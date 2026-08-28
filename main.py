#!/usr/bin/env python3
"""
main.py V19 — MESIN (engine).

V15 HARDENED: verified real-order execution, no blind mutating retries, exchange/local state reconciliation, protection-pair verification, and fail-closed emergency handling. Telegram handler, API client, monitoring,
stats, export /analyze, hot-swap /ganti. Logika analisa ada di
strategy_logic.py ("Otak"), diimpor di bawah.

V19: V18 observability retained; derived market-context telemetry added from existing cached M15/H1/D1 data with zero additional Binance requests. Adds breadth, relative strength vs BTC, efficiency, volume participation, volatility/regime descriptors, and market-context research export.
1. Setup logging dipindah ke awal (sebelumnya log dipanggil sebelum
   didefinisikan -> selalu NameError saat start).
2. Fallback strategy_logic gagal load: full_analyze jadi no-op (tidak
   entry baru), tapi TRAIL_R_LADDER dkk tetap terisi biar posisi yang
   sudah terbuka tetap ke-trailing.
3. full_analyze() terima df_h1/df_m15/df_d1 langsung, bukan symbol.
"""
import sys
import os, time, logging, threading
from collections import deque
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

import requests, pandas as pd, numpy as np, urllib3, json, html
from flask import Flask

try:
    import websocket   # pip: websocket-client
    _WS_LIB_OK = True
except ImportError:
    _WS_LIB_OK = False

# ── Logging: WAJIB disiapkan sebelum baris lain yang mungkin logging ──
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
MAX_PRICE       = 80.0
TOP_N_COINS     = 50
MONITOR_SLEEP       = 10
# Polling API SIGNED (posisi/order real) sengaja lebih jarang dari MONITOR_SLEEP
# biasa — TP/SL sudah dieksekusi Binance sendiri otomatis begitu tersentuh,
# polling di sini cuma buat TAHU KAPAN itu terjadi (pencatatan), bukan buat
# memicu eksekusinya. Terlalu sering polling = boros weight API tanpa manfaat
# nyata, malah berisiko kena limit/ban (lihat _binance_wait_if_banned).
REAL_TRADE_POLL_SLEEP = 30

# Telegram long-polling / Render watchdog. Telegram polling harus tetap hidup
# walaupun Binance sedang pause; error polling TIDAK boleh ditelan diam-diam.
TELEGRAM_LONGPOLL_TIMEOUT = 20
TELEGRAM_HTTP_TIMEOUT = 30
TELEGRAM_ERROR_BACKOFF_MAX = 60
TELEGRAM_KEEPALIVE_SEC = 300
# Jeda minimum antar-request HTTP ke Binance agar scan tidak menghantam API beruntun.
# 1 request / detik masih cukup untuk scan 50 koin tanpa burst besar.
BINANCE_REQUEST_INTERVAL = 2.5
# Setelah cooldown/ban Binance selesai, tunggu tambahan 60 detik sebelum request pertama.
BINANCE_POST_COOLDOWN_GRACE = 60.0
MAX_MARGIN_MULTIPLIER = 1.50  # HARD SAFETY CAP relative to configured MARGIN_USD
# Safety governor berbasis header usage; berhenti sebelum mendekati limit 1 menit.
BINANCE_WEIGHT_SOFT_LIMIT = 1800
BINANCE_WEIGHT_HARD_LIMIT = 2100
_binance_request_lock = threading.Lock()
_binance_last_request_at = 0.0
_binance_weight_1m = None
_binance_weight_seen_at = 0.0
MAX_POSITIONS       = 20   # runtime via /max — jangan pindah ke strategy_logic
MONITOR_INTERVAL    = 15 * 60
STRATEGY_MANAGE_INTERVAL = 60
STRATEGY_CONFIDENCE_THRESHOLD = 60  # filter orchestration; strategy tetap menghitung confidence
WIB = timezone(timedelta(hours=7))   # format jam entry di /trade
MAIN_ENGINE_VERSION = "machine_Learning_main_v1"

# ── SCAN MARKET-DATA CACHE ─────────────────────────────────────────────
# Scanner tidak boleh mengambil candle yang sama berulang-ulang. Cache ini
# hanya dipakai oleh pipeline scan; execution/position monitoring tetap memakai
# get_klines() normal sehingga tidak mengubah freshness data posisi.
SCAN_KLINE_TTL = {
    "15m": 8 * 60,      # refresh maksimum sekitar sekali per 8 menit
    "1h": 30 * 60,      # tidak perlu REST berulang di antara candle 1h
    "1d": 6 * 60 * 60,  # daily candle cukup direfresh berkala
}
_scan_kline_cache = {}          # {(symbol, interval): {df, fetched_at, source}}
_scan_kline_cache_lock = threading.RLock()
_scan_kline_fetch_locks = {}
_scan_kline_fetch_locks_guard = threading.Lock()
_scan_telemetry_lock = threading.Lock()
_last_scan_telemetry = {}

def _scan_key_lock(key):
    with _scan_kline_fetch_locks_guard:
        lock = _scan_kline_fetch_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _scan_kline_fetch_locks[key] = lock
        return lock

def _scan_cache_get(symbol, interval, limit):
    key = (symbol, interval)
    now = time.time()
    with _scan_kline_cache_lock:
        item = _scan_kline_cache.get(key)
        if not item:
            return None
        age = now - item["fetched_at"]
        ttl = SCAN_KLINE_TTL.get(interval, 10 * 60)
        df = item["df"]
        if age > ttl or df is None or df.empty or len(df) < min(limit, 40):
            return None
        return df.tail(limit).copy()

def _scan_cache_put(symbol, interval, df, source):
    if df is None or df.empty:
        return
    with _scan_kline_cache_lock:
        _scan_kline_cache[(symbol, interval)] = {
            "df": df.copy(), "fetched_at": time.time(), "source": source,
        }

def _scan_cache_stats():
    now = time.time()
    with _scan_kline_cache_lock:
        total = len(_scan_kline_cache)
        fresh = 0
        for (sym, itv), item in _scan_kline_cache.items():
            if now - item["fetched_at"] <= SCAN_KLINE_TTL.get(itv, 600):
                fresh += 1
    return total, fresh

# ─────────────────────────────────────────────

# Import OTAK — kalau gagal ATAU full_analyze() tidak ada di dalamnya
# (misal file strategy_logic.py yang salah/lama ke-upload), fallback aman.
try:
    from strategy_logic import *
    if "full_analyze" not in dir() or not callable(full_analyze):
        raise ImportError(
            "strategy_logic.py ke-import tapi TIDAK ADA fungsi full_analyze() di dalamnya "
            "— kemungkinan file yang salah/versi lama ter-upload.")
    log.info("[OTAK] strategy_logic.py berhasil dimuat & full_analyze() terverifikasi ada.")
except Exception as e:
    log.error(f"[OTAK] Gagal memuat strategy_logic.py ({e}) — fallback aman aktif.")
    # Engine fallback tidak memiliki aturan trading.
    def full_analyze(df_h1, df_m15, df_d1=None, symbol=None):
        return None
    _STRATEGY_LOAD_ERROR = str(e)
else:
    _STRATEGY_LOAD_ERROR = None

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN tidak ditemukan di environment. Cek file .env")

class TelegramLogHandler(logging.Handler):
    """
    Forward log ERROR/CRITICAL ke Telegram.
    Throttle: maks 1 pesan per 30 detik per pesan unik
    agar tidak flood saat error berulang.
    """
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self._last_sent: dict = {}   # {msg_key: timestamp}
        self._throttle  = 30         # detik

    def emit(self, record):
        # Hindari rekursi (error saat kirim TG itu sendiri)
        if "TG" in record.getMessage(): return
        try:
            msg_key = record.getMessage()[:80]
            now = time.time()
            if now - self._last_sent.get(msg_key, 0) < self._throttle:
                return
            self._last_sent[msg_key] = now

            cid = active_chat_id
            if not cid or not TELEGRAM_TOKEN: return

            level_em = "🔴" if record.levelno >= logging.CRITICAL else "⚠️"
            safe_msg = html.escape(record.getMessage()[:400], quote=False)
            text = (
                f"{level_em} <b>[{html.escape(record.levelname)}]</b>\n"
                f"<code>{safe_msg}</code>"
            )
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "HTML"},
                timeout=5
            )
        except Exception:
            pass   # jangan pernah raise dari handler log


_tg_log_handler = TelegramLogHandler()
log.addHandler(_tg_log_handler)

auto_mode      = False
auto_thread    = None
autostop_thread = None
active_chat_id = None
timeout_flag   = False
active_trade   = None   # dict posisi yang sedang dipantau, None jika tidak ada

STARTING_BALANCE = 10.0   # modal awal simulasi dalam USD

# Research/UI balance state. /mode on snapshots Binance balance once; /mode off
# restores the simulation anchor to $10. /resetstats NEVER changes this balance.
real_balance_snapshot = None
real_balance_snapshot_at = 0.0
real_balance_lock = threading.Lock()

stat_lock = threading.Lock()
stats = {
    "tp":0, "sl":0, "trail":0, "total":0,
    "balance"    : STARTING_BALANCE,
    "pnl_history": deque(maxlen=20),   # compatibility /backtest view
}

# FULL CLOSED-TRADE LEDGER UNTUK RESEARCH /analyze.
# Berbeda dari pnl_history yang sengaja hanya menyimpan 20 trade terakhir.
# Ledger ini tumbuh sepanjang research run dan dihapus oleh /resetstats.
trade_history_lock = threading.Lock()
trade_history: list[dict] = []
trade_sequence = 0
research_run_id = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")


# ==================== MACHINE LEARNING ====================
ML_STATE_DIR = Path(os.getenv("FULL_STATE_DIR", str(Path(__file__).resolve().parent / "machine_learning_state")))
ML_STATE_FILE = ML_STATE_DIR / "full_learning_state.json"
ML_EXPERIENCE_FILE = ML_STATE_DIR / "experience.jsonl"
ML_LOCK = threading.RLock()
FULL_MODE = False
FULL_THREAD = None
FULL_WAKE = threading.Event()
FULL_STOP = threading.Event()
FULL_MANUAL_THRESHOLD_SAVED = None
FULL_TRAIN_INTERVAL = max(60, int(os.getenv("FULL_TRAIN_INTERVAL_SEC", "180")))
FULL_MIN_TRAIN_SAMPLES = max(20, int(os.getenv("FULL_MIN_TRAIN_SAMPLES", "30")))
FULL_MIN_VALIDATION_SAMPLES = max(5, int(os.getenv("FULL_MIN_VALIDATION_SAMPLES", "10")))
FULL_PROMOTION_MIN_IMPROVEMENT = float(os.getenv("FULL_PROMOTION_MIN_IMPROVEMENT", "0.01"))
FULL_ALLOWED_THRESHOLDS = list(range(35, 81))
FULL_MIN_COVERAGE = 0.25
ML_FEATURE_NAMES = [
    "direction_confidence", "setup_quality", "entry_location_score", "rr",
    "range_position", "rsi_timing_score", "direction_edge", "m15_trigger_count",
    "poi_reacted", "selected_sweep", "m15_structure_alignment", "htf_alignment",
    "macro_alignment", "m15_relative_volume", "fib_position", "atr_pct_proxy",
    "entry_distance_atr", "risk_atr", "target_distance_atr", "data_quality",
    "entry_ob", "entry_fvg", "entry_eq", "entry_sweep", "entry_breakout",
    "entry_pullback", "htf_conflict", "m15_ranging"
]


def _ml_default_state():
    return {
        "schema": "machine_Learning_v2",
        "feature_names": list(ML_FEATURE_NAMES),
        "champion": None,
        "previous_champion": None,
        "last_training_at": None,
        "last_training_samples": 0,
        "last_training_result": None,
        "drift": {"status": "UNKNOWN", "score": 0.0},
        "learning_cycles": 0,
        "promotion_count": 0,
    }


def _ml_load_state():
    try:
        ML_STATE_DIR.mkdir(parents=True, exist_ok=True)
        if not ML_STATE_FILE.exists():
            return _ml_default_state()
        data = json.loads(ML_STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else _ml_default_state()
    except Exception as e:
        log.warning(f"[ML] state load gagal: {e}")
        return _ml_default_state()


ML_STATE = _ml_load_state()

def _ml_model_compatible(model):
    if not isinstance(model, dict):
        return False
    if str(model.get("schema") or "") != "machine_Learning_v2":
        return False
    names = model.get("feature_names")
    if not isinstance(names, list) or names != list(ML_FEATURE_NAMES):
        return False
    n = len(ML_FEATURE_NAMES)
    try:
        return all(len(model.get(k, [])) == n for k in ("mean", "scale", "w", "rw"))
    except Exception:
        return False

with ML_LOCK:
    if not _ml_model_compatible(ML_STATE.get("champion")):
        ML_STATE["previous_champion"] = None
        ML_STATE["champion"] = None
    if not _ml_model_compatible(ML_STATE.get("last_challenger")):
        ML_STATE["last_challenger"] = None
    _ml_save_state()

ML_EXPERIENCE_LOCK = threading.RLock()
ML_EXPERIENCE = []


def _ml_load_experience():
    try:
        ML_STATE_DIR.mkdir(parents=True, exist_ok=True)
        if not ML_EXPERIENCE_FILE.exists():
            return []
        rows = []
        with ML_EXPERIENCE_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
        return rows[-10000:]
    except Exception as e:
        log.warning(f"[ML] experience load gagal: {e}")
        return []


ML_EXPERIENCE = _ml_load_experience()


def _ml_append_experience(record):
    if not isinstance(record, dict) or not isinstance(record.get("learning_features"), dict):
        return
    sample = {
        "trade_uid": record.get("trade_uid"),
        "entry_time": record.get("entry_time"),
        "exit_time": record.get("exit_time"),
        "result": record.get("result"),
        "final_r": record.get("final_r"),
        "pnl_usd": record.get("pnl_usd"),
        "confidence": record.get("confidence"),
        "learning_features": record.get("learning_features"),
        "ml_model_version": record.get("ml_model_version", "static"),
    }
    try:
        with ML_EXPERIENCE_LOCK:
            key = str(sample.get("trade_uid") or "")
            if key and any(str(x.get("trade_uid") or "") == key for x in ML_EXPERIENCE[-200:]):
                return
            ML_EXPERIENCE.append(sample)
            if len(ML_EXPERIENCE) > 10000:
                del ML_EXPERIENCE[:-10000]
            ML_STATE_DIR.mkdir(parents=True, exist_ok=True)
            with ML_EXPERIENCE_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    except Exception as e:
        log.warning(f"[ML] experience append gagal: {e}")



def _ml_save_state():
    try:
        ML_STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = ML_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(ML_STATE, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, ML_STATE_FILE)
    except Exception as e:
        log.warning(f"[ML] state save gagal: {e}")


def _strategy_set_ml_model(model):
    setter = globals().get("set_learning_model")
    if callable(setter):
        try:
            setter(model)
        except Exception as e:
            log.warning(f"[ML] gagal bind model ke strategy: {e}")


def _ml_current_champion():
    with ML_LOCK:
        c = ML_STATE.get("champion")
        return dict(c) if isinstance(c, dict) else None


def _ml_sync_strategy():
    _strategy_set_ml_model(_ml_current_champion())


def _ml_feature_vector(record):
    feats = record.get("learning_features")
    if not isinstance(feats, dict):
        return None
    try:
        return np.asarray([float(feats.get(k, 0.0) or 0.0) for k in ML_FEATURE_NAMES], dtype=float)
    except Exception:
        return None


def _ml_collect_samples():
    with trade_history_lock:
        hist = [dict(x) for x in trade_history]
    with ML_EXPERIENCE_LOCK:
        persisted = [dict(x) for x in ML_EXPERIENCE]
    combined = {}
    for row in persisted + hist:
        key = str(row.get("trade_uid") or f"{row.get('symbol','')}|{row.get('entry_time','')}|{row.get('exit_time','')}")
        combined[key] = row
    samples = []
    for t in combined.values():
        x = _ml_feature_vector(t)
        if x is None:
            continue
        try:
            fr = float(t.get("final_r"))
        except (TypeError, ValueError):
            fr = None
        if fr is None:
            try:
                fr = float(t.get("pnl_usd", 0.0))
            except Exception:
                fr = 0.0
        result = str(t.get("result") or "sl").lower()
        y = 1.0 if fr > 0 and result not in {"strategy_error", "data_error"} else 0.0
        expected = max(-3.0, min(3.0, fr))
        try:
            baseline_conf = float(t.get("confidence", 50.0) or 50.0) / 100.0
        except (TypeError, ValueError):
            baseline_conf = 0.5
        samples.append((t.get("exit_time") or t.get("entry_time") or 0, x, y, expected, baseline_conf))
    samples.sort(key=lambda r: float(r[0] or 0))
    return samples

def _ml_sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _ml_fit(X, y, r_targets, reg=1.0, epochs=250, lr=0.03):
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale < 1e-8] = 1.0
    Z = (X - mean) / scale
    w = np.zeros(Z.shape[1], dtype=float)
    b = 0.0
    rw = np.zeros(Z.shape[1], dtype=float)
    rb = float(np.mean(r_targets)) if len(r_targets) else 0.0
    n = max(1, len(Z))
    for _ in range(epochs):
        p = _ml_sigmoid(Z @ w + b)
        gw = (Z.T @ (p - y)) / n + (reg / n) * w
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
        pred_r = Z @ rw + rb
        grw = (Z.T @ (pred_r - r_targets)) / n + (reg / n) * rw
        grb = float(np.mean(pred_r - r_targets))
        rw -= lr * grw
        rb -= lr * grb
    return {"mean": mean.tolist(), "scale": scale.tolist(), "w": w.tolist(), "b": b, "rw": rw.tolist(), "rb": rb}


def _ml_predict_model(model, X):
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    w = np.asarray(model["w"], dtype=float)
    b = float(model.get("b", 0.0))
    Z = (X - mean) / np.maximum(scale, 1e-8)
    p = _ml_sigmoid(Z @ w + b)
    rw = np.asarray(model.get("rw", np.zeros_like(w)), dtype=float)
    rb = float(model.get("rb", 0.0))
    er = Z @ rw + rb
    return p, er


def _ml_score_from_expected_r(expected_r):
    return 50.0 + 25.0 * np.tanh(np.asarray(expected_r, dtype=float))


def _ml_eval_predictions(p, rs, threshold):
    mask = p * 100.0 >= threshold
    if int(mask.sum()) < 1:
        return None
    actual = rs[mask]
    avg_r = float(np.mean(actual))
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for r in actual:
        equity += float(r)
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    win = float(np.mean(actual > 0)) if len(actual) else 0.0
    calibration = 1.0 - abs(float(np.mean(p[mask])) - win)
    coverage = float(mask.sum()) / max(1, len(rs))
    required = max(5, int(np.ceil(len(rs) * FULL_MIN_COVERAGE)))
    if int(mask.sum()) < required:
        return None
    objective = avg_r - 0.12 * dd + 0.18 * calibration + 0.05 * coverage
    return {"objective": float(objective), "threshold": int(threshold), "n": int(mask.sum()),
            "coverage": coverage, "avg_r": avg_r, "win_rate": win,
            "max_dd_r": dd, "calibration": calibration}


def _ml_eval_candidate(model, samples):
    if not samples:
        return {"objective": -999.0, "threshold": 60, "n": 0, "coverage": 0.0}
    X = np.vstack([s[1] for s in samples])
    rs = np.asarray([s[3] for s in samples], dtype=float)
    _p, er = _ml_predict_model(model, X)
    p = np.clip(_ml_score_from_expected_r(er) / 100.0, 0.0, 1.0)
    best = None
    for threshold in FULL_ALLOWED_THRESHOLDS:
        candidate = _ml_eval_predictions(p, rs, threshold)
        if candidate is not None and (best is None or candidate["objective"] > best["objective"]):
            best = candidate
    return best or {"objective": -999.0, "threshold": 60, "n": 0, "coverage": 0.0}


def _ml_eval_baseline(samples):
    if not samples:
        return {"objective": -999.0, "threshold": 60, "n": 0, "coverage": 0.0}
    p = np.asarray([s[4] for s in samples], dtype=float)
    rs = np.asarray([s[3] for s in samples], dtype=float)
    best = None
    for threshold in FULL_ALLOWED_THRESHOLDS:
        candidate = _ml_eval_predictions(p, rs, threshold)
        if candidate is not None and (best is None or candidate["objective"] > best["objective"]):
            best = candidate
    return best or {"objective": -999.0, "threshold": 60, "n": 0, "coverage": 0.0}

def _ml_train_once(force=False):
    global ML_STATE
    samples = _ml_collect_samples()
    now = datetime.now(WIB).isoformat()
    if len(samples) < FULL_MIN_TRAIN_SAMPLES:
        with ML_LOCK:
            ML_STATE["last_training_at"] = now
            ML_STATE["last_training_samples"] = len(samples)
            ML_STATE["last_training_result"] = {"status": "INSUFFICIENT_DATA", "samples": len(samples)}
            _ml_save_state()
        return False

    split = int(len(samples) * 0.70)
    if len(samples) - split < FULL_MIN_VALIDATION_SAMPLES:
        split = len(samples) - FULL_MIN_VALIDATION_SAMPLES
    train = samples[:split]
    valid = samples[split:]
    Xtr = np.vstack([s[1] for s in train])
    ytr = np.asarray([s[2] for s in train], dtype=float)
    rtr = np.asarray([s[3] for s in train], dtype=float)

    best = None
    for reg in (0.1, 0.3, 1.0, 3.0, 10.0):
        params = _ml_fit(Xtr, ytr, rtr, reg=reg)
        evaluation = _ml_eval_candidate(params, valid)
        evaluation["reg"] = reg
        if best is None or evaluation["objective"] > best["evaluation"]["objective"]:
            best = {"params": params, "evaluation": evaluation}
    if best is None:
        return False

    candidate_score = float(best["evaluation"].get("objective", -999.0))
    baseline_eval = _ml_eval_baseline(valid)
    champion = _ml_current_champion()
    champion_eval = {"objective": -999.0, "threshold": 60, "n": 0, "coverage": 0.0}
    if champion:
        try:
            champion_eval = _ml_eval_candidate(champion, valid)
        except Exception as e:
            log.warning(f"[ML] champion evaluation gagal: {e}")

    best_reference = max(float(baseline_eval.get("objective", -999.0)), float(champion_eval.get("objective", -999.0)))
    promote = candidate_score >= best_reference + FULL_PROMOTION_MIN_IMPROVEMENT
    if champion is None and candidate_score < baseline_eval.get("objective", -999.0) + FULL_PROMOTION_MIN_IMPROVEMENT:
        promote = False

    candidate = dict(best["params"])
    candidate.update({
        "schema": "machine_Learning_v2",
        "feature_names": list(ML_FEATURE_NAMES),
        "active": bool(promote),
        "model_version": f"ML-{int(time.time())}",
        "sample_count": len(samples),
        "train_samples": len(train),
        "validation_samples": len(valid),
        "validation_objective": candidate_score,
        "validation": best["evaluation"],
        "confidence_min": int(best["evaluation"].get("threshold", 60)),
        "live_weight": 0.35,
        "baseline_validation": baseline_eval,
        "champion_validation": champion_eval,
    })

    with ML_LOCK:
        ML_STATE["learning_cycles"] = int(ML_STATE.get("learning_cycles", 0)) + 1
        ML_STATE["last_training_at"] = now
        ML_STATE["last_training_samples"] = len(samples)
        ML_STATE["last_training_result"] = {
            "status": "PROMOTED" if promote else "CHALLENGER",
            "candidate": best["evaluation"],
            "baseline": baseline_eval,
            "champion": champion_eval,
        }
        if promote:
            ML_STATE["previous_champion"] = ML_STATE.get("champion")
            ML_STATE["champion"] = candidate
            ML_STATE["promotion_count"] = int(ML_STATE.get("promotion_count", 0)) + 1
        ML_STATE["last_challenger"] = candidate
        _ml_save_state()

    if promote:
        _ml_sync_strategy()
        if FULL_MODE:
            STRATEGY_CONFIDENCE_THRESHOLD = int(candidate["confidence_min"])
        log.info(f"[ML] Champion promoted {candidate['model_version']} threshold={candidate['confidence_min']} objective={candidate_score:.4f}")
    else:
        log.info(f"[ML] Challenger rejected/retained objective={candidate_score:.4f}; baseline={float(baseline_eval.get('objective',-999)):.4f}; champion={float(champion_eval.get('objective',-999)):.4f}")
    return promote

def _ml_learning_loop():
    while not FULL_STOP.is_set():
        try:
            if FULL_MODE:
                _ml_train_once()
        except Exception as e:
            log.exception(f"[ML] learning cycle gagal: {e}")
        FULL_WAKE.wait(FULL_TRAIN_INTERVAL)
        FULL_WAKE.clear()


def _full_status_text():
    champion = _ml_current_champion()
    with ML_LOCK:
        cycles = int(ML_STATE.get("learning_cycles", 0))
        promotions = int(ML_STATE.get("promotion_count", 0))
        last = ML_STATE.get("last_training_result") or {}
        last_samples = int(ML_STATE.get("last_training_samples", 0) or 0)
    if champion:
        return (f"🧠 <b>FULL LEARNING</b>: {'ON' if FULL_MODE else 'OFF'}\n"
                f"Champion: <code>{html.escape(str(champion.get('model_version')))}</code>\n"
                f"Samples: <b>{int(champion.get('sample_count',0) or 0)}</b> | OOS: <b>{int(champion.get('validation_samples',0) or 0)}</b>\n"
                f"Confidence min ML: <b>{int(champion.get('confidence_min',60) or 60)}%</b>\n"
                f"Objective OOS: <b>{float(champion.get('validation_objective',0) or 0):.4f}</b>\n"
                f"Learning cycles: <b>{cycles}</b> | Promotions: <b>{promotions}</b>\n"
                f"Last samples: <b>{last_samples}</b> | Last status: <b>{html.escape(str(last.get('status','—')))}</b>")
    return (f"🧠 <b>FULL LEARNING</b>: {'ON' if FULL_MODE else 'OFF'}\n"
            f"Champion: <b>Belum ada</b>\nLearning cycles: <b>{cycles}</b> | Samples: <b>{last_samples}</b>\n"
            f"Minimum training samples: <b>{FULL_MIN_TRAIN_SAMPLES}</b>")


def _full_on():
    global FULL_MODE, FULL_THREAD, FULL_MANUAL_THRESHOLD_SAVED
    with ML_LOCK:
        if not FULL_MODE:
            FULL_MANUAL_THRESHOLD_SAVED = int(STRATEGY_CONFIDENCE_THRESHOLD)
        FULL_MODE = True
        FULL_STOP.clear()
        FULL_WAKE.set()
    _ml_sync_strategy()
    if FULL_THREAD is None or not FULL_THREAD.is_alive():
        FULL_THREAD = threading.Thread(target=_ml_learning_loop, name="full-learning", daemon=True)
        FULL_THREAD.start()
    _ml_train_once(force=True)
    return _full_status_text()


def _full_off():
    global FULL_MODE, STRATEGY_CONFIDENCE_THRESHOLD, FULL_MANUAL_THRESHOLD_SAVED
    with ML_LOCK:
        FULL_MODE = False
        FULL_WAKE.clear()
        if FULL_MANUAL_THRESHOLD_SAVED is not None:
            STRATEGY_CONFIDENCE_THRESHOLD = int(FULL_MANUAL_THRESHOLD_SAVED)
            FULL_MANUAL_THRESHOLD_SAVED = None
    return _full_status_text()


def _ml_record_signal_metadata(signal):
    if not isinstance(signal, dict):
        return
    signal.setdefault("ml_schema", "machine_Learning_v1")
    signal.setdefault("ml_model_version", signal.get("learning_model_version", "static"))

_ml_sync_strategy()

# Research warmup: beberapa kandidat sinyal pertama setelah /resetstats sengaja
# ditolak tanpa dianggap trade gagal dan tanpa mengubah ban state.
EARLY_REJECT_DEFAULT = 5
early_reject_configured = EARLY_REJECT_DEFAULT
early_reject_remaining = EARLY_REJECT_DEFAULT
early_reject_lock = threading.Lock()

# Ban koin berbasis SCAN CYCLE.
# SHORT ban = pending/no-trade + low-confidence. CLOSED ban tetap terpisah.
ban_lock = threading.Lock()
banned_coins: dict = {}      # {symbol: {banned_at, duration, reason, kind, confidence}}
scan_counter = 0             # bertambah 1 setiap get_top_coins() dipanggil
BAN_DURATION_SCANS = 15.0
BAN_DURATION_TRADE_CLOSED = 50.0   # setelah trade BENAR-BENAR closed

# Research observability state. Low-confidence history intentionally survives
# /resetstats; use /resetlowconf to clear it explicitly.
low_conf_history_lock = threading.Lock()
low_conf_history: list[dict] = []
trail_events_lock = threading.Lock()
trail_events: list[dict] = []
trail_event_sequence = 0
scan_quality_lock = threading.Lock()
scan_quality_history: list[dict] = []
market_context_lock = threading.Lock()
market_context_history: list[dict] = []

# ── REAL TRADE (Binance Futures) — aktif otomatis kalau API key/secret diset ──
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

def _read_binance_credentials():
    """Read Binance credentials at runtime, not only at module import.

    Render/container environments can update secrets between process starts or
    expose a secret under a legacy alias. We keep the canonical names first,
    trim accidental whitespace, and return a single consistent pair.
    """
    key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_KEY")
    secret = (os.getenv("BINANCE_API_SECRET")
              or os.getenv("BINANCE_SECRET_KEY")
              or os.getenv("BINANCE_API_SECRET_KEY"))
    key = key.strip() if isinstance(key, str) else key
    secret = secret.strip() if isinstance(secret, str) else secret
    return key, secret

# BINANCE_KEYS_PRESENT: apakah kredensial ADA. Nilai ini juga disegarkan
# sebelum signed request/recovery sehingga secret yang tersedia setelah boot
# tidak dianggap hilang selamanya.
BINANCE_KEYS_PRESENT = bool(BINANCE_API_KEY and BINANCE_API_SECRET)
# REAL_TRADE_ENABLED: mode AKTIF SEKARANG (bisa di-toggle runtime via /mode
# on|off). Default tetap mengikuti ketersediaan key saat startup.
REAL_TRADE_ENABLED   = False
# New real trading is OFF until the operator explicitly runs /mode on.
# Existing REAL positions remain manageable even while the mode flag is OFF.


LEVERAGE          = 5      # runtime, via /leverage
MARGIN_USD        = 5.0    # runtime, via /margin
AUTOSTOP_PCT      = 3.0    # runtime, via /autostop
peak_real_balance = None   # diisi saat fetch balance real pertama kali sukses
autostop_lock     = threading.Lock()

def _position_execution_mode(pos):
    """Return the immutable execution class of a local position."""
    if not isinstance(pos, dict):
        return "SIMULATION"
    mode = str(pos.get("execution_mode") or "").strip().upper()
    if mode in {"REAL", "SIMULATION"}:
        return mode
    real_markers = ("order_id", "tp_order_id", "sl_order_id", "entry_client_order_id", "margin_used")
    if any(pos.get(k) is not None for k in real_markers):
        return "REAL"
    if str(pos.get("status") or "").upper() == "EMERGENCY":
        return "REAL"
    return "SIMULATION"


def _position_is_real(pos):
    return _position_execution_mode(pos) == "REAL"


def _has_real_recovery_work():
    """True only when private Binance reconciliation/protection work is required."""
    with positions_lock:
        if any(_position_is_real(pos) for pos in positions.values()):
            return True
    with _pending_protections_lock:
        if _pending_protections:
            return True
    with _pending_trails_lock:
        if _pending_trails:
            return True
    with _pending_cleanup_lock:
        if _pending_cleanup:
            return True
    return False


_binance_recovery_notice_generation = -1
_binance_recovery_notice_lock = threading.Lock()


def _ban_coin(sym, reason="", duration=None, kind="short", confidence=None):
    """Ban a symbol for scan cycles with explicit reason/kind metadata."""
    d = float(BAN_DURATION_SCANS if duration is None else duration)
    kind = str(kind or "short").lower()
    with ban_lock:
        banned_coins[sym] = {
            "banned_at": float(scan_counter),
            "duration": d,
            "reason": str(reason or ""),
            "kind": kind,
            "confidence": (float(confidence) if confidence is not None else None),
        }
    log.info(f"[ban] {sym} diban {d:g} scan ({kind})" + (f" [{reason}]" if reason else ""))

def _unban_coin(sym):
    with ban_lock:
        existed = sym in banned_coins
        if existed:
            del banned_coins[sym]
    if existed:
        log.info(f"[unban] {sym} dihapus dari ban manual/operator")
    return existed

def _ban_remaining(sym, current_scan=None):
    cur = float(scan_counter if current_scan is None else current_scan)
    with ban_lock:
        b = banned_coins.get(sym)
        if not b:
            return None
        if isinstance(b, tuple):
            banned_at, dur = b
            return max(0.0, float(dur) - (cur - float(banned_at)))
        return max(0.0, float(b["duration"]) - (cur - float(b["banned_at"])))


def _record_low_confidence_event(sym, confidence, cutoff, direction=None, entry_label=None):
    event = {
        "event_time": time.time(), "run_id": research_run_id, "scan_counter": scan_counter,
        "symbol": str(sym), "confidence": float(confidence),
        "confidence_min": float(STRATEGY_CONFIDENCE_THRESHOLD), "cutoff": float(cutoff),
        "decision": direction, "entry_label": entry_label,
    }
    with low_conf_history_lock: low_conf_history.append(event)

def _low_conf_summary():
    with low_conf_history_lock: rows=[dict(x) for x in low_conf_history]
    g={}
    for r in rows:
        sym=str(r.get("symbol") or "?"); c=float(r.get("confidence",0) or 0)
        x=g.setdefault(sym,{"symbol":sym,"count":0,"sum":0.0,"min":None,"max":None,"last":0.0})
        x["count"]+=1; x["sum"]+=c; x["min"]=c if x["min"] is None else min(x["min"],c); x["max"]=c if x["max"] is None else max(x["max"],c); x["last"]=max(x["last"],float(r.get("event_time") or 0))
    out=[]
    for x in g.values():
        x["avg"]=x["sum"]/x["count"] if x["count"] else 0.0; out.append(x)
    out.sort(key=lambda x:(-x["count"],x["avg"],x["symbol"]))
    return out

def _record_scan_quality(row):
    with scan_quality_lock: scan_quality_history.append(dict(row))

def _market_feature_row(sym, h1, m15, analysis):
    """Derive chart-context features from already-fetched scan data.

    This function intentionally performs ZERO Binance requests. It is a feature
    extraction layer for future strategy_logic versions: price velocity,
    directional efficiency, ATR/volatility regime, volume participation and
    structure/decision labels are recorded while the scanner already has the
    candles in memory.
    """
    def _safe_float(v, default=None):
        try:
            x=float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    out={"symbol":str(sym),"decision":(analysis or {}).get("decision"),
         "confidence":_safe_float((analysis or {}).get("confidence")),
         "entry_label":(analysis or {}).get("entry_label"),
         "struct_h1":(analysis or {}).get("struct_h1") or (analysis or {}).get("structure"),
         "d1_bias":(analysis or {}).get("d1_bias")}
    try:
        c=pd.to_numeric(m15["close"], errors="coerce").dropna()
        v=pd.to_numeric(m15["volume"], errors="coerce").fillna(0.0)
        atr_col=pd.to_numeric(m15.get("atr", pd.Series(index=m15.index,dtype=float)), errors="coerce")
        if len(c)<20:
            return out
        last=float(c.iloc[-1]);
        out["price_1h_pct"]=(float(c.iloc[-1]/c.iloc[-5]-1.0)*100.0) if c.iloc[-5] else None
        out["price_4h_pct"]=(float(c.iloc[-1]/c.iloc[-17]-1.0)*100.0) if c.iloc[-17] else None
        diffs=c.diff().abs().iloc[-17:].sum()
        out["efficiency_4h"]=float(abs(c.iloc[-1]-c.iloc[-17])/diffs) if diffs and np.isfinite(diffs) else 0.0
        atr_last=_safe_float(atr_col.iloc[-1])
        out["atr_pct"]=(atr_last/last*100.0) if atr_last and last else None
        vol_base=float(v.iloc[-21:-1].mean()) if len(v)>=21 else float(v.iloc[:-1].mean())
        out["relative_volume"]=(float(v.iloc[-1])/vol_base) if vol_base>0 else None
        true_range=np.maximum(pd.to_numeric(m15["high"],errors="coerce")-pd.to_numeric(m15["low"],errors="coerce"),
                              np.maximum((pd.to_numeric(m15["high"],errors="coerce")-pd.to_numeric(m15["close"].shift(1),errors="coerce")).abs(),
                                         (pd.to_numeric(m15["low"],errors="coerce")-pd.to_numeric(m15["close"].shift(1),errors="coerce")).abs()))
        tr=pd.Series(true_range,index=m15.index).replace([np.inf,-np.inf],np.nan).dropna()
        if len(tr)>=20:
            recent=float(tr.iloc[-4:].mean()); baseline=float(tr.iloc[-20:-4].median())
            out["range_expansion_ratio"]=recent/baseline if baseline>0 else None
        atr_series=atr_col.dropna()
        if len(atr_series)>=30 and atr_last is not None:
            med=float(atr_series.iloc[-31:-1].median())
            out["volatility_ratio"]=atr_last/med if med>0 else None
        # Simple, robust regime label derived from the same chart data. It is a
        # descriptor, not a trade rule.
        r1=out.get("price_1h_pct") or 0.0; eff=out.get("efficiency_4h") or 0.0
        rr=out.get("range_expansion_ratio") or 1.0
        if eff>=0.55 and rr>=1.15:
            regime="expansion"
        elif eff<=0.30 and rr<=1.10:
            regime="range/compression"
        elif eff>=0.45:
            regime="trend"
        else:
            regime="transition"
        out["chart_regime"]=regime
        out["directional_bias"]="bullish" if r1>0.15 else "bearish" if r1<-0.15 else "neutral"
    except Exception:
        pass
    return out


def _record_market_context(rows):
    if not rows:
        return
    with market_context_lock:
        market_context_history.extend(dict(x) for x in rows)


def _market_context_snapshot(run_id=None):
    with market_context_lock:
        rows=[dict(x) for x in market_context_history]
    return [x for x in rows if run_id is None or x.get("run_id")==run_id]


def _summarize_market_context(rows):
    rows=[dict(x) for x in (rows or [])]
    analyzed=len(rows)
    bull=sum(1 for x in rows if str(x.get("decision") or "").upper()=="BUY")
    bear=sum(1 for x in rows if str(x.get("decision") or "").upper()=="SELL")
    neutral=max(0, analyzed-bull-bear)
    def med(key):
        vals=[float(x[key]) for x in rows if x.get(key) is not None]
        return float(np.median(vals)) if vals else None
    breadth=(bull-bear)/analyzed if analyzed else 0.0
    eff=med("efficiency_4h") or 0.0
    rr=med("range_expansion_ratio") or 1.0
    avg_rv=float(np.mean([float(x["relative_volume"]) for x in rows if x.get("relative_volume") is not None])) if any(x.get("relative_volume") is not None for x in rows) else None
    med_r1=med("price_1h_pct")
    med_r4=med("price_4h_pct")
    if analyzed==0:
        regime="unknown"
    elif abs(breadth)>=0.35 and eff>=0.45:
        regime="bullish expansion" if breadth>0 else "bearish expansion"
    elif abs(breadth)<=0.15 and eff<=0.35:
        regime="range/compression"
    elif abs(breadth)>=0.20:
        regime="bullish trend" if breadth>0 else "bearish trend"
    else:
        regime="transition"
    btc=[x for x in rows if str(x.get("symbol"))=="BTCUSDT"]
    btc1=btc[0].get("price_1h_pct") if btc else None
    btc4=btc[0].get("price_4h_pct") if btc else None
    return {"market_regime":regime,"bullish_breadth_pct":100*bull/analyzed if analyzed else None,"bearish_breadth_pct":100*bear/analyzed if analyzed else None,"neutral_breadth_pct":100*neutral/analyzed if analyzed else None,"breadth_score":breadth,"median_price_1h_pct":med_r1,"median_price_4h_pct":med_r4,"median_efficiency_4h":eff,"median_range_expansion_ratio":rr,"avg_relative_volume":avg_rv,"btc_price_1h_pct":btc1,"btc_price_4h_pct":btc4,"analyzed_symbols":analyzed}

def _record_trail_event(sym, pos, update, old_sl, new_sl, status="APPLIED", error=None):
    global trail_event_sequence
    if old_sl is None or new_sl is None: return None
    try:
        sig=pos.get("signal",{}); entry=float(pos.get("entry") or sig.get("entry") or 0); initial_sl=float(pos.get("initial_sl") or sig.get("initial_sl") or old_sl); risk=abs(entry-initial_sl)
        side=str(sig.get("decision") or "BUY").upper(); price=float(pos.get("current_price") or pos.get("price") or entry)
        current_r=(((price-entry)/risk) if side=="BUY" else ((entry-price)/risk)) if risk else 0.0
        mfe=float(pos.get("mfe_r",0) or 0); mae=float(pos.get("mae_r",0) or 0); give=max(0.0,mfe-current_r); gr=(give/mfe) if mfe>0 else 0.0
        atr=float(sig.get("atr") or 0); protected=(((float(new_sl)-entry)/risk) if side=="BUY" else ((entry-float(new_sl))/risk)) if risk else 0.0
        reasons=update.get("reason",[]) if isinstance(update,dict) else []
        if isinstance(reasons,str): reasons=[reasons]
        now=time.time(); event={
            "event_id":None,"trade_uid":pos.get("trade_uid"),"run_id":research_run_id,"event_time":now,"symbol":sym,"decision":side,
            "entry":entry,"initial_sl":initial_sl,"old_sl":float(old_sl),"new_sl":float(new_sl),"tp":sig.get("tp"),"current_price":price,
            "current_r":current_r,"mfe_r":mfe,"mae_r":mae,"giveback_r":give,"giveback_ratio":gr,"protected_r":protected,
            "atr":atr,"sl_distance_atr":(abs(price-float(new_sl))/atr if atr>0 else None),
            "weakness_score":update.get("weakness_score") if isinstance(update,dict) else None,"state":update.get("state","TRAIL") if isinstance(update,dict) else "TRAIL",
            "trade_phase":update.get("trade_phase") if isinstance(update,dict) else pos.get("trade_phase"),
            "trail_source":(update.get("trail_source") or update.get("source") or "adaptive") if isinstance(update,dict) else "adaptive",
            "reasons":" | ".join(str(x) for x in reasons[:10]),"relative_volume":update.get("relative_volume") if isinstance(update,dict) else None,
            "candidate_type":update.get("candidate_type") if isinstance(update,dict) else None,"status":status,
            "error_code":None,"error_message":str(error)[:400] if error else "","time_since_entry_sec":max(0.0,now-float(pos.get("entry_time") or now)),
            "time_since_previous_trail_sec":None,"distance_to_tp_r":None}
        if sig.get("tp") is not None and risk:
            tp_r=(((float(sig["tp"])-entry)/risk) if side=="BUY" else ((entry-float(sig["tp"]))/risk)); event["distance_to_tp_r"]=tp_r-current_r
        with trail_events_lock:
            trail_event_sequence+=1; event["event_id"]=trail_event_sequence
            prev=next((x for x in reversed(trail_events) if x.get("trade_uid")==pos.get("trade_uid") and x.get("symbol")==sym),None)
            if prev: event["time_since_previous_trail_sec"]=max(0.0,now-float(prev.get("event_time") or now))
            trail_events.append(event)
        pos["trail_update_count"]=int(pos.get("trail_update_count",0))+1
        pos["trail_applied_count"]=int(pos.get("trail_applied_count",0))+(status=="APPLIED")
        pos["trail_failed_count"]=int(pos.get("trail_failed_count",0))+(status=="FAILED")
        pos["trail_queued_count"]=int(pos.get("trail_queued_count",0))+(status=="QUEUED")
        if pos.get("first_trail_r") is None: pos["first_trail_r"]=current_r
        pos["last_trail_r"]=current_r; pos["max_protected_r"]=max(float(pos.get("max_protected_r",-999)),protected)
        return event
    except Exception as e:
        log.debug(f"[trail-observe] {sym}: {e}"); return None

FAPI = "https://fapi.binance.com"
BINANCE_WS_URL = "wss://fstream.binance.com/ws"

# ── Flask ─────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    with stat_lock:
        t=stats["total"]; tp=stats["tp"]; sl=stats["sl"]; trail=stats.get("trail",0)
    with ban_lock:
        n_banned = len(banned_coins)
    wins = tp + trail
    wr=f"{wins/(wins+sl)*100:.1f}%" if (wins+sl)>0 else "–"
    ws_state = "REST (WS fallback siaga)" if ws_feed.is_fresh() else "REST (WS fallback belum siap)"
    return (f"<h3>SMC Signal Broadcaster</h3>"
            f"<p>Auto:{auto_mode} | Banned:{n_banned} | Data:{ws_state}</p>"
            f"<p>Total:{t} TP:{tp} SL:{sl} Trail:{trail} WR:{wr}</p>"), 200

@app.route("/health")
@app.route("/healthz")
def health():
    # Endpoint ini sengaja TIDAK memanggil Binance/API eksternal.
    # Aman dipakai Render/uptime monitor untuk menjaga service tetap hidup.
    with _telegram_state_lock:
        tg_ok = _telegram_polling_alive
        tg_last = _telegram_last_success_at
    with _binance_pause_lock:
        paused = _binance_scan_paused or _binance_recovering
    body = {
        "status": "ok",
        "telegram_polling": bool(tg_ok),
        "telegram_last_success_age": (round(time.time()-tg_last, 1) if tg_last else None),
        "binance_scan_paused": bool(paused),
        "timestamp": time.time(),
    }
    return body, 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    log.info(f"[flask] binding port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═════════════════════════════════════════════
# TELEGRAM
# ═════════════════════════════════════════════
def _tg_plain_fallback(text):
    """Strip HTML markup for a guaranteed-safe Telegram fallback message."""
    try:
        # The normal path keeps rich HTML. This path is used only after Telegram
        # rejects the markup, so sacrificing formatting is preferable to losing
        # the operational message entirely.
        plain = re.sub(r"<[^>]*>", "", str(text))
        plain = html.unescape(plain)
        return plain
    except Exception:
        return str(text)


def tg_send(chat_id, text):
    """Send Telegram HTML safely, retrying rejected HTML once as plain text."""
    if not chat_id or not TELEGRAM_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id":chat_id,"text":str(text),"parse_mode":"HTML"},
            timeout=10)
        if r.status_code < 400:
            return True

        body = r.text[:500]
        # Telegram 400s caused by malformed HTML must never become an operational
        # failure. Retry exactly once without parse_mode. Do not retry arbitrary
        # HTTP failures because that could amplify rate-limit/network problems.
        if r.status_code == 400 and "can't parse entities" in body.lower():
            fallback = _tg_plain_fallback(text)
            try:
                rr = requests.post(
                    url,
                    json={"chat_id":chat_id,"text":fallback},
                    timeout=10)
                if rr.status_code < 400:
                    log.warning("[TG/sendMessage] HTML rejected; delivered plain-text fallback")
                    return True
                log.warning(f"[TG/sendMessage] HTML rejected and fallback failed: HTTP {rr.status_code}: {rr.text[:300]}")
                return False
            except Exception as e:
                log.error(f"[TG/sendMessage] fallback error: {e}")
                return False

        log.warning(f"[TG/sendMessage] HTTP {r.status_code}: {body[:300]}")
        return False
    except Exception as e:
        log.error(f"[TG/sendMessage] {e}")
        return False

# ============================================================
# TAMBAHAN BARU (START) — Helper kirim file ke Telegram
# ============================================================
def tg_send_document(chat_id, file_path, caption=""):
    """Kirim file ke Telegram."""
    if not chat_id or not TELEGRAM_TOKEN:
        return
    try:
        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {"chat_id": chat_id, "caption": caption}
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
                files=files, data=data, timeout=30
            )
    except Exception as e:
        log.error(f"[TG doc] {e}")
# ============================================================
# TAMBAHAN BARU (END)
# ============================================================


# ============================================================
# TAMBAHAN BARU (START) — GitHub API untuk /ganti
# ============================================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME")  # format: "username/repo"

def _commit_to_github(content, path="strategy_logic.py", commit_msg="Update strategy_logic via Telegram /ganti"):
    """Commit file ke GitHub menggunakan API."""
    if not GITHUB_TOKEN or not REPO_NAME:
        raise ValueError("GITHUB_TOKEN atau REPO_NAME tidak diset di environment.")
    
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    # 1. Get current SHA (untuk update)
    sha = None
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception:
        pass
    
    # 2. Commit baru
    import base64
    data = {
        "message": commit_msg,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": "main"
    }
    if sha:
        data["sha"] = sha
    
    resp = requests.put(url, headers=headers, json=data)
    if resp.status_code not in (200, 201):
        raise ValueError(f"GitHub commit gagal: {resp.status_code} {resp.text}")
    
    return True
# ============================================================
# TAMBAHAN BARU (END)
# ============================================================

# ── TELEGRAM POLLING STATE ───────────────────────────────────────────────
_telegram_state_lock = threading.Lock()
_telegram_polling_alive = False
_telegram_last_success_at = 0.0
_telegram_last_error_at = 0.0
_telegram_last_conflict_alert_at = 0.0

class TelegramPollingConflict(ConnectionError):
    """Telegram 409: webhook/instance lain bentrok dengan getUpdates."""


def _telegram_mark_success():
    global _telegram_polling_alive, _telegram_last_success_at
    with _telegram_state_lock:
        _telegram_polling_alive = True
        _telegram_last_success_at = time.time()


def _telegram_mark_error():
    global _telegram_polling_alive, _telegram_last_error_at
    with _telegram_state_lock:
        _telegram_polling_alive = False
        _telegram_last_error_at = time.time()


def _telegram_bootstrap():
    """Pastikan token memakai long polling, bukan webhook lama yang tertinggal."""
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo",
            timeout=10)
        d = r.json()
        if not d.get("ok"):
            log.warning(f"[TG] getWebhookInfo gagal: {d}")
            return
        info = d.get("result", {})
        url = info.get("url") or ""
        if url:
            log.warning(f"[TG] Webhook aktif ({url}) — hapus agar long polling tidak 409.")
            rr = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook",
                params={"drop_pending_updates": False}, timeout=10)
            rd = rr.json()
            if rd.get("ok"):
                log.info("[TG] Webhook lama dihapus; long polling siap.")
            else:
                log.error(f"[TG] deleteWebhook gagal: {rd}")
    except Exception as e:
        log.warning(f"[TG] bootstrap error: {e}")


def tg_updates(offset=None):
    """Long poll Telegram dengan error visibility + backoff signal.

    Tidak pernah mengembalikan [] secara diam-diam saat HTTP/JSON gagal, karena
    itu membuat bot terlihat hidup padahal command tidak diterima.
    """
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"timeout": TELEGRAM_LONGPOLL_TIMEOUT, "offset": offset},
            timeout=TELEGRAM_HTTP_TIMEOUT)
        if r.status_code == 409:
            _telegram_mark_error()
            raise TelegramPollingConflict(
                "Telegram 409 Conflict: ada webhook atau instance bot lain yang memakai token ini.")
        if r.status_code == 429:
            _telegram_mark_error()
            try:
                d = r.json(); retry_after = int(d.get("parameters", {}).get("retry_after", 5))
            except Exception:
                retry_after = 5
            raise ConnectionError(f"Telegram rate limit 429; retry_after={retry_after}s")
        if r.status_code >= 500:
            _telegram_mark_error()
            raise ConnectionError(f"Telegram server HTTP {r.status_code}")
        r.raise_for_status()
        d = r.json()
        if not d.get("ok"):
            _telegram_mark_error()
            raise ConnectionError(f"Telegram getUpdates error: {d}")
        _telegram_mark_success()
        return d.get("result", [])
    except TelegramPollingConflict:
        raise
    except Exception as e:
        _telegram_mark_error()
        log.warning(f"[TG/getUpdates] {e}")
        raise


# ═════════════════════════════════════════════
# DATA LAYER — REST sebagai sumber UTAMA, WS cuma fallback TERAKHIR
#   Tier 1: Binance Futures REST        (sumber utama)
#   Tier 2: Bybit REST                  (kalau Binance REST error/kena
#           limit/ban — lihat fapi_get(): begitu Binance balas 418/429,
#           retry ke Binance langsung dihentikan, tidak ditunggu2)
#   Tier 3: Binance Futures WebSocket   (fallback TERAKHIR, dipakai hanya
#           kalau Tier 1 & Tier 2 dua-duanya gagal. WS tetap disubscribe
#           & di-backfill terus di background — lihat ensure_symbol_
#           interval() — supaya buffernya SIAP dipakai sewaktu-waktu,
#           tapi TIDAK dijadikan sumber utama krn koneksinya sering
#           putus-nyambung di lingkungan hosting ini)
#   Tier 4: CoinGecko REST — DARURAT, HARGA SAJA, hanya koin-koin di
#           COINGECKO_ID_MAP. TIDAK dipakai untuk klines: granularitas
#           candle CoinGecko (30m/4h/4hari tergantung rentang) tidak
#           cocok dengan kebutuhan M1/M15/H1/D1 presisi bot ini — kalau
#           dipaksakan, sinyal SMC yang butuh candle presisi (BOS/CHoCH/
#           swing point) bisa salah baca. Kalau semua REST+WS gagal
#           total, get_klines() balikin DataFrame kosong (sama seperti
#           perilaku lama) alih-alih pura-pura pakai data CoinGecko yang
#           tidak akurat.
# ═════════════════════════════════════════════
BYBIT = "https://api.bybit.com"

# Konversi interval Binance → Bybit
INTERVAL_MAP = {
    "1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
    "1h":"60","2h":"120","4h":"240","1d":"D","1w":"W",
}

# Simbol Binance Futures -> id CoinGecko, HANYA koin-koin besar yang aman
# di-mapping manual (ticker collision antar chain bikin auto-match ke
# CoinGecko berisiko fatal — bisa ambil harga koin yang salah). Tambah
# manual kalau perlu koin lain, JANGAN pernah generate otomatis dari nama.
COINGECKO_ID_MAP = {
    "BTCUSDT":"bitcoin", "ETHUSDT":"ethereum", "BNBUSDT":"binancecoin",
    "SOLUSDT":"solana", "XRPUSDT":"ripple", "ADAUSDT":"cardano",
    "DOGEUSDT":"dogecoin", "AVAXUSDT":"avalanche-2", "LINKUSDT":"chainlink",
    "DOTUSDT":"polkadot", "LTCUSDT":"litecoin", "TRXUSDT":"tron",
    "ATOMUSDT":"cosmos", "NEARUSDT":"near", "APTUSDT":"aptos",
    "ARBUSDT":"arbitrum", "OPUSDT":"optimism", "SUIUSDT":"sui",
    "TONUSDT":"the-open-network", "BCHUSDT":"bitcoin-cash",
}

import re

# ── State ban IP Binance — DIBAGI antara fapi_get (publik) & _binance_signed
# (private), karena ban itu per-IP, bukan per-endpoint/per-key. Begitu satu
# sisi kena ban, sisi lain juga harus tahu & berhenti nembak, bukan lanjut
# jalan sendiri-sendiri (itu yang bikin log kebanjiran "Skip ... HTTP 418").
_binance_ban_lock = threading.Lock()
_binance_banned_until = 0.0   # unix timestamp detik; 0 = tidak sedang ban
_BINANCE_BAN_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".binance_ban_state.json")
# Global circuit breaker: saat Binance 429/418, NEW SCAN/ENTRY berhenti.
# WS tetap hidup untuk memantau posisi aktif.
_binance_scan_paused = False
_binance_pause_reason = ""
_binance_recovering = False
_binance_pause_lock = threading.Lock()
# Monotonic pause generation + notification generation. A single Binance ban
# must produce at most ONE pause notification, even when several worker threads
# receive the same 418/429 a few milliseconds apart.
_binance_pause_generation = 0
_binance_pause_notice_generation = -1
_binance_pause_notice_lock = threading.Lock()
# Pending trail: satu state terbaru per simbol, bukan queue order lama berantai.
_pending_trails = {}   # {symbol: {sl, tp, quantity, updated_at, reason, side}}
_pending_trails_lock = threading.Lock()
_pending_protections = {}  # filled position awaiting TP/SL after Binance recovery
_pending_protections_lock = threading.Lock()
# V6: explicit emergency/cleanup state. These states are retained in /trade until
# Binance confirms the exchange-side truth; API errors must never silently remove them.
_pending_cleanup = {}          # {symbol: {reason, created_at, last_error}}
_pending_cleanup_lock = threading.Lock()
_binance_time_offset_ms = 0
_binance_time_sync_at = 0.0
_binance_time_sync_lock = threading.Lock()
BINANCE_TIME_SYNC_TTL = 60.0
_real_trade_preflight_cache = {"at": 0.0, "position_mode": None, "can_trade": None}
_real_trade_preflight_lock = threading.Lock()
REAL_TRADE_PREFLIGHT_TTL = 60.0

class BinanceCooldownError(ConnectionError):
    """Tidak mengirim request Binance selama cooldown aktif."""


class BinanceUnknownExecutionError(ConnectionError):
    """Mutating Binance request may have reached the exchange but response is unknown.

    IMPORTANT: callers must reconcile exchange state before submitting a duplicate
    mutating request. This prevents blind duplicate entry/exit/protection orders.
    """


def _load_binance_ban_state():
    global _binance_banned_until
    try:
        with open(_BINANCE_BAN_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        until = float(data.get("banned_until", 0.0))
        if until > time.time():
            _binance_banned_until = until
            with _binance_pause_lock:
                globals()["_binance_scan_paused"] = True
                globals()["_binance_pause_reason"] = "persisted Binance cooldown"
                globals()["_binance_pause_generation"] = 1
            log.warning(f"[binance] cooldown dipulihkan: {until-time.time():.0f} detik tersisa")
    except Exception:
        pass


def _save_binance_ban_state(until):
    try:
        tmp = _BINANCE_BAN_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"banned_until": float(until)}, f)
        os.replace(tmp, _BINANCE_BAN_STATE_FILE)
    except Exception as e:
        log.debug(f"[binance] gagal simpan cooldown: {e}")

_load_binance_ban_state()


def _binance_sync_time(force=False):
    """Sync local clock against Binance server time for signed requests.
    Public endpoint only; no API key required.
    """
    global _binance_time_offset_ms, _binance_time_sync_at
    now = time.time()
    with _binance_time_sync_lock:
        if not force and (now - _binance_time_sync_at) < BINANCE_TIME_SYNC_TTL:
            return _binance_time_offset_ms
    local_send = int(time.time() * 1000)
    try:
        r = requests.get(f"{FAPI}/fapi/v1/time", timeout=5, verify=False)
        r.raise_for_status()
        server_ms = int(r.json()["serverTime"])
        local_recv = int(time.time() * 1000)
        midpoint = (local_send + local_recv) // 2
        offset = server_ms - midpoint
        with _binance_time_sync_lock:
            _binance_time_offset_ms = int(offset)
            _binance_time_sync_at = time.time()
        log.info(f"[binance-time] sync OK offset={offset}ms")
        return int(offset)
    except Exception as e:
        log.warning(f"[binance-time] sync gagal: {e}")
        raise


def _binance_timestamp_ms(sync_if_stale=True):
    with _binance_time_sync_lock:
        offset = _binance_time_offset_ms
        synced_at = _binance_time_sync_at
    if sync_if_stale and (time.time() - synced_at) >= BINANCE_TIME_SYNC_TTL:
        try:
            offset = _binance_sync_time(force=True)
        except Exception:
            pass
    return int(time.time() * 1000) + int(offset)


def _queue_pending_cleanup(sym, reason="orphan algo cleanup", error=None):
    with _pending_cleanup_lock:
        item = _pending_cleanup.get(sym) or {"reason": reason, "created_at": time.time()}
        item.update({"reason": reason, "last_error": str(error)[:300] if error else item.get("last_error")})
        _pending_cleanup[sym] = item


def _clear_pending_cleanup(sym):
    with _pending_cleanup_lock:
        _pending_cleanup.pop(sym, None)


def _get_pending_cleanup(sym):
    with _pending_cleanup_lock:
        v = _pending_cleanup.get(sym)
        return dict(v) if v else None


def _get_open_algo_orders(sym):
    """Exchange-side verification of remaining Binance algo orders for symbol."""
    data = _binance_signed("GET", "/fapi/v1/openAlgoOrders", {"symbol": sym})
    if isinstance(data, dict):
        rows = data.get("orders") or data.get("openOrders") or data.get("data") or []
    else:
        rows = data or []
    return rows if isinstance(rows, list) else []


def _cleanup_algo_orders_verified(sym):
    """Cancel all algo orders and verify none remain."""
    _cancel_all_algo_orders_verified(sym)
    _clear_pending_cleanup(sym)
    return True


def _binance_register_ban(msg="", fallback_seconds=60, retry_after=None):
    """Aktifkan global Binance circuit breaker secara idempotent.

    Banyak worker boleh menerima 418/429 yang sama. Hanya transisi dari
    *tidak pause* -> *pause* yang membuat generasi pause baru. Error berikutnya
    hanya boleh memperpanjang cooldown bila memang lebih panjang; tidak boleh
    menghasilkan ban notification baru.
    """
    global _binance_banned_until, _binance_scan_paused, _binance_pause_reason
    global _binance_pause_generation
    m = re.search(r"banned until (\d+)", msg or "")
    candidates = [time.time() + fallback_seconds]
    if retry_after is not None:
        try:
            candidates.append(time.time() + max(float(retry_after), 0.0))
        except (TypeError, ValueError):
            pass
    if m:
        candidates.append(int(m.group(1)) / 1000)
    until = max(candidates) + BINANCE_POST_COOLDOWN_GRACE
    now = time.time()
    with _binance_ban_lock:
        previous_until = _binance_banned_until
        was_cooldown_active = previous_until > now
        _binance_banned_until = max(previous_until, until)
        current_until = _binance_banned_until
    with _binance_pause_lock:
        was_paused = bool(_binance_scan_paused or _binance_recovering or was_cooldown_active)
        _binance_scan_paused = True
        _binance_pause_reason = (msg or "Binance rate limit / ban")[:180]
        if not was_paused:
            _binance_pause_generation += 1
            transition = True
        else:
            transition = False
    _save_binance_ban_state(current_until)
    remaining = max(current_until - time.time(), 0)
    if transition:
        # ERROR is intentionally emitted only once per pause generation.
        # Subsequent 418/429 events are WARNING-only and therefore cannot flood
        # Telegram through TelegramLogHandler.
        log.error(f"[BINANCE PAUSE] Scanner & entry BARU dihentikan selama {remaining:.0f} detik. WS tetap memantau posisi.")
    else:
        log.warning(f"[BINANCE PAUSE] duplicate/parallel limit event ignored; cooldown tersisa {remaining:.0f} detik.")
    return transition


def _notify_binance_pause_once(chat_id):
    """Send one user-facing pause notice for the current pause generation."""
    if not chat_id or not _binance_is_scan_paused():
        return False
    global _binance_pause_notice_generation
    with _binance_pause_notice_lock:
        with _binance_pause_lock:
            generation = _binance_pause_generation
            reason = _binance_pause_reason
        if generation == _binance_pause_notice_generation:
            return False
        _binance_pause_notice_generation = generation
    remaining = _binance_cooldown_remaining()
    detail = f"\nCooldown: <b>{remaining:.0f} detik</b>" if remaining > 0 else ""
    tg_send(chat_id,
            "⏸️ <b>Binance RATE LIMIT/BAN</b>\n"
            "Scanning & entry baru dihentikan. Posisi aktif tetap dipantau via WS."
            f"{detail}")
    log.warning(f"[BINANCE PAUSE NOTICE] generation={generation} reason={reason[:120]}")
    return True


def _binance_is_scan_paused():
    with _binance_pause_lock:
        paused = _binance_scan_paused or _binance_recovering
    if paused:
        return True
    return _binance_cooldown_remaining() > 0


def _binance_cooldown_remaining():
    with _binance_ban_lock:
        return max(0.0, _binance_banned_until - time.time())


def _binance_try_resume():
    global _binance_scan_paused, _binance_pause_reason
    if _binance_cooldown_remaining() > 0:
        return False
    with _binance_pause_lock:
        _binance_scan_paused = False
        _binance_pause_reason = ""
    return True


def _queue_pending_trail(sym, new_sl, new_tp, qty, reason="strategy", side=None):
    """Simpan state trail terbaru per simbol; order lama tidak ditumpuk."""
    with _pending_trails_lock:
        old = _pending_trails.get(sym)
        if old is None:
            _pending_trails[sym] = {
                "sl": new_sl, "tp": new_tp, "quantity": qty,
                "updated_at": time.time(), "reason": reason, "side": side,
            }
            return
        buy = (side or old.get("side")) == "BUY"
        old_sl = old.get("sl")
        better_sl = (new_sl is not None and old_sl is None) or (new_sl is not None and ((new_sl > old_sl) if buy else (new_sl < old_sl)))
        if better_sl or new_tp != old.get("tp") or (qty and qty != old.get("quantity")):
            old.update({"sl": new_sl, "tp": new_tp, "quantity": qty, "updated_at": time.time(), "reason": reason, "side": side or old.get("side")})


def _get_pending_trail(sym):
    with _pending_trails_lock:
        v = _pending_trails.get(sym)
        return dict(v) if v else None


def _clear_pending_trail(sym):
    with _pending_trails_lock:
        _pending_trails.pop(sym, None)


def _binance_wait_if_banned():
    with _binance_ban_lock:
        until = _binance_banned_until
    remaining = until - time.time()
    if remaining > 0:
        raise BinanceCooldownError(f"Binance cooldown aktif {remaining:.0f}s")

def _binance_update_weight_from_response(r):
    """Catat request-weight 1m dari header Binance bila tersedia."""
    global _binance_weight_1m, _binance_weight_seen_at
    raw = None
    for key in ("X-MBX-USED-WEIGHT-1M", "x-mbx-used-weight-1m"):
        if key in r.headers:
            raw = r.headers.get(key)
            break
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    _binance_weight_1m = value
    _binance_weight_seen_at = time.time()
    return value


def _binance_request_pause():
    """Legacy throttle helper. New Binance calls should use _binance_request_slot()."""
    global _binance_last_request_at
    _binance_wait_if_banned()
    with _binance_request_lock:
        if _binance_weight_1m is not None and _binance_weight_1m >= BINANCE_WEIGHT_SOFT_LIMIT:
            wall_now = time.time()
            wait_window = max(0.0, 62.0 - (wall_now % 60.0))
            log.warning(f"[binance-weight] {_binance_weight_1m} weight/1m — throttle {wait_window:.1f}s ke window berikutnya.")
            time.sleep(wait_window)
        wait = BINANCE_REQUEST_INTERVAL - (time.monotonic() - _binance_last_request_at)
        if wait > 0:
            time.sleep(wait)
        _binance_last_request_at = time.monotonic()


@contextmanager
def _binance_request_slot():
    """Serialize the actual Binance HTTP request, not only its scheduling.

    This closes the race where worker A detects a ban while worker B has already
    passed the old throttle function but has not sent its HTTP request yet. B now
    re-checks the breaker immediately before its request. Once A registers the ban,
    the next worker waiting on this lock is stopped before it can hit Binance.
    """
    global _binance_last_request_at
    with _binance_request_lock:
        _binance_wait_if_banned()
        if _binance_weight_1m is not None and _binance_weight_1m >= BINANCE_WEIGHT_SOFT_LIMIT:
            wall_now = time.time()
            wait_window = max(0.0, 62.0 - (wall_now % 60.0))
            log.warning(f"[binance-weight] {_binance_weight_1m} weight/1m — throttle {wait_window:.1f}s ke window berikutnya.")
            time.sleep(wait_window)
            _binance_wait_if_banned()
        wait = BINANCE_REQUEST_INTERVAL - (time.monotonic() - _binance_last_request_at)
        if wait > 0:
            time.sleep(wait)
        # Final breaker check is deliberately immediately before yielding to the
        # HTTP call. This is the important anti-race point.
        _binance_wait_if_banned()
        _binance_last_request_at = time.monotonic()
        yield


def _raw_get(url, params=None, retries=3):
    """HTTP GET dengan retry — digunakan oleh Bybit & CoinGecko."""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10, verify=False)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"[http] {i+1}/{retries} {url}: {e}")
            time.sleep(2)
    raise ConnectionError(f"GET gagal: {url}")


# ── BINANCE REST (backfill awal WS + fallback tier-2) ─────────────────
def fapi_get(path, params=None):
    # Satu request saja per pemanggilan. Retry REST publik ke Binance dihapus:
    # saat gagal langsung gunakan fallback agar error tidak berubah menjadi burst.
    _binance_wait_if_banned()
    try:
        with _binance_request_slot():
            r = requests.get(f"{FAPI}{path}", params=params, timeout=10, verify=False)
        used = _binance_update_weight_from_response(r)
        if used is not None and used >= BINANCE_WEIGHT_HARD_LIMIT:
            log.warning(f"[binance-weight] {used} weight/1m — menahan request baru sampai window mereda.")
        if r.status_code in (418, 429):
            retry_after = r.headers.get("Retry-After")
            _binance_register_ban(r.text or "", retry_after=retry_after)
            raise BinanceCooldownError(f"Binance kena limit/ban (HTTP {r.status_code})")
        d = r.json()
        if isinstance(d, dict) and "code" in d:
            if d["code"] == -1003:
                retry_after = None
                _binance_register_ban(d.get("msg", ""), retry_after=retry_after)
                raise BinanceCooldownError(f"Binance {d['code']}: {d.get('msg')}")
            raise ValueError(f"Binance {d['code']}: {d.get('msg')}")
        return d
    except BinanceCooldownError:
        raise
    except Exception as e:
        log.warning(f"[binance] {path} gagal: {e} — langsung fallback")
        raise ConnectionError(f"Binance gagal: {path}: {e}") from e


# ============================================================
# REAL TRADE — Binance Futures signed API (order/leverage/posisi)
# Dipakai TERPISAH dari fapi_get di atas (yang publik, untuk cari
# sinyal) supaya limit rate keduanya tidak bercampur.
# ============================================================
import hmac, hashlib, urllib.parse, math
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN

def _binance_signed(method, path, params=None):
    """Signed Binance request with strict handling for mutating requests.

    GET requests may be retried after transient transport errors. Mutating
    requests (POST/PUT/DELETE) are NEVER blindly retried after a transport
    exception because Binance may have accepted the request even if the HTTP
    response was lost. The caller must reconcile first.
    """
    global BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_KEYS_PRESENT
    key, secret = _read_binance_credentials()
    if key and secret:
        BINANCE_API_KEY, BINANCE_API_SECRET = key, secret
        BINANCE_KEYS_PRESENT = True
    else:
        BINANCE_KEYS_PRESENT = False
    if not BINANCE_KEYS_PRESENT:
        raise RuntimeError("BINANCE_API_KEY/SECRET tidak tersedia di runtime Render")

    base_params = dict(params or {})
    method = str(method).upper()
    mutating = method in {"POST", "PUT", "DELETE"}
    max_attempts = 1 if mutating else 3
    last_err = None
    time_resync_attempted = False

    for attempt in range(max_attempts):
        _binance_wait_if_banned()
        try:
            with _binance_time_sync_lock:
                time_stale = (time.time() - _binance_time_sync_at) >= BINANCE_TIME_SYNC_TTL
            if time_stale:
                try:
                    _binance_sync_time(force=True)
                except Exception:
                    pass

            with _binance_request_slot():
                req = dict(base_params)
                req["timestamp"] = _binance_timestamp_ms(sync_if_stale=False)
                req["recvWindow"] = 10000
                query = urllib.parse.urlencode(req, safe=",")
                sig = hmac.new(BINANCE_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
                url = f"{FAPI}{path}?{query}&signature={sig}"
                headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
                r = requests.request(method, url, headers=headers, timeout=10, verify=False)

            used = _binance_update_weight_from_response(r)
            if used is not None and used >= BINANCE_WEIGHT_HARD_LIMIT:
                log.warning(f"[binance-weight] {used} weight/1m setelah signed {method} {path}")

            if r.status_code in (418, 429):
                retry_after = r.headers.get("Retry-After")
                _binance_register_ban(r.text, retry_after=retry_after)
                raise BinanceCooldownError(f"Binance kena limit/ban (HTTP {r.status_code})")

            data = r.json()
            if isinstance(data, dict) and "code" in data and data["code"] < 0:
                code = int(data["code"])
                if code == -1003:
                    _binance_register_ban(data.get("msg", ""))
                    raise BinanceCooldownError(f"Binance {code}: {data.get('msg')}")
                if code == -1021:
                    if time_resync_attempted:
                        raise RuntimeError(f"Binance -1021 setelah resync: {data.get('msg')}")
                    time_resync_attempted = True
                    last_err = RuntimeError(f"Binance -1021: {data.get('msg')}")
                    log.warning(f"[binance-time] -1021 pada {method} {path}; resync server time lalu retry dengan signature baru")
                    _binance_sync_time(force=True)
                    # Safe to retry because the exchange explicitly rejected the request.
                    if mutating and attempt + 1 >= max_attempts:
                        max_attempts = 2
                    continue
                raise RuntimeError(f"Binance {code}: {data.get('msg')}")
            return data

        except BinanceCooldownError:
            raise
        except RuntimeError:
            raise
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            if mutating:
                raise BinanceUnknownExecutionError(
                    f"Binance {method} {path} transport error; execution status unknown: {e}"
                ) from e
            log.warning(f"[binance-signed] GET {path} percobaan {attempt+1}: {e}")
            time.sleep(1.0 + attempt)
        except Exception as e:
            last_err = e
            if mutating:
                raise BinanceUnknownExecutionError(
                    f"Binance {method} {path} response error; execution status unknown: {e}"
                ) from e
            log.warning(f"[binance-signed] GET {path} percobaan {attempt+1}: {e}")
            time.sleep(1.0 + attempt)

    raise RuntimeError(f"Gagal request signed {method} {path}: {last_err}")


_symbol_filters_cache = {}
_exchange_info_cache = {"fetched_at": 0.0}
_exchange_info_lock = threading.Lock()

def _load_all_symbol_filters():
    """Fetch /fapi/v1/exchangeInfo SEKALI, parse SEMUA simbol sekaligus ke
    cache — supaya koin baru berikutnya tidak perlu fetch ulang endpoint
    berat ini. Refresh tiap 1 jam (filter simbol jarang berubah)."""
    with _exchange_info_lock:
        if time.time() - _exchange_info_cache["fetched_at"] < 3600 and _symbol_filters_cache:
            return
        data = fapi_get("/fapi/v1/exchangeInfo")
        for s in data["symbols"]:
            f = {x["filterType"]: x for x in s["filters"]}
            if "LOT_SIZE" not in f or "PRICE_FILTER" not in f:
                continue
            _symbol_filters_cache[s["symbol"]] = {
                "stepSize": float(f["LOT_SIZE"]["stepSize"]),
                "minQty": float(f["LOT_SIZE"]["minQty"]),
                "minNotional": float(f.get("MIN_NOTIONAL", {}).get("notional", 5.0)),
                "tickSize": float(f["PRICE_FILTER"]["tickSize"]),
                "qtyPrecision": s["quantityPrecision"],
                "pricePrecision": s["pricePrecision"],
            }
        _exchange_info_cache["fetched_at"] = time.time()

def get_symbol_filters(symbol):
    if symbol not in _symbol_filters_cache:
        _load_all_symbol_filters()
    if symbol not in _symbol_filters_cache:
        raise ValueError(f"{symbol} tidak ada di exchangeInfo")
    return _symbol_filters_cache[symbol]


def round_to_tick(price, tick_size):
    """Bulatkan ke kelipatan PERSIS tickSize (bukan cuma jumlah desimal —
    dua hal beda, sumber error -4014 'Price not increased by tick size').
    Pakai Decimal supaya tidak kena noise floating point (mis. 0.0005)."""
    if not tick_size or tick_size <= 0:
        return price
    d_price, d_tick = Decimal(str(price)), Decimal(str(tick_size))
    steps = (d_price / d_tick).to_integral_value(rounding=ROUND_HALF_UP)
    return float(steps * d_tick)


def round_qty(quantity, step, qty_prec, rounding=ROUND_HALF_UP):
    """Bulatkan quantity ke kelipatan PERSIS stepSize pakai Decimal.

    KENAPA INI PENTING (bug 'posisi tidak 100% tertutup saat SL'):
    ─────────────────────────────────────────────────────────────
    `math.floor(quantity / step) * step` pakai float biasa KENA noise
    binary floating point. Contoh nyata: quantity=1.2, step=0.1 →
    1.2/0.1 di Python = 11.999999999999998 (BUKAN 12.0), lalu
    math.floor() membulatkannya jadi 11 → hasil 1.1, padahal quantity
    aslinya 1.2. Order SL (reduceOnly=quantity) yang dipasang jadi
    0.1 koin LEBIH KECIL dari posisi riil → saat SL ter-trigger, 0.1
    koin itu TIDAK IKUT TERTUTUP dan posisi tersisa terbuka selamanya
    (tanpa proteksi SL sama sekali untuk sisa itu).

    Fix: pakai Decimal(str(...)) supaya representasi desimalnya EXACT
    (bukan biner), dan default ROUND_HALF_UP (bulatkan ke kelipatan
    step TERDEKAT) — karena quantity yang masuk ke sini (qty posisi
    aktif) SEHARUSNYA sudah persis kelipatan step sejak awal dibuka
    (lihat calc_auto_quantity), jadi tidak perlu di-floor lagi; floor
    kedua itulah sumber bug di atas. Kalau memang perlu floor murni
    (mis. saat menghitung qty MAKSIMUM yang boleh dibeli dari suatu
    notional — lihat calc_auto_quantity), panggil dengan
    rounding=ROUND_DOWN secara eksplisit.
    """
    if not step or step <= 0:
        return round(quantity, qty_prec)
    d_qty, d_step = Decimal(str(quantity)), Decimal(str(step))
    steps = (d_qty / d_step).to_integral_value(rounding=rounding)
    return float(round(steps * d_step, qty_prec))


def calc_auto_quantity(symbol, entry_price, margin_usd, leverage):
    """
    Quantity dari margin x leverage, dibulatkan ke stepSize Binance.
    Kalau di bawah minQty/minNotional (error -1013 LOT_SIZE / -4164
    MIN_NOTIONAL), margin dinaikkan SEDIKIT supaya order tetap valid.
    Cap kenaikan = mana yang LEBIH BESAR antara 3x margin awal ATAU
    margin awal + $5 — kombinasi ini supaya margin kecil (mis. $1) tetap
    dapat headroom wajar (cuma 1.5x dari $1 = $1.5, kelewat sempit utk
    banyak koin), sementara margin besar tidak melonjak tak terkendali.
    Return (qty, margin_terpakai, dinaikkan?) atau (None, None, False)
    kalau tetap gagal walau sudah disesuaikan.
    """
    info = get_symbol_filters(symbol)
    step, min_qty, min_notional = info["stepSize"], info["minQty"], info["minNotional"]

    def qty_from_notional(notional):
        # Sama seperti place_sl_order() — pakai Decimal (ROUND_DOWN, exact)
        # bukan math.floor(float) supaya tidak kehilangan 1 step ekstra
        # akibat noise floating point (mis. 1.2/0.1 = 11.999999999998 di
        # Python murni). Floor tetap dipertahankan di sini (memang harus
        # floor, bukan nearest) karena tujuannya membatasi qty MAKSIMUM
        # yang boleh dibeli dari notional yang tersedia.
        q = round_qty(notional / entry_price, step, info["qtyPrecision"], rounding=ROUND_DOWN)
        return q

    qty = qty_from_notional(margin_usd * leverage)
    if qty >= min_qty and qty * entry_price >= min_notional:
        actual_margin = (qty * entry_price) / max(float(leverage), 1e-12)
        hard_cap = float(margin_usd) * MAX_MARGIN_MULTIPLIER
        if actual_margin > hard_cap + 1e-9:
            log.warning(f"[calc_auto_quantity] {symbol}: computed margin ${actual_margin:.4f} > hard cap ${hard_cap:.4f}")
            return None, None, False
        return qty, float(margin_usd), False

    # Minimum-notional compensation is allowed only inside the hard cap.
    needed_notional = max(min_notional, min_qty * entry_price) * 1.01
    bumped_margin = needed_notional / max(float(leverage), 1e-12)
    cap = float(margin_usd) * MAX_MARGIN_MULTIPLIER
    if bumped_margin > cap:
        log.warning(f"[calc_auto_quantity] {symbol}: minimum notional needs ${bumped_margin:.4f} "
                    f"but hard cap is ${cap:.4f} (base margin ${margin_usd:.2f}, leverage {leverage}x)")
        return None, None, False
    qty = qty_from_notional(needed_notional)
    actual_margin = (qty * entry_price) / max(float(leverage), 1e-12)
    if qty < min_qty or qty * entry_price < min_notional or actual_margin > cap + 1e-9:
        return None, None, False
    return qty, round(actual_margin, 4), True


def _real_trade_preflight(force=False):
    """Verify account canTrade and require One-way/BOTH position mode.

    The engine intentionally does not auto-switch position mode because Binance
    rejects that while positions/orders exist. Entries are blocked when the
    account is in Hedge Mode, while existing positions remain manageable.
    """
    now = time.time()
    with _real_trade_preflight_lock:
        if (not force and now - _real_trade_preflight_cache["at"] < REAL_TRADE_PREFLIGHT_TTL
                and _real_trade_preflight_cache["position_mode"] is not None):
            if not _real_trade_preflight_cache["can_trade"]:
                raise RuntimeError("Binance account canTrade=false")
            if _real_trade_preflight_cache["position_mode"] is True:
                raise RuntimeError("Binance Hedge Mode aktif; bot V15 membutuhkan One-way Mode (positionSide=BOTH)")
            return dict(_real_trade_preflight_cache)

    mode = _binance_signed("GET", "/fapi/v1/positionSide/dual", {})
    acct = _binance_signed("GET", "/fapi/v2/account", {})
    dual = bool(mode.get("dualSidePosition")) if isinstance(mode, dict) else False
    can_trade = bool(acct.get("canTrade", True)) if isinstance(acct, dict) else True
    with _real_trade_preflight_lock:
        _real_trade_preflight_cache.update({"at": now, "position_mode": dual, "can_trade": can_trade})
    if not can_trade:
        raise RuntimeError("Binance account canTrade=false")
    if dual:
        raise RuntimeError("Binance Hedge Mode aktif; bot V15 membutuhkan One-way Mode (positionSide=BOTH)")
    return dict(_real_trade_preflight_cache)


def _new_client_id(prefix):
    # Binance allows up to 36 chars: ^[\.A-Z\:/a-z0-9_-]{1,36}$
    return f"{prefix}_{int(time.time()*1000)%10_000_000_000}_{threading.get_ident()%100000}"[:36]


def _order_query_by_client_id(symbol, client_id):
    try:
        return _binance_signed("GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": client_id})
    except Exception as e:
        msg = str(e).lower()
        if "order does not exist" in msg or "-2013" in msg:
            return None
        raise


def _find_open_algo_by_client_id(symbol, client_algo_id):
    for row in _get_open_algo_orders(symbol):
        if str(row.get("clientAlgoId") or "") == str(client_algo_id):
            return row
    return None


def _protection_matches(row, symbol, side, order_type, trigger_price, quantity, tick, step):
    if not row or row.get("symbol") != symbol:
        return False
    if str(row.get("side")) != str(side):
        return False
    typ = str(row.get("orderType") or row.get("type") or "").upper()
    expected = "TAKE_PROFIT" if order_type == "TAKE_PROFIT_MARKET" else "STOP"
    if expected not in typ:
        return False
    try:
        t_actual = round_to_tick(float(row.get("triggerPrice")), tick)
        t_expected = round_to_tick(float(trigger_price), tick)
        q_actual = round_qty(float(row.get("quantity")), step, 16)
        q_expected = round_qty(float(quantity), step, 16)
        return abs(t_actual - t_expected) <= max(tick, 1e-12) and abs(q_actual - q_expected) <= max(step, 1e-12) and bool(row.get("reduceOnly", False))
    except Exception:
        return False


def _verify_protection_pair(symbol, is_buy, tp_price, sl_price, quantity):
    info = get_symbol_filters(symbol)
    tick, step = info["tickSize"], info["stepSize"]
    qty = round_qty(quantity, step, info.get("qtyPrecision", 8))
    rows = _get_open_algo_orders(symbol)
    close_side = "SELL" if is_buy else "BUY"
    tp_ok = any(_protection_matches(r, symbol, close_side, "TAKE_PROFIT_MARKET", tp_price, qty, tick, step) for r in rows)
    sl_ok = any(_protection_matches(r, symbol, close_side, "STOP_MARKET", sl_price, qty, tick, step) for r in rows)
    if not (tp_ok and sl_ok):
        raise RuntimeError(f"protection verification gagal: TP={tp_ok}, SL={sl_ok}, algo={len(rows)}")
    return rows


def _reconcile_position_quantity(symbol):
    real = get_real_position(symbol)
    if not real:
        return None
    qty = abs(float(real.get("positionAmt", 0) or 0))
    return (real, qty) if qty > 0 else (real, 0.0)


def _verified_market_close(symbol, is_buy, reason, chat_id=None, max_retries=1):
    """Close actual Binance position with reconcile-before-retry semantics."""
    last_error = None
    for attempt in range(max_retries + 1):
        real_info = _reconcile_position_quantity(symbol)
        if real_info is None:
            return True, None
        real, qty = real_info
        if qty <= 0:
            return True, None

        side = "SELL" if is_buy else "BUY"
        try:
            resp = place_market_order(symbol, side, qty, reduce_only=True)
        except BinanceUnknownExecutionError as e:
            last_error = e
            # Do NOT immediately send another order. Reconcile first.
            try:
                real_after = get_real_position(symbol)
            except Exception:
                real_after = None
            if real_after is None or abs(float(real_after.get("positionAmt", 0) or 0)) <= 0:
                return True, None
            if attempt >= max_retries:
                raise
            continue
        except Exception as e:
            last_error = e
            # Even a known/rejected response can race with an exchange-side fill.
            # Reconcile actual position before declaring the close failed.
            try:
                real_after = get_real_position(symbol)
                remaining_after = abs(float(real_after.get("positionAmt", 0) or 0)) if real_after else 0.0
                if remaining_after <= 0:
                    return True, None
            except Exception:
                pass
            raise

        try:
            real_after = get_real_position(symbol)
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                raise RuntimeError(f"market close response received but position verification failed: {e}") from e
            continue
        remaining = abs(float(real_after.get("positionAmt", 0) or 0)) if real_after else 0.0
        if remaining <= 0:
            exit_price = None
            if isinstance(resp, dict):
                try:
                    exit_price = float(resp.get("avgPrice") or 0) or None
                except Exception:
                    exit_price = None
            return True, exit_price
        if attempt >= max_retries:
            raise RuntimeError(f"market close submitted but position remains open: {remaining}")

    raise RuntimeError(f"market close failed: {last_error}")


def set_leverage_verified(symbol, leverage):
    """Set leverage, and reconcile if the POST response is unknown."""
    try:
        return set_leverage(symbol, leverage)
    except BinanceUnknownExecutionError:
        rows = _binance_signed("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
        for row in rows or []:
            if row.get("symbol") == symbol and int(float(row.get("leverage", 0) or 0)) == int(leverage):
                return {"verified": True, "leverage": leverage}
        raise


def set_leverage(symbol, leverage):
    return _binance_signed("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})


def place_limit_order(symbol, side, quantity, price):
    tick = get_symbol_filters(symbol)["tickSize"]
    client_id = _new_client_id("ENTRY")
    params = {
        "symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTC",
        "quantity": quantity, "price": round_to_tick(price, tick),
        "newClientOrderId": client_id,
    }
    try:
        result = _binance_signed("POST", "/fapi/v1/order", params)
        if isinstance(result, dict):
            result.setdefault("clientOrderId", client_id)
        return result
    except BinanceUnknownExecutionError as e:
        reconciled = _order_query_by_client_id(symbol, client_id)
        if reconciled is not None:
            return reconciled
        e.client_order_id = client_id
        e.symbol = symbol
        raise


def place_market_order(symbol, side, quantity, reduce_only=False):
    params = {
        "symbol": symbol, "side": side, "type": "MARKET",
        "quantity": quantity, "newOrderRespType": "RESULT",
    }
    if reduce_only:
        params["reduceOnly"] = "true"
    return _binance_signed("POST", "/fapi/v1/order", params)


def cancel_order(symbol, order_id):
    """Cancel ordinary order. Unknown transport result is reconciled before reporting failure."""
    if not order_id:
        return None
    try:
        return _binance_signed("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id})
    except BinanceUnknownExecutionError as e:
        try:
            st = get_order_status(symbol, order_id)
            status = str(st.get("status", "")).upper()
            if status in {"CANCELED", "EXPIRED", "REJECTED"}:
                return st
            if status == "FILLED":
                raise RuntimeError(f"cancel #{order_id} terlambat: order sudah FILLED")
        except RuntimeError:
            raise
        except Exception:
            pass
        raise
    except Exception as e:
        log.warning(f"[cancel_order] {symbol} #{order_id}: {e}")
        return None


def get_order_status(symbol, order_id):
    return _binance_signed("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id})



def get_real_position(symbol):
    rows = _binance_signed("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
    for p in rows:
        if p["symbol"] == symbol:
            if abs(float(p.get("positionAmt", 0) or 0)) > 0:
                return p
            return None
    return None

def get_real_positions_all():
    """Return every non-zero Futures position visible on Binance."""
    rows = _binance_signed("GET", "/fapi/v2/positionRisk", {})
    return [p for p in (rows or []) if abs(float(p.get("positionAmt", 0) or 0)) > 0]


def get_open_orders_all(symbol=None):
    """Return ordinary open orders. With no symbol, query the whole account."""
    params = {"symbol": symbol} if symbol else {}
    rows = _binance_signed("GET", "/fapi/v1/openOrders", params)
    return rows if isinstance(rows, list) else []


def get_open_algo_orders_all(symbol=None):
    """Return open conditional/algo orders. With no symbol, query the whole account."""
    params = {"symbol": symbol} if symbol else {}
    data = _binance_signed("GET", "/fapi/v1/openAlgoOrders", params)
    if isinstance(data, dict):
        rows = data.get("orders") or data.get("openOrders") or data.get("data") or []
    else:
        rows = data or []
    return rows if isinstance(rows, list) else []


def _cancel_all_ordinary_orders_verified(sym, retries=2):
    """Cancel all ordinary orders for a symbol and verify exchange-side empty state.

    A DELETE transport error is treated as unknown: query open orders first. Only
    if orders are still present do we attempt another cancel.
    """
    last = None
    for attempt in range(max(1, int(retries))):
        try:
            _binance_signed("DELETE", "/fapi/v1/allOpenOrders", {"symbol": sym})
        except BinanceUnknownExecutionError as e:
            last = e
        except Exception as e:
            last = e
        try:
            rows = get_open_orders_all(sym)
            if not rows:
                return True
        except Exception as e:
            last = e
        if attempt + 1 < retries:
            time.sleep(0.35 * (attempt + 1))
    raise RuntimeError(f"{sym}: ordinary order cleanup belum terverifikasi ({last})")



def _cancel_all_algo_orders_verified(sym, retries=2):
    """Cancel all algo orders, then verify exchange-side empty state.

    A transport-unknown DELETE is reconciled by GET /openAlgoOrders before any
    retry. This avoids both orphan protection and blind repeated cancels.
    """
    last = None
    for attempt in range(max(1, int(retries))):
        try:
            cancel_all_algo_orders(sym)
        except BinanceUnknownExecutionError as e:
            last = e
        except Exception as e:
            last = e
        try:
            rows = _get_open_algo_orders(sym)
            if not rows:
                _clear_pending_cleanup(sym)
                return True
        except Exception as e:
            last = e
        if attempt + 1 < retries:
            time.sleep(0.35 * (attempt + 1))
    raise RuntimeError(f"{sym}: algo cleanup belum terverifikasi ({last})")


def _cancel_all_symbol_orders_verified(sym):
    """Cancel ordinary + algo orders for one symbol and verify both are empty."""
    _cancel_all_ordinary_orders_verified(sym)
    _cancel_all_algo_orders_verified(sym)
    ordinary = get_open_orders_all(sym)
    algo = _get_open_algo_orders(sym)
    if ordinary or algo:
        raise RuntimeError(f"{sym}: masih ada order setelah cancel (ordinary={len(ordinary)}, algo={len(algo)})")
    return True


def _verified_timeout_symbol(sym, chat_id, reason="manual timeout"):
    """Flatten one symbol and verify position + ordinary + algo orders are zero."""
    try:
        _cancel_all_symbol_orders_verified(sym)
        real = get_real_position(sym)
        exit_price = None
        if real is not None:
            live_qty = abs(float(real.get("positionAmt", 0) or 0))
            if live_qty > 0:
                is_buy = float(real.get("positionAmt", 0) or 0) > 0
                closed, exit_price = _verified_market_close(sym, is_buy, reason, chat_id=chat_id, max_retries=1)
                if not closed:
                    raise RuntimeError("market close belum terkonfirmasi")
        _cancel_all_symbol_orders_verified(sym)
        with positions_lock:
            local = positions.get(sym)
        if local and local.get("entry_time") is not None and local.get("status") != "pending":
            close_position(sym, "timeout", close_price=exit_price or get_price(sym) or local.get("entry"))
        elif local:
            with positions_lock:
                positions.pop(sym, None)
            _record_pending_cancel("manual_timeout")
        _clear_pending_cleanup(sym)
        tg_send(chat_id, f"✅ <b>TIMEOUT CLOSED</b> — {sym}\nPosisi Binance: <b>0</b>\nOrder biasa: <b>0</b>\nAlgo TP/SL/Trail: <b>0</b>\nSemua exposure {sym} sudah dibersihkan.")
        return True
    except Exception as e:
        with positions_lock:
            if sym in positions:
                positions[sym]["status"] = "EMERGENCY"
                positions[sym]["emergency_reason"] = reason
                positions[sym]["emergency_error"] = str(e)[:300]
        _queue_pending_cleanup(sym, "timeout cleanup", e)
        tg_send(chat_id, f"🚨 <b>TIMEOUT BELUM SELESAI</b> — {sym}\n<code>{str(e)[:350]}</code>\nPosisi tetap dipertahankan di /trade. Gunakan <code>/ok {sym}</code> setelah Binance/API normal.")
        return False


def _verified_timeout_all(chat_id):
    """GLOBAL emergency cleanup: cancel every Binance order and flatten every position."""
    try:
        positions_remote = get_real_positions_all()
        ordinary = get_open_orders_all()
        algo = get_open_algo_orders_all()
        symbols = {str(p.get("symbol")) for p in positions_remote if p.get("symbol")}
        symbols.update(str(o.get("symbol")) for o in ordinary if o.get("symbol"))
        symbols.update(str(o.get("symbol")) for o in algo if o.get("symbol"))

        for sym in sorted(symbols):
            _cancel_all_symbol_orders_verified(sym)

        exit_prices = {}
        for p in get_real_positions_all():
            sym = p.get("symbol")
            qty = abs(float(p.get("positionAmt", 0) or 0))
            if not sym or qty <= 0:
                continue
            is_buy = float(p.get("positionAmt", 0) or 0) > 0
            closed, exit_price = _verified_market_close(sym, is_buy, "manual timeout global", chat_id=chat_id, max_retries=1)
            if not closed:
                raise RuntimeError(f"{sym}: posisi global belum terkonfirmasi flat")
            exit_prices[sym] = exit_price

        remaining_pos = get_real_positions_all()
        remaining_orders = get_open_orders_all()
        remaining_algo = get_open_algo_orders_all()
        if remaining_pos or remaining_orders or remaining_algo:
            raise RuntimeError(f"verifikasi global gagal: positions={len(remaining_pos)}, ordinary_orders={len(remaining_orders)}, algo_orders={len(remaining_algo)}")

        with positions_lock:
            local_items = [(sym, dict(pos)) for sym, pos in positions.items()]
        for sym, pos in local_items:
            if pos.get("status") == "pending" or pos.get("entry_time") is None:
                with positions_lock:
                    positions.pop(sym, None)
                _record_pending_cancel("manual_timeout_global")
            else:
                close_position(sym, "timeout", close_price=exit_prices.get(sym) or get_price(sym) or pos.get("entry"))
        with positions_lock:
            positions.clear()
        tg_send(chat_id, "✅ <b>TIMEOUT GLOBAL SELESAI</b>\n"
                        f"Posisi ditutup: <b>{len(positions_remote)}</b>\n"
                        f"Symbol dibersihkan: <b>{len(symbols)}</b>\n"
                        "Semua posisi dan semua order Binance terverifikasi <b>0</b>.")
        return True
    except Exception as e:
        tg_send(chat_id, "🚨 <b>TIMEOUT GLOBAL BELUM SELESAI</b>\n"
                        f"<code>{str(e)[:500]}</code>\n"
                        "Scanner/entry baru tidak boleh dianggap aman. Cek Binance dan lakukan /ok SYMBOL untuk posisi yang masih tercatat.")
        log.error(f"[TIMEOUT GLOBAL] cleanup gagal: {e}")
        return False


# ── TP/SL sekarang WAJIB lewat Algo Order API (Binance migrasi order kondisional
# ke /fapi/v1/algoOrder per 9 Des 2025 — endpoint /fapi/v1/order lama menolaknya
# dengan error -4120). Field beda dari order biasa: stopPrice->triggerPrice,
# orderId->algoId, status->algoStatus. ──

def place_sl_order(symbol, is_buy, sl_price, quantity, client_algo_id=None):
    close_side = "SELL" if is_buy else "BUY"
    info = get_symbol_filters(symbol)
    tick = info["tickSize"]; step = info["stepSize"]; qty_prec = info.get("qtyPrecision", 8)
    qty_rounded = round_qty(quantity, step, qty_prec)
    client_algo_id = client_algo_id or _new_client_id("SL")
    params = {
        "algoType": "CONDITIONAL", "symbol": symbol, "side": close_side,
        "type": "STOP_MARKET", "triggerPrice": round_to_tick(sl_price, tick),
        "quantity": qty_rounded, "reduceOnly": "true", "workingType": "MARK_PRICE",
        "clientAlgoId": client_algo_id,
    }
    try:
        return _binance_signed("POST", "/fapi/v1/algoOrder", params)
    except BinanceUnknownExecutionError:
        found = _find_open_algo_by_client_id(symbol, client_algo_id)
        if found is not None:
            return found
        raise


def place_tp_sl(symbol, is_buy, tp_price, sl_price, quantity):
    """Create verified TP+SL pair. On partial creation, clean up and fail closed."""
    close_side = "SELL" if is_buy else "BUY"
    info = get_symbol_filters(symbol)
    tick, step = info["tickSize"], info["stepSize"]
    qty_prec = info.get("qtyPrecision", 8)
    qty_rounded = round_qty(quantity, step, qty_prec)
    tp_client = _new_client_id("TP")
    sl_client = _new_client_id("SL")
    tp = None; sl = None
    try:
        tp = _binance_signed("POST", "/fapi/v1/algoOrder", {
            "algoType": "CONDITIONAL", "symbol": symbol, "side": close_side,
            "type": "TAKE_PROFIT_MARKET", "triggerPrice": round_to_tick(tp_price, tick),
            "quantity": qty_rounded, "reduceOnly": "true", "workingType": "MARK_PRICE",
            "clientAlgoId": tp_client,
        })
    except BinanceUnknownExecutionError:
        tp = _find_open_algo_by_client_id(symbol, tp_client)
        if tp is None:
            raise

    try:
        sl = place_sl_order(symbol, is_buy, sl_price, quantity, client_algo_id=sl_client)
    except Exception:
        # If TP exists, best-effort remove it so we never leave a half-protected pair.
        try:
            _cancel_all_algo_orders_verified(symbol)
        finally:
            raise

    # Exchange-side verification is mandatory before local state is allowed to advance.
    _verify_protection_pair(symbol, is_buy, tp_price, sl_price, qty_rounded)
    return tp, sl


def cancel_algo_order(algo_id):
    if not algo_id: return None
    try:
        return _binance_signed("DELETE", "/fapi/v1/algoOrder", {"algoId": algo_id})
    except BinanceUnknownExecutionError:
        # Query-all-by-symbol is unavailable without symbol; caller verification
        # is authoritative. Do not blindly repeat a cancel.
        raise
    except Exception as e:
        log.warning(f"[cancel_algo_order] #{algo_id}: {e}")
        return None


def get_algo_order_status(algo_id):
    return _binance_signed("GET", "/fapi/v1/algoOrder", {"algoId": algo_id})


def cancel_all_algo_orders(symbol):
    try:
        return _binance_signed("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol})
    except BinanceUnknownExecutionError:
        # Verification immediately after the call determines whether cleanup happened.
        raise
    except Exception as e:
        log.warning(f"[cancel_all_algo_orders] {symbol}: {e}")
        return None


def get_real_balance():
    """Return (available, total) USDT, atau (None, None) kalau gagal."""
    try:
        rows = _binance_signed("GET", "/fapi/v2/balance", {})
        for r in rows:
            if r["asset"] == "USDT":
                return float(r["availableBalance"]), float(r["balance"])
    except Exception as e:
        log.warning(f"[get_real_balance] {e}")
    return None, None


def get_public_ip():
    try:
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except Exception:
        return "unknown"


def _binance_klines(symbol, interval, limit):
    raw = fapi_get("/fapi/v1/klines",
                   {"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(raw, list) or len(raw) < min(limit, 40):
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=[
        "ts","open","high","low","close","volume",
        "cts","qvol","trades","tbv","tbq","ign"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open","high","low","close","volume"]].dropna()

def _binance_price(symbol):
    d = fapi_get("/fapi/v1/ticker/price", {"symbol": symbol})
    return float(d["price"])

def _binance_top_coins(exclude_syms):
    tickers = fapi_get("/fapi/v1/ticker/24hr")
    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t["quoteVolume"]) > 5_000_000
        and abs(float(t.get("priceChangePercent","0"))) < 15
        and t["symbol"] not in exclude_syms
    ]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]


# ── BYBIT (fallback tier-3) ────────────────────────────────────────────
def _bybit_klines(symbol, interval, limit):
    iv = INTERVAL_MAP.get(interval, "15")
    d = _raw_get(f"{BYBIT}/v5/market/kline", {
        "category":"linear","symbol":symbol,
        "interval":iv,"limit":limit
    })
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit kline error: {d.get('retMsg')}")
    rows = d["result"]["list"]
    if not rows or len(rows) < min(limit, 40):
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["ts"].astype(float), unit="ms")
    df = df.sort_index()
    return df[["open","high","low","close","volume"]].dropna()

def _bybit_price(symbol):
    d = _raw_get(f"{BYBIT}/v5/market/tickers",
                 {"category":"linear","symbol":symbol})
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit ticker error: {d.get('retMsg')}")
    return float(d["result"]["list"][0]["lastPrice"])

def _bybit_top_coins(exclude_syms):
    d = _raw_get(f"{BYBIT}/v5/market/tickers", {"category":"linear"})
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit tickers error: {d.get('retMsg')}")
    items = d["result"]["list"]
    usdt = [
        t for t in items
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t.get("turnover24h","0")) > 5_000_000
        and abs(float(t.get("price24hPcnt","0"))) < 0.15
        and t["symbol"] not in exclude_syms
    ]
    usdt.sort(key=lambda x: float(x.get("turnover24h","0")), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]


# ── COINGECKO (fallback tier-4, DARURAT — harga saja) ──────────────────
def _coingecko_price(symbol):
    cid = COINGECKO_ID_MAP.get(symbol)
    if not cid:
        return None
    try:
        d = _raw_get("https://api.coingecko.com/api/v3/simple/price",
                     {"ids": cid, "vs_currencies": "usd"}, retries=1)
        p = d.get(cid, {}).get("usd")
        return float(p) if p is not None else None
    except Exception as e:
        log.warning(f"[price/coingecko] {symbol}: {e}")
        return None


# ── WEBSOCKET FEED (tier-1) ─────────────────────────────────────────────
class BinanceWSFeed:
    """
    Satu koneksi WS gabungan (raw stream endpoint, subscribe dinamis) ke
    Binance Futures:
      - !ticker@arr        → harga + statistik 24 jam SEMUA simbol tiap
                              ~1 detik. Menggantikan polling REST batch
                              utk get_price() & get_top_coins() sepenuhnya
                              begitu WS ini live — jauh lebih hemat rate
                              limit/risiko IP ban dibanding sebelumnya.
      - <sym>@kline_<itv>  → update candle real-time, HANYA utk pasangan
                              (simbol, interval) yang benar-benar diminta
                              get_klines() — subscribe on-demand (lazy),
                              bukan semua 50 koin x semua interval sekaligus,
                              biar hemat kuota stream & bandwidth.

    Catatan penting: WS TIDAK BISA memberi histori candle sebelum koneksi
    dibuka — itu keterbatasan protokol, bukan celah desain. Karena itu
    setiap (simbol, interval) yang baru pertama kali diminta di-backfill
    SEKALI via REST (Binance → Bybit), baru setelah itu WS yang menjaga
    buffer tetap update tanpa REST lagi.

    Auto-reconnect dgn exponential backoff (1s→30s), auto re-subscribe
    semua stream yang lagi aktif begitu reconnect berhasil.
    """
    KLINE_INTERVALS = ("1m", "15m", "1h", "1d")
    MAX_CANDLES  = {"1m": 300, "15m": 300, "1h": 300, "1d": 150}
    STALE_AFTER_SEC   = 30     # >30s tanpa pesan masuk → anggap WS mati
    STREAM_IDLE_SEC   = 1800   # (simbol,interval) tak dipakai 30menit → unsubscribe

    def __init__(self):
        self._lock       = threading.Lock()
        self._send_lock  = threading.Lock()
        self._klines     = {}     # {(sym,itv): deque([{t,o,h,l,c,v}, ...])}
        self._ticker     = {}     # {sym: {"symbol","price","qvol","chg"}}
        self._last_used  = {}     # {(sym,itv): timestamp terakhir diminta}
        self._subscribed = set()  # stream string yg lagi aktif di WS
        self._ws         = None
        self._last_msg   = 0.0
        self._connected  = False
        self._stop       = False
        self._backoff    = 1

    # ── public ──
    def start(self):
        if not _WS_LIB_OK:
            log.error("[ws] Modul 'websocket-client' belum terpasang — "
                      "TAMBAHKAN 'websocket-client' ke requirements.txt. "
                      "Bot tetap jalan tapi full REST-only (Binance→Bybit) "
                      "sampai modul ini ada.")
            return
        threading.Thread(target=self._run_forever, daemon=True).start()

    def is_fresh(self):
        return self._connected and (time.time() - self._last_msg) < self.STALE_AFTER_SEC

    def get_price(self, symbol):
        with self._lock:
            d = self._ticker.get(symbol)
            return d["price"] if d else None

    def get_top_coins_raw(self):
        with self._lock:
            return list(self._ticker.values())

    def get_klines(self, symbol, interval, limit=250):
        """Return klines dari buffer WS internal (data yg sudah di-backfill & live-update).
        Dipanggil dari module-level get_klines() sebagai fallback setelah REST gagal."""
        with self._lock:
            buf = self._klines.get((symbol, interval))
            if not buf:
                return pd.DataFrame()
            rows = list(buf)[-limit:]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], unit="ms")
        df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"}, inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    def seed_klines(self, symbol, interval, df):
        """Masukkan histori yang SUDAH diperoleh scanner ke buffer WS.

        Ini sengaja tidak melakukan REST request. Setelah seed, WS tinggal
        melanjutkan candle secara live sehingga request histori tidak diulang.
        """
        if not _WS_LIB_OK or df is None or df.empty:
            return
        try:
            rows = []
            for idx, row in df.tail(self.MAX_CANDLES.get(interval, 250)).iterrows():
                ts = int(pd.Timestamp(idx).timestamp() * 1000)
                rows.append({
                    "t": ts, "o": float(row["open"]), "h": float(row["high"]),
                    "l": float(row["low"]), "c": float(row["close"]), "v": float(row["volume"]),
                })
            if not rows:
                return
            with self._lock:
                self._klines[(symbol, interval)] = deque(rows, maxlen=self.MAX_CANDLES.get(interval, 250))
                self._last_used[(symbol, interval)] = time.time()
            self._subscribe_kline(symbol, interval)
        except Exception as e:
            log.warning(f"[ws-seed] {symbol} {interval}: {e}")

    def ensure_symbol_interval(self, symbol, interval):
        """Dipanggil tiap get_klines() — backfill SEKALI kalau baru,
        subscribe stream kalau belum, update timestamp pemakaian terakhir."""
        if not _WS_LIB_OK:
            return
        with self._lock:
            have = (symbol, interval) in self._klines
            self._last_used[(symbol, interval)] = time.time()
        if not have:
            self._backfill(symbol, interval)
        self._subscribe_kline(symbol, interval)

    def cleanup_stale_streams(self):
        """Unsubscribe & buang buffer (simbol,interval) yg tidak dipakai
        >30menit — dipanggil berkala dari watchdog thread, biar jumlah
        stream aktif tetap proporsional dgn pool koin yang sedang jalan."""
        now = time.time()
        with self._lock:
            stale = [k for k, ts in self._last_used.items()
                     if now - ts > self.STREAM_IDLE_SEC]
        for (sym, itv) in stale:
            self._unsubscribe_kline(sym, itv)
            with self._lock:
                self._klines.pop((sym, itv), None)
                self._last_used.pop((sym, itv), None)
        if stale:
            log.info(f"[ws] cleanup {len(stale)} stream idle >30menit")

    # ── internal: backfill histori awal via REST ──
    def _backfill(self, symbol, interval):
        limit = self.MAX_CANDLES.get(interval, 250)
        df, src = pd.DataFrame(), None
        try:
            df = _binance_klines(symbol, interval, limit)
            if not df.empty: src = "binance"
        except Exception as e:
            log.warning(f"[ws-backfill/binance] {symbol} {interval}: {e}")
        if df.empty:
            try:
                df = _bybit_klines(symbol, interval, limit)
                if not df.empty: src = "bybit"
            except Exception as e:
                log.warning(f"[ws-backfill/bybit] {symbol} {interval}: {e}")
        if df.empty:
            log.warning(f"[ws-backfill] {symbol} {interval} GAGAL TOTAL "
                        f"(binance+bybit) — coba lagi di pemanggilan berikutnya")
            return
        rows = deque(maxlen=limit)
        for ts, r in df.iterrows():
            rows.append({"t": int(ts.timestamp()*1000), "o": float(r.open),
                         "h": float(r.high), "l": float(r.low),
                         "c": float(r.close), "v": float(r.volume)})
        with self._lock:
            self._klines[(symbol, interval)] = rows
        log.info(f"[ws-backfill] {symbol} {interval} OK via {src} ({len(rows)} candle)")

    # ── internal: lifecycle WS ──
    def _run_forever(self):
        while not self._stop:
            try:
                self._connect()
            except Exception as e:
                log.warning(f"[ws] koneksi error: {e}")
            self._connected = False
            if self._stop:
                break
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, 30)

    def _connect(self):
        self._ws = websocket.WebSocketApp(
            BINANCE_WS_URL,
            on_open=self._on_open, on_message=self._on_message,
            on_error=self._on_error, on_close=self._on_close)
        self._ws.run_forever(ping_interval=180, ping_timeout=10)

    def _on_open(self, ws):
        self._connected = True
        self._backoff = 1
        self._last_msg = time.time()
        log.info("[ws] Binance Futures WS terhubung")
        self._send_subscribe(["!ticker@arr"])
        with self._lock:
            keys = list(self._klines.keys())
        if keys:
            streams = [f"{sym.lower()}@kline_{itv}" for sym, itv in keys]
            self._send_subscribe(streams)

    def _on_message(self, ws, raw):
        self._last_msg = time.time()
        try:
            msg = json.loads(raw)
        except Exception:
            return
        if isinstance(msg, list):
            self._handle_ticker_array(msg)
        elif isinstance(msg, dict) and msg.get("e") == "24hrTicker":
            self._handle_ticker_array([msg])
        elif isinstance(msg, dict) and msg.get("e") == "kline":
            self._handle_kline(msg)

    def _handle_ticker_array(self, arr):
        with self._lock:
            for t in arr:
                try:
                    sym = t["s"]
                    self._ticker[sym] = {
                        "symbol": sym, "price": float(t["c"]),
                        "qvol": float(t["q"]), "chg": float(t["P"]),
                    }
                except Exception:
                    continue

    def _handle_kline(self, msg):
        k = msg["k"]; sym = msg["s"]; itv = k["i"]
        key = (sym, itv)
        row = {"t": k["t"], "o": float(k["o"]), "h": float(k["h"]),
               "l": float(k["l"]), "c": float(k["c"]), "v": float(k["v"])}
        with self._lock:
            buf = self._klines.get(key)
            if buf is None:
                return   # belum di-backfill — abaikan sampai diminta
            if buf and buf[-1]["t"] == row["t"]:
                buf[-1] = row
            else:
                buf.append(row)

    def _on_error(self, ws, err):
        log.warning(f"[ws] error: {err}")

    def _on_close(self, ws, code, msg):
        self._connected = False
        log.warning(f"[ws] tertutup (code={code})")

    def _send_subscribe(self, streams):
        if not streams or not self._ws:
            return   # belum connect — akan di-resubscribe otomatis di _on_open
        try:
            with self._send_lock:
                self._ws.send(json.dumps({
                    "method":"SUBSCRIBE","params":streams,
                    "id": int(time.time()*1000) % 100000}))
            with self._lock:
                self._subscribed |= set(streams)
        except Exception as e:
            log.warning(f"[ws] gagal subscribe {streams}: {e}")

    def _subscribe_kline(self, symbol, interval):
        stream = f"{symbol.lower()}@kline_{interval}"
        with self._lock:
            already = stream in self._subscribed
        if not already:
            self._send_subscribe([stream])

    def _unsubscribe_kline(self, symbol, interval):
        stream = f"{symbol.lower()}@kline_{interval}"
        try:
            with self._send_lock:
                if self._ws:
                    self._ws.send(json.dumps({
                        "method":"UNSUBSCRIBE","params":[stream],
                        "id": int(time.time()*1000) % 100000}))
            with self._lock:
                self._subscribed.discard(stream)
        except Exception:
            pass


ws_feed = BinanceWSFeed()


# ── FUNGSI PUBLIK — signature SAMA PERSIS dgn sebelumnya, jadi seluruh
#    kode bot (scoring, monitor posisi, dsb) TIDAK perlu diubah sama sekali ──
def get_price(symbol):
    """Saat Binance pause: jangan hit REST fallback. Gunakan WS/local cache saja."""
    if _binance_is_scan_paused():
        p = ws_feed.get_price(symbol)
        return p
    try:
        return _binance_price(symbol)
    except Exception as e:
        log.warning(f"[price/binance] {symbol}: {e} — fallback")
        if _binance_is_scan_paused():
            return ws_feed.get_price(symbol)
    for _ in range(2):
        try:
            return _bybit_price(symbol)
        except Exception as e:
            log.warning(f"[price/bybit] {symbol}: {e}")
            time.sleep(1)
    if ws_feed.is_fresh():
        p = ws_feed.get_price(symbol)
        if p is not None:
            return p
    p = _coingecko_price(symbol)
    if p is not None:
        return p
    return None

def get_klines(symbol, interval, limit=250):
    """Normal market-data accessor. Existing behavior retained for execution.

    Scanner optimization lives in get_scan_klines(); it uses a dedicated
    freshness cache and never forces a WS backfill synchronously.
    """
    if _binance_is_scan_paused():
        df = ws_feed.get_klines(symbol, interval, limit) if ws_feed.is_fresh() else pd.DataFrame()
        return df if df is not None else pd.DataFrame()
    ws_feed.ensure_symbol_interval(symbol, interval)
    if ws_feed.is_fresh():
        df = ws_feed.get_klines(symbol, interval, limit)
        if df is not None and not df.empty:
            return df
    try:
        df = _binance_klines(symbol, interval, limit)
        if not df.empty:
            return df
    except Exception as e:
        log.warning(f"[klines/binance] {symbol}: {e}")
        if _binance_is_scan_paused():
            return ws_feed.get_klines(symbol, interval, limit) if ws_feed.is_fresh() else pd.DataFrame()
    try:
        df = _bybit_klines(symbol, interval, limit)
        if not df.empty:
            log.info(f"[klines/bybit fallback] {symbol} {interval} OK")
            return df
    except Exception as e:
        log.warning(f"[klines/bybit] {symbol}: {e}")
    return pd.DataFrame()

def get_scan_klines(symbol, interval, limit=250):
    """Scanner-only candle accessor: cache-first, single-flight, no duplicate backfill.

    Prinsip utama V11: mempercepat scan dengan MENGURANGI request, bukan dengan
    menaikkan concurrency/request rate. Jika cache masih fresh, tidak ada HTTP
    request sama sekali. Jika cache miss, hanya satu fetch untuk key tersebut;
    histori yang berhasil kemudian di-seed ke WS sehingga tidak di-backfill dua kali.
    """
    if _binance_is_scan_paused():
        cached = _scan_cache_get(symbol, interval, limit)
        if cached is not None:
            return cached
        df = ws_feed.get_klines(symbol, interval, limit) if ws_feed.is_fresh() else pd.DataFrame()
        return df if df is not None else pd.DataFrame()

    cached = _scan_cache_get(symbol, interval, limit)
    if cached is not None:
        return cached

    key = (symbol, interval)
    lock = _scan_key_lock(key)
    with lock:
        # Double-check after waiting for another worker/fetcher.
        cached = _scan_cache_get(symbol, interval, limit)
        if cached is not None:
            return cached

        # A pre-existing WS buffer is free data; promote it to scan cache.
        if ws_feed.is_fresh():
            ws_df = ws_feed.get_klines(symbol, interval, limit)
            if ws_df is not None and not ws_df.empty and len(ws_df) >= min(limit, 40):
                _scan_cache_put(symbol, interval, ws_df, "ws")
                return ws_df.tail(limit).copy()

        started = time.monotonic()
        source = None
        df = pd.DataFrame()

        # Do NOT call ensure_symbol_interval() here. That would synchronously
        # backfill through WS and then make the scanner wait on another REST path.
        # Fetch exactly once, cache it, seed WS, and let WS maintain it thereafter.
        try:
            # If Binance weight is already near the soft governor, prefer the
            # existing fallback instead of sleeping an entire minute inside a scan.
            high_weight = (_binance_weight_1m is not None and
                           _binance_weight_1m >= BINANCE_WEIGHT_SOFT_LIMIT)
            if not high_weight:
                df = _binance_klines(symbol, interval, limit)
                if not df.empty:
                    source = "binance"
        except BinanceCooldownError:
            raise
        except Exception as e:
            log.warning(f"[scan-data/binance] {symbol} {interval}: {e}")

        if df.empty:
            try:
                df = _bybit_klines(symbol, interval, limit)
                if not df.empty:
                    source = "bybit"
            except Exception as e:
                log.warning(f"[scan-data/bybit] {symbol} {interval}: {e}")

        if df.empty:
            return pd.DataFrame()

        _scan_cache_put(symbol, interval, df, source or "unknown")
        # Seed WS from the same dataframe. This adds zero REST requests.
        ws_feed.seed_klines(symbol, interval, df)
        elapsed = time.monotonic() - started
        log.info(f"[scan-data] {symbol} {interval} source={source} fetch={elapsed:.2f}s")
        return df.tail(limit).copy()

def _record_scan_telemetry(data):
    global _last_scan_telemetry
    with _scan_telemetry_lock:
        _last_scan_telemetry = dict(data)

def get_last_scan_telemetry():
    with _scan_telemetry_lock:
        return dict(_last_scan_telemetry)

last_scanned_coins = []
last_scanned_at = None
_last_scanned_lock = threading.Lock()

def get_top_coins():
    """Wrapper: panggil _get_top_coins_impl() lalu cache hasilnya ke
    last_scanned_coins — dipakai command /koin supaya bisa nampilin daftar
    koin yang di-scan TANPA perlu fetch ulang / ikut nambah scan_counter
    (yang dipakai buat hitung durasi ban)."""
    coins = _get_top_coins_impl()
    global last_scanned_coins, last_scanned_at
    with _last_scanned_lock:
        last_scanned_coins = coins
        last_scanned_at = time.time()
    return coins

def _get_top_coins_impl():
    """Ambil top coins. Saat Binance pause, seluruh scan berhenti.
    Fallback Bybit/WS hanya boleh dipakai ketika Binance tidak sedang dalam global pause.
    """
    if _binance_is_scan_paused():
        log.warning(f"[scan] DITAHAN — Binance cooldown aktif {_binance_cooldown_remaining():.0f}s")
        return []
    global scan_counter
    with ban_lock:
        scan_counter += 1
        to_unban = []
        for sym, meta in list(banned_coins.items()):
            if isinstance(meta, tuple):
                banned_at, dur = meta
                expired = scan_counter - banned_at >= dur
            else:
                expired = scan_counter - float(meta.get("banned_at", scan_counter)) >= float(meta.get("duration", 0))
            if expired and meta != "PERMANENT":
                to_unban.append(sym)
        for sym in to_unban:
            meta = banned_coins.pop(sym, {})
            dur = meta[1] if isinstance(meta, tuple) else meta.get("duration", 0)
            log.info(f"[unban] {sym} kembali aktif setelah {float(dur):g} scan")
        cur_ban = set(banned_coins.keys())

    with positions_lock:
        active_syms = set(positions.keys())

    exclude_syms = cur_ban | active_syms

    # Binance REST
    try:
        coins = _binance_top_coins(exclude_syms)
        if coins:
            return coins
        if _binance_is_scan_paused():
            log.warning("[top_coins/binance] kosong karena circuit breaker aktif — TIDAK fallback.")
            return []
        log.warning("[top_coins/binance] kosong, coba Bybit...")
    except BinanceCooldownError:
        log.warning("[top_coins/binance] rate-limit/ban — TIDAK fallback, scan cycle dihentikan.")
        return []
    except Exception as e:
        log.warning(f"[top_coins/binance] {e}")
        if _binance_is_scan_paused():
            return []
    # Bybit fallback
    try:
        coins = _bybit_top_coins(exclude_syms)
        if coins:
            log.info(f"[top_coins/bybit fallback] {len(coins)} koin")
            return coins
        log.warning("[top_coins/bybit] kosong, coba WS...")
    except Exception as e:
        log.warning(f"[top_coins/bybit] {e} — coba WS...")
    # WS fallback TERAKHIR
    if ws_feed.is_fresh():
        raw = ws_feed.get_top_coins_raw()
        usdt = [
            t for t in raw
            if t["symbol"].endswith("USDT")
            and 0.0001 < t["price"] < MAX_PRICE
            and t["qvol"] > 5_000_000
            and abs(t["chg"]) < 15
            and t["symbol"] not in exclude_syms
        ]
        if usdt:
            usdt.sort(key=lambda x: x["qvol"], reverse=True)
            log.warning("[top_coins/ws fallback] REST Binance & Bybit gagal")
            return [t["symbol"] for t in usdt[:TOP_N_COINS]]
    return []


_PRICE_REFRESH_SEC = 10   # interval cek watchdog (detik)

def _price_cache_loop():
    """
    DULU: thread polling REST batch tiap 10 detik utk cache harga posisi.
    SEKARANG: REST (Binance→Bybit) adalah sumber data UTAMA di get_price/
    get_klines/get_top_coins; WS cuma buffer fallback TERAKHIR yang
    disiapkan di background. Karena WS bukan sumber utama lagi, hidup-
    matinya WS BUKAN kejadian penting bagi operasional bot — jadi TIDAK
    lagi dikirim ke Telegram tiap kali flap (dulu ini yang bikin spam
    notifikasi "WS pulih"/"WS terputus" berulang-ulang). Status WS tetap
    dicatat di log untuk keperluan debug, dan stream kline yang sudah
    tidak dipakai >30 menit tetap dibersihkan di sini.
    """
    was_fresh = None   # None = belum pernah dicek
    while True:
        try:
            fresh = ws_feed.is_fresh()
            if was_fresh is not None and was_fresh != fresh:
                if fresh:
                    log.info("[ws-watchdog] WS fallback tersedia lagi (buffer siap)")
                else:
                    log.info("[ws-watchdog] WS fallback tidak tersedia — tidak masalah, REST tetap sumber utama")
            was_fresh = fresh
            ws_feed.cleanup_stale_streams()
        except Exception as e:
            log.error(f"[ws-watchdog] {e}")
        time.sleep(_PRICE_REFRESH_SEC)

# ═════════════════════════════════════════════
# INDIKATOR
# ═════════════════════════════════════════════
def run_scan_once(chat_id):
    global early_reject_remaining
    """Scan universe with cached market data and record rich market context.

    V19 adds *derived* market breadth/relative-strength/regime telemetry only.
    It does not add a Binance endpoint or a new request: the M15/H1/D1 frames
    already fetched for strategy analysis are reused in memory.
    """
    scan_started=time.monotonic()
    if _binance_is_scan_paused():
        _notify_binance_pause_once(chat_id); return []
    tg_send(chat_id, f"🔍 Scanning {TOP_N_COINS} koin...")
    if _binance_is_scan_paused():
        _notify_binance_pause_once(chat_id); return []
    try:
        symbols=get_top_coins()
    except BinanceCooldownError as e:
        tg_send(chat_id, f"⏸️ <b>Scan dihentikan</b> — Binance rate-limit/ban aktif.\n<code>{str(e)[:180]}</code>"); return []
    except Exception as e:
        tg_send(chat_id, f"⚠️ Market data error: <code>{str(e)[:150]}</code>"); return []
    if not symbols:
        tg_send(chat_id, "⚠️ Tidak ada koin tersedia untuk di-scan."); return []

    data_started=time.monotonic(); results=[]; all_scan_confidences=[]; market_rows=[]
    analyzed_symbols=cache_hits=cache_misses=failed_symbols=low_conf_count=below_threshold_count=0
    for idx,sym in enumerate(symbols,1):
        if _binance_is_scan_paused():
            log.warning("[scan] Binance pause aktif — scan cycle dihentikan di tengah jalan."); break
        log.info(f"[scan {idx:02d}/{len(symbols)}] {sym}")
        try:
            before=_scan_cache_stats()
            h1=get_scan_klines(sym,"1h",250); m15=get_scan_klines(sym,"15m",250)
            try: d1=get_scan_klines(sym,"1d",100)
            except BinanceCooldownError: raise
            except Exception: d1=None
            after=_scan_cache_stats(); cache_misses += max(0,after[0]-before[0])
            r=full_analyze(h1,m15,d1,symbol=sym)
            if isinstance(r,dict):
                _ml_record_signal_metadata(r)
                analyzed_symbols+=1; conf=float(r.get("confidence",0) or 0); all_scan_confidences.append(conf)
                row=_market_feature_row(sym,h1,m15,r)
                row.update({"scan_time":time.time(),"run_id":research_run_id,"scan_counter":scan_counter})
                lf = r.setdefault("learning_features", {})
                lf.update({
                    "market_bull_breadth": float(row.get("bullish_breadth_pct") or 0.0) / 100.0,
                    "market_bear_breadth": float(row.get("bearish_breadth_pct") or 0.0) / 100.0,
                    "market_efficiency": float(row.get("efficiency_4h") or 0.0),
                    "market_relative_volume": float(row.get("relative_volume") or 0.0),
                    "btc_1h": float(row.get("price_1h_pct") or 0.0) if sym == "BTCUSDT" else 0.0,
                    "btc_4h": float(row.get("price_4h_pct") or 0.0) if sym == "BTCUSDT" else 0.0,
                    "symbol_rs_1h": float(row.get("relative_strength_1h_pct") or 0.0) if row.get("relative_strength_1h_pct") is not None else 0.0,
                    "symbol_rs_4h": float(row.get("relative_strength_4h_pct") or 0.0) if row.get("relative_strength_4h_pct") is not None else 0.0,
                })
                market_rows.append(row)
                active_threshold = int((_ml_current_champion() or {}).get("confidence_min", STRATEGY_CONFIDENCE_THRESHOLD)) if FULL_MODE else STRATEGY_CONFIDENCE_THRESHOLD
                cutoff=float(active_threshold)/2.0
                if conf<=cutoff:
                    low_conf_count+=1; _record_low_confidence_event(sym,conf,cutoff,r.get("decision"),r.get("entry_label"))
                    _ban_coin(sym,reason=f"low confidence {conf:.1f} <= {cutoff:.1f}",duration=BAN_DURATION_SCANS,kind="low_confidence",confidence=conf)
                if conf<active_threshold: below_threshold_count+=1
                if conf>=active_threshold:
                    r["market_context"]={k:v for k,v in row.items() if k not in {"scan_time","run_id","scan_counter"}}
                    results.append(r); log.info(f"[SIGNAL] {sym} {r.get('decision')} confidence={conf:.1f}")
                else:
                    log.info(f"[FILTER] {sym} confidence={conf:.1f} < {STRATEGY_CONFIDENCE_THRESHOLD}")
        except BinanceCooldownError:
            log.warning(f"[scan] {sym}: Binance cooldown aktif — scan cycle dihentikan aman."); break
        except Exception as e:
            failed_symbols+=1; log.debug(f"[scan] {sym}: {e}")
        time.sleep(0.05)

    # Enrich per-symbol rows with relative strength vs BTC without a new request.
    btc_row=next((x for x in market_rows if x.get("symbol")=="BTCUSDT"),None)
    btc_r1=btc_row.get("price_1h_pct") if btc_row else None; btc_r4=btc_row.get("price_4h_pct") if btc_row else None
    for row in market_rows:
        row["relative_strength_1h_pct"]=(row.get("price_1h_pct")-btc_r1) if btc_r1 is not None and row.get("price_1h_pct") is not None else None
        row["relative_strength_4h_pct"]=(row.get("price_4h_pct")-btc_r4) if btc_r4 is not None and row.get("price_4h_pct") is not None else None
        if row.get("symbol")=="BTCUSDT":
            row["relative_strength_1h_pct"]=0.0; row["relative_strength_4h_pct"]=0.0
        for result in results:
            if result.get("symbol") == row.get("symbol"):
                lf = result.setdefault("learning_features", {})
                lf.update({
                    "market_bull_breadth": lf.get("market_bull_breadth", 0.0),
                    "btc_1h": float(btc_r1 or 0.0),
                    "btc_4h": float(btc_r4 or 0.0),
                    "symbol_rs_1h": float(row.get("relative_strength_1h_pct") or 0.0),
                    "symbol_rs_4h": float(row.get("relative_strength_4h_pct") or 0.0),
                })
    _record_market_context(market_rows)
    mc=_summarize_market_context(market_rows)

    data_elapsed=time.monotonic()-data_started; total_elapsed=time.monotonic()-scan_started
    cache_total,cache_fresh=_scan_cache_stats(); avg_conf=(sum(all_scan_confidences)/len(all_scan_confidences)) if all_scan_confidences else None
    telemetry={"duration_sec":round(total_elapsed,2),"data_phase_sec":round(data_elapsed,2),"symbols_requested":len(symbols),"analyzed_symbols":analyzed_symbols,"avg_confidence":round(avg_conf,2) if avg_conf is not None else None,"min_confidence":round(min(all_scan_confidences),2) if all_scan_confidences else None,"max_confidence":round(max(all_scan_confidences),2) if all_scan_confidences else None,"low_confidence_count":low_conf_count,"below_threshold_count":below_threshold_count,"results":len(results),"failed_symbols":failed_symbols,"cache_entries":cache_total,"cache_fresh":cache_fresh,"binance_weight_1m":_binance_weight_1m,"market_regime":mc.get("market_regime"),"bullish_breadth_pct":mc.get("bullish_breadth_pct"),"bearish_breadth_pct":mc.get("bearish_breadth_pct"),"median_efficiency_4h":mc.get("median_efficiency_4h"),"avg_relative_volume":mc.get("avg_relative_volume"),"btc_price_1h_pct":mc.get("btc_price_1h_pct"),"btc_price_4h_pct":mc.get("btc_price_4h_pct")}
    _record_scan_telemetry(telemetry)
    _record_scan_quality({"scan_time":time.time(),"run_id":research_run_id,"scan_counter":scan_counter,"symbols_requested":len(symbols),"symbols_analyzed":analyzed_symbols,"failed_symbols":failed_symbols,"avg_confidence":avg_conf,"min_confidence":(min(all_scan_confidences) if all_scan_confidences else None),"max_confidence":(max(all_scan_confidences) if all_scan_confidences else None),"low_confidence_count":low_conf_count,"below_threshold_count":below_threshold_count,"qualified_count":len(results),"early_rejected_count":0,"cache_entries":cache_total,"cache_fresh":cache_fresh,**mc})
    log.info("[SCAN SUMMARY] " + " | ".join(f"{k}={v}" for k,v in telemetry.items()))

    # /reject is expressed in SCAN CYCLES, not individual signals.
    # While warmup is active, every qualified signal from the current scan is
    # discarded together, none is banned, and exactly one scan is consumed.
    rejected_warmup=[]
    warmup_active=False
    with early_reject_lock:
        remaining=int(early_reject_remaining)
        if remaining>0:
            warmup_active=True
            rejected_warmup=list(results)
            results=[]
            early_reject_remaining=max(0, remaining-1)
    for r in rejected_warmup:
        log.info(f"[EARLY-REJECT][scan] {r.get('symbol','?')} confidence={float(r.get('confidence',0) or 0):.1f}")
    with scan_quality_lock:
        if scan_quality_history and scan_quality_history[-1].get("run_id")==research_run_id and scan_quality_history[-1].get("scan_counter")==scan_counter:
            scan_quality_history[-1]["early_rejected_count"]=len(rejected_warmup)
            scan_quality_history[-1]["warmup_reject_scan"]=warmup_active

    results.sort(key=lambda x:float(x.get("confidence",0) or 0),reverse=True)
    avg_txt=f"{avg_conf:.1f}%" if avg_conf is not None else "—"
    breadth_txt=(f"📈 Breadth BUY <b>{mc['bullish_breadth_pct']:.1f}%</b> | SELL <b>{mc['bearish_breadth_pct']:.1f}%</b> | Regime: <b>{mc['market_regime']}</b>" if mc.get('bullish_breadth_pct') is not None else "📈 Market context: <b>insufficient data</b>")
    rs_txt=(f"\n₿ BTC 1h: <b>{mc['btc_price_1h_pct']:+.2f}%</b> | BTC 4h: <b>{mc['btc_price_4h_pct']:+.2f}%</b>" if mc.get('btc_price_1h_pct') is not None else "")
    active_threshold = int((_ml_current_champion() or {}).get("confidence_min", STRATEGY_CONFIDENCE_THRESHOLD)) if FULL_MODE else STRATEGY_CONFIDENCE_THRESHOLD
    scan_meta=f"\n\n📊 Rata-rata confidence scan: <b>{avg_txt}</b> ({analyzed_symbols}/{len(symbols)} dianalisis)\nThreshold aktif: <b>{active_threshold}%</b>\n{breadth_txt}{rs_txt}"
    if warmup_active: scan_meta+=f"\n🛡️ Warmup reject: <b>{len(rejected_warmup)}</b> signal qualified dari scan ini ditolak"
    if not results:
        tg_send(chat_id,f"⚠️ Tidak ada setup dengan confidence ≥ {active_threshold}%."+scan_meta); return []
    summary="\n".join(f"• {r.get('symbol','?')} {r.get('decision','?')} — {float(r.get('confidence',0) or 0):.0f}%" for r in results)
    tg_send(chat_id,f"✅ <b>{len(results)} sinyal lolos</b> (threshold {active_threshold}%)\n\n{summary}{scan_meta}")
    return results


# ═════════════════════════════════════════════
# STATISTIK + BALANCE
# ═════════════════════════════════════════════

# ── Fee trading — dipakai update_stats() untuk PnL simulasi & real ────────
# Standar Binance USDT-M Futures VIP0, TANPA diskon BNB. SESUAIKAN kalau
# tier akun kamu beda (VIP lebih tinggi / diskon BNB aktif / dsb) —
# semakin akurat angka ini, semakin dekat statistik bot ke kenyataan.
ENTRY_FEE_PCT = 0.0002   # 0.02% — entry via limit order (biasanya maker)
EXIT_FEE_PCT  = 0.0005   # 0.05% — exit via SL/TP market-trigger (taker)
                            # P&L murni dari jarak SL/TP yang ditetapkan analisis:
                            #   TP hit → gain = posisi × (tp_dist / entry)
                            #   SL hit → loss = posisi × (sl_dist / entry)
                            # Nilai ini TIDAK mempengaruhi PENEMPATAN SL/TP —
                            # hanya memengaruhi simulasi saldo.
# POSITION_SIZE_PCT: SUDAH TIDAK DIPAKAI (lihat fix di update_stats di bawah)
# — dipertahankan sebagai konstanta supaya tidak menghapus definisi yang
# mungkin masih direferensikan dari luar, tapi update_stats() sekarang
# pakai MARGIN_USD × LEVERAGE (persis logika real trade), bukan ini lagi.
POSITION_SIZE_PCT = 100.0  # DEPRECATED — lihat catatan di atas

def _classify_close_result(result, entry=None, close_price=None, decision=None):
    """Normalize every closed trade into the three operational outcome buckets.

    TP is preserved when the exchange/engine explicitly confirms a take-profit path.
    Every other realized exit is classified economically: positive net PnL -> TRAIL,
    non-positive net PnL -> SL. The original reason is kept separately in
    ``close_reason`` for research, so UI/statistics never end up with an uncounted
    fourth outcome such as ``strategy`` or ``timeout``.
    """
    result = str(result or "strategy").strip().lower()
    if result == "tp":
        return "tp"
    if entry is not None and close_price is not None:
        try:
            entry = float(entry); close_price = float(close_price)
            if entry > 0:
                side = str(decision or "BUY").upper()
                direction = 1 if side == "BUY" else -1
                pnl_raw = ((close_price - entry) / entry) * direction
                net_pnl = pnl_raw - (ENTRY_FEE_PCT + EXIT_FEE_PCT)
                return "trail" if net_pnl > 0 else "sl"
        except Exception:
            pass
    # No realized price available: preserve explicit operational result, otherwise
    # fail closed into SL so every closed trade remains countable.
    if result == "trail":
        return "trail"
    if result == "sl":
        return "sl"
    return "sl"


def _update_trade_path_metrics(pos, price):
    """Track MFE/MAE and time-to-R milestones in-memory without API calls."""
    try:
        entry=float(pos.get("entry")); sl=float(pos["signal"].get("sl")); price=float(price)
        if entry <= 0: return
        side=str(pos["signal"].get("decision") or "BUY").upper()
        move=((price-entry)/entry*100.0) if side=="BUY" else ((entry-price)/entry*100.0)
        risk_pct=(abs(entry-sl)/entry*100.0)
        r=(move/risk_pct) if risk_pct else 0.0
        pos.setdefault("mfe_pct", 0.0); pos.setdefault("mae_pct", 0.0); pos.setdefault("mfe_r", 0.0); pos.setdefault("mae_r", 0.0)
        pos["mfe_pct"]=max(float(pos["mfe_pct"]), move); pos["mae_pct"]=min(float(pos["mae_pct"]), move)
        pos["mfe_r"]=max(float(pos["mfe_r"]), r); pos["mae_r"]=min(float(pos["mae_r"]), r)
        now=time.time(); start=pos.get("entry_time") or now; pos.setdefault("time_in_trade_sec", 0.0); pos["time_in_trade_sec"]=max(0.0, now-start)
        if r>=1.0 and pos.get("time_to_1r_sec") is None: pos["time_to_1r_sec"]=now-start
        if r>=2.0 and pos.get("time_to_2r_sec") is None: pos["time_to_2r_sec"]=now-start
    except Exception:
        return

def update_stats(result, entry=None, sl_p=None, tp_p=None, close_price=None,
                 sym=None, decision=None, entry_time=None, close_reason=None,
                 confidence=None, entry_label=None, rr=None, rsi=None,
                 struct_h1=None, d1_bias=None,
                 mfe_pct=None, mae_pct=None, mfe_r=None, mae_r=None,
                 time_in_trade_sec=None, time_to_1r_sec=None, time_to_2r_sec=None,
                 execution_mode=None, balance_anchor=None, trade_uid=None,
                 trail_update_count=0, trail_applied_count=0, trail_failed_count=0, trail_queued_count=0,
                 first_trail_r=None, last_trail_r=None, max_protected_r=None, learning_features=None, ml_model_version=None):
    """
    Hitung P&L simulasi murni dari jarak harga analisis (lihat komentar
    lama untuk detail model close_price). Tambahan: catat sym/decision/
    entry_time/exit_time + detail sinyal (confidence/entry_label/rr/rsi/
    struct_h1/d1_bias) ke pnl_history — bahan diagnosis strategy_logic.py
    tanpa perlu data tambahan lain (lihat /analyze).

    result: klasifikasi final yang sudah dinormalisasi. "trail" hanya dipakai
    jika exit trailing benar-benar menghasilkan PnL positif; trailing yang
    berakhir negatif dicatat sebagai "sl". "timeout" juga diklasifikasikan
    menjadi "trail" bila positif dan "sl" bila negatif.
    """
    result = _classify_close_result(result, entry=entry, close_price=close_price, decision=decision)
    with stat_lock:
        stats["total"] += 1
        if result in ("tp", "sl", "trail"):
            stats[result] = stats.get(result, 0) + 1

        if not entry or tp_p is None:
            return

        balance      = stats["balance"]
        # ── FIX "buat semirip mungkin" ──────────────────────────────────
        # Sebelumnya: position_usd = balance × 100% — simulasi selalu
        # bertaruh SELURUH saldo tiap trade (full compounding), padahal
        # real trading pakai MARGIN_USD × LEVERAGE (jumlah dolar FIXED,
        # kecil, diatur via /margin & /leverage), TIDAK ikut membesar
        # walau saldo real sudah tumbuh. Ini bikin bentuk kurva ekuitas
        # simulasi sama sekali beda dari real (simulasi: compounding
        # agresif; real: flat sizing) — bukan cuma soal fee/entry lagi,
        # tapi soal skala taruhan itu sendiri.
        # Sekarang KEDUA mode pakai rumus yang SAMA PERSIS seperti real
        # trade sizing, supaya kalau kamu ubah /margin atau /leverage,
        # simulasi otomatis ikut menyesuaikan — selaras terus dengan real.
        position_usd = round(MARGIN_USD * LEVERAGE, 6)
        direction_sign = 1 if tp_p > entry else -1

        if close_price is not None:
            ref_price = close_price
        elif result == "tp":
            ref_price = tp_p
        elif result == "sl" and sl_p is not None:
            ref_price = sl_p
        else:
            return

        pnl_pct_raw = (ref_price - entry) / entry * direction_sign
        # ── FIX "simulasi tidak real / win rate kelewat bagus" ──────────
        # Sebelumnya PnL dihitung MURNI dari selisih harga — nol biaya
        # trading. Di real trading, Binance SELALU potong fee tiap kali
        # entry (limit order → biasanya maker) DAN exit (SL/TP → market-
        # trigger → taker), otomatis kepotong dari saldo asli. Simulasi
        # tidak pernah mengurangi ini, jadi untuk trade RR ketat (SL 1-2%
        # dari harga, khas bot ini), fee round-trip yang kelihatannya kecil
        # bisa membalik hasil "breakeven/rugi tipis di real" jadi "menang"
        # di simulasi — bias sistemik yang bikin win rate simulasi selalu
        # kelihatan lebih bagus dari kenyataan.
        #
        # Angka ENTRY_FEE_PCT/EXIT_FEE_PCT di bawah = tarif standar Binance
        # USDT-M Futures VIP0 tanpa diskon BNB. Kalau akun kamu VIP lebih
        # tinggi / pakai diskon BNB / fee-nya beda, SESUAIKAN angka ini
        # (dekat bagian atas file) supaya makin presisi ke kondisi akunmu.
        # Diterapkan ke SIMULASI *dan* REAL supaya keduanya konsisten
        # mencerminkan biaya riil (real trading sebenarnya sudah kepotong
        # otomatis di Binance — ini menyamakan angka yang DITAMPILKAN bot
        # dengan kenyataan itu, bukan menambah biaya baru yang sungguhan).
        fee_pct = ENTRY_FEE_PCT + EXIT_FEE_PCT
        pnl_pct = pnl_pct_raw - fee_pct
        pnl_usd = round(position_usd * pnl_pct, 4)
        pct     = round(pnl_pct * 100, 3)
        stats["balance"] = round(balance + pnl_usd, 4)
        exit_ts = time.time()
        global trade_sequence
        trade_sequence += 1
        trade_record = {
            "trade_id": trade_sequence, "run_id": research_run_id, "trade_uid": trade_uid,
            "result": result, "close_reason": close_reason or result, "pct": pct,
            "pnl_usd": pnl_usd, "balance_after": stats["balance"],
            "symbol": sym, "decision": decision,
            "entry_time": entry_time, "exit_time": exit_ts,
            "entry": entry, "tp": tp_p, "sl": sl_p, "exit_price": ref_price,
            "confidence": confidence, "entry_label": entry_label, "rr": rr,
            "rsi": rsi, "struct_h1": struct_h1, "d1_bias": d1_bias,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "time_in_trade_sec": time_in_trade_sec,
            "time_to_1r_sec": time_to_1r_sec,
            "time_to_2r_sec": time_to_2r_sec,
            "execution_mode": execution_mode, "balance_anchor": balance_anchor,
            "trail_update_count": int(trail_update_count or 0), "trail_applied_count": int(trail_applied_count or 0),
            "trail_failed_count": int(trail_failed_count or 0), "trail_queued_count": int(trail_queued_count or 0),
            "first_trail_r": first_trail_r, "last_trail_r": last_trail_r, "max_protected_r": max_protected_r,
            "learning_features": dict(learning_features) if isinstance(learning_features, dict) else None,
            "ml_model_version": ml_model_version or "static",
        }
        # Full ledger: every closed trade in this research run.
        with trade_history_lock:
            trade_history.append(dict(trade_record))
        _ml_append_experience(trade_record)
        # Backward-compatible 20-trade view for /backtest and existing UI.
        stats["pnl_history"].append(dict(trade_record))

# Hitung alasan pending dibatalkan — biar bisa didiagnosis dari data,
# bukan tebak-tebakan (mis. "kenapa banyak batal?" jadi terjawab dari /stats).
pending_cancel_stats = {"tp_before_entry": 0, "expired": 0, "binance_reject": 0}
pending_cancel_lock = threading.Lock()

def _record_pending_cancel(reason_key):
    with pending_cancel_lock:
        pending_cancel_stats[reason_key] = pending_cancel_stats.get(reason_key, 0) + 1


def _avg_conf_for_result(hist, result_key):
    vals = []
    for row in hist:
        if str(row.get("result", "")).lower() == result_key:
            try:
                vals.append(float(row.get("confidence")))
            except (TypeError, ValueError):
                pass
    return (sum(vals) / len(vals)) if vals else None

def fmt_stats():
    with stat_lock:
        t, tp, sl = stats["total"], stats["tp"], stats["sl"]
        trail, bal = stats.get("trail", 0), stats["balance"]
        hist = list(stats["pnl_history"])
    with trade_history_lock:
        full_hist = [dict(x) for x in trade_history]
    wins = tp + trail
    wr = wins/(wins+sl)*100 if (wins+sl) > 0 else 0
    base = STARTING_BALANCE if not REAL_TRADE_ENABLED else (real_balance_snapshot if real_balance_snapshot is not None else bal)
    pnl = round(bal - base, 4)
    pnl_pct = round((pnl / base * 100), 2) if base else 0.0
    sgn = "+" if pnl >= 0 else ""
    hist_str = "\n".join(
        f"  {'🟢' if h.get('pnl_usd',0) > 0 else '🔴' if h.get('pnl_usd',0) < 0 else '⚪'} "
        f"{h.get('result','?').upper()} {'+' if h.get('pnl_usd',0)>=0 else ''}{h.get('pct',0):.2f}% → ${h.get('balance_after',0):.4f} | C{float(h.get('confidence',0) or 0):.0f}%"
        for h in reversed(hist[-5:])
    ) or "  (belum ada)"
    avg_all = None
    conf_vals = []
    for h in full_hist:
        try: conf_vals.append(float(h.get("confidence")))
        except (TypeError, ValueError): pass
    if conf_vals: avg_all = sum(conf_vals)/len(conf_vals)
    avg_tp = _avg_conf_for_result(full_hist, "tp")
    avg_trail = _avg_conf_for_result(full_hist, "trail")
    avg_sl = _avg_conf_for_result(full_hist, "sl")
    with pending_cancel_lock:
        pc = dict(pending_cancel_stats)
    total_cancel = sum(pc.values())
    cancel_line = ""
    if total_cancel > 0:
        cancel_line = (f"\n\n⏭ Pending batal: {total_cancel} total\n"
                       f"  TP sebelum entry: {pc['tp_before_entry']} | "
                       f"Expired: {pc['expired']} | Ditolak Binance: {pc['binance_reject']}")
    with ban_lock:
        banned_n = len(banned_coins)
    with early_reject_lock:
        reject_rem = early_reject_remaining
    lc=_low_conf_summary(); top_lc=", ".join(f"{x['symbol']}({x['count']}x)" for x in lc[:5]) if lc else "—"
    mode_label = "🔴 REAL" if REAL_TRADE_ENABLED else "🧪 SIMULASI"
    avg_line = f"{avg_all:.1f}%" if avg_all is not None else "—"
    tp_line = f"{avg_tp:.1f}%" if avg_tp is not None else "—"
    trail_line = f"{avg_trail:.1f}%" if avg_trail is not None else "—"
    sl_line = f"{avg_sl:.1f}%" if avg_sl is not None else "—"
    return (
        f"📊 <b>Statistik</b> — {t} trade | TP {tp} SL {sl} Trail {trail}\n"
        f"Mode: <b>{mode_label}</b>\n"
        f"Win Rate: <b>{wr:.1f}%</b> (TP+Trail vs SL)\n"
        f"\nModal anchor: <b>${base:.4f}</b> → Saldo statistik: <b>${bal:.4f}</b> "
        f"({sgn}{pnl_pct:.2f}%)\n"
        f"\nConfidence rata-rata closed: <b>{avg_line}</b>\n"
        f"🎯 TP: <b>{tp_line}</b> | 🔒 Trail: <b>{trail_line}</b> | 🛑 SL: <b>{sl_line}</b>\n"
        f"\n5 terakhir:\n{hist_str}\n\n"
        f"🚫 Banned: {banned_n} | 🧠 Low-conf teratas: {top_lc} | 🛡️ Early reject tersisa: {reject_rem}"
        f"{cancel_line}"
    )

def fmt_backtest():
    """20 trade terakhir dengan confidence per trade."""
    with stat_lock:
        hist = list(stats["pnl_history"])
    if not hist:
        return "📋 <b>Backtest</b>\nBelum ada trade."
    lines = []
    for h in reversed(hist):
        em  = "🟢" if float(h.get("pnl_usd", 0) or 0) > 0 else "🔴" if float(h.get("pnl_usd", 0) or 0) < 0 else "⚪"
        dec = h.get("decision") or "?"
        sym = h.get("symbol") or "?"
        et  = h.get("entry_time"); xt = h.get("exit_time")
        t_in  = datetime.fromtimestamp(et, WIB).strftime("%d/%m/%Y %H:%M") if et else "?"
        t_out = datetime.fromtimestamp(xt, WIB).strftime("%d/%m/%Y %H:%M") if xt else "?"
        sgn = "+" if h["pnl_usd"] >= 0 else ""
        entry_v, tp_v, sl_v = h.get("entry"), h.get("tp"), h.get("sl")
        sl_display = h.get("exit_price") if h.get("result") == "trail" else sl_v
        levels = (f"Entry: <code>{entry_v:.6g}</code> | TP: <code>{tp_v:.6g}</code> | SL: <code>{sl_display:.6g}</code>\n"
                  if entry_v is not None and tp_v is not None and sl_display is not None else "")
        try:
            conf_txt = f"{float(h.get('confidence')):.0f}%"
        except (TypeError, ValueError):
            conf_txt = "—"
        lines.append(
            f"{em} <b>{sym}</b> {dec} | {h['result'].upper()} {sgn}{h['pct']:.2f}% | Confidence: <b>{conf_txt}</b>\n"
            f"{levels}{t_in}→{t_out}"
        )
    return f"📋 <b>Backtest ({len(hist)} trade terakhir)</b>\n\n" + "\n\n".join(lines)

# ============================================================
# ANALYZE — CLOSED TRADE LEDGER (DUA FILE: MD + CSV)
# ============================================================
def _trade_analysis_rows(hist):
    rows = []
    for t in hist:
        rows.append({
            "trade_id": t.get("trade_id", ""),
            "run_id": t.get("run_id", ""),
            "symbol": t.get("symbol", ""),
            "result": t.get("result", ""),
            "close_reason": t.get("close_reason", ""),
            "decision": t.get("decision", ""),
            "entry": t.get("entry", ""),
            "sl": t.get("sl", ""),
            "tp": t.get("tp", ""),
            "exit_price": t.get("exit_price", ""),
            "rr": t.get("rr", ""),
            "final_r": t.get("final_r", ""),
            "confidence": t.get("confidence", ""),
            "entry_label": t.get("entry_label", ""),
            "rsi": t.get("rsi", ""),
            "struct_h1": t.get("struct_h1", ""),
            "d1_bias": t.get("d1_bias", ""),
            "pct": t.get("pct", ""),
            "pnl_usd": t.get("pnl_usd", ""),
            "balance_after": t.get("balance_after", ""),
            "mfe_pct": t.get("mfe_pct", ""),
            "mae_pct": t.get("mae_pct", ""),
            "mfe_r": t.get("mfe_r", ""),
            "mae_r": t.get("mae_r", ""),
            "time_in_trade_sec": t.get("time_in_trade_sec", ""),
            "time_to_1r_sec": t.get("time_to_1r_sec", ""),
            "time_to_2r_sec": t.get("time_to_2r_sec", ""),
            "execution_mode": t.get("execution_mode", ""), "balance_anchor": t.get("balance_anchor", ""), "trade_uid": t.get("trade_uid", ""),
            "trail_update_count": t.get("trail_update_count",0), "trail_applied_count": t.get("trail_applied_count",0), "trail_failed_count": t.get("trail_failed_count",0), "trail_queued_count": t.get("trail_queued_count",0),
            "first_trail_r": t.get("first_trail_r", ""), "last_trail_r": t.get("last_trail_r", ""), "max_protected_r": t.get("max_protected_r", ""),
            "ml_model_version": t.get("ml_model_version", "static"),
            "learning_features": json.dumps(t.get("learning_features"), ensure_ascii=False, sort_keys=True) if t.get("learning_features") else "",
            "entry_time": t.get("entry_time", ""), "exit_time": t.get("exit_time", ""),
        })
    return rows


def _analyze_trade_history():
    """Build one consistent, immutable-enough snapshot for the whole analysis pipeline."""
    with trade_history_lock:
        hist = [dict(x) for x in trade_history]
    if not hist:
        return [], {"trades": 0, "run_id": research_run_id}, []

    winners = [t for t in hist if t.get("result") in ("tp", "trail")]
    losers = [t for t in hist if t.get("result") == "sl"]
    gross_profit = sum(max(float(t.get("pnl_usd", 0.0)), 0.0) for t in hist)
    gross_loss = abs(sum(min(float(t.get("pnl_usd", 0.0)), 0.0) for t in hist))
    total_pnl = sum(float(t.get("pnl_usd", 0.0)) for t in hist)
    avg_pct = sum(float(t.get("pct", 0.0)) for t in hist) / len(hist)
    anchors=[]
    for t in hist:
        try:
            if t.get("balance_anchor") is not None: anchors.append(float(t.get("balance_anchor")))
        except (TypeError,ValueError): pass
    run_anchor = anchors[0] if anchors else STARTING_BALANCE
    equity = [run_anchor] + [float(t.get("balance_after", run_anchor)) for t in hist]
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    # Additional diagnostics derived from the closed-trade ledger.
    for t in hist:
        try:
            entry=float(t.get("entry")); exit_price=float(t.get("exit_price")); sl=float(t.get("sl")); side=str(t.get("decision") or "BUY").upper(); risk=abs(entry-sl)
            t["final_r"] = (((exit_price-entry)/risk) if side=="BUY" else ((entry-exit_price)/risk)) if risk else 0.0
        except Exception:
            t["final_r"] = None

    # Breakdown yang berguna untuk diagnosis strategy tanpa market rescan.
    by_result = {}
    by_symbol = {}
    by_entry = {}
    for t in hist:
        r = str(t.get("result") or "unknown")
        s = str(t.get("symbol") or "?")
        el = str(t.get("entry_label") or "?")
        for bucket, key in ((by_result, r), (by_symbol, s), (by_entry, el)):
            b = bucket.setdefault(key, {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
            b["trades"] += 1
            b["pnl"] += float(t.get("pnl_usd", 0.0))
            if float(t.get("pnl_usd", 0.0)) >= 0:
                b["wins"] += 1
            else:
                b["losses"] += 1

    summary = {
        "trades": len(hist),
        "run_id": hist[-1].get("run_id", research_run_id),
        "balance": float(hist[-1].get("balance_after", run_anchor)),
        "balance_anchor": run_anchor, "net": total_pnl,
        "win_rate": len(winners) / len(hist) * 100.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_dd": max_dd,
        "expectancy": total_pnl / len(hist),
        "avg_pct": avg_pct,
        "tp": sum(1 for t in hist if t.get("result") == "tp"),
        "trail": sum(1 for t in hist if t.get("result") == "trail"),
        "sl": sum(1 for t in hist if t.get("result") == "sl"),
        "by_result": by_result,
        "by_symbol": by_symbol,
        "by_entry": by_entry,
    }
    return _trade_analysis_rows(hist), summary, hist


def _analyze_snapshot():
    """Compatibility wrapper: /analyze tidak scan market; membaca satu ledger snapshot."""
    rows, summary, hist = _analyze_trade_history()
    return rows, summary, hist


def _analyze_runtime_stats():
    rows, summary, _hist = _analyze_snapshot()
    return {
        "trades": summary.get("trades", 0),
        "balance": summary.get("balance", STARTING_BALANCE),
        "net": summary.get("net", 0.0),
        "win_rate": summary.get("win_rate", 0.0),
        "profit_factor": summary.get("profit_factor", 0.0),
        "max_dd": summary.get("max_dd", 0.0),
        "expectancy": summary.get("expectancy", 0.0),
    }


def _write_analyze_csv(rows):
    path = "/tmp/analyze_data.csv"
    cols = [
        "trade_id", "run_id", "symbol", "result", "close_reason", "decision", "entry", "sl", "tp", "exit_price",
        "rr", "final_r", "confidence", "entry_label", "rsi", "struct_h1", "d1_bias", "pct", "pnl_usd",
        "balance_after", "balance_anchor", "execution_mode", "mfe_pct", "mae_pct", "mfe_r", "mae_r", "time_in_trade_sec",
        "time_to_1r_sec", "time_to_2r_sec", "trail_update_count", "trail_applied_count", "trail_failed_count", "trail_queued_count",
        "first_trail_r", "last_trail_r", "max_protected_r", "trade_uid", "ml_model_version", "learning_features", "entry_time", "exit_time",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def _trail_events_snapshot(run_id=None):
    with trail_events_lock: rows=[dict(x) for x in trail_events]
    return [x for x in rows if run_id is None or x.get("run_id")==run_id]

def _scan_quality_snapshot(run_id=None):
    with scan_quality_lock: rows=[dict(x) for x in scan_quality_history]
    return [x for x in rows if run_id is None or x.get("run_id")==run_id]

def _low_conf_snapshot():
    with low_conf_history_lock: return [dict(x) for x in low_conf_history]


def _write_research_support_files(summary):
    """Bundle all research support datasets into one lossless JSON artifact."""
    run_id = summary.get("run_id", research_run_id)
    events = _trail_events_snapshot(run_id)
    scans = _scan_quality_snapshot(run_id)
    lows = _low_conf_snapshot()
    market_rows = _market_context_snapshot(run_id)

    trail_cols = ["event_id","trade_uid","run_id","event_time","symbol","decision","entry","initial_sl","old_sl","new_sl","tp","current_price","current_r","mfe_r","mae_r","giveback_r","giveback_ratio","protected_r","atr","sl_distance_atr","weakness_score","state","trade_phase","trail_source","reasons","relative_volume","candidate_type","status","error_code","error_message","time_since_entry_sec","time_since_previous_trail_sec","distance_to_tp_r"]
    scan_cols = ["scan_time","run_id","scan_counter","symbols_requested","symbols_analyzed","failed_symbols","avg_confidence","min_confidence","max_confidence","low_confidence_count","below_threshold_count","qualified_count","early_rejected_count","cache_entries","cache_fresh","market_regime","bullish_breadth_pct","bearish_breadth_pct","neutral_breadth_pct","breadth_score","median_price_1h_pct","median_price_4h_pct","median_efficiency_4h","median_range_expansion_ratio","avg_relative_volume","btc_price_1h_pct","btc_price_4h_pct"]
    low_cols = ["event_time","run_id","scan_counter","symbol","confidence","confidence_min","cutoff","decision","entry_label"]
    market_cols = ["scan_time","run_id","scan_counter","symbol","decision","confidence","entry_label","struct_h1","d1_bias","price_1h_pct","price_4h_pct","efficiency_4h","atr_pct","relative_volume","range_expansion_ratio","volatility_ratio","chart_regime","directional_bias","relative_strength_1h_pct","relative_strength_4h_pct"]

    groups = {}
    for e in events:
        uid = e.get("trade_uid") or f"{e.get('symbol','?')}|{e.get('entry','')}"
        g = groups.setdefault(uid, {
            "trade_uid": uid, "symbol": e.get("symbol"), "trail_updates": 0,
            "trail_applied": 0, "trail_failed": 0, "trail_queued": 0,
            "first_trail_r": None, "max_protected_r": None,
            "max_mfe_at_trail": None, "max_giveback_ratio": None,
            "final_new_sl": None, "_dist": []
        })
        g["trail_updates"] += 1
        status = str(e.get("status") or "")
        g["trail_applied"] += int(status == "APPLIED")
        g["trail_failed"] += int(status == "FAILED")
        g["trail_queued"] += int(status == "QUEUED")
        if g["first_trail_r"] is None and e.get("current_r") not in (None, ""):
            g["first_trail_r"] = float(e["current_r"])
        if e.get("protected_r") not in (None, ""):
            v = float(e["protected_r"])
            g["max_protected_r"] = v if g["max_protected_r"] is None else max(g["max_protected_r"], v)
        if e.get("mfe_r") not in (None, ""):
            v = float(e["mfe_r"])
            g["max_mfe_at_trail"] = v if g["max_mfe_at_trail"] is None else max(g["max_mfe_at_trail"], v)
        if e.get("giveback_ratio") not in (None, ""):
            v = float(e["giveback_ratio"])
            g["max_giveback_ratio"] = v if g["max_giveback_ratio"] is None else max(g["max_giveback_ratio"], v)
        if e.get("sl_distance_atr") not in (None, ""):
            g["_dist"].append(float(e["sl_distance_atr"]))
        g["final_new_sl"] = e.get("new_sl")

    trail_summary = []
    for g in groups.values():
        ds = g.pop("_dist", [])
        g["avg_sl_distance_atr"] = (sum(ds) / len(ds)) if ds else None
        trail_summary.append(g)

    payload = {
        "schema_version": "analyze_research_bundle_v1",
        "generated_at": datetime.now(WIB).isoformat(),
        "run_id": run_id,
        "description": "Lossless bundle of the five research-support datasets formerly emitted as separate CSV files.",
        "trail_events": {"columns": trail_cols, "rows": events},
        "trail_summary": {"columns": ["trade_uid","symbol","trail_updates","trail_applied","trail_failed","trail_queued","first_trail_r","max_protected_r","max_mfe_at_trail","max_giveback_ratio","avg_sl_distance_atr","final_new_sl"], "rows": trail_summary},
        "scan_quality": {"columns": scan_cols, "rows": scans},
        "low_confidence_bans": {"columns": low_cols, "rows": lows},
        "market_context": {"columns": market_cols, "rows": market_rows},
        "machine_learning": _full_status_text(),
        "machine_learning_state": _ml_current_champion(),
        "machine_learning_experience_count": len(ML_EXPERIENCE),
    }

    path = "/tmp/analyze_research_bundle.json"
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    return path

def _write_analyze_report(rows, summary, hist):
    """Write the report from the SAME snapshot used to calculate summary."""
    path = "/tmp/analyze_report.md"
    now = datetime.now(WIB).strftime("%Y-%m-%d %H:%M:%S WIB")
    hist = [dict(t) for t in (hist or [])]
    if not rows or not hist:
        Path(path).write_text(
            "# SMCAutoTrade — Trade Analysis\n\n"
            f"**Waktu:** {now}\n\n"
            "Belum ada closed trade pada research run aktif.\n",
            encoding="utf-8",
        )
        return path

    def _pf(v):
        return "∞" if v == float("inf") else f"{v:.3f}"

    lines = [
        "# SMCAutoTrade — Trade Analysis",
        "",
        f"**Waktu analysis:** {now}",
        f"**Research run:** `{summary.get('run_id', research_run_id)}`",
        f"**Snapshot:** {len(hist)} closed trade",
        "**Sumber:** full closed-trade ledger dari `/stats`; tidak melakukan market scan.",
        "",
        "## Ringkasan",
        "",
        "| Metrik | Nilai |",
        "|---|---:|",
        f"| Closed trades | {summary['trades']} |",
        f"| TP | {summary['tp']} |",
        f"| Trail | {summary['trail']} |",
        f"| SL | {summary['sl']} |",
        f"| Win rate | {summary['win_rate']:.2f}% |",
        f"| Profit factor | {_pf(summary['profit_factor'])} |",
        f"| Expectancy/trade | ${summary['expectancy']:.5f} |",
        f"| Avg PnL/trade | {summary['avg_pct']:.3f}% |",
        f"| Net PnL | ${summary['net']:.4f} |",
        f"| Balance | ${summary['balance']:.4f} |",
        f"| Max drawdown | ${summary['max_dd']:.4f} |",
        "",
        "## Breakdown Result",
        "",
        "| Result | Trades | PnL |",
        "|---|---:|---:|",
    ]
    for k in ("tp", "trail", "sl"):
        b = summary["by_result"].get(k, {"trades": 0, "pnl": 0.0})
        lines.append(f"| {k.upper()} | {b['trades']} | ${b['pnl']:.4f} |")

    lines += ["", "## Breakdown Entry Type", "", "| Entry | Trades | Wins | Losses | PnL |", "|---|---:|---:|---:|---:|"]
    for key, b in sorted(summary["by_entry"].items(), key=lambda kv: kv[1]["pnl"], reverse=True):
        lines.append(f"| {key} | {b['trades']} | {b['wins']} | {b['losses']} | ${b['pnl']:.4f} |")

    lines += ["", "## Breakdown Symbol", "", "| Symbol | Trades | Wins | Losses | PnL |", "|---|---:|---:|---:|---:|"]
    for key, b in sorted(summary["by_symbol"].items(), key=lambda kv: kv[1]["pnl"], reverse=True):
        lines.append(f"| {key} | {b['trades']} | {b['wins']} | {b['losses']} | ${b['pnl']:.4f} |")

    buckets=[(50,59),(60,69),(70,79),(80,89),(90,100)]
    lines += ["", "## Breakdown Confidence", "", "| Bucket | Trades | Wins | Losses | Win Rate | Avg PnL | Avg R |", "|---|---:|---:|---:|---:|---:|---:|"]
    for lo,hi in buckets:
        bt=[t for t in hist if lo <= float(t.get("confidence",0) or 0) <= hi]
        if not bt: continue
        wins=sum(1 for t in bt if t.get("result") in ("tp","trail")); losses=len(bt)-wins
        lines.append(f"| {lo}-{hi} | {len(bt)} | {wins} | {losses} | {wins/len(bt)*100:.1f}% | {sum(float(t.get('pct',0) or 0) for t in bt)/len(bt):.3f}% | {sum(float(t.get('final_r',0) or 0) for t in bt)/len(bt):.2f}R |")
    lines += ["", "## Breakdown Entry Type", "", "| Entry | Trades | Wins | Losses | Win Rate | Avg PnL | Avg R |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key,b in sorted(summary["by_entry"].items(), key=lambda kv: kv[1]["pnl"], reverse=True):
        bt=[t for t in hist if str(t.get("entry_label") or "?")==str(key)]
        avgr=sum(float(t.get("final_r",0) or 0) for t in bt)/len(bt) if bt else 0
        lines.append(f"| {key} | {b['trades']} | {b['wins']} | {b['losses']} | {(b['wins']/b['trades']*100 if b['trades'] else 0):.1f}% | {(b['pnl']/b['trades'] if b['trades'] else 0):.3f} | {avgr:.2f}R |")
    path_rows=[t for t in hist if t.get("mfe_r") is not None or t.get("mae_r") is not None]
    if path_rows:
        mfe=sum(float(t.get("mfe_r",0) or 0) for t in path_rows)/len(path_rows); mae=sum(float(t.get("mae_r",0) or 0) for t in path_rows)/len(path_rows)
        ds=[float(t.get("time_in_trade_sec")) for t in path_rows if t.get("time_in_trade_sec") not in (None, "")]
        lines += ["", "## Path Analytics", "", f"- Trades tracked: **{len(path_rows)}**", f"- Avg MFE: **{mfe:.2f}R**", f"- Avg MAE: **{mae:.2f}R**"]
        if ds: lines.append(f"- Avg time in trade: **{sum(ds)/len(ds)/60:.1f} minutes**")

    # Guard against future regressions where summary and report receive different snapshots.
    if int(summary.get("trades", 0) or 0) != len(hist):
        raise RuntimeError(
            f"analysis snapshot mismatch: summary={summary.get('trades')} hist={len(hist)}"
        )

    trail_rows=_trail_events_snapshot(summary.get("run_id",research_run_id)); low_summary=_low_conf_summary(); scan_rows=_scan_quality_snapshot(summary.get("run_id",research_run_id))
    if trail_rows:
        protected=[float(x["protected_r"]) for x in trail_rows if x.get("protected_r") not in (None,"")]; give=[float(x["giveback_ratio"]) for x in trail_rows if x.get("giveback_ratio") not in (None,"")]
        lines += ["","## Trail Effectiveness","",f"- Trail events: **{len(trail_rows)}**",f"- Applied: **{sum(x.get('status')=='APPLIED' for x in trail_rows)}** | Queued: **{sum(x.get('status')=='QUEUED' for x in trail_rows)}** | Failed: **{sum(x.get('status')=='FAILED' for x in trail_rows)}**",f"- Avg protected R: **{(sum(protected)/len(protected) if protected else 0):.2f}R**",f"- Avg giveback ratio: **{(sum(give)/len(give)*100 if give else 0):.1f}%**"]
    if low_summary:
        lines += ["","## Low Confidence Ban Frequency","","| Symbol | Ban Count | Avg Conf | Min | Max |","|---|---:|---:|---:|---:|"]
        for g in low_summary[:20]: lines.append(f"| {g['symbol']} | {g['count']} | {g['avg']:.1f}% | {g['min']:.1f}% | {g['max']:.1f}% |")
    if scan_rows:
        av=[float(x['avg_confidence']) for x in scan_rows if x.get('avg_confidence') is not None]
        lines += ["","## Scan Quality","",f"- Scan cycles: **{len(scan_rows)}**",f"- Avg confidence by scan: **{(sum(av)/len(av) if av else 0):.2f}%**",f"- Total qualified signals: **{sum(int(x.get('qualified_count',0) or 0) for x in scan_rows)}**"]
        last_scan=scan_rows[-1]
        lines += [f"- Latest market regime: **{last_scan.get('market_regime','unknown')}**",f"- Latest breadth: **BUY {float(last_scan.get('bullish_breadth_pct',0) or 0):.1f}% / SELL {float(last_scan.get('bearish_breadth_pct',0) or 0):.1f}%**",f"- Latest median 4h efficiency: **{float(last_scan.get('median_efficiency_4h',0) or 0):.2f}**"]
    market_rows=_market_context_snapshot(summary.get("run_id",research_run_id))
    if market_rows:
        rs1=[float(x['relative_strength_1h_pct']) for x in market_rows if x.get('relative_strength_1h_pct') is not None and x.get('symbol')!='BTCUSDT']
        rv=[float(x['relative_volume']) for x in market_rows if x.get('relative_volume') is not None]
        lines += ["","## Market Context","",f"- Symbol-context observations: **{len(market_rows)}**",f"- Avg relative strength vs BTC (1h): **{(sum(rs1)/len(rs1) if rs1 else 0):+.2f}%**",f"- Avg relative volume: **{(sum(rv)/len(rv) if rv else 0):.2f}x**"]
    lines += [
        "", "## Catatan", "",
        "- `/analyze` hanya menganalisis trade yang benar-benar closed dan tercatat sejak `/resetstats` terakhir.",
        "- History penuh disimpan dalam memory `trade_history`; `/backtest` tetap menampilkan 20 trade terakhir untuk kompatibilitas UI.",
        "- Jalankan `/resetstats` untuk memulai research run baru dan mengosongkan ledger aktif.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

# ============================================================
# END ANALYZE
# ============================================================

def fmt_signal_msg(sig):
    d=sig.get("decision","?"); em="🟢" if d=="BUY" else "🔴" if d=="SELL" else "⚪"
    return (f"📡 <b>{sig.get('symbol','?')}</b> | {em} <b>{d}</b> | Confidence: {sig.get('confidence','—')}%\n"
            f"Entry: <code>{sig.get('entry',0):.8g}</code> | TP: <code>{sig.get('tp',0):.8g}</code> | SL: <code>{sig.get('sl',0):.8g}</code>")



# ═════════════════════════════════════════════
# MULTI-POSITION BROADCASTER
# ═════════════════════════════════════════════
# MAX_POSITIONS dikontrol lewat /max — lihat konstanta di bagian atas file
MONITOR_INTERVAL = 15 * 60  # cek posisi tiap 15 menit (detik)

positions_lock = threading.Lock()
positions: dict = {}   # {sym: {signal, entry, tp, sl, entry_time, thread}}

def close_position(sym, result, close_price=None):
    """Finalize a position that is already confirmed flat on Binance.

    This function is intentionally exception-safe: exchange-side closure must never be
    lost merely because statistics/Telegram formatting encounters an unexpected error.
    """
    global active_trade
    with positions_lock:
        pos = positions.get(sym)
    if pos is None:
        return False

    sig = pos.get("signal") or {}
    entry = pos.get("entry")
    sl_p = sig.get("sl")
    tp_p = sig.get("tp")
    cid = pos.get("chat_id") or active_chat_id

    # Final path metric update before the local state is removed.
    try:
        if close_price is not None:
            _update_trade_path_metrics(pos, close_price)
    except Exception:
        pass

    classified = _classify_close_result(
        result, entry=entry, close_price=close_price, decision=sig.get("decision")
    )

    # Remove local position exactly once. Exchange has already been confirmed flat by callers.
    with positions_lock:
        popped = positions.pop(sym, None)
        if popped is None:
            return False
        if not positions:
            active_trade = None

    stats_error = None
    try:
        update_stats(
            classified, entry=entry, sl_p=sl_p, tp_p=tp_p, close_price=close_price,
            sym=sym, decision=sig.get("decision"), entry_time=pos.get("entry_time"),
            close_reason=result,
            confidence=sig.get("confidence"), entry_label=sig.get("entry_label"),
            rr=sig.get("rr"), rsi=sig.get("rsi"), struct_h1=sig.get("struct_h1"),
            d1_bias=sig.get("d1_bias"),
            mfe_pct=pos.get("mfe_pct"), mae_pct=pos.get("mae_pct"),
            mfe_r=pos.get("mfe_r"), mae_r=pos.get("mae_r"),
            time_in_trade_sec=pos.get("time_in_trade_sec"),
            time_to_1r_sec=pos.get("time_to_1r_sec"), time_to_2r_sec=pos.get("time_to_2r_sec"),
            execution_mode=pos.get("execution_mode"), balance_anchor=(real_balance_snapshot if str(pos.get("execution_mode") or "").upper()=="REAL" else STARTING_BALANCE),
            trade_uid=pos.get("trade_uid"), trail_update_count=pos.get("trail_update_count",0), trail_applied_count=pos.get("trail_applied_count",0),
            trail_failed_count=pos.get("trail_failed_count",0), trail_queued_count=pos.get("trail_queued_count",0), first_trail_r=pos.get("first_trail_r"),
            last_trail_r=pos.get("last_trail_r"), max_protected_r=(pos.get("max_protected_r") if pos.get("max_protected_r",-999)>-998 else None),
            learning_features=(pos.get("signal") or {}).get("learning_features"),
            ml_model_version=(pos.get("signal") or {}).get("learning_model_version", "static")
        )
    except Exception as e:
        stats_error = e
        log.exception(f"[close_position] stats gagal {sym}: {e}")

    try:
        _ban_coin(sym, f"trade closed ({classified}; reason={result})", duration=BAN_DURATION_TRADE_CLOSED, kind="closed")
    except Exception as e:
        log.warning(f"[close_position] gagal ban {sym}: {e}")

    try:
        with stat_lock:
            last = stats["pnl_history"][-1] if stats["pnl_history"] else None
    except Exception:
        last = None

    emoji = {"tp":"🎯","sl":"🛑","trail":"🔒"}.get(classified, "🛑")
    label = {"tp":"TAKE PROFIT","sl":"STOP LOSS","trail":"TRAILING STOP"}.get(classified, "STOP LOSS")
    detail = ""
    if last and last.get("symbol") == sym:
        sgn = "+" if last.get("pct", 0) >= 0 else ""
        detail = (
            f"Entry: <code>{last.get('entry', entry):.6g}</code> → Exit: <code>{last.get('exit_price', close_price):.6g}</code>\n"
            f"Hasil: <b>{sgn}{last.get('pct', 0):.2f}%</b> (${sgn}{last.get('pnl_usd', 0):.4f})\n\n"
        )
    if stats_error:
        detail += "⚠️ Statistik lokal mengalami error dan akan dicoba dipulihkan oleh audit/recovery.\n\n"
    try:
        tg_send(cid, f"{emoji} <b>{label}</b> — {sym}\n\n{detail}" + fmt_stats())
    except Exception as e:
        log.error(f"[close_position] notifikasi gagal {sym}: {e}")
    return True


def _finalize_external_close(sym, pos, reason_hint="unknown", exit_price=None):
    """Finalize an exchange-side close detected by positionAmt==0.

    Binance positionRisk returning a symbol with positionAmt=0 is treated as a real close,
    not as a still-open position. Cleanup is best-effort but tracked separately so a
    transient API failure cannot erase the closed-trade record.
    """
    sig = pos.get("signal") or {}
    price = exit_price
    if price is None:
        try:
            price = get_price(sym)
        except Exception:
            price = None
    if price is None:
        price = pos.get("current_price") or pos.get("entry")

    # Keep final MFE/MAE snapshot without another Binance request.
    try:
        if price is not None:
            _update_trade_path_metrics(pos, price)
    except Exception:
        pass

    reason = reason_hint if reason_hint in ("tp", "sl") else "unknown"
    if reason == "unknown":
        # A trailing SL that was moved to/above breakeven and then triggered is a Trail
        # only if the realized outcome is positive; the common classifier below handles it.
        reason = "trail"
    elif reason == "sl":
        # Preserve explicit SL when Binance's algo status confirms a stop. The final
        # classifier will still convert it only when the caller marks it as trail/timeout.
        pass

    # Clean orphan protection. If Binance is temporarily unavailable, retain a pending
    # cleanup marker but still finalize the already-flat trade locally.
    try:
        _cancel_all_algo_orders_verified(sym)
    except Exception as e:
        _queue_pending_cleanup(sym, "post-external-close cleanup", e)
        log.warning(f"[external-close] {sym} cleanup tertunda: {e}")

    # If the exchange explicitly told us TP/SL, preserve that reason. Otherwise classify
    # from the realized exit outcome and trailing context.
    if reason_hint == "tp":
        final_result = "tp"
    elif reason_hint == "sl":
        # An SL trigger after a trailing move is a Trail only if the realized PnL is positive.
        trail_candidate = bool(pos.get("current_sl") is not None and pos.get("entry") is not None)
        final_result = "trail" if trail_candidate and _classify_close_result("trail", pos.get("entry"), price, sig.get("decision")) == "trail" else "sl"
    else:
        final_result = "trail" if _classify_close_result("trail", pos.get("entry"), price, sig.get("decision")) == "trail" else "sl"

    closed = close_position(sym, final_result, close_price=price)
    return closed


def check_tp_sl_order(sym, tp_p, sl_p, is_buy, lookback_min=15):
    """
    Ambil candle M1 dalam N menit terakhir, periksa urutan:
    mana yang kena duluan — TP atau SL?

    Return: "tp", "sl", atau None (tidak ada yang tersentuh)
    """
    try:
        df = get_klines(sym, "1m", lookback_min + 2)
        if df is None or df.empty: return None

        # Ambil hanya candle dalam lookback_min menit terakhir
        df = df.tail(lookback_min)

        for _, row in df.iterrows():
            high = row["high"]
            low  = row["low"]
            if is_buy:
                # Untuk BUY: TP di atas, SL di bawah
                # Kalau high >= TP dan low <= SL di candle yang sama → cek open lebih dekat ke mana
                if high >= tp_p and low <= sl_p:
                    # Harga open candle ini lebih dekat ke TP atau SL?
                    dist_tp = abs(row["open"] - tp_p)
                    dist_sl = abs(row["open"] - sl_p)
                    return "tp" if dist_tp < dist_sl else "sl"
                elif high >= tp_p:
                    return "tp"
                elif low <= sl_p:
                    return "sl"
            else:
                # Untuk SELL: TP di bawah, SL di atas
                if low <= tp_p and high >= sl_p:
                    dist_tp = abs(row["open"] - tp_p)
                    dist_sl = abs(row["open"] - sl_p)
                    return "tp" if dist_tp < dist_sl else "sl"
                elif low <= tp_p:
                    return "tp"
                elif high >= sl_p:
                    return "sl"
    except Exception as e:
        log.debug(f"[check_tp_sl_order] {sym}: {e}")
    return None




# ============================================================
# STRATEGY DISPATCH — ENGINE TIDAK MEMILIKI OTAK TRADING
# ============================================================

def _strategy_position_update(sym,pos):
    if _binance_is_scan_paused():
        return None
    manager=globals().get("manage_position")
    if not callable(manager): return None
    try:
        m15=get_klines(sym,"15m",250); h1=get_klines(sym,"1h",250)
        try: d1=get_klines(sym,"1d",100)
        except Exception: d1=None
        return manager(state=dict(pos),df_m15=m15,df_h1=h1,df_d1=d1,symbol=sym)
    except Exception as e:
        log.warning(f"[strategy/manage] {sym}: {e}"); return None

def _notify_trail_update(chat_id, sym, pos, update, old_sl, new_sl, status="APPLIED", error=None):
    """Kirim notifikasi Telegram hanya saat trailing/protection benar-benar berubah."""
    if not chat_id or new_sl is None or old_sl is None:
        return
    try:
        _record_trail_event(sym,pos,update if isinstance(update,dict) else {},old_sl,new_sl,status=status,error=error)
        sig = pos.get("signal", {})
        entry = float(pos.get("entry") or sig.get("entry") or 0.0)
        decision = str(sig.get("decision") or "BUY").upper()
        tp = sig.get("tp")
        atr = sig.get("atr") or 0.0
        initial_sl = float(pos.get("initial_sl") or sig.get("initial_sl") or old_sl)
        risk = abs(entry - initial_sl) if entry and initial_sl else 0.0
        current_price = float(pos.get("current_price") or pos.get("price") or entry)
        profit_r = ((current_price - entry) if decision == "BUY" else (entry - current_price)) / risk if risk > 0 else 0.0
        reasons = update.get("reason") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        state = update.get("state") or "TRAIL"
        source = update.get("trail_source") or update.get("source") or "adaptive"
        rv = update.get("relative_volume")
        weakness = update.get("weakness_score")
        arrow = "↑" if new_sl > old_sl else "↓"
        lines = [
            f"🔒 <b>TRAILING UPDATE — {sym}</b>",
            "",
            f"Arah: <b>{decision}</b> | State: <b>{state}</b>",
            f"Entry: <code>{entry:.8g}</code>",
            f"Harga: <code>{current_price:.8g}</code>",
            f"SL: <code>{old_sl:.8g}</code> → <code>{new_sl:.8g}</code> {arrow}",
            f"Profit: <b>{profit_r:+.2f}R</b>",
        ]
        if tp is not None:
            lines.append(f"TP saat ini: <code>{float(tp):.8g}</code>")
        if atr:
            lines.append(f"ATR M15: <code>{float(atr):.8g}</code>")
        if rv is not None:
            lines.append(f"Relative Volume: <b>{float(rv):.2f}x</b>")
        if weakness is not None:
            lines.append(f"Weakness score: <b>{int(weakness)}</b>")
        lines.append(f"Engine: <b>{source}</b>")
        if reasons:
            pretty = "\n".join(f"• {str(r).replace('_', ' ')}" for r in reasons[:6])
            lines.append("\n<b>Alasan:</b>\n" + pretty)
        if status == "QUEUED":
            lines += ["", "⏸️ <b>BINANCE PAUSE</b>", "Trail disimpan sebagai pending dan akan dipasang saat recovery selesai."]
        elif status == "APPLIED":
            lines += ["", "✅ <b>Protection order berhasil diperbarui di Binance.</b>"]
        elif status == "FAILED":
            lines += ["", f"🚨 <b>Update protection gagal:</b> <code>{str(error)[:220]}</code>", "⚠️ SL sebelumnya dipertahankan."]
        tg_send(chat_id, "\n".join(lines))
    except Exception as e:
        log.debug(f"[trail-notify] {sym}: {e}")


def _apply_strategy_update(sym,pos,update):
    if not isinstance(update,dict): return False
    sig=pos["signal"]; changed=False
    if update.get("tp") is not None:
        sig["tp"]=float(update["tp"]); changed=True
    if update.get("sl") is not None:
        new=float(update["sl"]); old=float(pos.get("current_sl",sig["sl"]))
        buy=sig["decision"]=="BUY"
        if (new>old) if buy else (new<old):
            pos["current_sl"]=new; sig["sl"]=new; changed=True
    return changed

def monitor_position(sym,pos):
    """Execution monitor. Tidak menentukan Entry/TP/SL/Trail."""
    next_strategy=0
    while True:
        with positions_lock:
            if sym not in positions:return
            pos=positions[sym]
        if pos.get("timeout_flag"):
            price=get_price(sym) or pos["entry"]; buy=pos["signal"]["decision"]=="BUY"
            close_position(sym,"timeout",close_price=price); return
        if time.time()>=next_strategy:
            upd=_strategy_position_update(sym,pos); next_strategy=time.time()+STRATEGY_MANAGE_INTERVAL
            if isinstance(upd,dict):
                if upd.get("close"):
                    price=upd.get("close_price") or get_price(sym) or pos["entry"]
                    reason=str(upd.get("reason") or "strategy")
                    close_position(sym,"trail" if reason=="trail" else "strategy",close_price=price); return
                old_sl=float(pos.get("current_sl",pos["signal"].get("sl")))
                changed=_apply_strategy_update(sym,pos,upd)
                new_sl=float(pos.get("current_sl",pos["signal"].get("sl")))
                if changed and new_sl != old_sl:
                    pos["current_price"] = get_price(sym) or pos.get("current_price") or pos["entry"]
                    _notify_trail_update(active_chat_id, sym, pos, upd, old_sl, new_sl, status="APPLIED")
        price=get_price(sym)
        if price is None: time.sleep(MONITOR_SLEEP); continue
        _update_trade_path_metrics(pos, price)
        sig=pos["signal"]; buy=sig["decision"]=="BUY"; tp=sig.get("tp"); sl=pos.get("current_sl",sig.get("sl"))
        hit_tp=tp is not None and ((price>=tp) if buy else (price<=tp))
        hit_sl=sl is not None and ((price<=sl) if buy else (price>=sl))
        if hit_tp or hit_sl:
            result="tp" if hit_tp and not hit_sl else "sl"
            if hit_tp and hit_sl: result=check_tp_sl_order(sym,tp,sl,buy,3) or "tp"
            close_position(sym,result,close_price=tp if result=="tp" else sl); return
        time.sleep(MONITOR_SLEEP)

def _open_position(sym,signal,actual_entry,chat_id,mode_label="strategy"):
    buy=signal["decision"]=="BUY"; sl=signal.get("sl"); tp=signal.get("tp")
    if sl is None or tp is None:
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,"strategy tidak mengirim SL/TP"); return
    valid=(sl<actual_entry<tp) if buy else (tp<actual_entry<sl)
    if not valid:
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,"level strategy invalid")
        tg_send(chat_id,f"⚠️ <b>Skip {sym}</b> — geometri level strategy invalid.")
        return
    with positions_lock:
        if sym not in positions:return
        pos=positions[sym]
        now=time.time()
        pos.update({"entry":actual_entry,"entry_time":now,"status":"active","trade_uid":f"{research_run_id}:{sym}:{int(now*1000)}",
                    "timeout_flag":False,"current_sl":sl,"initial_sl":sl,"execution_mode":"SIMULATION",
                    "trail_update_count":0,"trail_applied_count":0,"trail_failed_count":0,"trail_queued_count":0,"first_trail_r":None,"last_trail_r":None,"max_protected_r":-999.0})
    tg_send(chat_id,f"⚡ <b>ENTRY {mode_label.upper()}</b> — {sym}\n"
                    f"Entry: <code>{actual_entry:.8g}</code>\n"
                    f"TP: <code>{tp:.8g}</code> | SL: <code>{sl:.8g}</code>")
    threading.Thread(target=monitor_position,args=(sym,pos),daemon=True).start()


# ============================================================
# REAL TRADE — alur pending order, monitoring posisi, auto-stop
# ============================================================

def _open_pending_real(sym,signal,chat_id):
    if _binance_is_scan_paused():
        log.warning(f"[entry] {sym} ditahan — Binance pause aktif")
        return
    buy=signal["decision"]=="BUY"; entry=signal["entry"]; sl=signal.get("sl"); tp=signal.get("tp")
    if sl is None or tp is None:
        _ban_coin(sym,"strategy tidak mengirim SL/TP"); return
    valid=(sl<entry<tp) if buy else (tp<entry<sl)
    if not valid:
        _ban_coin(sym,"geometri strategy invalid"); tg_send(chat_id,f"⏭ <b>Skip {sym}</b> — geometri strategy invalid."); return
    side="BUY" if buy else "SELL"
    with positions_lock:
        if sym in positions or len(positions)>=MAX_POSITIONS:return
        positions[sym]={"signal":signal,"entry":entry,"chat_id":chat_id,"entry_time":None,"trade_uid":f"{research_run_id}:{sym}:pending:{int(time.time()*1000)}",
                        "timeout_flag":False,"status":"pending","execution_mode":"REAL"}
    try:
        _real_trade_preflight(force=False)
        avail,_=get_real_balance()
        if avail is not None and avail<MARGIN_USD: raise RuntimeError(f"saldo ${avail:.2f} < margin ${MARGIN_USD:.2f}")
        qty,margin,bumped=calc_auto_quantity(sym,entry,MARGIN_USD,LEVERAGE)
        if qty is None: raise RuntimeError("quantity di bawah minimum Binance")
        set_leverage_verified(sym,LEVERAGE); order=place_limit_order(sym,side,qty,entry)
        with positions_lock: positions[sym].update({"order_id":order["orderId"],"quantity":qty,"margin_used":margin})
        tg_send(chat_id,f"🎯 <b>PENDING ORDER REAL</b> — {sym}\n\n{fmt_signal_msg(signal)}")
        threading.Thread(target=_wait_entry_real,args=(sym,signal,chat_id,order["orderId"]),daemon=True).start()
    except BinanceUnknownExecutionError as e:
        # The entry POST may have reached Binance even though its response was lost.
        # Preserve the client order id so reconciliation can resolve the ambiguity.
        with positions_lock:
            if sym in positions:
                positions[sym]["status"]="EMERGENCY"
                positions[sym]["emergency_error"]=str(e)[:300]
                positions[sym]["entry_client_order_id"]=getattr(e, "client_order_id", None)
        tg_send(chat_id,f"🚨 <b>ENTRY STATUS UNKNOWN</b> — {sym}\n<code>{str(e)[:300]}</code>\nOrder tidak diulang secara buta. State dipertahankan untuk rekonsiliasi <code>/ok {sym}</code>.")
    except Exception as e:
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,f"gagal pasang order real ({e})"); tg_send(chat_id,f"⚠️ <b>Skip {sym}</b> — {e}")



def _wait_entry_real(sym,signal,chat_id,order_id):
    deadline=time.time()+8*3600
    while time.time()<deadline:
        with positions_lock:
            if sym not in positions:return
            if positions[sym].get("timeout_flag"):
                try:
                    cancel_order(sym,order_id)
                    time.sleep(0.2)
                    st=get_order_status(sym,order_id)
                    if str(st.get("status","")).upper()=="FILLED":
                        actual=float(st.get("avgPrice") or 0) or signal["entry"]
                        _open_position_real(sym,signal,actual,chat_id,st)
                        return
                    positions.pop(sym,None); return
                except Exception as e:
                    with positions_lock:
                        if sym in positions:
                            positions[sym]["status"]="EMERGENCY"
                            positions[sym]["emergency_error"]=str(e)[:300]
                    tg_send(chat_id, f"🚨 <b>ENTRY CANCEL BELUM TERKONFIRMASI</b> — {sym}\n<code>{str(e)[:300]}</code>\nPosisi tetap dipertahankan sampai <code>/ok {sym}</code>.")
                    return
        try:
            order=get_order_status(sym,order_id)
        except Exception as e:
            log.warning(f"[wait_entry_real] {sym}: {e}"); time.sleep(REAL_TRADE_POLL_SLEEP); continue
        status=str(order.get("status","")).upper()
        if status=="FILLED":
            actual=float(order.get("avgPrice") or 0) or signal["entry"]
            _open_position_real(sym,signal,actual,chat_id,order); return
        if status in ("CANCELED","EXPIRED","REJECTED"):
            with positions_lock: positions.pop(sym,None)
            _ban_coin(sym,f"order {status.lower()}"); _record_pending_cancel("binance_reject"); return
        time.sleep(REAL_TRADE_POLL_SLEEP)

    try:
        cancel_order(sym,order_id)
        st=get_order_status(sym,order_id)
        if str(st.get("status","")).upper()=="FILLED":
            actual=float(st.get("avgPrice") or 0) or signal["entry"]
            _open_position_real(sym,signal,actual,chat_id,st); return
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,"pending expired"); _record_pending_cancel("expired")
    except Exception as e:
        with positions_lock:
            if sym in positions:
                positions[sym]["status"]="EMERGENCY"
                positions[sym]["emergency_error"]=str(e)[:300]
        tg_send(chat_id, f"🚨 <b>PENDING ENTRY BELUM TERKONFIRMASI</b> — {sym}\n<code>{str(e)[:300]}</code>\nState tetap dipertahankan untuk <code>/ok {sym}</code>.")


def _emergency_close(sym, is_buy, qty, chat_id, reason):
    """Emergency flatten. Exchange confirmation is required before local close."""
    try:
        closed, exit_price = _verified_market_close(sym, is_buy, reason, chat_id=chat_id, max_retries=1)
        if not closed:
            raise RuntimeError("posisi belum terkonfirmasi flat")
        try:
            _cleanup_algo_orders_verified(sym)
        except Exception as ce:
            _queue_pending_cleanup(sym, "post-auto-out cleanup", ce)
            raise RuntimeError(f"posisi sudah flat tetapi cleanup protection belum terverifikasi: {ce}")
        with positions_lock:
            local_pos = positions.get(sym)
        if local_pos is not None:
            close_position(sym, "trail" if _classify_close_result("trail", local_pos.get("entry"), exit_price or get_price(sym), local_pos["signal"].get("decision")) == "trail" else "sl", close_price=exit_price or get_price(sym) or local_pos.get("entry"))
        tg_send(chat_id, f"✅ <b>AUTO-OUT</b> — {sym}\nPosisi Binance terkonfirmasi tertutup dan protection dibersihkan.")
        return True
    except Exception as e:
        with positions_lock:
            if sym in positions:
                positions[sym]["status"] = "EMERGENCY"
                positions[sym]["emergency_reason"] = reason
                positions[sym]["emergency_error"] = str(e)[:300]
        _queue_pending_cleanup(sym, "auto-out cleanup", e)
        tg_send(chat_id, f"🚨 <b>GAGAL AUTO-OUT</b> — {sym}: {e}\n⚠️ Posisi TETAP dicatat di /trade. Jalankan <code>/ok {sym}</code> untuk rekonsiliasi Binance.")
        return False


def _open_position_real(sym,signal,actual_entry,chat_id,order_info):
    buy=signal["decision"]=="BUY"; sl=signal.get("sl"); tp=signal.get("tp")
    qty=abs(float(order_info.get("executedQty",0)))
    if not qty:
        with positions_lock: qty=positions.get(sym,{}).get("quantity",0)
    if sl is None or tp is None:
        _emergency_close(sym,buy,qty,chat_id,"strategy tidak mengirim SL/TP"); return
    valid=(sl<actual_entry<tp) if buy else (tp<actual_entry<sl)
    if not valid:
        _emergency_close(sym,buy,qty,chat_id,"level strategy invalid setelah fill"); return
    tick=get_symbol_filters(sym)["tickSize"]; sl=round_to_tick(sl,tick); tp=round_to_tick(tp,tick)
    try:
        # Protection must be verified on Binance before local state is promoted to ACTIVE.
        t,s=place_tp_sl(sym,buy,tp,sl,qty)
    except BinanceCooldownError:
        with positions_lock:
            if sym in positions:
                now=time.time()
                positions[sym].update({"entry": actual_entry, "entry_time": now, "status": "active", "trade_uid":f"{research_run_id}:{sym}:{int(now*1000)}", "current_sl": sl, "initial_sl": sl, "quantity": qty, "tp_order_id": None, "sl_order_id": None, "trail_update_count":0,"trail_applied_count":0,"trail_failed_count":0,"trail_queued_count":0,"first_trail_r":None,"last_trail_r":None,"max_protected_r":-999.0})
        _queue_pending_protection(sym, buy, sl, tp, qty)
        tg_send(chat_id, f"⏸️ <b>PROTEKSI DITUNDA</b> — {sym}\nBinance sedang rate-limit/ban. TP/SL dicatat dan akan dipasang setelah recovery +60 detik.")
        threading.Thread(target=monitor_position_real,args=(sym,positions[sym]),daemon=True).start(); return
    except BinanceUnknownExecutionError as e:
        # place_tp_sl already reconciles algo client ids; if it still cannot prove the
        # pair exists, fail closed and flatten the real position.
        _emergency_close(sym,buy,qty,chat_id,f"status protection UNKNOWN setelah submit: {e}"); return
    except Exception as e:
        _emergency_close(sym,buy,qty,chat_id,f"gagal pasang protection ({e})"); return

    with positions_lock:
        if sym not in positions:return
        now=time.time()
        positions[sym].update({"entry":actual_entry,"entry_time":now,"status":"active","trade_uid":f"{research_run_id}:{sym}:{int(now*1000)}",
                               "current_sl":sl,"initial_sl":sl,"quantity":qty,"tp_order_id":t["algoId"],"sl_order_id":s["algoId"],
                               "execution_mode":"REAL","trail_update_count":0,"trail_applied_count":0,"trail_failed_count":0,"trail_queued_count":0,"first_trail_r":None,"last_trail_r":None,"max_protected_r":-999.0})
    tg_send(chat_id,f"⚡ <b>ENTRY REAL</b> — {sym}\nEntry: <code>{actual_entry:.8g}</code>\nTP: <code>{tp:.8g}</code> | SL: <code>{sl:.8g}</code>")
    threading.Thread(target=monitor_position_real,args=(sym,positions[sym]),daemon=True).start()


def _infer_close_reason(tp_algo_id, sl_algo_id):
    """Cek algo order mana yang TRIGGERED/FINISHED untuk tahu sebab posisi
    closed (tp/sl). TP/SL sekarang algo order (lihat place_tp_sl), jadi
    query-nya lewat get_algo_order_status, bukan get_order_status biasa."""
    tp_status = sl_status = None
    try:
        if tp_algo_id: tp_status = get_algo_order_status(tp_algo_id).get("algoStatus")
    except Exception: pass
    try:
        if sl_algo_id: sl_status = get_algo_order_status(sl_algo_id).get("algoStatus")
    except Exception: pass
    if tp_status in ("TRIGGERED", "FINISHED"): return "tp"
    if sl_status in ("TRIGGERED", "FINISHED"): return "sl"
    return "unknown"


def monitor_position_real(sym,pos):
    next_strategy=0
    while True:
        try:
                with positions_lock:
                    if sym not in positions:return
                    pos=positions[sym]

                if pos.get("timeout_flag"):
                    _verified_timeout_symbol(sym, pos.get("chat_id") or active_chat_id, reason="manual timeout")
                    return

                if _binance_is_scan_paused():
                    price = get_price(sym)
                    if price is not None:
                        _update_trade_path_metrics(pos, price)
                    time.sleep(MONITOR_SLEEP)
                    continue

                try:
                    real=get_real_position(sym)
                except BinanceCooldownError:
                    time.sleep(REAL_TRADE_POLL_SLEEP); continue
                except Exception as e:
                    log.warning(f"[monitor_real] {sym}: {e}"); time.sleep(REAL_TRADE_POLL_SLEEP); continue

                # Exchange is authoritative. get_real_position() returns None when the symbol's
                # positionAmt is 0, which is a legitimate FILLED/closed state, not a reason to keep
                # looping. Finalize it immediately so /stats, /analyze, ban, and Telegram all fire.
                if real is None:
                    reason = _infer_close_reason(pos.get("tp_order_id"), pos.get("sl_order_id"))
                    price = None
                    try:
                        price = get_price(sym)
                    except Exception:
                        price = None
                    _finalize_external_close(sym, pos, reason_hint=reason, exit_price=price)
                    return

                live=abs(float(real.get("positionAmt",0) or 0))
                if live <= 0:
                    # Defensive fallback for any future get_real_position() implementation that
                    # returns the raw zero-quantity row instead of None. Do not leave zombie local state.
                    reason = _infer_close_reason(pos.get("tp_order_id"), pos.get("sl_order_id"))
                    price = None
                    try:
                        price = get_price(sym)
                    except Exception:
                        price = None
                    _finalize_external_close(sym, pos, reason_hint=reason, exit_price=price)
                    return
                with positions_lock:
                    if sym in positions:
                        positions[sym]["quantity"]=live

                px=get_price(sym)
                if px is not None:
                    _update_trade_path_metrics(pos, px)

                if time.time()>=next_strategy:
                    upd=_strategy_position_update(sym,pos); next_strategy=time.time()+STRATEGY_MANAGE_INTERVAL
                    if isinstance(upd,dict):
                        # Strategy-requested market exit: use the same verified close path.
                        if upd.get("close"):
                            price=upd.get("close_price") or px or pos["entry"]
                            buy=pos["signal"]["decision"]=="BUY"
                            closed, exit_price = _verified_market_close(sym, buy, "strategy close", chat_id=pos.get("chat_id") or active_chat_id, max_retries=1)
                            if not closed:
                                with positions_lock:
                                    if sym in positions:
                                        positions[sym]["status"]="EMERGENCY"
                                return
                            try:
                                _cleanup_algo_orders_verified(sym)
                            except Exception as ce:
                                _queue_pending_cleanup(sym, "strategy close cleanup", ce)
                                with positions_lock:
                                    if sym in positions:
                                        positions[sym]["status"]="EMERGENCY"
                                return
                            result = "trail" if _classify_close_result("trail", pos.get("entry"), exit_price or price, pos["signal"].get("decision")) == "trail" else "sl"
                            close_position(sym,result,close_price=exit_price or price); return

                        oldsl=pos.get("current_sl",pos["signal"].get("sl")); oldtp=pos["signal"].get("tp")
                        # Calculate candidates WITHOUT mutating local state.
                        candidate_tp = float(upd.get("tp")) if upd.get("tp") is not None else float(oldtp) if oldtp is not None else None
                        candidate_sl = float(upd.get("sl")) if upd.get("sl") is not None else float(oldsl) if oldsl is not None else None
                        if candidate_sl is not None:
                            buy=pos["signal"]["decision"]=="BUY"
                            if not ((candidate_sl > oldsl) if buy else (candidate_sl < oldsl)) and candidate_sl != oldsl:
                                candidate_sl = oldsl

                        if candidate_sl != oldsl or candidate_tp != oldtp:
                            current_price = px or get_price(sym) or pos["entry"]
                            try:
                                if _binance_is_scan_paused():
                                    _queue_pending_trail(sym, candidate_sl, candidate_tp, live, reason="strategy", side=pos["signal"]["decision"])
                                    if candidate_sl != oldsl:
                                        _notify_trail_update(active_chat_id, sym, pos, upd, oldsl, candidate_sl, status="QUEUED")
                                else:
                                    # Refresh quantity from exchange immediately before protection mutation.
                                    latest = get_real_position(sym)
                                    live_qty = abs(float(latest.get("positionAmt",0) or 0)) if latest else 0.0
                                    if live_qty <= 0:
                                        continue
                                    # Cancel existing protection, then create+verify new pair. If creation
                                    # fails, restore the old pair before declaring an emergency.
                                    _cancel_all_algo_orders_verified(sym)
                                    try:
                                        nt, ns = place_tp_sl(sym, pos["signal"]["decision"]=="BUY", candidate_tp, candidate_sl, live_qty)
                                    except Exception as protect_err:
                                        restore_failed = False
                                        try:
                                            rt, rs = place_tp_sl(sym, pos["signal"]["decision"]=="BUY", oldtp, oldsl, live_qty)
                                            _verify_protection_pair(sym, pos["signal"]["decision"]=="BUY", oldtp, oldsl, live_qty)
                                        except Exception as restore_err:
                                            restore_failed = True
                                            _queue_pending_cleanup(sym, "trail protection restore failed", restore_err)
                                            log.critical(f"[trail] {sym}: restore old protection gagal: {restore_err}")
                                        if restore_failed:
                                            with positions_lock:
                                                if sym in positions:
                                                    positions[sym]["status"]="EMERGENCY"
                                                    positions[sym]["emergency_error"]=str(protect_err)[:300]
                                            raise RuntimeError(f"trail update gagal dan protection lama tidak bisa dipulihkan: {protect_err}")
                                        raise
                                    with positions_lock:
                                        if sym in positions:
                                            positions[sym]["current_sl"] = candidate_sl
                                            positions[sym]["signal"]["sl"] = candidate_sl
                                            if candidate_tp is not None:
                                                positions[sym]["signal"]["tp"] = candidate_tp
                                            positions[sym]["tp_order_id"] = nt["algoId"]
                                            positions[sym]["sl_order_id"] = ns["algoId"]
                                            positions[sym]["quantity"] = live_qty
                                            positions[sym]["current_price"] = current_price
                                    _clear_pending_trail(sym)
                                    if candidate_sl != oldsl:
                                        _notify_trail_update(active_chat_id, sym, positions.get(sym, pos), upd, oldsl, candidate_sl, status="APPLIED")
                            except BinanceCooldownError as e:
                                _queue_pending_trail(sym, candidate_sl, candidate_tp, live, reason="binance_cooldown", side=pos["signal"]["decision"])
                                if candidate_sl != oldsl:
                                    _notify_trail_update(active_chat_id, sym, pos, upd, oldsl, candidate_sl, status="QUEUED", error=e)
                            except Exception as e:
                                log.error(f"[strategy/manage real] {sym}: {e}")
                                if candidate_sl != oldsl:
                                    _notify_trail_update(active_chat_id, sym, pos, upd, oldsl, candidate_sl, status="FAILED", error=e)
                                # Do not commit candidate local state on failure.

                time.sleep(REAL_TRADE_POLL_SLEEP)

        except Exception as e:
            # Last-resort monitor guard: a single unexpected Python error must not silently
            # kill the real-position watchdog. Keep the position visible and retry.
            log.exception(f"[monitor_real] UNHANDLED {sym}: {e}")
            with positions_lock:
                if sym in positions:
                    positions[sym]["status"] = positions[sym].get("status") or "active"
                    positions[sym]["monitor_error"] = str(e)[:300]
            time.sleep(REAL_TRADE_POLL_SLEEP)

def _queue_pending_protection(sym, buy, sl, tp, qty):
    with _pending_protections_lock:
        _pending_protections[sym] = {
            "side": "BUY" if buy else "SELL", "sl": sl, "tp": tp,
            "quantity": qty, "updated_at": time.time()
        }


def _clear_pending_protection(sym):
    with _pending_protections_lock:
        _pending_protections.pop(sym, None)


def _resume_binance_and_flush_pending(chat_id=None):
    """Recover Binance without treating simulation positions as real exposure."""
    global _binance_recovering, _binance_scan_paused, _binance_pause_reason
    global _binance_recovery_notice_generation
    if _binance_cooldown_remaining() > 0:
        return False
    with _binance_pause_lock:
        _binance_recovering=True; _binance_scan_paused=True; _binance_pause_reason='recovery in progress'

    # SIMULATION-only: no private Binance sync, no credential dependency, no extra requests.
    # Existing REAL positions still force the real recovery path even if /mode is OFF.
    if not _has_real_recovery_work():
        with _binance_pause_lock:
            _binance_recovering=False; _binance_scan_paused=False; _binance_pause_reason=''
        log.info('[BINANCE RECOVERY] Simulation-only state; private Binance sync skipped. Scanner resumed.')
        return True

    failures=[]
    try:
        global BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_KEYS_PRESENT
        key,secret=_read_binance_credentials()
        if not key or not secret:
            BINANCE_KEYS_PRESENT=False
            with _binance_pause_lock:
                _binance_recovering=False; _binance_scan_paused=True; _binance_pause_reason='Binance credentials unavailable'
            log.error('[BINANCE RECOVERY] credential gate gagal: BINANCE_API_KEY/BINANCE_API_SECRET tidak tersedia di runtime Render')
            return False
        BINANCE_API_KEY, BINANCE_API_SECRET=key,secret; BINANCE_KEYS_PRESENT=True

        # REAL position sync. None means a successful request found zero exposure.
        with positions_lock:
            items=[(sym,dict(pos)) for sym,pos in positions.items() if _position_is_real(pos)]
        for sym,pos in items:
            if pos.get('status') not in ('active','EMERGENCY'):
                continue
            try:
                real=get_real_position(sym)
                if real is None or abs(float(real.get('positionAmt',0) or 0))<=0:
                    price=pos.get('current_price') or pos.get('entry')
                    try:
                        _finalize_external_close(sym,pos,reason_hint='unknown',exit_price=price)
                        log.info(f'[resume] {sym}: Binance flat -> local REAL position reconciled as closed.')
                    except Exception as e:
                        failures.append(f'{sym}: flat-position finalize {e}')
                        log.error(f'[resume] FINALIZE GAGAL {sym}: {e}')
                    continue
                live_qty=abs(float(real.get('positionAmt',0) or 0))
                with positions_lock:
                    if sym in positions:
                        positions[sym].update({'quantity':live_qty,'exchange_synced_at':time.time(),'execution_mode':'REAL'})
            except Exception as e:
                failures.append(f'{sym}: sync {e}')
                log.error(f'[resume] SYNC GAGAL {sym}: {e}')

        # Pending protection/trail/cleanup are real-only queues.
        with _pending_protections_lock:
            protections=[(sym,dict(v)) for sym,v in _pending_protections.items()]
        for sym,pr in protections:
            try:
                with positions_lock: pos=positions.get(sym)
                if not pos or not _position_is_real(pos) or pos.get('status')!='active':
                    _clear_pending_protection(sym); continue
                qty=pos.get('quantity') or pr.get('quantity'); buy=pr.get('side')=='BUY'
                if not qty or pr.get('tp') is None or pr.get('sl') is None: raise RuntimeError('pending protection tidak lengkap')
                t,s=place_tp_sl(sym,buy,pr['tp'],pr['sl'],qty)
                with positions_lock:
                    if sym in positions: positions[sym].update({'tp_order_id':t['algoId'],'sl_order_id':s['algoId'],'execution_mode':'REAL'})
                _clear_pending_protection(sym)
            except Exception as e:
                failures.append(f'{sym}: protection {e}'); log.error(f'[protection-resume] {sym} GAGAL: {e}')

        with _pending_trails_lock:
            pending=[(sym,dict(v)) for sym,v in _pending_trails.items()]
        for sym,tr in pending:
            try:
                with positions_lock: pos=positions.get(sym)
                if not pos or not _position_is_real(pos) or pos.get('status')!='active':
                    _clear_pending_trail(sym); continue
                buy=pos['signal']['decision']=='BUY'; qty=pos.get('quantity') or tr.get('quantity')
                tp=tr.get('tp') or pos['signal'].get('tp'); sl=tr.get('sl') or pos.get('current_sl')
                if not qty or sl is None or tp is None: raise RuntimeError('pending trail tidak lengkap')
                cancel_algo_order(pos.get('tp_order_id')); cancel_algo_order(pos.get('sl_order_id'))
                t,s=place_tp_sl(sym,buy,tp,sl,qty)
                old_sl = pos.get('current_sl')
                with positions_lock:
                    if sym in positions:
                        positions[sym].update({'tp_order_id':t['algoId'],'sl_order_id':s['algoId'],'exchange_synced_at':time.time(),'execution_mode':'REAL','current_sl':sl})
                if old_sl is not None and sl is not None and float(sl) != float(old_sl):
                    _record_trail_event(sym, pos, {'trail_source':'pending_resume','reason':['queued trail applied after Binance recovery']}, old_sl, sl, status='APPLIED')
                _clear_pending_trail(sym)
            except Exception as e:
                failures.append(f'{sym}: trail {e}'); log.error(f'[trail-resume] {sym} GAGAL: {e}')

        with _pending_cleanup_lock:
            cleanup_items=list(_pending_cleanup.items())
        for sym,item in cleanup_items:
            try:
                _cleanup_algo_orders_verified(sym)
            except Exception as e:
                failures.append(f'{sym}: cleanup {e}'); log.error(f'[cleanup-resume] {sym} GAGAL: {e}')

        if failures:
            with _binance_pause_lock:
                _binance_recovering=False; _binance_scan_paused=True; _binance_pause_reason='recovery incomplete'; generation=_binance_pause_generation
            msg=' | '.join(failures[:6])
            log.error(f'[BINANCE RECOVERY] BELUM SELESAI — scanner tetap PAUSED. {msg}')
            with _binance_recovery_notice_lock:
                should_notify=generation!=_binance_recovery_notice_generation
                if should_notify: _binance_recovery_notice_generation=generation
            if chat_id and should_notify:
                tg_send(chat_id,'⚠️ <b>Binance recovery belum selesai.</b>\nScanner tetap dihentikan karena sync/protection REAL masih gagal.\nDetail: <code>'+msg[:500]+'</code>')
            return False

        with _binance_pause_lock:
            _binance_recovering=False; _binance_scan_paused=False; _binance_pause_reason=''
        if chat_id: tg_send(chat_id,'✅ <b>Binance recovery selesai.</b>\nREAL position/protection state sudah konsisten. Scanning boleh resume.')
        return True
    except Exception as e:
        with _binance_pause_lock:
            _binance_recovering=False; _binance_scan_paused=True; _binance_pause_reason='recovery exception'
        log.error(f'[BINANCE RECOVERY] exception — scanner tetap PAUSED: {e}',exc_info=True)
        return False

def _binance_recovery_loop(chat_id_getter=lambda: active_chat_id):
    """Watchdog global. One pause notice per pause generation; strict recovery gates resume."""
    while True:
        try:
            if _binance_is_scan_paused():
                _notify_binance_pause_once(chat_id_getter())
                if _binance_cooldown_remaining() <= 0 and not _binance_recovering:
                    _resume_binance_and_flush_pending(chat_id_getter())
                else:
                    time.sleep(5)
                continue
        except Exception as e:
            log.warning(f"[binance-recovery] {e}")
        time.sleep(5)


def autostop_loop(chat_id):
    """Background: pantau saldo real, auto /stop kalau drawdown dari peak > AUTOSTOP_PCT."""
    global auto_mode, peak_real_balance
    while True:
        try:
            if REAL_TRADE_ENABLED and not _binance_is_scan_paused():
                _, total = get_real_balance()
                if total is not None:
                    with autostop_lock:
                        if peak_real_balance is None or total > peak_real_balance:
                            peak_real_balance = total
                        drawdown_pct = (peak_real_balance - total) / peak_real_balance * 100 if peak_real_balance else 0
                    if auto_mode and drawdown_pct >= AUTOSTOP_PCT:
                        auto_mode = False
                        tg_send(chat_id,
                            f"🛑 <b>AUTO-STOP TERPICU</b>\n\n"
                            f"Saldo turun <b>{drawdown_pct:.2f}%</b> dari peak "
                            f"(${peak_real_balance:.2f} → ${total:.2f})\n"
                            f"Threshold: {AUTOSTOP_PCT}%\n\n"
                            f"Scan sinyal baru dihentikan. Posisi aktif tetap dipantau.\n"
                            f"Jalankan lagi manual dengan /auto")
        except Exception as e:
            log.warning(f"[autostop_loop] {e}")
        time.sleep(60)


def simulation_loop(chat_id):
    """Koordinator runtime; seluruh keputusan trading berasal dari strategy."""
    global auto_mode
    tg_send(chat_id,"🤖 <b>Engine dimulai.</b>\nStrategy mengendalikan Entry/TP/SL/Trail.")
    scanning=False; scan_lock=threading.Lock(); last_scan=0.0

    def do_scan():
        nonlocal scanning
        try:
            signals = run_scan_once(chat_id)
            if not auto_mode or not signals:
                return

            opened = 0
            for signal in signals:
                if not auto_mode or _binance_is_scan_paused():
                    break
                sym = signal.get("symbol")
                if not sym:
                    continue
                with positions_lock:
                    if sym in positions or len(positions) >= MAX_POSITIONS:
                        continue

                if REAL_TRADE_ENABLED:
                    _open_pending_real(sym, signal, chat_id)
                    opened += 1
                    continue

                price = signal.get("price") or get_price(sym)
                entry = signal.get("entry")
                if price is None or entry is None:
                    continue
                mode = str(signal.get("execution_mode", "")).lower() or ("market" if signal.get("entry_label") == "market" else "limit")
                with positions_lock:
                    if sym in positions or len(positions) >= MAX_POSITIONS:
                        continue
                    positions[sym] = {"signal": signal, "entry": entry, "chat_id": chat_id,
                                      "entry_time": None, "timeout_flag": False, "status": "pending",
                                      "execution_mode": "SIMULATION"}
                if mode == "market":
                    _open_position(sym, signal, get_price(sym) or price, chat_id, "strategy")
                else:
                    tg_send(chat_id, f"🎯 <b>PENDING ORDER</b> — {sym}\n\n{fmt_signal_msg(signal)}")
                    threading.Thread(target=wait_entry, args=(sym, signal, chat_id), daemon=True).start()
                opened += 1

            log.info(f"[scan] {len(signals)} signal lolos, {opened} dikirim ke execution")
        finally:
            with scan_lock:
                scanning = False

    def wait_entry(sym,signal,chat_id):
        entry=signal["entry"]; buy=signal["decision"]=="BUY"; deadline=time.time()+8*3600
        while time.time()<deadline:
            with positions_lock:
                if sym not in positions:return
                if positions[sym].get("timeout_flag"): positions.pop(sym,None); return
            price=get_price(sym)
            if price is not None and ((price<=entry) if buy else (price>=entry)):
                fill=min(entry,price) if buy else max(entry,price)
                _open_position(sym,signal,fill,chat_id,"strategy"); return
            time.sleep(MONITOR_SLEEP)
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,"pending expired")

    while auto_mode:
        if _binance_is_scan_paused():
            time.sleep(5)
            continue
        with positions_lock: full=len(positions)>=MAX_POSITIONS
        if full: time.sleep(5); continue
        with scan_lock:
            if scanning: time.sleep(5); continue
            scanning=True
        if time.time()-last_scan<120:
            with scan_lock: scanning=False
            time.sleep(5); continue
        last_scan=time.time(); threading.Thread(target=do_scan,daemon=True).start(); time.sleep(5)
    tg_send(chat_id,"⏹ <b>Scanning dihentikan.</b>\n\n"+fmt_stats())




# ═════════════════════════════════════════════
# PESAN STATIS
# ═════════════════════════════════════════════
def get_start_msg():
    return (
        "👋 <b>SMC Signal Broadcaster</b>\n\n"
        f"Mode: <b>{'REAL TRADE' if REAL_TRADE_ENABLED else 'SIMULASI'}</b>\n"
        f"Posisi aktif: <b>{MAX_POSITIONS}</b> maksimum\n"
        f"Confidence minimum: <b>{STRATEGY_CONFIDENCE_THRESHOLD}%</b>\n"
        "TP minimum: <b>2R</b> • Max RR: <b>Unlimited</b>\n"
        "Trailing: <b>Adaptive / context-aware</b>\n\n"
        "━━━━━━━━ <b>TRADING</b> ━━━━━━━━\n"
        "/auto                — Mulai scanning & trading\n"
        "/stop                — Hentikan scanning; posisi aktif tetap dipantau\n"
        "/trade               — Lihat semua posisi aktif/pending/emergency\n"
        "/ok SYMBOL           — Rekonsiliasi posisi Binance + TP/SL\n"
        "/timeout SYMBOL      — Tutup paksa posisi tertentu\n"
        "/timeout             — Tutup paksa semua posisi + semua order\n\n"
        "━━━━━━━━ <b>CONFIG</b> ━━━━━━━━━\n"
        "/mode                — Lihat mode aktif\n"
        "/mode on             — Aktifkan REAL TRADE\n"
        "/mode off            — Aktifkan SIMULASI\n"
        "/max                 — Lihat/ubah batas posisi\n"
        "/leverage            — Lihat/ubah leverage\n"
        "/margin              — Lihat/ubah margin awal/trade\n"
        "/confidence_min      — Lihat confidence minimum\n"
        "/confidence_min 70   — Ubah threshold confidence\n"
        "/autostop            — Lihat/ubah auto-stop drawdown\n\n"
        "━━━━━━━━ <b>RESEARCH</b> ━━━━━━━━\n"
        "/stats               — Statistik & saldo aktif\n"
        "/backtest            — 20 trade terakhir\n"
        "/analyze             — Analisis seluruh closed trade sejak resetstats\n"
        "/resetstats           — Reset research stats/ledger tanpa mengubah modal\n\n"
        "━━━━━━━━ <b>TOOLS</b> ━━━━━━━━━━\n"
        "/ganti               — Upload/ganti strategy_logic.py\n"
        "/info                — Detail engine & metode analisis\n"
        "/IP                  — Lihat public IP Render saat ini\n"
        "/banned              — Lihat daftar koin yang diban\n"
        "/koin                — Lihat koin yang sedang/terakhir di-scan\n"
        "/resetban             — Hapus semua ban koin\n"
        "/unban SYMBOL         — Lepas ban satu koin\n"
        "/timer               — Lihat/ubah ban pendek (scan)\n"
        "/timer 20            — Ban pendek 20 scan\n"
        "/reject              — Lihat/ubah warmup reject setelah /resetstats\n"
        "/reject 5            — Tolak 5 sinyal awal setelah /resetstats\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        + ("🔴 <b>REAL TRADE AKTIF</b> — order sungguhan di Binance Futures, uang beneran."
           if REAL_TRADE_ENABLED else
           "⚠️ <i>Mode simulasi — tidak mengeksekusi order sungguhan.</i>")
    )

def get_info_msg():
    return ("ℹ️ <b>Engine</b>\n\n"
            "Strategy: Entry • Stop Loss • Take Profit • Trail • Confidence • Setup selection\n"
            "Engine: data transport • Telegram • order execution • position state • monitoring • statistics.")



# ═════════════════════════════════════════════
# RENDER KEEP-ALIVE / TELEGRAM WATCHDOG
# ═════════════════════════════════════════════
def _render_keepalive_loop():
    """Best-effort self health ping untuk mengurangi risiko Render Free idle.

    Render_EXTERNAL_URL dipakai kalau tersedia. Endpoint /healthz tidak menyentuh
    Binance, jadi loop ini tidak menambah Binance weight. Untuk jaminan terhadap
    spin-down Free, external uptime monitor tetap lebih kuat; loop ini adalah
    lapisan tambahan, bukan satu-satunya mekanisme.
    """
    base = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
    if not base:
        log.info("[render] RENDER_EXTERNAL_URL tidak tersedia — keepalive internal off")
        return
    url = f"{base}/healthz"
    while True:
        try:
            r = requests.get(url, timeout=10)
            if r.ok:
                log.debug("[render] keepalive OK")
            else:
                log.warning(f"[render] keepalive HTTP {r.status_code}")
        except Exception as e:
            log.debug(f"[render] keepalive gagal: {e}")
        time.sleep(TELEGRAM_KEEPALIVE_SEC)


def _telegram_watchdog_alert(cid, text):
    global _telegram_last_conflict_alert_at
    now = time.time()
    if now - _telegram_last_conflict_alert_at < 300:
        return
    _telegram_last_conflict_alert_at = now
    if cid:
        tg_send(cid, text)


# ═════════════════════════════════════════════
# BOT LOOP
# ═════════════════════════════════════════════
def bot_loop():
    global auto_mode, auto_thread, autostop_thread, active_chat_id, timeout_flag, MAX_POSITIONS, LEVERAGE, MARGIN_USD, AUTOSTOP_PCT, peak_real_balance, REAL_TRADE_ENABLED, STRATEGY_CONFIDENCE_THRESHOLD, BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_KEYS_PRESENT, real_balance_snapshot, real_balance_snapshot_at, early_reject_configured, early_reject_remaining, BAN_DURATION_SCANS

    # Set active_chat_id ke ALLOWED_USER_ID SEJAK AWAL — di chat pribadi
    # Telegram, chat_id sama dengan user_id, jadi bot bisa kirim pesan
    # proaktif (termasuk "Bot Siap" & notifikasi darurat) SEBELUM user
    # mengirim perintah apa pun. Sebelumnya active_chat_id cuma None
    # sampai user chat duluan, jadi notifikasi penting tidak pernah sampai.
    if ALLOWED_USER_ID:
        active_chat_id = ALLOWED_USER_ID

    # Tidak ada startup ping Binance. Request hanya dilakukan saat benar-benar dibutuhkan.

    try:
        _binance_sync_time(force=True)
    except Exception:
        pass
    _telegram_bootstrap()
    offset=None
    poll_backoff=1
    log.info(f"Bot siap — main.py {MAIN_ENGINE_VERSION}.")
    if ALLOWED_USER_ID:
        tg_send(ALLOWED_USER_ID,
            "✅ <b>Bot Siap</b>\n"
            "Semua sistem sudah menyala dan siap menerima perintah.\n"
            "Ketik /start untuk melihat menu.")
        # Always report the public Render IP at startup, regardless of trading mode.
        # This is a non-Binance request and is best-effort only; failure must never
        # prevent the bot from starting or receiving commands.
        ip = get_public_ip()
        if ip and ip != "unknown":
            tg_send(ALLOWED_USER_ID,
                    f"🌐 <b>Public IP Render</b>\n<code>{html.escape(ip)}</code>\n\n"
                    "Whitelist IP ini di Binance API Management jika IP restriction aktif.")
        else:
            tg_send(ALLOWED_USER_ID,
                    "⚠️ <b>Public IP Render</b>\nTidak berhasil diambil saat startup.\n"
                    "Gunakan /IP untuk mencoba lagi.")

    while True:
        try:
            updates = tg_updates(offset)
            poll_backoff = 1
            for upd in updates:
                offset=upd["update_id"]+1
                msg=upd.get("message",{})
                uid=msg.get("from",{}).get("id")
                chat_id=msg.get("chat",{}).get("id")
                # Pesan berisi DOKUMEN pakai field "caption", bukan "text" —
                # "text" cuma ada di pesan teks polos tanpa lampiran. Sebelumnya
                # cuma baca "text", jadi /ganti (dikirim sbg dokumen + caption)
                # selalu ke-skip diam-diam di baris `if ... not text: continue`
                # di bawah, sebelum sempat sampai ke handler manapun.
                text=(msg.get("text") or msg.get("caption") or "").strip().lower()
                if not uid or not chat_id or not text: continue
                if uid!=ALLOWED_USER_ID:
                    tg_send(chat_id,"⛔ Akses ditolak."); continue
                active_chat_id=chat_id

                if text in ("/start","start"):
                    tg_send(chat_id,get_start_msg())
                elif text.startswith("/confidence_min") or text.startswith("confidence_min"):
                    parts = text.split()
                    if len(parts) == 1:
                        tg_send(chat_id, f"🎯 <b>Confidence minimum:</b> {STRATEGY_CONFIDENCE_THRESHOLD}%\nGunakan <code>/confidence_min 70</code> untuk mengubahnya.")
                    else:
                        try:
                            val = float(parts[1].replace("%", ""))
                            if not (0 <= val <= 100):
                                raise ValueError("rentang 0-100")
                            STRATEGY_CONFIDENCE_THRESHOLD = int(round(val))
                            tg_send(chat_id, f"✅ Confidence minimum diubah menjadi <b>{STRATEGY_CONFIDENCE_THRESHOLD}%</b>.")
                        except Exception:
                            tg_send(chat_id, "❌ Format salah. Gunakan <code>/confidence_min 70</code> (0-100).")
                elif text in ("/info","info"):
                    tg_send(chat_id,get_info_msg())
                elif text in ("/ip","ip"):
                    ip = get_public_ip()
                    if ip and ip != "unknown":
                        tg_send(chat_id, f"🌐 <b>Public IP Render</b>\n<code>{html.escape(ip)}</code>\n\nGunakan IP ini untuk whitelist Binance API jika diperlukan.")
                    else:
                        tg_send(chat_id, "⚠️ Public IP Render tidak berhasil diambil dari layanan IP eksternal saat ini.")
                elif text in ("/stats","stats"):
                    tg_send(chat_id,fmt_stats())
                elif text in ("/backtest","backtest"):
                    tg_send(chat_id,fmt_backtest())
                # ============================================================
                # TAMBAHAN BARU (START) — Handler /analyze
                # ============================================================
                elif text in ("/full on", "full on"):
                    try:
                        tg_send(chat_id, _full_on())
                    except Exception as e:
                        log.exception(f"[full] on gagal: {e}")
                        tg_send(chat_id, f"❌ FULL gagal diaktifkan: <code>{html.escape(str(e)[:300])}</code>")
                elif text in ("/full off", "full off"):
                    try:
                        tg_send(chat_id, _full_off())
                    except Exception as e:
                        log.exception(f"[full] off gagal: {e}")
                        tg_send(chat_id, f"❌ FULL gagal dimatikan: <code>{html.escape(str(e)[:300])}</code>")
                elif text in ("/full", "full"):
                    tg_send(chat_id, _full_status_text())
                elif text in ("/analyze","analyze"):
                    # Research analysis dari FULL CLOSED-TRADE LEDGER; TIDAK scan Binance.
                    def _run_analyze(cid):
                        try:
                            with trade_history_lock:
                                trade_count = len(trade_history)
                            tg_send(cid,
                                f"🔎 <b>Mulai /analyze</b>\n"
                                f"Menganalisis <b>{trade_count}</b> closed trade yang tercatat sejak /resetstats terakhir.\n"
                                f"Tidak melakukan scan market baru.\n"
                                f"Dibuat: 3 file research terstruktur.")
                            rows, summary, hist = _analyze_snapshot()
                            report_path = _write_analyze_report(rows, summary, hist)
                            csv_path = _write_analyze_csv(rows)
                            support_paths = _write_research_support_files(summary)
                            tg_send(cid,
                                f"✅ <b>/analyze selesai</b>\n"
                                f"Closed trade dianalisis: <b>{len(rows)}</b>\n"
                                f"Trail events: <b>{len(_trail_events_snapshot(summary.get('run_id', research_run_id)))}</b>\n"
                                f"Low-confidence history: <b>{len(_low_conf_snapshot())}</b>\n"
                                f"Run: <code>{summary.get('run_id', research_run_id)}</code>\n\n"
                                f"Mengirim <b>3 file research</b> (7 dataset digabung losslessly)...")
                            tg_send_document(cid, report_path, caption="📊 analyze_report.md — ringkasan analisis")
                            tg_send_document(cid, csv_path, caption="📋 analyze_data.csv — seluruh closed trade")
                            tg_send_document(cid, support_paths, caption="🧠 analyze_research_bundle.json — 5 dataset research: trail events, trail summary, scan quality, low-confidence bans, market context")
                        except Exception as e:
                            log.error(f"[analyze] Error: {e}", exc_info=True)
                            tg_send(cid, f"❌ Error saat /analyze:\n<code>{str(e)[:300]}</code>")

                    threading.Thread(target=_run_analyze, args=(chat_id,), daemon=True).start()
                    tg_send(chat_id, "⏳ /analyze berjalan di background berdasarkan history trade. Bot tetap menerima perintah lain.")
# ============================================================
# TAMBAHAN BARU (END)
# ============================================================
                # ============================================================
# TAMBAHAN BARU (START) — Handler /ganti (Upload Otak Baru via GitHub API)
# ============================================================
                elif text in ("/ganti","ganti"):
                   doc = msg.get("document")
                   if not doc:
                       tg_send(chat_id, "📤 Kirim file strategy_logic.py sebagai dokumen dengan caption /ganti")
                       continue
               
                   file_name = doc.get("file_name", "")
                   if not file_name.endswith(".py"):
                       tg_send(chat_id, "❌ Harus file .py")
                       continue
               
                   try:
                       # 1. Download file dari Telegram
                       file_id = doc["file_id"]
                       file_info = requests.get(
                           f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile",
                           params={"file_id": file_id}, timeout=10
                       ).json()
                       file_path = file_info["result"]["file_path"]
                       file_content = requests.get(
                           f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}",
                           timeout=10
                       ).text
               
                       # 2. Validasi sintaks
                       try:
                           compiled = compile(file_content, "strategy_logic.py", "exec")
                       except SyntaxError as e:
                           tg_send(chat_id, f"❌ Error sintaks di file:\n<code>{e}</code>")
                           continue
               
                       # 3. Validasi full_analyze() ADA
                       check_ns = {}
                       try:
                           exec(compiled, check_ns)
                       except Exception as e:
                           tg_send(chat_id, f"❌ File error saat dijalankan (bukan cuma sintaks):\n<code>{e}</code>")
                           continue
                       if "full_analyze" not in check_ns or not callable(check_ns["full_analyze"]):
                           tg_send(chat_id, "❌ File ini tidak punya fungsi full_analyze() — ditolak.")
                           continue
               
                       # 4. Commit ke GitHub
                       try:
                           _commit_to_github(file_content, "strategy_logic.py", f"Update strategy_logic via Telegram /ganti")
                           tg_send(chat_id, "✅ File berhasil di-commit ke GitHub!")
                       except Exception as e:
                           tg_send(chat_id, f"❌ Gagal commit ke GitHub:\n<code>{str(e)[:200]}</code>")
                           continue
               
                       # 5. Tulis ke file LOKAL
                       local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_logic.py")
                       with open(local_path, "w", encoding="utf-8") as f:
                           f.write(file_content)
               
                       # 6. ========== ADAPTIVE RELOAD: Bind apa yang ADA, pertahankan yang tidak ==========
                       import importlib, sys

                       # Hapus modul dari cache supaya reload benar-benar dari disk
                       if "strategy_logic" in sys.modules:
                           del sys.modules["strategy_logic"]

                       import strategy_logic as sl
                       _strategy_set_ml_model(_ml_current_champion())

                       # --- SENTINEL untuk membedakan "tidak ada" vs None ---
                       _SL_SENTINEL = object()

                       def _sl_bind(name):
                           """Bind ke global HANYA kalau ada di modul baru.
                           Kalau tidak ada -> global lama tetap aktif, return False."""
                           val = getattr(sl, name, _SL_SENTINEL)
                           if val is not _SL_SENTINEL:
                               globals()[name] = val
                               return True
                           return False

                       # WAJIB: full_analyze sudah divalidasi ada di atas
                       globals()["full_analyze"] = sl.full_analyze

                       # -- Fungsi opsional --------------------------------------------------
                       # Kalau tidak ada di file baru -> versi lama di global tetap aktif.
                       # Kamu bebas ganti nama, tambah, atau hapus fungsi apapun
                       # selama full_analyze() tetap ada.
                       _OPT_FNS = ["manage_position"]
                       _bound_fns, _kept_fns = [], []
                       for _fn in _OPT_FNS:
                           (_bound_fns if _sl_bind(_fn) else _kept_fns).append(_fn)

                       # Tangkap semua public callable BARU yang tidak ada di daftar atas
                       for _attr in dir(sl):
                           if _attr.startswith("__"):
                               continue
                           if _attr not in _OPT_FNS and _attr != "full_analyze":
                               _v = getattr(sl, _attr, None)
                               if callable(_v):
                                   globals()[_attr] = _v
                                   if _attr not in _bound_fns:
                                       _bound_fns.append(f"✨{_attr}")

                       # -- Konstanta opsional -----------------------------------------------
                       # Kalau tidak ada di file baru, nilai lama dipertahankan.
                       _OPT_CONSTS = []
                       _bound_consts, _kept_consts = [], []
                       for _k in _OPT_CONSTS:
                           (_bound_consts if _sl_bind(_k) else _kept_consts).append(_k)

                       # -- Laporan ke user --------------------------------------------------
                       _rpt = ["✅ <b>Strategy logic aktif!</b>"]
                       if _bound_fns:
                           _rpt.append(f"🔄 Diperbarui: <code>{', '.join(_bound_fns)}</code>")
                       if _kept_fns:
                           _rpt.append(f"♻️ Versi lama dipertahankan: <code>{', '.join(_kept_fns)}</code>")
                       if _bound_consts:
                           _rpt.append(f"📐 Konstanta diperbarui: <code>{', '.join(_bound_consts)}</code>")
                       if _kept_consts:
                           _rpt.append(f"📌 Konstanta lama dipertahankan: <code>{', '.join(_kept_consts)}</code>")

                       log.info("[OTAK] Strategy logic di-reload (adaptive bind).")
                       tg_send(chat_id, "\n".join(_rpt))
               
                   except Exception as e:
                       log.error(f"[ganti] Error: {e}")
                       tg_send(chat_id, f"❌ Gagal mengganti strategy_logic:\n<code>{str(e)[:200]}</code>")
                # ============================================================
                # TAMBAHAN BARU (END)
                # ============================================================
                elif text.startswith("/banned") or text.startswith("banned"):
                    parts = text.split()
                    if len(parts) > 1:
                        target_sym = parts[1].upper()
                        _ban_coin(target_sym, reason="manual", duration=float("inf"), kind="manual")
                        tg_send(chat_id, f"🚫 <b>{html.escape(target_sym)}</b> diban PERMANEN.\nLepas dengan <code>/unban {html.escape(target_sym)}</code> atau <code>/resetban</code>.")
                    else:
                        with ban_lock:
                            cur_scan = scan_counter
                            b = sorted(banned_coins.items())
                        output_lines = [f"🚫 <b>Banned ({len(b)}):</b>"]
                        if b:
                            for sym, meta in b:
                                sym_e = html.escape(str(sym))
                                if isinstance(meta, tuple):
                                    banned_at, dur = meta; reason = "legacy"; kind = "short"; conf_txt = ""
                                else:
                                    banned_at, dur = meta.get("banned_at", cur_scan), meta.get("duration", 0)
                                    reason = html.escape(str(meta.get("reason", "") or ""))
                                    kind = html.escape(str(meta.get("kind", "short") or "short"))
                                    c = meta.get("confidence")
                                    conf_txt = f" C{float(c):.0f}%" if c is not None else ""
                                if dur == float("inf"):
                                    output_lines.append(f"• {sym_e} (PERMANEN) | {kind}{conf_txt} | {reason}")
                                else:
                                    remaining = max(0.0, float(dur) - (cur_scan - float(banned_at)))
                                    output_lines.append(f"• {sym_e} ({remaining:g} scan) | {kind}{conf_txt} | {reason}")
                        else:
                            output_lines.append("• (tidak ada ban aktif)")

                        lc = _low_conf_summary()
                        if lc:
                            output_lines.append("\n🧠 <b>Low-confidence frequency:</b>")
                            for x in lc[:10]:
                                output_lines.append(f"• {html.escape(str(x['symbol']))} — {x['count']}x | avg C{x['avg']:.1f}% | min C{x['min']:.1f}%")

                        # Telegram messages have a hard length limit. Send line-aware chunks.
                        chunk = ""
                        for line in output_lines:
                            candidate = (chunk + "\n" + line).strip()
                            if chunk and len(candidate) > 3500:
                                tg_send(chat_id, chunk)
                                chunk = line
                            else:
                                chunk = candidate
                        if chunk:
                            tg_send(chat_id, chunk)
                elif text.startswith("/unban") or text.startswith("unban"):
                    parts = text.split()
                    if len(parts) != 2:
                        tg_send(chat_id, "❌ Format: <code>/unban SYMBOL</code>")
                    else:
                        sym = parts[1].upper()
                        if _unban_coin(sym):
                            tg_send(chat_id, f"✅ <b>{sym}</b> di-unban.")
                        else:
                            tg_send(chat_id, f"ℹ️ <b>{sym}</b> tidak sedang diban.")
                elif text.startswith("/timer") or text.startswith("timer"):
                    parts = text.split()
                    if len(parts) == 1:
                        tg_send(chat_id, f"⏱️ <b>Ban pendek:</b> {BAN_DURATION_SCANS:g} scan\nDipakai untuk pending/no-trade dan low-confidence.\nUbah: <code>/timer 20</code> atau <code>/timer 7.5</code>")
                    elif len(parts) == 2:
                        try:
                            val = float(parts[1].replace(",", "."))
                            if not (val > 0 and val <= 10000): raise ValueError
                            BAN_DURATION_SCANS = val
                            tg_send(chat_id, f"✅ Ban pendek diubah menjadi <b>{BAN_DURATION_SCANS:g} scan</b>.")
                        except ValueError:
                            tg_send(chat_id, "❌ Format timer salah. Gunakan misalnya <code>/timer 20</code> atau <code>/timer 7.5</code>.")
                    else:
                        tg_send(chat_id, "❌ Format: <code>/timer</code> atau <code>/timer 20</code>")
                elif text.startswith("/reject") or text.startswith("reject"):
                    parts = text.split()
                    with early_reject_lock:
                        remaining_now = early_reject_remaining
                        configured_now = early_reject_configured
                    if len(parts) == 1:
                        tg_send(chat_id, f"🛡️ <b>Warmup reject:</b> {configured_now} scan\nTersisa: {remaining_now} scan\nSetiap scan warmup: semua signal qualified ditolak, tanpa ban.\nUbah: <code>/reject 5</code> atau matikan <code>/reject 0</code>")
                    elif len(parts) == 2:
                        try:
                            val = int(float(parts[1]))
                            if val < 0 or val > 1000: raise ValueError
                            with early_reject_lock:
                                early_reject_configured = val
                                early_reject_remaining = val
                            tg_send(chat_id, f"✅ Warmup reject diubah menjadi <b>{val} scan</b>. Counter aktif langsung di-reset ke {val} scan.")
                        except ValueError:
                            tg_send(chat_id, "❌ Format reject salah. Gunakan <code>/reject 5</code>.")
                    else:
                        tg_send(chat_id, "❌ Format: <code>/reject</code> atau <code>/reject 5</code> (satuan scan)")
                elif text in ("/koin","koin"):
                    with _last_scanned_lock:
                        coins = list(last_scanned_coins)
                        scanned_at = last_scanned_at
                    if not coins:
                        tg_send(chat_id, "⏳ Belum ada data — tunggu siklus scan pertama selesai.")
                    else:
                        age_min = (time.time() - scanned_at) / 60 if scanned_at else 0
                        tg_send(chat_id,
                            f"📋 <b>Koin yang di-scan ({len(coins)})</b> — update {age_min:.0f} menit lalu:\n\n"
                            + ", ".join(coins))
                elif text in ("/resetban","resetban"):
                    with ban_lock:
                        n=len(banned_coins); banned_coins.clear()
                    tg_send(chat_id,f"✅ Ban direset ({n} dihapus).")
                elif text in ("/resetlowconf","resetlowconf"):
                    with low_conf_history_lock:
                        cleared_lc=len(low_conf_history); low_conf_history.clear()
                    tg_send(chat_id,f"✅ <b>Low-confidence history direset.</b> Event dihapus: <b>{cleared_lc}</b>. Current ban tidak disentuh.")
                elif text in ("/resetstats","resetstats"):
                    global research_run_id, trade_sequence, trail_event_sequence
                    with stat_lock:
                        current_balance = stats["balance"]
                        stats["pnl_history"] = deque(maxlen=20)
                        stats["tp"]          = 0
                        stats["sl"]          = 0
                        stats["trail"]       = 0
                        stats["total"]       = 0
                        stats["balance"]     = current_balance
                    with trade_history_lock:
                        cleared = len(trade_history)
                        trade_history.clear()
                    trade_sequence = 0
                    with trail_events_lock:
                        trail_events.clear(); trail_event_sequence=0
                    with scan_quality_lock:
                        scan_quality_history.clear()
                    with market_context_lock:
                        market_context_history.clear()
                    research_run_id = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
                    with pending_cancel_lock:
                        pending_cancel_stats.clear()
                        pending_cancel_stats.update({"tp_before_entry": 0, "expired": 0, "binance_reject": 0})
                    with early_reject_lock:
                        early_reject_remaining = early_reject_configured
                    tg_send(chat_id,
                        f"✅ <b>Research stats direset.</b>\n"
                        f"Closed trade ledger dihapus: <b>{cleared}</b>\n"
                        f"Run baru: <code>{research_run_id}</code>\n"
                        f"💵 Balance TIDAK diubah: <b>${current_balance:.4f}</b>\n"
                        f"🛡️ Warmup reject aktif: <b>{early_reject_configured}</b> sinyal awal.")
                elif text in ("/auto","auto"):
                    if auto_mode:
                        tg_send(chat_id,"⚙️ Broadcaster sudah berjalan.")
                    else:
                        # Reset referensi peak ke saldo SEKARANG — supaya drawdown
                        # dihitung ulang dari titik ini, bukan dari peak lama yang
                        # bikin auto-stop langsung kepicu lagi begitu /auto ditekan.
                        if REAL_TRADE_ENABLED:
                            _, total = get_real_balance()
                            with autostop_lock:
                                peak_real_balance = total
                        auto_mode=True
                        auto_thread=threading.Thread(
                            target=simulation_loop,args=(chat_id,),daemon=True)
                        auto_thread.start()
                elif text in ("/stop","stop"):
                    # /stop hanya mematikan scanning sinyal baru — posisi
                    # yang sudah berjalan tetap dipantau sampai TP/SL alami.
                    if auto_mode:
                        auto_mode = False
                        with positions_lock:
                            n_active = len(positions)
                        tg_send(chat_id,
                            f"⏹ <b>Scanning dihentikan.</b>\n"
                            f"Posisi aktif ({n_active}) tetap dipantau sampai TP/SL.\n"
                            f"Pakai /timeout SYMBOL kalau mau tutup paksa.")
                    else:
                        tg_send(chat_id,"ℹ️ Broadcaster tidak berjalan.")
                elif text in ("/trade","trade"):
                    with positions_lock:
                        pos_list = list(positions.items())
                    # Safety-first display order: EMERGENCY, ACTIVE, then PENDING;
                    # within each status, highest confidence first.
                    status_rank = {"EMERGENCY": 0, "active": 1, "pending": 2}
                    pos_list.sort(key=lambda item: (
                        status_rank.get(str(item[1].get("status", "active")), 3),
                        -float((item[1].get("signal") or {}).get("confidence", 0) or 0),
                        str(item[0])
                    ))
                    if not pos_list:
                        tg_send(chat_id,"ℹ️ Tidak ada posisi aktif.")
                    else:
                        lines = [f"📡 <b>Posisi Aktif ({len(pos_list)}/{MAX_POSITIONS})</b>\n"]
                        for s, p in pos_list:
                            sig    = p["signal"]
                            is_buy = sig["decision"] == "BUY"
                            em     = "🟢" if is_buy else "🔴"
                            status = p.get("status", "active")

                            if status == "pending":
                                pr       = get_price(s) or p["entry"]
                                dist_pct = abs(p["entry"] - pr) / pr * 100
                                lines.append(
                                    f"\n⏳ <b>{s}</b> — PENDING\n"
                                    f"{em} {sig['decision']} | Entry zone: <code>{p['entry']:.6g}</code>\n"
                                    f"Harga kini: <code>{pr:.6g}</code> | Jarak: {dist_pct:.2f}%\n"
                                    f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{sig['sl']:.6g}</code> | Confidence: <b>{float(sig.get('confidence', 0) or 0):.0f}%</b>"
                                )
                            else:
                                pr  = get_price(s) or p["entry"]
                                pnl = (pr - p["entry"]) / p["entry"] * 100 * (1 if is_buy else -1)
                                entry_clock = datetime.fromtimestamp(
                                    p["entry_time"], tz=WIB).strftime("%H:%M") if p.get("entry_time") else "?"
                                cur_sl = p.get("current_sl", sig["sl"])
                                trail_note = " 🔒trailing" if cur_sl != sig["sl"] else ""
                                if status == "EMERGENCY":
                                    lines.append(
                                        f"\n🚨 <b>{s}</b> — EMERGENCY\n"
                                        f"Entry: <code>{p['entry']:.6g}</code> | Harga: <code>{pr:.6g}</code>\n"
                                        f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{cur_sl:.6g}</code>{trail_note} | Confidence: <b>{float(sig.get('confidence', 0) or 0):.0f}%</b>\n"
                                        f"PnL: <b>{pnl:+.2f}%</b>\n"
                                        f"⚠️ {p.get('emergency_error','Posisi Binance belum terverifikasi')[:180]}\n"
                                        f"➡️ Jalankan <code>/ok {s}</code>"
                                    )
                                else:
                                    lines.append(
                                        f"\n{em} <b>{s}</b> — AKTIF\n"
                                        f"Entry: <code>{p['entry']:.6g}</code> | Harga: <code>{pr:.6g}</code>\n"
                                        f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{cur_sl:.6g}</code>{trail_note} | Confidence: <b>{float(sig.get('confidence', 0) or 0):.0f}%</b>\n"
                                        f"PnL: <b>{pnl:+.2f}%</b> | 🕐 Entry jam {entry_clock}"
                                    )
                        tg_send(chat_id,"\n".join(lines))
                elif text.startswith("/ok") or (not text.startswith("/") and text.startswith("ok")):
                    parts = text.split()
                    target = parts[1].upper() if len(parts) > 1 else None
                    if not target:
                        tg_send(chat_id, "❌ Gunakan <code>/ok SYMBOL</code>, contoh <code>/ok HOMEUSDT</code>.")
                        continue
                    def _run_ok(cid, sym):
                        try:
                            with positions_lock:
                                pos = positions.get(sym)
                            if not pos:
                                tg_send(cid, f"ℹ️ <b>{sym}</b> tidak ada di state /trade. Tidak ada yang direkonsiliasi.")
                                return
                            tg_send(cid, f"🔄 <b>RECONCILING {sym}...</b>\nMemeriksa posisi Binance + TP/SL + pending protection/trail.")
                            _binance_sync_time(force=True)
                            real = get_real_position(sym)
                            live_qty = abs(float(real.get("positionAmt", 0))) if real else 0.0
                            if live_qty <= 0:
                                # Position closed: clean orphan algo orders, then remove local state.
                                _cleanup_algo_orders_verified(sym)
                                _clear_pending_trail(sym)
                                _clear_pending_protection(sym)
                                _clear_pending_cleanup(sym)
                                close_position(sym, "strategy", close_price=get_price(sym) or pos.get("entry"))
                                tg_send(cid, f"✅ <b>{sym} RECONCILED</b>\nPosition Binance = 0.\nOrphan TP/SL sudah dibersihkan.\nStatus: CLOSED.")
                                return
                            with positions_lock:
                                if sym in positions:
                                    positions[sym]["status"] = "active"
                                    positions[sym]["quantity"] = live_qty
                                    positions[sym]["exchange_synced_at"] = time.time()
                                    positions[sym]["emergency_reason"] = None
                            # If protection is missing/unknown, queue/restore it.
                            sig = pos["signal"]; buy = sig["decision"] == "BUY"
                            tp = sig.get("tp"); sl = pos.get("current_sl", sig.get("sl"))
                            pending = _get_pending_trail(sym)
                            if pending:
                                tp = pending.get("tp") or tp
                                sl = pending.get("sl") or sl
                            # Reconcile atomically: remove stale protective algo orders before
                            # placing the verified current TP/SL pair, preventing duplicates.
                            cancel_all_algo_orders(sym)
                            t, sl_order = place_tp_sl(sym, buy, tp, sl, live_qty)
                            with positions_lock:
                                if sym in positions:
                                    positions[sym].update({"tp_order_id": t["algoId"], "sl_order_id": sl_order["algoId"], "protection_state": "VERIFIED", "exchange_synced_at": time.time()})
                            _clear_pending_protection(sym)
                            _clear_pending_trail(sym)
                            _clear_pending_cleanup(sym)
                            tg_send(cid, f"✅ <b>{sym} RECONCILED</b>\nPosition Binance masih terbuka: <code>{live_qty:.8g}</code>\nTP/SL terpasang ulang dan state kembali ACTIVE.")
                        except Exception as e:
                            with positions_lock:
                                if sym in positions:
                                    positions[sym]["status"] = "EMERGENCY"
                                    positions[sym]["emergency_error"] = str(e)[:300]
                            _queue_pending_cleanup(sym, "/ok gagal — retry manual", e)
                            tg_send(cid, f"🚨 <b>{sym} RECONCILE GAGAL</b>\n<code>{str(e)[:350]}</code>\nPosisi tetap dipertahankan di /trade. Coba <code>/ok {sym}</code> lagi setelah Binance/API normal.")
                    threading.Thread(target=_run_ok, args=(chat_id, target), daemon=True).start()
                elif text.startswith("/timeout") or (not text.startswith("/") and text.startswith("timeout")):
                    parts = text.split()
                    target_sym = parts[1].upper() if len(parts) > 1 else None
                    if target_sym:
                        tg_send(chat_id, f"⏳ <b>TIMEOUT REQUESTED</b> — {target_sym}\n"
                                        "Membatalkan semua order Binance dan menutup posisi bila ada…")
                        threading.Thread(target=_verified_timeout_symbol, args=(target_sym, chat_id), daemon=True).start()
                    else:
                        tg_send(chat_id, "⏳ <b>TIMEOUT GLOBAL REQUESTED</b>\n"
                                        "Membatalkan SEMUA order Binance dan menutup SEMUA posisi…")
                        threading.Thread(target=_verified_timeout_all, args=(chat_id,), daemon=True).start()
                elif text.startswith("/mode"):
                    # /mode on snapshots Binance balance exactly once per OFF→ON transition.
                    parts = text.split()
                    arg = parts[1].lower() if len(parts) > 1 else None
                    with positions_lock:
                        n_open = len(positions)
                    if arg is None:
                        status = "🔴 REAL TRADE" if REAL_TRADE_ENABLED else "🧪 SIMULASI"
                        anchor = (f"${real_balance_snapshot:.4f}" if REAL_TRADE_ENABLED and real_balance_snapshot is not None else f"${STARTING_BALANCE:.4f}")
                        tg_send(chat_id, f"⚙️ <b>Mode:</b> {status}\nBalance anchor: <b>{anchor}</b>\n\nGunakan <code>/mode on</code> atau <code>/mode off</code>.")
                    elif arg == "on":
                        if REAL_TRADE_ENABLED:
                            tg_send(chat_id, "🔴 Mode real sudah aktif. Tidak fetch balance ulang.")
                            continue
                        key, secret = _read_binance_credentials()
                        BINANCE_KEYS_PRESENT = bool(key and secret)
                        if BINANCE_KEYS_PRESENT:
                            BINANCE_API_KEY, BINANCE_API_SECRET = key, secret
                        if not BINANCE_KEYS_PRESENT:
                            tg_send(chat_id, "❌ Tidak bisa aktifkan mode real — API key/secret belum diset.")
                            continue
                        # One signed balance call on the transition only. If it fails, remain OFF.
                        try:
                            avail, total = get_real_balance()
                            if total is None:
                                raise RuntimeError("saldo Binance tidak tersedia")
                            with real_balance_lock:
                                real_balance_snapshot = float(total)
                                real_balance_snapshot_at = time.time()
                            with stat_lock:
                                stats["balance"] = float(total)
                            with autostop_lock:
                                peak_real_balance = float(total)
                            REAL_TRADE_ENABLED = True
                            if autostop_thread is None or not autostop_thread.is_alive():
                                autostop_thread = threading.Thread(target=autostop_loop, args=(chat_id,), daemon=True)
                                autostop_thread.start()
                            extra = (f"\n\nℹ️ {n_open} posisi simulasi tetap dipantau sebagai simulasi." if n_open else "")
                            tg_send(chat_id, f"🔴 <b>Mode REAL TRADE diaktifkan.</b>\nBalance Binance snapshot: <b>${float(total):.4f}</b>\nSnapshot dibuat sekali pada transisi ini.{extra}")
                        except Exception as e:
                            REAL_TRADE_ENABLED = False
                            tg_send(chat_id, f"❌ <b>/mode on gagal.</b> Balance Binance tidak berhasil diambil.\n<code>{str(e)[:220]}</code>")
                    elif arg == "off":
                        if not REAL_TRADE_ENABLED:
                            with stat_lock:
                                stats["balance"] = STARTING_BALANCE
                            tg_send(chat_id, "🧪 Mode simulasi sudah aktif. Balance anchor: $10.0000.")
                        else:
                            REAL_TRADE_ENABLED = False
                            with stat_lock:
                                stats["balance"] = STARTING_BALANCE
                            # Do not clear active REAL positions; their execution mode is immutable.
                            extra = (f"\n\nℹ️ {n_open} posisi real tetap dipantau/ditutup via Binance." if n_open else "")
                            tg_send(chat_id, f"🧪 <b>Mode SIMULASI diaktifkan.</b>\nBalance anchor dikembalikan ke <b>${STARTING_BALANCE:.2f}</b>.{extra}")
                    else:
                        tg_send(chat_id, "❓ Pakai <code>/mode</code>, <code>/mode on</code>, atau <code>/mode off</code>.")
                elif text.startswith("/max"):
                    parts = text.split()
                    # ── /max (tampilkan info) ──────────────────────────────
                    if len(parts) == 1:
                        # Estimasi beban API saat ini
                        scan_weight_per_min  = 836   # ~100 kline req × weight5 / ~34s scan
                        price_weight_per_min = 12    # 1 batch ticker/price tiap 10 detik
                        total_weight         = scan_weight_per_min + price_weight_per_min
                        binance_limit        = 2400
                        usage_pct            = total_weight / binance_limit * 100
                        headroom_pct         = 100 - usage_pct
                        threads_now          = 4 + MAX_POSITIONS * 2   # bot+cache+flask+scan + monitor+wait_entry

                        # Batas aman: scan mendominasi, bukan jumlah posisi
                        # Posisi hanya menambah ~0.02 weight/mnt per posisi (SL check jarang)
                        # Batas praktis sebelum scan overload:
                        #   sisa headroom = 1552 weight/mnt, scan = 836/mnt
                        #   bisa ~2 scan paralel tapi kode hanya 1 scan sekaligus → aman tak terbatas dari sisi API
                        # Batas rekomendasi dari sisi KUALITAS SINYAL: ≤ 20
                        tg_send(chat_id,
                            f"⚙️ <b>Max Posisi</b>\n\n"
                            f"Saat ini     : <b>{MAX_POSITIONS} posisi</b>\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📡 <b>Info Beban API (Binance Futures)</b>\n\n"
                            f"Limit Binance    : <b>2.400 weight/mnt</b>\n"
                            f"Scan 50 koin     : ~{scan_weight_per_min} weight/mnt\n"
                            f"Price cache      : ~{price_weight_per_min} weight/mnt (1 batch/10 dtk)\n"
                            f"Total dipakai    : ~{total_weight} weight/mnt "
                            f"(<b>{usage_pct:.0f}%</b> dari limit)\n"
                            f"Headroom tersisa : ~{headroom_pct:.0f}%\n\n"
                            f"⚠️ <b>Penting:</b> MAX_POSITIONS <b>tidak</b> menambah beban\n"
                            f"API secara signifikan. Beban didominasi scan koin,\n"
                            f"bukan jumlah posisi yang dipantau.\n"
                            f"Monitor thread baca harga dari cache lokal — bukan API.\n\n"
                            f"🧵 Thread aktif est. : ~{threads_now}\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 <b>Batas yang Disarankan</b>\n\n"
                            f"API weight  : ✅ aman hingga 50+ posisi\n"
                            f"Thread      : ✅ aman hingga 50+ posisi\n"
                            f"Kualitas sinyal: ⚠️  disarankan ≤ 20\n"
                            f"  (lebih dari itu, scanner makin susah\n"
                            f"  temukan setup berkualitas karena koin\n"
                            f"  terbaik sudah terpakai)\n\n"
                            f"<b>Ubah: /max 5 | /max 10 | /max 15 | /max 20</b>")
                    # ── /max N (ubah nilai) ────────────────────────────────
                    elif len(parts) == 2:
                        try:
                            n = int(parts[1])
                            if n < 1 or n > 50:
                                tg_send(chat_id,
                                    f"❌ Nilai harus antara 1–50.\n"
                                    f"Contoh: /max 10")
                            else:
                                old = MAX_POSITIONS
                                MAX_POSITIONS = n
                                with positions_lock:
                                    n_active = len(positions)
                                note = ""
                                if n < n_active:
                                    note = (f"\n\n⚠️ Ada {n_active} posisi aktif saat ini.\n"
                                            f"Posisi yang sudah buka tetap dipantau.\n"
                                            f"Scan baru berhenti sampai posisi tutup ke ≤ {n}.")
                                tg_send(chat_id,
                                    f"✅ Max posisi diubah: <b>{old} → {MAX_POSITIONS}</b>{note}")
                        except ValueError:
                            tg_send(chat_id,"❌ Format salah. Contoh: /max 10")
                    else:
                        tg_send(chat_id,"❌ Format: /max  atau  /max 10")

                elif text.startswith("/leverage"):
                    parts = text.split()
                    if len(parts) == 1:
                        tg_send(chat_id,
                            f"⚙️ <b>Leverage</b>\n\nSaat ini: <b>{LEVERAGE}x</b>\n\n"
                            f"<b>Ubah: /leverage 5</b>")
                    elif len(parts) == 2:
                        try:
                            n = int(parts[1])
                            if n < 1 or n > 125:
                                tg_send(chat_id, "❌ Nilai harus antara 1–125.\nContoh: /leverage 5")
                            else:
                                old = LEVERAGE
                                LEVERAGE = n
                                tg_send(chat_id, f"✅ Leverage diubah: <b>{old}x → {LEVERAGE}x</b>")
                        except ValueError:
                            tg_send(chat_id, "❌ Format salah. Contoh: /leverage 5")
                    else:
                        tg_send(chat_id, "❌ Format: /leverage  atau  /leverage 5")

                elif text.startswith("/margin"):
                    parts = text.split()
                    if len(parts) == 1:
                        tg_send(chat_id,
                            f"⚙️ <b>Margin Awal</b>\n\nSaat ini: <b>${MARGIN_USD:.2f}</b>\n\n"
                            f"Kalau margin ini terlalu kecil untuk suatu koin (kena batas minimum\n"
                            f"quantity/notional Binance), bot otomatis menaikkan SEDIKIT (maks 1.5x)\n"
                            f"khusus untuk trade itu — bukan mengubah setting ini secara permanen.\n\n"
                            f"<b>Ubah: /margin 5</b>")
                    elif len(parts) == 2:
                        try:
                            n = float(parts[1])
                            if n <= 0 or n > 10000:
                                tg_send(chat_id, "❌ Nilai harus antara 0–10000.\nContoh: /margin 5")
                            else:
                                old = MARGIN_USD
                                MARGIN_USD = n
                                tg_send(chat_id, f"✅ Margin awal diubah: <b>${old:.2f} → ${MARGIN_USD:.2f}</b>")
                        except ValueError:
                            tg_send(chat_id, "❌ Format salah. Contoh: /margin 5")
                    else:
                        tg_send(chat_id, "❌ Format: /margin  atau  /margin 5")

                elif text.startswith("/autostop"):
                    parts = text.split()
                    if len(parts) == 1:
                        with autostop_lock:
                            peak_txt = f"${peak_real_balance:.2f}" if peak_real_balance else "belum ada data"
                        tg_send(chat_id,
                            f"⚙️ <b>Auto-Stop Drawdown</b>\n\nThreshold: <b>{AUTOSTOP_PCT}%</b>\n"
                            f"Peak saldo tercatat: {peak_txt}\n\n"
                            f"Kalau saldo turun segini persen dari peak, scan sinyal baru otomatis\n"
                            f"berhenti (posisi aktif tetap dipantau). Jalankan lagi manual dengan /auto.\n\n"
                            f"<b>Ubah: /autostop 3</b>")
                    elif len(parts) == 2:
                        try:
                            n = float(parts[1])
                            if n <= 0 or n > 100:
                                tg_send(chat_id, "❌ Nilai harus antara 0–100.\nContoh: /autostop 3")
                            else:
                                old = AUTOSTOP_PCT
                                AUTOSTOP_PCT = n
                                tg_send(chat_id, f"✅ Threshold auto-stop diubah: <b>{old}% → {AUTOSTOP_PCT}%</b>")
                        except ValueError:
                            tg_send(chat_id, "❌ Format salah. Contoh: /autostop 3")
                    else:
                        tg_send(chat_id, "❌ Format: /autostop  atau  /autostop 3")
                else:
                    tg_send(chat_id,"❓ Tidak dikenal. /start")

            time.sleep(0.2)
        except TelegramPollingConflict as e:
            log.error(f"[TG POLLING CONFLICT] {e}")
            _telegram_watchdog_alert(
                active_chat_id,
                "🚨 <b>Telegram polling conflict</b>\n\n"
                "Bot masih hidup, tetapi <code>getUpdates</code> bentrok. "
                "Pastikan hanya 1 instance bot memakai TELEGRAM_TOKEN ini."
            )
            time.sleep(min(max(poll_backoff, 5), TELEGRAM_ERROR_BACKOFF_MAX))
            poll_backoff = min(max(poll_backoff * 2, 5), TELEGRAM_ERROR_BACKOFF_MAX)
        except Exception as e:
            log.error(f"[TG/BOT LOOP] {e}", exc_info=True)
            time.sleep(min(max(poll_backoff, 2), TELEGRAM_ERROR_BACKOFF_MAX))
            poll_backoff = min(max(poll_backoff * 2, 2), TELEGRAM_ERROR_BACKOFF_MAX)


if __name__=="__main__":
    # Flask dijalankan di thread sendiri PALING AWAL supaya port langsung
    # bind & terdeteksi Render, tidak menunggu inisialisasi bot/WS selesai.
    threading.Thread(target=run_flask, daemon=True).start()
    ws_feed.start()
    threading.Thread(target=_price_cache_loop, daemon=True).start()
    threading.Thread(target=_binance_recovery_loop, daemon=True).start()
    threading.Thread(target=_render_keepalive_loop, daemon=True).start()
    threading.Thread(target=bot_loop, daemon=True).start()

    log.info(f"[ENGINE] {MAIN_ENGINE_VERSION} starting")
    if _STRATEGY_LOAD_ERROR and ALLOWED_USER_ID:
        tg_send(ALLOWED_USER_ID,
            f"🚨 <b>strategy_logic.py BERMASALAH</b>\n\n"
            f"{_STRATEGY_LOAD_ERROR}\n\n"
            f"Bot jalan pakai fallback AMAN (tidak akan cari/entry sinyal baru)\n"
            f"sampai file yang benar di-upload lewat /ganti.")

    if REAL_TRADE_ENABLED and ALLOWED_USER_ID:
        ip = get_public_ip()
        tg_send(ALLOWED_USER_ID,
            f"🔴 <b>REAL TRADE MODE</b>\n\n"
            f"IP Render saat ini: <code>{ip}</code>\n\n"
            f"Whitelist IP ini di Binance API Management dulu kalau belum,\n"
            f"lalu kirim /auto untuk mulai. Bot TIDAK akan mulai cari sinyal\n"
            f"sampai kamu kirim /auto secara manual.")
        threading.Thread(target=autostop_loop, args=(ALLOWED_USER_ID,), daemon=True).start()

    # Semua thread di atas daemon=True — main thread harus tetap hidup,
    # kalau tidak proses langsung exit begitu baris ini selesai.
    while True:
        time.sleep(3600)
