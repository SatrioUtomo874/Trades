#!/usr/bin/env python3
"""
main.py V72 — OPERATIONAL BODY / FINAL EXECUTION INFRASTRUCTURE.

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
import os, time, logging, threading, signal, uuid, inspect
from collections import deque
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

import requests, pandas as pd, numpy as np, urllib3, json, html, base64
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
BINANCE_REQUEST_INTERVAL = 0.55
# Setelah cooldown/ban Binance selesai, tunggu tambahan 60 detik sebelum request pertama.
BINANCE_POST_COOLDOWN_GRACE = 60.0
MAX_MARGIN_MULTIPLIER = 1.50  # HARD SAFETY CAP relative to configured MARGIN_USD
# Safety governor berbasis header usage; berhenti sebelum mendekati limit 1 menit.
BINANCE_WEIGHT_SOFT_LIMIT = 1400
BINANCE_WEIGHT_HARD_LIMIT = 1900
BINANCE_CRITICAL_HARD_LIMIT = 2300
BINANCE_EXECUTION_RESERVE = 350
BINANCE_WEIGHT_STALE_AFTER_SEC = 65.0
_binance_request_lock = threading.Lock()
_binance_priority_local = threading.local()
_binance_last_request_at = 0.0
_binance_weight_1m = None
_binance_weight_seen_at = 0.0
MAX_POSITIONS       = 20   # runtime via /max — jangan pindah ke strategy_logic
MONITOR_INTERVAL    = 15 * 60
STRATEGY_MANAGE_INTERVAL = 60
BRAIN_CONFIDENCE_DISPLAY_FALLBACK = "brain-owned"
WIB = timezone(timedelta(hours=7))   # format jam entry di /trade
MAIN_ENGINE_VERSION = "MAIN-BODY-V91-BYBIT-SCANNER-BINANCE-EXECUTION-SEP"

# ── SCAN MARKET-DATA CACHE ─────────────────────────────────────────────
# Scanner tidak boleh mengambil candle yang sama berulang-ulang. Cache ini
# hanya dipakai oleh pipeline scan; execution/position monitoring tetap memakai
# get_klines() normal sehingga tidak mengubah freshness data posisi.
def _env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value

SCAN_MAX_DURATION_SEC = _env_int("SCAN_MAX_DURATION_SEC", 180, minimum=60, maximum=3600)
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

# Scanner lifecycle is explicit: exactly one coordinator and at most one heavy scan task.
_SCAN_STATE_LOCK = threading.RLock()
_SCAN_STATE = {
    "enabled": False,
    "coordinator_alive": False,
    "cycle_running": False,
    "cycle_count": 0,
    "last_started_at": None,
    "last_finished_at": None,
    "last_success_at": None,
    "last_error": None,
    "last_result_count": 0,
    "last_symbols_processed": 0,
    "last_candidate_count": 0,
    "last_eligible_count": 0,
    "last_ban_count": 0,
    "last_low_confidence_count": 0,
    "last_rejection_reasons": {},
    "consecutive_no_signal_cycles": 0,
    "coordinator_heartbeat_at": None,
    "last_cycle_id": None,
    "last_data_source": "BYBIT_REST",
}
_SCAN_COORDINATOR_THREAD = None
_SCAN_CYCLE_THREAD = None
_SCAN_CYCLE_EVENT = threading.Event()
_SCAN_WAKE = threading.Event()

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

# ==================== BRAIN / STRATEGY INTERFACE ====================
# main.py never contains strategy reasoning. It exposes a small stable adapter
# to strategy_logic.py; missing optional research APIs never break execution.
try:
    import strategy_logic as _brain
    _STRATEGY_LOAD_ERROR = None
    log.info("[BRAIN] strategy_logic.py imported; runtime contract validation pending")
except Exception as e:
    _brain = None
    _STRATEGY_LOAD_ERROR = str(e)
    log.error(f"[BRAIN] load gagal: {e}; new entries disabled, existing positions remain managed.")

def _brain_fn(name):
    return getattr(_brain, name, None) if _brain is not None else None

def full_analyze(df_h1, df_m15, df_d1=None, symbol=None, **kwargs):
    """Stable body→brain adapter. The body never implements strategy logic."""
    fn = _brain_fn("full_analyze")
    if not callable(fn):
        return None
    try:
        return fn(df_h1, df_m15, df_d1=df_d1, symbol=symbol, **kwargs)
    except TypeError as exc:
        try:
            params = inspect.signature(fn).parameters
            legacy_ok = "df_btc_h1" not in params and "trade_history" not in params
        except Exception:
            legacy_ok = False
        if legacy_ok:
            return fn(df_h1, df_m15, df_d1, symbol=symbol)
        log.exception(f"[BRAIN] full_analyze internal TypeError {symbol}: {exc}")
        return None

def manage_position(state, df_m15, df_h1=None, df_d1=None, symbol=None, **kwargs):
    """Stable body→brain management adapter; no execution is performed here."""
    fn = _brain_fn("manage_position")
    if not callable(fn):
        return None
    try:
        return fn(state, df_m15, df_h1=df_h1, df_d1=df_d1, symbol=symbol, **kwargs)
    except TypeError as exc:
        try:
            params = inspect.signature(fn).parameters
            legacy_ok = "symbol" not in params
        except Exception:
            legacy_ok = False
        if legacy_ok:
            return fn(state, df_m15, df_h1, df_d1, symbol=symbol)
        log.exception(f"[BRAIN] manage_position internal TypeError {symbol}: {exc}")
        return None

def _call_brain_event(names, row, *, fallback_symbol=False):
    for name in names:
        fn = _brain_fn(name)
        if not callable(fn):
            continue
        try:
            return fn(row)
        except TypeError:
            if fallback_symbol:
                try:
                    return fn(row.get("symbol"), row)
                except Exception:
                    pass
        except Exception as exc:
            log.warning(f"[BRAIN] {name} gagal: {exc}")
        return None
    return None

def _brain_on_stats_snapshot(snapshot):
    fn=_brain_fn("evaluate_stats_decision")
    if not callable(fn): return None
    try:
        return fn(dict(snapshot or {}), source="main_stats")
    except Exception as exc:
        log.warning(f"[BRAIN] evaluate_stats_decision gagal: {exc}")
        return None

def _brain_on_candidate(row):
    for name in ("ingest_live_candidate", "record_candidate_observation"):
        fn=_brain_fn(name)
        if not callable(fn): continue
        try:
            if name=="ingest_live_candidate":
                return fn(dict(row or {}), source="bybit_market")
            return fn(dict(row or {}), source="bybit_market")
        except TypeError:
            try: return fn(dict(row or {}))
            except Exception as exc: log.warning(f"[BRAIN] {name} gagal: {exc}")
        except Exception as exc:
            log.warning(f"[BRAIN] {name} gagal: {exc}")
        break
    return None

def _brain_on_trade(row):
    """Canonical outcome bridge. Supports both modern ingest_* and legacy record_* APIs."""
    fn = _brain_fn("ingest_live_outcome")
    if callable(fn):
        try:
            outcome = row.get("outcome") if isinstance(row, dict) and isinstance(row.get("outcome"), dict) else (dict(row) if isinstance(row, dict) else {"result": row})
            return fn(row, outcome, source="binance_trade")
        except Exception as exc:
            log.warning(f"[BRAIN] ingest_live_outcome gagal: {exc}")
    fn = _brain_fn("record_trade_outcome")
    if callable(fn):
        try:
            outcome = row.get("outcome") if isinstance(row, dict) else {"result": row}
            return fn(row, outcome, source="binance_trade")
        except Exception as exc:
            log.warning(f"[BRAIN] record_trade_outcome gagal: {exc}")
    return None

def _record_brain_scan_summary(summary):
    """Canonical scan→brain frequency boundary. Legacy brains simply ignore it."""
    fn = _brain_fn("record_scan_summary")
    if not callable(fn):
        return None
    try:
        return fn(dict(summary or {}), source="main_scanner")
    except TypeError:
        return fn(dict(summary or {}))
    except Exception as exc:
        log.warning(f"[BRAIN] record_scan_summary gagal: {exc}")
        return None

def _brain_get_experience_count():
    for name in ("get_experience_count", "get_learning_model_info", "get_full_cognitive_status", "get_cognitive_status"):
        fn = _brain_fn(name)
        if not callable(fn):
            continue
        try:
            value = fn()
            if isinstance(value, dict):
                state = value.get("state") if isinstance(value.get("state"), dict) else value
                for key in ("experience_samples", "samples", "experience_count", "outcomes", "outcome_count", "labeled_outcomes"):
                    if key in state:
                        return int(state.get(key) or 0)
        except Exception:
            continue
    return 0

def _strategy_set_ml_model(model):
    fn = _brain_fn("set_learning_model")
    if callable(fn):
        try:
            return fn(model)
        except Exception as exc:
            log.debug(f"[BRAIN] set model ignored: {exc}")
    return None

def _brain_get_champion():
    fn = _brain_fn("get_learning_model_info")
    if callable(fn):
        try:
            info = fn()
            if isinstance(info, dict):
                return info.get("champion") or info.get("model") or {}
        except Exception:
            pass
    return {}

def _brain_full_command(action, chat_id=None):
    fn = _brain_fn("full_command")
    if callable(fn):
        try:
            return fn(action)
        except TypeError:
            try:
                return fn(action, chat_id)
            except Exception as exc:
                log.warning(f"[BRAIN] full command gagal: {exc}")
        except Exception as exc:
            log.warning(f"[BRAIN] full command gagal: {exc}")
    return None

def _brain_full_status():
    fn = _brain_fn("get_full_cognitive_status") or _brain_fn("get_cognitive_status")
    if callable(fn):
        try:
            return fn()
        except Exception as exc:
            return {"error": str(exc)}
    return {"error": "brain status unavailable"}

def _get_active_confidence_threshold():
    """Read-only compatibility display. Strategy policy remains brain-owned.
    If the brain does not expose a threshold, return None rather than inventing one.
    """
    for name in ("get_active_confidence_threshold", "suggest_confidence_threshold"):
        fn = _brain_fn(name)
        if not callable(fn):
            continue
        try:
            value = fn()
            if isinstance(value, dict):
                value = value.get("threshold") or value.get("active_threshold")
            if value is not None:
                return float(value)
        except Exception:
            pass
    return None

FULL_MODE=False
FULL_THREAD=None
FULL_MANUAL_THRESHOLD_SAVED=None
FULL_STOP=threading.Event()
FULL_WAKE=threading.Event()

def _full_on():
    global FULL_MODE, FULL_MANUAL_THRESHOLD_SAVED
    FULL_MODE=True; FULL_MANUAL_THRESHOLD_SAVED=None
    return _brain_full_command("on")

def _full_off():
    global FULL_MODE, FULL_MANUAL_THRESHOLD_SAVED
    FULL_MODE=False; FULL_MANUAL_THRESHOLD_SAVED=None
    return _brain_full_command("off")

def _full_reset_internal():
    global FULL_MODE, FULL_MANUAL_THRESHOLD_SAVED
    FULL_MODE=False; FULL_STOP.set(); FULL_WAKE.clear(); FULL_MANUAL_THRESHOLD_SAVED=None
    return _brain_full_command("reset")

def _full_strategy_command(action, chat_id=None):
    action=str(action or "status").strip().lower()
    # Human-readable brain command is the canonical Telegram surface. Never send raw status dicts.
    if action=="on": return _full_on()
    if action=="off": return _full_off()
    if action=="reset": return _full_reset_internal()
    return _brain_full_command("status")

def _full_status_text():
    return _brain_full_command("status")

def _full_controller_state():
    return {"mode": bool(FULL_MODE), "threshold": _get_active_confidence_threshold(), "brain": _brain_full_status()}

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
        self._throttle  = 900         # detik

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
BAN_DURATION_TRADE_CLOSED = 25.0   # setelah trade BENAR-BENAR closed

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

# Execution infrastructure invariants
EXECUTION_ENGINE_VERSION = "MAIN-BODY-V91-BYBIT-SCANNER-BINANCE-EXECUTION-SEP"
RUNTIME_SCHEMA_VERSION = "runtime_v1"
EVENT_SCHEMA_VERSION = "event_v1"
CHECKPOINT_SCHEMA_VERSION = "checkpoint_v1"
BRAIN_INTERFACE_VERSION = "brain_v1"
BRAIN_COMPATIBLE_LEGACY_VERSIONS = {"brain_v1", "V35_ADAPTIVE_BRAIN", "V34_CONTINUAL_COGNITIVE_AUDITED", "V35_CONTINUAL_ADAPTIVE_BRAIN_AUDITED", "V36_MULTI_AUDIT_EVOLUTION_BRAIN", "V37_FINAL_REGRESSION_HARDENED_BRAIN", "V38_EVENT_CONTRACT_HARDENED_BRAIN", "V40_FULL_BRAIN_REBUILT", "V52_OPERATIONAL_BRAIN", "V60_STATS_DRIVEN_FREQUENCY_BRAIN", "V71_UNIFIED_FULL_BRAIN", "STRATEGY-BRAIN-V100-BYBIT-WS-STATS-EVOLUTION"}
MAX_HEAVY_WORKERS = 5
HEAVY_WORKER_SEMAPHORE = threading.BoundedSemaphore(MAX_HEAVY_WORKERS)
_HEAVY_WORKER_LOCK = threading.RLock()
_HEAVY_WORKER_REGISTRY = {}
STOP_NEW_ENTRIES = False
CIRCUIT_BREAKER_OPEN = False
RUNTIME_STATE = "BOOTING"
RUNTIME_STATE_LOCK = threading.RLock()
SHUTDOWN_EVENT = threading.Event()
EXECUTION_MUTATION_LOCK = threading.RLock()
EVENT_SEQUENCE = 0
EVENT_SEQUENCE_LOCK = threading.Lock()
RUN_ID = uuid.uuid4().hex[:16]
RUNTIME_STATE_DIR = Path(os.getenv("RUNTIME_STATE_DIR", str(Path(__file__).resolve().parent / "runtime_state")))
RUNTIME_CHECKPOINT_DIR = RUNTIME_STATE_DIR / "checkpoints"
RUNTIME_CHECKPOINT_MANIFEST = RUNTIME_CHECKPOINT_DIR / "latest.json"
RUNTIME_STATE_LOCK = threading.RLock()

def _runtime_snapshot(include_brain=True):
    with stat_lock:
        stat_copy = {k: v for k, v in stats.items() if k != "pnl_history"}
        stat_copy["pnl_history"] = list(stats.get("pnl_history") or [])
    with positions_lock:
        pos_copy = {}
        for sym, pos in positions.items():
            # Current exchange is authoritative for REAL positions; checkpoint only
            # records metadata for them. SIMULATION positions can be restored locally.
            if _position_is_real(pos):
                pos_copy[sym] = {
                    "execution_mode": "REAL",
                    "position_id": pos.get("position_id"),
                    "trade_uid": pos.get("trade_uid"),
                    "strategy_version": (pos.get("signal") or {}).get("strategy_version"),
                }
            else:
                pos_copy[sym] = dict(pos)
    with _last_scanned_lock:
        scanned = list(last_scanned_coins)
        scanned_at = last_scanned_at
    with trade_history_lock:
        trade_copy = [dict(x) for x in trade_history]
    with ban_lock:
        ban_copy = dict(banned_coins)
    with low_conf_history_lock:
        low_conf_copy = [dict(x) for x in low_conf_history]
    with trail_events_lock:
        trail_copy = [dict(x) for x in trail_events]
    brain_state = None
    if include_brain:
        for name in ("export_checkpoint_state", "get_brain_state"):
            fn = _brain_fn(name)
            if callable(fn):
                try:
                    brain_state = fn()
                    break
                except Exception as e:
                    log.debug(f"[CHECKPOINT] brain export ignored: {e}")
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "engine_version": EXECUTION_ENGINE_VERSION,
        "run_id": RUN_ID,
        "created_at": time.time(),
        "event_sequence": EVENT_SEQUENCE,
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "brain_interface_version": BRAIN_INTERFACE_VERSION,
        "runtime_state": RUNTIME_STATE,
        "runtime": {
            "auto_mode": bool(auto_mode),
            "real_trade_enabled": bool(REAL_TRADE_ENABLED),
            "max_positions": int(MAX_POSITIONS),
            "leverage": int(LEVERAGE),
            "margin_usd": float(MARGIN_USD),
            "autostop_pct": float(AUTOSTOP_PCT),
            "full_mode": bool(FULL_MODE),
        },
        "stats": stat_copy,
        "trade_history": trade_copy,
        "trail_events": trail_copy,
        "banned_coins": ban_copy,
        "low_confidence_history": low_conf_copy,
        "positions": pos_copy,
        "scan_universe": {"symbols": scanned, "updated_at": scanned_at},
        "brain_state": brain_state,
    }

def _write_atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)

def _github_get_json(path):
    if not GITHUB_TOKEN or not REPO_NAME:
        raise RuntimeError("GITHUB_TOKEN/REPO_NAME belum diset")
    url=f"https://api.github.com/repos/{REPO_NAME}/contents/{path}"
    r=requests.get(url, headers={"Authorization":f"token {GITHUB_TOKEN}","Accept":"application/vnd.github+json"}, timeout=15)
    if r.status_code>=400:
        raise RuntimeError(f"GitHub GET {path}: HTTP {r.status_code} {r.text[:200]}")
    body=r.json(); raw=base64.b64decode(body["content"]).decode("utf-8")
    return json.loads(raw), body.get("sha")

_CHECKPOINT_TRANSACTION_LOCK=threading.RLock()
_CHECKPOINT_LAST_GOOD=None

def _save_runtime_checkpoint(push_github=True):
    global _CHECKPOINT_LAST_GOOD, STOP_NEW_ENTRIES
    with _CHECKPOINT_TRANSACTION_LOCK:
        prev_stop=STOP_NEW_ENTRIES; STOP_NEW_ENTRIES=True
        try:
            cp=_runtime_snapshot(include_brain=True); cid=f"cp-{int(time.time())}-{RUN_ID}"; cp["checkpoint_id"]=cid
            canonical=json.dumps(cp,ensure_ascii=False,allow_nan=False,sort_keys=True,default=str).encode()
            cp["content_hash"]=hashlib.sha256(canonical).hexdigest()
            local=RUNTIME_CHECKPOINT_DIR/f"{cid}.json"; _write_atomic_json(local,cp); _verify_checkpoint_integrity(cp)
            prev=None
            if RUNTIME_CHECKPOINT_MANIFEST.exists():
                try: prev=json.loads(RUNTIME_CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
                except Exception: prev=None
            manifest={"checkpoint_id":cid,"path":local.name,"created_at":cp["created_at"],"content_hash":cp["content_hash"],"previous_known_good":prev}
            _write_atomic_json(RUNTIME_CHECKPOINT_MANIFEST,manifest)
            if push_github:
                remote=f"runtime_state/checkpoints/{cid}.json"; _commit_to_github(json.dumps(cp,ensure_ascii=False,allow_nan=False,indent=2,default=str),remote,f"Runtime checkpoint {cid}")
                _commit_to_github(json.dumps({**manifest,"path":remote},indent=2),"runtime_state/latest.json",f"Update latest checkpoint {cid}")
            _CHECKPOINT_LAST_GOOD=cp
            return cid,local
        finally:
            STOP_NEW_ENTRIES=prev_stop


def _load_previous_known_good_manifest():
    """Return the last known-good checkpoint manifest without mutating runtime state."""
    try:
        if not RUNTIME_CHECKPOINT_MANIFEST.exists():
            return None
        manifest = json.loads(RUNTIME_CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
        prev = manifest.get("previous_known_good") if isinstance(manifest, dict) else None
        return prev if isinstance(prev, dict) else None
    except Exception as exc:
        log.warning(f"[CHECKPOINT] previous manifest read failed: {exc}")
        return None

def _load_runtime_checkpoint(reference=None):
    if reference in ("previous","previous_known_good"):
        prev=_load_previous_known_good_manifest()
        if not prev: raise RuntimeError("previous known-good checkpoint tidak tersedia")
        path=prev.get("path")
        if path and RUNTIME_CHECKPOINT_MANIFEST.exists():
            local=RUNTIME_CHECKPOINT_DIR/path
            if local.exists(): return json.loads(local.read_text(encoding="utf-8"))
        remote_path=prev.get("path")
        if remote_path and GITHUB_TOKEN and REPO_NAME:
            data,_=_github_get_json(remote_path)
            return data
        raise RuntimeError("previous known-good checkpoint tidak dapat dimuat")
    if reference:
        try:
            with open(reference, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    if RUNTIME_CHECKPOINT_MANIFEST.exists():
        m=json.loads(RUNTIME_CHECKPOINT_MANIFEST.read_text(encoding="utf-8")); p=RUNTIME_CHECKPOINT_DIR/m["path"]
        return json.loads(p.read_text(encoding="utf-8"))
    # Prefer GitHub only when local latest is absent.
    latest,_=_github_get_json("runtime_state/latest.json")
    cp=latest.get("path") if isinstance(latest,dict) else None
    if not cp: raise RuntimeError("latest checkpoint tidak ditemukan")
    data,_=_github_get_json(cp)
    return data

def _verify_checkpoint_integrity(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint bukan object")
    schema = checkpoint.get("checkpoint_schema")
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"checkpoint schema tidak kompatibel: {schema}")
    expected = checkpoint.get("content_hash")
    if expected:
        probe = dict(checkpoint)
        probe.pop("content_hash", None)
        canonical = json.dumps(probe, ensure_ascii=False, allow_nan=False, sort_keys=True, default=str).encode("utf-8")
        actual = hashlib.sha256(canonical).hexdigest()
        if actual != expected:
            raise RuntimeError("checkpoint content hash mismatch / possible corruption")
    return True


def _migrate_checkpoint(checkpoint):
    schema=str(checkpoint.get("checkpoint_schema") or "") if isinstance(checkpoint,dict) else ""
    if schema==CHECKPOINT_SCHEMA_VERSION: return checkpoint
    raise RuntimeError(f"unsupported checkpoint schema: {schema}; current={CHECKPOINT_SCHEMA_VERSION}")

def _restore_runtime_checkpoint(checkpoint):
    checkpoint=_migrate_checkpoint(checkpoint)
    _verify_checkpoint_integrity(checkpoint)
    # Restore simulation statistics/config only. REAL exposure remains exchange-authoritative.
    runtime=checkpoint.get("runtime") or {}
    with stat_lock:
        restored=checkpoint.get("stats") or {}
        for k in ("tp","sl","trail","total","balance"):
            if k in restored: stats[k]=restored[k]
        stats["pnl_history"]=deque(restored.get("pnl_history") or [], maxlen=20)
    global MAX_POSITIONS, LEVERAGE, MARGIN_USD, AUTOSTOP_PCT
    MAX_POSITIONS=int(runtime.get("max_positions",MAX_POSITIONS))
    LEVERAGE=int(runtime.get("leverage",LEVERAGE))
    MARGIN_USD=float(runtime.get("margin_usd",MARGIN_USD))
    AUTOSTOP_PCT=float(runtime.get("autostop_pct",AUTOSTOP_PCT))
    with trade_history_lock:
        trade_history.clear()
        trade_history.extend([dict(x) for x in (checkpoint.get("trade_history") or [])])
    with trail_events_lock:
        trail_events.clear()
        trail_events.extend([dict(x) for x in (checkpoint.get("trail_events") or [])])
    with ban_lock:
        banned_coins.clear()
        banned_coins.update(checkpoint.get("banned_coins") or {})
    with low_conf_history_lock:
        low_conf_history.clear()
        low_conf_history.extend([dict(x) for x in (checkpoint.get("low_confidence_history") or [])])
    sim_positions={}
    for sym,pos in (checkpoint.get("positions") or {}).items():
        if str(pos.get("execution_mode") or "SIMULATION").upper()=="SIMULATION": sim_positions[sym]=pos
    with positions_lock:
        for sym in list(positions):
            if _position_is_real(positions[sym]): continue
            positions.pop(sym,None)
        positions.update(sim_positions)
    universe=(checkpoint.get("scan_universe") or {})
    with _last_scanned_lock:
        last_scanned_coins.clear()
        last_scanned_coins.extend(list(universe.get("symbols") or []))
        globals()["last_scanned_at"] = universe.get("updated_at")
    brain_state=checkpoint.get("brain_state")
    for name in ("import_checkpoint_state", "apply_brain_state"):
        fn=_brain_fn(name)
        if callable(fn) and brain_state is not None:
            try: fn(brain_state); break
            except Exception as e: log.warning(f"[CHECKPOINT] brain restore warning: {e}")
    # Exchange truth always wins after restore.
    try:
        if _has_real_recovery_work(): _resume_binance_and_flush_pending(active_chat_id)
    except Exception as e: log.warning(f"[CHECKPOINT] reconcile after restore warning: {e}")
    return checkpoint.get("checkpoint_id","unknown")

def _set_runtime_state_legacy(state, reason=""):
    global RUNTIME_STATE
    with RUNTIME_STATE_LOCK:
        old=RUNTIME_STATE; RUNTIME_STATE=state
    if old!=state:
        log.info(f"[STATE] {old} -> {state}" + (f" | {reason}" if reason else ""))

def _next_event_id():
    global EVENT_SEQUENCE
    with EVENT_SEQUENCE_LOCK:
        EVENT_SEQUENCE += 1
        return EVENT_SEQUENCE

def _emit_execution_event(event_type, entity_id=None, correlation_id=None, payload=None, persist=False):
    event={"event_id":_next_event_id(),"event_type":event_type,"event_version":EVENT_SCHEMA_VERSION,
           "sequence":EVENT_SEQUENCE,"timestamp":time.time(),"source":"main.py","run_id":RUN_ID,
           "correlation_id":correlation_id or RUN_ID,"entity_id":entity_id,"payload":payload or {}}
    if persist:
        try:
            p=Path(os.getenv("RUNTIME_EVENT_FILE", "runtime_events.jsonl"))
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a",encoding="utf-8") as f:
                f.write(json.dumps(event,ensure_ascii=False,separators=(",",":")) + "\n")
        except Exception as e:
            log.debug(f"[EVENT] persist gagal: {e}")
    return event

def _new_request_id(prefix="REQ"):
    return f"{prefix}-{uuid.uuid4().hex[:20]}"


# =============================================================================
# V47 FINAL BODY CONTRACT — RUNTIME / HEALTH / EXECUTION AUTHORITY
# =============================================================================
STATE_TRANSITIONS = {
    "BOOTING": {"READY", "DEGRADED", "RECOVERING", "EMERGENCY", "STOPPING"},
    "READY": {"DEGRADED", "RECOVERING", "EMERGENCY", "STOPPING"},
    "DEGRADED": {"READY", "RECOVERING", "EMERGENCY", "STOPPING"},
    "RECOVERING": {"READY", "DEGRADED", "EMERGENCY", "STOPPING"},
    "EMERGENCY": {"RECOVERING", "STOPPING"},
    "STOPPING": set(),
}
HEALTH_STATES = {"HEALTHY", "DEGRADED", "RECOVERING", "BLOCKED", "EMERGENCY", "UNKNOWN"}
HEALTH_COMPONENTS = (
    "binance_rest", "binance_websocket", "brain", "scanner", "execution",
    "protection", "persistence", "telegram", "research", "resource"
)
_health_lock = threading.RLock()
_component_health = {name: {"state":"UNKNOWN", "updated_at":time.time(), "detail":"not checked"} for name in HEALTH_COMPONENTS}
_runtime_state_history = deque(maxlen=100)


def _set_component_health(component, state, detail=""):
    if component not in _component_health:
        return
    state=str(state).upper()
    if state not in HEALTH_STATES:
        state="UNKNOWN"
    with _health_lock:
        _component_health[component]={"state":state,"updated_at":time.time(),"detail":str(detail)[:500]}


def _health_snapshot():
    with _health_lock:
        comps={k:dict(v) for k,v in _component_health.items()}
    states={v["state"] for v in comps.values()}
    if "EMERGENCY" in states: overall="EMERGENCY"
    elif "BLOCKED" in states: overall="BLOCKED"
    elif "RECOVERING" in states: overall="RECOVERING"
    elif "DEGRADED" in states or "UNKNOWN" in states: overall="DEGRADED"
    else: overall="HEALTHY"
    return {"overall":overall,"components":comps,"timestamp":time.time()}


def _set_runtime_state(state, reason=""):
    global RUNTIME_STATE
    state=str(state).upper()
    if state not in STATE_TRANSITIONS:
        raise ValueError(f"invalid runtime state: {state}")
    with RUNTIME_STATE_LOCK:
        old=RUNTIME_STATE
        if old!=state and state not in STATE_TRANSITIONS.get(old,set()):
            raise RuntimeError(f"invalid runtime transition {old} -> {state}")
        RUNTIME_STATE=state
        _runtime_state_history.append({"from":old,"to":state,"timestamp":time.time(),"reason":str(reason)[:500]})
    if old!=state:
        log.info(f"[STATE] {old} -> {state}" + (f" | {reason}" if reason else ""))


POSITION_LIFECYCLE = {"DISCOVERED","ENTRY_PENDING","OPENING","PROTECTION_PENDING","OPEN","MANAGED","CLOSING","CLOSED","RECONCILING","EMERGENCY"}
POSITION_TRANSITIONS = {
    "DISCOVERED":{"ENTRY_PENDING","EMERGENCY"},"ENTRY_PENDING":{"OPENING","CLOSED","RECONCILING","EMERGENCY"},
    "OPENING":{"PROTECTION_PENDING","CLOSING","RECONCILING","EMERGENCY"},"PROTECTION_PENDING":{"OPEN","MANAGED","CLOSING","RECONCILING","EMERGENCY"},
    "OPEN":{"MANAGED","CLOSING","RECONCILING","EMERGENCY"},"MANAGED":{"CLOSING","RECONCILING","EMERGENCY"},
    "CLOSING":{"CLOSED","RECONCILING","EMERGENCY"},"CLOSED":{"RECONCILING"},
    "RECONCILING":{"OPEN","MANAGED","CLOSED","EMERGENCY"},"EMERGENCY":{"RECONCILING","CLOSING","CLOSED"},
}

def _position_lifecycle(pos):
    if not isinstance(pos,dict): return "EMERGENCY"
    lc=str(pos.get("lifecycle") or "").upper()
    if lc in POSITION_LIFECYCLE: return lc
    st=str(pos.get("status") or "").lower()
    if st=="pending": return "ENTRY_PENDING"
    if st=="active": return "MANAGED" if pos.get("current_sl") is not None else "OPEN"
    if st=="EMERGENCY": return "EMERGENCY"
    return "DISCOVERED"

def _force_position_emergency(sym, reason):
    with positions_lock:
        pos=positions.get(sym)
        if pos is None: return
        old=_position_lifecycle(pos)
        pos["lifecycle"]="EMERGENCY"; pos["status"]="EMERGENCY"; pos["emergency_error"]=str(reason)[:400]
        pid=pos.get("position_id")
    _emit_execution_event("POSITION_LIFECYCLE",entity_id=pid or sym,correlation_id=pid or RUN_ID,payload={"symbol":sym,"from":old,"to":"EMERGENCY","reason":str(reason)[:300]},persist=True)

def _transition_position_lifecycle(sym,new_state,reason="",expected_state=None):
    new_state=str(new_state).upper()
    with positions_lock:
        pos=positions.get(sym)
        if pos is None: raise KeyError(f"position not found: {sym}")
        old=_position_lifecycle(pos)
        if expected_state and old!=str(expected_state).upper(): raise RuntimeError(f"stale lifecycle mutation {sym}: expected {expected_state}, actual {old}")
        if old!=new_state and new_state not in POSITION_TRANSITIONS.get(old,set()): raise RuntimeError(f"invalid lifecycle transition {sym}: {old}->{new_state}")
        pos["lifecycle"]=new_state
        pos["status"]="pending" if new_state in {"DISCOVERED","ENTRY_PENDING","OPENING","PROTECTION_PENDING"} else ("active" if new_state not in {"CLOSED","EMERGENCY"} else ("closed" if new_state=="CLOSED" else "EMERGENCY"))
        pid=pos.get("position_id")
    _emit_execution_event("POSITION_LIFECYCLE",entity_id=pid or sym,correlation_id=pid or RUN_ID,payload={"symbol":sym,"from":old,"to":new_state,"reason":str(reason)[:300]},persist=True)
    return new_state


class ExecutionRequest:
    __slots__=("request_id","request_type","created_at","expires_at","source","correlation_id","position_id","symbol","execution_mode","strategy_version","requested_action","parameters")
    def __init__(self, **kw):
        now=time.time()
        self.request_id=kw.get("request_id") or _new_request_id("EXEC")
        self.request_type=str(kw.get("request_type") or "BINANCE_MUTATION")
        self.created_at=float(kw.get("created_at") or now)
        self.expires_at=float(kw.get("expires_at") or (now+30.0))
        self.source=str(kw.get("source") or "main")
        self.correlation_id=str(kw.get("correlation_id") or RUN_ID)
        self.position_id=kw.get("position_id")
        self.symbol=kw.get("symbol")
        self.execution_mode=kw.get("execution_mode")
        self.strategy_version=kw.get("strategy_version")
        self.requested_action=str(kw.get("requested_action") or "")
        self.parameters=dict(kw.get("parameters") or {})
    def expired(self): return time.time()>self.expires_at
    def as_dict(self): return {k:getattr(self,k) for k in self.__slots__}

class ProtectionMutationRequest:
    __slots__=("request_id","position_id","symbol","expected_version","created_at","expires_at","action")
    def __init__(self,position_id,symbol,expected_version,action,expires_sec=20):
        self.request_id=_new_request_id("PROT"); self.position_id=str(position_id or ""); self.symbol=str(symbol or "")
        self.expected_version=int(expected_version or 0); self.created_at=time.time(); self.expires_at=self.created_at+float(expires_sec); self.action=str(action)
    def expired(self): return time.time()>self.expires_at

def _validate_protection_mutation(sym,expected_version,request_id):
    with positions_lock:
        pos=positions.get(sym)
        if not pos: raise RuntimeError(f"protection mutation rejected: {sym} position missing")
        actual=int(pos.get("protection_version",0) or 0)
        if actual!=int(expected_version): raise RuntimeError(f"stale protection request {request_id}: expected v{expected_version}, actual v{actual}")
        return pos.get("position_id")

_EXECUTION_IDEMPOTENCY_LOCK=threading.RLock()
_EXECUTION_IDEMPOTENCY={}
_EXECUTION_IDEMPOTENCY_TTL=6*3600


def _prune_execution_idempotency():
    now=time.time()
    with _EXECUTION_IDEMPOTENCY_LOCK:
        for k in [k for k,v in _EXECUTION_IDEMPOTENCY.items() if now-v.get("at",now)>_EXECUTION_IDEMPOTENCY_TTL]:
            _EXECUTION_IDEMPOTENCY.pop(k,None)


class ExecutionController:
    """Single Binance mutation authority. High-level helpers ultimately use this path."""
    MUTATIONS={"POST","PUT","DELETE"}
    def submit_signed(self, method, path, params=None, *, critical=False, request_type="BINANCE_MUTATION", source="main", position_id=None, symbol=None, execution_mode="REAL", strategy_version=None, correlation_id=None, expires_sec=30.0):
        method=str(method).upper()
        if method not in self.MUTATIONS:
            return _binance_signed_impl(method,path,params=params,critical=critical)
        req=ExecutionRequest(request_type=request_type,source=source,position_id=position_id,symbol=symbol,execution_mode=execution_mode,strategy_version=strategy_version,correlation_id=correlation_id,requested_action=f"{method} {path}",parameters=params,expires_at=time.time()+float(expires_sec))
        if req.expired(): raise RuntimeError(f"execution request expired: {req.request_id}")
        _prune_execution_idempotency()
        # Exchange order IDs/client IDs remain the strongest duplicate guard. This in-process key catches identical concurrent requests.
        key=f"{req.request_type}|{req.symbol or ''}|{req.position_id or ''}|{req.requested_action}|{json.dumps(params or {},sort_keys=True,default=str)}"
        with _EXECUTION_IDEMPOTENCY_LOCK:
            prev=_EXECUTION_IDEMPOTENCY.get(key)
            if prev and prev.get("inflight"): raise RuntimeError(f"duplicate mutation blocked: {req.request_id}")
            if prev and prev.get("has_result"): return prev.get("result")
            _EXECUTION_IDEMPOTENCY[key]={"at":time.time(),"inflight":True,"request_id":req.request_id}
        _emit_execution_event("EXECUTION_REQUESTED",entity_id=req.position_id or req.symbol,correlation_id=req.correlation_id,payload=req.as_dict(),persist=False)
        try:
            result=_binance_signed_impl(method,path,params=params,critical=critical)
        except Exception as exc:
            msg=str(exc).lower()
            if any(k in msg for k in ("429","418","rate","timeout","connection","unknown")): _record_circuit_failure(msg[:120])
            with _EXECUTION_IDEMPOTENCY_LOCK:
                _EXECUTION_IDEMPOTENCY[key]={"at":time.time(),"error":str(exc)[:500],"request_id":req.request_id}
            _emit_execution_event("EXECUTION_RESULT",entity_id=req.position_id or req.symbol,correlation_id=req.correlation_id,payload={"request_id":req.request_id,"status":"ERROR","error":str(exc)[:500]},persist=True)
            raise
        with _EXECUTION_IDEMPOTENCY_LOCK:
            _EXECUTION_IDEMPOTENCY[key]={"at":time.time(),"has_result":True,"result":result,"request_id":req.request_id}
        _emit_execution_event("EXECUTION_RESULT",entity_id=req.position_id or req.symbol,correlation_id=req.correlation_id,payload={"request_id":req.request_id,"status":"SUCCESS","method":method,"path":path},persist=True)
        return result

_execution_controller=ExecutionController()
_CIRCUIT_LOCK=threading.RLock(); _CIRCUIT_FAILURES=deque(maxlen=50); _CIRCUIT_OPEN_AT=0.0
_CIRCUIT_WINDOW_SEC=300.0; _CIRCUIT_THRESHOLD=5; _CIRCUIT_COOLDOWN_SEC=180.0
def _record_circuit_failure(kind):
    global CIRCUIT_BREAKER_OPEN, STOP_NEW_ENTRIES, _CIRCUIT_OPEN_AT, _CIRCUIT_FAILURES
    now=time.time()
    with _CIRCUIT_LOCK:
        _CIRCUIT_FAILURES.append((now,str(kind))); recent=[x for x in _CIRCUIT_FAILURES if now-x[0]<=_CIRCUIT_WINDOW_SEC]
        if len(recent)>=_CIRCUIT_THRESHOLD and not CIRCUIT_BREAKER_OPEN:
            CIRCUIT_BREAKER_OPEN=True; STOP_NEW_ENTRIES=True; _CIRCUIT_OPEN_AT=now; _set_component_health("execution","BLOCKED",f"circuit opened: {len(recent)} failures")
            log.error("[CIRCUIT] OPEN")
def _circuit_health_tick():
    global CIRCUIT_BREAKER_OPEN, STOP_NEW_ENTRIES
    with _CIRCUIT_LOCK:
        if not CIRCUIT_BREAKER_OPEN or time.time()-_CIRCUIT_OPEN_AT<_CIRCUIT_COOLDOWN_SEC: return
        recent=[x for x in _CIRCUIT_FAILURES if time.time()-x[0]<=_CIRCUIT_WINDOW_SEC]
        if not recent:
            CIRCUIT_BREAKER_OPEN=False; STOP_NEW_ENTRIES=(RUNTIME_STATE!="READY"); _set_component_health("execution","HEALTHY","circuit recovered"); log.info("[CIRCUIT] CLOSED")
_execution_controller=ExecutionController()


def _validate_brain_contract(brain):
    if brain is None:
        return False, "strategy_logic unavailable"

    required = {
        "full_analyze": "callable",
        "manage_position": "callable",
        "record_candidate_observation": "callable",
        "record_trade_outcome": "callable",
        "get_learning_schema": "callable",
        "get_cognitive_status": "callable",
        "full_command": "callable",
    }
    missing = [name for name, kind in required.items() if not callable(getattr(brain, name, None))]
    # V35 brain historically exposed ingest_* aliases instead of the newer names.
    if "record_candidate_observation" in missing and callable(getattr(brain, "ingest_live_candidate", None)):
        missing.remove("record_candidate_observation")
    if "record_trade_outcome" in missing and callable(getattr(brain, "ingest_live_outcome", None)):
        missing.remove("record_trade_outcome")
    if missing:
        return False, f"missing required brain interface: {', '.join(missing)}"

    advertised = str(getattr(brain, "BRAIN_INTERFACE_VERSION", "") or "").strip()
    brain_version = advertised or str(getattr(brain, "V35_VERSION", "") or "").strip() or str(getattr(brain, "V32_VERSION", "") or "").strip()
    if brain_version and brain_version not in BRAIN_COMPATIBLE_LEGACY_VERSIONS:
        # Unknown explicit versions must not be silently accepted.
        return False, f"brain interface/version unsupported: {brain_version!r}"

    # Validate signatures with semantics, not just presence.
    checks = {
        "full_analyze": {"min_params": 2},
        "manage_position": {"min_params": 2},
    }
    for name, rules in checks.items():
        try:
            sig = inspect.signature(getattr(brain, name))
            positional = [p for p in sig.parameters.values()
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if len(positional) < rules["min_params"]:
                return False, f"{name} signature too small: {sig}"
        except Exception as exc:
            return False, f"signature inspection failed for {name}: {exc}"

    # Public schemas/status must be serializable and object-shaped when callable.
    for name in ("get_learning_schema", "get_cognitive_status"):
        fn = getattr(brain, name, None)
        if callable(fn):
            try:
                value = fn()
                json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
            except Exception as exc:
                return False, f"{name} validation failed: {exc}"

    # Optional but strongly preferred strategy descriptor.
    descriptor = getattr(brain, "get_strategy_descriptor", None)
    if callable(descriptor):
        try:
            value = descriptor()
            json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
        except Exception as exc:
            return False, f"strategy descriptor validation failed: {exc}"

    return True, f"contract OK ({brain_version or 'unversioned-compatible'})"

def _validate_decision_freshness(signal):
    if not isinstance(signal,dict): return False,"decision packet is not a dict"
    created=signal.get("decision_created_at") or signal.get("created_at") or signal.get("analysis_time")
    expires=signal.get("decision_expires_at") or signal.get("expires_at")
    if created is None and expires is None: return True,"legacy-current-scan"
    try:
        now=time.time()
        if expires is not None and now>float(expires): return False,"decision expired"
        if created is not None and now+1<float(created): return False,"decision timestamp is in the future"
    except (TypeError,ValueError): return False,"invalid decision timestamp"
    return True,"fresh"


def _bootstrap_validate_and_reconcile():
    ok,detail=_validate_brain_contract(_brain)
    _set_component_health("brain","HEALTHY" if ok else "BLOCKED",detail)
    try:
        fapi_get("/fapi/v1/time")
        _set_component_health("binance_rest","HEALTHY","public REST reachable")
    except Exception as exc:
        _set_component_health("binance_rest","DEGRADED",str(exc))
    try:
        _load_all_symbol_filters()
        _set_component_health("execution","HEALTHY","exchange metadata loaded")
    except Exception as exc:
        _set_component_health("execution","DEGRADED",f"exchange metadata: {exc}")
    key,secret=_read_binance_credentials()
    if key and secret:
        try:
            with _binance_critical_context():
                remote_positions=get_real_positions_all()
            with positions_lock:
                local_real={sym for sym,pos in positions.items() if _position_is_real(pos)}
            remote_by={str(p.get("symbol")):p for p in remote_positions if p.get("symbol")}
            unresolved=[]
            for sym in sorted(local_real):
                pos=positions.get(sym)
                remote=remote_by.get(sym)
                if remote is None:
                    unresolved.append(sym); continue
                try:
                    with _binance_critical_context():
                        if pos.get("signal") and pos.get("current_sl") is not None:
                            _verify_protection_pair(sym,pos["signal"].get("decision")=="BUY",pos["signal"].get("tp"),pos.get("current_sl"),abs(float(remote.get("positionAmt",0) or 0)))
                    with positions_lock:
                        if sym in positions:
                            positions[sym]["quantity"]=abs(float(remote.get("positionAmt",0) or 0))
                except Exception as exc:
                    unresolved.append(sym)
                    _force_position_emergency(sym, str(exc)[:400])
            unmanaged=sorted(set(remote_by)-local_real)
            if unresolved or unmanaged:
                _set_component_health("protection","EMERGENCY",f"unresolved={unresolved[:6]} unmanaged={unmanaged[:6]}")
                global STOP_NEW_ENTRIES
                STOP_NEW_ENTRIES=True
                _set_runtime_state("EMERGENCY", "startup real-state safety unresolved")
            else:
                _set_component_health("protection","HEALTHY","real protection verified")
        except Exception as exc:
            _set_component_health("binance_rest","RECOVERING",str(exc))
            STOP_NEW_ENTRIES=True
            _set_runtime_state("RECOVERING",f"private reconciliation unavailable: {exc}")
    else:
        _set_component_health("protection","HEALTHY","simulation/no private credentials")
    _set_component_health("persistence","HEALTHY","local state ready")
    _set_component_health("research","HEALTHY","research non-critical")
    _set_component_health("resource","HEALTHY",f"heavy cap={MAX_HEAVY_WORKERS}")
    return _health_snapshot()


def _graceful_shutdown(reason="shutdown"):
    global STOP_NEW_ENTRIES
    if RUNTIME_STATE!="STOPPING":
        try: _set_runtime_state("STOPPING",reason)
        except Exception: RUNTIME_STATE="STOPPING"
    STOP_NEW_ENTRIES=True
    try: _full_off()
    except Exception: pass
    try: _save_runtime_checkpoint(push_github=False)
    except Exception as exc: log.warning(f"[SHUTDOWN] checkpoint gagal: {exc}")
    deadline=time.time()+3.0
    for t in list(_ACTIVE_HEAVY_THREADS.values()):
        rem=max(0.0,deadline-time.time())
        if rem<=0: break
        try: t.join(rem)
        except Exception: pass
    SHUTDOWN_EVENT.set()




def _heavy_worker_target(name, fn, *args, worker_id=None, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        log.exception(f"[WORKER] {name} failed: {exc}")
        try:
            _set_component_health("resource", "DEGRADED", f"worker {name}: {exc}")
        except Exception:
            pass
    finally:
        with _HEAVY_WORKER_LOCK:
            if worker_id is not None:
                _HEAVY_WORKER_REGISTRY.pop(worker_id, None)
            _ACTIVE_HEAVY_THREADS.pop(threading.current_thread().name, None)
        HEAVY_WORKER_SEMAPHORE.release()


_ACTIVE_HEAVY_THREADS={}

def _start_heavy_worker(name, fn, *args, **kwargs):
    # Reserve the global heavy-worker slot BEFORE starting the thread.
    # This prevents callers from believing a worker started when the semaphore was full.
    with _HEAVY_WORKER_LOCK:
        if any(v.get("name") == name for v in _HEAVY_WORKER_REGISTRY.values()):
            log.warning(f"[RESOURCE] worker {name} already running; skip duplicate")
            return None
        if not HEAVY_WORKER_SEMAPHORE.acquire(blocking=False):
            log.warning(f"[RESOURCE] heavy worker limit reached ({MAX_HEAVY_WORKERS}); defer {name}")
            return None
        worker_id = _new_request_id("WRK")
        thread_name = f"heavy-{name}-{worker_id[-6:]}"
        _HEAVY_WORKER_REGISTRY[worker_id] = {"name": name, "started_at": time.time(), "thread": thread_name}
    try:
        t = threading.Thread(
            target=_heavy_worker_target,
            args=(name, fn, *args),
            kwargs={"worker_id": worker_id, **kwargs},
            name=thread_name,
            daemon=True,
        )
        with _HEAVY_WORKER_LOCK:
            _ACTIVE_HEAVY_THREADS[thread_name] = t
        t.start()
        return t
    except Exception:
        with _HEAVY_WORKER_LOCK:
            _HEAVY_WORKER_REGISTRY.pop(worker_id, None)
            _ACTIVE_HEAVY_THREADS.pop(thread_name, None)
        HEAVY_WORKER_SEMAPHORE.release()
        raise


def _heavy_worker_snapshot():
    with _HEAVY_WORKER_LOCK:
        return {k: dict(v) for k, v in _HEAVY_WORKER_REGISTRY.items()}

# Long-lived I/O waiters (e.g. pending entry watchers) are NOT heavy compute.
# Keeping them out of the heavy pool prevents four pending entries from starving
# the scanner/FULL/recovery workers.  The global heavy-worker ceiling remains 5.
_LIGHT_WORKER_LOCK = threading.RLock()
_LIGHT_WORKER_REGISTRY = {}
_MAX_LIGHT_WORKERS = max(4, _env_int("MAX_LIGHT_WORKERS", 16, minimum=4, maximum=64))

def _light_worker_target(name, fn, *args, worker_id=None, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        log.exception(f"[LIGHT WORKER] {name} failed: {exc}")
    finally:
        with _LIGHT_WORKER_LOCK:
            if worker_id:
                _LIGHT_WORKER_REGISTRY.pop(worker_id, None)

def _start_light_worker(name, fn, *args, **kwargs):
    with _LIGHT_WORKER_LOCK:
        if len(_LIGHT_WORKER_REGISTRY) >= _MAX_LIGHT_WORKERS:
            log.warning(f"[RESOURCE] light worker limit reached ({_MAX_LIGHT_WORKERS}); defer {name}")
            return None
        if any(v.get("name") == name for v in _LIGHT_WORKER_REGISTRY.values()):
            log.debug(f"[RESOURCE] light worker {name} already running")
            return None
        worker_id = _new_request_id("LWRK")
        _LIGHT_WORKER_REGISTRY[worker_id] = {"name": name, "started_at": time.time()}
    try:
        t = threading.Thread(target=_light_worker_target, args=(name, fn, *args), kwargs={"worker_id": worker_id, **kwargs}, name=f"light-{name}-{worker_id[-6:]}", daemon=True)
        t.start()
        return t
    except Exception:
        with _LIGHT_WORKER_LOCK:
            _LIGHT_WORKER_REGISTRY.pop(worker_id, None)
        raise

def _light_worker_snapshot():
    with _LIGHT_WORKER_LOCK:
        return {k: dict(v) for k,v in _LIGHT_WORKER_REGISTRY.items()}

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
        "confidence_min": None, "cutoff": float(cutoff),
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
    decision_rows=[x for x in rows if str(x.get("decision") or "").upper() in {"BUY","SELL"}]
    decision_count=len(decision_rows)
    bull=sum(1 for x in decision_rows if str(x.get("decision") or "").upper()=="BUY")
    bear=sum(1 for x in decision_rows if str(x.get("decision") or "").upper()=="SELL")
    neutral=max(0, decision_count-bull-bear)
    def med(key):
        vals=[float(x[key]) for x in rows if x.get(key) is not None]
        return float(np.median(vals)) if vals else None
    breadth=(bull-bear)/decision_count if decision_count else 0.0
    eff=med("efficiency_4h") or 0.0
    rr=med("range_expansion_ratio") or 1.0
    avg_rv=float(np.mean([float(x["relative_volume"]) for x in rows if x.get("relative_volume") is not None])) if any(x.get("relative_volume") is not None for x in rows) else None
    med_r1=med("price_1h_pct")
    med_r4=med("price_4h_pct")
    if analyzed==0:
        regime="unknown"
    elif abs(breadth)>=0.35 and eff>=0.45 and decision_count>0:
        regime="bullish expansion" if breadth>0 else "bearish expansion"
    elif abs(breadth)<=0.15 and eff<=0.35:
        regime="range/compression"
    elif abs(breadth)>=0.20 and decision_count>0:
        regime="bullish trend" if breadth>0 else "bearish trend"
    else:
        regime="transition"
    btc=[x for x in rows if str(x.get("symbol"))=="BTCUSDT"]
    btc1=btc[0].get("price_1h_pct") if btc else None
    btc4=btc[0].get("price_4h_pct") if btc else None
    return {"market_regime":regime,"bullish_breadth_pct":100*bull/decision_count if decision_count else None,"bearish_breadth_pct":100*bear/decision_count if decision_count else None,"neutral_breadth_pct":100*neutral/decision_count if decision_count else None,"breadth_score":breadth,"median_price_1h_pct":med_r1,"median_price_4h_pct":med_r4,"median_efficiency_4h":eff,"median_range_expansion_ratio":rr,"avg_relative_volume":avg_rv,"btc_price_1h_pct":btc1,"btc_price_4h_pct":btc4,"analyzed_symbols":analyzed}

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
                d = r.json(); retry_after = int((d.get("parameters") or {}).get("retry_after", 5))
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
# Bybit is the primary public market-data infrastructure. Binance REST is reserved
# for authenticated execution/reconciliation and is never required for scanning.
BYBIT_PUBLIC_REQUEST_INTERVAL = 0.18
BYBIT_PUBLIC_MIN_INTERVAL = 0.18  # <= ~5.5 req/s globally; comfortably below IP ceilings
_bybit_request_lock = threading.Lock()
_bybit_last_request_at = 0.0
_bybit_rate_limit_until = 0.0
_bybit_request_count = 0
_bybit_request_errors = 0


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
BINANCE_TIME_SYNC_TTL = 300.0
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
    """Sync clock using one governed Binance public request; never bypass the rate limiter."""
    global _binance_time_offset_ms, _binance_time_sync_at
    now=time.time()
    with _binance_time_sync_lock:
        if not force and (now-_binance_time_sync_at)<BINANCE_TIME_SYNC_TTL:
            return _binance_time_offset_ms
    if _binance_is_scan_paused() and not force:
        raise BinanceCooldownError(f"Binance cooldown aktif {_binance_cooldown_remaining():.0f}s")
    local_send=int(time.time()*1000)
    with _binance_request_slot(critical=True):
        try:
            r=requests.get(f"{FAPI}/fapi/v1/time",timeout=5,verify=False)
            _binance_update_weight_from_response(r)
            if r.status_code in (418,429):
                _binance_register_ban(r.text,retry_after=r.headers.get("Retry-After"))
                raise BinanceCooldownError(f"Binance time endpoint HTTP {r.status_code}")
            r.raise_for_status()
            server_ms=int(r.json()["serverTime"])
        except BinanceCooldownError:
            raise
        except Exception as e:
            log.warning(f"[binance-time] sync gagal: {e}")
            raise
    local_recv=int(time.time()*1000); midpoint=(local_send+local_recv)//2; offset=server_ms-midpoint
    with _binance_time_sync_lock:
        _binance_time_offset_ms=int(offset); _binance_time_sync_at=time.time()
    log.info(f"[binance-time] sync OK offset={offset}ms")
    return int(offset)

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
    data = _binance_signed("GET", "/fapi/v1/openAlgoOrders", {"symbol": sym}, critical=True)
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
        log.error(f"[BINANCE PAUSE] Entry baru Binance dihentikan selama {remaining:.0f} detik. Scanner Bybit tetap berjalan; WS tetap memantau posisi.")
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
            "Entry baru Binance dihentikan. <b>Scanner Bybit tetap berjalan.</b> Posisi aktif tetap dipantau via WS."
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
def _binance_critical_context():
    """Promote nested Binance calls to the bounded emergency/reconciliation lane."""
    previous = bool(getattr(_binance_priority_local, "critical", False))
    _binance_priority_local.critical = True
    try:
        yield
    finally:
        _binance_priority_local.critical = previous


@contextmanager
def _binance_request_slot(critical=False):
    """Serialize Binance calls; stale usage must expire and scan keeps execution reserve."""
    global _binance_last_request_at, _binance_weight_1m, _binance_weight_seen_at
    critical = bool(critical or getattr(_binance_priority_local, "critical", False))
    with _binance_request_lock:
        _binance_wait_if_banned()
        used = _binance_weight_1m
        age = time.time() - float(_binance_weight_seen_at or 0.0)
        if used is not None and age > BINANCE_WEIGHT_STALE_AFTER_SEC:
            used = None
            _binance_weight_1m = None
            _binance_weight_seen_at = 0.0
        if used is not None:
            if critical:
                if used >= BINANCE_CRITICAL_HARD_LIMIT:
                    raise BinanceCooldownError(f"Binance critical governor: {used}/min >= critical hard limit {BINANCE_CRITICAL_HARD_LIMIT}")
            else:
                normal_limit = min(BINANCE_WEIGHT_SOFT_LIMIT, max(1, BINANCE_WEIGHT_HARD_LIMIT - BINANCE_EXECUTION_RESERVE))
                if used >= normal_limit:
                    raise BinanceCooldownError(f"Binance scan governor: {used}/min >= normal limit {normal_limit} (execution reserve {BINANCE_EXECUTION_RESERVE})")
        wait = BINANCE_REQUEST_INTERVAL - (time.monotonic() - _binance_last_request_at)
        if wait > 0:
            time.sleep(wait)
        _binance_wait_if_banned()
        _binance_last_request_at = time.monotonic()
        yield

def _bybit_wait_slot():
    global _bybit_last_request_at
    wait_until = time.time()
    with _bybit_request_lock:
        now=time.monotonic()
        wait = BYBIT_PUBLIC_MIN_INTERVAL - (now - _bybit_last_request_at)
        if wait > 0:
            time.sleep(wait)
        _bybit_last_request_at = time.monotonic()

def _bybit_get(path, params=None):
    """Primary public Bybit REST accessor for market data. One controlled request per call."""
    global _bybit_rate_limit_until, _bybit_request_count, _bybit_request_errors
    now=time.time()
    if _bybit_rate_limit_until > now:
        raise ConnectionError(f"Bybit cooldown active {int(_bybit_rate_limit_until-now)}s")
    _bybit_wait_slot()
    _bybit_request_count += 1
    try:
        r=requests.get(f"{BYBIT}{path}", params=params, timeout=10, verify=False)
        # honor documented response rate-limit headers when present
        reset_ms = r.headers.get("X-Bapi-Limit-Reset-Timestamp")
        remain = r.headers.get("X-Bapi-Limit-Status")
        if remain is not None:
            try:
                if int(float(remain)) <= 1 and reset_ms:
                    reset_s=max(0.0,(float(reset_ms)/1000.0)-time.time())
                    _bybit_rate_limit_until=max(_bybit_rate_limit_until, time.time()+min(max(reset_s,0.0),5.0))
            except Exception:
                pass
        if r.status_code == 403 and "access too frequent" in r.text.lower():
            _bybit_rate_limit_until=time.time()+600.0
            raise ConnectionError("Bybit HTTP 403 access too frequent; 10m safety cooldown")
        r.raise_for_status()
        data=r.json()
        if isinstance(data,dict) and int(data.get("retCode",0) or 0) == 10006:
            _bybit_rate_limit_until=time.time()+10.0
            raise ConnectionError("Bybit API rate limit 10006")
        return data
    except Exception:
        _bybit_request_errors += 1
        raise

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

def _binance_signed_impl(method, path, params=None, critical=False):
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
    critical = bool(critical or mutating)
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

            with _binance_request_slot(critical=critical):
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
            time.sleep(0.25 + 0.25 * attempt)
        except Exception as e:
            last_err = e
            if mutating:
                raise BinanceUnknownExecutionError(
                    f"Binance {method} {path} response error; execution status unknown: {e}"
                ) from e
            log.warning(f"[binance-signed] GET {path} percobaan {attempt+1}: {e}")
            time.sleep(0.25 + 0.25 * attempt)

    raise RuntimeError(f"Gagal request signed {method} {path}: {last_err}")


def _binance_signed_legacy(method, path, params=None, critical=False):
    """Legacy compatibility helper; final authority is defined near runtime entry."""
    method_u=str(method).upper()
    if method_u in {"POST","PUT","DELETE"}:
        with EXECUTION_MUTATION_LOCK:
            return _binance_signed_impl(method_u, path, params=params, critical=critical)
    return _binance_signed_impl(method_u, path, params=params, critical=critical)

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
                "minNotional": float((f.get("MIN_NOTIONAL") or {}).get("notional", 5.0)),
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

    mode = _binance_signed("GET", "/fapi/v1/positionSide/dual", {}, critical=True)
    acct = _binance_signed("GET", "/fapi/v2/account", {}, critical=True)
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
        return _binance_signed("GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": client_id}, critical=True)
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
        with _binance_critical_context():
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
        rows = _binance_signed("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, critical=True)
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
    return _binance_signed("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, critical=True)



def get_real_position(symbol):
    rows = _binance_signed("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, critical=True)
    for p in rows:
        if p["symbol"] == symbol:
            if abs(float(p.get("positionAmt", 0) or 0)) > 0:
                return p
            return None
    return None

def get_real_positions_all():
    """Return every non-zero Futures position visible on Binance."""
    rows = _binance_signed("GET", "/fapi/v2/positionRisk", {}, critical=True)
    return [p for p in (rows or []) if abs(float(p.get("positionAmt", 0) or 0)) > 0]


def get_open_orders_all(symbol=None):
    """Return ordinary open orders. With no symbol, query the whole account."""
    params = {"symbol": symbol} if symbol else {}
    rows = _binance_signed("GET", "/fapi/v1/openOrders", params, critical=True)
    return rows if isinstance(rows, list) else []


def get_open_algo_orders_all(symbol=None):
    """Return open conditional/algo orders. With no symbol, query the whole account."""
    params = {"symbol": symbol} if symbol else {}
    data = _binance_signed("GET", "/fapi/v1/openAlgoOrders", params, critical=True)
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
    """Emergency flatten+cleanup using the bounded critical Binance lane."""
    try:
        with _binance_critical_context():
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
    except BinanceCooldownError as e:
        _force_position_emergency(sym, f"{reason}: {e}")
        _queue_pending_cleanup(sym, "timeout deferred by Binance governor/cooldown", e)
        tg_send(chat_id, f"🚨 <b>TIMEOUT TERTUNDA</b> — {sym}\n<code>{html.escape(str(e)[:350])}</code>\nTidak ada retry agresif; posisi tetap dicatat untuk rekonsiliasi.")
        return False
    except Exception as e:
        _force_position_emergency(sym, f"{reason}: {e}")
        _queue_pending_cleanup(sym, "timeout cleanup", e)
        tg_send(chat_id, f"🚨 <b>TIMEOUT BELUM SELESAI</b> — {sym}\n<code>{html.escape(str(e)[:350])}</code>\nPosisi tetap dipertahankan di /trade. Gunakan <code>/ok {sym}</code> untuk rekonsiliasi.")
        return False

def _verified_timeout_all(chat_id):
    """Global emergency cleanup using the bounded critical Binance lane."""
    try:
        with _binance_critical_context():
            positions_remote = get_real_positions_all()
            ordinary = get_open_orders_all()
            algo = get_open_algo_orders_all()
            symbols = {str(p.get("symbol")) for p in positions_remote if p.get("symbol")}
            symbols.update(str(o.get("symbol")) for o in ordinary if o.get("symbol"))
            symbols.update(str(o.get("symbol")) for o in algo if o.get("symbol"))
            for sym in sorted(symbols):
                _cancel_all_symbol_orders_verified(sym)
            exit_prices = {}
            for p in positions_remote:
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
    except BinanceCooldownError as e:
        tg_send(chat_id, "🚨 <b>TIMEOUT GLOBAL TERTUNDA</b>\n"
                        f"<code>{html.escape(str(e)[:500])}</code>\n"
                        "Critical lane sedang dibatasi; tidak ada retry agresif.")
        log.warning(f"[TIMEOUT GLOBAL] ditunda: {e}")
        return False
    except Exception as e:
        tg_send(chat_id, "🚨 <b>TIMEOUT GLOBAL BELUM SELESAI</b>\n"
                        f"<code>{html.escape(str(e)[:500])}</code>\n"
                        "Cek Binance dan gunakan /ok SYMBOL untuk rekonsiliasi.")
        log.error(f"[TIMEOUT GLOBAL] cleanup gagal: {e}")
        return False
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
    return _binance_signed("GET", "/fapi/v1/algoOrder", {"algoId": algo_id}, critical=True)


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
        rows = _binance_signed("GET", "/fapi/v2/balance", {}, critical=True)
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
    d = _bybit_get("/v5/market/kline", {
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
    d = _bybit_get("/v5/market/tickers",
                 {"category":"linear","symbol":symbol})
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit ticker error: {d.get('retMsg')}")
    return float(d["result"]["list"][0]["lastPrice"])

def _bybit_top_coins(exclude_syms):
    d = _bybit_get("/v5/market/tickers", {"category":"linear"})
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
        p = (d.get(cid) or {}).get("usd")
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
_LOCAL_PRICE_MAX_AGE = 30.0
_local_price_cache = {}
_local_price_lock = threading.Lock()

def get_price(symbol, prefer_binance=False):
    """Market price accessor: Bybit first for analysis; Binance only on explicit real-position path."""
    now=time.time()
    if prefer_binance and not _binance_is_scan_paused():
        try:
            with _binance_critical_context():
                bp=_binance_price(symbol)
            if bp is not None:
                with _local_price_lock: _local_price_cache[symbol]=(bp,now)
                return bp
        except Exception as e:
            log.debug(f"[price/binance-authoritative] {symbol}: {e}")
    with _local_price_lock:
        cached=_local_price_cache.get(symbol)
    # fresh local market cache first
    if cached and now-cached[1] <= _LOCAL_PRICE_MAX_AGE:
        return cached[0]
    try:
        byp=_bybit_price(symbol)
        with _local_price_lock: _local_price_cache[symbol]=(byp,now)
        return byp
    except Exception as e:
        log.debug(f"[price/bybit] {symbol}: {e}")
    # Only explicit real-position paths should use Binance fallback.
    if prefer_binance:
        try:
            with _binance_critical_context():
                bp=_binance_price(symbol)
            if bp is not None:
                with _local_price_lock: _local_price_cache[symbol]=(bp,now)
                return bp
        except Exception as e:
            log.debug(f"[price/binance-fallback] {symbol}: {e}")
    p=_coingecko_price(symbol)
    if p is not None:
        with _local_price_lock: _local_price_cache[symbol]=(p,now)
    return p

def get_klines(symbol, interval, limit=250):
    """Bybit-first market data. Binance REST is intentionally not part of normal analysis."""
    cached=_scan_cache_get(symbol,interval,limit)
    if cached is not None:
        return cached
    try:
        df=_bybit_klines(symbol,interval,limit)
        if not df.empty:
            _scan_cache_put(symbol,interval,df,"bybit")
            return df
    except Exception as e:
        log.warning(f"[klines/bybit] {symbol} {interval}: {e}")
    # free Binance WS buffer may still be used as a non-REST fallback for monitoring
    df=ws_feed.get_klines(symbol,interval,limit) if ws_feed.is_fresh() else pd.DataFrame()
    return df if df is not None else pd.DataFrame()

def get_scan_klines(symbol, interval, limit=250):
    """Scanner market-data path: cache -> Bybit REST -> optional WS buffer. Never Binance REST."""
    cached=_scan_cache_get(symbol,interval,limit)
    if cached is not None:
        return cached
    key=(symbol,interval); lock=_scan_key_lock(key)
    with lock:
        cached=_scan_cache_get(symbol,interval,limit)
        if cached is not None: return cached
        df=pd.DataFrame()
        try:
            df=_bybit_klines(symbol,interval,limit)
            if not df.empty:
                _scan_cache_put(symbol,interval,df,"bybit")
                # Keep Binance WS only as a free monitoring fallback; no REST backfill.
                try: ws_feed.seed_klines(symbol,interval,df)
                except Exception: pass
                return df.tail(limit).copy()
        except Exception as e:
            log.warning(f"[scan-data/bybit] {symbol} {interval}: {e}")
        if ws_feed.is_fresh():
            ws_df=ws_feed.get_klines(symbol,interval,limit)
            if ws_df is not None and not ws_df.empty and len(ws_df)>=min(limit,40):
                _scan_cache_put(symbol,interval,ws_df,"ws")
                return ws_df.tail(limit).copy()
        return pd.DataFrame()

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
    """Public scanner universe API.

    Signature intentionally has no required arguments. Any canonical
    implementation behind this wrapper must therefore accept exclude_syms
    as optional; this is also enforced by the runtime contract audit.
    """
    coins = _get_top_coins_impl(exclude_syms=None)
    global last_scanned_coins, last_scanned_at
    with _last_scanned_lock:
        last_scanned_coins = coins
        last_scanned_at = time.time()
    return coins

_TOP_COINS_CACHE_TTL = 120.0
_top_coins_cached_symbols = []
_top_coins_cached_at = 0.0
_top_coins_cache_lock = threading.Lock()

def _get_top_coins_impl():
    """Top-universe source is Bybit public market data. Binance is not queried for scanning."""
    global scan_counter,_top_coins_cached_symbols,_top_coins_cached_at
    with ban_lock:
        scan_counter+=1
        cur_ban=set(banned_coins.keys())
        now=time.time()
        expired=[]
        for sym,meta in list(banned_coins.items()):
            try:
                if isinstance(meta,tuple):
                    banned_at,dur=meta;
                    # legacy scan-count ban
                    exp=(scan_counter-int(banned_at))>=int(dur)
                else:
                    until=float(meta.get("until",0) or 0)
                    exp=until>0 and now>=until
                    if not until:
                        exp=(scan_counter-float(meta.get("banned_at",scan_counter)))>=float(meta.get("duration",0))
                if exp: expired.append(sym)
            except Exception: continue
        for sym in expired:
            banned_coins.pop(sym,None); cur_ban.discard(sym); log.info(f"[unban] {sym} kembali aktif")
    with positions_lock: active_syms=set(positions.keys())
    exclude_syms=cur_ban|active_syms
    with _top_coins_cache_lock:
        cached_syms=list(_top_coins_cached_symbols); cached_at=float(_top_coins_cached_at or 0.0)
    if cached_syms and time.time()-cached_at<=_TOP_COINS_CACHE_TTL:
        return [s for s in cached_syms if s not in exclude_syms][:TOP_N_COINS]
    try:
        coins=_bybit_top_coins(exclude_syms)
        if coins:
            with _top_coins_cache_lock:
                _top_coins_cached_symbols=list(coins); _top_coins_cached_at=time.time()
            return coins
    except Exception as e:
        log.warning(f"[top_coins/bybit] {e}")
    if ws_feed.is_fresh():
        raw=ws_feed.get_top_coins_raw()
        usdt=[t for t in raw if t.get("symbol","").endswith("USDT") and 0.0001<float(t.get("price",0))<MAX_PRICE and float(t.get("qvol",0))>5_000_000 and t.get("symbol") not in exclude_syms]
        usdt.sort(key=lambda x:float(x.get("qvol",0)),reverse=True)
        if usdt: return [t["symbol"] for t in usdt[:TOP_N_COINS]]
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
    # Scanner/analysis is intentionally decoupled from Binance entry pause.
    # STOP_NEW_ENTRIES and Binance cooldowns block NEW ORDER MUTATION only;
    # Bybit market analysis must continue so FULL can keep learning.
    # Analysis is fail-closed only for shutdown/emergency/brain-load failure.
    if SHUTDOWN_EVENT.is_set() or RUNTIME_STATE in {"STOPPING", "EMERGENCY"} or _STRATEGY_LOAD_ERROR:
        log.debug("[SCAN] analysis gate blocked by runtime safety/brain state")
        return []
    global early_reject_remaining
    """Scan universe with cached market data and record rich market context.

    V19 adds *derived* market breadth/relative-strength/regime telemetry only.
    It does not add a Binance endpoint or a new request: the M15/H1/D1 frames
    already fetched for strategy analysis are reused in memory.
    """
    scan_started=time.monotonic()
    tg_send(chat_id, f"🔍 Scanning {TOP_N_COINS} koin via Bybit...")
    try:
        symbols=get_top_coins()
    except BinanceCooldownError as e:
        tg_send(chat_id, f"⚠️ <b>Universe Bybit gagal</b> — sumber data analisis sedang tidak tersedia.\n<code>{html.escape(str(e)[:180])}</code>"); return []
    except Exception as e:
        tg_send(chat_id, f"⚠️ Market data error: <code>{str(e)[:150]}</code>"); return []
    if not symbols:
        tg_send(chat_id, "⚠️ Tidak ada koin tersedia untuk di-scan."); return []

    data_started=time.monotonic(); results=[]; all_scan_confidences=[]; market_rows=[]
    processed_symbols=analyzed_symbols=cache_hits=cache_misses=failed_symbols=low_conf_count=below_threshold_count=0
    candidate_count=eligible_count=ban_count=0
    rejection_reasons={}
    scan_deadline = scan_started + SCAN_MAX_DURATION_SEC
    for idx,sym in enumerate(symbols,1):
        if time.monotonic() >= scan_deadline:
            log.warning(f"[scan] hard deadline {SCAN_MAX_DURATION_SEC}s tercapai — cycle dihentikan aman.")
            break
        # Binance rate-limit must NOT stop market analysis; it only gates new Binance entry mutation.
        log.debug(f"[scan {idx:02d}/{len(symbols)}] {sym}")
        processed_symbols += 1
        try:
            before=_scan_cache_stats()
            h1=get_scan_klines(sym,"1h",250); m15=get_scan_klines(sym,"15m",250)
            try: d1=get_scan_klines(sym,"1d",100)
            except BinanceCooldownError: raise
            except Exception: d1=None
            after=_scan_cache_stats(); cache_misses += max(0,after[0]-before[0])
            r=full_analyze(h1,m15,d1,symbol=sym,market_data_source="bybit")
            # Market-context telemetry belongs to every successfully loaded chart,
            # not only to symbols where strategy_logic returned a trade candidate.
            # This keeps market context meaningful even when no setup is produced.
            row=_market_feature_row(sym,h1,m15,r if isinstance(r,dict) else {})
            row.update({"scan_time":time.time(),"run_id":research_run_id,"scan_counter":scan_counter})
            market_rows.append(row)
            if isinstance(r,dict):
                r.setdefault("candidate_uid", f"{research_run_id}|{scan_counter}|{sym}")
                if r.get("no_signal"):
                    log.info(f"[SCAN NO-SIGNAL] {sym} stage={r.get('analysis_stage')} reason={r.get('rejected_reason')}")
                _brain_on_candidate(r)
                analyzed_symbols+=1; conf=float(r.get("confidence",0) or 0); all_scan_confidences.append(conf)
                if bool(r.get("candidate") or r.get("is_candidate") or (r.get("decision") in {"BUY","SELL"} and not r.get("no_signal"))):
                    candidate_count += 1
                reason = str(r.get("rejected_reason") or r.get("eligibility_reason") or "UNCLASSIFIED")
                if bool(r.get("no_signal")) and reason == "UNCLASSIFIED":
                    reason = "NO_SIGNAL"
                if reason and reason not in {"UNCLASSIFIED", ""} and not bool(r.get("execution_eligible")):
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
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
                if bool(r.get("low_confidence")):
                    cutoff=float(r.get("low_confidence_cutoff") or 0.0)
                    low_conf_count+=1; _record_low_confidence_event(sym,conf,cutoff,r.get("decision"),r.get("entry_label"))
                if bool(r.get("ban_recommended")):
                    _ban_coin(sym, reason=str(r.get("ban_reason") or "brain recommendation"), duration=r.get("ban_duration"), kind=str(r.get("ban_kind") or "low_confidence"), confidence=conf)
                    ban_count += 1
                # Brain owns entry eligibility. Main only supplies a compatibility
                # fallback for truly legacy brains; current operational brain always
                # returns an explicit boolean + reason.
                if "execution_eligible" not in r:
                    r["execution_eligible"] = str(r.get("decision") or "").upper() in {"BUY", "SELL"}
                    r["eligibility_source"] = "legacy_decision_adapter"
                eligible = bool(r.get("execution_eligible"))
                if not eligible:
                    below_threshold_count+=1
                else:
                    eligible_count += 1
                    r["market_context"]={k:v for k,v in row.items() if k not in {"scan_time","run_id","scan_counter"}}
                    results.append(r); log.info(f"[SIGNAL] {sym} {r.get('decision')} confidence={conf:.1f}")
        except BinanceCooldownError:
            log.warning(f"[scan] {sym}: Binance cooldown aktif — scan cycle dihentikan aman."); break
        except Exception as e:
            failed_symbols+=1; log.warning(f"[scan] {sym}: {type(e).__name__}: {e}")
        # No per-symbol sleep here. Binance requests are serialized by _binance_request_slot().
        # Sleeping here only stretched scans without adding API safety.

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
    telemetry={"duration_sec":round(total_elapsed,2),"data_phase_sec":round(data_elapsed,2),"symbols_requested":len(symbols),"analyzed_symbols":analyzed_symbols,"avg_confidence":round(avg_conf,2) if avg_conf is not None else None,"min_confidence":round(min(all_scan_confidences),2) if all_scan_confidences else None,"max_confidence":round(max(all_scan_confidences),2) if all_scan_confidences else None,"low_confidence_count":low_conf_count,"below_threshold_count":below_threshold_count,"candidate_count":candidate_count,"eligible_count":eligible_count,"ban_count":ban_count,"rejection_reasons":dict(sorted(rejection_reasons.items(), key=lambda kv:(-kv[1],kv[0]))[:20]),"results":len(results),"failed_symbols":failed_symbols,"cache_entries":cache_total,"cache_fresh":cache_fresh,"binance_weight_1m":_binance_weight_1m,"binance_weight_seen_age_sec":round(max(0.0,time.time()-float(_binance_weight_seen_at or 0.0)),1) if _binance_weight_seen_at else None,"binance_execution_reserve":BINANCE_EXECUTION_RESERVE,"market_regime":mc.get("market_regime"),"bullish_breadth_pct":mc.get("bullish_breadth_pct"),"bearish_breadth_pct":mc.get("bearish_breadth_pct"),"median_efficiency_4h":mc.get("median_efficiency_4h"),"avg_relative_volume":mc.get("avg_relative_volume"),"btc_price_1h_pct":mc.get("btc_price_1h_pct"),"btc_price_4h_pct":mc.get("btc_price_4h_pct"),"source":"bybit_ws_primary_analysis","binance_entry_paused":bool(_binance_is_scan_paused()),"entry_mutations_blocked":bool(STOP_NEW_ENTRIES or CIRCUIT_BREAKER_OPEN or _binance_is_scan_paused())}
    _record_scan_telemetry(telemetry)
    # Give the brain one canonical summary per scan. This is the observation
    # boundary used for frequency health and controlled exploration; it never
    # mutates Binance and never bypasses the execution authority.
    try:
        _record_brain_scan_summary(dict(telemetry))
    except Exception as exc:
        log.warning(f"[BRAIN] scan summary bridge gagal: {exc}")
    _set_scan_state(last_candidate_count=candidate_count,last_eligible_count=eligible_count,last_ban_count=ban_count,last_low_confidence_count=low_conf_count,last_rejection_reasons=dict(rejection_reasons),last_symbols_processed=processed_symbols)
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
    avg_txt=f"{avg_conf:.1f}%" if avg_conf is not None else "belum tersedia (0 analisa strategy valid)"
    if mc.get('bullish_breadth_pct') is not None:
        breadth_txt=(f"📈 Breadth BUY <b>{mc['bullish_breadth_pct']:.1f}%</b> | SELL <b>{mc['bearish_breadth_pct']:.1f}%</b> | Regime: <b>{mc['market_regime']}</b>")
    elif market_rows:
        breadth_txt=(f"📈 Breadth BUY <b>—</b> | SELL <b>—</b> | Regime: <b>{mc.get('market_regime','unknown')}</b> | arah belum cukup untuk breadth")
    else:
        breadth_txt="📈 Market context: <b>belum tersedia</b>"
    rs_txt=(f"\n₿ BTC 1h: <b>{mc['btc_price_1h_pct']:+.2f}%</b> | BTC 4h: <b>{mc['btc_price_4h_pct']:+.2f}%</b>" if mc.get('btc_price_1h_pct') is not None else "")
    freq_txt = f"\n🎯 Candidate: <b>{candidate_count}</b> | Eligible: <b>{eligible_count}</b> | Low-conf ban: <b>{ban_count}</b>"
    reject_txt = ""
    if rejection_reasons:
        top_reject = ", ".join(f"{k}={v}" for k,v in sorted(rejection_reasons.items(), key=lambda kv:(-kv[1],kv[0]))[:4])
        reject_txt = f"\n🚫 Reject utama: <code>{html.escape(top_reject)}</code>"
    scan_meta=f"\n\n📊 Scan: <b>{TOP_N_COINS}</b> diminta | <b>{len(symbols)}</b> tersedia | <b>{processed_symbols}</b> diproses | <b>{analyzed_symbols}</b> analisa strategy valid\n🧠 Rata-rata confidence scan: <b>{avg_txt}</b>{freq_txt}{reject_txt}\n{breadth_txt}{rs_txt}"
    if warmup_active: scan_meta+=f"\n🛡️ Warmup reject: <b>{len(rejected_warmup)}</b> signal qualified dari scan ini ditolak"
    if not results:
        tg_send(chat_id,"⚠️ Tidak ada decision yang dinyatakan eligible oleh brain."+scan_meta); return []
    summary="\n".join(f"• {r.get('symbol','?')} {r.get('decision','?')} — {float(r.get('confidence',0) or 0):.0f}%" for r in results)
    tg_send(chat_id,f"✅ <b>{len(results)} decision brain eligible</b>\n\n{summary}{scan_meta}")
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
        _brain_on_trade(trade_record)
        # Backward-compatible 20-trade view for /backtest and existing UI.
        stats["pnl_history"].append(dict(trade_record))
        try:
            decision_payload={
                "total":int(stats.get("total",0) or 0),"tp":int(stats.get("tp",0) or 0),
                "sl":int(stats.get("sl",0) or 0),"trail":int(stats.get("trail",0) or 0),
                "balance":float(stats.get("balance",0) or 0),
                "recent":list(stats.get("pnl_history") or [])[-20:],
            }
            sd=_brain_on_stats_snapshot(decision_payload)
            if isinstance(sd,dict):
                trade_record["strategy_decision"]=sd
                stats["last_strategy_decision"]=dict(sd)
        except Exception as exc:
            log.warning(f"[STATS→BRAIN] decision bridge gagal: {exc}")

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
        t,tp,sl=stats["total"],stats["tp"],stats["sl"]
        trail,bal=stats.get("trail",0),stats["balance"]
        hist=list(stats["pnl_history"])
        last_dec=dict(stats.get("last_strategy_decision") or {})
    # Existing ledger must be evaluated too; otherwise /stats is only a report, not a brain feedback loop.
    try:
        sd=_brain_on_stats_snapshot({"total":t,"tp":tp,"sl":sl,"trail":trail,"balance":bal,"recent":hist[-20:]})
        if isinstance(sd,dict):
            last_dec=sd
            with stat_lock:
                stats["last_strategy_decision"]=dict(sd)
    except Exception as exc:
        log.debug(f"[STATS→BRAIN] fmt_stats evaluation gagal: {exc}")
    with trade_history_lock: full_hist=[dict(x) for x in trade_history]
    wins=tp+trail; wr=wins/(wins+sl)*100 if wins+sl>0 else 0.0
    base=STARTING_BALANCE if not REAL_TRADE_ENABLED else (real_balance_snapshot if real_balance_snapshot is not None else bal)
    pnl=bal-base; pnl_pct=(pnl/base*100) if base else 0.0
    def avg_res(k):
        vals=[]
        for h in full_hist:
            if str(h.get("result","")).lower()==k:
                try: vals.append(float(h.get("confidence")))
                except Exception: pass
        return sum(vals)/len(vals) if vals else None
    def fmt(v): return f"{v:.1f}%" if v is not None else "—"
    recent=[]
    for h in reversed(full_hist[-5:]):
        p=float(h.get("pnl_usd",0) or 0); icon="🟢" if p>0 else "🔴" if p<0 else "⚪"
        recent.append(f"{icon} {str(h.get('result','?')).upper()} {p:+.2f}% | {h.get('symbol','?')} | C{float(h.get('confidence',0) or 0):.0f}%")
    if not recent: recent=["—"]
    with ban_lock: banned_n=len(banned_coins)
    with early_reject_lock: reject_rem=early_reject_remaining
    lc=_low_conf_summary(); top_lc=", ".join(f"{x['symbol']} ({x['count']}x)" for x in lc[:3]) or "—"
    scan=get_last_scan_telemetry()
    mode="🔴 REAL" if REAL_TRADE_ENABLED else "🧪 SIMULASI"
    freq_state="—"; freq_action="—"
    try:
        bs=_brain_full_status()
        adaptive=bs.get("adaptive") if isinstance(bs,dict) else {}
        freq=adaptive.get("frequency") if isinstance(adaptive,dict) else {}
        freq_state=str(adaptive.get("frequency_state") or freq.get("status") or "—")
        fa=adaptive.get("last_frequency_action") if isinstance(adaptive,dict) else None
        if isinstance(fa,dict): freq_action=str(fa.get("action") or "—")
    except Exception: pass
    action=str(last_dec.get("action") or "WAITING_DATA")
    reason=str(last_dec.get("reason") or "—")
    proposal=str(last_dec.get("proposal") or "—")
    threshold=str(last_dec.get("active_threshold") or "—")
    freq=str(last_dec.get("frequency_action") or "—")
    source=str(scan.get("source") or "bybit_primary")
    return (
        f"📊 <b>STATISTIK</b> — {t} trade | ✅ TP {tp} | 🟢 Trail {trail} | 🔴 SL {sl}\n"
        f"Mode: <b>{mode}</b>\n"
        f"Win rate: <b>{wr:.1f}%</b> | Net: <b>{pnl:+.2f}%</b>\n"
        f"Saldo statistik: <b>${bal:.4f}</b>\n"
        f"Confidence closed: TP {fmt(avg_res('tp'))} | Trail {fmt(avg_res('trail'))} | SL {fmt(avg_res('sl'))}\n\n"
        f"🧠 <b>KEPUTUSAN OTAK</b>\n"
        f"Action: <b>{html.escape(action)}</b>\n"
        f"Reason: {html.escape(reason[:240])}\n"
        f"Proposal: <b>{html.escape(proposal[:200])}</b>\n"
        f"Threshold aktif: <b>{html.escape(threshold)}%</b> | Frequency state: <b>{html.escape(freq_state)}</b>\n"
        f"Frequency action: <b>{html.escape(freq_action)}</b>\n\n"
        f"5 terakhir:\n" + "\n".join(recent) + "\n\n"
        f"🚫 Ban: <b>{banned_n}</b> | Low-conf: <b>{html.escape(top_lc)}</b> | Early reject: <b>{reject_rem}</b>\n"
        f"🔎 Scan: <b>{_SCAN_STATE.get('cycle_count',0)}</b> cycle | status <b>{get_scanner_status().get('health')}</b> | last eligible <b>{_SCAN_STATE.get('last_result_count',0)}</b>\n"
        f"📡 Market data: <b>{html.escape(source)}</b> | Execution: <b>Binance</b> | Binance pause: <b>{"YA" if _binance_is_scan_paused() else "TIDAK"}</b>\n"
        f"Bybit REST: <b>{_bybit_request_count}</b> req | errors: <b>{_bybit_request_errors}</b>"
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
    scan_cols = ["scan_time","run_id","scan_counter","symbols_requested","symbols_available","symbols_processed","symbols_analyzed","failed_symbols","avg_confidence","min_confidence","max_confidence","low_confidence_count","below_threshold_count","qualified_count","early_rejected_count","cache_entries","cache_fresh","market_regime","bullish_breadth_pct","bearish_breadth_pct","neutral_breadth_pct","breadth_score","median_price_1h_pct","median_price_4h_pct","median_efficiency_4h","median_range_expansion_ratio","avg_relative_volume","btc_price_1h_pct","btc_price_4h_pct"]
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
        "machine_learning_state": _brain_get_champion(),
        "machine_learning_experience_count": int(_brain_get_experience_count()),
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
    sl_p = pos.get("initial_sl") or sig.get("sl")
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
    try:
        _transition_position_lifecycle(sym, "CLOSING", reason=f"finalizing {classified}")
    except Exception:
        pass

    # Remove local position exactly once. Exchange has already been confirmed flat by callers.
    with positions_lock:
        popped = positions.pop(sym, None)
        if popped is None:
            return False
        if not positions:
            active_trade = None
    _emit_execution_event("POSITION_LIFECYCLE", entity_id=pos.get("position_id") or sym, correlation_id=pos.get("position_id") or RUN_ID, payload={"symbol":sym,"from":"CLOSING","to":"CLOSED","reason":str(classified)}, persist=True)

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
        pm=last.get("price_move_pct")
        ap=last.get("account_impact_pct")
        sgn = "+" if float(last.get("pnl_usd", 0) or 0) >= 0 else ""
        detail = (
            f"Entry: <code>{last.get('entry', entry):.6g}</code> → Exit: <code>{last.get('exit_price', close_price):.6g}</code>\n"
            f"Pergerakan harga: <b>{float(pm):+.2f}%</b>\n" if pm is not None else
            f"Entry: <code>{last.get('entry', entry):.6g}</code> → Exit: <code>{last.get('exit_price', close_price):.6g}</code>\n")
        detail += (
            f"PnL posisi: <b>{float(last.get('pct', 0) or 0):+.2f}%</b> (${sgn}{float(last.get('pnl_usd', 0) or 0):.4f})\n"
            f"Dampak saldo statistik: <b>{float(ap):+.2f}%</b>\n\n" if ap is not None else
            f"PnL posisi: <b>{float(last.get('pct', 0) or 0):+.2f}%</b> (${sgn}{float(last.get('pnl_usd', 0) or 0):.4f})\n\n")
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
            price = get_price(sym, prefer_binance=True)
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
        with _binance_critical_context():
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
    if STOP_NEW_ENTRIES or CIRCUIT_BREAKER_OPEN: return
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
        pos.update({"entry":actual_entry,"entry_time":now,"status":"active","lifecycle":"PROTECTION_PENDING","trade_uid":f"{research_run_id}:{sym}:{int(now*1000)}",
                    "timeout_flag":False,"current_sl":sl,"initial_sl":sl,"execution_mode":"SIMULATION","position_id":_new_request_id("POS"),
                    "trail_update_count":0,"trail_applied_count":0,"trail_failed_count":0,"trail_queued_count":0,"first_trail_r":None,"last_trail_r":None,"max_protected_r":-999.0})
    try: _transition_position_lifecycle(sym,"MANAGED",reason="simulation protection state initialized")
    except Exception: pass
    tg_send(chat_id,f"⚡ <b>ENTRY {mode_label.upper()}</b> — {sym}\n"
                    f"Entry: <code>{actual_entry:.8g}</code>\n"
                    f"TP: <code>{tp:.8g}</code> | SL: <code>{sl:.8g}</code>")
    threading.Thread(target=monitor_position,args=(sym,pos),daemon=True).start()


# ============================================================
# REAL TRADE — alur pending order, monitoring posisi, auto-stop
# ============================================================

def _open_pending_real(sym,signal,chat_id):
    if STOP_NEW_ENTRIES or CIRCUIT_BREAKER_OPEN: return False
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
                        "timeout_flag":False,"status":"pending","lifecycle":"ENTRY_PENDING","execution_mode":"REAL","position_id":_new_request_id("POS")}
    try:
        with _binance_critical_context():
            _real_trade_preflight(force=False)
            avail,_=get_real_balance()
            if avail is not None and avail<MARGIN_USD: raise RuntimeError(f"saldo ${avail:.2f} < margin ${MARGIN_USD:.2f}")
            qty,margin,bumped=calc_auto_quantity(sym,entry,MARGIN_USD,LEVERAGE)
            if qty is None: raise RuntimeError("quantity di bawah minimum Binance")
            set_leverage_verified(sym,LEVERAGE); order=place_limit_order(sym,side,qty,entry)
        with positions_lock: positions[sym].update({"order_id":order["orderId"],"quantity":qty,"margin_used":margin})
        tg_send(chat_id,f"🎯 <b>PENDING ORDER REAL</b> — {sym}\n\n{fmt_signal_msg(signal)}")
        threading.Thread(target=_wait_entry_real,args=(sym,signal,chat_id,order["orderId"]),daemon=True).start()
        return True
    except BinanceCooldownError as e:
        with positions_lock:
            positions.pop(sym, None)
        log.warning(f"[entry] {sym} ditunda karena Binance budget/cooldown: {e}")
        tg_send(chat_id, f"⚠️ <b>ENTRY DITUNDA</b> — {sym}\n<code>{html.escape(str(e)[:300])}</code>\nSinyal tidak dianggap gagal dan tidak diban karena alasan API.")
        return False
    except BinanceUnknownExecutionError as e:
        # The entry POST may have reached Binance even though its response was lost.
        # Preserve the client order id so reconciliation can resolve the ambiguity.
        _force_position_emergency(sym, str(e)[:300])
        with positions_lock:
            if sym in positions:
                positions[sym]["entry_client_order_id"]=getattr(e, "client_order_id", None)
        tg_send(chat_id,f"🚨 <b>ENTRY STATUS UNKNOWN</b> — {sym}\n<code>{str(e)[:300]}</code>\nOrder tidak diulang secara buta. State dipertahankan untuk rekonsiliasi <code>/ok {sym}</code>.")
    except Exception as e:
        with positions_lock: positions.pop(sym,None)
        msg = str(e)
        # Infrastructure/account/API failures are not evidence that the symbol is bad.
        # Only known symbol-specific order validation errors may receive a short ban.
        symbol_specific = any(k in msg.lower() for k in (
            "precision", "min notional", "minimum quantity", "quantity", "invalid price",
            "-1111", "-1013", "-1121"
        ))
        if symbol_specific:
            _ban_coin(sym, f"order ditolak khusus simbol ({msg})")
        tg_send(chat_id,f"⚠️ <b>ENTRY GAGAL</b> — {sym}\n<code>{html.escape(msg[:300])}</code>\nSinyal tidak dihukum sebagai loss strategi.")
        return False



def _wait_entry_real(sym,signal,chat_id,order_id):
    deadline=time.time()+8*3600
    while time.time()<deadline:
        with positions_lock:
            if sym not in positions:return
            if positions[sym].get("timeout_flag"):
                try:
                    with _binance_critical_context():
                        cancel_order(sym,order_id)
                    time.sleep(0.2)
                    with _binance_critical_context():
                        st=get_order_status(sym,order_id)
                    if str(st.get("status","")).upper()=="FILLED":
                        actual=float(st.get("avgPrice") or 0) or signal["entry"]
                        _open_position_real(sym,signal,actual,chat_id,st)
                        return
                    positions.pop(sym,None); return
                except Exception as e:
                    _force_position_emergency(sym, str(e)[:300])
                    tg_send(chat_id, f"🚨 <b>ENTRY CANCEL BELUM TERKONFIRMASI</b> — {sym}\n<code>{str(e)[:300]}</code>\nPosisi tetap dipertahankan sampai <code>/ok {sym}</code>.")
                    return
        try:
            with _binance_critical_context():
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
        with _binance_critical_context():
            cancel_order(sym,order_id)
            st=get_order_status(sym,order_id)
        if str(st.get("status","")).upper()=="FILLED":
            actual=float(st.get("avgPrice") or 0) or signal["entry"]
            _open_position_real(sym,signal,actual,chat_id,st); return
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,"pending expired"); _record_pending_cancel("expired")
    except Exception as e:
        _force_position_emergency(sym, str(e)[:300])
        tg_send(chat_id, f"🚨 <b>PENDING ENTRY BELUM TERKONFIRMASI</b> — {sym}\n<code>{str(e)[:300]}</code>\nState tetap dipertahankan untuk <code>/ok {sym}</code>.")


def _emergency_close(sym, is_buy, qty, chat_id, reason):
    """Emergency flatten. Exchange confirmation is required before local close."""
    try:
        with _binance_critical_context():
            closed, exit_price = _verified_market_close(sym, is_buy, reason, chat_id=chat_id, max_retries=1)
        if not closed:
            raise RuntimeError("posisi belum terkonfirmasi flat")
        try:
            with _binance_critical_context():
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
        _force_position_emergency(sym, f"{reason}: {e}")
        _queue_pending_cleanup(sym, "auto-out cleanup", e)
        tg_send(chat_id, f"🚨 <b>GAGAL AUTO-OUT</b> — {sym}: {e}\n⚠️ Posisi TETAP dicatat di /trade. Jalankan <code>/ok {sym}</code> untuk rekonsiliasi Binance.")
        return False


def _open_position_real(sym,signal,actual_entry,chat_id,order_info):
    buy=signal["decision"]=="BUY"; sl=signal.get("sl"); tp=signal.get("tp")
    qty=abs(float(order_info.get("executedQty",0)))
    if not qty:
        with positions_lock: qty=(positions.get(sym) or {}).get("quantity",0)
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
                positions[sym].update({"entry": actual_entry, "entry_time": now, "status": "active", "lifecycle":"PROTECTION_PENDING", "trade_uid":f"{research_run_id}:{sym}:{int(now*1000)}", "current_sl": sl, "initial_sl": sl, "quantity": qty, "position_id": (positions.get(sym) or {}).get("position_id") or _new_request_id("POS"), "tp_order_id": None, "sl_order_id": None, "trail_update_count":0,"trail_applied_count":0,"trail_failed_count":0,"trail_queued_count":0,"first_trail_r":None,"last_trail_r":None,"max_protected_r":-999.0})
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
        positions[sym].update({"entry":actual_entry,"entry_time":now,"status":"active","lifecycle":"PROTECTION_PENDING","trade_uid":f"{research_run_id}:{sym}:{int(now*1000)}",
                               "current_sl":sl,"initial_sl":sl,"quantity":qty,"position_id":(positions.get(sym) or {}).get("position_id") or _new_request_id("POS"),"tp_order_id":t["algoId"],"sl_order_id":s["algoId"],
                               "execution_mode":"REAL","trail_update_count":0,"trail_applied_count":0,"trail_failed_count":0,"trail_queued_count":0,"first_trail_r":None,"last_trail_r":None,"max_protected_r":-999.0})
    try: _transition_position_lifecycle(sym,"MANAGED",reason="entry protection verified")
    except Exception: pass
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
                    with _binance_critical_context():
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
                        price = get_price(sym, prefer_binance=True)
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
                        price = get_price(sym, prefer_binance=True)
                    except Exception:
                        price = None
                    _finalize_external_close(sym, pos, reason_hint=reason, exit_price=price)
                    return
                with positions_lock:
                    if sym in positions:
                        positions[sym]["quantity"]=live

                px=get_price(sym, prefer_binance=True)
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
                                _force_position_emergency(sym, "strategy close cleanup gagal")
                                return
                            try:
                                with _binance_critical_context():
                                    _cleanup_algo_orders_verified(sym)
                            except Exception as ce:
                                _queue_pending_cleanup(sym, "strategy close cleanup", ce)
                                _force_position_emergency(sym, "strategy close cleanup gagal")
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
                            protection_request = ProtectionMutationRequest(pos.get("position_id"), sym, int(pos.get("protection_version",0) or 0), "REPLACE", expires_sec=20.0)
                            try:
                                _validate_protection_mutation(sym, protection_request.expected_version, protection_request.request_id)
                                if _binance_is_scan_paused():
                                    _queue_pending_trail(sym, candidate_sl, candidate_tp, live, reason="strategy", side=pos["signal"]["decision"])
                                    if candidate_sl != oldsl:
                                        _notify_trail_update(active_chat_id, sym, pos, upd, oldsl, candidate_sl, status="QUEUED")
                                else:
                                    # Refresh quantity from exchange immediately before protection mutation.
                                    with _binance_critical_context():
                                        latest = get_real_position(sym)
                                    live_qty = abs(float(latest.get("positionAmt",0) or 0)) if latest else 0.0
                                    if live_qty <= 0:
                                        continue
                                    # Cancel existing protection, then create+verify new pair. If creation
                                    # fails, restore the old pair before declaring an emergency.
                                    with _binance_critical_context():
                                        _cancel_all_algo_orders_verified(sym)
                                    try:
                                        with _binance_critical_context():
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
                                            _force_position_emergency(sym, str(protect_err)[:300])
                                            raise RuntimeError(f"trail update gagal dan protection lama tidak bisa dipulihkan: {protect_err}")
                                        raise
                                    _validate_protection_mutation(sym, protection_request.expected_version, protection_request.request_id)
                                    with positions_lock:
                                        if sym in positions:
                                            positions[sym]["current_sl"] = candidate_sl
                                            positions[sym]["protection_version"] = protection_request.expected_version + 1
                                            positions[sym]["protection_request_id"] = protection_request.request_id
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
                with _binance_critical_context():
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
                with _binance_critical_context():
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
                with _binance_critical_context():
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


def _mode_on_preflight_reconcile(chat_id=None):
    """One-time OFF→ON exchange reconciliation; clean orphan orders only."""
    with _binance_critical_context():
        _real_trade_preflight(force=True)
        remote_positions = get_real_positions_all()
        remote = {str(p.get("symbol")): p for p in remote_positions if p.get("symbol")}
        ordinary = get_open_orders_all()
        algo = get_open_algo_orders_all()
        orphan = set()
        for row in list(ordinary) + list(algo):
            sym = str(row.get("symbol") or "")
            if sym and sym not in remote:
                orphan.add(sym)
        for sym in sorted(orphan):
            _cancel_all_symbol_orders_verified(sym)
        with positions_lock:
            local_real = [(sym, dict(pos)) for sym, pos in positions.items() if _position_is_real(pos)]
        for sym, pos in local_real:
            live = remote.get(sym)
            if live is None or abs(float(live.get("positionAmt", 0) or 0)) <= 0:
                try:
                    _finalize_external_close(sym, pos, reason_hint="unknown", exit_price=pos.get("current_price") or pos.get("entry"))
                except Exception as e:
                    log.warning(f"[mode-on reconcile] finalize {sym} gagal: {e}")
    return {"remote_positions": len(remote_positions), "orphan_symbols_cleaned": len(orphan)}


def _set_scan_state(**updates):
    with _SCAN_STATE_LOCK:
        _SCAN_STATE.update(updates)
        return dict(_SCAN_STATE)

def get_scanner_status():
    with _SCAN_STATE_LOCK:
        out = dict(_SCAN_STATE)
    now=time.time()
    hb=float(out.get("coordinator_heartbeat_at") or 0.0)
    fin=float(out.get("last_finished_at") or 0.0)
    out["coordinator_heartbeat_age_sec"] = round(now-hb,1) if hb else None
    out["last_cycle_age_sec"] = round(now-fin,1) if fin else None
    out["thread_alive"] = bool(auto_thread is not None and auto_thread.is_alive())
    recent_hb = bool(hb and now-hb <= max(45.0, SCAN_MAX_DURATION_SEC*0.75))
    healthy = bool(out.get("enabled")) and bool(out.get("coordinator_alive")) and out["thread_alive"] and recent_hb
    out["health"] = "RUNNING" if healthy else ("STARTING" if out.get("enabled") else "STOPPED")
    out["binance_paused"] = _binance_is_scan_paused()
    out["market_data_source"] = "BYBIT_WS_PRIMARY"
    out["execution_exchange"] = "BINANCE"
    out["entry_mutations_blocked"] = bool(STOP_NEW_ENTRIES or CIRCUIT_BREAKER_OPEN or _binance_is_scan_paused())
    out["pause_remaining_sec"] = round(_binance_cooldown_remaining(), 2)
    out["heavy_workers"] = len(_heavy_worker_snapshot())
    out["light_workers"] = len(_light_worker_snapshot())
    out["top_coins_cached"] = len(_top_coins_cached_symbols)
    return out

def _scanner_thread_is_alive():
    t=auto_thread
    return bool(t is not None and t.is_alive())

def _ensure_scanner_running(chat_id, announce=False):
    global auto_mode, auto_thread, active_chat_id
    active_chat_id = chat_id or active_chat_id
    if _scanner_thread_is_alive():
        _set_scan_state(enabled=True, coordinator_alive=True, last_error=None)
        return auto_thread, False
    auto_mode=True
    _set_scan_state(enabled=True, coordinator_alive=False, last_error=None)
    _set_component_health("scanner", "STARTING", "scanner coordinator starting")
    t=threading.Thread(target=simulation_loop, args=(active_chat_id,), name="scanner-coordinator", daemon=True)
    auto_thread=t
    t.start()
    _SCAN_WAKE.set()
    if announce and active_chat_id:
        tg_send(active_chat_id, "🔎 <b>Scanner dimulai.</b>\nData analisis: <b>Bybit</b>\nExecution: <b>Binance</b>")
    return t, True

def _scanner_watchdog_loop():
    # Watchdog proves scanner liveness independently from the /auto flag.
    # It only supervises the coordinator; it does not create a second scanner.
    while not SHUTDOWN_EVENT.wait(10):
        try:
            if not auto_mode:
                continue
            t=auto_thread
            with _SCAN_STATE_LOCK:
                enabled=bool(_SCAN_STATE.get("enabled"))
                coordinator=bool(_SCAN_STATE.get("coordinator_alive"))
                hb=float(_SCAN_STATE.get("coordinator_heartbeat_at") or 0.0)
            stale=bool(hb and time.time()-hb > max(90.0, SCAN_MAX_DURATION_SEC+30))
            if t is None or not t.is_alive() or not coordinator or stale:
                log.error("[SCANNER WATCHDOG] scanner coordinator tidak sehat — restart terkontrol")
                _set_scan_state(last_error="scanner coordinator not healthy")
                try:
                    _ensure_scanner_running(active_chat_id, announce=True)
                    _set_component_health("scanner", "RECOVERING", "scanner coordinator restarted by watchdog")
                except Exception as exc:
                    _set_component_health("scanner", "DEGRADED", str(exc)[:250])
                    log.exception(f"[SCANNER WATCHDOG] restart gagal: {exc}")
        except Exception as exc:
            log.error(f"[SCANNER WATCHDOG] {exc}")

def simulation_loop(chat_id):
    """Long-lived scan coordinator. It never owns strategy logic and cannot deadlock on a rejected worker slot."""
    global auto_mode
    _set_scan_state(enabled=True, coordinator_alive=True, last_error=None)
    tg_send(chat_id, "🤖 <b>Engine dimulai.</b>\nStrategy mengendalikan Entry/TP/SL/Trail.")

    def wait_entry(sym, signal, chat_id):
        try:
            entry=float(signal["entry"]); buy=str(signal.get("decision")).upper()=="BUY"; deadline=time.time()+8*3600
            while time.time()<deadline and auto_mode:
                with positions_lock:
                    if sym not in positions: return
                    if positions[sym].get("timeout_flag"):
                        positions.pop(sym,None); return
                if _binance_is_scan_paused():
                    time.sleep(min(10.0, max(1.0, _binance_cooldown_remaining()))); continue
                price=get_price(sym)
                if price is not None and ((price<=entry) if buy else (price>=entry)):
                    fill=min(entry,price) if buy else max(entry,price)
                    _open_position(sym,signal,fill,chat_id,"strategy"); return
                time.sleep(MONITOR_SLEEP)
            with positions_lock:
                positions.pop(sym,None)
            _ban_coin(sym,"pending expired")
        except Exception as exc:
            log.exception(f"[ENTRY WAIT] {sym}: {exc}")

    def do_scan():
        _set_scan_state(cycle_running=True, last_started_at=time.time(), last_error=None)
        try:
            signals = run_scan_once(chat_id)
            _set_scan_state(last_result_count=len(signals or []))
            opened = 0
            if auto_mode and signals:
                entry_blocked = bool(_binance_is_scan_paused() or STOP_NEW_ENTRIES or CIRCUIT_BREAKER_OPEN)
                if entry_blocked:
                    log.info(f"[EXECUTION GATE] {len(signals)} eligible signals retained; Binance/new-entry gate blocked execution only")
                for sig in signals:
                    if not auto_mode: break
                    if _binance_is_scan_paused() or STOP_NEW_ENTRIES or CIRCUIT_BREAKER_OPEN: continue
                    sym=str(sig.get("symbol") or "").upper()
                    if not sym: continue
                    with positions_lock:
                        if sym in positions or len(positions)>=MAX_POSITIONS: continue
                    if REAL_TRADE_ENABLED:
                        if _open_pending_real(sym,sig,chat_id): opened += 1
                        continue
                    price=sig.get("price") or get_price(sym)
                    entry=sig.get("entry")
                    if price is None or entry is None: continue
                    mode=str(sig.get("execution_mode") or "").lower() or ("market" if sig.get("entry_label")=="market" else "limit")
                    with positions_lock:
                        if sym in positions or len(positions)>=MAX_POSITIONS: continue
                        positions[sym]={"signal":sig,"entry":entry,"chat_id":chat_id,"entry_time":None,"timeout_flag":False,"status":"pending","lifecycle":"ENTRY_PENDING","execution_mode":"SIMULATION"}
                    if mode=="market":
                        _open_position(sym,sig,get_price(sym) or price,chat_id,"strategy")
                    else:
                        tg_send(chat_id,f"🎯 <b>PENDING ORDER</b> — {sym}\n\n{fmt_signal_msg(sig)}")
                        _start_light_worker(f"entry-wait-{sym}", wait_entry, sym, sig, chat_id)
                    opened += 1
            log.info(f"[scan] {len(signals or [])} signal lolos, {opened} dikirim ke execution")
            _set_scan_state(last_success_at=time.time(), last_finished_at=time.time())
        except Exception as exc:
            _set_scan_state(last_error=str(exc)[:500], last_finished_at=time.time())
            _set_component_health("scanner", "DEGRADED", str(exc)[:250])
            log.exception(f"[SCAN CYCLE] gagal: {exc}")
        finally:
            _set_scan_state(cycle_running=False, cycle_count=_SCAN_STATE.get("cycle_count",0)+1)

    try:
        while auto_mode:
            _set_scan_state(coordinator_heartbeat_at=time.time(), coordinator_alive=True)
            # Binance cooldown/ban blocks new Binance entry mutation, not market analysis.
            # Full position capacity blocks new execution, NOT market analysis.
            # The scan continues so frequency/learning remain alive.
            with _SCAN_STATE_LOCK:
                running=bool(_SCAN_STATE.get("cycle_running"))
                last_finished=_SCAN_STATE.get("last_finished_at") or 0.0
            if running:
                _SCAN_WAKE.wait(1); _SCAN_WAKE.clear(); continue
            if time.time()-float(last_finished or 0.0) < 120:
                _SCAN_WAKE.wait(5); _SCAN_WAKE.clear(); continue
            worker=_start_heavy_worker("scan", do_scan)
            if worker is None:
                # Crucially: no permanent scanning=True flag when worker admission fails.
                _SCAN_WAKE.wait(2); _SCAN_WAKE.clear(); continue
            _SCAN_WAKE.wait(2); _SCAN_WAKE.clear()
    finally:
        _set_scan_state(enabled=False, coordinator_alive=False, cycle_running=False)
        tg_send(chat_id,"⏹ <b>Scanning dihentikan.</b>\n\n"+fmt_stats())


# ═════════════════════════════════════════════
# PESAN STATIS
# ═════════════════════════════════════════════
def get_start_msg():
    return (
        "👋 <b>SMC Signal Broadcaster</b>\n\n"
        f"Mode: <b>{'REAL TRADE' if REAL_TRADE_ENABLED else 'SIMULASI'}</b>\n"
        f"Posisi aktif: <b>{MAX_POSITIONS}</b> maksimum\n"
        f"Confidence policy: <b>{html.escape(str(_get_active_confidence_threshold()))}</b> (brain-owned)\n"
        "TP minimum: <b>2R</b> • Max RR: <b>Unlimited</b>\n"
        "Trailing: <b>Adaptive / context-aware</b>\n\n"
        "━━━━━━━━ <b>TRADING</b> ━━━━━━━━\n"
        "/auto                — Mulai scanning & trading\n"
        "/stop                — Hentikan scanning; posisi aktif tetap dipantau\n"
        "/trade               — Lihat semua posisi aktif/pending/emergency\n"
        "/ok SYMBOL           — Rekonsiliasi posisi Binance + TP/SL\n"
        "/timeout SYMBOL      — Tutup paksa posisi tertentu\n"
        "/timeout all          — Timeout semua posisi + semua order\n"
        "/timeout pending     — Timeout pending entry saja\n\n"
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
    global auto_mode, auto_thread, autostop_thread, active_chat_id, timeout_flag, MAX_POSITIONS, LEVERAGE, MARGIN_USD, AUTOSTOP_PCT, peak_real_balance, REAL_TRADE_ENABLED, BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_KEYS_PRESENT, real_balance_snapshot, real_balance_snapshot_at, early_reject_configured, early_reject_remaining, BAN_DURATION_SCANS, STOP_NEW_ENTRIES

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
                uid=(msg.get("from") or {}).get("id")
                chat_id=(msg.get("chat") or {}).get("id")
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
                    parts=text.split(); setter=_brain_fn("set_manual_confidence_threshold"); getter=_brain_fn("get_active_confidence_threshold")
                    if len(parts)==1:
                        try: current=getter() if callable(getter) else "brain-owned"
                        except Exception: current="brain-owned"
                        tg_send(chat_id,f"🎯 <b>Confidence policy:</b> <b>{html.escape(str(current))}</b> (owned by brain).")
                    elif callable(setter):
                        try:
                            token=parts[1].strip().lower()
                            if token == "auto":
                                auto_fn=_brain_fn("set_confidence_mode")
                                if callable(auto_fn): auto_fn("auto")
                                else: setter(None)
                                tg_send(chat_id,"✅ Confidence policy kembali ke <b>AUTO</b>. FULL boleh mengadaptasi frequency dengan guardrail.")
                            else:
                                val=float(token.replace("%",""));
                                if not 0<=val<=100: raise ValueError
                                setter(val); tg_send(chat_id,f"✅ Brain confidence policy diberi nilai manual <b>{val:.0f}%</b>.")
                        except Exception: tg_send(chat_id,"❌ Format salah. Gunakan <code>/confidence_min 70</code> atau <code>/confidence_min auto</code>.")
                    else: tg_send(chat_id,"⚠️ Brain tidak menyediakan kontrol confidence manual.")
                elif text in ("/info","info"):
                    tg_send(chat_id,get_info_msg())
                elif text in ("/ip","ip"):
                    ip = get_public_ip()
                    if ip and ip != "unknown":
                        tg_send(chat_id, f"🌐 <b>Public IP Render</b>\n<code>{html.escape(ip)}</code>\n\nGunakan IP ini untuk whitelist Binance API jika diperlukan.")
                    else:
                        tg_send(chat_id, "⚠️ Public IP Render tidak berhasil diambil dari layanan IP eksternal saat ini.")
                elif text in ("/status","status"):
                    tg_send(chat_id,fmt_runtime_status_v110())
                elif text in ("/stats","stats"):
                    tg_send(chat_id,fmt_stats())
                elif text in ("/backtest","backtest"):
                    tg_send(chat_id,fmt_backtest())
                # ============================================================
                # TAMBAHAN BARU (START) — Handler /analyze
                # ============================================================
                elif text in ("/full on", "full on"):
                    try:
                        tg_send(chat_id, _full_strategy_command("on", chat_id))
                    except Exception as e:
                        log.exception(f"[full] on gagal: {e}")
                        tg_send(chat_id, f"❌ FULL gagal diaktifkan: <code>{html.escape(str(e)[:300])}</code>")
                elif text in ("/full off", "full off"):
                    try:
                        tg_send(chat_id, _full_strategy_command("off", chat_id))
                    except Exception as e:
                        log.exception(f"[full] off gagal: {e}")
                        tg_send(chat_id, f"❌ FULL gagal dimatikan: <code>{html.escape(str(e)[:300])}</code>")
                elif text in ("/full reset", "full reset"):
                    try:
                        tg_send(chat_id, _full_strategy_command("reset", chat_id))
                    except Exception as e:
                        log.exception(f"[full] reset gagal: {e}")
                        tg_send(chat_id, f"❌ FULL reset gagal: <code>{html.escape(str(e)[:300])}</code>")
                elif text in ("/full", "full", "/full status", "full status"):
                    try:
                        tg_send(chat_id, _v110_full_text())
                    except Exception as e:
                        log.exception(f"[full] status gagal: {e}")
                        tg_send(chat_id, f"❌ FULL status gagal: <code>{html.escape(str(e)[:300])}</code>")
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

                    _start_heavy_worker("analyze", _run_analyze, chat_id)
                    tg_send(chat_id, "⏳ /analyze berjalan di background berdasarkan history trade. Bot tetap menerima perintah lain.")
# ============================================================
# TAMBAHAN BARU (END)
# ============================================================
                # ============================================================
                # FINAL /GANTI — strict brain hot-swap; no silent fallback.
                # ============================================================
                elif text in ("/ganti", "ganti"):
                    doc = msg.get("document")
                    if not doc:
                        tg_send(chat_id, "📤 Kirim file strategy_logic.py sebagai dokumen dengan caption /ganti")
                        continue
                    file_name = str(doc.get("file_name") or "")
                    if not file_name.endswith(".py"):
                        tg_send(chat_id, "❌ Strategy brain harus file .py")
                        continue
                    try:
                        file_id = doc["file_id"]
                        file_info = requests.get(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile",
                            params={"file_id": file_id}, timeout=10
                        ).json()
                        file_path = file_info["result"]["file_path"]
                        downloaded = requests.get(
                            f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}",
                            timeout=10
                        )
                        downloaded.raise_for_status()
                        file_content = downloaded.text

                        # Stage in a temporary module namespace. Never mutate the active brain first.
                        import ast as _ast, tempfile, pathlib as _pathlib, importlib.util as _importlib_util
                        try:
                            _ast.parse(file_content, filename="strategy_logic.py")
                            compile(file_content, "strategy_logic.py", "exec")
                        except SyntaxError as exc:
                            tg_send(chat_id, f"❌ Brain ditolak — syntax error:\n<code>{html.escape(str(exc)[:500])}</code>")
                            continue

                        stage_path = _pathlib.Path(tempfile.mkdtemp(prefix="brain-stage-") ) / "strategy_logic.py"
                        stage_path.write_text(file_content, encoding="utf-8")
                        spec = _importlib_util.spec_from_file_location("_strategy_candidate", stage_path)
                        candidate = _importlib_util.module_from_spec(spec)
                        spec.loader.exec_module(candidate)

                        ok, detail = _validate_brain_contract(candidate)
                        if not ok:
                            tg_send(chat_id, f"❌ Brain ditolak — contract gagal:\n<code>{html.escape(detail[:500])}</code>")
                            continue

                        # Require the mandatory decision/management contract. Optional APIs may evolve freely.
                        required = ("full_analyze", "manage_position")
                        if any(not callable(getattr(candidate, name, None)) for name in required):
                            tg_send(chat_id, "❌ Brain ditolak — contract mandatory tidak lengkap.")
                            continue

                        # Validate serialization-safe descriptor before committing.
                        descriptor = {
                            "brain_interface_version": getattr(candidate, "BRAIN_INTERFACE_VERSION", BRAIN_INTERFACE_VERSION),
                            "module_name": "strategy_logic",
                            "required": list(required),
                        }
                        json.dumps(descriptor, ensure_ascii=False, allow_nan=False)

                        # Commit locally only after validation. GitHub commit follows successful local stage.
                        local_path = Path(os.path.dirname(os.path.abspath(__file__))) / "strategy_logic.py"
                        backup_path = local_path.with_suffix(".py.previous")
                        previous = local_path.read_text(encoding="utf-8") if local_path.exists() else None
                        if previous is not None:
                            backup_path.write_text(previous, encoding="utf-8")
                        local_path.write_text(file_content, encoding="utf-8")
                        try:
                            _commit_to_github(file_content, "strategy_logic.py", "Update strategy_logic via Telegram /ganti")
                        except Exception:
                            if previous is not None:
                                local_path.write_text(previous, encoding="utf-8")
                            else:
                                try: local_path.unlink()
                                except Exception: pass
                            raise

                        # Only now activate the candidate as the new brain; no partial function binding.
                        import importlib, sys as _sys
                        if "strategy_logic" in _sys.modules:
                            del _sys.modules["strategy_logic"]
                        import strategy_logic as new_brain
                        ok2, detail2 = _validate_brain_contract(new_brain)
                        if not ok2:
                            # Restore previous local brain if possible; exchange state remains untouched.
                            if previous is not None:
                                local_path.write_text(previous, encoding="utf-8")
                            raise RuntimeError(f"post-commit brain reload contract failed: {detail2}")
                        globals()["_brain"] = new_brain
                        globals()["_STRATEGY_LOAD_ERROR"] = None
                        log.info("[BRAIN] strict hot-swap activated")
                        tg_send(chat_id, "✅ <b>Brain baru aktif.</b> Contract, load, dan post-load validation berhasil. Tidak ada fallback fungsi lama.")
                    except Exception as e:
                        log.exception(f"[ganti] strict hot-swap gagal: {e}")
                        tg_send(chat_id, f"❌ <b>Brain tidak diganti.</b>\n<code>{html.escape(str(e)[:500])}</code>")
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
                elif text in ("/save", "save"):
                    try:
                        cid, local = _save_runtime_checkpoint(push_github=True)
                        tg_send(chat_id, f"💾 <b>SAVE BERHASIL</b>\nCheckpoint: <code>{html.escape(cid)}</code>\nMemory/strategy tetap utuh.")
                    except Exception as e:
                        log.exception(f"[save] gagal: {e}")
                        tg_send(chat_id, f"❌ <b>/save gagal</b>\n<code>{html.escape(str(e)[:300])}</code>")
                elif text in ("/open", "open") or text.startswith("/open ") or text.startswith("open "):
                    parts=text.split(maxsplit=2)
                    if len(parts)==1:
                        tg_send(chat_id, "⚠️ <b>/open akan mengganti state lokal yang kompatibel.</b>\nREAL position tetap mengikuti Binance.\nKonfirmasi: <code>/open confirm</code> atau rollback: <code>/open previous</code>")
                    elif len(parts)==2 and parts[1].strip() in {"confirm","previous"}:
                        try:
                            STOP_NEW_ENTRIES=True
                            cp=_load_runtime_checkpoint("previous" if parts[1].strip()=="previous" else None)
                            cid=_restore_runtime_checkpoint(cp)
                            STOP_NEW_ENTRIES=False
                            tg_send(chat_id, f"✅ <b>OPEN BERHASIL</b>\nCheckpoint: <code>{html.escape(str(cid))}</code>\nREAL state direkonsiliasi ulang dengan Binance.")
                        except Exception as e:
                            STOP_NEW_ENTRIES=False
                            log.exception(f"[open] gagal: {e}")
                            tg_send(chat_id, f"❌ <b>/open gagal</b>\n<code>{html.escape(str(e)[:300])}</code>")
                    else:
                        tg_send(chat_id, "❌ Konfirmasi salah. Gunakan <code>/open confirm</code>.")
                elif text in ("/resetbalance", "resetbalance"):
                    if REAL_TRADE_ENABLED:
                        tg_send(chat_id, "⚠️ <b>RESET REAL BALANCE STATISTICS</b>\nIni tidak mengubah saldo Binance, posisi, atau order.\nKonfirmasi: <code>/resetbalance confirm</code>")
                    else:
                        with stat_lock:
                            stats["balance"] = STARTING_BALANCE
                        tg_send(chat_id, "✅ <b>Balance simulasi di-reset ke $10.00</b>\nStats, learning, strategy, dan trade evidence tetap utuh.")
                elif text in ("/resetbalance confirm", "resetbalance confirm"):
                    if not REAL_TRADE_ENABLED:
                        with stat_lock:
                            stats["balance"] = STARTING_BALANCE
                        tg_send(chat_id, "✅ <b>Balance simulasi di-reset ke $10.00</b>\nStats dan learning tetap utuh.")
                    else:
                        try:
                            with _binance_critical_context():
                                _, total = get_real_balance()
                            if total is None: raise RuntimeError("saldo Binance tidak tersedia")
                            with stat_lock:
                                stats["balance"] = float(total)
                            with real_balance_lock:
                                real_balance_snapshot=float(total); real_balance_snapshot_at=time.time()
                            tg_send(chat_id, f"✅ <b>Real statistics anchor diperbarui</b>\nSaldo Binance terbaru: <b>${float(total):.4f}</b>\nTidak ada order/posisi yang diubah. Learning tetap utuh.")
                        except Exception as e:
                            tg_send(chat_id, f"❌ <b>/resetbalance gagal</b>\n<code>{html.escape(str(e)[:300])}</code>")
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
                    if REAL_TRADE_ENABLED:
                        try:
                            _, total = get_real_balance()
                            with autostop_lock:
                                peak_real_balance = total
                        except Exception:
                            pass
                    try:
                        t, created = _ensure_scanner_running(chat_id, announce=False)
                        st=get_scanner_status()
                        if created:
                            tg_send(chat_id, "🔎 <b>SCANNER STARTING</b>\nData realtime: <b>Bybit WS</b>\nBackfill: <b>Bybit REST</b>\nExecution: <b>Binance</b>")
                        else:
                            tg_send(chat_id, "🔎 <b>SCANNER SUDAH AKTIF</b>\n"
                                            f"Cycle: <b>{st.get('cycle_count',0)}</b> | "
                                            f"last scan: <b>{st.get('last_cycle_age_sec','—')}s lalu</b>\n"
                                            f"Health: <b>{st.get('health')}</b> | Bybit: <b>AKTIF</b>")
                    except Exception as exc:
                        tg_send(chat_id, f"❌ <b>Scanner gagal dimulai</b>\n<code>{html.escape(str(exc)[:300])}</code>")
                elif text in ("/stop","stop"):
                    # /stop hanya mematikan scanning sinyal baru — posisi
                    # yang sudah berjalan tetap dipantau sampai TP/SL alami.
                    if auto_mode:
                        auto_mode = False
                        _SCAN_WAKE.set()
                        with positions_lock:
                            n_active = len(positions)
                        tg_send(chat_id,
                            f"⏹ <b>Scanning dihentikan.</b>\n"
                            f"Posisi aktif ({n_active}) tetap dipantau sampai TP/SL.\n"
                            f"Pakai /timeout SYMBOL kalau mau tutup paksa.")
                    else:
                        st=get_scanner_status()
                        tg_send(chat_id,"ℹ️ <b>Scanner tidak berjalan.</b>\n"
                                        f"Health: <b>{st.get('health')}</b>")
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
                                pr       = ws_feed.get_price(s) or get_price(s) or p["entry"]
                                dist_pct = abs(p["entry"] - pr) / pr * 100
                                lines.append(
                                    f"\n⏳ <b>{s}</b> — PENDING\n"
                                    f"{em} {sig['decision']} | Entry zone: <code>{p['entry']:.6g}</code>\n"
                                    f"Harga kini: <code>{pr:.6g}</code> | Jarak: {dist_pct:.2f}%\n"
                                    f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{sig['sl']:.6g}</code> | Confidence: <b>{float(sig.get('confidence', 0) or 0):.0f}%</b>"
                                )
                            else:
                                pr  = ws_feed.get_price(s) or get_price(s) or p["entry"]
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
                                    positions[sym]["quantity"] = live_qty
                                    positions[sym]["lifecycle"] = "RECONCILING"
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
                            _force_position_emergency(sym, str(e)[:300])
                            _queue_pending_cleanup(sym, "/ok gagal — retry manual", e)
                            tg_send(cid, f"🚨 <b>{sym} RECONCILE GAGAL</b>\n<code>{str(e)[:350]}</code>\nPosisi tetap dipertahankan di /trade. Coba <code>/ok {sym}</code> lagi setelah Binance/API normal.")
                    threading.Thread(target=_run_ok, args=(chat_id, target), daemon=True).start()
                elif text.startswith("/timeout") or (not text.startswith("/") and text.startswith("timeout")):
                    parts = text.split()
                    if len(parts) > 1 and parts[1].lower() == "pending":
                        threading.Thread(target=_verified_timeout_pending_only, args=(chat_id,), daemon=True).start()
                    elif len(parts) > 1 and parts[1].lower() == "all":
                        threading.Thread(target=_verified_timeout_all, args=(chat_id,), daemon=True).start()
                    else:
                        target_sym = parts[1].upper() if len(parts) > 1 else None
                        if target_sym:
                            tg_send(chat_id, f"⏳ <b>TIMEOUT REQUESTED</b> — {target_sym}\nMembatalkan order/menutup posisi untuk symbol tersebut…")
                            threading.Thread(target=_verified_timeout_symbol, args=(target_sym, chat_id), daemon=True).start()
                        else:
                            tg_send(chat_id, "⛔ Gunakan <code>/timeout pending</code> untuk pending entry saja, atau <code>/timeout SYMBOL</code> untuk satu symbol.")
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
                        # One-time exchange preflight/cleanup, then one balance snapshot.
                        try:
                            reconcile_meta = _mode_on_preflight_reconcile(chat_id)
                            with _binance_critical_context():
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
                            tg_send(chat_id, f"🔴 <b>Mode REAL TRADE diaktifkan.</b>\nBalance Binance snapshot: <b>${float(total):.4f}</b>\nOrphan symbols dibersihkan: <b>{int(reconcile_meta.get('orphan_symbols_cleaned', 0))}</b>.\nSnapshot dibuat sekali pada transisi ini.{extra}")
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



# Final runtime authority alias. All mutation helpers resolve through this symbol.
def _binance_signed(method, path, params=None, critical=False):
    method_u=str(method).upper()
    if method_u in ExecutionController.MUTATIONS:
        return _execution_controller.submit_signed(method_u,path,params=params,critical=critical)
    return _binance_signed_impl(method_u,path,params=params,critical=critical)


def _handle_shutdown(signum, frame):
    try:
        _graceful_shutdown(f"signal={signum}")
    except Exception as exc:
        log.critical(f"[SHUTDOWN] cleanup gagal: {exc}")
        SHUTDOWN_EVENT.set()

if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _handle_shutdown)
if hasattr(signal, "SIGINT"):
    signal.signal(signal.SIGINT, _handle_shutdown)


def start_runtime():
    global STOP_NEW_ENTRIES
    _set_runtime_state("BOOTING", "runtime start")
    STOP_NEW_ENTRIES=True
    health=_bootstrap_validate_and_reconcile()
    # Start non-entry infrastructure only after deterministic preflight.
    try:
        ws_feed.start(); _set_component_health("binance_websocket","HEALTHY","websocket started")
    except Exception as exc:
        _set_component_health("binance_websocket","DEGRADED",str(exc))
    threading.Thread(target=_price_cache_loop,name="price-cache",daemon=True).start()
    threading.Thread(target=_binance_recovery_loop,name="binance-recovery",daemon=True).start()
    threading.Thread(target=_scanner_watchdog_loop,name="scanner-watchdog",daemon=True).start()
    threading.Thread(target=_render_keepalive_loop,name="render-health",daemon=True).start()
    threading.Thread(target=run_flask,name="http",daemon=True).start()
    threading.Thread(target=bot_loop,name="telegram-runtime",daemon=True).start()
    _set_component_health("scanner","STOPPED","scanner menunggu /auto; watchdog siap memantau")
    if _brain is None or not _validate_brain_contract(_brain)[0]:
        STOP_NEW_ENTRIES=True
        if RUNTIME_STATE=="BOOTING": _set_runtime_state("DEGRADED","brain contract unavailable")
    elif RUNTIME_STATE=="BOOTING":
        STOP_NEW_ENTRIES=False
        _set_runtime_state("READY","deterministic startup checks completed")
    log.info(f"[ENGINE] {EXECUTION_ENGINE_VERSION} run={RUN_ID} health={health['overall']}")
    if _STRATEGY_LOAD_ERROR and ALLOWED_USER_ID:
        tg_send(ALLOWED_USER_ID,f"🚨 <b>strategy_logic.py bermasalah</b>\n<code>{html.escape(_STRATEGY_LOAD_ERROR[:500])}</code>\nNew entry ditahan.")



# ============================================================================
# FINAL V100 HARDENING OVERLAY
# Bybit WS primary market data + Binance WS user-state + REST governor,
# trail breach latch, order cleanup, pending-only timeout, PnL normalization,
# Telegram spam suppression, and explicit scanner/execution separation.
# ============================================================================
from collections import defaultdict

FINAL_BODY_VERSION = "MAIN-BODY-V100-BYBIT-WS-BINANCE-REST-GOVERNOR"
BYBIT_WS_URL = os.getenv("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/linear")
BYBIT_WS_STALE_SEC = max(5.0, float(os.getenv("BYBIT_WS_STALE_SEC", "15")))
BYBIT_WS_RECONNECT_MAX = max(5.0, float(os.getenv("BYBIT_WS_RECONNECT_MAX", "30")))
BYBIT_WS_MAX_TOPICS_PER_SUB = max(10, int(os.getenv("BYBIT_WS_MAX_TOPICS_PER_SUB", "50")))
BYBIT_WS_TOPIC_INTERVALS = ("15", "60", "D")
BYBIT_WS_PREWARM_SYMBOLS = max(0, int(os.getenv("BYBIT_WS_PREWARM_SYMBOLS", "50")))
PENDING_ENTRY_TIMEOUT_SEC = max(60.0, float(os.getenv("PENDING_ENTRY_TIMEOUT_SEC", "3600")))
BINANCE_WS_STALE_SEC = max(15.0, float(os.getenv("BINANCE_WS_STALE_SEC", "90")))
BINANCE_REST_NONCRITICAL_MIN_INTERVAL = max(0.5, float(os.getenv("BINANCE_REST_NONCRITICAL_MIN_INTERVAL", "2.0")))
BINANCE_REST_RECONCILE_MIN_INTERVAL = max(5.0, float(os.getenv("BINANCE_REST_RECONCILE_MIN_INTERVAL", "30")))
BINANCE_BULK_RECONCILE_MIN_INTERVAL = max(30.0, float(os.getenv("BINANCE_BULK_RECONCILE_MIN_INTERVAL", "300")))
TELEGRAM_REPEAT_SUPPRESS_SEC = max(30.0, float(os.getenv("TELEGRAM_REPEAT_SUPPRESS_SEC", "180")))
TELEGRAM_FLOOD_WINDOW_SEC = max(15.0, float(os.getenv("TELEGRAM_FLOOD_WINDOW_SEC", "60")))
TELEGRAM_FLOOD_MAX = max(3, int(os.getenv("TELEGRAM_FLOOD_MAX", "6")))
SL_BREACH_EPSILON_PCT = max(0.0, float(os.getenv("SL_BREACH_EPSILON_PCT", "0.0")))

# ---------- Bybit realtime market bus ----------
class BybitMarketWS:
    def __init__(self):
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._ws = None
        self._connected = False
        self._last_msg_at = 0.0
        self._last_error = None
        self._backoff = 1.0
        self._stop = threading.Event()
        self._topics = set()
        self._tickers = {}
        self._klines = {}  # (symbol, interval) -> deque rows
        self._seq = {}
        self._thread = None

    def start(self):
        if not _WS_LIB_OK:
            _set_component_health("bybit_websocket", "DEGRADED", "websocket-client belum terpasang")
            return None
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self._thread
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="bybit-market-ws", daemon=True)
            self._thread.start()
            return self._thread

    def stop(self):
        self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def is_fresh(self):
        with self._lock:
            return self._connected and self._last_msg_at > 0 and (time.time() - self._last_msg_at) <= BYBIT_WS_STALE_SEC

    def status(self):
        with self._lock:
            return {
                "connected": bool(self._connected),
                "thread_alive": bool(self._thread and self._thread.is_alive()),
                "fresh": bool(self._connected and self._last_msg_at and time.time()-self._last_msg_at <= BYBIT_WS_STALE_SEC),
                "last_msg_at": self._last_msg_at,
                "last_error": self._last_error,
                "topics": len(self._topics),
                "tickers": len(self._tickers),
                "kline_buffers": len(self._klines),
            }

    def subscribe_symbols(self, symbols):
        syms = [str(s).upper() for s in symbols if s]
        topics = []
        for sym in syms:
            topics.append(f"tickers.{sym}")
            for iv in BYBIT_WS_TOPIC_INTERVALS:
                topics.append(f"kline.{iv}.{sym}")
        with self._lock:
            new_topics = [t for t in topics if t not in self._topics]
        if not new_topics:
            return
        self._send_topics("subscribe", new_topics)

    def unsubscribe_symbols(self, symbols):
        topics=[]
        for sym in [str(s).upper() for s in symbols if s]:
            topics.append(f"tickers.{sym}")
            for iv in BYBIT_WS_TOPIC_INTERVALS:
                topics.append(f"kline.{iv}.{sym}")
        self._send_topics("unsubscribe", topics)

    def _send_topics(self, op, topics):
        if not topics:
            return
        for i in range(0, len(topics), BYBIT_WS_MAX_TOPICS_PER_SUB):
            chunk=topics[i:i+BYBIT_WS_MAX_TOPICS_PER_SUB]
            try:
                with self._send_lock:
                    ws=self._ws
                    if ws is None or not self._connected:
                        # Keep desired topics; on reconnect _on_open resubscribes.
                        if op == "subscribe":
                            with self._lock: self._topics.update(chunk)
                        elif op == "unsubscribe":
                            with self._lock: self._topics.difference_update(chunk)
                        continue
                    ws.send(json.dumps({"op":op,"args":chunk}))
                with self._lock:
                    if op == "subscribe": self._topics.update(chunk)
                    else: self._topics.difference_update(chunk)
            except Exception as exc:
                self._last_error=str(exc)[:300]
                log.warning(f"[BYBIT WS] {op} gagal: {exc}")

    def get_ticker(self, symbol):
        with self._lock:
            row=self._tickers.get(str(symbol).upper())
            return dict(row) if row else None


    def get_price(self, symbol):
        row=self.get_ticker(symbol)
        return row.get("price") if row else None

    def get_prices(self):
        with self._lock:
            return {k:dict(v) for k,v in self._tickers.items()}

    def get_klines(self, symbol, interval, limit=250):
        key=(str(symbol).upper(), str(interval))
        with self._lock:
            buf=self._klines.get(key)
            if not buf:
                return pd.DataFrame()
            rows=list(buf)[-int(limit):]
        if not rows: return pd.DataFrame()
        df=pd.DataFrame(rows)
        df.index=pd.to_datetime(df["t"], unit="ms")
        df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
        return df[["open","high","low","close","volume"]]

    def seed(self, symbol, interval, df):
        if df is None or df.empty: return
        key=(str(symbol).upper(), str(interval))
        rows=[]
        for idx,row in df.tail(300).iterrows():
            try:
                rows.append({"t":int(pd.Timestamp(idx).timestamp()*1000),"o":float(row["open"]),"h":float(row["high"]),"l":float(row["low"]),"c":float(row["close"]),"v":float(row["volume"])})
            except Exception:
                continue
        if rows:
            with self._lock:
                self._klines[key]=deque(rows,maxlen=300)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._ws=websocket.WebSocketApp(
                    BYBIT_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._last_error=str(exc)[:300]
                log.warning(f"[BYBIT WS] connection exception: {exc}")
            finally:
                with self._lock: self._connected=False
            if self._stop.is_set(): break
            time.sleep(self._backoff)
            self._backoff=min(BYBIT_WS_RECONNECT_MAX,self._backoff*2)

    def _on_open(self, ws):
        with self._lock:
            self._connected=True; self._last_msg_at=time.time(); self._backoff=1.0; topics=list(self._topics)
        _set_component_health("bybit_websocket","HEALTHY","public market WS connected")
        for i in range(0,len(topics),BYBIT_WS_MAX_TOPICS_PER_SUB):
            self._send_topics("subscribe", topics[i:i+BYBIT_WS_MAX_TOPICS_PER_SUB])

    def _on_error(self, ws, error):
        with self._lock: self._last_error=str(error)[:300]
        _set_component_health("bybit_websocket","DEGRADED",str(error)[:250])

    def _on_close(self, ws, code, msg):
        with self._lock: self._connected=False
        log.warning(f"[BYBIT WS] closed code={code} msg={msg}")

    def _on_message(self, ws, raw):
        now=time.time()
        try: msg=json.loads(raw)
        except Exception: return
        with self._lock: self._last_msg_at=now
        topic=str(msg.get("topic") or "") if isinstance(msg,dict) else ""
        if topic.startswith("tickers."):
            data=msg.get("data") or {}
            if isinstance(data,dict): data=[data]
            with self._lock:
                for item in data:
                    try:
                        sym=str(item.get("symbol") or "").upper()
                        if not sym: continue
                        prev=dict(self._tickers.get(sym) or {})
                        price=item.get("lastPrice") or item.get("markPrice")
                        volume=item.get("turnover24h") or item.get("volume24h")
                        chg=item.get("price24hPcnt")
                        row={"symbol":sym,"price":float(price) if price is not None else float(prev.get("price",0.0) or 0.0),"ts":float(msg.get("ts") or now*1000)/1000.0,"recv_at":now,"volume":float(volume) if volume is not None else float(prev.get("volume",0.0) or 0.0),"change_24h":float(chg) if chg is not None else float(prev.get("change_24h",0.0) or 0.0)}
                        row["cross_seq"]=msg.get("cs") or prev.get("cross_seq")
                        self._tickers[sym]=row
                        if msg.get("cs") is not None: self._seq[sym]=int(msg.get("cs"))
                    except Exception: continue
        elif topic.startswith("kline."):
            # kline.15.BTCUSDT / kline.60.BTCUSDT / kline.D.BTCUSDT
            parts=topic.split(".")
            if len(parts)!=3: return
            interval,sym=parts[1],parts[2].upper()
            data=msg.get("data") or []
            if isinstance(data,dict): data=[data]
            with self._lock:
                buf=self._klines.get((sym,interval))
                if buf is None: buf=deque(maxlen=300); self._klines[(sym,interval)]=buf
                for item in data:
                    try:
                        row={"t":int(item.get("start") or item.get("timestamp") or 0),"o":float(item["open"]),"h":float(item["high"]),"l":float(item["low"]),"c":float(item["close"]),"v":float(item.get("volume") or 0.0)}
                        if not row["t"]: continue
                        if buf and buf[-1]["t"]==row["t"]: buf[-1]=row
                        else: buf.append(row)
                    except Exception: continue

bybit_market_ws = BybitMarketWS()

# ---------- REST market-data wrappers become WS-first ----------
_ORIG_BYBIT_KLINES = globals().get("_bybit_klines")
_ORIG_BYBIT_PRICE = globals().get("_bybit_price")
_ORIG_BYBIT_TOP = globals().get("_bybit_top_coins")

# REST backfill locks prevent 50 parallel first-use requests.
_bybit_backfill_locks = {}
_bybit_backfill_guard = threading.Lock()

def _bybit_backfill_once(symbol, interval, limit=250):
    key=(str(symbol).upper(),str(interval))
    with _bybit_backfill_guard:
        lk=_bybit_backfill_locks.get(key)
        if lk is None: lk=threading.Lock(); _bybit_backfill_locks[key]=lk
    with lk:
        existing=bybit_market_ws.get_klines(*key,limit=limit)
        if len(existing)>=min(int(limit),40): return existing
        df=pd.DataFrame()
        try:
            if callable(_ORIG_BYBIT_KLINES): df=_ORIG_BYBIT_KLINES(symbol,interval,limit)
        except Exception as exc:
            log.warning(f"[BYBIT REST BACKFILL] {symbol} {interval}: {exc}")
        if df is not None and not df.empty:
            bybit_market_ws.seed(symbol, INTERVAL_MAP.get(interval,interval), df)
            bybit_market_ws.subscribe_symbols([symbol])
        return df if df is not None else pd.DataFrame()

def _final_bybit_price(symbol):
    sym=str(symbol).upper(); now=time.time()
    try: bybit_market_ws.subscribe_symbols([sym])
    except Exception: pass
    wsrow=bybit_market_ws.get_ticker(sym)
    if wsrow and wsrow.get("price") and now-float(wsrow.get("recv_at",0))<=BYBIT_WS_STALE_SEC:
        return float(wsrow["price"])
    # REST fallback is throttled per symbol so a disconnected WS cannot create a request storm.
    with _local_price_lock:
        cached=_local_price_cache.get(sym)
    if cached and now-cached[1] < 10.0:
        return float(cached[0])
    if callable(_ORIG_BYBIT_PRICE):
        try:
            price=float(_ORIG_BYBIT_PRICE(sym))
            with _local_price_lock: _local_price_cache[sym]=(price,now)
            return price
        except Exception: pass
    return None

def _final_bybit_top(exclude_syms):
    prices=bybit_market_ws.get_prices()
    if prices:
        exc=set(exclude_syms or ())
        arr=[]
        for s,row in prices.items():
            try:
                if not s.endswith("USDT") or s in exc: continue
                p=float(row.get("price",0)); vol=float(row.get("volume",0)); chg=abs(float(row.get("change_24h",0.0) or 0.0))
                if 0.0001<p<MAX_PRICE and vol>5_000_000 and chg<0.15: arr.append((s,vol))
            except Exception: continue
        if arr:
            arr.sort(key=lambda x:x[1],reverse=True)
            return [s for s,_ in arr[:TOP_N_COINS]]
    return _ORIG_BYBIT_TOP(exclude_syms) if callable(_ORIG_BYBIT_TOP) else []

def _final_get_scan_klines(symbol, interval, limit=250):
    mapped=INTERVAL_MAP.get(interval,interval)
    wsdf=bybit_market_ws.get_klines(symbol,mapped,limit)
    if len(wsdf)>=min(int(limit),40): return wsdf.tail(limit).copy()
    df=_bybit_backfill_once(symbol,interval,limit)
    if df is not None and not df.empty: return df.tail(limit).copy()
    return pd.DataFrame()

def _final_get_klines(symbol,interval,limit=250):
    return _final_get_scan_klines(symbol,interval,limit)

def get_scan_klines_final(symbol, interval, limit=250):
    return _final_get_scan_klines(symbol,interval,limit)

def get_klines_final(symbol, interval, limit=250):
    return _final_get_klines(symbol,interval,limit)

def get_price_final(symbol, prefer_binance=False):
    # Explicit Binance preference is retained for critical execution/reconciliation.
    if prefer_binance:
        try:
            with _binance_critical_context():
                return _binance_price(symbol)
        except Exception:
            pass
    return _final_bybit_price(symbol)

def _get_top_coins_impl_final(exclude_syms=None):
    """Canonical Bybit universe implementation.

    exclude_syms is optional by contract so the public get_top_coins() wrapper
    can never fail with a missing positional argument.
    """
    return _final_bybit_top(set(exclude_syms or ()))

# Existing get_top_coins wrapper resolves the global implementation symbol.
_get_top_coins_impl = _get_top_coins_impl_final

# ---------- Binance private user-data WS ----------
_binance_ws_state_lock=threading.RLock()
_binance_ws_positions={}
_binance_ws_orders={}
_binance_ws_balance={}
_binance_ws_last_event_at=0.0
_binance_ws_last_error=None
_binance_ws_thread=None
_binance_ws_stop=threading.Event()
_binance_ws_listen_key=None
_binance_ws_key_created_at=0.0

def _binance_ws_listen_key_create():
    if not BINANCE_API_KEY: return None
    _binance_wait_if_banned()
    with _binance_request_slot(critical=True):
        r=requests.post(f"{FAPI}/fapi/v1/listenKey",headers={"X-MBX-APIKEY":BINANCE_API_KEY},timeout=10,verify=False)
    _binance_update_weight_from_response(r)
    if r.status_code in (418,429):
        _binance_register_ban(r.text,retry_after=r.headers.get("Retry-After"))
        raise BinanceCooldownError(f"Binance listenKey HTTP {r.status_code}")
    r.raise_for_status(); d=r.json(); return d.get("listenKey")

def _binance_ws_listen_key_keepalive():
    if not BINANCE_API_KEY or not _binance_ws_listen_key: return
    try:
        _binance_wait_if_banned()
        with _binance_request_slot(critical=True):
            r=requests.put(f"{FAPI}/fapi/v1/listenKey",headers={"X-MBX-APIKEY":BINANCE_API_KEY},timeout=10,verify=False)
        _binance_update_weight_from_response(r)
        if r.status_code in (418,429):
            _binance_register_ban(r.text,retry_after=r.headers.get("Retry-After"))
            raise BinanceCooldownError(f"Binance listenKey keepalive HTTP {r.status_code}")
    except Exception as exc:
        log.warning(f"[BINANCE WS] listenKey keepalive gagal: {exc}")

def _binance_user_ws_start():
    global _binance_ws_thread
    if not _WS_LIB_OK or not BINANCE_API_KEY:
        _set_component_health("binance_user_websocket","DEGRADED","API key/WS library unavailable")
        return None
    if _binance_ws_thread and _binance_ws_thread.is_alive(): return _binance_ws_thread
    _binance_ws_stop.clear()
    _binance_ws_thread=threading.Thread(target=_binance_user_ws_loop,name="binance-user-ws",daemon=True)
    _binance_ws_thread.start()
    return _binance_ws_thread

def _binance_user_ws_loop():
    global _binance_ws_listen_key,_binance_ws_key_created_at,_binance_ws_last_error
    backoff=1.0
    while not _binance_ws_stop.is_set():
        try:
            _binance_ws_listen_key=_binance_ws_listen_key_create()
            _binance_ws_key_created_at=time.time()
            if not _binance_ws_listen_key: raise RuntimeError("listenKey kosong")
            url=f"wss://fstream.binance.com/ws/{_binance_ws_listen_key}"
            ws=websocket.WebSocketApp(url,on_open=lambda *_:_set_component_health("binance_user_websocket","HEALTHY","user-data WS connected"),on_message=_binance_user_ws_message,on_error=_binance_user_ws_error,on_close=_binance_user_ws_close)
            ws.run_forever(ping_interval=180,ping_timeout=30)
            backoff=1.0
        except Exception as exc:
            _binance_ws_last_error=str(exc)[:300]
            _set_component_health("binance_user_websocket","DEGRADED",_binance_ws_last_error)
        if _binance_ws_stop.is_set(): break
        if time.time()-_binance_ws_key_created_at>3300: _binance_ws_listen_key=None
        time.sleep(backoff); backoff=min(30.0,backoff*2)

def _binance_user_ws_error(ws,error):
    global _binance_ws_last_error
    _binance_ws_last_error=str(error)[:300]
    _set_component_health("binance_user_websocket","DEGRADED",_binance_ws_last_error)

def _binance_user_ws_close(ws,code,msg):
    log.warning(f"[BINANCE WS] user stream closed code={code}")


def _binance_user_ws_keepalive_loop():
    while not SHUTDOWN_EVENT.wait(45*60):
        try: _binance_ws_listen_key_keepalive()
        except Exception as exc: log.warning(f"[BINANCE WS] keepalive loop: {exc}")

def _binance_user_ws_message(ws,raw):
    global _binance_ws_last_event_at
    try: msg=json.loads(raw)
    except Exception:return
    now=time.time(); _binance_ws_last_event_at=now
    event=msg.get("e")
    with _binance_ws_state_lock:
        if event=="ACCOUNT_UPDATE":
            acct=msg.get("a") or {}
            for p in acct.get("P") or []:
                sym=str(p.get("s") or "").upper();
                if sym: _binance_ws_positions[sym]=dict(p)
            for b in acct.get("B") or []:
                asset=str(b.get("a") or "")
                if asset: _binance_ws_balance[asset]=dict(b)
        elif event=="ORDER_TRADE_UPDATE":
            o=msg.get("o") or {}; oid=str(o.get("i") or o.get("c") or "")
            if oid: _binance_ws_orders[oid]=dict(o)

def _binance_ws_position(symbol):
    with _binance_ws_state_lock:
        row=_binance_ws_positions.get(str(symbol).upper())
        return dict(row) if row else None

def _binance_ws_fresh():
    return _binance_ws_last_event_at>0 and time.time()-_binance_ws_last_event_at<=BINANCE_WS_STALE_SEC

def _binance_ws_status():
    return {"thread_alive":bool(_binance_ws_thread and _binance_ws_thread.is_alive()),"fresh":_binance_ws_fresh(),"last_event_at":_binance_ws_last_event_at,"last_error":_binance_ws_last_error,"positions":len(_binance_ws_positions),"orders":len(_binance_ws_orders)}

# Use WS for ordinary position polling, but preserve REST semantics for critical callers.
_ORIG_GET_REAL_POSITION=globals().get("get_real_position")
def get_real_position_final(symbol, prefer_ws=True):
    if prefer_ws and _binance_ws_fresh():
        row=_binance_ws_position(symbol)
        if row is not None:
            try:
                if abs(float(row.get("pa") or row.get("positionAmt") or 0))>0: return row
                return None
            except Exception: pass
    return _ORIG_GET_REAL_POSITION(symbol) if callable(_ORIG_GET_REAL_POSITION) else None

# ---------- REST governor helpers ----------
_binance_last_reconcile_by_symbol={}
_binance_last_bulk_reconcile=0.0

def _binance_reconcile_allowed(symbol=None, bulk=False, force=False):
    global _binance_last_bulk_reconcile
    now=time.time()
    if force: return True
    if bulk:
        return now-_binance_last_bulk_reconcile>=BINANCE_BULK_RECONCILE_MIN_INTERVAL
    key=str(symbol or "")
    last=float(_binance_last_reconcile_by_symbol.get(key,0.0))
    return now-last>=BINANCE_REST_RECONCILE_MIN_INTERVAL

def _mark_binance_reconcile(symbol=None, bulk=False):
    global _binance_last_bulk_reconcile
    now=time.time()
    if bulk: _binance_last_bulk_reconcile=now
    else: _binance_last_reconcile_by_symbol[str(symbol or "")]=now

# ---------- Unified PnL / trade accounting ----------
def _trade_price_move_pct(entry, exit_price, decision):
    try:
        e=float(entry); x=float(exit_price)
        if e<=0 or x<=0: return 0.0
        sign=1.0 if str(decision or "BUY").upper()=="BUY" else -1.0
        return (x-e)/e*100.0*sign
    except Exception:return 0.0

def _trade_net_position_pnl_usd(entry, exit_price, decision, quantity, fees_usd=0.0):
    try:
        e=float(entry); x=float(exit_price); q=abs(float(quantity or 0.0))
        if e<=0 or q<=0:return 0.0
        sign=1.0 if str(decision or "BUY").upper()=="BUY" else -1.0
        gross=(x-e)*q*sign
        return gross-float(fees_usd or 0.0)
    except Exception:return 0.0

def _compute_account_impact_pct(pnl_usd, anchor):
    try:
        a=float(anchor)
        return float(pnl_usd)/a*100.0 if a else 0.0
    except Exception:return 0.0

# Preserve original updater for mechanics; enrich record afterwards.
_ORIG_UPDATE_STATS=globals().get("update_stats")
def update_stats_final(*args, **kwargs):
    # Canonicalized front-end: the legacy updater still performs balance/ledger mutation.
    result=kwargs.get("result", args[0] if args else None)
    entry=kwargs.get("entry"); exit_price=kwargs.get("close_price")
    decision=kwargs.get("decision")
    quantity=kwargs.get("quantity")
    anchor=kwargs.get("balance_anchor")
    if quantity is None:
        margin=float(MARGIN_USD or 0.0); lev=float(LEVERAGE or 1.0); quantity=margin*lev/max(float(entry or 1.0),1e-12)
    canonical_result=_classify_close_result(result,entry=entry,close_price=exit_price,decision=decision)
    kwargs["result"]=canonical_result
    call_args=args[1:] if args and "result" in kwargs else args
    out=_ORIG_UPDATE_STATS(*call_args,**kwargs)
    # Add normalized fields to last ledger record atomically after original updater.
    try:
        with trade_history_lock:
            rec=trade_history[-1] if trade_history and trade_history[-1].get("symbol")==kwargs.get("sym") else None
            enriched={"price_move_pct":_trade_price_move_pct(entry,exit_price,decision),
                      "configured_sl_pct":(abs(float(entry)-float(kwargs.get("sl_p")))/float(entry)*100.0) if entry and kwargs.get("sl_p") else None,
                      "actual_exit_price":exit_price,
                      "account_impact_pct":_compute_account_impact_pct(float(rec.get("pnl_usd",0.0)), anchor or STARTING_BALANCE),
                      "position_pnl_pct":float(rec.get("pct",0.0)),
                      "pnl_semantics":{"price_move_pct":"price movement after side normalization","position_pnl_pct":"net PnL percentage used by legacy position accounting","account_impact_pct":"net PnL / balance anchor"}}
            rec.update(enriched)
            for h in stats.get("pnl_history",[]):
                if h is not rec and h.get("trade_uid") == rec.get("trade_uid") and h.get("symbol") == rec.get("symbol"):
                    h.update(enriched); break
    except Exception as exc:
        log.warning(f"[STATS][PNL NORMALIZE] {exc}")
    return out

# ---------- Pending-only timeout ----------
def _timeout_pending_entries(chat_id=None):
    now=time.time(); targets=[]
    with positions_lock:
        for sym,pos in list(positions.items()):
            lifecycle=str(pos.get("lifecycle") or "").upper()
            status=str(pos.get("status") or "").lower()
            if lifecycle=="ENTRY_PENDING" or status=="pending":
                age=now-float(pos.get("entry_created_at") or pos.get("entry_time") or now)
                if age>=0: targets.append((sym,dict(pos)))
    done=0; failed=[]
    for sym,pos in targets:
        try:
            if _position_is_real(pos):
                oid=pos.get("entry_order_id") or pos.get("order_id")
                if oid: cancel_order(sym,oid)
            with positions_lock:
                cur=positions.get(sym)
                if cur and str(cur.get("lifecycle") or "").upper()=="ENTRY_PENDING": positions.pop(sym,None)
            _record_pending_cancel("expired")
            done+=1
        except Exception as exc: failed.append(f"{sym}: {str(exc)[:120]}")
    return {"found":len(targets),"cancelled":done,"failed":failed}

def _pending_entry_watchdog():
    while not SHUTDOWN_EVENT.wait(15):
        try:
            now=time.time(); expired=[]
            with positions_lock:
                for sym,pos in list(positions.items()):
                    if str(pos.get("lifecycle") or "").upper()!="ENTRY_PENDING": continue
                    created=float(pos.get("entry_created_at") or pos.get("entry_time") or now)
                    if now-created>=PENDING_ENTRY_TIMEOUT_SEC: expired.append((sym,pos.get("chat_id") or active_chat_id))
            for sym,cid in expired:
                try:
                    if _position_is_real(positions.get(sym,{})):
                        oid=positions[sym].get("entry_order_id") or positions[sym].get("order_id")
                        if oid: cancel_order(sym,oid)
                    with positions_lock:
                        positions.pop(sym,None)
                    _record_pending_cancel("expired")
                    if cid: tg_send(cid,f"⏱️ <b>PENDING TIMEOUT</b> — {sym}\nEntry pending dibatalkan setelah {PENDING_ENTRY_TIMEOUT_SEC/60:.0f} menit.")
                except Exception as exc: log.warning(f"[PENDING TIMEOUT] {sym}: {exc}")
        except Exception as exc: log.warning(f"[PENDING TIMEOUT WATCHDOG] {exc}")

# ---------- Trail breach latch / recovery ----------
def _trail_breach_price_check(sym, pos, price):
    try:
        pending=_get_pending_trail(sym)
        if not pending or price is None: return False
        sl=float(pending.get("sl")); side=str(pos.get("signal",{}).get("decision") or pending.get("side") or "BUY").upper()
        crossed = float(price) <= sl*(1.0+SL_BREACH_EPSILON_PCT/100.0) if side=="BUY" else float(price) >= sl*(1.0-SL_BREACH_EPSILON_PCT/100.0)
        if crossed:
            with positions_lock:
                cur=positions.get(sym)
                if cur is not None:
                    cur["trail_breach_latched"]=True; cur["trail_breach_price"]=float(price); cur["trail_breach_at"]=time.time(); cur["forced_exit_pending"]=True
            log.warning(f"[TRAIL BREACH] {sym} pending SL crossed at {price}")
            return True
    except Exception: return False
    return False

def _process_trail_breach_after_recovery(sym,pos):
    with positions_lock: cur=dict(positions.get(sym) or pos)
    if not cur.get("forced_exit_pending"): return False
    try:
        buy=str(cur.get("signal",{}).get("decision") or "BUY").upper()=="BUY"
        closed,exit_price=_verified_market_close(sym,buy,"trail_breach",chat_id=cur.get("chat_id") or active_chat_id,max_retries=0)
        if not closed:return False
        try:
            with _binance_critical_context(): _cancel_all_symbol_orders_verified(sym)
        except Exception as exc: _queue_pending_cleanup(sym,"trail breach final cleanup",exc); raise
        entry=float(cur.get("entry")); xp=float(exit_price or cur.get("trail_breach_price") or entry)
        result="trail" if _classify_close_result("trail",entry,xp,cur.get("signal",{}).get("decision"))=="trail" else "sl"
        _clear_pending_trail(sym)
        with positions_lock:
            if sym in positions: positions[sym]["forced_exit_pending"]=False
        close_position(sym,result,close_price=xp)
        return True
    except BinanceCooldownError:
        return False
    except Exception as exc:
        _queue_pending_cleanup(sym,"trail breach close failed",exc)
        log.warning(f"[TRAIL BREACH RECOVERY] {sym}: {exc}")
        return False

# ---------- Cleanup: one verified coordinator ----------
def _final_cleanup_after_flat(sym, reason="flat"):
    try:
        with _binance_critical_context():
            _cancel_all_symbol_orders_verified(sym)
        _clear_pending_trail(sym); _clear_pending_protection(sym); _clear_pending_cleanup(sym)
        return True
    except Exception as exc:
        _queue_pending_cleanup(sym,f"flat cleanup: {reason}",exc)
        return False

# ---------- Telegram anti-spam ----------
_TG_SUPPRESS_LOCK=threading.Lock()
_TG_SUPPRESS_STATE={}
_ORIG_TG_SEND=globals().get("tg_send")
def tg_send_final(chat_id, text, *args, **kwargs):
    # Do not suppress critical user-directed responses; suppress only repeated system errors/trailing failures.
    txt=str(text or "")
    upper=txt.upper()
    critical=any(k in upper for k in ("EMERGENCY","POSITION UNPROTECTED","EXECUTION UNKNOWN"))
    system_error=any(k in upper for k in ("UPDATE PROTECTION GAGAL","ALGO CLEANUP","TRAILING UPDATE","[ERROR]","RECOVERY BELUM","PROTECTION GAGAL"))
    if not system_error or critical:
        return _ORIG_TG_SEND(chat_id,text,*args,**kwargs)
    import hashlib
    sig=hashlib.sha1(re.sub(r"\d+(\.\d+)?","#",txt).encode()).hexdigest()
    now=time.time(); key=(str(chat_id),sig)
    with _TG_SUPPRESS_LOCK:
        item=_TG_SUPPRESS_STATE.get(key,{"last":0.0,"count":0,"window_start":now,"suppressed":0})
        if now-item["window_start"]>TELEGRAM_FLOOD_WINDOW_SEC:
            item={"last":0.0,"count":0,"window_start":now,"suppressed":0}
        if now-item["last"]<TELEGRAM_REPEAT_SUPPRESS_SEC or item["count"]>=TELEGRAM_FLOOD_MAX:
            item["count"]+=1; item["suppressed"]+=1; _TG_SUPPRESS_STATE[key]=item
            return True
        item["count"]+=1; item["last"]=now; _TG_SUPPRESS_STATE[key]=item
    return _ORIG_TG_SEND(chat_id,text,*args,**kwargs)

# ---------- Human-readable status ----------
def _final_scanner_status():
    s=get_scanner_status()
    by=bybit_market_ws.status()
    bn=_binance_ws_status()
    return {
        "scanner":s,
        "bybit_ws":by,
        "binance_user_ws":bn,
        "binance_execution_paused":_binance_is_scan_paused(),
    }

def fmt_runtime_status():
    s=get_scanner_status(); by=bybit_market_ws.status(); bn=_binance_ws_status()
    lines=[
        "📡 <b>RUNTIME STATUS</b>",
        f"🔎 Scanner: <b>{s.get('health')}</b> | cycle <b>{s.get('cycle_count',0)}</b> | last {s.get('last_cycle_age_sec') if s.get('last_cycle_age_sec') is not None else '—'}s ago",
        f"   processed <b>{s.get('last_symbols_processed',0)}</b> | candidates <b>{s.get('last_candidate_count',0)}</b> | eligible <b>{s.get('last_eligible_count',0)}</b>",
        f"🟣 Bybit WS: <b>{'CONNECTED' if by.get('fresh') else 'DEGRADED'}</b> | topics {by.get('topics',0)} | tickers {by.get('tickers',0)} | buffers {by.get('kline_buffers',0)}",
        f"🔵 Binance WS: <b>{'FRESH' if bn.get('fresh') else 'STALE'}</b> | positions {bn.get('positions',0)} | orders {bn.get('orders',0)}",
        f"💳 Binance entry: <b>{'PAUSED' if _binance_is_scan_paused() else 'READY'}</b>",
    ]
    return "\n".join(lines)

# ---------- Override scanner worker start so watchdog never mislabels dead thread ----------
_ORIG_ENSURE_SCANNER_RUNNING=globals().get("_ensure_scanner_running")
def _ensure_scanner_running_final(chat_id, announce=False):
    global auto_mode,auto_thread,active_chat_id
    active_chat_id=chat_id or active_chat_id
    if _scanner_thread_is_alive():
        _set_scan_state(enabled=True,coordinator_alive=True,last_error=None)
        return auto_thread,False
    auto_mode=True
    _set_scan_state(enabled=True,coordinator_alive=False,cycle_running=False,last_error=None)
    _set_component_health("scanner","STARTING","scanner coordinator starting")
    try:
        t=threading.Thread(target=simulation_loop,args=(active_chat_id,),name="scanner-coordinator",daemon=True)
        auto_thread=t; t.start(); _SCAN_WAKE.set()
        if announce and active_chat_id: tg_send(active_chat_id,"🔎 <b>Scanner dimulai.</b>\nMarket data realtime: <b>Bybit WS</b>\nBackfill/fallback: <b>Bybit REST</b>\nExecution: <b>Binance</b>")
        return t,True
    except Exception as exc:
        auto_mode=False; _set_scan_state(enabled=False,coordinator_alive=False,last_error=str(exc)[:300]); _set_component_health("scanner","DEGRADED",str(exc)[:250]); raise

# ---------- Override run scan to use WS-first and feed brain ----------
_ORIG_RUN_SCAN_ONCE=globals().get("run_scan_once")
def run_scan_once_final(chat_id):
    result=_ORIG_RUN_SCAN_ONCE(chat_id) if callable(_ORIG_RUN_SCAN_ONCE) else []
    # Ensure desired universe is subscribed even when the legacy function already completed via REST.
    try:
        syms=list(last_scanned_coins[-TOP_N_COINS:]) if last_scanned_coins else []
        if syms: bybit_market_ws.subscribe_symbols(syms)
    except Exception: pass
    return result

# ---------- Better real position monitor: WS-first, REST only when stale/required ----------
_ORIG_MONITOR_POSITION_REAL=globals().get("monitor_position_real")
def monitor_position_real_final(sym,pos):
    next_strategy=0.0; next_rest_reconcile=0.0
    while True:
        with positions_lock:
            if sym not in positions:return
            pos=positions[sym]
        try:
            if pos.get("timeout_flag"):
                _verified_timeout_symbol(sym,pos.get("chat_id") or active_chat_id,reason="manual timeout"); return
            px=_final_bybit_price(sym)
            if px is not None:
                with positions_lock:
                    if sym in positions:
                        positions[sym]["current_price"]=px
                        _update_trade_path_metrics(positions[sym],px)
                        pos=positions[sym]
                if _trail_breach_price_check(sym,pos,px):
                    # Breach is latched; no Binance REST while exchange is blocked.
                    if _binance_is_scan_paused():
                        time.sleep(min(5.0,max(1.0,_binance_cooldown_remaining())))
                        continue
            if pos.get("forced_exit_pending") and not _binance_is_scan_paused():
                if _process_trail_breach_after_recovery(sym,pos): return
            # Strategy management can run from Bybit market data without Binance REST.
            if time.time()>=next_strategy:
                upd=_strategy_position_update(sym,pos)
                next_strategy=time.time()+STRATEGY_MANAGE_INTERVAL
                if isinstance(upd,dict):
                    oldsl=pos.get("current_sl",pos.get("signal",{}).get("sl")); oldtp=pos.get("signal",{}).get("tp")
                    cand_sl=upd.get("sl"); cand_tp=upd.get("tp") if upd.get("tp") is not None else oldtp
                    if cand_sl is not None and oldsl is not None:
                        buy=str(pos.get("signal",{}).get("decision") or "BUY").upper()=="BUY"
                        if not ((float(cand_sl)>float(oldsl)) if buy else (float(cand_sl)<float(oldsl))): cand_sl=oldsl
                    if cand_sl is not None and cand_sl!=oldsl:
                        if _binance_is_scan_paused():
                            _queue_pending_trail(sym,float(cand_sl),cand_tp,pos.get("quantity"),reason="strategy",side=pos.get("signal",{}).get("decision"))
                            _notify_trail_update(active_chat_id,sym,pos,upd,oldsl,cand_sl,status="QUEUED")
                        else:
                            try:
                                latest=_ORIG_GET_REAL_POSITION(sym) if callable(_ORIG_GET_REAL_POSITION) else None
                                live_qty=abs(float(latest.get("positionAmt",0) or 0)) if latest else float(pos.get("quantity") or 0)
                                if live_qty<=0: pass
                                else:
                                    with _binance_critical_context(): _cancel_all_algo_orders_verified(sym)
                                    nt,ns=place_tp_sl(sym,str(pos.get("signal",{}).get("decision") or "BUY").upper()=="BUY",cand_tp,float(cand_sl),live_qty)
                                    with positions_lock:
                                        if sym in positions:
                                            positions[sym]["current_sl"]=float(cand_sl); positions[sym]["signal"]["sl"]=float(cand_sl); positions[sym]["tp_order_id"]=nt.get("algoId"); positions[sym]["sl_order_id"]=ns.get("algoId"); positions[sym]["quantity"]=live_qty
                                    _clear_pending_trail(sym); _notify_trail_update(active_chat_id,sym,positions[sym],upd,oldsl,cand_sl,status="APPLIED")
                            except BinanceCooldownError as exc:
                                _queue_pending_trail(sym,float(cand_sl),cand_tp,pos.get("quantity"),reason="strategy",side=pos.get("signal",{}).get("decision"))
                                _notify_trail_update(active_chat_id,sym,pos,upd,oldsl,cand_sl,status="QUEUED",error=exc)
                            except Exception as exc:
                                # One alert only; previous SL remains unchanged.
                                _notify_trail_update(active_chat_id,sym,pos,upd,oldsl,cand_sl,status="FAILED",error=exc)
            # REST reconcile only if WS is stale and not rate-limited.
            if (not _binance_ws_fresh()) and time.time()>=next_rest_reconcile and _binance_reconcile_allowed(sym):
                try:
                    real=_ORIG_GET_REAL_POSITION(sym) if callable(_ORIG_GET_REAL_POSITION) else None
                    _mark_binance_reconcile(sym)
                    if real is None:
                        _finalize_external_close(sym,pos,reason_hint="unknown",exit_price=px); return
                except BinanceCooldownError:
                    pass
                except Exception as exc:
                    log.warning(f"[monitor_real/reconcile] {sym}: {exc}")
                next_rest_reconcile=time.time()+BINANCE_REST_RECONCILE_MIN_INTERVAL
            time.sleep(MONITOR_SLEEP)
        except Exception as exc:
            log.exception(f"[monitor_real/final] {sym}: {exc}")
            time.sleep(MONITOR_SLEEP)


# ---------- Pending real-entry watcher: Binance WS-first ----------
_ORIGINAL_WAIT_ENTRY_REAL_V100=globals().get("_wait_entry_real")
def _wait_entry_real_final(sym,signal,chat_id,order_id):
    deadline=time.time()+8*3600; next_rest=0.0
    while time.time()<deadline:
        with positions_lock:
            pos=positions.get(sym)
            if pos is None: return
            if pos.get("timeout_flag"):
                try:
                    with _binance_critical_context(): cancel_order(sym,order_id)
                    with _binance_critical_context(): st=get_order_status(sym,order_id)
                    if str(st.get("status","")).upper()=="FILLED":
                        _open_position_real(sym,signal,float(st.get("avgPrice") or signal["entry"]),chat_id,st); return
                    positions.pop(sym,None); return
                except Exception as exc:
                    _force_position_emergency(sym,str(exc)[:300]); return
        event=None
        with _binance_ws_state_lock:
            event=dict(_binance_ws_orders.get(str(order_id)) or _binance_ws_orders.get(str(signal.get("order_id") or "")) or {})
        status=str(event.get("X") or event.get("status") or "").upper()
        if status=="FILLED":
            actual=float(event.get("ap") or event.get("avgPrice") or signal.get("entry") or 0)
            _open_position_real(sym,signal,actual,chat_id,event); return
        if status in {"CANCELED","EXPIRED","REJECTED","EXPIRED_IN_MATCH"}:
            with positions_lock: positions.pop(sym,None)
            _ban_coin(sym,f"order {status.lower()}"); _record_pending_cancel("binance_reject"); return
        # REST fallback only when WS is stale or no event has appeared for a controlled interval.
        if (not _binance_ws_fresh() or time.time()>=next_rest):
            if not _binance_is_scan_paused() and time.time()>=next_rest and _binance_reconcile_allowed(sym):
                try:
                    with _binance_critical_context(): st=get_order_status(sym,order_id)
                    _mark_binance_reconcile(sym); next_rest=time.time()+BINANCE_REST_RECONCILE_MIN_INTERVAL
                    stt=str(st.get("status","")).upper()
                    if stt=="FILLED": _open_position_real(sym,signal,float(st.get("avgPrice") or signal["entry"]),chat_id,st); return
                    if stt in {"CANCELED","EXPIRED","REJECTED"}:
                        with positions_lock: positions.pop(sym,None)
                        _ban_coin(sym,f"order {stt.lower()}"); _record_pending_cancel("binance_reject"); return
                except BinanceCooldownError: pass
                except Exception as exc: log.warning(f"[wait_entry_real] REST fallback {sym}: {exc}")
        with positions_lock:
            created=float(positions.get(sym,{}).get("entry_created_at") or positions.get(sym,{}).get("entry_time") or time.time()) if sym in positions else time.time()
        if time.time()-created>=PENDING_ENTRY_TIMEOUT_SEC:
            try:
                with _binance_critical_context(): cancel_order(sym,order_id)
            except Exception: pass
            with positions_lock: positions.pop(sym,None)
            _ban_coin(sym,"pending expired"); _record_pending_cancel("expired"); return
        time.sleep(min(5.0,MONITOR_SLEEP))

_wait_entry_real=_wait_entry_real_final

# ---------- Flat-position cleanup wrapper ----------
_ORIG_FINALIZE_EXTERNAL_CLOSE=globals().get("_finalize_external_close")
def _finalize_external_close_final(sym,pos,reason_hint="unknown",exit_price=None):
    out=_ORIG_FINALIZE_EXTERNAL_CLOSE(sym,pos,reason_hint=reason_hint,exit_price=exit_price) if callable(_ORIG_FINALIZE_EXTERNAL_CLOSE) else False
    try: _final_cleanup_after_flat(sym,reason="external-close")
    except Exception: pass
    return out

# ---------- Human-readable stats formatter with correct percentage semantics ----------
_ORIG_FMT_STATS=globals().get("fmt_stats")
def fmt_stats_final():
    with stat_lock:
        hist=[dict(x) for x in stats.get("pnl_history",[])]; t=int(stats.get("total",0)); tp=int(stats.get("tp",0)); sl=int(stats.get("sl",0)); trail=int(stats.get("trail",0)); bal=float(stats.get("balance",0.0))
    wins=tp+trail; wr=wins/(wins+sl)*100 if wins+sl else 0.0
    anchor=float(real_balance_snapshot if REAL_TRADE_ENABLED and real_balance_snapshot is not None else STARTING_BALANCE)
    net_pct=(bal-anchor)/anchor*100 if anchor else 0.0
    def avg(k):
        vals=[float(x.get("confidence")) for x in hist if str(x.get("result"))==k and x.get("confidence") is not None]
        return sum(vals)/len(vals) if vals else None
    def pc(v): return f"{v:.1f}%" if v is not None else "—"
    recent=[]
    for x in reversed(hist[-5:]):
        pnl=float(x.get("pnl_usd",0) or 0); icon="🟢" if pnl>0 else "🔴" if pnl<0 else "⚪"; move=x.get("price_move_pct")
        move_txt=f"{float(move):+.2f}%" if move is not None else f"{float(x.get('pct',0)):+.2f}%"
        recent.append(f"{icon} {str(x.get('result','?')).upper()} {move_txt} price | {x.get('symbol','?')} | C{float(x.get('confidence',0) or 0):.0f}%")
    if not recent: recent=["—"]
    with ban_lock: bn=len(banned_coins)
    with early_reject_lock: er=early_reject_remaining
    try: low=", ".join(f"{x['symbol']} ({x['count']}x)" for x in _low_conf_summary()[:3]) or "—"
    except Exception: low="—"
    try:
        dec=_brain_on_stats_snapshot({"total":t,"tp":tp,"sl":sl,"trail":trail,"balance":bal,"recent":hist[-20:]}) or {}
    except Exception as exc: dec={"action":"REVIEW_ERROR","reason":str(exc)[:200]}
    bs=dec.get("active_threshold","—"); fstate=dec.get("frequency_state") or dec.get("frequency_action") or "—"; action=dec.get("action","WAITING_DATA"); reason=dec.get("reason","—"); proposal=dec.get("proposal","—")
    scan=get_last_scan_telemetry(); ss= get_scanner_status()
    lines=[
        f"📊 <b>STATISTIK</b> — {t} trade | ✅ TP {tp} | 🟢 Trail {trail} | 🔴 SL {sl}",
        f"Mode: <b>{'🔴 REAL' if REAL_TRADE_ENABLED else '🧪 SIMULASI'}</b>",
        f"Win rate: <b>{wr:.1f}%</b> | Net account: <b>{net_pct:+.2f}%</b>",
        f"Saldo statistik: <b>${bal:.4f}</b>",
        f"Confidence closed: TP {pc(avg('tp'))} | Trail {pc(avg('trail'))} | SL {pc(avg('sl'))}",
        "",
        "🧠 <b>KEPUTUSAN OTAK</b>",
        f"Action: <b>{html.escape(str(action))}</b>",
        f"Reason: {html.escape(str(reason)[:260])}",
        f"Proposal: <b>{html.escape(str(proposal)[:220])}</b>",
        f"Threshold: <b>{html.escape(str(bs))}%</b> | Frequency: <b>{html.escape(str(fstate))}</b>",
        "",
        "5 terakhir:",
        *recent,
        "",
        f"🚫 Ban: <b>{bn}</b> | Low-conf: <b>{html.escape(low)}</b> | Early reject: <b>{er}</b>",
        f"🔎 Scanner: <b>{ss.get('health')}</b> | cycle <b>{ss.get('cycle_count',0)}</b> | eligible <b>{ss.get('last_eligible_count',0)}</b>",
        f"📡 Data: <b>Bybit WS → REST backfill</b> | Execution: <b>Binance</b> | Binance pause: <b>{'YA' if _binance_is_scan_paused() else 'TIDAK'}</b>",
        f"Bybit REST: <b>{_bybit_request_count}</b> req | errors: <b>{_bybit_request_errors}</b>",
    ]
    return "\n".join(lines)

# ---------- Timeout command semantics ----------
_ORIG_VERIFIED_TIMEOUT_ALL=globals().get("_verified_timeout_all")
def _verified_timeout_pending_only(chat_id):
    r=_timeout_pending_entries(chat_id)
    tg_send(chat_id,f"⏱️ <b>/timeout pending</b>\nPending ditemukan: <b>{r['found']}</b>\nDibatalkan: <b>{r['cancelled']}</b>\nGagal: <b>{len(r['failed'])}</b>")
    if r["failed"]: tg_send(chat_id,"\n".join(f"• {x}" for x in r["failed"])[:1800])
    return r

# ---------- Runtime startup extension ----------
_ORIG_START_RUNTIME=globals().get("start_runtime")
def start_runtime_final():
    # Start upstream runtime first; then add final data/control buses.
    _ORIG_START_RUNTIME()
    try:
        bybit_market_ws.start()
        _binance_user_ws_start()
        threading.Thread(target=_binance_user_ws_keepalive_loop,name="binance-user-ws-keepalive",daemon=True).start()
        threading.Thread(target=_pending_entry_watchdog,name="pending-entry-watchdog",daemon=True).start()
        _set_component_health("scanner","STARTING","scanner waits for /auto; Bybit WS ready")
    except Exception as exc:
        log.exception(f"[FINAL STARTUP] extension failed: {exc}")

# ---------- Brain status formatting bridge ----------
def _format_brain_status_human(status):
    st=status if isinstance(status,dict) else {}
    adaptive=st.get("adaptive") if isinstance(st.get("adaptive"),dict) else {}
    freq=adaptive.get("frequency") if isinstance(adaptive.get("frequency"),dict) else {}
    la=adaptive.get("last_strategy_decision") if isinstance(adaptive.get("last_strategy_decision"),dict) else {}
    return (
        "🧠 <b>FULL STATUS</b>\n"
        f"Worker: <b>{'✅' if adaptive.get('worker_alive') else '❌'}</b>\n"
        f"Observations: <b>{adaptive.get('live',{}).get('observations',0)}</b> | Candidates: <b>{adaptive.get('live',{}).get('candidates',0)}</b> | Outcomes: <b>{adaptive.get('live',{}).get('outcomes',0)}</b>\n"
        f"Threshold: <b>{adaptive.get('active_threshold','—')}%</b> ({adaptive.get('threshold_mode','auto')})\n"
        f"Frequency: <b>{adaptive.get('frequency_state') or freq.get('status') or '—'}</b> | Action: <b>{(adaptive.get('last_frequency_action') or {}).get('action','—') if isinstance(adaptive.get('last_frequency_action'),dict) else '—'}</b>\n"
        f"Strategy: <b>{adaptive.get('strategy_version','S1')}</b> | revisions <b>{adaptive.get('strategy_revisions',0)}</b>\n"
        f"Last brain decision: <b>{la.get('action','—')}</b> — {html.escape(str(la.get('reason','—'))[:220])}"
    )

# Startup of this overlay is intentionally idempotent and only occurs when main runtime starts.

# Final symbol bindings: overrides are uniquely named to keep static audit unambiguous.
_FINAL_OVERRIDES = {
    "tg_send": tg_send_final, "get_real_position": get_real_position_final,
    "get_price": get_price_final, "get_klines": get_klines_final, "get_scan_klines": get_scan_klines_final,
    "run_scan_once": run_scan_once_final, "update_stats": update_stats_final, "fmt_stats": fmt_stats_final,
    "_finalize_external_close": _finalize_external_close_final, "monitor_position_real": monitor_position_real_final,
    "_ensure_scanner_running": _ensure_scanner_running_final, "start_runtime": start_runtime_final,
}
globals().update(_FINAL_OVERRIDES)


# ---------- Final recovery: Binance execution recovery never pauses Bybit scanning ----------
_ORIGINAL_RESUME_BINANCE_V100 = globals().get("_resume_binance_and_flush_pending")
def _resume_binance_and_flush_pending_final(chat_id=None):
    global _binance_recovering,_binance_scan_paused,_binance_pause_reason
    if _binance_cooldown_remaining()>0: return False
    with _binance_pause_lock:
        _binance_recovering=True; _binance_scan_paused=True; _binance_pause_reason="execution recovery"
    if not _has_real_recovery_work():
        with _binance_pause_lock: _binance_recovering=False; _binance_scan_paused=False; _binance_pause_reason=""
        return True
    failures=[]
    try:
        key,secret=_read_binance_credentials()
        if not key or not secret:
            failures.append("credentials unavailable")
        else:
            global BINANCE_API_KEY,BINANCE_API_SECRET,BINANCE_KEYS_PRESENT
            BINANCE_API_KEY,BINANCE_API_SECRET,BINANCE_KEYS_PRESENT=key,secret,True
            with positions_lock: items=[(sym,dict(pos)) for sym,pos in positions.items() if _position_is_real(pos) and str(pos.get("status")) in ("active","EMERGENCY")]
            for sym,pos in items:
                try:
                    real=_binance_ws_position(sym) if _binance_ws_fresh() else None
                    if real is None:
                        real=_ORIG_GET_REAL_POSITION(sym) if callable(_ORIG_GET_REAL_POSITION) else None
                    if real is None or abs(float(real.get("pa") or real.get("positionAmt") or 0))<=0:
                        price=_final_bybit_price(sym) or pos.get("current_price") or pos.get("entry")
                        _finalize_external_close_final(sym,pos,reason_hint="unknown",exit_price=price)
                        continue
                    qty=abs(float(real.get("pa") or real.get("positionAmt") or 0))
                    with positions_lock:
                        if sym in positions: positions[sym].update({"quantity":qty,"exchange_synced_at":time.time()})
                except Exception as exc: failures.append(f"{sym}: sync {exc}")
            # A breached queued trail has priority over re-installing the old protection.
            with _pending_trails_lock: pending=list((s,d.copy()) for s,d in _pending_trails.items())
            for sym,tr in pending:
                try:
                    with positions_lock: pos=dict(positions.get(sym) or {})
                    if not pos or not _position_is_real(pos): _clear_pending_trail(sym); continue
                    px=_final_bybit_price(sym)
                    _trail_breach_price_check(sym,pos,px)
                    if pos.get("forced_exit_pending"):
                        if _process_trail_breach_after_recovery(sym,pos): continue
                    buy=str(pos.get("signal",{}).get("decision") or "BUY").upper()=="BUY"
                    qty=pos.get("quantity") or tr.get("quantity"); tp=tr.get("tp") or pos.get("signal",{}).get("tp"); sl=tr.get("sl") or pos.get("current_sl")
                    if not qty or tp is None or sl is None: raise RuntimeError("pending trail incomplete")
                    with _binance_critical_context():
                        _cancel_all_algo_orders_verified(sym)
                        nt,ns=place_tp_sl(sym,buy,tp,sl,qty)
                    old=pos.get("current_sl")
                    with positions_lock:
                        if sym in positions: positions[sym].update({"tp_order_id":nt.get("algoId"),"sl_order_id":ns.get("algoId"),"current_sl":float(sl),"signal":{**positions[sym].get("signal",{}),"sl":float(sl)},"exchange_synced_at":time.time()})
                    _clear_pending_trail(sym)
                    if old is not None and float(old)!=float(sl): _record_trail_event(sym,pos,{"trail_source":"recovery","reason":["queued trail applied after Binance recovery"]},old,sl,status="APPLIED")
                except Exception as exc: failures.append(f"{sym}: trail {exc}")
            with _pending_protections_lock: prots=list((s,d.copy()) for s,d in _pending_protections.items())
            for sym,pr in prots:
                try:
                    with positions_lock: pos=positions.get(sym)
                    if not pos or not _position_is_real(pos): _clear_pending_protection(sym); continue
                    qty=pos.get("quantity") or pr.get("quantity"); buy=pr.get("side")=="BUY"; tp=pr.get("tp"); sl=pr.get("sl")
                    if not qty or tp is None or sl is None: raise RuntimeError("pending protection incomplete")
                    with _binance_critical_context():
                        nt,ns=place_tp_sl(sym,buy,tp,sl,qty)
                    with positions_lock:
                        if sym in positions: positions[sym].update({"tp_order_id":nt.get("algoId"),"sl_order_id":ns.get("algoId"),"protection_state":"VERIFIED"})
                    _clear_pending_protection(sym)
                except Exception as exc: failures.append(f"{sym}: protection {exc}")
            with _pending_cleanup_lock: cleans=list(_pending_cleanup.items())
            for sym,item in cleans:
                try: _cleanup_algo_orders_verified(sym)
                except Exception as exc: failures.append(f"{sym}: cleanup {exc}")
        if failures:
            with _binance_pause_lock: _binance_recovering=False; _binance_scan_paused=True; _binance_pause_reason="execution recovery incomplete"
            msg=" | ".join(failures[:6])
            log.error(f"[BINANCE RECOVERY] incomplete; Bybit scanner tetap berjalan. {msg}")
            if chat_id: tg_send(chat_id,f"⚠️ <b>Binance recovery belum selesai.</b>\n🔎 Scanner Bybit <b>tetap berjalan</b>.\nExecution/protection REAL masih menunggu recovery.\n<code>{html.escape(msg[:500])}</code>")
            return False
        with _binance_pause_lock: _binance_recovering=False; _binance_scan_paused=False; _binance_pause_reason=""
        if chat_id: tg_send(chat_id,"✅ <b>Binance recovery selesai.</b>\nExecution/protection REAL kembali konsisten.\n🔎 Scanner Bybit tetap aktif.")
        return True
    except Exception as exc:
        with _binance_pause_lock: _binance_recovering=False; _binance_scan_paused=True; _binance_pause_reason="execution recovery exception"
        log.exception(f"[BINANCE RECOVERY] {exc}")
        return False


def _binance_recovery_loop_final(chat_id_getter=lambda: active_chat_id):
    consecutive=0
    while not SHUTDOWN_EVENT.wait(5):
        try:
            if _binance_is_scan_paused():
                _notify_binance_pause_once(chat_id_getter())
                if _binance_cooldown_remaining()<=0 and not _binance_recovering:
                    ok=_resume_binance_and_flush_pending_final(chat_id_getter()); consecutive=0 if ok else consecutive+1
                    if consecutive>3: time.sleep(20)
                else: time.sleep(5)
        except Exception as exc: log.warning(f"[binance-recovery] {exc}")

# ---------- Autostop WS-first balance ----------
def autostop_loop_final(chat_id):
    global auto_mode,peak_real_balance
    last_rest=0.0
    while not SHUTDOWN_EVENT.wait(30):
        try:
            if not REAL_TRADE_ENABLED: continue
            total=None
            with _binance_ws_state_lock:
                row=dict(_binance_ws_balance.get("USDT") or {})
            if row and _binance_ws_fresh():
                try: total=float(row.get("wb") or row.get("walletBalance") or 0)
                except Exception: total=None
            if total is None and time.time()-last_rest>=300 and not _binance_is_scan_paused():
                try:
                    _,total=get_real_balance(); last_rest=time.time()
                except Exception: pass
            if total is None: continue
            with autostop_lock:
                if peak_real_balance is None or total>peak_real_balance: peak_real_balance=total
                dd=(peak_real_balance-total)/peak_real_balance*100 if peak_real_balance else 0
            if auto_mode and dd>=AUTOSTOP_PCT:
                auto_mode=False
                tg_send(chat_id,f"🛑 <b>AUTO-STOP TERPICU</b>\nDrawdown <b>{dd:.2f}%</b> dari peak ${peak_real_balance:.4f}.\nScanner dihentikan; posisi aktif tetap dipantau.")
        except Exception as exc: log.warning(f"[autostop] {exc}")

# ---------- Better analysis-row export for normalized PnL semantics ----------
_ORIGINAL_TRADE_ANALYSIS_ROWS_V100 = globals().get("_trade_analysis_rows")
def _trade_analysis_rows_final(hist):
    rows=_ORIGINAL_TRADE_ANALYSIS_ROWS_V100(hist) if callable(_ORIGINAL_TRADE_ANALYSIS_ROWS_V100) else []
    by_id={x.get("trade_id"):x for x in rows if isinstance(x,dict)}
    for rec in hist:
        r=by_id.get(rec.get("trade_id"))
        if r is not None:
            for k in ("price_move_pct","position_pnl_pct","account_impact_pct","configured_sl_pct","actual_exit_price","pnl_semantics"):
                r[k]=rec.get(k)
    return rows

# Final public symbol bindings for recovery/monitoring.
_binance_is_scan_paused_original=globals().get("_binance_is_scan_paused")
_resume_binance_and_flush_pending=_resume_binance_and_flush_pending_final
_binance_recovery_loop=_binance_recovery_loop_final
autostop_loop=autostop_loop_final
_trade_analysis_rows=_trade_analysis_rows_final



# =============================================================================
# V110 FINAL GUARDRAILS
# Bybit WS primary realtime market data; Binance WS primary account/order events;
# Binance REST restricted to execution/reconciliation/emergency lanes.
# =============================================================================
V110_MAIN_VERSION = "MAIN-V111-BYBIT-WS-BINANCE-EXECUTION-GUARDED"
try:
    BRAIN_COMPATIBLE_LEGACY_VERSIONS = tuple(dict.fromkeys(list(BRAIN_COMPATIBLE_LEGACY_VERSIONS) + ["STRATEGY-BRAIN-V111-BYBIT-WS-STATS-EVOLUTION-GUARDED"]))
except Exception:
    pass
BINANCE_REST_NORMAL_MIN_INTERVAL_FINAL = max(1.0, float(os.getenv("BINANCE_REST_NORMAL_MIN_INTERVAL", "3.0")))
_V110_RECONCILE_GUARD=threading.RLock()
BINANCE_REST_RECONCILE_MIN_INTERVAL_FINAL = max(15.0, float(os.getenv("BINANCE_REST_RECONCILE_MIN_INTERVAL", "45.0")))
TRAIL_FAILURE_SUPPRESS_SEC_FINAL = max(30.0, float(os.getenv("TRAIL_FAILURE_SUPPRESS_SEC", "120")))
TG_SYSTEM_REPEAT_SUPPRESS_SEC_FINAL = max(30.0, float(os.getenv("TG_SYSTEM_REPEAT_SUPPRESS_SEC", "180")))
PENDING_ENTRY_TIMEOUT_SEC_FINAL = max(60.0, float(os.getenv("PENDING_ENTRY_TIMEOUT_SEC", str(globals().get("PENDING_ENTRY_TIMEOUT_SEC", 900)))))

# ---- scanner liveness: flag != worker health ----
def _v110_scanner_healthy():
    try:
        with scanner_state_lock:
            st=dict(scanner_state)
        thread_ok=bool(auto_thread and auto_thread.is_alive())
        hb=st.get("last_heartbeat") or st.get("last_cycle_at") or 0.0
        fresh=(time.time()-float(hb) <= max(90.0, float(SCAN_INTERVAL)*4)) if hb else False
        return bool(auto_mode and thread_ok and fresh and str(st.get("health") or "RUNNING").upper() not in {"STOPPED","DEAD","STUCK"})
    except Exception:
        return False

def _v110_ensure_scanner_running(chat_id=None, announce=False):
    global auto_mode, auto_thread, active_chat_id
    active_chat_id=chat_id or active_chat_id
    if _v110_scanner_healthy():
        return auto_thread, False
    # A stale flag must never suppress restart.
    auto_mode=True
    try:
        _set_scan_state(enabled=True, coordinator_alive=False, cycle_running=False, last_error=None)
        _set_component_health("scanner", "STARTING", "scanner coordinator restart/starting")
        old=auto_thread
        if old and old.is_alive():
            return old, False
        t=threading.Thread(target=simulation_loop, args=(active_chat_id,), name="scanner-coordinator-final", daemon=True)
        auto_thread=t; t.start(); _SCAN_WAKE.set()
        if announce and active_chat_id:
            tg_send(active_chat_id, "🔎 <b>SCANNER STARTED</b>\nMarket data: <b>Bybit WS</b>\nBackfill: <b>Bybit REST</b>\nExecution: <b>Binance</b>")
        return t, True
    except Exception as exc:
        auto_mode=False
        _set_scan_state(enabled=False, coordinator_alive=False, last_error=str(exc)[:300])
        _set_component_health("scanner", "DEGRADED", str(exc)[:250])
        raise

# ---- rate-limit gate: normal Binance REST is never allowed to consume critical reserve ----
def _v110_binance_rest_allowed(kind="normal", symbol=None):
    now=time.time()
    if _binance_cooldown_remaining()>0 and kind not in {"critical","emergency"}: return False
    try:
        if kind=="normal":
            with _V110_RECONCILE_GUARD:
                last=float(_binance_last_reconcile_by_symbol.get(str(symbol or "_global"),0.0))
            if now-last < BINANCE_REST_RECONCILE_MIN_INTERVAL_FINAL: return False
    except Exception:
        pass
    return True

# ---- trail mutation deduplication / failure backoff ----
_V110_TRAIL_FAILURES={}
_V110_TRAIL_FAILURE_LOCK=threading.RLock()
def _v110_trail_failure_blocked(sym, desired_sl, error):
    import hashlib
    sig=hashlib.sha1(f"{sym}|{desired_sl}|{type(error).__name__}|{str(error)[:160]}".encode()).hexdigest()
    now=time.time()
    with _V110_TRAIL_FAILURE_LOCK:
        row=_V110_TRAIL_FAILURES.get(sym)
        if row and row.get("sig")==sig and now-row.get("at",0.0)<TRAIL_FAILURE_SUPPRESS_SEC_FINAL:
            row["count"]=int(row.get("count",1))+1
            return True
        _V110_TRAIL_FAILURES[sym]={"sig":sig,"at":now,"count":1}
    return False

def _v110_trail_failure_count(sym):
    with _V110_TRAIL_FAILURE_LOCK:
        return int((_V110_TRAIL_FAILURES.get(sym) or {}).get("count",0))

# ---- final Trail monitor: WS breach latch is never lost during Binance cooldown ----
def _v110_process_trail_breach(sym, pos):
    with positions_lock:
        cur=dict(positions.get(sym) or pos)
    if not cur.get("forced_exit_pending"): return False
    if _binance_is_scan_paused(): return False
    try:
        buy=str(cur.get("signal",{}).get("decision") or "BUY").upper()=="BUY"
        closed, exit_price = _verified_market_close(sym, buy, "trail_breach", chat_id=cur.get("chat_id") or active_chat_id, max_retries=0)
        if not closed: return False
        # Flat first, then one verified cleanup pass for ALL order families.
        _final_cleanup_after_flat(sym, reason="trail breach close")
        entry=float(cur.get("entry") or 0.0); xp=float(exit_price or cur.get("trail_breach_price") or entry)
        result="trail" if _trade_price_move_pct(entry,xp,cur.get("signal",{}).get("decision"))>=0 else "sl"
        close_position(sym,result,close_price=xp)
        _clear_pending_trail(sym)
        with positions_lock:
            if sym in positions: positions[sym]["forced_exit_pending"]=False
        return True
    except BinanceCooldownError:
        return False
    except Exception as exc:
        log.warning(f"[TRAIL BREACH V110] {sym}: {exc}")
        return False

# ---- final real position monitor: Binance WS flat event has priority over polling ----
def monitor_position_real_v110(sym, pos):
    next_strategy=0.0; next_rest=0.0
    while True:
        try:
            with positions_lock:
                if sym not in positions: return
                pos=positions[sym]
            if pos.get("timeout_flag"):
                _verified_timeout_symbol(sym, pos.get("chat_id") or active_chat_id, reason="manual timeout"); return
            # Binance WS account state is authoritative for position existence.
            if _binance_ws_fresh():
                wspos=_binance_ws_position(sym)
                if wspos is not None:
                    try:
                        amt=float(wspos.get("pa") or wspos.get("positionAmt") or 0.0)
                        if abs(amt)<=0:
                            px=_final_bybit_price(sym) or pos.get("current_price") or pos.get("entry")
                            _finalize_external_close_final(sym,pos,reason_hint=_infer_close_reason(pos.get("tp_order_id"),pos.get("sl_order_id")),exit_price=px)
                            _final_cleanup_after_flat(sym, reason="binance WS flat")
                            return
                        with positions_lock:
                            if sym in positions: positions[sym]["quantity"]=abs(amt); positions[sym]["exchange_synced_at"]=time.time()
                    except Exception: pass
            px=_final_bybit_price(sym)
            if px is not None:
                with positions_lock:
                    if sym in positions:
                        positions[sym]["current_price"]=px
                        _update_trade_path_metrics(positions[sym],px); pos=positions[sym]
                _trail_breach_price_check(sym,pos,px)
            if pos.get("forced_exit_pending") and not _binance_is_scan_paused():
                if _v110_process_trail_breach(sym,pos): return
            if time.time()>=next_strategy:
                upd=_strategy_position_update(sym,pos); next_strategy=time.time()+STRATEGY_MANAGE_INTERVAL
                if isinstance(upd,dict):
                    oldsl=pos.get("current_sl",pos.get("signal",{}).get("sl")); oldtp=pos.get("signal",{}).get("tp")
                    cand_sl=upd.get("sl"); cand_tp=upd.get("tp") if upd.get("tp") is not None else oldtp
                    buy=str(pos.get("signal",{}).get("decision") or "BUY").upper()=="BUY"
                    if cand_sl is not None and oldsl is not None and not ((float(cand_sl)>float(oldsl)) if buy else (float(cand_sl)<float(oldsl))): cand_sl=oldsl
                    if cand_sl is not None and oldsl is not None and float(cand_sl)!=float(oldsl):
                        if _binance_is_scan_paused():
                            _queue_pending_trail(sym,float(cand_sl),cand_tp,pos.get("quantity"),reason="strategy",side=pos.get("signal",{}).get("decision"))
                            _trail_breach_price_check(sym,pos,px)
                            if not _v110_trail_failure_blocked(sym,float(cand_sl),BinanceCooldownError("queued while Binance paused")):
                                _notify_trail_update(active_chat_id,sym,pos,upd,oldsl,cand_sl,status="QUEUED")
                        else:
                            try:
                                if not _v110_binance_rest_allowed("normal",sym):
                                    _queue_pending_trail(sym,float(cand_sl),cand_tp,pos.get("quantity"),reason="rest-governor",side=pos.get("signal",{}).get("decision")); next_strategy=time.time()+15
                                else:
                                    latest=_binance_ws_position(sym) if _binance_ws_fresh() else None
                                    live_qty=abs(float((latest or {}).get("pa") or (latest or {}).get("positionAmt") or 0.0)) or float(pos.get("quantity") or 0.0)
                                    if live_qty>0:
                                        with _binance_critical_context(): _cancel_all_algo_orders_verified(sym)
                                        nt,ns=place_tp_sl(sym,buy,cand_tp,float(cand_sl),live_qty)
                                        with positions_lock:
                                            if sym in positions: positions[sym].update({"current_sl":float(cand_sl),"signal":{**positions[sym].get("signal",{}),"sl":float(cand_sl)},"tp_order_id":nt.get("algoId"),"sl_order_id":ns.get("algoId"),"quantity":live_qty})
                                        _clear_pending_trail(sym); _notify_trail_update(active_chat_id,sym,positions[sym],upd,oldsl,cand_sl,status="APPLIED")
                            except BinanceCooldownError as exc:
                                _queue_pending_trail(sym,float(cand_sl),cand_tp,pos.get("quantity"),reason="strategy",side=pos.get("signal",{}).get("decision"))
                                if not _v110_trail_failure_blocked(sym,float(cand_sl),exc): _notify_trail_update(active_chat_id,sym,pos,upd,oldsl,cand_sl,status="QUEUED",error=exc)
                            except Exception as exc:
                                if not _v110_trail_failure_blocked(sym,float(cand_sl),exc): _notify_trail_update(active_chat_id,sym,pos,upd,oldsl,cand_sl,status="FAILED",error=exc)
            # Only stale Binance WS permits REST reconciliation, with a large interval.
            if (not _binance_ws_fresh()) and time.time()>=next_rest and _v110_binance_rest_allowed("normal",sym):
                try:
                    real=_ORIG_GET_REAL_POSITION(sym) if callable(_ORIG_GET_REAL_POSITION) else None
                    _mark_binance_reconcile(sym)
                    if real is None:
                        next_rest=time.time()+BINANCE_REST_RECONCILE_MIN_INTERVAL_FINAL
                    elif abs(float(real.get("positionAmt",0) or 0))<=0:
                        px=_final_bybit_price(sym) or pos.get("current_price") or pos.get("entry")
                        _finalize_external_close_final(sym,pos,reason_hint="unknown",exit_price=px); _final_cleanup_after_flat(sym); return
                    else:
                        next_rest=time.time()+BINANCE_REST_RECONCILE_MIN_INTERVAL_FINAL
                except Exception as exc:
                    log.warning(f"[monitor_real/V110] {sym}: {exc}"); next_rest=time.time()+BINANCE_REST_RECONCILE_MIN_INTERVAL_FINAL
            time.sleep(MONITOR_SLEEP)
        except Exception as exc:
            log.exception(f"[monitor_real/V110] {sym}: {exc}"); time.sleep(MONITOR_SLEEP)

# ---- final pending entry watcher: WS first, REST only controlled fallback ----
def _wait_entry_real_v110(sym,signal,chat_id,order_id):
    deadline=time.time()+8*3600; next_rest=0.0
    while time.time()<deadline:
        try:
            with positions_lock:
                if sym not in positions: return
                pos=positions[sym]
                timeout=bool(pos.get("timeout_flag"))
            event=None
            with _binance_ws_state_lock: event=dict(_binance_ws_orders.get(str(order_id)) or {})
            status=str(event.get("X") or event.get("status") or "").upper()
            if status=="FILLED":
                actual=float(event.get("ap") or event.get("avgPrice") or signal.get("entry") or 0); _open_position_real(sym,signal,actual,chat_id,event); return
            if status in {"CANCELED","EXPIRED","REJECTED","EXPIRED_IN_MATCH"}:
                with positions_lock: positions.pop(sym,None)
                _record_pending_cancel("binance_reject"); return
            with positions_lock:
                created=float(positions.get(sym,{}).get("entry_created_at") or time.time())
            if timeout or time.time()-created>=PENDING_ENTRY_TIMEOUT_SEC_FINAL:
                try:
                    with _binance_critical_context():
                        cancel_order(sym,order_id)
                        st=get_order_status(sym,order_id)
                    if str(st.get("status") or "").upper()=="FILLED":
                        _open_position_real(sym,signal,float(st.get("avgPrice") or signal.get("entry") or 0),chat_id,st); return
                    with positions_lock: positions.pop(sym,None)
                    _record_pending_cancel("expired");
                    if chat_id: tg_send(chat_id,f"⏱️ <b>PENDING TIMEOUT</b> — {sym}\nHanya pending entry yang dibatalkan setelah {PENDING_ENTRY_TIMEOUT_SEC_FINAL/60:.0f} menit.")
                    return
                except Exception as exc:
                    _force_position_emergency(sym,str(exc)[:300]); return
            if (not _binance_ws_fresh() or time.time()>=next_rest) and not _binance_is_scan_paused() and _v110_binance_rest_allowed("normal",sym):
                try:
                    with _binance_critical_context(): st=get_order_status(sym,order_id)
                    next_rest=time.time()+BINANCE_REST_RECONCILE_MIN_INTERVAL_FINAL
                    stt=str(st.get("status") or "").upper()
                    if stt=="FILLED": _open_position_real(sym,signal,float(st.get("avgPrice") or signal.get("entry") or 0),chat_id,st); return
                    if stt in {"CANCELED","EXPIRED","REJECTED"}:
                        with positions_lock: positions.pop(sym,None)
                        _record_pending_cancel("binance_reject"); return
                except BinanceCooldownError: next_rest=time.time()+30
                except Exception as exc: log.warning(f"[pending/V110] REST fallback {sym}: {exc}")
            time.sleep(1.0)
        except Exception as exc:
            log.warning(f"[pending/V110] {sym}: {exc}"); time.sleep(2.0)

# ---- order cleanup: verify flat -> cancel all ordinary/algo -> clear queues ----
def _v110_cleanup_after_flat(sym, reason="flat"):
    try:
        # Existing coordinator already verifies each class; call only once per generation.
        return bool(_cancel_all_symbol_orders_verified(sym))
    except Exception as exc:
        _queue_pending_cleanup(sym, f"V110 flat cleanup: {reason}", exc)
        return False

# ---- richer PnL semantics ----
def _v110_enrich_last_trade(sym, entry, exit_price, decision, quantity, configured_sl_price=None):
    try:
        with trade_history_lock:
            candidates=[x for x in trade_history if x.get("symbol")==sym]
            rec=candidates[-1] if candidates else None
            if not rec: return
            pm=_trade_price_move_pct(entry,exit_price,decision)
            q=abs(float(quantity or 0.0)); net=_trade_net_position_pnl_usd(entry,exit_price,decision,q,0.0)
            anchor=STARTING_BALANCE if not REAL_TRADE_ENABLED else (real_balance_snapshot or STARTING_BALANCE)
            rec.update({"price_move_pct":pm,"position_pnl_pct":float(rec.get("pct",0.0) or 0.0),"account_impact_pct":_compute_account_impact_pct(float(rec.get("pnl_usd",net) or net),anchor),"configured_sl_pct":(abs(float(entry)-float(configured_sl_price))/abs(float(entry))*100.0 if configured_sl_price else rec.get("configured_sl_pct")),"actual_exit_price":exit_price,"pnl_semantics":{"price_move_pct":"market movement after side normalization","position_pnl_pct":"canonical realized position return","account_impact_pct":"net realized PnL divided by statistics balance anchor"}})
    except Exception as exc: log.warning(f"[STATS V110] {sym}: {exc}")

# ---- human-readable FULL status + scanner status ----
def _v110_full_text():
    try:
        st=_brain_full_command("status")
    except Exception as exc:
        return f"🧠 <b>FULL STATUS</b>\nStatus error: <code>{html.escape(str(exc)[:250])}</code>"
    if isinstance(st,str): return st
    return _format_brain_status_human(st)

def fmt_runtime_status_v110():
    s=get_scanner_status(); by=bybit_market_ws.status(); bn=_binance_ws_status()
    scan_health=str(s.get("health") or "UNKNOWN"); last=s.get("last_cycle_age_sec")
    last_txt=f"{float(last):.0f}s lalu" if isinstance(last,(int,float)) else "—"
    return (
        "📡 <b>RUNTIME STATUS</b>\n"
        f"🔎 Scanner: <b>{scan_health}</b> | cycle <b>{s.get('cycle_count',0)}</b> | last scan <b>{last_txt}</b>\n"
        f"Symbols: <b>{s.get('symbols_processed',0)}/{s.get('symbols_requested',0)}</b> | candidates <b>{s.get('last_candidate_count',0)}</b> | eligible <b>{s.get('last_eligible_count',0)}</b>\n"
        f"📡 Bybit WS: <b>{'✅' if by.get('fresh') else '⚠️'}</b> | tickers {by.get('tickers',0)} | kline buffers {by.get('kline_buffers',0)}\n"
        f"🔐 Binance WS: <b>{'✅' if bn.get('fresh') else '⚠️'}</b> | positions {bn.get('positions',0)} | orders {bn.get('orders',0)}\n"
        f"💳 Binance execution: <b>{'PAUSED' if _binance_is_scan_paused() else 'READY'}</b>\n"
        f"🧠 FULL: {_v110_full_text().replace(chr(10),' • ')[:500]}"
    )

# ---- final Telegram suppression: aggregate repeated identical system failures ----
def tg_send_v110(chat_id, text, *args, **kwargs):
    txt=str(text or ""); upper=txt.upper()
    critical=any(k in upper for k in ("EMERGENCY","POSITION UNPROTECTED","EXECUTION UNKNOWN","FORCED EXIT"))
    system=any(k in upper for k in ("ALGO CLEANUP","UPDATE PROTECTION GAGAL","TRAILING UPDATE","RECOVERY BELUM","BINANCE RATE LIMIT","BINANCE PAUSE","PROTECTION DITUNDA","PENDING ENTRY BELUM"))
    if critical or not system: return _ORIG_TG_SEND(chat_id,text,*args,**kwargs)
    import hashlib
    sig=hashlib.sha1(re.sub(r"\d+(?:\.\d+)?", "#", txt).encode()).hexdigest(); key=(str(chat_id),sig); now=time.time()
    with _TG_SUPPRESS_LOCK:
        row=_TG_SUPPRESS_STATE.get(key,{"last":0.0,"count":0,"suppressed":0})
        last=float(row.get("last",0.0))
        if now-last<TG_SYSTEM_REPEAT_SUPPRESS_SEC_FINAL:
            row["count"]=int(row.get("count",0))+1; row["suppressed"]=int(row.get("suppressed",0))+1; _TG_SUPPRESS_STATE[key]=row; return True
        row["last"]=now; row["count"]=1; row["suppressed"]=int(row.get("suppressed",0)); _TG_SUPPRESS_STATE[key]=row
    return _ORIG_TG_SEND(chat_id,text,*args,**kwargs)

# ---- final /status, /full, /auto and /timeout pending handlers ----
def telegram_command_router_v110(text, chat_id):
    t=str(text or "").strip().lower()
    if t in {"/status","status"}:
        return tg_send(chat_id, fmt_runtime_status_v110())
    if t in {"/full","full","/full status","full status"}:
        return tg_send(chat_id, _v110_full_text())
    if t in {"/auto","auto"}:
        try:
            _,created=_v110_ensure_scanner_running(chat_id,announce=False)
            s=get_scanner_status();
            return tg_send(chat_id, ("🔎 <b>SCANNER STARTED</b>" if created else "🔎 <b>SCANNER HEALTHY</b>")+f"\nCycle: <b>{s.get('cycle_count',0)}</b> | last scan: <b>{s.get('last_cycle_age_sec','—')}s</b>\nMarket data: <b>Bybit WS</b>\nExecution: <b>Binance</b>")
        except Exception as exc:
            return tg_send(chat_id,f"❌ <b>Scanner gagal</b>\n<code>{html.escape(str(exc)[:300])}</code>")
    if t=="/timeout pending" or t=="timeout pending":
        return _verified_timeout_pending_only(chat_id)
    return None

# Patch legacy command handler only for the affected commands by replacing exact branches.
main_source_for_patch = globals().get("__file__", "")

# ---- final aliases; aliases are established before __main__ executes ----
_FINAL_V110_OVERRIDES = {
    "_ensure_scanner_running": _v110_ensure_scanner_running,
    "monitor_position_real": monitor_position_real_v110,
    "_wait_entry_real": _wait_entry_real_v110,
    "fmt_runtime_status": fmt_runtime_status_v110,
    "tg_send": tg_send_v110,
}
globals().update(_FINAL_V110_OVERRIDES)

# Make existing scanner entry point feed the final brain summary and preserve Bybit source marker.
_ORIG_RECORD_SCAN_SUMMARY_V110 = globals().get("_brain_on_scan_summary")
def _brain_on_scan_summary_v110(summary):
    rep=dict(summary or {}); rep.setdefault("source","main_scanner"); rep["market_data_source"]="bybit_ws_primary"
    try:
        fn=_brain_fn("record_scan_summary")
        if callable(fn): return fn(rep, source="main_scanner")
    except Exception as exc: log.warning(f"[BRAIN][SCAN SUMMARY V110] {exc}")
    return _ORIG_RECORD_SCAN_SUMMARY_V110(rep) if callable(_ORIG_RECORD_SCAN_SUMMARY_V110) else None
globals()["_brain_on_scan_summary"]=_brain_on_scan_summary_v110

# Replace Telegram branch text at the source level so the final status formatter is actually used.


# ============================================================
# V120 — BINANCE REST SINGLE GATE / WS-FIRST HARDENING
# Goal: eliminate avoidable Binance REST traffic before touching
# notification UX. All Binance REST calls are serialized here;
# non-critical reads are cached/coalesced; 429/418 create a global
# hard cooldown; scanner/Bybit are never gated by Binance state.
# ============================================================
from dataclasses import dataclass

BINANCE_REST_GLOBAL_MIN_INTERVAL = max(0.80, float(os.getenv("BINANCE_REST_GLOBAL_MIN_INTERVAL", "1.0")))
BINANCE_REST_RECOVERY_INTERVAL = max(2.0, float(os.getenv("BINANCE_REST_RECOVERY_INTERVAL", "3.0")))
BINANCE_REST_READ_TIMEOUT = max(3.0, float(os.getenv("BINANCE_REST_READ_TIMEOUT", "8.0")))
BINANCE_REST_GET_RETRY_COUNT = 1
BINANCE_REST_CACHE_DEFAULT_TTL = 5.0
BINANCE_REST_CACHE = {
    "/fapi/v1/exchangeInfo": 3600.0,
    "/fapi/v1/time": 300.0,
    "/fapi/v1/positionSide/dual": 300.0,
    "/fapi/v2/account": 30.0,
    "/fapi/v2/balance": 30.0,
    "/fapi/v2/positionRisk": 15.0,
    "/fapi/v1/order": 5.0,
    "/fapi/v1/openOrders": 15.0,
    "/fapi/v1/openAlgoOrders": 15.0,
    "/fapi/v1/algoOrder": 5.0,
}
BINANCE_PUBLIC_MARKET_ENDPOINTS = {
    "/fapi/v1/klines", "/fapi/v1/ticker/price", "/fapi/v1/ticker/24hr",
}

_binance_rest_state_lock = threading.RLock()
_binance_rest_cache = {}
_binance_rest_inflight = {}
_binance_rest_last_request_mono = 0.0
_binance_rest_probe_at = 0.0
_binance_rest_metrics = {
    "requests": 0, "success": 0, "errors": 0, "cache_hits": 0,
    "singleflight_hits": 0, "rate_limited": 0, "unknown_mutations": 0,
    "coalesced": 0, "last_endpoint": None, "last_status": None,
    "last_at": None, "last_429_at": None, "last_418_at": None,
}

@dataclass
class _BinanceInflight:
    event: object
    result: object = None
    error: object = None

class BinanceMarketDataDisabled(ConnectionError):
    """Legacy Binance market-data path is intentionally disabled."""


def _binance_rest_key(method, path, params=None):
    try:
        return (str(method).upper(), str(path), tuple(sorted((str(k), str(v)) for k, v in (params or {}).items())))
    except Exception:
        return (str(method).upper(), str(path), repr(params or {}))


def _binance_rest_cache_ttl(method, path):
    if str(method).upper() != "GET":
        return 0.0
    return float(BINANCE_REST_CACHE.get(str(path), BINANCE_REST_CACHE_DEFAULT_TTL))


def _binance_rest_cache_get(key, force=False):
    if force:
        return None
    now=time.time()
    with _binance_rest_state_lock:
        row=_binance_rest_cache.get(key)
        if not row:
            return None
        if now >= float(row.get("expires_at",0.0)):
            _binance_rest_cache.pop(key,None)
            return None
        _binance_rest_metrics["cache_hits"] += 1
        return row.get("value")


def _binance_rest_cache_put(key, value, ttl):
    if ttl <= 0:
        return
    with _binance_rest_state_lock:
        _binance_rest_cache[key] = {"value": value, "expires_at": time.time()+ttl}


def _binance_rest_mark_request(endpoint, status=None):
    with _binance_rest_state_lock:
        _binance_rest_metrics["requests"] += 1
        _binance_rest_metrics["last_endpoint"] = endpoint
        _binance_rest_metrics["last_status"] = status
        _binance_rest_metrics["last_at"] = time.time()


def _binance_rest_wait_for_global_slot(critical=False, mutation=False):
    """Global pacing. Critical traffic still shares one gate; only the post-ban
    recovery phase is intentionally slower to avoid immediately re-triggering 418."""
    global _binance_rest_last_request_mono
    interval = BINANCE_REST_RECOVERY_INTERVAL if _binance_recovering else BINANCE_REST_GLOBAL_MIN_INTERVAL
    now_mono=time.monotonic()
    with _binance_rest_state_lock:
        wait=max(0.0, interval-(now_mono-_binance_rest_last_request_mono))
    if wait>0:
        time.sleep(wait)
    with _binance_rest_state_lock:
        _binance_rest_last_request_mono=time.monotonic()


def _binance_parse_retry_after(response):
    raw=None
    try: raw=response.headers.get("Retry-After")
    except Exception: pass
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError,ValueError):
        return None


def _binance_register_rate_limit_response(response, body=""):
    status=int(getattr(response,"status_code",0) or 0)
    retry_after=_binance_parse_retry_after(response)
    if status not in (418,429):
        return False
    if status==418:
        with _binance_rest_state_lock: _binance_rest_metrics["last_418_at"]=time.time()
    else:
        with _binance_rest_state_lock: _binance_rest_metrics["last_429_at"]=time.time()
    with _binance_rest_state_lock: _binance_rest_metrics["rate_limited"]+=1
    _binance_register_ban(str(body or ""), retry_after=retry_after, fallback_seconds=(retry_after or (120.0 if status==418 else 60.0)))
    return True


@contextmanager
def _binance_request_slot_v120(critical=False):
    """Final global REST gate. A Binance ban blocks every REST caller until expiry."""
    _binance_wait_if_banned()
    _binance_rest_wait_for_global_slot(critical=critical, mutation=critical)
    _binance_wait_if_banned()
    yield


@contextmanager
def _binance_critical_context_v120(force_reconcile=True):
    prev_critical=bool(getattr(_binance_priority_local,"critical",False))
    prev_force=bool(getattr(_binance_priority_local,"force_reconcile",False))
    _binance_priority_local.critical=True
    _binance_priority_local.force_reconcile=bool(force_reconcile)
    try:
        yield
    finally:
        _binance_priority_local.critical=prev_critical
        _binance_priority_local.force_reconcile=prev_force


def _binance_signed_impl_v120(method, path, params=None, critical=False):
    """Single implementation for Binance REST. GET reads are cached/single-flight;
    mutations are never blindly retried. -1021 is the only deterministic mutation retry."""
    global BINANCE_API_KEY,BINANCE_API_SECRET,BINANCE_KEYS_PRESENT
    key,secret=_read_binance_credentials()
    if key and secret:
        BINANCE_API_KEY,BINANCE_API_SECRET=key,secret; BINANCE_KEYS_PRESENT=True
    else:
        BINANCE_KEYS_PRESENT=False
    if not BINANCE_KEYS_PRESENT:
        raise RuntimeError("BINANCE_API_KEY/SECRET tidak tersedia di runtime Render")

    method=str(method).upper(); path=str(path); base_params=dict(params or {})
    if path in BINANCE_PUBLIC_MARKET_ENDPOINTS:
        raise BinanceMarketDataDisabled(f"Binance market-data endpoint disabled: {path}; gunakan Bybit WS/REST")
    mutating=method in {"POST","PUT","DELETE"}
    critical=bool(critical or getattr(_binance_priority_local,"critical",False) or mutating)
    force_read=bool(getattr(_binance_priority_local,"force_reconcile",False))
    cache_key=_binance_rest_key(method,path,base_params)
    if not mutating:
        cached=_binance_rest_cache_get(cache_key,force=force_read)
        if cached is not None:
            return cached

        owner=False
        with _binance_rest_state_lock:
            slot=_binance_rest_inflight.get(cache_key)
            if slot is None:
                slot=_BinanceInflight(threading.Event()); _binance_rest_inflight[cache_key]=slot; owner=True
            else:
                _binance_rest_metrics["singleflight_hits"]+=1; _binance_rest_metrics["coalesced"]+=1
        if not owner:
            slot.event.wait(timeout=BINANCE_REST_READ_TIMEOUT+5)
            if slot.error is not None: raise slot.error
            if slot.result is not None: return slot.result
            raise RuntimeError(f"Binance single-flight timeout: {path}")
    else:
        slot=None

    try:
        attempts=2 if (not mutating) else 1
        time_resync=False
        last_err=None
        for attempt in range(attempts):
            try:
                _binance_wait_if_banned()
                with _binance_time_sync_lock:
                    stale=(time.time()-_binance_time_sync_at)>=BINANCE_TIME_SYNC_TTL
                if stale and path != "/fapi/v1/time":
                    try: _binance_sync_time(force=False)
                    except Exception: pass
                with _binance_request_slot_v120(critical=critical):
                    req=dict(base_params)
                    if path!="/fapi/v1/time":
                        req["timestamp"]=_binance_timestamp_ms(sync_if_stale=False)
                        req["recvWindow"]=10000
                    if method in {"POST","PUT","DELETE","GET"}:
                        query=urllib.parse.urlencode(req,safe=",")
                        sig=hmac.new(BINANCE_API_SECRET.encode(),query.encode(),hashlib.sha256).hexdigest() if method!="GET" or path!="/fapi/v1/time" else None
                        url=f"{FAPI}{path}"
                        if sig: url += f"?{query}&signature={sig}"
                        elif query: url += f"?{query}"
                        headers={"X-MBX-APIKEY":BINANCE_API_KEY} if path!="/fapi/v1/time" else {}
                        r=requests.request(method,url,headers=headers,timeout=BINANCE_REST_READ_TIMEOUT,verify=False)
                    else:
                        raise RuntimeError(f"unsupported Binance method {method}")
                used=_binance_update_weight_from_response(r)
                _binance_rest_mark_request(path,r.status_code)
                if _binance_register_rate_limit_response(r,r.text or ""):
                    raise BinanceCooldownError(f"Binance rate limited HTTP {r.status_code}")
                data=r.json()
                if isinstance(data,dict) and "code" in data and isinstance(data.get("code"),int) and data.get("code")<0:
                    code=int(data["code"]); msg=str(data.get("msg") or "")
                    if code==-1003:
                        _binance_register_ban(msg); raise BinanceCooldownError(f"Binance {code}: {msg}")
                    if code==-1021 and not time_resync:
                        time_resync=True
                        _binance_sync_time(force=True)
                        if not mutating: continue
                        raise RuntimeError(f"Binance -1021 on mutation: {msg}")
                    raise RuntimeError(f"Binance {code}: {msg}")
                with _binance_rest_state_lock: _binance_rest_metrics["success"]+=1
                if not mutating:
                    _binance_rest_cache_put(cache_key,data,_binance_rest_cache_ttl(method,path))
                return data
            except BinanceCooldownError:
                raise
            except BinanceMarketDataDisabled:
                raise
            except BinanceUnknownExecutionError:
                raise
            except (requests.Timeout,requests.ConnectionError) as e:
                last_err=e
                with _binance_rest_state_lock: _binance_rest_metrics["errors"]+=1
                if mutating:
                    _binance_rest_metrics["unknown_mutations"]+=1
                    raise BinanceUnknownExecutionError(f"Binance {method} {path} transport error; execution status unknown: {e}") from e
                if attempt+1<attempts:
                    time.sleep(0.75)
                    continue
                raise ConnectionError(f"Binance GET gagal {path}: {e}") from e
            except Exception as e:
                last_err=e
                with _binance_rest_state_lock: _binance_rest_metrics["errors"]+=1
                if mutating:
                    _binance_rest_metrics["unknown_mutations"]+=1
                    raise BinanceUnknownExecutionError(f"Binance {method} {path} response error; execution status unknown: {e}") from e
                if attempt+1<attempts and not isinstance(e, BinanceCooldownError):
                    time.sleep(0.75)
                    continue
                raise
        raise RuntimeError(f"Binance request failed: {path}: {last_err}")
    finally:
        if slot is not None:
            with _binance_rest_state_lock:
                _binance_rest_inflight.pop(cache_key,None)
                slot.event.set()


# Final REST aliases. These assignments happen after the legacy implementations,
# so every runtime caller resolves to this single guarded gateway.
globals()["_binance_request_slot"] = _binance_request_slot_v120
globals()["_binance_critical_context"] = _binance_critical_context_v120
globals()["_binance_signed_impl"] = _binance_signed_impl_v120

# Final signed gateway: mutations still pass through ExecutionController.
def _binance_signed_v120(method,path,params=None,critical=False):
    method_u=str(method).upper()
    if method_u in ExecutionController.MUTATIONS:
        return _execution_controller.submit_signed(method_u,path,params=params,critical=critical)
    return _binance_signed_impl_v120(method_u,path,params=params,critical=critical)

globals()["_binance_signed"] = _binance_signed_v120

# Binance public market data is forbidden; Bybit is the only analysis source.
def fapi_get_v120(path,params=None):
    if str(path) in BINANCE_PUBLIC_MARKET_ENDPOINTS:
        raise BinanceMarketDataDisabled(f"Binance market-data disabled for {path}; scanner must use Bybit")
    _binance_wait_if_banned()
    return _binance_signed_impl_v120("GET",path,params=params,critical=False)

globals()["fapi_get"] = fapi_get_v120

# WS-first position/account helpers: REST only as controlled reconciliation fallback.
_ORIG_GET_REAL_POSITION_V120 = globals().get("get_real_position")
_ORIG_GET_ORDER_STATUS_V120 = globals().get("get_order_status")
_ORIG_GET_OPEN_ORDERS_ALL_V120 = globals().get("get_open_orders_all")
_ORIG_GET_OPEN_ALGO_ORDERS_ALL_V120 = globals().get("get_open_algo_orders_all")
_ORIG_GET_REAL_POSITIONS_ALL_V120 = globals().get("get_real_positions_all")

_bn_position_rest_cache={}
_bn_position_rest_lock=threading.RLock()

def get_real_position_v120(symbol,prefer_ws=True,force=False):
    sym=str(symbol).upper()
    if prefer_ws and _binance_ws_fresh():
        row=_binance_ws_position(sym)
        if row is not None:
            try:
                return row if abs(float(row.get("pa") or row.get("positionAmt") or 0))>0 else None
            except Exception: pass
    key=(sym,)
    now=time.time()
    if not force:
        with _bn_position_rest_lock:
            row=_bn_position_rest_cache.get(key)
            if row and now-row["at"]<BINANCE_REST_RECONCILE_MIN_INTERVAL:
                return dict(row["value"]) if row["value"] is not None else None
    val=_ORIG_GET_REAL_POSITION_V120(sym) if callable(_ORIG_GET_REAL_POSITION_V120) else None
    with _bn_position_rest_lock: _bn_position_rest_cache[key]={"at":time.time(),"value":dict(val) if isinstance(val,dict) else None}
    return val

globals()["get_real_position"] = get_real_position_v120

# Order status: use Binance user-data WS mirror first.
def get_order_status_v120(symbol,order_id):
    oid=str(order_id or "")
    if _binance_ws_fresh():
        with _binance_ws_state_lock:
            row=dict(_binance_ws_orders.get(oid) or {})
        if row:
            row.setdefault("status",row.get("X")); row.setdefault("avgPrice",row.get("ap")); row.setdefault("orderId",row.get("i")); row.setdefault("clientOrderId",row.get("c")); return row
    return _ORIG_GET_ORDER_STATUS_V120(symbol,order_id) if callable(_ORIG_GET_ORDER_STATUS_V120) else None

globals()["get_order_status"] = get_order_status_v120

# Open-order verification remains REST, but global gateway caches/coalesces reads.
def get_open_orders_all_v120(symbol=None):
    return _ORIG_GET_OPEN_ORDERS_ALL_V120(symbol) if callable(_ORIG_GET_OPEN_ORDERS_ALL_V120) else []
globals()["get_open_orders_all"] = get_open_orders_all_v120

def get_open_algo_orders_all_v120(symbol=None):
    return _ORIG_GET_OPEN_ALGO_ORDERS_ALL_V120(symbol) if callable(_ORIG_GET_OPEN_ALGO_ORDERS_ALL_V120) else []
globals()["get_open_algo_orders_all"] = get_open_algo_orders_all_v120

def get_real_positions_all_v120():
    # Binance WS is authoritative for normal monitoring; REST bulk is reconciliation only.
    if _binance_ws_fresh():
        with _binance_ws_state_lock:
            rows=[]
            for row in _binance_ws_positions.values():
                try:
                    if abs(float(row.get("pa") or row.get("positionAmt") or 0))>0: rows.append(dict(row))
                except Exception: continue
            if rows: return rows
    return _ORIG_GET_REAL_POSITIONS_ALL_V120() if callable(_ORIG_GET_REAL_POSITIONS_ALL_V120) else []
globals()["get_real_positions_all"] = get_real_positions_all_v120

# REST diagnostics for /status and troubleshooting.
def get_binance_rest_status_v120():
    with _binance_rest_state_lock: m=dict(_binance_rest_metrics)
    m.update({
        "cooldown_sec": round(_binance_cooldown_remaining(),1),
        "blocked": bool(_binance_is_scan_paused()),
        "last_weight_1m": _binance_weight_1m,
        "weight_age_sec": round(time.time()-_binance_weight_seen_at,1) if _binance_weight_seen_at else None,
        "cache_items": len(_binance_rest_cache),
        "inflight": len(_binance_rest_inflight),
    })
    return m

globals()["get_binance_rest_status"] = get_binance_rest_status_v120

# Ensure startup always initializes market WS before scanner can be reported healthy.
_ORIG_START_RUNTIME_V120 = globals().get("start_runtime")
def start_runtime_v120():
    _v122_runtime_contract_audit()
    out = _ORIG_START_RUNTIME_V120() if callable(_ORIG_START_RUNTIME_V120) else None
    try:
        if not (bybit_market_ws._thread and bybit_market_ws._thread.is_alive()):
            bybit_market_ws.start()
    except Exception as exc:
        log.warning(f"[BYBIT WS] startup: {exc}")
    return out

globals()["start_runtime"] = start_runtime_v120

# ---------- Notification safety AFTER the API path is hardened ----------
_ORIG_TG_SEND_V120 = globals().get("tg_send")
_TG_API_INCIDENT_LOCK = threading.RLock()
_TG_API_INCIDENT_LAST = {}

def tg_send_v120(chat_id,text,*args,**kwargs):
    raw=str(text or "")
    low=raw.lower()
    is_api=("binance rate limit/ban" in low or "[binance pause]" in low or "http 418" in low or "http 429" in low or "binance cooldown" in low or "algo cleanup belum terverifikasi" in low)
    if is_api:
        if "http 418" in low or "http 429" in low or "binance rate limit/ban" in low or "[binance pause]" in low or "binance cooldown" in low:
            kind="binance-rate-limit"
        else:
            kind="binance-protection"
        now=time.time()
        with _TG_API_INCIDENT_LOCK:
            prev=float(_TG_API_INCIDENT_LAST.get(kind,0.0))
            # Initial incident and state change may pass; identical noise inside the window is dropped.
            if now-prev < 300.0:
                return False
            _TG_API_INCIDENT_LAST[kind]=now
    return _ORIG_TG_SEND_V120(chat_id,text,*args,**kwargs)

globals()["tg_send"] = tg_send_v120

# Human-readable REST status extension.
_ORIG_FMT_RUNTIME_STATUS_V120 = globals().get("fmt_runtime_status")
def fmt_runtime_status_v120():
    base=_ORIG_FMT_RUNTIME_STATUS_V120() if callable(_ORIG_FMT_RUNTIME_STATUS_V120) else ""
    bn=get_binance_rest_status_v120()
    extra=(
        "\n\n🔧 <b>BINANCE API</b>\n"
        f"REST state: <b>{'COOLDOWN' if bn['blocked'] else 'READY'}</b> | "
        f"requests {bn['requests']} | cache-hit {bn['cache_hits']} | coalesced {bn['coalesced']}\n"
        f"429: {bn['last_429_at'] or '—'} | 418: {bn['last_418_at'] or '—'} | "
        f"cooldown: {bn['cooldown_sec']}s\n"
        f"Market analysis: <b>Bybit WS</b> | Binance REST market-data: <b>DISABLED</b>\n"
        f"REST last endpoint: <code>{html.escape(str(bn.get('last_endpoint') or '—'))}</code>"
    )
    return base+extra

globals()["fmt_runtime_status"] = fmt_runtime_status_v120




# ============================================================
# V122 — RUNTIME CONTRACT AUDIT
# ============================================================
def _v122_runtime_contract_audit():
    """Validate critical public contracts at runtime before scanner health is exposed."""
    required_callables=("get_top_coins","get_scan_klines","full_analyze","manage_position")
    for name in required_callables:
        fn=globals().get(name)
        if not callable(fn):
            raise RuntimeError(f"runtime contract missing callable: {name}")
    # Public get_top_coins must accept no required positional/keyword args.
    sig=inspect.signature(globals()["get_top_coins"])
    required=[p for p in sig.parameters.values() if p.default is inspect._empty and p.kind in (inspect.Parameter.POSITIONAL_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD,inspect.Parameter.KEYWORD_ONLY)]
    if required:
        raise RuntimeError(f"get_top_coins unexpectedly requires arguments: {[p.name for p in required]}")
    impl=globals().get("_get_top_coins_impl")
    if not callable(impl):
        raise RuntimeError("runtime contract missing _get_top_coins_impl")
    impl_sig=inspect.signature(impl)
    param=impl_sig.parameters.get("exclude_syms")
    if param is None:
        raise RuntimeError("_get_top_coins_impl missing exclude_syms parameter")
    if param.default is inspect._empty:
        raise RuntimeError("_get_top_coins_impl.exclude_syms must be optional")
    # Brain signatures must retain the stable required arguments.
    for name, req_names in (("full_analyze",("df_h1","df_m15")),("manage_position",("state","df_m15"))):
        bs=inspect.signature(globals()[name])
        missing=[x for x in req_names if x not in bs.parameters]
        if missing: raise RuntimeError(f"{name} missing parameters: {missing}")
    brain_mod=globals().get("_brain")
    if brain_mod is not None:
        for name in ("full_analyze","manage_position","export_checkpoint_state","import_checkpoint_state"):
            if not callable(getattr(brain_mod,name,None)):
                raise RuntimeError(f"brain contract missing: {name}")
    return {"ok":True,"version":"V122_RUNTIME_CONTRACT_HARDENED"}

# ============================================================
# V121 — FINAL TRAFFIC CUT: NEVER PULL BINANCE FOR MARKET PRICE
# Existing legacy monitors sometimes requested get_price(...,
# prefer_binance=True). That defeats the Bybit market-data design.
# Rebind those legacy references to the WS-first position reader and
# make market price unambiguously Bybit-only unless an explicit caller
# invokes a Binance execution/reconciliation endpoint directly.
# ============================================================

def get_price_v121(symbol, prefer_binance=False):
    # Market price for analysis/monitoring is ALWAYS Bybit.
    # Binance REST price endpoints are intentionally disabled.
    return _final_bybit_price(symbol)

globals()["get_price"] = get_price_v121

# Legacy final monitor functions reference this captured name directly.
# Point it to the WS-first/cached position reader so those paths no longer
# fall through to per-loop Binance positionRisk polling.
globals()["_ORIG_GET_REAL_POSITION"] = get_real_position_v120

# Keep the final Binance public market helpers hard-disabled.
def _binance_price_v121(symbol):
    raise BinanceMarketDataDisabled("Binance REST price is disabled; use Bybit WS/REST")

def _binance_klines_v121(symbol, interval, limit):
    raise BinanceMarketDataDisabled("Binance REST klines are disabled; use Bybit WS/REST")

def _binance_top_coins_v121(exclude_syms):
    raise BinanceMarketDataDisabled("Binance REST ticker universe is disabled; use Bybit WS/REST")

globals()["_binance_price"] = _binance_price_v121
globals()["_binance_klines"] = _binance_klines_v121
globals()["_binance_top_coins"] = _binance_top_coins_v121

# /status REST metrics are human-friendly and do not expose raw headers/objects.
def fmt_runtime_status_v121():
    base=fmt_runtime_status_v120()
    bn=get_binance_rest_status_v120()
    cooldown=bn.get("cooldown_sec") or 0
    state="COOLDOWN" if cooldown>0 else "READY"
    return base+(
        "\n📊 <b>REST GUARD</b>\n"
        f"State: <b>{state}</b> | Calls: <b>{bn.get('requests',0)}</b> | "
        f"Cache: <b>{bn.get('cache_hits',0)}</b> | Coalesced: <b>{bn.get('coalesced',0)}</b>\n"
        f"429: <b>{bn.get('last_429_at') or '—'}</b> | 418: <b>{bn.get('last_418_at') or '—'}</b>\n"
        "Market REST: <b>DISABLED</b> | Analysis: <b>Bybit WS</b>"
    )

globals()["fmt_runtime_status"] = fmt_runtime_status_v121


# ============================================================
# V124 — PROTECTION REFERENCE / TRAIL / TIMEOUT HARDENING
# - Binance MARK_PRICE WS is authoritative for MARK_PRICE triggers.
# - Safe trail update never removes old protection before the new SL is verified.
# - -2021 is a deterministic immediate-trigger rejection, not a rate-limit error.
# - Pending trail keeps only the latest desired SL; breach is latched and resolved
#   after Binance recovers.
# - Flat positions converge to zero ordinary + zero algo orders.
# - /timeout all is an explicit global timeout; /timeout pending remains pending-only.
# ============================================================
V124_VERSION = "MAIN-V124-PROTECTION-TIMEOUT-ALL-HARDENED"
BINANCE_MARK_WS_STALE_SEC_V124 = max(3.0, float(os.getenv("BINANCE_MARK_WS_STALE_SEC", "5")))
BINANCE_TRIGGER_GUARD_TICKS_V124 = max(1, int(os.getenv("BINANCE_TRIGGER_GUARD_TICKS", "2")))
TIMEOUT_ALL_LOCK = threading.RLock()
_TIMEOUT_ALL_PENDING = {"requested": False, "at": None, "chat_id": None, "running": False}
_TIMEOUT_ALL_LAST_NOTICE = 0.0

class BinanceImmediateTriggerError(RuntimeError):
    """Protection trigger would already be active on Binance reference price."""
    code = -2021

class BinanceTriggerReferenceUnavailable(RuntimeError):
    """No sufficiently fresh Binance reference price is available for validation."""

class BinanceAlgoCleanupState(RuntimeError):
    """Explicit cleanup state when exchange-side algo order verification fails."""

# ---------- Binance public MARK_PRICE WS (validation only; no REST) ----------
class BinanceMarkPriceWSV124:
    def __init__(self):
        self._lock=threading.RLock()
        self._ws=None
        self._thread=None
        self._stop=threading.Event()
        self._symbols=set()
        self._prices={}
        self._connected=False
        self._last_error=None
        self._last_msg_at=0.0
        self._desired_version=0
        self._connected_version=-1

    def start(self):
        if not _WS_LIB_OK:
            self._last_error="websocket-client unavailable"
            return None
        with self._lock:
            if self._thread and self._thread.is_alive(): return self._thread
            self._stop.clear()
            self._thread=threading.Thread(target=self._run,name="binance-mark-price-ws-v124",daemon=True)
            self._thread.start(); return self._thread

    def stop(self):
        self._stop.set()
        try:
            if self._ws: self._ws.close()
        except Exception: pass

    def set_symbols(self, symbols):
        desired={str(s).upper() for s in (symbols or []) if s}
        if len(desired)>1000:
            desired=set(list(desired)[:1000])
        with self._lock:
            if desired==self._symbols: return
            self._symbols=desired; self._desired_version+=1
            ws=self._ws
        # Rebuild combined stream on next loop. This changes rarely (position set changes).
        if ws:
            try: ws.close()
            except Exception: pass

    def get(self,symbol):
        with self._lock:
            row=self._prices.get(str(symbol).upper())
            return dict(row) if row else None

    def get_fresh(self,symbol):
        row=self.get(symbol)
        if not row: return None
        if time.time()-float(row.get("recv_at",0.0))>BINANCE_MARK_WS_STALE_SEC_V124: return None
        return row

    def status(self):
        with self._lock:
            return {"thread_alive":bool(self._thread and self._thread.is_alive()),"connected":self._connected,"symbols":len(self._symbols),"prices":len(self._prices),"fresh":bool(self._last_msg_at and time.time()-self._last_msg_at<=BINANCE_MARK_WS_STALE_SEC_V124),"last_msg_at":self._last_msg_at,"last_error":self._last_error}

    def _run(self):
        backoff=1.0
        while not self._stop.is_set():
            with self._lock:
                syms=sorted(self._symbols); version=self._desired_version
            if not syms:
                time.sleep(1.0); continue
            try:
                streams="/".join(f"{s.lower()}@markPrice@1s" for s in syms)
                url=f"wss://fstream.binance.com/stream?streams={streams}"
                ws=websocket.WebSocketApp(url,on_open=self._on_open,on_message=self._on_message,on_error=self._on_error,on_close=self._on_close)
                with self._lock:
                    self._ws=ws; self._connected_version=version
                ws.run_forever(ping_interval=60,ping_timeout=20)
                backoff=1.0
            except Exception as exc:
                with self._lock: self._last_error=str(exc)[:300]
            finally:
                with self._lock:
                    self._connected=False; self._ws=None
            if self._stop.is_set(): break
            time.sleep(backoff); backoff=min(30.0,backoff*2.0)

    def _on_open(self,ws):
        with self._lock:
            self._connected=True; self._last_error=None; self._last_msg_at=time.time()

    def _on_error(self,ws,error):
        with self._lock: self._last_error=str(error)[:300]

    def _on_close(self,ws,code,msg):
        with self._lock: self._connected=False

    def _on_message(self,ws,raw):
        try: msg=json.loads(raw)
        except Exception: return
        data=msg.get("data") if isinstance(msg,dict) else None
        if not isinstance(data,dict): return
        sym=str(data.get("s") or "").upper()
        raw_p=data.get("p")
        try: price=float(raw_p)
        except Exception: return
        now=time.time()
        with self._lock:
            self._last_msg_at=now
            self._prices[sym]={"symbol":sym,"price":price,"event_at":float(data.get("E") or now*1000)/1000.0,"recv_at":now,"source":"binance_mark_price_ws"}

_binance_mark_ws_v124=BinanceMarkPriceWSV124()

def _sync_binance_mark_ws_symbols_v124():
    with positions_lock:
        syms=set(str(s).upper() for s,p in positions.items() if _position_is_real(p))
    with _pending_trails_lock:
        syms.update(str(s).upper() for s in _pending_trails.keys())
    _binance_mark_ws_v124.set_symbols(syms)


def _binance_mark_watchdog_v124():
    while not SHUTDOWN_EVENT.wait(2):
        try: _sync_binance_mark_ws_symbols_v124()
        except Exception: pass

_BINANCE_MARK_WATCHDOG_THREAD=None

def _start_binance_mark_ws_v124():
    global _BINANCE_MARK_WATCHDOG_THREAD
    _binance_mark_ws_v124.start()
    if not _BINANCE_MARK_WATCHDOG_THREAD or not _BINANCE_MARK_WATCHDOG_THREAD.is_alive():
        _BINANCE_MARK_WATCHDOG_THREAD=threading.Thread(target=_binance_mark_watchdog_v124,name="binance-mark-watchdog-v124",daemon=True)
        _BINANCE_MARK_WATCHDOG_THREAD.start()


def _binance_mark_price_v124(symbol, allow_rest_fallback=False):
    row=_binance_mark_ws_v124.get_fresh(symbol)
    if row is not None: return float(row["price"])
    if allow_rest_fallback:
        try:
            data=_binance_signed_impl_v120("GET","/fapi/v1/premiumIndex",{"symbol":str(symbol).upper()},critical=True)
            if isinstance(data,list): data=(data[0] if data else {})
            if isinstance(data,dict) and data.get("markPrice") is not None:
                return float(data["markPrice"])
        except Exception as exc:
            raise BinanceTriggerReferenceUnavailable(f"Binance mark price unavailable: {exc}") from exc
    raise BinanceTriggerReferenceUnavailable(f"Binance MARK_PRICE WS stale/unavailable: {symbol}")


def _validate_conditional_trigger_v124(symbol, close_side, order_type, trigger_price, *, allow_rest_fallback=True):
    info=get_symbol_filters(symbol); tick=max(float(info.get("tickSize") or 0.0),1e-12)
    trigger=round_to_tick(float(trigger_price),tick)
    mark=_binance_mark_price_v124(symbol,allow_rest_fallback=allow_rest_fallback)
    guard=max(tick*BINANCE_TRIGGER_GUARD_TICKS_V124, abs(mark)*1e-7)
    side=str(close_side).upper(); typ=str(order_type).upper()
    # STOP: BUY triggers when mark >= trigger, SELL when mark <= trigger.
    # TAKE_PROFIT is opposite for the same close side.
    invalid=False
    if typ in {"STOP","STOP_MARKET"}:
        invalid=(mark+guard>=trigger) if side=="BUY" else (mark-guard<=trigger)
    elif typ in {"TAKE_PROFIT","TAKE_PROFIT_MARKET"}:
        invalid=(mark-guard<=trigger) if side=="BUY" else (mark+guard>=trigger)
    if invalid:
        raise BinanceImmediateTriggerError(f"Binance -2021 guard: {symbol} {typ} {side} trigger={trigger:.12g} mark={mark:.12g} guard={guard:.12g}")
    return {"trigger_price":trigger,"mark_price":mark,"guard":guard,"working_type":"MARK_PRICE"}


def _validate_protection_pair_before_mutation_v124(symbol,is_buy,tp_price,sl_price):
    close_side="SELL" if is_buy else "BUY"
    tp=_validate_conditional_trigger_v124(symbol,close_side,"TAKE_PROFIT_MARKET",tp_price,allow_rest_fallback=True)
    sl=_validate_conditional_trigger_v124(symbol,close_side,"STOP_MARKET",sl_price,allow_rest_fallback=True)
    return tp,sl

# ---------- Safe algo cleanup: never return None as state ----------
def _cancel_algo_order_verified_v124(symbol, algo_id):
    if not algo_id: return {"state":"NO_ID"}
    target=str(algo_id)
    try:
        _binance_signed("DELETE","/fapi/v1/algoOrder",{"algoId":target},critical=True)
    except BinanceUnknownExecutionError as exc:
        # Reconcile instead of retrying the mutation.
        try:
            rows=_get_open_algo_orders(symbol)
            still=any(str(r.get("algoId") or r.get("strategyId") or "")==target for r in rows)
            return {"state":"STILL_PRESENT" if still else "VERIFIED_EMPTY","error":str(exc)[:220]}
        except Exception as verify_exc:
            return {"state":"VERIFY_FAILED","error":str(verify_exc)[:220]}
    except Exception as exc:
        try:
            rows=_get_open_algo_orders(symbol)
            still=any(str(r.get("algoId") or r.get("strategyId") or "")==target for r in rows)
            return {"state":"STILL_PRESENT" if still else "VERIFIED_EMPTY","error":str(exc)[:220]}
        except Exception as verify_exc:
            return {"state":"FAILED","error":str(verify_exc)[:220]}
    try:
        rows=_get_open_algo_orders(symbol)
        still=any(str(r.get("algoId") or r.get("strategyId") or "")==target for r in rows)
        return {"state":"STILL_PRESENT" if still else "VERIFIED_EMPTY"}
    except Exception as exc:
        return {"state":"VERIFY_FAILED","error":str(exc)[:220]}


def _cancel_all_algo_orders_verified_v124(sym,retries=1):
    # One cancel-all mutation maximum per invocation. Verification can be repeated without mutation.
    last=None
    try:
        _binance_signed("DELETE","/fapi/v1/algoOpenOrders",{"symbol":sym},critical=True)
    except BinanceUnknownExecutionError as exc:
        last=exc
    except Exception as exc:
        last=exc
    try:
        rows=_get_open_algo_orders(sym)
        if not rows:
            _clear_pending_cleanup(sym)
            return {"state":"VERIFIED_EMPTY","remaining":0}
        return {"state":"STILL_PRESENT","remaining":len(rows),"error":str(last)[:220] if last else None}
    except Exception as exc:
        return {"state":"VERIFY_FAILED","remaining":None,"error":str(exc)[:220]}


def _cancel_all_symbol_orders_verified_v124(sym):
    # Active-position safety: do not cancel protection before the position is flat.
    try: _cancel_all_ordinary_orders_verified(sym)
    except Exception as exc: return {"state":"ORDINARY_CLEANUP_FAILED","error":str(exc)[:220]}
    algo=_cancel_all_algo_orders_verified_v124(sym)
    if algo.get("state")!="VERIFIED_EMPTY": return {"state":"ALGO_CLEANUP_"+str(algo.get("state")),"algo":algo}
    try:
        ordinary=get_open_orders_all(sym)
    except Exception as exc:
        return {"state":"ORDINARY_VERIFY_FAILED","error":str(exc)[:220]}
    if ordinary: return {"state":"ORDINARY_STILL_PRESENT","remaining":len(ordinary)}
    return {"state":"VERIFIED_EMPTY","ordinary":0,"algo":0}

globals()["_cancel_all_algo_orders_verified"]=_cancel_all_algo_orders_verified_v124

# ---------- Safe single-SL placement/update ----------
_ORIG_PLACE_SL_ORDER_V124=globals().get("place_sl_order")
_ORIG_PLACE_TP_SL_V124=globals().get("place_tp_sl")

def place_sl_order_v124(symbol,is_buy,sl_price,quantity,client_algo_id=None):
    info=get_symbol_filters(symbol); tick=float(info.get("tickSize") or 0.0)
    validation=_validate_conditional_trigger_v124(symbol,"SELL" if is_buy else "BUY","STOP_MARKET",sl_price,allow_rest_fallback=True)
    client_algo_id=client_algo_id or _new_client_id("SL")
    params={"algoType":"CONDITIONAL","symbol":symbol,"side":"SELL" if is_buy else "BUY","type":"STOP_MARKET","triggerPrice":round_to_tick(float(sl_price),tick),"quantity":round_qty(quantity,info["stepSize"],info.get("qtyPrecision",8)),"reduceOnly":"true","workingType":"MARK_PRICE","clientAlgoId":client_algo_id}
    try:
        result=_binance_signed("POST","/fapi/v1/algoOrder",params)
    except BinanceUnknownExecutionError:
        found=_find_open_algo_by_client_id(symbol,client_algo_id)
        if found is not None: return found
        raise
    except Exception as exc:
        msg=str(exc)
        if "-2021" in msg or "immediately trigger" in msg.lower():
            try:
                fn=_brain_fn("record_protection_event")
                if callable(fn): fn({"type":"PROTECTION_REJECTED","symbol":symbol,"kind":"STOP_MARKET","trigger_price":float(sl_price),"reason":"-2021 immediate trigger","error":msg[:300]},source="binance_execution")
            except Exception: pass
            raise BinanceImmediateTriggerError(f"Binance -2021: {msg}") from exc
        raise
    if isinstance(result,dict):
        result.setdefault("_validation",validation)
    return result

globals()["place_sl_order"]=place_sl_order_v124

def place_tp_sl_v124(symbol,is_buy,tp_price,sl_price,quantity):
    # Validate BOTH triggers before any mutation, so an invalid SL never causes TP to be placed first.
    _validate_protection_pair_before_mutation_v124(symbol,is_buy,tp_price,sl_price)
    return _ORIG_PLACE_TP_SL_V124(symbol,is_buy,tp_price,sl_price,quantity) if callable(_ORIG_PLACE_TP_SL_V124) else (None,None)

globals()["place_tp_sl"] = place_tp_sl_v124

# ---------- Safe trail replacement: new SL first, old SL second ----------
def _apply_trail_update_safe_v124(sym,pos,new_sl):
    signal_data=pos.get("signal") or {}; is_buy=str(signal_data.get("decision") or "BUY").upper()=="BUY"
    qty=float(pos.get("quantity") or 0.0)
    if qty<=0: raise RuntimeError(f"{sym}: quantity unavailable")
    current_sl=float(pos.get("current_sl") or signal_data.get("sl") or 0.0)
    desired=float(new_sl)
    info=get_symbol_filters(sym); tick=float(info.get("tickSize") or 0.0); desired=round_to_tick(desired,tick)
    # Validate using Binance MARK_PRICE before any mutation.
    validation=_validate_conditional_trigger_v124(sym,"SELL" if is_buy else "BUY","STOP_MARKET",desired,allow_rest_fallback=True)
    old_algo_id=pos.get("sl_order_id")
    client=_new_client_id("TRL")
    try:
        new_order=place_sl_order_v124(sym,is_buy,desired,qty,client_algo_id=client)
    except BinanceImmediateTriggerError:
        raise
    # New SL must be visible exchange-side before old SL is touched.
    rows=_get_open_algo_orders(sym)
    new_id=str((new_order or {}).get("algoId") or (new_order or {}).get("strategyId") or "")
    matched=False
    for row in rows:
        rid=str(row.get("algoId") or row.get("strategyId") or "")
        if (new_id and rid==new_id) or (_protection_matches(row,sym,"SELL" if is_buy else "BUY","STOP_MARKET",desired,qty,tick,info["stepSize"])):
            matched=True; new_id=new_id or rid; break
    if not matched: raise RuntimeError(f"{sym}: new trail SL not verified on Binance")
    # State commit happens after new SL verification.
    with positions_lock:
        if sym in positions:
            positions[sym]["current_sl"]=desired
            positions[sym]["sl_order_id"]=new_id or positions[sym].get("sl_order_id")
            positions[sym]["signal"]={**positions[sym].get("signal",{}),"sl":desired}
            positions[sym]["protection_state"]="VERIFIED"
    # Only now remove prior SL; TP is untouched.
    old_cleanup=None
    if old_algo_id and str(old_algo_id)!=(new_id or ""):
        old_cleanup=_cancel_algo_order_verified_v124(sym,old_algo_id)
        if old_cleanup.get("state") not in {"VERIFIED_EMPTY","NO_ID"}:
            _queue_pending_cleanup(sym,"old trail SL cleanup pending",RuntimeError(str(old_cleanup)))
    _clear_pending_trail(sym)
    try:
        fn=_brain_fn("record_protection_event")
        if callable(fn): fn({"type":"TRAIL_APPLIED","symbol":sym,"old_sl":current_sl,"new_sl":desired,"mark_price":validation.get("mark_price"),"working_type":"MARK_PRICE"},source="binance_execution")
    except Exception: pass
    return {"ok":True,"new_sl":desired,"new_algo_id":new_id,"old_algo_id":old_algo_id,"old_cleanup":old_cleanup,"validation":validation}

# ---------- Safe trail breach processing ----------
def _process_trail_breach_after_recovery_v124(sym,pos):
    with positions_lock: cur=dict(positions.get(sym) or pos)
    if not cur.get("forced_exit_pending"): return False
    if _binance_is_scan_paused(): return False
    buy=str(cur.get("signal",{}).get("decision") or "BUY").upper()=="BUY"
    try:
        closed,exit_price=_verified_market_close(sym,buy,"trail_breach",chat_id=cur.get("chat_id") or active_chat_id,max_retries=0)
        if not closed: return False
        _final_cleanup_after_flat(sym,reason="trail breach close")
        entry=float(cur.get("entry") or 0.0); xp=float(exit_price or cur.get("trail_breach_price") or entry)
        result="trail" if _trade_price_move_pct(entry,xp,cur.get("signal",{}).get("decision"))>=0 else "sl"
        close_position(sym,result,close_price=xp)
        _clear_pending_trail(sym)
        with positions_lock:
            if sym in positions: positions[sym]["forced_exit_pending"]=False
        return True
    except BinanceCooldownError: return False
    except Exception as exc:
        _queue_pending_cleanup(sym,"trail breach close failed",exc); return False

globals()["_process_trail_breach_after_recovery"]=_process_trail_breach_after_recovery_v124

# ---------- Recovery: pending trail first, then protections/cleanup ----------
def _resume_binance_and_flush_pending_v124(chat_id_getter=lambda: active_chat_id):
    global _binance_recovering
    if _binance_cooldown_remaining()>0: return False
    if _binance_recovering: return False
    _binance_recovering=True
    failures=[]
    try:
        _sync_binance_mark_ws_symbols_v124()
        # Breached trail always wins over reinstallation.
        with positions_lock: active={s:dict(p) for s,p in positions.items() if _position_is_real(p)}
        for sym,pos in active.items():
            if pos.get("forced_exit_pending"):
                if not _process_trail_breach_after_recovery_v124(sym,pos):
                    failures.append(f"{sym}: forced trail exit pending")
        with _pending_trails_lock: pending=list((s,dict(v)) for s,v in _pending_trails.items())
        for sym,tr in pending:
            with positions_lock: pos=dict(positions.get(sym) or {})
            if not pos or not _position_is_real(pos): _clear_pending_trail(sym); continue
            try:
                px=_final_bybit_price(sym); _trail_breach_price_check(sym,pos,px)
                if pos.get("forced_exit_pending"):
                    if _process_trail_breach_after_recovery_v124(sym,pos): continue
                    failures.append(f"{sym}: pending trail breached; exit pending"); continue
                qty=float(pos.get("quantity") or tr.get("quantity") or 0.0)
                sl=float(tr.get("sl")); tp=tr.get("tp") or pos.get("signal",{}).get("tp")
                if qty<=0 or tp is None: raise RuntimeError("pending trail incomplete")
                result=_apply_trail_update_safe_v124(sym,pos,sl)
                if pos.get("current_sl")!=sl: pass
            except BinanceImmediateTriggerError as exc:
                _trail_breach_price_check(sym,pos,_final_bybit_price(sym))
                with positions_lock: nowpos=dict(positions.get(sym) or pos)
                if nowpos.get("forced_exit_pending") and _process_trail_breach_after_recovery_v124(sym,nowpos): continue
                failures.append(f"{sym}: trail trigger already crossed")
            except BinanceCooldownError as exc:
                failures.append(f"{sym}: cooldown {exc}")
            except Exception as exc: failures.append(f"{sym}: trail {exc}")
        with _pending_protections_lock: prots=list((s,dict(v)) for s,v in _pending_protections.items())
        for sym,pr in prots:
            try:
                with positions_lock: pos=dict(positions.get(sym) or {})
                if not pos or not _position_is_real(pos): _clear_pending_protection(sym); continue
                qty=float(pos.get("quantity") or pr.get("quantity") or 0.0); buy=str(pr.get("side") or "BUY").upper()=="BUY"; tp=pr.get("tp"); sl=pr.get("sl")
                if qty<=0 or tp is None or sl is None: raise RuntimeError("pending protection incomplete")
                _validate_protection_pair_before_mutation_v124(sym,buy,tp,sl)
                nt,ns=place_tp_sl_v124(sym,buy,tp,sl,qty)
                with positions_lock:
                    if sym in positions: positions[sym].update({"tp_order_id":(nt or {}).get("algoId"),"sl_order_id":(ns or {}).get("algoId"),"protection_state":"VERIFIED"})
                _clear_pending_protection(sym)
            except Exception as exc: failures.append(f"{sym}: protection {exc}")
        with _pending_cleanup_lock: cleans=list(_pending_cleanup.items())
        for sym,_item in cleans:
            try:
                res=_cancel_all_symbol_orders_verified_v124(sym)
                if res.get("state")=="VERIFIED_EMPTY": _clear_pending_cleanup(sym)
                else: failures.append(f"{sym}: cleanup {res.get('state')}")
            except Exception as exc: failures.append(f"{sym}: cleanup {exc}")
        _sync_binance_mark_ws_symbols_v124()
        if failures:
            with _binance_pause_lock: _binance_recovering=False
            _set_component_health("execution","DEGRADED","; ".join(failures[:3])[:300])
            msg=" | ".join(failures[:6])
            cid=chat_id_getter() if callable(chat_id_getter) else active_chat_id
            if cid: tg_send(cid,f"⚠️ <b>Binance recovery belum selesai.</b>\n🔎 Scanner Bybit tetap berjalan.\n<code>{html.escape(msg[:600])}</code>")
            return False
        with _binance_pause_lock: _binance_recovering=False; _binance_scan_paused=False; _binance_pause_reason=""
        cid=chat_id_getter() if callable(chat_id_getter) else active_chat_id
        if cid: tg_send(cid,"✅ <b>Binance recovery selesai.</b>\nExecution/protection kembali konsisten.\n🔎 Scanner Bybit tetap aktif.")
        return True
    except Exception as exc:
        with _binance_pause_lock: _binance_recovering=False
        log.exception(f"[BINANCE RECOVERY V124] {exc}")
        return False

# ---------- Proper global timeout ----------
def _verified_timeout_symbol_v124(sym,chat_id,reason="manual timeout"):
    try:
        with positions_lock: local=dict(positions.get(sym) or {})
        real=get_real_position(sym,prefer_ws=True,force=True)
        live_qty=abs(float(real.get("positionAmt") or 0.0)) if real else 0.0
        if live_qty>0:
            is_buy=float(real.get("positionAmt") or 0.0)>0
            closed,exit_price=_verified_market_close(sym,is_buy,reason,chat_id=chat_id,max_retries=0)
            if not closed: raise RuntimeError("position close not verified")
            _final_cleanup_after_flat(sym,reason=reason)
            if local: close_position(sym,"timeout",close_price=exit_price or local.get("entry"))
        else:
            _final_cleanup_after_flat(sym,reason="timeout flat cleanup")
            if local and local.get("entry_time") is None:
                with positions_lock: positions.pop(sym,None)
                _record_pending_cancel("manual_timeout")
            elif local:
                close_position(sym,"timeout",close_price=local.get("current_price") or local.get("entry"))
        tg_send(chat_id,f"✅ <b>TIMEOUT — {sym}</b>\nPosition: <b>0</b>\nOrdinary orders: <b>0</b>\nAlgo TP/SL/Trail: <b>0</b>")
        return True
    except BinanceCooldownError as exc:
        _queue_pending_cleanup(sym,"timeout deferred by Binance cooldown",exc)
        tg_send(chat_id,f"⏸️ <b>TIMEOUT TERTUNDA — {sym}</b>\nBinance masih cooldown; tidak ada retry agresif.")
        return False
    except Exception as exc:
        _force_position_emergency(sym,str(exc)[:300]); _queue_pending_cleanup(sym,"timeout failed",exc)
        tg_send(chat_id,f"🚨 <b>TIMEOUT BELUM SELESAI — {sym}</b>\n<code>{html.escape(str(exc)[:300])}</code>")
        return False

globals()["_verified_timeout_symbol"]=_verified_timeout_symbol_v124

def _verified_timeout_all_v124(chat_id):
    global _TIMEOUT_ALL_LAST_NOTICE
    with TIMEOUT_ALL_LOCK:
        if _TIMEOUT_ALL_PENDING.get("running"):
            tg_send(chat_id,"⏳ <b>/timeout all</b> sudah sedang diproses."); return False
        _TIMEOUT_ALL_PENDING.update({"requested":True,"at":time.time(),"chat_id":chat_id,"running":True})
    try:
        if _binance_cooldown_remaining()>0:
            tg_send(chat_id,"⏸️ <b>/timeout all diterima.</b>\nBinance cooldown aktif; permintaan global disimpan dan akan diproses saat recovery. Tidak ada retry agresif.")
            return False
        with _binance_critical_context():
            remote_positions=list(get_real_positions_all() or [])
            ordinary=list(get_open_orders_all() or [])
            algo=list(get_open_algo_orders_all() or [])
        remote_by_sym={str(p.get("symbol")).upper():dict(p) for p in remote_positions if p.get("symbol")}
        symbols=set(remote_by_sym)
        symbols.update(str(o.get("symbol")).upper() for o in ordinary if o.get("symbol"))
        symbols.update(str(o.get("symbol")).upper() for o in algo if o.get("symbol"))
        with positions_lock: local_items={s:dict(p) for s,p in positions.items()}
        symbols.update(local_items.keys())
        # Close active real positions FIRST; protection is retained until flat.
        exits={}
        for sym,p in sorted(remote_by_sym.items()):
            qty=abs(float(p.get("positionAmt") or 0.0))
            if qty<=0: continue
            is_buy=float(p.get("positionAmt") or 0.0)>0
            closed,exit_price=_verified_market_close(sym,is_buy,"manual timeout all",chat_id=chat_id,max_retries=0)
            if not closed: raise RuntimeError(f"{sym}: position not flat")
            exits[sym]=exit_price
        # Now cleanup all orders, including symbols that were only orphan orders.
        cleanup_failures=[]
        for sym in sorted(symbols):
            res=_cancel_all_symbol_orders_verified_v124(sym)
            if res.get("state")!="VERIFIED_EMPTY": cleanup_failures.append(f"{sym}:{res.get('state')}")
        # Simulation/pending local state is resolved locally after exchange cleanup.
        for sym,p in local_items.items():
            try:
                if _position_is_real(p):
                    if p.get("entry_time") is not None: close_position(sym,"timeout",close_price=exits.get(sym) or p.get("current_price") or p.get("entry"))
                    else:
                        with positions_lock: positions.pop(sym,None)
                        _record_pending_cancel("manual_timeout_global")
                else:
                    close_position(sym,"timeout",close_price=p.get("current_price") or p.get("entry"))
            except Exception as exc: cleanup_failures.append(f"{sym}:local:{exc}")
        # Final exchange verification.
        with _binance_critical_context():
            rem_pos=[p for p in (get_real_positions_all() or []) if abs(float(p.get("positionAmt") or 0.0))>0]
            rem_ord=list(get_open_orders_all() or [])
            rem_algo=list(get_open_algo_orders_all() or [])
        if rem_pos or rem_ord or rem_algo or cleanup_failures:
            raise RuntimeError(f"global cleanup incomplete positions={len(rem_pos)} ordinary={len(rem_ord)} algo={len(rem_algo)} failures={cleanup_failures[:4]}")
        with positions_lock: positions.clear()
        with _pending_trails_lock: _pending_trails.clear()
        with _pending_protections_lock: _pending_protections.clear()
        with _pending_cleanup_lock: _pending_cleanup.clear()
        tg_send(chat_id,f"✅ <b>/timeout all SELESAI</b>\nPosition: <b>0</b>\nOrdinary orders: <b>0</b>\nAlgo TP/SL/Trail: <b>0</b>\nSymbol diproses: <b>{len(symbols)}</b>")
        return True
    except BinanceCooldownError:
        tg_send(chat_id,"⏸️ <b>/timeout all TERTUNDA</b>\nBinance cooldown aktif; permintaan tetap menunggu recovery.")
        return False
    except Exception as exc:
        tg_send(chat_id,f"🚨 <b>/timeout all BELUM SELESAI</b>\n<code>{html.escape(str(exc)[:500])}</code>")
        log.error(f"[TIMEOUT ALL V124] {exc}")
        return False
    finally:
        with TIMEOUT_ALL_LOCK:
            _TIMEOUT_ALL_PENDING["running"]=False
            # Keep requested=True only while a cooldown/recovery retry remains.
            if _binance_cooldown_remaining()<=0: _TIMEOUT_ALL_PENDING["requested"]=False

globals()["_verified_timeout_all"]=_verified_timeout_all_v124

def _timeout_all_recovery_hook_v124():
    with TIMEOUT_ALL_LOCK:
        pending=bool(_TIMEOUT_ALL_PENDING.get("requested")) and not bool(_TIMEOUT_ALL_PENDING.get("running"))
        cid=_TIMEOUT_ALL_PENDING.get("chat_id")
    if pending and cid and _binance_cooldown_remaining()<=0:
        _verified_timeout_all_v124(cid)

# ---------- Final real monitor: safe trail + breach + no failed-loop spam ----------
def monitor_position_real_v124(sym,pos):
    next_strategy=0.0; next_rest=0.0
    while True:
        with positions_lock:
            if sym not in positions: return
            pos=positions[sym]
        try:
            if pos.get("timeout_flag"):
                _verified_timeout_symbol_v124(sym,pos.get("chat_id") or active_chat_id,reason="manual timeout"); return
            px=_final_bybit_price(sym)
            if px is not None:
                with positions_lock:
                    if sym in positions:
                        positions[sym]["current_price"]=px; _update_trade_path_metrics(positions[sym],px); pos=positions[sym]
                _trail_breach_price_check(sym,pos,px)
            if pos.get("forced_exit_pending") and not _binance_is_scan_paused():
                if _process_trail_breach_after_recovery_v124(sym,pos): return
            if time.time()>=next_strategy:
                upd=_strategy_position_update(sym,pos); next_strategy=time.time()+STRATEGY_MANAGE_INTERVAL
                if isinstance(upd,dict):
                    oldsl=pos.get("current_sl",pos.get("signal",{}).get("sl")); cand_sl=upd.get("sl")
                    if cand_sl is not None and oldsl is not None:
                        buy=str(pos.get("signal",{}).get("decision") or "BUY").upper()=="BUY"
                        if not ((float(cand_sl)>float(oldsl)) if buy else (float(cand_sl)<float(oldsl))): cand_sl=oldsl
                    if cand_sl is not None and oldsl is not None and float(cand_sl)!=float(oldsl):
                        desired=float(cand_sl)
                        try:
                            if _binance_is_scan_paused():
                                _queue_pending_trail(sym,desired,upd.get("tp") or pos.get("signal",{}).get("tp"),pos.get("quantity"),reason="binance-cooldown",side=pos.get("signal",{}).get("decision"))
                                _trail_breach_price_check(sym,pos,px)
                            else:
                                result=_apply_trail_update_safe_v124(sym,pos,desired)
                                _notify_trail_update(active_chat_id,sym,positions.get(sym,pos),upd,oldsl,desired,status="APPLIED")
                        except BinanceImmediateTriggerError as exc:
                            _queue_pending_trail(sym,desired,upd.get("tp") or pos.get("signal",{}).get("tp"),pos.get("quantity"),reason="trigger-crossed",side=pos.get("signal",{}).get("decision"))
                            _trail_breach_price_check(sym,pos,px)
                            if not _v110_trail_failure_blocked(sym,desired,exc):
                                _notify_trail_update(active_chat_id,sym,pos,upd,oldsl,desired,status="QUEUED",error="Binance trigger already crossed; waiting for forced-close/recovery")
                        except BinanceTriggerReferenceUnavailable as exc:
                            _queue_pending_trail(sym,desired,upd.get("tp") or pos.get("signal",{}).get("tp"),pos.get("quantity"),reason="mark-price-unavailable",side=pos.get("signal",{}).get("decision"))
                        except BinanceCooldownError as exc:
                            _queue_pending_trail(sym,desired,upd.get("tp") or pos.get("signal",{}).get("tp"),pos.get("quantity"),reason="binance-cooldown",side=pos.get("signal",{}).get("decision"))
                        except Exception as exc:
                            if not _v110_trail_failure_blocked(sym,desired,exc):
                                _notify_trail_update(active_chat_id,sym,pos,upd,oldsl,desired,status="FAILED",error=exc)
            # REST reconciliation remains slow and only when Binance WS is stale.
            if not _binance_ws_fresh() and time.time()>=next_rest and _v110_binance_rest_allowed("normal",sym):
                try:
                    real=get_real_position_v120(sym,prefer_ws=False,force=False)
                    _mark_binance_reconcile(sym); next_rest=time.time()+BINANCE_REST_RECONCILE_MIN_INTERVAL_FINAL
                    if real is None or abs(float(real.get("positionAmt",0) or 0))<=0:
                        px=_final_bybit_price(sym) or pos.get("current_price") or pos.get("entry")
                        _finalize_external_close_final(sym,pos,reason_hint="unknown",exit_price=px); _final_cleanup_after_flat(sym); return
                except Exception as exc: next_rest=time.time()+BINANCE_REST_RECONCILE_MIN_INTERVAL_FINAL
            _sync_binance_mark_ws_symbols_v124()
            time.sleep(MONITOR_SLEEP)
        except Exception as exc:
            log.exception(f"[monitor_real/V124] {sym}: {exc}"); time.sleep(MONITOR_SLEEP)

globals()["monitor_position_real"] = monitor_position_real_v124

# ---------- Recovery loop wrapper adds pending /timeout all processing ----------
_ORIG_BINANCE_RECOVERY_LOOP_V124=globals().get("_binance_recovery_loop")
def _binance_recovery_loop_v124(chat_id_getter=lambda: active_chat_id):
    consecutive=0
    while not SHUTDOWN_EVENT.wait(5):
        try:
            if _binance_is_scan_paused():
                _notify_binance_pause_once(chat_id_getter())
                if _binance_cooldown_remaining()<=0 and not _binance_recovering:
                    ok=_resume_binance_and_flush_pending_v124(chat_id_getter); consecutive=0 if ok else consecutive+1
                    if ok: _timeout_all_recovery_hook_v124()
                    elif consecutive>3: time.sleep(20)
                else:
                    _timeout_all_recovery_hook_v124() if _binance_cooldown_remaining()<=0 else None
            else:
                _timeout_all_recovery_hook_v124()
        except Exception as exc: log.warning(f"[binance-recovery/V124] {exc}")

globals()["_binance_recovery_loop"]=_binance_recovery_loop_v124

# ---------- Command handler: explicit /timeout all ----------
# Patch the legacy command branch in source-independent runtime by replacing the router,
# and expose a direct function for tests. The original long-poll loop still uses its own
# branch; the exact string replacement below is applied to the generated file after this block.
_ORIG_TELEGRAM_COMMAND_ROUTER_V124 = globals().get("telegram_command_router_v110")
def telegram_command_router_v124(text,chat_id):
    t=str(text or "").strip().lower()
    if t in {"/timeout all","timeout all"}:
        return _verified_timeout_all_v124(chat_id)
    if t in {"/timeout pending","timeout pending"}:
        return _verified_timeout_pending_only(chat_id)
    return _ORIG_TELEGRAM_COMMAND_ROUTER_V124(text,chat_id) if callable(_ORIG_TELEGRAM_COMMAND_ROUTER_V124) else None

globals()["telegram_command_router_v110"] = telegram_command_router_v124

# ---------- Start final mark-price infrastructure ----------
_ORIG_START_RUNTIME_V124=globals().get("start_runtime")
def start_runtime_v124():
    _start_binance_mark_ws_v124()
    return _ORIG_START_RUNTIME_V124() if callable(_ORIG_START_RUNTIME_V124) else None

globals()["start_runtime"] = start_runtime_v124

# ---------- Critical runtime audit ----------
def _v124_final_runtime_audit():
    checks={
        "place_sl_order": callable(globals().get("place_sl_order")),
        "place_tp_sl": callable(globals().get("place_tp_sl")),
        "verified_timeout_symbol": callable(globals().get("_verified_timeout_symbol")),
        "verified_timeout_all": callable(globals().get("_verified_timeout_all")),
        "timeout_pending": callable(globals().get("_verified_timeout_pending_only")),
        "trail_safe": callable(globals().get("_apply_trail_update_safe_v124")),
        "mark_ws": callable(globals().get("_binance_mark_price_v124")),
        "brain_save": callable(getattr(globals().get("_brain"),"export_checkpoint_state",None)),
    }
    bad=[k for k,v in checks.items() if not v]
    if bad: raise RuntimeError(f"V124 critical runtime contract missing: {bad}")
    return checks

# Ensure final audit occurs before main enters the runtime loop.
_v124_final_runtime_audit()



# ============================================================
# V125 — SAFE TP/SL PAIR MUTATION
# Existing protections are never bulk-canceled before a new pair is verified.
# A failed SL create only cleans a newly-created TP from this transaction.
# ============================================================
V125_VERSION="MAIN-V125-SAFE-TP-SL-PAIR"

def _cancel_new_algo_if_known_v125(symbol, order_obj):
    aid=str((order_obj or {}).get("algoId") or (order_obj or {}).get("strategyId") or "")
    if not aid: return {"state":"NO_ID"}
    return _cancel_algo_order_verified_v124(symbol,aid)


def place_tp_sl_v125(symbol,is_buy,tp_price,sl_price,quantity):
    # Validate both triggers against Binance MARK_PRICE before ANY mutation.
    tp_v,sl_v=_validate_protection_pair_before_mutation_v124(symbol,is_buy,tp_price,sl_price)
    info=get_symbol_filters(symbol); tick=float(info.get("tickSize") or 0.0); step=float(info.get("stepSize") or 0.0); qty=round_qty(quantity,step,info.get("qtyPrecision",8))
    close_side="SELL" if is_buy else "BUY"
    tp_client=_new_client_id("TP"); sl_client=_new_client_id("SL")
    tp=None
    try:
        tp=_binance_signed("POST","/fapi/v1/algoOrder",{
            "algoType":"CONDITIONAL","symbol":symbol,"side":close_side,
            "type":"TAKE_PROFIT_MARKET","triggerPrice":round_to_tick(float(tp_price),tick),
            "quantity":qty,"reduceOnly":"true","workingType":"MARK_PRICE","clientAlgoId":tp_client,
        })
    except BinanceUnknownExecutionError:
        tp=_find_open_algo_by_client_id(symbol,tp_client)
        if tp is None: raise
    try:
        sl=place_sl_order_v124(symbol,is_buy,sl_price,quantity,client_algo_id=sl_client)
    except BinanceImmediateTriggerError:
        # Do not touch existing protection. Remove ONLY the TP created by this transaction.
        if tp:
            _cancel_new_algo_if_known_v125(symbol,tp)
        raise
    except Exception:
        if tp:
            _cancel_new_algo_if_known_v125(symbol,tp)
        raise
    rows=_get_open_algo_orders(symbol)
    tp_ok=any(_protection_matches(r,symbol,close_side,"TAKE_PROFIT_MARKET",tp_price,qty,tick,step) for r in rows)
    sl_ok=any(_protection_matches(r,symbol,close_side,"STOP_MARKET",sl_price,qty,tick,step) for r in rows)
    if not (tp_ok and sl_ok):
        # The newly-created pair is not verified. Never bulk cancel pre-existing protection.
        for obj in (sl,tp):
            try: _cancel_new_algo_if_known_v125(symbol,obj)
            except Exception: pass
        raise RuntimeError(f"protection verification gagal: TP={tp_ok}, SL={sl_ok}, algo={len(rows)}")
    return tp,sl

globals()["place_tp_sl"] = place_tp_sl_v125

# Update the explicit global timeout handler to use the V124 safe close/cleanup order
# and do not clear local state until the exchange is confirmed flat/clean.

# Extend the final runtime audit with the new pair contract.
_ORIG_V124_AUDIT=globals().get("_v124_final_runtime_audit")
def _v125_final_runtime_audit():
    result=_ORIG_V124_AUDIT() if callable(_ORIG_V124_AUDIT) else {}
    if not callable(globals().get("place_tp_sl")): raise RuntimeError("V125 place_tp_sl missing")
    return result
_v125_final_runtime_audit()


if __name__ == "__main__":
    start_runtime()
    while not SHUTDOWN_EVENT.wait(15):
        try: _circuit_health_tick()
        except Exception: pass
        if RUNTIME_STATE=="STOPPING": break
    if RUNTIME_STATE!="STOPPING": _graceful_shutdown("main loop exit")


# V122 marker: runtime contract hardening applied.
MAIN_RUNTIME_CONTRACT_VERSION = "V122_RUNTIME_CONTRACT_HARDENED"
