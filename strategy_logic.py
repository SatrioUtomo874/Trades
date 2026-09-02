"""
strategy_logic.py — ADAPTIVE BRAIN v35 (OTAK) (Adaptive RR + Intelligent Target Selection + Predictive Reversal-Aware Trailing)
========================================================================================
Dibangun dari corpus transkrip video SMC/ICT (channel RUANG TRADER, ~39 video:
market structure, order block, FVG, liquidity sweep, inducement, ChoCH/BOS,
CISD, OTE & Fibonacci, external vs internal liquidity, Wyckoff, dsb).

Urutan proses tiap sinyal (SESUAI PERMINTAAN): Entry → SL → TP → Confidence
global (tanpa bias sesi). Arah HTF adalah gate keputusan, bukan sekadar bonus
skor. Setup lokal hanya boleh dipakai setelah POI HTF disentuh dan ada
konfirmasi displacement/structure yang baru. Ini sengaja mengurangi sinyal:
skor agregat lama dapat membuat OTE/RSI/sweep mengalahkan tren utama.

Poin revisi v3:
  1) RR minimal 1:2, maksimal 1:4. Jika target RR<2 → JANGAN auto‑tolak,
     dulu cari target lebih jauh dari pool TP (swing H1 lebih lama, EQ H1,
     fib extension) sebelum menyerah; jika ada RR>4 → cap ke 4 (bukan tolak).
  2) Entry presisi: OB/FVG/EQ dengan bonus konfluen HTF (overlap zona H1) dan
     bonus Inducement (liquidity minor sudah disapu sebelum POI asli) —
     sesuai transkrip "How the Market Traps Traders with Inducement" &
     "trading strategy using inducement [SNIPER ENTRY]".
  3) SL struktural anti‑Liquidity Sweep: SL ditaruh di luar invalidation
     level (OB/struktur) + buffer, supaya jika benar‑benar tersentuh berarti
     arah memang salah, bukan sekadar sweep likuiditas minor. Penanganan
     "SL tersentuh tapi ternyata sweep" (harga lanjut sesuai arah semula)
     ditangani di validate_and_adjust_geometry() (dipanggil main.py saat
     order terisi/di-monitor) dan di monitor_position() main.py (verifikasi
     candle M1 sebelum SL dikonfirmasi).
  4) Trail Ladder: seluruh profit-ladder dihapus. Trail hanya mengikuti
     struktur M15 (HL/LH + buffer), sehingga pergeseran SL berarti struktur
     mulai gagal, bukan sekadar memaksa profit terkunci.
  5) Proses per‑koin: Entry → SL → TP → confidence (global, tanpa sesi).
  6) Tidak ada confidence per‑sesi — confidence murni dari kualitas
     struktur+konfluensi chart (skema sama seperti v2, ditambah bonus baru).
  7) Konsep tambahan (inducement, external/internal liquidity, konfluensi
     HTF) masuk sebagai bonus skor kecil, TIDAK menambah filter penolakan,
     supaya kuantitas sinyal tidak turun.
  8) Entry-location layer: arah bullish/bearish tidak otomatis berarti entry
     sekarang. Kandidat dinilai terhadap range M15, adverse swing, dan timing
     RSI. Harga yang terlalu dekat sisi salah range + momentum RSI yang masih
     melawan entry menjadi WAIT_ENTRY, bukan langsung dipaksa trade.
  9) Candidate fallback: jika kandidat terbaik punya invalidation terlalu lebar
     atau TP tidak mencapai 2R, candidate berikutnya dicoba. Tidak ada synthetic
     ATR stop untuk menyelamatkan entry yang buruk.
 10) Tidak menambahkan logika scalping M1/Silver Bullet/killzone sesi yang
     tidak relevan untuk swing H1/M15 — hanya diambil konsep yang dipakai
     bot ini (structure, OB/FVG, liquidity, CISD, Fibonacci).

Kompatibel dengan main.py:
  - full_analyze(df_h1, df_m15, df_d1=None, symbol=None) → dict | None
  - score_direction(df_h1, df_m15, df_d1=None) → dict | None
  - swing_pts(df, lb) → (sh, sl)
  - TRAIL_R_LADDER, STRUCT_TRAIL_LB, STRUCT_TRAIL_BUF_PCT, STRUCT_TRAIL_LOOKBACK
  - MIN_RR, MAX_RR, FIB_EXT_1, FIB_EXT_2
"""

import logging
from typing import Optional, Tuple
from pathlib import Path
import os
import json
import time
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
import html
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

ML_COGNITIVE_VERSION = "V40_FULL_BRAIN_REBUILT"
V40_VERSION = "V40_FULL_BRAIN_REBUILT"
BRAIN_INTERFACE_VERSION = V40_VERSION
FULL_LEARNING_SCHEMA = "full_learning_cognitive_v2"


MACHINE_LEARNING_SCHEMA = "machine_Learning_v3_cognitive"
_LEARNED_MODEL = None
_LEARNED_MODEL_FILE = Path(os.getenv("FULL_MODEL_FILE", "machine_learning_state/active_model.json"))
_FULL_ENABLED = False

ML_FEATURE_NAMES = [
    "direction_confidence", "setup_quality", "entry_location_score", "rr",
    "range_position", "rsi_timing_score", "direction_edge", "m15_trigger_count",
    "poi_reacted", "selected_sweep", "m15_structure_alignment", "htf_alignment",
    "macro_alignment", "m15_relative_volume", "fib_position", "atr_pct_proxy",
    "entry_distance_atr", "risk_atr", "target_distance_atr", "data_quality",
    "entry_ob", "entry_fvg", "entry_eq", "entry_sweep", "entry_breakout",
    "entry_pullback", "htf_conflict", "m15_ranging"
]

def set_learning_model(model):
    global _LEARNED_MODEL
    _LEARNED_MODEL = model if isinstance(model, dict) and model.get("active") else None


def _load_active_learning_model():
    global _LEARNED_MODEL
    try:
        if _LEARNED_MODEL_FILE.exists():
            obj=json.loads(_LEARNED_MODEL_FILE.read_text(encoding="utf-8"))
            if isinstance(obj,dict) and obj.get("active"):
                _LEARNED_MODEL=obj
    except Exception as exc:
        log.warning(f"[MODEL] active model load gagal: {exc}")

def _save_active_learning_model(model):
    try:
        _LEARNED_MODEL_FILE.parent.mkdir(parents=True,exist_ok=True)
        tmp=_LEARNED_MODEL_FILE.with_suffix(_LEARNED_MODEL_FILE.suffix+".tmp")
        tmp.write_text(json.dumps(model,ensure_ascii=False,allow_nan=False,indent=2,default=str),encoding="utf-8")
        os.replace(tmp,_LEARNED_MODEL_FILE)
        return True
    except Exception as exc:
        log.warning(f"[MODEL] active model save gagal: {exc}")
        return False


def get_learning_model_info():
    if not _LEARNED_MODEL:
        return {"active": False, "model_version": "static", "sample_count": 0}
    return {
        "active": True,
        "model_version": _LEARNED_MODEL.get("model_version", "unknown"),
        "sample_count": int(_LEARNED_MODEL.get("sample_count", 0) or 0),
        "confidence_min": _LEARNED_MODEL.get("confidence_min"),
    }


def _rsi_timing_numeric(value):
    s = str(value or "").lower()
    if "strong" in s:
        return 1.0
    if "support" in s or "aligned" in s:
        return 0.75
    if "neutral" in s:
        return 0.5
    if "weak" in s or "against" in s:
        return 0.25
    return 0.5


def _feature_dict_to_vector(features):
    return np.asarray([float(features.get(k, 0.0) or 0.0) for k in ML_FEATURE_NAMES], dtype=float)


def _build_learning_features(score, loc, candidate, rr, entry, atr, cur_price, risk, htf_poi, poi_reacted):
    direction = str(score.get("direction") or "bull")
    m15_struct = str(score.get("m15_struct") or "ranging")
    htf_bias = str(score.get("htf_bias") or "neutral")
    macro = str(score.get("macro_bias") or "unknown")
    edge = float(score.get("direction_edge", 0) or 0)
    atr = max(float(atr or 0.0), 1e-12)
    entry = float(entry or 0.0)
    cur_price = float(cur_price or entry)
    risk = abs(float(risk or 0.0))
    distance_atr = abs(cur_price - entry) / atr
    risk_atr = risk / atr
    target_distance_atr = abs(float(rr or 0.0) * risk) / atr if risk else 0.0
    loc_pos = float(loc.get("range_position", 0.5) or 0.5)
    fib_r = float(score.get("fib_r", 0.5) or 0.5)
    htf_align = 1.0 if htf_bias == ("bullish" if direction == "bull" else "bearish") else 0.0
    macro_align = 1.0 if macro == ("bullish" if direction == "bull" else "bearish") else 0.0
    label = str(candidate.get("label") or "").lower()
    return {
        "direction_confidence": float(score.get("confidence", 0) or 0),
        "setup_quality": float(candidate.get("score", 0) or 0),
        "entry_location_score": float(loc.get("location_score", 50) or 50),
        "rr": min(float(rr or 0.0), 8.0),
        "range_position": min(max(loc_pos, 0.0), 1.0),
        "rsi_timing_score": _rsi_timing_numeric(loc.get("rsi_timing")),
        "direction_edge": min(max(edge / 100.0, 0.0), 1.0),
        "m15_trigger_count": min(float(score.get("trigger_count", 0) or 0) / 4.0, 1.0),
        "poi_reacted": float(bool(poi_reacted)),
        "selected_sweep": float(bool(score.get("selected_sweep"))),
        "m15_structure_alignment": 1.0 if m15_struct == ("bullish" if direction == "bull" else "bearish") else 0.0,
        "htf_alignment": htf_align,
        "macro_alignment": macro_align,
        "m15_relative_volume": min(max(float(score.get("m15_relative_volume", 1.0) or 1.0) / 2.0, 0.0), 2.0),
        "fib_position": min(max(fib_r, 0.0), 1.0),
        "atr_pct_proxy": min(max((atr / max(cur_price, 1e-12)) * 100.0, 0.0), 25.0),
        "entry_distance_atr": min(max(distance_atr, 0.0), 3.0),
        "risk_atr": min(max(risk_atr, 0.0), 6.0),
        "target_distance_atr": min(max(target_distance_atr, 0.0), 20.0),
        "data_quality": 1.0 if htf_poi is not None else 0.85,
        "entry_ob": float("ob" in label or "order" in label),
        "entry_fvg": float("fvg" in label),
        "entry_eq": float("eq" in label or "equilibrium" in label),
        "entry_sweep": float("sweep" in label),
        "entry_breakout": float("break" in label),
        "entry_pullback": float("pullback" in label or "retest" in label),
        "htf_conflict": float(htf_bias == "conflict"),
        "m15_ranging": float(m15_struct == "ranging"),
    }


_ML_SCHEMA_WARNED = False

def _predict_learning(features):
    global _ML_SCHEMA_WARNED
    if not _LEARNED_MODEL:
        return None
    try:
        model_schema = str(_LEARNED_MODEL.get("schema") or "")
        model_features = _LEARNED_MODEL.get("feature_names")
        if model_schema and model_schema != MACHINE_LEARNING_SCHEMA:
            if not _ML_SCHEMA_WARNED:
                log.warning("[ML] model schema mismatch; ML prediction disabled until a compatible model is trained")
                _ML_SCHEMA_WARNED = True
            return None
        expected = list(model_features) if isinstance(model_features, list) and model_features else list(ML_FEATURE_NAMES)
        if expected != list(ML_FEATURE_NAMES):
            if not _ML_SCHEMA_WARNED:
                log.warning("[ML] model feature schema mismatch; ML prediction disabled until a compatible model is trained")
                _ML_SCHEMA_WARNED = True
            return None
        mean = np.asarray(_LEARNED_MODEL.get("mean", [0.0] * len(expected)), dtype=float)
        scale = np.asarray(_LEARNED_MODEL.get("scale", [1.0] * len(expected)), dtype=float)
        w = np.asarray(_LEARNED_MODEL.get("w", [0.0] * len(expected)), dtype=float)
        b = float(_LEARNED_MODEL.get("b", 0.0) or 0.0)
        x = _feature_dict_to_vector(features)
        if not (len(x) == len(mean) == len(scale) == len(w) == len(expected)):
            if not _ML_SCHEMA_WARNED:
                log.warning("[ML] model dimension mismatch; ML prediction disabled until a compatible model is trained")
                _ML_SCHEMA_WARNED = True
            return None
        z = float(np.dot((x - mean) / np.maximum(scale, 1e-8), w) + b)
        z = max(-35.0, min(35.0, z))
        _classification_prob = 1.0 / (1.0 + np.exp(-z))
        rw = np.asarray(_LEARNED_MODEL.get("rw", [0.0] * len(expected)), dtype=float)
        rb = float(_LEARNED_MODEL.get("rb", 0.0) or 0.0)
        if len(rw) != len(expected):
            if not _ML_SCHEMA_WARNED:
                log.warning("[ML] expected-R model dimension mismatch; ML prediction disabled until a compatible model is trained")
                _ML_SCHEMA_WARNED = True
            return None
        expected_r = float(np.dot((x - mean) / np.maximum(scale, 1e-8), rw) + rb)
        expected_r = float(max(-3.0, min(3.0, expected_r)))
        model_conf = float(50.0 + 25.0 * np.tanh(expected_r))
        probability = float(1.0 / (1.0 + np.exp(-expected_r / 0.75)))
        return {
            "probability": probability,
            "classification_probability": float(_classification_prob),
            "model_confidence": model_conf,
            "expected_r": expected_r,
            "model_version": _LEARNED_MODEL.get("model_version", "unknown"),
            "sample_count": int(_LEARNED_MODEL.get("sample_count", 0) or 0),
            "confidence_min": _LEARNED_MODEL.get("confidence_min"),
        }
    except Exception:
        log.exception("[ML] prediction gagal")
        return None

# =============================================================================
# KONFIGURASI — Diimpor langsung oleh main.py
# =============================================================================

MIN_RR   = 2.0
MAX_RR   = None  # unlimited RR; target tetap harus struktural/berbasis liquidity

# ── Trail Ladder v3 ──────────────────────────────────────────────────────
# HANYA breakeven di 1.0R. Ini bukan "pengaman profit" — cuma menghilangkan
# risiko begitu trade sudah maju 1R, konsisten dengan permintaan: "Trail
# jangan dipaksa profit". Setelah breakeven tercapai, seluruh pergeseran SL
# selanjutnya murni mengikuti struktur M15 (lihat monitor_position main.py
# yang memakai kandidat "paling protektif" antara ladder ini vs structure —
# karena ladder cuma py 1 rung breakeven, structure yang akan mendominasi
# di hampir semua kasus setelah 1R).
TRAIL_R_LADDER = []  # Trail sepenuhnya struktural M15; tidak ada profit-lock ladder.

# Trailing struktural M15 — INI inti dari Trail (bukan ladder di atas).
# "Ketika harga menyentuh Trail artinya harga tidak kuat mengikuti trend
# sebelumnya" — SL baru = swing M15 terakhir (HL utk BUY / LH utk SELL)
# + buffer supaya tidak kena liquidity sweep biasa.
STRUCT_TRAIL_LB       = 3      # lookback swing_pts saat trailing
STRUCT_TRAIL_BUF_PCT  = 0.0025 # buffer 0.25% di luar swing agar tidak kena LS biasa
STRUCT_TRAIL_LOOKBACK = 60     # candle M15 untuk cari swing trailing

# Adaptive trailing engine V4
TRAIL_ENGINE_VERSION       = "8.0-predictive-reversal"
TRAIL_ARM_R                = 0.80
TRAIL_PROTECT_R            = 1.00
TRAIL_MATURE_R             = 1.50
TRAIL_ATR_MULT             = 2.40
TRAIL_ATR_TIGHT_MULT       = 1.65
TRAIL_MIN_GAP_ATR          = 0.28
TRAIL_BREAK_EVEN_OFFSET_R  = 0.02
TRAIL_WEAKNESS_MODERATE    = 3
TRAIL_WEAKNESS_STRONG      = 6
TRAIL_LIQUIDITY_NEAR_R     = 0.75
TRAIL_VOLUME_EXPANSION     = 1.30
TRAIL_VOLUME_WEAK          = 0.75
TRAIL_MIN_IMPROVEMENT_ATR  = 0.10

# Fibonacci extension TP (level 1.272 dan 1.618 dari impulse leg)
FIB_EXT_1 = 0.272   # 127.2%
FIB_EXT_2 = 0.618   # 161.8%

# ── Inducement & konfluensi ──────────────────────────────────────────────
INDUCEMENT_LOOKBACK  = 40     # candle M15 untuk cari liquidity minor sebelum POI
INDUCEMENT_MINOR_LB  = 2      # lookback swing_pts untuk swing "minor" (inducement)
CONFLUENCE_BONUS      = 2     # bonus skor kalau OB/FVG M15 overlap dengan zona H1
POI_REACTION_LOOKBACK = 16     # candle M15 untuk menunggu reaksi setelah POI HTF
CONFIRMATION_LOOKBACK = 4      # candle M15 yang dipakai untuk displacement terbaru
MIN_DISPLACEMENT_ATR  = 0.25   # body minimum agar candle bukan noise

# Entry-location / RSI timing — ditambahkan setelah audit CAPUSDT.
# Tujuannya bukan membuat bot anti-trade, tetapi mencegah BUY/SELL di lokasi
# yang sudah terlalu dekat sisi salah dari range saat momentum M15 masih
# bergerak melawan entry. RSI dipakai sebagai timing/context, bukan sinyal
# tunggal dan bukan hard overbought/oversold rule.
ENTRY_LOCATION_LOOKBACK = 16     # 4 jam M15 untuk konteks lokasi
ENTRY_CHASE_HIGH = 0.82         # BUY di atas 82% range / SELL di bawah 18% = chase
ENTRY_PREFERRED_BUY = 0.55      # BUY ideal di bawah ~55% range lokal
ENTRY_PREFERRED_SELL = 0.45     # SELL ideal di atas ~45% range lokal
ENTRY_SWING_NEAR_ATR = 0.55     # dekat swing adverse => kurangi kualitas
RSI_TIMING_SLOPE = 1.5          # perubahan RSI minimum agar dianggap bermakna
RSI_BUY_WEAK = 48.0              # BUY + RSI di bawah ini yang masih jatuh = tunggu
RSI_SELL_WEAK = 52.0              # SELL + RSI di atas ini yang masih naik = tunggu
ENTRY_LOCATION_HARD_FLOOR = 30   # kandidat di bawah ini tidak executable

# Must match main.py _validate_signal_before_entry().  Keeping this in the
# strategy engine prevents full_analyze() from returning a candidate that the
# execution gate will immediately reject as ENTRY_TOO_FAR.
MAIN_ENTRY_MAX_ATR = 1.50


# =============================================================================
# UTILITY — Indikator teknikal
# =============================================================================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n, min_periods=n).mean()
    lo = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    out = pd.Series(50.0, index=s.index, dtype=float)
    valid = g.notna() & lo.notna()
    both_zero = valid & (g <= 1e-12) & (lo <= 1e-12)
    gain_only = valid & (lo <= 1e-12) & (g > 1e-12)
    loss_only = valid & (g <= 1e-12) & (lo > 1e-12)
    normal = valid & (g > 1e-12) & (lo > 1e-12)
    out.loc[gain_only] = 100.0
    out.loc[loss_only] = 0.0
    out.loc[both_zero] = 50.0
    rs = g.loc[normal] / lo.loc[normal]
    out.loc[normal] = 100 - 100 / (1 + rs)
    return out

def atr_fn(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _closed_candles(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Keep only candles whose interval has fully closed.

    Binance REST and websocket kline feeds include the currently forming
    candle.  Using it for structure/EMA decisions makes a signal repaint
    during the same scan and was especially harmful on the fast altcoins in
    the supplied trade sample.  DataFrames with a non-datetime index are
    left untouched for compatibility with offline callers/tests.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    idx = out.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    boundary = now.floor(f"{interval_minutes}min")
    # Historical/offline datasets may be entirely older than the current
    # candle. In that case every row is already closed and must be preserved.
    if idx[-1] < boundary:
        return out
    return out.loc[idx < boundary].copy()


def build_df(df: pd.DataFrame, interval_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Tambahkan EMA, RSI, ATR, volume SMA ke DataFrame OHLCV."""
    if df is None or len(df) < 60:
        return None
    df = df.copy()
    if interval_minutes is not None:
        df = _closed_candles(df, interval_minutes)
    if len(df) < 60:
        return None
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200) if len(df) >= 200 else ema(df["close"], 50)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr_fn(df)
    df["vol_sma"] = df["volume"].rolling(20).mean()
    required = ["ema9", "ema21", "ema50", "ema200", "atr", "vol_sma"]
    df = df.dropna(subset=required)
    df["rsi"] = df["rsi"].fillna(50.0).clip(0.0, 100.0)
    return df

def swing_pts(df: pd.DataFrame, lb: int = 5):
    """Swing high & low — dipakai main.py untuk trailing."""
    sh, sl = [], []
    high = df["high"].values
    low = df["low"].values
    n = len(high)
    for i in range(lb, n - lb):
        window_h = high[max(0, i - lb): i + lb + 1]
        window_l = low[max(0, i - lb): i + lb + 1]
        if high[i] == window_h.max():
            sh.append(i)
        if low[i] == window_l.min():
            sl.append(i)
    return sh, sl

def _market_structure(df: pd.DataFrame, sh: list, sl: list) -> str:
    """HH+HL = bullish · LH+LL = bearish · else ranging."""
    if len(sh) < 2 or len(sl) < 2:
        return "ranging"
    hh = df["high"].iloc[sh[-1]] > df["high"].iloc[sh[-2]]
    hl = df["low"].iloc[sl[-1]] > df["low"].iloc[sl[-2]]
    lh = df["high"].iloc[sh[-1]] < df["high"].iloc[sh[-2]]
    ll = df["low"].iloc[sl[-1]] < df["low"].iloc[sl[-2]]
    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"

mkt_struct = _market_structure  # alias

def _macro_bias(df_btc_h1: Optional[pd.DataFrame]) -> str:
    """
    Bias arah MARKET SECARA KESELURUHAN (BTC H1 sebagai proxy), bukan
    struktur per-koin. Dipakai untuk meredam sinyal BUY di altcoin saat
    seluruh market lagi risk-off (dan sebaliknya) — analisa 7 Agu 2026:
    4/4 sinyal BUY kena SL, total market cap crypto turun 0.5% & BTC di
    bawah EMA50/EMA200 hari itu; satu-satunya SELL malah menang besar.
    Struktur per-koin (OB/FVG M15) bisa kelihatan valid sendiri, tapi
    kalah lawan gravitasi market kalau macro-nya jelas berlawanan.

    Return "bullish" / "bearish" / "ranging" / "unknown" (kalau data BTC
    tidak dikasih atau tidak cukup — caller HARUS treat "unknown" sebagai
    no-op, jangan menghukum arah manapun).
    """
    if df_btc_h1 is None or len(df_btc_h1) < 60:
        return "unknown"
    try:
        btc = build_df(df_btc_h1, interval_minutes=60)
        if btc is None or len(btc) < 60:
            return "unknown"
        LB = btc.iloc[-1]
        shb, slb = swing_pts(btc, lb=5)
        struct_btc = _market_structure(btc, shb, slb)
        ema_bull = LB["ema9"] > LB["ema21"] > LB["ema50"]
        ema_bear = LB["ema9"] < LB["ema21"] < LB["ema50"]
        if struct_btc == "bullish" or ema_bull:
            return "bullish"
        if struct_btc == "bearish" or ema_bear:
            return "bearish"
        return "ranging"
    except Exception:
        return "unknown"

# Penalti/bonus macro — sengaja MODERAT (bukan veto keras seperti d1_bias
# per-koin di atas), karena ini cuma satu lapis konteks tambahan. Tujuannya
# NGE-REM sinyal yang jelas melawan arus market, BUKAN mematikannya —
# kuantitas sinyal harus tetap terjaga (lihat instruksi awal strategy ini).
MACRO_ALIGN_BONUS   = 8      # searah macro → bonus kecil
MACRO_AGAINST_MULT  = 0.72   # berlawanan macro → skor sisi itu dikali ini (bukan 0)


# =============================================================================
# SMC / ICT DETECTORS (berdasarkan transkrip)
# =============================================================================

def is_zone_fresh(df: pd.DataFrame, top: float, bot: float,
                  formed_idx: int, end_idx: Optional[int] = None,
                  direction: Optional[str] = None) -> bool:
    """True jika zona masih aktif, bukan sekadar belum pernah disentuh.

    Untuk zona berarah, retest tidak otomatis membatalkan zona. Invalidasi
    terjadi bila candle close menembus sisi luar zona. Tanpa ``direction``
    fungsi mempertahankan perilaku konservatif lama untuk caller generik.
    """
    if formed_idx is None or formed_idx + 2 >= len(df):
        return True
    start = formed_idx + 2
    end = end_idx if end_idx is not None else len(df) - 1
    if start >= end:
        return True
    sub = df.iloc[start:end]
    if direction == "bull":
        return not bool((sub["close"] < bot).any())
    if direction == "bear":
        return not bool((sub["close"] > top).any())
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)

def fib_position(price: float, swing_low: float, swing_high: float) -> float:
    """Posisi harga dalam range 0–1, 0 = swing_low, 1 = swing_high."""
    rng = swing_high - swing_low
    if rng <= 0:
        return 0.5
    return max(0.0, min(1.0, (price - swing_low) / rng))

def is_in_ote(price: float, swing_low: float, swing_high: float,
              direction: str) -> bool:
    """
    OTE (Optimal Trade Entry) — zona 61.8%–78.6% retracement.
    Bull: fib_position antara 0.214 dan 0.382
    Bear: fib_position antara 0.618 dan 0.786
    """
    if swing_high <= swing_low:
        return False
    r = fib_position(price, swing_low, swing_high)
    if direction == "bull":
        return 0.214 <= r <= 0.382
    else:
        return 0.618 <= r <= 0.786

def detect_fvg(df: pd.DataFrame, direction: str, lb: int = 50) -> list:
    """
    Fair Value Gap (imbalance) — 3‑candle pattern.
    Hanya return zona yang FRESH.
    """
    sub = df.iloc[-lb:]
    base = len(df) - len(sub)
    out = []
    for i in range(len(sub) - 2):
        c0, c2 = sub.iloc[i], sub.iloc[i + 2]
        gap = None
        if direction == "bull" and c2["low"] > c0["high"]:
            gap = {"top": float(c2["low"]), "bot": float(c0["high"])}
        elif direction == "bear" and c2["high"] < c0["low"]:
            gap = {"top": float(c0["low"]), "bot": float(c2["high"])}
        if gap:
            gap["mid"] = (gap["top"] + gap["bot"]) / 2
            gap["idx"] = base + i + 2
            gap["is_fresh"] = is_zone_fresh(
                df, gap["top"], gap["bot"], gap["idx"], direction=direction
            )
            out.append(gap)
    fresh = [f for f in out if f["is_fresh"]]
    return fresh[-3:] if fresh else []

def detect_order_block(df: pd.DataFrame, direction: str, lb: int = 60,
                       sh: Optional[list] = None, sl: Optional[list] = None) -> list:
    """
    Order Block berkualitas (transkrip 1, 14, 26).
    Filter:
      - zona fresh
      - Fibonacci diskon (bull ≤ 0.618) / premium (bear ≥ 0.382)
      - kualitas: impulse kuat, FVG setelahnya, BOS, recency.
    Return list max 3, urut kualitas tertinggi.
    """
    is_demand = direction == "bull"
    sub = df.iloc[-lb:]
    base = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean() or 1e-8

    # Fibonacci context
    fib_sh = float(df["high"].iloc[sh[-1]]) if (sh and len(sh) > 0) else None
    fib_sl = float(df["low"].iloc[sl[-1]]) if (sl and len(sl) > 0) else None

    # BOS global
    has_bos_global = False
    if sh and sl:
        if is_demand and len(sh) >= 2:
            has_bos_global = float(df["high"].iloc[-1]) > float(df["high"].iloc[sh[-2]])
        elif not is_demand and len(sl) >= 2:
            has_bos_global = float(df["low"].iloc[-1]) < float(df["low"].iloc[sl[-2]])

    zones = []
    for i in range(1, len(sub) - 3):
        c, nx = sub.iloc[i], sub.iloc[i + 1]
        # Pola trigger + impulse
        if is_demand:
            if not (c["close"] < c["open"] and nx["close"] > nx["open"]):
                continue
        else:
            if not (c["close"] > c["open"] and nx["close"] < nx["open"]):
                continue

        impulse_body = abs(nx["close"] - nx["open"])
        if impulse_body < avg_body * 1.2:
            continue

        ob_top = float(max(c["open"], c["close"]))
        ob_bot = float(min(c["open"], c["close"]))
        df_idx = base + i

        if not is_zone_fresh(
            df, ob_top, ob_bot, df_idx, direction=direction
        ):
            continue

        q = 0
        if impulse_body >= avg_body * 1.5:
            q += 1
        if impulse_body >= avg_body * 2.5:
            q += 1

        # FVG setelah OB
        if i + 2 < len(sub):
            c2 = sub.iloc[i + 2]
            if is_demand and c2["low"] > c["high"]:
                q += 1
            elif not is_demand and c2["high"] < c["low"]:
                q += 1

        if has_bos_global:
            q += 1

        # Fibonacci diskon/premium (kunci dari transkrip 1)
        if fib_sh is not None and fib_sl is not None:
            ob_mid = (ob_top + ob_bot) / 2
            fib_r = fib_position(ob_mid, fib_sl, fib_sh)
            if is_demand and fib_r <= 0.618:
                q += 1
            elif not is_demand and fib_r >= 0.382:
                q += 1

        if df_idx >= len(df) - 20:
            q += 1

        if q >= 2:
            zones.append({
                "top": ob_top,
                "bot": ob_bot,
                "mid": (ob_top + ob_bot) / 2,
                "idx": df_idx,
                "quality": q,
                "has_bos": has_bos_global,
            })

    zones.sort(key=lambda z: (-z["quality"], -z["idx"]))
    return zones[:3]

def detect_choch(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """Change of Character (transkrip 17, 19, 20)."""
    result = {"bullish_choch": False, "bearish_choch": False}
    if len(sh) < 2 or len(sl) < 2:
        return result
    close = float(df["close"].iloc[-1])
    prev_high = float(df["high"].iloc[sh[-2]])
    last_high = float(df["high"].iloc[sh[-1]])
    prev_low = float(df["low"].iloc[sl[-2]])
    last_low = float(df["low"].iloc[sl[-1]])
    struct = _market_structure(df, sh, sl)

    # A ChoCH is a break of the *opposite* protected swing.  The old
    # implementation compared a bullish close with prev_low (and a bearish
    # close with prev_low), which is true on most ordinary candles and added
    # +20 points to both random pullbacks and continuations.
    if struct == "bearish" and close > last_high:
        result["bullish_choch"] = True
    if struct == "bullish" and close < last_low:
        result["bearish_choch"] = True

    # Raw reversal: require a close beyond the swing, never only a wick.
    if last_high > prev_high and last_low > prev_low and close > last_high:
        result["bullish_choch"] = True
    if last_high < prev_high and last_low < prev_low and close < last_low:
        result["bearish_choch"] = True
    return result

def detect_bos(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """Break of Structure (transkrip 17)."""
    result = {"bullish_bos": False, "bearish_bos": False}
    if len(sh) < 2 or len(sl) < 2:
        return result

    # Use the latest confirmed swing and a candle close.  Looking at the
    # previous swing plus the current wick made old structure look broken
    # long after the actual break had happened.
    close = float(df["close"].iloc[-1])
    last_high = float(df["high"].iloc[sh[-1]])
    last_low = float(df["low"].iloc[sl[-1]])
    prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else close
    result["bullish_bos"] = close > last_high and prev_close <= last_high
    result["bearish_bos"] = close < last_low and prev_close >= last_low
    return result

def detect_cisd(df: pd.DataFrame, lb: int = 8) -> dict:
    """Change In State of Delivery (transkrip 29)."""
    result = {"bullish_cisd": False, "bearish_cisd": False}
    if len(df) < lb + 1:
        return result
    sub = df.iloc[-lb:]
    opens = sub["open"].values
    closes = sub["close"].values
    n = len(closes)
    if n < 4:
        return result
    last_bull = closes[-1] > opens[-1]
    last_bear = closes[-1] < opens[-1]

    if last_bull:
        bear_run = 0
        for j in range(n - 2, -1, -1):
            if closes[j] < opens[j]:
                bear_run += 1
            else:
                break
        if bear_run >= 3:
            first_idx = n - 1 - bear_run
            if first_idx >= 0:
                bear_mid = (opens[first_idx] + closes[first_idx]) / 2
                if closes[-1] > bear_mid:
                    result["bullish_cisd"] = True
    elif last_bear:
        bull_run = 0
        for j in range(n - 2, -1, -1):
            if closes[j] > opens[j]:
                bull_run += 1
            else:
                break
        if bull_run >= 3:
            first_idx = n - 1 - bull_run
            if first_idx >= 0:
                bull_mid = (opens[first_idx] + closes[first_idx]) / 2
                if closes[-1] < bull_mid:
                    result["bearish_cisd"] = True
    return result

def detect_liquidity_sweep(df: pd.DataFrame, sh: list, sl: list,
                           direction: str) -> dict:
    """
    Liquidity Sweep (transkrip 4, 9, 15, 18).
    Valid: wick menembus swing, tapi close kembali di atas/bawah level.
    """
    result = {"type": "none", "level": None, "strength": 0}
    if direction == "bull" and sl:
        level = float(df["low"].iloc[sl[-1]])
        last_low = float(df["low"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_low < level and last_close > level:
            depth = (level - last_low) / max(level, 1e-10)
            result = {
                "type": "sweep",
                "level": level,
                "strength": min(3, int(depth / 0.002) + 1),
            }
    elif direction == "bear" and sh:
        level = float(df["high"].iloc[sh[-1]])
        last_high = float(df["high"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_high > level and last_close < level:
            depth = (last_high - level) / max(level, 1e-10)
            result = {
                "type": "sweep",
                "level": level,
                "strength": min(3, int(depth / 0.002) + 1),
            }
    return result

def detect_inducement(df: pd.DataFrame, direction: str,
                      lb: int = INDUCEMENT_LOOKBACK) -> dict:
    """
    Inducement (transkrip "How the Market Traps Traders with Inducement" &
    "trading strategy using inducement [SNIPER ENTRY]").

    Konsep: sebelum harga bereaksi di POI "asli" (OB/FVG yang lebih dalam),
    market sering menyapu dulu liquidity pool MINOR (swing kecil) yang
    memancing early entry ("inducement"). Kalau minor pool itu SUDAH disapu
    (wick tembus, close kembali ke sisi yang benar) sebelum harga mendekati
    POI kita, itu tanda bagus — bukan validasi wajib, cuma BONUS confidence,
    supaya kuantitas sinyal tidak berkurang untuk koin yang tidak
    menunjukkan pola ini secara eksplisit.

    Return: {"found": bool, "level": float|None, "swept": bool}
    """
    out = {"found": False, "level": None, "swept": False}
    if df is None or len(df) < lb + INDUCEMENT_MINOR_LB * 2 + 1:
        return out
    sub = df.iloc[-lb:].reset_index(drop=True)
    sh_m, sl_m = swing_pts(sub, lb=INDUCEMENT_MINOR_LB)
    try:
        if direction == "bull" and sl_m:
            idx = sl_m[-1]
            if idx >= len(sub) - 2:
                return out
            level = float(sub["low"].iloc[idx])
            after = sub.iloc[idx + 1:]
            swept = bool((after["low"] < level).any() and
                         float(after["close"].iloc[-1]) > level)
            out = {"found": True, "level": level, "swept": swept}
        elif direction == "bear" and sh_m:
            idx = sh_m[-1]
            if idx >= len(sub) - 2:
                return out
            level = float(sub["high"].iloc[idx])
            after = sub.iloc[idx + 1:]
            swept = bool((after["high"] > level).any() and
                         float(after["close"].iloc[-1]) < level)
            out = {"found": True, "level": level, "swept": swept}
    except Exception:
        return {"found": False, "level": None, "swept": False}
    return out

def zones_overlap(a_top: float, a_bot: float, b_top: float, b_bot: float) -> bool:
    """True jika dua rentang harga [bot, top] saling overlap (dipakai untuk
    cek konfluensi OB/FVG M15 dengan zona H1 — 'How to Choose the Best Order
    Block When All Zones Look Valid')."""
    lo = max(min(a_bot, a_top), min(b_bot, b_top))
    hi = min(max(a_bot, a_top), max(b_bot, b_top))
    return lo <= hi

def detect_equal_highs_lows(df: pd.DataFrame, kind: str = "high",
                            lb: int = 80, tol: float = 0.003) -> list:
    """Equal Highs/Lows (transkrip 15, 24)."""
    sub = df.iloc[-lb:]
    vals = sub["high"] if kind == "high" else sub["low"]
    clusters, visited = [], set()
    for i in range(len(vals)):
        if i in visited:
            continue
        group = [float(vals.iloc[i])]
        for j in range(i + 1, len(vals)):
            if abs(vals.iloc[i] - vals.iloc[j]) / max(abs(float(vals.iloc[i])), 1e-10) < tol:
                group.append(float(vals.iloc[j]))
                visited.add(j)
        if len(group) >= 2:
            clusters.append(sum(group) / len(group))
    return sorted(clusters)

def detect_rsi_divergence(df: pd.DataFrame, direction: str, lb: int = 30) -> dict:
    """RSI Divergence (transkrip 19)."""
    result = {"bull_div": False, "bear_div": False, "strong": False}
    if len(df) < lb + 1 or "rsi" not in df.columns:
        return result
    sub = df.iloc[-lb:]
    price = sub["close"].values
    rsi_v = sub["rsi"].values
    n = len(price)
    lb3 = 3

    lows = [i for i in range(lb3, n - lb3)
            if price[i] == min(price[max(0, i - lb3): i + lb3 + 1])]
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if price[i2] < price[i1] and rsi_v[i2] > rsi_v[i1]:
            result["bull_div"] = True
            if rsi_v[i2] < 35:
                result["strong"] = True

    highs = [i for i in range(lb3, n - lb3)
             if price[i] == max(price[max(0, i - lb3): i + lb3 + 1])]
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if price[i2] > price[i1] and rsi_v[i2] < rsi_v[i1]:
            result["bear_div"] = True
            if rsi_v[i2] > 65:
                result["strong"] = True

    if direction == "bull" and not result["bull_div"]:
        return {"bull_div": False, "bear_div": False, "strong": False}
    if direction == "bear" and not result["bear_div"]:
        return {"bull_div": False, "bear_div": False, "strong": False}
    return result

def detect_failed_retest(df: pd.DataFrame, sh: list, sl: list,
                         atr: float) -> dict:
    """Failed retest (untuk kompatibilitas)."""
    result = {"failed_retest_sell": False, "failed_retest_buy": False}
    if len(df) < 3 or not sh or not sl:
        return result
    L = df.iloc[-1]
    P = df.iloc[-2]
    if len(sh) >= 2:
        res = float(df["high"].iloc[sh[-2]])
        if P["high"] >= res - atr * 0.5 and L["close"] < res - atr * 0.3 and L["close"] < L["open"]:
            result["failed_retest_sell"] = True
    if len(sl) >= 2:
        sup = float(df["low"].iloc[sl[-2]])
        if P["low"] <= sup + atr * 0.5 and L["close"] > sup + atr * 0.3 and L["close"] > L["open"]:
            result["failed_retest_buy"] = True
    return result


def detect_entry_confirmation(df: pd.DataFrame, direction: str, atr: float,
                              lb: int = CONFIRMATION_LOOKBACK) -> dict:
    """Konfirmasi displacement yang baru, bukan sekadar lokasi entry.

    OTE, RSI divergence, discount/premium, dan sweep menjelaskan lokasi atau
    kemungkinan reaksi. Mereka tidak membuktikan bahwa buyer/seller sudah
    mengambil alih. Konfirmasi ini membutuhkan candle close searah yang
    menembus range pendek sebelumnya dengan body yang cukup besar.
    """
    out = {
        "confirmed": False,
        "direction": direction,
        "kind": "none",
        "idx": None,
        "body_atr": 0.0,
    }
    if df is None or len(df) < max(lb + 2, 8):
        return out

    last = df.iloc[-1]
    prior = df.iloc[-(lb + 1):-1]
    body = abs(float(last["close"]) - float(last["open"]))
    local_atr = max(float(atr), 1e-10)
    body_atr = body / local_atr
    out["body_atr"] = round(body_atr, 3)

    recent_bodies = (df["close"] - df["open"]).abs().iloc[-(lb + 5):-1]
    median_body = float(recent_bodies.median()) if not recent_bodies.empty else 0.0
    min_body = max(local_atr * MIN_DISPLACEMENT_ATR, median_body * 0.8)

    if direction == "bull":
        confirmed = (
            float(last["close"]) > float(last["open"])
            and float(last["close"]) > float(prior["high"].max())
            and body >= min_body
        )
    else:
        confirmed = (
            float(last["close"]) < float(last["open"])
            and float(last["close"]) < float(prior["low"].min())
            and body >= min_body
        )

    if confirmed:
        out.update({
            "confirmed": True,
            "kind": "displacement_close",
            "idx": len(df) - 1,
        })
    return out


def _collect_htf_poi_zones(h1: pd.DataFrame, direction: str,
                           score_ctx: dict) -> list:
    """Kumpulkan POI H1 searah untuk validasi reaksi sebelum entry."""
    zones = []
    try:
        for z in detect_order_block(
            h1, direction, lb=80,
            sh=score_ctx.get("sh1", []),
            sl=score_ctx.get("sl1", []),
        ):
            zones.append({
                "top": float(z["top"]),
                "bot": float(z["bot"]),
                "kind": "ob_h1",
            })
        for f in detect_fvg(h1, direction, lb=60):
            zones.append({
                "top": float(f["top"]),
                "bot": float(f["bot"]),
                "kind": "fvg_h1",
            })
    except Exception:
        return []
    return zones


def _recent_poi_reaction(m15: pd.DataFrame, zones: list, direction: str,
                         atr: float,
                         lookback: int = POI_REACTION_LOOKBACK) -> bool:
    """Pastikan POI HTF sudah disentuh dan mendapat rejection baru.

    Retest saja tidak cukup. Setelah harga masuk POI, harus ada close kembali
    keluar dari sisi reaksi zona. Ini mencegah pending order dipasang hanya
    karena OB/FVG lama masih terlihat fresh.
    """
    if m15 is None or m15.empty or not zones:
        return False
    start = max(0, len(m15) - lookback)
    sub = m15.iloc[start:]
    body_floor = max(float(atr) * MIN_DISPLACEMENT_ATR, 1e-10)

    for zone in zones:
        top, bot = float(zone["top"]), float(zone["bot"])
        for i in range(len(sub)):
            candle = sub.iloc[i]
            touched = float(candle["low"]) <= top and float(candle["high"]) >= bot
            if not touched:
                continue
            for j in range(i, min(i + 5, len(sub))):
                reaction = sub.iloc[j]
                body = abs(float(reaction["close"]) - float(reaction["open"]))
                if direction == "bull":
                    ok = (
                        float(reaction["close"]) > top
                        and float(reaction["close"]) > float(reaction["open"])
                        and body >= body_floor
                    )
                else:
                    ok = (
                        float(reaction["close"]) < bot
                        and float(reaction["close"]) < float(reaction["open"])
                        and body >= body_floor
                    )
                if ok:
                    return True
    return False


# =============================================================================
# SCORING — Confidence global tanpa bias sesi
# =============================================================================

def score_direction(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                    df_d1: Optional[pd.DataFrame] = None,
                    df_btc_h1: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """
    Tentukan arah dan skor confidence global (tanpa bias sesi).
    Faktor scoring (max ~165):
      - D1 bias (struktur/EMA)         +20
      - H1 structure                   +20
      - H1 ChoCH                       +10
      - H1 BOS                         +5
      - H1 EMA stack                   +10
      - M15 ChoCH                      +20
      - M15 BOS                        +10
      - M15 CISD                       +15
      - M15 Liquidity Sweep            +15
      - M15 OTE                        +10
      - M15 Fibonacci diskon/premium   +10
      - M15 RSI divergence             +10 (+5 if strong)
      - M15 failed retest              +10
    Penalti: jika setup M15 berlawanan dengan bias D1 → kurangi 50%.
    """
    h1 = build_df(df_h1, interval_minutes=60)
    m15 = build_df(df_m15, interval_minutes=15)
    if h1 is None or m15 is None:
        return None

    L1 = h1.iloc[-1]
    L15 = m15.iloc[-1]
    atr = max(float(L15["atr"]),
              float(L1["atr"]) / 4,
              float(L15["close"]) * 0.003)

    sh1, sl1 = swing_pts(h1, lb=5)
    sh15, sl15 = swing_pts(m15, lb=5)
    struct_h1 = _market_structure(h1, sh1, sl1)

    # ── D1 bias ──────────────────────────────────────────────────
    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            d1 = build_df(df_d1, interval_minutes=1440)
        else:
            d1 = build_df(
                df_h1.resample("1D").agg(
                    {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
                ).dropna()
            )
        if d1 is not None and len(d1) >= 10:
            LD = d1.iloc[-1]
            shd, sld = swing_pts(d1, lb=3)
            sd1 = _market_structure(d1, shd, sld)
            bull_d1 = sd1 == "bullish" or (LD["ema9"] > LD["ema21"] > LD["ema50"])
            bear_d1 = sd1 == "bearish" or (LD["ema9"] < LD["ema21"] < LD["ema50"])
            if bull_d1:
                d1_bias = "bullish"
            elif bear_d1:
                d1_bias = "bearish"
    except Exception:
        pass

    # ── H1 indikator ─────────────────────────────────────────────
    ema_h1_bull = L1["ema9"] > L1["ema21"] > L1["ema50"]
    ema_h1_bear = L1["ema9"] < L1["ema21"] < L1["ema50"]
    choch_h1 = detect_choch(h1, sh1, sl1)
    bos_h1 = detect_bos(h1, sh1, sl1)

    # ── M15 indikator ─────────────────────────────────────────────
    choch_m15 = detect_choch(m15, sh15, sl15)
    bos_m15 = detect_bos(m15, sh15, sl15)
    cisd_m15 = detect_cisd(m15, lb=8)
    liq_bull = detect_liquidity_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_sweep(m15, sh15, sl15, "bear")
    fr_m15 = detect_failed_retest(m15, sh15, sl15, atr)

    # ── Fibonacci M15 context ─────────────────────────────────────
    fib_sh = float(m15["high"].iloc[sh15[-1]]) if sh15 else None
    fib_sl = float(m15["low"].iloc[sl15[-1]]) if sl15 else None
    fib_r = fib_position(float(L15["close"]), fib_sl or 0, fib_sh or 1) \
            if (fib_sh and fib_sl) else 0.5

    ote_bull = is_in_ote(float(L15["close"]), fib_sl or 0, fib_sh or 1, "bull") if (fib_sh and fib_sl) else False
    ote_bear = is_in_ote(float(L15["close"]), fib_sl or 0, fib_sh or 1, "bear") if (fib_sh and fib_sl) else False

    in_discount = fib_r < 0.45
    in_premium = fib_r > 0.55

    rdiv_bull = detect_rsi_divergence(m15, "bull", lb=30)
    rdiv_bear = detect_rsi_divergence(m15, "bear", lb=30)

    # ── Inducement dan konfirmasi entry ────────────────────────────
    induce_bull = detect_inducement(m15, "bull")
    induce_bear = detect_inducement(m15, "bear")
    confirm_bull = detect_entry_confirmation(m15, "bull", atr)
    confirm_bear = detect_entry_confirmation(m15, "bear", atr)

    # ── Score BULL ───────────────────────────────────────────────
    bull = 0
    if d1_bias == "bullish":                    bull += 20
    if struct_h1 == "bullish":                  bull += 20
    elif struct_h1 == "ranging":                bull += 5
    if choch_h1["bullish_choch"]:               bull += 10
    if bos_h1["bullish_bos"]:                   bull += 5
    if ema_h1_bull:                             bull += 10
    if choch_m15["bullish_choch"]:              bull += 20
    if bos_m15["bullish_bos"]:                  bull += 10
    if cisd_m15["bullish_cisd"]:                bull += 15
    if liq_bull["type"] == "sweep":             bull += 15
    if ote_bull:                                bull += 10
    if in_discount:                             bull += 10
    if rdiv_bull.get("bull_div"):
        bull += 10 + (5 if rdiv_bull.get("strong") else 0)
    if fr_m15["failed_retest_buy"]:             bull += 10
    if induce_bull.get("swept"):                bull += 8

    # ── Score BEAR ───────────────────────────────────────────────
    bear = 0
    if d1_bias == "bearish":                    bear += 20
    if struct_h1 == "bearish":                  bear += 20
    elif struct_h1 == "ranging":                bear += 5
    if choch_h1["bearish_choch"]:               bear += 10
    if bos_h1["bearish_bos"]:                   bear += 5
    if ema_h1_bear:                             bear += 10
    if choch_m15["bearish_choch"]:              bear += 20
    if bos_m15["bearish_bos"]:                  bear += 10
    if cisd_m15["bearish_cisd"]:                bear += 15
    if liq_bear["type"] == "sweep":             bear += 15
    if ote_bear:                                bear += 10
    if in_premium:                              bear += 10
    if rdiv_bear.get("bear_div"):
        bear += 10 + (5 if rdiv_bear.get("strong") else 0)
    if fr_m15["failed_retest_sell"]:            bear += 10
    if induce_bear.get("swept"):                bear += 8

    # ── Macro bias (BTC H1) — konteks tambahan, bukan pengganti HTF koin ──
    macro_bias = _macro_bias(df_btc_h1)
    if macro_bias == "bullish":
        bull += MACRO_ALIGN_BONUS
        bear = int(bear * MACRO_AGAINST_MULT)
    elif macro_bias == "bearish":
        bear += MACRO_ALIGN_BONUS
        bull = int(bull * MACRO_AGAINST_MULT)
    # macro_bias == "ranging"/"unknown" → tidak ada perubahan sama sekali

    # ── HTF direction gate ─────────────────────────────────────────
    # D1 dan H1 bukan lagi bonus independen yang bisa dikalahkan oleh RSI,
    # OTE, atau sweep. Jika keduanya berlawanan, market sedang tidak memiliki
    # bias yang cukup jelas untuk limit order baru; tunggu alignment berikutnya.
    h1_bias = "bullish" if struct_h1 == "bullish" else (
        "bearish" if struct_h1 == "bearish" else "neutral"
    )
    if d1_bias in ("bullish", "bearish") and h1_bias in ("bullish", "bearish"):
        htf_bias = d1_bias if d1_bias == h1_bias else "conflict"
    elif d1_bias in ("bullish", "bearish"):
        htf_bias = d1_bias
    elif h1_bias in ("bullish", "bearish"):
        htf_bias = h1_bias
    else:
        htf_bias = "neutral"

    if htf_bias == "bullish":
        direction = "bull"
    elif htf_bias == "bearish":
        direction = "bear"
    else:
        # Konflik/ranging bukan auto-reject. Pilih sisi dengan confluence
        # terbesar dan biarkan confidence yang menghukum setup yang lemah.
        direction = "bull" if bull >= bear else "bear"
    raw = bull if direction == "bull" else bear
    # ── FIX BUG KRITIS (scan 50 koin selalu "Tidak ada setup valid") ──
    # MAX_SCORE=173 lama = jumlah SEMUA bonus sekaligus (D1+H1+ChoCH+BOS+
    # EMA+M15 ChoCH+BOS+CISD+Liquidity+OTE+Fib+RSIdiv+failed_retest+
    # inducement, semua di arah yang sama). Dites empiris lewat 300
    # skenario pasar acak (termasuk tren paling bersih): skor mentah
    # MAKSIMUM yang pernah tercapai cuma ~88, rata-rata ~40, p99 ~75 —
    # skala 173 bikin confidence yang ditampilkan jauh lebih rendah dari
    # kualitas setup sebenarnya (bahkan setup terbaik pun mentok ~50%).
    # Direkalibrasi ke 100 (dekat batas atas realistis) supaya angka
    # confidence yang tampil BERARTI sesuatu — setup kuat bisa mendekati
    # 80-90%, bukan mentok di 50%.
    MAX_SCORE = 100
    confidence = min(int(raw / MAX_SCORE * 100), 99)
    if htf_bias == "conflict":
        confidence = max(0, confidence - 12)
    elif htf_bias == "neutral":
        confidence = max(0, confidence - 5)

    # Direction quality is separate from the absolute score.  A high score
    # assembled from unrelated bonuses is not enough when H1 and M15 disagree.
    edge = abs(bull - bear)
    m15_struct = _market_structure(m15, sh15, sl15)
    selected_choch = choch_m15["bullish_choch"] if direction == "bull" else choch_m15["bearish_choch"]
    selected_bos = bos_m15["bullish_bos"] if direction == "bull" else bos_m15["bearish_bos"]
    selected_cisd = cisd_m15["bullish_cisd"] if direction == "bull" else cisd_m15["bearish_cisd"]
    selected_confirm = (
        confirm_bull if direction == "bull" else confirm_bear
    )
    selected_sweep = (
        liq_bull if direction == "bull" else liq_bear
    ).get("type") == "sweep"
    # Sweep hanya lokasi liquidity, bukan bukti arah. Trigger utama harus
    # berupa structure/displacement yang terjadi pada candle terbaru.
    trigger_count = sum(bool(x) for x in (
        selected_choch, selected_bos, selected_cisd,
        selected_confirm.get("confirmed"),
    ))

    return {
        "direction": direction,
        "confidence": confidence,
        "bull_score": bull,
        "bear_score": bear,
        "direction_edge": edge,
        "m15_struct": m15_struct,
        "trigger_count": trigger_count,
        "htf_bias": htf_bias,
        "h1_bias": h1_bias,
        "entry_confirmation": selected_confirm,
        "entry_confirmation_bull": confirm_bull,
        "entry_confirmation_bear": confirm_bear,
        "selected_sweep": selected_sweep,
        "price": float(L15["close"]),
        "atr": atr,
        "struct_h1": struct_h1,
        "d1_bias": d1_bias,
        "macro_bias": macro_bias,
        "choch_m15": choch_m15,
        "choch_h1": choch_h1,
        "cisd_m15": cisd_m15,
        "bos_m15": bos_m15,
        "bos_h1": bos_h1,
        "m15_rsi": float(L15["rsi"]),
        "m15_rsi_slope": float(L15["rsi"] - m15["rsi"].iloc[-2]) if len(m15) >= 2 else 0.0,
        "m15_relative_volume": _relative_volume(m15, 20) if len(m15) >= 23 else 1.0,
        "m15_momentum_aligned": bool((direction == "bull" and _momentum_context(m15).get("bull")) or (direction == "bear" and _momentum_context(m15).get("bear"))),
        "m15_divergence_against": bool((direction == "bull" and rdiv_bear.get("bear_div")) or (direction == "bear" and rdiv_bull.get("bull_div"))),
        "failed_retest": fr_m15,
        "liquidity_bull": liq_bull,
        "liquidity_bear": liq_bear,
        "inducement_bull": induce_bull,
        "inducement_bear": induce_bear,
        "sh15": sh15, "sl15": sl15,
        "sh1": sh1, "sl1": sl1,
        "fib_r": round(fib_r, 3),
        "ote_bull": ote_bull,
        "ote_bear": ote_bear,
    }


# =============================================================================
# CONFIDENCE CALIBRATION V3.0 — TRADEABILITY
# =============================================================================
CONFIDENCE_MODEL_VERSION = "4.0_calibrated_ready"

def _rr_band(rr: float) -> str:
    rr = float(rr or 0.0)
    if rr < 3.0:
        return "2-3R"
    if rr < 5.0:
        return "3-5R"
    if rr < 7.0:
        return "5-7R"
    if rr < 9.0:
        return "7-9R"
    return "9R+"


def calibrate_confidence_from_history(history, base_score: float, decision: str,
                                      entry_label: str, rr: float, setup_quality: float) -> tuple[int, dict]:
    """Empirical probability calibration hook.

    `history` is optional and must contain closed trades with a numeric PnL or
    a boolean `profitable`. With fewer observations the model falls back to the
    structural score; it never pretends that a tiny sample is a precise probability.
    Bayesian smoothing keeps sparse buckets conservative.
    """
    try:
        rows = [r for r in (history or []) if isinstance(r, dict)]
        if not rows:
            return int(max(0, min(99, round(base_score)))), {
                "calibration": "structural_fallback", "samples": 0,
                "empirical_confidence": None, "calibration_weight": 0.0
            }

        rrband = _rr_band(rr)
        direction = str(decision or "").upper()
        label = str(entry_label or "").lower()
        candidates = []
        for r in rows:
            d = str(r.get("decision", r.get("direction", ""))).upper()
            el = str(r.get("entry_label", "")).lower()
            rb = _rr_band(float(r.get("rr", 0) or 0))
            profitable = r.get("profitable")
            if profitable is None:
                pnl = r.get("pnl", r.get("pnl_usd", r.get("net_pnl")))
                try:
                    profitable = float(pnl) > 0
                except Exception:
                    continue
            # Hierarchical relevance: same direction + label + RR band is strongest.
            match = 0.0
            if d == direction: match += 0.35
            if el == label: match += 0.30
            if rb == rrband: match += 0.25
            if match >= 0.35:
                candidates.append((match, bool(profitable)))

        if not candidates:
            candidates = [(0.25, bool(r.get("profitable"))) for r in rows if "profitable" in r]

        if not candidates:
            return int(max(0, min(99, round(base_score)))), {
                "calibration": "structural_fallback", "samples": 0,
                "empirical_confidence": None, "calibration_weight": 0.0
            }

        # Weighted Beta posterior. Prior is intentionally neutral, not a claim of
        # 60% win probability. More observations are required before it dominates.
        alpha = beta = 2.0
        weighted_n = 0.0
        for w, win in candidates:
            alpha += w if win else 0.0
            beta += 0.0 if win else w
            weighted_n += w
        empirical = 100.0 * alpha / max(alpha + beta, 1e-9)
        weight = min(0.70, weighted_n / 40.0)
        calibrated = (1.0 - weight) * float(base_score) + weight * empirical
        calibrated = max(0, min(99, int(round(calibrated))))
        return calibrated, {
            "calibration": "empirical_bayesian",
            "samples": len(candidates),
            "weighted_samples": round(weighted_n, 2),
            "empirical_confidence": round(empirical, 2),
            "calibration_weight": round(weight, 3),
            "rr_band": rrband,
            "direction": direction,
            "entry_label": label,
        }
    except Exception:
        return int(max(0, min(99, round(base_score)))), {
            "calibration": "structural_fallback_error", "samples": 0,
            "empirical_confidence": None, "calibration_weight": 0.0
        }


def _confidence_band(conf: int) -> str:
    if conf >= 85:
        return "ELITE"
    if conf >= 75:
        return "VERY_STRONG"
    if conf >= 65:
        return "STRONG"
    if conf >= 55:
        return "NORMAL"
    if conf >= 45:
        return "WEAK"
    return "VERY_WEAK"


def _calibrate_confidence(base_confidence: int, score_ctx: dict, loc: dict,
                          candidate: dict, rr: float, htf_poi: list,
                          poi_reacted: bool, tp_diag: Optional[dict] = None) -> tuple[int, int, dict]:
    """Confidence V3.0 = tradeability, not raw confluence count.

    Direction establishes the thesis, but entry location and current-move health
    have larger influence on the final number. RR is payoff, not probability.
    """
    direction = score_ctx.get("direction", "bull")
    htf_bias = score_ctx.get("htf_bias", "neutral")
    m15_struct = score_ctx.get("m15_struct", "ranging")
    h1_bias = score_ctx.get("h1_bias", "neutral")
    trigger_count = int(score_ctx.get("trigger_count", 0) or 0)
    selected_confirm = score_ctx.get("entry_confirmation", {}) or {}
    selected_sweep = bool(score_ctx.get("selected_sweep"))
    loc_score = int(loc.get("location_score", 50) or 50)
    candidate_score = float(candidate.get("score", 0) or 0)
    entry_label = str(candidate.get("label", ""))

    # 1) DIRECTION / STRUCTURAL THESIS: 0-25
    direction_q = 0.0
    if h1_bias == ("bullish" if direction == "bull" else "bearish"):
        direction_q += 10
    elif h1_bias == "neutral":
        direction_q += 6
    if htf_bias == ("bullish" if direction == "bull" else "bearish"):
        direction_q += 9
    elif htf_bias == "neutral":
        direction_q += 5
    elif htf_bias == "conflict":
        direction_q -= 4
    if m15_struct == ("bullish" if direction == "bull" else "bearish"):
        direction_q += 6
    elif m15_struct == "ranging":
        direction_q += 3
    direction_q = max(0.0, min(25.0, direction_q))

    # 2) ENTRY QUALITY: 0-30 -- deliberately larger than raw direction.
    entry_q = 0.0
    entry_q += max(0.0, min(18.0, (loc_score - 30) * 0.38))
    if htf_poi:
        entry_q += 3.0
    if poi_reacted:
        entry_q += 4.0
    if entry_label in ("ob", "ob_retest"):
        entry_q += 2.0
    elif entry_label in ("fvg", "fvg_retest"):
        entry_q += 1.5
    # Chase / bad location penalties.
    if loc_score < 45:
        entry_q -= 4
    if loc.get("location_state") in {"chase", "bad_location", "premium_chase", "discount_chase"}:
        entry_q -= 4
    entry_q = max(0.0, min(30.0, entry_q))

    # 3) CURRENT MOVE HEALTH: 0-25
    move_q = 12.0
    rsi_val = float(loc.get("rsi", score_ctx.get("m15_rsi", 50.0)) or 50.0)
    rv = float(score_ctx.get("m15_relative_volume", 1.0) or 1.0)
    rsi_timing = str(loc.get("rsi_timing", "unknown"))
    rsi_slope = float(loc.get("rsi_slope", score_ctx.get("m15_rsi_slope", 0.0)) or 0.0)
    momentum_aligned = bool(score_ctx.get("m15_momentum_aligned", False))
    divergence = bool(score_ctx.get("m15_divergence_against", False))

    if momentum_aligned:
        move_q += 6
    if rv >= 1.30 and momentum_aligned:
        move_q += 3
    elif rv < 0.70 and rr >= 2.0:
        move_q -= 2
    if divergence:
        move_q -= 5
    if direction == "bull" and rsi_val >= 72:
        move_q -= 4
    elif direction == "bear" and rsi_val <= 28:
        move_q -= 4
    if rsi_timing in {"rising", "falling", "favorable"}:
        move_q += 1
    if direction == "bull" and rsi_slope < -1.5:
        move_q -= 2
    if direction == "bear" and rsi_slope > 1.5:
        move_q -= 2
    move_q = max(0.0, min(25.0, move_q))

    # 4) TRIGGER / CONFIRMATION: 0-10
    trigger_q = min(10.0, 2.0 + min(trigger_count, 3) * 1.5)
    if selected_confirm.get("confirmed"):
        trigger_q += 2.0
    if selected_sweep:
        trigger_q += 0.5
    trigger_q = max(0.0, min(10.0, trigger_q))

    # 5) TARGET / PAYOFF QUALITY: 0-10. RR is not treated as win probability.
    target_q = 4.0
    selected = (tp_diag or {}).get("selected") if isinstance(tp_diag, dict) else None
    if selected:
        reach = float(selected.get("reachability_proxy", 0.0) or 0.0)
        path = float(selected.get("path_clear", 0.0) or 0.0)
        tq = float(selected.get("target_quality", 0.0) or 0.0)
        target_q = min(10.0, 2.0 + 3.0 * reach + 2.0 * path + 3.0 * tq)
    elif rr >= 2.0:
        target_q = 5.0
    target_q = max(0.0, min(10.0, target_q))

    # Research prior: OB was materially healthier than FVG/EQ in the supplied
    # ledger, while SELL was weaker. These are soft penalties only; they are not
    # hard filters and must be re-calibrated once a larger history is available.
    if entry_label in ("fvg", "fvg_retest"):
        entry_q -= 1.5
    elif entry_label in ("eq", "equal", "equal_high", "equal_low"):
        entry_q -= 3.0
    if direction == "bear":
        direction_q -= 2.0
    entry_q = max(0.0, min(30.0, entry_q))
    direction_q = max(0.0, min(25.0, direction_q))

    # Direction should matter, but cannot dominate. This is the key change that
    # addresses high-confidence SLs caused by strong HTF structure + poor timing.
    calibrated = int(round(
        0.25 * (base_confidence)
        + 0.30 * (direction_q + 10 * 0)
        + 0.30 * (entry_q / 30.0 * 100.0)
        + 0.15 * ((move_q + trigger_q + target_q) / 45.0 * 100.0)
    ))

    # Harder contradiction penalties are still soft, so frequency does not collapse.
    if htf_bias == "conflict":
        calibrated -= 7
    if m15_struct not in (("bullish" if direction == "bull" else "bearish"), "ranging"):
        calibrated -= 4
    if loc_score < 40:
        calibrated -= 5
    if divergence and (direction == "bull" and rsi_val >= 70 or direction == "bear" and rsi_val <= 30):
        calibrated -= 3

    calibrated = max(0, min(99, calibrated))
    setup_quality = int(round(0.20 * direction_q + 0.40 * entry_q + 0.25 * move_q + 0.10 * trigger_q + 0.05 * target_q))
    diagnostics = {
        "model_version": CONFIDENCE_MODEL_VERSION,
        "direction_component": int(base_confidence),
        "direction_quality": int(round(direction_q)),
        "entry_quality": int(round(entry_q)),
        "move_health": int(round(move_q)),
        "trigger_quality": int(round(trigger_q)),
        "target_quality": int(round(target_q)),
        "setup_quality": max(0, min(100, setup_quality)),
        "rr_band": _rr_band(rr),
        "entry_type_prior": "OB_BASELINE" if entry_label in ("ob", "ob_retest") else ("FVG_SOFT_PENALTY" if entry_label in ("fvg", "fvg_retest") else "OTHER_SOFT_PENALTY"),
        "rsi": round(rsi_val, 1),
        "relative_volume": round(rv, 3),
        "divergence_against": bool(divergence),
        "confidence_band": _confidence_band(calibrated),
    }
    return calibrated, max(0, min(100, setup_quality)), diagnostics


# =============================================================================
# ENTRY LOCATION + RSI TIMING
# =============================================================================

def _entry_location_metrics(m15: pd.DataFrame, direction: str,
                            entry: float, atr: float) -> dict:
    """Nilai lokasi entry relatif terhadap range M15 + timing RSI.

    Prinsip CAP: arah HTF boleh benar, tetapi entry di bagian atas range saat
    M15 sedang melemah adalah entry yang buruk. Kita tidak memaksa harga
    masuk di titik tertentu; kita hanya memberi ranking lebih tinggi pada
    retracement yang sehat dan memblokir chase yang jelas.
    """
    if m15 is None or len(m15) < 8:
        return {
            "location_score": 50, "location_state": "unknown",
            "range_position": 0.5, "rsi_timing": "unknown",
            "rsi": None, "rsi_slope": 0.0, "hard_block": False,
            "entry_zone_low": None, "entry_zone_high": None,
        }

    n = min(ENTRY_LOCATION_LOOKBACK, len(m15))
    sub = m15.iloc[-n:]
    rh = float(sub["high"].max())
    rl = float(sub["low"].min())
    width = max(rh - rl, max(float(atr), 1e-10))
    pos = (float(entry) - rl) / width
    pos = max(0.0, min(1.0, pos))

    rsi_now = float(m15["rsi"].iloc[-1])
    rsi_1 = float(m15["rsi"].iloc[-2])
    rsi_2 = float(m15["rsi"].iloc[-3])
    slope = rsi_now - rsi_2
    rising = slope >= RSI_TIMING_SLOPE
    falling = slope <= -RSI_TIMING_SLOPE

    # Adverse-side swing: BUY terlalu dekat high, SELL terlalu dekat low.
    recent = m15.iloc[-8:]
    adverse_high = float(recent["high"].max())
    adverse_low = float(recent["low"].min())
    if direction == "bull":
        swing_dist_atr = (adverse_high - float(entry)) / max(float(atr), 1e-10)
    else:
        swing_dist_atr = (float(entry) - adverse_low) / max(float(atr), 1e-10)

    score = 50
    notes = []
    if direction == "bull":
        if pos <= 0.35:
            score += 15; notes.append("deep_discount")
        elif pos <= ENTRY_PREFERRED_BUY:
            score += 10; notes.append("good_pullback")
        elif pos <= 0.70:
            score += 0; notes.append("mid_range")
        elif pos <= ENTRY_CHASE_HIGH:
            score -= 10; notes.append("upper_range")
        else:
            score -= 22; notes.append("chasing_high")

        if swing_dist_atr <= ENTRY_SWING_NEAR_ATR:
            score -= 10; notes.append("near_swing_high")
        if 42 <= rsi_now <= 58 and rising:
            score += 10; notes.append("rsi_recovery")
        elif rsi_now > 68 and falling:
            score -= 14; notes.append("rsi_exhaustion")
        elif rsi_now < RSI_BUY_WEAK and falling and pos > ENTRY_PREFERRED_BUY:
            score -= 18; notes.append("rsi_weak_while_high")
        elif falling:
            score -= 5; notes.append("rsi_falling")
    else:
        if pos >= 0.65:
            score += 15; notes.append("deep_premium")
        elif pos >= ENTRY_PREFERRED_SELL:
            score += 10; notes.append("good_pullback")
        elif pos >= 0.30:
            score += 0; notes.append("mid_range")
        elif pos >= (1.0 - ENTRY_CHASE_HIGH):
            score -= 10; notes.append("lower_range")
        else:
            score -= 22; notes.append("chasing_low")

        if swing_dist_atr <= ENTRY_SWING_NEAR_ATR:
            score -= 10; notes.append("near_swing_low")
        if 42 <= rsi_now <= 58 and falling:
            score += 10; notes.append("rsi_recovery")
        elif rsi_now < 32 and rising:
            score -= 8; notes.append("rsi_bounce_against_sell")
        elif rsi_now > RSI_SELL_WEAK and rising and pos < ENTRY_PREFERRED_SELL:
            score -= 18; notes.append("rsi_weak_while_low")
        elif rising:
            score -= 5; notes.append("rsi_rising")

    score = int(max(0, min(100, score)))
    # Hard block hanya untuk lokasi yang benar-benar mengejar harga.
    # RSI tetap menjadi penalty/confluence, bukan gate keras, supaya setup
    # yang bagus tidak hilang hanya karena momentum belum ideal.
    if direction == "bull":
        hard_block = pos >= ENTRY_CHASE_HIGH
    else:
        hard_block = pos <= (1.0 - ENTRY_CHASE_HIGH)

    # Zona entry berbasis range lokal, bukan titik matematis palsu.
    if direction == "bull":
        zone_low = rl
        zone_high = rl + width * 0.60
    else:
        zone_low = rl + width * 0.40
        zone_high = rh

    if hard_block:
        state = "WAIT_ENTRY"
    elif score >= 70:
        state = "GOOD"
    elif score >= 50:
        state = "ACCEPTABLE"
    else:
        state = "WEAK"

    return {
        "location_score": score,
        "location_state": state,
        "range_position": round(pos, 3),
        "range_low": rl, "range_high": rh,
        "entry_zone_low": zone_low, "entry_zone_high": zone_high,
        "rsi": round(rsi_now, 2),
        "rsi_prev": round(rsi_1, 2),
        "rsi_slope": round(slope, 2),
        "rsi_timing": "rising" if rising else ("falling" if falling else "flat"),
        "swing_dist_atr": round(swing_dist_atr, 3),
        "notes": notes,
        "hard_block": hard_block,
    }


# =============================================================================
# STEP 1 — ENTRY CANDIDATES
# =============================================================================

def _collect_entry_candidates(m15: pd.DataFrame, h1: pd.DataFrame,
                              direction: str, current_price: float,
                              atr: float, score_ctx: dict) -> list:
    """
    Kumpulkan kandidat entry berdasarkan SMC/ICT.
    Prioritas (score tertinggi):
      1. OB + Liquidity Sweep (12–15)
      2. OB + ChoCH (9–12)
      3. OB + OTE (8–10)
      4. OB saja (5–8)
      5. FVG + CISD + LiqSweep (6–9)
      6. FVG saja (3–5)
      7. Equal Highs/Lows (2–4)
      8. Market entry fallback (1)
    """
    up = direction == "bull"
    cands = []

    liq = score_ctx.get("liquidity_bull" if up else "liquidity_bear", {})
    choch = score_ctx.get("choch_m15", {})
    cisd = score_ctx.get("cisd_m15", {})

    choch_ok = choch.get("bullish_choch") if up else choch.get("bearish_choch")
    cisd_ok = cisd.get("bullish_cisd") if up else cisd.get("bearish_cisd")
    liq_ok = liq.get("type") == "sweep"

    fib_sh = float(m15["high"].iloc[score_ctx["sh15"][-1]]) if score_ctx.get("sh15") else None
    fib_sl = float(m15["low"].iloc[score_ctx["sl15"][-1]]) if score_ctx.get("sl15") else None

    induce = score_ctx.get("inducement_bull" if up else "inducement_bear", {})
    induce_ok = bool(induce.get("swept"))

    # Zona H1 (OB + FVG searah) untuk cek konfluensi — "How to Choose the
    # Best Order Block When All Zones Look Valid": OB M15 yang overlap
    # dengan zona HTF lebih dipercaya daripada OB M15 berdiri sendiri.
    htf_zones = []
    try:
        for z in detect_order_block(h1, direction, lb=80,
                                    sh=score_ctx.get("sh1", []),
                                    sl=score_ctx.get("sl1", [])):
            htf_zones.append((z["top"], z["bot"]))
        for f in detect_fvg(h1, direction, lb=60):
            htf_zones.append((f["top"], f["bot"]))
    except Exception:
        htf_zones = []

    def _htf_confluence(top: float, bot: float) -> bool:
        return any(zones_overlap(top, bot, zt, zb) for zt, zb in htf_zones)

    # ── Order Block ──────────────────────────────────────────────────
    obs = detect_order_block(m15, direction, lb=60,
                             sh=score_ctx.get("sh15", []),
                             sl=score_ctx.get("sl15", []))
    for z in obs:
        entry_pt = float(z["top"]) if up else float(z["bot"])
        invalid_pt = float(z["bot"]) if up else float(z["top"])

        # A limit order must be a genuine retracement.  The previous 1%
        # tolerance allowed a BUY above market or a SELL below market;
        # Binance can fill those immediately at a different price, leaving
        # the original SL/TP geometry invalid.
        if up:
            if current_price < z["bot"] or entry_pt > current_price * 1.001:
                continue
        else:
            if current_price > z["top"] or entry_pt < current_price * 0.999:
                continue

        sc = 3 + z["quality"]  # base 3 + quality

        if liq_ok:
            sweep_lev = liq.get("level", 0)
            if up and entry_pt >= float(sweep_lev) * 0.995:
                sc += 3
            elif not up and entry_pt <= float(sweep_lev) * 1.005:
                sc += 3
        if choch_ok:
            sc += 2
        if fib_sh and fib_sl and is_in_ote(entry_pt, fib_sl, fib_sh, direction):
            sc += 1
        if _htf_confluence(z["top"], z["bot"]):
            sc += CONFLUENCE_BONUS
        if induce_ok:
            sc += 1

        cands.append({
            "price": round(entry_pt, 8),
            "invalid": round(invalid_pt, 8),
            "label": "ob",
            "score": sc,
        })

    # ── FVG ──────────────────────────────────────────────────────────
    fvgs = detect_fvg(m15, direction, lb=50)
    for f in fvgs:
        if not f["is_fresh"]:
            continue
        entry_pt = f["mid"]
        invalid_pt = f["top"] if up else f["bot"]
        if up:
            if current_price < f["bot"] or entry_pt > current_price * 1.001:
                continue
        else:
            if current_price > f["top"] or entry_pt < current_price * 0.999:
                continue
        sc = 3
        if cisd_ok: sc += 2
        if liq_ok: sc += 2
        if choch_ok: sc += 1
        if _htf_confluence(f["top"], f["bot"]):
            sc += CONFLUENCE_BONUS
        if induce_ok:
            sc += 1
        cands.append({
            "price": round(entry_pt, 8),
            "invalid": round(invalid_pt, 8),
            "label": "fvg",
            "score": sc,
        })

    # ── Equal Highs/Lows ─────────────────────────────────────────────
    eqs = detect_equal_highs_lows(m15, "low" if up else "high", lb=80)
    for eq in eqs[:2]:
        # ── Proximity filter: EQ entry harus REACHABLE dari harga sekarang.
        #
        # Bug asal: tidak ada filter proximity → EQ yang harganya sudah
        # "tersapu" (liquidity sweep) tetap masuk sebagai kandidat entry.
        # Akibatnya SELL limit dipasang DI BAWAH harga pasar → Binance langsung
        # fill di harga pasar (slippage besar) → actual_entry > SL → geometri
        # rusak → auto-out terpicu.
        #
        # SELL (not up): entry di equal HIGH → level harus ≥ current_price
        #   supaya SELL limit menunggu harga naik ke sana, bukan fill sekarang.
        #   Toleransi 0.3%: jika eq < current_price * 0.997 → sudah tersapu.
        #
        # BUY (up): entry di equal LOW → level harus ≤ current_price
        #   supaya BUY limit menunggu harga turun ke sana, bukan fill sekarang.
        #   Toleransi 0.3%: jika eq > current_price * 1.003 → sudah tersapu.
        if not up and float(eq) < current_price * 0.999:
            continue   # EQ high sudah di bawah harga pasar → skip
        if up and float(eq) > current_price * 1.001:
            continue   # EQ low sudah di atas harga pasar → skip

        invalid_pt = eq - atr * 0.8 if up else eq + atr * 0.8
        sc = 2
        if liq_ok: sc += 1
        if induce_ok: sc += 1
        cands.append({
            "price": round(float(eq), 8),
            "invalid": round(float(invalid_pt), 8),
            "label": "eq",
            "score": sc,
        })

    # No synthetic market entry. If no fresh/reachable POI exists, wait for
    # price to come to a real zone.  Before ranking, evaluate LOCATION + RSI
    # timing for every candidate so a good-direction but bad-location setup
    # cannot become the best trade merely because its RR is attractive.
    enriched = []
    for c in cands:
        loc = _entry_location_metrics(m15, direction, c["price"], atr)
        c = dict(c)
        c["location"] = loc
        c["location_score"] = loc["location_score"]
        c["location_state"] = loc["location_state"]
        c["rsi_timing"] = loc["rsi_timing"]
        c["hard_location_block"] = loc["hard_block"]
        c["reject_reason"] = "CHASE_LOCATION" if loc["hard_block"] else None

        # Candidate-specific quality. Location is deliberately meaningful but
        # not dominant: structure/POI can still win, while an obvious chase is
        # removed before SL/TP are calculated.
        c["score"] = int(c.get("score", 0) + round((loc["location_score"] - 50) * 0.35))
        if loc["hard_block"]:
            continue
        enriched.append(c)

    enriched.sort(key=lambda c: (-c["score"], -c.get("location_score", 0)))
    return enriched


# =============================================================================
# STEP 2 — SL STRUKTURAL (anti‑Liquidity Sweep)
# =============================================================================

def _compute_sl(m15: pd.DataFrame, h1: pd.DataFrame, direction: str,
                entry: float, atr: float, liq_sweep: dict,
                invalid_level: Optional[float] = None) -> Tuple[float, float]:
    """
    Hitung SL yang tepat secara struktural, ditempatkan sedemikian rupa
    sehingga jika tersentuh berarti arah benar‑benar salah (bukan LS biasa).
    Buffer anti‑Liquidity Sweep = 0.5 ATR.
    Pilih kandidat dengan prioritas: ob_invalid > struct_h1 > ls_level > struct_m15,
    dan di antara prioritas yang sama pilih yang LEBIH JAUH dari entry
    (lebih tahan noise).
    """
    up = direction == "bull"
    ls_buffer = atr * 0.5
    min_risk = atr * 1.0
    # Risk width is a QUALITY signal, not an automatic rejection. If the
    # real structural invalidation is wide, keep that invalidation rather than
    # moving SL inward just to manufacture a better RR. TP selection will then
    # decide whether a genuine >=2R target exists.
    # A very wide structural invalidation is not repaired by pulling SL inward.
    # The caller will try another entry candidate. This is exactly the CAP lesson:
    # a 4% stop is a property of the chosen entry, not a reason to keep the entry.
    max_risk = min(float(atr) * 2.0, float(entry) * 0.035)

    cands = []

    # 1. invalid level dari OB/FVG
    if invalid_level is not None:
        sl_raw = invalid_level + (-ls_buffer if up else ls_buffer)
        risk = abs(sl_raw - entry)
        if min_risk <= risk <= max_risk:
            cands.append(("ob_invalid", sl_raw, risk))

    # 2. M15 swing (prioritas lebih rendah karena lebih rentan noise)
    sh15, sl15 = swing_pts(m15, lb=3)
    if up and sl15:
        struct_low = float(m15["low"].iloc[sl15[-1]])
        if struct_low < entry:
            sl_raw = struct_low - ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_m15", sl_raw, risk))
    elif not up and sh15:
        struct_high = float(m15["high"].iloc[sh15[-1]])
        if struct_high > entry:
            sl_raw = struct_high + ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_m15", sl_raw, risk))

    # 3. level yang disweep
    if liq_sweep and liq_sweep.get("type") == "sweep" and liq_sweep.get("level"):
        lev = float(liq_sweep["level"])
        if up and lev < entry:
            sl_raw = lev - ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("ls_level", sl_raw, risk))
        elif not up and lev > entry:
            sl_raw = lev + ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("ls_level", sl_raw, risk))

    # 4. H1 swing (lebih lebar, lebih tahan noise)
    sh1, sl1 = swing_pts(h1, lb=5)
    if up and sl1:
        h1_low = float(h1["low"].iloc[sl1[-1]])
        if h1_low < entry:
            sl_raw = h1_low - ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_h1", sl_raw, risk))
    elif not up and sh1:
        h1_high = float(h1["high"].iloc[sh1[-1]])
        if h1_high > entry:
            sl_raw = h1_high + ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_h1", sl_raw, risk))

    if cands:
        # Prioritas: ob_invalid (paling presisi), struct_h1 (paling tahan),
        # ls_level, struct_m15. Dalam prioritas sama, pilih risk terbesar.
        _PRIO = {"ob_invalid": 0, "struct_h1": 1, "ls_level": 2, "struct_m15": 3}
        cands.sort(key=lambda x: (_PRIO.get(x[0], 9), -x[2]))
        _, sl_price, risk = cands[0]
        return sl_price, risk

    # No compact structural invalidation. Do not manufacture an ATR stop.
    raise ValueError("NO_COMPACT_STRUCTURAL_SL")


# =============================================================================
# STEP 3 — TP POOL DAN SELEKSI (dengan ekstensi jika RR < 2.0)
# =============================================================================

def _target_context(h1: pd.DataFrame, m15: pd.DataFrame, direction: str,
                    entry: float, atr: float) -> dict:
    """Context lokal untuk menilai apakah target yang lebih jauh masih credible."""
    up = direction == "bull"
    sh1, sl1 = swing_pts(h1, lb=5)
    struct_h1 = _market_structure(h1, sh1, sl1)
    struct_m15 = _market_structure(m15, *swing_pts(m15, lb=5)) if len(m15) >= 12 else "ranging"
    rsi_val = float(m15["rsi"].iloc[-1]) if ("rsi" in m15.columns and len(m15)) else 50.0
    rv = _relative_volume(m15, 20) if len(m15) >= 23 else 1.0
    mom = _momentum_context(m15) if len(m15) >= 8 else {"bull": False, "bear": False}
    return {
        "h1_structure": struct_h1,
        "m15_structure": struct_m15,
        "rsi": rsi_val,
        "relative_volume": rv,
        "momentum": mom,
        "atr": max(float(atr), 1e-12),
        "entry": float(entry),
        "direction": direction,
        "sh1": sh1,
        "sl1": sl1,
    }


def _build_tp_pool(h1: pd.DataFrame, m15: pd.DataFrame, direction: str,
                   entry: float, atr: float,
                   sh1: list, sl1: list, sh15: list, sl15: list) -> list:
    """Build a rich target pool; selection is performed by _select_tp()."""
    up = direction == "bull"
    sgn = 1 if up else -1
    pool = []

    def add(value, label, tier, anchor_strength=1.0):
        try:
            v = float(value)
        except Exception:
            return
        if sgn * (v - entry) <= atr * 0.35:
            return
        pool.append({
            "value": v,
            "label": str(label),
            "tier": int(tier),
            "anchor_strength": float(anchor_strength),
        })

    opp_dir = "bear" if up else "bull"

    # External liquidity / opposing HTF zones.
    for z in detect_order_block(h1, opp_dir, lb=80, sh=sh1, sl=sl1):
        edge = float(z["bot"]) if up else float(z["top"])
        fresh = 1.0 if z.get("quality", 0) >= 4 else 0.75
        add(edge, "ob_h1", 1, 1.10 * fresh)

    for f in detect_fvg(h1, opp_dir, lb=60):
        add(float(f["mid"]), "fvg_h1", 2, 0.95 if f.get("is_fresh", False) else 0.75)

    sw_vals = ([float(h1["high"].iloc[i]) for i in sh1]
               if up else [float(h1["low"].iloc[i]) for i in sl1])
    for v in sw_vals[-2:]:
        add(v, "sw_h1", 3, 1.00)

    for v in detect_equal_highs_lows(h1, "high" if up else "low", lb=100):
        add(v, "eq_h1", 4, 1.08)

    sw_all = ([float(h1["high"].iloc[i]) for i in sh1]
              if up else [float(h1["low"].iloc[i]) for i in sl1])
    for v in sw_all[:-2]:
        add(v, "sw_h1_far", 5, 0.82)

    # Internal liquidity is useful as a waypoint/path obstacle, but may be a
    # weaker final target than external H1 liquidity.
    for v in detect_equal_highs_lows(m15, "high" if up else "low", lb=80):
        add(v, "eq_m15", 6, 0.62)

    # Extensions are valid farther targets, but start with lower credibility.
    if sh1 and sl1:
        sh_val = float(h1["high"].iloc[sh1[-1]])
        sl_val = float(h1["low"].iloc[sl1[-1]])
        leg = sh_val - sl_val
        if leg > atr * 0.8:
            for ext, lbl, tier, strength in [
                (FIB_EXT_1, "fib127", 7, 0.58),
                (FIB_EXT_2, "fib162", 7, 0.48),
                (1.0, "fib200", 8, 0.38),
                (1.414, "fib241", 8, 0.30),
            ]:
                tp_v = (sh_val + leg * ext) if up else (sl_val - leg * ext)
                add(tp_v, lbl, tier, strength)

    # De-duplicate very close targets; preserve the strongest anchor.
    pool.sort(key=lambda x: (x["value"], x["tier"]))
    dedup = []
    for item in pool:
        if dedup:
            last = dedup[-1]
            if abs(item["value"] - last["value"]) <= atr * 0.12:
                if item["anchor_strength"] > last["anchor_strength"]:
                    dedup[-1] = item
                continue
        dedup.append(item)
    return dedup


def _target_path_score(pool: list, target: dict, entry: float, direction: str,
                       atr: float) -> tuple[float, int]:
    """Score how clear the price path is before a proposed final target."""
    sgn = 1 if direction == "bull" else -1
    target_v = float(target["value"])
    obstacles = 0
    penalty = 0.0
    for other in pool:
        ov = float(other["value"])
        if other is target:
            continue
        if sgn * (ov - entry) <= 0:
            continue
        if sgn * (target_v - ov) <= atr * 0.15:
            strength = max(0.2, float(other.get("anchor_strength", 0.5)))
            tier = int(other.get("tier", 8))
            tier_w = 1.20 if tier <= 4 else (0.80 if tier <= 6 else 0.45)
            distance = abs(ov - entry) / max(atr, 1e-12)
            # Obstacles closer to entry are less damaging because they may be
            # swept/consumed on the way; major levels close to target matter more.
            target_proximity = 1.0 / (1.0 + abs(target_v - ov) / max(atr, 1e-12))
            penalty += strength * tier_w * (0.55 + 0.90 * target_proximity)
            obstacles += 1
    return max(0.0, 1.0 - min(0.78, penalty * 0.10)), obstacles


def _select_tp(pool: list, entry: float, risk: float, direction: str,
               h1: Optional[pd.DataFrame] = None,
               m15: Optional[pd.DataFrame] = None,
               atr: Optional[float] = None) -> Tuple[Optional[float], Optional[str], Optional[float], dict]:
    """Choose the target with the best risk-adjusted plausibility, not merely the first >=2R.

    There is no artificial maximum RR. The engine prefers a target when its structural
    quality, path clarity, market regime and distance form a better *expected-value proxy*.
    The proxy is a ranking heuristic, not a calibrated probability.
    """
    if not pool or risk <= 0:
        return None, None, None, {}

    atr_v = max(float(atr or risk), 1e-12)
    ctx = _target_context(h1, m15, direction, entry, atr_v) if h1 is not None and m15 is not None else {
        "h1_structure": "unknown", "m15_structure": "unknown", "rsi": 50.0,
        "relative_volume": 1.0, "momentum": {"bull": False, "bear": False},
        "atr": atr_v,
    }
    sgn = 1 if direction == "bull" else -1
    candidates = []

    aligned_struct = ctx["h1_structure"] == ("bullish" if direction == "bull" else "bearish")
    aligned_m15 = ctx["m15_structure"] == ("bullish" if direction == "bull" else "bearish")
    momentum_aligned = ((direction == "bull" and ctx["momentum"].get("bull")) or
                        (direction == "bear" and ctx["momentum"].get("bear")))
    rsi = float(ctx["rsi"])
    # RSI is soft context: avoid rewarding an already exhausted extreme.
    rsi_support = 1.0
    if direction == "bull" and rsi >= 75:
        rsi_support = 0.78
    elif direction == "bear" and rsi <= 25:
        rsi_support = 0.78

    for target in pool:
        value = float(target["value"])
        distance = sgn * (value - entry)
        if distance <= 0:
            continue
        rr = distance / risk
        if rr < MIN_RR:
            continue

        path_clear, obstacles = _target_path_score(pool, target, entry, direction, atr_v)
        quality = 0.50
        quality += 0.12 * min(1.0, max(0.0, float(target.get("anchor_strength", 0.5))))

        tier = int(target.get("tier", 8))
        tier_base = {1: 0.14, 2: 0.12, 3: 0.13, 4: 0.15, 5: 0.09, 6: 0.04, 7: 0.02, 8: 0.0}.get(tier, 0.0)
        quality += tier_base
        quality += 0.10 if aligned_struct else (-0.05 if ctx["h1_structure"] in ("bullish", "bearish") else 0.0)
        quality += 0.06 if aligned_m15 else (-0.04 if ctx["m15_structure"] in ("bullish", "bearish") else 0.0)
        quality += 0.06 if momentum_aligned else 0.0
        quality *= rsi_support
        quality *= (0.88 + 0.16 * min(1.0, path_clear))

        # Distance is valuable, but target reachability should decay as the
        # requested move becomes increasingly remote. This prevents a raw
        # "largest RR wins" behaviour while still allowing 6R+ targets when
        # structure, momentum and path quality genuinely support them.
        far_support = 0.0
        if aligned_struct:
            far_support += 0.35
        if aligned_m15:
            far_support += 0.20
        if momentum_aligned:
            far_support += 0.25
        if path_clear > 0.85:
            far_support += 0.20
        decay = max(0.028, 0.085 - 0.055 * far_support)
        distance_excess = max(0.0, rr - 3.0)
        reachability = quality * np.exp(-decay * distance_excess)
        reachability = max(0.06, min(0.93, reachability))

        # Expected-value proxy with diminishing reward. log(1+RR) prevents a
        # mathematically huge target from dominating solely because it is far.
        # It is a ranking tool, NOT a calibrated probability model.
        ev_proxy = reachability * np.log1p(rr) - (1.0 - reachability) * 0.45
        distance_bonus = 0.08 * min(1.0, max(0.0, rr - 2.0) / 4.0)
        far_penalty = 0.0
        # Research showed the useful zone is not simply "lowest RR" or "highest RR".
        # Keep unlimited RR, but require stronger path evidence for extreme targets.
        if rr > 7.0:
            evidence = far_support
            if rr > 9.0:
                far_penalty += min(0.38, 0.055 * (rr - 9.0))
                if evidence < 0.80:
                    far_penalty += min(0.22, (0.80 - evidence) * 0.55)
            elif evidence < 0.60:
                far_penalty += min(0.16, (0.60 - evidence) * 0.45)
        # Prefer the empirically interesting middle-distance region when its
        # structural/path quality is comparable, without hard-capping RR.
        middle_rr_bonus = 0.035 if 5.0 <= rr <= 8.5 else 0.0
        score = ev_proxy + distance_bonus + middle_rr_bonus - far_penalty

        # A target directly beyond several strong obstacles is less attractive.
        score -= min(0.30, obstacles * 0.035)

        candidates.append({
            "label": target["label"],
            "value": round(value, 10),
            "rr": round(rr, 3),
            "tier": tier,
            "target_quality": round(quality, 4),
            "path_clear": round(path_clear, 4),
            "obstacles": int(obstacles),
            "reachability_proxy": round(reachability, 4),
            "ev_proxy": round(ev_proxy, 4),
            "score": round(score, 4),
        })

    if not candidates:
        return None, None, None, {}

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    diagnostics = {
        "selected": best,
        "candidates": candidates[:12],
        "context": {
            "h1_structure": ctx["h1_structure"],
            "m15_structure": ctx["m15_structure"],
            "rsi": round(rsi, 1),
            "relative_volume": round(float(ctx["relative_volume"]), 3),
            "momentum_aligned": bool(momentum_aligned),
            "min_rr": MIN_RR,
            "max_rr": None,
        },
        "method": "structural_target_quality + path_clarity + regime/momentum context + diminishing_distance + EV_proxy",
    }
    return best["value"], best["label"], best["rr"], diagnostics


# =============================================================================
# DECISION BRAIN V26
# =============================================================================

def _clamp(value, lo=0.0, hi=100.0):
    try:
        return float(max(lo, min(hi, float(value))))
    except Exception:
        return lo


def _bool01(value):
    return 1.0 if bool(value) else 0.0


def _regime_profile(score: dict, m15: pd.DataFrame, h1: pd.DataFrame) -> dict:
    h1_struct = str(score.get("struct_h1") or "ranging")
    m15_struct = str(score.get("m15_struct") or "ranging")
    rvol = float(score.get("m15_relative_volume", 1.0) or 1.0)
    atr = max(float(m15["atr"].iloc[-1]), 1e-12)
    body = abs(float(m15["close"].iloc[-1]) - float(m15["open"].iloc[-1])) / atr
    ret3 = abs(float(m15["close"].iloc[-1]) - float(m15["close"].iloc[-4])) / atr if len(m15) >= 4 else 0.0
    trend = 0.0
    if h1_struct in {"bullish", "bearish"}: trend += 0.45
    if m15_struct == h1_struct and h1_struct != "ranging": trend += 0.30
    if ret3 >= 0.7: trend += 0.15
    if rvol >= 1.2: trend += 0.10
    expansion = _clamp((body - 0.35) * 90 + max(0.0, rvol - 1.0) * 35, 0, 100)
    range_score = 100.0 - _clamp(trend * 100, 0, 100)
    transition = 0.0
    if h1_struct == "ranging" and m15_struct in {"bullish", "bearish"}: transition += 0.45
    if h1_struct in {"bullish", "bearish"} and m15_struct not in {h1_struct, "ranging"}: transition += 0.40
    if 0.7 <= rvol <= 1.15 and ret3 < 0.5: transition += 0.15
    if trend >= 0.70 and expansion >= 55: regime = "TREND_EXPANSION"
    elif trend >= 0.55: regime = "TREND"
    elif range_score >= 75 and expansion < 45: regime = "RANGE"
    elif transition >= 0.55: regime = "TRANSITION"
    elif expansion >= 70: regime = "EXPANSION"
    else: regime = "MIXED"
    return {"regime": regime, "trend": round(_clamp(trend * 100), 2), "range": round(range_score, 2), "transition": round(_clamp(transition * 100), 2), "expansion": round(expansion, 2), "rvol": round(rvol, 3), "ret3_atr": round(ret3, 3)}


def _setup_archetype(direction: str, score: dict, candidate: dict, loc: dict, poi_reacted: bool) -> str:
    label = str(candidate.get("label") or "").lower()
    sweep = bool(score.get("selected_sweep"))
    choch = bool((score.get("choch_m15") or {}).get("bullish_choch") if direction == "bull" else (score.get("choch_m15") or {}).get("bearish_choch"))
    cisd = bool((score.get("cisd_m15") or {}).get("bullish_cisd") if direction == "bull" else (score.get("cisd_m15") or {}).get("bearish_cisd"))
    confirmed = bool((score.get("entry_confirmation") or {}).get("confirmed"))
    if sweep and (choch or confirmed):
        return "LIQUIDITY_SWEEP_RECLAIM"
    if label == "ob" and poi_reacted and confirmed:
        return "HTF_POI_DISPLACEMENT_RETEST"
    if label == "fvg" and confirmed:
        return "FVG_DISPLACEMENT_RETEST"
    if choch and cisd:
        return "STRUCTURE_SHIFT_CONTINUATION"
    if label == "eq":
        return "LIQUIDITY_REACTION"
    if loc.get("location_state") == "GOOD":
        return "HEALTHY_PULLBACK"
    return "STRUCTURE_REACTION"


def _thesis_quality(score: dict, loc: dict, candidate: dict, tp_diag: dict, regime: dict, direction: str, poi_reacted: bool) -> dict:
    htf_bias = str(score.get("htf_bias") or "neutral")
    d1_bias = str(score.get("d1_bias") or "neutral")
    desired = "bullish" if direction == "bull" else "bearish"
    structure = 50.0
    if htf_bias == desired: structure += 22
    elif htf_bias == "conflict": structure -= 22
    elif d1_bias == desired: structure += 10
    elif d1_bias not in {"neutral", desired}: structure -= 10
    trend = float(score.get("trend_strength", score.get("direction_edge", 50)) or 50)
    structure += _clamp(trend) * 0.20
    location = float(loc.get("location_score", 50) or 50)
    setup = float(candidate.get("score", 0) or 0)
    setup_norm = _clamp(setup * 8.5, 0, 100)
    confirmation = 100.0 if (score.get("entry_confirmation") or {}).get("confirmed") else 45.0
    liquidity = 100.0 if score.get("selected_sweep") else 55.0
    reaction = 100.0 if poi_reacted else 55.0
    target = _clamp(float(tp_diag.get("target_quality", 0) or 0) * 100)
    path = _clamp(float(tp_diag.get("path_clear", 0) or 0) * 100)
    contradiction = 0.0
    reasons = []
    if regime["regime"] == "TRANSITION":
        contradiction += 12; reasons.append("transition regime")
    if regime["regime"] == "RANGE" and structure > 70 and not score.get("selected_sweep"):
        contradiction += 10; reasons.append("trend thesis inside range")
    if loc.get("location_state") == "WEAK":
        contradiction += 12; reasons.append("weak entry location")
    if loc.get("hard_block"):
        contradiction += 40; reasons.append("entry chase")
    if not (score.get("entry_confirmation") or {}).get("confirmed"):
        contradiction += 8; reasons.append("confirmation incomplete")
    if not poi_reacted:
        contradiction += 4; reasons.append("HTF reaction not confirmed")
    if float(tp_diag.get("obstacles", 0) or 0) >= 3:
        contradiction += 7; reasons.append("target path obstructed")
    # A balanced quality model: no single indicator can dominate.
    base = (
        structure * 0.25 + setup_norm * 0.20 + location * 0.16 +
        confirmation * 0.13 + liquidity * 0.08 + reaction * 0.06 +
        target * 0.07 + path * 0.05
    )
    # Stronger trend regimes reward aligned structure; range regimes reward location/liquidity.
    if regime["regime"] in {"TREND", "TREND_EXPANSION"}:
        base += (structure - 50.0) * 0.08
    elif regime["regime"] == "RANGE":
        base += (liquidity - 50.0) * 0.08 + (location - 50.0) * 0.05
    quality = _clamp(base - contradiction)
    return {
        "trade_quality": round(quality, 2),
        "structure_quality": round(_clamp(structure), 2),
        "setup_quality": round(setup_norm, 2),
        "location_quality": round(location, 2),
        "confirmation_quality": round(confirmation, 2),
        "liquidity_quality": round(liquidity, 2),
        "reaction_quality": round(reaction, 2),
        "target_quality": round(target, 2),
        "path_quality": round(path, 2),
        "contradiction_score": round(_clamp(contradiction), 2),
        "contradictions": reasons[:8],
    }


def _quality_to_confidence(quality: float, uncertainty: float = 0.0) -> int:
    q = _clamp(quality)
    u = _clamp(uncertainty)
    # Compress extremes: confidence is an interpretable quality rank, not a claim of certainty.
    conf = 30.0 + 0.68 * q - 0.18 * u
    return int(max(1, min(99, round(conf))))


def _selection_strength(evaluated: list) -> dict:
    if not evaluated:
        return {"leader_margin": 0.0, "candidate_count": 0}
    ordered = sorted(evaluated, key=lambda x: x.get("execution_score", -1), reverse=True)
    if len(ordered) == 1:
        return {"leader_margin": 100.0, "candidate_count": 1}
    a = float(ordered[0].get("execution_score", 0))
    b = float(ordered[1].get("execution_score", 0))
    return {"leader_margin": round(_clamp(a - b, 0, 100), 3), "candidate_count": len(ordered)}


def _trail_ml_adjustment(analysis: dict, ml: Optional[dict]) -> dict:
    if not ml:
        return {"model_used": False}
    er = float(ml.get("expected_r", 0.0) or 0.0)
    prob = float(ml.get("probability", 0.5) or 0.5)
    return {
        "model_used": True,
        "model_expected_r": round(er, 3),
        "model_probability": round(prob, 3),
        "model_confidence": round(float(ml.get("model_confidence", 50.0) or 50.0), 2),
        "model_version": ml.get("model_version", "unknown"),
    }

# =============================================================================
# FUNGSI UTAMA — Dipanggil oleh main.py
# =============================================================================

def _core_full_analyze(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                 df_d1: Optional[pd.DataFrame] = None,
                 symbol: Optional[str] = None,
                 df_btc_h1: Optional[pd.DataFrame] = None,
                 trade_history: Optional[list] = None) -> Optional[dict]:
    """
    Analisa penuh satu koin: Entry → SL → TP.

    df_btc_h1: candle H1 BTCUSDT, OPSIONAL — kalau dikasih, dipakai sebagai
    filter macro (lihat _macro_bias) supaya sinyal yang jelas melawan arah
    market keseluruhan sedikit diredam, bukan diloloskan mentah-mentah cuma
    berdasar struktur koin itu sendiri. Kalau tidak dikasih (default None),
    perilaku 100% sama seperti sebelumnya — tidak ada filter tambahan.
    """
    try:
        trade_history = trade_history if isinstance(trade_history, list) else []
        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            return None

        if symbol:
            log.info(f"[{symbol}] h1={len(df_h1)} m15={len(df_m15)}")

        # Structural analysis MUST use only fully closed candles.
        # Keep the latest market price separately for execution/pending-entry
        # geometry; the forming candle must not rewrite M15/H1 structure.
        live_price = float(df_m15["close"].iloc[-1])
        h1_closed = _closed_candles(df_h1, 60)
        m15_closed = _closed_candles(df_m15, 15)
        d1_closed = _closed_candles(df_d1, 1440) if df_d1 is not None else None
        score = score_direction(h1_closed, m15_closed, d1_closed, df_btc_h1)
        if score is None:
            if symbol:
                log.debug(f"[{symbol}] score_direction=None (data kurang)")
            return None

        direction = score["direction"]
        cur_price = live_price
        atr = score["atr"]
        confidence = score["confidence"]
        up = direction == "bull"

        if symbol:
            log.info(
                f"[{symbol}] dir={direction} conf={confidence}% "
                f"struct_h1={score['struct_h1']} d1={score['d1_bias']} "
                f"macro={score.get('macro_bias', 'unknown')}"
            )

        h1 = build_df(h1_closed, interval_minutes=60)
        m15 = build_df(m15_closed, interval_minutes=15)
        if h1 is None or m15 is None:
            return None

        # ── STEP 1: ENTRY ────────────────────────────────────────
        # Coba kandidat satu per satu. Candidate #1 tidak boleh membunuh
        # seluruh koin hanya karena SL-nya lebar atau lokasi entry buruk.
        cands = _collect_entry_candidates(m15, h1, direction, cur_price, atr, score)
        if not cands:
            if symbol:
                log.debug(f"[{symbol}] no executable entry candidate (location/POI)")
            return None

        # HTF POI, displacement, sweep, CHoCH/BOS/CISD, inducement, OTE, dan
        # reaction adalah confluence. Geometri entry/SL/TP tetap dihitung per
        # candidate, sehingga candidate buruk bisa dilewati tanpa membuang koin.
        htf_poi = _collect_htf_poi_zones(h1, direction, score)
        poi_reacted = _recent_poi_reaction(m15, htf_poi, direction, atr) if htf_poi else False
        confirmation = score.get("entry_confirmation", {})
        selected_m15_struct = score.get("m15_struct", "ranging")
        opposite_m15 = (
            (direction == "bull" and selected_m15_struct == "bearish") or
            (direction == "bear" and selected_m15_struct == "bullish")
        )

        confluence_bonus = 0
        if htf_poi:
            confluence_bonus += 5
        if poi_reacted:
            confluence_bonus += 7
        if confirmation.get("confirmed"):
            confluence_bonus += 8
        if score.get("selected_sweep"):
            confluence_bonus += 9
        if score.get("trigger_count", 0) >= 2:
            confluence_bonus += 4
        elif score.get("trigger_count", 0) == 1:
            confluence_bonus += 2
        if selected_m15_struct == ("bullish" if up else "bearish"):
            confluence_bonus += 4
        if opposite_m15:
            confluence_bonus -= 6
        base_confidence = max(0, min(99, confidence + confluence_bonus))

        liq_ctx = score["liquidity_bull"] if up else score["liquidity_bear"]
        sh1 = score.get("sh1", [])
        sl1 = score.get("sl1", [])
        sh15 = score.get("sh15", [])
        sl15 = score.get("sl15", [])

        evaluated = []
        for candidate in cands:
            entry = float(candidate["price"])
            entry_lbl = candidate["label"]
            invalid = candidate.get("invalid")
            loc = candidate.get("location", {})

            # Geometry checks per candidate.
            if up and entry > cur_price * 1.005:
                continue
            if not up and entry < cur_price * 0.995:
                continue

            # Keep strategy output compatible with main.py's pre-entry gate.
            # If candidate #1 is too far, evaluate candidate #2/#3 instead of
            # returning a signal that will be rejected immediately after scan.
            # The price can move after this snapshot; main.py still performs the
            # final live validation immediately before placing the order.
            if abs(cur_price - entry) > atr * MAIN_ENTRY_MAX_ATR:
                if symbol:
                    log.debug(
                        f"[{symbol}] candidate {entry_lbl}@{entry:.6g} skipped: "
                        f"ENTRY_TOO_FAR "
                        f"({abs(cur_price-entry)/max(atr,1e-12):.2f} ATR > "
                        f"{MAIN_ENTRY_MAX_ATR:.2f} ATR)"
                    )
                continue

            try:
                sl_price, risk = _compute_sl(
                    m15, h1, direction, entry, atr, liq_ctx, invalid
                )
            except ValueError as exc:
                if symbol:
                    log.debug(
                        f"[{symbol}] candidate {entry_lbl}@{entry:.6g} skipped: {exc}"
                    )
                continue

            if up and sl_price >= entry:
                continue
            if not up and sl_price <= entry:
                continue
            if risk <= 0:
                continue
            if up and cur_price <= sl_price:
                continue
            if not up and cur_price >= sl_price:
                continue

            tp_pool = _build_tp_pool(
                h1, m15, direction, entry, atr, sh1, sl1, sh15, sl15
            )
            tp_price, tp_lbl, rr, tp_diag = _select_tp(tp_pool, entry, risk, direction, h1=h1, m15=m15, atr=atr)
            if tp_price is None or rr is None or rr < MIN_RR:
                if symbol:
                    log.debug(
                        f"[{symbol}] candidate {entry_lbl}@{entry:.6g} skipped: "
                        f"no TP >= {MIN_RR}R"
                    )
                continue

            if up and cur_price >= tp_price:
                continue
            if not up and cur_price <= tp_price:
                continue

            # V3.0: confidence menilai tradeability; entry location + move health lebih berat
            # daripada raw structural confluence. Tidak ada hard rejection baru.
            loc_score = int(loc.get("location_score", 50))
            final_conf, setup_quality, conf_diag = _calibrate_confidence(
                base_confidence, score, loc, candidate, float(rr), htf_poi, poi_reacted, tp_diag=tp_diag
            )
            empirical_conf, empirical_diag = calibrate_confidence_from_history(
                trade_history, final_conf, "BUY" if up else "SELL", entry_lbl, float(rr), setup_quality
            )
            final_conf = empirical_conf
            conf_diag.update(empirical_diag)

            learning_features = _build_learning_features(
                score, loc, candidate, float(rr), entry, atr, cur_price, risk, htf_poi, poi_reacted
            )
            regime = _regime_profile(score, m15, h1)
            thesis = _thesis_quality(score, loc, candidate, tp_diag, regime, direction, poi_reacted)
            archetype = _setup_archetype(direction, score, candidate, loc, poi_reacted)
            uncertainty = min(65.0, thesis["contradiction_score"] * 0.70 + (20.0 if regime["regime"] in {"TRANSITION", "MIXED"} else 0.0))
            quality_conf = _quality_to_confidence(thesis["trade_quality"], uncertainty)
            final_conf = int(round(0.55 * final_conf + 0.45 * quality_conf))
            final_conf = max(1, min(99, final_conf))
            ml_prediction = _predict_learning(learning_features)
            if ml_prediction is not None:
                ml_conf = float(ml_prediction.get("model_confidence", 50.0))
                ml_weight = min(max(float((_LEARNED_MODEL or {}).get("live_weight", 0.35) or 0.35), 0.0), 0.50)
                # Learning model participates as a calibrated modifier, never as an override.
                final_conf = int(round((1.0 - ml_weight) * final_conf + ml_weight * ml_conf))
                final_conf = max(1, min(99, final_conf))
                conf_diag["ml_confidence"] = round(ml_conf, 2)
                conf_diag["ml_expected_r"] = round(float(ml_prediction.get("expected_r", 0.0) or 0.0), 3)
                conf_diag["ml_weight"] = round(ml_weight, 3)
                conf_diag["ml_model_version"] = ml_prediction.get("model_version")
            # Final selection score is based on quality first; RR is only a small feasibility factor.
            execution_score = (
                thesis["trade_quality"] * 1.30
                + final_conf * 0.55
                + min(float(rr), 6.0) * 1.50
                + thesis["target_quality"] * 0.20
                - thesis["contradiction_score"] * 0.65
            )
            # Learned policy is a bounded ranking mutation. It can change which
            # setup wins, but it cannot create an invalid trade or bypass safety.
            policy_effect = _agent_strategy_policy_adjustment(archetype, regime.get("regime", "UNKNOWN"))
            execution_score += float(policy_effect["score_adjustment"])

            evaluated.append({
                "candidate": candidate,
                "entry": entry,
                "entry_lbl": entry_lbl,
                "invalid": invalid,
                "sl": float(sl_price),
                "risk": float(risk),
                "tp": float(tp_price),
                "tp_lbl": tp_lbl,
                "rr": float(rr),
                "confidence": final_conf,
                "setup_quality": setup_quality,
                "confidence_diagnostics": conf_diag,
                "execution_score": execution_score,
                "strategy_policy_effect": policy_effect,
                "location": loc,
                "learning_features": learning_features,
                "learning_prediction": ml_prediction,
                "regime": regime,
                "archetype": archetype,
                "thesis": thesis,
                "uncertainty": round(uncertainty, 2),
            })

        if not evaluated:
            if symbol:
                log.debug(
                    f"[{symbol}] semua candidate gugur di geometry/TP/location/"
                    f"entry-distance; tidak mengembalikan setup yang main.py pasti tolak"
                )
            return None

        evaluated.sort(key=lambda x: x["execution_score"], reverse=True)
        best_eval = evaluated[0]
        best = best_eval["candidate"]
        entry = best_eval["entry"]
        entry_lbl = best_eval["entry_lbl"]
        sl_price = best_eval["sl"]
        risk = best_eval["risk"]
        tp_price = best_eval["tp"]
        tp_lbl = best_eval["tp_lbl"]
        rr = best_eval["rr"]
        confidence = best_eval["confidence"]
        loc = best_eval["location"]
        selection = _selection_strength(evaluated)

        if symbol:
            log.info(
                f"[{symbol}] ENTRY={entry:.6f} label={entry_lbl} "
                f"loc={loc.get('location_score')}%/{loc.get('location_state')} "
                f"RSI={loc.get('rsi')}({loc.get('rsi_timing')})"
            )
            log.info(f"[{symbol}] SL={sl_price:.6f} risk={risk:.6f}")
            log.info(f"[{symbol}] TP={tp_price:.6f} label={tp_lbl} RR={rr:.2f}")

        rsi_val = round(float(m15["rsi"].iloc[-1]), 1)

        return {
            "symbol": symbol,
            "original_dir": direction,
            "decision": "BUY" if up else "SELL",
            "confidence": confidence,
            "direction_confidence": max(0, min(99, base_confidence)),
            "setup_quality": int(round((best_eval.get("thesis") or {}).get("trade_quality", best_eval.get("setup_quality", 0)))),
            "trade_quality": round(float((best_eval.get("thesis") or {}).get("trade_quality", 0.0)), 2),
            "archetype": best_eval.get("archetype", "STRUCTURE_REACTION"),
            "thesis": best_eval.get("thesis", {}),
            "regime_profile": best_eval.get("regime", {}),
            "uncertainty": best_eval.get("uncertainty", 0.0),
            "confidence_band": (best_eval.get("confidence_diagnostics") or {}).get("confidence_band", _confidence_band(confidence)),
            "confidence_diagnostics": best_eval.get("confidence_diagnostics", {}),
            "confidence_model": CONFIDENCE_MODEL_VERSION,
            "confidence_is_probability": bool((best_eval.get("confidence_diagnostics") or {}).get("calibration") == "empirical_bayesian"),
            "learning_features": best_eval.get("learning_features", {}),
            "learning_prediction": best_eval.get("learning_prediction"),
            "learning_model_version": (best_eval.get("learning_prediction") or {}).get("model_version", "static"),
            "rr_band": _rr_band(rr),
            "entry_location_score": loc.get("location_score", 50),
            "entry_location_state": loc.get("location_state", "unknown"),
            "entry_range_position": loc.get("range_position", 0.5),
            "entry_zone_low": loc.get("entry_zone_low"),
            "entry_zone_high": loc.get("entry_zone_high"),
            "rsi_timing": loc.get("rsi_timing", "unknown"),
            "rsi_slope": loc.get("rsi_slope", 0.0),
            "entry": round(entry, 8),
            "price": cur_price,
            "entry_label": entry_lbl,
            "sl": round(sl_price, 8),
            "initial_sl": round(sl_price, 8),
            "initial_risk": round(abs(entry - sl_price), 8),
            "tp": round(tp_price, 8),
            "rr": round(rr, 2),
            "atr": round(atr, 8),
            "rsi": rsi_val,
            "struct_h1": score["struct_h1"],
            "d1_bias": score.get("d1_bias", "neutral"),
            "choch_m15": score["choch_m15"],
            "choch_h1": score["choch_h1"],
            "cisd_m15": score["cisd_m15"],
            "failed_retest": score.get("failed_retest", {}),
            "htf_bias": score.get("htf_bias", "unknown"),
            "h1_bias": score.get("h1_bias", "unknown"),
            "poi_reacted": poi_reacted,
            "entry_confirmation": confirmation,
            "selected_sweep": score.get("selected_sweep", False),
            "trigger_count": score.get("trigger_count", 0),
            "selection_diagnostics": selection,
            "strategy_policy_effect": best_eval.get("strategy_policy_effect", {"policy_active": False, "delta": 0.0}),
            "candidate_count": len(evaluated),
            "tp_sl_reason": (
                f"Entry@{entry:.5g}({entry_lbl}) | "
                f"SL@{sl_price:.5g}(struct) | "
                f"TP@{tp_price:.5g}({tp_lbl}) | RR={rr:.2f} | "
                f"Loc={loc.get('location_score')}({loc.get('location_state')}) | "
                f"RSI={rsi_val}/{loc.get('rsi_timing')}"
            ),
        }

    except Exception as e:
        if symbol:
            log.error(f"[full_analyze] {symbol}: {e}", exc_info=True)
        return None

def get_best_signal(candidates: list) -> Optional[dict]:
    """Pilih peluang terbaik berdasarkan kualitas trade, bukan confidence mentah."""
    if not candidates:
        return None

    def _rank(sig):
        quality = float(sig.get("trade_quality", sig.get("setup_quality", 0)) or 0.0)
        conf = float(sig.get("confidence", 0) or 0.0)
        rr = min(float(sig.get("rr", 0) or 0.0), 6.0)
        diag = sig.get("confidence_diagnostics") or {}
        ml_er = float((sig.get("learning_prediction") or {}).get("expected_r", 0.0) or 0.0)
        contradiction = float((sig.get("thesis") or {}).get("contradiction_score", 0.0) or 0.0)
        calibration_penalty = 0.0
        if diag.get("calibration") == "empirical_bayesian":
            samples = float(diag.get("samples", 0) or 0.0)
            calibration_penalty = max(0.0, 12.0 - min(12.0, samples * 0.15))
        return (
            quality * 1.45
            + conf * 0.55
            + rr * 1.10
            + ml_er * 6.0
            - contradiction * 0.90
            - calibration_penalty
        )

    return max(candidates, key=_rank)


# =============================================================================
# ADAPTIVE POSITION MANAGEMENT — TRAILING BRAIN V8
# =============================================================================
# V8 design principle:
#   Trail is not a static distance-from-price stop. It is a state machine that
#   evaluates continuation health, momentum decay, structural failure,
#   exhaustion and retracement depth before choosing a protection level.
#
# The goal is asymmetric:
#   - healthy winner -> preserve room and avoid noise exits
#   - weakening winner -> lock a meaningful portion of realized edge
#   - confirmed reversal -> move protection close enough to avoid giving back
#     the move, while still respecting the current candle/ATR noise envelope
#
# No API calls are made here. main.py remains the execution layer.

TRAIL_LOOKBACK_CANDLES = STRUCT_TRAIL_LOOKBACK
TRAIL_PREDICTIVE_VERSION = "5.0"
TRAIL_MIN_PROFIT_TO_PROTECT_R = 0.55
TRAIL_CAUTION_R = 0.90
TRAIL_MATURE_R = 1.50
TRAIL_REVERSAL_R = 1.00
TRAIL_STRONG_REVERSAL_R = 1.75
TRAIL_RETRACE_WARN_R = 0.30
TRAIL_RETRACE_STRONG_R = 0.55
TRAIL_LOCK_CAUTION_R = 0.12
TRAIL_LOCK_WEAK_R = 0.28
TRAIL_LOCK_STRONG_R = 0.55
TRAIL_LOCK_REVERSAL_R = 0.80
TRAIL_MIN_GAP_ATR_V8 = 0.32
TRAIL_STRUCT_BUFFER_ATR = 0.36
TRAIL_MOMENTUM_DECAY_ATR = 0.12
TRAIL_OPPOSITE_BODY_ATR = 0.70
TRAIL_REVERSAL_BODY_ATR = 0.95
TRAIL_TWO_BAR_REVERSAL = True
TRAIL_VOLUME_COUNTER = 1.20
TRAIL_VOLUME_EXHAUSTION = 0.72
TRAIL_PEAK_LOOKBACK = 40
TRAIL_PROTECTED_SWING_LB = 3
TRAIL_SCORE_CAUTION = 3
TRAIL_SCORE_WEAK = 5
TRAIL_SCORE_REVERSAL = 7
TRAIL_SCORE_STRONG_REVERSAL = 9


def _trail_directional_return(df: pd.DataFrame, direction: str, bars: int = 3) -> float:
    if df is None or len(df) < bars + 1:
        return 0.0
    atr = max(float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0, 1e-12)
    move = float(df["close"].iloc[-1] - df["close"].iloc[-1-bars])
    return (move if direction == "bull" else -move) / atr


def _trail_peak_metrics(df: pd.DataFrame, direction: str, entry: float,
                        initial_risk: float) -> dict:
    """Measure favorable excursion and give-back from the recent favorable extreme."""
    recent = df.iloc[-min(len(df), TRAIL_PEAK_LOOKBACK):]
    if recent.empty:
        return {"peak_price": entry, "peak_r": 0.0, "giveback_r": 0.0, "retracement_r": 0.0}
    peak_price = float(recent["high"].max()) if direction == "bull" else float(recent["low"].min())
    peak_r = ((peak_price - entry) if direction == "bull" else (entry - peak_price)) / max(initial_risk, 1e-12)
    current_price = float(df["close"].iloc[-1])
    giveback_r = ((peak_price - current_price) if direction == "bull" else (current_price - peak_price)) / max(initial_risk, 1e-12)
    retracement_r = max(0.0, giveback_r)
    return {
        "peak_price": peak_price,
        "peak_r": round(peak_r, 3),
        "giveback_r": round(giveback_r, 3),
        "retracement_r": round(retracement_r, 3),
    }


def _trail_protected_swing(df: pd.DataFrame, direction: str, atr: float) -> Optional[float]:
    """Return the nearest *confirmed* structural stop candidate."""
    recent = df.iloc[-min(len(df), TRAIL_LOOKBACK_CANDLES):]
    if len(recent) < TRAIL_PROTECTED_SWING_LB * 2 + 3:
        return None
    sh, sl = swing_pts(recent, lb=TRAIL_PROTECTED_SWING_LB)
    buf = max(atr * TRAIL_STRUCT_BUFFER_ATR, 1e-12)
    if direction == "bull" and sl:
        return float(recent["low"].iloc[sl[-1]]) - buf
    if direction == "bear" and sh:
        return float(recent["high"].iloc[sh[-1]]) + buf
    return None


def _trail_reversal_analysis(df: pd.DataFrame, direction: str, entry: float,
                             current_price: float, initial_risk: float,
                             tp: Optional[float] = None) -> dict:
    """Estimate continuation health vs. reversal risk from closed M15 candles.

    This is deliberately a *risk-management classifier*, not a claim that the
    market can be predicted perfectly. It looks for agreement across independent
    signals before calling a move a reversal.
    """
    atr = max(float(df["atr"].iloc[-1]), 1e-12)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    prev2 = df.iloc[-3] if len(df) >= 3 else prev
    peak = _trail_peak_metrics(df, direction, entry, initial_risk)
    structure = _market_structure(df, *swing_pts(df, lb=TRAIL_PROTECTED_SWING_LB))
    aligned_structure = structure == ("bullish" if direction == "bull" else "bearish")
    opposite_structure = structure == ("bearish" if direction == "bull" else "bullish")
    mom = _momentum_context(df)
    aligned_mom = (direction == "bull" and mom.get("bull")) or (direction == "bear" and mom.get("bear"))
    opposite_mom = (direction == "bull" and mom.get("bear")) or (direction == "bear" and mom.get("bull"))
    rdiv = detect_rsi_divergence(df, direction, lb=30)
    divergence = ((direction == "bull" and rdiv.get("bear_div")) or
                  (direction == "bear" and rdiv.get("bull_div")))
    vol = _relative_volume(df, 20)

    last_body_atr = abs(float(last["close"] - last["open"])) / atr
    prev_body_atr = abs(float(prev["close"] - prev["open"])) / atr
    opposite_last = (direction == "bull" and float(last["close"]) < float(last["open"])) or (direction == "bear" and float(last["close"]) > float(last["open"]))
    opposite_prev = (direction == "bull" and float(prev["close"]) < float(prev["open"])) or (direction == "bear" and float(prev["close"]) > float(prev["open"]))
    two_bar_opposite = opposite_last and opposite_prev if TRAIL_TWO_BAR_REVERSAL else False

    # Directional return is measured in ATR units. A healthy trade should retain
    # positive directional return over several lookbacks.
    ret3 = _trail_directional_return(df, direction, 3)
    ret5 = _trail_directional_return(df, direction, 5) if len(df) >= 6 else ret3
    momentum_decay = (ret5 < TRAIL_MOMENTUM_DECAY_ATR and ret3 < 0.0)

    # Detect a fresh opposite displacement / break of the short-term range.
    look = df.iloc[-6:-1] if len(df) >= 7 else df.iloc[:-1]
    prior_high = float(look["high"].max()) if not look.empty else float(prev["high"])
    prior_low = float(look["low"].min()) if not look.empty else float(prev["low"])
    opposite_break = ((direction == "bull" and float(last["close"]) < prior_low) or
                      (direction == "bear" and float(last["close"]) > prior_high))

    # Protected swing failure is stronger than a wick: the close must break the
    # latest confirmed swing in the adverse direction.
    sh, sl = swing_pts(df, lb=TRAIL_PROTECTED_SWING_LB)
    protected_break = False
    if direction == "bull" and sl:
        protected_break = float(last["close"]) < float(df["low"].iloc[sl[-1]])
    elif direction == "bear" and sh:
        protected_break = float(last["close"]) > float(df["high"].iloc[sh[-1]])

    # Liquidity rejection = extension followed by close back through the candle body.
    recent_high = float(df["high"].iloc[-15:].max())
    recent_low = float(df["low"].iloc[-15:].min())
    liquidity_rejection = False
    if direction == "bull":
        liquidity_rejection = (float(last["high"]) >= recent_high * (1 - 1e-9)
                               and opposite_last and last_body_atr >= TRAIL_OPPOSITE_BODY_ATR)
    else:
        liquidity_rejection = (float(last["low"]) <= recent_low * (1 + 1e-9)
                               and opposite_last and last_body_atr >= TRAIL_OPPOSITE_BODY_ATR)

    score = 0
    reasons = []
    confirmations = 0

    if opposite_structure:
        score += 3
        confirmations += 1
        reasons.append("opposite_structure")
    elif not aligned_structure:
        score += 1
        reasons.append("structure_ranging")
    else:
        reasons.append("structure_aligned")

    if opposite_mom:
        score += 2
        confirmations += 1
        reasons.append("opposite_momentum")
    elif aligned_mom:
        reasons.append("momentum_aligned")
    elif momentum_decay:
        score += 1
        reasons.append("momentum_decay")

    if divergence:
        score += 2 if not rdiv.get("strong") else 3
        confirmations += 1
        reasons.append("rsi_divergence")

    if two_bar_opposite:
        score += 2
        confirmations += 1
        reasons.append("two_bar_opposite")
    elif opposite_last and last_body_atr >= TRAIL_OPPOSITE_BODY_ATR:
        score += 1
        reasons.append("opposite_candle")

    if opposite_break:
        score += 2
        confirmations += 1
        reasons.append("opposite_range_break")

    if protected_break:
        score += 3
        confirmations += 1
        reasons.append("protected_swing_break")

    if liquidity_rejection:
        score += 2
        confirmations += 1
        reasons.append("liquidity_rejection")

    if vol >= TRAIL_VOLUME_COUNTER and opposite_last:
        score += 2
        confirmations += 1
        reasons.append("counter_volume_expansion")
    elif vol <= TRAIL_VOLUME_EXHAUSTION and peak["peak_r"] >= TRAIL_CAUTION_R:
        score += 1
        reasons.append("volume_exhaustion")

    if peak["retracement_r"] >= TRAIL_RETRACE_WARN_R and peak["peak_r"] >= TRAIL_MIN_PROFIT_TO_PROTECT_R:
        score += 2
        reasons.append("meaningful_giveback")
    if peak["retracement_r"] >= TRAIL_RETRACE_STRONG_R:
        score += 2
        confirmations += 1
        reasons.append("deep_giveback")

    if tp is not None:
        target_distance_r = abs(float(tp) - current_price) / max(initial_risk, 1e-12)
        if target_distance_r <= TRAIL_LIQUIDITY_NEAR_R:
            score += 1
            reasons.append("target_near")
    else:
        target_distance_r = None

    strong_reversal = (protected_break and opposite_break and confirmations >= 3) or score >= TRAIL_SCORE_STRONG_REVERSAL
    reversal = strong_reversal or (score >= TRAIL_SCORE_REVERSAL and confirmations >= 2)
    weakening = reversal or score >= TRAIL_SCORE_WEAK or peak["retracement_r"] >= TRAIL_RETRACE_WARN_R
    # Profit maturity arms the trailing engine, but is NOT itself evidence of
    # reversal. A large floating profit with healthy structure must still be
    # allowed to breathe.
    caution = weakening or score >= TRAIL_SCORE_CAUTION

    if strong_reversal:
        state = "REVERSAL_CONFIRMED"
    elif reversal:
        state = "REVERSAL"
    elif weakening:
        state = "WEAKENING"
    elif caution:
        state = "CAUTION"
    else:
        state = "HEALTHY"

    return {
        "state": state,
        "score": int(score),
        "confirmations": int(confirmations),
        "reasons": reasons,
        "aligned_structure": bool(aligned_structure),
        "opposite_structure": bool(opposite_structure),
        "aligned_momentum": bool(aligned_mom),
        "opposite_momentum": bool(opposite_mom),
        "divergence": bool(divergence),
        "protected_break": bool(protected_break),
        "opposite_break": bool(opposite_break),
        "liquidity_rejection": bool(liquidity_rejection),
        "last_body_atr": round(last_body_atr, 3),
        "prev_body_atr": round(prev_body_atr, 3),
        "relative_volume": round(vol, 3),
        "ret3_atr": round(ret3, 3),
        "ret5_atr": round(ret5, 3),
        "peak_price": round(peak["peak_price"], 10),
        "peak_r": peak["peak_r"],
        "giveback_r": peak["giveback_r"],
        "target_distance_r": round(target_distance_r, 3) if target_distance_r is not None else None,
    }


def _trail_lock_floor(entry: float, initial_risk: float, state: str, direction: str) -> float:
    lock_r = {
        "CAUTION": TRAIL_LOCK_CAUTION_R,
        "WEAKENING": TRAIL_LOCK_WEAK_R,
        "REVERSAL": TRAIL_LOCK_REVERSAL_R,
        "REVERSAL_CONFIRMED": TRAIL_LOCK_REVERSAL_R,
    }.get(state, 0.0)
    return entry + (initial_risk * lock_r if direction == "bull" else -initial_risk * lock_r)


def _trail_stop_candidates(df_m15: pd.DataFrame, direction: str, entry: float,
                           current_price: float, initial_risk: float,
                           analysis: dict) -> dict:
    """Build protection candidates from structure, retracement and momentum.

    The candidates are *floors/ceilings*, not direct orders. manage_position()
    selects the tightest candidate that remains outside the current noise band.
    """
    atr = max(float(df_m15["atr"].iloc[-1]), 1e-12)
    state = analysis["state"]
    peak_price = float(analysis["peak_price"])
    weakness = int(analysis["score"])

    structure_stop = _trail_protected_swing(df_m15, direction, atr)
    min_gap = atr * TRAIL_MIN_GAP_ATR_V8

    # Momentum stop is adaptive to weakness. More weakness => smaller distance
    # from the favorable extreme, but never inside the current noise gap.
    mult = 2.20
    if state == "CAUTION":
        mult = 1.80
    elif state == "WEAKENING":
        mult = 1.45
    elif state == "REVERSAL":
        mult = 1.10
    elif state == "REVERSAL_CONFIRMED":
        mult = 0.85
    momentum_stop = (peak_price - atr * mult) if direction == "bull" else (peak_price + atr * mult)

    # Retracement-based stop locks part of the move before a new confirmed swing
    # is available. This is the key V8 addition: it can protect a winner *before*
    # the market has completed a textbook BOS/ChoCH.
    retrace_lock_r = {
        "CAUTION": 0.05,
        "WEAKENING": 0.18,
        "REVERSAL": 0.38,
        "REVERSAL_CONFIRMED": 0.55,
    }.get(state, 0.0)
    retrace_stop = (peak_price - max(initial_risk * retrace_lock_r, atr * 0.20)
                    if direction == "bull"
                    else peak_price + max(initial_risk * retrace_lock_r, atr * 0.20))

    lock_floor = _trail_lock_floor(entry, initial_risk, state, direction)

    # In a confirmed reversal, use the tighter of retracement and momentum stops.
    # In healthy/caution phases structure remains the primary authority.
    if direction == "bull":
        usable_ceiling = current_price - min_gap
        candidates = {
            "structure": structure_stop,
            "momentum": min(momentum_stop, usable_ceiling),
            "retracement": min(retrace_stop, usable_ceiling),
            "lock_floor": min(lock_floor, usable_ceiling),
        }
    else:
        usable_floor = current_price + min_gap
        candidates = {
            "structure": structure_stop,
            "momentum": max(momentum_stop, usable_floor),
            "retracement": max(retrace_stop, usable_floor),
            "lock_floor": max(lock_floor, usable_floor),
        }

    return {
        "candidates": candidates,
        "atr": atr,
        "min_gap": min_gap,
        "weakness_score": weakness,
        "state": state,
    }


def _choose_trail_stop(candidates: dict, direction: str, current_price: float,
                       current_sl: float, atr: float, state: str) -> tuple[Optional[float], Optional[str]]:
    valid = []
    for name, value in candidates.items():
        if value is None:
            continue
        v = float(value)
        if direction == "bull":
            if current_sl < v < current_price - atr * 0.10:
                valid.append((v, name))
        else:
            if current_price + atr * 0.10 < v < current_sl:
                valid.append((v, name))

    if not valid:
        return None, None

    # Healthy/caution: prefer structural integrity. Weak/reversal: prefer the
    # tightest valid protection to minimize give-back.
    priority = {"structure": 0, "momentum": 1, "retracement": 2, "lock_floor": 3}
    if state in {"HEALTHY", "CAUTION"}:
        # Healthy/caution is explicitly anti-overtrailing. Structure is preferred;
        # when unavailable, use the wider momentum stop rather than a close
        # retracement/lock candidate.
        structural = [x for x in valid if x[1] == "structure"]
        if structural:
            return structural[0]
        momentum = [x for x in valid if x[1] == "momentum"]
        if momentum:
            return (max(momentum, key=lambda x: x[0]) if direction == "bull"
                    else min(momentum, key=lambda x: x[0]))
        lock = [x for x in valid if x[1] == "lock_floor"]
        if lock:
            return (max(lock, key=lambda x: x[0]) if direction == "bull"
                    else min(lock, key=lambda x: x[0]))
        return (max(valid, key=lambda x: x[0]) if direction == "bull"
                else min(valid, key=lambda x: x[0]))

    if direction == "bull":
        return max(valid, key=lambda x: x[0])
    return min(valid, key=lambda x: x[0])


def _relative_volume(df: pd.DataFrame, n: int = 20) -> float:
    """Relative volume proxy dari data OHLCV yang SUDAH DIMILIKI bot."""
    if df is None or df.empty or "volume" not in df.columns:
        return 1.0
    vol = pd.to_numeric(df["volume"], errors="coerce").dropna()
    if len(vol) < n + 3:
        return 1.0
    base = float(vol.iloc[-n-1:-1].mean())
    return float(vol.iloc[-1] / max(base, 1e-12))


def _momentum_context(df: pd.DataFrame) -> dict:
    """Momentum lokal: EMA slope + multi-horizon return + candle expansion."""
    if df is None or len(df) < 8:
        return {"bull": False, "bear": False, "expanding": False, "slope_atr": 0.0,
                "ret3_atr": 0.0, "ret5_atr": 0.0, "last_body_atr": 0.0}
    close = df["close"].astype(float)
    atr = max(float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0, 1e-12)
    e9 = ema(close, 9)
    slope = float(e9.iloc[-1] - e9.iloc[-4]) / atr
    ret3 = float(close.iloc[-1] - close.iloc[-4]) / atr
    ret5 = float(close.iloc[-1] - close.iloc[-6]) / atr if len(df) >= 6 else ret3
    last_body = abs(float(df["close"].iloc[-1] - df["open"].iloc[-1])) / atr
    return {
        "bull": slope > 0.20 and ret3 > 0.15,
        "bear": slope < -0.20 and ret3 < -0.15,
        "expanding": last_body >= 0.80,
        "slope_atr": round(slope, 3),
        "ret3_atr": round(ret3, 3),
        "ret5_atr": round(ret5, 3),
        "last_body_atr": round(last_body, 3),
    }


def _core_manage_position(state: dict, df_m15: pd.DataFrame, df_h1: Optional[pd.DataFrame] = None,
                    df_d1: Optional[pd.DataFrame] = None, symbol: Optional[str] = None) -> dict:
    """Position-management brain V8 — predictive, stateful-in-time, execution-free.

    The engine does not predict exact future prices. It detects when the evidence
    for continuation is degrading and moves protection *before* a full reversal
    is complete. This is intentionally asymmetric: healthy winners get room;
    weakening/reversing winners surrender progressively less of the move.
    """
    if df_m15 is None or df_m15.empty:
        return {"action": "HOLD", "state": "NO_DATA", "reason": ["M15 unavailable"]}

    live_price = float(df_m15["close"].iloc[-1])
    m15 = build_df(_closed_candles(df_m15, 15), interval_minutes=15)
    if m15 is None or len(m15) < 55:
        return {"action": "HOLD", "state": "NO_DATA", "reason": ["insufficient M15 data"]}

    sig = state.get("signal", state) or {}
    decision = str(sig.get("decision") or state.get("decision") or "BUY").upper()
    direction = "bull" if decision == "BUY" else "bear"
    entry = float(state.get("entry") or sig.get("entry") or m15["close"].iloc[-1])
    current_price = float(state.get("current_price") or state.get("price") or live_price)
    current_sl = float(state.get("current_sl") or sig.get("sl") or entry)
    initial_sl = float(state.get("initial_sl") or sig.get("initial_sl") or sig.get("sl") or current_sl)
    tp = sig.get("tp")
    tp = float(tp) if tp is not None else None
    initial_risk = max(abs(entry - initial_sl), 1e-12)
    profit_r = ((current_price - entry) if direction == "bull" else (entry - current_price)) / initial_risk

    analysis = _trail_reversal_analysis(
        m15, direction, entry, current_price, initial_risk, tp=tp
    )
    state_name = analysis["state"]
    reasons = list(analysis["reasons"])

    # Before enough profit exists, do not force a trail unless the price has already
    # invalidated the thesis. We still return diagnostics so main.py can log why.
    if profit_r < TRAIL_MIN_PROFIT_TO_PROTECT_R:
        return {
            "action": "HOLD",
            "state": "DEVELOPING",
            "profit_r": round(profit_r, 2),
            "weakness_score": analysis["score"],
            "reversal_state": state_name,
            "reversal_confirmations": analysis["confirmations"],
            "reversal_diagnostics": analysis,
            "reason": reasons + ["profit_not_mature_for_trail"],
        }

    trail_pack = _trail_stop_candidates(
        m15, direction, entry, current_price, initial_risk, analysis
    )
    candidates = trail_pack["candidates"]
    atr = trail_pack["atr"]
    chosen, source = _choose_trail_stop(
        candidates, direction, current_price, current_sl, atr, state_name
    )
    # The entry model is not reused as a trail predictor. Trail decisions are based
    # on current position-path evidence; a dedicated trail learner can be plugged
    # in later once enough trail-labelled samples exist.
    trail_model_diag = {"model_used": False, "reason": "entry_model_not_reused_for_trail"}

    if chosen is not None:
        minimum_lock_r = TRAIL_LOCK_WEAK_R
        if state_name == "CAUTION":
            minimum_lock_r = TRAIL_LOCK_CAUTION_R
        elif state_name == "WEAKENING":
            minimum_lock_r = TRAIL_LOCK_WEAK_R
        elif state_name == "REVERSAL":
            minimum_lock_r = TRAIL_LOCK_REVERSAL_R * 0.65
        elif state_name == "REVERSAL_CONFIRMED":
            minimum_lock_r = TRAIL_LOCK_REVERSAL_R

        locked_r = ((chosen - entry) if direction == "bull" else (entry - chosen)) / initial_risk
        if locked_r < minimum_lock_r and state_name in {"REVERSAL", "REVERSAL_CONFIRMED"}:
            # Fall back to the strongest valid candidate that actually preserves
            # at least the minimum lock; if none exists, keep the previous SL.
            eligible = []
            for name, value in candidates.items():
                if value is None:
                    continue
                v = float(value)
                lr = ((v - entry) if direction == "bull" else (entry - v)) / initial_risk
                if lr >= minimum_lock_r:
                    if direction == "bull" and current_sl < v < current_price - atr * 0.10:
                        eligible.append((v, name))
                    elif direction == "bear" and current_price + atr * 0.10 < v < current_sl:
                        eligible.append((v, name))
            if eligible:
                chosen, source = (max(eligible, key=lambda x: x[0]) if direction == "bull"
                                  else min(eligible, key=lambda x: x[0]))

        return {
            "action": "TRAIL",
            "state": state_name,
            "sl": round(float(chosen), 10),
            "profit_r": round(profit_r, 2),
            "locked_r": round((((chosen - entry) if direction == "bull" else (entry - chosen)) / initial_risk), 3),
            "weakness_score": int(analysis["score"]),
            "reversal_confirmations": int(analysis["confirmations"]),
            "trail_source": source,
            "reversal_diagnostics": analysis,
            "trail_model": trail_model_diag,
            "reason": reasons + [f"predictive_trail:{source}"],
        }

    # A confirmed reversal without an improvable stop is safer handled as a controlled
    # exit once there is enough realized profit. This avoids holding a winner all the
    # way back to its original SL just because no structural candidate exists.
    if state_name == "REVERSAL_CONFIRMED" and profit_r >= TRAIL_STRONG_REVERSAL_R:
        return {
            "action": "EXIT",
            "close": True,
            "state": state_name,
            "profit_r": round(profit_r, 2),
            "weakness_score": int(analysis["score"]),
            "reversal_confirmations": int(analysis["confirmations"]),
            "reversal_diagnostics": analysis,
            "reason": reasons + ["confirmed_reversal_no_safe_trail"]
        }

    if state_name in {"REVERSAL", "REVERSAL_CONFIRMED", "WEAKENING"}:
        return {
            "action": "PROTECT",
            "state": state_name,
            "profit_r": round(profit_r, 2),
            "weakness_score": int(analysis["score"]),
            "reversal_confirmations": int(analysis["confirmations"]),
            "reversal_diagnostics": analysis,
            "reason": reasons + ["reversal_detected_but_no_safe_improvement"]
        }

    return {
        "action": "PROTECT" if profit_r >= TRAIL_CAUTION_R else "HOLD",
        "state": state_name,
        "profit_r": round(profit_r, 2),
        "weakness_score": int(analysis["score"]),
        "reversal_confirmations": int(analysis["confirmations"]),
        "reversal_diagnostics": analysis,
        "reason": reasons + ["trend_still_healthy_or_no_better_stop"]
    }


# =============================================================================
# VALIDASI PRE-ORDER & KOREKSI GEOMETRY (dipanggil oleh main.py)
# =============================================================================

def validate_and_adjust_geometry(
    entry: float, sl: float, tp: float,
    current_price: float, atr: float,
    direction: str,
) -> Optional[dict]:
    """
    Validasi dan (jika perlu) koreksi geometri entry/SL/TP sebelum order dipasang
    atau setelah order terisi di harga yang berbeda dari target.

    Mengapa fungsi ini penting:
    ─────────────────────────────────────────────────────────────────────────────
    Sinyal dihitung pada waktu T. Saat order terpasang (T+beberapa detik/menit),
    harga pasar bisa sudah bergerak — khususnya:

      • Kasus "geometri invalid setelah order terisi":
        SELL limit di entry_target, tapi actual_fill = harga pasar (lebih tinggi
        dari entry_target karena market sudah di atas limit sell). Akibatnya
        actual_fill > SL → geometri rusak.

      • Kasus "harga sudah melewati SL setelah order terisi":
        Sinyal sudah kedaluwarsa atau fill terjadi di sisi yang salah.
        Posisi tidak boleh diselamatkan dengan menggeser entry/SL
        secara retroaktif.

    Logika:
    ─────────────────────────────────────────────────────────────────────────────
    1. Cek geometri dasar: SL di sisi yang benar dari entry, TP di sisi lain.
    2. Cek SL belum ditembus current_price.
    3. Jika SL ditembus atau geometri fill berubah, tolak posisi secara aman.
       Wick M1 tidak cukup untuk membuktikan bahwa posisi masih valid.
    4. Cek RR ≥ MIN_RR tanpa mengubah level trade.

    Return:
      dict  {entry, sl, tp, rr, adjusted} jika valid / bisa diselamatkan
      None  jika tidak bisa diperbaiki → TOLAK sinyal / auto-out
    """
    up = direction == "bull"
    def _geo_ok(e: float, s: float, t: float) -> bool:
        return (s < e < t) if up else (t < e < s)

    def _rr(e: float, s: float, t: float) -> float:
        return abs(t - e) / max(abs(e - s), 1e-10)

    sl_breached = (current_price <= sl) if up else (current_price >= sl)

    # ─── Kasus 1: sudah valid ────────────────────────────────────────────────
    if _geo_ok(entry, sl, tp) and not sl_breached:
        rr = _rr(entry, sl, tp)
        if rr < MIN_RR:
            return None
        return {"entry": entry, "sl": sl, "tp": tp, "rr": round(rr, 2), "adjusted": False}

    # ─── Kasus 2: SL ditembus → fail closed ─────────────────────────────────
    if sl_breached:
        log.info(
            f"[validate_geo] SL sudah ditembus sebelum validasi "
            f"(entry={entry:.6g}, sl={sl:.6g}, price={current_price:.6g}) — ditolak"
        )
        return None

    # Never move entry or SL after a fill. A changed geometry is a stale
    # signal, not a new setup.
    if not _geo_ok(entry, sl, tp):
        return None

    rr = _rr(entry, sl, tp)
    if rr < MIN_RR:
        log.debug(
            f"[validate_geo] RR={rr:.2f} < MIN_RR={MIN_RR} setelah koreksi — ditolak"
        )
        return None

    return {
        "entry": round(entry, 8),
        "sl":    round(sl, 8),
        "tp":    tp,
        "rr":    round(rr, 2),
        "adjusted": False,
    }

# =============================================================================
# MACHINE LEARNING COGNITIVE LAYER
# =============================================================================
# V27: belief revision, evidence arbitration, failure autopsy, candidate
# comparison, richer research snapshots, and FULL command interface.
# This layer is deliberately execution-free and Binance-free.

ML_COGNITIVE_VERSION = "V40_FULL_BRAIN_REBUILT"
FULL_LEARNING_SCHEMA = "full_learning_cognitive_v1"
FULL_BELIEF_DIR = Path(os.getenv("FULL_STATE_DIR", "machine_learning_state"))
FULL_BELIEF_FILE = FULL_BELIEF_DIR / "belief_state.json"
FULL_LESSONS_FILE = FULL_BELIEF_DIR / "lessons.jsonl"


def _finite(value, default=0.0):
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _v34_clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, _finite(value)))


def _load_json_file(path, default):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return default
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, type(default)) else default
    except Exception as exc:
        log.warning(f"[FULL] load state gagal: {exc}")
        return default


def _save_json_atomic(path, obj):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception as exc:
        log.warning(f"[FULL] save state gagal: {exc}")
        return False


_FULL_LOCK = threading.RLock()
_FULL_BELIEFS = _load_json_file(FULL_BELIEF_FILE, {
    "schema": FULL_LEARNING_SCHEMA,
    "beliefs": {},
    "lessons": 0,
    "revisions": 0,
    "last_review": None,
})
if _FULL_BELIEFS.get("schema") != FULL_LEARNING_SCHEMA:
    _FULL_BELIEFS = {"schema": FULL_LEARNING_SCHEMA, "beliefs": {}, "lessons": 0, "revisions": 0, "last_review": None}


def _belief_key(name, condition="global"):
    return f"{str(name).strip().lower()}::{str(condition).strip().lower()}"


def get_belief(name, condition="global", default_value=None):
    key = _belief_key(name, condition)
    with _FULL_LOCK:
        b = dict((_FULL_BELIEFS.get("beliefs") or {}).get(key) or {})
    if b:
        return b
    return {
        "name": name,
        "condition": condition,
        "value": default_value,
        "belief_strength": 0.35,
        "sample_size": 0,
        "evidence_for": [],
        "evidence_against": [],
        "last_revision": None,
        "status": "UNESTABLISHED",
    }


def revise_belief(name, observed_value, condition="global", evidence_strength=0.5,
                  source="unknown", reason="", sample_increment=1):
    """Bayes-like gradual belief revision; evidence changes the belief, not a hard override."""
    observed = _finite(observed_value)
    strength = _v34_clamp(evidence_strength)
    now = datetime.now(timezone.utc).isoformat()
    key = _belief_key(name, condition)
    with _FULL_LOCK:
        beliefs = _FULL_BELIEFS.setdefault("beliefs", {})
        old = dict(beliefs.get(key) or {})
        old_value = _finite(old.get("value"), observed)
        old_strength = _v34_clamp(old.get("belief_strength", 0.35))
        old_n = int(old.get("sample_size", 0) or 0)
        prior_weight = max(1.0, old_n * max(0.20, old_strength))
        evidence_weight = max(0.10, strength)
        new_value = (old_value * prior_weight + observed * evidence_weight) / (prior_weight + evidence_weight)
        new_strength = _v34_clamp((old_strength * prior_weight + strength * evidence_weight) / (prior_weight + evidence_weight))
        change = abs(new_value - old_value)
        status = "TENTATIVE" if old_n + sample_increment < 30 else ("SUPPORTED" if new_strength >= 0.65 else "MIXED")
        row = {
            "name": name,
            "condition": condition,
            "value": new_value,
            "belief_strength": new_strength,
            "sample_size": old_n + int(sample_increment),
            "evidence_for": (old.get("evidence_for") or [])[-9:],
            "evidence_against": (old.get("evidence_against") or [])[-9:],
            "last_revision": now,
            "last_change": change,
            "last_source": source,
            "last_reason": reason,
            "status": status,
        }
        if change > 0:
            if observed >= old_value:
                row["evidence_for"].append({"value": observed, "strength": strength, "source": source, "reason": reason, "at": now})
            else:
                row["evidence_against"].append({"value": observed, "strength": strength, "source": source, "reason": reason, "at": now})
        beliefs[key] = row
        _FULL_BELIEFS["revisions"] = int(_FULL_BELIEFS.get("revisions", 0) or 0) + 1
        _FULL_BELIEFS["last_review"] = now
        _save_json_atomic(FULL_BELIEF_FILE, _FULL_BELIEFS)
    return row


def _append_lesson(lesson):
    try:
        FULL_BELIEF_DIR.mkdir(parents=True, exist_ok=True)
        with FULL_LESSONS_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(lesson, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        return True
    except Exception as exc:
        log.warning(f"[FULL] lesson append gagal: {exc}")
        return False


def _safe_ratio(num, den, default=0.0):
    den = _finite(den)
    if abs(den) < 1e-12:
        return default
    return _finite(num) / den


def _detect_regime_local(h1, m15, d1=None):
    result = {"regime": "UNKNOWN", "trend_strength": 0.0, "volatility": 0.0, "range_position": 0.5}
    try:
        h = h1 if isinstance(h1, pd.DataFrame) else None
        m = m15 if isinstance(m15, pd.DataFrame) else None
        if h is None or m is None or len(h) < 20 or len(m) < 20:
            return result
        h_close = _finite(h["close"].iloc[-1])
        h_old = _finite(h["close"].iloc[-20])
        atr_h = _finite(h.get("atr", pd.Series(dtype=float)).iloc[-1] if "atr" in h else np.mean(np.abs(h["high"].tail(14).to_numpy() - h["low"].tail(14).to_numpy())))
        move = _safe_ratio(h_close - h_old, max(abs(h_old), 1e-12))
        trend = _v34_clamp(abs(move) / max(atr_h / max(abs(h_close), 1e-12), 1e-6), 0, 3.0) / 3.0
        window = m["close"].tail(32)
        lo, hi = float(window.min()), float(window.max())
        rp = _v34_clamp(_safe_ratio(float(m["close"].iloc[-1]) - lo, hi - lo, 0.5)) if hi > lo else 0.5
        vol = _safe_ratio(float(m["high"].tail(16).max() - m["low"].tail(16).min()), max(abs(float(m["close"].tail(16).mean())), 1e-12))
        result["trend_strength"] = trend
        result["volatility"] = _v34_clamp(vol / 0.25)
        result["range_position"] = rp
        if trend > 0.55 and move > 0:
            result["regime"] = "TREND_UP"
        elif trend > 0.55 and move < 0:
            result["regime"] = "TREND_DOWN"
        elif result["volatility"] > 0.75 and trend < 0.45:
            result["regime"] = "TRANSITION"
        else:
            result["regime"] = "RANGE"
    except Exception:
        pass
    return result


def _evidence_arbitration(signal, regime=None):
    thesis = signal.get("thesis") or {}
    quality = _finite(signal.get("trade_quality", signal.get("setup_quality", 0.0)))
    contradictions = _finite(thesis.get("contradiction_score", 0.0))
    confidence = _finite(signal.get("confidence", 50.0))
    uncertainty = _finite(signal.get("uncertainty", 0.5))
    support = []
    against = []
    for key, label in (
        ("selected_sweep", "liquidity sweep"),
        ("poi_reacted", "POI reaction"),
        ("entry_confirmation", "entry confirmation"),
    ):
        if signal.get(key):
            support.append(label)
    if contradictions > 30:
        against.append("meaningful contradictions")
    if _finite(signal.get("entry_location_score", 50)) < 45:
        against.append("weak entry location")
    if _finite(signal.get("rr", 0)) < 2.0:
        against.append("weak target geometry")
    regime_name = (regime or signal.get("regime_profile", {}) or {}).get("regime") if isinstance(regime, dict) else regime
    decision_quality = _v34_clamp((quality / 100.0) * 0.52 + (confidence / 100.0) * 0.28 + (1.0 - _v34_clamp(contradictions / 100.0)) * 0.12 + (1.0 - _v34_clamp(uncertainty / 100.0)) * 0.08) * 100.0
    return {
        "supporting_evidence": support,
        "contradicting_evidence": against,
        "regime": regime_name or "UNKNOWN",
        "decision_quality": round(decision_quality, 2),
        "evidence_balance": len(support) - len(against),
    }


def analyze_counterfactual(signal, hypothetical_changes=None):
    """Cheap, execution-free counterfactual reasoning for research/autopsy."""
    hypothetical_changes = hypothetical_changes if isinstance(hypothetical_changes, dict) else {}
    base = _finite(signal.get("trade_quality", signal.get("setup_quality", 0.0)))
    conf = _finite(signal.get("confidence", 50.0))
    loc = _finite(signal.get("entry_location_score", 50.0))
    rr = _finite(signal.get("rr", 0.0))
    delta = 0.0
    assumptions = []
    if hypothetical_changes.get("wait_for_retest"):
        delta += 4.0 if signal.get("entry_confirmation") else 2.0
        assumptions.append("entry delayed for retest")
    if hypothetical_changes.get("ignore_sweep_without_reclaim") and signal.get("selected_sweep") and not signal.get("entry_confirmation"):
        delta -= 12.0
        assumptions.append("sweep without reclaim treated as weak")
    if hypothetical_changes.get("boost_location"):
        delta += max(0.0, 55.0 - loc) * 0.10
        assumptions.append("location given more influence")
    if hypothetical_changes.get("penalize_low_rr") and rr < 2.0:
        delta -= (2.0 - rr) * 6.0
        assumptions.append("low RR penalized")
    return {
        "base_quality": round(base, 2),
        "counterfactual_quality": round(max(0.0, min(100.0, base + delta)), 2),
        "quality_delta": round(delta, 2),
        "base_confidence": round(conf, 2),
        "assumptions": assumptions,
    }


def autopsy_trade(record, path_rows=None):
    """Explain a closed trade without rewriting history."""
    record = record if isinstance(record, dict) else {}
    final_r = _finite(record.get("final_r"), _finite(record.get("pnl_usd")))
    mfe = _finite(record.get("mfe_r", record.get("mfe", 0.0)))
    mae = _finite(record.get("mae_r", record.get("mae", 0.0)))
    giveback = _finite(record.get("giveback_ratio", record.get("giveback_pct", 0.0)))
    result = str(record.get("result") or "").lower()
    if final_r > 0:
        outcome_class = "SUCCESS"
    elif mfe > 1.0 and final_r < 0:
        outcome_class = "MANAGEMENT_OR_PROTECTION_FAILURE"
    elif mfe < 0.35 and final_r < 0:
        outcome_class = "ENTRY_OR_THESIS_FAILURE"
    else:
        outcome_class = "AMBIGUOUS_FAILURE"
    reasons = []
    if outcome_class == "SUCCESS":
        reasons.append("successful_outcome")
    if outcome_class == "MANAGEMENT_OR_PROTECTION_FAILURE":
        reasons.append("trade demonstrated favorable excursion but failed to preserve it")
    if giveback >= 0.50:
        reasons.append("large giveback relative to MFE")
    if mae > 0 and mfe > 0 and mae > mfe:
        reasons.append("adverse excursion exceeded favorable excursion")
    if result in {"strategy_error", "data_error", "execution_error"}:
        outcome_class = "EXECUTION_OR_DATA_FAILURE"
    lesson = {
        "trade_uid": record.get("trade_uid"),
        "symbol": record.get("symbol"),
        "outcome_class": outcome_class,
        "final_r": final_r,
        "mfe_r": mfe,
        "mae_r": mae,
        "giveback": giveback,
        "reasons": reasons,
        "confidence": _finite(record.get("confidence"), 50.0),
        "archetype": record.get("archetype"),
        "regime": record.get("market_regime", record.get("regime")),
    }
    _append_lesson(lesson)
    return lesson


def build_research_snapshot(signal, h1=None, m15=None, d1=None, symbol=None):
    """Feature-rich decision snapshot. It never calls Binance/network."""
    signal = signal if isinstance(signal, dict) else {}
    regime = _detect_regime_local(h1, m15, d1)
    arb = _evidence_arbitration(signal, regime)
    snapshot = {
        "schema": FULL_LEARNING_SCHEMA,
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "time_context": extract_time_context(df=m15),
        "symbol": symbol or signal.get("symbol"),
        "strategy_version": ML_COGNITIVE_VERSION,
        "model_version": signal.get("learning_model_version") or (signal.get("learning_prediction") or {}).get("model_version", "static"),
        "decision": signal.get("decision"),
        "confidence": _finite(signal.get("confidence"), 50.0),
        "raw_confidence": _finite(signal.get("direction_confidence", signal.get("confidence")), 50.0),
        "trade_quality": _finite(signal.get("trade_quality", signal.get("setup_quality")), 0.0),
        "archetype": signal.get("archetype", "UNKNOWN"),
        "market": regime,
        "evidence": arb,
        "location": {
            "score": _finite(signal.get("entry_location_score"), 50.0),
            "state": signal.get("entry_location_state"),
            "range_position": _finite(signal.get("entry_range_position"), 0.5),
        },
        "geometry": {
            "entry": _finite(signal.get("entry")),
            "sl": _finite(signal.get("sl")),
            "tp": _finite(signal.get("tp")),
            "rr": _finite(signal.get("rr")),
            "atr": _finite(signal.get("atr")),
            "entry_distance_atr": _finite((signal.get("learning_features") or {}).get("entry_distance_atr")),
            "risk_atr": _finite((signal.get("learning_features") or {}).get("risk_atr")),
        },
        "thesis": signal.get("thesis", {}),
        "uncertainty": _finite(signal.get("uncertainty"), 0.5),
        "counter_thesis": arb.get("contradicting_evidence", []),
    }
    return snapshot




# ============================================================
# COGNITIVE LEARNING LAYER — V28
# ============================================================
_COG_LOCK = threading.RLock()
_COG_STATE_DIR = Path(os.getenv("FULL_STATE_DIR", "machine_learning_state"))
_COG_EXPERIENCE_FILE = _COG_STATE_DIR / "strategy_experience.jsonl"
_COG_LESSON_FILE = _COG_STATE_DIR / "strategy_lessons.jsonl"
_COG_BELIEF_FILE = _COG_STATE_DIR / "strategy_beliefs.json"
_COG_STATE_FILE = _COG_STATE_DIR / "strategy_cognitive_state.json"
_COG_STATE = {
    "schema": FULL_LEARNING_SCHEMA,
    "observations": 0,
    "candidates": 0,
    "labeled_outcomes": 0,
    "autopsies": 0,
    "counterfactuals": 0,
    "belief_revisions": 0,
    "research_cycles": 0,
    "market_clusters": [],
    "last_cycle": None,
    "drift": {"status": "UNKNOWN", "score": 0.0},
}
_COG_BELIEFS = {"schema": FULL_LEARNING_SCHEMA, "beliefs": {}}
_COG_EXPERIENCE_BUFFER = []
_COG_LESSON_BUFFER = []

# Autonomous FULL cognitive worker. It NEVER calls Binance/network; it only
# processes locally buffered experiences, builds research hypotheses, and
# revises cognitive state. Trading/execution remains owned by main.py.
_FULL_COG_THREAD = None
_FULL_COG_STOP = threading.Event()
_FULL_COG_WAKE = threading.Event()
_FULL_COG_LOCK = threading.RLock()
_FULL_COG_INTERVAL_SEC = 5.0
_FULL_COG_LAST_SIGNATURE = None
_FULL_COG_LAST_ERROR = None
_FULL_COG_TICKS = 0
_FULL_COG_LAST_RESULT = {}


def _cjson_load(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"[COG] load gagal {path.name}: {exc}")
    return default


def _cjson_save(path, payload):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        log.warning(f"[COG] save gagal {path.name}: {exc}")


def _full_cognitive_signature():
    with _COG_LOCK:
        return (
            int(_COG_STATE.get("observations", 0) or 0),
            int(_COG_STATE.get("candidates", 0) or 0),
            int(_COG_STATE.get("labeled_outcomes", 0) or 0),
            int(_COG_STATE.get("autopsies", 0) or 0),
            int(_COG_STATE.get("counterfactuals", 0) or 0),
            int(_COG_STATE.get("belief_revisions", 0) or 0),
        )


def _full_cognitive_tick():
    """Run one local-only FULL learning/research cycle.

    It deliberately does not change live trading parameters. Its purpose is
    to continuously process observations/outcomes, find contradictions and
    lessons, and maintain a research state that can later be consumed by the
    model-training layer.
    """
    global _FULL_COG_TICKS, _FULL_COG_LAST_RESULT, _FULL_COG_LAST_ERROR
    with _COG_LOCK:
        obs = list(_COG_EXPERIENCE_BUFFER)
    outcomes = [r for r in obs if isinstance(r, dict) and r.get("type") == "trade_outcome"]
    candidates = [r for r in obs if isinstance(r, dict) and r.get("type") == "candidate"]
    market_obs = [r for r in obs if isinstance(r, dict) and r.get("type") == "market_observation"]
    # Use recent local evidence; no network and bounded work.
    recent = obs[-1000:]
    research = research_brain_cycle(
        observations=recent,
        outcomes=outcomes[-200:],
        current_signal=(candidates[-1] if candidates else None),
    )
    # Learn from the distribution of actual outcomes without turning that
    # observation into a hard threshold change. This keeps FULL adaptive rather
    # than chronically tightening the live gate.
    if outcomes:
        rs = []
        for row in outcomes[-200:]:
            payload = row.get("outcome") if isinstance(row.get("outcome"), dict) else row
            rs.append(_finite_num(payload.get("final_r"), _finite_num(payload.get("r"), 0.0)))
        positive_rate = sum(1 for x in rs if x > 0) / max(1, len(rs))
        revise_belief(
            "positive_outcome_rate", positive_rate, condition="rolling",
            evidence_strength=min(0.90, len(rs) / 200.0),
            source="binance_trade", reason="continuous FULL outcome review",
            sample_increment=max(1, len(rs)),
        )
    with _COG_LOCK:
        _COG_STATE["observations"] = max(int(_COG_STATE.get("observations", 0) or 0), len(market_obs))
        _COG_STATE["candidates"] = max(int(_COG_STATE.get("candidates", 0) or 0), len(candidates))
        _COG_STATE["labeled_outcomes"] = max(int(_COG_STATE.get("labeled_outcomes", 0) or 0), len(outcomes))
        _COG_STATE["last_cycle"] = time.time()
        _COG_STATE["full_worker_ticks"] = int(_COG_STATE.get("full_worker_ticks", 0) or 0) + 1
        _COG_STATE["last_worker_mode"] = "ACTIVE"
        _cjson_save(_COG_STATE_FILE, _COG_STATE)
    _FULL_COG_TICKS += 1
    _FULL_COG_LAST_ERROR = None
    _FULL_COG_LAST_RESULT = research
    return research


def _full_cognitive_loop():
    global _FULL_COG_LAST_ERROR, _FULL_COG_LAST_RESULT
    while not _FULL_COG_STOP.is_set():
        try:
            if _full_cognitive_signature() != _FULL_COG_LAST_SIGNATURE:
                _FULL_COG_LAST_RESULT = _full_cognitive_tick()
                globals()["_FULL_COG_LAST_SIGNATURE"] = _full_cognitive_signature()
        except Exception as exc:
            _FULL_COG_LAST_ERROR = str(exc)[:500]
            log.exception("[FULL COG] learning tick gagal")
        _FULL_COG_WAKE.wait(_FULL_COG_INTERVAL_SEC)
        _FULL_COG_WAKE.clear()


def _start_full_cognitive_worker():
    global _FULL_COG_THREAD
    with _FULL_COG_LOCK:
        if _FULL_COG_THREAD is None or not _FULL_COG_THREAD.is_alive():
            _FULL_COG_STOP.clear()
            _FULL_COG_WAKE.set()
            _FULL_COG_THREAD = threading.Thread(
                target=_full_cognitive_loop, name="full-cognitive", daemon=True
            )
            _FULL_COG_THREAD.start()
        else:
            _FULL_COG_WAKE.set()


def _stop_full_cognitive_worker():
    _FULL_COG_STOP.set()
    _FULL_COG_WAKE.set()


def get_full_cognitive_status():
    """Authoritative FULL status used by main.py. Includes the real V32 and adaptive workers."""
    try:
        v32 = get_v32_status()
    except Exception as exc:
        v32 = {"worker_alive": False, "ticks": 0, "last_error": str(exc)[:300], "state": {}}
    try:
        adaptive = get_adaptive_status()
    except Exception as exc:
        adaptive = {"worker_alive": False, "last_error": str(exc)[:300], "strategy_revisions": 0, "strategy_version": "S1"}
    state = dict(v32.get("state") or {})
    state["adaptive"] = adaptive
    return {
        "worker_alive": bool(v32.get("worker_alive")),
        "worker_ticks": int(v32.get("ticks", 0) or 0),
        "last_cycle": state.get("last_review"),
        "last_error": v32.get("last_error") or adaptive.get("last_error"),
        "last_result": {},
        "full_enabled": bool(_FULL_ENABLED),
        "v32": v32,
        "adaptive": adaptive,
        "strategy_version": adaptive.get("strategy_version", "S1"),
        "strategy_revisions": int(adaptive.get("strategy_revisions", 0) or 0),
    }


def _load_cognitive_state():
    global _COG_STATE, _COG_BELIEFS
    with _COG_LOCK:
        state = _cjson_load(_COG_STATE_FILE, {})
        if isinstance(state, dict):
            _COG_STATE.update(state)
        beliefs = _cjson_load(_COG_BELIEF_FILE, {})
        if isinstance(beliefs, dict):
            _COG_BELIEFS.update(beliefs)


def _append_jsonl(path, record, memory, max_mem=5000):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=str)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        memory.append(record)
        if len(memory) > max_mem:
            del memory[:-max_mem]
    except Exception as exc:
        log.warning(f"[COG] jsonl append gagal {path.name}: {exc}")


def _finite_num(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _bounded(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, _finite_num(v, lo)))


def _safe_mean(values, default=0.0):
    vals = [_finite_num(v, np.nan) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else default


def _last_row_features(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    r = df.iloc[-1]
    out = {}
    for k, v in r.to_dict().items():
        if isinstance(v, (int, float, np.number, bool)):
            out[str(k)] = _finite_num(v)
    return out



# =============================================================================
# TIME-OF-DAY / SESSION CONTEXT
# =============================================================================
FULL_TIMEZONE_NAME = os.getenv("FULL_TIMEZONE", "Asia/Jakarta")

def _full_timezone():
    if ZoneInfo is not None:
        try:
            return ZoneInfo(FULL_TIMEZONE_NAME)
        except Exception:
            pass
    return timezone(timedelta(hours=7))


def _timestamp_from_df(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    try:
        if "timestamp" in df.columns:
            raw = df["timestamp"].iloc[-1]
        else:
            raw = df.index[-1]
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime().astimezone(_full_timezone())
    except Exception:
        return None


def _time_session(hour):
    h = int(hour) % 24
    if 0 <= h < 6:
        return "NIGHT_EARLY"
    if 6 <= h < 12:
        return "MORNING"
    if 12 <= h < 18:
        return "AFTERNOON"
    return "EVENING_NIGHT"


def extract_time_context(timestamp=None, df=None):
    """Time is an evidence feature, never a hard trading rule."""
    dt = None
    if timestamp is not None:
        try:
            ts = pd.Timestamp(timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            dt = ts.to_pydatetime().astimezone(_full_timezone())
        except Exception:
            dt = None
    if dt is None and df is not None:
        dt = _timestamp_from_df(df)
    if dt is None:
        return {
            "available": False,
            "timezone": FULL_TIMEZONE_NAME,
            "hour_local": None,
            "minute_local": None,
            "day_of_week": None,
            "session": "UNKNOWN",
            "hour_sin": 0.0,
            "hour_cos": 0.0,
            "is_weekend": False,
        }
    hour_float = dt.hour + dt.minute / 60.0
    angle = 2.0 * np.pi * hour_float / 24.0
    return {
        "available": True,
        "timezone": FULL_TIMEZONE_NAME,
        "hour_local": int(dt.hour),
        "minute_local": int(dt.minute),
        "day_of_week": int(dt.weekday()),
        "session": _time_session(dt.hour),
        "hour_sin": float(np.sin(angle)),
        "hour_cos": float(np.cos(angle)),
        "is_weekend": bool(dt.weekday() >= 5),
    }


def _time_effect_from_outcomes(outcomes):
    """Estimate whether time-of-day explains outcomes without forcing it into live decisions."""
    rows = []
    for r in outcomes or []:
        if not isinstance(r, dict):
            continue
        try:
            fr = float(r.get("final_r"))
        except (TypeError, ValueError):
            continue
        tc = r.get("time_context") or {}
        if not isinstance(tc, dict) or not tc.get("available"):
            tc = extract_time_context(r.get("entry_timestamp") or r.get("timestamp"))
        if not tc.get("available"):
            continue
        rows.append((fr, tc))
    if len(rows) < 10:
        return {"status": "INSUFFICIENT", "sample_size": len(rows), "overall_mean_r": 0.0, "effects": [], "strongest_effect": 0.0, "usable": False}
    rs = np.asarray([x[0] for x in rows], dtype=float)
    overall = float(np.mean(rs))
    groups = {}
    for fr, tc in rows:
        groups.setdefault(str(tc.get("session", "UNKNOWN")), []).append(fr)
    effects = []
    for session, vals in groups.items():
        n = len(vals)
        if n < 5:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        diff = mean - overall
        sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if sd > 0 else 0.0
        t = diff / se if se > 1e-12 else (999.0 if abs(diff) > 1e-12 else 0.0)
        # Shrink small groups toward zero; never turn this into a hard gate.
        shrink = min(1.0, n / 30.0)
        effect = float(np.tanh(diff) * shrink)
        effects.append({
            "session": session,
            "sample_size": n,
            "mean_r": round(mean, 5),
            "delta_vs_overall": round(diff, 5),
            "t_like": round(float(t), 3),
            "effect": round(effect, 5),
        })
    effects.sort(key=lambda x: abs(x["effect"]), reverse=True)
    strongest = effects[0]["effect"] if effects else 0.0
    strong = bool(effects and effects[0]["sample_size"] >= 20 and abs(effects[0]["t_like"]) >= 2.0)
    status = "SUPPORTED" if strong else ("WEAK_SIGNAL" if effects else "INSUFFICIENT")
    return {
        "status": status,
        "sample_size": len(rows),
        "overall_mean_r": round(overall, 5),
        "effects": effects,
        "strongest_effect": round(float(strongest), 5),
        "usable": strong,
        "timezone": FULL_TIMEZONE_NAME,
    }


def extract_market_features(df_h1, df_m15, df_d1=None, df_btc_h1=None):
    """Point-in-time numeric market representation. No network access."""
    raw_time_df = df_m15 if isinstance(df_m15, pd.DataFrame) else df_h1
    h1 = build_df(df_h1, 60)
    m15 = build_df(df_m15, 15)
    d1 = build_df(df_d1, 1440) if isinstance(df_d1, pd.DataFrame) else None
    btc = build_df(df_btc_h1, 60) if isinstance(df_btc_h1, pd.DataFrame) else None
    def ret(df, bars):
        if df is None or len(df) <= bars:
            return 0.0
        a = _finite_num(df["close"].iloc[-bars-1])
        b = _finite_num(df["close"].iloc[-1])
        return (b / a - 1.0) if a else 0.0
    def atr_pct(df):
        if df is None or df.empty:
            return 0.0
        atr = _finite_num(df["atr"].iloc[-1] if "atr" in df.columns else 0.0)
        px = _finite_num(df["close"].iloc[-1])
        return atr / px if px else 0.0
    mf = {
        "m15_ret_1": ret(m15, 1), "m15_ret_4": ret(m15, 4), "m15_ret_16": ret(m15, 16),
        "h1_ret_1": ret(h1, 1), "h1_ret_4": ret(h1, 4), "h1_ret_24": ret(h1, 24),
        "d1_ret_1": ret(d1, 1), "d1_ret_5": ret(d1, 5),
        "m15_atr_pct": atr_pct(m15), "h1_atr_pct": atr_pct(h1), "d1_atr_pct": atr_pct(d1),
        "m15_rv": _relative_volume(m15) if m15 is not None else 0.0,
        "h1_rv": _relative_volume(h1) if h1 is not None else 0.0,
        "btc_h1_ret_1": ret(btc, 1), "btc_h1_ret_4": ret(btc, 4),
    }
    if h1 is not None and not h1.empty:
        sh, sl = swing_pts(h1, 5)
        mf["h1_structure_numeric"] = {"bullish": 1.0, "bearish": -1.0}.get(_market_structure(h1, sh, sl), 0.0)
    else:
        mf["h1_structure_numeric"] = 0.0
    if m15 is not None and not m15.empty:
        sh, sl = swing_pts(m15, 5)
        mf["m15_structure_numeric"] = {"bullish": 1.0, "bearish": -1.0}.get(_market_structure(m15, sh, sl), 0.0)
    else:
        mf["m15_structure_numeric"] = 0.0
    tc = extract_time_context(df=raw_time_df)
    if tc.get("available"):
        mf.update({
            "time_hour_local": float(tc["hour_local"]),
            "time_day_of_week": float(tc["day_of_week"]),
            "time_hour_sin": float(tc["hour_sin"]),
            "time_hour_cos": float(tc["hour_cos"]),
            "time_is_weekend": float(1.0 if tc["is_weekend"] else 0.0),
        })
    return {k: float(v) for k, v in mf.items()}


def record_market_observation(symbol, features, source="binance", timestamp=None):
    record = {
        "type": "market_observation",
        "timestamp": timestamp or time.time(),
        "symbol": symbol,
        "source": source,
        "features": dict(features or {}),
        "time_context": (features or {}).get("time_context") if isinstance(features, dict) else None,
    }
    with _COG_LOCK:
        _append_jsonl(_COG_EXPERIENCE_FILE, record, _COG_EXPERIENCE_BUFFER, 10000)
        _COG_STATE["observations"] = int(_COG_STATE.get("observations", 0)) + 1
    return record


def record_candidate_observation(signal, outcome=None, rejected_reason=None, source="binance"):
    sig = dict(signal or {}) if isinstance(signal, dict) else {}
    snap = sig.get("research_snapshot") or {}
    record = {
        "type": "candidate",
        "timestamp": time.time(),
        "symbol": sig.get("symbol"),
        "source": source,
        "decision": sig.get("decision"),
        "confidence": _finite_num(sig.get("confidence"), 50.0),
        "quality": _finite_num(sig.get("trade_quality", sig.get("setup_quality")), 0.0),
        "archetype": sig.get("archetype", "UNKNOWN"),
        "regime": sig.get("market_regime") or (snap.get("market") or {}).get("regime"),
        "snapshot": snap,
        "rejected_reason": rejected_reason,
        "outcome": outcome,
        "time_context": sig.get("time_context") or snap.get("time_context"),
    }
    with _COG_LOCK:
        _append_jsonl(_COG_EXPERIENCE_FILE, record, _COG_EXPERIENCE_BUFFER, 10000)
        _COG_STATE["candidates"] = int(_COG_STATE.get("candidates", 0)) + 1
    return record


def record_trade_outcome(signal, outcome, source="binance"):
    """Canonical cognitive outcome writer; idempotent by trade_uid/order_id."""
    sig = dict(signal or {}) if isinstance(signal, dict) else {}
    payload = outcome if isinstance(outcome, dict) else {"result": outcome}
    uid = str(sig.get("trade_uid") or sig.get("order_id") or f"{sig.get('symbol','')}|{sig.get('entry_time','')}|{sig.get('exit_time','')}|{payload.get('result','')}")
    key=f"{uid}|{source}"
    with _COG_LOCK:
        seen = globals().setdefault("_COG_SEEN_OUTCOME_KEYS", set())
        if key in seen:
            return None
        seen.add(key)
        snap = sig.get("research_snapshot") if isinstance(sig.get("research_snapshot"), dict) else {}
        record = {"type":"trade_outcome","timestamp":time.time(),"trade_uid":uid,"symbol":sig.get("symbol"),"source":source,"decision":sig.get("decision"),"confidence":_finite_num(sig.get("confidence"),50),"quality":_finite_num(sig.get("trade_quality",sig.get("setup_quality")),0),"archetype":sig.get("archetype","UNKNOWN"),"regime":sig.get("market_regime") or (snap.get("market") or {}).get("regime"),"snapshot":snap,"time_context":sig.get("time_context") or snap.get("time_context"),"learning_features":dict(sig.get("learning_features") or {}),"signal":sig,"outcome":dict(payload)}
        _append_jsonl(_COG_EXPERIENCE_FILE, record, _COG_EXPERIENCE_BUFFER, 10000)
        _COG_STATE["labeled_outcomes"] = int(_COG_STATE.get("labeled_outcomes", 0)) + 1
    return record


def _evidence_quality(source, sample_size=0):
    base = {"binance_trade": 1.0, "binance_market": 0.85, "external": 0.55, "domain": 0.35}.get(source, 0.5)
    return float(base * min(1.0, 0.35 + 0.65 * np.sqrt(max(sample_size, 1) / 100.0)))


def evidence_discussion(evidence_items):
    """Make heterogeneous evidence explicit: support, contradiction, uncertainty."""
    rows = []
    for item in evidence_items or []:
        if not isinstance(item, dict):
            continue
        val = str(item.get("stance", "neutral")).lower()
        if val not in {"support", "contradict", "neutral"}:
            val = "neutral"
        strength = _bounded(item.get("strength", 0.5))
        reliability = _bounded(item.get("reliability", _evidence_quality(item.get("source", "external"), item.get("sample_size", 0))))
        rows.append({**item, "stance": val, "weighted_strength": strength * reliability})
    support = sum(x["weighted_strength"] for x in rows if x["stance"] == "support")
    contradict = sum(x["weighted_strength"] for x in rows if x["stance"] == "contradict")
    neutral = sum(x["weighted_strength"] for x in rows if x["stance"] == "neutral")
    total = support + contradict + neutral
    balance = (support - contradict) / max(total, 1e-9)
    return {
        "support": round(support, 5),
        "contradict": round(contradict, 5),
        "neutral": round(neutral, 5),
        "balance": round(balance, 5),
        "uncertainty": round(1.0 - min(1.0, abs(balance)), 5),
        "items": rows,
    }


def _trade_path(record):
    mfe = _finite_num(record.get("mfe_r"), 0.0)
    mae = _finite_num(record.get("mae_r"), 0.0)
    final_r = _finite_num(record.get("final_r"), 0.0)
    giveback = _finite_num(record.get("giveback_r"), max(0.0, mfe - final_r))
    return {
        "mfe_r": mfe,
        "mae_r": mae,
        "final_r": final_r,
        "giveback_r": max(0.0, giveback),
        "giveback_ratio": max(0.0, giveback) / max(mfe, 1e-9) if mfe > 0 else 0.0,
        "capture_ratio": final_r / mfe if mfe > 1e-9 else 0.0,
    }




def counterfactual_trade(record, changes=None):
    """Evaluate a small alternative hypothesis without using future data in features."""
    rec = dict(record or {})
    ch = dict(changes or {})
    path = _trade_path(rec)
    final_r = path["final_r"]
    simulated = final_r
    notes = []
    if ch.get("delay_entry_bars"):
        delay = max(0, int(ch.get("delay_entry_bars")))
        adverse = _finite_num(rec.get("mae_r"), 0.0)
        simulated = final_r - 0.10 * min(delay, 5) * abs(adverse)
        notes.append(f"entry_delay_{delay}")
    if ch.get("require_reclaim") and not ((rec.get("research_snapshot") or {}).get("evidence") or {}).get("reclaim"):
        simulated = min(simulated, 0.0)
        notes.append("missing_reclaim_penalty")
    if ch.get("trail_tighten_on_giveback"):
        gb = path["giveback_ratio"]
        if gb >= 0.50 and path["mfe_r"] > 1.0:
            simulated = max(simulated, 0.25 * path["mfe_r"])
            notes.append("hypothetical_giveback_protection")
    out = {
        "type": "counterfactual",
        "timestamp": time.time(),
        "symbol": rec.get("symbol"),
        "changes": ch,
        "actual_final_r": final_r,
        "simulated_final_r": float(simulated),
        "delta_r": float(simulated - final_r),
        "notes": notes,
        "causal_confidence": 0.25,
    }
    with _COG_LOCK:
        _COG_STATE["counterfactuals"] = int(_COG_STATE.get("counterfactuals", 0)) + 1
    return out


def get_belief_state(name, condition="global", default_value=None):
    key = f"{name}|{condition}"
    with _COG_LOCK:
        b = (_COG_BELIEFS.get("beliefs") or {}).get(key)
        if isinstance(b, dict):
            return dict(b)
    return {
        "name": name, "condition": condition, "value": default_value,
        "strength": 0.20, "evidence_for": 0, "evidence_against": 0,
        "last_revision": None,
    }




def _zscore_matrix(rows):
    if not rows:
        return np.empty((0, 0))
    keys = sorted({k for r in rows for k in r if isinstance(r.get(k), (int, float, np.number, bool))})
    if not keys:
        return np.empty((len(rows), 0))
    X = np.asarray([[ _finite_num(r.get(k), 0.0) for k in keys] for r in rows], dtype=float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, keys


def discover_market_clusters(observations, k=3, max_iter=15):
    """Lightweight k-means using only numpy; exploratory, not a live trade gate."""
    rows = [r.get("features", {}) for r in observations or [] if isinstance(r, dict)]
    rows = rows[-3000:]
    z = _zscore_matrix(rows)
    if isinstance(z, tuple):
        X, keys = z
    else:
        X, keys = z, []
    if X.size == 0 or len(X) < max(2, k):
        return []
    k = max(2, min(int(k), len(X)))
    seeds = np.linspace(0, len(X) - 1, k, dtype=int)
    centers = X[seeds].copy()
    for _ in range(max_iter):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist, axis=1)
        new_centers = []
        for i in range(k):
            subset = X[labels == i]
            new_centers.append(np.mean(subset, axis=0) if len(subset) else centers[i])
        new_centers = np.asarray(new_centers)
        if np.max(np.abs(new_centers - centers)) < 1e-5:
            centers = new_centers
            break
        centers = new_centers
    out = []
    for i in range(k):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        centroid = centers[i]
        ranked = np.argsort(np.abs(centroid))[::-1][:min(6, len(keys))]
        signature = [{"feature": keys[j], "z": round(float(centroid[j]), 3)} for j in ranked]
        out.append({"cluster": i, "samples": int(len(idx)), "signature": signature})
    with _COG_LOCK:
        _COG_STATE["market_clusters"] = out
    return out


def detect_feature_drift(reference_rows, recent_rows):
    ref = [r.get("features", {}) for r in reference_rows or [] if isinstance(r, dict)]
    rec = [r.get("features", {}) for r in recent_rows or [] if isinstance(r, dict)]
    if not ref or not rec:
        return {"status": "UNKNOWN", "score": 0.0, "features": []}
    keys = sorted(set().union(*(r.keys() for r in ref + rec)))
    diffs = []
    for key in keys:
        a = [_finite_num(r.get(key), 0.0) for r in ref]
        b = [_finite_num(r.get(key), 0.0) for r in rec]
        sa = np.std(a) or 1.0
        d = abs(float(np.mean(b) - np.mean(a))) / sa
        diffs.append((d, key))
    diffs.sort(reverse=True)
    score = _bounded(np.mean([min(d, 3.0) / 3.0 for d, _ in diffs]) if diffs else 0.0)
    status = "HIGH" if score >= 0.45 else "MEDIUM" if score >= 0.25 else "LOW"
    result = {"status": status, "score": round(float(score), 4), "features": [{"feature": k, "z_shift": round(float(d), 3)} for d, k in diffs[:10]]}
    with _COG_LOCK:
        _COG_STATE["drift"] = result
    return result


def build_evidence_discussion(signal, historical=None, lessons=None):
    sig = signal if isinstance(signal, dict) else {}
    items = []
    q = _finite_num(sig.get("trade_quality", sig.get("setup_quality")), 0.0)
    items.append({"source": "binance_market", "stance": "support" if q >= 70 else "neutral", "strength": q / 100.0, "sample_size": 1, "label": "current analytical quality"})
    hist = historical or {}
    if hist.get("expected_r") is not None:
        ex = _finite_num(hist.get("expected_r"))
        items.append({"source": "external", "stance": "support" if ex > 0 else "contradict", "strength": min(1.0, abs(ex) / 1.0), "sample_size": int(hist.get("sample", 0) or 0), "label": "historical conditional expectancy"})
    lesson_rows = lessons or []
    with _COG_LOCK:
        time_effect = dict(_COG_STATE.get("time_of_day") or {})
    if time_effect.get("usable"):
        strongest = time_effect.get("effects", [])[0] if time_effect.get("effects") else {}
        items.append({
            "source": "binance_trade",
            "stance": "support" if _finite_num(strongest.get("effect"), 0.0) > 0 else "contradict",
            "strength": abs(_finite_num(strongest.get("effect"), 0.0)),
            "sample_size": int(time_effect.get("sample_size", 0) or 0),
            "label": f"time-of-day effect: {strongest.get('session', 'UNKNOWN')}",
        })
    if lesson_rows:
        failure_rate = _safe_mean([1.0 if "FAILURE" in str(x.get("reasons")) else 0.0 for x in lesson_rows], 0.0)
        items.append({"source": "binance_trade", "stance": "contradict" if failure_rate > 0.5 else "neutral", "strength": failure_rate, "sample_size": len(lesson_rows), "label": "recent failure evidence"})
    return evidence_discussion(items)


def research_brain_cycle(observations=None, outcomes=None, current_signal=None):
    """One bounded research cycle. It does not call any network/API."""
    observations = observations or []
    outcomes = outcomes or []
    recent = observations[-250:] if len(observations) > 250 else observations
    reference = observations[-1500:-250] if len(observations) > 500 else observations
    clusters = discover_market_clusters(recent, k=3) if len(recent) >= 20 else []
    drift = detect_feature_drift(reference, recent) if reference and recent else {"status": "UNKNOWN", "score": 0.0, "features": []}
    lessons = []
    for row in outcomes[-100:]:
        if isinstance(row, dict) and row.get("type") == "trade_outcome":
            lessons.append(autopsy_trade(row))
    discussion = build_evidence_discussion(current_signal, lessons=lessons) if current_signal else {}
    time_effect = _time_effect_from_outcomes(outcomes)
    with _COG_LOCK:
        _COG_STATE["time_of_day"] = time_effect
    with _COG_LOCK:
        _COG_STATE["research_cycles"] = int(_COG_STATE.get("research_cycles", 0)) + 1
        _COG_STATE["last_cycle"] = time.time()
        _cjson_save(_COG_STATE_FILE, _COG_STATE)
    return {
        "schema": FULL_LEARNING_SCHEMA,
        "version": ML_COGNITIVE_VERSION,
        "clusters": clusters,
        "drift": drift,
        "evidence_discussion": discussion,
        "state": dict(_COG_STATE),
    }


def get_cognitive_status():
    with _COG_LOCK:
        return {
            "version": ML_COGNITIVE_VERSION,
            "schema": FULL_LEARNING_SCHEMA,
            "state": dict(_COG_STATE),
            "belief_count": len(_COG_BELIEFS.get("beliefs") or {}),
            "experience_buffer": len(_COG_EXPERIENCE_BUFFER),
            "lesson_buffer": len(_COG_LESSON_BUFFER),
        }


def get_learning_schema():
    return {
        "schema": MACHINE_LEARNING_SCHEMA,
        "feature_names": list(ML_FEATURE_NAMES),
        "count": len(ML_FEATURE_NAMES),
        "point_in_time": True,
        "network_access": False,
    }



def ingest_historical_ohlcv(path, source="external", symbol=None, interval_minutes=15, max_rows=50000):
    """Ingest local historical OHLCV only; never performs a network request."""
    fp = Path(path)
    if not fp.exists() or not fp.is_file():
        raise FileNotFoundError(str(fp))
    if fp.suffix.lower() == ".csv":
        df = pd.read_csv(fp)
    elif fp.suffix.lower() in {".json", ".jsonl"}:
        if fp.suffix.lower() == ".jsonl":
            df = pd.DataFrame([json.loads(line) for line in fp.read_text(encoding="utf-8").splitlines() if line.strip()])
        else:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            df = pd.DataFrame(obj)
    else:
        raise ValueError("historical file must be CSV, JSON, or JSONL")
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"timestamp":"timestamp", "time":"timestamp", "datetime":"timestamp", "date":"timestamp"}
    if "timestamp" not in df.columns:
        for old, new in aliases.items():
            if old in df.columns:
                df = df.rename(columns={old:new}); break
    required = {"open","high","low","close","volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = ts
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    else:
        df = df.sort_index()
    df = df.dropna(subset=list(required))
    df = df[(df["high"] >= df[["open","close"]].max(axis=1)) & (df["low"] <= df[["open","close"]].min(axis=1))]
    df = df.drop_duplicates(subset=["timestamp"] if "timestamp" in df.columns else None, keep="last")
    if max_rows and len(df) > int(max_rows):
        df = df.tail(int(max_rows)).copy()
    features = extract_market_features(df, df, df)
    rec = {
        "type": "historical_dataset",
        "source": source,
        "symbol": symbol,
        "interval_minutes": int(interval_minutes),
        "rows": int(len(df)),
        "start": df["timestamp"].iloc[0].isoformat() if "timestamp" in df.columns and len(df) else None,
        "end": df["timestamp"].iloc[-1].isoformat() if "timestamp" in df.columns and len(df) else None,
        "summary_features": features,
    }
    with _COG_LOCK:
        _append_jsonl(_COG_EXPERIENCE_FILE, rec, _COG_EXPERIENCE_BUFFER, 10000)
    return {"data": df, "record": rec}



_load_cognitive_state()
_load_active_learning_model()

# Preserve the battle-tested V26 public implementations and decorate their output
# rather than replacing their core detector math.
_CORE_FULL_ANALYZE = _core_full_analyze
_CORE_MANAGE_POSITION = _core_manage_position
_BASE_FULL_ANALYZE_V26 = _CORE_FULL_ANALYZE
_BASE_MANAGE_POSITION_V26 = _CORE_MANAGE_POSITION






def compare_candidates(candidates):
    """Global candidate comparison with quality, uncertainty, and contradiction penalties."""
    rows = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        q = _finite(c.get("trade_quality", c.get("setup_quality")), 0.0)
        conf = _finite(c.get("confidence"), 50.0)
        unc = _finite(c.get("uncertainty"), 50.0)
        rr = min(_finite(c.get("rr")), 6.0)
        contradiction = _finite((c.get("thesis") or {}).get("contradiction_score"), 0.0)
        dataq = _finite((c.get("research_snapshot") or {}).get("data_quality", 1.0), 1.0)
        score = q * 0.62 + conf * 0.16 + rr * 2.2 + dataq * 4.0 - contradiction * 0.55 - unc * 0.08
        rows.append((score, c))
    rows.sort(key=lambda x: x[0], reverse=True)
    for rank, (score, c) in enumerate(rows, 1):
        c["global_rank"] = rank
        c["global_candidate_score"] = round(float(score), 3)
    return [c for _, c in rows]






def _format_full_payload(payload):
    if not isinstance(payload, dict):
        return html.escape(str(payload))
    mode = "ON" if payload.get("mode") else "OFF"
    champ = payload.get("champion") or {}
    model = champ.get("model_version", "Belum ada") if isinstance(champ, dict) else "Belum ada"
    threshold = champ.get("confidence_min", payload.get("manual_threshold", "—")) if isinstance(champ, dict) else payload.get("manual_threshold", "—")
    worker = payload.get("cognitive_worker") or {}
    worker_state = "ACTIVE" if worker.get("worker_alive") else "OFF"
    ticks = int(worker.get("worker_ticks", 0) or 0)
    return (
        f"Status: <b>{mode}</b>\n"
        f"Champion: <code>{html.escape(str(model))}</code>\n"
        f"Confidence min: <b>{html.escape(str(threshold))}%</b>\n"
        f"Experience: <b>{int(payload.get('experience_samples', 0) or 0)}</b>\n"
        f"Learning cycles: <b>{int(payload.get('learning_cycles', 0) or 0)}</b>\n"
        f"Promotions: <b>{int(payload.get('promotion_count', 0) or 0)}</b>\n"
        f"Beliefs: <b>{len(_FULL_BELIEFS.get('beliefs') or {})}</b>\n"
        f"Belief revisions: <b>{int(_FULL_BELIEFS.get('revisions', 0) or 0)}</b>\n"
        f"Cognitive worker: <b>{worker_state}</b>\n"
        f"Research ticks: <b>{ticks}</b>"
    )


# =============================================================================
# LIVE EXPERIENCE BRIDGE
# Main.py may call these hooks after each scan/close. They never access Binance.
# =============================================================================
_COG_SEEN_OUTCOME_KEYS = set()





# Keep exported public API explicit and stable for main.py /ganti validation.
__all__ = [
    "full_analyze", "manage_position", "get_best_signal", "score_direction", "swing_pts",
    "set_learning_model", "get_learning_model_info", "full_command",
    "build_research_snapshot", "autopsy_trade", "analyze_counterfactual",
    "revise_belief", "get_belief", "compare_candidates", "ML_FEATURE_NAMES",
    "extract_market_features", "record_market_observation", "record_candidate_observation",
    "record_trade_outcome", "evidence_discussion", "get_belief_state", "revise_belief",
    "discover_market_clusters", "detect_feature_drift", "research_brain_cycle", "extract_time_context", "_time_effect_from_outcomes",
    "get_full_cognitive_status", "ingest_historical_ohlcv", "full_learning_review",
    "get_cognitive_status", "get_learning_schema", "reset_cognitive_memory", "ingest_historical_ohlcv",
    "ingest_live_candidate", "ingest_live_outcome", "full_learning_review",
    "TRAIL_R_LADDER", "TRAIL_ENGINE_VERSION", "MIN_RR", "MAX_RR",
]

# =============================================================================
# V32 COGNITIVE RESEARCH ENGINE
# =============================================================================
# This section intentionally lives after the older cognitive layer so the exported
# functions below become the single public behavior. It is research-only: it does
# not send orders, call Binance, or forcibly tighten live signal frequency.

V32_VERSION = "V38_EVENT_CONTRACT_HARDENED_BRAIN"
V32_SCHEMA = "full_learning_cognitive_v2"
V32_STATE_DIR = Path(os.getenv("FULL_STATE_DIR", "machine_learning_state"))
V32_STATE_FILE = V32_STATE_DIR / "v32_brain_state.json"
V32_EXPERIENCE_FILE = V32_STATE_DIR / "v32_experience.jsonl"
V32_LESSON_FILE = V32_STATE_DIR / "v32_lessons.jsonl"
V32_POLICY_FILE = V32_STATE_DIR / "v32_policy.json"
V32_TIMEZONE = os.getenv("FULL_TIMEZONE", "Asia/Jakarta")

V32_MIN_OUTCOMES = 8
V32_MIN_WIN = 3
V32_MIN_LOSS = 3
V32_MIN_CELL = 8
V32_RESEARCH_WINDOW = 2500
V32_REVIEW_INTERVAL = max(5.0, float(os.getenv("FULL_REVIEW_INTERVAL", "30")))
V32_SINGLE_EVENT_MAX_BELIEF_DELTA = 0.035
V32_POLICY_MIN_STABILITY_REVIEWS = 3
V32_EXPLORATION_SHARE = 0.15

try:
    V32_TZ = ZoneInfo(V32_TIMEZONE) if ZoneInfo else timezone(timedelta(hours=7))
except Exception:
    V32_TZ = timezone(timedelta(hours=7))

_V32_LOCK = threading.RLock()
_V32_STOP = threading.Event()
_V32_WAKE = threading.Event()
_V32_THREAD = None
_V32_TICKS = 0
_V32_LAST_ERROR = None
_V32_NOTIFY = None
_V32_LAST_TELEGRAM_NOTIFY = 0.0
_V32_TELEGRAM_MIN_INTERVAL = max(30.0, float(os.getenv("FULL_TELEGRAM_MIN_INTERVAL", "90")))

_V32_STATE = {
    "schema": V32_SCHEMA,
    "version": V32_VERSION,
    "observations": 0,
    "candidates": 0,
    "outcomes": 0,
    "wins": 0,
    "losses": 0,
    "autopsies": 0,
    "counterfactuals": 0,
    "belief_revisions": 0,
    "research_questions": 0,
    "resolved_questions": 0,
    "model_candidates": 0,
    "model_promotions": 0,
    "drift_score": 0.0,
    "drift_status": "UNKNOWN",
    "time_effect": {"status": "INSUFFICIENT", "usable": False},
    "coverage": {"candidate_count": 0, "live_candidate_count": 0, "shadow_count": 0, "coverage_rate": 0.0},
    "calibration": {"status": "INSUFFICIENT", "buckets": []},
    "last_review": None,
    "last_model_update": None,
    "last_model_sample_count": 0,
    "last_reviewed_outcomes": 0,
    "last_policy_revision": None,
    "policy_revision_count": 0,
}
_V32_BELIEFS = {}
_V32_BUFFER = []
_V32_QUESTIONS = []
_V32_LAST_FILE_OFFSET = 0


def _v32_json_load(path, default):
    try:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, type(default)) else default
    except Exception as exc:
        log.warning(f"[V32] load {path.name} gagal: {exc}")
    return default


def _v32_json_save(path, obj):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception as exc:
        log.warning(f"[V32] save {path.name} gagal: {exc}")
        return False


def _v32_load_state():
    global _V32_STATE, _V32_BELIEFS
    with _V32_LOCK:
        st = _v32_json_load(V32_STATE_FILE, {})
        if isinstance(st, dict):
            _V32_STATE.update(st)
        _V32_BELIEFS = _v32_json_load(V32_POLICY_FILE, {}) or {}
        if not isinstance(_V32_BELIEFS, dict):
            _V32_BELIEFS = {}


def _v32_append(path, row, memory, max_mem=8000):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        memory.append(row)
        if len(memory) > max_mem:
            del memory[:-max_mem]
    except Exception as exc:
        log.warning(f"[V32] append gagal: {exc}")


def _v32_f(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _v32_prob(v):
    x = _v32_f(v, 0.5)
    return max(0.0, min(1.0, x))


def _v32_hour_context(timestamp=None):
    dt = None
    try:
        if timestamp is not None:
            ts = pd.Timestamp(timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            dt = ts.to_pydatetime().astimezone(V32_TZ)
    except Exception:
        dt = None
    if dt is None:
        dt = datetime.now(timezone.utc).astimezone(V32_TZ)
    hour = dt.hour + dt.minute / 60.0
    angle = 2.0 * np.pi * hour / 24.0
    if 0 <= dt.hour < 6:
        session = "NIGHT_EARLY"
    elif dt.hour < 12:
        session = "MORNING"
    elif dt.hour < 18:
        session = "AFTERNOON"
    else:
        session = "EVENING_NIGHT"
    return {
        "timezone": V32_TIMEZONE,
        "hour_local": int(dt.hour),
        "minute_local": int(dt.minute),
        "day_of_week": int(dt.weekday()),
        "session": session,
        "hour_sin": float(np.sin(angle)),
        "hour_cos": float(np.cos(angle)),
        "is_weekend": bool(dt.weekday() >= 5),
    }


def _v32_record_experience(row):
    row = dict(row or {})
    with _V32_LOCK:
        _v32_append(V32_EXPERIENCE_FILE, row, _V32_BUFFER)
        typ = str(row.get("type") or "")
        if typ == "market_observation":
            _V32_STATE["observations"] = int(_V32_STATE.get("observations", 0)) + 1
        elif typ == "candidate":
            _V32_STATE["candidates"] = int(_V32_STATE.get("candidates", 0)) + 1
        elif typ == "trade_outcome":
            _V32_STATE["outcomes"] = int(_V32_STATE.get("outcomes", 0)) + 1
            result = row.get("outcome") if isinstance(row.get("outcome"), dict) else row
            final_r = _v32_f((result or {}).get("final_r", row.get("final_r", 0.0)))
            if final_r > 0:
                _V32_STATE["wins"] = int(_V32_STATE.get("wins", 0)) + 1
            elif final_r < 0:
                _V32_STATE["losses"] = int(_V32_STATE.get("losses", 0)) + 1
        _v32_json_save(V32_STATE_FILE, _V32_STATE)
    # Wake the worker immediately; it remains rate-limited by its own review cycle.
    _V32_WAKE.set()
    return row


def _v32_current_records(limit=V32_RESEARCH_WINDOW):
    records = []
    try:
        if V32_EXPERIENCE_FILE.exists():
            with V32_EXPERIENCE_FILE.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            records.append(row)
                    except Exception:
                        continue
    except Exception as exc:
        log.warning(f"[V32] read experience gagal: {exc}")
    return records[-max(1, int(limit)):]


def _v32_outcome_payload(row):
    if not isinstance(row, dict):
        return {}
    p = row.get("outcome")
    return p if isinstance(p, dict) else row


def _v32_outcomes(records=None):
    rows = records if records is not None else _v32_current_records()
    return [r for r in rows if isinstance(r, dict) and r.get("type") == "trade_outcome"]


def _v32_candidate_records(records=None):
    rows = records if records is not None else _v32_current_records()
    return [r for r in rows if isinstance(r, dict) and r.get("type") == "candidate"]


def _v32_market_records(records=None):
    rows = records if records is not None else _v32_current_records()
    return [r for r in rows if isinstance(r, dict) and r.get("type") == "market_observation"]


def _v32_effective_sample(rows):
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return 0.0
    symbols = len({str(r.get("symbol")) for r in rows if r.get("symbol")})
    regimes = len({str(r.get("regime") or r.get("market_regime")) for r in rows if r.get("regime") or r.get("market_regime")})
    times = len({str((r.get("time_context") or {}).get("session")) for r in rows})
    diversity = max(1.0, min(2.5, 1.0 + 0.12 * symbols + 0.10 * regimes + 0.05 * times))
    return float(len(rows) / diversity)


def _v32_wilson(pos, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = pos / n
    den = 1.0 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt(max(0.0, p*(1-p)/n + z*z/(4*n*n))) / den
    return (max(0.0, center-half), min(1.0, center+half))


def _v32_calibration(outcomes):
    if len(outcomes) < V32_MIN_OUTCOMES:
        return {"status": "INSUFFICIENT", "sample_size": len(outcomes), "buckets": [], "error": None}
    buckets = []
    for lo, hi in ((0, .40),(.40,.50),(.50,.60),(.60,.70),(.70,.80),(.80,1.01)):
        vals=[]
        for r in outcomes:
            p = _v32_f(r.get("confidence"), 50.0) / 100.0
            if lo <= p < hi:
                payload=_v32_outcome_payload(r)
                fr=_v32_f(payload.get("final_r"), 0.0)
                vals.append((p, 1.0 if fr>0 else 0.0))
        if not vals:
            continue
        pred=float(np.mean([x[0] for x in vals])); actual=float(np.mean([x[1] for x in vals])); n=len(vals)
        low, high=_v32_wilson(sum(x[1] for x in vals), n)
        buckets.append({"range":f"{int(lo*100)}-{min(99,int(hi*100))}%","n":n,"predicted":round(pred,4),"actual":round(actual,4),"error":round(actual-pred,4),"ci_low":round(low,4),"ci_high":round(high,4)})
    if len(buckets) < 2:
        return {"status":"INSUFFICIENT","sample_size":len(outcomes),"buckets":buckets,"error":None}
    monotonic = all(buckets[i]["actual"] <= buckets[i+1]["actual"] + 0.08 for i in range(len(buckets)-1))
    mae = float(np.mean([abs(b["error"]) for b in buckets]))
    return {"status":"GOOD" if monotonic and mae < .15 else "NEEDS_REVIEW","sample_size":len(outcomes),"buckets":buckets,"mae":round(mae,4),"monotonic":monotonic}


def _v32_time_effect(outcomes):
    if len(outcomes) < V32_MIN_OUTCOMES:
        return {"status":"INSUFFICIENT","usable":False,"sample_size":len(outcomes),"effects":[]}
    rows=[]
    for r in outcomes:
        tc=r.get("time_context") if isinstance(r.get("time_context"),dict) else _v32_hour_context(r.get("entry_timestamp") or r.get("timestamp"))
        session=tc.get("session")
        if not session or session=="UNKNOWN": continue
        fr=_v32_f(_v32_outcome_payload(r).get("final_r"),0.0)
        rows.append((session,fr))
    if len(rows)<V32_MIN_OUTCOMES:
        return {"status":"INSUFFICIENT","usable":False,"sample_size":len(rows),"effects":[]}
    overall=float(np.mean([x[1] for x in rows])); effects=[]
    for session in sorted({x[0] for x in rows}):
        vals=[x[1] for x in rows if x[0]==session]
        n=len(vals)
        if n<V32_MIN_CELL: continue
        mean=float(np.mean(vals)); sd=float(np.std(vals,ddof=1)) if n>1 else 0.0; se=sd/np.sqrt(n) if sd>0 else 0.0
        t=(mean-overall)/se if se>1e-12 else 0.0
        shrink=min(1.0,n/40.0)
        effect=float(np.tanh(mean-overall)*shrink)
        effects.append({"session":session,"n":n,"mean_r":round(mean,5),"delta":round(mean-overall,5),"t_like":round(float(t),3),"effect":round(effect,5)})
    effects.sort(key=lambda x:abs(x["effect"]),reverse=True)
    usable=bool(effects and effects[0]["n"]>=24 and abs(effects[0]["t_like"])>=2.0)
    return {"status":"SUPPORTED" if usable else "WEAK_SIGNAL","usable":usable,"sample_size":len(rows),"overall_mean_r":round(overall,5),"effects":effects,"timezone":V32_TIMEZONE}


def _v32_conditional_cells(outcomes, key_fn, min_cell=V32_MIN_CELL):
    cells={}
    for r in outcomes:
        key=key_fn(r)
        if key is None: continue
        payload=_v32_outcome_payload(r); fr=_v32_f(payload.get("final_r"),0.0)
        cells.setdefault(key,[]).append(fr)
    out=[]
    for key,vals in cells.items():
        if len(vals)<min_cell: continue
        pos=sum(1 for x in vals if x>0); n=len(vals); low,high=_v32_wilson(pos,n)
        out.append({"key":key,"n":n,"mean_r":round(float(np.mean(vals)),5),"positive_rate":round(pos/n,4),"ci_low":round(low,4),"ci_high":round(high,4)})
    return sorted(out,key=lambda x:x["n"],reverse=True)


def _v32_coverage(candidates):
    n=len(candidates)
    live=sum(1 for r in candidates if not r.get("rejected_reason") and str(r.get("decision") or "").upper() in {"BUY","SELL"})
    shadow=sum(1 for r in candidates if r.get("rejected_reason") or str(r.get("decision") or "").upper() not in {"BUY","SELL"})
    return {"candidate_count":n,"live_candidate_count":live,"shadow_count":shadow,"coverage_rate":round(live/max(1,n),4),"exploration_share":V32_EXPLORATION_SHARE}


def _v32_autopsy(record):
    payload=_v32_outcome_payload(record); fr=_v32_f(payload.get("final_r"),0.0); mfe=_v32_f(payload.get("mfe_r",record.get("mfe_r")),0.0); mae=_v32_f(payload.get("mae_r",record.get("mae_r")),0.0)
    gb=_v32_f(payload.get("giveback_ratio",record.get("giveback_ratio")),0.0)
    result=str(record.get("result") or payload.get("result") or "").lower()
    if fr>0: cls="SUCCESS"
    elif mfe>=1.0 and gb>=0.5: cls="MANAGEMENT_OR_PROTECTION_FAILURE"
    elif mfe<0.35 and fr<0: cls="ENTRY_OR_THESIS_FAILURE"
    else: cls="AMBIGUOUS_FAILURE"
    if result in {"execution_error","data_error","strategy_error"}: cls="EXECUTION_OR_DATA_FAILURE"
    return {"type":"autopsy","timestamp":time.time(),"symbol":record.get("symbol"),"class":cls,"final_r":fr,"mfe_r":mfe,"mae_r":mae,"giveback_ratio":gb,"confidence":_v32_f(record.get("confidence"),50),"archetype":record.get("archetype"),"regime":record.get("market_regime",record.get("regime")),"thesis":record.get("thesis") or (record.get("research_snapshot") or {}).get("thesis"),"lesson_type":"preserve" if cls=="SUCCESS" else "repair"}


def _v32_counterfactuals(record):
    p=_v32_outcome_payload(record); actual=_v32_f(p.get("final_r"),0.0); mfe=_v32_f(p.get("mfe_r"),0.0); gb=_v32_f(p.get("giveback_ratio"),0.0)
    rows=[]
    if actual<0:
        rows.append({"policy":"require_reclaim","simulated_delta_r":round(0.10*max(0,mfe),4) if record.get("selected_sweep") else 0.0})
        rows.append({"policy":"delay_entry_1_bar","simulated_delta_r":round(-0.05*abs(_v32_f(p.get("mae_r"),0.0)),4)})
    if actual<0 and mfe>=1.0 and gb>=0.5:
        rows.append({"policy":"adaptive_giveback_trail","simulated_delta_r":round(0.20*mfe,4)})
    return rows


def _v32_stable_policy_update(name, observed, evidence_n, condition="global", source="research"):
    """Update a belief only on fresh, batched evidence; repeated reviews of the same sample do nothing."""
    n = int(evidence_n or 0)
    if n < 5:
        return None
    key=f"{name}|{condition}"
    with _V32_LOCK:
        b=dict(_V32_BELIEFS.get(key) or {"name":name,"condition":condition,"value":observed,"strength":0.25,"samples":0,"stable_reviews":0,"last_change":0.0})
        old=_v32_f(b.get("value"),observed); strength=max(0.05,min(0.98,_v32_f(b.get("strength"),0.25)))
        effective=min(1.0,n/200.0)
        # Single review is capped. Fresh evidence must accumulate before policy moves.
        rate=min(V32_SINGLE_EVENT_MAX_BELIEF_DELTA, 0.01+0.025*effective)
        new=old + np.clip(_v32_f(observed)-old,-rate,rate)
        changed=abs(new-old)>1e-12
        stable=int(b.get("stable_reviews",0) or 0) + (1 if changed else 0)
        b.update({"value":float(new),"strength":float(min(.98,strength+0.01*effective)),"samples":int(b.get("samples",0) or 0)+n,"stable_reviews":stable,"last_change":float(abs(new-old)),"last_source":source,"last_revision":time.time()})
        _V32_BELIEFS[key]=b
        if changed:
            _V32_STATE["belief_revisions"]=int(_V32_STATE.get("belief_revisions",0))+1
            _V32_STATE["last_policy_revision"]=time.time()
            _V32_STATE["policy_revision_count"]=int(_V32_STATE.get("policy_revision_count",0))+1
        _v32_json_save(V32_POLICY_FILE,_V32_BELIEFS); _v32_json_save(V32_STATE_FILE,_V32_STATE)
        return dict(b) if changed else None


def get_v32_belief(name, condition="global", default=None):
    with _V32_LOCK:
        return dict(_V32_BELIEFS.get(f"{name}|{condition}") or {"name":name,"condition":condition,"value":default,"strength":0.20,"samples":0,"stable_reviews":0})


def _v32_make_questions(outcomes, candidates, time_effect, calibration, drift):
    qs=[]
    if time_effect.get("status") in {"WEAK_SIGNAL","SUPPORTED"}:
        qs.append("Apakah time-of-day effect bertahan setelah mengontrol setup dan regime?")
    if calibration.get("status")=="NEEDS_REVIEW":
        qs.append("Apakah confidence buckets masih monotonic dan terkalibrasi?")
    if drift.get("status") in {"MEDIUM","HIGH"}:
        qs.append("Apakah recent market distribution berbeda enough untuk memerlukan challenger?")
    if outcomes:
        failures=[_v32_autopsy(x) for x in outcomes if _v32_f(_v32_outcome_payload(x).get("final_r"),0)<0]
        mgmt=sum(1 for x in failures if x["class"]=="MANAGEMENT_OR_PROTECTION_FAILURE")
        entry=sum(1 for x in failures if x["class"]=="ENTRY_OR_THESIS_FAILURE")
        if mgmt and mgmt>entry: qs.append("Apakah management/trailing saat ini lebih lemah daripada entry thesis?")
        elif entry: qs.append("Apakah failure cluster sebenarnya berasal dari thesis/location/timing?")
    # Deduplicate while keeping the list bounded.
    uniq=[]
    for q in qs:
        if q not in uniq: uniq.append(q)
    return uniq[:8]


def _v32_model_fit(outcomes):
    # Lightweight research-only model. It does not replace the main live model.
    if len(outcomes) < V32_MIN_OUTCOMES:
        return {"status": "INSUFFICIENT", "reason": "too_few_outcomes"}
    rows, ys = [], []
    for r in outcomes:
        lf = r.get("learning_features")
        if not isinstance(lf, dict):
            continue
        p = _v32_f(r.get("confidence"), 50.0) / 100.0
        payload = _v32_outcome_payload(r)
        y = 1.0 if _v32_f(payload.get("final_r"), 0.0) > 0 else 0.0
        rows.append([p, _v32_f(lf.get("setup_quality"),0)/100.0, _v32_f(lf.get("entry_location_score"),50)/100.0, _v32_f(lf.get("rr"),0)/4.0, _v32_f(lf.get("selected_sweep"),0), _v32_f((r.get("time_context") or {}).get("hour_sin"),0), _v32_f((r.get("time_context") or {}).get("hour_cos"),1)])
        ys.append(y)
    if len(rows) < max(V32_MIN_OUTCOMES, 10):
        return {"status":"INSUFFICIENT", "reason":"too_few_labeled_rows", "sample_count":len(rows)}
    X=np.asarray(rows,float); y=np.asarray(ys,float)
    test_n=max(5, int(round(len(X)*0.30)))
    if len(X)-test_n < 5:
        test_n=max(1, len(X)//3)
    train_n=len(X)-test_n
    if train_n < 5 or test_n < 1:
        return {"status":"INSUFFICIENT", "reason":"invalid_oos_split", "sample_count":len(X)}
    Ztr_raw=X[:train_n]; Zte_raw=X[train_n:]; ytr=y[:train_n]; yte=y[train_n:]
    if len(np.unique(ytr)) < 2:
        return {"status":"INSUFFICIENT", "reason":"single_class_training", "sample_count":len(X), "train_count":train_n, "oos_count":test_n}
    mu=Ztr_raw.mean(axis=0); sd=Ztr_raw.std(axis=0); sd[sd<1e-8]=1.0
    Ztr=(Ztr_raw-mu)/sd; Zte=(Zte_raw-mu)/sd
    w=np.zeros(Ztr.shape[1]); b=0.0
    for _ in range(220):
        z=np.clip(Ztr@w+b,-8,8); p=1/(1+np.exp(-z)); grad=(Ztr.T@(p-ytr))/len(ytr)+0.01*w; gb=float(np.mean(p-ytr)); w-=0.08*grad; b-=0.08*gb
    pte=1/(1+np.exp(-np.clip(Zte@w+b,-8,8))); pred=(pte>=0.5).astype(float); acc=float(np.mean(pred==yte)) if len(yte) else 0.0
    return {"status":"READY","sample_count":len(X),"train_count":len(ytr),"oos_count":len(yte),"oos_accuracy":round(acc,4),"w":w.tolist(),"b":float(b),"mean":mu.tolist(),"scale":sd.tolist(),"feature_names":["confidence","setup_quality","location","rr","sweep","hour_sin","hour_cos"]}


def _v32_drift(market):
    if len(market)<80: return {"status":"UNKNOWN","score":0.0,"features":[]}
    ref=market[:-40]; recent=market[-40:]
    keys=sorted(set().union(*(r.get("features",{}).keys() for r in ref+recent)))
    ds=[]
    for k in keys:
        a=np.asarray([_v32_f((r.get("features") or {}).get(k),0.0) for r in ref]); b=np.asarray([_v32_f((r.get("features") or {}).get(k),0.0) for r in recent]); s=float(np.std(a)) or 1.0; d=abs(float(np.mean(b)-np.mean(a)))/s; ds.append((d,k))
    ds.sort(reverse=True); score=min(1.0,float(np.mean([min(x[0],3)/3 for x in ds]))) if ds else 0.0
    status="HIGH" if score>=0.45 else "MEDIUM" if score>=0.25 else "LOW"
    return {"status":status,"score":round(score,4),"features":[{"feature":k,"shift":round(d,3)} for d,k in ds[:10]]}


def _v33_log(phase, message, level=logging.INFO):
    """Structured FULL telemetry for Render logs."""
    log.log(level, f"[FULL][{phase}] {message}")

def _v33_notify(message, force=False):
    """Optional low-frequency Telegram heartbeat; never raises into learning."""
    global _V32_LAST_TELEGRAM_NOTIFY
    cb = _V32_NOTIFY
    if not callable(cb):
        return
    now = time.time()
    if not force and now - _V32_LAST_TELEGRAM_NOTIFY < _V32_TELEGRAM_MIN_INTERVAL:
        return
    try:
        cb(str(message))
        _V32_LAST_TELEGRAM_NOTIFY = now
    except Exception as exc:
        log.debug(f"[FULL][TELEGRAM] heartbeat gagal: {exc}")

def _v32_research_cycle():
    _v33_log("CYCLE", "START")
    records=_v32_current_records()
    outcomes=_v32_outcomes(records); candidates=_v32_candidate_records(records); market=_v32_market_records(records)
    with _V32_LOCK:
        prev_outcomes=int(_V32_STATE.get("last_reviewed_outcomes",0) or 0)
        prev_model_samples=int(_V32_STATE.get("last_model_sample_count",0) or 0)
    fresh_outcomes=max(0,len(outcomes)-prev_outcomes)
    _v33_log("DATA", f"loaded observations={len(market)} candidates={len(candidates)} outcomes={len(outcomes)} fresh_outcomes={fresh_outcomes}")
    coverage=_v32_coverage(candidates)
    _v33_log("FREQUENCY", f"candidates={coverage.get('candidate_count',0)} live={coverage.get('live_candidate_count',0)} shadow={coverage.get('shadow_count',0)} rate={coverage.get('coverage_rate',0):.1%}")
    calibration=_v32_calibration(outcomes)
    _v33_log("CALIBRATION", f"status={calibration.get('status','INSUFFICIENT')} samples={len(outcomes)}")
    time_effect=_v32_time_effect(outcomes)
    _v33_log("TIME", f"status={time_effect.get('status','INSUFFICIENT')} usable={bool(time_effect.get('usable'))}")
    drift=_v32_drift(market)
    _v33_log("DRIFT", f"status={drift.get('status','UNKNOWN')} score={drift.get('score',0):.3f}")
    questions=_v32_make_questions(outcomes,candidates,time_effect,calibration,drift)
    if questions:
        _v33_log("HYPOTHESIS", f"research_questions={len(questions)}")
        for q in questions[:5]: _v33_log("HYPOTHESIS", q, logging.DEBUG)
    model={"status":"INSUFFICIENT"}
    if len(outcomes) >= V32_MIN_OUTCOMES and (prev_model_samples == 0 or len(outcomes)-prev_model_samples >= 10):
        model=_v32_model_fit(outcomes)
        if model.get("status")=="READY":
            oos=float(model.get("oos_accuracy",0.0) or 0.0)
            if oos >= 0.52 and int(model.get("oos_count",0) or 0) >= 10:
                promoted=dict(model)
                promoted.update({"active":True,"schema":MACHINE_LEARNING_SCHEMA,"model_version":f"ML-{int(time.time())}","live_weight":0.25,"promoted_at":time.time()})
                if _save_active_learning_model(promoted):
                    set_learning_model(promoted)
                    with _V32_LOCK:
                        _V32_STATE["model_promotions"]=int(_V32_STATE.get("model_promotions",0))+1
            with _V32_LOCK:
                _V32_STATE["model_candidates"]=int(_V32_STATE.get("model_candidates",0))+1
                _V32_STATE["last_model_update"]=time.time()
                _V32_STATE["last_model_sample_count"]=len(outcomes)
            _v33_log("MODEL", f"research_fit=READY samples={model.get('sample_count',0)} oos={model.get('oos_count',0)} accuracy={oos:.3f}")
        else:
            _v33_log("MODEL", f"research_fit={model.get('status','INSUFFICIENT')} samples={model.get('sample_count',0)}")
    else:
        _v33_log("MODEL", f"research_fit=HOLD samples={len(outcomes)} fresh_outcomes={fresh_outcomes}")
    lessons=[]
    if fresh_outcomes>0:
        batch=outcomes[-min(25,fresh_outcomes):]
        for row in batch:
            dx=_v32_autopsy(row); lessons.append(dx); _v33_log("AUTOPSY", f"{dx.get('symbol','?')} class={dx.get('class','UNKNOWN')}", logging.DEBUG)
            for _ in _v32_counterfactuals(row):
                with _V32_LOCK: _V32_STATE["counterfactuals"]=int(_V32_STATE.get("counterfactuals",0))+1
        with _V32_LOCK: _V32_STATE["autopsies"]=int(_V32_STATE.get("autopsies",0))+len(lessons)
        for lesson in lessons[-10:]: _v32_append(V32_LESSON_FILE,lesson,[])
    _v33_log("AUTOPSY", f"reviewed={len(lessons)} counterfactuals_total={_V32_STATE.get('counterfactuals',0)}")
    # Policy/belief changes require fresh batches; repeated scans of the same outcomes cannot move them.
    if fresh_outcomes>=5 and time_effect.get("usable"):
        best=time_effect["effects"][0]
        revised=_v32_stable_policy_update("time_effect", float(best.get("delta",0.0)), best.get("n",0), condition=str(best.get("session")), source="time_research")
        _v33_log("BELIEF", f"time_effect reviewed; changed={bool(revised)}")
    else:
        _v33_log("BELIEF", "time effect observed; no policy change without fresh sufficient evidence", logging.DEBUG)
    if fresh_outcomes>=5 and calibration.get("status")=="GOOD":
        avg_error=float(calibration.get("mae",0.0) or 0.0)
        revised=_v32_stable_policy_update("confidence_calibration_error",avg_error,fresh_outcomes,condition="global",source="calibration")
        _v33_log("BELIEF", f"calibration belief reviewed; changed={bool(revised)}")
    else:
        _v33_log("BELIEF", "calibration observed; no policy change without fresh evidence", logging.DEBUG)
    with _V32_LOCK:
        _V32_STATE.update({"version":V32_VERSION,"time_effect":time_effect,"calibration":calibration,"coverage":coverage,"drift_score":drift.get("score",0.0),"drift_status":drift.get("status","UNKNOWN"),"research_questions":len(questions),"resolved_questions":max(0,int(_V32_STATE.get("resolved_questions",0))),"last_review":time.time(),"last_reviewed_outcomes":len(outcomes)})
        _v32_json_save(V32_STATE_FILE,_V32_STATE)
    _v33_log("CYCLE", f"COMPLETE observations={len(market)} candidates={len(candidates)} outcomes={len(outcomes)} fresh_outcomes={fresh_outcomes}")
    _v33_notify(f"🧠 FULL masih belajar…\nCycle selesai • {len(market)} observations • {len(candidates)} candidates", force=False)
    return {"schema":V32_SCHEMA,"version":V32_VERSION,"outcomes":len(outcomes),"candidates":len(candidates),"coverage":coverage,"calibration":calibration,"time_effect":time_effect,"drift":drift,"questions":questions,"model_research":model,"lessons_reviewed":len(lessons),"fresh_outcomes":fresh_outcomes}


def _v32_loop():
    global _V32_TICKS,_V32_LAST_ERROR
    _v33_log("WORKER", "STARTED")
    while not _V32_STOP.is_set():
        try:
            _v32_research_cycle()
            _V32_TICKS += 1
            _V32_LAST_ERROR = None
        except Exception as exc:
            _V32_LAST_ERROR = str(exc)[:500]
            log.exception("[FULL][ERROR] research cycle gagal")
            _v33_notify("⚠️ FULL mengalami error penelitian. Worker tetap hidup dan akan mencoba lagi.", force=True)
        _V32_WAKE.wait(V32_REVIEW_INTERVAL)
        _V32_WAKE.clear()
    _v33_log("WORKER", "STOPPED")


def _v32_start():
    global _V32_THREAD
    with _V32_LOCK:
        if _V32_THREAD is None or not _V32_THREAD.is_alive():
            _V32_STOP.clear(); _V32_WAKE.set()
            _V32_THREAD=threading.Thread(target=_v32_loop,name="full-v32-brain",daemon=True)
            _V32_THREAD.start()
        else:
            _V32_WAKE.set()


def _v32_stop():
    _V32_STOP.set(); _V32_WAKE.set()


def get_v32_status():
    with _V32_LOCK:
        return {"version":V32_VERSION,"worker_alive":bool(_V32_THREAD is not None and _V32_THREAD.is_alive()),"ticks":_V32_TICKS,"last_error":_V32_LAST_ERROR,"state":dict(_V32_STATE),"beliefs":len(_V32_BELIEFS)}


def _v32_time_features_into(signal, timestamp=None):
    tc=_v32_hour_context(timestamp)
    out=dict(signal or {})
    out["time_context"]=tc
    out["time_aware_research"] = True
    return out


# V32 public bridge: every strategy result becomes a research observation; actual
# execution remains owned by main.py.
_V31_BASE_FULL_ANALYZE = _CORE_FULL_ANALYZE
_V31_BASE_MANAGE_POSITION = _CORE_MANAGE_POSITION


def full_analyze(df_h1, df_m15, df_d1=None, symbol=None, df_btc_h1=None, trade_history=None):
    result = _V31_BASE_FULL_ANALYZE(df_h1, df_m15, df_d1, symbol=symbol, df_btc_h1=df_btc_h1, trade_history=trade_history)
    ts = _timestamp_from_df(df_m15)
    try:
        feats = extract_market_features(df_h1, df_m15, df_d1, df_btc_h1)
        feats["time_context"] = _v32_hour_context(ts)
        _v32_record_experience({"type":"market_observation","timestamp":time.time(),"symbol":symbol,"source":"binance","features":feats,"regime":(result or {}).get("market_regime") if isinstance(result,dict) else None,"result":"strategy_result" if isinstance(result,dict) else "no_strategy_result"})
    except Exception as exc:
        _v33_log("DATA", f"market observation bridge gagal {symbol}: {exc}", logging.DEBUG)
    if not isinstance(result, dict):
        return {
            "symbol": symbol, "decision": "HOLD", "confidence": 0,
            "execution_eligible": False, "no_signal": True,
            "analysis_stage": "CORE_NO_CANDIDATE",
            "rejected_reason": "NO_VALID_ENTRY_CANDIDATE",
            "brain_version": V40_VERSION, "cognitive_schema": V32_SCHEMA,
            "strategy_version": str(_AGENT_STATE.get("strategy_version") or "S1"),
            "time_context": _v32_hour_context(ts),
        }
    try:
        result = _v32_time_features_into(result, ts)
        result.setdefault("learning_features", {})
        tc = result["time_context"]
        result["learning_features"].update({"hour_sin":tc["hour_sin"],"hour_cos":tc["hour_cos"],"hour_local":float(tc["hour_local"])/23.0,"is_weekend":float(tc["is_weekend"])})
        snap = build_research_snapshot(result, df_h1, df_m15, df_d1, symbol=symbol)
        snap["time_context"] = tc
        snap["coverage_role"] = "candidate"
        result["research_snapshot"] = snap
        result["brain_version"] = V40_VERSION
        result["cognitive_schema"] = V32_SCHEMA
        result["strategy_version"] = str(_AGENT_STATE.get("strategy_version") or "S1")
    except Exception as exc:
        log.warning(f"[V32] full_analyze annotation gagal {symbol}: {exc}")
    return result


def manage_position(state, df_m15, df_h1=None, df_d1=None, symbol=None):
    result=_V31_BASE_MANAGE_POSITION(state,df_m15,df_h1,df_d1,symbol=symbol)
    if isinstance(result,dict):
        try:
            tc=_v32_hour_context(_timestamp_from_df(df_m15))
            result["time_context"]=tc
            result["trail_research"]={"frequency_neutral":True,"learning_target":"capture_ratio_vs_continuation_survival","uses_actual_position_only":True}
        except Exception:
            pass
    return result


def ingest_live_candidate(signal,h1=None,m15=None,d1=None,rejected_reason=None,source="binance"):
    """Record one live/shadow candidate idempotently. Never calls execution APIs."""
    sig=dict(signal or {}) if isinstance(signal,dict) else {}
    tc=sig.get("time_context") if isinstance(sig.get("time_context"),dict) else _v32_hour_context(sig.get("entry_timestamp") or sig.get("timestamp"))
    snap=sig.get("research_snapshot") if isinstance(sig.get("research_snapshot"),dict) else {}
    uid=str(sig.get("candidate_uid") or f"{sig.get('symbol','')}|{sig.get('timestamp','')}|{sig.get('decision','')}|{sig.get('entry','')}")
    key=f"{uid}|{source}"
    with _V32_LOCK:
        if key in _V34_SEEN_CANDIDATE_KEYS:
            _v33_log("DATA", f"duplicate candidate ignored uid={uid}", logging.DEBUG)
            return None
        _V34_SEEN_CANDIDATE_KEYS.add(key)
    row={"type":"candidate","timestamp":time.time(),"candidate_uid":uid,"symbol":sig.get("symbol"),"source":source,"decision":sig.get("decision"),"confidence":_v32_f(sig.get("confidence"),50),"trade_quality":_v32_f(sig.get("trade_quality",sig.get("setup_quality")),0),"archetype":sig.get("archetype"),"regime":sig.get("market_regime") or (snap.get("market") or {}).get("regime"),"time_context":tc,"learning_features":dict(sig.get("learning_features") or {}),"research_snapshot":snap,"signal":sig,"rejected_reason":rejected_reason}
    return _v32_record_experience(row)


_V34_SEEN_OUTCOME_KEYS = set()
_V34_SEEN_CANDIDATE_KEYS = set()

def ingest_live_outcome(signal,outcome,source="binance_trade"):
    sig=dict(signal or {}) if isinstance(signal,dict) else {}
    payload=outcome if isinstance(outcome,dict) else {"result":outcome}
    uid=str(sig.get("trade_uid") or sig.get("order_id") or f"{sig.get('symbol','')}|{sig.get('entry_time','')}|{sig.get('exit_time','')}|{payload.get('result','')}")
    key=f"{uid}|{str(source)}"
    with _V32_LOCK:
        if key in _V34_SEEN_OUTCOME_KEYS:
            _v33_log("DATA", f"duplicate outcome ignored uid={uid}", logging.DEBUG)
            return None
        _V34_SEEN_OUTCOME_KEYS.add(key)
    row={"type":"trade_outcome","timestamp":time.time(),"symbol":sig.get("symbol"),"source":source,"decision":sig.get("decision"),"confidence":sig.get("confidence"),"archetype":sig.get("archetype"),"market_regime":sig.get("market_regime"),"time_context":sig.get("time_context") or _v32_hour_context(sig.get("entry_timestamp")),"learning_features":dict(sig.get("learning_features") or {}),"research_snapshot":sig.get("research_snapshot") or {},"signal":sig,"outcome":dict(payload)}
    return _v32_record_experience(row)


def full_learning_review(max_rows=V32_RESEARCH_WINDOW):
    return _v32_research_cycle()


def reset_cognitive_memory():
    """Reset V32 cognitive research memory only; never touches trading ledger."""
    global _V32_STATE,_V32_BELIEFS,_V32_BUFFER,_V32_TICKS,_V32_LAST_ERROR
    _v32_stop()
    with _V32_LOCK:
        _V32_STATE={
            "schema":V32_SCHEMA,"version":V32_VERSION,"observations":0,"candidates":0,"outcomes":0,"wins":0,"losses":0,"autopsies":0,"counterfactuals":0,"belief_revisions":0,"research_questions":0,"resolved_questions":0,"model_candidates":0,"model_promotions":0,"drift_score":0.0,"drift_status":"UNKNOWN","time_effect":{"status":"INSUFFICIENT","usable":False},"coverage":{"candidate_count":0,"live_candidate_count":0,"shadow_count":0,"coverage_rate":0.0},"calibration":{"status":"INSUFFICIENT","buckets":[]},"last_review":None,"last_model_update":None,"last_model_sample_count":0,"last_reviewed_outcomes":0,"last_policy_revision":None,"policy_revision_count":0,
        }
        _V32_BELIEFS={}; _V32_BUFFER=[]; _V32_TICKS=0; _V32_LAST_ERROR=None
        _V34_SEEN_OUTCOME_KEYS.clear(); _V34_SEEN_CANDIDATE_KEYS.clear()
        if "_COG_SEEN_OUTCOME_KEYS" in globals():
            globals()["_COG_SEEN_OUTCOME_KEYS"].clear()
        for path in (V32_STATE_FILE,V32_EXPERIENCE_FILE,V32_LESSON_FILE,V32_POLICY_FILE):
            try: path.unlink(missing_ok=True)
            except Exception: pass
        _v32_json_save(V32_STATE_FILE,_V32_STATE); _v32_json_save(V32_POLICY_FILE,_V32_BELIEFS)
    return get_v32_status()


def full_command(action, callbacks=None):
    global _V32_NOTIFY, _V32_LAST_TELEGRAM_NOTIFY, _FULL_ENABLED, _LEARNED_MODEL
    callbacks=callbacks if isinstance(callbacks,dict) else {}
    if callable(callbacks.get("notify")):
        _V32_NOTIFY = callbacks.get("notify")
    action=str(action or "status").strip().lower()
    try:
        if action=="on":
            _FULL_ENABLED=True
            cb=callbacks.get("on"); payload=cb() if callable(cb) else {}
            _v32_start()
            try: adaptive_agent_start()
            except Exception as exc: log.warning(f"[FULL] adaptive worker start gagal: {exc}")
            _v33_notify("🧠 FULL ON — otak mulai belajar terus.", force=True)
            return ("🧠 <b>FULL LEARNING ON</b>\n"
                    "Otak belajar terus: observasi → kandidat → outcome → autopsy → counterfactual → calibration → hypothesis → adaptation.\n"
                    "<b>Frequency tetap menjadi objective</b>; learning tidak diperbolehkan mematikan opportunity hanya demi menjadi lebih ketat.\n"
                    "Satu SL tidak mengubah policy secara langsung; perubahan harus melewati bukti bertahap dan stability review.\n\n" + _v32_full_text(payload))
        if action=="off":
            _FULL_ENABLED=False
            _v32_stop()
            try: adaptive_agent_stop()
            except Exception as exc: log.warning(f"[FULL] adaptive worker stop gagal: {exc}")
            cb=callbacks.get("off"); payload=cb() if callable(cb) else {}
            _v33_notify("🧠 FULL OFF — learning dihentikan; model terakhir dipertahankan.", force=True)
            return "🧠 <b>FULL LEARNING OFF</b>\nResearch worker dihentikan; state/model terakhir dipertahankan.\n\n"+_v32_full_text(payload)
        if action=="reset":
            _FULL_ENABLED=False
            try: adaptive_agent_stop()
            except Exception: pass
            _LEARNED_MODEL=None
            try: _LEARNED_MODEL_FILE.unlink(missing_ok=True)
            except Exception: pass
            cb=callbacks.get("reset"); payload=cb() if callable(cb) else {}
            st=reset_cognitive_memory()
            return "🧠 <b>FULL RESET</b>\nCognitive research memory V32 dihapus; trading ledger/posisi tidak disentuh.\n\n"+_v32_full_text({"mode":False,"cognitive_worker":st})
        if action in {"review","research"}:
            rep=_v32_research_cycle(); return _v32_research_text(rep)
        cb=callbacks.get("status"); payload=cb() if callable(cb) else {}
        return "🧠 <b>FULL LEARNING STATUS</b>\n"+_v32_full_text(payload)
    except Exception as exc:
        log.exception("[V32 FULL] command gagal")
        return f"❌ <b>FULL gagal</b>\n<code>{html.escape(str(exc)[:400])}</code>"


def _v32_full_text(payload):
    p=payload if isinstance(payload,dict) else {}
    st=get_v32_status(); s=st["state"]; cw="ON" if st["worker_alive"] else "OFF"
    champion=p.get("champion") if isinstance(p.get("champion"),dict) else {}
    return (f"Main mode: <b>{'ON' if p.get('mode') else 'OFF'}</b>\n"
            f"Cognitive worker: <b>{cw}</b> | ticks: <b>{st['ticks']}</b>\n"
            f"Observations: <b>{s.get('observations',0)}</b> | Candidates: <b>{s.get('candidates',0)}</b>\n"
            f"Outcomes: <b>{s.get('outcomes',0)}</b> | W/L: <b>{s.get('wins',0)}/{s.get('losses',0)}</b>\n"
            f"Coverage: <b>{(s.get('coverage') or {}).get('coverage_rate',0)*100:.1f}%</b>\n"
            f"Calibration: <b>{(s.get('calibration') or {}).get('status','INSUFFICIENT')}</b>\n"
            f"Time effect: <b>{(s.get('time_effect') or {}).get('status','INSUFFICIENT')}</b>\n"
            f"Drift: <b>{s.get('drift_status','UNKNOWN')}</b>\n"
            f"Beliefs: <b>{st['beliefs']}</b>\n"
            f"Policy revisions: <b>{s.get('policy_revision_count',0)}</b>\n"
            f"Champion: <code>{html.escape(str(champion.get('model_version','—')))}</code>"
            f"\nWorker: <b>{'ON' if get_v32_status().get('worker_alive') else 'OFF'}</b> | Ticks: <b>{get_v32_status().get('ticks',0)}</b>"
            f"\nLast error: <b>{html.escape(str(get_v32_status().get('last_error') or '—')[:160])}</b>")


def _v32_research_text(rep):
    cal=rep.get("calibration",{}); te=rep.get("time_effect",{}); drift=rep.get("drift",{}); cov=rep.get("coverage",{})
    return ("🔬 <b>FULL RESEARCH REVIEW</b>\n"
            f"Candidates: <b>{rep.get('candidates',0)}</b> | Outcomes: <b>{rep.get('outcomes',0)}</b>\n"
            f"Coverage: <b>{cov.get('coverage_rate',0)*100:.1f}%</b>\n"
            f"Calibration: <b>{cal.get('status','INSUFFICIENT')}</b>\n"
            f"Time-of-day: <b>{te.get('status','INSUFFICIENT')}</b>\n"
            f"Drift: <b>{drift.get('status','UNKNOWN')}</b>\n"
            f"Questions: <b>{len(rep.get('questions',[]))}</b>\n"
            f"Research model: <b>{(rep.get('model_research') or {}).get('status','INSUFFICIENT')}</b>")


_v32_load_state()

# Explicitly advertise the V32 cognitive surface while preserving the stable API.
__all__ = list(dict.fromkeys(__all__ + [
    "V32_VERSION", "get_v32_status", "get_v32_belief", "full_command",
    "_v33_log", "_v33_notify",
    "full_analyze", "manage_position", "ingest_live_candidate", "ingest_live_outcome",
    "full_learning_review", "reset_cognitive_memory", "extract_time_context",
]))


# =============================================================================
# V35 ADAPTIVE AGENT OVERLAY
# =============================================================================
# Design contract:
#   strategy_logic.py = replaceable brain.
#   It may research, learn, rank and propose policy changes, but NEVER calls
#   Binance execution APIs. main.py remains the execution body.
#
# Learning starts immediately from observations. There is NO "wait 30 trades"
# requirement for observation/model bookkeeping. Policy changes still require
# batched evidence, validation and stability; one SL cannot rewrite the brain.
#
# Sources of evidence:
#   1) historical replay (local Binance-compatible OHLCV files),
#   2) live market observations,
#   3) every analyzed live candidate, including below-threshold candidates,
#   4) executed trade outcomes and management events.
#
# Frequency/opportunity is a first-class research metric. The brain is not
# rewarded for simply making the filter stricter and quieter.

AGENT_BRAIN_API_VERSION = "v35-body-brain-contract-1"
AGENT_STATE_DIR = Path(os.getenv("FULL_STATE_DIR", "machine_learning_state"))
AGENT_STATE_FILE = AGENT_STATE_DIR / "adaptive_brain_state.json"
AGENT_POLICY_FILE = AGENT_STATE_DIR / "adaptive_policy.json"
AGENT_HISTORY_DIR = Path(os.getenv("HISTORICAL_DATA_DIR", str(AGENT_STATE_DIR / "historical_data")))
AGENT_HISTORICAL_DAYS = max(30, int(os.getenv("HISTORICAL_LEARNING_DAYS", "90")))
AGENT_RESEARCH_INTERVAL = max(5.0, float(os.getenv("ADAPTIVE_RESEARCH_INTERVAL", "20")))
AGENT_REPLAY_WORKERS = max(1, min(4, int(os.getenv("ADAPTIVE_REPLAY_WORKERS", "2"))))
AGENT_REPLAY_STEP_M15 = max(1, int(os.getenv("ADAPTIVE_REPLAY_STEP_M15", "4")))
AGENT_MIN_POLICY_EVIDENCE = max(8, int(os.getenv("ADAPTIVE_MIN_POLICY_EVIDENCE", "20")))
AGENT_POLICY_MAX_DELTA = max(0.01, min(0.10, float(os.getenv("ADAPTIVE_POLICY_MAX_DELTA", "0.03"))))
AGENT_EXPLORATION_SHARE = max(0.05, min(0.30, float(os.getenv("ADAPTIVE_EXPLORATION_SHARE", "0.15"))))
AGENT_AUTO_HISTORICAL = str(os.getenv("ADAPTIVE_AUTO_HISTORICAL", "1")).strip().lower() not in {"0", "false", "off", "no"}
AGENT_MAX_HISTORY_ROWS = max(5000, int(os.getenv("ADAPTIVE_MAX_HISTORY_ROWS", "300000")))

_AGENT_LOCK = threading.RLock()
_AGENT_STOP = threading.Event()
_AGENT_WAKE = threading.Event()
_AGENT_THREAD = None
_AGENT_HISTORY_THREAD = None
_AGENT_HISTORY_DONE = set()
_AGENT_LAST_ERROR = None
_AGENT_TICKS = 0
_AGENT_LAST_REPORT = {}
_AGENT_STATE = {
    "version": AGENT_BRAIN_API_VERSION,
    "brain_version": V32_VERSION,
    "historical": {"files": 0, "rows": 0, "replay_candidates": 0, "status": "NOT_STARTED"},
    "live": {"observations": 0, "candidates": 0, "outcomes": 0},
    "frequency": {},
    "policy": {},
    "policy_revisions": 0,
    "strategy_revisions": 0,
    "strategy_version": "S1",
    "last_research": None,
}


def _agent_json_load(path, default):
    try:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, type(default)) else default
    except Exception as exc:
        log.warning(f"[ADAPTIVE] load {path.name} gagal: {exc}")
    return default


def _agent_json_save(path, obj):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception as exc:
        log.warning(f"[ADAPTIVE] save {path.name} gagal: {exc}")
        return False


with _AGENT_LOCK:
    _AGENT_STATE.update(_agent_json_load(AGENT_STATE_FILE, {}))
    _AGENT_POLICY = _agent_json_load(AGENT_POLICY_FILE, {})
    if not isinstance(_AGENT_POLICY, dict):
        _AGENT_POLICY = {}


def _agent_strategy_policy_adjustment(archetype, regime):
    """Translate learned policy into a small selection modifier, not a hard gate."""
    key=f"{archetype}|{regime}"
    try:
        raw=float(_AGENT_POLICY.get(key,0.0) or 0.0)
    except (TypeError, ValueError):
        raw=0.0
    delta=max(-0.25,min(0.25,raw))
    return {"key":key,"delta":round(delta,6),"score_adjustment":round(delta*10.0,4),"policy_active":bool(abs(delta)>1e-9)}


def _agent_read_records(limit=AGENT_MAX_HISTORY_ROWS):
    """Read the canonical v32/v35 experience log without touching Binance."""
    rows = []
    try:
        if V32_EXPERIENCE_FILE.exists():
            with V32_EXPERIENCE_FILE.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            rows.append(row)
                    except Exception:
                        continue
    except Exception as exc:
        log.warning(f"[ADAPTIVE] experience read gagal: {exc}")
    return rows[-int(limit):]


def _agent_candidate_cells(records):
    """Compute opportunity/frequency and quality cells from *all* candidates."""
    candidates = [r for r in records if isinstance(r, dict) and r.get("type") == "candidate"]
    cells = defaultdict(list)
    for r in candidates:
        regime = str(r.get("regime") or r.get("market_regime") or "UNKNOWN")
        archetype = str(r.get("archetype") or "UNKNOWN")
        decision = str(r.get("decision") or "UNKNOWN").upper()
        key = (archetype, regime, decision)
        q = float(r.get("trade_quality") or (r.get("confidence") or 0.0))
        cells[key].append(q)
    out = []
    for (archetype, regime, decision), vals in cells.items():
        n = len(vals)
        out.append({
            "archetype": archetype,
            "regime": regime,
            "decision": decision,
            "n": n,
            "avg_quality": round(float(np.mean(vals)), 3) if vals else 0.0,
            "opportunity_share": round(n / max(1, len(candidates)), 5),
        })
    out.sort(key=lambda x: (-x["n"], -x["avg_quality"]))
    return out


def _agent_outcome_cells(records):
    outcomes = [r for r in records if isinstance(r, dict) and r.get("type") == "trade_outcome"]
    cells = defaultdict(list)
    for r in outcomes:
        key = (str(r.get("archetype") or "UNKNOWN"), str(r.get("market_regime") or r.get("regime") or "UNKNOWN"))
        p = r.get("outcome") if isinstance(r.get("outcome"), dict) else r
        try:
            final_r = float(p.get("final_r", 0.0) or 0.0)
        except Exception:
            final_r = 0.0
        cells[key].append(final_r)
    out = []
    for (archetype, regime), vals in cells.items():
        n = len(vals)
        wins = sum(1 for x in vals if x > 0)
        out.append({
            "archetype": archetype,
            "regime": regime,
            "n": n,
            "mean_r": round(float(np.mean(vals)), 5),
            "win_rate": round(wins / max(1, n), 4),
        })
    out.sort(key=lambda x: (x["mean_r"], x["n"]), reverse=True)
    return out


def _agent_frequency_health(candidate_cells):
    total = sum(int(x.get("n", 0) or 0) for x in candidate_cells)
    if not total:
        return {"candidates": 0, "top_cell_share": 0.0, "status": "INSUFFICIENT"}
    top = max((int(x.get("n", 0) or 0) / total for x in candidate_cells), default=0.0)
    return {
        "candidates": total,
        "top_cell_share": round(top, 4),
        "exploration_share_target": AGENT_EXPLORATION_SHARE,
        "status": "HEALTHY" if total >= 20 else "WARMING_UP",
    }


def _agent_bounded_policy_proposal(candidate_cells, outcome_cells):
    """Propose small, auditable policy changes; never mutate core strategy code."""
    proposals = []
    outcome_map = {(x["archetype"], x["regime"]): x for x in outcome_cells if int(x.get("n", 0) or 0) >= AGENT_MIN_POLICY_EVIDENCE}
    for c in candidate_cells:
        key = (c["archetype"], c["regime"])
        o = outcome_map.get(key)
        if not o:
            continue
        mean_r = float(o.get("mean_r", 0.0))
        n = int(o.get("n", 0) or 0)
        # Positive edge => gentle support; negative edge => gentle caution.
        target = float(np.tanh(mean_r)) * 0.03
        target = max(-AGENT_POLICY_MAX_DELTA, min(AGENT_POLICY_MAX_DELTA, target))
        proposals.append({
            "key": f"{c['archetype']}|{c['regime']}",
            "delta": round(target, 5),
            "evidence": n,
            "mean_r": round(mean_r, 5),
            "frequency": int(c.get("n", 0) or 0),
        })
    return proposals


def _agent_apply_policy_proposals(proposals):
    """Apply only bounded, evidence-backed revisions; one trade cannot change it."""
    changed = []
    global _AGENT_POLICY
    with _AGENT_LOCK:
        for p in proposals:
            key = str(p.get("key"))
            n = int(p.get("evidence", 0) or 0)
            if n < AGENT_MIN_POLICY_EVIDENCE:
                continue
            old = float(_AGENT_POLICY.get(key, 0.0) or 0.0)
            delta = max(-AGENT_POLICY_MAX_DELTA, min(AGENT_POLICY_MAX_DELTA, float(p.get("delta", 0.0) or 0.0)))
            # Hysteresis: ignore tiny noise.
            if abs(delta) < 0.005:
                continue
            new = max(-0.25, min(0.25, old + delta * min(1.0, n / 100.0)))
            if abs(new - old) < 0.002:
                continue
            _AGENT_POLICY[key] = round(float(new), 6)
            changed.append({"key": key, "old": round(old, 6), "new": round(new, 6), "evidence": n})
        if changed:
            _AGENT_STATE["policy_revisions"] = int(_AGENT_STATE.get("policy_revisions", 0) or 0) + len(changed)
            _AGENT_STATE["strategy_revisions"] = int(_AGENT_STATE.get("strategy_revisions", 0) or 0) + len(changed)
            old_version = str(_AGENT_STATE.get("strategy_version") or "S1")
            m = re.match(r"S(\d+)", old_version)
            next_num = (int(m.group(1)) + len(changed)) if m else 1 + len(changed)
            _AGENT_STATE["strategy_version"] = f"S{next_num}"
            _AGENT_STATE["policy"] = dict(_AGENT_POLICY)
            _agent_json_save(AGENT_POLICY_FILE, _AGENT_POLICY)
            _agent_json_save(AGENT_STATE_FILE, _AGENT_STATE)
    return changed


def adaptive_research_cycle():
    """Fast research pass. It runs immediately with sparse evidence and scales with data."""
    records = _agent_read_records()
    candidate_cells = _agent_candidate_cells(records)
    outcome_cells = _agent_outcome_cells(records)
    frequency = _agent_frequency_health(candidate_cells)
    proposals = _agent_bounded_policy_proposal(candidate_cells, outcome_cells)
    changed = _agent_apply_policy_proposals(proposals)
    report = {
        "timestamp": time.time(),
        "brain_version": AGENT_BRAIN_API_VERSION,
        "records": len(records),
        "candidate_cells": candidate_cells[:100],
        "outcome_cells": outcome_cells[:100],
        "frequency": frequency,
        "policy_proposals": proposals[:100],
        "policy_changes": changed,
        "strategy_version": str(_AGENT_STATE.get("strategy_version") or "S1"),
        "strategy_revision_count": int(_AGENT_STATE.get("strategy_revisions", 0) or 0),
    }
    with _AGENT_LOCK:
        _AGENT_STATE["frequency"] = frequency
        _AGENT_STATE["live"] = {
            "observations": sum(1 for x in records if x.get("type") == "market_observation"),
            "candidates": sum(1 for x in records if x.get("type") == "candidate"),
            "outcomes": sum(1 for x in records if x.get("type") == "trade_outcome"),
        }
        _AGENT_STATE["last_research"] = time.time()
        _AGENT_STATE["policy"] = dict(_AGENT_POLICY)
        _agent_json_save(AGENT_STATE_FILE, _AGENT_STATE)
    return report


def get_adaptive_status():
    with _AGENT_LOCK:
        return {
            "api_version": AGENT_BRAIN_API_VERSION,
            "full_enabled": bool(_FULL_ENABLED),
            "brain_version": V32_VERSION,
            "worker_alive": bool(_AGENT_THREAD is not None and _AGENT_THREAD.is_alive()),
            "history_worker_alive": bool(_AGENT_HISTORY_THREAD is not None and _AGENT_HISTORY_THREAD.is_alive()),
            "ticks": int(_AGENT_TICKS),
            "last_error": _AGENT_LAST_ERROR,
            "history": dict(_AGENT_STATE.get("historical") or {}),
            "live": dict(_AGENT_STATE.get("live") or {}),
            "frequency": dict(_AGENT_STATE.get("frequency") or {}),
            "policy_revisions": int(_AGENT_STATE.get("policy_revisions", 0) or 0),
            "strategy_revisions": int(_AGENT_STATE.get("strategy_revisions", 0) or 0),
            "strategy_version": str(_AGENT_STATE.get("strategy_version") or "S1"),
            "exploration_share": AGENT_EXPLORATION_SHARE,
        }


def get_adaptive_policy():
    with _AGENT_LOCK:
        return dict(_AGENT_POLICY)


def _load_ohlcv_file(path):
    """Load CSV/JSON/JSONL historical OHLCV without contacting Binance."""
    suffix = path.suffix.lower()
    rows = []
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            df = pd.DataFrame(rows)
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            df = pd.DataFrame(obj if isinstance(obj, list) else obj.get("data", []))
    else:
        return pd.DataFrame()
    if df.empty:
        return df
    rename = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in rename: return rename[n]
        return None
    cols = {k: pick(k, k.replace("_", " ")) for k in ("open_time","timestamp","time","open","high","low","close","volume")}
    ts_col = cols.get("open_time") or cols.get("timestamp") or cols.get("time")
    if not ts_col:
        return pd.DataFrame()
    out = pd.DataFrame({
        "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
        "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
        "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
        "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
        "volume": pd.to_numeric(df[cols["volume"]], errors="coerce"),
    }, index=pd.to_datetime(df[ts_col], unit="ms", utc=True, errors="coerce"))
    if out.index.isna().all():
        out.index = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    out = out[~out.index.isna()].dropna(subset=["open","high","low","close","volume"]).sort_index()
    return out


def _historical_symbol_name(path):
    name = path.stem.upper()
    for token in ("-15M", "_15M", "15M", "-M15", "_M15", "M15"):
        if token in name:
            return name.split(token)[0].replace("-", "").replace("_", "") or name
    return name.split("-")[0].split("_")[0]


def _replay_one_history_file(path):
    """Replay one M15-ish OHLCV file; generate observation/candidate evidence only."""
    try:
        df = _load_ohlcv_file(path)
        if df is None or len(df) < 320:
            return {"file": str(path), "rows": 0, "candidates": 0, "status": "SKIPPED"}
        # Respect requested historical horizon without relying on current Binance REST.
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=AGENT_HISTORICAL_DAYS)
        df = df.loc[df.index >= cutoff]
        if len(df) < 320:
            return {"file": str(path), "rows": int(len(df)), "candidates": 0, "status": "SKIPPED_SHORT"}
        h1 = df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        d1 = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        symbol = _historical_symbol_name(path)
        emitted = 0
        step = AGENT_REPLAY_STEP_M15
        # Point-in-time replay: only rows up to t are visible to the analysis.
        for i in range(300, len(df) - 32, step):
            m15_now = df.iloc[:i].copy()
            now = m15_now.index[-1]
            h1_now = h1.loc[:now].copy()
            d1_now = d1.loc[:now].copy()
            if len(h1_now) < 80 or len(d1_now) < 20:
                continue
            try:
                result = _V31_BASE_FULL_ANALYZE(h1_now, m15_now, d1_now, symbol=symbol, df_btc_h1=None, trade_history=None)
            except Exception:
                continue
            # The original engine can return None; that itself is a useful scan observation.
            try:
                feats = extract_market_features(h1_now, m15_now, d1_now, None)
                _v32_record_experience({"type":"historical_market_observation","timestamp":time.time(),"historical_timestamp":str(now),"symbol":symbol,"source":"historical_replay","features":feats})
            except Exception:
                pass
            if isinstance(result, dict):
                # Label the hypothetical candidate with forward path information without
                # allowing those future values to enter the analysis itself.
                entry = float(result.get("entry") or result.get("price") or 0.0)
                direction = str(result.get("decision") or "BUY").upper()
                future = df.iloc[i:min(len(df), i + 32)]
                if entry > 0 and not future.empty:
                    if direction == "BUY":
                        mfe = max(0.0, float(future["high"].max()) - entry)
                        mae = max(0.0, entry - float(future["low"].min()))
                    else:
                        mfe = max(0.0, entry - float(future["low"].min()))
                        mae = max(0.0, float(future["high"].max()) - entry)
                    risk = abs(float(result.get("entry") or entry) - float(result.get("sl") or entry)) or np.nan
                    mfe_r = float(mfe / risk) if np.isfinite(risk) and risk > 0 else 0.0
                    mae_r = float(mae / risk) if np.isfinite(risk) and risk > 0 else 0.0
                else:
                    mfe_r = mae_r = 0.0
                result = dict(result)
                result["historical_timestamp"] = str(now)
                result["source"] = "historical_replay"
                result["shadow"] = True
                result["shadow_forward_bars"] = 32
                result["shadow_mfe_r"] = round(mfe_r, 4)
                result["shadow_mae_r"] = round(mae_r, 4)
                result["rejected_reason"] = "historical_shadow"
                ingest_live_candidate(result, h1=h1_now, m15=m15_now, d1=d1_now, rejected_reason="historical_shadow", source="historical_replay")
                emitted += 1
        return {"file": str(path), "rows": int(len(df)), "candidates": emitted, "status": "OK"}
    except Exception as exc:
        return {"file": str(path), "rows": 0, "candidates": 0, "status": "ERROR", "error": str(exc)[:300]}


def adaptive_replay_historical(force=False):
    """Replay local historical files in parallel; never uses Binance API."""
    global _AGENT_HISTORY_DONE
    if not AGENT_HISTORY_DIR.exists():
        with _AGENT_LOCK:
            _AGENT_STATE["historical"] = {"files": 0, "rows": 0, "replay_candidates": 0, "status": "NO_DIRECTORY"}
            _agent_json_save(AGENT_STATE_FILE, _AGENT_STATE)
        return dict(_AGENT_STATE["historical"])
    paths = [p for p in sorted(AGENT_HISTORY_DIR.rglob("*")) if p.suffix.lower() in {".csv", ".json", ".jsonl"}]
    if not force:
        paths = [p for p in paths if str(p) not in _AGENT_HISTORY_DONE]
    if not paths:
        with _AGENT_LOCK:
            _AGENT_STATE["historical"] = {"files": len(_AGENT_HISTORY_DONE), "rows": _AGENT_STATE.get("historical",{}).get("rows",0), "replay_candidates": _AGENT_STATE.get("historical",{}).get("replay_candidates",0), "status": "UP_TO_DATE"}
            _agent_json_save(AGENT_STATE_FILE, _AGENT_STATE)
        return dict(_AGENT_STATE["historical"])
    results=[]
    with ThreadPoolExecutor(max_workers=AGENT_REPLAY_WORKERS, thread_name_prefix="hist-replay") as ex:
        futs={ex.submit(_replay_one_history_file,p):p for p in paths}
        for fut in as_completed(futs):
            try: results.append(fut.result())
            except Exception as exc: results.append({"file":str(futs[fut]),"status":"ERROR","error":str(exc)[:300]})
    _AGENT_HISTORY_DONE.update(str(x.get("file")) for x in results if x.get("status") in {"OK","SKIPPED","SKIPPED_SHORT"})
    with _AGENT_LOCK:
        hist=_AGENT_STATE.get("historical") if isinstance(_AGENT_STATE.get("historical"),dict) else {}
        _AGENT_STATE["historical"]={
            "files": int(hist.get("files",0) or 0)+len(results),
            "rows": int(hist.get("rows",0) or 0)+sum(int(x.get("rows",0) or 0) for x in results),
            "replay_candidates": int(hist.get("replay_candidates",0) or 0)+sum(int(x.get("candidates",0) or 0) for x in results),
            "status": "RUNNING" if results else "IDLE",
            "last_results": results[-20:],
        }
        _agent_json_save(AGENT_STATE_FILE, _AGENT_STATE)
        return dict(_AGENT_STATE["historical"])


def _adaptive_worker_loop():
    global _AGENT_TICKS, _AGENT_LAST_ERROR
    log.info("[ADAPTIVE] brain worker started")
    while not _AGENT_STOP.is_set():
        try:
            if not _FULL_ENABLED:
                _AGENT_WAKE.wait(5)
                _AGENT_WAKE.clear()
                continue
            if AGENT_AUTO_HISTORICAL and _AGENT_TICKS == 0:
                try:
                    adaptive_replay_historical(force=False)
                except Exception:
                    log.exception("[ADAPTIVE] historical bootstrap gagal")
            _AGENT_LAST_REPORT = adaptive_research_cycle()
            _AGENT_TICKS += 1
            _AGENT_LAST_ERROR = None
        except Exception as exc:
            _AGENT_LAST_ERROR = str(exc)[:500]
            log.exception("[ADAPTIVE] research cycle gagal")
        _AGENT_WAKE.wait(AGENT_RESEARCH_INTERVAL)
        _AGENT_WAKE.clear()
    log.info("[ADAPTIVE] brain worker stopped")


def adaptive_agent_start():
    """Idempotent start. Called by main.py at boot and after brain hot-swap."""
    global _AGENT_THREAD, _AGENT_STOP
    with _AGENT_LOCK:
        if _AGENT_THREAD is not None and _AGENT_THREAD.is_alive():
            _AGENT_WAKE.set()
            return get_adaptive_status()
        _AGENT_STOP.clear()
        _AGENT_WAKE.set()
        _AGENT_THREAD = threading.Thread(target=_adaptive_worker_loop, name="adaptive-brain", daemon=True)
        _AGENT_THREAD.start()
    return get_adaptive_status()


def adaptive_agent_stop():
    with _AGENT_LOCK:
        _AGENT_STOP.set()
        _AGENT_WAKE.set()
    return get_adaptive_status()


# FULL workers are operator-controlled; no research thread is spawned at import.

# Explicit public API for the stable execution body.
try:
    __all__ = list(dict.fromkeys(__all__ + [
        "AGENT_BRAIN_API_VERSION", "adaptive_agent_start", "adaptive_agent_stop",
        "adaptive_research_cycle", "adaptive_replay_historical", "get_adaptive_status",
        "get_adaptive_policy",
    ]))
except Exception:
    pass
