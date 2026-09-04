"""
strategy_logic.py — STRATEGY BRAIN V2 / V127
================================================
Clean rebuild of the trading brain while preserving the stable main.py contract.

Goals
-----
1. Deterministic market understanding first: regime, trend strength, structure,
   liquidity, POI, entry location, risk geometry.
2. Frequency-aware decisions: avoid a beautiful strategy that almost never emits
   signals; threshold adapts only inside safety bounds and never fabricates setups.
3. Recency-aware learning: recent evidence dominates. Default effective half-life
   is 72h, with a hard age ceiling and context-drift tightening. Old data is used
   only when evidence is sparse and still similar.
4. FULL is an experience engine, not a second execution engine. It learns from
   candidates, rejected candidates, outcomes, drawdown, trail/protection events,
   and scan frequency.
5. Optional Ollama critic using OLLAMA_API_KEY + OLLAMA_MODEL. The LLM may critique
   a structured candidate but can never place/cancel/modify an order.
6. manage_position remains brain-owned for TP/trailing recommendations; main.py
   remains the actual execution authority.

Stable public contracts retained for main.py
---------------------------------------------
- full_analyze(df_h1, df_m15, df_d1=None, symbol=None, **kwargs)
- manage_position(state, df_m15, df_h1=None, df_d1=None, symbol=None, **kwargs)
- score_direction(df_h1, df_m15, df_d1=None)
- swing_pts(df, lb)
- MIN_RR, MAX_RR, TRAIL_R_LADDER, STRUCT_TRAIL_LB,
  STRUCT_TRAIL_BUF_PCT, STRUCT_TRAIL_LOOKBACK, FIB_EXT_1, FIB_EXT_2
- record_candidate_observation / ingest_live_candidate
- record_trade_outcome / ingest_live_outcome
- record_scan_summary
- evaluate_stats_decision
- record_protection_event
- full_command, adaptive_agent_start, adaptive_agent_stop
- get_learning_schema, get_cognitive_status, get_full_cognitive_status
- export_checkpoint_state / import_checkpoint_state
- set_learning_model
- get_active_confidence_threshold / set_manual_confidence_threshold

The module is self-contained and does not call Binance.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# VERSION / CONTRACT
# -----------------------------------------------------------------------------
FINAL_BRAIN_VERSION = "V133_STRATEGY_BRAIN_PENDING_ENTRY_AWARE"
BRAIN_INTERFACE_VERSION = "V128_COIN_ROTATION_BRAIN_PROGRESS"
V35_VERSION = "V128_COIN_ROTATION_BRAIN_PROGRESS"
V32_VERSION = "V128_COIN_ROTATION_BRAIN_PROGRESS"
FULL_LEARNING_SCHEMA = "full_learning_v3_strategy_brain_v2"
MACHINE_LEARNING_SCHEMA = "machine_learning_v4_strategy_brain_v2"
BRAIN_CHECKPOINT_SCHEMA = "brain_progress_checkpoint_v2"

# Main.py imports these directly.
MIN_RR = 2.0
MAX_RR = None
TRAIL_R_LADDER: list = []
STRUCT_TRAIL_LB = 3
STRUCT_TRAIL_BUF_PCT = 0.0025
STRUCT_TRAIL_LOOKBACK = 60
FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618

# -----------------------------------------------------------------------------
# CONFIG: FREQUENCY, RECENCY, OLLAMA
# -----------------------------------------------------------------------------
FULL_ENABLED = False
FULL_LEARNING_ACTIVE = False

# Signal frequency target. These are intentionally ranges, not promises.
FREQUENCY_TARGET_LOW = 0.05
FREQUENCY_TARGET_HIGH = 0.18
FREQUENCY_TARGET_IDEAL = 0.10
CONFIDENCE_BASE = 65.0
CONFIDENCE_SAFE_MIN = 55.0
CONFIDENCE_SAFE_MAX = 82.0
CONFIDENCE_ADAPT_STEP = 2.0
FREQUENCY_WINDOW = 80

# Recency: 72h is the default half-life, but effective windows can tighten with
# regime/context drift. Hard ceiling prevents three-month-old evidence from
# quietly influencing today's decision.
RECENCY_HALF_LIFE_HOURS = 72.0
RECENCY_DEFAULT_MAX_AGE_HOURS = 24.0 * 30.0
RECENCY_SAFE_MIN_AGE_HOURS = 24.0 * 3.0
RECENCY_EXTEND_IF_SPARSE = True
RECENCY_MIN_SIMILAR_EXAMPLES = 12
RECENCY_CONTEXT_DRIFT_TIGHTEN = True

# Ollama: optional critic only; failure is fail-open for trading logic.
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b").strip() or "gpt-oss:20b"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://ollama.com/api/chat").strip()
OLLAMA_ENABLED = str(os.getenv("OLLAMA_ENABLED", "auto")).strip().lower() not in {"0", "false", "off", "no"}
OLLAMA_MIN_CONFIDENCE = float(os.getenv("OLLAMA_MIN_CONFIDENCE", "72"))
OLLAMA_MAX_CALLS_PER_MIN = int(os.getenv("OLLAMA_MAX_CALLS_PER_MIN", "8"))
OLLAMA_CACHE_SEC = float(os.getenv("OLLAMA_CACHE_SEC", "90"))
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "15"))

# State persistence. main.py checkpoint remains authoritative for runtime save/open,
# but local persistence keeps the brain alive between normal restarts too.
STATE_DIR = Path(os.getenv("FULL_MODEL_DIR", "machine_learning_state"))
STATE_FILE = STATE_DIR / "strategy_brain_v2_state.json"

_LOCK = threading.RLock()
_AGENT_LOCK = _LOCK
_HISTORY: deque = deque(maxlen=5000)
_SCAN_HISTORY: deque = deque(maxlen=FREQUENCY_WINDOW)
_PROTECTION_EVENTS: deque = deque(maxlen=500)
_OLLAMA_CALL_TIMES: deque = deque(maxlen=100)
_OLLAMA_CACHE: dict[str, tuple[float, dict]] = {}
_FULL_THREAD: Optional[threading.Thread] = None
_FULL_STOP = threading.Event()
_FULL_WAKE = threading.Event()
_AGENT_TICKS = 0
_MANUAL_THRESHOLD: Optional[float] = None
_ADAPTIVE_THRESHOLD = CONFIDENCE_BASE
_LEARNED_MODEL: Optional[dict] = None
_STRATEGY_STATE = {
    "version": "S2.0",
    "revisions": 0,
    "last_reason": "startup",
    "last_update_at": 0.0,
    "champion": "S2.0",
    "challengers": [],
}


# -----------------------------------------------------------------------------
# LIGHTWEIGHT DATA OBJECTS
# -----------------------------------------------------------------------------
class MarketState:
    __slots__ = (
        "symbol", "macro_bias", "htf_bias", "m15_bias", "regime",
        "trend_strength", "volatility", "structure_strength",
        "liquidity_state", "range_position", "data_quality",
        "relative_volume", "timestamp"
    )
    def __init__(self, symbol, macro_bias, htf_bias, m15_bias, regime, trend_strength,
                 volatility, structure_strength, liquidity_state, range_position,
                 data_quality, relative_volume, timestamp):
        self.symbol=symbol; self.macro_bias=macro_bias; self.htf_bias=htf_bias
        self.m15_bias=m15_bias; self.regime=regime; self.trend_strength=float(trend_strength)
        self.volatility=float(volatility); self.structure_strength=float(structure_strength)
        self.liquidity_state=liquidity_state; self.range_position=float(range_position)
        self.data_quality=float(data_quality); self.relative_volume=float(relative_volume)
        self.timestamp=float(timestamp)
    def to_dict(self):
        return {k:getattr(self,k) for k in self.__slots__}

class Candidate:
    __slots__ = (
        "direction", "entry", "sl", "tp", "rr", "entry_label", "confidence",
        "setup_quality", "location_score", "trend_strength", "structure_strength",
        "liquidity_score", "htf_alignment", "macro_alignment", "poi_reacted",
        "trigger_confirmed", "reasons", "invalidations"
    )
    def __init__(self, direction, entry, sl, tp, rr, entry_label, confidence,
                 setup_quality, location_score, trend_strength, structure_strength,
                 liquidity_score, htf_alignment, macro_alignment, poi_reacted,
                 trigger_confirmed, reasons=None, invalidations=None):
        self.direction=str(direction); self.entry=float(entry); self.sl=float(sl); self.tp=float(tp); self.rr=float(rr)
        self.entry_label=str(entry_label); self.confidence=float(confidence); self.setup_quality=float(setup_quality)
        self.location_score=float(location_score); self.trend_strength=float(trend_strength)
        self.structure_strength=float(structure_strength); self.liquidity_score=float(liquidity_score)
        self.htf_alignment=float(htf_alignment); self.macro_alignment=float(macro_alignment)
        self.poi_reacted=bool(poi_reacted); self.trigger_confirmed=bool(trigger_confirmed)
        self.reasons=list(reasons or []); self.invalidations=list(invalidations or [])
    def to_dict(self):
        return {k:getattr(self,k) for k in self.__slots__}


# -----------------------------------------------------------------------------
# SAFE HELPERS
# -----------------------------------------------------------------------------
def _safe_float(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _clip(x, lo, hi):
    return max(lo, min(hi, _safe_float(x, lo)))


def _json_safe(v):
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, deque)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _now():
    return time.time()


def _latest_ts(row: dict) -> float:
    for key in ("timestamp", "scan_time", "exit_time", "entry_time", "recorded_at", "time"):
        v = _safe_float(row.get(key), 0.0)
        if v > 0:
            return v
    return _now()


def _side_sign(direction: str) -> float:
    return 1.0 if str(direction or "BUY").upper() == "BUY" else -1.0


def _price_return(entry: float, price: float, direction: str) -> float:
    if entry <= 0:
        return 0.0
    return ((price - entry) / entry) * _side_sign(direction)


# -----------------------------------------------------------------------------
# TECHNICAL CORE
# -----------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.astype(float).ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.astype(float).diff()
    gain = d.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    out = pd.Series(50.0, index=s.index, dtype=float)
    valid = gain.notna() & loss.notna()
    both_zero = valid & (gain <= 1e-12) & (loss <= 1e-12)
    gain_only = valid & (loss <= 1e-12) & (gain > 1e-12)
    loss_only = valid & (gain <= 1e-12) & (loss > 1e-12)
    normal = valid & (gain > 1e-12) & (loss > 1e-12)
    out.loc[both_zero] = 50.0
    out.loc[gain_only] = 100.0
    out.loc[loss_only] = 0.0
    rs = gain.loc[normal] / loss.loc[normal]
    out.loc[normal] = 100.0 - 100.0 / (1.0 + rs)
    return out


def atr_fn(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _closed_candles(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    idx = out.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    boundary = pd.Timestamp.now(tz="UTC").floor(f"{int(interval_minutes)}min")
    if idx[-1] < boundary:
        return out
    return out.loc[idx < boundary].copy()


def build_df(df: pd.DataFrame, interval_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 60:
        return None
    out = df.copy()
    if interval_minutes:
        out = _closed_candles(out, interval_minutes)
    if out is None or len(out) < 60:
        return None
    for c in ("open", "high", "low", "close", "volume"):
        if c not in out.columns:
            return None
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["ema9"] = ema(out["close"], 9)
    out["ema21"] = ema(out["close"], 21)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200) if len(out) >= 200 else ema(out["close"], 50)
    out["rsi"] = rsi(out["close"])
    out["atr"] = atr_fn(out)
    out["vol_sma"] = out["volume"].rolling(20).mean()
    out = out.dropna(subset=["ema9", "ema21", "ema50", "ema200", "atr", "vol_sma"])
    if len(out) < 30:
        return None
    out["rsi"] = out["rsi"].fillna(50.0).clip(0.0, 100.0)
    return out


def swing_pts(df: pd.DataFrame, lb: int = 5):
    if df is None or len(df) < max(2 * lb + 1, 5):
        return [], []
    sh, sl = [], []
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    for i in range(lb, len(df) - lb):
        if high[i] >= np.max(high[i - lb:i + lb + 1]):
            sh.append(i)
        if low[i] <= np.min(low[i - lb:i + lb + 1]):
            sl.append(i)
    return sh, sl


def _market_structure(df: pd.DataFrame, sh: list, sl: list) -> str:
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


def mkt_struct(df: pd.DataFrame, sh: list, sl: list) -> str:
    return _market_structure(df, sh, sl)


def fib_position(price: float, swing_low: float, swing_high: float) -> float:
    rng = swing_high - swing_low
    if rng <= 0:
        return 0.5
    return _clip((price - swing_low) / rng, 0.0, 1.0)


def _trend_strength(df: pd.DataFrame, sh: list, sl: list) -> float:
    if df is None or len(df) < 20:
        return 0.0
    struct = _market_structure(df, sh, sl)
    score = 50.0
    if struct == "bullish":
        score += 15
    elif struct == "bearish":
        score += 15
    else:
        score -= 10
    # Slope of confirmed extremes per candle, normalized by ATR.
    try:
        if struct == "bullish" and len(sh) >= 3:
            p1 = float(df["high"].iloc[sh[-3]])
            p2 = float(df["high"].iloc[sh[-2]])
            p3 = float(df["high"].iloc[sh[-1]])
            dt1 = max(1, sh[-2] - sh[-3]); dt2 = max(1, sh[-1] - sh[-2])
            impulse_accel = ((p3 - p2) / dt2) - ((p2 - p1) / dt1)
            atr = _safe_float(df["atr"].iloc[-1], 0.0)
            if atr > 0:
                score += _clip(impulse_accel / atr * 1000.0, -15, 15)
        elif struct == "bearish" and len(sl) >= 3:
            p1 = float(df["low"].iloc[sl[-3]])
            p2 = float(df["low"].iloc[sl[-2]])
            p3 = float(df["low"].iloc[sl[-1]])
            dt1 = max(1, sl[-2] - sl[-3]); dt2 = max(1, sl[-1] - sl[-2])
            impulse_accel = ((p2 - p3) / dt2) - ((p1 - p2) / dt1)
            atr = _safe_float(df["atr"].iloc[-1], 0.0)
            if atr > 0:
                score += _clip(impulse_accel / atr * 1000.0, -15, 15)
    except Exception:
        pass
    # EMA ordering and volume participation.
    last = df.iloc[-1]
    bull_ema = last["ema9"] > last["ema21"] > last["ema50"]
    bear_ema = last["ema9"] < last["ema21"] < last["ema50"]
    if bull_ema or bear_ema:
        score += 8
    rel_vol = _safe_float(last["volume"], 0.0) / max(_safe_float(last["vol_sma"], 1.0), 1e-12)
    score += _clip((rel_vol - 1.0) * 8.0, -8, 8)
    return _clip(score, 0, 100)


def _macro_bias(df_btc_h1: Optional[pd.DataFrame]) -> str:
    btc = build_df(df_btc_h1, 60) if df_btc_h1 is not None else None
    if btc is None or len(btc) < 50:
        return "unknown"
    sh, sl = swing_pts(btc, 5)
    struct = _market_structure(btc, sh, sl)
    last = btc.iloc[-1]
    if struct == "bullish" or last["ema9"] > last["ema21"] > last["ema50"]:
        return "bullish"
    if struct == "bearish" or last["ema9"] < last["ema21"] < last["ema50"]:
        return "bearish"
    return "ranging"


# -----------------------------------------------------------------------------
# SMC / ICT DETECTORS
# -----------------------------------------------------------------------------
def detect_bos(df: pd.DataFrame, sh: list, sl: list) -> dict:
    out = {"bullish_bos": False, "bearish_bos": False, "level": None}
    if len(sh) < 1 or len(sl) < 1 or len(df) < 3:
        return out
    close = _safe_float(df["close"].iloc[-1])
    prev_close = _safe_float(df["close"].iloc[-2])
    hi = _safe_float(df["high"].iloc[sh[-1]])
    lo = _safe_float(df["low"].iloc[sl[-1]])
    out["bullish_bos"] = close > hi and prev_close <= hi
    out["bearish_bos"] = close < lo and prev_close >= lo
    out["level"] = hi if out["bullish_bos"] else lo if out["bearish_bos"] else None
    return out


def detect_choch(df: pd.DataFrame, sh: list, sl: list) -> dict:
    out = {"bullish_choch": False, "bearish_choch": False}
    if len(sh) < 2 or len(sl) < 2:
        return out
    struct = _market_structure(df, sh, sl)
    close = _safe_float(df["close"].iloc[-1])
    last_hi = _safe_float(df["high"].iloc[sh[-1]])
    last_lo = _safe_float(df["low"].iloc[sl[-1]])
    if struct == "bearish" and close > last_hi:
        out["bullish_choch"] = True
    if struct == "bullish" and close < last_lo:
        out["bearish_choch"] = True
    return out


def detect_cisd(df: pd.DataFrame, lb: int = 8) -> dict:
    out = {"bullish_cisd": False, "bearish_cisd": False}
    if df is None or len(df) < lb + 1:
        return out
    sub = df.iloc[-lb:]
    o, c = sub["open"].to_numpy(float), sub["close"].to_numpy(float)
    if c[-1] > o[-1]:
        run = 0
        for j in range(len(c) - 2, -1, -1):
            if c[j] < o[j]: run += 1
            else: break
        if run >= 3:
            first = len(c) - 1 - run
            mid = (o[first] + c[first]) / 2.0
            out["bullish_cisd"] = c[-1] > mid
    elif c[-1] < o[-1]:
        run = 0
        for j in range(len(c) - 2, -1, -1):
            if c[j] > o[j]: run += 1
            else: break
        if run >= 3:
            first = len(c) - 1 - run
            mid = (o[first] + c[first]) / 2.0
            out["bearish_cisd"] = c[-1] < mid
    return out


def detect_liquidity_sweep(df: pd.DataFrame, sh: list, sl: list, direction: str) -> dict:
    if direction == "bull" and sl:
        level = _safe_float(df["low"].iloc[sl[-1]])
        low = _safe_float(df["low"].iloc[-1]); close = _safe_float(df["close"].iloc[-1])
        if low < level and close > level:
            return {"type":"sellside_sweep", "level":level, "strength":_clip((level-low)/max(abs(level),1e-9)*1000, 1, 3)}
    if direction == "bear" and sh:
        level = _safe_float(df["high"].iloc[sh[-1]])
        high = _safe_float(df["high"].iloc[-1]); close = _safe_float(df["close"].iloc[-1])
        if high > level and close < level:
            return {"type":"buyside_sweep", "level":level, "strength":_clip((high-level)/max(abs(level),1e-9)*1000, 1, 3)}
    return {"type":"none", "level":None, "strength":0}


def detect_inducement(df: pd.DataFrame, direction: str, lb: int = 40) -> dict:
    if df is None or len(df) < 15:
        return {"found":False,"swept":False,"level":None}
    sub = df.iloc[-min(lb, len(df)):].reset_index(drop=True)
    sh, sl = swing_pts(sub, 2)
    if direction == "bull" and sl:
        lvl = _safe_float(sub["low"].iloc[sl[-1]])
        aft = sub.iloc[sl[-1]+1:]
        return {"found":True,"swept":bool((aft["low"]<lvl).any()),"level":lvl}
    if direction == "bear" and sh:
        lvl = _safe_float(sub["high"].iloc[sh[-1]])
        aft = sub.iloc[sh[-1]+1:]
        return {"found":True,"swept":bool((aft["high"]>lvl).any()),"level":lvl}
    return {"found":False,"swept":False,"level":None}


def detect_fvg(df: pd.DataFrame, direction: str, lb: int = 60) -> list[dict]:
    if df is None or len(df) < 5:
        return []
    sub = df.iloc[-min(lb, len(df)):]
    base = len(df) - len(sub)
    out = []
    for i in range(len(sub)-2):
        c0, c2 = sub.iloc[i], sub.iloc[i+2]
        gap = None
        if direction == "bull" and c2["low"] > c0["high"]:
            gap = (float(c2["low"]), float(c0["high"]))
        elif direction == "bear" and c2["high"] < c0["low"]:
            gap = (float(c0["low"]), float(c2["high"]))
        if gap:
            top, bot = max(gap), min(gap)
            # Freshness by closes after formation, not mere wick touch.
            formed = base+i+2
            post = df.iloc[formed+1:]
            if direction == "bull":
                fresh = not bool((post["close"] < bot).any())
            else:
                fresh = not bool((post["close"] > top).any())
            out.append({"top":top,"bot":bot,"mid":(top+bot)/2,"idx":formed,"fresh":fresh})
    return [x for x in out if x["fresh"]][-5:]


def detect_order_blocks(df: pd.DataFrame, direction: str, lb: int = 80) -> list[dict]:
    if df is None or len(df) < 20:
        return []
    sub = df.iloc[-min(lb, len(df)):]
    base = len(df)-len(sub)
    atr = _safe_float(df["atr"].iloc[-1], 0.0)
    body_avg = _safe_float((sub["close"]-sub["open"]).abs().mean(), 1e-9)
    sh, sl = swing_pts(df, 5)
    swing_h = _safe_float(df["high"].iloc[sh[-1]], 0.0) if sh else None
    swing_l = _safe_float(df["low"].iloc[sl[-1]], 0.0) if sl else None
    zones=[]
    for i in range(1, len(sub)-3):
        c, nxt = sub.iloc[i], sub.iloc[i+1]
        bullish_pair = c["close"] < c["open"] and nxt["close"] > nxt["open"]
        bearish_pair = c["close"] > c["open"] and nxt["close"] < nxt["open"]
        if direction == "bull" and not bullish_pair: continue
        if direction == "bear" and not bearish_pair: continue
        impulse = abs(float(nxt["close"]-nxt["open"]))
        if impulse < body_avg*1.2: continue
        top = max(float(c["open"]), float(c["close"]))
        bot = min(float(c["open"]), float(c["close"]))
        idx = base+i
        post = df.iloc[idx+2:]
        if direction == "bull":
            fresh = not bool((post["close"] < bot).any())
        else:
            fresh = not bool((post["close"] > top).any())
        if not fresh: continue
        mid=(top+bot)/2
        fib=None
        if swing_l is not None and swing_h is not None and swing_h>swing_l:
            fib=fib_position(mid,swing_l,swing_h)
        q=50
        q += 10 if impulse >= body_avg*1.5 else 0
        q += 8 if impulse >= body_avg*2.5 else 0
        if atr>0: q += _clip(impulse/atr*8,0,12)
        if fib is not None:
            if direction=="bull" and fib<=0.618: q+=8
            if direction=="bear" and fib>=0.382: q+=8
        if idx>=len(df)-20: q+=5
        zones.append({"top":top,"bot":bot,"mid":mid,"idx":idx,"quality":_clip(q,0,100),"fib":fib})
    zones.sort(key=lambda z:(-z["quality"],-z["idx"]))
    return zones[:5]


def _entry_location(df: pd.DataFrame, direction: str, entry: float) -> dict:
    lb=min(24,len(df))
    sub=df.iloc[-lb:]
    hi=_safe_float(sub["high"].max(), entry)
    lo=_safe_float(sub["low"].min(), entry)
    rp=fib_position(entry,lo,hi)
    last_rsi=_safe_float(df["rsi"].iloc[-1],50)
    prev_rsi=_safe_float(df["rsi"].iloc[-2],last_rsi)
    score=70.0
    reasons=[]
    if direction=="bull":
        if rp<=0.55: score+=10; reasons.append("DISCOUNT_LOCATION")
        if rp>=0.82: score-=22; reasons.append("CHASE_HIGH")
        if last_rsi>=prev_rsi: score+=5
        elif last_rsi<48 and last_rsi<prev_rsi: score-=12; reasons.append("RSI_AGAINST_ENTRY")
    else:
        if rp>=0.45: score+=10; reasons.append("PREMIUM_LOCATION")
        if rp<=0.18: score-=22; reasons.append("CHASE_LOW")
        if last_rsi<=prev_rsi: score+=5
        elif last_rsi>52 and last_rsi>prev_rsi: score-=12; reasons.append("RSI_AGAINST_ENTRY")
    return {"location_score":_clip(score,0,100),"range_position":rp,"rsi_timing":"aligned" if score>=75 else "neutral" if score>=55 else "against","reasons":reasons}


def _confirmation(df: pd.DataFrame, direction: str) -> dict:
    sh, sl = swing_pts(df, 3)
    bos=detect_bos(df,sh,sl)
    choch=detect_choch(df,sh,sl)
    cisd=detect_cisd(df,8)
    atr=_safe_float(df["atr"].iloc[-1],0.0)
    body=abs(_safe_float(df["close"].iloc[-1])-_safe_float(df["open"].iloc[-1]))
    displacement=(body/max(atr,1e-12)) if atr else 0.0
    bullish = bool(bos["bullish_bos"] or choch["bullish_choch"] or cisd["bullish_cisd"]) and displacement>=0.20
    bearish = bool(bos["bearish_bos"] or choch["bearish_choch"] or cisd["bearish_cisd"]) and displacement>=0.20
    ok = bullish if direction=="bull" else bearish
    return {"confirmed":ok,"bos":bos,"choch":choch,"cisd":cisd,"displacement_atr":displacement}


# -----------------------------------------------------------------------------
# MARKET STATE / DIRECTION
# -----------------------------------------------------------------------------
def _frame_bias(df: Optional[pd.DataFrame], interval_minutes: Optional[int]) -> tuple[str,float,float,float]:
    d=build_df(df, interval_minutes)
    if d is None:
        return "unknown",0.0,0.0,0.0
    sh,sl=swing_pts(d,5)
    struct=_market_structure(d,sh,sl)
    trend=_trend_strength(d,sh,sl)
    last=d.iloc[-1]
    ema_bull=bool(last["ema9"]>last["ema21"]>last["ema50"])
    ema_bear=bool(last["ema9"]<last["ema21"]<last["ema50"])
    if struct=="bullish" and ema_bull: bias="bullish"
    elif struct=="bearish" and ema_bear: bias="bearish"
    elif struct=="bullish" or ema_bull: bias="bullish"
    elif struct=="bearish" or ema_bear: bias="bearish"
    else: bias="ranging"
    rel_vol=_safe_float(last["volume"],0)/max(_safe_float(last["vol_sma"],1),1e-9)
    return bias,trend,_clip(rel_vol/2.0,0,2),_clip(_safe_float(last["atr"],0)/max(_safe_float(last["close"],1),1e-9)*100,0,25)


def _market_state(symbol, h1, m15, d1, btc_h1) -> tuple[MarketState, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict]:
    h1d=build_df(h1,60); m15d=build_df(m15,15); d1d=build_df(d1,1440) if d1 is not None else None
    hb, ht, hvol, hv = _frame_bias(h1,60)
    mb, mt, mvol, mv = _frame_bias(m15,15)
    db, dt, dvol, dv = _frame_bias(d1,1440) if d1 is not None else ("unknown",0,0,0)
    macro=_macro_bias(btc_h1)
    trend=_clip((ht*0.55+mt*0.35+dt*0.10) if d1d is not None else (ht*0.60+mt*0.40),0,100)
    sh,sl=(swing_pts(m15d,5) if m15d is not None else ([],[]))
    struct_strength=_clip((60 if mb in {"bullish","bearish"} else 35)+(trend-50)*0.35,0,100)
    last=m15d.iloc[-1] if m15d is not None else (h1d.iloc[-1] if h1d is not None else None)
    rp=0.5
    if last is not None and m15d is not None:
        rp=fib_position(_safe_float(last["close"]),_safe_float(m15d["low"].tail(24).min()),_safe_float(m15d["high"].tail(24).max()))
    if hb=="bullish" and trend>=72: regime="BULL_TREND_STRONG"
    elif hb=="bearish" and trend>=72: regime="BEAR_TREND_STRONG"
    elif hb=="bullish": regime="BULL_TREND_WEAK"
    elif hb=="bearish": regime="BEAR_TREND_WEAK"
    elif mb in {"bullish","bearish"}: regime="TRANSITION"
    else: regime="RANGING"
    liq="UNKNOWN"
    if m15d is not None:
        bull_sw=detect_liquidity_sweep(m15d,sh,sl,"bull")
        bear_sw=detect_liquidity_sweep(m15d,sh,sl,"bear")
        if bull_sw["type"]!="none": liq="SELLSIDE_SWEPT"
        elif bear_sw["type"]!="none": liq="BUYSIDE_SWEPT"
        else: liq="UNTAKEN"
    quality=1.0 if h1d is not None and m15d is not None else 0.7 if h1d is not None else 0.0
    if d1d is None: quality*=0.95
    state=MarketState(symbol,macro,hb,mb,regime,trend,mv+hv,struct_strength,liq,rp,quality,mvol,_now())
    extra={"d1_bias":db,"d1_strength":dt,"h1_strength":ht,"m15_strength":mt,"h1_df":h1d,"m15_df":m15d,"d1_df":d1d}
    return state,h1d,m15d,d1d,extra


def score_direction(df_h1, df_m15, df_d1=None):
    state,_,m15d,_,extra=_market_state(None,df_h1,df_m15,df_d1,None)
    bull=0.0; bear=0.0
    if state.htf_bias=="bullish": bull+=35
    if state.htf_bias=="bearish": bear+=35
    if state.m15_bias=="bullish": bull+=25
    if state.m15_bias=="bearish": bear+=25
    if state.macro_bias=="bullish": bull+=10
    if state.macro_bias=="bearish": bear+=10
    if state.regime.startswith("BULL"): bull+=10
    if state.regime.startswith("BEAR"): bear+=10
    if m15d is not None:
        sh,sl=swing_pts(m15d,5)
        bs=detect_bos(m15d,sh,sl)
        ch=detect_choch(m15d,sh,sl)
        if bs["bullish_bos"] or ch["bullish_choch"]: bull+=10
        if bs["bearish_bos"] or ch["bearish_choch"]: bear+=10
    total=max(bull+bear,1.0)
    direction="bull" if bull>bear else "bear" if bear>bull else "neutral"
    conf=max(bull,bear)/total*100.0
    return {"direction":direction,"confidence":_clip(conf,0,100),"bull_score":bull,"bear_score":bear,
            "htf_bias":state.htf_bias,"macro_bias":state.macro_bias,"m15_struct":state.m15_bias,
            "trigger_count":int(round(max(bull,bear)/10.0)),"direction_edge":abs(bull-bear),
            "fib_r":state.range_position,"m15_relative_volume":state.relative_volume,
            "regime":state.regime,"trend_strength":state.trend_strength,"state":state.to_dict()}


# -----------------------------------------------------------------------------
# GEOMETRY / CANDIDATE ENGINE
# -----------------------------------------------------------------------------
def _candidate_for_direction(state: MarketState, m15d: pd.DataFrame, h1d: pd.DataFrame, direction: str) -> Optional[Candidate]:
    sign=direction
    p=_safe_float(m15d["close"].iloc[-1])
    atr=_safe_float(m15d["atr"].iloc[-1])
    if p<=0 or atr<=0: return None
    ob=detect_order_blocks(m15d,sign)
    fvg=detect_fvg(m15d,sign)
    poi=None
    if ob:
        poi=ob[0]
    elif fvg:
        poi=fvg[0]
    if poi is None:
        return None
    entry=_safe_float(poi.get("mid"),p)
    if not entry>0: return None
    loc=_entry_location(m15d,sign,entry)
    sh,sl=swing_pts(m15d,5)
    last_sw_hi=_safe_float(m15d["high"].iloc[sh[-1]],p) if sh else p+atr
    last_sw_lo=_safe_float(m15d["low"].iloc[sl[-1]],p) if sl else p-atr
    if direction=="bull":
        sl_price=min(last_sw_lo,entry-0.85*atr)
        # Structural target: next external swing / extension.
        target=max(p+2.0*atr, last_sw_hi+0.5*atr)
        if target<=entry: target=entry+2.0*abs(entry-sl_price)
    else:
        sl_price=max(last_sw_hi,entry+0.85*atr)
        target=min(p-2.0*atr, last_sw_lo-0.5*atr)
        if target>=entry: target=entry-2.0*abs(entry-sl_price)
    risk=abs(entry-sl_price)
    if risk<=0: return None
    rr=abs(target-entry)/risk
    # Search a farther structural target if RR < 2.
    if rr < MIN_RR:
        swing_targets=[]
        if direction=="bull":
            for idx in sh:
                val=_safe_float(m15d["high"].iloc[idx])
                if val>entry: swing_targets.append(val)
            ext=entry+risk*(2.5+FIB_EXT_1)
            target=max([target,*swing_targets,ext])
        else:
            for idx in sl:
                val=_safe_float(m15d["low"].iloc[idx])
                if val<entry: swing_targets.append(val)
            ext=entry-risk*(2.5+FIB_EXT_1)
            target=min([target,*swing_targets,ext])
        rr=abs(target-entry)/risk
    if rr<MIN_RR: return None
    confirm=_confirmation(m15d,direction)
    sweep=detect_liquidity_sweep(m15d,sh,sl,direction)
    inducement=detect_inducement(m15d,direction)
    htf_align=1.0 if state.htf_bias==("bullish" if direction=="bull" else "bearish") else 0.0
    macro_align=1.0 if state.macro_bias==("bullish" if direction=="bull" else "bearish") else 0.5 if state.macro_bias=="unknown" else 0.0
    poi_score=_safe_float(poi.get("quality"),60.0) if isinstance(poi,dict) else 55.0
    liq_score=60.0
    reasons=[]
    if htf_align: reasons.append("HTF_ALIGNED")
    if macro_align>=1: reasons.append("MACRO_ALIGNED")
    if state.trend_strength>=72: reasons.append("STRONG_TREND")
    if sweep["type"]!="none": liq_score+=20; reasons.append(sweep["type"].upper())
    if inducement.get("swept"): liq_score+=8; reasons.append("INDUCEMENT_SWEPT")
    if poi_score>=70: reasons.append("HIGH_QUALITY_POI")
    if confirm["confirmed"]: reasons.append("M15_CONFIRMATION")
    else: reasons.append("NO_FRESH_CONFIRMATION")
    if loc["location_score"]>=75: reasons.append("GOOD_LOCATION")
    else: reasons.extend(loc["reasons"][:2])
    setup_quality=_clip(
        poi_score*0.30 + state.trend_strength*0.20 + state.structure_strength*0.15 +
        liq_score*0.12 + loc["location_score"]*0.13 + htf_align*10 + macro_align*5, 0, 100)
    raw_conf=setup_quality
    raw_conf += 7 if confirm["confirmed"] else -10
    raw_conf += 5 if sweep["type"]!="none" else 0
    raw_conf += 4 if rr>=3 else 0
    raw_conf -= 10 if loc["location_score"]<45 else 0
    raw_conf=_clip(raw_conf,0,100)
    return Candidate(direction.upper(),entry,sl_price,target,rr,
                     "H1_OB" if ob else "M15_FVG",raw_conf,setup_quality,
                     loc["location_score"],state.trend_strength,state.structure_strength,
                     liq_score,htf_align,macro_align,True,confirm["confirmed"],reasons,
                     ["HTF_STRUCTURE_BREAK","POI_INVALIDATION","M15_THESIS_FAILURE"])


def _account_context(trade_history=None, kwargs=None) -> dict:
    kwargs=kwargs or {}
    supplied=kwargs.get("account_context") or kwargs.get("stats") or {}
    hist=trade_history if isinstance(trade_history,list) else []
    values=[]
    for row in hist:
        if isinstance(row,dict):
            values.append(_safe_float(row.get("balance_after"),0.0))
    base=_safe_float(supplied.get("starting_balance"),0.0)
    if base<=0 and hist:
        base=_safe_float(hist[0].get("balance_anchor"),0.0) or _safe_float(hist[0].get("balance_after"),0.0)
    current=_safe_float(supplied.get("balance"),0.0) or (_safe_float(hist[-1].get("balance_after"),0.0) if hist else base)
    peak=max(values+[base,current,1e-9])
    dd=(peak-current)/peak*100 if peak>0 else 0.0
    recent=hist[-10:]
    losses=sum(1 for x in recent if str(x.get("result","")).lower()=="sl")
    wins=sum(1 for x in recent if str(x.get("result","")).lower() in {"tp","trail"})
    explicit_dd=_safe_float(supplied.get("drawdown_pct"),dd)
    return {"balance":current,"peak":peak,"drawdown_pct":max(dd,explicit_dd),"recent_losses":losses,
            "recent_wins":wins,"consecutive_losses":_consecutive_losses(hist),
            "risk_regime":"DEFENSIVE" if max(dd,explicit_dd)>=6 else "CAUTIOUS" if max(dd,explicit_dd)>=3 else "NORMAL"}


def _consecutive_losses(hist):
    n=0
    for row in reversed(hist or []):
        if str(row.get("result","")).lower()=="sl": n+=1
        else: break
    return n


# -----------------------------------------------------------------------------
# RECENCY / HISTORICAL EXPERIENCE
# -----------------------------------------------------------------------------
def _regime_distance(current: dict, old: dict) -> float:
    fields=("regime","htf_bias","m15_bias","macro_bias")
    mismatches=0
    for f in fields:
        a=str(current.get(f,"unknown")); b=str(old.get(f,"unknown"))
        if a!="unknown" and b!="unknown" and a!=b: mismatches+=1
    return mismatches/len(fields)


def _effective_history_limit(current_state: MarketState, examples: list[dict]) -> float:
    # Start at 72h for today's decision. Extend only if sparse and similarity is good.
    now=_now()
    recent=0
    for x in examples:
        if now-_latest_ts(x) <= RECENCY_SAFE_MIN_AGE_HOURS*3600: recent+=1
    if recent>=RECENCY_MIN_SIMILAR_EXAMPLES:
        return RECENCY_SAFE_MIN_AGE_HOURS
    # Sparse -> extend progressively, never beyond hard ceiling.
    return min(RECENCY_DEFAULT_MAX_AGE_HOURS, 24.0 * (7 if len(examples)<30 else 14))


def _recency_weight(age_hours: float, max_age_hours: float, drift: float) -> float:
    if age_hours<0: age_hours=0
    if age_hours>max_age_hours: return 0.0
    half=RECENCY_HALF_LIFE_HOURS/max(1.0,1.0+drift*2.0) if RECENCY_CONTEXT_DRIFT_TIGHTEN else RECENCY_HALF_LIFE_HOURS
    w=math.exp(-math.log(2.0)*age_hours/max(half,1e-9))
    # A hard floor is intentionally absent: old information can become irrelevant.
    return _clip(w,0,1)


def _feature_similarity(current: dict, old: dict) -> float:
    numeric=("trend_strength","location_score","rr","confidence","setup_quality","drawdown_pct")
    vals=[]
    for k in numeric:
        a=_safe_float(current.get(k),0); b=_safe_float(old.get(k),0)
        if k=="rr": scale=3.0
        elif k=="drawdown_pct": scale=6.0
        else: scale=50.0
        vals.append(_clip(1.0-abs(a-b)/max(scale,1e-9),0,1))
    return float(np.mean(vals)) if vals else 0.0


def _historical_context(candidate: Candidate, state: MarketState, trade_history: list[dict]) -> dict:
    examples=[x for x in (trade_history or []) if isinstance(x,dict) and str(x.get("decision","")).upper()==candidate.direction]
    if not examples:
        return {"count":0,"win_rate":None,"avg_r":None,"effective_window_hours":72.0,"quality":"NO_DATA"}
    cur={"regime":state.regime,"htf_bias":state.htf_bias,"m15_bias":state.m15_bias,"macro_bias":state.macro_bias,
         "trend_strength":candidate.trend_strength,"location_score":candidate.location_score,"rr":candidate.rr,
         "confidence":candidate.confidence,"setup_quality":candidate.setup_quality,"drawdown_pct":0.0}
    window=_effective_history_limit(state,examples)
    weighted=[]; now=_now()
    for row in examples:
        ts=_latest_ts(row); age=(now-ts)/3600.0
        if age<0: age=0
        if age>window: continue
        drift=_regime_distance(state.to_dict(), row)
        sim=_feature_similarity(cur,row)
        if sim<0.35: continue
        w=_recency_weight(age,window,drift)*(0.55+0.45*sim)
        r=_safe_float(row.get("realized_r"),0.0)
        if r==0:
            pnl_pct=_safe_float(row.get("pct"),0.0)
            rr=_safe_float(row.get("rr"),candidate.rr)
            if rr>0: r=pnl_pct/100.0*max(rr,1.0)
        weighted.append((w,r,row))
    if not weighted:
        return {"count":0,"win_rate":None,"avg_r":None,"effective_window_hours":window,"quality":"INSUFFICIENT_SIMILAR_DATA"}
    sw=sum(w for w,_,_ in weighted)
    if sw<=0: return {"count":0,"win_rate":None,"avg_r":None,"effective_window_hours":window,"quality":"ZERO_WEIGHT"}
    wins=sum(w for w,r,_ in weighted if r>0)/sw
    avg_r=sum(w*r for w,r,_ in weighted)/sw
    return {"count":len(weighted),"win_rate":round(wins*100,2),"avg_r":round(avg_r,4),
            "effective_window_hours":round(window,2),"quality":"GOOD" if len(weighted)>=RECENCY_MIN_SIMILAR_EXAMPLES else "SPARSE"}


# -----------------------------------------------------------------------------
# LEARNED MODEL (lightweight, no external training dependency)
# -----------------------------------------------------------------------------
ML_FEATURE_NAMES=[
    "confidence","setup_quality","location_score","rr","trend_strength","structure_strength",
    "liquidity_score","htf_alignment","macro_alignment","trigger_confirmed","regime_bull","regime_bear",
    "drawdown_pct","consecutive_losses","recent_signal_rate","ollama_delta"
]


def set_learning_model(model):
    global _LEARNED_MODEL
    with _LOCK:
        _LEARNED_MODEL = model if isinstance(model,dict) and model.get("active") else None


def get_learning_model_info():
    with _LOCK:
        m=dict(_LEARNED_MODEL or {})
    return {"active":bool(m),"model_version":m.get("model_version","static"),"sample_count":int(m.get("sample_count",0) or 0),
            "confidence_min":m.get("confidence_min"),"champion":m.get("champion",m)}


def _predict_ml(features: dict) -> Optional[dict]:
    with _LOCK: model=dict(_LEARNED_MODEL or {})
    if not model: return None
    try:
        expected=list(model.get("feature_names") or ML_FEATURE_NAMES)
        if expected!=ML_FEATURE_NAMES: return None
        x=np.asarray([_safe_float(features.get(k),0.0) for k in expected],dtype=float)
        mean=np.asarray(model.get("mean",[0.0]*len(expected)),dtype=float)
        scale=np.asarray(model.get("scale",[1.0]*len(expected)),dtype=float)
        w=np.asarray(model.get("w",[0.0]*len(expected)),dtype=float)
        b=_safe_float(model.get("b"),0.0)
        z=float(np.dot((x-mean)/np.maximum(scale,1e-8),w)+b)
        z=max(-35,min(35,z)); p=1/(1+math.exp(-z))
        return {"probability":p,"model_confidence":50+50*p,"expected_r":_safe_float(model.get("expected_r"),0.0),
                "model_version":model.get("model_version","unknown"),"sample_count":int(model.get("sample_count",0) or 0)}
    except Exception:
        return None


# -----------------------------------------------------------------------------
# OLLAMA CRITIC
# -----------------------------------------------------------------------------
def _ollama_allowed() -> bool:
    if not OLLAMA_ENABLED or not OLLAMA_API_KEY or requests is None:
        return False
    now=_now()
    while _OLLAMA_CALL_TIMES and now-_OLLAMA_CALL_TIMES[0]>60:
        _OLLAMA_CALL_TIMES.popleft()
    return len(_OLLAMA_CALL_TIMES)<OLLAMA_MAX_CALLS_PER_MIN


def _ollama_critic(packet: dict) -> dict:
    if not _ollama_allowed():
        return {"enabled":False,"verdict":"SKIPPED","delta":0.0,"reason":"rate_limit_or_unconfigured"}
    key=json.dumps(packet,sort_keys=True,ensure_ascii=False,default=str)
    cache_key=str(abs(hash(key)))
    cached=_OLLAMA_CACHE.get(cache_key)
    if cached and _now()-cached[0]<=OLLAMA_CACHE_SEC:
        return dict(cached[1])
    system=(
        "You are a trading setup critic. You are NOT an execution agent. "
        "Critique the supplied structured market decision. Never invent market data. "
        "Return JSON only with keys verdict, thesis_quality, contradictions, risk_flags, "
        "confidence_adjustment, summary. verdict must be READY, WAIT, or INVALIDATE. "
        "Do not provide broker/API actions."
    )
    user=json.dumps(packet,ensure_ascii=False,default=str)
    payload={"model":OLLAMA_MODEL,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"stream":False}
    headers={"Authorization":f"Bearer {OLLAMA_API_KEY}","Content-Type":"application/json"}
    try:
        _OLLAMA_CALL_TIMES.append(_now())
        resp=requests.post(OLLAMA_API_URL,headers=headers,json=payload,timeout=OLLAMA_TIMEOUT_SEC)
        resp.raise_for_status()
        data=resp.json()
        content=((data.get("message") or {}).get("content") or "").strip()
        if content.startswith("```"):
            content=content.strip("`").replace("json\n", "", 1).strip()
        parsed=json.loads(content)
        adj=_clip(_safe_float(parsed.get("confidence_adjustment"),0.0),-12,12)
        out={"enabled":True,"verdict":str(parsed.get("verdict") or "WAIT").upper(),
             "thesis_quality":_clip(_safe_float(parsed.get("thesis_quality"),50),0,100),
             "contradictions":list(parsed.get("contradictions") or [])[:5],
             "risk_flags":list(parsed.get("risk_flags") or [])[:5],
             "delta":adj,"summary":str(parsed.get("summary") or "")[:700]}
        _OLLAMA_CACHE[cache_key]=(_now(),out)
        return out
    except Exception as exc:
        log.debug("[OLLAMA] critic failed: %s",exc)
        return {"enabled":False,"verdict":"ERROR","delta":0.0,"reason":str(exc)[:240]}


# -----------------------------------------------------------------------------
# DECISION ENGINE
# -----------------------------------------------------------------------------
def _current_signal_threshold() -> float:
    with _LOCK:
        base=_MANUAL_THRESHOLD if _MANUAL_THRESHOLD is not None else CONFIDENCE_BASE
        return _clip(_ADAPTIVE_THRESHOLD if _MANUAL_THRESHOLD is None else _ADAPTIVE_THRESHOLD, CONFIDENCE_SAFE_MIN, CONFIDENCE_SAFE_MAX)


def get_active_confidence_threshold():
    return round(_current_signal_threshold(),2)


def set_manual_confidence_threshold(value):
    global _MANUAL_THRESHOLD, _ADAPTIVE_THRESHOLD
    v=_clip(value,CONFIDENCE_SAFE_MIN,CONFIDENCE_SAFE_MAX)
    with _LOCK:
        _MANUAL_THRESHOLD=v
        _ADAPTIVE_THRESHOLD=v
    return v


def suggest_confidence_threshold():
    return get_active_confidence_threshold()


def _frequency_rate() -> float:
    with _LOCK:
        rows=list(_SCAN_HISTORY)
    if not rows: return FREQUENCY_TARGET_IDEAL
    denom=sum(max(1,int(x.get("analyzed_symbols",0) or 0)) for x in rows)
    num=sum(int(x.get("eligible_count",0) or 0) for x in rows)
    return num/denom if denom else FREQUENCY_TARGET_IDEAL


def _adapt_frequency(summary: dict):
    global _ADAPTIVE_THRESHOLD
    rate=_frequency_rate()
    quality=_safe_float(summary.get("avg_confidence"),0)
    with _LOCK:
        cur=_ADAPTIVE_THRESHOLD
        manual=_MANUAL_THRESHOLD
        # Manual threshold becomes the anchor. Adaptive drift is deliberately tiny.
        anchor=manual if manual is not None else CONFIDENCE_BASE
        if rate< FREQUENCY_TARGET_LOW and quality>=55:
            cur-=CONFIDENCE_ADAPT_STEP
        elif rate>FREQUENCY_TARGET_HIGH:
            cur+=CONFIDENCE_ADAPT_STEP
        else:
            # Slowly return toward anchor so a stale frequency shock does not stick forever.
            cur += (anchor-cur)*0.15
        _ADAPTIVE_THRESHOLD=_clip(cur,CONFIDENCE_SAFE_MIN,CONFIDENCE_SAFE_MAX)


def _build_candidate_learning_features(c: Candidate, account: dict, freq_rate: float, ollama_delta: float=0.0) -> dict:
    return {
        "confidence":c.confidence,"setup_quality":c.setup_quality,"location_score":c.location_score,
        "rr":min(c.rr,8),"trend_strength":c.trend_strength,"structure_strength":c.structure_strength,
        "liquidity_score":c.liquidity_score,"htf_alignment":c.htf_alignment,"macro_alignment":c.macro_alignment,
        "trigger_confirmed":1.0 if c.trigger_confirmed else 0.0,
        "regime_bull":1.0 if c.direction=="BUY" and c.trend_strength>=50 else 0.0,
        "regime_bear":1.0 if c.direction=="SELL" and c.trend_strength>=50 else 0.0,
        "drawdown_pct":account.get("drawdown_pct",0.0),"consecutive_losses":account.get("consecutive_losses",0),
        "recent_signal_rate":freq_rate,"ollama_delta":ollama_delta,
    }


def full_analyze(df_h1, df_m15, df_d1=None, symbol=None, df_btc_h1=None, trade_history=None,
                 market_data_source="bybit", **kwargs):
    """Main strategy boundary. No Binance calls. Always returns a structured packet when data is usable."""
    try:
        state,h1d,m15d,d1d,extra=_market_state(symbol,df_h1,df_m15,df_d1,df_btc_h1)
        if h1d is None or m15d is None:
            return {"symbol":symbol,"decision":"WAIT","no_signal":True,"candidate":False,
                    "execution_eligible":False,"analysis_stage":"DATA_QUALITY","rejected_reason":"INSUFFICIENT_DATA",
                    "eligibility_source":"brain_v2","confidence":0.0,"confidence_threshold":get_active_confidence_threshold(),
                    "market_data_source":market_data_source,"brain_version":FINAL_BRAIN_VERSION}
        score=score_direction(df_h1,df_m15,df_d1)
        direction=score.get("direction")
        if direction not in {"bull","bear"}:
            return _no_signal_packet(symbol,state,score,"NO_CLEAR_DIRECTION",market_data_source)
        # HTF gate: never trade strongly against HTF structure.
        if state.htf_bias not in {"bullish" if direction=="bull" else "bearish","unknown","ranging"} and state.trend_strength<80:
            return _no_signal_packet(symbol,state,score,"HTF_CONFLICT",market_data_source)
        c=_candidate_for_direction(state,m15d,h1d,direction)
        if c is None:
            return _no_signal_packet(symbol,state,score,"NO_VALID_POI_OR_GEOMETRY",market_data_source)
        if not isinstance(trade_history, list):
            with _LOCK:
                internal_outcomes=[dict(x) for x in _HISTORY if x.get("kind")=="outcome"]
            trade_history=internal_outcomes
        account=_account_context(trade_history,kwargs)
        hist=_historical_context(c,state,trade_history or [])
        freq=_frequency_rate()
        ml_features=_build_candidate_learning_features(c,account,freq,0)
        ml=_predict_ml(ml_features)
        if ml:
            c.confidence=(c.confidence*0.70)+(_safe_float(ml.get("model_confidence"),c.confidence)*0.30)
        packet={"symbol":symbol,"direction":c.direction,"entry":c.entry,"sl":c.sl,"tp":c.tp,"rr":c.rr,
                "regime":state.regime,"trend_strength":state.trend_strength,"structure_strength":state.structure_strength,
                "htf_bias":state.htf_bias,"m15_bias":state.m15_bias,"macro_bias":state.macro_bias,
                "liquidity_state":state.liquidity_state,"reasons":c.reasons,"account":account,
                "historical":hist,"candidate_confidence":c.confidence}
        ollama={"enabled":False,"verdict":"SKIPPED","delta":0.0}
        # LLM only critiques reasonably strong candidates, preserving scan frequency and latency.
        if c.confidence>=OLLAMA_MIN_CONFIDENCE:
            ollama=_ollama_critic(packet)
            c.confidence=_clip(c.confidence+_safe_float(ollama.get("delta"),0.0),0,100)
            if str(ollama.get("verdict"))=="INVALIDATE":
                return _no_signal_packet(symbol,state,score,"LLM_INVALIDATED",market_data_source,extra={"ollama":ollama,"candidate":c.to_dict()})
            if str(ollama.get("verdict"))=="WAIT":
                c.trigger_confirmed=False
        threshold=get_active_confidence_threshold()
        # Drawdown guard is not a strategy veto: it raises the quality requirement modestly.
        dd=account.get("drawdown_pct",0.0)
        required=threshold + (6 if dd>=6 else 3 if dd>=3 else 0)
        if not c.trigger_confirmed and c.confidence<required+5:
            reason="WAIT_CONFIRMATION"
            eligible=False
        elif c.confidence < required:
            reason="BELOW_ADAPTIVE_THRESHOLD"
            eligible=False
        else:
            reason="READY"
            eligible=True
        learning_features=_build_candidate_learning_features(c,account,freq,_safe_float(ollama.get("delta"),0.0))
        out={
            "symbol":symbol,"decision":("BUY" if c.direction=="BULL" else "SELL"),"candidate":True,"is_candidate":True,
            "execution_eligible":eligible,"eligibility_source":"brain_v2",
            "eligibility_reason":reason if not eligible else "BRAIN_READY",
            "analysis_stage":"READY" if eligible else "WAIT_ENTRY",
            "rejected_reason":None if eligible else reason,
            "no_signal":not eligible,
            "confidence":round(c.confidence,2),"confidence_threshold":round(required,2),"active_threshold":round(threshold,2),
            "entry":c.entry,"sl":c.sl,"initial_sl":c.sl,"tp":c.tp,"rr":round(c.rr,3),"entry_label":c.entry_label,
            "atr":_safe_float(m15d["atr"].iloc[-1]),"rsi":_safe_float(m15d["rsi"].iloc[-1]),
            "struct_h1":state.htf_bias,"d1_bias":extra.get("d1_bias"),
            "market_regime":state.regime,"trend_strength":round(state.trend_strength,2),"structure_strength":round(state.structure_strength,2),
            "macro_bias":state.macro_bias,"liquidity_state":state.liquidity_state,
            "poi_reacted":True,"trigger_confirmed":c.trigger_confirmed,"reasons":c.reasons,
            "invalidations":c.invalidations,"learning_features":learning_features,
            "historical_context":hist,"ml":ml,"ollama":ollama,
            "account_context":account,"market_data_source":market_data_source,
            "brain_version":FINAL_BRAIN_VERSION,"strategy_version":_STRATEGY_STATE.get("version","S2.0"),
            "low_confidence":(c.confidence < max(60,required)),
            "low_confidence_cutoff":required,"ban_recommended":False,
            # Canonical execution handoff. main.py is the only execution authority.
            "execution_contract":"V132",
            "execution":{
                "symbol":str(symbol or "").upper(),
                "side":("BUY" if c.direction=="BULL" else "SELL"),
                "entry":float(c.entry),
                "sl":float(c.sl),
                "tp":float(c.tp),
                "rr":float(c.rr),
                "type":"LIMIT",
            },
        }
        # Hard contract check before returning a signal to main.py. A strategy
        # signal is never marked eligible unless its complete risk geometry is
        # finite and directionally valid.
        try:
            vals=(float(out["entry"]),float(out["sl"]),float(out["tp"]))
            valid=all(math.isfinite(v) and v>0 for v in vals)
            valid = valid and ((out["sl"] < out["entry"] < out["tp"]) if out["decision"]=="BUY" else (out["tp"] < out["entry"] < out["sl"]))
            if not valid:
                out["execution_eligible"]=False
                out["no_signal"]=True
                out["analysis_stage"]="EXECUTION_CONTRACT_REJECTED"
                out["rejected_reason"]="INVALID_EXECUTION_GEOMETRY"
                out["eligibility_reason"]="INVALID_EXECUTION_GEOMETRY"
        except Exception:
            out["execution_eligible"]=False
            out["no_signal"]=True
            out["analysis_stage"]="EXECUTION_CONTRACT_REJECTED"
            out["rejected_reason"]="INVALID_EXECUTION_PACKET"
            out["eligibility_reason"]="INVALID_EXECUTION_PACKET"
        return out
    except Exception as exc:
        log.exception("[BRAIN V2] full_analyze failed")
        return {"symbol":symbol,"decision":"WAIT","no_signal":True,"candidate":False,"execution_eligible":False,
                "analysis_stage":"ERROR","rejected_reason":"BRAIN_ERROR","eligibility_source":"brain_v2",
                "confidence":0.0,"error":str(exc)[:240],"market_data_source":market_data_source,"brain_version":FINAL_BRAIN_VERSION}


def _no_signal_packet(symbol,state,score,reason,market_data_source,extra=None):
    return {"symbol":symbol,"decision":score.get("direction","WAIT").upper() if score.get("direction") in {"bull","bear"} else "WAIT",
            "candidate":False,"is_candidate":False,"execution_eligible":False,"eligibility_source":"brain_v2",
            "eligibility_reason":reason,"analysis_stage":"DIAGNOSTIC_NO_ENTRY","rejected_reason":reason,"no_signal":True,
            "confidence":round(_safe_float(score.get("confidence"),0),2),"confidence_threshold":get_active_confidence_threshold(),
            "market_regime":state.regime,"trend_strength":round(state.trend_strength,2),"structure_strength":round(state.structure_strength,2),
            "macro_bias":state.macro_bias,"liquidity_state":state.liquidity_state,"market_data_source":market_data_source,
            "brain_version":FINAL_BRAIN_VERSION,**(extra or {})}


# -----------------------------------------------------------------------------
# POSITION MANAGEMENT / TRAILING
# -----------------------------------------------------------------------------
def manage_position(state, df_m15, df_h1=None, df_d1=None, symbol=None, **kwargs):
    """Brain-owned management recommendation. Pending entries are valid analysis state.

    Main.py remains the execution authority. While an order is still pending, the
    brain must be able to inspect the setup without pretending a live position exists
    and without moving a protective stop before the entry is filled.
    """
    try:
        m15=build_df(df_m15,15); h1=build_df(df_h1,60) if df_h1 is not None else None
        status=str((state or {}).get("status") or (state or {}).get("lifecycle") or "").lower()
        if status in {"pending", "entry_pending", "waiting_entry"} or str((state or {}).get("lifecycle") or "").upper()=="ENTRY_PENDING":
            sig=state.get("signal") if isinstance(state.get("signal"),dict) else state
            tp=_safe_float(sig.get("tp"),0.0) if isinstance(sig,dict) else 0.0
            sl=_safe_float(sig.get("sl"),0.0) if isinstance(sig,dict) else 0.0
            return {
                "action":"HOLD", "close":False, "tp":tp or None, "sl":None, "new_sl":None,
                "profit_r":0.0, "trend_strength":None, "weakness_score":0, "relative_volume":None,
                "trail_source":"pending_entry_analysis", "state":"PENDING",
                "reason":["ENTRY_PENDING","NO_TRAILING_BEFORE_FILL"],
                "engine_version":FINAL_BRAIN_VERSION,
            }
        if m15 is None:
            return {"action":"HOLD","tp":None,"sl":None,"reason":["DATA_UNAVAILABLE"],"trail_source":"brain_v2"}
        sig=state.get("signal") if isinstance(state.get("signal"),dict) else state
        side=str(sig.get("decision") or state.get("decision") or "BUY").upper()
        entry=_safe_float(state.get("entry") or sig.get("entry"),_safe_float(m15["close"].iloc[-1]))
        current_price=_safe_float(state.get("current_price") or state.get("price"),_safe_float(m15["close"].iloc[-1]))
        current_sl=_safe_float(state.get("current_sl") or sig.get("sl"),0.0)
        initial_sl=_safe_float(state.get("initial_sl") or sig.get("sl"),current_sl)
        tp=_safe_float(sig.get("tp"),0.0)
        risk=abs(entry-initial_sl)
        if risk<=0:
            return {"action":"HOLD","tp":tp or None,"sl":None,"reason":["NO_RISK_REFERENCE"],"trail_source":"brain_v2"}
        profit_r=_price_return(entry,current_price,side)/max(risk/entry,1e-12)
        sh,sl=swing_pts(m15,STRUCT_TRAIL_LB)
        trend=_trend_strength(m15,sh,sl)
        struct=_market_structure(m15,sh,sl)
        rel_vol=_safe_float(m15["volume"].iloc[-1],0)/max(_safe_float(m15["vol_sma"].iloc[-1],1),1e-12)
        weakness=0
        if side=="BUY":
            if struct!="bullish": weakness+=3
            if trend<60: weakness+=2
            if rel_vol<0.75: weakness+=2
            if _safe_float(m15["rsi"].iloc[-1],50)<48: weakness+=1
        else:
            if struct!="bearish": weakness+=3
            if trend<60: weakness+=2
            if rel_vol<0.75: weakness+=2
            if _safe_float(m15["rsi"].iloc[-1],50)>52: weakness+=1
        reasons=[]
        new_sl=None
        if sh and side=="SELL":
            swing=_safe_float(m15["high"].iloc[sh[-1]])
            new_sl=swing*(1+STRUCT_TRAIL_BUF_PCT)
        if sl and side=="BUY":
            swing=_safe_float(m15["low"].iloc[sl[-1]])
            new_sl=swing*(1-STRUCT_TRAIL_BUF_PCT)
        # Only tighten after the trade has earned enough room.
        if profit_r < 0.8:
            new_sl=None
            reasons.append("TRAIL_NOT_ARMED")
        elif profit_r>=1.0:
            reasons.append("PROFIT_PROTECTED")
        if weakness>=6:
            reasons.append("REVERSAL_RISK_STRONG")
            if profit_r>=1.0 and new_sl is not None:
                # tighten toward break-even / current structural candidate
                be=entry*(1+0.0005 if side=="BUY" else 0.9995)
                new_sl=max(current_sl,be,new_sl) if side=="BUY" else min(current_sl,be,new_sl)
        elif weakness>=3:
            reasons.append("TREND_WEAKENING")
        else:
            reasons.append("STRUCTURE_HEALTHY")
        # Validate direction of movement: a brain recommendation may only tighten risk.
        if new_sl is not None:
            if side=="BUY" and new_sl <= current_sl: new_sl=None
            if side=="SELL" and new_sl >= current_sl and current_sl>0: new_sl=None
        close=False
        close_reason=None
        # Thesis invalidation is stronger than a generic weak trend.
        if side=="BUY" and struct=="bearish" and profit_r>0.5 and weakness>=6:
            close=True; close_reason="trail"
        elif side=="SELL" and struct=="bullish" and profit_r>0.5 and weakness>=6:
            close=True; close_reason="trail"
        return {"action":"CLOSE" if close else "UPDATE" if new_sl is not None else "HOLD",
                "close":close,"reason":reasons,"close_reason":close_reason,"tp":tp or None,"sl":new_sl,
                "new_sl":new_sl,"profit_r":round(profit_r,3),"trend_strength":round(trend,2),
                "weakness_score":weakness,"relative_volume":round(rel_vol,3),"trail_source":"structure_m15_v2",
                "state":"TRAIL" if new_sl is not None else "HOLD","engine_version":FINAL_BRAIN_VERSION}
    except Exception as exc:
        log.exception("[BRAIN V2] manage_position failed")
        return {"action":"HOLD","tp":None,"sl":None,"reason":["MANAGEMENT_ERROR"],"trail_source":"brain_v2","error":str(exc)[:220]}


# -----------------------------------------------------------------------------
# EXPERIENCE INGESTION / AUTOPSY
# -----------------------------------------------------------------------------
def _append_experience(row: dict, kind: str):
    rec=dict(row or {}); rec["kind"]=kind; rec.setdefault("timestamp",_now())
    with _LOCK:
        _HISTORY.append(_json_safe(rec))
    return {"ok":True,"count":len(_HISTORY),"kind":kind}


def record_candidate_observation(row, source="bybit_market"):
    rec=dict(row or {}); rec["source"]=source
    return _append_experience(rec,"candidate")


def ingest_live_candidate(row, source="bybit_market"):
    return record_candidate_observation(row,source)


def record_trade_outcome(trade, outcome=None, source="binance_trade"):
    rec=dict(trade or {})
    out=dict(outcome or rec.get("outcome") or {}) if isinstance(outcome or rec.get("outcome"),dict) else {}
    rec["source"]=source
    rec["outcome"]=out
    # Normalize realized R and explicit exit diagnosis.
    if "realized_r" not in rec:
        entry=_safe_float(rec.get("entry"),0); exit_price=_safe_float(rec.get("exit_price"),0); rr=_safe_float(rec.get("rr"),1)
        side=str(rec.get("decision") or "BUY")
        move=_price_return(entry,exit_price,side) if entry and exit_price else 0
        risk_pct=abs(_safe_float(rec.get("sl"),entry)-entry)/entry if entry else 0
        rec["realized_r"]=move/max(risk_pct,1e-12)
    rec["autopsy"]=_trade_autopsy(rec)
    return _append_experience(rec,"outcome")


def ingest_live_outcome(trade, outcome=None, source="binance_trade"):
    return record_trade_outcome(trade,outcome,source)


def _trade_autopsy(row: dict) -> dict:
    result=str(row.get("result") or row.get("close_reason") or "unknown").lower()
    mfe_r=_safe_float(row.get("mfe_r"),0); mae_r=_safe_float(row.get("mae_r"),0)
    trail_count=int(_safe_float(row.get("trail_update_count"),0))
    conf=_safe_float(row.get("confidence"),0)
    rr=_safe_float(row.get("rr"),0)
    realized=_safe_float(row.get("realized_r"),0)
    reasons=[]
    if result=="sl":
        if mfe_r>=1.0: reasons.append("LOSS_AFTER_FAVORABLE_MOVE")
        if mae_r<=-1.0: reasons.append("ADVERSE_MOVE_REACHED_INVALIDATION")
        if row.get("trail_failed_count"): reasons.append("TRAIL_INFRA_FAILURE")
        if conf>=80: reasons.append("HIGH_CONFIDENCE_LOSS")
    elif result=="trail":
        if mfe_r-realized>1.0: reasons.append("TRAIL_LEFT_MONEY")
        if trail_count>0: reasons.append("TRAIL_MANAGED_EXIT")
    elif result=="tp":
        reasons.append("TARGET_REALIZED")
    return {"result":result,"realized_r":realized,"mfe_r":mfe_r,"mae_r":mae_r,"reasons":reasons,
            "high_confidence_loss":bool(result=="sl" and conf>=80),"trail_count":trail_count,"initial_rr":rr}


def record_protection_event(event, source="main_execution"):
    rep=dict(event or {}); rep["source"]=source; rep["recorded_at"]=_now()
    with _LOCK: _PROTECTION_EVENTS.append(_json_safe(rep))
    return {"ok":True,"event_count":len(_PROTECTION_EVENTS)}


# -----------------------------------------------------------------------------
# SCAN/FREQUENCY / STATS LEARNING
# -----------------------------------------------------------------------------
def record_scan_summary(summary, source="main_scanner"):
    rep=dict(summary or {}); rep["source"]=source; rep.setdefault("timestamp",_now())
    with _LOCK: _SCAN_HISTORY.append(_json_safe(rep))
    _adapt_frequency(rep)
    return {"ok":True,"signal_rate":round(_frequency_rate(),4),"threshold":get_active_confidence_threshold()}


def _recent_trade_stats(max_age_h=72):
    now=_now(); rows=[]
    with _LOCK: hist=list(_HISTORY)
    for r in hist:
        if r.get("kind")!="outcome": continue
        age=(now-_latest_ts(r))/3600
        if age<=max_age_h: rows.append(r)
    wins=sum(1 for r in rows if _safe_float(r.get("realized_r"),0)>0)
    avg=sum(_safe_float(r.get("realized_r"),0) for r in rows)/len(rows) if rows else 0
    return {"count":len(rows),"wins":wins,"win_rate":wins/len(rows) if rows else None,"avg_r":avg}


def evaluate_stats_decision(snapshot, source="main_stats"):
    snap=dict(snapshot or {})
    dd=_safe_float(snap.get("drawdown_pct"),0)
    total=int(_safe_float(snap.get("total"),0)); tp=int(_safe_float(snap.get("tp"),0)); sl=int(_safe_float(snap.get("sl"),0)); trail=int(_safe_float(snap.get("trail"),0))
    recent=_recent_trade_stats(72)
    action="MAINTAIN"; reason=[]
    if dd>=8:
        action="DEFENSIVE"; reason.append("DRAWdown_HIGH")
    elif dd>=5:
        action="CAUTIOUS"; reason.append("DRAWDOWN_ELEVATED")
    if recent.get("count",0)>=10 and recent.get("avg_r",0)<0:
        action="REVIEW"; reason.append("RECENT_EXPECTED_R_NEGATIVE")
    return {"action":action,"reason":reason or ["NO_STRONG_STATS_CHANGE"],"drawdown_pct":dd,"total":total,
            "wins":tp+trail,"sl":sl,"recent_72h":recent,"frequency_rate":_frequency_rate(),
            "threshold":get_active_confidence_threshold(),"strategy_version":_STRATEGY_STATE.get("version","S2.0"),"source":source}


# -----------------------------------------------------------------------------
# CHECKPOINT / COGNITIVE STATUS
# -----------------------------------------------------------------------------
def export_checkpoint_state():
    with _LOCK:
        return _json_safe({
            "schema":BRAIN_CHECKPOINT_SCHEMA,"brain_version":FINAL_BRAIN_VERSION,"interface_version":BRAIN_INTERFACE_VERSION,
            "saved_at":_now(),"history":list(_HISTORY),"scan_history":list(_SCAN_HISTORY),
            "protection_events":list(_PROTECTION_EVENTS),"learned_model":_LEARNED_MODEL,
            "full_enabled":FULL_ENABLED,"ticks":_AGENT_TICKS,"manual_threshold":_MANUAL_THRESHOLD,
            "adaptive_threshold":_ADAPTIVE_THRESHOLD,"strategy_state":dict(_STRATEGY_STATE),
        })


def import_checkpoint_state(checkpoint):
    if not isinstance(checkpoint,dict) or checkpoint.get("schema") not in {BRAIN_CHECKPOINT_SCHEMA,"brain_progress_checkpoint_v1"}:
        raise ValueError(f"unsupported brain checkpoint schema: {checkpoint.get('schema') if isinstance(checkpoint,dict) else None}")
    global _LEARNED_MODEL,_MANUAL_THRESHOLD,_ADAPTIVE_THRESHOLD,_AGENT_TICKS, FULL_ENABLED
    with _LOCK:
        _HISTORY.clear(); _HISTORY.extend(checkpoint.get("history") or checkpoint.get("agent_state",{}).get("history") or [])
        _SCAN_HISTORY.clear(); _SCAN_HISTORY.extend(checkpoint.get("scan_history") or [])
        _PROTECTION_EVENTS.clear(); _PROTECTION_EVENTS.extend(checkpoint.get("protection_events") or [])
        learned=checkpoint.get("learned_model")
        _LEARNED_MODEL=dict(learned) if isinstance(learned,dict) else None
        _MANUAL_THRESHOLD=checkpoint.get("manual_threshold")
        _ADAPTIVE_THRESHOLD=_safe_float(checkpoint.get("adaptive_threshold"),CONFIDENCE_BASE)
        _AGENT_TICKS=int(checkpoint.get("ticks",0) or 0)
        FULL_ENABLED=bool(checkpoint.get("full_enabled",False))
        if isinstance(checkpoint.get("strategy_state"),dict): _STRATEGY_STATE.update(checkpoint["strategy_state"])
    return {"ok":True,"schema":BRAIN_CHECKPOINT_SCHEMA,"restored_at":_now(),"strategy_version":_STRATEGY_STATE.get("version","S2.0")}


get_brain_state=export_checkpoint_state
apply_brain_state=import_checkpoint_state


def get_learning_schema():
    return {"schema":FULL_LEARNING_SCHEMA,"brain_version":FINAL_BRAIN_VERSION,"ml_schema":MACHINE_LEARNING_SCHEMA,
            "recency":{"half_life_hours":RECENCY_HALF_LIFE_HOURS,"default_max_age_hours":RECENCY_DEFAULT_MAX_AGE_HOURS,
                        "sparse_extension":RECENCY_EXTEND_IF_SPARSE},
            "frequency":{"target_low":FREQUENCY_TARGET_LOW,"target_high":FREQUENCY_TARGET_HIGH,"ideal":FREQUENCY_TARGET_IDEAL},
            "ollama":{"enabled":bool(OLLAMA_ENABLED and OLLAMA_API_KEY),"model":OLLAMA_MODEL}}


def get_strategy_evolution_status():
    with _LOCK:
        return {"active_version":_STRATEGY_STATE.get("version","S2.0"),"champion_version":_STRATEGY_STATE.get("champion","S2.0"),
                "revisions":int(_STRATEGY_STATE.get("revisions",0) or 0),"challengers":list(_STRATEGY_STATE.get("challengers") or []),
                "last_reason":_STRATEGY_STATE.get("last_reason","startup")}


def get_cognitive_status():
    with _LOCK:
        history=list(_HISTORY); scans=list(_SCAN_HISTORY); protections=len(_PROTECTION_EVENTS)
    outcomes=[x for x in history if x.get("kind")=="outcome"]
    candidates=[x for x in history if x.get("kind")=="candidate"]
    return {"brain_version":FINAL_BRAIN_VERSION,"strategy":get_strategy_evolution_status(),
            "full_enabled":bool(FULL_ENABLED),"worker_alive":bool(_FULL_THREAD and _FULL_THREAD.is_alive()),"ticks":_AGENT_TICKS,
            "experience_samples":len(outcomes),"candidate_samples":len(candidates),"protection_events":protections,
            "scan_cycles":len(scans),"signal_rate":_frequency_rate(),"threshold":get_active_confidence_threshold(),
            "recent_72h":_recent_trade_stats(72),"recency_half_life_hours":RECENCY_HALF_LIFE_HOURS,
            "ollama":{"configured":bool(OLLAMA_API_KEY),"model":OLLAMA_MODEL,"enabled":bool(OLLAMA_ENABLED)},
            "learning_schema":FULL_LEARNING_SCHEMA}


def get_full_cognitive_status():
    return get_cognitive_status()


def get_adaptive_status():
    return get_cognitive_status()


def get_experience_count():
    with _LOCK: return sum(1 for x in _HISTORY if x.get("kind")=="outcome")


# -----------------------------------------------------------------------------
# FULL COMMAND + WORKER
# -----------------------------------------------------------------------------
def reset_cognitive_memory():
    global _AGENT_TICKS,_LEARNED_MODEL,_MANUAL_THRESHOLD,_ADAPTIVE_THRESHOLD,FULL_ENABLED
    with _LOCK:
        _HISTORY.clear(); _SCAN_HISTORY.clear(); _PROTECTION_EVENTS.clear()
        _LEARNED_MODEL=None; _MANUAL_THRESHOLD=None; _ADAPTIVE_THRESHOLD=CONFIDENCE_BASE
        _STRATEGY_STATE.update({"version":"S2.0","revisions":0,"last_reason":"full reset","last_update_at":_now(),"champion":"S2.0","challengers":[]})
        _AGENT_TICKS=0; FULL_ENABLED=False
    _save_state()
    return {"ok":True,"message":"cognitive memory reset"}


def reset_adaptive_learning():
    return reset_cognitive_memory()


def _save_state():
    try:
        STATE_DIR.mkdir(parents=True,exist_ok=True)
        tmp=STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(export_checkpoint_state(),ensure_ascii=False,allow_nan=False,indent=2,default=str),encoding="utf-8")
        os.replace(tmp,STATE_FILE)
        return True
    except Exception as exc:
        log.warning("[BRAIN V2] state save failed: %s",exc)
        return False


def _worker_loop():
    global _AGENT_TICKS
    while not _FULL_STOP.is_set():
        with _LOCK: _AGENT_TICKS+=1
        _periodic_learning_tick()
        _FULL_WAKE.wait(30.0); _FULL_WAKE.clear()
    log.info("[BRAIN V2] FULL worker stopped")


def _periodic_learning_tick():
    # Controlled adaptation from evidence, not blind parameter mutation.
    try:
        recent=_recent_trade_stats(72)
        with _LOCK:
            if recent["count"]>=20 and recent["avg_r"] < -0.15:
                _STRATEGY_STATE["last_reason"]="recent_72h_negative_expectancy"
            elif recent["count"]>=20 and recent["avg_r"] > 0.35:
                _STRATEGY_STATE["last_reason"]="recent_72h_positive_expectancy"
            else:
                _STRATEGY_STATE["last_reason"]="insufficient_or_neutral_recent_evidence"
            _STRATEGY_STATE["last_update_at"]=_now()
        _save_state()
    except Exception as exc:
        log.debug("[BRAIN V2] learning tick: %s",exc)


def adaptive_agent_start():
    global _FULL_THREAD, FULL_ENABLED, FULL_LEARNING_ACTIVE
    with _LOCK:
        FULL_ENABLED=True; FULL_LEARNING_ACTIVE=True
        if _FULL_THREAD is not None and _FULL_THREAD.is_alive():
            _FULL_WAKE.set(); return True
        _FULL_STOP.clear()
        _FULL_THREAD=threading.Thread(target=_worker_loop,name="strategy-brain-v2",daemon=True)
        _FULL_THREAD.start()
    return True


def adaptive_agent_stop():
    global FULL_ENABLED, FULL_LEARNING_ACTIVE, _FULL_THREAD
    with _LOCK: FULL_ENABLED=False; FULL_LEARNING_ACTIVE=False
    _FULL_STOP.set(); _FULL_WAKE.set();
    return True


def _full_status_text() -> str:
    st=get_cognitive_status(); evo=st["strategy"]
    return (f"🧠 <b>FULL V2</b>\n"
            f"Worker: <b>{'ON' if st['worker_alive'] else 'OFF'}</b> | ticks: <b>{st['ticks']}</b>\n"
            f"Strategy: <b>{evo['active_version']}</b> | revisions: <b>{evo['revisions']}</b>\n"
            f"Threshold: <b>{st['threshold']:.1f}%</b> | frequency: <b>{st['signal_rate']*100:.1f}%</b>\n"
            f"Experience: <b>{st['experience_samples']}</b> outcomes / <b>{st['candidate_samples']}</b> candidates\n"
            f"Recent 72h: <b>{st['recent_72h']['count']}</b> trades | avg R: <b>{st['recent_72h']['avg_r']:.3f}</b>\n"
            f"Ollama: <b>{OLLAMA_MODEL if st['ollama']['enabled'] else 'OFF/UNCONFIGURED'}</b>")


def full_command(action, callbacks=None):
    act=str(action or "status").strip().lower()
    if act in {"on","/full on","full on"}:
        adaptive_agent_start()
        return "🧠 <b>FULL ON</b>\nAdaptive brain V2 aktif. Evidence + frequency + recency + trade autopsy berjalan.\n\n"+_full_status_text()
    if act in {"off","/full off","full off"}:
        adaptive_agent_stop()
        return "🧠 <b>FULL OFF</b>\nLearning worker dihentikan. Memory/strategy tetap disimpan.\n\n"+_full_status_text()
    if act in {"reset","/full reset","full reset"}:
        adaptive_agent_stop(); reset_cognitive_memory()
        return "🧠 <b>FULL RESET</b>\nExperience, scan frequency, protection events, learned model dan strategy evolution direset. Execution ledger main.py tidak disentuh."
    if act in {"review","full review","/full review"}:
        return {"ok":True,"action":"review","status":get_cognitive_status(),"strategy":get_strategy_evolution_status()}
    if act in {"experiments","full experiments","/full experiments"}:
        return {"ok":True,"challengers":list(_STRATEGY_STATE.get("challengers") or []),"strategy":get_strategy_evolution_status()}
    return _full_status_text()


# -----------------------------------------------------------------------------
# LEGACY COMPATIBILITY ALIASES / OPTIONAL IMPORT-TIME LOAD
# -----------------------------------------------------------------------------
def load_persisted_state():
    if not STATE_FILE.exists(): return False
    try:
        obj=json.loads(STATE_FILE.read_text(encoding="utf-8"))
        import_checkpoint_state(obj)
        return True
    except Exception as exc:
        log.warning("[BRAIN V2] persisted state load failed: %s",exc)
        return False


try:
    load_persisted_state()
except Exception:
    pass

__all__=[
    "FINAL_BRAIN_VERSION","BRAIN_INTERFACE_VERSION","FULL_LEARNING_SCHEMA","MACHINE_LEARNING_SCHEMA","BRAIN_CHECKPOINT_SCHEMA",
    "MIN_RR","MAX_RR","TRAIL_R_LADDER","STRUCT_TRAIL_LB","STRUCT_TRAIL_BUF_PCT","STRUCT_TRAIL_LOOKBACK","FIB_EXT_1","FIB_EXT_2",
    "full_analyze","manage_position","score_direction","swing_pts","mkt_struct",
    "record_candidate_observation","ingest_live_candidate","record_trade_outcome","ingest_live_outcome","record_protection_event",
    "record_scan_summary","evaluate_stats_decision","set_learning_model","get_learning_model_info","get_learning_schema",
    "export_checkpoint_state","import_checkpoint_state","get_brain_state","apply_brain_state",
    "get_cognitive_status","get_full_cognitive_status","get_adaptive_status","get_experience_count",
    "get_strategy_evolution_status","get_active_confidence_threshold","set_manual_confidence_threshold","suggest_confidence_threshold",
    "reset_cognitive_memory","reset_adaptive_learning","full_command","adaptive_agent_start","adaptive_agent_stop",
    "ema","rsi","atr_fn","build_df","fib_position","detect_bos","detect_choch","detect_cisd","detect_liquidity_sweep",
    "detect_inducement","detect_fvg","detect_order_blocks",
]
