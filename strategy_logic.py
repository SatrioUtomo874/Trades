#!/usr/bin/env python3
"""
strategy_logic.py — ADAPTIVE STRATEGY BRAIN

Contract target: main_BODY_V50_BOUNDARY_FINAL.py

Design goals
------------
- Strategy logic is the decision brain; it never calls Binance mutation APIs.
- `full_analyze()` produces an entry decision packet compatible with main.py.
- `manage_position()` produces management intent only; main.py owns execution.
- Learning follows an explicit epistemic lifecycle:
  observation -> evidence -> pattern -> hypothesis -> experiment -> finding -> belief -> policy change.
- Research is bounded and synchronous by default; it never creates its own worker pool.
- State is serializable through export/import checkpoint APIs used by main.py.
- Low memory footprint: deques, bounded queues, compact summaries, lazy historical loading.

This implementation intentionally keeps the brain self-contained and dependency-light
(pandas + numpy only) so it can run inside the same Render process as main.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import statistics
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Public version / contract
# -----------------------------------------------------------------------------
BRAIN_INTERFACE_VERSION = "brain_v1"
BRAIN_SCHEMA_VERSION = "brain_state_v1"
ENGINE_NAME = "AdaptiveStrategyBrain"

# Hard resource boundaries. The main process owns the actual global worker
# governor. This module never creates a worker pool; it only limits internal
# queues and challenger counts.
MAX_HEAVY_WORKERS = 5
MAX_CANDIDATE_MEMORY = 5000
MAX_OUTCOME_MEMORY = 2000
MAX_PATTERN_MEMORY = 500
MAX_HYPOTHESIS_MEMORY = 200
MAX_EXPERIMENT_MEMORY = 100
MAX_CHALLENGER_MEMORY = 12
MAX_EVIDENCE_MEMORY = 5000
MAX_RESEARCH_QUEUE = 200
MAX_OLLAMA_QUEUE = 40

DEFAULT_DECISION_TTL_SEC = 20.0
DEFAULT_MANAGEMENT_TTL_SEC = 25.0
MIN_RR = 2.0

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def _finite(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _optional_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=max(2, span // 3)).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(3, period // 2)).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_dn = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _normalize_ohlcv(df: Any) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = df.copy()
    aliases = {str(c).lower(): c for c in out.columns}
    rename = {}
    for want in ("open", "high", "low", "close", "volume"):
        if want not in out.columns:
            src = aliases.get(want)
            if src is not None:
                rename[src] = want
    if rename:
        out = out.rename(columns=rename)
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in out.columns]
    if missing:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.loc[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _safe_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return _safe_jsonable(obj.item())
    if isinstance(obj, Mapping):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, deque)):
        return [_safe_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _percent_change(a: float, b: float) -> float:
    return ((b / a) - 1.0) * 100.0 if a else 0.0


def _quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.quantile(vals, q)) if vals else float(default)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float(default)


def _median(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else float(default)


# -----------------------------------------------------------------------------
# Explicit brain objects
# -----------------------------------------------------------------------------
@dataclass
class StrategyProfile:
    version: str
    parent_version: str
    entry_rules: dict[str, Any]
    preferences: dict[str, Any]
    regime_rules: dict[str, Any]
    management_rules: dict[str, Any]
    confidence_policy: dict[str, Any]
    created_at: float
    change_reason: str = "initial"
    hypothesis_id: Optional[str] = None
    experiment_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return _safe_jsonable(asdict(self))


@dataclass
class Evidence:
    evidence_id: str
    source: str
    timestamp: float
    sample_size: int
    freshness: float
    data_quality: float
    regime: str
    independence: float
    confidence: float
    statement: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _safe_jsonable(asdict(self))


@dataclass
class Hypothesis:
    hypothesis_id: str
    claim: str
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    sample_size: int
    confidence: float
    regimes: list[str]
    expected_change: str
    status: str
    created_at: float
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return _safe_jsonable(asdict(self))


@dataclass
class Experiment:
    experiment_id: str
    hypothesis_id: str
    base_strategy: str
    challenger_versions: list[str]
    design: dict[str, Any]
    result: dict[str, Any]
    status: str
    created_at: float
    completed_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return _safe_jsonable(asdict(self))


# -----------------------------------------------------------------------------
# Brain state
# -----------------------------------------------------------------------------
_STATE_LOCK = threading.RLock()

_DEFAULT_PROFILE = StrategyProfile(
    version="S01",
    parent_version="S00",
    entry_rules={
        "min_rr": MIN_RR,
        "min_structure": 0.52,
        "min_alignment": 0.48,
        "max_entry_extension_atr": 1.25,
        "require_location_edge": True,
        "allow_transition": True,
    },
    preferences={
        "preferred_setup": "trend_continuation",
        "avoid_extreme_extension": True,
        "prefer_liquidity_confirmation": True,
    },
    regime_rules={
        "TREND_UP": 1.08,
        "TREND_DOWN": 1.08,
        "RANGE": 0.96,
        "TRANSITION": 0.92,
        "EXPANSION": 1.05,
        "CONTRACTION": 0.92,
        "HIGH_VOLATILITY": 0.90,
        "LOW_VOLATILITY": 0.94,
    },
    management_rules={
        "trail_start_r": 1.0,
        "trail_lock_r": 0.35,
        "trail_buffer_atr": 0.35,
        "giveback_tolerance_r": 0.60,
        "extension_hold_r": 2.0,
    },
    confidence_policy={
        "baseline": 60.0,
        "near_threshold_band": 8.0,
        "calibration_strength": 0.25,
    },
    created_at=_now(),
)

_ACTIVE_STRATEGY = copy.deepcopy(_DEFAULT_PROFILE)
_CHAMPION = copy.deepcopy(_DEFAULT_PROFILE)

_candidates: deque[dict[str, Any]] = deque(maxlen=MAX_CANDIDATE_MEMORY)
_outcomes: deque[dict[str, Any]] = deque(maxlen=MAX_OUTCOME_MEMORY)
_evidence: deque[dict[str, Any]] = deque(maxlen=MAX_EVIDENCE_MEMORY)
_patterns: deque[dict[str, Any]] = deque(maxlen=MAX_PATTERN_MEMORY)
_hypotheses: dict[str, dict[str, Any]] = {}
_experiments: dict[str, dict[str, Any]] = {}
_challengers: dict[str, dict[str, Any]] = {}
_beliefs: dict[str, dict[str, Any]] = {}
_strategy_history: deque[dict[str, Any]] = deque(maxlen=100)
_research_journal: deque[dict[str, Any]] = deque(maxlen=1500)
_research_queue: deque[dict[str, Any]] = deque(maxlen=MAX_RESEARCH_QUEUE)
_ollama_queue: deque[dict[str, Any]] = deque(maxlen=MAX_OLLAMA_QUEUE)
_frequency_state: dict[str, Any] = {}
_drift_state: dict[str, Any] = {}
_calibration_state: dict[str, Any] = {
    "bins": {str(i): {"n": 0, "wins": 0, "r_sum": 0.0} for i in range(10, 101, 10)},
    "last_update": 0.0,
}
_brain_stats = {
    "observations": 0,
    "candidates": 0,
    "outcomes": 0,
    "patterns": 0,
    "hypotheses_created": 0,
    "experiments_completed": 0,
    "challengers_created": 0,
    "promotions": 0,
    "rejections": 0,
    "missed_opportunities": 0,
    "shadow_outcomes": 0,
    "last_learning_at": None,
    "last_strategy_change_at": None,
}


# -----------------------------------------------------------------------------
# Logging / journal
# -----------------------------------------------------------------------------

def _brain_log(level: int, message: str) -> None:
    log.log(level, message)


def _journal(event_type: str, payload: Mapping[str, Any], telegram: bool = False) -> None:
    row = {
        "journal_id": _id("JRN"),
        "event_type": str(event_type),
        "timestamp": _now(),
        "telegram": bool(telegram),
        "payload": _safe_jsonable(dict(payload)),
    }
    with _STATE_LOCK:
        _research_journal.append(row)
    if event_type == "research":
        _brain_log(logging.INFO, f"[BRAIN RESEARCH] {payload.get('message', event_type)}")
    elif event_type == "experiment":
        _brain_log(logging.INFO, f"[BRAIN EXPERIMENT] {payload.get('message', event_type)}")
    elif event_type == "evolution":
        _brain_log(logging.INFO, f"[BRAIN EVOLUTION] {payload.get('message', event_type)}")
    elif event_type == "promotion":
        _brain_log(logging.INFO, f"[BRAIN PROMOTION] {payload.get('message', event_type)}")


# -----------------------------------------------------------------------------
# Perception / context / regime
# -----------------------------------------------------------------------------

def _clean_snapshot_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_ohlcv(df)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["ema9"] = _ema(frame["close"], 9)
    frame["ema21"] = _ema(frame["close"], 21)
    frame["ema50"] = _ema(frame["close"], 50)
    frame["atr"] = _atr(frame, 14)
    frame["rsi"] = _rsi(frame["close"], 14)
    frame["range"] = (frame["high"] - frame["low"]).abs()
    vol_ma = frame["volume"].rolling(20, min_periods=5).mean()
    frame["relative_volume"] = frame["volume"] / vol_ma.replace(0, np.nan)
    return frame


def _structure_descriptor(df: pd.DataFrame) -> dict[str, Any]:
    f = _clean_snapshot_frame(df)
    if len(f) < 10:
        return {
            "trend": "UNKNOWN", "structure_quality": 0.0, "bos": False,
            "choch": False, "displacement": 0.0, "swing_high": None,
            "swing_low": None,
        }
    c = f["close"]
    recent = f.tail(min(80, len(f)))
    last = float(c.iloc[-1])
    ema9, ema21, ema50 = [float(f[x].iloc[-1]) for x in ("ema9", "ema21", "ema50")]
    atr = max(float(f["atr"].iloc[-1] or 0), last * 1e-5)
    slope21 = _finite(f["ema21"].iloc[-1] - f["ema21"].iloc[-6], 0.0) if len(f) >= 6 else 0.0
    slope50 = _finite(f["ema50"].iloc[-1] - f["ema50"].iloc[-11], 0.0) if len(f) >= 11 else 0.0
    high_ref = float(recent["high"].iloc[:-3].max()) if len(recent) > 3 else float(recent["high"].max())
    low_ref = float(recent["low"].iloc[:-3].min()) if len(recent) > 3 else float(recent["low"].min())
    bos_up = last > high_ref
    bos_down = last < low_ref
    up_alignment = 0.0
    down_alignment = 0.0
    if ema9 > ema21: up_alignment += 0.25
    if ema21 > ema50: up_alignment += 0.25
    if slope21 > 0: up_alignment += 0.20
    if slope50 > 0: up_alignment += 0.15
    if last > ema50: up_alignment += 0.15
    if ema9 < ema21: down_alignment += 0.25
    if ema21 < ema50: down_alignment += 0.25
    if slope21 < 0: down_alignment += 0.20
    if slope50 < 0: down_alignment += 0.15
    if last < ema50: down_alignment += 0.15
    trend = "TREND_UP" if up_alignment >= 0.68 else "TREND_DOWN" if down_alignment >= 0.68 else "RANGE"
    disp = _finite(abs(float(c.iloc[-1]) - float(c.iloc[-4])) / atr, 0.0) if len(f) >= 4 else 0.0
    structure_quality = _clamp(max(up_alignment, down_alignment) * 0.7 + min(disp / 2.5, 1.0) * 0.3, 0, 1)
    range_high = float(recent["high"].max())
    range_low = float(recent["low"].min())
    location = _clamp((last - range_low) / max(range_high - range_low, 1e-9), 0, 1)
    return {
        "trend": trend,
        "up_alignment": round(up_alignment, 4),
        "down_alignment": round(down_alignment, 4),
        "structure_quality": round(structure_quality, 4),
        "bos": bool(bos_up or bos_down),
        "bos_up": bool(bos_up),
        "bos_down": bool(bos_down),
        "choch": bool((trend == "TREND_UP" and last < ema21) or (trend == "TREND_DOWN" and last > ema21)),
        "displacement": round(disp, 4),
        "swing_high": high_ref,
        "swing_low": low_ref,
        "range_high": range_high,
        "range_low": range_low,
        "range_position": round(location, 4),
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "atr": atr,
        "rsi": float(f["rsi"].iloc[-1]),
        "relative_volume": _finite(f["relative_volume"].iloc[-1], 1.0),
    }


def _regime_descriptor(h1: pd.DataFrame, m15: pd.DataFrame) -> dict[str, Any]:
    h = _clean_snapshot_frame(h1)
    m = _clean_snapshot_frame(m15)
    if h.empty or m.empty:
        return {"primary": "UNKNOWN", "descriptors": [], "volatility": "UNKNOWN"}
    hs = _structure_descriptor(h)
    ms = _structure_descriptor(m)
    atr_pct = _finite(ms["atr"] / float(m["close"].iloc[-1]) * 100.0, 0.0)
    recent_ranges = m["range"].tail(30)
    range_median = _median(recent_ranges.tolist(), default=0.0)
    last_range = _finite(m["range"].iloc[-1], 0.0)
    expansion = last_range > range_median * 1.35 if range_median else False
    contraction = last_range < range_median * 0.72 if range_median else False
    descriptors = []
    primary = hs["trend"] if hs["trend"] in {"TREND_UP", "TREND_DOWN"} else ms["trend"]
    if primary == "RANGE":
        primary = "RANGE"
    elif primary not in {"TREND_UP", "TREND_DOWN"}:
        primary = "TRANSITION"
    if expansion:
        descriptors.append("EXPANSION")
    if contraction:
        descriptors.append("CONTRACTION")
    atr_series = _atr(m, 14)
    vol_samples = []
    for j in range(max(0, len(m) - 50), len(m)):
        atr_j = _optional_float(atr_series.iloc[j])
        close_j = _optional_float(m["close"].iloc[j])
        if atr_j is not None and close_j and close_j > 0:
            vol_samples.append(atr_j / close_j * 100.0)
    vol_q70 = _quantile(vol_samples, 0.70, atr_pct)
    vol = "HIGH_VOLATILITY" if atr_pct >= vol_q70 else "LOW_VOLATILITY"
    if atr_pct > 1.5:
        vol = "HIGH_VOLATILITY"
    elif atr_pct < 0.35:
        vol = "LOW_VOLATILITY"
    descriptors.append(vol)
    if ms["choch"]:
        descriptors.append("TRANSITION")
    return {
        "primary": primary,
        "descriptors": list(dict.fromkeys(descriptors)),
        "volatility": vol,
        "atr_pct": atr_pct,
        "h1_structure": hs,
        "m15_structure": ms,
    }


def _time_context(timestamp: Optional[float] = None) -> dict[str, Any]:
    ts = time.localtime(timestamp or _now())
    hour = int(ts.tm_hour)
    # UTC-based bucket keeps deployment timezone-independent; user can still
    # learn relative session behavior from actual outcomes.
    if 0 <= hour < 7:
        session = "ASIA"
    elif 7 <= hour < 13:
        session = "LONDON"
    elif 13 <= hour < 18:
        session = "NY"
    else:
        session = "NY_CLOSE"
    return {"hour": hour, "weekday": int(ts.tm_wday), "session": session}


def _btc_context(analysis_input: Any, h1: pd.DataFrame, m15: pd.DataFrame) -> dict[str, Any]:
    if isinstance(analysis_input, Mapping):
        explicit = dict(analysis_input)
    else:
        explicit = {}
    btc_bias = str(explicit.get("btc_bias") or explicit.get("btc_direction") or "UNKNOWN").upper()
    btc_momentum = _optional_float(explicit.get("btc_momentum"))
    return {
        "bias": btc_bias,
        "momentum": btc_momentum,
        "source": "provided_context" if explicit else "unavailable_in_brain",
    }


def perceive_market(symbol: str, h1: pd.DataFrame, m15: pd.DataFrame, d1: Optional[pd.DataFrame] = None, context: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    h = _clean_snapshot_frame(h1)
    m = _clean_snapshot_frame(m15)
    d = _clean_snapshot_frame(d1) if d1 is not None else pd.DataFrame()
    if h.empty or m.empty or len(m) < 25:
        raise ValueError(f"insufficient market data for {symbol}")
    hs = _structure_descriptor(h)
    ms = _structure_descriptor(m)
    ds = _structure_descriptor(d) if not d.empty else {}
    regime = _regime_descriptor(h, m)
    snapshot = {
        "snapshot_id": _id("MS").lower(),
        "symbol": str(symbol),
        "timestamp": _now(),
        "m15_bars": len(m),
        "h1_bars": len(h),
        "d1_bars": len(d),
        "m15": ms,
        "h1": hs,
        "d1": ds,
        "regime": regime,
        "time": _time_context(),
        "btc": _btc_context(context, h, m),
        "close": float(m["close"].iloc[-1]),
        "data_quality": _data_quality(m, h, d),
    }
    with _STATE_LOCK:
        _brain_stats["observations"] += 1
    return _safe_jsonable(snapshot)


def _data_quality(m15: pd.DataFrame, h1: pd.DataFrame, d1: pd.DataFrame) -> float:
    def frame_quality(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.5
        finite = np.isfinite(df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)).mean()
        duplicate = 0.0 if df.index.has_duplicates else 0.15
        ordered = 0.15 if df.index.is_monotonic_increasing else 0.0
        return _clamp(0.7 * finite + duplicate + ordered, 0, 1)
    weights = [frame_quality(m15), frame_quality(h1), frame_quality(d1) if not d1.empty else 0.7]
    return round(sum(weights) / len(weights), 4)


# -----------------------------------------------------------------------------
# Liquidity / setup discovery
# -----------------------------------------------------------------------------

def _liquidity_features(m15: pd.DataFrame) -> dict[str, Any]:
    f = _clean_snapshot_frame(m15)
    if len(f) < 20:
        return {"sweep_up": False, "sweep_down": False, "liquidity_score": 0.0}
    look = f.tail(min(60, len(f)))
    last = look.iloc[-1]
    prev = look.iloc[:-1]
    swing_high = float(prev["high"].tail(20).max())
    swing_low = float(prev["low"].tail(20).min())
    sweep_up = float(last["high"]) > swing_high and float(last["close"]) < swing_high
    sweep_down = float(last["low"]) < swing_low and float(last["close"]) > swing_low
    equal_high = abs(float(prev["high"].tail(12).max()) - swing_high) <= float(last["atr"] or 0) * 0.15
    equal_low = abs(float(prev["low"].tail(12).min()) - swing_low) <= float(last["atr"] or 0) * 0.15
    score = 0.0
    score += 0.35 if sweep_up or sweep_down else 0.0
    score += 0.20 if equal_high or equal_low else 0.0
    rv = _finite(last.get("relative_volume"), 1.0)
    score += 0.25 if rv >= 1.2 else 0.0
    score += 0.20 if float(last["range"]) >= _median(look["range"].tail(20).tolist(), 0.0) * 1.1 else 0.0
    return {
        "sweep_up": bool(sweep_up),
        "sweep_down": bool(sweep_down),
        "equal_high": bool(equal_high),
        "equal_low": bool(equal_low),
        "swing_high": swing_high,
        "swing_low": swing_low,
        "liquidity_score": round(_clamp(score, 0, 1), 4),
    }


def discover_setups(snapshot: Mapping[str, Any], h1: pd.DataFrame, m15: pd.DataFrame) -> list[dict[str, Any]]:
    m = _clean_snapshot_frame(m15)
    h = _clean_snapshot_frame(h1)
    ms = dict(snapshot.get("m15") or {})
    hs = dict(snapshot.get("h1") or {})
    regime = dict(snapshot.get("regime") or {})
    liq = _liquidity_features(m)
    last = float(m["close"].iloc[-1])
    atr = max(_finite(ms.get("atr"), 0.0), last * 1e-5)
    range_pos = _finite(ms.get("range_position"), 0.5)
    candidates: list[dict[str, Any]] = []
    # Trend continuation / breakout retest.
    if hs.get("trend") == "TREND_UP" and ms.get("up_alignment", 0) >= 0.55:
        entry = last
        sl = min(_finite(liq.get("swing_low"), last - 1.0*atr), last - 0.9*atr)
        tp = entry + max(2.0 * abs(entry - sl), 1.8 * atr)
        candidates.append(_setup_candidate(snapshot, "trend_continuation", "BUY", entry, sl, tp, liq, 0.68, ["h1_uptrend", "m15_alignment"]))
    if hs.get("trend") == "TREND_DOWN" and ms.get("down_alignment", 0) >= 0.55:
        entry = last
        sl = max(_finite(liq.get("swing_high"), last + 1.0*atr), last + 0.9*atr)
        tp = entry - max(2.0 * abs(entry - sl), 1.8 * atr)
        candidates.append(_setup_candidate(snapshot, "trend_continuation", "SELL", entry, sl, tp, liq, 0.68, ["h1_downtrend", "m15_alignment"]))
    # Liquidity sweep reversal.
    if liq.get("sweep_down"):
        entry = last
        sl = min(entry - 0.8*atr, _finite(liq.get("swing_low"), entry - 0.8*atr))
        tp = entry + max(2.1 * abs(entry - sl), 1.7 * atr)
        candidates.append(_setup_candidate(snapshot, "liquidity_sweep_reversal", "BUY", entry, sl, tp, liq, 0.72, ["sweep_down", "reclaim"]))
    if liq.get("sweep_up"):
        entry = last
        sl = max(entry + 0.8*atr, _finite(liq.get("swing_high"), entry + 0.8*atr))
        tp = entry - max(2.1 * abs(entry - sl), 1.7 * atr)
        candidates.append(_setup_candidate(snapshot, "liquidity_sweep_reversal", "SELL", entry, sl, tp, liq, 0.72, ["sweep_up", "reclaim"]))
    # Range rejection: only when clearly in range, and avoid midrange.
    if regime.get("primary") == "RANGE":
        if range_pos <= 0.25:
            entry = last; sl = min(last - 0.9*atr, float(ms.get("swing_low") or last - atr)); tp = last + 2.0*abs(last-sl)
            candidates.append(_setup_candidate(snapshot, "range_rejection", "BUY", entry, sl, tp, liq, 0.61, ["range_low_location"]))
        elif range_pos >= 0.75:
            entry = last; sl = max(last + 0.9*atr, float(ms.get("swing_high") or last + atr)); tp = last - 2.0*abs(last-sl)
            candidates.append(_setup_candidate(snapshot, "range_rejection", "SELL", entry, sl, tp, liq, 0.61, ["range_high_location"]))
    # Momentum continuation when displacement and volume align.
    if _finite(ms.get("displacement"), 0) >= 1.2 and _finite(ms.get("relative_volume"), 1) >= 1.15:
        if ms.get("up_alignment", 0) > ms.get("down_alignment", 0):
            entry = last; sl = last - 0.95*atr; tp = last + 2.2*abs(last-sl)
            candidates.append(_setup_candidate(snapshot, "momentum_continuation", "BUY", entry, sl, tp, liq, 0.64, ["displacement", "relative_volume"]))
        elif ms.get("down_alignment", 0) > ms.get("up_alignment", 0):
            entry = last; sl = last + 0.95*atr; tp = last - 2.2*abs(last-sl)
            candidates.append(_setup_candidate(snapshot, "momentum_continuation", "SELL", entry, sl, tp, liq, 0.64, ["displacement", "relative_volume"]))
    # Deduplicate by direction/setup; keep strongest raw evidence.
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for c in candidates:
        key = (c["setup_family"], c["direction"])
        if key not in best or c["raw_quality"] > best[key]["raw_quality"]:
            best[key] = c
    return sorted(best.values(), key=lambda x: x["raw_quality"], reverse=True)


def _setup_candidate(snapshot: Mapping[str, Any], family: str, direction: str, entry: float, sl: float, tp: float, liquidity: Mapping[str, Any], quality: float, reasons: list[str]) -> dict[str, Any]:
    risk = abs(entry - sl)
    rr = abs(tp - entry) / risk if risk else 0.0
    return {
        "candidate_id": _id("CAND").lower(),
        "symbol": str(snapshot.get("symbol")),
        "timestamp": _now(),
        "direction": direction,
        "setup_family": family,
        "market_context": dict(snapshot.get("regime") or {}),
        "regime": dict(snapshot.get("regime") or {}),
        "entry_zone": [entry - abs(entry-sl)*0.15, entry + abs(entry-sl)*0.15],
        "entry_price": float(entry),
        "potential_SL": float(sl),
        "potential_TP": float(tp),
        "feature_snapshot": {
            "m15": snapshot.get("m15"),
            "h1": snapshot.get("h1"),
            "btc": snapshot.get("btc"),
            "time": snapshot.get("time"),
            "liquidity": dict(liquidity),
        },
        "raw_confidence": float(_clamp(quality*100, 0, 100)),
        "raw_quality": float(quality),
        "brain_decision": "PENDING",
        "rejection_reason": None,
        "reason_codes": list(reasons),
        "rr": rr,
        "immutable": True,
    }


# -----------------------------------------------------------------------------
# Thesis / scoring / confidence
# -----------------------------------------------------------------------------

def _historical_similarity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    family = str(candidate.get("setup_family") or "unknown")
    direction = str(candidate.get("direction") or "")
    regime = str((candidate.get("regime") or {}).get("primary") or "UNKNOWN")
    with _STATE_LOCK:
        relevant = [x for x in _outcomes if x.get("setup_family") == family and x.get("direction") == direction]
        regime_rows = [x for x in relevant if x.get("regime") == regime]
    rs = [_finite(x.get("r_multiple"), 0.0) for x in regime_rows]
    all_rs = [_finite(x.get("r_multiple"), 0.0) for x in relevant]
    n = len(regime_rows)
    expectancy = _mean(rs, _mean(all_rs, 0.0))
    # Evidence strength grows slowly with sample size to avoid over-trusting tiny n.
    strength = _clamp(math.sqrt(n / 25.0), 0.0, 1.0)
    return {"sample": n, "expectancy_r": expectancy, "strength": strength, "all_sample": len(relevant)}


def evaluate_thesis(candidate: Mapping[str, Any], snapshot: Mapping[str, Any], profile: StrategyProfile) -> dict[str, Any]:
    direction = candidate["direction"]
    m15 = snapshot.get("m15") or {}
    h1 = snapshot.get("h1") or {}
    regime = snapshot.get("regime") or {}
    liq = (candidate.get("feature_snapshot") or {}).get("liquidity") or {}
    btc = snapshot.get("btc") or {}
    rr = _finite(candidate.get("rr"), 0.0)
    alignment = _finite(m15.get("up_alignment" if direction == "BUY" else "down_alignment"), 0.0)
    structure = _finite(h1.get("up_alignment" if direction == "BUY" else "down_alignment"), 0.0)
    liquidity = _finite(liq.get("liquidity_score"), 0.0)
    regime_multiplier = 1.0
    descriptors = list(regime.get("descriptors") or [])
    for descriptor in descriptors:
        regime_multiplier *= _finite(profile.regime_rules.get(descriptor), 1.0)
    primary = str(regime.get("primary") or "UNKNOWN")
    regime_multiplier *= _finite(profile.regime_rules.get(primary), 1.0)
    btc_alignment = 0.5
    btc_bias = str(btc.get("bias") or "UNKNOWN").upper()
    if btc_bias in {"BULLISH", "BUY", "UP"}:
        btc_alignment = 0.75 if direction == "BUY" else 0.35
    elif btc_bias in {"BEARISH", "SELL", "DOWN"}:
        btc_alignment = 0.75 if direction == "SELL" else 0.35
    hist = _historical_similarity(candidate)
    hist_boost = _clamp(0.5 + hist["expectancy_r"] * 0.08, 0.2, 0.8) * hist["strength"] + 0.5 * (1-hist["strength"])
    extension_atr = abs(float(candidate["entry_price"]) - float(h1.get("ema21") or candidate["entry_price"])) / max(_finite(m15.get("atr"), 1e-9), 1e-9)
    location = _finite(m15.get("range_position"), 0.5)
    location_edge = (0.75 if (direction == "BUY" and location < 0.60) or (direction == "SELL" and location > 0.40) else 0.45)
    scores = {
        "setup_quality": _finite(candidate.get("raw_quality"), 0.5),
        "structure_quality": structure,
        "alignment_quality": alignment,
        "liquidity_quality": liquidity,
        "regime_compatibility": _clamp(regime_multiplier, 0.55, 1.15),
        "btc_alignment": btc_alignment,
        "location_quality": location_edge,
        "historical_evidence": hist_boost,
        "rr_quality": _clamp(rr / 3.5, 0.0, 1.0),
        "extension_penalty": _clamp(extension_atr / max(profile.entry_rules.get("max_entry_extension_atr", 1.25), 0.1), 0.0, 2.0),
    }
    weighted = (
        scores["setup_quality"] * 0.18 +
        scores["structure_quality"] * 0.16 +
        scores["alignment_quality"] * 0.15 +
        scores["liquidity_quality"] * 0.10 +
        _clamp(scores["regime_compatibility"], 0, 1) * 0.10 +
        scores["btc_alignment"] * 0.08 +
        scores["location_quality"] * 0.08 +
        scores["historical_evidence"] * 0.08 +
        scores["rr_quality"] * 0.07
    )
    weighted -= max(0.0, scores["extension_penalty"] - 1.0) * 0.10
    weighted = _clamp(weighted, 0, 1)
    thesis = {
        "claim": f"{candidate['setup_family']} {direction} is supported by current structure, context and location",
        "invalidation": float(candidate["potential_SL"]),
        "scores": scores,
        "historical": hist,
        "regime_multiplier": regime_multiplier,
        "decision_score": weighted,
        "thesis_quality": weighted,
        "reason_codes": list(candidate.get("reason_codes") or []),
    }
    if extension_atr > _finite(profile.entry_rules.get("max_entry_extension_atr"), 1.25):
        thesis["reason_codes"].append("price_extended_from_ema21")
    if rr < _finite(profile.entry_rules.get("min_rr"), MIN_RR):
        thesis["reason_codes"].append("rr_below_minimum")
    return thesis


def _calibrated_confidence(raw_confidence: float, candidate: Mapping[str, Any], thesis: Mapping[str, Any]) -> float:
    family = str(candidate.get("setup_family") or "unknown")
    with _STATE_LOCK:
        rows = [x for x in _outcomes if x.get("setup_family") == family]
    if len(rows) < 5:
        return _clamp(raw_confidence * 0.96 + _finite(thesis.get("decision_score"), 0.5) * 4.0, 0, 100)
    rs = [_finite(x.get("r_multiple"), 0.0) for x in rows]
    positive = sum(1 for r in rs if r > 0) / len(rs)
    expectancy = _mean(rs, 0.0)
    empirical = _clamp(50 + positive*25 + expectancy*8, 0, 100)
    strength = _clamp(len(rows)/30.0, 0, 1)
    return _clamp(raw_confidence*(1-strength*0.35) + empirical*(strength*0.35), 0, 100)


def _effective_decision_score(thesis: Mapping[str, Any], calibrated_confidence: float) -> float:
    return _clamp(_finite(thesis.get("decision_score"), 0.0) * 100 * 0.65 + calibrated_confidence * 0.35, 0, 100)


def _current_frequency_state() -> dict[str, Any]:
    with _STATE_LOCK:
        rows = list(_candidates)
    total = len(rows)
    qualified = sum(1 for r in rows if str(r.get("brain_decision")) == "TRADE")
    near = sum(1 for r in rows if str(r.get("brain_decision")) in {"TRADE", "REJECT", "WATCH"} and abs(_finite(r.get("confidence"), 0) - _active_baseline()) <= 8)
    return {"candidate_supply": total, "qualified_supply": qualified, "near_threshold_supply": near, "acceptance_rate": qualified/total if total else 0.0}


def _active_baseline() -> float:
    with _STATE_LOCK:
        return _clamp(_finite(_ACTIVE_STRATEGY.confidence_policy.get("baseline"), 60), 0, 100)


def _candidate_decision(candidate: dict[str, Any], thesis: dict[str, Any], calibrated: float, effective: float, snapshot: Mapping[str, Any]) -> str:
    rr = _finite(candidate.get("rr"), 0.0)
    min_rr = _finite(_ACTIVE_STRATEGY.entry_rules.get("min_rr"), MIN_RR)
    baseline = _active_baseline()
    quality = _finite(thesis.get("thesis_quality"), 0.0)
    if rr < min_rr:
        return "REJECT"
    if quality < 0.47 or effective < baseline - 10:
        return "REJECT"
    if effective < baseline + 2:
        return "WATCH"
    return "TRADE"


def build_decision(candidate: dict[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    profile = copy.deepcopy(_ACTIVE_STRATEGY)
    thesis = evaluate_thesis(candidate, snapshot, profile)
    raw = _finite(candidate.get("raw_confidence"), 0.0)
    calibrated = _calibrated_confidence(raw, candidate, thesis)
    effective = _effective_decision_score(thesis, calibrated)
    decision = _candidate_decision(candidate, thesis, calibrated, effective, snapshot)
    now = _now()
    candidate_id = str(candidate["candidate_id"])
    analysis_id = _id("AN")
    decision_id = _id("DEC")
    direction = candidate["direction"]
    trade_decision = direction if decision == "TRADE" else decision
    baseline = _active_baseline()
    expires = now + DEFAULT_DECISION_TTL_SEC
    reason_codes = list(dict.fromkeys(list(thesis.get("reason_codes") or []) + [f"decision_{decision.lower()}"]))
    if decision != "TRADE":
        candidate["rejection_reason"] = ",".join(reason_codes)
    packet = {
        "decision_id": decision_id,
        "analysis_id": analysis_id,
        "candidate_id": candidate_id,
        "strategy_version": profile.version,
        "direction": direction,
        "decision": trade_decision,
        "setup_family": candidate.get("setup_family"),
        "entry": float(candidate["entry_price"]),
        "sl": float(candidate["potential_SL"]),
        "tp": float(candidate["potential_TP"]),
        "initial_sl": float(candidate["potential_SL"]),
        "rr": float(candidate["rr"]),
        "confidence": round(calibrated, 2),
        "raw_confidence": round(raw, 2),
        "calibrated_confidence": round(calibrated, 2),
        "effective_quality": round(effective, 2),
        "decision_score": round(effective, 2),
        "reason_codes": reason_codes,
        "market_snapshot_time": float(snapshot.get("timestamp") or now),
        "decision_created_at": now,
        "decision_expires_at": expires,
        "created_at": now,
        "expires_at": expires,
        "execution_mode": "SIMULATION",
        "entry_label": "strategy",
        "rsi": _finite((snapshot.get("m15") or {}).get("rsi"), 50),
        "atr": _finite((snapshot.get("m15") or {}).get("atr"), 0),
        "struct_h1": (snapshot.get("h1") or {}).get("trend"),
        "d1_bias": (snapshot.get("d1") or {}).get("trend") if snapshot.get("d1") else None,
        "thesis": thesis,
        "evidence_ids": [],
        "market_context": snapshot,
        "active_policy": {
            "baseline_confidence": baseline,
            "strategy_version": profile.version,
        },
        "learning_features": _extract_learning_features(candidate, snapshot, thesis),
    }
    return _safe_jsonable(packet)


def _extract_learning_features(candidate: Mapping[str, Any], snapshot: Mapping[str, Any], thesis: Mapping[str, Any]) -> dict[str, Any]:
    m = snapshot.get("m15") or {}
    h = snapshot.get("h1") or {}
    regime = snapshot.get("regime") or {}
    return {
        "setup_family": candidate.get("setup_family"),
        "direction": candidate.get("direction"),
        "structure_quality": _finite(thesis.get("scores", {}).get("structure_quality"), 0),
        "alignment_quality": _finite(thesis.get("scores", {}).get("alignment_quality"), 0),
        "liquidity_quality": _finite(thesis.get("scores", {}).get("liquidity_quality"), 0),
        "range_position": _finite(m.get("range_position"), 0.5),
        "relative_volume": _finite(m.get("relative_volume"), 1),
        "rsi": _finite(m.get("rsi"), 50),
        "atr": _finite(m.get("atr"), 0),
        "atr_pct": _finite(regime.get("atr_pct"), 0),
        "h1_trend": h.get("trend"),
        "regime": regime.get("primary"),
        "regime_descriptors": list(regime.get("descriptors") or []),
        "session": (snapshot.get("time") or {}).get("session"),
        "hour": (snapshot.get("time") or {}).get("hour"),
        "btc_bias": (snapshot.get("btc") or {}).get("bias"),
    }


# -----------------------------------------------------------------------------
# Public entry contract
# -----------------------------------------------------------------------------
def full_analyze(df_h1: pd.DataFrame, df_m15: pd.DataFrame, df_d1: Optional[pd.DataFrame] = None, symbol: Optional[str] = None, **kwargs: Any) -> dict[str, Any]:
    """Analyze current market and return a main.py-compatible decision packet.

    This function never calls Binance, never mutates exchange state, and never
    starts a thread. If no valid opportunity is found, it returns WAIT/REJECT
    rather than fabricating a trade.
    """
    sym = str(symbol or kwargs.get("symbol") or "UNKNOWN").upper()
    snapshot = perceive_market(sym, df_h1, df_m15, df_d1, kwargs.get("market_context"))
    setups = discover_setups(snapshot, df_h1, df_m15)
    if not setups:
        now = _now()
        return _safe_jsonable({
            "decision_id": _id("DEC"), "analysis_id": _id("AN"), "candidate_id": None,
            "symbol": sym, "strategy_version": _ACTIVE_STRATEGY.version,
            "decision": "WAIT", "direction": None,
            "entry": float(snapshot["close"]), "sl": None, "tp": None,
            "confidence": 0.0, "raw_confidence": 0.0, "calibrated_confidence": 0.0,
            "effective_quality": 0.0, "decision_score": 0.0,
            "reason_codes": ["no_valid_setup"], "market_snapshot_time": snapshot["timestamp"],
            "decision_created_at": now, "decision_expires_at": now+DEFAULT_DECISION_TTL_SEC,
            "created_at": now, "expires_at": now+DEFAULT_DECISION_TTL_SEC,
            "entry_label": "none", "atr": snapshot["m15"].get("atr"),
            "rsi": snapshot["m15"].get("rsi"), "struct_h1": snapshot["h1"].get("trend"),
            "d1_bias": snapshot.get("d1", {}).get("trend") if snapshot.get("d1") else None,
            "market_context": snapshot, "learning_features": {}, "evidence_ids": [],
        })
    decisions = []
    for c in setups:
        pkt = build_decision(c, snapshot)
        pkt["symbol"] = sym
        decisions.append(pkt)
        _record_candidate_packet(pkt)
    trades = [x for x in decisions if x.get("decision") in {"BUY", "SELL"}]
    if trades:
        best = max(trades, key=lambda x: _finite(x.get("decision_score"), 0))
    else:
        best = max(decisions, key=lambda x: _finite(x.get("decision_score"), 0))
        # Main's scanner owns the actual confidence threshold; brain still gives
        # an explicit WAIT/REJECT when below its own quality floor.
        if best.get("decision") not in {"BUY", "SELL"}:
            best["decision"] = "WAIT" if best.get("decision") == "WATCH" else "REJECT"
    best["symbol"] = sym
    return _safe_jsonable(best)


def _record_candidate_packet(packet: Mapping[str, Any]) -> str:
    row = copy.deepcopy(_safe_jsonable(dict(packet)))
    row["observation_type"] = "candidate"
    with _STATE_LOCK:
        _candidates.append(row)
        _brain_stats["candidates"] += 1
        baseline = _active_baseline()
        if row.get("confidence") is not None:
            _update_calibration_observation_pending(row)
    try:
        if _finite(row.get("confidence"), 0) < baseline:
            _queue_research("missed_opportunity_review", {"candidate_id": row.get("candidate_id"), "confidence": row.get("confidence")}, priority=5)
    except Exception:
        pass
    return str(row.get("candidate_id"))


def ingest_live_candidate(row: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
    """Main.py bridge: retain every analyzed candidate, not only traded ones."""
    if not isinstance(row, Mapping):
        return
    pkt = dict(row)
    # Avoid duplicate retention when full_analyze already recorded the packet.
    cid = pkt.get("candidate_id")
    with _STATE_LOCK:
        if cid and any(x.get("candidate_id") == cid for x in _candidates):
            return
    pkt.setdefault("observation_type", "candidate")
    pkt["ingested_at"] = _now()
    with _STATE_LOCK:
        _candidates.append(_safe_jsonable(pkt))
        _brain_stats["candidates"] += 1
    _update_frequency_state(pkt)


record_candidate_observation = ingest_live_candidate


# -----------------------------------------------------------------------------
# Outcome ingestion / autopsy / counterfactual
# -----------------------------------------------------------------------------

def _r_multiple(entry: float, sl: Optional[float], exit_price: float, direction: str) -> float:
    if sl is None:
        return 0.0
    risk = abs(entry - float(sl))
    if risk <= 1e-12:
        return 0.0
    move = (exit_price-entry) if direction.upper() == "BUY" else (entry-exit_price)
    return move / risk


def _entry_quality_from_outcome(row: Mapping[str, Any]) -> tuple[str, float, list[str]]:
    lf = row.get("learning_features") or {}
    thesis_quality = _finite(row.get("thesis", {}).get("thesis_quality"), _finite(row.get("effective_quality"), 50)/100)
    outcome_r = _finite(row.get("r_multiple"), 0.0)
    mfe_r = _finite(row.get("mfe_r"), outcome_r)
    if thesis_quality >= 0.62:
        quality = "GOOD"
        score = thesis_quality * 100
    elif thesis_quality <= 0.42:
        quality = "BAD"
        score = thesis_quality * 100
    else:
        quality = "MIXED"
        score = thesis_quality * 100
    reasons = []
    if outcome_r < 0 and mfe_r > 0.8:
        reasons.append("entry_had_favorable_excursion_but_failed_exit")
    if _finite(lf.get("liquidity_quality"), 0) < 0.3:
        reasons.append("weak_liquidity_confirmation")
    return quality, _clamp(score, 0, 100), reasons


def _management_quality(row: Mapping[str, Any]) -> tuple[str, float, list[str]]:
    outcome_r = _finite(row.get("r_multiple"), 0.0)
    mfe_r = _finite(row.get("mfe_r"), outcome_r)
    capture = (outcome_r / mfe_r) if mfe_r > 0.05 else (1.0 if outcome_r > 0 else 0.0)
    giveback = max(0.0, mfe_r - outcome_r) if mfe_r > 0 else 0.0
    if capture >= 0.70:
        quality = "GOOD"
        score = 85 + min(15, capture*15)
    elif capture >= 0.35:
        quality = "MIXED"
        score = 55 + capture*25
    else:
        quality = "POOR"
        score = capture*50
    reasons = [f"capture_ratio={capture:.2f}", f"giveback_r={giveback:.2f}"]
    return quality, _clamp(score, 0, 100), reasons


def _sl_classification(row: Mapping[str, Any], entry_quality: str) -> str:
    if str(row.get("result")).lower() not in {"sl", "stop", "loss", "stop_loss"}:
        return "N_A"
    thesis = _finite(row.get("thesis", {}).get("thesis_quality"), 0.5)
    mfe = _finite(row.get("mfe_r"), 0.0)
    if str(row.get("execution_error") or "").strip():
        return "EXECUTION_FAILURE"
    if thesis < 0.40 and mfe < 0.4:
        return "THESIS_FAILURE"
    if mfe >= 1.0 and _finite(row.get("mae_r"), 0.0) < 0:
        return "SL_GEOMETRY_FAILURE"
    if _finite(row.get("regime_change"), 0) > 0:
        return "REGIME_FAILURE"
    if _finite(row.get("mfe_r"), 0) >= 0.7:
        return "LIQUIDITY_TRAP"
    return "NORMAL_LOSS"


def _detect_lucky_rescue(row: Mapping[str, Any], entry_quality: str, management_quality: str) -> bool:
    return bool(
        entry_quality == "BAD" and
        management_quality == "GOOD" and
        _finite(row.get("r_multiple"), 0) > 0 and
        _finite(row.get("mfe_r"), 0) > 0
    )


def _autopsy(row: Mapping[str, Any]) -> dict[str, Any]:
    entry_quality, entry_score, entry_reasons = _entry_quality_from_outcome(row)
    management_quality, management_score, management_reasons = _management_quality(row)
    result = str(row.get("result") or row.get("classified_result") or "UNKNOWN").lower()
    lucky = _detect_lucky_rescue(row, entry_quality, management_quality)
    sl_class = _sl_classification(row, entry_quality)
    if entry_quality == "GOOD" and result in {"tp", "trail", "win", "profit"}:
        bucket = "GOOD_ENTRY_GOOD_OUTCOME"
    elif entry_quality == "GOOD":
        bucket = "GOOD_ENTRY_BAD_OUTCOME"
    elif entry_quality == "BAD" and result in {"tp", "trail", "win", "profit"}:
        bucket = "BAD_ENTRY_GOOD_OUTCOME"
    elif entry_quality == "BAD":
        bucket = "BAD_ENTRY_BAD_OUTCOME"
    else:
        bucket = "MIXED_ENTRY"
    if lucky:
        bucket = "RESCUED_TRADE"
    return {
        "entry_quality": entry_quality,
        "entry_score": round(entry_score, 2),
        "entry_reasons": entry_reasons,
        "management_quality": management_quality,
        "management_score": round(management_score, 2),
        "management_reasons": management_reasons,
        "outcome_bucket": bucket,
        "lucky_rescue": lucky,
        "sl_classification": sl_class,
        "thesis_quality": _finite(row.get("thesis", {}).get("thesis_quality"), 0.0),
    }


def _update_calibration(row: Mapping[str, Any]) -> None:
    confidence = _finite(row.get("confidence"), 0.0)
    bucket = min(100, max(10, int(round(confidence / 10.0) * 10)))
    key = str(bucket)
    r = _finite(row.get("r_multiple"), 0.0)
    with _STATE_LOCK:
        b = _calibration_state["bins"].setdefault(key, {"n": 0, "wins": 0, "r_sum": 0.0})
        b["n"] += 1
        b["wins"] += 1 if r > 0 else 0
        b["r_sum"] += r
        _calibration_state["last_update"] = _now()


def _update_calibration_observation_pending(row: Mapping[str, Any]) -> None:
    # Candidate observations are intentionally not treated as outcomes. This
    # function only keeps the presence of confidence bins for frequency analysis.
    _frequency_state.setdefault("confidence_distribution", Counter())
    bucket = int(round(_finite(row.get("confidence"), 0) / 10.0) * 10)
    _frequency_state["confidence_distribution"][str(max(0, min(100, bucket)))] += 1


def ingest_live_outcome(row: Mapping[str, Any], classified_result: Optional[str] = None, source: str = "unknown", *args: Any, **kwargs: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    x = dict(row)
    x["source"] = source
    x["observed_at"] = _now()
    if classified_result is not None:
        x["result"] = classified_result
    entry = _finite(x.get("entry"), _finite((x.get("signal") or {}).get("entry"), 0))
    exit_price = _finite(x.get("exit_price"), entry)
    direction = str(x.get("decision") or (x.get("signal") or {}).get("decision") or "BUY").upper()
    sl = _optional_float(x.get("initial_sl") or x.get("sl") or (x.get("signal") or {}).get("sl"))
    x["r_multiple"] = _r_multiple(entry, sl, exit_price, direction)
    x["setup_family"] = x.get("setup_family") or x.get("entry_label") or (x.get("signal") or {}).get("entry_label") or "unknown"
    x["direction"] = direction
    x["regime"] = x.get("regime") or x.get("market_context", {}).get("regime", {}).get("primary")
    x["autopsy"] = _autopsy(x)
    x["entry_quality"] = x["autopsy"]["entry_quality"]
    x["management_quality"] = x["autopsy"]["management_quality"]
    with _STATE_LOCK:
        _outcomes.append(_safe_jsonable(x))
        _brain_stats["outcomes"] += 1
        _brain_stats["last_learning_at"] = _now()
    _update_calibration(x)
    _record_trade_evidence(x)
    _queue_autopsy_research(x)
    _detect_missed_for_recent_candidate(x)
    return _safe_jsonable(x["autopsy"])


record_trade_outcome = ingest_live_outcome


def _record_trade_evidence(row: Mapping[str, Any]) -> None:
    aut = row.get("autopsy") or {}
    eid = _id("EVID")
    evidence = Evidence(
        evidence_id=eid,
        source=str(row.get("source") or "trade"),
        timestamp=_now(),
        sample_size=1,
        freshness=1.0,
        data_quality=_finite(row.get("data_quality"), 1.0),
        regime=str(row.get("regime") or "UNKNOWN"),
        independence=1.0,
        confidence=_finite(row.get("confidence"), 0.0),
        statement=f"Trade outcome {row.get('result')} with entry={aut.get('entry_quality')} management={aut.get('management_quality')}",
        payload={"trade_uid": row.get("trade_uid"), "r_multiple": row.get("r_multiple"), "autopsy": aut},
    )
    with _STATE_LOCK:
        _evidence.append(evidence.to_dict())
    return None


def _queue_autopsy_research(row: Mapping[str, Any]) -> None:
    aut = row.get("autopsy") or {}
    priority = 10 if aut.get("entry_quality") == "BAD" else 6
    _queue_research("trade_autopsy", {"trade_uid": row.get("trade_uid"), "r_multiple": row.get("r_multiple"), "autopsy": aut}, priority=priority)


def _detect_missed_for_recent_candidate(outcome: Mapping[str, Any]) -> None:
    sym = outcome.get("symbol")
    entry = _finite(outcome.get("entry"), 0)
    direction = str(outcome.get("direction") or "")
    with _STATE_LOCK:
        candidates = [x for x in list(_candidates)[-250:] if x.get("symbol") == sym and x.get("direction") == direction]
    for c in candidates[-20:]:
        if c.get("decision") not in {"TRADE", direction} and _finite(c.get("confidence"), 100) < _active_baseline():
            _brain_stats["missed_opportunities"] += 1
            _journal("research", {"message": f"MISSED_OPPORTUNITY {sym} {direction}", "candidate_id": c.get("candidate_id"), "source_outcome": outcome.get("trade_uid")})
            break


# -----------------------------------------------------------------------------
# Missed opportunity / shadow
# -----------------------------------------------------------------------------

def record_shadow_outcome(row: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
    if not isinstance(row, Mapping):
        return
    x = dict(row)
    x["source"] = "shadow"
    x["observed_at"] = _now()
    with _STATE_LOCK:
        _outcomes.append(_safe_jsonable(x))
        _brain_stats["shadow_outcomes"] += 1
    _record_trade_evidence(x)


# -----------------------------------------------------------------------------
# Frequency / drift / calibration
# -----------------------------------------------------------------------------

def _update_frequency_state(row: Mapping[str, Any]) -> None:
    with _STATE_LOCK:
        family = str(row.get("setup_family") or "unknown")
        fs = _frequency_state.setdefault("families", {})
        info = fs.setdefault(family, {"n": 0, "trades": 0, "near": 0})
        info["n"] += 1
        conf = _finite(row.get("confidence"), 0)
        if row.get("decision") in {"BUY", "SELL", "TRADE"}:
            info["trades"] += 1
        if abs(conf - _active_baseline()) <= 8:
            info["near"] += 1


def get_active_confidence_threshold() -> float:
    """Brain-owned confidence policy used by main's display/gate integration."""
    # During early learning, use the configured brain baseline. The brain may
    # nudge it only when frequency evidence is sufficient; it never performs
    # abrupt threshold shifts.
    return _active_baseline()


def suggest_confidence_threshold() -> float:
    return get_active_confidence_threshold()


def _update_drift() -> None:
    with _STATE_LOCK:
        recent = list(_outcomes)[-60:]
    if len(recent) < 10:
        return
    rs = [_finite(x.get("r_multiple"), 0) for x in recent]
    old = rs[:max(1, len(rs)//2)]
    new = rs[len(rs)//2:]
    old_e = _mean(old); new_e = _mean(new)
    with _STATE_LOCK:
        _drift_state.update({"recent_expectancy_r": new_e, "prior_expectancy_r": old_e, "expectancy_delta_r": new_e-old_e, "sample": len(recent), "updated_at": _now()})


# -----------------------------------------------------------------------------
# Pattern discovery / hypothesis / experiments
# -----------------------------------------------------------------------------

def discover_patterns(min_sample: int = 8, top_k: int = 20) -> list[dict[str, Any]]:
    with _STATE_LOCK:
        rows = list(_outcomes)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (str(r.get("setup_family") or "unknown"), str(r.get("regime") or "UNKNOWN"), str(r.get("direction") or ""))
        groups[key].append(r)
    discovered = []
    for (family, regime, direction), group in groups.items():
        if len(group) < min_sample:
            continue
        rs = [_finite(x.get("r_multiple"), 0) for x in group]
        expectancy = _mean(rs)
        win_rate = sum(1 for x in rs if x > 0) / len(rs)
        pattern = {
            "pattern_id": _id("PAT"),
            "timestamp": _now(),
            "setup_family": family,
            "regime": regime,
            "direction": direction,
            "sample_size": len(group),
            "expected_r": round(expectancy, 4),
            "win_rate": round(win_rate, 4),
            "stability": round(_clamp(1 - abs(_mean(rs[:len(rs)//2]) - _mean(rs[len(rs)//2:])), 0, 1), 4),
            "statement": f"{family} {direction} in {regime} has expectancy {expectancy:.2f}R over {len(group)} observations",
        }
        discovered.append(pattern)
    discovered.sort(key=lambda x: (x["expected_r"] * math.log1p(x["sample_size"])), reverse=True)
    with _STATE_LOCK:
        for p in discovered[:top_k]:
            _patterns.append(p)
        _brain_stats["patterns"] += min(len(discovered), top_k)
    for p in discovered[:top_k]:
        _journal("research", {"message": f"Pattern discovered: {p['statement']}", "pattern_id": p["pattern_id"]})
    return _safe_jsonable(discovered[:top_k])


def _hypothesis_fingerprint(claim: str, affected_regime: str = "") -> str:
    raw = f"{claim.strip().lower()}|{affected_regime.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def create_hypothesis(claim: str, supporting: Optional[list[str]] = None, contradicting: Optional[list[str]] = None, sample_size: int = 0, confidence: float = 0.0, regimes: Optional[list[str]] = None, expected_change: str = "") -> dict[str, Any]:
    fp = _hypothesis_fingerprint(claim, ",".join(regimes or []))
    with _STATE_LOCK:
        for h in _hypotheses.values():
            if h.get("fingerprint") == fp and h.get("status") not in {"REJECTED", "CONTRADICTED"}:
                return dict(h)
    h = Hypothesis(
        hypothesis_id=_id("H").upper(), claim=claim,
        supporting_evidence=list(supporting or []), contradicting_evidence=list(contradicting or []),
        sample_size=int(sample_size), confidence=float(_clamp(confidence, 0, 1)),
        regimes=list(regimes or []), expected_change=str(expected_change), status="NEW",
        created_at=_now(), fingerprint=fp,
    )
    with _STATE_LOCK:
        _hypotheses[h.hypothesis_id] = h.to_dict()
        _brain_stats["hypotheses_created"] += 1
    _journal("research", {"message": f"{h.hypothesis_id} created: {claim}", "hypothesis_id": h.hypothesis_id})
    return h.to_dict()


def _generate_hypotheses_from_patterns(patterns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in patterns:
        expected = _finite(p.get("expected_r"), 0)
        sample = int(p.get("sample_size") or 0)
        if sample < 10:
            continue
        if expected < -0.15:
            claim = f"{p.get('setup_family')} performs poorly in {p.get('regime')} and should receive a stricter acceptance condition."
            out.append(create_hypothesis(claim, sample_size=sample, confidence=_clamp(abs(expected)/1.5, 0, 1), regimes=[str(p.get("regime"))], expected_change="tighten acceptance in affected regime"))
        elif expected > 0.35:
            claim = f"{p.get('setup_family')} performs strongly in {p.get('regime')} and the current policy may be under-accepting it."
            out.append(create_hypothesis(claim, sample_size=sample, confidence=_clamp(expected/1.5, 0, 1), regimes=[str(p.get("regime"))], expected_change="protect or slightly relax acceptance for this context"))
    return out


def _strategy_variant_from_hypothesis(h: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    claim = str(h.get("claim") or "")
    base = copy.deepcopy(_ACTIVE_STRATEGY)
    if "stricter acceptance" in claim.lower():
        new_version = _next_strategy_version(base.version)
        base.entry_rules["min_structure"] = _clamp(_finite(base.entry_rules.get("min_structure"), 0.52) + 0.04, 0, 1)
        base.entry_rules["min_alignment"] = _clamp(_finite(base.entry_rules.get("min_alignment"), 0.48) + 0.03, 0, 1)
        base.change_reason = claim
        base.parent_version = base.version
        base.version = new_version
        base.hypothesis_id = h.get("hypothesis_id")
        return base.to_dict()
    if "under-accepting" in claim.lower():
        new_version = _next_strategy_version(base.version)
        base.entry_rules["min_structure"] = _clamp(_finite(base.entry_rules.get("min_structure"), 0.52) - 0.025, 0, 1)
        base.entry_rules["min_alignment"] = _clamp(_finite(base.entry_rules.get("min_alignment"), 0.48) - 0.02, 0, 1)
        base.change_reason = claim
        base.parent_version = base.version
        base.version = new_version
        base.hypothesis_id = h.get("hypothesis_id")
        return base.to_dict()
    return None


def _next_strategy_version(current: str) -> str:
    text = str(current or "S01").upper().replace("S", "")
    try:
        return f"S{int(text.split(".")[0]) + 1:02d}"
    except Exception:
        return f"S{len(_strategy_history)+2:02d}"


def _eligible_rows_for_profile(rows: Sequence[Mapping[str, Any]], profile_dict: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rules = dict(profile_dict.get("entry_rules") or {})
    min_structure = _finite(rules.get("min_structure"), 0.0)
    min_alignment = _finite(rules.get("min_alignment"), 0.0)
    min_rr = _finite(rules.get("min_rr"), MIN_RR)
    eligible = []
    for r in rows:
        lf = r.get("learning_features") or {}
        thesis = r.get("thesis") or {}
        scores = thesis.get("scores") or {}
        structure = _finite(lf.get("structure_quality"), _finite(scores.get("structure_quality"), 1.0))
        alignment = _finite(lf.get("alignment_quality"), _finite(scores.get("alignment_quality"), 1.0))
        rr = _finite(r.get("rr"), _finite(r.get("risk_reward"), MIN_RR))
        if structure + 1e-9 < min_structure:
            continue
        if alignment + 1e-9 < min_alignment:
            continue
        if rr + 1e-9 < min_rr:
            continue
        eligible.append(r)
    return eligible


def _evaluate_rows_for_profile(rows: Sequence[Mapping[str, Any]], profile_dict: Mapping[str, Any]) -> dict[str, Any]:
    rows = _eligible_rows_for_profile(rows, profile_dict)
    if not rows:
        return {"sample_size": 0, "expected_r": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown_r": 0.0, "consistency": 0.0}
    rs = [_finite(r.get("r_multiple"), 0) for r in rows]
    gains = sum(x for x in rs if x > 0)
    losses = abs(sum(x for x in rs if x < 0))
    eq = 0.0; peak = 0.0; dd = 0.0
    for r in rs:
        eq += r; peak = max(peak, eq); dd = max(dd, peak-eq)
    half = max(1, len(rs)//2)
    e1, e2 = _mean(rs[:half]), _mean(rs[-half:])
    return {
        "sample_size": len(rs),
        "expected_r": _mean(rs),
        "win_rate": sum(1 for x in rs if x > 0)/len(rs),
        "profit_factor": (gains/losses) if losses > 1e-12 else (999.0 if gains > 0 else 0.0),
        "max_drawdown_r": dd,
        "consistency": _clamp(1.0 - abs(e1-e2), 0, 1),
        "mfe_r": _mean([_finite(r.get("mfe_r"), 0) for r in rows]),
        "mae_r": _mean([_finite(r.get("mae_r"), 0) for r in rows]),
        "profile_version": profile_dict.get("version"),
    }


def run_experiment(hypothesis_id: str, challenger_profile: Mapping[str, Any]) -> dict[str, Any]:
    with _STATE_LOCK:
        base_rows = list(_outcomes)
        h = dict(_hypotheses.get(hypothesis_id) or {})
    if not h:
        raise ValueError(f"unknown hypothesis: {hypothesis_id}")
    # This is a policy-level experiment over the observed candidate set. The
    # strategy's core signal model is deterministic, so the variant is scored
    # without mutating any live trade.
    base_eval = _evaluate_rows_for_profile(base_rows, _ACTIVE_STRATEGY.to_dict())
    challenger_eval = _evaluate_rows_for_profile(base_rows, challenger_profile)
    # Recent/OOS split: last 35% is treated as out-of-sample for promotion.
    split = max(1, int(len(base_rows) * 0.65))
    train = base_rows[:split]
    oos = base_rows[split:]
    train_eval = _evaluate_rows_for_profile(train, challenger_profile)
    oos_eval = _evaluate_rows_for_profile(oos, challenger_profile)
    recent = base_rows[-min(30, len(base_rows)):]
    recent_eval = _evaluate_rows_for_profile(recent, challenger_profile)
    result = {
        "base": base_eval,
        "challenger": challenger_eval,
        "train": train_eval,
        "oos": oos_eval,
        "recent": recent_eval,
        "improvement_r": challenger_eval["expected_r"] - base_eval["expected_r"],
        "oos_improvement_r": oos_eval["expected_r"] - _evaluate_rows_for_profile(oos, _ACTIVE_STRATEGY.to_dict())["expected_r"],
        "frequency_delta": challenger_eval["sample_size"] - base_eval["sample_size"],
    }
    experiment = Experiment(
        experiment_id=_id("EXP"), hypothesis_id=hypothesis_id,
        base_strategy=_ACTIVE_STRATEGY.version,
        challenger_versions=[str(challenger_profile.get("version"))],
        design={"type": "replay_policy_comparison", "split": 0.65, "recent_window": min(30, len(base_rows))},
        result=result, status="COMPLETED", created_at=_now(), completed_at=_now(),
    )
    with _STATE_LOCK:
        _experiments[experiment.experiment_id] = experiment.to_dict()
        _brain_stats["experiments_completed"] += 1
        _challengers[str(challenger_profile.get("version"))] = {
            "version": challenger_profile.get("version"), "profile": _safe_jsonable(dict(challenger_profile)),
            "experiment_id": experiment.experiment_id, "status": "VALIDATED_PENDING", "created_at": _now(),
        }
        _hypotheses[hypothesis_id]["status"] = "TESTING"
    _journal("experiment", {"message": f"{experiment.experiment_id} completed: {hypothesis_id}", "experiment_id": experiment.experiment_id, "result": result})
    return experiment.to_dict()


def _validate_challenger(experiment: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    result = experiment.get("result") or {}
    oos = result.get("oos") or {}
    recent = result.get("recent") or {}
    base = result.get("base") or {}
    improvement = _finite(result.get("improvement_r"), 0)
    oos_imp = _finite(result.get("oos_improvement_r"), 0)
    robust = (
        improvement > 0.05 and
        oos_imp > 0.02 and
        _finite(oos.get("consistency"), 0) >= 0.35 and
        _finite(recent.get("expected_r"), -99) >= _finite(base.get("expected_r"), -99) - 0.05
    )
    return bool(robust), {"improvement": improvement, "oos_improvement": oos_imp, "robust": robust}


def promote_challenger(experiment_id: str) -> bool:
    with _STATE_LOCK:
        ex = _experiments.get(experiment_id)
        if not ex:
            return False
        challenger_versions = ex.get("challenger_versions") or []
        if not challenger_versions:
            return False
        version = str(challenger_versions[0])
        c = _challengers.get(version)
    ok, metrics = _validate_challenger(ex)
    if not ok or not c:
        with _STATE_LOCK:
            _brain_stats["rejections"] += 1
            if c:
                c["status"] = "REJECTED"
            h = _hypotheses.get(str(ex.get("hypothesis_id")))
            if h:
                h["status"] = "REJECTED"
        _journal("promotion", {"message": f"{version} rejected", "experiment_id": experiment_id, "metrics": metrics})
        return False
    profile_dict = c["profile"]
    profile = StrategyProfile(**profile_dict)
    with _STATE_LOCK:
        global _ACTIVE_STRATEGY, _CHAMPION
        old = _ACTIVE_STRATEGY.version
        _ACTIVE_STRATEGY = copy.deepcopy(profile)
        _CHAMPION = copy.deepcopy(profile)
        c["status"] = "CHAMPION"
        h = _hypotheses.get(str(ex.get("hypothesis_id")))
        if h:
            h["status"] = "PROMOTED"
        _strategy_history.append({"old_version": old, "new_version": profile.version, "change_type": "promotion", "change_reason": profile.change_reason, "hypothesis_id": profile.hypothesis_id, "experiment_id": experiment_id, "evidence": metrics, "timestamp": _now()})
        _brain_stats["promotions"] += 1
        _brain_stats["last_strategy_change_at"] = _now()
    _journal("promotion", {"message": f"{profile.version} promoted over {old}", "experiment_id": experiment_id, "metrics": metrics}, telegram=True)
    return True


# -----------------------------------------------------------------------------
# Research queue / Ollama adapter
# -----------------------------------------------------------------------------

def _queue_research(job_type: str, payload: Mapping[str, Any], priority: int = 1) -> str:
    job = {"job_id": _id("JOB"), "type": str(job_type), "priority": int(priority), "created_at": _now(), "payload": _safe_jsonable(dict(payload))}
    with _STATE_LOCK:
        _research_queue.append(job)
        rows = sorted(_research_queue, key=lambda x: (-int(x.get("priority", 0)), x.get("created_at", 0)))
        _research_queue.clear()
        _research_queue.extend(rows[:MAX_RESEARCH_QUEUE])
    return job["job_id"]


def queue_ollama_research(prompt: str, context: Optional[Mapping[str, Any]] = None, priority: int = 1) -> str:
    job = {"job_id": _id("OLLAMA"), "priority": int(priority), "created_at": _now(), "prompt": str(prompt), "context": _safe_jsonable(dict(context or {})), "status": "QUEUED"}
    with _STATE_LOCK:
        _ollama_queue.append(job)
    return job["job_id"]


def _ollama_endpoint() -> Optional[str]:
    return os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL")


def process_one_ollama_job() -> Optional[dict[str, Any]]:
    if requests is None:
        return None
    endpoint = _ollama_endpoint()
    if not endpoint:
        return None
    with _STATE_LOCK:
        if not _ollama_queue:
            return None
        job = max(_ollama_queue, key=lambda x: (int(x.get("priority", 0)), -float(x.get("created_at", 0))))
        try:
            _ollama_queue.remove(job)
        except ValueError:
            return None
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    prompt = json.dumps({"prompt": job["prompt"], "context": job.get("context", {}), "instruction": "Return JSON with hypothesis, supporting_evidence, contradicting_evidence, unknowns, suggested_tests, confidence."}, ensure_ascii=False)
    try:
        r = requests.post(endpoint.rstrip("/") + "/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=45)
        r.raise_for_status()
        data = r.json()
        raw = data.get("response") if isinstance(data, dict) else None
        parsed = None
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"raw": str(raw)}
        result = {"job_id": job["job_id"], "status": "COMPLETED", "output": _safe_jsonable(parsed or data)}
        _journal("research", {"message": "Ollama research completed", "job_id": job["job_id"]})
        return result
    except Exception as exc:
        _journal("research", {"message": f"Ollama research failed: {exc}", "job_id": job["job_id"]})
        return {"job_id": job["job_id"], "status": "FAILED", "error": str(exc)[:300]}


# -----------------------------------------------------------------------------
# /full research cycle
# -----------------------------------------------------------------------------

def research_cycle(*, include_ollama: bool = True, include_historical: bool = True, promote: bool = True) -> dict[str, Any]:
    """Run a bounded research cycle inside the caller's worker.

    It intentionally does not create threads. main.py's global heavy-worker
    governor remains the sole concurrency authority.
    """
    started = _now()
    patterns = discover_patterns(min_sample=8, top_k=12)
    hypotheses = _generate_hypotheses_from_patterns(patterns)
    experiments = []
    for h in hypotheses[:6]:
        variant = _strategy_variant_from_hypothesis(h)
        if not variant:
            continue
        try:
            experiments.append(run_experiment(h["hypothesis_id"], variant))
        except Exception as exc:
            _journal("experiment", {"message": f"Experiment failed: {exc}", "hypothesis_id": h.get("hypothesis_id")})
    promotions = []
    if promote:
        for e in experiments:
            try:
                if promote_challenger(e["experiment_id"]):
                    promotions.append(e["experiment_id"])
            except Exception as exc:
                _journal("promotion", {"message": f"Promotion evaluation failed: {exc}", "experiment_id": e.get("experiment_id")})
    ollama_result = None
    if include_ollama:
        with _STATE_LOCK:
            queued = len(_ollama_queue)
        if queued:
            ollama_result = process_one_ollama_job()
    _update_drift()
    duration = _now() - started
    summary = {
        "started_at": started,
        "completed_at": _now(),
        "duration_sec": round(duration, 3),
        "patterns": len(patterns),
        "hypotheses": len(hypotheses),
        "experiments": len(experiments),
        "promotions": promotions,
        "ollama": ollama_result,
        "active_strategy": _ACTIVE_STRATEGY.version,
    }
    _brain_log(logging.INFO, f"[BRAIN RESEARCH] cycle completed patterns={len(patterns)} hypotheses={len(hypotheses)} experiments={len(experiments)} promotions={len(promotions)}")
    return _safe_jsonable(summary)


def full_command(action: str, *args: Any, **kwargs: Any) -> str:
    """Main.py bridge for /full on/off/reset/status.

    `/full` remains brain-owned mode control. A status request is cheap. `on`
    enables continuous adaptive mode, while `off` disables automatic research.
    A caller can explicitly trigger the full research cycle with `action='cycle'`.
    """
    global _FULL_MODE
    action = str(action or "status").lower()
    with _STATE_LOCK:
        if action == "on":
            _FULL_MODE = True
        elif action == "off":
            _FULL_MODE = False
        elif action == "reset":
            _reset_learning_state(keep_strategy=True)
            _FULL_MODE = False
        elif action in {"cycle", "run", "research"}:
            _FULL_MODE = True
            summary = research_cycle(include_ollama=True, include_historical=True, promote=True)
            return _format_status(summary)
    return _format_status()


_FULL_MODE = False


def _format_status(extra: Optional[Mapping[str, Any]] = None) -> str:
    with _STATE_LOCK:
        challenger = list(_challengers.values())
        active = _ACTIVE_STRATEGY.version
        champion = _CHAMPION.version
        stats = dict(_brain_stats)
        freq = _current_frequency_state()
        hypotheses = {k: v.get("status") for k, v in _hypotheses.items()}
        experiments = {k: v.get("status") for k, v in _experiments.items()}
        beliefs = len(_beliefs)
        queue_len = len(_research_queue)
    lines = [
        "🧠 <b>ADAPTIVE BRAIN</b>",
        "",
        f"Strategy: <b>{active}</b>",
        f"Champion: <b>{champion}</b>",
        f"Full mode: <b>{'ON' if _FULL_MODE else 'OFF'}</b>",
        f"Observations: <b>{stats['observations']}</b>",
        f"Candidates: <b>{stats['candidates']}</b>",
        f"Outcomes: <b>{stats['outcomes']}</b>",
        f"Patterns: <b>{stats['patterns']}</b>",
        f"Hypotheses: <b>{len(hypotheses)}</b>",
        f"Experiments: <b>{len(experiments)}</b>",
        f"Challengers: <b>{len(challenger)}</b>",
        f"Beliefs: <b>{beliefs}</b>",
        f"Research queue: <b>{queue_len}</b>",
        f"Near-threshold supply: <b>{freq['near_threshold_supply']}</b>",
        f"Acceptance rate: <b>{freq['acceptance_rate']:.1%}</b>",
        f"Missed opportunities: <b>{stats['missed_opportunities']}</b>",
        f"Promotions: <b>{stats['promotions']}</b>",
        f"Rejected challengers: <b>{stats['rejections']}</b>",
        f"Last strategy change: <b>{stats['last_strategy_change_at'] or 'never'}</b>",
    ]
    if extra:
        lines += ["", "<b>Last cycle</b>", json.dumps(_safe_jsonable(dict(extra)), ensure_ascii=False)[:1600]]
    return "\n".join(lines)


def get_full_cognitive_status() -> str:
    return _format_status()


def get_cognitive_status() -> str:
    return _format_status()


# -----------------------------------------------------------------------------
# Management learning / position management
# -----------------------------------------------------------------------------

def _position_context(state: Mapping[str, Any], df_m15: pd.DataFrame) -> dict[str, Any]:
    f = _clean_snapshot_frame(df_m15)
    if f.empty:
        return {"price": _finite(state.get("current_price"), _finite(state.get("entry"), 0)), "atr": 0.0, "rsi": 50.0, "relative_volume": 1.0}
    return {
        "price": float(f["close"].iloc[-1]),
        "atr": _finite(f["atr"].iloc[-1], 0),
        "rsi": _finite(f["rsi"].iloc[-1], 50),
        "relative_volume": _finite(f["relative_volume"].iloc[-1], 1),
        "trend": _structure_descriptor(f).get("trend"),
        "liquidity": _liquidity_features(f),
    }


def manage_position(state: Mapping[str, Any], df_m15: pd.DataFrame, df_h1: Optional[pd.DataFrame] = None, df_d1: Optional[pd.DataFrame] = None, symbol: Optional[str] = None, **kwargs: Any) -> dict[str, Any]:
    """Return management intent; never executes or mutates exchange orders."""
    pos = dict(state or {})
    sig = dict(pos.get("signal") or pos)
    entry = _finite(pos.get("entry"), _finite(sig.get("entry"), 0))
    current_sl = _finite(pos.get("current_sl"), _finite(sig.get("sl"), entry))
    initial_sl = _finite(pos.get("initial_sl"), _finite(sig.get("initial_sl"), current_sl))
    tp = _optional_float(sig.get("tp"))
    direction = str(sig.get("decision") or "BUY").upper()
    ctx = _position_context(pos, df_m15)
    price = _finite(ctx.get("price"), entry)
    risk = abs(entry - initial_sl)
    if risk <= 1e-12:
        return {"state": "NO_MANAGEMENT", "reason": ["invalid_risk_geometry"]}
    profit_r = ((price-entry) if direction == "BUY" else (entry-price)) / risk
    max_protected_r = _finite(pos.get("max_protected_r"), -999)
    mgmt = _ACTIVE_STRATEGY.management_rules
    trail_start = _finite(mgmt.get("trail_start_r"), 1.0)
    lock_r = _finite(mgmt.get("trail_lock_r"), 0.35)
    buffer_atr = _finite(mgmt.get("trail_buffer_atr"), 0.35)
    giveback_tol = _finite(mgmt.get("giveback_tolerance_r"), 0.60)
    reasons = []
    update = {
        "state": "HOLD",
        "trail_source": "adaptive",
        "reason": reasons,
        "relative_volume": _finite(ctx.get("relative_volume"), 1.0),
        "weakness_score": 0,
        "profit_r": round(profit_r, 3),
        "decision_created_at": _now(),
        "decision_expires_at": _now() + DEFAULT_MANAGEMENT_TTL_SEC,
    }
    if profit_r < trail_start:
        return _safe_jsonable(update)
    max_protected_r = max(max_protected_r, profit_r)
    structure = _structure_descriptor(_clean_snapshot_frame(df_m15)) if not _normalize_ohlcv(df_m15).empty else {}
    weakness = 0
    if direction == "BUY":
        if _finite(ctx.get("rsi"), 50) < 48: weakness += 25
        if structure.get("trend") == "TREND_DOWN": weakness += 40
        if _finite(ctx.get("relative_volume"), 1) < 0.8: weakness += 15
        anchor = price - max(buffer_atr * _finite(ctx.get("atr"), risk), risk * lock_r)
        candidate_sl = max(current_sl, anchor)
        if weakness >= 65 and profit_r >= trail_start + 0.5:
            candidate_sl = max(candidate_sl, price - risk * 0.55)
            reasons.append("momentum_weakness")
        if profit_r >= _finite(mgmt.get("extension_hold_r"), 2.0) and weakness < 50:
            update["state"] = "RUN"
            reasons.append("favorable_extension_preserved")
        if candidate_sl > current_sl + risk * 0.03:
            update.update({"sl": candidate_sl, "tp": tp, "state": "TRAIL"})
            reasons.append("profit_protection_improved")
    else:
        if _finite(ctx.get("rsi"), 50) > 52: weakness += 25
        if structure.get("trend") == "TREND_UP": weakness += 40
        if _finite(ctx.get("relative_volume"), 1) < 0.8: weakness += 15
        anchor = price + max(buffer_atr * _finite(ctx.get("atr"), risk), risk * lock_r)
        candidate_sl = min(current_sl, anchor)
        if weakness >= 65 and profit_r >= trail_start + 0.5:
            candidate_sl = min(candidate_sl, price + risk * 0.55)
            reasons.append("momentum_weakness")
        if profit_r >= _finite(mgmt.get("extension_hold_r"), 2.0) and weakness < 50:
            update["state"] = "RUN"
            reasons.append("favorable_extension_preserved")
        if candidate_sl < current_sl - risk * 0.03:
            update.update({"sl": candidate_sl, "tp": tp, "state": "TRAIL"})
            reasons.append("profit_protection_improved")
    update["weakness_score"] = weakness
    # Do not issue market close solely from a management score in this baseline;
    # preserve main.py's hard close / protection authority.
    return _safe_jsonable(update)


# -----------------------------------------------------------------------------
# Strategy descriptor / model info
# -----------------------------------------------------------------------------

def get_active_strategy() -> dict[str, Any]:
    with _STATE_LOCK:
        return _safe_jsonable(_ACTIVE_STRATEGY.to_dict())


def get_learning_model_info() -> dict[str, Any]:
    with _STATE_LOCK:
        champion = _CHAMPION.to_dict()
        return {
            "champion": champion,
            "active_strategy": _ACTIVE_STRATEGY.to_dict(),
            "experience_samples": len(_outcomes),
            "observations": _brain_stats["observations"],
            "policy_revisions": len(_strategy_history),
        }


def get_experience_count() -> int:
    with _STATE_LOCK:
        return len(_outcomes)


def get_strategy_descriptor() -> dict[str, Any]:
    with _STATE_LOCK:
        return {
            "engine": ENGINE_NAME,
            "brain_interface_version": BRAIN_INTERFACE_VERSION,
            "brain_schema_version": BRAIN_SCHEMA_VERSION,
            "active_strategy": _ACTIVE_STRATEGY.to_dict(),
            "champion": _CHAMPION.to_dict(),
            "capabilities": [
                "perception", "context", "setup_detection", "thesis", "decision",
                "confidence_calibration", "memory", "autopsy", "counterfactual",
                "pattern_discovery", "hypothesis", "experiment", "shadow",
                "challenger", "validation", "promotion", "management_learning",
                "frequency", "drift", "ollama_research", "brain_state",
            ],
        }


def get_learning_schema() -> dict[str, Any]:
    return {
        "observation": ["observation_id", "timestamp", "symbol", "features"],
        "candidate": ["candidate_id", "setup_family", "direction", "confidence", "strategy_version"],
        "outcome": ["trade_uid", "result", "r_multiple", "autopsy", "strategy_version"],
        "hypothesis": ["hypothesis_id", "claim", "supporting_evidence", "contradicting_evidence", "status"],
        "experiment": ["experiment_id", "hypothesis_id", "result", "status"],
    }


def get_brain_state() -> dict[str, Any]:
    with _STATE_LOCK:
        return {
            "brain_interface_version": BRAIN_INTERFACE_VERSION,
            "brain_schema_version": BRAIN_SCHEMA_VERSION,
            "engine": ENGINE_NAME,
            "active_strategy": _ACTIVE_STRATEGY.to_dict(),
            "champion": _CHAMPION.to_dict(),
            "candidates": list(_candidates),
            "outcomes": list(_outcomes),
            "evidence": list(_evidence),
            "patterns": list(_patterns),
            "hypotheses": dict(_hypotheses),
            "experiments": dict(_experiments),
            "challengers": dict(_challengers),
            "beliefs": dict(_beliefs),
            "strategy_history": list(_strategy_history),
            "research_journal": list(_research_journal),
            "research_queue": list(_research_queue),
            "ollama_queue": list(_ollama_queue),
            "frequency_state": _safe_jsonable(_frequency_state),
            "drift_state": _safe_jsonable(_drift_state),
            "calibration_state": _safe_jsonable(_calibration_state),
            "brain_stats": dict(_brain_stats),
            "full_mode": bool(_FULL_MODE),
            "exported_at": _now(),
        }


def export_checkpoint_state() -> dict[str, Any]:
    return get_brain_state()


def _restore_deque(target: deque, values: Any, cap: int) -> None:
    target.clear()
    if isinstance(values, list):
        target.extend(values[-cap:])


def import_checkpoint_state(state: Mapping[str, Any]) -> bool:
    if not isinstance(state, Mapping):
        raise ValueError("brain checkpoint must be an object")
    version = str(state.get("brain_interface_version") or "")
    if version and version != BRAIN_INTERFACE_VERSION:
        raise ValueError(f"brain interface mismatch: {version} != {BRAIN_INTERFACE_VERSION}")
    schema = str(state.get("brain_schema_version") or BRAIN_SCHEMA_VERSION)
    if schema != BRAIN_SCHEMA_VERSION:
        raise ValueError(f"unsupported brain schema: {schema}")
    global _ACTIVE_STRATEGY, _CHAMPION, _FULL_MODE
    with _STATE_LOCK:
        active = state.get("active_strategy")
        champ = state.get("champion")
        if active:
            _ACTIVE_STRATEGY = StrategyProfile(**dict(active))
        if champ:
            _CHAMPION = StrategyProfile(**dict(champ))
        _restore_deque(_candidates, state.get("candidates"), MAX_CANDIDATE_MEMORY)
        _restore_deque(_outcomes, state.get("outcomes"), MAX_OUTCOME_MEMORY)
        _restore_deque(_evidence, state.get("evidence"), MAX_EVIDENCE_MEMORY)
        _restore_deque(_patterns, state.get("patterns"), MAX_PATTERN_MEMORY)
        _restore_deque(_strategy_history, state.get("strategy_history"), _strategy_history.maxlen or 100)
        _restore_deque(_research_journal, state.get("research_journal"), _research_journal.maxlen or 1500)
        _restore_deque(_research_queue, state.get("research_queue"), MAX_RESEARCH_QUEUE)
        _restore_deque(_ollama_queue, state.get("ollama_queue"), MAX_OLLAMA_QUEUE)
        _hypotheses.clear(); _hypotheses.update({str(k): dict(v) for k, v in (state.get("hypotheses") or {}).items()})
        _experiments.clear(); _experiments.update({str(k): dict(v) for k, v in (state.get("experiments") or {}).items()})
        _challengers.clear(); _challengers.update({str(k): dict(v) for k, v in (state.get("challengers") or {}).items()})
        _beliefs.clear(); _beliefs.update({str(k): dict(v) for k, v in (state.get("beliefs") or {}).items()})
        _frequency_state.clear(); _frequency_state.update(dict(state.get("frequency_state") or {}))
        _drift_state.clear(); _drift_state.update(dict(state.get("drift_state") or {}))
        _calibration_state.clear(); _calibration_state.update(dict(state.get("calibration_state") or _calibration_state))
        _brain_stats.update(dict(state.get("brain_stats") or {}))
        _FULL_MODE = bool(state.get("full_mode", _FULL_MODE))
    _brain_log(logging.INFO, f"[BRAIN] checkpoint restored active={_ACTIVE_STRATEGY.version} outcomes={len(_outcomes)}")
    return True


def load_brain_state(state: Mapping[str, Any]) -> bool:
    return import_checkpoint_state(state)


def apply_brain_state(state: Mapping[str, Any]) -> bool:
    return import_checkpoint_state(state)


def set_learning_model(model: Any) -> None:
    # Compatibility bridge: the brain's learning state is statistical/policy
    # based. An external ML object may be retained only as a serializable label.
    with _STATE_LOCK:
        _frequency_state["external_model"] = _safe_jsonable(model)


# -----------------------------------------------------------------------------
# Strategy validation
# -----------------------------------------------------------------------------

def validate_strategy(profile: Any = None) -> dict[str, Any]:
    target = profile or _ACTIVE_STRATEGY.to_dict()
    if isinstance(target, StrategyProfile):
        target = target.to_dict()
    required = ["version", "entry_rules", "preferences", "regime_rules", "management_rules", "confidence_policy"]
    missing = [k for k in required if k not in target]
    errors = []
    if missing:
        errors.append(f"missing fields: {missing}")
    if _finite((target.get("entry_rules") or {}).get("min_rr"), 0) < 1:
        errors.append("min_rr must be >= 1")
    version = str(target.get("version") or "")
    if not version.startswith("S"):
        errors.append("version must start with S")
    return {"valid": not errors, "errors": errors, "version": version}


# -----------------------------------------------------------------------------
# Brain reset / persistence helper
# -----------------------------------------------------------------------------

def _reset_learning_state(keep_strategy: bool = True) -> None:
    global _ACTIVE_STRATEGY, _CHAMPION
    with _STATE_LOCK:
        _candidates.clear(); _outcomes.clear(); _evidence.clear(); _patterns.clear()
        _hypotheses.clear(); _experiments.clear(); _challengers.clear(); _beliefs.clear()
        _research_journal.clear(); _research_queue.clear(); _ollama_queue.clear(); _strategy_history.clear()
        _brain_stats.update({
            "observations": 0, "candidates": 0, "outcomes": 0, "patterns": 0,
            "hypotheses_created": 0, "experiments_completed": 0, "challengers_created": 0,
            "promotions": 0, "rejections": 0, "missed_opportunities": 0, "shadow_outcomes": 0,
            "last_learning_at": None, "last_strategy_change_at": None,
        })
        _frequency_state.clear(); _drift_state.clear()
        _calibration_state.clear(); _calibration_state.update({"bins": {str(i): {"n": 0, "wins": 0, "r_sum": 0.0} for i in range(10, 101, 10)}, "last_update": 0.0})
        if keep_strategy:
            _brain_log(logging.INFO, f"[BRAIN] learning reset; strategy preserved at {_ACTIVE_STRATEGY.version}")
        else:
            _ACTIVE_STRATEGY = copy.deepcopy(_DEFAULT_PROFILE)
            _CHAMPION = copy.deepcopy(_DEFAULT_PROFILE)


def reset_brain_learning() -> None:
    _reset_learning_state(keep_strategy=True)


# -----------------------------------------------------------------------------
# Historical replay API — lazy, bounded
# -----------------------------------------------------------------------------

def _load_historical_frame(path: str, max_rows: int = 100_000) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    # Read only a bounded tail by default. Full multi-month research can be
    # chunked by a caller without keeping the entire file resident.
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    except Exception as exc:
        raise RuntimeError(f"historical load failed: {exc}") from exc
    if len(df) > max_rows:
        df = df.tail(max_rows).copy()
    return _normalize_ohlcv(df)


def historical_replay(data: Any, symbol: str = "HIST", interval: str = "M15", max_rows: int = 100_000) -> dict[str, Any]:
    if isinstance(data, (str, os.PathLike)):
        df = _load_historical_frame(str(data), max_rows=max_rows)
    else:
        df = _normalize_ohlcv(data)
        if len(df) > max_rows:
            df = df.tail(max_rows).copy()
    if len(df) < 80:
        return {"status": "INSUFFICIENT_DATA", "rows": len(df)}
    # Use a rolling reconstruction without holding per-step DataFrames.
    observations = 0
    signals = 0
    pseudo_outcomes = []
    step = 1
    for i in range(70, len(df)-5, step):
        window = df.iloc[max(0, i-250):i+1]
        h1 = window.iloc[::4].copy() if len(window) >= 50 else window.copy()
        m15 = window.copy()
        try:
            packet = full_analyze(h1, m15, None, symbol=symbol)
        except Exception:
            continue
        observations += 1
        if packet.get("decision") in {"BUY", "SELL"} and packet.get("sl") is not None and packet.get("tp") is not None:
            signals += 1
            future = df.iloc[i+1:i+6]
            entry = _finite(packet.get("entry"), 0)
            sl = _finite(packet.get("sl"), entry)
            tp = _finite(packet.get("tp"), entry)
            buy = packet["decision"] == "BUY"
            result = None; exit_px = None
            for _, r in future.iterrows():
                hi, lo = float(r["high"]), float(r["low"])
                if buy:
                    if hi >= tp and lo <= sl:
                        result, exit_px = "sl", sl
                        break
                    if hi >= tp:
                        result, exit_px = "tp", tp; break
                    if lo <= sl:
                        result, exit_px = "sl", sl; break
                else:
                    if lo <= tp and hi >= sl:
                        result, exit_px = "sl", sl; break
                    if lo <= tp:
                        result, exit_px = "tp", tp; break
                    if hi >= sl:
                        result, exit_px = "sl", sl; break
            if result:
                pseudo_outcomes.append(_r_multiple(entry, sl, float(exit_px), packet["decision"]))
    return {
        "status": "COMPLETED", "symbol": symbol, "interval": interval,
        "rows": len(df), "observations": observations, "signals": signals,
        "expected_r": _mean(pseudo_outcomes), "win_rate": (sum(1 for r in pseudo_outcomes if r > 0)/len(pseudo_outcomes) if pseudo_outcomes else 0.0),
    }


# -----------------------------------------------------------------------------
# Startup-safe optional auto-learning flag
# -----------------------------------------------------------------------------
def brain_mode_enabled() -> bool:
    return bool(_FULL_MODE)


# -----------------------------------------------------------------------------
# Compatibility aliases expected by main.py and future integrations
# -----------------------------------------------------------------------------
analyze_market = full_analyze
record_market_observation = ingest_live_candidate


# -----------------------------------------------------------------------------
# Self-check: pure, no network, no Binance, no worker creation.
# -----------------------------------------------------------------------------
def self_check() -> dict[str, Any]:
    checks = {}
    checks["brain_interface"] = BRAIN_INTERFACE_VERSION == "brain_v1"
    checks["brain_schema"] = BRAIN_SCHEMA_VERSION == "brain_state_v1"
    checks["strategy_valid"] = bool(validate_strategy()["valid"])
    required = [
        "full_analyze", "manage_position", "ingest_live_candidate", "ingest_live_outcome",
        "research_cycle", "get_active_strategy", "get_brain_state", "load_brain_state",
        "validate_strategy", "export_checkpoint_state", "import_checkpoint_state",
    ]
    checks["public_contract"] = all(callable(globals().get(x)) for x in required)
    checks["no_binance_mutation"] = not any(name.lower().startswith(("place_", "cancel_", "set_leverage")) for name in globals() if callable(globals().get(name)) and name.lower() in {"place_order", "place_market_order", "place_limit_order", "cancel_order", "set_leverage"})
    return {"ok": all(checks.values()), "checks": checks, "strategy_version": _ACTIVE_STRATEGY.version}


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2, ensure_ascii=False))
