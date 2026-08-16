# ============================================================
# STRATEGY LOGIC TERBAIK
# Date: 16 August 2026
# Time: 17:14 WIB (UTC+7)
#
# PURPOSE
# -------
# Replacement for the previous strategy_logic.py.
#
# Design rules:
# 1) ENTRY -> SL -> TP is strictly ordered.
# 2) Entry remains LIMIT-compatible with the existing main.py.
# 3) RR is MIN 1:2, MAX 1:4.
# 4) A sub-2R setup is NOT rejected immediately: TP is extended
#    through reachable structural/liquidity targets, capped at 4R.
# 5) Liquidity sweep is treated as an event, not as an entry by itself.
# 6) SL is an INVALIDATION level. It is deliberately outside the
#    likely liquidity sweep, not simply the nearest visible swing.
# 7) Trailing is STRUCTURE/WEAKENING aware, not an R-profit ladder.
#    Existing main.py consumes _raw_swing_pts() for its structural trail,
#    therefore _raw_swing_pts() below deliberately returns only
#    "protected" structural points.
# 8) RSI + volume are used as timing/strength measurements.
# 9) No additional API requests are made by this module.
#    Everything is derived from the OHLCV DataFrames passed by main.py.
#
# SOURCE PRINCIPLES
# -----------------
# combined(1).txt is treated as the primary playbook:
# - HTF POI/OB/FVG first, LTF structure confirmation next.
# - Fresh zones outrank mitigated zones.
# - Discount/premium is a filter, not a standalone signal.
# - Liquidity pools are magnets; a sweep + reclaim + structure shift
#   is materially stronger than a raw wick or EQ level.
# - RSI divergence is an early warning, not a standalone reversal trigger.
# - Trend strength is measured through the speed/quality of new extrema,
#   not merely through a pretty trendline.
#
# Web research was used only as a cross-check of the same ideas:
# liquidity sweep definitions consistently require a breach followed
# by rejection/close back inside; sweep alone is not a complete trade.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any, List

import math
import numpy as np
import pandas as pd


# ============================================================
# PUBLIC CONSTANTS — main.py compatibility
# ============================================================

MIN_RR = 2.0
MAX_RR = 4.0

# IMPORTANT:
# main.py's old R-ladder is explicitly disabled because the requested
# trail is not a profit-lock ladder. Trailing is structural and is
# driven through _raw_swing_pts().
TRAIL_R_LADDER: list = []

# main.py calls _raw_swing_pts(df, lb=STRUCT_TRAIL_LB), then:
# BUY  -> latest returned swing-low - entry*STRUCT_TRAIL_BUF_PCT
# SELL -> latest returned swing-high + entry*STRUCT_TRAIL_BUF_PCT
#
# We therefore make _raw_swing_pts() selective and return only structural
# anchors that survived a trend-strength / follow-through quality test.
STRUCT_TRAIL_LB = 3
STRUCT_TRAIL_BUF_PCT = 0.0018
STRUCT_TRAIL_LOOKBACK = 90

# Kept for compatibility with old main.py / /info.
FIB_EXT_1 = 1.272
FIB_EXT_2 = 1.618
H4_RSI_BUY_MIN, H4_RSI_BUY_MAX = 45, 68
H4_RSI_SELL_MIN, H4_RSI_SELL_MAX = 32, 55

# Strategy tuning.
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
EMA_LONG = 200
RSI_LEN = 14
ATR_LEN = 14
VOL_LEN = 20

# Zone quality.
ZONE_LOOKBACK_H1 = 140
ZONE_LOOKBACK_M15 = 180
SWING_LB_H1 = 3
SWING_LB_M15 = 3
EQ_ATR_TOL = 0.12
FVG_MIN_ATR = 0.10
OB_MIN_BODY_ATR = 0.35

# Signal timing.
RECENT_CONFIRM_BARS = 8
SWEEP_VALID_BARS = 8
MAX_ENTRY_DISTANCE_ATR = 1.45

# SL: deliberately outside the event which would invalidate the thesis.
SL_ATR_BUFFER = 0.20
SL_SWEEP_BUFFER = 0.18

# Trend strength / trailing.
TRAIL_MIN_R = 0.90
TRAIL_STRONG_THRESHOLD = 70.0
TRAIL_DECAY_THRESHOLD = -10.0


# ============================================================
# BASIC NUMERICAL HELPERS
# ============================================================

def _as_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if not math.isfinite(v):
        return lo
    return float(max(lo, min(hi, v)))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or abs(b) < 1e-12:
        return default
    return a / b


def _ensure_ohlcv(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 20:
        return None

    out = df.copy()
    rename = {c: c.lower() for c in out.columns}
    out.rename(columns=rename, inplace=True)

    required = ["open", "high", "low", "close", "volume"]
    if any(c not in out.columns for c in required):
        return None

    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=required).copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass

    out = out.sort_index()
    return out


def _closed_candles(df: pd.DataFrame, interval_minutes: Optional[int]) -> pd.DataFrame:
    """
    Drop only the currently-open candle when the timestamp tells us it is open.
    Historical backtests remain untouched because all candles are already closed.
    """
    out = df.copy()
    if out.empty or interval_minutes is None:
        return out

    try:
        ts = pd.Timestamp(out.index[-1])
        now = pd.Timestamp.utcnow()
        if now.tzinfo is not None:
            now = now.tz_localize(None)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)

        if ts + pd.Timedelta(minutes=interval_minutes) > now:
            return out.iloc[:-1].copy()
    except Exception:
        pass
    return out


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = RSI_LEN) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    avg_up = up.ewm(alpha=1 / length, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_up / avg_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def atr(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _volume_ratio(df: pd.DataFrame, length: int = VOL_LEN) -> pd.Series:
    base = df["volume"].rolling(length, min_periods=max(5, length // 2)).median()
    return df["volume"] / base.replace(0, np.nan)


def build_df(
    df: pd.DataFrame,
    interval_minutes: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Add the indicators required by the strategy. No network activity.
    """
    out = _ensure_ohlcv(df)
    if out is None:
        return None

    if interval_minutes is not None:
        out = _closed_candles(out, interval_minutes)

    if len(out) < 60:
        return None

    out["ema9"] = ema(out["close"], EMA_FAST)
    out["ema21"] = ema(out["close"], EMA_MID)
    out["ema50"] = ema(out["close"], EMA_SLOW)
    out["ema200"] = ema(out["close"], EMA_LONG) if len(out) >= EMA_LONG else ema(out["close"], EMA_SLOW)
    out["rsi"] = rsi(out["close"], RSI_LEN)
    out["atr"] = atr(out, ATR_LEN)
    out["vol_sma"] = out["volume"].rolling(VOL_LEN, min_periods=5).mean()
    out["vol_ratio"] = _volume_ratio(out, VOL_LEN)

    # Candle geometry, useful for displacement and rejection.
    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = (out["high"] - out["low"]).clip(lower=1e-12)
    out["body_ratio"] = out["body"] / out["range"]
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

    return out.dropna(subset=["ema9", "ema21", "ema50", "rsi", "atr", "vol_ratio"]).copy()


# ============================================================
# MARKET STRUCTURE
# ============================================================

def _raw_swing_pts(df: pd.DataFrame, lb: int = 3):
    """
    Confirmed pivot detector with trail-quality evidence.

    Internal structure analysis uses this raw-but-confirmed function.
    A pivot becomes higher quality when the move away from it is:
      - materially larger than noise (ATR-normalised),
      - supported by directional RSI movement,
      - accompanied by reasonable volume,
      - and, where possible, produces continuation.

    This is intentionally deterministic and uses only OHLCV already supplied
    by main.py.
    """
    d = _ensure_ohlcv(df)
    if d is None or len(d) < max(2 * lb + 7, 25):
        return [], []

    x = build_df(d)
    if x is None:
        x = d.copy()
        x["atr"] = atr(x)
        x["rsi"] = rsi(x["close"])
        x["vol_ratio"] = _volume_ratio(x)

    highs: List[int] = []
    lows: List[int] = []

    hi_vals = x["high"].to_numpy()
    lo_vals = x["low"].to_numpy()
    close_vals = x["close"].to_numpy()
    n = len(x)

    for i in range(lb, n - lb):
        wh = hi_vals[i - lb:i + lb + 1]
        wl = lo_vals[i - lb:i + lb + 1]
        a = _as_float(x["atr"].iloc[i], np.nan)
        if not math.isfinite(a) or a <= 0:
            continue

        # --------------------------------------------------------
        # HIGH QUALITY HIGH -> possible SELL trailing anchor.
        # --------------------------------------------------------
        if hi_vals[i] >= np.max(wh):
            post_end = min(n, i + 6)
            post = x.iloc[i + 1:post_end]
            if len(post):
                follow = hi_vals[i] - float(post["close"].min())
                move_atr = follow / a

                rsi_departure = float(x["rsi"].iloc[i]) - float(x["rsi"].iloc[post_end - 1])
                vol_participation = float(post["vol_ratio"].max())
                body_follow = float(((post["close"] - post["open"]).clip(upper=0).abs() / post["range"]).max())

                score = (
                    45.0 * _clip(move_atr / 1.0, 0, 1)
                    + 20.0 * _clip(rsi_departure / 8.0, 0, 1)
                    + 20.0 * _clip(vol_participation / 1.5, 0, 1)
                    + 15.0 * _clip(body_follow, 0, 1)
                )

                if move_atr >= 0.25 and score >= 22:
                    highs.append(i)

        # --------------------------------------------------------
        # HIGH QUALITY LOW -> possible BUY trailing anchor.
        # --------------------------------------------------------
        if lo_vals[i] <= np.min(wl):
            post_end = min(n, i + 6)
            post = x.iloc[i + 1:post_end]
            if len(post):
                follow = float(post["close"].max()) - lo_vals[i]
                move_atr = follow / a

                rsi_departure = float(x["rsi"].iloc[post_end - 1]) - float(x["rsi"].iloc[i])
                vol_participation = float(post["vol_ratio"].max())
                body_follow = float(((post["close"] - post["open"]).clip(lower=0).abs() / post["range"]).max())

                score = (
                    45.0 * _clip(move_atr / 1.0, 0, 1)
                    + 20.0 * _clip(rsi_departure / 8.0, 0, 1)
                    + 20.0 * _clip(vol_participation / 1.5, 0, 1)
                    + 15.0 * _clip(body_follow, 0, 1)
                )

                if move_atr >= 0.25 and score >= 22:
                    lows.append(i)

    # Safe fallback for very quiet assets.
    if not highs or not lows:
        for i in range(lb, n - lb):
            if hi_vals[i] >= np.max(hi_vals[i - lb:i + lb + 1]) and i not in highs:
                highs.append(i)
            if lo_vals[i] <= np.min(lo_vals[i - lb:i + lb + 1]) and i not in lows:
                lows.append(i)

    return highs, lows


def swing_pts(df: pd.DataFrame, lb: int = STRUCT_TRAIL_LB):
    """
    PUBLIC TRAILING HOOK FOR main.py.

    main.py uses the latest returned swing-low/high as a new SL anchor.
    Therefore this public version exposes only anchors on the correct side
    of the current market price:
        - highs must be ABOVE current close;
        - lows must be BELOW current close.

    The internal strategy uses _raw_swing_pts() so filtering here does not
    distort market-structure analysis.
    """
    d = _ensure_ohlcv(df)
    if d is None:
        return [], []

    highs, lows = _raw_swing_pts(d, lb=lb)
    current = float(d["close"].iloc[-1])

    # main.py adds entry*STRUCT_TRAIL_BUF_PCT to a high for SELL and
    # subtracts it from a low for BUY. Filter against that final trigger
    # geometry, not merely against the raw pivot.
    sell_margin = current * STRUCT_TRAIL_BUF_PCT
    buy_margin = current * STRUCT_TRAIL_BUF_PCT

    protected_highs = [
        i for i in highs
        if float(d["high"].iloc[i]) + sell_margin > current
    ]
    protected_lows = [
        i for i in lows
        if float(d["low"].iloc[i]) - buy_margin < current
    ]

    # No safe anchor => do not trail this cycle.
    return protected_highs, protected_lows


def mkt_struct(df: pd.DataFrame, lb: int = 3) -> str:
    d = _ensure_ohlcv(df)
    if d is None or len(d) < 20:
        return "ranging"

    sh, sl = _raw_swing_pts(d, lb=lb)

    if len(sh) < 2 or len(sl) < 2:
        return "ranging"

    h1, h2 = d["high"].iloc[sh[-2]], d["high"].iloc[sh[-1]]
    l1, l2 = d["low"].iloc[sl[-2]], d["low"].iloc[sl[-1]]

    bullish = h2 > h1 and l2 > l1
    bearish = h2 < h1 and l2 < l1

    if bullish:
        return "bullish"
    if bearish:
        return "bearish"
    return "ranging"


def detect_bos(df: pd.DataFrame, direction: Optional[str] = None, lb: int = 3) -> dict:
    d = _ensure_ohlcv(df)
    if d is None or len(d) < 20:
        return {}

    sh, sl = _raw_swing_pts(d, lb=lb)
    last_close = float(d["close"].iloc[-1])

    out = {
        "bullish_bos": False,
        "bearish_bos": False,
        "level": None,
        "bars_ago": None,
    }

    if sh:
        level = float(d["high"].iloc[sh[-1]])
        if last_close > level:
            out.update({"bullish_bos": True, "level": level, "bars_ago": len(d) - 1 - sh[-1]})

    if sl:
        level = float(d["low"].iloc[sl[-1]])
        if last_close < level:
            out.update({"bearish_bos": True, "level": level, "bars_ago": len(d) - 1 - sl[-1]})

    if direction == "bull":
        out["bearish_bos"] = False
    elif direction == "bear":
        out["bullish_bos"] = False

    return out


def detect_choch(df: pd.DataFrame, lb: int = 3) -> dict:
    """
    CHoCH requires a CLOSE through the prior valid swing, not merely a wick.
    This follows the supplied playbook's distinction between a shadow break
    and a genuine structural shift.
    """
    d = _ensure_ohlcv(df)
    if d is None or len(d) < 30:
        return {
            "bullish_choch": False,
            "bearish_choch": False,
            "level": None,
            "bars_ago": None,
        }

    sh, sl = _raw_swing_pts(d, lb=lb)
    if len(sh) < 2 or len(sl) < 2:
        return {
            "bullish_choch": False,
            "bearish_choch": False,
            "level": None,
            "bars_ago": None,
        }

    structure = mkt_struct(d, lb=lb)
    close = float(d["close"].iloc[-1])

    out = {
        "bullish_choch": False,
        "bearish_choch": False,
        "level": None,
        "bars_ago": None,
    }

    if structure == "bearish":
        level = float(d["high"].iloc[sh[-1]])
        if close > level:
            out.update({
                "bullish_choch": True,
                "level": level,
                "bars_ago": len(d) - 1 - sh[-1],
            })

    elif structure == "bullish":
        level = float(d["low"].iloc[sl[-1]])
        if close < level:
            out.update({
                "bearish_choch": True,
                "level": level,
                "bars_ago": len(d) - 1 - sl[-1],
            })

    return out


# ============================================================
# LIQUIDITY / EQUALITY / SWEEP
# ============================================================

def detect_equal_highs_lows(
    df: pd.DataFrame,
    lb: int = 3,
    atr_tolerance: float = EQ_ATR_TOL,
) -> dict:
    d = _ensure_ohlcv(df)
    if d is None:
        return {"equal_highs": [], "equal_lows": []}

    sh, sl = _raw_swing_pts(d, lb=lb)
    atr_s = atr(d)

    eq_highs: List[float] = []
    eq_lows: List[float] = []

    tol = float(atr_s.iloc[-1]) * atr_tolerance if len(atr_s) else 0.0

    for a, b in zip(sh[:-1], sh[1:]):
        va = float(d["high"].iloc[a])
        vb = float(d["high"].iloc[b])
        if abs(va - vb) <= max(tol, va * 0.0005):
            eq_highs.append((va + vb) / 2)

    for a, b in zip(sl[:-1], sl[1:]):
        va = float(d["low"].iloc[a])
        vb = float(d["low"].iloc[b])
        if abs(va - vb) <= max(tol, va * 0.0005):
            eq_lows.append((va + vb) / 2)

    return {
        "equal_highs": eq_highs,
        "equal_lows": eq_lows,
    }


def _nearest_swing_level(
    d: pd.DataFrame,
    direction: str,
    around: float,
    lb: int = 3,
    max_distance_atr: float = 4.0,
) -> Optional[float]:
    sh, sl = _raw_swing_pts(d, lb=lb)
    atr_v = float(atr(d).iloc[-1])

    if direction == "above":
        vals = [float(d["high"].iloc[i]) for i in sh if float(d["high"].iloc[i]) > around]
        vals = [v for v in vals if v - around <= max_distance_atr * atr_v]
        return min(vals) if vals else None

    vals = [float(d["low"].iloc[i]) for i in sl if float(d["low"].iloc[i]) < around]
    vals = [v for v in vals if around - v <= max_distance_atr * atr_v]
    return max(vals) if vals else None


def detect_liquidity_sweep(
    df: pd.DataFrame,
    lookback: int = 40,
    lb: int = 3,
) -> dict:
    """
    Strict definition:
    - Sweep low: low pierces a prior swing / equal-low liquidity pool,
      but candle CLOSES back above that level.
    - Sweep high: high pierces a prior swing / equal-high pool,
      but candle CLOSES back below that level.

    A wick alone is not treated as a trade signal.
    """
    d = _ensure_ohlcv(df)
    if d is None or len(d) < lookback + 10:
        return {
            "bullish_sweep": False,
            "bearish_sweep": False,
            "level": None,
            "extreme": None,
            "bars_ago": None,
            "volume_ratio": 0.0,
            "strength": 0.0,
        }

    sh, sl = _raw_swing_pts(d, lb=lb)
    start = max(lb + 1, len(d) - lookback)
    atr_s = atr(d)
    vr = _volume_ratio(d)

    best_low = None
    best_high = None

    prior_lows = [(i, float(d["low"].iloc[i])) for i in sl if i < len(d) - 1]
    prior_highs = [(i, float(d["high"].iloc[i])) for i in sh if i < len(d) - 1]

    for i in range(start, len(d)):
        row = d.iloc[i]
        a = float(atr_s.iloc[i]) if math.isfinite(float(atr_s.iloc[i])) else 0.0
        if a <= 0:
            continue

        # Bullish sweep.
        for j, level in reversed(prior_lows[-12:]):
            if j >= i:
                continue
            wick_depth = level - float(row["low"])
            if wick_depth <= 0:
                continue

            close_back = float(row["close"]) > level
            if close_back:
                score = (
                    35.0
                    + 20.0 * _clip(wick_depth / (0.5 * a), 0, 1)
                    + 20.0 * _clip(float(vr.iloc[i]) / 2.0, 0, 1)
                    + 10.0 * _clip(float(row["body_ratio"]), 0, 1)
                )
                best_low = {
                    "bullish_sweep": True,
                    "bearish_sweep": False,
                    "level": level,
                    "extreme": float(row["low"]),
                    "bars_ago": len(d) - 1 - i,
                    "volume_ratio": float(vr.iloc[i]),
                    "strength": _clip(score),
                }
                break

        # Bearish sweep.
        for j, level in reversed(prior_highs[-12:]):
            if j >= i:
                continue
            wick_depth = float(row["high"]) - level
            if wick_depth <= 0:
                continue

            close_back = float(row["close"]) < level
            if close_back:
                score = (
                    35.0
                    + 20.0 * _clip(wick_depth / (0.5 * a), 0, 1)
                    + 20.0 * _clip(float(vr.iloc[i]) / 2.0, 0, 1)
                    + 10.0 * _clip(float(row["body_ratio"]), 0, 1)
                )
                best_high = {
                    "bullish_sweep": False,
                    "bearish_sweep": True,
                    "level": level,
                    "extreme": float(row["high"]),
                    "bars_ago": len(d) - 1 - i,
                    "volume_ratio": float(vr.iloc[i]),
                    "strength": _clip(score),
                }
                break

    if best_low and best_high:
        return best_low if best_low["bars_ago"] <= best_high["bars_ago"] else best_high
    if best_low:
        return best_low
    if best_high:
        return best_high

    return {
        "bullish_sweep": False,
        "bearish_sweep": False,
        "level": None,
        "extreme": None,
        "bars_ago": None,
        "volume_ratio": 0.0,
        "strength": 0.0,
    }


# ============================================================
# FVG / ORDER BLOCK / POI
# ============================================================

def detect_fvg(df: pd.DataFrame, lookback: int = 100) -> List[dict]:
    d = _ensure_ohlcv(df)
    if d is None or len(d) < 10:
        return []

    atr_s = atr(d)
    start = max(2, len(d) - lookback)
    zones: List[dict] = []

    for i in range(start, len(d)):
        a = d.iloc[i - 2]
        c = d.iloc[i]
        a_val = float(atr_s.iloc[i]) if math.isfinite(float(atr_s.iloc[i])) else 0.0
        if a_val <= 0:
            continue

        # Bullish 3-candle imbalance.
        if float(c["low"]) > float(a["high"]):
            lo = float(a["high"])
            hi = float(c["low"])
            if hi - lo >= FVG_MIN_ATR * a_val:
                zones.append({
                    "type": "bullish_fvg",
                    "lo": lo,
                    "hi": hi,
                    "mid": (lo + hi) / 2,
                    "created": i,
                    "width": hi - lo,
                })

        # Bearish 3-candle imbalance.
        if float(c["high"]) < float(a["low"]):
            lo = float(c["high"])
            hi = float(a["low"])
            if hi - lo >= FVG_MIN_ATR * a_val:
                zones.append({
                    "type": "bearish_fvg",
                    "lo": lo,
                    "hi": hi,
                    "mid": (lo + hi) / 2,
                    "created": i,
                    "width": hi - lo,
                })

    # Mark mitigation.
    for z in zones:
        future = d.iloc[z["created"] + 1:]
        if z["type"] == "bullish_fvg":
            z["mitigated"] = bool(len(future) and (future["low"] <= z["lo"]).any())
        else:
            z["mitigated"] = bool(len(future) and (future["high"] >= z["hi"]).any())
        z["fresh"] = not z["mitigated"]

    return zones


def detect_order_block(df: pd.DataFrame, lookback: int = 120) -> List[dict]:
    """
    OB = last opposing candle before an impulsive displacement which breaks
    a recent structural level. A raw candle is not considered a high quality
    OB unless there is directional follow-through.
    """
    d = _ensure_ohlcv(df)
    if d is None or len(d) < 30:
        return []

    sh, sl = _raw_swing_pts(d, lb=3)
    atr_s = atr(d)
    vr = _volume_ratio(d)

    start = max(3, len(d) - lookback)
    zones: List[dict] = []

    for i in range(start, len(d) - 3):
        row = d.iloc[i]
        future = d.iloc[i + 1:min(len(d), i + 4)]

        a = float(atr_s.iloc[i]) if math.isfinite(float(atr_s.iloc[i])) else 0.0
        if a <= 0:
            continue

        body = abs(float(row["close"]) - float(row["open"]))
        if body < OB_MIN_BODY_ATR * a:
            continue

        # Bullish displacement after a bearish candle.
        if float(row["close"]) < float(row["open"]):
            highs = [float(d["high"].iloc[j]) for j in sh if j < i]
            prior_high = max(highs[-3:], default=np.nan)
            if not math.isfinite(prior_high):
                continue

            future_close = float(future["close"].max()) if len(future) else float(row["close"])
            excursion = future_close - float(row["low"])

            if future_close > prior_high and excursion >= 0.8 * a:
                lo = float(row["low"])
                hi = float(row["high"])
                zones.append({
                    "type": "bullish_ob",
                    "lo": lo,
                    "hi": hi,
                    "mid": (lo + hi) / 2,
                    "created": i,
                    "fresh": True,
                    "volume_ratio": float(vr.iloc[i]),
                    "displacement": excursion / a,
                })

        # Bearish displacement after a bullish candle.
        if float(row["close"]) > float(row["open"]):
            lows = [float(d["low"].iloc[j]) for j in sl if j < i]
            prior_low = min(lows[-3:], default=np.nan)
            if not math.isfinite(prior_low):
                continue

            future_close = float(future["close"].min()) if len(future) else float(row["close"])
            excursion = float(row["high"]) - future_close

            if future_close < prior_low and excursion >= 0.8 * a:
                lo = float(row["low"])
                hi = float(row["high"])
                zones.append({
                    "type": "bearish_ob",
                    "lo": lo,
                    "hi": hi,
                    "mid": (lo + hi) / 2,
                    "created": i,
                    "fresh": True,
                    "volume_ratio": float(vr.iloc[i]),
                    "displacement": excursion / a,
                })

    # A zone is mitigated even if only its wick is revisited.
    for z in zones:
        future = d.iloc[z["created"] + 1:]
        if z["type"] == "bullish_ob":
            z["fresh"] = not bool(len(future) and (future["low"] <= z["hi"]).any())
        else:
            z["fresh"] = not bool(len(future) and (future["high"] >= z["lo"]).any())

    return zones


# ============================================================
# RSI / VOLUME / TREND STRENGTH
# ============================================================

def detect_rsi_divergence(df: pd.DataFrame, lb: int = 3) -> dict:
    d = _ensure_ohlcv(df)
    if d is None or len(d) < 40:
        return {
            "bullish": False,
            "bearish": False,
            "strength": 0.0,
            "rsi": 50.0,
            "slope": 0.0,
        }

    x = build_df(d)
    if x is None:
        return {
            "bullish": False,
            "bearish": False,
            "strength": 0.0,
            "rsi": 50.0,
            "slope": 0.0,
        }

    sh, sl = _raw_swing_pts(x, lb=lb)
    bullish = bearish = False
    strength = 0.0

    if len(sl) >= 2:
        a, b = sl[-2], sl[-1]
        price_a, price_b = float(x["low"].iloc[a]), float(x["low"].iloc[b])
        rsi_a, rsi_b = float(x["rsi"].iloc[a]), float(x["rsi"].iloc[b])
        if price_b < price_a and rsi_b > rsi_a + 2.0:
            bullish = True
            strength = max(strength, min(100.0, (rsi_b - rsi_a) * 5.0))

    if len(sh) >= 2:
        a, b = sh[-2], sh[-1]
        price_a, price_b = float(x["high"].iloc[a]), float(x["high"].iloc[b])
        rsi_a, rsi_b = float(x["rsi"].iloc[a]), float(x["rsi"].iloc[b])
        if price_b > price_a and rsi_b < rsi_a - 2.0:
            bearish = True
            strength = max(strength, min(100.0, (rsi_a - rsi_b) * 5.0))

    slope = float(x["rsi"].iloc[-1] - x["rsi"].iloc[-4]) if len(x) >= 4 else 0.0

    return {
        "bullish": bullish,
        "bearish": bearish,
        "strength": _clip(strength),
        "rsi": float(x["rsi"].iloc[-1]),
        "slope": slope,
    }


def _price_time_trend_strength(df: pd.DataFrame, direction: str) -> float:
    """
    Primary trend-strength idea from combined.txt:
    measure how quickly the relevant new extrema are being produced,
    not just the visual angle of a trendline.

    We use:
    - distance moved per candle / ATR,
    - consistency of new extrema,
    - pullback depth,
    - EMA slope,
    - RSI slope,
    - volume participation.
    """
    d = build_df(df)
    if d is None or len(d) < 50:
        return 0.0

    sh, sl = _raw_swing_pts(d, lb=3)
    atr_now = max(float(d["atr"].iloc[-1]), 1e-12)

    if direction == "bull":
        pts = sh[-4:]
        if len(pts) < 2:
            return 25.0

        speeds = []
        for a, b in zip(pts[:-1], pts[1:]):
            dt = max(1, b - a)
            move = float(d["high"].iloc[b] - d["high"].iloc[a])
            speeds.append(move / (dt * atr_now))

        hh_ratio = sum(
            float(d["high"].iloc[b]) > float(d["high"].iloc[a])
            for a, b in zip(pts[:-1], pts[1:])
        ) / max(1, len(pts) - 1)

        ema_slope = _safe_div(
            float(d["ema21"].iloc[-1] - d["ema21"].iloc[-6]),
            atr_now * 5.0,
        )
        rsi_slope = float(d["rsi"].iloc[-1] - d["rsi"].iloc[-5]) / 10.0
        vol_ratio = float(d["vol_ratio"].iloc[-5:].mean())

        score = (
            35.0 * _clip(np.mean(speeds) / 0.35, 0, 1)
            + 25.0 * hh_ratio
            + 15.0 * _clip(max(ema_slope, 0), 0, 1)
            + 10.0 * _clip(max(rsi_slope, 0), 0, 1)
            + 15.0 * _clip(vol_ratio / 1.5, 0, 1)
        )
        return _clip(score)

    pts = sl[-4:]
    if len(pts) < 2:
        return 25.0

    speeds = []
    for a, b in zip(pts[:-1], pts[1:]):
        dt = max(1, b - a)
        move = float(d["low"].iloc[a] - d["low"].iloc[b])
        speeds.append(move / (dt * atr_now))

    ll_ratio = sum(
        float(d["low"].iloc[b]) < float(d["low"].iloc[a])
        for a, b in zip(pts[:-1], pts[1:])
    ) / max(1, len(pts) - 1)

    ema_slope = _safe_div(
        float(d["ema21"].iloc[5] - d["ema21"].iloc[0]),
        atr_now * max(5, len(d) - 1),
    ) if len(d) >= 10 else 0.0

    # For bearish trend, negative RSI/EMA slope is supportive.
    rsi_slope = (float(d["rsi"].iloc[-5]) - float(d["rsi"].iloc[-1])) / 10.0
    vol_ratio = float(d["vol_ratio"].iloc[-5:].mean())

    score = (
        35.0 * _clip(np.mean(speeds) / 0.35, 0, 1)
        + 25.0 * ll_ratio
        + 15.0 * _clip(max(-ema_slope, 0), 0, 1)
        + 10.0 * _clip(max(rsi_slope, 0), 0, 1)
        + 15.0 * _clip(vol_ratio / 1.5, 0, 1)
    )
    return _clip(score)


def _trend_snapshot(
    h1: pd.DataFrame,
    m15: pd.DataFrame,
    d1: Optional[pd.DataFrame],
) -> dict:
    h1b = build_df(h1)
    m15b = build_df(m15)
    d1b = build_df(d1) if d1 is not None else None

    if h1b is None or m15b is None:
        return {
            "h1": "ranging",
            "d1": "neutral",
            "m15": "ranging",
            "bull_strength": 0.0,
            "bear_strength": 0.0,
            "rsi_m15": 50.0,
            "rsi_slope": 0.0,
            "volume_ratio": 1.0,
        }

    h1_struct = mkt_struct(h1b, lb=3)
    m15_struct = mkt_struct(m15b, lb=3)

    d1_struct = "neutral"
    if d1b is not None and len(d1b) >= 20:
        d1_struct = mkt_struct(d1b, lb=3)

    bull_strength = (
        0.50 * _price_time_trend_strength(h1b, "bull")
        + 0.35 * _price_time_trend_strength(m15b, "bull")
    )
    bear_strength = (
        0.50 * _price_time_trend_strength(h1b, "bear")
        + 0.35 * _price_time_trend_strength(m15b, "bear")
    )

    # D1 is contextual, never allowed to overwhelm clear H1/M15 disagreement.
    if d1_struct == "bullish":
        bull_strength += 15
    elif d1_struct == "bearish":
        bear_strength += 15

    return {
        "h1": h1_struct,
        "d1": d1_struct,
        "m15": m15_struct,
        "bull_strength": _clip(bull_strength),
        "bear_strength": _clip(bear_strength),
        "rsi_m15": float(m15b["rsi"].iloc[-1]),
        "rsi_slope": float(m15b["rsi"].iloc[-1] - m15b["rsi"].iloc[-4]) if len(m15b) >= 4 else 0.0,
        "volume_ratio": float(m15b["vol_ratio"].iloc[-5:].mean()),
    }


# ============================================================
# VOLUME PROFILE PROXY
# ============================================================

def _volume_profile_poc(df: pd.DataFrame, bins: int = 32) -> Optional[float]:
    """
    Approximate POC from OHLCV only:
    allocate each candle's volume to its typical price
    and accumulate volume by price bins.
    No order-book request is needed.
    """
    d = _ensure_ohlcv(df)
    if d is None or len(d) < 25:
        return None

    lo = float(d["low"].min())
    hi = float(d["high"].max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None

    typical = (d["high"] + d["low"] + d["close"]) / 3.0
    vol = d["volume"].astype(float)

    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(typical.to_numpy(), edges) - 1, 0, bins - 1)

    hist = np.zeros(bins, dtype=float)
    np.add.at(hist, idx, vol.to_numpy())

    if not np.isfinite(hist).any() or hist.sum() <= 0:
        return None

    k = int(np.nanargmax(hist))
    return float((edges[k] + edges[k + 1]) / 2.0)


# ============================================================
# FIBONACCI / LOCATION
# ============================================================

def _fib_location(
    swing_low: float,
    swing_high: float,
    price: float,
    direction: str,
) -> float:
    rng = swing_high - swing_low
    if rng <= 0:
        return 0.5

    if direction == "bull":
        retr = (swing_high - price) / rng
    else:
        retr = (price - swing_low) / rng

    return float(retr)


def _choose_range(df: pd.DataFrame, direction: str) -> Tuple[float, float]:
    d = _ensure_ohlcv(df)
    if d is None:
        return (np.nan, np.nan)

    sh, sl = _raw_swing_pts(d, lb=3)
    highs = [float(d["high"].iloc[i]) for i in sh[-6:]]
    lows = [float(d["low"].iloc[i]) for i in sl[-6:]]

    if not highs or not lows:
        return float(d["low"].tail(50).min()), float(d["high"].tail(50).max())

    return min(lows), max(highs)


def _zone_confluence(
    zone: dict,
    direction: str,
    h1: pd.DataFrame,
    m15: pd.DataFrame,
    sweep: dict,
    rsi_div: dict,
) -> Tuple[float, List[str]]:
    score = 0.0
    evidence: List[str] = []

    # Freshness.
    if zone.get("fresh", False):
        score += 15
        evidence.append("fresh")
    else:
        score -= 10
        evidence.append("mitigated")

    # Discount/premium.
    rl, rh = _choose_range(h1, direction)
    loc = _fib_location(rl, rh, zone["mid"], direction)

    if direction == "bull":
        if loc >= 0.618:
            score += 15
            evidence.append("discount_0.618+")
        elif loc >= 0.50:
            score += 7
            evidence.append("discount_0.50+")
    else:
        if loc >= 0.618:
            score += 15
            evidence.append("premium_0.618+")
        elif loc >= 0.50:
            score += 7
            evidence.append("premium_0.50+")

    # Volume / displacement.
    if zone.get("volume_ratio", 0.0) >= 1.25:
        score += 10
        evidence.append("volume_expansion")

    if zone.get("displacement", 0.0) >= 1.0:
        score += 10
        evidence.append("displacement")

    # FVG overlap.
    fvg_zones = detect_fvg(m15, lookback=90)
    for f in fvg_zones:
        if direction == "bull" and f["type"] == "bullish_fvg":
            overlap = min(zone["hi"], f["hi"]) - max(zone["lo"], f["lo"])
            if overlap > 0:
                score += 12
                evidence.append("FVG_overlap")
                break
        if direction == "bear" and f["type"] == "bearish_fvg":
            overlap = min(zone["hi"], f["hi"]) - max(zone["lo"], f["lo"])
            if overlap > 0:
                score += 12
                evidence.append("FVG_overlap")
                break

    # Sweep.
    if direction == "bull" and sweep.get("bullish_sweep"):
        score += 20
        evidence.append("liquidity_sweep")
    if direction == "bear" and sweep.get("bearish_sweep"):
        score += 20
        evidence.append("liquidity_sweep")

    # RSI divergence is a reversal/timing bonus, never a standalone trigger.
    if direction == "bull" and rsi_div.get("bullish"):
        score += 10
        evidence.append("bullish_RSI_divergence")
    if direction == "bear" and rsi_div.get("bearish"):
        score += 10
        evidence.append("bearish_RSI_divergence")

    # Volume profile POC proximity.
    poc = _volume_profile_poc(m15, bins=32)
    if poc is not None:
        zone_atr = max(float(atr(m15).iloc[-1]), 1e-12)
        if abs(poc - zone["mid"]) <= 0.5 * zone_atr:
            score += 8
            evidence.append("POC_overlap")

    return _clip(score, -20, 100), evidence


# ============================================================
# ENTRY CANDIDATE ENGINE
# ============================================================

@dataclass
class EntryCandidate:
    direction: str
    entry: float
    zone_lo: float
    zone_hi: float
    label: str
    location_score: float
    evidence: List[str]
    sweep: dict
    rsi_div: dict
    trend_strength: float
    atr: float
    raw_score: float


def _retest_zone_from_sweep(
    m15: pd.DataFrame,
    direction: str,
    sweep: dict,
) -> Optional[dict]:
    if sweep.get("level") is None:
        return None

    x = build_df(m15)
    if x is None or len(x) < 15:
        return None

    # Look for a recent displacement FVG / opposing candle after sweep.
    sweep_idx = len(x) - 1 - int(sweep.get("bars_ago", 999))
    if sweep_idx < 0 or sweep_idx >= len(x) - 1:
        return None

    fvg = detect_fvg(x, lookback=min(40, len(x)))
    expected = "bullish_fvg" if direction == "bull" else "bearish_fvg"
    fvg = [z for z in fvg if z["type"] == expected and z["created"] >= sweep_idx]

    if fvg:
        z = sorted(fvg, key=lambda q: q["created"])[-1]
        return {
            "lo": z["lo"],
            "hi": z["hi"],
            "mid": z["mid"],
            "type": "sweep_fvg",
            "fresh": z.get("fresh", True),
            "displacement": max(0.5, z["width"] / max(float(x["atr"].iloc[-1]), 1e-12)),
            "volume_ratio": float(x["vol_ratio"].iloc[z["created"]]),
        }

    # Fallback: reaction candle zone around sweep.
    end = min(len(x), sweep_idx + 4)
    window = x.iloc[sweep_idx:end]
    if len(window) < 2:
        return None

    if direction == "bull":
        lo = float(window["low"].min())
        hi = float(window["open"].max())
        if hi <= lo:
            hi = float(window["high"].max())
        return {
            "lo": lo,
            "hi": hi,
            "mid": (lo + hi) / 2,
            "type": "sweep_reaction",
            "fresh": True,
            "displacement": 0.7,
            "volume_ratio": float(window["vol_ratio"].max()),
        }

    lo = float(window["close"].min())
    hi = float(window["high"].max())
    if hi <= lo:
        lo = float(window["low"].min())
    return {
        "lo": lo,
        "hi": hi,
        "mid": (lo + hi) / 2,
        "type": "sweep_reaction",
        "fresh": True,
        "displacement": 0.7,
        "volume_ratio": float(window["vol_ratio"].max()),
    }


def _candidate_entry_price(zone: dict, direction: str, current: float, atr_v: float) -> float:
    """
    LIMIT placement:
    - Bull: choose lower half of the demand zone.
    - Bear: choose upper half of the supply zone.
    This improves RR while remaining inside the same identified zone.
    """
    lo, hi = float(zone["lo"]), float(zone["hi"])
    width = max(hi - lo, 1e-12)

    if direction == "bull":
        entry = lo + 0.35 * width
        # Do not place absurdly far from the current market.
        if current - entry > MAX_ENTRY_DISTANCE_ATR * atr_v:
            entry = current - min(MAX_ENTRY_DISTANCE_ATR * atr_v, width * 0.60)
        return min(max(entry, lo), hi)

    entry = hi - 0.35 * width
    if entry - current > MAX_ENTRY_DISTANCE_ATR * atr_v:
        entry = current + min(MAX_ENTRY_DISTANCE_ATR * atr_v, width * 0.60)
    return min(max(entry, lo), hi)


def _collect_entry_candidates(
    h1: pd.DataFrame,
    m15: pd.DataFrame,
    d1: Optional[pd.DataFrame],
    preferred_direction: str,
) -> List[EntryCandidate]:
    h1b = build_df(h1, interval_minutes=60)
    m15b = build_df(m15, interval_minutes=15)
    if h1b is None or m15b is None:
        return []

    cur = float(m15b["close"].iloc[-1])
    atr_v = max(float(m15b["atr"].iloc[-1]), 1e-12)

    sweep = detect_liquidity_sweep(m15b, lookback=50, lb=3)
    rsi_div = detect_rsi_divergence(m15b, lb=3)

    choch15 = detect_choch(m15b, lb=3)
    h1_struct = mkt_struct(h1b, lb=3)

    zones: List[dict] = []

    # 1) H1 fresh OBs — primary POI class.
    for z in detect_order_block(h1b, lookback=ZONE_LOOKBACK_H1):
        if preferred_direction == "bull" and z["type"] == "bullish_ob":
            zones.append(z)
        elif preferred_direction == "bear" and z["type"] == "bearish_ob":
            zones.append(z)

    # 2) H1 FVGs — secondary POI class.
    for z in detect_fvg(h1b, lookback=ZONE_LOOKBACK_H1):
        if preferred_direction == "bull" and z["type"] == "bullish_fvg":
            zones.append(z)
        elif preferred_direction == "bear" and z["type"] == "bearish_fvg":
            zones.append(z)

    # 3) Sweep-derived reaction zone — highest timing quality when available.
    sweep_zone = None
    if preferred_direction == "bull" and sweep.get("bullish_sweep"):
        sweep_zone = _retest_zone_from_sweep(m15b, "bull", sweep)
    elif preferred_direction == "bear" and sweep.get("bearish_sweep"):
        sweep_zone = _retest_zone_from_sweep(m15b, "bear", sweep)

    if sweep_zone is not None:
        zones.insert(0, sweep_zone)

    # Deduplicate by center.
    unique = []
    seen = set()
    for z in zones:
        key = (
            z["type"],
            round(float(z["lo"]) / atr_v, 2),
            round(float(z["hi"]) / atr_v, 2),
        )
        if key not in seen:
            unique.append(z)
            seen.add(key)

    candidates: List[EntryCandidate] = []

    for z in unique:
        # Current price must be above a bullish zone or below a bearish zone
        # for a meaningful pending LIMIT.
        if preferred_direction == "bull" and float(z["lo"]) > cur + MAX_ENTRY_DISTANCE_ATR * atr_v:
            continue
        if preferred_direction == "bear" and float(z["hi"]) < cur - MAX_ENTRY_DISTANCE_ATR * atr_v:
            continue

        entry = _candidate_entry_price(z, preferred_direction, cur, atr_v)

        location_score, evidence = _zone_confluence(
            z, preferred_direction, h1b, m15b, sweep, rsi_div
        )

        # M15 structure confirmation is the critical low-TF timing layer.
        if preferred_direction == "bull" and choch15.get("bullish_choch"):
            location_score += 16
            evidence.append("M15_CHoCH_bull")
        if preferred_direction == "bear" and choch15.get("bearish_choch"):
            location_score += 16
            evidence.append("M15_CHoCH_bear")

        # Existing trend must not be blindly opposed.
        if preferred_direction == "bull" and h1_struct == "bullish":
            location_score += 10
            evidence.append("H1_trend_aligned")
        elif preferred_direction == "bear" and h1_struct == "bearish":
            location_score += 10
            evidence.append("H1_trend_aligned")
        elif h1_struct == "ranging":
            evidence.append("H1_range")
        else:
            location_score -= 10
            evidence.append("H1_countertrend")

        trend_strength = (
            _price_time_trend_strength(m15b, preferred_direction)
            + _price_time_trend_strength(h1b, preferred_direction)
        ) / 2.0

        # RSI timing: early reversal + re-acceleration is better than
        # blindly calling oversold/overbought.
        rsi_now = float(m15b["rsi"].iloc[-1])
        rsi_slope = float(m15b["rsi"].iloc[-1] - m15b["rsi"].iloc[-4])
        if preferred_direction == "bull":
            if rsi_slope > 0 and rsi_now >= 40:
                location_score += 8
                evidence.append("RSI_reaccelerating")
            elif rsi_now < 32 and rsi_slope < 0:
                location_score -= 8
                evidence.append("falling_knife_RSI")
        else:
            if rsi_slope < 0 and rsi_now <= 60:
                location_score += 8
                evidence.append("RSI_reaccelerating")
            elif rsi_now > 68 and rsi_slope > 0:
                location_score -= 8
                evidence.append("climbing_extreme_RSI")

        # Volume timing: increasing volume on the directional impulse is
        # evidence; high volume on the sweep is even more useful.
        recent_vol = float(m15b["vol_ratio"].iloc[-5:].mean())
        if recent_vol >= 1.25:
            location_score += 8
            evidence.append("volume_active")
        elif recent_vol < 0.75:
            location_score -= 3
            evidence.append("volume_thin")

        candidates.append(
            EntryCandidate(
                direction=preferred_direction,
                entry=float(entry),
                zone_lo=float(z["lo"]),
                zone_hi=float(z["hi"]),
                label=z["type"],
                location_score=_clip(location_score, 0, 100),
                evidence=evidence,
                sweep=sweep,
                rsi_div=rsi_div,
                trend_strength=_clip(trend_strength),
                atr=atr_v,
                raw_score=_clip(location_score),
            )
        )

    # Keep best 5 zones for deterministic ranking.
    candidates.sort(
        key=lambda c: (
            c.location_score,
            1 if "liquidity_sweep" in c.evidence else 0,
            1 if "M15_CHoCH_bull" in c.evidence or "M15_CHoCH_bear" in c.evidence else 0,
            c.trend_strength,
        ),
        reverse=True,
    )
    return candidates[:5]


# ============================================================
# SL ENGINE — INVALIDATION, NOT "NEAREST SWING"
# ============================================================

def _compute_sl(
    entry: float,
    direction: str,
    candidate: EntryCandidate,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
) -> Tuple[float, Dict[str, Any]]:
    m15b = build_df(m15)
    h1b = build_df(h1)

    if m15b is None:
        raise ValueError("M15 tidak cukup untuk SL.")

    atr15 = max(float(m15b["atr"].iloc[-1]), 1e-12)

    sh15, sl15 = _raw_swing_pts(m15b, lb=3)
    sh1, sl1 = _raw_swing_pts(h1b, lb=3) if h1b is not None else ([], [])

    recent_low = float(m15b["low"].tail(25).min())
    recent_high = float(m15b["high"].tail(25).max())

    sweep = candidate.sweep
    sweep_extreme = sweep.get("extreme")

    if direction == "bull":
        invalidations = [
            float(candidate.zone_lo),
            recent_low,
        ]
        if sl15:
            invalidations.append(float(m15b["low"].iloc[sl15[-1]]))
        if sl1:
            invalidations.append(float(h1b["low"].iloc[sl1[-1]]))

        base = min(invalidations)

        # If a sweep occurred, the sweep extreme is the key invalidation:
        # dropping through that level means the supposed liquidity rejection
        # did not hold.
        if sweep.get("bullish_sweep") and sweep_extreme is not None:
            base = min(base, float(sweep_extreme))
            buffer = max(SL_SWEEP_BUFFER * atr15, 0.5 * m15b["range"].iloc[-1] * 0.05)
            reason = "below_sweep_and_structure"
        else:
            buffer = SL_ATR_BUFFER * atr15
            reason = "below_structure_and_zone"

        sl = base - buffer

        # Never allow an invalid geometry.
        if sl >= entry:
            sl = entry - max(0.5 * atr15, entry * 0.001)

        return float(sl), {
            "reason": reason,
            "base": float(base),
            "buffer": float(buffer),
            "sweep_extreme": sweep_extreme,
        }

    invalidations = [
        float(candidate.zone_hi),
        recent_high,
    ]
    if sh15:
        invalidations.append(float(m15b["high"].iloc[sh15[-1]]))
    if sh1:
        invalidations.append(float(h1b["high"].iloc[sh1[-1]]))

    base = max(invalidations)

    if sweep.get("bearish_sweep") and sweep_extreme is not None:
        base = max(base, float(sweep_extreme))
        buffer = max(SL_SWEEP_BUFFER * atr15, 0.5 * m15b["range"].iloc[-1] * 0.05)
        reason = "above_sweep_and_structure"
    else:
        buffer = SL_ATR_BUFFER * atr15
        reason = "above_structure_and_zone"

    sl = base + buffer

    if sl <= entry:
        sl = entry + max(0.5 * atr15, entry * 0.001)

    return float(sl), {
        "reason": reason,
        "base": float(base),
        "buffer": float(buffer),
        "sweep_extreme": sweep_extreme,
    }


# ============================================================
# TP ENGINE — ONLY AFTER ENTRY + SL
# ============================================================

def _target_levels(
    direction: str,
    entry: float,
    sl: float,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
) -> List[Tuple[float, str, float]]:
    """
    Build reachable structural/liquidity targets.

    Priority:
    1) external liquidity / swing
    2) EQ pools
    3) opposite FVG/OB edges
    4) volume-profile POC/value proxy
    5) 2R technical fallback only when no nearer obstacle is present
    """
    m15b = build_df(m15)
    h1b = build_df(h1)
    if m15b is None or h1b is None:
        return []

    risk = abs(entry - sl)
    if risk <= 0:
        return []

    raw: List[Tuple[float, str, float]] = []

    if direction == "bull":
        sh15, _ = _raw_swing_pts(m15b, lb=3)
        sh1, _ = _raw_swing_pts(h1b, lb=3)

        for i in sh15[-8:]:
            p = float(m15b["high"].iloc[i])
            if p > entry:
                raw.append((p, "swing_high_m15", abs(p - entry) / risk))

        for i in sh1[-8:]:
            p = float(h1b["high"].iloc[i])
            if p > entry:
                raw.append((p, "swing_high_h1_external", abs(p - entry) / risk))

        eq = detect_equal_highs_lows(h1b, lb=3)
        for p in eq["equal_highs"]:
            if p > entry:
                raw.append((p, "equal_high_h1", abs(p - entry) / risk))

        for z in detect_fvg(h1b, lookback=100):
            if z["type"] == "bearish_fvg" and z["lo"] > entry:
                raw.append((z["lo"], "bearish_fvg_edge", abs(z["lo"] - entry) / risk))

        poc = _volume_profile_poc(h1b)
        if poc is not None and poc > entry:
            raw.append((poc, "volume_profile_poc", abs(poc - entry) / risk))

    else:
        _, sl15 = _raw_swing_pts(m15b, lb=3)
        _, sl1 = _raw_swing_pts(h1b, lb=3)

        for i in sl15[-8:]:
            p = float(m15b["low"].iloc[i])
            if p < entry:
                raw.append((p, "swing_low_m15", abs(entry - p) / risk))

        for i in sl1[-8:]:
            p = float(h1b["low"].iloc[i])
            if p < entry:
                raw.append((p, "swing_low_h1_external", abs(entry - p) / risk))

        eq = detect_equal_highs_lows(h1b, lb=3)
        for p in eq["equal_lows"]:
            if p < entry:
                raw.append((p, "equal_low_h1", abs(entry - p) / risk))

        for z in detect_fvg(h1b, lookback=100):
            if z["type"] == "bullish_fvg" and z["hi"] < entry:
                raw.append((z["hi"], "bullish_fvg_edge", abs(entry - z["hi"]) / risk))

        poc = _volume_profile_poc(h1b)
        if poc is not None and poc < entry:
            raw.append((poc, "volume_profile_poc", abs(entry - poc) / risk))

    # Deduplicate close prices and sort in travel order.
    raw.sort(key=lambda t: t[2])
    out = []
    seen = set()
    for p, label, r in raw:
        k = round(p, 8)
        if k in seen:
            continue
        seen.add(k)
        out.append((p, label, r))
    return out


def _choose_tp_after_sl(
    entry: float,
    sl: float,
    direction: str,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    trend_strength: float,
) -> Tuple[float, str, float, List[str]]:
    """
    TP logic runs only after Entry and SL exist.

    If the nearest target is <2R:
    - inspect subsequent structural/liquidity targets,
    - choose an extension only while it remains <=4R,
    - reject a target that has a major opposing obstacle before it,
    - if the trend is strong, permit a larger target inside the 4R cap.
    """
    risk = abs(entry - sl)
    if risk <= 0:
        raise ValueError("Risk nol.")

    targets = _target_levels(direction, entry, sl, m15, h1)

    reasons: List[str] = []
    min_target = 2.0
    max_target = 4.0

    reachable = [
        t for t in targets
        if min_target - 1e-9 <= t[2] <= max_target + 1e-9
    ]

    # First prefer the nearest truly reachable target.
    if reachable:
        # Strong trend can justify the further target, but we do not
        # automatically chase the furthest possible level.
        if trend_strength >= 78:
            chosen = reachable[min(len(reachable) - 1, 1)]
        else:
            chosen = reachable[0]

        reasons.append(f"target:{chosen[1]}")
        reasons.append(f"target_R:{chosen[2]:.2f}")
        return float(chosen[0]), chosen[1], float(chosen[2]), reasons

    # No structural target >=2R. If there is room and the trend is healthy,
    # use a synthetic 2R target but only if no identified resistance/support
    # sits between entry and 2R.
    target_2r = entry + (2.0 * risk if direction == "bull" else -2.0 * risk)
    target_4r = entry + (4.0 * risk if direction == "bull" else -4.0 * risk)

    obstructed = False
    if direction == "bull":
        obstacles = [t[0] for t in targets if entry < t[0] < target_2r]
    else:
        obstacles = [t[0] for t in targets if target_2r < t[0] < entry]
    if obstacles:
        obstructed = True

    if not obstructed and trend_strength >= 42:
        reasons.append("synthetic_2R_clear_path")
        return float(target_2r), "2R_clear_path", 2.0, reasons

    # A weak trend with a close structural target cannot honestly support a
    # 2R target. Return a capped 1.99R result so the caller can mark the
    # setup low-confidence; main.py will reject it before execution.
    reasons.append("no_valid_2R_target")
    return float(target_2r), "2R_unproven", 2.0, reasons


# ============================================================
# CONFIDENCE ENGINE
# ============================================================

def _confidence(
    direction: str,
    trend: dict,
    candidate: EntryCandidate,
    sl_reason: dict,
    tp_label: str,
    rr: float,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    d1: Optional[pd.DataFrame],
    choch15: dict,
) -> int:
    """
    0-100 QUALITY SCORE.

    This is intentionally not a mathematical probability. It is a
    deterministic evidence score. 100 is reserved for rare setups where
    multiple independent evidence buckets agree.
    """
    score = 20.0
    reasons = candidate.evidence

    # Regime / directional alignment.
    if direction == "bull":
        if trend["h1"] == "bullish":
            score += 16
        elif trend["h1"] == "ranging":
            score += 6
        if trend["d1"] == "bullish":
            score += 10
        elif trend["d1"] == "bearish":
            score -= 12
        score += 0.12 * trend["bull_strength"]
    else:
        if trend["h1"] == "bearish":
            score += 16
        elif trend["h1"] == "ranging":
            score += 6
        if trend["d1"] == "bearish":
            score += 10
        elif trend["d1"] == "bullish":
            score -= 12
        score += 0.12 * trend["bear_strength"]

    # Location quality.
    score += 0.25 * candidate.location_score

    # Sweep + CHoCH is the strongest reversal evidence.
    if direction == "bull" and candidate.sweep.get("bullish_sweep"):
        score += 10
    if direction == "bear" and candidate.sweep.get("bearish_sweep"):
        score += 10

    if direction == "bull" and choch15.get("bullish_choch"):
        score += 10
    if direction == "bear" and choch15.get("bearish_choch"):
        score += 10

    # Trend strength has a direct but bounded influence.
    score += 0.15 * candidate.trend_strength

    # RSI.
    rsi_now = float(m15["rsi"].iloc[-1])
    rsi_slope = float(m15["rsi"].iloc[-1] - m15["rsi"].iloc[-4])
    if direction == "bull":
        if rsi_slope > 0 and 40 <= rsi_now <= 72:
            score += 6
        if candidate.rsi_div.get("bullish"):
            score += 6
    else:
        if rsi_slope < 0 and 28 <= rsi_now <= 60:
            score += 6
        if candidate.rsi_div.get("bearish"):
            score += 6

    # Volume.
    vol_ratio = float(m15["vol_ratio"].iloc[-5:].mean())
    if 1.0 <= vol_ratio <= 2.5:
        score += 5
    elif vol_ratio > 3.0:
        # Extremely high volume can be climax; do not automatically reward it.
        score += 2

    # TP feasibility.
    if 2.0 <= rr <= 2.6:
        score += 4
    elif 2.6 < rr <= 3.5:
        score += 6
    elif rr > 3.5:
        score += 4

    if tp_label == "2R_unproven":
        score -= 15

    # A countertrend setup can still exist, but it should not masquerade
    # as premium quality.
    if "H1_countertrend" in reasons:
        score -= 10

    return int(round(_clip(score)))


# ============================================================
# PUBLIC ANALYSIS API
# ============================================================

def get_best_signal(
    df_h1: pd.DataFrame,
    df_m15: pd.DataFrame,
    df_d1: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
) -> Optional[dict]:
    return full_analyze(df_h1, df_m15, df_d1, symbol=symbol)


def full_analyze(
    df_h1: pd.DataFrame,
    df_m15: pd.DataFrame,
    df_d1: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
) -> Optional[dict]:
    """
    MAIN CONTRACT:

        full_analyze(df_h1, df_m15, df_d1, symbol=sym)

    Returns a signal dict compatible with current main.py.

    Analysis order:
        1. Build numerical data.
        2. Determine HTF direction / trend strength.
        3. Find POI / liquidity / entry zone.
        4. Determine Entry LIMIT.
        5. Determine SL invalidation.
        6. Determine TP and RR.
        7. Produce confidence and evidence.
    """
    h1b = build_df(df_h1, interval_minutes=60)
    m15b = build_df(df_m15, interval_minutes=15)
    d1b = build_df(df_d1, interval_minutes=1440) if df_d1 is not None else None

    if h1b is None or m15b is None or len(m15b) < 60 or len(h1b) < 60:
        return None

    symbol = symbol or "UNKNOWN"
    price = float(m15b["close"].iloc[-1])

    trend = _trend_snapshot(h1b, m15b, d1b)

    # Direction selection is evidence based, not a single indicator.
    bull = trend["bull_strength"]
    bear = trend["bear_strength"]

    # Recent M15 structure shift can override a stale H1 structure only when
    # the shift is confirmed by close and location.
    choch15 = detect_choch(m15b, lb=3)

    if choch15.get("bullish_choch") and bull >= bear - 12:
        direction = "bull"
    elif choch15.get("bearish_choch") and bear >= bull - 12:
        direction = "bear"
    elif bull >= bear:
        direction = "bull"
    else:
        direction = "bear"

    # Entry candidate stage.
    candidates = _collect_entry_candidates(
        h1b, m15b, d1b, preferred_direction=direction
    )

    # If preferred side has no zone, inspect the opposite side as a
    # legitimate low-confidence setup. This keeps every analyzed coin
    # diagnosable rather than returning an unexplained "no setup".
    if not candidates:
        opposite = "bear" if direction == "bull" else "bull"
        alt = _collect_entry_candidates(h1b, m15b, d1b, opposite)
        if alt:
            candidates = alt
            direction = opposite

    # Final fallback: a structural pullback zone from the latest protected
    # swing. This is deliberately low-confidence.
    if not candidates:
        sh, sl = _raw_swing_pts(m15b, lb=3)
        a = max(float(m15b["atr"].iloc[-1]), 1e-12)

        if direction == "bull" and sl:
            lo = float(m15b["low"].iloc[sl[-1]])
            hi = min(price, lo + 0.75 * a)
            if hi > lo:
                c = EntryCandidate(
                    direction="bull",
                    entry=lo + 0.35 * (hi - lo),
                    zone_lo=lo,
                    zone_hi=hi,
                    label="structural_pullback",
                    location_score=30,
                    evidence=["fallback_structure_only"],
                    sweep=detect_liquidity_sweep(m15b),
                    rsi_div=detect_rsi_divergence(m15b),
                    trend_strength=trend["bull_strength"],
                    atr=a,
                    raw_score=30,
                )
                candidates = [c]

        elif direction == "bear" and sh:
            hi = float(m15b["high"].iloc[sh[-1]])
            lo = max(price, hi - 0.75 * a)
            if hi > lo:
                c = EntryCandidate(
                    direction="bear",
                    entry=hi - 0.35 * (hi - lo),
                    zone_lo=lo,
                    zone_hi=hi,
                    label="structural_pullback",
                    location_score=30,
                    evidence=["fallback_structure_only"],
                    sweep=detect_liquidity_sweep(m15b),
                    rsi_div=detect_rsi_divergence(m15b),
                    trend_strength=trend["bear_strength"],
                    atr=a,
                    raw_score=30,
                )
                candidates = [c]

    if not candidates:
        return None

    # ========================================================
    # 1) ENTRY FIRST
    # ========================================================
    candidate = max(
        candidates,
        key=lambda c: (
            c.location_score,
            c.trend_strength,
            1 if "liquidity_sweep" in c.evidence else 0,
        ),
    )

    entry = float(candidate.entry)

    # ========================================================
    # 2) SL SECOND
    # ========================================================
    sl, sl_meta = _compute_sl(
        entry=entry,
        direction=candidate.direction,
        candidate=candidate,
        m15=m15b,
        h1=h1b,
    )

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    # ========================================================
    # 3) TP THIRD
    # ========================================================
    trend_strength = candidate.trend_strength

    tp, tp_label, rr, tp_reasons = _choose_tp_after_sl(
        entry=entry,
        sl=sl,
        direction=candidate.direction,
        m15=m15b,
        h1=h1b,
        trend_strength=trend_strength,
    )

    # Cap RR mechanically.
    if candidate.direction == "bull":
        tp = min(tp, entry + MAX_RR * risk)
    else:
        tp = max(tp, entry - MAX_RR * risk)

    rr = abs(tp - entry) / risk

    # If target still falls short, confidence is explicitly downgraded.
    # main.py will reject it at the pre-entry RR gate; we do not fabricate
    # a "great" signal to force execution.
    if rr < MIN_RR:
        rr_label = "RR_unproven"
    else:
        rr_label = tp_label

    # ========================================================
    # 4) CONFIDENCE
    # ========================================================
    confidence = _confidence(
        direction=candidate.direction,
        trend=trend,
        candidate=candidate,
        sl_reason=sl_meta,
        tp_label=rr_label,
        rr=rr,
        m15=m15b,
        h1=h1b,
        d1=d1b,
        choch15=choch15,
    )

    if rr < MIN_RR:
        confidence = min(confidence, 55)

    if candidate.direction == "bull":
        decision = "BUY"
        original_dir = "bull"
    else:
        decision = "SELL"
        original_dir = "bear"

    # Human-readable evidence; every claim is tied to computed features.
    evidence = list(dict.fromkeys(candidate.evidence + tp_reasons))
    tp_sl_reason = (
        f"Entry {candidate.label}; "
        f"SL={sl_meta['reason']}; "
        f"TP={tp_label}; "
        f"TrendStrength={candidate.trend_strength:.0f}; "
        f"RSI={float(m15b['rsi'].iloc[-1]):.1f}; "
        f"VolRatio={float(m15b['vol_ratio'].iloc[-5:].mean()):.2f}; "
        f"Evidence={','.join(evidence[:12])}"
    )

    # Location score displayed to main.py.
    loc = int(round(_clip(candidate.location_score)))
    loc_label = (
        "EXCEPTIONAL" if loc >= 85
        else "GOOD" if loc >= 70
        else "ACCEPTABLE" if loc >= 50
        else "WEAK"
    )

    return {
        "symbol": symbol,
        "decision": decision,
        "original_dir": original_dir,
        "price": price,
        "entry": entry,
        "sl": float(sl),
        "tp": float(tp),
        "rr": round(rr, 2),
        "confidence": int(confidence),
        "atr": float(m15b["atr"].iloc[-1]),
        "rsi": round(float(m15b["rsi"].iloc[-1]), 2),
        "rsi_slope": round(float(m15b["rsi"].iloc[-1] - m15b["rsi"].iloc[-4]), 2),
        "volume_ratio": round(float(m15b["vol_ratio"].iloc[-5:].mean()), 2),
        "trend_strength": int(round(candidate.trend_strength)),
        "trend_regime": (
            "VERY_STRONG" if candidate.trend_strength >= 85
            else "STRONG" if candidate.trend_strength >= 70
            else "HEALTHY" if candidate.trend_strength >= 55
            else "TRANSITION" if candidate.trend_strength >= 40
            else "WEAK"
        ),
        "entry_label": candidate.label,
        "location_score": loc,
        "location_label": loc_label,
        "tp_label": tp_label,
        "tp_sl_reason": tp_sl_reason,
        "struct_h1": trend["h1"],
        "struct_m15": trend["m15"],
        "d1_bias": (
            "bullish" if trend["d1"] == "bullish"
            else "bearish" if trend["d1"] == "bearish"
            else "neutral"
        ),
        "choch_m15": choch15,
        "choch_h1": detect_choch(h1b, lb=3),
        "failed_retest": {
            "failed_retest_buy": False,
            "failed_retest_sell": False,
        },
        "liquidity_sweep": candidate.sweep,
        "rsi_divergence": candidate.rsi_div,
        "evidence": evidence,
        "zone": {
            "lo": candidate.zone_lo,
            "hi": candidate.zone_hi,
            "type": candidate.label,
        },
        "sl_meta": sl_meta,
        "analysis_version": "strategy_logic_terbaik_2026-08-16",
    }


# ============================================================
# OPTIONAL TRAILING INTERFACE
# ============================================================

def strategy_trailing_stop(
    df_m15: pd.DataFrame,
    entry: float,
    current_sl: float,
    direction: str,
    risk: Optional[float] = None,
    current_price: Optional[float] = None,
    tp: Optional[float] = None,
    position: Optional[dict] = None,
) -> dict:
    """
    Optional public interface for future main.py.

    Current main.py does not call this function yet. It exists so the
    execution engine can later delegate ALL trail decisions here.

    The principle is:
        - no R-profit locking;
        - do not move SL merely because price is profitable;
        - a strong trend gets more breathing room;
        - a weakening trend selects a newer protected swing;
        - SL is a thesis-invalidation point.

    No API call is made here.
    """
    d = build_df(df_m15, interval_minutes=15)
    if d is None:
        return {
            "candidate": None,
            "strength": 0.0,
            "regime": "UNKNOWN",
            "reason": "insufficient_data",
        }

    is_buy = direction.upper() == "BUY"
    dir_key = "bull" if is_buy else "bear"

    strength = _price_time_trend_strength(d, dir_key)
    rsi_div = detect_rsi_divergence(d)

    sh, sl = swing_pts(d, lb=STRUCT_TRAIL_LB)

    candidate = None
    reason_parts = []

    if is_buy and sl:
        idx = sl[-1]
        a = max(float(d["atr"].iloc[idx]), 1e-12)
        candidate = float(d["low"].iloc[idx]) - SL_ATR_BUFFER * a
        reason_parts.append("protected_HL")
    elif not is_buy and sh:
        idx = sh[-1]
        a = max(float(d["atr"].iloc[idx]), 1e-12)
        candidate = float(d["high"].iloc[idx]) + SL_ATR_BUFFER * a
        reason_parts.append("protected_LH")

    # RSI divergence reinforces decay, but never creates an exit alone.
    if is_buy and rsi_div.get("bearish"):
        reason_parts.append("bearish_RSI_decay")
    if not is_buy and rsi_div.get("bullish"):
        reason_parts.append("bullish_RSI_decay")

    # Final geometry: a new SL must remain outside current price and improve
    # the existing SL. A structurally "better" price that sits inside current
    # market would cause an immediate/invalid stop.
    if candidate is not None:
        px = float(current_price) if current_price is not None else float(d["close"].iloc[-1])
        side_ok = (candidate < px) if is_buy else (candidate > px)
        improves = candidate > current_sl if is_buy else candidate < current_sl
        if not side_ok or not improves:
            candidate = None

    regime = (
        "VERY_STRONG" if strength >= 85
        else "STRONG" if strength >= 70
        else "HEALTHY" if strength >= 55
        else "TRANSITION" if strength >= 40
        else "WEAK"
    )

    return {
        "candidate": candidate,
        "strength": round(strength, 2),
        "regime": regime,
        "reason": "+".join(reason_parts) if reason_parts else "no_new_invalidation",
    }


# ============================================================
# END
# ============================================================
