from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

import learn

MIN_RR = 2.0
MAX_RR = None
TRAIL_R_LADDER = []
STRUCT_TRAIL_LB = 3
STRUCT_TRAIL_BUF_PCT = 0.0025
STRUCT_TRAIL_LOOKBACK = 60
FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618
STRATEGY_VERSION = "S1.0"


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _clip(value, low, high):
    return max(low, min(high, _safe_float(value, low)))


def ema(series, period):
    return series.astype(float).ewm(span=period, adjust=False).mean()


def atr_fn(df, period=14):
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def build_df(df, min_rows=60):
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < min_rows:
        return None

    out = df.copy()

    for column in ("open", "high", "low", "close", "volume"):
        if column not in out.columns:
            return None
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    if len(out) < min_rows:
        return None

    out["ema9"] = ema(out["close"], 9)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["atr"] = atr_fn(out, 14)
    out["vol_sma"] = out["volume"].rolling(20).mean()
    return out.dropna().copy()


def swing_pts(df, lb=5):
    if df is None or len(df) < 2 * lb + 1:
        return [], []

    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    swing_highs = []
    swing_lows = []

    for i in range(lb, len(df) - lb):
        if highs[i] >= np.max(highs[i - lb:i + lb + 1]):
            swing_highs.append(i)
        if lows[i] <= np.min(lows[i - lb:i + lb + 1]):
            swing_lows.append(i)

    return swing_highs, swing_lows


def mkt_struct(df, sh, sl):
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


def fib_position(price, swing_low, swing_high):
    span = swing_high - swing_low
    if span <= 0:
        return 0.5
    return _clip((price - swing_low) / span, 0.0, 1.0)


def trend_strength(df, sh, sl):
    if df is None or len(df) < 30:
        return 0.0

    atr = _safe_float(df["atr"].iloc[-1])
    if atr <= 0:
        return 0.0

    structure = mkt_struct(df, sh, sl)
    score = 35.0

    if structure in {"bullish", "bearish"}:
        score += 20.0
    else:
        score -= 10.0

    if structure == "bullish" and len(sh) >= 3:
        points = [(i, df["high"].iloc[i]) for i in sh[-3:]]
    elif structure == "bearish" and len(sl) >= 3:
        points = [(i, df["low"].iloc[i]) for i in sl[-3:]]
    else:
        points = []

    slopes = []
    for (i1, p1), (i2, p2) in zip(points[:-1], points[1:]):
        slopes.append(abs(float(p2) - float(p1)) / max(1, i2 - i1) / atr)

    if slopes:
        score += _clip(float(np.mean(slopes)) * 900.0, 0.0, 28.0)
        if len(slopes) >= 2:
            if slopes[-1] > slopes[-2] * 1.10:
                score += 7.0
            elif slopes[-1] < slopes[-2] * 0.75:
                score -= 7.0

    last = df.iloc[-1]
    if structure == "bullish" and last["ema9"] > last["ema20"] > last["ema50"]:
        score += 8
    elif structure == "bearish" and last["ema9"] < last["ema20"] < last["ema50"]:
        score += 8

    relative_volume = _safe_float(last["volume"]) / max(_safe_float(last["vol_sma"], 1.0), 1e-12)
    score += _clip((relative_volume - 1.0) * 6.0, -6.0, 6.0)
    return _clip(score, 0.0, 100.0)


def detect_bos(df, sh, sl):
    result = {"bullish": False, "bearish": False, "level": None}
    if not sh or not sl or len(df) < 3:
        return result

    close = _safe_float(df["close"].iloc[-1])
    prev_close = _safe_float(df["close"].iloc[-2])
    high_level = _safe_float(df["high"].iloc[sh[-1]])
    low_level = _safe_float(df["low"].iloc[sl[-1]])

    result["bullish"] = close > high_level and prev_close <= high_level
    result["bearish"] = close < low_level and prev_close >= low_level
    result["level"] = high_level if result["bullish"] else low_level if result["bearish"] else None
    return result


def detect_choch(df, sh, sl):
    result = {"bullish": False, "bearish": False}
    if len(sh) < 2 or len(sl) < 2:
        return result

    structure = mkt_struct(df, sh, sl)
    close = _safe_float(df["close"].iloc[-1])
    last_high = _safe_float(df["high"].iloc[sh[-1]])
    last_low = _safe_float(df["low"].iloc[sl[-1]])

    if structure == "bearish" and close > last_high:
        result["bullish"] = True
    if structure == "bullish" and close < last_low:
        result["bearish"] = True
    return result


def detect_liquidity_sweep(df, sh, sl, direction):
    if direction == "bull" and sl:
        level = _safe_float(df["low"].iloc[sl[-1]])
        low = _safe_float(df["low"].iloc[-1])
        close = _safe_float(df["close"].iloc[-1])
        if low < level and close > level:
            return {"type": "sellside_sweep", "level": level}

    if direction == "bear" and sh:
        level = _safe_float(df["high"].iloc[sh[-1]])
        high = _safe_float(df["high"].iloc[-1])
        close = _safe_float(df["close"].iloc[-1])
        if high > level and close < level:
            return {"type": "buyside_sweep", "level": level}

    return {"type": "none", "level": None}


def detect_fvg(df, direction, lookback=60):
    if df is None or len(df) < 5:
        return []

    start = max(0, len(df) - lookback)
    result = []

    for i in range(start, len(df) - 2):
        first = df.iloc[i]
        third = df.iloc[i + 2]

        if direction == "bull" and third["low"] > first["high"]:
            result.append({
                "top": float(third["low"]),
                "bottom": float(first["high"]),
                "mid": float((third["low"] + first["high"]) / 2),
                "index": i + 2,
            })

        elif direction == "bear" and third["high"] < first["low"]:
            result.append({
                "top": float(first["low"]),
                "bottom": float(third["high"]),
                "mid": float((first["low"] + third["high"]) / 2),
                "index": i + 2,
            })

    return result[-5:]


def detect_order_blocks(df, direction, lookback=80):
    if df is None or len(df) < 20:
        return []

    start = max(1, len(df) - lookback)
    body_average = _safe_float((df["close"] - df["open"]).abs().iloc[start:].mean())
    result = []

    for i in range(start, len(df) - 1):
        candle = df.iloc[i]
        next_candle = df.iloc[i + 1]
        impulse = abs(float(next_candle["close"] - next_candle["open"]))

        if body_average <= 0 or impulse < body_average * 1.2:
            continue

        if direction == "bull" and candle["close"] < candle["open"] and next_candle["close"] > next_candle["open"]:
            top = max(float(candle["open"]), float(candle["close"]))
            bottom = min(float(candle["open"]), float(candle["close"]))
        elif direction == "bear" and candle["close"] > candle["open"] and next_candle["close"] < next_candle["open"]:
            top = max(float(candle["open"]), float(candle["close"]))
            bottom = min(float(candle["open"]), float(candle["close"]))
        else:
            continue

        result.append({
            "top": top,
            "bottom": bottom,
            "mid": (top + bottom) / 2,
            "index": i,
            "quality": _clip(50.0 + (impulse / body_average) * 10.0, 0.0, 100.0),
        })

    return sorted(result, key=lambda x: (x["quality"], x["index"]), reverse=True)[:5]


def _find_zone(m15, direction):
    fvgs = detect_fvg(m15, direction)
    obs = detect_order_blocks(m15, direction)
    candidates = []

    for zone in fvgs:
        candidates.append((zone.get("quality", 65.0), zone))
    for zone in obs:
        candidates.append((zone.get("quality", 55.0), zone))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]["index"]), reverse=True)
    return candidates[0][1]


def _direction_from_frames(h1, m15, d1=None):
    frames = [h1, m15]
    if d1 is not None:
        frames.append(d1)

    directions = []
    for frame in frames:
        if frame is None or len(frame) < 25:
            continue
        sh, sl = swing_pts(frame, 5)
        structure = mkt_struct(frame, sh, sl)
        last = frame.iloc[-1]
        if structure == "bullish" or last["ema9"] > last["ema20"]:
            directions.append("bull")
        elif structure == "bearish" or last["ema9"] < last["ema20"]:
            directions.append("bear")
        else:
            directions.append("neutral")

    if directions.count("bull") >= 2 and directions.count("bull") > directions.count("bear"):
        return "bull"
    if directions.count("bear") >= 2 and directions.count("bear") > directions.count("bull"):
        return "bear"
    return "neutral"


def score_direction(df_h1, df_m15, df_d1=None):
    h1 = build_df(df_h1)
    m15 = build_df(df_m15)
    d1 = build_df(df_d1) if df_d1 is not None else None

    bull = 0.0
    bear = 0.0

    for frame, weight in ((h1, 40), (m15, 35), (d1, 25)):
        if frame is None:
            continue
        sh, sl = swing_pts(frame, 5)
        structure = mkt_struct(frame, sh, sl)
        last = frame.iloc[-1]
        if structure == "bullish" or last["ema9"] > last["ema20"]:
            bull += weight
        if structure == "bearish" or last["ema9"] < last["ema20"]:
            bear += weight

    direction = "bull" if bull > bear else "bear" if bear > bull else "neutral"
    total = max(bull + bear, 1.0)
    return {
        "direction": direction,
        "confidence": _clip(max(bull, bear) / total * 100.0, 0, 100),
        "bull_score": bull,
        "bear_score": bear,
    }


def _build_candidate(h1, m15, d1, direction):
    if h1 is None or m15 is None:
        return None

    sh15, sl15 = swing_pts(m15, 5)
    if not sh15 or not sl15:
        return None

    current = _safe_float(m15["close"].iloc[-1])
    atr = _safe_float(m15["atr"].iloc[-1])
    if current <= 0 or atr <= 0:
        return None

    h1_sh, h1_sl = swing_pts(h1, 5)
    if not h1_sh or not h1_sl:
        return None

    sweep = detect_liquidity_sweep(m15, sh15, sl15, direction)
    bos = detect_bos(m15, sh15, sl15)
    choch = detect_choch(m15, sh15, sl15)
    zone = _find_zone(m15, direction)

    if zone is None:
        return None

    last_swing_high = _safe_float(m15["high"].iloc[sh15[-1]])
    last_swing_low = _safe_float(m15["low"].iloc[sl15[-1]])

    h1_high = _safe_float(h1["high"].iloc[h1_sh[-1]])
    h1_low = _safe_float(h1["low"].iloc[h1_sl[-1]])
    swing_low = min(last_swing_low, h1_low)
    swing_high = max(last_swing_high, h1_high)

    location = fib_position(zone["mid"], swing_low, swing_high)
    location_good = location <= 0.618 if direction == "bull" else location >= 0.382

    trigger = (
        bos["bullish"] or choch["bullish"] or sweep["type"] == "sellside_sweep"
        if direction == "bull"
        else bos["bearish"] or choch["bearish"] or sweep["type"] == "buyside_sweep"
    )

    if not trigger:
        return None

    if direction == "bull":
        entry = zone["mid"]
        sl = min(last_swing_low, zone["bottom"]) - 0.10 * atr
        risk = entry - sl
        if risk <= 0:
            return None
        targets = [float(h1["high"].iloc[i]) for i in h1_sh if float(h1["high"].iloc[i]) > entry]
        tp = min(targets) if targets else entry + 2.0 * risk
        tp = max(tp, entry + 2.0 * risk)
    else:
        entry = zone["mid"]
        sl = max(last_swing_high, zone["top"]) + 0.10 * atr
        risk = sl - entry
        if risk <= 0:
            return None
        targets = [float(h1["low"].iloc[i]) for i in h1_sl if float(h1["low"].iloc[i]) < entry]
        tp = max(targets) if targets else entry - 2.0 * risk
        tp = min(tp, entry - 2.0 * risk)

    rr = abs(tp - entry) / max(risk, 1e-12)
    if rr < MIN_RR:
        return None

    trend = trend_strength(m15, sh15, sl15)
    htf_alignment = 1.0 if _direction_from_frames(h1, m15, d1) == direction else 0.0
    score = 50.0
    score += trend * 0.25
    score += 12.0 if location_good else -10.0
    score += 12.0 if htf_alignment else 0.0
    score += 10.0 if sweep["type"] != "none" else 0.0
    score += 8.0 if bos["bullish"] or bos["bearish"] else 0.0
    score += 6.0 if choch["bullish"] or choch["bearish"] else 0.0
    confidence = _clip(score, 0, 100)

    return {
        "side": "Buy" if direction == "bull" else "Sell",
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "rr": float(rr),
        "confidence": float(confidence),
        "strategy_version": STRATEGY_VERSION,
        "reasons": [
            "HTF_ALIGNMENT" if htf_alignment else "HTF_MIXED",
            "LIQUIDITY_SWEEP" if sweep["type"] != "none" else "NO_SWEEP",
            "BOS" if bos["bullish"] or bos["bearish"] else "NO_BOS",
            "CHOCH" if choch["bullish"] or choch["bearish"] else "NO_CHOCH",
            "FVG_OR_OB",
            "FIB_LOCATION" if location_good else "WEAK_LOCATION",
        ],
        "trail": {
            "enabled": True,
            "activation_r": 1.0,
            "distance_r": 0.5,
        },
    }



def candles_to_df(candles):
    rows = []
    for candle in candles:
        if len(candle) < 6:
            continue
        rows.append({
            "timestamp": pd.to_datetime(int(candle[0]), unit="ms", utc=True),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


def resample_ohlc(df, rule):
    if df is None or df.empty:
        return None
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def strategy_logic(symbol, candles):
    if not candles or len(candles) < 672:
        print(f"[STRATEGY] {symbol} FAILED | M15={len(candles) if candles else 0}")
        return None

    m15 = candles_to_df(candles[-672:])
    if m15 is None or len(m15) < 672:
        print(f"[STRATEGY] {symbol} FAILED | invalid M15 data")
        return None

    h1 = resample_ohlc(m15, "1h")
    h4 = resample_ohlc(m15, "4h")
    d1 = resample_ohlc(m15, "1D")

    result = full_analyze(
        h1,
        m15,
        d1,
        symbol=symbol,
        h4=h4,
    )

    if result and result.get("execution_eligible"):
        candidate = result.get("candidate") or {}
        print(
            f"[STRATEGY] {symbol} SIGNAL | "
            f"{candidate.get('side')} | "
            f"Entry={candidate.get('entry')} | "
            f"TP={candidate.get('tp')} | "
            f"SL={candidate.get('sl')}"
        )
    else:
        reason = result.get("reason", "NO_SIGNAL") if result else "NO_SIGNAL"
        print(f"[STRATEGY] {symbol} WAIT | {reason}")

    return result

def full_analyze(df_h1, df_m15, df_d1=None, symbol=None, **kwargs):
    """Deterministic strategy decision. Learning can adjust admission, not geometry."""
    h1 = build_df(df_h1)
    m15 = build_df(df_m15)
    d1 = build_df(df_d1) if df_d1 is not None else None

    if h1 is None or m15 is None:
        return {
            "symbol": symbol,
            "decision": "WAIT",
            "no_signal": True,
            "execution_eligible": False,
            "reason": "INSUFFICIENT_DATA",
            "strategy_version": STRATEGY_VERSION,
        }

    score = score_direction(h1, m15, d1)
    direction = score["direction"]
    if direction == "neutral":
        return {
            "symbol": symbol,
            "decision": "WAIT",
            "no_signal": True,
            "execution_eligible": False,
            "reason": "NO_CLEAR_DIRECTION",
            "score": score,
            "strategy_version": STRATEGY_VERSION,
        }

    candidate = _build_candidate(h1, m15, d1, direction)
    if candidate is None:
        return {
            "symbol": symbol,
            "decision": "WAIT",
            "no_signal": True,
            "execution_eligible": False,
            "reason": "NO_VALID_SETUP",
            "score": score,
            "strategy_version": STRATEGY_VERSION,
        }

    learning = learn.evaluate_candidate(symbol, candidate)
    threshold = learning.get("threshold", learn.get_confidence_threshold())
    eligible = candidate["confidence"] >= threshold

    packet = {
        "symbol": symbol,
        "decision": "READY" if eligible else "WAIT",
        "no_signal": not eligible,
        "execution_eligible": eligible,
        "confidence": candidate["confidence"],
        "confidence_threshold": threshold,
        "score": score,
        "candidate": candidate,
        "learning": learning,
        "strategy_version": STRATEGY_VERSION,
    }

    learn.record_candidate(packet)
    return packet


def manage_position(state, df_m15, df_h1=None, df_d1=None, symbol=None, **kwargs):
    """Return a trailing/protection recommendation; does not execute orders."""
    m15 = build_df(df_m15)
    if m15 is None or not state:
        return {"action": "HOLD"}

    direction = str(state.get("side") or state.get("direction") or "").lower()
    entry = _safe_float(state.get("entry"))
    current = _safe_float(m15["close"].iloc[-1])
    risk = abs(entry - _safe_float(state.get("sl")))

    if entry <= 0 or risk <= 0:
        return {"action": "HOLD"}

    profit_r = ((current - entry) / risk) if direction in {"buy", "bull"} else ((entry - current) / risk)

    if profit_r >= 1.0:
        return {
            "action": "TRAIL",
            "enabled": True,
            "distance": risk * 0.5,
            "profit_r": round(profit_r, 3),
        }

    return {"action": "HOLD", "profit_r": round(profit_r, 3)}
