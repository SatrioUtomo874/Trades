"""
strategy.py — SMC/ICT market-analysis and setup-generation engine.

Receives candle data only from main.py. Never touches Bybit/Binance/Telegram.

Public interface:
    analyze(candles, context) -> decision dict | None
    update_position(market_data, position_context) -> trail dict | None
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Optional

STRATEGY_VERSION = "1.0.0"

MIN_RR = 1.3
DISPLACEMENT_ATR_MULT = 1.5
SWEEP_TOLERANCE_ATR = 0.15
EQUAL_LEVEL_TOLERANCE_ATR = 0.12


# ----------------------------------------------------------------------
# candle helpers
# ----------------------------------------------------------------------

def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _hi(c: dict) -> float:
    return _f(c.get("high"))


def _lo(c: dict) -> float:
    return _f(c.get("low"))


def _cl(c: dict) -> float:
    return _f(c.get("close"))


def _op(c: dict) -> float:
    return _f(c.get("open"))


def true_range(prev: dict, cur: dict) -> float:
    return max(
        _hi(cur) - _lo(cur),
        abs(_hi(cur) - _cl(prev)),
        abs(_lo(cur) - _cl(prev)),
    )


def atr(candles: list[dict], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = [true_range(candles[i - 1], candles[i]) for i in range(len(candles) - period, len(candles))]
    return sum(trs) / len(trs) if trs else 0.0


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


# ----------------------------------------------------------------------
# structure: swings, BOS/CHoCH, FVG, order blocks, sweeps
# ----------------------------------------------------------------------

def find_swings(candles: list[dict], left: int = 3, right: int = 3) -> list[dict]:
    """Pivot highs/lows. Returns list of {index, type('high'|'low'), price}."""
    swings = []
    n = len(candles)
    for i in range(left, n - right):
        window = candles[i - left:i + right + 1]
        h = _hi(candles[i])
        l = _lo(candles[i])
        if h == max(_hi(c) for c in window):
            swings.append({"index": i, "type": "high", "price": h})
        if l == min(_lo(c) for c in window):
            swings.append({"index": i, "type": "low", "price": l})
    return swings


def detect_bos_choch(candles: list[dict], swings: list[dict]) -> dict:
    """Detect most recent BOS and whether it represents a CHoCH (trend flip)."""
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    if not highs or not lows:
        return {"bias": "neutral", "last_bos": None, "choch": False}

    events = []  # chronological list of ('bull'|'bear', index)
    last_swing_high = None
    last_swing_low = None
    for i, c in enumerate(candles):
        for s in highs:
            if s["index"] == i:
                last_swing_high = s["price"]
        for s in lows:
            if s["index"] == i:
                last_swing_low = s["price"]
        if last_swing_high is not None and _cl(c) > last_swing_high and i > 0:
            events.append(("bull", i))
            last_swing_high = None  # consumed
        if last_swing_low is not None and _cl(c) < last_swing_low and i > 0:
            events.append(("bear", i))
            last_swing_low = None

    if not events:
        return {"bias": "neutral", "last_bos": None, "choch": False}

    last_dir = events[-1][0]
    choch = len(events) >= 2 and events[-2][0] != last_dir
    bias = "bullish" if last_dir == "bull" else "bearish"
    return {"bias": bias, "last_bos": events[-1], "choch": choch, "event_count": len(events)}


def detect_fvg(candles: list[dict], lookback: int = 40) -> list[dict]:
    """3-candle imbalance (fair value gap). Returns recent unmitigated gaps."""
    gaps = []
    start = max(2, len(candles) - lookback)
    for i in range(start, len(candles)):
        c1, c3 = candles[i - 2], candles[i]
        if _hi(c1) < _lo(c3):
            gaps.append({"index": i, "type": "bullish", "top": _lo(c3), "bottom": _hi(c1)})
        elif _lo(c1) > _hi(c3):
            gaps.append({"index": i, "type": "bearish", "top": _lo(c1), "bottom": _hi(c3)})

    # mitigation: drop gaps whose range has since been fully traded through
    live = []
    for g in gaps:
        mitigated = False
        for c in candles[g["index"] + 1:]:
            if _lo(c) <= g["bottom"] and _hi(c) >= g["top"]:
                mitigated = True
                break
        if not mitigated:
            live.append(g)
    return live


def detect_order_blocks(candles: list[dict], atr_value: float, lookback: int = 60) -> list[dict]:
    """Last opposite-colored candle preceding a displacement move."""
    blocks = []
    start = max(1, len(candles) - lookback)
    for i in range(start, len(candles)):
        body = _cl(candles[i]) - _op(candles[i])
        if atr_value <= 0 or abs(body) < DISPLACEMENT_ATR_MULT * atr_value:
            continue
        prev = candles[i - 1]
        prev_bear = _cl(prev) < _op(prev)
        prev_bull = _cl(prev) > _op(prev)
        if body > 0 and prev_bear:
            blocks.append({"index": i - 1, "type": "bullish", "top": _hi(prev), "bottom": _lo(prev)})
        elif body < 0 and prev_bull:
            blocks.append({"index": i - 1, "type": "bearish", "top": _hi(prev), "bottom": _lo(prev)})
    return blocks[-6:]


def detect_liquidity_sweep(candles: list[dict], swings: list[dict], atr_value: float) -> Optional[dict]:
    """Wick beyond a recent swing extreme that closes back inside range."""
    if len(candles) < 5 or atr_value <= 0:
        return None
    last = candles[-1]
    recent_highs = [s["price"] for s in swings if s["type"] == "high" and s["index"] < len(candles) - 1]
    recent_lows = [s["price"] for s in swings if s["type"] == "low" and s["index"] < len(candles) - 1]
    tol = SWEEP_TOLERANCE_ATR * atr_value

    if recent_highs:
        level = max(recent_highs[-3:])
        if _hi(last) > level + tol * 0 and _hi(last) >= level and _cl(last) < level:
            return {"type": "sell_side_sweep", "level": level, "wick": _hi(last)}
    if recent_lows:
        level = min(recent_lows[-3:])
        if _lo(last) <= level and _cl(last) > level:
            return {"type": "buy_side_sweep", "level": level, "wick": _lo(last)}
    return None


def detect_equal_levels(swings: list[dict], atr_value: float) -> dict:
    highs = sorted([s["price"] for s in swings if s["type"] == "high"])
    lows = sorted([s["price"] for s in swings if s["type"] == "low"])
    tol = max(EQUAL_LEVEL_TOLERANCE_ATR * atr_value, 1e-9)

    def cluster(vals: list[float]) -> bool:
        for i in range(len(vals) - 1):
            if abs(vals[i + 1] - vals[i]) <= tol:
                return True
        return False

    return {"equal_highs": cluster(highs), "equal_lows": cluster(lows)}


def premium_discount(candles: list[dict], lookback: int = 60) -> float:
    """0 = bottom of range (discount), 1 = top of range (premium)."""
    window = candles[-lookback:]
    hi = max(_hi(c) for c in window)
    lo = min(_lo(c) for c in window)
    if hi == lo:
        return 0.5
    return (_cl(candles[-1]) - lo) / (hi - lo)


def market_regime(candles: list[dict]) -> str:
    closes = [_cl(c) for c in candles]
    if len(closes) < 50:
        return "unknown"
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    slope = ema20[-1] - ema20[-10] if len(ema20) > 10 else 0.0
    spread = (ema20[-1] - ema50[-1]) / ema50[-1] if ema50[-1] else 0.0
    if abs(spread) < 0.002 and abs(slope) < (ema20[-1] * 0.001):
        return "ranging"
    return "bullish_trend" if ema20[-1] > ema50[-1] else "bearish_trend"


def session_of(timestamp_ms: int) -> str:
    hour = time.gmtime(timestamp_ms / 1000).tm_hour
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "new_york"
    return "late_ny"


def btc_correlation(candles: list[dict], btc_candles: Optional[list[dict]], n: int = 60) -> float:
    if not btc_candles or len(candles) < n + 1 or len(btc_candles) < n + 1:
        return 0.0
    a = [_cl(c) for c in candles[-n - 1:]]
    b = [_cl(c) for c in btc_candles[-n - 1:]]
    ra = [(a[i] - a[i - 1]) / a[i - 1] for i in range(1, len(a)) if a[i - 1]]
    rb = [(b[i] - b[i - 1]) / b[i - 1] for i in range(1, len(b)) if b[i - 1]]
    m = min(len(ra), len(rb))
    if m < 5:
        return 0.0
    ra, rb = ra[-m:], rb[-m:]
    try:
        return statistics.correlation(ra, rb)
    except statistics.StatisticsError:
        return 0.0


# ----------------------------------------------------------------------
# confidence scoring
# ----------------------------------------------------------------------

def _score_structure(structure: dict, choch: bool) -> float:
    if structure["bias"] == "neutral":
        return 20.0
    score = 55.0
    if choch:
        score += 15.0
    ev = structure.get("event_count", 0)
    score += min(ev, 5) * 2.0
    return min(score, 100.0)


def _score_liquidity(sweep: Optional[dict], equal_levels: dict) -> float:
    score = 30.0
    if sweep:
        score += 35.0
    if equal_levels.get("equal_highs") or equal_levels.get("equal_lows"):
        score += 15.0
    return min(score, 100.0)


def _score_momentum(candles: list[dict], atr_value: float) -> float:
    if len(candles) < 6 or atr_value <= 0:
        return 40.0
    body = abs(_cl(candles[-1]) - _op(candles[-1]))
    ratio = body / atr_value
    return min(40.0 + ratio * 30.0, 100.0)


def _score_volatility(candles: list[dict], atr_value: float) -> float:
    if len(candles) < 30 or atr_value <= 0:
        return 50.0
    closes = [_cl(c) for c in candles[-30:]]
    px = closes[-1] or 1.0
    rel = atr_value / px
    if rel < 0.0005:
        return 35.0
    if rel > 0.05:
        return 40.0
    return 75.0


def _score_entry_quality(fvgs: list[dict], order_blocks: list[dict], pd_pos: float, bias: str) -> float:
    score = 35.0
    aligned_ob = [b for b in order_blocks if (bias == "bullish" and b["type"] == "bullish") or
                  (bias == "bearish" and b["type"] == "bearish")]
    aligned_fvg = [g for g in fvgs if (bias == "bullish" and g["type"] == "bullish") or
                   (bias == "bearish" and g["type"] == "bearish")]
    if aligned_ob:
        score += 25.0
    if aligned_fvg:
        score += 20.0
    if bias == "bullish" and pd_pos <= 0.5:
        score += 15.0
    elif bias == "bearish" and pd_pos >= 0.5:
        score += 15.0
    return min(score, 100.0)


def _score_rr(rr: float) -> float:
    return min(max((rr - 1.0) * 40.0, 0.0), 100.0)


def _score_regime(regime: str, bias: str) -> float:
    if regime == "unknown":
        return 45.0
    if regime == "ranging":
        return 40.0
    aligned = (regime == "bullish_trend" and bias == "bullish") or (regime == "bearish_trend" and bias == "bearish")
    return 80.0 if aligned else 30.0


def _score_btc(correlation: float, bias: str, btc_bias: str) -> float:
    if btc_bias == "neutral":
        return 50.0
    aligned = (bias == "bullish" and btc_bias == "bullish") or (bias == "bearish" and btc_bias == "bearish")
    if abs(correlation) < 0.2:
        return 50.0
    if aligned and correlation > 0:
        return 75.0
    if not aligned and correlation > 0:
        return 30.0
    return 50.0


def _score_session(session: str) -> float:
    return {"london": 70.0, "new_york": 70.0, "asia": 50.0, "late_ny": 45.0}.get(session, 50.0)


def _score_displacement(candles: list[dict], atr_value: float) -> float:
    if len(candles) < 3 or atr_value <= 0:
        return 40.0
    body = abs(_cl(candles[-2]) - _op(candles[-2]))
    return min(40.0 + (body / atr_value) * 35.0, 100.0)


def _score_confirmation(sweep: Optional[dict], choch: bool, aligned_ob: bool) -> float:
    score = 25.0
    if sweep:
        score += 25.0
    if choch:
        score += 25.0
    if aligned_ob:
        score += 25.0
    return score


def _score_contradictions(pd_pos: float, bias: str, regime: str) -> float:
    """Higher score = fewer contradictions."""
    score = 100.0
    if bias == "bullish" and pd_pos > 0.8:
        score -= 30.0
    if bias == "bearish" and pd_pos < 0.2:
        score -= 30.0
    if regime == "bullish_trend" and bias == "bearish":
        score -= 25.0
    if regime == "bearish_trend" and bias == "bullish":
        score -= 25.0
    return max(score, 0.0)


FEATURE_WEIGHTS = {
    "structure": 0.16,
    "liquidity": 0.12,
    "momentum": 0.08,
    "volatility": 0.05,
    "entry_quality": 0.14,
    "risk_reward": 0.10,
    "market_regime": 0.10,
    "btc_relationship": 0.06,
    "session": 0.04,
    "fvg_quality": 0.05,
    "order_block_quality": 0.05,
    "sweep_quality": 0.05,
    "displacement": 0.04,
    "confirmation": 0.04,
    "contradictions": 0.06,
}


def _weighted_confidence(scores: dict[str, float]) -> float:
    total_w = sum(FEATURE_WEIGHTS.values())
    total = sum(scores.get(k, 50.0) * w for k, w in FEATURE_WEIGHTS.items())
    return round(min(max(total / total_w, 0.0), 100.0), 2)


# ----------------------------------------------------------------------
# public: analyze
# ----------------------------------------------------------------------

def analyze(candles: list[dict], context: dict) -> Optional[dict]:
    """
    candles: ascending-time list of {open, high, low, close, volume, timestamp}
    context: {symbol, btc_candles?, active_threshold?, min_rr?}
    Returns a decision dict or None if no valid setup.
    """
    symbol = context.get("symbol", "UNKNOWN")
    min_rr = context.get("min_rr", MIN_RR)

    if len(candles) < 60:
        return None

    atr_value = atr(candles, 14)
    if atr_value <= 0:
        return None

    swings = find_swings(candles)
    structure = detect_bos_choch(candles, swings)
    bias = structure["bias"]
    if bias == "neutral":
        return None

    fvgs = detect_fvg(candles)
    order_blocks = detect_order_blocks(candles, atr_value)
    sweep = detect_liquidity_sweep(candles, swings, atr_value)
    equal_levels = detect_equal_levels(swings, atr_value)
    pd_pos = premium_discount(candles)
    regime = market_regime(candles)
    last_ts = int(candles[-1].get("timestamp", time.time() * 1000))
    session = session_of(last_ts)

    btc_candles = context.get("btc_candles")
    btc_bias = "neutral"
    if btc_candles and len(btc_candles) >= 60:
        btc_swings = find_swings(btc_candles)
        btc_bias = detect_bos_choch(btc_candles, btc_swings)["bias"]
    correlation = btc_correlation(candles, btc_candles)

    aligned_ob = [b for b in order_blocks if (bias == "bullish" and b["type"] == "bullish") or
                  (bias == "bearish" and b["type"] == "bearish")]
    aligned_fvg = [g for g in fvgs if (bias == "bullish" and g["type"] == "bullish") or
                   (bias == "bearish" and g["type"] == "bearish")]

    # --- entry / SL / TP construction ---
    price = _cl(candles[-1])
    recent_swing_highs = [s["price"] for s in swings if s["type"] == "high"]
    recent_swing_lows = [s["price"] for s in swings if s["type"] == "low"]
    if not recent_swing_highs or not recent_swing_lows:
        return None

    if bias == "bullish":
        zone = aligned_ob[-1] if aligned_ob else (aligned_fvg[-1] if aligned_fvg else None)
        entry = zone["top"] if zone else price
        sl_anchor = min(recent_swing_lows[-2:]) if len(recent_swing_lows) >= 2 else recent_swing_lows[-1]
        stop_loss = min(sl_anchor, zone["bottom"] if zone else sl_anchor) - atr_value * 0.15
        target_pool = max(recent_swing_highs[-3:])
        take_profit = max(target_pool, entry + (entry - stop_loss) * min_rr)
        if not (stop_loss < entry < take_profit):
            return None
    else:
        zone = aligned_ob[-1] if aligned_ob else (aligned_fvg[-1] if aligned_fvg else None)
        entry = zone["bottom"] if zone else price
        sl_anchor = max(recent_swing_highs[-2:]) if len(recent_swing_highs) >= 2 else recent_swing_highs[-1]
        stop_loss = max(sl_anchor, zone["top"] if zone else sl_anchor) + atr_value * 0.15
        target_pool = min(recent_swing_lows[-3:])
        take_profit = min(target_pool, entry - (stop_loss - entry) * min_rr)
        if not (take_profit < entry < stop_loss):
            return None

    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk <= 0:
        return None
    risk_reward = round(reward / risk, 3)
    if risk_reward < min_rr:
        return None

    feature_scores = {
        "structure": _score_structure(structure, structure.get("choch", False)),
        "liquidity": _score_liquidity(sweep, equal_levels),
        "momentum": _score_momentum(candles, atr_value),
        "volatility": _score_volatility(candles, atr_value),
        "entry_quality": _score_entry_quality(fvgs, order_blocks, pd_pos, bias),
        "risk_reward": _score_rr(risk_reward),
        "market_regime": _score_regime(regime, bias),
        "btc_relationship": _score_btc(correlation, bias, btc_bias),
        "session": _score_session(session),
        "fvg_quality": 70.0 if aligned_fvg else 35.0,
        "order_block_quality": 70.0 if aligned_ob else 35.0,
        "sweep_quality": 80.0 if sweep else 30.0,
        "displacement": _score_displacement(candles, atr_value),
        "confirmation": _score_confirmation(sweep, structure.get("choch", False), bool(aligned_ob)),
        "contradictions": _score_contradictions(pd_pos, bias, regime),
    }
    confidence = _weighted_confidence(feature_scores)

    reason_codes = []
    if structure.get("choch"):
        reason_codes.append("choch_confirmed")
    else:
        reason_codes.append("bos_continuation")
    if sweep:
        reason_codes.append(f"liquidity_{sweep['type']}")
    if aligned_ob:
        reason_codes.append("order_block_aligned")
    if aligned_fvg:
        reason_codes.append("fvg_aligned")
    if equal_levels.get("equal_highs"):
        reason_codes.append("equal_highs_pool")
    if equal_levels.get("equal_lows"):
        reason_codes.append("equal_lows_pool")
    reason_codes.append(f"regime_{regime}")
    reason_codes.append(f"session_{session}")

    setup_type = "sweep_reversal" if sweep else ("ob_continuation" if aligned_ob else "structure_break")

    return {
        "pair": symbol,
        "direction": "BUY" if bias == "bullish" else "SELL",
        "entry": round(entry, 10),
        "take_profit": round(take_profit, 10),
        "stop_loss": round(stop_loss, 10),
        "confidence": confidence,
        "risk_reward": risk_reward,
        "setup_type": setup_type,
        "market_regime": regime,
        "reason_codes": reason_codes,
        "feature_scores": feature_scores,
        "timestamp": int(time.time() * 1000),
        "strategy_version": STRATEGY_VERSION,
    }


# ----------------------------------------------------------------------
# public: update_position (dynamic trailing)
# ----------------------------------------------------------------------

def update_position(market_data: dict, position_context: dict) -> Optional[dict]:
    """
    market_data: {candles: [...]} recent candles for the symbol (live).
    position_context: {direction, entry, current_sl, current_tp, engine_state?, best_price?}
    Returns a trail decision dict or None (no change).
    """
    candles = market_data.get("candles") or []
    if len(candles) < 20:
        return None

    direction = position_context.get("direction")
    entry = _f(position_context.get("entry"))
    current_sl = _f(position_context.get("current_sl"))
    tp = _f(position_context.get("current_tp"))
    if not direction or entry <= 0 or current_sl <= 0:
        return None

    atr_value = atr(candles, 14)
    if atr_value <= 0:
        return None

    price = _cl(candles[-1])
    risk = abs(entry - current_sl)
    if risk <= 0:
        return None

    if direction == "BUY":
        profit_r = (price - entry) / risk
    else:
        profit_r = (entry - price) / risk

    if profit_r <= 0.3:
        return None  # not enough profit cushion to trail yet

    swings = find_swings(candles)
    reason_codes = []
    weakness_score = 0

    last_bodies = [abs(_cl(c) - _op(c)) for c in candles[-4:]]
    shrinking = len(last_bodies) >= 3 and last_bodies[-1] < last_bodies[-2] < last_bodies[-3]
    if shrinking:
        weakness_score += 2
        reason_codes.append("momentum exhaustion")

    last = candles[-1]
    opposite_candle = (_cl(last) < _op(last)) if direction == "BUY" else (_cl(last) > _op(last))
    if opposite_candle:
        weakness_score += 1
        reason_codes.append("opposite candle")

    best_price = position_context.get("best_price")
    giveback = 0.0
    if best_price:
        best_price = _f(best_price)
        span = abs(best_price - entry) or 1e-9
        if direction == "BUY":
            giveback = max(best_price - price, 0.0) / span
        else:
            giveback = max(price - best_price, 0.0) / span
        if giveback > 0.15:
            weakness_score += 1
            reason_codes.append("meaningful giveback")
        if giveback > 0.35:
            weakness_score += 2
            reason_codes.append("deep giveback")

    structure = detect_bos_choch(candles, swings)
    structure_failure = (direction == "BUY" and structure["bias"] == "bearish") or \
                         (direction == "SELL" and structure["bias"] == "bullish")
    if structure_failure:
        weakness_score += 2
        reason_codes.append("structure aligned against position")

    if weakness_score < 2:
        return None

    # candidate new SL: lock in a fraction of profit, anchored to recent structure
    lock_r = min(profit_r * 0.5, profit_r - 0.15)
    if lock_r <= 0:
        return None

    if direction == "BUY":
        struct_anchor = max([s["price"] for s in swings if s["type"] == "low" and s["price"] < price] or [current_sl])
        candidate_sl = max(entry + lock_r * risk, min(struct_anchor, price - atr_value * 0.5))
        new_sl = max(candidate_sl, current_sl)  # never loosen
        if new_sl <= current_sl:
            return None
    else:
        struct_anchor = min([s["price"] for s in swings if s["type"] == "high" and s["price"] > price] or [current_sl])
        candidate_sl = min(entry - lock_r * risk, max(struct_anchor, price + atr_value * 0.5))
        new_sl = min(candidate_sl, current_sl)  # never loosen
        if new_sl >= current_sl:
            return None

    engine = "structure" if structure_failure else ("giveback" if giveback > 0.15 else "momentum")
    state = "REVERSING" if structure_failure else ("WEAKENING" if weakness_score >= 3 else "TIGHTENING")

    return {
        "direction": direction,
        "state": state,
        "entry": entry,
        "current_price": price,
        "old_sl": current_sl,
        "new_sl": round(new_sl, 10),
        "profit_r": round(profit_r, 3),
        "tp": tp,
        "atr": round(atr_value, 10),
        "weakness_score": weakness_score,
        "engine": engine,
        "reason_codes": reason_codes,
        "timestamp": int(time.time() * 1000),
    }
