
from __future__ import annotations

"""
SMCAutoTrade strategy_v2.py

Rule-based, simulation-only strategy engine.
Source basis: combined.txt supplied by the user.

Design goals:
- Candidate != confirmed setup.
- Event-driven reevaluation per symbol/timeframe.
- Explicit reason codes and human-readable thesis.
- Setup lifecycle: CANDIDATE -> WATCHING -> IN_ZONE -> WAITING_CONFIRMATION
  -> PENDING_LIMIT / FILLED -> CLOSED / INVALIDATED / EXPIRED.
- Ranking across the entire universe, without forcing one setup per pair.
- Initial scan progress + summary even when zero setups are found.
- Simulation journal only; no broker/exchange order calls.

Timeframes:
- Native: 1m, 5m, 15m from start.py.
- Derived: 1H/4H/1D from 15m by aggregation.
"""

import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any

log = logging.getLogger("strategy")

# ---------------- configuration ----------------
MIN_RR = float(os.getenv("STRAT_V2_MIN_RR", "2.0"))
MIN_SCORE = int(os.getenv("STRAT_V2_MIN_SCORE", "62"))
MAX_ACTIVE_PER_SYMBOL = max(1, int(os.getenv("STRAT_V2_MAX_ACTIVE_PER_SYMBOL", "3")))
MAX_WATCHING_PER_SYMBOL = max(1, int(os.getenv("STRAT_V2_MAX_WATCHING_PER_SYMBOL", "4")))
MAX_TELEGRAM_SETUPS = max(1, int(os.getenv("STRAT_V2_MAX_TELEGRAM_SETUPS", "8")))
SCAN_LOG_EVERY = max(1, int(os.getenv("STRAT_V2_SCAN_LOG_EVERY", "10")))

SWING_LEFT = max(1, int(os.getenv("STRAT_V2_SWING_LEFT", "2")))
SWING_RIGHT = max(1, int(os.getenv("STRAT_V2_SWING_RIGHT", "2")))
ENTRY_ATR_PAD = float(os.getenv("STRAT_V2_ENTRY_ATR_PAD", "0.12"))
SL_ATR_PAD = float(os.getenv("STRAT_V2_SL_ATR_PAD", "0.20"))
EXPIRY_MINUTES = max(15, int(os.getenv("STRAT_V2_EXPIRY_MINUTES", "720")))

# A setup is allowed to be reported when it has a coherent score.
# Score is a ranking/filter, NOT a probability of winning.

API: Any = None
CONTEXT: dict[str, Any] = {}
LOCK = threading.RLock()

SETUPS: dict[str, "Setup"] = {}
JOURNAL: list[dict[str, Any]] = []
LAST_ANALYSIS: dict[str, dict[str, Any]] = {}
LAST_CLOSED_TS: dict[str, dict[str, int]] = {}
COUNTERS = {
    "initial_scans": 0,
    "event_scans": 0,
    "candidates": 0,
    "confirmed": 0,
    "fills": 0,
    "wins": 0,
    "losses": 0,
    "expired": 0,
    "invalidated": 0,
}
INITIAL_SCAN_DONE = False


@dataclass
class Setup:
    id: str
    symbol: str
    direction: str
    model: str
    state: str
    entry_type: str
    entry: float
    sl: float
    tp: float
    rr: float
    score: int
    created_ts: int
    expires_ts: int
    reason: str
    reason_codes: list[str]
    confluences: list[str]
    trigger_tf: str
    zone_low: float
    zone_high: float
    thesis: str
    confirmation_required: list[str]
    filled_ts: int | None = None
    outcome: str | None = None
    r_multiple: float | None = None


@dataclass
class Position:
    setup_id: str
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    opened_ts: int
    closed_ts: int | None = None
    outcome: str | None = None
    r_multiple: float | None = None


POSITIONS: dict[str, Position] = {}


# ---------------- public lifecycle ----------------
def initialize(api: Any, context: dict[str, Any]) -> None:
    global API, CONTEXT, INITIAL_SCAN_DONE
    API = api
    CONTEXT = dict(context)
    INITIAL_SCAN_DONE = False
    log.info(
        "[STRATEGY V2] initialized | min_score=%d min_rr=%.2f "
        "max_active=%d max_watch=%d",
        MIN_SCORE, MIN_RR, MAX_ACTIVE_PER_SYMBOL, MAX_WATCHING_PER_SYMBOL
    )


def shutdown() -> None:
    log.info(
        "[STRATEGY V2] shutdown | setups=%d journal=%d active=%d",
        len(SETUPS),
        len(JOURNAL),
        _active_count(),
    )


# ---------------- generic math/helpers ----------------
def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    e = sum(values[:period]) / period
    for x in values[period:]:
        e = alpha * x + (1.0 - alpha) * e
    return e


def _atr(candles: list[dict[str, Any]], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    tr = []
    for i in range(1, len(candles)):
        cur, prev = candles[i], candles[i - 1]
        tr.append(
            max(
                cur["high"] - cur["low"],
                abs(cur["high"] - prev["close"]),
                abs(cur["low"] - prev["close"]),
            )
        )
    return sum(tr[-period:]) / period


def _aggregate(candles: list[dict[str, Any]], minutes: int) -> list[dict[str, Any]]:
    if not candles:
        return []
    bucket_ms = minutes * 60_000
    groups: dict[int, list[dict[str, Any]]] = {}
    for c in candles:
        ts = int(c["timestamp"])
        key = (ts // bucket_ms) * bucket_ms
        groups.setdefault(key, []).append(c)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: x["timestamp"])
        out.append(
            {
                "timestamp": key,
                "open": float(group[0]["open"]),
                "high": max(float(x["high"]) for x in group),
                "low": min(float(x["low"]) for x in group),
                "close": float(group[-1]["close"]),
                "volume": sum(float(x.get("volume", 0.0)) for x in group),
                "turnover": sum(float(x.get("turnover", 0.0)) for x in group),
                "confirmed": all(bool(x.get("confirmed", True)) for x in group),
            }
        )
    return out


def _swing_points(candles: list[dict[str, Any]]) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    if len(candles) < SWING_LEFT + SWING_RIGHT + 1:
        return highs, lows
    for i in range(SWING_LEFT, len(candles) - SWING_RIGHT):
        h = candles[i]["high"]
        l = candles[i]["low"]
        left_highs = [candles[j]["high"] for j in range(i - SWING_LEFT, i)]
        right_highs = [candles[j]["high"] for j in range(i + 1, i + SWING_RIGHT + 1)]
        left_lows = [candles[j]["low"] for j in range(i - SWING_LEFT, i)]
        right_lows = [candles[j]["low"] for j in range(i + 1, i + SWING_RIGHT + 1)]

        if all(h > x for x in left_highs) and all(h >= x for x in right_highs):
            highs.append((i, float(h)))
        if all(l < x for x in left_lows) and all(l <= x for x in right_lows):
            lows.append((i, float(l)))
    return highs, lows


def _fvg(candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    zones = []
    for i in range(2, len(candles)):
        a = candles[i - 2]
        c = candles[i]
        if c["low"] > a["high"]:
            zones.append(
                {
                    "direction": "LONG",
                    "low": float(a["high"]),
                    "high": float(c["low"]),
                    "index": i,
                }
            )
        if c["high"] < a["low"]:
            zones.append(
                {
                    "direction": "SHORT",
                    "low": float(c["high"]),
                    "high": float(a["low"]),
                    "index": i,
                }
            )
    return zones


def _zone_fresh(candles: list[dict[str, Any]], zone: dict[str, Any], direction: str) -> bool:
    lo, hi = zone["low"], zone["high"]
    for c in candles[zone["index"] + 1 :]:
        if direction == "LONG" and c["low"] <= hi and c["high"] >= lo:
            return False
        if direction == "SHORT" and c["high"] >= lo and c["low"] <= hi:
            return False
    return True


def _order_blocks(candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    zones = []
    atr = _atr(candles, 14) or 0.0
    for i in range(2, len(candles)):
        prev = candles[i - 1]
        cur = candles[i]
        prev_body = abs(prev["close"] - prev["open"])
        up = cur["close"] > prev["high"] and (atr == 0.0 or prev_body >= atr * 0.12)
        down = cur["close"] < prev["low"] and (atr == 0.0 or prev_body >= atr * 0.12)

        # Heuristic "last opposite candle" definition.
        if up and prev["close"] < prev["open"]:
            zones.append(
                {
                    "direction": "LONG",
                    "low": float(prev["low"]),
                    "high": float(prev["open"]),
                    "index": i - 1,
                }
            )
        if down and prev["close"] > prev["open"]:
            zones.append(
                {
                    "direction": "SHORT",
                    "low": float(prev["open"]),
                    "high": float(prev["high"]),
                    "index": i - 1,
                }
            )
    return zones


def _trend(candles_1h: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candles_1h) < 30:
        return {
            "bias": "NEUTRAL",
            "score": 0,
            "codes": [],
            "labels": [],
            "ema9": None,
            "ema20": None,
        }
    closes = [float(c["close"]) for c in candles_1h]
    ema9 = _ema(closes, 9)
    ema20 = _ema(closes, 20)
    highs, lows = _swing_points(candles_1h)

    if ema9 is None or ema20 is None:
        return {"bias": "NEUTRAL", "score": 0, "codes": [], "labels": [], "ema9": ema9, "ema20": ema20}

    if ema9 > ema20:
        bias = "BULL"
        score = 20
        codes = ["HTF_EMA_BULL"]
        labels = ["1H EMA9 > EMA20"]
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
                score += 15
                codes.append("HTF_HH_HL")
                labels.append("1H HH + HL")
    elif ema9 < ema20:
        bias = "BEAR"
        score = 20
        codes = ["HTF_EMA_BEAR"]
        labels = ["1H EMA9 < EMA20"]
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
                score += 15
                codes.append("HTF_LH_LL")
                labels.append("1H LH + LL")
    else:
        bias = "NEUTRAL"
        score = 0
        codes = []
        labels = []

    return {
        "bias": bias,
        "score": score,
        "codes": codes,
        "labels": labels,
        "ema9": ema9,
        "ema20": ema20,
    }


def _fib_context(candles_1h: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    highs, lows = _swing_points(candles_1h)
    if not highs or not lows:
        return {"ok": False, "level": None, "code": None, "label": None}

    if direction == "LONG":
        hi_i, hi = highs[-1]
        prior_lows = [x for x in lows if x[0] < hi_i]
        if not prior_lows:
            return {"ok": False, "level": None, "code": None, "label": None}
        lo = prior_lows[-1][1]
        level = hi - 0.618 * (hi - lo)
        px = candles_1h[-1]["close"]
        return {
            "ok": px <= level,
            "level": level,
            "code": "FIB_DISCOUNT",
            "label": "1H price in 0.618 discount",
        }

    lo_i, lo = lows[-1]
    prior_highs = [x for x in highs if x[0] < lo_i]
    if not prior_highs:
        return {"ok": False, "level": None, "code": None, "label": None}
    hi = prior_highs[-1][1]
    level = lo + 0.618 * (hi - lo)
    px = candles_1h[-1]["close"]
    return {
        "ok": px >= level,
        "level": level,
        "code": "FIB_PREMIUM",
        "label": "1H price in 0.618 premium",
    }


def _liquidity_sweep(candles_5m: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    if len(candles_5m) < 10:
        return {"ok": False, "level": None, "code": None, "label": None}

    closed = candles_5m[:-1] if not candles_5m[-1].get("confirmed", True) else candles_5m
    if len(closed) < 8:
        return {"ok": False, "level": None, "code": None, "label": None}

    highs, lows = _swing_points(closed[:-1])
    c = closed[-1]
    if direction == "LONG" and lows:
        level = lows[-1][1]
        if c["low"] < level and c["close"] > level:
            return {"ok": True, "level": level, "code": "SSL_SWEEP", "label": "5M sell-side liquidity sweep"}
    if direction == "SHORT" and highs:
        level = highs[-1][1]
        if c["high"] > level and c["close"] < level:
            return {"ok": True, "level": level, "code": "BSL_SWEEP", "label": "5M buy-side liquidity sweep"}
    return {"ok": False, "level": None, "code": None, "label": None}


def _mss(candles_5m: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    closed = candles_5m[:-1] if candles_5m and not candles_5m[-1].get("confirmed", True) else candles_5m
    if len(closed) < 15:
        return {"ok": False, "level": None, "code": None, "label": None}

    highs, lows = _swing_points(closed[:-1])
    last_close = closed[-1]["close"]

    if direction == "LONG" and highs:
        level = highs[-1][1]
        if last_close > level:
            return {"ok": True, "level": level, "code": "MSS_BULL", "label": "5M bullish MSS/ChoCH"}
    if direction == "SHORT" and lows:
        level = lows[-1][1]
        if last_close < level:
            return {"ok": True, "level": level, "code": "MSS_BEAR", "label": "5M bearish MSS/ChoCH"}
    return {"ok": False, "level": None, "code": None, "label": None}


def _in_zone(price: float, zone: dict[str, Any], atr: float) -> bool:
    pad = atr * 0.25
    return zone["low"] - pad <= price <= zone["high"] + pad


def _best_zones(candles_1h: list[dict[str, Any]], direction: str) -> list[tuple[dict[str, Any], str]]:
    wanted = "LONG" if direction == "LONG" else "SHORT"
    out: list[tuple[dict[str, Any], str]] = []

    for z in _fvg(candles_1h):
        if z["direction"] == wanted and _zone_fresh(candles_1h, z, direction):
            out.append((z, "FVG"))

    for z in _order_blocks(candles_1h):
        if z["direction"] == wanted:
            out.append((z, "OB"))

    out.sort(key=lambda item: item[0]["index"], reverse=True)
    return out[:8]


def _active_setups(symbol: str) -> list[Setup]:
    return [
        s
        for s in SETUPS.values()
        if s.symbol == symbol and s.state in {
            "CANDIDATE", "WATCHING", "IN_ZONE", "WAITING_CONFIRMATION",
            "PENDING_LIMIT", "FILLED"
        }
    ]


def _active_count() -> int:
    return sum(
        1
        for s in SETUPS.values()
        if s.state in {"CANDIDATE", "WATCHING", "IN_ZONE", "WAITING_CONFIRMATION", "PENDING_LIMIT", "FILLED"}
    )


def _build_plan(
    symbol: str,
    direction: str,
    zone: dict[str, Any],
    model: str,
    score: int,
    codes: list[str],
    labels: list[str],
    ts: int,
    atr: float,
    confirmation_required: list[str],
) -> Setup | None:
    zl, zh = float(zone["low"]), float(zone["high"])
    entry = (zl + zh) / 2.0

    if direction == "LONG":
        sl = zl - max(atr * SL_ATR_PAD, (zh - zl) * 0.15)
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + MIN_RR * risk
    else:
        sl = zh + max(atr * SL_ATR_PAD, (zh - zl) * 0.15)
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - MIN_RR * risk

    dedup_key = f"{symbol}:{direction}:{model}:{int(zl*1e8)}:{int(zh*1e8)}"
    sid = f"{symbol}-{direction}-{int(ts)}-{uuid.uuid4().hex[:6]}"

    return Setup(
        id=sid,
        symbol=symbol,
        direction=direction,
        model=model,
        state="WATCHING",
        entry_type="LIMIT",
        entry=entry,
        sl=sl,
        tp=tp,
        rr=abs(tp - entry) / max(abs(entry - sl), 1e-12),
        score=min(100, int(score)),
        created_ts=ts,
        expires_ts=ts + EXPIRY_MINUTES * 60_000,
        reason=" | ".join(labels),
        reason_codes=list(dict.fromkeys(codes)),
        confluences=list(dict.fromkeys(labels)),
        trigger_tf="5",
        zone_low=zl,
        zone_high=zh,
        thesis=(
            f"{direction} because HTF bias and POI align; "
            f"entry is planned from {model} while waiting for micro confirmation."
        ),
        confirmation_required=list(dict.fromkeys(confirmation_required)),
    )


def _candidate_signature(setup: Setup) -> tuple:
    return (
        setup.symbol,
        setup.direction,
        setup.model,
        round(setup.zone_low, 8),
        round(setup.zone_high, 8),
    )


def _register_or_merge(setup: Setup) -> bool:
    signature = _candidate_signature(setup)
    for existing in SETUPS.values():
        if existing.state in {"CLOSED", "EXPIRED", "INVALIDATED"}:
            continue
        if _candidate_signature(existing) == signature:
            # Keep better score and preserve confirmation state.
            if setup.score > existing.score:
                existing.score = setup.score
                existing.reason = setup.reason
                existing.reason_codes = setup.reason_codes
                existing.confluences = setup.confluences
                existing.thesis = setup.thesis
            return False

    current = _active_setups(setup.symbol)
    if len(current) >= MAX_WATCHING_PER_SYMBOL:
        weakest = min(current, key=lambda x: (x.score, x.created_ts))
        if weakest.score >= setup.score:
            return False
        weakest.state = "INVALIDATED"
        weakest.outcome = "replaced_by_higher_score"
        COUNTERS["invalidated"] += 1

    SETUPS[setup.id] = setup
    COUNTERS["candidates"] += 1
    log.info(
        "[CANDIDATE] %s %s model=%s score=%d entry=%.8f rr=%.2f codes=%s",
        setup.symbol, setup.direction, setup.model, setup.score, setup.entry, setup.rr,
        ",".join(setup.reason_codes)
    )
    return True


def _scan_symbol(symbol: str, event_tf: str | None = None) -> dict[str, Any]:
    c15 = API.get_candles(symbol, "15", 700)
    c5 = API.get_candles(symbol, "5", 500)
    c1 = API.get_candles(symbol, "1", 500)

    result = {
        "symbol": symbol,
        "bias": "NEUTRAL",
        "score": 0,
        "candidates": [],
        "reason_codes": [],
        "labels": [],
        "event_tf": event_tf,
    }

    if len(c15) < 120 or len(c5) < 80 or len(c1) < 80:
        result["labels"] = ["insufficient history"]
        return result

    h1 = _aggregate(c15, 60)
    h4 = _aggregate(c15, 240)
    d1 = _aggregate(c15, 1440)
    if len(h1) < 30 or len(h4) < 8:
        result["labels"] = ["insufficient derived HTF history"]
        return result

    trend = _trend(h1)
    result["bias"] = trend["bias"]
    result["score"] = trend["score"]
    result["reason_codes"].extend(trend["codes"])
    result["labels"].extend(trend["labels"])

    if trend["bias"] == "NEUTRAL":
        result["labels"].append("HTF neutral")
        return result

    direction = "LONG" if trend["bias"] == "BULL" else "SHORT"
    fib = _fib_context(h1, direction)

    price = float(API.get_price(symbol) or c5[-1]["close"])
    atr = _atr(c5, 14) or max(price * 0.001, 1e-9)
    sweep = _liquidity_sweep(c5, direction)
    mss = _mss(c5, direction)

    # HTF zones based on H1; H4/D1 are context only in v2.
    zones = _best_zones(h1, direction)

    candidate_rows: list[dict[str, Any]] = []
    for zone, model in zones:
        in_zone = _in_zone(price, zone, atr)
        score = trend["score"]
        codes = list(trend["codes"])
        labels = list(trend["labels"])
        missing: list[str] = []

        if h4:
            h4trend = _trend(h4)
            if h4trend["bias"] == trend["bias"]:
                score += 10
                codes.append("H4_ALIGN")
                labels.append("4H bias aligned")
            else:
                missing.append("H4 alignment")

        if d1 and len(d1) >= 4:
            d1trend = _trend(d1)
            if d1trend["bias"] in {"BULL", "BEAR"} and d1trend["bias"] == trend["bias"]:
                score += 5
                codes.append("D1_ALIGN")
                labels.append("1D bias aligned")

        if fib["ok"]:
            score += 15
            if fib["code"]:
                codes.append(fib["code"])
            if fib["label"]:
                labels.append(fib["label"])
        else:
            missing.append("Fib 0.618 location")

        if model == "FVG":
            score += 15
            codes.append("FRESH_FVG")
            labels.append("Fresh H1 FVG")
        else:
            score += 15
            codes.append("ORDER_BLOCK")
            labels.append("H1 Order Block")

        if sweep["ok"]:
            score += 20
            if sweep["code"]:
                codes.append(sweep["code"])
            if sweep["label"]:
                labels.append(sweep["label"])
        else:
            missing.append("liquidity sweep")

        if mss["ok"]:
            score += 25
            if mss["code"]:
                codes.append(mss["code"])
            if mss["label"]:
                labels.append(mss["label"])
        else:
            missing.append("5M MSS/ChoCH")

        if not in_zone:
            missing.append("price at POI")

        state = "WATCHING"
        confirmations_needed: list[str] = []
        if in_zone:
            state = "IN_ZONE"
            confirmations_needed = [x for x in ("liquidity sweep", "5M MSS/ChoCH") if x in missing]
            if confirmations_needed:
                state = "WAITING_CONFIRMATION"
            else:
                state = "PENDING_LIMIT"

        candidate_rows.append(
            {
                "zone": zone,
                "model": model,
                "score": min(100, score),
                "codes": list(dict.fromkeys(codes)),
                "labels": list(dict.fromkeys(labels)),
                "missing": list(dict.fromkeys(missing)),
                "state": state,
            }
        )

    candidate_rows.sort(
        key=lambda x: (
            -int(x["score"]),
            0 if x["state"] in {"PENDING_LIMIT", "WAITING_CONFIRMATION", "IN_ZONE"} else 1,
        )
    )

    for row in candidate_rows[:MAX_ACTIVE_PER_SYMBOL]:
        if row["score"] < MIN_SCORE:
            continue
        plan = _build_plan(
            symbol=symbol,
            direction=direction,
            zone=row["zone"],
            model=row["model"],
            score=row["score"],
            codes=row["codes"],
            labels=row["labels"],
            ts=int(c5[-1]["timestamp"]),
            atr=atr,
            confirmation_required=row["missing"],
        )
        if plan:
            plan.state = row["state"]
            result["candidates"].append(plan)

    return result


def _transition_setup(setup: Setup, analysis: dict[str, Any], now_ts: int) -> list[str]:
    notices: list[str] = []
    rows = [
        r for r in analysis.get("candidates", [])
        if r.model == setup.model
        and r.direction == setup.direction
        and abs(r.zone_low - setup.zone_low) / max(abs(setup.zone_low), 1e-12) < 0.002
        and abs(r.zone_high - setup.zone_high) / max(abs(setup.zone_high), 1e-12) < 0.002
    ]
    if not rows:
        return notices

    row = rows[0]
    old_state = setup.state

    if now_ts >= setup.expires_ts and setup.state not in {"CLOSED", "FILLED"}:
        setup.state = "EXPIRED"
        setup.outcome = "expired"
        COUNTERS["expired"] += 1
        notices.append(f"⏳ {setup.symbol} {setup.direction} setup expired")
        return notices

    setup.score = max(setup.score, int(row["score"]))
    setup.reason_codes = list(dict.fromkeys(setup.reason_codes + row["codes"]))
    setup.confluences = list(dict.fromkeys(setup.confluences + row["labels"]))
    setup.reason = " | ".join(setup.confluences)
    setup.confirmation_required = list(row["missing"])

    if setup.state != "FILLED":
        target_state = str(row["state"])
        if target_state == "PENDING_LIMIT":
            setup.state = "PENDING_LIMIT"
        elif target_state == "WAITING_CONFIRMATION":
            setup.state = "WAITING_CONFIRMATION"
        elif target_state == "IN_ZONE":
            setup.state = "IN_ZONE"
        else:
            setup.state = "WATCHING"

        if setup.state != old_state:
            notices.append(_format_transition(setup, old_state))

    return notices


def _simulation_update(symbol: str, now_ts: int) -> list[str]:
    notices: list[str] = []
    price = API.get_price(symbol)
    if price is None:
        return notices

    for setup in list(SETUPS.values()):
        if setup.symbol != symbol:
            continue

        if setup.state == "PENDING_LIMIT":
            if now_ts >= setup.expires_ts:
                setup.state = "EXPIRED"
                setup.outcome = "expired"
                COUNTERS["expired"] += 1
                notices.append(f"⏳ {symbol} {setup.direction} limit expired")
                _journal_setup(setup)
                continue

            if setup.direction == "LONG" and setup.sl < price <= setup.entry:
                setup.state = "FILLED"
                setup.filled_ts = now_ts
                POSITIONS[setup.id] = Position(
                    setup.id, symbol, setup.direction, setup.entry, setup.sl, setup.tp, now_ts
                )
                COUNTERS["fills"] += 1
                notices.append(_format_fill(setup))
                log.info("[SIM] LIMIT FILLED %s %s entry=%.8f", symbol, setup.direction, setup.entry)

            elif setup.direction == "SHORT" and setup.entry <= price < setup.sl:
                setup.state = "FILLED"
                setup.filled_ts = now_ts
                POSITIONS[setup.id] = Position(
                    setup.id, symbol, setup.direction, setup.entry, setup.sl, setup.tp, now_ts
                )
                COUNTERS["fills"] += 1
                notices.append(_format_fill(setup))
                log.info("[SIM] LIMIT FILLED %s %s entry=%.8f", symbol, setup.direction, setup.entry)

        elif setup.state == "FILLED":
            pos = POSITIONS.get(setup.id)
            if not pos or pos.closed_ts is not None:
                continue

            # Price updates arrive from the live candle stream. This is intentionally
            # conservative: it does not invent intrabar ordering beyond current price.
            if setup.direction == "LONG":
                if price <= setup.sl:
                    _close_position(setup, now_ts, "SL", -1.0)
                    notices.append(_format_exit(setup))
                elif price >= setup.tp:
                    _close_position(setup, now_ts, "TP", setup.rr)
                    notices.append(_format_exit(setup))
            else:
                if price >= setup.sl:
                    _close_position(setup, now_ts, "SL", -1.0)
                    notices.append(_format_exit(setup))
                elif price <= setup.tp:
                    _close_position(setup, now_ts, "TP", setup.rr)
                    notices.append(_format_exit(setup))

    return notices


def _close_position(setup: Setup, ts: int, outcome: str, r_multiple: float) -> None:
    setup.state = "CLOSED"
    setup.outcome = outcome
    setup.r_multiple = r_multiple
    pos = POSITIONS.get(setup.id)
    if pos:
        pos.closed_ts = ts
        pos.outcome = outcome
        pos.r_multiple = r_multiple

    if outcome == "TP":
        COUNTERS["wins"] += 1
    elif outcome == "SL":
        COUNTERS["losses"] += 1

    _journal_setup(setup)
    log.info("[SIM] CLOSED %s %s %s R=%.2f", setup.symbol, setup.direction, outcome, r_multiple)


def _journal_setup(setup: Setup) -> None:
    JOURNAL.append(
        {
            "ts": int(time.time() * 1000),
            "setup": asdict(setup),
        }
    )
    if len(JOURNAL) > 5000:
        del JOURNAL[:-5000]


def _format_transition(setup: Setup, old_state: str) -> str:
    if setup.state == "IN_ZONE":
        return f"📍 {setup.symbol} {setup.direction}\nState: {old_state} → IN_ZONE\nWaiting for confirmation."
    if setup.state == "WAITING_CONFIRMATION":
        missing = ", ".join(setup.confirmation_required) or "micro confirmation"
        return f"⏳ {setup.symbol} {setup.direction}\nState: {old_state} → WAITING_CONFIRMATION\nMissing: {missing}"
    if setup.state == "PENDING_LIMIT":
        return _format_setup(setup, header="🟢 SETUP CONFIRMED")
    return ""


def _format_setup(setup: Setup, header: str = "🧠 SETUP") -> str:
    missing = ", ".join(setup.confirmation_required) if setup.confirmation_required else "-"
    codes = ", ".join(setup.reason_codes) if setup.reason_codes else "-"
    return (
        f"{header}\n\n"
        f"{setup.symbol} {setup.direction}\n"
        f"Model: {setup.model}\n"
        f"Score: {setup.score}/100\n"
        f"State: {setup.state}\n"
        f"Entry: {setup.entry_type} {setup.entry:.8f}\n"
        f"SL: {setup.sl:.8f}\n"
        f"TP: {setup.tp:.8f}\n"
        f"RR: {setup.rr:.2f}\n"
        f"Confluence: {', '.join(setup.confluences) or '-'}\n"
        f"Reason codes: {codes}\n"
        f"Waiting: {missing}"
    )


def _format_fill(setup: Setup) -> str:
    return (
        "🟢 SIMULATION FILLED\n\n"
        f"{setup.symbol} {setup.direction}\n"
        f"Entry: {setup.entry:.8f}\n"
        f"SL: {setup.sl:.8f}\n"
        f"TP: {setup.tp:.8f}\n"
        f"RR: {setup.rr:.2f}\n"
        f"Model: {setup.model}"
    )


def _format_exit(setup: Setup) -> str:
    icon = "✅" if setup.outcome == "TP" else "🛑"
    return (
        f"{icon} SIMULATION {setup.outcome}\n\n"
        f"{setup.symbol} {setup.direction}\n"
        f"R: {setup.r_multiple:.2f}\n"
        f"Model: {setup.model}"
    )


def _active_setups_sorted() -> list[Setup]:
    active = [
        s for s in SETUPS.values()
        if s.state in {
            "CANDIDATE", "WATCHING", "IN_ZONE",
            "WAITING_CONFIRMATION", "PENDING_LIMIT", "FILLED"
        }
    ]
    return sorted(active, key=lambda x: (-x.score, x.symbol, x.created_ts))


def scan_all(initial: bool = False) -> list[str]:
    if API is None or not API.is_bootstrap_complete():
        return ["ℹ️ Strategy menunggu historical data lengkap."]

    global INITIAL_SCAN_DONE
    symbols = API.get_symbols()
    total = len(symbols)
    found_new = 0
    watching = 0
    confirmed = 0
    no_setup = 0
    best_candidates: list[Setup] = []

    log.info(
        "[SCAN] %s scan start | symbols=%d",
        "INITIAL" if initial else "FULL",
        total,
    )

    for idx, symbol in enumerate(symbols, 1):
        with LOCK:
            try:
                analysis = _scan_symbol(symbol)
                LAST_ANALYSIS[symbol] = analysis
                COUNTERS["initial_scans" if initial else "event_scans"] += 1

                rows = analysis.get("candidates", [])
                if not rows:
                    no_setup += 1
                for setup in rows:
                    if _register_or_merge(setup):
                        found_new += 1
                    best_candidates.append(setup)
                    if setup.state in {"IN_ZONE", "WAITING_CONFIRMATION", "PENDING_LIMIT"}:
                        confirmed += 1
                    else:
                        watching += 1
            except Exception:
                log.exception("[SCAN] %s failed", symbol)

        if idx == 1 or idx % SCAN_LOG_EVERY == 0 or idx == total:
            log.info(
                "[SCAN] progress %d/%d | new=%d watching=%d actionable=%d no_setup=%d active=%d",
                idx, total, found_new, watching, confirmed, no_setup, _active_count()
            )

    INITIAL_SCAN_DONE = INITIAL_SCAN_DONE or initial

    active = _active_setups_sorted()
    summary = [
        "🔎 INITIAL STRATEGY SCAN COMPLETE" if initial else "🔎 STRATEGY SCAN COMPLETE",
        "",
        f"Pairs scanned: {total}",
        f"New candidates: {found_new}",
        f"Actionable/watch states: {len(active)}",
        f"Pairs without candidate: {no_setup}",
    ]

    if best_candidates:
        ranked = sorted(
            {s.id: s for s in best_candidates}.values(),
            key=lambda s: (-s.score, s.symbol),
        )
        summary += ["", "Top candidates:"]
        for i, s in enumerate(ranked[:MAX_TELEGRAM_SETUPS], 1):
            summary.append(
                f"{i}. {s.symbol} {s.direction} | {s.model} | "
                f"{s.state} | score={s.score} | RR={s.rr:.2f}"
            )
        if len(ranked) > MAX_TELEGRAM_SETUPS:
            summary.append(f"… +{len(ranked) - MAX_TELEGRAM_SETUPS} lainnya. Gunakan /setups.")
    else:
        summary += ["", "No candidate met the current rule threshold."]

    return ["\n".join(summary)]


def on_data_ready() -> str | None:
    with LOCK:
        messages = scan_all(initial=True)
    return "\n\n".join(messages)


def on_market_event(event: dict[str, Any]) -> str | None:
    if API is None or event.get("type") != "candle":
        return None

    symbol = str(event.get("symbol") or "").upper()
    tf = str(event.get("timeframe") or "")
    candle = event.get("candle") or {}
    if not symbol or tf not in {"1", "5", "15"}:
        return None

    now_ts = int(candle.get("timestamp") or int(time.time() * 1000))
    notices: list[str] = []

    with LOCK:
        # Simulation can react to every live price update.
        notices.extend(_simulation_update(symbol, now_ts))

        # Only closed candles are allowed to advance structural analysis.
        if not candle.get("confirmed", False):
            return "\n\n".join(notices[:2]) if notices else None

        # Micro timeframe updates: inspect only this symbol.
        if tf in {"5", "1"}:
            analysis = _scan_symbol(symbol, event_tf=tf)
            LAST_ANALYSIS[symbol] = analysis

            for setup in list(_active_setups(symbol)):
                notices.extend(_transition_setup(setup, analysis, now_ts))

            # Register new candidate if it is materially new.
            for setup in analysis.get("candidates", []):
                if _register_or_merge(setup):
                    if setup.state == "PENDING_LIMIT":
                        COUNTERS["confirmed"] += 1
                        notices.append(_format_setup(setup, header="🟢 NEW CONFIRMED SETUP"))
                    elif setup.state in {"IN_ZONE", "WAITING_CONFIRMATION"}:
                        notices.append(
                            f"⏳ {setup.symbol} {setup.direction} "
                            f"{setup.state}\nMissing: "
                            f"{', '.join(setup.confirmation_required) or '-'}"
                        )

            log.info(
                "[EVENT] %s %s closed | bias=%s candidates=%d",
                symbol, tf, analysis.get("bias"), len(analysis.get("candidates", []))
            )

        # 15M close can alter 1H/4H context, so rescan just this symbol.
        elif tf == "15":
            analysis = _scan_symbol(symbol, event_tf=tf)
            LAST_ANALYSIS[symbol] = analysis
            for setup in list(_active_setups(symbol)):
                notices.extend(_transition_setup(setup, analysis, now_ts))

            for setup in analysis.get("candidates", []):
                if _register_or_merge(setup):
                    notices.append(_format_setup(setup, header="🆕 NEW WATCH"))
            log.info(
                "[HTF] %s 15M close | bias=%s candidates=%d active=%d",
                symbol, analysis.get("bias"), len(analysis.get("candidates", [])), len(_active_setups(symbol))
            )

    # Telegram remains intentionally quiet unless something changed.
    unique: list[str] = []
    seen = set()
    for msg in notices:
        if not msg or msg in seen:
            continue
        seen.add(msg)
        unique.append(msg)
    return "\n\n".join(unique[:3]) if unique else None


# ---------------- Telegram commands ----------------
def _why(symbol: str) -> str:
    a = LAST_ANALYSIS.get(symbol)
    if not a:
        return f"ℹ️ Belum ada analysis snapshot untuk {symbol}. Gunakan /rescan."

    labels = a.get("labels") or []
    codes = a.get("reason_codes") or []
    candidates = a.get("candidates") or []

    out = [
        f"🔍 WHY {symbol}",
        "",
        f"Bias: {a.get('bias')}",
        f"Base score: {a.get('score', 0)}",
        f"Reason codes: {', '.join(codes) or '-'}",
        f"Context: {', '.join(labels) or '-'}",
        "",
        f"Candidates: {len(candidates)}",
    ]
    for i, s in enumerate(candidates[:5], 1):
        out += [
            "",
            f"{i}. {s.symbol} {s.direction} / {s.model}",
            f"State: {s.state}",
            f"Score: {s.score}",
            f"Confluence: {', '.join(s.confluences) or '-'}",
            f"Waiting: {', '.join(s.confirmation_required) or '-'}",
        ]
    return "\n".join(out)


def _setup_detail(setup_id: str) -> str:
    s = SETUPS.get(setup_id)
    if not s:
        return "❌ Setup ID tidak ditemukan."
    return _format_setup(s) + f"\n\nThesis: {s.thesis}"


def handle_command(text: str) -> str | None:
    parts = text.split()
    cmd = parts[0].lower() if parts else ""

    if cmd == "/setups":
        active = _active_setups_sorted()
        if not active:
            return "📭 Tidak ada active setup/candidate saat ini."
        lines = ["🧠 ACTIVE SETUPS"]
        for i, s in enumerate(active[:40], 1):
            lines.append(
                f"{i}. {s.symbol} {s.direction} | {s.model} | "
                f"{s.state} | {s.score} | RR {s.rr:.2f}"
            )
        if len(active) > 40:
            lines.append(f"… +{len(active)-40} lainnya")
        return "\n".join(lines)

    if cmd == "/setup":
        if len(parts) < 2:
            return "Format: /setup SETUP_ID"
        return _setup_detail(parts[1])

    if cmd == "/why":
        if len(parts) < 2:
            return "Format: /why BTCUSDT"
        return _why(parts[1].upper())

    if cmd == "/top":
        active = _active_setups_sorted()
        if not active:
            return "📭 Belum ada candidate."
        lines = ["🏆 TOP SETUPS"]
        for i, s in enumerate(active[:15], 1):
            lines.append(
                f"{i}. {s.symbol} {s.direction} | {s.model} | "
                f"score={s.score} | {s.state}"
            )
        return "\n".join(lines)

    if cmd == "/strategystatus":
        active = _active_setups_sorted()
        state_counts: dict[str, int] = {}
        for s in active:
            state_counts[s.state] = state_counts.get(s.state, 0) + 1
        return (
            "🧠 STRATEGY V2 STATUS\n"
            f"Symbols: {len(API.get_symbols()) if API else 0}\n"
            f"Initial scan done: {INITIAL_SCAN_DONE}\n"
            f"Initial scans: {COUNTERS['initial_scans']}\n"
            f"Event scans: {COUNTERS['event_scans']}\n"
            f"Candidates created: {COUNTERS['candidates']}\n"
            f"Confirmed count: {COUNTERS['confirmed']}\n"
            f"Fills: {COUNTERS['fills']}\n"
            f"Wins: {COUNTERS['wins']} | Losses: {COUNTERS['losses']}\n"
            f"Expired: {COUNTERS['expired']} | Invalidated: {COUNTERS['invalidated']}\n"
            f"Active: {_active_count()}\n"
            f"States: {state_counts or '-'}\n"
            f"Min score: {MIN_SCORE}\n"
            f"Min RR: {MIN_RR:.2f}"
        )

    if cmd == "/rescan":
        with LOCK:
            messages = scan_all(initial=False)
        return "\n\n".join(messages)

    if cmd == "/journal":
        if not JOURNAL:
            return "📓 Journal simulasi masih kosong."
        rows = JOURNAL[-20:]
        lines = ["📓 SIMULATION JOURNAL"]
        for row in reversed(rows):
            s = row["setup"]
            lines.append(
                f"{s['symbol']} {s['direction']} | {s['model']} | "
                f"{s['outcome']} | R={s['r_multiple']}"
            )
        return "\n".join(lines)

    if cmd == "/debug":
        if len(parts) < 2:
            return "Format: /debug BTCUSDT"
        symbol = parts[1].upper()
        a = LAST_ANALYSIS.get(symbol)
        if not a:
            return f"ℹ️ Tidak ada analysis snapshot untuk {symbol}."
        return repr(a)[:3800]

    return None
