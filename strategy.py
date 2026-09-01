from __future__ import annotations

"""
SMCAutoTrade strategy_v4.py

Simulation-only strategy engine.
Source basis: user's combined.txt trading transcripts.

Key design changes from v3:
- One PRIMARY THESIS per symbol+direction instead of multiple duplicate-looking setups.
- Stable POI identity to prevent repeated registrations on every websocket event.
- Alternative POIs are stored inside the same thesis, not as separate Telegram signals.
- Score is normalized from explicit components; it no longer simply saturates at 100.
- Signal output contains entry, confirmation price/condition, SL, TP, invalidation, RR,
  confluence and waiting conditions.
- Candidate/watch/confirmation lifecycle is explicit.
- Initial scan and live event logs always explain what happened, including zero-setup cases.
- Simulation only; no exchange order placement.
"""

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any

log = logging.getLogger("strategy")

MIN_RR = float(os.getenv("STRAT_V4_MIN_RR", "2.0"))
MIN_SCORE = int(os.getenv("STRAT_V4_MIN_SCORE", "60"))
MAX_TELEGRAM_SETUPS = max(1, int(os.getenv("STRAT_V4_MAX_TELEGRAM_SETUPS", "10")))
SCAN_LOG_EVERY = max(1, int(os.getenv("STRAT_V4_SCAN_LOG_EVERY", "10")))
EXPIRY_MINUTES = max(15, int(os.getenv("STRAT_V4_EXPIRY_MINUTES", "720")))
SWING_LEFT = max(1, int(os.getenv("STRAT_V4_SWING_LEFT", "2")))
SWING_RIGHT = max(1, int(os.getenv("STRAT_V4_SWING_RIGHT", "2")))
SL_ATR_PAD = float(os.getenv("STRAT_V4_SL_ATR_PAD", "0.20"))
ZONE_TOLERANCE = float(os.getenv("STRAT_V4_ZONE_TOLERANCE", "0.0025"))

API: Any = None
CONTEXT: dict[str, Any] = {}
LOCK = threading.RLock()
INITIAL_SCAN_DONE = False

SETUPS: dict[str, "Setup"] = {}
THESIS_INDEX: dict[str, str] = {}  # stable thesis_key -> setup_id
JOURNAL: list[dict[str, Any]] = []
POSITIONS: dict[str, "Position"] = {}
LAST_ANALYSIS: dict[str, dict[str, Any]] = {}

COUNTERS = {
    "symbols_scanned": 0,
    "event_scans": 0,
    "theses_created": 0,
    "theses_updated": 0,
    "confirmed": 0,
    "fills": 0,
    "wins": 0,
    "losses": 0,
    "expired": 0,
    "invalidated": 0,
}


@dataclass
class POI:
    poi_id: str
    model: str
    direction: str
    low: float
    high: float
    created_index: int
    fresh: bool
    rank_hint: float


@dataclass
class Setup:
    id: str
    thesis_key: str
    symbol: str
    direction: str
    state: str
    model: str
    entry_type: str
    entry_price: float
    confirmation_price: float | None
    confirmation_condition: str
    stop_loss: float
    take_profit: float
    tp_model: str
    invalidation_price: float
    rr: float
    score: int
    created_ts: int
    updated_ts: int
    expires_ts: int
    primary_poi: POI
    alternative_pois: list[POI] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    confluences: list[str] = field(default_factory=list)
    waiting_for: list[str] = field(default_factory=list)
    thesis: str = ""
    filled_ts: int | None = None
    outcome: str | None = None
    r_multiple: float | None = None
    confirmation_ts: int | None = None


@dataclass
class Position:
    setup_id: str
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    opened_ts: int
    closed_ts: int | None = None
    outcome: str | None = None
    r_multiple: float | None = None


# ---------------- basic helpers ----------------
def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    value = sum(values[:period]) / period
    for x in values[period:]:
        value = alpha * x + (1.0 - alpha) * value
    return value


def _atr(candles: list[dict[str, Any]], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    tr = []
    for i in range(1, len(candles)):
        cur, prev = candles[i], candles[i - 1]
        tr.append(max(
            cur["high"] - cur["low"],
            abs(cur["high"] - prev["close"]),
            abs(cur["low"] - prev["close"]),
        ))
    return sum(tr[-period:]) / period


def _aggregate(candles: list[dict[str, Any]], minutes: int) -> list[dict[str, Any]]:
    if not candles:
        return []
    bucket = minutes * 60_000
    groups: dict[int, list[dict[str, Any]]] = {}
    for c in candles:
        key = (int(c["timestamp"]) // bucket) * bucket
        groups.setdefault(key, []).append(c)
    out = []
    for ts, group in sorted(groups.items()):
        group = sorted(group, key=lambda x: x["timestamp"])
        out.append({
            "timestamp": ts,
            "open": float(group[0]["open"]),
            "high": max(float(x["high"]) for x in group),
            "low": min(float(x["low"]) for x in group),
            "close": float(group[-1]["close"]),
            "volume": sum(float(x.get("volume", 0.0)) for x in group),
            "turnover": sum(float(x.get("turnover", 0.0)) for x in group),
            "confirmed": all(bool(x.get("confirmed", True)) for x in group),
        })
    return out


def _swings(candles: list[dict[str, Any]]) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs, lows = [], []
    n = len(candles)
    if n < SWING_LEFT + SWING_RIGHT + 1:
        return highs, lows
    for i in range(SWING_LEFT, n - SWING_RIGHT):
        h = candles[i]["high"]
        l = candles[i]["low"]
        if all(h > candles[j]["high"] for j in range(i - SWING_LEFT, i)) and all(
            h >= candles[j]["high"] for j in range(i + 1, i + SWING_RIGHT + 1)
        ):
            highs.append((i, float(h)))
        if all(l < candles[j]["low"] for j in range(i - SWING_LEFT, i)) and all(
            l <= candles[j]["low"] for j in range(i + 1, i + SWING_RIGHT + 1)
        ):
            lows.append((i, float(l)))
    return highs, lows


def _fvg(candles: list[dict[str, Any]]) -> list[POI]:
    out: list[POI] = []
    for i in range(2, len(candles)):
        a, c = candles[i - 2], candles[i]
        if c["low"] > a["high"]:
            out.append(POI(f"FVG-L-{i}", "FVG", "LONG", float(a["high"]), float(c["low"]), i, True, float(i)))
        elif c["high"] < a["low"]:
            out.append(POI(f"FVG-S-{i}", "FVG", "SHORT", float(c["high"]), float(a["low"]), i, True, float(i)))
    return out


def _fvg_fresh(candles: list[dict[str, Any]], poi: POI, direction: str) -> bool:
    for c in candles[poi.created_index + 1:]:
        if direction == "LONG" and c["low"] <= poi.high and c["high"] >= poi.low:
            return False
        if direction == "SHORT" and c["high"] >= poi.low and c["low"] <= poi.high:
            return False
    return True


def _order_blocks(candles: list[dict[str, Any]]) -> list[POI]:
    atr = _atr(candles, 14) or 0.0
    out: list[POI] = []
    for i in range(2, len(candles)):
        prev, cur = candles[i - 1], candles[i]
        body = abs(prev["close"] - prev["open"])
        if cur["close"] > prev["high"] and prev["close"] < prev["open"] and (atr == 0 or body >= atr * 0.10):
            out.append(POI(f"OB-L-{i-1}", "OB", "LONG", float(prev["low"]), float(prev["open"]), i - 1, True, float(i)))
        elif cur["close"] < prev["low"] and prev["close"] > prev["open"] and (atr == 0 or body >= atr * 0.10):
            out.append(POI(f"OB-S-{i-1}", "OB", "SHORT", float(prev["open"]), float(prev["high"]), i - 1, True, float(i)))
    return out


def _trend(candles_1h: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candles_1h) < 30:
        return {"bias": "NEUTRAL", "components": [], "score": 0, "ema9": None, "ema20": None}
    closes = [float(c["close"]) for c in candles_1h]
    ema9, ema20 = _ema(closes, 9), _ema(closes, 20)
    highs, lows = _swings(candles_1h)
    if ema9 is None or ema20 is None:
        return {"bias": "NEUTRAL", "components": [], "score": 0, "ema9": ema9, "ema20": ema20}
    if ema9 > ema20:
        bias = "BULL"
        comps = [("HTF_EMA_BULL", 22, "1H EMA9 > EMA20")]
        if len(highs) >= 2 and len(lows) >= 2 and highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
            comps.append(("HTF_HH_HL", 13, "1H HH + HL"))
    elif ema9 < ema20:
        bias = "BEAR"
        comps = [("HTF_EMA_BEAR", 22, "1H EMA9 < EMA20")]
        if len(highs) >= 2 and len(lows) >= 2 and highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
            comps.append(("HTF_LH_LL", 13, "1H LH + LL"))
    else:
        bias, comps = "NEUTRAL", []
    return {"bias": bias, "components": comps, "score": sum(x[1] for x in comps), "ema9": ema9, "ema20": ema20}


def _fib(candles_1h: list[dict[str, Any]], direction: str) -> tuple[bool, float | None, str, str]:
    highs, lows = _swings(candles_1h)
    if not highs or not lows:
        return False, None, "", ""
    if direction == "LONG":
        hi_i, hi = highs[-1]
        prior_lows = [x for x in lows if x[0] < hi_i]
        if not prior_lows:
            return False, None, "", ""
        lo = prior_lows[-1][1]
        level = hi - 0.618 * (hi - lo)
        return candles_1h[-1]["close"] <= level, level, "FIB_DISCOUNT", "1H price in 0.618 discount"
    lo_i, lo = lows[-1]
    prior_highs = [x for x in highs if x[0] < lo_i]
    if not prior_highs:
        return False, None, "", ""
    hi = prior_highs[-1][1]
    level = lo + 0.618 * (hi - lo)
    return candles_1h[-1]["close"] >= level, level, "FIB_PREMIUM", "1H price in 0.618 premium"


def _micro_confirmation(c5: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    closed = c5[:-1] if c5 and not c5[-1].get("confirmed", True) else c5
    if len(closed) < 20:
        return {"sweep": False, "mss": False, "sweep_code": "", "mss_code": "", "level": None}
    highs, lows = _swings(closed[:-1])
    last = closed[-1]
    sweep = False
    sweep_code = ""
    sweep_level = None
    if direction == "LONG" and lows:
        sweep_level = lows[-1][1]
        sweep = last["low"] < sweep_level and last["close"] > sweep_level
        sweep_code = "SSL_SWEEP" if sweep else ""
    elif direction == "SHORT" and highs:
        sweep_level = highs[-1][1]
        sweep = last["high"] > sweep_level and last["close"] < sweep_level
        sweep_code = "BSL_SWEEP" if sweep else ""

    mss = False
    mss_code = ""
    mss_level = None
    if direction == "LONG" and highs:
        mss_level = highs[-1][1]
        mss = last["close"] > mss_level
        mss_code = "MSS_BULL" if mss else ""
    elif direction == "SHORT" and lows:
        mss_level = lows[-1][1]
        mss = last["close"] < mss_level
        mss_code = "MSS_BEAR" if mss else ""
    return {"sweep": sweep, "mss": mss, "sweep_code": sweep_code, "mss_code": mss_code, "level": mss_level or sweep_level}


def _in_zone(price: float, poi: POI, atr: float) -> bool:
    pad = atr * 0.25
    return poi.low - pad <= price <= poi.high + pad


def _poi_key(symbol: str, direction: str, poi: POI) -> str:
    return f"{symbol}:{direction}:{poi.model}:{poi.poi_id}"


def _thesis_key(symbol: str, direction: str) -> str:
    # Exactly one primary thesis per symbol. Direction can change only if the
    # existing thesis is still pre-entry and the new thesis is materially stronger.
    return symbol


def _calculate_score(components: list[tuple[str, float, str]]) -> int:
    # Weights are intentionally bounded. This is a ranking/filter score, not probability.
    total = sum(max(0.0, weight) for _, weight, _ in components)
    max_possible = 108.0
    return max(0, min(100, round((total / max_possible) * 100.0)))


def _build_setup(symbol: str, direction: str, poi: POI, alternatives: list[POI], a: dict[str, Any], now_ts: int) -> Setup | None:
    price = float(API.get_price(symbol) or a["price"])
    atr = float(a["atr"])
    entry = (poi.low + poi.high) / 2.0
    pad = max(atr * SL_ATR_PAD, (poi.high - poi.low) * 0.15)
    if direction == "LONG":
        sl = poi.low - pad
        risk = entry - sl
        tp = entry + MIN_RR * risk
    else:
        sl = poi.high + pad
        risk = sl - entry
        tp = entry - MIN_RR * risk
    if risk <= 0:
        return None

    micro = a["micro"]
    in_zone = _in_zone(price, poi, atr)
    waiting = []
    score_components = list(a["components"])
    codes = [x[0] for x in score_components]
    labels = [x[2] for x in score_components]

    # POI
    score_components.append(("POI", 15, f"{poi.model} primary POI"))
    codes.append("ORDER_BLOCK" if poi.model == "OB" else "FRESH_FVG")
    labels.append(f"H1 {poi.model} primary zone")

    if a["fib_ok"]:
        score_components.append((a["fib_code"], 12, a["fib_label"]))
        codes.append(a["fib_code"])
        labels.append(a["fib_label"])
    else:
        waiting.append("0.618 Fibonacci location")

    if a["h4_align"]:
        score_components.append(("H4_ALIGN", 8, "4H bias aligned"))
        codes.append("H4_ALIGN")
        labels.append("4H bias aligned")
    if a["d1_align"]:
        score_components.append(("D1_ALIGN", 5, "1D bias aligned"))
        codes.append("D1_ALIGN")
        labels.append("1D bias aligned")

    if micro["sweep"]:
        score_components.append((micro["sweep_code"], 15, "5M liquidity sweep"))
        codes.append(micro["sweep_code"])
        labels.append("5M liquidity sweep")
    else:
        waiting.append("5M liquidity sweep")

    if micro["mss"]:
        score_components.append((micro["mss_code"], 18, "5M MSS/ChoCH"))
        codes.append(micro["mss_code"])
        labels.append("5M MSS/ChoCH")
    else:
        waiting.append("5M MSS/ChoCH")

    if not in_zone:
        waiting.append("price returns to primary POI")

    state = "WATCHING"
    if in_zone:
        state = "WAITING_CONFIRMATION" if waiting else "PENDING_LIMIT"

    confirmation_price = micro["level"]
    confirmation_condition = (
        f"5M close {'above' if direction == 'LONG' else 'below'} {confirmation_price:.8f}"
        if confirmation_price is not None else
        f"5M liquidity sweep + {'bullish' if direction == 'LONG' else 'bearish'} MSS"
    )

    score = _calculate_score(score_components)
    if score < MIN_SCORE:
        return None

    thesis_key = _thesis_key(symbol, direction)
    sid = f"S4-{symbol}-{direction}-{uuid.uuid4().hex[:8]}"
    thesis = (
        f"{direction} thesis: {a['bias_label']}; price is being evaluated around "
        f"the {poi.model} POI, with liquidity/structure confirmation required before entry."
    )
    return Setup(
        id=sid,
        thesis_key=thesis_key,
        symbol=symbol,
        direction=direction,
        state=state,
        model=poi.model,
        entry_type="LIMIT",
        entry_price=entry,
        confirmation_price=confirmation_price,
        confirmation_condition=confirmation_condition,
        stop_loss=sl,
        take_profit=tp,
        tp_model="FIXED_RR",
        invalidation_price=poi.low if direction == "LONG" else poi.high,
        rr=abs(tp - entry) / max(abs(entry - sl), 1e-12),
        score=score,
        created_ts=now_ts,
        updated_ts=now_ts,
        expires_ts=now_ts + EXPIRY_MINUTES * 60_000,
        primary_poi=poi,
        alternative_pois=alternatives,
        reason_codes=list(dict.fromkeys(codes)),
        confluences=list(dict.fromkeys(labels)),
        waiting_for=list(dict.fromkeys(waiting)),
        thesis=thesis,
    )


def _active_thesis(symbol: str, direction: str | None = None) -> Setup | None:
    sid = THESIS_INDEX.get(_thesis_key(symbol, direction or ""))
    if not sid:
        return None
    setup = SETUPS.get(sid)
    if not setup or setup.state in {"CLOSED", "EXPIRED", "INVALIDATED"}:
        return None
    return setup


def _merge_thesis(existing: Setup, incoming: Setup) -> bool:
    changed = False
    if incoming.score > existing.score:
        existing.score = incoming.score
        changed = True
    if incoming.updated_ts > existing.updated_ts:
        existing.updated_ts = incoming.updated_ts
    if incoming.state != existing.state:
        priority = {
            "WATCHING": 1,
            "IN_ZONE": 2,
            "WAITING_CONFIRMATION": 3,
            "PENDING_LIMIT": 4,
            "FILLED": 5,
            "CLOSED": 6,
            "EXPIRED": 0,
            "INVALIDATED": 0,
        }
        if priority.get(incoming.state, 0) > priority.get(existing.state, 0):
            existing.state = incoming.state
            changed = True
    if existing.primary_poi.poi_id != incoming.primary_poi.poi_id and incoming.score >= existing.score:
        existing.alternative_pois.append(existing.primary_poi)
        existing.primary_poi = incoming.primary_poi
        existing.model = incoming.model
        changed = True

    known = {p.poi_id for p in existing.alternative_pois}
    for p in incoming.alternative_pois + [existing.primary_poi]:
        if p.poi_id != existing.primary_poi.poi_id and p.poi_id not in known:
            existing.alternative_pois.append(p)
            known.add(p.poi_id)
            changed = True

    existing.reason_codes = list(dict.fromkeys(existing.reason_codes + incoming.reason_codes))
    existing.confluences = list(dict.fromkeys(existing.confluences + incoming.confluences))
    existing.waiting_for = list(dict.fromkeys(incoming.waiting_for))
    existing.confirmation_price = incoming.confirmation_price
    existing.confirmation_condition = incoming.confirmation_condition
    existing.entry_price = incoming.entry_price
    existing.stop_loss = incoming.stop_loss
    existing.take_profit = incoming.take_profit
    existing.tp_model = incoming.tp_model
    existing.invalidation_price = incoming.invalidation_price
    existing.rr = incoming.rr
    existing.thesis = incoming.thesis
    if changed:
        COUNTERS["theses_updated"] += 1
    return changed


def _register_thesis(setup: Setup) -> tuple[bool, Setup]:
    existing = _active_thesis(setup.symbol)
    if existing:
        # Never replace a filled/live position with an unrelated thesis.
        if existing.state == "FILLED":
            return False, existing

        # Same-direction thesis: merge POIs/confluence and refresh prices.
        if existing.direction == setup.direction:
            _merge_thesis(existing, setup)
            return False, existing

        # Opposite thesis: only switch when the incoming thesis is materially stronger.
        if setup.score >= existing.score + 8:
            existing.state = "INVALIDATED"
            existing.outcome = "replaced_by_stronger_opposite_thesis"
            COUNTERS["invalidated"] += 1
            THESIS_INDEX.pop(existing.thesis_key, None)
            SETUPS.pop(existing.id, None)
        else:
            return False, existing

    SETUPS[setup.id] = setup
    THESIS_INDEX[setup.thesis_key] = setup.id
    COUNTERS["theses_created"] += 1
    log.info(
        "[THESIS] NEW %s | %s | model=%s score=%d state=%s entry=%.8f sl=%.8f tp=%.8f",
        setup.symbol, setup.direction, setup.model, setup.score, setup.state,
        setup.entry_price, setup.stop_loss, setup.take_profit,
    )
    return True, setup


def _analysis_for_symbol(symbol: str, event_tf: str | None = None) -> dict[str, Any]:
    c15 = API.get_candles(symbol, "15", 700)
    c5 = API.get_candles(symbol, "5", 500)
    c1 = API.get_candles(symbol, "1", 500)
    price = float(API.get_price(symbol) or (c1[-1]["close"] if c1 else 0.0))
    a: dict[str, Any] = {
        "symbol": symbol,
        "bias": "NEUTRAL",
        "bias_label": "HTF neutral",
        "price": price,
        "atr": _atr(c5, 14) or max(price * 0.001, 1e-9),
        "components": [],
        "fib_ok": False,
        "fib_code": "",
        "fib_label": "",
        "h4_align": False,
        "d1_align": False,
        "micro": _micro_confirmation(c5, "LONG"),
        "candidates": [],
        "event_tf": event_tf,
    }
    if len(c15) < 120 or len(c5) < 80 or len(c1) < 80:
        a["labels"] = ["insufficient history"]
        return a

    h1 = _aggregate(c15, 60)
    h4 = _aggregate(c15, 240)
    d1 = _aggregate(c15, 1440)
    if len(h1) < 30 or len(h4) < 8:
        a["labels"] = ["insufficient derived HTF history"]
        return a

    t1 = _trend(h1)
    t4 = _trend(h4)
    td = _trend(d1) if len(d1) >= 30 else {"bias": "NEUTRAL", "components": []}
    a["bias"] = t1["bias"]
    a["bias_label"] = "1H bullish" if t1["bias"] == "BULL" else "1H bearish" if t1["bias"] == "BEAR" else "HTF neutral"
    a["components"] = list(t1["components"])
    a["h4_align"] = t4["bias"] == t1["bias"] and t1["bias"] != "NEUTRAL"
    a["d1_align"] = td["bias"] == t1["bias"] and t1["bias"] != "NEUTRAL"
    a["micro"] = _micro_confirmation(c5, "LONG" if t1["bias"] == "BULL" else "SHORT") if t1["bias"] != "NEUTRAL" else a["micro"]

    if t1["bias"] == "NEUTRAL":
        return a

    direction = "LONG" if t1["bias"] == "BULL" else "SHORT"
    fib_ok, fib_level, fib_code, fib_label = _fib(h1, direction)
    a["fib_ok"], a["fib_level"], a["fib_code"], a["fib_label"] = fib_ok, fib_level, fib_code, fib_label

    pois = []
    for p in _fvg(h1):
        if p.direction == direction and p.high > p.low and _fvg_fresh(h1, p, direction):
            pois.append(p)
    for p in _order_blocks(h1):
        if p.direction == direction:
            pois.append(p)

    pois.sort(key=lambda p: p.created_index, reverse=True)
    if not pois:
        return a

    primary = pois[0]
    alternatives = pois[1:5]
    primary, alternatives = primary, alternatives
    setup = _build_setup(symbol, direction, primary, alternatives, a, int(c15[-1]["timestamp"]))
    if setup:
        a["candidates"] = [setup]
    return a


# ---------------- public lifecycle ----------------
def initialize(api: Any, context: dict[str, Any]) -> None:
    global API, CONTEXT, INITIAL_SCAN_DONE
    API = api
    CONTEXT = dict(context)
    INITIAL_SCAN_DONE = False
    log.info(
        "[STRATEGY V4] initialized | min_score=%d min_rr=%.2f expiry=%dm",
        MIN_SCORE, MIN_RR, EXPIRY_MINUTES,
    )


def shutdown() -> None:
    log.info(
        "[STRATEGY V4] shutdown | active=%d theses=%d journal=%d",
        _active_count(), len(SETUPS), len(JOURNAL),
    )


def _active_count() -> int:
    return sum(
        1 for s in SETUPS.values()
        if s.state in {"WATCHING", "IN_ZONE", "WAITING_CONFIRMATION", "PENDING_LIMIT", "FILLED"}
    )


def scan_all(initial: bool = False) -> list[str]:
    if API is None or not API.is_bootstrap_complete():
        return ["ℹ️ Strategy masih menunggu historical data lengkap."]

    symbols = API.get_symbols()
    total = len(symbols)
    no_candidate = 0
    created = 0
    seen_before = set()
    log.info("[SCAN] %s start | %d symbols", "INITIAL" if initial else "FULL", total)

    for idx, symbol in enumerate(symbols, 1):
        try:
            analysis = _analysis_for_symbol(symbol)
            LAST_ANALYSIS[symbol] = analysis
            if not analysis.get("candidates"):
                no_candidate += 1
            for setup in analysis.get("candidates", []):
                seen_before.add(setup.thesis_key)
                is_new, current = _register_thesis(setup)
                if is_new:
                    created += 1
        except Exception:
            log.exception("[SCAN] %s failed", symbol)
            no_candidate += 1
        COUNTERS["symbols_scanned"] += 1
        if idx == 1 or idx % SCAN_LOG_EVERY == 0 or idx == total:
            log.info(
                "[SCAN] progress %d/%d | new=%d no_candidate=%d active=%d",
                idx, total, created, no_candidate, _active_count(),
            )

    global INITIAL_SCAN_DONE
    INITIAL_SCAN_DONE = True

    active = sorted(_active_setups(), key=lambda s: (-s.score, s.symbol, s.direction))
    lines = [
        "🔎 INITIAL STRATEGY SCAN COMPLETE" if initial else "🔎 STRATEGY SCAN COMPLETE",
        "",
        f"Pairs scanned: {total}",
        f"New primary theses: {created}",
        f"Active theses: {len(active)}",
        f"Pairs with no candidate: {no_candidate}",
    ]
    if not active:
        lines += ["", "No setup met the minimum rule threshold."]
    else:
        lines += ["", "Top signals:"]
        for i, s in enumerate(active[:MAX_TELEGRAM_SETUPS], 1):
            lines.append(f"{i}. {s.symbol} {s.direction} | {s.model} | {s.state} | score={s.score} | RR={s.rr:.2f}")
        if len(active) > MAX_TELEGRAM_SETUPS:
            lines.append(f"… +{len(active)-MAX_TELEGRAM_SETUPS} lainnya. Gunakan /setups.")
    return ["\n".join(lines)]


def on_data_ready() -> str | None:
    return "\n\n".join(scan_all(initial=True))


def _active_setups(symbol: str | None = None) -> list[Setup]:
    states = {"WATCHING", "IN_ZONE", "WAITING_CONFIRMATION", "PENDING_LIMIT", "FILLED"}
    rows = [s for s in SETUPS.values() if s.state in states]
    if symbol:
        rows = [s for s in rows if s.symbol == symbol.upper()]
    return sorted(rows, key=lambda s: (-s.score, s.symbol, s.direction))


def _expire_and_update_simulation(symbol: str, now_ts: int) -> list[str]:
    notices = []
    price = API.get_price(symbol)
    if price is None:
        return notices
    for s in list(_active_setups(symbol)):
        if s.state != "FILLED" and now_ts >= s.expires_ts:
            s.state = "EXPIRED"
            s.outcome = "expired"
            COUNTERS["expired"] += 1
            notices.append(f"⏳ SIGNAL EXPIRED\n{s.symbol} {s.direction}\nID: {s.id}")
            _journal(s)
            continue
        if s.state == "PENDING_LIMIT":
            filled = (s.direction == "LONG" and s.stop_loss < price <= s.entry_price) or (s.direction == "SHORT" and s.entry_price <= price < s.stop_loss)
            if filled:
                s.state = "FILLED"
                s.filled_ts = now_ts
                POSITIONS[s.id] = Position(s.id, s.symbol, s.direction, s.entry_price, s.stop_loss, s.take_profit, now_ts)
                COUNTERS["fills"] += 1
                notices.append(_format_signal(s, "🟢 SIMULATION FILLED"))
        elif s.state == "FILLED":
            pos = POSITIONS.get(s.id)
            if not pos or pos.closed_ts:
                continue
            hit_sl = price <= s.stop_loss if s.direction == "LONG" else price >= s.stop_loss
            hit_tp = price >= s.take_profit if s.direction == "LONG" else price <= s.take_profit
            if hit_sl:
                _close(s, now_ts, "SL", -1.0)
                notices.append(_format_exit(s))
            elif hit_tp:
                _close(s, now_ts, "TP", s.rr)
                notices.append(_format_exit(s))
    return notices


def _close(s: Setup, ts: int, outcome: str, r: float) -> None:
    s.state = "CLOSED"
    s.outcome = outcome
    s.r_multiple = r
    pos = POSITIONS.get(s.id)
    if pos:
        pos.closed_ts = ts
        pos.outcome = outcome
        pos.r_multiple = r
    if outcome == "TP":
        COUNTERS["wins"] += 1
    elif outcome == "SL":
        COUNTERS["losses"] += 1
    _journal(s)


def _journal(s: Setup) -> None:
    JOURNAL.append({"ts": int(time.time() * 1000), "setup": asdict(s)})
    if len(JOURNAL) > 5000:
        del JOURNAL[:-5000]


def on_market_event(event: dict[str, Any]) -> str | None:
    if API is None or event.get("type") != "candle":
        return None
    symbol = str(event.get("symbol") or "").upper()
    tf = str(event.get("timeframe") or "")
    candle = event.get("candle") or {}
    if not symbol or tf not in {"1", "5", "15"}:
        return None

    now_ts = int(candle.get("timestamp") or int(time.time() * 1000))
    notices = []
    with LOCK:
        notices.extend(_expire_and_update_simulation(symbol, now_ts))
        if not candle.get("confirmed", False):
            return "\n\n".join(notices[:2]) if notices else None

        analysis = _analysis_for_symbol(symbol, tf)
        LAST_ANALYSIS[symbol] = analysis
        COUNTERS["event_scans"] += 1

        for incoming in analysis.get("candidates", []):
            existing = _active_thesis(incoming.symbol, incoming.direction)
            if existing:
                old_state = existing.state
                _merge_thesis(existing, incoming)
                if existing.state != old_state:
                    if existing.state == "WAITING_CONFIRMATION":
                        notices.append(_format_signal(existing, "⏳ CONFIRMATION NEEDED"))
                    elif existing.state == "PENDING_LIMIT":
                        COUNTERS["confirmed"] += 1
                        existing.confirmation_ts = now_ts
                        notices.append(_format_signal(existing, "🟢 SETUP CONFIRMED"))
            else:
                is_new, setup = _register_thesis(incoming)
                if is_new:
                    notices.append(_format_signal(setup, "🧠 NEW TRADING SIGNAL"))

        log.info(
            "[EVENT] %s %s CLOSED | bias=%s candidate=%d active=%d",
            symbol, tf, analysis.get("bias"), len(analysis.get("candidates", [])), len(_active_setups(symbol)),
        )
    return "\n\n".join(dict.fromkeys(notices[:3])) if notices else None


# ---------------- signal formatting ----------------
def _format_signal(s: Setup, header: str = "🧠 TRADING SIGNAL") -> str:
    waiting = ", ".join(s.waiting_for) if s.waiting_for else "-"
    alts = ", ".join(f"{p.model} {p.low:.8f}-{p.high:.8f}" for p in s.alternative_pois[:3]) or "-"
    return (
        f"{header}\n\n"
        f"{s.symbol} — {s.direction}\n"
        f"Model: {s.model}\n"
        f"Score: {s.score}/100\n"
        f"Status: {s.state}\n\n"
        f"📍 Entry: {s.entry_type} @ {s.entry_price:.8f}\n"
        f"🔔 Confirmation: {s.confirmation_price:.8f} ({s.confirmation_condition})\n" if s.confirmation_price is not None else
        f"{header}\n\n{s.symbol} — {s.direction}\nModel: {s.model}\nScore: {s.score}/100\nStatus: {s.state}\n\n"
        f"📍 Entry: {s.entry_type} @ {s.entry_price:.8f}\n"
        f"🔔 Confirmation: {s.confirmation_condition}\n"
    ) + (
        f"🛑 Stop Loss: {s.stop_loss:.8f}\n"
        f"⚠️ Invalidation: {s.invalidation_price:.8f}\n"
        f"🎯 Take Profit: {s.take_profit:.8f}\n"
        f"TP Model: {s.tp_model}\n"
        f"📐 RR: 1:{s.rr:.2f}\n\n"
        f"Confluence: {', '.join(s.confluences) or '-'}\n"
        f"Reason codes: {', '.join(s.reason_codes) or '-'}\n"
        f"Waiting: {waiting}\n"
        f"Alternative POIs: {alts}\n"
        f"Setup ID: {s.id}"
    )


def _format_exit(s: Setup) -> str:
    icon = "✅" if s.outcome == "TP" else "🛑"
    return f"{icon} SIMULATION {s.outcome}\n{s.symbol} {s.direction}\nR: {s.r_multiple:.2f}\nSetup: {s.id}"


# ---------------- commands ----------------
def _why(symbol: str) -> str:
    a = LAST_ANALYSIS.get(symbol.upper())
    if not a:
        return f"ℹ️ Belum ada analysis untuk {symbol.upper()}. Gunakan /rescan."
    codes = ", ".join(x[0] for x in a.get("components", [])) or "-"
    candidates = a.get("candidates") or []
    lines = [
        f"🔍 WHY {symbol.upper()}",
        "",
        f"Bias: {a.get('bias_label')}",
        f"Base structure: {codes}",
        f"Fib: {'YES' if a.get('fib_ok') else 'NO'}",
        f"H4 align: {'YES' if a.get('h4_align') else 'NO'}",
        f"D1 align: {'YES' if a.get('d1_align') else 'NO'}",
        f"Candidates: {len(candidates)}",
    ]
    for s in candidates:
        lines.extend(["", _format_signal(s)])
    return "\n".join(lines)[:3900]


def handle_command(text: str) -> str | None:
    parts = text.split()
    cmd = parts[0].lower() if parts else ""

    if cmd in {"/setups", "/signals", "/top"}:
        rows = _active_setups()
        if not rows:
            return "📭 Tidak ada active primary thesis saat ini."
        limit = 20 if cmd == "/setups" else 10
        lines = ["🧠 ACTIVE TRADING SIGNALS"]
        for i, s in enumerate(rows[:limit], 1):
            lines.append(
                f"\n{i}. {s.symbol} — {s.direction}\n"
                f"Score {s.score} | {s.model} | {s.state} | RR 1:{s.rr:.2f}\n"
                f"Entry {s.entry_type} {s.entry_price:.8f} | Confirm {s.confirmation_price:.8f} | SL {s.stop_loss:.8f} | TP {s.take_profit:.8f}"
            )
        return "".join(lines)[:3900]

    if cmd == "/setup":
        if len(parts) < 2:
            return "Format: /setup SETUP_ID"
        s = SETUPS.get(parts[1])
        return _format_signal(s) if s else "❌ Setup ID tidak ditemukan."

    if cmd == "/watch":
        if len(parts) < 2:
            return "Format: /watch BTCUSDT"
        rows = _active_setups(parts[1].upper())
        return "\n\n".join(_format_signal(s) for s in rows)[:3900] if rows else f"📭 Tidak ada signal untuk {parts[1].upper()}."

    if cmd == "/why":
        if len(parts) < 2:
            return "Format: /why BTCUSDT"
        return _why(parts[1])

    if cmd == "/strategystatus":
        states: dict[str, int] = {}
        for s in _active_setups():
            states[s.state] = states.get(s.state, 0) + 1
        return (
            "🧠 STRATEGY V4 STATUS\n"
            f"Symbols: {len(API.get_symbols()) if API else 0}\n"
            f"Initial scan: {INITIAL_SCAN_DONE}\n"
            f"Symbols scanned: {COUNTERS['symbols_scanned']}\n"
            f"Event scans: {COUNTERS['event_scans']}\n"
            f"Primary theses created: {COUNTERS['theses_created']}\n"
            f"Theses updated: {COUNTERS['theses_updated']}\n"
            f"Confirmed transitions: {COUNTERS['confirmed']}\n"
            f"Fills: {COUNTERS['fills']}\n"
            f"Wins: {COUNTERS['wins']} | Losses: {COUNTERS['losses']}\n"
            f"Expired: {COUNTERS['expired']} | Invalidated: {COUNTERS['invalidated']}\n"
            f"Active: {_active_count()}\n"
            f"States: {states or '-'}\n"
            f"Min score: {MIN_SCORE} | Min RR: {MIN_RR:.2f}"
        )

    if cmd == "/rescan":
        with LOCK:
            return "\n\n".join(scan_all(initial=False))

    if cmd == "/journal":
        if not JOURNAL:
            return "📓 Simulation journal masih kosong."
        lines = ["📓 SIMULATION JOURNAL"]
        for row in reversed(JOURNAL[-20:]):
            s = row["setup"]
            lines.append(f"{s['symbol']} {s['direction']} | {s['model']} | {s['outcome']} | R={s['r_multiple']}")
        return "\n".join(lines)

    if cmd == "/debug":
        if len(parts) < 2:
            return "Format: /debug BTCUSDT"
        a = LAST_ANALYSIS.get(parts[1].upper())
        return repr(a)[:3900] if a else f"ℹ️ Belum ada snapshot untuk {parts[1].upper()}."

    return None
