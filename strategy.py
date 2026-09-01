from __future__ import annotations

"""SMCAutoTrade strategy_v5.py

Strategy-only engine. No real/simulated order execution lives here.

Responsibilities:
- read market data from start.py DataAPI
- detect HTF context + POI + liquidity + MSS/FVG confluence
- maintain one primary thesis per symbol
- keep candidates that need more confirmation in terminal logs only
- emit CONFIRMED signal objects to start.py through drain_signals()/events
- respect start.py ban state

Source basis: user's combined.txt. Definitions are algorithmic heuristics of the
visual concepts in that source, not claims of a unique canonical SMC definition.
"""

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any

log = logging.getLogger("strategy")

MIN_RR = float(os.getenv("STRAT_V5_MIN_RR", "2.0"))
MIN_SCORE = int(os.getenv("STRAT_V5_MIN_SCORE", "60"))
MAX_TELEGRAM_SETUPS = max(1, int(os.getenv("STRAT_V5_MAX_TELEGRAM_SETUPS", "10")))
SCAN_LOG_EVERY = max(1, int(os.getenv("STRAT_V5_SCAN_LOG_EVERY", "10")))
EXPIRY_MINUTES = max(15, int(os.getenv("STRAT_V5_EXPIRY_MINUTES", "720")))
SWING_LEFT = max(1, int(os.getenv("STRAT_V5_SWING_LEFT", "2")))
SWING_RIGHT = max(1, int(os.getenv("STRAT_V5_SWING_RIGHT", "2")))
SL_ATR_PAD = float(os.getenv("STRAT_V5_SL_ATR_PAD", "0.20"))

API: Any = None
CONTEXT: dict[str, Any] = {}
LOCK = threading.RLock()
INITIAL_SCAN_DONE = False

SETUPS: dict[str, "Setup"] = {}
THESIS_INDEX: dict[str, str] = {}  # symbol -> setup_id
LAST_ANALYSIS: dict[str, dict[str, Any]] = {}
SIGNAL_QUEUE: list[dict[str, Any]] = []

COUNTERS = {
    "symbols_scanned": 0,
    "event_scans": 0,
    "theses_created": 0,
    "theses_updated": 0,
    "confirmed": 0,
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
    fresh: bool = True
    rank_hint: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    confirmation_ts: int | None = None
    signal_emitted: bool = False

    def to_signal(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thesis_key": self.thesis_key,
            "symbol": self.symbol,
            "direction": self.direction,
            "model": self.model,
            "state": self.state,
            "entry_type": self.entry_type,
            "entry_price": self.entry_price,
            "confirmation_price": self.confirmation_price,
            "confirmation_condition": self.confirmation_condition,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "tp_model": self.tp_model,
            "invalidation_price": self.invalidation_price,
            "rr": self.rr,
            "score": self.score,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "expires_ts": self.expires_ts,
            "primary_poi": self.primary_poi.to_dict(),
            "alternative_pois": [p.to_dict() for p in self.alternative_pois],
            "reason_codes": list(self.reason_codes),
            "confluences": list(self.confluences),
            "waiting_for": list(self.waiting_for),
            "thesis": self.thesis,
            "confirmation_ts": self.confirmation_ts,
        }


# ---------------- math / market helpers ----------------
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
        h = float(candles[i]["high"])
        l = float(candles[i]["low"])
        if all(h > float(candles[j]["high"]) for j in range(i - SWING_LEFT, i)) and all(
            h >= float(candles[j]["high"]) for j in range(i + 1, i + SWING_RIGHT + 1)
        ):
            highs.append((i, h))
        if all(l < float(candles[j]["low"]) for j in range(i - SWING_LEFT, i)) and all(
            l <= float(candles[j]["low"]) for j in range(i + 1, i + SWING_RIGHT + 1)
        ):
            lows.append((i, l))
    return highs, lows


def _fvg(candles: list[dict[str, Any]]) -> list[POI]:
    out: list[POI] = []
    for i in range(2, len(candles)):
        a, c = candles[i - 2], candles[i]
        if float(c["low"]) > float(a["high"]):
            out.append(POI(f"FVG-L-{i}", "FVG", "LONG", float(a["high"]), float(c["low"]), i, True, i))
        elif float(c["high"]) < float(a["low"]):
            out.append(POI(f"FVG-S-{i}", "FVG", "SHORT", float(c["high"]), float(a["low"]), i, True, i))
    return out


def _fvg_fresh(candles: list[dict[str, Any]], poi: POI) -> bool:
    for c in candles[poi.created_index + 1 :]:
        if poi.direction == "LONG" and float(c["low"]) <= poi.high and float(c["high"]) >= poi.low:
            return False
        if poi.direction == "SHORT" and float(c["high"]) >= poi.low and float(c["low"]) <= poi.high:
            return False
    return True


def _order_blocks(candles: list[dict[str, Any]]) -> list[POI]:
    atr = _atr(candles, 14) or 0.0
    out: list[POI] = []
    seen: set[str] = set()
    for i in range(2, len(candles)):
        prev, cur = candles[i - 1], candles[i]
        body = abs(float(prev["close"]) - float(prev["open"]))
        if float(cur["close"]) > float(prev["high"]) and float(prev["close"]) < float(prev["open"]) and (atr == 0 or body >= atr * 0.10):
            p = POI(f"OB-L-{i-1}", "OB", "LONG", float(prev["low"]), float(prev["open"]), i - 1, True, i)
            if p.poi_id not in seen:
                out.append(p); seen.add(p.poi_id)
        elif float(cur["close"]) < float(prev["low"]) and float(prev["close"]) > float(prev["open"]) and (atr == 0 or body >= atr * 0.10):
            p = POI(f"OB-S-{i-1}", "OB", "SHORT", float(prev["open"]), float(prev["high"]), i - 1, True, i)
            if p.poi_id not in seen:
                out.append(p); seen.add(p.poi_id)
    return out


def _trend(c1h: list[dict[str, Any]]) -> dict[str, Any]:
    if len(c1h) < 30:
        return {"bias": "NEUTRAL", "components": [], "score": 0}
    closes = [float(c["close"]) for c in c1h]
    ema9, ema20 = _ema(closes, 9), _ema(closes, 20)
    highs, lows = _swings(c1h)
    if ema9 is None or ema20 is None:
        return {"bias": "NEUTRAL", "components": [], "score": 0}
    if ema9 > ema20:
        comps = [("HTF_EMA_BULL", 20, "1H EMA9 > EMA20")]
        if len(highs) >= 2 and len(lows) >= 2 and highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
            comps.append(("HTF_HH_HL", 15, "1H HH + HL"))
        return {"bias": "BULL", "components": comps, "score": sum(x[1] for x in comps)}
    if ema9 < ema20:
        comps = [("HTF_EMA_BEAR", 20, "1H EMA9 < EMA20")]
        if len(highs) >= 2 and len(lows) >= 2 and highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
            comps.append(("HTF_LH_LL", 15, "1H LH + LL"))
        return {"bias": "BEAR", "components": comps, "score": sum(x[1] for x in comps)}
    return {"bias": "NEUTRAL", "components": [], "score": 0}


def _fib(c1h: list[dict[str, Any]], direction: str) -> tuple[bool, str, str]:
    highs, lows = _swings(c1h)
    if not highs or not lows:
        return False, "", ""
    if direction == "LONG":
        hi_i, hi = highs[-1]
        lows_before = [x for x in lows if x[0] < hi_i]
        if not lows_before:
            return False, "", ""
        lo = lows_before[-1][1]
        level = hi - 0.618 * (hi - lo)
        if float(c1h[-1]["close"]) <= level:
            return True, "FIB_DISCOUNT", "1H price in 0.618 discount"
    else:
        lo_i, lo = lows[-1]
        highs_before = [x for x in highs if x[0] < lo_i]
        if not highs_before:
            return False, "", ""
        hi = highs_before[-1][1]
        level = lo + 0.618 * (hi - lo)
        if float(c1h[-1]["close"]) >= level:
            return True, "FIB_PREMIUM", "1H price in 0.618 premium"
    return False, "", ""


def _micro(c5: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    closed = c5[:-1] if c5 and not c5[-1].get("confirmed", True) else c5
    if len(closed) < 20:
        return {"sweep": False, "mss": False, "level": None, "codes": [], "labels": []}
    highs, lows = _swings(closed[:-1])
    last = closed[-1]
    sweep = False
    sweep_code = ""
    level = None
    if direction == "LONG" and lows:
        level = lows[-1][1]
        sweep = float(last["low"]) < level and float(last["close"]) > level
        sweep_code = "SSL_SWEEP"
    elif direction == "SHORT" and highs:
        level = highs[-1][1]
        sweep = float(last["high"]) > level and float(last["close"]) < level
        sweep_code = "BSL_SWEEP"
    mss = False
    mss_code = ""
    mss_level = None
    if direction == "LONG" and highs:
        mss_level = highs[-1][1]
        mss = float(last["close"]) > mss_level
        mss_code = "MSS_BULL"
    elif direction == "SHORT" and lows:
        mss_level = lows[-1][1]
        mss = float(last["close"]) < mss_level
        mss_code = "MSS_BEAR"
    codes = [x for x in (sweep_code if sweep else "", mss_code if mss else "") if x]
    labels = ["5M liquidity sweep" if sweep else "", "5M MSS/ChoCH" if mss else ""]
    labels = [x for x in labels if x]
    return {"sweep": sweep, "mss": mss, "level": mss_level or level, "codes": codes, "labels": labels}


def _in_zone(price: float, poi: POI, atr: float) -> bool:
    pad = atr * 0.20
    return poi.low - pad <= price <= poi.high + pad


def _score(components: list[tuple[str, float, str]]) -> int:
    max_possible = 100.0
    total = sum(max(0.0, x[1]) for x in components)
    return max(0, min(100, round((total / max_possible) * 100.0)))


def _build_setup(symbol: str, direction: str, primary: POI, alternatives: list[POI], a: dict[str, Any], now_ts: int) -> Setup | None:
    price = float(a["price"])
    atr = float(a["atr"])
    entry = (primary.low + primary.high) / 2.0
    pad = max(atr * SL_ATR_PAD, (primary.high - primary.low) * 0.15)
    if direction == "LONG":
        stop_loss = primary.low - pad
        risk = entry - stop_loss
        take_profit = entry + MIN_RR * risk
    else:
        stop_loss = primary.high + pad
        risk = stop_loss - entry
        take_profit = entry - MIN_RR * risk
    if risk <= 0:
        return None

    components = list(a["components"])
    codes = [x[0] for x in components]
    labels = [x[2] for x in components]
    waiting: list[str] = []
    components.append(("POI", 15, f"H1 {primary.model} primary zone"))
    codes.append("ORDER_BLOCK" if primary.model == "OB" else "FRESH_FVG")
    labels.append(f"H1 {primary.model} primary zone")

    if a["fib_ok"]:
        components.append((a["fib_code"], 12, a["fib_label"]))
        codes.append(a["fib_code"]); labels.append(a["fib_label"])
    else:
        waiting.append("0.618 Fibonacci location")
    if a["h4_align"]:
        components.append(("H4_ALIGN", 8, "4H bias aligned")); codes.append("H4_ALIGN"); labels.append("4H bias aligned")
    if a["d1_align"]:
        components.append(("D1_ALIGN", 5, "1D bias aligned")); codes.append("D1_ALIGN"); labels.append("1D bias aligned")

    micro = a["micro"]
    if micro["sweep"]:
        components.append((micro["codes"][0], 15, "5M liquidity sweep")); codes.append(micro["codes"][0]); labels.append("5M liquidity sweep")
    else:
        waiting.append("5M liquidity sweep")
    if micro["mss"]:
        mcode = micro["codes"][-1]; components.append((mcode, 18, "5M MSS/ChoCH")); codes.append(mcode); labels.append("5M MSS/ChoCH")
    else:
        waiting.append("5M MSS/ChoCH")

    in_zone = _in_zone(price, primary, atr)
    if not in_zone:
        waiting.append("price returns to primary POI")

    state = "WATCHING"
    if in_zone:
        state = "WAITING_CONFIRMATION" if waiting else "CONFIRMED"

    conf_level = micro["level"]
    if conf_level is not None:
        condition = f"5M close {'above' if direction == 'LONG' else 'below'} {conf_level:.8f}"
    else:
        condition = f"5M liquidity sweep + {'bullish' if direction == 'LONG' else 'bearish'} MSS"

    score = _score(components)
    if score < MIN_SCORE:
        return None

    return Setup(
        id=f"S5-{symbol}-{uuid.uuid4().hex[:10]}",
        thesis_key=symbol,
        symbol=symbol,
        direction=direction,
        state=state,
        model=primary.model,
        entry_type="LIMIT",
        entry_price=entry,
        confirmation_price=conf_level,
        confirmation_condition=condition,
        stop_loss=stop_loss,
        take_profit=take_profit,
        tp_model="FIXED_RR",
        invalidation_price=primary.low if direction == "LONG" else primary.high,
        rr=abs(take_profit - entry) / max(abs(entry - stop_loss), 1e-12),
        score=score,
        created_ts=now_ts,
        updated_ts=now_ts,
        expires_ts=now_ts + EXPIRY_MINUTES * 60_000,
        primary_poi=primary,
        alternative_pois=alternatives,
        reason_codes=list(dict.fromkeys(codes)),
        confluences=list(dict.fromkeys(labels)),
        waiting_for=list(dict.fromkeys(waiting)),
        thesis=(f"{direction} thesis from 1H context + {primary.model} POI; "
                "entry requires the stated lower-timeframe confirmation."),
    )


def _active(symbol: str | None = None) -> list[Setup]:
    states = {"WATCHING", "IN_ZONE", "WAITING_CONFIRMATION", "CONFIRMED"}
    rows = [s for s in SETUPS.values() if s.state in states]
    if symbol:
        rows = [s for s in rows if s.symbol == symbol.upper()]
    return sorted(rows, key=lambda s: (-s.score, s.symbol, s.direction))


def _active_thesis(symbol: str) -> Setup | None:
    sid = THESIS_INDEX.get(symbol.upper())
    if not sid:
        return None
    s = SETUPS.get(sid)
    if not s or s.state in {"EXPIRED", "INVALIDATED", "CLOSED"}:
        return None
    return s


def _queue_signal(s: Setup, now_ts: int) -> None:
    if s.signal_emitted:
        return
    s.state = "CONFIRMED"
    s.confirmation_ts = now_ts
    s.updated_ts = now_ts
    s.signal_emitted = True
    COUNTERS["confirmed"] += 1
    payload = s.to_signal()
    SIGNAL_QUEUE.append(payload)
    log.warning(
        "[SIGNAL CONFIRMED] %s %s model=%s score=%d ENTRY=%s @ %.8f SL=%.8f TP=%.8f RR=%.2f",
        s.symbol, s.direction, s.model, s.score, s.entry_type,
        s.entry_price, s.stop_loss, s.take_profit, s.rr,
    )


def _merge(existing: Setup, incoming: Setup) -> None:
    old_state = existing.state
    if incoming.score >= existing.score:
        existing.primary_poi = incoming.primary_poi
        existing.model = incoming.model
        existing.entry_price = incoming.entry_price
        existing.stop_loss = incoming.stop_loss
        existing.take_profit = incoming.take_profit
        existing.rr = incoming.rr
        existing.score = incoming.score
    existing.updated_ts = max(existing.updated_ts, incoming.updated_ts)
    existing.reason_codes = list(dict.fromkeys(existing.reason_codes + incoming.reason_codes))
    existing.confluences = list(dict.fromkeys(existing.confluences + incoming.confluences))
    existing.waiting_for = list(incoming.waiting_for)
    existing.confirmation_price = incoming.confirmation_price
    existing.confirmation_condition = incoming.confirmation_condition
    existing.invalidation_price = incoming.invalidation_price
    existing.alternative_pois = incoming.alternative_pois[:5]

    if existing.state not in {"CONFIRMED", "CLOSED"}:
        if incoming.state == "CONFIRMED":
            existing.state = "CONFIRMED"
        elif incoming.state == "WAITING_CONFIRMATION":
            existing.state = "WAITING_CONFIRMATION"
        elif incoming.state == "WATCHING":
            existing.state = "WATCHING"
    if old_state != existing.state:
        log.info("[STATE] %s %s %s -> %s | score=%d waiting=%s", existing.symbol, existing.direction, old_state, existing.state, existing.score, ",".join(existing.waiting_for) or "-")
    if old_state != "CONFIRMED" and existing.state == "CONFIRMED":
        _queue_signal(existing, incoming.updated_ts)
        COUNTERS["theses_updated"] += 1


def _register(setup: Setup) -> None:
    existing = _active_thesis(setup.symbol)
    if existing:
        if existing.direction == setup.direction:
            _merge(existing, setup)
            return
        if setup.score >= existing.score + 10 and existing.state != "CONFIRMED":
            existing.state = "INVALIDATED"
            existing.reason_codes.append("OPPOSITE_THESIS_REPLACED")
            COUNTERS["invalidated"] += 1
            THESIS_INDEX.pop(setup.symbol, None)
        else:
            return

    SETUPS[setup.id] = setup
    THESIS_INDEX[setup.symbol] = setup.id
    COUNTERS["theses_created"] += 1
    if setup.state == "CONFIRMED":
        _queue_signal(setup, setup.updated_ts)
    else:
        missing = ", ".join(setup.waiting_for) or "confirmation"
        log.info(
            "[WATCH] %s %s model=%s score=%d state=%s missing=%s entry=%.8f sl=%.8f tp=%.8f",
            setup.symbol, setup.direction, setup.model, setup.score, setup.state,
            missing, setup.entry_price, setup.stop_loss, setup.take_profit,
        )


def _analysis(symbol: str, event_tf: str | None = None) -> dict[str, Any]:
    symbol = symbol.upper()
    if API is not None and hasattr(API, "is_symbol_banned"):
        try:
            if API.is_symbol_banned(symbol):
                return {"symbol": symbol, "bias": "BANNED", "candidates": [], "labels": ["symbol banned"]}
        except Exception:
            log.exception("[BANNED CHECK] failed for %s", symbol)

    c15 = API.get_candles(symbol, "15", 700)
    c5 = API.get_candles(symbol, "5", 500)
    c1 = API.get_candles(symbol, "1", 500)
    price = float(API.get_price(symbol) or (c1[-1]["close"] if c1 else c15[-1]["close"] if c15 else 0.0))
    result: dict[str, Any] = {
        "symbol": symbol, "price": price, "bias": "NEUTRAL", "bias_label": "HTF neutral",
        "atr": _atr(c5, 14) or max(price * 0.001, 1e-9), "components": [],
        "fib_ok": False, "fib_code": "", "fib_label": "", "h4_align": False, "d1_align": False,
        "micro": {"sweep": False, "mss": False, "level": None, "codes": [], "labels": []},
        "candidates": [], "event_tf": event_tf,
    }
    if len(c15) < 120 or len(c5) < 80 or len(c1) < 80:
        result["labels"] = ["insufficient history"]
        return result

    h1, h4, d1 = _aggregate(c15, 60), _aggregate(c15, 240), _aggregate(c15, 1440)
    if len(h1) < 30 or len(h4) < 8:
        result["labels"] = ["insufficient derived HTF history"]
        return result

    t1, t4 = _trend(h1), _trend(h4)
    td = _trend(d1) if len(d1) >= 30 else {"bias": "NEUTRAL", "components": []}
    result["bias"] = t1["bias"]
    result["bias_label"] = {"BULL": "1H bullish", "BEAR": "1H bearish"}.get(t1["bias"], "HTF neutral")
    result["components"] = list(t1["components"])
    result["h4_align"] = t1["bias"] != "NEUTRAL" and t4["bias"] == t1["bias"]
    result["d1_align"] = t1["bias"] != "NEUTRAL" and td.get("bias") == t1["bias"]
    if t1["bias"] == "NEUTRAL":
        return result

    direction = "LONG" if t1["bias"] == "BULL" else "SHORT"
    result["fib_ok"], result["fib_code"], result["fib_label"] = _fib(h1, direction)
    result["micro"] = _micro(c5, direction)

    pois: list[POI] = []
    for p in _fvg(h1):
        if p.direction == direction and _fvg_fresh(h1, p):
            pois.append(p)
    for p in _order_blocks(h1):
        if p.direction == direction:
            pois.append(p)
    pois.sort(key=lambda p: p.created_index, reverse=True)
    if not pois:
        return result

    setup = _build_setup(symbol, direction, pois[0], pois[1:5], result, int(c15[-1]["timestamp"]))
    if setup:
        result["candidates"] = [setup]
    return result


# ---------------- lifecycle ----------------
def initialize(api: Any, context: dict[str, Any]) -> None:
    global API, CONTEXT, INITIAL_SCAN_DONE
    API = api
    CONTEXT = dict(context)
    INITIAL_SCAN_DONE = False
    log.info("[STRATEGY V5] initialized | min_score=%d min_rr=%.2f expiry=%dm", MIN_SCORE, MIN_RR, EXPIRY_MINUTES)


def shutdown() -> None:
    log.info("[STRATEGY V5] shutdown | active=%d queued_signals=%d", len(_active()), len(SIGNAL_QUEUE))


def scan_all(initial: bool = False) -> list[str]:
    if API is None or not API.is_bootstrap_complete():
        return ["ℹ️ Strategy menunggu historical data lengkap."]
    symbols = API.get_symbols()
    total = len(symbols)
    no_candidate = 0
    banned = 0
    created_before = COUNTERS["theses_created"]
    log.info("[SCAN] %s start | symbols=%d", "INITIAL" if initial else "FULL", total)

    for idx, symbol in enumerate(symbols, 1):
        try:
            analysis = _analysis(symbol)
            LAST_ANALYSIS[symbol] = analysis
            if analysis.get("bias") == "BANNED":
                banned += 1
            if not analysis.get("candidates"):
                no_candidate += 1
            for setup in analysis.get("candidates", []):
                _register(setup)
        except Exception:
            log.exception("[SCAN] %s failed", symbol)
            no_candidate += 1
        COUNTERS["symbols_scanned"] += 1
        if idx == 1 or idx % SCAN_LOG_EVERY == 0 or idx == total:
            log.info(
                "[SCAN] progress %d/%d | new=%d no_candidate=%d banned=%d active=%d queued=%d",
                idx, total, COUNTERS["theses_created"] - created_before, no_candidate, banned,
                len(_active()), len(SIGNAL_QUEUE),
            )

    global INITIAL_SCAN_DONE
    INITIAL_SCAN_DONE = True
    active = _active()
    lines = [
        "🔎 INITIAL STRATEGY SCAN COMPLETE" if initial else "🔎 STRATEGY SCAN COMPLETE",
        "",
        f"Pairs scanned: {total}",
        f"Primary theses active: {len(active)}",
        f"New confirmed signals queued: {len(SIGNAL_QUEUE)}",
        f"Pairs without candidate: {no_candidate}",
        f"Banned pairs skipped: {banned}",
    ]
    if active:
        lines += ["", "Top primary theses (confirmation-needed items stay in terminal):"]
        for i, s in enumerate(active[:MAX_TELEGRAM_SETUPS], 1):
            lines.append(f"{i}. {s.symbol} {s.direction} | {s.model} | {s.state} | score={s.score} | RR={s.rr:.2f}")
    else:
        lines += ["", "No active thesis met the current filter."]
    return ["\n".join(lines)]


def on_data_ready() -> str | None:
    return "\n\n".join(scan_all(initial=True))


def drain_signals() -> list[dict[str, Any]]:
    with LOCK:
        out = list(SIGNAL_QUEUE)
        SIGNAL_QUEUE.clear()
        return out


def get_active_signals() -> list[dict[str, Any]]:
    return [s.to_signal() for s in _active()]


def on_market_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if API is None or event.get("type") != "candle":
        return None
    symbol = str(event.get("symbol") or "").upper()
    tf = str(event.get("timeframe") or "")
    candle = event.get("candle") or {}
    if not symbol or tf not in {"1", "5", "15"} or not candle.get("confirmed", False):
        return None

    with LOCK:
        analysis = _analysis(symbol, tf)
        LAST_ANALYSIS[symbol] = analysis
        COUNTERS["event_scans"] += 1
        incoming = analysis.get("candidates", [])
        if not incoming:
            return None
        setup = incoming[0]
        existing = _active_thesis(symbol)
        if existing:
            old_emitted = existing.signal_emitted
            _merge(existing, setup)
            if existing.signal_emitted and not old_emitted:
                return {"type": "signal", "signal": existing.to_signal()}
        else:
            _register(setup)
            queued = drain_signals()
            if queued:
                return {"type": "signal", "signal": queued[0]}
    return None


# ---------------- terminal/Telegram helper commands ----------------
def _format_signal(s: dict[str, Any] | Setup) -> str:
    d = s if isinstance(s, dict) else s.to_signal()
    waiting = ", ".join(d.get("waiting_for") or []) or "-"
    return (
        f"{d['symbol']} — {d['direction']}\n"
        f"Model: {d['model']} | State: {d['state']} | Score: {d['score']}/100\n"
        f"Entry: {d['entry_type']} @ {d['entry_price']:.8f}\n"
        f"Confirmation: {d.get('confirmation_condition') or '-'}\n"
        f"SL: {d['stop_loss']:.8f}\n"
        f"TP: {d['take_profit']:.8f}\n"
        f"RR: 1:{d['rr']:.2f}\n"
        f"Waiting: {waiting}\n"
        f"Confluence: {', '.join(d.get('confluences') or []) or '-'}\n"
        f"ID: {d['id']}"
    )


def _why(symbol: str) -> str:
    a = LAST_ANALYSIS.get(symbol.upper())
    if not a:
        return f"ℹ️ Belum ada snapshot {symbol.upper()}."
    return (
        f"🔍 WHY {symbol.upper()}\n\n"
        f"Bias: {a.get('bias_label')}\n"
        f"Fib: {'YES' if a.get('fib_ok') else 'NO'}\n"
        f"H4 align: {'YES' if a.get('h4_align') else 'NO'}\n"
        f"D1 align: {'YES' if a.get('d1_align') else 'NO'}\n"
        f"Micro sweep: {'YES' if a.get('micro', {}).get('sweep') else 'NO'}\n"
        f"Micro MSS: {'YES' if a.get('micro', {}).get('mss') else 'NO'}\n"
        f"Candidates: {len(a.get('candidates') or [])}\n\n"
        + ("\n\n".join(_format_signal(x) for x in a.get("candidates", [])) or "No candidate.")
    )[:3900]


def handle_command(text: str) -> str | None:
    parts = text.split()
    cmd = parts[0].lower() if parts else ""
    if cmd in {"/setups", "/signals", "/top"}:
        rows = get_active_signals()
        if not rows:
            return "📭 Tidak ada active strategy thesis. Confirmation-needed detail ada di terminal."
        lines = ["🧠 ACTIVE STRATEGY THESES"]
        for i, d in enumerate(rows[:20], 1):
            lines.append(
                f"{i}. {d['symbol']} {d['direction']} | {d['model']} | {d['state']} | "
                f"score={d['score']} | Entry={d['entry_price']:.8f} | SL={d['stop_loss']:.8f} | TP={d['take_profit']:.8f}"
            )
        return "\n".join(lines)[:3900]
    if cmd == "/why":
        return _why(parts[1].upper()) if len(parts) > 1 else "Format: /why BTCUSDT"
    if cmd == "/strategystatus":
        states: dict[str, int] = {}
        for d in get_active_signals():
            states[d["state"]] = states.get(d["state"], 0) + 1
        return (
            "🧠 STRATEGY V5 STATUS\n"
            f"Symbols: {len(API.get_symbols()) if API else 0}\n"
            f"Initial scan: {INITIAL_SCAN_DONE}\n"
            f"Event scans: {COUNTERS['event_scans']}\n"
            f"Theses created: {COUNTERS['theses_created']}\n"
            f"Confirmed signals: {COUNTERS['confirmed']}\n"
            f"Invalidated: {COUNTERS['invalidated']}\n"
            f"Queued signals: {len(SIGNAL_QUEUE)}\n"
            f"Active: {len(get_active_signals())}\n"
            f"States: {states or '-'}\n"
            f"Min score: {MIN_SCORE} | Min RR: {MIN_RR:.2f}"
        )
    if cmd == "/rescan":
        return "\n\n".join(scan_all(initial=False))
    return None
