from __future__ import annotations

"""
SMCAutoTrade - strategy.py

Simulation-only strategy engine.

Design:
- Consumes market data exposed by start.py / DataEngine.
- Builds higher timeframes from the 15M base data:
    15M -> H1 -> H4 -> D1
- Does NOT fabricate M5/M1 candles. Those cannot be reconstructed faithfully
  from 15M OHLCV.
- Scans ALL available symbols; it does not force one setup per symbol.
- Produces simulated setups only. No real orders are sent.
- Maintains state so a setup can WAIT for a future confirmation / retest.
- Designed to be attached to start.py through context["data_engine"].

Expected data engine interface:
    get_symbols() -> list[str]
    get_candles(symbol, limit=None) -> list[dict]
    get_price(symbol) -> float | None

Optional context:
    chat_id
    send_message(chat_id, text)
    data_engine

Strategy source basis:
- trend strength / trade with dominant direction
- multi-timeframe context
- order block + FVG
- Fibonacci discount / premium, especially 0.618+
- liquidity sweep / inducement
- market structure shift / change of character
- confluence
- fixed RR 1:2 or 1:3
"""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_TF_MINUTES = 15
H1_FACTOR = 4
H4_FACTOR = 16
D1_FACTOR = 96

MIN_BASE_CANDLES = max(96, int(os.getenv("STRATEGY_MIN_CANDLES", "180")))
SCAN_INTERVAL_SECONDS = max(
    0.5, float(os.getenv("STRATEGY_SCAN_INTERVAL", "2.0"))
)
MAX_ACTIVE_SETUP_PER_SYMBOL = max(
    1, int(os.getenv("STRATEGY_MAX_SETUP_PER_SYMBOL", "2"))
)
MAX_TOTAL_ACTIVE_SETUPS = max(
    1, int(os.getenv("STRATEGY_MAX_ACTIVE_SETUPS", "100"))
)
SETUP_EXPIRY_CANDLES = max(4, int(os.getenv("STRATEGY_SETUP_EXPIRY_CANDLES", "32")))

SWING_LEFT = max(1, int(os.getenv("STRATEGY_SWING_LEFT", "2")))
SWING_RIGHT = max(1, int(os.getenv("STRATEGY_SWING_RIGHT", "2")))

MIN_SCORE = max(1, int(os.getenv("STRATEGY_MIN_SCORE", "5")))
MIN_RR = float(os.getenv("STRATEGY_MIN_RR", "2.0"))
DEFAULT_RR = max(MIN_RR, float(os.getenv("STRATEGY_DEFAULT_RR", "2.0")))

# Confluence thresholds.
STRONG_BODY_RATIO = max(0.1, float(os.getenv("STRATEGY_STRONG_BODY_RATIO", "0.60")))
ATR_MULTIPLIER = max(0.1, float(os.getenv("STRATEGY_ATR_MULTIPLIER", "1.20")))
ZONE_TOUCH_EPSILON = max(0.0, float(os.getenv("STRATEGY_ZONE_EPSILON", "0.0002")))

LOG_SUMMARY_SECONDS = max(
    10.0, float(os.getenv("STRATEGY_LOG_SUMMARY_SECONDS", "30"))
)

# Conservative simulation constraints.
SIM_MAX_RISK_FRACTION = max(
    0.0001, float(os.getenv("STRATEGY_SIM_RISK", "0.005"))
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=(os.getenv("LOG_LEVEL") or "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("strategy")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class Bar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(0.0, self.high - self.low)

    @property
    def body_ratio(self) -> float:
        return self.body / self.range if self.range > 0 else 0.0


@dataclass(slots=True)
class Zone:
    kind: str
    low: float
    high: float
    created_at: int
    source_tf: str
    reference_index: int
    fresh: bool = True

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(slots=True)
class Setup:
    setup_id: str
    symbol: str
    direction: str
    family: str
    status: str
    score: float
    created_at: int
    expiry_at: int | None

    source_tf: str
    confirmation_tf: str

    entry_type: str
    entry_low: float
    entry_high: float
    entry_price: float

    stop_loss: float
    take_profit: float
    risk_distance: float
    rr: float

    reasons: list[str] = field(default_factory=list)
    confluences: list[str] = field(default_factory=list)

    origin_bar: int = 0
    last_seen_bar: int = 0
    trigger_bar: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "family": self.family,
            "status": self.status,
            "score": round(self.score, 2),
            "created_at": self.created_at,
            "expiry_at": self.expiry_at,
            "source_tf": self.source_tf,
            "confirmation_tf": self.confirmation_tf,
            "entry_type": self.entry_type,
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_distance": self.risk_distance,
            "rr": round(self.rr, 3),
            "reasons": list(self.reasons),
            "confluences": list(self.confluences),
            "origin_bar": self.origin_bar,
            "last_seen_bar": self.last_seen_bar,
            "trigger_bar": self.trigger_bar,
        }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _to_bar(raw: dict[str, Any]) -> Bar | None:
    try:
        return Bar(
            timestamp=int(raw["timestamp"]),
            open=float(raw["open"]),
            high=float(raw["high"]),
            low=float(raw["low"]),
            close=float(raw["close"]),
            volume=float(raw.get("volume") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def normalize_bars(rows: Iterable[dict[str, Any]]) -> list[Bar]:
    bars: list[Bar] = []
    for row in rows:
        bar = _to_bar(row)
        if bar is not None:
            bars.append(bar)

    bars.sort(key=lambda x: x.timestamp)

    # Deduplicate by timestamp, keeping the last version.
    dedup: dict[int, Bar] = {}
    for bar in bars:
        dedup[bar.timestamp] = bar

    return [dedup[k] for k in sorted(dedup)]


def aggregate_bars(bars: list[Bar], factor: int) -> list[Bar]:
    """
    Aggregate contiguous base bars into a higher timeframe.

    This is valid for higher timeframes because 15M divides evenly into H1/H4/D1.
    """
    if factor <= 1:
        return list(bars)

    grouped: list[Bar] = []
    bucket: list[Bar] = []

    expected_step_ms = BASE_TF_MINUTES * 60_000

    for bar in bars:
        if bucket:
            previous = bucket[-1]
            continuous = bar.timestamp - previous.timestamp == expected_step_ms
            if not continuous:
                if bucket:
                    grouped.append(
                        Bar(
                            timestamp=bucket[0].timestamp,
                            open=bucket[0].open,
                            high=max(x.high for x in bucket),
                            low=min(x.low for x in bucket),
                            close=bucket[-1].close,
                            volume=sum(x.volume for x in bucket),
                        )
                    )
                bucket = []

        bucket.append(bar)

        if len(bucket) == factor:
            grouped.append(
                Bar(
                    timestamp=bucket[0].timestamp,
                    open=bucket[0].open,
                    high=max(x.high for x in bucket),
                    low=min(x.low for x in bucket),
                    close=bucket[-1].close,
                    volume=sum(x.volume for x in bucket),
                )
            )
            bucket = []

    # Do not append an incomplete higher-timeframe bar.
    return grouped


def ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period harus > 0")

    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result

    sma = sum(values[:period]) / period
    result[period - 1] = sma
    alpha = 2.0 / (period + 1.0)
    prev = sma

    for i in range(period, len(values)):
        prev = (values[i] - prev) * alpha + prev
        result[i] = prev

    return result


def atr(bars: list[Bar], period: int = 14) -> list[float | None]:
    if len(bars) < period:
        return [None] * len(bars)

    trs: list[float] = []
    previous_close: float | None = None

    for bar in bars:
        if previous_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - previous_close),
                abs(bar.low - previous_close),
            )
        trs.append(tr)
        previous_close = bar.close

    out: list[float | None] = [None] * len(bars)
    first = sum(trs[:period]) / period
    out[period - 1] = first
    prev = first

    # Wilder-style smoothing.
    for i in range(period, len(trs)):
        prev = ((prev * (period - 1)) + trs[i]) / period
        out[i] = prev

    return out


def find_swings(
    bars: list[Bar],
    left: int = SWING_LEFT,
    right: int = SWING_RIGHT,
) -> tuple[list[int], list[int]]:
    highs: list[int] = []
    lows: list[int] = []

    if len(bars) < left + right + 1:
        return highs, lows

    for i in range(left, len(bars) - right):
        high = bars[i].high
        low = bars[i].low

        left_highs = all(high >= bars[j].high for j in range(i - left, i))
        right_highs = all(high >= bars[j].high for j in range(i + 1, i + right + 1))

        left_lows = all(low <= bars[j].low for j in range(i - left, i))
        right_lows = all(low <= bars[j].low for j in range(i + 1, i + right + 1))

        if left_highs and right_highs:
            highs.append(i)
        if left_lows and right_lows:
            lows.append(i)

    return highs, lows


def market_structure(bars: list[Bar]) -> dict[str, Any]:
    high_idx, low_idx = find_swings(bars)

    trend = "neutral"
    structure_break = False
    last_swing_high = bars[high_idx[-1]].high if high_idx else None
    last_swing_low = bars[low_idx[-1]].low if low_idx else None

    if len(high_idx) >= 2 and len(low_idx) >= 2:
        h1, h2 = bars[high_idx[-2]], bars[high_idx[-1]]
        l1, l2 = bars[low_idx[-2]], bars[low_idx[-1]]

        if h2.high > h1.high and l2.low > l1.low:
            trend = "bullish"
        elif h2.high < h1.high and l2.low < l1.low:
            trend = "bearish"

        # Latest closed bar breaking latest confirmed swing.
        last = bars[-1]
        if trend == "bullish" and last_swing_high is not None and last.close > last_swing_high:
            structure_break = True
        elif trend == "bearish" and last_swing_low is not None and last.close < last_swing_low:
            structure_break = True

    return {
        "trend": trend,
        "high_idx": high_idx,
        "low_idx": low_idx,
        "last_swing_high": last_swing_high,
        "last_swing_low": last_swing_low,
        "structure_break": structure_break,
    }


def trend_strength(bars: list[Bar]) -> dict[str, Any]:
    if len(bars) < 25:
        return {"direction": "neutral", "score": 0.0, "ema9": None, "ema20": None}

    closes = [b.close for b in bars]
    e9 = ema(closes, 9)[-1]
    e20 = ema(closes, 20)[-1]

    if e9 is None or e20 is None:
        return {"direction": "neutral", "score": 0.0, "ema9": e9, "ema20": e20}

    direction = "bullish" if e9 > e20 else "bearish" if e9 < e20 else "neutral"

    # Directional strength is based on the rate of change over recent bars,
    # consistent with the source's price-over-time idea.
    lookback = min(12, len(bars) - 1)
    start = bars[-1 - lookback].close
    end = bars[-1].close
    roc = (end - start) / start if start else 0.0

    same_direction = (
        (direction == "bullish" and roc > 0)
        or (direction == "bearish" and roc < 0)
    )
    score = abs(roc) * 1000.0
    if same_direction:
        score *= 1.25

    return {
        "direction": direction,
        "score": score,
        "ema9": e9,
        "ema20": e20,
        "roc": roc,
    }


def detect_fvgs(bars: list[Bar], limit: int = 8) -> list[Zone]:
    zones: list[Zone] = []
    start = max(2, len(bars) - 80)

    for i in range(start, len(bars)):
        a = bars[i - 2]
        b = bars[i - 1]
        c = bars[i]

        # Bullish FVG: current low above two-bars-back high.
        if c.low > a.high and b.bullish:
            zones.append(
                Zone(
                    kind="bullish_fvg",
                    low=a.high,
                    high=c.low,
                    created_at=c.timestamp,
                    source_tf="TF",
                    reference_index=i,
                )
            )

        # Bearish FVG: current high below two-bars-back low.
        if c.high < a.low and b.bearish:
            zones.append(
                Zone(
                    kind="bearish_fvg",
                    low=c.high,
                    high=a.low,
                    created_at=c.timestamp,
                    source_tf="TF",
                    reference_index=i,
                )
            )

    return zones[-limit:]


def detect_order_blocks(bars: list[Bar], limit: int = 8) -> list[Zone]:
    """
    Heuristic order-block detector based on:
    - a final opposite candle
    - followed by a strong displacement
    - that closes through a recent swing.
    """
    zones: list[Zone] = []
    swing_highs, swing_lows = find_swings(bars)

    recent_high_prices = {
        idx: bars[idx].high for idx in swing_highs[-8:]
    }
    recent_low_prices = {
        idx: bars[idx].low for idx in swing_lows[-8:]
    }

    for i in range(max(3, len(bars) - 100), len(bars)):
        if i >= len(bars) - 1:
            continue

        base = bars[i]
        impulse = bars[i + 1]

        recent_high = max(recent_high_prices.values(), default=None)
        recent_low = min(recent_low_prices.values(), default=None)

        if (
            base.bearish
            and impulse.bullish
            and impulse.body_ratio >= STRONG_BODY_RATIO
            and recent_high is not None
            and impulse.close > recent_high
        ):
            zones.append(
                Zone(
                    kind="bullish_ob",
                    low=base.low,
                    high=max(base.open, base.close),
                    created_at=base.timestamp,
                    source_tf="TF",
                    reference_index=i,
                )
            )

        if (
            base.bullish
            and impulse.bearish
            and impulse.body_ratio >= STRONG_BODY_RATIO
            and recent_low is not None
            and impulse.close < recent_low
        ):
            zones.append(
                Zone(
                    kind="bearish_ob",
                    low=min(base.open, base.close),
                    high=base.high,
                    created_at=base.timestamp,
                    source_tf="TF",
                    reference_index=i,
                )
            )

    return zones[-limit:]


def fib_levels(
    swing_low: float,
    swing_high: float,
) -> dict[str, float]:
    distance = swing_high - swing_low
    if distance <= 0:
        return {}

    return {
        "0.382": swing_high - distance * 0.382,
        "0.500": swing_high - distance * 0.500,
        "0.618": swing_high - distance * 0.618,
        "0.786": swing_high - distance * 0.786,
    }


def fib_retracement_fraction(
    price: float,
    swing_low: float,
    swing_high: float,
) -> float:
    distance = swing_high - swing_low
    if distance <= 0:
        return math.nan
    return (swing_high - price) / distance


def bearish_fib_retracement_fraction(
    price: float,
    swing_low: float,
    swing_high: float,
) -> float:
    distance = swing_high - swing_low
    if distance <= 0:
        return math.nan
    return (price - swing_low) / distance


def zone_contains_price(zone: Zone, price: float) -> bool:
    epsilon = abs(price) * ZONE_TOUCH_EPSILON
    return zone.low - epsilon <= price <= zone.high + epsilon


def zone_is_discount(
    zone: Zone,
    swing_low: float,
    swing_high: float,
) -> bool:
    retracement = fib_retracement_fraction(zone.midpoint, swing_low, swing_high)
    return math.isfinite(retracement) and retracement >= 0.618


def zone_is_premium(
    zone: Zone,
    swing_low: float,
    swing_high: float,
) -> bool:
    retracement = bearish_fib_retracement_fraction(
        zone.midpoint, swing_low, swing_high
    )
    return math.isfinite(retracement) and retracement >= 0.618


def bullish_liquidity_sweep(
    bars: list[Bar],
    lookback: int = 20,
) -> dict[str, Any] | None:
    if len(bars) < lookback + 5:
        return None

    swing_highs, swing_lows = find_swings(bars)
    if not swing_lows:
        return None

    candidate_idx = swing_lows[-1]
    if candidate_idx >= len(bars) - 2:
        return None

    level = bars[candidate_idx].low
    window = bars[candidate_idx + 1 : -1]
    if not window:
        return None

    latest = bars[-1]
    had_break = any(bar.low < level for bar in window[-lookback:])
    reclaimed = latest.close > level and latest.low < level

    if had_break and reclaimed:
        return {"level": level, "bar": latest, "swing_index": candidate_idx}

    return None


def bearish_liquidity_sweep(
    bars: list[Bar],
    lookback: int = 20,
) -> dict[str, Any] | None:
    if len(bars) < lookback + 5:
        return None

    swing_highs, swing_lows = find_swings(bars)
    if not swing_highs:
        return None

    candidate_idx = swing_highs[-1]
    if candidate_idx >= len(bars) - 2:
        return None

    level = bars[candidate_idx].high
    window = bars[candidate_idx + 1 : -1]
    if not window:
        return None

    latest = bars[-1]
    had_break = any(bar.high > level for bar in window[-lookback:])
    reclaimed = latest.close < level and latest.high > level

    if had_break and reclaimed:
        return {"level": level, "bar": latest, "swing_index": candidate_idx}

    return None


def mss_after_sweep(
    bars: list[Bar],
    direction: str,
) -> bool:
    """
    Confirmation proxy on the 15M stream.

    A true M5/M1 confirmation cannot be reconstructed from 15M candles.
    Therefore this uses the latest confirmed 15M structure shift as the live
    confirmation layer until start.py exposes lower-timeframe data.
    """
    if len(bars) < 10:
        return False

    swing_highs, swing_lows = find_swings(bars)

    if direction == "bullish" and swing_highs:
        level = bars[swing_highs[-1]].high
        return bars[-1].close > level

    if direction == "bearish" and swing_lows:
        level = bars[swing_lows[-1]].low
        return bars[-1].close < level

    return False


def nearest_target(
    bars: list[Bar],
    direction: str,
    entry: float,
    stop: float,
) -> float | None:
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    # Prefer a recent opposing liquidity / swing level if it gives >= 2R.
    swing_highs, swing_lows = find_swings(bars)

    if direction == "bullish":
        candidates = [
            bars[i].high for i in swing_highs[-8:]
            if bars[i].high > entry
        ]
        for target in sorted(candidates):
            if target - entry >= risk * DEFAULT_RR:
                return target
        return entry + risk * DEFAULT_RR

    candidates = [
        bars[i].low for i in swing_lows[-8:]
        if bars[i].low < entry
    ]
    for target in sorted(candidates, reverse=True):
        if entry - target >= risk * DEFAULT_RR:
            return target
    return entry - risk * DEFAULT_RR


def setup_key(
    symbol: str,
    direction: str,
    family: str,
    created_at: int,
    entry_low: float,
    entry_high: float,
) -> str:
    return (
        f"{symbol}:{direction}:{family}:{created_at}:"
        f"{entry_low:.12g}:{entry_high:.12g}"
    )


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    Scans all symbols and maintains simulated setup state.

    The engine intentionally separates:
        detect -> score -> publish -> wait -> trigger -> expire
    """

    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context
        self.stop_event: threading.Event = context["stop_event"]
        self.send_message: Callable[[int, Any], None] = context["send_message"]
        self.chat_id: int | None = context.get("chat_id")

        self.data_engine = context.get("data_engine")
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

        self.setups: dict[str, Setup] = {}
        self._reported_ids: set[str] = set()
        self._last_processed_timestamp: dict[str, int] = {}
        self._last_scan = 0.0
        self._last_summary = 0.0

        self._stats = {
            "scans": 0,
            "symbols_scanned": 0,
            "setups_created": 0,
            "setups_triggered": 0,
            "setups_expired": 0,
        }

    # ----------------------------- IO -----------------------------

    def _notify(self, text: str) -> None:
        if self.chat_id is None:
            return
        try:
            self.send_message(self.chat_id, text)
        except Exception:
            log.exception("[TG] send failed")

    def _get_symbols(self) -> list[str]:
        if self.data_engine is None:
            return []
        try:
            return list(self.data_engine.get_symbols())
        except Exception:
            log.exception("[DATA] get_symbols failed")
            return []

    def _get_candles(self, symbol: str) -> list[dict[str, Any]]:
        if self.data_engine is None:
            return []
        try:
            return list(self.data_engine.get_candles(symbol, 700))
        except Exception:
            log.exception("[DATA] get_candles failed | %s", symbol)
            return []

    def _get_price(self, symbol: str, bars: list[Bar]) -> float | None:
        if self.data_engine is not None:
            try:
                value = self.data_engine.get_price(symbol)
                if value is not None:
                    return float(value)
            except Exception:
                log.exception("[DATA] get_price failed | %s", symbol)

        return bars[-1].close if bars else None

    # ----------------------------- analysis -----------------------------

    def _build_timeframes(
        self,
        base: list[Bar],
    ) -> dict[str, list[Bar]]:
        return {
            "15M": base,
            "1H": aggregate_bars(base, H1_FACTOR),
            "4H": aggregate_bars(base, H4_FACTOR),
            "1D": aggregate_bars(base, D1_FACTOR),
        }

    def _source_context(
        self,
        frames: dict[str, list[Bar]],
    ) -> dict[str, Any]:
        h1 = frames["1H"]
        h4 = frames["4H"]
        d1 = frames["1D"]

        h1_trend = trend_strength(h1)
        h4_trend = trend_strength(h4)
        d1_trend = trend_strength(d1)

        h1_struct = market_structure(h1)
        h4_struct = market_structure(h4)
        d1_struct = market_structure(d1)

        return {
            "h1_trend": h1_trend,
            "h4_trend": h4_trend,
            "d1_trend": d1_trend,
            "h1_struct": h1_struct,
            "h4_struct": h4_struct,
            "d1_struct": d1_struct,
        }

    def _dominant_bias(self, ctx: dict[str, Any]) -> str:
        scores = {"bullish": 0, "bearish": 0}

        for tf in ("d1_trend", "h4_trend", "h1_trend"):
            item = ctx[tf]
            direction = item.get("direction")
            if direction in scores:
                weight = 3 if tf == "d1_trend" else 2 if tf == "h4_trend" else 1
                scores[direction] += weight

        for tf in ("d1_struct", "h4_struct", "h1_struct"):
            direction = ctx[tf].get("trend")
            if direction in scores:
                scores[direction] += 2 if tf != "h1_struct" else 1

        if scores["bullish"] > scores["bearish"]:
            return "bullish"
        if scores["bearish"] > scores["bullish"]:
            return "bearish"
        return "neutral"

    def _find_latest_swing_pair(
        self,
        bars: list[Bar],
        direction: str,
    ) -> tuple[float, float] | None:
        highs, lows = find_swings(bars)
        if not highs or not lows:
            return None

        if direction == "bullish":
            low_idx = lows[-1]
            later_highs = [i for i in highs if i > low_idx]
            if not later_highs:
                return None
            high_idx = later_highs[-1]
            if bars[high_idx].high <= bars[low_idx].low:
                return None
            return bars[low_idx].low, bars[high_idx].high

        high_idx = highs[-1]
        later_lows = [i for i in lows if i > high_idx]
        if not later_lows:
            return None
        low_idx = later_lows[-1]
        if bars[high_idx].high <= bars[low_idx].low:
            return None
        return bars[low_idx].low, bars[high_idx].high

    def _zone_candidates(
        self,
        h1: list[Bar],
        direction: str,
    ) -> list[Zone]:
        fvg = detect_fvgs(h1)
        ob = detect_order_blocks(h1)

        zones = []
        for zone in fvg + ob:
            zone.source_tf = "1H"
            if direction == "bullish" and zone.kind in {"bullish_fvg", "bullish_ob"}:
                zones.append(zone)
            elif direction == "bearish" and zone.kind in {"bearish_fvg", "bearish_ob"}:
                zones.append(zone)

        # Newest first.
        return sorted(zones, key=lambda z: z.created_at, reverse=True)

    def _make_setup(
        self,
        symbol: str,
        bars: list[Bar],
        frames: dict[str, list[Bar]],
        ctx: dict[str, Any],
        bias: str,
    ) -> Setup | None:
        h1 = frames["1H"]
        base = bars

        swing = self._find_latest_swing_pair(h1, bias)
        if swing is None:
            return None

        swing_low, swing_high = swing
        zones = self._zone_candidates(h1, bias)
        if not zones:
            return None

        selected: Zone | None = None
        best_score = -1.0
        best_reasons: list[str] = []
        best_confluences: list[str] = []

        current_price = base[-1].close

        for zone in zones:
            if bias == "bullish":
                fib_ok = zone_is_discount(zone, swing_low, swing_high)
            else:
                fib_ok = zone_is_premium(zone, swing_low, swing_high)

            score = 0.0
            reasons: list[str] = []
            confluences: list[str] = []

            if fib_ok:
                score += 2
                confluences.append("Fibonacci 0.618+")
            else:
                continue

            if (
                (bias == "bullish" and zone.kind == "bullish_ob")
                or (bias == "bearish" and zone.kind == "bearish_ob")
            ):
                score += 2
                confluences.append("Order Block")

            if (
                (bias == "bullish" and zone.kind == "bullish_fvg")
                or (bias == "bearish" and zone.kind == "bearish_fvg")
            ):
                score += 2
                confluences.append("Fair Value Gap")

            h4_dir = ctx["h4_struct"].get("trend")
            d1_dir = ctx["d1_struct"].get("trend")
            if h4_dir == bias:
                score += 1
                confluences.append("H4 structure aligned")
            if d1_dir == bias:
                score += 1
                confluences.append("D1 structure aligned")

            h1_dir = ctx["h1_struct"].get("trend")
            if h1_dir == bias:
                score += 1
                confluences.append("H1 structure aligned")

            # Detect a current/very recent liquidity event.
            sweep = (
                bullish_liquidity_sweep(base)
                if bias == "bullish"
                else bearish_liquidity_sweep(base)
            )
            if sweep:
                score += 2
                confluences.append("Liquidity sweep")
                reasons.append(
                    f"Liquidity taken at {sweep['level']:.8g}"
                )

            if zone_contains_price(zone, current_price):
                score += 1
                reasons.append("Price is inside POI")
            else:
                reasons.append("Waiting for POI retest")

            if score > best_score:
                selected = zone
                best_score = score
                best_reasons = reasons
                best_confluences = confluences

        if selected is None or best_score < MIN_SCORE:
            return None

        # Entry zone.
        entry_low = selected.low
        entry_high = selected.high

        # Prefer limit while waiting for the POI. If price is already in the
        # zone and 15M structure has confirmed the direction, use simulated
        # market execution.
        confirmed = mss_after_sweep(base, bias)
        touched = zone_contains_price(selected, current_price)

        if touched and confirmed:
            entry_type = "MARKET"
            entry_price = current_price
            status = "READY"
        else:
            entry_type = "LIMIT"
            entry_price = selected.midpoint
            status = "WATCHING"

        risk_buffer = max(
            selected.high - selected.low,
            (atr(h1, 14)[-1] or 0.0) * 0.15,
        )

        if bias == "bullish":
            stop = min(selected.low, swing_low) - risk_buffer * 0.10
        else:
            stop = max(selected.high, swing_high) + risk_buffer * 0.10

        target = nearest_target(h1, bias, entry_price, stop)
        if target is None:
            return None

        risk = abs(entry_price - stop)
        reward = abs(target - entry_price)
        rr = reward / risk if risk > 0 else 0.0

        if rr < MIN_RR:
            # Recalculate with strict 2R/3R style target rather than force a
            # structurally too-close level.
            target = (
                entry_price + risk * DEFAULT_RR
                if bias == "bullish"
                else entry_price - risk * DEFAULT_RR
            )
            reward = abs(target - entry_price)
            rr = reward / risk if risk > 0 else 0.0

        now_bar = base[-1]
        setup_id = setup_key(
            symbol,
            bias,
            "H1_POI_CONFLUENCE",
            selected.created_at,
            entry_low,
            entry_high,
        )

        return Setup(
            setup_id=setup_id,
            symbol=symbol,
            direction="BUY" if bias == "bullish" else "SELL",
            family="H1_POI_CONFLUENCE",
            status=status,
            score=best_score,
            created_at=now_bar.timestamp,
            expiry_at=now_bar.timestamp + SETUP_EXPIRY_CANDLES * BASE_TF_MINUTES * 60_000,
            source_tf="1H",
            confirmation_tf="15M",
            entry_type=entry_type,
            entry_low=entry_low,
            entry_high=entry_high,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit=target,
            risk_distance=risk,
            rr=rr,
            reasons=best_reasons + [
                "15M confirmation layer; M5/M1 unavailable from 15M source"
            ],
            confluences=best_confluences,
            origin_bar=now_bar.timestamp,
            last_seen_bar=now_bar.timestamp,
            trigger_bar=None,
        )

    def _make_sweep_setup(
        self,
        symbol: str,
        bars: list[Bar],
        frames: dict[str, list[Bar]],
        ctx: dict[str, Any],
    ) -> Setup | None:
        """
        Reversal candidate:
            liquidity sweep -> 15M MSS -> FVG/OB retracement.
        """
        bias: str
        sweep: dict[str, Any] | None

        bull_sweep = bullish_liquidity_sweep(bars)
        bear_sweep = bearish_liquidity_sweep(bars)

        if bull_sweep and not bear_sweep:
            bias = "bullish"
            sweep = bull_sweep
        elif bear_sweep and not bull_sweep:
            bias = "bearish"
            sweep = bear_sweep
        else:
            return None

        if not mss_after_sweep(bars, bias):
            return None

        h1 = frames["1H"]
        zones = self._zone_candidates(h1, bias)
        if not zones:
            return None

        swing = self._find_latest_swing_pair(h1, bias)
        if swing is None:
            return None

        swing_low, swing_high = swing
        eligible: list[Zone] = []
        for z in zones:
            good = (
                zone_is_discount(z, swing_low, swing_high)
                if bias == "bullish"
                else zone_is_premium(z, swing_low, swing_high)
            )
            if good:
                eligible.append(z)

        if not eligible:
            return None

        zone = eligible[0]
        price = bars[-1].close

        if bias == "bullish":
            stop = min(zone.low, sweep["level"]) * (1.0 - 0.00015)
        else:
            stop = max(zone.high, sweep["level"]) * (1.0 + 0.00015)

        entry = zone.midpoint
        target = nearest_target(h1, bias, entry, stop)
        if target is None:
            return None

        risk = abs(entry - stop)
        rr = abs(target - entry) / risk if risk > 0 else 0.0
        if rr < MIN_RR:
            target = entry + risk * DEFAULT_RR if bias == "bullish" else entry - risk * DEFAULT_RR
            rr = DEFAULT_RR

        touched = zone_contains_price(zone, price)

        return Setup(
            setup_id=setup_key(
                symbol,
                bias,
                "LIQUIDITY_SWEEP_MSS",
                int(sweep["bar"].timestamp),
                zone.low,
                zone.high,
            ),
            symbol=symbol,
            direction="BUY" if bias == "bullish" else "SELL",
            family="LIQUIDITY_SWEEP_MSS",
            status="READY" if touched else "WATCHING",
            score=7.0,
            created_at=int(sweep["bar"].timestamp),
            expiry_at=int(sweep["bar"].timestamp)
            + SETUP_EXPIRY_CANDLES * BASE_TF_MINUTES * 60_000,
            source_tf="1H/15M",
            confirmation_tf="15M",
            entry_type="MARKET" if touched else "LIMIT",
            entry_low=zone.low,
            entry_high=zone.high,
            entry_price=price if touched else entry,
            stop_loss=stop,
            take_profit=target,
            risk_distance=risk,
            rr=rr,
            reasons=[
                f"Liquidity sweep at {sweep['level']:.8g}",
                "15M market structure shift confirmed",
                "H1 POI retest model",
                "M5/M1 not available from 15M source",
            ],
            confluences=[
                "Liquidity sweep",
                "Market structure shift",
                zone.kind.replace("_", " ").upper(),
                "Fibonacci 0.618+",
            ],
            origin_bar=int(sweep["bar"].timestamp),
            last_seen_bar=bars[-1].timestamp,
        )

    # ----------------------------- state -----------------------------

    def _existing_for_symbol(self, symbol: str) -> list[Setup]:
        with self._lock:
            return [
                s for s in self.setups.values()
                if s.symbol == symbol and s.status in {"WATCHING", "READY", "PENDING"}
            ]

    def _accept_setup(self, setup: Setup) -> bool:
        if setup.setup_id in self._reported_ids:
            return False

        existing = self._existing_for_symbol(setup.symbol)
        if len(existing) >= MAX_ACTIVE_SETUP_PER_SYMBOL:
            return False

        with self._lock:
            active = [
                s for s in self.setups.values()
                if s.status in {"WATCHING", "READY", "PENDING"}
            ]
            if len(active) >= MAX_TOTAL_ACTIVE_SETUPS:
                return False

            self.setups[setup.setup_id] = setup
            self._reported_ids.add(setup.setup_id)
            self._stats["setups_created"] += 1

        return True

    def _update_setup_state(
        self,
        setup: Setup,
        bars: list[Bar],
        current_price: float,
    ) -> str | None:
        latest_ts = bars[-1].timestamp

        if setup.expiry_at is not None and latest_ts >= setup.expiry_at:
            setup.status = "EXPIRED"
            self._stats["setups_expired"] += 1
            return "EXPIRED"

        # Invalidate a setup if the market closes clearly beyond the protective
        # zone before entry.
        if setup.direction == "BUY":
            if bars[-1].close < setup.stop_loss:
                setup.status = "INVALIDATED"
                return "INVALIDATED"
            touched = (
                setup.entry_low <= current_price <= setup.entry_high
                or bars[-1].low <= setup.entry_high
            )
        else:
            if bars[-1].close > setup.stop_loss:
                setup.status = "INVALIDATED"
                return "INVALIDATED"
            touched = (
                setup.entry_low <= current_price <= setup.entry_high
                or bars[-1].high >= setup.entry_low
            )

        if setup.status == "WATCHING" and touched:
            # Wait for 15M directional confirmation before simulated execution.
            confirmed = mss_after_sweep(bars, "bullish" if setup.direction == "BUY" else "bearish")
            if confirmed:
                setup.status = "TRIGGERED"
                setup.entry_type = "MARKET"
                setup.entry_price = current_price
                setup.trigger_bar = latest_ts
                self._stats["setups_triggered"] += 1
                return "TRIGGERED"
            setup.status = "PENDING"
            return "PENDING"

        if setup.status == "PENDING":
            confirmed = mss_after_sweep(bars, "bullish" if setup.direction == "BUY" else "bearish")
            if confirmed and touched:
                setup.status = "TRIGGERED"
                setup.entry_type = "MARKET"
                setup.entry_price = current_price
                setup.trigger_bar = latest_ts
                self._stats["setups_triggered"] += 1
                return "TRIGGERED"

        setup.last_seen_bar = latest_ts
        return None

    def _format_setup(self, setup: Setup, event: str = "NEW") -> str:
        emoji = "🟢" if setup.direction == "BUY" else "🔴"
        return (
            f"{emoji} <b>{event} SETUP</b>\n"
            f"Pair: <code>{setup.symbol}</code>\n"
            f"Direction: <b>{setup.direction}</b>\n"
            f"Family: <code>{setup.family}</code>\n"
            f"Status: <b>{setup.status}</b>\n"
            f"Entry: {setup.entry_type} @ {setup.entry_price:.8g}\n"
            f"Zone: {setup.entry_low:.8g} → {setup.entry_high:.8g}\n"
            f"SL: {setup.stop_loss:.8g}\n"
            f"TP: {setup.take_profit:.8g}\n"
            f"RR: 1:{setup.rr:.2f}\n"
            f"Score: {setup.score:.1f}\n"
            f"Confluence: {', '.join(setup.confluences)}\n"
            f"Reason: {', '.join(setup.reasons[:4])}"
        )

    # ----------------------------- scanning -----------------------------

    def scan_symbol(self, symbol: str) -> list[Setup]:
        raw = self._get_candles(symbol)
        bars = normalize_bars(raw)

        if len(bars) < MIN_BASE_CANDLES:
            log.warning(
                "[SCAN] %s | insufficient data %d/%d",
                symbol,
                len(bars),
                MIN_BASE_CANDLES,
            )
            return []

        frames = self._build_timeframes(bars)
        if len(frames["1H"]) < 25:
            return []

        ctx = self._source_context(frames)
        bias = self._dominant_bias(ctx)

        if bias == "neutral":
            return []

        current_price = self._get_price(symbol, bars)
        if current_price is None:
            return []

        newest = bars[-1].timestamp
        last_seen = self._last_processed_timestamp.get(symbol)
        if last_seen == newest:
            # Still update state because the live candle price can change.
            pass
        else:
            self._last_processed_timestamp[symbol] = newest

        candidates: list[Setup] = []

        regular = self._make_setup(symbol, bars, frames, ctx, bias)
        if regular is not None:
            candidates.append(regular)

        reversal = self._make_sweep_setup(symbol, bars, frames, ctx)
        if reversal is not None:
            candidates.append(reversal)

        # Prefer stronger setup families first.
        candidates.sort(
            key=lambda s: (
                s.score,
                1 if s.family == "LIQUIDITY_SWEEP_MSS" else 0,
            ),
            reverse=True,
        )

        return candidates[:MAX_ACTIVE_SETUP_PER_SYMBOL]

    def scan_all(self) -> tuple[list[Setup], int]:
        symbols = self._get_symbols()

        created: list[Setup] = []
        scanned = 0

        for symbol in symbols:
            if self.stop_event.is_set() or not self._running:
                break

            try:
                scanned += 1
                candidates = self.scan_symbol(symbol)
                for candidate in candidates:
                    if self._accept_setup(candidate):
                        created.append(candidate)
                        log.info(
                            "[SETUP] NEW | %s | %s | family=%s score=%.1f entry=%s rr=1:%.2f",
                            candidate.symbol,
                            candidate.direction,
                            candidate.family,
                            candidate.score,
                            candidate.entry_type,
                            candidate.rr,
                        )
            except Exception:
                log.exception("[SCAN] %s failed", symbol)

        self._stats["scans"] += 1
        self._stats["symbols_scanned"] += scanned
        return created, scanned

    def update_live_states(self) -> list[tuple[Setup, str]]:
        events: list[tuple[Setup, str]] = []

        with self._lock:
            active = [
                s for s in self.setups.values()
                if s.status in {"WATCHING", "PENDING", "READY"}
            ]

        for setup in active:
            raw = self._get_candles(setup.symbol)
            bars = normalize_bars(raw)
            if not bars:
                continue

            price = self._get_price(setup.symbol, bars)
            if price is None:
                continue

            old = setup.status
            result = self._update_setup_state(setup, bars, price)

            if result and result != old:
                events.append((setup, result))
                log.info(
                    "[SETUP] STATE | %s | %s -> %s | entry=%s",
                    setup.symbol,
                    old,
                    result,
                    setup.entry_price,
                )

        return events

    def _summary(self) -> None:
        now = time.monotonic()
        if now - self._last_summary < LOG_SUMMARY_SECONDS:
            return
        self._last_summary = now

        with self._lock:
            active = [
                s for s in self.setups.values()
                if s.status in {"WATCHING", "PENDING", "READY"}
            ]

        log.info(
            "[SUMMARY] scans=%d symbols=%d new_setups=%d triggered=%d expired=%d active=%d",
            self._stats["scans"],
            self._stats["symbols_scanned"],
            self._stats["setups_created"],
            self._stats["setups_triggered"],
            self._stats["setups_expired"],
            len(active),
        )

    # ----------------------------- lifecycle -----------------------------

    def run(self) -> None:
        log.info("[STRATEGY] worker started")
        boot_notified = False

        while self._running and not self.stop_event.is_set():
            try:
                if self.data_engine is None:
                    log.warning("[STRATEGY] data_engine unavailable; waiting")
                    self.stop_event.wait(5)
                    continue

                status = {}
                try:
                    status = self.data_engine.get_status()
                except Exception:
                    pass

                if not status.get("bootstrap_done", False):
                    if not boot_notified:
                        log.info("[STRATEGY] waiting for historical bootstrap")
                        boot_notified = True
                    self.stop_event.wait(SCAN_INTERVAL_SECONDS)
                    continue

                if not self._last_scan or (
                    time.monotonic() - self._last_scan >= SCAN_INTERVAL_SECONDS
                ):
                    self._last_scan = time.monotonic()

                    created, scanned = self.scan_all()

                    for setup in created:
                        # Telegram only gets meaningful setup messages.
                        self._notify(self._format_setup(setup, "NEW"))

                    state_events = self.update_live_states()
                    for setup, event in state_events:
                        if event == "TRIGGERED":
                            self._notify(self._format_setup(setup, "TRIGGERED"))
                        elif event in {"EXPIRED", "INVALIDATED"}:
                            self._notify(
                                f"ℹ️ <b>SETUP {event}</b>\n"
                                f"{setup.symbol} {setup.direction}\n"
                                f"{setup.family}\n"
                                f"ID: <code>{setup.setup_id}</code>"
                            )

                    self._summary()

                self.stop_event.wait(SCAN_INTERVAL_SECONDS)

            except Exception:
                log.exception("[STRATEGY] worker loop failed")
                self.stop_event.wait(3)

        log.info("[STRATEGY] worker stopped")

    def start(self) -> str:
        with self._lock:
            if self._running:
                return "ℹ️ Strategy engine sudah aktif."

            if self.data_engine is None:
                return (
                    "❌ Strategy engine belum mendapat data_engine dari start.py."
                )

            self._running = True

        self._thread = threading.Thread(
            target=self.run,
            name="strategy-engine",
            daemon=True,
        )
        self._thread.start()

        self._notify(
            "🧠 <b>STRATEGY ENGINE ACTIVE</b>\n"
            "Mode: SIMULATION ONLY\n"
            "Base data: 15M\n"
            "Derived TF: 1H / 4H / 1D\n"
            "Scanning: ALL available pairs\n"
            "Execution: virtual only"
        )
        return "🟢 Strategy engine aktif."

    def stop(self) -> None:
        with self._lock:
            self._running = False
        log.info("[STRATEGY] stop requested")

    # ----------------------------- commands / API -----------------------------

    def get_setups(self, active_only: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            values = list(self.setups.values())

        if active_only:
            values = [
                x for x in values
                if x.status in {"WATCHING", "PENDING", "READY", "TRIGGERED"}
            ]

        values.sort(key=lambda x: x.created_at, reverse=True)
        return [x.as_dict() for x in values]

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            active = sum(
                1 for x in self.setups.values()
                if x.status in {"WATCHING", "PENDING", "READY"}
            )
            triggered = sum(1 for x in self.setups.values() if x.status == "TRIGGERED")

        return {
            "running": self._running,
            "active_setups": active,
            "triggered_setups": triggered,
            **self._stats,
        }


# ---------------------------------------------------------------------------
# Module-level integration
# ---------------------------------------------------------------------------

_ENGINE_LOCK = threading.RLock()
_STRATEGY: StrategyEngine | None = None


def on_start(context: dict[str, Any]) -> bool:
    """
    Called by an integrated start.py.

    Important:
    Current start.py must pass context["data_engine"] = its DataEngine instance
    for this strategy module to access the market data.
    """
    global _STRATEGY

    data_engine = context.get("data_engine")
    if data_engine is None:
        log.warning(
            "[STRATEGY] context[data_engine] missing. "
            "Strategy file loaded but cannot scan yet."
        )

    with _ENGINE_LOCK:
        _STRATEGY = StrategyEngine(context)

    # Strategy can start its worker immediately; it will wait until bootstrap_done.
    return _STRATEGY.start().startswith("🟢") if _STRATEGY else False


def on_stop(context: dict[str, Any]) -> bool:
    global _STRATEGY

    with _ENGINE_LOCK:
        engine = _STRATEGY
        _STRATEGY = None

    if engine is not None:
        engine.stop()

    return True


def _strategy_or_raise() -> StrategyEngine:
    with _ENGINE_LOCK:
        engine = _STRATEGY
    if engine is None:
        raise RuntimeError("Strategy engine belum diinisialisasi.")
    return engine


def handle_update(update: dict[str, Any], context: dict[str, Any]) -> str | None:
    """
    Telegram command namespace for strategy.py.

    The module is intentionally simulation-only.
    """
    message = update.get("message") or {}
    text = str(message.get("text") or message.get("caption") or "").strip()
    if not text:
        return None

    command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()

    if command == "/strategystatus":
        engine = _strategy_or_raise()
        s = engine.get_status()
        return (
            "🧠 <b>STRATEGY STATUS</b>\n"
            f"Running: {'ON' if s['running'] else 'OFF'}\n"
            f"Scans: {s['scans']}\n"
            f"Symbols scanned: {s['symbols_scanned']}\n"
            f"Setups created: {s['setups_created']}\n"
            f"Triggered: {s['setups_triggered']}\n"
            f"Expired: {s['setups_expired']}\n"
            f"Active: {s['active_setups']}"
        )

    if command == "/setups":
        parts = text.split()
        limit = 10
        if len(parts) > 1:
            try:
                limit = max(1, min(25, int(parts[1])))
            except ValueError:
                return "❌ Limit harus angka."

        setups = _strategy_or_raise().get_setups(active_only=True)[:limit]
        if not setups:
            return "ℹ️ Belum ada setup aktif."

        lines = ["📋 <b>ACTIVE SETUPS</b>"]
        for s in setups:
            lines.append(
                f"{'🟢' if s['direction'] == 'BUY' else '🔴'} "
                f"{s['symbol']} | {s['direction']} | {s['family']} | "
                f"{s['status']} | RR 1:{s['rr']}"
            )
        return "\n".join(lines)

    if command == "/setup":
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /setup SETUP_ID"

        setup_id = parts[1].strip()
        setups = _strategy_or_raise().get_setups(active_only=False)
        for s in setups:
            if s["setup_id"] == setup_id:
                return (
                    f"🧾 <b>SETUP DETAIL</b>\n"
                    f"Pair: {s['symbol']}\n"
                    f"Direction: {s['direction']}\n"
                    f"Family: {s['family']}\n"
                    f"Status: {s['status']}\n"
                    f"Entry: {s['entry_type']} @ {s['entry_price']}\n"
                    f"Zone: {s['entry_low']} → {s['entry_high']}\n"
                    f"SL: {s['stop_loss']}\n"
                    f"TP: {s['take_profit']}\n"
                    f"RR: 1:{s['rr']}\n"
                    f"Confluence: {', '.join(s['confluences'])}\n"
                    f"Reason: {', '.join(s['reasons'])}"
                )
        return "❌ Setup ID tidak ditemukan."

    if command == "/strategyhelp":
        return (
            "🧠 <b>STRATEGY COMMANDS</b>\n\n"
            "/strategystatus — status engine\n"
            "/setups [n] — daftar setup aktif\n"
            "/setup SETUP_ID — detail setup\n"
            "/strategyhelp — bantuan"
        )

    return None


if __name__ == "__main__":
    print(
        "strategy.py is a simulation strategy module. "
        "Load it from start.py with context['data_engine']."
    )
