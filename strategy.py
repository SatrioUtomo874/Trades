from __future__ import annotations

"""
STRATEGY.PY
===========

Strategy intelligence untuk main.py.

Kontrak utama:
    async def generate_setup(pair, context) -> dict

Prinsip desain v0.2.0
---------------------
1. SMC adalah kerangka utama; RSI 14 M15 dan VLT/OHLCV adalah konfirmasi.
2. Semua keputusan struktur memakai CLOSED candles saja.
3. Target pair dianalisis multi-timeframe:
      H4 -> bias struktur pasangan
      H1 -> lokasi / dealing range / POI
      M15 -> timing / trigger / RSI / VLT
4. Untuk altcoin, BTCUSDT H4 adalah directional constraint utama:
      BULLISH -> hanya cari BUY
      BEARISH -> hanya cari SELL
      RANGE   -> BUY dan SELL boleh dicari.
5. Multi-timeframe target pair tetap menentukan kualitas, lokasi, trigger,
   dan timing; ia tidak boleh membalikkan arah yang ditentukan BTC H4.
6. Data M15 target = 672 CLOSED candles (7 hari).
7. Data provider: Bybit public REST -> Binance public REST fallback.
8. Strategy selalu memilih kandidat terbaik bila market data cukup.
   Tidak ada hard gate "NO VALID SETUP" hanya karena confidence rendah.
9. Confidence = quality score setup 0-100, bukan probabilitas profit.
10. Price Exp adalah batas relevansi setup: bila terlewati sebelum entry,
   thesis lama dianggap tidak relevan dan pola baru harus dicari.
11. strategy.py tidak menyentuh state trade, Telegram, WebSocket, GitHub,
    atau order execution. Ia hanya menganalisis dan mengembalikan dict.

SMC yang dibuat objektif dalam kode:
    - Swing High / Swing Low
    - HH / HL / LH / LL
    - BOS
    - CHOCH
    - MSS (CHOCH + displacement)
    - Equal High / Equal Low
    - Swing liquidity
    - Liquidity sweep
    - Fair Value Gap (FVG)
    - Order Block (OB)
    - Breaker Block (basic, dari OB failure)
    - Premium / Discount / Equilibrium
    - Displacement
    - ATR / RSI(14) / relative-volume / trend slope (VLT-style)

Catatan penting:
- SMC bukan standar teknikal tunggal dengan definisi universal. Karena itu
  istilah di atas diberi definisi operasional yang deterministik agar dapat
  dites dari trade history.
- "VLT" di sini adalah modul custom berbasis OHLCV. Ia bukan klaim memiliki
  true order-flow / footprint / delta data.
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable

import requests


# ============================================================================
# CONSTANTS / CONFIG
# ============================================================================

STRATEGY_NAME = "SMC_VLT_RSI"
STRATEGY_VERSION = "0.3.0"

BYBIT_BASE_URL = "https://api.bybit.com"
BINANCE_BASE_URL = "https://fapi.binance.com"

M15 = "15"
H1 = "60"
H4 = "240"

M15_MS = 15 * 60 * 1000
H1_MS = 60 * 60 * 1000
H4_MS = 4 * 60 * 60 * 1000

M15_CANDLES_REQUIRED = 672
PAIR_H4_CANDLES_REQUIRED = 120
BTC_H4_CANDLES_REQUIRED = 240

HTTP_TIMEOUT_SECONDS = 20

SWING_SPAN_M15 = 3
SWING_SPAN_H1 = 2
SWING_SPAN_H4 = 2

FVG_MAX_AGE_M15 = 160
POI_MAX_DISTANCE_ATR = 3.5
H4_POI_MAX_DISTANCE_ATR = 8.0
H1_POI_MAX_DISTANCE_ATR = 5.0
H1_REFINEMENT_OVERLAP_ATR = 2.0
M15_EXECUTION_OVERLAP_ATR = 1.5
FIB_DEEP_RATIO = 0.618
FIB_DEEPEST_RATIO = 0.786
FIB_SHALLOW_RATIO = 0.382
FIB_MID_RATIO = 0.500
EXP_ATR_MULTIPLIER = 1.25
SL_BUFFER_ATR = 0.15

WEIGHTS = {
    "btc_h4_bias": 15.0,
    "pair_h4_structure": 10.0,
    "h4_poi": 14.0,
    "h4_fibonacci": 11.0,
    "h1_refinement": 10.0,
    "liquidity": 12.0,
    "smc_trigger": 14.0,
    "displacement": 5.0,
    "rsi_m15": 3.0,
    "vlt": 3.0,
    "planned_rr": 3.0,
}
DIRECTION_BUY = "BUY"
DIRECTION_SELL = "SELL"
DIRECTIONS_BOTH = (DIRECTION_BUY, DIRECTION_SELL)

EPS = 1e-12


# ============================================================================
# BASIC NUMERIC HELPERS
# ============================================================================


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return default


def round_price(value: float) -> float:
    """Reasonable display precision; final tick validation remains in main.py."""
    if not math.isfinite(value):
        return value
    if abs(value) >= 1000:
        return round(value, 2)
    if abs(value) >= 100:
        return round(value, 3)
    if abs(value) >= 1:
        return round(value, 5)
    if abs(value) >= 0.01:
        return round(value, 7)
    return round(value, 10)


def pct_change(a: float, b: float) -> float:
    if abs(b) <= EPS:
        return 0.0
    return (a - b) / b * 100.0


def distance_pct(price: float, level: float) -> float:
    if abs(price) <= EPS:
        return 0.0
    return abs(level - price) / price * 100.0


def normalize_pair(value: str) -> str:
    pair = str(value or "").upper().strip()
    pair = pair.replace("/", "").replace("-", "").replace(" ", "")
    if not pair.endswith("USDT"):
        pair += "USDT"
    return pair


def is_altcoin(pair: str) -> bool:
    return normalize_pair(pair) != "BTCUSDT"


def median_or(values: Iterable[float], fallback: float) -> float:
    data = [x for x in values if math.isfinite(x)]
    return float(median(data)) if data else fallback


def linear_slope(values: list[float], lookback: int = 20) -> float:
    data = values[-lookback:]
    if len(data) < 3:
        return 0.0
    n = len(data)
    mean_x = (n - 1) / 2.0
    mean_y = sum(data) / n
    denom = sum((i - mean_x) ** 2 for i in range(n))
    if denom <= EPS:
        return 0.0
    return sum((i - mean_x) * (y - mean_y) for i, y in enumerate(data)) / denom


# ============================================================================
# CANDLE DATA MODEL
# ============================================================================

@dataclass(slots=True)
class Candle:
    time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(self.high - self.low, EPS)

    @property
    def body_ratio(self) -> float:
        return self.body / self.range

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open


@dataclass(slots=True)
class Pivot:
    index: int
    price: float
    kind: str  # HIGH / LOW
    confirmed_at_index: int


@dataclass(slots=True)
class StructureSnapshot:
    trend: str
    swing_highs: list[Pivot]
    swing_lows: list[Pivot]
    labels: list[dict[str, Any]]
    events: list[dict[str, Any]]
    last_bos: dict[str, Any] | None
    last_mss: dict[str, Any] | None
    protected_high: float | None
    protected_low: float | None


@dataclass(slots=True)
class Zone:
    kind: str
    low: float
    high: float
    index: int
    source: str
    status: str = "ACTIVE"
    strength: float = 50.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(slots=True)
class LiquidityPool:
    kind: str
    level: float
    index: int
    strength: float
    source: str
    swept: bool = False
    sweep_index: int | None = None
    distance_pct: float = 0.0
    distance_atr: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FibonacciContext:
    timeframe: str
    direction: str
    swing_low: float
    swing_high: float
    low_index: int
    high_index: int
    anchor_reason: str
    trend_strength: dict[str, Any]
    levels: dict[str, float]

    @property
    def range(self) -> float:
        return max(self.swing_high - self.swing_low, EPS)

    def price_at(self, ratio: float) -> float:
        ratio = max(0.0, min(1.0, float(ratio)))
        if self.direction == "BUY":
            return self.swing_high - ratio * self.range
        return self.swing_low + ratio * self.range

    def retracement_ratio(self, price: float) -> float:
        if self.direction == "BUY":
            return (self.swing_high - price) / self.range
        return (price - self.swing_low) / self.range


@dataclass(slots=True)
class Candidate:
    direction: str
    model: str
    entry: float
    sl: float
    tp: float
    price_exp: float
    entry_reason: str
    sl_reason: str
    tp_reason: str
    price_exp_reason: str
    evidence: dict[str, Any]
    scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


# ============================================================================
# HTTP PROVIDERS
# ============================================================================

class HTTPProvider:
    def __init__(self, base_url: str, name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name

    async def get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"

        def request() -> Any:
            response = requests.get(
                url,
                params=params,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()

        return await asyncio.to_thread(request)


class BybitProvider(HTTPProvider):
    def __init__(self) -> None:
        super().__init__(BYBIT_BASE_URL, "BYBIT")

    async def klines(
        self,
        symbol: str,
        interval: str,
        count: int,
        interval_ms: int,
    ) -> list[Candle]:
        collected: dict[int, Candle] = {}
        end_ms = int(time.time() * 1000)
        attempts = 0

        while len(collected) < count and attempts < 5:
            attempts += 1
            limit = min(1000, max(200, count - len(collected) + 5))
            payload = await self.get(
                "/v5/market/kline",
                {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "end": end_ms,
                    "limit": limit,
                },
            )

            if int(payload.get("retCode", -1)) != 0:
                raise RuntimeError(
                    f"Bybit retCode={payload.get('retCode')}: "
                    f"{payload.get('retMsg', 'unknown error')}"
                )

            rows = ((payload.get("result") or {}).get("list") or [])
            if not rows:
                break

            oldest: int | None = None
            now_ms = int(time.time() * 1000)

            for row in rows:
                if len(row) < 7:
                    continue
                start = int(row[0])
                oldest = start if oldest is None else min(oldest, start)

                # Hanya candle yang sudah selesai sepenuhnya.
                if start + interval_ms > now_ms:
                    continue

                collected[start] = Candle(
                    time_ms=start,
                    open=safe_float(row[1]),
                    high=safe_float(row[2]),
                    low=safe_float(row[3]),
                    close=safe_float(row[4]),
                    volume=safe_float(row[5]),
                    turnover=safe_float(row[6]),
                )

            if oldest is None:
                break

            end_ms = oldest - 1

            # Prevent an accidental loop on a provider response that did not move.
            if len(rows) == 1:
                break

        candles = [collected[k] for k in sorted(collected)]
        if len(candles) < count:
            raise RuntimeError(
                f"Bybit hanya mengembalikan {len(candles)} candle closed "
                f"untuk {symbol} {interval}; diperlukan {count}."
            )
        return candles[-count:]

    async def last_price(self, symbol: str) -> float:
        payload = await self.get(
            "/v5/market/tickers",
            {
                "category": "linear",
                "symbol": symbol,
            },
        )
        if int(payload.get("retCode", -1)) != 0:
            raise RuntimeError(
                f"Bybit ticker retCode={payload.get('retCode')}: "
                f"{payload.get('retMsg', 'unknown error')}"
            )
        rows = ((payload.get("result") or {}).get("list") or [])
        if not rows:
            raise RuntimeError(f"Bybit ticker tidak memiliki {symbol}.")
        return safe_float(rows[0].get("lastPrice"), 0.0)


class BinanceProvider(HTTPProvider):
    def __init__(self) -> None:
        super().__init__(BINANCE_BASE_URL, "BINANCE_FALLBACK")

    async def klines(
        self,
        symbol: str,
        interval: str,
        count: int,
        interval_ms: int,
    ) -> list[Candle]:
        collected: dict[int, Candle] = {}
        end_ms = int(time.time() * 1000)
        attempts = 0

        while len(collected) < count and attempts < 5:
            attempts += 1
            limit = min(1000, max(200, count - len(collected) + 5))
            payload = await self.get(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": _binance_interval(interval),
                    "endTime": end_ms,
                    "limit": limit,
                },
            )

            if not isinstance(payload, list):
                raise RuntimeError("Binance klines response bukan list.")

            if not payload:
                break

            oldest: int | None = None
            now_ms = int(time.time() * 1000)

            for row in payload:
                if len(row) < 7:
                    continue
                start = int(row[0])
                oldest = start if oldest is None else min(oldest, start)
                if start + interval_ms > now_ms:
                    continue

                collected[start] = Candle(
                    time_ms=start,
                    open=safe_float(row[1]),
                    high=safe_float(row[2]),
                    low=safe_float(row[3]),
                    close=safe_float(row[4]),
                    volume=safe_float(row[5]),
                    turnover=safe_float(row[7]) if len(row) > 7 else 0.0,
                )

            if oldest is None:
                break
            end_ms = oldest - 1
            if len(payload) == 1:
                break

        candles = [collected[k] for k in sorted(collected)]
        if len(candles) < count:
            raise RuntimeError(
                f"Binance hanya mengembalikan {len(candles)} candle closed "
                f"untuk {symbol} {_binance_interval(interval)}; diperlukan {count}."
            )
        return candles[-count:]

    async def last_price(self, symbol: str) -> float:
        payload = await self.get(
            "/fapi/v1/ticker/price",
            {"symbol": symbol},
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Binance ticker response tidak valid.")
        price = safe_float(payload.get("price"), 0.0)
        if price <= 0:
            raise RuntimeError(f"Binance ticker tidak memiliki harga {symbol}.")
        return price

    async def tick_size(self, symbol: str) -> float:
        payload = await self.get(
            "/fapi/v1/exchangeInfo",
            {},
        )
        for raw in payload.get("symbols", []):
            if str(raw.get("symbol") or "").upper() != symbol:
                continue
            for filt in raw.get("filters", []):
                if filt.get("filterType") == "PRICE_FILTER":
                    tick = safe_float(filt.get("tickSize"), 0.0)
                    if tick > 0:
                        return tick
        return 0.0


def _binance_interval(interval: str) -> str:
    mapping = {"15": "15m", "60": "1h", "240": "4h"}
    return mapping.get(str(interval), str(interval))


# ============================================================================
# FETCHING / NORMALIZATION
# ============================================================================

async def fetch_series(
    pair: str,
    interval: str,
    count: int,
    interval_ms: int,
) -> tuple[list[Candle], str]:
    bybit = BybitProvider()
    try:
        candles = await bybit.klines(pair, interval, count, interval_ms)
        return candles, "BYBIT"
    except Exception as bybit_exc:
        binance = BinanceProvider()
        try:
            candles = await binance.klines(pair, interval, count, interval_ms)
            return candles, "BINANCE_FALLBACK"
        except Exception as binance_exc:
            raise RuntimeError(
                f"Gagal mengambil {pair} {interval}: "
                f"Bybit={bybit_exc}; Binance={binance_exc}"
            ) from binance_exc


async def fetch_price(pair: str, preferred_source: str | None = None) -> tuple[float, str]:
    errors: list[str] = []
    providers = []

    if preferred_source == "BINANCE_FALLBACK":
        providers = [BinanceProvider(), BybitProvider()]
    else:
        providers = [BybitProvider(), BinanceProvider()]

    for provider in providers:
        try:
            return await provider.last_price(pair), provider.name
        except Exception as exc:
            errors.append(f"{provider.name}: {exc}")

    raise RuntimeError("Gagal mengambil current price. " + " | ".join(errors))


async def get_tick_size(pair: str) -> float:
    try:
        return await BinanceProvider().tick_size(pair)
    except Exception:
        return 0.0


# ============================================================================
# TIMEFRAME RESAMPLING
# ============================================================================


def resample_candles(candles: list[Candle], bucket_ms: int) -> list[Candle]:
    """Aggregate lower timeframe candles into closed higher timeframe buckets."""
    buckets: dict[int, list[Candle]] = {}

    for candle in candles:
        bucket = (candle.time_ms // bucket_ms) * bucket_ms
        buckets.setdefault(bucket, []).append(candle)

    result: list[Candle] = []
    for start in sorted(buckets):
        rows = sorted(buckets[start], key=lambda x: x.time_ms)
        if not rows:
            continue
        # A bucket is only valid if it has all expected component candles.
        expected = bucket_ms // M15_MS
        if bucket_ms % M15_MS == 0 and len(rows) < expected:
            continue
        result.append(
            Candle(
                time_ms=start,
                open=rows[0].open,
                high=max(r.high for r in rows),
                low=min(r.low for r in rows),
                close=rows[-1].close,
                volume=sum(r.volume for r in rows),
                turnover=sum(r.turnover for r in rows),
            )
        )
    return result


# ============================================================================
# INDICATORS
# ============================================================================


def true_ranges(candles: list[Candle]) -> list[float]:
    result: list[float] = []
    prev_close: float | None = None
    for c in candles:
        if prev_close is None:
            tr = c.high - c.low
        else:
            tr = max(
                c.high - c.low,
                abs(c.high - prev_close),
                abs(c.low - prev_close),
            )
        result.append(max(tr, EPS))
        prev_close = c.close
    return result


def atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    trs = true_ranges(candles)
    if not trs:
        return []
    if len(trs) <= period:
        seed = sum(trs) / len(trs)
        return [seed] * len(trs)

    result = [0.0] * len(trs)
    seed = sum(trs[:period]) / period
    result[period - 1] = seed
    prev = seed
    for i in range(period, len(trs)):
        prev = ((prev * (period - 1)) + trs[i]) / period
        result[i] = prev
    for i in range(period - 1):
        result[i] = seed
    return result


def rsi_series(candles: list[Candle], period: int = 14) -> list[float]:
    if not candles:
        return []
    closes = [c.close for c in candles]
    if len(closes) <= period:
        return [50.0] * len(closes)

    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    result = [50.0] * len(closes)

    def calc(g: float, l: float) -> float:
        if l <= EPS and g <= EPS:
            return 50.0
        if l <= EPS:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))

    result[period] = calc(avg_gain, avg_loss)
    for i in range(period + 1, len(closes)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        result[i] = calc(avg_gain, avg_loss)
    return result


def relative_volume(candles: list[Candle], lookback: int = 20) -> list[float]:
    result: list[float] = []
    for i, c in enumerate(candles):
        start = max(0, i - lookback)
        baseline = median_or([x.volume for x in candles[start:i]], c.volume)
        result.append(c.volume / max(baseline, EPS))
    return result


def volume_trend_score(candles: list[Candle], atr_values: list[float] | None = None) -> dict[str, Any]:
    if not candles:
        return {
            "direction": "NEUTRAL",
            "score": 50.0,
            "relative_volume": 1.0,
            "slope": 0.0,
            "signed_pressure": 0.0,
        }

    rel = relative_volume(candles, 20)
    recent = candles[-40:]
    signed_values: list[float] = []
    for c in recent:
        body = c.close - c.open
        signed_values.append((body / c.range) * c.volume)

    signed_pressure = sum(signed_values) / max(sum(c.volume for c in recent), EPS)
    slope = linear_slope([c.close for c in recent], min(20, len(recent)))
    current_rv = rel[-1]

    if signed_pressure > 0.08 and slope >= 0:
        direction = "BULLISH"
    elif signed_pressure < -0.08 and slope <= 0:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # Combine price slope and signed volume pressure. This is intentionally
    # OHLCV-only; no claim of bid/ask delta is made.
    pressure_score = clamp(50.0 + signed_pressure * 300.0)
    if direction == "BULLISH":
        score = clamp(pressure_score + max(0.0, min(current_rv - 1.0, 2.0)) * 10.0)
    elif direction == "BEARISH":
        score = clamp(100.0 - pressure_score + max(0.0, min(current_rv - 1.0, 2.0)) * 10.0)
    else:
        score = 50.0 + max(0.0, min(current_rv - 1.0, 2.0)) * 4.0

    return {
        "direction": direction,
        "score": clamp(score),
        "relative_volume": round(current_rv, 3),
        "slope": slope,
        "signed_pressure": signed_pressure,
    }


# ============================================================================
# SWINGS / STRUCTURE
# ============================================================================


def detect_pivots(candles: list[Candle], span: int) -> tuple[list[Pivot], list[Pivot]]:
    highs: list[Pivot] = []
    lows: list[Pivot] = []
    if len(candles) < 2 * span + 1:
        return highs, lows

    for i in range(span, len(candles) - span):
        center = candles[i]
        left = candles[i - span : i]
        right = candles[i + 1 : i + span + 1]

        high_values = [c.high for c in left + [center] + right]
        low_values = [c.low for c in left + [center] + right]

        if center.high >= max(high_values) and center.high > max(c.high for c in left):
            highs.append(
                Pivot(i, center.high, "HIGH", i + span)
            )

        if center.low <= min(low_values) and center.low < min(c.low for c in left):
            lows.append(
                Pivot(i, center.low, "LOW", i + span)
            )

    return highs, lows


def label_pivots(highs: list[Pivot], lows: list[Pivot]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    previous_high: float | None = None
    previous_low: float | None = None

    for pivot in sorted([*highs, *lows], key=lambda x: x.index):
        if pivot.kind == "HIGH":
            if previous_high is None:
                label = "SH"
            elif pivot.price > previous_high:
                label = "HH"
            elif pivot.price < previous_high:
                label = "LH"
            else:
                label = "EQH"
            previous_high = pivot.price
        else:
            if previous_low is None:
                label = "SL"
            elif pivot.price > previous_low:
                label = "HL"
            elif pivot.price < previous_low:
                label = "LL"
            else:
                label = "EQL"
            previous_low = pivot.price

        labels.append(
            {
                "index": pivot.index,
                "kind": pivot.kind,
                "price": round_price(pivot.price),
                "label": label,
            }
        )
    return labels


def displacement_strength(
    candles: list[Candle],
    index: int,
    atr_values: list[float],
    rel_volume: list[float],
) -> float:
    if index < 0 or index >= len(candles):
        return 0.0
    c = candles[index]
    atr = max(atr_values[index], EPS)
    range_ratio = c.range / atr
    body_ratio = c.body_ratio
    rv = rel_volume[index]

    score = 0.0
    score += clamp((range_ratio - 0.8) / 1.6 * 45.0)
    score += clamp((body_ratio - 0.45) / 0.5 * 30.0)
    score += clamp((rv - 0.8) / 1.5 * 25.0)
    return clamp(score)


def build_structure(
    candles: list[Candle],
    span: int,
) -> StructureSnapshot:
    highs, lows = detect_pivots(candles, span)
    labels = label_pivots(highs, lows)
    atr_values = atr_series(candles, 14)
    rel_vol = relative_volume(candles, 20)

    events: list[dict[str, Any]] = []
    high_ptr = 0
    low_ptr = 0
    current_high: Pivot | None = None
    current_low: Pivot | None = None
    broken_high_indices: set[int] = set()
    broken_low_indices: set[int] = set()
    trend = "RANGE"

    for i, candle in enumerate(candles):
        while high_ptr < len(highs) and highs[high_ptr].confirmed_at_index <= i:
            current_high = highs[high_ptr]
            high_ptr += 1
        while low_ptr < len(lows) and lows[low_ptr].confirmed_at_index <= i:
            current_low = lows[low_ptr]
            low_ptr += 1

        if current_high and current_high.index not in broken_high_indices and candle.close > current_high.price:
            prior_trend = trend
            if prior_trend == "BEARISH":
                event_type = "CHOCH"
            else:
                event_type = "BOS"
            disp = displacement_strength(candles, i, atr_values, rel_vol)
            if prior_trend == "BEARISH" and disp >= 60:
                event_type = "MSS"
                trend = "BULLISH"
            elif prior_trend in {"BULLISH", "RANGE"}:
                trend = "BULLISH"
            elif prior_trend == "BEARISH":
                trend = "BULLISH"
            events.append(
                {
                    "index": i,
                    "type": event_type,
                    "direction": "BULLISH",
                    "level": round_price(current_high.price),
                    "displacement": round(disp, 2),
                    "prior_trend": prior_trend,
                    "confirmed": True,
                }
            )
            broken_high_indices.add(current_high.index)

        if current_low and current_low.index not in broken_low_indices and candle.close < current_low.price:
            prior_trend = trend
            if prior_trend == "BULLISH":
                event_type = "CHOCH"
            else:
                event_type = "BOS"
            disp = displacement_strength(candles, i, atr_values, rel_vol)
            if prior_trend == "BULLISH" and disp >= 60:
                event_type = "MSS"
                trend = "BEARISH"
            elif prior_trend in {"BEARISH", "RANGE"}:
                trend = "BEARISH"
            elif prior_trend == "BULLISH":
                trend = "BEARISH"
            events.append(
                {
                    "index": i,
                    "type": event_type,
                    "direction": "BEARISH",
                    "level": round_price(current_low.price),
                    "displacement": round(disp, 2),
                    "prior_trend": prior_trend,
                    "confirmed": True,
                }
            )
            broken_low_indices.add(current_low.index)

    # If events are sparse, infer trend from the last two pivot relations.
    if trend == "RANGE":
        recent_labels = labels[-6:]
        high_bias = sum(1 for x in recent_labels if x["label"] == "HH") - sum(
            1 for x in recent_labels if x["label"] == "LH"
        )
        low_bias = sum(1 for x in recent_labels if x["label"] == "HL") - sum(
            1 for x in recent_labels if x["label"] == "LL"
        )
        if high_bias + low_bias >= 2:
            trend = "BULLISH"
        elif high_bias + low_bias <= -2:
            trend = "BEARISH"

    bullish_highs = [p for p in highs if p.index < len(candles)]
    bullish_lows = [p for p in lows if p.index < len(candles)]
    protected_high = bullish_highs[-1].price if bullish_highs else None
    protected_low = bullish_lows[-1].price if bullish_lows else None

    last_bos = next(
        (e for e in reversed(events) if e["type"] == "BOS"),
        None,
    )
    last_mss = next(
        (e for e in reversed(events) if e["type"] == "MSS"),
        None,
    )

    return StructureSnapshot(
        trend=trend,
        swing_highs=highs,
        swing_lows=lows,
        labels=labels,
        events=events,
        last_bos=last_bos,
        last_mss=last_mss,
        protected_high=protected_high,
        protected_low=protected_low,
    )


def latest_event(structure: StructureSnapshot, event_types: set[str], direction: str | None = None, max_age: int | None = None, candle_count: int | None = None) -> dict[str, Any] | None:
    for event in reversed(structure.events):
        if event["type"] not in event_types:
            continue
        if direction and event.get("direction") != direction:
            continue
        if max_age is not None and candle_count is not None:
            if candle_count - 1 - int(event["index"]) > max_age:
                continue
        return event
    return None



# ============================================================================
# TREND STRENGTH / FIBONACCI
# ============================================================================


def trend_strength_metrics(
    candles: list[Candle],
    structure: StructureSnapshot,
) -> dict[str, Any]:
    """
    Measure trend energy using the time taken to print new extremes.

    This follows the source material's operational idea: bullish strength is
    read from the progression/slope of highs, while bearish strength is read
    from the progression/slope of lows. It is a context measurement, not an
    entry trigger.
    """
    atr_values = atr_series(candles, 14)
    atr = median_or(atr_values[-30:], max(candles[-1].range, EPS))
    if structure.trend == "BULLISH":
        pivots = structure.swing_highs[-6:]
    elif structure.trend == "BEARISH":
        pivots = structure.swing_lows[-6:]
    else:
        pivots = []

    if len(pivots) < 2:
        return {
            "direction": structure.trend,
            "score": 50.0,
            "normalized_slope": 0.0,
            "pivot_count": len(pivots),
            "interpretation": "DATA_INSUFFICIENT",
        }

    x0 = pivots[0].index
    x1 = pivots[-1].index
    y0 = pivots[0].price
    y1 = pivots[-1].price
    bars = max(1, x1 - x0)
    price_per_bar = (y1 - y0) / bars
    normalized_slope = price_per_bar / max(atr, EPS)

    if structure.trend == "BULLISH":
        score = clamp(50.0 + normalized_slope * 55.0)
        interpretation = "BULLISH_STRONG" if score >= 70 else "BULLISH_MODERATE"
    elif structure.trend == "BEARISH":
        score = clamp(50.0 - normalized_slope * 55.0)
        interpretation = "BEARISH_STRONG" if score >= 70 else "BEARISH_MODERATE"
    else:
        score = 50.0
        interpretation = "RANGE"

    # Compare most recent impulse magnitude with recent corrective movement.
    impulse_ratio = 1.0
    if len(pivots) >= 3:
        impulse_move = abs(pivots[-1].price - pivots[-2].price)
        prior_move = abs(pivots[-2].price - pivots[-3].price)
        if prior_move > EPS:
            impulse_ratio = impulse_move / prior_move
            score = clamp(score + min(15.0, max(-15.0, (impulse_ratio - 1.0) * 20.0)))

    if structure.trend == "BULLISH":
        score = clamp(score)
    elif structure.trend == "BEARISH":
        score = clamp(score)

    return {
        "direction": structure.trend,
        "score": round(score, 2),
        "normalized_slope": round(normalized_slope, 6),
        "pivot_count": len(pivots),
        "bars_between_first_last_extreme": bars,
        "impulse_ratio": round(impulse_ratio, 4),
        "interpretation": interpretation,
    }


def _fib_anchor_from_structure(
    candles: list[Candle],
    structure: StructureSnapshot,
    direction: str,
) -> tuple[float, float, int, int, str] | None:
    """Choose a non-future-looking impulse for Fibonacci measurement."""
    events = [
        e for e in structure.events
        if e.get("direction") == ("BULLISH" if direction == "BUY" else "BEARISH")
        and e.get("type") in {"BOS", "MSS"}
    ]

    if events:
        event = events[-1]
        e_idx = int(event["index"])
        if direction == "BUY":
            lows = [p for p in structure.swing_lows if p.index < e_idx]
            if lows:
                low_pivot = lows[-1]
                segment = candles[low_pivot.index : e_idx + 1]
                if segment:
                    high_idx_local, high_candle = max(
                        enumerate(segment, start=low_pivot.index),
                        key=lambda item: item[1].high,
                    )
                    return (
                        low_pivot.price,
                        high_candle.high,
                        low_pivot.index,
                        high_idx_local,
                        f"Fib ditarik dari swing low sebelum {event['type']} bullish terakhir ke impuls yang menghasilkan breakout.",
                    )
        else:
            highs = [p for p in structure.swing_highs if p.index < e_idx]
            if highs:
                high_pivot = highs[-1]
                segment = candles[high_pivot.index : e_idx + 1]
                if segment:
                    low_idx_local, low_candle = min(
                        enumerate(segment, start=high_pivot.index),
                        key=lambda item: item[1].low,
                    )
                    return (
                        low_candle.low,
                        high_pivot.price,
                        low_idx_local,
                        high_pivot.index,
                        f"Fib ditarik dari impuls yang membentuk {event['type']} bearish terakhir, dari swing high ke swing low impuls tersebut.",
                    )

    # Fallback: use the latest confirmed directional swing pair.
    if direction == "BUY":
        lows = structure.swing_lows[-8:]
        highs = structure.swing_highs[-8:]
        pairs = [(lo, hi) for lo in lows for hi in highs if hi.index > lo.index]
        if pairs:
            lo, hi = max(pairs, key=lambda pair: pair[1].index)
            return (
                lo.price,
                hi.price,
                lo.index,
                hi.index,
                "Fib fallback memakai swing low → swing high terbaru yang terkonfirmasi.",
            )
    else:
        highs = structure.swing_highs[-8:]
        lows = structure.swing_lows[-8:]
        pairs = [(hi, lo) for hi in highs for lo in lows if lo.index > hi.index]
        if pairs:
            hi, lo = max(pairs, key=lambda pair: pair[1].index)
            return (
                lo.price,
                hi.price,
                lo.index,
                hi.index,
                "Fib fallback memakai swing high → swing low terbaru yang terkonfirmasi.",
            )
    return None


def build_fibonacci(
    candles: list[Candle],
    structure: StructureSnapshot,
    direction: str,
    timeframe: str,
) -> FibonacciContext | None:
    anchors = _fib_anchor_from_structure(candles, structure, direction)
    if not anchors:
        return None
    low, high, low_index, high_index, reason = anchors
    if high <= low:
        return None

    trend_strength = trend_strength_metrics(candles, structure)
    ratios = (0.0, 0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 1.0)
    levels = {f"{ratio:.3f}": round_price(
        high - ratio * (high - low) if direction == "BUY" else low + ratio * (high - low)
    ) for ratio in ratios}

    return FibonacciContext(
        timeframe=timeframe,
        direction=direction,
        swing_low=low,
        swing_high=high,
        low_index=low_index,
        high_index=high_index,
        anchor_reason=reason,
        trend_strength=trend_strength,
        levels=levels,
    )


def fib_zone_for_direction(
    fib: FibonacciContext,
    low_ratio: float = FIB_DEEP_RATIO,
    high_ratio: float = FIB_DEEPEST_RATIO,
) -> tuple[float, float]:
    p1 = fib.price_at(low_ratio)
    p2 = fib.price_at(high_ratio)
    return min(p1, p2), max(p1, p2)


def fib_confluence_score(
    zone: Zone,
    fib: FibonacciContext | None,
    direction: str,
) -> tuple[float, dict[str, Any]]:
    if fib is None:
        return 50.0, {"available": False}

    midpoint_ratio = fib.retracement_ratio(zone.midpoint)
    zone_low_ratio = fib.retracement_ratio(zone.low)
    zone_high_ratio = fib.retracement_ratio(zone.high)
    ratio_min = min(zone_low_ratio, zone_high_ratio)
    ratio_max = max(zone_low_ratio, zone_high_ratio)

    # Required directional context: BUY values deeper in discount, SELL values
    # deeper in premium. A shallower zone is still possible, but it scores less.
    if ratio_min >= 0.618 and ratio_max <= 0.90:
        score = 96.0
        band = "DEEP_0.618_0.786"
    elif ratio_max >= 0.618 and ratio_min <= 0.786:
        score = 92.0
        band = "OVERLAPS_0.618"
    elif 0.50 <= midpoint_ratio < 0.618:
        score = 82.0
        band = "0.500_0.618"
    elif 0.382 <= midpoint_ratio < 0.50:
        score = 72.0
        band = "0.382_0.500"
    elif midpoint_ratio < 0.382:
        score = 45.0
        band = "SHALLOW"
    else:
        score = 65.0
        band = "DEEPER_THAN_0.786"

    # Trend strength changes the *preferred* retracement depth, but does not
    # invalidate the source's 0.618 filter concept.
    ts = safe_float(fib.trend_strength.get("score"), 50.0)
    if ts >= 75 and 0.382 <= midpoint_ratio <= 0.618:
        score += 4.0
    elif ts < 55 and midpoint_ratio >= 0.618:
        score += 4.0

    return clamp(score), {
        "available": True,
        "midpoint_ratio": round(midpoint_ratio, 4),
        "zone_ratio_min": round(ratio_min, 4),
        "zone_ratio_max": round(ratio_max, 4),
        "band": band,
        "deep_filter_ratio": FIB_DEEP_RATIO,
        "preferred_by_trend_strength": (
            "0.382-0.618" if ts >= 75 else
            "0.618-0.786" if ts < 55 else
            "0.500-0.618"
        ),
        "trend_strength": round(ts, 2),
        "direction": direction,
    }


def zone_overlap_ratio(a: Zone, b: Zone) -> float:
    overlap = max(0.0, min(a.high, b.high) - max(a.low, b.low))
    smaller = max(min(a.high - a.low, b.high - b.low), EPS)
    return clamp(overlap / smaller, 0.0, 1.0)


def fib_level_confluence(
    zone: Zone,
    fib: FibonacciContext | None,
    direction: str,
) -> float:
    if fib is None:
        return 0.0
    fib_zone_low, fib_zone_high = fib_zone_for_direction(fib, 0.618, 0.786)
    synthetic = Zone(
        kind="FIB_DEEP_ZONE",
        low=fib_zone_low,
        high=fib_zone_high,
        index=0,
        source="FIBONACCI",
    )
    return zone_overlap_ratio(zone, synthetic) * 100.0


def zone_quality_score(
    zone: Zone,
    structure: StructureSnapshot,
    fib: FibonacciContext | None,
    liquidity: list[LiquidityPool],
    sweeps: list[dict[str, Any]],
    direction: str,
    candles: list[Candle],
) -> tuple[float, dict[str, Any]]:
    wanted = "BULLISH" if direction == "BUY" else "BEARISH"
    score = clamp(zone.strength)
    details: dict[str, Any] = {
        "zone_kind": zone.kind,
        "status": zone.status,
        "source": zone.source,
    }

    if zone.status == "ACTIVE":
        score += 10.0
    elif zone.status == "MITIGATED":
        score -= 12.0
    elif zone.status == "FAILED":
        score -= 45.0

    be = zone.details.get("break_event")
    if isinstance(be, dict):
        if be.get("type") == "MSS":
            score += 12.0
        elif be.get("type") == "BOS":
            score += 8.0
        score += min(12.0, safe_float(be.get("displacement"), 0.0) * 0.12)

    formation_direction = str(zone.details.get("formation_direction") or "")
    if formation_direction == wanted:
        score += 8.0
    elif formation_direction and formation_direction != wanted:
        score -= 8.0

    if zone.kind.endswith("FVG"):
        if zone.details.get("breakaway_gap"):
            score += 8.0
        if zone.details.get("rejection_gap"):
            score -= 10.0

    # A POI formed after a relevant liquidity sweep gets additional context
    # weight. We use price/time proximity only; no intent is inferred.
    candidate_sweeps = [
        s for s in sweeps
        if s.get("direction") == wanted
        and int(s.get("index", -999999)) < zone.index
        and zone.index - int(s.get("index", -999999)) <= 20
    ]
    if candidate_sweeps:
        score += 12.0
        details["formed_after_liquidity_sweep"] = True
    else:
        details["formed_after_liquidity_sweep"] = False

    fib_score, fib_details = fib_confluence_score(zone, fib, direction)
    score += (fib_score - 50.0) * 0.25
    details["fibonacci"] = fib_details
    details["fibonacci_overlap_deep_zone"] = round(fib_level_confluence(zone, fib, direction), 2)

    # Prefer zones with a nearby liquidity objective/context.
    nearby = [
        p for p in liquidity
        if abs(p.level - zone.midpoint) <= max(zone.high - zone.low, EPS) * 2.0
    ]
    if nearby:
        score += min(8.0, max(p.strength for p in nearby) * 0.08)
        details["nearby_liquidity"] = True
    else:
        details["nearby_liquidity"] = False

    if zone.index >= 0 and candles:
        details["age_candles"] = max(0, len(candles) - 1 - zone.index)

    return clamp(score), details


def htf_poi_candidates(
    candles: list[Candle],
    structure: StructureSnapshot,
    direction: str,
    timeframe: str,
) -> dict[str, Any]:
    atr_values = atr_series(candles, 14)
    rel_vol = relative_volume(candles, 20)
    fvgs = fvg_zones(candles, atr_values)
    obs = find_ob_for_displacement(candles, structure, atr_values, rel_vol)
    breakers = breaker_zones(obs, candles)
    pools = build_liquidity_pools(candles, structure)
    sweeps = detect_sweeps(candles, pools, atr_values, min(120, len(candles)))
    fib_buy = build_fibonacci(candles, structure, "BUY", timeframe)
    fib_sell = build_fibonacci(candles, structure, "SELL", timeframe)

    zones = [*fvgs, *obs, *breakers]
    for zone in zones:
        zone.source = f"{timeframe}_{zone.source}"
        quality_direction = "BUY" if "BULLISH" in zone.kind else "SELL"
        fib = fib_buy if quality_direction == "BUY" else fib_sell
        q, qd = zone_quality_score(
            zone,
            structure,
            fib,
            pools,
            sweeps,
            quality_direction,
            candles,
        )
        zone.strength = q
        zone.details["timeframe"] = timeframe
        zone.details["quality"] = qd

    return {
        "timeframe": timeframe,
        "atr": atr_values[-1] if atr_values else 0.0,
        "structure": structure,
        "fvg": fvgs,
        "obs": obs,
        "breakers": breakers,
        "zones": zones,
        "liquidity": pools,
        "sweeps": sweeps,
        "fib": {"BUY": fib_buy, "SELL": fib_sell},
        "trend_strength": trend_strength_metrics(candles, structure),
    }


def select_primary_poi(
    ctx: dict[str, Any],
    direction: str,
    current: float,
) -> tuple[Zone | None, dict[str, Any]]:
    wanted = "BULLISH" if direction == "BUY" else "BEARISH"
    fib = ctx.get("fib", {}).get(direction)
    atr = safe_float(ctx.get("atr"), 0.0)
    zones = [
        z for z in ctx.get("zones", [])
        if wanted in z.kind and z.status != "FAILED"
    ]

    scored: list[tuple[float, Zone, dict[str, Any]]] = []
    for zone in zones:
        dist = abs(current - zone.midpoint) / max(atr, EPS)
        if dist > H4_POI_MAX_DISTANCE_ATR:
            continue
        quality = safe_float(zone.strength, 50.0)
        fib_score, fib_details = fib_confluence_score(zone, fib, direction)
        freshness_bonus = 12.0 if zone.status == "ACTIVE" else 0.0
        distance_penalty = min(30.0, max(0.0, dist - 1.0) * 4.0)
        overlap = fib_level_confluence(zone, fib, direction)
        total = quality * 0.55 + fib_score * 0.25 + overlap * 0.10 + freshness_bonus - distance_penalty
        scored.append((total, zone, {
            "score": round(clamp(total), 2),
            "distance_atr": round(dist, 3),
            "fib": fib_details,
            "freshness_bonus": freshness_bonus,
            "distance_penalty": round(distance_penalty, 2),
        }))

    if not scored:
        return None, {"available": False, "reason": "Tidak ada H4 POI aktif yang cukup dekat."}
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_zone, details = scored[0]
    details.update({
        "available": True,
        "timeframe": ctx.get("timeframe"),
        "type": best_zone.kind,
        "low": round_price(best_zone.low),
        "high": round_price(best_zone.high),
        "midpoint": round_price(best_zone.midpoint),
        "quality": round(best_score, 2),
        "status": best_zone.status,
    })
    return best_zone, details


def select_refinement_poi(
    ctx: dict[str, Any],
    direction: str,
    primary: Zone | None,
    current: float,
) -> tuple[Zone | None, dict[str, Any]]:
    wanted = "BULLISH" if direction == "BUY" else "BEARISH"
    fib = ctx.get("fib", {}).get(direction)
    atr = safe_float(ctx.get("atr"), 0.0)
    zones = [
        z for z in ctx.get("zones", [])
        if wanted in z.kind and z.status != "FAILED"
    ]
    scored: list[tuple[float, Zone, dict[str, Any]]] = []
    for zone in zones:
        if primary is not None:
            overlap = zone_overlap_ratio(zone, primary)
            center_dist_atr = abs(zone.midpoint - primary.midpoint) / max(atr, EPS)
            if overlap <= 0 and center_dist_atr > H1_REFINEMENT_OVERLAP_ATR:
                continue
        else:
            overlap = 0.0
            center_dist_atr = abs(zone.midpoint - current) / max(atr, EPS)
            if center_dist_atr > H1_POI_MAX_DISTANCE_ATR:
                continue

        fib_score, fib_details = fib_confluence_score(zone, fib, direction)
        quality = safe_float(zone.strength, 50.0)
        freshness = 10.0 if zone.status == "ACTIVE" else 0.0
        total = quality * 0.55 + fib_score * 0.25 + overlap * 20.0 + freshness
        scored.append((total, zone, {
            "score": round(clamp(total), 2),
            "overlap_with_h4": round(overlap, 3),
            "center_distance_atr": round(center_dist_atr, 3),
            "fib": fib_details,
            "freshness_bonus": freshness,
        }))

    if not scored:
        return None, {"available": False, "reason": "Tidak ada H1 refinement yang beririsan/dekat."}
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_zone, details = scored[0]
    details.update({
        "available": True,
        "timeframe": ctx.get("timeframe"),
        "type": best_zone.kind,
        "low": round_price(best_zone.low),
        "high": round_price(best_zone.high),
        "midpoint": round_price(best_zone.midpoint),
        "quality": round(best_score, 2),
        "status": best_zone.status,
    })
    return best_zone, details


def select_execution_poi(
    ctx: dict[str, Any],
    direction: str,
    parent_zone: Zone | None,
    current: float,
) -> tuple[Zone | None, dict[str, Any]]:
    wanted = "BULLISH" if direction == "BUY" else "BEARISH"
    atr = safe_float(ctx.get("atr"), 0.0)
    zones = [
        z for z in ctx.get("zones", [])
        if wanted in z.kind and z.status != "FAILED"
    ]
    scored: list[tuple[float, Zone, dict[str, Any]]] = []
    for zone in zones:
        if direction == "BUY" and zone.low >= current:
            continue
        if direction == "SELL" and zone.high <= current:
            continue
        if parent_zone is not None:
            overlap = zone_overlap_ratio(zone, parent_zone)
            center_dist_atr = abs(zone.midpoint - parent_zone.midpoint) / max(atr, EPS)
            if overlap <= 0 and center_dist_atr > M15_EXECUTION_OVERLAP_ATR:
                continue
        else:
            overlap = 0.0
            center_dist_atr = abs(zone.midpoint - current) / max(atr, EPS)
        freshness = 12.0 if zone.status == "ACTIVE" else 0.0
        total = safe_float(zone.strength, 50.0) * 0.70 + overlap * 25.0 + freshness - min(15.0, center_dist_atr * 1.5)
        scored.append((total, zone, {
            "score": round(clamp(total), 2),
            "overlap_with_parent": round(overlap, 3),
            "center_distance_atr": round(center_dist_atr, 3),
        }))

    if not scored:
        return None, {"available": False, "reason": "Tidak ada execution POI M15 yang berada di parent zone dan sisi entry."}
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_zone, details = scored[0]
    details.update({
        "available": True,
        "timeframe": ctx.get("timeframe"),
        "type": best_zone.kind,
        "low": round_price(best_zone.low),
        "high": round_price(best_zone.high),
        "midpoint": round_price(best_zone.midpoint),
        "quality": round(best_score, 2),
        "status": best_zone.status,
    })
    return best_zone, details


def fibonacci_summary(fib: FibonacciContext | None, current: float) -> dict[str, Any]:
    if fib is None:
        return {"available": False}
    ratio = fib.retracement_ratio(current)
    return {
        "available": True,
        "timeframe": fib.timeframe,
        "direction": fib.direction,
        "swing_low": round_price(fib.swing_low),
        "swing_high": round_price(fib.swing_high),
        "low_index": fib.low_index,
        "high_index": fib.high_index,
        "current_retracement_ratio": round(ratio, 4),
        "current_retracement_pct": round(ratio * 100.0, 2),
        "levels": fib.levels,
        "anchor_reason": fib.anchor_reason,
        "trend_strength": fib.trend_strength,
        "deep_zone_0618_0786": {
            "low": round_price(fib_zone_for_direction(fib, 0.618, 0.786)[0]),
            "high": round_price(fib_zone_for_direction(fib, 0.618, 0.786)[1]),
        },
    }


def find_target_liquidity_topdown(
    pools: list[LiquidityPool],
    direction: str,
    entry: float,
    atr: float,
) -> LiquidityPool | None:
    wanted = "BUY_SIDE" if direction == "BUY" else "SELL_SIDE"
    candidates = []
    tf_priority = {"H4": 4, "H1": 3, "M15": 1}
    for p in pools:
        if p.kind != wanted:
            continue
        if direction == "BUY" and p.level <= entry:
            continue
        if direction == "SELL" and p.level >= entry:
            continue
        dist = abs(p.level - entry) / max(atr, EPS)
        if dist > 8.0:
            continue
        source = str(p.details.get("timeframe") or "M15")
        priority = tf_priority.get(source.split("_")[0], 1)
        candidates.append((priority, p.strength, -dist, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][3]


def structural_invalidation_level(
    direction: str,
    entry: float,
    execution: Zone | None,
    refinement: Zone | None,
    primary: Zone | None,
    m15_structure: StructureSnapshot,
    trigger_sweep: dict[str, Any] | None,
) -> float:
    levels: list[float] = []
    if direction == "BUY":
        if trigger_sweep and safe_float(trigger_sweep.get("level"), 0.0) < entry:
            levels.append(safe_float(trigger_sweep.get("level")))
        for pivot in reversed(m15_structure.swing_lows[-8:]):
            if pivot.price < entry:
                levels.append(pivot.price)
                break
        for zone in (execution, refinement, primary):
            if zone and zone.low < entry:
                levels.append(zone.low)
        return max(levels) if levels else entry

    if trigger_sweep and safe_float(trigger_sweep.get("level"), 0.0) > entry:
        levels.append(safe_float(trigger_sweep.get("level")))
    for pivot in reversed(m15_structure.swing_highs[-8:]):
        if pivot.price > entry:
            levels.append(pivot.price)
            break
    for zone in (execution, refinement, primary):
        if zone and zone.high > entry:
            levels.append(zone.high)
    return min(levels) if levels else entry


def topdown_candidate(
    *,
    pair: str,
    direction: str,
    current: float,
    m15_atr: float,
    primary: Zone | None,
    refinement: Zone | None,
    execution: Zone | None,
    h4_ctx: dict[str, Any],
    h1_ctx: dict[str, Any],
    m15_ctx: dict[str, Any],
    m15_structure: StructureSnapshot,
    target: LiquidityPool | None,
    sweep: dict[str, Any] | None,
    mss: dict[str, Any] | None,
    bos: dict[str, Any] | None,
    rsi: float,
    vlt_direction: str,
) -> Candidate | None:
    entry_zone = execution or refinement or primary
    if entry_zone is None:
        return None

    # The final entry must remain on the retracement side required by main.py.
    if direction == "BUY":
        if entry_zone.midpoint < current:
            entry = entry_zone.midpoint
        elif entry_zone.low < current:
            entry = max(entry_zone.low, current - max(m15_atr * 0.20, (entry_zone.high - entry_zone.low) * 0.25))
        else:
            return None
    else:
        if entry_zone.midpoint > current:
            entry = entry_zone.midpoint
        elif entry_zone.high > current:
            entry = min(entry_zone.high, current + max(m15_atr * 0.20, (entry_zone.high - entry_zone.low) * 0.25))
        else:
            return None

    structural = structural_invalidation_level(
        direction,
        entry,
        execution,
        refinement,
        primary,
        m15_structure,
        sweep,
    )
    if direction == "BUY":
        sl = structural - m15_atr * SL_BUFFER_ATR
        if sl >= entry:
            sl = entry - max(m15_atr * 0.60, entry * 0.002)
    else:
        sl = structural + m15_atr * SL_BUFFER_ATR
        if sl <= entry:
            sl = entry + max(m15_atr * 0.60, entry * 0.002)

    tp = target.level if target else 0.0
    if direction == "BUY":
        if tp <= entry:
            tp = max(current + 1.75 * m15_atr, entry + 1.75 * m15_atr)
        tp = max(tp, entry + 1.25 * m15_atr)
        price_exp = current + max(
            EXP_ATR_MULTIPLIER * m15_atr,
            abs(current - entry) * 1.20,
            (tp - current) * 0.28,
        )
        if tp > current:
            price_exp = min(price_exp, current + (tp - current) * 0.60)
        if price_exp <= current:
            price_exp = current + max(m15_atr * 0.6, current * 0.001)
        if price_exp >= tp:
            price_exp = current + max(m15_atr * 0.60, (tp - current) * 0.35)
        if not (sl < entry < current < price_exp < tp):
            return None
    else:
        if tp >= entry or tp <= 0:
            tp = min(current - 1.75 * m15_atr, entry - 1.75 * m15_atr)
        tp = min(tp, entry - 1.25 * m15_atr)
        price_exp = current - max(
            EXP_ATR_MULTIPLIER * m15_atr,
            abs(current - entry) * 1.20,
            (current - tp) * 0.28,
        )
        if tp < current:
            price_exp = max(price_exp, current - (current - tp) * 0.60)
        if price_exp >= current:
            price_exp = current - max(m15_atr * 0.60, current * 0.001)
        if price_exp <= tp:
            price_exp = current - max(m15_atr * 0.60, (current - tp) * 0.35)
        if not (tp < price_exp < current < entry < sl):
            return None

    model = "HTF_POI_M15_MSS" if primary and (mss or sweep) else "HTF_POI_M15_REFINEMENT"
    hierarchy = {
        "primary_h4": _zone_evidence(primary, h4_ctx.get("atr"), current),
        "refinement_h1": _zone_evidence(refinement, h1_ctx.get("atr"), current),
        "execution_m15": _zone_evidence(execution, m15_ctx.get("atr"), current),
    }
    fib_info = {
        "h4": fib_confluence_score(primary, h4_ctx.get("fib", {}).get(direction), direction)[1] if primary else {"available": False},
        "h1": fib_confluence_score(refinement, h1_ctx.get("fib", {}).get(direction), direction)[1] if refinement else {"available": False},
    }
    evidence = {
        "top_down": hierarchy,
        "fibonacci": fib_info,
        "sweep": sweep,
        "mss": mss,
        "bos": bos,
        "target_liquidity": _liquidity_evidence(target),
        "direction": direction,
        "btc_direction_constraint": True,
        "pair_h4_trend": h4_ctx.get("structure").trend if h4_ctx.get("structure") else None,
        "notes_architecture": "H4 primary POI -> H1 refinement -> M15 execution confirmation",
    }
    entry_reason = _topdown_entry_reason(
        pair,
        direction,
        primary,
        refinement,
        execution,
        h4_ctx,
        h1_ctx,
        sweep,
        mss,
        bos,
        rsi,
        vlt_direction,
    )
    sl_reason = (
        f"SL ditempatkan di bawah/atas structural invalidation terdekat "
        f"({round_price(structural)}) + buffer {SL_BUFFER_ATR:.2f} ATR M15."
    )
    tp_reason = (
        f"TP diarahkan ke {target.source} {round_price(target.level)} sebagai liquidity target HTF."
        if target else
        f"TP fallback {round_price(tp)} menggunakan range/ATR setelah target liquidity HTF tidak tersedia."
    )
    exp_reason = (
        f"Price Exp {round_price(price_exp)} adalah batas ekspansi thesis H4/H1 sebelum entry. "
        "Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru."
    )

    return Candidate(
        direction=direction,
        model=model,
        entry=round_price(entry),
        sl=round_price(sl),
        tp=round_price(tp),
        price_exp=round_price(price_exp),
        entry_reason=entry_reason,
        sl_reason=sl_reason,
        tp_reason=tp_reason,
        price_exp_reason=exp_reason,
        evidence=evidence,
    )


def _zone_evidence(zone: Zone | None, atr: float | None, current: float | None = None) -> dict[str, Any]:
    if zone is None:
        return {"available": False}
    a = safe_float(atr, 0.0)
    return {
        "available": True,
        "timeframe": zone.details.get("timeframe"),
        "kind": zone.kind,
        "low": round_price(zone.low),
        "high": round_price(zone.high),
        "midpoint": round_price(zone.midpoint),
        "status": zone.status,
        "strength": round(zone.strength, 2),
        "distance_atr_to_current": round(abs(zone.midpoint - safe_float(current, zone.midpoint)) / max(a, EPS), 3) if a > 0 and current is not None else None,
        "details": zone.details,
    }


def _liquidity_evidence(pool: LiquidityPool | None) -> dict[str, Any]:
    if pool is None:
        return {"available": False}
    return {
        "available": True,
        "kind": pool.kind,
        "level": round_price(pool.level),
        "strength": round(pool.strength, 2),
        "source": pool.source,
        "distance_pct": round(pool.distance_pct, 4),
        "distance_atr": round(pool.distance_atr, 3),
        "timeframe": pool.details.get("timeframe"),
    }


def _topdown_entry_reason(
    pair: str,
    direction: str,
    primary: Zone | None,
    refinement: Zone | None,
    execution: Zone | None,
    h4_ctx: dict[str, Any],
    h1_ctx: dict[str, Any],
    sweep: dict[str, Any] | None,
    mss: dict[str, Any] | None,
    bos: dict[str, Any] | None,
    rsi: float,
    vlt_direction: str,
) -> str:
    fragments = [
        f"{pair} {direction}: thesis dimulai dari POI H4",
        f"({primary.kind} {round_price(primary.midpoint)})" if primary else "(H4 POI tidak tersedia)",
    ]
    if refinement:
        fragments.append(f"H1 me-refine ke {refinement.kind} {round_price(refinement.midpoint)}")
    if execution:
        fragments.append(f"M15 memberi execution POI {execution.kind} {round_price(execution.midpoint)}")
    fib = h4_ctx.get("fib", {}).get(direction)
    if primary and fib:
        fib_score, fib_details = fib_confluence_score(primary, fib, direction)
        if fib_details.get("band"):
            fragments.append(f"Fib H4 berada pada band {fib_details['band']} (score {fib_score:.0f})")
    if sweep:
        fragments.append("terdapat liquidity sweep searah reversal")
    if mss:
        fragments.append("diikuti MSS pada M15")
    elif bos:
        fragments.append("diikuti BOS pada M15")
    if rsi >= 50 and direction == "BUY":
        fragments.append(f"RSI 14 M15 {rsi:.1f} mendukung momentum")
    elif rsi <= 50 and direction == "SELL":
        fragments.append(f"RSI 14 M15 {rsi:.1f} mendukung momentum")
    if vlt_direction == ("BULLISH" if direction == "BUY" else "BEARISH"):
        fragments.append("VLT/OHLCV searah dengan setup")
    return ". ".join(fragments) + "."

# ============================================================================
# LIQUIDITY
# ============================================================================


def liquidity_tolerance(price: float, atr: float) -> float:
    return max(price * 0.001, atr * 0.15)


def find_equal_pools(
    pivots: list[Pivot],
    price_now: float,
    atr: float,
    kind: str,
) -> list[LiquidityPool]:
    pools: list[LiquidityPool] = []
    sorted_pivots = sorted(pivots, key=lambda x: x.index)
    tol = liquidity_tolerance(price_now, atr)

    for i in range(len(sorted_pivots) - 1):
        a = sorted_pivots[i]
        for j in range(i + 1, min(i + 6, len(sorted_pivots))):
            b = sorted_pivots[j]
            if abs(a.price - b.price) > tol:
                continue
            level = (a.price + b.price) / 2.0
            strength = clamp(
                60.0
                + min(25.0, (j - i) * 5.0)
                + (10.0 if b.index - a.index >= 3 else 0.0)
            )
            pools.append(
                LiquidityPool(
                    kind=kind,
                    level=level,
                    index=b.index,
                    strength=strength,
                    source="EQUAL_LEVELS",
                    details={
                        "first_index": a.index,
                        "second_index": b.index,
                        "distance": abs(a.price - b.price),
                    },
                )
            )
    # Keep strongest/newest levels.
    pools.sort(key=lambda p: (p.strength, p.index), reverse=True)
    deduped: list[LiquidityPool] = []
    for pool in pools:
        if any(abs(pool.level - existing.level) <= tol for existing in deduped):
            continue
        deduped.append(pool)
        if len(deduped) >= 8:
            break
    return deduped


def build_liquidity_pools(
    candles: list[Candle],
    structure: StructureSnapshot,
) -> list[LiquidityPool]:
    if not candles:
        return []
    price = candles[-1].close
    atr = atr_series(candles, 14)[-1]

    pools: list[LiquidityPool] = []
    pools.extend(
        find_equal_pools(
            structure.swing_highs[-20:], price, atr, "BUY_SIDE",
        )
    )
    pools.extend(
        find_equal_pools(
            structure.swing_lows[-20:], price, atr, "SELL_SIDE",
        )
    )

    for p in structure.swing_highs[-12:]:
        pools.append(
            LiquidityPool(
                kind="BUY_SIDE",
                level=p.price,
                index=p.index,
                strength=55.0,
                source="SWING_HIGH",
            )
        )
    for p in structure.swing_lows[-12:]:
        pools.append(
            LiquidityPool(
                kind="SELL_SIDE",
                level=p.price,
                index=p.index,
                strength=55.0,
                source="SWING_LOW",
            )
        )

    # Recent range boundaries act as broader external liquidity references.
    recent = candles[-96:] if len(candles) >= 96 else candles
    pools.append(
        LiquidityPool(
            kind="BUY_SIDE",
            level=max(c.high for c in recent),
            index=len(candles) - len(recent),
            strength=70.0,
            source="RECENT_RANGE_HIGH",
        )
    )
    pools.append(
        LiquidityPool(
            kind="SELL_SIDE",
            level=min(c.low for c in recent),
            index=len(candles) - len(recent),
            strength=70.0,
            source="RECENT_RANGE_LOW",
        )
    )

    tol = liquidity_tolerance(price, atr)
    deduped: list[LiquidityPool] = []
    for pool in sorted(pools, key=lambda x: (x.strength, x.index), reverse=True):
        if pool.level <= 0:
            continue
        if any(pool.kind == x.kind and abs(pool.level - x.level) <= tol for x in deduped):
            continue
        pool.distance_pct = distance_pct(price, pool.level)
        pool.distance_atr = abs(pool.level - price) / max(atr, EPS)
        deduped.append(pool)
    return deduped[:30]


def detect_sweeps(
    candles: list[Candle],
    pools: list[LiquidityPool],
    atr_values: list[float],
    lookback: int = 80,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start = max(1, len(candles) - lookback)

    relevant = pools[:20]
    for i in range(start, len(candles)):
        c = candles[i]
        atr = atr_values[i]
        for pool in relevant:
            tol = max(pool.level * 0.0002, atr * 0.03)
            if pool.kind == "SELL_SIDE":
                swept = c.low < pool.level - tol and c.close > pool.level
                if swept:
                    result.append(
                        {
                            "index": i,
                            "direction": "BULLISH",
                            "type": "SELL_SIDE_SWEEP",
                            "level": round_price(pool.level),
                            "pool_source": pool.source,
                            "pool_strength": pool.strength,
                            "rejection": (c.close - c.low) / max(c.range, EPS),
                        }
                    )
            elif pool.kind == "BUY_SIDE":
                swept = c.high > pool.level + tol and c.close < pool.level
                if swept:
                    result.append(
                        {
                            "index": i,
                            "direction": "BEARISH",
                            "type": "BUY_SIDE_SWEEP",
                            "level": round_price(pool.level),
                            "pool_source": pool.source,
                            "pool_strength": pool.strength,
                            "rejection": (c.high - c.close) / max(c.range, EPS),
                        }
                    )
    return result


# ============================================================================
# FVG / ORDER BLOCK / BREAKER
# ============================================================================


def fvg_zones(candles: list[Candle], atr_values: list[float]) -> list[Zone]:
    zones: list[Zone] = []
    for i in range(1, len(candles) - 1):
        a, _b, c = candles[i - 1], candles[i], candles[i + 1]
        atr = atr_values[i]

        if c.low > a.high:
            gap_low = a.high
            gap_high = c.low
            size = gap_high - gap_low
            strength = clamp(
                55.0
                + min(30.0, size / max(atr, EPS) * 35.0)
                + min(15.0, _b.body_ratio * 15.0)
            )
            zones.append(
                Zone(
                    kind="BULLISH_FVG",
                    low=gap_low,
                    high=gap_high,
                    index=i,
                    source="FVG",
                    strength=strength,
                    details={
                        "gap_atr": size / max(atr, EPS),
                        "formation_direction": "BULLISH" if c.bullish else "BEARISH",
                        "breakaway_gap": bool(c.bullish and c.body_ratio >= 0.55),
                        "rejection_gap": bool(not c.bullish),
                    },
                )
            )

        if c.high < a.low:
            gap_low = c.high
            gap_high = a.low
            size = gap_high - gap_low
            strength = clamp(
                55.0
                + min(30.0, size / max(atr, EPS) * 35.0)
                + min(15.0, _b.body_ratio * 15.0)
            )
            zones.append(
                Zone(
                    kind="BEARISH_FVG",
                    low=gap_low,
                    high=gap_high,
                    index=i,
                    source="FVG",
                    strength=strength,
                    details={
                        "gap_atr": size / max(atr, EPS),
                        "formation_direction": "BEARISH" if c.bearish else "BULLISH",
                        "breakaway_gap": bool(c.bearish and c.body_ratio >= 0.55),
                        "rejection_gap": bool(not c.bearish),
                    },
                )
            )

    # Determine mitigation after creation.
    for zone in zones:
        for j in range(zone.index + 2, len(candles)):
            c = candles[j]
            overlaps = c.low <= zone.high and c.high >= zone.low
            if overlaps:
                zone.status = "MITIGATED"
                zone.details["mitigated_at"] = j
                break
    return zones


def find_ob_for_displacement(
    candles: list[Candle],
    structure: StructureSnapshot,
    atr_values: list[float],
    rel_vol: list[float],
) -> list[Zone]:
    zones: list[Zone] = []
    for event in structure.events[-40:]:
        if event["type"] not in {"BOS", "MSS"}:
            continue
        i = int(event["index"])
        direction = event["direction"]
        if i <= 0:
            continue

        search_start = max(0, i - 6)
        origin: int | None = None
        if direction == "BULLISH":
            for j in range(i - 1, search_start - 1, -1):
                if candles[j].bearish:
                    origin = j
                    break
            if origin is None:
                continue
            zone = Zone(
                kind="BULLISH_OB",
                low=candles[origin].low,
                high=candles[origin].high,
                index=origin,
                source="ORDER_BLOCK",
                strength=clamp(60.0 + event.get("displacement", 0) * 0.35),
                details={
                    "break_event": event,
                    "relative_volume": round(rel_vol[i], 3),
                    "formation_direction": "BULLISH",
                },
            )
        else:
            for j in range(i - 1, search_start - 1, -1):
                if candles[j].bullish:
                    origin = j
                    break
            if origin is None:
                continue
            zone = Zone(
                kind="BEARISH_OB",
                low=candles[origin].low,
                high=candles[origin].high,
                index=origin,
                source="ORDER_BLOCK",
                strength=clamp(60.0 + event.get("displacement", 0) * 0.35),
                details={
                    "break_event": event,
                    "relative_volume": round(rel_vol[i], 3),
                    "formation_direction": "BEARISH",
                },
            )

        # A return into the zone is mitigation; a later close through the
        # entire zone is failure and can create a breaker block.
        for j in range(i + 1, len(candles)):
            c = candles[j]
            overlaps = c.low <= zone.high and c.high >= zone.low
            if overlaps and zone.status == "ACTIVE":
                zone.status = "MITIGATED"
                zone.details["mitigated_at"] = j
            if zone.kind == "BULLISH_OB" and c.close < zone.low:
                zone.status = "FAILED"
                zone.details["failed_at"] = j
                break
            if zone.kind == "BEARISH_OB" and c.close > zone.high:
                zone.status = "FAILED"
                zone.details["failed_at"] = j
                break
        zones.append(zone)

    # Keep newest distinct zones.
    zones.sort(key=lambda z: (z.strength, z.index), reverse=True)
    deduped: list[Zone] = []
    for zone in zones:
        if any(
            zone.kind == x.kind
            and abs(zone.midpoint - x.midpoint) <= max(zone.high - zone.low, EPS) * 0.5
            for x in deduped
        ):
            continue
        deduped.append(zone)
    return deduped[:16]


def breaker_zones(order_blocks: list[Zone], candles: list[Candle]) -> list[Zone]:
    result: list[Zone] = []
    for ob in order_blocks:
        if ob.status != "FAILED":
            continue
        failed_at = int(ob.details.get("failed_at", -1))
        if failed_at < 0:
            continue
        if ob.kind == "BULLISH_OB":
            # Failed bullish OB becomes a bearish breaker.
            result.append(
                Zone(
                    kind="BEARISH_BREAKER",
                    low=ob.low,
                    high=ob.high,
                    index=failed_at,
                    source="BREAKER",
                    strength=clamp(ob.strength - 5),
                    details={"from": ob.kind, "failed_at": failed_at},
                )
            )
        elif ob.kind == "BEARISH_OB":
            result.append(
                Zone(
                    kind="BULLISH_BREAKER",
                    low=ob.low,
                    high=ob.high,
                    index=failed_at,
                    source="BREAKER",
                    strength=clamp(ob.strength - 5),
                    details={"from": ob.kind, "failed_at": failed_at},
                )
            )
    return result


# ============================================================================
# PREMIUM / DISCOUNT / LOCATION
# ============================================================================


def dealing_range(candles: list[Candle], structure: StructureSnapshot) -> dict[str, Any]:
    # Prefer the latest meaningful swing range. Fall back to recent range.
    highs = structure.swing_highs[-6:]
    lows = structure.swing_lows[-6:]

    high = max([p.price for p in highs], default=max(c.high for c in candles[-48:]))
    low = min([p.price for p in lows], default=min(c.low for c in candles[-48:]))
    if high <= low:
        recent = candles[-48:]
        high = max(c.high for c in recent)
        low = min(c.low for c in recent)

    mid = (high + low) / 2.0
    return {
        "high": high,
        "low": low,
        "mid": mid,
        "range": max(high - low, EPS),
    }


def location_ratio(price: float, dr: dict[str, Any]) -> float:
    return clamp((price - dr["low"]) / max(dr["range"], EPS), 0.0, 1.0)


def location_score(direction: str, entry: float, dr: dict[str, Any]) -> float:
    ratio = location_ratio(entry, dr)
    if direction == "BUY":
        if ratio <= 0.25:
            return 100.0
        if ratio <= 0.35:
            return 92.0
        if ratio <= 0.50:
            return 78.0
        if ratio <= 0.65:
            return 55.0
        return 25.0
    if ratio >= 0.75:
        return 100.0
    if ratio >= 0.65:
        return 92.0
    if ratio >= 0.50:
        return 78.0
    if ratio >= 0.35:
        return 55.0
    return 25.0


# ============================================================================
# NOTE INTERPRETATION
# ============================================================================


def analyze_notes(notes: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(n.get("text") or "").strip() for n in notes if isinstance(n, dict)]
    combined = " | ".join(texts).lower()

    applied: list[dict[str, Any]] = []

    if "btc" in combined and "h4" in combined:
        applied.append({
            "rule": "BTC_H4_BIAS",
            "status": "APPLIED",
            "detail": "BTCUSDT H4 digunakan sebagai macro bias untuk altcoin.",
        })

    if "multi-timeframe" in combined or "multi timeframe" in combined or "mtf" in combined:
        applied.append({
            "rule": "MULTI_TIMEFRAME",
            "status": "APPLIED",
            "detail": "Analisis H4 → H1 → M15 dijalankan.",
        })

    if "rsi" in combined and "14" in combined and "m15" in combined:
        applied.append({
            "rule": "RSI14_M15",
            "status": "APPLIED",
            "detail": "RSI periode 14 pada M15 menjadi konfirmasi momentum.",
        })

    if "price exp" in combined or "price_exp" in combined or "price exp" in combined:
        applied.append({
            "rule": "PRICE_EXP",
            "status": "APPLIED",
            "detail": "Price Exp dipakai sebagai batas relevansi; jika tersentuh sebelum entry, thesis setup lama dianggap expired.",
        })

    if "liquidity" in combined and ("percentage" in combined or "%" in combined or "persen" in combined):
        applied.append({
            "rule": "LIQUIDITY_DISTANCE",
            "status": "APPLIED",
            "detail": "Jarak liquidity pool dihitung dalam persen harga dan ATR.",
        })

    applied.append({
        "rule": "HTF_POI_HIERARCHY",
        "status": "APPLIED",
        "detail": "Primary POI H4 -> refinement H1 -> execution confirmation M15.",
    })
    applied.append({
        "rule": "FIBONACCI_CONFLUENCE",
        "status": "APPLIED",
        "detail": "Fibonacci retracement dihitung dari impuls/swing terkonfirmasi pada H4 dan H1; 0.618 menjadi filter confluence utama, bukan trigger tunggal.",
    })

    return {
        "raw_notes": texts,
        "applied_rules": applied,
        "count": len(texts),
    }


# ============================================================================
# SMC CONTEXT / CANDIDATE HELPERS
# ============================================================================


def find_matching_poi(
    zones: list[Zone],
    direction: str,
    current_price: float,
    atr: float,
) -> Zone | None:
    wanted = {
        "BUY": {"BULLISH_FVG", "BULLISH_OB", "BULLISH_BREAKER"},
        "SELL": {"BEARISH_FVG", "BEARISH_OB", "BEARISH_BREAKER"},
    }[direction]

    eligible: list[tuple[float, Zone]] = []
    for zone in zones:
        if zone.kind not in wanted:
            continue
        if zone.status == "FAILED":
            continue
        dist = abs(current_price - zone.midpoint) / max(atr, EPS)
        if dist > POI_MAX_DISTANCE_ATR:
            continue

        # Pending BUY should be below current. Pending SELL should be above current.
        if direction == "BUY" and zone.midpoint >= current_price:
            continue
        if direction == "SELL" and zone.midpoint <= current_price:
            continue
        eligible.append((dist - zone.strength / 500.0, zone))

    eligible.sort(key=lambda x: x[0])
    return eligible[0][1] if eligible else None


def find_target_liquidity(
    pools: list[LiquidityPool],
    direction: str,
    entry: float,
    current_price: float,
    atr: float,
) -> LiquidityPool | None:
    if direction == "BUY":
        candidates = [
            p for p in pools
            if p.kind == "BUY_SIDE" and p.level > entry + 0.75 * atr
        ]
        candidates.sort(key=lambda p: (max(0.0, p.level - current_price), -p.strength))
    else:
        candidates = [
            p for p in pools
            if p.kind == "SELL_SIDE" and p.level < entry - 0.75 * atr
        ]
        candidates.sort(key=lambda p: (max(0.0, current_price - p.level), -p.strength))

    for pool in candidates:
        if abs(pool.level - entry) / max(atr, EPS) <= 0.75:
            continue
        if abs(pool.level - entry) / max(atr, EPS) > 5.5:
            continue
        return pool
    return None


def find_recent_sweep_mss(
    sweeps: list[dict[str, Any]],
    structure: StructureSnapshot,
    direction: str,
    max_gap: int = 12,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for sweep in reversed(sweeps):
        if sweep["direction"] != direction:
            continue
        s_idx = int(sweep["index"])
        for event in reversed(structure.events):
            if event["type"] != "MSS" or event["direction"] != direction:
                continue
            e_idx = int(event["index"])
            if e_idx <= s_idx:
                break
            if e_idx - s_idx <= max_gap:
                return sweep, event
    return None, None


def candidate_from_poi(
    direction: str,
    model: str,
    poi: Zone,
    current: float,
    atr: float,
    target: LiquidityPool | None,
    evidence: dict[str, Any],
    rsi: float,
    vlt_direction: str,
    structure: StructureSnapshot,
    dr: dict[str, Any],
) -> Candidate | None:
    entry = poi.midpoint

    if direction == "BUY":
        sl = poi.low - atr * SL_BUFFER_ATR
        if sl >= entry:
            sl = entry - max(atr * 0.6, entry * 0.002)
        target_price = target.level if target else max(dr["high"], entry + atr * 2.0)
        tp = max(target_price, current + atr * 1.25, entry + atr * 1.25)
        # Keep Price Exp between current and the forward target when possible.
        price_exp = current + max(atr * EXP_ATR_MULTIPLIER, abs(current - entry) * 1.35)
        if tp > current:
            midpoint_to_tp = current + (tp - current) * 0.45
            price_exp = min(price_exp, midpoint_to_tp)
        price_exp = max(price_exp, current + min(atr * 0.50, max(tp - current, atr * 0.50) * 0.35))

        if not (sl < entry < current < price_exp < tp and tp > entry):
            return None

        entry_reason = _entry_reason_buy(model, poi, evidence, structure, rsi, vlt_direction)
        sl_reason = (
            f"SL di bawah low {poi.kind} ({round_price(poi.low)}) + buffer {SL_BUFFER_ATR:.2f} ATR "
            "sebagai batas invalidasi struktur."
        )
        tp_reason = _tp_reason("BUY", target, tp, current, atr)
        exp_reason = (
            f"Price Exp {round_price(price_exp)} menjadi batas ekspansi sebelum entry. "
            "Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari."
        )
    else:
        sl = poi.high + atr * SL_BUFFER_ATR
        if sl <= entry:
            sl = entry + max(atr * 0.6, entry * 0.002)
        target_price = target.level if target else min(dr["low"], entry - atr * 2.0)
        tp = min(target_price, current - atr * 1.25, entry - atr * 1.25)
        # Keep Price Exp between the forward target and current when possible.
        price_exp = current - max(atr * EXP_ATR_MULTIPLIER, abs(current - entry) * 1.35)
        if tp < current:
            midpoint_to_tp = current - (current - tp) * 0.45
            price_exp = max(price_exp, midpoint_to_tp)
        price_exp = min(price_exp, current - min(atr * 0.50, max(current - tp, atr * 0.50) * 0.35))

        if not (tp < price_exp < current < entry < sl and tp < entry):
            return None

        entry_reason = _entry_reason_sell(model, poi, evidence, structure, rsi, vlt_direction)
        sl_reason = (
            f"SL di atas high {poi.kind} ({round_price(poi.high)}) + buffer {SL_BUFFER_ATR:.2f} ATR "
            "sebagai batas invalidasi struktur."
        )
        tp_reason = _tp_reason("SELL", target, tp, current, atr)
        exp_reason = (
            f"Price Exp {round_price(price_exp)} menjadi batas ekspansi sebelum entry. "
            "Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari."
        )

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / max(risk, EPS)

    evidence = dict(evidence)
    evidence.update(
        {
            "poi": {
                "kind": poi.kind,
                "low": round_price(poi.low),
                "high": round_price(poi.high),
                "status": poi.status,
                "strength": round(poi.strength, 2),
                "age_candles": max(0, len(evidence.get("_candles", [])) - 1 - poi.index) if evidence.get("_candles") else None,
            },
            "planned_rr": round(rr, 3),
        }
    )
    evidence.pop("_candles", None)

    return Candidate(
        direction=direction,
        model=model,
        entry=round_price(entry),
        sl=round_price(sl),
        tp=round_price(tp),
        price_exp=round_price(price_exp),
        entry_reason=entry_reason,
        sl_reason=sl_reason,
        tp_reason=tp_reason,
        price_exp_reason=exp_reason,
        evidence=evidence,
    )


def _entry_reason_buy(model: str, poi: Zone, evidence: dict[str, Any], structure: StructureSnapshot, rsi: float, vlt_direction: str) -> str:
    fragments = [
        f"Model {model}: area {poi.kind} di {round_price(poi.midpoint)}.",
    ]
    if evidence.get("sweep"):
        fragments.append("Ada sell-side liquidity sweep sebelum perubahan struktur.")
    if evidence.get("mss"):
        fragments.append("M15 membentuk MSS bullish dengan displacement.")
    elif evidence.get("bos"):
        fragments.append("M15 membentuk bullish BOS.")
    if structure.trend == "BULLISH":
        fragments.append("Struktur pasangan mendukung bullish.")
    if rsi >= 50:
        fragments.append(f"RSI 14 M15 berada di {rsi:.1f}, mendukung momentum bullish.")
    if vlt_direction == "BULLISH":
        fragments.append("Volume/price pressure OHLCV mendukung bullish.")
    return " ".join(fragments)


def _entry_reason_sell(model: str, poi: Zone, evidence: dict[str, Any], structure: StructureSnapshot, rsi: float, vlt_direction: str) -> str:
    fragments = [
        f"Model {model}: area {poi.kind} di {round_price(poi.midpoint)}.",
    ]
    if evidence.get("sweep"):
        fragments.append("Ada buy-side liquidity sweep sebelum perubahan struktur.")
    if evidence.get("mss"):
        fragments.append("M15 membentuk MSS bearish dengan displacement.")
    elif evidence.get("bos"):
        fragments.append("M15 membentuk bearish BOS.")
    if structure.trend == "BEARISH":
        fragments.append("Struktur pasangan mendukung bearish.")
    if rsi <= 50:
        fragments.append(f"RSI 14 M15 berada di {rsi:.1f}, mendukung momentum bearish.")
    if vlt_direction == "BEARISH":
        fragments.append("Volume/price pressure OHLCV mendukung bearish.")
    return " ".join(fragments)


def _tp_reason(direction: str, target: LiquidityPool | None, tp: float, current: float, atr: float) -> str:
    if target:
        return (
            f"TP diarahkan ke {target.source} {round_price(target.level)} "
            f"({target.distance_atr:.2f} ATR dari current) sebagai target liquidity."
        )
    return f"Tidak ada liquidity target yang cukup dekat; TP fallback {round_price(tp)} berdasarkan ATR / range."


def candidate_rr_score(candidate: Candidate) -> float:
    risk = abs(candidate.entry - candidate.sl)
    reward = abs(candidate.tp - candidate.entry)
    rr = reward / max(risk, EPS)
    if rr >= 3:
        return 100.0
    if rr >= 2.5:
        return 94.0
    if rr >= 2:
        return 86.0
    if rr >= 1.5:
        return 72.0
    if rr >= 1.0:
        return 48.0
    return 20.0


def macro_bias_score(direction: str, macro_trend: str) -> float:
    if macro_trend == "BULLISH":
        return 100.0 if direction == "BUY" else 25.0
    if macro_trend == "BEARISH":
        return 100.0 if direction == "SELL" else 25.0
    return 58.0


def structure_direction_score(direction: str, trend: str) -> float:
    if trend == "BULLISH":
        return 100.0 if direction == "BUY" else 25.0
    if trend == "BEARISH":
        return 100.0 if direction == "SELL" else 25.0
    return 58.0


def rsi_score(direction: str, rsi: float, slope: float) -> float:
    if direction == "BUY":
        score = 55.0
        if 50 <= rsi <= 68:
            score = 88.0
        elif 45 <= rsi < 50:
            score = 72.0
        elif 68 < rsi <= 75:
            score = 70.0
        elif rsi < 40:
            score = 38.0
        elif rsi > 78:
            score = 45.0
        if slope > 0:
            score += 8
        elif slope < 0:
            score -= 8
    else:
        score = 55.0
        if 32 <= rsi <= 50:
            score = 88.0
        elif 50 < rsi <= 55:
            score = 72.0
        elif 25 <= rsi < 32:
            score = 70.0
        elif rsi > 60:
            score = 38.0
        elif rsi < 22:
            score = 45.0
        if slope < 0:
            score += 8
        elif slope > 0:
            score -= 8
    return clamp(score)


def displacement_score_for_candidate(candidate: Candidate, latest_disp: float, vlt_rv: float) -> float:
    base = latest_disp
    if candidate.evidence.get("mss"):
        base += 8
    if vlt_rv >= 1.2:
        base += 5
    return clamp(base)


def liquidity_score(candidate: Candidate, target: LiquidityPool | None, sweep: dict[str, Any] | None, current: float, atr: float) -> float:
    score = 50.0
    if target:
        score += min(30.0, target.strength * 0.30)
        dist = abs(target.level - current) / max(atr, EPS)
        if 1.0 <= dist <= 4.0:
            score += 15
        elif dist <= 6:
            score += 7
    if sweep:
        score += 20
    return clamp(score)


def smc_trigger_score(candidate: Candidate) -> float:
    score = 50.0
    if candidate.evidence.get("mss"):
        score += 30
    elif candidate.evidence.get("bos"):
        score += 22
    if candidate.evidence.get("sweep"):
        score += 12

    poi_kind = str(candidate.evidence.get("poi", {}).get("kind", ""))
    topdown = candidate.evidence.get("top_down") or {}
    execution = topdown.get("execution_m15") or {}
    execution_kind = str(execution.get("kind") or "")
    combined_kinds = f"{poi_kind} {execution_kind}"
    if "FVG" in combined_kinds:
        score += 8
    if "OB" in combined_kinds:
        score += 7
    if "BREAKER" in combined_kinds:
        score += 4

    if candidate.evidence.get("btc_direction_constraint"):
        score += 4
    if candidate.evidence.get("trigger_status") == "M15_CONFIRMATION_AVAILABLE":
        score += 5
    return clamp(score)


def score_candidate(
    candidate: Candidate,
    *,
    macro_trend: str,
    pair_h4_trend: str,
    h1_dr: dict[str, Any],
    target: LiquidityPool | None,
    sweep: dict[str, Any] | None,
    current: float,
    m15_atr: float,
    m15_rsi: float,
    m15_rsi_slope: float,
    vlt: dict[str, Any],
    latest_displacement: float,
    h4_poi_score: float | None = None,
    h4_fib_score: float | None = None,
    h1_refinement_score: float | None = None,
) -> Candidate:
    scores = {
        "btc_h4_bias": macro_bias_score(candidate.direction, macro_trend),
        "pair_h4_structure": structure_direction_score(candidate.direction, pair_h4_trend),
        "h4_poi": clamp(h4_poi_score if h4_poi_score is not None else 50.0),
        "h4_fibonacci": clamp(h4_fib_score if h4_fib_score is not None else 50.0),
        "h1_refinement": clamp(h1_refinement_score if h1_refinement_score is not None else location_score(candidate.direction, candidate.entry, h1_dr)),
        "liquidity": liquidity_score(candidate, target, sweep, current, m15_atr),
        "smc_trigger": smc_trigger_score(candidate),
        "displacement": displacement_score_for_candidate(candidate, latest_displacement, vlt["relative_volume"]),
        "rsi_m15": rsi_score(candidate.direction, m15_rsi, m15_rsi_slope),
        "vlt": _vlt_direction_score(candidate.direction, vlt),
        "planned_rr": candidate_rr_score(candidate),
    }

    confidence = sum(
        WEIGHTS[name] * scores[name] / 100.0
        for name in WEIGHTS
    )

    candidate.scores = {k: round(v, 2) for k, v in scores.items()}
    candidate.confidence = round(clamp(confidence), 2)
    return candidate


def _vlt_direction_score(direction: str, vlt: dict[str, Any]) -> float:
    trend = str(vlt.get("direction") or "NEUTRAL")
    base = safe_float(vlt.get("score"), 50.0)
    # The VLT module's score is interpreted in the direction reported by the
    # module. A bearish module with score 80 should therefore give SELL=80,
    # while a bullish module with score 80 should give BUY=80.
    if direction == "BUY":
        if trend == "BULLISH":
            return clamp(base)
        if trend == "BEARISH":
            return clamp(100.0 - base)
        return 50.0
    if trend == "BEARISH":
        return clamp(base)
    if trend == "BULLISH":
        return clamp(100.0 - base)
    return 50.0


# ============================================================================
# FALLBACK CANDIDATE
# ============================================================================


def fallback_candidate(
    direction: str,
    current: float,
    atr: float,
    dr: dict[str, Any],
    structure: StructureSnapshot,
    target_pools: list[LiquidityPool],
) -> Candidate:
    if atr <= 0:
        atr = max(current * 0.005, 1e-8)

    if direction == "BUY":
        entry = current - 0.55 * atr
        sl = entry - 1.0 * atr
        target = find_target_liquidity(target_pools, "BUY", entry, current, atr)
        tp = target.level if target and target.level > current else max(dr["high"], current + 1.8 * atr)
        tp = max(tp, current + 1.25 * atr, entry + 1.8 * atr)
        price_exp = current + max(EXP_ATR_MULTIPLIER * atr, (tp - current) * 0.45)
        price_exp = min(price_exp, current + (tp - current) * 0.65) if tp > current else current + EXP_ATR_MULTIPLIER * atr
        return Candidate(
            direction="BUY",
            model="STRUCTURE_PULLBACK_FALLBACK",
            entry=round_price(entry),
            sl=round_price(min(sl, entry - EPS)),
            tp=round_price(max(tp, entry + atr)),
            price_exp=round_price(max(price_exp, current + EPS)),
            entry_reason=(
                "Fallback: tidak ditemukan POI SMC yang cukup dekat untuk model utama. "
                f"Entry pullback {round_price(entry)} mengikuti bias {structure.trend}."
            ),
            sl_reason="SL fallback 1 ATR di bawah entry sebagai structural/volatility invalidation.",
            tp_reason=(
                f"TP menuju liquidity {round_price(tp)} / range high terdekat."
            ),
            price_exp_reason=(
                f"Setup BUY expired bila harga mencapai {round_price(price_exp)} sebelum retrace ke entry. "
                "Sesudah itu market perlu membentuk pola baru."
            ),
            evidence={"fallback": True, "structure": structure.trend},
        )

    entry = current + 0.55 * atr
    sl = entry + 1.0 * atr
    target = find_target_liquidity(target_pools, "SELL", entry, current, atr)
    tp = target.level if target and target.level < current else min(dr["low"], current - 1.8 * atr)
    tp = min(tp, current - 1.25 * atr, entry - 1.8 * atr)
    price_exp = current - max(EXP_ATR_MULTIPLIER * atr, (current - tp) * 0.45)
    price_exp = max(price_exp, current - (current - tp) * 0.65) if tp < current else current - EXP_ATR_MULTIPLIER * atr
    return Candidate(
        direction="SELL",
        model="STRUCTURE_PULLBACK_FALLBACK",
        entry=round_price(entry),
        sl=round_price(max(sl, entry + EPS)),
        tp=round_price(min(tp, entry - atr)),
        price_exp=round_price(min(price_exp, current - EPS)),
        entry_reason=(
            "Fallback: tidak ditemukan POI SMC yang cukup dekat untuk model utama. "
            f"Entry pullback {round_price(entry)} mengikuti bias {structure.trend}."
        ),
        sl_reason="SL fallback 1 ATR di atas entry sebagai structural/volatility invalidation.",
        tp_reason=(
            f"TP menuju liquidity {round_price(tp)} / range low terdekat."
        ),
        price_exp_reason=(
            f"Setup SELL expired bila harga mencapai {round_price(price_exp)} sebelum retrace ke entry. "
            "Sesudah itu market perlu membentuk pola baru."
        ),
        evidence={"fallback": True, "structure": structure.trend},
    )


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================


def _latest_rsi_context(candles: list[Candle]) -> dict[str, Any]:
    values = rsi_series(candles, 14)
    if not values:
        return {"rsi14": 50.0, "slope": 0.0}
    slope = linear_slope(values[-8:], min(8, len(values)))
    return {
        "rsi14": round(values[-1], 3),
        "slope": round(slope, 4),
        "previous": round(values[-2], 3) if len(values) >= 2 else round(values[-1], 3),
    }


def _structure_summary(structure: StructureSnapshot, candles: list[Candle]) -> dict[str, Any]:
    last_labels = structure.labels[-8:]
    last_events = structure.events[-8:]
    return {
        "trend": structure.trend,
        "last_pivot_labels": last_labels,
        "recent_events": last_events,
        "last_bos": structure.last_bos,
        "last_mss": structure.last_mss,
        "protected_high": round_price(structure.protected_high) if structure.protected_high else None,
        "protected_low": round_price(structure.protected_low) if structure.protected_low else None,
        "swing_high_count": len(structure.swing_highs),
        "swing_low_count": len(structure.swing_lows),
        "bars": len(candles),
    }


def _select_macro_trend(
    btc_h4: StructureSnapshot,
    target_h4: StructureSnapshot,
    pair: str,
) -> str:
    """Return the higher-timeframe directional source used by the strategy.

    For altcoins, BTCUSDT H4 is the primary directional source. For BTCUSDT
    itself, the target's own H4 structure is used so BTC is not self-compared.
    """
    if pair == "BTCUSDT":
        return target_h4.trend
    return btc_h4.trend


def allowed_directions_for_macro(
    macro_trend: str,
    pair_h4_trend: str,
    pair: str,
) -> tuple[str, ...]:
    """Define the strategy search-space from the macro directional regime.

    Core rule:
        Altcoin + BTC H4 bullish -> BUY only.
        Altcoin + BTC H4 bearish -> SELL only.
        Altcoin + BTC H4 range   -> both directions allowed.

    For BTCUSDT itself, target H4 is the macro source.

    Pair H4 never reverses the macro direction. It only affects confidence.
    """
    trend = pair_h4_trend if pair == "BTCUSDT" else macro_trend
    if trend == "BULLISH":
        return (DIRECTION_BUY,)
    if trend == "BEARISH":
        return (DIRECTION_SELL,)
    return DIRECTIONS_BOTH


def directional_regime_label(
    allowed_directions: tuple[str, ...],
) -> str:
    if allowed_directions == (DIRECTION_BUY,):
        return "BULLISH_ONLY"
    if allowed_directions == (DIRECTION_SELL,):
        return "BEARISH_ONLY"
    return "NEUTRAL_BOTH"


def _collect_directional_candidates(
    *,
    pair: str,
    current: float,
    m15: list[Candle],
    h1: list[Candle],
    h4: list[Candle],
    btc_h4: list[Candle],
    m15_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
    h4_structure: StructureSnapshot,
    btc_structure: StructureSnapshot,
    liquidity: list[LiquidityPool],
    sweeps: list[dict[str, Any]],
    fvg: list[Zone],
    obs: list[Zone],
    breakers: list[Zone],
    m15_atr_values: list[float],
    h4_ctx: dict[str, Any] | None = None,
    h1_ctx: dict[str, Any] | None = None,
    m15_ctx: dict[str, Any] | None = None,
    topdown_liquidity: list[LiquidityPool] | None = None,
) -> list[Candidate]:
    atr = m15_atr_values[-1]
    rsi_ctx = _latest_rsi_context(m15)
    rsi = rsi_ctx["rsi14"]
    vlt = volume_trend_score(m15, m15_atr_values)
    m15_rel_vol = relative_volume(m15, 20)
    latest_disp = displacement_strength(
        m15,
        len(m15) - 1,
        m15_atr_values,
        m15_rel_vol,
    )
    h1_dr = dealing_range(h1, h1_structure)

    macro_trend = _select_macro_trend(btc_structure, h4_structure, pair)
    allowed_directions = allowed_directions_for_macro(
        macro_trend,
        h4_structure.trend,
        pair,
    )
    candidates: list[Candidate] = []

    h4_ctx = h4_ctx or htf_poi_candidates(h4, h4_structure, DIRECTION_BUY, "H4")
    h1_ctx = h1_ctx or htf_poi_candidates(h1, h1_structure, DIRECTION_BUY, "H1")
    m15_ctx = m15_ctx or htf_poi_candidates(m15, m15_structure, DIRECTION_BUY, "M15")
    combined_liquidity = topdown_liquidity or [*h4_ctx.get("liquidity", []), *h1_ctx.get("liquidity", []), *m15_ctx.get("liquidity", [])]

    for direction in allowed_directions:
        primary, primary_details = select_primary_poi(h4_ctx, direction, current)
        refinement, refinement_details = select_refinement_poi(h1_ctx, direction, primary, current)
        parent = refinement or primary
        execution, execution_details = select_execution_poi(m15_ctx, direction, parent, current)

        sweep, mss = find_recent_sweep_mss(sweeps, m15_structure, direction)
        recent_bos = latest_event(
            m15_structure,
            {"BOS", "MSS"},
            direction=direction,
            max_age=40,
            candle_count=len(m15),
        )

        # Primary model: HTF POI -> H1 refinement -> M15 confirmation.
        if primary is not None:
            target = find_target_liquidity_topdown(
                combined_liquidity,
                direction,
                execution.midpoint if execution else refinement.midpoint if refinement else primary.midpoint,
                atr,
            )
            cand = topdown_candidate(
                pair=pair,
                direction=direction,
                current=current,
                m15_atr=atr,
                primary=primary,
                refinement=refinement,
                execution=execution,
                h4_ctx=h4_ctx,
                h1_ctx=h1_ctx,
                m15_ctx=m15_ctx,
                m15_structure=m15_structure,
                target=target,
                sweep=sweep,
                mss=mss,
                bos=recent_bos,
                rsi=rsi,
                vlt_direction=vlt["direction"],
            )
            if cand is not None:
                cand.evidence["poi_selection"] = {
                    "h4": primary_details,
                    "h1": refinement_details,
                    "m15": execution_details,
                }
                # Dedicated top-down scoring.
                h4_fib = h4_ctx.get("fib", {}).get(direction)
                h1_fib = h1_ctx.get("fib", {}).get(direction)
                h4_poi_score = safe_float(primary_details.get("quality"), 50.0)
                h4_fib_score = fib_confluence_score(primary, h4_fib, direction)[0] if primary else 50.0
                h1_ref_score = safe_float(refinement_details.get("quality"), 50.0) if refinement else 35.0
                trigger_event = mss or recent_bos
                trigger_disp = safe_float(trigger_event.get("displacement"), latest_disp) if trigger_event else latest_disp
                score_candidate(
                    cand,
                    macro_trend=macro_trend,
                    pair_h4_trend=h4_structure.trend,
                    h1_dr=h1_dr,
                    target=target,
                    sweep=sweep,
                    current=current,
                    m15_atr=atr,
                    m15_rsi=rsi,
                    m15_rsi_slope=rsi_ctx["slope"],
                    vlt=vlt,
                    latest_displacement=trigger_disp,
                    h4_poi_score=h4_poi_score,
                    h4_fib_score=h4_fib_score,
                    h1_refinement_score=h1_ref_score,
                )
                # Trigger quality is materially lower when no M15 structural
                # confirmation exists, but we do not hard-gate the setup.
                if not (sweep or mss or recent_bos):
                    cand.confidence = round(max(0.0, cand.confidence - 18.0), 2)
                    cand.evidence["trigger_status"] = "WAITING_FOR_M15_CONFIRMATION"
                else:
                    cand.evidence["trigger_status"] = "M15_CONFIRMATION_AVAILABLE"
                candidates.append(cand)

        # Secondary legacy-style candidate is kept only as a resilience path;
        # it remains inside the BTC-defined direction and is scored lower than
        # a top-down candidate when both exist.
        if not candidates or not primary:
            recent_fvg = [
                z for z in fvg
                if len(m15) - 1 - z.index <= FVG_MAX_AGE_M15
            ]
            matching_zones = [*recent_fvg, *obs, *breakers]
            poi = find_matching_poi(matching_zones, direction, current, atr)
            if poi:
                target = find_target_liquidity(liquidity, direction, poi.midpoint, current, atr)
                cand = candidate_from_poi(
                    direction,
                    "M15_SMC_FALLBACK",
                    poi,
                    current,
                    atr,
                    target,
                    {
                        "poi": poi.kind,
                        "sweep": sweep,
                        "mss": mss,
                        "bos": recent_bos,
                        "macro_trend": macro_trend,
                        "allowed_directions": list(allowed_directions),
                        "top_down_primary_available": primary is not None,
                    },
                    rsi,
                    vlt["direction"],
                    h4_structure,
                    h1_dr,
                )
                if cand:
                    score_candidate(
                        cand,
                        macro_trend=macro_trend,
                        pair_h4_trend=h4_structure.trend,
                        h1_dr=h1_dr,
                        target=target,
                        sweep=sweep,
                        current=current,
                        m15_atr=atr,
                        m15_rsi=rsi,
                        m15_rsi_slope=rsi_ctx["slope"],
                        vlt=vlt,
                        latest_displacement=latest_disp,
                        h4_poi_score=35.0,
                        h4_fib_score=35.0,
                        h1_refinement_score=location_score(direction, cand.entry, h1_dr),
                    )
                    cand.confidence = round(max(0.0, cand.confidence - 10.0), 2)
                    candidates.append(cand)

        # Preserve secondary breaker retest as an explicit model.
        breaker = find_matching_poi(breakers, direction, current, atr)
        if breaker and primary is None:
            target = find_target_liquidity(liquidity, direction, breaker.midpoint, current, atr)
            cand = candidate_from_poi(
                direction,
                "BREAKER_RETEST",
                breaker,
                current,
                atr,
                target,
                {
                    "poi": breaker.kind,
                    "breaker": True,
                    "macro_trend": macro_trend,
                    "allowed_directions": list(allowed_directions),
                },
                rsi,
                vlt["direction"],
                h4_structure,
                h1_dr,
            )
            if cand:
                score_candidate(
                    cand,
                    macro_trend=macro_trend,
                    pair_h4_trend=h4_structure.trend,
                    h1_dr=h1_dr,
                    target=target,
                    sweep=sweep,
                    current=current,
                    m15_atr=atr,
                    m15_rsi=rsi,
                    m15_rsi_slope=rsi_ctx["slope"],
                    vlt=vlt,
                    latest_displacement=latest_disp,
                    h4_poi_score=30.0,
                    h4_fib_score=30.0,
                    h1_refinement_score=location_score(direction, cand.entry, h1_dr),
                )
                candidates.append(cand)

    # Deterministic fallback remains, but it can now carry the HTF context.
    if not candidates:
        if len(allowed_directions) == 1:
            direction = allowed_directions[0]
        else:
            direction = DIRECTION_BUY if h4_structure.trend != "BEARISH" else DIRECTION_SELL
        fallback = fallback_candidate(
            direction,
            current,
            atr,
            h1_dr,
            h4_structure,
            combined_liquidity,
        )
        fallback.evidence.update({
            "macro_trend": macro_trend,
            "allowed_directions": list(allowed_directions),
            "directional_regime": directional_regime_label(allowed_directions),
            "top_down": {
                "h4_primary_poi": False,
                "h1_refinement": False,
                "m15_execution": False,
            },
        })
        candidates.append(fallback)

    # Defensive invariant: no candidate may escape the macro-defined search
    # space.
    candidates = [c for c in candidates if c.direction in allowed_directions]
    if not candidates:
        raise RuntimeError(
            "Directional strategy invariant failed: no candidate remains "
            f"for allowed directions {list(allowed_directions)}."
        )

    return candidates


def _best_candidate(candidates: list[Candidate]) -> Candidate:
    # Main criterion is confidence. Ties prefer explicit SMC trigger over fallback.
    model_priority = {
        "LIQUIDITY_SWEEP_MSS": 4,
        "BOS_PULLBACK": 3,
        "BREAKER_RETEST": 2,
        "STRUCTURE_PULLBACK_FALLBACK": 1,
    }
    return sorted(
        candidates,
        key=lambda c: (c.confidence, model_priority.get(c.model, 0)),
        reverse=True,
    )[0]


# ============================================================================
# OUTPUT SERIALIZATION
# ============================================================================


def _zone_summary(zones: list[Zone], current: float, atr: float, total_candles: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for z in zones[:12]:
        row = {
            "kind": z.kind,
            "timeframe": z.details.get("timeframe"),
            "low": round_price(z.low),
            "high": round_price(z.high),
            "midpoint": round_price(z.midpoint),
            "status": z.status,
            "strength": round(z.strength, 2),
            "distance_pct": round(distance_pct(current, z.midpoint), 4),
            "distance_atr": round(abs(z.midpoint - current) / max(atr, EPS), 3),
        }
        if total_candles is not None:
            row["age_candles"] = max(0, total_candles - 1 - z.index)
        if isinstance(z.details.get("quality"), dict):
            row["quality_details"] = z.details.get("quality")
        rows.append(row)
    return rows


def _liquidity_summary(pools: list[LiquidityPool], current: float, atr: float) -> list[dict[str, Any]]:
    rows = []
    for p in pools[:20]:
        rows.append(
            {
                "kind": p.kind,
                "level": round_price(p.level),
                "strength": round(p.strength, 2),
                "source": p.source,
                "swept": p.swept,
                "distance_pct": round(distance_pct(current, p.level), 4),
                "distance_atr": round(abs(p.level - current) / max(atr, EPS), 3),
            }
        )
    return rows


def _candidate_summary(candidate: Candidate) -> dict[str, Any]:
    return {
        "model": candidate.model,
        "direction": candidate.direction,
        "entry": candidate.entry,
        "sl": candidate.sl,
        "tp": candidate.tp,
        "price_exp": candidate.price_exp,
        "confidence": candidate.confidence,
        "scores": candidate.scores,
    }


def _ensure_price_geometry(candidate: Candidate, current: float) -> bool:
    if candidate.direction == "BUY":
        return candidate.sl < candidate.entry < current < candidate.price_exp and candidate.tp > candidate.entry
    return candidate.price_exp < current < candidate.entry < candidate.sl and candidate.tp < candidate.entry


def _round_tick(value: float, tick_size: float, direction: str) -> float:
    """Align to Binance tick when available; otherwise leave analytical value."""
    if tick_size <= 0:
        return value
    steps = value / tick_size
    if direction == "DOWN":
        steps = math.floor(steps + 1e-12)
    elif direction == "UP":
        steps = math.ceil(steps - 1e-12)
    else:
        steps = round(steps)
    return steps * tick_size


def align_candidate_to_tick(candidate: Candidate, tick_size: float, current: float) -> Candidate:
    if tick_size <= 0:
        return candidate

    # Direction-aware rounding maintains safety geometry as much as possible.
    if candidate.direction == "BUY":
        candidate.entry = round_price(_round_tick(candidate.entry, tick_size, "DOWN"))
        candidate.sl = round_price(_round_tick(candidate.sl, tick_size, "DOWN"))
        candidate.tp = round_price(_round_tick(candidate.tp, tick_size, "UP"))
        candidate.price_exp = round_price(_round_tick(candidate.price_exp, tick_size, "UP"))

        # Fix tiny boundary reversals after quantization.
        if candidate.entry >= current:
            candidate.entry = round_price(_round_tick(current - tick_size, tick_size, "DOWN"))
        if candidate.sl >= candidate.entry:
            candidate.sl = round_price(_round_tick(candidate.entry - tick_size, tick_size, "DOWN"))
        if candidate.price_exp <= current:
            candidate.price_exp = round_price(_round_tick(current + tick_size, tick_size, "UP"))
        if candidate.tp <= current:
            candidate.tp = round_price(_round_tick(max(candidate.entry + tick_size, current + tick_size), tick_size, "UP"))
        if candidate.price_exp >= candidate.tp:
            candidate.price_exp = round_price(_round_tick(current + max(tick_size, (candidate.tp - current) * 0.45), tick_size, "UP"))
    else:
        candidate.entry = round_price(_round_tick(candidate.entry, tick_size, "UP"))
        candidate.sl = round_price(_round_tick(candidate.sl, tick_size, "UP"))
        candidate.tp = round_price(_round_tick(candidate.tp, tick_size, "DOWN"))
        candidate.price_exp = round_price(_round_tick(candidate.price_exp, tick_size, "DOWN"))

        if candidate.entry <= current:
            candidate.entry = round_price(_round_tick(current + tick_size, tick_size, "UP"))
        if candidate.sl <= candidate.entry:
            candidate.sl = round_price(_round_tick(candidate.entry + tick_size, tick_size, "UP"))
        if candidate.price_exp >= current:
            candidate.price_exp = round_price(_round_tick(current - tick_size, tick_size, "DOWN"))
        if candidate.tp >= current:
            candidate.tp = round_price(_round_tick(min(candidate.entry - tick_size, current - tick_size), tick_size, "DOWN"))
        if candidate.price_exp <= candidate.tp:
            candidate.price_exp = round_price(_round_tick(current - max(tick_size, (current - candidate.tp) * 0.45), tick_size, "DOWN"))

    return candidate


# ============================================================================
# PUBLIC CONTRACT
# ============================================================================

async def generate_setup(pair: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Analyze one symbol and return exactly the setup contract main.py expects.

    context fields used:
        notes, timeframe, candles_requested, market, session_id
    """
    context = context or {}
    normalized_pair = normalize_pair(pair)

    if not normalized_pair.endswith("USDT"):
        raise ValueError("strategy.py hanya mendukung USDT perpetual.")

    notes = context.get("notes") or []
    notes_context = analyze_notes(notes if isinstance(notes, list) else [])

    # ------------------------------------------------------------------
    # 1) Target M15: 672 CLOSED candles exactly.
    # ------------------------------------------------------------------
    m15, m15_source = await fetch_series(
        normalized_pair,
        M15,
        M15_CANDLES_REQUIRED,
        M15_MS,
    )

    # ------------------------------------------------------------------
    # 2) Higher timeframes.
    #    H1 and target H4 are fetched independently when possible so the
    #    H4 structure is not limited to only 42 bars from the 672 M15 window.
    #    We still resample H1/H4 from M15 for an explicitly comparable view.
    # ------------------------------------------------------------------
    pair_h4, pair_h4_source = await fetch_series(
        normalized_pair,
        H4,
        PAIR_H4_CANDLES_REQUIRED,
        H4_MS,
    )
    btc_h4, btc_h4_source = await fetch_series(
        "BTCUSDT",
        H4,
        BTC_H4_CANDLES_REQUIRED,
        H4_MS,
    )

    # H1 derived from the same M15 dataset, preserving the 672-candle scope.
    h1_derived = resample_candles(m15, H1_MS)
    h4_derived = resample_candles(m15, H4_MS)
    h1 = h1_derived
    if len(h1) < 20:
        # Should never happen with 672 closed M15 candles, but keep a safe fallback.
        h1 = await _fetch_if_short(normalized_pair, H1, H1_MS, 168)

    # ------------------------------------------------------------------
    # 3) Current price. Prefer the same provider as target M15 data.
    # ------------------------------------------------------------------
    current, price_source = await fetch_price(normalized_pair, m15_source)
    if current <= 0:
        raise RuntimeError("Current price tidak valid.")

    tick_size = await get_tick_size(normalized_pair)

    # ------------------------------------------------------------------
    # 4) Build all technical contexts.
    # ------------------------------------------------------------------
    m15_atr_values = atr_series(m15, 14)
    m15_rsi_ctx = _latest_rsi_context(m15)
    m15_rel_volume = relative_volume(m15, 20)
    m15_vlt = volume_trend_score(m15, m15_atr_values)

    m15_structure = build_structure(m15, SWING_SPAN_M15)
    h1_structure = build_structure(h1, SWING_SPAN_H1)
    pair_h4_structure = build_structure(pair_h4, SWING_SPAN_H4)
    btc_h4_structure = build_structure(btc_h4, SWING_SPAN_H4)

    # Full top-down POI contexts. H4 is the primary analytical layer, H1
    # refines it, and M15 is the execution layer.
    h4_ctx = htf_poi_candidates(pair_h4, pair_h4_structure, DIRECTION_BUY, "H4")
    h1_ctx = htf_poi_candidates(h1, h1_structure, DIRECTION_BUY, "H1")
    m15_ctx = htf_poi_candidates(m15, m15_structure, DIRECTION_BUY, "M15")

    liquidity = list(m15_ctx.get("liquidity", []))
    sweeps = list(m15_ctx.get("sweeps", []))
    fvg = list(m15_ctx.get("fvg", []))
    obs = list(m15_ctx.get("obs", []))
    breakers = list(m15_ctx.get("breakers", []))

    # Tag each liquidity pool with its timeframe so target selection can prefer
    # external HTF liquidity over micro/internal targets.
    for pool in h4_ctx.get("liquidity", []):
        pool.details["timeframe"] = "H4"
        pool.source = f"H4_{pool.source}" if not pool.source.startswith("H4_") else pool.source
    for pool in h1_ctx.get("liquidity", []):
        pool.details["timeframe"] = "H1"
        pool.source = f"H1_{pool.source}" if not pool.source.startswith("H1_") else pool.source
    for pool in m15_ctx.get("liquidity", []):
        pool.details["timeframe"] = "M15"
        pool.source = f"M15_{pool.source}" if not pool.source.startswith("M15_") else pool.source

    # Add H1/H4 liquidity context as broader external references.
    for structure, source in (
        (h1_structure, "H1_SWING"),
        (pair_h4_structure, "H4_SWING"),
    ):
        for p in structure.swing_highs[-8:]:
            liquidity.append(
                LiquidityPool(
                    kind="BUY_SIDE",
                    level=p.price,
                    index=p.index,
                    strength=68.0 if source == "H4_SWING" else 62.0,
                    source=source,
                    distance_pct=distance_pct(current, p.price),
                    distance_atr=abs(p.price - current) / max(m15_atr_values[-1], EPS),
                    details={"timeframe": source.split("_")[0]},
                )
            )
        for p in structure.swing_lows[-8:]:
            liquidity.append(
                LiquidityPool(
                    kind="SELL_SIDE",
                    level=p.price,
                    index=p.index,
                    strength=68.0 if source == "H4_SWING" else 62.0,
                    source=source,
                    distance_pct=distance_pct(current, p.price),
                    distance_atr=abs(p.price - current) / max(m15_atr_values[-1], EPS),
                    details={"timeframe": source.split("_")[0]},
                )
            )

    # Candidate generation is intentionally based on explicit SMC sequences
    # inside the BTC-defined directional search space, then fallback if there
    # is no complete sequence.
    candidates = _collect_directional_candidates(
        pair=normalized_pair,
        current=current,
        m15=m15,
        h1=h1,
        h4=pair_h4,
        btc_h4=btc_h4,
        m15_structure=m15_structure,
        h1_structure=h1_structure,
        h4_structure=pair_h4_structure,
        btc_structure=btc_h4_structure,
        liquidity=liquidity,
        sweeps=sweeps,
        fvg=fvg,
        obs=obs,
        breakers=breakers,
        m15_atr_values=m15_atr_values,
        h4_ctx=h4_ctx,
        h1_ctx=h1_ctx,
        m15_ctx=m15_ctx,
        topdown_liquidity=[*h4_ctx.get("liquidity", []), *h1_ctx.get("liquidity", []), *m15_ctx.get("liquidity", [])],
    )

    best = _best_candidate(candidates)

    # Quantize to actual Binance price tick when available, then re-check the
    # geometry that main.py will enforce.
    align_candidate_to_tick(best, tick_size, current)
    if not _ensure_price_geometry(best, current):
        # Rebuild a guaranteed-valid fallback around the exact live price.
        direction = best.direction
        h1_dr = dealing_range(h1, h1_structure)
        best = fallback_candidate(
            direction,
            current,
            m15_atr_values[-1],
            h1_dr,
            pair_h4_structure,
            liquidity,
        )
        align_candidate_to_tick(best, tick_size, current)

    if not _ensure_price_geometry(best, current):
        # Extremely defensive: use one-tick offsets when exchange tick is known.
        step = tick_size if tick_size > 0 else max(current * 1e-6, 1e-8)
        if best.direction == "BUY":
            best.entry = round_price(current - step)
            best.sl = round_price(max(step, best.entry - step))
            best.price_exp = round_price(current + step)
            best.tp = round_price(best.entry + step)
        else:
            best.entry = round_price(current + step)
            best.sl = round_price(best.entry + step)
            best.price_exp = round_price(max(step, current - step))
            best.tp = round_price(max(step, best.entry - step))

    # Recalculate confidence after the final level geometry is locked.
    final_macro = _select_macro_trend(btc_h4_structure, pair_h4_structure, normalized_pair)
    h1_dr = dealing_range(h1, h1_structure)
    final_target = None
    for pool in sorted(liquidity, key=lambda x: (x.strength, -abs(x.level - best.tp)), reverse=True):
        if best.direction == "BUY" and pool.kind == "BUY_SIDE" and pool.level > best.entry:
            final_target = pool
            break
        if best.direction == "SELL" and pool.kind == "SELL_SIDE" and pool.level < best.entry:
            final_target = pool
            break

    final_sweep = None
    sweep_direction = "BULLISH" if best.direction == "BUY" else "BEARISH"
    for sweep in reversed(sweeps):
        if sweep["direction"] == sweep_direction and len(m15) - 1 - int(sweep["index"]) <= 40:
            final_sweep = sweep
            break

    score_candidate(
        best,
        macro_trend=final_macro,
        pair_h4_trend=pair_h4_structure.trend,
        h1_dr=h1_dr,
        target=final_target,
        sweep=final_sweep,
        current=current,
        m15_atr=m15_atr_values[-1],
        m15_rsi=m15_rsi_ctx["rsi14"],
        m15_rsi_slope=m15_rsi_ctx["slope"],
        vlt=m15_vlt,
        latest_displacement=displacement_strength(
            m15,
            len(m15) - 1,
            m15_atr_values,
            m15_rel_volume,
        ),
    )

    risk = abs(best.entry - best.sl)
    reward = abs(best.tp - best.entry)
    planned_rr = reward / max(risk, EPS)

    # ------------------------------------------------------------------
    # Compact but rich machine-readable analysis.
    # ------------------------------------------------------------------
    analysis = {
        "macro": {
            "pair": normalized_pair,
            "btc_h4": _structure_summary(btc_h4_structure, btc_h4),
            "pair_h4": _structure_summary(pair_h4_structure, pair_h4),
            "macro_bias": final_macro,
            "directional_regime": directional_regime_label(
                allowed_directions_for_macro(
                    final_macro,
                    pair_h4_structure.trend,
                    normalized_pair,
                )
            ),
            "allowed_directions": list(
                allowed_directions_for_macro(
                    final_macro,
                    pair_h4_structure.trend,
                    normalized_pair,
                )
            ),
            "direction_rule": (
                "BTCUSDT H4 determines the search direction for altcoins; "
                "pair H4/H1/M15 can reduce confidence but cannot flip the direction."
                if is_altcoin(normalized_pair)
                else "BTCUSDT uses its own H4 structure as the directional source."
            ),
            "altcoin_rule_applied": is_altcoin(normalized_pair),
        },
        "multi_timeframe": {
            "h4": {
                **_structure_summary(pair_h4_structure, pair_h4),
                "trend_strength": trend_strength_metrics(pair_h4, pair_h4_structure),
                "fibonacci": fibonacci_summary(h4_ctx.get("fib", {}).get(best.direction), current),
                "poi": _zone_summary(h4_ctx.get("zones", [])[:10], current, h4_ctx.get("atr", m15_atr_values[-1]), len(pair_h4)),
            },
            "h1": {
                **_structure_summary(h1_structure, h1),
                "dealing_range": {
                    "high": round_price(h1_dr["high"]),
                    "low": round_price(h1_dr["low"]),
                    "mid": round_price(h1_dr["mid"]),
                    "location_current": round(location_ratio(current, h1_dr), 4),
                    "zone": (
                        "DISCOUNT" if location_ratio(current, h1_dr) < 0.5
                        else "PREMIUM" if location_ratio(current, h1_dr) > 0.5
                        else "EQUILIBRIUM"
                    ),
                },
                "trend_strength": trend_strength_metrics(h1, h1_structure),
                "fibonacci": fibonacci_summary(h1_ctx.get("fib", {}).get(best.direction), current),
                "poi": _zone_summary(h1_ctx.get("zones", [])[:10], current, h1_ctx.get("atr", m15_atr_values[-1]), len(h1)),
            },
            "m15": {
                **_structure_summary(m15_structure, m15),
                "rsi14": m15_rsi_ctx,
                "vlt": m15_vlt,
                "latest_displacement_score": round(
                    displacement_strength(
                        m15,
                        len(m15) - 1,
                        m15_atr_values,
                        m15_rel_volume,
                    ),
                    2,
                ),
                "atr14": round(m15_atr_values[-1], 8),
                "trend_strength": trend_strength_metrics(m15, m15_structure),
                "fibonacci": fibonacci_summary(m15_ctx.get("fib", {}).get(best.direction), current),
                "poi": _zone_summary(m15_ctx.get("zones", [])[:10], current, m15_atr_values[-1], len(m15)),
            },
        },
        "smc": {
            "liquidity_sweeps_recent": sweeps[-12:],
            "htf_poi_hierarchy": {
                "h4_primary_candidates": _zone_summary(h4_ctx.get("zones", [])[:12], current, h4_ctx.get("atr", m15_atr_values[-1]), len(pair_h4)),
                "h1_refinement_candidates": _zone_summary(h1_ctx.get("zones", [])[:12], current, h1_ctx.get("atr", m15_atr_values[-1]), len(h1)),
                "m15_execution_candidates": _zone_summary(m15_ctx.get("zones", [])[:12], current, m15_ctx.get("atr", m15_atr_values[-1]), len(m15)),
            },
            "fibonacci": {
                "h4_buy": fibonacci_summary(h4_ctx.get("fib", {}).get("BUY"), current),
                "h4_sell": fibonacci_summary(h4_ctx.get("fib", {}).get("SELL"), current),
                "h1_buy": fibonacci_summary(h1_ctx.get("fib", {}).get("BUY"), current),
                "h1_sell": fibonacci_summary(h1_ctx.get("fib", {}).get("SELL"), current),
                "m15_buy": fibonacci_summary(m15_ctx.get("fib", {}).get("BUY"), current),
                "m15_sell": fibonacci_summary(m15_ctx.get("fib", {}).get("SELL"), current),
            },
            "trend_strength": {
                "btc_h4": trend_strength_metrics(btc_h4, btc_h4_structure),
                "pair_h4": trend_strength_metrics(pair_h4, pair_h4_structure),
                "h1": trend_strength_metrics(h1, h1_structure),
                "m15": trend_strength_metrics(m15, m15_structure),
            },
            "fvg_recent": _zone_summary(fvg[-10:], current, m15_atr_values[-1], len(m15)),
            "order_blocks": _zone_summary(obs, current, m15_atr_values[-1], len(m15)),
            "breakers": _zone_summary(breakers, current, m15_atr_values[-1], len(m15)),
            "liquidity_pools": _liquidity_summary(liquidity, current, m15_atr_values[-1]),
        },
        "selected_setup": {
            "model": best.model,
            "direction": best.direction,
            "entry": best.entry,
            "entry_reason": best.entry_reason,
            "price_exp": best.price_exp,
            "price_exp_reason": best.price_exp_reason,
            "sl": best.sl,
            "sl_reason": best.sl_reason,
            "tp": best.tp,
            "tp_reason": best.tp_reason,
            "planned_rr": round(planned_rr, 4),
            "evidence": best.evidence,
        },
        "confidence": {
            "score": best.confidence,
            "components": best.scores,
            "meaning": "Kualitas internal setup 0-100; bukan probabilitas profit.",
        },
        "notes": notes_context,
        "candidate_set": [_candidate_summary(x) for x in sorted(candidates, key=lambda c: c.confidence, reverse=True)],
        "price_context": {
            "current": round_price(current),
            "atr14_m15": round_price(m15_atr_values[-1]),
            "current_distance_to_entry_pct": round(distance_pct(current, best.entry), 4),
            "entry_distance_to_exp_pct": round(distance_pct(best.entry, best.price_exp), 4),
        },
    }

    # Strategy output matches the bridge contract in main.py.
    return {
        "pair": normalized_pair,
        "direction": best.direction,
        "price_now_reference": round_price(current),
        "entry": best.entry,
        "entry_reason": best.entry_reason,
        "price_exp": best.price_exp,
        "price_exp_reason": best.price_exp_reason,
        "sl": best.sl,
        "sl_reason": best.sl_reason,
        "tp": best.tp,
        "tp_reason": best.tp_reason,
        "confidence": best.confidence,
        "confidence_components": best.scores,
        "analysis": analysis,
        "data": {
            "source": m15_source,
            "price_source": price_source,
            "fallback_used": m15_source == "BINANCE_FALLBACK",
            "directional_bias": final_macro,
            "directional_regime": directional_regime_label(
                allowed_directions_for_macro(
                    final_macro,
                    pair_h4_structure.trend,
                    normalized_pair,
                )
            ),
            "allowed_directions": list(
                allowed_directions_for_macro(
                    final_macro,
                    pair_h4_structure.trend,
                    normalized_pair,
                )
            ),
            "sources_by_timeframe": {
                "pair_m15": m15_source,
                "pair_h4": pair_h4_source,
                "btc_h4": btc_h4_source,
                "current_price": price_source,
            },
            "timeframe": "15m",
            "candles_requested": M15_CANDLES_REQUIRED,
            "candles_used": len(m15),
            "closed_candles_only": True,
            "pair_h4_candles": len(pair_h4),
            "btc_h4_candles": len(btc_h4),
            "h1_derived_candles": len(h1_derived),
            "h4_derived_candles": len(h4_derived),
            "tick_size": round_price(tick_size) if tick_size else None,
            "htf_poi_hierarchy": "H4_PRIMARY -> H1_REFINEMENT -> M15_EXECUTION",
            "fibonacci_enabled": True,
            "fibonacci_deep_filter": 0.618,
        },
        "strategy": {
            "name": STRATEGY_NAME,
            "version": STRATEGY_VERSION,
            "architecture": (
                "BTC H4 directional constraint + H4 primary POI + "
                "H4/H1 Fibonacci confluence + H1 refinement + M15 execution + "
                "SMC + RSI14 M15 + VLT/OHLCV"
            ),
            "confidence_semantics": "quality_score_not_profit_probability",
            "directional_rule": (
                "Altcoin BTC H4 BULLISH -> BUY only; "
                "BTC H4 BEARISH -> SELL only; BTC H4 RANGE -> both."
            ),
        },
    }


async def _fetch_if_short(pair: str, interval: str, interval_ms: int, count: int) -> list[Candle]:
    candles, _source = await fetch_series(pair, interval, count, interval_ms)
    return candles


# ============================================================================
# OPTIONAL SELF-TEST UTILITIES
# ============================================================================


def validate_directional_alignment(result: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate the v0.3 directional/top-down invariant from a strategy result."""
    errors: list[str] = []
    direction = str(result.get("direction") or "")
    analysis = result.get("analysis")
    macro = analysis.get("macro") if isinstance(analysis, dict) else None
    if not isinstance(macro, dict):
        errors.append("analysis.macro tidak tersedia")
        return False, errors

    allowed = macro.get("allowed_directions")
    if not isinstance(allowed, list) or not allowed:
        errors.append("analysis.macro.allowed_directions kosong")
    elif direction not in allowed:
        errors.append(
            f"direction {direction} tidak termasuk allowed_directions {allowed}"
        )

    return (not errors, errors)


def validate_result_contract(result: dict[str, Any]) -> tuple[bool, list[str]]:
    required = [
        "pair",
        "direction",
        "price_now_reference",
        "entry",
        "entry_reason",
        "price_exp",
        "price_exp_reason",
        "sl",
        "sl_reason",
        "tp",
        "tp_reason",
        "confidence",
        "confidence_components",
        "analysis",
        "data",
        "strategy",
    ]
    missing = [key for key in required if key not in result]
    errors: list[str] = []
    if missing:
        errors.append("missing: " + ", ".join(missing))
    if str(result.get("direction")) not in {"BUY", "SELL"}:
        errors.append("direction harus BUY/SELL")
    confidence = safe_float(result.get("confidence"), -1)
    if not 0 <= confidence <= 100:
        errors.append("confidence di luar 0-100")
    return (not errors, errors)


if __name__ == "__main__":
    print(f"{STRATEGY_NAME} v{STRATEGY_VERSION}")
    print("Module contract: async generate_setup(pair, context)")
