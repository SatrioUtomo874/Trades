from __future__ import annotations

"""
STRATEGY.PY
===========

Structural Prediction Engine V2 untuk main.py: market structure → thesis →
liquidity map → adaptive Fibonacci → FVG/OB refinement → projected RSI →
entry zone → prediction invalidation → structural SL → target map → RR →
thesis quality.

Kontrak utama tetap kompatibel dengan main.py:
    async def generate_setup(pair, context) -> dict
    async def analyze_btc_regime(context) -> dict
    async def analyze_scan_structure(pair, context) -> dict
    async def validate_setup(pair, initial_setup, context) -> dict
    async def analyze_trailing(trade, context) -> dict

Prinsip inti V2:
1. H4 structure menjadi sumber thesis utama; BTC hanya konteks confidence
   (bukan veto arah) untuk altcoin. D1/H1/M15 menjadi konteks dan refinement, bukan
   alasan untuk mengganti thesis H4 secara gegabah.
2. Semua keputusan struktur menggunakan CLOSED candles saja.
3. Liquidity dibangun sebagai zone map dari swing/equal/repeated/range levels
   lintas H4/H1/M15, kemudian diberi significance dan role.
4. Fibonacci adaptif: impulse clean → 0.382–0.500 sebagai lokasi utama;
   impulse struggle → 0.618–0.786 sebagai deeper retracement; weak impulse
   tetap dianalisis namun kualitasnya dibatasi.
5. FVG/OB hanya menjadi refinement setelah liquidity × Fibonacci menghasilkan
   lokasi yang relevan.
6. RSI digunakan dalam dua keadaan: current momentum dan projected RSI at
   entry zone melalui beberapa bounded scenario path. Projection adalah scenario
   analysis, bukan prediksi exact future price path.
7. Entry adalah zone, bukan angka arbitrary. SL mengikuti prediction
   invalidation structure dan dapat memakai liquidity guard; tidak memaksa ATR
   floor yang merusak thesis. TP memprioritaskan significant liquidity dan
   H4/H1 structural targets dengan RR minimum.
8. Confidence adalah thesis-quality score 0–100, bukan probabilitas profit.
9. Waiting state tetap informatif; fallback harus tetap rendah kualitas.
10. strategy.py tidak menyentuh state trade, Telegram, WebSocket, GitHub, atau
    order execution.

SMC yang dibuat objektif dalam kode:
    - Swing High / Swing Low
    - HH / HL / LH / LL
    - BOS / CHOCH / MSS
    - Equal High / Equal Low
    - Liquidity pool / sweep / magnet
    - Fair Value Gap / Order Block / Breaker
    - Premium / Discount / Fibonacci
    - Displacement / ATR / RSI(14) / relative-volume / trend slope

Catatan penting:
- SMC bukan standar teknikal tunggal dengan definisi universal; semua istilah
  diberi definisi operasional deterministik agar dapat diuji pada histori.
- VLT di sini adalah modul custom berbasis OHLCV, bukan true footprint/order-flow.
"""

import asyncio
import hashlib
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable

import requests


# ============================================================================
# CONSTANTS / CONFIG
# ============================================================================

STRATEGY_NAME = "SMC_VLT_RSI"
STRATEGY_ENGINE = "V2"
STRATEGY_SCHEMA_VERSION = 1
SCORING_FORMULA_VERSION = "V2.5-binance-orderflow-volume-pricelegs"
# Explicit provider contract consumed by main.py. Both analysis and active-price
# monitoring are intentionally Binance USD-M only; no cross-exchange fallback.
SCAN_PRIMARY_DATA_SOURCE = "BINANCE_PUBLIC"
PRIMARY_DATA_SOURCE = "BINANCE_PUBLIC"
STRATEGY_DATA_SOURCE = "BINANCE_PUBLIC"

_log = logging.getLogger("strategy")


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """Backward-compatible helper: all strategy parameters are code-owned, never env-tuned."""
    return float(default)


def _env_int(name: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    return int(_env_float(name, float(default), None if lo is None else float(lo), None if hi is None else float(hi)))


def _env_bool(name: str, default: bool = False) -> bool:
    """Backward-compatible helper: strategy switches are fixed in code."""
    return bool(default)


# Candle cache (RAM, bounded). Nilai akhir harus diukur di host deployment.
CANDLE_CACHE_ENABLED = _env_bool("CANDLE_CACHE_ENABLED", True)
CANDLE_CACHE_MAX_MB = _env_float("CANDLE_CACHE_MAX_MB", 24.0, 1.0, 512.0)
CANDLE_CACHE_MAX_ENTRIES = _env_int("CANDLE_CACHE_MAX_ENTRIES", 600, 10, 20000)
CANDLE_CACHE_TTL_M15 = _env_float("CANDLE_CACHE_TTL_M15", 45.0, 1.0, 3600.0)
CANDLE_CACHE_TTL_H1 = _env_float("CANDLE_CACHE_TTL_H1", 180.0, 1.0, 7200.0)
CANDLE_CACHE_TTL_H4 = _env_float("CANDLE_CACHE_TTL_H4", 600.0, 1.0, 14400.0)
CANDLE_BYTES_ESTIMATE = 120
SERIES_MIN_REFETCH_SECONDS = 3.0
SERIES_PROVIDER_LAG_GRACE_SECONDS = 20.0
SERIES_INCREMENTAL_MAX_CANDLES = 60  # cache basi <= N candle: ambil candle baru saja (bobot 1)
SERIES_MAX_CANDLES = 1500
SCAN_WEIGHT_BUDGET_PER_MINUTE = _env_int("SCAN_WEIGHT_BUDGET_PER_MINUTE", 1200, 100, 2400)  # jatah scan dari 2400 bobot IP/menit
SCAN_HEADER_PAUSE_WEIGHT = 1900  # header bobot IP setinggi ini: scan jeda dulu
SERIES_MAX_INVALID_FRACTION = 0.02
VALIDATION_MAX_AGE_M15 = _env_float("VALIDATION_MAX_AGE_M15", 30.0, 1.0, 600.0)
VALIDATION_MAX_AGE_H4 = _env_float("VALIDATION_MAX_AGE_H4", 300.0, 1.0, 3600.0)
TRAIL_M15_MAX_AGE_SECONDS = _env_float("TRAIL_M15_MAX_AGE_SECONDS", 60.0, 1.0, 900.0)

# Biaya (estimasi, per sisi). Funding hanya dihitung bila datanya tersedia.
EST_FEE_PCT_PER_SIDE = _env_float("EST_FEE_PCT_PER_SIDE", 0.05, 0.0, 5.0)
EST_SLIPPAGE_PCT_PER_SIDE = _env_float("EST_SLIPPAGE_PCT_PER_SIDE", 0.03, 0.0, 5.0)
# Entry limit = maker tanpa slippage; exit (SL/TP market) = taker + slippage.
EST_ENTRY_FEE_PCT = _env_float("EST_ENTRY_FEE_PCT", 0.02, 0.0, 5.0)
EST_ENTRY_SLIPPAGE_PCT = _env_float("EST_ENTRY_SLIPPAGE_PCT", 0.0, 0.0, 5.0)
# Entry: batas jarak (ATR H1), kedalaman maksimum di zona, pengali Price Exp.
ENTRY_MAX_DISTANCE_H1_ATR = 4.0
ENTRY_MAX_DEPTH_BIAS = 0.50
EXP_ENTRY_DISTANCE_MULT = 1.5
# Struktur: swing minimum (ATR) dan buffer break.
SWING_MIN_ATR_H4 = 1.0
SWING_MIN_ATR_H1 = 0.8
STRUCTURE_BREAK_BUFFER_ATR = 0.10
NO_REFINEMENT_PENALTY = 4.0
MIN_RR_AFTER_COSTS = _env_float("MIN_RR_AFTER_COSTS", 1.5, 0.0, 20.0)  # 0 = hanya laporan

# BTC hanya konteks confidence, bukan veto arah.
BTC_COUNTER_TREND_PENALTY = _env_float("BTC_COUNTER_TREND_PENALTY", 12.0, 0.0, 60.0)
BTC_EXTREME_CONFIDENCE = _env_float("BTC_EXTREME_CONFIDENCE", 90.0, 50.0, 100.0)

# Kode state kandidat & alasan (stabil, machine-readable).
CANDIDATE_STATES = (
    "DATA_UNAVAILABLE", "REJECTED_HARD", "REJECTED_QUALITY", "WAITING_PRICE",
    "WAITING_TRIGGER", "READY", "PENDING", "FILLED", "CLOSED", "EXPIRED",
    "SUPERSEDED", "REPLAY_PENDING", "REPLAY_COMPLETE", "REPLAY_AMBIGUOUS",
    "REPLAY_UNAVAILABLE",
)
REASON_CODES = (
    "GEOMETRY_INVALID", "ENTRY_OUTSIDE_ZONE", "ZONE_INVERTED", "ZONE_TOO_WIDE",
    "SL_MISSING", "TP_MISSING", "PRICE_INVALID", "RR_BELOW_MIN", "RR_AFTER_COST_BELOW_MIN",
    "DATA_STALE", "DATA_INSUFFICIENT", "DATA_INVALID", "TRIGGER_NOT_CONFIRMED",
    "LOW_QUALITY_FALLBACK", "BELOW_QUALITY_FLOOR", "BTC_COUNTER_TREND",
    "BTC_HIGH_RISK_CONFLICT", "PRICE_IN_ZONE", "PRICE_NOT_IN_ZONE", "READY_FOR_VALIDATION",
)
ENTRY_ZONE_MAX_WIDTH_PCT = 15.0
STRATEGY_VERSION = "2.6.0"

BINANCE_BASE_URL = "https://fapi.binance.com"

M15 = "15"
H1 = "60"
H4 = "240"
D1 = "D"

M15_MS = 15 * 60 * 1000
H1_MS = 60 * 60 * 1000
H4_MS = 4 * 60 * 60 * 1000

M15_CANDLES_REQUIRED = 672
PAIR_H4_CANDLES_REQUIRED = 120
BTC_H4_CANDLES_REQUIRED = 240

HTTP_TIMEOUT_SECONDS = 20
REQUEST_OBSERVER: Any = None  # diisi main.py: mencatat request & bobot IP

# Binance public REST requests are serialized and spaced by BinanceProvider.
# When Binance replies with HTTP 418/429 or error -1003, its provider raises a
# rate-limit exception for main.py to pause scanning and resume after cooldown.

SWING_SPAN_M15 = 3
SWING_SPAN_H1 = 2
SWING_SPAN_H4 = 2

FVG_MAX_AGE_M15 = 160
POI_MAX_DISTANCE_ATR = 3.5
H4_POI_MAX_DISTANCE_ATR = 8.0
H1_POI_MAX_DISTANCE_ATR = 4.0
H1_REFINEMENT_OVERLAP_ATR = 2.0
M15_EXECUTION_OVERLAP_ATR = 1.5
MAX_ENTRY_DISTANCE_PCT = 10.0
MAX_ENTRY_DISTANCE_H4_ATR = 2.5
MAX_PRIMARY_POI_CANDIDATES = 6
FIB_DEEP_RATIO = 0.618
FIB_DEEPEST_RATIO = 0.786
FIB_SHALLOW_RATIO = 0.382
FIB_MID_RATIO = 0.500
EXP_ATR_MULTIPLIER = 1.25
SL_BUFFER_ATR = 0.15
# Model risiko multi-timeframe: SL, Price Exp, dan target dihitung dari ATR H1
# dan invalidasi struktur H1/H4 (bukan noise M15).
SL_BUFFER_H1_ATR = 0.40
SL_MIN_H1_ATR = 1.5
SL_MAX_H1_ATR = 5.0
SL_MIN_PCT = 0.5
SL_SWING_EXTEND_H1_ATR = 1.0
SL_SWEEP_EXTEND_H1_ATR = 1.5
EXP_H1_ATR_MULTIPLIER = _env_float("EXP_H1_ATR_MULTIPLIER", 5.0, 0.5, 8.0)
SL_MIN_RISK_H1_ATR = _env_float("SL_MIN_RISK_H1_ATR", 0.8, 0.0, 3.0)
SL_MIN_RISK_PCT = _env_float("SL_MIN_RISK_PCT", 1.0, 0.0, 5.0)
MIN_PLANNED_RR = 2.0
TP_FALLBACK_RR = 2.5
FALLBACK_CONFIDENCE_CAP = 60.0
# Kerangka momentum RSI (regime -> archetype -> gate).
RSI_PERIOD = 14
RSI_REGIME_WINDOW = 60
DIV_MAX_AGE = {"H4": 6, "H1": 24, "M15": 40}
DIV_MIN_RSI_DIFF = 3.0
DIV_MIN_PRICE_PCT = 0.05
OVEREXT_H1_BUY = 72.0
OVEREXT_M15_BUY = 78.0
OVEREXT_H1_SELL = 28.0
OVEREXT_M15_SELL = 22.0
EQUILIBRIUM_BUY_MAX = 0.60
EQUILIBRIUM_SELL_MIN = 0.40
MOMENTUM_FACTOR_FLOOR = 0.50
VETO_CONFIDENCE_CAP = 55.0
UNCONFIRMED_CONFIDENCE_CAP = 64.0
# Gerbang kualitas pasar: anti pisau jatuh, pasca pompa, dan setup tanpa konfirmasi struktur.
KNIFE_MOVE_24H_PCT = 18.0
POST_PUMP_7D_PCT = 60.0
POST_PUMP_DROP_PCT = 25.0
DRY_VOLUME_RATIO = 0.45
TRIGGER_MAX_AGE_M15 = 32
SL_MAX_PCT = 4.5
RSI_PULLBACK_FLOOR = 32.0
RSI_PULLBACK_CEIL_SELL = 68.0
# Model likuiditas: pool signifikan, belum disapu, magnet terdekat, dan retest leg.
SIGNIFICANT_POOL_MIN = 70.0
MAGNET_MAX_H1_ATR = 3.0
SL_POOL_REACH_H1_ATR = 0.75
SL_POOL_BUFFER_H1_ATR = 0.25
OTE_LOW = 0.62
OTE_HIGH = 0.79
LOW_QUALITY_MODELS = {"M15_SMC_FALLBACK", "STRUCTURE_PULLBACK_FALLBACK"}


# ============================================================================
# STRUCTURAL PREDICTION ENGINE V2 CONFIG
# ============================================================================

# Liquidity clustering / map.
LIQ_MIN_TOLERANCE_PCT = 0.0010       # 0.10% of price
LIQ_ATR_TOLERANCE = 0.15            # adaptive clustering tolerance
LIQ_MIN_TOUCHES = 1
LIQ_H4_BONUS = 20.0
LIQ_H1_BONUS = 10.0
LIQ_EQUAL_BONUS = 10.0
LIQ_RECENT_BONUS = 8.0
LIQ_MULTI_SOURCE_BONUS = 8.0
LIQ_RANGE_EXTREME_BONUS = 5.0
LIQ_SWEEP_PENALTY = 30.0
LIQ_ZONE_MAX_PER_TF = 30
LIQ_SIGNIFICANT_THRESHOLD_V2 = 70.0
LIQ_NEAR_ENTRY_ATR = 0.80
LIQ_MAGNET_MAX_H1_ATR_V2 = 3.0

# Adaptive Fibonacci.
FIB_NORMAL_LOW = 0.382
FIB_NORMAL_HIGH = 0.500
FIB_DEEP_LOW = 0.618
FIB_DEEP_HIGH = 0.786
FIB_SECONDARY_SHALLOW_LOW = 0.500
FIB_SECONDARY_SHALLOW_HIGH = 0.618
FIB_IMPULSE_MIN_MOVE_PCT = 1.0
FIB_CLEAN_QUALITY = 72.0
FIB_STRUGGLE_QUALITY = 48.0
FIB_MIN_STRUCTURE_PRESERVED = True

# Refinement / entry zone.
REFINEMENT_MIN_OVERLAP = 0.10
REFINEMENT_GOOD_OVERLAP = 0.35
ENTRY_ZONE_MIN_WIDTH_PCT = 0.03
ENTRY_ZONE_MAX_WIDTH_PCT = 1.80
ENTRY_ZONE_MAX_COMPONENTS = 8
ENTRY_SELECTION_RSI_WEIGHT = 0.35
ENTRY_SELECTION_CONFLUENCE_WEIGHT = 0.35
ENTRY_SELECTION_RISK_WEIGHT = 0.15
ENTRY_SELECTION_REACH_WEIGHT = 0.15

# Projected RSI.
RSI_PROJECTION_MIN_STEPS_M15 = 3
RSI_PROJECTION_MAX_STEPS_M15 = 12
RSI_PROJECTION_STEPS_H1 = 2
RSI_PROJECTION_STEPS_H4 = 4
RSI_PROJECTION_SCENARIOS = ("FAST_RETRACE", "NORMAL_RETRACE", "SLOW_RETRACE")
RSI_BUY_IDEAL_MAX_M15 = 40.0
RSI_BUY_ACCEPTABLE_MAX_M15 = 48.0
RSI_BUY_WARNING_MAX_M15 = 55.0
RSI_SELL_IDEAL_MIN_M15 = 60.0
RSI_SELL_ACCEPTABLE_MIN_M15 = 52.0
RSI_SELL_WARNING_MIN_M15 = 45.0
RSI_H4_BREAKDOWN_BUY = 35.0
RSI_H4_BREAKDOWN_SELL = 65.0
RSI_PROJECTION_MIN_CONSENSUS = 55.0
# Momentum RSI: jangan jual saat RSI H4 sudah jenuh turun (beli: jenuh naik); entry harus
# berada di zona yang diproyeksikan membuat RSI M15 mencapai level momentum.
RSI_H4_EXHAUST_SELL = _env_float("RSI_H4_EXHAUST_SELL", 40.0, 20.0, 50.0)
RSI_H4_EXHAUST_BUY = _env_float("RSI_H4_EXHAUST_BUY", 60.0, 50.0, 80.0)
RSI_SELL_ENTRY_TARGET_M15 = _env_float("RSI_SELL_ENTRY_TARGET_M15", 66.0, 50.0, 85.0)
RSI_BUY_ENTRY_TARGET_M15 = _env_float("RSI_BUY_ENTRY_TARGET_M15", 34.0, 15.0, 50.0)
VOLUME_BASELINE_BARS = 30
RSI_EXHAUST_PENALTY_PER_POINT = _env_float("RSI_EXHAUST_PENALTY_PER_POINT", 1.5, 0.0, 5.0)
RSI_EXHAUST_MAX_PENALTY = _env_float("RSI_EXHAUST_MAX_PENALTY", 18.0, 0.0, 40.0)
RSI_TARGET_PENALTY_PER_POINT = _env_float("RSI_TARGET_PENALTY_PER_POINT", 0.6, 0.0, 3.0)
RSI_TARGET_MAX_PENALTY = _env_float("RSI_TARGET_MAX_PENALTY", 10.0, 0.0, 30.0)

# Structural invalidation / SL V2.
MIN_INVALIDATION_STRENGTH = 55.0
SL_INVALIDATION_BUFFER_ATR = 0.10
SL_INVALIDATION_BUFFER_MAX_ATR = 0.35
SL_MIN_BUFFER_PCT = 0.03
SL_MAX_RISK_H1_ATR_V2 = 5.0
SL_MAX_RISK_PCT_V2 = 4.5
SL_TOO_TIGHT_ATR_RATIO = 0.25
SL_IDEAL_H1_ATR_LOW = 0.70
SL_IDEAL_H1_ATR_HIGH = 3.00

# Target / RR V2.
TARGET_MIN_RR = 2.0
TARGET_PREFERRED_RR_LOW = 2.0
TARGET_PREFERRED_RR_HIGH = 3.5
TARGET_MAX_DISTANCE_H1_ATR = 8.0
TARGET_MAJOR_BARRIER_THRESHOLD = 82.0
TARGET_OBSTACLE_PENALTY = 15.0
TARGET_H4_SWING_BONUS = 25.0
TARGET_H1_SWING_BONUS = 15.0
TARGET_EXTREME_BONUS = 20.0
TARGET_LIQUIDITY_BASE = 35.0

# Thesis scoring V2. Weights total 100.
THESIS_WEIGHTS_V2 = {
    "structure_alignment": 8.0,
    "liquidity_confluence": 13.0,
    "fib_location": 7.0,
    "fvg_ob_refinement": 5.0,
    "rsi_projection": 11.0,
    "momentum_current": 4.0,
    "invalidation_quality": 8.0,
    "target_quality": 7.0,
    "rr_quality": 4.0,
    "entry_reachability": 3.0,
    "volume_confirmation": 5.0,
    "orderflow_leg": 8.0,
    "price_leg": 8.0,
    "support_resistance": 9.0,
}
THESIS_WAITING_CAP_V2 = 78.0  # batas atas lunak; pengurangnya ada di THESIS_WAITING_PENALTY_V2
THESIS_WAITING_PENALTY_V2 = _env_float("THESIS_WAITING_PENALTY_V2", 6.0, 0.0, 30.0)
THESIS_VETO_CAP_V2 = 55.0
THESIS_FALLBACK_CAP_V2 = 60.0
THESIS_MIN_QUALITY_V2 = 45.0

# Thesis freshness / trigger age.
THESIS_MAX_TRIGGER_AGE_M15 = 32
THESIS_EXPIRY_ATR_MULT = 1.50

# V2 model names.
PRIMARY_V2_MODEL = "STRUCTURAL_PREDICTION_V2"
WAITING_V2_MODEL = "STRUCTURAL_PREDICTION_V2_WAITING"
FALLBACK_V2_MODEL = "STRUCTURAL_PREDICTION_V2_FALLBACK"

WEIGHTS = {
    # BTC regime is a directional constraint; the always-saturated gates
    # (H1 refinement, M15 trigger) keep small weights so scores can separate.
    "htf_alignment": 14.0,
    "h4_poi": 10.0,
    "h4_fibonacci": 14.0,
    "h1_refinement": 6.0,
    "liquidity": 8.0,
    "smc_trigger": 6.0,
    "displacement": 8.0,
    "vlt": 8.0,
    "planned_rr": 8.0,
    "entry_reachability": 5.0,
    "risk_quality": 5.0,
}
DIRECTION_BUY = "BUY"
DIRECTION_SELL = "SELL"
DIRECTIONS_BOTH = (DIRECTION_BUY, DIRECTION_SELL)

EPS = 1e-12

# Cache khusus scan: BTC H4 dihitung sekali per cycle lalu dipakai ulang oleh
# seluruh pair agar directional regime dalam satu cycle konsisten dan tidak
# membebani API berulang-ulang. Tidak digunakan untuk order/trade state.
_SCAN_BTC_REGIME_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


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
class LiquidityZone:
    side: str                  # BUY_SIDE / SELL_SIDE
    low: float
    high: float
    midpoint: float
    strength: float
    timeframe: str
    touches: int
    source_types: list[str]
    first_index: int
    last_index: int
    swept: bool = False
    sweep_index: int | None = None
    distance_pct: float = 0.0
    distance_atr: float = 0.0
    significance: float = 0.0
    role: str = "INTERNAL_LIQUIDITY"


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
    # V2 analytical objects. Defaults preserve compatibility with all V1 callers.
    entry_low: float = 0.0
    entry_high: float = 0.0
    thesis_status: str = ""
    thesis_type: str = ""
    liquidity_zone: dict[str, Any] | None = None
    fib_zone: dict[str, Any] | None = None
    refinement_zone: dict[str, Any] | None = None
    projected_rsi: dict[str, Any] | None = None
    invalidation: dict[str, Any] | None = None
    target_map: list[dict[str, Any]] | None = None


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


class InsufficientHistoryError(RuntimeError):
    """Pair belum punya cukup candle closed (mis. listing baru)."""

    insufficient_history = True


class BinanceRateLimitError(RuntimeError):
    """Raised on Binance HTTP 418/429 so main.py can pause and resume scanner."""
    binance_rate_limited = True

    def __init__(self, message: str, *, status_code: int, endpoint: str, retry_after_seconds: float = 60.0) -> None:
        self.status_code = int(status_code)
        self.code = None
        self.endpoint = endpoint
        self.server_cooldown_seconds = max(1.0, float(retry_after_seconds))
        self.bot_cooldown_seconds = self.server_cooldown_seconds + 60.0
        self.retry_after_seconds = self.server_cooldown_seconds
        super().__init__(message)


def _request_weight(path: str, params: dict[str, Any]) -> int:
    if path.endswith("/klines"):
        limit = int(params.get("limit") or 500)
        return 1 if limit < 100 else 2 if limit < 500 else 5 if limit <= 1000 else 10
    if path.endswith("/ticker/24hr") and not params.get("symbol"):
        return 40
    return 1


class BinanceProvider(HTTPProvider):
    _request_lock: asyncio.Lock | None = None
    _last_request_monotonic: float = 0.0
    _cooldown_until_monotonic: float = 0.0
    _min_request_interval_seconds: float = 0.12
    _weight_log: list = []
    _used_weight: int = 0
    _used_weight_at: float = 0.0
    _tick_size_cache: dict[str, float] = {}
    _tick_size_cache_expires_monotonic: float = 0.0
    _tick_size_cache_ttl_seconds: float = 1800.0

    def __init__(self) -> None:
        super().__init__(BINANCE_BASE_URL, "BINANCE")

    @classmethod
    def _get_request_lock(cls) -> asyncio.Lock:
        if cls._request_lock is None:
            cls._request_lock = asyncio.Lock()
        return cls._request_lock

    @classmethod
    async def _pace_weight(cls, weight: int) -> None:
        """Jaga bobot request scan di bawah jatah per menit agar IP tidak kena 429."""
        while True:
            now = time.monotonic()
            cls._weight_log = [(t, w) for t, w in cls._weight_log if now - t < 60.0]
            header_age = now - cls._used_weight_at
            if cls._used_weight >= SCAN_HEADER_PAUSE_WEIGHT and header_age < 20.0:
                await asyncio.sleep(min(5.0, 20.0 - header_age))
                continue
            used = sum(w for _, w in cls._weight_log)
            if used + weight <= SCAN_WEIGHT_BUDGET_PER_MINUTE or not cls._weight_log:
                cls._weight_log.append((now, weight))
                return
            wait = 60.0 - (now - cls._weight_log[0][0]) + 0.05
            await asyncio.sleep(min(max(wait, 0.1), 5.0))

    async def get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        lock = self._get_request_lock()
        async with lock:
            wait = self.__class__._cooldown_until_monotonic - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            spacing = self.__class__._last_request_monotonic + self._min_request_interval_seconds - time.monotonic()
            if spacing > 0:
                await asyncio.sleep(spacing)
            await self._pace_weight(_request_weight(path, params or {}))
            def request() -> tuple[int, Any, dict[str, Any], str]:
                response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
                try:
                    body = response.json()
                except ValueError:
                    body = None
                return response.status_code, body, dict(response.headers), response.text[:500]
            status, body, headers, body_text = await asyncio.to_thread(request)
            self.__class__._last_request_monotonic = time.monotonic()
            raw_used = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
            if raw_used:
                try:
                    self.__class__._used_weight = int(raw_used)
                    self.__class__._used_weight_at = time.monotonic()
                except ValueError:
                    pass
            if REQUEST_OBSERVER is not None:
                try:
                    REQUEST_OBSERVER(path, status, raw_used)
                except Exception:
                    pass
        code = body.get("code") if isinstance(body, dict) else None
        msg = body.get("msg") if isinstance(body, dict) else body_text
        rate_limited = status in (418, 429) or str(code) == "-1003" or "too many requests" in str(msg).lower()
        if rate_limited:
            raw_retry = headers.get("Retry-After") or headers.get("retry-after")
            try:
                retry = max(1.0, float(raw_retry)) if raw_retry not in (None, "") else 60.0
            except (TypeError, ValueError):
                retry = 60.0
            self.__class__._cooldown_until_monotonic = max(
                self.__class__._cooldown_until_monotonic, time.monotonic() + retry
            )
            raise BinanceRateLimitError(
                f"Binance public REST rate limit HTTP {status} code={code}: {body_text}",
                status_code=status if status else 429, endpoint=path, retry_after_seconds=retry,
            )
        if status >= 400:
            raise RuntimeError(f"Binance REST {path}: HTTP {status} code={code} msg={msg}")
        if body is None:
            raise RuntimeError(f"Binance REST {path}: JSON response invalid")
        return body

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
            raise InsufficientHistoryError(
                f"{symbol} {_binance_interval(interval)}: candle closed {len(candles)} "
                f"dari {count} yang dibutuhkan (kemungkinan listing baru)"
            )
        return candles[-count:]

    async def klines_tail(self, symbol: str, interval: str, limit: int, interval_ms: int) -> list[Candle]:
        """Candle closed terbaru saja (limit kecil = bobot 1)."""
        payload = await self.get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": _binance_interval(interval), "limit": int(limit)},
        )
        if not isinstance(payload, list):
            raise RuntimeError("Binance klines response bukan list.")
        now_ms = int(time.time() * 1000)
        out: list[Candle] = []
        for row in payload:
            if len(row) < 7:
                continue
            start = int(row[0])
            if start + interval_ms > now_ms:
                continue
            out.append(
                Candle(
                    time_ms=start,
                    open=safe_float(row[1]),
                    high=safe_float(row[2]),
                    low=safe_float(row[3]),
                    close=safe_float(row[4]),
                    volume=safe_float(row[5]),
                    turnover=safe_float(row[7]) if len(row) > 7 else 0.0,
                )
            )
        return sorted(out, key=lambda c: c.time_ms)

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
        symbol = str(symbol).upper().strip()
        now = time.monotonic()
        cache = self.__class__._tick_size_cache
        if now < self.__class__._tick_size_cache_expires_monotonic and symbol in cache:
            return cache[symbol]

        # One exchangeInfo fetch populates tick sizes for all symbols. Scan batches
        # query many symbols sequentially, so this avoids repeated heavy JSON reads.
        payload = await self.get("/fapi/v1/exchangeInfo", {})
        refreshed: dict[str, float] = {}
        for raw in payload.get("symbols", []):
            raw_symbol = str(raw.get("symbol") or "").upper().strip()
            for filt in raw.get("filters", []):
                if filt.get("filterType") == "PRICE_FILTER":
                    tick = safe_float(filt.get("tickSize"), 0.0)
                    if raw_symbol and tick > 0:
                        refreshed[raw_symbol] = tick
                    break
        cache.clear()
        cache.update(refreshed)
        self.__class__._tick_size_cache_expires_monotonic = now + self.__class__._tick_size_cache_ttl_seconds
        return cache.get(symbol, 0.0)


def _binance_interval(interval: str) -> str:
    mapping = {"15": "15m", "60": "1h", "240": "4h"}
    return mapping.get(str(interval), str(interval))


# ============================================================================
# FETCHING / NORMALIZATION
# ============================================================================

class DataInvalidError(RuntimeError):
    """Seri candle rusak (geometri OHLC tidak masuk akal)."""


def _valid_candle(c: Candle) -> bool:
    try:
        vals = (c.open, c.high, c.low, c.close)
        if not all(math.isfinite(v) and v > 0 for v in vals):
            return False
        return c.high >= c.low and c.low <= min(c.open, c.close) and c.high >= max(c.open, c.close)
    except (TypeError, ValueError):
        return False


def normalize_candles(
    candles: list[Candle],
    interval_ms: int,
    *,
    now_ms: int | None = None,
) -> tuple[list[Candle], dict[str, int]]:
    """Urutkan, dedupe, buang candle belum close / geometri rusak. Tidak memutasi input."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    by_time: dict[int, Candle] = {}
    stats = {"input": len(candles), "open_dropped": 0, "invalid_dropped": 0, "duplicates": 0, "gaps": 0}
    for c in candles:
        if c.time_ms + interval_ms > now_ms:
            stats["open_dropped"] += 1
            continue
        if not _valid_candle(c):
            stats["invalid_dropped"] += 1
            continue
        if c.time_ms in by_time:
            stats["duplicates"] += 1
        by_time[c.time_ms] = c
    ordered = [by_time[k] for k in sorted(by_time)]
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.time_ms - prev.time_ms > interval_ms:
            stats["gaps"] += 1
    seen = max(1, stats["input"] - stats["open_dropped"])
    if stats["invalid_dropped"] / seen > SERIES_MAX_INVALID_FRACTION:
        raise DataInvalidError(
            f"{stats['invalid_dropped']} dari {seen} candle memiliki geometri OHLC tidak valid."
        )
    return ordered, stats


@dataclass(slots=True)
class _SeriesEntry:
    candles: list
    source: str
    fetched_at: float
    last_closed_ms: int
    interval_ms: int
    approx_bytes: int
    gaps: int = 0


_SERIES_CACHE: "OrderedDict[tuple[str, str, str], _SeriesEntry]" = OrderedDict()
_SERIES_INFLIGHT: dict[tuple, Any] = {}
_SERIES_STATS: dict[str, int] = {
    "hits": 0, "misses": 0, "evictions": 0, "refreshes": 0, "failures": 0,
    "inflight_shared": 0, "stores": 0, "bypass": 0,
}
_SERIES_LOCK = threading.Lock()


def _series_ttl(interval_ms: int) -> float:
    if interval_ms <= 15 * 60_000:
        return CANDLE_CACHE_TTL_M15
    if interval_ms <= 60 * 60_000:
        return CANDLE_CACHE_TTL_H1
    return CANDLE_CACHE_TTL_H4


def _expected_last_closed_ms(now_ms: int, interval_ms: int) -> int:
    return (now_ms // interval_ms) * interval_ms - interval_ms


def _series_is_fresh(entry: _SeriesEntry, now: float, max_age: float | None) -> bool:
    now_ms = int(now * 1000)
    expected = _expected_last_closed_ms(now_ms, entry.interval_ms)
    age = now - entry.fetched_at
    if entry.last_closed_ms >= expected:
        return True  # candle closed terbaru sudah ada; data closed tidak berubah
    if entry.last_closed_ms < expected:
        # Candle baru seharusnya sudah ada. Beri jeda singkat agar tidak refetch beruntun,
        # dan toleransi bila provider memang tidak punya candle itu (diambil >20 dtk setelah close).
        close_ms = expected + entry.interval_ms
        if entry.fetched_at * 1000 >= close_ms + SERIES_PROVIDER_LAG_GRACE_SECONDS * 1000:
            pass
        else:
            return age < SERIES_MIN_REFETCH_SECONDS
    limit = max_age if max_age is not None else _series_ttl(entry.interval_ms)
    return age <= limit


def _series_total_bytes() -> int:
    return sum(e.approx_bytes for e in _SERIES_CACHE.values())


def _series_enforce_bounds() -> None:
    max_bytes = int(CANDLE_CACHE_MAX_MB * 1024 * 1024)
    total = _series_total_bytes()
    while _SERIES_CACHE and (len(_SERIES_CACHE) > CANDLE_CACHE_MAX_ENTRIES or total > max_bytes):
        _key, evicted = _SERIES_CACHE.popitem(last=False)
        total -= evicted.approx_bytes
        _SERIES_STATS["evictions"] += 1


def _series_cache_get(
    pair: str, interval: str, count: int, allow_fallback: bool, max_age: float | None
) -> tuple[list[Candle], str] | None:
    providers = ("BINANCE",)
    now = time.time()
    with _SERIES_LOCK:
        for provider in providers:
            entry = _SERIES_CACHE.get((provider, pair, interval))
            if entry is None or len(entry.candles) < count:
                continue
            if not _series_is_fresh(entry, now, max_age):
                continue
            _SERIES_CACHE.move_to_end((provider, pair, interval))
            _SERIES_STATS["hits"] += 1
            return list(entry.candles[-count:]), entry.source
    return None


def _series_cache_store(
    pair: str, interval: str, candles: list[Candle], source: str, interval_ms: int, gaps: int = 0
) -> None:
    if not candles:
        return
    key = ("BINANCE", pair, interval)
    with _SERIES_LOCK:
        old = _SERIES_CACHE.get(key)
        merged = list(candles)
        if old is not None and old.candles and old.candles[0].time_ms < merged[0].time_ms:
            merged = [c for c in old.candles if c.time_ms < merged[0].time_ms] + merged
        merged = merged[-SERIES_MAX_CANDLES:]
        _SERIES_CACHE[key] = _SeriesEntry(
            candles=merged,
            source=source,
            fetched_at=time.time(),
            last_closed_ms=merged[-1].time_ms,
            interval_ms=interval_ms,
            approx_bytes=len(merged) * CANDLE_BYTES_ESTIMATE,
            gaps=gaps,
        )
        _SERIES_CACHE.move_to_end(key)
        _SERIES_STATS["stores"] += 1
        _series_enforce_bounds()


def series_cache_stats() -> dict[str, Any]:
    with _SERIES_LOCK:
        hits, misses = _SERIES_STATS["hits"], _SERIES_STATS["misses"]
        total = hits + misses
        return {
            **_SERIES_STATS,
            "entries": len(_SERIES_CACHE),
            "approx_mb": round(_series_total_bytes() / 1024 / 1024, 3),
            "max_mb": CANDLE_CACHE_MAX_MB,
            "max_entries": CANDLE_CACHE_MAX_ENTRIES,
            "hit_rate": round(hits / total, 4) if total else None,
            "enabled": CANDLE_CACHE_ENABLED,
        }


def series_cache_clear(pair: str | None = None) -> int:
    with _SERIES_LOCK:
        if pair is None:
            removed = len(_SERIES_CACHE)
            _SERIES_CACHE.clear()
            return removed
        keys = [k for k in _SERIES_CACHE if k[1] == pair.upper()]
        for key in keys:
            _SERIES_CACHE.pop(key, None)
        return len(keys)


def series_cache_shrink(fraction: float = 0.5) -> int:
    """Buang entri LRU (dipakai saat tekanan memori)."""
    with _SERIES_LOCK:
        target = int(len(_SERIES_CACHE) * max(0.0, min(1.0, fraction)))
        for _ in range(target):
            _SERIES_CACHE.popitem(last=False)
            _SERIES_STATS["evictions"] += 1
        return target


async def _fetch_series_uncached(
    pair: str,
    interval: str,
    count: int,
    interval_ms: int,
    *,
    allow_fallback: bool,
) -> tuple[list[Candle], str, dict[str, int]]:
    # Binance REST is the sole source for all candle timeframes.
    binance = BinanceProvider()
    raw = await binance.klines(pair, interval, count, interval_ms)
    source = "BINANCE"
    candles, qstats = normalize_candles(raw, interval_ms)
    if len(candles) < count:
        raise InsufficientHistoryError(
            f"{source} hanya memiliki {len(candles)} candle valid untuk {pair} {interval}; diperlukan {count}."
        )
    return candles[-count:], source, qstats


async def _fetch_series_incremental(
    pair: str, interval: str, count: int, interval_ms: int
) -> tuple[list[Candle], str, dict[str, int]] | None:
    """Cache lama + candle baru saja, menggantikan fetch ulang penuh."""
    with _SERIES_LOCK:
        entry = _SERIES_CACHE.get(("BINANCE", pair, interval))
        if entry is None or len(entry.candles) < count:
            return None
        old = list(entry.candles)
        last_closed = entry.last_closed_ms
    expected = _expected_last_closed_ms(int(time.time() * 1000), interval_ms)
    missing = int((expected - last_closed) // interval_ms)
    if missing < 1 or missing > SERIES_INCREMENTAL_MAX_CANDLES:
        return None
    tail = await BinanceProvider().klines_tail(pair, interval, missing + 2, interval_ms)
    if not tail or tail[0].time_ms > last_closed + interval_ms:
        return None  # ada celah: ambil penuh
    merged = [c for c in old if c.time_ms < tail[0].time_ms] + tail
    candles, qstats = normalize_candles(merged, interval_ms)
    if len(candles) < count:
        return None
    return candles[-count:], "BINANCE", qstats


async def fetch_series(
    pair: str,
    interval: str,
    count: int,
    interval_ms: int,
    *,
    allow_fallback: bool = True,
    max_age_seconds: float | None = None,
    force_refresh: bool = False,
    use_cache: bool = True,
) -> tuple[list[Candle], str]:
    """Ambil candle closed melalui Binance REST only.

    Cache bounded + single-flight. Seluruh mode memakai Binance sebagai satu-satunya sumber.
    """
    symbol = str(pair).upper()
    if not (use_cache and CANDLE_CACHE_ENABLED):
        _SERIES_STATS["bypass"] += 1
        candles, source, _ = await _fetch_series_uncached(
            symbol, interval, count, interval_ms, allow_fallback=allow_fallback
        )
        return candles, source

    if not force_refresh:
        hit = _series_cache_get(symbol, interval, count, allow_fallback, max_age_seconds)
        if hit is not None:
            return hit

    flight_key = (symbol, interval, bool(allow_fallback), int(count))
    pending = _SERIES_INFLIGHT.get(flight_key)
    if pending is not None:
        _SERIES_STATS["inflight_shared"] += 1
        candles, source = await asyncio.shield(pending)
        return list(candles), source

    _SERIES_STATS["misses"] += 1
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    future.add_done_callback(lambda f: f.cancelled() or f.exception())
    _SERIES_INFLIGHT[flight_key] = future
    try:
        incremental = await _fetch_series_incremental(symbol, interval, count, interval_ms)
        if incremental is not None:
            candles, source, qstats = incremental
        else:
            candles, source, qstats = await _fetch_series_uncached(
                symbol, interval, count, interval_ms, allow_fallback=allow_fallback
            )
        _series_cache_store(symbol, interval, candles, source, interval_ms, qstats.get("gaps", 0))
        if force_refresh:
            _SERIES_STATS["refreshes"] += 1
        if not future.done():
            future.set_result((candles, source))
        return list(candles), source
    except BaseException as exc:
        _SERIES_STATS["failures"] += 1
        if not future.done():
            if isinstance(exc, asyncio.CancelledError):
                future.set_exception(RuntimeError("fetch dibatalkan"))
            else:
                future.set_exception(exc)
        raise
    finally:
        if _SERIES_INFLIGHT.get(flight_key) is future:
            _SERIES_INFLIGHT.pop(flight_key, None)


async def fetch_price(
    pair: str,
    preferred_source: str | None = None,
    *,
    allow_fallback: bool = True,
) -> tuple[float, str]:
    """Fetch last price from Binance REST only; signature retained for compatibility."""
    price = await BinanceProvider().last_price(pair)
    return price, "BINANCE"


async def get_tick_size(
    pair: str,
    *,
    source: str = "BINANCE",
) -> float:
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

    # Infer base candle size so D1 can be built from H4 and H1 from M15.
    spans = sorted(
        b.time_ms - a.time_ms
        for a, b in zip(candles, candles[1:])
        if b.time_ms > a.time_ms
    )
    base_ms = spans[len(spans) // 2] if spans else M15_MS
    expected = max(1, bucket_ms // base_ms) if bucket_ms % base_ms == 0 else 1

    result: list[Candle] = []
    for start in sorted(buckets):
        rows = sorted(buckets[start], key=lambda x: x.time_ms)
        if not rows:
            continue
        # A bucket is only valid if it has all expected component candles.
        if len(rows) < expected:
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


def _significant_pivots(
    atr_values: list[float],
    highs: list[Pivot],
    lows: list[Pivot],
    min_atr: float,
) -> tuple[list[Pivot], list[Pivot]]:
    """Zigzag ATR: buang swing kecil (noise), pivot sejenis berurutan diambil yang paling ekstrem."""
    if min_atr <= 0 or len(highs) + len(lows) < 3:
        return highs, lows
    kept: list[Pivot] = []
    for pv in sorted([*highs, *lows], key=lambda x: x.index):
        if not kept:
            kept.append(pv)
            continue
        last = kept[-1]
        if pv.kind == last.kind:
            if (pv.kind == "HIGH" and pv.price > last.price) or (pv.kind == "LOW" and pv.price < last.price):
                kept[-1] = pv
            continue
        atr = max(atr_values[min(pv.index, len(atr_values) - 1)], EPS) if atr_values else EPS
        if abs(pv.price - last.price) >= min_atr * atr:
            kept.append(pv)
    new_highs = [x for x in kept if x.kind == "HIGH"]
    new_lows = [x for x in kept if x.kind == "LOW"]
    if len(new_highs) < 2 or len(new_lows) < 2:
        return highs, lows
    return new_highs, new_lows


def _swing_min_atr(timeframe: str) -> float:
    return {"H4": SWING_MIN_ATR_H4, "H1": SWING_MIN_ATR_H1}.get(str(timeframe).upper(), 0.0)


def build_structure(
    candles: list[Candle],
    span: int,
    min_swing_atr: float = 0.0,
) -> StructureSnapshot:
    highs, lows = detect_pivots(candles, span)
    atr_values = atr_series(candles, 14)
    highs, lows = _significant_pivots(atr_values, highs, lows, min_swing_atr)
    labels = label_pivots(highs, lows)
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

        if current_high and current_high.index not in broken_high_indices and candle.close > current_high.price + STRUCTURE_BREAK_BUFFER_ATR * max(atr_values[i], EPS):
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

        if current_low and current_low.index not in broken_low_indices and candle.close < current_low.price - STRUCTURE_BREAK_BUFFER_ATR * max(atr_values[i], EPS):
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

    confirmed_highs = [p for p in highs if p.confirmed_at_index < len(candles)]
    confirmed_lows = [p for p in lows if p.confirmed_at_index < len(candles)]

    # Final structural regime: do not let one internal break flip the whole
    # timeframe. Require agreement from the latest confirmed high/low
    # progression, or a recent structural break that is still supported by the
    # opposite protected swing.
    atr_now = max(median_or(atr_values[-20:], candles[-1].range), EPS)
    high_bull = len(confirmed_highs) >= 2 and confirmed_highs[-1].price > confirmed_highs[-2].price + 0.35 * atr_now
    high_bear = len(confirmed_highs) >= 2 and confirmed_highs[-1].price < confirmed_highs[-2].price - 0.35 * atr_now
    low_bull = len(confirmed_lows) >= 2 and confirmed_lows[-1].price > confirmed_lows[-2].price + 0.35 * atr_now
    low_bear = len(confirmed_lows) >= 2 and confirmed_lows[-1].price < confirmed_lows[-2].price - 0.35 * atr_now
    if high_bull and low_bull:
        trend = "BULLISH"
    elif high_bear and low_bear:
        trend = "BEARISH"
    else:
        # Mixed structure: require a clear majority before declaring a direction.
        bullish_parts = int(high_bull) + int(low_bull)
        bearish_parts = int(high_bear) + int(low_bear)
        if bullish_parts > bearish_parts and bullish_parts >= 1 and trend == "BULLISH":
            trend = "BULLISH"
        elif bearish_parts > bullish_parts and bearish_parts >= 1 and trend == "BEARISH":
            trend = "BEARISH"
        else:
            trend = "RANGE"

    last_bull_break = next((e for e in reversed(events) if e.get("direction") == "BULLISH" and e.get("type") in {"BOS", "MSS"}), None)
    last_bear_break = next((e for e in reversed(events) if e.get("direction") == "BEARISH" and e.get("type") in {"BOS", "MSS"}), None)
    if trend == "BULLISH":
        ref = int(last_bull_break.get("index")) if last_bull_break else len(candles)
        protected_candidates = [p for p in confirmed_lows if p.index < ref]
        protected_low = protected_candidates[-1].price if protected_candidates else (confirmed_lows[-1].price if confirmed_lows else None)
        protected_high = confirmed_highs[-1].price if confirmed_highs else None
    elif trend == "BEARISH":
        ref = int(last_bear_break.get("index")) if last_bear_break else len(candles)
        protected_candidates = [p for p in confirmed_highs if p.index < ref]
        protected_high = protected_candidates[-1].price if protected_candidates else (confirmed_highs[-1].price if confirmed_highs else None)
        protected_low = confirmed_lows[-1].price if confirmed_lows else None
    else:
        protected_high = confirmed_highs[-1].price if confirmed_highs else None
        protected_low = confirmed_lows[-1].price if confirmed_lows else None

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


def _zone_is_retracement_side(zone: Zone, direction: str, current: float) -> bool:
    """A pending entry zone must sit on the retracement side of current price."""
    if direction == "BUY":
        return zone.midpoint < current
    return zone.midpoint > current


def _path_position(zone: Zone, primary: Zone | None, current: float, direction: str) -> str:
    """Describe a refinement zone relative to the current->HTF POI path."""
    if primary is None:
        return "NO_PRIMARY"
    if zone_overlap_ratio(zone, primary) > 0:
        return "OVERLAPS_PRIMARY"
    if direction == "BUY" and primary.midpoint <= zone.midpoint < current:
        return "BETWEEN_CURRENT_AND_PRIMARY"
    if direction == "SELL" and current < zone.midpoint <= primary.midpoint:
        return "BETWEEN_CURRENT_AND_PRIMARY"
    return "OUTSIDE_PRIMARY_PATH"


def _primary_poi_rank(
    zone: Zone,
    direction: str,
    current: float,
    atr: float,
    fib: FibonacciContext | None,
) -> tuple[float, dict[str, Any]]:
    dist_atr = abs(zone.midpoint - current) / max(atr, EPS)
    dist_pct = distance_pct(current, zone.midpoint)
    fib_score, fib_details = fib_confluence_score(zone, fib, direction)
    age = safe_float(zone.details.get("age_candles"), 0.0)

    # Proximity is deliberately strong: a far historical OB cannot win merely
    # because its structural quality is high when a newer POI sits on the
    # current retracement path.
    proximity = clamp(100.0 - (dist_atr / max(H4_POI_MAX_DISTANCE_ATR, EPS)) * 100.0)
    recency = clamp(100.0 - (age / 60.0) * 100.0)
    freshness = 15.0 if zone.status == "ACTIVE" else -12.0 if zone.status == "MITIGATED" else -50.0
    fvg_bonus = 6.0 if "FVG" in zone.kind else 0.0
    fib_overlap_bonus = 5.0 if fib_details.get("band") in {"DEEP_0.618_0.786", "OVERLAPS_0.618"} else 0.0

    total = (
        safe_float(zone.strength, 50.0) * 0.40
        + fib_score * 0.20
        + proximity * 0.30
        + recency * 0.10
        + freshness
        + fvg_bonus
        + fib_overlap_bonus
    )
    return total, {
        "score": round(clamp(total), 2),
        "distance_atr": round(dist_atr, 3),
        "distance_pct": round(dist_pct, 4),
        "proximity_score": round(proximity, 2),
        "recency_score": round(recency, 2),
        "freshness_adjustment": round(freshness, 2),
        "fib": fib_details,
        "fvg_bonus": fvg_bonus,
        "fib_overlap_bonus": fib_overlap_bonus,
    }


def rank_primary_pois(
    ctx: dict[str, Any],
    direction: str,
    current: float,
    limit: int = MAX_PRIMARY_POI_CANDIDATES,
) -> list[tuple[Zone, dict[str, Any]]]:
    wanted = "BULLISH" if direction == "BUY" else "BEARISH"
    fib = ctx.get("fib", {}).get(direction)
    atr = safe_float(ctx.get("atr"), 0.0)
    ranked: list[tuple[float, Zone, dict[str, Any]]] = []

    for zone in ctx.get("zones", []):
        if wanted not in zone.kind or zone.status == "FAILED":
            continue
        if not _zone_is_retracement_side(zone, direction, current):
            continue
        dist_atr = abs(zone.midpoint - current) / max(atr, EPS)
        if dist_atr > H4_POI_MAX_DISTANCE_ATR:
            continue
        score, details = _primary_poi_rank(zone, direction, current, atr, fib)
        details.update({
            "available": True,
            "timeframe": ctx.get("timeframe"),
            "type": zone.kind,
            "low": round_price(zone.low),
            "high": round_price(zone.high),
            "midpoint": round_price(zone.midpoint),
            "quality": round(score, 2),
            "status": zone.status,
        })
        ranked.append((score, zone, details))

    ranked.sort(key=lambda item: (-item[0], abs(item[1].midpoint - current)))
    return [(zone, details) for _score, zone, details in ranked[:max(1, limit)]]


def select_primary_poi(
    ctx: dict[str, Any],
    direction: str,
    current: float,
) -> tuple[Zone | None, dict[str, Any]]:
    ranked = rank_primary_pois(ctx, direction, current, limit=1)
    if not ranked:
        return None, {"available": False, "reason": "Tidak ada H4 POI aktif yang berada pada sisi retracement dan cukup dekat."}
    return ranked[0]


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
        if wanted in z.kind
        and z.status != "FAILED"
        and _zone_is_retracement_side(z, direction, current)
    ]
    scored: list[tuple[float, Zone, dict[str, Any]]] = []
    for zone in zones:
        overlap = zone_overlap_ratio(zone, primary) if primary is not None else 0.0
        anchor = primary.midpoint if primary else current
        center_dist_atr = abs(zone.midpoint - anchor) / max(atr, EPS)
        relation = _path_position(zone, primary, current, direction)

        if primary is not None and relation not in {"OVERLAPS_PRIMARY", "BETWEEN_CURRENT_AND_PRIMARY"}:
            continue
        if primary is None and center_dist_atr > H1_POI_MAX_DISTANCE_ATR:
            continue

        fib_score, fib_details = fib_confluence_score(zone, fib, direction)
        quality = safe_float(zone.strength, 50.0)
        freshness = 12.0 if zone.status == "ACTIVE" else 0.0
        path_bonus = 25.0 if relation == "OVERLAPS_PRIMARY" else 20.0 if relation == "BETWEEN_CURRENT_AND_PRIMARY" else 0.0
        proximity = clamp(100.0 - (abs(zone.midpoint - current) / max(atr, EPS)) / max(H1_POI_MAX_DISTANCE_ATR, EPS) * 100.0)
        total = quality * 0.44 + fib_score * 0.20 + overlap * 18.0 + proximity * 0.18 + freshness + path_bonus
        scored.append((total, zone, {
            "score": round(clamp(total), 2),
            "overlap_with_h4": round(overlap, 3),
            "center_distance_atr": round(center_dist_atr, 3),
            "relation_to_h4": relation,
            "proximity_score": round(proximity, 2),
            "path_bonus": round(path_bonus, 2),
            "fib": fib_details,
            "freshness_bonus": freshness,
        }))

    if not scored:
        return None, {"available": False, "reason": "Tidak ada H1 refinement yang overlap atau berada pada retracement path menuju H4 POI."}
    scored.sort(key=lambda item: (-item[0], abs(item[1].midpoint - current)))
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
        if wanted in z.kind
        and z.status != "FAILED"
        and _zone_is_retracement_side(z, direction, current)
    ]
    scored: list[tuple[float, Zone, dict[str, Any]]] = []
    for zone in zones:
        overlap = zone_overlap_ratio(zone, parent_zone) if parent_zone is not None else 0.0
        relation = _path_position(zone, parent_zone, current, direction)
        center_dist_atr = abs(zone.midpoint - (parent_zone.midpoint if parent_zone else current)) / max(atr, EPS)
        if parent_zone is not None and relation not in {"OVERLAPS_PRIMARY", "BETWEEN_CURRENT_AND_PRIMARY"}:
            continue
        if parent_zone is None and center_dist_atr > M15_EXECUTION_OVERLAP_ATR:
            continue

        freshness = 16.0 if zone.status == "ACTIVE" else 0.0
        proximity = clamp(100.0 - (abs(zone.midpoint - current) / max(atr, EPS)) / max(M15_EXECUTION_OVERLAP_ATR, EPS) * 100.0)
        total = safe_float(zone.strength, 50.0) * 0.50 + overlap * 25.0 + proximity * 0.15 + freshness + (14.0 if relation == "OVERLAPS_PRIMARY" else 10.0 if relation == "BETWEEN_CURRENT_AND_PRIMARY" else 0.0)
        scored.append((total, zone, {
            "score": round(clamp(total), 2),
            "overlap_with_parent": round(overlap, 3),
            "center_distance_atr": round(center_dist_atr, 3),
            "relation_to_parent": relation,
            "proximity_score": round(proximity, 2),
            "freshness_bonus": freshness,
        }))

    if not scored:
        return None, {"available": False, "reason": "Tidak ada execution POI M15 yang berada pada parent/path dan di sisi entry."}
    scored.sort(key=lambda item: (-item[0], abs(item[1].midpoint - current)))
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


def entry_reachability(
    direction: str,
    current: float,
    entry: float,
    h4_atr: float,
) -> tuple[float, dict[str, Any]]:
    """Measure whether a pending entry is realistically reachable from current."""
    pct = distance_pct(current, entry)
    h4_dist = abs(current - entry) / max(h4_atr, EPS) if h4_atr > 0 else float("inf")
    pct_score = clamp(100.0 - (pct / MAX_ENTRY_DISTANCE_PCT) * 100.0)
    atr_score = clamp(100.0 - (h4_dist / MAX_ENTRY_DISTANCE_H4_ATR) * 100.0) if math.isfinite(h4_dist) else 0.0
    score = clamp(pct_score * 0.55 + atr_score * 0.45)
    allowed = pct <= MAX_ENTRY_DISTANCE_PCT and h4_dist <= MAX_ENTRY_DISTANCE_H4_ATR
    return score, {
        "allowed": allowed,
        "distance_pct": round(pct, 4),
        "distance_h4_atr": round(h4_dist, 3) if math.isfinite(h4_dist) else None,
        "max_distance_pct": MAX_ENTRY_DISTANCE_PCT,
        "max_distance_h4_atr": MAX_ENTRY_DISTANCE_H4_ATR,
        "score": round(score, 2),
    }


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


def htf_invalidation_level(
    direction: str,
    entry: float,
    zone: Zone,
    h1_structure: StructureSnapshot | None,
    h1_atr: float,
    sweep_level: float | None = None,
) -> float:
    """Invalidasi di sisi jauh zona HTF, diperluas ke swing H1 dan ekstrem sweep terdekat."""
    reach = h1_atr * SL_SWING_EXTEND_H1_ATR
    sweep_reach = h1_atr * SL_SWEEP_EXTEND_H1_ATR
    if direction == "BUY":
        level = zone.low if zone.low < entry else entry
        if h1_structure is not None:
            for pivot in reversed(h1_structure.swing_lows[-8:]):
                if pivot.price < level and level - pivot.price <= reach:
                    level = pivot.price
                    break
        if sweep_level and sweep_level < level and level - sweep_level <= sweep_reach:
            level = sweep_level
        return level
    level = zone.high if zone.high > entry else entry
    if h1_structure is not None:
        for pivot in reversed(h1_structure.swing_highs[-8:]):
            if pivot.price > level and pivot.price - level <= reach:
                level = pivot.price
                break
    if sweep_level and sweep_level > level and sweep_level - level <= sweep_reach:
        level = sweep_level
    return level


def enforce_risk_floor(candidate: Candidate, h1_atr: float) -> None:
    """Lebarkan SL ke risiko minimum agar tidak tersapu noise intraday."""
    min_risk = max(h1_atr * SL_MIN_H1_ATR, candidate.entry * SL_MIN_PCT / 100.0)
    if min_risk <= 0 or abs(candidate.entry - candidate.sl) >= min_risk:
        return
    if candidate.direction == "BUY":
        candidate.sl = round_price(candidate.entry - min_risk)
    else:
        candidate.sl = round_price(candidate.entry + min_risk)
    candidate.sl_reason += f" SL dilebarkan ke risiko minimum {SL_MIN_H1_ATR:.1f} ATR H1."
    candidate.evidence["risk_floor_applied"] = True


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
    leg: dict[str, Any] | None = None,
    target_pools: list[LiquidityPool] | None = None,
    sl_pools: list[LiquidityPool] | None = None,
) -> Candidate | None:
    h1_atr = safe_float(h1_ctx.get("atr"), 0.0)
    if h1_atr <= 0:
        h1_atr = m15_atr * 3.0
    h4_atr = safe_float(h4_ctx.get("atr"), h1_atr * 2.0)

    entry_zone = None
    entry_zone_tf = None
    reach_score = 0.0
    reach_details: dict[str, Any] = {}

    # Zona entry mengikuti HTF: H1 refinement dulu, lalu H4 primary.
    # M15 hanya mempersempit entry bila overlap dengan zona HTF tersebut.
    for zone, tf in ((refinement, "H1"), (primary, "H4")):
        if zone is None or not _zone_is_retracement_side(zone, direction, current):
            continue
        score, details = entry_reachability(direction, current, zone.midpoint, h4_atr)
        if not details["allowed"]:
            continue
        entry_zone = zone
        entry_zone_tf = tf
        reach_score = score
        reach_details = details
        break

    if entry_zone is None:
        return None

    entry = entry_zone.midpoint
    if (
        execution is not None
        and _zone_is_retracement_side(execution, direction, current)
        and zone_overlap_ratio(execution, entry_zone) > 0
    ):
        low = max(execution.low, entry_zone.low)
        high = min(execution.high, entry_zone.high)
        refined = (low + high) / 2.0
        if high > low and (
            (direction == "BUY" and refined < current)
            or (direction == "SELL" and refined > current)
        ):
            entry = refined
            entry_zone_tf = f"{entry_zone_tf}+M15"

    leg_used = False
    if leg:
        anchored = entry_zone.low - h1_atr <= leg["extreme"] <= entry_zone.high + h1_atr
        if anchored:
            if (direction == "BUY" and leg["entry"] < current) or (direction == "SELL" and leg["entry"] > current):
                leg_score, leg_details = entry_reachability(direction, current, leg["entry"], h4_atr)
                if not leg_details["allowed"]:
                    return None
                entry = leg["entry"]
                reach_score, reach_details = leg_score, leg_details
                leg_used = True
                entry_zone_tf = f"{entry_zone_tf}+LEG"
            else:
                # Retest sudah terlewat: jangan mengejar harga.
                return None

    structural = htf_invalidation_level(
        direction,
        entry,
        entry_zone,
        h1_ctx.get("structure"),
        h1_atr,
        safe_float(sweep.get("level"), 0.0) if sweep else None,
    )
    if leg_used:
        structural = min(structural, leg["extreme"]) if direction == "BUY" else max(structural, leg["extreme"])
    buffer = h1_atr * SL_BUFFER_H1_ATR
    sl = structural - buffer if direction == "BUY" else structural + buffer
    sl_pool_adjusted = None
    for guard in sl_pools or []:
        if direction == "BUY" and sl > guard.level >= sl - h1_atr * SL_POOL_REACH_H1_ATR:
            sl = guard.level - h1_atr * SL_POOL_BUFFER_H1_ATR
            sl_pool_adjusted = round_price(guard.level)
        elif direction == "SELL" and sl < guard.level <= sl + h1_atr * SL_POOL_REACH_H1_ATR:
            sl = guard.level + h1_atr * SL_POOL_BUFFER_H1_ATR
            sl_pool_adjusted = round_price(guard.level)
    min_risk = max(h1_atr * SL_MIN_H1_ATR, entry * SL_MIN_PCT / 100.0)
    risk = abs(entry - sl)
    if risk < min_risk:
        risk = min_risk
        sl = entry - risk if direction == "BUY" else entry + risk
    if risk > h1_atr * SL_MAX_H1_ATR or risk / max(entry, EPS) * 100.0 > SL_MAX_PCT:
        return None

    if target_pools is not None:
        picked = None
        for pool in target_pools:
            reward = (pool.level - entry) if direction == "BUY" else (entry - pool.level)
            if reward / risk >= MIN_PLANNED_RR:
                picked = pool
                break
        if picked is None:
            # Tanpa pool target yang layak, tidak ada setup (bukan TP di udara).
            return None
        target = picked
    tp = target.level if target else 0.0
    tp_from_target = target is not None
    # No artificial RR ceiling: TP remains the selected liquidity/structure target.
    if direction == "BUY":
        if tp <= entry:
            tp = entry + TP_FALLBACK_RR * risk
            tp_from_target = False
        if (tp - entry) / risk < MIN_PLANNED_RR:
            return None
        price_exp = current + max(
            EXP_H1_ATR_MULTIPLIER * h1_atr,
            abs(current - entry) * 1.20,
            (tp - current) * 0.45,
        )
        if tp > current:
            # Price Exp harus tetap di antara current dan TP, tetapi sedekat mungkin
            # ke TP agar thesis tidak kedaluwarsa hanya karena ekspansi normal kecil.
            price_exp = min(price_exp, current + (tp - current) * 0.95)
        if price_exp <= current:
            price_exp = current + max(h1_atr * 0.60, current * 0.001)
        if tp > current:
            price_exp = min(price_exp, current + (tp - current) * 0.95)
        if not (sl < entry < current < price_exp < tp):
            return None
    else:
        if tp >= entry or tp <= 0:
            tp = entry - TP_FALLBACK_RR * risk
            tp_from_target = False
        if tp <= 0 or (entry - tp) / risk < MIN_PLANNED_RR:
            return None
        price_exp = current - max(
            EXP_H1_ATR_MULTIPLIER * h1_atr,
            abs(current - entry) * 1.20,
            (current - tp) * 0.45,
        )
        if tp < current:
            price_exp = max(price_exp, current - (current - tp) * 0.95)
        if price_exp >= current:
            price_exp = current - max(h1_atr * 0.60, current * 0.001)
        if tp < current:
            price_exp = max(price_exp, current - (current - tp) * 0.95)
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
        "entry_zone_timeframe": entry_zone_tf,
        "entry_reachability": {"score": round(reach_score, 2), **reach_details},
        "risk_model": {
            "h1_atr": round_price(h1_atr),
            "risk": round_price(risk),
            "risk_h1_atr": round(risk / max(h1_atr, EPS), 2),
            "planned_rr": round(abs(tp - entry) / max(risk, EPS), 2),
        },
        "liquidity_model": {
            "leg_used": leg_used,
            "leg_kind": leg.get("kind") if (leg and leg_used) else None,
            "leg_extreme": round_price(leg["extreme"]) if (leg and leg_used) else None,
            "sl_pool_adjusted": sl_pool_adjusted,
            "target_pool": (
                f"{pool_timeframe(target)} {target.source} {round_price(target.level)}" if target else None
            ),
        },
        "notes_architecture": "H4 primary POI -> H1 entry zone + SL anchor -> M15 timing refinement",
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
        f"SL di luar invalidasi struktur H1/H4 ({round_price(structural)}) + buffer "
        f"{SL_BUFFER_H1_ATR:.2f} ATR H1; risiko minimum {SL_MIN_H1_ATR:.1f} ATR H1."
    )
    if target and tp_from_target:
        tp_reason = f"TP diarahkan ke {target.source} {round_price(target.level)} sebagai liquidity target HTF."
    else:
        tp_reason = (
            f"TP fallback {round_price(tp)} menggunakan RR minimum dari risiko "
            "karena target liquidity HTF tidak tersedia."
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
            tol = max(pool.level * 0.0004, atr * 0.10)
            if pool.kind == "SELL_SIDE":
                swept = (
                    c.low < pool.level - tol
                    and c.close > pool.level
                    and (c.close - c.low) / max(c.range, EPS) >= 0.40
                )
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
                swept = (
                    c.high > pool.level + tol
                    and c.close < pool.level
                    and (c.high - c.close) / max(c.range, EPS) >= 0.40
                )
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
            "detail": "BTCUSDT H4 direkam sebagai konteks numerik; bukan filter arah.",
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
        expected_sweep = "BULLISH" if direction == DIRECTION_BUY else "BEARISH"
        if sweep.get("direction") != expected_sweep:
            continue
        s_idx = int(sweep["index"])
        for event in reversed(structure.events):
            expected_break = "BULLISH" if direction == DIRECTION_BUY else "BEARISH"
            if event.get("type") != "MSS" or event.get("direction") != expected_break:
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
        # Price Exp sits near the forward target to avoid premature expiry.
        price_exp = current + max(atr * EXP_ATR_MULTIPLIER, abs(current - entry) * 1.35)
        price_exp = max(price_exp, current + min(atr * 0.50, max(tp - current, atr * 0.50) * 0.90))
        if tp > current:
            price_exp = min(price_exp, current + (tp - current) * 0.95)

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
        price_exp = current - max(atr * EXP_ATR_MULTIPLIER, abs(current - entry) * 1.35)
        price_exp = min(price_exp, current - min(atr * 0.50, max(current - tp, atr * 0.50) * 0.90))
        if tp < current:
            price_exp = max(price_exp, current - (current - tp) * 0.95)

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
    if rr + 1e-9 < MIN_PLANNED_RR:
        return None

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
    # RR ekstrem tinggi biasanya tanda SL terlalu ketat, bukan setup bagus.
    if rr >= 8:
        return 40.0
    if rr >= 5:
        return 70.0
    if rr >= 2:
        return 100.0
    if rr >= 1.5:
        return 78.0
    if rr >= 1.0:
        return 40.0
    return 15.0


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


def htf_alignment_score(
    direction: str,
    regime: dict[str, Any] | None,
    fallback_trend: str,
) -> float:
    """Kesepakatan D1/H4/H1/M15 dengan arah setup (continuation > pullback > range)."""
    want = "BULLISH" if direction == "BUY" else "BEARISH"
    frames = (regime or {}).get("timeframes") or {}
    weights = {"D1": 0.20, "H4": 0.40, "H1": 0.25, "M15": 0.15}
    total = 0.0
    used = 0.0
    for name, weight in weights.items():
        item = frames.get(name)
        if not isinstance(item, dict):
            continue
        state = str(item.get("regime") or "")
        trend = str(item.get("trend") or "")
        if state.startswith(want):
            score = 100.0 if state.endswith("CONTINUATION") else 85.0
        elif state in ("RANGE", "TRANSITION", ""):
            score = 50.0
        elif trend == want:
            score = 60.0
        else:
            score = 10.0
        total += weight * score
        used += weight
    if used <= 0:
        return structure_direction_score(direction, fallback_trend)
    return clamp(total / used)


def risk_quality_score(risk_h1_atr: float) -> float:
    """Jarak SL ideal 1.5-3 ATR H1; terlalu ketat atau terlalu lebar dihukum."""
    if risk_h1_atr <= 0:
        return 40.0
    if 1.5 <= risk_h1_atr <= 3.0:
        return 100.0
    if risk_h1_atr < 1.5:
        return clamp(100.0 - (1.5 - risk_h1_atr) * 100.0)
    if risk_h1_atr <= 4.0:
        return 100.0 - (risk_h1_atr - 3.0) * 25.0
    if risk_h1_atr <= 5.0:
        return 75.0 - (risk_h1_atr - 4.0) * 35.0
    return 20.0


def rsi_pivot_divergence(
    candles: list[Candle],
    rsi_values: list[float],
    pivots: list[Pivot],
    direction: str,
    max_age: int,
) -> dict[str, Any] | None:
    """Divergence RSI vs dua pivot terakhir. BUY: swing low, SELL: swing high."""
    if len(pivots) < 2 or not candles:
        return None
    last = len(candles) - 1
    p1, p2 = pivots[-2], pivots[-1]
    if last - p2.index > max_age or p1.index < RSI_PERIOD:
        return None
    if p2.index >= len(rsi_values) or p1.index >= len(rsi_values):
        return None
    r1, r2 = rsi_values[p1.index], rsi_values[p2.index]
    move_pct = (p2.price - p1.price) / max(abs(p1.price), EPS) * 100.0
    if direction == "BUY":
        if move_pct <= -DIV_MIN_PRICE_PCT and r2 >= r1 + DIV_MIN_RSI_DIFF:
            kind = "REGULAR"
        elif move_pct >= DIV_MIN_PRICE_PCT and r2 <= r1 - DIV_MIN_RSI_DIFF:
            kind = "HIDDEN"
        else:
            return None
    else:
        if move_pct >= DIV_MIN_PRICE_PCT and r2 <= r1 - DIV_MIN_RSI_DIFF:
            kind = "REGULAR"
        elif move_pct <= -DIV_MIN_PRICE_PCT and r2 >= r1 + DIV_MIN_RSI_DIFF:
            kind = "HIDDEN"
        else:
            return None
    return {
        "kind": kind,
        "strength": round(clamp(abs(r2 - r1) * 5.0), 1),
        "rsi_prev": round(r1, 1),
        "rsi_last": round(r2, 1),
        "pivot_age": last - p2.index,
    }


def build_rsi_context(
    candles: list[Candle],
    structure: StructureSnapshot,
    tf: str,
) -> dict[str, Any]:
    """Ringkasan RSI satu timeframe: regime range, kondisi 12 bar, dan divergence."""
    values = rsi_series(candles, RSI_PERIOD)
    empty = {
        "valid": False,
        "tf": tf,
        "cur": 50.0,
        "slope": 0.0,
        "regime": "NEUTRAL",
        "min12": 50.0,
        "max12": 50.0,
        "div": {"BUY": None, "SELL": None},
    }
    if len(values) < RSI_PERIOD + 10:
        return empty
    tail = values[-RSI_REGIME_WINDOW:]
    avg = sum(tail) / len(tail)
    lo, hi = min(tail), max(tail)
    if avg >= 52.0 and lo >= 33.0:
        regime = "BULL_RANGE"
    elif avg <= 48.0 and hi <= 67.0:
        regime = "BEAR_RANGE"
    else:
        regime = "NEUTRAL"
    last12 = values[-12:]
    max_age = DIV_MAX_AGE.get(tf, 24)
    return {
        "valid": True,
        "tf": tf,
        "cur": round(values[-1], 2),
        "slope": round(linear_slope(values, 8), 3),
        "avg": round(avg, 1),
        "lo": round(lo, 1),
        "hi": round(hi, 1),
        "regime": regime,
        "min12": round(min(last12), 1),
        "max12": round(max(last12), 1),
        "div": {
            "BUY": rsi_pivot_divergence(candles, values, structure.swing_lows, "BUY", max_age),
            "SELL": rsi_pivot_divergence(candles, values, structure.swing_highs, "SELL", max_age),
        },
    }


def simple_rsi_state(candles: list[Candle]) -> dict[str, Any]:
    values = rsi_series(candles, RSI_PERIOD)
    if len(values) < RSI_PERIOD + 10:
        return {"valid": False, "cur": 50.0, "slope": 0.0}
    return {"valid": True, "cur": round(values[-1], 2), "slope": round(linear_slope(values, 8), 3)}


def evaluate_momentum(
    direction: str,
    rsi_pack: dict[str, Any] | None,
    has_sweep: bool,
) -> dict[str, Any]:
    """Regime RSI -> archetype -> veto. Hasilnya menjadi pengali confidence."""
    if not rsi_pack:
        return {"score": 50.0, "archetype": "N/A", "waiting": False, "vetoes": [], "tags": []}
    buy = direction == "BUY"
    h4r = rsi_pack.get("H4") or {}
    h1r = rsi_pack.get("H1") or {}
    m15r = rsi_pack.get("M15") or {}
    btcr = rsi_pack.get("BTC") or {}
    good = "BULL_RANGE" if buy else "BEAR_RANGE"
    bad = "BEAR_RANGE" if buy else "BULL_RANGE"
    side = "bullish" if buy else "bearish"
    score = 50.0
    tags: list[str] = []
    vetoes: list[str] = []

    divs = []
    for tf, ctx in (("H4", h4r), ("H1", h1r), ("M15", m15r)):
        item = (ctx.get("div") or {}).get(direction)
        if item:
            divs.append((tf, item))
    reg_div = next(((tf, d) for tf, d in divs if d["kind"] == "REGULAR"), None)
    hid_div = next(((tf, d) for tf, d in divs if d["kind"] == "HIDDEN"), None)

    # 1) Regime RSI H4 (bobot besar) dan H1.
    for name, ctx, up, down in (("H4", h4r, 15.0, 22.0), ("H1", h1r, 10.0, 10.0)):
        if not ctx.get("valid"):
            continue
        regime = ctx.get("regime")
        if regime == good:
            score += up
            tags.append(f"RSI {name} regime {side}")
        elif regime == bad:
            if reg_div:
                score -= 5.0
                tags.append(f"RSI {name} melawan arah, tertahan divergence reguler")
            else:
                score -= down
                tags.append(f"RSI {name} regime melawan arah")
                if name == "H4":
                    vetoes.append("RSI H4 melawan arah tanpa divergence")

    # 2) Koreksi RSI H1 ke value zone lalu berbalik (trend pullback).
    pullback = False
    if h1r.get("valid"):
        cur = h1r["cur"]
        slope = h1r["slope"]
        if buy:
            dipped = RSI_PULLBACK_FLOOR <= h1r["min12"] <= 48.0
            broken = h1r["min12"] < RSI_PULLBACK_FLOOR
            turning = cur >= h1r["min12"] + 3.0 and slope > 0
            pressing = slope <= -0.8
            extreme = h1r["min12"]
        else:
            dipped = 52.0 <= h1r["max12"] <= RSI_PULLBACK_CEIL_SELL
            broken = h1r["max12"] > RSI_PULLBACK_CEIL_SELL
            turning = cur <= h1r["max12"] - 3.0 and slope < 0
            pressing = slope >= 0.8
            extreme = h1r["max12"]
        pullback = dipped and turning
        if pullback:
            score += 20.0
            tags.append(f"RSI H1 koreksi ke {extreme:.0f} lalu berbalik ({cur:.0f})")
        elif pressing and not reg_div:
            score -= 15.0
            tags.append("RSI H1 masih menekan melawan arah")
        elif broken and not reg_div:
            score -= 12.0
            tags.append("RSI H1 jatuh ke zona breakdown, bukan koreksi sehat")

    # 3) Divergence.
    if reg_div:
        tf, d = reg_div
        score += 15.0 + 10.0 * d["strength"] / 100.0
        tags.append(f"divergence reguler RSI {tf} ({d['rsi_prev']:.0f}->{d['rsi_last']:.0f})")
    if hid_div:
        tf, d = hid_div
        score += 12.0 + 8.0 * d["strength"] / 100.0
        tags.append(f"hidden divergence RSI {tf} ({d['rsi_prev']:.0f}->{d['rsi_last']:.0f})")

    # 4) Timing M15.
    if m15r.get("valid"):
        m_slope = m15r["slope"]
        if (buy and m_slope > 0) or (not buy and m_slope < 0):
            score += 5.0
        elif abs(m_slope) >= 1.0:
            score -= 5.0

    # 5) Overextension: jangan mengejar harga.
    h1_cur = h1r.get("cur", 50.0)
    m15_cur = m15r.get("cur", 50.0)
    if (buy and (h1_cur >= OVEREXT_H1_BUY or m15_cur >= OVEREXT_M15_BUY)) or (
        not buy and (h1_cur <= OVEREXT_H1_SELL or m15_cur <= OVEREXT_M15_SELL)
    ):
        score -= 20.0
        vetoes.append("RSI overextended (mengejar harga)")
        tags.append(f"RSI terlalu ekstrem (H1 {h1_cur:.0f}, M15 {m15_cur:.0f})")

    # 6) Momentum BTC H1 (alt berkorelasi).
    if btcr.get("valid"):
        weak = btcr["cur"] < 45.0 and btcr["slope"] < 0
        strong = btcr["cur"] > 55.0 and btcr["slope"] > 0
        if (buy and weak) or (not buy and strong):
            score -= 15.0
            tags.append("momentum RSI BTC H1 melawan arah")
        elif (buy and strong) or (not buy and weak):
            score += 5.0

    if reg_div:
        archetype = "SWEEP_DIVERGENCE_REVERSAL" if has_sweep else "DIVERGENCE_REVERSAL"
    elif pullback or hid_div:
        archetype = "TREND_PULLBACK_RSI"
    else:
        archetype = "UNCONFIRMED"
    return {
        "score": round(clamp(score), 1),
        "archetype": archetype,
        "waiting": archetype == "UNCONFIRMED",
        "vetoes": vetoes,
        "tags": tags,
    }


def momentum_note(momentum: dict[str, Any]) -> str:
    archetype = momentum.get("archetype")
    if archetype in (None, "N/A"):
        return ""
    head = (
        "Belum ada trigger RSI (butuh koreksi ke value zone atau divergence)"
        if archetype == "UNCONFIRMED"
        else f"Kerangka RSI {archetype}"
    )
    body = "; ".join(momentum.get("tags") or [])
    note = f"{head}: {body}." if body else f"{head}."
    vetoes = momentum.get("vetoes") or []
    if vetoes:
        note += " Veto: " + "; ".join(vetoes) + "."
    return note


def momentum_factor(score: float) -> float:
    return MOMENTUM_FACTOR_FLOOR + (1.0 - MOMENTUM_FACTOR_FLOOR) * clamp(score) / 100.0


def combine_confidence(scores: dict[str, float], evidence: dict[str, Any]) -> float:
    """Skor struktur x pengali momentum RSI, lalu penalti dan batas gate."""
    total_weight = sum(WEIGHTS.values())
    base = sum(WEIGHTS[name] * scores.get(name, 50.0) for name in WEIGHTS) / total_weight
    gates = evidence.get("gates") or {}
    momentum_score = safe_float((evidence.get("momentum") or {}).get("score"), 50.0) + safe_float(
        gates.get("score_adj"), 0.0
    )
    confidence = base * momentum_factor(momentum_score)
    confidence -= safe_float(evidence.get("confidence_penalty"), 0.0)
    if gates.get("waiting"):
        confidence = min(confidence, UNCONFIRMED_CONFIDENCE_CAP)
    if gates.get("vetoes"):
        confidence = min(confidence, VETO_CONFIDENCE_CAP)
    return round(clamp(confidence), 2)


def pool_timeframe(pool: LiquidityPool) -> str:
    return str(pool.details.get("timeframe") or "M15").split("_")[0]


def pool_significance(pool: LiquidityPool) -> float:
    tf_bonus = {"H4": 20.0, "H1": 10.0}.get(pool_timeframe(pool), 0.0)
    source = str(pool.source)
    src_bonus = 10.0 if source == "EQUAL_LEVELS" else 5.0 if source.startswith("RECENT_RANGE") else 0.0
    return clamp(pool.strength + tf_bonus + src_bonus)


def pool_consumed(pool: LiquidityPool, candles: list[Candle], atr: float) -> bool:
    """Pool sudah diambil jika harga menembusnya setelah terbentuk."""
    tol = max(pool.level * 0.0002, atr * 0.05)
    for c in candles[max(pool.index + 1, 0):]:
        if pool.kind == "SELL_SIDE" and c.low < pool.level - tol:
            return True
        if pool.kind == "BUY_SIDE" and c.high > pool.level + tol:
            return True
    return False


def live_pools(
    pools: list[LiquidityPool],
    candles_by_tf: dict[str, list[Candle]],
    atr_by_tf: dict[str, float],
) -> list[LiquidityPool]:
    live: list[LiquidityPool] = []
    for pool in pools:
        tf = pool_timeframe(pool)
        candles = candles_by_tf.get(tf)
        if candles and pool_consumed(pool, candles, atr_by_tf.get(tf, 0.0)):
            continue
        live.append(pool)
    return live


def nearest_magnet(
    direction: str,
    live: list[LiquidityPool],
    price: float,
    h1_atr: float,
) -> dict[str, Any] | None:
    """Pool signifikan terdekat di sisi lawan yang belum disapu (harga cenderung menujunya dulu)."""
    want = "SELL_SIDE" if direction == "BUY" else "BUY_SIDE"
    best: dict[str, Any] | None = None
    for pool in live:
        if pool.kind != want or pool_significance(pool) < SIGNIFICANT_POOL_MIN:
            continue
        dist = (price - pool.level) if direction == "BUY" else (pool.level - price)
        if dist <= 0:
            continue
        d_atr = dist / max(h1_atr, EPS)
        if d_atr > MAGNET_MAX_H1_ATR:
            continue
        if best is None or d_atr < best["d_atr"]:
            best = {
                "level": round_price(pool.level),
                "d_atr": round(d_atr, 2),
                "tf": pool_timeframe(pool),
                "source": pool.source,
            }
    return best


def ranked_targets(
    live: list[LiquidityPool],
    direction: str,
    ref_entry: float,
    h1_atr: float,
) -> list[LiquidityPool]:
    """Pool target signifikan yang belum disapu, dari yang terdekat."""
    want = "BUY_SIDE" if direction == "BUY" else "SELL_SIDE"
    ranked: list[tuple[float, LiquidityPool]] = []
    for pool in live:
        if pool.kind != want or pool_significance(pool) < SIGNIFICANT_POOL_MIN:
            continue
        dist = (pool.level - ref_entry) if direction == "BUY" else (ref_entry - pool.level)
        if dist <= 0 or dist / max(h1_atr, EPS) > 8.0:
            continue
        ranked.append((dist, pool))
    ranked.sort(key=lambda item: item[0])
    return [pool for _dist, pool in ranked]


def sl_guard_pools(
    live: list[LiquidityPool],
    direction: str,
    entry: float,
) -> list[LiquidityPool]:
    """Pool signifikan di sisi stop yang berpotensi diburu."""
    want = "SELL_SIDE" if direction == "BUY" else "BUY_SIDE"
    return [
        pool
        for pool in live
        if pool.kind == want
        and pool_significance(pool) >= SIGNIFICANT_POOL_MIN
        and ((pool.level < entry) if direction == "BUY" else (pool.level > entry))
    ]


def sweep_mss_leg(
    direction: str,
    sweep: dict[str, Any] | None,
    mss: dict[str, Any] | None,
    m15: list[Candle],
    fvg: list[Zone],
) -> dict[str, Any] | None:
    """Leg sweep -> MSS dan zona retest: FVG di dalam OTE, atau OTE 0.62-0.79."""
    if not sweep or not mss:
        return None
    try:
        s_idx = int(sweep["index"])
        e_idx = int(mss["index"])
    except (KeyError, TypeError, ValueError):
        return None
    if e_idx <= s_idx or e_idx >= len(m15):
        return None
    buy = direction == "BUY"
    seg = m15[s_idx : e_idx + 1]
    if buy:
        extreme = min(c.low for c in seg)
        peak = max(c.high for c in seg)
    else:
        extreme = max(c.high for c in seg)
        peak = min(c.low for c in seg)
    span = abs(peak - extreme)
    if span <= 0 or span / max(abs(extreme), EPS) * 100.0 < 0.3:
        return None
    if buy:
        band_hi = peak - OTE_LOW * span
        band_lo = peak - OTE_HIGH * span
    else:
        band_lo = peak + OTE_LOW * span
        band_hi = peak + OTE_HIGH * span
    wanted = "BULLISH_FVG" if buy else "BEARISH_FVG"
    zones = [
        z
        for z in fvg
        if z.kind == wanted and s_idx <= z.index <= e_idx + 1 and z.high >= band_lo and z.low <= band_hi
    ]
    if zones:
        zone = max(zones, key=lambda item: item.strength)
        entry = (max(zone.low, band_lo) + min(zone.high, band_hi)) / 2.0
        kind = "FVG_OTE"
    else:
        entry = (band_lo + band_hi) / 2.0
        kind = "OTE"
    return {
        "entry": entry,
        "kind": kind,
        "extreme": extreme,
        "peak": peak,
        "ote": [round_price(band_lo), round_price(band_hi)],
        "sweep_idx": s_idx,
        "mss_idx": e_idx,
    }


def ema_values(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(value * k + out[-1] * (1.0 - k))
    return out


def market_quality_pack(
    m15: list[Candle],
    h1: list[Candle],
    pair_h4: list[Candle],
) -> dict[str, Any]:
    """Konteks pasar: guncangan 24j/7h, jarak dari ekstrem, tumpukan EMA, dan partisipasi volume."""
    if len(m15) < 100 or len(h1) < 50:
        return {"valid": False}
    now = m15[-1].close
    win24 = m15[-96:]
    hi24 = max(c.high for c in win24)
    lo24 = min(c.low for c in win24)
    win48 = h1[-48:]
    hi48 = max(c.high for c in win48)
    lo48 = min(c.low for c in win48)
    chg7d = 0.0
    if len(pair_h4) > 43:
        base = pair_h4[-43].close
        chg7d = (pair_h4[-1].close - base) / max(abs(base), EPS) * 100.0
    c15 = [c.close for c in m15]
    c1 = [c.close for c in h1]
    e7, e25, e50 = (ema_values(c15, n)[-1] for n in (7, 25, 50))
    h1_e25 = ema_values(c1, 25)[-1]
    vol_recent = sum(c.volume for c in m15[-8:]) / 8.0
    vol_base = sum(c.volume for c in win24) / max(len(win24), 1)
    return {
        "valid": True,
        "chg24": round((now - m15[-97].close) / max(abs(m15[-97].close), EPS) * 100.0, 2),
        "drop24": round((hi24 - now) / max(hi24, EPS) * 100.0, 2),
        "rise24": round((now - lo24) / max(lo24, EPS) * 100.0, 2),
        "drop48": round((hi48 - now) / max(hi48, EPS) * 100.0, 2),
        "rise48": round((now - lo48) / max(lo48, EPS) * 100.0, 2),
        "chg7d": round(chg7d, 2),
        "bear15": bool(e7 < e25 < e50 and now < e25),
        "bull15": bool(e7 > e25 > e50 and now > e25),
        "h1_below_ema25": bool(c1[-1] < h1_e25),
        "h1_above_ema25": bool(c1[-1] > h1_e25),
        "vol_ratio": round(vol_recent / max(vol_base, EPS), 2),
    }


def trigger_info(
    m15: list[Candle],
    sweep: dict[str, Any] | None,
    mss: dict[str, Any] | None,
    bos: dict[str, Any] | None,
    magnet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    age = 999
    if mss:
        try:
            age = len(m15) - 1 - int(mss.get("index"))
        except (TypeError, ValueError):
            age = 999
    return {"sweep": bool(sweep), "mss": bool(mss), "mss_age": age, "bos": bool(bos), "magnet": magnet}


def evaluate_market_gates(
    direction: str,
    quality: dict[str, Any] | None,
    trigger: dict[str, Any] | None,
    entry: float,
    sl: float,
) -> dict[str, Any]:
    """Veto dan status menunggu berdasarkan konteks pasar dan konfirmasi struktur M15."""
    buy = direction == "BUY"
    vetoes: list[str] = []
    notes: list[str] = []
    adj = 0.0
    trig = trigger or {}
    mss_ok = bool(trig.get("mss")) and int(trig.get("mss_age", 999)) <= TRIGGER_MAX_AGE_M15
    bos_ok = bool(trig.get("bos"))
    waiting = not (mss_ok or bos_ok)
    risk_pct = abs(entry - sl) / max(abs(entry), EPS) * 100.0
    if risk_pct > SL_MAX_PCT:
        vetoes.append(f"SL terlalu lebar ({risk_pct:.1f}% > {SL_MAX_PCT:.1f}% harga)")
    if waiting:
        notes.append("belum ada konfirmasi struktur M15 (butuh MSS atau BOS)")
    magnet = trig.get("magnet")
    if magnet:
        lvl = safe_float(magnet.get("level"))
        if (buy and entry > lvl) or ((not buy) and entry < lvl):
            vetoes.append(
                f"Entry terlalu cepat: likuiditas {'sell' if buy else 'buy'}-side {magnet.get('tf')} "
                f"{lvl} belum disapu ({magnet.get('d_atr')} ATR H1 dari harga)"
            )
    q = quality or {}
    if not q.get("valid"):
        return {"vetoes": vetoes, "waiting": waiting, "notes": notes, "score_adj": adj}

    adverse24 = q["drop24"] if buy else q["rise24"]
    adverse48 = q["drop48"] if buy else q["rise48"]
    word = "turun" if buy else "naik"
    if adverse24 >= KNIFE_MOVE_24H_PCT and not mss_ok:
        vetoes.append(f"Pisau jatuh: harga {word} {adverse24:.0f}% dari ekstrem 24j tanpa MSS M15")
    if buy and q["chg7d"] >= POST_PUMP_7D_PCT and adverse48 >= POST_PUMP_DROP_PCT and not (
        mss_ok and q["h1_above_ema25"]
    ):
        vetoes.append(
            f"Pasca pompa: +{q['chg7d']:.0f}% dalam 7 hari lalu turun {adverse48:.0f}% dari high 48j"
        )
    if (not buy) and q["chg7d"] <= -POST_PUMP_7D_PCT and adverse48 >= POST_PUMP_DROP_PCT and not (
        mss_ok and q["h1_below_ema25"]
    ):
        vetoes.append(
            f"Pasca dump: {q['chg7d']:.0f}% dalam 7 hari lalu naik {adverse48:.0f}% dari low 48j"
        )
    against15 = q["bear15"] if buy else q["bull15"]
    against1 = q["h1_below_ema25"] if buy else q["h1_above_ema25"]
    if against15 and against1:
        if mss_ok:
            adj -= 5.0
            notes.append("EMA M15/H1 masih melawan arah, tetapi MSS M15 sudah terkonfirmasi")
        else:
            vetoes.append("EMA 7/25/50 M15 dan H1 melawan arah tanpa MSS M15")
    if q["vol_ratio"] < DRY_VOLUME_RATIO:
        adj -= 8.0
        notes.append(f"partisipasi volume rendah ({q['vol_ratio']:.2f}x rata-rata 24j)")
    return {"vetoes": vetoes, "waiting": waiting, "notes": notes, "score_adj": adj}


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
    pair_regime: dict[str, Any] | None = None,
    momentum: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    trigger: dict[str, Any] | None = None,
) -> Candidate:
    reach = candidate.evidence.get("entry_reachability") or {}
    reach_score = safe_float(reach.get("score"), 50.0)
    scores = {
        "htf_alignment": htf_alignment_score(candidate.direction, pair_regime, pair_h4_trend),
        "h4_poi": clamp(h4_poi_score if h4_poi_score is not None else 50.0),
        "h4_fibonacci": clamp(h4_fib_score if h4_fib_score is not None else 50.0),
        "h1_refinement": clamp(h1_refinement_score if h1_refinement_score is not None else location_score(candidate.direction, candidate.entry, h1_dr)),
        "liquidity": liquidity_score(candidate, target, sweep, current, m15_atr),
        "smc_trigger": smc_trigger_score(candidate),
        "displacement": displacement_score_for_candidate(candidate, latest_displacement, vlt["relative_volume"]),
        "vlt": _vlt_direction_score(candidate.direction, vlt),
        "planned_rr": candidate_rr_score(candidate),
        "entry_reachability": reach_score,
        "risk_quality": risk_quality_score(
            safe_float((candidate.evidence.get("risk_model") or {}).get("risk_h1_atr"), 0.0)
        ),
    }

    if momentum is not None:
        candidate.evidence["momentum"] = momentum
    mom = candidate.evidence.get("momentum") or {}
    vetoes = list(mom.get("vetoes") or [])
    ratio = location_ratio(candidate.entry, h1_dr)
    if candidate.direction == "BUY" and ratio > EQUILIBRIUM_BUY_MAX:
        vetoes.append(f"BUY di zona premium H1 (lokasi {ratio:.2f})")
    elif candidate.direction == "SELL" and ratio < EQUILIBRIUM_SELL_MIN:
        vetoes.append(f"SELL di zona discount H1 (lokasi {ratio:.2f})")
    gate = evaluate_market_gates(candidate.direction, quality, trigger, candidate.entry, candidate.sl)
    vetoes.extend(gate["vetoes"])
    lm = candidate.evidence.get("liquidity_model") or {}
    if lm.get("leg_used"):
        gate["notes"].append(
            f"entry di retest leg sweep->MSS ({lm.get('leg_kind')}, ekstrem sweep {lm.get('leg_extreme')})"
        )
    if lm.get("sl_pool_adjusted"):
        gate["notes"].append(f"SL digeser ke luar pool likuiditas {lm.get('sl_pool_adjusted')}")
    if lm.get("target_pool"):
        gate["notes"].append(f"TP ke pool {lm.get('target_pool')}")
    candidate.evidence["gates"] = {
        "vetoes": vetoes,
        "waiting": bool(mom.get("waiting")) or bool(gate["waiting"]),
        "archetype": mom.get("archetype"),
        "notes": gate["notes"],
        "score_adj": gate["score_adj"],
    }
    note = momentum_note({**mom, "vetoes": vetoes}) if mom else ""
    if gate["notes"]:
        note = f"{note} Gerbang pasar: " + "; ".join(gate["notes"]) + "."
    if (
        note
        and "Kerangka RSI" not in candidate.entry_reason
        and "trigger RSI" not in candidate.entry_reason
        and "Gerbang pasar" not in candidate.entry_reason
    ):
        candidate.entry_reason = f"{candidate.entry_reason} {note}".strip()

    scores["momentum"] = safe_float(mom.get("score"), 50.0)
    candidate.scores = {k: round(v, 2) for k, v in scores.items()}
    candidate.confidence = combine_confidence(scores, candidate.evidence)
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
        tp = max(tp, current + 1.25 * atr, entry + 2.0 * abs(entry - sl))
        price_exp = current + max(EXP_ATR_MULTIPLIER * atr, (tp - current) * 0.90)
        price_exp = min(price_exp, current + (tp - current) * 0.95) if tp > current else current + EXP_ATR_MULTIPLIER * atr
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
    tp = min(tp, current - 1.25 * atr, entry - 2.0 * abs(entry - sl))
    price_exp = current - max(EXP_ATR_MULTIPLIER * atr, (current - tp) * 0.90)
    price_exp = max(price_exp, current - (current - tp) * 0.95) if tp < current else current - EXP_ATR_MULTIPLIER * atr
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


def _structure_direction_evidence(candles: list[Candle], structure: StructureSnapshot) -> dict[str, Any]:
    """Directional evidence that resists flips from one minor/internal break."""
    if not candles:
        return {"bullish_score": 0.0, "bearish_score": 0.0, "state": "RANGE", "reason": "NO_DATA"}
    atr_values = atr_series(candles, 14)
    atr = max(median_or(atr_values[-20:], candles[-1].range), EPS)
    highs = structure.swing_highs[-6:]
    lows = structure.swing_lows[-6:]
    bull = 0.0
    bear = 0.0
    reasons: list[str] = []
    if len(highs) >= 2:
        dh = highs[-1].price - highs[-2].price
        if dh > 0.35 * atr:
            bull += 30.0; reasons.append("HH_SIGNIFICANT")
        elif dh < -0.35 * atr:
            bear += 30.0; reasons.append("LH_SIGNIFICANT")
    if len(lows) >= 2:
        dl = lows[-1].price - lows[-2].price
        if dl > 0.35 * atr:
            bull += 30.0; reasons.append("HL_SIGNIFICANT")
        elif dl < -0.35 * atr:
            bear += 30.0; reasons.append("LL_SIGNIFICANT")
    recent_labels = [x.get("label") for x in structure.labels[-10:]]
    bull += min(16.0, (recent_labels.count("HH") + recent_labels.count("HL")) * 4.0)
    bear += min(16.0, (recent_labels.count("LH") + recent_labels.count("LL")) * 4.0)
    recent_events = structure.events[-8:]
    latest_bull = next((e for e in reversed(recent_events) if e.get("direction") == "BULLISH" and e.get("type") in {"BOS", "MSS"}), None)
    latest_bear = next((e for e in reversed(recent_events) if e.get("direction") == "BEARISH" and e.get("type") in {"BOS", "MSS"}), None)
    if latest_bull: bull += 18.0 if latest_bull.get("type") == "MSS" else 12.0
    if latest_bear: bear += 18.0 if latest_bear.get("type") == "MSS" else 12.0
    closes = [c.close for c in candles]
    slope = linear_slope(closes[-20:], min(20, len(closes))) if len(closes) >= 5 else 0.0
    normalized_slope = slope / max(atr, EPS)
    if normalized_slope > 0.08: bull += 10.0
    elif normalized_slope < -0.08: bear += 10.0
    gap = abs(bull - bear)
    if bull >= 54 and gap >= 14 and bull > bear: regime = "BULLISH"
    elif bear >= 54 and gap >= 14 and bear > bull: regime = "BEARISH"
    elif max(bull, bear) < 50 or gap < 10: regime = "RANGE"
    else: regime = "TRANSITION"
    state = regime
    if regime == "BULLISH" and normalized_slope < -0.05: state = "BULLISH_PULLBACK"
    elif regime == "BULLISH" and normalized_slope >= 0.05: state = "BULLISH_CONTINUATION"
    elif regime == "BEARISH" and normalized_slope > 0.05: state = "BEARISH_PULLBACK"
    elif regime == "BEARISH" and normalized_slope <= -0.05: state = "BEARISH_CONTINUATION"
    return {"bullish_score": round(clamp(bull),2), "bearish_score": round(clamp(bear),2), "gap": round(gap,2), "raw_structure": structure.trend, "state": state, "dominant": "BULLISH" if bull > bear else "BEARISH" if bear > bull else "RANGE", "normalized_slope": round(normalized_slope,6), "reasons": reasons}


def _timeframe_regime_report(candles: list[Candle], timeframe: str, span: int) -> dict[str, Any]:
    structure = build_structure(candles, span, min_swing_atr=_swing_min_atr(timeframe))
    evidence = _structure_direction_evidence(candles, structure)
    strength = trend_strength_metrics(candles, structure)
    atr_values = atr_series(candles, 14)
    trend = "BULLISH" if evidence["bullish_score"] > evidence["bearish_score"] else "BEARISH" if evidence["bearish_score"] > evidence["bullish_score"] else "RANGE"
    return {"timeframe": timeframe, "trend": trend, "regime": evidence["state"], "state": evidence["state"], "bullish_score": evidence["bullish_score"], "bearish_score": evidence["bearish_score"], "evidence": evidence, "structure": _structure_summary(structure, candles), "trend_strength": strength, "atr": round(atr_values[-1],10) if atr_values else None, "candles": len(candles)}


def build_multi_timeframe_regime(timeframes: dict[str, tuple[list[Candle], int]]) -> dict[str, Any]:
    """Determine direction from multiple timeframes; M15 cannot flip strong HTF structure."""
    weights = {"D1":0.18, "H4":0.42, "H1":0.30, "M15":0.10}
    reports = {}; bull = 0.0; bear = 0.0; total = 0.0
    for tf,(candles,span) in timeframes.items():
        if not candles: continue
        r = _timeframe_regime_report(candles, tf, span); reports[tf]=r
        w=weights.get(tf,0.20); bull += w*r["bullish_score"]; bear += w*r["bearish_score"]; total += w
    if total > EPS: bull/=total; bear/=total
    gap=abs(bull-bear)
    if bull>=57 and bull-bear>=15: trend="BULLISH"
    elif bear>=57 and bear-bull>=15: trend="BEARISH"
    else: trend="RANGE"
    states=[str((reports.get(tf) or {}).get("state") or "") for tf in ("H4","H1","M15")]
    if trend=="BULLISH": state="BULLISH_PULLBACK" if any("PULLBACK" in x for x in states) else "BULLISH_CONTINUATION" if any("CONTINUATION" in x for x in states) else "BULLISH"
    elif trend=="BEARISH": state="BEARISH_PULLBACK" if any("PULLBACK" in x for x in states) else "BEARISH_CONTINUATION" if any("CONTINUATION" in x for x in states) else "BEARISH"
    else: state="TRANSITION" if gap>=6 else "RANGE"
    confidence=clamp(50+gap*2+abs(max(bull,bear)-50)*0.45)
    return {"trend":trend,"regime":trend,"state":state,"confidence":round(confidence,2),"bullish_score":round(bull,2),"bearish_score":round(bear,2),"gap":round(gap,2),"dominant_strength":round(max(bull,bear),2),"timeframes":reports,"regime_rule":"D1+H4 define external direction; H1 confirms; M15 classifies current state and cannot flip external regime alone."}


def _select_macro_trend(btc_h4: StructureSnapshot, target_h4: StructureSnapshot, pair: str, *, btc_regime: str | None=None, pair_regime: str | None=None) -> str:
    if pair=="BTCUSDT": return pair_regime or target_h4.trend
    return btc_regime or btc_h4.trend


def allowed_directions_for_macro(
    macro_trend: str,
    pair_h4_trend: str,
    pair: str,
) -> tuple[str, ...]:
    """BUY dan SELL selalu dievaluasi; BTC/regime hanya memengaruhi skor."""
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
    macro_trend_override: str | None = None,
    pair_regime_override: str | None = None,
    pair_regime_detail: dict[str, Any] | None = None,
    rsi_pack: dict[str, Any] | None = None,
    quality_pack: dict[str, Any] | None = None,
) -> list[Candidate]:
    atr = m15_atr_values[-1]
    rsi_ctx = _latest_rsi_context(m15)
    rsi = rsi_ctx["rsi14"]
    vlt = volume_trend_score(m15, m15_atr_values)
    m15_rel_vol = relative_volume(m15, 20)
    latest_disp = displacement_strength(m15, len(m15) - 1, m15_atr_values, m15_rel_vol)
    h1_dr = dealing_range(h1, h1_structure)

    macro_trend = _select_macro_trend(btc_structure, h4_structure, pair, btc_regime=macro_trend_override, pair_regime=pair_regime_override)
    pair_directional_regime = pair_regime_override or h4_structure.trend
    allowed_directions = allowed_directions_for_macro(macro_trend, pair_directional_regime, pair)
    candidates: list[Candidate] = []

    h4_ctx = h4_ctx or htf_poi_candidates(h4, h4_structure, DIRECTION_BUY, "H4")
    h1_ctx = h1_ctx or htf_poi_candidates(h1, h1_structure, DIRECTION_BUY, "H1")
    m15_ctx = m15_ctx or htf_poi_candidates(m15, m15_structure, DIRECTION_BUY, "M15")
    combined_liquidity = topdown_liquidity or [
        *h4_ctx.get("liquidity", []),
        *h1_ctx.get("liquidity", []),
        *m15_ctx.get("liquidity", []),
    ]
    h1_atr_ctx = safe_float(h1_ctx.get("atr"), atr * 3.0)
    live = live_pools(
        combined_liquidity,
        {"H4": h4, "H1": h1, "M15": m15},
        {"H4": safe_float(h4_ctx.get("atr"), h1_atr_ctx * 2.0), "H1": h1_atr_ctx, "M15": atr},
    )

    for direction in allowed_directions:
        primary_ranked = rank_primary_pois(h4_ctx, direction, current, limit=MAX_PRIMARY_POI_CANDIDATES)
        magnet = nearest_magnet(direction, live, current, h1_atr_ctx)
        topdown_count = 0
        last_sweep = None
        last_mss = None
        last_bos = None

        for rank_idx, (primary, primary_details) in enumerate(primary_ranked, start=1):
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
            last_sweep, last_mss, last_bos = sweep, mss, recent_bos

            ref_entry = refinement.midpoint if refinement else primary.midpoint
            target = find_target_liquidity_topdown(live, direction, ref_entry, h1_atr_ctx)
            target_pools = ranked_targets(live, direction, ref_entry, h1_atr_ctx)
            leg = sweep_mss_leg(direction, sweep, mss, m15, fvg)
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
                leg=leg,
                target_pools=target_pools,
                sl_pools=sl_guard_pools(live, direction, ref_entry),
            )
            if cand is None:
                continue

            cand.evidence["poi_selection"] = {
                "h4": primary_details,
                "h1": refinement_details,
                "m15": execution_details,
                "h4_candidate_rank": rank_idx,
            }
            h4_fib = h4_ctx.get("fib", {}).get(direction)
            h4_poi_score = safe_float(primary_details.get("quality"), 50.0)
            h4_fib_score = fib_confluence_score(primary, h4_fib, direction)[0]
            h1_ref_score = safe_float(refinement_details.get("quality"), 35.0) if refinement else 35.0
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
                pair_regime=pair_regime_detail,
                momentum=evaluate_momentum(direction, rsi_pack, bool(sweep)),
                quality=quality_pack,
                trigger=trigger_info(m15, sweep, mss, recent_bos, magnet),
                h4_poi_score=h4_poi_score,
                h4_fib_score=h4_fib_score,
                h1_refinement_score=h1_ref_score,
            )
            cand.evidence["trigger_status"] = (
                "M15_CONFIRMATION_AVAILABLE" if (sweep or mss or recent_bos) else "WAITING_FOR_M15_CONFIRMATION"
            )
            if not (sweep or mss or recent_bos):
                cand.confidence = round(max(0.0, cand.confidence - 18.0), 2)
                cand.evidence["confidence_penalty"] = 18.0

            candidates.append(cand)
            topdown_count += 1

        # Only use the M15 resilience path when no reachable top-down candidate
        # could be produced. This protects the HTF thesis rather than bypassing it.
        if topdown_count == 0:
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
                        "sweep": last_sweep,
                        "mss": last_mss,
                        "bos": last_bos,
                        "macro_trend": macro_trend,
                        "allowed_directions": list(allowed_directions),
                        "top_down_primary_available": bool(primary_ranked),
                        "fallback_reason": "Tidak ada H4 POI/path menghasilkan entry yang reachable; M15 resilience path dipakai.",
                    },
                    rsi,
                    vlt["direction"],
                    h4_structure,
                    h1_dr,
                )
                if cand:
                    reach_score, reach_details = entry_reachability(
                        direction,
                        current,
                        cand.entry,
                        safe_float(h4_ctx.get("atr"), atr * 4.0),
                    )
                    cand.evidence["entry_reachability"] = {"score": reach_score, **reach_details}
                    score_candidate(
                        cand,
                        macro_trend=macro_trend,
                        pair_h4_trend=h4_structure.trend,
                        h1_dr=h1_dr,
                        target=target,
                        sweep=last_sweep,
                        current=current,
                        m15_atr=atr,
                        m15_rsi=rsi,
                        m15_rsi_slope=rsi_ctx["slope"],
                        vlt=vlt,
                        latest_displacement=latest_disp,
                        pair_regime=pair_regime_detail,
                        momentum=evaluate_momentum(direction, rsi_pack, bool(last_sweep)),
                        quality=quality_pack,
                        trigger=trigger_info(m15, last_sweep, last_mss, last_bos, magnet),
                        h4_poi_score=25.0,
                        h4_fib_score=25.0,
                        h1_refinement_score=location_score(direction, cand.entry, h1_dr),
                    )
                    cand.confidence = round(max(0.0, cand.confidence - 10.0), 2)
                    cand.evidence["confidence_penalty"] = 10.0
                    cand.confidence = min(cand.confidence, FALLBACK_CONFIDENCE_CAP)
                    cand.evidence["trigger_status"] = "M15_FALLBACK"
                    candidates.append(cand)

    if not candidates:
        if len(allowed_directions) == 1:
            direction = allowed_directions[0]
        else:
            direction = DIRECTION_BUY if h4_structure.trend != "BEARISH" else DIRECTION_SELL
        fallback = fallback_candidate(direction, current, atr, h1_dr, h4_structure, combined_liquidity)
        reach_score, reach_details = entry_reachability(
            direction,
            current,
            fallback.entry,
            safe_float(h4_ctx.get("atr"), atr * 4.0),
        )
        fallback.evidence.update({
            "macro_trend": macro_trend,
            "allowed_directions": list(allowed_directions),
            "directional_regime": directional_regime_label(allowed_directions),
            "entry_reachability": {"score": reach_score, **reach_details},
            "top_down": {
                "h4_primary_poi": False,
                "h1_refinement": False,
                "m15_execution": False,
            },
        })
        candidates.append(fallback)

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
        "HTF_POI_M15_MSS": 5,
        "HTF_POI_M15_REFINEMENT": 4,
        "M15_SMC_FALLBACK": 2,
        "BREAKER_RETEST": 1,
        "STRUCTURE_PULLBACK_FALLBACK": 0,
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
# STRUCTURAL PREDICTION ENGINE V2
# ============================================================================


def _v2_tf_span(timeframe: str) -> int:
    return {
        "D1": 2,
        "H4": SWING_SPAN_H4,
        "H1": SWING_SPAN_H1,
        "M15": SWING_SPAN_M15,
    }.get(str(timeframe).upper(), SWING_SPAN_M15)


def _v2_structure_signature(structure: StructureSnapshot) -> tuple[Any, ...]:
    def _event_sig(event: dict[str, Any] | None) -> tuple[Any, ...] | None:
        if not event:
            return None
        return (
            event.get("type"),
            event.get("direction"),
            event.get("index"),
            round(safe_float(event.get("level")), 8),
        )

    return (
        structure.trend,
        _event_sig(structure.last_bos),
        _event_sig(structure.last_mss),
        round(safe_float(structure.protected_high), 8),
        round(safe_float(structure.protected_low), 8),
        tuple((p.kind, p.index, round(p.price, 8)) for p in structure.swing_highs[-3:]),
        tuple((p.kind, p.index, round(p.price, 8)) for p in structure.swing_lows[-3:]),
    )


def _v2_pivot_raw_levels(
    structure: StructureSnapshot,
    timeframe: str,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for p in structure.swing_highs[-24:]:
        raw.append({
            "side": "BUY_SIDE",
            "level": p.price,
            "index": p.index,
            "source": "SWING_HIGH",
        })
    for p in structure.swing_lows[-24:]:
        raw.append({
            "side": "SELL_SIDE",
            "level": p.price,
            "index": p.index,
            "source": "SWING_LOW",
        })

    pivots_high = structure.swing_highs[-24:]
    pivots_low = structure.swing_lows[-24:]
    price = 0.0
    if pivots_high or pivots_low:
        prices = [p.price for p in (*pivots_high, *pivots_low)]
        price = median_or(prices, 0.0)

    # Equal/repeated levels from confirmed pivots. We retain source metadata,
    # then clustering later consolidates everything into zones.
    for pivots, side in ((pivots_high, "BUY_SIDE"), (pivots_low, "SELL_SIDE")):
        for i in range(len(pivots) - 1):
            for j in range(i + 1, min(i + 7, len(pivots))):
                a, b = pivots[i], pivots[j]
                if abs(a.price - b.price) <= max(price * LIQ_MIN_TOLERANCE_PCT, 1e-12):
                    raw.append({
                        "side": side,
                        "level": (a.price + b.price) / 2.0,
                        "index": b.index,
                        "source": "EQUAL_LEVEL",
                    })

    return raw


def _v2_cluster_liquidity_levels(
    raw_levels: list[dict[str, Any]],
    candles: list[Candle],
    timeframe: str,
    current: float,
) -> list[LiquidityZone]:
    if not candles or not raw_levels:
        return []
    atr_values = atr_series(candles, 14)
    atr = atr_values[-1] if atr_values else max(candles[-1].range, current * 0.005)
    tolerance = max(current * LIQ_MIN_TOLERANCE_PCT, atr * LIQ_ATR_TOLERANCE)
    groups: list[dict[str, Any]] = []

    for item in sorted(raw_levels, key=lambda x: (x.get("side", ""), safe_float(x.get("level")))):
        side = str(item.get("side") or "")
        level = safe_float(item.get("level"), 0.0)
        if level <= 0 or side not in {"BUY_SIDE", "SELL_SIDE"}:
            continue
        selected = None
        for group in groups:
            if group["side"] != side:
                continue
            if abs(level - group["midpoint"]) <= tolerance:
                selected = group
                break
        if selected is None:
            selected = {
                "side": side,
                "levels": [],
                "midpoint": level,
                "timeframe": timeframe,
            }
            groups.append(selected)
        selected["levels"].append(item)
        selected["midpoint"] = sum(safe_float(x["level"]) for x in selected["levels"]) / max(len(selected["levels"]), 1)

    result: list[LiquidityZone] = []
    for group in groups:
        levels = group["levels"]
        midpoint = safe_float(group["midpoint"])
        half = max(
            max(safe_float(x["level"]) for x in levels) - min(safe_float(x["level"]) for x in levels),
            atr * 0.05,
        ) * 0.5
        low = min(safe_float(x["level"]) for x in levels) - half
        high = max(safe_float(x["level"]) for x in levels) + half
        source_types = sorted({str(x.get("source") or "UNKNOWN") for x in levels})
        touches = len(levels)
        first_index = min(int(x.get("index", 0)) for x in levels)
        last_index = max(int(x.get("index", 0)) for x in levels)

        # Recent range extreme as an explicit liquidity source.
        window_size = 32 if timeframe == "H4" else 48 if timeframe == "H1" else 96
        window_start = max(0, len(candles) - window_size)
        window = candles[window_start:]
        if window:
            if group["side"] == "BUY_SIDE":
                extreme_idx = max(range(window_start, len(candles)), key=lambda i: candles[i].high)
                extreme = candles[extreme_idx].high
            else:
                extreme_idx = min(range(window_start, len(candles)), key=lambda i: candles[i].low)
                extreme = candles[extreme_idx].low
            if abs(extreme - midpoint) <= tolerance:
                source_types.append("RECENT_RANGE_EXTREME")
                # Preserve the actual extreme timestamp for recency/sweep logic.
                last_index = max(last_index, int(extreme_idx))

        raw_strength = 0.0
        raw_strength += min(25.0, touches * 8.0)
        if "EQUAL_LEVEL" in source_types:
            raw_strength += LIQ_EQUAL_BONUS
        if timeframe == "H4":
            raw_strength += LIQ_H4_BONUS
        elif timeframe == "H1":
            raw_strength += LIQ_H1_BONUS
        if last_index >= len(candles) - max(8, len(candles) // 12):
            raw_strength += LIQ_RECENT_BONUS
        if len(source_types) >= 2:
            raw_strength += LIQ_MULTI_SOURCE_BONUS
        if "RECENT_RANGE_EXTREME" in source_types:
            raw_strength += LIQ_RANGE_EXTREME_BONUS

        # Confirm whether the level was actually consumed/swept after formation.
        sweep_tol = max(midpoint * 0.0002, atr * 0.05)
        swept = False
        sweep_index: int | None = None
        for idx in range(min(len(candles) - 1, last_index + 1), len(candles)):
            c = candles[idx]
            if group["side"] == "SELL_SIDE" and c.low < midpoint - sweep_tol:
                swept = True
                sweep_index = idx
                break
            if group["side"] == "BUY_SIDE" and c.high > midpoint + sweep_tol:
                swept = True
                sweep_index = idx
                break
        significance = clamp(raw_strength - (LIQ_SWEEP_PENALTY if swept else 0.0))
        result.append(
            LiquidityZone(
                side=group["side"],
                low=round_price(low),
                high=round_price(high),
                midpoint=round_price(midpoint),
                strength=round(clamp(raw_strength), 2),
                timeframe=timeframe,
                touches=touches,
                source_types=sorted(set(source_types)),
                first_index=first_index,
                last_index=last_index,
                swept=swept,
                sweep_index=sweep_index,
                distance_pct=round(distance_pct(current, midpoint), 5),
                distance_atr=round(abs(midpoint - current) / max(atr, EPS), 4),
                significance=round(significance, 2),
            )
        )

    result.sort(key=lambda z: (z.significance, z.strength, z.touches, -z.distance_atr), reverse=True)
    return result[:LIQ_ZONE_MAX_PER_TF]


def _v2_mark_liquidity_roles(
    zones: list[LiquidityZone],
    direction: str,
    current: float,
    entry_low: float | None = None,
    entry_high: float | None = None,
) -> None:
    entry_low = safe_float(entry_low, 0.0) if entry_low is not None else 0.0
    entry_high = safe_float(entry_high, 0.0) if entry_high is not None else 0.0
    for zone in zones:
        if direction == "BUY":
            if zone.side == "SELL_SIDE":
                if entry_low and zone.high <= entry_high:
                    zone.role = "ENTRY_LIQUIDITY"
                elif zone.midpoint < current and zone.significance >= LIQ_SIGNIFICANT_THRESHOLD_V2:
                    zone.role = "SL_GUARD_LIQUIDITY"
                else:
                    zone.role = "INTERNAL_LIQUIDITY"
            else:
                zone.role = "TARGET_LIQUIDITY" if zone.midpoint > current else "INTERNAL_LIQUIDITY"
        else:
            if zone.side == "BUY_SIDE":
                if entry_high and zone.low >= entry_low:
                    zone.role = "ENTRY_LIQUIDITY"
                elif zone.midpoint > current and zone.significance >= LIQ_SIGNIFICANT_THRESHOLD_V2:
                    zone.role = "SL_GUARD_LIQUIDITY"
                else:
                    zone.role = "INTERNAL_LIQUIDITY"
            else:
                zone.role = "TARGET_LIQUIDITY" if zone.midpoint < current else "INTERNAL_LIQUIDITY"


def _v2_build_liquidity_map(
    candles_by_tf: dict[str, list[Candle]],
    structures: dict[str, StructureSnapshot],
    current: float,
    direction: str,
) -> dict[str, Any]:
    all_zones: list[LiquidityZone] = []
    for tf in ("H4", "H1", "M15"):
        candles = candles_by_tf.get(tf) or []
        structure = structures.get(tf)
        if not candles or structure is None:
            continue
        raw = _v2_pivot_raw_levels(structure, tf)
        # Recent range boundaries are first-class liquidity references.
        window = candles[-96:] if tf == "M15" else candles[-48:] if tf == "H1" else candles[-32:]
        if window:
            raw.append({"side": "BUY_SIDE", "level": max(c.high for c in window), "index": max(0, len(candles) - len(window)), "source": "RECENT_RANGE_HIGH"})
            raw.append({"side": "SELL_SIDE", "level": min(c.low for c in window), "index": max(0, len(candles) - len(window)), "source": "RECENT_RANGE_LOW"})
        zones = _v2_cluster_liquidity_levels(raw, candles, tf, current)
        all_zones.extend(zones)

    # Deduplicate cross-timeframe zones while preserving the most important TF.
    all_zones.sort(key=lambda z: (z.significance, z.strength, z.timeframe == "H4", z.timeframe == "H1"), reverse=True)
    merged: list[LiquidityZone] = []
    for zone in all_zones:
        overlap = False
        for existing in merged:
            tol = max(current * LIQ_MIN_TOLERANCE_PCT, (zone.distance_atr + existing.distance_atr) * 0.0 + current * 0.0005)
            if zone.side == existing.side and abs(zone.midpoint - existing.midpoint) <= tol:
                # Keep separate TF detail in source types; strengthen the more significant zone.
                existing.low = min(existing.low, zone.low)
                existing.high = max(existing.high, zone.high)
                existing.midpoint = round_price((existing.midpoint + zone.midpoint) / 2.0)
                existing.strength = round(clamp(max(existing.strength, zone.strength) + (5.0 if zone.timeframe != existing.timeframe else 0.0)), 2)
                existing.significance = round(clamp(max(existing.significance, zone.significance) + (8.0 if zone.timeframe != existing.timeframe else 0.0)), 2)
                existing.touches += zone.touches
                existing.source_types = sorted(set(existing.source_types + zone.source_types + [zone.timeframe]))
                existing.swept = existing.swept and zone.swept
                overlap = True
                break
        if not overlap:
            merged.append(zone)

    for z in merged:
        z.distance_pct = round(distance_pct(current, z.midpoint), 5)
        z.role = "INTERNAL_LIQUIDITY"

    above = sorted([z for z in merged if z.midpoint > current], key=lambda z: z.midpoint)
    below = sorted([z for z in merged if z.midpoint < current], key=lambda z: z.midpoint, reverse=True)
    significant_above = [z for z in above if z.significance >= LIQ_SIGNIFICANT_THRESHOLD_V2]
    significant_below = [z for z in below if z.significance >= LIQ_SIGNIFICANT_THRESHOLD_V2]
    opposing = [z for z in (below if direction == "BUY" else above) if not z.swept and z.significance >= LIQ_SIGNIFICANT_THRESHOLD_V2]
    magnet = None
    for z in opposing:
        if z.distance_atr <= LIQ_MAGNET_MAX_H1_ATR_V2:
            magnet = z
            break

    _v2_mark_liquidity_roles(merged, direction, current)
    return {
        "all": merged,
        "above": above,
        "below": below,
        "significant_above": significant_above,
        "significant_below": significant_below,
        "magnet": magnet,
        "direction": direction,
    }


def _v2_zone_overlap_values(
    a_low: float,
    a_high: float,
    b_low: float,
    b_high: float,
) -> float:
    overlap_low = max(a_low, b_low)
    overlap_high = min(a_high, b_high)
    overlap = max(0.0, overlap_high - overlap_low)
    reference = max(min(a_high - a_low, b_high - b_low), EPS)
    return clamp(overlap / reference)


def _v2_assess_impulse_quality(
    candles: list[Candle],
    structure: StructureSnapshot,
    direction: str,
    fib: FibonacciContext | None,
) -> dict[str, Any]:
    if not candles or fib is None:
        return {
            "quality": 0.0,
            "classification": "WEAK",
            "net_move_pct": 0.0,
            "body_efficiency": 0.0,
            "opposite_candle_ratio": 1.0,
            "displacement_count": 0,
            "structure_preserved": False,
        }
    start = min(fib.low_index, fib.high_index)
    end = max(fib.low_index, fib.high_index)
    seg = candles[start : end + 1]
    if len(seg) < 3:
        return {
            "quality": 0.0,
            "classification": "WEAK",
            "net_move_pct": 0.0,
            "body_efficiency": 0.0,
            "opposite_candle_ratio": 1.0,
            "displacement_count": 0,
            "structure_preserved": False,
        }
    net_move = abs(seg[-1].close - seg[0].open)
    net_move_pct = net_move / max(abs(seg[0].open), EPS) * 100.0
    total_range = sum(max(c.range, EPS) for c in seg)
    net_eff = min(1.0, net_move / max(total_range, EPS))
    if direction == "BUY":
        opposite = sum(1 for c in seg if c.bearish)
    else:
        opposite = sum(1 for c in seg if c.bullish)
    opposite_ratio = opposite / len(seg)
    body_eff = sum(c.body for c in seg) / max(total_range, EPS)
    atr_values = atr_series(candles, 14)
    rel_vol = relative_volume(candles, 20)
    disp_count = 0
    for idx in range(start, min(end + 1, len(candles))):
        if idx < len(atr_values) and idx < len(rel_vol):
            if displacement_strength(candles, idx, atr_values, rel_vol) >= 60:
                disp_count += 1
    preserved = (
        (direction == "BUY" and structure.trend == "BULLISH" and structure.protected_low is not None)
        or (direction == "SELL" and structure.trend == "BEARISH" and structure.protected_high is not None)
    )
    quality = (
        clamp(net_move_pct / max(FIB_IMPULSE_MIN_MOVE_PCT * 2.5, 1.0) * 30.0)
        + clamp(net_eff * 100.0) * 0.30
        + clamp(body_eff * 100.0) * 0.20
        + clamp((1.0 - opposite_ratio) * 100.0) * 0.20
        + min(10.0, disp_count * 2.5)
    )
    if not preserved:
        quality -= 20.0
    quality = clamp(quality)
    if quality >= FIB_CLEAN_QUALITY:
        classification = "CLEAN"
    elif quality >= FIB_STRUGGLE_QUALITY:
        classification = "STRUGGLE"
    else:
        classification = "WEAK"
    return {
        "quality": round(quality, 2),
        "classification": classification,
        "net_move_pct": round(net_move_pct, 3),
        "body_efficiency": round(body_eff, 4),
        "opposite_candle_ratio": round(opposite_ratio, 4),
        "displacement_count": disp_count,
        "structure_preserved": preserved,
        "anchor_indices": {"start": start, "end": end},
    }


def _v2_build_adaptive_fib_zones(
    candles: list[Candle],
    structure: StructureSnapshot,
    direction: str,
    timeframe: str,
) -> dict[str, Any]:
    fib = build_fibonacci(candles, structure, direction, timeframe)
    if fib is None and candles:
        # V2 fallback anchor: use the latest ordered structural extreme pair in a bounded window.
        window_start = max(0, len(candles) - 120)
        segment = candles[window_start:]
        if direction == "BUY":
            low_rel = min(range(len(segment)), key=lambda i: segment[i].low)
            later = segment[low_rel:] or segment
            high_rel_local = max(range(len(later)), key=lambda i: later[i].high)
            low_idx = window_start + low_rel
            high_idx = window_start + low_rel + high_rel_local
            low = segment[low_rel].low
            high = later[high_rel_local].high
            reason = "V2 fallback Fib menggunakan structural low -> high terbaru karena anchor event belum cukup eksplisit."
        else:
            high_rel = max(range(len(segment)), key=lambda i: segment[i].high)
            later = segment[high_rel:] or segment
            low_rel_local = min(range(len(later)), key=lambda i: later[i].low)
            high_idx = window_start + high_rel
            low_idx = window_start + high_rel + low_rel_local
            high = segment[high_rel].high
            low = later[low_rel_local].low
            reason = "V2 fallback Fib menggunakan structural high -> low terbaru karena anchor event belum cukup eksplisit."
        if high > low and high_idx != low_idx:
            fib = FibonacciContext(
                timeframe=timeframe,
                direction=direction,
                swing_low=low,
                swing_high=high,
                low_index=min(low_idx, high_idx),
                high_index=max(low_idx, high_idx),
                anchor_reason=reason,
                trend_strength=trend_strength_metrics(candles, structure),
                levels={f"{ratio:.3f}": round_price(high - ratio * (high - low) if direction == "BUY" else low + ratio * (high - low)) for ratio in (0.0, 0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 1.0)},
            )
    if fib is None:
        return {"available": False, "zones": [], "fib": None, "impulse_quality": {"classification": "WEAK", "quality": 0.0}}
    impulse = _v2_assess_impulse_quality(candles, structure, direction, fib)
    if impulse["classification"] == "CLEAN":
        primary = (FIB_NORMAL_LOW, FIB_NORMAL_HIGH)
        secondary = (FIB_SECONDARY_SHALLOW_LOW, FIB_DEEP_LOW)
    elif impulse["classification"] == "STRUGGLE":
        primary = (FIB_DEEP_LOW, FIB_DEEP_HIGH)
        secondary = (FIB_NORMAL_LOW, FIB_NORMAL_HIGH)
    else:
        primary = (FIB_DEEP_LOW, FIB_DEEP_HIGH)
        secondary = (FIB_NORMAL_LOW, FIB_NORMAL_HIGH)

    def _zone(r1: float, r2: float, label: str) -> dict[str, Any]:
        low, high = fib_zone_for_direction(fib, r1, r2)
        return {
            "label": label,
            "ratio_low": min(r1, r2),
            "ratio_high": max(r1, r2),
            "price_low": round_price(low),
            "price_high": round_price(high),
            "midpoint": round_price((low + high) / 2.0),
        }

    return {
        "available": True,
        "direction": direction,
        "timeframe": timeframe,
        "fib": fib,
        "impulse_quality": impulse,
        "primary_zone": _zone(primary[0], primary[1], "PRIMARY"),
        "secondary_zone": _zone(secondary[0], secondary[1], "SECONDARY"),
        "deep_zone": _zone(FIB_DEEP_LOW, FIB_DEEP_HIGH, "DEEP_0.618_0.786"),
        "normal_zone": _zone(FIB_NORMAL_LOW, FIB_NORMAL_HIGH, "NORMAL_0.382_0.500"),
        "reason": (
            "Clean impulse → normal retracement preferred (0.382-0.500)."
            if impulse["classification"] == "CLEAN"
            else "Struggle impulse → deeper discount/premium retracement preferred (0.618-0.786)."
            if impulse["classification"] == "STRUGGLE"
            else "Weak impulse → deep and normal zones retained only as low-confidence candidates."
        ),
    }


def _v2_zone_as_zone(item: dict[str, Any], kind: str = "V2_ZONE") -> Zone:
    return Zone(
        kind=kind,
        low=safe_float(item.get("price_low")),
        high=safe_float(item.get("price_high")),
        index=0,
        source=str(item.get("label") or kind),
        strength=50.0,
        details={},
    )


def _v2_liquidity_fib_candidates(
    liquidity_map: dict[str, Any],
    fib_analysis: dict[str, Any],
    direction: str,
    current: float,
    h1_atr: float,
) -> list[dict[str, Any]]:
    if not fib_analysis.get("available"):
        return []
    zones = [fib_analysis["primary_zone"], fib_analysis["secondary_zone"], fib_analysis["deep_zone"], fib_analysis["normal_zone"]]
    unique_zones: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for z in zones:
        key = (safe_float(z["price_low"]), safe_float(z["price_high"]))
        if key in seen:
            continue
        seen.add(key)
        unique_zones.append(z)

    pool_side = "SELL_SIDE" if direction == "BUY" else "BUY_SIDE"
    pools = [z for z in liquidity_map.get("all", []) if z.side == pool_side and not z.swept]
    candidates: list[dict[str, Any]] = []
    for fib_zone in unique_zones:
        f_low = safe_float(fib_zone["price_low"])
        f_high = safe_float(fib_zone["price_high"])
        for pool in pools:
            overlap = _v2_zone_overlap_values(f_low, f_high, pool.low, pool.high)
            distance = abs(pool.midpoint - current) / max(h1_atr, EPS)
            # POI wajib overlap nyata dengan pool; zona sintetis dari dua level dibuang.
            if overlap <= 0 or distance > 8.0:
                continue
            structure_bonus = 10.0 if pool.timeframe == "H4" else 6.0 if pool.timeframe == "H1" else 2.0
            confluence = clamp(
                pool.significance * 0.40
                + overlap * 30.0
                + structure_bonus
                + (10.0 if pool.touches >= 3 else 0.0)
            )
            low = max(f_low, pool.low)
            high = min(f_high, pool.high)
            if high <= low:
                continue
            candidates.append({
                "zone_low": round_price(low),
                "zone_high": round_price(high),
                "zone_mid": round_price((low + high) / 2.0),
                "fib": fib_zone,
                "pool": pool,
                "overlap_ratio": round(overlap, 4),
                "confluence": round(confluence, 2),
            })

    # Keep best unique location zones.
    candidates.sort(key=lambda x: (x["confluence"], x["pool"].significance, x["overlap_ratio"]), reverse=True)
    dedupe: list[dict[str, Any]] = []
    for item in candidates:
        if any(
            _v2_zone_overlap_values(item["zone_low"], item["zone_high"], d["zone_low"], d["zone_high"]) > 0.65
            for d in dedupe
        ):
            continue
        dedupe.append(item)
        if len(dedupe) >= 8:
            break
    return dedupe


def _v2_refine_location(
    location: dict[str, Any],
    direction: str,
    h1_ctx: dict[str, Any],
    m15_ctx: dict[str, Any],
) -> dict[str, Any]:
    low = safe_float(location["zone_low"])
    high = safe_float(location["zone_high"])
    parent = Zone("LOCATION_ZONE", low, high, 0, "LIQUIDITY_FIB", strength=location["confluence"])
    refinements: list[tuple[float, Zone, str]] = []
    for source_name, ctx in (("H1", h1_ctx), ("M15", m15_ctx)):
        for z in (ctx.get("fvg") or []) + (ctx.get("obs") or []) + (ctx.get("breakers") or []):
            if z.status == "FAILED":
                continue
            overlap = zone_overlap_ratio(parent, z)
            if overlap >= REFINEMENT_MIN_OVERLAP:
                q = clamp(z.strength * 0.55 + overlap * 45.0)
                if z.status == "MITIGATED":
                    q *= 0.7
                refinements.append((q, z, source_name))
    refinements.sort(key=lambda x: x[0], reverse=True)
    best = refinements[0] if refinements else None
    if best:
        q, z, source_name = best
        refined_low = max(low, z.low)
        refined_high = min(high, z.high)
        if refined_high <= refined_low:
            refined_low, refined_high = low, high
        return {
            "low": round_price(refined_low),
            "high": round_price(refined_high),
            "midpoint": round_price((refined_low + refined_high) / 2.0),
            "source": source_name,
            "kind": z.kind,
            "quality": round(q, 2),
            "overlap_ratio": round(zone_overlap_ratio(parent, z), 4),
            "parent_zone": {"low": low, "high": high},
        }
    return {
        "low": round_price(low),
        "high": round_price(high),
        "midpoint": round_price((low + high) / 2.0),
        "source": "LIQUIDITY_FIB_ONLY",
        "kind": None,
        "quality": round(location["confluence"] * 0.55, 2),
        "overlap_ratio": 0.0,
        "parent_zone": {"low": low, "high": high},
    }


def _v2_rsi_value_from_closes(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    for i in range(period + 1, len(closes)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
    if avg_loss <= EPS and avg_gain <= EPS:
        return 50.0
    if avg_loss <= EPS:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _v2_simulate_rsi_path(
    candles: list[Candle],
    target: float,
    steps: int,
    scenario: str,
) -> float:
    if not candles:
        return 50.0
    base_closes = [c.close for c in candles[-max(80, RSI_PERIOD + 20):]]
    current = base_closes[-1]
    steps = max(1, int(steps))
    path: list[float] = []
    for i in range(1, steps + 1):
        x = i / steps
        if scenario == "FAST_RETRACE":
            weight = min(1.0, x * 1.65)
            price = current + (target - current) * weight
        elif scenario == "SLOW_RETRACE":
            weight = x * x
            price = current + (target - current) * weight
        else:
            price = current + (target - current) * x
        path.append(price)
    return _v2_rsi_value_from_closes(base_closes + path, RSI_PERIOD)


def _v2_projection_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"min": 50.0, "median": 50.0, "max": 50.0, "spread": 100.0, "consensus": 0.0}
    ordered = sorted(values)
    med = float(median(ordered))
    spread = max(ordered) - min(ordered)
    consensus = clamp(100.0 - spread * 2.5)
    return {
        "min": round(min(ordered), 2),
        "median": round(med, 2),
        "max": round(max(ordered), 2),
        "spread": round(spread, 2),
        "consensus": round(consensus, 2),
    }


def _v2_classify_projected_rsi(direction: str, h4: dict[str, Any], m15: dict[str, Any]) -> str:
    h4_med = safe_float(h4.get("median"), 50.0)
    m15_med = safe_float(m15.get("median"), 50.0)
    if direction == "BUY":
        if m15_med <= RSI_BUY_IDEAL_MAX_M15 and h4_med >= RSI_H4_BREAKDOWN_BUY:
            return "SUPPORTIVE_BUY_RETRACEMENT"
        if m15_med <= RSI_BUY_ACCEPTABLE_MAX_M15 and h4_med >= RSI_H4_BREAKDOWN_BUY:
            return "ACCEPTABLE_BUY_RETRACEMENT"
        if m15_med <= RSI_BUY_WARNING_MAX_M15:
            return "WARNING_SHALLOW_BUY_RETRACEMENT"
        return "WEAK_BUY_RETRACEMENT"
    if m15_med >= RSI_SELL_IDEAL_MIN_M15 and h4_med <= RSI_H4_BREAKDOWN_SELL:
        return "SUPPORTIVE_SELL_RETRACEMENT"
    if m15_med >= RSI_SELL_ACCEPTABLE_MIN_M15 and h4_med <= RSI_H4_BREAKDOWN_SELL:
        return "ACCEPTABLE_SELL_RETRACEMENT"
    if m15_med >= RSI_SELL_WARNING_MIN_M15:
        return "WARNING_SHALLOW_SELL_RETRACEMENT"
    return "WEAK_SELL_RETRACEMENT"


def _v2_project_rsi_to_entry_zone(
    m15: list[Candle],
    h4: list[Candle],
    entry_low: float,
    entry_high: float,
    direction: str,
) -> dict[str, Any]:
    m15_rsi_now = _latest_rsi_context(m15).get("rsi14", 50.0)
    h4_rsi_now = _latest_rsi_context(h4).get("rsi14", 50.0)
    current_price = m15[-1].close if m15 else (entry_low + entry_high) / 2.0
    midpoint = (entry_low + entry_high) / 2.0
    if direction == "BUY":
        entry_target = min(midpoint, current_price - max(current_price * 1e-6, EPS))
        entry_target = max(entry_low, entry_target)
    else:
        entry_target = max(midpoint, current_price + max(current_price * 1e-6, EPS))
        entry_target = min(entry_high, entry_target)
    entry_distance_pct = distance_pct(entry_target, current_price) if m15 else 0.0
    m15_steps = max(RSI_PROJECTION_MIN_STEPS_M15, min(RSI_PROJECTION_MAX_STEPS_M15, int(abs(entry_distance_pct) / 0.5) + 3))
    m15_values = [
        _v2_simulate_rsi_path(m15, entry_target, m15_steps, scenario)
        for scenario in RSI_PROJECTION_SCENARIOS
    ]
    h4_values = [
        _v2_simulate_rsi_path(h4, entry_target, RSI_PROJECTION_STEPS_H4, scenario)
        for scenario in RSI_PROJECTION_SCENARIOS
    ]
    m15_stats = _v2_projection_stats(m15_values)
    h4_stats = _v2_projection_stats(h4_values)
    classification = _v2_classify_projected_rsi(direction, h4_stats, m15_stats)
    consensus = (m15_stats["consensus"] * 0.70) + (h4_stats["consensus"] * 0.30)
    return {
        "current": {"H4": round(h4_rsi_now, 2), "M15": round(m15_rsi_now, 2)},
        "at_entry": {"H4": h4_stats, "M15": m15_stats},
        "entry_target": round_price(entry_target),
        "scenario_values": {
            "H4": {k: round(v, 2) for k, v in zip(RSI_PROJECTION_SCENARIOS, h4_values)},
            "M15": {k: round(v, 2) for k, v in zip(RSI_PROJECTION_SCENARIOS, m15_values)},
        },
        "scenario_consensus": round(consensus, 2),
        "classification": classification,
        "distance_pct": round(entry_distance_pct, 4),
    }


def _v2_score_rsi_projection(direction: str, projection: dict[str, Any]) -> float:
    classification = str(projection.get("classification") or "")
    consensus = safe_float(projection.get("scenario_consensus"), 0.0)
    base = {
        "SUPPORTIVE_BUY_RETRACEMENT": 96.0,
        "SUPPORTIVE_SELL_RETRACEMENT": 96.0,
        "ACCEPTABLE_BUY_RETRACEMENT": 82.0,
        "ACCEPTABLE_SELL_RETRACEMENT": 82.0,
        "WARNING_SHALLOW_BUY_RETRACEMENT": 58.0,
        "WARNING_SHALLOW_SELL_RETRACEMENT": 58.0,
        "WEAK_BUY_RETRACEMENT": 30.0,
        "WEAK_SELL_RETRACEMENT": 30.0,
    }.get(classification, 40.0)
    return clamp(base * 0.70 + consensus * 0.30)


def _lin(x: float, pts: list[tuple[float, float]]) -> float:
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def _v2_volume_context(direction: str, h1: list[Candle], h4: list[Candle]) -> dict[str, Any]:
    """Konfirmasi volume atas gerakan searah thesis; volume tipis = risiko fake move."""
    want_down = direction == "SELL"

    def leg(candles: list[Candle], lookback: int) -> dict[str, float] | None:
        if len(candles) < lookback + 12:
            return None
        seg = candles[-lookback:]
        base = sorted(c.volume for c in candles[-(lookback + VOLUME_BASELINE_BARS):-lookback] if c.volume > 0)
        total = sum(c.volume for c in seg)
        if len(base) < 8 or total <= 0:
            return None
        median = base[len(base) // 2]
        with_t = [c for c in seg if c.close != c.open and ((c.close < c.open) == want_down)]
        counter = [c for c in seg if c.close != c.open and ((c.close < c.open) != want_down)]
        tv = sum(c.volume for c in with_t) / len(with_t) if with_t else 0.0
        cv = sum(c.volume for c in counter) / len(counter) if counter else 0.0
        return {
            "rvol": tv / median if median > 0 else 1.0,
            "dominance": sum(c.volume for c in with_t) / total,
            "trend_vs_counter": tv / cv if cv > 0 else 2.0,
        }

    legs = {"H1": leg(h1, 24), "H4": leg(h4, 6)}
    parts: list[tuple[float, float, float]] = []
    for name, weight in (("H1", 0.6), ("H4", 0.4)):
        lg = legs[name]
        if lg is None:
            continue
        sc = (0.4 * _lin(lg["rvol"], [(0.5, 10), (0.8, 35), (1.1, 60), (1.5, 85), (2.0, 95)])
              + 0.3 * _lin(lg["dominance"], [(0.35, 15), (0.5, 55), (0.65, 90)])
              + 0.3 * _lin(lg["trend_vs_counter"], [(0.6, 10), (1.0, 55), (1.5, 85)]))
        parts.append((weight, sc, lg["rvol"]))
    if not parts:
        return {"score": 50.0, "class": "UNKNOWN", "rvol": 1.0, "legs": {}, "note": "data volume tidak cukup"}
    wsum = sum(w for w, _, _ in parts)
    score = sum(w * sc for w, sc, _ in parts) / wsum
    rvol = sum(w * rv for w, _, rv in parts) / wsum
    cls = "CONFIRMED" if score >= 65 else ("NEUTRAL" if score >= 45 else "WEAK_VOLUME")
    h1l = legs["H1"]
    if cls == "WEAK_VOLUME" and h1l and h1l["rvol"] < 0.8 and h1l["dominance"] < 0.5:
        cls = "FAKE_MOVE_RISK"
    return {"score": round(score, 2), "class": cls, "rvol": round(rvol, 3),
            "legs": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in legs.items() if v}}


def _v2_rsi_momentum_gate(direction: str, projection: dict[str, Any]) -> dict[str, Any]:
    cur_h4 = safe_float((projection.get("current") or {}).get("H4"), 50.0)
    at_entry = projection.get("at_entry") or {}
    proj_h4 = safe_float((at_entry.get("H4") or {}).get("median"), cur_h4)
    m15_med = safe_float((at_entry.get("M15") or {}).get("median"), 50.0)
    if direction == "SELL":
        h4_depth = max(0.0, RSI_H4_EXHAUST_SELL - min(cur_h4, proj_h4))
        shortfall = max(0.0, RSI_SELL_ENTRY_TARGET_M15 - m15_med)
        target, limit = RSI_SELL_ENTRY_TARGET_M15, RSI_H4_EXHAUST_SELL
    else:
        h4_depth = max(0.0, max(cur_h4, proj_h4) - RSI_H4_EXHAUST_BUY)
        shortfall = max(0.0, m15_med - RSI_BUY_ENTRY_TARGET_M15)
        target, limit = RSI_BUY_ENTRY_TARGET_M15, RSI_H4_EXHAUST_BUY
    exhaustion_pen = min(RSI_EXHAUST_MAX_PENALTY, h4_depth * RSI_EXHAUST_PENALTY_PER_POINT)
    target_pen = min(RSI_TARGET_MAX_PENALTY, shortfall * RSI_TARGET_PENALTY_PER_POINT)
    state = "EXHAUSTED" if exhaustion_pen > 0 else ("OK" if target_pen <= 0 else "BELOW_TARGET")
    return {"state": state, "penalty": round(exhaustion_pen + target_pen, 2), "exhaustion_penalty": round(exhaustion_pen, 2),
            "target_penalty": round(target_pen, 2), "h4_current": round(cur_h4, 2), "h4_projected": round(proj_h4, 2),
            "m15_projected_median": round(m15_med, 2), "m15_target": target, "h4_exhaustion_limit": limit}


def _v2_momentum_entry_price(direction: str, m15: list[Candle], h4: list[Candle], zone_low: float, zone_high: float,
                             current: float, tick_size: float) -> float | None:
    """Cari harga di zona tempat RSI M15 diproyeksikan mencapai level momentum (terdekat ke harga sekarang)."""
    lo, hi = min(zone_low, zone_high), max(zone_low, zone_high)
    if hi <= lo or not m15:
        return None
    points = [lo + (hi - lo) * i / 5.0 for i in range(6)]
    if direction == "SELL":
        prices = sorted(p for p in points if p > current)
    else:
        prices = sorted((p for p in points if p < current), reverse=True)
    for price in prices[:3]:
        proj = _v2_project_rsi_to_entry_zone(m15, h4, price, price, direction)
        med = safe_float(((proj.get("at_entry") or {}).get("M15") or {}).get("median"), 50.0)
        hit = med >= RSI_SELL_ENTRY_TARGET_M15 if direction == "SELL" else med <= RSI_BUY_ENTRY_TARGET_M15
        if hit:
            return _round_tick(price, tick_size, "DOWN" if direction == "BUY" else "UP") if tick_size > 0 else price
    return None


def _v2_select_entry_price(
    direction: str,
    zone_low: float,
    zone_high: float,
    projection: dict[str, Any],
    current: float,
    h4_atr: float,
    confluence: float,
) -> float:
    zone_low, zone_high = min(zone_low, zone_high), max(zone_low, zone_high)
    rsi_med = safe_float((projection.get("at_entry") or {}).get("M15", {}).get("median"), 50.0)
    if direction == "BUY":
        ideal_rsi = RSI_BUY_IDEAL_MAX_M15
        quality = clamp((ideal_rsi - rsi_med + 25.0) / 55.0 * 100.0)
        bias = min(clamp((confluence * 0.65 + quality * 0.35) / 100.0), ENTRY_MAX_DEPTH_BIAS)
        # Stronger projected cooling prefers the deeper part of the zone, but
        # never beyond zone boundaries.
        price = zone_high - (zone_high - zone_low) * bias
        return max(zone_low, min(current - max(h4_atr * 0.01, EPS), price))
    ideal_rsi = RSI_SELL_IDEAL_MIN_M15
    quality = clamp((rsi_med - ideal_rsi + 25.0) / 55.0 * 100.0)
    bias = min(clamp((confluence * 0.65 + quality * 0.35) / 100.0), ENTRY_MAX_DEPTH_BIAS)
    price = zone_low + (zone_high - zone_low) * bias
    return min(zone_high, max(current + max(h4_atr * 0.01, EPS), price))


def _v2_find_invalidation_candidates(
    direction: str,
    entry: float,
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
    liquidity_map: dict[str, Any],
    sweeps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if direction == "BUY":
        if h4_structure.protected_low is not None and h4_structure.protected_low < entry:
            candidates.append({"level": h4_structure.protected_low, "source": "H4_PROTECTED_LOW", "strength": 96.0})
        if h1_structure.protected_low is not None and h1_structure.protected_low < entry:
            candidates.append({"level": h1_structure.protected_low, "source": "H1_PROTECTED_LOW", "strength": 82.0})
        for pool in liquidity_map.get("below", []):
            # A pool may straddle the selected entry zone. Its lower boundary
            # is still a legitimate prediction-invalidation candidate for BUY.
            if pool.low < entry and pool.significance >= MIN_INVALIDATION_STRENGTH:
                candidates.append({
                    "level": pool.low,
                    "source": f"{pool.timeframe}_SELL_SIDE_LIQUIDITY",
                    "strength": pool.significance,
                    "pool": pool,
                })
        for sweep in reversed(sweeps[-12:]):
            if sweep.get("direction") == "BULLISH" and safe_float(sweep.get("level")) < entry:
                candidates.append({"level": safe_float(sweep.get("level")), "source": "BULLISH_SWEEP_EXTREME", "strength": 90.0, "sweep": sweep})
    else:
        if h4_structure.protected_high is not None and h4_structure.protected_high > entry:
            candidates.append({"level": h4_structure.protected_high, "source": "H4_PROTECTED_HIGH", "strength": 96.0})
        if h1_structure.protected_high is not None and h1_structure.protected_high > entry:
            candidates.append({"level": h1_structure.protected_high, "source": "H1_PROTECTED_HIGH", "strength": 82.0})
        for pool in liquidity_map.get("above", []):
            # Mirror logic for SELL: the upper boundary of an overlapping pool
            # can invalidate the bearish thesis.
            if pool.high > entry and pool.significance >= MIN_INVALIDATION_STRENGTH:
                candidates.append({
                    "level": pool.high,
                    "source": f"{pool.timeframe}_BUY_SIDE_LIQUIDITY",
                    "strength": pool.significance,
                    "pool": pool,
                })
        for sweep in reversed(sweeps[-12:]):
            if sweep.get("direction") == "BEARISH" and safe_float(sweep.get("level")) > entry:
                candidates.append({"level": safe_float(sweep.get("level")), "source": "BEARISH_SWEEP_EXTREME", "strength": 90.0, "sweep": sweep})

    # Remove near-duplicates, keeping the stronger structural source.
    deduped: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda x: (safe_float(x["level"]), -safe_float(x["strength"])), reverse=direction == "SELL"):
        if any(abs(safe_float(item["level"]) - safe_float(x["level"])) <= max(entry * 0.0002, entry * SL_MIN_BUFFER_PCT / 100.0) for x in deduped):
            continue
        deduped.append(item)
    return deduped


def _v2_select_prediction_invalidation(
    direction: str,
    entry: float,
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
    liquidity_map: dict[str, Any],
    sweeps: list[dict[str, Any]],
    h1_atr: float,
) -> dict[str, Any] | None:
    candidates = _v2_find_invalidation_candidates(direction, entry, h4_structure, h1_structure, liquidity_map, sweeps)
    if not candidates:
        return None
    if direction == "BUY":
        below = [c for c in candidates if safe_float(c["level"]) < entry]
        below.sort(key=lambda x: (abs(entry - safe_float(x["level"])), -safe_float(x["strength"])))
    else:
        below = [c for c in candidates if safe_float(c["level"]) > entry]
        below.sort(key=lambda x: (abs(entry - safe_float(x["level"])), -safe_float(x["strength"])))
    if not below:
        return None

    selected = None
    for item in below:
        dist_atr = abs(entry - safe_float(item["level"])) / max(h1_atr, EPS)
        if dist_atr <= SL_MAX_RISK_H1_ATR_V2 and safe_float(item["strength"]) >= MIN_INVALIDATION_STRENGTH:
            selected = item
            break
    if selected is None:
        selected = below[0]

    anchor = safe_float(selected["level"])
    buffer = min(
        h1_atr * SL_INVALIDATION_BUFFER_ATR,
        h1_atr * SL_INVALIDATION_BUFFER_MAX_ATR,
    )
    buffer = max(buffer, entry * SL_MIN_BUFFER_PCT / 100.0)
    sl = anchor - buffer if direction == "BUY" else anchor + buffer

    # If a significant liquidity pool sits between entry and structural anchor,
    # push SL only just beyond the pool; do not create an arbitrary ATR floor.
    guard_pool = None
    if direction == "BUY":
        guards = [
            z for z in liquidity_map.get("below", [])
            if z.low < entry and z.significance >= MIN_INVALIDATION_STRENGTH
        ]
        guards.sort(key=lambda z: z.midpoint, reverse=True)
        for pool in guards:
            if pool.low <= anchor <= pool.high or pool.high >= anchor - h1_atr * 0.35:
                guard_pool = pool
                break
        if guard_pool is not None and guard_pool.low < sl:
            sl = guard_pool.low - min(h1_atr * SL_INVALIDATION_BUFFER_ATR, h1_atr * 0.20)
    else:
        guards = [
            z for z in liquidity_map.get("above", [])
            if z.high > entry and z.significance >= MIN_INVALIDATION_STRENGTH
        ]
        guards.sort(key=lambda z: z.midpoint)
        for pool in guards:
            if pool.low <= anchor <= pool.high or pool.low <= anchor + h1_atr * 0.35:
                guard_pool = pool
                break
        if guard_pool is not None and guard_pool.high > sl:
            sl = guard_pool.high + min(h1_atr * SL_INVALIDATION_BUFFER_ATR, h1_atr * 0.20)

    min_risk = max(h1_atr * SL_MIN_RISK_H1_ATR, entry * SL_MIN_RISK_PCT / 100.0)
    noise_widened = abs(entry - sl) < min_risk
    if noise_widened:
        sl = entry - min_risk if direction == "BUY" else entry + min_risk

    risk = abs(entry - sl)
    risk_h1_atr = risk / max(h1_atr, EPS)
    risk_pct = risk / max(entry, EPS) * 100.0
    # HARD SAFETY GATE: never widen a structural stop simply to satisfy an ATR
    # floor. If prediction invalidation creates excessive risk, reject the
    # candidate rather than changing the thesis geometry.
    if risk_h1_atr > SL_MAX_RISK_H1_ATR_V2 or risk_pct > SL_MAX_RISK_PCT_V2:
        return None
    quality = 100.0
    if risk_h1_atr < SL_TOO_TIGHT_ATR_RATIO:
        quality -= 25.0
    elif SL_IDEAL_H1_ATR_LOW <= risk_h1_atr <= SL_IDEAL_H1_ATR_HIGH:
        quality += 0.0
    elif risk_h1_atr > 3.5:
        quality -= min(35.0, (risk_h1_atr - 3.5) * 12.0)
    if risk_pct > SL_MAX_RISK_PCT_V2:
        quality -= 30.0
    quality = clamp(quality)

    return {
        "anchor_level": round_price(anchor),
        "noise_floor_widened": noise_widened,
        "anchor_source": selected["source"],
        "anchor_strength": round(safe_float(selected["strength"]), 2),
        "liquidity_guard": (
            {
                "low": guard_pool.low,
                "high": guard_pool.high,
                "midpoint": guard_pool.midpoint,
                "significance": guard_pool.significance,
                "timeframe": guard_pool.timeframe,
            }
            if guard_pool is not None else None
        ),
        "buffer": round(buffer, 10),
        "sl": round_price(sl),
        "risk": round_price(risk),
        "risk_h1_atr": round(risk_h1_atr, 3),
        "risk_pct": round(risk_pct, 3),
        "quality": round(quality, 2),
        "classification": (
            "TIGHT_BUT_STRUCTURAL" if risk_h1_atr < SL_IDEAL_H1_ATR_LOW else
            "IDEAL" if risk_h1_atr <= SL_IDEAL_H1_ATR_HIGH else
            "WIDE" if risk_h1_atr <= SL_MAX_RISK_H1_ATR_V2 else
            "TOO_WIDE"
        ),
        "reason": (
            f"Prediction invalidation berasal dari {selected['source']} di {round_price(anchor)}; "
            f"SL ditempatkan di luar invalidation" +
            (f" dan liquidity guard {round_price(guard_pool.midpoint)}." if guard_pool else ".")
        ),
    }


def _v2_structural_targets(
    direction: str,
    entry: float,
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    if direction == "BUY":
        for p in reversed(h1_structure.swing_highs[-10:]):
            if p.price > entry:
                targets.append({"level": p.price, "type": "H1_SWING_HIGH", "strength": 70.0 + (5.0 if p.kind == "HIGH" else 0.0), "priority": 80.0})
        for p in reversed(h4_structure.swing_highs[-10:]):
            if p.price > entry:
                labels = [x for x in h4_structure.labels if x.get("index") == p.index]
                label = labels[-1].get("label") if labels else None
                bonus = TARGET_EXTREME_BONUS if label == "HH" else 0.0
                targets.append({"level": p.price, "type": "H4_HIGHER_HIGH" if label == "HH" else "H4_SWING_HIGH", "strength": 78.0 + bonus, "priority": 95.0})
    else:
        for p in reversed(h1_structure.swing_lows[-10:]):
            if p.price < entry:
                targets.append({"level": p.price, "type": "H1_SWING_LOW", "strength": 70.0, "priority": 80.0})
        for p in reversed(h4_structure.swing_lows[-10:]):
            if p.price < entry:
                labels = [x for x in h4_structure.labels if x.get("index") == p.index]
                label = labels[-1].get("label") if labels else None
                bonus = TARGET_EXTREME_BONUS if label == "LL" else 0.0
                targets.append({"level": p.price, "type": "H4_LOWER_LOW" if label == "LL" else "H4_SWING_LOW", "strength": 78.0 + bonus, "priority": 95.0})
    return targets


def _v2_build_target_map(
    direction: str,
    entry: float,
    sl: float,
    liquidity_map: dict[str, Any],
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
    h1_atr: float,
    current: float | None = None,
) -> list[dict[str, Any]]:
    risk = abs(entry - sl)
    targets: list[dict[str, Any]] = []
    pools = liquidity_map.get("above", []) if direction == "BUY" else liquidity_map.get("below", [])
    wanted = "BUY_SIDE" if direction == "BUY" else "SELL_SIDE"
    for pool in pools:
        if pool.side != wanted or pool.swept or pool.significance < 45:
            continue
        dist = abs(pool.midpoint - entry)
        if dist <= 0 or dist / max(h1_atr, EPS) > TARGET_MAX_DISTANCE_H1_ATR:
            continue
        if current is not None:
            if direction == "BUY" and pool.midpoint <= current:
                continue
            if direction == "SELL" and pool.midpoint >= current:
                continue
        targets.append({
            "level": pool.midpoint,
            "type": f"{pool.timeframe}_{wanted}_LIQUIDITY",
            "strength": pool.significance,
            "priority": TARGET_LIQUIDITY_BASE + pool.significance * 0.50,
            "source": pool.source_types,
            "timeframe": pool.timeframe,
            "distance_h1_atr": round(dist / max(h1_atr, EPS), 3),
        })

    structural_targets = _v2_structural_targets(direction, entry, h4_structure, h1_structure)
    if current is not None:
        structural_targets = [
            t for t in structural_targets
            if (safe_float(t["level"]) > current if direction == "BUY" else safe_float(t["level"]) < current)
        ]
    targets.extend(structural_targets)

    # Remove duplicate levels.
    unique: list[dict[str, Any]] = []
    for t in sorted(targets, key=lambda x: (safe_float(x["level"]), -safe_float(x.get("strength"))), reverse=direction == "SELL"):
        if direction == "BUY" and t["level"] <= entry:
            continue
        if direction == "SELL" and t["level"] >= entry:
            continue
        if any(abs(safe_float(t["level"]) - safe_float(u["level"])) <= max(entry * 0.0003, h1_atr * 0.05) for u in unique):
            continue
        reward = abs(safe_float(t["level"]) - entry)
        rr = reward / max(risk, EPS)
        # Count meaningful barriers before this target. Both liquidity and
        # structural targets matter: a farther liquidity pool beyond an H4
        # higher-high/lower-low is not automatically the better first target.
        between_liquidity = []
        for b in pools:
            if direction == "BUY" and entry < b.midpoint < t["level"]:
                between_liquidity.append(b)
            elif direction == "SELL" and t["level"] < b.midpoint < entry:
                between_liquidity.append(b)
        structural_levels = _v2_structural_targets(direction, entry, h4_structure, h1_structure)
        between_structural = []
        for b in structural_levels:
            if abs(safe_float(b["level"]) - safe_float(t["level"])) <= max(entry * 0.0003, h1_atr * 0.05):
                continue
            if direction == "BUY" and entry < safe_float(b["level"]) < t["level"]:
                between_structural.append(b)
            elif direction == "SELL" and t["level"] < safe_float(b["level"]) < entry:
                between_structural.append(b)
        major_barriers = [
            {"level": b.midpoint, "significance": b.significance, "source": b.source_types}
            for b in between_liquidity
            if b.significance >= TARGET_MAJOR_BARRIER_THRESHOLD
        ]
        major_barriers.extend(
            {"level": safe_float(b["level"]), "significance": safe_float(b.get("priority"), 50.0), "source": [b.get("type")]}
            for b in between_structural
            if safe_float(b.get("priority"), 0.0) >= TARGET_MAJOR_BARRIER_THRESHOLD
        )
        obstacle_penalty = min(40.0, len(major_barriers) * TARGET_OBSTACLE_PENALTY)
        rr_score = (
            100.0 if TARGET_PREFERRED_RR_LOW <= rr <= TARGET_PREFERRED_RR_HIGH else
            85.0 if rr > TARGET_PREFERRED_RR_HIGH else
            45.0 if rr >= TARGET_MIN_RR else 15.0
        )
        distance_score = clamp(100.0 - (safe_float(t.get("distance_h1_atr"), reward / max(h1_atr, EPS)) / TARGET_MAX_DISTANCE_H1_ATR) * 100.0)
        quality = clamp(
            min(100.0, safe_float(t.get("strength"), 50.0)) * 0.35
            + safe_float(t.get("priority"), 50.0) * 0.25
            + distance_score * 0.15
            + rr_score * 0.20
            + max(0.0, 5.0 - obstacle_penalty) * 1.0
        )
        unique.append({
            **t,
            "rr": round(rr, 3),
            "major_barriers": [
                {"level": round_price(safe_float(b["level"])), "significance": round(safe_float(b["significance"]), 2), "source": b.get("source")}
                for b in major_barriers[:5]
            ],
            "quality": round(quality, 2),
        })
    unique.sort(key=lambda x: (x["quality"], x["rr"], x.get("priority", 0)), reverse=True)
    return unique[:12]


def _v2_select_target(target_map: list[dict[str, Any]], direction: str) -> dict[str, Any] | None:
    # RR and obstacles are scoring dimensions, not candidate-veto conditions.
    valid = list(target_map)
    if not valid:
        return None

    # Path-first target selection: prefer the strongest reasonable destination
    # that is actually in front of price. A farther liquidity pool must not
    # jump ahead of a nearer H4 HH/LL simply because its liquidity strength is
    # high. target_map is already quality-ranked after barrier penalties.
    valid.sort(
        key=lambda x: (
            safe_float(x.get("quality"), 0.0),
            safe_float(x.get("priority"), 0.0),
            safe_float(x.get("rr"), 0.0),
        ),
        reverse=True,
    )
    return valid[0]


def _v2_structure_alignment_score(
    direction: str,
    btc_regime: dict[str, Any],
    pair_regime: dict[str, Any],
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
) -> float:
    want = "BULLISH" if direction == "BUY" else "BEARISH"
    score = 50.0
    btc_trend = str(btc_regime.get("trend") or "RANGE")
    pair_trend = str(pair_regime.get("trend") or "RANGE")
    if btc_trend == want:
        score += 8.0
    elif btc_trend == "RANGE":
        score += 0.0
    else:
        score -= 8.0  # soft context only; never a direction veto
    if pair_trend == want:
        score += 20.0
    elif pair_trend == "RANGE":
        score += 0.0
    else:
        score -= 20.0  # represented numerically; direction still gets a candidate
    if h4_structure.trend == want:
        score += 10.0
    if h1_structure.trend == want:
        score += 5.0
    return clamp(score)


def _v2_score_liquidity_confluence(location: dict[str, Any], refinement: dict[str, Any]) -> float:
    base = safe_float(location.get("confluence"), 0.0)
    pool = location.get("pool")
    if pool is not None:
        base = base * 0.75 + pool.significance * 0.25
    base = base * 0.80 + safe_float(refinement.get("quality"), 0.0) * 0.20
    return clamp(base)


def _v2_score_fib_location(location: dict[str, Any], fib_analysis: dict[str, Any]) -> float:
    fib = location.get("fib") or {}
    low = safe_float(fib.get("price_low"), 0.0)
    high = safe_float(fib.get("price_high"), 0.0)
    if high <= low:
        return 20.0
    zone_mid = safe_float(location.get("zone_mid"), (low + high) / 2.0)
    r1 = safe_float(fib.get("ratio_low"), 0.5)
    r2 = safe_float(fib.get("ratio_high"), 0.5)
    quality = 60.0
    if 0.382 <= r1 <= 0.500 and 0.382 <= r2 <= 0.500:
        quality = 90.0 if fib_analysis.get("impulse_quality", {}).get("classification") == "CLEAN" else 78.0
    elif 0.618 <= r1 <= 0.786 and 0.618 <= r2 <= 0.786:
        quality = 96.0 if fib_analysis.get("impulse_quality", {}).get("classification") in {"STRUGGLE", "WEAK"} else 88.0
    elif r2 >= 0.618 or r1 >= 0.618:
        quality = 82.0
    # Zone center remaining near current decreases quality mildly.
    if current := safe_float(fib_analysis.get("current_price"), 0.0):
        _ = current
    return clamp(quality + min(8.0, safe_float(location.get("overlap_ratio"), 0.0) * 8.0))


def _v2_momentum_current_score(
    direction: str,
    rsi_pack: dict[str, Any],
    vlt: dict[str, Any],
    m15_structure: StructureSnapshot,
) -> float:
    rsi = rsi_pack.get("M15") or {}
    cur = safe_float(rsi.get("cur"), 50.0)
    slope = safe_float(rsi.get("slope"), 0.0)
    if direction == "BUY":
        if cur >= 68:
            score = 35.0
        elif cur >= 55:
            score = 55.0
        elif 35 <= cur < 55:
            score = 85.0
        else:
            score = 72.0
        score += 8.0 if slope > 0 else -5.0 if slope < -0.8 else 0.0
    else:
        if cur <= 32:
            score = 35.0
        elif cur <= 45:
            score = 55.0
        elif 45 < cur <= 65:
            score = 85.0
        else:
            score = 72.0
        score += 8.0 if slope < 0 else -5.0 if slope > 0.8 else 0.0
    vlt_dir = str(vlt.get("direction") or "NEUTRAL")
    if (direction == "BUY" and vlt_dir == "BULLISH") or (direction == "SELL" and vlt_dir == "BEARISH"):
        score += 8.0
    elif vlt_dir in {"BULLISH", "BEARISH"}:
        score -= 4.0
    if m15_structure.trend == ("BULLISH" if direction == "BUY" else "BEARISH"):
        score += 5.0
    return clamp(score)


def _v2_score_refinement(refinement: dict[str, Any]) -> float:
    if refinement.get("kind") is None:
        return clamp(safe_float(refinement.get("quality"), 0.0) * 0.60)
    return clamp(safe_float(refinement.get("quality"), 0.0) * 0.80 + safe_float(refinement.get("overlap_ratio"), 0.0) * 20.0)


def _v2_score_invalidation(invalidation: dict[str, Any] | None, entry: float) -> float:
    if not invalidation:
        return 0.0
    quality = safe_float(invalidation.get("quality"), 0.0)
    risk_h1 = safe_float(invalidation.get("risk_h1_atr"), 0.0)
    if risk_h1 <= 0:
        return clamp(quality * 0.50)
    if risk_h1 > SL_MAX_RISK_H1_ATR_V2:
        return min(25.0, quality * 0.25)
    # Structural quality dominates; moderate risk is preferable.
    geometry = 100.0
    if risk_h1 < SL_TOO_TIGHT_ATR_RATIO:
        geometry = 50.0
    elif SL_IDEAL_H1_ATR_LOW <= risk_h1 <= SL_IDEAL_H1_ATR_HIGH:
        geometry = 100.0
    elif risk_h1 <= 4.0:
        geometry = 82.0
    else:
        geometry = 65.0
    return clamp(quality * 0.75 + geometry * 0.25)


def _v2_score_target_quality(target: dict[str, Any] | None) -> float:
    if not target:
        return 0.0
    score = safe_float(target.get("quality"), 50.0)
    if target.get("major_barriers"):
        score -= min(30.0, 10.0 * len(target["major_barriers"]))
    if safe_float(target.get("rr")) >= 2.0:
        score += 5.0
    return clamp(score)


def _v2_score_rr(rr: float) -> float:
    if rr < 1.0:
        return 10.0
    if rr < 1.5:
        return 35.0
    if rr < 2.0:
        return 55.0
    if rr <= 3.5:
        return 100.0
    if rr <= 5.0:
        return 88.0
    if rr <= 7.0:
        return 72.0
    return 48.0


def _v2_entry_reachability_v2(
    direction: str,
    current: float,
    zone_low: float,
    zone_high: float,
    h4_atr: float,
    projection_consensus: float,
) -> tuple[float, dict[str, Any]]:
    midpoint = (zone_low + zone_high) / 2.0
    pct = distance_pct(current, midpoint)
    h4_dist = abs(current - midpoint) / max(h4_atr, EPS)
    width_pct = (zone_high - zone_low) / max(abs(midpoint), EPS) * 100.0
    physical = clamp(100.0 - (pct / MAX_ENTRY_DISTANCE_PCT) * 100.0)
    atr_score = clamp(100.0 - (h4_dist / MAX_ENTRY_DISTANCE_H4_ATR) * 100.0)
    zone_width_score = 100.0 if ENTRY_ZONE_MIN_WIDTH_PCT <= width_pct <= ENTRY_ZONE_MAX_WIDTH_PCT else 65.0
    score = physical * 0.45 + atr_score * 0.30 + zone_width_score * 0.10 + projection_consensus * 0.15
    allowed = pct <= MAX_ENTRY_DISTANCE_PCT and h4_dist <= MAX_ENTRY_DISTANCE_H4_ATR
    return clamp(score), {
        "allowed": allowed,
        "distance_pct": round(pct, 4),
        "distance_h4_atr": round(h4_dist, 3),
        "zone_width_pct": round(width_pct, 4),
        "physical_score": round(physical, 2),
        "atr_score": round(atr_score, 2),
        "projection_consensus": round(projection_consensus, 2),
        "score": round(score, 2),
    }


def _v2_price_exp(
    direction: str,
    current: float,
    entry: float,
    tp: float,
    h1_atr: float,
) -> float:
    if direction == "BUY":
        exp = current + max(EXP_H1_ATR_MULTIPLIER * h1_atr, abs(current - entry) * EXP_ENTRY_DISTANCE_MULT)
        exp = min(exp, entry + abs(tp - entry) * 0.98)
        exp = min(exp, tp - max(h1_atr * 0.20, EPS))
        if exp <= current:
            exp = current + max(h1_atr * 0.50, current * 0.001)
        return round_price(exp)
    exp = current - max(EXP_H1_ATR_MULTIPLIER * h1_atr, abs(current - entry) * EXP_ENTRY_DISTANCE_MULT)
    exp = max(exp, entry - abs(tp - entry) * 0.98)
    exp = max(exp, tp + max(h1_atr * 0.20, EPS))
    if exp >= current:
        exp = current - max(h1_atr * 0.50, current * 0.001)
    return round_price(exp)


def _v2_orderflow_leg_score(direction: str, candles: list[Candle]) -> dict[str, Any]:
    """
    OHLCV order-flow proxy (not true exchange aggressor/footprint data).

    Uses signed candle-body volume, relative volume, displacement and the change
    in directional participation. Green/red candle volume is only a buyer/seller
    pressure proxy; OHLCV alone cannot identify the actual taker side reliably.
    """
    sample = candles[-20:]
    if len(sample) < 4:
        return {
            "score": 50.0, "signed_flow": 0.0, "aligned_flow": 0.0,
            "displacement": 0.0, "relative_volume": 1.0, "bars": len(sample),
            "buy_volume": 0.0, "sell_volume": 0.0,
            "buy_volume_trend": 0.0, "sell_volume_trend": 0.0,
            "participation_trend": 0.0, "method": "OHLCV_PROXY_NOT_TRUE_FOOTPRINT",
        }
    signed = gross = 0.0
    buy_volume = sell_volume = 0.0
    for c in sample:
        vol = max(c.volume, 0.0)
        body_value = (c.close - c.open) * vol
        signed += body_value
        gross += abs(body_value)
        if c.close > c.open:
            buy_volume += vol
        elif c.close < c.open:
            sell_volume += vol
    signed_flow = signed / max(gross, EPS)
    move = sample[-1].close - sample[0].open
    path = sum(abs(c.close - c.open) for c in sample)
    displacement = clamp(abs(move) / max(path, EPS) * 100.0)
    direction_sign = 1.0 if direction == "BUY" else -1.0
    aligned_flow = signed_flow * direction_sign

    previous = sample[:-5]
    recent = sample[-5:]
    def _signed_ratio(items: list[Candle]) -> float:
        numerator = sum((c.close - c.open) * max(c.volume, 0.0) for c in items)
        denominator = sum(abs(c.close - c.open) * max(c.volume, 0.0) for c in items)
        return numerator / max(denominator, EPS)

    previous_flow = _signed_ratio(previous) if previous else 0.0
    recent_flow = _signed_ratio(recent)
    participation_trend = (recent_flow - previous_flow) * direction_sign

    def _volume_side(items: list[Candle], side: str) -> float:
        selected = [max(c.volume, 0.0) for c in items if (c.close > c.open if side == "buy" else c.close < c.open)]
        return sum(selected) / max(len(items), 1)

    previous_buy = _volume_side(previous, "buy")
    recent_buy = _volume_side(recent, "buy")
    previous_sell = _volume_side(previous, "sell")
    recent_sell = _volume_side(recent, "sell")
    buy_volume_trend = (recent_buy - previous_buy) / max(previous_buy, EPS) * 100.0 if previous_buy > 0 else (100.0 if recent_buy > 0 else 0.0)
    sell_volume_trend = (recent_sell - previous_sell) / max(previous_sell, EPS) * 100.0 if previous_sell > 0 else (100.0 if recent_sell > 0 else 0.0)

    volume_base = median([c.volume for c in previous if c.volume > 0]) if any(c.volume > 0 for c in previous) else 0.0
    latest_vol = sum(max(c.volume, 0.0) for c in recent) / len(recent)
    rvol = latest_vol / volume_base if volume_base > 0 else 1.0
    flow_score = 50.0 + 28.0 * aligned_flow
    displacement_score = 0.20 * displacement
    volume_score = clamp(50.0 + (rvol - 1.0) * 18.0, 25.0, 85.0)
    trend_score = clamp(50.0 + 35.0 * participation_trend, 15.0, 85.0)
    score = clamp(flow_score * 0.50 + displacement_score * 0.15 + volume_score * 0.20 + trend_score * 0.15)
    return {
        "score": round(score, 2),
        "signed_flow": round(signed_flow, 4),
        "aligned_flow": round(aligned_flow, 4),
        "displacement": round(displacement, 2),
        "relative_volume": round(rvol, 3),
        "buy_volume": round(buy_volume, 6),
        "sell_volume": round(sell_volume, 6),
        "buy_volume_trend": round(buy_volume_trend, 2),
        "sell_volume_trend": round(sell_volume_trend, 2),
        "participation_trend": round(participation_trend, 4),
        "recent_signed_flow": round(recent_flow, 4),
        "previous_signed_flow": round(previous_flow, 4),
        "bars": len(sample),
        "method": "OHLCV_PROXY_NOT_TRUE_FOOTPRINT",
    }


def _v2_price_leg_score(direction: str, candles: list[Candle], zone_low: float, zone_high: float, current: float) -> dict[str, Any]:
    """Score impulse leg plus retracement depth/location without hard directional veto."""
    sample = candles[-24:]
    if len(sample) < 4:
        return {"score": 50.0, "impulse_pct": 0.0, "retracement_pct": 50.0}
    first = sample[0].close
    last = sample[-1].close
    impulse_pct = (last - first) / max(abs(first), EPS) * 100.0
    sign = 1.0 if direction == "BUY" else -1.0
    aligned_impulse = impulse_pct * sign
    highs = [c.high for c in sample]
    lows = [c.low for c in sample]
    full_range = max(max(highs) - min(lows), EPS)
    if direction == "BUY":
        leg_low, leg_high = min(lows), max(highs)
        retracement_pct = clamp((leg_high - current) / max(leg_high - leg_low, EPS) * 100.0)
    else:
        leg_high, leg_low = max(highs), min(lows)
        retracement_pct = clamp((current - leg_low) / max(leg_high - leg_low, EPS) * 100.0)
    zone_mid = (zone_low + zone_high) / 2.0
    loc_distance = abs(current - zone_mid) / max(full_range, EPS)
    impulse_score = clamp(50.0 + aligned_impulse * 10.0, 10.0, 90.0)
    retrace_fit = clamp(100.0 - abs(retracement_pct - 50.0) * 1.5)
    location_score = clamp(100.0 - loc_distance * 70.0)
    score = clamp(impulse_score * 0.45 + retrace_fit * 0.30 + location_score * 0.25)
    return {"score": round(score, 2), "impulse_pct": round(impulse_pct, 4), "aligned_impulse_pct": round(aligned_impulse, 4), "retracement_pct": round(retracement_pct, 2), "zone_distance_range": round(loc_distance, 4), "bars": len(sample)}


def _v2_support_resistance_score(direction: str, current: float, location: dict[str, Any], *contexts: dict[str, Any]) -> dict[str, Any]:
    """Numeric S/R score combining active FVG/OB/breaker zones, swing pivots, and liquidity."""
    zones: list[Zone] = []
    pivots: list[dict[str, Any]] = []
    wanted_pivot_kind = "LOW" if direction == "BUY" else "HIGH"
    target_low = safe_float(location.get("zone_low"), current)
    target_high = safe_float(location.get("zone_high"), current)
    zone_mid = (target_low + target_high) / 2.0
    for ctx in contexts:
        if not isinstance(ctx, dict):
            continue
        for z in ctx.get("zones", []) or []:
            if isinstance(z, Zone) and z.status in {"ACTIVE", "MITIGATED"}:
                zones.append(z)
        structure = ctx.get("structure")
        if isinstance(structure, StructureSnapshot):
            selected = structure.swing_lows if wanted_pivot_kind == "LOW" else structure.swing_highs
            for p in selected[-16:]:
                distance_pct = abs(p.price - zone_mid) / max(abs(current), EPS) * 100.0
                # Nearer confirmed structural pivots contribute more; distance never rejects the thesis.
                proximity = clamp(100.0 - distance_pct * 45.0, 10.0, 100.0)
                recentness = clamp(100.0 - max(0, len(selected) - 1 - selected.index(p)) * 4.0, 35.0, 100.0)
                pivot_score = clamp(proximity * 0.80 + recentness * 0.20)
                pivots.append({
                    "timeframe": ctx.get("timeframe", "UNKNOWN"),
                    "kind": p.kind,
                    "price": round_price(p.price),
                    "distance_pct": round(distance_pct, 4),
                    "score": round(pivot_score, 2),
                })
    zone_scores = []
    zone_details = []
    for z in zones:
        overlap = _v2_zone_overlap_values(target_low, target_high, z.low, z.high)
        dist = abs(z.midpoint - zone_mid) / max(abs(current), EPS) * 100.0
        formation = str(z.details.get("formation_direction") or "")
        direction_match = formation == ("BULLISH" if direction == "BUY" else "BEARISH")
        kind = str(z.kind).upper()
        is_ob = "OB" in kind or "ORDER_BLOCK" in kind
        is_fvg = "FVG" in kind
        # Explicit weights maintain both OB and FVG evidence within S/R, alongside their refinement score.
        class_bonus = (12.0 if is_ob else 0.0) + (8.0 if is_fvg else 0.0)
        score = clamp(z.strength * 0.60 + (22.0 if overlap > 0 else 0.0) + (8.0 if direction_match else 0.0) + class_bonus - min(25.0, dist * 8.0))
        zone_scores.append(score)
        if len(zone_details) < 8:
            zone_details.append({"kind": z.kind, "source": z.source, "strength": round(z.strength, 2), "overlap": round(overlap, 4), "distance_pct": round(dist, 4), "order_block": is_ob, "fvg": is_fvg, "score": round(score, 2)})
    pool = location.get("pool")
    pool_score = safe_float(getattr(pool, "significance", 50.0), 50.0) if pool is not None else 35.0
    zone_score = max(zone_scores) if zone_scores else 35.0
    pivot_score = max((x["score"] for x in pivots), default=35.0)
    score = clamp(zone_score * 0.45 + pivot_score * 0.30 + pool_score * 0.25)
    return {
        "score": round(score, 2),
        "zone_count": len(zones),
        "swing_level_count": len(pivots),
        "pool_score": round(pool_score, 2),
        "poi_score": round(zone_score, 2),
        "swing_sr_score": round(pivot_score, 2),
        "top_zones": zone_details,
        "top_swing_levels": sorted(pivots, key=lambda x: x["score"], reverse=True)[:6],
    }


def _v2_build_thesis_candidate(
    *,
    direction: str,
    pair: str,
    current: float,
    btc_regime: dict[str, Any],
    pair_regime: dict[str, Any],
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
    m15_structure: StructureSnapshot,
    h4: list[Candle],
    h1: list[Candle],
    m15: list[Candle],
    h1_ctx: dict[str, Any],
    m15_ctx: dict[str, Any],
    h4_ctx: dict[str, Any],
    liquidity_map: dict[str, Any],
    fib_analysis: dict[str, Any],
    location: dict[str, Any],
    rsi_pack: dict[str, Any],
    m15_vlt: dict[str, Any],
    quality_pack: dict[str, Any],
    tick_size: float,
) -> Candidate | None:
    # V2: H4 structure is the authoritative thesis direction. Aggregate
    # D1/H4/H1/M15 regime is retained as a soft quality input so a temporary
    # H1/M15 pullback does not erase a valid H4 continuation thesis.
    refinement = _v2_refine_location(location, direction, h1_ctx, m15_ctx)
    zone_low = safe_float(refinement.get("low"))
    zone_high = safe_float(refinement.get("high"))
    if zone_high <= zone_low:
        return None

    # Opposing unswept liquidity is a scored obstacle, never a directional veto.
    magnet = liquidity_map.get("magnet")
    magnet_conflict = False
    if magnet is not None and not magnet.swept:
        overlaps_magnet = _v2_zone_overlap_values(zone_low, zone_high, magnet.low, magnet.high) > 0.0
        magnet_conflict = bool(
            (direction == "BUY" and magnet.midpoint < current < zone_high and not overlaps_magnet)
            or (direction == "SELL" and magnet.midpoint > current > zone_low and not overlaps_magnet)
        )

    h4_atr_values = atr_series(h4, 14)
    h1_atr_values = atr_series(h1, 14)
    h4_atr = h4_atr_values[-1] if h4_atr_values else max(current * 0.01, EPS)
    h1_atr = h1_atr_values[-1] if h1_atr_values else max(current * 0.005, EPS)
    projection = _v2_project_rsi_to_entry_zone(m15, h4, zone_low, zone_high, direction)
    projection_score = _v2_score_rsi_projection(direction, projection)
    reach_score, reach_details = _v2_entry_reachability_v2(
        direction, current, zone_low, zone_high, h4_atr, safe_float(projection.get("scenario_consensus"), 50.0)
    )
    # Reachability is reflected in its score and distance fields, not a veto.

    confluence_score = _v2_score_liquidity_confluence(location, refinement)
    entry = _v2_select_entry_price(
        direction,
        zone_low,
        zone_high,
        projection,
        current,
        h4_atr,
        confluence_score,
    )
    if tick_size > 0:
        entry = _round_tick(entry, tick_size, "DOWN" if direction == "BUY" else "UP")
    min_gap = max(tick_size, current * 0.0005, EPS)
    if direction == "BUY" and entry >= current:
        entry = current - min_gap
    elif direction == "SELL" and entry <= current:
        entry = current + min_gap
    if tick_size > 0:
        entry = _round_tick(entry, tick_size, "DOWN" if direction == "BUY" else "UP")

    mom_entry = _v2_momentum_entry_price(direction, m15, h4, zone_low, zone_high, current, tick_size)
    if mom_entry is not None and ((direction == "BUY" and mom_entry < current) or (direction == "SELL" and mom_entry > current)):
        entry = mom_entry
    # Entry terlalu jauh hampir pasti expired lewat Price Exp.
    if abs(current - entry) > ENTRY_MAX_DISTANCE_H1_ATR * h1_atr:
        return None

    # Final RSI projection is always calculated against the actual selected
    # entry, not the raw zone midpoint. This prevents a zone that straddles
    # current price from producing an impossible "future" RSI scenario.
    projection = _v2_project_rsi_to_entry_zone(m15, h4, entry, entry, direction)
    projection_score = _v2_score_rsi_projection(direction, projection)
    volume_ctx = _v2_volume_context(direction, h1, h4)
    momentum_gate = _v2_rsi_momentum_gate(direction, projection)

    invalidation = _v2_select_prediction_invalidation(
        direction, entry, h4_structure, h1_structure, liquidity_map,
        list(reversed((h1_ctx.get("sweeps") or []) + (m15_ctx.get("sweeps") or []))),
        h1_atr,
    )
    if invalidation is None:
        return None
    sl = safe_float(invalidation["sl"])

    target_map = _v2_build_target_map(direction, entry, sl, liquidity_map, h4_structure, h1_structure, h1_atr, current=current)
    target = _v2_select_target(target_map, direction)
    if target is None:
        fallback_distance = max(h1_atr * 2.0, abs(entry - safe_float(invalidation.get("sl"), entry)) * 1.5, current * 0.002)
        fallback_tp = entry + fallback_distance if direction == "BUY" else entry - fallback_distance
        target = {"level": fallback_tp, "type": "ATR_SCORE_FALLBACK", "quality": 35.0, "rr": fallback_distance / max(abs(entry - sl), EPS), "major_barriers": [], "priority": 25.0}
    tp = safe_float(target["level"])
    if direction == "BUY" and tp <= entry:
        tp = entry + max(h1_atr * 2.0, current * 0.002)
    elif direction == "SELL" and tp >= entry:
        tp = entry - max(h1_atr * 2.0, current * 0.002)

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / max(risk, EPS)
    if rr < TARGET_MIN_RR:
        # Retain the setup and reflect the RR shortfall numerically in rr_quality.
        alternatives = sorted(target_map, key=lambda x: safe_float(x.get("rr"), 0.0), reverse=True)
        if alternatives and safe_float(alternatives[0].get("rr"), 0.0) > rr:
            target = alternatives[0]
            tp = safe_float(target["level"])
            reward = abs(tp - entry)
            rr = reward / max(risk, EPS)

    price_exp = _v2_price_exp(direction, current, entry, tp, h1_atr)
    # Price Exp is a forecast boundary, not a trade-veto condition.
    if direction == "BUY":
        price_exp = min(max(price_exp, current + max(h1_atr * 0.1, current * 0.0005)), tp - max(h1_atr * 0.05, EPS))
        if price_exp <= current or price_exp >= tp:
            price_exp = current + max((tp - current) * 0.90, current * 0.0005)
    else:
        price_exp = max(min(price_exp, current - max(h1_atr * 0.1, current * 0.0005)), tp + max(h1_atr * 0.05, EPS))
        if price_exp >= current or price_exp <= tp:
            price_exp = current - max((current - tp) * 0.90, current * 0.0005)

    last_mss = latest_event(m15_structure, {"MSS"}, direction=("BULLISH" if direction == "BUY" else "BEARISH"), max_age=THESIS_MAX_TRIGGER_AGE_M15, candle_count=len(m15))
    last_bos = latest_event(m15_structure, {"BOS"}, direction=("BULLISH" if direction == "BUY" else "BEARISH"), max_age=THESIS_MAX_TRIGGER_AGE_M15, candle_count=len(m15))
    trigger_ok = bool(last_mss or last_bos)
    thesis_status = "ACTIVE_TRIGGER_CONFIRMED" if trigger_ok else "WAITING_TRIGGER"
    thesis_type = "BUY_PULLBACK" if direction == "BUY" else "SELL_PULLBACK"
    if projection.get("classification") in {"SUPPORTIVE_BUY_RETRACEMENT", "SUPPORTIVE_SELL_RETRACEMENT"} and (rsi_pack.get("H1") or {}).get("div", {}).get(direction):
        thesis_type += "_RSI_DIVERGENCE"

    structure_score = _v2_structure_alignment_score(direction, btc_regime, pair_regime, h4_structure, h1_structure)
    fib_score = _v2_score_fib_location(location, fib_analysis)
    fib_class = str((fib_analysis.get("impulse_quality") or {}).get("classification") or "")
    refinement_score = _v2_score_refinement(refinement)
    momentum_score = _v2_momentum_current_score(direction, rsi_pack, m15_vlt, m15_structure)
    invalidation_score = _v2_score_invalidation(invalidation, entry)
    target_score = _v2_score_target_quality(target)
    rr_score = _v2_score_rr(rr)
    orderflow_leg = _v2_orderflow_leg_score(direction, m15)
    price_leg = _v2_price_leg_score(direction, m15, zone_low, zone_high, current)
    sr_score = _v2_support_resistance_score(direction, current, location, h4_ctx, h1_ctx, m15_ctx)

    scores = {
        "structure_alignment": round(structure_score, 2),
        "liquidity_confluence": round(confluence_score, 2),
        "fib_location": round(fib_score, 2),
        "fvg_ob_refinement": round(refinement_score, 2),
        "rsi_projection": round(projection_score, 2),
        "momentum_current": round(momentum_score, 2),
        "invalidation_quality": round(invalidation_score, 2),
        "target_quality": round(target_score, 2),
        "rr_quality": round(rr_score, 2),
        "entry_reachability": round(reach_score, 2),
        "volume_confirmation": round(volume_ctx["score"], 2),
        "orderflow_leg": round(orderflow_leg["score"], 2),
        "price_leg": round(price_leg["score"], 2),
        "support_resistance": round(sr_score["score"], 2),
    }
    base = sum(THESIS_WEIGHTS_V2[k] * scores[k] for k in THESIS_WEIGHTS_V2) / 100.0
    penalties: list[dict[str, Any]] = []
    if fib_class == "WEAK":
        # Weak impulse is analytically useful, but it must not score like a
        # clean/struggle impulse. Keep it as a low-confidence candidate rather
        # than fabricating certainty about the retracement depth.
        fib_score = min(fib_score, 58.0)
        scores["fib_location"] = round(fib_score, 2)
        base = min(base, 62.0)
        penalties.append({"type": "WEAK_IMPULSE", "amount": 0.0, "cap": 62.0})
    if not trigger_ok:
        base = min(base - THESIS_WAITING_PENALTY_V2, THESIS_WAITING_CAP_V2)
        penalties.append({"type": "WAITING_TRIGGER", "amount": THESIS_WAITING_PENALTY_V2})
    if momentum_gate["exhaustion_penalty"] > 0:
        base -= momentum_gate["exhaustion_penalty"]
        penalties.append({"type": "RSI_H4_EXHAUSTION", "amount": momentum_gate["exhaustion_penalty"]})
    if momentum_gate["target_penalty"] > 0:
        base -= momentum_gate["target_penalty"]
        penalties.append({"type": "RSI_M15_BELOW_TARGET", "amount": momentum_gate["target_penalty"]})
    if volume_ctx["class"] == "FAKE_MOVE_RISK":
        base -= 10.0
        penalties.append({"type": "FAKE_MOVE_LOW_VOLUME", "amount": 10.0})
    elif volume_ctx["class"] == "WEAK_VOLUME":
        base -= 5.0
        penalties.append({"type": "WEAK_VOLUME", "amount": 5.0})
    if refinement.get("source") == "LIQUIDITY_FIB_ONLY":
        base -= NO_REFINEMENT_PENALTY
        penalties.append({"type": "NO_FVG_OB_REFINEMENT", "amount": NO_REFINEMENT_PENALTY})
    if magnet_conflict:
        base -= 8.0
        penalties.append({"type": "OPPOSING_LIQUIDITY_OBSTACLE", "amount": 8.0})
    projection_class = str(projection.get("classification") or "")
    if projection_class.startswith("WEAK"):
        base -= 10.0
        penalties.append({"type": "WEAK_RSI_PROJECTION", "amount": 10.0})
    if safe_float(projection.get("scenario_consensus"), 0.0) < RSI_PROJECTION_MIN_CONSENSUS:
        base -= 8.0
        penalties.append({"type": "LOW_RSI_PROJECTION_CONSENSUS", "amount": 8.0})
    if invalidation.get("classification") == "TOO_WIDE":
        base -= 18.0
        penalties.append({"type": "WIDE_INVALIDATION", "amount": 18.0})
    if target.get("major_barriers"):
        base -= min(12.0, 4.0 * len(target["major_barriers"]))
        penalties.append({"type": "TARGET_BARRIER", "amount": min(12.0, 4.0 * len(target["major_barriers"]))})
    confidence = clamp(base)
    if not trigger_ok:
        confidence = min(confidence, THESIS_WAITING_CAP_V2)
    # No confidence floor veto: low quality remains an observable, scored candidate.

    if tick_size > 0:
        sl = _round_tick(sl, tick_size, "DOWN" if direction == "BUY" else "UP")
        tp = _round_tick(tp, tick_size, "UP" if direction == "BUY" else "DOWN")
        entry = _round_tick(entry, tick_size, "DOWN" if direction == "BUY" else "UP")
        # Recalculate after tick alignment; a tick-rounded target must still satisfy RR >= 2.
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / max(risk, EPS)
        if rr + 1e-9 < TARGET_MIN_RR:
            # Do not emit a tradable setup that violates the user's hard 1:2 minimum.
            return None
        price_exp = _round_tick(price_exp, tick_size, "UP" if direction == "BUY" else "DOWN")

    if rr + 1e-9 < TARGET_MIN_RR:
        return None

    liquidity_zone = location.get("pool")
    liquidity_evidence = (
        {
            "side": liquidity_zone.side,
            "low": liquidity_zone.low,
            "high": liquidity_zone.high,
            "midpoint": liquidity_zone.midpoint,
            "strength": liquidity_zone.strength,
            "significance": liquidity_zone.significance,
            "timeframe": liquidity_zone.timeframe,
            "touches": liquidity_zone.touches,
            "source_types": liquidity_zone.source_types,
            "swept": liquidity_zone.swept,
            "role": "ENTRY_LIQUIDITY",
        }
        if liquidity_zone is not None else None
    )
    fib_evidence = {
        **location.get("fib", {}),
        "impulse_quality": fib_analysis.get("impulse_quality"),
        "adaptive_reason": fib_analysis.get("reason"),
        "liquidity_overlap_ratio": location.get("overlap_ratio"),
    }
    final_model = PRIMARY_V2_MODEL if trigger_ok else WAITING_V2_MODEL
    entry_reason = (
        f"{pair} {direction}: {thesis_type}. H4 structure {h4_structure.trend}, "
        f"liquidity {liquidity_zone.timeframe if liquidity_zone else '-'} "
        f"{round_price(liquidity_zone.midpoint) if liquidity_zone else '-'} dipadukan dengan Fib {location.get('fib', {}).get('label', '-')}; "
        f"refinement {refinement.get('kind') or 'none'}. "
        f"Projected M15 RSI {projection['at_entry']['M15']['median']:.1f} pada entry zone."
    )
    sl_reason = invalidation["reason"]
    tp_reason = (
        f"TP {round_price(tp)} dipilih dari {target.get('type')} dengan RR {rr:.2f}; "
        f"target quality {target.get('quality', 0):.0f}."
    )
    price_exp_reason = (
        f"Price Exp {price_exp}: retracement thesis dianggap basi jika harga bergerak terlalu jauh "
        f"tanpa menyentuh entry zone sebelum menembus boundary relevansi."
    )
    context_age_m15 = safe_float((quality_pack or {}).get("thesis_age_m15_bars"), 0.0)
    thesis_age_minutes = safe_float((quality_pack or {}).get("thesis_age_minutes"), 0.0)
    thesis_freshness = clamp(100.0 - min(60.0, context_age_m15 * 2.5) - min(25.0, thesis_age_minutes / 12.0))

    evidence = {
        "v2": True,
        "thesis_freshness": {
            "age_m15_bars": round(context_age_m15, 2),
            "age_minutes": round(thesis_age_minutes, 2),
            "score": round(thesis_freshness, 2),
        },
        "thesis": {
            "type": thesis_type,
            "status": thesis_status,
            "btc_regime": btc_regime.get("trend"),
            "pair_regime": pair_regime.get("trend"),
            "structure_signature_h4": _v2_structure_signature(h4_structure),
            "structure_signature_h1": _v2_structure_signature(h1_structure),
        },
        "liquidity_map": liquidity_evidence,
        "fib": fib_evidence,
        "refinement": refinement,
        "projected_rsi": projection,
        "volume_context": volume_ctx,
        "orderflow_leg": orderflow_leg,
        "price_leg": price_leg,
        "support_resistance": sr_score,
        "rsi_momentum_gate": momentum_gate,
        "entry_zone": {
            "low": round_price(zone_low),
            "high": round_price(zone_high),
            "selected": round_price(entry),
            "reachability": reach_details,
        },
        "invalidation": invalidation,
        "target": target,
        "target_map": target_map,
        "planned_rr": round(rr, 4),
        "gates": {
            "waiting": not trigger_ok,
            "vetoes": [],
            "notes": [],
        },
        "current_rsi": rsi_pack,
        "market_quality": quality_pack,
        "liquidity_role": "ENTRY_LIQUIDITY",
        "penalties": penalties,
    }
    notes = [
        f"V2 thesis {thesis_type}",
        f"Fib {location.get('fib', {}).get('label', '-')}",
        f"Projected M15 RSI median {projection['at_entry']['M15']['median']:.1f}",
        f"Volume {volume_ctx['class']} (rvol {volume_ctx['rvol']:.2f}); momentum RSI {momentum_gate['state']}",
        f"Invalidation {invalidation['anchor_source']} @ {invalidation['anchor_level']}",
        f"Target {target['type']} @ {target['level']}",
    ]
    if not trigger_ok:
        notes.append("WAITING_TRIGGER: M15 MSS/BOS belum terkonfirmasi")

    return Candidate(
        direction=direction,
        model=final_model,
        entry=round_price(entry),
        sl=round_price(sl),
        tp=round_price(tp),
        price_exp=round_price(price_exp),
        entry_reason=entry_reason,
        sl_reason=sl_reason,
        tp_reason=tp_reason,
        price_exp_reason=price_exp_reason,
        evidence=evidence,
        scores=scores,
        confidence=round(confidence, 2),
        notes=notes,
        entry_low=round_price(zone_low),
        entry_high=round_price(zone_high),
        thesis_status=thesis_status,
        thesis_type=thesis_type,
        liquidity_zone=liquidity_evidence,
        fib_zone=fib_evidence,
        refinement_zone=refinement,
        projected_rsi=projection,
        invalidation=invalidation,
        target_map=target_map,
    )


def _v2_candidate_summary(candidate: Candidate) -> dict[str, Any]:
    return {
        "model": candidate.model,
        "direction": candidate.direction,
        "entry": candidate.entry,
        "entry_zone_low": candidate.entry_low,
        "entry_zone_high": candidate.entry_high,
        "sl": candidate.sl,
        "tp": candidate.tp,
        "price_exp": candidate.price_exp,
        "planned_rr": round(abs(candidate.tp - candidate.entry) / max(abs(candidate.entry - candidate.sl), EPS), 4),
        "confidence": candidate.confidence,
        "thesis_status": candidate.thesis_status,
        "thesis_type": candidate.thesis_type,
        "scores": candidate.scores,
    }


def _v2_serialize_liquidity_zone(z: LiquidityZone) -> dict[str, Any]:
    return {
        "side": z.side,
        "low": z.low,
        "high": z.high,
        "midpoint": z.midpoint,
        "strength": z.strength,
        "significance": z.significance,
        "timeframe": z.timeframe,
        "touches": z.touches,
        "source_types": z.source_types,
        "first_index": z.first_index,
        "last_index": z.last_index,
        "swept": z.swept,
        "sweep_index": z.sweep_index,
        "distance_pct": z.distance_pct,
        "distance_atr": z.distance_atr,
        "role": z.role,
    }


def _v2_choose_fallback_candidate(
    direction: str,
    current: float,
    h1_atr: float,
    h1_structure: StructureSnapshot,
    liquidity_map: dict[str, Any],
    tick_size: float,
) -> Candidate:
    target_pool = None
    if direction == "BUY":
        for z in liquidity_map.get("above", []):
            if z.significance >= 45 and z.midpoint > current:
                target_pool = z
                break
        entry = current - max(h1_atr * 0.55, current * 0.003)
        sl = entry - max(h1_atr * 1.0, current * 0.008)
        tp = target_pool.midpoint if target_pool else current + h1_atr * 2.5
        tp = max(tp, entry + h1_atr * 2.0)
        exp = min(current + h1_atr * 1.5, tp - h1_atr * 0.2)
    else:
        for z in liquidity_map.get("below", []):
            if z.significance >= 45 and z.midpoint < current:
                target_pool = z
                break
        entry = current + max(h1_atr * 0.55, current * 0.003)
        sl = entry + max(h1_atr * 1.0, current * 0.008)
        tp = target_pool.midpoint if target_pool else current - h1_atr * 2.5
        tp = min(tp, entry - h1_atr * 2.0)
        exp = max(current - h1_atr * 1.5, tp + h1_atr * 0.2)
    if tick_size > 0:
        entry = _round_tick(entry, tick_size, "DOWN" if direction == "BUY" else "UP")
        sl = _round_tick(sl, tick_size, "DOWN" if direction == "BUY" else "UP")
        tp = _round_tick(tp, tick_size, "UP" if direction == "BUY" else "DOWN")
        exp = _round_tick(exp, tick_size, "UP" if direction == "BUY" else "DOWN")
    rr = abs(tp - entry) / max(abs(entry - sl), EPS)
    return Candidate(
        direction=direction,
        model=FALLBACK_V2_MODEL,
        entry=round_price(entry),
        sl=round_price(sl),
        tp=round_price(tp),
        price_exp=round_price(exp),
        entry_reason=f"Fallback V2: structural direction {direction} dipertahankan namun tidak ditemukan thesis location lengkap.",
        sl_reason="Fallback volatility stop; bukan structural prediction invalidation utama.",
        tp_reason=f"Fallback target {target_pool.timeframe if target_pool else 'ATR'}.",
        price_exp_reason="Fallback relevance boundary.",
        evidence={"v2": True, "fallback": True, "gates": {"waiting": True, "vetoes": [], "notes": []}, "planned_rr": round(rr, 3)},
        scores={
            "structure_alignment": 50.0,
            "liquidity_confluence": 45.0,
            "fib_location": 35.0,
            "fvg_ob_refinement": 25.0,
            "rsi_projection": 35.0,
            "momentum_current": 50.0,
            "invalidation_quality": 35.0,
            "target_quality": 45.0,
            "rr_quality": _v2_score_rr(rr),
            "entry_reachability": 55.0,
            "volume_confirmation": 45.0,
            "orderflow_leg": 50.0,
            "price_leg": 45.0,
            "support_resistance": 40.0,
        },
        confidence=THESIS_FALLBACK_CAP_V2,
        notes=["LOW_QUALITY_FALLBACK"],
        entry_low=round_price(entry),
        entry_high=round_price(entry),
        thesis_status="LOW_QUALITY_FALLBACK",
        thesis_type=f"{direction}_FALLBACK",
    )


async def _generate_setup_v2(pair: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    normalized_pair = normalize_pair(pair)
    scan_mode = str(context.get("mode") or "").upper() == "SCAN"
    allow_binance_fallback = bool(context.get("allow_binance_fallback", True)) and not scan_mode
    force_fresh_btc = bool(context.get("force_fresh_btc_regime", False))
    fresh_validation = bool(context.get("validation"))
    m15_age = VALIDATION_MAX_AGE_M15 if fresh_validation else None
    h4_age = VALIDATION_MAX_AGE_H4 if fresh_validation else None
    if not normalized_pair.endswith("USDT"):
        raise ValueError("strategy.py hanya mendukung USDT perpetual.")

    notes = context.get("notes") or []
    notes_context = analyze_notes(notes if isinstance(notes, list) else [])

    # ------------------------------
    # 1) Market data / closed candles
    # ------------------------------
    m15, m15_source = await fetch_series(normalized_pair, M15, M15_CANDLES_REQUIRED, M15_MS, allow_fallback=allow_binance_fallback, max_age_seconds=m15_age)
    pair_h4, pair_h4_source = await fetch_series(normalized_pair, H4, PAIR_H4_CANDLES_REQUIRED, H4_MS, allow_fallback=allow_binance_fallback, max_age_seconds=h4_age)

    btc_cache_key = _scan_cache_key(context) if scan_mode else None
    cached_btc = _SCAN_BTC_REGIME_CACHE.get(btc_cache_key) if btc_cache_key else None
    if cached_btc and not force_fresh_btc:
        btc_h4 = cached_btc.get("candles_h4") or cached_btc.get("candles")
        btc_h1 = cached_btc.get("candles_h1") or []
        btc_h4_source = str(cached_btc.get("sources_by_timeframe", {}).get("H4") or "BINANCE")
        btc_h1_source = str(cached_btc.get("sources_by_timeframe", {}).get("H1") or "BINANCE")
    else:
        btc_h4, btc_h4_source = await fetch_series("BTCUSDT", H4, BTC_H4_CANDLES_REQUIRED, H4_MS, allow_fallback=allow_binance_fallback)
        btc_h1, btc_h1_source = await fetch_series("BTCUSDT", H1, 168, H1_MS, allow_fallback=allow_binance_fallback)
        if btc_cache_key and btc_h4_source == "BINANCE" and btc_h1_source == "BINANCE":
            btc_payload = _btc_regime_payload(btc_h4, btc_h1, {"D1": "BINANCE_DERIVED", "H4": btc_h4_source, "H1": btc_h1_source})
            _SCAN_BTC_REGIME_CACHE[btc_cache_key] = {**btc_payload, "candles_h4": btc_h4, "candles_h1": btc_h1, "created_at": time.time()}
            _prune_scan_btc_cache(btc_cache_key[0])

    h1 = resample_candles(m15, H1_MS)
    if len(h1) < 20:
        h1 = await _fetch_if_short(normalized_pair, H1, H1_MS, 168, allow_fallback=allow_binance_fallback)
    h4_derived = resample_candles(m15, H4_MS)

    scan_snapshot = context.get("scan_market_snapshot") if scan_mode else None
    if not isinstance(scan_snapshot, dict):
        scan_snapshot = {}
    snapshot_age_ms = max(0, int(time.time() * 1000) - int(safe_float(scan_snapshot.get("snapshot_at_ms"), 0.0))) if scan_snapshot.get("snapshot_at_ms") else None
    snapshot_price = safe_float(scan_snapshot.get("price"), 0.0)
    snapshot_fresh = snapshot_age_ms is not None and snapshot_age_ms <= 120_000
    if scan_mode and snapshot_fresh and snapshot_price > 0:
        current = snapshot_price
        price_source = str(scan_snapshot.get("source") or "BINANCE_PUBLIC_TICKER_24H")
    else:
        current, price_source = await fetch_price(normalized_pair, m15_source, allow_fallback=allow_binance_fallback)
    if current <= 0:
        raise RuntimeError("Current price tidak valid.")

    tick_size = safe_float(scan_snapshot.get("tick_size"), 0.0) if scan_mode else 0.0
    if tick_size <= 0:
        tick_size = await get_tick_size(normalized_pair, source="BINANCE")

    # ------------------------------
    # 2) Structure / regime first
    # ------------------------------
    btc_h4_structure = build_structure(btc_h4, SWING_SPAN_H4, min_swing_atr=SWING_MIN_ATR_H4)
    btc_h1_structure = build_structure(btc_h1, SWING_SPAN_H1, min_swing_atr=SWING_MIN_ATR_H1)
    btc_d1 = resample_candles(btc_h4, 24 * 60 * 60 * 1000)
    btc_regime = build_multi_timeframe_regime({"D1": (btc_d1, 2), "H4": (btc_h4, SWING_SPAN_H4), "H1": (btc_h1, SWING_SPAN_H1)})

    h4_structure = build_structure(pair_h4, SWING_SPAN_H4, min_swing_atr=SWING_MIN_ATR_H4)
    h1_structure = build_structure(h1, SWING_SPAN_H1, min_swing_atr=SWING_MIN_ATR_H1)
    m15_structure = build_structure(m15, SWING_SPAN_M15)
    pair_d1 = resample_candles(pair_h4, 24 * 60 * 60 * 1000)
    pair_regime = build_multi_timeframe_regime({
        "D1": (pair_d1, 2),
        "H4": (pair_h4, SWING_SPAN_H4),
        "H1": (h1, SWING_SPAN_H1),
        "M15": (m15, SWING_SPAN_M15),
    })

    btc_trend = str(btc_regime.get("trend") or "RANGE")
    # V2 thesis direction is anchored to H4 structure. Aggregate D1/H4/H1/M15
    # regime remains a soft quality input so a temporary M15/H1 pullback does
    # not erase a valid H4 pullback thesis. BTC remains a soft numerical context only.
    h4_trend = str(h4_structure.trend or "RANGE")
    pair_trend = h4_trend
    # BTC and pair direction are scoring inputs only. Always evaluate BUY and SELL theses.
    allowed_directions = DIRECTIONS_BOTH

    # ------------------------------
    # 3) Technical contexts
    # ------------------------------
    h4_ctx = htf_poi_candidates(pair_h4, h4_structure, DIRECTION_BUY, "H4")
    h1_ctx = htf_poi_candidates(h1, h1_structure, DIRECTION_BUY, "H1")
    m15_ctx = htf_poi_candidates(m15, m15_structure, DIRECTION_BUY, "M15")
    rsi_pack = {
        "H4": build_rsi_context(pair_h4, h4_structure, "H4"),
        "H1": build_rsi_context(h1, h1_structure, "H1"),
        "M15": build_rsi_context(m15, m15_structure, "M15"),
        "BTC": simple_rsi_state(btc_h1),
    }
    quality_pack = market_quality_pack(m15, h1, pair_h4)
    m15_atr_values = atr_series(m15, 14)
    m15_vlt = volume_trend_score(m15, m15_atr_values)

    # ------------------------------
    # 4) Build structure-driven candidates
    # ------------------------------
    candidates: list[Candidate] = []
    directional_regime = directional_regime_label(allowed_directions)
    for direction in allowed_directions:
        # Contradictory structure lowers the score; it never removes the direction.
        fib_analysis = _v2_build_adaptive_fib_zones(pair_h4, h4_structure, direction, "H4")
        if not fib_analysis.get("available"):
            continue
        h1_atr_values = atr_series(h1, 14)
        h1_atr = h1_atr_values[-1] if h1_atr_values else max(current * 0.005, EPS)
        liquidity_map = _v2_build_liquidity_map(
            {"H4": pair_h4, "H1": h1, "M15": m15},
            {"H4": h4_structure, "H1": h1_structure, "M15": m15_structure},
            current,
            direction,
        )
        location_candidates = _v2_liquidity_fib_candidates(liquidity_map, fib_analysis, direction, current, h1_atr)
        if not location_candidates:
            # Still permit a strong Fibonacci location near a significant pool.
            zone = fib_analysis.get("primary_zone") or fib_analysis.get("secondary_zone")
            if zone:
                pools = [
                    p for p in liquidity_map.get("all", [])
                    if p.side == ("SELL_SIDE" if direction == "BUY" else "BUY_SIDE") and not p.swept and p.significance >= 60
                    and abs(p.midpoint - safe_float(zone["midpoint"])) <= 1.0 * h1_atr
                ]
                if pools:
                    pools.sort(key=lambda p: abs(p.midpoint - safe_float(zone["midpoint"])))
                    pool = pools[0]
                    location_candidates.append({
                        "zone_low": safe_float(zone["price_low"]),
                        "zone_high": safe_float(zone["price_high"]),
                        "zone_mid": safe_float(zone["midpoint"]),
                        "fib": zone,
                        "pool": pool,
                        "overlap_ratio": round(_v2_zone_overlap_values(safe_float(zone["price_low"]), safe_float(zone["price_high"]), pool.low, pool.high), 4),
                        "confluence": round(clamp(pool.significance * 0.65), 2),
                    })

        for location in location_candidates:
            candidate = _v2_build_thesis_candidate(
                direction=direction,
                pair=normalized_pair,
                current=current,
                btc_regime=btc_regime,
                pair_regime=pair_regime,
                h4_structure=h4_structure,
                h1_structure=h1_structure,
                m15_structure=m15_structure,
                h4=pair_h4,
                h1=h1,
                m15=m15,
                h1_ctx=h1_ctx,
                m15_ctx=m15_ctx,
                h4_ctx=h4_ctx,
                liquidity_map=liquidity_map,
                fib_analysis=fib_analysis,
                location=location,
                rsi_pack=rsi_pack,
                m15_vlt=m15_vlt,
                quality_pack=quality_pack,
                tick_size=tick_size,
            )
            if candidate:
                candidates.append(candidate)

    # BTC alignment is included in structure_alignment as a soft numerical contribution only.

    # Never drop a direction only because one confluence component is missing.
    # Complete its record with a geometry-safe fallback whose confidence is capped.
    present_directions = {c.direction for c in candidates}
    for missing_direction in allowed_directions:
        if missing_direction in present_directions:
            continue
        fallback_atr_values = atr_series(h1, 14)
        fallback_atr = fallback_atr_values[-1] if fallback_atr_values else max(current * 0.005, EPS)
        fallback_liquidity = _v2_build_liquidity_map(
            {"H4": pair_h4, "H1": h1, "M15": m15},
            {"H4": h4_structure, "H1": h1_structure, "M15": m15_structure},
            current,
            missing_direction,
        )
        fallback = _v2_choose_fallback_candidate(
            missing_direction, current, fallback_atr, h1_structure, fallback_liquidity, tick_size
        )
        fallback.confidence = min(fallback.confidence, FALLBACK_CONFIDENCE_CAP)
        fallback.evidence.setdefault("scoring_policy", "NO_ANALYTICAL_VETO_FALLBACK")
        fallback.notes.append("Fallback scored because one or more confluences were unavailable; not an analytical rejection.")
        candidates.append(fallback)

    if not candidates:
        # This is reachable only if fallback construction itself fails.
        direction = allowed_directions[0] if allowed_directions else ("BUY" if pair_trend != "BEARISH" else "SELL")
        h1_atr = atr_series(h1, 14)[-1] if h1 else max(current * 0.005, EPS)
        liquidity_map = _v2_build_liquidity_map(
            {"H4": pair_h4, "H1": h1, "M15": m15},
            {"H4": h4_structure, "H1": h1_structure, "M15": m15_structure},
            current,
            direction,
        )
        best = _v2_choose_fallback_candidate(direction, current, h1_atr, h1_structure, liquidity_map, tick_size)
        candidates = [best]
    else:
        candidates.sort(key=lambda c: (c.confidence, c.scores.get("liquidity_confluence", 0.0), c.scores.get("rsi_projection", 0.0)), reverse=True)
        best = candidates[0]

    # Final geometry validation after tick normalization.
    if not _ensure_price_geometry(best, current):
        h1_atr = atr_series(h1, 14)[-1] if h1 else max(current * 0.005, EPS)
        liquidity_map = _v2_build_liquidity_map(
            {"H4": pair_h4, "H1": h1, "M15": m15},
            {"H4": h4_structure, "H1": h1_structure, "M15": m15_structure},
            current,
            best.direction,
        )
        best = _v2_choose_fallback_candidate(best.direction, current, h1_atr, h1_structure, liquidity_map, tick_size)
        candidates = [best] + [c for c in candidates if c is not best]

    risk = abs(best.entry - best.sl)
    reward = abs(best.tp - best.entry)
    planned_rr = reward / max(risk, EPS)

    # Build final analysis using the actual selected candidate's direction/location.
    final_liquidity_map = _v2_build_liquidity_map(
        {"H4": pair_h4, "H1": h1, "M15": m15},
        {"H4": h4_structure, "H1": h1_structure, "M15": m15_structure},
        current,
        best.direction,
    )
    final_fib = _v2_build_adaptive_fib_zones(pair_h4, h4_structure, best.direction, "H4")
    final_target_map = best.target_map or []
    final_projection = best.projected_rsi or {}
    target_selected = None
    if final_target_map:
        target_selected = min(
            final_target_map,
            key=lambda x: abs(safe_float(x.get("level"), 0.0) - safe_float(best.tp, 0.0)),
        )
    selected_entry_zone = {
        "low": best.entry_low,
        "high": best.entry_high,
        "selected": best.entry,
        "width_pct": round((best.entry_high - best.entry_low) / max(abs(best.entry), EPS) * 100.0, 4),
    }
    liquidity_serialized = {
        "above": [_v2_serialize_liquidity_zone(z) for z in final_liquidity_map.get("above", [])[:12]],
        "below": [_v2_serialize_liquidity_zone(z) for z in final_liquidity_map.get("below", [])[:12]],
        "significant_above": [_v2_serialize_liquidity_zone(z) for z in final_liquidity_map.get("significant_above", [])[:10]],
        "significant_below": [_v2_serialize_liquidity_zone(z) for z in final_liquidity_map.get("significant_below", [])[:10]],
        "magnet": _v2_serialize_liquidity_zone(final_liquidity_map["magnet"]) if final_liquidity_map.get("magnet") else None,
    }
    analysis = {
        "version": "V2",
        "macro": {
            "pair": normalized_pair,
            "btc_regime": btc_regime,
            "btc_h4_structure_trend": btc_h4_structure.trend,
            "btc_aggregate_regime": btc_regime.get("aggregate_regime", btc_regime.get("regime")),
            "btc_h4": _structure_summary(btc_h4_structure, btc_h4),
            "pair_regime": pair_regime,
            "pair_h4": _structure_summary(h4_structure, pair_h4),
            "macro_bias": btc_trend,
            "allowed_directions": list(allowed_directions),
            "directional_regime": directional_regime,
        },
        "pair_structure": {
            "H4": _structure_summary(h4_structure, pair_h4),
            "H1": _structure_summary(h1_structure, h1),
            "M15": _structure_summary(m15_structure, m15),
            "signature_h4": _v2_structure_signature(h4_structure),
            "signature_h1": _v2_structure_signature(h1_structure),
        },
        "thesis": {
            "type": best.thesis_type,
            "status": best.thesis_status,
            "reason": best.entry_reason,
            "freshness": (best.evidence.get("thesis_freshness") if isinstance(best.evidence, dict) else None),
        },
        "liquidity_map": liquidity_serialized,
        "fib_analysis": {
            "H4": {
                **{k: v for k, v in final_fib.items() if k != "fib"},
                "current_price": current,
            },
        },
        "refinement": best.refinement_zone,
        "rsi": rsi_pack,
        "rsi_projection": final_projection,
        "entry_zone": selected_entry_zone,
        "invalidation": best.invalidation,
        "target_map": {
            "candidates": final_target_map,
            "selected": target_selected,
        },
        "risk": {
            "risk": round_price(risk),
            "reward": round_price(reward),
            "planned_rr": round(planned_rr, 4),
            "risk_pct": round(risk / max(best.entry, EPS) * 100.0, 4),
            "risk_h1_atr": round(risk / max(atr_series(h1, 14)[-1] if h1 else 1.0, EPS), 4),
            "classification": (best.invalidation or {}).get("classification"),
        },
        "confidence": {
            "score": best.confidence,
            "components": best.scores,
            "weights": THESIS_WEIGHTS_V2,
            "penalties": (best.evidence.get("penalties") if isinstance(best.evidence, dict) else []) or [],
            "meaning": "Kualitas internal thesis 0-100; bukan probabilitas profit.",
        },
        "score_details": {
            "orderflow_leg": best.evidence.get("orderflow_leg"),
            "price_leg": best.evidence.get("price_leg"),
            "support_resistance": best.evidence.get("support_resistance"),
            "liquidity_pool": best.evidence.get("liquidity_map"),
            "fvg_order_block_refinement": best.evidence.get("refinement"),
            "rsi_momentum": best.evidence.get("rsi_momentum_gate"),
            "current_rsi": best.evidence.get("current_rsi"),
            "projected_rsi": best.evidence.get("projected_rsi"),
            "volume_context": best.evidence.get("volume_context"),
        },
        "candidate_set": [_v2_candidate_summary(c) for c in sorted(candidates, key=lambda c: c.confidence, reverse=True)],
        "price_context": {
            "current": round_price(current),
            "atr14_m15": round_price(m15_atr_values[-1] if m15_atr_values else 0.0),
            "current_distance_to_entry_pct": round(distance_pct(current, best.entry), 4),
            "entry_distance_to_exp_pct": round(distance_pct(best.entry, best.price_exp), 4),
            "entry_zone_timeframe": "H4+H1+M15",
        },
        "vlt": m15_vlt,
        "market_quality": quality_pack,
        "smc": {
            "sweeps_recent": (m15_ctx.get("sweeps") or [])[-12:],
            "fvg_recent": _zone_summary(m15_ctx.get("fvg") or [], current, m15_atr_values[-1] if m15_atr_values else 1.0, len(m15)),
            "order_blocks": _zone_summary(m15_ctx.get("obs") or [], current, m15_atr_values[-1] if m15_atr_values else 1.0, len(m15)),
            "breakers": _zone_summary(m15_ctx.get("breakers") or [], current, m15_atr_values[-1] if m15_atr_values else 1.0, len(m15)),
        },
    }

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
            "scan_mode": scan_mode,
            "data_policy": "BINANCE_PUBLIC_ONLY" if scan_mode else "BINANCE_ONLY",
            "fallback_used": False,
            "directional_bias": btc_trend,
            "directional_regime": directional_regime,
            "allowed_directions": list(allowed_directions),
            "sources_by_timeframe": {
                "pair_m15": m15_source,
                "pair_h4": pair_h4_source,
                "btc_h4": btc_h4_source,
                "btc_h1": btc_h1_source,
                "current_price": price_source,
            },
            "timeframe": "15m",
            "candles_requested": M15_CANDLES_REQUIRED,
            "candles_used": len(m15),
            "closed_candles_only": True,
            "pair_h4_candles": len(pair_h4),
            "btc_h4_candles": len(btc_h4),
            "btc_h1_candles": len(btc_h1),
            "h1_derived_candles": len(h1),
            "h4_derived_candles": len(h4_derived),
            "data_asof": {
                "M15": m15[-1].time_ms if m15 else None,
                "H1": h1[-1].time_ms if h1 else None,
                "H4": pair_h4[-1].time_ms if pair_h4 else None,
                "BTC_H4": btc_h4[-1].time_ms if btc_h4 else None,
                "BTC_H1": btc_h1[-1].time_ms if btc_h1 else None,
            },
            "tick_size": round_price(tick_size) if tick_size else None,
            "tick_size_source": "BINANCE" if scan_mode else ("BINANCE" if m15_source == "BINANCE" else "BINANCE"),
            "engine": STRATEGY_ENGINE,
        },
        "strategy": {
            "name": STRATEGY_NAME,
            "version": STRATEGY_VERSION,
            "engine": STRATEGY_ENGINE,
            "architecture": "Binance-only closed candles -> H4 structure -> orderflow leg/price leg -> liquidity pools + adaptive Fibonacci + FVG/OB + support-resistance -> RSI momentum -> numeric confidence -> score-based validation",
            "confidence_semantics": "weighted_thesis_quality_score_not_profit_probability",
            "scan_contract": "analyze_scan_structure -> generate_setup -> score_based_validate_setup",
            "validator_tolerance": "confidence changes are values, not veto; only stale data or unsafe geometry blocks execution",
            "directional_rule": "BUY and SELL are always evaluated; BTC and pair context contribute scores only.",
        },
    }


# ============================================================================
# BLUEPRINT V2 PUBLIC ANALYSIS HELPERS
# ============================================================================


def build_liquidity_map(
    candles_by_tf: dict[str, list[Candle]],
    structures: dict[str, StructureSnapshot],
    current: float,
    direction: str,
) -> dict[str, Any]:
    """Blueprint-facing name for the V2 liquidity map engine."""
    return _v2_build_liquidity_map(candles_by_tf, structures, current, direction)


def cluster_liquidity_levels(
    raw_levels: list[dict[str, Any]],
    candles: list[Candle],
    timeframe: str,
    current: float,
) -> list[LiquidityZone]:
    """Blueprint-facing name for liquidity clustering."""
    return _v2_cluster_liquidity_levels(raw_levels, candles, timeframe, current)


def assess_impulse_quality(
    candles: list[Candle],
    structure: StructureSnapshot,
    direction: str,
    fib: FibonacciContext,
) -> dict[str, Any]:
    return _v2_assess_impulse_quality(candles, structure, direction, fib)


def build_adaptive_fib_zones(
    candles: list[Candle],
    structure: StructureSnapshot,
    direction: str,
    timeframe: str = "H4",
) -> dict[str, Any]:
    return _v2_build_adaptive_fib_zones(candles, structure, direction, timeframe)


def liquidity_fib_confluence(
    liquidity_map: dict[str, Any],
    fib_analysis: dict[str, Any],
    direction: str,
    current: float,
    h1_atr: float,
) -> list[dict[str, Any]]:
    return _v2_liquidity_fib_candidates(liquidity_map, fib_analysis, direction, current, h1_atr)


def find_refinement_confluence(
    location: dict[str, Any],
    direction: str,
    h1_ctx: dict[str, Any],
    m15_ctx: dict[str, Any],
) -> dict[str, Any]:
    return _v2_refine_location(location, direction, h1_ctx, m15_ctx)


def project_rsi_to_entry_zone(
    m15: list[Candle],
    h4: list[Candle],
    entry_low: float,
    entry_high: float,
    direction: str,
) -> dict[str, Any]:
    return _v2_project_rsi_to_entry_zone(m15, h4, entry_low, entry_high, direction)


def classify_projected_rsi(direction: str, h4: dict[str, Any], m15: dict[str, Any]) -> str:
    return _v2_classify_projected_rsi(direction, h4, m15)


def find_invalidation_candidates(
    direction: str,
    entry: float,
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
    liquidity_map: dict[str, Any],
    sweeps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _v2_find_invalidation_candidates(direction, entry, h4_structure, h1_structure, liquidity_map, sweeps)


def select_prediction_invalidation(
    direction: str,
    entry: float,
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
    liquidity_map: dict[str, Any],
    sweeps: list[dict[str, Any]],
    h1_atr: float,
) -> dict[str, Any] | None:
    return _v2_select_prediction_invalidation(direction, entry, h4_structure, h1_structure, liquidity_map, sweeps, h1_atr)


def build_target_map(
    direction: str,
    entry: float,
    sl: float,
    liquidity_map: dict[str, Any],
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot,
    h1_atr: float,
    current: float | None = None,
) -> list[dict[str, Any]]:
    return _v2_build_target_map(direction, entry, sl, liquidity_map, h4_structure, h1_structure, h1_atr, current=current)


def score_target_quality(target: dict[str, Any] | None) -> float:
    return _v2_score_target_quality(target)


def score_liquidity_zone(zone: LiquidityZone | None) -> float:
    """Return the final V2 significance score for a liquidity zone."""
    return clamp(safe_float(zone.significance, 0.0)) if zone is not None else 0.0


def classify_liquidity_role(
    zones: list[LiquidityZone],
    direction: str,
    current: float,
    entry_low: float | None = None,
    entry_high: float | None = None,
) -> list[LiquidityZone]:
    """Assign V2 liquidity roles in-place and return the same list."""
    _v2_mark_liquidity_roles(zones, direction, current, entry_low, entry_high)
    return zones


def simulate_rsi_path(
    candles: list[Candle],
    target: float,
    steps: int,
    scenario: str = "NORMAL_RETRACE",
) -> float:
    return _v2_simulate_rsi_path(candles, target, steps, scenario)


def score_entry_zone(
    location: dict[str, Any],
    refinement: dict[str, Any],
    projection: dict[str, Any],
    invalidation: dict[str, Any] | None,
    current: float,
    direction: str,
    h4_atr: float,
) -> float:
    """Standalone entry-zone quality score for audits/backtests."""
    zone_low = safe_float(refinement.get("low"), location.get("zone_low", 0.0))
    zone_high = safe_float(refinement.get("high"), location.get("zone_high", 0.0))
    if zone_high <= zone_low:
        return 0.0
    confluence = _v2_score_liquidity_confluence(location, refinement)
    rsi_score = _v2_score_rsi_projection(direction, projection)
    invalidation_score = _v2_score_invalidation(invalidation, (zone_low + zone_high) / 2.0)
    reach_score, _ = _v2_entry_reachability_v2(
        direction, current, zone_low, zone_high, h4_atr,
        safe_float(projection.get("scenario_consensus"), 50.0),
    )
    return clamp(
        confluence * 0.35
        + rsi_score * 0.30
        + invalidation_score * 0.20
        + reach_score * 0.15
    )


def build_thesis(
    direction: str,
    btc_regime: dict[str, Any],
    pair_regime: dict[str, Any],
    h4_structure: StructureSnapshot,
    h1_structure: StructureSnapshot | None = None,
) -> dict[str, Any]:
    """Blueprint-facing thesis object used by audits and integration tooling."""
    want = "BULLISH" if direction == DIRECTION_BUY else "BEARISH"
    btc_trend = str(btc_regime.get("trend") or "RANGE").upper()
    pair_trend = str(pair_regime.get("trend") or h4_structure.trend or "RANGE").upper()
    h1_trend = str(h1_structure.trend if h1_structure is not None else "RANGE").upper()
    macro_aligned = btc_trend in {want, "RANGE"}
    pair_aligned = pair_trend in {want, "RANGE"}
    h4_aligned = h4_structure.trend in {want, "RANGE"}
    return {
        "direction": direction,
        "valid": bool(pair_aligned and h4_aligned),
        "type": f"H4_{'BULLISH' if direction == DIRECTION_BUY else 'BEARISH'}_PULLBACK",
        "btc_trend": btc_trend,
        "pair_trend": pair_trend,
        "h4_trend": h4_structure.trend,
        "h1_trend": h1_trend,
        "macro_aligned": macro_aligned,
        "pair_aligned": pair_aligned,
        "h4_aligned": h4_aligned,
        "structure_signature_h4": _v2_structure_signature(h4_structure),
        "structure_signature_h1": _v2_structure_signature(h1_structure) if h1_structure is not None else None,
    }


def score_thesis_quality(candidate: Candidate) -> float:
    """Recompute V2 thesis quality from stored component scores when available."""
    if not candidate.scores:
        return float(candidate.confidence)
    base = sum(
        THESIS_WEIGHTS_V2[name] * safe_float(candidate.scores.get(name), 0.0)
        for name in THESIS_WEIGHTS_V2
    ) / 100.0
    if candidate.thesis_status == "WAITING_TRIGGER":
        base = min(base, THESIS_WAITING_CAP_V2)
    return clamp(base)


def validate_thesis_consistency(candidate: Candidate) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if candidate.direction not in {DIRECTION_BUY, DIRECTION_SELL}:
        errors.append("invalid direction")
    if candidate.entry_low > candidate.entry_high:
        errors.append("entry zone inverted")
    if not (candidate.entry_low <= candidate.entry <= candidate.entry_high):
        errors.append("entry outside entry zone")
    if not candidate.liquidity_zone:
        errors.append("missing liquidity thesis")
    if not candidate.fib_zone:
        errors.append("missing fibonacci thesis")
    if not candidate.invalidation:
        errors.append("missing prediction invalidation")
    if candidate.direction == DIRECTION_BUY:
        if not (candidate.sl < candidate.entry < candidate.tp):
            errors.append("BUY geometry invalid")
        if safe_float((candidate.invalidation or {}).get("anchor_level"), candidate.entry) >= candidate.entry:
            errors.append("BUY invalidation on wrong side")
    else:
        if not (candidate.tp < candidate.entry < candidate.sl):
            errors.append("SELL geometry invalid")
        if safe_float((candidate.invalidation or {}).get("anchor_level"), candidate.entry) <= candidate.entry:
            errors.append("SELL invalidation on wrong side")
    risk = abs(candidate.entry - candidate.sl)
    reward = abs(candidate.tp - candidate.entry)
    rr = reward / max(risk, EPS)
    if rr < TARGET_MIN_RR:
        errors.append("RR below minimum")
    if candidate.model != FALLBACK_V2_MODEL and not 0.0 <= candidate.confidence <= 100.0:
        errors.append("confidence outside 0-100")
    return (not errors, errors)


# ============================================================================
# PUBLIC CONTRACT
# ============================================================================

# ============================================================================
# CANDIDATE CONTRACT (schema v1): state, reason codes, geometri, biaya, BTC
# ============================================================================


def classify_btc_alignment(btc_trend: str | None, direction: str | None, btc_confidence: Any = None) -> str:
    """ALIGNED / NEUTRAL / COUNTER_TREND / HIGH_RISK_CONFLICT."""
    trend = str(btc_trend or "RANGE").upper()
    side = str(direction or "").upper()
    if trend not in {"BULLISH", "BEARISH"} or side not in {"BUY", "SELL"}:
        return "NEUTRAL"
    if (trend == "BULLISH" and side == "BUY") or (trend == "BEARISH" and side == "SELL"):
        return "ALIGNED"
    if btc_confidence is not None and safe_float(btc_confidence, 0.0) >= BTC_EXTREME_CONFIDENCE:
        return "HIGH_RISK_CONFLICT"
    return "COUNTER_TREND"


def validate_setup_geometry(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    zone_low: float | None = None,
    zone_high: float | None = None,
    tick_size: float = 0.0,
) -> list[dict[str, Any]]:
    """Daftar kegagalan geometri (hard gate). Kosong = valid."""
    issues: list[dict[str, Any]] = []

    def add(code: str, text: str) -> None:
        issues.append({"code": code, "text": text, "hard": True})

    side = str(direction or "").upper()
    if side not in {"BUY", "SELL"} or not all(math.isfinite(v) and v > 0 for v in (entry, sl, tp)):
        add("PRICE_INVALID", "entry/SL/TP tidak valid atau arah tidak dikenal.")
        return issues
    if side == "BUY" and not (sl < entry < tp):
        add("GEOMETRY_INVALID", f"BUY wajib SL < entry < TP; dapat SL={sl}, entry={entry}, TP={tp}.")
    if side == "SELL" and not (tp < entry < sl):
        add("GEOMETRY_INVALID", f"SELL wajib TP < entry < SL; dapat TP={tp}, entry={entry}, SL={sl}.")
    if zone_low is not None and zone_high is not None and zone_low > 0 and zone_high > 0:
        if zone_low > zone_high:
            add("ZONE_INVERTED", f"Zona entry terbalik: low={zone_low} > high={zone_high}.")
        else:
            tol = max(tick_size, entry * 0.001)
            if entry < zone_low - tol or entry > zone_high + tol:
                add("ENTRY_OUTSIDE_ZONE", f"Entry {entry} di luar zona [{zone_low}, {zone_high}].")
            mid = (zone_low + zone_high) / 2.0
            if mid > 0 and (zone_high - zone_low) / mid * 100.0 > ENTRY_ZONE_MAX_WIDTH_PCT:
                add("ZONE_TOO_WIDE", "Zona entry terlalu lebar untuk satu harga representatif.")
    return issues


def estimate_after_cost_rr(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    *,
    fee_pct: float | None = None,
    slippage_pct: float | None = None,
    funding_pct: float = 0.0,
) -> dict[str, Any]:
    """RR gross vs setelah biaya (fee + slippage, entry dan exit)."""
    fee = EST_FEE_PCT_PER_SIDE if fee_pct is None else fee_pct
    slip = EST_SLIPPAGE_PCT_PER_SIDE if slippage_pct is None else slippage_pct
    c = (fee + slip) / 100.0
    ce = (EST_ENTRY_FEE_PCT + EST_ENTRY_SLIPPAGE_PCT) / 100.0
    buy = str(direction or "").upper() == "BUY"
    risk = (entry - sl) if buy else (sl - entry)
    reward = (tp - entry) if buy else (entry - tp)
    if entry <= 0 or risk <= 0 or reward <= 0:
        return {"valid": False, "gross_rr": 0.0, "after_cost_rr": 0.0}
    funding = entry * funding_pct / 100.0
    cost_tp = entry * ce + tp * c + funding
    cost_sl = entry * ce + sl * c + funding
    net_reward = reward - cost_tp
    net_risk = risk + cost_sl
    return {
        "valid": True,
        "gross_rr": reward / risk,
        "after_cost_rr": (net_reward / net_risk) if net_reward > 0 else 0.0,
        "risk": risk,
        "reward": reward,
        "cost_at_tp": cost_tp,
        "cost_at_sl": cost_sl,
        "round_trip_cost_pct": (c + ce) * 100.0,
        "assumptions": {"fee_pct_per_side": fee, "slippage_pct_per_side": slip, "entry_fee_pct": EST_ENTRY_FEE_PCT, "entry_slippage_pct": EST_ENTRY_SLIPPAGE_PCT, "funding_pct": funding_pct},
    }


_DATA_STALE_FACTOR = {"M15": 3, "H1": 3, "H4": 2}
_TF_MS = {"M15": 15 * 60_000, "H1": 60 * 60_000, "H4": 4 * 60 * 60_000}


def _data_quality(data_asof: dict[str, Any] | None, now_ms: int | None = None) -> dict[str, Any]:
    """Umur candle closed terbaru per timeframe; stale bila terlalu tertinggal."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    ages: dict[str, float | None] = {}
    stale: list[str] = []
    for tf, span in _TF_MS.items():
        ts = (data_asof or {}).get(tf)
        if not ts:
            ages[tf] = None
            continue
        age = (now_ms - (int(ts) + span)) / 1000.0
        ages[tf] = round(max(0.0, age), 1)
        if age > _DATA_STALE_FACTOR[tf] * span / 1000.0:
            stale.append(tf)
    return {"newest_closed_age_seconds": ages, "stale_timeframes": stale, "fresh": not stale}


def thesis_weights_total() -> float:
    return float(sum(THESIS_WEIGHTS_V2.values()))


if abs(thesis_weights_total() - 100.0) > 1e-6:
    _log.error("THESIS_WEIGHTS_V2 total %.4f != 100", thesis_weights_total())


def _confidence_breakdown(out: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    comps = out.get("confidence_components") or {}
    raw = sum(THESIS_WEIGHTS_V2[k] * safe_float(comps.get(k), 0.0) for k in THESIS_WEIGHTS_V2) / 100.0
    conf = safe_float(out.get("confidence"), 0.0)
    penalties = ((analysis.get("confidence") or {}).get("penalties")) or []
    return {
        "scoring_version": SCORING_FORMULA_VERSION,
        "weights_total": thesis_weights_total(),
        "components": comps,
        "raw_weighted_score": round(raw, 2),
        "penalties": penalties,
        "caps": {
            "waiting": THESIS_WAITING_CAP_V2,
            "veto": THESIS_VETO_CAP_V2,
            "fallback": THESIS_FALLBACK_CAP_V2,
            "min_quality": THESIS_MIN_QUALITY_V2,
        },
        "final": round(conf, 2),
        "delta_from_raw": round(conf - raw, 2),
        "meaning": "thesis_quality_score_not_profit_probability",
    }


def _enrich_setup_contract(setup: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tambahkan field kontrak v1 secara kompatibel; tidak mengubah field lama."""
    out = dict(setup)
    analysis = out.get("analysis") if isinstance(out.get("analysis"), dict) else {}
    data = out.get("data") if isinstance(out.get("data"), dict) else {}
    pair = str(out.get("pair") or "")
    direction = str(out.get("direction") or "").upper()
    entry, sl, tp = (safe_float(out.get(k)) for k in ("entry", "sl", "tp"))
    zone = analysis.get("entry_zone") if isinstance(analysis.get("entry_zone"), dict) else {}
    zlow = safe_float(zone.get("low"), entry)
    zhigh = safe_float(zone.get("high"), entry)
    tick = safe_float(data.get("tick_size"), 0.0)
    status = str((analysis.get("thesis") or {}).get("status") or "")
    conf = safe_float(out.get("confidence"), 0.0)

    gates: list[dict[str, Any]] = validate_setup_geometry(direction, entry, sl, tp, zlow, zhigh, tick)
    risk_block = analysis.get("risk") if isinstance(analysis.get("risk"), dict) else {}
    gross_rr = safe_float(risk_block.get("planned_rr"), 0.0)
    if gross_rr + 1e-9 < TARGET_MIN_RR:
        gates.append({"code": "RR_BELOW_MIN", "text": f"RR gross {gross_rr:.2f} < {TARGET_MIN_RR}; direkam sebagai skor.", "hard": False})
    cost = estimate_after_cost_rr(direction, entry, sl, tp)
    if MIN_RR_AFTER_COSTS > 0 and cost.get("valid") and cost["after_cost_rr"] < MIN_RR_AFTER_COSTS:
        gates.append({
            "code": "RR_AFTER_COST_BELOW_MIN",
            "text": f"RR setelah biaya {cost['after_cost_rr']:.2f} < {MIN_RR_AFTER_COSTS}; biaya terlalu besar, ditolak.",
            "hard": True,
        })

    data_asof = data.get("data_asof") if isinstance(data.get("data_asof"), dict) else {}
    quality = _data_quality(data_asof) if data_asof else {"fresh": True, "stale_timeframes": [], "newest_closed_age_seconds": {}}
    if not quality["fresh"]:
        gates.append({"code": "DATA_STALE", "text": f"Data basi: {','.join(quality['stale_timeframes'])}.", "hard": True})

    btc_trend = data.get("directional_bias")
    btc_conf = (((analysis.get("macro") or {}).get("btc_regime")) or {}).get("confidence")
    alignment = "NEUTRAL" if pair == "BTCUSDT" else classify_btc_alignment(btc_trend, direction, btc_conf)
    if alignment in {"COUNTER_TREND", "HIGH_RISK_CONFLICT"}:
        gates.append({"code": "BTC_CONTEXT_COUNTER_TREND", "text": "Arah berbeda dari BTC H4; dicatat sebagai konteks skor, bukan veto.", "hard": False})

    current = safe_float(out.get("price_now_reference"), 0.0)
    if zlow > 0 and zhigh > 0 and current > 0:
        price_vs_zone = "IN_ZONE" if zlow <= current <= zhigh else ("ABOVE_ZONE" if current > zhigh else "BELOW_ZONE")
    else:
        price_vs_zone = "UNKNOWN"

    hard = [g for g in gates if g.get("hard")]
    safety_geometry_codes = {"PRICE_INVALID", "GEOMETRY_INVALID", "ZONE_INVERTED", "ENTRY_OUTSIDE_ZONE", "ZONE_TOO_WIDE"}
    geometry_errors = [g for g in hard if g.get("code") in safety_geometry_codes]
    if any(g["code"] == "DATA_STALE" for g in hard):
        state, reason = "DATA_UNAVAILABLE", "DATA_STALE"
    elif geometry_errors:
        state, reason = "REJECTED_HARD", geometry_errors[0]["code"]  # execution safety only
    elif any(g["code"] == "RR_AFTER_COST_BELOW_MIN" for g in hard):
        state, reason = "REJECTED_QUALITY", "RR_AFTER_COST_BELOW_MIN"
    elif "WAITING" in status:
        state, reason = "WAITING_TRIGGER", "MOMENTUM_NOT_READY" if status == "WAITING_MOMENTUM" else "TRIGGER_NOT_CONFIRMED"
        gates.append({"code": reason, "text": "Setup belum memiliki trigger penuh; kualitas tetap direpresentasikan oleh confidence.", "hard": False})
    else:
        state = "READY"
        reason = "PRICE_IN_ZONE" if price_vs_zone == "IN_ZONE" else ("LOW_SCORE_FALLBACK" if status == "LOW_QUALITY_FALLBACK" else "PRICE_NOT_IN_ZONE")
    # RR gross, kualitas komponen, RSI exhaustion dan BTC tetap angka; hanya RR setelah biaya yang menolak.

    sig_h4 = ((analysis.get("pair_structure") or {}).get("signature_h4"))
    anchor = f"{pair}|{direction}|{STRATEGY_VERSION}|{sig_h4!r}|{round(zlow, 6)}|{round(zhigh, 6)}"
    thesis_signature = hashlib.sha1(anchor.encode("utf-8")).hexdigest()[:20]

    src = str(data.get("source") or "")
    out.update({
        "schema_version": STRATEGY_SCHEMA_VERSION,
        "candidate_state": state,
        "state_reason": reason,
        "gate_reasons": gates,
        "entry_zone": {"low": zlow, "high": zhigh, "selected": entry},
        "price_vs_zone": price_vs_zone,
        "planned_rr_gross": round(gross_rr, 4),
        "planned_rr_after_costs": round(cost["after_cost_rr"], 4) if cost.get("valid") else None,
        "cost_assumptions": cost.get("assumptions"),
        "data_asof": data_asof,
        "data_quality": quality,
        "setup_archetype": str((analysis.get("thesis") or {}).get("type") or PRIMARY_V2_MODEL),
        "analysis_source": "BINANCE_PUBLIC" if src == "BINANCE" else (src or "UNKNOWN"),
        "btc_alignment": alignment,
        "thesis_signature": thesis_signature,
        "scoring_version": SCORING_FORMULA_VERSION,
        "confidence_breakdown": _confidence_breakdown(out, analysis),
        "risk": {
            "entry_to_sl_distance": round_price(abs(entry - sl)) if entry and sl else None,
            "risk_pct": risk_block.get("risk_pct"),
            "risk_h1_atr": risk_block.get("risk_h1_atr"),
            "estimated_round_trip_cost_pct": round(cost.get("round_trip_cost_pct", 0.0), 4) if cost.get("valid") else None,
            "invalidation_anchor": ((analysis.get("invalidation") or {}).get("anchor") or (analysis.get("invalidation") or {}).get("reason")),
        },
    })
    return out


async def generate_setup(pair: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Public strategy contract; delegates to the Structural Prediction Engine V2."""
    return _enrich_setup_contract(await _generate_setup_v2(pair, context), context)


# ============================================================================
# SCAN / VALIDATION CONTRACTS
# ============================================================================


def _scan_cache_key(context: dict[str, Any] | None) -> tuple[str, str] | None:
    context = context or {}
    cycle = context.get("scan_cycle")
    session = str(context.get("session_id") or "")
    if cycle in (None, ""):
        return None
    return session, str(cycle)


def _prune_scan_btc_cache(session_id: str, keep_cycles: int = 4) -> None:
    """Keep only the latest few BTC cycle snapshots for bounded memory use."""
    session_id = str(session_id or "")
    if not session_id:
        return
    items = []
    for key, value in _SCAN_BTC_REGIME_CACHE.items():
        if key[0] != session_id:
            continue
        try:
            cycle = int(key[1])
        except (TypeError, ValueError):
            cycle = -1
        items.append((cycle, key, value.get("created_at", 0.0)))
    for _cycle, key, _created in sorted(items, reverse=True)[keep_cycles:]:
        _SCAN_BTC_REGIME_CACHE.pop(key, None)


def _scan_context(context: dict[str, Any] | None) -> dict[str, Any]:
    base = dict(context or {})
    base["mode"] = "SCAN"
    base["data_provider"] = "BINANCE_PUBLIC_ONLY"
    base["primary_data_source"] = "BINANCE_PUBLIC"
    base["allow_binance_fallback"] = False
    return base


def _btc_regime_payload(h4_candles: list[Candle], h1_candles: list[Candle], source_by_tf: dict[str,str]) -> dict[str,Any]:
    d1_candles = resample_candles(h4_candles, 24 * 60 * 60 * 1000)
    regime = build_multi_timeframe_regime({
        "D1": (d1_candles, 2),
        "H4": (h4_candles, SWING_SPAN_H4),
        "H1": (h1_candles, SWING_SPAN_H1),
    })
    h4_structure = build_structure(h4_candles, SWING_SPAN_H4, min_swing_atr=SWING_MIN_ATR_H4)
    # V2 macro direction is the confirmed BTC H4 structure. The aggregate
    # D1/H4/H1 regime remains available as a soft context field.
    return {
        "trend": h4_structure.trend,
        "regime": regime["trend"],
        "aggregate_regime": regime["trend"],
        "state": regime["state"],
        "confidence": regime["confidence"],
        "bullish_score": regime["bullish_score"],
        "bearish_score": regime["bearish_score"],
        "gap": regime["gap"],
        "structure": _structure_summary(h4_structure, h4_candles),
        "trend_strength": trend_strength_metrics(h4_candles, h4_structure),
        "fibonacci": {
            d: fibonacci_summary(build_fibonacci(h4_candles, h4_structure, d, "H4"), h4_candles[-1].close)
            for d in DIRECTIONS_BOTH
        },
        "timeframes": regime["timeframes"],
        "sources_by_timeframe": source_by_tf,
        "candles": len(h4_candles),
        "last_closed_ms": {
            "H4": h4_candles[-1].time_ms if h4_candles else None,
            "H1": h1_candles[-1].time_ms if h1_candles else None,
        },
        "closed_candles_only": True,
    }


async def analyze_btc_regime(context: dict[str,Any] | None=None) -> dict[str,Any]:
    context=_scan_context(context); key=_scan_cache_key(context); force=bool(context.get("force_fresh_btc_regime",False))
    if key and key in _SCAN_BTC_REGIME_CACHE and not force:
        cached=_SCAN_BTC_REGIME_CACHE[key]
        return {"pair":"BTCUSDT",**{k:v for k,v in cached.items() if k not in {"candles_h4","candles_h1"}},"cached":True}
    h4,h4_source=await fetch_series("BTCUSDT",H4,BTC_H4_CANDLES_REQUIRED,H4_MS,allow_fallback=False)
    h1,h1_source=await fetch_series("BTCUSDT",H1,168,H1_MS,allow_fallback=False)
    payload=_btc_regime_payload(h4,h1,{"D1":h4_source+"_DERIVED","H4":h4_source,"H1":h1_source})
    if key:
        _SCAN_BTC_REGIME_CACHE[key]={**payload,"candles_h4":h4,"candles_h1":h1,"created_at":time.time()}
        _prune_scan_btc_cache(key[0])
    return {"pair":"BTCUSDT",**payload,"cached":False}


async def analyze_scan_structure(pair: str, context: dict[str,Any] | None=None) -> dict[str,Any]:
    context=_scan_context(context); symbol=normalize_pair(pair); btc=await analyze_btc_regime(context); btc_trend=str(btc.get("trend") or "RANGE").upper()
    if symbol=="BTCUSDT":
        return {"pair":symbol,"trend":btc_trend,"regime":btc.get("regime",btc_trend),"state":btc.get("state",btc_trend),"confidence":btc.get("confidence",50),"btc_h4_trend":btc_trend,"aligned":True,"analysis":{"btc_regime":btc},"data":{"source":"BINANCE","timeframe":"D1/H4/H1","candles_used":btc.get("candles"),"closed_candles_only":True,"data_policy":"BINANCE_PUBLIC_ONLY"}}
    h4, h4_source = await fetch_series(symbol, H4, PAIR_H4_CANDLES_REQUIRED, H4_MS, allow_fallback=False)
    # Reuse the M15 series already needed by generate_setup and derive H1 locally.
    # This removes one standalone H1 REST call per scanned pair while preserving
    # H1 structure/regime detail from the same closed M15 candles.
    m15, m15_source = await fetch_series(symbol, M15, M15_CANDLES_REQUIRED, M15_MS, allow_fallback=False)
    h1 = resample_candles(m15, H1_MS)
    if len(h1) < 20:
        raise InsufficientHistoryError(
            f"Riwayat M15 Binance {symbol} hanya membentuk {len(h1)} candle H1; tidak cukup untuk struktur scan."
        )
    h1_source = f"{m15_source}_DERIVED"
    d1 = resample_candles(h4, 24 * 60 * 60 * 1000)
    h4_structure = build_structure(h4, SWING_SPAN_H4, min_swing_atr=SWING_MIN_ATR_H4)
    regime = build_multi_timeframe_regime({
        "D1": (d1, 2),
        "H4": (h4, SWING_SPAN_H4),
        "H1": (h1, SWING_SPAN_H1),
    })
    pair_h4_trend = str(h4_structure.trend or "RANGE").upper()
    aligned = (
        (btc_trend == "RANGE" and pair_h4_trend in {"BULLISH", "BEARISH", "RANGE"})
        or (btc_trend in {"BULLISH", "BEARISH"} and pair_h4_trend in {btc_trend, "RANGE"})
    )
    pair_side = "BUY" if pair_h4_trend == "BULLISH" else ("SELL" if pair_h4_trend == "BEARISH" else "")
    alignment_label = classify_btc_alignment(btc_trend, pair_side, btc.get("confidence"))
    aligned = True  # kept for compatibility; BTC alignment never blocks a pair
    return {
        "pair": symbol,
        "btc_alignment": alignment_label,
        "trend": pair_h4_trend,
        "regime": regime["trend"],
        "state": regime["state"],
        "confidence": regime["confidence"],
        "btc_h4_trend": btc_trend,
        "btc_regime": btc.get("regime", btc_trend),
        "aligned": True,
        "btc_alignment_score": 80.0 if alignment_label == "ALIGNED" else (50.0 if alignment_label == "NEUTRAL" else 40.0),
        "analysis": {
            "pair_h4": _structure_summary(h4_structure, h4),
            "pair_regime": regime,
            "btc_regime": btc,
        },
        "data": {
            "source": "BINANCE",
            "timeframe": "D1/H4/H1",
            "candles_used": {"D1": len(d1), "H4": len(h4), "H1": len(h1)},
            "closed_candles_only": True,
            "data_policy": "BINANCE_PUBLIC_ONLY",
            "sources_by_timeframe": {"D1": h4_source + "_DERIVED", "H4": h4_source, "H1": h1_source},
        },
    }


def _setup_signature(setup: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(setup.get("direction") or ""),
        safe_float(setup.get("entry")),
        safe_float(setup.get("price_exp")),
        safe_float(setup.get("sl")),
        safe_float(setup.get("tp")),
    )


async def validate_setup(
    pair: str,
    initial_setup: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fresh score-based validation. BTC mismatch and score delta never veto a candidate."""
    if not isinstance(initial_setup, dict):
        raise ValueError("initial_setup harus berupa dict.")
    fresh_context = _scan_context(context)
    fresh_context["validation"] = True
    fresh_context["force_fresh_btc_regime"] = False
    fresh = await generate_setup(pair, fresh_context)
    initial_conf = safe_float(initial_setup.get("confidence"), 0.0)
    fresh_conf = safe_float(fresh.get("confidence"), 0.0)
    initial_direction = str(initial_setup.get("direction") or "").upper()
    fresh_direction = str(fresh.get("direction") or "").upper()
    data = fresh.get("data") if isinstance(fresh.get("data"), dict) else {}
    analysis = fresh.get("analysis") if isinstance(fresh.get("analysis"), dict) else {}
    geometry = validate_setup_geometry(
        fresh_direction,
        safe_float(fresh.get("entry")),
        safe_float(fresh.get("sl")),
        safe_float(fresh.get("tp")),
        safe_float((analysis.get("entry_zone") or {}).get("low"), safe_float(fresh.get("entry"))),
        safe_float((analysis.get("entry_zone") or {}).get("high"), safe_float(fresh.get("entry"))),
        safe_float(data.get("tick_size"), 0.0),
    )
    data_quality = fresh.get("data_quality") if isinstance(fresh.get("data_quality"), dict) else {}
    stale_data = (not data_quality.get("fresh", True)) or fresh.get("candidate_state") == "DATA_UNAVAILABLE"
    geometry_codes = {"PRICE_INVALID", "GEOMETRY_INVALID", "ZONE_INVERTED", "ENTRY_OUTSIDE_ZONE", "ZONE_TOO_WIDE"}
    unsafe = any(item.get("code") in geometry_codes for item in geometry)
    valid = not stale_data and not unsafe
    initial_sig = _setup_signature(initial_setup)
    fresh_sig = _setup_signature(fresh)
    initial_vs_fresh = {
        "confidence": {"initial": round(initial_conf, 2), "fresh": round(fresh_conf, 2), "delta": round(fresh_conf - initial_conf, 2)},
        "direction": {"initial": initial_direction, "fresh": fresh_direction, "changed": initial_direction != fresh_direction},
        "entry": {"initial": safe_float(initial_setup.get("entry")), "fresh": safe_float(fresh.get("entry"))},
        "sl": {"initial": safe_float(initial_setup.get("sl")), "fresh": safe_float(fresh.get("sl"))},
        "tp": {"initial": safe_float(initial_setup.get("tp")), "fresh": safe_float(fresh.get("tp"))},
        "setup_replaced": initial_sig != fresh_sig,
        "btc_filter": "DISABLED_SCORE_ONLY",
        "veto_policy": "ONLY_STALE_DATA_OR_UNSAFE_EXECUTION_GEOMETRY",
    }
    fresh_analysis = dict(analysis)
    fresh_analysis["validation"] = {
        "initial_confidence": round(initial_conf, 2),
        "validated_confidence": round(fresh_conf, 2),
        "confidence_delta": round(fresh_conf - initial_conf, 2),
        "initial_direction": initial_direction,
        "validated_direction": fresh_direction,
        "direction_changed": initial_direction != fresh_direction,
        "geometry_issues": geometry,
        "stale_data": stale_data,
        "score_only": True,
        "reason": "Skor/arah direfresh; BTC mismatch tidak mem-veto kandidat." if valid else ("Data candle basi/tidak tersedia." if stale_data else "Geometri order tidak aman."),
    }
    fresh = dict(fresh)
    fresh["analysis"] = fresh_analysis
    return {
        "valid": valid,
        "reason": fresh_analysis["validation"]["reason"],
        "validation_reason": fresh_analysis["validation"]["reason"],
        "validation_status": "VALID" if valid else ("STALE_DATA" if stale_data else "UNSAFE_GEOMETRY"),
        "reason_codes": (["DATA_STALE"] if stale_data else [x.get("code") for x in geometry]) if not valid else [],
        "stale_data": stale_data,
        "initial_vs_fresh": initial_vs_fresh,
        "confidence": fresh_conf,
        "setup": fresh,
        "validation": fresh_analysis["validation"],
    }


async def _fetch_if_short(
    pair: str,
    interval: str,
    interval_ms: int,
    count: int,
    *,
    allow_fallback: bool = True,
) -> list[Candle]:
    candles, _source = await fetch_series(
        pair,
        interval,
        count,
        interval_ms,
        allow_fallback=allow_fallback,
    )
    return candles


# ============================================================================
# OPTIONAL SELF-TEST UTILITIES
# ============================================================================


def validate_directional_alignment(result: dict[str, Any]) -> tuple[bool, list[str]]:
    """Compatibility check: validate the output side only, never BTC/pair alignment."""
    direction = str(result.get("direction") or "").upper()
    if direction not in {"BUY", "SELL"}:
        return False, ["direction harus BUY/SELL"]
    # BTC regime, pair regime and H4 alignment are score/context fields, not vetoes.
    return True, []


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



def validate_liquidity_map(liquidity_map: dict[str, Any], current: float) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for bucket in ("all", "above", "below"):
        if not isinstance(liquidity_map.get(bucket), list):
            errors.append(f"liquidity_map.{bucket} not list")
    for zone in liquidity_map.get("all", []):
        if not (zone.low <= zone.midpoint <= zone.high):
            errors.append("liquidity zone geometry invalid")
        if not 0.0 <= zone.significance <= 100.0:
            errors.append("liquidity significance outside 0-100")
    return (not errors, errors)


def validate_fib_zone(fib_analysis: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not fib_analysis.get("available"):
        return False, ["fib unavailable"]
    for name in ("primary_zone", "secondary_zone", "deep_zone", "normal_zone"):
        zone = fib_analysis.get(name)
        if not isinstance(zone, dict):
            errors.append(f"missing {name}")
            continue
        low, high = safe_float(zone.get("price_low")), safe_float(zone.get("price_high"))
        if high < low:
            errors.append(f"{name} inverted")
    return (not errors, errors)


def validate_entry_zone(entry_low: float, entry_high: float, entry: float, current: float, direction: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if entry_low > entry_high:
        errors.append("entry zone inverted")
    if not (entry_low - EPS <= entry <= entry_high + EPS):
        errors.append("entry outside zone")
    if direction == DIRECTION_BUY and entry >= current:
        errors.append("BUY entry not below current")
    if direction == DIRECTION_SELL and entry <= current:
        errors.append("SELL entry not above current")
    return (not errors, errors)


def validate_invalidation(direction: str, entry: float, invalidation: dict[str, Any] | None, sl: float, h1_atr: float) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not invalidation:
        return False, ["missing invalidation"]
    anchor = safe_float(invalidation.get("anchor_level"), 0.0)
    if direction == DIRECTION_BUY:
        if not anchor < entry or not sl < anchor:
            errors.append("BUY invalidation/SL geometry invalid")
    else:
        if not anchor > entry or not sl > anchor:
            errors.append("SELL invalidation/SL geometry invalid")
    risk_h1 = abs(entry - sl) / max(h1_atr, EPS)
    risk_pct = abs(entry - sl) / max(entry, EPS) * 100.0
    if risk_h1 > SL_MAX_RISK_H1_ATR_V2:
        errors.append("SL risk exceeds H1 ATR maximum")
    if risk_pct > SL_MAX_RISK_PCT_V2:
        errors.append("SL risk exceeds percentage maximum")
    return (not errors, errors)


def validate_target_map(target_map: list[dict[str, Any]], direction: str, entry: float, sl: float) -> tuple[bool, list[str]]:
    if not target_map:
        return False, ["empty target map"]
    risk = abs(entry - sl)
    for target in target_map:
        level = safe_float(target.get("level"), 0.0)
        rr = safe_float(target.get("rr"), 0.0)
        valid_side = level > entry if direction == DIRECTION_BUY else level < entry
        if valid_side and rr >= TARGET_MIN_RR and risk > 0:
            return True, []
    return False, ["no target satisfies direction and minimum RR"]


def validate_rsi_projection(projection: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for tf in ("M15", "H4"):
        stats = (projection.get("at_entry") or {}).get(tf) or {}
        lo = safe_float(stats.get("min"), -1.0)
        med = safe_float(stats.get("median"), -1.0)
        hi = safe_float(stats.get("max"), -1.0)
        if not (0.0 <= lo <= med <= hi <= 100.0):
            errors.append(f"projected {tf} RSI bounds invalid")
    consensus = safe_float(projection.get("scenario_consensus"), -1.0)
    if not 0.0 <= consensus <= 100.0:
        errors.append("projection consensus outside 0-100")
    return (not errors, errors)


def validate_thesis(candidate: Candidate) -> tuple[bool, list[str]]:
    return validate_thesis_consistency(candidate)


# --------------------------------------------------------------------------
# STRUCTURAL PREDICTION V2 SELF-TEST
# --------------------------------------------------------------------------


def _v2_make_synthetic_candles(
    *,
    start: float = 100.0,
    bars: int = 220,
    direction: str = "BUY",
) -> list[Candle]:
    """Deterministic closed-candle series for unit/invariant tests."""
    data: list[Candle] = []
    price = start
    for i in range(bars):
        # Alternating structural impulses and controlled pullbacks.
        phase = i % 40
        if direction == "BUY":
            if phase < 26:
                drift = 0.45
            else:
                drift = -0.28
        else:
            if phase < 26:
                drift = -0.45
            else:
                drift = 0.28
        open_price = price
        close = max(1.0, price + drift)
        wick = 0.12 + (0.04 if phase % 7 == 0 else 0.0)
        high = max(open_price, close) + wick
        low = min(open_price, close) - wick
        volume = 1000.0 + (250.0 if phase in {2, 3, 4, 26} else 0.0)
        data.append(Candle(
            time_ms=i * M15_MS,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            turnover=volume * close,
        ))
        price = close
    return data


def _self_test_structural_prediction_v2_core() -> dict[str, Any]:
    """Deterministic local regression suite; never hits network."""
    tests: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # DATA / STRUCTURE
    # ------------------------------------------------------------------
    m15_buy = _v2_make_synthetic_candles(direction="BUY", bars=720)
    h1_buy = _v2_make_synthetic_candles(direction="BUY", bars=220)
    h4_buy = _v2_make_synthetic_candles(direction="BUY", bars=160)
    m15_sell = _v2_make_synthetic_candles(direction="SELL", bars=220)
    h1_sell = _v2_make_synthetic_candles(direction="SELL", bars=220)
    h4_sell = _v2_make_synthetic_candles(direction="SELL", bars=160)
    for series, minimum, name in (
        (m15_buy, 100, "M15"),
        (h1_buy, 40, "H1"),
        (h4_buy, 40, "H4"),
    ):
        assert len(series) >= minimum, f"synthetic {name} history too short"

    structures_buy = {
        "H4": build_structure(h4_buy, SWING_SPAN_H4),
        "H1": build_structure(h1_buy, SWING_SPAN_H1),
        "M15": build_structure(m15_buy, SWING_SPAN_M15),
    }
    structures_sell = {
        "H4": build_structure(h4_sell, SWING_SPAN_H4),
        "H1": build_structure(h1_sell, SWING_SPAN_H1),
        "M15": build_structure(m15_sell, SWING_SPAN_M15),
    }
    assert structures_buy["H4"].trend == "BULLISH", "synthetic BUY H4 structure not bullish"
    assert structures_sell["H4"].trend == "BEARISH", "synthetic SELL H4 structure not bearish"
    tests["closed_candles_and_structure"] = True

    # ------------------------------------------------------------------
    # LIQUIDITY MAP / CLUSTER / ROLE / SWEEP
    # ------------------------------------------------------------------
    current_buy = m15_buy[-1].close
    liq_buy = _v2_build_liquidity_map(
        {"H4": h4_buy, "H1": h1_buy, "M15": m15_buy},
        structures_buy,
        current_buy,
        DIRECTION_BUY,
    )
    ok, errors = validate_liquidity_map(liq_buy, current_buy)
    assert ok, "; ".join(errors)
    assert isinstance(liq_buy.get("above"), list) and isinstance(liq_buy.get("below"), list)

    # Explicit equal-level cluster regression.
    cluster_candles = [
        Candle(i * M15_MS, 100, 101 + (0.01 if i == 1 else 0), 99, 100.2, 1000, 100000)
        for i in range(30)
    ]
    raw_equal = [
        {"side": "BUY_SIDE", "level": 101.0, "index": 10, "source": "SWING_HIGH"},
        {"side": "BUY_SIDE", "level": 101.04, "index": 14, "source": "SWING_HIGH"},
        {"side": "BUY_SIDE", "level": 101.02, "index": 18, "source": "EQUAL_LEVEL"},
    ]
    clusters = _v2_cluster_liquidity_levels(raw_equal, cluster_candles, "H1", 100.0)
    assert clusters, "equal liquidity levels did not cluster"
    assert clusters[0].touches >= 2, "cluster touch count missing"

    # Pool swept/consumed regression: after formation price breaks the pool.
    sweep_candles = [
        Candle(i * M15_MS, 100, 101, 99 if i < 8 else 96, 100, 1000, 100000)
        for i in range(10)
    ]
    swept = _v2_cluster_liquidity_levels(
        [{"side": "SELL_SIDE", "level": 99.5, "index": 4, "source": "SWING_LOW"}],
        sweep_candles,
        "M15",
        100.0,
    )
    assert swept and swept[0].swept, "liquidity sweep detection failed"
    tests["liquidity_map_cluster_sweep"] = True

    # ------------------------------------------------------------------
    # ADAPTIVE FIBONACCI
    # ------------------------------------------------------------------
    fib_buy = _v2_build_adaptive_fib_zones(h4_buy, structures_buy["H4"], DIRECTION_BUY, "H4")
    fib_sell = _v2_build_adaptive_fib_zones(h4_sell, structures_sell["H4"], DIRECTION_SELL, "H4")
    for fib in (fib_buy, fib_sell):
        ok, errors = validate_fib_zone(fib)
        assert ok, "; ".join(errors)
        assert fib["primary_zone"]["price_low"] <= fib["primary_zone"]["price_high"]
        assert fib["deep_zone"]["ratio_low"] == FIB_DEEP_LOW
        assert fib["deep_zone"]["ratio_high"] == FIB_DEEP_HIGH
    tests["adaptive_fibonacci"] = True

    # ------------------------------------------------------------------
    # PROJECTED RSI — ensemble bounded and scenario-aware
    # ------------------------------------------------------------------
    projection_buy = _v2_project_rsi_to_entry_zone(
        m15_buy,
        h4_buy,
        safe_float(fib_buy["deep_zone"]["price_low"]),
        safe_float(fib_buy["deep_zone"]["price_high"]),
        DIRECTION_BUY,
    )
    projection_sell = _v2_project_rsi_to_entry_zone(
        m15_sell,
        h4_sell,
        safe_float(fib_sell["deep_zone"]["price_low"]),
        safe_float(fib_sell["deep_zone"]["price_high"]),
        DIRECTION_SELL,
    )
    for projection in (projection_buy, projection_sell):
        ok, errors = validate_rsi_projection(projection)
        assert ok, "; ".join(errors)
        assert len(projection["scenario_values"]["M15"]) == len(RSI_PROJECTION_SCENARIOS)
        assert 0.0 <= projection["scenario_consensus"] <= 100.0
    assert simulate_rsi_path(m15_buy, projection_buy["entry_target"], 4, "FAST_RETRACE") >= 0.0
    tests["projected_rsi"] = True

    # ------------------------------------------------------------------
    # INVALIDATION / STRUCTURAL SL
    # ------------------------------------------------------------------
    h1_atr_buy = atr_series(h1_buy, 14)[-1]
    buy_current = current_buy
    buy_pool = LiquidityZone(
        "SELL_SIDE", buy_current - 3.0, buy_current - 2.5, buy_current - 2.75,
        85, "H4", 3, ["EQUAL_LEVEL"], len(h4_buy) - 10, len(h4_buy) - 4,
        False, None, 0, 0, 85, "SL_GUARD_LIQUIDITY"
    )
    liq_buy["below"].insert(0, buy_pool)
    inv_buy = _v2_select_prediction_invalidation(
        DIRECTION_BUY, buy_current, structures_buy["H4"], structures_buy["H1"], liq_buy, [], h1_atr_buy
    )
    assert inv_buy is not None and inv_buy["sl"] < inv_buy["anchor_level"] < buy_current
    ok, errors = validate_invalidation(DIRECTION_BUY, buy_current, inv_buy, inv_buy["sl"], h1_atr_buy)
    assert ok, "; ".join(errors)

    sell_current = m15_sell[-1].close
    sell_pool = LiquidityZone(
        "BUY_SIDE", sell_current + 0.8, sell_current + 1.2, sell_current + 1.0,
        85, "H4", 3, ["EQUAL_LEVEL"], len(h4_sell) - 10, len(h4_sell) - 4,
        False, None, 0, 0, 85, "SL_GUARD_LIQUIDITY"
    )
    liq_sell = _v2_build_liquidity_map(
        {"H4": h4_sell, "H1": h1_sell, "M15": m15_sell},
        structures_sell,
        sell_current,
        DIRECTION_SELL,
    )
    liq_sell["above"].insert(0, sell_pool)
    h1_atr_sell = atr_series(h1_sell, 14)[-1]
    inv_sell = _v2_select_prediction_invalidation(
        DIRECTION_SELL, sell_current, structures_sell["H4"], structures_sell["H1"], liq_sell, [], h1_atr_sell
    )
    assert inv_sell is not None and sell_current < inv_sell["anchor_level"] < inv_sell["sl"]
    ok, errors = validate_invalidation(DIRECTION_SELL, sell_current, inv_sell, inv_sell["sl"], h1_atr_sell)
    assert ok, "; ".join(errors)
    tests["prediction_invalidation_and_structural_sl"] = True

    # Tight structural stop is allowed; V2 must not widen it to a 1.5 ATR floor.
    tight_struct = StructureSnapshot("BULLISH", [], [], [], [], None, None, 110.0, 99.0)
    tight_map = {
        "all": [
            LiquidityZone("SELL_SIDE", 98.9, 99.4, 99.15, 80, "H1", 2, ["SWING_LOW"], 5, 10, False, None, 0, 0, 80, "SL_GUARD_LIQUIDITY")
        ],
        "above": [],
        "below": [
            LiquidityZone("SELL_SIDE", 98.9, 99.4, 99.15, 80, "H1", 2, ["SWING_LOW"], 5, 10, False, None, 0, 0, 80, "SL_GUARD_LIQUIDITY")
        ],
        "significant_above": [],
        "significant_below": [],
        "magnet": None,
    }
    tight_inv = _v2_select_prediction_invalidation(DIRECTION_BUY, 100.0, tight_struct, tight_struct, tight_map, [], 4.0)
    assert tight_inv is not None
    assert tight_inv["risk_h1_atr"] < 1.5, "V2 unexpectedly forced an ATR floor"
    tests["no_forced_atr_stop_floor"] = True

    # ------------------------------------------------------------------
    # TARGET MAP / RR / HARD GATES
    # ------------------------------------------------------------------
    target_map_buy = _v2_build_target_map(DIRECTION_BUY, buy_current, inv_buy["sl"], liq_buy, structures_buy["H4"], structures_buy["H1"], h1_atr_buy, current=buy_current)
    if target_map_buy:
        ok, errors = validate_target_map(target_map_buy, DIRECTION_BUY, buy_current, inv_buy["sl"])
        assert ok, "; ".join(errors)
        chosen = _v2_select_target(target_map_buy, DIRECTION_BUY)
        if chosen is not None:
            assert chosen["level"] > buy_current and chosen["rr"] >= TARGET_MIN_RR

    # Add an explicit structural destination so the RR gate has a deterministic
    # SELL target in the synthetic fixture.
    synthetic_sell_target = LiquidityZone(
        "SELL_SIDE", sell_current - 5.0, sell_current - 4.0, sell_current - 4.5,
        90, "H4", 4, ["H4_SWING_LOW"], len(h4_sell) - 8, len(h4_sell) - 3,
        False, None, 0, 0, 90, "TARGET_LIQUIDITY"
    )
    liq_sell["below"].insert(0, synthetic_sell_target)
    liq_sell["all"].append(synthetic_sell_target)
    target_map_sell = _v2_build_target_map(DIRECTION_SELL, sell_current, inv_sell["sl"], liq_sell, structures_sell["H4"], structures_sell["H1"], h1_atr_sell, current=sell_current)
    if target_map_sell:
        ok, errors = validate_target_map(target_map_sell, DIRECTION_SELL, sell_current, inv_sell["sl"])
        assert ok, "; ".join(errors)
        chosen = _v2_select_target(target_map_sell, DIRECTION_SELL)
        if chosen is not None:
            assert chosen["level"] < sell_current and chosen["rr"] >= TARGET_MIN_RR
    tests["target_map_and_rr"] = True

    # Entry-zone geometry validators in both directions.
    buy_low, buy_high = buy_current - 3.0, buy_current - 2.0
    sell_low, sell_high = sell_current + 2.0, sell_current + 3.0
    ok, errors = validate_entry_zone(buy_low, buy_high, (buy_low + buy_high) / 2.0, buy_current, DIRECTION_BUY)
    assert ok, "; ".join(errors)
    ok, errors = validate_entry_zone(sell_low, sell_high, (sell_low + sell_high) / 2.0, sell_current, DIRECTION_SELL)
    assert ok, "; ".join(errors)
    tests["entry_zone_geometry"] = True

    # ------------------------------------------------------------------
    # RESULT / THESIS CONTRACTS
    # ------------------------------------------------------------------
    minimal_result = {
        "pair": "TESTUSDT", "direction": "BUY", "price_now_reference": 100.0,
        "entry": 99.0, "entry_reason": "test", "price_exp": 101.0,
        "price_exp_reason": "test", "sl": 97.0, "sl_reason": "test",
        "tp": 104.0, "tp_reason": "test", "confidence": 75.0,
        "confidence_components": {}, "analysis": {}, "data": {}, "strategy": {},
    }
    ok, errors = validate_result_contract(minimal_result)
    assert ok, "; ".join(errors)
    ok, errors = validate_directional_alignment({
        **minimal_result,
        "analysis": {"macro": {"allowed_directions": ["BUY"], "pair_regime": {"trend": "BULLISH"}}},
    })
    assert ok, "; ".join(errors)
    tests["output_contracts"] = True

    # Exposed blueprint-facing helper contract.
    assert isinstance(build_thesis(DIRECTION_BUY, {"trend": "BULLISH"}, {"trend": "BULLISH"}, structures_buy["H4"], structures_buy["H1"]), dict)
    tests["blueprint_public_helpers"] = True

    return {"ok": True, "tests": tests}


# --------------------------------------------------------------------------
# TRAILING: otak yang mengusulkan SL baru; main.py yang memvalidasi dan mengeksekusi.
# Tiga sumber: R-ladder, struktur M15 (higher low / lower high setelah entry), dan
# kelelahan RSI (divergence berlawanan / RSI ekstrem). Yang paling melindungi dipakai.
# --------------------------------------------------------------------------
TRAIL_R_LADDER = [
    (0.75, -0.05),
    (1.00, 0.15),
    (1.25, 0.40),
    (1.50, 0.65),
    (2.00, 1.10),
    (2.50, 1.60),
    (3.00, 2.10),
]
TRAIL_FEE_BUFFER_PCT = 0.12
STRUCT_TRAIL_MIN_R = 0.75
STRUCT_TRAIL_BUFFER_ATR = 0.5
EXHAUST_MIN_R = 1.5
EXHAUST_GIVEBACK_R = 0.8
TRAIL_MIN_GAP_R = 0.30
TRAIL_REJ_WICK_RATIO = _env_float("TRAIL_REJ_WICK_RATIO", 0.5, 0.2, 0.95)
TRAIL_REJ_MIN_PROGRESS = _env_float("TRAIL_REJ_MIN_PROGRESS", 0.6, 0.2, 0.95)
TRAIL_REJ_BUFFER_ATR = _env_float("TRAIL_REJ_BUFFER_ATR", 0.25, 0.0, 2.0)
TRAIL_REJ_LOOKBACK = _env_int("TRAIL_REJ_LOOKBACK", 8, 3, 40)



def _run_coro_sync(coro: Any) -> Any:
    """Jalankan coroutine pada loop sendiri (aman walau dipanggil dari loop aktif)."""
    box: dict[str, Any] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            box["v"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001
            box["e"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _synthetic_series(count: int, interval_ms: int, *, end_closed: bool = True, price: float = 100.0) -> list[Candle]:
    now_ms = int(time.time() * 1000)
    last_open = (now_ms // interval_ms) * interval_ms - (interval_ms if end_closed else 0)
    out: list[Candle] = []
    for i in range(count):
        t = last_open - (count - 1 - i) * interval_ms
        px = price + (i % 7) * 0.1
        out.append(Candle(time_ms=t, open=px, high=px + 0.5, low=px - 0.5, close=px + 0.1, volume=10.0))
    return out


def _self_test_v22() -> dict[str, Any]:
    """Tes kontrak/cache/geometri/biaya/trailing (tanpa jaringan)."""
    tests: dict[str, bool] = {}
    failures: list[dict[str, Any]] = []

    def check(name: str, rule: str, cond: bool, expected: Any = None, actual: Any = None) -> None:
        tests[name] = bool(cond)
        if not cond:
            failures.append({"check": name, "rule": rule, "expected": expected, "actual": actual})

    saved_cache = OrderedDict(_SERIES_CACHE)
    saved_stats = dict(_SERIES_STATS)
    saved_entries, saved_mb = CANDLE_CACHE_MAX_ENTRIES, CANDLE_CACHE_MAX_MB
    saved_fetch = globals()["_fetch_series_uncached"]
    saved_binance = BinanceProvider.klines
    try:
        _SERIES_CACHE.clear()
        for k in _SERIES_STATS:
            _SERIES_STATS[k] = 0

        check("weights_total_100", "THESIS_WEIGHTS_V2 harus berjumlah 100", abs(thesis_weights_total() - 100.0) < 1e-9, 100.0, thesis_weights_total())

        # ---- normalisasi candle
        base = _synthetic_series(100, M15_MS)
        # ---- new score-only components (orderflow proxy, price leg, explicit S/R)
        orderflow = _v2_orderflow_leg_score("BUY", base)
        price_leg = _v2_price_leg_score("BUY", base, base[-1].close - 0.5, base[-1].close + 0.5, base[-1].close)
        sr_structure = StructureSnapshot(
            trend="BULLISH",
            swing_highs=[Pivot(70, 102.0, "HIGH", 73)],
            swing_lows=[Pivot(80, 98.9, "LOW", 83)],
            labels=[], events=[], last_bos=None, last_mss=None,
            protected_high=102.0, protected_low=98.9,
        )
        sr_zone = Zone("BULLISH_OB", 99.0, 100.0, 90, "H4_OB", "ACTIVE", 80.0, {"formation_direction": "BULLISH"})
        sr_score = _v2_support_resistance_score(
            "BUY", base[-1].close,
            {"zone_low": 99.0, "zone_high": 100.0, "pool": None},
            {"timeframe": "H4", "structure": sr_structure, "zones": [sr_zone]},
        )
        check("orderflow_leg_score_only", "orderflow proxy produces score and is explicitly not footprint data", 0 <= orderflow["score"] <= 100 and orderflow["method"] == "OHLCV_PROXY_NOT_TRUE_FOOTPRINT", "score 0..100 with OHLCV proxy label", orderflow)
        check("price_leg_score_only", "price leg produces numeric score without directional veto", 0 <= price_leg["score"] <= 100 and "impulse_pct" in price_leg, "score 0..100 plus impulse", price_leg)
        check("support_resistance_uses_swing_and_ob", "S/R integrates swing levels and active OB zones", 0 <= sr_score["score"] <= 100 and sr_score["zone_count"] == 1 and sr_score["swing_level_count"] >= 1, "zone and swing count", sr_score)
        ok_mismatch, mismatch_errors = validate_directional_alignment({"direction": "BUY", "analysis": {"macro": {"allowed_directions": ["SELL"], "pair_regime": {"trend": "BEARISH"}}}})
        check("btc_pair_alignment_no_veto", "BTC/pair structure mismatch does not reject BUY/SELL direction", ok_mismatch and not mismatch_errors, True, mismatch_errors)

        messy = list(reversed(base)) + [base[10]]  # terbalik + duplikat
        open_c = Candle(time_ms=(int(time.time() * 1000) // M15_MS) * M15_MS, open=1, high=2, low=0.5, close=1.5, volume=1)
        bad_c = Candle(time_ms=base[0].time_ms - M15_MS, open=10, high=5, low=6, close=7, volume=1)  # high<low
        norm, qs = normalize_candles(messy + [open_c, bad_c], M15_MS)
        times = [c.time_ms for c in norm]
        check("normalize_sorted_unique", "ascending & unik", times == sorted(set(times)), "sorted", times[:3])
        check("normalize_drops_open_and_invalid", "candle open/invalid dibuang", qs["open_dropped"] == 1 and qs["invalid_dropped"] == 1 and len(norm) == 100, (1, 1, 100), (qs["open_dropped"], qs["invalid_dropped"], len(norm)))
        invalid_raised = False
        try:
            normalize_candles(base[:50] + [Candle(time_ms=base[0].time_ms - (i + 1) * M15_MS, open=10, high=5, low=6, close=7, volume=1) for i in range(10)], M15_MS)
        except DataInvalidError:
            invalid_raised = True
        check("normalize_rejects_invalid_series", "terlalu banyak OHLC rusak -> DataInvalidError", invalid_raised, True, invalid_raised)
        gap_series = base[:20] + base[25:]
        _, gstats = normalize_candles(gap_series, M15_MS)
        check("normalize_counts_gaps", "gap terdeteksi", gstats["gaps"] == 1, 1, gstats["gaps"])

        # ---- cache hit/miss/mutasi
        _series_cache_store("TESTUSDT", M15, base, "BINANCE", M15_MS)
        hit = _series_cache_get("TESTUSDT", M15, 50, False, None)
        check("cache_hit_smaller_lookback", "lookback kecil dilayani seri besar", hit is not None and len(hit[0]) == 50, 50, None if hit is None else len(hit[0]))
        check("cache_miss_larger_lookback", "tidak mengembalikan 100 bila diminta 672", _series_cache_get("TESTUSDT", M15, 672, False, None) is None, None, "hit")
        if hit:
            hit[0].clear()
        again = _series_cache_get("TESTUSDT", M15, 50, False, None)
        check("cache_no_shared_mutation", "mutasi list pemanggil tidak mengubah cache", again is not None and len(again[0]) == 50, 50, None if again is None else len(again[0]))

        # ---- TTL & boundary
        entry = _SERIES_CACHE[("BINANCE", "TESTUSDT", M15)]
        entry.fetched_at = time.time() - 1000.0
        expected_ms = _expected_last_closed_ms(int(time.time() * 1000), M15_MS)
        entry.last_closed_ms = expected_ms - M15_MS
        check("cache_ttl_expiry", "melewati TTL dan candle close baru -> stale", _series_cache_get("TESTUSDT", M15, 50, False, None) is None, None, "hit")
        entry.last_closed_ms = expected_ms - M15_MS
        entry.fetched_at = (expected_ms + M15_MS) / 1000.0 - 5.0  # diambil 5 dtk sebelum candle terbaru close
        check("cache_boundary_invalidation", "candle baru close -> cache tidak boleh menyembunyikannya", _series_cache_get("TESTUSDT", M15, 50, False, None) is None, None, "hit")

        # ---- bound entries / bytes
        _SERIES_CACHE.clear()
        globals()["CANDLE_CACHE_MAX_ENTRIES"] = 3
        for i in range(5):
            _series_cache_store(f"P{i}USDT", M15, base[-20:], "BINANCE", M15_MS)
        check("cache_bounded_entries", "maks entri dipatuhi", len(_SERIES_CACHE) == 3 and _SERIES_STATS["evictions"] >= 2, 3, len(_SERIES_CACHE))
        globals()["CANDLE_CACHE_MAX_ENTRIES"] = saved_entries
        _SERIES_CACHE.clear()
        globals()["CANDLE_CACHE_MAX_MB"] = 0.01
        before_ev = _SERIES_STATS["evictions"]
        _series_cache_store("BIGUSDT", M15, base, "BINANCE", M15_MS)
        check("cache_bounded_bytes", "maks byte dipatuhi", _series_total_bytes() <= 0.01 * 1024 * 1024 and _SERIES_STATS["evictions"] > before_ev, "<=10KB", _series_total_bytes())
        globals()["CANDLE_CACHE_MAX_MB"] = saved_mb
        _SERIES_CACHE.clear()

        # ---- single-flight, force refresh, pemisahan provider
        calls = {"n": 0}

        async def fake_fetch(pair, interval, count, interval_ms, *, allow_fallback):
            calls["n"] += 1
            await asyncio.sleep(0.05)
            return _synthetic_series(count, interval_ms), "BINANCE", {"gaps": 0}

        globals()["_fetch_series_uncached"] = fake_fetch

        async def concurrent() -> list[Any]:
            return await asyncio.gather(*[fetch_series("SFUSDT", M15, 60, M15_MS, allow_fallback=False) for _ in range(5)])

        results = _run_coro_sync(concurrent())
        check("single_flight_dedup", "5 pemanggil serentak -> 1 fetch", calls["n"] == 1 and len(results) == 5, 1, calls["n"])
        check("returns_independent_lists", "tiap pemanggil mendapat list sendiri", all(results[i][0] is not results[j][0] for i in range(5) for j in range(i + 1, 5)), True, False)
        n_before = calls["n"]
        _run_coro_sync(fetch_series("SFUSDT", M15, 60, M15_MS, allow_fallback=False))
        check("second_call_cache_hit", "panggilan berikutnya dari cache", calls["n"] == n_before, n_before, calls["n"])
        _run_coro_sync(fetch_series("SFUSDT", M15, 60, M15_MS, allow_fallback=False, force_refresh=True))
        check("force_refresh_bypasses_cache", "force_refresh memaksa fetch baru", calls["n"] == n_before + 1, n_before + 1, calls["n"])
        _SERIES_CACHE.clear()
        _series_cache_store("PSUSDT", M15, base, "BINANCE", M15_MS)
        calls["n"] = 0
        got = _run_coro_sync(fetch_series("PSUSDT", M15, 50, M15_MS, allow_fallback=False))
        check("provider_separation", "Binance-only SCAN boleh memakai cache Binance", calls["n"] == 0 and got[1] == "BINANCE", (0, "BINANCE"), (calls["n"], got[1]))
        globals()["_fetch_series_uncached"] = saved_fetch

        # ---- one market-data source for all scan modes
        called = {"binance": 0}

        async def binance_ok(self, *a, **k):
            called["binance"] += 1
            return base

        saved_binance = BinanceProvider.klines
        BinanceProvider.klines = binance_ok
        try:
            no_fallback = _run_coro_sync(fetch_series("NOFBUSDT", M15, 50, M15_MS, allow_fallback=False, use_cache=False))
            with_fallback = _run_coro_sync(fetch_series("FBUSDT", M15, 50, M15_MS, allow_fallback=True, use_cache=False))
            check(
                "binance_only_provider",
                "semua mode memakai Binance tanpa cross-exchange fallback",
                called["binance"] == 2 and no_fallback[1] == "BINANCE" and with_fallback[1] == "BINANCE",
                (2, "BINANCE", "BINANCE"),
                (called["binance"], no_fallback[1], with_fallback[1]),
            )
        finally:
            BinanceProvider.klines = saved_binance

        # ---- geometri BUY/SELL
        check("geometry_buy_valid", "BUY SL<entry<TP valid", validate_setup_geometry("BUY", 100, 98, 106, 99.5, 100.5) == [], [], validate_setup_geometry("BUY", 100, 98, 106, 99.5, 100.5))
        check("geometry_sell_valid", "SELL TP<entry<SL valid", validate_setup_geometry("SELL", 100, 102, 94, 99.5, 100.5) == [], [], validate_setup_geometry("SELL", 100, 102, 94, 99.5, 100.5))
        buy_bad = validate_setup_geometry("BUY", 100, 101, 106)
        sell_bad = validate_setup_geometry("SELL", 100, 98, 94)
        check("geometry_buy_sl_wrong_side", "SL di sisi salah ditolak", bool(buy_bad) and buy_bad[0]["code"] == "GEOMETRY_INVALID", "GEOMETRY_INVALID", buy_bad)
        check("geometry_sell_sl_wrong_side", "SELL SL di bawah entry ditolak", bool(sell_bad) and sell_bad[0]["code"] == "GEOMETRY_INVALID", "GEOMETRY_INVALID", sell_bad)
        outside = validate_setup_geometry("BUY", 100, 98, 106, 90, 95)
        inverted = validate_setup_geometry("BUY", 100, 98, 106, 101, 99)
        check("geometry_entry_outside_zone", "entry di luar zona ditolak", any(i["code"] == "ENTRY_OUTSIDE_ZONE" for i in outside), "ENTRY_OUTSIDE_ZONE", outside)
        check("geometry_zone_inverted", "zona terbalik ditolak", any(i["code"] == "ZONE_INVERTED" for i in inverted), "ZONE_INVERTED", inverted)
        check("geometry_invalid_prices", "harga nol/NaN ditolak", validate_setup_geometry("BUY", 0, 98, 106)[0]["code"] == "PRICE_INVALID", "PRICE_INVALID", None)

        # ---- RR & biaya
        rr = estimate_after_cost_rr("BUY", 100, 99.7, 100.6)
        check("rr_costs_reduce_net", "biaya menurunkan RR bersih", rr["valid"] and rr["after_cost_rr"] < rr["gross_rr"], "net<gross", (rr["gross_rr"], rr["after_cost_rr"]))
        marginal = estimate_after_cost_rr("SELL", 100, 100.3, 99.6, fee_pct=0.2, slippage_pct=0.2)
        check("rr_marginal_turns_net_zero", "profit gross marginal jadi nol setelah biaya", marginal["after_cost_rr"] == 0.0, 0.0, marginal["after_cost_rr"])
        check("rr_impossible_invalid", "RR mustahil -> invalid", estimate_after_cost_rr("BUY", 100, 101, 99)["valid"] is False, False, None)

        # ---- kontrak state kandidat
        def make(direction="BUY", conf=72.0, status="ACTIVE_TRIGGER_CONFIRMED", entry=100.0, sl=98.0, tp=105.0, rr=2.5, asof_age_ms=0):
            now_ms = int(time.time() * 1000)
            return {
                "pair": "SOLUSDT", "direction": direction, "entry": entry, "sl": sl, "tp": tp, "confidence": conf,
                "price_now_reference": entry, "confidence_components": {k: 60.0 for k in THESIS_WEIGHTS_V2},
                "analysis": {"thesis": {"status": status, "type": "BUY_PULLBACK"}, "entry_zone": {"low": entry - 0.3, "high": entry + 0.3},
                             "risk": {"planned_rr": rr, "risk_pct": 2.0, "risk_h1_atr": 1.1}, "invalidation": {"reason": "swing"},
                             "pair_structure": {"signature_h4": "abc"}, "macro": {"btc_regime": {"confidence": 60}}, "confidence": {"penalties": []}},
                "data": {"source": "BINANCE", "tick_size": 0.01, "directional_bias": "BULLISH",
                         "data_asof": {"M15": ((now_ms - asof_age_ms) // M15_MS) * M15_MS - M15_MS, "H1": ((now_ms - asof_age_ms) // (4 * M15_MS)) * 4 * M15_MS - 4 * M15_MS,
                                       "H4": ((now_ms - asof_age_ms) // (16 * M15_MS)) * 16 * M15_MS - 16 * M15_MS}},
            }

        ready = _enrich_setup_contract(make())
        waiting = _enrich_setup_contract(make(status="WAITING_TRIGGER", conf=60.0))
        hard = _enrich_setup_contract(make(sl=101.0))
        stale = _enrich_setup_contract(make(asof_age_ms=3 * 24 * 3600 * 1000))
        fallback = _enrich_setup_contract(make(status="LOW_QUALITY_FALLBACK", conf=58.0))
        low_rr = _enrich_setup_contract(make(rr=1.2, tp=102.0))
        check("state_ready", "setup valid -> READY", ready["candidate_state"] == "READY", "READY", ready["candidate_state"])
        check("state_waiting_trigger", "trigger belum -> WAITING_TRIGGER", waiting["candidate_state"] == "WAITING_TRIGGER", "WAITING_TRIGGER", waiting["candidate_state"])
        check("state_rejected_hard_geometry", "SL salah sisi -> REJECTED_HARD", hard["candidate_state"] == "REJECTED_HARD" and hard["state_reason"] == "GEOMETRY_INVALID", "REJECTED_HARD/GEOMETRY_INVALID", (hard["candidate_state"], hard["state_reason"]))
        check("state_data_unavailable_stale", "data basi -> DATA_UNAVAILABLE", stale["candidate_state"] == "DATA_UNAVAILABLE", "DATA_UNAVAILABLE", stale["candidate_state"])
        check("state_scored_fallback", "fallback tetap kandidat bernilai, bukan rejection kualitas", fallback["candidate_state"] == "READY", "READY", fallback["candidate_state"])
        check("state_rr_below_minimum_rejected", "RR di bawah minimum 1:2 tidak menjadi setup READY", low_rr["candidate_state"] != "READY" and low_rr["planned_rr_gross"] < TARGET_MIN_RR, ("not READY", "low RR"), (low_rr["candidate_state"], low_rr["planned_rr_gross"]))
        required = {"schema_version", "candidate_state", "state_reason", "gate_reasons", "entry_zone", "planned_rr_gross", "planned_rr_after_costs", "data_asof", "data_quality", "setup_archetype", "analysis_source", "risk", "thesis_signature", "confidence_breakdown"}
        check("contract_required_fields", "field kontrak lengkap", required <= set(ready), "semua field", sorted(required - set(ready)))
        check("contract_preserves_legacy_fields", "field lama tidak hilang", all(k in ready for k in ("pair", "direction", "entry", "sl", "tp", "confidence", "analysis", "data")), True, None)
        check("contract_signature_stable", "signature thesis deterministik", ready["thesis_signature"] == _enrich_setup_contract(make())["thesis_signature"], True, None)
        cb = ready["confidence_breakdown"]
        check("confidence_breakdown", "komponen, raw, cap, final tersedia", abs(cb["raw_weighted_score"] - 60.0) < 0.01 and cb["weights_total"] == 100.0 and "waiting" in cb["caps"], 60.0, cb["raw_weighted_score"])

        # ---- BTC alignment
        check("btc_alignment_labels", "label ALIGNED/NEUTRAL/COUNTER/HIGH_RISK",
              (classify_btc_alignment("BULLISH", "BUY"), classify_btc_alignment("RANGE", "BUY"), classify_btc_alignment("BULLISH", "SELL", 50),
               classify_btc_alignment("BEARISH", "BUY", 95)) == ("ALIGNED", "NEUTRAL", "COUNTER_TREND", "HIGH_RISK_CONFLICT"), None, None)

        # ---- kualitas data
        dq_old = _data_quality({"M15": int(time.time() * 1000) - 5 * 3600_000})
        check("data_quality_stale", "M15 lama -> tidak fresh", dq_old["fresh"] is False and "M15" in dq_old["stale_timeframes"], False, dq_old)

        # ---- trailing (R-ladder, tanpa jaringan)
        async def trail(trade):
            return await analyze_trailing(trade, {"fetch_structure": False, "trail_mode": "ladder"})

        buy = _run_coro_sync(trail({"direction": "BUY", "fill_price": 100.0, "initial_sl": 95.0, "sl": 95.0, "price": 110.0, "pair": "XUSDT"}))
        sell = _run_coro_sync(trail({"direction": "SELL", "fill_price": 100.0, "initial_sl": 105.0, "sl": 105.0, "price": 90.0, "pair": "XUSDT"}))
        none_low = _run_coro_sync(trail({"direction": "BUY", "fill_price": 100.0, "initial_sl": 95.0, "sl": 95.0, "price": 100.5, "pair": "XUSDT"}))
        tighter = _run_coro_sync(trail({"direction": "SELL", "fill_price": 100.0, "initial_sl": 105.0, "sl": 90.5, "price": 90.0, "pair": "XUSDT"}))
        check("trailing_buy_improves", "BUY: SL baru > SL lama dan < harga", buy["new_sl"] is not None and 95.0 < buy["new_sl"] < 110.0, "95<sl<110", buy["new_sl"])
        check("trailing_sell_improves", "SELL: SL baru < SL lama dan > harga", sell["new_sl"] is not None and 90.0 < sell["new_sl"] < 105.0, "90<sl<105", sell["new_sl"])
        check("trailing_no_suggestion_below_threshold", "R rendah -> tidak ada usulan", none_low["new_sl"] is None, None, none_low["new_sl"])
        check("trailing_never_widens", "tidak pernah melonggarkan SL", tighter["new_sl"] is None or tighter["new_sl"] < 90.5, "None atau lebih ketat", tighter["new_sl"])
        check("trailing_error_field", "field error terstruktur ada", "error" in buy and buy["schema_version"] == STRATEGY_SCHEMA_VERSION, True, None)

        def mk(rows: list[tuple[float, float, float, float]]) -> list[Candle]:
            return [Candle(time_ms=1_700_000_000_000 + i * M15_MS, open=o, high=h, low=l, close=c, volume=1.0) for i, (o, h, l, c) in enumerate(rows)]

        up = [(100, 101, 99.8, 100.8), (100.8, 102, 100.5, 101.8), (101.8, 103, 101.5, 102.8), (102.8, 104, 102.5, 103.8)]
        buy_rej = mk(up + [(105, 107, 104.5, 105.2), (105.2, 106.0, 104.8, 105.0), (105.0, 105.5, 104.6, 104.9)])
        check("rejection_buy_wick_then_lower_high", "BUY: sumbu atas + lower high terdeteksi", (_detect_rejection(buy_rej, True, 100.0, 110.0) or {}).get("kind") == "WICK", "WICK", _detect_rejection(buy_rej, True, 100.0, 110.0))
        buy_no_lh = mk(up + [(105, 107, 104.5, 105.2), (105.2, 107.5, 104.8, 107.0), (107.0, 108.0, 106.5, 107.8)])
        check("rejection_requires_lower_high", "tanpa lower high tidak ada rejection", _detect_rejection(buy_no_lh, True, 100.0, 110.0) is None, None, _detect_rejection(buy_no_lh, True, 100.0, 110.0))
        buy_far = mk([(100, 101, 99.8, 100.8), (100.8, 102.5, 100.5, 101.0), (101.0, 102.0, 100.8, 101.5), (101.5, 102.2, 101.0, 101.2), (101.2, 101.9, 100.9, 101.1)])
        check("rejection_needs_progress_to_tp", "wick jauh dari TP diabaikan", _detect_rejection(buy_far, True, 100.0, 110.0) is None, None, None)
        sell_dn = [(100, 100.2, 99, 99.2), (99.2, 99.5, 98, 98.2), (98.2, 98.5, 97, 97.2), (97.2, 97.5, 96, 96.2)]
        sell_rej = mk(sell_dn + [(95, 95.5, 93, 94.8), (94.8, 95.2, 94.0, 95.0), (95.0, 95.4, 94.2, 95.1)])
        check("rejection_sell_mirrored", "SELL: sumbu bawah + higher low terdeteksi", (_detect_rejection(sell_rej, False, 100.0, 90.0) or {}).get("kind") == "WICK", "WICK", _detect_rejection(sell_rej, False, 100.0, 90.0))
        struct_none = _run_coro_sync(analyze_trailing({"direction": "BUY", "fill_price": 100.0, "initial_sl": 95.0, "sl": 95.0, "price": 110.0, "pair": "XUSDT", "tp": 120.0}, {"fetch_structure": False}))
        check("structure_mode_ignores_r_ladder", "mode structure tidak memakai R-ladder", struct_none["new_sl"] is None and struct_none.get("mode") == "structure", None, struct_none)

        stats = series_cache_stats()
        check("cache_stats_observable", "metrik cache tersedia", {"hits", "misses", "evictions", "refreshes", "failures", "entries", "approx_mb", "hit_rate"} <= set(stats), True, sorted(stats))
    except Exception as exc:  # noqa: BLE001
        failures.append({"check": "self_test_v22_runtime", "rule": "tes tidak boleh melempar", "expected": "no exception", "actual": f"{type(exc).__name__}: {exc}"})
        tests["self_test_v22_runtime"] = False
    finally:
        globals()["_fetch_series_uncached"] = saved_fetch
        BinanceProvider.klines = saved_binance
        globals()["CANDLE_CACHE_MAX_ENTRIES"], globals()["CANDLE_CACHE_MAX_MB"] = saved_entries, saved_mb
        with _SERIES_LOCK:
            _SERIES_CACHE.clear()
            _SERIES_CACHE.update(saved_cache)
            _SERIES_STATS.update(saved_stats)
    return {"ok": not failures, "tests": tests, "failures": failures}


def self_test_structural_prediction_v2() -> dict[str, Any]:
    """Suite lokal deterministik (tanpa jaringan): core V2 + kontrak/cache v2.2."""
    tests: dict[str, bool] = {}
    failures: list[dict[str, Any]] = []
    try:
        core = _self_test_structural_prediction_v2_core()
        tests.update({f"core.{k}": bool(v) for k, v in core.get("tests", {}).items()})
    except AssertionError as exc:
        tests["core.suite"] = False
        failures.append({"check": "core.suite", "rule": "regresi core V2", "expected": "pass", "actual": f"AssertionError: {exc}"})
    extra = _self_test_v22()
    tests.update({f"v22.{k}": v for k, v in extra["tests"].items()})
    failures.extend(extra["failures"])
    return {"ok": not failures, "tests": tests, "failures": failures}


def _detect_rejection(post: list[Candle], buy: bool, fill: float, tp: float) -> dict[str, Any] | None:
    """Rejection M15 (candle closed): sumbu panjang atau sweep gagal di area dekat TP, lalu lower high / higher low."""
    dist = abs(tp - fill)
    if dist <= 0 or (buy and tp <= fill) or (not buy and tp >= fill):
        return None
    need = TRAIL_REJ_MIN_PROGRESS * dist
    last = len(post) - 1
    for idx in range(last - 1, max(-1, last - 1 - TRAIL_REJ_LOOKBACK), -1):
        c = post[idx]
        rng = c.high - c.low
        if rng <= 0:
            continue
        reached = (c.high - fill) if buy else (fill - c.low)
        if reached < need:
            continue
        wick = (c.high - max(c.open, c.close)) if buy else (min(c.open, c.close) - c.low)
        kind = "WICK" if wick / rng >= TRAIL_REJ_WICK_RATIO else None
        prior = post[max(0, idx - 10):idx]
        if kind is None and len(prior) >= 3:
            if buy:
                ref = max(x.high for x in prior)
                kind = "SWEEP" if c.high > ref and c.close < ref else None
            else:
                ref = min(x.low for x in prior)
                kind = "SWEEP" if c.low < ref and c.close > ref else None
        if kind is None:
            continue
        confirmed = any((d.high < c.high) if buy else (d.low > c.low) for d in post[idx + 1:])
        if confirmed:
            return {"candle_ms": c.time_ms, "kind": kind, "wick_ratio": round(wick / rng, 3), "progress": round(reached / dist, 3)}
    return None


async def _analyze_trailing_rejection(
    trade: dict[str, Any], context: dict[str, Any], result: dict[str, Any],
    buy: bool, fill: float, current_sl: float, price: float, risk: float,
) -> dict[str, Any]:
    """Mode structure: SL baru hanya setelah rejection; SL ditaruh di invalidation M15."""
    result["mode"] = "structure"
    tp = safe_float(trade.get("tp"))
    if not context.get("fetch_structure", True):
        return result
    try:
        pair = normalize_pair(str(trade.get("pair") or ""))
        m15, _ = await fetch_series(pair, M15, 200, M15_MS, allow_fallback=False, max_age_seconds=TRAIL_M15_MAX_AGE_SECONDS)
        if len(m15) < 20:
            return result
        last_ms = m15[-1].time_ms
        result["candle_ms"] = last_ms
        if int(trade.get("last_trail_candle_ms") or 0) >= last_ms:
            result["reason"] = "candle M15 ini sudah diproses"
            return result
        filled_ms = int(trade.get("filled_at_ms") or 0)
        post = [c for c in m15 if c.time_ms >= filled_ms]
        if len(post) < 3:
            return result
        rejection = None
        if not trade.get("armed"):
            rejection = _detect_rejection(post, buy, fill, tp)
            if rejection is None:
                result["reason"] = "belum ada rejection M15"
                return result
            result["rejection"] = rejection
        atr_vals = atr_series(m15, 14)
        atr = atr_vals[-1] if atr_vals else 0.0
        structure = build_structure(m15, SWING_SPAN_M15)
        level_price, label = None, ""
        for pivot in reversed(structure.swing_lows if buy else structure.swing_highs):
            if pivot.index >= len(m15) or m15[pivot.index].time_ms < filled_ms:
                continue
            if (buy and pivot.price > current_sl) or (not buy and pivot.price < current_sl):
                level_price, label = pivot.price, ("higher low" if buy else "lower high")
                break
        if level_price is None and rejection is not None:
            seg = [c for c in post if c.time_ms >= rejection["candle_ms"]]
            level_price = min(c.low for c in seg) if buy else max(c.high for c in seg)
            label = "ekstrem sejak rejection"
        if level_price is None:
            return result
        buffer = max(atr * TRAIL_REJ_BUFFER_ATR, price * 0.0005)
        level = level_price - buffer if buy else level_price + buffer
        cost = (EST_FEE_PCT_PER_SIDE + EST_SLIPPAGE_PCT_PER_SIDE + EST_ENTRY_FEE_PCT + EST_ENTRY_SLIPPAGE_PCT) / 100.0
        min_lock = fill * (1 + cost) if buy else fill * (1 - cost)
        result["candidates"] = {"REJECTION_STRUCTURE": round_price(level)}
        if (buy and level < min_lock) or (not buy and level > min_lock):
            result["reason"] = "invalidation M15 belum mengunci profit di atas biaya"
            return result
        improves = level > current_sl if buy else level < current_sl
        gap = (price - level) if buy else (level - price)
        if not improves or gap < risk * TRAIL_MIN_GAP_R:
            return result
        tag = f"rejection {rejection['kind']}" if rejection else "trailing lanjut"
        result.update(new_sl=round_price(level), source="REJECTION_STRUCTURE",
                      reason=f"{tag}: SL di {label} M15 {round_price(level_price)} (buffer ATR).")
    except Exception as exc:
        result["structure_error"] = str(exc)[:160]
        result["error"] = {"code": "STRUCTURE_ANALYSIS_FAILED", "message": str(exc)[:160]}
    return result


async def analyze_trailing(
    trade: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Usulan SL trailing posisi FILLED. trail_mode=structure (default): hanya setelah rejection M15; ladder: mode lama."""
    context = context or {}
    direction = str(trade.get("direction") or "").upper()
    buy = direction == "BUY"
    fill = safe_float(trade.get("fill_price"))
    initial_sl = safe_float(trade.get("initial_sl"))
    current_sl = safe_float(trade.get("sl"))
    price = safe_float(trade.get("price"))
    risk = abs(fill - initial_sl)
    result: dict[str, Any] = {
        "schema_version": STRATEGY_SCHEMA_VERSION,
        "error": None,
        "new_sl": None,
        "source": None,
        "reason": "",
        "r_now": 0.0,
        "candidates": {},
    }
    if direction not in {"BUY", "SELL"} or fill <= 0 or risk <= 0 or price <= 0:
        return result
    r_now = (price - fill) / risk if buy else (fill - price) / risk
    result["r_now"] = round(r_now, 3)
    if str(context.get("trail_mode") or "structure").lower() != "ladder":
        return await _analyze_trailing_rejection(trade, context, result, buy, fill, current_sl, price, risk)

    candidates: dict[str, tuple[float, str]] = {}
    lock_r = None
    for trigger, lock in TRAIL_R_LADDER:
        if r_now >= trigger:
            lock_r = lock
    if lock_r is not None:
        gain = max(lock_r * risk, fill * TRAIL_FEE_BUFFER_PCT / 100.0)
        candidates["R_LADDER"] = (
            fill + gain if buy else fill - gain,
            f"R-ladder: harga +{r_now:.2f}R, SL dikunci di {lock_r:.1f}R.",
        )

    if context.get("fetch_structure", True) and r_now >= STRUCT_TRAIL_MIN_R:
        try:
            pair = normalize_pair(str(trade.get("pair") or ""))
            m15, _ = await fetch_series(pair, M15, 200, M15_MS, allow_fallback=False, max_age_seconds=TRAIL_M15_MAX_AGE_SECONDS)
            atr_vals = atr_series(m15, 14)
            atr = atr_vals[-1] if atr_vals else 0.0
            structure = build_structure(m15, SWING_SPAN_M15)
            filled_ms = int(trade.get("filled_at_ms") or 0)
            buffer = max(atr * STRUCT_TRAIL_BUFFER_ATR, price * 0.001)

            pivots = structure.swing_lows if buy else structure.swing_highs
            for pivot in reversed(pivots):
                if pivot.index >= len(m15) or m15[pivot.index].time_ms < filled_ms:
                    continue
                if (buy and pivot.price > current_sl) or (not buy and pivot.price < current_sl):
                    candidates["STRUCTURE"] = (
                        pivot.price - buffer if buy else pivot.price + buffer,
                        f"struktur M15: {'higher low' if buy else 'lower high'} "
                        f"{round_price(pivot.price)} terbentuk setelah entry.",
                    )
                    break

            rsi_ctx = build_rsi_context(m15, structure, "M15")
            adverse = "SELL" if buy else "BUY"
            div = (rsi_ctx.get("div") or {}).get(adverse)
            cur = safe_float(rsi_ctx.get("cur"), 50.0)
            exhausted = cur >= 80.0 if buy else cur <= 20.0
            if r_now >= EXHAUST_MIN_R and ((div and div.get("kind") == "REGULAR") or exhausted):
                why = "divergence reguler berlawanan" if div and div.get("kind") == "REGULAR" else f"RSI M15 ekstrem ({cur:.0f})"
                candidates["RSI_EXHAUSTION"] = (
                    price - EXHAUST_GIVEBACK_R * risk if buy else price + EXHAUST_GIVEBACK_R * risk,
                    f"kelelahan momentum: {why}; ruang balik dibatasi {EXHAUST_GIVEBACK_R:.1f}R.",
                )
        except Exception as exc:
            result["structure_error"] = str(exc)[:160]
            result["error"] = {"code": "STRUCTURE_ANALYSIS_FAILED", "message": str(exc)[:160]}

    if not candidates:
        return result
    pick = max if buy else min
    source, (level, reason) = pick(candidates.items(), key=lambda item: item[1][0])
    result["candidates"] = {name: round_price(value[0]) for name, value in candidates.items()}
    improves = level > current_sl if buy else level < current_sl
    gap = (price - level) if buy else (level - price)
    if not improves or gap < risk * TRAIL_MIN_GAP_R:
        return result
    result.update(new_sl=round_price(level), source=source, reason=reason)
    return result


if __name__ == "__main__":
    print(f"{STRATEGY_NAME} v{STRATEGY_VERSION}")
    print("Module contracts: async generate_setup(pair, context), async analyze_btc_regime(context), async analyze_scan_structure(pair, context), async validate_setup(pair, initial_setup, context), async analyze_trailing(trade, context)")
    print("V2 self-test:", self_test_structural_prediction_v2())
