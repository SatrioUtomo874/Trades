"""
strategy.py — Mesin Analisis & Pencarian Setup (Adaptive Trading Bot)
======================================================================

PRINSIP UTAMA (WAJIB DIPATUHI):
    - Modul ini TIDAK PERNAH melakukan request API / network apapun.
    - Seluruh data market (candle OHLCV) HARUS diberikan oleh main.py.
    - strategy.py hanya bertugas sebagai "analis": menerima data,
      mengeluarkan kesimpulan (setup / trailing decision).
    - Parameter strategy hanya boleh diubah lewat apply_update(), yang
      dipanggil oleh learn.py setelah melalui proses validasi statistik.
      Tidak boleh berubah impulsif karena satu trade.

Pendekatan analisis yang digunakan (lihat spesifikasi §8 & combined.txt):
    - Market Structure (swing high/low, BOS, CHOCH)
    - Liquidity (equal high/low, liquidity sweep / stop hunt)
    - Displacement & Imbalance (Fair Value Gap)
    - Momentum (rate of change)
    - Trend Strength (kemiringan regresi harga-terhadap-waktu — lihat
      catatan "steepness" dari combined.txt: tren yang menempuh jarak
      harga sama dalam waktu lebih singkat = tren lebih kuat)
    - Volatility regime (ATR percentile)
    - BTC correlation / BTC cross
    - Market regime (bullish/bearish/sideways/high-vol/low-vol)
    - Session (Asia/London/NewYork)

Confidence Score (0-100%) adalah penjumlahan kontribusi komponen di atas,
masing-masing dengan bobot yang terdokumentasi (lihat CONFIDENCE_WEIGHTS).
Tidak ada angka acak — setiap poin confidence bisa dijelaskan (reason[]).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # optional compatibility surface for legacy dataframe callers
    pd = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy should always be available
    np = None


logger = logging.getLogger("strategy")

STRATEGY_NAME = "adaptive-smc-ict"
EXECUTION_CONFIDENCE_FLOOR = 55.0
EXECUTION_MIN_RR_FLOOR = 1.20

# Market-context assets are analyzed for correlation/regime but are never trade candidates.
TRADE_EXCLUDED_SYMBOLS = frozenset({"BTCUSDT"})

def is_trade_allowed(symbol: str) -> bool:
    return str(symbol or "").upper() not in TRADE_EXCLUDED_SYMBOLS

# ---------------------------------------------------------------------------
# Bobot komponen confidence — total harus 100. Setiap perubahan bobot HARUS
# lewat apply_update() (dicatat versi + alasan + evidence).
# ---------------------------------------------------------------------------
CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "structure": 20.0,       # BOS/CHOCH searah + kekuatan tren (steepness)
    "liquidity": 15.0,       # liquidity sweep + jarak ke equal high/low
    "entry_quality": 15.0,   # OTE / FVG retracement quality
    "risk_reward": 15.0,     # rasio TP/SL
    "momentum": 10.0,        # rate of change searah arah setup
    "volatility": 5.0,       # ATR regime yang wajar (bukan ekstrem/kosong)
    "btc_correlation": 10.0, # korelasi & keselarasan tren BTC
    "regime": 5.0,           # keselarasan dengan market regime keseluruhan
    "session": 3.0,          # sesi trading dengan likuiditas lebih baik
    "confirmation": 2.0,     # confluence tambahan (FVG fill, sweep, dsb)
}
assert abs(sum(CONFIDENCE_WEIGHTS.values()) - 100.0) < 1e-6

DEFAULT_PARAMS: Dict[str, Any] = {
    "ACTIVE_THRESHOLD": 65.0,      # % — execution threshold awal; Learn boleh adaptif tetapi tidak di bawah safety floor
    "swing_left": 2,
    "swing_right": 2,
    "equal_level_tol_atr": 0.15,   # toleransi "equal high/low" dalam satuan ATR
    "displacement_atr_mult": 1.5,  # body candle > mult * ATR = displacement
    "min_rr": 1.2,                 # minimum risk/reward yang dianggap layak
    "sweep_lookback": 40,
    "structure_lookback": 80,
    "momentum_lookback": 10,
    "atr_period": 14,
    "vol_regime_lookback": 100,
    "trend_lookback": 30,
    "btc_corr_lookback": 50,
    "sl_atr_buffer": 0.25,          # buffer SL tambahan dalam satuan ATR
    "min_price_distance_ticks": 2,  # jarak minimum entry/SL/TP dalam tick
    "entry_retracement_fib": 0.618,   # level OTE pullback dari impulse leg (§17/§25)
    "entry_min_offset_atr": 0.25,     # jarak minimum entry dari harga saat ini (satuan ATR)
}


# ---------------------------------------------------------------------------
# Data contract / safety helpers
# ---------------------------------------------------------------------------

_REQUIRED_OHLCV = ("t", "o", "h", "l", "c", "v")

def validate_candles(candles: Sequence[Dict[str, float]], min_len: int = 1) -> Tuple[bool, str]:
    """Validasi keras data yang masuk dari main.py. Tidak melakukan I/O."""
    if candles is None or len(candles) < min_len:
        return False, "INSUFFICIENT_CANDLES"
    prev_t = None
    for i, c in enumerate(candles):
        if not isinstance(c, dict) or any(k not in c for k in _REQUIRED_OHLCV):
            return False, f"MALFORMED_CANDLE_{i}"
        try:
            vals = [float(c[k]) for k in _REQUIRED_OHLCV]
        except (TypeError, ValueError):
            return False, f"NON_NUMERIC_CANDLE_{i}"
        if any(not math.isfinite(v) for v in vals):
            return False, f"NON_FINITE_CANDLE_{i}"
        t, o, h, l, close, v = vals
        if t <= 0 or v < 0 or min(o, h, l, close) <= 0:
            return False, f"INVALID_CANDLE_RANGE_{i}"
        if h < max(o, close) or l > min(o, close) or h < l:
            return False, f"INVALID_OHLC_RELATION_{i}"
        if prev_t is not None and t <= prev_t:
            return False, f"TIMESTAMP_NOT_ASCENDING_{i}"
        prev_t = t
    return True, "OK"


def _last_confirmed(candles: Sequence[Dict[str, float]]) -> Sequence[Dict[str, float]]:
    """Gunakan candle tertutup bila caller menyertakan flag confirm=False/True.
    REST candle tanpa field confirm dianggap sudah closed."""
    if not candles:
        return candles
    last = candles[-1]
    if last.get("confirm", True) is False:
        return candles[:-1]
    return candles


# ---------------------------------------------------------------------------
# Utility indikator — murni matematis, tidak butuh network.
# ---------------------------------------------------------------------------

def _closes(candles: Sequence[Dict[str, float]]) -> List[float]:
    return [c["c"] for c in candles]


def _highs(candles: Sequence[Dict[str, float]]) -> List[float]:
    return [c["h"] for c in candles]


def _lows(candles: Sequence[Dict[str, float]]) -> List[float]:
    return [c["l"] for c in candles]


def ema(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def true_range(candles: Sequence[Dict[str, float]]) -> List[float]:
    tr = []
    prev_close = None
    for c in candles:
        h, l = c["h"], c["l"]
        if prev_close is None:
            tr.append(h - l)
        else:
            tr.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
        prev_close = c["c"]
    return tr


def atr_series(candles: Sequence[Dict[str, float]], period: int = 14) -> List[float]:
    tr = true_range(candles)
    if len(tr) < period:
        avg = sum(tr) / len(tr) if tr else 0.0
        return [avg] * len(tr)
    out: List[float] = []
    running = sum(tr[:period]) / period
    out.extend([running] * period)
    for v in tr[period:]:
        running = (running * (period - 1) + v) / period
        out.append(running)
    return out


def linreg_slope(values: Sequence[float]) -> Tuple[float, float]:
    """Regresi linear sederhana. Return (slope, r_squared).

    Konsep "trend strength = kemiringan pergerakan harga per satuan waktu"
    (lihat combined.txt): dua tren yang menempuh jarak harga sama tapi salah
    satu lebih curam (lebih cepat) dianggap punya tenaga lebih besar.
    """
    n = len(values)
    if n < 3:
        return 0.0, 0.0
    if np is not None:
        x = np.arange(n, dtype=float)
        y = np.asarray(values, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-9
        r2 = 1.0 - ss_res / ss_tot
        return float(slope), max(0.0, min(1.0, r2))
    # fallback tanpa numpy
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n)) or 1e-9
    slope = num / den
    return slope, 0.0


def pct_returns(values: Sequence[float]) -> List[float]:
    out = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        out.append(0.0 if prev == 0 else (values[i] - prev) / prev)
    return out


def correlation(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    a, b = list(a[-n:]), list(b[-n:])
    if np is not None:
        try:
            m = np.corrcoef(a, b)
            v = float(m[0, 1])
            return 0.0 if math.isnan(v) else v
        except Exception:
            return 0.0
    # fallback
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    if va * vb == 0:
        return 0.0
    return cov / (va * vb)


def swing_points(
    candles: Sequence[Dict[str, float]], left: int = 2, right: int = 2
) -> List[Tuple[int, float, str]]:
    """Deteksi fractal swing high/low. Return list of (index, price, 'H'|'L')."""
    highs, lows = _highs(candles), _lows(candles)
    n = len(candles)
    swings: List[Tuple[int, float, str]] = []
    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i - left : i + right + 1]
        if highs[i] == max(window_h) and window_h.count(highs[i]) == 1:
            swings.append((i, highs[i], "H"))
        if lows[i] == min(window_l) and window_l.count(lows[i]) == 1:
            swings.append((i, lows[i], "L"))
    return swings


def equal_levels(
    swings: Sequence[Tuple[int, float, str]], atr_val: float, tol_atr: float
) -> Dict[str, List[float]]:
    """Cari kluster equal-high / equal-low (liquidity pool)."""
    tol = max(atr_val * tol_atr, 1e-9)
    highs = sorted(p for _, p, t in swings if t == "H")
    lows = sorted(p for _, p, t in swings if t == "L")

    def cluster(levels: List[float]) -> List[float]:
        pools = []
        i = 0
        while i < len(levels):
            j = i
            group = [levels[i]]
            while j + 1 < len(levels) and levels[j + 1] - levels[i] <= tol:
                j += 1
                group.append(levels[j])
            if len(group) >= 2:
                pools.append(sum(group) / len(group))
            i = j + 1
        return pools

    return {"equal_highs": cluster(highs), "equal_lows": cluster(lows)}


def detect_liquidity_sweep(
    candles: Sequence[Dict[str, float]], lookback: int
) -> Optional[Dict[str, Any]]:
    """Deteksi liquidity sweep: wick menembus swing sebelumnya lalu close
    kembali di dalam range (stop hunt), sinyal potensi pembalikan/lanjutan.
    """
    if len(candles) < lookback + 3:
        lookback = max(5, len(candles) - 3)
    window = candles[-lookback:-1]
    if not window:
        return None
    last = candles[-1]
    prior_high = max(_highs(window))
    prior_low = min(_lows(window))

    if last["h"] > prior_high and last["c"] < prior_high:
        return {"type": "BEARISH_SWEEP", "level": prior_high}
    if last["l"] < prior_low and last["c"] > prior_low:
        return {"type": "BULLISH_SWEEP", "level": prior_low}
    return None


def detect_displacement(
    candles: Sequence[Dict[str, float]], atr_val: float, mult: float
) -> Optional[Dict[str, Any]]:
    if not candles or atr_val <= 0:
        return None
    last = candles[-1]
    body = abs(last["c"] - last["o"])
    if body >= atr_val * mult:
        direction = "BUY" if last["c"] > last["o"] else "SELL"
        return {"direction": direction, "body": body, "strength": body / atr_val}
    return None


def detect_fvg(candles: Sequence[Dict[str, float]]) -> Optional[Dict[str, Any]]:
    """Fair Value Gap / imbalance 3-candle: candle1.high < candle3.low
    (bullish FVG) atau candle1.low > candle3.high (bearish FVG)."""
    if len(candles) < 3:
        return None
    c1, _, c3 = candles[-3], candles[-2], candles[-1]
    if c1["h"] < c3["l"]:
        return {"type": "BULLISH_FVG", "top": c3["l"], "bottom": c1["h"]}
    if c1["l"] > c3["h"]:
        return {"type": "BEARISH_FVG", "top": c1["l"], "bottom": c3["h"]}
    return None


def classify_session(ts_ms: float) -> str:
    hour = time.gmtime(ts_ms / 1000.0).tm_hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 21:
        return "NEWYORK"
    return "OFF_HOURS"


def classify_volatility_regime(candles: Sequence[Dict[str, float]], params: Dict[str, Any]) -> str:
    """Classify current volatility using ATR relative to price and its historical percentile.

    This helper is intentionally deterministic and input-only. It is used by the
    vNext engine for a separate volatility diagnostic so ``classify_regime`` can
    continue to describe directional market regime.
    """
    if not candles:
        return "NORMAL"
    period = max(2, int(params.get("atr_period", 14)))
    lookback = max(period + 5, int(params.get("vol_regime_lookback", 100)))
    work = list(candles)[-lookback:]
    closes = _closes(work)
    atrs = atr_series(work, period)
    if not closes or not atrs:
        return "NORMAL"
    price = float(closes[-1] or 0.0)
    atr_now = float(atrs[-1] or 0.0)
    if price <= 0 or atr_now <= 0:
        return "NORMAL"

    atr_pct = (atr_now / price) * 100.0
    low_pct = float(params.get("low_vol_pct", 0.15))
    high_pct = float(params.get("high_vol_pct", 3.0))
    if atr_pct < low_pct:
        return "LOW_VOLATILITY"
    if atr_pct > high_pct:
        return "HIGH_VOLATILITY"

    # Relative ATR percentile prevents a symbol with a normally large ATR from
    # being misclassified solely because its absolute ATR is large.
    history = [float(x) for x in atrs if math.isfinite(float(x)) and float(x) > 0]
    if len(history) >= 10:
        rank = _vn_pct_rank(history[-max(10, min(len(history), lookback)):], atr_now) if "_vn_pct_rank" in globals() else 0.5
        if rank < 0.05:
            return "LOW_VOLATILITY"
        if rank > 0.95:
            return "HIGH_VOLATILITY"
    return "NORMAL"


def classify_regime(btc_candles: Sequence[Dict[str, float]], params: Dict[str, Any]) -> str:
    lb = params["trend_lookback"]
    closes = _closes(btc_candles)[-lb:]
    atrs = atr_series(btc_candles, params["atr_period"])
    if len(closes) < 5 or not atrs:
        return "SIDEWAYS"
    slope, r2 = linreg_slope(closes)
    avg_price = sum(closes) / len(closes)
    avg_atr = sum(atrs[-lb:]) / max(1, len(atrs[-lb:]))
    if avg_price == 0:
        return "SIDEWAYS"
    normalized_slope = (slope * len(closes)) / avg_price  # total move (% of price) over window
    vol_pct = (avg_atr / avg_price) * 100.0

    # regime volatilitas ekstrem menang dulu (mempengaruhi validitas semua setup)
    if vol_pct > 3.0:
        return "HIGH_VOLATILITY"
    if vol_pct < 0.15:
        return "LOW_VOLATILITY"
    if normalized_slope > 0.02 and r2 > 0.25:
        return "BULLISH_TREND"
    if normalized_slope < -0.02 and r2 > 0.25:
        return "BEARISH_TREND"
    return "SIDEWAYS"


# ---------------------------------------------------------------------------
# Struktur output
# ---------------------------------------------------------------------------

@dataclass
class Setup:
    pair: str
    direction: str  # BUY / SELL
    entry: float
    tp: float
    sl: float
    confidence: float
    reason: List[str]
    components: Dict[str, float]
    setup_type: str
    regime: str
    session: str
    atr: float
    timestamp: float
    strategy_version: str
    threshold_passed: bool = True
    reference_levels: Dict[str, Any] = field(default_factory=dict)
    viability: str = "UNKNOWN"
    quality_score: float = 0.0
    execution_score: float = 0.0
    context_score: float = 0.0
    freshness_score: float = 0.0
    expected_value_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "direction": self.direction,
            "entry": self.entry,
            "tp": self.tp,
            "sl": self.sl,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "components": {k: round(v, 2) for k, v in self.components.items()},
            "setup_type": self.setup_type,
            "regime": self.regime,
            "session": self.session,
            "atr": self.atr,
            "timestamp": self.timestamp,
            "strategy_version": self.strategy_version,
            "threshold_passed": bool(self.threshold_passed),
            "reference_levels": self.reference_levels,
            "viability": self.viability,
            "quality_score": round(float(self.quality_score), 2),
            "execution_score": round(float(self.execution_score), 2),
            "context_score": round(float(self.context_score), 2),
            "freshness_score": round(float(self.freshness_score), 2),
            "expected_value_score": round(float(self.expected_value_score), 2),
        }


# ---------------------------------------------------------------------------
# Geometry / validasi (dipakai juga oleh main.py sebelum kirim order)
# ---------------------------------------------------------------------------

def validate_geometry(
    direction: str, entry: float, sl: float, tp: float, tick_size: float = 0.0, atr_val: float = 0.0
) -> Tuple[bool, str]:
    for name, val in (("entry", entry), ("sl", sl), ("tp", tp)):
        if val is None or math.isnan(val) or math.isinf(val) or val <= 0:
            return False, f"INVALID_PRICE_{name.upper()}"

    if direction == "BUY":
        if not (sl < entry < tp):
            return False, "GEOMETRY_ORDER_INVALID_BUY"
    elif direction == "SELL":
        if not (tp < entry < sl):
            return False, "GEOMETRY_ORDER_INVALID_SELL"
    else:
        return False, "INVALID_DIRECTION"

    min_dist = max(tick_size * 2, atr_val * 0.05, entry * 0.0005)
    if abs(entry - sl) < min_dist:
        return False, "SL_TOO_CLOSE"
    if abs(entry - tp) < min_dist:
        return False, "TP_TOO_CLOSE"
    return True, "OK"


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------

class Strategy:
    """Mesin analisis. Semua data diberikan lewat argumen — tidak ada I/O."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.version = "1.00"
        self.params: Dict[str, Any] = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.version_history: List[Dict[str, Any]] = [
            {
                "version": self.version,
                "timestamp": time.time(),
                "reason": "INITIAL",
                "old_params": None,
                "new_params": dict(self.params),
                "evidence": None,
            }
        ]

    # -- parameter lifecycle -------------------------------------------------
    def get_active_threshold(self) -> float:
        return float(self.params.get("ACTIVE_THRESHOLD", 0.0))

    def apply_update(self, new_params: Dict[str, Any], reason: str, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dipanggil HANYA oleh learn.py setelah validasi statistik.
        Tidak melakukan validasi ulang di sini secara sengaja — validasi
        (sample size, backtest counterfactual, perbandingan performa)
        adalah tanggung jawab learn.py sesuai prinsip §41/§47/§49.
        """
        old_params = dict(self.params)
        self.params.update(new_params)
        major, minor = self.version.split(".")
        self.version = f"{major}.{int(minor) + 1:02d}"
        record = {
            "version": self.version,
            "timestamp": time.time(),
            "reason": reason,
            "old_params": old_params,
            "new_params": dict(self.params),
            "evidence": evidence,
        }
        self.version_history.append(record)
        return record

    def rollback(self) -> Optional[Dict[str, Any]]:
        if len(self.version_history) < 2:
            return None
        self.version_history.pop()  # buang versi saat ini
        previous = self.version_history[-1]
        self.params = dict(previous["new_params"])
        self.version = previous["version"]
        return previous

    def export_state(self) -> Dict[str, Any]:
        return {"version": self.version, "params": dict(self.params), "version_history": list(self.version_history)}

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.version = state.get("version", self.version)
        self.params.update(state.get("params", {}))
        if isinstance(state.get("version_history"), list) and state.get("version_history"):
            self.version_history = list(state["version_history"])

    # -- analisis utama -------------------------------------------------------
    def analyze(
        self,
        symbol: str,
        candles: Sequence[Dict[str, float]],
        btc_candles: Optional[Sequence[Dict[str, float]]] = None,
        enforce_threshold: bool = True,
    ) -> Optional[Setup]:
        p = self.params
        weights = dict(CONFIDENCE_WEIGHTS)
        weights.update(p.get("CONFIDENCE_WEIGHTS", {}))
        min_len = max(p["structure_lookback"], p["vol_regime_lookback"], p["atr_period"]) + 5
        work = list(_last_confirmed(candles))
        ok, _ = validate_candles(work, min_len=min_len)
        if not ok:
            return None

        atrs = atr_series(work, p["atr_period"])
        atr_now = atrs[-1]
        if atr_now <= 0:
            return None

        closes = _closes(work)
        last_close = closes[-1]

        # --- market structure ---
        struct_window = work[-p["structure_lookback"]:]
        swings = swing_points(struct_window, p["swing_left"], p["swing_right"])
        trend_slice = closes[-p["trend_lookback"]:]
        slope, r2 = linreg_slope(trend_slice)
        trend_dir = "BUY" if slope > 0 else "SELL"

        bos = None
        if swings:
            last_highs = [s for s in swings if s[2] == "H"]
            last_lows = [s for s in swings if s[2] == "L"]
            if last_highs and last_close > last_highs[-1][1]:
                bos = "BOS_UP"
            elif last_lows and last_close < last_lows[-1][1]:
                bos = "BOS_DOWN"

        direction = None
        reasons: List[str] = []
        if bos == "BOS_UP":
            direction = "BUY"
            reasons.append("structure break bullish (BOS)")
        elif bos == "BOS_DOWN":
            direction = "SELL"
            reasons.append("structure break bearish (BOS)")
        else:
            return None  # tanpa structure break, tidak ada dasar entry
        if trend_dir == direction:
            reasons.append("trend slope searah structure")
        else:
            reasons.append("trend slope berlawanan — confidence dikurangi")

        # --- liquidity ---
        levels = equal_levels(swings, atr_now, p["equal_level_tol_atr"])
        sweep = detect_liquidity_sweep(work, p["sweep_lookback"])
        liquidity_score = 0.0
        if sweep:
            if (direction == "BUY" and sweep["type"] == "BULLISH_SWEEP") or (
                direction == "SELL" and sweep["type"] == "BEARISH_SWEEP"
            ):
                liquidity_score += weights["liquidity"] * 0.7
                reasons.append(f"liquidity sweep searah ({sweep['type']})")
        pool = levels["equal_highs"] if direction == "BUY" else levels["equal_lows"]
        if pool:
            liquidity_score += weights["liquidity"] * 0.3
            reasons.append("equal high/low terdeteksi sebagai target likuiditas")
        liquidity_score = min(liquidity_score, weights["liquidity"])

        # --- displacement & FVG (entry quality) ---
        disp = detect_displacement(work, atr_now, p["displacement_atr_mult"])
        fvg = detect_fvg(work)
        entry_quality_score = 0.0
        setup_type_parts = ["SMC_BOS"]
        if disp and disp["direction"] == direction:
            entry_quality_score += weights["entry_quality"] * 0.6
            reasons.append("displacement candle searah")
            setup_type_parts.append("DISPLACEMENT")
        if fvg and (
            (direction == "BUY" and fvg["type"] == "BULLISH_FVG")
            or (direction == "SELL" and fvg["type"] == "BEARISH_FVG")
        ):
            entry_quality_score += weights["entry_quality"] * 0.4
            reasons.append("imbalance/FVG mendukung entry")
            setup_type_parts.append("FVG")
        entry_quality_score = min(entry_quality_score, weights["entry_quality"])

        # --- entry / TP / SL ---
        # PENTING (revisi): entry TIDAK BOLEH sama dengan harga saat ini
        # (last_close) — itu penyebab pending order "terisi" hampir instan
        # begitu WebSocket mulai memantau (harga live sudah pasti dekat
        # dengan harga candle terakhir). Sesuai §17/§25 spesifikasi, entry
        # yang valid adalah level pullback/retracement (OTE) dari impulse
        # leg yang baru terbentuk — bot MENUNGGU harga kembali ke zona
        # tersebut, baru dianggap FILLED. Kalau harga keburu ke TP duluan
        # sebelum pullback terjadi, itu memang seharusnya jadi TIMEOUT
        # ("strategy terlambat entry atau entry terlalu konservatif").
        buffer_ = atr_now * p["sl_atr_buffer"]
        fib = p["entry_retracement_fib"]
        min_offset = atr_now * p["entry_min_offset_atr"]

        if direction == "BUY":
            leg_low = last_lows[-1][1] if last_lows else (last_close - atr_now * 2.0)
            leg_high = last_close
            leg_range = max(leg_high - leg_low, atr_now * 1e-6)
            entry = leg_high - leg_range * fib
            if leg_high - entry < min_offset:
                entry = leg_high - min_offset
            entry = max(entry, leg_low + atr_now * 0.05)  # jangan sampai lewati awal leg

            sl = min(leg_low, entry - atr_now * 0.5) - buffer_
            target_pool = levels["equal_highs"]
            valid_targets = [x for x in target_pool if x > entry]
            tp = min(valid_targets) if valid_targets else entry + (entry - sl) * 2.0
            if tp <= entry:
                tp = entry + (entry - sl) * 2.0
        else:
            leg_high = last_highs[-1][1] if last_highs else (last_close + atr_now * 2.0)
            leg_low = last_close
            leg_range = max(leg_high - leg_low, atr_now * 1e-6)
            entry = leg_low + leg_range * fib
            if entry - leg_low < min_offset:
                entry = leg_low + min_offset
            entry = min(entry, leg_high - atr_now * 0.05)

            sl = max(leg_high, entry + atr_now * 0.5) + buffer_
            target_pool = levels["equal_lows"]
            valid_targets = [x for x in target_pool if x < entry]
            tp = max(valid_targets) if valid_targets else entry - (sl - entry) * 2.0
            if tp >= entry:
                tp = entry - (sl - entry) * 2.0

        reasons.append(f"entry pullback OTE {fib*100:.0f}% dari impulse leg (bukan harga pasar saat ini)")

        ok, geom_reason = validate_geometry(direction, entry, sl, tp, atr_val=atr_now)
        if not ok:
            return None

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0.0
        rr_score = 0.0
        if rr >= p["min_rr"]:
            rr_score = min(weights["risk_reward"], weights["risk_reward"] * (rr / max(2.0, p["min_rr"] * 1.5)))
            reasons.append(f"risk/reward {rr:.2f}R memenuhi minimum")
        else:
            return None  # RR di bawah minimum -> bukan kandidat valid

        # --- momentum ---
        mlb = p["momentum_lookback"]
        roc = 0.0
        if len(closes) > mlb and closes[-mlb - 1] != 0:
            roc = (closes[-1] - closes[-mlb - 1]) / closes[-mlb - 1]
        momentum_aligned = (direction == "BUY" and roc > 0) or (direction == "SELL" and roc < 0)
        momentum_score = weights["momentum"] * min(1.0, abs(roc) * 20) if momentum_aligned else 0.0
        if momentum_aligned:
            reasons.append("momentum (ROC) searah")

        # --- volatility regime ---
        vol_lb = atrs[-p["vol_regime_lookback"]:] if len(atrs) >= p["vol_regime_lookback"] else atrs
        vol_rank = sorted(vol_lb).index(min(vol_lb, key=lambda x: abs(x - atr_now))) / max(1, len(vol_lb) - 1)
        volatility_score = weights["volatility"] * (1.0 - abs(vol_rank - 0.5) * 2)
        if 0.2 <= vol_rank <= 0.85:
            reasons.append("volatility (ATR) berada di rentang wajar")

        # --- BTC correlation ---
        btc_corr_score = 0.0
        if btc_candles and symbol.upper() != "BTCUSDT":
            lb = p["btc_corr_lookback"]
            sym_ret = pct_returns(closes[-lb:])
            btc_ret = pct_returns(_closes(btc_candles)[-lb:])
            corr = correlation(sym_ret, btc_ret)
            btc_slope, _ = linreg_slope(_closes(btc_candles)[-p["trend_lookback"]:])
            btc_dir = "BUY" if btc_slope > 0 else "SELL"
            if corr > 0.3 and btc_dir == direction:
                btc_corr_score = weights["btc_correlation"] * min(1.0, corr)
                reasons.append(f"selaras dengan tren BTC (corr={corr:.2f})")
            elif corr < -0.3 and btc_dir != direction:
                btc_corr_score = weights["btc_correlation"] * min(1.0, abs(corr)) * 0.7
                reasons.append(f"korelasi negatif terhadap BTC mendukung arah (corr={corr:.2f})")
        else:
            btc_corr_score = weights["btc_correlation"] * 0.5  # netral utk BTCUSDT sendiri / data tak tersedia

        # --- regime & session ---
        regime = classify_regime(btc_candles if btc_candles else candles, p)
        regime_score = 0.0
        if (regime == "BULLISH_TREND" and direction == "BUY") or (
            regime == "BEARISH_TREND" and direction == "SELL"
        ):
            regime_score = weights["regime"]
            reasons.append(f"searah market regime ({regime})")
        elif regime == "SIDEWAYS":
            regime_score = weights["regime"] * 0.4

        session = classify_session(work[-1].get("t", time.time() * 1000))
        session_score = weights["session"] if session in ("LONDON", "NEWYORK") else weights["session"] * 0.3

        confirmation_count = sum([bool(sweep), bool(fvg), bool(disp), bool(pool)])
        confirmation_score = weights["confirmation"] * min(1.0, confirmation_count / 3)

        structure_alignment = 1.0 if trend_dir == direction else 0.45
        structure_score = weights["structure"] * min(1.0, (0.45 + r2 * 0.35) * structure_alignment) if bos else 0.0

        components = {
            "structure": structure_score,
            "liquidity": liquidity_score,
            "entry_quality": entry_quality_score,
            "risk_reward": rr_score,
            "momentum": momentum_score,
            "volatility": volatility_score,
            "btc_correlation": btc_corr_score,
            "regime": regime_score,
            "session": session_score,
            "confirmation": confirmation_score,
        }
        confidence = max(0.0, min(100.0, sum(components.values())))

        threshold_passed = confidence >= self.get_active_threshold()
        if enforce_threshold and not threshold_passed:
            return None

        return Setup(
            pair=symbol,
            direction=direction,
            entry=entry,
            tp=tp,
            sl=sl,
            confidence=confidence,
            reason=reasons,
            components=components,
            setup_type="+".join(setup_type_parts),
            regime=regime,
            session=session,
            atr=atr_now,
            timestamp=work[-1].get("t", time.time() * 1000),
            strategy_version=self.version,
            threshold_passed=threshold_passed,
            reference_levels={
                "bos": bos,
                "equal_highs": levels["equal_highs"][-5:],
                "equal_lows": levels["equal_lows"][-5:],
                "sweep": sweep,
                "fvg": fvg,
                "rr": round(rr, 4),
                "risk": risk,
                "reward": reward,
                "geometry": geom_reason,
            },
        )

    # -- monitoring posisi aktif (trailing) -----------------------------------
    def monitor_position(
        self, position: Dict[str, Any], candles: Sequence[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Evaluasi posisi aktif untuk trailing. Tujuannya BUKAN mencari
        entry baru, melainkan structure/momentum/weakness monitoring (§18/19).
        """
        p = self.params
        work = list(_last_confirmed(candles))
        if len(work) < p["atr_period"] + 5:
            return {"action": "HOLD", "new_sl": None, "reason": ["data belum cukup"], "weakness_score": 0, "engine": "none"}
        ok, reason_data = validate_candles(work, min_len=p["atr_period"] + 5)
        if not ok:
            return {"action": "HOLD", "new_sl": None, "reason": [f"data invalid: {reason_data}"], "weakness_score": 0, "engine": "none"}

        atrs = atr_series(work, p["atr_period"])
        atr_now = atrs[-1]
        direction = position["direction"]
        entry = position["entry"]
        current_sl = position["sl"]
        tp = position["tp"]
        last = work[-1]
        price = last["c"]

        initial_risk = abs(entry - float(position.get("initial_sl", current_sl))) or atr_now
        risk = initial_risk
        profit_r = (price - entry) / risk if direction == "BUY" else (entry - price) / risk

        reasons: List[str] = []
        weakness = 0

        closes = _closes(work[-p["momentum_lookback"] - 1 :])
        slope, _ = linreg_slope(closes)
        structure_aligned = (direction == "BUY" and slope > 0) or (direction == "SELL" and slope < 0)
        if structure_aligned:
            reasons.append("structure aligned")
        else:
            weakness += 1
            reasons.append("structure melemah")

        opposite_candle = (direction == "BUY" and last["c"] < last["o"]) or (
            direction == "SELL" and last["c"] > last["o"]
        )
        if opposite_candle:
            weakness += 1
            reasons.append("opposite candle")

        fill_time = float(position.get("fill_time", 0.0) or 0.0)
        post_fill = [c for c in work if not fill_time or float(c.get("t", 0.0)) >= fill_time] or list(work[-min(20, len(work)):])
        peak_since_entry = max(_highs(post_fill)) if direction == "BUY" else min(_lows(post_fill))
        giveback = (peak_since_entry - price) / atr_now if direction == "BUY" else (price - peak_since_entry) / atr_now
        if giveback > 0.5:
            weakness += 1
            reasons.append("meaningful giveback")
        if giveback > 1.2:
            weakness += 1
            reasons.append("deep giveback")

        roc = 0.0
        if len(closes) > 1 and closes[0] != 0:
            roc = (closes[-1] - closes[0]) / closes[0]
        momentum_weak = (direction == "BUY" and roc < 0) or (direction == "SELL" and roc > 0)
        if momentum_weak:
            weakness += 1
            reasons.append("predictive trail: momentum")

        action = "HOLD"
        new_sl = None
        if profit_r >= 0.3 and weakness >= 2:
            # geser SL mengikuti struktur, tidak boleh mundur (kurang protektif)
            buffer_ = atr_now * p["sl_atr_buffer"]
            # Hanya swing yang sudah confirmed (memiliki right-side bars) yang boleh menjadi checkpoint trail.
            recent_swings = swing_points(post_fill, p["swing_left"], p["swing_right"]) if len(post_fill) >= (p["swing_left"] + p["swing_right"] + 3) else []
            if direction == "BUY":
                lows = [v for _, v, t in recent_swings if t == "L"]
                structural = max(lows[-3:]) if lows else price - buffer_
                candidate = structural - buffer_ * 0.5
                candidate = min(candidate, price - max(atr_now * 0.05, buffer_ * 0.25))
                if candidate > current_sl:
                    new_sl = candidate
            else:
                highs = [v for _, v, t in recent_swings if t == "H"]
                structural = min(highs[-3:]) if highs else price + buffer_
                candidate = structural + buffer_ * 0.5
                candidate = max(candidate, price + max(atr_now * 0.05, buffer_ * 0.25))
                if candidate < current_sl:
                    new_sl = candidate
            if new_sl is not None:
                action = "TRAIL"

        return {
            "action": action,
            "new_sl": new_sl,
            "reason": reasons,
            "weakness_score": weakness,
            "engine": "momentum",
            "profit_r": profit_r,
        }


def new_default_strategy() -> Strategy:
    return Strategy()

# =============================================================================
# STRATEGY vNEXT EXTENSION
# =============================================================================
# The original implementation above is kept as a compatibility reference.
# The public Strategy symbol is rebound to StrategyVNext below.  This keeps
# old helper imports stable while giving main.py the complete vNext contract.

STRATEGY_SCHEMA_VERSION = 2
SIGNAL_STATUSES = (
    "NO_SETUP", "INVALID_GEOMETRY", "STALE_SETUP", "LOW_EXPECTED_VALUE",
    "TOO_CLOSE", "TOO_FAR", "LOW_LIQUIDITY_CONTEXT", "REGIME_MISMATCH",
    "BTC_CONFLICT", "VALID_LOW_CONF", "VALID_HIGH_CONF",
)
MONITOR_ACTIONS = ("HOLD", "TRAIL", "NO_TRAIL", "EXIT_RISK", "STALE")

VNEXT_DEFAULTS: Dict[str, Any] = {
    "structure_min_swings": 3,
    "structure_age_max_bars": 40,
    "structure_break_buffer_atr": 0.05,
    "trend_r2_min": 0.20,
    "sideways_slope_pct": 0.20,
    "vol_low_pct": 0.15,
    "vol_high_pct": 3.0,
    "vol_extreme_low_rank": 0.05,
    "vol_extreme_high_rank": 0.95,
    "sweep_wick_min_atr": 0.05,
    "sweep_close_reclaim_pct": 0.35,
    "liquidity_distance_max_atr": 4.5,
    "fvg_min_size_atr": 0.08,
    "fvg_max_age_bars": 24,
    "entry_max_distance_atr": 2.25,
    "entry_stale_bars": 18,
    "entry_freshness_half_life": 8.0,
    "entry_likelihood_window": 30,
    "entry_mfe_penalty_atr": 1.5,
    "target_max_atr": 8.0,
    "trail_activation_r": 0.75,
    "trail_min_profit_r": 0.80,
    "trail_weakness_score": 3,
    "trail_structure_buffer_atr": 0.20,
    "trail_min_step_atr": 0.10,
    "trail_max_giveback_atr": 1.0,
    "trail_deep_giveback_atr": 1.5,
    "trail_tp_priority_r": 2.25,
    "trail_protection_floor_r": 0.05,
    "stale_snapshot_seconds": 180.0,
    "max_candle_gap_factor": 2.0,
    "allow_countertrend": True,
    "allow_sideways": True,
    "score_high_confidence": 70.0,
    "score_low_confidence": 45.0,
    "HISTORICAL_EXPECTANCY_R": 0.0,
    "HISTORICAL_TP_RATE": 0.50,
    "HISTORICAL_SL_RATE": 0.50,
    "HISTORICAL_MFE_R": 1.50,
    "HISTORICAL_MAE_R": 1.00,
    "TRAIL_PREFERENCE_SCORE": 0.50,
    "NO_TRAIL_PREFERENCE_SCORE": 0.50,
    # Frequency is observed/audited by learn.py; strategy only exposes the targets.
    "frequency_target_low": 0.05,
    "frequency_target_high": 0.18,
    "frequency_target_ideal": 0.10,
    "frequency_window": 80,
}

VNEXT_WEIGHTS: Dict[str, float] = {
    "structure": 18.0,
    "liquidity": 12.0,
    "entry_quality": 14.0,
    "risk_reward": 13.0,
    "momentum": 9.0,
    "volatility": 5.0,
    "btc_correlation": 9.0,
    "regime": 6.0,
    "session": 3.0,
    "confirmation": 2.0,
    "freshness": 3.0,
    "expected_value": 6.0,
}

# Stable public constants retained from the original strategy brain.
MIN_RR = 2.0
MAX_RR = None
TRAIL_R_LADDER: list = []
STRUCT_TRAIL_LB = 3
STRUCT_TRAIL_BUF_PCT = 0.0025
STRUCT_TRAIL_LOOKBACK = 60
FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618
FINAL_BRAIN_VERSION = "V136_COMBINED_FREQUENCY_LEARNING_REBUILD"
BRAIN_INTERFACE_VERSION = "V128_COIN_ROTATION_BRAIN_PROGRESS"
FULL_LEARNING_SCHEMA = "full_learning_v3_strategy_brain_v2"
MACHINE_LEARNING_SCHEMA = "machine_learning_v4_strategy_brain_v2"
BRAIN_CHECKPOINT_SCHEMA = "brain_progress_checkpoint_v2"


def _vn_clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _vn_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _vn_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _vn_mean(values: Sequence[float], default: float = 0.0) -> float:
    xs = [_vn_float(v) for v in values if math.isfinite(_vn_float(v))]
    return sum(xs) / len(xs) if xs else default


def _vn_median(values: Sequence[float], default: float = 0.0) -> float:
    xs = sorted(_vn_float(v) for v in values if math.isfinite(_vn_float(v)))
    if not xs:
        return default
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _vn_pct_rank(values: Sequence[float], value: float) -> float:
    xs = [_vn_float(v) for v in values if math.isfinite(_vn_float(v))]
    if not xs:
        return 0.5
    if len(xs) == 1:
        return 0.5
    below = sum(1 for x in xs if x < value)
    equal = sum(1 for x in xs if x == value)
    return _vn_clip((below + equal * 0.5) / len(xs))


def _vn_return(values: Sequence[float], lookback: int) -> float:
    lb = max(1, int(lookback))
    if len(values) <= lb:
        return 0.0
    a = _vn_float(values[-lb - 1])
    b = _vn_float(values[-1])
    return 0.0 if a == 0 else (b - a) / a


def _vn_candle_body(c: Dict[str, Any]) -> float:
    return abs(_vn_float(c.get("c")) - _vn_float(c.get("o")))


def _vn_candle_range(c: Dict[str, Any]) -> float:
    return max(0.0, _vn_float(c.get("h")) - _vn_float(c.get("l")))


def _vn_body_ratio(c: Dict[str, Any]) -> float:
    rng = _vn_candle_range(c)
    return 0.0 if rng <= 0 else _vn_clip(_vn_candle_body(c) / rng)


def _vn_upper_wick(c: Dict[str, Any]) -> float:
    return max(0.0, _vn_float(c.get("h")) - max(_vn_float(c.get("o")), _vn_float(c.get("c"))))


def _vn_lower_wick(c: Dict[str, Any]) -> float:
    return max(0.0, min(_vn_float(c.get("o")), _vn_float(c.get("c"))) - _vn_float(c.get("l")))


def vn_validate_data_quality(
    candles: Sequence[Dict[str, Any]],
    expected_interval_ms: float = 900000.0,
    stale_seconds: Optional[float] = None,
    max_gap_factor: float = 2.0,
) -> Dict[str, Any]:
    result = {
        "valid": False, "reason": "UNKNOWN", "candle_count": len(candles or []),
        "duplicate_count": 0, "gap_count": 0, "outlier_count": 0,
        "stale": False, "stale_seconds": 0.0, "median_interval_ms": 0.0,
        "last_timestamp_ms": 0.0,
    }
    if not candles:
        result["reason"] = "INSUFFICIENT_CANDLES"
        return result
    previous = None
    intervals = []
    returns = []
    closes = []
    for i, candle in enumerate(candles):
        if not isinstance(candle, dict):
            result["reason"] = f"MALFORMED_CANDLE_{i}"
            return result
        for key in ("t", "o", "h", "l", "c", "v"):
            if key not in candle:
                result["reason"] = f"MISSING_FIELD_{i}"
                return result
        try:
            t = float(candle["t"]); o = float(candle["o"]); h = float(candle["h"])
            l = float(candle["l"]); c = float(candle["c"]); v = float(candle["v"])
        except (TypeError, ValueError):
            result["reason"] = f"NON_NUMERIC_CANDLE_{i}"
            return result
        if not all(math.isfinite(x) for x in (t, o, h, l, c, v)):
            result["reason"] = f"NON_FINITE_CANDLE_{i}"
            return result
        if min(o, h, l, c) <= 0 or v < 0 or t <= 0 or h < l or h < max(o, c) or l > min(o, c):
            result["reason"] = f"INVALID_CANDLE_RANGE_{i}"
            return result
        if previous is not None:
            delta = t - previous
            if delta <= 0:
                if delta == 0:
                    result["duplicate_count"] += 1
                result["reason"] = f"TIMESTAMP_NOT_ASCENDING_{i}"
                return result
            intervals.append(delta)
            if delta > expected_interval_ms * max_gap_factor:
                result["gap_count"] += 1
        if closes:
            prev_close = closes[-1]
            if prev_close != 0:
                returns.append((c - prev_close) / prev_close)
        closes.append(c)
        previous = t
    result["last_timestamp_ms"] = previous or 0.0
    result["median_interval_ms"] = _vn_median(intervals, expected_interval_ms)
    result["outlier_count"] = sum(1 for x in returns if abs(x) > 0.25)
    if stale_seconds is not None and _vn_float(stale_seconds) > 0:
        result["stale_seconds"] = max(0.0, _vn_float(stale_seconds))
        result["stale"] = result["stale_seconds"] > 180.0
    if result["gap_count"]:
        result["reason"] = "CANDLE_GAP"
        return result
    if result["duplicate_count"]:
        result["reason"] = "DUPLICATE_TIMESTAMPS"
        return result
    if result["stale"]:
        result["reason"] = "STALE_SNAPSHOT"
        return result
    result["valid"] = True
    result["reason"] = "OK"
    return result


def vn_detect_swings(
    candles: Sequence[Dict[str, Any]], left: int, right: int
) -> List[Dict[str, Any]]:
    left = max(1, _vn_int(left, 2)); right = max(1, _vn_int(right, 2))
    highs = [_vn_float(c.get("h")) for c in candles]
    lows = [_vn_float(c.get("l")) for c in candles]
    output = []
    n = len(candles)
    for i in range(left, n - right):
        hw = highs[i - left:i + right + 1]
        lw = lows[i - left:i + right + 1]
        if highs[i] == max(hw) and hw.count(highs[i]) == 1:
            avg_rng = _vn_mean([max(1e-9, _vn_candle_range(c)) for c in candles[i-left:i+right+1]], 1.0)
            output.append({"index": i, "price": highs[i], "type": "H",
                           "prominence": _vn_clip((highs[i] - _vn_median(lw)) / avg_rng, 0, 5)})
        if lows[i] == min(lw) and lw.count(lows[i]) == 1:
            avg_rng = _vn_mean([max(1e-9, _vn_candle_range(c)) for c in candles[i-left:i+right+1]], 1.0)
            output.append({"index": i, "price": lows[i], "type": "L",
                           "prominence": _vn_clip((_vn_median(hw) - lows[i]) / avg_rng, 0, 5)})
    output.sort(key=lambda x: x["index"])
    for row in output:
        row["age_bars"] = max(0, n - row["index"] - 1)
    return output


def vn_swing_hierarchy(swings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    highs = [x for x in swings if x["type"] == "H"]
    lows = [x for x in swings if x["type"] == "L"]
    def rel(xs: Sequence[Dict[str, Any]]) -> str:
        if len(xs) < 2: return "UNKNOWN"
        if xs[-1]["price"] > xs[-2]["price"]: return "UP"
        if xs[-1]["price"] < xs[-2]["price"]: return "DOWN"
        return "EQ"
    hr = rel(highs); lr = rel(lows)
    trend = "SIDEWAYS"
    if hr == "UP" and lr == "UP": trend = "BULLISH"
    elif hr == "DOWN" and lr == "DOWN": trend = "BEARISH"
    elif hr in ("UP", "DOWN") or lr in ("UP", "DOWN"): trend = "MIXED"
    return {"trend": trend, "high_relation": hr, "low_relation": lr,
            "highs": highs[-6:], "lows": lows[-6:]}


def vn_structure_event(
    candles: Sequence[Dict[str, Any]],
    swings: Sequence[Dict[str, Any]],
    atr_now: float,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    if not candles:
        return {"bos": None, "choch": None, "direction": "NEUTRAL", "level": None, "age_bars": 0}
    close = _vn_float(candles[-1].get("c"))
    buffer_ = max(0.0, atr_now * _vn_float(params.get("structure_break_buffer_atr"), 0.05))
    hierarchy = vn_swing_hierarchy(swings)
    highs = [x for x in swings if x["type"] == "H"]
    lows = [x for x in swings if x["type"] == "L"]
    up = highs[-1] if highs and close > highs[-1]["price"] + buffer_ else None
    dn = lows[-1] if lows and close < lows[-1]["price"] - buffer_ else None
    if up and not dn:
        direction = "BUY"
        is_choch = hierarchy["trend"] == "BEARISH"
        return {"bos": None if is_choch else "BOS_UP", "choch": "CHOCH_UP" if is_choch else None,
                "direction": direction, "level": up["price"], "age_bars": up["age_bars"],
                "strength_atr": _vn_clip(abs(close-up["price"])/max(atr_now, 1e-9), 0, 5)}
    if dn and not up:
        direction = "SELL"
        is_choch = hierarchy["trend"] == "BULLISH"
        return {"bos": None if is_choch else "BOS_DOWN", "choch": "CHOCH_DOWN" if is_choch else None,
                "direction": direction, "level": dn["price"], "age_bars": dn["age_bars"],
                "strength_atr": _vn_clip(abs(close-dn["price"])/max(atr_now, 1e-9), 0, 5)}
    return {"bos": None, "choch": None, "direction": "NEUTRAL", "level": None, "age_bars": 0, "strength_atr": 0.0}


def vn_equal_levels(swings: Sequence[Dict[str, Any]], atr_now: float, tol_atr: float) -> Dict[str, List[float]]:
    tolerance = max(1e-12, atr_now * max(0.0, _vn_float(tol_atr, 0.15)))
    out: Dict[str, List[float]] = {"equal_highs": [], "equal_lows": []}
    for key, kind in (("equal_highs", "H"), ("equal_lows", "L")):
        xs = sorted(x["price"] for x in swings if x["type"] == kind)
        i = 0
        while i < len(xs):
            j = i; group = [xs[i]]
            while j + 1 < len(xs) and xs[j+1] - xs[i] <= tolerance:
                j += 1; group.append(xs[j])
            if len(group) >= 2: out[key].append(_vn_mean(group))
            i = j + 1
    return out


def vn_liquidity_sweep(
    candles: Sequence[Dict[str, Any]], lookback: int, atr_now: float, params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if len(candles) < 5: return None
    lb = min(max(5, _vn_int(lookback, 50)), len(candles)-1)
    prior = candles[-lb:-1]; last = candles[-1]
    ph = max(_vn_float(c["h"]) for c in prior); pl = min(_vn_float(c["l"]) for c in prior)
    h = _vn_float(last["h"]); l = _vn_float(last["l"]); c = _vn_float(last["c"])
    rng = max(_vn_candle_range(last), atr_now, 1e-9)
    wick_min = _vn_float(params.get("sweep_wick_min_atr"), 0.05) * max(atr_now, 1e-9)
    reclaim_min = _vn_float(params.get("sweep_close_reclaim_pct"), 0.35)
    if h > ph and c < ph:
        penetration = h - ph; wick = _vn_upper_wick(last); reclaim = (ph-c)/rng
        if wick >= wick_min or penetration >= wick_min:
            quality = _vn_clip(0.45 + 0.35*_vn_clip(reclaim/max(reclaim_min,1e-9)) + 0.20*_vn_clip(penetration/max(atr_now,1e-9)))
            return {"type":"BEARISH_SWEEP","level":ph,"penetration":penetration,"wick":wick,"reclaim":reclaim,"quality":quality}
    if l < pl and c > pl:
        penetration = pl - l; wick = _vn_lower_wick(last); reclaim = (c-pl)/rng
        if wick >= wick_min or penetration >= wick_min:
            quality = _vn_clip(0.45 + 0.35*_vn_clip(reclaim/max(reclaim_min,1e-9)) + 0.20*_vn_clip(penetration/max(atr_now,1e-9)))
            return {"type":"BULLISH_SWEEP","level":pl,"penetration":penetration,"wick":wick,"reclaim":reclaim,"quality":quality}
    return None


def vn_displacement(candles: Sequence[Dict[str, Any]], atr_now: float, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not candles or atr_now <= 0: return None
    c = candles[-1]; body = _vn_candle_body(c); ratio = _vn_body_ratio(c)
    mult = _vn_float(params.get("displacement_atr_mult"), 1.35)
    minimum_ratio = _vn_float(params.get("displacement_body_ratio_min"), 0.55)
    if body >= atr_now*mult and ratio >= minimum_ratio:
        direction = "BUY" if _vn_float(c["c"]) > _vn_float(c["o"]) else "SELL"
        return {"direction":direction,"body":body,"body_atr":body/atr_now,"body_ratio":ratio,"strength":_vn_clip(body/atr_now/max(mult,1e-9),0,2)}
    return None


def vn_fvgs(candles: Sequence[Dict[str, Any]], atr_values: Sequence[float], min_size_atr: float, max_age: int) -> List[Dict[str, Any]]:
    if len(candles) < 3: return []
    result=[]; start=max(0,len(candles)-max(3,_vn_int(max_age,24)+3)); current=len(candles)-1
    for i in range(start,len(candles)-2):
        a,b,c=candles[i],candles[i+1],candles[i+2]
        atr=atr_values[min(i+2,len(atr_values)-1)] if atr_values else 0.0
        if atr<=0: continue
        if _vn_float(a["h"]) < _vn_float(c["l"]):
            bottom=_vn_float(a["h"]); top=_vn_float(c["l"]); size=top-bottom
            if size >= atr*min_size_atr:
                age=current-(i+2); touched=False; fill=0.0
                for cc in candles[i+3:]:
                    if _vn_float(cc["l"]) <= bottom: touched=True; fill=1.0; break
                    fill=max(fill,_vn_clip((top-_vn_float(cc["l"]))/max(size,1e-9)))
                result.append({"type":"BULLISH_FVG","top":top,"bottom":bottom,"size":size,"size_atr":size/atr,"index":i+2,"age_bars":age,"filled_fraction":fill,"touched":touched})
        if _vn_float(a["l"]) > _vn_float(c["h"]):
            top=_vn_float(a["l"]); bottom=_vn_float(c["h"]); size=top-bottom
            if size >= atr*min_size_atr:
                age=current-(i+2); touched=False; fill=0.0
                for cc in candles[i+3:]:
                    if _vn_float(cc["h"]) >= top: touched=True; fill=1.0; break
                    fill=max(fill,_vn_clip((_vn_float(cc["h"])-bottom)/max(size,1e-9)))
                result.append({"type":"BEARISH_FVG","top":top,"bottom":bottom,"size":size,"size_atr":size/atr,"index":i+2,"age_bars":age,"filled_fraction":fill,"touched":touched})
    return result


def vn_fvg_quality(fvg: Optional[Dict[str, Any]], direction: str, params: Dict[str, Any]) -> float:
    if not fvg or fvg.get("type") != ("BULLISH_FVG" if direction=="BUY" else "BEARISH_FVG"): return 0.0
    age=_vn_int(fvg.get("age_bars"),999); max_age=max(1,_vn_int(params.get("fvg_max_age_bars"),24)); fill=_vn_float(fvg.get("filled_fraction"),0.0)
    age_score=_vn_clip(1.0-age/max_age)
    size_score=_vn_clip(_vn_float(fvg.get("size_atr"),0.0))
    fresh=_vn_clip(1.0-0.75*fill)
    return _vn_clip(0.45*age_score+0.35*size_score+0.20*fresh)


def vn_impulse(
    candles: Sequence[Dict[str, Any]], swings: Sequence[Dict[str, Any]], atr_now: float, fib: float, direction: str
) -> Optional[Dict[str, Any]]:
    if atr_now <= 0: return None
    highs=[x for x in swings if x["type"]=="H"]; lows=[x for x in swings if x["type"]=="L"]
    if direction=="BUY":
        candidates=[]
        for low in lows[-10:]:
            later=[h for h in highs if h["index"]>low["index"] and h["price"]>low["price"]]
            if later:
                high=later[-1]; rng=high["price"]-low["price"]
                if rng>0: candidates.append((rng/atr_now,high["index"],low,high))
        if not candidates: return None
        _,_,low,high=max(candidates,key=lambda x:(x[0],x[1]))
        return {"direction":"BUY","low":low["price"],"high":high["price"],"start_index":low["index"],"end_index":high["index"],"range":high["price"]-low["price"],"range_atr":(high["price"]-low["price"])/atr_now,"entry":high["price"]-(high["price"]-low["price"])*fib,"age_bars":len(candles)-1-high["index"]}
    candidates=[]
    for high in highs[-10:]:
        later=[l for l in lows if l["index"]>high["index"] and high["price"]>l["price"]]
        if later:
            low=later[-1]; rng=high["price"]-low["price"]
            if rng>0: candidates.append((rng/atr_now,low["index"],high,low))
    if not candidates: return None
    _,_,high,low=max(candidates,key=lambda x:(x[0],x[1]))
    return {"direction":"SELL","low":low["price"],"high":high["price"],"start_index":high["index"],"end_index":low["index"],"range":high["price"]-low["price"],"range_atr":(high["price"]-low["price"])/atr_now,"entry":low["price"]+(high["price"]-low["price"])*fib,"age_bars":len(candles)-1-low["index"]}


def vn_entry_assessment(
    candles: Sequence[Dict[str, Any]], current: float, entry: float, direction: str, impulse: Dict[str, Any], atr: float, params: Dict[str, Any]
) -> Dict[str, Any]:
    distance=abs(current-entry)/max(atr,1e-9)
    rng=max(_vn_float(impulse.get("range")),1e-9)
    if direction=="BUY": pullback=(impulse["high"]-current)/rng
    else: pullback=(current-impulse["low"])/rng
    retrace=_vn_clip(1.0-abs(pullback-_vn_float(params.get("entry_retracement_fib"),0.618))/0.50)
    min_offset=_vn_float(params.get("entry_min_offset_atr"),0.25)
    max_offset=_vn_float(params.get("entry_max_distance_atr"),2.25)
    too_close=distance<min_offset
    too_far=distance>max_offset
    age=_vn_int(impulse.get("age_bars"),999); stale=age>_vn_int(params.get("entry_stale_bars"),18)
    half=max(1.0,_vn_float(params.get("entry_freshness_half_life"),8.0)); fresh=0.5**(age/half)
    window=candles[-max(5,_vn_int(params.get("entry_likelihood_window"),30)):]
    touched=sum(1 for c in window if _vn_float(c["l"])<=entry<=_vn_float(c["h"]))/max(1,len(window))
    proximity=_vn_clip(1.0-distance/3.0)
    fill=_vn_clip(0.50*_vn_clip(touched*2)+0.25*proximity+0.25*fresh)
    adverse=[]
    for c in window:
        if direction=="BUY" and _vn_float(c["h"])>=entry: adverse.append(max(0.0,(entry-_vn_float(c["l"])))/max(atr,1e-9))
        if direction=="SELL" and _vn_float(c["l"])<=entry: adverse.append(max(0.0,(_vn_float(c["h"])-entry))/max(atr,1e-9))
    adverse_score=_vn_clip(_vn_median(adverse,0.5)/2.0)
    return {"distance_atr":distance,"pullback_fraction":pullback,"retracement_quality":_vn_clip(0.60*retrace+0.40*proximity),"fill_likelihood":fill,"freshness":fresh,"adverse_excursion_risk":adverse_score,"too_close":too_close,"too_far":too_far,"stale":stale}


def vn_btc_alignment(
    symbol: str, direction: str, candles: Sequence[Dict[str, Any]], btc: Optional[Sequence[Dict[str, Any]]], params: Dict[str, Any], regime: str
) -> Dict[str, Any]:
    if symbol.upper()=="BTCUSDT" or not btc:
        return {"available":bool(btc),"correlation":1.0 if symbol.upper()=="BTCUSDT" else 0.0,"direction":direction if symbol.upper()=="BTCUSDT" else "NEUTRAL","aligned":True,"conflict":False,"alignment_score":1.0 if symbol.upper()=="BTCUSDT" else 0.5,"regime_alignment":1.0 if regime=="SIDEWAYS" else 0.5,"relative_strength":0.0,"reason":["BTC self-context"]}
    lb=max(5,_vn_int(params.get("btc_corr_lookback"),50))
    cr=correlation(pct_returns(_closes(candles)[-lb:]),pct_returns(_closes(btc)[-lb:]))
    btc_sample=_closes(btc)[-max(5,_vn_int(params.get("trend_lookback"),30)):]
    slope,r2=linreg_slope(btc_sample); bd="BUY" if slope>0 else "SELL" if slope<0 else "NEUTRAL"
    coinret=_vn_return(_closes(candles),_vn_int(params.get("btc_relative_strength_window"),20)); btcret=_vn_return(_closes(btc),_vn_int(params.get("btc_relative_strength_window"),20))
    rel=coinret-btcret; aligned=False; conflict=False; score=0.5; reasons=[]
    if cr>=_vn_float(params.get("btc_alignment_corr_min"),0.30) and bd==direction and r2>=_vn_float(params.get("btc_trend_r2_min"),0.15):
        aligned=True; score=_vn_clip(0.55+0.45*cr); reasons.append(f"BTC aligned corr={cr:.2f}")
    elif cr>=_vn_float(params.get("btc_conflict_corr_min"),0.55) and bd in ("BUY","SELL") and bd!=direction:
        conflict=True; score=_vn_clip(0.40-0.25*cr); reasons.append(f"BTC conflict corr={cr:.2f}")
    elif cr< -_vn_float(params.get("btc_alignment_corr_min"),0.30) and bd!=direction:
        aligned=True; score=_vn_clip(0.55+0.30*abs(cr)); reasons.append(f"BTC negative-corr divergence={cr:.2f}")
    else: reasons.append(f"BTC neutral corr={cr:.2f}")
    if regime=="BULLISH_TREND": regime_alignment=1.0 if direction=="BUY" else 0.2
    elif regime=="BEARISH_TREND": regime_alignment=1.0 if direction=="SELL" else 0.2
    elif regime=="SIDEWAYS": regime_alignment=0.55
    else: regime_alignment=0.35
    return {"available":True,"correlation":cr,"direction":bd,"btc_r2":r2,"aligned":aligned,"conflict":conflict,"alignment_score":score,"regime_alignment":regime_alignment,"relative_strength":rel,"coin_return":coinret,"btc_return":btcret,"reason":reasons}


def vn_target_probability(rr: float, momentum: float, fvg: float, liquidity: bool, params: Dict[str, Any]) -> float:
    base=max(0.5,_vn_float(params.get("HISTORICAL_MFE_R"),1.5))
    p=math.exp(-max(0.0,rr-base)/max(0.75,base))
    p*=1.10 if momentum>=0.60 else 0.90
    p*=1.08 if fvg>=0.45 else 0.94
    p*=1.10 if liquidity else 0.90
    return _vn_clip(p,0.05,0.95)


def vn_build_tp(direction: str, entry: float, risk: float, atr: float, liquidity: Dict[str, Any], swings: Sequence[Dict[str, Any]], params: Dict[str, Any], momentum: float, fvg_score: float) -> Dict[str, Any]:
    highs=[x["price"] for x in swings if x["type"]=="H"]; lows=[x["price"] for x in swings if x["type"]=="L"]
    eql=liquidity.get("equal_highs",[]); eqs=liquidity.get("equal_lows",[])
    if direction=="BUY":
        candidates=[x for x in eql+highs if x>entry]
        candidates=[x for x in candidates if abs(x-entry)/max(atr,1e-9)<=_vn_float(params.get("target_max_atr"),8.0)]
        tp=min(candidates) if candidates else entry+max(risk*2.0,atr*1.5); is_liq=tp in eql
    else:
        candidates=[x for x in eqs+lows if x<entry]
        candidates=[x for x in candidates if abs(x-entry)/max(atr,1e-9)<=_vn_float(params.get("target_max_atr"),8.0)]
        tp=max(candidates) if candidates else entry-max(risk*2.0,atr*1.5); is_liq=tp in eqs
    reward=abs(tp-entry); rr=reward/max(risk,1e-9); pr=vn_target_probability(rr,momentum,fvg_score,is_liq,params); expected=pr*rr-(1-pr)
    dist=reward/max(atr,1e-9); quality=_vn_clip(0.30*(1.0 if is_liq else 0.35)+0.25*_vn_clip(1-dist/8.0)+0.20*_vn_clip(rr/3.0)+0.25*pr)
    return {"tp":tp,"rr":rr,"reward":reward,"reach_probability":pr,"expected_r":expected,"quality":quality,"liquidity_target":is_liq,"distance_atr":dist}


def vn_build_sl(direction: str, entry: float, atr: float, impulse: Dict[str, Any], swings: Sequence[Dict[str, Any]], sweep: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    buffer_=atr*_vn_float(params.get("sl_atr_buffer"),0.25); guard=atr*_vn_float(params.get("sl_wick_guard_atr"),0.10)
    min_r=atr*_vn_float(params.get("sl_min_atr"),0.45); max_r=atr*_vn_float(params.get("sl_max_atr"),3.5)
    lows=[x["price"] for x in swings if x["type"]=="L"]; highs=[x["price"] for x in swings if x["type"]=="H"]
    if direction=="BUY":
        structural=impulse.get("low",min(lows[-3:]) if lows else entry-atr)
        if sweep and sweep.get("type")=="BULLISH_SWEEP": structural=min(structural,_vn_float(sweep["level"],structural))
        sl=structural-buffer_-guard
    else:
        structural=impulse.get("high",max(highs[-3:]) if highs else entry+atr)
        if sweep and sweep.get("type")=="BEARISH_SWEEP": structural=max(structural,_vn_float(sweep["level"],structural))
        sl=structural+buffer_+guard
    risk=abs(entry-sl)
    if risk<min_r: sl=entry-min_r if direction=="BUY" else entry+min_r; risk=min_r
    if risk>max_r: sl=entry-max_r if direction=="BUY" else entry+max_r; risk=max_r
    risk_atr=risk/max(atr,1e-9); hist_mae=max(0.5,_vn_float(params.get("HISTORICAL_MAE_R"),1.0)); mae_score=_vn_clip(1-abs(risk_atr-hist_mae)/max(1.0,hist_mae)); wick_risk=_vn_clip(1-(_vn_clip(1-abs(risk_atr-1)/2)))
    quality=_vn_clip(0.35*_vn_clip(1-abs(risk_atr-1)/2.5)+0.20*(0.80 if sweep else 0.35)+0.25*mae_score+0.20*(1-wick_risk))
    return {"sl":sl,"risk":risk,"risk_atr":risk_atr,"structural_score":_vn_clip(1-abs(risk_atr-1)/2.5),"mae_score":mae_score,"wickout_risk":wick_risk,"quality":quality}


def vn_component_scores(
    structure_signal: float, liquidity_signal: float, entry_signal: float, rr_signal: float,
    momentum_signal: float, volatility_signal: float, btc_signal: float, regime_signal: float,
    session_signal: float, confirmation_signal: float, freshness_signal: float, ev_signal: float,
    params: Dict[str, Any]
) -> Dict[str, float]:
    weights=dict(VNEXT_WEIGHTS)
    custom=params.get("CONFIDENCE_WEIGHTS")
    if isinstance(custom,dict):
        for k,v in custom.items():
            if k in weights and _vn_float(v,-1)<0: continue
            if k in weights: weights[k]=_vn_float(v,weights[k])
    total=sum(weights.values()) or 100.0
    factor=100.0/total
    weights={k:v*factor for k,v in weights.items()}
    sig={"structure":structure_signal,"liquidity":liquidity_signal,"entry_quality":entry_signal,"risk_reward":rr_signal,"momentum":momentum_signal,"volatility":volatility_signal,"btc_correlation":btc_signal,"regime":regime_signal,"session":session_signal,"confirmation":confirmation_signal,"freshness":freshness_signal,"expected_value":ev_signal}
    return {k:_vn_clip(sig[k])*weights[k] for k in weights}


def vn_dynamic_confidence(
    components: Dict[str,float], entry: Dict[str,Any], tp: Dict[str,Any], sl: Dict[str,Any],
    regime_ok: bool, btc_conflict: bool, params: Dict[str,Any]
) -> Dict[str,Any]:
    raw=sum(components.values()); gate=1.0
    if not regime_ok: gate*=0.88
    if btc_conflict: gate*=0.72
    if entry.get("stale"): gate*=0.70
    if entry.get("too_far"): gate*=0.72
    if entry.get("too_close"): gate*=0.76
    if _vn_float(tp.get("expected_r"),0)<0: gate*=0.72
    if _vn_float(sl.get("wickout_risk"),0)>0.75: gate*=0.90
    final=_vn_clip(raw*gate,0,100)
    setup_keys=("structure","liquidity","entry_quality","risk_reward","confirmation")
    context_keys=("momentum","volatility","btc_correlation","regime","session")
    setup_score=sum(components.get(k,0) for k in setup_keys)/max(1,sum(VNEXT_WEIGHTS[k] for k in setup_keys))*100
    context_score=sum(components.get(k,0) for k in context_keys)/max(1,sum(VNEXT_WEIGHTS[k] for k in context_keys))*100
    execution_score=_vn_clip(0.40*entry.get("fill_likelihood",0)+0.25*entry.get("retracement_quality",0)+0.20*tp.get("quality",0)+0.15*sl.get("quality",0))*100
    freshness_score=_vn_clip(entry.get("freshness",0))*100
    ev_score=_vn_clip(0.5+_vn_float(tp.get("expected_r"),0)/max(2,abs(_vn_float(tp.get("rr"),0))+1)*0.5)*100
    return {"final":final,"setup_quality":setup_score,"execution":execution_score,"context":context_score,"freshness":freshness_score,"expected_value":ev_score,"gate":gate}


def vn_diagnosis(
    setup: bool, geometry_ok: bool, entry: Optional[Dict[str,Any]], tp: Optional[Dict[str,Any]], confidence: float,
    regime_ok: bool, btc_conflict: bool, low_liquidity: bool, data_quality: Dict[str,Any], params: Dict[str,Any]
) -> str:
    if not setup: return "NO_SETUP"
    if data_quality.get("stale"): return "STALE_SETUP"
    if not geometry_ok: return "INVALID_GEOMETRY"
    if entry and entry.get("too_close"): return "TOO_CLOSE"
    if entry and entry.get("too_far"): return "TOO_FAR"
    if entry and entry.get("stale"): return "STALE_SETUP"
    if low_liquidity: return "LOW_LIQUIDITY_CONTEXT"
    if btc_conflict: return "BTC_CONFLICT"
    if not regime_ok: return "REGIME_MISMATCH"
    if tp and _vn_float(tp.get("expected_r"),0)<0: return "LOW_EXPECTED_VALUE"
    return "VALID_HIGH_CONF" if confidence>=_vn_float(params.get("score_high_confidence"),70) else "VALID_LOW_CONF"


def validate_trailing_geometry(direction: str, current_sl: float, proposed_sl: float, price: float, entry: float, tp: float) -> Tuple[bool, str]:
    """Safety geometry for trailing stop updates. Never permits a less-protective or crossed stop."""
    d = str(direction or "").upper()
    cur = _vn_float(current_sl)
    new = _vn_float(proposed_sl)
    px = _vn_float(price)
    en = _vn_float(entry)
    target = _vn_float(tp)
    if not all(math.isfinite(x) for x in (cur, new, px, en, target)):
        return False, "NON_FINITE_TRAIL_GEOMETRY"
    if d == "BUY":
        if cur >= en:
            # Once protected above entry, a new SL cannot go back below entry.
            if new < cur:
                return False, "TRAIL_WOULD_REDUCE_PROTECTION"
        if new >= px:
            return False, "TRAIL_SL_NOT_BELOW_PRICE"
        if target > en and new >= target:
            return False, "TRAIL_SL_CROSSES_TP"
        if new <= 0 or en <= 0:
            return False, "INVALID_PRICE_GEOMETRY"
        return True, "OK"
    if d == "SELL":
        if cur <= en:
            if new > cur:
                return False, "TRAIL_WOULD_REDUCE_PROTECTION"
        if new <= px:
            return False, "TRAIL_SL_NOT_ABOVE_PRICE"
        if target < en and new <= target:
            return False, "TRAIL_SL_CROSSES_TP"
        if new <= 0 or en <= 0:
            return False, "INVALID_PRICE_GEOMETRY"
        return True, "OK"
    return False, "INVALID_DIRECTION"


class StrategyVNext:
    """Full Strategy vNext engine; input-only, deterministic analysis."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged=dict(DEFAULT_PARAMS)
        merged.update(VNEXT_DEFAULTS)
        if params: merged.update(params)
        merged["CONFIDENCE_WEIGHTS"]=dict(params.get("CONFIDENCE_WEIGHTS", VNEXT_WEIGHTS)) if isinstance(params,dict) and isinstance(params.get("CONFIDENCE_WEIGHTS"),dict) else dict(VNEXT_WEIGHTS)
        self.params=merged
        self.version="2.00"
        self.version_history=[{"version":self.version,"timestamp":time.time(),"reason":"INITIAL_VNEXT","old_params":None,"new_params":dict(self.params),"evidence":None}]
        self.last_diagnostics={}

    def get_active_threshold(self)->float:
        # Adaptive threshold may move, but execution must never drop below the hard safety floor.
        raw = _vn_float(self.params.get("ACTIVE_THRESHOLD"), EXECUTION_CONFIDENCE_FLOOR)
        return max(EXECUTION_CONFIDENCE_FLOOR, min(100.0, raw))

    def export_state(self)->Dict[str,Any]:
        return {"schema_version":STRATEGY_SCHEMA_VERSION,"strategy_name":STRATEGY_NAME,"version":self.version,"params":dict(self.params),"version_history":list(self.version_history)}

    def load_state(self,state:Dict[str,Any])->None:
        if not isinstance(state,dict): return
        if isinstance(state.get("params"),dict):
            self.params.update(state["params"])
            if isinstance(state["params"].get("CONFIDENCE_WEIGHTS"),dict): self.params["CONFIDENCE_WEIGHTS"]=dict(state["params"]["CONFIDENCE_WEIGHTS"])
        # Migrate legacy checkpoints that allowed an execution threshold of 0.
        self.params["ACTIVE_THRESHOLD"] = max(EXECUTION_CONFIDENCE_FLOOR, min(100.0, _vn_float(self.params.get("ACTIVE_THRESHOLD"), EXECUTION_CONFIDENCE_FLOOR)))
        if isinstance(state.get("version"),str): self.version=state["version"]
        if isinstance(state.get("version_history"),list) and state["version_history"]: self.version_history=list(state["version_history"])

    def _validate_update(self, updates: Dict[str,Any])->Tuple[bool,str]:
        if not isinstance(updates,dict) or not updates: return False,"EMPTY_UPDATE"
        known=set(self.params)|set(DEFAULT_PARAMS)|set(VNEXT_DEFAULTS)
        for key in updates:
            if key not in known: return False,f"UNKNOWN_PARAM:{key}"
        for key,value in updates.items():
            if key=="CONFIDENCE_WEIGHTS":
                if not isinstance(value,dict) or not value: return False,"INVALID_CONFIDENCE_WEIGHTS"
                if any(_vn_float(v,-1)<0 for v in value.values()): return False,"NEGATIVE_CONFIDENCE_WEIGHT"
                if sum(_vn_float(v,0) for v in value.values())<=0: return False,"ZERO_CONFIDENCE_WEIGHT_SUM"
        return True,"OK"

    def apply_update(self,new_params:Dict[str,Any],reason:str,evidence:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
        ok,msg=self._validate_update(new_params)
        if not ok: raise ValueError(msg)
        old=dict(self.params); merged=dict(self.params)
        if "CONFIDENCE_WEIGHTS" in new_params:
            w=dict(self.params.get("CONFIDENCE_WEIGHTS",VNEXT_WEIGHTS)); w.update(new_params["CONFIDENCE_WEIGHTS"]); merged["CONFIDENCE_WEIGHTS"]=w
        for k,v in new_params.items():
            if k!="CONFIDENCE_WEIGHTS": merged[k]=v
        self.params=merged
        major,minor=self.version.split(".",1); self.version=f"{major}.{int(minor)+1:02d}"
        record={"version":self.version,"timestamp":time.time(),"reason":reason or "UNSPECIFIED","old_params":old,"new_params":dict(self.params),"evidence":evidence}
        self.version_history.append(record); return record

    def rollback(self)->Optional[Dict[str,Any]]:
        if len(self.version_history)<2: return None
        self.version_history.pop(); previous=self.version_history[-1]
        if isinstance(previous.get("new_params"),dict): self.params=dict(previous["new_params"])
        self.version=str(previous.get("version",self.version)); return previous

    def _market_regime(self,btc:Sequence[Dict[str,Any]],coin:Sequence[Dict[str,Any]])->str:
        source=btc if btc else coin; p=self.params
        return classify_regime(source,p)

    def analyze_with_diagnostics(
        self,symbol:str,candles:Sequence[Dict[str,Any]],btc_candles:Optional[Sequence[Dict[str,Any]]]=None,
        market_context:Optional[Dict[str,Any]]=None,enforce_threshold:bool=True,current_timestamp_ms:Optional[float]=None,
    )->Tuple[Optional[Setup],Dict[str,Any]]:
        symbol = str(symbol or "").upper()
        if symbol in TRADE_EXCLUDED_SYMBOLS:
            diagnostics = {"strategy_version": self.version, "symbol": symbol, "status": "BTC_CONTEXT_ONLY",
                           "data_quality": {}, "structure": {}, "liquidity": {}, "entry": {}, "tp": {},
                           "sl": {}, "momentum": {}, "volatility": {}, "btc": {}, "market": {},
                           "score": {}, "reasons": ["BTCUSDT context-only; trade disabled"],
                           "threshold": {"active": self.get_active_threshold(), "passed": False}}
            self.last_diagnostics = diagnostics
            return None, diagnostics
        p=self.params; work=list(_last_confirmed(candles)); btc=list(_last_confirmed(btc_candles or []))
        min_len=max(_vn_int(p.get("structure_lookback"),80),_vn_int(p.get("vol_regime_lookback"),100),_vn_int(p.get("atr_period"),14))+5
        quality=vn_validate_data_quality(work,max_gap_factor=_vn_float(p.get("max_candle_gap_factor"),2.0),stale_seconds=((current_timestamp_ms-work[-1]["t"])/1000.0 if current_timestamp_ms and work else None)) if work else {"valid":False,"reason":"INSUFFICIENT_CANDLES","candle_count":0}
        diagnostics={"strategy_version":self.version,"symbol":symbol,"status":"NO_SETUP","data_quality":quality,"structure":{},"liquidity":{},"entry":{},"tp":{},"sl":{},"momentum":{},"volatility":{},"btc":{},"market":{},"score":{},"reasons":[]}
        if len(work)<min_len or not quality.get("valid"):
            diagnostics["status"]="STALE_SETUP" if quality.get("reason")=="STALE_SNAPSHOT" else "NO_SETUP"
            diagnostics["reasons"].append(quality.get("reason","INSUFFICIENT_CANDLES")); self.last_diagnostics=diagnostics; return None,diagnostics
        atrs=atr_series(work,_vn_int(p.get("atr_period"),14)); atr=atrs[-1] if atrs else 0.0
        if atr<=0:
            diagnostics["reasons"].append("ATR_INVALID"); self.last_diagnostics=diagnostics; return None,diagnostics
        closes=_closes(work); current=closes[-1]
        struct_window=work[-_vn_int(p.get("structure_lookback"),120):]
        swings=vn_detect_swings(struct_window,_vn_int(p.get("swing_left"),2),_vn_int(p.get("swing_right"),2))
        offset=len(work)-len(struct_window)
        if offset:
            for s in swings: s["index"]+=offset
        event=vn_structure_event(work,swings,atr,p); hierarchy=vn_swing_hierarchy(swings)
        slope,r2=linreg_slope(closes[-_vn_int(p.get("trend_lookback"),30):]); trend_dir="BUY" if slope>0 else "SELL" if slope<0 else "NEUTRAL"
        diagnostics["structure"]={"bos":event.get("bos"),"choch":event.get("choch"),"direction":event.get("direction"),"level":event.get("level"),"age_bars":event.get("age_bars"),"strength_atr":event.get("strength_atr"),"hierarchy":hierarchy,"trend_direction":trend_dir,"trend_r2":r2}
        if event["direction"]=="NEUTRAL":
            diagnostics["reasons"].append("NO_STRUCTURE_BREAK_OR_CHOCH"); self.last_diagnostics=diagnostics; return None,diagnostics
        direction=event["direction"]
        regime=self._market_regime(btc,work); session=classify_session(work[-1].get("t",0))
        momentum_fast=_vn_return(closes,_vn_int(p.get("momentum_fast"),5)); momentum_main=_vn_return(closes,_vn_int(p.get("momentum_lookback"),10)); momentum_slow=_vn_return(closes,_vn_int(p.get("momentum_slow"),20))
        momentum_dir="BUY" if momentum_main>0 else "SELL" if momentum_main<0 else "NEUTRAL"
        momentum_alignment=(_vn_clip((1 if (direction=="BUY" and momentum_fast>0) or (direction=="SELL" and momentum_fast<0) else 0)*0.35 + (1 if (direction=="BUY" and momentum_main>0) or (direction=="SELL" and momentum_main<0) else 0)*0.35 + (1 if (direction=="BUY" and momentum_slow>0) or (direction=="SELL" and momentum_slow<0) else 0)*0.20 + (1 if trend_dir==direction else 0)*0.10))
        diagnostics["momentum"]={"roc_fast":momentum_fast,"roc_main":momentum_main,"roc_slow":momentum_slow,"direction":momentum_dir,"alignment":momentum_alignment}
        vol_rank=_vn_pct_rank(atrs[-_vn_int(p.get("vol_regime_lookback"),100):],atr); vol_regime=classify_volatility_regime(work,p)
        normality=1.0 if 0.15<=vol_rank<=0.90 and vol_regime=="NORMAL" else 0.35 if vol_rank<0.05 or vol_rank>0.95 else 0.65
        diagnostics["volatility"]={"atr":atr,"atr_percentile":vol_rank,"regime":vol_regime,"normality":normality}
        pools=vn_equal_levels(swings,atr,_vn_float(p.get("equal_level_tol_atr"),0.15)); sweep=vn_liquidity_sweep(work,_vn_int(p.get("sweep_lookback"),50),atr,p); pools["sweep"]=sweep
        pools["nearest_equal_high"]=min([x for x in pools["equal_highs"] if x>current],default=None); pools["nearest_equal_low"]=max([x for x in pools["equal_lows"] if x<current],default=None)
        diagnostics["liquidity"]=pools
        disp=vn_displacement(work,atr,p); fvg_values=vn_fvgs(work,atrs,_vn_float(p.get("fvg_min_size_atr"),0.08),_vn_int(p.get("fvg_max_age_bars"),24)); aligned=[x for x in fvg_values if x["type"]==("BULLISH_FVG" if direction=="BUY" else "BEARISH_FVG")]; fvg=aligned[-1] if aligned else None; fvg_score=vn_fvg_quality(fvg,direction,p)
        impulse=vn_impulse(work,swings,atr,_vn_float(p.get("entry_retracement_fib"),0.618),direction)
        if not impulse:
            diagnostics["status"]="NO_SETUP"; diagnostics["reasons"].append("NO_USABLE_IMPULSE"); self.last_diagnostics=diagnostics; return None,diagnostics
        entry=_vn_float(impulse["entry"])
        if direction=="BUY" and current-entry < atr*_vn_float(p.get("entry_min_offset_atr"),0.25): entry=current-atr*_vn_float(p.get("entry_min_offset_atr"),0.25)
        if direction=="SELL" and entry-current < atr*_vn_float(p.get("entry_min_offset_atr"),0.25): entry=current+atr*_vn_float(p.get("entry_min_offset_atr"),0.25)
        entry_info=vn_entry_assessment(work,current,entry,direction,impulse,atr,p); diagnostics["entry"]={**entry_info,"entry":entry,"impulse":impulse}
        regime_ok=direction=="BUY" if regime=="BULLISH_TREND" else direction=="SELL" if regime=="BEARISH_TREND" else (regime=="SIDEWAYS" and bool(p.get("allow_sideways",True))) or regime not in ("LOW_VOLATILITY","HIGH_VOLATILITY")
        btc_info=vn_btc_alignment(symbol,direction,work,btc,p,regime); diagnostics["btc"]=btc_info
        sl=vn_build_sl(direction,entry,atr,impulse,swings,sweep,p); diagnostics["sl"]=sl
        tp=vn_build_tp(direction,entry,sl["risk"],atr,pools,swings,p,momentum_alignment,fvg_score); diagnostics["tp"]=tp
        # HARD EXECUTION GATES: bad RR/EV is never a tradable setup.
        hard_rejects=[]
        rr_value=_vn_float(tp.get("rr"),0.0)
        expected_r=_vn_float(tp.get("expected_r"),0.0)
        if rr_value < max(EXECUTION_MIN_RR_FLOOR, _vn_float(p.get("min_rr"), EXECUTION_MIN_RR_FLOOR)):
            hard_rejects.append(f"RR_BELOW_MIN:{rr_value:.2f}")
        if expected_r <= 0.0:
            hard_rejects.append(f"NEGATIVE_EXPECTED_R:{expected_r:.3f}")
        if entry_info.get("stale"):
            hard_rejects.append("STALE_ENTRY")
        diagnostics["hard_gates"]={"passed":not hard_rejects,"rejects":hard_rejects,"min_rr":max(EXECUTION_MIN_RR_FLOOR,_vn_float(p.get("min_rr"),EXECUTION_MIN_RR_FLOOR))}
        if hard_rejects:
            diagnostics["status"]="LOW_EXPECTED_VALUE" if any(x.startswith(("RR_BELOW_MIN","NEGATIVE_EXPECTED_R")) for x in hard_rejects) else "STALE_SETUP"
            diagnostics["reasons"].extend(hard_rejects)
        geom_ok,geom_reason=validate_geometry(direction,entry,sl["sl"],tp["tp"],atr_val=atr); diagnostics["geometry"]={"valid":geom_ok,"reason":geom_reason}
        if not geom_ok:
            diagnostics["status"]="INVALID_GEOMETRY"; diagnostics["reasons"].append(geom_reason); self.last_diagnostics=diagnostics; return None,diagnostics
        liquidity_signal=0.0
        if sweep and sweep.get("type")==("BULLISH_SWEEP" if direction=="BUY" else "BEARISH_SWEEP"): liquidity_signal+=0.65*_vn_float(sweep.get("quality"),0)
        if direction=="BUY" and pools.get("nearest_equal_high") is not None: liquidity_signal+=0.35
        if direction=="SELL" and pools.get("nearest_equal_low") is not None: liquidity_signal+=0.35
        structure_signal=0.35 + 0.35*(1 if event["direction"]==direction else 0) + 0.20*(1 if hierarchy["trend"]==("BULLISH" if direction=="BUY" else "BEARISH") else 0) + 0.10*(1 if trend_dir==direction else 0); structure_signal*=(0.55+0.45*r2)
        entry_signal=0.40*entry_info["retracement_quality"]+0.20*entry_info["fill_likelihood"]+0.15*entry_info["freshness"]+0.15*(1-entry_info["adverse_excursion_risk"])+0.10*fvg_score
        rr_signal=_vn_clip(0.50*_vn_float(tp["rr"])/max(1.0,_vn_float(p.get("min_rr"),1.2)*2)+0.25*tp["quality"]+0.25*sl["quality"])
        session_signal=1.0 if session in ("LONDON","NEWYORK") else 0.35
        confirmation=0.35*(1 if sweep else 0)+0.25*(1 if disp and disp.get("direction")==direction else 0)+0.25*fvg_score+0.15*momentum_alignment
        freshness=_vn_clip(0.65*entry_info["freshness"]+0.35*(1-event.get("age_bars",0)/max(1,_vn_int(p.get("structure_age_max_bars"),40))))
        ev_signal=_vn_clip(0.5+_vn_float(tp.get("expected_r"),0)/max(2,abs(_vn_float(tp.get("rr"),0))+1)*0.5)
        components=vn_component_scores(structure_signal,liquidity_signal,entry_signal,rr_signal,momentum_alignment,normality,btc_info.get("alignment_score",0.5),btc_info.get("regime_alignment",0.5),session_signal,confirmation,freshness,ev_signal,p)
        score=vn_dynamic_confidence(components,entry_info,tp,sl,regime_ok,bool(btc_info.get("conflict")),p)
        low_liq=vol_rank<=0.10 and not sweep
        status=vn_diagnosis(True,geom_ok,entry_info,tp,score["final"],regime_ok,bool(btc_info.get("conflict")),low_liq,quality,p)
        diagnostics["market"]={"regime":regime,"session":session,"breadth":dict(market_context or {}),"regime_ok":regime_ok}
        diagnostics["score"]={**score,"components":components}
        diagnostics["status"]=status
        threshold=self.get_active_threshold(); passed=bool(not hard_rejects and score["final"]>=threshold)
        diagnostics["threshold"]={"active":threshold,"passed":passed}
        reasons=[f"{event.get('bos') or event.get('choch')} {direction}",f"trend slope={'aligned' if trend_dir==direction else 'opposed'}",f"entry OTE={_vn_float(p.get('entry_retracement_fib'),0.618)*100:.0f}%",f"RR={tp['rr']:.2f}",f"expectedR={tp['expected_r']:.2f}",f"viability={status}"]
        if sweep: reasons.append(f"sweep={sweep['type']}")
        if fvg: reasons.append("fresh FVG")
        if btc_info.get("aligned"): reasons.append("BTC aligned")
        setup=Setup(pair=symbol,direction=direction,entry=entry,tp=tp["tp"],sl=sl["sl"],confidence=score["final"],reason=reasons,components=components,setup_type="+".join([x for x in (event.get("bos") or event.get("choch") or "STRUCTURE", "SWEEP" if sweep and sweep.get("type")==("BULLISH_SWEEP" if direction=="BUY" else "BEARISH_SWEEP") else "", "DISPLACEMENT" if disp and disp.get("direction")==direction else "", "FVG" if fvg else "") if x]),regime=regime,session=session,atr=atr,timestamp=_vn_float(work[-1].get("t")),strategy_version=self.version,threshold_passed=passed,reference_levels={"bos":event.get("bos"),"choch":event.get("choch"),"broken_level":event.get("level"),"swing_hierarchy":hierarchy,"equal_highs":pools.get("equal_highs",[])[-5:],"equal_lows":pools.get("equal_lows",[])[-5:],"sweep":sweep,"fvg":fvg,"impulse":impulse,"rr":tp["rr"],"expected_r":tp["expected_r"],"tp_reach_probability":tp["reach_probability"],"entry_distance_atr":entry_info["distance_atr"],"fill_likelihood":entry_info["fill_likelihood"],"stale":entry_info["stale"],"geometry":geom_reason,"diagnosis":status,"btc_correlation":btc_info.get("correlation"),"btc_aligned":btc_info.get("aligned")},viability=status,quality_score=score["setup_quality"],execution_score=score["execution"],context_score=score["context"],freshness_score=score["freshness"],expected_value_score=score["expected_value"])
        self.last_diagnostics=diagnostics
        if enforce_threshold and not passed: return None,diagnostics
        return setup,diagnostics

    def analyze(self,symbol:str,candles:Sequence[Dict[str,Any]],btc_candles:Optional[Sequence[Dict[str,Any]]]=None,enforce_threshold:bool=True)->Optional[Setup]:
        setup,_=self.analyze_with_diagnostics(symbol,candles,btc_candles=btc_candles,enforce_threshold=enforce_threshold)
        return setup

    def monitor_position(self,position:Dict[str,Any],candles:Sequence[Dict[str,Any]],btc_candles:Optional[Sequence[Dict[str,Any]]]=None,market_context:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
        p=self.params; work=list(_last_confirmed(candles))
        if len(work)<_vn_int(p.get("atr_period"),14)+5:
            return {"action":"HOLD","new_sl":None,"reason":["data belum cukup"],"weakness_score":0,"engine":"vnext","profit_r":0.0,"trigger":"INSUFFICIENT_DATA","tp_still_superior":True,"trail_statistically_preferable":False}
        quality=vn_validate_data_quality(work,max_gap_factor=_vn_float(p.get("max_candle_gap_factor"),2.0))
        if not quality["valid"]:
            return {"action":"STALE" if quality["reason"]=="STALE_SNAPSHOT" else "HOLD","new_sl":None,"reason":[quality["reason"]],"weakness_score":0,"engine":"vnext","profit_r":0.0,"trigger":quality["reason"],"tp_still_superior":True,"trail_statistically_preferable":False,"data_quality":quality}
        atrs=atr_series(work,_vn_int(p.get("atr_period"),14)); atr=atrs[-1]; direction=str(position.get("direction","BUY")).upper(); entry=_vn_float(position.get("fill_price",position.get("entry"))); current_sl=_vn_float(position.get("protected_sl",position.get("sl"))); tp=_vn_float(position.get("tp")); initial_sl=_vn_float(position.get("initial_sl",current_sl)); risk=abs(entry-initial_sl) or atr; price=_vn_float(work[-1].get("c")); profit_r=((price-entry)/risk if direction=="BUY" else (entry-price)/risk)
        fill_time=_vn_float(position.get("fill_time"),0.0); path=[c for c in work if not fill_time or _vn_float(c.get("t"))>=fill_time] or work[-20:]
        short=work[-max(6,_vn_int(p.get("trail_momentum_lookback"),6)):]; slope,r2=linreg_slope(_closes(short)); aligned=(direction=="BUY" and slope>0) or (direction=="SELL" and slope<0); roc=_vn_return(_closes(work),_vn_int(p.get("momentum_lookback"),10)); momentum_weak=(direction=="BUY" and roc<0) or (direction=="SELL" and roc>0); last=work[-1]; opposite=(direction=="BUY" and last["c"]<last["o"]) or (direction=="SELL" and last["c"]>last["o"]); peak=max(_vn_float(c["h"]) for c in path); trough=min(_vn_float(c["l"]) for c in path); giveback=(peak-price)/max(atr,1e-9) if direction=="BUY" else (price-trough)/max(atr,1e-9)
        weakness=0; reasons=[]
        if not aligned: weakness+=1; reasons.append("short-term structure melemah")
        else: reasons.append("short-term structure aligned")
        if opposite: weakness+=1; reasons.append("opposite candle")
        if momentum_weak: weakness+=1; reasons.append("momentum melemah")
        if giveback>=_vn_float(p.get("trail_max_giveback_atr"),1.0): weakness+=1; reasons.append("giveback signifikan")
        if giveback>=_vn_float(p.get("trail_deep_giveback_atr"),1.5): weakness+=1; reasons.append("deep giveback")
        regime=classify_regime(btc_candles or work,p); rem_tp=((tp-price)/risk if direction=="BUY" else (price-tp)/risk); tp_superior=rem_tp>=_vn_float(p.get("trail_tp_priority_r"),2.25); trail_pref=_vn_float(p.get("TRAIL_PREFERENCE_SCORE"),0.5)>=_vn_float(p.get("NO_TRAIL_PREFERENCE_SCORE"),0.5)
        trigger="INSUFFICIENT_WEAKNESS"
        proposed=None; checkpoint=None; action="HOLD"
        if profit_r>=max(_vn_float(p.get("trail_activation_r"),0.75),_vn_float(p.get("trail_min_profit_r"),0.80)) and weakness>=_vn_int(p.get("trail_weakness_score"),3):
            swings=vn_detect_swings(path,_vn_int(p.get("swing_left"),2),_vn_int(p.get("swing_right"),2)); buffer_=atr*_vn_float(p.get("trail_structure_buffer_atr"),0.20)
            if direction=="BUY":
                lows=[s["price"] for s in swings if s["type"]=="L"]; checkpoint=max(lows[-3:]) if lows else price-atr; proposed=min(checkpoint-buffer_,price-atr*_vn_float(p.get("trail_min_step_atr"),0.10))
            else:
                highs=[s["price"] for s in swings if s["type"]=="H"]; checkpoint=min(highs[-3:]) if highs else price+atr; proposed=max(checkpoint+buffer_,price+atr*_vn_float(p.get("trail_min_step_atr"),0.10))
            valid,_=validate_trailing_geometry(direction,current_sl,proposed,price,entry,tp)
            if valid and ((direction=="BUY" and proposed>current_sl) or (direction=="SELL" and proposed<current_sl)):
                locked=((proposed-entry)/risk if direction=="BUY" else (entry-proposed)/risk); old_locked=((current_sl-entry)/risk if direction=="BUY" else (entry-current_sl)/risk); gain=locked-old_locked; premature=_vn_clip(0.25+0.15*weakness+0.20*giveback+0.20*(0 if trail_pref else 1),0,1)
                if gain>=_vn_float(p.get("trail_protection_floor_r"),0.05):
                    action="TRAIL"; trigger="DEEP_GIVEBACK" if giveback>=_vn_float(p.get("trail_deep_giveback_atr"),1.5) else "MOMENTUM_WEAKNESS" if momentum_weak else "STRUCTURE_WEAKNESS"; reasons.append(f"trigger={trigger}"); reasons.append(f"protection gain={gain:.2f}R")
                    return {"action":action,"new_sl":proposed,"reason":reasons,"weakness_score":weakness,"engine":"vnext","profit_r":profit_r,"trigger":trigger,"structure_checkpoint":checkpoint,"old_sl":current_sl,"proposed_sl":proposed,"protection_gain_r":gain,"locked_r":locked,"risk_premature_stop":premature,"tp_still_superior":tp_superior,"trail_statistically_preferable":trail_pref,"regime":regime,"giveback_atr":giveback,"data_quality":quality}
        return {"action":"NO_TRAIL" if profit_r>=max(_vn_float(p.get("trail_activation_r"),0.75),_vn_float(p.get("trail_min_profit_r"),0.80)) else "HOLD","new_sl":None,"reason":reasons+[trigger],"weakness_score":weakness,"engine":"vnext","profit_r":profit_r,"trigger":trigger,"structure_checkpoint":checkpoint,"old_sl":current_sl,"proposed_sl":proposed,"protection_gain_r":0.0,"locked_r":((current_sl-entry)/risk if direction=="BUY" else (entry-current_sl)/risk),"risk_premature_stop":0.25,"tp_still_superior":tp_superior,"trail_statistically_preferable":trail_pref,"regime":regime,"giveback_atr":giveback,"data_quality":quality}


# Public API rebinding: main.py now receives the vNext implementation.
Strategy = StrategyVNext


def new_default_strategy() -> Strategy:
    return Strategy()


__all__ = [
    "STRATEGY_NAME", "CONFIDENCE_WEIGHTS", "DEFAULT_PARAMS", "Setup", "Strategy",
    "StrategyVNext", "new_default_strategy", "validate_candles", "validate_geometry",
    "classify_session", "classify_regime", "classify_volatility_regime", "true_range", "atr_series", "ema",
    "linreg_slope", "pct_returns", "correlation", "swing_points", "equal_levels",
    "detect_liquidity_sweep", "detect_displacement", "detect_fvg", "validate_trailing_geometry", "SIGNAL_STATUSES",
    "MONITOR_ACTIONS",
]



# =============================================================================
# COMBINED-BRAIN COMPATIBILITY ANALYSIS SURFACES
# =============================================================================
# The functions below are deterministic portions of strategy_logic.py that are
# useful to callers outside the vNext object API. They do not perform network I/O.

def _safe_float(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _clip(x, lo, hi):
    return max(lo, min(hi, _safe_float(x, lo)))


def _now():
    return time.time()

class LegacyMarketState:
    __slots__ = (
        "symbol", "macro_bias", "htf_bias", "m15_bias", "regime",
        "trend_strength", "volatility", "structure_strength",
        "liquidity_state", "range_position", "data_quality",
        "relative_volume", "timestamp"
    )
    def __init__(self, symbol, macro_bias, htf_bias, m15_bias, regime, trend_strength,
                 volatility, structure_strength, liquidity_state, range_position,
                 data_quality, relative_volume, timestamp):
        self.symbol=symbol; self.macro_bias=macro_bias; self.htf_bias=htf_bias
        self.m15_bias=m15_bias; self.regime=regime; self.trend_strength=float(trend_strength)
        self.volatility=float(volatility); self.structure_strength=float(structure_strength)
        self.liquidity_state=liquidity_state; self.range_position=float(range_position)
        self.data_quality=float(data_quality); self.relative_volume=float(relative_volume)
        self.timestamp=float(timestamp)
    def to_dict(self):
        return {k:getattr(self,k) for k in self.__slots__}

class LegacyCandidate:
    __slots__ = (
        "direction", "entry", "sl", "tp", "rr", "entry_label", "confidence",
        "setup_quality", "location_score", "trend_strength", "structure_strength",
        "liquidity_score", "htf_alignment", "macro_alignment", "poi_reacted",
        "trigger_confirmed", "reasons", "invalidations"
    )
    def __init__(self, direction, entry, sl, tp, rr, entry_label, confidence,
                 setup_quality, location_score, trend_strength, structure_strength,
                 liquidity_score, htf_alignment, macro_alignment, poi_reacted,
                 trigger_confirmed, reasons=None, invalidations=None):
        self.direction=str(direction); self.entry=float(entry); self.sl=float(sl); self.tp=float(tp); self.rr=float(rr)
        self.entry_label=str(entry_label); self.confidence=float(confidence); self.setup_quality=float(setup_quality)
        self.location_score=float(location_score); self.trend_strength=float(trend_strength)
        self.structure_strength=float(structure_strength); self.liquidity_score=float(liquidity_score)
        self.htf_alignment=float(htf_alignment); self.macro_alignment=float(macro_alignment)
        self.poi_reacted=bool(poi_reacted); self.trigger_confirmed=bool(trigger_confirmed)
        self.reasons=list(reasons or []); self.invalidations=list(invalidations or [])
    def to_dict(self):
        return {k:getattr(self,k) for k in self.__slots__}

def _logic_rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.astype(float).diff()
    gain = d.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    out = pd.Series(50.0, index=s.index, dtype=float)
    valid = gain.notna() & loss.notna()
    both_zero = valid & (gain <= 1e-12) & (loss <= 1e-12)
    gain_only = valid & (loss <= 1e-12) & (gain > 1e-12)
    loss_only = valid & (gain <= 1e-12) & (loss > 1e-12)
    normal = valid & (gain > 1e-12) & (loss > 1e-12)
    out.loc[both_zero] = 50.0
    out.loc[gain_only] = 100.0
    out.loc[loss_only] = 0.0
    rs = gain.loc[normal] / loss.loc[normal]
    out.loc[normal] = 100.0 - 100.0 / (1.0 + rs)
    return out

def _logic_atr_fn(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _logic_closed_candles(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    idx = out.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    boundary = pd.Timestamp.now(tz="UTC").floor(f"{int(interval_minutes)}min")
    if idx[-1] < boundary:
        return out
    return out.loc[idx < boundary].copy()

def _logic_build_df(df: pd.DataFrame, interval_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 60:
        return None
    out = df.copy()
    if interval_minutes:
        out = _logic_closed_candles(out, interval_minutes)
    if out is None or len(out) < 60:
        return None
    for c in ("open", "high", "low", "close", "volume"):
        if c not in out.columns:
            return None
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["ema9"] = ema(out["close"], 9)
    out["ema21"] = ema(out["close"], 21)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200) if len(out) >= 200 else ema(out["close"], 50)
    out["_logic_rsi"] = _logic_rsi(out["close"])
    out["atr"] = _logic_atr_fn(out)
    out["vol_sma"] = out["volume"].rolling(20).mean()
    out = out.dropna(subset=["ema9", "ema21", "ema50", "ema200", "atr", "vol_sma"])
    if len(out) < 30:
        return None
    out["_logic_rsi"] = out["_logic_rsi"].fillna(50.0).clip(0.0, 100.0)
    return out

def _logic_swing_pts(df: pd.DataFrame, lb: int = 5):
    if df is None or len(df) < max(2 * lb + 1, 5):
        return [], []
    sh, sl = [], []
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    for i in range(lb, len(df) - lb):
        if high[i] >= np.max(high[i - lb:i + lb + 1]):
            sh.append(i)
        if low[i] <= np.min(low[i - lb:i + lb + 1]):
            sl.append(i)
    return sh, sl

def _logic_market_structure(df: pd.DataFrame, sh: list, sl: list) -> str:
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

def _logic_mkt_struct(df: pd.DataFrame, sh: list, sl: list) -> str:
    return _logic_market_structure(df, sh, sl)

def _logic_fib_position(price: float, swing_low: float, swing_high: float) -> float:
    rng = swing_high - swing_low
    if rng <= 0:
        return 0.5
    return _clip((price - swing_low) / rng, 0.0, 1.0)

def _logic_trend_strength(df: pd.DataFrame, sh: list, sl: list) -> float:
    """Directional trend strength based on confirmed swing progress and price/time slope.

    The baseline treats trend strength as directional energy: how far successive
    directional swing extremes travel per unit of time, normalized by ATR.
    """
    if df is None or len(df) < 30:
        return 0.0
    last = df.iloc[-1]
    atr = _safe_float(last.get("atr"), 0.0)
    if atr <= 0:
        return 0.0

    struct = _logic_market_structure(df, sh, sl)
    score = 35.0
    if struct in {"bullish", "bearish"}:
        score += 20.0
    else:
        score -= 8.0

    try:
        if struct == "bullish" and len(sh) >= 3:
            pts = [(int(i), _safe_float(df["high"].iloc[i])) for i in sh[-3:]]
        elif struct == "bearish" and len(sl) >= 3:
            pts = [(int(i), _safe_float(df["low"].iloc[i])) for i in sl[-3:]]
        else:
            pts = []
        if len(pts) >= 2:
            slopes = []
            for (i1, p1), (i2, p2) in zip(pts[:-1], pts[1:]):
                dt = max(1, i2 - i1)
                slopes.append(abs(p2 - p1) / dt / atr)
            slope = float(np.mean(slopes))
            score += _clip(slope * 900.0, 0.0, 28.0)

            # Acceleration in the directional swing sequence.
            if len(slopes) >= 2 and slopes[-1] > slopes[-2] * 1.10:
                score += 7.0
            elif len(slopes) >= 2 and slopes[-1] < slopes[-2] * 0.75:
                score -= 7.0
    except Exception:
        pass

    ema9 = _safe_float(last.get("ema9"), 0)
    ema21 = _safe_float(last.get("ema21"), 0)
    ema50 = _safe_float(last.get("ema50"), 0)
    if struct == "bullish" and ema9 > ema21 > ema50:
        score += 8
    elif struct == "bearish" and ema9 < ema21 < ema50:
        score += 8
    elif struct in {"bullish", "bearish"}:
        score -= 4

    rel_vol = _safe_float(last.get("volume"), 0.0) / max(_safe_float(last.get("vol_sma"), 1.0), 1e-12)
    score += _clip((rel_vol - 1.0) * 6.0, -6.0, 6.0)
    return _clip(score, 0.0, 100.0)

def _logic_macro_bias(df_btc_h1: Optional[pd.DataFrame]) -> str:
    btc = _logic_build_df(df_btc_h1, 60) if df_btc_h1 is not None else None
    if btc is None or len(btc) < 50:
        return "unknown"
    sh, sl = _logic_swing_pts(btc, 5)
    struct = _logic_market_structure(btc, sh, sl)
    last = btc.iloc[-1]
    if struct == "bullish" or last["ema9"] > last["ema21"] > last["ema50"]:
        return "bullish"
    if struct == "bearish" or last["ema9"] < last["ema21"] < last["ema50"]:
        return "bearish"
    return "ranging"

def _logic_detect_bos(df: pd.DataFrame, sh: list, sl: list) -> dict:
    out = {"bullish_bos": False, "bearish_bos": False, "level": None}
    if len(sh) < 1 or len(sl) < 1 or len(df) < 3:
        return out
    close = _safe_float(df["close"].iloc[-1])
    prev_close = _safe_float(df["close"].iloc[-2])
    hi = _safe_float(df["high"].iloc[sh[-1]])
    lo = _safe_float(df["low"].iloc[sl[-1]])
    out["bullish_bos"] = close > hi and prev_close <= hi
    out["bearish_bos"] = close < lo and prev_close >= lo
    out["level"] = hi if out["bullish_bos"] else lo if out["bearish_bos"] else None
    return out

def _logic_detect_choch(df: pd.DataFrame, sh: list, sl: list) -> dict:
    out = {"bullish_choch": False, "bearish_choch": False}
    if len(sh) < 2 or len(sl) < 2:
        return out
    struct = _logic_market_structure(df, sh, sl)
    close = _safe_float(df["close"].iloc[-1])
    last_hi = _safe_float(df["high"].iloc[sh[-1]])
    last_lo = _safe_float(df["low"].iloc[sl[-1]])
    if struct == "bearish" and close > last_hi:
        out["bullish_choch"] = True
    if struct == "bullish" and close < last_lo:
        out["bearish_choch"] = True
    return out

def _logic_detect_cisd(df: pd.DataFrame, lb: int = 8) -> dict:
    out = {"bullish_cisd": False, "bearish_cisd": False}
    if df is None or len(df) < lb + 1:
        return out
    sub = df.iloc[-lb:]
    o, c = sub["open"].to_numpy(float), sub["close"].to_numpy(float)
    if c[-1] > o[-1]:
        run = 0
        for j in range(len(c) - 2, -1, -1):
            if c[j] < o[j]: run += 1
            else: break
        if run >= 3:
            first = len(c) - 1 - run
            mid = (o[first] + c[first]) / 2.0
            out["bullish_cisd"] = c[-1] > mid
    elif c[-1] < o[-1]:
        run = 0
        for j in range(len(c) - 2, -1, -1):
            if c[j] > o[j]: run += 1
            else: break
        if run >= 3:
            first = len(c) - 1 - run
            mid = (o[first] + c[first]) / 2.0
            out["bearish_cisd"] = c[-1] < mid
    return out

def _logic_detect_inducement(df: pd.DataFrame, direction: str, lb: int = 40) -> dict:
    if df is None or len(df) < 15:
        return {"found":False,"swept":False,"level":None}
    sub = df.iloc[-min(lb, len(df)):].reset_index(drop=True)
    sh, sl = _logic_swing_pts(sub, 2)
    if direction == "bull" and sl:
        lvl = _safe_float(sub["low"].iloc[sl[-1]])
        aft = sub.iloc[sl[-1]+1:]
        return {"found":True,"swept":bool((aft["low"]<lvl).any()),"level":lvl}
    if direction == "bear" and sh:
        lvl = _safe_float(sub["high"].iloc[sh[-1]])
        aft = sub.iloc[sh[-1]+1:]
        return {"found":True,"swept":bool((aft["high"]>lvl).any()),"level":lvl}
    return {"found":False,"swept":False,"level":None}

def _logic_detect_order_blocks(df: pd.DataFrame, direction: str, lb: int = 80) -> list[dict]:
    if df is None or len(df) < 20:
        return []
    sub = df.iloc[-min(lb, len(df)):]
    base = len(df)-len(sub)
    atr = _safe_float(df["atr"].iloc[-1], 0.0)
    body_avg = _safe_float((sub["close"]-sub["open"]).abs().mean(), 1e-9)
    sh, sl = _logic_swing_pts(df, 5)
    swing_h = _safe_float(df["high"].iloc[sh[-1]], 0.0) if sh else None
    swing_l = _safe_float(df["low"].iloc[sl[-1]], 0.0) if sl else None
    zones=[]
    for i in range(1, len(sub)-3):
        c, nxt = sub.iloc[i], sub.iloc[i+1]
        bullish_pair = c["close"] < c["open"] and nxt["close"] > nxt["open"]
        bearish_pair = c["close"] > c["open"] and nxt["close"] < nxt["open"]
        if direction == "bull" and not bullish_pair: continue
        if direction == "bear" and not bearish_pair: continue
        impulse = abs(float(nxt["close"]-nxt["open"]))
        if impulse < body_avg*1.2: continue
        top = max(float(c["open"]), float(c["close"]))
        bot = min(float(c["open"]), float(c["close"]))
        idx = base+i
        post = df.iloc[idx+2:]
        if direction == "bull":
            fresh = not bool((post["close"] < bot).any())
        else:
            fresh = not bool((post["close"] > top).any())
        if not fresh: continue
        mid=(top+bot)/2
        fib=None
        if swing_l is not None and swing_h is not None and swing_h>swing_l:
            fib=_logic_fib_position(mid,swing_l,swing_h)
        q=50
        q += 10 if impulse >= body_avg*1.5 else 0
        q += 8 if impulse >= body_avg*2.5 else 0
        if atr>0: q += _clip(impulse/atr*8,0,12)
        if fib is not None:
            if direction=="bull" and fib<=0.618: q+=8
            if direction=="bear" and fib>=0.382: q+=8
        if idx>=len(df)-20: q+=5
        zones.append({"top":top,"bot":bot,"mid":mid,"idx":idx,"quality":_clip(q,0,100),"fib":fib})
    zones.sort(key=lambda z:(-z["quality"],-z["idx"]))
    return zones[:5]

def _legacy_entry_location(df: pd.DataFrame, direction: str, entry: float) -> dict:
    lb=min(24,len(df))
    sub=df.iloc[-lb:]
    hi=_safe_float(sub["high"].max(), entry)
    lo=_safe_float(sub["low"].min(), entry)
    rp=_logic_fib_position(entry,lo,hi)
    last_rsi=_safe_float(df["_logic_rsi"].iloc[-1],50)
    prev_rsi=_safe_float(df["_logic_rsi"].iloc[-2],last_rsi)
    score=70.0
    reasons=[]
    if direction=="bull":
        if rp<=0.55: score+=10; reasons.append("DISCOUNT_LOCATION")
        if rp>=0.82: score-=22; reasons.append("CHASE_HIGH")
        if last_rsi>=prev_rsi: score+=5
        elif last_rsi<48 and last_rsi<prev_rsi: score-=12; reasons.append("RSI_AGAINST_ENTRY")
    else:
        if rp>=0.45: score+=10; reasons.append("PREMIUM_LOCATION")
        if rp<=0.18: score-=22; reasons.append("CHASE_LOW")
        if last_rsi<=prev_rsi: score+=5
        elif last_rsi>52 and last_rsi>prev_rsi: score-=12; reasons.append("RSI_AGAINST_ENTRY")
    return {"location_score":_clip(score,0,100),"range_position":rp,"rsi_timing":"aligned" if score>=75 else "neutral" if score>=55 else "against","reasons":reasons}

def _logic_confirmation(df: pd.DataFrame, direction: str) -> dict:
    """Lower-timeframe confirmation based on liquidity + structure + displacement.

    A confirmation does not require the current candle to be the original swing
    candle. It checks the most recent completed structure and the latest closed
    candle, which is materially more stable for a scanner.
    """
    out = {
        "confirmed": False, "bos": {}, "choch": {}, "cisd": {},
        "sweep": {}, "displacement_atr": 0.0, "reason": "NO_CONFIRMATION"
    }
    if df is None or len(df) < 40:
        return out

    sh, sl = _logic_swing_pts(df, 3)
    bos = _logic_detect_bos(df, sh, sl)
    choch = _logic_detect_choch(df, sh, sl)
    cisd = _logic_detect_cisd(df, 8)
    sweep = _logic_detect_liquidity_sweep(df, sh, sl, direction)

    close = _safe_float(df["close"].iloc[-1])
    op = _safe_float(df["open"].iloc[-1])
    atr = _safe_float(df["atr"].iloc[-1], 0.0)
    displacement = abs(close - op) / max(atr, 1e-12) if atr > 0 else 0.0

    # Robust recent structure break: use swings that are not formed by the
    # final 3 candles, then require the latest close to break that level.
    recent_sh = [i for i in sh if i <= len(df) - 4]
    recent_sl = [i for i in sl if i <= len(df) - 4]
    bull_break = False
    bear_break = False
    if recent_sh:
        level = _safe_float(df["high"].iloc[recent_sh[-1]], 0.0)
        prev = _safe_float(df["close"].iloc[-2], 0.0)
        bull_break = close > level and prev <= level
    if recent_sl:
        level = _safe_float(df["low"].iloc[recent_sl[-1]], 0.0)
        prev = _safe_float(df["close"].iloc[-2], 0.0)
        bear_break = close < level and prev >= level

    bullish_event = bool(bos.get("bullish_bos") or choch.get("bullish_choch") or
                          cisd.get("bullish_cisd") or bull_break)
    bearish_event = bool(bos.get("bearish_bos") or choch.get("bearish_choch") or
                          cisd.get("bearish_cisd") or bear_break)

    # Combined baseline: a liquidity sweep is preferred, while a clean
    # structure break/displacement can also validate a continuation setup.
    directional_event = bullish_event if direction == "bull" else bearish_event
    sweep_bonus = sweep.get("type") != "none"
    body_ok = displacement >= 0.15
    confirmed = directional_event and body_ok

    if sweep_bonus and directional_event:
        confirmed = confirmed or displacement >= 0.10

    reasons = []
    if sweep_bonus:
        reasons.append(str(sweep.get("type")).upper())
    if bull_break and direction == "bull":
        reasons.append("RECENT_BULLISH_STRUCTURE_BREAK")
    if bear_break and direction == "bear":
        reasons.append("RECENT_BEARISH_STRUCTURE_BREAK")
    if direction == "bull" and choch.get("bullish_choch"):
        reasons.append("BULLISH_CHOCH")
    if direction == "bear" and choch.get("bearish_choch"):
        reasons.append("BEARISH_CHOCH")
    if direction == "bull" and cisd.get("bullish_cisd"):
        reasons.append("BULLISH_CISD")
    if direction == "bear" and cisd.get("bearish_cisd"):
        reasons.append("BEARISH_CISD")
    if confirmed:
        reasons.append("DISPLACEMENT_CONFIRMED")
    out.update({
        "confirmed": bool(confirmed),
        "bos": bos, "choch": choch, "cisd": cisd, "sweep": sweep,
        "displacement_atr": round(displacement, 4),
        "reason": reasons or ["NO_CONFIRMATION"]
    })
    return out

def _logic_frame_bias(df: Optional[pd.DataFrame], interval_minutes: Optional[int]) -> tuple[str,float,float,float]:
    d=_logic_build_df(df, interval_minutes)
    if d is None:
        return "unknown",0.0,0.0,0.0
    sh,sl=_logic_swing_pts(d,5)
    struct=_logic_market_structure(d,sh,sl)
    trend=_logic_trend_strength(d,sh,sl)
    last=d.iloc[-1]
    ema_bull=bool(last["ema9"]>last["ema21"]>last["ema50"])
    ema_bear=bool(last["ema9"]<last["ema21"]<last["ema50"])
    if struct=="bullish" and ema_bull: bias="bullish"
    elif struct=="bearish" and ema_bear: bias="bearish"
    elif struct=="bullish" or ema_bull: bias="bullish"
    elif struct=="bearish" or ema_bear: bias="bearish"
    else: bias="ranging"
    rel_vol=_safe_float(last["volume"],0)/max(_safe_float(last["vol_sma"],1),1e-9)
    return bias,trend,_clip(rel_vol/2.0,0,2),_clip(_safe_float(last["atr"],0)/max(_safe_float(last["close"],1),1e-9)*100,0,25)

def _logic_market_state(symbol, h1, m15, d1, btc_h1) -> tuple[MarketState, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict]:
    h1d=_logic_build_df(h1,60); m15d=_logic_build_df(m15,15); d1d=_logic_build_df(d1,1440) if d1 is not None else None
    hb, ht, hvol, hv = _logic_frame_bias(h1,60)
    mb, mt, mvol, mv = _logic_frame_bias(m15,15)
    db, dt, dvol, dv = _logic_frame_bias(d1,1440) if d1 is not None else ("unknown",0,0,0)
    macro=_logic_macro_bias(btc_h1)
    trend=_clip((ht*0.55+mt*0.35+dt*0.10) if d1d is not None else (ht*0.60+mt*0.40),0,100)
    sh,sl=(_logic_swing_pts(m15d,5) if m15d is not None else ([],[]))
    struct_strength=_clip((60 if mb in {"bullish","bearish"} else 35)+(trend-50)*0.35,0,100)
    last=m15d.iloc[-1] if m15d is not None else (h1d.iloc[-1] if h1d is not None else None)
    rp=0.5
    if last is not None and m15d is not None:
        rp=_logic_fib_position(_safe_float(last["close"]),_safe_float(m15d["low"].tail(24).min()),_safe_float(m15d["high"].tail(24).max()))
    if hb=="bullish" and trend>=72: regime="BULL_TREND_STRONG"
    elif hb=="bearish" and trend>=72: regime="BEAR_TREND_STRONG"
    elif hb=="bullish": regime="BULL_TREND_WEAK"
    elif hb=="bearish": regime="BEAR_TREND_WEAK"
    elif mb in {"bullish","bearish"}: regime="TRANSITION"
    else: regime="RANGING"
    liq="UNKNOWN"
    if m15d is not None:
        bull_sw=_logic_detect_liquidity_sweep(m15d,sh,sl,"bull")
        bear_sw=_logic_detect_liquidity_sweep(m15d,sh,sl,"bear")
        if bull_sw["type"]!="none": liq="SELLSIDE_SWEPT"
        elif bear_sw["type"]!="none": liq="BUYSIDE_SWEPT"
        else: liq="UNTAKEN"
    quality=1.0 if h1d is not None and m15d is not None else 0.7 if h1d is not None else 0.0
    if d1d is None: quality*=0.95
    state=MarketState(symbol,macro,hb,mb,regime,trend,mv+hv,struct_strength,liq,rp,quality,mvol,_now())
    extra={"d1_bias":db,"d1_strength":dt,"h1_strength":ht,"m15_strength":mt,"h1_df":h1d,"m15_df":m15d,"d1_df":d1d}
    return state,h1d,m15d,d1d,extra

def _logic_score_direction(df_h1, df_m15, df_d1=None):
    state,_,m15d,_,extra=_logic_market_state(None,df_h1,df_m15,df_d1,None)
    bull=0.0; bear=0.0
    if state.htf_bias=="bullish": bull+=35
    if state.htf_bias=="bearish": bear+=35
    if state.m15_bias=="bullish": bull+=25
    if state.m15_bias=="bearish": bear+=25
    if state.macro_bias=="bullish": bull+=10
    if state.macro_bias=="bearish": bear+=10
    if state.regime.startswith("BULL"): bull+=10
    if state.regime.startswith("BEAR"): bear+=10
    if m15d is not None:
        sh,sl=_logic_swing_pts(m15d,5)
        bs=_logic_detect_bos(m15d,sh,sl)
        ch=_logic_detect_choch(m15d,sh,sl)
        if bs["bullish_bos"] or ch["bullish_choch"]: bull+=10
        if bs["bearish_bos"] or ch["bearish_choch"]: bear+=10
    total=max(bull+bear,1.0)
    direction="bull" if bull>bear else "bear" if bear>bull else "neutral"
    conf=max(bull,bear)/total*100.0
    return {"direction":direction,"confidence":_clip(conf,0,100),"bull_score":bull,"bear_score":bear,
            "htf_bias":state.htf_bias,"macro_bias":state.macro_bias,"m15_struct":state.m15_bias,
            "trigger_count":int(round(max(bull,bear)/10.0)),"direction_edge":abs(bull-bear),
            "fib_r":state.range_position,"m15_relative_volume":state.relative_volume,
            "regime":state.regime,"trend_strength":state.trend_strength,"state":state.to_dict()}

def _logic_candidate_for_direction(state: MarketState, m15d: pd.DataFrame, h1d: pd.DataFrame, direction: str) -> Optional[Candidate]:
    """Build a staged candidate from the combined.txt baseline.

    Important design change in V136:
    - POI discovery and entry eligibility are separate stages.
    - A valid directional POI may be *approaching* and does not need to be
      touched on the current candle to become a candidate.
    - Execution still requires the actual reaction/confirmation at the POI.
    This lets the brain learn opportunity frequency without fabricating entries.
    """
    if h1d is None or m15d is None or len(h1d) < 60 or len(m15d) < 60:
        return None

    p = _safe_float(m15d["close"].iloc[-1])
    atr = _safe_float(m15d["atr"].iloc[-1], 0.0)
    h1_atr = _safe_float(h1d["atr"].iloc[-1], 0.0)
    if p <= 0 or atr <= 0 or h1_atr <= 0:
        return None

    # H1 is the primary POI timeframe, matching combined.txt.
    h1_obs = _logic_detect_order_blocks(h1d, direction, 100)
    h1_fvgs = _logic_detect_fvg(h1d, direction, 100)
    h1_zones = []
    for z in h1_obs:
        zz = dict(z); zz["kind"] = "H1_OB"; h1_zones.append(zz)
    for z in h1_fvgs:
        zz = dict(z); zz["quality"] = 62.0; zz["kind"] = "H1_FVG"; h1_zones.append(zz)

    # Detector fallback: derive a structural demand/supply zone from a recent
    # displacement when the strict OB/FVG detector found no zone. This is still
    # price-action geometry; it does not create a directional signal by itself.
    if not h1_zones:
        sh1, sl1 = _logic_swing_pts(h1d, 5)
        look = min(18, len(h1d) - 1)
        sub = h1d.iloc[-look:]
        body = (sub["close"] - sub["open"]).abs()
        avg_body = max(_safe_float(body.mean(), 0.0), 1e-12)
        for j in range(max(1, len(h1d)-look), len(h1d)-1):
            row = h1d.iloc[j]
            nxt = h1d.iloc[j+1]
            impulse = abs(_safe_float(nxt["close"] - nxt["open"]))
            if direction == "bull" and row["close"] < row["open"] and nxt["close"] > nxt["open"] and impulse >= avg_body*1.15:
                h1_zones.append({"top":float(max(row["open"],row["close"])),
                                 "bot":float(min(row["open"],row["close"])),
                                 "mid":float((row["open"]+row["close"])/2),
                                 "idx":j,"quality":58.0,"kind":"H1_STRUCTURAL_DEMAND"})
            elif direction == "bear" and row["close"] > row["open"] and nxt["close"] < nxt["open"] and impulse >= avg_body*1.15:
                h1_zones.append({"top":float(max(row["open"],row["close"])),
                                 "bot":float(min(row["open"],row["close"])),
                                 "mid":float((row["open"]+row["close"])/2),
                                 "idx":j,"quality":58.0,"kind":"H1_STRUCTURAL_SUPPLY"})

    if not h1_zones:
        return None

    # Choose the best zone by quality + freshness + distance. Distance is now a
    # score, not a hard rejection. A setup can be watched while price approaches.
    h1_zones.sort(key=lambda z: (-_safe_float(z.get("quality"),50), -_safe_float(z.get("idx"),0)))
    max_watch_distance = max(8.0 * atr, 3.0 * h1_atr)
    ranked = []
    for z in h1_zones:
        top = _safe_float(z.get("top"), 0); bot = _safe_float(z.get("bot"), 0)
        if top <= 0 or bot <= 0 or top < bot:
            continue
        dist = 0.0 if bot <= p <= top else min(abs(p-bot), abs(p-top))
        if dist <= max_watch_distance:
            ranked.append((dist, z))
    if not ranked:
        # If all zones are stale/far, retain the freshest valid zone only as a
        # monitoring candidate when it remains within a broad structural range.
        ranked = [(min(abs(p-_safe_float(z.get("bot"),p)), abs(p-_safe_float(z.get("top"),p))), z)
                  for z in h1_zones[:3]
                  if _safe_float(z.get("top"),0)>0 and _safe_float(z.get("bot"),0)>0]

    if not ranked:
        return None
    ranked.sort(key=lambda x: (x[0] / max(h1_atr,atr,1e-12), -_safe_float(x[1].get("quality"),50), -_safe_float(x[1].get("idx"),0)))
    distance, poi = ranked[0]
    poi_top = _safe_float(poi.get("top"),0); poi_bot = _safe_float(poi.get("bot"),0)
    in_poi = poi_bot <= p <= poi_top
    proximity = max(1.25 * atr, 0.55 * h1_atr)
    near_poi = in_poi or distance <= proximity

    # Confirmation is evaluated only as execution evidence. A distant/approaching
    # POI is a legitimate learning candidate, not an executable signal.
    confirm = _logic_confirmation(m15d, direction) if near_poi else {
        "confirmed":False,"bos":False,"choch":False,"cisd":False,"sweep":{"type":"none"},
        "displacement":0.0,"reason":["POI_APPROACHING","AWAIT_POI_REACTION"]
    }

    sh15, sl15 = _logic_swing_pts(m15d, 3)
    lower_obs = _logic_detect_order_blocks(m15d, direction, 80)
    lower_fvgs = _logic_detect_fvg(m15d, direction, 80)

    entry_zone = None
    if confirm["confirmed"]:
        for z in reversed(lower_fvgs):
            top = _safe_float(z.get("top"),0); bot = _safe_float(z.get("bot"),0)
            if top <= 0 or bot <= 0: continue
            if direction == "bull" and bot <= p + proximity and top >= p - proximity:
                entry_zone = dict(z); entry_zone["kind"]="M15_FVG"; break
            if direction == "bear" and bot <= p + proximity and top >= p - proximity:
                entry_zone = dict(z); entry_zone["kind"]="M15_FVG"; break
        if entry_zone is None and lower_obs:
            entry_zone = dict(lower_obs[0]); entry_zone["kind"]="M15_OB"

    if entry_zone is None:
        entry_zone = dict(poi)
        entry_zone["kind"] = poi.get("kind","H1_POI")

    entry = _safe_float(entry_zone.get("mid"), 0.0)
    if entry <= 0:
        return None

    # For an approaching POI the zone midpoint is a planning price, not a
    # permission to chase. Execution remains disabled until confirmation.
    last_hi = _safe_float(m15d["high"].iloc[sh15[-1]], p + atr) if sh15 else p + atr
    last_lo = _safe_float(m15d["low"].iloc[sl15[-1]], p - atr) if sl15 else p - atr

    if direction == "bull":
        candidates = [x for x in (last_lo, poi_bot) if 0 < x < entry]
        sl_price = min(candidates) if candidates else entry - 0.85 * atr
        sl_price -= 0.10 * atr
        targets = [_safe_float(h1d["high"].iloc[i],0) for i in _logic_swing_pts(h1d,5)[0]]
        targets += [_safe_float(m15d["high"].iloc[i],0) for i in sh15]
        targets = [v for v in targets if v > entry]
        target = min(targets, default=entry + 2.5*abs(entry-sl_price))
        target = max(target, entry + 2.0*abs(entry-sl_price))
    else:
        candidates = [x for x in (last_hi, poi_top) if x > entry]
        sl_price = max(candidates) if candidates else entry + 0.85 * atr
        sl_price += 0.10 * atr
        targets = [_safe_float(h1d["low"].iloc[i],0) for i in _logic_swing_pts(h1d,5)[1]]
        targets += [_safe_float(m15d["low"].iloc[i],0) for i in sl15]
        targets = [v for v in targets if 0 < v < entry]
        target = max(targets, default=entry - 2.5*abs(entry-sl_price))
        target = min(target, entry - 2.0*abs(entry-sl_price))

    risk = abs(entry-sl_price)
    if risk <= 0:
        return None
    rr = abs(target-entry)/risk
    if rr < MIN_RR:
        return None

    loc = _legacy_entry_location(m15d,direction,entry)
    htf_align = 1.0 if state.htf_bias == ("bullish" if direction=="bull" else "bearish") else 0.0
    macro_align = 1.0 if state.macro_bias == ("bullish" if direction=="bull" else "bearish") else 0.5 if state.macro_bias=="unknown" else 0.0
    poi_score = _safe_float(poi.get("quality"),58.0)
    sweep = confirm.get("sweep") or {"type":"none"}
    liq_score = 62.0 + (18.0 if sweep.get("type")!="none" else 0.0)
    distance_score = _clip(100.0 - (distance/max(h1_atr,atr,1e-12))*18.0, 10.0, 100.0)

    reasons = [
        "HTF_ALIGNED" if htf_align else "HTF_NEUTRAL",
        str(poi.get("kind","H1_POI")),
        "FRESH_POI" if bool(poi.get("fresh",True)) else "POI_AGED",
        "POI_AT_PRICE" if in_poi else "POI_NEAR_PRICE" if near_poi else "POI_APPROACHING",
    ]
    if macro_align >= 1: reasons.append("MACRO_ALIGNED")
    if state.trend_strength >= 65: reasons.append("TREND_STRENGTH")
    if sweep.get("type")!="none": reasons.append(str(sweep["type"]).upper())
    reasons.extend([r for r in confirm.get("reason",[]) if r not in reasons])
    if loc["location_score"]>=70: reasons.append("GOOD_LOCATION")
    if rr>=3: reasons.append("STRUCTURAL_RR_3R_PLUS")

    setup_quality = _clip(
        poi_score*0.26 + state.trend_strength*0.20 + state.structure_strength*0.15 +
        liq_score*0.12 + loc["location_score"]*0.10 + distance_score*0.10 +
        htf_align*4.0 + macro_align*3.0, 0,100)
    raw_conf = _clip(setup_quality + (10 if confirm["confirmed"] else -8) +
                     (5 if sweep.get("type")!="none" else 0) + (4 if rr>=3 else 0),0,100)

    return Candidate(
        direction.upper(),entry,sl_price,target,rr,str(entry_zone.get("kind","H1_FVG")),
        raw_conf,setup_quality,loc["location_score"],state.trend_strength,
        state.structure_strength,liq_score,htf_align,macro_align,True,
        confirm["confirmed"],reasons,
        ["HTF_POI_INVALIDATION","M15_STRUCTURE_FAILURE","LIQUIDITY_THESIS_FAILURE"]
    )

# Legacy detector adapters: current vNext implementations remain canonical for runtime.
def _logic_detect_fvg(candles, direction=None, lb=60):
    # Recreate the strategy_logic 3-candle / scan-window behavior without I/O.
    if candles is None or len(candles) < 3:
        return []
    data=list(candles)
    start=max(0, len(data)-max(3,int(lb or 60)))
    result=[]
    for i in range(start+2, len(data)):
        a,b,c=data[i-2],data[i-1],data[i]
        ah,al,cl,ch=_safe_float(a.get("h",a.get("high"))),_safe_float(a.get("l",a.get("low"))),_safe_float(c.get("l",c.get("low"))),_safe_float(c.get("h",c.get("high")))
        if ah < cl:
            result.append({"type":"BULLISH_FVG","top":cl,"bottom":ah,"mid":(cl+ah)/2,"idx":i,"quality":60.0,"fresh":True})
        elif al > ch:
            result.append({"type":"BEARISH_FVG","top":al,"bottom":ch,"mid":(al+ch)/2,"idx":i,"quality":60.0,"fresh":True})
    return [z for z in result if direction is None or z["type"]==("BULLISH_FVG" if str(direction).lower() in {"buy","bull"} else "BEARISH_FVG")]

def _logic_detect_liquidity_sweep(candles, *args):
    # Supports both strategy_logic(df, sh, sl, direction) and simplified (candles, lookback).
    if len(args)==1 and isinstance(args[0], (int,float)):
        lb=int(args[0]); data=list(candles);
        if len(data) < 5: return None
        win=data[-max(5, min(len(data)-1, lb)):-1]; last=data[-1]
        prior_high=max(_safe_float(x.get("h",x.get("high"))) for x in win); prior_low=min(_safe_float(x.get("l",x.get("low"))) for x in win)
        if _safe_float(last.get("h",last.get("high")))>prior_high and _safe_float(last.get("c",last.get("close")))<prior_high: return {"type":"BEARISH_SWEEP","level":prior_high}
        if _safe_float(last.get("l",last.get("low")))<prior_low and _safe_float(last.get("c",last.get("close")))>prior_low: return {"type":"BULLISH_SWEEP","level":prior_low}
        return None
    if len(args) >= 3:
        sh, sl, direction = args[:3]
        try:
            d=str(direction).lower(); highs=[_safe_float(candles.iloc[i]["high"]) for i in sh] if hasattr(candles,'iloc') else [_safe_float(candles[i].get("h")) for i in sh]
            lows=[_safe_float(candles.iloc[i]["low"]) for i in sl] if hasattr(candles,'iloc') else [_safe_float(candles[i].get("l")) for i in sl]
            last=candles.iloc[-1] if hasattr(candles,'iloc') else candles[-1]
            h=_safe_float(last.get("high",last.get("h"))); l=_safe_float(last.get("low",last.get("l"))); c=_safe_float(last.get("close",last.get("c")))
            if d in {"buy","bull"} and lows and l<lows[-1] and c>lows[-1]: return {"type":"BULLISH_SWEEP","level":lows[-1]}
            if d in {"sell","bear"} and highs and h>highs[-1] and c<highs[-1]: return {"type":"BEARISH_SWEEP","level":highs[-1]}
        except Exception:
            return None
    return None


def _legacy_entry_location(df, direction, entry):
    try:
        closes=df["close"].astype(float)
        atr=float(df["atr"].iloc[-1])
        px=float(entry)
        recent=closes.iloc[-30:]
        lo=float(recent.min()); hi=float(recent.max())
        pos=0.5 if hi<=lo else (px-lo)/(hi-lo)
        good=(0.15<=pos<=0.65) if str(direction).lower() in {"buy","bull"} else (0.35<=pos<=0.85)
        dist=abs(float(df["close"].iloc[-1])-px)/max(atr,1e-9)
        return {"location_score":float(_clip(100-35*dist-20*abs(pos-0.5),0,100)),"range_position":pos,"distance_atr":dist,"good":good}
    except Exception:
        return {"location_score":50.0,"range_position":0.5,"distance_atr":0.0,"good":False}

# Public legacy dataframe helpers.
def rsi(s, n=14):
    return _logic_rsi(s,n)

def atr_fn(df, n=14):
    return _logic_atr_fn(df,n)

def build_df(df, interval_minutes=None):
    if pd is None: return None
    return _logic_build_df(df,interval_minutes)

def fib_position(price, swing_low, swing_high):
    return _logic_fib_position(price,swing_low,swing_high)

def swing_pts(df, lb=5):
    return _logic_swing_pts(df,lb)

def mkt_struct(df, sh, sl):
    return _logic_mkt_struct(df,sh,sl)

def detect_bos(df, sh, sl):
    return _logic_detect_bos(df,sh,sl)

def detect_choch(df, sh, sl):
    return _logic_detect_choch(df,sh,sl)

def detect_cisd(df, lb=8):
    return _logic_detect_cisd(df,lb)

def detect_inducement(df, direction, lb=40):
    return _logic_detect_inducement(df,direction,lb)

def detect_order_blocks(df, direction, lb=80):
    return _logic_detect_order_blocks(df,direction,lb)

def score_direction(df_h1, df_m15, df_d1=None):
    return _logic_score_direction(df_h1,df_m15,df_d1)

def _to_candle_records(data):
    if data is None:
        return []
    if isinstance(data, (list, tuple)):
        return [dict(x) for x in data if isinstance(x, dict)]
    if pd is not None and isinstance(data, pd.DataFrame):
        out = []
        for n, (idx, row) in enumerate(data.iterrows(), start=1):
            def gv(*names, default=0.0):
                for name in names:
                    if name in row.index:
                        try: return float(row[name])
                        except Exception: return default
                return default
            if hasattr(idx, "timestamp"):
                ts = float(idx.timestamp() * 1000.0)
            else:
                ts = float(n * 900000)
            out.append({"t": ts, "o": gv("open", "o"), "h": gv("high", "h"),
                        "l": gv("low", "l"), "c": gv("close", "c"), "v": gv("volume", "v")})
        return out
    return []

def full_analyze(df_h1, df_m15, df_d1=None, symbol=None, df_btc_h1=None, trade_history=None, market_data_source="main", **kwargs):
    """Compatibility facade for the original strategy_logic.full_analyze API.

    Accepts both pandas DataFrames and the newer list-of-candle records. No exchange/API calls.
    Learning/trade history is observational context only; LearnEngine owns mutations.
    """
    try:
        engine = kwargs.pop("strategy_engine", None) or Strategy()
        candles = _to_candle_records(df_m15)
        btc = _to_candle_records(df_btc_h1 if df_btc_h1 is not None else df_h1)
        setup, diag = engine.analyze_with_diagnostics(
            str(symbol or "UNKNOWN"), candles, btc_candles=btc,
            market_context=kwargs.get("market_context"),
            enforce_threshold=kwargs.get("enforce_threshold", False),
        )
        if setup is None:
            return {
                "symbol": symbol, "decision": "WAIT", "candidate": False, "is_candidate": False,
                "execution_eligible": False, "no_signal": True, "analysis_stage": "DIAGNOSTIC_NO_ENTRY",
                "rejected_reason": diag.get("status", "NO_SETUP"), "confidence": 0.0,
                "confidence_threshold": engine.get_active_threshold(), "market_data_source": market_data_source,
                "brain_version": FINAL_BRAIN_VERSION, "diagnostics": diag,
            }
        rr = float(setup.reference_levels.get("rr", 0.0) or 0.0)
        eligible = bool(setup.threshold_passed)
        return {
            "symbol": symbol, "decision": setup.direction, "candidate": True, "is_candidate": True,
            "execution_eligible": eligible, "no_signal": not eligible,
            "analysis_stage": "READY" if eligible else "WAIT_ENTRY",
            "rejected_reason": None if eligible else "BELOW_ACTIVE_THRESHOLD",
            "eligibility_reason": "BRAIN_READY" if eligible else "BELOW_ACTIVE_THRESHOLD",
            "confidence": round(setup.confidence, 2), "confidence_threshold": engine.get_active_threshold(),
            "entry": setup.entry, "sl": setup.sl, "initial_sl": setup.sl, "tp": setup.tp, "rr": rr,
            "entry_label": setup.setup_type, "regime": setup.regime, "session": setup.session,
            "atr": setup.atr, "reason": setup.reason, "reasons": setup.reason,
            "components": setup.components, "strategy_version": setup.strategy_version,
            "brain_version": FINAL_BRAIN_VERSION, "diagnostics": diag,
            "execution_contract": "V136",
            "execution": {"symbol": str(symbol or "").upper(), "side": setup.direction,
                           "entry": setup.entry, "sl": setup.sl, "tp": setup.tp, "rr": rr, "type": "LIMIT"},
        }
    except Exception as exc:
        logger.exception("[BRAIN] full_analyze failed")
        return {"symbol": symbol, "decision": "WAIT", "candidate": False, "is_candidate": False,
                "execution_eligible": False, "no_signal": True, "analysis_stage": "ERROR",
                "rejected_reason": "BRAIN_ERROR", "error": str(exc)[:240],
                "market_data_source": market_data_source, "brain_version": FINAL_BRAIN_VERSION}


def manage_position(state, df_m15, df_h1=None, df_d1=None, symbol=None, **kwargs):
    engine = kwargs.pop("strategy_engine", None) or Strategy()
    btc = _to_candle_records(kwargs.pop("df_btc_h1", None))
    candles = _to_candle_records(df_m15)
    return engine.monitor_position(state, candles, btc_candles=btc, market_context=kwargs.get("market_context"))


def get_active_confidence_threshold(strategy_engine=None):
    return float((strategy_engine or Strategy()).get_active_threshold())


def set_manual_confidence_threshold(value, strategy_engine=None):
    engine=strategy_engine or Strategy()
    v=max(55.0,min(82.0,float(value)))
    engine.apply_update({"ACTIVE_THRESHOLD":v},"manual compatibility threshold",{"source":"compatibility_api"})
    return v

def suggest_confidence_threshold(strategy_engine=None):
    return get_active_confidence_threshold(strategy_engine)

def get_learning_schema():
    return FULL_LEARNING_SCHEMA

def get_cognitive_status():
    return {"strategy_version":Strategy().version,"threshold":Strategy().get_active_threshold(),"engine":"strategy_only","learning_owner":"learn.py"}

def get_full_cognitive_status():
    return get_cognitive_status()

def get_adaptive_status():
    return get_cognitive_status()


__all__ = [
    "STRATEGY_NAME", "FINAL_BRAIN_VERSION", "BRAIN_INTERFACE_VERSION", "FULL_LEARNING_SCHEMA",
    "MACHINE_LEARNING_SCHEMA", "BRAIN_CHECKPOINT_SCHEMA", "MIN_RR", "MAX_RR",
    "TRAIL_R_LADDER", "STRUCT_TRAIL_LB", "STRUCT_TRAIL_BUF_PCT", "STRUCT_TRAIL_LOOKBACK",
    "FIB_EXT_1", "FIB_EXT_2", "EXECUTION_CONFIDENCE_FLOOR", "EXECUTION_MIN_RR_FLOOR", "CONFIDENCE_WEIGHTS", "DEFAULT_PARAMS", "Setup", "Strategy",
    "StrategyVNext", "new_default_strategy", "validate_candles", "validate_geometry",
    "classify_session", "classify_regime", "classify_volatility_regime", "true_range", "atr_series",
    "ema", "rsi", "atr_fn", "build_df", "linreg_slope", "pct_returns", "correlation",
    "swing_points", "swing_pts", "mkt_struct", "fib_position", "detect_bos", "detect_choch",
    "detect_cisd", "detect_inducement", "detect_liquidity_sweep", "detect_displacement",
    "detect_fvg", "detect_order_blocks", "full_analyze", "manage_position",
    "get_active_confidence_threshold", "set_manual_confidence_threshold", "suggest_confidence_threshold",
    "get_learning_schema", "get_cognitive_status", "get_full_cognitive_status", "get_adaptive_status",
    "validate_trailing_geometry", "SIGNAL_STATUSES", "MONITOR_ACTIONS",
]


# Public detector adapters supporting both the newer list API and the original dataframe API.
def detect_liquidity_sweep(candles, *args):
    return _logic_detect_liquidity_sweep(candles, *args)

def detect_fvg(candles, *args):
    if not args:
        out = _logic_detect_fvg(candles, None, 60)
        return out[-1] if out else None
    direction = args[0] if args else None
    lb = args[1] if len(args) > 1 else 60
    return _logic_detect_fvg(candles, direction, lb)
