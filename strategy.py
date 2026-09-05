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

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy should always be available
    np = None


STRATEGY_NAME = "adaptive-smc-ict"

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
    "ACTIVE_THRESHOLD": 0.0,       # % — dimulai rendah agar learn.py punya data (§10)
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
        return max(0.0,min(100.0,_vn_float(self.params.get("ACTIVE_THRESHOLD"),0.0)))

    def export_state(self)->Dict[str,Any]:
        return {"schema_version":STRATEGY_SCHEMA_VERSION,"strategy_name":STRATEGY_NAME,"version":self.version,"params":dict(self.params),"version_history":list(self.version_history)}

    def load_state(self,state:Dict[str,Any])->None:
        if not isinstance(state,dict): return
        if isinstance(state.get("params"),dict):
            self.params.update(state["params"])
            if isinstance(state["params"].get("CONFIDENCE_WEIGHTS"),dict): self.params["CONFIDENCE_WEIGHTS"]=dict(state["params"]["CONFIDENCE_WEIGHTS"])
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
        threshold=self.get_active_threshold(); passed=score["final"]>=threshold
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
