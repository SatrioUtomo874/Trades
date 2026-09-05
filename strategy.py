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
    """Deterministic, data-only strategy engine.

    The public surface intentionally remains compatible with main.py:
    get_active_threshold(), apply_update(), rollback(), export_state(),
    load_state(), analyze(), monitor_position().

    vNext adds diagnostics for data quality, context, freshness, entry/TP/SL
    viability and trail decisions without making any network/API calls.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.version = "2.00"
        self.params: Dict[str, Any] = dict(DEFAULT_PARAMS)
        self.params.update({
            "stale_bars": 8,
            "max_entry_distance_atr": 4.0,
            "min_fill_viability": 0.20,
            "min_tp_viability": 0.20,
            "min_sl_viability": 0.20,
            "trail_activation_r": 0.60,
            "trail_min_step_atr": 0.15,
            "trail_min_profit_r": 0.30,
            "freshness_decay_bars": 12,
        })
        if params:
            self.params.update(params)
        self.version_history: List[Dict[str, Any]] = [{
            "version": self.version, "timestamp": time.time(), "reason": "INITIAL_VNEXT",
            "old_params": None, "new_params": dict(self.params), "evidence": None,
        }]

    def get_active_threshold(self) -> float:
        return float(self.params.get("ACTIVE_THRESHOLD", 0.0))

    def apply_update(self, new_params: Dict[str, Any], reason: str, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        old = dict(self.params)
        self.params.update(new_params)
        major, minor = self.version.split(".")
        self.version = f"{major}.{int(minor) + 1:02d}"
        rec = {"version": self.version, "timestamp": time.time(), "reason": reason,
               "old_params": old, "new_params": dict(self.params), "evidence": evidence}
        self.version_history.append(rec)
        return rec

    def rollback(self) -> Optional[Dict[str, Any]]:
        if len(self.version_history) < 2:
            return None
        self.version_history.pop()
        prev = self.version_history[-1]
        self.params = dict(prev["new_params"])
        self.version = prev["version"]
        return prev

    def export_state(self) -> Dict[str, Any]:
        return {"version": self.version, "params": dict(self.params), "version_history": list(self.version_history)}

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.version = state.get("version", self.version)
        self.params.update(state.get("params", {}))
        hist = state.get("version_history")
        if isinstance(hist, list) and hist:
            self.version_history = list(hist)

    @staticmethod
    def _data_quality(candles: Sequence[Dict[str, float]], interval_ms: float = 900_000.0) -> Dict[str, Any]:
        if not candles:
            return {"valid": False, "reason": "NO_CANDLES", "stale_bars": None, "gaps": 0, "duplicates": 0}
        ts = [float(c.get("t", 0)) for c in candles]
        gaps = sum(1 for a,b in zip(ts, ts[1:]) if b - a > interval_ms * 1.5)
        duplicates = sum(1 for a,b in zip(ts, ts[1:]) if b == a)
        stale_bars = None
        if len(ts) >= 2:
            median_step = sorted(b-a for a,b in zip(ts, ts[1:]))[len(ts)//2-1]
            if median_step > 0:
                stale_bars = max(0, int(round((ts[-1] - ts[-1]) / median_step)))
        ok, reason = validate_candles(candles, 1)
        return {"valid": bool(ok), "reason": reason if not ok else "OK", "gaps": gaps,
                "duplicates": duplicates, "n": len(candles), "last_ts": ts[-1] if ts else 0}

    @staticmethod
    def _freshness(last_ts: float, now_ms: Optional[float], bar_ms: float, decay_bars: float) -> Dict[str, Any]:
        now_ms = time.time() * 1000 if now_ms is None else float(now_ms)
        age_bars = max(0.0, (now_ms - last_ts) / max(1.0, bar_ms))
        score = max(0.0, min(100.0, 100.0 * math.exp(-age_bars / max(0.5, decay_bars))))
        return {"age_bars": round(age_bars, 3), "score": round(score, 2), "stale": age_bars > decay_bars}

    @staticmethod
    def _relative_strength(coin: Sequence[Dict[str, float]], btc: Sequence[Dict[str, float]], lb: int = 30) -> Dict[str, float]:
        cc = _closes(coin)[-lb:]
        bc = _closes(btc)[-lb:]
        if len(cc) < 5 or len(bc) < 5 or cc[0] == 0 or bc[0] == 0:
            return {"coin_return": 0.0, "btc_return": 0.0, "relative": 0.0}
        cr = cc[-1] / cc[0] - 1.0
        br = bc[-1] / bc[0] - 1.0
        return {"coin_return": cr, "btc_return": br, "relative": cr - br}

    def _score_confidence(self, values: Dict[str, float]) -> float:
        # Harmonic blend: a near-zero dimension cannot be hidden by another strong dimension.
        xs = [max(0.01, min(1.0, values.get(k, 0.0))) for k in
              ("setup_quality", "execution_viability", "context", "freshness", "expected_value")]
        harmonic = len(xs) / sum(1.0/x for x in xs)
        arithmetic = sum(xs) / len(xs)
        return max(0.0, min(100.0, 100.0 * (0.65 * harmonic + 0.35 * arithmetic)))

    def _build_setup(self, symbol: str, work: Sequence[Dict[str, float]], btc_candles: Optional[Sequence[Dict[str, float]]]) -> Tuple[Optional[Setup], Dict[str, Any]]:
        p = self.params
        closes = _closes(work)
        atrs = atr_series(work, p["atr_period"])
        atr_now = atrs[-1] if atrs else 0.0
        if atr_now <= 0:
            return None, {"diagnosis": "INVALID_VOLATILITY", "reason": ["ATR <= 0"]}
        struct_window = work[-p["structure_lookback"]:]
        swings = swing_points(struct_window, p["swing_left"], p["swing_right"])
        highs = [s for s in swings if s[2] == "H"]
        lows = [s for s in swings if s[2] == "L"]
        slope, r2 = linreg_slope(closes[-p["trend_lookback"]:])
        trend_dir = "BUY" if slope > 0 else "SELL"
        last_close = closes[-1]
        bos = None
        if highs and last_close > highs[-1][1]: bos = "BOS_UP"
        elif lows and last_close < lows[-1][1]: bos = "BOS_DOWN"
        if not bos:
            return None, {"diagnosis": "NO_SETUP", "reason": ["no confirmed BOS"]}
        direction = "BUY" if bos == "BOS_UP" else "SELL"
        reasons = [f"confirmed {bos}"]

        levels = equal_levels(swings, atr_now, p["equal_level_tol_atr"])
        sweep = detect_liquidity_sweep(work, p["sweep_lookback"])
        disp = detect_displacement(work, atr_now, p["displacement_atr_mult"])
        fvg = detect_fvg(work)
        sweep_aligned = bool(sweep and ((direction == "BUY" and sweep["type"] == "BULLISH_SWEEP") or (direction == "SELL" and sweep["type"] == "BEARISH_SWEEP")))
        disp_aligned = bool(disp and disp["direction"] == direction)
        fvg_aligned = bool(fvg and ((direction == "BUY" and fvg["type"] == "BULLISH_FVG") or (direction == "SELL" and fvg["type"] == "BEARISH_FVG")))
        pool = levels["equal_highs"] if direction == "BUY" else levels["equal_lows"]

        fib = float(p["entry_retracement_fib"])
        buffer_ = atr_now * float(p["sl_atr_buffer"])
        if direction == "BUY":
            leg_low = lows[-1][1] if lows else last_close - 2*atr_now
            leg_high = last_close
            rng = max(leg_high-leg_low, atr_now*1e-6)
            entry = leg_high-rng*fib
            entry = max(leg_low+0.05*atr_now, min(entry, leg_high-max(float(p["entry_min_offset_atr"])*atr_now, 0.05*atr_now)))
            sl = min(leg_low, entry-0.5*atr_now)-buffer_
            candidates = [x for x in levels["equal_highs"] if x > entry]
            tp = min(candidates) if candidates else entry + 2.0*(entry-sl)
        else:
            leg_high = highs[-1][1] if highs else last_close + 2*atr_now
            leg_low = last_close
            rng = max(leg_high-leg_low, atr_now*1e-6)
            entry = leg_low+rng*fib
            entry = min(leg_high-0.05*atr_now, max(entry, leg_low+max(float(p["entry_min_offset_atr"])*atr_now, 0.05*atr_now)))
            sl = max(leg_high, entry+0.5*atr_now)+buffer_
            candidates = [x for x in levels["equal_lows"] if x < entry]
            tp = max(candidates) if candidates else entry - 2.0*(sl-entry)

        ok, geom_reason = validate_geometry(direction, entry, sl, tp, atr_val=atr_now)
        if not ok:
            return None, {"diagnosis": "INVALID_GEOMETRY", "reason": [geom_reason]}
        risk = abs(entry-sl); reward = abs(tp-entry); rr = reward/max(risk, 1e-9)
        if rr < float(p["min_rr"]):
            return None, {"diagnosis": "LOW_EXPECTED_VALUE", "reason": [f"RR {rr:.2f} < minimum {p['min_rr']}"]}

        entry_dist_atr = abs(last_close-entry)/max(atr_now, 1e-9)
        fill_viability = max(0.0, min(1.0, math.exp(-entry_dist_atr / max(0.5, float(p["max_entry_distance_atr"]))) ))
        if entry_dist_atr > float(p["max_entry_distance_atr"]):
            diagnosis = "TOO_FAR"
        elif entry_dist_atr < 0.10:
            diagnosis = "TOO_CLOSE"
        else:
            diagnosis = "VALID"

        # Freshness is based on the age of the structural trigger relative to latest confirmed bar.
        last_swing_idx = max([i for i,_,_ in swings], default=max(0, len(struct_window)-1))
        structure_age = max(0, len(struct_window)-1-last_swing_idx)
        freshness_score = max(0.0, 1.0 - structure_age/max(1.0, float(p["freshness_decay_bars"])))
        if structure_age > int(p["stale_bars"]): diagnosis = "STALE_SETUP"

        roc = (closes[-1]-closes[-p["momentum_lookback"]-1])/closes[-p["momentum_lookback"]-1] if len(closes)>p["momentum_lookback"] and closes[-p["momentum_lookback"]-1] else 0.0
        momentum_aligned = (direction == "BUY" and roc > 0) or (direction == "SELL" and roc < 0)
        atr_rank = sum(a <= atr_now for a in atrs[-p["vol_regime_lookback"]:]) / max(1, len(atrs[-p["vol_regime_lookback"]:]))
        volatility_fit = max(0.0, 1.0-abs(atr_rank-0.55)*1.7)

        btc = {"corr":0.0,"direction":None,"return":0.0,"relative_strength":0.0,"aligned":None}
        if btc_candles:
            lb=int(p["btc_corr_lookback"])
            corr=correlation(pct_returns(closes[-lb:]), pct_returns(_closes(btc_candles)[-lb:]))
            bslope,_=linreg_slope(_closes(btc_candles)[-p["trend_lookback"]:])
            btc["corr"]=corr; btc["direction"]="BUY" if bslope>0 else "SELL"
            rs=self._relative_strength(work, btc_candles, min(lb, 30)); btc.update(rs)
            btc["aligned"]=(btc["direction"]==direction and corr>=0) or (btc["direction"]!=direction and corr<0)
        btc_context = 0.5
        btc_conflict = False
        if btc["direction"]:
            btc_context = max(0.0, min(1.0, 0.5 + 0.4*(1 if btc["aligned"] else -1)*abs(float(btc["corr"]))))
            btc_conflict = bool(not btc["aligned"] and abs(float(btc["corr"])) >= 0.45)
            if btc_conflict: diagnosis = "BTC_CONFLICT"

        regime = classify_regime(btc_candles if btc_candles else work, p)
        regime_fit = 1.0 if ((regime=="BULLISH_TREND" and direction=="BUY") or (regime=="BEARISH_TREND" and direction=="SELL")) else (0.5 if regime=="SIDEWAYS" else 0.2)
        session = classify_session(work[-1].get("t", time.time()*1000))
        session_fit = 1.0 if session in ("LONDON","NEWYORK") else 0.45

        target_liq = min((x for x in pool if ((direction=="BUY" and x>entry) or (direction=="SELL" and x<entry))), default=tp)
        tp_distance_atr=abs(tp-entry)/max(atr_now,1e-9)
        liq_distance_atr=abs(target_liq-entry)/max(atr_now,1e-9)
        tp_viability=max(0.0,min(1.0, 0.65*min(1.0,rr/3.0)+0.35*min(1.0,liq_distance_atr/max(tp_distance_atr,1e-9))))
        # Higher RR is not capped; only probability/market structure feasibility matters.
        sl_distance_atr=abs(entry-sl)/max(atr_now,1e-9)
        sl_viability=max(0.0,min(1.0, math.exp(-abs(sl_distance_atr-1.5)/2.0)))
        setup_quality=(0.35*(1 if bos else 0)+0.20*sweep_aligned+0.15*disp_aligned+0.10*fvg_aligned+0.10*min(1.0,r2)+0.10*(1 if trend_dir==direction else 0.4))
        context=(0.45*btc_context+0.35*regime_fit+0.20*session_fit)
        execution=(0.55*fill_viability+0.45*tp_viability)
        expected_value=max(0.0,min(1.0, 0.45*min(1.0,rr/2.5)+0.30*tp_viability+0.25*sl_viability))
        confidence=self._score_confidence({"setup_quality":setup_quality,"execution_viability":execution,"context":context,"freshness":freshness_score,"expected_value":expected_value})

        reasons += [
            f"entry distance {entry_dist_atr:.2f} ATR; fill viability {fill_viability:.2f}",
            f"RR {rr:.2f}; TP viability {tp_viability:.2f}; SL viability {sl_viability:.2f}",
            f"structure age {structure_age} bars; freshness {freshness_score:.2f}",
        ]
        if sweep_aligned: reasons.append("liquidity sweep aligned")
        if disp_aligned: reasons.append("displacement aligned")
        if fvg_aligned: reasons.append("FVG aligned")
        if momentum_aligned: reasons.append("momentum aligned")
        if btc["direction"]: reasons.append(f"BTC {btc['direction']} corr={btc['corr']:.2f} rel={btc['relative']:.4f}")
        if regime: reasons.append(f"regime={regime}")

        components={
            "structure":20.0*setup_quality,
            "liquidity":15.0*(0.7*float(sweep_aligned)+0.3*min(1.0,len(pool)/2)),
            "entry_quality":15.0*(0.55*float(fvg_aligned)+0.45*execution),
            "risk_reward":15.0*min(1.0,rr/3.0),
            "momentum":10.0*(1.0 if momentum_aligned else 0.0),
            "volatility":5.0*volatility_fit,
            "btc_correlation":10.0*btc_context,
            "regime":5.0*regime_fit,
            "session":3.0*session_fit,
            "confirmation":2.0*min(1.0,(float(sweep_aligned)+float(disp_aligned)+float(fvg_aligned))/2),
        }
        threshold_passed=confidence>=self.get_active_threshold()
        setup=Setup(symbol if False else symbol,direction,entry,tp,sl,confidence,reasons,components,
                    "+".join(["SMC_BOS"]+(["SWEEP"] if sweep_aligned else [])+(["DISPLACEMENT"] if disp_aligned else [])+(["FVG"] if fvg_aligned else [])),
                    regime,session,atr_now,work[-1].get("t",time.time()*1000),self.version,threshold_passed,
                    reference_levels={
                        "bos":bos,"rr":round(rr,5),"risk":risk,"reward":reward,"geometry":geom_reason,
                        "entry_distance_atr":round(entry_dist_atr,4),"fill_viability":round(fill_viability,4),
                        "tp_viability":round(tp_viability,4),"sl_viability":round(sl_viability,4),
                        "target_liquidity":target_liq,"tp_distance_atr":round(tp_distance_atr,4),"sl_distance_atr":round(sl_distance_atr,4),
                        "sweep":sweep,"fvg":fvg,"equal_highs":levels["equal_highs"][-6:],"equal_lows":levels["equal_lows"][-6:],
                        "freshness":{"structure_age_bars":structure_age,"score":round(freshness_score,4)},
                        "btc_context":btc,"diagnosis":diagnosis,
                        "quality_dimensions":{"setup_quality":round(setup_quality,4),"execution_viability":round(execution,4),
                                              "context":round(context,4),"freshness":round(freshness_score,4),"expected_value":round(expected_value,4)},
                    })
        if diagnosis not in ("VALID",) and confidence < self.get_active_threshold():
            # Still return it when caller requests diagnostic mode.
            pass
        return setup, {"diagnosis":diagnosis,"btc":btc,"freshness":freshness_score}

    def analyze_with_diagnostics(self, symbol: str, candles: Sequence[Dict[str, float]], btc_candles: Optional[Sequence[Dict[str, float]]] = None, market_context: Optional[Dict[str, Any]] = None, enforce_threshold: bool = True) -> Tuple[Optional[Setup], Dict[str, Any]]:
        """Compatibility + rich diagnostics API for main.py. No network access."""
        work = list(_last_confirmed(candles))
        ok, reason = validate_candles(work, min_len=max(self.params["structure_lookback"], self.params["vol_regime_lookback"], self.params["atr_period"]) + 5)
        if not ok:
            return None, {"data_quality": {"valid": False, "reason": reason}, "diagnosis": "INVALID_DATA"}
        btc_work = list(_last_confirmed(btc_candles)) if btc_candles else None
        setup, diag = self._build_setup(symbol, work, btc_work)
        if setup is None:
            return None, {"data_quality": {"valid": True, "coin_n": len(work), "btc_n": len(btc_work or [])}, **diag}
        ref = setup.reference_levels
        diagnostics = {
            "data_quality": {"valid": True, "coin_n": len(work), "btc_n": len(btc_work or [])},
            "structure": {"bos": ref.get("bos"), "trend": setup.direction, "age_bars": (ref.get("freshness") or {}).get("structure_age_bars")},
            "liquidity": {"sweep": ref.get("sweep"), "equal_highs": ref.get("equal_highs"), "equal_lows": ref.get("equal_lows")},
            "entry": {"price": setup.entry, "distance_atr": ref.get("entry_distance_atr"), "fill_viability": ref.get("fill_viability")},
            "tp": {"price": setup.tp, "rr": ref.get("rr"), "viability": ref.get("tp_viability"), "distance_atr": ref.get("tp_distance_atr")},
            "sl": {"price": setup.sl, "viability": ref.get("sl_viability"), "distance_atr": ref.get("sl_distance_atr"), "geometry": ref.get("geometry")},
            "btc": ref.get("btc_context", {}),
            "freshness": ref.get("freshness", {}),
            "quality_dimensions": ref.get("quality_dimensions", {}),
            "diagnosis": ref.get("viability_diagnosis", "VALID"),
            "market_context": dict(market_context or {}),
        }
        if enforce_threshold and setup.confidence < self.get_active_threshold():
            return None, diagnostics
        return setup, diagnostics

    def analyze(self, symbol: str, candles: Sequence[Dict[str, float]], btc_candles: Optional[Sequence[Dict[str, float]]] = None, enforce_threshold: bool = True) -> Optional[Setup]:
        p=self.params
        work=list(_last_confirmed(candles))
        min_len=max(p["structure_lookback"],p["vol_regime_lookback"],p["atr_period"])+5
        ok,reason=validate_candles(work,min_len=min_len)
        if not ok:
            return None
        if btc_candles:
            btc_work=list(_last_confirmed(btc_candles))
            bok,_=validate_candles(btc_work,min_len=min(20,p["trend_lookback"]+5))
            if not bok: btc_work=None
        else:
            btc_work=None
        setup,diag=self._build_setup(symbol,work,btc_work)
        if setup is None:
            return None
        setup.reference_levels["data_quality"]={"coin_n":len(work),"btc_n":len(btc_work) if btc_work else 0}
        if diag.get("diagnosis") not in ("VALID", "VALID_HIGH_CONF", "VALID_LOW_CONF"):
            setup.reference_levels["viability_diagnosis"]=diag.get("diagnosis")
        setup.reference_levels["viability_diagnosis"] = setup.reference_levels.get("diagnosis",diag.get("diagnosis","VALID"))
        setup.reference_levels["confidence_tier"] = "HIGH" if setup.confidence>=70 else ("MEDIUM" if setup.confidence>=50 else "LOW")
        if enforce_threshold and setup.confidence < self.get_active_threshold():
            return None
        return setup

    def monitor_position(self, position: Dict[str, Any], candles: Sequence[Dict[str, float]], btc_candles: Optional[Sequence[Dict[str, float]]] = None) -> Dict[str, Any]:
        p=self.params; work=list(_last_confirmed(candles))
        ok,reason=validate_candles(work,min_len=p["atr_period"]+5)
        if not ok:
            return {"action":"STALE","new_sl":None,"reason":[f"data invalid: {reason}"],"weakness_score":0,"engine":"data"}
        atr=atr_series(work,p["atr_period"])[-1]
        direction=position["direction"]; entry=float(position.get("entry",0)); current_sl=float(position.get("sl",position.get("initial_sl",0))); tp=float(position.get("tp",0)); price=work[-1]["c"]
        risk=abs(entry-float(position.get("initial_sl",current_sl))) or atr
        profit_r=(price-entry)/risk if direction=="BUY" else (entry-price)/risk
        reasons=[]; weakness=0
        recent=work[-min(30,len(work)):]
        slope,r2=linreg_slope(_closes(recent)); aligned=(slope>0 if direction=="BUY" else slope<0)
        if not aligned: weakness+=1; reasons.append("trend weakened")
        last=work[-1]
        opposite=(last["c"]<last["o"] if direction=="BUY" else last["c"]>last["o"])
        if opposite: weakness+=1; reasons.append("opposite candle")
        fill_ts=float(position.get("fill_time",0) or 0)
        post=[c for c in work if not fill_ts or c.get("t",0)>=fill_ts] or work[-20:]
        if post:
            peak=max(_highs(post)) if direction=="BUY" else min(_lows(post))
            giveback=((peak-price)/atr if direction=="BUY" else (price-peak)/atr)
        else: giveback=0.0
        if giveback>0.6: weakness+=1; reasons.append(f"giveback {giveback:.2f} ATR")
        if giveback>1.2: weakness+=1; reasons.append("deep giveback")
        btc_conflict=False
        if btc_candles:
            bcl=_closes(btc_candles); bs,_=linreg_slope(bcl[-p["trend_lookback"]:]); btc_dir="BUY" if bs>0 else "SELL"
            corr=correlation(pct_returns(_closes(work)[-p["btc_corr_lookback"]:]),pct_returns(bcl[-p["btc_corr_lookback"]:]))
            btc_conflict=(btc_dir!=direction and abs(corr)>=0.45)
            if btc_conflict: weakness+=1; reasons.append(f"BTC conflict corr={corr:.2f}")
        action="HOLD"; new_sl=None; trail_reason=[]
        trail_count=int(position.get("trail_count",0) or 0)
        if profit_r>=float(p["trail_activation_r"]) and weakness>=2:
            swings=swing_points(post,p["swing_left"],p["swing_right"]) if len(post)>=p["swing_left"]+p["swing_right"]+3 else []
            buffer_=atr*float(p["sl_atr_buffer"])
            if direction=="BUY":
                lows=[v for _,v,t in swings if t=="L"]; structural=max(lows[-3:]) if lows else price-buffer_; candidate=structural-buffer_*0.5
                candidate=min(candidate,price-max(atr*0.05,buffer_*0.25))
                min_step=atr*float(p["trail_min_step_atr"])
                if candidate>current_sl+min_step and profit_r>=float(p["trail_min_profit_r"]): new_sl=candidate
            else:
                highs=[v for _,v,t in swings if t=="H"]; structural=min(highs[-3:]) if highs else price+buffer_; candidate=structural+buffer_*0.5
                candidate=max(candidate,price+max(atr*0.05,buffer_*0.25))
                min_step=atr*float(p["trail_min_step_atr"])
                if candidate<current_sl-min_step and profit_r>=float(p["trail_min_profit_r"]): new_sl=candidate
            if new_sl is not None:
                action="TRAIL"; trail_reason=["structure checkpoint",f"profit={profit_r:.2f}R",f"weakness={weakness}"]
            else:
                reasons.append("no sufficiently better protected SL")
        if profit_r<0 and btc_conflict and weakness>=2:
            action="EXIT_RISK"
        age_bars=max(0.0,(work[-1].get("t",0)-fill_ts)/900000.0) if fill_ts else 0.0
        if age_bars>max(24.0,float(p["stale_bars"])*4) and profit_r<0:
            action="STALE"
        return {"action":action,"new_sl":new_sl,"reason":reasons+trail_reason,"weakness_score":weakness,"engine":"vnext",
                "profit_r":profit_r,"giveback_atr":giveback,"trail_reason":trail_reason,"age_bars":age_bars,
                "tp_preference":("TP" if profit_r<max(0.5,float(p["trail_activation_r"])) and weakness<2 else "TRAIL" if action=="TRAIL" else "HOLD")}

def new_default_strategy() -> Strategy:
    return Strategy()
