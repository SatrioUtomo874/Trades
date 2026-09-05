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
    "ACTIVE_THRESHOLD": 45.0,       # % — dimulai rendah agar learn.py punya data (§10)
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
}


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
        return {"version": self.version, "params": dict(self.params)}

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.version = state.get("version", self.version)
        self.params.update(state.get("params", {}))

    # -- analisis utama -------------------------------------------------------
    def analyze(
        self,
        symbol: str,
        candles: Sequence[Dict[str, float]],
        btc_candles: Optional[Sequence[Dict[str, float]]] = None,
    ) -> Optional[Setup]:
        p = self.params
        min_len = max(p["structure_lookback"], p["vol_regime_lookback"], p["atr_period"]) + 5
        if len(candles) < min_len:
            return None

        # Entry hanya memakai candle yang sudah confirm. Candle terakhir WebSocket
        # dapat berubah dan menyebabkan entry instan.
        if len(candles) < 3:
            return None
        candles = list(candles[:-1])
        atrs = atr_series(candles, p["atr_period"])
        atr_now = atrs[-1]
        if atr_now <= 0:
            return None

        closes = _closes(candles)
        last_close = closes[-1]

        # --- market structure ---
        struct_window = candles[-p["structure_lookback"]:]
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

        # --- liquidity ---
        levels = equal_levels(swings, atr_now, p["equal_level_tol_atr"])
        sweep = detect_liquidity_sweep(candles, p["sweep_lookback"])
        liquidity_score = 0.0
        if sweep:
            if (direction == "BUY" and sweep["type"] == "BULLISH_SWEEP") or (
                direction == "SELL" and sweep["type"] == "BEARISH_SWEEP"
            ):
                liquidity_score += CONFIDENCE_WEIGHTS["liquidity"] * 0.7
                reasons.append(f"liquidity sweep searah ({sweep['type']})")
        pool = levels["equal_highs"] if direction == "BUY" else levels["equal_lows"]
        if pool:
            liquidity_score += CONFIDENCE_WEIGHTS["liquidity"] * 0.3
            reasons.append("equal high/low terdeteksi sebagai target likuiditas")
        liquidity_score = min(liquidity_score, CONFIDENCE_WEIGHTS["liquidity"])

        # --- displacement & FVG (entry quality) ---
        disp = detect_displacement(candles, atr_now, p["displacement_atr_mult"])
        fvg = detect_fvg(candles)
        entry_quality_score = 0.0
        setup_type_parts = ["SMC_BOS"]
        if disp and disp["direction"] == direction:
            entry_quality_score += CONFIDENCE_WEIGHTS["entry_quality"] * 0.6
            reasons.append("displacement candle searah")
            setup_type_parts.append("DISPLACEMENT")
        if fvg and (
            (direction == "BUY" and fvg["type"] == "BULLISH_FVG")
            or (direction == "SELL" and fvg["type"] == "BEARISH_FVG")
        ):
            entry_quality_score += CONFIDENCE_WEIGHTS["entry_quality"] * 0.4
            reasons.append("imbalance/FVG mendukung entry")
            setup_type_parts.append("FVG")
        entry_quality_score = min(entry_quality_score, CONFIDENCE_WEIGHTS["entry_quality"])

        # --- entry / TP / SL ---
        buffer_ = atr_now * p["sl_atr_buffer"]
        if direction == "BUY":
            entry = last_close
            recent_low = min(_lows(candles[-p["swing_left"] - p["swing_right"] - 3:]))
            sl = min(recent_low, entry - atr_now * 0.5) - buffer_
            target_pool = levels["equal_highs"]
            tp = max(target_pool) if target_pool else entry + (entry - sl) * 2.0
            if tp <= entry:
                tp = entry + (entry - sl) * 2.0
        else:
            entry = last_close
            recent_high = max(_highs(candles[-p["swing_left"] - p["swing_right"] - 3:]))
            sl = max(recent_high, entry + atr_now * 0.5) + buffer_
            target_pool = levels["equal_lows"]
            tp = min(target_pool) if target_pool else entry - (sl - entry) * 2.0
            if tp >= entry:
                tp = entry - (sl - entry) * 2.0

        ok, geom_reason = validate_geometry(direction, entry, sl, tp, atr_val=atr_now)
        if not ok:
            return None

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0.0
        rr_score = 0.0
        if rr >= p["min_rr"]:
            rr_score = min(CONFIDENCE_WEIGHTS["risk_reward"], CONFIDENCE_WEIGHTS["risk_reward"] * (rr / max(2.0, p["min_rr"] * 1.5)))
            reasons.append(f"risk/reward {rr:.2f}R memenuhi minimum")
        else:
            return None  # RR di bawah minimum -> bukan kandidat valid

        # --- momentum ---
        mlb = p["momentum_lookback"]
        roc = 0.0
        if len(closes) > mlb and closes[-mlb - 1] != 0:
            roc = (closes[-1] - closes[-mlb - 1]) / closes[-mlb - 1]
        momentum_aligned = (direction == "BUY" and roc > 0) or (direction == "SELL" and roc < 0)
        momentum_score = CONFIDENCE_WEIGHTS["momentum"] * min(1.0, abs(roc) * 20) if momentum_aligned else 0.0
        if momentum_aligned:
            reasons.append("momentum (ROC) searah")

        # --- volatility regime ---
        vol_lb = atrs[-p["vol_regime_lookback"]:] if len(atrs) >= p["vol_regime_lookback"] else atrs
        vol_rank = sorted(vol_lb).index(min(vol_lb, key=lambda x: abs(x - atr_now))) / max(1, len(vol_lb) - 1)
        volatility_score = CONFIDENCE_WEIGHTS["volatility"] * (1.0 - abs(vol_rank - 0.5) * 2)
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
                btc_corr_score = CONFIDENCE_WEIGHTS["btc_correlation"] * min(1.0, corr)
                reasons.append(f"selaras dengan tren BTC (corr={corr:.2f})")
            elif corr < -0.3 and btc_dir != direction:
                btc_corr_score = CONFIDENCE_WEIGHTS["btc_correlation"] * min(1.0, abs(corr)) * 0.7
                reasons.append(f"korelasi negatif terhadap BTC mendukung arah (corr={corr:.2f})")
        else:
            btc_corr_score = CONFIDENCE_WEIGHTS["btc_correlation"] * 0.5  # netral utk BTCUSDT sendiri / data tak tersedia

        # --- regime & session ---
        regime = classify_regime(btc_candles if btc_candles else candles, p)
        regime_score = 0.0
        if (regime == "BULLISH_TREND" and direction == "BUY") or (
            regime == "BEARISH_TREND" and direction == "SELL"
        ):
            regime_score = CONFIDENCE_WEIGHTS["regime"]
            reasons.append(f"searah market regime ({regime})")
        elif regime == "SIDEWAYS":
            regime_score = CONFIDENCE_WEIGHTS["regime"] * 0.4

        session = classify_session(candles[-1].get("t", time.time() * 1000))
        session_score = CONFIDENCE_WEIGHTS["session"] if session in ("LONDON", "NEWYORK") else CONFIDENCE_WEIGHTS["session"] * 0.3

        confirmation_count = sum([bool(sweep), bool(fvg), bool(disp), bool(pool)])
        confirmation_score = CONFIDENCE_WEIGHTS["confirmation"] * min(1.0, confirmation_count / 3)

        structure_score = CONFIDENCE_WEIGHTS["structure"] * min(1.0, 0.5 + r2 * 0.5) if bos else 0.0

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

        if confidence < self.get_active_threshold():
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
            timestamp=candles[-1].get("t", time.time() * 1000),
            strategy_version=self.version,
        )

    # -- monitoring posisi aktif (trailing) -----------------------------------
    def monitor_position(
        self, position: Dict[str, Any], candles: Sequence[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Evaluasi posisi aktif untuk trailing. Tujuannya BUKAN mencari
        entry baru, melainkan structure/momentum/weakness monitoring (§18/19).
        """
        p = self.params
        if len(candles) < p["atr_period"] + 5:
            return {"action": "HOLD", "new_sl": None, "reason": ["data belum cukup"], "weakness_score": 0, "engine": "none"}

        atrs = atr_series(candles, p["atr_period"])
        atr_now = atrs[-1]
        direction = position["direction"]
        entry = position["entry"]
        current_sl = position["sl"]
        tp = position["tp"]
        last = candles[-1]
        price = last["c"]

        risk = abs(entry - current_sl) or atr_now
        profit_r = (price - entry) / risk if direction == "BUY" else (entry - price) / risk

        reasons: List[str] = []
        weakness = 0

        closes = _closes(candles[-p["momentum_lookback"] - 1 :])
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

        peak_since_entry = max(_highs(candles)) if direction == "BUY" else min(_lows(candles))
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
            if direction == "BUY":
                candidate = price - buffer_
                if candidate > current_sl:
                    new_sl = candidate
            else:
                candidate = price + buffer_
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
