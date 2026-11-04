from __future__ import annotations
"""Minimal Volumatic Trend [ChartPrime] signal engine.

Original indicator attribution: © ChartPrime, Mozilla Public License 2.0 (MPL-2.0).
Only EMA-SWMA(40), EMA(close,40), ATR/RMA(200), trend state and flips.
Volume histogram/bands/delta are visual-only and intentionally omitted.
"""
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np

LENGTH = 40
ATR_LENGTH = 200

@dataclass(frozen=True)
class VolumaticSignal:
    symbol: str
    timeframe: str
    candle_open_time: int
    candle_close_time: int
    direction: str
    bubble: str
    signal_price: float
    ema1: float
    ema2: float
    atr: float
    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _smooth_seed_sma(values: np.ndarray, length: int, alpha: float) -> np.ndarray:
    """EMA/RMA seeded by SMA of first `length` valid samples (backtest parity)."""
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(x))
    if valid.size == 0:
        return out
    first = int(valid[0])
    end = first + length
    if end > x.size or not np.all(np.isfinite(x[first:end])):
        return out
    prev = float(np.mean(x[first:end]))
    out[end - 1] = prev
    for i in range(end, x.size):
        if not np.isfinite(x[i]):
            prev = np.nan
            continue
        if not np.isfinite(prev):
            seed_end = i + length
            if seed_end > x.size or not np.all(np.isfinite(x[i:seed_end])):
                continue
            prev = float(np.mean(x[i:seed_end]))
            out[seed_end - 1] = prev
            for j in range(seed_end, x.size):
                if not np.isfinite(x[j]):
                    prev = np.nan
                    continue
                prev += alpha * (x[j] - prev)
                out[j] = prev
            break
        prev += alpha * (x[i] - prev)
        out[i] = prev
    return out


def calculate_indicators(candles: Iterable[dict[str, Any]] | np.ndarray) -> dict[str, np.ndarray]:
    if isinstance(candles, np.ndarray):
        a = np.asarray(candles, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] < 5:
            raise ValueError("candles harus matrix [time,open,high,low,close,volume]")
        t, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    else:
        rows = list(candles)
        t = np.asarray([r.get("open_time", r.get("timestamp", 0)) for r in rows], dtype=np.float64)
        o = np.asarray([r["open"] for r in rows], dtype=np.float64)
        h = np.asarray([r["high"] for r in rows], dtype=np.float64)
        l = np.asarray([r["low"] for r in rows], dtype=np.float64)
        c = np.asarray([r["close"] for r in rows], dtype=np.float64)
    n = c.size
    if not n or not (t.size == o.size == h.size == l.size == c.size):
        raise ValueError("Data candle kosong atau kolom OHLC tidak sama panjang")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Candle harus unik dan berurutan menaik")
    swma = np.full(n, np.nan, dtype=np.float64)
    if n >= 4:
        swma[3:] = (c[:-3] + 2*c[1:-2] + 2*c[2:-1] + c[3:]) / 6.0
    ema1 = _smooth_seed_sma(swma, LENGTH, 2.0/(LENGTH+1.0))
    ema2 = _smooth_seed_sma(c, LENGTH, 2.0/(LENGTH+1.0))
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    if n > 1:
        tr[1:] = np.maximum.reduce((h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr = _smooth_seed_sma(tr, ATR_LENGTH, 1.0/ATR_LENGTH)
    valid = np.zeros(n, dtype=bool)
    trend = np.zeros(n, dtype=bool)
    if n > 1:
        valid[1:] = np.isfinite(ema1[:-1]) & np.isfinite(ema2[1:])
        trend[1:] = valid[1:] & (ema1[:-1] < ema2[1:])
    flip = np.zeros(n, dtype=bool)
    if n > 2:
        flip[2:] = valid[2:] & valid[1:-1] & (trend[2:] != trend[1:-1])
    return {"timestamp":t.astype(np.int64),"open":o,"high":h,"low":l,"close":c,
            "ema1":ema1,"ema2":ema2,"atr":atr,"trend":trend,"trend_valid":valid,"flip":flip}


def latest_signal(symbol: str, timeframe: str, candles: Iterable[dict[str, Any]] | np.ndarray, interval_ms: int = 0) -> VolumaticSignal | None:
    x = calculate_indicators(candles)
    i = len(x["close"])-1
    if i < 2 or not bool(x["flip"][i]):
        return None
    if not (np.isfinite(x["ema1"][i-1]) and np.isfinite(x["ema2"][i])):
        return None
    bullish = bool(x["trend"][i])
    ts = int(x["timestamp"][i])
    return VolumaticSignal(symbol.upper(), timeframe.upper(), ts, ts+int(interval_ms),
        "LONG" if bullish else "SHORT", "BLUE" if bullish else "RED", float(x["close"][i]),
        float(x["ema1"][i]), float(x["ema2"][i]), float(x["atr"][i]) if np.isfinite(x["atr"][i]) else 0.0)


def signal_on_latest_closed(symbol: str, timeframe: str, rows: list[list[Any]], interval_ms: int, now_ms: int | None = None) -> VolumaticSignal | None:
    import time
    now_ms = int(now_ms if now_ms is not None else time.time()*1000)
    closed = []
    for r in rows:
        if len(r) >= 7 and int(r[6]) < now_ms:
            closed.append({"open_time":int(r[0]),"open":float(r[1]),"high":float(r[2]),
                "low":float(r[3]),"close":float(r[4]),"volume":float(r[5])})
    if len(closed) < ATR_LENGTH + LENGTH + 5:
        return None
    return latest_signal(symbol, timeframe, closed, interval_ms)


def signals_since(symbol: str, timeframe: str, candles: Iterable[dict[str, Any]] | np.ndarray,
                  interval_ms: int, after_open_time: int | None = None,
                  after_close_time_ms: int | None = None) -> list[VolumaticSignal]:
    """Return confirmed flips newer than a persisted candle cursor/baseline.

    Used after downtime so an opposite bubble cannot be silently missed. Old
    entry signals are still subject to their one-next-candle expiry in main.py.
    """
    x = calculate_indicators(candles)
    mask = x["flip"].copy()
    if after_open_time is not None:
        mask &= x["timestamp"] > int(after_open_time)
    elif after_close_time_ms is not None:
        mask &= (x["timestamp"] + int(interval_ms)) > int(after_close_time_ms)
    out: list[VolumaticSignal] = []
    for i in np.flatnonzero(mask).tolist():
        if not (np.isfinite(x["ema1"][i-1]) and np.isfinite(x["ema2"][i])):
            continue
        bullish = bool(x["trend"][i]); ts = int(x["timestamp"][i])
        out.append(VolumaticSignal(symbol.upper(), timeframe.upper(), ts, ts+int(interval_ms),
            "LONG" if bullish else "SHORT", "BLUE" if bullish else "RED", float(x["close"][i]),
            float(x["ema1"][i]), float(x["ema2"][i]), float(x["atr"][i]) if np.isfinite(x["atr"][i]) else 0.0))
    return out
