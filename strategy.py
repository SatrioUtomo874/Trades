#!/usr/bin/env python3
"""Volumatic Trend [ChartPrime] bubble engine.

This module is deliberately pure: it does not access Binance, Telegram, disk,
WebSocket, or exchange orders. main.py supplies already-closed OHLCV candles.

Trading rules implemented by main.py:
- blue/bullish flip -> BUY LIMIT at the confirmed signal candle close;
- red/bearish flip -> SELL LIMIT at the confirmed signal candle close;
- the order is valid only for the immediately following candle;
- a filled position is held until the opposite confirmed bubble.

The source indicator attribution/license notice must be retained when the
original Pine-derived implementation is redistributed: © ChartPrime, MPL-2.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

STRATEGY_NAME = "VOLUMATIC_TREND"
STRATEGY_VERSION = "1.2.0"
TREND_LENGTH = 40
ATR_LENGTH = 200
REFERENCE_BUFFER_ATR = 0.30
TIMEFRAME_MS = {"H4": 14_400_000, "D1": 86_400_000}
TIMEFRAME_INTERVALS = {"H4": "4h", "D1": "1d"}
SUPPORTED_TIMEFRAMES = tuple(TIMEFRAME_MS)


@dataclass(frozen=True, slots=True)
class VolumaticSignal:
    symbol: str
    timeframe: str
    signal: str                 # BLUE or RED
    direction: str              # LONG or SHORT
    signal_index: int
    signal_open_time_ms: int
    signal_close_time_ms: int
    bubble_visual_time_ms: int
    bubble_price: float         # EMA1[t-1], matching Pine diamond location; NOT the entry price
    signal_atr: float           # ATR[t-1]
    close_price: float          # exact LIMIT target: close of the confirmed signal candle
    reference_sl: float         # reference only; NEVER a live stop order
    ema1: float
    ema2: float
    trend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": STRATEGY_NAME,
            "strategy_version": STRATEGY_VERSION,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal": self.signal,
            "direction": self.direction,
            "signal_index": self.signal_index,
            "signal_open_time_ms": self.signal_open_time_ms,
            "signal_close_time_ms": self.signal_close_time_ms,
            "bubble_visual_time_ms": self.bubble_visual_time_ms,
            "bubble_price": self.bubble_price,
            "signal_atr": self.signal_atr,
            "close_price": self.close_price,
            "signal_close_price": self.close_price,
            "limit_price": self.close_price,
            "reference_sl": self.reference_sl,
            "ema1": self.ema1,
            "ema2": self.ema2,
            "trend": self.trend,
        }


def _extract_columns(candles: Any) -> tuple[np.ndarray, ...]:
    """Accept Binance structured arrays, dict-of-arrays, or sequence rows.

    BinanceREST's compact dtype fields are t/o/h/l/c/v/qv. Standard names
    timestamp/open/high/low/close/volume are accepted as well.
    """
    if isinstance(candles, Mapping):
        def pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in candles:
                    return candles[name]
            return default
        t = pick("t", "timestamp", "open_time")
        o = pick("o", "open")
        h = pick("h", "high")
        l = pick("l", "low")
        c = pick("c", "close")
        v = pick("v", "volume", default=np.zeros(len(c) if c is not None else 0))
    else:
        arr = np.asarray(candles)
        if arr.dtype.names:
            names = set(arr.dtype.names)
            def field(*choices: str, default: Any = None) -> Any:
                for name in choices:
                    if name in names:
                        return arr[name]
                return default
            t = field("t", "timestamp", "open_time")
            o = field("o", "open")
            h = field("h", "high")
            l = field("l", "low")
            c = field("c", "close")
            v = field("v", "volume", default=np.zeros(len(arr)))
        else:
            if arr.ndim != 2 or arr.shape[1] < 5:
                raise ValueError("Candles harus berupa structured array, dict, atau array Nx5+.")
            # Conventional layout: timestamp, open, high, low, close, volume (optional).
            t, o, h, l, c = (arr[:, i] for i in range(5))
            v = arr[:, 5] if arr.shape[1] > 5 else np.zeros(len(arr))

    if any(x is None for x in (t, o, h, l, c)):
        raise ValueError("Kolom timestamp/open/high/low/close tidak lengkap.")
    arrays = tuple(np.asarray(x, dtype=np.float64).reshape(-1) for x in (t, o, h, l, c, v))
    lengths = {len(x) for x in arrays}
    if len(lengths) != 1:
        raise ValueError("Panjang kolom OHLCV tidak sama.")
    timestamps, opens, highs, lows, closes, volumes = arrays
    if len(closes) < 3:
        raise ValueError("Minimal 3 candle diperlukan.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Timestamp candle mengandung nilai non-finite.")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Candle harus terurut menaik dan tidak duplikat.")
    if np.any(~np.isfinite(opens) | ~np.isfinite(highs) | ~np.isfinite(lows) | ~np.isfinite(closes)):
        raise ValueError("OHLC mengandung nilai non-finite.")
    if np.any(highs < lows):
        raise ValueError("Ada candle dengan high lebih rendah dari low.")
    return timestamps.astype(np.int64), opens, highs, lows, closes, volumes


def _ema_tv(values: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.ema-style recursive EMA seeded by first valid source value."""
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if length < 1:
        raise ValueError("EMA length harus >= 1.")
    alpha = 2.0 / (float(length) + 1.0)
    previous = np.nan
    for i, value in enumerate(x):
        if not np.isfinite(value):
            continue
        if not np.isfinite(previous):
            previous = float(value)
        else:
            previous = alpha * float(value) + (1.0 - alpha) * previous
        out[i] = previous
    return out


def _rma_tv(values: np.ndarray, length: int) -> np.ndarray:
    """Wilder RMA seeded with SMA of the first `length` valid observations."""
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if length < 1:
        raise ValueError("RMA length harus >= 1.")
    valid_count = 0
    seed_sum = 0.0
    previous = np.nan
    for i, value in enumerate(x):
        if not np.isfinite(value):
            continue
        value = float(value)
        if valid_count < length:
            seed_sum += value
            valid_count += 1
            if valid_count == length:
                previous = seed_sum / length
                out[i] = previous
            continue
        previous = (previous * (length - 1) + value) / length
        out[i] = previous
    return out


def calculate_indicators(
    candles: Any,
    *,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
) -> dict[str, np.ndarray]:
    """Calculate EMA-SWMA, EMA close, ATR-RMA, trend, and confirmed flips.

    Exact trend formula: trend[t] = ema1[t-1] < ema2[t].
    signal[t] is +1 for blue/bullish flip, -1 for red/bearish flip, and 0
    when no valid flip is present. Indices without enough history are invalid.
    """
    ts, opens, highs, lows, closes, volumes = _extract_columns(candles)
    n = len(closes)

    swma = np.full(n, np.nan, dtype=np.float64)
    if n >= 4:
        swma[3:] = (
            closes[:-3] / 6.0
            + 2.0 * closes[1:-2] / 6.0
            + 2.0 * closes[2:-1] / 6.0
            + closes[3:] / 6.0
        )

    ema1 = _ema_tv(swma, trend_length)
    ema2 = _ema_tv(closes, trend_length)

    tr = np.full(n, np.nan, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    if n > 1:
        previous_close = closes[:-1]
        tr[1:] = np.maximum.reduce((
            highs[1:] - lows[1:],
            np.abs(highs[1:] - previous_close),
            np.abs(lows[1:] - previous_close),
        ))
    atr = _rma_tv(tr, atr_length)

    # int8 state: +1 bullish, -1 bearish, 0 not yet calculable.
    trend = np.zeros(n, dtype=np.int8)
    valid = np.zeros(n, dtype=bool)
    if n > 1:
        valid[1:] = np.isfinite(ema1[:-1]) & np.isfinite(ema2[1:])
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            trend[valid_indices] = np.where(
                ema1[valid_indices - 1] < ema2[valid_indices], 1, -1
            )

    signal = np.zeros(n, dtype=np.int8)
    if n > 2:
        flip_valid = valid[1:] & valid[:-1]
        idx = np.flatnonzero(flip_valid) + 1
        changed = trend[idx] != trend[idx - 1]
        flip_idx = idx[changed]
        signal[flip_idx] = trend[flip_idx]

    return {
        "timestamp": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "swma": swma,
        "ema1": ema1,
        "ema2": ema2,
        "tr": tr,
        "atr": atr,
        "trend": trend,
        "valid": valid,
        "signal": signal,
    }


def _signal_from_indicators(
    indicators: dict[str, np.ndarray],
    *,
    index: int,
    symbol: str,
    timeframe: str,
    interval_ms: int,
    reference_buffer_atr: float = REFERENCE_BUFFER_ATR,
) -> dict[str, Any] | None:
    i = int(index)
    code = int(indicators["signal"][i])
    if code == 0:
        return None
    bubble_index = i - 1
    bubble_price = float(indicators["ema1"][bubble_index])
    signal_atr = float(indicators["atr"][bubble_index])
    if not (np.isfinite(bubble_price) and np.isfinite(signal_atr) and signal_atr > 0):
        reference_sl = None
    elif code == 1:
        reference_sl = bubble_price - float(reference_buffer_atr) * signal_atr
    else:
        reference_sl = bubble_price + float(reference_buffer_atr) * signal_atr

    direction = "LONG" if code == 1 else "SHORT"
    signal_close_ms = int(indicators["timestamp"][i]) + int(interval_ms)
    return VolumaticSignal(
        symbol=str(symbol).upper(),
        timeframe=str(timeframe).upper(),
        signal="BLUE" if code == 1 else "RED",
        direction=direction,
        signal_index=i,
        signal_open_time_ms=int(indicators["timestamp"][i]),
        signal_close_time_ms=signal_close_ms,
        bubble_visual_time_ms=int(indicators["timestamp"][bubble_index]),
        bubble_price=bubble_price,
        signal_atr=signal_atr,
        close_price=float(indicators["close"][i]),
        reference_sl=float(reference_sl) if reference_sl is not None else float("nan"),
        ema1=float(indicators["ema1"][i]),
        ema2=float(indicators["ema2"][i]),
        trend="BULLISH" if code == 1 else "BEARISH",
    ).to_dict()


def detect_latest_signal(
    candles: Any,
    *,
    symbol: str,
    timeframe: str,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
    reference_buffer_atr: float = REFERENCE_BUFFER_ATR,
    interval_ms: int | None = None,
) -> dict[str, Any] | None:
    """Return a signal only when the newest supplied (already closed) candle flips."""
    tf = str(timeframe).upper()
    if tf not in TIMEFRAME_MS:
        raise ValueError(f"Timeframe tidak didukung: {timeframe}. Gunakan H4 atau D1.")
    span_ms = int(interval_ms or TIMEFRAME_MS[tf])
    indicators = calculate_indicators(candles, trend_length=trend_length, atr_length=atr_length)
    index = len(indicators["close"]) - 1
    return _signal_from_indicators(
        indicators, index=index, symbol=symbol, timeframe=tf, interval_ms=span_ms,
        reference_buffer_atr=reference_buffer_atr,
    )


def detect_signals_since(
    candles: Any,
    *,
    symbol: str,
    timeframe: str,
    after_close_time_ms: int,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
    reference_buffer_atr: float = REFERENCE_BUFFER_ATR,
    interval_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Return every confirmed bubble newer than a checkpoint, oldest first.

    This is used to recover filled-position monitoring after a restart or API
    delay so an opposite bubble is not silently missed just because it is no
    longer the newest candle. Entry orders are still allowed only for the
    immediately following candle; main.py decides whether that window is fresh.
    """
    tf = str(timeframe).upper()
    if tf not in TIMEFRAME_MS:
        raise ValueError(f"Timeframe tidak didukung: {timeframe}. Gunakan H4 atau D1.")
    span_ms = int(interval_ms or TIMEFRAME_MS[tf])
    indicators = calculate_indicators(candles, trend_length=trend_length, atr_length=atr_length)
    codes = indicators["signal"]
    indices = np.flatnonzero(codes)
    output: list[dict[str, Any]] = []
    checkpoint = int(after_close_time_ms)
    for index in indices.tolist():
        close_ms = int(indicators["timestamp"][index]) + span_ms
        if close_ms <= checkpoint:
            continue
        item = _signal_from_indicators(
            indicators, index=index, symbol=symbol, timeframe=tf, interval_ms=span_ms,
            reference_buffer_atr=reference_buffer_atr,
        )
        if item is not None:
            output.append(item)
    return output

def signals_since(
    candles: Any,
    *,
    symbol: str,
    timeframe: str,
    after_close_time_ms: int,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
    reference_buffer_atr: float = REFERENCE_BUFFER_ATR,
    interval_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Launcher-compatible public API for retrieving confirmed bubble events.

    try.py validates the strategy module by requiring a top-level
    ``signals_since`` function. main.py currently calls the explicit
    ``detect_signals_since`` name; this wrapper intentionally exposes both
    names with identical behavior so launcher validation and the running
    engine remain compatible.
    """
    return detect_signals_since(
        candles,
        symbol=symbol,
        timeframe=timeframe,
        after_close_time_ms=after_close_time_ms,
        trend_length=trend_length,
        atr_length=atr_length,
        reference_buffer_atr=reference_buffer_atr,
        interval_ms=interval_ms,
    )


def analyze(
    candles: Any,
    *,
    symbol: str,
    timeframe: str,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
) -> dict[str, Any]:
    """Full latest-bar analysis response, including no-signal state."""
    indicators = calculate_indicators(candles, trend_length=trend_length, atr_length=atr_length)
    tf = str(timeframe).upper()
    if tf not in TIMEFRAME_MS:
        raise ValueError("Timeframe harus H4 atau D1.")
    i = len(indicators["close"]) - 1
    signal = detect_latest_signal(
        candles, symbol=symbol, timeframe=tf,
        trend_length=trend_length, atr_length=atr_length,
    )
    return {
        "strategy": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "symbol": str(symbol).upper(),
        "timeframe": tf,
        "latest_open_time_ms": int(indicators["timestamp"][i]),
        "latest_close_time_ms": int(indicators["timestamp"][i]) + TIMEFRAME_MS[tf],
        "latest_close": float(indicators["close"][i]),
        "trend": (
            "BULLISH" if indicators["trend"][i] == 1
            else "BEARISH" if indicators["trend"][i] == -1
            else "WARMING_UP"
        ),
        "signal": signal,
        "indicators": indicators,
    }


def reference_risk(direction: str, entry_price: float, reference_sl: float) -> float | None:
    """Return positive initial risk distance, or None; reference level is not an SL."""
    try:
        entry = float(entry_price)
        reference = float(reference_sl)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(entry) or not np.isfinite(reference) or entry <= 0:
        return None
    side = str(direction).upper()
    risk = entry - reference if side == "LONG" else reference - entry if side == "SHORT" else -1.0
    return float(risk) if np.isfinite(risk) and risk > 0 else None


def trade_result_r(direction: str, entry_price: float, exit_price: float, initial_risk: float | None) -> float | None:
    """Calculate realized R without creating or implying a live stop order."""
    if initial_risk is None or not np.isfinite(float(initial_risk)) or float(initial_risk) <= 0:
        return None
    entry, exit_ = float(entry_price), float(exit_price)
    if not np.isfinite(entry) or not np.isfinite(exit_):
        return None
    side = str(direction).upper()
    move = exit_ - entry if side == "LONG" else entry - exit_ if side == "SHORT" else None
    if move is None:
        return None
    return float(move / float(initial_risk))


__all__ = [
    "STRATEGY_NAME", "STRATEGY_VERSION", "TREND_LENGTH", "ATR_LENGTH",
    "REFERENCE_BUFFER_ATR", "TIMEFRAME_MS", "TIMEFRAME_INTERVALS", "SUPPORTED_TIMEFRAMES", "VolumaticSignal",
    "calculate_indicators", "detect_latest_signal", "detect_signals_since", "signals_since", "analyze",
    "reference_risk", "trade_result_r",
]
