#!/usr/bin/env python3
"""Volumatic M15 + H4 EMA filter + M15 Supertrend protection.

The original ChartPrime-derived EMA-SWMA bubble logic is retained. This module
adds the closed-candle execution model used by backtest.py: a confirmed M15
bubble is filtered by the latest fully closed H4 EMA state, enters at the next
M15 candle open, takes the M15 Supertrend line as SL and computes a fixed 3R TP.
No intrabar/visual-offset Mode-B signal is used by the live engine.

Original indicator attribution/license notice: © ChartPrime, MPL-2.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

STRATEGY_NAME = "VOLUMATIC_TREND"
STRATEGY_VERSION = "2.0.0"
DEFAULT_TREND_LENGTH = 100
TREND_LENGTH = DEFAULT_TREND_LENGTH  # backward-compatible public default
MIN_TREND_LENGTH = 1
MAX_TREND_LENGTH = (2 ** 63) - 1
ATR_LENGTH = 200
REFERENCE_BUFFER_ATR = 0.30
SUPPORTED_EMA_FILTERS = ()  # kept for backward compatibility; all positive periods are accepted
MIN_EMA_FILTER_LENGTH = 1
MAX_EMA_FILTER_LENGTH = (2 ** 63) - 1
DEFAULT_EMA_FILTER_LENGTH = 100
TIMEFRAME_MS = {"M15": 900_000, "H4": 14_400_000, "D1": 86_400_000}
TIMEFRAME_INTERVALS = {"M15": "15m", "H4": "4h", "D1": "1d"}
SUPPORTED_TIMEFRAMES = ("M15",)
EMA_TIMEFRAME = "H4"
SIGNAL_TIMEFRAME = "M15"
SUPERTREND_ATR_LENGTH = 10
SUPERTREND_FACTOR = 3.0
RR_TARGET = 3.0


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
    trend_length: int
    ema_filter_length: int | tuple[int, int] | None
    ema_filter_value: float | None
    ema_filter_slow_value: float | None
    ema_filter_passed: bool
    ema_filter_ready: bool

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
            "trend_length": self.trend_length,
            "ema_filter_length": list(self.ema_filter_length) if isinstance(self.ema_filter_length, tuple) else self.ema_filter_length,
            "ema_filter_mode": "DUAL" if isinstance(self.ema_filter_length, tuple) else ("SINGLE" if self.ema_filter_length is not None else "OFF"),
            "ema_filter_value": self.ema_filter_value,
            "ema_filter_slow_value": self.ema_filter_slow_value,
            "ema_filter_passed": self.ema_filter_passed,
            "ema_filter_ready": self.ema_filter_ready,
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


def normalize_trend_length(value: Any) -> int:
    """Validate a positive integer Volumatic Trend length without preset choices."""
    if isinstance(value, bool):
        raise ValueError("Trend length harus berupa bilangan bulat positif.")
    raw = str(value).strip()
    try:
        length = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Trend length harus berupa bilangan bulat positif.") from exc
    if raw not in {str(length), f"+{length}"}:
        raise ValueError("Trend length harus berupa bilangan bulat positif.")
    if not MIN_TREND_LENGTH <= length <= MAX_TREND_LENGTH:
        raise ValueError("Trend length harus lebih besar dari 0 dan masih dalam batas integer 64-bit.")
    return length


def normalize_ema_filter_length(value: Any) -> int | tuple[int, int] | None:
    """Normalize /EMA configuration with unrestricted positive integer periods.

    - None/off disables filtering.
    - Any single positive integer keeps the close-vs-one-EMA rule.
    - Two positive integers enable EMA alignment. They are sorted so the
      numerically smaller period is always fast and the larger is slow.
    """
    if value is None or (isinstance(value, str) and value.strip().lower() in {"off", "none", "0", ""}):
        return None

    if isinstance(value, str):
        raw_parts = value.replace(",", " ").split()
        if len(raw_parts) == 2:
            value = raw_parts
        elif len(raw_parts) == 1:
            value = raw_parts[0]
        else:
            raise ValueError("Gunakan /EMA <angka>, /EMA off, atau /EMA <angka1> <angka2>.")

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Mode EMA ganda membutuhkan tepat dua angka: /EMA FAST SLOW.")
        parsed: list[int] = []
        for item in value:
            if isinstance(item, bool):
                raise ValueError("EMA harus berupa bilangan bulat positif.")
            raw = str(item).strip()
            try:
                period = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("EMA harus berupa bilangan bulat positif.") from exc
            if raw not in {str(period), f"+{period}"}:
                raise ValueError("EMA harus berupa bilangan bulat positif.")
            if not MIN_EMA_FILTER_LENGTH <= period <= MAX_EMA_FILTER_LENGTH:
                raise ValueError("Periode EMA harus lebih besar dari 0 dan masih dalam batas integer 64-bit.")
            parsed.append(period)
        if parsed[0] == parsed[1]:
            raise ValueError("Mode dual EMA membutuhkan dua periode berbeda, contoh: /EMA 89 95.")
        return tuple(sorted(parsed))

    if isinstance(value, bool):
        raise ValueError("EMA tunggal harus berupa bilangan bulat positif atau OFF.")
    raw = str(value).strip()
    try:
        length = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("EMA tunggal harus berupa bilangan bulat positif atau OFF.") from exc
    if raw not in {str(length), f"+{length}"}:
        raise ValueError("EMA tunggal harus berupa bilangan bulat positif.")
    if not MIN_EMA_FILTER_LENGTH <= length <= MAX_EMA_FILTER_LENGTH:
        raise ValueError("Periode EMA harus lebih besar dari 0 dan masih dalam batas integer 64-bit.")
    return length


def format_ema_filter_length(value: Any) -> str:
    """Human-readable EMA mode label for telemetry and Telegram."""
    selected = normalize_ema_filter_length(value)
    if selected is None:
        return "OFF"
    if isinstance(selected, tuple):
        return f"EMA{selected[0]}/EMA{selected[1]}"
    return f"EMA{selected}"


def ema_filter_lengths(value: Any) -> tuple[int, int] | tuple[int] | tuple[()]:
    selected = normalize_ema_filter_length(value)
    if selected is None:
        return ()
    if isinstance(selected, tuple):
        return selected
    return (selected,)


def _ema_filter_decision(
    direction: str,
    close_price: float,
    config: int | tuple[int, int] | None,
    fast_value: float | None,
    slow_value: float | None = None,
) -> tuple[bool, bool]:
    """Return (passed, ready) for single-EMA or dual-EMA directional mode."""
    selected = normalize_ema_filter_length(config)
    if selected is None:
        return True, True
    if fast_value is None or not np.isfinite(float(fast_value)):
        return False, False
    side = str(direction).upper()
    if isinstance(selected, tuple):
        if slow_value is None or not np.isfinite(float(slow_value)):
            return False, False
        if side == "LONG":
            return bool(float(fast_value) > float(slow_value)), True
        if side == "SHORT":
            return bool(float(fast_value) < float(slow_value)), True
        return False, True
    if side == "LONG":
        return bool(float(close_price) > float(fast_value)), True
    if side == "SHORT":
        return bool(float(close_price) < float(fast_value)), True
    return False, True


def calculate_indicators(
    candles: Any,
    *,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
    ema_filter_length: int | tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Calculate EMA-SWMA, EMA close, ATR-RMA, trend, and confirmed flips.

    Exact trend formula: trend[t] = ema1[t-1] < ema2[t].
    signal[t] is +1 for blue/bullish flip, -1 for red/bearish flip, and 0
    when no valid flip is present. Indices without enough history are invalid.
    """
    trend_length = normalize_trend_length(trend_length)
    atr_length = int(atr_length)
    if atr_length < 1:
        raise ValueError("ATR length harus >= 1.")
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
    filter_length = normalize_ema_filter_length(ema_filter_length)
    if isinstance(filter_length, tuple):
        ema_filter = _ema_tv(closes, filter_length[0])
        ema_filter_slow = _ema_tv(closes, filter_length[1])
        filter_length_meta = np.asarray(filter_length, dtype=np.int64)
    elif filter_length is not None:
        ema_filter = _ema_tv(closes, filter_length)
        ema_filter_slow = np.full(n, np.nan, dtype=np.float64)
        filter_length_meta = np.asarray([filter_length, 0], dtype=np.int64)
    else:
        ema_filter = np.full(n, np.nan, dtype=np.float64)
        ema_filter_slow = np.full(n, np.nan, dtype=np.float64)
        filter_length_meta = np.asarray([0, 0], dtype=np.int64)

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
        "ema_filter": ema_filter,
        "ema_filter_slow": ema_filter_slow,
        "ema_filter_length": filter_length_meta,
        "trend_length": np.asarray([trend_length], dtype=np.int64),
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
    signal_close_price = float(indicators["close"][i])
    raw_lengths = np.asarray(indicators.get("ema_filter_length", np.asarray([0, 0]))).reshape(-1)
    fast_len = int(raw_lengths[0]) if raw_lengths.size else 0
    slow_len = int(raw_lengths[1]) if raw_lengths.size > 1 else 0
    filter_length: int | tuple[int, int] | None = ((fast_len, slow_len) if fast_len and slow_len else (fast_len or None))
    raw_filter_value = float(indicators["ema_filter"][i]) if fast_len else float("nan")
    raw_filter_slow_value = float(indicators["ema_filter_slow"][i]) if slow_len else float("nan")
    filter_value = raw_filter_value if np.isfinite(raw_filter_value) else None
    filter_slow_value = raw_filter_slow_value if np.isfinite(raw_filter_slow_value) else None
    filter_passed, filter_ready = _ema_filter_decision(
        direction, signal_close_price, filter_length, filter_value, filter_slow_value
    )
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
        close_price=signal_close_price,
        reference_sl=float(reference_sl) if reference_sl is not None else float("nan"),
        ema1=float(indicators["ema1"][i]),
        ema2=float(indicators["ema2"][i]),
        trend="BULLISH" if code == 1 else "BEARISH",
        trend_length=int(indicators.get("trend_length", np.asarray([TREND_LENGTH]))[0]),
        ema_filter_length=filter_length,
        ema_filter_value=filter_value,
        ema_filter_slow_value=filter_slow_value,
        ema_filter_passed=bool(filter_passed),
        ema_filter_ready=bool(filter_ready),
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
    ema_filter_length: int | tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """Return a signal only when the newest supplied (already closed) candle flips."""
    tf = str(timeframe).upper()
    if tf not in TIMEFRAME_MS:
        raise ValueError(f"Timeframe tidak didukung: {timeframe}. Gunakan H4 atau D1.")
    span_ms = int(interval_ms or TIMEFRAME_MS[tf])
    indicators = calculate_indicators(
        candles, trend_length=trend_length, atr_length=atr_length,
        ema_filter_length=ema_filter_length,
    )
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
    ema_filter_length: int | tuple[int, int] | None = None,
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
    indicators = calculate_indicators(
        candles, trend_length=trend_length, atr_length=atr_length,
        ema_filter_length=ema_filter_length,
    )
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


def prepare_live_visual_context(
    candles: Any,
    *,
    symbol: str,
    timeframe: str,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
    reference_buffer_atr: float = REFERENCE_BUFFER_ATR,
    interval_ms: int | None = None,
    ema_filter_length: int | tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """Precompute the closed-candle state needed for intrabar mode-B checks.

    ``candles`` includes one currently-forming last row. The function computes
    EMA/ATR/filter once; subsequent WebSocket ticks only need a scalar EMA2
    update, avoiding a full historical indicator recalculation per tick.
    """
    tf = str(timeframe).upper()
    if tf not in TIMEFRAME_MS:
        raise ValueError(f"Timeframe tidak didukung: {timeframe}. Gunakan H4 atau D1.")
    trend_length = normalize_trend_length(trend_length)
    span_ms = int(interval_ms or TIMEFRAME_MS[tf])
    indicators = calculate_indicators(
        candles, trend_length=trend_length, atr_length=atr_length,
        ema_filter_length=ema_filter_length,
    )
    n = len(indicators["close"])
    if n < 3:
        return None
    live_index = n - 1
    bubble_index = live_index - 1
    live_open_ms = int(indicators["timestamp"][live_index])
    bubble_open_ms = int(indicators["timestamp"][bubble_index])
    if live_open_ms != bubble_open_ms + span_ms:
        return None

    ema1_bubble = float(indicators["ema1"][bubble_index])
    ema2_bubble = float(indicators["ema2"][bubble_index])
    atr_bubble = float(indicators["atr"][bubble_index])
    bubble_close = float(indicators["close"][bubble_index])
    prior_trend = int(indicators["trend"][bubble_index])
    if prior_trend not in (-1, 1) or not all(
        np.isfinite(x) for x in (ema1_bubble, ema2_bubble, atr_bubble, bubble_close)
    ) or atr_bubble <= 0 or bubble_close <= 0:
        return None

    raw_lengths = np.asarray(indicators.get("ema_filter_length", np.asarray([0, 0]))).reshape(-1)
    fast_len = int(raw_lengths[0]) if raw_lengths.size else 0
    slow_len = int(raw_lengths[1]) if raw_lengths.size > 1 else 0
    filter_length: int | tuple[int, int] | None = ((fast_len, slow_len) if fast_len and slow_len else (fast_len or None))
    raw_filter_fast = float(indicators["ema_filter"][bubble_index]) if fast_len else float("nan")
    raw_filter_slow = float(indicators["ema_filter_slow"][bubble_index]) if slow_len else float("nan")
    filter_value = raw_filter_fast if np.isfinite(raw_filter_fast) else None
    filter_slow_value = raw_filter_slow if np.isfinite(raw_filter_slow) else None
    filter_ready = (
        True if filter_length is None
        else filter_value is not None and (not isinstance(filter_length, tuple) or filter_slow_value is not None)
    )

    return {
        "strategy": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "symbol": str(symbol).upper(),
        "timeframe": tf,
        "trend_length": trend_length,
        "signal_mode": "VISUAL_OFFSET_INTRABAR_B",
        "repaint_risk": True,
        # The signal's visual candle is the penultimate CLOSED row. Pine's
        # offset=-1 paints the flip on that candle while the following candle
        # is live. Mode B intentionally acts on that visual marker as it forms.
        "signal_index": bubble_index,
        "detection_index": live_index,
        "signal_open_time_ms": bubble_open_ms,
        "signal_close_time_ms": live_open_ms,
        "bubble_visual_time_ms": bubble_open_ms,
        "bubble_visual_close_time_ms": live_open_ms,
        "visual_candle_open_time_ms": bubble_open_ms,
        "visual_candle_close_time_ms": live_open_ms,
        "entry_candle_open_time_ms": live_open_ms,
        "entry_candle_close_time_ms": live_open_ms + span_ms,
        "entry_window_expiry_ms": live_open_ms + span_ms,
        "bubble_price": ema1_bubble,
        "signal_atr": atr_bubble,
        "close_price": bubble_close,
        "visual_candle_close_price": bubble_close,
        "signal_close_price": bubble_close,
        "limit_price": bubble_close,
        "limit_price_source": "VISUAL_BUBBLE_CANDLE_CLOSE",
        "reference_sl": None,  # direction-specific value is populated at evaluation
        "ema1": ema1_bubble,
        "ema2": ema2_bubble,
        "prior_trend": prior_trend,
        "ema2_alpha": 2.0 / (float(trend_length) + 1.0),
        "ema_filter_length": filter_length,
        "ema_filter_mode": "DUAL" if isinstance(filter_length, tuple) else ("SINGLE" if filter_length is not None else "OFF"),
        "ema_filter_value": filter_value,
        "ema_filter_slow_value": filter_slow_value,
        "ema_filter_ready": filter_ready,
        "ema_filter_passed": False,
        "reference_buffer_atr": float(reference_buffer_atr),
        "interval_ms": span_ms,
    }


def evaluate_live_visual_context(
    context: Mapping[str, Any] | None,
    live_price: float,
    *,
    now_ms: int,
) -> dict[str, Any] | None:
    """Evaluate one live price against a prepared mode-B visual bubble context."""
    if not context:
        return None
    price = float(live_price)
    if not np.isfinite(price) or price <= 0:
        return None
    live_open_ms = int(context["entry_candle_open_time_ms"])
    span_ms = int(context["interval_ms"])
    live_close_ms = live_open_ms + span_ms
    current_ms = int(now_ms)
    if not (live_open_ms <= current_ms < live_close_ms):
        return None

    alpha = float(context["ema2_alpha"])
    ema2_live = alpha * price + (1.0 - alpha) * float(context["ema2"])
    current_trend = 1 if float(context["ema1"]) < ema2_live else -1
    prior_trend = int(context["prior_trend"])
    if current_trend == prior_trend:
        return None

    direction = "LONG" if current_trend == 1 else "SHORT"
    bubble_price = float(context["bubble_price"])
    atr = float(context["signal_atr"])
    buffer = float(context["reference_buffer_atr"])
    reference_sl = bubble_price - buffer * atr if direction == "LONG" else bubble_price + buffer * atr
    filter_length_raw = context.get("ema_filter_length")
    filter_length = normalize_ema_filter_length(filter_length_raw)
    filter_value = context.get("ema_filter_value")
    filter_slow_value = context.get("ema_filter_slow_value")
    filter_passed, filter_ready = _ema_filter_decision(
        direction, float(context["close_price"]), filter_length,
        float(filter_value) if filter_value is not None else None,
        float(filter_slow_value) if filter_slow_value is not None else None,
    )

    result = dict(context)
    result.update({
        "signal": "BLUE" if current_trend == 1 else "RED",
        "direction": direction,
        "trend": "BULLISH" if current_trend == 1 else "BEARISH",
        "detection_time_ms": current_ms,
        "detection_price": price,
        "ema2_live": ema2_live,
        "limit_price": float(context["close_price"]),
        "signal_close_price": float(context["close_price"]),
        "close_price": float(context["close_price"]),
        "limit_price_source": "VISUAL_BUBBLE_CANDLE_CLOSE",
        "reference_sl": float(reference_sl),
        "ema_filter_passed": bool(filter_passed),
        "ema_filter_ready": bool(filter_ready),
        "ema_filter_mode": "DUAL" if isinstance(filter_length, tuple) else ("SINGLE" if filter_length is not None else "OFF"),
        "ema_filter_value": filter_value,
        "ema_filter_slow_value": filter_slow_value,
    })
    return result


def detect_live_visual_signal(
    candles: Any,
    *,
    symbol: str,
    timeframe: str,
    now_ms: int | None = None,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
    reference_buffer_atr: float = REFERENCE_BUFFER_ATR,
    interval_ms: int | None = None,
    ema_filter_length: int | tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """One-shot wrapper for preparing and evaluating a live mode-B kline."""
    context = prepare_live_visual_context(
        candles, symbol=symbol, timeframe=timeframe,
        trend_length=trend_length, atr_length=atr_length,
        reference_buffer_atr=reference_buffer_atr, interval_ms=interval_ms,
        ema_filter_length=ema_filter_length,
    )
    if context is None:
        return None
    # The last supplied row is a live/forming candle; its close is the latest
    # Binance snapshot, used as the current-price input for the intrabar flip.
    _ts, _o, _h, _l, closes, _v = _extract_columns(candles)
    current_ms = int(now_ms if now_ms is not None else context["entry_candle_open_time_ms"])
    return evaluate_live_visual_context(context, float(closes[-1]), now_ms=current_ms)

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
    ema_filter_length: int | tuple[int, int] | None = None,
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
        ema_filter_length=ema_filter_length,
    )


def analyze(
    candles: Any,
    *,
    symbol: str,
    timeframe: str,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
    ema_filter_length: int | tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Full latest-bar analysis response, including no-signal state."""
    indicators = calculate_indicators(
        candles, trend_length=trend_length, atr_length=atr_length,
        ema_filter_length=ema_filter_length,
    )
    tf = str(timeframe).upper()
    if tf not in TIMEFRAME_MS:
        raise ValueError("Timeframe harus H4 atau D1.")
    i = len(indicators["close"]) - 1
    signal = detect_latest_signal(
        candles, symbol=symbol, timeframe=tf,
        trend_length=trend_length, atr_length=atr_length,
        ema_filter_length=ema_filter_length,
    )
    return {
        "strategy": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "symbol": str(symbol).upper(),
        "timeframe": tf,
        "trend_length": int(indicators.get("trend_length", np.asarray([TREND_LENGTH]))[0]),
        "latest_open_time_ms": int(indicators["timestamp"][i]),
        "latest_close_time_ms": int(indicators["timestamp"][i]) + TIMEFRAME_MS[tf],
        "latest_close": float(indicators["close"][i]),
        "ema_filter_length": (list(normalize_ema_filter_length(ema_filter_length)) if isinstance(normalize_ema_filter_length(ema_filter_length), tuple) else normalize_ema_filter_length(ema_filter_length)),
        "ema_filter_value": (float(indicators["ema_filter"][i]) if np.isfinite(indicators["ema_filter"][i]) else None),
        "ema_filter_slow_value": (float(indicators["ema_filter_slow"][i]) if np.isfinite(indicators["ema_filter_slow"][i]) else None),
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
    "STRATEGY_NAME", "STRATEGY_VERSION", "TREND_LENGTH", "DEFAULT_TREND_LENGTH",
    "EMA_TIMEFRAME", "SIGNAL_TIMEFRAME", "SUPERTREND_ATR_LENGTH", "SUPERTREND_FACTOR", "RR_TARGET", "detect_backtest_signal",
    "MIN_TREND_LENGTH", "MAX_TREND_LENGTH", "normalize_trend_length", "ATR_LENGTH",
    "REFERENCE_BUFFER_ATR", "TIMEFRAME_MS", "TIMEFRAME_INTERVALS", "SUPPORTED_TIMEFRAMES", "VolumaticSignal",
    "SUPPORTED_EMA_FILTERS", "MIN_EMA_FILTER_LENGTH", "MAX_EMA_FILTER_LENGTH", "DEFAULT_EMA_FILTER_LENGTH", "normalize_ema_filter_length", "format_ema_filter_length", "ema_filter_lengths",
    "calculate_indicators", "detect_latest_signal", "detect_signals_since", "signals_since", "analyze",
    "prepare_live_visual_context", "evaluate_live_visual_context", "detect_live_visual_signal",
    "reference_risk", "trade_result_r",
]


# ---------------------------------------------------------------------------
# Live multi-timeframe strategy matching backtest.py
# ---------------------------------------------------------------------------

def _backtest_smooth(values: np.ndarray, length: int, alpha: float) -> np.ndarray:
    """Backtest.py EMA/RMA seed: SMA of first `length` valid values."""
    x = np.asarray(values, dtype=np.float64)
    out = np.full(len(x), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(x))
    if valid.size == 0 or len(x) - int(valid[0]) < int(length):
        return out
    start = int(valid[0])
    first = start + int(length) - 1
    seed = x[start:start + int(length)]
    if len(seed) != int(length) or not np.all(np.isfinite(seed)):
        return out
    previous = float(np.mean(seed))
    out[first] = previous
    for i in range(first + 1, len(x)):
        value = float(x[i])
        if not np.isfinite(value):
            previous = np.nan
            continue
        if not np.isfinite(previous):
            # This branch is only relevant for non-contiguous external inputs.
            window = x[max(start, i - int(length) + 1):i + 1]
            if len(window) == int(length) and np.all(np.isfinite(window)):
                previous = float(np.mean(window))
                out[i] = previous
            continue
        previous += float(alpha) * (value - previous)
        out[i] = previous
    return out


def _backtest_volumatic_events(close: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return event(+1 BLUE/-1 RED), EMA(SWMA), EMA(close), exact backtest formula."""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    swma = np.full(n, np.nan, dtype=np.float64)
    if n >= 4:
        swma[3:] = (
            close[:-3] + 2.0 * close[1:-2] + 2.0 * close[2:-1] + close[3:]
        ) / 6.0
    alpha = 2.0 / (float(length) + 1.0)
    ema1 = _backtest_smooth(swma, length, alpha)
    ema2 = _backtest_smooth(close, length, alpha)
    valid = np.zeros(n, dtype=bool)
    trend = np.zeros(n, dtype=bool)
    if n > 1:
        valid[1:] = np.isfinite(ema1[:-1]) & np.isfinite(ema2[1:])
        trend[1:] = ema1[:-1] < ema2[1:]
        trend &= valid
    flip = np.zeros(n, dtype=bool)
    if n > 1:
        flip[1:] = valid[1:] & valid[:-1] & (trend[1:] != trend[:-1])
    events = np.zeros(n, dtype=np.int8)
    events[flip & trend] = 1
    events[flip & ~trend] = -1
    return events, ema1, ema2


def _backtest_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    n = len(close)
    if not n:
        return np.asarray([], dtype=np.float64)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = float(high[0] - low[0])
    if n > 1:
        tr[1:] = np.maximum.reduce((
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ))
    return _backtest_smooth(tr, length, 1.0 / float(length))


def _backtest_supertrend(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    length: int = SUPERTREND_ATR_LENGTH,
    factor: float = SUPERTREND_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """TradingView-style Supertrend line and direction, seeded like backtest.py."""
    n = len(close)
    line = np.full(n, np.nan, dtype=np.float64)
    direction = np.zeros(n, dtype=np.int8)
    if n == 0:
        return line, direction
    atr = _backtest_atr(high, low, close, length)
    mid = (high + low) / 2.0
    basic_upper = mid + factor * atr
    basic_lower = mid - factor * atr
    prev_upper = prev_lower = prev_line = None
    for i in range(n):
        if not np.isfinite(atr[i]):
            continue
        if prev_upper is None:
            prev_upper = float(basic_upper[i])
            prev_lower = float(basic_lower[i])
            prev_line = prev_upper
            line[i] = prev_line
            direction[i] = 1  # initial line is upper/bearish in TradingView convention
            continue
        prev_close = float(close[i - 1])
        final_upper = (float(basic_upper[i]) if
                       (basic_upper[i] < prev_upper or prev_close > prev_upper)
                       else prev_upper)
        final_lower = (float(basic_lower[i]) if
                       (basic_lower[i] > prev_lower or prev_close < prev_lower)
                       else prev_lower)
        if prev_line == prev_upper:
            up = float(close[i]) > final_upper
        else:
            up = not (float(close[i]) < final_lower)
        prev_line = final_lower if up else final_upper
        prev_upper, prev_lower = final_upper, final_lower
        line[i] = prev_line
        direction[i] = -1 if up else 1
    return line, direction


def _ema_pair_from_config(
    h4_close: np.ndarray, ema_config: int | tuple[int, int] | None,
) -> tuple[float | None, float | None, bool, bool]:
    """Return filter fast/single, slow, ready; computed on closed H4 data."""
    selected = normalize_ema_filter_length(ema_config)
    if selected is None:
        return None, None, True, True
    if isinstance(selected, tuple):
        fast = _backtest_smooth(h4_close, selected[0], 2.0 / (selected[0] + 1.0))
        slow = _backtest_smooth(h4_close, selected[1], 2.0 / (selected[1] + 1.0))
        if not len(h4_close) or not (np.isfinite(fast[-1]) and np.isfinite(slow[-1])):
            return None, None, False, False
        return float(fast[-1]), float(slow[-1]), True, True
    ema = _backtest_smooth(h4_close, selected, 2.0 / (selected + 1.0))
    if not len(h4_close) or not np.isfinite(ema[-1]):
        return None, None, False, True
    return float(ema[-1]), None, True, True


def detect_backtest_signal(
    m15_with_forming: Any,
    h4_closed: Any,
    *,
    symbol: str,
    trend_length: int = DEFAULT_TREND_LENGTH,
    ema_filter_length: int | tuple[int, int] | None = DEFAULT_EMA_FILTER_LENGTH,
    supertrend_atr_length: int = SUPERTREND_ATR_LENGTH,
    supertrend_factor: float = SUPERTREND_FACTOR,
    rr_target: float = RR_TARGET,
) -> dict[str, Any] | None:
    """Detect only the newest just-closed M15 flip and prepare next-open entry.

    `m15_with_forming` must contain closed M15 candles plus exactly one current
    forming M15 row. `h4_closed` must contain only completed H4 candles. The H4
    sample is selected with the same timestamp convention as backtest.py.
    """
    length = normalize_trend_length(trend_length)
    filter_config = normalize_ema_filter_length(ema_filter_length)
    mt, mo, mh, ml, mc, _mv = _extract_columns(m15_with_forming)
    ht, ho, hh, hl, hc, _hv = _extract_columns(h4_closed)
    if len(mt) < max(12, length + 8) or len(ht) < 3:
        return None
    if len(mt) < 2:
        return None
    # Verify that the last M15 row really is the immediately following entry bar.
    signal_open_ms = int(mt[-2])
    signal_close_ms = signal_open_ms + 900_000
    entry_open_ms = int(mt[-1])
    entry_close_ms = entry_open_ms + 900_000
    if entry_open_ms != signal_close_ms:
        return None
    now_ms = int(__import__("time").time() * 1000)
    if not (entry_open_ms <= now_ms < entry_close_ms):
        return None

    closed_close = mc[:-1]
    closed_high = mh[:-1]
    closed_low = ml[:-1]
    events, ema1, ema2 = _backtest_volumatic_events(closed_close, length)
    signal_index = len(events) - 1
    code = int(events[signal_index])
    if code == 0:
        return None

    # H4 EMA filter: latest fully closed H4 candle at this M15 signal close.
    h4_index = int(np.searchsorted(ht, signal_close_ms - 14_400_000, side="right") - 1)
    if h4_index < 0 or h4_index >= len(hc):
        return None
    h4_closes = hc[:h4_index + 1]
    fast_value, slow_value, filter_ready, _ = _ema_pair_from_config(h4_closes, filter_config)
    direction = "LONG" if code == 1 else "SHORT"
    signal_close = float(closed_close[-1])
    h4_reference_close = float(hc[h4_index])
    if filter_config is None:
        filter_passed = True
    elif not filter_ready or fast_value is None:
        filter_passed = False
    elif isinstance(filter_config, tuple):
        filter_passed = bool(fast_value > slow_value) if direction == "LONG" else bool(fast_value < slow_value)
    else:
        filter_passed = bool(h4_reference_close > fast_value) if direction == "LONG" else bool(h4_reference_close < fast_value)

    st_line, st_direction = _backtest_supertrend(
        closed_high, closed_low, closed_close,
        int(supertrend_atr_length), float(supertrend_factor),
    )
    stop = float(st_line[-1]) if len(st_line) else float("nan")
    entry = float(mo[-1])
    if not (np.isfinite(stop) and np.isfinite(entry) and entry > 0):
        return None
    protective = stop < entry if direction == "LONG" else stop > entry
    risk = abs(entry - stop)
    if not protective or not np.isfinite(risk) or risk <= max(1e-12, entry * 1e-12):
        return None
    target = entry + (1 if direction == "LONG" else -1) * float(rr_target) * risk
    if not np.isfinite(target) or target <= 0:
        return None

    # Backtest.py's bubble event occurs on this closed M15 index; EMA1[i-1]
    # is retained only as a visual/debug reference, not as the entry price.
    bubble_price = float(ema1[-2]) if len(ema1) >= 2 and np.isfinite(ema1[-2]) else signal_close
    atr_for_record = _backtest_atr(closed_high, closed_low, closed_close, int(supertrend_atr_length))[-1]
    selected_meta = list(filter_config) if isinstance(filter_config, tuple) else filter_config
    return {
        "strategy": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "signal_mode": "CONFIRMED_M15_NEXT_OPEN",
        "symbol": str(symbol).upper(),
        "timeframe": "M15",
        "ema_timeframe": "H4",
        "signal": "BLUE" if code == 1 else "RED",
        "direction": direction,
        "signal_index": int(signal_index),
        "signal_open_time_ms": signal_open_ms,
        "signal_close_time_ms": signal_close_ms,
        "bubble_visual_time_ms": signal_open_ms,
        "bubble_price": bubble_price,
        "signal_atr": float(atr_for_record) if np.isfinite(atr_for_record) else None,
        "close_price": signal_close,
        "signal_close_price": signal_close,
        "entry_price": entry,
        "limit_price": entry,
        "limit_price_source": "NEXT_M15_OPEN",
        "entry_candle_open_time_ms": entry_open_ms,
        "entry_candle_close_time_ms": entry_close_ms,
        "reference_sl": stop,
        "stop_loss": stop,
        "take_profit": target,
        "initial_risk": risk,
        "rr_target": float(rr_target),
        "supertrend_line": stop,
        "supertrend_direction": int(st_direction[-1]) if len(st_direction) else 0,
        "supertrend_atr_length": int(supertrend_atr_length),
        "supertrend_factor": float(supertrend_factor),
        "ema_filter_length": selected_meta,
        "ema_filter_mode": "DUAL" if isinstance(filter_config, tuple) else ("SINGLE" if filter_config is not None else "OFF"),
        "ema_filter_value": fast_value,
        "ema_filter_slow_value": slow_value,
        "ema_reference_close": h4_reference_close,
        "ema_filter_passed": bool(filter_passed),
        "ema_filter_ready": bool(filter_ready),
        "ema_h4_open_time_ms": int(ht[h4_index]),
        "trend_length": length,
        "repaint_risk": False,
    }
