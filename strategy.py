#!/usr/bin/env python3
"""Volumatic Trend [ChartPrime] bubble engine.

This module is deliberately pure: it does not access Binance, Telegram, disk,
WebSocket, or exchange orders. main.py supplies closed history plus the current forming candle for mode B.

Trading rules implemented by main.py:
- Pine's bubble is visually offset one candle backwards (offset=-1).
- Mode B detects the intrabar flip on the immediately following live candle,
  uses the close of the candle carrying the visual bubble as LIMIT price, and
  expires the order at the close of that adjacent live candle.
- This intentionally accepts realtime repaint/lookahead risk inherent in
  observing Pine's offset visual marker before the following candle closes.
- A signal is stale once that adjacent candle has closed; never chase it.

The source indicator attribution/license notice must be retained when the
original Pine-derived implementation is redistributed: © ChartPrime, MPL-2.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

STRATEGY_NAME = "VOLUMATIC_TREND"
STRATEGY_VERSION = "1.4.1"
TREND_LENGTH = 40
ATR_LENGTH = 200
REFERENCE_BUFFER_ATR = 0.30
SUPPORTED_EMA_FILTERS = (50, 100, 200)
DEFAULT_EMA_FILTER_LENGTH = 100
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
    ema_filter_length: int | None
    ema_filter_value: float | None
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
            "ema_filter_length": self.ema_filter_length,
            "ema_filter_value": self.ema_filter_value,
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


def normalize_ema_filter_length(value: Any) -> int | None:
    """Normalize /EMA selection. None/"off" disables the directional filter."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"off", "none", "0", ""}):
        return None
    try:
        length = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("EMA filter harus 50, 100, 200, atau OFF.") from exc
    if length not in SUPPORTED_EMA_FILTERS:
        raise ValueError("EMA filter hanya mendukung 50, 100, 200, atau OFF.")
    return length


def calculate_indicators(
    candles: Any,
    *,
    trend_length: int = TREND_LENGTH,
    atr_length: int = ATR_LENGTH,
    ema_filter_length: int | None = None,
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
    filter_length = normalize_ema_filter_length(ema_filter_length)
    ema_filter = (
        _ema_tv(closes, filter_length)
        if filter_length is not None
        else np.full(n, np.nan, dtype=np.float64)
    )

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
        "ema_filter_length": np.asarray([filter_length if filter_length is not None else 0], dtype=np.int16),
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
    raw_filter_length = int(indicators.get("ema_filter_length", np.asarray([0]))[0])
    filter_length = raw_filter_length or None
    raw_filter_value = float(indicators["ema_filter"][i]) if filter_length is not None else float("nan")
    filter_ready = filter_length is None or np.isfinite(raw_filter_value)
    if filter_length is None:
        filter_passed = True
        filter_value: float | None = None
    elif not filter_ready:
        # Fail closed for new exposure while preserving the opposite-bubble exit.
        filter_passed = False
        filter_value = None
    else:
        filter_value = raw_filter_value
        filter_passed = (
            signal_close_price > raw_filter_value
            if direction == "LONG"
            else signal_close_price < raw_filter_value
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
        ema_filter_length=filter_length,
        ema_filter_value=filter_value,
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
    ema_filter_length: int | None = None,
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
    ema_filter_length: int | None = None,
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
    ema_filter_length: int | None = None,
) -> dict[str, Any] | None:
    """Precompute the closed-candle state needed for intrabar mode-B checks.

    ``candles`` includes one currently-forming last row. The function computes
    EMA/ATR/filter once; subsequent WebSocket ticks only need a scalar EMA2
    update, avoiding a full historical indicator recalculation per tick.
    """
    tf = str(timeframe).upper()
    if tf not in TIMEFRAME_MS:
        raise ValueError(f"Timeframe tidak didukung: {timeframe}. Gunakan H4 atau D1.")
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

    filter_length_raw = int(indicators.get("ema_filter_length", np.asarray([0]))[0])
    filter_length = filter_length_raw or None
    if filter_length is None:
        filter_value = None
        filter_ready = True
    else:
        raw_filter = float(indicators["ema_filter"][bubble_index])
        filter_ready = bool(np.isfinite(raw_filter))
        filter_value = raw_filter if filter_ready else None

    return {
        "strategy": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "symbol": str(symbol).upper(),
        "timeframe": tf,
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
        "ema_filter_value": filter_value,
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
    filter_length = context.get("ema_filter_length")
    filter_value = context.get("ema_filter_value")
    filter_ready = bool(context.get("ema_filter_ready", True))
    if filter_length is None:
        filter_passed = True
    elif not filter_ready or filter_value is None:
        filter_passed = False
    else:
        filter_passed = (
            float(context["close_price"]) > float(filter_value)
            if direction == "LONG"
            else float(context["close_price"]) < float(filter_value)
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
        "ema_filter_ready": filter_ready,
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
    ema_filter_length: int | None = None,
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
    ema_filter_length: int | None = None,
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
    ema_filter_length: int | None = None,
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
        "latest_open_time_ms": int(indicators["timestamp"][i]),
        "latest_close_time_ms": int(indicators["timestamp"][i]) + TIMEFRAME_MS[tf],
        "latest_close": float(indicators["close"][i]),
        "ema_filter_length": normalize_ema_filter_length(ema_filter_length),
        "ema_filter_value": (float(indicators["ema_filter"][i]) if np.isfinite(indicators["ema_filter"][i]) else None),
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
    "SUPPORTED_EMA_FILTERS", "DEFAULT_EMA_FILTER_LENGTH", "normalize_ema_filter_length",
    "calculate_indicators", "detect_latest_signal", "detect_signals_since", "signals_since", "analyze",
    "prepare_live_visual_context", "evaluate_live_visual_context", "detect_live_visual_signal",
    "reference_risk", "trade_result_r",
]
