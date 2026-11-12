#!/usr/bin/env python3
from __future__ import annotations

"""Volumatic Trend M15 + H4 EMA + M15 Supertrend + H1 inverse EMA exit engine.

The lifecycle functions on_start(), handle_update(), and on_stop() remain
available to an external launcher. Running this file directly also starts a
local Telegram Bot API long-polling adapter, so no separate try.py is needed.

The strategy combines confirmed M15 Volumatic bubble flips, the latest
closed H4 EMA direction filter, and M15 Supertrend SL. TP starts at the
configurable RR_MIN, then extends to the furthest unmitigated Volume Liquidity
Trend pool no farther than the configurable RR_MAX.
Entries are MARKET-on-touch: a signal stores its entry level locally, the
market WebSocket watches it, and one MARKET order is sent when price touches
the level; closePosition TP/SL are installed right after the fill.
The main process owns Binance REST/private execution, WebSocket prices,
Telegram, PAPER/REAL ledger, scanner schedule, logging and GitHub history.
"""

import asyncio
import base64
import contextvars
import ctypes
import gc
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN, ROUND_UP
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

import requests
import numpy as np
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed

try:
    from websockets.asyncio.client import connect as ws_connect
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Dependency 'websockets' belum terpasang. Tambahkan "
        "'websockets>=17,<18' ke requirements.txt lalu redeploy."
    ) from exc
except ImportError:
    try:
        from websockets import connect as ws_connect
    except ImportError as exc:
        raise RuntimeError("Library websockets tidak kompatibel; gunakan websockets>=17,<18.") from exc

import strategy as volumatic

# Fail fast on an incompatible/stale strategy module. try.py installs the exact
# STRATEGY_FILE as sys.modules["strategy"] before executing this engine.
if not isinstance(getattr(volumatic, "TIMEFRAME_MS", None), dict) or not {"M15", "H4"}.issubset(volumatic.TIMEFRAME_MS):
    raise RuntimeError("strategy module mismatch: TIMEFRAME_MS M15/H4 tidak tersedia. Pastikan try.py dan STRATEGY_FILE satu versi.")
if not isinstance(getattr(volumatic, "TIMEFRAME_INTERVALS", None), dict) or not {"M15", "H4"}.issubset(volumatic.TIMEFRAME_INTERVALS):
    raise RuntimeError("strategy module mismatch: TIMEFRAME_INTERVALS M15/H4 tidak tersedia.")
if str(getattr(volumatic, "STRATEGY_VERSION", "")) != "2.4.0":
    raise RuntimeError(
        "strategy module mismatch: dibutuhkan STRATEGY_VERSION 2.4.0 "
        "dengan RR_MIN/RR_MAX configurable dan Volume Liquidity TP. Sinkronkan main.py dan strategy.py."
    )
if (
    not callable(getattr(volumatic, "normalize_trend_length", None))
    or not hasattr(volumatic, "DEFAULT_TREND_LENGTH")
    or int(getattr(volumatic, "DEFAULT_TREND_LENGTH", 0)) != 100
    or not callable(getattr(volumatic, "normalize_supertrend_atr_length", None))
    or not callable(getattr(volumatic, "normalize_supertrend_factor", None))
    or not callable(getattr(volumatic, "normalize_rr_value", None))
    or not callable(getattr(volumatic, "normalize_rr_range", None))
):
    raise RuntimeError(
        "strategy module mismatch: API /length, SuperTrend ATR/Factor, atau RR_MIN/RR_MAX tidak tersedia. "
        "Sinkronkan main.py dan strategy.py dengan versi Volumatic configurable SuperTrend."
    )

# ---------------------------------------------------------------------------
# Environment and runtime configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / "trades.env")
load_dotenv(BASE_DIR / ".env")

ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0") or "0")
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()
MAIN_FILE = "main.py"  # canonical runtime/source name; descriptive names are download labels only
BINANCE_API_KEY = (os.getenv("BINANCE_API_KEY") or "").strip()
BINANCE_API_SECRET = (os.getenv("BINANCE_API_SECRET") or "").strip()
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()

TIMEZONE_NAME = "Asia/Jakarta"
TZ = ZoneInfo(TIMEZONE_NAME)
BINANCE_REST_BASE = "https://fapi.binance.com"
BINANCE_REAL_REST_BASE = "https://fapi.binance.com"
BINANCE_WS_BASE = "wss://fstream.binance.com/market/stream"
# Binance migrated USD-M streams to separated public/market/private routes in 2026.
BINANCE_PRIVATE_WS_BASE = "wss://fstream.binance.com/private/ws"
BINANCE_WS_API_BASE = "wss://ws-fapi.binance.com/ws-fapi/v1"
# Optional authenticated HTTP(S) proxy for Binance only. The proxy is applied
# to USD-M REST requests and all Binance WebSocket connections; Telegram and
# GitHub keep their normal network route.
BINANCE_PROXY_URL = (os.getenv("BINANCE_PROXY_URL") or "").strip()
if BINANCE_PROXY_URL:
    _binance_proxy_parts = urlsplit(BINANCE_PROXY_URL)
    if _binance_proxy_parts.scheme not in {"http", "https"} or not _binance_proxy_parts.hostname:
        raise RuntimeError("BINANCE_PROXY_URL harus berupa URL proxy http:// atau https:// yang valid.")
    BINANCE_REQUEST_PROXIES = {"http": BINANCE_PROXY_URL, "https": BINANCE_PROXY_URL}
    BINANCE_WS_PROXY_KWARGS = {"proxy": BINANCE_PROXY_URL}
else:
    BINANCE_REQUEST_PROXIES = None
    BINANCE_WS_PROXY_KWARGS = {}
# Cooldown markers are scoped to the actual configured egress identity. Set
# BINANCE_EGRESS_ID to the proxy's public exit IP when known; otherwise derive
# a stable identity from the proxy endpoint (or "direct" without a proxy).
BINANCE_EGRESS_ID = (os.getenv("BINANCE_EGRESS_ID") or "").strip()
if not BINANCE_EGRESS_ID:
    if BINANCE_PROXY_URL:
        BINANCE_EGRESS_ID = f"proxy:{_binance_proxy_parts.hostname}:{_binance_proxy_parts.port or 0}"
    else:
        BINANCE_EGRESS_ID = "direct"
BINANCE_TRADING_MODE = "live"
REAL_EXECUTION_ENABLED = (os.getenv("REAL_EXECUTION_ENABLED", "1") or "1").strip().lower() not in {
    "0", "false", "off", "no", "disabled"
}
GITHUB_API = "https://api.github.com"
# IP cooldown markers are machine/egress-IP specific. Keep them local by default;
# enable GitHub sharing only when the process restarts behind the same public IP.
BINANCE_COOLDOWN_GITHUB_SYNC = (os.getenv("BINANCE_COOLDOWN_GITHUB_SYNC", "0") or "0").strip().lower() in {
    "1", "true", "yes", "on", "enabled"
}

HISTORY_TRADES_PATH = "data/trades.json"
HISTORY_EVENTS_PATH = "data/events.jsonl"
HISTORY_MARKDOWN_PATH = "data/trade_history.md"
NOTES_PATH = "data/notes.json"
SETUPS_PATH = "data/setups.json"
RUNTIME_SETTINGS_PATH = "data/runtime_settings.json"
REAL_GUARD_PATH = "data/real_equity_guard.json"
DEFERRED_COMMANDS_PATH = "data/deferred_commands.json"
BINANCE_COOLDOWN_PATH = "data/binance_cooldown_state.json"
SIGNAL_DEDUPE_PATH = "data/volumatic_signal_dedupe.json"
SIGNAL_DEDUPE_DB_NAME = "volumatic_signal_dedupe.sqlite3"

BINANCE_USER_STREAM_KEEPALIVE_SECONDS = 45 * 60
BINANCE_USER_STREAM_RETRY_SECONDS = 30.0
BINANCE_RECOVERY_NOTICE_INTERVAL_SECONDS = 60.0

HTTP_REQUEST_SECONDS = 20
GITHUB_REQUEST_SECONDS = 30
BINANCE_RECV_WINDOW = 5000
# Conservative per-IP pacing. Dynamic throttling below increases this when the
# exchange reports high X-MBX-USED-WEIGHT-1M. A Render egress IP can be shared.
BINANCE_PUBLIC_REQUEST_SPACING_SECONDS = float(os.getenv("BINANCE_PUBLIC_REQUEST_SPACING_SECONDS", "0.75") or "0.75")
BINANCE_RATE_LIMIT_FALLBACK_SECONDS = 60
BINANCE_RATE_LIMIT_SAFETY_SECONDS = 60
BINANCE_BAN_418_SAFETY_SECONDS = 600  # extra quiet time after an IP ban (418); retrying early escalates the ban


def rate_limit_safety(status: Any) -> int:
    return BINANCE_BAN_418_SAFETY_SECONDS if int(status or 0) == 418 else BINANCE_RATE_LIMIT_SAFETY_SECONDS
BINANCE_RATE_LIMIT_PRIORITY_SAFETY_SECONDS = 60
BINANCE_BAN_FALLBACK_SECONDS = 120
REAL_PENDING_POLL_SECONDS = 5.0
REAL_RECONCILE_INTERVAL_SECONDS = 20.0
REAL_RECONCILE_WS_COVERED_SECONDS = 300.0  # private WS (ACCOUNT_UPDATE/ALGO_UPDATE) already reports fills/closes
H1_MS = 3_600_000
EMA_EXIT_INIT_RETRY_SECONDS = 30.0
EMA_EXIT_RECONCILE_SECONDS = 20.0
EMA_EXIT_UNKNOWN_RECHECK_SECONDS = 120.0
ENTRY_ORDER_TTL_SECONDS = 60 * 60  # local entry watch (and any legacy exchange LIMIT) expires 1 hour after creation
REAL_POSITION_SETTLE_SECONDS = 15.0  # grace window after a WebSocket fill before REST position reconciliation
POSITION_CACHE_MAX_AGE_SECONDS = 30.0
BALANCE_STALE_SECONDS = 60.0
WS_RECONNECT_MIN = 2
WS_RECONNECT_MAX = 60
PRICE_STALE_SECONDS = 15
HISTORY_REFRESH_SECONDS = 30
AUTOSTOP_CHECK_SECONDS = 5.0
AUTOSTOP_BALANCE_REFRESH_SECONDS = 60.0
AUTOSTOP_PERSIST_SECONDS = 60.0
MEMORY_SOFT_LIMIT_MB = float(os.getenv("MEMORY_SOFT_LIMIT_MB", "360") or "360")

DEFAULT_MARGIN_USDT = Decimal(os.getenv("DEFAULT_MARGIN_USDT", "0.5") or "0.5")
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "10") or "10")
SIM_START_BALANCE = Decimal(os.getenv("SIM_START_BALANCE", "100") or "100")
CONFIG_AUTOSTOP = (os.getenv("AUTOSTOP_PERCENT") or "").strip()
# Fixed-margin sizing must stay close to the requested margin. Binance LOT_SIZE
# rounding that cannot fit inside this +/-5% band is rejected, never rounded up
# into a materially larger position. Market-fill slippage can still move final margin.
REAL_NOTIONAL_MIN_RATIO = Decimal("0.95")
REAL_NOTIONAL_MAX_RATIO = Decimal("1.05")
REAL_FEE_BUFFER = Decimal("0.001")


MARKET_ENTRY_PROTECTION_ATTEMPTS = 3
MARKET_ENTRY_MAX_PRESEND_ATTEMPTS = 3
LEVERAGE_CACHE_SECONDS = 900.0
QTY_FILTER_CODES = (-1013, -1111, -4004, -4005, -4164)
MAX_ACTIVE_TRADES = max(1, int(os.getenv("MAX_ACTIVE_TRADES", "100") or "100"))
SCAN_MAX_PAIRS_PER_CYCLE = 50
UNIVERSE_REFRESH_SECONDS = 60 * 60  # refresh symbol/filter metadata hourly; retain cache during recovery
SCAN_PAIR_DELAY_SECONDS = float(os.getenv("SCAN_PAIR_DELAY_SECONDS", "0.15") or "0.15")
SCAN_POLL_SECONDS = 2.0
# One-time scanner activation slots in Asia/Jakarta (WIB). 12:00 is an
# additional user-requested slot; the other six align with H4 closes.
SCAN_SCHEDULE_SLOTS = (
    ("1", "07:00"),
    ("2", "11:00"),
    ("3", "15:00"),
    ("4", "19:00"),
    ("5", "23:00"),
    ("6", "03:00"),
    ("7", "12:00"),
)
SCAN_SCHEDULE_BY_NUMBER = dict(SCAN_SCHEDULE_SLOTS)
MAINTENANCE_POLL_SECONDS = 2.0
# Base request remains 499 candles. _kline_history_limit() expands history only
# when /length is raised, capped below Binance USD-M per-request maximum.
KLINE_HISTORY = max(400, min(499, int(os.getenv("VOLUMATIC_KLINE_HISTORY", "499") or "499")))
EXCLUDED_SYMBOLS = {
    x.strip().upper() for x in (os.getenv("EXCLUDED_SYMBOLS", "BTCUSDT,ETHUSDT") or "").split(",") if x.strip()
}
BTC_EMA_FILTER_SYMBOL = "BTCUSDT"  # TradingView BTCUSDT.P USD-M perpetual reference

REAL_PRIORITY: contextvars.ContextVar[bool] = contextvars.ContextVar("real_priority", default=False)
REAL_EARLY: contextvars.ContextVar[bool] = contextvars.ContextVar("real_early", default=False)

if not ALLOWED_USER_ID:
    raise RuntimeError("ALLOWED_USER_ID belum diset.")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN belum diset.")
if not REPO_NAME or "/" not in REPO_NAME:
    raise RuntimeError("REPO_NAME belum diset atau formatnya tidak valid.")

# ---------------------------------------------------------------------------
# Logging and common helpers
# ---------------------------------------------------------------------------
log = logging.getLogger("main.trading_engine")


def ema_filter_label(value: Any) -> str:
    """Stable label for single, dual, or disabled EMA filter mode."""
    try:
        return volumatic.format_ema_filter_length(value)
    except Exception:
        return "EMA CONFIG INVALID"


def ema_filter_values_label(value: Any, fast_value: Any, slow_value: Any = None) -> str:
    """Render the EMA values calculated from the latest fully closed H4 candle."""
    try:
        selected = volumatic.normalize_ema_filter_length(value)
    except Exception:
        return "EMA CONFIG INVALID"
    if selected is None:
        return "OFF"
    if isinstance(selected, tuple):
        fast_text = fmt_price(Decimal(str(fast_value))) if fast_value not in (None, "") else "WARMING UP"
        slow_text = fmt_price(Decimal(str(slow_value))) if slow_value not in (None, "") else "WARMING UP"
        return f"EMA{selected[0]}={fast_text} · EMA{selected[1]}={slow_text}"
    one_text = fmt_price(Decimal(str(fast_value))) if fast_value not in (None, "") else "WARMING UP"
    return f"EMA{selected}={one_text}"
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_local() -> datetime:
    return now_utc().astimezone(TZ)


def iso_utc(value: datetime | None = None) -> str:
    dt = value or now_utc()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def format_wib(value: datetime | None) -> str:
    if value is None:
        return "-"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(TZ).strftime("%d-%m-%Y %H:%M WIB")


def decimal_to_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def parse_decimal(value: Any) -> Decimal:
    try:
        number = Decimal(str(value).strip().replace(",", "."))
    except (InvalidOperation, AttributeError, TypeError) as exc:
        raise ValueError(f"Nilai desimal tidak valid: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError("Nilai harus berupa angka positif.")
    return number


def parse_signed_decimal(value: Any) -> Decimal:
    try:
        number = Decimal(str(value).strip().replace(",", "."))
    except (InvalidOperation, AttributeError, TypeError) as exc:
        raise ValueError(f"Angka bertanda tidak valid: {value!r}") from exc
    if not number.is_finite():
        raise ValueError("Angka harus finite.")
    return number


def normalize_symbol(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").upper().replace("/", "").replace("-", "").replace("_", ""))


def generate_trade_id(pair: str) -> str:
    return f"{pair}-{now_local():%Y%m%d-%H%M%S}-{uuid4().hex[:6].upper()}"


def generate_session_id() -> str:
    return f"{now_local():%Y%m%d-%H%M%S}-{uuid4().hex[:4].upper()}"


def duration_text(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    days, total = divmod(total, 86400)
    hours, total = divmod(total, 3600)
    minutes, secs = divmod(total, 60)
    parts = []
    if days: parts.append(f"{days}h")
    if hours: parts.append(f"{hours}j")
    if minutes: parts.append(f"{minutes}m")
    if secs or not parts: parts.append(f"{secs}s")
    return " ".join(parts)


def fmt_price(value: Decimal | None) -> str:
    if value is None:
        return "-"
    try:
        return decimal_to_str(value.quantize(Decimal("0.0000000001"))) or "0"
    except InvalidOperation:
        return decimal_to_str(value) or "0"


def fmt_num(value: Any, digits: int = 2) -> str:
    try:
        return str(Decimal(str(value)).quantize(Decimal("1").scaleb(-digits)))
    except (InvalidOperation, ValueError, TypeError):
        return "-" if value is None else str(value)


def directional_reward_r(
    direction: str, entry: Decimal, target: Decimal, risk: Decimal
) -> Decimal | None:
    """Calculate the direction-adjusted reward multiple for a candidate TP."""
    if risk is None or risk <= 0:
        return None
    side = Decimal("1") if str(direction).upper() == "LONG" else (
        Decimal("-1") if str(direction).upper() == "SHORT" else Decimal("0")
    )
    if side == 0:
        return None
    value = ((target - entry) * side) / risk
    return value if value.is_finite() else None


def is_valid_liquidity_reward_r(
    value: Decimal | None,
    rr_min: Decimal = Decimal("3"),
    rr_max: Decimal = Decimal("4"),
) -> bool:
    """A liquidity TP must remain inside the trade's configured RR band."""
    if value is None:
        return False
    try:
        low, high = Decimal(str(rr_min)), Decimal(str(rr_max))
        return value.is_finite() and low.is_finite() and high.is_finite() and low <= value <= high
    except (InvalidOperation, TypeError, ValueError):
        return False


def as_decimal_number(value: Any, field_name: str = "nilai") -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} bukan angka yang valid.")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} bukan angka yang valid: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} harus finite.")
    return result


def quantized_price(value: Decimal, tick_size: Decimal) -> Decimal:
    value, tick_size = as_decimal_number(value, "harga"), as_decimal_number(tick_size, "tick size")
    return value if tick_size <= 0 else (value / tick_size).to_integral_value(rounding=ROUND_DOWN) * tick_size


def quantized_price_ceiling(value: Decimal, tick_size: Decimal) -> Decimal:
    value, tick_size = as_decimal_number(value, "harga"), as_decimal_number(tick_size, "tick size")
    return value if tick_size <= 0 else (value / tick_size).to_integral_value(rounding=ROUND_UP) * tick_size


def round_step_down(value: Decimal, step: Decimal) -> Decimal:
    return value if step <= 0 else (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def round_step_up(value: Decimal, step: Decimal) -> Decimal:
    return value if step <= 0 else (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def compute_minimum_order(
    entry: Decimal,
    leverage: int,
    filters: dict[str, Any],
    available_balance: Decimal | None = None,
    leverage_max_notional: Decimal | None = None,
) -> tuple[Decimal | None, Decimal | None, Decimal | None, str]:
    """Find the smallest exchange-valid LIMIT quantity and its initial margin.

    The minimum is derived from Binance LOT_SIZE (minQty/stepSize) and
    MIN_NOTIONAL/NOTIONAL filters. Leverage changes margin, not minimum
    order notional. No arbitrary percentage buffer is added to the quantity.
    """
    try:
        price = Decimal(str(entry))
        lev = Decimal(str(leverage))
        step = Decimal(str(filters.get("step_size") or "0"))
        min_qty = Decimal(str(filters.get("min_qty") or "0"))
        max_qty = Decimal(str(filters.get("max_qty") or "0"))
        min_notional = Decimal(str(filters.get("min_notional") or "0"))
        max_notional = Decimal(str(filters.get("max_notional") or "0"))
        min_price = Decimal(str(filters.get("min_price") or "0"))
        max_price = Decimal(str(filters.get("max_price") or "0"))
    except (InvalidOperation, TypeError, ValueError):
        return None, None, None, "INVALID_FILTER_NUMERIC"

    values = (price, lev, step, min_qty, max_qty, min_notional, max_notional, min_price, max_price)
    if not all(v.is_finite() for v in values):
        return None, None, None, "NON_FINITE_FILTER"
    if price <= 0 or lev < 1 or lev != lev.to_integral_value() or step <= 0:
        return None, None, None, "INVALID_INPUT_OR_STEP_SIZE"
    if min_price > 0 and price < min_price:
        return None, None, None, "PRICE_BELOW_EXCHANGE_MINIMUM"
    if max_price > 0 and price > max_price:
        return None, None, None, "PRICE_ABOVE_EXCHANGE_MAXIMUM"

    # The quantity must satisfy both LOT_SIZE and notional minimums. Round UP
    # to the next exchange step; rounding down can create an invalid order.
    required_qty = max(min_qty, step)
    if min_notional > 0:
        required_qty = max(required_qty, min_notional / price)
    quantity = round_step_up(required_qty, step)
    notional = quantity * price

    # Protect against decimal boundary effects and ensure all minimums pass.
    while min_notional > 0 and notional < min_notional:
        quantity += step
        notional = quantity * price

    caps = [v for v in (max_notional, leverage_max_notional) if v is not None and v > 0]
    effective_max_notional = min(caps) if caps else Decimal("0")
    if max_qty > 0 and quantity > max_qty:
        return None, None, None, "MINIMUM_QUANTITY_EXCEEDS_MAX_QTY"
    if effective_max_notional > 0 and notional > effective_max_notional:
        return None, None, None, "MINIMUM_NOTIONAL_EXCEEDS_LEVERAGE_OR_SYMBOL_CAP"

    margin = notional / lev
    if available_balance is not None:
        balance = Decimal(str(available_balance))
        if not balance.is_finite() or balance < 0:
            return None, None, None, "INVALID_AVAILABLE_BALANCE"
        if margin + notional * REAL_FEE_BUFFER > balance:
            return None, None, None, "INSUFFICIENT_AVAILABLE_BALANCE"
    return quantity, notional, margin, "OK"


def format_qty(qty: Decimal) -> str:
    return format(qty.normalize(), "f")


def parse_max_notional(response: Any) -> Decimal | None:
    raw = (response or {}).get("maxNotionalValue") if isinstance(response, dict) else None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, TypeError):
        return None
    return value if value.is_finite() and value > 0 else None


def compute_quantity(entry: Decimal, margin: Decimal, leverage: int, filters: dict[str, Any], available_balance: Decimal | None = None) -> tuple[Decimal | None, str]:
    try:
        price, margin_d = Decimal(str(entry)), Decimal(str(margin))
        lev = Decimal(str(leverage))
    except (InvalidOperation, ValueError, TypeError):
        return None, "INVALID_NUMERIC_INPUT"
    if not all(x.is_finite() for x in (price, margin_d, lev)) or price <= 0 or margin_d <= 0 or lev < 1 or lev != lev.to_integral_value():
        return None, "INVALID_INPUT_RANGE"
    step = Decimal(str(filters.get("step_size") or "0"))
    min_qty = Decimal(str(filters.get("min_qty") or "0"))
    max_qty = Decimal(str(filters.get("max_qty") or "0"))
    min_notional = Decimal(str(filters.get("min_notional") or "0"))
    max_notional = Decimal(str(filters.get("max_notional") or "0"))
    if step <= 0:
        return None, "INVALID_STEP_SIZE"
    target = margin_d * lev
    lo = max(target * REAL_NOTIONAL_MIN_RATIO / price, min_qty)
    hi = target * REAL_NOTIONAL_MAX_RATIO / price
    if max_qty > 0: hi = min(hi, max_qty)
    if min_notional > 0: lo = max(lo, min_notional / price)
    if max_notional > 0: hi = min(hi, max_notional / price)
    lower = ((lo / step).to_integral_value(rounding=ROUND_CEILING)) * step
    upper = ((hi / step).to_integral_value(rounding=ROUND_DOWN)) * step
    if lower <= 0 or lower > upper:
        return None, "MARGIN_DEVIATION_OUT_OF_BOUND"
    target_qty = target / price
    candidates = {lower, upper, round_step_down(target_qty, step), ((target_qty / step).to_integral_value(rounding=ROUND_CEILING)) * step}
    valid = [q for q in candidates if lower <= q <= upper and q > 0 and (max_qty <= 0 or q <= max_qty)]
    if not valid:
        return None, "NO_VALID_QUANTITY"
    qty = min(valid, key=lambda q: abs(q * price - target))
    notional = qty * price
    if not (target * REAL_NOTIONAL_MIN_RATIO <= notional <= target * REAL_NOTIONAL_MAX_RATIO):
        return None, "MARGIN_DEVIATION_OUT_OF_BOUND"
    required_margin = notional / lev
    if available_balance is not None and required_margin + notional * REAL_FEE_BUFFER > available_balance:
        return None, "INSUFFICIENT_AVAILABLE_BALANCE"
    return qty, "OK"


def rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class TelegramErrorHandler(logging.Handler):
    def __init__(self, engine: "TradingEngine") -> None:
        super().__init__(level=logging.WARNING)
        self.engine = engine
        self.loop: asyncio.AbstractEventLoop | None = None

    def attach(self) -> None:
        try: self.loop = asyncio.get_running_loop()
        except RuntimeError: self.loop = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.loop is None or "Gagal mengirim Telegram" in record.getMessage():
            return
        try:
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self.engine._send_log_error_to_telegram(record)))
        except Exception:
            pass


@dataclass(slots=True)
class SymbolMeta:
    symbol: str
    status: str
    contract_type: str
    quote_asset: str
    tick_size: Decimal
    min_price: Decimal = Decimal("0")
    max_price: Decimal = Decimal("0")
    min_qty: Decimal = Decimal("0")
    max_qty: Decimal = Decimal("0")
    step_size: Decimal = Decimal("0")
    min_notional: Decimal = Decimal("0")
    max_notional: Decimal = Decimal("0")
    market_max_qty: Decimal = Decimal("0")
    quantity_precision: int = 0
    price_precision: int = 0


@dataclass(slots=True)
class PriceSnapshot:
    symbol: str
    price: Decimal
    event_time_ms: int
    received_at: datetime
    source: str = "WEBSOCKET"

    @property
    def age_seconds(self) -> float:
        return max(0.0, (now_utc() - self.received_at).total_seconds())

    @property
    def live(self) -> bool:
        return self.age_seconds <= PRICE_STALE_SECONDS


@dataclass(slots=True)
class Trade:
    trade_id: str
    session_id: str
    pair: str
    direction: str                    # LONG / SHORT
    timeframe: str
    trend_length: int
    signal: str                       # BLUE / RED
    signal_time: datetime
    bubble_price: Decimal
    signal_atr: Decimal | None
    reference_sl: Decimal | None      # M15 Supertrend line from confirmed signal candle
    entry: Decimal                    # LIMIT price
    expires_at: datetime              # entry order creation time + one-hour TTL
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    rr_target: Decimal = Decimal("3")  # legacy alias; always equals this trade's RR_MIN
    rr_min: Decimal = Decimal("3")
    rr_max: Decimal = Decimal("4")
    take_profit_source: str = "RR_MIN_BASE"
    take_profit_rr: Decimal = Decimal("3")
    liquidity_pool_price: Decimal | None = None
    supertrend_atr_length: int = volumatic.SUPERTREND_ATR_LENGTH
    supertrend_factor: float = volumatic.SUPERTREND_FACTOR
    sl_algo_id: int | None = None
    tp_algo_id: int | None = None
    sl_client_algo_id: str | None = None
    tp_client_algo_id: str | None = None
    sl_algo_status: str = "NOT_PLACED"
    tp_algo_status: str = "NOT_PLACED"
    protection_state: str = "NOT_REQUIRED"
    protection_error: str | None = None
    protection_warning_sent: bool = False
    protection_exit_reason: str | None = None
    sl_submit_ambiguous: bool = False
    tp_submit_ambiguous: bool = False
    signal_close_price: Decimal | None = None  # close of the M15 signal candle
    ema_filter_length: int | tuple[int, int] | None = 100
    ema_filter_value: Decimal | None = None
    ema_filter_slow_value: Decimal | None = None
    ema_filter_passed: bool | None = None
    # H1 inverse-EMA exit state. EMA levels are seeded from closed H1 candles
    # using the same SMA-seeded recurrence as backtest(10).py.
    ema_exit_state: str = "UNINITIALIZED"
    ema_exit_last_open_ms: int = 0
    ema_exit_last_fast: Decimal | None = None
    ema_exit_last_slow: Decimal | None = None
    ema_exit_trigger_close_ms: int | None = None
    ema_exit_trigger_price: Decimal | None = None
    ema_exit_reason: str | None = None
    ema_exit_client_order_id: str | None = None
    ema_exit_order_id: int | None = None
    ema_exit_order_status: str = "NOT_SUBMITTED"
    ema_exit_attempts: int = 0
    ema_exit_last_error: str | None = None
    status: str = "PENDING"
    created_at: datetime = field(default_factory=now_utc)
    entry_order_created_at: datetime | None = None  # actual local/Binance LIMIT creation timestamp
    entry_window_closes_at: datetime | None = None  # next M15 candle close; deadline to CREATE the LIMIT
    filled_at: datetime | None = None
    closed_at: datetime | None = None
    fill_price: Decimal | None = None
    exit_price: Decimal | None = None
    result: str | None = None
    result_reason: str | None = None
    pnl_percent: Decimal | None = None
    pnl_usdt: Decimal | None = None
    initial_risk: Decimal | None = None
    realized_r: Decimal | None = None
    quantity: Decimal | None = None
    margin_usdt: Decimal = DEFAULT_MARGIN_USDT
    leverage: int = DEFAULT_LEVERAGE
    min_margin_mode: bool = False
    target_notional: Decimal | None = None
    actual_notional: Decimal | None = None
    real_enabled: bool = False
    real_state: str = "SIMULATION"
    position_side: str | None = None
    entry_order_id: int | None = None
    entry_client_order_id: str | None = None
    real_error: str | None = None
    # True when a POST may have reached Binance but its result was not confirmed.
    # Such an order is queried by clientOrderId and NEVER re-POSTed automatically.
    real_submit_ambiguous: bool = False
    last_real_check_monotonic: float = 0.0
    last_real_order_status_check: float = 0.0
    # Entry order may be partially filled while the remaining LIMIT is still open.
    entry_executed_qty: Decimal | None = None
    entry_order_status: str = "UNKNOWN"
    entry_order_terminal: bool = False
    pending_cancel_reason: str | None = None
    last_ws_event_key: str | None = None
    late_fill_during_api_cooldown: bool = False
    # Preserve an already-submitted exchange GTC LIMIT if its candle deadline
    # passed while Binance REST was IP-banned. It remains until fill, reversal,
    # or explicit manual cancellation.
    preserve_gtc_after_cooldown: bool = False
    # Live exchange snapshot adoption is not a local-history restore. It lets a
    # fresh /try session display/manage existing bot-created Binance exposure.
    exchange_adopted: bool = False
    exchange_adoption_source: str | None = None
    # MARKET-on-touch: while waiting, `entry` is only a local watch level and no
    # exchange order exists (real_enabled stays False). real_intent marks a setup
    # that must be executed as a REAL MARKET order when price touches the level.
    real_intent: bool = False
    entry_inflight: bool = False  # runtime-only single-flight guard; never persisted
    entry_triggered_at: datetime | None = None
    entry_trigger_price: Decimal | None = None

    @property
    def entry_side(self) -> str:
        return "BUY" if self.direction == "LONG" else "SELL"

    def to_record(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id, "session_id": self.session_id,
            "pair": self.pair, "direction": self.direction,
            "timeframe": self.timeframe, "trend_length": self.trend_length,
            "signal": self.signal,
            "signal_time": iso_utc(self.signal_time),
            "bubble_price": decimal_to_str(self.bubble_price),
            "signal_atr": decimal_to_str(self.signal_atr),
            "reference_sl": decimal_to_str(self.reference_sl),
            "stop_loss": decimal_to_str(self.stop_loss), "take_profit": decimal_to_str(self.take_profit),
            "rr_target": decimal_to_str(self.rr_target),
            "rr_min": decimal_to_str(self.rr_min),
            "rr_max": decimal_to_str(self.rr_max),
            "take_profit_source": self.take_profit_source,
            "take_profit_rr": decimal_to_str(self.take_profit_rr),
            "liquidity_pool_price": decimal_to_str(self.liquidity_pool_price),
            "supertrend_atr_length": self.supertrend_atr_length,
            "supertrend_factor": self.supertrend_factor,
            "sl_algo_id": self.sl_algo_id, "tp_algo_id": self.tp_algo_id,
            "sl_client_algo_id": self.sl_client_algo_id, "tp_client_algo_id": self.tp_client_algo_id,
            "sl_algo_status": self.sl_algo_status, "tp_algo_status": self.tp_algo_status,
            "protection_state": self.protection_state, "protection_error": self.protection_error,
            "protection_warning_sent": self.protection_warning_sent,
            "protection_exit_reason": self.protection_exit_reason,
            "sl_submit_ambiguous": self.sl_submit_ambiguous, "tp_submit_ambiguous": self.tp_submit_ambiguous,
            "entry": decimal_to_str(self.entry), "signal_close_price": decimal_to_str(self.signal_close_price),
            "ema_filter_length": list(self.ema_filter_length) if isinstance(self.ema_filter_length, tuple) else self.ema_filter_length,
            "ema_filter_value": decimal_to_str(self.ema_filter_value),
            "ema_filter_slow_value": decimal_to_str(self.ema_filter_slow_value),
            "ema_filter_passed": self.ema_filter_passed,
            "ema_exit_state": self.ema_exit_state,
            "ema_exit_last_open_ms": int(self.ema_exit_last_open_ms or 0),
            "ema_exit_last_fast": decimal_to_str(self.ema_exit_last_fast),
            "ema_exit_last_slow": decimal_to_str(self.ema_exit_last_slow),
            "ema_exit_trigger_close_ms": self.ema_exit_trigger_close_ms,
            "ema_exit_trigger_price": decimal_to_str(self.ema_exit_trigger_price),
            "ema_exit_reason": self.ema_exit_reason,
            "ema_exit_client_order_id": self.ema_exit_client_order_id,
            "ema_exit_order_id": self.ema_exit_order_id,
            "ema_exit_order_status": self.ema_exit_order_status,
            "ema_exit_attempts": int(self.ema_exit_attempts or 0),
            "ema_exit_last_error": self.ema_exit_last_error,
            "expires_at": iso_utc(self.expires_at),
            "status": self.status, "created_at": iso_utc(self.created_at),
            "entry_order_created_at": iso_utc(self.entry_order_created_at) if self.entry_order_created_at else None,
            "entry_window_closes_at": iso_utc(self.entry_window_closes_at) if self.entry_window_closes_at else None,
            "filled_at": iso_utc(self.filled_at) if self.filled_at else None,
            "closed_at": iso_utc(self.closed_at) if self.closed_at else None,
            "fill_price": decimal_to_str(self.fill_price), "exit_price": decimal_to_str(self.exit_price),
            "result": self.result, "result_reason": self.result_reason,
            "pnl_percent": decimal_to_str(self.pnl_percent), "pnl_usdt": decimal_to_str(self.pnl_usdt),
            "initial_risk": decimal_to_str(self.initial_risk), "realized_r": decimal_to_str(self.realized_r),
            "quantity": decimal_to_str(self.quantity), "margin_usdt": decimal_to_str(self.margin_usdt),
            "leverage": self.leverage, "min_margin_mode": self.min_margin_mode,
            "target_notional": decimal_to_str(self.target_notional),
            "actual_notional": decimal_to_str(self.actual_notional), "real_enabled": self.real_enabled,
            "real_state": self.real_state, "position_side": self.position_side,
            "entry_order_id": self.entry_order_id, "entry_client_order_id": self.entry_client_order_id,
            "real_error": self.real_error,
            "real_submit_ambiguous": self.real_submit_ambiguous,
            "entry_executed_qty": decimal_to_str(self.entry_executed_qty),
            "entry_order_status": self.entry_order_status,
            "entry_order_terminal": self.entry_order_terminal,
            "pending_cancel_reason": self.pending_cancel_reason,
            "last_ws_event_key": self.last_ws_event_key,
            "late_fill_during_api_cooldown": self.late_fill_during_api_cooldown,
            "preserve_gtc_after_cooldown": self.preserve_gtc_after_cooldown,
            "exchange_adopted": self.exchange_adopted,
            "exchange_adoption_source": self.exchange_adoption_source,
            "real_intent": self.real_intent,
            "entry_triggered_at": iso_utc(self.entry_triggered_at) if self.entry_triggered_at else None,
            "entry_trigger_price": decimal_to_str(self.entry_trigger_price),
            "strategy_name": volumatic.STRATEGY_NAME, "strategy_version": volumatic.STRATEGY_VERSION,
            "entry_reason": f"Confirmed M15 Volumatic bubble + H4 EMA; entry level = next M15 OPEN; TP baseline {self.rr_min}R with farthest unmitigated Volume Liquidity Trend pool extension up to {self.rr_max}R",
            "price_exp_reason": (
                "Unfilled local entry watch expires 1 hour after the signal; no exchange order exists while waiting."
            ),
        }

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> "Trade":
        def dec(key: str, default: Decimal | None = None) -> Decimal | None:
            value = row.get(key)
            if value in (None, ""):
                return default
            try:
                result = Decimal(str(value))
                return result if result.is_finite() else default
            except (InvalidOperation, ValueError):
                return default
        signal_time = parse_iso(row.get("signal_time") or row.get("created_at")) or now_utc()
        try:
            saved_trend_length = volumatic.normalize_trend_length(row.get("trend_length", volumatic.DEFAULT_TREND_LENGTH))
        except (TypeError, ValueError):
            saved_trend_length = volumatic.DEFAULT_TREND_LENGTH
        try:
            saved_supertrend_atr_length = volumatic.normalize_supertrend_atr_length(
                row.get("supertrend_atr_length", volumatic.SUPERTREND_ATR_LENGTH)
            )
        except (TypeError, ValueError):
            saved_supertrend_atr_length = volumatic.SUPERTREND_ATR_LENGTH
        try:
            saved_supertrend_factor = volumatic.normalize_supertrend_factor(
                row.get("supertrend_factor", volumatic.SUPERTREND_FACTOR)
            )
        except (TypeError, ValueError):
            saved_supertrend_factor = volumatic.SUPERTREND_FACTOR
        saved_created_at = parse_iso(row.get("created_at")) or signal_time
        saved_entry_window_closes_at = parse_iso(row.get("entry_window_closes_at"))
        if saved_entry_window_closes_at is None and str(row.get("timeframe") or "M15").upper() == "M15":
            saved_entry_window_closes_at = signal_time + timedelta(milliseconds=volumatic.TIMEFRAME_MS["M15"])
        saved_real_enabled = bool(row.get("real_enabled"))
        saved_real_state = str(row.get("real_state") or ("REAL_PENDING" if saved_real_enabled else "SIMULATION")).upper()
        saved_entry_order_created_at = parse_iso(row.get("entry_order_created_at"))
        # Backward compatibility for records written before entry_order_created_at existed.
        # PREPARED means Binance has not accepted an order yet, so its one-hour clock
        # must not start until the entry order is actually sent.
        if saved_entry_order_created_at is None:
            has_submitted_order = bool(row.get("entry_order_id") or row.get("entry_client_order_id"))
            if not saved_real_enabled or (saved_real_state != "REAL_PREPARED" and has_submitted_order):
                saved_entry_order_created_at = saved_created_at
        saved_expiry = (
            saved_entry_order_created_at + timedelta(seconds=ENTRY_ORDER_TTL_SECONDS)
            if saved_entry_order_created_at else
            (parse_iso(row.get("expires_at")) or saved_created_at + timedelta(seconds=ENTRY_ORDER_TTL_SECONDS))
        )
        return cls(
            trade_id=str(row.get("trade_id") or generate_trade_id(str(row.get("pair") or "UNKNOWN"))),
            session_id=str(row.get("session_id") or "RECOVERED"),
            pair=normalize_symbol(str(row.get("pair") or "")),
            direction=str(row.get("direction") or "LONG").upper(),
            timeframe=str(row.get("timeframe") or "M15").upper(),
            trend_length=saved_trend_length,
            signal=str(row.get("signal") or ("BLUE" if str(row.get("direction")).upper() == "LONG" else "RED")).upper(),
            signal_time=signal_time,
            bubble_price=dec("bubble_price", dec("entry", Decimal("0"))) or Decimal("0"),
            signal_atr=dec("signal_atr"), reference_sl=dec("reference_sl"),
            stop_loss=dec("stop_loss", dec("reference_sl")), take_profit=dec("take_profit"),
            rr_target=dec("rr_target", dec("rr_min", Decimal("3"))) or Decimal("3"),
            rr_min=dec("rr_min", dec("rr_target", Decimal("3"))) or Decimal("3"),
            rr_max=dec("rr_max", Decimal("4")) or Decimal("4"),
            take_profit_source=str(row.get("take_profit_source") or "RR_MIN_BASE").upper(),
            take_profit_rr=dec("take_profit_rr", dec("rr_min", Decimal("3"))) or Decimal("3"),
            liquidity_pool_price=dec("liquidity_pool_price"),
            supertrend_atr_length=saved_supertrend_atr_length,
            supertrend_factor=saved_supertrend_factor,
            sl_algo_id=int(row["sl_algo_id"]) if row.get("sl_algo_id") not in (None, "") else None,
            tp_algo_id=int(row["tp_algo_id"]) if row.get("tp_algo_id") not in (None, "") else None,
            sl_client_algo_id=row.get("sl_client_algo_id"), tp_client_algo_id=row.get("tp_client_algo_id"),
            sl_algo_status=str(row.get("sl_algo_status") or "NOT_PLACED"),
            tp_algo_status=str(row.get("tp_algo_status") or "NOT_PLACED"),
            protection_state=str(row.get("protection_state") or "NOT_REQUIRED"),
            protection_error=row.get("protection_error"),
            protection_warning_sent=bool(row.get("protection_warning_sent", False)),
            protection_exit_reason=row.get("protection_exit_reason"),
            sl_submit_ambiguous=bool(row.get("sl_submit_ambiguous", False)),
            tp_submit_ambiguous=bool(row.get("tp_submit_ambiguous", False)),
            entry=dec("entry", Decimal("0")) or Decimal("0"),
            expires_at=saved_expiry,
            status=str(row.get("status") or "PENDING").upper(),
            created_at=saved_created_at,
            entry_order_created_at=saved_entry_order_created_at,
            entry_window_closes_at=saved_entry_window_closes_at,
            filled_at=parse_iso(row.get("filled_at")), closed_at=parse_iso(row.get("closed_at")),
            fill_price=dec("fill_price"), exit_price=dec("exit_price"),
            result=row.get("result"), result_reason=row.get("result_reason"),
            pnl_percent=dec("pnl_percent"), pnl_usdt=dec("pnl_usdt"),
            initial_risk=dec("initial_risk"), realized_r=dec("realized_r"),
            quantity=dec("quantity"), margin_usdt=dec("margin_usdt", DEFAULT_MARGIN_USDT) or DEFAULT_MARGIN_USDT,
            leverage=int(row.get("leverage") or DEFAULT_LEVERAGE),
            min_margin_mode=bool(row.get("min_margin_mode", False)),
            target_notional=dec("target_notional"),
            actual_notional=dec("actual_notional"), real_enabled=bool(row.get("real_enabled")),
            real_state=str(row.get("real_state") or ("REAL_PENDING" if row.get("real_enabled") else "SIMULATION")),
            position_side=row.get("position_side"),
            entry_order_id=int(row["entry_order_id"]) if row.get("entry_order_id") not in (None, "") else None,
            entry_client_order_id=row.get("entry_client_order_id"), real_error=row.get("real_error"),
            real_submit_ambiguous=bool(row.get("real_submit_ambiguous", False)),
            signal_close_price=dec("signal_close_price", dec("entry", Decimal("0"))),
            ema_filter_length=volumatic.normalize_ema_filter_length(row.get("ema_filter_length")),
            ema_filter_value=dec("ema_filter_value"),
            ema_filter_slow_value=dec("ema_filter_slow_value"),
            ema_filter_passed=(bool(row.get("ema_filter_passed")) if row.get("ema_filter_passed") is not None else None),
            ema_exit_state=str(row.get("ema_exit_state") or "UNINITIALIZED").upper(),
            ema_exit_last_open_ms=max(0, int(row.get("ema_exit_last_open_ms") or 0)),
            ema_exit_last_fast=dec("ema_exit_last_fast"),
            ema_exit_last_slow=dec("ema_exit_last_slow"),
            ema_exit_trigger_close_ms=(int(row["ema_exit_trigger_close_ms"]) if row.get("ema_exit_trigger_close_ms") not in (None, "") else None),
            ema_exit_trigger_price=dec("ema_exit_trigger_price"),
            ema_exit_reason=row.get("ema_exit_reason"),
            ema_exit_client_order_id=row.get("ema_exit_client_order_id"),
            ema_exit_order_id=(int(row["ema_exit_order_id"]) if row.get("ema_exit_order_id") not in (None, "") else None),
            ema_exit_order_status=str(row.get("ema_exit_order_status") or "NOT_SUBMITTED").upper(),
            ema_exit_attempts=max(0, int(row.get("ema_exit_attempts") or 0)),
            ema_exit_last_error=row.get("ema_exit_last_error"),
            entry_executed_qty=dec("entry_executed_qty"),
            entry_order_status=str(row.get("entry_order_status") or "UNKNOWN").upper(),
            entry_order_terminal=bool(row.get("entry_order_terminal", False)),
            pending_cancel_reason=row.get("pending_cancel_reason"),
            last_ws_event_key=row.get("last_ws_event_key"),
            late_fill_during_api_cooldown=bool(row.get("late_fill_during_api_cooldown", False)),
            preserve_gtc_after_cooldown=bool(row.get("preserve_gtc_after_cooldown", False)),
            exchange_adopted=bool(row.get("exchange_adopted", False)),
            exchange_adoption_source=row.get("exchange_adoption_source"),
            real_intent=bool(row.get("real_intent", False)),
            entry_triggered_at=parse_iso(row.get("entry_triggered_at")),
            entry_trigger_price=dec("entry_trigger_price"),
        )


# ============================================================
# BINANCE REST MARKET DATA
# ============================================================

BINANCE_IP_WEIGHT_LIMIT = 2400
# Bobot per endpoint (estimasi dari dokumentasi Binance; default 1).
ENDPOINT_WEIGHTS = {
    "/fapi/v1/ticker/24hr": 40,
    "/fapi/v2/positionRisk": 5,
    "/fapi/v3/balance": 5,
    "/fapi/v2/account": 5,
    "/fapi/v1/accountConfig": 5,
    "/fapi/v1/klines": 2,  # fallback; estimator below accounts for the requested limit
    "/fapi/v1/exchangeInfo": 1,
    "/fapi/v1/time": 1,
    "/fapi/v2/ticker/price": 1,
}


def estimate_request_weight(path: str, params: dict[str, Any] | None = None) -> int:
    """Best local estimate; Binance response header remains authoritative."""
    params = params or {}
    if path == "/fapi/v1/klines":
        try:
            limit = int(params.get("limit", 500) or 500)
        except (TypeError, ValueError):
            limit = 500
        if limit < 100:
            return 1
        if limit < 500:
            return 2
        if limit <= 1000:
            return 5
        return 10
    if path == "/fapi/v1/ticker/24hr" and params.get("symbol"):
        return 1
    if path == "/fapi/v2/positionRisk" and not params.get("symbol"):
        return 10
    return max(1, int(ENDPOINT_WEIGHTS.get(path, 1)))


def parse_ban_until_ms(text: Any) -> int | None:
    match = re.search(r"banned until (\d{10,13})", str(text or ""))
    if match is None:
        return None
    value = int(match.group(1))
    return value * 1000 if value < 10**11 else value


def rate_limit_wait(response: Any, offset_ms: int = 0) -> tuple[float, bool]:
    """Detik tunggu dari Retry-After dan/atau 'banned until' (ambil yang terpanjang)."""
    retry: float | None = None
    try:
        retry = max(0.0, float(response.headers.get("Retry-After") or ""))
    except (TypeError, ValueError):
        retry = None
    ban_ms = parse_ban_until_ms(getattr(response, "text", ""))
    if ban_ms is not None:
        ban_s = max(0.0, ban_ms / 1000.0 - (time.time() + offset_ms / 1000.0))
        retry = ban_s if retry is None else max(retry, ban_s)
    if retry is not None:
        return retry, True
    status = getattr(response, "status_code", 0)
    return float(BINANCE_BAN_FALLBACK_SECONDS if status == 418 else BINANCE_RATE_LIMIT_FALLBACK_SECONDS), False


# ---------------------------------------------------------------------------
# Automatic Binance egress failover (DIRECT -> configured proxy -> ProxyScrape)
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "1") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_number(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = float(default)
    return min(high, max(low, value))


PROXY_AUTO_FAILOVER = _env_flag("PROXY_AUTO_FAILOVER", "1")
# ProxyScrape free-list API v4 (no key). Override with PROXYSCRAPE_API_URL; the
# URL may also return `ip:port`, `scheme://ip:port` or JSON {"proxies": [...]}.
PROXYSCRAPE_API_URL = (os.getenv("PROXYSCRAPE_API_URL") or (
    "https://api.proxyscrape.com/v4/free-proxy-list/get"
    "?request=display_proxies&protocol=http&proxy_format=ipport&format=text&timeout=10000"
)).strip()
PROXYSCRAPE_MIRROR_URLS = tuple(
    item.strip() for item in (os.getenv("PROXYSCRAPE_MIRROR_URLS") or (
        "https://cdn.jsdelivr.net/gh/proxyscrape/free-proxy-list@main/proxies/protocols/https/data.txt,"
        "https://cdn.jsdelivr.net/gh/proxyscrape/free-proxy-list@main/proxies/protocols/http/data.txt"
    )).split(",") if item.strip()
)
PROXY_POOL_SIZE = int(_env_number("PROXY_POOL_SIZE", 3, 1, 10))
PROXY_MAX_CANDIDATES = int(_env_number("PROXY_MAX_CANDIDATES", 400, 10, 3000))
PROXY_TEST_TIMEOUT_SECONDS = _env_number("PROXY_TEST_TIMEOUT_SECONDS", 6.0, 2.0, 30.0)
PROXY_TEST_CONCURRENCY = int(_env_number("PROXY_TEST_CONCURRENCY", 40, 1, 150))
PROXY_FAIL_THRESHOLD = int(_env_number("PROXY_FAIL_THRESHOLD", 3, 1, 20))
PROXY_BLACKLIST_SECONDS = _env_number("PROXY_BLACKLIST_SECONDS", 1800.0, 60.0, 86400.0)
PROXY_HOME_CHECK_SECONDS = _env_number("PROXY_HOME_CHECK_SECONDS", 120.0, 30.0, 3600.0)
PROXY_REFETCH_MIN_SECONDS = _env_number("PROXY_REFETCH_MIN_SECONDS", 45.0, 10.0, 900.0)
# 418 (IP ban) and 451 (restricted location) always trigger a switch; a plain
# 429 only when the resulting cooldown is long enough to justify a new IP.
PROXY_SWITCH_MIN_COOLDOWN_SECONDS = _env_number("PROXY_SWITCH_MIN_COOLDOWN_SECONDS", 120.0, 0.0, 86400.0)


def mask_proxy_url(url: str | None) -> str:
    """Never print proxy credentials: scheme://***@host:port."""
    if not url:
        return "DIRECT"
    try:
        parts = urlsplit(url)
        auth = "***@" if (parts.username or parts.password) else ""
        return f"{parts.scheme or 'http'}://{auth}{parts.hostname or '?'}:{parts.port or ''}".rstrip(":")
    except Exception:
        return "proxy:?"


def parse_proxy_list(text: str, limit: int = 5000) -> list[str]:
    """Parse ProxyScrape text/JSON into unique http(s) proxy URLs."""
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        value = str(raw or "").strip()
        if not value or len(value) > 200:
            return
        if "://" not in value:
            value = f"http://{value}"
        try:
            parts = urlsplit(value)
            port = parts.port
        except ValueError:
            return
        if parts.scheme not in {"http", "https"} or not parts.hostname or not port or not 1 <= port <= 65535:
            return  # SOCKS is not supported by requests/websockets proxy settings here
        if not re.fullmatch(r"[A-Za-z0-9.\-]+", parts.hostname):
            return
        if value not in seen:
            seen.add(value)
            found.append(value)

    body = (text or "").strip()
    if body[:1] in "{[":
        try:
            data = json.loads(body)
            rows = data.get("proxies") if isinstance(data, dict) else data
            for row in rows or []:
                if isinstance(row, dict):
                    add(row.get("proxy") or (f"{row.get('ip')}:{row.get('port')}" if row.get("ip") and row.get("port") else ""))
                else:
                    add(row)
        except (ValueError, TypeError):
            pass
    else:
        for line in body.splitlines():
            add(line)
    return found[:limit]


class ProxyFailoverManager:
    """Switch Binance traffic automatically when the active egress IP is unusable.

    Route order: DIRECT/Render -> configured BINANCE_PROXY_URL -> AUTO pool built
    from the ProxyScrape API. A route is abandoned on 418/451 (IP ban/geo block),
    on proxy-side failures (402/407/tunnel errors/repeated transport errors), and
    the manager returns to the primary route when its ban/dead window expires.
    Every candidate is validated against Binance /fapi/v1/time over verified TLS
    before use. All steps are logged with the "[PROXY]" prefix.
    """

    DIRECT, STATIC, AUTO = "DIRECT", "STATIC", "AUTO"
    PROXY_FAILURE_MARKERS = (
        "proxyerror", "unable to connect to proxy", "tunnel connection failed",
        "proxy authentication", "cannot connect to proxy", "proxy connect", "502 bad gateway",
        "402 payment required", "407 proxy",
    )

    def __init__(self) -> None:
        self.enabled = PROXY_AUTO_FAILOVER
        self._lock = threading.RLock()
        self.static_url = BINANCE_PROXY_URL or ""
        self._direct_egress = "direct" if self.static_url else BINANCE_EGRESS_ID
        self._static_egress = BINANCE_EGRESS_ID if self.static_url else ""
        self.mode = self.STATIC if self.static_url else self.DIRECT
        self.active_url: str = self.static_url
        self.egress_id: str = self._static_egress if self.static_url else self._direct_egress
        self.generation = 0
        self.fail_count = 0
        self.pool: list[tuple[str, float]] = []  # (url, latency_ms) best first
        self.blacklist: dict[str, float] = {}
        self.static_dead_until = 0.0
        self.direct_ban_until = 0.0
        self._needs_failover = False
        self._retry_at = 0.0
        self._retry_backoff = 60.0
        self._last_fetch = 0.0
        self._home_check_at = 0.0
        self._heartbeat_at = 0.0
        self._worker_running = False
        self._pending_reason = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listener: Any = None
        self.switch_count = 0
        self.last_switch_at = 0.0
        # Test seam: tests replace these with fakes (no network).
        self._probe_fn = self._probe_binance
        self._fetch_fn = self._fetch_candidates
        self.exit_ip = ""  # public IP Binance sees for the active route (for API-key IP whitelist)
        self._exit_ip_fn = self._lookup_exit_ip

    # ------------------------------------------------------------ accessors
    def request_proxies(self) -> dict[str, str] | None:
        url = self.active_url
        return {"http": url, "https": url} if url else None

    def ws_kwargs(self) -> dict[str, str]:
        url = self.active_url
        return {"proxy": url} if url else {}

    @property
    def home_mode(self) -> str:
        return self.STATIC if self.static_url else self.DIRECT

    def describe(self) -> str:
        return f"{self.mode}:{mask_proxy_url(self.active_url)} egress={self.egress_id}"

    def bind(self, loop: asyncio.AbstractEventLoop | None, listener: Any) -> None:
        self._loop = loop
        self._listener = listener

    # ------------------------------------------------------------- observers
    def observe_http(self, status: int | None, cooldown_seconds: float = 0.0) -> None:
        """Called for every Binance HTTP response (status only)."""
        if not self.enabled or status is None:
            return
        status = int(status)
        if status in (418, 451) or (status == 429 and cooldown_seconds >= PROXY_SWITCH_MIN_COOLDOWN_SECONDS):
            self._on_blocked(status, cooldown_seconds)
        elif status < 500 and status != 429:
            self.fail_count = 0

    def observe_transport_error(self, text: str) -> None:
        """Called for Binance transport failures (requests/websocket exceptions)."""
        if not self.enabled or self.mode == self.DIRECT:
            return  # without a proxy a transport error is not a proxy problem
        low = str(text or "").lower()
        fatal_billing = "402 payment required" in low or "407 proxy" in low or "proxy authentication" in low
        proxy_side = fatal_billing or any(marker in low for marker in self.PROXY_FAILURE_MARKERS)
        with self._lock:
            self.fail_count += 1
            count = self.fail_count
            url, mode = self.active_url, self.mode
        if not (proxy_side or count >= PROXY_FAIL_THRESHOLD):
            log.info("[PROXY] transport error #%d/%d via %s: %s", count, PROXY_FAIL_THRESHOLD, mask_proxy_url(url), low[:160])
            return
        now = time.monotonic()
        if fatal_billing:
            ttl = 6 * 3600.0
            why = "HTTP 402/407 dari proxy (saldo/kuota/izin proxy habis atau ditolak)"
        else:
            ttl = PROXY_BLACKLIST_SECONDS if mode == self.AUTO else 600.0
            why = "proxy tidak dapat dipakai (%s)" % (low[:120] or "transport error")
        with self._lock:
            if url != self.active_url:
                return  # route already changed by another thread
            if mode == self.STATIC:
                self.static_dead_until = max(self.static_dead_until, now + ttl)
            else:
                self.blacklist[url] = now + ttl
            self.pool = [item for item in self.pool if item[0] != url]
        log.warning("[PROXY] %s mati: %s | diblokir %.0f menit", mask_proxy_url(url), why, ttl / 60.0)
        self._schedule(f"{mode} proxy {mask_proxy_url(url)} gagal: {why}")

    def _on_blocked(self, status: int, seconds: float) -> None:
        now = time.monotonic()
        ttl = max(float(seconds or 0.0), 600.0 if status == 451 else 60.0)
        with self._lock:
            mode, url = self.mode, self.active_url
            if mode == self.DIRECT:
                self.direct_ban_until = max(self.direct_ban_until, now + ttl)
            elif mode == self.STATIC:
                self.static_dead_until = max(self.static_dead_until, now + ttl)
            else:
                self.blacklist[url] = max(self.blacklist.get(url, 0.0), now + ttl)
                self.pool = [item for item in self.pool if item[0] != url]
            if self._needs_failover and self._worker_running:
                return
        label = {418: "IP BANNED (HTTP 418)", 451: "LOKASI DIBLOKIR (HTTP 451)"}.get(status, f"RATE LIMIT PANJANG (HTTP {status})")
        log.warning("[PROXY] Binance menolak egress %s: %s, ban/cooldown %.0fs", self.describe(), label, ttl)
        self._schedule(f"egress {mode} {mask_proxy_url(url)} -> {label}")

    # ------------------------------------------------------------- scheduling
    def _schedule(self, reason: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._needs_failover = True
            if self._worker_running:
                self._pending_reason = reason
                return
            self._worker_running = True
        threading.Thread(target=self._worker, args=("failover", reason), name="proxy-failover", daemon=True).start()

    def tick(self) -> None:
        """Cheap periodic hook from the maintenance loop."""
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            if self._worker_running:
                return
            kind = ""
            if self._needs_failover and now >= self._retry_at:
                kind = "failover"
            elif self._home_check_useful(now) and now >= self._home_check_at:
                kind = "home"
            if self.mode == self.AUTO and now >= self._heartbeat_at:
                self._heartbeat_at = now + 600.0
                log.info(
                    "[PROXY] status | route=%s | pool=%d sehat | blacklist=%d | switches=%d | cek rute utama tiap %.0fs",
                    self.describe(), len(self.pool), len(self.blacklist), self.switch_count, PROXY_HOME_CHECK_SECONDS,
                )
            if not kind:
                return
            self._worker_running = True
        threading.Thread(
            target=self._worker, args=(kind, "retry terjadwal" if kind == "failover" else "cek rute utama"),
            name="proxy-failover", daemon=True,
        ).start()

    def _home_check_useful(self, now: float) -> bool:
        """A return-home probe only makes sense if it could land on a *different* route."""
        if self.mode == self.AUTO:
            return True
        if self.mode == self.DIRECT and self.static_url and now >= self.static_dead_until:
            return True  # configured proxy's dead window expired; try it again
        return False

    def _worker(self, kind: str, reason: str) -> None:
        try:
            if kind == "failover":
                self._failover(reason)
            else:
                self._return_home()
        except Exception:
            log.exception("[PROXY] worker error")
        finally:
            with self._lock:
                self._worker_running = False
                pending = self._pending_reason
                self._pending_reason = ""
            if pending and self._needs_failover:
                self._schedule(pending)

    # ------------------------------------------------------------- probing
    @staticmethod
    def _probe_binance(url: str | None, timeout: float) -> tuple[bool, str, float]:
        """GET /fapi/v1/time through `url` (None=direct). Returns (ok, detail, latency_ms)."""
        proxies = {"http": url, "https": url} if url else None
        started = time.monotonic()
        try:
            response = requests.get(
                f"{BINANCE_REST_BASE}/fapi/v1/time", proxies=proxies, timeout=(timeout, timeout),
                headers={"Accept": "application/json"},
            )
        except requests.RequestException as exc:
            return False, f"{type(exc).__name__}: {str(exc)[:100]}", 0.0
        latency = (time.monotonic() - started) * 1000.0
        if response.status_code == 200:
            try:
                if isinstance(response.json().get("serverTime"), int):
                    return True, "OK", latency
            except (ValueError, AttributeError):
                pass
            return False, "respons bukan JSON Binance", latency
        if response.status_code in (418, 429, 451):
            return False, f"Binance HTTP {response.status_code}", latency
        return False, f"HTTP {response.status_code}", latency

    @staticmethod
    def _lookup_exit_ip(url: str | None) -> str:
        """Public egress IP of a route (None=direct), via echo services; '' if unknown."""
        proxies = {"http": url, "https": url} if url else None
        for service in ("https://api.ipify.org", "https://checkip.amazonaws.com", "https://ifconfig.me/ip"):
            try:
                text = requests.get(service, proxies=proxies, timeout=(8, 8)).text.strip()
                ipaddress.ip_address(text)
                return text
            except (requests.RequestException, ValueError):
                continue
        return ""

    def refresh_exit_ip(self, reason: str = "") -> str:
        """Resolve and log the exit IP of the active route (blocking; run in a thread)."""
        url, mode = self.active_url, self.mode
        ip = self._exit_ip_fn(url or None)
        with self._lock:
            if url == self.active_url:
                self.exit_ip = ip
        if ip:
            log.info("[PROXY] IP KELUAR terdeteksi: %s | route=%s %s | %s", ip, mode, mask_proxy_url(url), reason)
        else:
            log.warning("[PROXY] IP keluar route %s tidak bisa dideteksi (%s)", self.describe(), reason)
        return ip

    def refresh_exit_ip_async(self, reason: str = "") -> None:
        threading.Thread(target=self.refresh_exit_ip, args=(reason,), name="proxy-exit-ip", daemon=True).start()

    def _probe(self, url: str | None, timeout: float | None = None) -> tuple[bool, str, float]:
        return self._probe_fn(url, timeout or PROXY_TEST_TIMEOUT_SECONDS)

    def _fetch_candidates(self) -> list[str]:
        sources = [("ProxyScrape API v4", PROXYSCRAPE_API_URL)] + [
            (f"mirror {index + 1}", url) for index, url in enumerate(PROXYSCRAPE_MIRROR_URLS)
        ]
        for label, url in sources:
            try:
                started = time.monotonic()
                response = requests.get(url, timeout=20, headers={"Accept": "text/plain,application/json"})
                if response.status_code != 200:
                    log.warning("[PROXY] %s HTTP %s", label, response.status_code)
                    continue
                items = parse_proxy_list(response.text)
                log.info("[PROXY] %s -> %d proxy dalam %.1fs", label, len(items), time.monotonic() - started)
                if items:
                    return items
            except requests.RequestException as exc:
                log.warning("[PROXY] %s gagal: %s", label, str(exc)[:160])
        return []

    # ------------------------------------------------------------- failover
    def _usable(self, url: str) -> bool:
        return self.blacklist.get(url, 0.0) <= time.monotonic() and url != self.static_url

    def _home_candidates(self, exclude_current: bool) -> list[tuple[str, str | None]]:
        now = time.monotonic()
        rows: list[tuple[str, str | None]] = []
        if self.static_url and now >= self.static_dead_until and not (exclude_current and self.mode == self.STATIC):
            rows.append((self.STATIC, self.static_url))
        if now >= self.direct_ban_until and not (exclude_current and self.mode == self.DIRECT):
            rows.append((self.DIRECT, None))
        return rows

    def _try_home(self, exclude_current: bool, reason: str) -> bool:
        for kind, url in self._home_candidates(exclude_current):
            label = "DIRECT (IP Render)" if kind == self.DIRECT else f"proxy terkonfigurasi {mask_proxy_url(url)}"
            log.info("[PROXY] STEP uji rute utama: %s", label)
            ok, detail, latency = self._probe(url)
            if ok:
                self._switch(kind, url, reason, latency)
                return True
            log.info("[PROXY] rute %s belum bisa: %s", label, detail)
            if "418" in detail or "451" in detail or "429" in detail:
                with self._lock:
                    if kind == self.DIRECT:
                        self.direct_ban_until = time.monotonic() + (600.0 if "451" in detail else 300.0)
                    else:
                        self.static_dead_until = time.monotonic() + 300.0
        return False

    def _failover(self, reason: str) -> None:
        started = time.monotonic()
        log.info("[PROXY] ===== FAILOVER MULAI | alasan=%s | route sekarang=%s =====", reason, self.describe())
        if self._try_home(exclude_current=True, reason=reason):
            return
        # AUTO pool: reuse validated entries first, otherwise refresh via ProxyScrape.
        refreshed = False
        for _ in range(4):
            candidate = self._next_pool_entry()
            if candidate is None:
                if refreshed or time.monotonic() - self._last_fetch < PROXY_REFETCH_MIN_SECONDS and self._last_fetch:
                    break
                refreshed = True
                self._refresh_pool()
                candidate = self._next_pool_entry()
                if candidate is None:
                    break
            url, latency = candidate
            ok, detail, latency_now = self._probe(url)
            if ok:
                self._switch(self.AUTO, url, reason, latency_now or latency)
                return
            log.info("[PROXY] kandidat %s gagal uji ulang: %s", mask_proxy_url(url), detail)
            with self._lock:
                self.blacklist[url] = time.monotonic() + PROXY_BLACKLIST_SECONDS
                self.pool = [item for item in self.pool if item[0] != url]
        with self._lock:
            self._retry_at = time.monotonic() + self._retry_backoff
            wait = self._retry_backoff
            self._retry_backoff = min(300.0, self._retry_backoff * 2)
        log.error(
            "[PROXY] ===== FAILOVER GAGAL setelah %.1fs: tidak ada rute Binance yang sehat. Coba lagi %.0fs lagi "
            "(route tetap %s) =====", time.monotonic() - started, wait, self.describe(),
        )
        self._notify({"kind": "exhausted", "reason": reason, "retry_seconds": wait, "route": self.describe()})

    def _next_pool_entry(self) -> tuple[str, float] | None:
        with self._lock:
            now = time.monotonic()
            self.blacklist = {k: v for k, v in self.blacklist.items() if v > now}
            for url, latency in self.pool:
                if self._usable(url) and not (self.mode == self.AUTO and url == self.active_url):
                    return url, latency
        return None

    def _refresh_pool(self) -> None:
        self._last_fetch = time.monotonic()
        log.info("[PROXY] STEP ambil daftar proxy dari ProxyScrape ...")
        raw = self._fetch_fn()
        with self._lock:
            candidates = [url for url in raw if self._usable(url) and url != self.active_url]
        if not candidates:
            log.warning("[PROXY] daftar proxy kosong / semua sudah diblokir")
            return
        random.shuffle(candidates)
        candidates = candidates[:PROXY_MAX_CANDIDATES]
        log.info(
            "[PROXY] STEP validasi %d kandidat ke Binance /fapi/v1/time (concurrency=%d, timeout=%.0fs, target=%d sehat)",
            len(candidates), PROXY_TEST_CONCURRENCY, PROXY_TEST_TIMEOUT_SECONDS, PROXY_POOL_SIZE,
        )
        healthy: list[tuple[str, float]] = []
        tested = failed = 0
        executor = ThreadPoolExecutor(max_workers=PROXY_TEST_CONCURRENCY, thread_name_prefix="proxy-test")
        try:
            futures = {executor.submit(self._probe, url): url for url in candidates}
            for future in as_completed(futures):
                url = futures[future]
                tested += 1
                try:
                    ok, detail, latency = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    ok, detail, latency = False, f"error {exc}", 0.0
                if ok:
                    healthy.append((url, latency))
                    log.info("[PROXY] OK   %s | %.0f ms (%d/%d sehat)", mask_proxy_url(url), latency, len(healthy), PROXY_POOL_SIZE)
                    if len(healthy) >= PROXY_POOL_SIZE:
                        break
                else:
                    failed += 1
                    if "Binance HTTP" in detail:
                        log.info("[PROXY] BAN %s | %s", mask_proxy_url(url), detail)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        healthy.sort(key=lambda item: item[1])
        with self._lock:
            self.pool = healthy
        log.info("[PROXY] validasi selesai | diuji=%d | gagal=%d | sehat=%d", tested, failed, len(healthy))

    def _return_home(self) -> None:
        with self._lock:
            self._home_check_at = time.monotonic() + PROXY_HOME_CHECK_SECONDS
        if not self._home_candidates(exclude_current=True):
            return
        log.info("[PROXY] cek apakah rute utama sudah pulih (route sekarang=%s)", self.describe())
        self._try_home(exclude_current=True, reason="rute utama sudah pulih")

    # ------------------------------------------------------------- switching
    def _switch(self, kind: str, url: str | None, reason: str, latency_ms: float) -> None:
        with self._lock:
            if kind == self.mode and (url or "") == self.active_url:
                self._needs_failover = False
                log.info("[PROXY] route %s masih sehat; tidak ada pergantian", self.describe())
                return
            old_egress, old_desc = self.egress_id, self.describe()
            self.mode = kind
            self.active_url = url or ""
            if kind == self.DIRECT:
                self.egress_id = self._direct_egress
            elif kind == self.STATIC:
                self.egress_id = self._static_egress
            else:
                parts = urlsplit(self.active_url)
                self.egress_id = f"auto-proxy:{parts.hostname}:{parts.port or 0}"
            self.generation += 1
            self.switch_count += 1
            self.fail_count = 0
            self._needs_failover = False
            self._retry_at = 0.0
            self._retry_backoff = 60.0
            self._home_check_at = time.monotonic() + PROXY_HOME_CHECK_SECONDS
            self.last_switch_at = time.time()
            new_egress, new_desc = self.egress_id, self.describe()
            if kind == self.STATIC:
                self.static_dead_until = 0.0
            if kind == self.DIRECT:
                self.direct_ban_until = 0.0
        BINANCE_COOLDOWN.rebind(old_egress, new_egress)
        old_ip = self.exit_ip
        self.exit_ip = ""
        new_ip = self.refresh_exit_ip(f"setelah switch ({reason})")
        log.warning(
            "[PROXY] ===== SWITCH EGRESS: %s  ->  %s | IP KELUAR BARU=%s (sebelumnya %s) | latency=%.0fms | alasan=%s "
            "| pasang ke REST + WebSocket =====",
            old_desc, new_desc, new_ip or "tidak terdeteksi", old_ip or "?", latency_ms, reason,
        )
        self._notify({
            "kind": "switched", "reason": reason, "old": old_desc, "new": new_desc,
            "mode": kind, "latency_ms": latency_ms, "generation": self.generation,
            "exit_ip": new_ip, "old_ip": old_ip,
        })

    def _notify(self, info: dict[str, Any]) -> None:
        loop, listener = self._loop, self._listener
        if loop is None or listener is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(listener(info), loop)
        except Exception:
            log.debug("[PROXY] listener gagal dijadwalkan", exc_info=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            return {
                "enabled": self.enabled, "mode": self.mode, "route": mask_proxy_url(self.active_url),
                "egress_id": self.egress_id, "pool": [mask_proxy_url(url) for url, _ in self.pool],
                "blacklist": sum(1 for v in self.blacklist.values() if v > now),
                "static_dead_s": max(0.0, self.static_dead_until - now),
                "direct_ban_s": max(0.0, self.direct_ban_until - now),
                "switches": self.switch_count, "exit_ip": self.exit_ip,
            }


PROXY_MANAGER = ProxyFailoverManager()


class BinanceCooldownGate:
    """One IP-wide cooldown shared by public REST and signed Futures REST.

    Keep the last ban interval on disk so an order that expired while the bot
    was offline can still be recognized after the IP cooldown has elapsed.
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until = 0.0
        self._last_request = 0.0
        self._reason = ""
        self._started_epoch = 0.0
        self._until_epoch = 0.0
        # Preserve multiple recent ban windows. A bot may be offline across more
        # than one rate-limit event; old trade expiries must remain attributable.
        self._ban_intervals: list[list[float]] = []
        # Fresh-engine policy: cooldown is process/session memory only.
        # Do not load a previous process' IP ban marker on /try.
        self._state_path = BASE_DIR / "data" / "binance_cooldown_state.json"
        # Cooldowns belong to one egress IP. When the proxy manager changes the
        # route, the old IP's cooldown is parked here and the new IP starts clean.
        self._parked: dict[str, tuple[float, str]] = {}

    def _restore_persisted_cooldown(self) -> None:
        """Restore a server IP ban and its historical time range."""
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            saved_egress = str(payload.get("egress_id") or "").strip()
            if not saved_egress or saved_egress != PROXY_MANAGER.egress_id:
                log.info(
                    "[BINANCE] persisted cooldown diabaikan: egress berbeda/legacy "
                    "(saved=%s, current=%s).",
                    saved_egress or "unknown", PROXY_MANAGER.egress_id,
                )
                return
            now_epoch = time.time()
            self._started_epoch = float(payload.get("started_epoch") or 0.0)
            self._until_epoch = float(payload.get("until_epoch") or 0.0)
            intervals = payload.get("ban_intervals")
            if isinstance(intervals, list):
                for item in intervals[-64:]:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        start, end = float(item[0]), float(item[1])
                        if start > 0 and end > start:
                            self._ban_intervals.append([start, end])
            if not self._ban_intervals and self._started_epoch > 0 and self._until_epoch > self._started_epoch:
                self._ban_intervals.append([self._started_epoch, self._until_epoch])
            remaining = max(0.0, self._until_epoch - now_epoch)
            self._reason = str(payload.get("reason") or "Persisted Binance cooldown")[:500]
            if remaining > 0:
                self._until = time.monotonic() + remaining
                log.info("[BINANCE] persisted cooldown restored | remaining=%.0fs | reason=%s", remaining, self._reason)
            elif self._until_epoch and now_epoch - self._until_epoch > 14 * 86400:
                self._state_path.unlink(missing_ok=True)
                self._started_epoch = self._until_epoch = 0.0
                self._ban_intervals = []
                self._reason = ""
        except FileNotFoundError:
            return
        except Exception:
            log.warning("[BINANCE] cooldown marker invalid; ignoring it", exc_info=True)

    def merge_persisted_payload(self, payload: dict[str, Any]) -> None:
        """Merge a GitHub/local snapshot without shortening an active cooldown."""
        if not isinstance(payload, dict):
            return
        saved_egress = str(payload.get("egress_id") or "").strip()
        if not saved_egress or saved_egress != PROXY_MANAGER.egress_id:
            log.info(
                "[BINANCE] cooldown snapshot eksternal diabaikan: egress berbeda/legacy "
                "(saved=%s, current=%s).",
                saved_egress or "unknown", PROXY_MANAGER.egress_id,
            )
            return
        try:
            now_epoch = time.time()
            remote_start = float(payload.get("started_epoch") or 0.0)
            remote_until = float(payload.get("until_epoch") or 0.0)
            remote_intervals: list[list[float]] = []
            rows = payload.get("ban_intervals")
            if isinstance(rows, list):
                for item in rows[-64:]:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        a, b = float(item[0]), float(item[1])
                        if a > 0 and b > a:
                            remote_intervals.append([a, b])
            if not remote_intervals and remote_start > 0 and remote_until > remote_start:
                remote_intervals.append([remote_start, remote_until])
        except (TypeError, ValueError):
            return

        with self._lock:
            combined = self._ban_intervals + remote_intervals
            combined.sort(key=lambda item: item[0])
            deduped: list[list[float]] = []
            for start, end in combined:
                if deduped and start <= deduped[-1][1] + 1.0:
                    deduped[-1][1] = max(deduped[-1][1], end)
                else:
                    deduped.append([start, end])
            self._ban_intervals = deduped[-64:]

            local_until_epoch = self._until_epoch
            if remote_until > local_until_epoch:
                self._started_epoch = remote_start
                self._until_epoch = remote_until
                self._reason = str(payload.get("reason") or "Persisted Binance cooldown")[:500]
            elif not self._reason and payload.get("reason"):
                self._reason = str(payload.get("reason"))[:500]

            remaining = max(0.0, self._until_epoch - now_epoch)
            if remaining > 0:
                self._until = max(self._until, time.monotonic() + remaining)
            if self._until_epoch > 0:
                self._persist_locked()

    def _persist_locked(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            now_epoch = time.time()
            remaining = max(0.0, self._until - time.monotonic())
            if self._until > 0:
                self._until_epoch = max(self._until_epoch, now_epoch + remaining)
            payload = {
                "egress_id": PROXY_MANAGER.egress_id,
                "started_epoch": self._started_epoch,
                "until_epoch": self._until_epoch,
                "ban_intervals": self._ban_intervals[-64:],
                "reason": self._reason,
                "updated_epoch": now_epoch,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            temp = self._state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, self._state_path)
        except Exception:
            log.warning("[BINANCE] gagal menyimpan cooldown marker lokal", exc_info=True)

    def rebind(self, old_egress: str, new_egress: str) -> None:
        """Park the old egress cooldown and load the new egress' (usually none)."""
        if old_egress == new_egress:
            return
        with self._lock:
            now = time.monotonic()
            if self._until > now:
                self._parked[old_egress] = (self._until, self._reason)
            parked = self._parked.pop(new_egress, None)
            if parked and parked[0] > now:
                self._until, self._reason = parked
                log.info("[PROXY] cooldown egress %s dipulihkan: %.0fs tersisa", new_egress, parked[0] - now)
            else:
                self._until, self._reason = 0.0, ""
                log.info("[PROXY] cooldown Binance dikosongkan untuk egress baru %s (cooldown lama %s disimpan)", new_egress, old_egress)

    def remaining(self) -> float:
        with self._lock:
            remaining = max(0.0, self._until - time.monotonic())
            if remaining <= 0 and self._until > 0:
                self._until = 0.0
                self._reason = ""
                # Retain started/until epochs to reconcile expiries after restart.
                self._persist_locked()
            return remaining

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def covered_by_last_ban(self, epoch_seconds: float) -> bool:
        """Return True when a timestamp fell inside the most recently recorded IP ban."""
        try:
            stamp = float(epoch_seconds)
        except (TypeError, ValueError):
            return False
        with self._lock:
            intervals = self._ban_intervals or (
                [[self._started_epoch, self._until_epoch]]
                if self._started_epoch > 0 and self._until_epoch > self._started_epoch else []
            )
            return any(start <= stamp <= end for start, end in intervals)

    def extend(self, seconds: float, reason: str = "Binance rate limit") -> float:
        seconds = max(0.0, float(seconds))
        with self._lock:
            now_mono = time.monotonic()
            now_epoch = time.time()
            if self._until <= now_mono:
                self._started_epoch = now_epoch
                self._until_epoch = now_epoch + seconds
                self._ban_intervals.append([self._started_epoch, self._until_epoch])
            else:
                self._until_epoch = max(self._until_epoch, now_epoch + seconds)
                if self._ban_intervals:
                    self._ban_intervals[-1][1] = max(self._ban_intervals[-1][1], self._until_epoch)
                else:
                    self._ban_intervals.append([self._started_epoch or now_epoch, self._until_epoch])
            self._ban_intervals = self._ban_intervals[-64:]
            self._until = max(self._until, now_mono + seconds)
            self._reason = str(reason or "Binance rate limit")[:500]
            self._persist_locked()
            return max(0.0, self._until - time.monotonic())

    def wait_for_request_slot(self, spacing: float) -> float:
        adaptive = max(0.0, float(spacing))
        stats = globals().get("API_STATS")
        if stats is not None:
            try:
                adaptive = max(adaptive, float(stats.adaptive_spacing()))
            except Exception:
                pass
        with self._lock:
            return max(0.0, self._last_request + adaptive - time.monotonic())

    def record_request(self) -> None:
        with self._lock:
            self._last_request = time.monotonic()


BINANCE_COOLDOWN = BinanceCooldownGate()
BINANCE_HTTP_LOCK = asyncio.Lock()


def _local_rate_limit_error(endpoint: str) -> "BinanceRateLimitError":
    remaining = max(1.0, BINANCE_COOLDOWN.remaining())
    exc = BinanceRateLimitError(
        f"Binance REST sedang cooldown global ({remaining:.0f}s); request ditahan tanpa dikirim.",
        status_code=429, code=-1003, endpoint=endpoint,
        server_cooldown_seconds=remaining,
        bot_cooldown_seconds=remaining,
        retry_after_known=False,
    )
    exc.internal = True
    return exc


def binance_limit_log(exc: Any, cooldown: float, where: str, impact: str) -> str:
    """Pesan log rate limit Binance yang mudah dibaca."""
    status = getattr(exc, "status_code", None) or 429
    server = max(
        float(getattr(exc, "server_cooldown_seconds", 0.0) or 0.0),
        float(getattr(exc, "retry_after_seconds", 0.0) or 0.0),
    )
    known = bool(getattr(exc, "retry_after_known", True))
    match = re.search(r"limit of IP\([^)]*\) is (\d+) requests? per minute", str(exc))
    limit = int(match.group(1)) if match else BINANCE_IP_WEIGHT_LIMIT
    if status == 418:
        cause = "IP server diblokir sementara (ban) karena request tetap dikirim setelah kena limit"
    else:
        cause = f"total request REST dari IP server melewati batas {limit}/menit"
    retry_text = f"Retry-After Binance {server:.0f} dtk" if known else f"perkiraan {server:.0f} dtk (Binance tidak kirim Retry-After)"
    lines = [
        f"⛔ BINANCE MEMBATASI REQUEST (HTTP {status})",
        f"• Penyebab  : {cause}",
        f"• Terjadi di: {where}",
        f"• Jeda bot  : {cooldown:.0f} dtk ({retry_text} + pengaman {BINANCE_RATE_LIMIT_SAFETY_SECONDS} dtk)",
    ]
    now = time.monotonic()
    if API_STATS.used_weight is not None and API_STATS.used_weight_at is not None and now - API_STATS.used_weight_at <= 120.0:
        lines.append(f"• Bobot IP  : {API_STATS.used_weight}/{limit} per menit")
    counts = API_STATS.window(60.0)
    top = sorted(counts.items(), key=lambda kv: ENDPOINT_WEIGHTS.get(kv[0], 1) * kv[1], reverse=True)[:3]
    if top:
        lines.append("• Terbanyak 1 menit terakhir: " + ", ".join(f"{p.replace('/fapi/v1/', '')} {n}x" for p, n in top))
    lines.append(f"• Dampak    : {impact}")
    return "\n".join(lines)


class ApiStats:
    """Hitung request Binance per endpoint + bobot IP dari header Binance."""

    def __init__(self) -> None:
        # Store the estimate per request so variable-weight endpoints (notably
        # klines) are not flattened into one inaccurate constant.
        self.calls: deque[tuple[float, str, int, int]] = deque(maxlen=50000)
        self.total = 0
        self.limit_hits = 0
        self.last_limit_at: datetime | None = None
        self.used_weight: int | None = None
        self.used_weight_at: float | None = None
        self.max_used_weight = 0

    def record(
        self, path: str, status: int, used_weight: str | None,
        estimated_weight: int | None = None,
    ) -> None:
        now = time.monotonic()
        estimate = max(1, int(estimated_weight if estimated_weight is not None else estimate_request_weight(path)))
        self.calls.append((now, path, status, estimate))
        self.total += 1
        if status in (418, 429):
            self.limit_hits += 1
            self.last_limit_at = now_utc()
        try:
            weight = int(used_weight) if used_weight not in (None, "") else None
        except (TypeError, ValueError):
            weight = None
        if weight is not None:
            self.used_weight = weight
            self.used_weight_at = now
            self.max_used_weight = max(self.max_used_weight, weight)

    def window(self, seconds: float, end: float | None = None) -> dict[str, int]:
        end = end if end is not None else time.monotonic()
        cutoff = end - seconds
        counts: dict[str, int] = {}
        for ts, path, _, _ in reversed(self.calls):
            if ts > end:
                continue
            if ts < cutoff:
                break
            counts[path] = counts.get(path, 0) + 1
        return counts

    def window_weight(self, seconds: float, end: float | None = None) -> int:
        end = end if end is not None else time.monotonic()
        cutoff = end - seconds
        total = 0
        for ts, _, _, estimate in reversed(self.calls):
            if ts > end:
                continue
            if ts < cutoff:
                break
            total += estimate
        return total

    @staticmethod
    def weight_of(counts: dict[str, int]) -> int:
        return sum(ENDPOINT_WEIGHTS.get(path, 1) * n for path, n in counts.items())

    def adaptive_spacing(self) -> float:
        """Back off before Binance reaches its shared IP request-weight ceiling."""
        base = max(0.10, BINANCE_PUBLIC_REQUEST_SPACING_SECONDS)
        if self.used_weight is None or self.used_weight_at is None:
            return base
        age = time.monotonic() - self.used_weight_at
        if age > 60.0:
            return base
        ratio = max(0.0, float(self.used_weight) / float(BINANCE_IP_WEIGHT_LIMIT))
        if ratio >= 0.92:
            return max(base, 6.0)
        if ratio >= 0.85:
            return max(base, 4.0)
        if ratio >= 0.75:
            return max(base, 2.5)
        if ratio >= 0.65:
            return max(base, 1.5)
        if ratio >= 0.50:
            return max(base, 1.0)
        return base


API_STATS = ApiStats()

# Compact representation of the seven fields used by the H4/D1 strategy:
# open time, OHLC, base volume, and quote volume. Taker-buy volume is unused.
# NumPy arrays keep historical OHLCV data compact and reduce RAM pressure.
KLINE_CACHE_DTYPE = np.dtype([
    ("t", "<i8"), ("o", "<f8"), ("h", "<f8"), ("l", "<f8"),
    ("c", "<f8"), ("v", "<f8"), ("qv", "<f8"),
])


class BinanceREST:
    """
    Public market-data REST only.

    Tidak ada signing.
    Tidak ada API order.
    """

    def __init__(self) -> None:
        self.base_url = BINANCE_REST_BASE
        self._exchange_info_cache: dict[str, SymbolMeta] | None = None
        self._exchange_info_cached_at = 0.0
        self._exchange_info_cache_ttl_seconds = float(UNIVERSE_REFRESH_SECONDS)
        self._ticker_24h_cache: list[dict[str, Any]] | None = None
        self._ticker_24h_cached_at = 0.0
        # Universe ranking is based on rolling 24h volume; refreshing every 5 minutes
        # is sufficient and avoids repeating the 40-weight all-symbol ticker call per batch.
        self._ticker_24h_cache_ttl_seconds = 300.0
        # One REST owner for the whole process. Kline cache is bounded by both
        # entry count and candle count; strategy.py never opens its own socket.
        self._request_lock = asyncio.Lock()
        # Reuse TCP/TLS connections for sequential REST calls; request lock means
        # this session is never used concurrently by the public market-data client.
        self._session = requests.Session()
        self._last_request_monotonic = 0.0
        self._rest_cooldown_until = 0.0
        # Bounded LRU and ephemeral-disk cache for repeated H4/D1 kline requests.
        self._kline_disk_cache_dir = Path(
            os.getenv("BINANCE_KLINE_DISK_CACHE_DIR")
            or (Path(tempfile.gettempdir()) / "volumatic_kline_cache")
        )
        self._kline_disk_cache_max_bytes = max(32, int(os.getenv("KLINE_DISK_CACHE_MAX_MB", "512") or "512")) * 1024 * 1024
        self._kline_disk_cache_lock = threading.RLock()
        self._kline_disk_cache_hits = 0
        self._kline_disk_cache_misses = 0
        self._kline_disk_cache_writes = 0
        self._kline_cache: OrderedDict[tuple[str, str, int], tuple[float, np.ndarray]] = OrderedDict()
        self._kline_locks: dict[tuple[str, str, int], asyncio.Lock] = {}
        self._kline_cache_candles = 0
        self._kline_cache_max_entries = 32
        self._kline_cache_max_candles = 350_000
        self._kline_cache_ttl = {"5m": 300.0, "15m": 45.0, "1h": 120.0, "4h": 300.0, "1d": 900.0}
        # One current M15 row plus at most eight closed WebSocket rows per
        # symbol. The maps are explicitly bounded for Render Free memory.
        self._forming_klines: OrderedDict[tuple[str, str], tuple[int, float, float, float, float, float, float]] = OrderedDict()
        self._ws_closed_klines: OrderedDict[tuple[str, str], deque[tuple[int, float, float, float, float, float, float]]] = OrderedDict()
        self._ws_kline_max_symbols = 1024
        self._ws_kline_closed_rows_per_symbol = 8

    def update_kline_from_ws(self, symbol: str, kline: dict[str, Any]) -> None:
        """Accept one Binance kline event without retaining the full event payload."""
        if not isinstance(kline, dict):
            return
        pair = normalize_symbol(symbol)
        interval = self._binance_interval(str(kline.get("i") or "15m"))
        if not pair:
            return
        try:
            open_ms = int(kline["t"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        key = (pair, interval)
        is_closed = kline.get("x") is True
        if not is_closed:
            # The active candle's open timestamp and open price never change.
            # Skip repeated 250ms updates before converting the remaining
            # OHLCV strings; the final x=True event remains authoritative.
            previous = self._forming_klines.get(key)
            if previous is not None and int(previous[0]) == open_ms:
                self._forming_klines.move_to_end(key)
                return
        try:
            row = (
                open_ms,
                float(kline["o"]), float(kline["h"]), float(kline["l"]),
                float(kline["c"]), float(kline.get("v") or 0.0),
                float(kline.get("q") or 0.0),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if not all(math.isfinite(value) for value in row[1:]):
            return
        if min(row[1:5]) <= 0 or row[2] < max(row[1], row[4], row[3]) or row[3] > min(row[1], row[2], row[4]):
            return
        if row[5] < 0 or row[6] < 0:
            return

        if not is_closed:
            self._forming_klines[key] = row
            self._forming_klines.move_to_end(key)
            while len(self._forming_klines) > self._ws_kline_max_symbols:
                self._forming_klines.popitem(last=False)
            return

        # The completed candle is appended to a tiny bounded replay buffer.
        closed = self._ws_closed_klines.get(key)
        if closed is None:
            closed = deque(maxlen=self._ws_kline_closed_rows_per_symbol)
            self._ws_closed_klines[key] = closed
        if not closed or open_ms > closed[-1][0]:
            closed.append(row)
        elif open_ms == closed[-1][0]:
            closed[-1] = row
        self._ws_closed_klines.move_to_end(key)
        while len(self._ws_closed_klines) > self._ws_kline_max_symbols:
            self._ws_closed_klines.popitem(last=False)
        self._forming_klines.pop(key, None)

        # Update only cache entries already resident in RAM. Disk-only histories
        # are reconciled from the small replay buffer when next requested.
        one = np.asarray([row], dtype=KLINE_CACHE_DTYPE)
        span_ms = self._interval_ms(interval)
        for cache_key in list(self._kline_cache.keys()):
            if cache_key[0] != pair or cache_key[1] != interval:
                continue
            cached_rows = self._kline_cache[cache_key][1]
            if not cached_rows.size:
                continue
            last_open = int(cached_rows["t"][-1])
            if open_ms == last_open:
                merged = np.concatenate((cached_rows[:-1], one))
            elif open_ms == last_open + span_ms:
                merged = np.concatenate((cached_rows, one))
            else:
                continue
            self._store_kline_cache(cache_key, merged[-cache_key[2]:])

    def _merge_ws_closed_klines(
        self, symbol: str, interval: str, rows: np.ndarray, count: int,
    ) -> tuple[np.ndarray, bool, bool]:
        """Merge contiguous closed WS rows; report a gap so REST can fill it."""
        key = (normalize_symbol(symbol), self._binance_interval(interval))
        buffered = self._ws_closed_klines.get(key)
        if not buffered or not isinstance(rows, np.ndarray) or not rows.size:
            return rows, False, False
        span_ms = self._interval_ms(key[1])
        merged = rows
        last_open = int(merged["t"][-1])
        changed = False
        gap = False
        for row in buffered:
            open_ms = int(row[0])
            if open_ms <= last_open:
                continue
            if open_ms != last_open + span_ms:
                gap = True
                break
            one = np.asarray([row], dtype=KLINE_CACHE_DTYPE)
            merged = np.concatenate((merged, one))
            last_open = open_ms
            changed = True
        return np.ascontiguousarray(merged[-int(count):]), gap, changed

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Public REST with a shared IP cooldown; never fires during a 418/429 ban."""
        params = params or {}
        async with self._request_lock:
            async with BINANCE_HTTP_LOCK:
                cooldown = BINANCE_COOLDOWN.remaining()
                if cooldown > 0:
                    raise _local_rate_limit_error(path)
                spacing = BINANCE_COOLDOWN.wait_for_request_slot(BINANCE_PUBLIC_REQUEST_SPACING_SECONDS)
                if spacing > 0:
                    await asyncio.sleep(spacing)
                if BINANCE_COOLDOWN.remaining() > 0:
                    raise _local_rate_limit_error(path)

                def request() -> tuple[int, Any, dict[str, str], str]:
                    response = self._session.get(
                        f"{self.base_url}{path}", params=params,
                        proxies=PROXY_MANAGER.request_proxies(),
                        timeout=HTTP_REQUEST_SECONDS,
                    )
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = None
                    return response.status_code, payload, dict(response.headers), response.text[:700]

                try:
                    status, payload, headers, body_text = await asyncio.to_thread(request)
                except requests.RequestException as exc:
                    PROXY_MANAGER.observe_transport_error(str(exc))
                    raise BinanceAPIError(
                        f"Transport error Binance REST {path}: {exc}",
                        status_code=None, code=None, endpoint=path,
                    ) from exc
                self._last_request_monotonic = time.monotonic()
                BINANCE_COOLDOWN.record_request()

                # IMPORTANT: publish the IP cooldown BEFORE releasing the shared
                # HTTP lock. Otherwise a scanner/maintenance waiter can slip one
                # more request between the 429/418 response and gate activation.
                used = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
                API_STATS.record(path, status, used, estimate_request_weight(path, params))
                code = payload.get("code") if isinstance(payload, dict) else None
                message = payload.get("msg") if isinstance(payload, dict) else body_text
                limited = (
                    status in {418, 429}
                    or str(code) in {"-1003", "-1015"}
                    or "too many requests" in str(message).lower()
                    or "way too many requests" in str(message).lower()
                )
                if limited:
                    server_seconds, known = rate_limit_wait(type("ResponseLike", (), {
                        "headers": headers, "status_code": status, "text": body_text,
                    })())
                    cooldown = max(1.0, server_seconds + rate_limit_safety(status))
                    self._rest_cooldown_until = max(self._rest_cooldown_until, time.monotonic() + cooldown)
                    BINANCE_COOLDOWN.extend(cooldown, f"HTTP {status} {path}: {message}")
                    PROXY_MANAGER.observe_http(status, cooldown)
                    raise BinanceRateLimitError(
                        f"Binance REST {path}: HTTP {status}, code={code}, {message}",
                        status_code=status or 429,
                        code=int(code) if str(code).lstrip("-").isdigit() else None,
                        endpoint=path,
                        server_cooldown_seconds=server_seconds,
                        bot_cooldown_seconds=cooldown,
                        retry_after_known=known,
                    )

        # `code` and `message` are set for every HTTP response above.
        PROXY_MANAGER.observe_http(status)
        if status >= 400:
            raise BinanceAPIError(
                f"Binance REST {path}: HTTP {status}: {message}",
                status_code=status, code=int(code) if str(code).lstrip("-").isdigit() else None,
                endpoint=path,
            )
        if payload is None:
            raise RuntimeError(f"Binance REST {path}: respons JSON tidak valid")
        return payload

    @staticmethod
    def _interval_ms(interval: str) -> int:
        token = str(interval or "").strip().lower()
        aliases = {
            "5": 300_000, "5m": 300_000, "15": 900_000, "15m": 900_000,
            "60": 3_600_000, "1h": 3_600_000, "240": 14_400_000,
            "4h": 14_400_000, "d": 86_400_000, "1d": 86_400_000,
        }
        if token not in aliases:
            raise ValueError(f"Interval Binance tidak didukung: {interval}")
        return aliases[token]

    @staticmethod
    def _binance_interval(interval: str) -> str:
        token = str(interval or "").strip()
        aliases = {"5": "5m", "15": "15m", "60": "1h", "240": "4h", "D": "1d"}
        return aliases.get(token, aliases.get(token.upper(), token.lower()))

    @staticmethod
    def _pack_kline_rows(
        rows: list[list[Any]], span_ms: int, now_ms: int, *, include_forming: bool = False
    ) -> np.ndarray:
        """Convert Binance kline rows; default drops the active candle."""
        packed: list[tuple[int, float, float, float, float, float, float]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 10:
                continue
            try:
                open_ms = int(row[0])
                if open_ms > now_ms:
                    continue
                if not include_forming and open_ms + span_ms > now_ms:
                    continue  # closed candles only
                # The strategy consumes OHLC; volume is retained for audit/extension.
                values = (
                    float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                    float(row[5]), float(row[7]),
                )
                if not all(math.isfinite(value) for value in values):
                    continue
                if min(values[0:4]) <= 0 or values[1] < max(values[0], values[2], values[3]):
                    continue
                if values[2] > min(values[0], values[1], values[3]):
                    continue
                packed.append((open_ms, *values))
            except (TypeError, ValueError, IndexError, OverflowError):
                continue
        if not packed:
            return np.empty(0, dtype=KLINE_CACHE_DTYPE)
        return np.asarray(packed, dtype=KLINE_CACHE_DTYPE)

    @staticmethod
    def _merge_packed_klines(parts: list[np.ndarray], count: int) -> np.ndarray:
        nonempty = [part for part in parts if isinstance(part, np.ndarray) and part.size]
        if not nonempty:
            return np.empty(0, dtype=KLINE_CACHE_DTYPE)
        rows = nonempty[0] if len(nonempty) == 1 else np.concatenate(nonempty)
        if rows.size > 1 and np.any(rows["t"][1:] <= rows["t"][:-1]):
            order = np.argsort(rows["t"], kind="stable")
            rows = rows[order]
            # Keep the last copy of a duplicated open timestamp, matching the old dict merge.
            keep = np.r_[rows["t"][1:] != rows["t"][:-1], True]
            rows = rows[keep]
        if rows.shape[0] > count:
            rows = rows[-count:]
        return np.ascontiguousarray(rows)

    def _store_kline_cache(self, key: tuple[str, str, int], rows: np.ndarray) -> None:
        old = self._kline_cache.pop(key, None)
        if old is not None:
            self._kline_cache_candles -= int(old[1].shape[0])
        packed = np.ascontiguousarray(rows, dtype=KLINE_CACHE_DTYPE)
        self._kline_cache[key] = (time.monotonic(), packed)
        self._kline_cache_candles += int(packed.shape[0])
        self._kline_cache.move_to_end(key)
        while (len(self._kline_cache) > self._kline_cache_max_entries
               or self._kline_cache_candles > self._kline_cache_max_candles):
            _old_key, (_at, old_rows) = self._kline_cache.popitem(last=False)
            self._kline_cache_candles = max(0, self._kline_cache_candles - int(old_rows.shape[0]))

    def _kline_disk_path(self, key: tuple[str, str, int]) -> Path:
        symbol, interval, count = key
        safe_symbol = re.sub(r"[^A-Z0-9]", "", str(symbol).upper())
        safe_interval = re.sub(r"[^a-z0-9]", "", str(interval).lower())
        return self._kline_disk_cache_dir / f"{safe_symbol}_{safe_interval}_{int(count)}.npy"

    def _load_kline_disk_cache(
        self, key: tuple[str, str, int], count: int
    ) -> tuple[float, np.ndarray] | None:
        """Load a validated read-only .npy memmap; disk is only a rebuildable cache."""
        path = self._kline_disk_path(key)
        try:
            stat = path.stat()
            if stat.st_size <= 0:
                raise ValueError("file cache kosong")
            # On Windows, a read-only memmap keeps the .npy file mapped. Later
            # os.replace(tmp_path, path) cannot replace a mapped destination and
            # raises WinError 5 (Access is denied). Load a regular ndarray there;
            # the in-memory LRU below is already bounded, so this remains safe.
            mmap_mode = None if os.name == "nt" else "r"
            rows = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
            if (
                not isinstance(rows, np.ndarray)
                or rows.ndim != 1
                or rows.dtype != KLINE_CACHE_DTYPE
                or rows.shape[0] < int(count)
            ):
                try:
                    rows._mmap.close()
                except Exception:
                    pass
                raise ValueError("schema/ukuran cache tidak cocok")
            rows = rows[-int(count):]
            age_seconds = max(0.0, time.time() - stat.st_mtime)
            self._kline_disk_cache_hits += 1
            return time.monotonic() - age_seconds, rows
        except FileNotFoundError:
            self._kline_disk_cache_misses += 1
            return None
        except Exception as exc:
            self._kline_disk_cache_misses += 1
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            log.warning("[KLINE CACHE] disk cache %s diabaikan: %s", path.name, str(exc)[:120])
            return None

    def _prune_kline_disk_cache(self, keep_path: Path | None = None) -> None:
        """Bound ephemeral disk use without touching the in-memory/watch cache."""
        try:
            files = []
            total_bytes = 0
            for path in self._kline_disk_cache_dir.glob("*.npy"):
                try:
                    stat = path.stat()
                    files.append((stat.st_mtime, stat.st_size, path))
                    total_bytes += stat.st_size
                except OSError:
                    continue
            if total_bytes <= self._kline_disk_cache_max_bytes:
                return
            files.sort(key=lambda item: item[0])
            for _mtime, size, path in files:
                if total_bytes <= self._kline_disk_cache_max_bytes:
                    break
                if keep_path is not None and path == keep_path:
                    continue
                try:
                    path.unlink(missing_ok=True)
                    total_bytes -= size
                except OSError:
                    continue
        except Exception as exc:
            log.debug("[KLINE CACHE] prune disk dilewati: %s", str(exc)[:120])

    def _write_kline_disk_cache(self, key: tuple[str, str, int], rows: np.ndarray) -> None:
        """Atomically persist one compact history; failure never blocks Binance trading."""
        path = self._kline_disk_path(key)
        tmp_path: Path | None = None
        try:
            with self._kline_disk_cache_lock:
                self._kline_disk_cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=path.name + ".", suffix=".tmp",
                    dir=self._kline_disk_cache_dir, delete=False,
                ) as handle:
                    tmp_path = Path(handle.name)
                    np.save(handle, rows, allow_pickle=False)
                    handle.flush()
                os.replace(tmp_path, path)
                self._kline_disk_cache_writes += 1
                self._prune_kline_disk_cache(keep_path=path)
        except Exception as exc:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            log.warning("[KLINE CACHE] gagal menyimpan disk cache %s: %s", path.name, str(exc)[:120])

    def _adopt_disk_kline_cache(
        self, key: tuple[str, str, int], cached: tuple[float, np.ndarray]
    ) -> None:
        """Adopt a validated disk cache into the bounded RAM LRU."""
        old = self._kline_cache.pop(key, None)
        if old is not None:
            self._kline_cache_candles = max(0, self._kline_cache_candles - int(old[1].shape[0]))
        self._kline_cache[key] = cached
        self._kline_cache_candles += int(cached[1].shape[0])
        self._kline_cache.move_to_end(key)
        while (
            len(self._kline_cache) > self._kline_cache_max_entries
            or self._kline_cache_candles > self._kline_cache_max_candles
        ):
            _old_key, (_at, old_rows) = self._kline_cache.popitem(last=False)
            self._kline_cache_candles = max(0, self._kline_cache_candles - int(old_rows.shape[0]))

    async def get_klines(
        self, symbol: str, interval: str, count: int, interval_ms: int | None = None,
        *, force_refresh: bool = False,
    ) -> np.ndarray:
        """Single-flight wrapper; returns a read-only-by-convention compact kline array."""
        symbol = normalize_symbol(symbol)
        api_interval = self._binance_interval(interval)
        lock_key = (symbol, api_interval, 0)
        lock = self._kline_locks.setdefault(lock_key, asyncio.Lock())
        async with lock:
            result = await self._get_klines_locked(
                symbol, api_interval, count, interval_ms, force_refresh=force_refresh
            )
        if len(self._kline_locks) > 512:
            for old_key, old_lock in list(self._kline_locks.items()):
                if old_key != lock_key and not old_lock.locked():
                    self._kline_locks.pop(old_key, None)
                    if len(self._kline_locks) <= 384:
                        break
        return result

    async def get_klines_with_forming(
        self, symbol: str, interval: str, count: int, interval_ms: int | None = None,
    ) -> np.ndarray:
        """Return closed history plus the current row, preferring market WebSocket.

        The strategy consumes the current row's open timestamp/open price; all
        indicators, Supertrend and liquidity calculations use closed rows only.
        REST remains a safe fallback when this symbol has no current WS kline.
        """
        symbol = normalize_symbol(symbol)
        api_interval = self._binance_interval(interval)
        span_ms = int(interval_ms or self._interval_ms(api_interval))
        count = max(3, min(1500, int(count)))
        now_ms = int(time.time() * 1000)
        expected_open = (now_ms // span_ms) * span_ms

        forming = self._forming_klines.get((symbol, api_interval))
        if forming is not None and int(forming[0]) == expected_open:
            try:
                closed = await self.get_klines(symbol, api_interval, count - 1, span_ms)
            except BinanceRateLimitError:
                raise
            except Exception as exc:
                # Newly listed contracts may have fewer historical candles than
                # the requested window. Preserve the legacy fallback's shorter
                # ATR warm-up allowance instead of rejecting those symbols.
                if not getattr(exc, "insufficient_history", False):
                    raise
                closed = None
            if (
                isinstance(closed, np.ndarray)
                and closed.size >= count - 1
                and int(closed["t"][-1]) + span_ms == expected_open
            ):
                current = np.asarray([forming], dtype=KLINE_CACHE_DTYPE)
                return np.ascontiguousarray(np.concatenate((closed[-(count - 1):], current)))

        # Fallback when a current WS row has not arrived yet. Reuse the cached
        # closed history and fetch only the active candle (limit=1). On cold
        # start, keep the one-request full snapshot and seed the history cache.
        history_key = (symbol, api_interval, count - 1)
        has_history_cache = (
            (history_key in self._kline_cache
             and self._kline_cache[history_key][1].shape[0] >= count - 1)
            or self._kline_disk_path(history_key).exists()
            or any(
                cache_key[0] == symbol and cache_key[1] == api_interval
                and cached[1].shape[0] >= count - 1
                for cache_key, cached in self._kline_cache.items()
            )
        )

        if has_history_cache:
            try:
                closed = await self.get_klines(symbol, api_interval, count - 1, span_ms)
            except BinanceRateLimitError:
                raise
            except Exception as exc:
                if not getattr(exc, "insufficient_history", False):
                    raise
                closed = None

            if (
                isinstance(closed, np.ndarray)
                and closed.size >= count - 1
                and int(closed["t"][-1]) + span_ms == expected_open
            ):
                payload = None
                current = np.empty(0, dtype=KLINE_CACHE_DTYPE)
                for attempt in range(2):
                    request_now = int(time.time() * 1000)
                    payload = await self._get("/fapi/v1/klines", {
                        "symbol": symbol, "interval": api_interval,
                        "endTime": request_now + 20_000, "limit": 1,
                    })
                    if isinstance(payload, list) and payload:
                        current = self._pack_kline_rows(
                            payload, span_ms, request_now + 20_000, include_forming=True
                        )
                        if current.size and int(current["t"][-1]) == expected_open:
                            break
                    if attempt == 0:
                        await asyncio.sleep(0.8)
                        now_ms = int(time.time() * 1000)
                        new_expected_open = (now_ms // span_ms) * span_ms
                        if new_expected_open != expected_open:
                            # The candle we just crossed is now closed; update
                            # the history tail before appending the next active row.
                            closed = await self.get_klines(
                                symbol, api_interval, count - 1, span_ms
                            )
                        expected_open = new_expected_open
                if current.size and int(current["t"][-1]) == expected_open:
                    return np.ascontiguousarray(
                        np.concatenate((closed[-(count - 1):], current[-1:]))
                    )
                raise RuntimeError(
                    f"Candle aktif belum tersedia untuk {symbol} {api_interval}; "
                    f"expected_open={expected_open}; REST tidak mengembalikan candle berjalan."
                )

        # Cold start or a newly-listed symbol without a reusable full history:
        # one REST snapshot supplies closed history and the current row.
        payload = None
        rows = np.empty(0, dtype=KLINE_CACHE_DTYPE)
        for attempt in range(2):
            request_now = int(time.time() * 1000)
            try:
                payload = await self._get("/fapi/v1/klines", {
                    "symbol": symbol, "interval": api_interval,
                    "endTime": request_now + 20_000, "limit": count,
                })
                if isinstance(payload, list) and payload:
                    rows = self._pack_kline_rows(
                        payload, span_ms, request_now + 20_000, include_forming=True
                    )
                    if rows.size and int(rows["t"][-1]) == expected_open:
                        break
            except BinanceRateLimitError:
                # Never retry a request after Binance has explicitly limited the IP.
                raise
            except Exception:
                if attempt == 1:
                    raise
            if attempt == 0:
                await asyncio.sleep(0.8)
                now_ms = int(time.time() * 1000)
                expected_open = (now_ms // span_ms) * span_ms

        if not isinstance(payload, list) or not payload:
            raise RuntimeError(f"Binance tidak mengembalikan kline live untuk {symbol} {api_interval}.")
        if rows.size < max(3, min(count - 1, volumatic.ATR_LENGTH + 5)):
            raise RuntimeError(f"Kline live {symbol} {api_interval} tidak cukup: {rows.shape[0]}/{count}.")
        last_open = int(rows["t"][-1])
        now_ms = int(time.time() * 1000)
        if last_open != expected_open or not (last_open <= now_ms < last_open + span_ms):
            raise RuntimeError(
                f"Candle aktif belum dipublikasikan Binance untuk {symbol} {api_interval}; "
                f"last_open={last_open}, expected_open={expected_open}, now={now_ms}; retry pada batch berikutnya."
            )

        # Seed a bounded closed-history cache from the same fallback payload so
        # later no-WS cycles need only the single current-candle endpoint call.
        closed_rows = np.ascontiguousarray(rows[:-1])
        if closed_rows.shape[0] >= count - 1:
            cache_key = (symbol, api_interval, count - 1)
            self._store_kline_cache(cache_key, closed_rows[-(count - 1):])
            await asyncio.to_thread(
                self._write_kline_disk_cache, cache_key, self._kline_cache[cache_key][1]
            )
        return rows

    async def _get_klines_locked(
        self, symbol: str, interval: str, count: int, interval_ms: int | None = None,
        *, force_refresh: bool = False,
    ) -> np.ndarray:
        """Return closed USD-M klines in compact arrays; refresh only bars after the cached close."""
        symbol = normalize_symbol(symbol)
        count = max(1, min(20_000, int(count)))
        api_interval = self._binance_interval(interval)
        span_ms = int(interval_ms or self._interval_ms(api_interval))
        key = (symbol, api_interval, count)
        cached_exact = self._kline_cache.get(key)
        if cached_exact is None:
            disk_cached = self._load_kline_disk_cache(key, count)
            if disk_cached is not None:
                self._adopt_disk_kline_cache(key, disk_cached)
                cached_exact = self._kline_cache.get(key)

        if cached_exact is None and not force_refresh:
            reusable = [
                (cache_key, cache_value)
                for cache_key, cache_value in self._kline_cache.items()
                if cache_key[0] == symbol and cache_key[1] == api_interval
                and cache_value[1].shape[0] >= count
            ]
            if reusable:
                _cache_key, cached = min(reusable, key=lambda item: item[1][1].shape[0])
                self._store_kline_cache(key, cached[1][-count:])
                cached_exact = self._kline_cache.get(key)

        if cached_exact and cached_exact[1].shape[0] >= count:
            cached_rows, ws_gap, ws_changed = self._merge_ws_closed_klines(
                symbol, api_interval, cached_exact[1][-count:], count
            )
            if ws_changed:
                self._store_kline_cache(key, cached_rows)
                cached_exact = self._kline_cache.get(key)
                cached_rows = cached_exact[1][-count:]

            now_ms = int(time.time() * 1000)
            expected_last_open = (now_ms // span_ms) * span_ms - span_ms
            cache_is_current = int(cached_rows["t"][-1]) >= expected_last_open
            if not force_refresh and cache_is_current and not ws_gap:
                self._kline_cache.move_to_end(key)
                return cached_rows

            last_open = int(cached_rows["t"][-1])
            next_open = last_open + span_ms
            additions: list[np.ndarray] = []
            attempts = 0
            while next_open + span_ms <= now_ms and attempts < 24:
                attempts += 1
                remaining_closed = max(1, (now_ms - next_open) // span_ms)
                page_limit = min(1000, int(remaining_closed) + 1)
                payload = await self._get("/fapi/v1/klines", {
                    "symbol": symbol, "interval": api_interval,
                    "startTime": next_open, "limit": page_limit,
                })
                if not isinstance(payload, list) or not payload:
                    break
                page_max_open = None
                for row in payload:
                    if isinstance(row, (list, tuple)) and row:
                        try:
                            open_ms = int(row[0])
                        except (TypeError, ValueError):
                            continue
                        page_max_open = open_ms if page_max_open is None else max(page_max_open, open_ms)
                packed = self._pack_kline_rows(payload, span_ms, now_ms)
                if packed.size:
                    packed = packed[packed["t"] >= next_open]
                    if packed.size:
                        additions.append(packed)
                if page_max_open is None or len(payload) < page_limit:
                    break
                advanced = page_max_open + span_ms
                if advanced <= next_open:
                    break
                next_open = advanced

            rows = self._merge_packed_klines([cached_rows, *additions], count)
            rows, _ws_gap_after_rest, _ws_changed_after_rest = self._merge_ws_closed_klines(
                symbol, api_interval, rows, count
            )
            if rows.shape[0] < count:
                raise RuntimeError(
                    f"Riwayat incremental Binance {symbol} {api_interval} tidak lengkap: "
                    f"{rows.shape[0]}/{count}."
                )
            self._store_kline_cache(key, rows)
            # Never block the Telegram/WebSocket event loop with disk writes.
            await asyncio.to_thread(self._write_kline_disk_cache, key, self._kline_cache[key][1])
            return self._kline_cache[key][1]

        # Cold start: convert each response page immediately and discard JSON strings.
        now_ms = int(time.time() * 1000)
        end_ms = now_ms
        chunks: list[np.ndarray] = []
        collected_count = 0
        attempts = 0
        while collected_count < count and attempts < 24:
            attempts += 1
            remaining = count - collected_count
            page_limit = min(1000, max(1, remaining + 1))
            payload = await self._get("/fapi/v1/klines", {
                "symbol": symbol, "interval": api_interval,
                "endTime": end_ms, "limit": page_limit,
            })
            if not isinstance(payload, list) or not payload:
                break
            oldest: int | None = None
            for row in payload:
                if isinstance(row, (list, tuple)) and row:
                    try:
                        open_ms = int(row[0])
                    except (TypeError, ValueError):
                        continue
                    oldest = open_ms if oldest is None else min(oldest, open_ms)
            packed = self._pack_kline_rows(payload, span_ms, now_ms)
            if packed.size:
                chunks.append(packed)
                collected_count += int(packed.shape[0])
            if oldest is None or len(payload) < page_limit:
                break
            end_ms = oldest - 1

        rows = self._merge_packed_klines(chunks, count)
        if rows.shape[0] < count:
            error = RuntimeError(
                f"Riwayat closed candle Binance {symbol} {api_interval}: {rows.shape[0]}/{count}; "
                "History OHLCV H4/D1 belum mencukupi warm-up indikator."
            )
            error.insufficient_history = True
            raise error

        self._store_kline_cache(key, rows)
        await asyncio.to_thread(self._write_kline_disk_cache, key, self._kline_cache[key][1])
        # Return the cached ndarray itself; consumer promises not to mutate it.
        return self._kline_cache[key][1]

    def kline_cache_stats(self) -> dict[str, Any]:
        approx_bytes = sum(int(rows.nbytes) for _at, rows in self._kline_cache.values())
        disk_bytes = 0
        disk_entries = 0
        try:
            for path in self._kline_disk_cache_dir.glob("*.npy"):
                try:
                    disk_bytes += path.stat().st_size
                    disk_entries += 1
                except OSError:
                    continue
        except OSError:
            pass
        return {
            "entries": len(self._kline_cache),
            "candles": self._kline_cache_candles,
            "approx_mb": round(approx_bytes / (1024 * 1024), 3),
            "max_entries": self._kline_cache_max_entries,
            "max_candles": self._kline_cache_max_candles,
            "disk_entries": disk_entries,
            "disk_mb": round(disk_bytes / (1024 * 1024), 2),
            "disk_max_mb": round(self._kline_disk_cache_max_bytes / (1024 * 1024), 0),
            "disk_hits": self._kline_disk_cache_hits,
            "disk_misses": self._kline_disk_cache_misses,
            "disk_writes": self._kline_disk_cache_writes,
            "owner": "main.py",
            "storage": "numpy_structured_array + bounded_ephemeral_npy_cache",
        }

    def trim_kline_cache(self, fraction: float = 0.5, pair: str | None = None) -> int:
        """Evict least-recently-used kline entries; fraction=1 clears matching entries."""
        fraction = max(0.0, min(1.0, float(fraction)))
        keys = [key for key in self._kline_cache if pair is None or key[0] == normalize_symbol(pair)]
        target = len(keys) if fraction >= 1.0 else int(len(keys) * fraction)
        removed = 0
        for key in keys[:target]:
            old = self._kline_cache.pop(key, None)
            if old is not None:
                self._kline_cache_candles = max(0, self._kline_cache_candles - int(old[1].shape[0]))
                removed += 1
        return removed

    def retain_kline_cache_for(self, pairs: set[str] | list[str] | tuple[str, ...]) -> int:
        """Keep only bounded compact OHLCV arrays for active H4/D1 analysis."""
        keep = {normalize_symbol(value) for value in pairs if value}
        removed = 0
        for key in list(self._kline_cache.keys()):
            if key[0] in keep:
                continue
            old = self._kline_cache.pop(key, None)
            if old is not None:
                self._kline_cache_candles = max(0, self._kline_cache_candles - int(old[1].shape[0]))
                removed += 1
        return removed

    async def get_exchange_info(self, force: bool = False) -> dict[str, SymbolMeta]:
        now = time.monotonic()
        if (
            not force
            and self._exchange_info_cache is not None
            and now - self._exchange_info_cached_at <= self._exchange_info_cache_ttl_seconds
        ):
            return self._exchange_info_cache

        payload = await self._get("/fapi/v1/exchangeInfo")

        result: dict[str, SymbolMeta] = {}

        for raw in payload.get("symbols", []):
            symbol = str(raw.get("symbol") or "").upper()
            status = str(raw.get("status") or "")
            contract_type = str(raw.get("contractType") or "")
            quote_asset = str(raw.get("quoteAsset") or "")

            # Bot ini fokus pada USDⓈ-M USDT perpetual.
            if status != "TRADING":
                continue

            if contract_type != "PERPETUAL":
                continue

            if quote_asset != "USDT":
                continue

            tick_size = Decimal("0")
            min_price = Decimal("0")
            max_price = Decimal("0")
            min_qty = Decimal("0")
            max_qty = Decimal("0")
            market_max_qty = Decimal("0")
            step_size = Decimal("0")
            min_notional = Decimal("0")
            max_notional = Decimal("0")

            for filt in raw.get("filters", []):
                kind = str(filt.get("filterType") or "")
                if kind == "PRICE_FILTER":
                    tick_size = Decimal(str(filt.get("tickSize") or "0"))
                    min_price = Decimal(str(filt.get("minPrice") or "0"))
                    max_price = Decimal(str(filt.get("maxPrice") or "0"))
                elif kind in {"LOT_SIZE", "MARKET_LOT_SIZE"}:
                    candidate_step = Decimal(str(filt.get("stepSize") or "0"))
                    candidate_min = Decimal(str(filt.get("minQty") or "0"))
                    candidate_max = Decimal(str(filt.get("maxQty") or "0"))
                    if kind == "MARKET_LOT_SIZE":
                        market_max_qty = candidate_max
                    if kind == "LOT_SIZE" or step_size <= 0:
                        step_size = candidate_step
                        min_qty = candidate_min
                        max_qty = candidate_max
                elif kind in {"MIN_NOTIONAL", "NOTIONAL"}:
                    min_notional = Decimal(str(
                        filt.get("minNotional")
                        or filt.get("notional")
                        or "0"
                    ))
                    max_notional = Decimal(str(filt.get("maxNotional") or "0"))

            result[symbol] = SymbolMeta(
                symbol=symbol,
                status=status,
                contract_type=contract_type,
                quote_asset=quote_asset,
                tick_size=tick_size,
                min_price=min_price,
                max_price=max_price,
                min_qty=min_qty,
                max_qty=max_qty,
                step_size=step_size,
                min_notional=min_notional,
                max_notional=max_notional,
                market_max_qty=market_max_qty,
                quantity_precision=int(raw.get("quantityPrecision") or 0),
                price_precision=int(raw.get("pricePrecision") or 0),
            )

        self._exchange_info_cache = result
        self._exchange_info_cached_at = time.monotonic()
        return result

    async def get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        """Return current Binance symbol filters used by REAL quantity validation."""
        symbols = await self.get_exchange_info(force=False)
        meta = symbols.get(normalize_symbol(symbol))
        if meta is None:
            raise ValueError(f"{symbol} tidak ditemukan di Binance exchangeInfo.")
        return {
            "step_size": meta.step_size,
            "min_qty": meta.min_qty,
            "max_qty": meta.max_qty,
            "market_max_qty": meta.market_max_qty,
            "min_notional": meta.min_notional,
            "max_notional": meta.max_notional,
            "min_price": meta.min_price,
            "max_price": meta.max_price,
            "tick_size": meta.tick_size,
            "quantity_precision": meta.quantity_precision,
            "price_precision": meta.price_precision,
        }

    async def get_price(self, symbol: str) -> Decimal:
        payload = await self._get(
            "/fapi/v2/ticker/price",
            params={"symbol": symbol},
        )

        return parse_decimal(str(payload["price"]))

    async def get_server_time(self) -> int:
        """One-weight public probe used only to refresh Binance IP-weight headers."""
        payload = await self._get("/fapi/v1/time")
        if not isinstance(payload, dict) or payload.get("serverTime") is None:
            raise RuntimeError("Binance server time response tidak valid.")
        return int(payload["serverTime"])

    async def get_24h_tickers(self, force: bool = False) -> list[dict[str, Any]]:
        now = time.monotonic()
        if (
            not force
            and self._ticker_24h_cache is not None
            and now - self._ticker_24h_cached_at <= self._ticker_24h_cache_ttl_seconds
        ):
            return self._ticker_24h_cache
        payload = await self._get("/fapi/v1/ticker/24hr")
        if not isinstance(payload, list):
            raise RuntimeError("Binance 24h ticker response bukan list.")
        self._ticker_24h_cache = [dict(row) for row in payload if isinstance(row, dict)]
        self._ticker_24h_cached_at = time.monotonic()
        return self._ticker_24h_cache


# ============================================================
# BINANCE REAL EXECUTION
# ============================================================
# REAL_EXECUTION_ENABLED is an environment-level kill switch. The bot starts
# with real_mode=False every session and requires the explicit /real on command.

class BinanceAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: int | None = None,
        endpoint: str = "",
        retry_after_seconds: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.endpoint = endpoint
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)

    def __str__(self) -> str:
        parts = []
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        if self.code is not None:
            parts.append(f"code {self.code}")
        if self.endpoint:
            parts.append(self.endpoint)
        prefix = " | ".join(parts)
        if prefix:
            return f"{prefix}: {super().__str__()}"
        return super().__str__()


class BinanceRateLimitError(BinanceAPIError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: int | None,
        endpoint: str,
        server_cooldown_seconds: float,
        bot_cooldown_seconds: float,
        retry_after_known: bool = False,
    ) -> None:
        self.server_cooldown_seconds = server_cooldown_seconds
        self.bot_cooldown_seconds = bot_cooldown_seconds
        self.retry_after_known = bool(retry_after_known)
        super().__init__(
            message,
            status_code=status_code,
            code=code,
            endpoint=endpoint,
            retry_after_seconds=(
                server_cooldown_seconds
                if self.retry_after_known
                else None
            ),
        )


class MarketEntryRejected(Exception):
    """Definitive refusal: no MARKET order exists for this setup and none will be sent."""

    def __init__(self, reason: str, detail: str = "", *, result: str = "CANCELED", notify: bool = True) -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason
        self.result = result
        self.notify = notify


class BinanceRealClient:
    """Signed USD-M Futures client used only when /real is ON."""

    def __init__(self) -> None:
        self.base_url = BINANCE_REAL_REST_BASE
        self.api_key = BINANCE_API_KEY
        self.api_secret = BINANCE_API_SECRET
        self.recv_window = BINANCE_RECV_WINDOW
        self._time_offset_ms = 0
        self._time_sync_at_monotonic = 0.0
        self._time_sync_rtt_ms: float | None = None
        self._time_sync_lock = asyncio.Lock()
        self._time_sync_interval_seconds = 600.0
        self._cooldown_until = 0.0
        self._server_cooldown_until = 0.0
        self._early_cooldown_until = 0.0
        self._request_lock = asyncio.Lock()
        self._dual_side_position: bool | None = None
        # Snapshot terakhir dari Futures balance. Semua consumer lokal (autostop
        # dan /trade) membaca cache ini agar tidak melakukan request berulang.
        self.last_balance: dict[str, Any] | None = None
        self.last_balance_at: float | None = None
        # Position snapshot per symbol; dipakai sebagai filter awal sebelum final
        # safety-check fresh pada saat real entry.
        self.last_positions: dict[str, list[dict[str, Any]]] = {}
        self.last_positions_at: dict[str, float] = {}

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.monotonic())

    @property
    def server_cooldown_remaining(self) -> float:
        return max(0.0, self._server_cooldown_until - time.monotonic())

    @property
    def early_cooldown_remaining(self) -> float:
        return max(0.0, self._early_cooldown_until - time.monotonic())

    def apply_cooldown(self, server_seconds: float, bot_seconds: float) -> None:
        now = time.monotonic()
        self._cooldown_until = max(self._cooldown_until, now + bot_seconds)
        self._early_cooldown_until = max(self._early_cooldown_until, now + server_seconds)
        self._server_cooldown_until = max(
            self._server_cooldown_until,
            now + server_seconds + BINANCE_RATE_LIMIT_PRIORITY_SAFETY_SECONDS,
        )

    def _signed_query(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build a signed payload with a timestamp as close as possible to send time."""
        result = dict(params)
        # Always refresh timestamp here. _request calls this inside request_once,
        # after all asyncio locks and Binance pacing delays have completed.
        result["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
        result.setdefault("recvWindow", self.recv_window)

        query = urlencode(result)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        result["signature"] = signature
        return result

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return max(0.0, value)

    @staticmethod
    def _extract_binance_error(response: requests.Response) -> tuple[int | None, str]:
        try:
            body = response.json()
        except ValueError:
            body = {}
        code = body.get("code") if isinstance(body, dict) else None
        msg = body.get("msg") if isinstance(body, dict) else None
        return (
            int(code) if isinstance(code, (int, float, str)) and str(code).lstrip("-").isdigit() else None,
            str(msg or response.text[:700] or "Unknown Binance API error"),
        )

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = True,
        retry_timestamp_error: bool = True,
        _timing: dict[str, float] | None = None,
    ) -> Any:
        if not REAL_EXECUTION_ENABLED:
            raise BinanceAPIError(
                "REAL_EXECUTION_ENABLED=0; private Binance API dinonaktifkan oleh environment.",
                endpoint=path,
            )
        if signed and not self.configured:
            raise BinanceAPIError(
                "BINANCE_API_KEY / BINANCE_API_SECRET belum tersedia di environment.",
                endpoint=path,
            )

        if signed and retry_timestamp_error:
            await self._ensure_time_sync()

        remaining = max(
            BINANCE_COOLDOWN.remaining(),
            (self.early_cooldown_remaining if REAL_EARLY.get() else self.server_cooldown_remaining)
            if REAL_PRIORITY.get() else self.cooldown_remaining,
        )
        if remaining > 0:
            blocked = BinanceRateLimitError(
                f"REST Binance sedang ditahan oleh rate-limit cooldown ({remaining:.0f}s tersisa).",
                status_code=429,
                code=None,
                endpoint=path,
                server_cooldown_seconds=remaining,
                bot_cooldown_seconds=remaining,
                retry_after_known=False,
            )
            blocked.internal = True
            raise blocked

        base_params = dict(params or {})
        headers = {"Accept": "application/json"}
        if signed:
            headers["X-MBX-APIKEY"] = self.api_key

        def request_once() -> requests.Response:
            # Do not create the signed timestamp before acquiring the request
            # locks/pacing slot: a busy scanner can otherwise make it stale
            # while the request waits locally.
            payload = self._signed_query(base_params) if signed else base_params
            kwargs: dict[str, Any] = {
                "headers": headers,
                "timeout": HTTP_REQUEST_SECONDS,
            }
            if method.upper() in {"GET", "DELETE"}:
                kwargs["params"] = payload
            else:
                kwargs["data"] = payload
                kwargs["headers"] = {
                    **headers,
                    "Content-Type": "application/x-www-form-urlencoded",
                }
            if _timing is not None:
                _timing["started_wall_ms"] = time.time() * 1000.0
                _timing["started_monotonic"] = time.monotonic()
            try:
                response = requests.request(
                    method.upper(),
                    f"{self.base_url}{path}",
                    proxies=PROXY_MANAGER.request_proxies(),
                    **kwargs,
                )
            finally:
                if _timing is not None:
                    _timing["finished_monotonic"] = time.monotonic()
            return response

        async with self._request_lock:
            async with BINANCE_HTTP_LOCK:
                cooldown = BINANCE_COOLDOWN.remaining()
                if cooldown > 0:
                    raise _local_rate_limit_error(path)
                spacing = BINANCE_COOLDOWN.wait_for_request_slot(BINANCE_PUBLIC_REQUEST_SPACING_SECONDS)
                if spacing > 0:
                    await asyncio.sleep(spacing)
                if BINANCE_COOLDOWN.remaining() > 0:
                    raise _local_rate_limit_error(path)
                try:
                    response = await asyncio.to_thread(request_once)
                except requests.RequestException as exc:
                    PROXY_MANAGER.observe_transport_error(str(exc))
                    raise BinanceAPIError(
                        f"Transport error Binance: {exc}", endpoint=path,
                    ) from exc
                except (OSError, TimeoutError) as exc:
                    raise BinanceAPIError(
                        f"Transport error Binance: {exc}", endpoint=path,
                    ) from exc
                BINANCE_COOLDOWN.record_request()

                code, msg = self._extract_binance_error(response)
                API_STATS.record(
                    path, response.status_code,
                    response.headers.get("X-MBX-USED-WEIGHT-1M"),
                    estimate_request_weight(path, base_params),
                )
                is_rate_limited = (
                    response.status_code in {418, 429}
                    or code in {-1003, -1015}
                    or "too many requests" in str(msg).lower()
                    or "way too many requests" in str(msg).lower()
                )
                if is_rate_limited:
                    server_seconds, retry_known = rate_limit_wait(response, self._time_offset_ms)
                    bot_seconds = max(1.0, server_seconds + rate_limit_safety(response.status_code))
                    self.apply_cooldown(server_seconds, bot_seconds)
                    BINANCE_COOLDOWN.extend(bot_seconds, f"HTTP {response.status_code} {path}: {msg}")
                    PROXY_MANAGER.observe_http(response.status_code, bot_seconds)
                    raise BinanceRateLimitError(
                        msg,
                        status_code=response.status_code or 429,
                        code=code,
                        endpoint=path,
                        server_cooldown_seconds=server_seconds,
                        bot_cooldown_seconds=bot_seconds,
                        retry_after_known=retry_known,
                    )

        PROXY_MANAGER.observe_http(response.status_code)
        if response.status_code >= 400:
            code, msg = self._extract_binance_error(response)
            if code == -1021 and retry_timestamp_error:
                log.warning(
                    "[BINANCE TIME] received -1021 on %s; resynchronizing server clock and retrying once.",
                    path,
                )
                try:
                    offset_ms, rtt_ms = await self.sync_server_time(force=True, samples=3)
                    log.info(
                        "[BINANCE TIME] resynchronized after -1021 | offset=%+dms | best_rtt=%.1fms | retry=%s",
                        offset_ms, rtt_ms, path,
                    )
                    return await self._request(
                        method,
                        path,
                        base_params,
                        signed=signed,
                        retry_timestamp_error=False,
                    )
                except BinanceRateLimitError:
                    raise
                except Exception as sync_exc:
                    raise BinanceAPIError(
                        f"{msg}; sinkronisasi waktu Binance gagal: {type(sync_exc).__name__}: {sync_exc}",
                        status_code=response.status_code,
                        code=code,
                        endpoint=path,
                    ) from sync_exc

            raise BinanceAPIError(
                msg,
                status_code=response.status_code,
                code=code,
                endpoint=path,
            )

        try:
            return response.json()
        except ValueError as exc:
            raise BinanceAPIError(
                "Respons Binance bukan JSON valid.",
                status_code=response.status_code,
                endpoint=path,
            ) from exc

    async def sync_server_time(
        self, *, force: bool = False, samples: int = 3,
    ) -> tuple[int, float]:
        """Estimate Binance clock offset using request-midpoint timing.

        Each sample records its timestamps inside the worker thread immediately
        around the actual HTTP call, so local queue/pacing delays are excluded.
        The lowest-RTT sample is selected to minimize proxy/network asymmetry.
        """
        sample_count = max(1, min(5, int(samples)))
        async with self._time_sync_lock:
            age = time.monotonic() - self._time_sync_at_monotonic
            if (
                not force
                and self._time_sync_at_monotonic > 0
                and age < self._time_sync_interval_seconds
            ):
                return self._time_offset_ms, float(self._time_sync_rtt_ms or 0.0)

            best: tuple[float, int, float] | None = None
            for _ in range(sample_count):
                timing: dict[str, float] = {}
                body = await self._request(
                    "GET", "/fapi/v1/time", {}, signed=False,
                    retry_timestamp_error=False, _timing=timing,
                )
                try:
                    server_ms = int(body["serverTime"])
                    started_wall_ms = float(timing["started_wall_ms"])
                    started_mono = float(timing["started_monotonic"])
                    finished_mono = float(timing["finished_monotonic"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise BinanceAPIError(
                        "Sampel sinkronisasi waktu Binance tidak lengkap.",
                        endpoint="/fapi/v1/time",
                    ) from exc

                rtt_ms = max(0.0, (finished_mono - started_mono) * 1000.0)
                local_midpoint_ms = started_wall_ms + (rtt_ms / 2.0)
                offset_ms = int(round(server_ms - local_midpoint_ms))
                if best is None or rtt_ms < best[0]:
                    best = (rtt_ms, offset_ms, float(server_ms))

            if best is None:
                raise BinanceAPIError(
                    "Binance tidak memberikan sampel waktu yang valid.",
                    endpoint="/fapi/v1/time",
                )

            self._time_sync_rtt_ms = best[0]
            self._time_offset_ms = best[1]
            self._time_sync_at_monotonic = time.monotonic()
            log.info(
                "[BINANCE TIME] synchronized | offset=%+dms | best_rtt=%.1fms | samples=%d",
                self._time_offset_ms, self._time_sync_rtt_ms, sample_count,
            )
            return self._time_offset_ms, self._time_sync_rtt_ms

    async def _ensure_time_sync(self) -> None:
        """Refresh the Binance clock offset periodically before signed requests."""
        age = time.monotonic() - self._time_sync_at_monotonic
        if self._time_sync_at_monotonic <= 0 or age >= self._time_sync_interval_seconds:
            await self.sync_server_time(force=False, samples=1)

    async def _public_server_time(self) -> int:
        # Backward-compatible helper; the robust synchronizer owns offset updates.
        body = await self._request(
            "GET", "/fapi/v1/time", {}, signed=False,
            retry_timestamp_error=False,
        )
        return int(body["serverTime"])

    async def _user_stream_request(self, method: str, listen_key: str | None = None) -> dict[str, Any]:
        """Create/renew USD-M listenKey; API-key header only, no HMAC signature."""
        if not REAL_EXECUTION_ENABLED:
            raise BinanceAPIError("REAL_EXECUTION_ENABLED=0; user stream REAL dinonaktifkan.", endpoint="/fapi/v1/listenKey")
        if not self.configured:
            raise BinanceAPIError("BINANCE_API_KEY / BINANCE_API_SECRET belum tersedia.", endpoint="/fapi/v1/listenKey")
        path = "/fapi/v1/listenKey"
        params = {"listenKey": listen_key} if method.upper() == "PUT" and listen_key else {}
        async with self._request_lock:
            async with BINANCE_HTTP_LOCK:
                cooldown = BINANCE_COOLDOWN.remaining()
                if cooldown > 0:
                    raise _local_rate_limit_error(path)
                spacing = BINANCE_COOLDOWN.wait_for_request_slot(BINANCE_PUBLIC_REQUEST_SPACING_SECONDS)
                if spacing > 0:
                    await asyncio.sleep(spacing)
                if BINANCE_COOLDOWN.remaining() > 0:
                    raise _local_rate_limit_error(path)

                def request_once() -> requests.Response:
                    return requests.request(
                        method.upper(), f"{self.base_url}{path}",
                        headers={"X-MBX-APIKEY": self.api_key, "Accept": "application/json"},
                        params=params or None,
                        proxies=PROXY_MANAGER.request_proxies(),
                        timeout=HTTP_REQUEST_SECONDS,
                    )

                try:
                    response = await asyncio.to_thread(request_once)
                except requests.RequestException as exc:
                    PROXY_MANAGER.observe_transport_error(str(exc))
                    raise BinanceAPIError(f"User stream transport error: {exc}", endpoint=path) from exc
                BINANCE_COOLDOWN.record_request()

                code, message = self._extract_binance_error(response)
                API_STATS.record(
                    path, response.status_code,
                    response.headers.get("X-MBX-USED-WEIGHT-1M"),
                    estimate_request_weight(path, params),
                )
                is_rate_limited = (
                    response.status_code in {418, 429}
                    or code in {-1003, -1015}
                    or "too many requests" in str(message).lower()
                    or "way too many requests" in str(message).lower()
                )
                if is_rate_limited:
                    server_seconds, retry_known = rate_limit_wait(response, self._time_offset_ms)
                    bot_seconds = max(1.0, server_seconds + rate_limit_safety(response.status_code))
                    self.apply_cooldown(server_seconds, bot_seconds)
                    BINANCE_COOLDOWN.extend(bot_seconds, f"HTTP {response.status_code} {path}: {message}")
                    PROXY_MANAGER.observe_http(response.status_code, bot_seconds)
                    raise BinanceRateLimitError(
                        message, status_code=response.status_code or 429, code=code, endpoint=path,
                        server_cooldown_seconds=server_seconds, bot_cooldown_seconds=bot_seconds,
                        retry_after_known=retry_known,
                    )
        PROXY_MANAGER.observe_http(response.status_code)
        if response.status_code >= 400:
            code, message = self._extract_binance_error(response)
            raise BinanceAPIError(message, status_code=response.status_code, code=code, endpoint=path)
        try:
            payload = response.json()
        except ValueError as exc:
            raise BinanceAPIError("User stream response bukan JSON valid.", endpoint=path) from exc
        if not isinstance(payload, dict):
            raise BinanceAPIError("User stream response tidak valid.", endpoint=path)
        return payload

    async def create_user_stream(self) -> str:
        payload = await self._user_stream_request("POST")
        listen_key = str(payload.get("listenKey") or "").strip()
        if not listen_key:
            raise BinanceAPIError("Binance tidak mengembalikan listenKey.", endpoint="/fapi/v1/listenKey")
        return listen_key

    async def keepalive_user_stream(self, listen_key: str) -> str:
        payload = await self._user_stream_request("PUT", listen_key)
        return str(payload.get("listenKey") or listen_key)

    async def get_futures_balance(self) -> list[dict[str, Any]]:
        """Read USD-M Futures balance and refresh the local USDT cache."""
        payload = await self._request("GET", "/fapi/v3/balance")
        if not isinstance(payload, list):
            raise BinanceAPIError(
                "Respons /fapi/v3/balance tidak berbentuk list.",
                endpoint="/fapi/v3/balance",
            )
        rows = [item for item in payload if isinstance(item, dict)]
        usdt = next(
            (row for row in rows if str(row.get("asset") or "").upper() == "USDT"),
            None,
        )
        if usdt is not None:
            self.last_balance = dict(usdt)
            self.last_balance_at = time.monotonic()
        return rows

    def cached_usdt_balance(self) -> dict[str, Any] | None:
        """Return the latest Binance USDT balance without making a REST request."""
        return dict(self.last_balance) if self.last_balance is not None else None

    def cached_position(self, symbol: str, desired_side: str | None = None) -> dict[str, Any] | None:
        """Return a fresh cached non-zero position for symbol/direction, if known."""
        key = symbol.upper()
        cached_at = self.last_positions_at.get(key)
        if cached_at is None or time.monotonic() - cached_at > POSITION_CACHE_MAX_AGE_SECONDS:
            return None
        rows = self.last_positions.get(key, [])
        if not rows:
            return None
        dual = bool(self._dual_side_position)
        target_side = (
            ("LONG" if desired_side == "BUY" else "SHORT")
            if dual and desired_side in {"BUY", "SELL"}
            else None
        )
        for item in rows:
            if target_side is not None and str(item.get("positionSide") or "BOTH") != target_side:
                continue
            amount = item.get("_position_amount_decimal")
            if not isinstance(amount, Decimal):
                try:
                    amount = parse_signed_decimal(str(amount or item.get("positionAmt") or "0"))
                except (InvalidOperation, ValueError):
                    continue
            if amount == 0:
                continue
            if not dual and desired_side in {"BUY", "SELL"}:
                if desired_side == "BUY" and amount <= 0:
                    continue
                if desired_side == "SELL" and amount >= 0:
                    continue
            return dict(item)
        return None

    async def get_account_info(self) -> dict[str, Any]:
        payload = await self._request("GET", "/fapi/v2/account")
        if not isinstance(payload, dict):
            raise BinanceAPIError(
                "Respons /fapi/v2/account tidak berbentuk object.",
                endpoint="/fapi/v2/account",
            )
        positions = payload.get("positions") or []
        cached_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for item in positions:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            try:
                amount = parse_signed_decimal(str(item.get("positionAmt") or "0"))
            except ValueError:
                continue
            if amount == 0:
                continue
            copied = dict(item)
            copied["_abs_position_amt"] = abs(amount)
            copied["_position_amount_decimal"] = amount
            cached_by_symbol.setdefault(symbol, []).append(copied)
        now_mono = time.monotonic()
        self.last_positions = cached_by_symbol
        self.last_positions_at = {symbol: now_mono for symbol in cached_by_symbol}
        sides = {
            str(item.get("positionSide") or "")
            for item in positions
            if isinstance(item, dict)
        }
        if {"LONG", "SHORT"} & sides:
            self._dual_side_position = True
        elif "BOTH" in sides:
            self._dual_side_position = False
        return payload

    async def get_account_config(self) -> dict[str, Any]:
        payload = await self._request("GET", "/fapi/v1/accountConfig")
        if isinstance(payload, dict) and "dualSidePosition" in payload:
            self._dual_side_position = bool(payload["dualSidePosition"])
        return payload

    async def ensure_position_mode(self) -> bool:
        if self._dual_side_position is None:
            await self.get_account_config()
        return bool(self._dual_side_position)

    async def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": leverage},
        )
        if not isinstance(payload, dict):
            raise BinanceAPIError(
                "Respons leverage Binance tidak valid.",
                endpoint="/fapi/v1/leverage",
            )
        return payload

    async def get_order(self, symbol: str, *, order_id: int | None = None, client_order_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("order_id atau client_order_id harus tersedia.")
        payload = await self._request("GET", "/fapi/v1/order", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons order Binance tidak valid.", endpoint="/fapi/v1/order")
        return payload

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else None
        payload = await self._request(
            "GET",
            "/fapi/v1/openOrders",
            params,
        )
        if not isinstance(payload, list):
            raise BinanceAPIError("Respons openOrders Binance tidak valid.", endpoint="/fapi/v1/openOrders")
        return [item for item in payload if isinstance(item, dict)]

    async def cancel_order(self, symbol: str, *, order_id: int | None = None, client_order_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("order_id atau client_order_id harus tersedia.")
        payload = await self._request("DELETE", "/fapi/v1/order", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons cancel order Binance tidak valid.", endpoint="/fapi/v1/order")
        return payload

    async def place_algo_order(
        self, *, symbol: str, side: str, order_type: str, trigger_price: Decimal,
        position_side: str, client_algo_id: str, working_type: str = "CONTRACT_PRICE",
    ) -> dict[str, Any]:
        """Submit a close-position Futures conditional order (SL/TP)."""
        working_type = str(working_type or "").upper()
        if working_type not in {"MARK_PRICE", "CONTRACT_PRICE"}:
            raise ValueError(f"working_type tidak valid: {working_type!r}")
        params: dict[str, Any] = {
            "algoType": "CONDITIONAL", "symbol": symbol, "side": side,
            "type": order_type, "triggerPrice": decimal_to_str(trigger_price),
            "workingType": working_type, "closePosition": "true",
            "clientAlgoId": client_algo_id,
        }
        if position_side and position_side != "BOTH":
            params["positionSide"] = position_side
        payload = await self._request("POST", "/fapi/v1/algoOrder", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons Algo Order Binance tidak valid.", endpoint="/fapi/v1/algoOrder")
        return payload

    async def get_algo_order(
        self, *, symbol: str, algo_id: int | None = None,
        client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if algo_id is not None:
            params["algoId"] = algo_id
        elif client_algo_id:
            params["clientAlgoId"] = client_algo_id
        else:
            raise ValueError("algo_id atau client_algo_id harus tersedia.")
        payload = await self._request("GET", "/fapi/v1/algoOrder", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons query Algo Order Binance tidak valid.", endpoint="/fapi/v1/algoOrder")
        return payload

    async def cancel_algo_order(
        self, *, symbol: str, algo_id: int | None = None,
        client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if algo_id is not None:
            params["algoId"] = algo_id
        elif client_algo_id:
            params["clientAlgoId"] = client_algo_id
        else:
            raise ValueError("algo_id atau client_algo_id harus tersedia.")
        payload = await self._request("DELETE", "/fapi/v1/algoOrder", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons cancel Algo Order Binance tidak valid.", endpoint="/fapi/v1/algoOrder")
        return payload

    async def place_market_entry(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        position_side: str,
        client_order_id: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": format_qty(quantity),
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }
        if position_side:
            params["positionSide"] = position_side
        payload = await self._request("POST", "/fapi/v1/order", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons market entry Binance tidak valid.", endpoint="/fapi/v1/order")
        return payload

    async def get_positions(self, symbol: str) -> list[dict[str, Any]]:
        payload = await self._request(
            "GET",
            "/fapi/v2/positionRisk",
            {"symbol": symbol},
        )
        if not isinstance(payload, list):
            raise BinanceAPIError(
                "Respons positionRisk Binance tidak valid.",
                endpoint="/fapi/v2/positionRisk",
            )

        result: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("symbol") or "") != symbol:
                continue
            try:
                amount = parse_signed_decimal(str(item.get("positionAmt") or "0"))
            except ValueError:
                continue
            if amount == 0:
                continue
            copied = dict(item)
            copied["_abs_position_amt"] = abs(amount)
            copied["_position_amount_decimal"] = amount
            result.append(copied)
        self.last_positions[symbol.upper()] = [dict(item) for item in result]
        self.last_positions_at[symbol.upper()] = time.monotonic()
        return result

    async def get_position(self, symbol: str, desired_side: str) -> dict[str, Any] | None:
        dual = await self.ensure_position_mode()
        target_side = (
            "LONG" if desired_side == "BUY" else "SHORT"
        ) if dual else "BOTH"

        positions = await self.get_positions(symbol)
        for item in positions:
            if str(item.get("positionSide") or "BOTH") != target_side:
                continue

            amount = item.get("_position_amount_decimal")
            if not isinstance(amount, Decimal):
                try:
                    amount = parse_signed_decimal(str(amount or "0"))
                except ValueError:
                    continue

            if amount == 0:
                continue

            # In ONE-WAY mode BOTH represents the signed actual position:
            # positive = LONG, negative = SHORT.
            if not dual:
                if desired_side == "BUY" and amount <= 0:
                    continue
                if desired_side == "SELL" and amount >= 0:
                    continue

            return item

        return None

    async def place_market_close(
        self,
        *,
        symbol: str,
        position: dict[str, Any],
        entry_direction: str,
        new_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        dual = await self.ensure_position_mode()
        position_side = str(position.get("positionSide") or "BOTH")
        quantity = parse_decimal(str(position["_abs_position_amt"]))
        side = "SELL" if entry_direction == "BUY" else "BUY"

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": format_qty(quantity),
            "newOrderRespType": "RESULT",
        }
        if new_client_order_id:
            if len(new_client_order_id) > 36:
                raise ValueError("newClientOrderId melebihi batas 36 karakter.")
            params["newClientOrderId"] = new_client_order_id
        if dual:
            params["positionSide"] = position_side
        else:
            params["positionSide"] = "BOTH"
            params["reduceOnly"] = "true"

        payload = await self._request("POST", "/fapi/v1/order", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons market close Binance tidak valid.", endpoint="/fapi/v1/order")
        return payload


# ============================================================
# BINANCE WEBSOCKET MARKET DATA
# ============================================================

class BinanceWebSocket:
    """
    Satu koneksi market WebSocket untuk seluruh symbol aktif.

    Streams:
        <symbol>@aggTrade       untuk harga / entry trade aktif
        <symbol>@kline_15m     untuk candle scanner
        !markPrice@arr@1s      satu stream global

    Symbol dikirim lowercase sesuai format Binance.
    """

    def __init__(
        self,
        on_price,
        on_kline=None,
    ) -> None:
        self.on_price = on_price
        self.on_kline = on_kline

        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._ws = None

        self._desired_symbols: set[str] = set()  # union; retained for status/backward compatibility
        self._desired_trade_symbols: set[str] = set()
        self._desired_kline_symbols: set[str] = set()
        self._desired_exit_symbols: set[str] = set()  # only filled strategy positions
        self._subscribed_symbols: set[str] = set()  # aggTrade
        self._subscribed_kline_symbols: set[str] = set()  # scanner M15
        self._subscribed_exit_symbols: set[str] = set()  # position H1 exit

        self._command_lock = asyncio.Lock()

        # Runtime market-event ordering guard. Binance may reconnect and
        # deliver an event after a connection transition; stale events must
        # never be allowed to move a setup backwards/forwards in time.
        self._last_market_event_key: dict[str, tuple[int, int]] = {}

        self.connected = False
        self.last_message_at: datetime | None = None
        self.reconnect_count = 0

    @property
    def status(self) -> str:
        if self.connected:
            if self.last_message_at is None or not self._desired_symbols:
                return "CONNECTED"
            age = (
                now_utc() - self.last_message_at
            ).total_seconds()

            if age > PRICE_STALE_SECONDS:
                return "STALE"

            return "LIVE"

        if self._task is not None and not self._task.done():
            if self.reconnect_count > 0:
                return "RECONNECTING"
            return "CONNECTING"

        return "OFFLINE"

    def symbols(self) -> list[str]:
        return sorted(self._desired_symbols)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return

        self._stop.clear()

        self._task = asyncio.create_task(
            self._run(),
            name="binance-market-ws",
        )

    async def stop(self) -> None:
        self._stop.set()

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                log.exception("Gagal menutup Binance WebSocket.")

        task = self._task

        if task is not None:
            try:
                await asyncio.wait_for(
                    task,
                    timeout=5,
                )
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception(
                    "WebSocket task berhenti dengan error."
                )

        self._task = None
        self._ws = None
        self.connected = False
        self._subscribed_symbols.clear()
        self._subscribed_kline_symbols.clear()
        self._subscribed_exit_symbols.clear()
        self._desired_symbols.clear()
        self._desired_trade_symbols.clear()
        self._desired_kline_symbols.clear()
        self._desired_exit_symbols.clear()
        self._last_market_event_key.clear()

    async def set_symbols(
        self,
        symbols: set[str] | list[str] | tuple[str, ...],
        *,
        scanner_symbols: set[str] | list[str] | tuple[str, ...] | None = None,
        exit_symbols: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Reconcile trade ticks, scanner M15 candles, and active-position H1 candles."""
        trade = {normalize_symbol(value) for value in symbols if str(value).strip()}
        scanner = {normalize_symbol(value) for value in (scanner_symbols or ()) if str(value).strip()}
        exits = {normalize_symbol(value) for value in (exit_symbols or ()) if str(value).strip()}
        old_trade = set(self._desired_trade_symbols)
        old_scanner = set(self._desired_kline_symbols)
        old_exits = set(self._desired_exit_symbols)
        self._desired_trade_symbols = trade
        self._desired_kline_symbols = scanner
        self._desired_exit_symbols = exits
        self._desired_symbols = trade | scanner | exits
        if not self.connected or self._ws is None:
            return

        batch_size = 100  # 4 control messages/sec at 250ms spacing, below Binance's limit.
        plans = (
            (self._subscribed_symbols, trade, "aggTrade"),
            (self._subscribed_kline_symbols, scanner, "kline_15m"),
            (self._subscribed_exit_symbols, exits, "kline_1h"),
        )
        for subscribed, desired, stream_type in plans:
            remove = sorted(subscribed - desired)
            add = sorted(desired - subscribed)
            for start in range(0, len(remove), batch_size):
                await self._send_unsubscribe(remove[start:start + batch_size], stream_type)
                await asyncio.sleep(0.25)
            for start in range(0, len(add), batch_size):
                await self._send_subscribe(add[start:start + batch_size], stream_type)
                await asyncio.sleep(0.25)

        if old_trade != trade or old_scanner != scanner or old_exits != exits:
            log.info(
                "[WS] subscriptions reconciled | aggTrade=%s M15=%s H1-exit=%s total=%s",
                len(trade), len(scanner), len(exits), len(self._desired_symbols),
            )

    async def add_symbol(self, symbol: str) -> None:
        symbol = normalize_symbol(symbol)
        if not symbol or symbol in self._desired_trade_symbols:
            return
        self._desired_trade_symbols.add(symbol)
        self._desired_symbols.add(symbol)
        if self.connected and self._ws is not None:
            sent = await self._send_subscribe([symbol])
            if not sent:
                log.warning(
                    "Subscribe aggTrade %s ditunda; akan diselaraskan saat reconnect.", symbol
                )

    async def remove_symbol(self, symbol: str) -> None:
        symbol = normalize_symbol(symbol)
        self._desired_trade_symbols.discard(symbol)
        self._desired_kline_symbols.discard(symbol)
        self._desired_symbols = self._desired_trade_symbols | self._desired_kline_symbols
        if self.connected and self._ws is not None:
            if symbol in self._subscribed_symbols:
                await self._send_unsubscribe([symbol])
            if symbol in self._subscribed_kline_symbols:
                await self._send_unsubscribe([symbol], "kline_15m")
            if symbol in self._subscribed_exit_symbols:
                await self._send_unsubscribe([symbol], "kline_1h")

    async def _send_request(
        self,
        method: str,
        symbols: list[str],
        stream_type: str = "aggTrade",
    ) -> bool:
        """
        Kirim SUBSCRIBE/UNSUBSCRIBE.

        Return True kalau payload benar-benar terkirim, False kalau
        koneksi ternyata sudah tertutup (server menutup 1000/OK,
        close handshake timeout, dsb). Kasus "koneksi sudah tertutup"
        BUKAN exception yang perlu di-propagate: loop _run() akan
        reconnect sendiri lalu _resubscribe_all() mengirim ulang semua
        simbol di _desired_symbols.
        """
        if self._ws is None or not self.connected:
            return False

        if not symbols:
            return False

        # Scanner symbols use kline_15m; only actual active trade symbols use
        # aggTrade. Global mark price remains a single stream for price guards.
        params = [f"{symbol.lower()}@{stream_type}" for symbol in symbols]

        payload = {
            "method": method,
            "params": params,
            "id": int(time.time() * 1000) % 2_000_000_000,
        }

        async with self._command_lock:
            # Ambil ulang referensi setelah lock: selama menunggu lock,
            # _run() bisa saja sudah menutup koneksi dan mengosongkan _ws.
            ws = self._ws

            if ws is None or not self.connected or bool(getattr(ws, "closed", False)):
                return False

            try:
                await ws.send(
                    json.dumps(payload)
                )
            except (
                ConnectionClosed,
                ConnectionError,
                OSError,
            ) as exc:
                # Tandai koneksi mati supaya tidak ada send lain yang
                # menabrak socket yang sama sebelum _run() sempat
                # membersihkan state-nya.
                self.connected = False

                if method == "UNSUBSCRIBE" and isinstance(exc, ConnectionClosed):
                    log.info(
                        "[WS] UNSUBSCRIBE %s dilewati karena socket Binance sudah closed; desired subscription state dipertahankan dan reconnect akan menyelaraskan ulang.",
                        ",".join(symbols),
                    )
                else:
                    log.warning(
                        "Gagal kirim %s ke Binance WebSocket (koneksi sudah tertutup): %s",
                        method,
                        exc,
                    )

                return False

        return True

    async def _send_subscribe(
        self,
        symbols: list[str],
        stream_type: str = "aggTrade",
    ) -> bool:
        sent = await self._send_request("SUBSCRIBE", symbols, stream_type)
        if sent:
            target = (
                self._subscribed_kline_symbols if stream_type == "kline_15m" else
                self._subscribed_exit_symbols if stream_type == "kline_1h" else
                self._subscribed_symbols
            )
            target.update(normalize_symbol(symbol) for symbol in symbols)
        return sent

    async def _send_unsubscribe(
        self,
        symbols: list[str],
        stream_type: str = "aggTrade",
    ) -> bool:
        sent = await self._send_request("UNSUBSCRIBE", symbols, stream_type)
        if sent:
            target = (
                self._subscribed_kline_symbols if stream_type == "kline_15m" else
                self._subscribed_exit_symbols if stream_type == "kline_1h" else
                self._subscribed_symbols
            )
            target.difference_update(normalize_symbol(symbol) for symbol in symbols)
        return sent

    async def _send_global_markprice_stream(self) -> None:
        """Subscribe once to Binance's all-symbol mark-price array stream."""
        if self._ws is None or not self.connected:
            return
        payload = {
            "method": "SUBSCRIBE",
            "params": ["!markPrice@arr@1s"],
            "id": int(time.time() * 1000) % 2_000_000_000,
        }
        async with self._command_lock:
            ws = self._ws
            if ws is None or not self.connected:
                return
            await ws.send(json.dumps(payload))

    async def _resubscribe_all(self) -> None:
        self._subscribed_symbols.clear()
        self._subscribed_kline_symbols.clear()
        self._subscribed_exit_symbols.clear()
        trade_symbols = sorted(self._desired_trade_symbols)
        scanner_symbols = sorted(self._desired_kline_symbols)
        exit_symbols = sorted(self._desired_exit_symbols)
        # One global mark-price stream plus small batches of active trades and
        # scanner kline streams. This remains comfortably below 1,024 streams
        # for the configured 522-symbol universe + normal active-trade count.
        await self._send_global_markprice_stream()
        batch_size = 100
        for symbols, stream_type in (
            (trade_symbols, "aggTrade"),
            (scanner_symbols, "kline_15m"),
            (exit_symbols, "kline_1h"),
        ):
            for start in range(0, len(symbols), batch_size):
                if self._stop.is_set():
                    break
                await self._send_subscribe(symbols[start:start + batch_size], stream_type)
                await asyncio.sleep(0.25)

    async def _handle_market_event(self, data: dict[str, Any]) -> None:
        event_type = str(data.get("e") or "")
        if event_type == "kline":
            symbol = normalize_symbol(str(data.get("s") or ""))
            kline = data.get("k")
            if symbol and isinstance(kline, dict) and self.on_kline is not None:
                try:
                    result = self.on_kline(symbol, kline)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    log.debug("[WS/KLINE] gagal memproses %s %s", symbol, kline.get("i"), exc_info=True)
            return
        if event_type not in {"aggTrade", "markPriceUpdate"}:
            return
        symbol = normalize_symbol(str(data.get("s") or ""))
        price_raw = str(data.get("p") or "")
        event_time_ms = int(data.get("E") or data.get("T") or int(time.time() * 1000))
        try:
            aggregate_trade_id = int(data.get("a")) if event_type == "aggTrade" else -1
        except (TypeError, ValueError):
            aggregate_trade_id = -1
        if not symbol or not price_raw:
            return
        event_key = (event_time_ms, aggregate_trade_id)
        last_key = self._last_market_event_key.get(symbol)
        if last_key is not None and event_key <= last_key:
            return
        self._last_market_event_key[symbol] = event_key
        try:
            price = parse_decimal(price_raw)
        except ValueError:
            log.warning("Harga WebSocket tidak valid: %s", price_raw)
            return
        await self.on_price(symbol, price, event_time_ms, aggregate_trade_id)

    async def _run(self) -> None:
        backoff = WS_RECONNECT_MIN

        while not self._stop.is_set():
            try:
                log.info(
                    "Menghubungkan Binance WebSocket: %s",
                    BINANCE_WS_BASE,
                )

                async with ws_connect(
                    BINANCE_WS_BASE,
                    **PROXY_MANAGER.ws_kwargs(),
                    # Binance sends server ping frames; websockets automatically replies.
                    # Disable client keepalive to avoid client/server ping races that can
                    # produce a misleading 1011 keepalive timeout during command sends.
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=3,
                    max_queue=2048,
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    self.last_message_at = now_utc()
                    self.reconnect_count += 1

                    log.info(
                        "Binance WebSocket connected. reconnect=%s",
                        self.reconnect_count,
                    )

                    await self._resubscribe_all()

                    backoff = WS_RECONNECT_MIN

                    while not self._stop.is_set():
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(),
                                timeout=PRICE_STALE_SECONDS + 30,
                            )
                        except asyncio.TimeoutError:
                            # Pair sepi bisa tanpa trade >45s; cek koneksi via ping.
                            try:
                                pong_waiter = await ws.ping()
                                await asyncio.wait_for(pong_waiter, timeout=10)
                            except Exception as exc:
                                raise ConnectionError(
                                    "Ping WebSocket tidak dibalas."
                                ) from exc
                            self.last_message_at = now_utc()
                            continue

                        if raw is None:
                            raise ConnectionError(
                                "WebSocket menerima close."
                            )

                        self.last_message_at = now_utc()

                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            log.warning(
                                "WebSocket payload bukan JSON."
                            )
                            continue

                        # SUBSCRIBE response:
                        # {"result": null, "id": ...}
                        #
                        # /market/stream memakai combined-stream subscription dan
                        # membungkus event sebagai {"stream":"...","data":{...}}.
                        # Parser tetap menerima raw payload untuk kompatibilitas.
                        # Combined !markPrice@arr@1s frames contain a list;
                        # individual aggTrade frames contain one dictionary.
                        data = payload.get("data", payload) if isinstance(payload, dict) else payload
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    await self._handle_market_event(item)
                        elif isinstance(data, dict):
                            await self._handle_market_event(data)

            except asyncio.CancelledError:
                break

            except (
                ConnectionClosed,
                asyncio.TimeoutError,
                ConnectionError,
                OSError,
            ) as exc:
                # ConnectionClosed (termasuk ConnectionClosedOK 1000)
                # adalah putus koneksi normal, mis. server Binance
                # menutup koneksi berkala. Cukup reconnect, bukan ERROR.
                if self._stop.is_set():
                    break

                log.warning(
                    "Binance WebSocket terputus: %s",
                    exc,
                )

            except Exception as exc:
                if self._stop.is_set():
                    break
                PROXY_MANAGER.observe_transport_error(f"WS market: {exc}")

                log.exception(
                    "Error Binance WebSocket: %s",
                    exc,
                )

            finally:
                self.connected = False
                self._ws = None
                self._subscribed_symbols.clear()
                self._subscribed_kline_symbols.clear()
                self._subscribed_exit_symbols.clear()

            if self._stop.is_set():
                break

            await asyncio.sleep(backoff)

            backoff = min(
                backoff * 2,
                WS_RECONNECT_MAX,
            )


# ============================================================
# GITHUB STORE
# ============================================================

class BinanceUserDataWebSocket:
    """Private USD-M user stream for authoritative order/fill notifications.

    REST is used only to create/renew the listenKey. Once connected, order
    updates continue over WebSocket during an HTTP 418/429 cooldown.
    """
    def __init__(self, real_client: BinanceRealClient, on_event) -> None:
        self.real = real_client
        self.on_event = on_event
        self.listen_key: str | None = None
        self.connected = False
        self.last_message_at: datetime | None = None
        self.last_keepalive_monotonic = 0.0
        self.reconnect_count = 0
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._ws = None
        self._start_lock = asyncio.Lock()
        # listenKey is a short-lived bearer credential. Store only on the local
        # runtime filesystem (never GitHub) so a process restart during an IP ban
        # can reconnect while the existing key remains valid.
        self._state_path = BASE_DIR / "data" / "private_user_stream_state.json"
        # Fresh-engine policy: never restore a private listenKey from a prior /try.

    def _load_local_state(self) -> None:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            key = str(payload.get("listen_key") or "").strip()
            expires_at = float(payload.get("expires_at") or 0)
            if key and expires_at > time.time() + 30:
                self.listen_key = key
                # The persisted expiry is one hour after the last successful
                # start/ping. Recover elapsed lifetime accurately so a /try
                # reload does not accidentally let the key expire before the
                # next 45-minute keepalive.
                remaining_lifetime = max(0.0, min(3600.0, expires_at - time.time()))
                elapsed_since_keepalive = max(0.0, 3600.0 - remaining_lifetime)
                self.last_keepalive_monotonic = time.monotonic() - elapsed_since_keepalive
        except Exception:
            pass

    def _save_local_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._state_path.with_suffix(".tmp")
            temp.write_text(json.dumps({
                "listen_key": self.listen_key or "",
                "expires_at": time.time() + 3600 if self.listen_key else 0,
                "updated_at": iso_utc(),
            }), encoding="utf-8")
            os.replace(temp, self._state_path)
        except Exception:
            log.debug("Tidak dapat menyimpan user-stream listenKey lokal", exc_info=True)

    @property
    def status(self) -> str:
        if self.connected:
            return "LIVE"
        if self._task is not None and not self._task.done():
            return "RECONNECTING" if self.reconnect_count else "CONNECTING"
        return "OFFLINE"

    async def start(self) -> None:
        async with self._start_lock:
            if self._task is not None and not self._task.done():
                return
            self._stop.clear()
            if not self.listen_key and BINANCE_COOLDOWN.remaining() <= 0:
                # Normal path: acquire a listenKey through REST before starting.
                self.listen_key = await self.real.create_user_stream()
                self.last_keepalive_monotonic = time.monotonic()
            # During an IP ban, deliberately start the task without a listenKey.
            # _run() will try the low-frequency USD-M WebSocket API start method;
            # if Binance blocks that too, it backs off and retries after recovery.
            self._task = asyncio.create_task(self._run(), name="binance-private-user-stream")

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                log.debug("Private WebSocket close: %s", traceback.format_exc(limit=1))
        task = self._task
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("Private user stream shutdown error")
        self._task = None
        self._ws = None
        self.connected = False

    async def _user_stream_ws_api(self, method: str) -> str:
        """Use the USD-M WebSocket API to start/ping a listenKey when REST is banned.

        This is a low-frequency fallback (weight 1) and is not treated as a
        guaranteed way around an IP ban. If Binance blocks it too, the existing
        private stream is left running and REST reconciliation takes over later.
        """
        if not self.real.configured:
            raise BinanceAPIError("API key REAL belum tersedia untuk user-stream WS API.")
        request_id = uuid4().hex
        payload = {
            "id": request_id,
            "method": method,
            "params": {"apiKey": self.real.api_key},
        }
        async with ws_connect(
            BINANCE_WS_API_BASE, **PROXY_MANAGER.ws_kwargs(),
            ping_interval=None, ping_timeout=None, close_timeout=3, max_queue=16,
        ) as ws:
            await ws.send(json.dumps(payload))
            raw = await asyncio.wait_for(ws.recv(), timeout=12.0)
        response = json.loads(raw)
        if str(response.get("id") or "") != request_id:
            raise RuntimeError("Respons userDataStream WS API memiliki request ID berbeda.")
        if int(response.get("status") or 0) != 200:
            error = response.get("error") or response
            raise RuntimeError(f"userDataStream WS API {method} ditolak: {error}")
        result = response.get("result") or {}
        key = str(result.get("listenKey") or self.listen_key or "").strip()
        if not key:
            raise RuntimeError(f"userDataStream WS API {method} tidak mengembalikan listenKey.")
        return key

    async def _renew_key_if_due(self) -> None:
        if not self.listen_key:
            return
        if time.monotonic() - self.last_keepalive_monotonic < BINANCE_USER_STREAM_KEEPALIVE_SECONDS:
            return
        if BINANCE_COOLDOWN.remaining() > 0:
            # Prefer WebSocket API keepalive during an HTTP REST ban. It is
            # attempted only on the 45-minute keepalive cadence, not per tick.
            # A failure is backed off by five minutes; the current private
            # socket remains active while Binance's listenKey is still valid.
            try:
                self.listen_key = await self._user_stream_ws_api("userDataStream.ping")
                self.last_keepalive_monotonic = time.monotonic()
                log.info("[USER WS] listenKey renewed through WebSocket API during REST cooldown")
            except Exception as exc:
                self.last_keepalive_monotonic = time.monotonic() - BINANCE_USER_STREAM_KEEPALIVE_SECONDS + 300.0
                log.info("[USER WS] keepalive tertunda selama REST cooldown; retry dalam 5 menit | %s", exc)
            return
        self.listen_key = await self.real.keepalive_user_stream(self.listen_key)
        self.last_keepalive_monotonic = time.monotonic()
        log.info("[USER WS] listenKey renewed")

    # The private stream carries only ORDER_TRADE_UPDATE / ACCOUNT_UPDATE /
    # ALGO_UPDATE events. Market ticks are handled exclusively by the public
    # BinanceWebSocket class; keeping a duplicate market handler here would
    # reference callbacks/state that this private-stream class does not own.

    async def _run(self) -> None:
        backoff = WS_RECONNECT_MIN
        while not self._stop.is_set():
            try:
                if not self.listen_key:
                    if BINANCE_COOLDOWN.remaining() > 0:
                        # If no persisted listenKey survived /try or restart,
                        # attempt the native WS API start endpoint once per
                        # cooldown retry interval. This may also be rate-limited;
                        # failure is non-fatal and REST reconciliation remains.
                        try:
                            self.listen_key = await self._user_stream_ws_api("userDataStream.start")
                            self.last_keepalive_monotonic = time.monotonic()
                        except Exception as exc:
                            log.info("[USER WS] userDataStream.start belum tersedia saat REST cooldown; retry terjadwal | %s", exc)
                            await asyncio.sleep(max(300.0, min(900.0, BINANCE_COOLDOWN.remaining() or 300.0)))
                            continue
                    else:
                        self.listen_key = await self.real.create_user_stream()
                        self.last_keepalive_monotonic = time.monotonic()
                        self._save_local_state()

                query = urlencode({"listenKey": self.listen_key, "events": "ORDER_TRADE_UPDATE/ACCOUNT_UPDATE"})
                url = f"{BINANCE_PRIVATE_WS_BASE}?{query}"
                log.info("[USER WS] connecting private order stream")
                async with ws_connect(
                    url, **PROXY_MANAGER.ws_kwargs(),
                    ping_interval=None, ping_timeout=None, close_timeout=3, max_queue=2048,
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    self.last_message_at = now_utc()
                    self.reconnect_count += 1
                    backoff = WS_RECONNECT_MIN
                    log.info("[USER WS] connected; reconnect=%s", self.reconnect_count)

                    while not self._stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
                        except asyncio.TimeoutError:
                            await self._renew_key_if_due()
                            continue
                        if raw is None:
                            raise ConnectionError("Private user stream closed by Binance")
                        self.last_message_at = now_utc()
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
                            payload = payload["data"]
                        if not isinstance(payload, dict):
                            continue
                        event_name = str(payload.get("e") or "")
                        if event_name == "listenKeyExpired":
                            log.warning("[USER WS] listenKey expired; create a new one after cooldown")
                            self.listen_key = None
                            self._save_local_state()
                            break
                        if event_name in {"ORDER_TRADE_UPDATE", "ACCOUNT_UPDATE", "ALGO_UPDATE", "MARGIN_CALL"}:
                            try:
                                await self.on_event(payload)
                            except Exception:
                                log.exception("[USER WS] order/account event handler failed")
                        await self._renew_key_if_due()

            except asyncio.CancelledError:
                break
            except BinanceRateLimitError as exc:
                log.info("[USER WS] listenKey operation ditahan rate governor | remaining=%.0fs | %s", BINANCE_COOLDOWN.remaining(), exc)
            except (ConnectionClosed, asyncio.TimeoutError, ConnectionError, OSError) as exc:
                if not self._stop.is_set():
                    PROXY_MANAGER.observe_transport_error(f"WS user: {exc}")
                    log.warning("[USER WS] disconnected: %s", exc)
            except Exception as exc:
                if not self._stop.is_set():
                    PROXY_MANAGER.observe_transport_error(f"WS user: {exc}")
                    log.exception("[USER WS] unexpected error: %s", exc)
            finally:
                self.connected = False
                self._ws = None

            if self._stop.is_set():
                break
            wait = BINANCE_COOLDOWN.remaining()
            await asyncio.sleep(max(wait, backoff))
            backoff = min(backoff * 2, WS_RECONNECT_MAX)


class GitHubStore:
    def __init__(self) -> None:
        self.token = GITHUB_TOKEN
        self.repo_name = REPO_NAME
        self.branch = GITHUB_BRANCH

        self._write_lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        }

    def _url(self, path: str) -> str:
        encoded = "/".join(
            quote(part, safe="")
            for part in path.strip("/").split("/")
        )

        return (
            f"{GITHUB_API}/repos/"
            f"{self.repo_name}/contents/{encoded}"
        )

    async def get_file(
        self,
        path: str,
    ) -> tuple[bytes | None, str | None]:
        def request() -> tuple[bytes | None, str | None]:
            response = requests.get(
                self._url(path),
                headers=self._headers(),
                params={"ref": self.branch},
                timeout=GITHUB_REQUEST_SECONDS,
            )

            if response.status_code == 404:
                return None, None

            if response.status_code >= 400:
                raise RuntimeError(
                    f"GitHub GET {path}: HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

            body = response.json()

            if body.get("type") != "file":
                raise RuntimeError(
                    f"GitHub path bukan file: {path}"
                )

            content = str(
                body.get("content") or ""
            ).replace("\n", "")

            if not content and int(body.get("size") or 0) > 0:
                # File >1 MB: kolom content kosong, ambil isi lewat media type raw.
                raw_headers = dict(self._headers())
                raw_headers["Accept"] = "application/vnd.github.raw+json"
                raw_response = requests.get(
                    self._url(path),
                    headers=raw_headers,
                    params={"ref": self.branch},
                    timeout=GITHUB_REQUEST_SECONDS,
                )
                if raw_response.status_code >= 400:
                    raise RuntimeError(
                        f"GitHub GET raw {path}: HTTP "
                        f"{raw_response.status_code}: {raw_response.text[:500]}"
                    )
                return raw_response.content, str(body.get("sha") or "")

            try:
                decoded = base64.b64decode(
                    content
                )
            except Exception as exc:
                raise RuntimeError(
                    f"GitHub file {path} gagal di-decode."
                ) from exc

            return (
                decoded,
                str(body.get("sha") or ""),
            )

        return await asyncio.to_thread(request)

    async def list_dir(self, path: str) -> list[str]:
        def request() -> list[str]:
            response = requests.get(
                self._url(path),
                headers=self._headers(),
                params={"ref": self.branch},
                timeout=GITHUB_REQUEST_SECONDS,
            )
            if response.status_code == 404:
                return []
            if response.status_code >= 400:
                raise RuntimeError(f"GitHub LIST {path}: HTTP {response.status_code}: {response.text[:300]}")
            body = response.json()
            if not isinstance(body, list):
                return []
            return [str(item.get("name")) for item in body if item.get("type") == "file"]

        return await asyncio.to_thread(request)

    async def delete_file(
        self,
        path: str,
        commit_message: str,
    ) -> bool:
        """Hapus satu file GitHub. Return False jika file memang tidak ada."""
        async with self._write_lock:
            for _attempt in range(3):
                _current, sha = await self.get_file(path)

                if not sha:
                    return False

                payload = {
                    "message": commit_message,
                    "sha": sha,
                    "branch": self.branch,
                }

                def request() -> bool:
                    response = requests.delete(
                        self._url(path),
                        headers=self._headers(),
                        json=payload,
                        timeout=GITHUB_REQUEST_SECONDS,
                    )

                    if response.status_code == 404:
                        return False

                    if response.status_code == 409:
                        raise RuntimeError("GITHUB_CONFLICT")

                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"GitHub DELETE {path}: HTTP "
                            f"{response.status_code}: "
                            f"{response.text[:700]}"
                        )

                    return True

                try:
                    return await asyncio.to_thread(request)
                except RuntimeError as exc:
                    if str(exc) != "GITHUB_CONFLICT":
                        raise
                    await asyncio.sleep(0.5)

            raise RuntimeError(
                f"GitHub gagal menghapus {path}: conflict berulang."
            )

    async def replace_file(
        self,
        path: str,
        content: bytes,
        commit_message: str,
    ) -> str:
        """
        Create/update satu file GitHub secara serial.

        Retry conflict 409 dilakukan dengan mengambil SHA terbaru.
        """
        async with self._write_lock:
            last_error: Exception | None = None

            for _attempt in range(3):
                current, sha = await self.get_file(
                    path
                )

                # current sengaja tidak digunakan; SHA yang dibutuhkan
                # untuk update sudah diambil dari GitHub.
                del current

                payload = {
                    "message": commit_message,
                    "content": base64.b64encode(
                        content
                    ).decode("ascii"),
                    "branch": self.branch,
                }

                if sha:
                    payload["sha"] = sha

                def request() -> str:
                    response = requests.put(
                        self._url(path),
                        headers=self._headers(),
                        json=payload,
                        timeout=GITHUB_REQUEST_SECONDS,
                    )

                    if response.status_code == 409:
                        raise RuntimeError(
                            "GITHUB_CONFLICT"
                        )

                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"GitHub PUT {path}: HTTP "
                            f"{response.status_code}: "
                            f"{response.text[:700]}"
                        )

                    body = response.json()

                    return str(
                        (body.get("commit") or {}).get("sha")
                        or ""
                    )

                try:
                    return await asyncio.to_thread(
                        request
                    )

                except RuntimeError as exc:
                    last_error = exc

                    if str(exc) != "GITHUB_CONFLICT":
                        raise

                    await asyncio.sleep(0.5)

            if last_error:
                raise last_error

            raise RuntimeError(
                f"GitHub gagal meng-update {path}."
            )

    async def append_text_file(
        self,
        path: str,
        addition: str,
        commit_message: str,
    ) -> str:
        async with self._write_lock:
            last_error: Exception | None = None

            for _attempt in range(3):
                current, sha = await self.get_file(path)

                if current is None:
                    content = "# Trading Journal\n\n" + addition.lstrip("\n")
                else:
                    old_text = current.decode(
                        "utf-8",
                        errors="replace",
                    )

                    if old_text and not old_text.endswith("\n"):
                        old_text += "\n"

                    content = old_text + addition

                payload = {
                    "message": commit_message,
                    "content": base64.b64encode(
                        content.encode("utf-8")
                    ).decode("ascii"),
                    "branch": self.branch,
                }

                if sha:
                    payload["sha"] = sha

                def request() -> str:
                    response = requests.put(
                        self._url(path),
                        headers=self._headers(),
                        json=payload,
                        timeout=GITHUB_REQUEST_SECONDS,
                    )

                    if response.status_code == 409:
                        raise RuntimeError(
                            "GITHUB_CONFLICT"
                        )

                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"GitHub PUT {path}: HTTP "
                            f"{response.status_code}: "
                            f"{response.text[:700]}"
                        )

                    body = response.json()

                    return str(
                        (body.get("commit") or {}).get("sha")
                        or ""
                    )

                try:
                    return await asyncio.to_thread(
                        request
                    )
                except RuntimeError as exc:
                    last_error = exc

                    if str(exc) != "GITHUB_CONFLICT":
                        raise

                    await asyncio.sleep(0.5)

            if last_error:
                raise last_error

            raise RuntimeError(
                f"GitHub gagal meng-append {path}."
            )






# ---------------------------------------------------------------------------
# Volumatic Trend trading engine
# ---------------------------------------------------------------------------
class TradingEngine:
    def __init__(self, context: dict[str, Any]) -> None:
        self.context = dict(context)
        # Initialize the local data root before any path derived from it.
        # Signal-dedupe DB is configured later in __init__, so this must exist first.
        self._local_data_dir = BASE_DIR / "data"
        self._local_data_dir.mkdir(parents=True, exist_ok=True)
        self.chat_id = int(context.get("chat_id") or ALLOWED_USER_ID)
        self.user_id = int(context.get("user_id") or ALLOWED_USER_ID)
        self.send_message = context["send_message"]
        self.session_id = generate_session_id()
        self.session_data_dir = f"data/sessions/{self.session_id}"
        self.session_history_trades_path = f"{self.session_data_dir}/trades.json"
        self.session_history_events_path = f"{self.session_data_dir}/events.jsonl"
        self.session_history_markdown_path = f"{self.session_data_dir}/trade_history.md"
        self.session_notes_path = f"{self.session_data_dir}/notes.json"
        self.session_deferred_commands_path = f"{self.session_data_dir}/deferred_commands.json"

        self.rest = BinanceREST()
        self.real = BinanceRealClient()
        self.github = GitHubStore()
        self.ws = BinanceWebSocket(self._on_price, self._on_kline)
        self.user_ws = BinanceUserDataWebSocket(self.real, self._on_user_data_event)

        self.real_mode = False  # New REAL entries always require explicit /real on.
        self.margin_usdt = DEFAULT_MARGIN_USDT
        self.leverage = DEFAULT_LEVERAGE
        self.min_margin_enabled = False
        self.max_active_trades = MAX_ACTIVE_TRADES
        self.timeframe = "M15"
        self.trend_length = volumatic.DEFAULT_TREND_LENGTH
        self.ema_filter_length: int | tuple[int, int] | None = volumatic.DEFAULT_EMA_FILTER_LENGTH
        self.h1_enabled = True  # one master switch for H1 entry filter + H1 inverse-cross forced exit
        self._h1_toggle_generation = 0
        self.supertrend_atr_length = volumatic.SUPERTREND_ATR_LENGTH
        self.supertrend_factor = volumatic.SUPERTREND_FACTOR
        self.rr_min = Decimal(str(volumatic.RR_MIN))
        self.rr_max = Decimal(str(volumatic.RR_MAX))
        self.scan_enabled = False
        # Session-only delayed scanner activation. It is deliberately not
        # persisted: every fresh /try starts with scanner OFF and no schedule.
        self.scan_schedule_slots: tuple[str, ...] = ()
        self.scan_schedule_waiting = False
        self.scan_schedule_activated_at: str | None = None
        self._scan_schedule_last_fired_minute: str | None = None

        # Account safety: trailing equity drawdown guard. It pauses NEW scanning
        # only; it never silently closes existing positions or cancels live orders.
        self.autostop_percent: Decimal | None = None
        if CONFIG_AUTOSTOP:
            try:
                configured_guard = Decimal(CONFIG_AUTOSTOP.replace(",", "."))
                if configured_guard.is_finite() and Decimal("0") < configured_guard < Decimal("100"):
                    self.autostop_percent = configured_guard
                else:
                    log.warning("AUTOSTOP_PERCENT harus >0 dan <100; konfigurasi env diabaikan.")
            except (InvalidOperation, ValueError):
                log.warning("AUTOSTOP_PERCENT tidak valid; konfigurasi env diabaikan.")
        self._equity_peak: Decimal | None = None
        self._equity_last: Decimal | None = None
        self._autostop_triggered = False
        self._autostop_last_check = 0.0
        self._autostop_last_persist = 0.0
        self._autostop_dirty = False
        self._autostop_missing_price_notice_at = 0.0
        self._autostop_last_state_notice_at = 0.0
        self._autostop_notice_state: str | None = None
        self._autostop_data_hold = bool(self.autostop_percent is not None)
        self._real_equity_baseline: Decimal | None = None
        self._realized_since_baseline = Decimal("0")
        self._autostop_source: str | None = None

        self.symbols: dict[str, SymbolMeta] = {}
        self._scanner_symbols: set[str] = set()
        self.prices: dict[str, PriceSnapshot] = {}
        self.active_trades: dict[str, Trade] = {}
        # Live exchange projection. Seeded once from the account-wide snapshot
        # already requested by /real on, then maintained by ACCOUNT_UPDATE and
        # ORDER_TRADE_UPDATE. /trade reads this cache only; it never REST-polls.
        self.exchange_positions: dict[tuple[str, str], dict[str, Any]] = {}
        self.exchange_open_orders: dict[str, dict[str, Any]] = {}
        self.exchange_snapshot_at: datetime | None = None
        self._last_user_ws_reconnect_count = 0
        self.legacy_active_snapshot_count = 0
        self.history_records: list[dict[str, Any]] = []
        self.history_events: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []
        self.last_processed_candles: dict[str, int] = {}
        self.last_position_candles: dict[str, int] = {}
        # Retained empty caches for compatibility with the launcher/state code;
        # confirmed M15 strategy signals are not evaluated intrabar by WebSocket.
        self._live_signal_candles: dict[str, dict[str, Any]] = {}
        self._live_signal_last_eval: dict[str, float] = {}
        self._live_signal_seen: dict[str, set[str]] = {}
        self._signal_retries: dict[str, dict[str, Any]] = {}
        self.deferred_commands: list[dict[str, Any]] = []
        self._deferred_commands_lock = asyncio.Lock()
        self._last_api_recovery_notice = 0.0
        self._last_cooldown_github_update = 0.0
        self._binance_notice_last_sent = 0.0
        self._binance_notice_status: int | None = None
        self._binance_notice_until_epoch = 0.0
        self._scan_resume_cursor = 0
        self._scan_resume_boundary_ms = 0
        self._scan_resume_timeframe = self.timeframe
        self._last_exchange_info_retry = 0.0
        self._was_in_binance_cooldown = self._binance_cooldown_remaining() > 0
        self._scan_interrupted_by_cooldown = False
        # Scanner must wait until the maintenance loop has completed the first
        # post-cooldown reconciliation pass. It prevents scanner REST bursts from
        # racing position/order recovery at the exact moment the ban expires.
        self._rest_recovery_active = self._was_in_binance_cooldown
        self._scan_capacity_paused = False
        self._scan_protection_paused = False
        self._last_user_stream_start_attempt = 0.0

        self._running = False
        self._scan_task: asyncio.Task | None = None
        self._maintenance_task: asyncio.Task | None = None
        self._scan_trigger_now = False
        self._last_scan_boundary_ms = 0
        self._scan_cycle_lock = asyncio.Lock()
        self._trade_lock = asyncio.Lock()
        # MARKET-on-touch execution: one entry at a time so every margin check
        # sees the balance left by the previous fill; tasks are kept referenced.
        self._entry_exec_lock = asyncio.Lock()
        self._entry_tasks: set[asyncio.Task] = set()
        self._entry_attempts: dict[str, int] = {}
        self._entry_retry_after: dict[str, float] = {}
        self._protection_locks: dict[str, asyncio.Lock] = {}
        self._ema_exit_init_tasks: dict[str, asyncio.Task] = {}
        self._ema_exit_init_semaphore = asyncio.Semaphore(1)
        self._ema_exit_close_locks: dict[str, asyncio.Lock] = {}
        self._ema_exit_close_last_check: dict[str, float] = {}
        self._ema_exit_last_retry: dict[str, float] = {}
        self._leverage_cache: dict[str, tuple[int, Decimal | None, float]] = {}
        self._signal_dedupe_lock = asyncio.Lock()
        self._signal_dedupe_loaded = False
        self._signal_dedupe_db_path = self._local_data_dir / SIGNAL_DEDUPE_DB_NAME
        self._history_lock = asyncio.Lock()
        self._settings_lock = asyncio.Lock()
        self._telegram_error_handler = TelegramErrorHandler(self)
        self._telegram_error_handler_added = False
        self._history_dirty = False
        self._last_market_event_key: dict[str, tuple[int, int]] = {}
        self._last_api_notice = 0.0
        self._api_recovery_until = 0.0
        self._last_universe_refresh = 0.0

    # ----------------------- Telegram and persistence --------------------
    async def reply(self, message: str) -> None:
        try:
            await asyncio.to_thread(self.send_message, self.chat_id, str(message))
        except Exception:
            log.warning("Gagal mengirim Telegram.", exc_info=True)

    async def _send_log_error_to_telegram(self, record: logging.LogRecord) -> None:
        if not self._running:
            return
        try:
            await self.reply(f"⚠️ {record.levelname} • {record.name}\n{record.getMessage()[:3000]}")
        except Exception:
            pass

    def _local_path(self, relative: str) -> Path:
        rel = Path(str(relative).replace("\\", "/"))
        parts = rel.parts[1:] if rel.parts and rel.parts[0] == "data" else rel.parts
        path = self._local_data_dir.joinpath(*parts) if parts else self._local_data_dir / "state.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def _remote_or_local(self, remote_path: str) -> bytes | None:
        local = self._local_path(remote_path)
        local_data: bytes | None = None
        try:
            local_data = local.read_bytes() if local.exists() else None
        except OSError:
            local_data = None

        remote_data: bytes | None = None
        try:
            remote_data, _ = await self.github.get_file(remote_path)
        except Exception as exc:
            log.warning("GitHub GET %s gagal; fallback lokal dipakai bila tersedia: %s", remote_path, exc)

        # Critical JSON snapshots are written locally before GitHub. If a GitHub
        # PUT failed, blindly preferring a still-readable older remote version
        # would resurrect stale active trades/settings/commands on restart.
        if local_data and remote_data and remote_path in {SETUPS_PATH, RUNTIME_SETTINGS_PATH, REAL_GUARD_PATH, DEFERRED_COMMANDS_PATH, BINANCE_COOLDOWN_PATH}:
            def updated_at(blob: bytes) -> float:
                try:
                    obj = json.loads(blob.decode("utf-8"))
                    stamp = parse_iso(obj.get("updated_at")) if isinstance(obj, dict) else None
                    return stamp.timestamp() if stamp else 0.0
                except Exception:
                    return 0.0
            local_stamp = updated_at(local_data)
            remote_stamp = updated_at(remote_data)
            if local_stamp >= remote_stamp:
                return local_data
            return remote_data
        if remote_data is not None:
            return remote_data
        return local_data

    async def _persist_bytes(self, remote_path: str, content: bytes, message: str) -> None:
        local = self._local_path(remote_path)
        temp = local.with_suffix(local.suffix + ".tmp")
        try:
            temp.write_bytes(content)
            temp.replace(local)
        except OSError:
            log.exception("Local fallback write gagal: %s", local)
        try:
            await self.github.replace_file(remote_path, content, message)
        except Exception as exc:
            self._history_dirty = True
            log.error("GitHub persistence gagal untuk %s; local fallback tersimpan: %s", remote_path, exc)

    async def _load_runtime_settings(self) -> None:
        raw = await self._remote_or_local(RUNTIME_SETTINGS_PATH)
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
            self.timeframe = "M15"
            raw_trend_length = payload.get("trend_length", volumatic.DEFAULT_TREND_LENGTH)
            try:
                self.trend_length = volumatic.normalize_trend_length(raw_trend_length)
            except ValueError:
                self.trend_length = volumatic.DEFAULT_TREND_LENGTH
                log.warning("trend_length tidak valid; default %s dipakai.", self.trend_length)
            raw_ema_filter = payload.get("ema_filter_length", volumatic.DEFAULT_EMA_FILTER_LENGTH)
            try:
                self.ema_filter_length = volumatic.normalize_ema_filter_length(raw_ema_filter)
            except ValueError:
                self.ema_filter_length = volumatic.DEFAULT_EMA_FILTER_LENGTH
                log.warning("ema_filter_length tidak valid; default EMA %s dipakai.", self.ema_filter_length)
            raw_h1_enabled = payload.get("h1_enabled", True)
            self.h1_enabled = raw_h1_enabled if isinstance(raw_h1_enabled, bool) else True
            try:
                self.supertrend_atr_length = volumatic.normalize_supertrend_atr_length(
                    payload.get("supertrend_atr_length", volumatic.SUPERTREND_ATR_LENGTH)
                )
            except (TypeError, ValueError):
                self.supertrend_atr_length = volumatic.SUPERTREND_ATR_LENGTH
                log.warning("supertrend_atr_length tidak valid; default %s dipakai.", self.supertrend_atr_length)
            try:
                self.supertrend_factor = volumatic.normalize_supertrend_factor(
                    payload.get("supertrend_factor", volumatic.SUPERTREND_FACTOR)
                )
            except (TypeError, ValueError):
                self.supertrend_factor = volumatic.SUPERTREND_FACTOR
                log.warning("supertrend_factor tidak valid; default %s dipakai.", self.supertrend_factor)
            try:
                loaded_rr_min, loaded_rr_max = volumatic.normalize_rr_range(
                    payload.get("rr_min", volumatic.RR_MIN),
                    payload.get("rr_max", volumatic.RR_MAX),
                )
                self.rr_min = Decimal(str(loaded_rr_min))
                self.rr_max = Decimal(str(loaded_rr_max))
            except (TypeError, ValueError):
                self.rr_min = Decimal(str(volumatic.RR_MIN))
                self.rr_max = Decimal(str(volumatic.RR_MAX))
                log.warning("RR_MIN/RR_MAX runtime tidak valid; default RR 1:%s–1:%s dipakai.", self.rr_min, self.rr_max)
            # Scanner arming is intentionally SESSION-ONLY. A fresh /try (or
            # Render restart) must never resume a previous scan automatically.
            # The only way to arm scanning is an explicit Telegram `/scan on`.
            self.scan_enabled = False
            self._scan_trigger_now = False
            self.max_active_trades = max(1, min(1000, int(payload.get("max_active_trades") or MAX_ACTIVE_TRADES)))
            self.margin_usdt = parse_decimal(payload.get("margin_usdt") or DEFAULT_MARGIN_USDT)
            self.leverage = max(1, min(125, int(payload.get("leverage") or DEFAULT_LEVERAGE)))
            self.min_margin_enabled = bool(payload.get("min_margin_enabled", False))
            if "autostop_percent" in payload:
                raw_autostop = payload.get("autostop_percent")
                if raw_autostop in (None, "", "0", 0):
                    self.autostop_percent = None
                else:
                    try:
                        guard_value = Decimal(str(raw_autostop))
                        if guard_value.is_finite() and Decimal("0") < guard_value < Decimal("100"):
                            self.autostop_percent = guard_value
                        else:
                            self.autostop_percent = None
                            log.warning("Nilai autostop_percent di runtime settings tidak valid; guard OFF.")
                    except (InvalidOperation, ValueError, TypeError):
                        self.autostop_percent = None
                        log.warning("Nilai autostop_percent di runtime settings tidak valid; guard OFF.")
            raw_peak = payload.get("autostop_peak_equity")
            raw_last = payload.get("autostop_last_equity")
            try:
                peak = Decimal(str(raw_peak)) if raw_peak not in (None, "") else None
                self._equity_peak = peak if peak is not None and peak.is_finite() and peak > 0 else None
            except (InvalidOperation, ValueError, TypeError):
                self._equity_peak = None
            try:
                last = Decimal(str(raw_last)) if raw_last not in (None, "") else None
                self._equity_last = last if last is not None and last.is_finite() else None
            except (InvalidOperation, ValueError, TypeError):
                self._equity_last = None
            self._autostop_triggered = bool(payload.get("autostop_triggered", False))
            saved_source = str(payload.get("autostop_source") or "").upper()
            self._autostop_source = saved_source if saved_source in {"REAL", "PAPER"} else None
            # Never scan between startup and the first fresh equity/mark snapshot.
            self._autostop_data_hold = bool(self.autostop_percent is not None and not self._autostop_triggered)
            if self._autostop_triggered:
                # A triggered guard must never resume scanning automatically after restart.
                self.scan_enabled = False
            processed = payload.get("last_processed_candles") or {}
            if isinstance(processed, dict):
                self.last_processed_candles = {str(k): int(v) for k, v in processed.items() if str(v).isdigit()}
            position_processed = payload.get("last_position_candles") or {}
            if isinstance(position_processed, dict):
                self.last_position_candles = {str(k): int(v) for k, v in position_processed.items() if str(v).isdigit()}
            raw_retries = payload.get("signal_retries") or {}
            if isinstance(raw_retries, dict):
                restored_retries: dict[str, dict[str, Any]] = {}
                for retry_key, retry_item in list(raw_retries.items())[-500:]:
                    if not isinstance(retry_item, dict) or not isinstance(retry_item.get("signal"), dict):
                        continue
                    try:
                        expiry = int(retry_item.get("expiry_ms") or 0)
                        signal_row = dict(retry_item["signal"])
                        if expiry > int(time.time() * 1000) and signal_row.get("symbol") and signal_row.get("direction"):
                            restored_retries[str(retry_key)] = {
                                "signal": signal_row, "expiry_ms": expiry, "key": str(retry_key),
                                "attempts": max(0, int(retry_item.get("attempts") or 0)),
                                "next_retry_at": retry_item.get("next_retry_at"),
                                "last_error": str(retry_item.get("last_error") or "")[:500],
                            }
                    except (TypeError, ValueError):
                        continue
                self._signal_retries = restored_retries
            self._scan_resume_cursor = max(0, int(payload.get("scan_resume_cursor") or 0))
            self._scan_resume_boundary_ms = max(0, int(payload.get("scan_resume_boundary_ms") or 0))
            saved_scan_tf = str(payload.get("scan_resume_timeframe") or self.timeframe).upper()
            self._scan_resume_timeframe = saved_scan_tf if saved_scan_tf in volumatic.TIMEFRAME_MS else self.timeframe
        except Exception as exc:
            log.warning("Runtime settings tidak valid; gunakan default: %s", exc)

    async def _load_real_guard_checkpoint(self) -> None:
        """Load only the persisted REAL equity ledger; never restore scanner/trades."""
        if self._real_equity_baseline is not None:
            return
        raw = await self._remote_or_local(REAL_GUARD_PATH)
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
            if "autostop_percent" in payload:
                raw_percent = payload.get("autostop_percent")
                if raw_percent in (None, "", "0", 0):
                    self.autostop_percent = None
                else:
                    try:
                        restored_percent = Decimal(str(raw_percent))
                        self.autostop_percent = restored_percent if restored_percent.is_finite() and Decimal("0") < restored_percent < Decimal("100") else None
                    except (InvalidOperation, ValueError, TypeError):
                        self.autostop_percent = None
            baseline_raw = payload.get("real_equity_baseline")
            realized_raw = payload.get("realized_since_baseline", "0")
            baseline = Decimal(str(baseline_raw)) if baseline_raw not in (None, "") else None
            realized = Decimal(str(realized_raw or "0"))
            if baseline is not None and baseline.is_finite() and baseline > 0 and realized.is_finite():
                self._real_equity_baseline = baseline
                self._realized_since_baseline = realized
                saved_source = str(payload.get("autostop_source") or "").upper()
                if saved_source == "REAL":
                    self._autostop_triggered = bool(payload.get("autostop_triggered", False))
                    for key, attr in (("autostop_peak_equity", "_equity_peak"), ("autostop_last_equity", "_equity_last")):
                        raw_value = payload.get(key)
                        try:
                            value = Decimal(str(raw_value)) if raw_value not in (None, "") else None
                        except (InvalidOperation, ValueError, TypeError):
                            value = None
                        if value is not None and value.is_finite() and value > 0:
                            setattr(self, attr, value)
                    self._autostop_source = "REAL"
                log.info("[AUTOSTOP] REAL checkpoint restored; balance baseline retained (no new balance baseline).")
        except Exception as exc:
            log.warning("[AUTOSTOP] REAL checkpoint invalid; fresh /real on snapshot will establish baseline: %s", exc)

    async def _save_real_guard_checkpoint(self) -> None:
        """Persist only the REAL account-risk ledger, independent of runtime/scanner settings."""
        if self._real_equity_baseline is None or self._real_equity_baseline <= 0:
            return  # Never overwrite a prior REAL checkpoint from a PAPER-only session.
        payload = {
            "strategy": volumatic.STRATEGY_NAME,
            "real_equity_baseline": decimal_to_str(self._real_equity_baseline),
            "realized_since_baseline": decimal_to_str(self._realized_since_baseline),
            "autostop_percent": decimal_to_str(self.autostop_percent),
            "autostop_peak_equity": decimal_to_str(self._equity_peak),
            "autostop_last_equity": decimal_to_str(self._equity_last),
            "autostop_triggered": bool(self._autostop_triggered),
            "autostop_source": self._autostop_source,
            "updated_at": iso_utc(),
        }
        await self._persist_bytes(REAL_GUARD_PATH, json.dumps(payload, indent=2).encode(), "volumatic: real equity guard checkpoint")

    async def _save_runtime_settings(self) -> None:
        async with self._settings_lock:
            payload = {
                "strategy": volumatic.STRATEGY_NAME,
                "timeframe": self.timeframe,
                "trend_length": self.trend_length,
                "ema_filter_length": list(self.ema_filter_length) if isinstance(self.ema_filter_length, tuple) else self.ema_filter_length,
                "h1_enabled": bool(self.h1_enabled),
                "supertrend_atr_length": self.supertrend_atr_length,
                "supertrend_factor": self.supertrend_factor,
                "rr_min": decimal_to_str(self.rr_min),
                "rr_max": decimal_to_str(self.rr_max),
                # Deliberately do not persist scan_enabled. Scanner requires an
                # explicit `/scan on` in every fresh engine session.
                "max_active_trades": self.max_active_trades,
                "margin_usdt": decimal_to_str(self.margin_usdt),
                "leverage": self.leverage,
                "min_margin_enabled": self.min_margin_enabled,
                "last_processed_candles": dict(list(self.last_processed_candles.items())[-3000:]),
                "last_position_candles": dict(list(self.last_position_candles.items())[-3000:]),
                "signal_retries": {
                    k: {
                        "signal": v.get("signal"),
                        "expiry_ms": int(v.get("expiry_ms") or 0),
                        "attempts": int(v.get("attempts") or 0),
                        "next_retry_at": v.get("next_retry_at"),
                        "last_error": str(v.get("last_error") or "")[:500],
                    }
                    for k, v in list(self._signal_retries.items())[-500:]
                },
                "scan_resume_cursor": int(self._scan_resume_cursor),
                "scan_resume_boundary_ms": int(self._scan_resume_boundary_ms),
                "scan_resume_timeframe": self._scan_resume_timeframe,
                "updated_at": iso_utc(),
            }
            await self._persist_bytes(RUNTIME_SETTINGS_PATH, json.dumps(payload, indent=2).encode(), "volumatic: runtime settings")

    @staticmethod
    def _deferred_command_allowed(command_text: str) -> bool:
        parts = str(command_text or "").strip().split(maxsplit=1)
        if not parts:
            return False
        command = parts[0].split("@", 1)[0].lower()
        argument = parts[1].strip().lower() if len(parts) > 1 else ""
        if command == "/real":
            return argument == "on"  # /real off is local and must always execute immediately.
        return command in {"/close", "/del", "/balance"}

    async def _save_deferred_commands(self) -> None:
        payload = {
            "strategy": volumatic.STRATEGY_NAME,
            "commands": self.deferred_commands[-500:],
            "updated_at": iso_utc(),
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        # _persist_bytes writes the local fallback first, then attempts GitHub.
        await self._persist_bytes(self.session_deferred_commands_path, content, "volumatic: session deferred command queue")

    async def _load_deferred_commands(self) -> None:
        raw = await self._remote_or_local(DEFERRED_COMMANDS_PATH)
        if not raw:
            self.deferred_commands = []
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
            rows = payload.get("commands", []) if isinstance(payload, dict) else []
            restored: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                command_text = str(row.get("command") or "").strip()
                if not self._deferred_command_allowed(command_text):
                    continue
                restored.append({
                    "command_id": str(row.get("command_id") or uuid4().hex),
                    "command": command_text,
                    "queued_at": str(row.get("queued_at") or iso_utc()),
                    "attempts": max(0, int(row.get("attempts") or 0)),
                    "next_retry_at": str(row.get("next_retry_at") or ""),
                    "last_error": str(row.get("last_error") or "")[:500],
                })
            self.deferred_commands = restored[-500:]
        except Exception as exc:
            log.warning("Deferred command queue tidak valid; queue lokal dimulai kosong: %s", exc)
            self.deferred_commands = []

    async def _enqueue_deferred_command(self, command_text: str, reason: str) -> None:
        command_text = str(command_text or "").strip()
        if not self._deferred_command_allowed(command_text):
            raise ValueError("Command ini tidak diizinkan masuk antrean Binance.")
        async with self._deferred_commands_lock:
            existing = next((x for x in self.deferred_commands if x.get("command") == command_text), None)
            if existing is None:
                if len(self.deferred_commands) >= 500:
                    raise RuntimeError("Antrean perintah penuh (500). Selesaikan antrean lebih dahulu.")
                self.deferred_commands.append({
                    "command_id": uuid4().hex,
                    "command": command_text,
                    "queued_at": iso_utc(),
                    "attempts": 0,
                    "next_retry_at": "",
                    "last_error": str(reason)[:500],
                })
            else:
                existing["last_error"] = str(reason)[:500]
            await self._save_deferred_commands()

    async def _remove_deferred_command(self, command_id: str) -> None:
        async with self._deferred_commands_lock:
            self.deferred_commands = [x for x in self.deferred_commands if x.get("command_id") != command_id]
            await self._save_deferred_commands()

    async def _cancel_queued_real_on(self) -> None:
        async with self._deferred_commands_lock:
            before = len(self.deferred_commands)
            def is_queued_real_on(item: dict[str, Any]) -> bool:
                parts = str(item.get("command") or "").strip().split(maxsplit=1)
                return bool(parts) and parts[0].split("@", 1)[0].lower() == "/real" and len(parts) > 1 and parts[1].strip().lower() == "on"
            self.deferred_commands = [x for x in self.deferred_commands if not is_queued_real_on(x)]
            if len(self.deferred_commands) != before:
                await self._save_deferred_commands()

    async def _process_deferred_commands(self) -> None:
        if not self.deferred_commands or self._binance_cooldown_remaining() > 0:
            return
        # FIFO, one API-dependent command per maintenance pass. The item is
        # retained until the handler completes; repeated execution is designed
        # to be idempotent (e.g. /close only sees trades still active).
        item = self.deferred_commands[0]
        command_text = str(item.get("command") or "")
        retry_at = parse_iso(item.get("next_retry_at")) if item.get("next_retry_at") else None
        if retry_at is not None and now_utc() < retry_at:
            return

        async def defer_retry(error: Exception) -> None:
            attempts = int(item.get("attempts") or 0) + 1
            item["attempts"] = attempts
            delay = min(300, max(5, 2 ** min(attempts, 8)))
            item["next_retry_at"] = iso_utc(now_utc() + timedelta(seconds=delay))
            item["last_error"] = str(error)[:500]
            await self._save_deferred_commands()
            log.info("[COMMAND QUEUE] deferred retry scheduled | command=%s | retry_in=%ss | reason=%s", command_text, delay, error)

        try:
            await self._dispatch_command(command_text)
        except BinanceRateLimitError as exc:
            await defer_retry(exc)
            return
        except BinanceAPIError as exc:
            # Transport errors / 5xx are transient. Authentication/parameter
            # errors are permanent and should be reported rather than retried forever.
            if exc.status_code is None or exc.status_code >= 500:
                await defer_retry(exc)
                return
            log.error("Deferred command %s ditolak Binance secara permanen: %s", command_text, exc)
            await self.reply(f"❌ Perintah tertunda {command_text} ditolak Binance: {str(exc)[:700]}")
        except (requests.RequestException, ConnectionError, TimeoutError, asyncio.TimeoutError) as exc:
            await defer_retry(exc)
            return
        except Exception as exc:
            log.exception("Deferred command %s gagal permanen", command_text)
            await self.reply(f"❌ Perintah tertunda {command_text} gagal dan dikeluarkan dari antrean: {str(exc)[:700]}")
        await self._remove_deferred_command(str(item.get("command_id") or ""))
        await self.reply(f"✅ Perintah tertunda selesai diproses: {command_text}")

    async def _on_egress_changed(self, info: dict[str, Any]) -> None:
        """Apply a proxy-manager decision: reset local cooldown copies, reconnect sockets, tell Telegram."""
        try:
            if info.get("kind") == "switched":
                if BINANCE_COOLDOWN.remaining() <= 0:
                    # The new egress is clean: drop per-client copies of the old IP's cooldown.
                    self.rest._rest_cooldown_until = 0.0
                    self.real._cooldown_until = 0.0
                    self.real._server_cooldown_until = 0.0
                    self.real._early_cooldown_until = 0.0
                for label, sock in (("market", self.ws), ("user", self.user_ws)):
                    raw_ws = getattr(sock, "_ws", None)
                    if raw_ws is not None:
                        try:
                            await raw_ws.close()
                            log.info("[PROXY] WebSocket %s ditutup agar tersambung ulang lewat route baru", label)
                        except Exception:
                            log.debug("[PROXY] gagal menutup WS %s", label, exc_info=True)
                if info.get("mode") == PROXY_MANAGER.AUTO:
                    risk = "\nCatatan: proxy gratis tidak dijamin stabil; bot berotasi otomatis bila gagal."
                else:
                    risk = ""
                await self.reply(
                    "╔══ BINANCE / EGRESS SWITCH ══╗\n"
                    f"DARI      : {info.get('old')}\n"
                    f"KE        : {info.get('new')}\n"
                    f"IP KELUAR : {info.get('exit_ip') or 'tidak terdeteksi'}  <- whitelist IP ini di Binance API key\n"
                    f"IP LAMA   : {info.get('old_ip') or '?'}\n"
                    f"LATENSI   : {float(info.get('latency_ms') or 0):.0f} ms (uji /fapi/v1/time OK)\n"
                    f"ALASAN    : {str(info.get('reason'))[:300]}\n"
                    "TERPASANG : REST + market WS + user WS (reconnect otomatis)"
                    f"{risk}"
                )
            elif info.get("kind") == "exhausted":
                await self.reply(
                    "╔══ BINANCE / EGRESS FAILOVER GAGAL ══╗\n"
                    f"ALASAN    : {str(info.get('reason'))[:300]}\n"
                    f"ROUTE     : {info.get('route')}\n"
                    f"RETRY     : {float(info.get('retry_seconds') or 0):.0f}s lagi\n"
                    "Tidak ada proxy sehat yang lolos uji ke Binance. Order LIMIT yang sudah ada tetap aman di Binance."
                )
        except Exception:
            log.exception("[PROXY] gagal memproses perubahan egress")

    def _binance_cooldown_remaining(self) -> float:
        return max(
            BINANCE_COOLDOWN.remaining(),
            self.rest._rest_cooldown_until - time.monotonic(),
            self.real.cooldown_remaining,
        )

    @staticmethod
    def _short_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    async def _notify_binance_cooldown(
        self, exc: Exception | None = None, *, source: str = "ENGINE", symbol: str | None = None
    ) -> None:
        """Send a single robot-style state notice per meaningful cooldown transition."""
        if isinstance(exc, BinanceRateLimitError) and getattr(exc, "internal", False):
            return  # locally-held requests are not new exchange events
        remaining = self._binance_cooldown_remaining()
        if remaining <= 0:
            return
        text_value = str(exc or BINANCE_COOLDOWN.reason())
        match = re.search(r"HTTP\s+(418|429)", text_value)
        status = int(getattr(exc, "status_code", 0) or (match.group(1) if match else 429))
        endpoint = str(getattr(exc, "endpoint", "") or "")
        if not endpoint:
            endpoint_match = re.search(r"(/fapi/[^ :]+)", text_value)
            endpoint = endpoint_match.group(1) if endpoint_match else "REST"
        until_epoch = float(BINANCE_COOLDOWN._until_epoch or (time.time() + remaining))
        now_mono = time.monotonic()
        # Notify at first occurrence, status escalation 429->418, or materially
        # extended server ban. Suppress repeated warnings from the same cooldown.
        first = self._binance_notice_last_sent <= 0
        escalated = self._binance_notice_status != status
        extended = until_epoch > self._binance_notice_until_epoch + 120.0
        heartbeat = now_mono - self._binance_notice_last_sent >= 900.0
        if not (first or escalated or extended or heartbeat):
            return
        self._binance_notice_last_sent = now_mono
        self._binance_notice_status = status
        self._binance_notice_until_epoch = max(self._binance_notice_until_epoch, until_epoch)
        local_until = datetime.fromtimestamp(until_epoch, timezone.utc).astimezone(TZ)
        used = "—"
        if API_STATS.used_weight is not None and API_STATS.used_weight_at is not None and time.monotonic() - API_STATS.used_weight_at <= 60:
            used = f"{API_STATS.used_weight}/{BINANCE_IP_WEIGHT_LIMIT}"
        active_real = sum(1 for t in self.active_trades.values() if t.real_enabled and t.status in {"PENDING", "FILLED"})
        gtc_real = sum(1 for t in self.active_trades.values() if t.real_enabled and t.status in {"PENDING", "FILLED"} and (t.preserve_gtc_after_cooldown or t.real_submit_ambiguous))
        mode = "IP BAN" if status == 418 else "RATE THROTTLE"
        lines = [
            "╔══ BINANCE / RATE GOVERNOR ══╗",
            f"STATE     : {mode} · HTTP {status}",
            f"SOURCE    : {source}" + (f" · {symbol}" if symbol else ""),
            f"ENDPOINT  : {endpoint}",
            f"IP WEIGHT : {used}",
            f"COOLDOWN  : {self._short_duration(remaining)} remaining",
            f"RESUME ETA: {local_until.strftime('%d-%m-%Y %H:%M:%S WIB')}",
            "──────────────────────────────",
            "REST      : PAUSED · no request burst",
            f"ORDERS    : {active_real} REAL active · {gtc_real} preserved/uncertain",
            f"USER WS   : {self.user_ws.status} · private",
            f"MARKET WS : {self.ws.status} · {len(self.ws.symbols())} symbols",
            f"CMD QUEUE : {len(self.deferred_commands)} waiting",
            f"SCANNER   : {'PAUSED' if self.scan_enabled else 'OFF'} · cursor {self._scan_resume_cursor}",
            "──────────────────────────────",
            "Existing Binance LIMIT orders are not canceled by cooldown.",
            "WebSocket events continue; REST reconciliation resumes automatically.",
        ]
        await self.reply("\n".join(lines))
        log.info("[BINANCE] rate governor notified | status=%s endpoint=%s source=%s remaining=%.0fs", status, endpoint, source, remaining)

    async def _notify_binance_recovered(self) -> None:
        if not self._was_in_binance_cooldown:
            return
        await self.reply(
            "╔══ BINANCE / LINK RESTORED ══╗\n"
            "STATE     : NORMAL\n"
            "REST      : AVAILABLE\n"
            "NEXT      : reconcile orders → positions → deferred commands → scanner\n"
            f"MARKET WS : {self.ws.status} · {len(self.ws.symbols())} symbols\n"
            f"CMD QUEUE : {len(self.deferred_commands)} waiting\n"
            "Existing LIMIT orders have been preserved during cooldown."
        )
        self._binance_notice_last_sent = 0.0
        self._binance_notice_status = None
        self._binance_notice_until_epoch = 0.0

    async def _restore_cooldown_from_github(self) -> None:
        """Restore a shared IP ban only when GitHub sharing is explicitly enabled.

        A GitHub marker may belong to a different machine/public egress IP.
        Local Windows runs therefore restore only their own data/ marker by default.
        """
        if not BINANCE_COOLDOWN_GITHUB_SYNC:
            log.info("[BINANCE] GitHub cooldown marker diabaikan; memakai state IP lokal saja.")
            return
        raw = await self._remote_or_local(BINANCE_COOLDOWN_PATH)
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
                BINANCE_COOLDOWN.merge_persisted_payload(payload)
                self._last_cooldown_github_update = float(payload.get("updated_epoch") or 0.0)
            except Exception:
                log.warning("Persisted Binance cooldown JSON invalid; keep local marker only.", exc_info=True)
        self._was_in_binance_cooldown = self._binance_cooldown_remaining() > 0

    async def _sync_cooldown_marker_to_github(self) -> None:
        """Publish a changed cooldown marker only for a deliberately shared IP."""
        if not BINANCE_COOLDOWN_GITHUB_SYNC:
            return
        path = BINANCE_COOLDOWN._state_path
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            updated = float(payload.get("updated_epoch") or 0.0)
            if updated <= self._last_cooldown_github_update:
                return
            await self.github.replace_file(
                BINANCE_COOLDOWN_PATH, raw, "volumatic: persist Binance IP cooldown"
            )
            self._last_cooldown_github_update = updated
            log.info("[BINANCE] cooldown marker tersinkron ke GitHub untuk recovery lintas-restart.")
        except FileNotFoundError:
            return
        except Exception as exc:
            log.warning("[BINANCE] cooldown marker GitHub gagal disinkron; local fallback tetap aktif: %s", exc)

    async def _load_history(self) -> None:
        raw = await self._remote_or_local(HISTORY_TRADES_PATH)
        if raw:
            try:
                records = json.loads(raw.decode("utf-8"))
                self.history_records = [x for x in records if isinstance(x, dict)] if isinstance(records, list) else []
            except Exception:
                log.exception("Trade history JSON tidak valid; fallback kosong.")
                self.history_records = []
        raw_events = await self._remote_or_local(HISTORY_EVENTS_PATH)
        self.history_events = []
        if raw_events:
            for line in raw_events.decode("utf-8", errors="replace").splitlines():
                try:
                    item = json.loads(line)
                    if isinstance(item, dict): self.history_events.append(item)
                except json.JSONDecodeError:
                    log.warning("events.jsonl berisi satu baris invalid; dilewati.")
        self._history_dirty = False

    async def _load_notes(self) -> None:
        raw = await self._remote_or_local(NOTES_PATH)
        if not raw:
            self.notes = []
            return
        try:
            data = json.loads(raw.decode("utf-8"))
            self.notes = [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
        except Exception:
            log.exception("Notes JSON tidak valid.")
            self.notes = []

    async def add_note(self, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            await self.show_notes()
            return
        self.notes.append({"note_id": uuid4().hex[:12].upper(), "text": value[:3000], "created_at": iso_utc(), "created_at_wib": format_wib(now_utc())})
        await self._persist_bytes(self.session_notes_path, json.dumps(self.notes, ensure_ascii=False, indent=2).encode(), "volumatic: add session note")
        await self.reply("✅ Catatan tersimpan.")

    async def show_notes(self) -> None:
        if not self.notes:
            await self.reply("Belum ada catatan. Gunakan /catatan isi catatan.")
            return
        lines = [f"{x.get('note_id')} • {x.get('created_at_wib', '-')}\n{x.get('text', '')}" for x in self.notes[-30:]]
        await self.reply("CATATAN\n\n" + "\n\n".join(lines))

    async def _persist_history(self) -> None:
        async with self._history_lock:
            trades = json.dumps(self.history_records, ensure_ascii=False, indent=2).encode("utf-8")
            events = ("\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in self.history_events[-20000:]) + ("\n" if self.history_events else "")).encode("utf-8")
            await self._persist_bytes(self.session_history_trades_path, trades, "volumatic: update session trade history")
            await self._persist_bytes(self.session_history_events_path, events, "volumatic: update session event log")

    async def _record_event(self, event: str, trade: Trade | None = None, **extra: Any) -> None:
        item = {
            "event_id": uuid4().hex,
            "event": event,
            "timestamp": iso_utc(),
            "timestamp_wib": format_wib(now_utc()),
            "strategy": volumatic.STRATEGY_NAME,
            "trade_id": trade.trade_id if trade else extra.pop("trade_id", None),
            "pair": trade.pair if trade else extra.pop("pair", None),
            "direction": trade.direction if trade else extra.pop("direction", None),
            "timeframe": trade.timeframe if trade else extra.pop("timeframe", self.timeframe),
            **extra,
        }
        self.history_events.append(item)
        await self._persist_history()
        log.info("[EVENT] %s %s %s", event, item.get("pair") or "-", item.get("trade_id") or "-")

    async def _save_active_state(self) -> None:
        payload = {
            "strategy": volumatic.STRATEGY_NAME,
            "version": volumatic.STRATEGY_VERSION,
            "updated_at": iso_utc(),
            "active_trades": [t.to_record() for t in self.active_trades.values() if t.status in {"PENDING", "FILLED"}],
        }
        await self._persist_bytes(SETUPS_PATH, json.dumps(payload, ensure_ascii=False, indent=2).encode(), "volumatic: active trade state")

    async def _load_active_state(self) -> None:
        self.legacy_active_snapshot_count = 0
        raw = await self._remote_or_local(SETUPS_PATH)
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
            rows = payload.get("active_trades", []) if isinstance(payload, dict) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                status = str(row.get("status") or "").upper()
                row_strategy = str(row.get("strategy_name") or row.get("strategy") or "").upper()
                if row_strategy != volumatic.STRATEGY_NAME:
                    if status in {"PENDING", "FILLED"}:
                        self.legacy_active_snapshot_count += 1
                    continue
                try:
                    trade = Trade.from_record(row)
                    if trade.direction not in {"LONG", "SHORT"} or trade.timeframe not in volumatic.TIMEFRAME_MS:
                        raise ValueError("Arah/timeframe snapshot tidak valid.")
                except Exception:
                    log.exception("Trade Volumatic gagal dipulihkan.")
                    continue
                if trade.status in {"PENDING", "FILLED"} and trade.pair:
                    self.active_trades[trade.trade_id] = trade
            if self.legacy_active_snapshot_count:
                log.critical(
                    "[MIGRATION] %s trade aktif dari strategi lama tidak diadopsi otomatis. "
                    "Periksa/selesaikan posisi, pending order, dan algo order lama di Binance.",
                    self.legacy_active_snapshot_count,
                )
        except Exception as exc:
            log.warning("Active trade snapshot gagal dibaca: %s", exc)

    async def save_setups(self) -> None:
        await self._save_active_state()
        await self.reply(f"✅ {sum(t.status in {'PENDING','FILLED'} for t in self.active_trades.values())} trade aktif disimpan ke GitHub.")

    async def open_setups(self) -> None:
        if self.active_trades:
            await self.reply("Trade aktif sudah dimuat. Tidak akan menimpa state yang sedang berjalan.")
            return
        await self._load_active_state()
        if self._binance_cooldown_remaining() > 0:
            if self.active_trades:
                await self.ws.start()
            await self._sync_ws_symbols()
            for restored_trade in list(self.active_trades.values()):
                if restored_trade.status == "FILLED" and not restored_trade.real_enabled:
                    self._schedule_ema_exit_initialization(restored_trade)
            await self.reply(
                f"✅ Snapshot lokal dimuat. Trade aktif: {len(self.active_trades)}. "
                f"Binance REST cooldown masih {self._binance_cooldown_remaining():.0f}s; "
                "rekonsiliasi exchange akan berjalan otomatis setelah API pulih."
            )
            return
        await self._reconcile_restored_trades()
        if self.active_trades:
            await self.ws.start()
        await self._sync_ws_symbols()
        for restored_trade in list(self.active_trades.values()):
            if restored_trade.status == "FILLED" and not restored_trade.real_enabled:
                self._schedule_ema_exit_initialization(restored_trade)
        await self.reply(f"✅ Snapshot dimuat dan direkonsiliasi. Trade aktif: {len(self.active_trades)}.")

    # ----------------------------- Startup -------------------------------
    async def start(self) -> None:
        """Start a fresh in-memory engine session without restoring prior data.

        /try is intentionally a clean boot: no runtime settings, history, notes,
        active-trade snapshots, deferred commands, cooldown markers, saved
        listenKeys, scanner cursors, or previous Autostop state are imported.
        Exchange market data is fetched only after an explicit /scan on; private
        account data is fetched only after an explicit /real on.
        """
        if self._running:
            return
        self._running = True
        log.info(
            "[BINANCE] outbound route configured | egress_id=%s | proxy=%s (REST + WebSocket)",
            PROXY_MANAGER.egress_id, "ON" if PROXY_MANAGER.active_url else "OFF",
        )
        log.info(
            "[PROXY] auto-failover=%s | route=%s | home=%s | sumber=ProxyScrape | pool=%d | uji<=%d kandidat, timeout=%.0fs | "
            "urutan: DIRECT/Render -> BINANCE_PROXY_URL -> pool ProxyScrape (divalidasi ke Binance sebelum dipakai)",
            "ON" if PROXY_MANAGER.enabled else "OFF", PROXY_MANAGER.describe(), PROXY_MANAGER.home_mode,
            PROXY_POOL_SIZE, PROXY_MAX_CANDIDATES, PROXY_TEST_TIMEOUT_SECONDS,
        )
        PROXY_MANAGER.bind(asyncio.get_running_loop(), self._on_egress_changed)
        PROXY_MANAGER.refresh_exit_ip_async("startup")
        self.real_mode = False
        self.scan_enabled = False
        self._scan_trigger_now = False
        self.scan_schedule_slots = ()
        self.scan_schedule_waiting = False
        self.scan_schedule_activated_at = None
        self._scan_schedule_last_fired_minute = None
        self.timeframe = "M15"
        self.trend_length = volumatic.DEFAULT_TREND_LENGTH
        self.ema_filter_length = volumatic.DEFAULT_EMA_FILTER_LENGTH
        self.h1_enabled = True
        self._h1_toggle_generation = 0
        self.supertrend_atr_length = volumatic.SUPERTREND_ATR_LENGTH
        self.supertrend_factor = volumatic.SUPERTREND_FACTOR
        self.margin_usdt = DEFAULT_MARGIN_USDT
        self.leverage = DEFAULT_LEVERAGE
        self.min_margin_enabled = False
        self.max_active_trades = MAX_ACTIVE_TRADES
        self.active_trades.clear()
        self._ema_exit_init_tasks.clear()
        self._ema_exit_last_retry.clear()
        self._ema_exit_close_last_check.clear()
        self.exchange_positions.clear()
        self.exchange_open_orders.clear()
        self.exchange_snapshot_at = None
        self._last_user_ws_reconnect_count = 0
        self.history_records.clear()
        self.history_events.clear()
        self.notes.clear()
        self.deferred_commands.clear()
        self.last_processed_candles.clear()
        self.last_position_candles.clear()
        self._live_signal_candles.clear()
        self._live_signal_last_eval.clear()
        self._live_signal_seen.clear()
        self._signal_retries.clear()
        self._scan_resume_cursor = 0
        self._scan_resume_boundary_ms = 0
        self._scan_resume_timeframe = "M15"
        self._scan_resume_trend_length = self.trend_length
        self._scan_interrupted_by_cooldown = False
        self._scan_capacity_paused = False
        self._scan_protection_paused = False
        self._was_in_binance_cooldown = False
        # Restore a persisted Binance ban only when its saved egress identity matches
        # this process. Legacy markers without egress_id and markers from another
        # proxy/direct IP are ignored; matching-IP bans remain protected on restart.
        try:
            BINANCE_COOLDOWN._restore_persisted_cooldown()
            await self._restore_cooldown_from_github()
            if self._binance_cooldown_remaining() > 0:
                self._rest_recovery_active = True
                log.warning("[START] Binance IP cooldown restored: %.0fs remaining; REST held.", self._binance_cooldown_remaining())
        except Exception:
            log.warning("[START] cooldown restore failed", exc_info=True)
        self._last_scan_boundary_ms = 0
        self.symbols.clear()
        self._scanner_symbols.clear()
        self.prices.clear()

        if not self._telegram_error_handler_added:
            for handler in list(log.handlers):
                if isinstance(handler, TelegramErrorHandler) and handler is not self._telegram_error_handler:
                    log.removeHandler(handler)
            self._telegram_error_handler.attach()
            log.addHandler(self._telegram_error_handler)
            self._telegram_error_handler_added = True

        # No exchangeInfo, market WS, private WS, account, or GitHub state request
        # is performed during startup. Telegram remains responsive and scanner OFF.
        self._real_equity_baseline = None
        self._realized_since_baseline = Decimal("0")
        self._autostop_notice_state = None
        self._autostop_last_state_notice_at = 0.0
        if self.autostop_percent is not None:
            self._equity_peak = None
            self._equity_last = None
            self._autostop_triggered = False
            self._autostop_source = None
            self._autostop_data_hold = True
        else:
            self._equity_peak = None
            self._equity_last = None
            self._autostop_triggered = False
            self._autostop_data_hold = False
            self._autostop_source = None

        self._maintenance_task = asyncio.create_task(self._maintenance_loop(), name="volumatic-maintenance")
        self._scan_task = asyncio.create_task(self._scan_loop(), name="volumatic-scanner")
        log.info(
            "[START] FRESH Volumatic session | scanner=OFF | timeframe=M15 | length=%s | "
            "EMA=%s | no persisted engine state imported",
            self.trend_length,
            ema_filter_label(self.ema_filter_length),
        )
        await self.reply(
            "🟢 VOLUMATIC TREND / FRESH SESSION\n\n"
            "Runtime state : CLEAN (tidak memuat data sesi sebelumnya)\n"
            "Signal TF     : M15\nEMA Trend TF  : H4\n"
            f"Trend length  : {self.trend_length} (default {volumatic.DEFAULT_TREND_LENGTH})\n"
            f"EMA filter    : {ema_filter_label(self.ema_filter_length)}\n"
            f"H1 logic      : {'ON' if self.h1_enabled else 'OFF'} (entry filter + inverse-cross exit)\n"
            f"SuperTrend    : ATR {self.supertrend_atr_length} • Factor {self.supertrend_factor:g}\n"
            f"Max positions : {self.max_active_trades}\n"
            "Scanner       : OFF — menunggu /scan on atau /scan schedule\n"
            "REAL          : OFF — menunggu /real on\n"
            "Active        : 0\n"
            "Market WS     : OFF — aktif saat scanner dimulai\n"
            "Private WS    : OFF — aktif setelah /real on\n"
            "Startup I/O   : tidak mengambil exchangeInfo, balance, posisi, order, atau snapshot GitHub.\n\n"
            "PERHATIAN: posisi dan order LIVE yang masih ada di Binance tidak diadopsi oleh sesi baru. "
            "Periksa akun Binance sebelum mengaktifkan /real on."
        )

    async def stop(self) -> None:
        """Immediately stop this in-memory engine session without touching exchange or saved data.

        This does NOT cancel Binance orders, close positions, delete listenKeys,
        save snapshots, write runtime settings, or alter journals. Those exchange
        objects (if any) remain on Binance and must be reviewed before a fresh /try.
        """
        if not self._running:
            return
        self._running = False
        PROXY_MANAGER.bind(None, None)
        self.scan_enabled = False
        self._scan_trigger_now = False
        self.scan_schedule_slots = ()
        self.scan_schedule_waiting = False
        self.scan_schedule_activated_at = None
        self._scan_schedule_last_fired_minute = None
        self.real_mode = False

        # Signal every loop/socket worker and every per-trade H1 warm-up task
        # to stop. EMA initialization tasks are explicitly tracked so they cannot
        # outlive /end or a Render redeploy and mutate a fresh engine instance.
        ema_workers = list(self._ema_exit_init_tasks.values())
        workers = [self._scan_task, self._maintenance_task, self.ws._task, self.user_ws._task, *ema_workers]
        for component in (self.ws, self.user_ws):
            try:
                component._stop.set()
            except Exception:
                pass
        for task in workers:
            if task is not None and not task.done():
                task.cancel()

        # Close sockets best-effort with a strict short bound; never send REST
        # DELETE/cancel/close actions to Binance during /end.
        close_jobs = []
        for component in (self.ws, self.user_ws):
            socket = getattr(component, "_ws", None)
            if socket is not None:
                try:
                    close_jobs.append(asyncio.create_task(socket.close()))
                except Exception:
                    pass
        if close_jobs:
            done, pending = await asyncio.wait(close_jobs, timeout=0.20)
            for task in pending:
                task.cancel()
        live_workers = [task for task in workers if task is not None and not task.done()]
        if live_workers:
            done, pending = await asyncio.wait(live_workers, timeout=0.35)
            for task in pending:
                task.cancel()

        self._scan_task = None
        self._maintenance_task = None
        self._ema_exit_init_tasks.clear()
        self._ema_exit_close_locks.clear()
        self._ema_exit_close_last_check.clear()
        self._ema_exit_last_retry.clear()
        self.ws._task = None
        self.user_ws._task = None
        self.ws._ws = None
        self.user_ws._ws = None
        self.ws.connected = False
        self.user_ws.connected = False
        self.ws._desired_symbols.clear()
        self.ws._subscribed_symbols.clear()

        # Local resource release only; no network request is performed here.
        try:
            self.rest._session.close()
        except Exception:
            pass
        self.active_trades.clear()
        self.prices.clear()
        self._live_signal_candles.clear()
        self._live_signal_last_eval.clear()
        self._live_signal_seen.clear()
        self._scanner_symbols.clear()
        self._signal_retries.clear()
        if self._telegram_error_handler_added:
            try:
                log.removeHandler(self._telegram_error_handler)
            except Exception:
                pass
            self._telegram_error_handler_added = False
        self.session_id = ""
        log.info("[STOP] Engine OFF (fresh-stop); no persistence writes and no Binance order/position mutation.")

    # ------------------------- Universe / scanner ------------------------
    async def _universe(self) -> list[str]:
        now = time.monotonic()
        if not self.symbols:
            # First boot still requires an authoritative exchangeInfo snapshot.
            self.symbols = await self.rest.get_exchange_info(force=False)
            self._last_universe_refresh = now
        elif now - self._last_universe_refresh > UNIVERSE_REFRESH_SECONDS:
            # Recovery has two independent asyncio loops (scanner + maintenance).
            # The scanner can reach _universe before maintenance clears the ban
            # flag, so retain the last-good metadata and advance its refresh gate
            # here too. This prevents a stale exchangeInfo request racing the
            # first post-cooldown scan batch.
            if self._was_in_binance_cooldown and self.symbols:
                self._last_universe_refresh = now
                log.info(
                    "[UNIVERSE] post-cooldown recovery uses %s cached symbols; "
                    "hourly exchangeInfo refresh deferred.", len(self.symbols),
                )
            else:
                try:
                    refreshed = await self.rest.get_exchange_info(force=True)
                except BinanceRateLimitError as exc:
                    self._last_universe_refresh = now
                    log.warning(
                        "[UNIVERSE] exchangeInfo refresh deferred by Binance rate governor; "
                        "using %s cached symbols for this session.", len(self.symbols),
                    )
                    await self._notify_binance_cooldown(exc, source="UNIVERSE REFRESH")
                else:
                    self.symbols = refreshed
                    self._last_universe_refresh = now
        result = [
            symbol for symbol, meta in self.symbols.items()
            if meta.status == "TRADING" and meta.contract_type == "PERPETUAL"
            and meta.quote_asset == "USDT" and symbol not in EXCLUDED_SYMBOLS
        ]
        return sorted(result)

    async def _send_scan_batch_report(
        self, *, timeframe: str, trend_length: int, ema_length: int | tuple[int, int] | None, batch_number: int,
        batch_start: int, batch_end: int, universe_total: int,
        stats: dict[str, Any], elapsed: float, interrupted: bool = False,
        supertrend_atr_length: int, supertrend_factor: float,
        rr_min: Decimal, rr_max: Decimal,
        configuration_changed: bool = False, pass_start: int | None = None,
    ) -> None:
        """Telegram telemetry for every completed or interrupted logical batch."""
        ema = ema_filter_label(ema_length)
        cooldown = self._binance_cooldown_remaining()
        used_slots, pending_only_slots, filled_slots, live_limits = self._active_slot_counts()
        capacity_full = used_slots >= self.max_active_trades
        pending_limit_exists = self._has_pending_entry_limit()
        state = (
            "REST COOLDOWN / PAUSED" if interrupted or cooldown > 0 else
            "PAUSED — TP/SL NOT ACTIVE" if self._has_unprotected_real_position() else
            "PAUSED — MAX SLOT REACHED" if capacity_full else
            "CONFIG CHANGED / RESTART" if configuration_changed else
            "SCANNING · PENDING LIMIT RESERVED" if pending_limit_exists else
            "BATCH COMPLETE"
        )
        analyzed = int(stats.get("analyzed", stats.get("processed", 0)))
        rejected = int(stats.get("rejected", 0))
        rejection_reasons = stats.get("rejection_reasons") or {}
        requested = int(stats.get("requested", 0))
        pass_start = batch_start if pass_start is None else pass_start
        pass_expected = max(0, batch_end - pass_start + 1)
        blue = int(stats.get("blue", 0))
        red = int(stats.get("red", 0))
        no_signal = int(stats.get("no_signal", 0))
        skipped = sum(int(stats.get(k, 0)) for k in ("already_seen", "warmup", "stale"))
        filled = int(stats.get("filled", 0))
        created = int(stats.get("orders", 0))
        deferred = int(stats.get("deferred", 0))
        unknown = int(stats.get("unknown", 0))
        lines = [
            "╔══ VOL / SCAN BATCH REPORT ══╗",
            f"ENGINE    : M15 VOLUMATIC · H4 EMA ALT+BTC · M15 SUPERTREND · LEN{trend_length} · {ema} · ATR{supertrend_atr_length} · FACTOR{supertrend_factor:g}",
            f"H1 LOGIC  : {'ON' if self.h1_enabled else 'OFF'} · ENTRY FILTER + INVERSE-CROSS EXIT",
            f"TP RULE   : BASE RR 1:{fmt_num(rr_min, 3)} · LIQUIDITY RANGE 1:{fmt_num(rr_min, 3)}–1:{fmt_num(rr_max, 3)}",
            f"BATCH     : {batch_number:02d} / COIN #{batch_start:04d}–#{batch_end:04d}",
            f"UNIVERSE  : {universe_total} symbols",
            f"STATE     : {state}",
            f"SLOTS     : {used_slots}/{self.max_active_trades} · PENDING {pending_only_slots} · FILLED {filled_slots} · LIVE LIMITS {live_limits}",
            "──────────────────────────────",
            f"PASS      : {pass_start:04d}–{min(batch_end, pass_start + requested - 1):04d}  ({requested}/{pass_expected} requested)",
            f"ANALYZED  : {analyzed}  ·  SKIPPED {skipped}",
            f"BUBBLES   : BLUE {blue}  ·  RED {red}  ·  NONE {no_signal}",
            f"REJECTED  : {rejected}",
            f"EXECUTION : NEW {created}  ·  FILLED {filled}  ·  DEFERRED {deferred}  ·  UNKNOWN {unknown}",
            f"BTC FILTER: {stats.get('btc_filter_status', 'UNKNOWN')}",
            f"FILTER    : ALT H4 BLOCK {int(stats.get('ema_blocked', 0))}  ·  ALT H1 BLOCK {int(stats.get('h1_ema_blocked', 0))}  ·  BTC EMA BLOCK {int(stats.get('btc_ema_blocked', 0))}  ·  MAX BLOCK {int(stats.get('max_blocked', 0))}",
            f"OTHER     : DUPLICATE {int(stats.get('duplicate', 0))}  ·  ERRORS {int(stats.get('errors', 0))}",
            f"DURATION  : {elapsed:.1f}s",
        ]
        if rejection_reasons:
            reason_text = " · ".join(
                f"{name} {count}" for name, count in sorted(
                    rejection_reasons.items(), key=lambda item: (-int(item[1]), str(item[0]))
                )[:6]
            )
            lines.append(f"REASONS   : {reason_text[:220]}")
        details = stats.get("details") or []
        if details:
            lines.extend(["──────────────────────────────", "SIGNAL TRACE:"])
            for detail in details[:16]:
                lines.append(f"› {detail}")
            if len(details) > 16:
                lines.append(f"› +{len(details) - 16} signal lain; detail lengkap tersimpan di event history")
        lines.append("──────────────────────────────")
        if interrupted or cooldown > 0:
            lines.append(f"GOVERNOR  : REST HOLD · {self._short_duration(cooldown)} remaining")
            lines.append(f"RESUME    : coin #{self._scan_resume_cursor + 1:04d} setelah cooldown")
            lines.append("ORDERS    : existing GTC tetap aktif · User Data WS tetap dipantau")
        elif self._has_unprotected_real_position():
            lines.append("PAUSE     : menunggu Binance TP dan SL terkonfirmasi aktif; scanner tidak membuat entry baru")
        elif capacity_full:
            lines.append("PAUSE     : kapasitas slot maksimum tercapai; scanner lanjut otomatis saat slot tersedia")
            lines.append("NEXT      : menunggu slot tersedia; scanner akan resume otomatis")
        elif configuration_changed:
            lines.append("NEXT      : mengulang siklus dengan parameter terbaru dari coin #0001")
        else:
            if pending_limit_exists:
                lines.append("INFO      : pending LIMIT tetap memakan slot; scanner lanjut selama masih ada kapasitas")
            if batch_end < universe_total:
                next_start = batch_end + 1
                next_end = min(universe_total, next_start + SCAN_MAX_PAIRS_PER_CYCLE - 1)
                lines.append(f"NEXT      : BATCH {batch_number + 1:02d} · COIN #{next_start:04d}–#{next_end:04d}")
            else:
                lines.append("NEXT      : menunggu close candle timeframe berikutnya")
        await self.reply("\n".join(lines))

    async def _check_scan_schedule_activation(self) -> bool:
        """Activate scanner once when a selected WIB slot is reached.

        After activation, scanning remains ON continuously until /scan off.
        A selected schedule is only a delayed-start gate, not a daily on/off
        window. Schedule state is intentionally in-memory for this fresh
        engine session only.
        """
        if not self.scan_schedule_waiting or not self.scan_schedule_slots:
            return False
        now = datetime.now(TZ)
        hhmm = now.strftime("%H:%M")
        if hhmm not in self.scan_schedule_slots:
            return False
        minute_key = now.strftime("%Y-%m-%d %H:%M")
        if minute_key == self._scan_schedule_last_fired_minute:
            return False
        if self._autostop_triggered:
            return False
        if self.autostop_percent is not None and self._autostop_data_hold:
            return False

        self._scan_schedule_last_fired_minute = minute_key
        self.scan_schedule_waiting = False
        self.scan_schedule_activated_at = now.strftime("%d-%m-%Y %H:%M WIB")
        self.scan_enabled = True
        self._scan_trigger_now = True
        self._last_scan_boundary_ms = 0
        await self.ws.start()
        await self._sync_ws_symbols()
        selected = ", ".join(self.scan_schedule_slots)
        message = (
            "🟢 VOLUMATIC / SCHEDULE TRIGGERED\n\n"
            f"SLOT      : {hhmm} WIB\n"
            f"SELECTED  : {selected} WIB\n"
            f"SIGNAL TF : M15 • EMA TF: H4\n"
            f"BATCH     : {SCAN_MAX_PAIRS_PER_CYCLE} symbols\n"
            "SCANNER   : ON — continuous\n"
            "CONTROL   : gunakan /scan off untuk menghentikan."
        )
        await self.reply(message)
        log.info("[SCAN SCHEDULE] activated | slot=%s | selected=%s | tf=%s", hhmm, selected, self.timeframe)
        return True

    async def _scan_loop(self) -> None:
        while self._running:
            try:
                if not self.scan_enabled and self.scan_schedule_waiting:
                    await self._check_scan_schedule_activation()
                if not self.scan_enabled:
                    await asyncio.sleep(SCAN_POLL_SECONDS)
                    continue
                if self.autostop_percent is not None and self._autostop_data_hold:
                    await asyncio.sleep(SCAN_POLL_SECONDS)
                    continue
                cooldown = self._binance_cooldown_remaining()
                if cooldown > 0:
                    self._scan_interrupted_by_cooldown = True
                    self._was_in_binance_cooldown = True
                    self._rest_recovery_active = True
                    await asyncio.sleep(min(10.0, max(2.0, cooldown)))
                    continue
                if self._rest_recovery_active:
                    await asyncio.sleep(SCAN_POLL_SECONDS)
                    continue
                if self._scan_interrupted_by_cooldown:
                    self._scan_interrupted_by_cooldown = False
                    self._scan_trigger_now = True
                    log.info("[SCAN] REST pulih; lanjutkan scanner dari cursor tersimpan.")
                if self._has_unprotected_real_position():
                    self._scan_protection_paused = True
                    await asyncio.sleep(SCAN_POLL_SECONDS)
                    continue
                if self._scan_protection_paused:
                    self._scan_protection_paused = False
                    self._scan_trigger_now = True
                    log.info("[SCAN] Binance TP/SL active; scanner resumed.")
                active_slots = len(self._active_slot_projection())
                if active_slots >= self.max_active_trades:
                    self._scan_capacity_paused = True
                    await asyncio.sleep(SCAN_POLL_SECONDS)
                    continue
                if self._scan_capacity_paused:
                    self._scan_capacity_paused = False
                    self._scan_trigger_now = True
                interval_ms = volumatic.TIMEFRAME_MS["M15"]
                boundary = (int(time.time() * 1000) // interval_ms) * interval_ms
                due = self._scan_trigger_now or boundary > self._last_scan_boundary_ms
                if due:
                    self._scan_trigger_now = False
                    self._last_scan_boundary_ms = boundary
                    await self._run_scan_cycle(boundary)
                await asyncio.sleep(SCAN_POLL_SECONDS)
            except asyncio.CancelledError:
                break
            except BinanceRateLimitError as exc:
                self._scan_interrupted_by_cooldown = True
                self._scan_trigger_now = True
                await self._notify_binance_cooldown(exc, source="SCANNER LOOP")
                log.info("[SCAN] paused by shared Binance rate governor; cursor=%s", self._scan_resume_cursor)
                await asyncio.sleep(5.0)
            except Exception as exc:
                log.exception("[SCAN] loop gagal: %s", exc)
                await asyncio.sleep(5.0)

    async def _run_scan_cycle(self, boundary_ms: int) -> None:
        if self.autostop_percent is not None and self._autostop_data_hold:
            log.info("[SCAN] held by AUTOSTOP data guard; equity snapshot belum fresh.")
            return
        if self._scan_cycle_lock.locked():
            log.info("[SCAN] cycle sebelumnya masih aktif; trigger ditahan.")
            return
        async with self._scan_cycle_lock:
            timeframe = "M15"
            trend_length = self.trend_length
            ema_filter_length = self.ema_filter_length
            h1_enabled = self.h1_enabled
            supertrend_atr_length = self.supertrend_atr_length
            supertrend_factor = self.supertrend_factor
            rr_min = self.rr_min
            rr_max = self.rr_max
            interval_ms = volumatic.TIMEFRAME_MS["M15"]
            universe = await self._universe()
            self._scanner_symbols = set(universe) if self.scan_enabled else set()
            if self.scan_enabled:
                await self._sync_ws_symbols()
            started = time.monotonic()
            processed_total = errors_total = 0
            if (
                self._scan_resume_boundary_ms != boundary_ms
                or self._scan_resume_timeframe != timeframe
                or getattr(self, "_scan_resume_trend_length", trend_length) != trend_length
                or self._scan_resume_cursor >= len(universe)
            ):
                self._scan_resume_boundary_ms = boundary_ms
                self._scan_resume_timeframe = timeframe
                self._scan_resume_trend_length = trend_length
                self._scan_resume_cursor = 0
            if not universe:
                await self.reply("╔══ VOL / SCANNER ══╗\nUNIVERSE : 0 symbols\nACTION : tidak ada pair USDT Perpetual yang memenuhi filter.")
                return

            log.info(
                "[SCAN] MTF cycle start | signal_tf=M15 | ema_tf=H4 | length=%s | ema=%s | ST_ATR=%s | ST_FACTOR=%s | pairs=%s | batch_size=%s | cursor=%s",
                trend_length, ema_filter_label(ema_filter_length), supertrend_atr_length, supertrend_factor, len(universe),
                SCAN_MAX_PAIRS_PER_CYCLE, self._scan_resume_cursor,
            )
            # The BTC H4 reference is fetched once per scan cycle, not once per
            # altcoin. A dual /EMA pair is required because this market-regime
            # gate is mandatory even when the per-altcoin filter is single/OFF.
            btc_h4_closed = None
            btc_filter_status = "NEED /EMA FAST SLOW"
            if isinstance(ema_filter_length, tuple):
                try:
                    btc_h4_closed = await self.rest.get_klines(
                        BTC_EMA_FILTER_SYMBOL,
                        "4h",
                        self._h4_ema_history_limit(ema_filter_length),
                        volumatic.TIMEFRAME_MS["H4"],
                        force_refresh=False,
                    )
                    if btc_h4_closed is None or len(btc_h4_closed) < 3:
                        raise RuntimeError("Riwayat H4 BTC tidak cukup.")
                    required_bars = max(ema_filter_length)
                    if len(btc_h4_closed) < required_bars:
                        btc_filter_status = f"WARMUP {len(btc_h4_closed)}/{required_bars} H4"
                    else:
                        btc_filter_status = (
                            f"EMA{ema_filter_length[0]}/EMA{ema_filter_length[1]} DATA READY"
                        )
                    log.info(
                        "[BTC EMA] H4 reference loaded once | symbol=%s | periods=%s/%s | candles=%s | status=%s",
                        BTC_EMA_FILTER_SYMBOL, ema_filter_length[0], ema_filter_length[1],
                        len(btc_h4_closed), btc_filter_status,
                    )
                except BinanceRateLimitError as exc:
                    self._scan_interrupted_by_cooldown = True
                    self._scan_trigger_now = True
                    await self._notify_binance_cooldown(
                        exc, source="BTC H4 EMA REFERENCE", symbol=BTC_EMA_FILTER_SYMBOL
                    )
                    await self.reply(
                        "⚠️ VOL / SCANNER PAUSED BEFORE BATCH\n"
                        f"CAUSE     : BTC H4 reference tertahan REST cooldown ({self._short_duration(self._binance_cooldown_remaining())})\n"
                        f"CURSOR    : coin #{self._scan_resume_cursor + 1:04d} belum diproses pada siklus ini\n"
                        "DATA      : metadata universe tetap disimpan; tidak ada candle lama yang akan dipakai untuk entry\n"
                        "RESUME    : otomatis setelah cooldown; scanner memakai candle M15 terbaru yang masih valid."
                    )
                    return
                except Exception as exc:
                    btc_h4_closed = None
                    btc_filter_status = f"DATA UNAVAILABLE: {str(exc)[:90]}"
                    log.exception("[BTC EMA] mandatory BTC H4 filter data unavailable; all new signals will fail closed.")
            else:
                log.warning(
                    "[BTC EMA] new entries are blocked until /EMA FAST SLOW is configured; current=%s",
                    ema_filter_label(ema_filter_length),
                )

            await self._expire_due_pending()
            offset = self._scan_resume_cursor
            capacity_reached = False
            protection_paused = False
            while offset < len(universe):
                if not self._running or not self.scan_enabled:
                    break
                if (
                    self.trend_length != trend_length
                    or self.ema_filter_length != ema_filter_length
                    or self.h1_enabled != h1_enabled
                    or self.rr_min != rr_min
                    or self.rr_max != rr_max
                ):
                    self._scan_resume_cursor = 0
                    self._scan_resume_boundary_ms = 0
                    break
                if len(self._active_slot_projection()) >= self.max_active_trades:
                    self._scan_capacity_paused = True
                    self._scan_resume_cursor = offset
                    capacity_reached = True
                    break

                logical_start = (offset // SCAN_MAX_PAIRS_PER_CYCLE) * SCAN_MAX_PAIRS_PER_CYCLE
                logical_end = min(logical_start + SCAN_MAX_PAIRS_PER_CYCLE, len(universe))
                batch = universe[offset:logical_end]
                batch_number = logical_start // SCAN_MAX_PAIRS_PER_CYCLE + 1
                batch_start, batch_end = logical_start + 1, logical_end
                pass_start = offset + 1
                batch_started = time.monotonic()
                stats: dict[str, Any] = {
                    "requested": 0, "processed": 0, "analyzed": 0,
                    "already_seen": 0, "warmup": 0, "stale": 0,
                    "no_signal": 0, "rejected": 0, "rejection_reasons": {},
                    "blue": 0, "red": 0,
                    "orders": 0, "filled": 0, "deferred": 0, "unknown": 0,
                    "ema_blocked": 0, "h1_ema_blocked": 0, "btc_ema_blocked": 0, "max_blocked": 0, "duplicate": 0,
                    "btc_filter_status": btc_filter_status,
                    "errors": 0, "details": [],
                }
                interrupted = False
                configuration_changed = False
                log.info(
                    "[SCAN] MTF batch start | batch=%02d | coins=%04d-%04d/%04d | pass_from=%04d | M15 LEN=%s | H4 EMA=%s",
                    batch_number, batch_start, batch_end, len(universe), pass_start,
                    trend_length, ema_filter_label(ema_filter_length),
                )
                for local_index, symbol in enumerate(batch):
                    absolute_index = offset + local_index
                    if (
                        self.trend_length != trend_length
                        or self.ema_filter_length != ema_filter_length
                        or self.h1_enabled != h1_enabled
                        or self.supertrend_atr_length != supertrend_atr_length
                        or self.supertrend_factor != supertrend_factor
                        or self.rr_min != rr_min
                        or self.rr_max != rr_max
                    ):
                        configuration_changed = True
                        self._scan_resume_cursor = 0
                        self._scan_resume_boundary_ms = 0
                        break
                    if self._binance_cooldown_remaining() > 0:
                        interrupted = True
                        self._scan_interrupted_by_cooldown = True
                        self._scan_trigger_now = True
                        self._scan_resume_cursor = absolute_index
                        break
                    if len(self._active_slot_projection()) >= self.max_active_trades:
                        self._scan_capacity_paused = True
                        self._scan_resume_cursor = absolute_index
                        capacity_reached = True
                        break
                    stats["requested"] += 1
                    try:
                        m15 = await self.rest.get_klines_with_forming(
                            symbol, "15m", self._kline_history_limit(trend_length, supertrend_atr_length), interval_ms
                        )
                        h4 = await self.rest.get_klines(
                            symbol, "4h", self._h4_ema_history_limit(ema_filter_length),
                            volumatic.TIMEFRAME_MS["H4"],
                        )
                        if (
                            self.trend_length != trend_length
                            or self.ema_filter_length != ema_filter_length
                            or self.h1_enabled != h1_enabled
                            or self.supertrend_atr_length != supertrend_atr_length
                            or self.supertrend_factor != supertrend_factor
                            or self.rr_min != rr_min
                            or self.rr_max != rr_max
                        ):
                            configuration_changed = True
                            self._scan_resume_cursor = 0
                            self._scan_resume_boundary_ms = 0
                            break
                        min_m15 = max(12, trend_length + 8, supertrend_atr_length + 1)
                        min_h4 = max(3, (max(ema_filter_length) if isinstance(ema_filter_length, tuple) else (ema_filter_length or 1)))
                        if m15 is None or len(m15) < min_m15 or h4 is None or len(h4) < min_h4:
                            stats["warmup"] += 1
                            self._scan_resume_cursor = absolute_index + 1
                            continue
                        entry_open_ms = int(m15["t"][-1])
                        expiry_ms = entry_open_ms + interval_ms
                        key = f"M15:{symbol}"
                        if entry_open_ms <= int(self.last_processed_candles.get(key, 0)):
                            stats["already_seen"] += 1
                            self._scan_resume_cursor = absolute_index + 1
                            continue
                        now_ms = int(time.time() * 1000)
                        if not (entry_open_ms <= now_ms < expiry_ms) or int(m15["t"][-2]) + interval_ms != entry_open_ms:
                            self.last_processed_candles[key] = entry_open_ms
                            stats["stale"] += 1
                            self._scan_resume_cursor = absolute_index + 1
                            continue
                        signal_diagnostics: dict[str, Any] = {}
                        signal = volumatic.detect_backtest_signal(
                            m15, h4, symbol=symbol, trend_length=trend_length,
                            ema_filter_length=ema_filter_length,
                            btc_h4_closed=btc_h4_closed,
                            supertrend_atr_length=supertrend_atr_length,
                            supertrend_factor=supertrend_factor,
                            rr_min=float(rr_min),
                            rr_max=float(rr_max),
                            diagnostics=signal_diagnostics,
                        )
                        stats["analyzed"] += 1
                        if signal is not None:
                            signal["h1_enabled_at_scan"] = h1_enabled
                        if signal is None:
                            rejection_reason = str(signal_diagnostics.get("reason") or "UNCLASSIFIED_REJECTION")
                            if rejection_reason == "NO_NEW_BUBBLE":
                                stats["no_signal"] += 1
                            else:
                                stats["rejected"] += 1
                                reasons = stats["rejection_reasons"]
                                reasons[rejection_reason] = int(reasons.get(rejection_reason, 0)) + 1
                            self.last_processed_candles[key] = entry_open_ms
                            self._scan_resume_cursor = absolute_index + 1
                            continue
                        # H1 EMA entry gate: fetched only for a real signal that already
                        # passed the H4 gate, to avoid extra REST weight on every coin.
                        if h1_enabled and ema_filter_length is not None and (
                            signal.get("ema_filter_passed") is True
                        ):
                            try:
                                h1 = await self.rest.get_klines(
                                    symbol, "1h", self._h4_ema_history_limit(ema_filter_length),
                                    H1_MS, force_refresh=False,
                                )
                            except BinanceRateLimitError:
                                raise
                            except Exception as h1_exc:
                                h1 = None
                                log.warning("[H1 EMA] %s data H1 gagal; entry diblokir fail-closed: %s", symbol, h1_exc)
                            signal.update(volumatic.h1_ema_filter_state(
                                h1, ema_filter_length, str(signal.get("direction") or ""),
                                int(signal.get("signal_close_time_ms") or 0),
                            ))
                        bubble = str(signal.get("signal") or "?").upper()
                        direction = str(signal.get("direction") or "?").upper()
                        stats["blue" if bubble == "BLUE" else "red"] += 1
                        try:
                            outcome = await self._dispatch_mode_b_signal(signal, expiry_ms, source="SCANNER_M15_CONFIRMED")
                        except Exception as signal_exc:
                            self._signal_retries[key] = {
                                "signal": signal, "expiry_ms": expiry_ms, "key": key,
                                "attempts": 0, "last_error": str(signal_exc)[:500],
                                "source": "SCANNER_M15_CONFIRMED",
                            }
                            stats["errors"] += 1
                            errors_total += 1
                            outcome = "ERROR_RETRY"
                            log.exception("[SIGNAL] %s %s failed: %s", symbol, bubble, signal_exc)
                        if outcome == "ORDER_CREATED": stats["orders"] += 1
                        elif outcome == "ORDER_FILLED": stats["filled"] += 1
                        elif outcome == "ORDER_DEFERRED": stats["deferred"] += 1
                        elif outcome == "ORDER_UNKNOWN": stats["unknown"] += 1
                        elif outcome == "EMA_BLOCKED": stats["ema_blocked"] += 1
                        elif outcome == "H1_EMA_BLOCKED": stats["h1_ema_blocked"] += 1
                        elif outcome == "BTC_EMA_BLOCKED": stats["btc_ema_blocked"] += 1
                        elif outcome == "MAX_BLOCKED": stats["max_blocked"] += 1
                        elif outcome in {"DUPLICATE", "IGNORED_TIMEFRAME", "EXCHANGE_SLOT_EXISTS"}: stats["duplicate"] += 1
                        stats["details"].append(
                            f"#{absolute_index+1:04d} {symbol} {bubble}/{direction} M15LEN{trend_length} "
                            f"H4EMA={ema_filter_values_label(ema_filter_length, signal.get('ema_filter_value'), signal.get('ema_filter_slow_value'))} "
                            f"H1EMA={signal.get('h1_ema_filter_value')}/{signal.get('h1_ema_filter_slow_value')}({signal.get('h1_ema_filter_reason', '-')}) "
                            f"BTCEMA={signal.get('btc_ema_fast_value')}/{signal.get('btc_ema_slow_value')} "
                            f"BTC={signal.get('btc_ema_filter_reason', 'UNKNOWN')} "
                            f"ENTRY={signal.get('entry_price')} SL={signal.get('stop_loss')} "
                            f"TP={signal.get('take_profit')} RR={signal.get('take_profit_rr')} "
                            f"SRC={signal.get('take_profit_source')} {outcome}"[:230]
                        )
                        self.last_processed_candles[key] = entry_open_ms
                        if outcome != "ERROR_RETRY":
                            self._signal_retries.pop(key, None)
                        stats["processed"] += 1
                        processed_total += 1
                        self._scan_resume_cursor = absolute_index + 1
                        # A newly-created LIMIT reserves one slot but must not
                        # stop this scan pass. The next symbol re-checks live
                        # slot usage before it can create another order.
                        if self._has_unprotected_real_position():
                            self._scan_protection_paused = True
                            protection_paused = True
                            break
                    except BinanceRateLimitError as exc:
                        stats["errors"] += 1
                        errors_total += 1
                        interrupted = True
                        self._scan_interrupted_by_cooldown = True
                        self._scan_trigger_now = True
                        self._scan_resume_cursor = absolute_index
                        await self._notify_binance_cooldown(exc, source="SCANNER M15/H4 OR ORDER", symbol=symbol)
                        break
                    except Exception as exc:
                        stats["errors"] += 1
                        errors_total += 1
                        self._scan_resume_cursor = absolute_index + 1
                        log.exception("[SCAN] %s M15/H4 failed: %s", symbol, exc)
                    if SCAN_PAIR_DELAY_SECONDS > 0:
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)

                release_memory()
                await self._send_scan_batch_report(
                    timeframe="M15 (EMA H4)", trend_length=trend_length,
                    ema_length=ema_filter_length, supertrend_atr_length=supertrend_atr_length,
                    supertrend_factor=supertrend_factor, rr_min=rr_min, rr_max=rr_max,
                    batch_number=batch_number, batch_start=batch_start, batch_end=batch_end,
                    universe_total=len(universe), stats=stats,
                    elapsed=time.monotonic()-batch_started, interrupted=interrupted,
                    configuration_changed=configuration_changed, pass_start=pass_start,
                )
                await self._save_runtime_settings()
                await self._save_active_state()
                if configuration_changed:
                    self._scan_trigger_now = bool(self.scan_enabled and not self._autostop_triggered)
                    break
                if interrupted or self._binance_cooldown_remaining() > 0 or self._scan_interrupted_by_cooldown:
                    self._scan_trigger_now = True
                    break
                if protection_paused:
                    log.warning("[SCAN] paused: Binance TP/SL are not both confirmed active yet.")
                    break
                if capacity_reached:
                    break
                if self._scan_resume_cursor >= logical_end:
                    offset = logical_end
                else:
                    break

            completed = self._scan_resume_cursor >= len(universe)
            if completed:
                self._scan_resume_cursor = 0
                self._scan_resume_boundary_ms = 0
                self._scan_resume_timeframe = "M15"
            await self._save_runtime_settings()
            await self._save_active_state()
            await self._sync_ws_symbols()
            if self._scan_interrupted_by_cooldown or self._binance_cooldown_remaining() > 0:
                self._scan_trigger_now = True
            elapsed = time.monotonic() - started
            log.info(
                "[SCAN] cycle %s | signal_tf=M15 ema_tf=H4 btc_filter=%s | processed=%s/%s | errors=%s | cursor=%s | "
                "capacity_pause=%s pending_reserved=%s protection_pause=%s | duration=%.1fs",
                "complete" if completed else "paused", btc_filter_status, processed_total, len(universe), errors_total,
                self._scan_resume_cursor, capacity_reached, self._has_pending_entry_limit(),
                self._has_unprotected_real_position(), elapsed,
            )

    async def _dispatch_mode_b_signal(
        self, signal: dict[str, Any], expiry_ms: int, *, source: str
    ) -> str:
        """Dispatch one closed M15 bubble; name kept for launcher/runtime compatibility."""
        pair = normalize_symbol(signal.get("symbol") or "")
        timeframe = "M15"
        direction = str(signal.get("direction") or "").upper()
        signal_close = int(signal.get("signal_close_time_ms") or 0)
        entry_open = int(signal.get("entry_candle_open_time_ms") or 0)
        token = f"{entry_open}:{str(signal.get('signal') or direction).upper()}"
        key = f"M15:{pair}"
        seen = self._live_signal_seen.setdefault(key, set())
        if token in seen:
            return "DUPLICATE"
        seen.add(token)
        try:
            outcome = await self._handle_signal(signal, int(expiry_ms))
        except BinanceRateLimitError as exc:
            self._signal_retries[key] = {
                "signal": dict(signal), "expiry_ms": int(expiry_ms), "key": key,
                "attempts": 0, "last_error": str(exc)[:500], "source": str(source),
            }
            self._was_in_binance_cooldown = True
            self._scan_interrupted_by_cooldown = True
            await self._notify_binance_cooldown(exc, source=f"{source} / M15 ORDER", symbol=pair)
            await self._record_event("M15_SIGNAL_DEFERRED_RATE_LIMIT", None,
                                     pair=pair, timeframe=timeframe, direction=direction,
                                     signal=signal.get("signal"), signal_close_ms=signal_close,
                                     entry_open_ms=entry_open, expiry_ms=int(expiry_ms), source=source)
            return "ORDER_DEFERRED"
        await self._record_event("M15_CONFIRMED_SIGNAL_DISPATCHED", None,
                                 pair=pair, timeframe=timeframe, direction=direction,
                                 signal=signal.get("signal"), signal_close_ms=signal_close,
                                 entry_open_ms=entry_open, entry_price=signal.get("entry_price"),
                                 sl=signal.get("stop_loss"), tp=signal.get("take_profit"),
                                 source=source, outcome=outcome)
        return outcome

    def _active_slot_projection(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Union local trades, Binance open LIMIT orders, and Binance positions.

        A pair+direction reserves one /max slot. A partially-filled LIMIT plus
        its resulting position on the same side is still one slot; an opposite
        side exposure/order is a separate slot. This method reads RAM caches
        only and never calls Binance REST.
        """
        slots: dict[tuple[str, str], dict[str, Any]] = {}

        def ensure(symbol: str, direction: str) -> dict[str, Any] | None:
            pair = normalize_symbol(symbol)
            side = str(direction or "").upper()
            if not pair or side not in {"LONG", "SHORT"}:
                return None
            return slots.setdefault((pair, side), {"pending": False, "filled": False, "orders": [], "trades": []})

        for trade in self.active_trades.values():
            if trade.status not in {"PENDING", "FILLED"}:
                continue
            item = ensure(trade.pair, trade.direction)
            if item is None:
                continue
            item["pending"] = item["pending"] or trade.status == "PENDING" or (
                trade.real_enabled and not trade.exchange_adopted and not trade.entry_order_terminal
            )
            item["filled"] = item["filled"] or trade.status == "FILLED"
            item["trades"].append(trade)

        for order in self.exchange_open_orders.values():
            if not isinstance(order, dict):
                continue
            status = str(order.get("status") or order.get("X") or "NEW").upper()
            if status not in {"NEW", "PARTIALLY_FILLED"}:
                continue
            side = str(order.get("side") or order.get("S") or "").upper()
            direction = "LONG" if side == "BUY" else "SHORT" if side == "SELL" else None
            if direction is None:
                continue
            symbol = str(order.get("symbol") or order.get("s") or "")
            item = ensure(symbol, direction)
            if item is not None:
                item["pending"] = True
                item["orders"].append(order)

        for (symbol, _position_side), position in self.exchange_positions.items():
            if not isinstance(position, dict):
                continue
            raw_amount = position.get("_position_amount_decimal", position.get("positionAmt", "0"))
            try:
                amount = parse_signed_decimal(str(raw_amount or "0"))
            except Exception:
                amount = Decimal("0")
            if amount != 0:
                item = ensure(symbol, "LONG" if amount > 0 else "SHORT")
                if item is not None:
                    item["filled"] = True

        return slots

    def _has_pending_entry_limit(self) -> bool:
        """True while any local/WS projected entry LIMIT still has open quantity."""
        return any(bool(slot.get("pending")) for slot in self._active_slot_projection().values())

    def _scanner_state_label(self) -> str:
        """Return the effective scanner state; pending LIMITs reserve slots but never pause scanning."""
        if self._autostop_triggered:
            return "LOCKED AUTOSTOP"
        if not self.scan_enabled:
            return "WAITING SCHEDULE" if self.scan_schedule_waiting else "OFF"
        if self._binance_cooldown_remaining() > 0:
            return "PAUSED — REST COOLDOWN"
        if self.autostop_percent is not None and self._autostop_data_hold:
            return "PAUSED — WAITING EQUITY DATA"
        if self._has_unprotected_real_position():
            return "PAUSED — TP/SL BINANCE BELUM AKTIF"
        used_slots = len(self._active_slot_projection())
        if used_slots >= self.max_active_trades:
            return "PAUSED — MAX SLOT TERCAPAI"
        return "ON"

    def _has_unprotected_real_position(self) -> bool:
        """Hold new scans while a bot-managed LIVE position lacks confirmed TP+SL."""
        return any(
            trade.real_enabled and not trade.exchange_adopted and trade.status == "FILLED"
            and trade.protection_state != "ACTIVE"
            for trade in self.active_trades.values()
        )

    def _active_slot_counts(self) -> tuple[int, int, int, int]:
        """Return (slots, pending-only slots, filled slots, live LIMIT orders)."""
        slots = self._active_slot_projection()
        pending_only = sum(1 for item in slots.values() if item["pending"] and not item["filled"])
        filled = sum(1 for item in slots.values() if item["filled"])
        live_limits = sum(len(item["orders"]) for item in slots.values())
        return len(slots), pending_only, filled, live_limits

    @staticmethod
    def _exchange_order_fields(order: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize Binance REST snapshot or ORDER_TRADE_UPDATE order fields."""
        if not isinstance(order, dict):
            return None
        symbol = normalize_symbol(str(order.get("symbol") or order.get("s") or ""))
        side = str(order.get("side") or order.get("S") or "").upper()
        direction = "LONG" if side == "BUY" else "SHORT" if side == "SELL" else None
        if not symbol or direction is None:
            return None
        try:
            price = Decimal(str(order.get("price") or order.get("p") or "0"))
            original = Decimal(str(order.get("origQty") or order.get("q") or "0"))
            executed = Decimal(str(order.get("executedQty") or order.get("z") or "0"))
        except (InvalidOperation, ValueError, TypeError):
            return None
        order_id = order.get("orderId", order.get("i"))
        client_id = str(order.get("clientOrderId") or order.get("origClientOrderId") or order.get("c") or "")
        status = str(order.get("status") or order.get("X") or "NEW").upper()
        return {
            "symbol": symbol, "direction": direction, "side": side,
            "price": price, "original_qty": original, "executed_qty": executed,
            "remaining_qty": max(Decimal("0"), original - executed),
            "order_id": order_id, "client_id": client_id, "status": status,
        }

    def _remove_exchange_order_cache(self, trade: Trade) -> None:
        """Remove a terminal entry order from the realtime open-order projection."""
        for key, order in list(self.exchange_open_orders.items()):
            if not isinstance(order, dict):
                continue
            order_id = order.get("orderId", order.get("i"))
            client_id = str(order.get("clientOrderId") or order.get("c") or "")
            same_id = (
                (trade.entry_client_order_id and client_id == trade.entry_client_order_id)
                or (trade.entry_order_id is not None and str(order_id) == str(trade.entry_order_id))
            )
            if same_id:
                self.exchange_open_orders.pop(key, None)

    async def _ensure_signal_dedupe_loaded(self) -> None:
        """Import durable consumed M15 signal keys into the local SQLite uniqueness ledger.

        The SQLite UNIQUE primary key is the cross-process mutex: two engine
        workers sharing the Render filesystem cannot both claim one signal,
        even if Binance has already filled the first order and would allow the
        same newClientOrderId to be reused.
        """
        if self._signal_dedupe_loaded:
            return
        async with self._signal_dedupe_lock:
            if self._signal_dedupe_loaded:
                return
            self._signal_dedupe_db_path.parent.mkdir(parents=True, exist_ok=True)
            raw = await self._remote_or_local(SIGNAL_DEDUPE_PATH)
            imported: list[str] = []
            if raw:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    values = payload.get("keys", []) if isinstance(payload, dict) else []
                    imported = [str(x) for x in values if isinstance(x, str) and x]
                except Exception:
                    log.warning("[SCAN/DEDUP] persisted signal ledger invalid; SQLite/local ledger retained.")
            db = sqlite3.connect(str(self._signal_dedupe_db_path), timeout=30)
            try:
                db.execute("PRAGMA busy_timeout=30000")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS consumed_signals ("
                    "signal_key TEXT PRIMARY KEY, claimed_at TEXT NOT NULL)"
                )
                db.execute("BEGIN IMMEDIATE")
                db.executemany(
                    "INSERT OR IGNORE INTO consumed_signals(signal_key, claimed_at) VALUES (?, ?)",
                    [(key, iso_utc()) for key in imported],
                )
                db.commit()
            finally:
                db.close()
            self._signal_dedupe_loaded = True
            log.info("[SCAN/DEDUP] durable signal ledger loaded | imported=%s", len(imported))

    async def _claim_m15_signal(self, signal_key: str) -> bool:
        """Atomically reserve a pair/candle signal before any Binance order POST."""
        await self._ensure_signal_dedupe_loaded()
        async with self._signal_dedupe_lock:
            db = sqlite3.connect(str(self._signal_dedupe_db_path), timeout=30)
            try:
                db.execute("PRAGMA busy_timeout=30000")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS consumed_signals ("
                    "signal_key TEXT PRIMARY KEY, claimed_at TEXT NOT NULL)"
                )
                db.execute("BEGIN IMMEDIATE")
                cur = db.execute(
                    "INSERT OR IGNORE INTO consumed_signals(signal_key, claimed_at) VALUES (?, ?)",
                    (signal_key, iso_utc()),
                )
                if cur.rowcount != 1:
                    db.rollback()
                    return False
                rows = db.execute(
                    "SELECT signal_key FROM consumed_signals ORDER BY claimed_at DESC LIMIT 10000"
                ).fetchall()
                keys = [row[0] for row in rows]
                payload = {
                    "updated_at": iso_utc(),
                    "keys": keys,
                    "count": len(keys),
                    "policy": "one_entry_per_symbol_per_m15_signal_candle",
                }
                # Persist while holding the SQLite write transaction. Another
                # process cannot claim/overwrite a newer key until this snapshot
                # is safely written to the local fallback and GitHub attempt ends.
                await self._persist_bytes(
                    SIGNAL_DEDUPE_PATH,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                    "volumatic: reserve unique M15 signal candle",
                )
                db.commit()
                return True
            except BaseException:
                try:
                    db.rollback()
                except Exception:
                    pass
                raise
            finally:
                db.close()

    async def _handle_signal(self, signal: dict[str, Any], expiry_ms: int) -> str:
        pair = normalize_symbol(signal["symbol"])
        direction = str(signal["direction"]).upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError(f"Arah sinyal tidak valid: {direction}")
        if str(signal.get("signal_mode") or "").upper() != "CONFIRMED_M15_NEXT_OPEN":
            raise ValueError("Signal ditolak: hanya sinyal confirmed M15 next-open yang diterima.")
        async with self._trade_lock:
            signal_close_ms = int(signal.get("signal_close_time_ms") or 0)
            entry_open_ms = int(signal.get("entry_candle_open_time_ms") or 0)
            entry_close_ms = int(signal.get("entry_candle_close_time_ms") or 0)
            if signal_close_ms != entry_open_ms or entry_close_ms != entry_open_ms + 900_000 or int(expiry_ms) != entry_close_ms:
                raise ValueError("Urutan candle tidak konsisten: sinyal M15 harus masuk pada open M15 berikutnya.")
            now_ms = int(time.time() * 1000)
            if not (entry_open_ms <= now_ms < entry_close_ms):
                await self._record_event("M15_SIGNAL_EXPIRED_BEFORE_ORDER", None, pair=pair, direction=direction)
                return "EXPIRED"

            # Backtest ignores any further signal while a trade/order on the pair is active.
            existing = next((t for t in self.active_trades.values()
                             if t.pair == pair and t.status in {"PENDING", "FILLED"}), None)
            if existing is not None:
                await self._record_event("SIGNAL_IGNORED_ACTIVE_TRADE", existing,
                                         signal_direction=direction, signal=signal.get("signal"),
                                         reason="Backtest model ignores signals until the active trade reaches SL/TP.")
                return "DUPLICATE"
            slot_projection = self._active_slot_projection()
            pair_slots = {k:v for k,v in slot_projection.items() if k[0] == pair}
            if pair_slots:
                await self._record_event("SIGNAL_BLOCKED_EXCHANGE_SLOT", None, pair=pair, direction=direction)
                return "EXCHANGE_SLOT_EXISTS"
            if len(slot_projection) >= self.max_active_trades:
                self._scan_capacity_paused = True
                await self._record_event("SIGNAL_BLOCKED_MAX_ACTIVE", None, pair=pair, direction=direction,
                                         active_slots=len(slot_projection), max_active_trades=self.max_active_trades)
                return "MAX_BLOCKED"

            filter_length = volumatic.normalize_ema_filter_length(signal.get("ema_filter_length", self.ema_filter_length))
            def positive_decimal(raw: Any) -> Decimal | None:
                if raw in (None, ""):
                    return None
                try:
                    value = Decimal(str(raw))
                    return value if value.is_finite() and value > 0 else None
                except (InvalidOperation, ValueError, TypeError):
                    return None
            filter_value = positive_decimal(signal.get("ema_filter_value"))
            filter_slow_value = positive_decimal(signal.get("ema_filter_slow_value"))
            filter_passed = bool(signal.get("ema_filter_passed", False))
            filter_ready = bool(signal.get("ema_filter_ready", filter_length is None))
            needs_slow = isinstance(filter_length, tuple)
            if filter_length is not None and (not filter_ready or filter_value is None or (needs_slow and filter_slow_value is None) or not filter_passed):
                reason = "H4_EMA_NOT_READY" if not filter_ready or filter_value is None or (needs_slow and filter_slow_value is None) else "H4_EMA_DIRECTION_NOT_ALIGNED"
                await self._record_event("SIGNAL_REJECTED_BY_H4_EMA", None, pair=pair, direction=direction,
                                         ema_filter_length=list(filter_length) if isinstance(filter_length, tuple) else filter_length,
                                         ema_filter_value=decimal_to_str(filter_value),
                                         ema_filter_slow_value=decimal_to_str(filter_slow_value), reason=reason)
                await self.reply(
                    f"⛔ SINYAL {direction} DIBLOKIR H4 EMA • {pair}\n"
                    f"Filter: {ema_filter_values_label(filter_length, filter_value, filter_slow_value)}\n"
                    f"Alasan: {reason}. Tidak ada order yang dibuat."
                )
                return "EMA_BLOCKED"

            # H1 EMA entry gate: same /EMA periods and direction rule as H4.
            h1_fast_value = positive_decimal(signal.get("h1_ema_filter_value"))
            h1_slow_value = positive_decimal(signal.get("h1_ema_filter_slow_value"))
            h1_filter_enabled_for_signal = bool(signal.get("h1_enabled_at_scan", self.h1_enabled)) and self.h1_enabled
            if h1_filter_enabled_for_signal and filter_length is not None:
                if not signal.get("h1_ema_checked"):
                    h1_reason = "H1_EMA_NOT_EVALUATED"
                elif not signal.get("h1_ema_filter_ready") or h1_fast_value is None or (needs_slow and h1_slow_value is None):
                    h1_reason = str(signal.get("h1_ema_filter_reason") or "H1_EMA_NOT_READY")
                elif not signal.get("h1_ema_filter_passed"):
                    h1_reason = str(signal.get("h1_ema_filter_reason") or "H1_EMA_DIRECTION_NOT_ALIGNED")
                else:
                    h1_reason = "OK"
                if h1_reason != "OK":
                    await self._record_event(
                        "SIGNAL_REJECTED_BY_H1_EMA", None, pair=pair, direction=direction,
                        ema_filter_length=list(filter_length) if isinstance(filter_length, tuple) else filter_length,
                        h1_ema_filter_value=decimal_to_str(h1_fast_value),
                        h1_ema_filter_slow_value=decimal_to_str(h1_slow_value),
                        h1_ema_open_time_ms=signal.get("h1_ema_open_time_ms"),
                        h1_ema_reference_close=signal.get("h1_ema_reference_close"),
                        reason=h1_reason,
                    )
                    await self.reply(
                        f"⛔ SINYAL {direction} DIBLOKIR H1 EMA • {pair}\n"
                        f"Filter: {ema_filter_values_label(filter_length, h1_fast_value, h1_slow_value)}\n"
                        f"Alasan: {h1_reason}. Tidak ada order yang dibuat."
                    )
                    return "H1_EMA_BLOCKED"

            btc_filter_length_raw = signal.get("btc_ema_filter_length")
            try:
                btc_filter_length = volumatic.normalize_ema_filter_length(btc_filter_length_raw)
            except (TypeError, ValueError):
                btc_filter_length = None
            btc_fast_value = positive_decimal(signal.get("btc_ema_fast_value"))
            btc_slow_value = positive_decimal(signal.get("btc_ema_slow_value"))
            btc_filter_ready = bool(signal.get("btc_ema_filter_ready", False))
            btc_filter_passed = bool(signal.get("btc_ema_filter_passed", False))
            if not isinstance(filter_length, tuple):
                btc_reason = "BTC_EMA_REQUIRES_DUAL_EMA_CONFIG"
            elif btc_filter_length != filter_length:
                btc_reason = "BTC_EMA_CONFIG_MISMATCH"
            elif not btc_filter_ready or btc_fast_value is None or btc_slow_value is None:
                btc_reason = str(signal.get("btc_ema_filter_reason") or "BTC_H4_EMA_NOT_READY")
            elif not btc_filter_passed:
                btc_reason = str(signal.get("btc_ema_filter_reason") or "BTC_H4_EMA_DIRECTION_NOT_ALIGNED")
            else:
                btc_reason = "OK"
            if btc_reason != "OK":
                await self._record_event(
                    "SIGNAL_REJECTED_BY_BTC_H4_EMA", None,
                    pair=pair, direction=direction,
                    btc_ema_filter_length=(list(btc_filter_length) if isinstance(btc_filter_length, tuple) else None),
                    btc_ema_fast_value=decimal_to_str(btc_fast_value),
                    btc_ema_slow_value=decimal_to_str(btc_slow_value),
                    btc_ema_filter_ready=btc_filter_ready,
                    btc_ema_filter_passed=btc_filter_passed,
                    btc_ema_h4_open_time_ms=signal.get("btc_ema_h4_open_time_ms"),
                    reason=btc_reason,
                )
                btc_label = (
                    f"EMA{filter_length[0]}/EMA{filter_length[1]}"
                    if isinstance(filter_length, tuple) else "DUAL EMA NOT CONFIGURED"
                )
                await self.reply(
                    f"⛔ SINYAL {direction} DIBLOKIR BTC H4 EMA • {pair}\n"
                    f"Filter BTCUSDT.P: {btc_label}\n"
                    f"Fast: {fmt_price(btc_fast_value)} · Slow: {fmt_price(btc_slow_value)}\n"
                    f"Alasan: {btc_reason}. Tidak ada order yang dibuat."
                )
                return "BTC_EMA_BLOCKED"

            meta = self.symbols.get(pair)
            if meta is None:
                raise ValueError(f"Metadata Binance untuk {pair} tidak ditemukan.")
            entry_raw = positive_decimal(signal.get("entry_price") or signal.get("limit_price"))
            stop_raw = positive_decimal(signal.get("stop_loss") or signal.get("reference_sl"))
            if entry_raw is None or stop_raw is None:
                raise ValueError("Harga next M15 open atau Supertrend SL tidak valid.")
            entry = quantized_price(entry_raw, meta.tick_size) if direction == "LONG" else quantized_price_ceiling(entry_raw, meta.tick_size)
            stop = quantized_price(stop_raw, meta.tick_size) if direction == "LONG" else quantized_price_ceiling(stop_raw, meta.tick_size)
            risk = (entry - stop) if direction == "LONG" else (stop - entry)
            if risk <= 0:
                await self._record_event("SIGNAL_REJECTED_INVALID_SUPERTREND_SL", None, pair=pair,
                                         direction=direction, entry=decimal_to_str(entry), stop=decimal_to_str(stop))
                return "INVALID_SL"
            # RR settings are snapshotted in the signal/trade. Later command
            # changes apply only to future signals and never rewrite active targets.
            try:
                signal_rr_min = Decimal(str(signal.get("rr_min", self.rr_min)))
                signal_rr_max = Decimal(str(signal.get("rr_max", self.rr_max)))
                normalized_rr_min, normalized_rr_max = volumatic.normalize_rr_range(
                    signal_rr_min, signal_rr_max
                )
                signal_rr_min = Decimal(str(normalized_rr_min))
                signal_rr_max = Decimal(str(normalized_rr_max))
            except (InvalidOperation, TypeError, ValueError) as exc:
                await self._record_event(
                    "SIGNAL_REJECTED_INVALID_RR_CONFIG", None, pair=pair, direction=direction,
                    rr_min=str(signal.get("rr_min")), rr_max=str(signal.get("rr_max")), reason=str(exc),
                )
                return "RR_CONFIG_INVALID"
            base_target_raw = entry + (
                signal_rr_min * risk if direction == "LONG" else -signal_rr_min * risk
            )
            base_target = (
                quantized_price_ceiling(base_target_raw, meta.tick_size)
                if direction == "LONG" else quantized_price(base_target_raw, meta.tick_size)
            )
            target_source = str(signal.get("take_profit_source") or "RR_MIN_BASE").upper()
            liquidity_pool_raw = positive_decimal(
                signal.get("liquidity_pool_price") or signal.get("take_profit")
            )
            target = base_target
            liquidity_pool_price = None
            if target_source == "VOLUME_LIQUIDITY" and liquidity_pool_raw is not None:
                # Round a pool toward entry so the target never overshoots
                # the structural level or crosses this trade's configured RR_MAX.
                candidate_target = (
                    quantized_price(liquidity_pool_raw, meta.tick_size)
                    if direction == "LONG" else quantized_price_ceiling(liquidity_pool_raw, meta.tick_size)
                )
                candidate_rr = directional_reward_r(direction, entry, candidate_target, risk)
                if (
                    is_valid_liquidity_reward_r(candidate_rr, signal_rr_min, signal_rr_max)
                    and (candidate_target > entry if direction == "LONG" else candidate_target < entry)
                ):
                    target = candidate_target
                    liquidity_pool_price = candidate_target
                else:
                    target_source = "RR_MIN_BASE"
            else:
                target_source = "RR_MIN_BASE"
            target_rr = directional_reward_r(direction, entry, target, risk) or signal_rr_min
            if target <= 0:
                raise ValueError("Target TP tidak valid setelah tick-size rounding.")
            signal_close_price = positive_decimal(signal.get("signal_close_price") or signal.get("close_price")) or entry
            bubble_price = positive_decimal(signal.get("bubble_price")) or signal_close_price
            signal_close = datetime.fromtimestamp(signal_close_ms / 1000, tz=timezone.utc)
            trade_created_at = now_utc()
            # PAPER LIMIT is created locally here. REAL's timer starts immediately
            # before its first Binance POST, not while exchange preflight is running.
            entry_order_created_at = trade_created_at
            entry_window_closes_at = datetime.fromtimestamp(entry_close_ms / 1000, tz=timezone.utc)
            expiry_dt = trade_created_at + timedelta(seconds=ENTRY_ORDER_TTL_SECONDS)
            atr_value = positive_decimal(signal.get("signal_atr"))
            trade = Trade(
                trade_id=generate_trade_id(pair), session_id=self.session_id, pair=pair,
                direction=direction, timeframe="M15", trend_length=volumatic.normalize_trend_length(signal.get("trend_length", self.trend_length)),
                signal=str(signal.get("signal") or ("BLUE" if direction == "LONG" else "RED")),
                signal_time=signal_close, bubble_price=bubble_price, signal_atr=atr_value,
                reference_sl=stop, stop_loss=stop, take_profit=target, rr_target=signal_rr_min,
                rr_min=signal_rr_min, rr_max=signal_rr_max,
                take_profit_source=target_source, take_profit_rr=target_rr,
                liquidity_pool_price=liquidity_pool_price,
                supertrend_atr_length=volumatic.normalize_supertrend_atr_length(
                    signal.get("supertrend_atr_length", self.supertrend_atr_length)
                ),
                supertrend_factor=volumatic.normalize_supertrend_factor(
                    signal.get("supertrend_factor", self.supertrend_factor)
                ),
                entry=entry, expires_at=expiry_dt, signal_close_price=signal_close_price,
                created_at=trade_created_at, entry_order_created_at=entry_order_created_at,
                entry_window_closes_at=entry_window_closes_at,
                ema_filter_length=filter_length, ema_filter_value=filter_value,
                ema_filter_slow_value=filter_slow_value,
                ema_filter_passed=(filter_passed if filter_length is not None else None),
                margin_usdt=self.margin_usdt, leverage=self.leverage,
                min_margin_mode=self.min_margin_enabled,
                target_notional=self.margin_usdt * Decimal(self.leverage),
            )
            # A signal candle is consumable only once per symbol, regardless of
            # direction. Binance clientOrderId uniqueness ends when an order is
            # terminal, so it cannot by itself stop a second order 10-15 seconds
            # later after the first LIMIT has already filled.
            signal_dedupe_key = f"{pair}|M15|{entry_open_ms}"
            if not await self._claim_m15_signal(signal_dedupe_key):
                await self._record_event(
                    "M15_SIGNAL_DUPLICATE_CANDLE_BLOCKED", None,
                    pair=pair, direction=direction,
                    signal_key=signal_dedupe_key,
                    signal_close_time_ms=signal_close_ms,
                    entry_candle_open_time_ms=entry_open_ms,
                    reason="Satu simbol hanya boleh membuat satu entry pada candle sinyal M15 yang sama.",
                )
                log.warning("[SCAN/DEDUP] %s duplicate signal candle blocked | key=%s", pair, signal_dedupe_key)
                return "DUPLICATE_SIGNAL_CANDLE"

            if self.real_mode:
                if not REAL_EXECUTION_ENABLED:
                    raise RuntimeError("REAL_EXECUTION_ENABLED=0; eksekusi LIVE dikunci oleh environment.")
                if not self.real.configured:
                    raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET belum lengkap.")
            sizing_filters = {
                "step_size": meta.step_size, "min_qty": meta.min_qty,
                "max_qty": meta.max_qty, "min_notional": meta.min_notional,
                "max_notional": meta.max_notional, "min_price": meta.min_price,
                "max_price": meta.max_price,
            }
            if trade.min_margin_mode:
                quantity, notional, minimum_margin, reason = compute_minimum_order(
                    trade.entry, trade.leverage, sizing_filters
                )
                if quantity is None or notional is None or minimum_margin is None:
                    raise RuntimeError(f"Minimum margin {pair} tidak valid: {reason}")
                trade.quantity, trade.target_notional, trade.actual_notional = quantity, notional, notional
                trade.margin_usdt = minimum_margin
            elif self.real_mode:
                quantity, reason = compute_quantity(trade.entry, trade.margin_usdt, trade.leverage, sizing_filters)
                if quantity is None:
                    raise RuntimeError(
                        f"Ukuran order REAL ditolak: {reason}. Fixed margin harus tetap dalam "
                        f"rentang {REAL_NOTIONAL_MIN_RATIO * 100:.0f}–{REAL_NOTIONAL_MAX_RATIO * 100:.0f}% dari target; "
                        "jika LOT_SIZE Binance tidak memungkinkan, order tidak dipaksakan lebih besar."
                    )
                trade.quantity = quantity
                trade.actual_notional = quantity * trade.entry
            else:
                trade.quantity = (trade.target_notional or (trade.margin_usdt * trade.leverage)) / trade.entry
                trade.actual_notional = trade.quantity * trade.entry
            # No exchange order is created here. REAL setups only store the entry
            # level; the WebSocket price watch sends one MARKET order on touch.
            trade.real_enabled = False
            trade.real_intent = bool(self.real_mode)
            trade.real_state = "REAL_WAITING_TRIGGER" if trade.real_intent else "PAPER_PENDING"
            # Hold the trigger until the signal message has been sent.
            trade.entry_inflight = True
            self.active_trades[trade.trade_id] = trade
            try:
                await self._record_event(
                    "M15_ENTRY_PENDING_CREATED", trade,
                    order_type="MARKET_ON_TOUCH" if trade.real_intent else "PAPER_TOUCH",
                    signal_mode="CONFIRMED_M15_NEXT_OPEN",
                    entry=decimal_to_str(trade.entry), stop_loss=decimal_to_str(trade.stop_loss),
                    take_profit=decimal_to_str(trade.take_profit),
                    rr_target=f"1:{decimal_to_str(trade.take_profit_rr)}",
                    rr_min=decimal_to_str(trade.rr_min), rr_max=decimal_to_str(trade.rr_max),
                    take_profit_source=trade.take_profit_source,
                    liquidity_pool_price=decimal_to_str(trade.liquidity_pool_price),
                    signal_close_time_ms=signal_close_ms, entry_candle_open_time_ms=entry_open_ms,
                    expires_at=iso_utc(trade.expires_at), expiry_seconds=ENTRY_ORDER_TTL_SECONDS,
                    real=trade.real_intent, quantity=decimal_to_str(trade.quantity),
                    margin_usdt=decimal_to_str(trade.margin_usdt),
                    leverage=trade.leverage, min_margin_mode=trade.min_margin_mode,
                )
                await self._save_active_state()
                await self.ws.add_symbol(pair)
                await self.reply(
                    f"{'🟢' if direction == 'LONG' else '🔴'} VOLUMATIC {direction} • {pair}\n\n"
                    f"Signal TF: M15 • EMA Trend TF: H4 • Volumatic Length: {trade.trend_length}\n"
                    f"Signal: {trade.signal} confirmed M15 bubble\n"
                    f"H4 EMA: {ema_filter_values_label(trade.ema_filter_length, trade.ema_filter_value, trade.ema_filter_slow_value)}\n"
                    f"H1 EMA: {ema_filter_values_label(trade.ema_filter_length, h1_fast_value, h1_slow_value)}\n"
                    f"Entry level (next M15 OPEN): {fmt_price(trade.entry)}\n"
                    f"Eksekusi: {'MARKET saat harga menyentuh level (dipantau WebSocket)' if trade.real_intent else 'PAPER saat harga menyentuh level'}\n"
                    f"Supertrend M15 SL: {fmt_price(trade.stop_loss)} • ATR {trade.supertrend_atr_length} / Factor {trade.supertrend_factor:g}\n"
                    f"TP: {fmt_price(trade.take_profit)} • RR 1:{fmt_num(trade.take_profit_rr, 2)} "
                    f"({'Volume Liquidity Trend' if trade.take_profit_source == 'VOLUME_LIQUIDITY' else f'baseline {trade.rr_min}R'})\n"
                    f"Auto-expire: 1 jam ({format_wib(trade.expires_at)})\n"
                    f"Margin target: {trade.margin_usdt} USDT • Leverage: {trade.leverage}x\n"
                    + (
                        f"Estimasi margin qty: {fmt_num((trade.actual_notional or Decimal('0')) / Decimal(trade.leverage), 5)} USDT "
                        f"({'guard ±5% REAL' if trade.real_intent else 'PAPER exact'})\n"
                        if not trade.min_margin_mode and trade.actual_notional is not None else ""
                    )
                    + f"Sizing: {'MINIMUM AUTO' if trade.min_margin_mode else 'FIXED TARGET'} • Qty (estimasi): {format_qty(trade.quantity or Decimal('0'))}\n"
                    f"Mode: {'REAL' if trade.real_intent else 'PAPER'}"
                )
            finally:
                trade.entry_inflight = False
            observed = self._current_price(pair)
            if observed is not None and (observed <= trade.entry if direction == "LONG" else observed >= trade.entry):
                if trade.real_intent:
                    self._schedule_market_entry(trade, observed, "SIGNAL_TIME_TOUCH")
                else:
                    await self._mark_filled(trade, trade.entry, "PAPER_MARKETABLE_AT_SUBMIT")
            return "ORDER_CREATED" if trade.status == "PENDING" else "ORDER_FILLED"

    # --------------- MARKET-on-touch entry (no exchange LIMIT orders) --------------
    @staticmethod
    def _safe_decimal(raw: Any) -> Decimal:
        try:
            value = Decimal(str(raw if raw not in (None, "") else "0"))
            return value if value.is_finite() else Decimal("0")
        except (InvalidOperation, ValueError, TypeError):
            return Decimal("0")

    @staticmethod
    def _definitive_reject(exc: Exception) -> bool:
        """A 4xx Binance answer (not rate limit) means the request was refused, not executed."""
        return (
            isinstance(exc, BinanceAPIError) and not isinstance(exc, BinanceRateLimitError)
            and exc.status_code is not None and 400 <= exc.status_code < 500
        )

    def _usdt_margin_snapshot(self, account: dict[str, Any]) -> tuple[Decimal | None, Decimal | None]:
        available = wallet = None
        for row in account.get("assets") or []:
            if isinstance(row, dict) and str(row.get("asset") or "").upper() == "USDT":
                if row.get("availableBalance") not in (None, ""):
                    available = self._safe_decimal(row.get("availableBalance"))
                if row.get("walletBalance") not in (None, ""):
                    wallet = self._safe_decimal(row.get("walletBalance"))
                break
        if available is None and account.get("availableBalance") not in (None, ""):
            available = self._safe_decimal(account.get("availableBalance"))
        if wallet is None and account.get("totalWalletBalance") not in (None, ""):
            wallet = self._safe_decimal(account.get("totalWalletBalance"))
        return available, wallet

    async def _ensure_leverage(self, pair: str, leverage: int) -> Decimal | None:
        cached = self._leverage_cache.get(pair)
        if cached and cached[0] == leverage and time.monotonic() - cached[2] < LEVERAGE_CACHE_SECONDS:
            return cached[1]
        response = await self.real.set_leverage(pair, leverage)
        max_notional = parse_max_notional(response)
        self._leverage_cache[pair] = (leverage, max_notional, time.monotonic())
        return max_notional

    def _schedule_market_entry(self, trade: Trade, price: Decimal, source: str) -> None:
        """Claim the one-shot trigger before any await, then run it off the WS loop.

        entry_inflight is the single-flight guard: later ticks, the signal-time
        check and every other caller see it set and return, so one setup can
        never start a second MARKET execution.
        """
        if (
            not self._running or trade.status != "PENDING" or not trade.real_intent
            or trade.real_enabled or trade.entry_inflight
        ):
            return
        if time.monotonic() < self._entry_retry_after.get(trade.trade_id, 0.0):
            return
        if self._binance_cooldown_remaining() > 0:
            return
        trade.entry_inflight = True
        trade.entry_triggered_at = now_utc()
        trade.entry_trigger_price = price
        task = asyncio.create_task(
            self._execute_market_entry(trade, price, source), name=f"market-entry-{trade.pair}"
        )
        self._entry_tasks.add(task)
        task.add_done_callback(self._entry_tasks.discard)

    async def _execute_market_entry(self, trade: Trade, trigger_price: Decimal, source: str) -> None:
        try:
            async with self._entry_exec_lock:
                await self._run_market_entry(trade, trigger_price, source)
        except asyncio.CancelledError:
            raise
        except MarketEntryRejected as rejection:
            await self._abort_market_entry(trade, rejection)
        except BinanceRateLimitError as exc:
            self._was_in_binance_cooldown = True
            await self._notify_binance_cooldown(exc, source="MARKET ENTRY", symbol=trade.pair)
            log.info("[ENTRY] %s ditahan rate governor; setup tetap pending dan dicoba lagi setelah cooldown.", trade.pair)
        except Exception as exc:
            if trade.real_enabled:
                # An exchange order may already exist: never abort, never re-POST.
                if trade.status == "PENDING":
                    trade.real_submit_ambiguous = True
                    trade.real_state = "REAL_SUBMIT_UNKNOWN"
                    trade.real_error = str(exc)[:500]
                log.exception("[ENTRY] %s penanganan setelah kirim gagal; direkonsiliasi dari Binance: %s", trade.pair, exc)
            else:
                attempts = self._entry_attempts.get(trade.trade_id, 0) + 1
                self._entry_attempts[trade.trade_id] = attempts
                if attempts >= MARKET_ENTRY_MAX_PRESEND_ATTEMPTS:
                    await self._abort_market_entry(
                        trade, MarketEntryRejected("ENTRY_PRESEND_FAILED", f"Preflight gagal {attempts}x: {str(exc)[:300]}")
                    )
                else:
                    self._entry_retry_after[trade.trade_id] = time.monotonic() + 5.0 * attempts
                    log.warning("[ENTRY] %s preflight gagal (%s/%s); coba lagi: %s", trade.pair, attempts, MARKET_ENTRY_MAX_PRESEND_ATTEMPTS, exc)
        finally:
            trade.entry_inflight = False

    async def _abort_market_entry(self, trade: Trade, rejection: MarketEntryRejected) -> None:
        if trade.status != "PENDING":
            return
        trade.real_enabled = False
        trade.real_submit_ambiguous = False
        trade.real_state = "REAL_ENTRY_REJECTED"
        trade.real_error = rejection.detail[:500]
        await self._finalize_trade(trade, rejection.result, None, rejection.reason)
        log.info("[ENTRY] %s %s dibatalkan: %s | %s", trade.pair, trade.direction, rejection.reason, rejection.detail)
        if rejection.notify:
            await self.reply(
                f"⛔ ENTRY {trade.direction} DIBATALKAN • {trade.pair}\n"
                f"Alasan: {rejection.reason}\n{rejection.detail[:300]}\nTidak ada order yang terbuka."
            )

    @staticmethod
    def _revert_unsent_entry(trade: Trade) -> None:
        """The POST provably did not execute: return to the local waiting state."""
        trade.real_enabled = False
        trade.real_state = "REAL_WAITING_TRIGGER"
        trade.entry_order_status = "UNKNOWN"
        trade.entry_order_id = None

    async def _run_market_entry(self, trade: Trade, trigger_price: Decimal, source: str) -> None:
        if trade.status != "PENDING" or trade.real_enabled:
            return
        direction = trade.direction
        if not self.real_mode:
            raise MarketEntryRejected("REAL_OFF_BEFORE_ENTRY", "REAL sedang OFF; entry baru tidak dikirim ke Binance.")
        if self._autostop_triggered:
            raise MarketEntryRejected("AUTOSTOP_BEFORE_ENTRY", "AUTOSTOP aktif; entry baru diblokir.")
        if now_utc() >= trade.expires_at:
            raise MarketEntryRejected("ENTRY_WATCH_EXPIRED_1H", "Batas 1 jam habis sebelum eksekusi.", result="EXPIRED", notify=False)
        if not REAL_EXECUTION_ENABLED or not self.real.configured:
            raise MarketEntryRejected("REAL_UNAVAILABLE", "API Binance tidak tersedia/tidak dikonfigurasi.")
        if any(
            other is not trade and other.pair == trade.pair and other.status in {"PENDING", "FILLED"}
            and (other.real_enabled or other.exchange_adopted)
            for other in self.active_trades.values()
        ):
            raise MarketEntryRejected("DUPLICATE_ACTIVE_SLOT", f"{trade.pair} sudah punya order/posisi aktif.")
        for order in self.exchange_open_orders.values():
            if not isinstance(order, dict):
                continue
            if normalize_symbol(str(order.get("symbol") or order.get("s") or "")) != trade.pair:
                continue
            if str(order.get("status") or order.get("X") or "NEW").upper() in {"NEW", "PARTIALLY_FILLED"}:
                raise MarketEntryRejected("EXCHANGE_OPEN_ORDER_EXISTS", f"{trade.pair} masih punya open order Binance; entry diblokir.")
        stop = trade.reference_sl or trade.stop_loss
        if stop is None:
            raise MarketEntryRejected("NO_SL_REFERENCE", "Referensi SL Supertrend tidak tersedia.")

        filters = dict(await self.rest.get_symbol_filters(trade.pair))
        try:
            leverage_max_notional = await self._ensure_leverage(trade.pair, trade.leverage)
            account = await self.real.get_account_info()
            dual = await self.real.ensure_position_mode()
        except BinanceRateLimitError:
            raise
        except BinanceAPIError as exc:
            if self._definitive_reject(exc):
                raise MarketEntryRejected("PREFLIGHT_REJECTED", str(exc)[:400]) from exc
            raise
        for row in account.get("positions") or []:
            if not isinstance(row, dict) or normalize_symbol(str(row.get("symbol") or "")) != trade.pair:
                continue
            if self._safe_decimal(row.get("positionAmt")) != 0:
                raise MarketEntryRejected(
                    "EXCHANGE_POSITION_EXISTS",
                    f"{trade.pair} masih punya posisi Binance; entry baru diblokir untuk mencegah order ganda.",
                )

        available, _wallet = self._usdt_margin_snapshot(account)
        if available is None:
            raise MarketEntryRejected("BALANCE_UNAVAILABLE", "availableBalance tidak terbaca; entry tidak dikirim.")
        sizing_available = available
        if sizing_available <= 0:
            raise MarketEntryRejected("INSUFFICIENT_AVAILABLE_BALANCE", f"Available balance {fmt_num(available, 4)} USDT tidak cukup.")

        live = self._current_price(trade.pair) or trigger_price
        if (direction == "LONG" and live <= stop) or (direction == "SHORT" and live >= stop):
            raise MarketEntryRejected(
                "PRICE_BEYOND_SL_BEFORE_ENTRY",
                f"Harga {fmt_price(live)} sudah melewati SL {fmt_price(stop)}; setup dibatalkan.",
            )
        symbol_max = Decimal(str(filters.get("max_notional") or "0"))
        if leverage_max_notional is not None:
            filters["max_notional"] = min(symbol_max, leverage_max_notional) if symbol_max > 0 else leverage_max_notional
        if trade.min_margin_mode:
            quantity, notional, minimum_margin, reason = compute_minimum_order(
                live, trade.leverage, filters, sizing_available, leverage_max_notional
            )
            if quantity is None or notional is None or minimum_margin is None:
                raise MarketEntryRejected("MIN_MARGIN_REJECTED", f"Minimum margin ditolak: {reason}")
            trade.margin_usdt, trade.target_notional = minimum_margin, notional
        else:
            quantity, reason = compute_quantity(live, trade.margin_usdt, trade.leverage, filters, sizing_available)
            if quantity is None:
                raise MarketEntryRejected(
                    "SIZE_REJECTED",
                    f"Ukuran order ditolak: {reason}. Fixed margin dijaga pada rentang "
                    f"{REAL_NOTIONAL_MIN_RATIO * 100:.0f}–{REAL_NOTIONAL_MAX_RATIO * 100:.0f}% dari target; "
                    "order ditolak bila aturan LOT_SIZE memaksa margin terlalu jauh.",
                )
        market_cap = Decimal(str(filters.get("market_max_qty") or "0"))
        if market_cap > 0 and quantity > market_cap:
            raise MarketEntryRejected("QTY_ABOVE_MARKET_MAX", f"Qty {format_qty(quantity)} melebihi batas MARKET {format_qty(market_cap)}.")

        trade.position_side = ("LONG" if direction == "LONG" else "SHORT") if dual else "BOTH"
        identity = f"{trade.pair}|{direction}|M15|{int(trade.signal_time.timestamp() * 1000)}|MARKET"
        trade.entry_client_order_id = "VT" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:28]
        trade.quantity = quantity
        trade.actual_notional = quantity * live
        # Synchronous commit point: from here on an exchange order may exist, so
        # the trade is REAL, matched by clientOrderId, and never POSTed again.
        trade.real_enabled = True
        trade.real_state = "REAL_SUBMITTING"
        trade.real_submit_ambiguous = False
        trade.real_error = None
        trade.entry_order_terminal = False
        trade.entry_order_status = "SUBMITTING"

        response: dict[str, Any] | None = None
        error: Exception | None = None
        try:
            response = await self.real.place_market_entry(
                symbol=trade.pair, side=trade.entry_side, quantity=quantity,
                position_side=trade.position_side, client_order_id=trade.entry_client_order_id,
            )
        except BinanceRateLimitError as exc:
            if getattr(exc, "internal", False) or exc.status_code in {418, 429}:
                self._revert_unsent_entry(trade)  # rejected by rate limit: nothing executed
                raise
            error = exc
        except Exception as exc:
            error = exc

        if error is not None:
            definitive = self._definitive_reject(error)
            try:
                response = await self.real.get_order(trade.pair, client_order_id=trade.entry_client_order_id)
            except BinanceAPIError as lookup_exc:
                if lookup_exc.code == -2013 and definitive:
                    self._revert_unsent_entry(trade)
                    raise MarketEntryRejected("BINANCE_MARKET_REJECTED", str(error)[:400]) from error
                response = None
            except Exception:
                response = None
            if response is None:
                trade.real_submit_ambiguous = True
                trade.real_state = "REAL_SUBMIT_UNKNOWN"
                trade.real_error = str(error)[:500]
                await self._save_active_state()
                log.warning("[ENTRY] %s hasil POST market tidak pasti; hanya lookup clientOrderId=%s, tidak ada POST ulang | %s",
                            trade.pair, trade.entry_client_order_id, error)
                await self.reply(
                    "╔══ VOL / REAL MARKET ENTRY ══╗\n"
                    f"PAIR      : {trade.pair} · {trade.direction}\n"
                    f"CLIENT ID : {trade.entry_client_order_id}\n"
                    "STATE     : ACK BINANCE BELUM PASTI\n"
                    "SAFETY    : POST ULANG DIMATIKAN · HANYA LOOKUP"
                )
                return
        await self._apply_market_entry_response(trade, response, live, source)

    async def _apply_market_entry_response(
        self, trade: Trade, response: dict[str, Any], live: Decimal, source: str
    ) -> None:
        terminal_states = {"FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}

        def read(order: dict[str, Any]) -> tuple[str, Decimal, Decimal]:
            return (
                str(order.get("status") or "UNKNOWN").upper(),
                self._safe_decimal(order.get("executedQty")),
                self._safe_decimal(order.get("avgPrice")),
            )

        status, executed, avg = read(response)
        trade.real_submit_ambiguous = False
        if response.get("orderId") is not None:
            trade.entry_order_id = int(response["orderId"])
        # RESULT normally carries the final MARKET state; confirm by GET if not.
        if status not in terminal_states or (status == "FILLED" and avg <= 0):
            for _ in range(3):
                try:
                    polled = await self.real.get_order(
                        trade.pair, order_id=trade.entry_order_id, client_order_id=trade.entry_client_order_id
                    )
                except BinanceAPIError:
                    break
                response = polled
                status, executed, avg = read(polled)
                if status in terminal_states and (status != "FILLED" or avg > 0):
                    break
                await asyncio.sleep(0.3)
        trade.entry_order_status = status
        trade.entry_executed_qty = executed
        if executed > 0:
            cum_quote = self._safe_decimal(response.get("cumQuote"))
            fill_price = avg if avg > 0 else (cum_quote / executed if cum_quote > 0 else live)
            trade.quantity = executed
            trade.entry_order_terminal = status in terminal_states
            trade.real_state = "REAL_FILLED" if trade.entry_order_terminal else "REAL_PARTIALLY_FILLED"
            deferred = await self._mark_filled(
                trade, fill_price, source or "BINANCE_MARKET_RESPONSE",
                defer_persist=True, filled_at=self._exchange_event_time(response),
            )
            # TP/SL first; GitHub persistence of the fill follows.
            await self._install_protection_now(trade)
            if deferred is not None:
                await self._record_event(deferred[0], trade, **deferred[1])
            await self._record_event(
                "MARKET_ENTRY_FILLED", trade, client_order_id=trade.entry_client_order_id,
                order_id=trade.entry_order_id, trigger_price=decimal_to_str(trade.entry_trigger_price),
                fill_price=decimal_to_str(trade.fill_price), quantity=decimal_to_str(trade.quantity),
                protection_state=trade.protection_state, source=source,
            )
            await self._save_active_state()
            if not trade.exchange_adopted:
                self._schedule_ema_exit_initialization(trade)
                await self._sync_ws_symbols()
            return
        if status in terminal_states:
            self._revert_unsent_entry(trade)
            raise MarketEntryRejected(f"BINANCE_MARKET_{status}", f"Order market berakhir {status} tanpa eksekusi.")
        # Unexpected live MARKET order without execution: keep it as a REAL pending
        # order; the existing reconcile path polls it by clientOrderId.
        trade.real_state = "REAL_PENDING"
        await self._save_active_state()

    async def _install_protection_now(self, trade: Trade) -> None:
        """Place closePosition SL/TP immediately; maintenance continues on any failure."""
        if trade.protection_state == "INVALID_RISK":
            return  # _reconcile_filled_real raises the one-time warning for this state
        for attempt in range(1, MARKET_ENTRY_PROTECTION_ATTEMPTS + 1):
            try:
                await self._ensure_real_protection(trade)
                if trade.protection_state == "ACTIVE":
                    return
            except BinanceRateLimitError:
                break
            except Exception as exc:
                log.warning("[ENTRY] %s proteksi percobaan %s/%s gagal: %s", trade.pair, attempt, MARKET_ENTRY_PROTECTION_ATTEMPTS, exc)
                if trade.protection_state == "ERROR_REJECTED":
                    break
            if attempt < MARKET_ENTRY_PROTECTION_ATTEMPTS:
                await asyncio.sleep(0.5)
        if trade.protection_state != "ACTIVE":
            log.error("[ENTRY] %s %s terisi tetapi TP/SL belum lengkap (%s); maintenance melanjutkan.",
                      trade.pair, trade.direction, trade.protection_state)

    # -------------------------- Orders / positions -----------------------
    @staticmethod
    def _order_fill_price(order: dict[str, Any], fallback: Decimal) -> Decimal:
        for key in ("avgPrice", "price"):
            try:
                value = Decimal(str(order.get(key) or "0"))
                if value > 0: return value
            except (InvalidOperation, TypeError):
                pass
        return fallback

    async def _mark_filled(
        self, trade: Trade, price: Decimal, source: str, *, defer_persist: bool = False,
        filled_at: datetime | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        # Once Binance has confirmed both closePosition Algo Orders, protection is
        # intentionally locked. Later cumulative-fill/account/REST events may update
        # size and average entry for PnL, but must not reprice, reinstall, re-query,
        # or re-notify TP/SL. This also keeps the original user-confirmed levels.
        protection_locked = bool(
            trade.real_enabled
            and trade.protection_state == "ACTIVE"
            and trade.sl_algo_id is not None
            and trade.tp_algo_id is not None
        )
        first_fill = trade.status == "PENDING"
        if first_fill:
            trade.status = "FILLED"
            trade.filled_at = filled_at or now_utc()
        if price > 0:
            trade.fill_price = price
        if trade.fill_price is None or trade.fill_price <= 0:
            trade.fill_price = trade.entry
        if trade.quantity is None:
            trade.quantity = (trade.target_notional or trade.margin_usdt * trade.leverage) / trade.fill_price
        trade.actual_notional = trade.quantity * trade.fill_price
        if trade.leverage > 0:
            trade.margin_usdt = trade.actual_notional / Decimal(trade.leverage)

        stop = trade.reference_sl or trade.stop_loss
        risk_decimal = trade.initial_risk if protection_locked else None
        if protection_locked:
            # Keep the exact Binance-confirmed SL/TP. Do not change protection_state.
            trade.protection_state = "ACTIVE"
            trade.protection_error = None
        else:
            if stop is not None:
                candidate_risk = trade.fill_price - stop if trade.direction == "LONG" else stop - trade.fill_price
                if candidate_risk > 0:
                    risk_decimal = candidate_risk
            if risk_decimal is None:
                trade.protection_state = "INVALID_RISK"
                trade.protection_error = "Actual fill price and M15 Supertrend SL do not form a positive risk distance."
            else:
                trade.initial_risk = risk_decimal
                meta = self.symbols.get(trade.pair)
                tick_size = meta.tick_size if meta is not None else Decimal("0")
                if meta is not None:
                    trade.stop_loss = (
                        quantized_price(stop, tick_size)
                        if trade.direction == "LONG"
                        else quantized_price_ceiling(stop, tick_size)
                    )
                else:
                    trade.stop_loss = stop

                raw_base_target = trade.fill_price + (
                    trade.initial_risk * trade.rr_target
                    if trade.direction == "LONG"
                    else -trade.initial_risk * trade.rr_target
                )
                base_target = (
                    quantized_price_ceiling(raw_base_target, tick_size)
                    if meta is not None and trade.direction == "LONG"
                    else (
                        quantized_price(raw_base_target, tick_size)
                        if meta is not None else raw_base_target
                    )
                )

                # A selected pool is an absolute structural price. Keep it only
                # when the actual/cumulative fill still places it inside this
                # trade's saved RR_MIN..RR_MAX band; otherwise use RR_MIN fallback.
                candidate_pool = trade.liquidity_pool_price or trade.take_profit
                pool_rr = (
                    directional_reward_r(trade.direction, trade.fill_price, candidate_pool, trade.initial_risk)
                    if candidate_pool is not None else None
                )
                pool_still_valid = (
                    trade.take_profit_source == "VOLUME_LIQUIDITY"
                    and candidate_pool is not None
                    and is_valid_liquidity_reward_r(pool_rr, trade.rr_min, trade.rr_max)
                    and (
                        candidate_pool > trade.fill_price
                        if trade.direction == "LONG"
                        else candidate_pool < trade.fill_price
                    )
                )
                if pool_still_valid:
                    trade.take_profit = candidate_pool
                    trade.take_profit_rr = pool_rr or trade.rr_min
                else:
                    trade.take_profit = base_target
                    trade.take_profit_rr = (
                        directional_reward_r(trade.direction, trade.fill_price, base_target, trade.initial_risk)
                        or trade.rr_min
                    )
                    if trade.take_profit_source == "VOLUME_LIQUIDITY":
                        trade.take_profit_source = "RR_MIN_FALLBACK_AFTER_FILL"
                trade.protection_state = "WAITING_FOR_EXCHANGE" if trade.real_enabled else "PAPER_ACTIVE"

        trade.real_state = "REAL_FILLED" if trade.real_enabled else "PAPER_FILLED"
        event_name = "ORDER_FILLED" if first_fill else "ORDER_PARTIAL_FILL_UPDATED"
        event_extra = dict(
            fill_price=decimal_to_str(trade.fill_price), source=source,
            quantity=decimal_to_str(trade.quantity), stop_loss=decimal_to_str(trade.stop_loss),
            take_profit=decimal_to_str(trade.take_profit),
            rr_target=f"1:{decimal_to_str(trade.take_profit_rr)}",
            rr_min=decimal_to_str(trade.rr_min), rr_max=decimal_to_str(trade.rr_max),
            take_profit_source=trade.take_profit_source,
            liquidity_pool_price=decimal_to_str(trade.liquidity_pool_price),
            protection_locked=protection_locked,
        )
        if defer_persist:
            return event_name, event_extra
        await self._record_event(event_name, trade, **event_extra)
        await self._save_active_state()
        # REAL protection is deferred to _reconcile_filled_real, which first confirms
        # that this exact LONG/SHORT exposure still exists on Binance. This prevents
        # false "SL cannot be placed" alerts after an already-flat close event.
        log.info("[TRADE] FILLED %s %s @ %s (%s) | SL=%s TP=%s protection=%s locked=%s",
                 trade.pair, trade.direction, trade.fill_price, source,
                 trade.stop_loss, trade.take_profit, trade.protection_state, protection_locked)
        if trade.status == "FILLED" and not trade.exchange_adopted:
            # PAPER can warm up immediately. REAL first gets its exchange-side
            # SL/TP attempt from maintenance; H1 history must not compete with
            # the initial protection requests for the shared Binance REST lock.
            if not trade.real_enabled:
                self._schedule_ema_exit_initialization(trade)
            await self._sync_ws_symbols()
        return None

    async def _ensure_real_protection(self, trade: Trade) -> None:
        """Single-flight wrapper: the entry task and the maintenance loop never install in parallel."""
        lock = self._protection_locks.setdefault(trade.trade_id, asyncio.Lock())
        async with lock:
            await self._ensure_real_protection_locked(trade)

    async def _ensure_real_protection_locked(self, trade: Trade) -> None:
        """Install closePosition SL/TP once; confirmed protection is left untouched."""
        if not trade.real_enabled or trade.status != "FILLED" or trade.exchange_adopted:
            return
        # User-requested behavior: after BOTH protection orders are confirmed,
        # leave them on Binance. Do not poll/replace them from maintenance and do
        # not send the success notification again. /trade shows the cached status.
        if (trade.protection_state == "ACTIVE"
                and trade.sl_algo_id is not None
                and trade.tp_algo_id is not None):
            return
        if self._binance_cooldown_remaining() > 0:
            raise _local_rate_limit_error("/fapi/v1/algoOrder")
        if trade.fill_price is None or trade.reference_sl is None:
            raise RuntimeError("Fill/SL reference belum tersedia untuk proteksi.")
        risk_decimal = (trade.fill_price - trade.reference_sl if trade.direction == "LONG"
                        else trade.reference_sl - trade.fill_price)
        if risk_decimal <= 0:
            raise RuntimeError("SL Supertrend tidak berada di sisi protektif dari harga fill; proteksi ditolak.")
        trade.initial_risk = risk_decimal
        meta = self.symbols.get(trade.pair)
        if meta is None:
            filters = await self.rest.get_symbol_filters(trade.pair)
            # Convert the compact dict to the same tickSize contract used elsewhere.
            tick_size = Decimal(str(filters.get("tick_size") or "0"))
        else:
            tick_size = meta.tick_size
        if tick_size <= 0:
            raise RuntimeError("tickSize tidak tersedia untuk quantize TP/SL.")
        tp_still_working = bool(trade.tp_algo_id is not None and trade.tp_algo_status.upper() not in {"CANCELED", "FINISHED", "REJECTED", "EXPIRED", "TRIGGERED", "REPRICE_REQUIRED"})
        if trade.direction == "LONG":
            stop = quantized_price(trade.reference_sl, tick_size)
            raw_tp = trade.fill_price + trade.initial_risk * trade.rr_target
            base_target = quantized_price_ceiling(raw_tp, tick_size)
            close_side = "SELL"
        else:
            stop = quantized_price_ceiling(trade.reference_sl, tick_size)
            raw_tp = trade.fill_price - trade.initial_risk * trade.rr_target
            base_target = quantized_price(raw_tp, tick_size)
            close_side = "BUY"

        target = base_target
        if tp_still_working and trade.take_profit is not None:
            # Never alter an already-working Binance TP protection.
            target = trade.take_profit
        elif trade.take_profit_source == "VOLUME_LIQUIDITY":
            candidate_pool = trade.liquidity_pool_price or trade.take_profit
            pool_rr = (
                directional_reward_r(trade.direction, trade.fill_price, candidate_pool, trade.initial_risk)
                if candidate_pool is not None else None
            )
            if (
                candidate_pool is not None
                and is_valid_liquidity_reward_r(pool_rr, trade.rr_min, trade.rr_max)
                and (candidate_pool > trade.fill_price if trade.direction == "LONG" else candidate_pool < trade.fill_price)
            ):
                target = candidate_pool
                trade.take_profit_rr = pool_rr or trade.rr_min
            else:
                trade.take_profit_source = "RR_MIN_FALLBACK_AFTER_FILL"
                trade.take_profit_rr = directional_reward_r(
                    trade.direction, trade.fill_price, base_target, trade.initial_risk
                ) or trade.rr_min
        else:
            trade.take_profit_rr = directional_reward_r(
                trade.direction, trade.fill_price, base_target, trade.initial_risk
            ) or trade.rr_min
        if stop <= 0 or target <= 0 or (trade.direction == "LONG" and not stop < trade.fill_price < target) or (trade.direction == "SHORT" and not target < trade.fill_price < stop):
            raise RuntimeError("Level TP/SL tidak valid pada harga fill aktual.")
        trade.stop_loss, trade.take_profit = stop, target
        trade.take_profit_rr = directional_reward_r(
            trade.direction, trade.fill_price, target, trade.initial_risk
        ) or trade.take_profit_rr
        base = re.sub(r"[^A-Za-z0-9_-]", "", trade.trade_id.upper())[-26:]
        trade.sl_client_algo_id = trade.sl_client_algo_id or f"VTSL{base}"[:36]
        trade.tp_client_algo_id = trade.tp_client_algo_id or f"VTTP{base}"[:36]
        await self._save_active_state()

        async def ensure_one(kind: str, order_type: str, trigger: Decimal) -> tuple[int | None, str]:
            id_attr = "sl_algo_id" if kind == "SL" else "tp_algo_id"
            client_attr = "sl_client_algo_id" if kind == "SL" else "tp_client_algo_id"
            status_attr = "sl_algo_status" if kind == "SL" else "tp_algo_status"
            ambiguous_attr = "sl_submit_ambiguous" if kind == "SL" else "tp_submit_ambiguous"
            algo_id = getattr(trade, id_attr)
            client_id = getattr(trade, client_attr)
            current_status = str(getattr(trade, status_attr) or "").upper()
            if kind == "TP" and current_status == "REPRICE_REQUIRED":
                old_status = "UNKNOWN"
                if algo_id is not None or client_id:
                    try:
                        existing = await self.real.get_algo_order(
                            symbol=trade.pair, algo_id=algo_id,
                            client_algo_id=(client_id if algo_id is None else None),
                        )
                        old_status = str(existing.get("algoStatus") or existing.get("status") or "UNKNOWN").upper()
                        if existing.get("algoId") is not None:
                            setattr(trade, id_attr, int(existing["algoId"]))
                    except BinanceRateLimitError:
                        raise
                    except BinanceAPIError as lookup_exc:
                        if lookup_exc.code not in {-2011, -2013}:
                            raise
                        old_status = "CANCELED"
                if old_status in {"TRIGGERED", "FINISHED"}:
                    trade.protection_exit_reason = "TAKE_PROFIT"
                    setattr(trade, status_attr, old_status)
                    await self._save_active_state()
                    return getattr(trade, id_attr), old_status
                if old_status not in {"CANCELED", "EXPIRED", "REJECTED"}:
                    try:
                        canceled = await self.real.cancel_algo_order(
                            symbol=trade.pair, algo_id=algo_id,
                            client_algo_id=(client_id if algo_id is None else None),
                        )
                        old_status = str(canceled.get("algoStatus") or canceled.get("status") or "CANCELED").upper()
                    except BinanceRateLimitError:
                        raise
                    except BinanceAPIError as cancel_exc:
                        if cancel_exc.code not in {-2011, -2013}:
                            raise
                        try:
                            existing = await self.real.get_algo_order(
                                symbol=trade.pair, algo_id=algo_id,
                                client_algo_id=(client_id if algo_id is None else None),
                            )
                            old_status = str(existing.get("algoStatus") or existing.get("status") or "UNKNOWN").upper()
                        except BinanceAPIError as lookup_exc:
                            if lookup_exc.code in {-2011, -2013}:
                                old_status = "CANCELED"
                            else:
                                raise
                if old_status not in {"CANCELED", "FINISHED", "REJECTED", "EXPIRED", "TRIGGERED"}:
                    raise RuntimeError(f"TP lama belum terminal saat reprice ({old_status}); TP baru tidak dikirim.")
                if old_status in {"TRIGGERED", "FINISHED"}:
                    trade.protection_exit_reason = "TAKE_PROFIT"
                    setattr(trade, status_attr, old_status)
                    await self._save_active_state()
                    return getattr(trade, id_attr), old_status
                setattr(trade, id_attr, None)
                setattr(trade, client_attr, f"VTTP{uuid4().hex[:28]}")
                setattr(trade, status_attr, "NOT_PLACED")
                setattr(trade, ambiguous_attr, False)
                algo_id = None
                client_id = getattr(trade, client_attr)
                current_status = "NOT_PLACED"
                await self._save_active_state()
            if algo_id is not None or client_id:
                try:
                    existing = await self.real.get_algo_order(symbol=trade.pair, algo_id=algo_id, client_algo_id=(client_id if algo_id is None else None))
                    found_id = existing.get("algoId", existing.get("ai"))
                    if found_id is not None:
                        setattr(trade, id_attr, int(found_id))
                    status = str(existing.get("algoStatus") or existing.get("status") or current_status or "WORKING").upper()
                    setattr(trade, status_attr, status)
                    if status == "REJECTED":
                        trade.protection_state = "ERROR_REJECTED"
                        trade.protection_error = f"{kind} Algo Order ditolak Binance saat lookup; POST ulang otomatis dihentikan."
                        await self._save_active_state()
                        raise RuntimeError(trade.protection_error)
                    if status not in {"CANCELED", "FINISHED", "EXPIRED", "TRIGGERED"}:
                        return getattr(trade, id_attr), status
                except BinanceRateLimitError:
                    raise
                except BinanceAPIError as lookup_exc:
                    if lookup_exc.code == -2013:
                        if getattr(trade, ambiguous_attr):
                            raise RuntimeError(f"{kind} Algo Order masih ambigu; POST ulang dinonaktifkan sampai Binance mengonfirmasi clientAlgoId.")
                    else:
                        raise
            if current_status == "REJECTED":
                trade.protection_state = "ERROR_REJECTED"
                trade.protection_error = f"{kind} Algo Order ditolak Binance; POST ulang otomatis dihentikan."
                await self._save_active_state()
                raise RuntimeError(trade.protection_error)
            if getattr(trade, ambiguous_attr):
                raise RuntimeError(f"{kind} Algo Order berstatus submit ambiguous; tidak akan POST ulang.")
            # Binance trigger source follows the user's requested TP/SL split:
            # TP triggers from Mark Price; SL triggers from Last/Contract Price.
            working_type = "MARK_PRICE" if kind == "TP" else "CONTRACT_PRICE"
            try:
                response = await self.real.place_algo_order(
                    symbol=trade.pair, side=close_side, order_type=order_type,
                    trigger_price=trigger, position_side=trade.position_side or "BOTH",
                    client_algo_id=client_id, working_type=working_type,
                )
            except BinanceRateLimitError:
                raise
            except BinanceAPIError as submit_exc:
                if submit_exc.code == -4509:
                    # Binance says this closePosition/GTE protection has no open
                    # exposure to attach to. Treat it as a flat/race condition, not
                    # a protection rejection; the next side-specific reconcile closes it.
                    setattr(trade, status_attr, "NOT_PLACED")
                    trade.protection_state = "WAITING_FOR_EXCHANGE"
                    trade.protection_error = "Binance reports no matching open position (-4509); waiting for side-specific reconciliation."
                    await self._save_active_state()
                    log.info("[PROTECTION] %s %s became flat before %s could be placed (-4509).", trade.pair, trade.direction, kind)
                    return None, "NO_POSITION"
                if submit_exc.status_code is not None and 400 <= submit_exc.status_code < 500 and submit_exc.status_code not in {418, 429}:
                    setattr(trade, status_attr, "REJECTED")
                    trade.protection_state = "ERROR_REJECTED"
                    trade.protection_error = f"{kind} Algo Order ditolak Binance: {str(submit_exc)[:400]}"
                    await self._save_active_state()
                    raise RuntimeError(trade.protection_error) from submit_exc
                setattr(trade, ambiguous_attr, True)
                setattr(trade, status_attr, "SUBMIT_UNKNOWN")
                await self._save_active_state()
                try:
                    existing = await self.real.get_algo_order(symbol=trade.pair, client_algo_id=client_id)
                    found_id = existing.get("algoId", existing.get("ai"))
                    if found_id is not None:
                        setattr(trade, id_attr, int(found_id))
                        setattr(trade, status_attr, str(existing.get("algoStatus") or "WORKING").upper())
                        setattr(trade, ambiguous_attr, False)
                        return int(found_id), getattr(trade, status_attr)
                except Exception:
                    pass
                raise RuntimeError(f"{kind} Algo Order acknowledgement unknown; lookup-only recovery: {submit_exc}") from submit_exc
            except Exception as submit_exc:
                setattr(trade, ambiguous_attr, True)
                setattr(trade, status_attr, "SUBMIT_UNKNOWN")
                await self._save_active_state()
                # Only a lookup is permitted after uncertain transport/server result.
                try:
                    existing = await self.real.get_algo_order(symbol=trade.pair, client_algo_id=client_id)
                    found_id = existing.get("algoId", existing.get("ai"))
                    if found_id is not None:
                        setattr(trade, id_attr, int(found_id))
                        setattr(trade, status_attr, str(existing.get("algoStatus") or "WORKING").upper())
                        setattr(trade, ambiguous_attr, False)
                        return int(found_id), getattr(trade, status_attr)
                except Exception:
                    pass
                raise RuntimeError(f"{kind} Algo Order acknowledgement unknown; lookup-only recovery: {submit_exc}") from submit_exc
            found_id = response.get("algoId", response.get("ai"))
            if found_id is not None:
                setattr(trade, id_attr, int(found_id))
            setattr(trade, ambiguous_attr, False)
            status = str(response.get("algoStatus") or "NEW").upper()
            setattr(trade, status_attr, status)
            return getattr(trade, id_attr), status

        # SL first: protection is never intentionally left with TP only.
        if trade.sl_algo_id is None or trade.sl_algo_status.upper() in {"CANCELED", "FINISHED", "REJECTED", "EXPIRED", "TRIGGERED", "NOT_PLACED"}:
            _sl_id, _sl_status = await ensure_one("SL", "STOP_MARKET", stop)
            await self._save_active_state()
            if _sl_status == "NO_POSITION":
                return
        if trade.tp_algo_id is None or trade.tp_algo_status.upper() in {"CANCELED", "FINISHED", "REJECTED", "EXPIRED", "TRIGGERED", "NOT_PLACED", "REPRICE_REQUIRED"}:
            _tp_id, _tp_status = await ensure_one("TP", "TAKE_PROFIT_MARKET", target)
            await self._save_active_state()
            if _tp_status == "NO_POSITION":
                return
        # Recheck position after submitting closePosition protections; if it's already flat,
        # the ACCOUNT_UPDATE/User WS or next reconciliation will finalize it.
        trade.protection_state = "ACTIVE" if trade.sl_algo_id is not None and trade.tp_algo_id is not None else "PARTIAL"
        trade.protection_error = None if trade.protection_state == "ACTIVE" else "One protective Algo Order is not yet confirmed."
        await self._record_event("REAL_TP_SL_PROTECTION_ACTIVE", trade,
                                 sl=decimal_to_str(stop), tp=decimal_to_str(target),
                                 rr_target=f"1:{decimal_to_str(trade.take_profit_rr)}",
                                 take_profit_source=trade.take_profit_source,
                                 liquidity_pool_price=decimal_to_str(trade.liquidity_pool_price),
                                 sl_algo_id=trade.sl_algo_id, tp_algo_id=trade.tp_algo_id,
                                 sl_working_type="CONTRACT_PRICE",
                                 tp_working_type="MARK_PRICE", close_position=True)
        await self._save_active_state()
        if trade.protection_state == "ACTIVE":
            await self.reply(
                f"🛡️ BINANCE TP/SL AKTIF • {trade.pair} {trade.direction}\n"
                f"Entry avg: {fmt_price(trade.fill_price)}\nSL Supertrend M15: {fmt_price(trade.stop_loss)}\n"
                f"TP RR 1:{fmt_num(trade.take_profit_rr, 2)} "
                f"({'Volume Liquidity Trend' if trade.take_profit_source == 'VOLUME_LIQUIDITY' else 'RR baseline'}): "
                f"{fmt_price(trade.take_profit)}\n"
                f"Trigger: SL Last Price / TP Mark Price\n"
                "Mode: closePosition Algo Orders"
            )

    async def _cancel_real_protections(self, trade: Trade, *, keep_triggered: str | None = None) -> bool:
        """Cancel sibling Algo exits and confirm cleanup before releasing the trade slot."""
        if not trade.real_enabled:
            return True
        if self._binance_cooldown_remaining() > 0:
            trade.protection_state = "CANCEL_WAIT_API"
            trade.protection_error = "Proteksi closePosition menunggu REST pulih; tidak dianggap sudah dibatalkan."
            await self._save_active_state()
            return False
        all_cleared = True
        errors: list[str] = []
        terminal = {"CANCELED", "FINISHED", "REJECTED", "EXPIRED", "TRIGGERED"}
        for kind, algo_id, client_id, status in (
            ("SL", trade.sl_algo_id, trade.sl_client_algo_id, trade.sl_algo_status),
            ("TP", trade.tp_algo_id, trade.tp_client_algo_id, trade.tp_algo_status),
        ):
            current = str(status or "").upper()
            if kind == keep_triggered or (algo_id is None and not client_id) or current in terminal:
                continue
            try:
                response = await self.real.cancel_algo_order(
                    symbol=trade.pair, algo_id=algo_id,
                    client_algo_id=(client_id if algo_id is None else None),
                )
                confirmed = str(response.get("algoStatus") or response.get("status") or "CANCELED").upper()
                if confirmed not in terminal:
                    confirmed = "CANCELED" if response else "UNKNOWN"
                if kind == "SL":
                    trade.sl_algo_status = confirmed
                else:
                    trade.tp_algo_status = confirmed
                if confirmed not in terminal:
                    all_cleared = False
                    errors.append(f"{kind} cancel belum terminal ({confirmed})")
            except BinanceRateLimitError as exc:
                all_cleared = False
                errors.append(f"{kind}: REST cooldown ({str(exc)[:180]})")
            except BinanceAPIError as exc:
                # -2011/-2013 can mean the Algo Order is already gone; query once
                # before declaring cancellation complete.
                try:
                    existing = await self.real.get_algo_order(
                        symbol=trade.pair, algo_id=algo_id,
                        client_algo_id=(client_id if algo_id is None else None),
                    )
                    confirmed = str(existing.get("algoStatus") or existing.get("status") or "UNKNOWN").upper()
                    if kind == "SL": trade.sl_algo_status = confirmed
                    else: trade.tp_algo_status = confirmed
                    if confirmed not in terminal:
                        all_cleared = False
                        errors.append(f"{kind}: status tetap {confirmed}")
                except BinanceAPIError as lookup_exc:
                    if lookup_exc.code in {-2011, -2013}:
                        if kind == "SL": trade.sl_algo_status = "CANCELED"
                        else: trade.tp_algo_status = "CANCELED"
                    else:
                        all_cleared = False
                        errors.append(f"{kind}: {str(lookup_exc)[:180]}")
                except Exception as lookup_exc:
                    all_cleared = False
                    errors.append(f"{kind}: {str(lookup_exc)[:180]}")
            except Exception as exc:
                all_cleared = False
                errors.append(f"{kind}: {str(exc)[:180]}")
        if not all_cleared:
            trade.protection_state = "CANCEL_WAIT_API"
            trade.protection_error = "; ".join(errors)[:500] or "Algo Order belum terkonfirmasi dibatalkan."
            await self._save_active_state()
            log.warning("[PROTECTION] %s cancellation incomplete: %s", trade.pair, trade.protection_error)
            return False
        trade.protection_state = "CANCELED"
        trade.protection_error = None
        await self._save_active_state()
        return True

    async def _cancel_real_entry_remainder(self, trade: Trade, reason: str) -> None:
        """Cancel only the unfilled remainder; never treat a rate-limit as cancellation.

        A partial fill is an actual position. Its filled quantity stays FILLED while
        the remaining GTC LIMIT is canceled/reconciled once REST is available.
        """
        if not trade.real_enabled or trade.entry_order_terminal:
            return
        if self._binance_cooldown_remaining() > 0:
            trade.pending_cancel_reason = reason
            trade.real_state = "REAL_EXPIRY_WAIT_API" if "EXPIRE" in reason else "REAL_CANCEL_WAIT_API"
            trade.real_error = f"REST cooldown aktif; LIMIT tetap dibiarkan berjalan. {reason}"
            await self._save_active_state()
            # Bubble/maintenance expiry does not call this while cooling down.
            # A manual /close or /del receives this signal and is durably queued.
            raise _local_rate_limit_error("/fapi/v1/order")
        if not (trade.entry_order_id or trade.entry_client_order_id):
            trade.pending_cancel_reason = reason
            trade.real_state = "REAL_CANCEL_UNKNOWN"
            trade.real_error = "Order REAL belum memiliki ID untuk rekonsiliasi/cancel."
            await self._save_active_state()
            raise RuntimeError(trade.real_error)

        try:
            order = await self.real.get_order(
                trade.pair, order_id=trade.entry_order_id,
                client_order_id=trade.entry_client_order_id,
            )
            status = str(order.get("status") or "UNKNOWN").upper()
            executed = Decimal(str(order.get("executedQty") or "0"))
            if status in {"NEW", "PARTIALLY_FILLED", "UNKNOWN"}:
                if status == "UNKNOWN":
                    raise RuntimeError(f"Status order belum diketahui saat cancel: {status}")
                await self.real.cancel_order(
                    trade.pair, order_id=trade.entry_order_id,
                    client_order_id=trade.entry_client_order_id,
                )
                # Confirm terminal state. A cancel response alone is not enough.
                order = await self.real.get_order(
                    trade.pair, order_id=trade.entry_order_id,
                    client_order_id=trade.entry_client_order_id,
                )
                status = str(order.get("status") or "UNKNOWN").upper()
                executed = Decimal(str(order.get("executedQty") or "0"))

            trade.entry_order_status = status
            trade.entry_executed_qty = executed
            if order.get("orderId") is not None:
                trade.entry_order_id = int(order["orderId"])
            if status in {"FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}:
                trade.entry_order_terminal = True
                trade.pending_cancel_reason = None
                self._remove_exchange_order_cache(trade)
                if executed > 0:
                    trade.quantity = executed
                    avg = self._order_fill_price(order, trade.fill_price or trade.entry)
                    if trade.status == "PENDING":
                        await self._mark_filled(
                            trade, avg, "BINANCE_CANCEL_RECONCILED_FILL",
                            filled_at=self._exchange_event_time(order),
                        )
                    else:
                        trade.fill_price = avg
                        trade.actual_notional = executed * avg
                        trade.margin_usdt = trade.actual_notional / Decimal(max(1, trade.leverage))
                        trade.real_state = "REAL_PARTIAL_TERMINAL" if status != "FILLED" else "REAL_FILLED"
                        await self._record_event("ENTRY_REMAINDER_TERMINAL", trade, order_status=status, executed_qty=decimal_to_str(executed), reason=reason)
                        await self._save_active_state()
                    return
                if trade.status == "PENDING":
                    result = "EXPIRED" if status == "EXPIRED" or "EXPIRE" in reason else "CANCELED"
                    await self._finalize_trade(trade, result, None, f"{reason}; BINANCE_{status}")
                else:
                    trade.real_state = "REAL_FILLED"
                    await self._save_active_state()
                return

            trade.pending_cancel_reason = reason
            trade.real_state = "REAL_CANCEL_UNKNOWN"
            trade.real_error = f"Cancel belum terkonfirmasi; status Binance={status}."
            await self._save_active_state()
            raise RuntimeError(trade.real_error)
        except BinanceRateLimitError:
            trade.pending_cancel_reason = reason
            trade.real_state = "REAL_CANCEL_WAIT_API"
            trade.real_error = "Cancel ditunda sampai Binance REST kembali tersedia; order tidak dibatalkan saat cooldown."
            await self._save_active_state()
            raise
        except Exception as exc:
            trade.pending_cancel_reason = reason
            trade.real_state = "REAL_CANCEL_UNKNOWN"
            trade.real_error = str(exc)[:500]
            await self._save_active_state()
            log.exception("[REAL] cancel/reconcile remainder %s gagal; order tetap ditahan", trade.pair)
            raise

    async def _cancel_pending_trade(self, trade: Trade, reason: str) -> None:
        """Cancel a pending order only after exchange confirmation.

        When Binance is rate-limited, the real LIMIT remains on Binance. The
        user-data stream can still turn it into FILLED while REST is unavailable.
        """
        if trade.status != "PENDING":
            return
        if trade.entry_inflight:
            log.info("[ENTRY] %s sedang dieksekusi; pembatalan %s ditolak.", trade.pair, reason)
            return
        if trade.real_enabled:
            trade.pending_cancel_reason = reason
            await self._cancel_real_entry_remainder(trade, reason)
            return
        await self._finalize_trade(
            trade, "EXPIRED" if "EXPIRE" in reason else "CANCELED", None, reason
        )

    async def _close_trade(
        self, trade: Trade, reference_price: Decimal, reason: str, *,
        keep_protections_until_flat: bool = False, client_order_id: str | None = None,
    ) -> None:
        if trade.status != "FILLED":
            return
        exit_price = reference_price
        market_order_submitted = False
        position_was_open = False
        if trade.real_enabled:
            try:
                # Manual closes retain the legacy cancel-first behavior. EMA exits
                # deliberately keep SL/TP armed until the MARKET position is flat.
                if not keep_protections_until_flat:
                    if not await self._cancel_real_protections(trade):
                        raise RuntimeError(
                            trade.protection_error or "Proteksi Binance belum terkonfirmasi dibatalkan; MARKET close ditahan."
                        )
                # A partially-filled GTC entry may still have an unfilled
                # remainder. Cancel and confirm that remainder before MARKET
                # close so it cannot fill later and unexpectedly reopen exposure.
                if not trade.entry_order_terminal:
                    await self._cancel_real_entry_remainder(
                        trade, "CLOSE_POSITION_CANCEL_ENTRY_REMAINDER"
                    )
                position = await self.real.get_position(trade.pair, trade.entry_side)
                position_was_open = position is not None
                if position is not None:
                    if keep_protections_until_flat:
                        trade.ema_exit_state = "CLOSE_SUBMITTING"
                        await self._save_active_state()
                    response = await self.real.place_market_close(
                        symbol=trade.pair, position=position, entry_direction=trade.entry_side,
                        new_client_order_id=(client_order_id if keep_protections_until_flat else None),
                    )
                    market_order_submitted = True
                    avg = Decimal(str(response.get("avgPrice") or "0"))
                    if avg > 0:
                        exit_price = avg
                    if keep_protections_until_flat:
                        trade.ema_exit_order_id = (int(response.get("orderId")) if response.get("orderId") not in (None, "") else trade.ema_exit_order_id)
                        trade.ema_exit_order_status = str(response.get("status") or "UNKNOWN").upper()
                        trade.ema_exit_state = "CLOSE_VERIFY"
                        await self._save_active_state()
                        # In Hedge Mode an opposite-side position on this symbol
                        # is unrelated; verify only the side being closed.
                        remaining = await self.real.get_position(trade.pair, trade.entry_side)
                        if remaining is not None:
                            raise RuntimeError(
                                f"EMA MARKET close {trade.pair} {trade.direction} belum flat; "
                                "Algo SL/TP tetap aktif dan close akan direkonsiliasi tanpa POST ulang."
                            )
                    else:
                        # A successful REST response is not enough: verify that no
                        # exchange exposure remains before recording close/reversal.
                        remaining = await self.real.get_positions(trade.pair)
                        if remaining:
                            raise RuntimeError(
                                f"Market close {trade.pair} belum flat; reversal/ledger close diblokir."
                            )
                else:
                    if not keep_protections_until_flat:
                        remaining = await self.real.get_positions(trade.pair)
                        if remaining:
                            raise RuntimeError(
                                "Posisi Binance tidak cocok dengan arah trade; reversal diblokir."
                            )
                    log.warning("[REAL] %s posisi %s tidak ditemukan; kemungkinan ditutup oleh SL/TP atau manual.", trade.pair, trade.direction)
                if keep_protections_until_flat:
                    if not await self._cancel_real_protections(trade):
                        if trade.protection_exit_reason is None:
                            trade.protection_exit_reason = (
                                reason if market_order_submitted else
                                ("TP_SL_EXIT_RECONCILED" if trade.protection_state == "ACTIVE" else "EXCHANGE_POSITION_FLAT_RECONCILED")
                            )
                        trade.ema_exit_state = "CLEANUP_WAIT_API"
                        await self._save_active_state()
                        return
            except Exception as exc:
                trade.real_error = str(exc)[:500]
                await self._save_active_state()
                raise
        if trade.protection_exit_reason in {"STOP_LOSS", "TAKE_PROFIT"}:
            final_reason = trade.protection_exit_reason
        elif keep_protections_until_flat and not market_order_submitted and not position_was_open:
            final_reason = "TP_SL_EXIT_RECONCILED" if trade.protection_state == "ACTIVE" else "EXCHANGE_POSITION_FLAT_RECONCILED"
        else:
            final_reason = reason
        await self._finalize_trade(trade, "CLOSED", exit_price, final_reason)

    async def _finalize_trade(self, trade: Trade, result: str, exit_price: Decimal | None, reason: str) -> None:
        # Idempotent terminal transition: WS flat, Algo exit reconciliation, and
        # REST position reconciliation may all report the same close. The first
        # caller marks the object terminal before its first await; later callers exit.
        if trade.status not in {"PENDING", "FILLED"} or any(
            str(row.get("trade_id") or "") == trade.trade_id
            and str(row.get("result") or "").upper() == "CLOSED"
            for row in self.history_records[-1000:]
            if isinstance(row, dict)
        ):
            log.debug("[TRADE] duplicate terminal event ignored | id=%s status=%s reason=%s", trade.trade_id, trade.status, reason)
            return
        trade.status = "CLOSED" if result == "CLOSED" else result
        trade.closed_at = now_utc()
        trade.result = result
        trade.result_reason = reason
        if result == "CLOSED" and exit_price is not None and trade.fill_price is not None:
            trade.exit_price = exit_price
            move = exit_price - trade.fill_price if trade.direction == "LONG" else trade.fill_price - exit_price
            trade.pnl_percent = (move / trade.fill_price * Decimal("100")) if trade.fill_price > 0 else Decimal("0")
            trade.pnl_usdt = move * (trade.quantity or Decimal("0"))
            if trade.initial_risk is not None and trade.initial_risk > 0:
                trade.realized_r = move / trade.initial_risk
            if trade.real_enabled and self._real_equity_baseline is not None:
                # Book a REAL close exactly once against the /real on baseline.
                self._realized_since_baseline += trade.pnl_usdt
        self.active_trades.pop(trade.trade_id, None)
        self._protection_locks.pop(trade.trade_id, None)
        self._entry_attempts.pop(trade.trade_id, None)
        self._entry_retry_after.pop(trade.trade_id, None)
        # The local/exchange ledger must converge immediately after a confirmed
        # terminal order or confirmed flat position; otherwise stale cache rows
        # would reserve /max slots or reappear in /trade.
        self._remove_exchange_order_cache(trade)
        if trade.real_enabled and result == "CLOSED":
            # Remove only the closed direction. In Hedge Mode the opposite side
            # of the same symbol may still be a valid live position.
            for position_key, position_row in list(self.exchange_positions.items()):
                if position_key[0] != trade.pair:
                    continue
                amount = position_row.get("_position_amount_decimal")
                direction = self._exchange_position_direction(position_row, amount if isinstance(amount, Decimal) else None)
                if direction == trade.direction:
                    self.exchange_positions.pop(position_key, None)
            self._refresh_real_position_cache(trade.pair)
        record = trade.to_record()
        self.history_records.append(record)
        await self._record_event("POSITION_CLOSED" if result == "CLOSED" else f"ORDER_{result}", trade, reason=reason, exit_price=decimal_to_str(exit_price))
        await self._persist_markdown_trade(record)
        await self._save_active_state()
        if trade.real_enabled and self._real_equity_baseline is not None:
            await self._save_real_guard_checkpoint()
        await self._sync_ws_symbols()
        log.info("[TRADE] %s %s %s reason=%s pnl=%s", result, trade.pair, trade.direction, reason, decimal_to_str(trade.pnl_usdt))
        if result == "CLOSED":
            await self.reply(
                f"✅ POSITION CLOSED • {trade.pair} {trade.direction}\n"
                f"TF: {trade.timeframe}\nEntry: {fmt_price(trade.fill_price)}\nExit: {fmt_price(trade.exit_price)}\n"
                f"PnL: {fmt_num(trade.pnl_percent, 2)}% | {fmt_num(trade.pnl_usdt, 4)} USDT\n"
                f"R: {fmt_num(trade.realized_r, 3)}\nReason: {reason}"
            )

    async def _persist_markdown_trade(self, record: dict[str, Any]) -> None:
        lines = [
            f"## {record.get('pair')} {record.get('direction')} — {record.get('result')}",
            f"- Trade ID: `{record.get('trade_id')}`",
            f"- Timeframe: {record.get('timeframe')} | Signal: {record.get('signal')}",
            f"- Entry: {record.get('fill_price')} | Exit: {record.get('exit_price')}",
            f"- PnL: {record.get('pnl_usdt')} USDT ({record.get('pnl_percent')}%) | R: {record.get('realized_r')}",
            f"- Closed: {record.get('closed_at')} | Reason: {record.get('result_reason')}",
            "",
        ]
        addition = "\n".join(lines)
        try:
            await self.github.append_text_file(self.session_history_markdown_path, addition, "volumatic: session trade closed")
            local = self._local_path(self.session_history_markdown_path)
            old = local.read_text(encoding="utf-8") if local.exists() else "# Volumatic Trend Trade Journal\n\n"
            local.write_text(old + addition, encoding="utf-8")
        except Exception as exc:
            log.error("Markdown journal GitHub gagal; detail tetap ada di session trades.json: %s", exc)

    async def _expire_due_pending(self) -> None:
        now = now_utc()
        for trade in list(self.active_trades.values()):
            if trade.entry_inflight:
                continue
            if trade.real_enabled and trade.status == "PENDING" and trade.real_state == "REAL_PREPARED" and trade.entry_order_created_at is None:
                # No Binance order exists yet. The one-hour timer has not started,
                # but the setup may only CREATE its LIMIT during the next M15 candle.
                if trade.entry_window_closes_at and now >= trade.entry_window_closes_at:
                    await self._finalize_trade(trade, "EXPIRED", None, "ENTRY_WINDOW_CLOSED_BEFORE_SUBMIT")
                continue
            if trade.real_enabled and trade.preserve_gtc_after_cooldown:
                # Never touch a live GTC while REST is rate-limited. Once REST
                # recovers, resume normal one-hour expiry handling for bot-tagged
                # orders, including those adopted from the exchange snapshot.
                if self._binance_cooldown_remaining() > 0:
                    continue
                trade.preserve_gtc_after_cooldown = False
                await self._save_active_state()
                log.info("[REAL] %s GTC REST protection cleared; one-hour expiry handling resumed", trade.pair)

            if trade.status == "PENDING" and now >= trade.expires_at:
                if trade.real_enabled:
                    cooldown_active = self._binance_cooldown_remaining() > 0
                    # A PREPARED order has never been submitted to Binance. Never
                    # turn an expired local intent into a late exchange order.
                    if trade.real_state == "REAL_PREPARED" or not trade.entry_client_order_id:
                        if cooldown_active:
                            continue
                        await self._finalize_trade(
                            trade, "EXPIRED", None, "REAL_ORDER_NEVER_SUBMITTED"
                        )
                        continue
                    order_may_exist = bool(trade.entry_client_order_id)
                    if order_may_exist and cooldown_active:
                        trade.preserve_gtc_after_cooldown = True
                        trade.pending_cancel_reason = "ENTRY_LIMIT_TIMEOUT_1H"
                        trade.real_state = "REAL_GTC_PRESERVED_AFTER_COOLDOWN"
                        trade.real_error = (
                            "Order GTC dibiarkan tanpa perubahan selama Binance REST cooldown. "
                            "Pembatalan karena timeout satu jam akan dilanjutkan begitu REST pulih."
                        )
                        await self._record_event(
                            "ORDER_GTC_PRESERVED_AFTER_API_COOLDOWN", trade,
                            order_status=trade.entry_order_status,
                            expires_at=iso_utc(trade.expires_at), expiry_seconds=ENTRY_ORDER_TTL_SECONDS,
                            ban_until_epoch=BINANCE_COOLDOWN._until_epoch,
                        )
                        await self._save_active_state()
                        log.info("[REAL] GTC preserved | pair=%s | 1h timeout during REST cooldown; order remains on Binance until fill/reversal/cancel", trade.pair)
                        continue
                await self._cancel_pending_trade(
                    trade, "ENTRY_WATCH_EXPIRED_1H" if trade.real_intent else "ENTRY_LIMIT_TIMEOUT_1H"
                )

            elif trade.status == "FILLED" and trade.real_enabled and not trade.entry_order_terminal and now >= trade.expires_at:
                cooldown_active = self._binance_cooldown_remaining() > 0
                if cooldown_active:
                    trade.preserve_gtc_after_cooldown = True
                    trade.pending_cancel_reason = "ENTRY_LIMIT_TIMEOUT_1H"
                    trade.real_state = "REAL_PARTIAL_GTC_PRESERVED_AFTER_COOLDOWN"
                    trade.real_error = (
                        "Sisa quantity LIMIT GTC dipertahankan setelah partial fill karena batas satu jam "
                        "terjadi selama Binance REST cooldown."
                    )
                    await self._record_event(
                        "PARTIAL_ENTRY_REMAINDER_GTC_PRESERVED_AFTER_API_COOLDOWN", trade,
                        order_status=trade.entry_order_status,
                        executed_qty=decimal_to_str(trade.entry_executed_qty),
                    )
                    await self._save_active_state()
                else:
                    await self._cancel_real_entry_remainder(trade, "ENTRY_LIMIT_TIMEOUT_1H")

    async def _reconcile_pending_real(self, trade: Trade) -> None:
        if not trade.real_enabled or trade.status != "PENDING" or trade.entry_inflight:
            return
        if self._binance_cooldown_remaining() > 0:
            return  # open Binance LIMIT remains untouched; private WS still reports fills.
        if trade.real_state == "REAL_PREPARED":
            # Legacy LIMIT intent that never reached Binance. LIMIT entries no
            # longer exist, so it is discarded instead of being submitted.
            await self._finalize_trade(trade, "EXPIRED", None, "LEGACY_LIMIT_INTENT_DISCARDED")
            return
        if time.monotonic() - trade.last_real_check_monotonic < REAL_PENDING_POLL_SECONDS:
            return
        trade.last_real_check_monotonic = time.monotonic()
        if not (trade.entry_order_id or trade.entry_client_order_id):
            trade.real_error = "REAL order tidak memiliki order ID/client ID."
            trade.real_state = "REAL_CANCEL_UNKNOWN"
            await self._save_active_state()
            return
        try:
            order = await self.real.get_order(
                trade.pair, order_id=trade.entry_order_id,
                client_order_id=trade.entry_client_order_id,
            )
        except BinanceRateLimitError:
            # A shared IP cooldown is not an order failure. Leave the Binance
            # LIMIT untouched and let the user-data stream keep reporting fills.
            raise
        except BinanceAPIError as exc:
            # Crash recovery for the narrow window after REAL_SUBMITTING was
            # persisted but before the POST reached Binance. -2013 means the
            # stable clientOrderId is not present. Reuse that same ID (never mint
            # a second one); if a first POST actually exists, Binance duplicate
            # detection plus the subsequent lookup prevents a second exposure.
            if exc.code == -2013 and trade.real_submit_ambiguous:
                # A missing lookup immediately after a transport/5xx ambiguity is
                # not proof that the POST never arrived. Keep the slot UNKNOWN and
                # retry only the GET lookup on later maintenance passes.
                trade.real_state = "REAL_SUBMIT_UNKNOWN"
                trade.real_error = "ACK submit belum ditemukan pada lookup ini; status UNKNOWN tetap ditahan dan POST tidak diulang."
                await self._save_active_state()
                log.info("[REAL/RECON] %s clientOrderId belum terlihat; lanjut lookup GET berkala tanpa POST ulang.", trade.pair)
                return
            if exc.code == -2013 and trade.preserve_gtc_after_cooldown:
                # This path is for an order that was previously acknowledged by
                # Binance. Do not create a new order after its preserved GTC vanishes.
                trade.entry_order_terminal = True
                trade.preserve_gtc_after_cooldown = False
                trade.real_state = "REAL_ORDER_NOT_FOUND"
                trade.real_error = "Order GTC yang sebelumnya dikenal tidak ditemukan; posisi akan diverifikasi terpisah."
                await self._save_active_state()
                if trade.status == "FILLED":
                    await self._reconcile_filled_real(trade)
                else:
                    await self._finalize_trade(trade, "EXPIRED", None, "PRESERVED_GTC_NOT_FOUND_AFTER_COOLDOWN")
                return
            if (now_utc() >= trade.expires_at and not trade.preserve_gtc_after_cooldown) or trade.real_state == "REAL_CANCEL_UNKNOWN":
                reason = "ENTRY_LIMIT_TIMEOUT_1H" if now_utc() >= trade.expires_at else "RETRY_CANCEL_UNKNOWN"
                await self._cancel_pending_trade(trade, reason)
                return
            raise
        except Exception:
            if (now_utc() >= trade.expires_at and not trade.preserve_gtc_after_cooldown) or trade.real_state == "REAL_CANCEL_UNKNOWN":
                reason = "ENTRY_LIMIT_TIMEOUT_1H" if now_utc() >= trade.expires_at else "RETRY_CANCEL_UNKNOWN"
                await self._cancel_pending_trade(trade, reason)
                return
            raise
        status = str(order.get("status") or "UNKNOWN").upper()
        trade.entry_order_status = status
        executed = Decimal(str(order.get("executedQty") or "0"))
        trade.entry_executed_qty = executed
        if status == "FILLED":
            trade.entry_order_terminal = True
            if executed > 0:
                trade.quantity = executed
            await self._mark_filled(
                trade, self._order_fill_price(order, trade.entry), "BINANCE_POLL",
                filled_at=self._exchange_event_time(order),
            )
            trade.real_state = "REAL_FILLED"
            await self._save_active_state()
        elif status == "PARTIALLY_FILLED":
            # The filled amount is live exposure. Normally cancel the remaining
            # LIMIT at one-hour entry timeout/reversal; if that expiry occurred during
            # a REST IP ban, preserve the GTC remainder until fill/reversal/manual cancel.
            trade.entry_order_terminal = False
            if executed > 0:
                trade.quantity = executed
                await self._mark_filled(
                    trade, self._order_fill_price(order, trade.entry), "BINANCE_POLL_PARTIAL",
                    filled_at=self._exchange_event_time(order),
                )
            trade.real_state = "REAL_PARTIALLY_FILLED"
            await self._save_active_state()
        elif status in {"NEW", "UNKNOWN"} and (
            (now_utc() >= trade.expires_at and not trade.preserve_gtc_after_cooldown)
            or trade.real_state == "REAL_CANCEL_UNKNOWN"
        ):
            # Retry a failed manual/expiry cancellation until Binance confirms
            # a terminal order state. An ambiguous order keeps its capacity slot.
            reason = "ENTRY_LIMIT_TIMEOUT_1H" if now_utc() >= trade.expires_at else "RETRY_CANCEL_UNKNOWN"
            await self._cancel_pending_trade(trade, reason)
        elif status in {"CANCELED", "EXPIRED", "REJECTED"}:
            executed = Decimal(str(order.get("executedQty") or "0"))
            if executed > 0:
                trade.quantity = executed
                await self._mark_filled(
                    trade, self._order_fill_price(order, trade.entry), "BINANCE_TERMINAL_PARTIAL_FILL",
                    filled_at=self._exchange_event_time(order),
                )
            else:
                await self._finalize_trade(
                    trade, "EXPIRED" if status == "EXPIRED" else "CANCELED",
                    None, f"BINANCE_{status}",
                )
        elif now_utc() >= trade.expires_at and not trade.preserve_gtc_after_cooldown:
            await self._cancel_pending_trade(trade, "ENTRY_LIMIT_TIMEOUT_1H")

    async def _sync_real_entry_remainder(self, trade: Trade) -> None:
        """Refresh cumulative fill/status for a still-open entry LIMIT.

        This is important when User Data WS disconnected during an IP ban: after
        REST recovers, reconciliation can discover additional fills or that the
        remaining GTC order became terminal without inventing/canceling exposure.
        """
        if trade.entry_order_terminal or not (trade.entry_order_id or trade.entry_client_order_id):
            return
        try:
            order = await self.real.get_order(
                trade.pair, order_id=trade.entry_order_id,
                client_order_id=trade.entry_client_order_id,
            )
        except BinanceAPIError as exc:
            if exc.code == -2013 and trade.real_submit_ambiguous:
                trade.real_state = "REAL_SUBMIT_UNKNOWN"
                trade.real_error = "Order submit ambigu belum ditemukan; tetap UNKNOWN dan tidak dibuat ulang."
                await self._save_active_state()
                return
            if exc.code == -2013 and trade.preserve_gtc_after_cooldown:
                trade.entry_order_terminal = True
                trade.preserve_gtc_after_cooldown = False
                trade.real_error = "GTC yang dipertahankan tidak ditemukan lagi di Binance; lanjut verifikasi posisi."
                await self._record_event("PRESERVED_GTC_ORDER_NOT_FOUND", trade, error=str(exc)[:300])
                await self._save_active_state()
                return
            raise

        status = str(order.get("status") or "UNKNOWN").upper()
        executed = Decimal(str(order.get("executedQty") or "0"))
        previous_executed = trade.entry_executed_qty or Decimal("0")
        trade.entry_order_status = status
        trade.entry_executed_qty = executed
        if order.get("orderId") is not None:
            trade.entry_order_id = int(order["orderId"])
        terminal = status in {"FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}
        trade.entry_order_terminal = terminal

        if executed > 0:
            avg = self._order_fill_price(order, trade.fill_price or trade.entry)
            trade.quantity = executed
            # A REST reconciliation can discover additional GTC fills while the
            # private stream was disconnected. Route every cumulative-size/avg-price
            # change through _mark_filled so the fixed Supertrend SL remains intact
            # and the TP is revalidated against the new cumulative average using the trade's saved RR band.
            if trade.status != "FILLED" or executed != previous_executed or avg != (trade.fill_price or trade.entry):
                await self._mark_filled(
                    trade, avg, "BINANCE_REST_CUMULATIVE_FILL_RECONCILIATION",
                    filled_at=self._exchange_event_time(order),
                )
            else:
                trade.fill_price = avg
                trade.actual_notional = executed * avg
                trade.margin_usdt = trade.actual_notional / Decimal(max(1, trade.leverage))
            if executed != previous_executed or status != "PARTIALLY_FILLED":
                await self._record_event(
                    "ENTRY_REMAINDER_RECONCILED", trade,
                    order_status=status,
                    previous_executed_qty=decimal_to_str(previous_executed),
                    executed_qty=decimal_to_str(executed),
                    average_price=decimal_to_str(avg),
                )
        elif terminal and trade.status == "PENDING":
            await self._finalize_trade(
                trade, "EXPIRED" if status in {"EXPIRED", "EXPIRED_IN_MATCH"} else "CANCELED",
                None, f"BINANCE_ENTRY_{status}_WITHOUT_FILL",
            )
            return

        if terminal:
            trade.pending_cancel_reason = None
            if status == "FILLED":
                trade.real_state = "REAL_FILLED"
            else:
                trade.real_state = "REAL_PARTIAL_TERMINAL" if executed > 0 else f"REAL_{status}"
        else:
            trade.real_state = "REAL_PARTIALLY_FILLED" if executed > 0 else "REAL_PENDING"
        await self._save_active_state()

    async def _reconcile_filled_real(self, trade: Trade) -> None:
        if not trade.real_enabled or trade.status != "FILLED":
            return
        if self._binance_cooldown_remaining() > 0:
            return
        if trade.ema_exit_state in {"EXIT_PENDING", "CLOSE_SUBMITTING", "CLOSE_VERIFY", "CLOSE_UNKNOWN"}:
            await self._process_ema_exit_real(trade)
            if trade.status != "FILLED" or trade.ema_exit_state in {"EXIT_PENDING", "CLOSE_SUBMITTING", "CLOSE_VERIFY", "CLOSE_UNKNOWN", "CLEANUP_WAIT_API"}:
                return
        if trade.protection_state == "CANCEL_WAIT_API":
            if trade.ema_exit_state == "CLEANUP_WAIT_API":
                now_mono = time.monotonic()
                if now_mono - self._ema_exit_close_last_check.get(trade.trade_id, 0.0) < EMA_EXIT_RECONCILE_SECONDS:
                    return
                self._ema_exit_close_last_check[trade.trade_id] = now_mono
            remaining_now = await self.real.get_position(trade.pair, trade.entry_side)
            if remaining_now is None:
                keep = "SL" if trade.protection_exit_reason == "STOP_LOSS" else "TP" if trade.protection_exit_reason == "TAKE_PROFIT" else None
                if await self._cancel_real_protections(trade, keep_triggered=keep):
                    price = self._current_price(trade.pair) or trade.fill_price or trade.entry
                    reason = trade.protection_exit_reason or "EXCHANGE_POSITION_FLAT_WS"
                    await self._finalize_trade(trade, "CLOSED", price, reason)
                return
            # Position still exists. Rebuild any protection that had to be canceled
            # for a manual close that did not reach the exchange.
            trade.protection_state = "PARTIAL"
            await self._save_active_state()
        if trade.pending_cancel_reason == "PROTECTION_EXIT_CANCEL_ENTRY_REMAINDER" and not trade.entry_order_terminal:
            await self._cancel_real_entry_remainder(trade, "PROTECTION_EXIT_CANCEL_ENTRY_REMAINDER")
            if trade.status != "FILLED":
                return
        if (not trade.entry_order_terminal and trade.expires_at <= now_utc()
                and not trade.preserve_gtc_after_cooldown):
            await self._cancel_real_entry_remainder(trade, trade.pending_cancel_reason or "ENTRY_LIMIT_TIMEOUT_1H")
            if trade.status != "FILLED":
                return
        ws_covered = self.user_ws.status == "LIVE" and trade.protection_state == "ACTIVE" and trade.entry_order_terminal
        reconcile_interval = REAL_RECONCILE_WS_COVERED_SECONDS if ws_covered else REAL_RECONCILE_INTERVAL_SECONDS
        if time.monotonic() - trade.last_real_check_monotonic < reconcile_interval:
            return
        trade.last_real_check_monotonic = time.monotonic()

        # A late/partial GTC preserved during a REST ban is not canceled while
        # the ban is active. The maintenance loop queues/resumes the timeout cancel
        # immediately after REST recovers.
        if trade.preserve_gtc_after_cooldown and not trade.entry_order_terminal:
            if time.monotonic() - trade.last_real_order_status_check >= REAL_PENDING_POLL_SECONDS:
                trade.last_real_order_status_check = time.monotonic()
                await self._sync_real_entry_remainder(trade)
                if trade.status != "FILLED":
                    return
                if not trade.entry_order_terminal and trade.entry_order_status in {"NEW", "PARTIALLY_FILLED", "UNKNOWN"}:
                    # It is intentionally still a live GTC. A missing position
                    # snapshot must not cause a cancel that violates preservation.
                    position = await self.real.get_position(trade.pair, trade.entry_side)
                    if position is None:
                        return

        position = await self.real.get_position(trade.pair, trade.entry_side)
        if position is None:
            # Give the exchange a brief settlement window after a WS fill before
            # treating a temporarily invisible position as externally closed.
            if trade.filled_at and (now_utc() - trade.filled_at).total_seconds() < REAL_POSITION_SETTLE_SECONDS:
                return
            if not trade.entry_order_terminal:
                if trade.preserve_gtc_after_cooldown and trade.pending_cancel_reason != "ENTRY_LIMIT_TIMEOUT_1H":
                    # A preserved/adopted live GTC is still allowed to fill; do not
                    # remove it only because the position endpoint is momentarily flat.
                    return
                await self._cancel_real_entry_remainder(
                    trade, "EXTERNAL_POSITION_MISSING_CANCEL_ENTRY_REMAINDER"
                )
            # get_position() is direction-aware (LONG/SHORT or signed BOTH). An
            # opposite-side position on the same symbol must not keep this ledger alive.
            price = self._current_price(trade.pair) or trade.fill_price or trade.entry
            log.warning("[REAL] %s %s exposure hilang di Binance; menutup ledger untuk arah ini.", trade.pair, trade.direction)
            reason = trade.protection_exit_reason or ("TP_SL_EXIT_RECONCILED" if trade.protection_state == "ACTIVE" else "EXTERNAL_CLOSE_RECONCILED")
            if not await self._cancel_real_protections(trade, keep_triggered=("SL" if reason == "STOP_LOSS" else "TP" if reason == "TAKE_PROFIT" else None)):
                trade.protection_exit_reason = reason
                return
            await self._finalize_trade(trade, "CLOSED", price, reason)
            return

        # The exact side is confirmed open. Only now may we install protection or
        # warn that the confirmed live position has an invalid Supertrend risk level.
        if trade.protection_state == "INVALID_RISK":
            if not trade.protection_warning_sent:
                trade.protection_warning_sent = True
                await self._save_active_state()
                await self.reply(
                    f"🚨 PROTEKSI BELUM BISA DIPASANG • {trade.pair} {trade.direction}\n"
                    f"Fill: {fmt_price(trade.fill_price)} • Supertrend SL: {fmt_price(trade.reference_sl)}\n"
                    "Exposure masih aktif di Binance, tetapi SL tidak membentuk jarak risiko valid. Amankan posisi LIVE segera."
                )
        elif trade.protection_state not in {"ACTIVE", "ERROR_REJECTED"}:
            try:
                # Give the first exchange SL/TP installation attempt priority.
                # EMA history warm-up is scheduled in finally, never ahead of it.
                await self._ensure_real_protection(trade)
            finally:
                if (
                    trade.status == "FILLED" and not trade.exchange_adopted
                    and trade.ema_exit_state in {"UNINITIALIZED", "WAITING_API", "INIT_RETRY", "BACKFILL_PENDING"}
                ):
                    self._schedule_ema_exit_initialization(trade)
                    await self._sync_ws_symbols()

        if (
            trade.status == "FILLED" and not trade.exchange_adopted
            and trade.protection_state in {"ACTIVE", "PARTIAL", "INVALID_RISK", "ERROR_REJECTED"}
            and trade.ema_exit_state in {"UNINITIALIZED", "WAITING_API", "INIT_RETRY", "BACKFILL_PENDING"}
        ):
            self._schedule_ema_exit_initialization(trade)
            await self._sync_ws_symbols()

    async def _reconcile_restored_trades(self) -> None:
        if self._binance_cooldown_remaining() > 0:
            log.info("[RECOVERY] REST reconciliation ditunda selama Binance cooldown; local state dan User WS tetap aktif.")
            return
        for trade in list(self.active_trades.values()):
            if trade.real_enabled:
                try:
                    if trade.status == "PENDING":
                        if trade.real_state == "REAL_SUBMITTING":
                            trade.real_submit_ambiguous = True
                            trade.real_state = "REAL_SUBMIT_UNKNOWN"
                            trade.real_error = "Proses berhenti saat submit; status dipulihkan sebagai UNKNOWN dan hanya lookup clientOrderId."
                            await self._save_active_state()
                        if trade.real_state == "REAL_PREPARED":
                            await self._finalize_trade(trade, "EXPIRED", None, "LEGACY_LIMIT_INTENT_DISCARDED")
                            continue
                        try:
                            order = await self.real.get_order(
                                trade.pair, order_id=trade.entry_order_id,
                                client_order_id=trade.entry_client_order_id,
                            )
                        except BinanceAPIError as exc:
                            if exc.code == -2013 and (
                                trade.real_submit_ambiguous
                                or trade.real_state in {"REAL_SUBMITTING", "REAL_SUBMIT_UNKNOWN"}
                            ):
                                # A process may have died between POST and response.
                                # Never infer "not submitted" from one missing lookup;
                                # never POST again with a possibly-live clientOrderId.
                                trade.real_submit_ambiguous = True
                                trade.real_state = "REAL_SUBMIT_UNKNOWN"
                                trade.real_error = (
                                    "Startup lookup belum menemukan clientOrderId; POST tidak diulang. "
                                    "Order harus direkonsiliasi berkala atau diverifikasi manual di Binance."
                                )
                                await self._save_active_state()
                                log.info(
                                    "[REAL] startup submit remains UNKNOWN | pair=%s | clientOrderId=%s | lookup-only",
                                    trade.pair, trade.entry_client_order_id,
                                )
                                continue
                            if exc.code == -2013 and trade.preserve_gtc_after_cooldown:
                                if trade.status == "FILLED":
                                    trade.entry_order_terminal = True
                                    trade.preserve_gtc_after_cooldown = False
                                    trade.real_state = "REAL_ORDER_NOT_FOUND"
                                    trade.real_error = "GTC tidak ditemukan; posisi terisi akan diverifikasi terpisah."
                                    await self._save_active_state()
                                else:
                                    await self._finalize_trade(
                                        trade, "EXPIRED", None,
                                        "STARTUP_PRESERVED_GTC_NOT_FOUND_AFTER_COOLDOWN",
                                    )
                                continue
                            raise
                        status = str(order.get("status") or "UNKNOWN").upper()
                        executed = Decimal(str(order.get("executedQty") or "0"))
                        trade.entry_order_status = status
                        trade.entry_executed_qty = executed
                        if order.get("orderId") is not None:
                            trade.entry_order_id = int(order["orderId"])
                        if status in {"FILLED", "PARTIALLY_FILLED"} and executed > 0:
                            trade.real_submit_ambiguous = False
                            trade.entry_order_terminal = status == "FILLED"
                            await self._mark_filled(
                                trade, self._order_fill_price(order, trade.entry),
                                "STARTUP_RECONCILIATION" if status == "FILLED" else "STARTUP_PARTIAL_FILL_RECONCILIATION",
                                filled_at=self._exchange_event_time(order),
                            )
                            trade.real_state = "REAL_FILLED" if status == "FILLED" else "REAL_PARTIALLY_FILLED"
                            await self._save_active_state()
                        elif status == "NEW":
                            trade.real_submit_ambiguous = False
                            trade.entry_order_terminal = False
                            trade.real_state = "REAL_PENDING"
                            trade.real_error = None
                            await self._save_active_state()
                        elif status in {"CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}:
                            trade.real_submit_ambiguous = False
                            if executed > 0:
                                trade.entry_order_terminal = True
                                await self._mark_filled(
                                    trade, self._order_fill_price(order, trade.entry),
                                    "STARTUP_TERMINAL_PARTIAL_FILL_RECONCILIATION",
                                    filled_at=self._exchange_event_time(order),
                                )
                            else:
                                await self._finalize_trade(
                                    trade, "EXPIRED" if status.startswith("EXPIRED") else "CANCELED",
                                    None, f"STARTUP_{status}",
                                )
                        else:
                            # Unknown exchange state is retained; do not free max slot.
                            trade.real_submit_ambiguous = True
                            trade.real_state = "REAL_SUBMIT_UNKNOWN"
                            trade.real_error = f"Startup order status belum dikenal: {status}"
                            await self._save_active_state()
                    elif trade.status == "FILLED":
                        position = await self.real.get_position(trade.pair, trade.entry_side)
                        if position is None:
                            remaining = await self.real.get_positions(trade.pair)
                            if not remaining:
                                trade.real_error = "Posisi ledger tidak ditemukan di Binance; cek manual sebelum trading pair ini."
                                log.error("[RECOVERY] %s: %s", trade.pair, trade.real_error)
                except BinanceRateLimitError as exc:
                    trade.real_error = str(exc)[:500]
                    self._was_in_binance_cooldown = True
                    await self._notify_binance_cooldown(exc, source="STARTUP RECONCILIATION", symbol=trade.pair)
                    log.info("[RECOVERY] reconciliation dijeda oleh rate governor | pair=%s", trade.pair)
                    break
                except Exception as exc:
                    trade.real_error = str(exc)[:500]
                    log.exception("[RECOVERY] reconcile %s gagal.", trade.pair)
        await self._save_active_state()

    async def _maintenance_loop(self) -> None:
        """Serialized recovery loop: never poll REST during an IP cooldown."""
        while self._running:
            try:
                PROXY_MANAGER.tick()  # non-blocking; worker thread does fetch/validate/switch
                # Local-only guard runs even while Binance REST is rate-limited.
                await self._autostop_check()
                cooldown = self._binance_cooldown_remaining()
                if cooldown > 0:
                    self._was_in_binance_cooldown = True
                    self._rest_recovery_active = True
                    await self._sync_cooldown_marker_to_github()
                    # Local PAPER expiration still happens. REAL orders are left
                    # untouched on Binance; WS events continue updating the ledger.
                    await self._expire_due_pending()
                    await asyncio.sleep(min(10.0, max(2.0, cooldown)))
                    continue

                if self._was_in_binance_cooldown:
                    self._rest_recovery_active = True
                    self._scan_trigger_now = False
                    await self._sync_cooldown_marker_to_github()
                    # Retain warm symbol metadata after recovery. Do not wake the
                    # scanner here: maintenance must first reconcile live REAL
                    # trades and drain at most one deferred command per pass.
                    self._last_universe_refresh = (
                        time.monotonic() if self.symbols else 0.0
                    )
                    now_mono = time.monotonic()
                    if now_mono - self._last_api_recovery_notice >= BINANCE_RECOVERY_NOTICE_INTERVAL_SECONDS:
                        self._last_api_recovery_notice = now_mono
                        await self._notify_binance_recovered()
                    self._was_in_binance_cooldown = False
                    log.info("[BINANCE] REST restored; recovery queue started.")

                # Reconnect private stream after cooldown if it was unable to
                # obtain a listenKey at startup. If an existing listenKey is
                # valid, this starts/reuses the socket without another REST call.
                needs_private_stream = self.real_mode or any(
                    t.real_enabled and t.status in {"PENDING", "FILLED"}
                    for t in self.active_trades.values()
                )
                if needs_private_stream and self.user_ws.status == "OFFLINE":
                    if time.monotonic() - self._last_user_stream_start_attempt >= 15.0:
                        self._last_user_stream_start_attempt = time.monotonic()
                        try:
                            await self.user_ws.start()
                        except BinanceRateLimitError:
                            await asyncio.sleep(2.0)
                            continue
                        except Exception:
                            log.exception("[USER WS] start/reconnect ditunda")

                # Account Guard uses a one-time /real on wallet baseline, then local
                # realized PnL + live floating PnL. No periodic wallet-balance REST poll.

                # Commands are FIFO and only one API-dependent command per pass. /real off
                # remains immediate and can cancel a queued /real on.
                await self._process_deferred_commands()

                # Refresh universe only when scanner needs it. ExchangeInfo is
                # cached for one hour; no forced repeat on every recovery pass.
                if self.scan_enabled and (not self.symbols or time.monotonic() - self._last_universe_refresh > UNIVERSE_REFRESH_SECONDS):
                    if time.monotonic() - self._last_exchange_info_retry >= 30.0:
                        self._last_exchange_info_retry = time.monotonic()
                        try:
                            await self._universe()
                        except BinanceRateLimitError as exc:
                            self._was_in_binance_cooldown = True
                            await self._notify_binance_cooldown(exc, source="UNIVERSE REFRESH")
                            await asyncio.sleep(2.0)
                            continue
                        except Exception as exc:
                            log.warning("[UNIVERSE] refresh belum berhasil; retry bertahap: %s", exc)

                await self._expire_due_pending()
                if self._scan_capacity_paused and len(self._active_slot_projection()) < self.max_active_trades:
                    self._scan_capacity_paused = False
                    if self.scan_enabled:
                        self._scan_trigger_now = True
                    log.info("[SCAN] capacity slot released; scanner resumed.")
                for trade in list(self.active_trades.values()):
                    if self._binance_cooldown_remaining() > 0:
                        break
                    if trade.status == "FILLED" and not trade.real_enabled and not trade.exchange_adopted and trade.ema_exit_state in {"UNINITIALIZED", "WAITING_API", "INIT_RETRY", "BACKFILL_PENDING"}:
                        self._schedule_ema_exit_initialization(trade)
                    if trade.real_enabled and trade.status == "PENDING":
                        try:
                            await self._reconcile_pending_real(trade)
                        except BinanceRateLimitError as exc:
                            self._was_in_binance_cooldown = True
                            await self._notify_binance_cooldown(exc, source="ORDER RECONCILIATION", symbol=trade.pair)
                            break
                        except Exception as exc:
                            log.warning("[REAL] pending sync %s gagal: %s", trade.pair, exc)
                    elif trade.real_enabled and trade.status == "FILLED":
                        try:
                            await self._reconcile_filled_real(trade)
                        except BinanceRateLimitError as exc:
                            self._was_in_binance_cooldown = True
                            await self._notify_binance_cooldown(exc, source="POSITION RECONCILIATION", symbol=trade.pair)
                            break
                        except Exception as exc:
                            log.warning("[REAL] position sync %s gagal: %s", trade.pair, exc)

                if self._binance_cooldown_remaining() <= 0:
                    await self._retry_signals()
                    await self._monitor_active_positions()

                # Recovery has now had one complete serialized maintenance pass.
                # Release scanner only if no request in that pass re-triggered a ban.
                if self._rest_recovery_active and self._binance_cooldown_remaining() <= 0:
                    self._rest_recovery_active = False
                    if self.scan_enabled and not self._autostop_triggered:
                        self._scan_trigger_now = True
                    log.info("[BINANCE] post-cooldown reconciliation pass complete; scanner gate released.")
                await asyncio.sleep(MAINTENANCE_POLL_SECONDS)
            except asyncio.CancelledError:
                break
            except BinanceRateLimitError as exc:
                self._was_in_binance_cooldown = True
                await self._notify_binance_cooldown(exc, source="MAINTENANCE")
                log.info("[MAINTENANCE] REST maintenance paused by shared rate governor.")
                await asyncio.sleep(3.0)
            except Exception as exc:
                # Include the exception message in the Telegram-forwarded log
                # record so a future maintenance failure is diagnosable without
                # SSH access; the full traceback remains in Render logs.
                log.exception("[MAINTENANCE] loop error: %s", exc)
                await asyncio.sleep(3)

    def _kline_history_limit(
        self, trend_length: int | None = None, supertrend_atr_length: int | None = None
    ) -> int:
        """M15 history sized for Volumatic EMA-SWMA and configured Supertrend warm-up."""
        length = volumatic.normalize_trend_length(
            self.trend_length if trend_length is None else trend_length
        )
        atr_length = volumatic.normalize_supertrend_atr_length(
            self.supertrend_atr_length if supertrend_atr_length is None else supertrend_atr_length
        )
        needed = max(KLINE_HISTORY, atr_length + 20, length * 4 + 12)
        return min(1500, int(needed))

    def _h4_ema_history_limit(self, ema_config: Any = None) -> int:
        selected = volumatic.normalize_ema_filter_length(
            self.ema_filter_length if ema_config is None else ema_config
        )
        periods = max(selected) if isinstance(selected, tuple) else (selected or 1)
        return min(1500, max(100, int(periods) * 4 + 12))

    def _cache_live_signal_context(
        self, symbol: str, timeframe: str, candles: np.ndarray,
        *, trend_length: int | None = None, ema_filter_length: int | tuple[int, int] | None = None,
    ) -> None:
        """Precompute a small context once; WebSocket ticks then use scalar math."""
        key = f"{str(timeframe).upper()}:{normalize_symbol(symbol)}"
        if not isinstance(candles, np.ndarray) or candles.dtype.names is None or not candles.size:
            return
        selected_trend_length = volumatic.normalize_trend_length(
            self.trend_length if trend_length is None else trend_length
        )
        context = volumatic.prepare_live_visual_context(
            candles, symbol=symbol, timeframe=timeframe,
            trend_length=selected_trend_length,
            interval_ms=volumatic.TIMEFRAME_MS[str(timeframe).upper()],
            ema_filter_length=ema_filter_length,
        )
        if context is None:
            self._live_signal_candles.pop(key, None)
            self._live_signal_last_eval.pop(key, None)
            self._live_signal_seen.pop(key, None)
            return
        previous = self._live_signal_candles.get(key)
        same_live_candle = bool(
            previous
            and int(previous.get("entry_candle_open_time_ms") or 0)
                == int(context.get("entry_candle_open_time_ms") or 0)
            and previous.get("ema_filter_length") == context.get("ema_filter_length")
            and previous.get("trend_length") == context.get("trend_length")
        )
        # A repeated scanner refresh inside the same active candle must not
        # clear the per-candle bubble latch; otherwise a repaint can submit the
        # same direction more than once. The closed visual candle is immutable.
        if same_live_candle:
            return
        self._live_signal_candles[key] = context
        self._live_signal_last_eval[key] = 0.0
        self._live_signal_seen[key] = set()

    async def _monitor_active_positions(self) -> None:
        """Position protection/reconciliation has one owner: _reconcile_filled_real.

        Keeping a second protection installer here caused duplicate POST/alerts and
        could race with a WS close. This hook intentionally performs no REST work.
        """
        return

    async def _retry_signals(self) -> None:
        if not self._signal_retries or self._binance_cooldown_remaining() > 0:
            return
        now = now_utc()
        for key, item in list(self._signal_retries.items()):
            if int(time.time() * 1000) >= int(item["expiry_ms"]):
                try:
                    # Expired confirmed-M15 signals must never open a late LIMIT.
                    outcome = await self._handle_signal(item["signal"], int(item["expiry_ms"]))
                except BinanceRateLimitError as exc:
                    self._was_in_binance_cooldown = True
                    await self._notify_binance_cooldown(exc, source="EXPIRED M15 SIGNAL RETRY", symbol=item["signal"].get("symbol"))
                    break
                except Exception as exc:
                    item["attempts"] = int(item.get("attempts") or 0) + 1
                    item["last_error"] = str(exc)[:500]
                    log.info("[SIGNAL RETRY] stale opposite-close %s deferred: %s", key, exc)
                    continue
                self._signal_retries.pop(key, None)
                self.last_processed_candles[key] = int(item["signal"].get("entry_candle_open_time_ms") or item["signal"]["signal_close_time_ms"])
                await self._record_event(
                    "SIGNAL_RETRY_EXPIRED", None,
                    pair=item["signal"].get("symbol"),
                    direction=item["signal"].get("direction"),
                    timeframe=item["signal"].get("timeframe"),
                    attempts=int(item.get("attempts") or 0), outcome=outcome,
                )
                continue
            retry_at = parse_iso(item.get("next_retry_at")) if item.get("next_retry_at") else None
            if retry_at is not None and now < retry_at:
                continue
            try:
                outcome = await self._handle_signal(item["signal"], int(item["expiry_ms"]))
                self.last_processed_candles[key] = int(item["signal"]["signal_close_time_ms"])
                self._signal_retries.pop(key, None)
                await self._save_runtime_settings()
                log.info("[SIGNAL RETRY] %s selesai | outcome=%s", key, outcome)
            except BinanceRateLimitError as exc:
                self._was_in_binance_cooldown = True
                await self._notify_binance_cooldown(exc, source="SIGNAL RETRY", symbol=item["signal"].get("symbol"))
                break
            except Exception as exc:
                attempts = int(item.get("attempts") or 0) + 1
                delay = min(120, max(5, 5 * (2 ** min(attempts - 1, 5))))
                item["attempts"] = attempts
                item["next_retry_at"] = iso_utc(now_utc() + timedelta(seconds=delay))
                item["last_error"] = str(exc)[:500]
                log.info("[SIGNAL RETRY] %s ditunda %ss | attempt=%s | %s", key, delay, attempts, exc)
                await self._record_event(
                    "SIGNAL_RETRY_DEFERRED", None,
                    pair=item["signal"].get("symbol"),
                    direction=item["signal"].get("direction"),
                    timeframe=item["signal"].get("timeframe"),
                    attempts=attempts, retry_in_seconds=delay, error=str(exc)[:300],
                )

    # ----------------------------- WebSocket ------------------------------
    def _current_price(self, symbol: str) -> Decimal | None:
        snapshot = self.prices.get(symbol)
        return snapshot.price if snapshot and snapshot.live else None

    async def _evaluate_live_visual_bubble(self, symbol: str, price: Decimal, event_time_ms: int) -> None:
        """Evaluate mode-B bubbles on WS ticks for scanned symbols and positions.

        The scanner subscribes the selected USDT perpetual universe in one
        batched WS request. Contexts are precomputed once per pair/candle, so
        each tick only performs scalar EMA2 + trend-flip math.
        """
        active = [
            t for t in self.active_trades.values()
            if t.pair == symbol and t.status in {"PENDING", "FILLED"}
        ]
        scanner_watch = self.scan_enabled and symbol in self._scanner_symbols
        if not active and not scanner_watch:
            return
        now_mono = time.monotonic()
        px = float(price)
        if not math.isfinite(px) or px <= 0:
            return
        timeframes = {t.timeframe for t in active}
        if scanner_watch:
            timeframes.add(self.timeframe)
        for tf in sorted(timeframes):
            if tf not in volumatic.TIMEFRAME_MS:
                continue
            # Do not let the selected scanner timeframe hijack a position that
            # belongs to another timeframe; active positions use their own key.
            key = f"{tf}:{symbol}"
            context = self._live_signal_candles.get(key)
            if context is None:
                continue
            live_open = int(context["entry_candle_open_time_ms"])
            live_close = int(context["entry_candle_close_time_ms"])
            event_ms = int(event_time_ms)
            if not (live_open <= event_ms < live_close):
                continue  # maintenance/scanner refreshes the context at the next candle
            if now_mono - self._live_signal_last_eval.get(key, 0.0) < 0.15:
                continue
            self._live_signal_last_eval[key] = now_mono
            if (
                context.get("ema_filter_length") != self.ema_filter_length
                or int(context.get("trend_length") or volumatic.DEFAULT_TREND_LENGTH) != self.trend_length
            ):
                # Runtime /EMA or /length change invalidates the old indicator snapshot.
                # A fresh strategy context will be built on the next scan/bar.
                continue
            signal = volumatic.evaluate_live_visual_context(context, px, now_ms=event_ms)
            if signal is None:
                continue
            try:
                outcome = await self._dispatch_mode_b_signal(
                    signal, live_close, source="MARKET_WEBSOCKET"
                )
                log.info(
                    "[LIVE BUBBLE] %s %s | visual_close=%s | limit=%s | outcome=%s",
                    symbol, signal.get("signal"), signal.get("visual_candle_close_price"),
                    signal.get("limit_price"), outcome,
                )
            except Exception as exc:
                # Keep the latch after ambiguous outcomes to prevent duplicate
                # POSTs. Queue one retry for transient local execution failures.
                log.exception("[LIVE BUBBLE] %s %s processing failed: %s", symbol, signal.get("signal"), exc)
                retry_key = f"{tf}:{symbol}"
                if retry_key not in self._signal_retries:
                    self._signal_retries[retry_key] = {
                        "signal": signal, "expiry_ms": live_close, "key": retry_key,
                        "attempts": 0, "last_error": str(exc)[:500],
                        "source": "MARKET_WEBSOCKET",
                    }

    @staticmethod
    def _ema_sma_seeded(values: np.ndarray, period: int) -> np.ndarray:
        """EMA identical to backtest(10).py smooth(): SMA seed, then recursive EMA."""
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        out = np.full(x.shape, np.nan, dtype=np.float64)
        n = int(period)
        if n < 1 or x.size < n or not np.all(np.isfinite(x)):
            return out
        alpha = 2.0 / (float(n) + 1.0)
        previous = float(np.mean(x[:n]))
        out[n - 1] = previous
        for i in range(n, x.size):
            previous += alpha * (float(x[i]) - previous)
            out[i] = previous
        return out

    @staticmethod
    def _ema_exit_periods(trade: Trade) -> tuple[int, int] | None:
        try:
            selected = volumatic.normalize_ema_filter_length(trade.ema_filter_length)
        except (TypeError, ValueError):
            return None
        return selected if isinstance(selected, tuple) else None

    def _schedule_ema_exit_initialization(self, trade: Trade) -> None:
        """Start at most one low-priority H1 history warm-up per trade."""
        if trade.status != "FILLED" or trade.exchange_adopted:
            return
        if trade.ema_exit_state in {
            "ACTIVE", "ACTIVE_WITH_GAP", "EXIT_PENDING", "CLOSE_SUBMITTING", "CLOSE_VERIFY",
            "CLOSE_UNKNOWN", "CLEANUP_WAIT_API", "SUPERSEDED_BY_PROTECTION",
            "DISABLED", "UNAVAILABLE",
        }:
            return
        periods = self._ema_exit_periods(trade)
        if periods is None:
            trade.ema_exit_state = "DISABLED"
            trade.ema_exit_last_error = "Trade tidak memiliki konfigurasi dual EMA yang tersimpan."
            return
        if self._binance_cooldown_remaining() > 0:
            trade.ema_exit_state = "WAITING_API"
            trade.ema_exit_last_error = "H1 EMA warm-up menunggu Binance REST cooldown."
            return
        now = time.monotonic()
        last_attempt = self._ema_exit_last_retry.get(trade.trade_id, 0.0)
        if now - last_attempt < EMA_EXIT_INIT_RETRY_SECONDS:
            return
        existing = self._ema_exit_init_tasks.get(trade.trade_id)
        if existing is not None and not existing.done():
            return
        self._ema_exit_last_retry[trade.trade_id] = now
        task = asyncio.create_task(
            self._initialize_ema_exit_trade(trade.trade_id),
            name=f"ema-h1-init-{trade.pair}",
        )
        self._ema_exit_init_tasks[trade.trade_id] = task

    async def _initialize_ema_exit_trade(self, trade_id: str) -> None:
        """Seed/repair the H1 EMA exit state and replay any missed closed candles."""
        exit_enabled_at_start = self.h1_enabled
        h1_generation_at_start = self._h1_toggle_generation
        try:
            async with self._ema_exit_init_semaphore:
                trade = self.active_trades.get(trade_id)
                if trade is None or trade.status != "FILLED" or trade.exchange_adopted:
                    return
                periods = self._ema_exit_periods(trade)
                if periods is None:
                    trade.ema_exit_state = "DISABLED"
                    trade.ema_exit_last_error = "Dual EMA tidak tersedia pada trade snapshot."
                    await self._save_active_state()
                    return
                fast_len, slow_len = periods
                # Four EMA periods provide a stable SMA-seeded warm-up, while
                # also replaying the interval since the last persisted H1 state.
                slow_bars = max(99, min(1500, slow_len * 4 + 12))
                saved_open = int(trade.ema_exit_last_open_ms or 0)
                filled_ms = int((trade.filled_at or trade.created_at).timestamp() * 1000)
                if saved_open:
                    replay_bars = max(99, int(max(0, int(time.time() * 1000) - saved_open) / H1_MS) + 4)
                    count = min(1500, max(slow_bars, replay_bars))
                else:
                    age_bars = max(0, int((int(time.time() * 1000) - filled_ms) / H1_MS) + 4)
                    count = min(1500, max(slow_bars, age_bars))
                if slow_len > count:
                    trade.ema_exit_state = "UNAVAILABLE"
                    trade.ema_exit_last_error = (
                        f"EMA slow {slow_len} membutuhkan lebih dari {count} candle H1; "
                        "histori Binance yang diminta tidak cukup."
                    )
                    await self._save_active_state()
                    await self.reply(
                        f"⚠️ EMA H1 EXIT BELUM AKTIF • {trade.pair}\n"
                        f"EMA: {fast_len}/{slow_len}\n"
                        f"Alasan: {trade.ema_exit_last_error}\n"
                        "SL/TP tetap aktif di Binance; periksa konfigurasi EMA/history."
                    )
                    return

                trade.ema_exit_state = "INITIALIZING"
                await self._save_active_state()
                candles = await self.rest.get_klines(
                    trade.pair, "1h", count, H1_MS, force_refresh=False
                )
                if not isinstance(candles, np.ndarray) or candles.size < slow_len:
                    raise RuntimeError(
                        f"Histori H1 tidak cukup untuk EMA{slow_len}: "
                        f"{len(candles) if isinstance(candles, np.ndarray) else 0}/{slow_len}."
                    )
                opens = np.asarray(candles["t"], dtype=np.int64)
                closes = np.asarray(candles["c"], dtype=np.float64)
                fast_values = self._ema_sma_seeded(closes, fast_len)
                slow_values = self._ema_sma_seeded(closes, slow_len)
                valid = np.isfinite(fast_values) & np.isfinite(slow_values)
                valid_indices = np.flatnonzero(valid)
                if valid_indices.size == 0:
                    raise RuntimeError("EMA H1 belum memiliki warm-up yang valid.")

                # If a legacy/opened trade predates the available history window,
                # its past cross cannot be reconstructed safely. Resume from the
                # latest confirmed candle and surface the gap instead of inventing
                # an exit signal from the current EMA relationship.
                first_valid = int(valid_indices[0])
                history_first_open = int(opens[first_valid])
                uncovered_gap = False
                if saved_open and history_first_open > saved_open + H1_MS:
                    uncovered_gap = True
                elif not saved_open and history_first_open > (filled_ms // H1_MS) * H1_MS:
                    uncovered_gap = True

                start_index = first_valid + 1
                if saved_open:
                    start_index = max(start_index, int(np.searchsorted(opens, saved_open, side="right")))
                elif not uncovered_gap:
                    fill_bucket = (filled_ms // H1_MS) * H1_MS
                    start_index = max(start_index, int(np.searchsorted(opens, fill_bucket, side="left")))

                triggered = False
                for i in range(start_index, len(opens)):
                    if not (np.isfinite(fast_values[i - 1]) and np.isfinite(slow_values[i - 1])
                            and np.isfinite(fast_values[i]) and np.isfinite(slow_values[i])):
                        continue
                    candle_close_ms = int(opens[i]) + H1_MS
                    if candle_close_ms <= filled_ms:
                        continue
                    previous_fast = float(fast_values[i - 1])
                    previous_slow = float(slow_values[i - 1])
                    current_fast = float(fast_values[i])
                    current_slow = float(slow_values[i])
                    bearish_cross = current_fast < current_slow and previous_fast >= previous_slow
                    bullish_cross = current_fast > current_slow and previous_fast <= previous_slow
                    trade.ema_exit_last_open_ms = int(opens[i])
                    trade.ema_exit_last_fast = Decimal(str(current_fast))
                    trade.ema_exit_last_slow = Decimal(str(current_slow))
                    should_exit = (
                        (trade.direction == "LONG" and bearish_cross)
                        or (trade.direction == "SHORT" and bullish_cross)
                    )
                    if (
                        should_exit and exit_enabled_at_start and self.h1_enabled
                        and self._h1_toggle_generation == h1_generation_at_start
                    ):
                        reason = "EMA_H1_BEARISH_CROSS" if trade.direction == "LONG" else "EMA_H1_BULLISH_CROSS"
                        trade.ema_exit_state = "EXIT_PENDING"
                        trade.ema_exit_trigger_close_ms = candle_close_ms
                        trade.ema_exit_trigger_price = Decimal(str(float(closes[i])))
                        trade.ema_exit_reason = reason
                        trade.ema_exit_last_error = None
                        triggered = True
                        log.warning(
                            "[EMA EXIT] recovered H1 inverse cross | %s %s EMA%s/%s close=%s",
                            trade.pair, trade.direction, fast_len, slow_len,
                            trade.ema_exit_trigger_price,
                        )
                        break

                if not triggered:
                    last_index = int(valid_indices[-1])
                    trade.ema_exit_last_open_ms = int(opens[last_index])
                    trade.ema_exit_last_fast = Decimal(str(float(fast_values[last_index])))
                    trade.ema_exit_last_slow = Decimal(str(float(slow_values[last_index])))
                    trade.ema_exit_state = "ACTIVE_WITH_GAP" if uncovered_gap else "ACTIVE"
                    trade.ema_exit_last_error = (
                        "Sebagian riwayat H1 sebelum warm-up tidak tersedia; exit dilanjutkan dari candle terbaru. "
                        "Cross yang terjadi saat bot tidak aktif di luar histori ini tidak dapat direkonstruksi."
                        if uncovered_gap else None
                    )
                await self._save_active_state()
                await self._sync_ws_symbols()
                if uncovered_gap:
                    await self.reply(
                        f"⚠️ EMA H1 HISTORY GAP • {trade.pair} {trade.direction}\n"
                        f"EMA: {fast_len}/{slow_len}\n"
                        "Histori candle sebelum warm-up tidak cukup untuk merekonstruksi semua cross saat bot tidak aktif.\n"
                        "Exit EMA tetap aktif untuk cross baru; SL/TP tetap dipertahankan. Periksa posisi ini."
                    )
                if (
                    triggered and trade.ema_exit_state == "EXIT_PENDING"
                    and self.h1_enabled
                    and self._h1_toggle_generation == h1_generation_at_start
                ):
                    if not trade.real_enabled:
                        await self._finalize_trade(
                            trade, "CLOSED", trade.ema_exit_trigger_price,
                            trade.ema_exit_reason or "EMA_H1_INVERSE_CROSS",
                        )
                    else:
                        await self._sync_ws_symbols()
        except BinanceRateLimitError as exc:
            trade = self.active_trades.get(trade_id)
            if trade is not None and trade.status == "FILLED":
                trade.ema_exit_state = "WAITING_API"
                trade.ema_exit_last_error = str(exc)[:400]
                await self._save_active_state()
            log.info("[EMA EXIT] H1 initialization deferred by rate governor | trade=%s | %s", trade_id, exc)
        except Exception as exc:
            trade = self.active_trades.get(trade_id)
            if trade is not None and trade.status == "FILLED":
                trade.ema_exit_state = "INIT_RETRY"
                trade.ema_exit_last_error = str(exc)[:500]
                await self._save_active_state()
            log.warning("[EMA EXIT] H1 initialization failed | trade=%s | %s", trade_id, exc)
        finally:
            current = asyncio.current_task()
            if self._ema_exit_init_tasks.get(trade_id) is current:
                self._ema_exit_init_tasks.pop(trade_id, None)

    async def _finalize_ema_exit_after_flat(
        self, trade: Trade, reason: str | None = None, *, market_close_confirmed: bool = False,
    ) -> None:
        """Position is confirmed flat; only now cancel remaining closePosition protections."""
        if trade.protection_exit_reason in {"STOP_LOSS", "TAKE_PROFIT"}:
            final_reason = trade.protection_exit_reason
        elif market_close_confirmed:
            final_reason = trade.ema_exit_reason or reason or "EMA_H1_INVERSE_CROSS"
        else:
            final_reason = reason or ("TP_SL_EXIT_RECONCILED" if trade.protection_state == "ACTIVE" else "EXTERNAL_CLOSE_RECONCILED")
        keep = None
        if final_reason == "STOP_LOSS":
            keep = "SL"
        elif final_reason == "TAKE_PROFIT":
            keep = "TP"
        if not await self._cancel_real_protections(trade, keep_triggered=keep):
            trade.protection_exit_reason = final_reason
            trade.ema_exit_state = "CLEANUP_WAIT_API"
            await self._save_active_state()
            return
        await self._finalize_trade(
            trade, "CLOSED", self._current_price(trade.pair) or trade.fill_price or trade.entry,
            final_reason,
        )

    async def _reconcile_ema_exit_close(self, trade: Trade) -> None:
        """Resolve a possibly-sent MARKET close by its stable clientOrderId; never POST twice."""
        client_id = str(trade.ema_exit_client_order_id or "").strip()
        if not client_id:
            trade.ema_exit_state = "CLOSE_UNKNOWN"
            trade.ema_exit_last_error = "Close state ambigu tetapi clientOrderId tidak tersimpan; POST ulang dikunci."
            await self._save_active_state()
            return
        try:
            order = await self.real.get_order(trade.pair, client_order_id=client_id)
        except BinanceAPIError as exc:
            if exc.code != -2013:
                raise
            # An absent order is not enough to prove that the original POST never
            # reached Binance. Check exposure; if still open, keep SL/TP and do not
            # manufacture a duplicate MARKET order.
            position = await self.real.get_position(trade.pair, trade.entry_side)
            if position is None:
                await self._finalize_ema_exit_after_flat(trade)
            else:
                trade.ema_exit_state = "CLOSE_UNKNOWN"
                trade.ema_exit_last_error = (
                    "Binance belum menemukan close clientOrderId, tetapi posisi masih terbuka. "
                    "Tidak ada POST ulang otomatis; SL/TP tetap aktif. Perlu verifikasi manual bila berlarut."
                )
                await self._save_active_state()
                log.error("[EMA EXIT] ambiguous close remains open; no duplicate MARKET POST | %s %s", trade.pair, trade.direction)
            return

        status = str(order.get("status") or "UNKNOWN").upper()
        trade.ema_exit_order_id = int(order.get("orderId")) if order.get("orderId") not in (None, "") else trade.ema_exit_order_id
        trade.ema_exit_order_status = status
        await self._save_active_state()
        position = await self.real.get_position(trade.pair, trade.entry_side)
        if position is None:
            await self._finalize_ema_exit_after_flat(
                trade, market_close_confirmed=(status == "FILLED"),
            )
            return
        if status in {"NEW", "PARTIALLY_FILLED", "UNKNOWN"}:
            trade.ema_exit_state = "CLOSE_VERIFY"
            trade.ema_exit_last_error = f"Close order masih {status}; menunggu konfirmasi posisi flat."
            await self._save_active_state()
            return
        if status in {"FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}:
            trade.ema_exit_state = "CLOSE_UNKNOWN"
            trade.ema_exit_last_error = (
                f"Close order berstatus {status}, tetapi posisi arah {trade.direction} masih terbuka. "
                "Bot tidak mengirim MARKET kedua secara otomatis agar tidak menutup exposure tambahan."
            )
            await self._save_active_state()
            log.error("[EMA EXIT] terminal close order but position still open | %s %s | %s", trade.pair, trade.direction, status)
            return

    async def _process_ema_exit_real(self, trade: Trade) -> None:
        """Execute/reconcile one H1 exit while leaving Binance SL/TP armed until flat."""
        if not trade.real_enabled or trade.status != "FILLED":
            return
        if self._binance_cooldown_remaining() > 0:
            return
        lock = self._ema_exit_close_locks.setdefault(trade.trade_id, asyncio.Lock())
        async with lock:
            if trade.status != "FILLED":
                return
            if trade.ema_exit_state in {"CLOSE_SUBMITTING", "CLOSE_VERIFY", "CLOSE_UNKNOWN"}:
                now_mono = time.monotonic()
                interval = (
                    EMA_EXIT_UNKNOWN_RECHECK_SECONDS
                    if trade.ema_exit_state in {"CLOSE_SUBMITTING", "CLOSE_UNKNOWN"}
                    else EMA_EXIT_RECONCILE_SECONDS
                )
                if now_mono - self._ema_exit_close_last_check.get(trade.trade_id, 0.0) < interval:
                    return
                self._ema_exit_close_last_check[trade.trade_id] = now_mono
                await self._reconcile_ema_exit_close(trade)
                return
            if trade.ema_exit_state != "EXIT_PENDING":
                return
            if (
                trade.protection_exit_reason in {"STOP_LOSS", "TAKE_PROFIT"}
                or trade.sl_algo_status.upper() in {"TRIGGERED", "FINISHED"}
                or trade.tp_algo_status.upper() in {"TRIGGERED", "FINISHED"}
            ):
                trade.ema_exit_state = "SUPERSEDED_BY_PROTECTION"
                trade.ema_exit_last_error = "SL/TP Binance terpicu; EMA market close dibatalkan."
                await self._save_active_state()
                return
            if not trade.ema_exit_client_order_id:
                identity = f"{trade.trade_id}|{trade.ema_exit_trigger_close_ms or 0}|{trade.ema_exit_reason or ''}"
                trade.ema_exit_client_order_id = "emaexit" + hashlib.sha256(identity.encode()).hexdigest()[:22]
            trade.ema_exit_attempts += 1
            trade.ema_exit_last_error = None
            await self._save_active_state()
            reference = trade.ema_exit_trigger_price or self._current_price(trade.pair) or trade.fill_price or trade.entry
            try:
                await self._close_trade(
                    trade, reference,
                    trade.ema_exit_reason or "EMA_H1_INVERSE_CROSS",
                    keep_protections_until_flat=True,
                    client_order_id=trade.ema_exit_client_order_id,
                )
            except BinanceRateLimitError as exc:
                # If the MARKET order itself was rate-limited or preflight was
                # blocked, it was not accepted. If it had been accepted, the
                # state was already moved to CLOSE_VERIFY before verification.
                if trade.ema_exit_state == "CLOSE_SUBMITTING":
                    trade.ema_exit_state = "EXIT_PENDING"
                trade.ema_exit_last_error = str(exc)[:500]
                await self._save_active_state()
                raise
            except Exception as exc:
                if trade.ema_exit_state == "CLOSE_SUBMITTING":
                    trade.ema_exit_state = "CLOSE_UNKNOWN"
                trade.ema_exit_last_error = str(exc)[:500]
                await self._save_active_state()
                log.exception("[EMA EXIT] close operation needs reconciliation | %s %s", trade.pair, trade.direction)
            else:
                # _close_trade may leave the ledger open only while Algo cleanup
                # is rate-limited; its CANCEL_WAIT_API state is handled by the
                # existing position reconciler without exposing the position.
                if trade.status == "FILLED" and trade.protection_state == "CANCEL_WAIT_API":
                    trade.ema_exit_state = "CLEANUP_WAIT_API"
                    await self._save_active_state()

    async def _process_ema_exit_candle(self, symbol: str, kline: dict[str, Any]) -> None:
        """Consume only a confirmed H1 close; intrabar EMA crosses are ignored."""
        if not isinstance(kline, dict) or kline.get("i") != "1h" or kline.get("x") is not True:
            return
        try:
            open_ms = int(kline["t"])
            close_ms = open_ms + H1_MS
            close_price = Decimal(str(kline["c"]))
            if not close_price.is_finite() or close_price <= 0:
                return
        except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
            return

        changed = False
        for trade in list(self.active_trades.values()):
            if (
                trade.pair != symbol or trade.status != "FILLED" or trade.exchange_adopted
                or self._ema_exit_periods(trade) is None
            ):
                continue
            if trade.ema_exit_state in {"EXIT_PENDING", "CLOSE_SUBMITTING", "CLOSE_VERIFY", "CLOSE_UNKNOWN", "CLEANUP_WAIT_API", "SUPERSEDED_BY_PROTECTION", "DISABLED", "UNAVAILABLE"}:
                continue
            if trade.ema_exit_state not in {"ACTIVE", "ACTIVE_WITH_GAP"}:
                self._schedule_ema_exit_initialization(trade)
                continue
            last_open = int(trade.ema_exit_last_open_ms or 0)
            if open_ms <= last_open:
                continue  # duplicate close event after reconnect
            if last_open <= 0 or open_ms != last_open + H1_MS:
                trade.ema_exit_state = "BACKFILL_PENDING"
                trade.ema_exit_last_error = (
                    f"H1 event gap: last={last_open}, current={open_ms}; reconstructing closed candles."
                )
                changed = True
                self._schedule_ema_exit_initialization(trade)
                continue
            if trade.ema_exit_last_fast is None or trade.ema_exit_last_slow is None:
                trade.ema_exit_state = "INIT_RETRY"
                self._schedule_ema_exit_initialization(trade)
                changed = True
                continue
            periods = self._ema_exit_periods(trade)
            if periods is None:
                trade.ema_exit_state = "DISABLED"
                continue
            fast_len, slow_len = periods
            alpha_fast = 2.0 / (fast_len + 1.0)
            alpha_slow = 2.0 / (slow_len + 1.0)
            previous_fast = float(trade.ema_exit_last_fast)
            previous_slow = float(trade.ema_exit_last_slow)
            current_fast = previous_fast + alpha_fast * (float(close_price) - previous_fast)
            current_slow = previous_slow + alpha_slow * (float(close_price) - previous_slow)
            bearish_cross = current_fast < current_slow and previous_fast >= previous_slow
            bullish_cross = current_fast > current_slow and previous_fast <= previous_slow
            trade.ema_exit_last_open_ms = open_ms
            trade.ema_exit_last_fast = Decimal(str(current_fast))
            trade.ema_exit_last_slow = Decimal(str(current_slow))
            changed = True
            should_exit = (
                (trade.direction == "LONG" and bearish_cross)
                or (trade.direction == "SHORT" and bullish_cross)
            )
            if not self.h1_enabled:
                continue  # update EMA baseline only; ignore H1 crosses while the master switch is OFF
            if not should_exit or close_ms <= int((trade.filled_at or trade.created_at).timestamp() * 1000):
                continue
            reason = "EMA_H1_BEARISH_CROSS" if trade.direction == "LONG" else "EMA_H1_BULLISH_CROSS"
            trade.ema_exit_state = "EXIT_PENDING"
            trade.ema_exit_trigger_close_ms = close_ms
            trade.ema_exit_trigger_price = close_price
            trade.ema_exit_reason = reason
            trade.ema_exit_last_error = None
            log.warning(
                "[EMA EXIT] confirmed H1 inverse cross | %s %s EMA%s/%s close=%s | SL=%s TP=%s remain armed",
                trade.pair, trade.direction, fast_len, slow_len, close_price,
                trade.stop_loss, trade.take_profit,
            )
            if not trade.real_enabled:
                # Paper mirrors backtest's H1-close price. Earlier local SL/TP
                # handling removes the trade before this callback can reach it.
                await self._finalize_trade(trade, "CLOSED", close_price, reason)
            else:
                await self.reply(
                    f"🔔 EMA H1 EXIT DETECTED • {trade.pair} {trade.direction}\n"
                    f"EMA: {fast_len}/{slow_len}\n"
                    f"H1 close: {fmt_price(close_price)}\n"
                    f"SL: {fmt_price(trade.stop_loss)} · TP: {fmt_price(trade.take_profit)}\n"
                    "SL/TP tetap aktif sampai MARKET close terkonfirmasi."
                )
        if changed:
            await self._save_active_state()
            await self._sync_ws_symbols()

    async def _on_kline(self, symbol: str, kline: dict[str, Any]) -> None:
        """Route M15 scanner and H1 position-exit candles through one cache owner."""
        if not isinstance(kline, dict):
            return
        self.rest.update_kline_from_ws(symbol, kline)
        interval = str(kline.get("i") or "").lower()
        if interval == "1h":
            await self._process_ema_exit_candle(normalize_symbol(symbol), kline)

    async def _on_price(self, symbol: str, price: Decimal, event_time_ms: int, aggregate_trade_id: int = -1) -> None:
        symbol = normalize_symbol(symbol)
        key = (int(event_time_ms), int(aggregate_trade_id))
        previous_key = self._last_market_event_key.get(symbol)
        if previous_key is not None and key <= previous_key:
            return
        self._last_market_event_key[symbol] = key
        self.prices[symbol] = PriceSnapshot(symbol, price, int(event_time_ms), now_utc())
        for trade in list(self.active_trades.values()):
            if trade.pair != symbol:
                continue
            if trade.status == "PENDING" and trade.real_intent and not trade.real_enabled:
                if not trade.entry_inflight and now_utc() < trade.expires_at:
                    touched = price <= trade.entry if trade.direction == "LONG" else price >= trade.entry
                    if touched:
                        self._schedule_market_entry(trade, price, "WEBSOCKET_TOUCH")
                continue
            if trade.status == "PENDING" and not trade.real_enabled:
                if now_utc() < trade.expires_at:
                    touched = price <= trade.entry if trade.direction == "LONG" else price >= trade.entry
                    if touched:
                        await self._mark_filled(trade, trade.entry, "PAPER_WEBSOCKET_LIMIT_TOUCH")
            elif trade.status == "FILLED" and not trade.real_enabled:
                stop = trade.stop_loss or trade.reference_sl
                target = trade.take_profit
                if stop is None or target is None:
                    continue
                reason = None
                fill = None
                if trade.direction == "LONG":
                    if price <= stop: reason, fill = "STOP_LOSS", stop
                    elif price >= target: reason, fill = "TAKE_PROFIT", target
                else:
                    if price >= stop: reason, fill = "STOP_LOSS", stop
                    elif price <= target: reason, fill = "TAKE_PROFIT", target
                if reason:
                    await self._finalize_trade(trade, "CLOSED", fill, reason)

    @staticmethod
    def _exchange_event_time(row: dict[str, Any]) -> datetime:
        """Best available exchange timestamp, falling back to current UTC."""
        for key in ("updateTime", "time", "T", "E", "T"):
            raw = row.get(key)
            if raw in (None, "", 0, "0"):
                continue
            try:
                value = float(raw)
                if value > 10_000_000_000:
                    value /= 1000.0
                return datetime.fromtimestamp(value, tz=timezone.utc)
            except (TypeError, ValueError, OSError, OverflowError):
                continue
        return now_utc()

    @staticmethod
    def _exchange_position_direction(row: dict[str, Any], amount: Decimal | None = None) -> str | None:
        side = str(row.get("positionSide") or row.get("ps") or "BOTH").upper()
        if side == "LONG":
            return "LONG"
        if side == "SHORT":
            return "SHORT"
        if amount is None:
            raw = row.get("positionAmt", row.get("pa", "0"))
            try:
                amount = parse_signed_decimal(str(raw or "0"))
            except Exception:
                return None
        if amount > 0:
            return "LONG"
        if amount < 0:
            return "SHORT"
        return None

    def _refresh_real_position_cache(self, symbol: str) -> None:
        rows = [
            dict(row) for (pair, _side), row in self.exchange_positions.items()
            if pair == symbol
        ]
        if rows:
            self.real.last_positions[symbol] = rows
            self.real.last_positions_at[symbol] = time.monotonic()
        else:
            self.real.last_positions.pop(symbol, None)
            self.real.last_positions_at[symbol] = time.monotonic()

    def _find_real_trade(self, symbol: str, direction: str) -> Trade | None:
        return next((
            trade for trade in self.active_trades.values()
            if trade.real_enabled and trade.pair == symbol
            and trade.direction == direction and trade.status in {"PENDING", "FILLED"}
        ), None)

    def _new_exchange_adopted_trade(
        self,
        *,
        symbol: str,
        direction: str,
        entry: Decimal,
        quantity: Decimal,
        position_side: str,
        source: str,
        status: str = "FILLED",
        signal_time: datetime | None = None,
        leverage: int | None = None,
        margin: Decimal | None = None,
        client_order_id: str | None = None,
        order_id: int | None = None,
        order_status: str = "FILLED",
        executed_qty: Decimal | None = None,
        preserve_gtc: bool = False,
    ) -> Trade:
        """Create an in-memory view from live Binance state, never from old local files."""
        stamp = signal_time or now_utc()
        lev = max(1, int(leverage or self.leverage or DEFAULT_LEVERAGE))
        notional = abs(quantity) * entry if quantity > 0 and entry > 0 else Decimal("0")
        margin_value = margin if margin is not None and margin > 0 else (
            notional / Decimal(lev) if notional > 0 else self.margin_usdt
        )
        direction = direction.upper()
        order_is_still_open = status == "PENDING" or order_status in {"NEW", "PARTIALLY_FILLED"}
        order_created_at = stamp if order_is_still_open else None
        expiry_anchor = order_created_at or stamp
        return Trade(
            trade_id=generate_trade_id(symbol),
            session_id=self.session_id,
            pair=symbol,
            direction=direction,
            timeframe="EXCHANGE",
            trend_length=self.trend_length,
            signal="BLUE" if direction == "LONG" else "RED",
            signal_time=stamp,
            bubble_price=entry,
            signal_atr=None,
            reference_sl=None,
            entry=entry,
            expires_at=expiry_anchor + timedelta(seconds=ENTRY_ORDER_TTL_SECONDS),
            signal_close_price=entry,
            entry_order_created_at=order_created_at,
            ema_filter_length=None,
            ema_filter_value=None,
            ema_filter_passed=None,
            status=status,
            created_at=stamp,
            filled_at=(stamp if status == "FILLED" else None),
            fill_price=(entry if status == "FILLED" else None),
            quantity=quantity if quantity > 0 else None,
            margin_usdt=margin_value,
            leverage=lev,
            target_notional=notional if notional > 0 else None,
            actual_notional=notional if notional > 0 else None,
            real_enabled=True,
            real_state=("REAL_EXCHANGE_ADOPTED" if status == "FILLED" else "REAL_EXCHANGE_ADOPTED_PENDING"),
            ema_exit_state="DISABLED",
            position_side=position_side or "BOTH",
            entry_order_id=order_id,
            entry_client_order_id=client_order_id,
            entry_executed_qty=executed_qty,
            entry_order_status=order_status,
            entry_order_terminal=(order_status in {"FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}),
            preserve_gtc_after_cooldown=preserve_gtc,
            exchange_adopted=True,
            exchange_adoption_source=source,
        )

    def _recover_terminal_local_order_trade(
        self,
        *,
        symbol: str,
        client_order_id: str,
        order_id: int | None,
        exchange_status: str,
    ) -> Trade | None:
        """Reopen a locally finalized bot trade when Binance proves its LIMIT is live.

        Exact exchange identifiers are required. This prevents a stale local
        CANCELED record from hiding an active order or losing its original SL/TP
        setup when the private WebSocket event was missed.
        """
        if exchange_status not in {"NEW", "PARTIALLY_FILLED", "FILLED"}:
            return None
        for trade in self.active_trades.values():
            if not trade.real_enabled or trade.pair != symbol:
                continue
            same_client = bool(client_order_id and trade.entry_client_order_id == client_order_id)
            same_order = bool(order_id is not None and trade.entry_order_id is not None and int(trade.entry_order_id) == int(order_id))
            if not (same_client or same_order):
                continue
            if trade.status in {"PENDING", "FILLED"}:
                return trade

            previous_status = trade.status
            trade.status = "PENDING"
            trade.result = None
            trade.result_reason = None
            trade.closed_at = None
            trade.exit_price = None
            trade.pnl_percent = None
            trade.pnl_usdt = None
            trade.realized_r = None
            trade.filled_at = None
            trade.entry_order_status = exchange_status
            trade.entry_order_terminal = False
            trade.pending_cancel_reason = None
            trade.real_submit_ambiguous = False
            trade.real_state = "REAL_PENDING"
            trade.exchange_adopted = False
            trade.exchange_adoption_source = None
            # The original local setup is retained, especially Supertrend SL and
            # RR target, so a later fill can receive the intended TP/SL protection.
            self.history_records = [
                record for record in self.history_records
                if str(record.get("trade_id") or "") != trade.trade_id
            ]
            log.warning(
                "[REAL/RECOVERY] Binance confirms live LIMIT; restored local trade %s %s %s from %s to PENDING | clientId=%s orderId=%s",
                trade.pair, trade.direction, trade.trade_id, previous_status,
                client_order_id or "-", order_id or "-",
            )
            return trade
        return None

    async def _upsert_exchange_position(
        self, row: dict[str, Any], *, source: str, finalize_on_flat: bool = True
    ) -> Trade | None:
        """Apply one position row from /account snapshot or ACCOUNT_UPDATE."""
        symbol = normalize_symbol(str(row.get("symbol") or row.get("s") or ""))
        if not symbol:
            return None
        side = str(row.get("positionSide") or row.get("ps") or "BOTH").upper()
        amount_raw = row.get("positionAmt", row.get("pa", "0"))
        try:
            amount = parse_signed_decimal(str(amount_raw or "0"))
        except ValueError:
            return None
        key = (symbol, side)
        previous = self.exchange_positions.get(key)

        if amount == 0:
            old = self.exchange_positions.pop(key, None)
            self._refresh_real_position_cache(symbol)
            if old and finalize_on_flat:
                old_amount = old.get("_position_amount_decimal")
                try:
                    old_amount = old_amount if isinstance(old_amount, Decimal) else parse_signed_decimal(str(old.get("positionAmt") or "0"))
                except Exception:
                    old_amount = Decimal("0")
                old_direction = self._exchange_position_direction(old, old_amount)
                if old_direction:
                    trade = self._find_real_trade(symbol, old_direction)
                    if trade and trade.exchange_adopted and trade.status == "FILLED":
                        close_price = self._current_price(symbol)
                        if close_price is None:
                            try:
                                close_price = Decimal(str(old.get("markPrice") or old.get("entryPrice") or trade.fill_price or trade.entry))
                            except Exception:
                                close_price = trade.fill_price or trade.entry
                        await self._finalize_trade(
                            trade, "CLOSED", close_price,
                            f"ACCOUNT_UPDATE_POSITION_FLAT:{source}",
                        )
            return None

        normalized = dict(previous or {})
        normalized.update(row)
        normalized["symbol"] = symbol
        normalized["positionSide"] = side
        normalized["positionAmt"] = str(amount)
        normalized["_position_amount_decimal"] = amount
        normalized["_abs_position_amt"] = abs(amount)
        if row.get("ep") not in (None, ""):
            normalized["entryPrice"] = str(row.get("ep"))
        if row.get("mp") not in (None, ""):
            normalized["markPrice"] = str(row.get("mp"))
        if row.get("up") not in (None, ""):
            normalized["unRealizedProfit"] = str(row.get("up"))
        self.exchange_positions[key] = normalized
        self._refresh_real_position_cache(symbol)

        direction = self._exchange_position_direction(normalized, amount)
        if direction is None:
            return None
        try:
            entry = Decimal(str(normalized.get("entryPrice") or "0"))
        except (InvalidOperation, ValueError):
            entry = Decimal("0")
        if entry <= 0:
            existing = self._find_real_trade(symbol, direction)
            entry = (existing.fill_price or existing.entry) if existing else Decimal("0")
        quantity = abs(amount)
        try:
            leverage = max(1, int(normalized.get("leverage") or self.leverage or DEFAULT_LEVERAGE))
        except (TypeError, ValueError):
            leverage = self.leverage
        margin = None
        for margin_key in ("initialMargin", "isolatedMargin", "positionInitialMargin"):
            try:
                candidate = Decimal(str(normalized.get(margin_key) or "0"))
                if candidate > 0:
                    margin = candidate
                    break
            except (InvalidOperation, ValueError):
                pass
        stamp = self._exchange_event_time(normalized)
        trade = self._find_real_trade(symbol, direction)
        if trade is None:
            trade = self._new_exchange_adopted_trade(
                symbol=symbol, direction=direction, entry=entry, quantity=quantity,
                position_side=side, source=source, status="FILLED", signal_time=stamp,
                leverage=leverage, margin=margin, order_status="FILLED",
                executed_qty=quantity,
            )
            self.active_trades[trade.trade_id] = trade
            log.info("[REAL/WS] exchange position adopted | %s %s qty=%s entry=%s source=%s", symbol, direction, quantity, entry, source)
        else:
            was_pending = trade.status == "PENDING"
            trade.quantity = quantity
            trade.entry_executed_qty = quantity
            trade.fill_price = entry if entry > 0 else (trade.fill_price or trade.entry)
            trade.actual_notional = quantity * trade.fill_price
            trade.margin_usdt = margin if margin is not None else trade.actual_notional / Decimal(max(1, trade.leverage))
            trade.position_side = side
            if was_pending:
                await self._mark_filled(
                    trade, trade.fill_price, f"BINANCE_ACCOUNT_UPDATE_{source}", filled_at=stamp
                )
            elif trade.exchange_adopted:
                trade.real_state = "REAL_EXCHANGE_ADOPTED"
        await self.ws.add_symbol(symbol)
        return trade

    async def _upsert_exchange_order_snapshot(self, row: dict[str, Any], *, source: str) -> Trade | None:
        """Adopt one live open LIMIT order from Binance's account-wide openOrders snapshot."""
        symbol = normalize_symbol(str(row.get("symbol") or row.get("s") or ""))
        if not symbol:
            return None
        client_id = str(row.get("clientOrderId") or row.get("origClientOrderId") or row.get("c") or "")
        order_id_raw = row.get("orderId", row.get("i"))
        try:
            order_id = int(order_id_raw) if order_id_raw not in (None, "") else None
        except (TypeError, ValueError):
            order_id = None
        key = client_id or (f"{symbol}:{order_id}" if order_id is not None else "")
        if not key:
            return None
        normalized = dict(row)
        normalized["symbol"] = symbol
        normalized["clientOrderId"] = client_id
        normalized["orderId"] = order_id
        self.exchange_open_orders[key] = normalized

        side = str(row.get("side") or row.get("S") or "").upper()
        direction = "LONG" if side == "BUY" else "SHORT" if side == "SELL" else None
        if direction is None:
            return None
        try:
            price = Decimal(str(row.get("price") or row.get("p") or "0"))
            orig_qty = Decimal(str(row.get("origQty") or row.get("q") or "0"))
            executed = Decimal(str(row.get("executedQty") or row.get("z") or "0"))
        except (InvalidOperation, ValueError):
            return None
        if price <= 0:
            return None
        status = str(row.get("status") or row.get("X") or "NEW").upper()
        stamp = self._exchange_event_time(row)
        trade = next((
            item for item in self.active_trades.values()
            if item.real_enabled and item.pair == symbol and item.direction == direction
            and item.status in {"PENDING", "FILLED"}
            and (
                (client_id and item.entry_client_order_id == client_id)
                or (order_id is not None and item.entry_order_id == order_id)
            )
        ), None)
        if trade is None:
            trade = self._recover_terminal_local_order_trade(
                symbol=symbol, client_order_id=client_id, order_id=order_id,
                exchange_status=status,
            )
        if trade is None:
            trade = next((
                item for item in self.active_trades.values()
                if item.real_enabled and item.pair == symbol and item.direction == direction
                and item.status in {"PENDING", "FILLED"} and item.exchange_adopted
            ), None)
        if trade is None:
            trade_status = "FILLED" if executed > 0 else "PENDING"
            trade = self._new_exchange_adopted_trade(
                symbol=symbol, direction=direction, entry=price,
                quantity=(executed if executed > 0 else orig_qty),
                position_side=str(row.get("positionSide") or row.get("ps") or "BOTH"),
                source=source, status=trade_status, signal_time=stamp,
                leverage=self.leverage,
                client_order_id=client_id or None, order_id=order_id,
                order_status=status, executed_qty=executed,
                preserve_gtc=(self._binance_cooldown_remaining() > 0),
            )
            if executed > 0:
                try:
                    avg = Decimal(str(row.get("avgPrice") or row.get("ap") or price))
                    if avg > 0:
                        trade.fill_price = avg
                except (InvalidOperation, ValueError):
                    pass
            self.active_trades[trade.trade_id] = trade
            log.info("[REAL/WS] open order adopted | %s %s clientId=%s status=%s", symbol, direction, client_id, status)
        else:
            trade.entry_order_id = order_id or trade.entry_order_id
            trade.entry_client_order_id = client_id or trade.entry_client_order_id
            trade.entry_order_status = status
            trade.entry_order_terminal = False
            trade.entry_executed_qty = executed
            if trade.entry_order_created_at is None:
                trade.entry_order_created_at = stamp
                trade.expires_at = stamp + timedelta(seconds=ENTRY_ORDER_TTL_SECONDS)
            trade.preserve_gtc_after_cooldown = (self._binance_cooldown_remaining() > 0)
            if executed > 0:
                trade.quantity = executed
                try:
                    avg = Decimal(str(row.get("avgPrice") or row.get("ap") or price))
                    if avg > 0:
                        trade.fill_price = avg
                except (InvalidOperation, ValueError):
                    pass
                if trade.status == "PENDING":
                    await self._mark_filled(
                        trade, trade.fill_price or price, f"BINANCE_OPEN_ORDER_SNAPSHOT_{source}",
                        filled_at=stamp,
                    )
        await self.ws.add_symbol(symbol)
        return trade

    async def _adopt_exchange_snapshot(
        self,
        account_snapshot: dict[str, Any],
        open_orders: list[dict[str, Any]],
        *,
        source: str,
    ) -> tuple[int, int]:
        """Seed current live exchange exposure; deliberately does not read old local/GitHub state."""
        self.exchange_positions.clear()
        self.exchange_open_orders.clear()
        position_rows = account_snapshot.get("positions") if isinstance(account_snapshot, dict) else []
        for row in position_rows or []:
            if not isinstance(row, dict):
                continue
            try:
                amount = parse_signed_decimal(str(row.get("positionAmt") or "0"))
            except Exception:
                continue
            if amount == 0:
                continue
            await self._upsert_exchange_position(row, source=source, finalize_on_flat=False)
        for row in open_orders or []:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "NEW").upper()
            if status in {"NEW", "PARTIALLY_FILLED"}:
                await self._upsert_exchange_order_snapshot(row, source=source)
        self.exchange_snapshot_at = now_utc()
        await self._sync_ws_symbols()
        await self._save_active_state()
        positions_count = len(self.exchange_positions)
        orders_count = len(self.exchange_open_orders)
        log.info(
            "[REAL/WS] account snapshot seeded | positions=%s open_orders=%s source=%s",
            positions_count, orders_count, source,
        )
        return positions_count, orders_count

    async def _apply_account_update_position(self, row: dict[str, Any]) -> None:
        """Update Binance position projection and finalize bot trades on a WS flat event."""
        symbol = normalize_symbol(str(row.get("s") or row.get("symbol") or ""))
        if not symbol:
            return
        side = str(row.get("ps") or row.get("positionSide") or "BOTH").upper()
        try:
            amount = parse_signed_decimal(str(row.get("pa", row.get("positionAmt", "0")) or "0"))
        except ValueError:
            return
        key = (symbol, side)
        old = self.exchange_positions.get(key)
        if amount == 0:
            old_direction = self._exchange_position_direction(old, old.get("_position_amount_decimal") if old else None) if old else None
            self.exchange_positions.pop(key, None)
            self._refresh_real_position_cache(symbol)
            if old_direction is None:
                matching = [t for t in self.active_trades.values()
                            if t.real_enabled and t.status == "FILLED" and t.pair == symbol
                            and (t.position_side in {None, "BOTH", side} or side == "BOTH")]
                if len(matching) == 1:
                    old_direction = matching[0].direction
            if old_direction:
                trade = self._find_real_trade(symbol, old_direction)
                if trade and trade.status == "FILLED" and trade.exchange_adopted:
                    # Adopted exchange positions have no bot-created entry remainder
                    # or Algo exits; keep them as a read-only exchange ledger row.
                    close_price = self._current_price(symbol)
                    if close_price is None:
                        close_price = Decimal(str((old or {}).get("markPrice") or (old or {}).get("entryPrice") or trade.fill_price or trade.entry))
                    await self._finalize_trade(trade, "CLOSED", close_price, "EXCHANGE_POSITION_FLAT_WS")
                    return
                if trade and trade.status == "FILLED":
                    # A partial entry remainder must be confirmed canceled before releasing the slot.
                    if not trade.entry_order_terminal:
                        try:
                            await self._cancel_real_entry_remainder(trade, "PROTECTION_EXIT_CANCEL_ENTRY_REMAINDER")
                        except (BinanceRateLimitError, Exception) as exc:
                            trade.pending_cancel_reason = "PROTECTION_EXIT_CANCEL_ENTRY_REMAINDER"
                            trade.real_error = f"Position sudah flat; entry remainder menunggu cancel: {str(exc)[:250]}"
                            await self._save_active_state()
                            return
                        if not trade.entry_order_terminal:
                            return
                    close_price = self._current_price(symbol)
                    if close_price is None:
                        close_price = Decimal(str((old or {}).get("markPrice") or (old or {}).get("entryPrice") or trade.fill_price or trade.entry))
                    reason = trade.protection_exit_reason or "EXCHANGE_POSITION_FLAT_WS"
                    if not await self._cancel_real_protections(trade, keep_triggered=("SL" if reason == "STOP_LOSS" else "TP" if reason == "TAKE_PROFIT" else None)):
                        trade.protection_exit_reason = reason
                        return
                    await self._finalize_trade(trade, "CLOSED", close_price, reason)
            return
        await self._upsert_exchange_position(row, source="USER_DATA_WS", finalize_on_flat=False)

    async def _upsert_exchange_order_event(self, order: dict[str, Any]) -> Trade | None:
        """Update order cache from ORDER_TRADE_UPDATE and adopt bot-tagged orders if needed."""
        symbol = normalize_symbol(str(order.get("s") or order.get("symbol") or ""))
        if not symbol:
            return None
        client_id = str(order.get("c") or order.get("clientOrderId") or "")
        order_id_raw = order.get("i", order.get("orderId"))
        try:
            order_id = int(order_id_raw) if order_id_raw not in (None, "") else None
        except (TypeError, ValueError):
            order_id = None
        status = str(order.get("X") or order.get("status") or "UNKNOWN").upper()
        execution = str(order.get("x") or "UNKNOWN").upper()
        cache_key = client_id or (f"{symbol}:{order_id}" if order_id is not None else "")
        if status in {"NEW", "PARTIALLY_FILLED"} and cache_key:
            cached = dict(order)
            cached.update({"symbol": symbol, "clientOrderId": client_id, "orderId": order_id, "status": status})
            self.exchange_open_orders[cache_key] = cached
        elif cache_key:
            self.exchange_open_orders.pop(cache_key, None)

        trade = next((
            item for item in self.active_trades.values()
            if item.real_enabled and item.pair == symbol and (
                (client_id and item.entry_client_order_id == client_id)
                or (order_id is not None and item.entry_order_id == order_id)
            ) and item.status in {"PENDING", "FILLED"}
        ), None)
        if trade is None and client_id.startswith("VT") and status in {"NEW", "PARTIALLY_FILLED", "FILLED"}:
            trade = self._recover_terminal_local_order_trade(
                symbol=symbol, client_order_id=client_id, order_id=order_id,
                exchange_status=status,
            )
        if trade is None and client_id.startswith("VT") and status in {"NEW", "PARTIALLY_FILLED"}:
            try:
                price = Decimal(str(order.get("p") or "0"))
                qty = Decimal(str(order.get("q") or "0"))
                executed = Decimal(str(order.get("z") or "0"))
            except (InvalidOperation, ValueError):
                return None
            side = str(order.get("S") or "").upper()
            direction = "LONG" if side == "BUY" else "SHORT" if side == "SELL" else None
            if direction and price > 0:
                trade = self._new_exchange_adopted_trade(
                    symbol=symbol, direction=direction, entry=price,
                    quantity=(executed if executed > 0 else qty),
                    position_side=str(order.get("ps") or "BOTH"),
                    source="USER_DATA_WS_ORDER_TRADE_UPDATE",
                    status=("FILLED" if executed > 0 else "PENDING"),
                    signal_time=self._exchange_event_time(order),
                    client_order_id=client_id, order_id=order_id,
                    order_status=status, executed_qty=executed,
                    preserve_gtc=(self._binance_cooldown_remaining() > 0),
                )
                self.active_trades[trade.trade_id] = trade
        elif trade is None and client_id.startswith("VT") and status in {"FILLED", "PARTIALLY_FILLED"}:
            # A fill event may arrive after the order snapshot was missed. Use the
            # account projection from ACCOUNT_UPDATE when it follows; retain a
            # temporary in-memory trade view meanwhile.
            try:
                price = Decimal(str(order.get("ap") or order.get("L") or order.get("p") or "0"))
                executed = Decimal(str(order.get("z") or "0"))
            except (InvalidOperation, ValueError):
                return None
            side = str(order.get("S") or "").upper()
            direction = "LONG" if side == "BUY" else "SHORT" if side == "SELL" else None
            if direction and price > 0 and executed > 0:
                trade = self._new_exchange_adopted_trade(
                    symbol=symbol, direction=direction, entry=price, quantity=executed,
                    position_side=str(order.get("ps") or "BOTH"),
                    source="USER_DATA_WS_ORDER_TRADE_UPDATE",
                    status="FILLED", signal_time=self._exchange_event_time(order),
                    client_order_id=client_id, order_id=order_id,
                    order_status=status, executed_qty=executed,
                )
                self.active_trades[trade.trade_id] = trade
        return trade

    async def _on_user_data_event(self, event: dict[str, Any]) -> None:
        """Apply Binance user-stream events to the in-memory exchange ledger."""
        event_name = str(event.get("e") or "")
        if event_name == "ACCOUNT_UPDATE":
            account = event.get("a") if isinstance(event.get("a"), dict) else {}
            balances = account.get("B") if isinstance(account.get("B"), list) else []
            for item in balances:
                if not isinstance(item, dict) or str(item.get("a") or "").upper() != "USDT":
                    continue
                cached = self.real.cached_usdt_balance() or {"asset": "USDT"}
                if item.get("wb") not in (None, ""):
                    cached["balance"] = str(item.get("wb"))
                if item.get("cw") not in (None, ""):
                    cached["crossWalletBalance"] = str(item.get("cw"))
                self.real.last_balance = cached
                self.real.last_balance_at = time.monotonic()
                break

            # ACCOUNT_UPDATE contains only changed position rows (P); an empty P
            # means there was no position change. Never clear other cached symbols.
            positions = account.get("P") if isinstance(account.get("P"), list) else []
            changed_symbols: set[str] = set()
            for row in positions:
                if not isinstance(row, dict):
                    continue
                symbol = normalize_symbol(str(row.get("s") or row.get("symbol") or ""))
                if symbol:
                    changed_symbols.add(symbol)
                await self._apply_account_update_position(row)
            for symbol in changed_symbols:
                self._refresh_real_position_cache(symbol)
            await self._sync_ws_symbols()
            await self._autostop_check()
            await self._save_active_state()
            return

        if event_name == "ALGO_UPDATE":
            payload = event.get("o") if isinstance(event.get("o"), dict) else (event.get("a") if isinstance(event.get("a"), dict) else event)
            symbol = normalize_symbol(str(payload.get("s") or payload.get("symbol") or ""))
            algo_id = payload.get("aid", payload.get("algoId"))
            if algo_id is None:
                # Legacy payload fallback; `ai` in the current schema is the
                # triggered matching-engine order ID, not the Algo ID.
                algo_id = payload.get("legacyAlgoId")
            client_id = str(payload.get("caid", payload.get("clientAlgoId", "")) or "")
            status = str(payload.get("X") or payload.get("algoStatus") or payload.get("status") or "UNKNOWN").upper()
            for trade in list(self.active_trades.values()):
                if trade.pair != symbol or not trade.real_enabled:
                    continue
                matched_kind = None
                if (client_id and client_id == trade.sl_client_algo_id) or (algo_id is not None and trade.sl_algo_id is not None and str(algo_id) == str(trade.sl_algo_id)):
                    trade.sl_algo_status = status
                    matched_kind = "SL"
                    if status in {"TRIGGERED", "FINISHED"}: trade.protection_exit_reason = "STOP_LOSS"
                if (client_id and client_id == trade.tp_client_algo_id) or (algo_id is not None and trade.tp_algo_id is not None and str(algo_id) == str(trade.tp_algo_id)):
                    trade.tp_algo_status = status
                    matched_kind = "TP"
                    if status in {"TRIGGERED", "FINISHED"}: trade.protection_exit_reason = "TAKE_PROFIT"
                if status == "REJECTED" and matched_kind:
                    reject_message = str(payload.get("rm") or payload.get("rejectReason") or "Binance rejected conditional Algo Order")
                    trade.protection_state = "ERROR_REJECTED"
                    trade.protection_error = f"{matched_kind} rejected: {reject_message}"[:500]
                    await self.reply(
                        f"🚨 BINANCE {matched_kind} DITOLAK • {trade.pair} {trade.direction}\n"
                        f"Detail: {reject_message[:300]}\n"
                        "Posisi masih terbuka tanpa proteksi lengkap. Periksa posisi dan tutup manual bila perlu."
                    )
                await self._save_active_state()
            return

        if event_name != "ORDER_TRADE_UPDATE":
            if event_name == "MARGIN_CALL":
                await self.reply("🚨 BINANCE MARGIN_CALL diterima. Periksa posisi dan risiko akun LIVE segera.")
            return

        order = event.get("o") if isinstance(event.get("o"), dict) else {}
        symbol = normalize_symbol(str(order.get("s") or ""))
        client_id = str(order.get("c") or "")
        if not symbol:
            return

        # Always maintain the exchange open-order projection first. The event is
        # useful even if the current /try session has no matching local trade row.
        trade = await self._upsert_exchange_order_event(order)
        if trade is None:
            await self._sync_ws_symbols()
            return

        if trade.entry_order_created_at is None:
            trade.entry_order_created_at = self._exchange_event_time(order)
            trade.expires_at = trade.entry_order_created_at + timedelta(seconds=ENTRY_ORDER_TTL_SECONDS)

        status = str(order.get("X") or "UNKNOWN").upper()
        execution = str(order.get("x") or "UNKNOWN").upper()
        order_id = order.get("i")
        try:
            order_id = int(order_id) if order_id not in (None, "") else None
        except (TypeError, ValueError):
            order_id = None
        try:
            cumulative = Decimal(str(order.get("z") or "0"))
        except (InvalidOperation, ValueError):
            cumulative = Decimal("0")
        try:
            average = Decimal(str(order.get("ap") or "0"))
            last_price = Decimal(str(order.get("L") or "0"))
        except (InvalidOperation, ValueError):
            average = Decimal("0")
            last_price = Decimal("0")
        fill_price = average if average > 0 else (last_price if last_price > 0 else trade.fill_price or trade.entry)
        event_key = f"{order_id}:{execution}:{status}:{cumulative}:{order.get('t', '')}"
        if trade.last_ws_event_key == event_key:
            return
        trade.last_ws_event_key = event_key
        trade.entry_order_status = status
        if order_id is not None:
            trade.entry_order_id = order_id
        if client_id:
            trade.entry_client_order_id = client_id

        terminal = status in {"FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}
        trade.entry_order_terminal = terminal
        if cumulative > 0:
            if trade.expires_at <= now_utc() and self._binance_cooldown_remaining() > 0 and not trade.late_fill_during_api_cooldown:
                trade.late_fill_during_api_cooldown = True
                await self._record_event(
                    "LATE_FILL_DURING_API_COOLDOWN", trade,
                    order_status=status, executed_qty=decimal_to_str(cumulative),
                    note="GTC order dibiarkan berjalan ketika REST banned; fill diterima melalui User Data WS setelah batas waktu satu jam.",
                )
            trade.entry_executed_qty = cumulative
            trade.quantity = cumulative
            trade.fill_price = fill_price
            trade.actual_notional = cumulative * fill_price
            trade.margin_usdt = trade.actual_notional / Decimal(max(1, trade.leverage))
            if trade.status == "PENDING":
                await self._mark_filled(
                    trade, fill_price, f"BINANCE_USER_WS_{status}",
                    filled_at=self._exchange_event_time(order),
                )
            else:
                trade.status = "FILLED"
                await self._mark_filled(
                    trade, fill_price, f"BINANCE_USER_WS_PARTIAL_{status}",
                    filled_at=self._exchange_event_time(order),
                )
                await self._record_event(
                    "ORDER_CUMULATIVE_FILL_UPDATE", trade,
                    order_status=status, execution_type=execution,
                    executed_qty=decimal_to_str(cumulative), average_price=decimal_to_str(fill_price),
                )
            trade.real_state = (
                "REAL_FILLED" if status == "FILLED" else
                "REAL_PARTIAL_TERMINAL" if terminal else
                ("REAL_EXCHANGE_ADOPTED_PARTIAL" if trade.exchange_adopted else "REAL_PARTIALLY_FILLED")
            )
            if terminal:
                trade.pending_cancel_reason = None
            elif trade.expires_at <= now_utc() and not trade.preserve_gtc_after_cooldown:
                trade.pending_cancel_reason = trade.pending_cancel_reason or "ENTRY_LIMIT_TIMEOUT_1H"
            await self._save_active_state()
            log.info("[USER WS] %s %s order=%s status=%s cumQty=%s avg=%s", symbol, trade.direction, order_id, status, cumulative, fill_price)
            return

        if terminal and trade.status == "PENDING":
            trade.pending_cancel_reason = None
            result = "EXPIRED" if status in {"EXPIRED", "EXPIRED_IN_MATCH"} else "CANCELED"
            await self._finalize_trade(trade, result, None, f"BINANCE_USER_WS_{status}")
        else:
            await self._save_active_state()

    async def _sync_ws_symbols(self) -> None:
        trade_symbols = {
            t.pair for t in self.active_trades.values()
            if t.status in {"PENDING", "FILLED"}
        }
        trade_symbols.update(symbol for symbol, _side in self.exchange_positions)
        trade_symbols.update(
            normalize_symbol(str(order.get("symbol") or ""))
            for order in self.exchange_open_orders.values()
            if order.get("symbol")
        )
        scanner_symbols = set(self._scanner_symbols) if self.scan_enabled else set()
        exit_symbols = {
            t.pair for t in self.active_trades.values()
            if t.status == "FILLED" and not t.exchange_adopted
            and self._ema_exit_periods(t) is not None
            and (
                not t.real_enabled
                or t.protection_state in {"ACTIVE", "PARTIAL", "INVALID_RISK", "ERROR_REJECTED"}
                or t.ema_exit_state in {"INITIALIZING", "WAITING_API", "INIT_RETRY", "BACKFILL_PENDING", "ACTIVE", "ACTIVE_WITH_GAP"}
            )
            and t.ema_exit_state not in {
                "DISABLED", "UNAVAILABLE", "EXIT_PENDING", "CLOSE_SUBMITTING",
                "CLOSE_VERIFY", "CLOSE_UNKNOWN", "CLEANUP_WAIT_API",
                "SUPERSEDED_BY_PROTECTION",
            }
        }
        desired = trade_symbols | scanner_symbols | exit_symbols
        await self.ws.set_symbols(
            trade_symbols, scanner_symbols=scanner_symbols, exit_symbols=exit_symbols
        )
        for key in list(self._live_signal_candles):
            _tf, pair = key.split(":", 1)
            if pair not in desired:
                self._live_signal_candles.pop(key, None)
                self._live_signal_last_eval.pop(key, None)
                self._live_signal_seen.pop(key, None)

    # ------------------------- Telegram commands -------------------------
    async def _set_real(self, command: str) -> None:
        arg = command.split(maxsplit=1)[1].strip().lower() if len(command.split(maxsplit=1)) > 1 else ""
        if arg == "on":
            if not REAL_EXECUTION_ENABLED:
                await self.reply("❌ REAL dikunci oleh REAL_EXECUTION_ENABLED=0.")
                return
            if not self.real.configured:
                await self.reply("❌ BINANCE_API_KEY / BINANCE_API_SECRET belum lengkap.")
                return
            # Restore only the REAL equity ledger, never scanner/trade state. The
            # first /real on takes one wallet-balance baseline; later process starts
            # retain that baseline and the locally accumulated realized PnL.
            await self._load_real_guard_checkpoint()
            # Synchronize against Binance immediately before the first signed
            # account/preflight calls. REST timestamp generation also refreshes
            # this offset periodically and retries one -1021 after resync.
            await self.real.sync_server_time(force=True, samples=3)
            account_snapshot = await self.real.get_account_info()
            all_open_orders = await self.real.get_open_orders(None)
            await self.real.get_account_config()
            balance_baseline_created = False
            if self._real_equity_baseline is None:
                balance_rows = await self.real.get_futures_balance()
                usdt_row = next((x for x in balance_rows if str(x.get("asset") or "").upper() == "USDT"), None)
                try:
                    wallet = Decimal(str((usdt_row or {}).get("balance") or "0"))
                except (InvalidOperation, TypeError, ValueError):
                    wallet = Decimal("0")
                if not wallet.is_finite() or wallet <= 0:
                    raise RuntimeError("Baseline USDT Futures balance tidak valid; REAL tidak diaktifkan.")
                self._real_equity_baseline = wallet
                self._realized_since_baseline = Decimal("0")
                self._equity_peak = wallet
                self._equity_last = wallet
                self._autostop_source = "REAL"
                self._autostop_data_hold = False
                balance_baseline_created = True
            adopted_positions, adopted_orders = await self._adopt_exchange_snapshot(
                account_snapshot, all_open_orders, source="REAL_ON_ACCOUNT_SNAPSHOT"
            )
            # Even with the entry scanner OFF, adopted live exposure/orders still
            # need market prices for floating-PnL guard and position monitoring.
            if adopted_positions or adopted_orders:
                await self.ws.start()
                await self._sync_ws_symbols()
            # Private stream becomes the live source of incremental order/position
            # updates. No REST request is made by /trade after this point.
            await self.user_ws.start()
            self._last_user_ws_reconnect_count = self.user_ws.reconnect_count
            self.real_mode = True
            if self.autostop_percent is not None:
                await self._autostop_check(force=True)
            await self._save_runtime_settings()
            await self._save_real_guard_checkpoint()
            await self.reply(
                "🟢 REAL ON. Entry baru: level disimpan lokal, MARKET order dikirim ke Binance USD-M Futures LIVE saat harga menyentuh level.\n"
                f"Live snapshot : {adopted_positions} posisi · {adopted_orders} open order dimuat dari Binance.\n"
                f"Equity baseline: {'baru diambil dari Binance' if balance_baseline_created else 'checkpoint lokal dipertahankan'}; selanjutnya dihitung lokal dari PnL + floating.\n"
                f"Private User Data WS: {self.user_ws.status}. ORDER_TRADE_UPDATE + ACCOUNT_UPDATE menjaga /trade realtime; /trade tidak memanggil REST.\n"
                "Posisi/order yang sudah ada tetap dikelola walaupun /real off."
            )
        elif arg == "off":
            self.real_mode = False
            await self.reply("🟡 REAL OFF untuk entry baru. Pending lokal tidak akan dieksekusi (dibatalkan saat tersentuh); posisi REAL yang sudah ada tetap direkonsiliasi dan dikelola sampai selesai.")
        else:
            await self.reply(f"REAL: {'ON' if self.real_mode else 'OFF'}\nGunakan /real on atau /real off.")

    async def _handle_length(self, text: str) -> None:
        parts = str(text or "").split()
        if len(parts) == 1:
            await self.reply(
                f"VOLUMATIC TREND LENGTH: {self.trend_length}\n"
                f"Default fresh session: {volumatic.DEFAULT_TREND_LENGTH}\n"
                "Input: semua bilangan bulat positif. Data Binance per request maksimal 1.500 candle; nilai sangat besar dapat menunggu warm-up dan belum menghasilkan sinyal.\n\n"
                "Gunakan /length <angka>, misalnya /length 89 atau /length 500.\n"
                "Length berlaku untuk EMA-SWMA dan EMA close Volumatic Trend; ATR tetap 200.\n"
                "Perubahan berlaku pada sinyal baru dan scanner akan membangun ulang konteks candle aktif."
            )
            return
        if len(parts) != 2:
            await self.reply("Gunakan /length <angka> dengan bilangan bulat positif, misalnya /length 89.")
            return
        try:
            selected = volumatic.normalize_trend_length(parts[1])
        except ValueError as exc:
            await self.reply(f"❌ {exc}")
            return
        old = self.trend_length
        if selected == old:
            await self.reply(f"Trend length sudah {selected}.")
            return

        self.trend_length = selected
        # Invalidate cached contexts and restart scanner cursor so every symbol
        # in the next pass is evaluated with the same new indicator length.
        self._live_signal_candles.clear()
        self._live_signal_last_eval.clear()
        self._live_signal_seen.clear()
        self.last_processed_candles.clear()
        self._scan_resume_cursor = 0
        self._scan_resume_boundary_ms = 0
        self._scan_resume_trend_length = selected
        self._scan_trigger_now = bool(self.scan_enabled and not self._autostop_triggered)
        await self._save_runtime_settings()
        await self.reply(
            f"✅ VOLUMATIC LENGTH UPDATED\n\n"
            f"Previous : {old}\n"
            f"Current  : {selected}\n"
            f"EMA1     : EMA-SWMA({selected})\n"
            f"EMA2     : EMA(Close, {selected})\n"
            f"ATR      : {volumatic.ATR_LENGTH} (unchanged)\n\n"
            "Cache indikator dibangun ulang. Pengaturan berlaku untuk sinyal berikutnya.\n"
            "Catatan: /try berikutnya memulai fresh session dengan default 100."
        )
        log.info("[CONFIG] Volumatic length changed: %s -> %s", old, selected)

    async def _handle_h1(self, text: str) -> None:
        parts = str(text or "").split()
        if len(parts) == 1:
            state = "ON" if self.h1_enabled else "OFF"
            await self.reply(
                f"H1 LOGIC: {state}\n\n"
                f"ENTRY FILTER : {state} — sinyal baru wajib lolos directional EMA H1 saat ON.\n"
                f"FORCED CLOSE: {state} — posisi LONG ditutup pada bearish EMA cross H1; "
                "posisi SHORT ditutup pada bullish EMA cross H1 saat ON.\n\n"
                "Gunakan /H1 on atau /H1 off. Saklar ini mengatur kedua logika sekaligus.\n"
                "Saat OFF, EMA H1 tetap diperbarui sebagai baseline agar cross yang terjadi selama OFF "
                "tidak memicu forced close setelah H1 dinyalakan kembali. SL/TP exchange tidak diubah."
            )
            return
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            await self.reply("Gunakan /H1 on atau /H1 off.")
            return

        enabled = parts[1].lower() == "on"
        previous = self.h1_enabled
        if enabled == previous:
            await self.reply(f"H1 LOGIC sudah {'ON' if enabled else 'OFF'}. Kedua logika tidak berubah.")
            return

        self.h1_enabled = enabled
        self._h1_toggle_generation += 1
        # A scan must not mix H1 filter modes inside one batch. Re-scan the
        # current universe with the new setting; no active trade is modified.
        self._signal_retries.clear()
        self._live_signal_candles.clear()
        self._live_signal_last_eval.clear()
        self._live_signal_seen.clear()
        self.last_processed_candles.clear()
        self._scan_resume_cursor = 0
        self._scan_resume_boundary_ms = 0
        self._scan_trigger_now = bool(self.scan_enabled and not self._autostop_triggered)
        await self._save_runtime_settings()

        state = "ON" if enabled else "OFF"
        if enabled:
            for trade in list(self.active_trades.values()):
                if trade.status == "FILLED" and not trade.exchange_adopted:
                    self._schedule_ema_exit_initialization(trade)
            detail = (
                "Filter entry H1 kembali wajib lolos. Forced close mengikuti cross H1 yang "
                "terkonfirmasi setelah baseline EMA terbaru; cross yang terjadi ketika OFF diabaikan."
            )
        else:
            # An inverse cross that has been detected but whose MARKET close has
            # not been submitted yet can still be cancelled by this switch.
            # Once submission starts, retain reconciliation to avoid duplicate or
            # ambiguous close orders.
            cancelled_pending = 0
            for trade in self.active_trades.values():
                if (
                    trade.status == "FILLED"
                    and trade.ema_exit_state == "EXIT_PENDING"
                    and not trade.ema_exit_client_order_id
                    and int(trade.ema_exit_attempts or 0) == 0
                ):
                    trade.ema_exit_state = "ACTIVE" if trade.ema_exit_last_fast is not None and trade.ema_exit_last_slow is not None else "UNINITIALIZED"
                    trade.ema_exit_trigger_close_ms = None
                    trade.ema_exit_trigger_price = None
                    trade.ema_exit_reason = None
                    trade.ema_exit_last_error = "Forced close H1 dibatalkan karena /H1 off sebelum MARKET close dikirim."
                    cancelled_pending += 1
            detail = (
                "Filter entry H1 dinonaktifkan. Cross H1 tidak lagi memicu forced close. "
                "Close yang sudah mulai dikirim ke Binance tetap direkonsiliasi; SL/TP tetap aktif."
            )
            if cancelled_pending:
                detail += f"\nPending close yang belum dikirim dibatalkan: {cancelled_pending}."

        await self._save_active_state()
        await self._sync_ws_symbols()
        await self.reply(
            f"✅ H1 LOGIC {state}\n\n"
            f"ENTRY FILTER : {state}\nFORCED CLOSE: {state}\n\n{detail}"
        )
        log.info("[CONFIG] H1 master switch: %s -> %s", "ON" if previous else "OFF", state)

    async def _handle_ema(self, text: str) -> None:
        parts = str(text or "").split()
        if len(parts) == 1:
            active = ema_filter_label(self.ema_filter_length)
            if isinstance(self.ema_filter_length, tuple):
                rule = (
                    f"BLUE/LONG hanya jika EMA{self.ema_filter_length[0]} > EMA{self.ema_filter_length[1]}.\n"
                    f"RED/SHORT hanya jika EMA{self.ema_filter_length[0]} < EMA{self.ema_filter_length[1]}."
                )
            elif self.ema_filter_length is not None:
                rule = (
                    f"BLUE/LONG hanya jika close candle H4 terakhir > EMA{self.ema_filter_length}.\n"
                    f"RED/SHORT hanya jika close candle H4 terakhir < EMA{self.ema_filter_length}."
                )
            else:
                rule = "Filter EMA OFF: BLUE dan RED tidak memakai filter EMA."
            await self.reply(
                f"Directional EMA filter: {active}\n{rule}\n"
                "Pair yang masih punya posisi/order aktif diabaikan sampai SL/TP atau close manual.\n\n"
                "Single: /EMA 89 (close H4 dibandingkan EMA89) atau /EMA off.\n"
                "Dual: /EMA 89 95 (LONG jika EMA89 > EMA95; SHORT jika EMA89 < EMA95).\n"
                "Dua angka otomatis diurutkan: periode lebih kecil menjadi EMA fast, lebih besar menjadi EMA slow.\n\n"
                "BTCUSDT.P H4 juga menjadi filter wajib: LONG hanya jika BTC EMA fast > slow; SHORT hanya jika fast < slow. "
                "Filter BTC memakai periode dual /EMA yang sama. Mode single atau OFF tidak dapat membentuk pasangan BTC fast/slow, sehingga entry baru diblokir."
            )
            return
        if len(parts) not in (2, 3):
            await self.reply("Gunakan /EMA <angka>, /EMA off, atau dua angka seperti /EMA 89 95.")
            return
        raw_selection: Any = parts[1] if len(parts) == 2 else (parts[1], parts[2])
        try:
            selected = volumatic.normalize_ema_filter_length(raw_selection)
        except ValueError as exc:
            await self.reply(f"❌ {exc}\nContoh: /EMA 89, /EMA 89 95, atau /EMA off.")
            return
        old = self.ema_filter_length
        self.ema_filter_length = selected
        # Queued signals carry their old EMA snapshot; discard them when periods change.
        self._signal_retries.clear()
        # A new EMA configuration invalidates cached signal and scan cursors.
        self._live_signal_candles.clear()
        self._live_signal_last_eval.clear()
        self._live_signal_seen.clear()
        self.last_processed_candles.clear()
        self._scan_resume_cursor = 0
        self._scan_resume_boundary_ms = 0
        self._scan_trigger_now = bool(self.scan_enabled and not self._autostop_triggered)
        await self._save_runtime_settings()
        old_text = ema_filter_label(old)
        new_text = ema_filter_label(selected)
        if selected is None:
            rule = "Filter arah dinonaktifkan; BLUE dapat membuka LONG dan RED dapat membuka SHORT."
        elif isinstance(selected, tuple):
            rule = (
                f"LONG hanya jika EMA{selected[0]} > EMA{selected[1]}.\n"
                f"SHORT hanya jika EMA{selected[0]} < EMA{selected[1]}.\n"
                "Jika kedua EMA sama, entry baru ditolak."
            )
        else:
            rule = (
                f"LONG hanya jika close candle H4 terakhir > EMA{selected}.\n"
                f"SHORT hanya jika close candle H4 terakhir < EMA{selected}.\n"
                "Close tepat pada EMA menolak entry baru."
            )
        btc_rule = (
            f"BTCUSDT.P H4: LONG jika EMA{selected[0]} > EMA{selected[1]}; "
            f"SHORT jika EMA{selected[0]} < EMA{selected[1]}."
            if isinstance(selected, tuple)
            else "BTCUSDT.P H4: filter wajib membutuhkan /EMA FAST SLOW. Mode single/OFF akan memblokir entry baru."
        )
        await self.reply(
            f"✅ EMA FILTER UPDATED\n\nPrevious : {old_text}\nCurrent  : {new_text}\n\n{rule}\n"
            f"{btc_rule}\n"
            "Filter H1 (pair yang sama, periode /EMA yang sama, aturan arah sama dengan H4) juga wajib lolos.\n"
            "Kedua filter H4 memakai candle yang sudah ditutup.\n"
            "Sinyal ketika pair masih memiliki posisi/order aktif diabaikan sampai SL/TP atau close manual."
        )
        log.info("[CONFIG] EMA directional filter: %s -> %s", old_text, new_text)

    def _reset_scan_after_supertrend_change(self) -> None:
        self._live_signal_candles.clear()
        self._live_signal_last_eval.clear()
        self._live_signal_seen.clear()
        self._signal_retries.clear()
        self.last_processed_candles.clear()
        self._scan_resume_cursor = 0
        self._scan_resume_boundary_ms = 0
        self._scan_trigger_now = bool(self.scan_enabled and not self._autostop_triggered)

    def _reset_scan_after_rr_change(self) -> None:
        """Discard only unsubmitted signal contexts; active trades keep saved RR bounds."""
        self._live_signal_candles.clear()
        self._live_signal_last_eval.clear()
        self._live_signal_seen.clear()
        self._signal_retries.clear()
        self.last_processed_candles.clear()
        self._scan_resume_cursor = 0
        self._scan_resume_boundary_ms = 0
        self._scan_trigger_now = bool(self.scan_enabled and not self._autostop_triggered)

    async def _handle_rr_min(self, text: str) -> None:
        parts = str(text or "").split()
        if len(parts) == 1:
            await self.reply(
                f"RR_MIN: {self.rr_min}R\nRR_MAX: {self.rr_max}R\n\n"
                "RR_MIN adalah TP dasar/fallback dan batas bawah pencarian liquidity pool.\n"
                "Gunakan /RR_MIN <angka>, contoh /RR_MIN 3 atau /RR_MIN 2.5.\n"
                f"Nilai harus > 0 dan tidak boleh melebihi RR_MAX ({self.rr_max}).\n"
                "Perubahan berlaku untuk sinyal baru; pending/filled trade yang sudah ada tetap memakai RR saat dibuat."
            )
            return
        if len(parts) != 2:
            await self.reply("Gunakan /RR_MIN <angka>, contoh /RR_MIN 3 atau /RR_MIN 2.5.")
            return
        try:
            selected = Decimal(str(volumatic.normalize_rr_value(parts[1], "RR_MIN")))
        except ValueError as exc:
            await self.reply(f"❌ {exc}")
            return
        if selected > self.rr_max:
            await self.reply(f"❌ RR_MIN {selected} tidak boleh lebih besar dari RR_MAX {self.rr_max}.")
            return
        old = self.rr_min
        if selected == old:
            await self.reply(f"RR_MIN sudah {selected}R.")
            return
        self.rr_min = selected
        self._reset_scan_after_rr_change()
        await self._save_runtime_settings()
        await self.reply(
            f"✅ RR_MIN UPDATED\n\nPrevious: {old}R\nCurrent : {self.rr_min}R\n"
            f"RR_MAX  : {self.rr_max}R\n\n"
            f"TP dasar/fallback sekarang 1:{self.rr_min}. Liquidity pool dicari pada rentang "
            f"1:{self.rr_min} sampai 1:{self.rr_max}, lalu dipilih pool paling jauh yang masih masuk batas.\n"
            "Trade yang sudah dibuat tidak berubah."
        )
        log.info("[CONFIG] RR_MIN changed: %s -> %s | RR_MAX=%s", old, self.rr_min, self.rr_max)

    async def _handle_rr_max(self, text: str) -> None:
        parts = str(text or "").split()
        if len(parts) == 1:
            await self.reply(
                f"RR_MIN: {self.rr_min}R\nRR_MAX: {self.rr_max}R\n\n"
                "RR_MAX adalah batas maksimal reward untuk memperpanjang TP ke liquidity pool.\n"
                "Gunakan /RR_MAX <angka>, contoh /RR_MAX 4 atau /RR_MAX 4.5.\n"
                f"Nilai harus >= RR_MIN ({self.rr_min}) dan maksimal {volumatic.MAX_RR_VALUE:g}.\n"
                "Perubahan berlaku untuk sinyal baru; pending/filled trade yang sudah ada tetap memakai RR saat dibuat."
            )
            return
        if len(parts) != 2:
            await self.reply("Gunakan /RR_MAX <angka>, contoh /RR_MAX 4 atau /RR_MAX 4.5.")
            return
        try:
            selected = Decimal(str(volumatic.normalize_rr_value(parts[1], "RR_MAX")))
        except ValueError as exc:
            await self.reply(f"❌ {exc}")
            return
        if selected < self.rr_min:
            await self.reply(f"❌ RR_MAX {selected} tidak boleh lebih kecil dari RR_MIN {self.rr_min}.")
            return
        old = self.rr_max
        if selected == old:
            await self.reply(f"RR_MAX sudah {selected}R.")
            return
        self.rr_max = selected
        self._reset_scan_after_rr_change()
        await self._save_runtime_settings()
        await self.reply(
            f"✅ RR_MAX UPDATED\n\nPrevious: {old}R\nCurrent : {self.rr_max}R\n"
            f"RR_MIN  : {self.rr_min}R\n\n"
            f"TP dasar/fallback tetap 1:{self.rr_min}. Liquidity pool dapat memperpanjang TP hingga "
            f"maksimal 1:{self.rr_max}; dipilih pool paling jauh yang masih masuk rentang.\n"
            "Trade yang sudah dibuat tidak berubah."
        )
        log.info("[CONFIG] RR_MAX changed: %s -> %s | RR_MIN=%s", old, self.rr_max, self.rr_min)

    async def _handle_atr(self, text: str) -> None:
        parts = str(text or "").split()
        if len(parts) == 1:
            await self.reply(
                f"SUPERTREND M15 ATR LENGTH: {self.supertrend_atr_length}\n"
                f"Default: {volumatic.SUPERTREND_ATR_LENGTH}\n\n"
                "Gunakan /ATR <angka>, contoh /ATR 14.\n"
                f"Rentang: {volumatic.MIN_SUPERTREND_ATR_LENGTH}–{volumatic.MAX_SUPERTREND_ATR_LENGTH}.\n"
                "Ini hanya mengubah ATR untuk garis SuperTrend/SL M15; ATR internal Volumatic tetap 200."
            )
            return
        if len(parts) != 2:
            await self.reply("Gunakan /ATR <angka> dengan bilangan bulat positif, contoh /ATR 14.")
            return
        try:
            selected = volumatic.normalize_supertrend_atr_length(parts[1])
        except ValueError as exc:
            await self.reply(f"❌ {exc}")
            return
        old = self.supertrend_atr_length
        if selected == old:
            await self.reply(f"SuperTrend ATR Length sudah {selected}.")
            return
        self.supertrend_atr_length = selected
        self._reset_scan_after_supertrend_change()
        await self._save_runtime_settings()
        await self.reply(
            f"✅ SUPERTREND ATR LENGTH UPDATED\n\nPrevious : {old}\nCurrent  : {selected}\n"
            f"Factor   : {self.supertrend_factor:g}\n"
            "History M15 dan cursor scanner disesuaikan. Berlaku untuk sinyal/order baru; SL/TP trade aktif tidak berubah."
        )
        log.info("[CONFIG] SuperTrend M15 ATR Length changed: %s -> %s", old, selected)

    async def _handle_factor(self, text: str) -> None:
        parts = str(text or "").split()
        if len(parts) == 1:
            await self.reply(
                f"SUPERTREND M15 FACTOR: {self.supertrend_factor:g}\n"
                f"Default: {volumatic.SUPERTREND_FACTOR:g}\n\n"
                "Gunakan /Factor <angka>, contoh /Factor 3 atau /Factor 2.5.\n"
                "Desimal boleh memakai titik atau koma. Rentang: lebih besar dari 0 sampai 1000."
            )
            return
        if len(parts) != 2:
            await self.reply("Gunakan /Factor <angka>, contoh /Factor 3 atau /Factor 2.5.")
            return
        try:
            selected = volumatic.normalize_supertrend_factor(parts[1])
        except ValueError as exc:
            await self.reply(f"❌ {exc}")
            return
        old = self.supertrend_factor
        if selected == old:
            await self.reply(f"SuperTrend Factor sudah {selected:g}.")
            return
        self.supertrend_factor = selected
        self._reset_scan_after_supertrend_change()
        await self._save_runtime_settings()
        await self.reply(
            f"✅ SUPERTREND FACTOR UPDATED\n\nPrevious : {old:g}\nCurrent  : {selected:g}\n"
            f"ATR Length: {self.supertrend_atr_length}\n"
            "Cache/cursor scanner dibangun ulang. Berlaku untuk sinyal/order baru; SL/TP trade aktif tidak berubah."
        )
        log.info("[CONFIG] SuperTrend M15 Factor changed: %s -> %s", old, selected)

    async def _handle_timeframe(self, text: str) -> None:
        await self.reply(
            "Timeframe strategi dikunci agar sama dengan backtest:\n"
            "EMA Trend : H4 (candle terakhir yang sudah close)\n"
            "Volumatic: M15 (bubble terkonfirmasi saat candle close)\n"
            "Supertrend: M15 (SL dari candle sinyal).\n"
            "Perintah /timeframe tidak mengubah timeframe strategi ini."
        )

    async def _handle_scan(self, text: str) -> None:
        parts = text.split()
        action = parts[1].lower() if len(parts) > 1 else "status"

        if action == "status":
            slots = self._active_slot_projection()
            pending = sum(1 for item in slots.values() if item.get("pending"))
            filled = sum(1 for item in slots.values() if item.get("filled"))
            state = self._scanner_state_label()
            await self.reply(
                "VOLUMATIC SCAN STATUS\n\n"
                f"State: {state}\n"
                f"Signal: M15 Volumatic + H4 EMA + M15 Supertrend\n"
                f"Pending entry LIMIT: {pending}\nFilled positions: {filled}\n"
                f"Slots: {len(slots)}/{self.max_active_trades}\n"
                f"TP RR: base 1:{self.rr_min} · liquidity cap 1:{self.rr_max}\n"
                "Rule: pending LIMIT memakai slot, bukan alasan pause scanner.\n"
                f"Schedule: {', '.join(self.scan_schedule_slots) if self.scan_schedule_slots else '-'}"
            )
            return

        if action == "schedule":
            if len(parts) == 2:
                lines = [
                    "╔════ VOLUMATIC / SCAN SCHEDULE ════╗",
                    "TIMEZONE : ASIA/JAKARTA (WIB)",
                    "MODE     : delayed activation → continuous ON",
                    "──────────────────────────────────",
                ]
                for number, hhmm in SCAN_SCHEDULE_SLOTS:
                    selected = " ✓" if hhmm in self.scan_schedule_slots else ""
                    lines.append(f"{number}. {hhmm} WIB{selected}")
                lines += [
                    "──────────────────────────────────",
                    f"SCANNER  : {self._scanner_state_label()}",
                    f"SELECTED : {', '.join(self.scan_schedule_slots) if self.scan_schedule_slots else '-'}",
                    "PILIH    : /scan schedule 1,3,6",
                    "CANCEL   : /scan schedule off",
                    "INFO     : setelah salah satu slot terpilih tercapai, scanner ON terus sampai /scan off.",
                    "INFO     : slot 12:00 adalah slot tambahan; enam slot lainnya mengikuti close H4 WIB.",
                ]
                await self.reply("\n".join(lines))
                return

            selection = " ".join(parts[2:]).strip().lower()
            if selection in {"off", "clear", "none"}:
                self.scan_schedule_slots = ()
                self.scan_schedule_waiting = False
                self._scan_schedule_last_fired_minute = None
                await self.reply(
                    "🟡 SCAN SCHEDULE CLEARED\n"
                    f"Scanner saat ini: {'ON' if self.scan_enabled else 'OFF'}. "
                    "Jika scanner sudah ON, gunakan /scan off untuk menghentikannya."
                )
                return

            raw_numbers = [x for x in re.split(r"[,;\s]+", selection) if x]
            if not raw_numbers:
                await self.reply("Gunakan /scan schedule 1,3,6 atau /scan schedule off.")
                return
            invalid = [x for x in raw_numbers if x not in SCAN_SCHEDULE_BY_NUMBER]
            if invalid:
                await self.reply(
                    "❌ Nomor jadwal tidak valid: " + ", ".join(invalid) +
                    "\nPilihan hanya 1 sampai 7. Kirim /scan schedule untuk melihat daftar."
                )
                return

            selected_numbers = sorted(set(raw_numbers), key=lambda x: int(x))
            selected_slots = tuple(SCAN_SCHEDULE_BY_NUMBER[x] for x in selected_numbers)
            if self._autostop_triggered:
                await self.reply("⛔ AUTOSTOP sedang TRIGGERED. Reset pengaman sebelum memasang jadwal scanner.")
                return

            # Selecting a schedule explicitly arms a delayed start and forces
            # scanner OFF now. Once the first selected slot is reached, it
            # remains ON continuously until a manual /scan off.
            self.scan_enabled = False
            self._scan_trigger_now = False
            self._scanner_symbols.clear()
            self.scan_schedule_slots = selected_slots
            self.scan_schedule_waiting = True
            self.scan_schedule_activated_at = None
            self._scan_schedule_last_fired_minute = None
            await self.reply(
                "⏳ VOLUMATIC / SCAN SCHEDULE ARMED\n\n"
                f"SELECTED  : {', '.join(selected_slots)} WIB\n"
                f"SIGNAL TF : M15 • EMA TF: H4\n"
                "SCANNER   : OFF — menunggu slot terpilih\n"
                "TRIGGER   : slot terpilih pertama yang tercapai\n"
                "AFTER     : scanner ON terus menerus\n"
                "STOP      : /scan off"
            )
            log.info("[SCAN SCHEDULE] armed | selected=%s | tf=%s", selected_slots, self.timeframe)
            return

        if action == "status":
            schedule_state = (
                f"WAITING ({', '.join(self.scan_schedule_slots)} WIB)"
                if self.scan_schedule_waiting else
                (f"ACTIVATED {self.scan_schedule_activated_at}" if self.scan_schedule_activated_at else "OFF")
            )
            await self.reply(
                f"Scanner: {'ON' if self.scan_enabled else 'OFF'}\n"
                f"Schedule: {schedule_state}\n"
                "Gunakan /scan on, /scan off, atau /scan schedule."
            )
            return

        if action not in {"on", "off"}:
            await self.reply("Gunakan /scan on, /scan off, atau /scan schedule.")
            return

        enable = action == "on"
        if enable and self._autostop_triggered:
            await self.reply(
                "⛔ Scanner masih dikunci oleh AUTOSTOP. Periksa equity/posisi, "
                "jalankan /autostop reset, lalu /scan on untuk mengaktifkan kembali."
            )
            return
        if enable and self.autostop_percent is not None and self._equity_peak is None:
            await self.reply(
                "⏳ AUTOSTOP belum memiliki baseline equity yang valid. Tunggu balance/mark price "
                "tersedia, lalu cek /autostop sebelum mengaktifkan scanner."
            )
            return

        self.scan_enabled = enable
        if enable:
            # Manual ON overrides and clears any pending schedule activation.
            self.scan_schedule_waiting = False
            self.scan_schedule_slots = ()
            self.scan_schedule_activated_at = None
            await self.ws.start()
            self._scan_trigger_now = True
            log.info("[SCAN] explicitly armed by Telegram /scan on | tf=%s | batch=%s", self.timeframe, SCAN_MAX_PAIRS_PER_CYCLE)
            await self.reply(
                f"🟢 Scanner ON • M15 Volumatic / H4 EMA • batch {SCAN_MAX_PAIRS_PER_CYCLE} "
                f"• Confirmed M15 Volumatic bubble + H4 EMA + M15 Supertrend SL / TP liquidity {self.rr_min}R-{self.rr_max}R. Scanner jeda selama entry LIMIT masih terbuka."
            )
        else:
            self._scanner_symbols.clear()
            self._scan_trigger_now = False
            self.scan_schedule_waiting = False
            self.scan_schedule_slots = ()
            self.scan_schedule_activated_at = None
            self._scan_schedule_last_fired_minute = None
            await self.reply(
                "🟡 Scanner OFF. Bubble baru tidak dipindai; posisi dan pending order "
                "tetap dimonitor via WebSocket/maintenance. Jadwal tertunda dibersihkan."
            )
        await self._sync_ws_symbols()
        await self._save_runtime_settings()

    async def _handle_max(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            total, pending_only, filled, live_limits = self._active_slot_counts()
            await self.reply(
                f"Max active slots: {self.max_active_trades}\n"
                f"Occupied: {total}/{self.max_active_trades}\n"
                f"Pending-only: {pending_only} | Filled positions: {filled} | Live LIMIT orders: {live_limits}"
            )
            return
        try: value = int(parts[1])
        except ValueError:
            await self.reply("Gunakan /max angka. Contoh: /max 50")
            return
        if value < 1 or value > 1000:
            await self.reply("/max harus antara 1 sampai 1000.")
            return
        self.max_active_trades = value
        await self._save_runtime_settings()
        await self.reply(f"✅ Max active slots menjadi {value}. Posisi yang sudah ada tidak ditutup otomatis.")

    async def _handle_margin(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            await self.reply(f"Margin target: {self.margin_usdt} USDT")
            return
        value = parse_decimal(parts[1])
        if value < Decimal("0.1"):
            raise ValueError("Margin minimal 0.1 USDT.")
        self.margin_usdt = value
        await self._save_runtime_settings()
        if self.min_margin_enabled:
            await self.reply(
                f"✅ Margin tetap disimpan: {value} USDT. Saat /min ON nilai ini tidak dipakai; "
                "margin dihitung dari minimum order Binance dan leverage."
            )
        else:
            await self.reply(f"✅ Margin entry baru: {value} USDT.")

    async def _handle_leverage(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            await self.reply(f"Leverage target: {self.leverage}x")
            return
        value = int(parts[1])
        if not 1 <= value <= 125:
            raise ValueError("Leverage harus 1–125x; batas pair diverifikasi Binance saat entry REAL.")
        self.leverage = value
        await self._save_runtime_settings()
        await self.reply(f"✅ Leverage untuk entry baru: {value}x.")

    async def _handle_min(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            state = "ON" if self.min_margin_enabled else "OFF"
            mode = (
                "margin minimum otomatis berdasarkan filter quantity/notional Binance"
                if self.min_margin_enabled else f"margin tetap {self.margin_usdt} USDT"
            )
            await self.reply(
                f"Minimum Margin: {state}\nMode: {mode}\nLeverage: {self.leverage}x\n\n"
                "Gunakan /min on atau /min off. Pengaturan berlaku untuk order baru; "
                "order dan posisi yang sudah ada tidak diubah."
            )
            return
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            await self.reply("Gunakan /min on atau /min off.")
            return
        enabled = parts[1].lower() == "on"
        self.min_margin_enabled = enabled
        await self._save_runtime_settings()
        if enabled:
            await self.reply(
                "✅ Minimum Margin ON. Setiap order baru akan memakai quantity terkecil "
                "yang memenuhi LOT_SIZE dan MIN_NOTIONAL Binance pada harga saat eksekusi, "
                f"dengan leverage {self.leverage}x. /margin disimpan tetapi tidak dipakai "
                "sampai /min off. Order aktif tidak berubah."
            )
        else:
            await self.reply(
                f"✅ Minimum Margin OFF. Order baru kembali memakai margin target "
                f"/margin {self.margin_usdt} USDT dengan leverage {self.leverage}x. "
                "Order aktif tidak berubah."
            )

    async def _handle_del(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await self.reply("Gunakan /del PAIR atau /del all untuk membatalkan pending order. Posisi FILLED tidak ditutup oleh /del.")
            return
        target = parts[1].strip().upper()
        pending = [t for t in self.active_trades.values() if t.status == "PENDING" and (target == "ALL" or t.pair == normalize_symbol(target))]
        if not pending:
            await self.reply("Tidak ada pending order yang cocok. Gunakan /close untuk menutup posisi FILLED.")
            return
        for trade in pending:
            await self._cancel_pending_trade(trade, "MANUAL_CANCEL")
        unresolved = sum(t.trade_id in self.active_trades for t in pending)
        confirmed = len(pending) - unresolved
        if unresolved:
            await self.reply(
                f"Cancel diproses: {confirmed}/{len(pending)}. {unresolved} order masih berstatus belum pasti di Binance; "
                "slot tetap ditahan dan bot akan mencoba rekonsiliasi ulang. Periksa tab Orders Binance."
            )
        else:
            await self.reply(f"✅ {confirmed} pending order berhasil dibatalkan.")

    async def _handle_close(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await self.reply("Gunakan /close PAIR atau /close all untuk menutup posisi FILLED dan membatalkan pending pada pair tersebut.")
            return
        target = parts[1].strip().upper()
        matches = [t for t in self.active_trades.values() if target == "ALL" or t.pair == normalize_symbol(target)]
        if not matches:
            await self.reply("Tidak ada trade aktif untuk pair tersebut.")
            return
        for trade in matches:
            if trade.status == "PENDING":
                await self._cancel_pending_trade(trade, "MANUAL_CANCEL")
            elif trade.status == "FILLED":
                price = self._current_price(trade.pair)
                if price is None:
                    price = await self.rest.get_price(trade.pair)
                await self._close_trade(trade, price, "MANUAL_CLOSE")

    async def _handle_remove(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await self.reply("Gunakan /remove TRADE_ID atau /remove PAIR untuk menghapus satu record trade dari statistik. Event audit tidak dihapus.")
            return
        key = parts[1].strip().upper()
        index = next((i for i, r in enumerate(self.history_records) if str(r.get("trade_id", "")).upper() == key), None)
        if index is None:
            index = next((i for i in range(len(self.history_records)-1, -1, -1) if str(self.history_records[i].get("pair", "")).upper() == normalize_symbol(key)), None)
        if index is None:
            await self.reply("Record trade tidak ditemukan.")
            return
        removed = self.history_records.pop(index)
        await self._persist_history()
        await self._record_event("HISTORY_RECORD_REMOVED", None, pair=removed.get("pair"), removed_trade_id=removed.get("trade_id"))
        await self.reply(f"✅ Record {removed.get('trade_id')} dihapus dari trade history. Event audit dipertahankan.")

    async def _handle_resetstats(self) -> None:
        stamp = now_local().strftime("%Y%m%d-%H%M%S")
        archive_path = f"{self.session_data_dir}/archive/trades-{stamp}.json"
        archive = json.dumps(self.history_records, ensure_ascii=False, indent=2).encode()
        await self._persist_bytes(archive_path, archive, "volumatic: archive stats")
        count = len(self.history_records)
        self.history_records = []
        await self._persist_history()
        await self.reply(f"✅ Statistik trade direset. {count} record diarsipkan ke {archive_path}. Posisi aktif dan event log tidak dihapus.")

    async def _handle_reset(self, text: str) -> None:
        if len(text.split()) < 2 or text.split()[1].upper() != "CONFIRM":
            await self.reply("⚠️ Reset hanya pencatatan (bukan posisi Binance). Pastikan semua pending/posisi sudah selesai, lalu gunakan /reset CONFIRM.")
            return
        if any(t.status in {"PENDING", "FILLED"} for t in self.active_trades.values()):
            await self.reply("❌ Reset diblokir karena masih ada pending order/posisi aktif. Selesaikan terlebih dahulu.")
            return
        self.history_records = []
        self.history_events = []
        self.notes = []
        await self._persist_history()
        await self._persist_bytes(self.session_notes_path, b"[]", "volumatic: reset session notes")
        await self.reply("✅ Riwayat, event, dan catatan dikosongkan. Snapshot runtime tetap dipertahankan.")

    async def _show_balance(self) -> None:
        if self.real_mode or any(t.real_enabled for t in self.active_trades.values()):
            rows = await self.real.get_futures_balance()
            usdt = next((r for r in rows if str(r.get("asset", "")).upper() == "USDT"), {})
            await self.reply("BINANCE USD-M BALANCE\n\n" + f"Balance: {usdt.get('balance', '-')} USDT\nAvailable: {usdt.get('availableBalance', '-')} USDT\nUnrealized PnL: {usdt.get('crossUnPnl', '-')} USDT")
        else:
            realized = sum((
                Decimal(str(r.get("pnl_usdt") or "0"))
                for r in self.history_records
                if r.get("strategy_name") == volumatic.STRATEGY_NAME and not r.get("real_enabled")
            ), Decimal("0"))
            await self.reply(f"PAPER BALANCE\n\nInitial: {SIM_START_BALANCE} USDT\nRealized PnL: {realized:.4f} USDT\nEquity estimate: {(SIM_START_BALANCE + realized):.4f} USDT")

    async def _api_report(self) -> str:
        calls = API_STATS.window(60.0)
        weight = API_STATS.window_weight(60.0)
        header_age = (
            max(0.0, time.monotonic() - API_STATS.used_weight_at)
            if API_STATS.used_weight_at is not None else None
        )
        if API_STATS.used_weight is None or header_age is None:
            header_text = "-"
            gap_text = "n/a"
        elif header_age <= 60.0:
            header_text = f"{API_STATS.used_weight}/{BINANCE_IP_WEIGHT_LIMIT} ({header_age:.0f}s ago)"
            gap_text = f"~{max(0, API_STATS.used_weight - weight)} weight not explained by this process estimate"
        else:
            header_text = f"STALE {API_STATS.used_weight} ({header_age:.0f}s ago)"
            gap_text = "n/a (header older than 60s)"
        return (
            "BINANCE API (60s)\n"
            f"Requests: {sum(calls.values())}\nEstimated weight: {weight}/{BINANCE_IP_WEIGHT_LIMIT}\n"
            f"Header weight: {header_text}\nIP vs local estimate: {gap_text}\n"
            f"Lifetime requests: {API_STATS.total}\nRate-limit hits: {API_STATS.limit_hits}\n"
            f"Cooldown remaining: {BINANCE_COOLDOWN.remaining():.0f}s\n"
            f"Cooldown reason: {BINANCE_COOLDOWN.reason() or '-'}\n"
        )

    async def _refresh_exchange_open_orders_for_trade_view(self) -> str | None:
        """Refresh live Binance open orders only when the user explicitly asks /trade.

        The User Data WS remains the realtime path. This one-shot REST snapshot is
        a recovery path for missed ORDER_TRADE_UPDATE events or WS reconnect gaps.
        It deliberately skips requests during the global Binance cooldown.
        """
        if not getattr(self.real, "configured", False):
            return "Binance snapshot: API belum dikonfigurasi; menampilkan cache lokal."
        cooldown = self._binance_cooldown_remaining()
        if cooldown > 0:
            return f"Binance snapshot: REST cooldown {cooldown:.0f}s; menampilkan cache WebSocket/lokal."
        try:
            rows = await self.real.get_open_orders(None)
        except BinanceRateLimitError as exc:
            await self._notify_binance_cooldown(exc, source="TRADE VIEW OPEN-ORDER SYNC")
            return f"Binance snapshot gagal karena rate limit; cache WebSocket/lokal dipakai. {str(exc)[:180]}"
        except Exception as exc:
            log.warning("[TRADE VIEW] openOrders snapshot gagal; fallback ke cache | %s", exc)
            return f"Binance snapshot gagal; cache WebSocket/lokal dipakai. {str(exc)[:180]}"

        live_rows = [
            row for row in rows
            if str(row.get("status") or row.get("X") or "NEW").upper()
            in {"NEW", "PARTIALLY_FILLED"}
        ]
        # openOrders is an account-wide snapshot. Replace only the open-order
        # projection, then reconcile/adopt each current exchange order. Existing
        # local pending trades are retained if an order is temporarily absent.
        self.exchange_open_orders.clear()
        for row in live_rows:
            await self._upsert_exchange_order_snapshot(row, source="TRADE_COMMAND_SYNC")
        self.exchange_snapshot_at = now_utc()
        await self._sync_ws_symbols()
        log.info(
            "[TRADE VIEW] Binance openOrders snapshot refreshed | active_orders=%s | adopted_cache=%s",
            len(live_rows), len(self.exchange_open_orders),
        )
        return f"Binance snapshot: {len(live_rows)} open order aktif berhasil disinkronkan."

    async def show_trades(self) -> None:
        snapshot_note = await self._refresh_exchange_open_orders_for_trade_view()
        raw_active = [t for t in self.active_trades.values() if t.status in {"PENDING", "FILLED"}]
        # Build per-pair exchange-order projection from WS/snapshot cache. This
        # is deliberately memory-only: /trade never makes a REST request.
        orders_by_pair_direction: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for raw_order in self.exchange_open_orders.values():
            fields = self._exchange_order_fields(raw_order)
            if not fields or fields["status"] not in {"NEW", "PARTIALLY_FILLED"}:
                continue
            orders_by_pair_direction.setdefault((fields["symbol"], fields["direction"]), []).append(fields)

        # Deduplicate races between ORDER_TRADE_UPDATE and ACCOUNT_UPDATE while
        # keeping a single canonical trade row per pair/direction/status.
        deduped: dict[tuple[str, str, str], Trade] = {}
        for trade in raw_active:
            key = (trade.pair, trade.direction, trade.status)
            current = deduped.get(key)
            score = lambda item: (
                int(bool(item.real_enabled)),
                int(bool(item.entry_order_id or item.entry_client_order_id)),
                int(bool(item.exchange_adopted)),
                item.filled_at.timestamp() if item.filled_at else item.created_at.timestamp(),
            )
            if current is None or score(trade) > score(current):
                deduped[key] = trade
        active = list(deduped.values())
        represented_directions = {(t.pair, t.direction) for t in active}
        represented_filled_directions = {
            (t.pair, t.direction) for t in active if t.status == "FILLED"
        }
        slot_count, pending_slots, filled_slots, live_limits = self._active_slot_counts()
        current_ema = ema_filter_label(self.ema_filter_length)
        lines = [
            f"ACTIVE TRADES • {slot_count}/{self.max_active_trades} slots\n"
            f"PENDING SLOTS: {pending_slots} • FILLED SLOTS: {filled_slots} • LIVE LIMITS: {live_limits}\n"
            f"Scanner TF: {self.timeframe} • Length: {self.trend_length} • EMA filter: {current_ema}\n"
            f"SuperTrend M15: ATR {self.supertrend_atr_length} • Factor {self.supertrend_factor:g}\n"
            f"TP RR: MIN {self.rr_min}R • MAX {self.rr_max}R\n"
            + (f"{snapshot_note}\n" if snapshot_note else "")
        ]

        for t in sorted(active, key=lambda x: (x.status, x.pair)):
            price = self._current_price(t.pair)
            matching_orders = orders_by_pair_direction.get((t.pair, t.direction), [])
            matching_order = next((
                item for item in matching_orders
                if (t.entry_client_order_id and item["client_id"] == t.entry_client_order_id)
                or (t.entry_order_id is not None and str(item["order_id"]) == str(t.entry_order_id))
            ), None)
            # If exchange sends an order update before the local IDs are
            # reconciled, pair+direction is still enough to display its live
            # remainder without generating a duplicate trade row.
            if matching_order is None and matching_orders and t.real_enabled:
                matching_order = matching_orders[0]

            if t.status == "PENDING":
                order_detail = ""
                if matching_order:
                    order_detail = (
                        f"\nBinance: {matching_order['status']} • Order ID: {matching_order['order_id'] or '-'}"
                        f"\nClient ID: {matching_order['client_id'] or '-'}"
                        f"\nExecuted: {format_qty(matching_order['executed_qty'])} / {format_qty(matching_order['original_qty'])}"
                        f" • Remaining: {format_qty(matching_order['remaining_qty'])}"
                    )
                elif t.real_intent and not t.real_enabled:
                    order_detail = "\nStatus: " + ("sedang dieksekusi (MARKET)" if t.entry_inflight else "menunggu harga menyentuh level → MARKET order (tanpa order di Binance)")
                elif t.real_enabled:
                    order_detail = f"\nBinance: {t.entry_order_status} • {t.real_state}"
                lines.append(
                    f"⏳ {t.pair} • PENDING {t.direction} ({t.timeframe} • LEN {t.trend_length})\n"
                    f"Signal: {t.signal} | Signal close: {fmt_price(t.signal_close_price or t.entry)} | Entry level (next M15 open): {fmt_price(t.entry)}\n"
                    f"Planned SL: {fmt_price(t.stop_loss or t.reference_sl)} | Planned TP: {fmt_price(t.take_profit)} "
                    f"(RR 1:{fmt_num(t.take_profit_rr, 2)}; {t.take_profit_source}; band {t.rr_min}-{t.rr_max}R; installed after fill)\n"
                    f"SuperTrend source: ATR {t.supertrend_atr_length} • Factor {t.supertrend_factor:g}\n"
                    f"EMA H4 at signal: {ema_filter_values_label(t.ema_filter_length, t.ema_filter_value, t.ema_filter_slow_value)}\n"
                    f"Qty: {format_qty(t.quantity or Decimal('0'))} | Margin: {t.margin_usdt} USDT | Lev: {t.leverage}x\n"
                    f"Sizing: {'MINIMUM AUTO' if t.min_margin_mode else 'FIXED TARGET'}\n"
                    + (
                        "EXCHANGE ACK UNKNOWN: verifikasi clientOrderId; POST tidak akan dikirim ulang\n"
                        if t.real_enabled and t.real_submit_ambiguous
                        else (
                            "Validity: GTC tidak disentuh selama REST cooldown; timeout 1 jam dibatalkan setelah REST pulih\n"
                            if t.real_enabled and t.preserve_gtc_after_cooldown
                            else (
                                f"Expires: {format_wib(t.expires_at)}\n"
                                if t.entry_order_created_at else
                                "Auto-expire: 1 jam setelah Binance menerima LIMIT (timer belum dimulai)\n"
                            )
                        )
                    )
                    + f"Mode: {'REAL' if (t.real_enabled or t.real_intent) else 'PAPER'}"
                    + order_detail
                    + (f"\nCancel/expiry ditunda: {t.pending_cancel_reason}" if t.pending_cancel_reason else "")
                    + ("\n⚠️ Late fill saat Binance REST cooldown; segera direkonsiliasi." if t.late_fill_during_api_cooldown else "")
                )
            else:
                entry = t.fill_price or t.entry
                pnl_pct = None
                pnl_usdt = None
                if price is not None and entry > 0:
                    move = price - entry if t.direction == "LONG" else entry - price
                    pnl_pct = move / entry * Decimal("100")
                    pnl_usdt = move * (t.quantity or Decimal("0"))
                remainder_note = ""
                if matching_order:
                    remainder_note = (
                        f"\nENTRY LIMIT REMAINDER: {matching_order['status']}"
                        f" • Remaining {format_qty(matching_order['remaining_qty'])}"
                        f" • Order ID {matching_order['order_id'] or '-'}"
                    )
                elif t.real_enabled and not t.entry_order_terminal:
                    remainder_note = "\nEntry LIMIT remainder: status menunggu konfirmasi exchange"
                lines.append(
                    f"{'🟢' if t.direction == 'LONG' else '🔴'} {t.pair} • {t.direction} FILLED ({t.timeframe} • LEN {t.trend_length})\n"
                    f"Entry: {fmt_price(entry)} | Live: {fmt_price(price)}\n"
                    f"SL M15 Supertrend: {fmt_price(t.stop_loss or t.reference_sl)} | "
                    f"TP {fmt_num(t.take_profit_rr, 2)}R: {fmt_price(t.take_profit)} ({t.take_profit_source}; band {t.rr_min}-{t.rr_max}R)\n"
                    f"SuperTrend source: ATR {t.supertrend_atr_length} • Factor {t.supertrend_factor:g}\n"
                    f"Protection: {t.protection_state} (SL {t.sl_algo_status} / TP {t.tp_algo_status})\n"
                    f"Algo IDs: SL {t.sl_algo_id or '-'} / TP {t.tp_algo_id or '-'}\n"
                    f"EMA H4 at signal: {ema_filter_values_label(t.ema_filter_length, t.ema_filter_value, t.ema_filter_slow_value)}\n"
                    f"Floating: {fmt_num(pnl_pct, 2)}% | {fmt_num(pnl_usdt, 4)} USDT\n"
                    f"Qty: {format_qty(t.quantity or Decimal('0'))} | Margin est.: {t.margin_usdt} USDT | Lev: {t.leverage}x\n"
                    f"Sizing: {'MINIMUM AUTO' if t.min_margin_mode else 'FIXED TARGET'}\n"
                    + remainder_note
                    + f"\nHolding: {duration_text((now_utc() - (t.filled_at or t.created_at)).total_seconds())} | Mode: {'REAL' if t.real_enabled else 'PAPER'}"
                )

        # Crucial: open LIMIT orders in Binance's WS/snapshot cache must be
        # visible even when no matching local Trade object exists. These rows
        # also reserve /max capacity through _active_slot_projection().
        for (symbol, direction), orders in sorted(orders_by_pair_direction.items()):
            if (symbol, direction) in represented_directions:
                continue
            for order in orders:
                lines.append(
                    f"⏳ {symbol} • PENDING {direction} (EXCHANGE ORDER)\n"
                    f"Binance: {order['status']} • Order ID: {order['order_id'] or '-'}\n"
                    f"Client ID: {order['client_id'] or '-'}\n"
                    f"LIMIT: {fmt_price(order['price'])}\n"
                    f"Executed: {format_qty(order['executed_qty'])} / {format_qty(order['original_qty'])}"
                    f" • Remaining: {format_qty(order['remaining_qty'])}\n"
                    "Capacity: RESERVED • Mode: REAL"
                )
                represented_directions.add((symbol, direction))

        # Fallback: show Binance positions directly from ACCOUNT_UPDATE/snapshot
        # cache when an exchange position has not yet been paired to a Trade row.
        # This is cache-only and prevents a real position from disappearing from
        # /trade during an order/position event race.
        exchange_position_rows = 0
        for (symbol, position_side), position in sorted(self.exchange_positions.items()):
            if not isinstance(position, dict):
                continue
            raw_amount = position.get("_position_amount_decimal", position.get("positionAmt", "0"))
            try:
                amount = parse_signed_decimal(str(raw_amount or "0"))
            except Exception:
                continue
            if amount == 0:
                continue
            direction = "LONG" if amount > 0 else "SHORT"
            if (symbol, direction) in represented_filled_directions:
                continue
            try:
                entry = Decimal(str(position.get("entryPrice") or position.get("ep") or "0"))
                mark = Decimal(str(position.get("markPrice") or position.get("mp") or "0"))
                upnl = Decimal(str(position.get("unRealizedProfit") or position.get("up") or "0"))
                leverage = int(position.get("leverage") or self.leverage or DEFAULT_LEVERAGE)
            except (InvalidOperation, ValueError, TypeError):
                entry, mark, upnl, leverage = Decimal("0"), Decimal("0"), Decimal("0"), self.leverage
            pnl_pct = None
            if entry > 0 and mark > 0:
                move = mark - entry if direction == "LONG" else entry - mark
                pnl_pct = move / entry * Decimal("100")
            lines.append(
                f"{'🟢' if direction == 'LONG' else '🔴'} {symbol} • {direction} FILLED (EXCHANGE POSITION)\n"
                f"Position side: {position_side}\n"
                f"Entry: {fmt_price(entry if entry > 0 else None)} | Mark: {fmt_price(mark if mark > 0 else None)}\n"
                f"Floating: {fmt_num(pnl_pct, 2)}% | {fmt_num(upnl, 4)} USDT\n"
                f"Qty: {format_qty(abs(amount))} | Lev: {leverage}x\n"
                "Source: ACCOUNT_UPDATE / exchange snapshot • Capacity: RESERVED"
            )
            exchange_position_rows += 1
            represented_filled_directions.add((symbol, direction))

        if not active and not orders_by_pair_direction and exchange_position_rows == 0:
            await self.reply("Tidak ada pending order atau posisi aktif.")
            return
        await self.reply(
            "\n\n".join(lines)
            + f"\n\nSL: M15 Supertrend • new-trade TP baseline {self.rr_min}R, liquidity extension up to {self.rr_max}R • REAL: Binance closePosition Algo Orders"
        )

    def _autostop_source_name(self) -> str:
        """Select the equity account the guard must monitor.

        Existing REAL exposure remains REAL even after `/real off`; PAPER is
        used only when no live REAL entry mode or exchange position is present.
        """
        if self.real_mode:
            return "REAL"
        if any(
            trade.real_enabled and trade.status in {"PENDING", "FILLED"}
            for trade in self.active_trades.values()
        ):
            return "REAL"
        for position in self.exchange_positions.values():
            if not isinstance(position, dict):
                continue
            raw_amount = position.get("_position_amount_decimal", position.get("positionAmt", "0"))
            try:
                if parse_signed_decimal(str(raw_amount or "0")) != 0:
                    return "REAL"
            except (InvalidOperation, ValueError, TypeError):
                # An unreadable non-empty exchange position must not silently
                # switch the guard to PAPER while live risk may still exist.
                if raw_amount not in (None, "", "0", 0):
                    return "REAL"
        return "PAPER"

    def _autostop_equity(self) -> Decimal | None:
        """Return current account equity from cached data only; never calls REST.

        REAL equity = cached USDT wallet balance + unrealized PnL recomputed
        from fresh WebSocket prices for every known exchange position. If an
        exposed position cannot be priced, return None so the scanner remains
        held instead of evaluating an incomplete equity snapshot.

        PAPER equity = configured starting balance + closed PAPER PnL + live
        floating PnL for filled PAPER trades.
        """
        source = self._autostop_source_name()
        equity: Decimal
        if source == "REAL":
            if self._real_equity_baseline is None or not self._real_equity_baseline.is_finite() or self._real_equity_baseline <= 0:
                return None
            if not self._realized_since_baseline.is_finite():
                return None
            # Deliberately do not re-read walletBalance: Binance wallet balance
            # already moves with realized PnL and would double count locally-added PnL.
            equity = self._real_equity_baseline + self._realized_since_baseline
            represented: set[tuple[str, str]] = set()

            for position in self.exchange_positions.values():
                if not isinstance(position, dict):
                    continue
                symbol = normalize_symbol(str(position.get("symbol") or ""))
                raw_amount = position.get("_position_amount_decimal", position.get("positionAmt", "0"))
                try:
                    amount = raw_amount if isinstance(raw_amount, Decimal) else parse_signed_decimal(str(raw_amount or "0"))
                except (InvalidOperation, TypeError, ValueError):
                    return None
                if amount == 0:
                    continue
                try:
                    entry = Decimal(str(position.get("entryPrice") or position.get("ep") or "0"))
                except (InvalidOperation, TypeError, ValueError):
                    return None
                if not symbol or not entry.is_finite() or entry <= 0:
                    return None
                mark = self._current_price(symbol)
                if mark is None or not mark.is_finite() or mark <= 0:
                    return None
                direction = self._exchange_position_direction(position, amount)
                if direction not in {"LONG", "SHORT"}:
                    return None
                signed_amount = abs(amount) if direction == "LONG" else -abs(amount)
                equity += (mark - entry) * signed_amount
                represented.add((symbol, direction))

            # Bridge the short WS ordering window: a fill can enter the trade
            # ledger before ACCOUNT_UPDATE has populated exchange_positions.
            for trade in self.active_trades.values():
                if not trade.real_enabled or trade.status not in {"PENDING", "FILLED"}:
                    continue
                qty = trade.entry_executed_qty or (trade.quantity if trade.status == "FILLED" else None)
                if qty is None or qty <= 0 or (trade.pair, trade.direction) in represented:
                    continue
                entry = trade.fill_price or trade.entry
                mark = self._current_price(trade.pair)
                if entry is None or entry <= 0 or mark is None or mark <= 0:
                    return None
                signed_qty = qty if trade.direction == "LONG" else -qty
                equity += (mark - entry) * signed_qty

        else:
            realized = Decimal("0")
            for row in self.history_records:
                if not isinstance(row, dict):
                    continue
                if row.get("strategy_name") != volumatic.STRATEGY_NAME or bool(row.get("real_enabled")):
                    continue
                try:
                    pnl = Decimal(str(row.get("pnl_usdt") or "0"))
                except (InvalidOperation, TypeError, ValueError):
                    continue
                if pnl.is_finite():
                    realized += pnl
            equity = SIM_START_BALANCE + realized
            for trade in self.active_trades.values():
                if trade.real_enabled or trade.status not in {"PENDING", "FILLED"}:
                    continue
                qty = trade.entry_executed_qty or (trade.quantity if trade.status == "FILLED" else None)
                if qty is None or qty <= 0:
                    continue
                entry = trade.fill_price or trade.entry
                mark = self._current_price(trade.pair)
                if entry is None or entry <= 0 or mark is None or mark <= 0:
                    return None
                signed_qty = qty if trade.direction == "LONG" else -qty
                equity += (mark - entry) * signed_qty

        return equity if equity.is_finite() and equity > 0 else None

    def _autostop_floor(self) -> Decimal | None:
        """Calculate the trailing equity floor from the high-water mark."""
        if self.autostop_percent is None or self._equity_peak is None:
            return None
        if not self.autostop_percent.is_finite() or not self._equity_peak.is_finite():
            return None
        if self.autostop_percent <= 0 or self._equity_peak <= 0:
            return None
        return self._equity_peak * (Decimal("1") - self.autostop_percent / Decimal("100"))

    def _autostop_status_text(self) -> str:
        """Human-readable `/autostop` status with no exchange/network calls."""
        if self.autostop_percent is None:
            state = "OFF"
            source = "—"
        elif self._autostop_triggered:
            state = "TRIGGERED"
            source = self._autostop_source or self._autostop_source_name()
        elif self._autostop_data_hold:
            state = "ARMED · WAITING DATA"
            source = self._autostop_source or self._autostop_source_name()
        else:
            state = f"ARMED · {decimal_to_str(self.autostop_percent)}%"
            source = self._autostop_source or self._autostop_source_name()
        floor = self._autostop_floor()
        scanner = (
            "LOCKED BY AUTOSTOP" if self._autostop_triggered else
            "PAUSED — WAITING EQUITY DATA" if self.autostop_percent is not None and self._autostop_data_hold else
            "ON" if self.scan_enabled else "OFF"
        )
        return (
            "╔══ ACCOUNT DRAWDOWN GUARD ══╗\n"
            f"STATE   : {state}\n"
            f"SOURCE  : {source}\n"
            f"EQUITY  : {fmt_num(self._equity_last, 4) if self._equity_last is not None else '—'} USDT\n"
            f"PEAK    : {fmt_num(self._equity_peak, 4) if self._equity_peak is not None else '—'} USDT\n"
            f"FLOOR   : {fmt_num(floor, 4) if floor is not None else '—'} USDT\n"
            f"SCANNER : {scanner}\n\n"
            "Guard hanya menghentikan entry/scanner baru. Posisi dan order yang sudah ada tidak ditutup otomatis.\n"
            "Perintah: /autostop <persen> | /autostop reset | /autostop off"
        )

    async def _autostop_check(self, force: bool = False) -> None:
        if self.autostop_percent is None or self._autostop_triggered:
            return
        now_mono = time.monotonic()
        if not force and now_mono - self._autostop_last_check < AUTOSTOP_CHECK_SECONDS:
            return
        self._autostop_last_check = now_mono
        equity = self._autostop_equity()
        if equity is None or not equity.is_finite() or equity <= 0:
            self._autostop_data_hold = True
            if (self._autostop_notice_state != "WAITING"
                    and now_mono - self._autostop_last_state_notice_at >= 900.0):
                self._autostop_missing_price_notice_at = now_mono
                self._autostop_last_state_notice_at = now_mono
                self._autostop_notice_state = "WAITING"
                source = self._autostop_source_name()
                if source == "REAL" and self.real.cached_usdt_balance() is None:
                    detail = "wallet balance belum tersedia dari REST atau ACCOUNT_UPDATE"
                else:
                    missing = [
                        t.pair for t in self.active_trades.values()
                        if t.status == "FILLED" and t.real_enabled == (source == "REAL")
                        and (self.prices.get(t.pair) is None or not self.prices[t.pair].live)
                    ]
                    detail = "mark price belum fresh: " + ", ".join(missing[:6]) if missing else "snapshot equity belum lengkap"
                log.info("[AUTOSTOP] snapshot WAIT; guard remains ARMED | source=%s | %s", source, detail)
                await self.reply(
                    "╔══ ACCOUNT GUARD / SNAPSHOT ══╗\n"
                    "STATE   : ARMED · WAITING DATA\n"
                    f"SOURCE  : {source}\n"
                    f"DETAIL  : {detail}\n"
                    "ACTION  : tidak mengubah peak/floor dan tidak membuka entry baru dari guard\n"
                    "GUARD   : tetap aktif; evaluasi dilanjutkan otomatis ketika data fresh"
                )
            return

        if self._autostop_data_hold:
            self._autostop_data_hold = False
            self._scan_trigger_now = bool(self.scan_enabled and not self._autostop_triggered)
            was_waiting_notice = self._autostop_notice_state == "WAITING"
            can_notify = now_mono - self._autostop_last_state_notice_at >= 900.0
            self._autostop_notice_state = "RESTORED"
            if was_waiting_notice and can_notify:
                self._autostop_last_state_notice_at = now_mono
                await self.reply(
                    "╔══ ACCOUNT GUARD / DATA RESTORED ══╗\n"
                    "STATE   : ARMED · EQUITY FRESH\n"
                    "ACTION  : pemantauan drawdown aktif kembali\n"
                    f"SCANNER : {'RESUME QUEUED' if self._scan_trigger_now else 'UNCHANGED'}"
                )
            log.info("[AUTOSTOP] equity data fresh; entry scanner hold released.")

        current_source = self._autostop_source_name()
        source_changed = self._autostop_source is not None and self._autostop_source != current_source
        if source_changed and not self._autostop_triggered:
            # PAPER and REAL accounts have different equity bases. Never compare
            # the first REAL snapshot against a previous PAPER high-water mark.
            self._equity_peak = equity
            log.info("[AUTOSTOP] equity source changed %s -> %s; baseline re-established.", self._autostop_source, current_source)
        self._autostop_source = current_source
        self._equity_last = equity
        peak_changed = source_changed or self._equity_peak is None or equity > self._equity_peak
        if peak_changed:
            self._equity_peak = equity
            self._autostop_dirty = True

        floor = self._autostop_floor()
        triggered_now = bool(
            floor is not None and not self._autostop_triggered and equity <= floor
        )
        if triggered_now:
            self._autostop_triggered = True
            self.scan_enabled = False
            self._scan_trigger_now = False
            drawdown = (
                (self._equity_peak - equity) / self._equity_peak * Decimal("100")
                if self._equity_peak and self._equity_peak > 0 else Decimal("0")
            )
            log.critical(
                "[AUTOSTOP] TRIGGERED | equity=%s peak=%s floor=%s dd=%s%%",
                decimal_to_str(equity), decimal_to_str(self._equity_peak),
                decimal_to_str(floor), decimal_to_str(drawdown),
            )
            await self._save_runtime_settings()
            await self._save_real_guard_checkpoint()
            self._autostop_dirty = False
            self._autostop_last_persist = now_mono
            await self._record_event(
                "AUTOSTOP_TRIGGERED", None,
                equity=decimal_to_str(equity), peak_equity=decimal_to_str(self._equity_peak),
                floor_equity=decimal_to_str(floor), drawdown_percent=decimal_to_str(drawdown),
                threshold_percent=decimal_to_str(self.autostop_percent),
            )
            await self.reply(
                "🛑 AUTOSTOP TRIGGERED\n\n"
                f"Drawdown dari peak: {drawdown.quantize(Decimal('0.01'))}% "
                f"(limit {self.autostop_percent}%).\n"
                f"Peak: {fmt_num(self._equity_peak, 4)} USDT\n"
                f"Equity: {fmt_num(equity, 4)} USDT\n"
                f"Floor: {fmt_num(floor, 4)} USDT\n\n"
                "SCANNER dikunci OFF untuk mencegah entry baru. Pending order dan posisi "
                "yang sudah ada TIDAK ditutup otomatis; manajemen reversal/close tetap berjalan.\n\n"
                "Setelah memeriksa akun: /autostop reset, lalu /scan on."
            )
            return

        if self._autostop_dirty and (now_mono - self._autostop_last_persist >= AUTOSTOP_PERSIST_SECONDS):
            self._autostop_last_persist = now_mono
            await self._save_runtime_settings()
            await self._save_real_guard_checkpoint()
            self._autostop_dirty = False

    async def _handle_autostop_command(self, text: str) -> None:
        # Preserve the prior REAL checkpoint when /autostop is used after /try
        # but before /real on; this reads only the guard ledger, not trades/scanner.
        await self._load_real_guard_checkpoint()
        parts = text.split()
        if len(parts) == 1:
            await self.reply(self._autostop_status_text())
            return
        if len(parts) != 2:
            raise ValueError("Gunakan /autostop [persen], /autostop off, atau /autostop reset.")

        arg = parts[1].lower()
        if arg == "off":
            self.autostop_percent = None
            self._autostop_data_hold = False
            self._autostop_triggered = False
            self._equity_peak = None
            self._equity_last = None
            self._autostop_source = None
            self._autostop_dirty = False
            await self._save_runtime_settings()
            await self._save_real_guard_checkpoint()
            await self.reply(
                "🛡 AUTOSTOP OFF. Scanner tidak otomatis dinyalakan kembali; "
                "gunakan /scan on setelah meninjau risiko akun."
            )
            return

        if arg == "reset":
            equity = self._autostop_equity()
            self._equity_peak = equity if equity is not None and equity > 0 else None
            self._equity_last = equity if equity is not None and equity > 0 else None
            self._autostop_source = self._autostop_source_name() if equity is not None else None
            self._autostop_data_hold = bool(equity is None or equity <= 0)
            self._autostop_triggered = False
            self._autostop_last_persist = 0.0
            self._autostop_dirty = False
            await self._save_runtime_settings()
            await self._save_real_guard_checkpoint()
            await self.reply(
                "♻ AUTOSTOP RE-ARMED. Peak equity diambil dari snapshot saat ini "
                "(atau akan diinisialisasi saat snapshot valid tersedia). Scanner tetap OFF; "
                "nyalakan kembali secara manual setelah memeriksa akun.\n\n"
                + self._autostop_status_text()
            )
            return

        try:
            value = Decimal(arg.replace(",", "."))
        except (InvalidOperation, ValueError):
            raise ValueError("Gunakan /autostop [persen], /autostop off, atau /autostop reset.")
        if not value.is_finite() or value <= 0 or value >= 100:
            raise ValueError("Persentase autostop harus lebih dari 0 dan kurang dari 100.")

        was_off = self.autostop_percent is None
        self.autostop_percent = value
        if was_off or self._autostop_triggered or self._equity_peak is None:
            equity = self._autostop_equity()
            self._equity_peak = equity if equity is not None and equity > 0 else None
            self._equity_last = equity if equity is not None and equity > 0 else None
            self._autostop_source = self._autostop_source_name() if equity is not None else None
            self._autostop_data_hold = bool(self._equity_peak is None)
        self._autostop_triggered = False
        self._autostop_last_persist = 0.0
        self._autostop_dirty = False
        await self._save_runtime_settings()
        await self._save_real_guard_checkpoint()
        await self.reply(
            f"🛡 AUTOSTOP ARMED • {decimal_to_str(value)}% dari peak equity\n\n"
            + self._autostop_status_text()
            + "\n\nAutostop akan menghentikan scanner saat batas tercapai; "
              "posisi/order yang sudah ada tetap dikelola."
        )

    @staticmethod
    def _max_drawdown(values: list[float]) -> float:
        peak = 0.0
        equity = 0.0
        max_dd = 0.0
        for pnl in values:
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return max_dd

    async def show_stats(self) -> None:
        closed = [
            r for r in self.history_records
            if r.get("strategy_name") == volumatic.STRATEGY_NAME and r.get("result") == "CLOSED"
        ]
        active = [t for t in self.active_trades.values() if t.status in {"PENDING", "FILLED"}]
        slot_count, pending_slots, filled_slots, live_limits = self._active_slot_counts()
        realized = [float(Decimal(str(r.get("pnl_usdt") or "0"))) for r in closed]
        wins = [x for x in realized if x > 0]
        losses = [x for x in realized if x < 0]
        be = len(realized) - len(wins) - len(losses)
        gross_profit, gross_loss = sum(wins), abs(sum(losses))
        win_rate = len(wins) / len(realized) * 100 if realized else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        floating = 0.0
        for trade in active:
            if trade.status != "FILLED" or trade.fill_price is None:
                continue
            price = self._current_price(trade.pair)
            if price is None:
                continue
            move = price - trade.fill_price if trade.direction == "LONG" else trade.fill_price - price
            floating += float(move * (trade.quantity or Decimal("0")))

        chronological = sorted(closed, key=lambda r: str(r.get("closed_at") or ""))
        dd = self._max_drawdown([float(Decimal(str(r.get("pnl_usdt") or "0"))) for r in chronological])
        pf_text = f"{gross_profit / gross_loss:.2f}" if gross_loss > 0 else ("∞" if gross_profit > 0 else "—")
        total_pnl = sum(realized)
        expectancy = total_pnl / len(realized) if realized else 0.0
        total_r = sum(
            float(Decimal(str(r.get("realized_r") or "0")))
            for r in closed if r.get("realized_r") is not None
        )
        holding = []
        for record in closed:
            opened = parse_iso(record.get("filled_at"))
            finished = parse_iso(record.get("closed_at"))
            if opened and finished and finished >= opened:
                holding.append((finished - opened).total_seconds())
        avg_holding = duration_text(sum(holding) / len(holding)) if holding else "—"

        pending = pending_slots
        filled = filled_slots
        expired = sum(
            r.get("result") == "EXPIRED"
            for r in self.history_records
            if r.get("strategy_name") == volumatic.STRATEGY_NAME
        )
        lines = [
            "╔════ VOLUMATIC / TELEMETRY ════╗",
            f"SYS  {'ONLINE' if self._running else 'OFFLINE'}   TF  {self.timeframe}   LEN  {self.trend_length}   EMA  {ema_filter_label(self.ema_filter_length)}   MODE  {'REAL' if self.real_mode else 'PAPER'}",
            "────────────────────────────────",
            "[ PERFORMANCE ]",
            f"CLOSED       {len(closed):>5}     W/L/BE   {len(wins)}/{len(losses)}/{be}",
            f"WIN RATE     {win_rate:>6.2f}%   PF       {pf_text}",
            f"PNL REALIZED {total_pnl:+.4f} USDT",
            f"PNL FLOATING {floating:+.4f} USDT",
            f"EXPECTANCY   {expectancy:+.4f} USDT/trade",
            f"AVG W/L      {avg_win:+.4f} / {avg_loss:+.4f} USDT",
            f"MAX DD       {dd:.4f} USDT   TOTAL R  {total_r:+.3f}R",
            f"AVG HOLD     {avg_holding}",
            "────────────────────────────────",
            "[ EXPOSURE ]",
            f"ACTIVE       {slot_count:>3}/{self.max_active_trades:<3}  PENDING  {pending:<3} FILLED {filled:<3}",
            f"LIVE LIMITS  {live_limits}",
            f"EXPIRED      {expired}",
        ]
        for tf in ("M15",):
            long_rows = [r for r in closed if r.get("timeframe") == tf and r.get("direction") == "LONG"]
            short_rows = [r for r in closed if r.get("timeframe") == tf and r.get("direction") == "SHORT"]
            long_pnl = sum(float(Decimal(str(r.get("pnl_usdt") or "0"))) for r in long_rows)
            short_pnl = sum(float(Decimal(str(r.get("pnl_usdt") or "0"))) for r in short_rows)
            lines.append(
                f"{tf:<2}  L {len(long_rows):>3} ({long_pnl:+.3f})  "
                f"S {len(short_rows):>3} ({short_pnl:+.3f}) USDT"
            )
        guard_state = (
            "TRIGGERED" if self._autostop_triggered else
            (f"ARMED {decimal_to_str(self.autostop_percent)}%" if self.autostop_percent is not None else "OFF")
        )
        lines += [
            "────────────────────────────────",
            "[ SAFETY ]",
            f"AUTOSTOP     {guard_state}",
            f"EQUITY       {fmt_num(self._equity_last, 4) if self._equity_last is not None else '—'} USDT",
            f"PEAK / FLOOR {fmt_num(self._equity_peak, 4) if self._equity_peak is not None else '—'} / {fmt_num(self._autostop_floor(), 4) if self._autostop_floor() is not None else '—'} USDT",
            f"SCANNER      {self._scanner_state_label()}",
            f"SCAN SCHEDULE {', '.join(self.scan_schedule_slots) + ' WIB' if self.scan_schedule_slots else ('ACTIVATED ' + self.scan_schedule_activated_at if self.scan_schedule_activated_at else 'OFF')}",
            "────────────────────────────────",
            f"PnL belum termasuk seluruh fee/funding aktual. R = jarak entry aktual ke SL Supertrend M15; TP baseline {self.rr_min}R dengan ekstensi liquidity maksimal {self.rr_max}R.",
        ]
        await self.reply("\n".join(lines))

    async def show_status(self) -> None:
        _slots, pending, filled, live_limits = self._active_slot_counts()
        fresh = sum(x.live for x in self.prices.values())
        await self.reply(
            "VOLUMATIC STATUS\n\n"
            f"Main: {'ONLINE' if self._running else 'OFFLINE'}\n"
            f"Strategy: {volumatic.STRATEGY_NAME} {volumatic.STRATEGY_VERSION}\n"
            f"Signal TF: M15 | EMA TF: H4 | Supertrend TF: M15\nTrend length: {self.trend_length}\nEMA filter: {ema_filter_label(self.ema_filter_length)}\nH1 logic: {'ON' if self.h1_enabled else 'OFF'} (entry filter + inverse-cross forced exit)\nSuperTrend M15: ATR {self.supertrend_atr_length} • Factor {self.supertrend_factor:g}\nTP RR: MIN {self.rr_min}R • MAX {self.rr_max}R\nScanner: {'ON' if self.scan_enabled else ('WAITING' if self.scan_schedule_waiting else 'OFF')}\n"
            f"Scan schedule: {', '.join(self.scan_schedule_slots) + ' WIB' if self.scan_schedule_slots else ('ACTIVATED ' + self.scan_schedule_activated_at if self.scan_schedule_activated_at else 'OFF')}\n"
            f"Universe: {len(self.symbols)} | Market WS: {self.ws.status} | User WS: {self.user_ws.status} | Fresh prices: {fresh}\n"
            f"Pending slots: {pending} | Filled slots: {filled} | Live LIMITs: {live_limits} | Max: {self.max_active_trades}\n"
            f"Binance REST cooldown: {self._binance_cooldown_remaining():.0f}s | Deferred commands: {len(self.deferred_commands)}\n"
            f"REAL entry: {'ON' if self.real_mode else 'OFF'} | Margin mode: {'MINIMUM AUTO' if self.min_margin_enabled else 'FIXED'} | Fixed margin: {self.margin_usdt} USDT | Leverage: {self.leverage}x\n"
            f"Fixed margin guard: {'MINIMUM AUTO (not fixed)' if self.min_margin_enabled else '95%–105% of target (REAL sizing)'}\n"
            f"AUTOSTOP: {'TRIGGERED' if self._autostop_triggered else (decimal_to_str(self.autostop_percent) + '%' if self.autostop_percent is not None else 'OFF')} | Peak: {fmt_num(self._equity_peak, 4) if self._equity_peak is not None else '-'} USDT\n"
            f"History: {len(self.history_records)} | Events: {len(self.history_events)} | Session: {self.session_id}"
        )

    async def show_help(self) -> None:
        await self.reply(
            "COMMAND VOLUMATIC TREND\n\n"
            "/scan on|off|status - aktif/nonaktifkan/cek scanner M15 Volumatic + H4 EMA\n"
            "/scan schedule - lihat jadwal WIB; pilih dengan /scan schedule 1,3,6\n"
            "/timeframe - strategi tetap: H4 EMA + M15 Volumatic/Supertrend\n"
            "/length <angka> - atur Trend Length Volumatic bebas (bilangan bulat positif; default 100)\n"
            "/EMA <angka|off> atau /EMA <angka1> <angka2> - filter EMA single/dual bebas\n"
            "/H1 on|off - saklar bersama filter entry H1 dan forced close saat EMA H1 cross\n"
            f"/ATR <angka> - atur SuperTrend M15 ATR Length (default {volumatic.SUPERTREND_ATR_LENGTH})\n"
            f"/Factor <angka> - atur SuperTrend M15 Factor (default {volumatic.SUPERTREND_FACTOR:g})\n"
            f"/RR_MIN <angka> - RR dasar/fallback dan batas bawah liquidity (default {volumatic.RR_MIN:g})\n"
            f"/RR_MAX <angka> - batas atas TP liquidity (default {volumatic.RR_MAX:g})\n"
            "/max [angka] - batas total pending + filled\n"
            "/trade - lihat pending order dan posisi aktif\n"
            "/stats - statistik trade, PnL, win rate, DD, R\n"
            "/setup - simpan snapshot trade aktif ke GitHub\n"
            "/open - pulihkan snapshot secara manual (tidak otomatis saat /try)\n"
            "/del PAIR|all - batalkan pending order\n"
            "/close PAIR|all - tutup posisi atau batalkan pending\n"
            "/remove TRADE_ID|PAIR - hapus record dari statistik (event audit tetap)\n"
            "/real on|off - aktif/nonaktifkan entry baru REAL\n"
            "/autostop [persen|off|reset] - hentikan scanner saat equity turun dari peak\n"
            "/margin [USDT] - margin target entry baru (dipakai saat /min off)\n"
            "/min on|off - gunakan margin minimum valid Binance atau margin tetap\n"
            "/leverage [angka] - leverage entry baru\n"
            "/balance - saldo Futures atau estimasi PAPER\n"
            "/api - statistik bobot/request Binance\n"
            "/catatan [teks] - tambah atau lihat catatan\n"
            "/resetstats - arsipkan lalu reset statistik trade\n"
            "/reset CONFIRM - reset history/event/catatan (tanpa mengubah posisi)\n"
            "/status - status sistem\n/help - bantuan\n\n"
            f"Aturan: BLUE=LONG, RED=SHORT; bubble M15 confirmed + filter EMA H4 + BTC H4; entry pada OPEN M15 berikutnya; SL Supertrend M15; TP dasar RR_MIN {self.rr_min}R, diperpanjang ke liquidity pool belum termitigasi paling jauh hingga RR_MAX {self.rr_max}R; posisi tidak reversal oleh bubble lawan."
        )

    async def _dispatch_command(self, text: str) -> None:
        """Execute a command. Shared by Telegram updates and deferred recovery."""
        text = str(text or "").strip()
        if not text:
            return
        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if command in {"/try", "/end", "/ganti", "/healthz"}:
            return
        if command == "/timeframe": await self._handle_timeframe(text)
        elif command == "/length": await self._handle_length(text)
        elif command == "/ema": await self._handle_ema(text)
        elif command == "/h1": await self._handle_h1(text)
        elif command == "/atr": await self._handle_atr(text)
        elif command == "/factor": await self._handle_factor(text)
        elif command == "/rr_min": await self._handle_rr_min(text)
        elif command == "/rr_max": await self._handle_rr_max(text)
        elif command == "/scan": await self._handle_scan(text)
        elif command == "/max": await self._handle_max(text)
        elif command == "/margin": await self._handle_margin(text)
        elif command == "/min": await self._handle_min(text)
        elif command == "/leverage": await self._handle_leverage(text)
        elif command == "/real":
            arg = text.split(maxsplit=1)[1].strip().lower() if len(text.split(maxsplit=1)) > 1 else ""
            if arg == "off":
                await self._cancel_queued_real_on()
            await self._set_real(text)
        elif command == "/autostop": await self._handle_autostop_command(text)
        elif command == "/trade": await self.show_trades()
        elif command == "/stats": await self.show_stats()
        elif command == "/status": await self.show_status()
        elif command in {"/help", "/start"}: await self.show_help()
        elif command == "/setup": await self.save_setups()
        elif command == "/open": await self.open_setups()
        elif command == "/del": await self._handle_del(text)
        elif command == "/close": await self._handle_close(text)
        elif command == "/remove": await self._handle_remove(text)
        elif command == "/resetstats": await self._handle_resetstats()
        elif command == "/reset": await self._handle_reset(text)
        elif command == "/balance": await self._show_balance()
        elif command == "/api": await self.reply(await self._api_report())
        elif command == "/catatan":
            arg = text[len(text.split(maxsplit=1)[0]):].strip() if len(text.split(maxsplit=1)) > 1 else ""
            if arg: await self.add_note(arg)
            else: await self.show_notes()
        else:
            await self.reply("Command tidak dikenal. Gunakan /help.")

    async def handle_update(self, update: dict[str, Any], context: dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message") or {}
        user_id = int((message.get("from") or {}).get("id") or 0)
        if user_id and user_id != ALLOWED_USER_ID:
            return
        text = str(message.get("text") or "").strip()
        if not text:
            return
        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        try:
            await self._dispatch_command(text)
        except BinanceRateLimitError as exc:
            if self._deferred_command_allowed(text):
                await self._enqueue_deferred_command(text, str(exc))
                await self.reply(
                    f"⏳ {command} diterima dan masuk antrean. Binance REST sedang cooldown "
                    f"({self._binance_cooldown_remaining():.0f}s). Perintah akan dijalankan otomatis setelah API pulih.\n"
                    "Order LIMIT REAL yang sudah berada di Binance tidak dibatalkan hanya karena cooldown; fill tetap dipantau melalui User Data WebSocket bila stream tersambung."
                )
                return
            log.info("[COMMAND] %s ditahan rate governor; tidak hilang dan dapat dicoba ulang setelah cooldown | %s", command, exc)
            await self.reply(
                f"⏳ {command} belum dijalankan karena Binance REST cooldown "
                f"({self._binance_cooldown_remaining():.0f}s). Perintah ini bersifat lokal/read-only dan dapat diulang setelah API pulih."
            )
        except Exception as exc:
            log.exception("Command %s gagal: %s", command, exc)
            await self.reply(f"❌ {command} gagal: {str(exc)[:1500]}")


# ---------------------------------------------------------------------------
# Engine lifecycle + standalone local Telegram long-polling runner
# ---------------------------------------------------------------------------
_ENGINE: TradingEngine | None = None


async def on_start(context: dict[str, Any]):
    global _ENGINE
    if _ENGINE is not None:
        await _ENGINE.stop()
        _ENGINE = None
    engine = TradingEngine(context)
    _ENGINE = engine
    try:
        await engine.start()
    except Exception:
        try: await engine.stop()
        except Exception: log.exception("Cleanup gagal setelah startup.")
        _ENGINE = None
        raise
    return True


async def handle_update(update: dict[str, Any], context: dict[str, Any]):
    if _ENGINE is not None:
        await _ENGINE.handle_update(update, context)


async def on_stop(context: dict[str, Any]):
    global _ENGINE
    engine = _ENGINE
    if engine is None:
        return
    try: await engine.stop()
    finally: _ENGINE = None


class TelegramBotAPIError(RuntimeError):
    """Telegram Bot API response error with retry metadata when available."""

    def __init__(self, message: str, *, status_code: int = 0, retry_after: float | None = None):
        super().__init__(message)
        self.status_code = int(status_code or 0)
        self.retry_after = retry_after


class LocalTelegramBot:
    """Small synchronous Telegram Bot API adapter used by the local runner.

    TradingEngine expects a synchronous ``send_message(chat_id, text)`` callback;
    its ``reply`` method already dispatches that callback through asyncio.to_thread.
    Polling is also dispatched through asyncio.to_thread so long polling never
    blocks the trading engine's event loop.
    """

    def __init__(self, token: str, offset_path: Path) -> None:
        self.token = str(token or "").strip()
        if not self.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN belum diset di .env.")
        self.api_root = f"https://api.telegram.org/bot{self.token}"
        self.offset_path = Path(offset_path)
        self.offset_path.parent.mkdir(parents=True, exist_ok=True)
        self.connect_timeout = 10.0
        self.request_timeout = 40.0

    def _url(self, method: str) -> str:
        return f"{self.api_root}/{method}"

    @staticmethod
    def _result_or_raise(response: requests.Response, method: str) -> Any:
        try:
            body = response.json()
        except ValueError as exc:
            raise TelegramBotAPIError(
                f"Telegram {method}: respons bukan JSON (HTTP {response.status_code}).",
                status_code=response.status_code,
            ) from exc

        if response.status_code >= 400 or not isinstance(body, dict) or not body.get("ok"):
            description = str(body.get("description") or response.text or "Unknown Telegram API error") if isinstance(body, dict) else str(response.text)
            parameters = body.get("parameters") if isinstance(body, dict) else None
            retry_after = None
            if isinstance(parameters, dict):
                try:
                    retry_after = float(parameters.get("retry_after"))
                except (TypeError, ValueError):
                    retry_after = None
            raise TelegramBotAPIError(
                f"Telegram {method} gagal (HTTP {response.status_code}): {description[:700]}",
                status_code=response.status_code,
                retry_after=retry_after,
            )
        return body.get("result")

    def get_me(self) -> dict[str, Any]:
        response = requests.get(
            self._url("getMe"),
            timeout=(self.connect_timeout, self.request_timeout),
        )
        result = self._result_or_raise(response, "getMe")
        if not isinstance(result, dict):
            raise TelegramBotAPIError("Telegram getMe mengembalikan data bot yang tidak valid.")
        return result

    def delete_webhook(self) -> None:
        # getUpdates is unavailable while a webhook is configured. Pending updates
        # are deliberately preserved; no old Telegram messages are discarded.
        response = requests.post(
            self._url("deleteWebhook"),
            json={"drop_pending_updates": False},
            timeout=(self.connect_timeout, self.request_timeout),
        )
        self._result_or_raise(response, "deleteWebhook")

    def get_updates(self, offset: int | None, poll_timeout: int) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timeout": int(poll_timeout),
            "limit": 100,
            "allowed_updates": json.dumps(["message", "edited_message"]),
        }
        if offset is not None:
            params["offset"] = int(offset)
        response = requests.get(
            self._url("getUpdates"),
            params=params,
            timeout=(self.connect_timeout, max(self.request_timeout, float(poll_timeout) + 15.0)),
        )
        result = self._result_or_raise(response, "getUpdates")
        if not isinstance(result, list):
            raise TelegramBotAPIError("Telegram getUpdates mengembalikan daftar update yang tidak valid.")
        return [item for item in result if isinstance(item, dict)]

    @staticmethod
    def _split_message(message: str, limit: int = 4000) -> list[str]:
        text = str(message or "")
        if not text:
            return []
        chunks: list[str] = []
        while len(text) > limit:
            split_at = text.rfind("\n", 0, limit + 1)
            if split_at < limit // 2:
                split_at = limit
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        if text:
            chunks.append(text)
        return chunks

    def send_message(self, chat_id: int, message: str) -> None:
        # Telegram text messages are limited to 4,096 characters. Keep a margin
        # below that limit and send plain text to preserve the engine's formatting.
        for chunk in self._split_message(str(message)):
            response = requests.post(
                self._url("sendMessage"),
                json={"chat_id": int(chat_id), "text": chunk},
                timeout=(self.connect_timeout, self.request_timeout),
            )
            self._result_or_raise(response, "sendMessage")

    def load_offset(self) -> int | None:
        try:
            raw = self.offset_path.read_text(encoding="utf-8").strip()
            return int(raw) if raw else None
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            log.warning("[TELEGRAM] Offset lokal tidak terbaca; polling dimulai dari update yang masih tertunda.", exc_info=True)
            return None

    def save_offset(self, offset: int) -> None:
        temporary = self.offset_path.with_suffix(self.offset_path.suffix + ".tmp")
        try:
            temporary.write_text(str(int(offset)), encoding="utf-8")
            temporary.replace(self.offset_path)
        except OSError:
            log.warning("[TELEGRAM] Offset lokal gagal disimpan; update mungkin diproses ulang setelah restart.", exc_info=True)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


async def _wait_for_local_stop(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=max(0.1, float(seconds)))
    except asyncio.TimeoutError:
        pass


async def _local_telegram_polling_loop(
    telegram: LocalTelegramBot,
    context: dict[str, Any],
    stop_event: asyncio.Event,
) -> None:
    """Receive private messages from the configured Telegram user only."""
    offset = telegram.load_offset()
    poll_timeout = max(5, min(50, int(os.getenv("TELEGRAM_POLL_TIMEOUT_SECONDS", "25") or "25")))
    backoff = 2.0
    log.info("[TELEGRAM] Long polling aktif; hanya private chat ALLOWED_USER_ID yang diproses.")

    while not stop_event.is_set():
        try:
            updates = await asyncio.to_thread(telegram.get_updates, offset, poll_timeout)
            backoff = 2.0
        except TelegramBotAPIError as exc:
            if exc.status_code == 401:
                raise RuntimeError("TELEGRAM_BOT_TOKEN ditolak Telegram (HTTP 401). Periksa token di .env.") from exc
            delay = max(1.0, exc.retry_after or backoff)
            log.warning("[TELEGRAM] Polling API gagal; mencoba lagi dalam %.1fs: %s", delay, exc)
            await _wait_for_local_stop(stop_event, delay)
            backoff = min(30.0, max(backoff * 2.0, delay))
            continue
        except requests.RequestException as exc:
            log.warning("[TELEGRAM] Koneksi polling gagal; mencoba lagi dalam %.1fs: %s", backoff, exc)
            await _wait_for_local_stop(stop_event, backoff)
            backoff = min(30.0, backoff * 2.0)
            continue

        for update in updates:
            if stop_event.is_set():
                break
            try:
                update_id = int(update.get("update_id"))
            except (TypeError, ValueError):
                log.warning("[TELEGRAM] Mengabaikan update tanpa update_id yang valid.")
                continue

            message = update.get("message") or update.get("edited_message") or {}
            sender = message.get("from") or {}
            chat = message.get("chat") or {}
            try:
                sender_id = int(sender.get("id") or 0)
                chat_id = int(chat.get("id") or 0)
            except (TypeError, ValueError):
                sender_id, chat_id = 0, 0

            # ALLOWED_USER_ID is the Telegram user ID. Restricting the local
            # adapter to that user's private chat also keeps reply routing exact.
            if sender_id == ALLOWED_USER_ID and chat_id == ALLOWED_USER_ID:
                try:
                    await handle_update(update, context)
                except Exception:
                    log.exception("[TELEGRAM] Gagal meneruskan update %s ke trading engine.", update_id)

            # Acknowledge every valid update (including unauthorized messages) so
            # one unrelated update cannot permanently block the polling queue.
            offset = max(offset or 0, update_id + 1)
            telegram.save_offset(offset)


async def _run_local_telegram() -> None:
    """Run the same TradingEngine directly with Telegram long polling on a PC."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN belum diset. Tambahkan ke file .env.")

    telegram = LocalTelegramBot(
        TELEGRAM_BOT_TOKEN,
        BASE_DIR / "data" / "telegram_update_offset.txt",
    )
    context: dict[str, Any] = {
        "chat_id": ALLOWED_USER_ID,
        "user_id": ALLOWED_USER_ID,
        "send_message": telegram.send_message,
    }
    stop_event = asyncio.Event()
    engine_started = False

    try:
        bot_info = await asyncio.to_thread(telegram.get_me)
        await asyncio.to_thread(telegram.delete_webhook)
        bot_username = bot_info.get("username")
        bot_name = f"@{bot_username}" if bot_username else str(bot_info.get("first_name") or "Telegram bot")
        log.info("[TELEGRAM] Terhubung sebagai %s.", bot_name)

        await on_start(context)
        engine_started = True
        await _local_telegram_polling_loop(telegram, context, stop_event)
    finally:
        stop_event.set()
        if engine_started:
            await on_stop(context)
        log.info("[LOCAL] Proses bot dihentikan.")


if __name__ == "__main__":
    try:
        asyncio.run(_run_local_telegram())
    except KeyboardInterrupt:
        print("\n[STOP] Bot dihentikan oleh pengguna.")
