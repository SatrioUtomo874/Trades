#!/usr/bin/env python3
from __future__ import annotations

"""Volumatic Trend [ChartPrime] H4/D1 execution engine.

Contract for try.py:
    async def on_start(context)
    async def handle_update(update, context)
    async def on_stop(context)

The strategy engine is intentionally limited to Volumatic Trend bubble flips.
The main process owns Binance REST/private execution, WebSocket prices,
H4/D1 scanner, Telegram, PAPER/REAL ledger, logging and GitHub history.
Mode B detects the offset=-1 visual bubble from the adjacent live candle and
uses the visually marked prior candle's close as its one-candle LIMIT price.
"""

import asyncio
import base64
import contextvars
import ctypes
import gc
import hashlib
import hmac
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN, ROUND_UP
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
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
if not isinstance(getattr(volumatic, "TIMEFRAME_MS", None), dict) or set(volumatic.TIMEFRAME_MS) != {"H4", "D1"}:
    raise RuntimeError("strategy module mismatch: TIMEFRAME_MS H4/D1 tidak tersedia. Pastikan try.py dan STRATEGY_FILE satu versi.")
if not isinstance(getattr(volumatic, "TIMEFRAME_INTERVALS", None), dict) or set(volumatic.TIMEFRAME_INTERVALS) != {"H4", "D1"}:
    raise RuntimeError("strategy module mismatch: TIMEFRAME_INTERVALS H4/D1 tidak tersedia.")
if (
    not callable(getattr(volumatic, "normalize_trend_length", None))
    or not hasattr(volumatic, "DEFAULT_TREND_LENGTH")
    or int(getattr(volumatic, "DEFAULT_TREND_LENGTH", 0)) != 100
):
    raise RuntimeError(
        "strategy module mismatch: API /length tidak tersedia atau default Trend Length bukan 100. "
        "Sinkronkan main.py dan strategy.py dengan versi Volumatic adjustable length."
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

TIMEZONE_NAME = "Asia/Jakarta"
TZ = ZoneInfo(TIMEZONE_NAME)
BINANCE_REST_BASE = "https://fapi.binance.com"
BINANCE_REAL_REST_BASE = "https://fapi.binance.com"
BINANCE_WS_BASE = "wss://fstream.binance.com/market/stream"
# Binance migrated USD-M streams to separated public/market/private routes in 2026.
BINANCE_PRIVATE_WS_BASE = "wss://fstream.binance.com/private/ws"
BINANCE_WS_API_BASE = "wss://ws-fapi.binance.com/ws-fapi/v1"
BINANCE_TRADING_MODE = "live"
REAL_EXECUTION_ENABLED = (os.getenv("REAL_EXECUTION_ENABLED", "1") or "1").strip().lower() not in {
    "0", "false", "off", "no", "disabled"
}
GITHUB_API = "https://api.github.com"

HISTORY_TRADES_PATH = "data/trades.json"
HISTORY_EVENTS_PATH = "data/events.jsonl"
HISTORY_MARKDOWN_PATH = "data/trade_history.md"
NOTES_PATH = "data/notes.json"
SETUPS_PATH = "data/setups.json"
RUNTIME_SETTINGS_PATH = "data/runtime_settings.json"
DEFERRED_COMMANDS_PATH = "data/deferred_commands.json"
BINANCE_COOLDOWN_PATH = "data/binance_cooldown_state.json"

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
BINANCE_RATE_LIMIT_PRIORITY_SAFETY_SECONDS = 60
BINANCE_BAN_FALLBACK_SECONDS = 120
REAL_PENDING_POLL_SECONDS = 5.0
REAL_RECONCILE_INTERVAL_SECONDS = 20.0
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
REAL_NOTIONAL_MIN_RATIO = Decimal("0.90")
REAL_NOTIONAL_MAX_RATIO = Decimal("1.10")
REAL_FEE_BUFFER = Decimal("0.001")
QTY_FILTER_CODES = (-1013, -1111, -4004, -4005, -4164)
MAX_ACTIVE_TRADES = max(1, int(os.getenv("MAX_ACTIVE_TRADES", "100") or "100"))
SCAN_MAX_PAIRS_PER_CYCLE = 50
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
    lo = max(target * Decimal("0.90") / price, min_qty)
    hi = target * Decimal("1.10") / price
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
    if not (target * Decimal("0.90") <= notional <= target * Decimal("1.10")):
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
    reference_sl: Decimal | None      # reference R only; never a live stop
    entry: Decimal                    # LIMIT price
    expires_at: datetime              # close of the next candle
    signal_close_price: Decimal | None = None  # exact close used as LIMIT price
    ema_filter_length: int | None = 100
    ema_filter_value: Decimal | None = None
    ema_filter_passed: bool | None = None
    status: str = "PENDING"
    created_at: datetime = field(default_factory=now_utc)
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
            "entry": decimal_to_str(self.entry), "signal_close_price": decimal_to_str(self.signal_close_price),
            "ema_filter_length": self.ema_filter_length,
            "ema_filter_value": decimal_to_str(self.ema_filter_value),
            "ema_filter_passed": self.ema_filter_passed,
            "expires_at": iso_utc(self.expires_at),
            "status": self.status, "created_at": iso_utc(self.created_at),
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
            "strategy_name": volumatic.STRATEGY_NAME, "strategy_version": volumatic.STRATEGY_VERSION,
            "entry_reason": "Volumatic Trend visual offset=-1 bubble candle CLOSE; LIMIT valid on adjacent live candle only (mode B)",
            "price_exp_reason": (
                "Normal expiry at next candle close; exception: an already-submitted REAL GTC order "
                "that crosses expiry during Binance REST cooldown is preserved until fill/reversal/manual cancel."
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
        return cls(
            trade_id=str(row.get("trade_id") or generate_trade_id(str(row.get("pair") or "UNKNOWN"))),
            session_id=str(row.get("session_id") or "RECOVERED"),
            pair=normalize_symbol(str(row.get("pair") or "")),
            direction=str(row.get("direction") or "LONG").upper(),
            timeframe=str(row.get("timeframe") or "H4").upper(),
            trend_length=saved_trend_length,
            signal=str(row.get("signal") or ("BLUE" if str(row.get("direction")).upper() == "LONG" else "RED")).upper(),
            signal_time=signal_time,
            bubble_price=dec("bubble_price", dec("entry", Decimal("0"))) or Decimal("0"),
            signal_atr=dec("signal_atr"), reference_sl=dec("reference_sl"),
            entry=dec("entry", Decimal("0")) or Decimal("0"),
            expires_at=parse_iso(row.get("expires_at")) or (
                signal_time + timedelta(milliseconds=volumatic.TIMEFRAME_MS.get(str(row.get("timeframe") or "H4").upper(), volumatic.TIMEFRAME_MS["H4"]))
            ),
            status=str(row.get("status") or "PENDING").upper(),
            created_at=parse_iso(row.get("created_at")) or signal_time,
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
            ema_filter_length=(int(row["ema_filter_length"]) if row.get("ema_filter_length") not in (None, "", 0, "0") else None),
            ema_filter_value=dec("ema_filter_value"),
            ema_filter_passed=(bool(row.get("ema_filter_passed")) if row.get("ema_filter_passed") is not None else None),
            entry_executed_qty=dec("entry_executed_qty"),
            entry_order_status=str(row.get("entry_order_status") or "UNKNOWN").upper(),
            entry_order_terminal=bool(row.get("entry_order_terminal", False)),
            pending_cancel_reason=row.get("pending_cancel_reason"),
            last_ws_event_key=row.get("last_ws_event_key"),
            late_fill_during_api_cooldown=bool(row.get("late_fill_during_api_cooldown", False)),
            preserve_gtc_after_cooldown=bool(row.get("preserve_gtc_after_cooldown", False)),
            exchange_adopted=bool(row.get("exchange_adopted", False)),
            exchange_adoption_source=row.get("exchange_adoption_source"),
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
    "/fapi/v1/klines": 2,  # limit 499 berada pada tier 100-499; header tetap sumber kebenaran
    "/fapi/v1/exchangeInfo": 1,
    "/fapi/v1/time": 1,
    "/fapi/v2/ticker/price": 1,
}


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

    def _restore_persisted_cooldown(self) -> None:
        """Restore a server IP ban and its historical time range."""
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
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
        self.calls: deque[tuple[float, str, int]] = deque(maxlen=50000)
        self.total = 0
        self.limit_hits = 0
        self.last_limit_at: datetime | None = None
        self.used_weight: int | None = None
        self.used_weight_at: float | None = None
        self.max_used_weight = 0

    def record(self, path: str, status: int, used_weight: str | None) -> None:
        now = time.monotonic()
        self.calls.append((now, path, status))
        self.total += 1
        if status in (418, 429):
            self.limit_hits += 1
            self.last_limit_at = now_utc()
        try:
            weight = int(used_weight) if used_weight not in (None, "") else None
        except ValueError:
            weight = None
        if weight is not None:
            self.used_weight = weight
            self.used_weight_at = now
            self.max_used_weight = max(self.max_used_weight, weight)

    def window(self, seconds: float, end: float | None = None) -> dict[str, int]:
        end = end if end is not None else time.monotonic()
        cutoff = end - seconds
        counts: dict[str, int] = {}
        for ts, path, _ in reversed(self.calls):
            if ts > end:
                continue
            if ts < cutoff:
                break
            counts[path] = counts.get(path, 0) + 1
        return counts

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
            return max(base, 5.0)
        if ratio >= 0.85:
            return max(base, 3.0)
        if ratio >= 0.75:
            return max(base, 1.8)
        if ratio >= 0.65:
            return max(base, 1.1)
        if ratio >= 0.50:
            return max(base, 0.70)
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
        self._exchange_info_cache_ttl_seconds = 900.0
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
                    raise BinanceAPIError(
                        f"Transport error Binance REST {path}: {exc}",
                        status_code=None, code=None, endpoint=path,
                    ) from exc
                self._last_request_monotonic = time.monotonic()
                BINANCE_COOLDOWN.record_request()

        used = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
        API_STATS.record(path, status, used)
        code = payload.get("code") if isinstance(payload, dict) else None
        message = payload.get("msg") if isinstance(payload, dict) else body_text
        limited = status in {418, 429} or str(code) in {"-1003", "-1015"} or "too many requests" in str(message).lower()
        if limited:
            server_seconds, known = rate_limit_wait(type("ResponseLike", (), {
                "headers": headers, "status_code": status, "text": body_text,
            })())
            cooldown = max(1.0, server_seconds + BINANCE_RATE_LIMIT_SAFETY_SECONDS)
            self._rest_cooldown_until = max(self._rest_cooldown_until, time.monotonic() + cooldown)
            BINANCE_COOLDOWN.extend(cooldown, f"HTTP {status} {path}: {message}")
            raise BinanceRateLimitError(
                f"Binance REST {path}: HTTP {status}, code={code}, {message}",
                status_code=status or 429,
                code=int(code) if str(code).lstrip("-").isdigit() else None,
                endpoint=path,
                server_cooldown_seconds=server_seconds,
                bot_cooldown_seconds=cooldown,
                retry_after_known=known,
            )
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
            rows = np.load(path, mmap_mode="r", allow_pickle=False)
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
        """Attach a read-only memmap to the bounded RAM LRU without copying its bytes."""
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
        """Fetch historical closed bars plus the current forming kline.

        Deliberately bypasses the closed-bar cache: the last row's close field
        changes intrabar and must represent the current Binance kline snapshot.
        The response is never persisted into the closed-candle cache/disk cache.
        """
        symbol = normalize_symbol(symbol)
        api_interval = self._binance_interval(interval)
        span_ms = int(interval_ms or self._interval_ms(api_interval))
        count = max(3, min(1500, int(count)))
        now_ms = int(time.time() * 1000)
        payload = await self._get("/fapi/v1/klines", {
            "symbol": symbol, "interval": api_interval,
            "endTime": now_ms, "limit": count,
        })
        if not isinstance(payload, list) or not payload:
            raise RuntimeError(f"Binance tidak mengembalikan kline live untuk {symbol} {api_interval}.")
        rows = self._pack_kline_rows(payload, span_ms, now_ms, include_forming=True)
        if rows.size < max(3, min(count - 1, volumatic.ATR_LENGTH + 5)):
            raise RuntimeError(
                f"Kline live {symbol} {api_interval} tidak cukup: {rows.shape[0]}/{count}."
            )
        last_open = int(rows["t"][-1])
        if not (last_open <= now_ms < last_open + span_ms):
            raise RuntimeError(
                f"Tidak ada candle aktif yang valid untuk {symbol} {api_interval}; "
                f"last_open={last_open}, now={now_ms}."
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
        now = time.monotonic()
        ttl = self._kline_cache_ttl.get(api_interval, 60.0)
        cached_exact = self._kline_cache.get(key)
        if cached_exact is None:
            disk_cached = self._load_kline_disk_cache(key, count)
            if disk_cached is not None:
                self._adopt_disk_kline_cache(key, disk_cached)
                cached_exact = self._kline_cache.get(key)

        if cached_exact and cached_exact[1].shape[0] >= count:
            cached_rows = cached_exact[1][-count:]
            if not force_refresh and now - cached_exact[0] <= ttl:
                self._kline_cache.move_to_end(key)
                return cached_rows

            now_ms = int(time.time() * 1000)
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
            if rows.shape[0] < count:
                raise RuntimeError(
                    f"Riwayat incremental Binance {symbol} {api_interval} tidak lengkap: "
                    f"{rows.shape[0]}/{count}."
                )
            self._store_kline_cache(key, rows)
            # Never block the Telegram/WebSocket event loop with disk writes.
            await asyncio.to_thread(self._write_kline_disk_cache, key, self._kline_cache[key][1])
            return self._kline_cache[key][1]

        if not force_refresh:
            reusable = [
                (cache_key, cache_value)
                for cache_key, cache_value in self._kline_cache.items()
                if cache_key[0] == symbol and cache_key[1] == api_interval
                and cache_value[1].shape[0] >= count and now - cache_value[0] <= ttl
            ]
            if reusable:
                cache_key, cached = min(reusable, key=lambda item: item[1][1].shape[0])
                self._kline_cache.move_to_end(cache_key)
                return cached[1][-count:]

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


class BinanceRealClient:
    """Signed USD-M Futures client used only when /real is ON."""

    def __init__(self) -> None:
        self.base_url = BINANCE_REAL_REST_BASE
        self.api_key = BINANCE_API_KEY
        self.api_secret = BINANCE_API_SECRET
        self.recv_window = BINANCE_RECV_WINDOW
        self._time_offset_ms = 0
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
        result = dict(params)
        result.setdefault(
            "timestamp",
            int(time.time() * 1000) + self._time_offset_ms,
        )
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
        payload = self._signed_query(base_params) if signed else base_params
        headers = {"Accept": "application/json"}
        if signed:
            headers["X-MBX-APIKEY"] = self.api_key

        def request_once() -> requests.Response:
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
            return requests.request(
                method.upper(),
                f"{self.base_url}{path}",
                **kwargs,
            )

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
                    raise BinanceAPIError(
                        f"Transport error Binance: {exc}", endpoint=path,
                    ) from exc
                except (OSError, TimeoutError) as exc:
                    raise BinanceAPIError(
                        f"Transport error Binance: {exc}", endpoint=path,
                    ) from exc
                BINANCE_COOLDOWN.record_request()

        API_STATS.record(
            path,
            response.status_code,
            response.headers.get("X-MBX-USED-WEIGHT-1M"),
        )
        code, msg = self._extract_binance_error(response)
        is_rate_limited = (
            response.status_code in {418, 429}
            or code in {-1003, -1015}
            or "too many requests" in str(msg).lower()
            or "way too many requests" in str(msg).lower()
        )
        if is_rate_limited:
            server_seconds, retry_known = rate_limit_wait(response, self._time_offset_ms)
            bot_seconds = max(1.0, server_seconds + BINANCE_RATE_LIMIT_SAFETY_SECONDS)
            self.apply_cooldown(server_seconds, bot_seconds)
            BINANCE_COOLDOWN.extend(bot_seconds, f"HTTP {response.status_code} {path}: {msg}")
            raise BinanceRateLimitError(
                msg,
                status_code=response.status_code or 429,
                code=code,
                endpoint=path,
                server_cooldown_seconds=server_seconds,
                bot_cooldown_seconds=bot_seconds,
                retry_after_known=retry_known,
            )

        if response.status_code >= 400:
            code, msg = self._extract_binance_error(response)
            if code == -1021 and retry_timestamp_error:
                try:
                    server = await self._public_server_time()
                    self._time_offset_ms = int(server) - int(time.time() * 1000)
                    return await self._request(
                        method,
                        path,
                        base_params,
                        signed=signed,
                        retry_timestamp_error=False,
                    )
                except Exception:
                    pass

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

    async def _public_server_time(self) -> int:
        def request() -> int:
            response = requests.get(
                f"{self.base_url}/fapi/v1/time",
                timeout=HTTP_REQUEST_SECONDS,
            )
            API_STATS.record(
                "/fapi/v1/time",
                response.status_code,
                response.headers.get("X-MBX-USED-WEIGHT-1M"),
            )
            response.raise_for_status()
            body = response.json()
            return int(body["serverTime"])

        return await asyncio.to_thread(request)

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
                        timeout=HTTP_REQUEST_SECONDS,
                    )

                try:
                    response = await asyncio.to_thread(request_once)
                except requests.RequestException as exc:
                    raise BinanceAPIError(f"User stream transport error: {exc}", endpoint=path) from exc
                BINANCE_COOLDOWN.record_request()

        API_STATS.record(path, response.status_code, response.headers.get("X-MBX-USED-WEIGHT-1M"))
        code, message = self._extract_binance_error(response)
        is_rate_limited = (
            response.status_code in {418, 429}
            or code in {-1003, -1015}
            or "too many requests" in str(message).lower()
        )
        if is_rate_limited:
            server_seconds, retry_known = rate_limit_wait(response, self._time_offset_ms)
            bot_seconds = max(1.0, server_seconds + BINANCE_RATE_LIMIT_SAFETY_SECONDS)
            self.apply_cooldown(server_seconds, bot_seconds)
            BINANCE_COOLDOWN.extend(bot_seconds, f"HTTP {response.status_code} {path}: {message}")
            raise BinanceRateLimitError(
                message, status_code=response.status_code or 429, code=code, endpoint=path,
                server_cooldown_seconds=server_seconds, bot_cooldown_seconds=bot_seconds,
                retry_after_known=retry_known,
            )
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

    async def place_limit_entry(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        position_side: str,
        client_order_id: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": format_qty(quantity),
            "price": decimal_to_str(price),
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }
        if position_side:
            params["positionSide"] = position_side
        payload = await self._request("POST", "/fapi/v1/order", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons new order Binance tidak valid.", endpoint="/fapi/v1/order")
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

    Stream:
        <symbol>@aggTrade

    Symbol dikirim lowercase sesuai format Binance.
    """

    def __init__(
        self,
        on_price,
    ) -> None:
        self.on_price = on_price

        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._ws = None

        self._desired_symbols: set[str] = set()
        self._subscribed_symbols: set[str] = set()

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
        self._desired_symbols.clear()
        self._last_market_event_key.clear()

    async def set_symbols(self, symbols: set[str] | list[str] | tuple[str, ...]) -> None:
        """Replace the desired stream universe using batched subscribe commands."""
        normalized = {normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()}
        old_desired = set(self._desired_symbols)
        self._desired_symbols = normalized
        if not self.connected or self._ws is None:
            return
        remove = sorted(self._subscribed_symbols - normalized)
        add = sorted(normalized - self._subscribed_symbols)
        # Keep each command comfortably under Binance's stream/message limits.
        batch_size = 200
        for start in range(0, len(remove), batch_size):
            await self._send_unsubscribe(remove[start:start + batch_size])
        for start in range(0, len(add), batch_size):
            await self._send_subscribe(add[start:start + batch_size])
        if old_desired != normalized:
            log.info("[WS] desired market symbols replaced | old=%s new=%s subscribed=%s",
                     len(old_desired), len(normalized), len(self._subscribed_symbols))

    async def add_symbol(self, symbol: str) -> None:
        symbol = normalize_symbol(symbol)

        if symbol in self._desired_symbols:
            return

        # Simbol dicatat dulu di _desired_symbols. Kalau koneksi sedang
        # putus / baru saja ditutup server, simbol ini otomatis
        # di-subscribe ulang oleh _resubscribe_all() saat reconnect,
        # jadi kegagalan kirim di sini BUKAN error fatal.
        self._desired_symbols.add(symbol)

        if self.connected and self._ws is not None:
            sent = await self._send_subscribe([symbol])

            if not sent:
                log.warning(
                    "Subscribe %s ditunda: koneksi WebSocket sedang "
                    "tertutup, akan di-subscribe otomatis saat reconnect.",
                    symbol,
                )

    async def remove_symbol(self, symbol: str) -> None:
        symbol = normalize_symbol(symbol)

        self._desired_symbols.discard(symbol)

        if (
            self.connected
            and self._ws is not None
            and symbol in self._subscribed_symbols
        ):
            await self._send_unsubscribe([symbol])

    async def _send_request(
        self,
        method: str,
        symbols: list[str],
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

        params = [
            f"{symbol.lower()}@aggTrade"
            for symbol in symbols
        ]

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
    ) -> bool:
        sent = await self._send_request(
            "SUBSCRIBE",
            symbols,
        )

        # Hanya tandai subscribed kalau payload benar-benar terkirim.
        if sent:
            for symbol in symbols:
                self._subscribed_symbols.add(
                    normalize_symbol(symbol)
                )

        return sent

    async def _send_unsubscribe(
        self,
        symbols: list[str],
    ) -> bool:
        sent = await self._send_request(
            "UNSUBSCRIBE",
            symbols,
        )

        if sent:
            for symbol in symbols:
                self._subscribed_symbols.discard(
                    normalize_symbol(symbol)
                )

        return sent

    async def _resubscribe_all(self) -> None:
        self._subscribed_symbols.clear()

        symbols = sorted(
            self._desired_symbols
        )

        if symbols:
            await self._send_subscribe(
                symbols
            )

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
                        if "data" in payload and isinstance(
                            payload.get("data"),
                            dict,
                        ):
                            data = payload["data"]
                        else:
                            data = payload

                        if data.get("e") != "aggTrade":
                            continue

                        symbol = normalize_symbol(
                            str(data.get("s") or "")
                        )

                        price_raw = str(
                            data.get("p") or ""
                        )

                        event_time_ms = int(
                            data.get("E")
                            or data.get("T")
                            or int(time.time() * 1000)
                        )

                        # aggTrade memiliki aggregate trade ID ("a").
                        # Simpan bersama event time supaya event lama yang
                        # terlambat datang setelah reconnect tidak diproses
                        # sebagai harga terbaru. Untuk aggTrade normal, ID
                        # tersedia; fallback  -1 hanya untuk payload yang
                        # tidak menyediakannya.
                        try:
                            aggregate_trade_id = int(data.get("a"))
                        except (TypeError, ValueError):
                            aggregate_trade_id = -1

                        if not symbol or not price_raw:
                            continue

                        event_key = (
                            event_time_ms,
                            aggregate_trade_id,
                        )

                        last_key = self._last_market_event_key.get(
                            symbol
                        )

                        if last_key is not None and event_key <= last_key:
                            # Event lama/duplikat. Abaikan agar tick stale tidak
                            # mengubah status fill/expiry atau PnL secara mundur.
                            continue

                        self._last_market_event_key[symbol] = event_key

                        try:
                            price = parse_decimal(
                                price_raw
                            )
                        except ValueError:
                            log.warning(
                                "Harga WebSocket tidak valid: %s",
                                price_raw,
                            )
                            continue

                        await self.on_price(
                            symbol,
                            price,
                            event_time_ms,
                            aggregate_trade_id,
                        )

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

                log.exception(
                    "Error Binance WebSocket: %s",
                    exc,
                )

            finally:
                self.connected = False
                self._ws = None
                self._subscribed_symbols.clear()

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
            BINANCE_WS_API_BASE, ping_interval=None, ping_timeout=None,
            close_timeout=3, max_queue=16,
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
                    url, ping_interval=None, ping_timeout=None, close_timeout=3, max_queue=2048,
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
                        if event_name in {"ORDER_TRADE_UPDATE", "ACCOUNT_UPDATE", "MARGIN_CALL"}:
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
                    log.warning("[USER WS] disconnected: %s", exc)
            except Exception as exc:
                if not self._stop.is_set():
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
        self.ws = BinanceWebSocket(self._on_price)
        self.user_ws = BinanceUserDataWebSocket(self.real, self._on_user_data_event)

        self.real_mode = False  # New REAL entries always require explicit /real on.
        self.margin_usdt = DEFAULT_MARGIN_USDT
        self.leverage = DEFAULT_LEVERAGE
        self.min_margin_enabled = False
        self.max_active_trades = MAX_ACTIVE_TRADES
        self.timeframe = "H4"
        self.trend_length = volumatic.DEFAULT_TREND_LENGTH
        self.ema_filter_length: int | None = volumatic.DEFAULT_EMA_FILTER_LENGTH
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
        self._autostop_data_hold = bool(self.autostop_percent is not None)
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
        # Cached immutable visual-bubble candle context + one adjacent live
        # candle for Mode-B intrabar evaluation. Context values are dictionaries
        # prepared by strategy.py, not full candle arrays.
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
        self._last_user_stream_start_attempt = 0.0

        self._running = False
        self._scan_task: asyncio.Task | None = None
        self._maintenance_task: asyncio.Task | None = None
        self._scan_trigger_now = False
        self._last_scan_boundary_ms = 0
        self._scan_cycle_lock = asyncio.Lock()
        self._trade_lock = asyncio.Lock()
        self._history_lock = asyncio.Lock()
        self._settings_lock = asyncio.Lock()
        self._telegram_error_handler = TelegramErrorHandler(self)
        self._telegram_error_handler_added = False
        self._history_dirty = False
        self._last_market_event_key: dict[str, tuple[int, int]] = {}
        self._last_api_notice = 0.0
        self._api_recovery_until = 0.0
        self._last_universe_refresh = 0.0
        self._local_data_dir = BASE_DIR / "data"
        self._local_data_dir.mkdir(parents=True, exist_ok=True)

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
        if local_data and remote_data and remote_path in {SETUPS_PATH, RUNTIME_SETTINGS_PATH, DEFERRED_COMMANDS_PATH, BINANCE_COOLDOWN_PATH}:
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
            tf = str(payload.get("timeframe") or "H4").upper()
            self.timeframe = tf if tf in volumatic.TIMEFRAME_MS else "H4"
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

    async def _save_runtime_settings(self) -> None:
        async with self._settings_lock:
            payload = {
                "strategy": volumatic.STRATEGY_NAME,
                "timeframe": self.timeframe,
                "trend_length": self.trend_length,
                "ema_filter_length": self.ema_filter_length,
                # Deliberately do not persist scan_enabled. Scanner requires an
                # explicit `/scan on` in every fresh engine session.
                "max_active_trades": self.max_active_trades,
                "margin_usdt": decimal_to_str(self.margin_usdt),
                "leverage": self.leverage,
                "min_margin_enabled": self.min_margin_enabled,
                "autostop_percent": decimal_to_str(self.autostop_percent),
                "autostop_peak_equity": decimal_to_str(self._equity_peak),
                "autostop_last_equity": decimal_to_str(self._equity_last),
                "autostop_triggered": self._autostop_triggered,
                "autostop_source": self._autostop_source,
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
            f"USER WS   : {self.user_ws.status}",
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
            f"CMD QUEUE : {len(self.deferred_commands)} waiting\n"
            "Existing LIMIT orders have been preserved during cooldown."
        )
        self._binance_notice_last_sent = 0.0
        self._binance_notice_status = None
        self._binance_notice_until_epoch = 0.0

    async def _restore_cooldown_from_github(self) -> None:
        """Restore a persisted IP ban before the first Binance REST request."""
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
        """Publish a changed cooldown marker once per ban transition, not per poll."""
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
            await self._sync_ws_symbols()
            await self.reply(
                f"✅ Snapshot lokal dimuat. Trade aktif: {len(self.active_trades)}. "
                f"Binance REST cooldown masih {self._binance_cooldown_remaining():.0f}s; "
                "rekonsiliasi exchange akan berjalan otomatis setelah API pulih."
            )
            return
        await self._reconcile_restored_trades()
        await self._sync_ws_symbols()
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
        self.real_mode = False
        self.scan_enabled = False
        self._scan_trigger_now = False
        self.scan_schedule_slots = ()
        self.scan_schedule_waiting = False
        self.scan_schedule_activated_at = None
        self._scan_schedule_last_fired_minute = None
        self.timeframe = "H4"
        self.trend_length = volumatic.DEFAULT_TREND_LENGTH
        self.ema_filter_length = volumatic.DEFAULT_EMA_FILTER_LENGTH
        self.margin_usdt = DEFAULT_MARGIN_USDT
        self.leverage = DEFAULT_LEVERAGE
        self.min_margin_enabled = False
        self.max_active_trades = MAX_ACTIVE_TRADES
        self.active_trades.clear()
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
        self._scan_resume_timeframe = "H4"
        self._scan_resume_trend_length = self.trend_length
        self._scan_interrupted_by_cooldown = False
        self._was_in_binance_cooldown = False
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
            "[START] FRESH Volumatic session | scanner=OFF | timeframe=H4 | length=%s | "
            "EMA=%s | no persisted engine state imported",
            self.trend_length,
            f"EMA{self.ema_filter_length}" if self.ema_filter_length is not None else "OFF",
        )
        await self.reply(
            "🟢 VOLUMATIC TREND / FRESH SESSION\n\n"
            "Runtime state : CLEAN (tidak memuat data sesi sebelumnya)\n"
            "Timeframe     : H4 (default)\n"
            f"Trend length  : {self.trend_length} (default {volumatic.DEFAULT_TREND_LENGTH})\n"
            f"EMA filter    : {'EMA' + str(self.ema_filter_length) if self.ema_filter_length is not None else 'OFF'}\n"
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
        self.scan_enabled = False
        self._scan_trigger_now = False
        self.scan_schedule_slots = ()
        self.scan_schedule_waiting = False
        self.scan_schedule_activated_at = None
        self._scan_schedule_last_fired_minute = None
        self.real_mode = False

        # Signal every loop/socket worker to stop, then cancel concurrently.
        workers = [self._scan_task, self._maintenance_task, self.ws._task, self.user_ws._task]
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
        try:
            self.real._session.close()
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
        if not self.symbols or time.monotonic() - self._last_universe_refresh > 900:
            self.symbols = await self.rest.get_exchange_info(force=bool(self.symbols))
            self._last_universe_refresh = time.monotonic()
        result = [
            symbol for symbol, meta in self.symbols.items()
            if meta.status == "TRADING" and meta.contract_type == "PERPETUAL"
            and meta.quote_asset == "USDT" and symbol not in EXCLUDED_SYMBOLS
        ]
        return sorted(result)

    async def _send_scan_batch_report(
        self, *, timeframe: str, trend_length: int, ema_length: int | None, batch_number: int,
        batch_start: int, batch_end: int, universe_total: int,
        stats: dict[str, Any], elapsed: float, interrupted: bool = False,
        configuration_changed: bool = False, pass_start: int | None = None,
    ) -> None:
        """Telegram telemetry for every completed or interrupted logical batch."""
        ema = f"EMA{ema_length}" if ema_length is not None else "OFF"
        cooldown = self._binance_cooldown_remaining()
        state = (
            "REST COOLDOWN / PAUSED" if interrupted or cooldown > 0 else
            "CONFIG CHANGED / RESTART" if configuration_changed else
            "BATCH COMPLETE"
        )
        processed = int(stats.get("processed", 0))
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
            f"ENGINE    : VOLUMATIC TREND · {timeframe} · LEN{trend_length} · {ema} · VISUAL OFFSET B",
            f"BATCH     : {batch_number:02d} / COIN #{batch_start:04d}–#{batch_end:04d}",
            f"UNIVERSE  : {universe_total} symbols",
            f"STATE     : {state}",
            "──────────────────────────────",
            f"PASS      : {pass_start:04d}–{min(batch_end, pass_start + requested - 1):04d}  ({requested}/{pass_expected} requested)",
            f"ANALYZED  : {processed}  ·  SKIPPED {skipped}",
            f"BUBBLES   : BLUE {blue}  ·  RED {red}  ·  NONE {no_signal}",
            f"EXECUTION : NEW {created}  ·  FILLED {filled}  ·  DEFERRED {deferred}  ·  UNKNOWN {unknown}",
            f"FILTER    : EMA BLOCK {int(stats.get('ema_blocked', 0))}  ·  MAX BLOCK {int(stats.get('max_blocked', 0))}",
            f"OTHER     : DUPLICATE {int(stats.get('duplicate', 0))}  ·  ERRORS {int(stats.get('errors', 0))}",
            f"DURATION  : {elapsed:.1f}s",
        ]
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
        elif configuration_changed:
            lines.append("NEXT      : mengulang siklus dengan parameter terbaru dari coin #0001")
        elif batch_end < universe_total:
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
            f"TIMEFRAME : {self.timeframe}\n"
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
                    await asyncio.sleep(min(10.0, max(2.0, cooldown)))
                    continue
                if self._scan_interrupted_by_cooldown:
                    self._scan_interrupted_by_cooldown = False
                    self._scan_trigger_now = True
                    log.info("[SCAN] REST pulih; lanjutkan scanner dari cursor tersimpan.")
                interval_ms = volumatic.TIMEFRAME_MS[self.timeframe]
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
            timeframe = self.timeframe
            trend_length = self.trend_length
            ema_filter_length = self.ema_filter_length
            interval_ms = volumatic.TIMEFRAME_MS[timeframe]
            universe = await self._universe()
            self._scanner_symbols = set(universe) if self.scan_enabled else set()
            if self.scan_enabled:
                await self._sync_ws_symbols()
            started = time.monotonic()
            processed_total = 0
            errors_total = 0
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
                await self.reply(
                    "╔══ VOL / SCANNER ══╗\n"
                    "UNIVERSE : 0 symbols\n"
                    "ACTION   : tidak ada pair USDT Perpetual yang memenuhi filter."
                )
                return

            log.info(
                "[SCAN] cycle start | tf=%s | length=%s | ema=%s | signal_mode=VISUAL_OFFSET_INTRABAR_B | universe=%s | batch_size=%s | cursor=%s",
                timeframe, trend_length, f"EMA{ema_filter_length}" if ema_filter_length is not None else "OFF",
                len(universe), SCAN_MAX_PAIRS_PER_CYCLE, self._scan_resume_cursor,
            )
            await self._expire_due_pending()

            # Logical batches remain fixed at [1..50], [51..100], etc., even
            # when a cooldown resumes from a cursor in the middle of a batch.
            offset = self._scan_resume_cursor
            while offset < len(universe):
                if not self._running or not self.scan_enabled:
                    break
                if self.timeframe != timeframe or self.trend_length != trend_length:
                    log.info("[SCAN] timeframe/trend length changed mid-cycle; restart next cycle.")
                    self._scan_resume_cursor = 0
                    self._scan_resume_boundary_ms = 0
                    break

                logical_start = (offset // SCAN_MAX_PAIRS_PER_CYCLE) * SCAN_MAX_PAIRS_PER_CYCLE
                logical_end = min(logical_start + SCAN_MAX_PAIRS_PER_CYCLE, len(universe))
                batch = universe[offset:logical_end]
                batch_number = logical_start // SCAN_MAX_PAIRS_PER_CYCLE + 1
                batch_start = logical_start + 1
                batch_end = logical_end
                pass_start = offset + 1
                batch_started = time.monotonic()
                stats: dict[str, Any] = {
                    "requested": 0, "processed": 0, "already_seen": 0, "warmup": 0,
                    "stale": 0, "no_signal": 0, "blue": 0, "red": 0,
                    "orders": 0, "filled": 0, "deferred": 0, "unknown": 0,
                    "ema_blocked": 0, "max_blocked": 0, "duplicate": 0,
                    "errors": 0, "details": [],
                }
                interrupted = False
                configuration_changed = False
                log.info(
                    "[SCAN] batch start | batch=%02d | logical_coins=%04d-%04d/%04d | pass_from=%04d | tf=%s | len=%s | ema=%s",
                    batch_number, batch_start, batch_end, len(universe), pass_start,
                    timeframe, trend_length, ema_filter_length or "OFF",
                )

                for local_index, symbol in enumerate(batch):
                    absolute_index = offset + local_index
                    if (self.timeframe != timeframe or self.trend_length != trend_length
                            or self.ema_filter_length != ema_filter_length):
                        configuration_changed = True
                        self._scan_resume_cursor = 0
                        self._scan_resume_boundary_ms = 0
                        break
                    if self._binance_cooldown_remaining() > 0:
                        self._scan_interrupted_by_cooldown = True
                        self._scan_trigger_now = True
                        interrupted = True
                        self._scan_resume_cursor = absolute_index
                        break

                    stats["requested"] += 1
                    try:
                        api_interval = volumatic.TIMEFRAME_INTERVALS[timeframe]
                        candles = await self.rest.get_klines_with_forming(
                            symbol, api_interval, self._kline_history_limit(trend_length), interval_ms
                        )
                        # Settings may change while the REST call is in flight.
                        # Do not evaluate this symbol with a stale cycle snapshot.
                        if (self.timeframe != timeframe or self.trend_length != trend_length
                                or self.ema_filter_length != ema_filter_length):
                            configuration_changed = True
                            self._scan_resume_cursor = 0
                            self._scan_resume_boundary_ms = 0
                            break
                        if candles is None or len(candles) < max(volumatic.ATR_LENGTH + 10, trend_length + 10):
                            stats["warmup"] += 1
                            self._scan_resume_cursor = absolute_index + 1
                            continue

                        # MODE B: the last row is the currently-forming candle.
                        # Pine draws trendFlip at offset=-1, i.e. on the previous
                        # closed candle. That previous candle is the bubble candle;
                        # its CLOSE is the LIMIT price. The current row is the only
                        # candle in which that visual-offset signal may be traded.
                        entry_open_ms = int(candles["t"][-1])
                        bubble_open_ms = int(candles["t"][-2])
                        bubble_close_ms = entry_open_ms
                        expiry_ms = entry_open_ms + interval_ms
                        key = f"{timeframe}:{symbol}"
                        previous_close = self.last_processed_candles.get(key)
                        if previous_close is not None and entry_open_ms <= previous_close:
                            stats["already_seen"] += 1
                            self._scan_resume_cursor = absolute_index + 1
                            continue

                        now_ms = int(time.time() * 1000)
                        if not (entry_open_ms <= now_ms < expiry_ms) or bubble_open_ms + interval_ms != entry_open_ms:
                            self.last_processed_candles[key] = entry_open_ms
                            stats["stale"] += 1
                            self._scan_resume_cursor = absolute_index + 1
                            continue

                        self._cache_live_signal_context(
                            symbol, timeframe, candles, trend_length=trend_length,
                            ema_filter_length=ema_filter_length
                        )
                        context_key = f"{timeframe}:{symbol}"
                        live_context = self._live_signal_candles.get(context_key)
                        signal = volumatic.evaluate_live_visual_context(
                            live_context, float(candles["c"][-1]), now_ms=now_ms
                        )
                        if signal is None:
                            stats["no_signal"] += 1
                            # One snapshot per active entry candle. If the B-mode
                            # visual bubble is absent during this scan, do not chase
                            # it in a later candle.
                            self.last_processed_candles[key] = entry_open_ms
                        else:
                            bubble = str(signal.get("signal") or "?").upper()
                            direction = str(signal.get("direction") or "?").upper()
                            if bubble == "BLUE":
                                stats["blue"] += 1
                            elif bubble == "RED":
                                stats["red"] += 1
                            try:
                                outcome = await self._dispatch_mode_b_signal(
                                    signal, expiry_ms, source="SCANNER"
                                )
                            except Exception as signal_exc:
                                self._signal_retries[key] = {
                                    "signal": signal, "expiry_ms": expiry_ms, "key": key,
                                    "attempts": 0, "last_error": str(signal_exc)[:500],
                                    "source": "SCANNER",
                                }
                                stats["errors"] += 1
                                errors_total += 1
                                outcome = "ERROR_RETRY"
                                log.exception(
                                    "[SIGNAL] %s %s failed; queued for safe retry: %s",
                                    symbol, bubble, signal_exc,
                                )

                            if outcome == "ORDER_CREATED":
                                stats["orders"] += 1
                            if outcome == "ORDER_FILLED":
                                stats["filled"] += 1
                            elif outcome == "ORDER_DEFERRED":
                                stats["deferred"] += 1
                            elif outcome == "ORDER_UNKNOWN":
                                stats["unknown"] += 1
                            elif outcome == "EMA_BLOCKED":
                                stats["ema_blocked"] += 1
                            elif outcome == "MAX_BLOCKED":
                                stats["max_blocked"] += 1
                            elif outcome in {"DUPLICATE", "IGNORED_TIMEFRAME"}:
                                stats["duplicate"] += 1

                            close_price = signal.get("close_price") or signal.get("signal_close_price") or "-"
                            ema_len = signal.get("ema_filter_length", ema_filter_length)
                            ema_val = signal.get("ema_filter_value")
                            ema_text = "EMA OFF" if ema_len is None else f"EMA{ema_len}={ema_val if ema_val is not None else 'N/A'}"
                            closed_old = bool(signal.get("_closed_opposite"))
                            close_note = f"CLOSE {signal.get('_closed_direction', 'OPPOSITE')} MARKET → " if closed_old else ""
                            trace = f"#{absolute_index + 1:04d} {symbol} {bubble}/{direction} LEN{trend_length} VISUAL-CLOSE={close_price} · {ema_text} · {close_note}{outcome}"
                            stats["details"].append(trace[:190])

                            self.last_processed_candles[key] = entry_open_ms
                            if outcome != "ERROR_RETRY":
                                self._signal_retries.pop(key, None)

                        stats["processed"] += 1
                        processed_total += 1
                        self._scan_resume_cursor = absolute_index + 1

                    except BinanceRateLimitError as exc:
                        stats["errors"] += 1
                        errors_total += 1
                        interrupted = True
                        self._scan_interrupted_by_cooldown = True
                        self._scan_trigger_now = True
                        self._scan_resume_cursor = absolute_index
                        await self._notify_binance_cooldown(exc, source="SCANNER / KLINES OR ORDER", symbol=symbol)
                        log.info("[SCAN] paused at symbol=%s | resume_cursor=%s", symbol, self._scan_resume_cursor)
                        break
                    except Exception as exc:
                        stats["errors"] += 1
                        errors_total += 1
                        self._scan_resume_cursor = absolute_index + 1
                        log.exception("[SCAN] %s failed: %s", symbol, exc)

                    if SCAN_PAIR_DELAY_SECONDS > 0:
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)

                release_memory()
                await self._send_scan_batch_report(
                    timeframe=timeframe, trend_length=trend_length, ema_length=ema_filter_length,
                    batch_number=batch_number, batch_start=batch_start, batch_end=batch_end,
                    universe_total=len(universe), stats=stats,
                    elapsed=time.monotonic() - batch_started, interrupted=interrupted,
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
                if self._scan_resume_cursor >= logical_end:
                    offset = logical_end
                    log.info("[SCAN] batch complete | batch=%02d | range=%04d-%04d", batch_number, batch_start, batch_end)
                else:
                    break

            completed = self._scan_resume_cursor >= len(universe)
            if completed:
                self._scan_resume_cursor = 0
                self._scan_resume_boundary_ms = 0
                self._scan_resume_timeframe = timeframe
            await self._save_runtime_settings()
            await self._save_active_state()
            await self._sync_ws_symbols()
            if self._scan_interrupted_by_cooldown or self._binance_cooldown_remaining() > 0:
                self._scan_trigger_now = True
            elapsed = time.monotonic() - started
            log.info(
                "[SCAN] cycle %s | tf=%s | processed=%s/%s | errors=%s | cursor=%s | duration=%.1fs",
                "complete" if completed else "paused", timeframe,
                processed_total, len(universe), errors_total, self._scan_resume_cursor, elapsed,
            )

    async def _dispatch_mode_b_signal(
        self, signal: dict[str, Any], expiry_ms: int, *, source: str
    ) -> str:
        """Deduplicate one visual bubble and route it through the trade manager.

        The latch is per pair/timeframe/adjacent candle/direction. A rate-limit
        response creates exactly one in-memory retry item and leaves the latch
        armed, preventing every incoming aggTrade tick from retrying the same
        Binance action while the REST governor is closed.
        """
        pair = normalize_symbol(signal.get("symbol") or "")
        timeframe = str(signal.get("timeframe") or self.timeframe).upper()
        direction = str(signal.get("direction") or "").upper()
        visual_open = int(signal.get("bubble_visual_time_ms") or 0)
        entry_open = int(signal.get("entry_candle_open_time_ms") or 0)
        token = f"{entry_open}:{str(signal.get('signal') or direction).upper()}"
        key = f"{timeframe}:{pair}"
        seen = self._live_signal_seen.setdefault(key, set())
        if token in seen:
            return "DUPLICATE"
        seen.add(token)
        try:
            outcome = await self._handle_signal(signal, int(expiry_ms))
        except BinanceRateLimitError as exc:
            self._signal_retries[key] = {
                "signal": dict(signal), "expiry_ms": int(expiry_ms), "key": key,
                "attempts": 0, "last_error": str(exc)[:500],
                "source": str(source),
            }
            self._was_in_binance_cooldown = True
            await self._notify_binance_cooldown(
                exc, source=f"{source} / MODE-B BUBBLE", symbol=pair
            )
            await self._record_event(
                "VISUAL_BUBBLE_DEFERRED_RATE_LIMIT", None,
                pair=pair, timeframe=timeframe, direction=direction,
                signal=signal.get("signal"), visual_candle_close_price=signal.get("close_price"),
                limit_price=signal.get("limit_price"), entry_candle_open_ms=entry_open,
                expiry_ms=int(expiry_ms), source=source,
            )
            return "ORDER_DEFERRED"
        except Exception:
            # Keep the latch after an ambiguous/error outcome. The retry queue
            # or order reconciliation owns recovery; do not emit duplicate POSTs.
            raise

        await self._record_event(
            "VISUAL_BUBBLE_DISPATCHED", None,
            pair=pair, timeframe=timeframe, direction=direction,
            signal=signal.get("signal"),
            visual_candle_open_ms=visual_open,
            visual_candle_close_ms=signal.get("visual_candle_close_time_ms"),
            visual_candle_close_price=signal.get("visual_candle_close_price", signal.get("close_price")),
            limit_price=signal.get("limit_price"),
            limit_price_source=signal.get("limit_price_source", "VISUAL_BUBBLE_CANDLE_CLOSE"),
            entry_candle_open_ms=entry_open, expiry_ms=int(expiry_ms),
            detection_price=signal.get("detection_price"), repaint_risk=True,
            source=source, outcome=outcome,
        )
        return outcome

    async def _handle_signal(self, signal: dict[str, Any], expiry_ms: int) -> str:
        pair = normalize_symbol(signal["symbol"])
        direction = str(signal["direction"]).upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError(f"Arah sinyal tidak valid: {direction}")
        # Even if the adjacent entry candle has just expired, an opposite
        # bubble must still close/cancel the old exposure. A later freshness
        # guard below prevents placing a stale new LIMIT.
        async with self._trade_lock:
            if str(signal.get("signal_mode") or "").upper() == "VISUAL_OFFSET_INTRABAR_B":
                visual_open = int(signal.get("bubble_visual_time_ms") or 0)
                entry_open = int(signal.get("entry_candle_open_time_ms") or 0)
                entry_close = int(signal.get("entry_candle_close_time_ms") or 0)
                interval = volumatic.TIMEFRAME_MS.get(str(signal.get("timeframe") or "").upper(), 0)
                if not interval or entry_open != visual_open + interval or entry_close != entry_open + interval:
                    raise ValueError("Identitas candle visual/entry Mode B tidak konsisten.")
                visual_close_ms = int(
                    signal.get("visual_candle_close_time_ms")
                    or signal.get("bubble_visual_close_time_ms")
                    or 0
                )
                if int(signal.get("signal_close_time_ms") or 0) != entry_open:
                    raise ValueError("Signal close Mode B harus sama dengan open candle entry berikutnya.")
                if visual_close_ms != entry_open:
                    raise ValueError("Candle close visual bubble tidak tepat sebelum candle entry.")
                if int(expiry_ms) != entry_close:
                    raise ValueError("Expiry Mode B tidak sama dengan close candle entry.")
            existing = next((t for t in self.active_trades.values() if t.pair == pair and t.status in {"PENDING", "FILLED"}), None)
            closed_opposite = False
            closed_direction = existing.direction if existing and existing.direction != direction else None
            # A trade keeps the timeframe that created it. A newly selected
            # timeframe must not reverse or otherwise hijack that existing
            # position; only its own timeframe can manage its lifecycle.
            if (existing and not existing.exchange_adopted
                    and existing.timeframe != str(signal.get("timeframe") or self.timeframe).upper()):
                log.info(
                    "[SIGNAL] %s %s %s diabaikan: trade aktif memakai timeframe %s.",
                    pair, direction, signal.get("timeframe"), existing.timeframe,
                )
                await self._record_event(
                    "SIGNAL_IGNORED_DIFFERENT_TIMEFRAME_ACTIVE", existing,
                    signal_direction=direction, signal_timeframe=signal.get("timeframe"),
                )
                return "IGNORED_TIMEFRAME"
            if existing and existing.direction == direction:
                log.info("[SIGNAL] %s %s diabaikan: trade arah sama masih aktif.", pair, direction)
                await self._record_event("DUPLICATE_SIGNAL_IGNORED", existing, signal=signal.get("signal"))
                return "DUPLICATE"
            # Opposite confirmed bubble closes a filled position immediately.
            # A pending order is canceled before a new direction is submitted.
            if existing:
                if existing.status == "PENDING":
                    await self._cancel_pending_trade(existing, "OPPOSITE_BUBBLE_CANCEL")
                    if existing.status == "PENDING" and existing.trade_id in self.active_trades:
                        raise RuntimeError(
                            f"Order lama {pair} belum terkonfirmasi canceled/terminal; reversal ditahan."
                        )
                    closed_opposite = existing.trade_id not in self.active_trades
                if existing.status == "FILLED":
                    # A partial entry can have filled exposure plus an open remainder.
                    # Cancel only the remainder before market-closing the live position.
                    if existing.real_enabled and not existing.entry_order_terminal:
                        await self._cancel_real_entry_remainder(existing, "OPPOSITE_BUBBLE_CANCEL_REMAINDER")
                    signal_close_price = Decimal(str(signal.get("close_price") or signal.get("signal_close_price") or "0"))
                    market_reference_price = self._current_price(pair)
                    if market_reference_price is None:
                        try:
                            market_reference_price = await self.rest.get_price(pair)
                        except BinanceRateLimitError:
                            raise
                        except Exception:
                            market_reference_price = None
                    if market_reference_price is None:
                        market_reference_price = signal_close_price
                    if market_reference_price <= 0:
                        raise RuntimeError(f"Harga market untuk close {pair} tidak valid.")
                    await self._close_trade(existing, market_reference_price, "OPPOSITE_BUBBLE_MARKET")
                    closed_opposite = existing.trade_id not in self.active_trades
                elif existing.trade_id in self.active_trades:
                    raise RuntimeError(f"Order lama {pair} belum selesai direkonsiliasi; reversal ditahan.")
            # EMA is a symmetric directional entry filter only. Opposite positions
            # above have already been canceled/market-closed and must stay closed
            # even when this new direction is rejected by EMA.
            filter_length_raw = signal.get("ema_filter_length", self.ema_filter_length)
            try:
                filter_length = volumatic.normalize_ema_filter_length(filter_length_raw)
            except ValueError:
                filter_length = self.ema_filter_length
            filter_value_raw = signal.get("ema_filter_value")
            filter_value: Decimal | None = None
            if filter_value_raw not in (None, ""):
                try:
                    candidate_filter_value = Decimal(str(filter_value_raw))
                    if candidate_filter_value.is_finite() and candidate_filter_value > 0:
                        filter_value = candidate_filter_value
                except (InvalidOperation, ValueError, TypeError):
                    filter_value = None
            filter_passed = bool(signal.get("ema_filter_passed", False))
            filter_ready = bool(signal.get("ema_filter_ready", filter_length is None))
            if filter_length is not None and (not filter_ready or filter_value is None or not filter_passed):
                await self._record_event(
                    "SIGNAL_REJECTED_BY_EMA", None, pair=pair,
                    direction=direction, signal=signal.get("signal"),
                    timeframe=signal.get("timeframe"),
                    ema_filter_length=filter_length,
                    ema_filter_value=decimal_to_str(filter_value),
                    signal_close_price=str(signal.get("close_price") or signal.get("signal_close_price") or ""),
                    reason=("EMA_NOT_READY" if not filter_ready or filter_value is None else "CLOSE_NOT_ON_ALLOWED_SIDE"),
                )
                log.info(
                    "[SIGNAL] %s %s ditolak EMA%s | close=%s ema=%s ready=%s; opposite exposure sudah diselesaikan.",
                    pair, direction, filter_length,
                    signal.get("close_price"), filter_value, filter_ready,
                )
                await self.reply(
                    f"⛔ VOLUMATIC {direction} DIBLOKIR EMA{filter_length} • {pair}\n"
                    f"Signal close: {signal.get('close_price')} | EMA{filter_length}: "
                    f"{decimal_to_str(filter_value) if filter_value is not None else 'WARMING UP'}\n"
                    "Posisi/order arah berlawanan sudah ditutup/dibatalkan; entry baru tidak dibuat."
                )
                signal["_closed_opposite"] = closed_opposite
                signal["_closed_direction"] = closed_direction
                return "EMA_BLOCKED"
            if int(time.time() * 1000) >= int(expiry_ms):
                await self._record_event(
                    "SIGNAL_EXPIRED_DURING_REVERSAL", None, pair=pair,
                    direction=direction, timeframe=signal.get("timeframe"),
                )
                signal["_closed_opposite"] = closed_opposite
                signal["_closed_direction"] = closed_direction
                return "EXPIRED"
            active_count = sum(t.status in {"PENDING", "FILLED"} for t in self.active_trades.values())
            if active_count >= self.max_active_trades:
                log.warning("[SIGNAL] %s %s dilewati: kapasitas /max penuh (%s/%s).", pair, direction, active_count, self.max_active_trades)
                await self._record_event("SIGNAL_SKIPPED_MAX_CAPACITY", None, pair=pair, direction=direction, timeframe=signal.get("timeframe"), active_count=active_count)
                signal["_closed_opposite"] = closed_opposite
                signal["_closed_direction"] = closed_direction
                return "MAX_BLOCKED"

            bubble_price = Decimal(str(signal["bubble_price"]))  # EMA1 on the visually marked bubble candle
            if str(signal.get("signal_mode") or "").upper() == "VISUAL_OFFSET_INTRABAR_B":
                # Hard invariant for Mode B: price source is exclusively the
                # close of the candle on which Pine visually paints offset=-1.
                visual_close_raw = signal.get(
                    "visual_candle_close_price", signal.get("close_price")
                )
                signal_close_price = Decimal(str(visual_close_raw or "0"))
                declared_limit = Decimal(str(signal.get("limit_price") or "0"))
                ordinary_close = Decimal(str(signal.get("close_price") or "0"))
                if signal_close_price <= 0 or declared_limit != signal_close_price or ordinary_close != signal_close_price:
                    raise ValueError(
                        "Mode B ditolak: LIMIT/close harus identik dengan close candle visual bubble, "
                        "bukan close candle aktif atau harga EMA1."
                    )
            else:
                signal_close_price = Decimal(str(
                    signal.get("limit_price") or signal.get("signal_close_price")
                    or signal.get("close_price") or "0"
                ))
            if signal_close_price <= 0:
                raise ValueError("Close candle sinyal tidak tersedia atau tidak valid.")
            meta = self.symbols.get(pair)
            if meta is None:
                raise ValueError(f"Metadata Binance untuk {pair} tidak ditemukan.")
            # MODE B: LIMIT is anchored to the CLOSE of the candle where Pine
            # visually draws the offset=-1 bubble. The adjacent candle is live
            # now and defines the one-candle entry expiry. This accepts intrabar
            # repaint/lookahead behavior by explicit user choice.
            # Respect tickSize without making the order price worse than target:
            # BUY rounds down, SELL rounds up.
            limit_price = quantized_price(signal_close_price, meta.tick_size) if direction == "LONG" else quantized_price_ceiling(signal_close_price, meta.tick_size)
            if limit_price <= 0:
                raise ValueError("Harga limit dari close candle setelah pembulatan tidak valid.")
            signal_close = datetime.fromtimestamp(int(signal["signal_close_time_ms"]) / 1000, tz=timezone.utc)
            expiry_dt = datetime.fromtimestamp(int(expiry_ms) / 1000, tz=timezone.utc)
            atr_value = Decimal(str(signal["signal_atr"])) if np.isfinite(float(signal.get("signal_atr", float("nan")))) else None
            reference_value = Decimal(str(signal["reference_sl"])) if np.isfinite(float(signal.get("reference_sl", float("nan")))) else None
            trade = Trade(
                trade_id=generate_trade_id(pair), session_id=self.session_id, pair=pair,
                direction=direction, timeframe=str(signal["timeframe"]),
                trend_length=volumatic.normalize_trend_length(signal.get("trend_length", self.trend_length)),
                signal=str(signal["signal"]), signal_time=signal_close,
                bubble_price=bubble_price, signal_atr=atr_value, reference_sl=reference_value,
                entry=limit_price, expires_at=expiry_dt, signal_close_price=signal_close_price,
                ema_filter_length=filter_length,
                ema_filter_value=filter_value,
                ema_filter_passed=(bool(filter_passed) if filter_length is not None else None),
                margin_usdt=self.margin_usdt,
                leverage=self.leverage, min_margin_mode=self.min_margin_enabled,
                target_notional=self.margin_usdt * Decimal(self.leverage),
            )
            if self.real_mode:
                # Complete all preflight checks before reserving the trade. No
                # exchange order has been sent if this method raises.
                await self._prepare_real_limit(trade)
            else:
                if trade.min_margin_mode:
                    paper_filters = {
                        "step_size": meta.step_size, "min_qty": meta.min_qty,
                        "max_qty": meta.max_qty, "min_notional": meta.min_notional,
                        "max_notional": meta.max_notional, "min_price": meta.min_price,
                        "max_price": meta.max_price,
                    }
                    quantity, notional, minimum_margin, reason = compute_minimum_order(
                        trade.entry, trade.leverage, paper_filters
                    )
                    if quantity is None or notional is None or minimum_margin is None:
                        raise RuntimeError(f"Minimum margin {pair} tidak valid: {reason}")
                    trade.quantity = quantity
                    trade.target_notional = notional
                    trade.actual_notional = notional
                    trade.margin_usdt = minimum_margin
                else:
                    trade.quantity = (trade.target_notional or (trade.margin_usdt * trade.leverage)) / trade.entry
                    trade.actual_notional = trade.quantity * trade.entry
                trade.real_enabled = False
                trade.real_state = "PAPER_PENDING"
            self.active_trades[trade.trade_id] = trade
            await self._record_event(
                "LIMIT_SUBMITTED", trade, order_type="LIMIT",
                signal_mode=signal.get("signal_mode", "VISUAL_OFFSET_INTRABAR_B"),
                trend_length=trade.trend_length,
                bubble_visual_time_ms=signal.get("bubble_visual_time_ms"),
                visual_candle_close_time_ms=signal.get("visual_candle_close_time_ms"),
                visual_candle_close_price=signal.get("visual_candle_close_price", signal.get("close_price")),
                limit_price_source=signal.get("limit_price_source", "VISUAL_BUBBLE_CANDLE_CLOSE"),
                entry_candle_open_time_ms=signal.get("entry_candle_open_time_ms"),
                entry_candle_close_time_ms=signal.get("entry_candle_close_time_ms"),
                repaint_risk=bool(signal.get("repaint_risk", True)),
                entry=decimal_to_str(trade.entry), expires_at=iso_utc(trade.expires_at),
                real=trade.real_enabled, quantity=decimal_to_str(trade.quantity),
                notional=decimal_to_str(trade.target_notional or trade.actual_notional),
                margin_usdt=decimal_to_str(trade.margin_usdt), leverage=trade.leverage,
                min_margin_mode=trade.min_margin_mode,
            )
            await self._save_active_state()
            await self.ws.add_symbol(pair)
            if trade.real_enabled:
                # The prepared trade is now durable. A timeout after POST can
                # therefore be reconciled by clientOrderId without duplicating it.
                await self._submit_real_limit(trade)
                if trade.status == "PENDING" and trade.real_state == "REAL_FILLED":
                    await self._mark_filled(trade, trade.fill_price or trade.entry, source="BINANCE_ORDER_RESPONSE")
                if trade.status not in {"PENDING", "FILLED"}:
                    return "NOT_ACTIVE"
            else:
                observed = self._current_price(pair)
                if observed is not None:
                    touched = observed <= trade.entry if trade.direction == "LONG" else observed >= trade.entry
                    if touched:
                        await self._mark_filled(trade, observed, "PAPER_MARKETABLE_AT_SUBMIT")
            signal["_closed_opposite"] = closed_opposite
            signal["_closed_direction"] = closed_direction
            await self.reply(
                f"{'🟢' if direction == 'LONG' else '🔴'} VOLUMATIC {direction} • {pair}\n\n"
                f"Timeframe: {trade.timeframe} • Volumatic Length: {trade.trend_length}\nSignal: {trade.signal} BUBBLE\n"
                f"Bubble line (EMA1): {fmt_price(trade.bubble_price)}\n"
                f"VISUAL bubble candle close: {fmt_price(trade.signal_close_price or trade.entry)}\n"
                f"Bubble candle close time: {format_wib(signal_close)}\n"
                f"LIMIT entry: {fmt_price(trade.entry)} (source: visual candle close)\n"
                f"Entry candle window: {format_wib(signal_close)} → {format_wib(trade.expires_at)}\n"
                f"EMA filter: {('OFF' if trade.ema_filter_length is None else f'EMA{trade.ema_filter_length} @ {fmt_price(trade.ema_filter_value)}')}\n"
                f"Margin: {trade.margin_usdt} USDT • Leverage: {trade.leverage}x\n"
                f"Sizing: {'MINIMUM AUTO' if trade.min_margin_mode else 'FIXED TARGET'}\n"
                f"Quantity: {format_qty(trade.quantity or Decimal('0'))}\n"
                f"Expiry: {format_wib(trade.expires_at)}\n"
                f"Mode: {'REAL' if trade.real_enabled else 'PAPER'}\n"
                "SL: NONE • TP: NONE"
            )
            if trade.real_submit_ambiguous:
                return "ORDER_UNKNOWN"
            if trade.real_enabled and trade.real_state == "REAL_PREPARED":
                return "ORDER_DEFERRED"
            return "ORDER_CREATED" if trade.status == "PENDING" else "ORDER_FILLED"

    async def _prepare_real_limit(self, trade: Trade) -> None:
        """Run all deterministic account/pair checks without submitting an order."""
        if not REAL_EXECUTION_ENABLED:
            raise RuntimeError("REAL_EXECUTION_ENABLED=0; eksekusi LIVE dikunci oleh environment.")
        if not self.real.configured:
            raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET belum lengkap.")
        if now_utc() >= trade.expires_at:
            raise RuntimeError("Candle entry sudah berakhir; sinyal tidak dikejar.")
        filters = await self.rest.get_symbol_filters(trade.pair)
        leverage_response = await self.real.set_leverage(trade.pair, trade.leverage)
        leverage_max_notional = parse_max_notional(leverage_response)
        current_symbol_max = Decimal(str(filters.get("max_notional") or "0"))
        if leverage_max_notional is not None:
            filters["max_notional"] = (
                min(current_symbol_max, leverage_max_notional) if current_symbol_max > 0
                else leverage_max_notional
            )
        dual = await self.real.ensure_position_mode()
        positions = await self.real.get_positions(trade.pair)
        if positions:
            raise RuntimeError(f"{trade.pair} masih memiliki posisi Binance yang tidak diadopsi trade ini; entry baru diblokir.")
        open_orders = await self.real.get_open_orders(trade.pair)
        if open_orders:
            raise RuntimeError(f"{trade.pair} masih memiliki open order Binance; entry baru diblokir untuk mencegah order ganda.")
        balance_rows = await self.real.get_futures_balance()
        usdt = next((x for x in balance_rows if str(x.get("asset", "")).upper() == "USDT"), {})
        available = Decimal(str(usdt.get("availableBalance") or usdt.get("balance") or "0"))
        if trade.min_margin_mode:
            quantity, notional, minimum_margin, reason = compute_minimum_order(
                trade.entry, trade.leverage, filters, available, leverage_max_notional
            )
            if quantity is None or notional is None or minimum_margin is None:
                raise RuntimeError(f"Minimum margin REAL ditolak: {reason}")
            trade.margin_usdt = minimum_margin
            trade.target_notional = notional
            trade.actual_notional = notional
        else:
            quantity, reason = compute_quantity(trade.entry, trade.margin_usdt, trade.leverage, filters, available)
            if quantity is None:
                raise RuntimeError(f"Ukuran order REAL ditolak: {reason}")
            trade.actual_notional = quantity * trade.entry
        trade.quantity = quantity
        trade.position_side = ("LONG" if trade.direction == "LONG" else "SHORT") if dual else "BOTH"
        trade.real_enabled = True
        trade.real_state = "REAL_PREPARED"
        trade.entry_client_order_id = f"VT{uuid4().hex[:28]}"
        trade.real_error = None

    async def _submit_real_limit(self, trade: Trade) -> None:
        """Submit once; ambiguous transport/5xx outcomes are lookup-only, never blind-reposted."""
        if not trade.real_enabled or not trade.entry_client_order_id:
            raise RuntimeError("REAL order belum melewati preflight atau clientOrderId tidak tersedia.")
        if now_utc() >= trade.expires_at:
            await self._finalize_trade(trade, "EXPIRED", None, "CANDLE_EXPIRED_BEFORE_REAL_SUBMIT")
            return
        if trade.real_submit_ambiguous or trade.real_state in {"REAL_SUBMIT_UNKNOWN", "REAL_SUBMITTING"}:
            # This clientOrderId may already identify a live or completed exchange order.
            # Never POST again until exchange state is positively resolved.
            await self._save_active_state()
            return

        trade.real_state = "REAL_SUBMITTING"
        await self._save_active_state()
        try:
            response = await self.real.place_limit_entry(
                symbol=trade.pair, side=trade.entry_side, quantity=trade.quantity or Decimal("0"),
                price=trade.entry, position_side=trade.position_side or "BOTH",
                client_order_id=trade.entry_client_order_id,
            )
        except BinanceRateLimitError as exc:
            # An internally blocked request was never sent. A server 429/418 is a
            # definite rate-limit rejection. Keep the durable intent PREPARED and
            # let maintenance submit the same clientOrderId after cooldown.
            if getattr(exc, "internal", False) or exc.status_code in {418, 429}:
                trade.real_state = "REAL_PREPARED"
                trade.real_error = f"Order belum diterima Binance karena rate limit: {str(exc)[:350]}"
                trade.real_submit_ambiguous = False
                await self._save_active_state()
                await self._notify_binance_cooldown(exc, source="REAL ORDER SUBMIT", symbol=trade.pair)
                log.info("[REAL] %s POST ditahan oleh rate governor; intent disimpan PREPARED, clientOrderId=%s", trade.pair, trade.entry_client_order_id)
                return
            raise
        except BinanceAPIError as exc:
            if exc.status_code is not None and 400 <= exc.status_code < 500 and exc.status_code not in {418, 429}:
                trade.real_state = "REAL_REJECTED"
                trade.real_error = str(exc)[:500]
                await self._save_active_state()
                await self._record_event("REAL_LIMIT_REJECTED", trade, error=str(exc)[:500])
                await self._finalize_trade(trade, "CANCELED", None, f"BINANCE_REJECTED: {exc}")
                return
            # Binance explicitly documents 5xx/unknown execution results as ambiguous.
            trade.real_state = "REAL_SUBMIT_UNKNOWN"
            trade.real_submit_ambiguous = True
            trade.real_error = str(exc)[:500]
            await self._save_active_state()
            log.info("[REAL/RECON] %s acknowledgement uncertain; lookup-only by clientOrderId=%s; duplicate POST disabled | %s", trade.pair, trade.entry_client_order_id, exc)
            await self.reply(
                "╔══ VOL / REAL ORDER RECONCILIATION ══╗\n"
                f"PAIR      : {trade.pair} · {trade.direction}\n"
                f"CLIENT ID : {trade.entry_client_order_id}\n"
                "STATE     : EXCHANGE ACK NOT CONFIRMED\n"
                "NEXT      : QUERY EXISTING CLIENT ORDER ID\n"
                "SAFETY    : POST REPEAT DISABLED · DUPLICATE ORDER BLOCKED\n"
                "SLOT      : RESERVED UNTIL EXCHANGE STATE IS KNOWN"
            )
            # Resolve a lost POST acknowledgement immediately using GET only.
            # If Binance still cannot find the clientOrderId, retain UNKNOWN and
            # let the normal maintenance loop repeat lookup; never POST again.
            try:
                response = await self.real.get_order(
                    trade.pair, client_order_id=trade.entry_client_order_id
                )
            except BinanceRateLimitError as query_exc:
                await self._notify_binance_cooldown(query_exc, source="REAL ORDER VERIFY", symbol=trade.pair)
                return
            except BinanceAPIError as query_exc:
                if query_exc.code == -2013:
                    trade.real_error = "Lookup awal belum menemukan clientOrderId; status UNKNOWN dipertahankan, GET akan diulang berkala."
                else:
                    trade.real_error = (trade.real_error + " | lookup: " + str(query_exc))[:500]
                await self._save_active_state()
                return
            except Exception as query_exc:
                trade.real_error = (trade.real_error + " | lookup: " + str(query_exc))[:500]
                await self._save_active_state()
                return
        except Exception as exc:
            # Timeouts/resets can happen after Binance accepted the POST.
            trade.real_state = "REAL_SUBMIT_UNKNOWN"
            trade.real_submit_ambiguous = True
            trade.real_error = str(exc)[:500]
            await self._save_active_state()
            log.info("[REAL/RECON] %s acknowledgement uncertain; lookup-only by clientOrderId=%s; duplicate POST disabled | %s", trade.pair, trade.entry_client_order_id, exc)
            await self.reply(
                "╔══ VOL / REAL ORDER RECONCILIATION ══╗\n"
                f"PAIR      : {trade.pair} · {trade.direction}\n"
                f"CLIENT ID : {trade.entry_client_order_id}\n"
                "STATE     : EXCHANGE ACK NOT CONFIRMED\n"
                "NEXT      : QUERY EXISTING CLIENT ORDER ID\n"
                "SAFETY    : POST REPEAT DISABLED · DUPLICATE ORDER BLOCKED\n"
                "SLOT      : RESERVED UNTIL EXCHANGE STATE IS KNOWN"
            )
            try:
                response = await self.real.get_order(trade.pair, client_order_id=trade.entry_client_order_id)
            except BinanceRateLimitError as query_exc:
                await self._notify_binance_cooldown(query_exc, source="REAL ORDER VERIFY", symbol=trade.pair)
                return
            except BinanceAPIError as query_exc:
                if query_exc.code == -2013:
                    trade.real_error = "Order belum ditemukan pada lookup pertama; status tetap UNKNOWN dan akan dicek ulang."
                    await self._save_active_state()
                    return
                trade.real_error = (trade.real_error + " | lookup: " + str(query_exc))[:500]
                await self._save_active_state()
                return
            except Exception as query_exc:
                trade.real_error = (trade.real_error + " | lookup: " + str(query_exc))[:500]
                await self._save_active_state()
                return

        trade.real_submit_ambiguous = False
        trade.entry_order_id = int(response["orderId"]) if response.get("orderId") is not None else trade.entry_order_id
        status = str(response.get("status") or "UNKNOWN").upper()
        executed = Decimal(str(response.get("executedQty") or "0"))
        trade.entry_order_status = status
        trade.entry_executed_qty = executed
        if status in {"FILLED", "PARTIALLY_FILLED"} and executed > 0:
            trade.quantity = executed
            avg = Decimal(str(response.get("avgPrice") or trade.entry))
            trade.fill_price = avg if avg > 0 else trade.entry
            trade.entry_order_terminal = status == "FILLED"
            trade.real_state = "REAL_FILLED" if status == "FILLED" else "REAL_PARTIALLY_FILLED"
            trade.real_error = None
        elif status == "NEW":
            trade.entry_order_terminal = False
            trade.real_state = "REAL_PENDING"
            trade.real_error = None
        elif status in {"CANCELED", "EXPIRED", "REJECTED"}:
            if executed > 0:
                trade.quantity = executed
                avg = Decimal(str(response.get("avgPrice") or trade.entry))
                trade.fill_price = avg if avg > 0 else trade.entry
                trade.real_state = "REAL_FILLED"
            else:
                await self._finalize_trade(
                    trade, "EXPIRED" if status == "EXPIRED" else "CANCELED",
                    None, f"BINANCE_SUBMIT_RESPONSE_{status}",
                )
                return
        else:
            trade.real_state = "REAL_SUBMIT_UNKNOWN"
            trade.real_submit_ambiguous = True
            trade.real_error = f"Status submit Binance tidak dikenal: {status}"
        await self._save_active_state()
        log.info(
            "[REAL] %s %s LIMIT client_id=%s order_id=%s status=%s",
            trade.pair, trade.direction, trade.entry_client_order_id, trade.entry_order_id, status,
        )

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

    async def _mark_filled(self, trade: Trade, price: Decimal, source: str) -> None:
        if trade.status != "PENDING":
            return
        trade.status = "FILLED"
        trade.filled_at = now_utc()
        trade.fill_price = price if price > 0 else trade.entry
        if trade.quantity is None:
            trade.quantity = (trade.target_notional or trade.margin_usdt * trade.leverage) / trade.fill_price
        trade.actual_notional = trade.quantity * trade.fill_price
        if trade.leverage > 0:
            trade.margin_usdt = trade.actual_notional / Decimal(trade.leverage)
        risk = volumatic.reference_risk(trade.direction, float(trade.fill_price), float(trade.reference_sl)) if trade.reference_sl is not None else None
        trade.initial_risk = Decimal(str(risk)) if risk is not None else None
        trade.real_state = "REAL_FILLED" if trade.real_enabled else "PAPER_FILLED"
        await self._record_event("ORDER_FILLED", trade, fill_price=decimal_to_str(trade.fill_price), source=source)
        await self._save_active_state()
        log.info("[TRADE] FILLED %s %s @ %s (%s)", trade.pair, trade.direction, trade.fill_price, source)

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
                if executed > 0:
                    trade.quantity = executed
                    avg = self._order_fill_price(order, trade.fill_price or trade.entry)
                    if trade.status == "PENDING":
                        await self._mark_filled(trade, avg, "BINANCE_CANCEL_RECONCILED_FILL")
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
        if trade.real_enabled:
            trade.pending_cancel_reason = reason
            await self._cancel_real_entry_remainder(trade, reason)
            return
        await self._finalize_trade(
            trade, "EXPIRED" if "EXPIRE" in reason else "CANCELED", None, reason
        )

    async def _close_trade(self, trade: Trade, reference_price: Decimal, reason: str) -> None:
        if trade.status != "FILLED":
            return
        exit_price = reference_price
        if trade.real_enabled:
            try:
                # A partially-filled GTC entry may still have an unfilled
                # remainder. Cancel and confirm that remainder before MARKET
                # close so it cannot fill later and unexpectedly reopen exposure.
                if not trade.entry_order_terminal:
                    await self._cancel_real_entry_remainder(
                        trade, "CLOSE_POSITION_CANCEL_ENTRY_REMAINDER"
                    )
                position = await self.real.get_position(trade.pair, trade.entry_side)
                if position is not None:
                    response = await self.real.place_market_close(
                        symbol=trade.pair, position=position, entry_direction=trade.entry_side
                    )
                    avg = Decimal(str(response.get("avgPrice") or "0"))
                    if avg > 0:
                        exit_price = avg
                    # A successful REST response is not enough: verify that no
                    # exchange exposure remains before recording close/reversal.
                    remaining = await self.real.get_positions(trade.pair)
                    if remaining:
                        raise RuntimeError(
                            f"Market close {trade.pair} belum flat; reversal/ledger close diblokir."
                        )
                else:
                    remaining = await self.real.get_positions(trade.pair)
                    if remaining:
                        raise RuntimeError(
                            "Posisi Binance tidak cocok dengan arah trade; reversal diblokir."
                        )
                    log.warning("[REAL] %s posisi tidak ditemukan; kemungkinan ditutup manual.", trade.pair)
            except Exception as exc:
                trade.real_error = str(exc)[:500]
                await self._save_active_state()
                raise
        await self._finalize_trade(trade, "CLOSED", exit_price, reason)

    async def _finalize_trade(self, trade: Trade, result: str, exit_price: Decimal | None, reason: str) -> None:
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
        self.active_trades.pop(trade.trade_id, None)
        record = trade.to_record()
        self.history_records.append(record)
        await self._record_event("POSITION_CLOSED" if result == "CLOSED" else f"ORDER_{result}", trade, reason=reason, exit_price=decimal_to_str(exit_price))
        await self._persist_markdown_trade(record)
        await self._save_active_state()
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
            if trade.real_enabled and trade.preserve_gtc_after_cooldown:
                # Already-submitted REAL GTC orders preserved during a ban are
                # intentionally not auto-canceled after REST becomes available.
                continue

            if trade.status == "PENDING" and now >= trade.expires_at:
                if trade.real_enabled:
                    cooldown_active = self._binance_cooldown_remaining() > 0
                    # A PREPARED order has never been submitted to Binance. Never
                    # turn an expired local intent into a late exchange order.
                    if trade.real_state == "REAL_PREPARED" or not trade.entry_client_order_id:
                        if cooldown_active:
                            continue
                        await self._finalize_trade(
                            trade, "EXPIRED", None, "REAL_ORDER_NEVER_SUBMITTED_BEFORE_CANDLE_EXPIRY"
                        )
                        continue
                    expired_inside_ban = BINANCE_COOLDOWN.covered_by_last_ban(trade.expires_at.timestamp())
                    order_may_exist = bool(trade.entry_client_order_id)
                    if order_may_exist and (cooldown_active or expired_inside_ban):
                        trade.preserve_gtc_after_cooldown = True
                        trade.pending_cancel_reason = None
                        trade.real_state = "REAL_GTC_PRESERVED_AFTER_COOLDOWN"
                        trade.real_error = (
                            "Order GTC dipertahankan: candle expiry terjadi selama Binance REST cooldown. "
                            "Order tetap aktif sampai fill, opposite bubble, atau pembatalan manual."
                        )
                        await self._record_event(
                            "ORDER_GTC_PRESERVED_AFTER_API_COOLDOWN", trade,
                            order_status=trade.entry_order_status,
                            expires_at=iso_utc(trade.expires_at),
                            ban_until_epoch=BINANCE_COOLDOWN._until_epoch,
                        )
                        await self._save_active_state()
                        log.info("[REAL] GTC preserved | pair=%s | expiry during REST cooldown; order remains on Binance until fill/reversal/cancel", trade.pair)
                        continue
                await self._cancel_pending_trade(trade, "ORDER_EXPIRED_ONE_CANDLE")

            elif trade.status == "FILLED" and trade.real_enabled and not trade.entry_order_terminal and now >= trade.expires_at:
                cooldown_active = self._binance_cooldown_remaining() > 0
                expired_inside_ban = BINANCE_COOLDOWN.covered_by_last_ban(trade.expires_at.timestamp())
                if cooldown_active or expired_inside_ban:
                    trade.preserve_gtc_after_cooldown = True
                    trade.pending_cancel_reason = None
                    trade.real_state = "REAL_PARTIAL_GTC_PRESERVED_AFTER_COOLDOWN"
                    trade.real_error = (
                        "Sisa quantity LIMIT GTC dipertahankan setelah partial fill karena candle expiry "
                        "terjadi selama Binance REST cooldown."
                    )
                    await self._record_event(
                        "PARTIAL_ENTRY_REMAINDER_GTC_PRESERVED_AFTER_API_COOLDOWN", trade,
                        order_status=trade.entry_order_status,
                        executed_qty=decimal_to_str(trade.entry_executed_qty),
                    )
                    await self._save_active_state()
                else:
                    await self._cancel_real_entry_remainder(trade, "ORDER_EXPIRED_ONE_CANDLE")

    async def _reconcile_pending_real(self, trade: Trade) -> None:
        if not trade.real_enabled or trade.status != "PENDING":
            return
        if self._binance_cooldown_remaining() > 0:
            return  # open Binance LIMIT remains untouched; private WS still reports fills.
        if trade.real_state == "REAL_PREPARED":
            # Startup was allowed to finish during a ban; now resume the one
            # already-persisted submission with its original clientOrderId.
            if now_utc() >= trade.expires_at:
                await self._finalize_trade(trade, "EXPIRED", None, "PREPARED_ORDER_WINDOW_EXPIRED_DURING_COOLDOWN")
                return
            await self._submit_real_limit(trade)
            if trade.status == "PENDING" and trade.real_state == "REAL_FILLED":
                await self._mark_filled(trade, trade.fill_price or trade.entry, "DEFERRED_PREPARED_SUBMISSION")
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
                reason = "ORDER_EXPIRED_ONE_CANDLE" if now_utc() >= trade.expires_at else "RETRY_CANCEL_UNKNOWN"
                await self._cancel_pending_trade(trade, reason)
                return
            raise
        except Exception:
            if (now_utc() >= trade.expires_at and not trade.preserve_gtc_after_cooldown) or trade.real_state == "REAL_CANCEL_UNKNOWN":
                reason = "ORDER_EXPIRED_ONE_CANDLE" if now_utc() >= trade.expires_at else "RETRY_CANCEL_UNKNOWN"
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
            await self._mark_filled(trade, self._order_fill_price(order, trade.entry), "BINANCE_POLL")
            trade.real_state = "REAL_FILLED"
            await self._save_active_state()
        elif status == "PARTIALLY_FILLED":
            # The filled amount is live exposure. Normally cancel the remaining
            # LIMIT at one-candle expiry/reversal; if that expiry occurred during
            # a REST IP ban, preserve the GTC remainder until fill/reversal/manual cancel.
            trade.entry_order_terminal = False
            if executed > 0:
                trade.quantity = executed
                await self._mark_filled(trade, self._order_fill_price(order, trade.entry), "BINANCE_POLL_PARTIAL")
            trade.real_state = "REAL_PARTIALLY_FILLED"
            await self._save_active_state()
        elif status in {"NEW", "UNKNOWN"} and (
            (now_utc() >= trade.expires_at and not trade.preserve_gtc_after_cooldown)
            or trade.real_state == "REAL_CANCEL_UNKNOWN"
        ):
            # Retry a failed manual/expiry cancellation until Binance confirms
            # a terminal order state. An ambiguous order keeps its capacity slot.
            reason = "ORDER_EXPIRED_ONE_CANDLE" if now_utc() >= trade.expires_at else "RETRY_CANCEL_UNKNOWN"
            await self._cancel_pending_trade(trade, reason)
        elif status in {"CANCELED", "EXPIRED", "REJECTED"}:
            executed = Decimal(str(order.get("executedQty") or "0"))
            if executed > 0:
                trade.quantity = executed
                await self._mark_filled(trade, self._order_fill_price(order, trade.entry), "BINANCE_TERMINAL_PARTIAL_FILL")
            else:
                await self._finalize_trade(
                    trade, "EXPIRED" if status == "EXPIRED" else "CANCELED",
                    None, f"BINANCE_{status}",
                )
        elif now_utc() >= trade.expires_at and not trade.preserve_gtc_after_cooldown:
            await self._cancel_pending_trade(trade, "ORDER_EXPIRED_ONE_CANDLE")

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
            trade.fill_price = avg
            trade.actual_notional = executed * avg
            trade.margin_usdt = trade.actual_notional / Decimal(max(1, trade.leverage))
            if trade.reference_sl is not None:
                risk = volumatic.reference_risk(
                    trade.direction, float(avg), float(trade.reference_sl)
                )
                trade.initial_risk = Decimal(str(risk)) if risk is not None else None
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
        if (not trade.entry_order_terminal and trade.expires_at <= now_utc()
                and not trade.preserve_gtc_after_cooldown):
            await self._cancel_real_entry_remainder(trade, trade.pending_cancel_reason or "ORDER_EXPIRED_ONE_CANDLE")
            if trade.status != "FILLED":
                return
        if time.monotonic() - trade.last_real_check_monotonic < REAL_RECONCILE_INTERVAL_SECONDS:
            return
        trade.last_real_check_monotonic = time.monotonic()

        # A late/partial GTC that was deliberately preserved during an API ban
        # remains live until fill, reversal, or manual cancel. Poll its exchange
        # status after REST recovers, but never auto-cancel it merely because the
        # original one-candle deadline has passed.
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
                if trade.preserve_gtc_after_cooldown:
                    # A preserved live GTC is still allowed to fill; do not
                    # remove it only because the position endpoint is momentarily flat.
                    return
                await self._cancel_real_entry_remainder(
                    trade, "EXTERNAL_POSITION_MISSING_CANCEL_ENTRY_REMAINDER"
                )
            remaining = await self.real.get_positions(trade.pair)
            if not remaining:
                price = self._current_price(trade.pair) or trade.fill_price or trade.entry
                log.warning("[REAL] %s %s exposure hilang di Binance; menutup ledger sebagai external close.", trade.pair, trade.direction)
                await self._finalize_trade(trade, "CLOSED", price, "EXTERNAL_CLOSE_RECONCILED")

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
                            await self._submit_real_limit(trade)
                            if trade.status == "PENDING" and trade.real_state == "REAL_FILLED":
                                await self._mark_filled(trade, trade.fill_price or trade.entry, "STARTUP_SUBMIT_RESUME")
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
                # Local-only guard runs even while Binance REST is rate-limited.
                await self._autostop_check()
                cooldown = self._binance_cooldown_remaining()
                if cooldown > 0:
                    self._was_in_binance_cooldown = True
                    await self._sync_cooldown_marker_to_github()
                    # Local PAPER expiration still happens. REAL orders are left
                    # untouched on Binance; WS events continue updating the ledger.
                    await self._expire_due_pending()
                    await asyncio.sleep(min(10.0, max(2.0, cooldown)))
                    continue

                if self._was_in_binance_cooldown:
                    await self._sync_cooldown_marker_to_github()
                    self._last_universe_refresh = 0.0
                    if self.scan_enabled:
                        self._scan_trigger_now = True
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

                # Refresh the REAL wallet snapshot at a low cadence. Autostop itself
                # never calls REST and therefore remains live during an IP cooldown.
                has_real_exposure = self.real_mode or any(
                    t.real_enabled and t.status in {"PENDING", "FILLED"}
                    for t in self.active_trades.values()
                )
                balance_age = (
                    time.monotonic() - self.real.last_balance_at
                    if self.real.last_balance_at is not None else float("inf")
                )
                if has_real_exposure and balance_age >= AUTOSTOP_BALANCE_REFRESH_SECONDS:
                    try:
                        await self.real.get_futures_balance()
                        await self._autostop_check(force=True)
                    except BinanceRateLimitError as exc:
                        # Cached equity + WebSocket prices continue while REST is held.
                        self._was_in_binance_cooldown = True
                        await self._notify_binance_cooldown(exc, source="AUTOSTOP / BALANCE")
                    except Exception as exc:
                        log.info("[AUTOSTOP] balance snapshot refresh ditunda; guard tetap armed: %s", exc)

                # Commands are FIFO and only one API-dependent command per pass. /real off
                # remains immediate and can cancel a queued /real on.
                await self._process_deferred_commands()

                # Refresh universe only when scanner needs it. ExchangeInfo is
                # cached for 15 minutes; no forced repeat on every recovery pass.
                if self.scan_enabled and (not self.symbols or time.monotonic() - self._last_universe_refresh > 900):
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
                for trade in list(self.active_trades.values()):
                    if self._binance_cooldown_remaining() > 0:
                        break
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
                await asyncio.sleep(MAINTENANCE_POLL_SECONDS)
            except asyncio.CancelledError:
                break
            except BinanceRateLimitError as exc:
                self._was_in_binance_cooldown = True
                await self._notify_binance_cooldown(exc, source="MAINTENANCE")
                log.info("[MAINTENANCE] REST maintenance paused by shared rate governor.")
                await asyncio.sleep(3.0)
            except Exception:
                log.exception("[MAINTENANCE] loop error")
                await asyncio.sleep(3)

    def _kline_history_limit(self, trend_length: int | None = None) -> int:
        """Return enough closed history for the selected trend EMA and ATR warm-up.

        The default 100-length keeps the existing 499-row request. Longer lengths
        expand history to roughly four EMA lengths, capped at 1,500 Binance rows.
        """
        length = volumatic.normalize_trend_length(
            self.trend_length if trend_length is None else trend_length
        )
        needed = max(KLINE_HISTORY, volumatic.ATR_LENGTH + 10, length * 4 + 10)
        return min(1500, int(needed))

    def _cache_live_signal_context(
        self, symbol: str, timeframe: str, candles: np.ndarray,
        *, trend_length: int | None = None, ema_filter_length: int | None = None,
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
        """Refresh one active candle per position; WebSocket handles intrabar ticks.

        On each new H4/D1 bar, fetch the current forming kline once, cache its
        closed history, and evaluate immediately. Subsequent live ticks update
        the cached current close and re-evaluate the Pine offset=-1 bubble
        without repeatedly polling REST.
        """
        by_key: dict[str, Trade] = {}
        trend_length = self.trend_length
        ema_filter_length = self.ema_filter_length
        for trade in self.active_trades.values():
            if trade.status in {"PENDING", "FILLED"}:
                by_key.setdefault(f"{trade.timeframe}:{trade.pair}", trade)

        now_ms = int(time.time() * 1000)
        for key, seed_trade in list(by_key.items()):
            tf = seed_trade.timeframe if seed_trade.timeframe in volumatic.TIMEFRAME_MS else "H4"
            span = volumatic.TIMEFRAME_MS[tf]
            active_open_ms = (now_ms // span) * span
            checkpoint = max(
                self.last_position_candles.get(key, 0),
                int(seed_trade.signal_time.timestamp() * 1000),
            )
            if active_open_ms <= checkpoint:
                # A current context might have been cached by the scanner when
                # the entry was created. No additional REST request is needed.
                if key in self._live_signal_candles:
                    continue
            try:
                candles = await self.rest.get_klines_with_forming(
                    seed_trade.pair, "4h" if tf == "H4" else "1d",
                    self._kline_history_limit(trend_length), span
                )
                if candles is None or len(candles) < max(volumatic.ATR_LENGTH + 10, trend_length + 10):
                    continue
                live_open = int(candles["t"][-1])
                live_close = live_open + span
                if live_open != active_open_ms or not (live_open <= now_ms < live_close):
                    continue

                self._cache_live_signal_context(
                    seed_trade.pair, tf, candles, trend_length=trend_length,
                    ema_filter_length=ema_filter_length
                )
                signal = volumatic.evaluate_live_visual_context(
                    self._live_signal_candles.get(key), float(candles["c"][-1]), now_ms=now_ms
                )
                if signal is not None:
                    existing = next((
                        t for t in self.active_trades.values()
                        if t.pair == seed_trade.pair and t.status in {"PENDING", "FILLED"}
                    ), None)
                    opposite = existing is not None and existing.direction != signal["direction"]
                    outcome = await self._dispatch_mode_b_signal(
                        signal, live_close, source="POSITION_MONITOR"
                    )
                    await self._record_event(
                        "POSITION_VISUAL_BUBBLE_EVALUATED", None,
                        pair=seed_trade.pair, timeframe=tf,
                        signal=signal.get("signal"), direction=signal.get("direction"),
                        visual_candle_close=signal.get("close_price"),
                        limit_price=signal.get("limit_price"),
                        entry_candle_open_ms=live_open,
                        outcome=outcome, opposite_position=bool(opposite), repaint_risk=True,
                    )
                self.last_position_candles[key] = live_open
            except BinanceRateLimitError:
                # Non-signal monitoring REST was rate-limited; do not advance the
                # checkpoint so the same active candle can be fetched later.
                raise
            except Exception as exc:
                log.warning(
                    "[POSITION MONITOR] %s %s gagal dianalisis/dieksekusi: %s",
                    seed_trade.pair, tf, exc, exc_info=True,
                )

    async def _retry_signals(self) -> None:
        if not self._signal_retries or self._binance_cooldown_remaining() > 0:
            return
        now = now_utc()
        for key, item in list(self._signal_retries.items()):
            if int(time.time() * 1000) >= int(item["expiry_ms"]):
                try:
                    # Stale Mode-B signals must never open a late LIMIT, but an
                    # opposite bubble can still close/cancel old exposure.
                    outcome = await self._handle_signal(item["signal"], int(item["expiry_ms"]))
                except BinanceRateLimitError as exc:
                    self._was_in_binance_cooldown = True
                    await self._notify_binance_cooldown(exc, source="EXPIRED SIGNAL / OPPOSITE CLOSE", symbol=item["signal"].get("symbol"))
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

    async def _on_price(self, symbol: str, price: Decimal, event_time_ms: int, aggregate_trade_id: int = -1) -> None:
        symbol = normalize_symbol(symbol)
        key = (int(event_time_ms), int(aggregate_trade_id))
        previous_key = self._last_market_event_key.get(symbol)
        if previous_key is not None and key <= previous_key:
            return
        self._last_market_event_key[symbol] = key
        self.prices[symbol] = PriceSnapshot(symbol, price, int(event_time_ms), now_utc())
        for trade in list(self.active_trades.values()):
            if trade.pair != symbol or trade.status != "PENDING" or trade.real_enabled:
                continue
            if now_utc() >= trade.expires_at:
                continue
            touched = price <= trade.entry if trade.direction == "LONG" else price >= trade.entry
            if touched:
                await self._mark_filled(trade, trade.entry, "PAPER_WEBSOCKET_LIMIT_TOUCH")
        await self._evaluate_live_visual_bubble(symbol, price, int(event_time_ms))

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
        timeframe_ms = volumatic.TIMEFRAME_MS.get(self.timeframe, volumatic.TIMEFRAME_MS["H4"])
        direction = direction.upper()
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
            expires_at=now_utc() + timedelta(milliseconds=timeframe_ms),
            signal_close_price=entry,
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
                await self._mark_filled(trade, trade.fill_price, f"BINANCE_ACCOUNT_UPDATE_{source}")
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
                or item.exchange_adopted
            )
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
                preserve_gtc=True,
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
            trade.preserve_gtc_after_cooldown = True
            if executed > 0:
                trade.quantity = executed
                try:
                    avg = Decimal(str(row.get("avgPrice") or row.get("ap") or price))
                    if avg > 0:
                        trade.fill_price = avg
                except (InvalidOperation, ValueError):
                    pass
                if trade.status == "PENDING":
                    await self._mark_filled(trade, trade.fill_price or price, f"BINANCE_OPEN_ORDER_SNAPSHOT_{source}")
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
        """Incrementally update the position projection from ACCOUNT_UPDATE.a.P."""
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
            if old_direction:
                trade = self._find_real_trade(symbol, old_direction)
                if trade and trade.exchange_adopted and trade.status == "FILLED":
                    close_price = self._current_price(symbol)
                    if close_price is None:
                        try:
                            close_price = Decimal(str(old.get("markPrice") or old.get("entryPrice") or trade.fill_price or trade.entry))
                        except Exception:
                            close_price = trade.fill_price or trade.entry
                    await self._finalize_trade(trade, "CLOSED", close_price, "ACCOUNT_UPDATE_POSITION_FLAT_WS")
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
                    preserve_gtc=True,
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
                    note="GTC order dibiarkan berjalan ketika REST banned; fill diterima melalui User Data WS setelah expiry one-candle.",
                )
            trade.entry_executed_qty = cumulative
            trade.quantity = cumulative
            trade.fill_price = fill_price
            trade.actual_notional = cumulative * fill_price
            trade.margin_usdt = trade.actual_notional / Decimal(max(1, trade.leverage))
            if trade.status == "PENDING":
                await self._mark_filled(trade, fill_price, f"BINANCE_USER_WS_{status}")
            else:
                risk = volumatic.reference_risk(trade.direction, float(fill_price), float(trade.reference_sl)) if trade.reference_sl is not None else None
                trade.initial_risk = Decimal(str(risk)) if risk is not None else None
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
                trade.pending_cancel_reason = trade.pending_cancel_reason or "ORDER_EXPIRED_ONE_CANDLE"
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
        desired = {
            t.pair for t in self.active_trades.values()
            if t.status in {"PENDING", "FILLED"}
        }
        desired.update(symbol for symbol, _side in self.exchange_positions)
        desired.update(
            normalize_symbol(str(order.get("symbol") or ""))
            for order in self.exchange_open_orders.values()
            if order.get("symbol")
        )
        if self.scan_enabled:
            desired.update(self._scanner_symbols)
        await self.ws.set_symbols(desired)
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
            # These are the same account-wide REST snapshots /real on already
            # needed for its safety preflight. Reuse them to seed the in-memory
            # exchange projection; do not restore old local session/history files.
            account_snapshot = await self.real.get_account_info()
            all_open_orders = await self.real.get_open_orders(None)
            await self.real.get_account_config()
            await self.real.get_futures_balance()
            adopted_positions, adopted_orders = await self._adopt_exchange_snapshot(
                account_snapshot, all_open_orders, source="REAL_ON_ACCOUNT_SNAPSHOT"
            )
            # Private stream becomes the live source of incremental order/position
            # updates. No REST request is made by /trade after this point.
            await self.user_ws.start()
            self._last_user_ws_reconnect_count = self.user_ws.reconnect_count
            self.real_mode = True
            if self.autostop_percent is not None and self._equity_peak is None:
                await self._autostop_check(force=True)
            await self.reply(
                "🟢 REAL ON. Order baru akan dikirim ke Binance USD-M Futures LIVE.\n"
                f"Live snapshot : {adopted_positions} posisi · {adopted_orders} open order dimuat dari Binance (tanpa restore file lama).\n"
                f"Private User Data WS: {self.user_ws.status}. ORDER_TRADE_UPDATE + ACCOUNT_UPDATE menjaga /trade realtime; /trade tidak memanggil REST.\n"
                "Posisi/order yang sudah ada tetap dikelola walaupun /real off."
            )
        elif arg == "off":
            self.real_mode = False
            await self.reply("🟡 REAL OFF untuk entry baru. Pending/posisi REAL yang sudah ada tetap direkonsiliasi dan dikelola sampai selesai.")
        else:
            await self.reply(f"REAL: {'ON' if self.real_mode else 'OFF'}\nGunakan /real on atau /real off.")

    async def _handle_length(self, text: str) -> None:
        parts = str(text or "").split()
        if len(parts) == 1:
            await self.reply(
                f"VOLUMATIC TREND LENGTH: {self.trend_length}\n"
                f"Default fresh session: {volumatic.DEFAULT_TREND_LENGTH}\n"
                f"Range: {volumatic.MIN_TREND_LENGTH}–{volumatic.MAX_TREND_LENGTH}\n\n"
                "Gunakan /length 100 atau /length <angka>.\n"
                "Length berlaku untuk EMA-SWMA dan EMA close Volumatic Trend; ATR tetap 200.\n"
                "Perubahan berlaku pada sinyal baru dan scanner akan membangun ulang konteks candle aktif."
            )
            return
        if len(parts) != 2:
            await self.reply(f"Gunakan /length <angka> (1–{volumatic.MAX_TREND_LENGTH}).")
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

    async def _handle_ema(self, text: str) -> None:
        parts = str(text or "").split()
        if len(parts) == 1:
            active = f"EMA{self.ema_filter_length}" if self.ema_filter_length is not None else "OFF"
            await self.reply(
                f"Directional EMA filter: {active}\n"
                "BLUE/LONG hanya jika candle close > EMA.\n"
                "RED/SHORT hanya jika candle close < EMA.\n"
                "Opposite position tetap ditutup meskipun entry baru ditolak.\n\n"
                "Pilihan: /EMA 50 | /EMA 100 | /EMA 200 | /EMA off"
            )
            return
        if len(parts) != 2:
            await self.reply("Gunakan /EMA 50, /EMA 100, /EMA 200, atau /EMA off.")
            return
        try:
            selected = volumatic.normalize_ema_filter_length(parts[1])
        except ValueError:
            await self.reply("❌ EMA hanya mendukung 50, 100, 200, atau off.")
            return
        old = self.ema_filter_length
        self.ema_filter_length = selected
        # Invalidate cached visual-candle filters. If scanner is armed, rebuild
        # contexts from current candle data without waiting for another H4/D1 close.
        self._live_signal_candles.clear()
        self._live_signal_last_eval.clear()
        self._live_signal_seen.clear()
        self.last_processed_candles.clear()
        self._scan_resume_cursor = 0
        self._scan_resume_boundary_ms = 0
        self._scan_trigger_now = bool(self.scan_enabled and not self._autostop_triggered)
        await self._save_runtime_settings()
        old_text = f"EMA{old}" if old is not None else "OFF"
        new_text = f"EMA{selected}" if selected is not None else "OFF"
        if selected is None:
            message = (
                f"✅ EMA filter OFF ({old_text} → OFF).\n"
                "Filter arah dinonaktifkan: BLUE dapat membuka LONG dan RED dapat membuka SHORT."
            )
        else:
            message = (
                f"✅ EMA filter diubah: {old_text} → EMA{selected}.\n"
                f"BLUE/LONG hanya jika close candle > EMA{selected}; "
                f"RED/SHORT hanya jika close candle < EMA{selected}.\n"
                "Close posisi berlawanan tetap dilakukan walaupun entry arah baru ditolak."
            )
        await self.reply(message + "\nPengaturan disimpan dan berlaku untuk sinyal baru.")
        log.info("[CONFIG] EMA directional filter: %s -> %s", old_text, new_text)

    async def _handle_timeframe(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            await self.reply(f"Timeframe aktif: {self.timeframe}\nPilihan: /timeframe H4 atau /timeframe D1")
            return
        tf = parts[1].upper()
        if tf not in volumatic.TIMEFRAME_MS:
            await self.reply("❌ Timeframe hanya H4 atau D1.")
            return
        if tf == self.timeframe:
            await self.reply(f"Timeframe sudah {tf}.")
            return
        old = self.timeframe
        self.timeframe = tf
        self._last_scan_boundary_ms = (int(time.time() * 1000) // volumatic.TIMEFRAME_MS[tf]) * volumatic.TIMEFRAME_MS[tf]
        self._scan_trigger_now = False  # wait for the next completed candle of the selected timeframe
        await self._save_runtime_settings()
        await self.reply(f"✅ Timeframe diubah: {old} → {tf}.\nPending/posisi yang sudah ada tetap dikelola menggunakan timeframe asalnya. Sinyal baru menunggu close {tf} berikutnya.")

    async def _handle_scan(self, text: str) -> None:
        parts = text.split()
        action = parts[1].lower() if len(parts) > 1 else "status"

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
                    f"SCANNER  : {'ON' if self.scan_enabled else ('WAITING' if self.scan_schedule_waiting else 'OFF')}",
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
                f"TIMEFRAME : {self.timeframe}\n"
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
                f"🟢 Scanner ON • Volumatic {self.timeframe} • batch {SCAN_MAX_PAIRS_PER_CYCLE} "
                "• Mode B visual bubble / live candle."
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
            await self.reply(f"Max active slots: {self.max_active_trades}\nAktif sekarang: {sum(t.status in {'PENDING','FILLED'} for t in self.active_trades.values())}")
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
                "yang memenuhi LOT_SIZE dan MIN_NOTIONAL Binance pada harga limit, "
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
        weight = API_STATS.weight_of(calls)
        return (
            "BINANCE API (60s)\n"
            f"Requests: {sum(calls.values())}\nEstimated weight: {weight}/{BINANCE_IP_WEIGHT_LIMIT}\n"
            f"Header weight: {API_STATS.used_weight if API_STATS.used_weight is not None else '-'}\n"
            f"Lifetime requests: {API_STATS.total}\nRate-limit hits: {API_STATS.limit_hits}\n"
            f"Cooldown remaining: {BINANCE_COOLDOWN.remaining():.0f}s\n"
            f"Cooldown reason: {BINANCE_COOLDOWN.reason() or '-'}\n"
        )

    async def show_trades(self) -> None:
        active = [t for t in self.active_trades.values() if t.status in {"PENDING", "FILLED"}]
        if not active:
            await self.reply("Tidak ada pending order atau posisi aktif.")
            return
        current_ema = f"EMA{self.ema_filter_length}" if self.ema_filter_length is not None else "OFF"
        lines = [
            f"ACTIVE TRADES • {len(active)}/{self.max_active_trades} slots\n"
            f"Scanner TF: {self.timeframe} • Length: {self.trend_length} • EMA filter: {current_ema}\n"
        ]
        for t in sorted(active, key=lambda x: (x.status, x.pair)):
            price = self._current_price(t.pair)
            if t.status == "PENDING":
                lines.append(
                    f"⏳ {t.pair} • PENDING {t.direction} ({t.timeframe} • LEN {t.trend_length})\n"
                    f"Signal: {t.signal} | Candle close: {fmt_price(t.signal_close_price or t.entry)} | Limit: {fmt_price(t.entry)}\n"
                    f"EMA at signal: {('OFF' if t.ema_filter_length is None else f'EMA{t.ema_filter_length} @ {fmt_price(t.ema_filter_value)}')}\n"
                    f"Qty: {format_qty(t.quantity or Decimal('0'))} | Margin: {t.margin_usdt} USDT | Lev: {t.leverage}x\n"
                    f"Sizing: {'MINIMUM AUTO' if t.min_margin_mode else 'FIXED TARGET'}\n"
                    + (
                        "EXCHANGE ACK UNKNOWN: verifikasi clientOrderId; POST tidak akan dikirim ulang\n"
                        if t.real_enabled and t.real_submit_ambiguous
                        else (
                            "Validity: GTC dipertahankan sampai fill/reversal/manual cancel\n"
                            if t.real_enabled and t.preserve_gtc_after_cooldown
                            else f"Expires: {format_wib(t.expires_at)}\n"
                        )
                    )
                    + f"Mode: {'REAL' if t.real_enabled else 'PAPER'}"
                    + (f"\nBinance: {t.entry_order_status} • {t.real_state}" if t.real_enabled else "")
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
                lines.append(
                    f"{'🟢' if t.direction == 'LONG' else '🔴'} {t.pair} • {t.direction} FILLED ({t.timeframe} • LEN {t.trend_length})\n"
                    f"Entry: {fmt_price(entry)} | Live: {fmt_price(price)}\n"
                    f"EMA at signal: {('OFF' if t.ema_filter_length is None else f'EMA{t.ema_filter_length} @ {fmt_price(t.ema_filter_value)}')}\n"
                    f"Floating: {fmt_num(pnl_pct, 2)}% | {fmt_num(pnl_usdt, 4)} USDT\n"
                    f"Qty: {format_qty(t.quantity or Decimal('0'))} | Margin est.: {t.margin_usdt} USDT | Lev: {t.leverage}x\n"
                    f"Sizing: {'MINIMUM AUTO' if t.min_margin_mode else 'FIXED TARGET'}\n"
                    + ("Remaining entry LIMIT: GTC dipertahankan sampai fill/reversal/manual cancel\n" if t.real_enabled and t.preserve_gtc_after_cooldown and not t.entry_order_terminal else "")
                    + f"Holding: {duration_text((now_utc() - (t.filled_at or t.created_at)).total_seconds())} | Mode: {'REAL' if t.real_enabled else 'PAPER'}"
                )
        await self.reply("\n\n".join(lines) + "\n\nSL: NONE • TP: NONE • Exit: opposite bubble")

    # -------------------------- Account safety --------------------------
    def _autostop_source_name(self) -> str:
        return "REAL" if (
            self.real_mode or any(
                t.real_enabled and t.status in {"PENDING", "FILLED"}
                for t in self.active_trades.values()
            )
        ) else "PAPER"

    def _autostop_equity(self) -> Decimal | None:
        """Cached equity only: no Binance REST request is made from the guard."""
        real_exposure = self._autostop_source_name() == "REAL"
        floating = Decimal("0")
        if real_exposure:
            balance = self.real.cached_usdt_balance()
            if balance is None:
                return None
            try:
                equity = parse_signed_decimal(str(balance.get("balance") or "0"))
            except ValueError:
                return None
            positions = [
                t for t in self.active_trades.values()
                if t.real_enabled and t.status == "FILLED" and t.quantity and t.quantity > 0
            ]
            for trade in positions:
                snapshot = self.prices.get(trade.pair)
                if snapshot is None or not snapshot.live or trade.fill_price is None:
                    # Missing/stale mark price must not be interpreted as zero PnL.
                    return None
                move = (
                    snapshot.price - trade.fill_price
                    if trade.direction == "LONG"
                    else trade.fill_price - snapshot.price
                )
                floating += move * trade.quantity
            return equity + floating

        # PAPER guard uses the simulation starting balance, realized PAPER PnL,
        # and locally marked PAPER positions. REAL records are never mixed in.
        realized = sum((
            Decimal(str(r.get("pnl_usdt") or "0"))
            for r in self.history_records
            if r.get("strategy_name") == volumatic.STRATEGY_NAME
            and not bool(r.get("real_enabled"))
            and r.get("result") == "CLOSED"
        ), Decimal("0"))
        for trade in self.active_trades.values():
            if trade.real_enabled or trade.status != "FILLED" or not trade.quantity or trade.quantity <= 0:
                continue
            snapshot = self.prices.get(trade.pair)
            if snapshot is None or not snapshot.live or trade.fill_price is None:
                return None
            move = (
                snapshot.price - trade.fill_price
                if trade.direction == "LONG"
                else trade.fill_price - snapshot.price
            )
            floating += move * trade.quantity
        return SIM_START_BALANCE + realized + floating

    def _autostop_floor(self) -> Decimal | None:
        if self.autostop_percent is None or self._equity_peak is None:
            return None
        return self._equity_peak * (Decimal("100") - self.autostop_percent) / Decimal("100")

    def _autostop_status_text(self) -> str:
        source = self._autostop_source or self._autostop_source_name()
        floor = self._autostop_floor()
        state = (
            "TRIGGERED" if self._autostop_triggered else
            ("ARMED · DATA HOLD" if self.autostop_percent is not None and self._autostop_data_hold else
             ("ARMED" if self.autostop_percent is not None else "OFF"))
        )
        lines = [
            "╔══ ACCOUNT GUARD / AUTOSTOP ══╗",
            f"STATE   : {state}",
            f"SOURCE  : {source}",
            f"LIMIT   : {decimal_to_str(self.autostop_percent) + '%' if self.autostop_percent is not None else 'OFF'}",
            f"PEAK    : {fmt_num(self._equity_peak, 4) if self._equity_peak is not None else '-'} USDT",
            f"EQUITY  : {fmt_num(self._equity_last, 4) if self._equity_last is not None else '-'} USDT",
            f"FLOOR   : {fmt_num(floor, 4) if floor is not None else '-'} USDT",
            f"SCANNER : {'LOCKED OFF' if self._autostop_triggered else ('HOLD · EQUITY DATA' if self._autostop_data_hold and self.autostop_percent is not None else ('ON' if self.scan_enabled else 'OFF'))}",
            "ACTION  : pause new scans; keep existing orders/positions managed",
            "",
            "Commands: /autostop 10 | /autostop off | /autostop reset",
        ]
        return "\n".join(lines)

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
            if now_mono - self._autostop_missing_price_notice_at >= 300.0:
                self._autostop_missing_price_notice_at = now_mono
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
            await self.reply(
                "╔══ ACCOUNT GUARD / DATA RESTORED ══╗\n"
                "STATE   : ARMED · EQUITY FRESH\n"
                "ACTION  : pemantauan drawdown aktif kembali\n"
                f"SCANNER : {'RESUME QUEUED' if self._scan_trigger_now else 'UNCHANGED'}"
            )
            log.info("[AUTOSTOP] equity snapshot fresh; entry scanner hold released.")

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
            self._autostop_dirty = False

    async def _handle_autostop_command(self, text: str) -> None:
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

        pending = sum(t.status == "PENDING" for t in active)
        filled = sum(t.status == "FILLED" for t in active)
        expired = sum(
            r.get("result") == "EXPIRED"
            for r in self.history_records
            if r.get("strategy_name") == volumatic.STRATEGY_NAME
        )
        lines = [
            "╔════ VOLUMATIC / TELEMETRY ════╗",
            f"SYS  {'ONLINE' if self._running else 'OFFLINE'}   TF  {self.timeframe}   LEN  {self.trend_length}   EMA  {('EMA' + str(self.ema_filter_length)) if self.ema_filter_length is not None else 'OFF'}   MODE  {'REAL' if self.real_mode else 'PAPER'}",
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
            f"ACTIVE       {len(active):>3}/{self.max_active_trades:<3}  PENDING  {pending:<3} FILLED {filled:<3}",
            f"EXPIRED      {expired}",
        ]
        for tf in ("H4", "D1"):
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
            f"SCANNER      {'LOCKED' if self._autostop_triggered else ('ON' if self.scan_enabled else ('WAITING' if self.scan_schedule_waiting else 'OFF'))}",
            f"SCAN SCHEDULE {', '.join(self.scan_schedule_slots) + ' WIB' if self.scan_schedule_slots else ('ACTIVATED ' + self.scan_schedule_activated_at if self.scan_schedule_activated_at else 'OFF')}",
            "────────────────────────────────",
            "PnL belum termasuk seluruh fee/funding aktual. R = reference risk 0.30 ATR; bukan SL aktif.",
        ]
        await self.reply("\n".join(lines))

    async def show_status(self) -> None:
        pending = sum(t.status == "PENDING" for t in self.active_trades.values())
        filled = sum(t.status == "FILLED" for t in self.active_trades.values())
        fresh = sum(x.live for x in self.prices.values())
        await self.reply(
            "VOLUMATIC STATUS\n\n"
            f"Main: {'ONLINE' if self._running else 'OFFLINE'}\n"
            f"Strategy: {volumatic.STRATEGY_NAME} {volumatic.STRATEGY_VERSION}\n"
            f"Timeframe: {self.timeframe}\nTrend length: {self.trend_length}\nEMA filter: {('EMA' + str(self.ema_filter_length)) if self.ema_filter_length is not None else 'OFF'}\nScanner: {'ON' if self.scan_enabled else ('WAITING' if self.scan_schedule_waiting else 'OFF')}\n"
            f"Scan schedule: {', '.join(self.scan_schedule_slots) + ' WIB' if self.scan_schedule_slots else ('ACTIVATED ' + self.scan_schedule_activated_at if self.scan_schedule_activated_at else 'OFF')}\n"
            f"Universe: {len(self.symbols)} | Market WS: {self.ws.status} | User WS: {self.user_ws.status} | Fresh prices: {fresh}\n"
            f"Pending: {pending} | Filled: {filled} | Max: {self.max_active_trades}\n"
            f"Binance REST cooldown: {self._binance_cooldown_remaining():.0f}s | Deferred commands: {len(self.deferred_commands)}\n"
            f"REAL entry: {'ON' if self.real_mode else 'OFF'} | Margin mode: {'MINIMUM AUTO' if self.min_margin_enabled else 'FIXED'} | Fixed margin: {self.margin_usdt} USDT | Leverage: {self.leverage}x\n"
            f"AUTOSTOP: {'TRIGGERED' if self._autostop_triggered else (decimal_to_str(self.autostop_percent) + '%' if self.autostop_percent is not None else 'OFF')} | Peak: {fmt_num(self._equity_peak, 4) if self._equity_peak is not None else '-'} USDT\n"
            f"History: {len(self.history_records)} | Events: {len(self.history_events)} | Session: {self.session_id}"
        )

    async def show_help(self) -> None:
        await self.reply(
            "COMMAND VOLUMATIC TREND\n\n"
            "/scan on|off - aktif/nonaktifkan scanner Volumatic Mode B\n"
            "/scan schedule - lihat jadwal WIB; pilih dengan /scan schedule 1,3,6\n"
            "/timeframe [H4|D1] - pilih timeframe (default H4)\n"
            f"/length [1–{volumatic.MAX_TREND_LENGTH}] - atur Trend Length Volumatic (default 100)\n"
            "/EMA [50|100|200|off] - filter arah LONG/SHORT berdasarkan EMA timeframe aktif\n"
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
            "Aturan: BLUE=LONG, RED=SHORT; Volumatic Length mengatur EMA-SWMA dan EMA close; LIMIT di CLOSE candle visual bubble untuk satu candle berikutnya; bubble berlawanan menutup posisi via MARKET lalu membuat LIMIT arah baru; tanpa SL/TP/trailing."
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
# Module contract for try.py
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


if __name__ == "__main__":
    raise SystemExit("main.py dijalankan melalui try.py dengan on_start(), handle_update(), dan on_stop().")
