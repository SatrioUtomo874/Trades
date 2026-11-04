#!/usr/bin/env python3
from __future__ import annotations

"""Volumatic Trend [ChartPrime] H4/D1 execution engine.

Contract for try.py:
    async def on_start(context)
    async def handle_update(update, context)
    async def on_stop(context)

The strategy engine is intentionally limited to Volumatic Trend bubble flips.
The main process owns Binance REST/private execution, WebSocket prices,
closed-candle scanner, Telegram, PAPER/REAL ledger, logging and GitHub history.
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
MAIN_FILE = (os.getenv("MAIN_FILE") or "main.py").strip()
BINANCE_API_KEY = (os.getenv("BINANCE_API_KEY") or "").strip()
BINANCE_API_SECRET = (os.getenv("BINANCE_API_SECRET") or "").strip()

TIMEZONE_NAME = "Asia/Jakarta"
TZ = ZoneInfo(TIMEZONE_NAME)
BINANCE_REST_BASE = "https://fapi.binance.com"
BINANCE_REAL_REST_BASE = "https://fapi.binance.com"
BINANCE_WS_BASE = "wss://fstream.binance.com/market/stream"
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

HTTP_REQUEST_SECONDS = 20
GITHUB_REQUEST_SECONDS = 30
BINANCE_RECV_WINDOW = 5000
BINANCE_PUBLIC_REQUEST_SPACING_SECONDS = float(os.getenv("BINANCE_PUBLIC_REQUEST_SPACING_SECONDS", "0.18") or "0.18")
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
MEMORY_SOFT_LIMIT_MB = float(os.getenv("MEMORY_SOFT_LIMIT_MB", "360") or "360")

DEFAULT_MARGIN_USDT = Decimal(os.getenv("DEFAULT_MARGIN_USDT", "0.5") or "0.5")
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "10") or "10")
SIM_START_BALANCE = Decimal(os.getenv("SIM_START_BALANCE", "100") or "100")
REAL_NOTIONAL_MIN_RATIO = Decimal("0.90")
REAL_NOTIONAL_MAX_RATIO = Decimal("1.10")
REAL_FEE_BUFFER = Decimal("0.001")
QTY_FILTER_CODES = (-1013, -1111, -4004, -4005, -4164)
MAX_ACTIVE_TRADES = max(1, int(os.getenv("MAX_ACTIVE_TRADES", "100") or "100"))
SCAN_MAX_PAIRS_PER_CYCLE = 50
SCAN_PAIR_DELAY_SECONDS = float(os.getenv("SCAN_PAIR_DELAY_SECONDS", "0.02") or "0.02")
SCAN_POLL_SECONDS = 2.0
MAINTENANCE_POLL_SECONDS = 2.0
KLINE_HISTORY = max(500, min(1500, int(os.getenv("VOLUMATIC_KLINE_HISTORY", "1200") or "1200")))
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
    signal: str                       # BLUE / RED
    signal_time: datetime
    bubble_price: Decimal
    signal_atr: Decimal | None
    reference_sl: Decimal | None      # reference R only; never a live stop
    entry: Decimal                    # LIMIT price
    expires_at: datetime              # close of the next candle
    signal_close_price: Decimal | None = None  # exact close used as LIMIT price
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
    last_real_check_monotonic: float = 0.0

    @property
    def entry_side(self) -> str:
        return "BUY" if self.direction == "LONG" else "SELL"

    def to_record(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id, "session_id": self.session_id,
            "pair": self.pair, "direction": self.direction,
            "timeframe": self.timeframe, "signal": self.signal,
            "signal_time": iso_utc(self.signal_time),
            "bubble_price": decimal_to_str(self.bubble_price),
            "signal_atr": decimal_to_str(self.signal_atr),
            "reference_sl": decimal_to_str(self.reference_sl),
            "entry": decimal_to_str(self.entry), "signal_close_price": decimal_to_str(self.signal_close_price),
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
            "strategy_name": volumatic.STRATEGY_NAME, "strategy_version": volumatic.STRATEGY_VERSION,
            "entry_reason": "Volumatic Trend confirmed signal candle CLOSE; one-candle LIMIT",
            "price_exp_reason": "Order expires at close of the next candle",
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
        return cls(
            trade_id=str(row.get("trade_id") or generate_trade_id(str(row.get("pair") or "UNKNOWN"))),
            session_id=str(row.get("session_id") or "RECOVERED"),
            pair=normalize_symbol(str(row.get("pair") or "")),
            direction=str(row.get("direction") or "LONG").upper(),
            timeframe=str(row.get("timeframe") or "H4").upper(),
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
            signal_close_price=dec("signal_close_price", dec("entry", Decimal("0"))),
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
    "/fapi/v1/klines": 5,  # konservatif untuk limit candle 500-1000
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
        """Serialized public REST request with central cooldown and telemetry."""
        params = params or {}
        async with self._request_lock:
            wait = self._rest_cooldown_until - time.monotonic()
            if wait > 0:
                log.warning("[BINANCE REST] cooldown aktif %.1fs; request %s ditahan", wait, path)
                await asyncio.sleep(wait)
            spacing = self._last_request_monotonic + BINANCE_PUBLIC_REQUEST_SPACING_SECONDS - time.monotonic()
            if spacing > 0:
                await asyncio.sleep(spacing)

            def request() -> tuple[int, Any, dict[str, str], str]:
                response = self._session.get(
                    f"{self.base_url}{path}", params=params,
                    timeout=HTTP_REQUEST_SECONDS,
                )
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                return response.status_code, payload, dict(response.headers), response.text[:500]

            status, payload, headers, body_text = await asyncio.to_thread(request)
            self._last_request_monotonic = time.monotonic()
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
                raise RuntimeError(f"Binance REST {path}: HTTP {status}: {message}")
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
    def _pack_kline_rows(rows: list[list[Any]], span_ms: int, now_ms: int) -> np.ndarray:
        """Convert Binance JSON kline rows immediately into a compact numeric array."""
        packed: list[tuple[int, float, float, float, float, float, float]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 10:
                continue
            try:
                open_ms = int(row[0])
                if open_ms + span_ms > now_ms:
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
        symbols = await self.get_exchange_info(force=True)
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

        remaining = (
            (self.early_cooldown_remaining if REAL_EARLY.get() else self.server_cooldown_remaining)
            if REAL_PRIORITY.get()
            else self.cooldown_remaining
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
            try:
                response = await asyncio.to_thread(request_once)
            except requests.RequestException as exc:
                raise BinanceAPIError(
                    f"Transport error Binance: {exc}",
                    endpoint=path,
                ) from exc
            except (OSError, TimeoutError) as exc:
                raise BinanceAPIError(
                    f"Transport error Binance: {exc}",
                    endpoint=path,
                ) from exc

        API_STATS.record(
            path,
            response.status_code,
            response.headers.get("X-MBX-USED-WEIGHT-1M"),
        )
        if response.status_code in {418, 429}:
            server_seconds, retry_known = rate_limit_wait(response, self._time_offset_ms)
            bot_seconds = server_seconds + BINANCE_RATE_LIMIT_SAFETY_SECONDS
            self.apply_cooldown(server_seconds, bot_seconds)
            code, msg = self._extract_binance_error(response)
            raise BinanceRateLimitError(
                msg,
                status_code=response.status_code,
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

    async def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        payload = await self._request(
            "GET",
            "/fapi/v1/openOrders",
            {"symbol": symbol},
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

        self.rest = BinanceREST()
        self.real = BinanceRealClient()
        self.github = GitHubStore()
        self.ws = BinanceWebSocket(self._on_price)

        self.real_mode = False  # New REAL entries always require explicit /real on.
        self.margin_usdt = DEFAULT_MARGIN_USDT
        self.leverage = DEFAULT_LEVERAGE
        self.min_margin_enabled = False
        self.max_active_trades = MAX_ACTIVE_TRADES
        self.timeframe = "H4"
        self.scan_enabled = False

        self.symbols: dict[str, SymbolMeta] = {}
        self.prices: dict[str, PriceSnapshot] = {}
        self.active_trades: dict[str, Trade] = {}
        self.legacy_active_snapshot_count = 0
        self.history_records: list[dict[str, Any]] = []
        self.history_events: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []
        self.last_processed_candles: dict[str, int] = {}
        self.last_position_candles: dict[str, int] = {}
        self._signal_retries: dict[str, dict[str, Any]] = {}

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
        try:
            data, _ = await self.github.get_file(remote_path)
            if data is not None:
                return data
        except Exception as exc:
            log.warning("GitHub GET %s gagal; mencoba fallback lokal: %s", remote_path, exc)
        local = self._local_path(remote_path)
        try:
            return local.read_bytes() if local.exists() else None
        except OSError:
            return None

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
            self.scan_enabled = bool(payload.get("scan_enabled", False))
            self.max_active_trades = max(1, min(1000, int(payload.get("max_active_trades") or MAX_ACTIVE_TRADES)))
            self.margin_usdt = parse_decimal(payload.get("margin_usdt") or DEFAULT_MARGIN_USDT)
            self.leverage = max(1, min(125, int(payload.get("leverage") or DEFAULT_LEVERAGE)))
            self.min_margin_enabled = bool(payload.get("min_margin_enabled", False))
            processed = payload.get("last_processed_candles") or {}
            if isinstance(processed, dict):
                self.last_processed_candles = {str(k): int(v) for k, v in processed.items() if str(v).isdigit()}
            position_processed = payload.get("last_position_candles") or {}
            if isinstance(position_processed, dict):
                self.last_position_candles = {str(k): int(v) for k, v in position_processed.items() if str(v).isdigit()}
        except Exception as exc:
            log.warning("Runtime settings tidak valid; gunakan default: %s", exc)

    async def _save_runtime_settings(self) -> None:
        async with self._settings_lock:
            payload = {
                "strategy": volumatic.STRATEGY_NAME,
                "timeframe": self.timeframe,
                "scan_enabled": self.scan_enabled,
                "max_active_trades": self.max_active_trades,
                "margin_usdt": decimal_to_str(self.margin_usdt),
                "leverage": self.leverage,
                "min_margin_enabled": self.min_margin_enabled,
                "last_processed_candles": dict(list(self.last_processed_candles.items())[-3000:]),
                "last_position_candles": dict(list(self.last_position_candles.items())[-3000:]),
                "updated_at": iso_utc(),
            }
            await self._persist_bytes(RUNTIME_SETTINGS_PATH, json.dumps(payload, indent=2).encode(), "volumatic: runtime settings")

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
        await self._persist_bytes(NOTES_PATH, json.dumps(self.notes, ensure_ascii=False, indent=2).encode(), "volumatic: add note")
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
            await self._persist_bytes(HISTORY_TRADES_PATH, trades, "volumatic: update trade history")
            await self._persist_bytes(HISTORY_EVENTS_PATH, events, "volumatic: append event log")

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
        await self._reconcile_restored_trades()
        await self._sync_ws_symbols()
        await self.reply(f"✅ Snapshot dimuat. Trade aktif: {len(self.active_trades)}.")

    # ----------------------------- Startup -------------------------------
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if not self._telegram_error_handler_added:
            for handler in list(log.handlers):
                if isinstance(handler, TelegramErrorHandler) and handler is not self._telegram_error_handler:
                    log.removeHandler(handler)
            self._telegram_error_handler.attach()
            log.addHandler(self._telegram_error_handler)
            self._telegram_error_handler_added = True

        await self._load_runtime_settings()
        await self._load_history()
        await self._load_notes()
        await self._load_active_state()
        self.symbols = await self.rest.get_exchange_info()
        self._last_universe_refresh = time.monotonic()
        await self._reconcile_restored_trades()
        await self.ws.start()
        await self._sync_ws_symbols()

        self._maintenance_task = asyncio.create_task(self._maintenance_loop(), name="volumatic-maintenance")
        self._scan_task = asyncio.create_task(self._scan_loop(), name="volumatic-scanner")
        log.info(
            "[START] Volumatic Trend aktif | timeframe=%s | symbols=%s | restored=%s | "
            "min_margin=%s | fixed_margin=%s | leverage=%sx",
            self.timeframe, len(self.symbols), len(self.active_trades),
            "ON" if self.min_margin_enabled else "OFF", self.margin_usdt, self.leverage,
        )
        migration_warning = (
            f"\n⚠️ {self.legacy_active_snapshot_count} snapshot trade strategi lama tidak dimuat. "
            "Periksa posisi/pending/algo order lama di Binance secara manual sebelum REAL dipakai.\n"
            if self.legacy_active_snapshot_count else ""
        )
        await self.reply(
            "🟢 VOLUMATIC TREND ONLINE\n\n"
            f"Timeframe: {self.timeframe} (default H4)\n"
            f"Universe: {len(self.symbols)} USDT Perpetual\n"
            f"Batch scanner: {SCAN_MAX_PAIRS_PER_CYCLE} pair\n"
            f"Active restored: {len(self.active_trades)}\n"
            "REAL: OFF untuk entry baru\n"
            "Gunakan /scan on untuk memulai scanner dan /timeframe H4 atau D1 untuk memilih timeframe."
            + migration_warning
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for task in (self._scan_task, self._maintenance_task):
            if task is not None and not task.done():
                task.cancel()
                try: await task
                except asyncio.CancelledError: pass
        self._scan_task = self._maintenance_task = None
        try:
            await self._save_active_state()
            await self._save_runtime_settings()
        except Exception:
            log.exception("Persistence saat shutdown gagal.")
        await self.ws.stop()
        try: self.rest._session.close()
        except Exception: pass
        self.active_trades.clear()
        self.prices.clear()
        if self._telegram_error_handler_added:
            try: log.removeHandler(self._telegram_error_handler)
            except Exception: pass
            self._telegram_error_handler_added = False
        self.session_id = ""
        log.info("[STOP] Main session berhenti dengan state aktif sudah disnapshot.")

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

    async def _scan_loop(self) -> None:
        while self._running:
            try:
                if not self.scan_enabled:
                    await asyncio.sleep(SCAN_POLL_SECONDS)
                    continue
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
            except Exception as exc:
                log.exception("[SCAN] loop gagal: %s", exc)
                await asyncio.sleep(5.0)

    async def _run_scan_cycle(self, boundary_ms: int) -> None:
        if self._scan_cycle_lock.locked():
            log.warning("[SCAN] cycle sebelumnya masih aktif; trigger baru dilewati.")
            return
        async with self._scan_cycle_lock:
            timeframe = self.timeframe
            interval_ms = volumatic.TIMEFRAME_MS[timeframe]
            universe = await self._universe()
            started = time.monotonic()
            processed = signals = errors = 0
            log.info("[SCAN] Volumatic %s cycle mulai | universe=%s | batch=%s", timeframe, len(universe), SCAN_MAX_PAIRS_PER_CYCLE)
            await self._expire_due_pending()
            for offset in range(0, len(universe), SCAN_MAX_PAIRS_PER_CYCLE):
                if not self._running or not self.scan_enabled:
                    break
                batch = universe[offset:offset + SCAN_MAX_PAIRS_PER_CYCLE]
                batch_number = offset // SCAN_MAX_PAIRS_PER_CYCLE + 1
                log.info("[SCAN] batch %s | %s-%s dari %s", batch_number, offset + 1, offset + len(batch), len(universe))
                for symbol in batch:
                    if self.timeframe != timeframe:
                        log.info("[SCAN] timeframe berubah saat cycle; hentikan batch lama.")
                        break
                    try:
                        api_interval = "4h" if timeframe == "H4" else "1d"
                        candles = await self.rest.get_klines(
                            symbol, api_interval, KLINE_HISTORY, interval_ms, force_refresh=True
                        )
                        if candles is None or len(candles) < max(volumatic.ATR_LENGTH + 10, volumatic.TREND_LENGTH + 10):
                            log.debug("[SCAN] %s dilewati: history warm-up belum cukup.", symbol)
                            continue
                        latest_open_ms = int(candles["t"][-1])
                        close_ms = latest_open_ms + interval_ms
                        key = f"{timeframe}:{symbol}"
                        previous_close = self.last_processed_candles.get(key)
                        if previous_close is not None and close_ms <= previous_close:
                            processed += 1
                            continue
                        # Only act during the immediately following candle. Signals
                        # from an older candle are historical and must not be chased.
                        now_ms = int(time.time() * 1000)
                        expiry_ms = close_ms + interval_ms
                        if now_ms >= expiry_ms:
                            self.last_processed_candles[key] = close_ms
                            processed += 1
                            continue
                        signal = volumatic.detect_latest_signal(
                            candles, symbol=symbol, timeframe=timeframe, interval_ms=interval_ms
                        )
                        if signal:
                            signals += 1
                            try:
                                await self._handle_signal(signal, expiry_ms)
                                self.last_processed_candles[key] = close_ms
                                self._signal_retries.pop(key, None)
                            except Exception as exc:
                                self._signal_retries[key] = {"signal": signal, "expiry_ms": expiry_ms, "key": key}
                                errors += 1
                                log.exception("[SIGNAL] %s %s gagal diproses; masuk retry: %s", symbol, signal.get("signal"), exc)
                        else:
                            self.last_processed_candles[key] = close_ms
                        processed += 1
                    except Exception as exc:
                        errors += 1
                        log.exception("[SCAN] %s gagal: %s", symbol, exc)
                    if SCAN_PAIR_DELAY_SECONDS > 0:
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                release_memory()
                if offset + len(batch) < len(universe):
                    log.info("[SCAN] batch %s selesai; lanjut batch berikutnya.", batch_number)
            await self._save_runtime_settings()
            await self._save_active_state()
            await self._sync_ws_symbols()
            elapsed = time.monotonic() - started
            log.info("[SCAN] cycle selesai | TF=%s | processed=%s/%s | signals=%s | errors=%s | durasi=%.1fs", timeframe, processed, len(universe), signals, errors, elapsed)

    async def _handle_signal(self, signal: dict[str, Any], expiry_ms: int) -> None:
        pair = normalize_symbol(signal["symbol"])
        direction = str(signal["direction"]).upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError(f"Arah sinyal tidak valid: {direction}")
        if int(time.time() * 1000) >= int(expiry_ms):
            await self._record_event("SIGNAL_EXPIRED_BEFORE_SUBMIT", None, pair=pair, direction=direction, timeframe=signal.get("timeframe"))
            return
        async with self._trade_lock:
            existing = next((t for t in self.active_trades.values() if t.pair == pair and t.status in {"PENDING", "FILLED"}), None)
            # A trade keeps the timeframe that created it. A newly selected
            # timeframe must not reverse or otherwise hijack that existing
            # position; only its own timeframe can manage its lifecycle.
            if existing and existing.timeframe != str(signal.get("timeframe") or self.timeframe).upper():
                log.info(
                    "[SIGNAL] %s %s %s diabaikan: trade aktif memakai timeframe %s.",
                    pair, direction, signal.get("timeframe"), existing.timeframe,
                )
                await self._record_event(
                    "SIGNAL_IGNORED_DIFFERENT_TIMEFRAME_ACTIVE", existing,
                    signal_direction=direction, signal_timeframe=signal.get("timeframe"),
                )
                return
            if existing and existing.direction == direction:
                log.info("[SIGNAL] %s %s diabaikan: trade arah sama masih aktif.", pair, direction)
                await self._record_event("DUPLICATE_SIGNAL_IGNORED", existing, signal=signal.get("signal"))
                return
            # Opposite confirmed bubble closes a filled position immediately.
            # A pending order is canceled before a new direction is submitted.
            if existing:
                if existing.status == "PENDING":
                    await self._cancel_pending_trade(existing, "OPPOSITE_BUBBLE_CANCEL")
                    if existing.trade_id in self.active_trades:
                        raise RuntimeError(
                            f"Order lama {pair} belum terkonfirmasi canceled/terminal; reversal ditahan."
                        )
                elif existing.status == "FILLED":
                    signal_close_price = Decimal(str(signal.get("close_price") or signal.get("signal_close_price") or "0"))
                    # Strategy exit is a MARKET close at the time the opposite
                    # bubble is confirmed. PAPER uses the freshest available
                    # market price; REAL sends a Binance MARKET reduce/close order
                    # and uses this price only as a fallback if Binance omits avgPrice.
                    market_reference_price = self._current_price(pair)
                    if market_reference_price is None:
                        try:
                            market_reference_price = await self.rest.get_price(pair)
                        except Exception:
                            market_reference_price = None
                    if market_reference_price is None:
                        market_reference_price = signal_close_price
                    if market_reference_price <= 0:
                        raise RuntimeError(f"Harga market untuk close {pair} tidak valid.")
                    await self._close_trade(existing, market_reference_price, "OPPOSITE_BUBBLE_MARKET")
            if int(time.time() * 1000) >= int(expiry_ms):
                await self._record_event(
                    "SIGNAL_EXPIRED_DURING_REVERSAL", None, pair=pair,
                    direction=direction, timeframe=signal.get("timeframe"),
                )
                return
            active_count = sum(t.status in {"PENDING", "FILLED"} for t in self.active_trades.values())
            if active_count >= self.max_active_trades:
                log.warning("[SIGNAL] %s %s dilewati: kapasitas /max penuh (%s/%s).", pair, direction, active_count, self.max_active_trades)
                await self._record_event("SIGNAL_SKIPPED_MAX_CAPACITY", None, pair=pair, direction=direction, timeframe=signal.get("timeframe"), active_count=active_count)
                return

            bubble_price = Decimal(str(signal["bubble_price"]))  # EMA1[t-1]: visual/reference only
            signal_close_price = Decimal(str(signal.get("limit_price") or signal.get("signal_close_price") or signal.get("close_price") or "0"))
            if signal_close_price <= 0:
                raise ValueError("Close candle sinyal tidak tersedia atau tidak valid.")
            meta = self.symbols.get(pair)
            if meta is None:
                raise ValueError(f"Metadata Binance untuk {pair} tidak ditemukan.")
            # The LIMIT is anchored to the CLOSE of the candle that confirmed
            # the bubble, not to the EMA-SWMA line where Pine draws the diamond.
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
                signal=str(signal["signal"]), signal_time=signal_close,
                bubble_price=bubble_price, signal_atr=atr_value, reference_sl=reference_value,
                entry=limit_price, expires_at=expiry_dt, signal_close_price=signal_close_price,
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
                    return
            else:
                observed = self._current_price(pair)
                if observed is not None:
                    touched = observed <= trade.entry if trade.direction == "LONG" else observed >= trade.entry
                    if touched:
                        await self._mark_filled(trade, observed, "PAPER_MARKETABLE_AT_SUBMIT")
            await self.reply(
                f"{'🟢' if direction == 'LONG' else '🔴'} VOLUMATIC {direction} • {pair}\n\n"
                f"Timeframe: {trade.timeframe}\nSignal: {trade.signal} BUBBLE\n"
                f"Bubble line (EMA1): {fmt_price(trade.bubble_price)}\n"
                f"Signal candle close: {fmt_price(trade.signal_close_price or trade.entry)}\n"
                f"LIMIT entry: {fmt_price(trade.entry)}\n"
                f"Margin: {trade.margin_usdt} USDT • Leverage: {trade.leverage}x\n"
                f"Sizing: {'MINIMUM AUTO' if trade.min_margin_mode else 'FIXED TARGET'}\n"
                f"Quantity: {format_qty(trade.quantity or Decimal('0'))}\n"
                f"Expiry: {format_wib(trade.expires_at)}\n"
                f"Mode: {'REAL' if trade.real_enabled else 'PAPER'}\n"
                "SL: NONE • TP: NONE"
            )

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
        """Submit one prepared LIMIT; preserve ambiguous submissions for reconciliation."""
        if not trade.real_enabled or not trade.entry_client_order_id:
            raise RuntimeError("REAL order belum melewati preflight atau clientOrderId tidak tersedia.")
        if now_utc() >= trade.expires_at:
            await self._finalize_trade(trade, "EXPIRED", None, "CANDLE_EXPIRED_BEFORE_REAL_SUBMIT")
            return

        trade.real_state = "REAL_SUBMITTING"
        await self._save_active_state()
        try:
            response = await self.real.place_limit_entry(
                symbol=trade.pair, side=trade.entry_side, quantity=trade.quantity or Decimal("0"),
                price=trade.entry, position_side=trade.position_side or "BOTH",
                client_order_id=trade.entry_client_order_id,
            )
        except Exception as exc:
            # The POST may have reached Binance even when the response timed out.
            # Query by the stable client order ID; never create a second ID here.
            trade.real_state = "REAL_SUBMIT_UNKNOWN"
            trade.real_error = str(exc)[:500]
            log.exception("[REAL] submit %s ambiguous; reconcile by clientOrderId", trade.pair)
            try:
                response = await self.real.get_order(
                    trade.pair, client_order_id=trade.entry_client_order_id
                )
            except Exception as query_exc:
                trade.real_error = (str(exc) + " | reconcile: " + str(query_exc))[:500]
                await self._save_active_state()
                return

        trade.entry_order_id = int(response["orderId"]) if response.get("orderId") is not None else trade.entry_order_id
        status = str(response.get("status") or "UNKNOWN").upper()
        executed = Decimal(str(response.get("executedQty") or "0"))
        if status == "FILLED":
            if executed > 0:
                trade.quantity = executed
            avg = Decimal(str(response.get("avgPrice") or trade.entry))
            trade.fill_price = avg if avg > 0 else trade.entry
            trade.real_state = "REAL_FILLED"
            trade.real_error = None
        elif status in {"NEW", "PARTIALLY_FILLED"}:
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

    async def _cancel_pending_trade(self, trade: Trade, reason: str) -> None:
        """Cancel a pending order; REAL state changes only after exchange confirmation."""
        if trade.status != "PENDING":
            return
        if trade.real_enabled:
            if not (trade.entry_order_id or trade.entry_client_order_id):
                trade.real_state = "REAL_CANCEL_UNKNOWN"
                trade.real_error = "Order REAL belum memiliki ID untuk rekonsiliasi/cancel."
                await self._save_active_state()
                return
            try:
                try:
                    order = await self.real.get_order(
                        trade.pair, order_id=trade.entry_order_id,
                        client_order_id=trade.entry_client_order_id,
                    )
                except Exception as lookup_exc:
                    # A lookup timeout/404 does not prove that the order is gone.
                    order = {}
                    log.warning("[REAL] lookup sebelum cancel %s gagal; tetap mencoba DELETE: %s", trade.pair, lookup_exc)
                status = str(order.get("status") or "UNKNOWN").upper()
                executed = Decimal(str(order.get("executedQty") or "0"))

                if status in {"NEW", "PARTIALLY_FILLED", "UNKNOWN"}:
                    try:
                        await self.real.cancel_order(
                            trade.pair, order_id=trade.entry_order_id,
                            client_order_id=trade.entry_client_order_id,
                        )
                    except Exception as cancel_exc:
                        log.warning(
                            "[REAL] cancel request %s belum terkonfirmasi; cek status lagi: %s",
                            trade.pair, cancel_exc,
                        )
                    order = await self.real.get_order(
                        trade.pair, order_id=trade.entry_order_id,
                        client_order_id=trade.entry_client_order_id,
                    )
                    status = str(order.get("status") or "UNKNOWN").upper()
                    executed = Decimal(str(order.get("executedQty") or "0"))

                if status == "FILLED":
                    trade.quantity = executed if executed > 0 else trade.quantity
                    await self._mark_filled(
                        trade, self._order_fill_price(order, trade.entry),
                        "BINANCE_CANCEL_RACE_FILL",
                    )
                    return

                if status in {"CANCELED", "EXPIRED", "REJECTED"}:
                    if executed > 0:
                        trade.quantity = executed
                        await self._mark_filled(
                            trade, self._order_fill_price(order, trade.entry),
                            "BINANCE_PARTIAL_FILL_REMAINDER_CANCELED",
                        )
                        return
                    result = "EXPIRED" if status == "EXPIRED" or "EXPIRE" in reason else "CANCELED"
                    await self._finalize_trade(trade, result, None, f"{reason}; BINANCE_{status}")
                    return

                trade.real_state = "REAL_CANCEL_UNKNOWN"
                trade.real_error = f"Cancel belum terkonfirmasi; status Binance={status}."
                await self._save_active_state()
                log.error("[REAL] %s tetap aktif karena status cancel belum final: %s", trade.pair, status)
                return
            except Exception as exc:
                trade.real_state = "REAL_CANCEL_UNKNOWN"
                trade.real_error = str(exc)[:500]
                log.exception("[REAL] cancel pending %s gagal; order tetap aktif untuk retry", trade.pair)
                await self._save_active_state()
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
            await self.github.append_text_file(HISTORY_MARKDOWN_PATH, addition, "volumatic: trade closed")
            local = self._local_path(HISTORY_MARKDOWN_PATH)
            old = local.read_text(encoding="utf-8") if local.exists() else "# Volumatic Trend Trade Journal\n\n"
            local.write_text(old + addition, encoding="utf-8")
        except Exception as exc:
            log.error("Markdown journal GitHub gagal; detail tetap ada di trades.json: %s", exc)

    async def _expire_due_pending(self) -> None:
        now = now_utc()
        for trade in list(self.active_trades.values()):
            if trade.status == "PENDING" and now >= trade.expires_at:
                await self._cancel_pending_trade(trade, "ORDER_EXPIRED_ONE_CANDLE")

    async def _reconcile_pending_real(self, trade: Trade) -> None:
        if not trade.real_enabled or trade.status != "PENDING":
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
        except BinanceAPIError as exc:
            # Crash recovery for the narrow window after REAL_SUBMITTING was
            # persisted but before the POST reached Binance. -2013 means the
            # stable clientOrderId is not present. Reuse that same ID (never mint
            # a second one); if a first POST actually exists, Binance duplicate
            # detection plus the subsequent lookup prevents a second exposure.
            if exc.code == -2013 and trade.real_state in {"REAL_SUBMITTING", "REAL_SUBMIT_UNKNOWN"}:
                if now_utc() >= trade.expires_at:
                    await self._finalize_trade(
                        trade, "EXPIRED", None, "REAL_SUBMISSION_NOT_FOUND_AFTER_CANDLE_EXPIRY"
                    )
                else:
                    await self._submit_real_limit(trade)
                    if trade.status == "PENDING" and trade.real_state == "REAL_FILLED":
                        await self._mark_filled(
                            trade, trade.fill_price or trade.entry, "REAL_SUBMISSION_RECOVERED"
                        )
                return
            if now_utc() >= trade.expires_at or trade.real_state == "REAL_CANCEL_UNKNOWN":
                reason = "ORDER_EXPIRED_ONE_CANDLE" if now_utc() >= trade.expires_at else "RETRY_CANCEL_UNKNOWN"
                await self._cancel_pending_trade(trade, reason)
                return
            raise
        except Exception:
            if now_utc() >= trade.expires_at or trade.real_state == "REAL_CANCEL_UNKNOWN":
                reason = "ORDER_EXPIRED_ONE_CANDLE" if now_utc() >= trade.expires_at else "RETRY_CANCEL_UNKNOWN"
                await self._cancel_pending_trade(trade, reason)
                return
            raise
        status = str(order.get("status") or "UNKNOWN").upper()
        if status == "FILLED":
            executed = Decimal(str(order.get("executedQty") or "0"))
            if executed > 0:
                trade.quantity = executed
            await self._mark_filled(trade, self._order_fill_price(order, trade.entry), "BINANCE_POLL")
        elif status == "PARTIALLY_FILLED":
            # A partial fill is a real exposure. Cancel the remaining quantity
            # and adopt only after Binance confirms the order is terminal.
            await self._cancel_pending_trade(trade, "PARTIAL_FILL_REMAINDER_CANCEL")
        elif status in {"NEW", "UNKNOWN"} and (
            now_utc() >= trade.expires_at or trade.real_state == "REAL_CANCEL_UNKNOWN"
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
        elif now_utc() >= trade.expires_at:
            await self._cancel_pending_trade(trade, "ORDER_EXPIRED_ONE_CANDLE")

    async def _reconcile_filled_real(self, trade: Trade) -> None:
        if not trade.real_enabled or trade.status != "FILLED":
            return
        if time.monotonic() - trade.last_real_check_monotonic < REAL_RECONCILE_INTERVAL_SECONDS:
            return
        trade.last_real_check_monotonic = time.monotonic()
        position = await self.real.get_position(trade.pair, trade.entry_side)
        if position is None:
            remaining = await self.real.get_positions(trade.pair)
            if not remaining:
                price = self._current_price(trade.pair) or trade.fill_price or trade.entry
                log.warning("[REAL] %s %s exposure hilang di Binance; menutup ledger sebagai external close.", trade.pair, trade.direction)
                await self._finalize_trade(trade, "CLOSED", price, "EXTERNAL_CLOSE_RECONCILED")

    async def _reconcile_restored_trades(self) -> None:
        for trade in list(self.active_trades.values()):
            if trade.real_enabled:
                try:
                    if trade.status == "PENDING":
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
                            if exc.code == -2013 and trade.real_state in {"REAL_SUBMITTING", "REAL_SUBMIT_UNKNOWN"}:
                                if now_utc() >= trade.expires_at:
                                    await self._finalize_trade(
                                        trade, "EXPIRED", None,
                                        "STARTUP_REAL_SUBMISSION_NOT_FOUND_AFTER_EXPIRY",
                                    )
                                else:
                                    await self._submit_real_limit(trade)
                                    if trade.status == "PENDING" and trade.real_state == "REAL_FILLED":
                                        await self._mark_filled(
                                            trade, trade.fill_price or trade.entry,
                                            "STARTUP_SUBMISSION_RECOVERED",
                                        )
                                continue
                            raise
                        status = str(order.get("status") or "UNKNOWN").upper()
                        if status == "FILLED":
                            await self._mark_filled(trade, self._order_fill_price(order, trade.entry), "STARTUP_RECONCILIATION")
                        elif status in {"CANCELED", "EXPIRED", "REJECTED"}:
                            await self._finalize_trade(trade, "EXPIRED" if status == "EXPIRED" else "CANCELED", None, f"STARTUP_{status}")
                    elif trade.status == "FILLED":
                        position = await self.real.get_position(trade.pair, trade.entry_side)
                        if position is None:
                            remaining = await self.real.get_positions(trade.pair)
                            if not remaining:
                                trade.real_error = "Posisi ledger tidak ditemukan di Binance; cek manual sebelum trading pair ini."
                                log.error("[RECOVERY] %s: %s", trade.pair, trade.real_error)
                except Exception as exc:
                    trade.real_error = str(exc)[:500]
                    log.exception("[RECOVERY] reconcile %s gagal.", trade.pair)
        await self._save_active_state()

    async def _maintenance_loop(self) -> None:
        while self._running:
            try:
                await self._expire_due_pending()
                for trade in list(self.active_trades.values()):
                    if trade.real_enabled and trade.status == "PENDING":
                        try: await self._reconcile_pending_real(trade)
                        except Exception as exc: log.warning("[REAL] pending sync %s gagal: %s", trade.pair, exc)
                    elif trade.real_enabled and trade.status == "FILLED":
                        try: await self._reconcile_filled_real(trade)
                        except Exception as exc: log.warning("[REAL] position sync %s gagal: %s", trade.pair, exc)
                await self._retry_signals()
                await self._monitor_active_positions()
                await asyncio.sleep(MAINTENANCE_POLL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("[MAINTENANCE] loop error")
                await asyncio.sleep(3)

    async def _monitor_active_positions(self) -> None:
        """Monitor every filled position on its own timeframe, including missed bars.

        Position exits must not be missed if the process/API was unavailable for
        several candles. Old opposite bubbles close the position on recovery but
        do not create a stale reverse LIMIT whose one-candle window has passed.
        """
        by_key: dict[str, Trade] = {}
        for trade in self.active_trades.values():
            if trade.status == "FILLED":
                by_key.setdefault(f"{trade.timeframe}:{trade.pair}", trade)
        now_ms = int(time.time() * 1000)
        for key, seed_trade in list(by_key.items()):
            tf = seed_trade.timeframe if seed_trade.timeframe in volumatic.TIMEFRAME_MS else "H4"
            span = volumatic.TIMEFRAME_MS[tf]
            expected_close = (now_ms // span) * span
            checkpoint = max(
                self.last_position_candles.get(key, 0),
                int(seed_trade.signal_time.timestamp() * 1000),
            )
            if expected_close <= checkpoint:
                continue
            try:
                candles = await self.rest.get_klines(
                    seed_trade.pair, "4h" if tf == "H4" else "1d",
                    KLINE_HISTORY, span, force_refresh=True,
                )
                if candles is None or len(candles) < volumatic.ATR_LENGTH + 10:
                    continue
                latest_close_ms = int(candles["t"][-1]) + span
                if latest_close_ms <= checkpoint:
                    continue
                events = volumatic.detect_signals_since(
                    candles, symbol=seed_trade.pair, timeframe=tf,
                    after_close_time_ms=checkpoint, interval_ms=span,
                )
                for signal in events:
                    signal_close_ms = int(signal["signal_close_time_ms"])
                    expiry_ms = signal_close_ms + span
                    existing = next((
                        t for t in self.active_trades.values()
                        if t.pair == seed_trade.pair and t.status in {"PENDING", "FILLED"}
                    ), None)
                    opposite = existing is not None and existing.direction != signal["direction"]
                    stale = now_ms >= expiry_ms

                    if opposite and stale:
                        if existing.status == "FILLED":
                            price = self._current_price(existing.pair)
                            if price is None:
                                price = await self.rest.get_price(existing.pair)
                            await self._close_trade(
                                existing, price, "LATE_OPPOSITE_BUBBLE_NO_REVERSE"
                            )
                            await self._record_event(
                                "LATE_OPPOSITE_BUBBLE_NO_REVERSE", None,
                                pair=existing.pair, direction=signal["direction"],
                                timeframe=tf, signal_close_time_ms=signal_close_ms,
                            )
                        elif existing.status == "PENDING":
                            await self._cancel_pending_trade(
                                existing, "LATE_OPPOSITE_BUBBLE_CANCEL"
                            )
                        continue

                    if not stale:
                        await self._handle_signal(signal, expiry_ms)
                    elif opposite and existing is not None and existing.trade_id in self.active_trades:
                        # REAL cancel/close may still be unresolved. Do not advance
                        # the checkpoint; retry this bubble on the next maintenance pass.
                        raise RuntimeError(
                            f"Opposite bubble {seed_trade.pair} belum selesai direkonsiliasi."
                        )

                self.last_position_candles[key] = latest_close_ms
                await self._save_runtime_settings()
            except Exception as exc:
                log.warning(
                    "[POSITION MONITOR] %s %s gagal dianalisis/dieksekusi: %s",
                    seed_trade.pair, tf, exc, exc_info=True,
                )

    async def _retry_signals(self) -> None:
        if not self._signal_retries:
            return
        for key, item in list(self._signal_retries.items()):
            if int(time.time() * 1000) >= int(item["expiry_ms"]):
                self._signal_retries.pop(key, None)
                self.last_processed_candles[key] = int(item["signal"]["signal_close_time_ms"])
                await self._record_event("SIGNAL_RETRY_EXPIRED", None, pair=item["signal"].get("symbol"), direction=item["signal"].get("direction"), timeframe=item["signal"].get("timeframe"))
                continue
            try:
                await self._handle_signal(item["signal"], int(item["expiry_ms"]))
                self.last_processed_candles[key] = int(item["signal"]["signal_close_time_ms"])
                self._signal_retries.pop(key, None)
                await self._save_runtime_settings()
            except Exception as exc:
                log.warning("[SIGNAL] retry %s belum berhasil: %s", key, exc)

    # ----------------------------- WebSocket ------------------------------
    def _current_price(self, symbol: str) -> Decimal | None:
        snapshot = self.prices.get(symbol)
        return snapshot.price if snapshot and snapshot.live else None

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

    async def _sync_ws_symbols(self) -> None:
        desired = {t.pair for t in self.active_trades.values() if t.status in {"PENDING", "FILLED"}}
        current = set(self.ws.symbols())
        for symbol in sorted(desired - current):
            await self.ws.add_symbol(symbol)
        for symbol in sorted(current - desired):
            await self.ws.remove_symbol(symbol)

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
            await self.real.get_account_info()
            await self.real.get_account_config()
            await self.real.get_futures_balance()
            self.real_mode = True
            await self.reply("🟢 REAL ON. Order BARU akan dikirim ke Binance USD-M Futures LIVE. Posisi REAL yang sudah ada tetap dikelola walaupun /real off.")
        elif arg == "off":
            self.real_mode = False
            await self.reply("🟡 REAL OFF untuk entry baru. Pending/posisi REAL yang sudah ada tetap direkonsiliasi dan dikelola sampai selesai.")
        else:
            await self.reply(f"REAL: {'ON' if self.real_mode else 'OFF'}\nGunakan /real on atau /real off.")

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
        if len(parts) < 2 or parts[1].lower() not in {"on", "off"}:
            await self.reply(f"Scanner: {'ON' if self.scan_enabled else 'OFF'}\nGunakan /scan on atau /scan off.")
            return
        enable = parts[1].lower() == "on"
        self.scan_enabled = enable
        if enable:
            self._scan_trigger_now = True
            await self.reply(f"🟢 Scanner ON • Volumatic {self.timeframe} • batch {SCAN_MAX_PAIRS_PER_CYCLE}.")
        else:
            await self.reply("🟡 Scanner OFF. Posisi dan pending order tetap dimonitor; order tetap expired pada akhir candle.")
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
        archive_path = f"data/archive/trades-{stamp}.json"
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
        await self._persist_bytes(NOTES_PATH, b"[]", "volumatic: reset notes")
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
        )

    async def show_trades(self) -> None:
        active = [t for t in self.active_trades.values() if t.status in {"PENDING", "FILLED"}]
        if not active:
            await self.reply("Tidak ada pending order atau posisi aktif.")
            return
        lines = [f"ACTIVE TRADES • {len(active)}/{self.max_active_trades} slots\nTimeframe scanner: {self.timeframe}\n"]
        for t in sorted(active, key=lambda x: (x.status, x.pair)):
            price = self._current_price(t.pair)
            if t.status == "PENDING":
                lines.append(
                    f"⏳ {t.pair} • PENDING {t.direction} ({t.timeframe})\n"
                    f"Signal: {t.signal} | Candle close: {fmt_price(t.signal_close_price or t.entry)} | Limit: {fmt_price(t.entry)}\n"
                    f"Qty: {format_qty(t.quantity or Decimal('0'))} | Margin: {t.margin_usdt} USDT | Lev: {t.leverage}x\n"
                    f"Sizing: {'MINIMUM AUTO' if t.min_margin_mode else 'FIXED TARGET'}\n"
                    f"Expires: {format_wib(t.expires_at)} | Mode: {'REAL' if t.real_enabled else 'PAPER'}"
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
                    f"{'🟢' if t.direction == 'LONG' else '🔴'} {t.pair} • {t.direction} FILLED ({t.timeframe})\n"
                    f"Entry: {fmt_price(entry)} | Live: {fmt_price(price)}\n"
                    f"Floating: {fmt_num(pnl_pct, 2)}% | {fmt_num(pnl_usdt, 4)} USDT\n"
                    f"Qty: {format_qty(t.quantity or Decimal('0'))} | Margin est.: {t.margin_usdt} USDT | Lev: {t.leverage}x\n"
                    f"Sizing: {'MINIMUM AUTO' if t.min_margin_mode else 'FIXED TARGET'}\n"
                    f"Holding: {duration_text((now_utc() - (t.filled_at or t.created_at)).total_seconds())} | Mode: {'REAL' if t.real_enabled else 'PAPER'}"
                )
        await self.reply("\n\n".join(lines) + "\n\nSL: NONE • TP: NONE • Exit: opposite bubble")

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
        closed = [r for r in self.history_records if r.get("strategy_name") == volumatic.STRATEGY_NAME and r.get("result") == "CLOSED"]
        active = [t for t in self.active_trades.values() if t.status in {"PENDING", "FILLED"}]
        realized = [float(Decimal(str(r.get("pnl_usdt") or "0"))) for r in closed]
        wins = [x for x in realized if x > 0]
        losses = [x for x in realized if x < 0]
        gross_profit, gross_loss = sum(wins), abs(sum(losses))
        win_rate = len(wins) / len(realized) * 100 if realized else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        floating = 0.0
        for t in active:
            if t.status != "FILLED": continue
            price = self._current_price(t.pair)
            if price is None or t.fill_price is None: continue
            move = price - t.fill_price if t.direction == "LONG" else t.fill_price - price
            floating += float(move * (t.quantity or Decimal("0")))
        pnls_by_time = sorted(closed, key=lambda r: str(r.get("closed_at") or ""))
        dd = self._max_drawdown([float(Decimal(str(r.get("pnl_usdt") or "0"))) for r in pnls_by_time])
        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        pending = sum(t.status == "PENDING" for t in active)
        filled = sum(t.status == "FILLED" for t in active)
        expired = sum(r.get("result") == "EXPIRED" for r in self.history_records if r.get("strategy_name") == volumatic.STRATEGY_NAME)
        total_r = sum(float(Decimal(str(r.get("realized_r") or "0"))) for r in closed if r.get("realized_r") is not None)
        total_pnl = sum(realized)
        expectancy = total_pnl / len(realized) if realized else 0.0
        holding_seconds = []
        for record in closed:
            filled_at = parse_iso(record.get("filled_at"))
            closed_at = parse_iso(record.get("closed_at"))
            if filled_at is not None and closed_at is not None and closed_at >= filled_at:
                holding_seconds.append((closed_at - filled_at).total_seconds())
        average_holding = duration_text(sum(holding_seconds) / len(holding_seconds)) if holding_seconds else "-"
        lines = [
            "VOLUMATIC TREND • STATISTICS",
            f"Scanner timeframe: {self.timeframe}",
            f"Total closed: {len(closed)} | Win/Loss/BE: {len(wins)}/{len(losses)}/{len(realized)-len(wins)-len(losses)}",
            f"Win rate: {win_rate:.2f}%",
            f"Average win: {avg_win:.4f} USDT | Average loss: {avg_loss:.4f} USDT",
            f"Profit factor: {pf:.3f}" if np.isfinite(pf) else "Profit factor: ∞",
            f"Realized PnL: {total_pnl:+.4f} USDT | Floating: {floating:+.4f} USDT",
            f"Expectancy: {expectancy:+.4f} USDT/trade | Average holding: {average_holding}",
            f"Max drawdown (realized equity): {dd:.4f} USDT",
            f"Total R: {total_r:+.3f}R",
            f"Active: {len(active)} / {self.max_active_trades} | Pending: {pending} | Filled: {filled} | Expired: {expired}",
        ]
        for tf in ("H4", "D1"):
            for direction in ("LONG", "SHORT"):
                subset = [r for r in closed if r.get("timeframe") == tf and r.get("direction") == direction]
                pnl = sum(float(Decimal(str(r.get("pnl_usdt") or "0"))) for r in subset)
                lines.append(f"{tf} {direction}: {len(subset)} trades | PnL {pnl:+.4f} USDT")
        lines.append("Catatan: PnL belum memasukkan funding/komisi aktual Binance; R memakai reference risk 0.30 ATR, bukan SL aktif.")
        await self.reply("\n".join(lines))

    async def show_status(self) -> None:
        pending = sum(t.status == "PENDING" for t in self.active_trades.values())
        filled = sum(t.status == "FILLED" for t in self.active_trades.values())
        fresh = sum(x.live for x in self.prices.values())
        await self.reply(
            "VOLUMATIC STATUS\n\n"
            f"Main: {'ONLINE' if self._running else 'OFFLINE'}\n"
            f"Strategy: {volumatic.STRATEGY_NAME} {volumatic.STRATEGY_VERSION}\n"
            f"Timeframe: {self.timeframe}\nScanner: {'ON' if self.scan_enabled else 'OFF'}\n"
            f"Universe: {len(self.symbols)} | WS: {self.ws.status} | Fresh prices: {fresh}\n"
            f"Pending: {pending} | Filled: {filled} | Max: {self.max_active_trades}\n"
            f"REAL entry: {'ON' if self.real_mode else 'OFF'} | Margin mode: {'MINIMUM AUTO' if self.min_margin_enabled else 'FIXED'} | Fixed margin: {self.margin_usdt} USDT | Leverage: {self.leverage}x\n"
            f"History: {len(self.history_records)} | Events: {len(self.history_events)} | Session: {self.session_id}"
        )

    async def show_help(self) -> None:
        await self.reply(
            "COMMAND VOLUMATIC TREND\n\n"
            "/scan on|off - aktif/nonaktifkan scanner closed-candle\n"
            "/timeframe [H4|D1] - pilih timeframe (default H4)\n"
            "/max [angka] - batas total pending + filled\n"
            "/trade - lihat pending order dan posisi aktif\n"
            "/stats - statistik trade, PnL, win rate, DD, R\n"
            "/setup - simpan snapshot trade aktif ke GitHub\n"
            "/open - pulihkan snapshot jika belum ada trade aktif\n"
            "/del PAIR|all - batalkan pending order\n"
            "/close PAIR|all - tutup posisi atau batalkan pending\n"
            "/remove TRADE_ID|PAIR - hapus record dari statistik (event audit tetap)\n"
            "/real on|off - aktif/nonaktifkan entry baru REAL\n"
            "/margin [USDT] - margin target entry baru (dipakai saat /min off)\n"
            "/min on|off - gunakan margin minimum valid Binance atau margin tetap\n"
            "/leverage [angka] - leverage entry baru\n"
            "/balance - saldo Futures atau estimasi PAPER\n"
            "/api - statistik bobot/request Binance\n"
            "/catatan [teks] - tambah atau lihat catatan\n"
            "/resetstats - arsipkan lalu reset statistik trade\n"
            "/reset CONFIRM - reset history/event/catatan (tanpa mengubah posisi)\n"
            "/status - status sistem\n/help - bantuan\n\n"
            "Aturan: BLUE=LONG, RED=SHORT; LIMIT di CLOSE candle sinyal untuk satu candle berikutnya; bubble berlawanan menutup posisi via MARKET lalu membuat LIMIT arah baru; tanpa SL/TP/trailing."
        )

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
            if command in {"/try", "/end", "/ganti", "/healthz"}:
                return
            if command == "/timeframe": await self._handle_timeframe(text)
            elif command == "/scan": await self._handle_scan(text)
            elif command == "/max": await self._handle_max(text)
            elif command == "/margin": await self._handle_margin(text)
            elif command == "/min": await self._handle_min(text)
            elif command == "/leverage": await self._handle_leverage(text)
            elif command == "/real": await self._set_real(text)
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
