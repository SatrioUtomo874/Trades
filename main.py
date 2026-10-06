from __future__ import annotations

"""
MAIN.PY
========
Trading simulator engine untuk try.py launcher.

Kontrak dengan try.py:
    async def on_start(context)
    async def handle_update(update, context)
    async def on_stop(context)

Prinsip:
- Binance USDⓈ-M Futures = market data publik + real execution opsional.
- REAL selalu OFF pada session baru dan hanya aktif setelah /real on.
- Price Now pada /add diambil REST satu kali.
- Harga live setelah setup dibuat berasal dari Binance WebSocket.
- Active trade hanya ada di RAM selama sesi main.py.
- /end membersihkan seluruh active state.
- Histori permanen ditulis langsung ke GitHub:
    data/trades.json
    data/events.jsonl
    data/trade_history.md
- /stats membaca histori dan menghitung statistik.
- /analyze membuat:
    analysis/full_data.json
    analysis/analysis.md
- /setup menyimpan snapshot setup aktif ke data/setups.json.
- /open memulihkan snapshot setup tersebut setelah main.py diganti.
- /auto menjembatani main.py ke strategy.py dan meminta konfirmasi user.
- /scan menjalankan scanner otomatis: universe Binance ∩ Bybit -> directional H4 -> strategy -> threshold -> validator -> /trade.
- /H4 mengatur gate scheduler scanner pada close H4: 03:00, 07:00, 11:00, 15:00, 19:00, 23:00 WIB.
- /threshold mengatur ambang confidence scanner.
- /max membatasi total active PENDING + FILLED dan dapat mem-pause/resume /scan otomatis.
- /close PAIR|all menutup paksa active setup dan tetap mencatat hasil ke history.
- /banned dan /unban mengelola ban pair; ban otomatis diterapkan untuk Price Exp/TP/SL/Margin Influence.
- /wrong menghapus active setup yang benar-benar salah tanpa mencatat history.
- /remove menghapus satu trade dari histori GitHub beserta event dan journal terkait.
- strategy.py wajib menyediakan generate_setup(pair, context); mode SCAN juga mendukung kontrak optimized scan structure dan validate_setup.
- SCAN maksimal 50 pair dianalisis per cycle; pair mismatch H4 diban 24 jam dan below-threshold diban 8 jam.
"""

import asyncio
import base64
import contextvars
import hashlib
import hmac
import copy
import html  # kept out of user output; used only for safe GitHub text if needed
import importlib
import inspect
import sys
import json
import logging
import os
import re
import time
from collections import deque
import traceback
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN, ROUND_UP
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from uuid import uuid4
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

from websockets.exceptions import ConnectionClosed

try:
    from websockets.asyncio.client import connect as ws_connect
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Dependency 'websockets' belum terpasang. "
        "Tambahkan 'websockets>=17,<18' ke requirements.txt "
        "lalu redeploy."
    ) from exc
except ImportError:
    # Fallback hanya untuk environment dengan versi lama.
    try:
        from websockets import connect as ws_connect
    except ImportError as exc:
        raise RuntimeError(
            "Library 'websockets' tersedia tetapi API client-nya tidak kompatibel. "
            "Gunakan websockets>=17,<18."
        ) from exc


# ============================================================
# ENV / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / "trades.env")
load_dotenv(BASE_DIR / ".env")

# Acuan nama key mengikuti Trades.env yang diberikan user.
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()
MAIN_FILE = (os.getenv("MAIN_FILE") or "main.py").strip()

# Credential Binance dipakai hanya oleh BinanceRealClient ketika /real ON.
# BinanceREST tetap public-only untuk market data/scanner.
BINANCE_API_KEY = (os.getenv("BINANCE_API_KEY") or "").strip()
BINANCE_API_SECRET = (os.getenv("BINANCE_API_SECRET") or "").strip()
BINANCE_API_KEY_1 = (os.getenv("BINANCE_API_KEY_1") or "").strip()
BINANCE_API_SECRET_1 = (os.getenv("BINANCE_API_SECRET_1") or "").strip()

TIMEZONE_NAME = "Asia/Jakarta"
TZ = ZoneInfo(TIMEZONE_NAME)

BINANCE_REST_BASE = "https://fapi.binance.com"
BINANCE_WS_BASE = "wss://fstream.binance.com/market/ws"

GITHUB_API = "https://api.github.com"

HISTORY_TRADES_PATH = "data/trades.json"
HISTORY_EVENTS_PATH = "data/events.jsonl"
HISTORY_MARKDOWN_PATH = "data/trade_history.md"

ANALYSIS_JSON_PATH = "analysis/full_data.json"
ANALYSIS_MD_PATH = "analysis/analysis.md"
NOTES_PATH = "data/notes.json"
SETUPS_PATH = "data/setups.json"
RESET_PATHS = (
    HISTORY_TRADES_PATH,
    HISTORY_EVENTS_PATH,
    HISTORY_MARKDOWN_PATH,
    ANALYSIS_JSON_PATH,
    ANALYSIS_MD_PATH,
    NOTES_PATH,
)

HTTP_REQUEST_SECONDS = 20
GITHUB_REQUEST_SECONDS = 30
BINANCE_RECV_WINDOW = 5000
BINANCE_RATE_LIMIT_FALLBACK_SECONDS = 60
BINANCE_RATE_LIMIT_SAFETY_SECONDS = 60
REAL_RECONCILE_INTERVAL_SECONDS = 2.0
REAL_PENDING_POLL_SECONDS = 5.0
REAL_PROTECT_CHECK_SECONDS = 60.0
AUTOSTOP_CHECK_SECONDS = 30.0
# Auto-trailing: saat profit >= trigger_R, SL pindah ke entry +/- lock_R x risiko awal.
AUTO_TRAIL_ENABLED = os.getenv("AUTO_TRAIL", "1").strip().lower() not in {"0", "false", "off"}
TRAIL_R_LADDER = [(1.0, 0.0), (1.5, 0.5), (2.0, 1.0), (3.0, 2.0)]
TRAIL_FEE_BUFFER_PCT = Decimal("0.12")  # lantai kunci >= fee taker round-trip + slippage
TRAIL_MIN_STEP_R = Decimal("0.10")
TRAIL_MIN_GAP_R = Decimal("0.30")
TRAIL_STRATEGY_INTERVAL_SECONDS = 90.0
# Pemasangan TP/SL boleh jalan setelah jeda server selesai (tanpa jeda aman bot).
REAL_PRIORITY: contextvars.ContextVar[bool] = contextvars.ContextVar("real_priority", default=False)
# Setup scan berkorelasi tinggi (alt searah): batasi eksposur sekaligus.
SCAN_MAX_NEW_PER_CYCLE = int(os.getenv("SCAN_MAX_NEW_PER_CYCLE", "3") or "3")
SCAN_MAX_PER_DIRECTION = int(os.getenv("SCAN_MAX_PER_DIRECTION", "6") or "6")
AUTOSTOP_INCLUDE_UNREALIZED = True  # equity = wallet balance + unrealized PnL
CONFIG_AUTOSTOP = os.getenv("AUTOSTOP_PERCENT", "").strip()
DEFAULT_MARGIN_USDT = Decimal("0.5")
DEFAULT_LEVERAGE = 10
REAL_NOTIONAL_MIN_RATIO = Decimal("0.90")
REAL_NOTIONAL_MAX_RATIO = Decimal("1.10")
REAL_FEE_BUFFER = Decimal("0.001")
QTY_FILTER_CODES = (-1013, -1111, -4004, -4005, -4164)
WS_RECONNECT_MIN = 2
WS_RECONNECT_MAX = 60
PRICE_STALE_SECONDS = 15
MAX_ACTIVE_TRADES = 100
HISTORY_REFRESH_SECONDS = 30

# SCAN configuration. /max and /threshold can override these at runtime.
DEFAULT_SCAN_THRESHOLD = Decimal("70")
SCAN_PAIR_DELAY_SECONDS = 1.0
SCAN_MAX_PAIRS_PER_CYCLE = 50
SCAN_CYCLE_DELAY_SECONDS = 30.0
# H4 scanner gate: scanner baru membuka window pada penutupan H4 WIB.
# Binance H4 boundaries dipetakan ke 03,07,11,15,19,23 WIB.
H4_SCAN_HOURS_WIB = (3, 7, 11, 15, 19, 23)
H4_SCAN_TRIGGER_LEEWAY_SECONDS = 60.0
H4_SCAN_SLOT_VALIDITY_SECONDS = 3600.0  # slot tetap valid maksimal +1 jam bila tertahan Binance.
H4_SCAN_SCHEDULER_POLL_SECONDS = 5.0
POSITION_CACHE_MAX_AGE_SECONDS = 30.0
SCAN_VALIDATION_TOLERANCE = Decimal("0.90")
SCAN_BANNED_PRICE_EXP_HOURS = Decimal("8")
SCAN_BANNED_TP_SL_HOURS = Decimal("24")
SCAN_BANNED_HISTORY_HOURS = Decimal("24")
SCAN_BANNED_LEVERAGE_HOURS = Decimal("4")
SCAN_BANNED_WAITING_HOURS = Decimal("2")  # menunggu trigger RSI
SCAN_MARGIN_TOLERANCE = Decimal("0.10")
SCAN_MARGIN_FAIL_CONFIRMATIONS = 3
BANNED_PATH = "data/banned_pairs.json"

# Optional margin/leverage inputs used only by the conservative Margin Influence detector.
# They may be supplied by the future /margin and /leverage execution layer.
CONFIG_MARGIN_USD = os.getenv("MARGIN_USD", "").strip()
CONFIG_LEVERAGE = os.getenv("LEVERAGE", "").strip()

if not ALLOWED_USER_ID:
    raise RuntimeError("ALLOWED_USER_ID belum diset.")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN belum diset.")

if not REPO_NAME or "/" not in REPO_NAME:
    raise RuntimeError("REPO_NAME belum diset atau formatnya tidak valid.")


# ============================================================
# LOGGING + TELEGRAM ERROR BRIDGE
# ============================================================

log = logging.getLogger("main.trading_engine")


class TelegramErrorHandler(logging.Handler):
    """Forward WARNING/ERROR logs + scan-cycle completion summaries to Telegram."""

    def __init__(self, engine: "TradingEngine") -> None:
        # INFO disaring manual di emit(); hanya summary cycle tertentu yang lolos.
        super().__init__(level=logging.INFO)
        self.engine = engine
        self.loop: asyncio.AbstractEventLoop | None = None

    def attach(self) -> None:
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    @staticmethod
    def _should_forward(record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return False

    def emit(self, record: logging.LogRecord) -> None:
        if self.loop is None or self.engine is None:
            return

        # Jangan membuat loop error baru ketika Telegram sender sendiri gagal.
        if "Gagal mengirim Telegram" in record.getMessage():
            return

        if not self._should_forward(record):
            return

        try:
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self.engine._send_log_error_to_telegram(record)
                )
            )
        except Exception:
            # Error handler tidak boleh menjatuhkan aplikasi.
            pass


# Render selalu menerima INFO proses + WARNING/ERROR. Telegram menerima WARNING/ERROR.
if not log.handlers:
    _render_handler = logging.StreamHandler()
    _render_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    log.addHandler(_render_handler)
log.setLevel(logging.INFO)
log.propagate = False


# ============================================================
# BASIC HELPERS
# ============================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_local() -> datetime:
    return now_utc().astimezone(TZ)


def iso_utc(dt: datetime | None = None) -> str:
    value = dt or now_utc()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def format_wib(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ).strftime("%d-%m-%Y, %H:%M WIB")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def decimal_to_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def parse_decimal(value: str) -> Decimal:
    text = str(value or "").strip().replace(",", ".")

    # Harga input dibuat ketat supaya tidak menerima karakter aneh.
    if not re.fullmatch(r"(?:\d+(?:\.\d+)?|\.\d+)", text):
        raise ValueError("Harga harus berupa angka positif, contoh 0.31555.")

    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Harga tidak valid.") from exc

    if not number.is_finite() or number <= 0:
        raise ValueError("Harga harus lebih besar dari 0.")

    return number


def parse_signed_decimal(value: str) -> Decimal:
    """Parse angka bertanda untuk field yang sah bernilai negatif/0/positif."""
    text = str(value or "").strip().replace(",", ".")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Angka bertanda tidak valid.") from exc
    if not number.is_finite():
        raise ValueError("Angka bertanda tidak valid.")
    return number


def parse_nonnegative_decimal(value: str) -> Decimal:
    """Parse angka yang boleh 0 tetapi tidak boleh negatif."""
    number = parse_signed_decimal(value)
    if number < 0:
        raise ValueError("Angka tidak boleh negatif.")
    return number


def normalize_symbol(value: str) -> str:
    text = str(value or "").upper().strip()
    text = text.replace("/", "").replace("-", "").replace("_", "")
    text = re.sub(r"\s+", "", text)
    return text


def safe_int(value: str, label: str) -> int:
    text = str(value or "").strip()
    if not text.isdigit():
        raise ValueError(f"{label} harus berupa angka bulat.")
    return int(text)


def generate_trade_id(pair: str) -> str:
    stamp = now_local().strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:6].upper()
    return f"{pair}-{stamp}-{suffix}"


def generate_session_id() -> str:
    stamp = now_local().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid4().hex[:4].upper()}"


def pct_change(direction: str, entry: Decimal, current: Decimal) -> Decimal:
    if entry == 0:
        return Decimal("0")

    if direction == "BUY":
        return ((current - entry) / entry) * Decimal("100")

    return ((entry - current) / entry) * Decimal("100")


def format_pct(value: Decimal | None) -> str:
    if value is None:
        return "-"
    return f"{value.quantize(Decimal('0.01')):+.2f}%"


def duration_text(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"

    total = max(0, int(seconds))
    days, total = divmod(total, 86400)
    hours, total = divmod(total, 3600)
    minutes, secs = divmod(total, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}h")
    if hours:
        parts.append(f"{hours}j")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def fmt_price(value: Decimal | None) -> str:
    """Harga untuk tampilan; membuang artefak desimal (mis. 0.013179999999999999)."""
    if value is None:
        return "-"
    try:
        return decimal_to_str(value.quantize(Decimal("0.0000000001"))) or "0"
    except InvalidOperation:
        return decimal_to_str(value) or "0"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return str(Decimal(str(value)).quantize(Decimal("1").scaleb(-digits)))
    except (InvalidOperation, ValueError):
        return str(value)


def card(title: str, rows: list[str]) -> str:
    body = "\n".join(f"│ {row}" for row in rows if row)
    return f"╭─ {title} ─╮\n{body}\n╰──────────────────╯"


def quantized_price(value: Decimal, tick_size: Decimal) -> Decimal:
    if tick_size <= 0:
        return value
    steps = (value / tick_size).to_integral_value(rounding=ROUND_DOWN)
    return steps * tick_size


def quantized_price_ceiling(value: Decimal, tick_size: Decimal) -> Decimal:
    if tick_size <= 0:
        return value
    steps = (value / tick_size).to_integral_value(rounding=ROUND_UP)
    return steps * tick_size


def validate_tick(value: Decimal, tick_size: Decimal) -> bool:
    if tick_size <= 0:
        return True
    remainder = value % tick_size
    return remainder == 0



# ============================================================
# REAL QUANTITY / PRECISION HELPERS
# ============================================================

def round_step_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def round_step_up(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def format_qty(qty: Decimal) -> str:
    """Plain decimal quantity string; never scientific notation."""
    return format(qty.normalize(), "f")


def parse_max_notional(response: Any) -> Decimal | None:
    raw = (response or {}).get("maxNotionalValue") if isinstance(response, dict) else None
    if raw in (None, ""):
        return None
    try:
        value = Decimal(str(raw))
    except InvalidOperation:
        return None
    return value if value.is_finite() and value > 0 else None


def compute_quantity(
    entry: Decimal,
    margin: Decimal,
    leverage: int,
    filters: dict[str, Any],
    available_balance: Decimal | None = None,
    max_notional: Decimal | None = None,
) -> tuple[Decimal | None, str]:
    """Compute a Binance-legal quantity constrained to target notional ±10%."""
    try:
        price = Decimal(str(entry))
        margin_d = Decimal(str(margin))
        lev = Decimal(str(leverage))
    except (InvalidOperation, ValueError, TypeError):
        return None, "INVALID_NUMERIC_INPUT"

    if not all(value.is_finite() for value in (price, margin_d, lev)):
        return None, "INVALID_NUMERIC_INPUT"
    if price <= 0 or margin_d <= 0 or lev < 1 or lev != lev.to_integral_value():
        return None, "INVALID_INPUT_RANGE"

    step = Decimal(str(filters.get("step_size") or "0"))
    if step <= 0:
        return None, "INVALID_STEP_SIZE"

    min_qty = Decimal(str(filters.get("min_qty") or "0"))
    max_qty = Decimal(str(filters.get("max_qty") or "0"))
    min_notional = Decimal(str(filters.get("min_notional") or "0"))
    filter_max_notional = Decimal(str(filters.get("max_notional") or "0"))
    if filter_max_notional > 0:
        max_notional = (
            filter_max_notional
            if max_notional is None or max_notional <= 0
            else min(max_notional, filter_max_notional)
        )

    target = margin_d * lev
    lower_notional = target * REAL_NOTIONAL_MIN_RATIO
    upper_notional = target * REAL_NOTIONAL_MAX_RATIO

    lower_qty = lower_notional / price
    upper_qty = upper_notional / price
    if min_qty > 0:
        lower_qty = max(lower_qty, min_qty)
    if max_qty > 0:
        upper_qty = min(upper_qty, max_qty)
    if min_notional > 0:
        lower_qty = max(lower_qty, min_notional / price)
    if max_notional is not None and max_notional > 0:
        upper_qty = min(upper_qty, max_notional / price)

    lower_step = round_step_up(lower_qty, step)
    upper_step = round_step_down(upper_qty, step)
    if lower_step <= 0:
        lower_step = step

    if lower_step > upper_step:
        candidates = [x for x in (lower_step, upper_step, round_step_down(target / price, step), round_step_up(target / price, step)) if x > 0]
        nearest = min(candidates, key=lambda q: abs(q * price - target), default=Decimal("0"))
        nearest_notional = nearest * price if nearest > 0 else Decimal("0")
        return None, (
            "MARGIN_DEVIATION_OUT_OF_BOUND "
            f"(target={decimal_to_str(target)}, nearest_notional={decimal_to_str(nearest_notional)})"
        )

    raw_qty = target / price
    candidates = {
        lower_step,
        upper_step,
        round_step_down(raw_qty, step),
        round_step_up(raw_qty, step),
    }
    valid: list[tuple[Decimal, Decimal]] = []
    for qty in candidates:
        if qty <= 0 or qty < lower_step or qty > upper_step:
            continue
        if min_qty > 0 and qty < min_qty:
            continue
        if max_qty > 0 and qty > max_qty:
            continue
        notional = qty * price
        if not (lower_notional <= notional <= upper_notional):
            continue
        if min_notional > 0 and notional < min_notional:
            continue
        if max_notional is not None and max_notional > 0 and notional > max_notional:
            continue
        valid.append((qty, notional))

    if not valid:
        nearest = min((lower_step, upper_step), key=lambda q: abs(q * price - target))
        return None, (
            "MARGIN_DEVIATION_OUT_OF_BOUND "
            f"(target={decimal_to_str(target)}, nearest_notional={decimal_to_str(nearest * price)})"
        )

    qty, notional = min(valid, key=lambda item: abs(item[1] - target))
    required_margin = notional / lev
    if required_margin < margin_d * REAL_NOTIONAL_MIN_RATIO or required_margin > margin_d * REAL_NOTIONAL_MAX_RATIO:
        return None, (
            "MARGIN_DEVIATION_OUT_OF_BOUND "
            f"(target={decimal_to_str(margin_d)}, actual={decimal_to_str(required_margin)})"
        )

    if available_balance is not None:
        required_cash = required_margin + notional * REAL_FEE_BUFFER
        if required_cash > available_balance:
            return None, (
                "INSUFFICIENT_AVAILABLE_BALANCE "
                f"(need={decimal_to_str(required_cash)}, avail={decimal_to_str(available_balance)})"
            )

    return qty, "OK"


def is_qty_filter_error(exc: Exception) -> bool:
    text = str(exc)
    return any(f"code {code}" in text or f"error {code}" in text for code in QTY_FILTER_CODES)


# ============================================================
# DATA MODELS
# ============================================================

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
        return max(
            0.0,
            (now_utc() - self.received_at).total_seconds(),
        )

    @property
    def live(self) -> bool:
        return self.age_seconds <= PRICE_STALE_SECONDS


@dataclass(slots=True)
class Trade:
    trade_id: str
    session_id: str
    pair: str
    direction: str

    price_now_reference: Decimal
    entry: Decimal
    entry_reason: str

    price_exp: Decimal
    price_exp_reason: str


    sl: Decimal
    sl_reason: str

    tp: Decimal
    tp_reason: str

    status: str = "PENDING"
    trailing: bool = False

    created_at: datetime = field(default_factory=now_utc)
    filled_at: datetime | None = None
    closed_at: datetime | None = None

    fill_price: Decimal | None = None
    exit_price: Decimal | None = None
    result: str | None = None
    result_reason: str | None = None

    pnl_percent: Decimal | None = None
    max_favorable_pct: Decimal | None = None
    max_adverse_pct: Decimal | None = None

    trail_history: list[dict[str, Any]] = field(default_factory=list)

    strategy_name: str = "MANUAL"
    strategy_version: str = "1.0"
    strategy_source: str = "MANUAL"
    strategy_confidence: Decimal | None = None
    strategy_data_source: str | None = None
    strategy_analysis: dict[str, Any] = field(default_factory=dict)

    # Real execution metadata. REAL is always opt-in at runtime.
    margin_usdt: Decimal = DEFAULT_MARGIN_USDT
    leverage: int = DEFAULT_LEVERAGE
    quantity: Decimal | None = None
    target_notional: Decimal | None = None
    actual_notional: Decimal | None = None
    real_enabled: bool = False
    real_state: str = "SIMULATION"
    position_side: str | None = None
    entry_order_id: int | None = None
    entry_client_order_id: str | None = None
    tp_algo_id: int | None = None
    sl_algo_id: int | None = None
    tp_client_algo_id: str | None = None
    sl_client_algo_id: str | None = None

    # Transactional SL replacement state.
    # ACTIVE SL is kept in sl_* until the replacement has been independently
    # confirmed on Binance. pending_sl_* describes the candidate being
    # installed/confirmed; it must never replace the active metadata early.
    pending_sl_algo_id: int | None = None
    pending_sl_client_algo_id: str | None = None
    pending_sl_price: Decimal | None = None
    pending_sl_state: str = "NONE"
    pending_sl_created_at: datetime | None = None
    pending_sl_error: str | None = None

    real_error: str | None = None
    last_real_check_monotonic: float = 0.0
    real_exit_check: str | None = None

    def to_record(self) -> dict[str, Any]:
        pending_seconds: float | None = None
        holding_seconds: float | None = None

        if self.filled_at:
            pending_seconds = (
                self.filled_at - self.created_at
            ).total_seconds()

        if self.filled_at and self.closed_at:
            holding_seconds = (
                self.closed_at - self.filled_at
            ).total_seconds()

        planned_risk = abs(self.entry - self.sl)
        planned_reward = abs(self.tp - self.entry)

        planned_rr: Decimal | None
        if planned_risk == 0:
            planned_rr = None
        else:
            planned_rr = planned_reward / planned_risk

        return {
            "trade_id": self.trade_id,
            "session_id": self.session_id,
            "pair": self.pair,
            "direction": self.direction,

            "price_now_reference": decimal_to_str(
                self.price_now_reference
            ),
            "entry": decimal_to_str(self.entry),
            "entry_reason": self.entry_reason,

            "price_exp": decimal_to_str(self.price_exp),
            "price_exp_reason": self.price_exp_reason,

            "sl": decimal_to_str(self.sl),
            "sl_reason": self.sl_reason,

            "tp": decimal_to_str(self.tp),
            "tp_reason": self.tp_reason,

            "status": self.status,
            "trailing": self.trailing,

            "created_at": iso_utc(self.created_at),
            "filled_at": iso_utc(self.filled_at) if self.filled_at else None,
            "closed_at": iso_utc(self.closed_at) if self.closed_at else None,

            "fill_price": decimal_to_str(self.fill_price),
            "exit_price": decimal_to_str(self.exit_price),

            "result": self.result,
            "result_reason": self.result_reason,

            "pnl_percent": (
                decimal_to_str(self.pnl_percent)
                if self.pnl_percent is not None
                else None
            ),
            "max_favorable_pct": (
                decimal_to_str(self.max_favorable_pct)
                if self.max_favorable_pct is not None
                else None
            ),
            "max_adverse_pct": (
                decimal_to_str(self.max_adverse_pct)
                if self.max_adverse_pct is not None
                else None
            ),

            "planned_rr": (
                decimal_to_str(planned_rr)
                if planned_rr is not None
                else None
            ),

            "pending_seconds": pending_seconds,
            "holding_seconds": holding_seconds,

            "trail_history": copy.deepcopy(self.trail_history),

            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "strategy_source": self.strategy_source,
            "strategy_confidence": (
                decimal_to_str(self.strategy_confidence)
                if self.strategy_confidence is not None
                else None
            ),
            "strategy_data_source": self.strategy_data_source,
            "strategy_analysis": copy.deepcopy(self.strategy_analysis),

            "margin_usdt": decimal_to_str(self.margin_usdt),
            "leverage": self.leverage,
            "quantity": decimal_to_str(self.quantity),
            "target_notional": decimal_to_str(self.target_notional),
            "actual_notional": decimal_to_str(self.actual_notional),
            "real_enabled": self.real_enabled,
            "real_state": self.real_state,
            "position_side": self.position_side,
            "entry_order_id": self.entry_order_id,
            "entry_client_order_id": self.entry_client_order_id,
            "tp_algo_id": self.tp_algo_id,
            "sl_algo_id": self.sl_algo_id,
            "tp_client_algo_id": self.tp_client_algo_id,
            "sl_client_algo_id": self.sl_client_algo_id,
            "pending_sl_algo_id": self.pending_sl_algo_id,
            "pending_sl_client_algo_id": self.pending_sl_client_algo_id,
            "pending_sl_price": decimal_to_str(self.pending_sl_price),
            "pending_sl_state": self.pending_sl_state,
            "pending_sl_created_at": (
                iso_utc(self.pending_sl_created_at)
                if self.pending_sl_created_at
                else None
            ),
            "pending_sl_error": self.pending_sl_error,
            "real_error": self.real_error,
            "real_exit_check": self.real_exit_check,
        }


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
}


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


class BinanceREST:
    """
    Public market-data REST only.

    Tidak ada signing.
    Tidak ada API order.
    """

    def __init__(self) -> None:
        self.base_url = BINANCE_REST_BASE

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        def request() -> Any:
            response = requests.get(
                f"{self.base_url}{path}",
                params=params or {},
                timeout=HTTP_REQUEST_SECONDS,
            )

            API_STATS.record(
                path,
                response.status_code,
                response.headers.get("X-MBX-USED-WEIGHT-1M"),
            )
            if response.status_code in {418, 429}:
                try:
                    retry = float(response.headers.get("Retry-After") or "")
                except ValueError:
                    retry = None
                server = retry if retry is not None else BINANCE_RATE_LIMIT_FALLBACK_SECONDS
                raise BinanceRateLimitError(
                    f"Binance REST {path}: HTTP {response.status_code}",
                    status_code=response.status_code,
                    code=None,
                    endpoint=path,
                    server_cooldown_seconds=server,
                    bot_cooldown_seconds=server + BINANCE_RATE_LIMIT_SAFETY_SECONDS,
                    retry_after_known=retry is not None,
                )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Binance REST {path}: HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

            return response.json()

        return await asyncio.to_thread(request)

    async def get_exchange_info(self, force: bool = False) -> dict[str, SymbolMeta]:
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

    async def get_24h_tickers(self) -> list[dict[str, Any]]:
        payload = await self._get("/fapi/v1/ticker/24hr")
        if not isinstance(payload, list):
            raise RuntimeError("Binance 24h ticker response bukan list.")
        return payload


# ============================================================
# BYBIT PUBLIC MARKET DATA
# ============================================================

class BybitPublicREST:
    """Public Bybit V5 market-data client for linear USDT perpetuals."""

    BASE_URL = "https://api.bybit.com"

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        def request() -> Any:
            response = requests.get(
                f"{self.BASE_URL}{path}",
                params=params or {},
                timeout=HTTP_REQUEST_SECONDS,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Bybit REST {path}: HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            body = response.json()
            if body.get("retCode") not in (0, None):
                raise RuntimeError(
                    f"Bybit REST {path}: retCode={body.get('retCode')} "
                    f"retMsg={body.get('retMsg')}"
                )
            return body

        return await asyncio.to_thread(request)

    async def get_linear_perpetual_symbols(self) -> set[str]:
        symbols: set[str] = set()
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {
                "category": "linear",
                "status": "Trading",
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor

            body = await self._get(
                "/v5/market/instruments-info",
                params=params,
            )
            result = body.get("result") or {}

            for item in result.get("list") or []:
                symbol = normalize_symbol(str(item.get("symbol") or ""))
                contract_type = str(item.get("contractType") or "")
                quote_coin = str(item.get("quoteCoin") or "")
                status = str(item.get("status") or "")
                if (
                    symbol
                    and quote_coin == "USDT"
                    and contract_type == "LinearPerpetual"
                    and status == "Trading"
                ):
                    symbols.add(symbol)

            cursor = str(result.get("nextPageCursor") or "") or None
            if not cursor:
                break

        return symbols

    async def get_linear_tickers(self) -> list[dict[str, Any]]:
        body = await self._get(
            "/v5/market/tickers",
            params={"category": "linear"},
        )
        result = body.get("result") or {}
        rows = result.get("list") or []
        if not isinstance(rows, list):
            raise RuntimeError("Bybit ticker response bukan list.")
        return rows


# ============================================================
# BINANCE REAL EXECUTION
# ============================================================

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


class LeverageNotSupportedError(RuntimeError):
    """Leverage yang diatur melebihi batas pair di Binance (code -4028)."""


class RealPositionConflictError(RuntimeError):
    """Real entry ditolak karena symbol sudah memiliki exposure Binance."""


class MarginInfluenceError(RuntimeError):
    pass


class BinanceRealClient:
    """Signed USD-M Futures client used only when /real is ON."""

    def __init__(self) -> None:
        self.base_url = BINANCE_REST_BASE
        self.api_key = BINANCE_API_KEY
        self.api_secret = BINANCE_API_SECRET
        self.recv_window = BINANCE_RECV_WINDOW
        self._time_offset_ms = 0
        self._cooldown_until = 0.0
        self._server_cooldown_until = 0.0
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

    def apply_cooldown(self, server_seconds: float, bot_seconds: float) -> None:
        now = time.monotonic()
        self._cooldown_until = max(self._cooldown_until, now + bot_seconds)
        self._server_cooldown_until = max(self._server_cooldown_until, now + server_seconds)

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
        if signed and not self.configured:
            raise BinanceAPIError(
                "BINANCE_API_KEY / BINANCE_API_SECRET belum tersedia di environment.",
                endpoint=path,
            )

        remaining = (
            self.server_cooldown_remaining if REAL_PRIORITY.get() else self.cooldown_remaining
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
            retry_after = self._parse_retry_after(response)
            server_seconds = retry_after if retry_after is not None else BINANCE_RATE_LIMIT_FALLBACK_SECONDS
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
                retry_after_known=(retry_after is not None),
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

    async def place_close_algo(
        self,
        *,
        symbol: str,
        side: str,
        position_side: str,
        order_type: str,
        trigger_price: Decimal,
        client_algo_id: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "triggerPrice": decimal_to_str(trigger_price),
            "workingType": "CONTRACT_PRICE",
            "closePosition": "true",
            "clientAlgoId": client_algo_id,
        }
        if position_side:
            params["positionSide"] = position_side
        payload = await self._request("POST", "/fapi/v1/algoOrder", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons new algo order Binance tidak valid.", endpoint="/fapi/v1/algoOrder")
        return payload

    async def get_open_algo_orders(self, symbol: str) -> list[dict[str, Any]]:
        payload = await self._request(
            "GET",
            "/fapi/v1/openAlgoOrders",
            {"symbol": symbol, "algoType": "CONDITIONAL"},
        )
        if not isinstance(payload, list):
            raise BinanceAPIError("Respons openAlgoOrders Binance tidak valid.", endpoint="/fapi/v1/openAlgoOrders")
        return [item for item in payload if isinstance(item, dict)]

    async def get_algo_order(self, *, algo_id: int) -> dict[str, Any]:
        payload = await self._request(
            "GET",
            "/fapi/v1/algoOrder",
            {"algoId": algo_id},
        )
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons algoOrder Binance tidak valid.", endpoint="/fapi/v1/algoOrder")
        return payload

    async def cancel_algo_order(self, *, symbol: str, algo_id: int | None = None, client_algo_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if algo_id is not None:
            params["algoId"] = algo_id
        elif client_algo_id:
            params["clientAlgoId"] = client_algo_id
        else:
            raise ValueError("algo_id atau client_algo_id harus tersedia.")
        payload = await self._request("DELETE", "/fapi/v1/algoOrder", params)
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons cancel algo Binance tidak valid.", endpoint="/fapi/v1/algoOrder")
        return payload

    async def cancel_all_algo_orders(self, symbol: str) -> dict[str, Any]:
        payload = await self._request(
            "DELETE",
            "/fapi/v1/algoOpenOrders",
            {"symbol": symbol},
        )
        if not isinstance(payload, dict):
            raise BinanceAPIError("Respons cancel all algo Binance tidak valid.", endpoint="/fapi/v1/algoOpenOrders")
        return payload

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
                        # /market/ws adalah raw-stream mode.
                        # Dalam mode ini event aggTrade datang langsung
                        # di root payload:
                        # {"e":"aggTrade", ...}
                        #
                        # Combined mode (/market/stream) membungkusnya:
                        # {"stream":"...","data":{"e":"aggTrade", ...}}
                        #
                        # Dukung keduanya supaya parser tahan terhadap
                        # perubahan mode stream di kemudian hari.
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
                            # Event lama/duplikat. Abaikan supaya event stale
                            # tidak bisa memicu Entry, Price Exp, SL, atau TP.
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



# ============================================================
# MAIN ENGINE
# ============================================================

class TradingEngine:
    def __init__(self, context: dict[str, Any]) -> None:
        self.context = dict(context)

        self.chat_id = int(
            context.get("chat_id")
            or ALLOWED_USER_ID
        )

        self.user_id = int(
            context.get("user_id")
            or ALLOWED_USER_ID
        )

        self.send_message = context[
            "send_message"
        ]
        self.send_document = context.get("send_document")

        self.session_id = generate_session_id()

        self.rest = BinanceREST()
        self.bybit = BybitPublicREST()
        self.github = GitHubStore()

        # REAL execution is always OFF on every fresh engine session.
        self.real_mode = False
        self.margin_usdt = self._parse_optional_decimal(CONFIG_MARGIN_USD) or DEFAULT_MARGIN_USDT
        leverage_cfg = self._parse_optional_decimal(CONFIG_LEVERAGE)
        self.leverage = int(leverage_cfg) if leverage_cfg and leverage_cfg == leverage_cfg.to_integral_value() else DEFAULT_LEVERAGE
        self.real = BinanceRealClient()
        self._real_reconcile_guard: dict[str, float] = {}
        self._real_poll_guard: dict[str, float] = {}
        self._real_protect_fail: dict[str, int] = {}

        self.symbols: dict[str, SymbolMeta] = {}
        self.prices: dict[str, PriceSnapshot] = {}

        self.active_trades: dict[str, Trade] = {}

        self.history_records: list[dict[str, Any]] = []
        self.history_events: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []

        # Market-event ordering is also guarded at engine level. This is a
        # second safety layer so an out-of-order callback can never trigger
        # a state transition. Key = (Binance event time, aggregate trade ID).
        self._last_market_event_key: dict[str, tuple[int, int]] = {}
        self._last_live_price: dict[str, Decimal] = {}


        self.flow: dict[str, Any] | None = None
        self._auto_task: asyncio.Task[Any] | None = None
        self._auto_job_id: str | None = None

        # SCAN runtime state. Manual ON/OFF intent is kept separate from the
        # temporary capacity pause caused by /max. /scan defaults OFF on restart.
        self.scan_threshold = DEFAULT_SCAN_THRESHOLD
        self.max_active_trades = MAX_ACTIVE_TRADES
        self.scan_margin_usd = self._parse_optional_decimal(CONFIG_MARGIN_USD) or self.margin_usdt
        self.scan_leverage = self._parse_optional_decimal(CONFIG_LEVERAGE) or Decimal(str(self.leverage))
        self._scan_user_enabled = False
        self._scan_auto_paused = False
        self._scan_task: asyncio.Task[Any] | None = None
        self._scan_cycle_number = 0
        self._scan_last_report: dict[str, Any] = {}
        self._scan_margin_streak: dict[str, int] = {}

        # H4 scanner gate. H4 defaults OFF per fresh main.py session.
        # _h4_window_active means the current H4 close has opened a scanning
        # window; it is closed again when max active trade is reached.
        self._h4_enabled = False
        self._h4_window_active = False
        self._h4_last_trigger_slot: str | None = None
        self._h4_window_slot: datetime | None = None
        self._h4_window_deadline: datetime | None = None
        self._h4_invalid_slot_key: str | None = None
        self._h4_opening_gate = False
        self._scan_generation = 0

        # AUTOSTOP: trailing max drawdown atas equity Binance (hanya saat REAL ON).
        default_autostop = self._parse_optional_decimal(CONFIG_AUTOSTOP)
        self.autostop_percent: Decimal | None = (
            default_autostop if default_autostop is not None and default_autostop < 100 else None
        )
        self._equity_peak: Decimal | None = None
        self._equity_last: Decimal | None = None
        self._autostop_triggered = False
        self._autostop_failures = 0
        self._autostop_task: asyncio.Task[Any] | None = None

        # Rate limit Binance: satu notifikasi per jeda + watcher yang melanjutkan.
        self._rl_task: asyncio.Task[Any] | None = None
        self._rl_notified_until = 0.0
        self._rl_started_at: float | None = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        # Antrian saat API dibatasi (bisa menumpuk): konfirmasi fill & TP/SL tertunda.
        self._deferred_fill: dict[str, float] = {}
        self._deferred_protect: dict[str, float] = {}
        self._deferred_cleanup: dict[str, Trade] = {}
        self._trail_busy: set[str] = set()
        self._trail_next_check: dict[str, float] = {}
        self._deferred_trail: dict[str, tuple[Decimal, str, str, Decimal]] = {}
        # Serialize the two-phase SL transition separately from the trailing
        # calculator. Recovery/reconcile may run even when no price tick is
        # currently processing the trade.
        self._sl_transition_busy: set[str] = set()
        self._close_batch_in_progress = False

        self.banned_pairs: dict[str, dict[str, Any]] = {}
        self._ban_lock = asyncio.Lock()
        self._strategy_runtime_module = None

        self.ws = BinanceWebSocket(
            self._on_price
        )

        self._running = False

        self._trade_lock = asyncio.Lock()
        # Menserialkan satu lifecycle history (event + trade + markdown)
        # agar dua trade yang selesai bersamaan tidak saling menimpa snapshot.
        self._history_lock = asyncio.Lock()

        self._telegram_error_handler = TelegramErrorHandler(self)
        self._telegram_error_handler_added = False

        self.last_history_refresh: datetime | None = None
        # True ketika histori lokal sudah berubah tetapi seluruh snapshot
        # GitHub belum berhasil ditulis. Selama dirty, refresh berkala tidak
        # boleh menimpa histori RAM dengan hasil remote yang lebih lama/kosong.
        self._history_dirty = False

    @staticmethod
    def _parse_optional_decimal(value: Any) -> Decimal | None:
        if value in (None, ""):
            return None
        try:
            parsed = Decimal(str(value))
        except InvalidOperation:
            return None
        return parsed if parsed > 0 else None

    # --------------------------------------------------------
    # Telegram send
    # --------------------------------------------------------

    async def reply(self, text: str) -> None:
        try:
            await asyncio.to_thread(
                self.send_message,
                self.chat_id,
                str(text),
            )
        except Exception:
            # Tidak memanggil log.exception di sini karena error handler
            # sendiri menggunakan Telegram dan bisa membuat recursion.
            pass

    async def _send_log_error_to_telegram(
        self,
        record: logging.LogRecord,
    ) -> None:
        """Send a readable backend error to the active Telegram chat."""
        try:
            trace = ""
            if record.exc_info:
                trace = "".join(
                    traceback.format_exception(
                        *record.exc_info
                    )
                )

            message = record.getMessage()
            if record.levelno >= logging.ERROR:
                title = "🚨 ERROR BACKEND"
            elif record.levelno >= logging.WARNING:
                title = "⚠️ WARNING BACKEND"
            else:
                title = "🔄 SCAN INFO"
            text = (
                f"{title}\n\n"
                f"Module: {record.name}\n"
                f"Level: {record.levelname}\n"
                f"Message: {record.getMessage()}\n"
                f"Time: {format_wib(now_utc())}"
            )

            if trace:
                # Telegram message limit ≈4096. Keep the useful tail of trace.
                trace = trace.strip()
                available = 3000
                if len(trace) > available:
                    trace = "..." + trace[-available:]
                text += f"\n\nTraceback:\n{trace}"

            await self.reply(text)
        except Exception:
            # Jangan sampai error reporter sendiri menjadi sumber error baru.
            pass

    async def send_document_file(
        self,
        path: str | Path,
        caption: str = "",
    ) -> None:
        """Kirim file lokal ke Telegram melalui callback dari try.py."""
        if not callable(self.send_document):
            raise RuntimeError(
                "Launcher belum menyediakan callback send_document."
            )

        await asyncio.to_thread(
            self.send_document,
            self.chat_id,
            str(path),
            str(caption or ""),
        )

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return

        self._running = True

        if not self._telegram_error_handler_added:
            # Hapus bridge dari engine lama agar /try reload tidak menggandakan
            # WARNING/ERROR ke Telegram. Render StreamHandler tetap dipertahankan.
            for handler in list(log.handlers):
                if isinstance(handler, TelegramErrorHandler) and handler is not self._telegram_error_handler:
                    log.removeHandler(handler)
            self._telegram_error_handler.attach()
            log.addHandler(self._telegram_error_handler)
            self._telegram_error_handler_added = True

        await self._load_history()
        await self._load_notes()
        await self._load_banned_pairs()

        try:
            self.symbols = await self.rest.get_exchange_info()
            log.info("[MAIN] Binance symbols loaded: %s", len(self.symbols))
        except BinanceRateLimitError as exc:
            self.real.apply_cooldown(exc.server_cooldown_seconds, exc.bot_cooldown_seconds)
            self._notify_rate_limit(exc, "START")
            log.info("[MAIN] exchangeInfo tertunda (rate limit); symbols dimuat ulang saat API pulih.")

        await self.ws.start()

        self.last_history_refresh = now_utc()

        await self.reply(
            "🟢 MAIN.PY ONLINE\n\n"
            f"Mode: {'REAL' if self.real_mode else 'SIMULATION'}\n"
            "Market: Binance USDⓈ-M Futures\n"
            "WebSocket: CONNECTING\n"
            f"Session: {self.session_id}\n\n"
            "Active Trade: 0\n\n"
            "Gunakan /add untuk membuat setup."
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        await self._stop_autostop_task()
        await self._stop_rate_limit_task()

        if self._scan_task is not None and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        self._scan_task = None

        if self._auto_task is not None and not self._auto_task.done():
            self._auto_task.cancel()
            try:
                await self._auto_task
            except asyncio.CancelledError:
                pass
        self._auto_task = None
        self._auto_job_id = None

        await self.ws.stop()

        # /end harus benar-benar menghapus state sesi dari RAM.
        self.flow = None
        self.active_trades.clear()
        self.prices.clear()
        self.symbols.clear()

        self.history_records.clear()
        self.history_events.clear()
        self._last_market_event_key.clear()
        self._last_live_price.clear()

        self.last_history_refresh = None
        self.session_id = ""

        if self._telegram_error_handler_added:
            try:
                log.removeHandler(
                    self._telegram_error_handler
                )
            except Exception:
                pass
            self._telegram_error_handler_added = False

        # Tidak ada write state/checkpoint ke disk.
        log.info(
            "Main session dibersihkan tanpa persistence: %s",
            MAIN_FILE,
        )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    async def _load_history(self) -> None:
        raw_trades, _ = await self.github.get_file(
            HISTORY_TRADES_PATH
        )

        if raw_trades:
            try:
                payload = json.loads(
                    raw_trades.decode(
                        "utf-8"
                    )
                )

                if isinstance(payload, list):
                    self.history_records = payload
                else:
                    self.history_records = []

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{HISTORY_TRADES_PATH} bukan JSON valid."
                ) from exc

        else:
            self.history_records = []

        raw_events, _ = await self.github.get_file(
            HISTORY_EVENTS_PATH
        )

        self.history_events = []
        self._history_dirty = False

        if raw_events:
            for line in raw_events.decode(
                "utf-8",
                errors="replace",
            ).splitlines():

                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(
                        "events.jsonl memiliki baris invalid."
                    )
                    continue

                if isinstance(item, dict):
                    self.history_events.append(
                        item
                    )

    async def _refresh_history_if_needed(
        self,
    ) -> None:
        if self.last_history_refresh is None:
            await self._load_history()
            self.last_history_refresh = now_utc()
            return

        age = (
            now_utc() - self.last_history_refresh
        ).total_seconds()

        if age < HISTORY_REFRESH_SECONDS:
            return

        # Jangan overwrite histori lokal yang baru saja tercatat tetapi belum
        # berhasil tersinkron penuh ke GitHub. Ini mencegah /stats berubah
        # menjadi 0 hanya karena refresh remote membaca snapshot lama/kosong.
        if self._history_dirty:
            log.warning(
                "[HISTORY] refresh GitHub dilewati karena histori lokal masih dirty; "
                "RAM dipertahankan (%s record).",
                len(self.history_records),
            )
            self.last_history_refresh = now_utc()
            return

        await self._load_history()
        self.last_history_refresh = now_utc()

    # --------------------------------------------------------
    # CATATAN / NOTES
    # --------------------------------------------------------

    async def _load_notes(self) -> None:
        raw, _ = await self.github.get_file(NOTES_PATH)

        if not raw:
            self.notes = []
            return

        try:
            payload = json.loads(
                raw.decode("utf-8")
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{NOTES_PATH} bukan JSON valid."
            ) from exc

        if not isinstance(payload, list):
            raise RuntimeError(
                f"{NOTES_PATH} harus berisi JSON list."
            )

        self.notes = [
            item
            for item in payload
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]

    async def add_note(self, text: str) -> None:
        note_text = str(text or "").strip()
        if not note_text:
            raise ValueError(
                "Catatan tidak boleh kosong. Gunakan: /catatan isi catatan"
            )

        if len(note_text) > 3000:
            raise ValueError(
                "Catatan terlalu panjang. Maksimum 3000 karakter."
            )

        await self._load_notes()

        note = {
            "note_id": uuid4().hex[:12].upper(),
            "text": note_text,
            "created_at": iso_utc(),
            "created_at_wib": format_wib(now_utc()),
        }

        self.notes.append(note)

        content = json.dumps(
            self.notes,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        try:
            await self.github.replace_file(
                NOTES_PATH,
                content,
                "notes: add catatan",
            )
        except Exception:
            # Jangan meninggalkan catatan lokal seolah-olah tersimpan
            # permanen ketika commit GitHub gagal.
            self.notes.pop()
            raise

        await self.reply(
            "📝 CATATAN DITAMBAHKAN\n\n"
            f"{note_text}\n\n"
            f"Waktu: {note['created_at_wib']}"
        )

    async def show_notes(self) -> None:
        await self._load_notes()

        if not self.notes:
            await self.reply(
                "📝 CATATAN\n\n"
                "Belum ada catatan.\n"
                "Gunakan /catatan isi catatan untuk menambahkan."
            )
            return

        lines = [
            "📝 CATATAN",
            "",
        ]

        for index, note in enumerate(self.notes, start=1):
            text = str(note.get("text") or "").strip()
            created = str(
                note.get("created_at_wib")
                or note.get("created_at")
                or "-"
            )
            lines.append(
                f"{index}. {text}\n"
                f"   🕒 {created}"
            )

        await self.reply("\n\n".join(lines))

    async def _set_real_on(self) -> None:
        if self.real_mode:
            await self.reply("🟢 REAL MODE SUDAH ON")
            return

        if not self.real.configured:
            raise ValueError(
                "BINANCE_API_KEY dan BINANCE_API_SECRET belum tersedia di environment."
            )

        if self.real.cooldown_remaining > 0:
            remaining = self.real.cooldown_remaining
            raise ValueError(
                f"Binance sedang cooldown rate-limit. Tunggu sekitar {remaining:.0f} detik."
            )

        try:
            # Exactly one private balance request when /real is enabled.
            balances = await self.real.get_futures_balance()
            usdt = next(
                (row for row in balances if str(row.get("asset") or "").upper() == "USDT"),
                None,
            )
            if usdt is None:
                raise BinanceAPIError(
                    "USDT tidak ditemukan pada Futures balance.",
                    endpoint="/fapi/v3/balance",
                )
            wallet_balance = parse_signed_decimal(str(usdt.get("balance") or "0"))

            # Cache the current exchange account position mode once so the
            # first real order can choose BOTH/LONG/SHORT correctly.
            await self.real.ensure_position_mode()
            try:
                await self.real.get_account_info()
            except Exception as exc:
                # Position cache is an optimization/filter; real entry still has
                # its final fresh positionRisk safety check.
                log.info("[REAL MODE] initial position snapshot gagal: %s", exc)

            self.real_mode = True
            self.real.last_balance = dict(usdt)
            self.real.last_balance_at = time.monotonic()
            self._equity_peak = self._current_equity_cached()
            self._equity_last = self._equity_peak
            self._autostop_triggered = False
            self._autostop_failures = 0
            self._start_autostop_task()

            await self.reply(
                "🔴🤖 REAL MODE AKTIF\n\n"
                "Private Binance API: ✅ Terhubung\n"
                f"USDT Balance: {decimal_to_str(wallet_balance)}\n"
                f"Current Equity: {self._fmt_usd(self._equity_last)} USDT\n"
                f"Margin: {decimal_to_str(self.margin_usdt)} USDT\n"
                f"Leverage: {self.leverage}x\n"
                f"Autostop: {self._autostop_short()}\n\n"
                "⚠️ Order real sekarang dapat dibuat oleh setup yang memakai mode REAL."
            )
        except BinanceRateLimitError as exc:
            self._notify_rate_limit(exc, "REAL MODE ON")
            raise
        except Exception as exc:
            self.real_mode = False
            await self._handle_real_exception(exc, "REAL MODE ON")
            raise

    async def _set_real_off(self) -> None:
        if not self.real_mode:
            await self.reply("🔴 REAL MODE SUDAH OFF")
            return

        protected_real = [
            trade
            for trade in self.active_trades.values()
            if trade.real_enabled and trade.result is None
        ]

        if protected_real:
            pairs = ", ".join(sorted({trade.pair for trade in protected_real}))
            await self.reply(
                "⚠️ REAL MODE TIDAK DIMATIKAN\n\n"
                "Masih ada setup yang terhubung ke Binance real.\n"
                f"Pair: {pairs}\n\n"
                "Gunakan /del untuk membersihkan order/posisi real terlebih dahulu."
            )
            return

        self.real_mode = False
        await self._stop_autostop_task()
        self._equity_peak = None
        self._equity_last = None
        self._autostop_triggered = False
        await self.reply(
            "🔴 REAL MODE OFF\n\n"
            "Tidak ada order real baru yang akan dibuat.\n"
            "WebSocket tetap berjalan.\n"
            "Setup simulasi tetap berjalan.\n"
            "Tidak ada order/posisi Binance yang disentuh otomatis."
        )

    @staticmethod
    def _api_weight_line() -> str:
        used = API_STATS.used_weight
        bot = API_STATS.weight_of(API_STATS.window(60, API_STATS.used_weight_at))
        ip_text = f"{used}/{BINANCE_IP_WEIGHT_LIMIT}" if used is not None else "-"
        return f"📊 Bobot IP  {ip_text}  (bot ≈{bot}/menit)"

    def _api_report(self) -> str:
        m1 = API_STATS.window(60)
        m5 = API_STATS.window(300)
        w1 = API_STATS.weight_of(m1)
        w5 = API_STATS.weight_of(m5)
        used = API_STATS.used_weight
        at = API_STATS.used_weight_at
        age = time.monotonic() - at if at is not None else None

        ip_text = "-"
        verdict = "Belum ada header bobot dari Binance; tunggu beberapa request."
        if used is not None and age is not None:
            ip_text = f"{used}/{BINANCE_IP_WEIGHT_LIMIT} ({age:.0f}s lalu) • puncak {API_STATS.max_used_weight}"
            bot_then = API_STATS.weight_of(API_STATS.window(60, at))
            if age > 180:
                verdict = "Header terakhir sudah lama; belum bisa disimpulkan."
            elif used > bot_then * 2 + 100:
                verdict = (
                    f"Bobot IP ({used}) jauh di atas bobot bot (≈{bot_then}): "
                    "kuota IP dipakai pihak lain."
                )
            else:
                verdict = (
                    f"Bobot IP ({used}) sejalan dengan bobot bot (≈{bot_then}): "
                    "pemakaian terutama dari bot."
                )

        cooldown = self.real.cooldown_remaining
        last_limit = format_wib(API_STATS.last_limit_at) if API_STATS.last_limit_at else "-"
        head = card(
            "📡 BINANCE API",
            [
                f"⏱ 1 menit   {sum(m1.values())} req  ≈{w1} bobot",
                f"⏱ 5 menit   {sum(m5.values())} req  ≈{w5} bobot (≈{w5 / 5:.0f}/menit)",
                f"📊 Bobot IP  {ip_text}",
                f"🚦 Limit     {API_STATS.limit_hits}x 429/418 • terakhir {last_limit}",
                f"⏸ Cooldown   {f'{cooldown:.0f}s tersisa' if cooldown > 0 else 'tidak aktif'}",
                f"🧮 Total     {API_STATS.total} request sejak bot start",
            ],
        )
        ranked = sorted(
            m5.items(),
            key=lambda item: ENDPOINT_WEIGHTS.get(item[0], 1) * item[1],
            reverse=True,
        )[:6]
        lines = [head, "", "Endpoint teratas (5 menit):"]
        if ranked:
            for path, count in ranked:
                lines.append(f"• {path.replace('/fapi/', '')}  {count}x  ≈{ENDPOINT_WEIGHTS.get(path, 1) * count}")
        else:
            lines.append("• belum ada request")
        lines += [
            "",
            f"🔎 {verdict}",
            "Bobot bot = estimasi dari tabel bobot; bobot IP = header Binance (semua pengguna IP yang sama).",
        ]
        return "\n".join(lines)

    def _fire(self, coro: Any) -> None:
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
            return
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _notify_rate_limit(self, exc: BinanceRateLimitError, action: str) -> None:
        """Satu pesan per jeda limit; penolakan cooldown internal tidak dikirim."""
        self._start_rate_limit_watcher()
        if getattr(exc, "internal", False):
            return
        bot = max(0.0, exc.bot_cooldown_seconds)
        now = time.monotonic()
        if now + bot <= self._rl_notified_until + 5.0:
            return
        self._rl_notified_until = now + bot
        if self._rl_started_at is None:
            self._rl_started_at = now
        log.info("Binance API LIMIT | action=%s | endpoint=%s | HTTP=%s | code=%s | %s",
                 action, exc.endpoint, exc.status_code, exc.code, str(exc)[:300])
        self._fire(
            self.reply(
                card(
                    "⏸ BINANCE API LIMIT",
                    [
                        f"🔧 Aksi      {action}",
                        f"🌐 Endpoint  {exc.endpoint}",
                        f"⏱ Jeda      {duration_text(bot)}",
                        self._api_weight_line(),
                        f"▶️ Lanjut    {format_wib(now_utc() + timedelta(seconds=bot))}",
                        "📌 Ditunda: scan & operasi REAL yang membutuhkan REST. Autostop tetap dihitung dari snapshot lokal.",
                        "👀 Harga tetap dipantau via WebSocket; autostop tetap aktif secara lokal dan tidak menunggu REST.",
                        "Lanjut otomatis dan dikabari saat Binance tersambung.",
                    ],
                )
            )
        )

    def _start_rate_limit_watcher(self) -> None:
        if not self._running:
            return
        if self._rl_task is not None and not self._rl_task.done():
            return
        try:
            self._rl_task = asyncio.get_running_loop().create_task(
                self._rate_limit_watcher(),
                name="main-rate-limit-watcher",
            )
        except RuntimeError:
            return

    async def _stop_rate_limit_task(self) -> None:
        task = self._rl_task
        self._rl_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _rate_limit_watcher(self) -> None:
        """Tunggu jeda selesai, tes koneksi, lalu kerjakan yang tertunda."""
        failures = 0
        while self._running:
            bot_rem = self.real.cooldown_remaining
            srv_rem = self.real.server_cooldown_remaining
            if bot_rem > 0:
                if self._deferred_any() and srv_rem <= 0:
                    await self._flush_deferred()
                    if self._deferred_any():
                        await asyncio.sleep(2.0)
                        continue
                wait = srv_rem if (self._deferred_any() and srv_rem > 0) else bot_rem
                await asyncio.sleep(min(wait + 1.0, 60.0))
                continue
            try:
                await self.real._public_server_time()
            except Exception as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if status in (418, 429):
                    retry = self.real._parse_retry_after(response)
                    server = retry if retry is not None else BINANCE_RATE_LIMIT_FALLBACK_SECONDS
                    bot = server + BINANCE_RATE_LIMIT_SAFETY_SECONDS
                    self.real.apply_cooldown(server, bot)
                    self._notify_rate_limit(
                        BinanceRateLimitError(
                            f"HTTP {status}",
                            status_code=status,
                            code=None,
                            endpoint="/fapi/v1/time",
                            server_cooldown_seconds=server,
                            bot_cooldown_seconds=bot,
                            retry_after_known=retry is not None,
                        ),
                        "CEK KONEKSI",
                    )
                    continue
                failures += 1
                log.info("[RATE LIMIT] cek koneksi gagal (%s/6): %s", failures, exc)
                if failures >= 6:
                    return
                await asyncio.sleep(15)
                continue

            started = self._rl_started_at
            self._rl_started_at = None
            self._rl_notified_until = 0.0
            if started is not None:
                await self.reply(
                    card(
                        "▶️ BINANCE API TERSAMBUNG",
                        [
                            f"⏱ Terhenti  {duration_text(time.monotonic() - started)}",
                            "✅ Melanjutkan: scan, cek fill & TP/SL, autostop",
                        ],
                    )
                )
            await self._resume_after_rate_limit()
            if self.real.cooldown_remaining > 0:
                continue
            return

    async def _resume_after_rate_limit(self) -> None:
        """Kerjakan pengecekan REAL yang tertunda selama jeda."""
        if not self.symbols:
            try:
                self.symbols = await self.rest.get_exchange_info()
                log.info("[MAIN] Binance symbols loaded: %s", len(self.symbols))
            except BinanceRateLimitError as exc:
                self.real.apply_cooldown(exc.server_cooldown_seconds, exc.bot_cooldown_seconds)
                self._notify_rate_limit(exc, "LOAD SYMBOLS")
                return
            except Exception as exc:
                log.info("[RATE LIMIT] load symbols gagal: %s", exc)
        if not self.real_mode:
            return
        try:
            await self.real.get_futures_balance()
        except BinanceRateLimitError as exc:
            self.real.apply_cooldown(exc.server_cooldown_seconds, exc.bot_cooldown_seconds)
            self._notify_rate_limit(exc, "REFRESH BALANCE")
            return
        except Exception as exc:
            log.info("[RATE LIMIT] refresh balance gagal: %s", exc)
        try:
            await self.real.get_account_info()
        except BinanceRateLimitError as exc:
            self.real.apply_cooldown(exc.server_cooldown_seconds, exc.bot_cooldown_seconds)
            self._notify_rate_limit(exc, "REFRESH POSITIONS")
            return
        except Exception as exc:
            log.info("[RATE LIMIT] refresh positions gagal: %s", exc)
        await self._flush_deferred()
        for trade in list(self.active_trades.values()):
            if not (trade.real_enabled and trade.status in {"PENDING", "FILLED"}):
                continue
            if self.real.cooldown_remaining > 0:
                return
            try:
                await self._reconcile_real_trade(trade)
            except BinanceRateLimitError as exc:
                self._notify_rate_limit(exc, "LANJUT TERTUNDA")
                return
            except Exception as exc:
                log.info("[RATE LIMIT] reconcile %s gagal: %s", trade.trade_id, exc)
            await asyncio.sleep(0.5)
        await self._autostop_check()

    async def _handle_real_exception(self, exc: Exception, action: str, trade: Trade | None = None) -> None:
        pair = trade.pair if trade else "-"
        trade_id = trade.trade_id if trade else "-"
        if isinstance(exc, BinanceRateLimitError):
            self._notify_rate_limit(exc, action)
            return
        log.exception(
            "REAL API error | action=%s | pair=%s | trade_id=%s | %s",
            action,
            pair,
            trade_id,
            exc,
        )

    def _position_side_for_direction(self, direction: str) -> str:
        if self.real._dual_side_position:
            return "LONG" if direction == "BUY" else "SHORT"
        return "BOTH"

    @staticmethod
    def _client_id(prefix: str, trade_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_:\-./]", "-", trade_id)
        return (f"{prefix}-{safe}")[:36]

    @staticmethod
    def _ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        steps = (value / step).to_integral_value(rounding=ROUND_CEILING)
        return steps * step

    @staticmethod
    def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        steps = (value / step).to_integral_value(rounding=ROUND_DOWN)
        return steps * step

    def _auto_quantity(
        self,
        meta: SymbolMeta,
        entry: Decimal,
        margin: Decimal,
        leverage: int,
        available_balance: Decimal | None = None,
        max_notional: Decimal | None = None,
    ) -> tuple[Decimal, Decimal, Decimal]:
        filters = {
            "step_size": meta.step_size,
            "min_qty": meta.min_qty,
            "max_qty": meta.max_qty,
            "min_notional": meta.min_notional,
            "max_notional": meta.max_notional,
            "market_max_qty": meta.market_max_qty,
        }
        effective_max_notional = max_notional
        if meta.max_notional > 0:
            effective_max_notional = (
                meta.max_notional
                if effective_max_notional is None
                else min(effective_max_notional, meta.max_notional)
            )
        quantity, reason = compute_quantity(
            entry,
            margin,
            leverage,
            filters,
            available_balance=available_balance,
            max_notional=effective_max_notional,
        )
        if quantity is None:
            raise MarginInfluenceError(f"{meta.symbol}: {reason}")
        actual = quantity * entry
        return quantity, margin * Decimal(leverage), actual

    async def _ensure_real_entry(self, trade: Trade) -> None:
        if not self.real_mode or trade.entry_order_id or trade.entry_client_order_id:
            return

        client_id = self._client_id("ENT", trade.trade_id)

        cached_position = self.real.cached_position(trade.pair, trade.direction)
        if cached_position is not None:
            amount = cached_position.get("_position_amount_decimal") or cached_position.get("positionAmt") or "0"
            side = cached_position.get("positionSide") or "BOTH"
            trade.real_state = "REAL_CONFLICT"
            trade.real_error = (
                f"Position Binance sudah ada pada {trade.pair} ({side}:{amount}). "
                "Entry baru tidak dibuat."
            )
            raise RealPositionConflictError(trade.real_error)

        async def prepare_and_place(force_refresh: bool = False) -> dict[str, Any]:
            if force_refresh:
                self.symbols = await self.rest.get_exchange_info(force=True)
            else:
                # REAL entry must use a fresh symbol-filter snapshot; stale
                # exchangeInfo is a known source of quantity/filter rejects.
                self.symbols = await self.rest.get_exchange_info(force=True)
            meta = self._get_symbol(trade.pair)

            await self.real.ensure_position_mode()
            existing_positions = await self.real.get_positions(trade.pair)
            if existing_positions:
                descriptions = ", ".join(
                    f"{item.get('positionSide', 'BOTH')}:{item.get('positionAmt', '0')}"
                    for item in existing_positions
                )
                trade.real_state = "REAL_CONFLICT"
                trade.real_error = (
                    f"Position Binance sudah ada pada {trade.pair} ({descriptions}). "
                    "Entry baru tidak dibuat agar exposure tidak bertambah."
                )
                raise RealPositionConflictError(trade.real_error)

            # IMPORTANT: leverage first. maxNotionalValue from the response is
            # part of the quantity calculation; availableBalance is also read
            # before placing the order.
            lev_response = await self.real.set_leverage(trade.pair, trade.leverage)
            max_notional = parse_max_notional(lev_response)
            balances = await self.real.get_futures_balance()
            usdt = next(
                (row for row in balances if str(row.get("asset") or "").upper() == "USDT"),
                None,
            )
            if usdt is None:
                raise BinanceAPIError(
                    "USDT balance tidak ditemukan saat menghitung real quantity.",
                    endpoint="/fapi/v3/balance",
                )
            available = parse_decimal(str(usdt.get("availableBalance") or "0"))

            quantity, target, actual = self._auto_quantity(
                meta,
                trade.entry,
                trade.margin_usdt,
                trade.leverage,
                available_balance=available,
                max_notional=max_notional,
            )

            position_side = self._position_side_for_direction(trade.direction)
            order = await self.real.place_limit_entry(
                symbol=trade.pair,
                side=trade.direction,
                quantity=quantity,
                price=trade.entry,
                position_side=position_side,
                client_order_id=client_id,
            )
            order["_quantity"] = quantity
            order["_target_notional"] = target
            order["_actual_notional"] = actual
            order["_position_side"] = position_side
            return order

        last_exc: Exception | None = None
        order: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                order = await prepare_and_place(force_refresh=attempt == 1)
                break
            except MarginInfluenceError as exc:
                last_exc = exc
                break
            except BinanceRateLimitError as exc:
                last_exc = exc
                self._notify_rate_limit(exc, f"PLACE LIMIT {trade.pair}")
                break
            except RealPositionConflictError:
                # Conflict sudah merupakan keputusan safety yang final; jangan
                # ubah state menjadi REAL_ERROR dan jangan kirim backend error.
                raise
            except BinanceAPIError as exc:
                last_exc = exc
                if exc.code == -4028:
                    last_exc = LeverageNotSupportedError(
                        f"{trade.pair}: leverage {trade.leverage}x tidak didukung Binance untuk pair ini."
                    )
                    break
                if is_qty_filter_error(exc) and attempt == 0:
                    log.warning(
                        "REAL quantity filter rejected for %s; refreshing exchangeInfo and recalculating once.",
                        trade.pair,
                    )
                    continue
                # Unknown execution after 5xx/transport: query by unique clientOrderId
                # before considering any retry. Never blindly duplicate the entry.
                if exc.status_code is None or exc.status_code >= 500:
                    try:
                        recovered = await self.real.get_order(
                            trade.pair,
                            client_order_id=client_id,
                        )
                        if recovered.get("orderId") is not None:
                            order = recovered
                            meta = self._get_symbol(trade.pair)
                            qty = parse_decimal(str(recovered.get("origQty") or "0"))
                            entry_price = parse_decimal(str(recovered.get("price") or trade.entry))
                            order["_quantity"] = qty
                            order["_target_notional"] = trade.margin_usdt * Decimal(trade.leverage)
                            order["_actual_notional"] = qty * entry_price
                            order["_position_side"] = self._position_side_for_direction(trade.direction)
                            break
                    except BinanceAPIError as lookup_exc:
                        if lookup_exc.code != -2013:
                            last_exc = lookup_exc
                            break
                break
            except Exception as exc:
                last_exc = exc
                break

        if order is None:
            trade.real_state = "REAL_ERROR"
            trade.real_error = str(last_exc or "Real entry gagal.")
            if isinstance(last_exc, MarginInfluenceError):
                log.warning(
                    "MARGIN INFLUENCE | pair=%s | trade_id=%s | %s",
                    trade.pair,
                    trade.trade_id,
                    last_exc,
                )
            elif isinstance(last_exc, LeverageNotSupportedError):
                log.info("LEVERAGE TIDAK DIDUKUNG | pair=%s | %s", trade.pair, last_exc)
            elif last_exc is not None:
                await self._handle_real_exception(last_exc, "PLACE LIMIT ENTRY", trade)
            raise last_exc or RuntimeError("Real entry gagal.")

        trade.quantity = order.get("_quantity")
        trade.target_notional = order.get("_target_notional")
        trade.actual_notional = order.get("_actual_notional")
        trade.real_enabled = True
        trade.real_state = "REAL_PENDING"
        trade.real_error = None
        trade.position_side = str(order.get("_position_side") or self._position_side_for_direction(trade.direction))
        try:
            trade.entry_order_id = int(order.get("orderId"))
        except (TypeError, ValueError):
            trade.entry_order_id = None
        trade.entry_client_order_id = str(order.get("clientOrderId") or client_id)

        if trade.entry_order_id is None and not trade.entry_client_order_id:
            raise BinanceAPIError(
                "Binance menerima response order tanpa orderId/clientOrderId.",
                endpoint="/fapi/v1/order",
            )

        status = str(order.get("status") or "NEW").upper()
        if status in {"FILLED", "PARTIALLY_FILLED"}:
            await self._confirm_real_fill(trade, verified_order=order)

    async def _confirm_real_fill(
        self,
        trade: Trade,
        verified_order: dict[str, Any] | None = None,
    ) -> bool:
        if not self.real_mode or not trade.real_enabled:
            return False

        if trade.entry_order_id is None and not trade.entry_client_order_id and verified_order is None:
            trade.real_state = "REAL_ERROR"
            trade.real_error = "REAL fill tidak dapat diverifikasi tanpa entry order ID/clientOrderId."
            log.warning(
                "REAL fill verification skipped: %s tidak memiliki entry order metadata.",
                trade.trade_id,
            )
            return False

        if verified_order is not None:
            order = dict(verified_order)
        else:
            try:
                order = await self.real.get_order(
                    trade.pair,
                    order_id=trade.entry_order_id,
                    client_order_id=(
                        trade.entry_client_order_id
                        if trade.entry_order_id is None
                        else None
                    ),
                )
            except BinanceAPIError as exc:
                if exc.code == -2013:
                    trade.real_state = "REAL_ERROR"
                    trade.real_error = str(exc)
                    log.warning(
                        "Entry order REAL %s tidak ditemukan saat fill verification.",
                        trade.trade_id,
                    )
                    return False
                raise

        order_status = str(order.get("status") or "").upper()
        if order_status not in {"FILLED", "PARTIALLY_FILLED"}:
            trade.real_state = "REAL_PENDING"
            trade.real_error = None
            return False

        position = await self.real.get_position(trade.pair, trade.direction)
        if position is None:
            # Exchange order is filled but position propagation may not be
            # visible yet. Do not invent local FILLED state.
            trade.real_state = "REAL_SYNCING"
            trade.real_error = (
                f"Entry order {order_status}, tetapi position belum terkonfirmasi."
            )
            return False

        position_amount = abs(
            parse_signed_decimal(
                str(position.get("_abs_position_amt") or position.get("positionAmt") or "0")
            )
        )
        if position_amount <= 0:
            return False

        actual_entry = parse_decimal(str(position.get("entryPrice") or trade.entry))
        trade.position_side = str(
            position.get("positionSide")
            or self._position_side_for_direction(trade.direction)
        )

        # For partial fills, cancel the remaining LIMIT quantity before
        # arming protective orders. Keep the already-verified order snapshot
        # because Binance may report it as CANCELED immediately after this call.
        if order_status == "PARTIALLY_FILLED" and trade.entry_order_id is not None:
            executed_qty = parse_decimal(str(order.get("executedQty") or position_amount))
            orig_qty = parse_decimal(str(order.get("origQty") or executed_qty))
            if executed_qty < orig_qty:
                try:
                    await self.real.cancel_order(
                        trade.pair,
                        order_id=trade.entry_order_id,
                    )
                except BinanceAPIError as exc:
                    if exc.code not in {-2011, -2013}:
                        raise

        trade.quantity = position_amount
        trade.actual_notional = position_amount * actual_entry
        trade.fill_price = actual_entry
        trade.filled_at = trade.filled_at or now_utc()
        trade.pnl_percent = Decimal("0")
        trade.real_state = "REAL_FILLED"
        trade.real_error = None
        trade.status = "FILLED"

        # Notifikasi dulu (cepat), proteksi TP/SL tetap diprioritaskan sebelum pencatatan.
        await self._notify_filled(trade)
        await self._protect_or_close(trade)
        try:
            await self._record_event(
                trade,
                "FILLED",
                event_price=actual_entry,
                reason=trade.entry_reason,
                extra={"fill_price": decimal_to_str(actual_entry)},
            )
        except Exception:
            log.exception("Gagal mencatat event FILLED REAL %s", trade.trade_id)
        return True

    def _clear_pending_sl(self, trade: Trade) -> None:
        trade.pending_sl_algo_id = None
        trade.pending_sl_client_algo_id = None
        trade.pending_sl_price = None
        trade.pending_sl_state = "NONE"
        trade.pending_sl_created_at = None
        trade.pending_sl_error = None

    @staticmethod
    def _is_open_algo_status(status: str) -> bool:
        return status.upper() in {"NEW", "PENDING", "WORKING", "ACCEPTED"}

    def _find_owned_sl_orders(
        self,
        trade: Trade,
        open_algos: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prefix = f"SL-{trade.trade_id}"
        tracked = {
            str(value)
            for value in (
                trade.sl_algo_id,
                trade.sl_client_algo_id,
                trade.pending_sl_algo_id,
                trade.pending_sl_client_algo_id,
            )
            if value not in (None, "")
        }
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in open_algos:
            client_id = str(item.get("clientAlgoId") or "")
            algo_id = str(item.get("algoId") or "")
            owned = (
                client_id in tracked
                or algo_id in tracked
                or client_id == prefix
                or client_id.startswith(prefix + "-")
            )
            if not owned:
                continue
            marker = algo_id or client_id
            if marker and marker in seen:
                continue
            seen.add(marker)
            result.append(item)
        return result

    def _find_specific_algo(
        self,
        open_algos: list[dict[str, Any]],
        *,
        algo_id: int | None = None,
        client_algo_id: str | None = None,
    ) -> dict[str, Any] | None:
        wanted_id = str(algo_id) if algo_id is not None else None
        wanted_client = str(client_algo_id or "") or None
        for item in open_algos:
            if wanted_id is not None and str(item.get("algoId") or "") == wanted_id:
                return item
            if wanted_client and str(item.get("clientAlgoId") or "") == wanted_client:
                return item
        return None

    async def _confirm_sl_algo_active(
        self,
        trade: Trade,
        *,
        algo_id: int | None,
        client_algo_id: str | None,
        expected_price: Decimal,
        attempts: int = 3,
    ) -> dict[str, Any] | None:
        """Confirm a specific SL is actually present in Binance open-algo state."""
        for attempt in range(max(1, attempts)):
            open_algos = await self.real.get_open_algo_orders(trade.pair)
            item = self._find_specific_algo(
                open_algos,
                algo_id=algo_id,
                client_algo_id=client_algo_id,
            )
            if item is not None:
                status = str(item.get("algoStatus") or "NEW").upper()
                raw_trigger = item.get("triggerPrice")
                try:
                    trigger = Decimal(str(raw_trigger)) if raw_trigger not in (None, "") else None
                except InvalidOperation:
                    trigger = None
                if self._is_open_algo_status(status) and trigger is not None and trigger == expected_price:
                    return item
            if attempt + 1 < max(1, attempts):
                await asyncio.sleep(0.35)
        return None

    async def _cancel_specific_algo_confirmed(
        self,
        trade: Trade,
        *,
        algo_id: int | None,
        client_algo_id: str | None,
        attempts: int = 3,
    ) -> bool:
        if algo_id is None and not client_algo_id:
            return True

        last_exists: bool = True
        for attempt in range(max(1, attempts)):
            open_algos = await self.real.get_open_algo_orders(trade.pair)
            item = self._find_specific_algo(
                open_algos,
                algo_id=algo_id,
                client_algo_id=client_algo_id,
            )
            if item is None:
                return True
            last_exists = True
            current_id = item.get("algoId")
            current_client = str(item.get("clientAlgoId") or "") or None
            try:
                await self.real.cancel_algo_order(
                    symbol=trade.pair,
                    algo_id=(int(current_id) if current_id not in (None, "") else None),
                    client_algo_id=(current_client if current_id in (None, "") else None),
                )
            except BinanceAPIError as exc:
                if exc.code not in {-2011, -2013}:
                    raise
            if attempt + 1 < max(1, attempts):
                await asyncio.sleep(0.35)
        # One final read after the last cancel request.
        open_algos = await self.real.get_open_algo_orders(trade.pair)
        last_exists = (
            self._find_specific_algo(
                open_algos,
                algo_id=algo_id,
                client_algo_id=client_algo_id,
            )
            is not None
        )
        return not last_exists

    async def _resume_sl_transition(self, trade: Trade) -> bool:
        """Resume/finish an in-flight SL replacement without ever discarding the active SL prematurely."""
        if not self.real_mode or not trade.real_enabled or trade.status != "FILLED":
            return False
        if trade.pending_sl_state == "NONE" or trade.pending_sl_price is None:
            return False

        position = await self.real.get_position(trade.pair, trade.direction)
        if position is None:
            return False

        if trade.trade_id in self._sl_transition_busy:
            return False
        self._sl_transition_busy.add(trade.trade_id)
        try:
            target = trade.pending_sl_price
            pending_id = trade.pending_sl_algo_id
            pending_client = trade.pending_sl_client_algo_id
            old_id = trade.sl_algo_id
            old_client = trade.sl_client_algo_id

            # First recover a possibly-created candidate by clientAlgoId. This
            # prevents duplicate placement when the POST response was lost.
            if pending_id is None and pending_client:
                open_algos = await self.real.get_open_algo_orders(trade.pair)
                recovered = self._find_specific_algo(
                    open_algos,
                    client_algo_id=pending_client,
                )
                if recovered is not None:
                    raw_recovered_id = recovered.get("algoId")
                    try:
                        trade.pending_sl_algo_id = int(raw_recovered_id) if raw_recovered_id not in (None, "") else None
                    except (TypeError, ValueError):
                        trade.pending_sl_algo_id = None
                    pending_id = trade.pending_sl_algo_id

            # If the candidate still was not created, create it now.
            if pending_id is None:
                pending_client = (
                    pending_client
                    or f"{self._client_id('SL', trade.trade_id)[:29]}-{uuid4().hex[:6]}"
                )
                trade.pending_sl_client_algo_id = pending_client
                trade.pending_sl_created_at = trade.pending_sl_created_at or now_utc()
                trade.pending_sl_state = "PLACING"
                trade.pending_sl_error = None
                try:
                    order = await self.real.place_close_algo(
                        symbol=trade.pair,
                        side=("SELL" if trade.direction == "BUY" else "BUY"),
                        position_side=str(position.get("positionSide") or self._position_side_for_direction(trade.direction)),
                        order_type="STOP_MARKET",
                        trigger_price=target,
                        client_algo_id=pending_client,
                    )
                except Exception as exc:
                    trade.pending_sl_state = (
                        "RATE_LIMIT_WAIT" if isinstance(exc, BinanceRateLimitError) else "PLACE_FAILED"
                    )
                    trade.pending_sl_error = str(exc)[:500]
                    if isinstance(exc, BinanceRateLimitError):
                        self._notify_rate_limit(exc, "PLACE TRAILING SL")
                        self._start_rate_limit_watcher()
                    # IMPORTANT: active SL metadata and Binance order are untouched.
                    return False
                trade.pending_sl_client_algo_id = str(order.get("clientAlgoId") or pending_client)
                raw_algo_id = order.get("algoId")
                try:
                    trade.pending_sl_algo_id = int(raw_algo_id) if raw_algo_id not in (None, "") else None
                except (TypeError, ValueError):
                    trade.pending_sl_algo_id = None
                pending_id = trade.pending_sl_algo_id
                pending_client = trade.pending_sl_client_algo_id

            # Resolve the old SL early when its exact metadata is missing.
            # This lets us distinguish "new not yet visible, old still safe"
            # from the rare case where neither protection exists.
            open_algos = await self.real.get_open_algo_orders(trade.pair)
            if old_id is None and not old_client:
                owned = self._find_owned_sl_orders(trade, open_algos)
                other_sl = [
                    item
                    for item in owned
                    if not self._find_specific_algo(
                        [item],
                        algo_id=pending_id,
                        client_algo_id=pending_client,
                    )
                ]
                if len(other_sl) == 1:
                    old_item = other_sl[0]
                    try:
                        old_id = int(old_item.get("algoId")) if old_item.get("algoId") not in (None, "") else None
                    except (TypeError, ValueError):
                        old_id = None
                    old_client = str(old_item.get("clientAlgoId") or "") or None

            trade.pending_sl_state = "CONFIRMING_NEW"
            confirmed_new = await self._confirm_sl_algo_active(
                trade,
                algo_id=pending_id,
                client_algo_id=pending_client,
                expected_price=target,
            )
            if confirmed_new is None:
                trade.pending_sl_state = "NEW_UNCONFIRMED"
                trade.pending_sl_error = "SL baru belum dapat dikonfirmasi sebagai open algo order di Binance."
                # Critical invariant: if old SL is still visible, keep it and
                # retry. If old SL is also gone while the position remains open,
                # there is no longer a safe protection layer; close immediately.
                latest_open_algos = await self.real.get_open_algo_orders(trade.pair)
                old_still_open = self._find_specific_algo(
                    latest_open_algos,
                    algo_id=old_id,
                    client_algo_id=old_client,
                )
                if old_still_open is None:
                    position_after = await self.real.get_position(trade.pair, trade.direction)
                    if position_after is not None:
                        trade.pending_sl_state = "NO_PROTECTION"
                        trade.pending_sl_error = "SL lama dan SL baru tidak sama-sama terkonfirmasi sementara posisi masih terbuka."
                        await self._emergency_close(
                            trade,
                            f"Tidak ada SL terkonfirmasi untuk {trade.pair} saat replacement trailing.",
                        )
                # Never cancel the old SL when it is still present.
                return False

            trade.pending_sl_state = "NEW_CONFIRMED"
            trade.pending_sl_error = None

            # There is no old SL to remove only when metadata is absent and
            # Binance also reports no other bot-owned SL besides the new one.
            open_algos = await self.real.get_open_algo_orders(trade.pair)
            old_item = self._find_specific_algo(
                open_algos,
                algo_id=old_id,
                client_algo_id=old_client,
            )
            if old_item is None and (old_id is not None or old_client):
                # Metadata says an old SL exists, yet it is no longer open.
                # Safe to promote the already-confirmed candidate.
                old_id = None
                old_client = None

            if old_id is None and not old_client:
                owned = self._find_owned_sl_orders(trade, open_algos)
                other_sl = [
                    item
                    for item in owned
                    if not self._find_specific_algo(
                        [item],
                        algo_id=pending_id,
                        client_algo_id=pending_client,
                    )
                ]
                if len(other_sl) == 1:
                    old_item = other_sl[0]
                    try:
                        old_id = int(old_item.get("algoId")) if old_item.get("algoId") not in (None, "") else None
                    except (TypeError, ValueError):
                        old_id = None
                    old_client = str(old_item.get("clientAlgoId") or "") or None
                elif len(other_sl) > 1:
                    trade.pending_sl_state = "AMBIGUOUS_OLD_SL"
                    trade.pending_sl_error = "Lebih dari satu SL lama terdeteksi; bot tidak akan menghapus order secara membabi buta."
                    return False

            if old_id is not None or old_client:
                trade.pending_sl_state = "CANCELING_OLD"
                old_canceled = await self._cancel_specific_algo_confirmed(
                    trade,
                    algo_id=old_id,
                    client_algo_id=old_client,
                )
                if not old_canceled:
                    trade.pending_sl_state = "WAITING_OLD_CANCEL"
                    trade.pending_sl_error = "SL baru sudah confirmed, namun SL lama belum berhasil dihapus."
                    # Both protections may temporarily coexist; retry later.
                    return False

            # Final invariant: NEW is still open and OLD is gone.
            trade.pending_sl_state = "FINAL_CONFIRM"
            final_new = await self._confirm_sl_algo_active(
                trade,
                algo_id=pending_id,
                client_algo_id=pending_client,
                expected_price=target,
                attempts=2,
            )
            open_algos = await self.real.get_open_algo_orders(trade.pair)
            final_old = self._find_specific_algo(
                open_algos,
                algo_id=old_id,
                client_algo_id=old_client,
            )
            if final_new is None:
                position_after = await self.real.get_position(trade.pair, trade.direction)
                if position_after is not None:
                    raise RuntimeError(
                        f"SL baru {decimal_to_str(target)} hilang setelah SL lama dihapus sementara posisi masih terbuka."
                    )
                return False
            if final_old is not None:
                trade.pending_sl_state = "WAITING_OLD_CANCEL"
                trade.pending_sl_error = "Verifikasi akhir masih menemukan SL lama di Binance."
                return False

            # COMMIT POINT. Only here does pending become active local state.
            trade.sl = target
            trade.sl_algo_id = pending_id
            trade.sl_client_algo_id = pending_client
            self._clear_pending_sl(trade)
            trade.real_error = None
            return True
        finally:
            self._sl_transition_busy.discard(trade.trade_id)

    async def _ensure_real_protective_orders(self, trade: Trade) -> None:
        if not self.real_mode or not trade.real_enabled:
            return

        position = await self.real.get_position(trade.pair, trade.direction)
        if position is None:
            return

        # Complete any interrupted two-phase SL transition first. If it cannot
        # be completed, the existing active SL remains the protection of record.
        if trade.pending_sl_state != "NONE" and trade.pending_sl_price is not None:
            await self._resume_sl_transition(trade)

        position_side = str(
            position.get("positionSide")
            or self._position_side_for_direction(trade.direction)
        )
        exit_side = "SELL" if trade.direction == "BUY" else "BUY"

        open_algos = await self.real.get_open_algo_orders(trade.pair)
        own_by_client = {
            str(item.get("clientAlgoId") or ""): item
            for item in open_algos
        }
        own_by_id = {
            str(item.get("algoId") or ""): item
            for item in open_algos
        }

        def find_existing(kind: str, tracked_id: int | None, tracked_client: str | None) -> dict[str, Any] | None:
            if tracked_client and tracked_client in own_by_client:
                return own_by_client[tracked_client]
            if tracked_id is not None and str(tracked_id) in own_by_id:
                return own_by_id[str(tracked_id)]

            prefix = f"{kind}-{trade.trade_id}-"
            for item in open_algos:
                client_id = str(item.get("clientAlgoId") or "")
                if client_id == f"{kind}-{trade.trade_id}" or client_id.startswith(prefix):
                    return item
            return None

        # If a replacement is still pending, do not create another SL while
        # the old one is still the active local protection.
        if trade.pending_sl_state != "NONE":
            existing_sl = find_existing("SL", trade.sl_algo_id, trade.sl_client_algo_id)
            if existing_sl is None and trade.pending_sl_algo_id is None and trade.pending_sl_client_algo_id:
                # _resume_sl_transition may have failed before POST; leave the
                # pending state intact so periodic reconcile can retry.
                pass
        else:
            existing_sl = find_existing("SL", trade.sl_algo_id, trade.sl_client_algo_id)
            if existing_sl is not None:
                trade.sl_client_algo_id = str(existing_sl.get("clientAlgoId") or trade.sl_client_algo_id or "") or None
                try:
                    trade.sl_algo_id = int(existing_sl.get("algoId"))
                except (TypeError, ValueError):
                    pass
            else:
                sl_client = trade.sl_client_algo_id or self._client_id("SL", trade.trade_id)
                sl_order = await self.real.place_close_algo(
                    symbol=trade.pair,
                    side=exit_side,
                    position_side=position_side,
                    order_type="STOP_MARKET",
                    trigger_price=trade.sl,
                    client_algo_id=sl_client,
                )
                trade.sl_client_algo_id = str(sl_order.get("clientAlgoId") or sl_client)
                try:
                    trade.sl_algo_id = int(sl_order.get("algoId"))
                except (TypeError, ValueError):
                    trade.sl_algo_id = None
                confirmed = await self._confirm_sl_algo_active(
                    trade,
                    algo_id=trade.sl_algo_id,
                    client_algo_id=trade.sl_client_algo_id,
                    expected_price=trade.sl,
                    attempts=3,
                )
                if confirmed is None:
                    raise RuntimeError(
                        f"SL awal {decimal_to_str(trade.sl)} berhasil dikirim namun belum confirmed aktif."
                    )

        existing_tp = find_existing("TP", trade.tp_algo_id, trade.tp_client_algo_id)
        if existing_tp is not None:
            trade.tp_client_algo_id = str(existing_tp.get("clientAlgoId") or trade.tp_client_algo_id or "") or None
            try:
                trade.tp_algo_id = int(existing_tp.get("algoId"))
            except (TypeError, ValueError):
                pass
        else:
            tp_client = trade.tp_client_algo_id or self._client_id("TP", trade.trade_id)
            tp_order = await self.real.place_close_algo(
                symbol=trade.pair,
                side=exit_side,
                position_side=position_side,
                order_type="TAKE_PROFIT_MARKET",
                trigger_price=trade.tp,
                client_algo_id=tp_client,
            )
            trade.tp_client_algo_id = str(tp_order.get("clientAlgoId") or tp_client)
            try:
                trade.tp_algo_id = int(tp_order.get("algoId"))
            except (TypeError, ValueError):
                trade.tp_algo_id = None

    async def _real_pending_price_exp(self, trade: Trade) -> None:
        """Reconcile the real entry when Price Exp is reached.

        Rule: if REAL is ON, inspect the bot's own entry LIMIT. If it exists
        and is still open, cancel it. If there is no such LIMIT, normal local
        EXPIRED processing is allowed to continue. A confirmed real position
        always wins over local expiry and is promoted to FILLED first.
        """
        # A position that was already created by this setup must be treated as
        # a real fill even if the LIMIT order is no longer open.
        if trade.entry_order_id is not None or trade.entry_client_order_id:
            try:
                position = await self.real.get_position(trade.pair, trade.direction)
                if position is not None:
                    if await self._confirm_real_fill(trade):
                        return
            except BinanceRateLimitError:
                raise
            except BinanceAPIError as exc:
                # Do not swallow real API failures; caller will report them to Telegram.
                if exc.code != -2013:
                    raise

        order: dict[str, Any] | None = None

        if trade.entry_order_id is not None or trade.entry_client_order_id:
            try:
                order = await self.real.get_order(
                    trade.pair,
                    order_id=trade.entry_order_id,
                    client_order_id=(
                        trade.entry_client_order_id
                        if trade.entry_order_id is None
                        else None
                    ),
                )
            except BinanceAPIError as exc:
                if exc.code != -2013:
                    raise
                # Missing bot order means there is no known LIMIT to cancel.
                # Continue with the normal local EXPIRED decision.
                order = None

        # If metadata is missing, try to locate only this bot's entry LIMIT
        # by its deterministic clientOrderId. Never cancel an unrelated order.
        if order is None:
            expected_client = trade.entry_client_order_id or self._client_id("ENT", trade.trade_id)
            open_orders = await self.real.get_open_orders(trade.pair)
            for candidate in open_orders:
                candidate_type = str(candidate.get("type") or "").upper()
                candidate_id = str(candidate.get("clientOrderId") or "")
                candidate_order_id = candidate.get("orderId")
                id_match = (
                    bool(expected_client) and candidate_id == expected_client
                )
                order_id_match = (
                    trade.entry_order_id is not None
                    and candidate_order_id is not None
                    and str(candidate_order_id) == str(trade.entry_order_id)
                )
                if candidate_type == "LIMIT" and (id_match or order_id_match):
                    order = candidate
                    break

        if order is not None:
            status = str(order.get("status") or "").upper()

            if status in {"FILLED", "PARTIALLY_FILLED"}:
                if await self._confirm_real_fill(trade):
                    return

            if status in {"NEW", "PARTIALLY_FILLED"}:
                order_id = order.get("orderId")
                try:
                    if order_id is not None:
                        await self.real.cancel_order(
                            trade.pair,
                            order_id=int(order_id),
                        )
                    elif trade.entry_client_order_id:
                        await self.real.cancel_order(
                            trade.pair,
                            client_order_id=trade.entry_client_order_id,
                        )
                except BinanceAPIError as exc:
                    if exc.code not in {-2011, -2013}:
                        raise

                # A partial fill can leave a live position after cancel.
                if status == "PARTIALLY_FILLED" and await self._confirm_real_fill(
                    trade,
                    verified_order=order,
                ):
                    return

        # No confirmed live position and no still-open bot entry LIMIT. Let
        # the normal PENDING -> EXPIRED transition happen in _on_price().
        trade.real_state = "REAL_CLOSED"
        trade.real_error = None

    async def _cancel_bot_algo_orders(self, trade: Trade, only: str | None = None) -> None:
        """Cancel tracked bot algo orders. SL replacement uses specific IDs separately."""
        open_algos = await self.real.get_open_algo_orders(trade.pair)
        kinds = ("TP", "SL") if only is None else (only,)
        ids = {"TP": trade.tp_algo_id, "SL": trade.sl_algo_id}
        clients = {"TP": trade.tp_client_algo_id, "SL": trade.sl_client_algo_id}
        tracked_ids = {str(ids[k]) for k in kinds if ids[k] is not None}
        tracked_clients = {str(clients[k]) for k in kinds if clients[k]}
        prefixes = tuple(f"{k}-{trade.trade_id}" for k in kinds)

        for item in open_algos:
            algo_id = str(item.get("algoId") or "")
            client_id = str(item.get("clientAlgoId") or "")
            owned = (
                algo_id in tracked_ids
                or client_id in tracked_clients
                or any(client_id == p or client_id.startswith(p + "-") for p in prefixes)
            )
            if not owned:
                continue

            try:
                if algo_id:
                    await self.real.cancel_algo_order(
                        symbol=trade.pair,
                        algo_id=int(algo_id),
                    )
                elif client_id:
                    await self.real.cancel_algo_order(
                        symbol=trade.pair,
                        client_algo_id=client_id,
                    )
            except BinanceAPIError as exc:
                if exc.code not in {-2011, -2013}:
                    raise

    async def _cleanup_real_leftovers(self, trade: Trade) -> None:
        """OCO manual: setelah salah satu TP/SL tercapai, hapus algo yang tersisa."""
        if not self.real_mode:
            return
        if self.real.cooldown_remaining > 0:
            self._deferred_cleanup[trade.trade_id] = trade
            return
        try:
            await self._cancel_bot_algo_orders(trade)
            self._deferred_cleanup.pop(trade.trade_id, None)
        except BinanceRateLimitError as exc:
            self._deferred_cleanup[trade.trade_id] = trade
            self._notify_rate_limit(exc, "HAPUS ORDER SISA")
        except Exception as exc:
            log.warning("Hapus algo sisa %s gagal: %s", trade.trade_id, exc)

    def _real_poll_due(self, trade: Trade, kind: str, interval: float) -> bool:
        """Throttle polling REST Binance per trade agar tidak memicu ban."""
        key = f"{kind}:{trade.trade_id}"
        now = time.monotonic()
        if now - self._real_poll_guard.get(key, 0.0) < interval:
            return False
        self._real_poll_guard[key] = now
        return True

    async def _protect_or_close(self, trade: Trade) -> bool:
        """Pasang TP/SL (retry 3x). Saat API dibatasi, antre; gagal terus -> close paksa."""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                await self._ensure_real_protective_orders(trade)
                self._deferred_protect.pop(trade.trade_id, None)
                return True
            except BinanceRateLimitError as exc:
                # Antre; watcher memasang otomatis begitu jeda server selesai.
                self._deferred_protect.setdefault(trade.trade_id, time.monotonic())
                await self._handle_real_exception(exc, "PLACE TP/SL", trade)
                return False
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "TP/SL %s gagal dipasang (%s/3): %s",
                    trade.trade_id,
                    attempt + 1,
                    exc,
                )
                if attempt < 2:
                    await asyncio.sleep(2.0)
        await self._emergency_close(trade, f"TP/SL gagal dipasang 3x: {last_exc}")
        self._deferred_protect.pop(trade.trade_id, None)
        return False

    def _deferred_any(self) -> bool:
        return bool(
            self._deferred_fill or self._deferred_protect or self._deferred_cleanup or self._deferred_trail
        )

    def _defer_fill_check(self, trade: Trade, price: Decimal) -> None:
        """Harga menyentuh entry saat API dibatasi: antre konfirmasi fill + TP/SL."""
        if trade.trade_id in self._deferred_fill:
            return
        self._deferred_fill[trade.trade_id] = time.monotonic()
        icon = "🟢" if trade.direction == "BUY" else "🔴"
        self._fire(
            self.reply(
                card(
                    "⚠️ KEMUNGKINAN FILLED",
                    [
                        f"{icon} {trade.pair}  •  {trade.direction}",
                        f"📡 Harga {fmt_price(price)} menyentuh entry {fmt_price(trade.entry)}",
                        "⏳ Konfirmasi order + TP/SL ditunda (Binance API limit)",
                        f"🛡 SL {fmt_price(trade.sl)}  •  🏁 TP {fmt_price(trade.tp)}",
                        "▶️ Dipasang otomatis begitu API pulih",
                    ],
                )
            )
        )

    async def _flush_deferred(self) -> None:
        """Konfirmasi fill + pasang TP/SL SEMUA trade yang antre (stack), tanpa jeda aman bot."""
        if not self.real_mode:
            self._deferred_fill.clear()
            self._deferred_protect.clear()
            self._deferred_cleanup.clear()
            self._deferred_trail.clear()
            return
        protected: list[str] = []
        for trade_id in list(dict.fromkeys([*self._deferred_protect, *self._deferred_fill])):
            trade = self.active_trades.get(trade_id)
            if trade is None or trade.status not in {"PENDING", "FILLED"}:
                self._deferred_fill.pop(trade_id, None)
                self._deferred_protect.pop(trade_id, None)
                continue
            if self.real.server_cooldown_remaining > 0:
                break
            token = REAL_PRIORITY.set(True)
            try:
                if trade.status == "PENDING":
                    await self._confirm_real_fill(trade)
                else:
                    await self._protect_or_close(trade)
                self._deferred_fill.pop(trade_id, None)
                if trade.status == "FILLED" and trade_id not in self._deferred_protect:
                    protected.append(trade.pair)
            except BinanceRateLimitError as exc:
                self._notify_rate_limit(exc, "PROTEKSI TERTUNDA")
                break
            except Exception as exc:
                log.warning("[RATE LIMIT] proteksi tertunda %s gagal: %s", trade_id, exc)
                self._deferred_fill.pop(trade_id, None)
                self._deferred_protect.pop(trade_id, None)
            finally:
                REAL_PRIORITY.reset(token)
            await asyncio.sleep(0.3)
        for trade_id, old_trade in list(self._deferred_cleanup.items()):
            if self.real.server_cooldown_remaining > 0:
                break
            token = REAL_PRIORITY.set(True)
            try:
                await self._cancel_bot_algo_orders(old_trade)
                self._deferred_cleanup.pop(trade_id, None)
            except BinanceRateLimitError as exc:
                self._notify_rate_limit(exc, "HAPUS ORDER SISA")
                break
            except Exception as exc:
                log.warning("Hapus algo sisa %s gagal: %s", trade_id, exc)
                self._deferred_cleanup.pop(trade_id, None)
            finally:
                REAL_PRIORITY.reset(token)
            await asyncio.sleep(0.3)
        for trade_id, item in list(self._deferred_trail.items()):
            trade = self.active_trades.get(trade_id)
            if trade is None or trade.status != "FILLED":
                self._deferred_trail.pop(trade_id, None)
                continue
            if self.real.server_cooldown_remaining > 0:
                break
            snapshot = self.prices.get(trade.pair)
            price = snapshot.price if snapshot is not None else None
            token = REAL_PRIORITY.set(True)
            try:
                await self._apply_deferred_trail(trade, item, price)
                self._deferred_trail.pop(trade_id, None)
            except BinanceRateLimitError as exc:
                self._notify_rate_limit(exc, "TRAILING TERTUNDA")
                break
            except Exception as exc:
                log.warning("[TRAIL] trailing tertunda %s gagal: %s", trade_id, exc)
                self._deferred_trail.pop(trade_id, None)
            finally:
                REAL_PRIORITY.reset(token)
            await asyncio.sleep(0.3)
        if protected:
            await self.reply(
                card(
                    "🛡️ PROTEKSI TERTUNDA TERPASANG",
                    [
                        f"✅ {len(protected)} posisi: {', '.join(protected[:10])}",
                        "TP/SL aktif di Binance.",
                    ],
                )
            )

    async def _verify_real_protection(self, trade: Trade) -> None:
        """Pastikan posisi REAL selalu punya TP/SL; gagal berulang -> close paksa."""
        if self.real.cooldown_remaining > 0:
            return
        try:
            await self._reconcile_real_trade(trade)
            self._real_protect_fail.pop(trade.trade_id, None)
        except BinanceRateLimitError as exc:
            self._notify_rate_limit(exc, "VERIFY TP/SL")
        except Exception as exc:
            count = self._real_protect_fail.get(trade.trade_id, 0) + 1
            self._real_protect_fail[trade.trade_id] = count
            log.warning("Verifikasi TP/SL %s gagal (%s/3): %s", trade.trade_id, count, exc)
            if count >= 3 and trade.status == "FILLED":
                await self._emergency_close(trade, f"TP/SL tidak bisa dipastikan aktif: {exc}")

    async def _emergency_close(self, trade: Trade, reason: str) -> None:
        """Market close paksa posisi REAL tanpa proteksi, lalu catat hasilnya."""
        log.error("EMERGENCY CLOSE %s: %s", trade.trade_id, reason)
        exit_price: Decimal | None = None
        try:
            position = await self.real.get_position(trade.pair, trade.direction)
            if position is not None:
                response = await self.real.place_market_close(
                    symbol=trade.pair,
                    position=position,
                    entry_direction=trade.direction,
                )
                for _ in range(3):
                    await asyncio.sleep(0.25)
                    if await self.real.get_position(trade.pair, trade.direction) is None:
                        break
                for key in ("avgPrice", "price"):
                    try:
                        exit_price = parse_decimal(str(response.get(key) or "0"))
                        break
                    except (InvalidOperation, ValueError):
                        continue
        except Exception as exc:
            await self._handle_real_exception(exc, "EMERGENCY CLOSE", trade)
            await self.reply(
                f"🚨 EMERGENCY CLOSE GAGAL {trade.pair}\n\n"
                "Posisi mungkin TANPA TP/SL. Tutup manual di Binance!\n"
                f"{exc}"
            )
            return

        try:
            await self._cancel_bot_algo_orders(trade)
        except Exception as exc:
            log.warning("Cancel algo setelah emergency close %s gagal: %s", trade.trade_id, exc)

        if exit_price is None:
            snapshot = self.prices.get(trade.pair)
            exit_price = snapshot.price if snapshot is not None else (trade.fill_price or trade.entry)

        pnl = pct_change(trade.direction, trade.fill_price or trade.entry, exit_price)
        if trade.trailing and pnl >= 0:
            result = "TRAIL"
        else:
            result = "TP" if pnl > 0 else "SL" if pnl < 0 else "MANUAL_CLOSE"
        trade.real_state = "REAL_CLOSED"
        if trade.trade_id not in self.active_trades:
            self.active_trades[trade.trade_id] = trade
        await self.reply(
            f"{'🔒 TRAILING TERTUNDA' if result == 'TRAIL' else '🚨 EMERGENCY CLOSE'} {trade.pair}\n\n"
            f"{reason}\n"
            f"Posisi ditutup market ({result}, PnL {format_pct(pnl)})."
        )
        await self._finalize_trade(
            trade,
            result=result,
            exit_price=exit_price,
            reason=f"Emergency close: {reason}",
        )

    async def _real_exit_trigger(self, trade: Trade, result: str, price: Decimal) -> bool:
        now = time.monotonic()
        last = self._real_reconcile_guard.get(trade.trade_id, 0.0)
        if now - last < REAL_RECONCILE_INTERVAL_SECONDS:
            return False
        self._real_reconcile_guard[trade.trade_id] = now

        trade.real_exit_check = result
        position = await self.real.get_position(trade.pair, trade.direction)
        if position is not None:
            return False

        # Position is zero. Try to read the triggered algo's actual execution
        # price before cleaning the remaining protection.
        triggered_algo_id = (
            trade.tp_algo_id if result == "TP" else trade.sl_algo_id
        )
        if triggered_algo_id is not None:
            try:
                algo = await self.real.get_algo_order(algo_id=triggered_algo_id)
                status = str(algo.get("algoStatus") or "").upper()
                if status in {"TRIGGERED", "FINISHED"}:
                    actual_raw = algo.get("actualPrice") or algo.get("triggerPrice")
                    if actual_raw not in (None, "", "0", 0):
                        trade.exit_price = parse_decimal(str(actual_raw))
            except BinanceAPIError as exc:
                if exc.code not in {-2013}:
                    raise

        if trade.exit_price is None:
            trade.exit_price = price
        return True

    async def _real_delete_cleanup(self, trade: Trade) -> None:
        # Cancel entry order if it is still open.
        if trade.entry_order_id is not None or trade.entry_client_order_id:
            try:
                order = await self.real.get_order(
                    trade.pair,
                    order_id=trade.entry_order_id,
                    client_order_id=trade.entry_client_order_id if trade.entry_order_id is None else None,
                )
                if str(order.get("status") or "") in {"NEW", "PARTIALLY_FILLED"}:
                    await self.real.cancel_order(
                        trade.pair,
                        order_id=int(order["orderId"]) if order.get("orderId") is not None else None,
                        client_order_id=None,
                    )
            except BinanceAPIError as exc:
                if exc.code not in {-2013, -2011}:
                    raise

        # Explicit /del requirement: clear all algo orders on this symbol.
        await self.real.cancel_all_algo_orders(trade.pair)

        # Close any remaining position.
        position = await self.real.get_position(trade.pair, trade.direction)
        if position is not None:
            await self.real.place_market_close(
                symbol=trade.pair,
                position=position,
                entry_direction=trade.direction,
            )

            # One immediate verification + one short reconciliation retry.
            for _ in range(2):
                await asyncio.sleep(0.25)
                position = await self.real.get_position(trade.pair, trade.direction)
                if position is None:
                    break

            if position is not None:
                raise BinanceAPIError(
                    f"Position {trade.pair} masih terbuka setelah /del.",
                    endpoint="/fapi/v2/positionRisk",
                )

        trade.real_state = "REAL_CLOSED"

    async def _replace_real_sl(self, trade: Trade, new_sl: Decimal) -> bool:
        """Two-phase SL replacement: PLACE NEW -> CONFIRM NEW -> CANCEL OLD -> VERIFY BOTH.

        The active SL metadata is never mutated until the complete transaction
        succeeds. Any exception before NEW is confirmed leaves the old SL alive.
        """
        if not self.real_mode or not trade.real_enabled or trade.status != "FILLED":
            return False

        if trade.pending_sl_state != "NONE" and trade.pending_sl_price is not None:
            # Resume the existing transition rather than creating a second candidate.
            if trade.pending_sl_price != new_sl:
                buy = trade.direction == "BUY"
                better = new_sl > trade.pending_sl_price if buy else new_sl < trade.pending_sl_price
                if better:
                    trade.pending_sl_price = new_sl
                    trade.pending_sl_error = None
                else:
                    new_sl = trade.pending_sl_price
            return await self._resume_sl_transition(trade)

        trade.pending_sl_algo_id = None
        trade.pending_sl_client_algo_id = None
        trade.pending_sl_price = new_sl
        trade.pending_sl_state = "PLACING"
        trade.pending_sl_created_at = now_utc()
        trade.pending_sl_error = None

        try:
            return await self._resume_sl_transition(trade)
        except BinanceRateLimitError as exc:
            # Old SL is untouched. Leave pending state for periodic reconcile.
            trade.pending_sl_state = "RATE_LIMIT_WAIT"
            trade.pending_sl_error = str(exc)[:500]
            self._notify_rate_limit(exc, "REPLACE TRAIL SL")
            return False
        except Exception as exc:
            # The transition may have reached a later state; never clear pending
            # metadata here. Reconcile can safely resume it.
            trade.pending_sl_error = str(exc)[:500]
            if trade.pending_sl_state == "NONE":
                trade.pending_sl_state = "ERROR"
            log.warning(
                "[TRAIL] transactional SL replacement %s gagal pada state=%s: %s",
                trade.trade_id,
                trade.pending_sl_state,
                exc,
            )
            return False

    def _trail_params(self) -> tuple[list[tuple[float, float]], Decimal]:
        """Tangga R dan buffer fee: strategy.py (otak) lebih utama, main.py sebagai cadangan."""
        module = getattr(self, "_strategy_runtime_module", None)
        ladder = getattr(module, "TRAIL_R_LADDER", None) or TRAIL_R_LADDER
        fee = getattr(module, "TRAIL_FEE_BUFFER_PCT", None)
        return list(ladder), (Decimal(str(fee)) if fee is not None else TRAIL_FEE_BUFFER_PCT)

    def _trail_initial_sl(self, trade: Trade) -> Decimal:
        try:
            if trade.trail_history:
                return Decimal(str(trade.trail_history[0]["old_sl"]))
        except (InvalidOperation, KeyError, TypeError):
            pass
        return trade.sl

    def _trail_metrics(self, trade: Trade, price: Decimal) -> tuple[Decimal, Decimal, Decimal] | None:
        fill = trade.fill_price or trade.entry
        risk = abs(fill - self._trail_initial_sl(trade))
        if risk <= 0:
            return None
        r_now = (price - fill) / risk if trade.direction == "BUY" else (fill - price) / risk
        return fill, risk, r_now

    def _trail_target(
        self,
        trade: Trade,
        price: Decimal,
        extra: tuple[Decimal, str, str] | None = None,
    ) -> tuple[Decimal, Decimal, Decimal, str, str] | None:
        """SL baru terbaik dari R-ladder + usulan strategy: (sl, r_now, lock_r, sumber, catatan)."""
        metrics = self._trail_metrics(trade, price)
        if metrics is None:
            return None
        fill, risk, r_now = metrics
        buy = trade.direction == "BUY"
        ladder, fee_pct = self._trail_params()
        cands: list[tuple[Decimal, str, str]] = []
        lock_r = None
        for trigger, lock in ladder:
            if r_now >= Decimal(str(trigger)):
                lock_r = Decimal(str(lock))
        if lock_r is not None:
            gain = max(lock_r * risk, fill * fee_pct / Decimal("100"))
            cands.append(
                (
                    fill + gain if buy else fill - gain,
                    "R_LADDER",
                    f"R-ladder: harga {fmt_num(r_now)}R, SL dikunci di {fmt_num(lock_r)}R.",
                )
            )
        if extra is not None:
            cands.append(extra)
        if not cands:
            return None
        raw_sl, source, note = (max if buy else min)(cands, key=lambda item: item[0])
        meta = self.symbols.get(trade.pair)
        if meta is not None and meta.tick_size > 0:
            tick = meta.tick_size
            new_sl = (raw_sl / tick).to_integral_value(rounding=ROUND_DOWN if buy else ROUND_UP) * tick
        else:
            new_sl = raw_sl
        improves = new_sl > trade.sl if buy else new_sl < trade.sl
        if not improves or abs(new_sl - trade.sl) < risk * TRAIL_MIN_STEP_R:
            return None
        gap = (price - new_sl) if buy else (new_sl - price)
        if gap < risk * TRAIL_MIN_GAP_R:
            return None
        lock_eff = (new_sl - fill) / risk if buy else (fill - new_sl) / risk
        return new_sl, r_now, lock_eff, source, note

    async def _strategy_trailing(self, trade: Trade, price: Decimal) -> tuple[Decimal, str, str] | None:
        """Minta analyze_trailing di strategy.py (struktur M15 + RSI); dibatasi per trade."""
        now = time.monotonic()
        if now < self._trail_next_check.get(trade.trade_id, 0.0):
            return None
        self._trail_next_check[trade.trade_id] = now + TRAIL_STRATEGY_INTERVAL_SECONDS
        try:
            module = getattr(self, "_strategy_runtime_module", None) or await self._load_strategy_runtime()
            meta = self.symbols.get(trade.pair)
            payload = {
                "pair": trade.pair,
                "direction": trade.direction,
                "fill_price": float(trade.fill_price or trade.entry),
                "initial_sl": float(self._trail_initial_sl(trade)),
                "sl": float(trade.sl),
                "price": float(price),
                "tick_size": float(meta.tick_size) if meta is not None else 0.0,
                "filled_at_ms": int(trade.filled_at.timestamp() * 1000) if trade.filled_at else 0,
            }
            result = await self._call_strategy_function(
                module, "analyze_trailing", payload, {"fetch_structure": True}
            )
        except Exception as exc:
            log.info("[TRAIL] analisa strategy %s gagal: %s", trade.trade_id, exc)
            return None
        if not isinstance(result, dict) or result.get("new_sl") is None:
            return None
        try:
            level = Decimal(str(result["new_sl"]))
        except InvalidOperation:
            return None
        return level, str(result.get("source") or "STRATEGY"), str(result.get("reason") or "")

    async def _commit_trail(
        self,
        trade: Trade,
        old_sl: Decimal,
        new_sl: Decimal,
        price: Decimal,
        r_now: Decimal,
        lock_eff: Decimal,
        source: str,
        note: str,
        real: bool,
    ) -> None:
        """Catat trailing yang sudah berhasil: riwayat, event, dan notifikasi."""
        reason = f"Auto trail [{source}]: {note}"
        trade.trailing = True
        trade.trail_history.append(
            {
                "old_sl": decimal_to_str(old_sl),
                "new_sl": decimal_to_str(new_sl),
                "price": decimal_to_str(price),
                "reason": reason,
                "source": source,
                "timestamp": iso_utc(),
                "timestamp_wib": format_wib(now_utc()),
            }
        )
        try:
            await self._record_event(
                trade,
                "TRAIL",
                event_price=price,
                reason=reason,
                extra={
                    "old_sl": decimal_to_str(old_sl),
                    "new_sl": decimal_to_str(new_sl),
                    "auto": True,
                    "source": source,
                },
            )
        except Exception:
            log.exception("Gagal mencatat event TRAIL %s", trade.trade_id)
        locked = pct_change(trade.direction, trade.fill_price or trade.entry, new_sl)
        rows = [
            f"{'🟢' if trade.direction == 'BUY' else '🔴'} {trade.pair}  •  {trade.direction}",
            f"📡 Harga {fmt_price(price)}  (+{fmt_num(r_now)}R)",
            f"🛡 SL {fmt_price(old_sl)} → {fmt_price(new_sl)}",
            f"🔐 Terkunci {format_pct(locked)}  ({fmt_num(lock_eff)}R)",
            f"🧭 {source}: {note}"[:220],
            f"⚙️ Mode {'REAL' if real else 'SIMULASI'}",
        ]
        if trade.trade_id in self._deferred_protect:
            rows.append("⏳ SL baru dipasang otomatis begitu API pulih")
        await self.reply(card("🔒 TRAILING OTOMATIS", rows))

    def _queue_trail(self, trade: Trade, new_sl: Decimal, source: str, note: str, r_now: Decimal) -> None:
        """Antre trailing saat API dibatasi; per trade disimpan target paling melindungi (puncak)."""
        buy = trade.direction == "BUY"
        prev = self._deferred_trail.get(trade.trade_id)
        if prev is not None and (prev[0] >= new_sl if buy else prev[0] <= new_sl):
            return
        self._deferred_trail[trade.trade_id] = (new_sl, source, note, r_now)
        self._start_rate_limit_watcher()
        if prev is None:
            self._fire(
                self.reply(
                    card(
                        "⏸ TRAILING DITUNDA",
                        [
                            f"{'🟢' if trade.direction == 'BUY' else '🔴'} {trade.pair}  •  {trade.direction}",
                            f"🛡 Target SL {fmt_price(new_sl)}  (+{fmt_num(r_now)}R)",
                            "⏳ Binance API limit; target mengikuti puncak harga",
                            "▶️ Dijalankan otomatis begitu API pulih",
                        ],
                    )
                )
            )

    def _defer_trail(self, trade: Trade, price: Decimal) -> None:
        if not AUTO_TRAIL_ENABLED or trade.status != "FILLED":
            return
        target = self._trail_target(trade, price)
        if target is None:
            return
        new_sl, r_now, _lock, source, note = target
        self._queue_trail(trade, new_sl, source, note, r_now)

    async def _apply_deferred_trail(
        self,
        trade: Trade,
        item: tuple[Decimal, str, str, Decimal],
        price: Decimal | None,
    ) -> None:
        new_sl, source, note, r_peak = item
        buy = trade.direction == "BUY"
        if (buy and new_sl <= trade.sl) or (not buy and new_sl >= trade.sl):
            return
        if price is not None and ((buy and price <= new_sl) or (not buy and price >= new_sl)):
            # Harga sudah melewati SL trailing saat API dibatasi: setara SL ter-trigger.
            trade.trailing = True
            await self._emergency_close(
                trade,
                f"Trailing tertunda: harga {fmt_price(price)} sudah melewati "
                f"SL trailing {fmt_price(new_sl)} [{source}].",
            )
            return
        old_sl = trade.sl
        replaced = await self._replace_real_sl(trade, new_sl)
        if not replaced or trade.status != "FILLED":
            return
        ref_price = price if price is not None else new_sl
        metrics = self._trail_metrics(trade, ref_price)
        lock_eff = Decimal("0")
        if metrics is not None:
            fill, risk, _r = metrics
            lock_eff = (new_sl - fill) / risk if buy else (fill - new_sl) / risk
        await self._commit_trail(
            trade, old_sl, new_sl, ref_price, r_peak, lock_eff, source, f"(tertunda) {note}", True
        )

    async def _auto_trail(self, trade: Trade, price: Decimal) -> None:
        if not AUTO_TRAIL_ENABLED or trade.status != "FILLED" or trade.trade_id in self._trail_busy:
            return
        if trade.pending_sl_state != "NONE":
            # A replacement is already in flight/recovery. Never start a third SL
            # or hammer Binance on every WebSocket price tick. The periodic
            # protection reconcile resumes the transaction.
            return
        real = trade.real_enabled and self.real_mode
        if real and self.real.cooldown_remaining > 0:
            self._defer_trail(trade, price)
            return
        metrics = self._trail_metrics(trade, price)
        if metrics is None:
            return
        extra = None
        if metrics[2] >= Decimal("1.0"):
            extra = await self._strategy_trailing(trade, price)
        target = self._trail_target(trade, price, extra)
        if target is None:
            return
        new_sl, r_now, lock_eff, source, note = target
        if real and not self._real_poll_due(trade, "trail", 3.0):
            return
        old_sl = trade.sl
        self._trail_busy.add(trade.trade_id)
        try:
            if real:
                replaced = await self._replace_real_sl(trade, new_sl)
                if not replaced:
                    return
            else:
                trade.sl = new_sl
        except BinanceRateLimitError as exc:
            # SL lama masih utuh (hapus gagal); antre target agar tidak hilang saat harga berbalik.
            self._queue_trail(trade, new_sl, source, note, r_now)
            self._notify_rate_limit(exc, "AUTO TRAIL")
            return
        except Exception as exc:
            # Backoff ~60 detik agar tidak mengulang tiap tick.
            self._real_poll_guard[f"trail:{trade.trade_id}"] = time.monotonic() + 57.0
            log.warning("[TRAIL] %s gagal memindah SL: %s", trade.trade_id, exc)
            return
        finally:
            self._trail_busy.discard(trade.trade_id)

        if trade.status != "FILLED":
            return
        await self._commit_trail(trade, old_sl, new_sl, price, r_now, lock_eff, source, note, real)

    async def _reconcile_real_trade(self, trade: Trade) -> None:
        if not self.real_mode or not trade.real_enabled:
            return

        if trade.status == "PENDING":
            position = await self.real.get_position(trade.pair, trade.direction)
            if position is not None:
                await self._confirm_real_fill(trade)
                return

            if not trade.entry_order_id and not trade.entry_client_order_id:
                trade.real_state = "REAL_ERROR"
                trade.real_error = "Setup real tidak memiliki entry order metadata."
                log.warning(
                    "REAL setup %s dipulihkan tanpa entry order metadata; "
                    "bot tidak membuat order baru otomatis.",
                    trade.trade_id,
                )
                return

            try:
                order = await self.real.get_order(
                    trade.pair,
                    order_id=trade.entry_order_id,
                    client_order_id=(
                        trade.entry_client_order_id
                        if trade.entry_order_id is None
                        else None
                    ),
                )
            except BinanceAPIError as exc:
                if exc.code == -2013:
                    trade.real_state = "REAL_ERROR"
                    trade.real_error = str(exc)
                    log.warning(
                        "Entry order REAL %s tidak ditemukan saat /open; "
                        "bot tidak membuat order baru otomatis.",
                        trade.trade_id,
                    )
                    return
                raise

            status = str(order.get("status") or "").upper()
            if status in {"FILLED", "PARTIALLY_FILLED"}:
                if not await self._confirm_real_fill(trade):
                    trade.real_state = "REAL_ERROR"
                    trade.real_error = (
                        f"Order {status} tetapi position belum terkonfirmasi "
                        f"untuk {trade.pair}."
                    )
            elif status == "NEW":
                trade.real_state = "REAL_PENDING"
                trade.real_error = None
            elif status in {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}:
                trade.real_state = "REAL_ERROR"
                trade.real_error = f"Entry order status saat /open: {status}"
            else:
                trade.real_state = "REAL_ERROR"
                trade.real_error = f"Entry order status saat /open: {status}"
            return

        if trade.status == "FILLED":
            position = await self.real.get_position(trade.pair, trade.direction)
            if position is not None:
                if trade.pending_sl_state != "NONE" and trade.pending_sl_price is not None:
                    await self._resume_sl_transition(trade)
                await self._ensure_real_protective_orders(trade)
                trade.real_state = "REAL_FILLED"
                trade.real_error = None
                return

            # Position zero. Check a pending replacement first: it may have
            # triggered after the new SL was confirmed and before the old SL
            # cleanup/promotion completed.
            tracked_exit_algos: list[tuple[str, int | None, str | None, Decimal]] = []
            if trade.pending_sl_algo_id is not None or trade.pending_sl_client_algo_id:
                tracked_exit_algos.append(
                    (
                        "SL",
                        trade.pending_sl_algo_id,
                        trade.pending_sl_client_algo_id,
                        trade.pending_sl_price or trade.sl,
                    )
                )
            tracked_exit_algos.extend(
                [
                    ("TP", trade.tp_algo_id, trade.tp_client_algo_id, trade.tp),
                    ("SL", trade.sl_algo_id, trade.sl_client_algo_id, trade.sl),
                ]
            )

            seen_exit_keys: set[tuple[str, int | None, str | None]] = set()
            for result, algo_id, client_algo_id, fallback_price in tracked_exit_algos:
                key = (result, algo_id, client_algo_id)
                if key in seen_exit_keys:
                    continue
                seen_exit_keys.add(key)
                if algo_id is None and not client_algo_id:
                    continue
                try:
                    if algo_id is not None:
                        algo = await self.real.get_algo_order(algo_id=algo_id)
                    else:
                        open_algos = await self.real.get_open_algo_orders(trade.pair)
                        algo = self._find_specific_algo(
                            open_algos,
                            client_algo_id=client_algo_id,
                        )
                        if algo is None:
                            continue
                except BinanceAPIError as exc:
                    if exc.code == -2013:
                        continue
                    raise

                status = str(algo.get("algoStatus") or "").upper()
                if status in {"TRIGGERED", "FINISHED"}:
                    actual_raw = algo.get("actualPrice") or algo.get("triggerPrice")
                    actual_price = (
                        fallback_price
                        if actual_raw in (None, "", "0", 0)
                        else parse_decimal(str(actual_raw))
                    )
                    trade.real_state = "REAL_CLOSED"
                    trade.real_error = None
                    trade.real_exit_check = result
                    if result == "SL" and (
                        algo_id == trade.pending_sl_algo_id
                        or (client_algo_id and client_algo_id == trade.pending_sl_client_algo_id)
                    ):
                        self._clear_pending_sl(trade)
                    await self._finalize_trade(
                        trade,
                        result=result,
                        exit_price=actual_price,
                        reason=(
                            trade.tp_reason
                            if result == "TP"
                            else trade.sl_reason
                        ),
                    )
                    return

            already_flagged = trade.real_state == "REAL_ERROR"
            trade.real_state = "REAL_ERROR"
            trade.real_error = (
                "Position Binance tidak ada saat /open dan penyebab penutupan "
                "tidak dapat dipastikan dari tracked algo order."
            )
            if not already_flagged:
                log.warning(
                    "REAL position %s tidak ditemukan saat /open dan hasil close tidak dapat dipastikan.",
                    trade.trade_id,
                )

    async def _clear_margin_bans(self) -> None:
        """Ban margin hanya valid untuk konfigurasi margin/leverage lamanya."""
        stale = [
            pair
            for pair, item in self.banned_pairs.items()
            if item.get("source") in {"AUTO_MARGIN_INFLUENCE", "AUTO_LEVERAGE_UNSUPPORTED"}
        ]
        for pair in stale:
            self.banned_pairs.pop(pair, None)
        self._scan_margin_streak.clear()
        if stale:
            try:
                await self._persist_banned_pairs()
            except Exception:
                log.exception("Gagal menyimpan ban setelah reset ban margin.")

    async def _set_margin_command(self, argument: str) -> None:
        if not argument:
            await self.reply(
                f"💰 MARGIN\n\nSaat ini: {decimal_to_str(self.margin_usdt)} USDT\n\n"
                "Gunakan /margin 0.5"
            )
            return
        margin = parse_decimal(argument)
        if margin <= 0:
            raise ValueError("Margin harus lebih besar dari 0.")
        self.margin_usdt = margin
        await self._clear_margin_bans()
        await self.reply(
            "✅ MARGIN DIPERBARUI\n\n"
            f"Margin default: {decimal_to_str(self.margin_usdt)} USDT\n"
            f"Leverage: {self.leverage}x\n"
            f"Target Notional: {decimal_to_str(self.margin_usdt * Decimal(self.leverage))} USDT"
        )

    async def _set_leverage_command(self, argument: str) -> None:
        if not argument:
            await self.reply(
                f"⚡ LEVERAGE\n\nSaat ini: {self.leverage}x\n\n"
                "Gunakan /leverage 10"
            )
            return
        leverage = safe_int(argument, "Leverage")
        if not 1 <= leverage <= 125:
            raise ValueError("Leverage harus 1 sampai 125x.")
        self.leverage = leverage
        await self._clear_margin_bans()
        await self.reply(
            "✅ LEVERAGE DIPERBARUI\n\n"
            f"Leverage default: {self.leverage}x\n"
            f"Margin: {decimal_to_str(self.margin_usdt)} USDT\n"
            f"Target Notional: {decimal_to_str(self.margin_usdt * Decimal(self.leverage))} USDT"
        )

    # --------------------------------------------------------
    # SETUP PERSISTENCE
    # --------------------------------------------------------

    async def save_setups(self) -> None:
        """Simpan snapshot seluruh setup aktif (/trade) ke GitHub.

        File ini sengaja dipisahkan dari histori pencatatan supaya setup
        aktif dapat dibawa ke versi main.py berikutnya. /reset tidak
        menghapus file ini.
        """
        async with self._trade_lock:
            records = [
                trade.to_record()
                for trade in self.active_trades.values()
                if trade.result is None and trade.status in {
                    "PENDING",
                    "FILLED",
                }
            ]

        payload = {
            "version": 1,
            "saved_at": iso_utc(),
            "saved_at_wib": format_wib(now_utc()),
            "count": len(records),
            "setups": records,
        }

        content = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        await self.github.replace_file(
            SETUPS_PATH,
            content,
            f"setup: save {len(records)} active setup(s)",
        )

        pending = sum(
            1
            for record in records
            if record.get("status") == "PENDING"
        )
        filled = sum(
            1
            for record in records
            if record.get("status") == "FILLED"
        )

        await self.reply(
            "💾 SETUP TERSIMPAN KE GITHUB\n\n"
            f"Total: {len(records)}\n"
            f"Pending: {pending}\n"
            f"Filled: {filled}\n\n"
            f"File: {SETUPS_PATH}\n"
            "Setup dapat dipulihkan dengan /open setelah main.py diganti."
        )

    def _trade_from_saved_record(self, record: dict[str, Any]) -> Trade:
        """Rebuild satu Trade aktif dari data /setup yang tersimpan."""
        required = [
            "trade_id",
            "pair",
            "direction",
            "price_now_reference",
            "entry",
            "price_exp",
            "sl",
            "tp",
        ]
        missing = [
            key
            for key in required
            if record.get(key) in (None, "")
        ]
        if missing:
            raise ValueError(
                "Field setup kurang: " + ", ".join(missing)
            )

        status = str(record.get("status") or "PENDING").upper().strip()
        if status not in {"PENDING", "FILLED"}:
            raise ValueError(
                f"Status setup {record.get('trade_id')} tidak aktif: {status}."
            )

        if record.get("result") not in (None, ""):
            raise ValueError(
                f"Setup {record.get('trade_id')} sudah memiliki result dan tidak dapat dibuka."
            )

        if record.get("closed_at") not in (None, ""):
            raise ValueError(
                f"Setup {record.get('trade_id')} memiliki closed_at dan tidak dapat dibuka."
            )

        created_at = parse_iso(record.get("created_at")) or now_utc()
        filled_at = parse_iso(record.get("filled_at"))

        def saved_number(
            field: str,
            parser: Any,
            *,
            required: bool = False,
        ) -> Decimal | None:
            raw = record.get(field)
            if raw in (None, ""):
                if required:
                    raise ValueError(
                        f"Setup {record.get('trade_id', '-')} field '{field}' kosong."
                    )
                return None
            try:
                return parser(str(raw))
            except (ValueError, TypeError, InvalidOperation) as exc:
                raise ValueError(
                    f"Setup {record.get('trade_id', '-')} field '{field}' tidak valid: {exc}"
                ) from exc

        fill_price = saved_number("fill_price", parse_decimal)
        pnl_percent = saved_number("pnl_percent", parse_signed_decimal)
        max_favorable_pct = saved_number("max_favorable_pct", parse_signed_decimal)
        max_adverse_pct = saved_number("max_adverse_pct", parse_signed_decimal)

        trail_history = record.get("trail_history")
        if not isinstance(trail_history, list):
            trail_history = []

        return Trade(
            trade_id=str(record["trade_id"]),
            # Setup yang dibuka kembali menjadi bagian dari session main.py
            # yang sedang berjalan, sementara created_at tetap dipertahankan.
            session_id=self.session_id,
            pair=normalize_symbol(str(record["pair"])),
            direction=str(record["direction"]).upper().strip(),
            price_now_reference=saved_number("price_now_reference", parse_decimal, required=True),
            entry=saved_number("entry", parse_decimal, required=True),
            entry_reason=str(record.get("entry_reason") or ""),
            price_exp=saved_number("price_exp", parse_decimal, required=True),
            price_exp_reason=str(record.get("price_exp_reason") or ""),
            sl=saved_number("sl", parse_decimal, required=True),
            sl_reason=str(record.get("sl_reason") or ""),
            tp=saved_number("tp", parse_decimal, required=True),
            tp_reason=str(record.get("tp_reason") or ""),
            status=status,
            trailing=bool(record.get("trailing", False)),
            created_at=created_at,
            filled_at=filled_at,
            closed_at=None,
            fill_price=fill_price,
            exit_price=None,
            result=None,
            result_reason=None,
            pnl_percent=pnl_percent if status == "FILLED" else None,
            max_favorable_pct=max_favorable_pct,
            max_adverse_pct=max_adverse_pct,
            trail_history=copy.deepcopy(trail_history),
            strategy_name=str(record.get("strategy_name") or "MANUAL"),
            strategy_version=str(record.get("strategy_version") or "1.0"),
            strategy_source=str(record.get("strategy_source") or "MANUAL"),
            strategy_confidence=(
                saved_number("strategy_confidence", parse_nonnegative_decimal)
            ),
            strategy_data_source=(
                str(record.get("strategy_data_source"))
                if record.get("strategy_data_source") not in (None, "")
                else None
            ),
            strategy_analysis=(
                copy.deepcopy(record.get("strategy_analysis"))
                if isinstance(record.get("strategy_analysis"), dict)
                else {}
            ),
            margin_usdt=(
                saved_number("margin_usdt", parse_decimal)
                if record.get("margin_usdt") not in (None, "")
                else self.margin_usdt
            ),
            leverage=int(record.get("leverage") or self.leverage),
            quantity=saved_number("quantity", parse_decimal),
            target_notional=saved_number("target_notional", parse_decimal),
            actual_notional=saved_number("actual_notional", parse_decimal),
            real_enabled=bool(record.get("real_enabled", False)),
            real_state=str(record.get("real_state") or "SIMULATION"),
            position_side=(str(record.get("position_side")) if record.get("position_side") not in (None, "") else None),
            entry_order_id=(int(record.get("entry_order_id")) if str(record.get("entry_order_id") or "").isdigit() else None),
            entry_client_order_id=(str(record.get("entry_client_order_id")) if record.get("entry_client_order_id") not in (None, "") else None),
            tp_algo_id=(int(record.get("tp_algo_id")) if str(record.get("tp_algo_id") or "").isdigit() else None),
            sl_algo_id=(int(record.get("sl_algo_id")) if str(record.get("sl_algo_id") or "").isdigit() else None),
            tp_client_algo_id=(str(record.get("tp_client_algo_id")) if record.get("tp_client_algo_id") not in (None, "") else None),
            sl_client_algo_id=(str(record.get("sl_client_algo_id")) if record.get("sl_client_algo_id") not in (None, "") else None),
            pending_sl_algo_id=(int(record.get("pending_sl_algo_id")) if str(record.get("pending_sl_algo_id") or "").isdigit() else None),
            pending_sl_client_algo_id=(str(record.get("pending_sl_client_algo_id")) if record.get("pending_sl_client_algo_id") not in (None, "") else None),
            pending_sl_price=saved_number("pending_sl_price", parse_decimal),
            pending_sl_state=str(record.get("pending_sl_state") or "NONE"),
            pending_sl_created_at=parse_iso(record.get("pending_sl_created_at")),
            pending_sl_error=(str(record.get("pending_sl_error")) if record.get("pending_sl_error") not in (None, "") else None),
            real_error=(str(record.get("real_error")) if record.get("real_error") not in (None, "") else None),
            real_exit_check=(str(record.get("real_exit_check")) if record.get("real_exit_check") not in (None, "") else None),
        )

    async def open_setups(self) -> None:
        """Pulihkan setup aktif yang sebelumnya disimpan dengan /setup."""
        raw, _ = await self.github.get_file(SETUPS_PATH)

        if not raw:
            await self.reply(
                "📂 OPEN SETUP\n\n"
                f"Belum ada file {SETUPS_PATH} di GitHub.\n"
                "Gunakan /setup terlebih dahulu."
            )
            return

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{SETUPS_PATH} bukan JSON valid."
            ) from exc

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("setups", [])
        else:
            raise RuntimeError(
                f"{SETUPS_PATH} harus berisi list setup atau object dengan key 'setups'."
            )

        if not isinstance(records, list):
            raise RuntimeError(
                f"{SETUPS_PATH}: key 'setups' harus berupa list."
            )

        loaded = 0
        skipped = 0
        skipped_details: list[str] = []
        symbols_to_subscribe: set[str] = set()

        for raw_record in records:
            if not isinstance(raw_record, dict):
                skipped += 1
                skipped_details.append("record bukan object")
                continue

            try:
                trade = self._trade_from_saved_record(raw_record)
            except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
                skipped += 1
                skipped_details.append(str(exc))
                continue

            if trade.trade_id in self.active_trades:
                skipped += 1
                skipped_details.append(
                    f"{trade.trade_id}: sudah aktif"
                )
                continue

            if trade.real_enabled and not self.real_mode:
                skipped += 1
                skipped_details.append(
                    f"{trade.trade_id}: setup REAL butuh /real on dulu"
                )
                continue

            if len(self.active_trades) >= self.max_active_trades:
                skipped += 1
                skipped_details.append(
                    f"{trade.trade_id}: batas active trade tercapai"
                )
                continue

            self.active_trades[trade.trade_id] = trade
            loaded += 1
            if trade.real_enabled:
                try:
                    await self._reconcile_real_trade(trade)
                except Exception as exc:
                    await self._handle_real_exception(exc, "RECONCILE /open", trade)
            if trade.trade_id in self.active_trades:
                symbols_to_subscribe.add(trade.pair)

        for symbol in sorted(symbols_to_subscribe):
            try:
                await self.ws.add_symbol(symbol)
            except Exception:
                # Setup tetap dipulihkan ke RAM. WebSocket akan retry/reconnect.
                log.exception(
                    "Subscription WebSocket %s gagal saat /open.",
                    symbol,
                )

        lines = [
            "📂 SETUP DIBUKA",
            "",
            f"Berhasil: {loaded}",
            f"Dilewati: {skipped}",
        ]

        if skipped_details:
            lines.extend([
                "",
                "Detail dilewati:",
            ])
            lines.extend(
                f"• {detail}"
                for detail in skipped_details[:10]
            )

            if len(skipped_details) > 10:
                lines.append(
                    f"• ...dan {len(skipped_details) - 10} lainnya"
                )

        lines.extend([
            "",
            "Gunakan /trade untuk melihat setup yang aktif.",
        ])

        await self.reply("\n".join(lines))
        await self._resume_scan_after_capacity_change()

    # --------------------------------------------------------
    # BANNED PAIRS
    # --------------------------------------------------------

    @staticmethod
    def _parse_ban_until(value: Any) -> datetime | None:
        if value in (None, "", False):
            return None
        try:
            parsed = parse_iso(str(value))
        except Exception:
            return None
        return parsed

    async def _load_banned_pairs(self) -> None:
        raw, _ = await self.github.get_file(BANNED_PATH)
        loaded: dict[str, dict[str, Any]] = {}

        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
                rows = payload.get("bans", payload) if isinstance(payload, dict) else payload
                if isinstance(rows, list):
                    for item in rows:
                        if not isinstance(item, dict):
                            continue
                        pair = normalize_symbol(str(item.get("pair") or ""))
                        if not pair:
                            continue
                        loaded[pair] = {
                            "pair": pair,
                            "banned_at": item.get("banned_at") or iso_utc(),
                            "until": item.get("until"),
                            "reason": str(item.get("reason") or "").strip(),
                            "source": str(item.get("source") or "MANUAL").strip(),
                        }
                elif isinstance(rows, dict):
                    for raw_pair, item in rows.items():
                        pair = normalize_symbol(str(raw_pair))
                        if not pair or not isinstance(item, dict):
                            continue
                        loaded[pair] = {
                            "pair": pair,
                            "banned_at": item.get("banned_at") or iso_utc(),
                            "until": item.get("until"),
                            "reason": str(item.get("reason") or "").strip(),
                            "source": str(item.get("source") or "MANUAL").strip(),
                        }
            except Exception:
                log.exception("Gagal membaca %s; daftar ban di-reset di RAM.", BANNED_PATH)
                loaded = {}

        self.banned_pairs = loaded
        await self._purge_expired_bans(persist=False)
        log.info("[BAN] Loaded active banned pairs: %s", len(self.banned_pairs))

    async def _persist_banned_pairs(self) -> None:
        async with self._ban_lock:
            rows = [self.banned_pairs[pair] for pair in sorted(self.banned_pairs)]
            payload = {
                "version": 1,
                "saved_at": iso_utc(),
                "bans": rows,
            }
            await self.github.replace_file(
                BANNED_PATH,
                json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                f"ban: update {len(rows)} pair(s)",
            )

    async def _purge_expired_bans(self, *, persist: bool = True) -> None:
        now = now_utc()
        removed = []
        for pair, item in list(self.banned_pairs.items()):
            until = self._parse_ban_until(item.get("until"))
            if until is not None and until <= now:
                self.banned_pairs.pop(pair, None)
                removed.append(pair)
        if removed:
            log.info("[BAN] Expired bans removed: %s", ", ".join(sorted(removed)))
            if persist:
                await self._persist_banned_pairs()

    def _is_pair_banned(self, pair: str) -> bool:
        pair = normalize_symbol(pair)
        item = self.banned_pairs.get(pair)
        if not item:
            return False
        until = self._parse_ban_until(item.get("until"))
        if until is not None and until <= now_utc():
            return False
        return True

    def _ban_remaining_text(self, item: dict[str, Any]) -> str:
        until = self._parse_ban_until(item.get("until"))
        if until is None:
            return "PERMANENT"
        seconds = max(0, int((until - now_utc()).total_seconds()))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}j {minutes}m"

    async def _ban_pair(
        self,
        pair: str,
        *,
        hours: Decimal | None = None,
        reason: str = "",
        source: str = "MANUAL",
    ) -> None:
        pair = normalize_symbol(pair)
        if not pair:
            raise ValueError("Pair ban tidak valid.")
        until: str | None
        if hours is None:
            until = None
        else:
            if hours <= 0:
                raise ValueError("Durasi ban harus > 0 jam.")
            until = iso_utc(now_utc() + timedelta(hours=float(hours)))

        new_source = str(source or "MANUAL").strip()
        new_reason = str(reason or "").strip()
        existing = self.banned_pairs.get(pair)

        # Automatic bans may extend/protect an existing ban, but never weaken
        # a permanent or longer-running ban. Manual /banned explicitly wins.
        preserve_existing = False
        if existing and new_source != "MANUAL":
            existing_until = self._parse_ban_until(existing.get("until"))
            new_until_dt = self._parse_ban_until(until)
            if existing_until is None and new_until_dt is not None:
                until = None
                preserve_existing = True
            elif (
                existing_until is not None
                and new_until_dt is not None
                and existing_until > new_until_dt
            ):
                until = existing.get("until")
                preserve_existing = True
            elif existing_until is None and new_until_dt is None:
                preserve_existing = True

            if preserve_existing:
                new_reason = str(existing.get("reason") or new_reason).strip()
                new_source = str(existing.get("source") or new_source).strip()

        self.banned_pairs[pair] = {
            "pair": pair,
            "banned_at": existing.get("banned_at") if existing and new_source != "MANUAL" else iso_utc(),
            "until": until,
            "reason": new_reason,
            "source": new_source,
        }
        await self._persist_banned_pairs()
        log.info(
            "[BAN] %s | duration=%s | source=%s | reason=%s",
            pair,
            "PERMANENT" if until is None else f"{hours}h",
            source,
            reason or "-",
        )

    async def _auto_ban_for_result(self, trade: Trade, result: str) -> None:
        result = str(result or "").upper()
        if result == "EXPIRED":
            await self._ban_pair(
                trade.pair,
                hours=SCAN_BANNED_PRICE_EXP_HOURS,
                reason="Setup berakhir karena Price Exp tercapai sebelum entry.",
                source="AUTO_PRICE_EXP",
            )
        elif result in {"TP", "SL", "TRAIL"}:
            await self._ban_pair(
                trade.pair,
                hours=SCAN_BANNED_TP_SL_HOURS,
                reason=f"Setup berakhir {result}; pair dikunci sementara dari scanner.",
                source=f"AUTO_{result}",
            )

    async def _handle_banned_command(self, text: str) -> None:
        await self._purge_expired_bans()
        parts = text.split(maxsplit=3)
        if len(parts) == 1:
            if not self.banned_pairs:
                await self.reply(
                    "🚫 BAN LIST\n\n"
                    "Total pair banned: 0\n\n"
                    "Tidak ada pair yang sedang dibanned."
                )
                return
            groups: dict[str, list[str]] = {}
            for pair in sorted(self.banned_pairs):
                item = self.banned_pairs[pair]
                source = str(item.get("source") or "")
                reason = str(item.get("reason") or "")
                if source == "AUTO_MARGIN_INFLUENCE":
                    label = "💸 Margin influence"
                elif source == "AUTO_LEVERAGE_UNSUPPORTED":
                    label = "⚙️ Leverage tidak didukung"
                elif source == "AUTO_INSUFFICIENT_HISTORY":
                    label = "🕒 Riwayat candle kurang"
                elif source == "MANUAL":
                    label = "✋ Manual"
                elif reason.startswith("H4 structure") or source == "AUTO_STRUCTURE_MISMATCH":
                    label = "🧭 Tidak searah BTC"
                elif reason.startswith("Confidence") or source == "AUTO_BELOW_THRESHOLD":
                    label = "📉 Di bawah threshold"
                elif reason.startswith("Setup berakhir") or source == "AUTO_PRICE_EXP":
                    label = "🔒 Setelah trade selesai"
                else:
                    label = "📌 Lainnya"
                remaining = self._ban_remaining_text(item)
                short = remaining.split(" ")[0] if remaining != "PERMANENT" else "∞"
                groups.setdefault(label, []).append(f"{pair} {short}")

            chunks: list[str] = []
            current = f"🚫 BAN LIST  •  {len(self.banned_pairs)} pair\n"
            for label, entries in groups.items():
                block = f"\n{label} ({len(entries)})\n" + "  ·  ".join(entries) + "\n"
                if len(current) + len(block) > 3500:
                    chunks.append(current)
                    current = ""
                while len(block) > 3500:
                    cut = block.rfind("  ·  ", 0, 3500)
                    cut = cut if cut > 0 else 3500
                    chunks.append(current + block[:cut])
                    current = ""
                    block = block[cut:].lstrip(" ·\n")
                current += block
            if current.strip():
                chunks.append(current)
            for chunk in chunks:
                await self.reply(chunk)
            return

        pair = normalize_symbol(parts[1])
        hours: Decimal | None = None
        reason = ""
        if len(parts) >= 3:
            try:
                hours = Decimal(parts[2])
                reason = parts[3].strip() if len(parts) == 4 else ""
            except InvalidOperation:
                # /banned PAIR alasan
                hours = None
                reason = " ".join(parts[2:]).strip()

        await self._ban_pair(pair, hours=hours, reason=reason, source="MANUAL")
        await self.reply(
            "🚫 PAIR DIBANNED\n\n"
            f"Pair: {pair}\n"
            f"Durasi: {'PERMANENT' if hours is None else f'{hours} jam'}\n"
            f"Alasan: {reason or 'Tanpa alasan'}"
        )

    async def _handle_unban_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            await self.reply("Gunakan /unban PAIR atau /unban all")
            return
        target = normalize_symbol(parts[1])
        if parts[1].strip().lower() == "all":
            count = len(self.banned_pairs)
            self.banned_pairs.clear()
            await self._persist_banned_pairs()
            await self.reply(f"✅ SEMUA BAN DIHAPUS\n\nTotal: {count}")
            return
        if target not in self.banned_pairs:
            await self.reply(f"Pair {target} tidak ada di ban list.")
            return
        self.banned_pairs.pop(target, None)
        await self._persist_banned_pairs()
        await self.reply(f"✅ UNBAN {target}")

    # --------------------------------------------------------
    # SCAN CONFIG / COMMANDS
    # --------------------------------------------------------

    def _h4_slot_key(self, dt: datetime) -> str:
        return dt.astimezone(TZ).strftime("%Y%m%d-%H%M")

    def _h4_latest_scheduled_slot(self, now: datetime | None = None) -> datetime | None:
        """Return the most recent scheduled H4 close at/before now (WIB)."""
        local = (now or now_utc()).astimezone(TZ)
        candidates: list[datetime] = []
        for day_offset in (0, -1):
            day = local.date() + timedelta(days=day_offset)
            for hour in H4_SCAN_HOURS_WIB:
                candidates.append(
                    datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=TZ)
                )
        past = [item for item in candidates if item <= local]
        return max(past) if past else None

    def _h4_next_scheduled_slot(self, now: datetime | None = None) -> datetime:
        """Return the next H4 close strictly after now (WIB)."""
        local = (now or now_utc()).astimezone(TZ)
        today = local.date()
        for hour in H4_SCAN_HOURS_WIB:
            candidate = datetime(today.year, today.month, today.day, hour, 0, 0, tzinfo=TZ)
            if candidate > local:
                return candidate
        tomorrow = today + timedelta(days=1)
        return datetime(
            tomorrow.year,
            tomorrow.month,
            tomorrow.day,
            H4_SCAN_HOURS_WIB[0],
            0,
            0,
            tzinfo=TZ,
        )

    def _h4_trigger_slot(self, now: datetime | None = None) -> datetime | None:
        """Return an unconsumed H4 slot that is due now."""
        local = (now or now_utc()).astimezone(TZ)
        latest = self._h4_latest_scheduled_slot(local)
        if latest is None:
            return None
        elapsed = (local - latest).total_seconds()
        if elapsed < 0 or elapsed > H4_SCAN_TRIGGER_LEEWAY_SECONDS:
            return None
        if self._h4_slot_key(latest) == self._h4_last_trigger_slot:
            return None
        return latest

    def _h4_next_slot_text(self, now: datetime | None = None) -> str:
        local = (now or now_utc()).astimezone(TZ)
        due = self._h4_trigger_slot(local)
        slot = due or self._h4_next_scheduled_slot(local)
        return slot.strftime("%d-%m-%Y %H:%M WIB")

    def _h4_slot_deadline(self, slot: datetime) -> datetime:
        return slot + timedelta(seconds=H4_SCAN_SLOT_VALIDITY_SECONDS)

    def _h4_window_remaining_seconds(self, now: datetime | None = None) -> float | None:
        if not self._h4_window_active or self._h4_window_deadline is None:
            return None
        current = (now or now_utc()).astimezone(TZ)
        return (self._h4_window_deadline - current).total_seconds()

    def _h4_window_expired(self, now: datetime | None = None) -> bool:
        remaining = self._h4_window_remaining_seconds(now)
        return remaining is not None and remaining <= 0

    def _invalidate_h4_window(self, reason: str) -> None:
        slot_key = self._h4_slot_key(self._h4_window_slot) if self._h4_window_slot else self._h4_last_trigger_slot
        self._h4_invalid_slot_key = slot_key
        self._h4_window_active = False
        self._h4_window_slot = None
        self._h4_window_deadline = None
        self._h4_opening_gate = False
        self._scan_auto_paused = True
        log.warning("[SCAN][H4] slot %s invalid: %s", slot_key or "-", reason)
        self._fire(
            self.reply(
                card(
                    "⛔ H4 SLOT INVALID",
                    [
                        f"🕓 Slot      {slot_key or '-'} WIB",
                        "⏱ Batas     +1 jam dari jadwal",
                        f"📌 Alasan    {reason}",
                        "▶️ Berikut  menunggu jadwal H4 selanjutnya.",
                    ],
                )
            )
        )

    def _h4_window_text(self) -> str:
        if not self._h4_enabled:
            return "OFF"
        if self._h4_window_active:
            remaining = self._h4_window_remaining_seconds()
            if remaining is not None:
                return f"ACTIVE ({max(0, remaining):.0f}s)"
            return "ACTIVE"
        if self._h4_invalid_slot_key == self._h4_last_trigger_slot and self._h4_invalid_slot_key:
            return "WAITING_NEXT_H4"
        return "WAITING_H4"

    def _h4_status_text(self) -> str:
        lines = [
            "🕓 H4 SCANNER GATE",
            "",
            f"H4 Mode: {'ON' if self._h4_enabled else 'OFF'}",
            f"Scan: {'ON' if self._scan_user_enabled else 'OFF'}",
            f"Window: {self._h4_window_text()}",
            f"Active: {len(self.active_trades)}/{self.max_active_trades}",
        ]
        if self._h4_enabled:
            lines.extend([
                f"Next H4: {self._h4_next_slot_text()}",
                "Jadwal: 03:00 • 07:00 • 11:00 • 15:00 • 19:00 • 23:00 WIB",
                (
                    "Status: scanner akan terus berjalan sampai MAX penuh."
                    if self._h4_window_active
                    else "Status: menunggu close H4 berikutnya."
                ),
            ])
        else:
            lines.append("Status: scanner berjalan normal tanpa gate H4.")
        return "\n".join(lines)

    async def _handle_h4_command(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            await self.reply(self._h4_status_text())
            return

        mode = parts[1].lower()
        if mode not in {"on", "off"}:
            raise ValueError("Gunakan /H4 on atau /H4 off.")

        if mode == "off":
            was_enabled = self._h4_enabled
            was_window_active = self._h4_window_active
            self._h4_enabled = False
            self._h4_window_active = False
            self._h4_last_trigger_slot = None
            self._h4_window_slot = None
            self._h4_window_deadline = None
            self._h4_invalid_slot_key = None
            self._h4_opening_gate = False

            # Jika masih berada di H4 window yang sedang berjalan, jangan
            # membatalkan cycle: H4 gate dicabut dan scan boleh lanjut normal.
            # Jika masih WAITING_H4, restart task agar tidak tetap tidur sampai
            # jadwal berikutnya.
            if self._scan_user_enabled and was_enabled:
                if was_window_active:
                    self._scan_auto_paused = False
                    if self._scan_task is None or self._scan_task.done():
                        await self._start_scan_task()
                else:
                    self._scan_generation += 1
                    await self._stop_scan_task()
                    self._scan_auto_paused = False
                    await self._start_scan_task()

            await self.reply(
                "🟢 H4 OFF\n\n"
                "SCAN kembali memakai mode normal.\n"
                f"Active: {len(self.active_trades)}/{self.max_active_trades}"
            )
            return

        # H4 ON selalu memulai ulang scheduler boundary. Jika scan sedang
        # berjalan bebas, current cycle dibatalkan dan hasilnya di-invalidasi.
        self._h4_enabled = True
        self._h4_window_active = False
        self._h4_last_trigger_slot = None
        self._h4_window_slot = None
        self._h4_window_deadline = None
        self._h4_invalid_slot_key = None
        self._h4_opening_gate = False
        self._scan_generation += 1

        if self._scan_user_enabled:
            await self._stop_scan_task()
            self._scan_auto_paused = False
            await self._start_scan_task()

        await self.reply(
            "🕓 H4 ON\n\n"
            "Scanning sekarang menunggu jadwal close H4.\n"
            f"Next H4: {self._h4_next_slot_text()}\n"
            "Jadwal: 03:00 • 07:00 • 11:00 • 15:00 • 19:00 • 23:00 WIB\n"
            f"Active: {len(self.active_trades)}/{self.max_active_trades}"
        )

    async def _handle_scan_command(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            state = "ON" if self._scan_user_enabled else "OFF"
            runtime = (
                "WAITING_H4"
                if self._scan_user_enabled and self._h4_enabled and not self._h4_window_active
                else "PAUSED_BY_MAX" if self._scan_auto_paused
                else state
            )
            await self.reply(
                "🔎 SCAN STATUS\n\n"
                f"User State: {state}\n"
                f"Runtime State: {runtime}\n"
                f"H4 Mode: {'ON' if self._h4_enabled else 'OFF'}\n"
                f"H4 Window: {self._h4_window_text()}\n"
                + (f"Next H4: {self._h4_next_slot_text()}\n" if self._h4_enabled else "")
                + f"Threshold: {decimal_to_str(self.scan_threshold)}\n"
                f"Max Active Trade: {self.max_active_trades}\n"
                f"Max Pair/Cycle: {SCAN_MAX_PAIRS_PER_CYCLE}\n"
                f"Interval Cycle: {SCAN_CYCLE_DELAY_SECONDS:g}s\n"
                f"Delay Pair: {SCAN_PAIR_DELAY_SECONDS:g}s"
            )
            return

        mode = parts[1].lower()
        if mode not in {"on", "off"}:
            raise ValueError("Gunakan /scan on atau /scan off.")

        if mode == "off":
            self._scan_user_enabled = False
            self._scan_auto_paused = False
            self._h4_window_active = False
            self._scan_generation += 1
            await self._stop_scan_task()
            await self.reply("🛑 SCAN OFF")
            return

        self._scan_user_enabled = True
        self._scan_auto_paused = False
        if self._h4_enabled:
            if len(self.active_trades) >= self.max_active_trades:
                self._scan_auto_paused = True
            await self._start_scan_task()
            await self.reply(
                "🟢 SCAN ON\n\n"
                "H4 Mode: ON\n"
                f"Window: {self._h4_window_text()}\n"
                f"Next H4: {self._h4_next_slot_text()}\n"
                f"Threshold: {decimal_to_str(self.scan_threshold)}\n"
                f"Max Active Trade: {self.max_active_trades}"
            )
            return

        if len(self.active_trades) >= self.max_active_trades:
            self._scan_auto_paused = True
            await self._stop_scan_task()
            await self.reply(
                "⏸️ SCAN belum berjalan karena MAX ACTIVE TRADE tercapai.\n"
                f"Max: {self.max_active_trades}"
            )
            return

        await self._start_scan_task()
        await self.reply(
            "🟢 SCAN ON\n\n"
            f"Threshold: {decimal_to_str(self.scan_threshold)}\n"
            f"Max Active Trade: {self.max_active_trades}"
        )

    # --------------------------------------------------------
    # AUTOSTOP (trailing max drawdown equity, hanya saat REAL ON)
    # --------------------------------------------------------

    @staticmethod
    def _equity_from_balance_row(row: dict[str, Any]) -> Decimal:
        # Autostop uses the cached Binance wallet balance as its fixed base and
        # adds current local REAL temporary PnL separately. This avoids double
        # counting Binance crossUnPnl while still remaining REST-independent
        # during a rate-limit window.
        return parse_signed_decimal(str(row.get("balance") or "0"))

    def _read_equity_cached(self) -> Decimal:
        balance = self.real.cached_usdt_balance()
        if balance is None:
            raise BinanceAPIError(
                "Snapshot USDT Binance belum tersedia. /real harus berhasil diaktifkan terlebih dahulu.",
                endpoint="CACHE:/fapi/v3/balance",
            )
        return self._equity_from_balance_row(balance)

    def _temporary_real_pnl_usdt(self) -> Decimal:
        """Hitung unrealized PnL lokal dari REAL FILLED trades dan harga live."""
        total = Decimal("0")
        for trade in self.active_trades.values():
            if trade.status != "FILLED" or not trade.real_enabled or trade.quantity is None:
                continue
            snapshot = self.prices.get(trade.pair)
            fill = trade.fill_price or trade.entry
            if snapshot is None or fill <= 0 or trade.quantity <= 0:
                continue
            if trade.direction == "BUY":
                total += (snapshot.price - fill) * trade.quantity
            else:
                total += (fill - snapshot.price) * trade.quantity
        return total

    def _current_equity_cached(self) -> Decimal:
        balance = self.real.cached_usdt_balance()
        if balance is None:
            raise BinanceAPIError(
                "Snapshot USDT Binance belum tersedia.",
                endpoint="CACHE:/fapi/v3/balance",
            )
        wallet = parse_signed_decimal(str(balance.get("balance") or "0"))
        return wallet + self._temporary_real_pnl_usdt()

    @staticmethod
    def _fmt_usd(value: Decimal | None) -> str:
        if value is None:
            return "-"
        return decimal_to_str(value.quantize(Decimal("0.0001"))) or "0"

    def _autostop_floor(self) -> Decimal | None:
        if self.autostop_percent is None or self._equity_peak is None:
            return None
        return self._equity_peak * (Decimal("100") - self.autostop_percent) / Decimal("100")

    def _autostop_short(self) -> str:
        if self.autostop_percent is None:
            return "OFF (atur dengan /autostop [persen])"
        state = "TERPICU" if self._autostop_triggered else "aktif"
        return f"{decimal_to_str(self.autostop_percent)}% dari puncak equity ({state})"

    def _autostop_status_text(self) -> str:
        lines = ["🛡️ AUTOSTOP", ""]
        if self.autostop_percent is None:
            lines.append("Threshold: OFF")
        else:
            lines.append(f"Threshold: {decimal_to_str(self.autostop_percent)}% dari puncak equity")
        if not self.real_mode:
            lines.append("Real Mode: OFF (autostop bekerja hanya saat /real on)")
            return "\n".join(lines)
        lines.append("Real Mode: ON")
        lines.append(f"Peak Equity: {self._fmt_usd(self._equity_peak)} USDT")
        lines.append(f"Equity Terakhir: {self._fmt_usd(self._equity_last)} USDT")
        floor = self._autostop_floor()
        if floor is not None:
            lines.append(f"Batas Stop: {self._fmt_usd(floor)} USDT")
        lines.append(f"Status: {'TERPICU (scan dimatikan)' if self._autostop_triggered else 'AMAN'}")
        return "\n".join(lines)

    def _start_autostop_task(self) -> None:
        if not self._running:
            return
        if self._autostop_task is not None and not self._autostop_task.done():
            return
        self._autostop_task = asyncio.create_task(
            self._autostop_loop(),
            name="main-autostop-loop",
        )

    async def _stop_autostop_task(self) -> None:
        task = self._autostop_task
        self._autostop_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _autostop_loop(self) -> None:
        while self._running and self.real_mode:
            try:
                await asyncio.sleep(AUTOSTOP_CHECK_SECONDS)
                await self._autostop_check()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("[AUTOSTOP] check error")

    async def _autostop_check(self) -> None:
        # Autostop is intentionally REST-independent. It uses the latest cached
        # Binance wallet balance plus current WebSocket-derived temporary PnL.
        if not self.real_mode:
            return
        try:
            equity = self._current_equity_cached()
        except Exception as exc:
            self._autostop_failures += 1
            if self._autostop_failures == 3:
                log.warning("[AUTOSTOP] snapshot equity lokal belum tersedia 3x berturut-turut: %s", exc)
            return
        self._autostop_failures = 0
        self._equity_last = equity
        # Puncak hanya naik; turun tidak mengubah puncak.
        if self._equity_peak is None or equity > self._equity_peak:
            self._equity_peak = equity
        floor = self._autostop_floor()
        if floor is None or self._autostop_triggered or equity > floor:
            return

        self._autostop_triggered = True
        self._scan_user_enabled = False
        self._scan_auto_paused = False
        drawdown = (self._equity_peak - equity) / self._equity_peak * Decimal("100")
        log.info("[AUTOSTOP] terpicu: equity=%s peak=%s", equity, self._equity_peak)
        await self.reply(
            "🛑 AUTOSTOP TERPICU\n\n"
            f"Equity turun {decimal_to_str(drawdown.quantize(Decimal('0.01')))}% dari puncak "
            f"(batas {decimal_to_str(self.autostop_percent)}%).\n"
            f"Peak: {self._fmt_usd(self._equity_peak)} USDT\n"
            f"Sekarang: {self._fmt_usd(equity)} USDT\n"
            f"Batas: {self._fmt_usd(floor)} USDT\n\n"
            "SCAN dimatikan; cycle yang sedang berjalan tidak memasukkan setup baru.\n"
            "Posisi dan order yang sudah ada tidak ditutup.\n\n"
            "Mulai lagi: /autostop [persen] lalu /scan on"
        )

    async def _autostop_rebaseline(self) -> None:
        equity = self._current_equity_cached()
        self._equity_peak = equity
        self._equity_last = equity
        self._autostop_triggered = False
        self._autostop_failures = 0

    async def _handle_autostop_command(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            await self.reply(self._autostop_status_text())
            return

        arg = parts[1].lower()
        if arg == "off":
            self.autostop_percent = None
            await self.reply("🛑 AUTOSTOP OFF")
            return
        if arg == "reset":
            if not self.real_mode:
                raise ValueError("Autostop bekerja hanya saat /real on.")
            await self._autostop_rebaseline()
            await self.reply("♻️ Puncak equity diset ulang ke saldo sekarang.\n\n" + self._autostop_status_text())
            return

        try:
            value = Decimal(arg.replace(",", "."))
        except InvalidOperation:
            raise ValueError("Gunakan /autostop [persen], /autostop off, atau /autostop reset.")
        if not value.is_finite() or value <= 0 or value >= 100:
            raise ValueError("Autostop harus di antara 0 dan 100 (persen).")

        self.autostop_percent = value
        if self.real_mode and self._autostop_triggered:
            await self._autostop_rebaseline()
        suffix = (
            "" if self.real_mode
            else "\n\nBaseline saldo diambil saat /real on."
        )
        await self.reply(
            f"✅ AUTOSTOP = {decimal_to_str(value)}%\n\n"
            + self._autostop_status_text()
            + suffix
        )

    async def _handle_threshold_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            await self.reply(f"🎚 Threshold saat ini: {decimal_to_str(self.scan_threshold)}")
            return
        value = Decimal(parts[1])
        if value < 0 or value > 100:
            raise ValueError("Threshold harus 0–100.")
        self.scan_threshold = value
        await self.reply(f"✅ Threshold diubah menjadi {decimal_to_str(value)}")

    async def _handle_max_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            await self.reply(f"📌 Max Active Trade: {self.max_active_trades}")
            return
        value = int(parts[1])
        if value <= 0:
            raise ValueError("/max harus berupa angka > 0.")
        self.max_active_trades = value

        if self._h4_enabled:
            # Dalam H4 mode scheduler tetap dipertahankan; window aktif hanya
            # boleh berjalan jika capacity tersedia. Jika menjadi penuh, window
            # ditutup dan slot berikutnya menjadi trigger selanjutnya.
            if len(self.active_trades) >= self.max_active_trades:
                self._h4_window_active = False
                self._scan_auto_paused = bool(self._scan_user_enabled)
            else:
                self._scan_auto_paused = False
            if self._scan_user_enabled:
                await self._start_scan_task()
            await self.reply(
                f"✅ /max = {value}\n"
                f"H4 Mode: ON | Window: {self._h4_window_text()}\n"
                f"Active: {len(self.active_trades)}/{self.max_active_trades}\n"
                f"Next H4: {self._h4_next_slot_text()}"
            )
            return

        if len(self.active_trades) >= self.max_active_trades:
            if self._scan_user_enabled:
                self._scan_auto_paused = True
                await self._stop_scan_task()
            await self.reply(
                f"✅ /max = {value}\n"
                "SCAN akan pause otomatis karena kapasitas sudah penuh."
            )
            return
        was_paused = self._scan_auto_paused
        self._scan_auto_paused = False
        if self._scan_user_enabled and (was_paused or self._scan_task is None):
            await self._start_scan_task()
        await self.reply(f"✅ /max = {value}")

    async def _stop_scan_task(self) -> None:
        task = self._scan_task
        self._scan_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _start_scan_task(self) -> None:
        if not self._running:
            return
        if not self._scan_user_enabled:
            return
        if not self._h4_enabled and len(self.active_trades) >= self.max_active_trades:
            self._scan_auto_paused = True
            return
        if self._scan_task is not None and not self._scan_task.done():
            return
        # H4 mode may legitimately keep a scheduler task alive while full so
        # it can wait for the next close and re-evaluate capacity there.
        if not self._h4_enabled:
            self._scan_auto_paused = False
        self._scan_task = asyncio.create_task(
            self._scan_loop(),
            name="main-scan-loop",
        )
        log.info("[SCAN] scanner task started | H4=%s window=%s", self._h4_enabled, self._h4_window_active)

    async def _resume_scan_after_capacity_change(self) -> None:
        if self.flow is not None:
            return
        if not self._scan_user_enabled:
            return
        if self._h4_enabled:
            # H4 scheduler owns the lifecycle. Never launch a fresh scan outside
            # an active H4 window merely because capacity became available.
            if self._scan_task is None or self._scan_task.done():
                await self._start_scan_task()
            return
        if len(self.active_trades) < self.max_active_trades:
            self._scan_auto_paused = False
            await self._start_scan_task()

    # --------------------------------------------------------
    # STRATEGY BRIDGE FOR SCAN
    # --------------------------------------------------------

    async def _load_strategy_runtime(self):
        strategy_path = BASE_DIR / "strategy.py"
        if not strategy_path.is_file():
            raise RuntimeError(f"strategy.py tidak ditemukan: {strategy_path}")
        module_name = f"trading_strategy_scan_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(module_name, strategy_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Loader strategy.py tidak tersedia.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        previous = getattr(self, "_strategy_runtime_module", None)
        if previous is not None:
            sys.modules.pop(getattr(previous, "__name__", ""), None)
        self._strategy_runtime_module = module
        return module

    async def _call_strategy_function(self, module, name: str, *args, **kwargs):
        fn = getattr(module, name, None)
        if not callable(fn):
            return None
        if inspect.iscoroutinefunction(fn):
            result = await fn(*args, **kwargs)
        else:
            result = await asyncio.to_thread(fn, *args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _extract_structure_trend(self, analysis: dict[str, Any] | None, key: str) -> str | None:
        if not isinstance(analysis, dict):
            return None
        direct = analysis.get(key)
        if isinstance(direct, dict) and direct.get("trend"):
            return str(direct.get("trend")).upper()
        macro = analysis.get("macro")
        if isinstance(macro, dict):
            node = macro.get(key)
            if isinstance(node, dict) and node.get("trend"):
                return str(node.get("trend")).upper()
        mtf = analysis.get("multi_timeframe")
        if isinstance(mtf, dict) and key == "pair_h4":
            node = mtf.get("h4")
            if isinstance(node, dict) and node.get("trend"):
                return str(node.get("trend")).upper()
        return None

    def _make_scan_context(self, pair: str, cycle_number: int) -> dict[str, Any]:
        return {
            "pair": pair,
            "session_id": self.session_id,
            "mode": "SCAN",
            "scan_cycle": cycle_number,
            "market": "USDT Futures",
            "primary_data_source": "BYBIT_PUBLIC",
            "data_provider": "BYBIT_PUBLIC_ONLY",
            "allow_binance_fallback": False,
            "timeframe": "15m",
            "candles_requested": 672,
            "notes": copy.deepcopy(self.notes),
            "engine_version": "SCAN_BRIDGE_1",
        }

    async def _scan_generate_for_pair(self, module, pair: str, cycle_number: int) -> dict[str, Any]:
        result = await self._call_strategy_function(
            module,
            "generate_setup",
            pair,
            self._make_scan_context(pair, cycle_number),
        )
        normalized = self._normalize_auto_result(result, pair)
        data_source = str(normalized.get("data_source") or "").upper()
        analysis = normalized.get("raw_result") if isinstance(normalized.get("raw_result"), dict) else {}
        data_info = analysis.get("data") if isinstance(analysis, dict) else {}
        if not data_source.startswith("BYBIT"):
            raise RuntimeError(
                f"SCAN menolak {pair}: strategy mengembalikan data source {data_source or '-'}, bukan BYBIT PUBLIC."
            )
        if bool(data_info.get("fallback_used")):
            raise RuntimeError(
                f"SCAN menolak {pair}: strategy melaporkan fallback data aktif."
            )
        sources = data_info.get("sources_by_timeframe")
        if isinstance(sources, dict):
            non_bybit = [
                f"{key}={value}"
                for key, value in sources.items()
                if value and not str(value).upper().startswith("BYBIT")
            ]
            if non_bybit:
                raise RuntimeError(
                    f"SCAN menolak {pair}: ada timeframe bukan BYBIT PUBLIC ({', '.join(non_bybit)})."
                )
        return normalized

    async def _scan_get_structure(self, module, pair: str, cycle_number: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
        # Optimized contract expected in the next strategy revision:
        # analyze_scan_structure(pair, context) -> {trend, analysis...}
        for name in ("analyze_scan_structure", "get_scan_structure", "analyze_structure"):
            result = await self._call_strategy_function(
                module,
                name,
                pair,
                self._make_scan_context(pair, cycle_number),
            )
            if result is not None:
                if not isinstance(result, dict):
                    raise ValueError(f"strategy.{name}() harus mengembalikan dict.")
                return result, None

        # Backward-compatible fallback: current strategy already returns pair_h4 and btc_h4
        # inside analysis, so one full generation can serve as structure + candidate setup.
        log.info(
            "[SCAN] structure API optimized belum tersedia; fallback generate_setup(%s)",
            pair,
        )
        normalized = await self._scan_generate_for_pair(module, pair, cycle_number)
        return {
            "trend": self._extract_structure_trend(normalized.get("analysis"), "pair_h4"),
            "btc_h4_trend": self._extract_structure_trend(normalized.get("analysis"), "btc_h4"),
            "analysis": normalized.get("analysis") or {},
        }, normalized

    def _normalize_validator_result(
        self,
        raw_result: Any,
        pair: str,
        initial: dict[str, Any],
    ) -> tuple[dict[str, Any], bool, str]:
        if not isinstance(raw_result, dict):
            raise ValueError("strategy.validate_setup() harus mengembalikan dict.")
        payload = copy.deepcopy(raw_result)
        replacement = payload.get("setup")
        if replacement is None and all(k in payload for k in ("direction", "entry", "price_exp", "sl", "tp")):
            replacement = payload
        if replacement is None:
            replacement = copy.deepcopy(initial)
        elif not isinstance(replacement, dict):
            raise ValueError("Field setup validator harus berupa dict.")
        else:
            merged = copy.deepcopy(initial)
            merged.update(replacement)
            replacement = merged
        replacement["pair"] = pair
        if replacement.get("price_now_reference") in (None, ""):
            replacement["price_now_reference"] = initial.get("price_now_reference")
        if not replacement.get("entry_reason"):
            replacement["entry_reason"] = initial.get("entry_reason")
        if not replacement.get("price_exp_reason"):
            replacement["price_exp_reason"] = initial.get("price_exp_reason")
        if not replacement.get("sl_reason"):
            replacement["sl_reason"] = initial.get("sl_reason")
        if not replacement.get("tp_reason"):
            replacement["tp_reason"] = initial.get("tp_reason")
        if "confidence" not in payload:
            replacement["confidence"] = initial["confidence"]
        else:
            replacement["confidence"] = payload["confidence"]
        valid = bool(payload.get("valid", True))
        reason = str(payload.get("reason") or payload.get("validation_reason") or "").strip()
        return replacement, valid, reason

    async def _scan_validate_candidate(self, module, candidate: dict[str, Any], cycle_number: int) -> dict[str, Any] | None:
        raw = await self._call_strategy_function(
            module,
            "validate_setup",
            candidate["pair"],
            copy.deepcopy(candidate),
            self._make_scan_context(candidate["pair"], cycle_number),
        )
        if raw is None:
            log.info(
                "[SCAN] Validator strategy belum tersedia untuk %s; candidate tidak dimasukkan /trade.",
                candidate["pair"],
            )
            return None

        replacement, valid, reason = self._normalize_validator_result(
            raw,
            candidate["pair"],
            candidate,
        )
        replacement = self._normalize_auto_result(replacement, candidate["pair"])
        initial_conf = Decimal(str(candidate["confidence"]))
        final_conf = Decimal(str(replacement["confidence"]))
        tolerance_floor = initial_conf * SCAN_VALIDATION_TOLERANCE
        within_tolerance = final_conf >= tolerance_floor
        if not valid or not within_tolerance:
            log.info(
                "[SCAN] validator reject %s | initial=%s final=%s floor=%s valid=%s reason=%s",
                candidate["pair"], decimal_to_str(initial_conf), decimal_to_str(final_conf),
                decimal_to_str(tolerance_floor), valid, reason or "-",
            )
            return None

        initial_analysis = candidate.get("analysis") if isinstance(candidate.get("analysis"), dict) else {}
        final_analysis = replacement.get("analysis") if isinstance(replacement.get("analysis"), dict) else {}
        merged_analysis = {
            "scan": {
                "initial_confidence": decimal_to_str(initial_conf),
                "validated_confidence": decimal_to_str(final_conf),
                "validation_ratio_percent": decimal_to_str(
                    (final_conf / initial_conf * Decimal("100")) if initial_conf else Decimal("0")
                ),
                "validation_tolerance_percent": "90",
                "validation_reason": reason or "Validator menyatakan setup valid.",
                "setup_replaced": replacement != candidate,
                "scan_cycle": cycle_number,
            },
            **initial_analysis,
            "validator": final_analysis,
        }
        replacement["analysis"] = merged_analysis
        return replacement

    # --------------------------------------------------------
    # SCAN UNIVERSE / MARGIN FILTER
    # --------------------------------------------------------

    async def _build_scan_universe(self) -> dict[str, Any]:
        await self._purge_expired_bans()
        # Symbol rules can change while the process is alive; refresh once per cycle.
        self.symbols = await self.rest.get_exchange_info()
        binance_tickers = await self.rest.get_24h_tickers()
        bybit_symbols = await self.bybit.get_linear_perpetual_symbols()
        bybit_tickers = await self.bybit.get_linear_tickers()

        bn: dict[str, dict[str, Any]] = {}
        for item in binance_tickers:
            pair = normalize_symbol(str(item.get("symbol") or ""))
            if pair not in self.symbols:
                continue
            if pair not in bybit_symbols:
                continue
            quote_volume = item.get("quoteVolume")
            if quote_volume in (None, ""):
                quote_volume = item.get("volume")
            try:
                volume = Decimal(str(quote_volume or "0"))
            except InvalidOperation:
                volume = Decimal("0")
            bn[pair] = {
                "volume_quote": volume,
                "price": Decimal(str(item.get("lastPrice") or "0")),
            }

        by: dict[str, dict[str, Any]] = {}
        for item in bybit_tickers:
            pair = normalize_symbol(str(item.get("symbol") or ""))
            if pair not in bybit_symbols:
                continue
            try:
                turnover = Decimal(str(item.get("turnover24h") or "0"))
            except InvalidOperation:
                turnover = Decimal("0")
            try:
                price = Decimal(str(item.get("lastPrice") or "0"))
            except InvalidOperation:
                price = Decimal("0")
            by[pair] = {"volume_quote": turnover, "price": price}

        common = sorted(set(bn) & set(by))
        rows = []
        for pair in common:
            combined = bn[pair]["volume_quote"] + by[pair]["volume_quote"]
            rows.append({
                "pair": pair,
                "binance_volume_quote": bn[pair]["volume_quote"],
                "bybit_volume_quote": by[pair]["volume_quote"],
                "combined_volume_quote": combined,
                "binance_price": bn[pair]["price"],
                "bybit_price": by[pair]["price"],
            })
        rows.sort(key=lambda x: x["combined_volume_quote"], reverse=True)
        return {
            "rows": rows,
            "binance_count": len(bn),
            "bybit_count": len(by),
            "common_count": len(rows),
        }

    def _margin_config(self) -> tuple[Decimal | None, Decimal | None]:
        return self.margin_usdt, Decimal(self.leverage)

    def _margin_influence_reason(self, pair: str, price: Decimal) -> str | None:
        margin, leverage = self._margin_config()
        if margin is None or leverage is None or price <= 0:
            return None
        meta = self.symbols.get(pair)
        if meta is None:
            return None
        target = margin * leverage
        low_target = target * (Decimal("1") - SCAN_MARGIN_TOLERANCE)
        high_target = target * (Decimal("1") + SCAN_MARGIN_TOLERANCE)
        if target <= 0:
            return None

        raw_qty = target / price
        candidates = []
        if meta.step_size > 0:
            floor_qty = (raw_qty / meta.step_size).to_integral_value(rounding=ROUND_DOWN) * meta.step_size
            ceil_qty = floor_qty + meta.step_size
            candidates.extend([floor_qty, ceil_qty])
        else:
            candidates.append(raw_qty)

        best_notional: Decimal | None = None
        for qty in candidates:
            if qty <= 0:
                continue
            if meta.min_qty > 0 and qty < meta.min_qty:
                continue
            if meta.max_qty > 0 and qty > meta.max_qty:
                continue
            notional = qty * price
            if meta.min_notional > 0 and notional < meta.min_notional:
                continue
            if low_target <= notional <= high_target:
                return None
            distance = abs(notional - target)
            if best_notional is None or distance < abs(best_notional - target):
                best_notional = notional

        if best_notional is None:
            return (
                f"Tidak ada quantity legal pada target notional {decimal_to_str(target)} "
                f"(toleransi ±{decimal_to_str(SCAN_MARGIN_TOLERANCE*100)}%)."
            )
        return (
            f"Quantity terdekat menghasilkan notional {decimal_to_str(best_notional)}, "
            f"di luar target {decimal_to_str(target)} ±{decimal_to_str(SCAN_MARGIN_TOLERANCE*100)}%."
        )

    async def _process_margin_filter(self, pair: str, price: Decimal) -> tuple[bool, str | None]:
        reason = self._margin_influence_reason(pair, price)
        if reason is None:
            self._scan_margin_streak[pair] = 0
            return False, None
        streak = self._scan_margin_streak.get(pair, 0) + 1
        self._scan_margin_streak[pair] = streak
        log.info(
            "[SCAN] Margin Influence candidate %s streak=%s/%s | %s",
            pair, streak, SCAN_MARGIN_FAIL_CONFIRMATIONS, reason,
        )
        if streak >= SCAN_MARGIN_FAIL_CONFIRMATIONS:
            await self._ban_pair(
                pair,
                hours=None,
                reason=(
                    f"Margin Influence terdeteksi {streak} cycle berturut-turut; "
                    f"{reason}"
                ),
                source="AUTO_MARGIN_INFLUENCE",
            )
            return True, reason
        return True, f"Konfirmasi margin {streak}/{SCAN_MARGIN_FAIL_CONFIRMATIONS}: {reason}"

    # --------------------------------------------------------
    # SCAN LOOP
    # --------------------------------------------------------

    async def _wait_for_h4_trigger(self) -> datetime | None:
        """Wait until the next H4 close, while remaining responsive to /H4 off."""
        while self._running and self._scan_user_enabled and self._h4_enabled and not self._h4_window_active:
            due = self._h4_trigger_slot()
            if due is not None:
                slot_key = self._h4_slot_key(due)
                self._h4_last_trigger_slot = slot_key
                return due

            next_slot = self._h4_next_scheduled_slot()
            wait_seconds = max(0.25, (next_slot - now_utc().astimezone(TZ)).total_seconds())
            # Short sleep chunks allow /H4 off, /scan off, and max/capacity
            # state changes to be noticed without waiting for the full interval.
            await asyncio.sleep(min(wait_seconds, H4_SCAN_SCHEDULER_POLL_SECONDS))
        return None

    async def _scan_loop(self) -> None:
        while self._running and self._scan_user_enabled:
            try:
                if self._h4_enabled:
                    if not self._h4_window_active:
                        slot = await self._wait_for_h4_trigger()
                        if slot is None:
                            break
                        if len(self.active_trades) >= self.max_active_trades:
                            self._scan_auto_paused = True
                            log.info(
                                "[SCAN][H4] slot %s dilewati karena MAX active=%s/%s; tunggu slot berikutnya.",
                                self._h4_slot_key(slot),
                                len(self.active_trades),
                                self.max_active_trades,
                            )
                            continue
                        self._h4_window_active = True
                        self._h4_window_slot = slot
                        self._h4_window_deadline = self._h4_slot_deadline(slot)
                        self._h4_invalid_slot_key = None
                        self._h4_opening_gate = True
                        self._scan_auto_paused = False
                        log.info(
                            "[SCAN][H4] scan window dibuka pada slot %s | deadline=%s",
                            self._h4_slot_key(slot),
                            self._h4_window_deadline.astimezone(TZ).strftime("%d-%m-%Y %H:%M:%S WIB"),
                        )

                # Dalam mode normal, capacity full menghentikan task seperti perilaku lama.
                if not self._h4_enabled and len(self.active_trades) >= self.max_active_trades:
                    self._scan_auto_paused = True
                    break

                while self._running and self._scan_user_enabled and self.real.cooldown_remaining > 0:
                    if self._h4_enabled and self._h4_window_expired():
                        self._invalidate_h4_window("Binance rate-limit/cooldown melewati +1 jam dari jadwal H4.")
                        break
                    wait_for_cooldown = self.real.cooldown_remaining
                    wait_for_h4 = self._h4_window_remaining_seconds() if self._h4_enabled else None
                    waits = [wait_for_cooldown, 30.0]
                    if wait_for_h4 is not None:
                        waits.append(max(0.25, wait_for_h4))
                    await asyncio.sleep(max(0.25, min(waits)))
                if not (self._running and self._scan_user_enabled):
                    break
                if self._h4_enabled and not self._h4_window_active:
                    continue
                if self._h4_enabled and self._h4_opening_gate and self._h4_window_expired():
                    self._invalidate_h4_window("Slot H4 sudah melewati batas valid +1 jam sebelum scan dimulai.")
                    continue

                # H4 can be disabled while sleeping on a scheduler boundary;
                # after the state flips, the next iteration becomes normal scan.
                await self._run_scan_cycle()

                # Satu cycle berhasil dimulai/selesai tanpa rate-limit fatal:
                # opening gate selesai, sehingga window berjalan terus sampai MAX.
                if self._h4_enabled and self._h4_window_active and self._h4_opening_gate:
                    self._h4_opening_gate = False
                    self._h4_window_deadline = None

                if len(self.active_trades) >= self.max_active_trades:
                    if self._h4_enabled:
                        self._h4_window_active = False
                        self._h4_window_slot = None
                        self._h4_window_deadline = None
                        self._h4_opening_gate = False
                        self._scan_auto_paused = True
                        log.info(
                            "[SCAN][H4] window slot %s selesai: MAX active=%s/%s.",
                            self._h4_last_trigger_slot or "-",
                            len(self.active_trades),
                            self.max_active_trades,
                        )
                        continue
                    self._scan_auto_paused = True
                    break

                if not (self._running and self._scan_user_enabled):
                    break

                # In H4 mode, keep scanning inside the active window. In normal
                # mode retain the original cycle delay.
                await asyncio.sleep(SCAN_CYCLE_DELAY_SECONDS)
            except asyncio.CancelledError:
                raise
            except BinanceRateLimitError as exc:
                self.real.apply_cooldown(exc.server_cooldown_seconds, exc.bot_cooldown_seconds)
                self._notify_rate_limit(exc, "SCAN")
                continue
            except Exception:
                log.exception("[SCAN] cycle fatal error")
                # A fatal cycle must not silently bypass an active H4 gate.
                await asyncio.sleep(SCAN_CYCLE_DELAY_SECONDS)

    async def _run_scan_cycle(self) -> None:
        if len(self.active_trades) >= self.max_active_trades:
            self._scan_auto_paused = True
            log.info("[SCAN] auto-pause: max active trade reached (%s)", self.max_active_trades)
            return

        self._scan_cycle_number += 1
        cycle = self._scan_cycle_number
        scan_generation = self._scan_generation
        started = time.monotonic()
        log.info(
            "[SCAN] Cycle #%s started | generation=%s | H4=%s window=%s",
            cycle, scan_generation, self._h4_enabled, self._h4_window_active,
        )
        await self.reply(
            card(
                f"🛰️ SCAN #{cycle} DIMULAI",
                [
                    "📡 Regime BTC → universe Binance ∩ Bybit",
                    f"🎯 Batch maksimum {SCAN_MAX_PAIRS_PER_CYCLE} pair",
                    "⏳ Memindai...",
                ],
            )
        )

        module = await self._load_strategy_runtime()
        if scan_generation != self._scan_generation or not self._scan_user_enabled:
            log.info("[SCAN] Cycle #%s dibatalkan sebelum analisis: generation berubah/off.", cycle)
            return
        btc_raw, btc_fallback = await self._scan_get_structure(module, "BTCUSDT", cycle)
        btc_analysis = btc_raw.get("analysis") if isinstance(btc_raw, dict) else {}
        btc_trend = str(
            btc_raw.get("trend")
            or self._extract_structure_trend(btc_analysis, "btc_h4")
            or "UNKNOWN"
        ).upper()
        if btc_fallback is not None:
            btc_analysis = btc_fallback.get("analysis") or btc_analysis
            btc_trend = self._extract_structure_trend(btc_analysis, "btc_h4") or btc_trend

        log.info("[SCAN] Cycle #%s BTC H4 trend=%s", cycle, btc_trend)

        universe = await self._build_scan_universe()
        rows = universe["rows"]
        already_trade = 0
        banned_count = 0
        margin_blocked = 0
        eligible: list[str] = []

        active_pairs = {trade.pair for trade in self.active_trades.values()}
        for row in rows:
            pair = row["pair"]
            if pair == "BTCUSDT":
                continue
            if pair in active_pairs:
                already_trade += 1
                continue
            if self._is_pair_banned(pair):
                banned_count += 1
                continue

            # Margin/quantity legality follows the Binance execution symbol,
            # therefore prefer Binance ticker price and only fall back to Bybit
            # when Binance did not return a usable price.
            price = row.get("binance_price") or row.get("bybit_price") or Decimal("0")
            try:
                margin_block, margin_reason = await self._process_margin_filter(pair, price)
            except Exception:
                log.exception("[SCAN] margin filter gagal %s", pair)
                margin_block = False
                margin_reason = None
            if margin_block:
                margin_blocked += 1
                continue

            eligible.append(pair)

        log.info(
            "[SCAN] Cycle #%s universe common=%s already_trade=%s banned=%s margin_blocked=%s eligible=%s top_batch=%s",
            cycle,
            universe["common_count"],
            already_trade,
            banned_count,
            margin_blocked,
            len(eligible),
            min(SCAN_MAX_PAIRS_PER_CYCLE, len(eligible)),
        )

        # Satu cycle hanya boleh melakukan full structure/setup analysis
        # terhadap maksimal 50 pair. Karena rows sudah diurutkan dari volume
        # terbesar ke terkecil, eligible[:50] adalah batch top-volume cycle ini.
        batch = eligible[:SCAN_MAX_PAIRS_PER_CYCLE]
        deferred = max(0, len(eligible) - len(batch))
        volume_rank_by_pair = {
            row["pair"]: index
            for index, row in enumerate(rows, start=1)
        }

        directional: list[str] = []
        threshold_candidates: list[dict[str, Any]] = []
        rejected_structure = 0
        analysis_errors = 0
        threshold_banned = 0
        structure_banned = 0
        scanned = 0
        direction_counts: dict[str, int] = {
            "BULLISH": 0,
            "BEARISH": 0,
            "RANGE": 0,
            "UNKNOWN": 0,
        }

        if btc_trend not in {"BULLISH", "BEARISH", "RANGE"}:
            log.error(
                "[SCAN] Cycle #%s dibatalkan: BTC H4 trend tidak valid (%s)",
                cycle,
                btc_trend,
            )
        else:
            for index, pair in enumerate(batch, start=1):
                if scan_generation != self._scan_generation or not self._scan_user_enabled:
                    log.info("[SCAN] Cycle #%s dibatalkan di tengah batch: generation berubah/off.", cycle)
                    return
                if len(self.active_trades) >= self.max_active_trades:
                    self._scan_auto_paused = True
                    log.info("[SCAN] Cycle #%s stopped early at max active trade.", cycle)
                    break

                scanned += 1
                log.info(
                    "[SCAN] Cycle #%s pair %s/%s (volume-rank=%s): %s",
                    cycle,
                    index,
                    len(batch),
                    volume_rank_by_pair.get(pair, index),
                    pair,
                )

                try:
                    structure, reused_setup = await self._scan_get_structure(module, pair, cycle)
                    pair_trend = str(structure.get("trend") or "UNKNOWN").upper()
                    direction_counts[pair_trend if pair_trend in direction_counts else "UNKNOWN"] += 1

                    aligned = (
                        (btc_trend == "BULLISH" and pair_trend == "BULLISH")
                        or (btc_trend == "BEARISH" and pair_trend == "BEARISH")
                        or (btc_trend == "RANGE" and pair_trend in {"BULLISH", "BEARISH", "RANGE"})
                    )
                    if not aligned:
                        rejected_structure += 1
                        structure_banned += 1
                        await self._ban_pair(
                            pair,
                            hours=SCAN_BANNED_TP_SL_HOURS,
                            reason=(
                                f"H4 structure {pair_trend} tidak searah dengan BTC H4 {btc_trend}; "
                                "pair dikeluarkan dari batch scanner."
                            ),
                            source="AUTO_STRUCTURE_MISMATCH",
                        )
                        log.info(
                            "[SCAN] %s rejected + banned 24h | pair=%s BTC=%s",
                            pair,
                            pair_trend,
                            btc_trend,
                        )
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue

                    directional.append(pair)

                    candidate = reused_setup
                    if candidate is None:
                        candidate = await self._scan_generate_for_pair(module, pair, cycle)

                    candidate["analysis"] = candidate.get("analysis") or {}
                    candidate["analysis"].setdefault("scan", {})
                    candidate["analysis"]["scan"].update({
                        "cycle": cycle,
                        "btc_h4_trend": btc_trend,
                        "pair_h4_trend": pair_trend,
                        "batch_rank": index,
                        "batch_limit": SCAN_MAX_PAIRS_PER_CYCLE,
                    })

                    confidence = Decimal(str(candidate["confidence"]))
                    if confidence < self.scan_threshold:
                        threshold_banned += 1
                        gates = (
                            ((candidate.get("analysis") or {}).get("selected_setup") or {}).get("evidence")
                            or {}
                        ).get("gates") or {}
                        waiting = bool(gates.get("waiting"))
                        ban_hours = SCAN_BANNED_WAITING_HOURS if waiting else SCAN_BANNED_PRICE_EXP_HOURS
                        await self._ban_pair(
                            pair,
                            hours=ban_hours,
                            reason=(
                                f"Confidence {decimal_to_str(confidence)} di bawah threshold "
                                f"{decimal_to_str(self.scan_threshold)}."
                                + (" Menunggu trigger RSI." if waiting else "")
                            ),
                            source="AUTO_BELOW_THRESHOLD",
                        )
                        log.info(
                            "[SCAN] %s below threshold + banned %sh | confidence=%s threshold=%s",
                            pair,
                            decimal_to_str(ban_hours),
                            decimal_to_str(confidence),
                            decimal_to_str(self.scan_threshold),
                        )
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue

                    threshold_candidates.append(candidate)
                    log.info(
                        "[SCAN] %s threshold PASS confidence=%s",
                        pair,
                        decimal_to_str(confidence),
                    )
                except Exception as exc:
                    analysis_errors += 1
                    if getattr(exc, "bybit_rate_limited", False):
                        retry_after = getattr(exc, "retry_after_seconds", 0.0)
                        log.warning(
                            "[SCAN] analysis %s dilewati setelah Bybit rate-limit backoff | retry_after=%.2fs | %s",
                            pair,
                            float(retry_after or 0.0),
                            exc,
                        )
                    elif getattr(exc, "insufficient_history", False):
                        # Listing baru / candle kurang: ban sementara, tanpa traceback.
                        try:
                            await self._ban_pair(
                                pair,
                                hours=SCAN_BANNED_HISTORY_HOURS,
                                reason=f"Riwayat candle belum cukup: {exc}",
                                source="AUTO_INSUFFICIENT_HISTORY",
                            )
                        except Exception as ban_exc:
                            log.warning("[SCAN] gagal ban %s (riwayat kurang): %s", pair, ban_exc)
                        log.info("[SCAN] %s dilewati, riwayat candle kurang | %s", pair, exc)
                    else:
                        log.exception("[SCAN] analysis error %s", pair)

                await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)

        # Kandidat yang threshold-pass selalu divalidasi berdasarkan confidence
        # terbesar terlebih dahulu.
        threshold_candidates.sort(
            key=lambda item: Decimal(str(item.get("confidence") or "0")),
            reverse=True,
        )
        initial_conf_values = [
            Decimal(str(x["confidence"]))
            for x in threshold_candidates
        ]

        validated: list[dict[str, Any]] = []
        validator_rejects = 0
        validator_missing = 0

        if threshold_candidates:
            log.info(
                "[SCAN] Cycle #%s validator started candidates=%s",
                cycle,
                len(threshold_candidates),
            )

        for rank, candidate in enumerate(threshold_candidates, start=1):
            if scan_generation != self._scan_generation or not self._scan_user_enabled:
                log.info("[SCAN] Cycle #%s validator dibatalkan: generation berubah/off.", cycle)
                return
            if len(self.active_trades) >= self.max_active_trades:
                self._scan_auto_paused = True
                break

            log.info(
                "[SCAN] Cycle #%s validator %s/%s: %s",
                cycle,
                rank,
                len(threshold_candidates),
                candidate["pair"],
            )
            try:
                checked = await self._scan_validate_candidate(module, candidate, cycle)
                if checked is None:
                    validator_missing += 1
                else:
                    validated.append(checked)
            except Exception as exc:
                validator_rejects += 1
                if getattr(exc, "bybit_rate_limited", False):
                    retry_after = getattr(exc, "retry_after_seconds", 0.0)
                    log.warning(
                        "[SCAN] validator %s dilewati setelah Bybit rate-limit backoff | retry_after=%.2fs | %s",
                        candidate["pair"],
                        float(retry_after or 0.0),
                        exc,
                    )
                else:
                    log.exception(
                        "[SCAN] validator error %s",
                        candidate["pair"],
                    )

            await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)

        final_added: list[Trade] = []
        final_add_errors = 0
        final_margin_banned: list[str] = []
        final_capped = 0
        final_leverage_banned: list[str] = []

        for candidate in validated:
            if scan_generation != self._scan_generation or not self._scan_user_enabled:
                log.info("[SCAN] Cycle #%s final add dibatalkan: generation berubah/off; tidak ada candidate lama yang dimasukkan.", cycle)
                return
            if not self._scan_user_enabled:
                break
            if len(self.active_trades) >= self.max_active_trades:
                self._scan_auto_paused = True
                break

            if len(final_added) >= SCAN_MAX_NEW_PER_CYCLE:
                final_capped += 1
                continue
            cand_dir = str(candidate.get("direction") or "").upper()
            if cand_dir and sum(
                1 for t in self.active_trades.values() if t.direction == cand_dir
            ) >= SCAN_MAX_PER_DIRECTION:
                final_capped += 1
                continue

            try:
                meta = {
                    "strategy_name": candidate.get("strategy_name") or "SMC_VLT_RSI",
                    "strategy_version": candidate.get("strategy_version") or "SCAN",
                    "strategy_source": "SCAN",
                    "strategy_confidence": candidate.get("confidence"),
                    "strategy_data_source": candidate.get("data_source") or "BYBIT",
                    "strategy_analysis": candidate.get("analysis") or {},
                }
                meta["margin_usdt"] = self.margin_usdt
                meta["leverage"] = self.leverage
                trade = self._build_trade_from_setup(
                    candidate,
                    strategy_meta=meta,
                )
                if self.real_mode:
                    if self.real.cooldown_remaining > 0:
                        break
                    try:
                        await self._ensure_real_entry(trade)
                    except RealPositionConflictError as exc:
                        log.info("[SCAN] %s dilewati: %s", trade.pair, exc)
                        final_add_errors += 1
                        await self.reply(
                            card(
                                "⚠️ REAL ENTRY DIBLOKIR",
                                [
                                    f"Pair      {trade.pair}",
                                    f"Alasan    {exc}",
                                    "Exposure Binance sudah ada; setup tidak dibuat sebagai real entry baru.",
                                ],
                            )
                        )
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue
                    except BinanceRateLimitError:
                        break
                    except MarginInfluenceError as exc:
                        await self._ban_pair(
                            trade.pair,
                            hours=None,
                            reason=f"Real quantity tidak memenuhi target margin ±10%: {exc}",
                            source="AUTO_MARGIN_INFLUENCE",
                        )
                        final_margin_banned.append(trade.pair)
                        log.info("[SCAN] %s diban permanen (margin influence): %s", trade.pair, exc)
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue
                    except LeverageNotSupportedError as exc:
                        await self._ban_pair(
                            trade.pair,
                            hours=SCAN_BANNED_LEVERAGE_HOURS,
                            reason=str(exc),
                            source="AUTO_LEVERAGE_UNSUPPORTED",
                        )
                        final_leverage_banned.append(trade.pair)
                        log.info("[SCAN] %s dilewati, diban %sj (leverage tidak didukung)", trade.pair, SCAN_BANNED_LEVERAGE_HOURS)
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue
                    except BinanceAPIError as exc:
                        # Sudah dilog sekali oleh _handle_real_exception; jangan ulang traceback.
                        final_add_errors += 1
                        log.info("[SCAN] %s gagal entry real: %s", trade.pair, exc)
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue
                    if trade.status == "CLOSED":
                        continue
                self.active_trades[trade.trade_id] = trade
                final_added.append(trade)
                await self.ws.add_symbol(trade.pair)
                log.info(
                    "[SCAN] FINAL ADD %s confidence=%s trade_id=%s",
                    trade.pair,
                    decimal_to_str(trade.strategy_confidence),
                    trade.trade_id,
                )
            except Exception:
                final_add_errors += 1
                log.exception(
                    "[SCAN] gagal memasukkan validated setup %s ke /trade",
                    candidate.get("pair"),
                )

            await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)

        if len(self.active_trades) >= self.max_active_trades:
            self._scan_auto_paused = True

        avg_initial = (
            sum(initial_conf_values, Decimal("0")) / Decimal(len(initial_conf_values))
            if initial_conf_values
            else None
        )
        validated_conf_values = [
            Decimal(str(x["confidence"]))
            for x in validated
            if x.get("confidence") is not None
        ]
        avg_validated = (
            sum(validated_conf_values, Decimal("0")) / Decimal(len(validated_conf_values))
            if validated_conf_values
            else None
        )
        highest = (
            max(validated_conf_values)
            if validated_conf_values
            else (max(initial_conf_values) if initial_conf_values else None)
        )
        lowest = (
            min(validated_conf_values)
            if validated_conf_values
            else (min(initial_conf_values) if initial_conf_values else None)
        )
        validation_ratios: list[Decimal] = []
        for checked in validated:
            try:
                scan_meta = (checked.get("analysis") or {}).get("scan")
                ratio = Decimal(str(scan_meta.get("validation_ratio_percent"))) if isinstance(scan_meta, dict) and scan_meta.get("validation_ratio_percent") not in (None, "") else None
            except (InvalidOperation, AttributeError):
                ratio = None
            if ratio is not None:
                validation_ratios.append(ratio)

        avg_validation_ratio = (
            sum(validation_ratios, Decimal("0")) / Decimal(len(validation_ratios))
            if validation_ratios
            else None
        )
        duration = time.monotonic() - started

        self._scan_last_report = {
            "cycle": cycle,
            "btc_h4_trend": btc_trend,
            "btc_analysis": btc_analysis,
            "universe": {
                "common": universe["common_count"],
                "binance": universe.get("binance_count"),
                "bybit": universe.get("bybit_count"),
                "already_trade": already_trade,
                "banned": banned_count,
                "margin_blocked": margin_blocked,
                "eligible": len(eligible),
                "batch_limit": SCAN_MAX_PAIRS_PER_CYCLE,
                "batch_selected": len(batch),
                "deferred_to_next_cycle": deferred,
            },
            "direction": {
                "aligned": len(directional),
                "rejected_structure": rejected_structure,
                "structure_banned_24h": structure_banned,
                "counts": direction_counts,
            },
            "analysis": {
                "scanned": scanned,
                "threshold_candidates": len(threshold_candidates),
                "below_threshold_banned_8h": threshold_banned,
                "validator_valid": len(validated),
                "validator_missing": validator_missing,
                "validator_errors_or_rejects": validator_rejects,
                "final_added": len(final_added),
                "final_add_errors": final_add_errors,
                "final_margin_banned": list(final_margin_banned),
                "final_leverage_banned": list(final_leverage_banned),
                "analysis_errors": analysis_errors,
            },
            "confidence": {
                "average_initial": avg_initial,
                "average_validated": avg_validated,
                "average_validation_ratio_percent": avg_validation_ratio,
                "highest": highest,
                "lowest": lowest,
                "threshold": self.scan_threshold,
            },
            "duration_seconds": duration,
            "next_cycle_delay_seconds": SCAN_CYCLE_DELAY_SECONDS,
        }

        # Informasi proses detail tetap di Render; ringkasan cycle completion
        # dikirim ke Telegram oleh TelegramErrorHandler karena itu adalah INFO
        # khusus yang kita izinkan.
        log.info(
            "[SCAN] Cycle #%s completed | scanned=%s aligned=%s threshold=%s validated=%s added=%s avg_initial=%s avg_validated=%s duration=%.2fs",
            cycle,
            scanned,
            len(directional),
            len(threshold_candidates),
            len(validated),
            len(final_added),
            decimal_to_str(avg_initial),
            decimal_to_str(avg_validated),
            duration,
        )

        btc_lines: list[str] = []
        if isinstance(btc_analysis, dict):
            btc_node = btc_analysis.get("btc_h4")
            if isinstance(btc_node, dict):
                for key in (
                    "trend",
                    "last_bos",
                    "last_mss",
                    "protected_high",
                    "protected_low",
                    "swing_high_count",
                    "swing_low_count",
                ):
                    if key in btc_node:
                        btc_lines.append(f"{key}: {btc_node.get(key)}")

        ratio_line = (
            f"   Rasio validasi   {fmt_num(avg_validation_ratio)}%\n"
            if avg_validation_ratio is not None
            else ""
        )
        margin_line = f"├ Ban margin       {len(final_margin_banned)}"
        if final_margin_banned:
            margin_line += f"  ({', '.join(final_margin_banned[:10])})"
        leverage_line = f"├ Ban leverage      {len(final_leverage_banned)}"
        if final_leverage_banned:
            leverage_line += f"  ({', '.join(final_leverage_banned[:10])})"
        await self.reply(
            f"╭─ 🔄 SCAN #{cycle} SELESAI ─╮\n"
            f"│ ₿ BTC H4   {btc_trend}\n"
            f"│ ⏱ {duration:.0f}s  •  siklus berikut {SCAN_CYCLE_DELAY_SECONDS:g}s\n"
            "╰──────────────────╯\n\n"
            "🌐 UNIVERSE\n"
            f"├ Binance ∩ Bybit  {universe['common_count']}\n"
            f"├ Banned           {banned_count}\n"
            f"├ Sudah di /trade  {already_trade}\n"
            f"├ Margin blocked   {margin_blocked}\n"
            f"└ Eligible         {len(eligible)}  (batch {len(batch)}/{SCAN_MAX_PAIRS_PER_CYCLE}, tunda {deferred})\n\n"
            "🧭 ARAH vs BTC\n"
            f"├ Searah {len(directional)}  •  tidak searah {rejected_structure} (ban 24j)\n"
            f"└ H4  🟢 {direction_counts['BULLISH']}  🔴 {direction_counts['BEARISH']}"
            f"  ⚪ {direction_counts['RANGE']}  ❓ {direction_counts['UNKNOWN']}\n\n"
            "🔬 ANALISIS\n"
            f"├ Dipindai         {scanned}/{SCAN_MAX_PAIRS_PER_CYCLE}\n"
            f"├ ≥ Threshold      {len(threshold_candidates)}  (di bawah {threshold_banned}, ban 2-8j)\n"
            f"├ Validator lolos  {len(validated)}  (tolak {validator_missing + validator_rejects})\n"
            f"├ Masuk /trade     {len(final_added)}  (error {final_add_errors}, dibatasi {final_capped})\n"
            f"{margin_line}\n"
            f"{leverage_line}\n"
            f"└ Error analisis   {analysis_errors}\n\n"
            f"🎯 CONFIDENCE (min {decimal_to_str(self.scan_threshold)})\n"
            f"├ Awal {fmt_num(avg_initial)}  →  Valid {fmt_num(avg_validated)}\n"
            f"{ratio_line}"
            f"└ Tertinggi {fmt_num(highest)}  •  Terendah {fmt_num(lowest)}"
        )

    async def reset_github_records(self) -> None:
        """Hapus seluruh file pencatatan bot dari GitHub.

        Tidak menghapus main.py atau file konfigurasi/repository lain.
        Active setup di RAM juga tidak disentuh karena /reset hanya untuk
        pencatatan historis.
        """
        deleted: list[str] = []
        missing: list[str] = []

        for path in RESET_PATHS:
            removed = await self.github.delete_file(
                path,
                f"reset: delete {path}",
            )
            if removed:
                deleted.append(path)
            else:
                missing.append(path)

        self.history_records.clear()
        self.history_events.clear()
        self.notes.clear()
        self._history_dirty = False
        self.last_history_refresh = now_utc()

        await self.reply(
            "♻️ RESET PENCATATAN SELESAI\n\n"
            f"Dihapus: {len(deleted)} file\n"
            f"Tidak ditemukan: {len(missing)} file\n\n"
            "Yang dihapus: histori trade, event, journal, hasil analyze, dan catatan.\n"
            "Active setup /trade TIDAK dihapus."
        )

    async def _write_history(
        self,
        trade: Trade,
    ) -> None:
        """
        Persist satu trade yang sudah CLOSED.

        Final event dibuat di sini supaya:
        - final event tidak dobel,
        - final event + trade record berada dalam satu history lock,
        - dua trade yang close hampir bersamaan tidak saling overwrite.
        """
        async with self._history_lock:
            self._history_dirty = True
            record = trade.to_record()

            final_event = {
                "event_id": uuid4().hex,
                "trade_id": trade.trade_id,
                "session_id": trade.session_id,
                "event": trade.result,
                "timestamp": iso_utc(trade.closed_at),
                "timestamp_wib": format_wib(
                    trade.closed_at
                ),
                "pair": trade.pair,
                "direction": trade.direction,
                "price": decimal_to_str(
                    trade.exit_price
                ),
                "reason": trade.result_reason,
                "pnl_percent": (
                    decimal_to_str(trade.pnl_percent)
                    if trade.pnl_percent is not None
                    else None
                ),
            }

            self.history_records.append(
                record
            )

            self.history_events.append(
                final_event
            )

            trades_json = json.dumps(
                self.history_records,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")

            events_jsonl = (
                "".join(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                    for item in self.history_events
                )
            ).encode("utf-8")

            journal_entry = self._trade_to_markdown(
                trade
            )

            try:
                await self.github.replace_file(
                    HISTORY_TRADES_PATH,
                    trades_json,
                    (
                        f"trade: {trade.pair} "
                        f"{trade.direction} "
                        f"{trade.result}"
                    ),
                )

                await self.github.replace_file(
                    HISTORY_EVENTS_PATH,
                    events_jsonl,
                    (
                        f"history: {trade.pair} "
                        f"{trade.result}"
                    ),
                )

                await self.github.append_text_file(
                    HISTORY_MARKDOWN_PATH,
                    journal_entry,
                    (
                        f"journal: {trade.pair} "
                        f"{trade.result}"
                    ),
                )

                # Ketiga file histori berhasil diperbarui. Refresh remote
                # boleh dipercaya kembali pada cycle /stats berikutnya.
                self._history_dirty = False

            except Exception:
                log.exception(
                    "Gagal menulis histori %s ke GitHub.",
                    trade.trade_id,
                )

                await self.reply(
                    "⚠️ GitHub gagal diperbarui.\n\n"
                    f"Trade: {trade.pair} {trade.direction}\n"
                    f"Hasil: {trade.result}\n"
                    "Data tetap ada di session RAM. "
                    "Commit histori akan dicoba lagi pada event "
                    "berikutnya dalam session ini."
                )

    def _trade_to_markdown(
        self,
        trade: Trade,
    ) -> str:
        record = trade.to_record()

        trail_lines: list[str] = []

        for index, item in enumerate(
            trade.trail_history,
            start=1,
        ):
            trail_lines.append(
                f"Trail #{index}\n"
                f"- Old SL: {item.get('old_sl', '-')}\n"
                f"- New SL: {item.get('new_sl', '-')}\n"
                f"- Price: {item.get('price', '-')}\n"
                f"- Reason: {item.get('reason', '-')}\n"
                f"- Time: {item.get('timestamp_wib', '-')}\n"
            )

        trails = (
            "\n".join(trail_lines)
            if trail_lines
            else "Tidak ada trailing.\n"
        )

        return (
            "\n---\n\n"
            f"## {record['pair']} — {record['direction']}\n\n"
            f"Trade ID: `{record['trade_id']}`\n\n"
            "### Setup\n\n"
            f"- Price Now Reference: {record['price_now_reference']}\n"
            f"- Entry: {record['entry']}\n"
            f"- Reason Entry: {record['entry_reason']}\n"
            f"- Price Exp: {record['price_exp']}\n"
            f"- Reason Price Exp: {record['price_exp_reason']}\n"
            f"- SL: {record['sl']}\n"
            f"- Reason SL: {record['sl_reason']}\n"
            f"- TP: {record['tp']}\n"
            f"- Reason TP: {record['tp_reason']}\n\n"
            "### Management\n\n"
            f"{trails}"
            "\n### Result\n\n"
            f"- Result: {record['result']}\n"
            f"- Exit Price: {record['exit_price'] or '-'}\n"
            f"- Result Reason: {record['result_reason'] or '-'}\n"
            f"- PnL: {(record['pnl_percent'] + '%') if record['pnl_percent'] else '-'}\n"
            f"- Created: {record['created_at']}\n"
            f"- Filled: {record['filled_at'] or '-'}\n"
            f"- Closed: {record['closed_at'] or '-'}\n"
            f"- Strategy: {record['strategy_name']} "
            f"v{record['strategy_version']}\n"
        )

    # --------------------------------------------------------
    # Symbol / price
    # --------------------------------------------------------

    def _get_symbol(
        self,
        symbol: str,
    ) -> SymbolMeta:
        normalized = normalize_symbol(symbol)

        if not self.symbols:
            raise ValueError(
                "Data symbol Binance belum dimuat (API sedang dibatasi). "
                "Coba lagi setelah jeda selesai."
            )

        meta = self.symbols.get(
            normalized
        )

        if meta is None:
            raise ValueError(
                f"{normalized or symbol} tidak ditemukan "
                "sebagai USDT-M perpetual yang aktif."
            )

        return meta

    def _validate_price(
        self,
        symbol: str,
        value: Decimal,
    ) -> None:
        meta = self._get_symbol(symbol)

        if not validate_tick(
            value,
            meta.tick_size,
        ):
            raise ValueError(
                "Harga tidak mengikuti tick size Binance.\n"
                f"Tick size {symbol}: "
                f"{decimal_to_str(meta.tick_size)}"
            )

        if (
            meta.min_price > 0
            and value < meta.min_price
        ):
            raise ValueError(
                "Harga berada di bawah minimum price symbol."
            )

        if (
            meta.max_price > 0
            and value > meta.max_price
        ):
            raise ValueError(
                "Harga berada di atas maximum price symbol."
            )

    async def _reference_price(
        self,
        symbol: str,
    ) -> Decimal:
        # REST hanya dipanggil satu kali untuk /add.
        price = await self.rest.get_price(
            symbol
        )

        return price

    # --------------------------------------------------------
    # AUTO STRATEGY BRIDGE
    # --------------------------------------------------------

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert strategy output into JSON-safe primitive structures."""
        if isinstance(value, Decimal):
            return decimal_to_str(value)
        if isinstance(value, dict):
            return {
                str(key): TradingEngine._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                TradingEngine._json_safe(item)
                for item in value
            ]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _normalize_auto_result(
        self,
        raw_result: Any,
        pair: str,
    ) -> dict[str, Any]:
        """
        Validate the strict return contract from strategy.py.

        strategy.py may return either:
            {"setup": {...}, "confidence": ...}
        or a flat result containing the setup fields directly.
        The bridge normalizes both forms to one internal structure.
        """
        if not isinstance(raw_result, dict):
            raise ValueError(
                "strategy.generate_setup() harus mengembalikan dict."
            )

        payload = copy.deepcopy(raw_result)
        nested_setup = payload.get("setup")

        if nested_setup is not None:
            if not isinstance(nested_setup, dict):
                raise ValueError(
                    "Field 'setup' dari strategy.py harus berupa dict."
                )
            merged = copy.deepcopy(nested_setup)
            for key, value in payload.items():
                if key != "setup":
                    merged[key] = value
            payload = merged

        required = (
            "direction",
            "entry",
            "entry_reason",
            "price_exp",
            "price_exp_reason",
            "sl",
            "sl_reason",
            "tp",
            "tp_reason",
            "confidence",
        )

        missing = [
            key
            for key in required
            if payload.get(key) in (None, "")
        ]
        if missing:
            raise ValueError(
                "Hasil strategy.py kurang field: "
                + ", ".join(missing)
            )

        result_pair = normalize_symbol(
            str(payload.get("pair") or pair)
        )
        if result_pair != pair:
            raise ValueError(
                f"strategy.py mengembalikan pair {result_pair}, "
                f"padahal pair yang diminta {pair}."
            )

        direction = str(payload["direction"]).upper().strip()
        if direction not in {"BUY", "SELL"}:
            raise ValueError(
                "Direction hasil strategy.py harus BUY atau SELL."
            )

        def decimal_field(name: str) -> Decimal:
            try:
                raw_value = payload[name]
                if isinstance(raw_value, float):
                    # Hindari notasi ilmiah (mis. 5e-05) yang ditolak parse_decimal.
                    raw_value = format(Decimal(repr(raw_value)), "f")
                return parse_decimal(str(raw_value))
            except Exception as exc:
                raise ValueError(
                    f"Field {name} dari strategy.py bukan angka valid."
                ) from exc

        confidence = decimal_field("confidence")
        if confidence < 0 or confidence > 100:
            raise ValueError(
                "Confidence strategy.py harus berada pada 0–100."
            )

        reasons = {}
        for key in (
            "entry_reason",
            "price_exp_reason",
            "sl_reason",
            "tp_reason",
        ):
            value = str(payload[key]).strip()
            if not value:
                raise ValueError(
                    f"{key} dari strategy.py tidak boleh kosong."
                )
            reasons[key] = value

        confidence_components = payload.get("confidence_components")
        if confidence_components is None:
            confidence_components = {}
        if not isinstance(confidence_components, dict):
            raise ValueError(
                "confidence_components harus berupa dict."
            )

        for name, raw_score in confidence_components.items():
            try:
                score = Decimal(str(raw_score))
            except Exception as exc:
                raise ValueError(
                    f"Confidence component '{name}' bukan angka."
                ) from exc
            if score < 0 or score > 100:
                raise ValueError(
                    f"Confidence component '{name}' harus 0–100."
                )

        strategy_info = payload.get("strategy")
        if strategy_info is None:
            strategy_info = {}
        if not isinstance(strategy_info, dict):
            raise ValueError(
                "Field 'strategy' dari strategy.py harus berupa dict."
            )

        analysis = payload.get("analysis")
        if analysis is None:
            analysis = {}
        if not isinstance(analysis, dict):
            raise ValueError(
                "Field 'analysis' dari strategy.py harus berupa dict."
            )

        data_info = payload.get("data")
        if data_info is None:
            data_info = {}
        if not isinstance(data_info, dict):
            raise ValueError(
                "Field 'data' dari strategy.py harus berupa dict."
            )

        normalized = {
            "pair": result_pair,
            "direction": direction,
            "price_now_reference": (
                decimal_field("price_now_reference")
                if payload.get("price_now_reference") not in (None, "")
                else None
            ),
            "entry": decimal_field("entry"),
            "entry_reason": reasons["entry_reason"],
            "price_exp": decimal_field("price_exp"),
            "price_exp_reason": reasons["price_exp_reason"],
            "sl": decimal_field("sl"),
            "sl_reason": reasons["sl_reason"],
            "tp": decimal_field("tp"),
            "tp_reason": reasons["tp_reason"],
            "confidence": confidence,
            "confidence_components": self._json_safe(
                confidence_components
            ),
            "strategy_name": str(
                strategy_info.get("name")
                or payload.get("strategy_name")
                or "SMC_VLT_RSI"
            ).strip(),
            "strategy_version": str(
                strategy_info.get("version")
                or payload.get("strategy_version")
                or "0.1.0"
            ).strip(),
            "data_source": str(
                data_info.get("source")
                or payload.get("data_source")
                or "UNKNOWN"
            ).strip(),
            "timeframe": str(
                data_info.get("timeframe")
                or "15m"
            ).strip(),
            "candles_requested": data_info.get(
                "candles_requested", 672
            ),
            "candles_used": data_info.get(
                "candles_used"
            ),
            "analysis": self._json_safe(analysis),
            "raw_result": self._json_safe(payload),
        }

        return normalized

    async def _run_strategy_for_auto(
        self,
        pair: str,
        job_id: str,
    ) -> None:
        """Run strategy.py and return its result into the active AUTO flow."""
        try:
            strategy_path = BASE_DIR / "strategy.py"
            if not strategy_path.is_file():
                raise RuntimeError(
                    "strategy.py tidak ditemukan di folder main.py. "
                    f"Path yang diperiksa: {strategy_path}"
                )

            try:
                spec = importlib.util.spec_from_file_location(
                    "trading_strategy_runtime",
                    strategy_path,
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError(
                        "Python tidak dapat membuat loader untuk strategy.py."
                    )

                strategy_module = importlib.util.module_from_spec(spec)
                sys.modules["trading_strategy_runtime"] = strategy_module
                spec.loader.exec_module(strategy_module)
            except Exception as exc:
                raise RuntimeError(
                    "Module strategy.py tidak dapat dimuat. "
                    "Periksa syntax/import strategy.py."
                ) from exc

            generate_setup = getattr(
                strategy_module,
                "generate_setup",
                None,
            )
            if not callable(generate_setup):
                raise RuntimeError(
                    "strategy.py wajib menyediakan "
                    "async def generate_setup(pair, context)."
                )

            context = {
                "pair": pair,
                "session_id": self.session_id,
                "market": "Binance USDⓈ-M Futures",
                "timeframe": "15m",
                "candles_requested": 672,
                "price_trigger": "aggTrade last price",
                "notes": copy.deepcopy(self.notes),
                "engine_version": "AUTO_BRIDGE_1",
            }

            if inspect.iscoroutinefunction(generate_setup):
                result = await generate_setup(
                    pair,
                    context,
                )
            else:
                result = await asyncio.to_thread(
                    generate_setup,
                    pair,
                    context,
                )

            if inspect.isawaitable(result):
                result = await result

            normalized = self._normalize_auto_result(
                result,
                pair,
            )

            # If strategy.py doesn't provide a current reference, main.py
            # uses its existing public market-data REST exactly once here.
            if normalized["price_now_reference"] is None:
                normalized["price_now_reference"] = await self._reference_price(
                    pair
                )

            if self.flow is None:
                return
            if self.flow.get("kind") != "AUTO":
                return
            if self.flow.get("job_id") != job_id:
                return
            if not self._running:
                return

            self.flow["step"] = "CONFIRM"
            self.flow["data"] = normalized
            self._auto_task = None

            await self.reply(
                self._render_auto_result(normalized)
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception(
                "AUTO strategy gagal untuk %s.",
                pair,
            )

            if (
                self.flow
                and self.flow.get("kind") == "AUTO"
                and self.flow.get("job_id") == job_id
            ):
                self.flow = None
                self._auto_task = None
                await self.reply(
                    "❌ AUTO ANALYSIS GAGAL\n\n"
                    f"Pair: {pair}\n"
                    f"Error: {exc}\n\n"
                    "Periksa log backend untuk detail."
                )

    def _render_auto_result(
        self,
        result: dict[str, Any],
    ) -> str:
        components = result.get("confidence_components") or {}

        lines = [
            "🤖 AUTO ANALYSIS SELESAI",
            "",
            f"Pair: {result['pair']}",
            f"Data: {result.get('data_source', '-')}",
            f"Timeframe: {result.get('timeframe', '15m')}",
            f"Candle: {result.get('candles_used') or result.get('candles_requested') or 672}",
            "",
            f"Direction: {result['direction']}",
            f"Price Now: {decimal_to_str(result['price_now_reference'])}",
            f"Entry: {decimal_to_str(result['entry'])}",
            f"Price Exp: {decimal_to_str(result['price_exp'])}",
            f"SL: {decimal_to_str(result['sl'])}",
            f"TP: {decimal_to_str(result['tp'])}",
            "",
            f"Confidence: {decimal_to_str(result['confidence'])}/100",
        ]

        if components:
            lines.extend([
                "",
                "Confidence Components:",
            ])
            for name, score in components.items():
                lines.append(
                    f"• {name}: {score}"
                )

        lines.extend([
            "",
            "🧠 Reason Entry:",
            result["entry_reason"],
            "",
            "⛔ Reason Price Exp:",
            result["price_exp_reason"],
            "",
            "🛡 Reason SL:",
            result["sl_reason"],
            "",
            "🎯 Reason TP:",
            result["tp_reason"],
            "",
            "Strategy:",
            f"{result['strategy_name']} v{result['strategy_version']}",
            "",
            "Gunakan setup ini?",
            "1. ✅ Ya, masukkan ke /trade",
            "2. ❌ Tidak",
        ])

        return "\n".join(lines)

    async def _start_auto(self) -> None:
        if self.flow is not None:
            await self.reply(
                "Masih ada sesi yang sedang berjalan.\n"
                "Gunakan /back terlebih dahulu."
            )
            return

        if len(self.active_trades) >= self.max_active_trades:
            await self.reply(
                "Maksimum active trade tercapai.\n"
                f"Batas: {self.max_active_trades}"
            )
            return

        job_id = uuid4().hex
        self._auto_job_id = job_id
        self.flow = {
            "kind": "AUTO",
            "step": "PAIR",
            "job_id": job_id,
            "data": {},
        }

        await self.reply(
            "🤖 AUTO SETUP\n\n"
            "Pair apa yang ingin dicari setupnya?\n"
            "Contoh: AVAXUSDT"
        )

    async def _handle_auto_input(
        self,
        text: str,
    ) -> None:
        if not self.flow or self.flow.get("kind") != "AUTO":
            return

        step = self.flow.get("step")
        if step == "PAIR":
            pair = normalize_symbol(text)
            self._get_symbol(pair)

            job_id = str(self.flow.get("job_id") or "")
            if not job_id:
                self.flow = None
                raise RuntimeError("AUTO job ID tidak tersedia.")

            self.flow["step"] = "ANALYZING"

            await self.reply(
                "🔎 AUTO ANALYSIS\n\n"
                f"Pair: {pair}\n"
                "Sedang menganalisis 672 candle M15 yang sudah close...\n"
                "Sumber data: strategy.py (Bybit → Binance public fallback)"
            )

            task = asyncio.create_task(
                self._run_strategy_for_auto(
                    pair,
                    job_id,
                ),
                name=f"auto-strategy-{pair}-{job_id[:8]}",
            )
            self._auto_task = task
            return

        if step == "ANALYZING":
            await self.reply(
                "⏳ AUTO masih menganalisis.\n"
                "Gunakan /back untuk membatalkan."
            )
            return

        if step == "CONFIRM":
            answer = str(text or "").strip()
            if answer == "1":
                await self._confirm_auto()
                return
            if answer == "2":
                self.flow = None
                self._auto_job_id = None
                await self.reply("❌ AUTO SETUP ditolak.")
                return

            raise ValueError(
                "Jawab 1 untuk menggunakan setup atau 2 untuk menolak."
            )

        raise ValueError(
            f"AUTO step tidak dikenal: {step}"
        )

    async def _confirm_auto(self) -> None:
        if not self.flow or self.flow.get("kind") != "AUTO":
            raise RuntimeError("AUTO flow tidak aktif.")

        result = self.flow.get("data")
        if not isinstance(result, dict):
            raise ValueError("Hasil AUTO belum tersedia.")

        data = {
            "pair": result["pair"],
            "direction": result["direction"],
            "price_now_reference": result["price_now_reference"],
            "entry": result["entry"],
            "entry_reason": result["entry_reason"],
            "price_exp": result["price_exp"],
            "price_exp_reason": result["price_exp_reason"],
            "sl": result["sl"],
            "sl_reason": result["sl_reason"],
            "tp": result["tp"],
            "tp_reason": result["tp_reason"],
        }

        strategy_meta = {
            "strategy_name": result["strategy_name"],
            "strategy_version": result["strategy_version"],
            "strategy_source": "AUTO",
            "strategy_confidence": result["confidence"],
            "strategy_data_source": result["data_source"],
            "strategy_analysis": {
                "confidence_components": result.get(
                    "confidence_components", {}
                ),
                "timeframe": result.get("timeframe", "15m"),
                "candles_requested": result.get(
                    "candles_requested", 672
                ),
                "candles_used": result.get("candles_used"),
                "analysis": result.get("analysis", {}),
                "raw_result": result.get("raw_result", {}),
            },
        }

        strategy_meta["margin_usdt"] = self.margin_usdt
        strategy_meta["leverage"] = self.leverage
        trade = self._build_trade_from_setup(
            data,
            strategy_meta=strategy_meta,
        )

        if self.real_mode:
            try:
                await self._ensure_real_entry(trade)
            except RealPositionConflictError as exc:
                self.flow = None
                self._auto_job_id = None
                self._auto_task = None
                await self.reply(
                    card(
                        "⚠️ REAL ENTRY DIBLOKIR",
                        [
                            f"Pair   {trade.pair}",
                            f"Alasan {exc}",
                            "Setup tidak diaktifkan agar exposure Binance tidak bertambah.",
                        ],
                    )
                )
                return
            if trade.status == "CLOSED":
                self.flow = None
                self._auto_job_id = None
                self._auto_task = None
                await self.reply("🚨 Setup REAL langsung ditutup otomatis karena TP/SL gagal dipasang.")
                return

        self.active_trades[trade.trade_id] = trade
        self.flow = None
        self._auto_job_id = None
        self._auto_task = None

        try:
            await self.ws.add_symbol(trade.pair)
        except Exception:
            log.exception(
                "Subscription WebSocket %s gagal saat /auto.",
                trade.pair,
            )

        level_note = ""
        if trade.price_exp == trade.tp:
            level_note = (
                "\n\n⚠️ Catatan: Price Exp sama dengan TP. "
                "Saat status PENDING, sentuhan level ini tetap dihitung "
                "sebagai PRICE EXPIRED, bukan TP."
            )

        await self.reply(
            "✅ SETUP AUTO DITAMBAHKAN\n\n"
            f"Trade ID: {trade.trade_id}\n"
            f"Pair: {trade.pair}\n"
            f"Direction: {trade.direction.title()}\n"
            f"Price Now: {decimal_to_str(trade.price_now_reference)}\n"
            f"Price Entry: {decimal_to_str(trade.entry)}\n"
            f"Reason Entry: {trade.entry_reason}\n\n"
            f"Price Exp: {decimal_to_str(trade.price_exp)}\n"
            f"Reason Price Exp: {trade.price_exp_reason}\n\n"
            f"Price SL: {decimal_to_str(trade.sl)}\n"
            f"Reason SL: {trade.sl_reason}\n\n"
            f"Price TP: {decimal_to_str(trade.tp)}\n"
            f"Reason TP: {trade.tp_reason}\n\n"
            f"Confidence: {decimal_to_str(result['confidence'])}/100\n"
            f"Strategy: {trade.strategy_name} v{trade.strategy_version}\n"
            f"Data Source: {trade.strategy_data_source or '-'}\n\n"
            "Status: PENDING"
            f"{level_note}"
        )

    def _build_trade_from_setup(
        self,
        data: dict[str, Any],
        strategy_meta: dict[str, Any] | None = None,
    ) -> Trade:
        pair = data.get("pair")
        direction = data.get("direction")
        reference = data.get("price_now_reference")
        entry = data.get("entry")
        price_exp = data.get("price_exp")
        sl = data.get("sl")
        tp = data.get("tp")

        if any(
            value is None
            for value in [
                pair,
                direction,
                reference,
                entry,
                price_exp,
                sl,
                tp,
            ]
        ):
            raise ValueError("Setup belum lengkap.")

        pair = normalize_symbol(str(pair))
        direction = str(direction).upper().strip()
        meta = strategy_meta or {}
        strategy_source = str(meta.get("strategy_source") or "MANUAL").upper().strip()

        # Strategy menghasilkan harga dari data Bybit. Sebelum setup masuk ke
        # /trade, harga executable dinormalisasi ke tick size Binance sehingga
        # setup seperti TUTUSDT tidak gagal hanya karena precision berbeda.
        # Price Now Reference bukan harga order, jadi tidak diwajibkan mengikuti
        # tick size Binance.
        if strategy_source in {"AUTO", "SCAN"}:
            symbol_meta = self._get_symbol(pair)
            tick = symbol_meta.tick_size
            if direction == "BUY":
                sl = quantized_price(sl, tick)
                entry = quantized_price(entry, tick)
                price_exp = quantized_price_ceiling(price_exp, tick)
                tp = quantized_price_ceiling(tp, tick)
            else:
                sl = quantized_price_ceiling(sl, tick)
                entry = quantized_price_ceiling(entry, tick)
                price_exp = quantized_price(price_exp, tick)
                tp = quantized_price(tp, tick)

            log.info(
                "[PRICE] strategy %s normalized to Binance tick | tick=%s entry=%s exp=%s sl=%s tp=%s",
                pair,
                decimal_to_str(tick),
                decimal_to_str(entry),
                decimal_to_str(price_exp),
                decimal_to_str(sl),
                decimal_to_str(tp),
            )

        # Price Now Reference hanya data konteks. Entry / Exp / SL / TP adalah
        # harga yang harus legal pada Binance.
        for price_value in (
            entry,
            price_exp,
            sl,
            tp,
        ):
            self._validate_price(
                pair,
                price_value,
            )

        if direction == "BUY":
            valid_geometry = (
                sl < entry < reference < price_exp
                and tp > entry
            )
        else:
            valid_geometry = (
                price_exp < reference < entry < sl
                and tp < entry
            )

        if not valid_geometry:
            raise ValueError(
                "Struktur harga setup tidak valid."
            )

        return Trade(
            trade_id=generate_trade_id(pair),
            session_id=self.session_id,
            pair=pair,
            direction=direction,
            price_now_reference=reference,
            entry=entry,
            entry_reason=str(data.get("entry_reason") or ""),
            price_exp=price_exp,
            price_exp_reason=str(data.get("price_exp_reason") or ""),
            sl=sl,
            sl_reason=str(data.get("sl_reason") or ""),
            tp=tp,
            tp_reason=str(data.get("tp_reason") or ""),
            strategy_name=str(meta.get("strategy_name") or "MANUAL"),
            strategy_version=str(meta.get("strategy_version") or "1.0"),
            strategy_source=str(meta.get("strategy_source") or "MANUAL"),
            strategy_confidence=(
                meta.get("strategy_confidence")
                if isinstance(meta.get("strategy_confidence"), Decimal)
                else (
                    parse_decimal(str(meta["strategy_confidence"]))
                    if meta.get("strategy_confidence") not in (None, "")
                    else None
                )
            ),
            strategy_data_source=(
                str(meta.get("strategy_data_source"))
                if meta.get("strategy_data_source") not in (None, "")
                else None
            ),
            strategy_analysis=(
                copy.deepcopy(meta.get("strategy_analysis"))
                if isinstance(meta.get("strategy_analysis"), dict)
                else {}
            ),
            margin_usdt=(
                meta.get("margin_usdt")
                if isinstance(meta.get("margin_usdt"), Decimal)
                else parse_decimal(str(meta.get("margin_usdt")))
                if meta.get("margin_usdt") not in (None, "")
                else self.margin_usdt
            ),
            leverage=int(meta.get("leverage") or self.leverage),
        )

    # --------------------------------------------------------
    # ADD flow
    # --------------------------------------------------------

    def _new_add_flow(self) -> None:
        self.flow = {
            "kind": "ADD",
            "step": "PAIR",
            "data": {
                "pair": None,
                "direction": None,
                "price_now_reference": None,
                "entry": None,
                "entry_reason": None,
                "price_exp": None,
                "price_exp_reason": None,
                "sl": None,
                "sl_reason": None,
                "tp": None,
                "tp_reason": None,
            },
        }

    def _add_data(self) -> dict[str, Any]:
        if not self.flow:
            raise RuntimeError(
                "ADD flow tidak aktif."
            )
        return self.flow["data"]

    def _render_add(self) -> str:
        if not self.flow or self.flow.get("kind") != "ADD":
            return ""

        data = self._add_data()
        step = self.flow["step"]

        def value_text(value: Any) -> str:
            if value is None:
                return "-"

            if isinstance(value, Decimal):
                return decimal_to_str(value) or "-"

            return str(value)

        lines = [
            "ADD",
            "",
            f"Pair: {value_text(data['pair'])}",
            f"Direction: {value_text(data['direction'])}",
            f"Price Now: {value_text(data['price_now_reference'])}",
            f"Price Entry: {value_text(data['entry'])}",
            f"Reason Entry: {value_text(data['entry_reason'])}",
            f"Price Exp: {value_text(data['price_exp'])}",
            f"Reason Price Exp: {value_text(data['price_exp_reason'])}",
            f"Price SL: {value_text(data['sl'])}",
            f"Reason SL: {value_text(data['sl_reason'])}",
            f"Price TP: {value_text(data['tp'])}",
            f"Reason TP: {value_text(data['tp_reason'])}",
            "",
        ]

        prompt_map = {
            "PAIR": "Pair:",
            "DIRECTION": (
                "Direction:\n"
                "1. Buy\n"
                "2. Sell"
            ),
            "ENTRY": "Price Entry:",
            "ENTRY_REASON": "Reason Entry:",
            "EXP": "Price Exp:",
            "EXP_REASON": "Reason Price Exp:",
            "SL": "Price SL:",
            "SL_REASON": "Reason SL:",
            "TP": "Price TP:",
            "TP_REASON": "Reason TP:",
            "CONFIRM": (
                "Confirm setup?\n"
                "1. Yes\n"
                "2. No"
            ),
        }

        lines.append(
            prompt_map.get(
                step,
                "Input:",
            )
        )

        return "\n".join(lines)

    async def _start_add(self) -> None:
        if self.flow is not None:
            await self.reply(
                "Masih ada sesi yang sedang berjalan.\n"
                "Gunakan /back untuk kembali atau "
                "selesaikan sesi tersebut."
            )
            return

        if len(self.active_trades) >= self.max_active_trades:
            await self.reply(
                "Maksimum active trade tercapai.\n"
                f"Batas: {self.max_active_trades}"
            )
            return

        self._new_add_flow()

        await self.reply(
            self._render_add()
        )

    async def _handle_add_input(
        self,
        text: str,
    ) -> None:
        if not self.flow:
            return

        data = self._add_data()
        step = self.flow["step"]

        try:
            if step == "PAIR":
                pair = normalize_symbol(text)
                meta = self._get_symbol(pair)

                # REST digunakan satu kali untuk reference price.
                price_now = await self._reference_price(pair)
                self._validate_price(
                    pair,
                    price_now,
                )

                data["pair"] = meta.symbol
                data["price_now_reference"] = price_now

                # Seed Current Price dengan reference awal.
                self.prices[meta.symbol] = PriceSnapshot(
                    symbol=meta.symbol,
                    price=price_now,
                    event_time_ms=int(time.time() * 1000),
                    received_at=now_utc(),
                    source="REST_REFERENCE",
                )

                self.flow["step"] = "DIRECTION"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "DIRECTION":
                answer = text.strip()

                if answer == "1":
                    direction = "BUY"
                elif answer == "2":
                    direction = "SELL"
                else:
                    raise ValueError(
                        "Jawab 1 untuk Buy atau 2 untuk Sell."
                    )

                data["direction"] = direction
                self.flow["step"] = "ENTRY"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "ENTRY":
                entry = parse_decimal(text)
                pair = data["pair"]

                self._validate_price(
                    pair,
                    entry,
                )

                reference = data["price_now_reference"]
                direction = data["direction"]

                if direction == "BUY" and entry >= reference:
                    raise ValueError(
                        "Untuk Buy, Price Entry harus "
                        "lebih rendah dari Price Now."
                    )

                if direction == "SELL" and entry <= reference:
                    raise ValueError(
                        "Untuk Sell, Price Entry harus "
                        "lebih tinggi dari Price Now."
                    )

                data["entry"] = entry
                self.flow["step"] = "ENTRY_REASON"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "ENTRY_REASON":
                reason = text.strip()

                if not reason:
                    raise ValueError(
                        "Reason Entry tidak boleh kosong."
                    )

                data["entry_reason"] = reason
                self.flow["step"] = "EXP"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "EXP":
                price_exp = parse_decimal(text)
                pair = data["pair"]

                self._validate_price(
                    pair,
                    price_exp,
                )

                reference = data["price_now_reference"]
                direction = data["direction"]

                if direction == "BUY" and price_exp <= reference:
                    raise ValueError(
                        "Untuk Buy, Price Exp harus "
                        "lebih tinggi dari Price Now."
                    )

                if direction == "SELL" and price_exp >= reference:
                    raise ValueError(
                        "Untuk Sell, Price Exp harus "
                        "lebih rendah dari Price Now."
                    )

                data["price_exp"] = price_exp
                self.flow["step"] = "EXP_REASON"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "EXP_REASON":
                reason = text.strip()

                if not reason:
                    raise ValueError(
                        "Reason Price Exp tidak boleh kosong."
                    )

                data["price_exp_reason"] = reason
                self.flow["step"] = "SL"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "SL":
                sl = parse_decimal(text)
                pair = data["pair"]

                self._validate_price(
                    pair,
                    sl,
                )

                entry = data["entry"]
                direction = data["direction"]

                if direction == "BUY" and sl >= entry:
                    raise ValueError(
                        "Untuk Buy, SL harus di bawah Entry."
                    )

                if direction == "SELL" and sl <= entry:
                    raise ValueError(
                        "Untuk Sell, SL harus di atas Entry."
                    )

                data["sl"] = sl
                self.flow["step"] = "SL_REASON"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "SL_REASON":
                reason = text.strip()

                if not reason:
                    raise ValueError(
                        "Reason SL tidak boleh kosong."
                    )

                data["sl_reason"] = reason
                self.flow["step"] = "TP"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "TP":
                tp = parse_decimal(text)
                pair = data["pair"]

                self._validate_price(
                    pair,
                    tp,
                )

                entry = data["entry"]
                direction = data["direction"]

                if direction == "BUY" and tp <= entry:
                    raise ValueError(
                        "Untuk Buy, TP harus di atas Entry."
                    )

                if direction == "SELL" and tp >= entry:
                    raise ValueError(
                        "Untuk Sell, TP harus di bawah Entry."
                    )

                data["tp"] = tp
                self.flow["step"] = "TP_REASON"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "TP_REASON":
                reason = text.strip()

                if not reason:
                    raise ValueError(
                        "Reason TP tidak boleh kosong."
                    )

                data["tp_reason"] = reason
                self.flow["step"] = "CONFIRM"

                await self.reply(
                    self._render_add()
                )
                return

            if step == "CONFIRM":
                answer = text.strip()

                if answer == "1":
                    await self._confirm_add()
                    return

                if answer == "2":
                    self.flow = None
                    await self.reply(
                        "ADD dibatalkan."
                    )
                    return

                raise ValueError(
                    "Jawab 1 untuk Yes atau 2 untuk No."
                )

            raise ValueError(
                f"Step ADD tidak dikenal: {step}"
            )

        except Exception as exc:
            await self.reply(
                f"❌ {exc}\n\n"
                f"{self._render_add()}"
            )

    async def _confirm_add(self) -> None:
        data = self._add_data()

        trade = self._build_trade_from_setup(
            data,
            strategy_meta={
                "strategy_name": "MANUAL",
                "strategy_version": "1.0",
                "strategy_source": "MANUAL",
                "margin_usdt": self.margin_usdt,
                "leverage": self.leverage,
            },
        )

        if self.real_mode:
            try:
                await self._ensure_real_entry(trade)
            except RealPositionConflictError as exc:
                self.flow = None
                await self.reply(
                    card(
                        "⚠️ REAL ENTRY DIBLOKIR",
                        [
                            f"Pair   {trade.pair}",
                            f"Alasan {exc}",
                            "Setup tidak dibuat agar exposure Binance tidak bertambah.",
                        ],
                    )
                )
                return
            if trade.status == "CLOSED":
                self.flow = None
                await self.reply("🚨 Setup REAL langsung ditutup otomatis karena TP/SL gagal dipasang.")
                return

        self.active_trades[
            trade.trade_id
        ] = trade

        self.flow = None

        level_note = ""
        if trade.price_exp == trade.tp:
            level_note = (
                "\n\n⚠️ Catatan: Price Exp sama dengan TP. "
                "Saat status PENDING, sentuhan level ini tetap dihitung "
                "sebagai PRICE EXPIRED, bukan TP."
            )

        try:
            await self.ws.add_symbol(
                trade.pair
            )
        except Exception:
            log.exception(
                "Subscription WebSocket %s gagal saat /add.",
                trade.pair,
            )

        await self.reply(
            "✅ SETUP DITAMBAHKAN\n\n"
            f"Trade ID: {trade.trade_id}\n"
            f"Pair: {trade.pair}\n"
            f"Direction: {trade.direction.title()}\n"
            f"Price Now: {decimal_to_str(trade.price_now_reference)}\n"
            f"Price Entry: {decimal_to_str(trade.entry)}\n"
            f"Reason Entry: {trade.entry_reason}\n\n"
            f"Price Exp: {decimal_to_str(trade.price_exp)}\n"
            f"Reason Price Exp: {trade.price_exp_reason}\n\n"
            f"Price SL: {decimal_to_str(trade.sl)}\n"
            f"Reason SL: {trade.sl_reason}\n\n"
            f"Price TP: {decimal_to_str(trade.tp)}\n"
            f"Reason TP: {trade.tp_reason}\n\n"
            "Status: PENDING"
            f"{level_note}"
        )

    # --------------------------------------------------------
    # BACK
    # --------------------------------------------------------

    async def _handle_back(self) -> None:
        if not self.flow:
            await self.reply(
                "Tidak ada sesi yang bisa dikembalikan."
            )
            return

        kind = self.flow["kind"]

        if kind == "ADD":
            previous = {
                "PAIR": None,
                "DIRECTION": "PAIR",
                "ENTRY": "DIRECTION",
                "ENTRY_REASON": "ENTRY",
                "EXP": "ENTRY_REASON",
                "EXP_REASON": "EXP",
                "SL": "EXP_REASON",
                "SL_REASON": "SL",
                "TP": "SL_REASON",
                "TP_REASON": "TP",
                "CONFIRM": "TP_REASON",
            }

            current = self.flow["step"]
            prior = previous.get(current)

            if prior is None:
                self.flow = None
                await self.reply(
                    "ADD dibatalkan."
                )
                return

            self.flow["step"] = prior

            await self.reply(
                self._render_add()
            )
            return

        if kind == "TRAIL":
            step = self.flow["step"]

            previous = {
                "SELECT": None,
                "PRICE": "SELECT",
                "REASON": "PRICE",
                "CONFIRM": "REASON",
            }

            prior = previous.get(step)

            if prior is None:
                self.flow = None
                await self.reply(
                    "TRAIL dibatalkan."
                )
                return

            self.flow["step"] = prior
            await self.reply(
                self._render_trail()
            )
            return

        if kind == "AUTO":
            if self._auto_task is not None and not self._auto_task.done():
                self._auto_task.cancel()
                try:
                    await self._auto_task
                except asyncio.CancelledError:
                    pass
            self._auto_task = None
            self._auto_job_id = None
            self.flow = None
            await self.reply(
                "❌ AUTO dibatalkan."
            )
            return

        if kind == "FILLED":
            self.flow = None
            await self.reply(
                "FILLED dibatalkan."
            )
            return

        if kind == "WRONG":
            self.flow = None
            await self.reply(
                "WRONG dibatalkan."
            )
            return

        if kind == "REMOVE":
            if self.flow["step"] == "SELECT":
                self.flow = None
                await self.reply(
                    "REMOVE dibatalkan."
                )
                return

            self.flow["step"] = "SELECT"
            self.flow["data"] = {}

            await self.reply(
                self._render_remove()
            )
            return

        if kind == "DEL":
            if self.flow["step"] == "SELECT":
                self.flow = None
                await self.reply(
                    "DEL dibatalkan."
                )
                return

            self.flow["step"] = "SELECT"

            await self.reply(
                self._render_del()
            )
            return

        self.flow = None

        await self.reply(
            "Sesi dibatalkan."
        )

    # --------------------------------------------------------
    # TRAIL
    # --------------------------------------------------------

    def _trade_list_text(
        self,
    ) -> list[tuple[int, Trade]]:
        return list(
            enumerate(
                self.active_trades.values(),
                start=1,
            )
        )

    def _render_trail(self) -> str:
        if not self.flow:
            return ""

        step = self.flow["step"]

        if step == "SELECT":
            lines = [
                "TRAIL",
                "",
                "Pilih setup:",
            ]

            items = self._trade_list_text()

            if not items:
                return (
                    "TRAIL\n\n"
                    "Tidak ada active setup."
                )

            for number, trade in items:
                snapshot = self.prices.get(
                    trade.pair
                )

                current = (
                    decimal_to_str(snapshot.price)
                    if snapshot
                    else "-"
                )

                lines.append(
                    f"{number}. "
                    f"{trade.pair} "
                    f"{trade.direction} | "
                    f"{trade.status} | "
                    f"Now: {current}"
                )

            lines.append(
                "\nKetik nomor setup."
            )

            return "\n".join(lines)

        trade: Trade = self.flow[
            "data"
        ]["trade"]

        snapshot = self.prices.get(
            trade.pair
        )

        current = (
            decimal_to_str(snapshot.price)
            if snapshot
            else "-"
        )

        lines = [
            "TRAIL",
            "",
            f"Pair: {trade.pair}",
            f"Direction: {trade.direction.title()}",
            f"Status: {trade.status}",
            f"Entry: {decimal_to_str(trade.entry)}",
            f"Current: {current}",
            f"SL Now: {decimal_to_str(trade.sl)}",
            f"New SL: {decimal_to_str(self.flow['data'].get('new_sl'))}",
            f"Reason: {self.flow['data'].get('reason') or '-'}",
            "",
        ]

        prompts = {
            "PRICE": "New SL:",
            "REASON": "Reason Trailing:",
            "CONFIRM": (
                "Confirm perubahan SL?\n"
                "1. Yes\n"
                "2. No"
            ),
        }

        lines.append(
            prompts.get(
                self.flow["step"],
                "",
            )
        )

        return "\n".join(lines)

    async def _start_trail(self) -> None:
        if self.flow:
            await self.reply(
                "Masih ada sesi yang sedang berjalan.\n"
                "Gunakan /back terlebih dahulu."
            )
            return

        if not self.active_trades:
            await self.reply(
                "Tidak ada active setup."
            )
            return

        self.flow = {
            "kind": "TRAIL",
            "step": "SELECT",
            "data": {},
        }

        await self.reply(
            self._render_trail()
        )

    async def _handle_trail_input(
        self,
        text: str,
    ) -> None:
        if not self.flow:
            return

        step = self.flow["step"]
        data = self.flow["data"]

        try:
            if step == "SELECT":
                number = safe_int(
                    text,
                    "Nomor setup",
                )

                items = self._trade_list_text()

                if not 1 <= number <= len(items):
                    raise ValueError(
                        "Nomor setup tidak valid."
                    )

                _index, trade = items[
                    number - 1
                ]

                data["trade"] = trade
                self.flow["step"] = "PRICE"

                await self.reply(
                    self._render_trail()
                )
                return

            if step == "PRICE":
                trade: Trade = data["trade"]

                new_sl = parse_decimal(
                    text
                )

                self._validate_price(
                    trade.pair,
                    new_sl,
                )

                if trade.direction == "BUY":
                    if not new_sl > trade.sl:
                        raise ValueError(
                            "Untuk Buy, New SL harus "
                            "lebih tinggi dari SL sekarang."
                        )
                else:
                    if not new_sl < trade.sl:
                        raise ValueError(
                            "Untuk Sell, New SL harus "
                            "lebih rendah dari SL sekarang."
                        )

                if trade.status == "PENDING":
                    # Sebelum entry, SL tetap harus berada
                    # di sisi protektif terhadap Entry.
                    if trade.direction == "BUY":
                        if new_sl >= trade.entry:
                            raise ValueError(
                                "Setup Pending Buy: New SL harus "
                                "tetap di bawah Entry."
                            )
                    else:
                        if new_sl <= trade.entry:
                            raise ValueError(
                                "Setup Pending Sell: New SL harus "
                                "tetap di atas Entry."
                            )

                elif trade.status == "FILLED":
                    snapshot = self.prices.get(
                        trade.pair
                    )

                    if snapshot is None or not snapshot.live:
                        raise ValueError(
                            "Harga live symbol belum tersedia/"
                            "sudah stale. Tunggu WebSocket LIVE."
                        )

                    if trade.direction == "BUY":
                        if new_sl >= snapshot.price:
                            raise ValueError(
                                "New SL Buy harus berada "
                                "di bawah Current Price."
                            )
                    else:
                        if new_sl <= snapshot.price:
                            raise ValueError(
                                "New SL Sell harus berada "
                                "di atas Current Price."
                            )

                if trade.direction == "BUY":
                    if new_sl >= trade.tp:
                        raise ValueError(
                            "New SL tidak boleh berada "
                            "di atas/menyamai TP."
                        )
                else:
                    if new_sl <= trade.tp:
                        raise ValueError(
                            "New SL tidak boleh berada "
                            "di bawah/menyamai TP."
                        )

                data["new_sl"] = new_sl
                self.flow["step"] = "REASON"

                await self.reply(
                    self._render_trail()
                )
                return

            if step == "REASON":
                reason = text.strip()

                if not reason:
                    raise ValueError(
                        "Reason Trailing tidak boleh kosong."
                    )

                data["reason"] = reason
                self.flow["step"] = "CONFIRM"

                await self.reply(
                    self._render_trail()
                )
                return

            if step == "CONFIRM":
                if text.strip() == "1":
                    await self._confirm_trail()
                    return

                if text.strip() == "2":
                    self.flow = None

                    await self.reply(
                        "TRAIL dibatalkan."
                    )
                    return

                raise ValueError(
                    "Jawab 1 untuk Yes atau 2 untuk No."
                )

        except Exception as exc:
            await self.reply(
                f"❌ {exc}\n\n"
                f"{self._render_trail()}"
            )

    async def _confirm_trail(self) -> None:
        data = self.flow["data"]
        trade: Trade = data["trade"]

        new_sl: Decimal = data["new_sl"]
        old_sl = trade.sl

        snapshot = self.prices.get(
            trade.pair
        )

        current_price = (
            snapshot.price
            if snapshot
            else None
        )

        trail_item = {
            "old_sl": decimal_to_str(
                old_sl
            ),
            "new_sl": decimal_to_str(
                new_sl
            ),
            "price": decimal_to_str(
                current_price
            ),
            "reason": data["reason"],
            "timestamp": iso_utc(),
            "timestamp_wib": format_wib(
                now_utc()
            ),
        }

        if trade.real_enabled and trade.status == "FILLED":
            replaced = await self._replace_real_sl(trade, new_sl)
            if not replaced:
                raise RuntimeError(
                    "SL baru belum berhasil dikonfirmasi. SL lama tetap dipertahankan; trailing belum di-commit."
                )
        else:
            trade.sl = new_sl

        trade.trailing = True
        trade.trail_history.append(
            trail_item
        )

        await self._record_event(
            trade,
            "TRAIL",
            event_price=current_price,
            reason=data["reason"],
            extra={
                "old_sl": decimal_to_str(
                    old_sl
                ),
                "new_sl": decimal_to_str(
                    new_sl
                ),
            },
        )

        self.flow = None

        await self.reply(
            "✅ TRAILING DIPERBARUI\n\n"
            f"Pair: {trade.pair}\n"
            f"Direction: {trade.direction.title()}\n"
            f"Old SL: {decimal_to_str(old_sl)}\n"
            f"New SL: {decimal_to_str(new_sl)}\n"
            f"Reason: {data['reason']}\n"
            f"Status: {trade.status}\n"
            "Mode: TRAILING"
        )

    # --------------------------------------------------------
    # MANUAL FILL
    # --------------------------------------------------------

    def _render_filled(self) -> str:
        """Tampilkan daftar /trade dan pilih setup untuk fill manual."""
        if not self.flow or self.flow.get("kind") != "FILLED":
            return ""

        items = self._trade_list_text()

        if not items:
            return (
                "FILLED\n\n"
                "Tidak ada active setup."
            )

        lines = [
            "╭──────────────────╮",
            "│  ✅ MANUAL FILLED │",
            "╰──────────────────╯",
            "",
            "Daftar setup aktif:",
            "",
        ]

        for number, trade in items:
            lines.append(
                self._render_trade(number, trade)
            )
            lines.append("")

        lines.extend([
            "Pilih nomor setup yang entry-nya",
            "sudah benar-benar terjadi / sudah running.",
            "",
            "Hanya status PENDING yang bisa diubah.",
            "Ketik nomor setup.",
        ])

        return "\n".join(lines)

    async def _start_filled(self) -> None:
        if self.flow:
            await self.reply(
                "Masih ada sesi yang sedang berjalan.\n"
                "Gunakan /back terlebih dahulu."
            )
            return

        if not self.active_trades:
            await self.reply(
                "Tidak ada active setup."
            )
            return

        if not any(
            trade.status == "PENDING"
            for trade in self.active_trades.values()
        ):
            await self.reply(
                "✅ Tidak ada setup PENDING yang bisa diubah menjadi FILLED."
            )
            return

        self.flow = {
            "kind": "FILLED",
            "step": "SELECT",
            "data": {},
        }

        await self.reply(
            self._render_filled()
        )

    async def _handle_filled_input(
        self,
        text: str,
    ) -> None:
        if not self.flow or self.flow.get("kind") != "FILLED":
            return

        try:
            if self.flow["step"] != "SELECT":
                self.flow = None
                return

            number = safe_int(
                text,
                "Nomor setup",
            )

            items = self._trade_list_text()

            if not 1 <= number <= len(items):
                raise ValueError(
                    "Nomor setup tidak valid."
                )

            _index, trade = items[number - 1]

            current = self.active_trades.get(trade.trade_id)
            if current is None:
                raise ValueError(
                    "Setup sudah tidak aktif. Gunakan /trade untuk melihat daftar terbaru."
                )
            if current.status != "PENDING":
                raise ValueError(
                    f"Setup {current.pair} sudah berstatus {current.status}. Hanya PENDING yang bisa diproses /filled."
                )

            if current.real_enabled:
                if not self.real_mode:
                    raise ValueError("REAL setup membutuhkan /real on untuk verifikasi /filled.")
                confirmed = await self._confirm_real_fill(current)
                if not confirmed:
                    raise ValueError(
                        f"Position Binance {current.pair} belum terkonfirmasi. /filled tidak membuat posisi baru."
                    )
                trade = current
            else:
                async with self._trade_lock:
                    current.status = "FILLED"
                    current.filled_at = now_utc()
                    current.fill_price = current.entry
                    current.pnl_percent = Decimal("0")
                    trade = current

            snapshot = self.prices.get(trade.pair)
            current_market_price = (
                snapshot.price
                if snapshot is not None
                else None
            )

            await self._record_event(
                trade,
                "FILLED",
                event_price=current_market_price,
                reason=(
                    "Manual /filled — entry dianggap sudah terjadi "
                    "dan setup sudah running."
                ),
                extra={
                    "source": "MANUAL /filled",
                    "fill_price": decimal_to_str(trade.entry),
                    "market_price_at_manual_fill": decimal_to_str(
                        current_market_price
                    ),
                },
            )

            self.flow = None

            await self.reply(
                "✅ MANUAL ENTRY FILLED\n\n"
                f"Pair: {trade.pair}\n"
                f"Direction: {trade.direction.title()}\n"
                f"Entry: {decimal_to_str(trade.entry)}\n"
                f"Fill Price: {decimal_to_str(trade.entry)}\n"
                f"Current saat /filled: {decimal_to_str(current_market_price) or '-'}\n"
                "Source: /filled (manual)\n"
                "Status: FILLED\n\n"
                "Bot sekarang menganggap posisi sudah running "
                "dan melanjutkan monitoring SL/TP."
            )

        except Exception as exc:
            await self.reply(
                f"❌ {exc}\n\n"
                f"{self._render_filled()}"
            )

    # --------------------------------------------------------
    # WRONG — remove active setup WITHOUT ANY HISTORY RECORD
    # --------------------------------------------------------

    def _render_wrong(self) -> str:
        if not self.flow:
            return ""

        items = self._trade_list_text()

        if not items:
            return (
                "WRONG\n\n"
                "Tidak ada active setup."
            )

        lines = [
            "⚠️ WRONG",
            "",
            "Pilih setup yang benar-benar salah:",
            "",
        ]

        for number, trade in items:
            lines.append(
                f"{number}. "
                f"{trade.pair} "
                f"{trade.direction} | "
                f"{trade.status}"
            )

        lines.extend([
            "",
            "Setup akan dihapus dari active session.",
            "Tidak dibuat record TRADE, EVENT, atau JOURNAL.",
            "",
            "Ketik nomor setup.",
        ])

        return "\n".join(lines)

    async def _start_wrong(self) -> None:
        if self.flow:
            await self.reply(
                "Masih ada sesi yang sedang berjalan.\n"
                "Gunakan /back terlebih dahulu."
            )
            return

        if not self.active_trades:
            await self.reply(
                "Tidak ada active setup."
            )
            return

        self.flow = {
            "kind": "WRONG",
            "step": "SELECT",
            "data": {},
        }

        await self.reply(
            self._render_wrong()
        )

    async def _persist_active_setups_silently(self) -> None:
        """Persist current active setups without creating history records/events."""
        async with self._trade_lock:
            records = [
                trade.to_record()
                for trade in self.active_trades.values()
                if trade.result is None and trade.status in {
                    "PENDING",
                    "FILLED",
                }
            ]

        payload = {
            "version": 1,
            "saved_at": iso_utc(),
            "saved_at_wib": format_wib(now_utc()),
            "count": len(records),
            "setups": records,
        }

        content = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        await self.github.replace_file(
            SETUPS_PATH,
            content,
            f"wrong: update active setup snapshot ({len(records)} remaining)",
        )

    async def _handle_wrong_input(
        self,
        text: str,
    ) -> None:
        if not self.flow:
            return

        try:
            if self.flow["step"] != "SELECT":
                self.flow = None
                return

            number = safe_int(
                text,
                "Nomor setup",
            )

            items = self._trade_list_text()

            if not 1 <= number <= len(items):
                raise ValueError(
                    "Nomor setup tidak valid."
                )

            _index, trade = items[number - 1]
            trade_id = trade.trade_id

            # WRONG tetap tidak membuat history, tetapi REAL exposure wajib
            # dibersihkan lebih dulu agar penghapusan "tanpa catatan" tidak
            # meninggalkan order/position Binance.
            if trade.real_enabled:
                if not self.real_mode:
                    raise ValueError("Setup REAL membutuhkan /real on sebelum WRONG.")
                await self._real_delete_cleanup(copy.deepcopy(trade))

            # WRONG sengaja tidak memanggil _finalize_trade(),
            # _record_event(), atau _write_history().
            self.active_trades.pop(
                trade_id,
                None,
            )

            try:
                # Bila setup pernah disimpan dengan /setup, snapshot GitHub
                # harus ikut menghapus setup yang salah agar /open tidak
                # menghidupkannya kembali. Ini bukan history.
                await self._persist_active_setups_silently()
            except Exception:
                # Jangan biarkan RAM dan snapshot GitHub berbeda jika write gagal.
                self.active_trades[trade_id] = trade
                raise

            if not any(
                item.pair == trade.pair
                for item in self.active_trades.values()
            ):
                try:
                    await self.ws.remove_symbol(trade.pair)
                except Exception:
                    log.exception(
                        "Gagal unsubscribe WebSocket %s setelah WRONG.",
                        trade.pair,
                    )

            self.flow = None
            await self._resume_scan_after_capacity_change()

            await self.reply(
                "🚫 SETUP DITANDAI WRONG\n\n"
                f"Trade ID: {trade.trade_id}\n"
                f"Pair: {trade.pair}\n"
                f"Direction: {trade.direction.title()}\n"
                "History: TIDAK DICATAT\n"
                "Events: TIDAK DICATAT\n"
                "Journal: TIDAK DICATAT\n\n"
                "Setup dihapus dari active session."
            )

        except Exception as exc:
            await self.reply(
                f"❌ WRONG gagal.\n\n{exc}\n\n"
                f"{self._render_wrong()}"
            )

    # --------------------------------------------------------
    # REMOVE — delete existing HISTORY TRADE DATA
    # --------------------------------------------------------

    def _history_list_for_remove(self) -> list[tuple[int, dict[str, Any]]]:
        rows: list[tuple[int, dict[str, Any]]] = []

        for number, record in enumerate(reversed(self.history_records), start=1):
            if isinstance(record, dict):
                rows.append((number, record))

        return rows

    def _render_remove(self) -> str:
        if not self.flow:
            return ""

        rows = self._history_list_for_remove()

        if not rows:
            return (
                "🧹 REMOVE HISTORY\n\n"
                "Tidak ada data trade di history."
            )

        lines = [
            "🧹 REMOVE HISTORY",
            "",
            "Pilih data trade yang ingin dihapus:",
            "",
        ]

        display_limit = 20
        for number, record in rows[:display_limit]:
            trade_id = str(record.get("trade_id") or "-")
            pair = str(record.get("pair") or "-")
            direction = str(record.get("direction") or "-")
            result = str(record.get("result") or "-")
            closed = str(
                record.get("closed_at")
                or record.get("created_at")
                or "-"
            )

            lines.append(
                f"{number}. {pair} {direction} | {result}"
            )
            lines.append(
                f"   ID: {trade_id}"
            )
            lines.append(
                f"   Time: {closed}"
            )

        if len(rows) > display_limit:
            lines.extend([
                "",
                f"... dan {len(rows) - display_limit} data lainnya.",
            ])

        lines.extend([
            "",
            "Data akan dihapus dari:",
            "- data/trades.json",
            "- data/events.jsonl",
            "- data/trade_history.md",
            "",
            "Ketik nomor data.",
        ])

        return "\n".join(lines)

    @staticmethod
    def _remove_trade_markdown_block(
        content: str,
        trade_id: str,
    ) -> tuple[str, bool]:
        marker = f"Trade ID: `{trade_id}`"
        chunks = content.split("\n---\n\n")

        if len(chunks) == 1:
            return content, False

        kept = [chunks[0]]
        removed = False

        for block in chunks[1:]:
            if marker in block:
                removed = True
                continue
            kept.append(block)

        if not removed:
            return content, False

        return "\n---\n\n".join(kept), True

    async def _remove_history_trade(self, trade_id: str) -> dict[str, int | bool]:
        """Remove one historical trade and all related event/journal data.

        The live GitHub files are re-read before mutation so the operation
        targets current repository state rather than a stale RAM snapshot.
        """
        trade_id = str(trade_id or "").strip()
        if not trade_id:
            raise ValueError("Trade ID tidak valid.")

        async with self._history_lock:
            raw_trades, _ = await self.github.get_file(HISTORY_TRADES_PATH)
            raw_events, _ = await self.github.get_file(HISTORY_EVENTS_PATH)
            raw_journal, _ = await self.github.get_file(HISTORY_MARKDOWN_PATH)

            if not raw_trades:
                raise ValueError("History trade kosong di GitHub.")

            try:
                trade_records = json.loads(raw_trades.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"{HISTORY_TRADES_PATH} bukan JSON valid."
                ) from exc

            if not isinstance(trade_records, list):
                raise RuntimeError(
                    f"{HISTORY_TRADES_PATH} harus berisi JSON list."
                )

            removed_trade_count = sum(
                1
                for record in trade_records
                if isinstance(record, dict)
                and str(record.get("trade_id") or "") == trade_id
            )

            if removed_trade_count == 0:
                raise ValueError(
                    f"Trade ID {trade_id} tidak ditemukan di history GitHub."
                )

            filtered_trades = [
                record
                for record in trade_records
                if not (
                    isinstance(record, dict)
                    and str(record.get("trade_id") or "") == trade_id
                )
            ]

            event_lines: list[str] = []
            removed_event_count = 0

            if raw_events:
                for line in raw_events.decode(
                    "utf-8",
                    errors="replace",
                ).splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue

                    try:
                        item = json.loads(stripped)
                    except json.JSONDecodeError:
                        # Preserve existing invalid event lines instead of
                        # silently deleting unrelated data.
                        event_lines.append(line)
                        continue

                    if (
                        isinstance(item, dict)
                        and str(item.get("trade_id") or "") == trade_id
                    ):
                        removed_event_count += 1
                        continue

                    event_lines.append(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )

            if raw_events is not None:
                filtered_events = (
                    "\n".join(event_lines) + ("\n" if event_lines else "")
                ).encode("utf-8")
            else:
                filtered_events = b""

            journal_text = (
                raw_journal.decode("utf-8", errors="replace")
                if raw_journal
                else ""
            )
            filtered_journal_text, removed_journal = self._remove_trade_markdown_block(
                journal_text,
                trade_id,
            )

            trades_bytes = json.dumps(
                filtered_trades,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")

            events_bytes = filtered_events
            journal_bytes = filtered_journal_text.encode("utf-8")

            # Keep GitHub writes in one history lock so no concurrent history
            # writer can interleave with this removal.
            await self.github.replace_file(
                HISTORY_TRADES_PATH,
                trades_bytes,
                f"remove history trade: {trade_id}",
            )

            await self.github.replace_file(
                HISTORY_EVENTS_PATH,
                events_bytes,
                f"remove history events: {trade_id}",
            )

            if raw_journal is not None:
                await self.github.replace_file(
                    HISTORY_MARKDOWN_PATH,
                    journal_bytes,
                    f"remove history journal: {trade_id}",
                )

            # analysis/* is derived from history. Remove stale exports so
            # /analyze will regenerate them from the current dataset.
            analysis_deleted = 0
            for path in (ANALYSIS_JSON_PATH, ANALYSIS_MD_PATH):
                try:
                    if await self.github.delete_file(
                        path,
                        f"remove history: invalidate {path}",
                    ):
                        analysis_deleted += 1
                except Exception:
                    log.exception(
                        "Gagal menghapus derived analysis %s setelah REMOVE %s.",
                        path,
                        trade_id,
                    )

            # Update RAM only after all primary history writes succeeded.
            self.history_records = filtered_trades
            self.history_events = []
            self._history_dirty = False
            for item in event_lines:
                try:
                    parsed = json.loads(item)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    self.history_events.append(parsed)

            self.last_history_refresh = now_utc()

            return {
                "removed_trades": removed_trade_count,
                "removed_events": removed_event_count,
                "removed_journal": bool(removed_journal),
                "analysis_deleted": analysis_deleted,
            }

    async def _start_remove(self) -> None:
        if self.flow:
            await self.reply(
                "Masih ada sesi yang sedang berjalan.\n"
                "Gunakan /back terlebih dahulu."
            )
            return

        await self._load_history()

        if not self.history_records:
            await self.reply(
                "Tidak ada data trade di history."
            )
            return

        self.flow = {
            "kind": "REMOVE",
            "step": "SELECT",
            "data": {},
        }

        await self.reply(
            self._render_remove()
        )

    async def _handle_remove_input(
        self,
        text: str,
    ) -> None:
        if not self.flow:
            return

        try:
            step = self.flow["step"]

            if step == "SELECT":
                number = safe_int(
                    text,
                    "Nomor data",
                )

                rows = self._history_list_for_remove()
                if not 1 <= number <= len(rows):
                    raise ValueError(
                        "Nomor data tidak valid."
                    )

                _display_number, record = rows[number - 1]
                trade_id = str(record.get("trade_id") or "").strip()
                if not trade_id:
                    raise ValueError(
                        "Data history yang dipilih tidak memiliki Trade ID."
                    )

                self.flow["data"] = {
                    "trade_id": trade_id,
                    "record": copy.deepcopy(record),
                }
                self.flow["step"] = "CONFIRM"

                await self.reply(
                    "⚠️ KONFIRMASI REMOVE\n\n"
                    f"Trade ID: {trade_id}\n"
                    f"Pair: {record.get('pair') or '-'}\n"
                    f"Direction: {record.get('direction') or '-'}\n"
                    f"Result: {record.get('result') or '-'}\n"
                    f"Created: {record.get('created_at') or '-'}\n\n"
                    "Data ini akan dihapus dari trades, events, dan journal.\n"
                    "File analysis lama juga dihapus karena menjadi stale.\n\n"
                    "1. HAPUS PERMANEN\n"
                    "2. BATAL"
                )
                return

            if step == "CONFIRM":
                answer = text.strip()

                if answer == "2":
                    self.flow = None
                    await self.reply(
                        "✅ REMOVE dibatalkan."
                    )
                    return

                if answer != "1":
                    raise ValueError(
                        "Jawab 1 untuk hapus permanen atau 2 untuk batal."
                    )

                trade_id = str(
                    self.flow["data"].get("trade_id") or ""
                ).strip()

                # Simpan konteks flow. Saat network I/O berjalan kita tidak
                # ingin /back mengubah state di tengah operasi, tetapi jika
                # operasi gagal kita kembalikan user ke menu REMOVE yang fresh.
                self.flow = None

                try:
                    result = await self._remove_history_trade(trade_id)
                except Exception:
                    self.flow = {
                        "kind": "REMOVE",
                        "step": "SELECT",
                        "data": {},
                    }
                    try:
                        await self._load_history()
                    except Exception:
                        pass
                    raise

                await self.reply(
                    "🧹 HISTORY DIHAPUS\n\n"
                    f"Trade ID: {trade_id}\n"
                    f"Trade record: {result['removed_trades']} dihapus\n"
                    f"Event: {result['removed_events']} dihapus\n"
                    f"Journal: {'dihapus' if result['removed_journal'] else 'tidak ditemukan'}\n"
                    f"Analysis lama: {result['analysis_deleted']} file dihapus\n\n"
                    "Data tersebut tidak lagi dihitung oleh /stats maupun /analyze."
                )
                return

        except Exception as exc:
            await self.reply(
                f"❌ REMOVE gagal.\n\n{exc}\n\n"
                f"{self._render_remove()}"
            )

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    def _render_del(self) -> str:
        if not self.flow:
            return ""

        items = self._trade_list_text()

        if not items:
            return (
                "DEL\n\n"
                "Tidak ada active setup."
            )

        lines = [
            "DEL",
            "",
            "Pilih setup:",
        ]

        for number, trade in items:
            lines.append(
                f"{number}. "
                f"{trade.pair} "
                f"{trade.direction} | "
                f"{trade.status}"
            )

        lines.append(
            "\nKetik nomor setup."
        )

        return "\n".join(lines)

    async def _start_del(self) -> None:
        if self.flow:
            await self.reply(
                "Masih ada sesi yang sedang berjalan.\n"
                "Gunakan /back terlebih dahulu."
            )
            return

        if not self.active_trades:
            await self.reply(
                "Tidak ada active setup."
            )
            return

        self.flow = {
            "kind": "DEL",
            "step": "SELECT",
            "data": {},
        }

        await self.reply(
            self._render_del()
        )

    async def _handle_del_input(
        self,
        text: str,
    ) -> None:
        if not self.flow:
            return

        try:
            if self.flow["step"] != "SELECT":
                self.flow = None
                return

            number = safe_int(
                text,
                "Nomor setup",
            )

            items = self._trade_list_text()

            if not 1 <= number <= len(items):
                raise ValueError(
                    "Nomor setup tidak valid."
                )

            _index, trade = items[
                number - 1
            ]

            # Snapshot sebelum delete.
            trade_copy = copy.deepcopy(trade)

            if trade_copy.real_enabled:
                if not self.real_mode:
                    raise ValueError("Setup REAL tidak dapat /del saat /real OFF. Aktifkan /real untuk cleanup Binance.")
                await self._real_delete_cleanup(trade_copy)

            await self._finalize_trade(
                trade_copy,
                result="DELETED",
                exit_price=None,
                reason="Dihapus oleh user.",
                send_notification=False,
            )

            self.active_trades.pop(
                trade.trade_id,
                None,
            )

            if not any(
                item.pair == trade.pair
                for item in self.active_trades.values()
            ):
                await self.ws.remove_symbol(
                    trade.pair
                )

            self.flow = None
            await self._resume_scan_after_capacity_change()

            await self.reply(
                "🗑️ SETUP DIHAPUS\n\n"
                f"Pair: {trade.pair}\n"
                f"Direction: {trade.direction.title()}\n"
                "Result: DELETED"
            )

        except Exception as exc:
            await self.reply(
                f"❌ {exc}\n\n"
                f"{self._render_del()}"
            )

    # --------------------------------------------------------
    # FORCE CLOSE
    # --------------------------------------------------------

    async def _close_one_trade(self, trade: Trade) -> tuple[bool, str]:
        """Force-close one active setup, using Binance truth when REAL is attached."""
        current = self.active_trades.get(trade.trade_id)
        if current is None:
            return False, "Setup sudah tidak aktif."

        if current.real_enabled:
            if not self.real_mode:
                raise ValueError("Setup REAL membutuhkan /real on untuk /close.")
            if current.status == "PENDING":
                # Cancel only this bot-owned entry; do not invent a fill.
                if current.entry_order_id is not None or current.entry_client_order_id:
                    try:
                        await self.real.cancel_order(
                            current.pair,
                            order_id=current.entry_order_id,
                            client_order_id=(current.entry_client_order_id if current.entry_order_id is None else None),
                        )
                    except BinanceAPIError as exc:
                        if exc.code not in {-2011, -2013}:
                            raise
                current.real_state = "REAL_CLOSED"
                await self._finalize_trade(
                    copy.deepcopy(current),
                    result="MANUAL_CLOSE_PENDING",
                    exit_price=None,
                    reason="Manual /close pada setup REAL PENDING sebelum entry.",
                    send_notification=False,
                )
                return True, "REAL PENDING dibatalkan."

            if current.status != "FILLED":
                return False, f"Status setup tidak dapat di-close: {current.status}."

            position = await self.real.get_position(current.pair, current.direction)
            if position is None:
                raise BinanceAPIError(
                    f"Position Binance {current.pair} tidak ditemukan saat /close.",
                    endpoint="/fapi/v2/positionRisk",
                )
            close_response = await self.real.place_market_close(
                symbol=current.pair,
                position=position,
                entry_direction=current.direction,
            )

            exit_price = None
            for _ in range(3):
                await asyncio.sleep(0.25)
                remaining = await self.real.get_position(current.pair, current.direction)
                if remaining is None:
                    break
            else:
                raise BinanceAPIError(
                    f"Position Binance {current.pair} masih terbuka setelah /close.",
                    endpoint="/fapi/v2/positionRisk",
                )

            try:
                if close_response.get("avgPrice") not in (None, "", "0"):
                    exit_price = parse_decimal(str(close_response["avgPrice"]))
                elif close_response.get("price") not in (None, "", "0"):
                    exit_price = parse_decimal(str(close_response["price"]))
            except (InvalidOperation, ValueError):
                exit_price = None

            if exit_price is None:
                snapshot = self.prices.get(current.pair)
                if snapshot is None or not snapshot.live:
                    raise ValueError("Exit price real tidak tersedia dan harga WebSocket stale.")
                exit_price = snapshot.price

            pnl = pct_change(current.direction, current.fill_price or current.entry, exit_price)
            result = "TP" if pnl > 0 else "SL" if pnl < 0 else "MANUAL_CLOSE"
            reason = (
                "Manual /close REAL. PnL positif → TP."
                if result == "TP" else
                "Manual /close REAL. PnL negatif → SL."
                if result == "SL" else
                "Manual /close REAL. PnL 0% → MANUAL_CLOSE."
            )
            current.real_state = "REAL_CLOSED"
            await self._cancel_bot_algo_orders(current)
            await self._finalize_trade(
                copy.deepcopy(current),
                result=result,
                exit_price=exit_price,
                reason=reason,
                send_notification=False,
            )
            return True, f"REAL FILLED ditutup sebagai {result}."

        if current.status == "PENDING":
            await self._finalize_trade(
                copy.deepcopy(current),
                result="MANUAL_CLOSE_PENDING",
                exit_price=None,
                reason="Manual /close pada setup PENDING sebelum entry.",
                send_notification=False,
            )
            return True, "PENDING ditutup."

        if current.status != "FILLED":
            return False, f"Status setup tidak dapat di-close: {current.status}."

        snapshot = self.prices.get(current.pair)
        if snapshot is None or not snapshot.live:
            raise ValueError(
                f"{current.pair}: harga live WebSocket belum tersedia/stale; FILLED tidak ditutup agar PnL tidak salah."
            )
        exit_price = snapshot.price
        pnl = pct_change(current.direction, current.fill_price or current.entry, exit_price)
        result = "TP" if pnl > 0 else "SL" if pnl < 0 else "MANUAL_CLOSE"
        reason = (
            "Manual /close. PnL positif, sehingga hasil dicatat sebagai TP "
            f"(PnL {format_pct(pnl)})."
            if result == "TP" else
            "Manual /close. PnL negatif, sehingga hasil dicatat sebagai SL "
            f"(PnL {format_pct(pnl)})."
            if result == "SL" else
            "Manual /close. PnL tepat 0%; tidak dikategorikan TP atau SL."
        )
        await self._finalize_trade(
            copy.deepcopy(current),
            result=result,
            exit_price=exit_price,
            reason=reason,
            send_notification=False,
        )
        return True, f"{current.status} ditutup sebagai {result}."

    async def _handle_close_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            raise ValueError("Gunakan /close PAIR atau /close all.")

        target = parts[1].strip()
        is_batch = target.lower() == "all"

        if is_batch:
            trades = list(self.active_trades.values())
            if not trades:
                await self.reply("🔒 /close all: tidak ada active setup.")
                return

            success = 0
            failed = 0
            details: list[str] = []
            self._close_batch_in_progress = True
            try:
                for trade in trades:
                    try:
                        ok, detail = await self._close_one_trade(trade)
                        if ok:
                            success += 1
                        else:
                            failed += 1
                        details.append(f"{trade.pair} → {detail}")
                    except Exception as exc:
                        failed += 1
                        log.exception("[CLOSE] gagal menutup %s", trade.trade_id)
                        details.append(f"{trade.pair} → ERROR: {exc}")
            finally:
                self._close_batch_in_progress = False
                # Setelah semua close diproses, baru scanner boleh resume.
                await self._resume_scan_after_capacity_change()

            await self.reply(
                "🔒 CLOSE ALL SELESAI\n\n"
                f"Diproses: {len(trades)}\n"
                f"Berhasil: {success}\n"
                f"Gagal: {failed}\n\n"
                + "\n".join(details)
            )
            return

        pair = normalize_symbol(target)
        matches = [
            trade
            for trade in self.active_trades.values()
            if trade.pair == pair
        ]
        if not matches:
            raise ValueError(f"Tidak ada active setup untuk {pair}.")

        results: list[str] = []
        if len(matches) > 1:
            self._close_batch_in_progress = True
        try:
            for trade in matches:
                try:
                    ok, detail = await self._close_one_trade(trade)
                    results.append(f"{trade.trade_id}: {detail}")
                    if not ok:
                        log.warning("[CLOSE] setup %s tidak ditutup: %s", trade.trade_id, detail)
                except Exception as exc:
                    log.exception("[CLOSE] gagal menutup %s", trade.trade_id)
                    results.append(f"{trade.trade_id}: ERROR: {exc}")
        finally:
            if len(matches) > 1:
                self._close_batch_in_progress = False
                await self._resume_scan_after_capacity_change()

        await self.reply(
            f"🔒 CLOSE {pair}\n\n"
            + "\n".join(results)
        )

    # --------------------------------------------------------
    # TRADE DISPLAY
    # --------------------------------------------------------

    @staticmethod
    def _distance_pct(current: Decimal, target: Decimal) -> Decimal | None:
        if current <= 0 or target <= 0:
            return None
        return abs(target - current) / current * Decimal("100")

    @staticmethod
    def _temporary_trade_pnl_usdt(trade: Trade, price: Decimal) -> Decimal | None:
        if trade.status != "FILLED" or not trade.real_enabled or trade.quantity is None:
            return None
        fill = trade.fill_price or trade.entry
        if fill <= 0 or trade.quantity <= 0:
            return None
        if trade.direction == "BUY":
            return (price - fill) * trade.quantity
        return (fill - price) * trade.quantity

    def _render_trade(
        self,
        number: int,
        trade: Trade,
    ) -> str:
        snapshot = self.prices.get(
            trade.pair
        )

        current_text = "-"
        feed = "NO DATA"
        pnl_text = "-"
        entry_distance_text = "-"
        tp_distance_text = "-"
        sl_distance_text = "-"

        if snapshot:
            current_text = (
                decimal_to_str(snapshot.price)
                or "-"
            )

            if (
                snapshot.live
                and snapshot.source == "REST_REFERENCE"
            ):
                feed = "REFERENCE"
            elif snapshot.live:
                feed = "LIVE"
            elif self.ws.status == "RECONNECTING":
                feed = "RECONNECTING"
            else:
                feed = "STALE"

            if trade.status == "PENDING":
                distance = self._distance_pct(snapshot.price, trade.entry)
                if distance is not None:
                    entry_distance_text = f"{decimal_to_str(distance.quantize(Decimal('0.01')))}%"
            elif trade.status == "FILLED":
                pnl_text = format_pct(
                    pct_change(
                        trade.direction,
                        trade.fill_price or trade.entry,
                        snapshot.price,
                    )
                )
                tp_distance = self._distance_pct(snapshot.price, trade.tp)
                sl_distance = self._distance_pct(snapshot.price, trade.sl)
                if tp_distance is not None:
                    tp_distance_text = f"{decimal_to_str(tp_distance.quantize(Decimal('0.01')))}%"
                if sl_distance is not None:
                    sl_distance_text = f"{decimal_to_str(sl_distance.quantize(Decimal('0.01')))}%"

        direction_icon = "🟢" if trade.direction == "BUY" else "🔴"
        status_icon = {
            "PENDING": "⏳",
            "FILLED": "✅",
        }.get(
            trade.status,
            "•",
        )

        mode = "TRAIL" if trade.trailing else "NORMAL"
        execution = "REAL" if trade.real_enabled else "SIM"
        feed_icon = {
            "LIVE": "📡",
            "REFERENCE": "📍",
            "RECONNECTING": "🔄",
            "STALE": "⚠️",
            "NO DATA": "❔",
        }.get(
            feed,
            "•",
        )

        reason = trade.entry_reason.strip()
        if len(reason) > 55:
            reason = reason[:52] + "..."

        lines = [
            f"╭─ {number} • {trade.pair} ─╮",
            f"│ {direction_icon} {trade.direction}   {status_icon} {trade.status}",
            f"│ {feed_icon} Now   {current_text}  •  {feed}",
            f"│ 🎯 Entry {decimal_to_str(trade.entry)}   →   TP {decimal_to_str(trade.tp)}",
            f"│ 🛡 SL    {decimal_to_str(trade.sl)}",
            f"│ ⛔ Exp   {decimal_to_str(trade.price_exp)}",
            (
                f"│ 📍 Jarak Entry  {entry_distance_text}"
                if trade.status == "PENDING"
                else f"│ 📈 PnL   {pnl_text}"
            ),
            f"│ 🎯 Jarak TP     {tp_distance_text}" if trade.status == "FILLED" else "",
            f"│ 🛡 Jarak SL     {sl_distance_text}" if trade.status == "FILLED" else "",
            f"│ ⚙️ Mode   {mode} / {execution}",
            f"│ 🧠 {reason}",
            "╰────────────────────────╯",
        ]

        return "\n".join(line for line in lines if line != "")

    async def show_trades(self) -> None:
        items = self._trade_list_text()

        if not items:
            await self.reply(
                "╭──────────────╮\n"
                "│  📊 TRADE   │\n"
                "╰──────────────╯\n\n"
                "Tidak ada setup aktif.\n"
                "Gunakan /add untuk membuat setup."
            )
            return

        pending = sum(
            1
            for _number, trade in items
            if trade.status == "PENDING"
        )

        filled = sum(
            1
            for _number, trade in items
            if trade.status == "FILLED"
        )

        trailing = sum(
            1
            for _number, trade in items
            if trade.trailing
        )

        live = sum(
            1
            for trade in self.active_trades.values()
            if (
                self.prices.get(trade.pair)
                and self.prices[trade.pair].live
            )
        )

        temp_pnl_usdt = Decimal("0")
        temp_pnl_count = 0
        stale_filled = 0
        for trade in self.active_trades.values():
            if trade.status != "FILLED" or not trade.real_enabled:
                continue
            snapshot = self.prices.get(trade.pair)
            if snapshot is None:
                stale_filled += 1
                continue
            if not snapshot.live:
                stale_filled += 1
            try:
                pnl_usdt = self._temporary_trade_pnl_usdt(trade, snapshot.price)
                if pnl_usdt is not None:
                    temp_pnl_usdt += pnl_usdt
                    temp_pnl_count += 1
            except Exception:
                log.exception("Gagal menghitung temporary PnL %s.", trade.trade_id)

        balance_row = self.real.cached_usdt_balance() if self.real_mode else None
        balance_text = "-"
        total_balance_text = "-"
        balance_cache_age = "-"
        if balance_row is not None:
            wallet_balance = parse_signed_decimal(str(balance_row.get("balance") or "0"))
            balance_text = self._fmt_usd(wallet_balance)
            total_balance_text = self._fmt_usd(wallet_balance + temp_pnl_usdt)
            if self.real.last_balance_at is not None:
                balance_cache_age = duration_text(max(0.0, time.monotonic() - self.real.last_balance_at))

        blocks = [
            "╭──────────────╮",
            "│  📊 TRADE   │",
            "╰──────────────╯",
            "",
            f"Active  {len(items)}   •   ⏳ {pending}   •   ✅ {filled}",
            f"Trail   {trailing}   •   📡 Live Feed {live}",
            f"💰 Saldo Binance       {balance_text} USDT",
            f"📊 Temporary Total PnL {self._fmt_usd(temp_pnl_usdt)} USDT",
            f"💵 Saldo + Temp PnL    {total_balance_text} USDT",
            f"🧊 Balance Cache       {balance_cache_age} ago • {temp_pnl_count}/{filled} REAL FILLED",
            f"PnL stale/no feed: {stale_filled}",
            "",
        ]

        for number, trade in items:
            blocks.append(
                self._render_trade(
                    number,
                    trade,
                )
            )
            blocks.append("")

        await self.reply(
            "\n".join(blocks).rstrip()
        )

    # --------------------------------------------------------
    # EVENTS / CLOSE
    # --------------------------------------------------------

    async def _record_event(
        self,
        trade: Trade,
        event: str,
        event_price: Decimal | None,
        reason: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        item = {
            "event_id": uuid4().hex,
            "trade_id": trade.trade_id,
            "session_id": trade.session_id,
            "event": event,
            "timestamp": iso_utc(),
            "timestamp_wib": format_wib(
                now_utc()
            ),
            "pair": trade.pair,
            "direction": trade.direction,
            "price": decimal_to_str(
                event_price
            ),
            "reason": reason,
            "extra": extra or {},
        }

        async with self._history_lock:
            self.history_events.append(
                item
            )

            events_jsonl = (
                "".join(
                    json.dumps(
                        event_item,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                    for event_item in self.history_events
                )
            ).encode("utf-8")

            try:
                await self.github.replace_file(
                    HISTORY_EVENTS_PATH,
                    events_jsonl,
                    (
                        f"event: {trade.pair} "
                        f"{event}"
                    ),
                )
            except Exception:
                log.exception(
                    "Gagal menyimpan event %s ke GitHub.",
                    event,
                )

    async def _finalize_trade(
        self,
        trade: Trade,
        result: str,
        exit_price: Decimal | None,
        reason: str,
        send_notification: bool = True,
    ) -> None:
        # State transition harus atomic dan cepat.
        # Semua network I/O dilakukan setelah lock dilepas.
        async with self._trade_lock:
            current = self.active_trades.get(
                trade.trade_id
            )

            if current is None:
                return

            if current.result is not None:
                return

            # SL hasil trailing yang menutup di atas entry dihitung TRAIL (profit terkunci).
            if (
                result == "SL"
                and current.trailing
                and current.fill_price is not None
                and exit_price is not None
                and pct_change(current.direction, current.fill_price, exit_price) >= 0
            ):
                result = "TRAIL"
                reason = f"SL trailing tercapai; profit terkunci. {reason}"

            current.status = "CLOSED"
            current.result = result
            current.closed_at = now_utc()
            current.exit_price = exit_price
            current.result_reason = reason

            if (
                current.fill_price is not None
                and exit_price is not None
            ):
                current.pnl_percent = pct_change(
                    current.direction,
                    current.fill_price,
                    exit_price,
                )

            trade = current

            self.active_trades.pop(
                trade.trade_id,
                None,
            )

            need_unsubscribe = not any(
                active.pair == trade.pair
                for active in self.active_trades.values()
            )

        # _write_history() membuat final event + trade record
        # di bawah satu history lock.
        await self._write_history(
            trade
        )

        if trade.real_enabled and result in {"TP", "SL", "TRAIL"}:
            await self._cleanup_real_leftovers(trade)

        try:
            await self._auto_ban_for_result(trade, result)
        except Exception:
            log.exception("Gagal memperbarui ban otomatis untuk %s %s.", trade.pair, result)

        # Snapshot active setup harus ikut dibersihkan agar /open tidak
        # menghidupkan kembali trade yang sudah ditutup setelah /setup.
        try:
            await self._persist_active_setups_silently()
        except Exception:
            log.exception(
                "Gagal menyinkronkan data/setups.json setelah close %s.",
                trade.trade_id,
            )

        if need_unsubscribe:
            try:
                await self.ws.remove_symbol(
                    trade.pair
                )
            except Exception:
                log.exception(
                    "Gagal unsubscribe WebSocket %s.",
                    trade.pair,
                )

        if send_notification:
            await self._notify_close(
                trade
            )

        if not self._close_batch_in_progress:
            await self._resume_scan_after_capacity_change()

    async def _notify_close(
        self,
        trade: Trade,
    ) -> None:
        result = trade.result or "CLOSED"
        title = {
            "TP": "✅ TP TERCAPAI",
            "SL": "🛑 SL TERCAPAI",
            "TRAIL": "🔒 TRAILING SL TERCAPAI",
            "EXPIRED": "⏳ PRICE EXPIRED",
            "DELETED": "🗑️ DELETED",
            "MANUAL_CLOSE_PENDING": "🔒 PENDING DITUTUP",
            "MANUAL_CLOSE": "🔒 DITUTUP MANUAL",
        }.get(result, "🏁 TRADE CLOSED")

        icon = "🟢" if trade.direction == "BUY" else "🔴"
        mode = "REAL" if trade.real_enabled else "SIMULASI"
        reason = (trade.result_reason or "").strip()[:240]

        if result == "EXPIRED":
            rows = [
                f"{icon} {trade.pair}  •  {trade.direction}",
                f"🎯 Entry    {fmt_price(trade.entry)}",
                f"⛔ Exp      {fmt_price(trade.price_exp)}",
                f"📡 Pemicu   {fmt_price(trade.exit_price)}",
                f"⚙️ Mode     {mode}",
            ]
            if reason:
                rows.append(f"🧠 {reason}")
            await self.reply(card(title, rows))
            return

        pnl = format_pct(trade.pnl_percent) if trade.pnl_percent is not None else "-"
        rows = [
            f"{icon} {trade.pair}  •  {trade.direction}",
            f"🎯 Entry   {fmt_price(trade.fill_price or trade.entry)}",
            f"🏁 Exit    {fmt_price(trade.exit_price)}",
            f"📈 PnL     {pnl}",
        ]
        if trade.filled_at is not None and trade.closed_at is not None:
            held = (trade.closed_at - trade.filled_at).total_seconds()
            rows.append(f"⏱ Durasi  {duration_text(held)}")
        rows.append(f"⚙️ Mode    {mode}")
        if reason:
            rows.append(f"🧠 {reason}")
        await self.reply(card(title, rows))

    async def _fill_trade(
        self,
        trade: Trade,
        price: Decimal,
    ) -> None:
        async with self._trade_lock:
            current = self.active_trades.get(
                trade.trade_id
            )

            if current is None:
                return

            if current.status != "PENDING":
                return

            current.status = "FILLED"
            current.filled_at = now_utc()
            current.fill_price = current.entry
            current.pnl_percent = Decimal("0")

            trade = current

        await self._record_event(
            trade,
            "FILLED",
            event_price=price,
            reason=trade.entry_reason,
            extra={
                "fill_price": decimal_to_str(
                    trade.entry
                ),
            },
        )

        await self._notify_filled(trade)

    async def _notify_filled(self, trade: Trade) -> None:
        icon = "🟢" if trade.direction == "BUY" else "🔴"
        rows = [
            f"{icon} {trade.pair}  •  {trade.direction}",
            f"🎯 Entry   {fmt_price(trade.entry)}",
            f"✅ Fill    {fmt_price(trade.fill_price or trade.entry)}",
            f"🛡 SL      {fmt_price(trade.sl)}",
            f"🏁 TP      {fmt_price(trade.tp)}",
            f"⛔ Exp     {fmt_price(trade.price_exp)}",
        ]
        if trade.real_enabled and trade.quantity is not None:
            rows.append(f"📦 Qty     {fmt_price(trade.quantity)}")
        rows.append(f"⚙️ Mode    {'REAL' if trade.real_enabled else 'SIMULASI'}")
        reason = (trade.entry_reason or "").strip()
        if reason:
            rows.append(f"🧠 {reason[:200]}")
        await self.reply(card("✅ ENTRY FILLED", rows))

    # --------------------------------------------------------
    # LIVE PRICE EVENT
    # --------------------------------------------------------
    # LIVE PRICE EVENT
    # --------------------------------------------------------

    async def _on_price(
        self,
        symbol: str,
        price: Decimal,
        event_time_ms: int,
        aggregate_trade_id: int | None = None,
    ) -> None:
        # Defense-in-depth: jangan pernah memproses market event yang lebih
        # lama dari event terakhir untuk symbol yang sama.
        event_key = (
            int(event_time_ms),
            int(aggregate_trade_id)
            if aggregate_trade_id is not None
            else -1,
        )

        last_key = self._last_market_event_key.get(symbol)
        if last_key is not None and event_key <= last_key:
            return

        self._last_market_event_key[symbol] = event_key

        self.prices[symbol] = PriceSnapshot(
            symbol=symbol,
            price=price,
            event_time_ms=event_time_ms,
            received_at=now_utc(),
            source="WEBSOCKET",
        )
        self._last_live_price[symbol] = price

        # Snapshot daftar trade tanpa menahan lock saat network I/O.
        relevant = [
            trade
            for trade in list(self.active_trades.values())
            if trade.pair == symbol
        ]

        for trade in relevant:
            if trade.trade_id not in self.active_trades:
                continue

            try:
                if trade.status == "PENDING":
                    entry_hit = (
                        price <= trade.entry if trade.direction == "BUY" else price >= trade.entry
                    )
                    exp_hit = (
                        price >= trade.price_exp if trade.direction == "BUY" else price <= trade.price_exp
                    )

                    if trade.real_enabled and self.real_mode:
                        if self.real.cooldown_remaining > 0:
                            # API dibatasi: harga tetap dipantau; fill diantrekan.
                            if entry_hit:
                                self._defer_fill_check(trade, price)
                            continue
                        if entry_hit and self._real_poll_due(trade, "fill", REAL_PENDING_POLL_SECONDS):
                            if await self._confirm_real_fill(trade):
                                continue
                        if exp_hit and trade.status == "PENDING":
                            # Jangan EXPIRED lokal sebelum order real dicek.
                            if not self._real_poll_due(trade, "exp", REAL_PENDING_POLL_SECONDS):
                                continue
                            await self._real_pending_price_exp(trade)
                            if trade.status != "PENDING":
                                continue
                    else:
                        if entry_hit:
                            await self._fill_trade(trade, price)
                            continue

                    if exp_hit and trade.status == "PENDING":
                        await self._finalize_trade(
                            trade,
                            result="EXPIRED",
                            exit_price=price,
                            reason=trade.price_exp_reason,
                        )
                        continue

                elif trade.status == "FILLED":
                    async with self._trade_lock:
                        current = self.active_trades.get(
                            trade.trade_id
                        )

                        if current is None:
                            continue

                        current.pnl_percent = pct_change(
                            current.direction,
                            current.fill_price or current.entry,
                            price,
                        )
                        # Catat ekskursi terbaik/terburuk untuk kalibrasi SL/TP berikutnya.
                        if current.max_favorable_pct is None or current.pnl_percent > current.max_favorable_pct:
                            current.max_favorable_pct = current.pnl_percent
                        if current.max_adverse_pct is None or current.pnl_percent < current.max_adverse_pct:
                            current.max_adverse_pct = current.pnl_percent

                        trade = current

                    if trade.direction == "BUY":
                        sl_hit = price <= trade.sl
                        tp_hit = price >= trade.tp
                    else:
                        sl_hit = price >= trade.sl
                        tp_hit = price <= trade.tp

                    if trade.real_enabled and self.real_mode:
                        if self.real.cooldown_remaining > 0:
                            self._defer_trail(trade, price)
                            continue
                        if sl_hit or tp_hit:
                            result_probe = "SL" if sl_hit else "TP"
                            if await self._real_exit_trigger(trade, result_probe, price):
                                result_price = trade.exit_price or price
                                await self._finalize_trade(
                                    trade,
                                    result=result_probe,
                                    exit_price=result_price,
                                    reason=(trade.sl_reason if result_probe == "SL" else trade.tp_reason),
                                )
                                continue
                        if self._real_poll_due(trade, "protect", REAL_PROTECT_CHECK_SECONDS):
                            await self._verify_real_protection(trade)
                    else:
                        if sl_hit:
                            await self._finalize_trade(
                                trade,
                                result="SL",
                                exit_price=price,
                                reason=trade.sl_reason,
                            )
                            continue
                        if tp_hit:
                            await self._finalize_trade(
                                trade,
                                result="TP",
                                exit_price=price,
                                reason=trade.tp_reason,
                            )
                            continue

                    await self._auto_trail(trade, price)

            except BinanceRateLimitError as exc:
                self._notify_rate_limit(exc, f"PRICE EVENT {symbol}")
            except Exception:
                log.exception(
                    "Gagal memproses price event "
                    f"{symbol} untuk {trade.trade_id}"
                )

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    def _calculate_stats(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate descriptive history statistics.

        Confidence statistics are calculated only from records that actually
        contain a numeric ``strategy_confidence``. "Successfully entered"
        means a historical record with ``filled_at`` present or a TP/SL
        outcome; EXPIRED records without a fill are therefore excluded.
        """
        total_records = len(records)

        tp = sum(
            1
            for record in records
            if record.get("result") in {"TP", "TRAIL"}
        )

        trail_exit = sum(
            1
            for record in records
            if record.get("result") == "TRAIL"
        )

        sl = sum(
            1
            for record in records
            if record.get("result") == "SL"
        )

        expired = sum(
            1
            for record in records
            if record.get("result") == "EXPIRED"
        )

        deleted = sum(
            1
            for record in records
            if record.get("result") == "DELETED"
        )

        manual_close_pending = sum(
            1
            for record in records
            if record.get("result") == "MANUAL_CLOSE_PENDING"
        )

        manual_close_filled = sum(
            1
            for record in records
            if record.get("result") == "MANUAL_CLOSE"
        )

        total_filled = tp + sl

        tp_rate = (
            Decimal(tp)
            / Decimal(total_filled)
            * Decimal("100")
            if total_filled
            else Decimal("0")
        )
        sl_rate = (
            Decimal(sl)
            / Decimal(total_filled)
            * Decimal("100")
            if total_filled
            else Decimal("0")
        )
        win_rate = tp_rate

        pnls: list[Decimal] = []
        for record in records:
            value = record.get("pnl_percent")
            if value is None:
                continue
            try:
                pnls.append(Decimal(str(value)))
            except InvalidOperation:
                continue

        gross_net = sum(pnls, Decimal("0"))

        wins = [value for value in pnls if value > 0]
        losses = [value for value in pnls if value < 0]

        average_win = (
            sum(wins, Decimal("0")) / Decimal(len(wins))
            if wins
            else Decimal("0")
        )
        average_loss = (
            sum(losses, Decimal("0")) / Decimal(len(losses))
            if losses
            else Decimal("0")
        )

        trail_count = sum(
            len(record.get("trail_history") or [])
            for record in records
        )

        def average_decimal(
            values: list[Decimal],
        ) -> Decimal | None:
            if not values:
                return None
            return sum(values, Decimal("0")) / Decimal(len(values))

        def read_decimal(
            record: dict[str, Any],
            key: str,
        ) -> Decimal | None:
            value = record.get(key)
            if value in (None, ""):
                return None
            try:
                return Decimal(str(value))
            except InvalidOperation:
                return None

        confidence_all: list[Decimal] = []
        confidence_tp: list[Decimal] = []
        confidence_sl: list[Decimal] = []
        confidence_entered: list[Decimal] = []

        planned_rr_entered: list[Decimal] = []
        holding_tp: list[Decimal] = []
        holding_sl: list[Decimal] = []
        pending_entered: list[Decimal] = []

        auto_records = 0
        manual_records = 0
        confidence_missing_entered = 0
        entered_records_count = 0

        for record in records:
            confidence = read_decimal(record, "strategy_confidence")
            result = str(record.get("result") or "").upper()
            has_entry = (
                result in {"TP", "SL", "TRAIL"}
                or record.get("filled_at") not in (None, "")
            )

            if has_entry:
                entered_records_count += 1

            if confidence is not None:
                confidence_all.append(confidence)
                if result in {"TP", "TRAIL"}:
                    confidence_tp.append(confidence)
                elif result == "SL":
                    confidence_sl.append(confidence)
                if has_entry:
                    confidence_entered.append(confidence)
            elif has_entry:
                confidence_missing_entered += 1

            if has_entry:
                rr = read_decimal(record, "planned_rr")
                if rr is not None:
                    planned_rr_entered.append(rr)

                holding = read_decimal(record, "holding_seconds")
                if holding is not None:
                    if result in {"TP", "TRAIL"}:
                        holding_tp.append(holding)
                    elif result == "SL":
                        holding_sl.append(holding)

                pending = read_decimal(record, "pending_seconds")
                if pending is not None:
                    pending_entered.append(pending)

            source = str(record.get("strategy_source") or "").upper()
            if source == "AUTO":
                auto_records += 1
            elif source == "MANUAL":
                manual_records += 1

        expired_rate_all = (
            Decimal(expired)
            / Decimal(total_records)
            * Decimal("100")
            if total_records
            else Decimal("0")
        )
        deleted_rate_all = (
            Decimal(deleted)
            / Decimal(total_records)
            * Decimal("100")
            if total_records
            else Decimal("0")
        )

        avg_confidence_all = average_decimal(confidence_all)
        avg_confidence_tp = average_decimal(confidence_tp)
        avg_confidence_sl = average_decimal(confidence_sl)
        avg_confidence_entered = average_decimal(confidence_entered)
        avg_planned_rr_entered = average_decimal(planned_rr_entered)
        avg_holding_tp = average_decimal(holding_tp)
        avg_holding_sl = average_decimal(holding_sl)
        avg_pending_entered = average_decimal(pending_entered)

        confidence_gap_tp_sl: Decimal | None = None
        if avg_confidence_tp is not None and avg_confidence_sl is not None:
            confidence_gap_tp_sl = avg_confidence_tp - avg_confidence_sl

        return {
            "total_records": total_records,
            "total_filled": total_filled,
            "entered_records_count": entered_records_count,
            "tp": tp,
            "trail_exit": trail_exit,
            "sl": sl,
            "tp_rate": tp_rate,
            "sl_rate": sl_rate,
            "expired": expired,
            "expired_rate_all": expired_rate_all,
            "deleted": deleted,
            "deleted_rate_all": deleted_rate_all,
            "manual_close_pending": manual_close_pending,
            "manual_close_filled": manual_close_filled,
            "win_rate": win_rate,
            "gross_pnl_percent": gross_net,
            "average_win_percent": average_win,
            "average_loss_percent": average_loss,
            "trail_count": trail_count,
            "confidence_count_all": len(confidence_all),
            "confidence_count_entered": len(confidence_entered),
            "confidence_missing_entered": confidence_missing_entered,
            "average_confidence_all": avg_confidence_all,
            "average_confidence_tp": avg_confidence_tp,
            "average_confidence_sl": avg_confidence_sl,
            "average_confidence_entered": avg_confidence_entered,
            "confidence_gap_tp_sl": confidence_gap_tp_sl,
            "average_planned_rr_entered": avg_planned_rr_entered,
            "average_holding_tp_seconds": avg_holding_tp,
            "average_holding_sl_seconds": avg_holding_sl,
            "average_pending_entered_seconds": avg_pending_entered,
            "auto_records": auto_records,
            "manual_records": manual_records,
        }

    async def show_stats(self) -> None:
        await self._refresh_history_if_needed()

        stats = self._calculate_stats(
            self.history_records
        )

        active_pending = sum(
            1
            for trade in self.active_trades.values()
            if trade.status == "PENDING"
        )

        active_filled = sum(
            1
            for trade in self.active_trades.values()
            if trade.status == "FILLED"
        )

        def fmt_confidence(value: Decimal | None) -> str:
            if value is None:
                return "-"
            return decimal_to_str(value)

        confidence_gap = stats["confidence_gap_tp_sl"]
        if confidence_gap is None:
            confidence_conclusion = (
                "Belum cukup data confidence yang terisi di kedua outcome."
            )
        elif confidence_gap > 0:
            confidence_conclusion = (
                "Secara historis, rerata confidence setup TP lebih tinggi "
                f"{decimal_to_str(confidence_gap)} poin daripada setup SL."
            )
        elif confidence_gap < 0:
            confidence_conclusion = (
                "Secara historis, rerata confidence setup TP lebih rendah "
                f"{decimal_to_str(abs(confidence_gap))} poin daripada setup SL."
            )
        else:
            confidence_conclusion = (
                "Secara historis, rerata confidence setup TP dan SL sama."
            )

        if stats["total_filled"]:
            outcome_conclusion = (
                f"Dari {stats['total_filled']} trade yang berakhir TP/SL, "
                f"TP {stats['tp']} ({format_pct(stats['tp_rate']).replace('+', '')}) dan "
                f"SL {stats['sl']} ({format_pct(stats['sl_rate']).replace('+', '')})."
            )
        else:
            outcome_conclusion = (
                "Belum ada trade historis yang berakhir TP/SL."
            )

        confidence_coverage = (
            Decimal(stats["confidence_count_entered"])
            / Decimal(stats["entered_records_count"])
            * Decimal("100")
            if stats["entered_records_count"]
            else Decimal("0")
        )

        rr_text = fmt_confidence(stats["average_planned_rr_entered"])

        await self.reply(
            "📊 STATS\n\n"
            f"Total Setup History: {stats['total_records']}\n"
            f"Total Setup Pernah Entry: {stats['entered_records_count']}\n"
            f"Total Trade Selesai TP/SL: {stats['total_filled']}\n"
            f"Active Pending: {active_pending}\n"
            f"Active Filled: {active_filled}\n\n"
            "OUTCOME\n"
            f"TP: {stats['tp']} ({format_pct(stats['tp_rate']).replace('+', '')} dari trade entry)\n"
            f"  ↳ termasuk TRAIL (SL trailing, profit terkunci): {stats.get('trail_exit', 0)}\n"
            f"SL: {stats['sl']} ({format_pct(stats['sl_rate']).replace('+', '')} dari trade entry)\n"
            f"Win Rate: {format_pct(stats['win_rate']).replace('+', '')}\n"
            f"Expired: {stats['expired']} ({format_pct(stats['expired_rate_all']).replace('+', '')} dari seluruh history)\n"
            f"Deleted: {stats['deleted']} ({format_pct(stats['deleted_rate_all']).replace('+', '')} dari seluruh history)\n"
            f"Manual Close Pending: {stats['manual_close_pending']}\n"
            f"Manual Close Filled @ 0%: {stats['manual_close_filled']}\n\n"
            "CONFIDENCE\n"
            f"Rerata Confidence Semua History: {fmt_confidence(stats['average_confidence_all'])}\n"
            f"Rerata Confidence TP: {fmt_confidence(stats['average_confidence_tp'])}\n"
            f"Rerata Confidence SL: {fmt_confidence(stats['average_confidence_sl'])}\n"
            f"Rerata Confidence Setup Berhasil Entry: {fmt_confidence(stats['average_confidence_entered'])}\n"
            f"Coverage Confidence pada Trade Entry: {decimal_to_str(confidence_coverage)}%\n"
            f"Entry tanpa Confidence: {stats['confidence_missing_entered']}\n\n"
            "ANGKA TAMBAHAN\n"
            f"Net PnL History: {format_pct(stats['gross_pnl_percent'])}\n"
            f"Average Win: {format_pct(stats['average_win_percent'])}\n"
            f"Average Loss: {format_pct(stats['average_loss_percent'])}\n"
            f"Average Planned RR (trade entry): {rr_text}\n"
            f"Average Pending Time: {duration_text(float(stats['average_pending_entered_seconds'])) if stats['average_pending_entered_seconds'] is not None else '-'}\n"
            f"Average Holding TP: {duration_text(float(stats['average_holding_tp_seconds'])) if stats['average_holding_tp_seconds'] is not None else '-'}\n"
            f"Average Holding SL: {duration_text(float(stats['average_holding_sl_seconds'])) if stats['average_holding_sl_seconds'] is not None else '-'}\n"
            f"Total Trail Event: {stats['trail_count']}\n"
            f"History AUTO: {stats['auto_records']} | MANUAL: {stats['manual_records']}\n\n"
            "KESIMPULAN BERBASIS ANGKA\n"
            f"{outcome_conclusion}\n"
            f"{confidence_conclusion}\n"
            f"{stats['expired']} setup berakhir Expired dan {stats['deleted']} setup berakhir Deleted.\n"
            f"Manual Close Pending: {stats['manual_close_pending']}; Manual Close Filled @ 0%: {stats['manual_close_filled']}.\n"
            "Expired, Deleted, dan manual close non-TP/SL tidak dihitung sebagai win/loss."
        )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    def _build_analysis(
        self,
        records: list[dict[str, Any]],
        events: list[dict[str, Any]],
        notes: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Build a complete JSON export and an AI-friendly Markdown report.

        The report is intentionally descriptive and traceable to the stored
        trade/event/note data. Calculated percentages are derived directly
        from those records and do not invent strategy conclusions.
        """
        notes_data = list(notes if notes is not None else self.notes)
        active_records = [
            trade.to_record()
            for trade in self.active_trades.values()
        ]

        stats = self._calculate_stats(
            records
        )

        total_setup_history = stats["total_records"]
        historical_filled = stats["total_filled"]
        expired = stats["expired"]
        deleted = stats["deleted"]

        def percent(numerator: int, denominator: int) -> Decimal:
            if denominator <= 0:
                return Decimal("0")
            return (
                Decimal(numerator)
                / Decimal(denominator)
                * Decimal("100")
            )

        historical_fill_rate = percent(
            historical_filled,
            total_setup_history,
        )
        expiration_rate = percent(
            expired,
            total_setup_history,
        )
        deletion_rate = percent(
            deleted,
            total_setup_history,
        )

        by_pair: dict[str, list[dict[str, Any]]] = {}

        for record in records:
            pair = str(
                record.get("pair") or "UNKNOWN"
            )
            by_pair.setdefault(
                pair,
                [],
            ).append(record)

        pair_rows: list[dict[str, Any]] = []

        for pair, pair_records in sorted(
            by_pair.items()
        ):
            pair_stats = self._calculate_stats(
                pair_records
            )

            pair_rows.append(
                {
                    "pair": pair,
                    "records": pair_stats[
                        "total_records"
                    ],
                    "filled": pair_stats[
                        "total_filled"
                    ],
                    "tp": pair_stats["tp"],
                    "sl": pair_stats["sl"],
                    "expired": pair_stats["expired"],
                    "deleted": pair_stats["deleted"],
                    "win_rate": decimal_to_str(
                        pair_stats["win_rate"]
                    ),
                    "pnl_percent": decimal_to_str(
                        pair_stats[
                            "gross_pnl_percent"
                        ]
                    ),
                }
            )

        by_direction: dict[str, list[dict[str, Any]]] = {}

        for record in records:
            direction = str(
                record.get("direction") or "UNKNOWN"
            )
            by_direction.setdefault(
                direction,
                [],
            ).append(record)

        direction_rows: list[dict[str, Any]] = []

        for direction, direction_records in sorted(
            by_direction.items()
        ):
            direction_stats = self._calculate_stats(
                direction_records
            )

            direction_rows.append(
                {
                    "direction": direction,
                    "records": direction_stats[
                        "total_records"
                    ],
                    "filled": direction_stats[
                        "total_filled"
                    ],
                    "tp": direction_stats["tp"],
                    "sl": direction_stats["sl"],
                    "expired": direction_stats["expired"],
                    "deleted": direction_stats["deleted"],
                    "win_rate": decimal_to_str(
                        direction_stats["win_rate"]
                    ),
                    "pnl_percent": decimal_to_str(
                        direction_stats[
                            "gross_pnl_percent"
                        ]
                    ),
                }
            )

        strategy_groups: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = {}

        for record in records:
            key = (
                str(
                    record.get("strategy_name")
                    or "MANUAL"
                ),
                str(
                    record.get("strategy_version")
                    or "1.0"
                ),
            )

            strategy_groups.setdefault(
                key,
                [],
            ).append(record)

        strategy_rows: list[dict[str, Any]] = []

        for (name, version), group in sorted(
            strategy_groups.items()
        ):
            group_stats = self._calculate_stats(
                group
            )

            strategy_rows.append(
                {
                    "strategy_name": name,
                    "strategy_version": version,
                    "records": group_stats[
                        "total_records"
                    ],
                    "filled": group_stats[
                        "total_filled"
                    ],
                    "tp": group_stats["tp"],
                    "sl": group_stats["sl"],
                    "expired": group_stats["expired"],
                    "deleted": group_stats["deleted"],
                    "win_rate": decimal_to_str(
                        group_stats["win_rate"]
                    ),
                    "pnl_percent": decimal_to_str(
                        group_stats[
                            "gross_pnl_percent"
                        ]
                    ),
                }
            )

        total_trailing_trades = sum(
            1
            for record in records
            if record.get("trail_history")
        )

        event_counts: dict[str, int] = {}
        for event in events:
            event_name = str(
                event.get("event") or "UNKNOWN"
            ).upper()
            event_counts[event_name] = (
                event_counts.get(event_name, 0) + 1
            )

        active_pending = sum(
            1
            for trade in active_records
            if trade.get("status") == "PENDING"
        )
        active_filled = sum(
            1
            for trade in active_records
            if trade.get("status") == "FILLED"
        )

        pending_seconds = []
        holding_seconds = []
        planned_rr_values = []

        for record in records:
            try:
                if record.get("pending_seconds") is not None:
                    pending_seconds.append(
                        Decimal(str(record["pending_seconds"]))
                    )
            except (InvalidOperation, TypeError, ValueError):
                pass

            try:
                if record.get("holding_seconds") is not None:
                    holding_seconds.append(
                        Decimal(str(record["holding_seconds"]))
                    )
            except (InvalidOperation, TypeError, ValueError):
                pass

            try:
                if record.get("planned_rr") is not None:
                    planned_rr_values.append(
                        Decimal(str(record["planned_rr"]))
                    )
            except (InvalidOperation, TypeError, ValueError):
                pass

        avg_pending_seconds = (
            sum(pending_seconds, Decimal("0"))
            / Decimal(len(pending_seconds))
            if pending_seconds
            else None
        )
        avg_holding_seconds = (
            sum(holding_seconds, Decimal("0"))
            / Decimal(len(holding_seconds))
            if holding_seconds
            else None
        )
        avg_planned_rr = (
            sum(planned_rr_values, Decimal("0"))
            / Decimal(len(planned_rr_values))
            if planned_rr_values
            else None
        )

        expiration_reasons: dict[str, int] = {}
        for record in records:
            if record.get("result") != "EXPIRED":
                continue
            reason = str(
                record.get("result_reason") or "-"
            ).strip() or "-"
            expiration_reasons[reason] = (
                expiration_reasons.get(reason, 0) + 1
            )

        # Data-quality / interpretation flags are descriptive only.
        exp_equals_tp = sum(
            1
            for record in records
            if (
                record.get("price_exp") is not None
                and record.get("tp") is not None
                and str(record.get("price_exp"))
                == str(record.get("tp"))
            )
        )

        expired_reason_contains_tp = sum(
            1
            for record in records
            if (
                record.get("result") == "EXPIRED"
                and "tp" in str(
                    record.get("result_reason") or ""
                ).lower()
            )
        )

        full_data = {
            "generated_at": iso_utc(),
            "generated_at_wib": format_wib(
                now_utc()
            ),
            "session_id": self.session_id,
            "source": {
                "market": "Binance USDⓈ-M Futures",
                "execution": "SIMULATION + REAL (lihat real_enabled per trade)",
                "price_trigger": "aggTrade last price",
            },
            "summary": {
                "total_setup_history": total_setup_history,
                "total_filled_trades": historical_filled,
                "tp": stats["tp"],
                "sl": stats["sl"],
                "expired": expired,
                "deleted": deleted,
                "win_rate_percent": decimal_to_str(
                    stats["win_rate"]
                ),
                "historical_fill_rate_percent": decimal_to_str(
                    historical_fill_rate
                ),
                "expiration_rate_percent": decimal_to_str(
                    expiration_rate
                ),
                "deletion_rate_percent": decimal_to_str(
                    deletion_rate
                ),
                "pnl_percent_sum": decimal_to_str(
                    stats["gross_pnl_percent"]
                ),
                "average_win_percent": decimal_to_str(
                    stats["average_win_percent"]
                ),
                "average_loss_percent": decimal_to_str(
                    stats["average_loss_percent"]
                ),
                "average_pending_seconds": (
                    decimal_to_str(avg_pending_seconds)
                    if avg_pending_seconds is not None
                    else None
                ),
                "average_holding_seconds": (
                    decimal_to_str(avg_holding_seconds)
                    if avg_holding_seconds is not None
                    else None
                ),
                "average_planned_rr": (
                    decimal_to_str(avg_planned_rr)
                    if avg_planned_rr is not None
                    else None
                ),
                "trail_event_count": stats[
                    "trail_count"
                ],
                "trailing_trade_count": total_trailing_trades,
            },
            "active_session": {
                "active_trade_count": len(
                    active_records
                ),
                "pending_count": active_pending,
                "filled_count": active_filled,
                "active_trade_ids": list(
                    self.active_trades.keys()
                ),
                "trades": active_records,
            },
            "trades": records,
            "events": events,
            "notes": notes_data,
            "analysis": {
                "by_pair": pair_rows,
                "by_direction": direction_rows,
                "by_strategy": strategy_rows,
                "event_counts": dict(
                    sorted(event_counts.items())
                ),
                "expiration_reasons": dict(
                    sorted(expiration_reasons.items())
                ),
                "data_quality_flags": {
                    "price_exp_equals_tp_count": exp_equals_tp,
                    "expired_reason_mentions_tp_count": expired_reason_contains_tp,
                },
            },
        }

        # --------------------------------------------------------
        # Markdown report: human-readable + AI-readable.
        # --------------------------------------------------------
        report_lines = [
            "# Trading Analysis & AI Reading Report",
            "",
            f"Generated: {format_wib(now_utc())}",
            f"Session: `{self.session_id or '-'}`",
            "",
            "## 1. Scope and Data Source",
            "",
            "- Market: Binance USDⓈ-M Futures",
            "- Execution in this export: Simulation + Real (see real_enabled per trade)",
            "- Price trigger: Binance `aggTrade` last price",
            "- Historical trade records: "
            f"{len(records)}",
            "- Event records: "
            f"{len(events)}",
            "- User notes: "
            f"{len(notes_data)}",
            "- Active setups at export time: "
            f"{len(active_records)}",
            "",
            "## 2. Executive Summary",
            "",
            f"- Total historical setup records: {total_setup_history}",
            f"- Historical filled-and-closed trades: {historical_filled}",
            f"- Historical TP: {stats['tp']}",
            f"- Historical SL: {stats['sl']}",
            f"- Historical Expired: {expired}",
            f"- Historical Deleted: {deleted}",
            f"- Historical fill rate: {decimal_to_str(historical_fill_rate)}%",
            f"- Historical expiration rate: {decimal_to_str(expiration_rate)}%",
            f"- Historical deletion rate: {decimal_to_str(deletion_rate)}%",
            f"- Win Rate: {decimal_to_str(stats['win_rate'])}%",
            f"- Historical PnL sum: {decimal_to_str(stats['gross_pnl_percent'])}%",
            f"- Average win: {decimal_to_str(stats['average_win_percent'])}%",
            f"- Average loss: {decimal_to_str(stats['average_loss_percent'])}%",
            f"- Average planned RR: {decimal_to_str(avg_planned_rr) if avg_planned_rr is not None else '-'}",
            f"- Average pending duration among completed filled trades: {duration_text(avg_pending_seconds) if avg_pending_seconds is not None else '-'}",
            f"- Average holding duration among completed filled trades: {duration_text(avg_holding_seconds) if avg_holding_seconds is not None else '-'}",
            f"- Trailing trade count: {total_trailing_trades}",
            f"- Total trail events: {stats['trail_count']}",
            "",
            "**Win Rate definition:** TP / (TP + SL).",
            "Expired dan Deleted tidak dihitung sebagai win/loss.",
            "Historical metrics above only use records already closed and stored in history.",
            "Active setup data is reported separately and is not counted as historical closed performance.",
            "",
            "## 3. Current Active Session",
            "",
            f"- Active setups: {len(active_records)}",
            f"- Pending: {active_pending}",
            f"- Filled: {active_filled}",
        ]

        if active_records:
            report_lines.extend([
                "",
                "| Pair | Direction | Status | Entry | SL | TP | Price Exp | Trailing |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ])
            for record in active_records:
                report_lines.append(
                    "| "
                    f"{record.get('pair', '-') } | "
                    f"{record.get('direction', '-') } | "
                    f"{record.get('status', '-') } | "
                    f"{record.get('entry', '-') } | "
                    f"{record.get('sl', '-') } | "
                    f"{record.get('tp', '-') } | "
                    f"{record.get('price_exp', '-') } | "
                    f"{'YES' if record.get('trailing') else 'NO'} |"
                )
        else:
            report_lines.append("")
            report_lines.append("Tidak ada active setup saat export.")

        report_lines.extend([
            "",
            "## 4. By Pair",
            "",
            "| Pair | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])

        for row in pair_rows:
            report_lines.append(
                "| "
                f"{row['pair']} | "
                f"{row['records']} | "
                f"{row['filled']} | "
                f"{row['tp']} | "
                f"{row['sl']} | "
                f"{row['expired']} | "
                f"{row['deleted']} | "
                f"{row['win_rate'] or '0'}% | "
                f"{row['pnl_percent'] or '0'} |"
            )

        report_lines.extend([
            "",
            "## 5. By Direction",
            "",
            "| Direction | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])

        for row in direction_rows:
            report_lines.append(
                "| "
                f"{row['direction']} | "
                f"{row['records']} | "
                f"{row['filled']} | "
                f"{row['tp']} | "
                f"{row['sl']} | "
                f"{row['expired']} | "
                f"{row['deleted']} | "
                f"{row['win_rate'] or '0'}% | "
                f"{row['pnl_percent'] or '0'} |"
            )

        report_lines.extend([
            "",
            "## 6. By Strategy",
            "",
            "| Strategy | Version | Setup | Filled | TP | SL | Expired | Win Rate | PnL % |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])

        for row in strategy_rows:
            report_lines.append(
                "| "
                f"{row['strategy_name']} | "
                f"{row['strategy_version']} | "
                f"{row['records']} | "
                f"{row['filled']} | "
                f"{row['tp']} | "
                f"{row['sl']} | "
                f"{row['expired']} | "
                f"{row['win_rate'] or '0'}% | "
                f"{row['pnl_percent'] or '0'} |"
            )

        report_lines.extend([
            "",
            "## 7. Event Distribution",
            "",
            "| Event | Count |",
            "|---|---:|",
        ])
        if event_counts:
            for name, count in sorted(event_counts.items()):
                report_lines.append(
                    f"| {name} | {count} |"
                )
        else:
            report_lines.append("| - | 0 |")

        report_lines.extend([
            "",
            "## 8. Expiration Reasons",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ])
        if expiration_reasons:
            for reason, count in sorted(
                expiration_reasons.items(),
                key=lambda item: (-item[1], item[0]),
            ):
                safe_reason = reason.replace("|", "\\|").replace("\n", " ")
                report_lines.append(
                    f"| {safe_reason} | {count} |"
                )
        else:
            report_lines.append("| - | 0 |")

        report_lines.extend([
            "",
            "## 9. Data Quality / Interpretation Flags",
            "",
            f"- Records where Price Exp exactly equals TP: {exp_equals_tp}",
            f"- Expired records whose reason text mentions TP: {expired_reason_contains_tp}",
            "- These are flags for review, not automatic strategy judgments.",
            "",
            "## 10. Trade-by-Trade Detail",
            "",
        ])

        if records:
            for index, record in enumerate(records, start=1):
                report_lines.extend([
                    f"### Trade {index}: {record.get('trade_id', '-')}",
                    "",
                    f"- Pair: {record.get('pair', '-')}",
                    f"- Direction: {record.get('direction', '-')}",
                    f"- Status: {record.get('status', '-')}",
                    f"- Result: {record.get('result', '-')}",
                    f"- Price Now Reference: {record.get('price_now_reference', '-')}",
                    f"- Entry: {record.get('entry', '-')}",
                    f"- Reason Entry: {record.get('entry_reason', '-')}",
                    f"- Price Exp: {record.get('price_exp', '-')}",
                    f"- Reason Price Exp: {record.get('price_exp_reason', '-')}",
                    f"- SL: {record.get('sl', '-')}",
                    f"- Reason SL: {record.get('sl_reason', '-')}",
                    f"- TP: {record.get('tp', '-')}",
                    f"- Reason TP: {record.get('tp_reason', '-')}",
                    f"- Created: {record.get('created_at', '-')}",
                    f"- Filled: {record.get('filled_at', '-')}",
                    f"- Closed: {record.get('closed_at', '-')}",
                    f"- Fill Price: {record.get('fill_price', '-')}",
                    f"- Exit Price: {record.get('exit_price', '-')}",
                    f"- Result Reason: {record.get('result_reason', '-')}",
                    f"- PnL %: {record.get('pnl_percent', '-')}",
                    f"- Planned RR: {record.get('planned_rr', '-')}",
                    f"- Pending Seconds: {record.get('pending_seconds', '-')}",
                    f"- Holding Seconds: {record.get('holding_seconds', '-')}",
                    f"- Trailing: {'YES' if record.get('trailing') else 'NO'}",
                    f"- Trail History Count: {len(record.get('trail_history') or [])}",
                    f"- Strategy: {record.get('strategy_name', '-')}",
                    f"- Strategy Version: {record.get('strategy_version', '-')}",
                    "",
                ])
        else:
            report_lines.append("Tidak ada trade record historis.")
            report_lines.append("")

        report_lines.extend([
            "## 11. User Notes (/catatan)",
            "",
        ])

        if notes_data:
            for index, note in enumerate(notes_data, start=1):
                note_text = str(
                    note.get("text") or ""
                ).strip()
                created = str(
                    note.get("created_at_wib")
                    or note.get("created_at")
                    or "-"
                )
                report_lines.extend([
                    f"### Note {index}",
                    "",
                    f"- Note ID: {note.get('note_id', '-')}",
                    f"- Time: {created}",
                    f"- Text: {note_text or '-'}",
                    "",
                ])
        else:
            report_lines.append("Belum ada catatan.")
            report_lines.append("")

        report_lines.extend([
            "## 12. AI-Readable Conclusions",
            "",
        ])

        if total_setup_history == 0:
            report_lines.extend([
                "1. Belum ada historical setup yang closed di dataset.",
                "2. Tidak ada dasar statistik untuk menilai outcome historis.",
            ])
        else:
            report_lines.extend([
                f"1. Dari {total_setup_history} historical setup, {expired} berakhir EXPIRED dan {historical_filled} berakhir sebagai filled trade yang kemudian closed.",
                f"2. Expiration rate historis adalah {decimal_to_str(expiration_rate)}%; angka ini menggambarkan frekuensi setup berakhir sebelum menjadi trade yang selesai, bukan win/loss rate.",
                f"3. Historical filled sample berjumlah {historical_filled}; TP={stats['tp']} dan SL={stats['sl']}. Win Rate historis menurut definisi sistem adalah {decimal_to_str(stats['win_rate'])}%.",
                f"4. Historical PnL sum hanya berasal dari record yang memiliki PnL, dengan nilai {decimal_to_str(stats['gross_pnl_percent'])}%.",
                f"5. Active session saat export memiliki {len(active_records)} setup ({active_pending} PENDING, {active_filled} FILLED); setup aktif tidak dimasukkan ke historical closed performance.",
                f"6. Total trailing event yang tercatat adalah {stats['trail_count']} dan jumlah trade historis yang memiliki trailing adalah {total_trailing_trades}.",
            ])

            directions = sorted({
                str(record.get("direction") or "UNKNOWN")
                for record in records
            })
            if directions:
                report_lines.append(
                    "7. Direction historis yang muncul: "
                    + ", ".join(directions)
                    + "."
                )
            else:
                report_lines.append(
                    "7. Tidak ada direction historis yang dapat dianalisis."
                )

            pairs = sorted({
                str(record.get("pair") or "UNKNOWN")
                for record in records
            })
            report_lines.append(
                "8. Pair historis yang muncul: "
                + (", ".join(pairs) if pairs else "-")
                + "."
            )

            if expiration_reasons:
                top_reason, top_count = sorted(
                    expiration_reasons.items(),
                    key=lambda item: (-item[1], item[0]),
                )[0]
                report_lines.append(
                    "9. Reason EXPIRED yang paling sering tercatat adalah "
                    f"'{top_reason}' ({top_count} record)."
                )
            else:
                report_lines.append(
                    "9. Tidak ada reason EXPIRED yang tercatat."
                )

            report_lines.append(
                "10. Dataset ini belum cukup untuk menyimpulkan efektivitas strategi secara umum; kesimpulan yang valid di sini bersifat deskriptif terhadap data yang tersimpan."
            )

        report_lines.extend([
            "",
            "## 13. Guidance for Future AI Analysis",
            "",
            "Gunakan bagian berikut sebagai aturan membaca dataset:",
            "",
            "- `trades` = historical records yang sudah closed.",
            "- `active_session.trades` = setup yang masih aktif saat export dan belum termasuk historical closed performance.",
            "- `events` = kronologi event yang tercatat; satu trade dapat memiliki beberapa event.",
            "- `notes` = catatan manual pengguna; jangan perlakukan isi catatan sebagai fakta pasar tanpa konteks tambahan.",
            "- Expired dan Deleted tidak dihitung sebagai win/loss pada Win Rate.",
            "- PnL hanya relevan pada trade yang memiliki `pnl_percent`.",
            "- Reason Entry / Price Exp / SL / TP adalah alasan yang ditulis pengguna, bukan label otomatis yang diverifikasi oleh bot.",
            "- Jangan menganggap active FILLED sebagai historical closed trade.",
            "- Saat membandingkan periode atau strategi, gunakan sample size dan pisahkan pending/expired dari TP/SL.",
            "",
            "## 14. Raw Data Location",
            "",
            "- Full JSON export: `analysis/full_data.json`",
            "- Full Markdown report: `analysis/analysis.md`",
            "- Historical trades source: `data/trades.json`",
            "- Event source: `data/events.jsonl`",
            "- User notes source: `data/notes.json`",
            "",
            "---",
            "Generated by main.py analysis exporter.",
        ])

        return (
            full_data,
            "\n".join(report_lines)
            + "\n",
        )

    async def analyze(self) -> None:
        # Gunakan history RAM yang sudah dimuat saat startup dan
        # terus diperbarui selama session.
        # Refresh notes so /analyze selalu mengambil data /catatan terbaru dari GitHub.
        await self._load_notes()

        full_data, report = self._build_analysis(
            self.history_records,
            self.history_events,
            self.notes,
        )

        full_data_bytes = json.dumps(
            full_data,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        report_bytes = report.encode("utf-8")

        temp_paths: list[Path] = []

        try:
            commit_json = await self.github.replace_file(
                ANALYSIS_JSON_PATH,
                full_data_bytes,
                "analysis: update full dataset",
            )

            commit_md = await self.github.replace_file(
                ANALYSIS_MD_PATH,
                report_bytes,
                "analysis: update report",
            )

            # Buat salinan sementara untuk dikirim sebagai Telegram Document.
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".json",
                prefix="trading_analysis_",
                delete=False,
            ) as json_file:
                json_file.write(full_data_bytes)
                json_path = Path(json_file.name)
                temp_paths.append(json_path)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".md",
                prefix="trading_analysis_",
                delete=False,
            ) as md_file:
                md_file.write(report_bytes)
                md_path = Path(md_file.name)
                temp_paths.append(md_path)

            stats = self._calculate_stats(
                self.history_records
            )

            await self.reply(
                "📊 ANALYZE SELESAI\n\n"
                f"Total Setup History: {stats['total_records']}\n"
                f"Total Trade Filled: {stats['total_filled']}\n"
                f"Win Rate: {decimal_to_str(stats['win_rate'])}%\n"
                f"PnL History: {decimal_to_str(stats['gross_pnl_percent'])}%\n\n"
                "File hasil analisis dikirim sebagai dokumen Telegram."
            )

            await self.send_document_file(
                json_path,
                "📄 Full dataset — trading history export",
            )
            await self.send_document_file(
                md_path,
                "📄 Analysis report — trading history report",
            )

        except Exception as exc:
            log.exception(
                "ANALYZE gagal."
            )

            await self.reply(
                "❌ /analyze gagal.\n\n"
                f"{exc}"
            )

        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    async def _start_reset(self) -> None:
        if self.flow is not None:
            await self.reply(
                "Masih ada sesi yang sedang berjalan.\n"
                "Gunakan /back terlebih dahulu."
            )
            return

        self.flow = {
            "kind": "RESET",
            "step": "CONFIRM",
        }

        await self.reply(
            "⚠️ RESET PENCATATAN\n\n"
            "Perintah ini akan menghapus dari GitHub:\n"
            "• histori trade\n"
            "• events\n"
            "• journal\n"
            "• hasil /analyze\n"
            "• seluruh /catatan\n\n"
            "Active setup tidak dihapus.\n\n"
            "1. Ya, hapus semua pencatatan\n"
            "2. Batal"
        )

    async def _handle_reset_input(self, text: str) -> None:
        answer = str(text or "").strip()

        if answer == "2":
            self.flow = None
            await self.reply("✅ RESET dibatalkan.")
            return

        if answer != "1":
            await self.reply(
                "Jawab 1 untuk menjalankan reset atau 2 untuk batal."
            )
            return

        self.flow = None

        try:
            await self.reset_github_records()
        except Exception as exc:
            log.exception("RESET pencatatan GitHub gagal.")
            await self.reply(
                "❌ /reset gagal.\n\n"
                f"{exc}"
            )

    # --------------------------------------------------------
    # STATUS / HELP
    # --------------------------------------------------------

    async def show_status(self) -> None:
        pending = sum(
            1
            for trade in self.active_trades.values()
            if trade.status == "PENDING"
        )

        filled = sum(
            1
            for trade in self.active_trades.values()
            if trade.status == "FILLED"
        )

        fresh_prices = sum(
            1
            for snapshot in self.prices.values()
            if snapshot.live
        )

        await self.reply(
            "STATUS\n\n"
            "Main: ONLINE\n"
            f"Mode: {'REAL OFF' if not self.real_mode else 'REAL ON'}\n"
            f"Session: {self.session_id}\n\n"
            f"Binance REST: {'REAL API READY' if self.real.configured else 'PUBLIC ONLY'}\n"
            f"WebSocket: {self.ws.status}\n"
            f"Symbols Subscribed: {len(self.ws.symbols())}\n"
            f"Fresh Price Feed: {fresh_prices}\n\n"
            f"Active Setup: {len(self.active_trades)}\n"
            f"Pending: {pending}\n"
            f"Filled: {filled}\n\n"
            f"History Records: {len(self.history_records)}\n"
            f"Catatan: {len(self.notes)}\n\n"
            f"Scan: {'ON' if self._scan_user_enabled else 'OFF'}"
            f" | Runtime: {'WAITING_H4' if self._scan_user_enabled and self._h4_enabled and not self._h4_window_active else ('PAUSED_BY_MAX' if self._scan_auto_paused else ('RUNNING' if self._scan_task is not None and not self._scan_task.done() else 'OFF'))}\n"
            f"H4 Gate: {'ON' if self._h4_enabled else 'OFF'}\n"
            f"H4 Window: {self._h4_window_text()}\n"
            + (f"H4 Next: {self._h4_next_slot_text()}\n" if self._h4_enabled else "")
            + f"Threshold: {decimal_to_str(self.scan_threshold)} | Max: {self.max_active_trades}\n"
            f"Autostop: {self._autostop_short()}\n"
            f"Margin: {decimal_to_str(self.margin_usdt)} USDT | Leverage: {self.leverage}x | Target Notional: {decimal_to_str(self.margin_usdt * Decimal(self.leverage))} USDT\n"
        )

    async def show_help(self) -> None:
        await self.reply(
            "COMMAND MAIN.PY\n\n"
            "/auto - cari setup otomatis dari strategy.py + confidence\n"
            "/add - membuat setup baru\n"
            "/back - kembali satu langkah / batalkan sesi\n"
            "/trade - melihat setup aktif + harga live\n"
            "/setup - simpan semua setup aktif /trade ke GitHub\n"
            "/open - buka kembali setup yang tersimpan di GitHub\n"
            "/trail - mengubah SL setup\n"
            "/del - menghapus setup dan mencatat DELETED di history\n"
            "/wrong - menghapus setup yang benar-benar salah TANPA pencatatan history\n"
            "/remove - menghapus data trade yang sudah masuk history\n"
            "/filled - ubah setup PENDING menjadi FILLED secara manual\n"
            "/close PAIR|all - tutup paksa setup dan catat hasil\n"
            "/real on|off - aktif/nonaktifkan real execution\n"
            "/margin [USDT] - atur margin target\n"
            "/leverage [angka] - atur leverage target\n"
            "/scan on|off - scanner otomatis\n"
            "/H4 on|off - gate scanner berdasarkan close H4 03/07/11/15/19/23 WIB\n"
            "/threshold [angka] - ambang confidence scanner\n"
            "/max [angka] - batas total PENDING + FILLED\n"
            "/autostop [persen|off|reset] - matikan scan saat equity turun dari puncak (REAL ON)\n"
            "/api - statistik request & bobot Binance (diagnosa rate limit)\n"
            "/banned [PAIR] [jam] [alasan] - ban pair / lihat daftar ban\n"
            "/unban PAIR|all - hapus ban\n"
            "/stats - statistik histori\n"
            "/catatan [teks] - tambah catatan / tanpa teks = lihat catatan\n"
            "/reset - hapus seluruh pencatatan GitHub\n"
            "/analyze - generate full dataset + report, lalu kirim file Telegram\n"
            "/status - status engine dan WebSocket\n"
            "/help - menu command\n\n"
            "Launcher:\n"
            "/try /end /ganti /healthz"
        )

    # --------------------------------------------------------
    # UPDATE ROUTER
    # --------------------------------------------------------

    async def handle_update(
        self,
        update: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        message = update.get(
            "message"
        ) or {}

        from_user = int(
            (message.get("from") or {}).get(
                "id"
            )
            or 0
        )

        if from_user and from_user != ALLOWED_USER_ID:
            return

        text = str(
            message.get("text") or ""
        ).strip()

        if not text:
            return

        command = (
            text.split(maxsplit=1)[0]
            .split("@", 1)[0]
            .lower()
        )

        # /back selalu memiliki prioritas di atas flow.
        if command == "/back":
            await self._handle_back()
            return

        # Launcher commands tidak ditangani ulang di main.
        if command in {
            "/try",
            "/end",
            "/ganti",
            "/healthz",
        }:
            return

        # Command baru saat flow aktif.
        if command.startswith("/"):
            if self.flow:
                if command in {
                    "/auto",
                    "/add",
                    "/trail",
                    "/del",
                    "/wrong",
                    "/remove",
                    "/filled",
                    "/close",
                    "/scan",
                    "/h4",
                    "/threshold",
                    "/real",
                    "/margin",
                    "/leverage",
                    "/max",
                    "/autostop",
                    "/banned",
                    "/unban",
                    "/trade",
                    "/setup",
                    "/open",
                    "/stats",
                    "/analyze",
                    "/status",
                    "/catatan",
                    "/reset",
                }:
                    await self.reply(
                        "Masih ada sesi yang sedang berjalan.\n"
                        "Gunakan /back terlebih dahulu."
                    )
                    return

            if command == "/auto":
                await self._start_auto()
                return

            if command == "/add":
                await self._start_add()
                return

            if command == "/trade":
                await self.show_trades()
                return

            if command == "/setup":
                try:
                    await self.save_setups()
                except Exception as exc:
                    log.exception("SETUP gagal disimpan ke GitHub.")
                    await self.reply(
                        "❌ /setup gagal.\n\n"
                        f"{exc}"
                    )
                return

            if command == "/open":
                try:
                    await self.open_setups()
                except Exception as exc:
                    log.exception("OPEN setup dari GitHub gagal.")
                    await self.reply(
                        "❌ /open gagal.\n\n"
                        f"{exc}"
                    )
                return

            if command == "/trail":
                await self._start_trail()
                return

            if command == "/del":
                await self._start_del()
                return

            if command == "/wrong":
                await self._start_wrong()
                return

            if command == "/remove":
                await self._start_remove()
                return

            if command == "/filled":
                await self._start_filled()
                return

            if command == "/close":
                try:
                    await self._handle_close_command(text)
                except Exception as exc:
                    log.exception("CLOSE command gagal.")
                    await self.reply(f"❌ /close gagal.\n\n{exc}")
                return

            if command == "/real":
                try:
                    parts = text.split(maxsplit=1)
                    if len(parts) == 1:
                        await self.reply(f"🔐 REAL MODE: {'ON' if self.real_mode else 'OFF'}")
                    elif parts[1].strip().lower() == "on":
                        await self._set_real_on()
                    elif parts[1].strip().lower() == "off":
                        await self._set_real_off()
                    else:
                        raise ValueError("Gunakan /real on atau /real off.")
                except Exception as exc:
                    log.exception("REAL command gagal.")
                    await self.reply(f"❌ /real gagal.\n\n{exc}")
                return

            if command == "/margin":
                try:
                    await self._set_margin_command(text[len(text.split(maxsplit=1)[0]):].strip())
                except Exception as exc:
                    log.exception("MARGIN command gagal.")
                    await self.reply(f"❌ /margin gagal.\n\n{exc}")
                return

            if command == "/leverage":
                try:
                    await self._set_leverage_command(text[len(text.split(maxsplit=1)[0]):].strip())
                except Exception as exc:
                    log.exception("LEVERAGE command gagal.")
                    await self.reply(f"❌ /leverage gagal.\n\n{exc}")
                return

            if command == "/scan":
                try:
                    await self._handle_scan_command(text)
                except Exception as exc:
                    log.exception("SCAN command gagal.")
                    await self.reply(f"❌ /scan gagal.\n\n{exc}")
                return

            if command == "/h4":
                try:
                    await self._handle_h4_command(text)
                except Exception as exc:
                    log.exception("H4 command gagal.")
                    await self.reply(f"❌ /H4 gagal.\n\n{exc}")
                return

            if command == "/threshold":
                try:
                    await self._handle_threshold_command(text)
                except Exception as exc:
                    log.exception("THRESHOLD command gagal.")
                    await self.reply(f"❌ /threshold gagal.\n\n{exc}")
                return

            if command == "/max":
                try:
                    await self._handle_max_command(text)
                except Exception as exc:
                    log.exception("MAX command gagal.")
                    await self.reply(f"❌ /max gagal.\n\n{exc}")
                return

            if command == "/api":
                await self.reply(self._api_report())
                return

            if command == "/autostop":
                try:
                    await self._handle_autostop_command(text)
                except Exception as exc:
                    log.exception("AUTOSTOP command gagal.")
                    await self.reply(f"❌ /autostop gagal.\n\n{exc}")
                return

            if command == "/banned":
                try:
                    await self._handle_banned_command(text)
                except Exception as exc:
                    log.exception("BANNED command gagal.")
                    await self.reply(f"❌ /banned gagal.\n\n{exc}")
                return

            if command == "/unban":
                try:
                    await self._handle_unban_command(text)
                except Exception as exc:
                    log.exception("UNBAN command gagal.")
                    await self.reply(f"❌ /unban gagal.\n\n{exc}")
                return

            if command == "/stats":
                await self.show_stats()
                return

            if command == "/analyze":
                await self.analyze()
                return

            if command == "/catatan":
                argument = text[len(text.split(maxsplit=1)[0]):].strip()
                if argument:
                    await self.add_note(argument)
                else:
                    await self.show_notes()
                return

            if command == "/reset":
                await self._start_reset()
                return

            if command == "/status":
                await self.show_status()
                return

            if command in {
                "/help",
                "/start",
            }:
                await self.show_help()
                return

            await self.reply(
                "Command tidak dikenal.\n"
                "Gunakan /help."
            )
            return

        # Plain text masuk ke active flow.
        if self.flow:
            if self.flow["kind"] == "AUTO":
                await self._handle_auto_input(
                    text
                )
                return

            if self.flow["kind"] == "ADD":
                await self._handle_add_input(
                    text
                )
                return

            if self.flow["kind"] == "TRAIL":
                await self._handle_trail_input(
                    text
                )
                return

            if self.flow["kind"] == "DEL":
                await self._handle_del_input(
                    text
                )
                return

            if self.flow["kind"] == "WRONG":
                await self._handle_wrong_input(
                    text
                )
                return

            if self.flow["kind"] == "REMOVE":
                await self._handle_remove_input(
                    text
                )
                return

            if self.flow["kind"] == "FILLED":
                await self._handle_filled_input(
                    text
                )
                return

            if self.flow["kind"] == "RESET":
                await self._handle_reset_input(
                    text
                )
                return

        await self.reply(
            "Tidak ada sesi input aktif.\n"
            "Gunakan /help atau /add."
        )


# ============================================================
# MODULE-LEVEL CONTRACT FOR try.py
# ============================================================

_ENGINE: TradingEngine | None = None


async def on_start(
    context: dict[str, Any],
):
    global _ENGINE

    if _ENGINE is not None:
        await _ENGINE.stop()
        _ENGINE = None

    engine = TradingEngine(
        context
    )

    _ENGINE = engine

    try:
        await engine.start()
    except Exception:
        try:
            await engine.stop()
        except Exception:
            log.exception(
                "Cleanup gagal setelah startup main.py gagal."
            )

        _ENGINE = None
        raise

    return True


async def handle_update(
    update: dict[str, Any],
    context: dict[str, Any],
):
    if _ENGINE is None:
        return

    await _ENGINE.handle_update(
        update,
        context,
    )


async def on_stop(
    context: dict[str, Any],
):
    global _ENGINE

    engine = _ENGINE

    if engine is None:
        return

    try:
        await engine.stop()
    finally:
        _ENGINE = None


# ============================================================
# DIRECT EXECUTION IS NOT SUPPORTED
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        "main.py dijalankan melalui try.py menggunakan "
        "on_start(), handle_update(), dan on_stop()."
    )
