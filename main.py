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
- Binance USDⓈ-M Futures = market data utama.
- Real execution bersifat opsional melalui /real dan signed Binance Futures API.
- Saat /real OFF, sistem tetap murni simulasi seperti sebelumnya.
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
"""

import asyncio
import base64
import copy
import hashlib
import hmac
import html  # kept out of user output; used only for safe GitHub text if needed
import json
import logging
import os
import re
import time
import traceback
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN
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

load_dotenv(BASE_DIR / "trades.env")
load_dotenv(BASE_DIR / ".env")

# Acuan nama key mengikuti Trades.env yang diberikan user.
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()
MAIN_FILE = (os.getenv("MAIN_FILE") or "main.py").strip()

# Kredensial real Binance hanya dipakai ketika /real ON.
# Jangan pernah menulis key/secret ini ke GitHub, /setup, log, atau Telegram.
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
WS_RECONNECT_MIN = 2
WS_RECONNECT_MAX = 60
PRICE_STALE_SECONDS = 15
MAX_ACTIVE_TRADES = 100
HISTORY_REFRESH_SECONDS = 30

DEFAULT_MARGIN_USDT = Decimal("0.5")
DEFAULT_LEVERAGE = 10
REAL_NOTIONAL_MIN_RATIO = Decimal("0.90")
REAL_NOTIONAL_MAX_RATIO = Decimal("1.10")
BINANCE_RECV_WINDOW = 5000
BINANCE_RATE_LIMIT_FALLBACK_SECONDS = 60
BINANCE_RATE_LIMIT_SAFETY_SECONDS = 60
REAL_RECONCILE_INTERVAL_SECONDS = 1.0

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
    """Forward WARNING/ERROR logs from backend tasks to Telegram."""

    def __init__(self, engine: "TradingEngine") -> None:
        super().__init__(level=logging.WARNING)
        self.engine = engine
        self.loop: asyncio.AbstractEventLoop | None = None
        self._last_signature = ""
        self._last_sent = 0.0

    def attach(self) -> None:
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.loop is None or self.engine is None:
            return

        # Jangan membuat loop error baru ketika Telegram sender sendiri gagal.
        if "Gagal mengirim Telegram" in record.getMessage():
            return

        message = record.getMessage()
        signature = f"{record.name}|{record.levelno}|{message}"
        now = time.monotonic()

        # Deduplicate spam identik selama 5 detik.
        if signature == self._last_signature and now - self._last_sent < 5:
            return

        self._last_signature = signature
        self._last_sent = now

        try:
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self.engine._send_log_error_to_telegram(record)
                )
            )
        except Exception:
            # Error handler tidak boleh menjatuhkan aplikasi.
            pass


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
    text = str(value or "").strip().replace(",", ".")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", text):
        raise ValueError("Angka desimal tidak valid.")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Angka desimal tidak valid.") from exc
    if not number.is_finite():
        raise ValueError("Angka desimal tidak finite.")
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


def quantized_price(value: Decimal, tick_size: Decimal) -> Decimal:
    if tick_size <= 0:
        return value
    steps = (value / tick_size).to_integral_value(rounding=ROUND_DOWN)
    return steps * tick_size


def validate_tick(value: Decimal, tick_size: Decimal) -> bool:
    if tick_size <= 0:
        return True
    remainder = value % tick_size
    return remainder == 0


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
    quantity_precision: int = 0


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

    trail_history: list[dict[str, Any]] = field(default_factory=list)

    strategy_name: str = "MANUAL"
    strategy_version: str = "1.0"

    # Real execution metadata. None/False means no real order is attached.
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
            "real_error": self.real_error,
            "real_exit_check": self.real_exit_check,
        }


# ============================================================
# BINANCE REST MARKET DATA
# ============================================================

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

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Binance REST {path}: HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

            return response.json()

        return await asyncio.to_thread(request)

    async def get_exchange_info(self) -> dict[str, SymbolMeta]:
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
            step_size = Decimal("0")
            min_notional = Decimal("0")
            max_notional = Decimal("0")

            for filt in raw.get("filters", []):
                filter_type = str(filt.get("filterType") or "")

                if filter_type == "PRICE_FILTER":
                    tick_size = Decimal(
                        str(filt.get("tickSize") or "0")
                    )
                    min_price = Decimal(
                        str(filt.get("minPrice") or "0")
                    )
                    max_price = Decimal(
                        str(filt.get("maxPrice") or "0")
                    )

                elif filter_type in {"LOT_SIZE", "MARKET_LOT_SIZE"}:
                    # LOT_SIZE adalah acuan utama untuk LIMIT entry.
                    if filter_type == "LOT_SIZE" or step_size == 0:
                        min_qty = Decimal(
                            str(filt.get("minQty") or "0")
                        )
                        max_qty = Decimal(
                            str(filt.get("maxQty") or "0")
                        )
                        step_size = Decimal(
                            str(filt.get("stepSize") or "0")
                        )

                elif filter_type == "MIN_NOTIONAL":
                    min_notional = Decimal(
                        str(filt.get("notional") or filt.get("minNotional") or "0")
                    )

                elif filter_type == "NOTIONAL":
                    min_notional = Decimal(
                        str(filt.get("minNotional") or min_notional or "0")
                    )
                    max_notional = Decimal(
                        str(filt.get("maxNotional") or "0")
                    )

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
                quantity_precision=int(raw.get("quantityPrecision") or 0),
            )

        return result

    async def get_price(self, symbol: str) -> Decimal:
        payload = await self._get(
            "/fapi/v2/ticker/price",
            params={"symbol": symbol},
        )

        return parse_decimal(str(payload["price"]))


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
        self._request_lock = asyncio.Lock()
        self._dual_side_position: bool | None = None
        self.last_balance: dict[str, Any] | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.monotonic())

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

        remaining = self.cooldown_remaining
        if remaining > 0:
            raise BinanceRateLimitError(
                f"REST Binance sedang ditahan oleh rate-limit cooldown ({remaining:.0f}s tersisa).",
                status_code=429,
                code=None,
                endpoint=path,
                server_cooldown_seconds=remaining,
                bot_cooldown_seconds=remaining,
                retry_after_known=False,
            )

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

        if response.status_code in {418, 429}:
            retry_after = self._parse_retry_after(response)
            server_seconds = retry_after if retry_after is not None else BINANCE_RATE_LIMIT_FALLBACK_SECONDS
            bot_seconds = server_seconds + BINANCE_RATE_LIMIT_SAFETY_SECONDS
            self._cooldown_until = max(
                self._cooldown_until,
                time.monotonic() + bot_seconds,
            )
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
            response.raise_for_status()
            body = response.json()
            return int(body["serverTime"])

        return await asyncio.to_thread(request)

    async def get_futures_balance(self) -> list[dict[str, Any]]:
        """Single balance test used when /real is turned ON."""
        payload = await self._request("GET", "/fapi/v3/balance")
        if not isinstance(payload, list):
            raise BinanceAPIError(
                "Respons /fapi/v3/balance tidak berbentuk list.",
                endpoint="/fapi/v3/balance",
            )
        return [item for item in payload if isinstance(item, dict)]

    async def get_account_info(self) -> dict[str, Any]:
        payload = await self._request("GET", "/fapi/v2/account")
        if not isinstance(payload, dict):
            raise BinanceAPIError(
                "Respons /fapi/v2/account tidak berbentuk object.",
                endpoint="/fapi/v2/account",
            )
        self.last_balance = payload
        positions = payload.get("positions") or []
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
            "quantity": decimal_to_str(quantity),
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
            "quantity": decimal_to_str(quantity),
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
            if self.last_message_at is None:
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

            if ws is None or not self.connected:
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

                log.warning(
                    "Gagal kirim %s ke Binance WebSocket "
                    "(koneksi sudah tertutup): %s",
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
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
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
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=PRICE_STALE_SECONDS + 30,
                        )

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
        self.github = GitHubStore()

        self.symbols: dict[str, SymbolMeta] = {}
        self.prices: dict[str, PriceSnapshot] = {}

        self.active_trades: dict[str, Trade] = {}

        self.history_records: list[dict[str, Any]] = []
        self.history_events: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []

        # Runtime real-trading controls. REAL selalu OFF saat engine baru dibuat.
        self.real_mode = False
        self.margin_usdt = DEFAULT_MARGIN_USDT
        self.leverage = DEFAULT_LEVERAGE
        self.real = BinanceRealClient()
        self._real_reconcile_guard: dict[str, float] = {}

        # Market-event ordering is also guarded at engine level. This is a
        # second safety layer so an out-of-order callback can never trigger
        # a state transition. Key = (Binance event time, aggregate trade ID).
        self._last_market_event_key: dict[str, tuple[int, int]] = {}
        self._last_live_price: dict[str, Decimal] = {}


        self.flow: dict[str, Any] | None = None

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

            text = (
                f"{'🚨 ERROR BACKEND' if record.levelno >= logging.ERROR else '⚠️ WARNING BACKEND'}\n\n"
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
            self._telegram_error_handler.attach()
            log.addHandler(self._telegram_error_handler)
            self._telegram_error_handler_added = True

        await self._load_history()
        await self._load_notes()

        self.symbols = await self.rest.get_exchange_info()

        await self.ws.start()

        self.last_history_refresh = now_utc()

        await self.reply(
            "🟢 MAIN.PY ONLINE\n\n"
            "Execution: REAL OFF (SIMULATION)\n"
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
            "version": 2,
            "saved_at": iso_utc(),
            "saved_at_wib": format_wib(now_utc()),
            "count": len(records),
            "defaults": {
                "margin_usdt": decimal_to_str(self.margin_usdt),
                "leverage": self.leverage,
            },
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
            "Setup dapat dipulihkan dengan /open setelah main.py diganti.\n"
            "Credential Binance TIDAK pernah disimpan ke file setup."
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

        fill_price = (
            parse_decimal(str(record["fill_price"]))
            if record.get("fill_price") not in (None, "")
            else None
        )
        pnl_percent = (
            parse_signed_decimal(str(record["pnl_percent"]))
            if record.get("pnl_percent") not in (None, "")
            else None
        )

        def optional_decimal(key: str) -> Decimal | None:
            value = record.get(key)
            if value in (None, ""):
                return None
            return parse_decimal(str(value))

        def optional_int(key: str) -> int | None:
            value = record.get(key)
            if value in (None, ""):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

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
            price_now_reference=parse_decimal(str(record["price_now_reference"])),
            entry=parse_decimal(str(record["entry"])),
            entry_reason=str(record.get("entry_reason") or ""),
            price_exp=parse_decimal(str(record["price_exp"])),
            price_exp_reason=str(record.get("price_exp_reason") or ""),
            sl=parse_decimal(str(record["sl"])),
            sl_reason=str(record.get("sl_reason") or ""),
            tp=parse_decimal(str(record["tp"])),
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
            trail_history=copy.deepcopy(trail_history),
            strategy_name=str(record.get("strategy_name") or "MANUAL"),
            strategy_version=str(record.get("strategy_version") or "1.0"),
            margin_usdt=optional_decimal("margin_usdt") or DEFAULT_MARGIN_USDT,
            leverage=int(record.get("leverage") or DEFAULT_LEVERAGE),
            quantity=optional_decimal("quantity"),
            target_notional=optional_decimal("target_notional"),
            actual_notional=optional_decimal("actual_notional"),
            real_enabled=bool(record.get("real_enabled", False)),
            real_state=(
                str(record.get("real_state") or "").strip()
                or (
                    "REAL_FILLED"
                    if status == "FILLED" and record.get("real_enabled")
                    else "REAL_PENDING"
                    if status == "PENDING" and record.get("real_enabled")
                    else "SIMULATION"
                )
            ),
            position_side=(str(record.get("position_side")) if record.get("position_side") not in (None, "") else None),
            entry_order_id=optional_int("entry_order_id"),
            entry_client_order_id=(str(record.get("entry_client_order_id")) if record.get("entry_client_order_id") not in (None, "") else None),
            tp_algo_id=optional_int("tp_algo_id"),
            sl_algo_id=optional_int("sl_algo_id"),
            tp_client_algo_id=(str(record.get("tp_client_algo_id")) if record.get("tp_client_algo_id") not in (None, "") else None),
            sl_client_algo_id=(str(record.get("sl_client_algo_id")) if record.get("sl_client_algo_id") not in (None, "") else None),
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
            defaults = payload.get("defaults")
            if isinstance(defaults, dict):
                saved_margin = defaults.get("margin_usdt")
                saved_leverage = defaults.get("leverage")

                if saved_margin not in (None, ""):
                    try:
                        restored_margin = parse_decimal(str(saved_margin))
                        if restored_margin > 0:
                            self.margin_usdt = restored_margin
                    except ValueError:
                        log.warning(
                            "Default margin pada %s tidak valid; margin runtime dipertahankan.",
                            SETUPS_PATH,
                        )

                if saved_leverage not in (None, ""):
                    try:
                        restored_leverage = int(saved_leverage)
                    except (TypeError, ValueError):
                        restored_leverage = self.leverage
                    if 1 <= restored_leverage <= 125:
                        self.leverage = restored_leverage
                    else:
                        log.warning(
                            "Default leverage pada %s di luar 1-125; leverage runtime dipertahankan.",
                            SETUPS_PATH,
                        )

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

            if len(self.active_trades) >= MAX_ACTIVE_TRADES:
                skipped += 1
                skipped_details.append(
                    f"{trade.trade_id}: batas active trade tercapai"
                )
                continue

            self.active_trades[trade.trade_id] = trade
            loaded += 1
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

        if self.real_mode:
            for trade in list(self.active_trades.values()):
                if not trade.real_enabled:
                    continue
                try:
                    await self._reconcile_real_trade(trade)
                except BinanceRateLimitError as exc:
                    self._notify_rate_limit(exc, f"RECONCILE /open {trade.pair}")
                except Exception:
                    log.exception(
                        "Rekonsiliasi REAL %s gagal saat /open.",
                        trade.trade_id,
                    )
        else:
            for trade in self.active_trades.values():
                if trade.real_enabled:
                    trade.real_state = "REAL_DETACHED"

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
            "### Real Execution\n\n"
            f"- Real Enabled: {record.get('real_enabled', False)}\n"
            f"- Margin: {record.get('margin_usdt', '-')} USDT\n"
            f"- Leverage: {record.get('leverage', '-')}x\n"
            f"- Quantity: {record.get('quantity', '-')}\n"
            f"- Target Notional: {record.get('target_notional', '-')} USDT\n"
            f"- Actual Notional: {record.get('actual_notional', '-')} USDT\n"
            f"- Entry Order ID: {record.get('entry_order_id', '-')}\n"
            f"- TP Algo ID: {record.get('tp_algo_id', '-')}\n"
            f"- SL Algo ID: {record.get('sl_algo_id', '-')}\n\n"
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
    # REAL MODE / MONEY MANAGEMENT
    # --------------------------------------------------------

    async def _start_real_on(self) -> None:
        if self.real_mode:
            await self.reply(
                "🟢 REAL MODE SUDAH ON\n\n"
                f"Margin default: {decimal_to_str(self.margin_usdt)} USDT\n"
                f"Leverage default: {self.leverage}x"
            )
            return

        if not self.real.configured:
            await self.reply(
                "❌ REAL MODE GAGAL DINYALAKAN\n\n"
                "BINANCE_API_KEY / BINANCE_API_SECRET belum tersedia."
            )
            return

        try:
            # Exactly one signed balance request is used as the /real ON access test.
            balances = await self.real.get_futures_balance()
            usdt = next(
                (
                    item
                    for item in balances
                    if str(item.get("asset") or "").upper() == "USDT"
                ),
                None,
            )

            if usdt is None:
                raise BinanceAPIError(
                    "Endpoint balance berhasil merespons, tetapi asset USDT tidak ditemukan.",
                    endpoint="/fapi/v3/balance",
                )

            balance = parse_decimal(str(usdt.get("balance") or "0"))
            available = parse_decimal(
                str(usdt.get("availableBalance") or usdt.get("balance") or "0")
            )

            self.real_mode = True
            dual_text = (
                "HEDGE" if self.real._dual_side_position
                else "ONE-WAY" if self.real._dual_side_position is False
                else "AUTO (akan dibaca saat eksekusi)"
            )

            await self.reply(
                "🟢 REAL MODE ON\n\n"
                "Binance API: CONNECTED\n"
                "Account: VERIFIED\n"
                f"USDT Balance: {decimal_to_str(balance)} USDT\n"
                f"Available: {decimal_to_str(available)} USDT\n"
                f"Position Mode: {dual_text}\n\n"
                f"Margin default: {decimal_to_str(self.margin_usdt)} USDT\n"
                f"Leverage default: {self.leverage}x\n"
                f"Target Notional: {decimal_to_str(self.margin_usdt * Decimal(self.leverage))} USDT\n\n"
                "Execution: REAL\n"
                "Market Feed: WEBSOCKET"
            )

            # Existing saved real setups are reconciled only here. This does
            # not create a new entry order when the saved metadata is missing.
            for trade in list(self.active_trades.values()):
                if not trade.real_enabled:
                    continue
                try:
                    await self._reconcile_real_trade(trade)
                except BinanceRateLimitError as exc:
                    self._notify_rate_limit(exc, f"RECONCILE /real ON {trade.pair}")
                except Exception:
                    log.exception(
                        "Rekonsiliasi REAL gagal saat /real ON untuk %s.",
                        trade.trade_id,
                    )

        except BinanceRateLimitError as exc:
            self.real_mode = False
            self._notify_rate_limit(exc, "REAL ON / balance test")
            await self.reply(
                "❌ REAL MODE tetap OFF karena Binance sedang rate-limited.\n"
                f"Cooldown safety: {exc.bot_cooldown_seconds:.0f} detik."
            )
        except Exception as exc:
            self.real_mode = False
            log.exception("REAL MODE gagal dinyalakan: %s", exc)
            await self.reply(
                "❌ REAL MODE tetap OFF.\n\n"
                f"Binance API test gagal: {exc}"
            )

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
        await self.reply(
            "🔴 REAL MODE OFF\n\n"
            "Tidak ada order real baru yang akan dibuat.\n"
            "WebSocket tetap berjalan.\n"
            "Setup simulasi tetap berjalan.\n"
            "Tidak ada order/posisi Binance yang disentuh otomatis."
        )

    def _notify_rate_limit(self, exc: BinanceRateLimitError, action: str) -> None:
        server = max(0.0, exc.server_cooldown_seconds)
        bot = max(0.0, exc.bot_cooldown_seconds)
        retry_known = getattr(exc, "retry_after_known", False)
        if retry_known:
            timing = f"Retry-After Binance={server:.0f}s; bot pause={bot:.0f}s"
        elif exc.code is None and exc.status_code == 429:
            timing = (
                "Cooldown internal bot sedang berjalan; "
                f"sisa jeda={bot:.0f}s"
            )
        else:
            timing = (
                "Retry-After tidak diberikan; "
                f"fallback Binance={server:.0f}s; bot pause={bot:.0f}s"
            )
        log.warning(
            "Binance API LIMIT | action=%s | endpoint=%s | HTTP=%s | code=%s | %s | message=%s",
            action,
            exc.endpoint,
            exc.status_code,
            exc.code,
            timing,
            str(exc),
        )

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
    ) -> tuple[Decimal, Decimal, Decimal]:
        if entry <= 0:
            raise MarginInfluenceError(f"{meta.symbol}: Entry harus lebih besar dari 0.")
        if margin <= 0:
            raise MarginInfluenceError("Margin harus lebih besar dari 0.")
        if leverage < 1:
            raise MarginInfluenceError("Leverage harus minimal 1x.")

        target = margin * Decimal(leverage)
        lower_notional = target * REAL_NOTIONAL_MIN_RATIO
        upper_notional = target * REAL_NOTIONAL_MAX_RATIO

        raw_quantity = target / entry

        step = meta.step_size
        if step <= 0:
            precision = max(0, meta.quantity_precision)
            step = Decimal(1).scaleb(-precision)
            if step <= 0:
                step = Decimal("0.00000001")

        # Convert all exchange constraints into one legal quantity window.
        lower_qty = lower_notional / entry
        upper_qty = upper_notional / entry

        if meta.min_qty > 0:
            lower_qty = max(lower_qty, meta.min_qty)
        if meta.max_qty > 0:
            upper_qty = min(upper_qty, meta.max_qty)
        if meta.min_notional > 0:
            lower_qty = max(lower_qty, meta.min_notional / entry)
        if meta.max_notional > 0:
            upper_qty = min(upper_qty, meta.max_notional / entry)

        lower_step_qty = self._ceil_to_step(lower_qty, step)
        upper_step_qty = self._floor_to_step(upper_qty, step)

        if lower_step_qty <= 0:
            lower_step_qty = step

        if lower_step_qty > upper_step_qty:
            nearest_candidates = [
                lower_step_qty,
                upper_step_qty,
                self._floor_to_step(raw_quantity, step),
                self._ceil_to_step(raw_quantity, step),
            ]
            nearest_candidates = [q for q in nearest_candidates if q > 0]
            nearest = min(
                nearest_candidates,
                key=lambda q: abs((q * entry) - target),
                default=Decimal("0"),
            )
            nearest_notional = nearest * entry
            raise MarginInfluenceError(
                f"{meta.symbol}: tidak ada quantity valid dalam rentang "
                f"±10% dari target {decimal_to_str(target)} USDT. "
                f"Quantity terdekat menghasilkan {decimal_to_str(nearest_notional)} USDT."
            )

        # Normally the nearest step to the raw quantity is enough. Also check
        # the exact legal window boundaries so min/max notional constraints do
        # not create false negatives.
        floor_raw = self._floor_to_step(raw_quantity, step)
        ceil_raw = self._ceil_to_step(raw_quantity, step)
        candidates = {
            lower_step_qty,
            upper_step_qty,
            floor_raw,
            ceil_raw,
        }
        candidates = {
            q for q in candidates
            if lower_step_qty <= q <= upper_step_qty and q > 0
        }

        valid: list[tuple[Decimal, Decimal]] = []
        for qty in sorted(candidates):
            if meta.min_qty > 0 and qty < meta.min_qty:
                continue
            if meta.max_qty > 0 and qty > meta.max_qty:
                continue

            notional = qty * entry
            if notional < lower_notional or notional > upper_notional:
                continue
            if meta.min_notional > 0 and notional < meta.min_notional:
                continue
            if meta.max_notional > 0 and notional > meta.max_notional:
                continue

            valid.append((qty, notional))

        # If raw floor/ceil were not in the candidate set because of a very
        # restrictive boundary, use the first and last legal step as fallback.
        if not valid:
            for qty in (lower_step_qty, upper_step_qty):
                if lower_step_qty <= qty <= upper_step_qty and qty > 0:
                    notional = qty * entry
                    if (
                        lower_notional <= notional <= upper_notional
                        and (meta.min_qty <= 0 or qty >= meta.min_qty)
                        and (meta.max_qty <= 0 or qty <= meta.max_qty)
                        and (meta.min_notional <= 0 or notional >= meta.min_notional)
                        and (meta.max_notional <= 0 or notional <= meta.max_notional)
                    ):
                        valid.append((qty, notional))

        if not valid:
            nearest = min(
                [lower_step_qty, upper_step_qty],
                key=lambda q: abs((q * entry) - target),
            )
            nearest_notional = nearest * entry
            raise MarginInfluenceError(
                f"{meta.symbol}: quantity valid tidak ditemukan dalam ±10% target "
                f"{decimal_to_str(target)} USDT. "
                f"Nearest notional: {decimal_to_str(nearest_notional)} USDT."
            )

        quantity, actual_notional = min(
            valid,
            key=lambda item: abs(item[1] - target),
        )
        return quantity, target, actual_notional

    async def _ensure_real_entry(self, trade: Trade) -> None:
        if not self.real_mode:
            return
        if trade.entry_order_id or trade.entry_client_order_id:
            return

        client_id = self._client_id("ENT", trade.trade_id)

        try:
            meta = self._get_symbol(trade.pair)
            quantity, target, actual = self._auto_quantity(
                meta,
                trade.entry,
                trade.margin_usdt,
                trade.leverage,
            )

            await self.real.ensure_position_mode()
            existing_positions = await self.real.get_positions(trade.pair)
            if existing_positions:
                descriptions = ", ".join(
                    f"{item.get('positionSide', 'BOTH')}:{item.get('positionAmt', '0')}"
                    for item in existing_positions
                )
                raise BinanceAPIError(
                    f"Position Binance sudah ada pada {trade.pair} ({descriptions}). "
                    "Bot tidak membuat real entry baru pada symbol yang sudah memiliki exposure.",
                    endpoint="/fapi/v2/positionRisk",
                )

            position_side = self._position_side_for_direction(trade.direction)
            await self.real.set_leverage(trade.pair, trade.leverage)

            order = await self.real.place_limit_entry(
                symbol=trade.pair,
                side=trade.direction,
                quantity=quantity,
                price=trade.entry,
                position_side=position_side,
                client_order_id=client_id,
            )
        except MarginInfluenceError as exc:
            trade.real_state = "REAL_ERROR"
            trade.real_error = str(exc)
            log.warning(
                "MARGIN INFLUENCE | pair=%s | trade_id=%s | %s",
                trade.pair,
                trade.trade_id,
                exc,
            )
            raise
        except BinanceRateLimitError as exc:
            trade.real_state = "REAL_ERROR"
            trade.real_error = str(exc)
            self._notify_rate_limit(exc, f"PLACE LIMIT {trade.pair}")
            raise
        except BinanceAPIError as exc:
            # POST /order can have unknown execution after a 5XX/transport
            # failure. Query by the unique clientOrderId before deciding that
            # no order exists; never blindly submit a second LIMIT order.
            if trade.pair and (exc.status_code is None or exc.status_code >= 500):
                try:
                    recovered = await self.real.get_order(
                        trade.pair,
                        client_order_id=client_id,
                    )
                    if recovered.get("orderId") is not None:
                        order = recovered
                    else:
                        raise exc
                except BinanceAPIError as lookup_exc:
                    if lookup_exc.code != -2013:
                        trade.real_state = "REAL_ERROR"
                        trade.real_error = str(lookup_exc)
                        await self._handle_real_exception(
                            lookup_exc,
                            "RECOVER LIMIT ORDER AFTER UNKNOWN EXECUTION",
                            trade,
                        )
                        raise lookup_exc
                    trade.real_state = "REAL_ERROR"
                    trade.real_error = str(exc)
                    await self._handle_real_exception(
                        exc,
                        "PLACE LIMIT ENTRY",
                        trade,
                    )
                    raise
            else:
                trade.real_state = "REAL_ERROR"
                trade.real_error = str(exc)
                await self._handle_real_exception(
                    exc,
                    "PLACE LIMIT ENTRY",
                    trade,
                )
                raise
        except Exception as exc:
            trade.real_state = "REAL_ERROR"
            trade.real_error = str(exc)
            await self._handle_real_exception(
                exc,
                "PLACE LIMIT ENTRY",
                trade,
            )
            raise

        trade.quantity = quantity
        trade.target_notional = target
        trade.actual_notional = actual
        trade.real_enabled = True
        trade.real_state = "REAL_PENDING"
        trade.real_error = None
        trade.position_side = position_side

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
            # If a LIMIT is only partially filled, leave the remaining order
            # controlled by Binance but immediately reconcile the position.
            # Protective closePosition algos protect the actual live position.
            if not await self._confirm_real_fill(trade):
                trade.real_state = "REAL_PENDING"
                trade.real_error = (
                    f"Order {status}, tetapi position belum terkonfirmasi."
                )

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

        await self._ensure_real_protective_orders(trade)
        return True

    async def _ensure_real_protective_orders(self, trade: Trade) -> None:
        if not self.real_mode or not trade.real_enabled:
            return

        position = await self.real.get_position(trade.pair, trade.direction)
        if position is None:
            return

        position_side = str(position.get("positionSide") or self._position_side_for_direction(trade.direction))
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

        existing_sl = find_existing("SL", trade.sl_algo_id, trade.sl_client_algo_id)
        if existing_sl is not None:
            trade.sl_client_algo_id = str(existing_sl.get("clientAlgoId") or trade.sl_client_algo_id or "") or None
            try:
                trade.sl_algo_id = int(existing_sl.get("algoId"))
            except (TypeError, ValueError):
                trade.sl_algo_id = trade.sl_algo_id
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

        existing_tp = find_existing("TP", trade.tp_algo_id, trade.tp_client_algo_id)
        if existing_tp is not None:
            trade.tp_client_algo_id = str(existing_tp.get("clientAlgoId") or trade.tp_client_algo_id or "") or None
            try:
                trade.tp_algo_id = int(existing_tp.get("algoId"))
            except (TypeError, ValueError):
                trade.tp_algo_id = trade.tp_algo_id
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

    async def _cancel_bot_algo_orders(self, trade: Trade) -> None:
        """Cancel this bot trade's TP/SL algos without touching unrelated algos."""
        open_algos = await self.real.get_open_algo_orders(trade.pair)
        tracked_ids = {
            str(value)
            for value in (trade.tp_algo_id, trade.sl_algo_id)
            if value is not None
        }
        tracked_clients = {
            str(value)
            for value in (trade.tp_client_algo_id, trade.sl_client_algo_id)
            if value
        }
        prefixes = (
            f"TP-{trade.trade_id}",
            f"SL-{trade.trade_id}",
        )

        for item in open_algos:
            algo_id = str(item.get("algoId") or "")
            client_id = str(item.get("clientAlgoId") or "")
            owned = (
                algo_id in tracked_ids
                or client_id in tracked_clients
                or client_id.startswith(prefixes[0] + "-")
                or client_id.startswith(prefixes[1] + "-")
                or client_id in prefixes
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

        await self._cancel_bot_algo_orders(trade)

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

    async def _replace_real_sl(self, trade: Trade, new_sl: Decimal) -> None:
        if not self.real_mode or not trade.real_enabled or trade.status != "FILLED":
            return

        position = await self.real.get_position(trade.pair, trade.direction)
        if position is None:
            raise BinanceAPIError(
                f"Position Binance {trade.pair} tidak ditemukan saat /trail.",
                endpoint="/fapi/v2/positionRisk",
            )

        position_side = str(position.get("positionSide") or self._position_side_for_direction(trade.direction))
        exit_side = "SELL" if trade.direction == "BUY" else "BUY"
        new_client = self._client_id("SL", f"{trade.trade_id}-{uuid4().hex[:6]}")

        # Place new protection first, then remove old protection to avoid a gap.
        new_order = await self.real.place_close_algo(
            symbol=trade.pair,
            side=exit_side,
            position_side=position_side,
            order_type="STOP_MARKET",
            trigger_price=new_sl,
            client_algo_id=new_client,
        )

        # Record the newly created protection immediately so an exception during
        # old-order cancellation can never leave the new SL orphaned.
        new_client_id = str(new_order.get("clientAlgoId") or new_client)
        try:
            new_algo_id = int(new_order.get("algoId"))
        except (TypeError, ValueError):
            new_algo_id = None

        old_id = trade.sl_algo_id

        if old_id is not None:
            try:
                await self.real.cancel_algo_order(
                    symbol=trade.pair,
                    algo_id=old_id,
                )
            except BinanceAPIError as exc:
                if exc.code not in {-2011, -2013}:
                    # First reconcile the old/new protection state. If the old
                    # order is still open, rollback the new order so we do not
                    # accidentally maintain two competing SLs.
                    try:
                        open_algos = await self.real.get_open_algo_orders(trade.pair)
                    except Exception:
                        # We keep the newly created protection tracked because
                        # its existence is known, and surface the original error.
                        trade.sl_client_algo_id = new_client_id
                        trade.sl_algo_id = new_algo_id
                        trade.real_error = (
                            "Old SL cancellation failed and Binance open-algo "
                            "reconciliation also failed."
                        )
                        raise

                    old_still_open = any(
                        str(item.get("algoId") or "") == str(old_id)
                        for item in open_algos
                    )
                    new_is_open = any(
                        (
                            str(item.get("algoId") or "") == str(new_algo_id)
                            if new_algo_id is not None
                            else False
                        )
                        or str(item.get("clientAlgoId") or "") == new_client_id
                        for item in open_algos
                    )

                    if old_still_open and new_is_open:
                        try:
                            await self.real.cancel_algo_order(
                                symbol=trade.pair,
                                algo_id=new_algo_id if new_algo_id is not None else None,
                                client_algo_id=(
                                    None if new_algo_id is not None else new_client_id
                                ),
                            )
                        except Exception:
                            trade.sl_client_algo_id = new_client_id
                            trade.sl_algo_id = new_algo_id
                            trade.real_error = (
                                "Old dan new SL sama-sama terdeteksi aktif; "
                                "rollback new SL gagal."
                            )
                            raise

                        raise exc

                    if not old_still_open and new_is_open:
                        # Old order is already gone; the new order is the active protection.
                        trade.sl_client_algo_id = new_client_id
                        trade.sl_algo_id = new_algo_id
                    else:
                        raise exc

        trade.sl_client_algo_id = new_client_id
        trade.sl_algo_id = new_algo_id
        trade.real_error = None

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
                await self._ensure_real_protective_orders(trade)
                trade.real_state = "REAL_FILLED"
                trade.real_error = None
                return

            # Position zero. Determine whether one of the bot's tracked algos
            # is already triggered. If so, finalize through the normal close
            # pipeline so the setup is removed from /trade, history is written
            # once, and WebSocket symbol cleanup still happens.
            for result, algo_id, fallback_price in (
                ("TP", trade.tp_algo_id, trade.tp),
                ("SL", trade.sl_algo_id, trade.sl),
            ):
                if algo_id is None:
                    continue
                try:
                    algo = await self.real.get_algo_order(algo_id=algo_id)
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

            trade.real_state = "REAL_ERROR"
            trade.real_error = (
                "Position Binance tidak ada saat /open dan penyebab penutupan "
                "tidak dapat dipastikan dari tracked algo order."
            )
            log.warning(
                "REAL position %s tidak ditemukan saat /open dan hasil close tidak dapat dipastikan.",
                trade.trade_id,
            )

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
        await self.reply(
            "✅ LEVERAGE DIPERBARUI\n\n"
            f"Leverage default: {self.leverage}x\n"
            f"Margin: {decimal_to_str(self.margin_usdt)} USDT\n"
            f"Target Notional: {decimal_to_str(self.margin_usdt * Decimal(self.leverage))} USDT"
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

        if len(self.active_trades) >= MAX_ACTIVE_TRADES:
            await self.reply(
                "Maksimum active trade tercapai.\n"
                f"Batas: {MAX_ACTIVE_TRADES}"
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

        pair = data["pair"]
        direction = data["direction"]
        reference = data["price_now_reference"]
        entry = data["entry"]
        price_exp = data["price_exp"]
        sl = data["sl"]
        tp = data["tp"]

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
            raise ValueError(
                "Setup belum lengkap."
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

        if self.real_mode:
            existing_real = [
                active
                for active in self.active_trades.values()
                if active.real_enabled and active.result is None and active.pair == pair
            ]
            if existing_real:
                raise ValueError(
                    f"REAL setup untuk {pair} sudah aktif. "
                    "Satu symbol hanya boleh memiliki satu active REAL setup agar "
                    "position Binance tidak tertukar antar setup."
                )

        trade = Trade(
            trade_id=generate_trade_id(pair),
            session_id=self.session_id,
            pair=pair,
            direction=direction,
            price_now_reference=reference,
            entry=entry,
            entry_reason=data["entry_reason"],
            price_exp=price_exp,
            price_exp_reason=data["price_exp_reason"],
            sl=sl,
            sl_reason=data["sl_reason"],
            tp=tp,
            tp_reason=data["tp_reason"],
            margin_usdt=self.margin_usdt,
            leverage=self.leverage,
            real_enabled=self.real_mode,
            real_state="REAL_PENDING" if self.real_mode else "SIMULATION",
            position_side=self._position_side_for_direction(direction) if self.real_mode else None,
        )

        self.active_trades[
            trade.trade_id
        ] = trade

        real_entry_error: Exception | None = None

        if self.real_mode:
            try:
                await self._ensure_real_entry(trade)
            except Exception as exc:
                # Setup tetap dipertahankan di /trade untuk troubleshooting.
                # Jangan pernah melaporkan "order real berhasil" ketika entry
                # belum terkonfirmasi oleh Binance.
                real_entry_error = exc

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
                pair
            )
        except Exception:
            log.exception(
                "Subscription WebSocket %s gagal saat /add.",
                pair,
            )

        title = "✅ SETUP DITAMBAHKAN"
        if real_entry_error is not None:
            title = "⚠️ SETUP DITAMBAHKAN — ORDER REAL BELUM BERHASIL"

        real_block = ""
        if trade.real_enabled:
            real_block = (
                f"Real: ON\n"
                f"Real State: {trade.real_state}\n"
                f"Margin: {decimal_to_str(trade.margin_usdt)} USDT\n"
                f"Leverage: {trade.leverage}x\n"
                f"Quantity: {decimal_to_str(trade.quantity) or '-'}\n"
                f"Target Notional: {decimal_to_str(trade.target_notional) or '-'} USDT\n"
                f"Actual Notional: {decimal_to_str(trade.actual_notional) or '-'} USDT\n\n"
            )

        error_block = ""
        if real_entry_error is not None:
            label = (
                "MARGIN INFLUENCE"
                if isinstance(real_entry_error, MarginInfluenceError)
                else "REAL ORDER ERROR"
            )
            error_block = (
                f"{label}: {real_entry_error}\n\n"
                "Order real belum dianggap berhasil.\n"
            )

        await self.reply(
            f"{title}\n\n"
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
            f"{real_block}"
            f"{error_block}"
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

        if kind == "FILLED":
            self.flow = None
            await self.reply(
                "FILLED dibatalkan."
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

        if self.real_mode and trade.real_enabled and trade.status == "FILLED":
            await self._replace_real_sl(trade, new_sl)

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

            async with self._trade_lock:
                current = self.active_trades.get(trade.trade_id)

                if current is None:
                    raise ValueError(
                        "Setup sudah tidak aktif. Gunakan /trade untuk melihat daftar terbaru."
                    )

                if current.status != "PENDING":
                    raise ValueError(
                        f"Setup {current.pair} sudah berstatus {current.status}. "
                        "Hanya PENDING yang bisa diproses /filled."
                    )

                trade = current

            if trade.real_enabled:
                if not self.real_mode:
                    trade.real_state = "REAL_DETACHED"
                    raise ValueError(
                        "Setup ini terhubung ke Binance REAL. Nyalakan /real on "
                        "untuk melakukan verifikasi position sebelum /filled."
                    )

                if not await self._confirm_real_fill(trade):
                    raise ValueError(
                        f"Binance belum mengonfirmasi position aktif untuk {trade.pair}. "
                        "/filled tidak membuat position baru."
                    )
            else:
                async with self._trade_lock:
                    current = self.active_trades.get(trade.trade_id)
                    if current is None or current.status != "PENDING":
                        raise ValueError("Setup berubah sebelum /filled diproses.")
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
            trade_copy = copy.deepcopy(
                trade
            )

            if self.real_mode and trade.real_enabled:
                try:
                    await self._real_delete_cleanup(trade)
                except Exception as exc:
                    await self._handle_real_exception(exc, "DELETE/CLOSE REAL", trade)
                    await self.reply(
                        "❌ /del dibatalkan. Binance belum terkonfirmasi bersih.\n\n"
                        f"Pair: {trade.pair}\n"
                        f"Error: {exc}"
                    )
                    return

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
    # TRADE DISPLAY
    # --------------------------------------------------------

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

            if trade.status == "FILLED":
                pnl_text = format_pct(
                    pct_change(
                        trade.direction,
                        trade.entry,
                        snapshot.price,
                    )
                )

        direction_icon = "🟢" if trade.direction == "BUY" else "🔴"
        status_icon = {
            "PENDING": "⏳",
            "FILLED": "✅",
        }.get(
            trade.status,
            "•",
        )

        mode = "TRAIL" if trade.trailing else "NORMAL"
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
            f"│ 📈 PnL   {pnl_text}",
            (
                f"│ 💰 Real   ON • {decimal_to_str(trade.margin_usdt)} USDT • {trade.leverage}x"
                if trade.real_enabled else ""
            ),
            (
                f"│ 📦 Qty    {decimal_to_str(trade.quantity) or '-'} • {trade.real_state}"
                if trade.real_enabled else ""
            ),
            f"│ ⚙️ Mode   {mode}",
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

        blocks = [
            "╭──────────────╮",
            "│  📊 TRADE   │",
            "╰──────────────╯",
            "",
            f"Active  {len(items)}   •   ⏳ {pending}   •   ✅ {filled}",
            f"Trail   {trailing}   •   📡 Live Feed {live}",
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

    async def _notify_close(
        self,
        trade: Trade,
    ) -> None:
        result = trade.result or "CLOSED"

        title = {
            "TP": "✅ TP TERCAPAI",
            "SL": "🛑 SL TERCAPAI",
            "EXPIRED": "⏳ PRICE EXPIRED",
            "DELETED": "🗑️ DELETED",
        }.get(
            result,
            "TRADE CLOSED",
        )

        pnl = (
            format_pct(trade.pnl_percent)
            if trade.pnl_percent is not None
            else "-"
        )

        if result == "EXPIRED":
            # Notifikasi khusus supaya jelas bahwa EXPIRED berasal dari
            # Price Exp, bukan dari TP/SL atau aturan lain.
            await self.reply(
                f"{title}\n\n"
                f"Pair: {trade.pair}\n"
                f"Direction: {trade.direction.title()}\n"
                f"Entry: {decimal_to_str(trade.entry)}\n"
                f"Price Exp: {decimal_to_str(trade.price_exp)}\n"
                f"Harga Pemicu: {decimal_to_str(trade.exit_price)}\n"
                f"Result: EXPIRED\n"
                f"PnL: -\n\n"
                "Pemicu: PRICE EXP SAJA\n"
                f"Reason Price Exp: {trade.result_reason or '-'}"
            )
            return

        await self.reply(
            f"{title}\n\n"
            f"Pair: {trade.pair}\n"
            f"Direction: {trade.direction.title()}\n"
            f"Entry: {decimal_to_str(trade.fill_price or trade.entry)}\n"
            f"Exit: {decimal_to_str(trade.exit_price)}\n"
            f"Result: {result}\n"
            f"PnL: {pnl}\n\n"
            f"Reason: {trade.result_reason or '-'}"
        )

    async def _fill_trade(
        self,
        trade: Trade,
        price: Decimal,
    ) -> None:
        if trade.real_enabled and not self.real_mode:
            trade.real_state = "REAL_DETACHED"
            return

        if self.real_mode and trade.real_enabled:
            try:
                confirmed = await self._confirm_real_fill(trade)
            except Exception as exc:
                await self._handle_real_exception(exc, "CONFIRM REAL ENTRY", trade)
                return

            if not confirmed:
                # Harga WS menyentuh Entry, tetapi Binance belum mengonfirmasi
                # position aktif. Jangan mengubah local status menjadi FILLED.
                return

        else:
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
                    trade.fill_price or trade.entry
                ),
                "real_mode": trade.real_enabled,
            },
        )

        await self.reply(
            "✅ ENTRY FILLED\n\n"
            f"Pair: {trade.pair}\n"
            f"Direction: {trade.direction.title()}\n"
            f"Entry: {decimal_to_str(trade.entry)}\n"
            f"Fill Price: {decimal_to_str(trade.fill_price or trade.entry)}\n"
            f"Reason Entry: {trade.entry_reason}\n"
            + (
                f"Quantity: {decimal_to_str(trade.quantity) or '-'}\n"
                f"Notional: {decimal_to_str(trade.actual_notional) or '-'} USDT\n"
                "Execution: REAL\n"
                if trade.real_enabled else ""
            )
            + "Status: FILLED"
        )

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

            # A setup that already belongs to a REAL Binance order/position
            # must never silently switch into simulation when /real is OFF.
            if trade.real_enabled and not self.real_mode:
                trade.real_state = "REAL_DETACHED"
                continue

            try:
                if trade.status == "PENDING":
                    # PENDING hanya dipengaruhi oleh Entry dan Price Exp.
                    # Price Exp diperiksa langsung terhadap tick Binance yang
                    # sedang diterima. Tidak memakai previous-price crossing
                    # atau baseline REST sebagai syarat tambahan.
                    if trade.direction == "BUY":
                        if price <= trade.entry:
                            await self._fill_trade(
                                trade,
                                price,
                            )
                            continue

                        if price >= trade.price_exp:
                            if trade.real_enabled and not self.real_mode:
                                trade.real_state = "REAL_DETACHED"
                                continue

                            if self.real_mode and trade.real_enabled:
                                try:
                                    await self._real_pending_price_exp(trade)
                                except Exception as exc:
                                    await self._handle_real_exception(exc, "PRICE EXP REAL RECONCILIATION", trade)
                                    continue
                                if trade.status == "FILLED":
                                    continue

                            await self._finalize_trade(
                                trade,
                                result="EXPIRED",
                                exit_price=price,
                                reason=trade.price_exp_reason,
                            )
                            continue

                    else:
                        if price >= trade.entry:
                            await self._fill_trade(
                                trade,
                                price,
                            )
                            continue

                        if price <= trade.price_exp:
                            if trade.real_enabled and not self.real_mode:
                                trade.real_state = "REAL_DETACHED"
                                continue

                            if self.real_mode and trade.real_enabled:
                                try:
                                    await self._real_pending_price_exp(trade)
                                except Exception as exc:
                                    await self._handle_real_exception(exc, "PRICE EXP REAL RECONCILIATION", trade)
                                    continue
                                if trade.status == "FILLED":
                                    continue

                            await self._finalize_trade(
                                trade,
                                result="EXPIRED",
                                exit_price=price,
                                reason=trade.price_exp_reason,
                            )
                            continue

                elif trade.status == "FILLED":
                    if trade.real_enabled and not self.real_mode:
                        trade.real_state = "REAL_DETACHED"
                        continue

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

                        trade = current

                    if trade.direction == "BUY":
                        # SL diperiksa lebih dulu.
                        if price <= trade.sl:
                            if self.real_mode and trade.real_enabled:
                                try:
                                    if await self._real_exit_trigger(trade, "SL", price):
                                        await self._finalize_trade(
                                            trade,
                                            result="SL",
                                            exit_price=price,
                                            reason=trade.sl_reason,
                                        )
                                except Exception as exc:
                                    await self._handle_real_exception(exc, "VERIFY REAL SL CLOSE", trade)
                                continue
                            await self._finalize_trade(
                                trade,
                                result="SL",
                                exit_price=price,
                                reason=trade.sl_reason,
                            )
                            continue

                        if price >= trade.tp:
                            if self.real_mode and trade.real_enabled:
                                try:
                                    if await self._real_exit_trigger(trade, "TP", price):
                                        await self._finalize_trade(
                                            trade,
                                            result="TP",
                                            exit_price=price,
                                            reason=trade.tp_reason,
                                        )
                                except Exception as exc:
                                    await self._handle_real_exception(exc, "VERIFY REAL TP CLOSE", trade)
                                continue
                            await self._finalize_trade(
                                trade,
                                result="TP",
                                exit_price=price,
                                reason=trade.tp_reason,
                            )
                            continue

                    else:
                        if price >= trade.sl:
                            if self.real_mode and trade.real_enabled:
                                try:
                                    if await self._real_exit_trigger(trade, "SL", price):
                                        await self._finalize_trade(
                                            trade,
                                            result="SL",
                                            exit_price=price,
                                            reason=trade.sl_reason,
                                        )
                                except Exception as exc:
                                    await self._handle_real_exception(exc, "VERIFY REAL SL CLOSE", trade)
                                continue
                            await self._finalize_trade(
                                trade,
                                result="SL",
                                exit_price=price,
                                reason=trade.sl_reason,
                            )
                            continue

                        if price <= trade.tp:
                            if self.real_mode and trade.real_enabled:
                                try:
                                    if await self._real_exit_trigger(trade, "TP", price):
                                        await self._finalize_trade(
                                            trade,
                                            result="TP",
                                            exit_price=price,
                                            reason=trade.tp_reason,
                                        )
                                except Exception as exc:
                                    await self._handle_real_exception(exc, "VERIFY REAL TP CLOSE", trade)
                                continue
                            await self._finalize_trade(
                                trade,
                                result="TP",
                                exit_price=price,
                                reason=trade.tp_reason,
                            )
                            continue

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
        total_records = len(records)

        tp = sum(
            1
            for record in records
            if record.get("result") == "TP"
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

        total_filled = tp + sl

        if total_filled:
            win_rate = (
                Decimal(tp)
                / Decimal(total_filled)
                * Decimal("100")
            )
        else:
            win_rate = Decimal("0")

        pnls = []

        for record in records:
            value = record.get("pnl_percent")

            if value is None:
                continue

            try:
                pnls.append(
                    Decimal(str(value))
                )
            except InvalidOperation:
                continue

        gross_net = sum(
            pnls,
            Decimal("0"),
        )

        wins = [
            value
            for value in pnls
            if value > 0
        ]

        losses = [
            value
            for value in pnls
            if value < 0
        ]

        average_win = (
            sum(
                wins,
                Decimal("0"),
            )
            / Decimal(len(wins))
            if wins
            else Decimal("0")
        )

        average_loss = (
            sum(
                losses,
                Decimal("0"),
            )
            / Decimal(len(losses))
            if losses
            else Decimal("0")
        )

        trail_count = sum(
            len(
                record.get("trail_history") or []
            )
            for record in records
        )

        return {
            "total_records": total_records,
            "total_filled": total_filled,
            "tp": tp,
            "sl": sl,
            "expired": expired,
            "deleted": deleted,
            "win_rate": win_rate,
            "gross_pnl_percent": gross_net,
            "average_win_percent": average_win,
            "average_loss_percent": average_loss,
            "trail_count": trail_count,
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

        await self.reply(
            "📊 STATS\n\n"
            f"Total Setup History: {stats['total_records']}\n"
            f"Total Trade Filled: {stats['total_filled']}\n"
            f"Active Pending: {active_pending}\n"
            f"Active Filled: {active_filled}\n\n"
            f"TP: {stats['tp']}\n"
            f"SL: {stats['sl']}\n"
            f"Expired: {stats['expired']}\n"
            f"Deleted: {stats['deleted']}\n\n"
            f"Win Rate: {format_pct(stats['win_rate']).replace('+', '')}\n"
            f"PnL History: {format_pct(stats['gross_pnl_percent'])}\n"
            f"Average Win: {format_pct(stats['average_win_percent'])}\n"
            f"Average Loss: {format_pct(stats['average_loss_percent'])}\n"
            f"Total Trail Event: {stats['trail_count']}\n\n"
            "Definisi Win Rate:\n"
            "TP / (TP + SL)\n"
            "Expired dan Deleted tidak dihitung sebagai win/loss."
        )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    def _build_analysis(
        self,
        records: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        stats = self._calculate_stats(
            records
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

        full_data = {
            "generated_at": iso_utc(),
            "generated_at_wib": format_wib(
                now_utc()
            ),
            "session_id": self.session_id,
            "source": {
                "market": "Binance USDⓈ-M Futures",
                "execution": "SIMULATION/REAL (runtime controlled by /real)",
                "price_trigger": "aggTrade last price",
            },
            "summary": {
                "total_setup_history": stats[
                    "total_records"
                ],
                "total_filled_trades": stats[
                    "total_filled"
                ],
                "tp": stats["tp"],
                "sl": stats["sl"],
                "expired": stats["expired"],
                "deleted": stats["deleted"],
                "win_rate_percent": decimal_to_str(
                    stats["win_rate"]
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
                "trail_event_count": stats[
                    "trail_count"
                ],
                "trailing_trade_count": total_trailing_trades,
            },
            "active_session": {
                "active_trade_count": len(
                    self.active_trades
                ),
                "active_trade_ids": list(
                    self.active_trades.keys()
                ),
            },
            "trades": records,
            "events": events,
            "analysis": {
                "by_pair": pair_rows,
                "by_direction": direction_rows,
                "by_strategy": strategy_rows,
            },
        }

        report_lines = [
            "# Trading Analysis",
            "",
            f"Generated: {format_wib(now_utc())}",
            "",
            "## System",
            "",
            "- Market: Binance USDⓈ-M Futures",
            "- Execution: Simulation/Real (runtime controlled by /real)",
            "- Price Trigger: aggTrade last price",
            "",
            "## Summary",
            "",
            f"- Total Setup History: {stats['total_records']}",
            f"- Total Filled Trade: {stats['total_filled']}",
            f"- TP: {stats['tp']}",
            f"- SL: {stats['sl']}",
            f"- Expired: {stats['expired']}",
            f"- Deleted: {stats['deleted']}",
            f"- Win Rate: {decimal_to_str(stats['win_rate'])}%",
            f"- PnL History: {decimal_to_str(stats['gross_pnl_percent'])}%",
            f"- Average Win: {decimal_to_str(stats['average_win_percent'])}%",
            f"- Average Loss: {decimal_to_str(stats['average_loss_percent'])}%",
            f"- Trail Event: {stats['trail_count']}",
            f"- Trade dengan Trailing: {total_trailing_trades}",
            "",
            "Win Rate = TP / (TP + SL).",
            "Expired dan Deleted tidak dihitung sebagai win/loss.",
            "",
            "## By Pair",
            "",
            "| Pair | Setup | Filled | TP | SL | Win Rate | PnL % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]

        for row in pair_rows:
            report_lines.append(
                "| "
                f"{row['pair']} | "
                f"{row['records']} | "
                f"{row['filled']} | "
                f"{row['tp']} | "
                f"{row['sl']} | "
                f"{row['win_rate'] or '0'}% | "
                f"{row['pnl_percent'] or '0'} |"
            )

        report_lines.extend(
            [
                "",
                "## By Direction",
                "",
                "| Direction | Setup | Filled | TP | SL | Win Rate | PnL % |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for row in direction_rows:
            report_lines.append(
                "| "
                f"{row['direction']} | "
                f"{row['records']} | "
                f"{row['filled']} | "
                f"{row['tp']} | "
                f"{row['sl']} | "
                f"{row['win_rate'] or '0'}% | "
                f"{row['pnl_percent'] or '0'} |"
            )

        report_lines.extend(
            [
                "",
                "## By Strategy",
                "",
                "| Strategy | Version | Setup | Filled | TP | SL | Win Rate | PnL % |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for row in strategy_rows:
            report_lines.append(
                "| "
                f"{row['strategy_name']} | "
                f"{row['strategy_version']} | "
                f"{row['records']} | "
                f"{row['filled']} | "
                f"{row['tp']} | "
                f"{row['sl']} | "
                f"{row['win_rate'] or '0'}% | "
                f"{row['pnl_percent'] or '0'} |"
            )

        report_lines.extend(
            [
                "",
                "## Dataset",
                "",
                f"- Trade records: {len(records)}",
                f"- Event records: {len(events)}",
                "",
                "File ini adalah hasil export dari histori simulator.",
            ]
        )

        return (
            full_data,
            "\n".join(report_lines)
            + "\n",
        )

    async def analyze(self) -> None:
        # Gunakan history RAM yang sudah dimuat saat startup dan
        # terus diperbarui selama session.
        full_data, report = self._build_analysis(
            self.history_records,
            self.history_events,
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
            f"Execution: {'REAL' if self.real_mode else 'SIMULATION'}\n"
            f"Session: {self.session_id}\n\n"
            f"Binance REST: {'REAL READY' if self.real_mode else 'MARKET DATA ONLY'}\n"
            f"Real Mode: {'ON' if self.real_mode else 'OFF'}\n"
            f"Margin Default: {decimal_to_str(self.margin_usdt)} USDT\n"
            f"Leverage Default: {self.leverage}x\n"
            f"WebSocket: {self.ws.status}\n"
            f"Symbols Subscribed: {len(self.ws.symbols())}\n"
            f"Fresh Price Feed: {fresh_prices}\n\n"
            f"Active Setup: {len(self.active_trades)}\n"
            f"Pending: {pending}\n"
            f"Filled: {filled}\n\n"
            f"History Records: {len(self.history_records)}\n"
            f"Catatan: {len(self.notes)}\n"
        )

    async def show_help(self) -> None:
        await self.reply(
            "COMMAND MAIN.PY\n\n"
            "/add - membuat setup baru\n"
            "/back - kembali satu langkah / batalkan sesi\n"
            "/trade - melihat setup aktif + harga live\n"
            "/setup - simpan semua setup aktif /trade ke GitHub\n"
            "/open - buka kembali setup yang tersimpan di GitHub\n"
            "/trail - mengubah SL setup\n"
            "/del - menghapus setup\n"
            "/filled - ubah setup PENDING menjadi FILLED secara manual\n"
            "/real [on|off] - aktif/nonaktif real execution Binance\n"
            "/margin [USDT] - atur margin default\n"
            "/leverage [1-125] - atur leverage default\n"
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
                    "/add",
                    "/trail",
                    "/del",
                    "/filled",
                    "/real",
                    "/margin",
                    "/leverage",
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

            if command == "/filled":
                await self._start_filled()
                return

            if command == "/real":
                argument = text[len(text.split(maxsplit=1)[0]):].strip().lower()
                if argument in {"", "status"}:
                    await self.reply(
                        "🌐 REAL MODE\n\n"
                        f"Status: {'ON' if self.real_mode else 'OFF'}\n"
                        f"Margin: {decimal_to_str(self.margin_usdt)} USDT\n"
                        f"Leverage: {self.leverage}x\n\n"
                        "Gunakan /real on atau /real off.\n"
                        "REAL OFF tidak akan melepas posisi/order real yang sedang aktif."
                    )
                elif argument in {"on", "1"}:
                    await self._start_real_on()
                elif argument in {"off", "0"}:
                    await self._set_real_off()
                else:
                    raise ValueError("Gunakan /real on, /real off, atau /real.")
                return

            if command == "/margin":
                argument = text[len(text.split(maxsplit=1)[0]):].strip()
                await self._set_margin_command(argument)
                return

            if command == "/leverage":
                argument = text[len(text.split(maxsplit=1)[0]):].strip()
                await self._set_leverage_command(argument)
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
