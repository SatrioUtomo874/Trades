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
- Binance USDⓈ-M Futures = market data saja.
- Tidak ada fungsi order / trading API Binance.
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
- /wrong menghapus active setup yang benar-benar salah tanpa mencatat history.
- /remove menghapus satu trade dari histori GitHub beserta event dan journal terkait.
- strategy.py wajib menyediakan generate_setup(pair, context).
"""

import asyncio
import base64
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
import traceback
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any
from urllib.parse import quote
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

# Key Binance tersedia di env, tetapi sengaja TIDAK dipakai.
# Market data publik tidak membutuhkan signing / order credentials.
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
    """Forward ERROR/EXCEPTION logs from background tasks to Telegram."""

    def __init__(self, engine: "TradingEngine") -> None:
        super().__init__(level=logging.ERROR)
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
    strategy_source: str = "MANUAL"
    strategy_confidence: Decimal | None = None
    strategy_data_source: str | None = None
    strategy_analysis: dict[str, Any] = field(default_factory=dict)

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
            "strategy_source": self.strategy_source,
            "strategy_confidence": (
                decimal_to_str(self.strategy_confidence)
                if self.strategy_confidence is not None
                else None
            ),
            "strategy_data_source": self.strategy_data_source,
            "strategy_analysis": copy.deepcopy(self.strategy_analysis),
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

            for filt in raw.get("filters", []):
                if filt.get("filterType") == "PRICE_FILTER":
                    tick_size = Decimal(
                        str(filt.get("tickSize") or "0")
                    )
                    min_price = Decimal(
                        str(filt.get("minPrice") or "0")
                    )
                    max_price = Decimal(
                        str(filt.get("maxPrice") or "0")
                    )
                    break

            result[symbol] = SymbolMeta(
                symbol=symbol,
                status=status,
                contract_type=contract_type,
                quote_asset=quote_asset,
                tick_size=tick_size,
                min_price=min_price,
                max_price=max_price,
            )

        return result

    async def get_price(self, symbol: str) -> Decimal:
        payload = await self._get(
            "/fapi/v2/ticker/price",
            params={"symbol": symbol},
        )

        return parse_decimal(str(payload["price"]))


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

        # Market-event ordering is also guarded at engine level. This is a
        # second safety layer so an out-of-order callback can never trigger
        # a state transition. Key = (Binance event time, aggregate trade ID).
        self._last_market_event_key: dict[str, tuple[int, int]] = {}
        self._last_live_price: dict[str, Decimal] = {}


        self.flow: dict[str, Any] | None = None
        self._auto_task: asyncio.Task[Any] | None = None
        self._auto_job_id: str | None = None

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
                "🚨 ERROR BACKEND\n\n"
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
            "Mode: SIMULATION ONLY\n"
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

        fill_price = (
            parse_decimal(str(record["fill_price"]))
            if record.get("fill_price") not in (None, "")
            else None
        )
        pnl_percent = (
            parse_decimal(str(record["pnl_percent"]))
            if record.get("pnl_percent") not in (None, "")
            else None
        )

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
            strategy_source=str(record.get("strategy_source") or "MANUAL"),
            strategy_confidence=(
                parse_decimal(str(record["strategy_confidence"]))
                if record.get("strategy_confidence") not in (None, "")
                else None
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
                return parse_decimal(str(payload[name]))
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

        if len(self.active_trades) >= MAX_ACTIVE_TRADES:
            await self.reply(
                "Maksimum active trade tercapai.\n"
                f"Batas: {MAX_ACTIVE_TRADES}"
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

        trade = self._build_trade_from_setup(
            data,
            strategy_meta=strategy_meta,
        )

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

        # Shared technical validation for both /add and /auto. Strategy.py
        # decides the setup; main.py only ensures all prices are legal for
        # the symbol and preserve the existing setup geometry.
        for price_value in (
            reference,
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

        meta = strategy_meta or {}

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

        trade = self._build_trade_from_setup(
            data,
            strategy_meta={
                "strategy_name": "MANUAL",
                "strategy_version": "1.0",
                "strategy_source": "MANUAL",
            },
        )

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

                current.status = "FILLED"
                current.filled_at = now_utc()

                # Konsisten dengan fill otomatis: posisi dianggap masuk pada
                # level Entry, sementara harga saat command hanya dicatat
                # sebagai informasi tambahan.
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
            trade_copy = copy.deepcopy(
                trade
            )

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
            f"│ ⚙️ Mode   {mode}",
            f"│ 🧠 {reason}",
            "╰────────────────────────╯",
        ]

        return "\n".join(lines)

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

        await self.reply(
            "✅ ENTRY FILLED\n\n"
            f"Pair: {trade.pair}\n"
            f"Direction: {trade.direction.title()}\n"
            f"Entry: {decimal_to_str(trade.entry)}\n"
            f"Fill Price: {decimal_to_str(trade.entry)}\n"
            f"Reason Entry: {trade.entry_reason}\n"
            "Status: FILLED"
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

                        trade = current

                    if trade.direction == "BUY":
                        # SL diperiksa lebih dulu.
                        if price <= trade.sl:
                            await self._finalize_trade(
                                trade,
                                result="SL",
                                exit_price=price,
                                reason=trade.sl_reason,
                            )
                            continue

                        if price >= trade.tp:
                            await self._finalize_trade(
                                trade,
                                result="TP",
                                exit_price=price,
                                reason=trade.tp_reason,
                            )
                            continue

                    else:
                        if price >= trade.sl:
                            await self._finalize_trade(
                                trade,
                                result="SL",
                                exit_price=price,
                                reason=trade.sl_reason,
                            )
                            continue

                        if price <= trade.tp:
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
                "execution": "SIMULATION ONLY",
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
            "- Execution in this export: Simulation Only",
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
            "Mode: SIMULATION ONLY\n"
            f"Session: {self.session_id}\n\n"
            "Binance REST: READY\n"
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
