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
- /resetstats mengarsipkan lalu mereset histori trade/event agar stats mulai dari nol (tes strategy baru).
- strategy.py wajib menyediakan generate_setup(pair, context); mode SCAN juga mendukung kontrak optimized scan structure dan validate_setup.
- SCAN maksimal 50 pair dianalisis per cycle; pair mismatch H4 dan below-threshold diban sampai close H4 berikutnya.
"""

import asyncio
import base64
import contextvars
import hashlib
import hmac
import copy
import ctypes
import gc
import gzip
import math
import sqlite3
import threading
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
BALANCE_STALE_SECONDS = 60.0  # /trade segarkan saldo bila cache lebih tua
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
SIM_START_BALANCE = Decimal("100")
SIM_ACCOUNT_PATH = "data/sim_account.json"
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
MEMORY_SOFT_LIMIT_MB = float(os.getenv("MEMORY_SOFT_LIMIT_MB", "400") or "400")
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
CANDIDATES_STATE_PATH = "data/candidates_state.json"
RUNTIME_SETTINGS_PATH = "data/runtime_settings.json"
RESEARCH_EXPORT_INTERVAL_SECONDS = 1800.0  # ringkasan kecil ke GitHub, bukan data candle

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


HEAVY_ANALYSIS_KEYS = frozenset(
    {
        "smc", "raw_result", "htf_poi_hierarchy", "breakers", "order_blocks", "fvg_recent",
        "h4_primary_ranking_for_direction", "liquidity_pools", "liquidity_sweeps_recent",
        "recent_events", "last_pivot_labels", "zones", "candles", "pivots",
    }
)
VALIDATOR_DUPLICATE_KEYS = frozenset({"macro", "multi_timeframe", "smc"})


def _shrink_value(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, dict):
        return {k: _shrink_value(v, depth + 1) for k, v in value.items() if k not in HEAVY_ANALYSIS_KEYS}
    if isinstance(value, (list, tuple)):
        return [_shrink_value(v, depth + 1) for v in value[:8]]
    if isinstance(value, str) and len(value) > 500:
        return value[:500]
    return value


def compact_analysis(analysis: Any) -> dict[str, Any]:
    """Pangkas analisis strategy (salinan ganda, daftar zona) agar RAM dan file history kecil."""
    if not isinstance(analysis, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in analysis.items():
        if key in HEAVY_ANALYSIS_KEYS:
            continue
        if key == "validator" and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in VALIDATOR_DUPLICATE_KEYS}
        out[key] = _shrink_value(value)
    return out


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
    """Kembalikan memori bebas ke OS (gc + malloc_trim)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


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


def gauge(percent: Any, width: int = 10) -> str:
    try:
        value = max(0.0, min(100.0, float(percent)))
    except (TypeError, ValueError):
        value = 0.0
    filled = round(value / 100 * width)
    return "▰" * filled + "▱" * (width - filled)


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
    candidate_id: str = ""
    sizing_mode: str = ""
    risk_info: dict[str, Any] = field(default_factory=dict)

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
            "candidate_id": self.candidate_id,
            "sizing_mode": self.sizing_mode,
            "risk_info": self.risk_info,
            "risk_pct_initial": self._risk_pct_initial(),
            "realized_r": self._realized_r(),
        }

    def _initial_sl_value(self) -> Decimal:
        try:
            if self.trail_history:
                return Decimal(str(self.trail_history[0]["old_sl"]))
        except (InvalidOperation, KeyError, TypeError, IndexError):
            pass
        return self.sl

    def _risk_pct_initial(self) -> str | None:
        base = self.fill_price or self.entry
        if not base or base <= 0:
            return None
        return decimal_to_str(abs(base - self._initial_sl_value()) / base * Decimal("100"))

    def _realized_r(self) -> str | None:
        risk = self._risk_pct_initial()
        if risk is None or self.pnl_percent is None or Decimal(risk) <= 0:
            return None
        return decimal_to_str(self.pnl_percent / Decimal(risk))


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


class InsufficientBalanceError(RuntimeError):
    """Saldo tersedia akun kurang untuk margin entry (kondisi akun, bukan sifat pair)."""


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

# ============================================================
# RESEARCH / RESILIENCE LAYER (blueprint V1)
# Candidate ledger, penyimpanan riset (SQLite), replay M1, risk sizing.
# Layer ini TIDAK memanggil API order; research tidak memegang kredensial.
# ============================================================

rlog = logging.getLogger("trading-research")


def _cfg_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
        if not math.isfinite(value) or (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError("di luar rentang")
        return value
    except (TypeError, ValueError):
        rlog.error("Config %s=%r tidak valid; memakai default %s", name, raw, default)
        return default


def _cfg_int(name: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    return int(_cfg_float(name, float(default), None if lo is None else float(lo), None if hi is None else float(hi)))


def _cfg_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _cfg_opt_float(name: str, lo: float, hi: float) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
        if not math.isfinite(value) or value < lo or value > hi:
            raise ValueError("di luar rentang")
        return value
    except (TypeError, ValueError):
        rlog.error("Config %s=%r tidak valid; fitur terkait dinonaktifkan.", name, raw)
        return None


RESEARCH_ENABLED = _cfg_bool("RESEARCH_ENABLED", False)  # OFF sampai storage/API teruji
BOT_DATA_DIR = (os.getenv("BOT_DATA_DIR") or "").strip()
RESEARCH_MAX_WORKERS = 1
RESEARCH_MAX_PAIRS_PER_BATCH = _cfg_int("RESEARCH_MAX_PAIRS_PER_BATCH", 2, 1, 20)
RESEARCH_MAX_CANDIDATES_PER_DAY = _cfg_int("RESEARCH_MAX_CANDIDATES_PER_DAY", 40, 1, 2000)
RESEARCH_MAX_CANDLES_PER_JOB = _cfg_int("RESEARCH_MAX_CANDLES_PER_JOB", 4320, 100, 100000)
RESEARCH_MAX_RUNTIME_SECONDS = _cfg_float("RESEARCH_MAX_RUNTIME_SECONDS", 20.0, 2.0, 300.0)
RESEARCH_MEMORY_BUDGET_MB = _cfg_float("RESEARCH_MEMORY_BUDGET_MB", 0.0, 0.0, 100000.0)  # 0 = otomatis
RESEARCH_MIN_API_HEADROOM_PCT = _cfg_float("RESEARCH_MIN_API_HEADROOM_PCT", 50.0, 5.0, 95.0)
RESEARCH_MAX_DISK_MB = _cfg_float("RESEARCH_MAX_DISK_MB", 200.0, 10.0, 100000.0)
RESEARCH_REPLAY_HORIZON_HOURS = _cfg_float("RESEARCH_REPLAY_HORIZON_HOURS", 48.0, 1.0, 720.0)
RESEARCH_SAMPLE_REJECT_RATE = _cfg_float("RESEARCH_SAMPLE_REJECT_RATE", 0.05, 0.0, 1.0)
RESEARCH_NEAR_THRESHOLD_POINTS = _cfg_float("RESEARCH_NEAR_THRESHOLD_POINTS", 8.0, 1.0, 40.0)
CANDIDATE_QUEUE_MAX = _cfg_int("CANDIDATE_QUEUE_MAX", 300, 20, 5000)
CANDIDATE_TTL_HOURS = _cfg_float("CANDIDATE_TTL_HOURS", 12.0, 0.5, 168.0)
SCAN_FULL_UNIVERSE_INTERVAL = _cfg_float("SCAN_FULL_UNIVERSE_INTERVAL", 1800.0, 60.0, 86400.0)
SCAN_WATCHLIST_INTERVAL = _cfg_float("SCAN_WATCHLIST_INTERVAL", 900.0, 60.0, 86400.0)
SCAN_VALIDATE_TOP_N = _cfg_int("SCAN_VALIDATE_TOP_N", 6, 1, 50)
SCAN_WATCH_MAX_PER_CYCLE = _cfg_int("SCAN_WATCH_MAX_PER_CYCLE", 15, 0, 50)
SCAN_MIN_API_HEADROOM_PCT = _cfg_float("SCAN_MIN_API_HEADROOM_PCT", 10.0, 0.0, 90.0)
SCAN_READY_RETRY_SECONDS = _cfg_float("SCAN_READY_RETRY_SECONDS", 300.0, 30.0, 3600.0)
SCAN_CYCLE_TIME_BUDGET_SECONDS = _cfg_float("SCAN_CYCLE_TIME_BUDGET_SECONDS", 240.0, 30.0, 3600.0)
RESEARCH_MIN_BUCKET_SAMPLE = _cfg_int("RESEARCH_MIN_BUCKET_SAMPLE", 30, 5, 10000)
EST_FEE_PCT_PER_SIDE = _cfg_float("EST_FEE_PCT_PER_SIDE", 0.05, 0.0, 5.0)
EST_SLIPPAGE_PCT_PER_SIDE = _cfg_float("EST_SLIPPAGE_PCT_PER_SIDE", 0.03, 0.0, 5.0)

# Risk sizing: default LEGACY_NOTIONAL; RISK_BUDGET butuh konfigurasi eksplisit.
RISK_SIZING_MODE_REQUESTED = (os.getenv("RISK_SIZING_MODE") or "LEGACY_NOTIONAL").strip().upper()
RISK_PER_TRADE_PCT = _cfg_opt_float("RISK_PER_TRADE_PCT", 0.01, 10.0)
MAX_PORTFOLIO_RISK_PCT = _cfg_opt_float("MAX_PORTFOLIO_RISK_PCT", 0.01, 50.0)
MAX_DIRECTION_RISK_PCT = _cfg_opt_float("MAX_DIRECTION_RISK_PCT", 0.01, 50.0)
MAX_DAILY_DRAWDOWN_PCT = _cfg_opt_float("MAX_DAILY_DRAWDOWN_PCT", 0.1, 50.0)
PAPER_EQUITY_USDT = _cfg_opt_float("PAPER_EQUITY_USDT", 1.0, 1e9)
if RISK_SIZING_MODE_REQUESTED not in {"LEGACY_NOTIONAL", "RISK_BUDGET"}:
    rlog.error("RISK_SIZING_MODE=%r tidak dikenal; memakai LEGACY_NOTIONAL.", RISK_SIZING_MODE_REQUESTED)
    RISK_SIZING_MODE = "LEGACY_NOTIONAL"
elif RISK_SIZING_MODE_REQUESTED == "RISK_BUDGET" and (RISK_PER_TRADE_PCT is None or MAX_PORTFOLIO_RISK_PCT is None):
    rlog.error("RISK_BUDGET butuh RISK_PER_TRADE_PCT dan MAX_PORTFOLIO_RISK_PCT eksplisit; kembali ke LEGACY_NOTIONAL.")
    RISK_SIZING_MODE = "LEGACY_NOTIONAL"
else:
    RISK_SIZING_MODE = RISK_SIZING_MODE_REQUESTED

CANDIDATE_STATES = (
    "DATA_UNAVAILABLE", "REJECTED_HARD", "REJECTED_QUALITY", "WAITING_PRICE", "WAITING_TRIGGER",
    "READY", "PENDING", "FILLED", "CLOSED", "EXPIRED", "SUPERSEDED",
    "REPLAY_PENDING", "REPLAY_COMPLETE", "REPLAY_AMBIGUOUS", "REPLAY_UNAVAILABLE",
)
ACTIVE_CANDIDATE_STATES = {"WAITING_PRICE", "WAITING_TRIGGER", "READY", "PENDING", "FILLED"}
WAITING_CANDIDATE_STATES = {"WAITING_PRICE", "WAITING_TRIGGER"}
CANDIDATE_SCHEMA_VERSION = 1
REPLAY_ENGINE_VERSION = "1.0"


class ErrorCategory:
    RATE_LIMIT = "RATE_LIMIT"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    DATA_STALE = "DATA_STALE"
    DATA_INVALID = "DATA_INVALID"
    STRATEGY_CONTRACT_ERROR = "STRATEGY_CONTRACT_ERROR"
    VALIDATION_REJECT = "VALIDATION_REJECT"
    RISK_REJECT = "RISK_REJECT"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"
    EXECUTION_UNKNOWN = "EXECUTION_UNKNOWN"
    PROTECTION_ERROR = "PROTECTION_ERROR"
    RESEARCH_ERROR = "RESEARCH_ERROR"


def classify_error(exc: BaseException) -> str:
    """Peta exception ke kategori error blueprint (bukan satu catch-all)."""
    name = type(exc).__name__
    text = str(exc).lower()
    if name in {"BinanceRateLimitError", "BybitRateLimitError"} or "rate limit" in text or "429" in text or "418" in text:
        return ErrorCategory.RATE_LIMIT
    if name == "InsufficientHistoryError" or "candle closed" in text or "diperlukan" in text:
        return ErrorCategory.DATA_INSUFFICIENT
    if name == "DataInvalidError":
        return ErrorCategory.DATA_INVALID
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timed out" in text or "timeout" in text:
        return ErrorCategory.NETWORK_TIMEOUT
    if name in {"InsufficientBalanceError", "MarginInfluenceError", "RiskRejectError"}:
        return ErrorCategory.RISK_REJECT
    if isinstance(exc, (KeyError, TypeError, ValueError, AttributeError)):
        return ErrorCategory.STRATEGY_CONTRACT_ERROR
    return ErrorCategory.STRATEGY_CONTRACT_ERROR if "strategy" in text else ErrorCategory.DATA_INVALID


class RiskRejectError(RuntimeError):
    """Entry ditolak oleh risk budget / batas portofolio (bukan alasan ban)."""


def _r_now() -> datetime:
    return datetime.now(timezone.utc)


def _r_iso(dt: datetime | None = None) -> str:
    return (dt or _r_now()).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _r_parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None


# ------------------------------------------------------------ models

@dataclass
class CandidateRecord:
    candidate_id: str
    pair: str
    state: str
    schema_version: int = CANDIDATE_SCHEMA_VERSION
    scan_cycle_id: str = ""
    created_at_utc: str = ""
    updated_at_utc: str = ""
    analysis_source: str = "BYBIT_PUBLIC"
    execution_source: str = "BINANCE_USDM"
    strategy_name: str = ""
    strategy_version: str = ""
    direction: str = ""
    btc_h4_regime: str = ""
    pair_h4_regime: str = ""
    btc_alignment: str = ""
    setup_archetype: str = ""
    confidence: float = 0.0
    confidence_components: dict[str, Any] = field(default_factory=dict)
    entry_zone: dict[str, Any] = field(default_factory=dict)
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    price_exp: float = 0.0
    planned_rr_gross: float | None = None
    planned_rr_after_costs: float | None = None
    data_asof: dict[str, Any] = field(default_factory=dict)
    gate_reasons: list[Any] = field(default_factory=list)
    state_reason: str = ""
    price_vs_zone: str = ""
    next_evaluate_after_utc: str | None = None
    expires_at_utc: str | None = None
    replay_eligible: bool = False
    replay_sampling_reason: str | None = None
    linked_trade_id: str | None = None
    priority: float = 0.0
    thesis_signature: str = ""
    decision_ms: int = 0
    retries: int = 0
    fill_state: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRecord":
        names = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class CandidateEvent:
    candidate_id: str
    event_type: str
    new_state: str
    old_state: str = ""
    reason_code: str = ""
    strategy_version: str = ""
    scan_cycle_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=_r_iso)
    event_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class ReplayJob:
    job_id: str
    candidate_id: str
    status: str = "PENDING"
    source: str = "BINANCE_USDM"
    from_ms: int = 0
    to_ms: int = 0
    cursor_ms: int = 0
    attempts: int = 0
    last_error: str | None = None


@dataclass
class ResourceBudgetSnapshot:
    rss_mb: float = 0.0
    limit_mb: float = 0.0
    pressure_pct: float = 0.0
    level: str = "NORMAL"
    api_headroom_pct: float | None = None
    research_state: str = "OFF"
    research_queue: int = 0
    cache: dict[str, Any] = field(default_factory=dict)
    scan_last_seconds: float | None = None
    scan_avg_seconds: float | None = None


# ------------------------------------------------------------ storage (SQLite)

_SCHEMA_V1 = (
    "CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at_utc TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS candidates(
        candidate_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL, pair TEXT NOT NULL, state TEXT NOT NULL, strategy_version TEXT,
        payload_json TEXT NOT NULL, next_evaluate_after_utc TEXT, expires_at_utc TEXT,
        record_type TEXT NOT NULL DEFAULT 'CANDIDATE')""",
    "CREATE INDEX IF NOT EXISTS idx_cand_state ON candidates(state)",
    "CREATE INDEX IF NOT EXISTS idx_cand_pair ON candidates(pair)",
    "CREATE INDEX IF NOT EXISTS idx_cand_created ON candidates(created_at_utc)",
    "CREATE INDEX IF NOT EXISTS idx_cand_next ON candidates(next_evaluate_after_utc)",
    """CREATE TABLE IF NOT EXISTS candidate_events(
        event_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
        timestamp_utc TEXT NOT NULL, event_type TEXT NOT NULL, reason_code TEXT, payload_json TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_evt_cand ON candidate_events(candidate_id)",
    """CREATE TABLE IF NOT EXISTS replay_jobs(
        job_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
        status TEXT NOT NULL, source TEXT NOT NULL, from_ms INTEGER NOT NULL, to_ms INTEGER NOT NULL,
        cursor_ms INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
        updated_at_utc TEXT NOT NULL, state_json TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_job_status ON replay_jobs(status)",
    """CREATE TABLE IF NOT EXISTS replay_results(
        replay_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
        strategy_version TEXT, replay_engine_version TEXT NOT NULL, outcome TEXT NOT NULL,
        realized_r REAL, gross_pnl REAL, net_pnl REAL, metrics_json TEXT NOT NULL, created_at_utc TEXT NOT NULL,
        parameter_set_id TEXT NOT NULL DEFAULT 'baseline', record_type TEXT NOT NULL DEFAULT 'REPLAY',
        UNIQUE(candidate_id, replay_engine_version, parameter_set_id))""",
    """CREATE TABLE IF NOT EXISTS candle_cache_meta(
        source TEXT NOT NULL, pair TEXT NOT NULL, timeframe TEXT NOT NULL, first_ms INTEGER, last_ms INTEGER,
        row_count INTEGER, path_or_table TEXT, updated_at_utc TEXT NOT NULL, PRIMARY KEY(source, pair, timeframe))""",
    """CREATE TABLE IF NOT EXISTS research_runs(
        run_id TEXT PRIMARY KEY, parameter_set_id TEXT, data_window TEXT, metrics_json TEXT NOT NULL,
        created_at_utc TEXT NOT NULL)""",
)


class ResearchStore:
    """SQLite stdlib; satu writer, transaksi singkat, tanpa network I/O di dalam transaksi."""

    def __init__(self, data_dir: str) -> None:
        self.root = Path(data_dir)
        self.path = self.root / "research.sqlite3"
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.ok = False
        self.error: str | None = None
        self.wal = False

    def open(self) -> bool:
        with self._lock:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                probe = self.root / ".write_probe"
                probe.write_text("ok")
                probe.unlink()
                conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None, timeout=5.0)
                conn.execute("PRAGMA foreign_keys=ON")
                try:
                    row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                    self.wal = bool(row and str(row[0]).lower() == "wal")
                except sqlite3.DatabaseError:
                    self.wal = False
                self._conn = conn
                self._migrate()
                self.ok = True
                self.error = None
            except Exception as exc:  # noqa: BLE001
                self.ok = False
                self.error = f"{type(exc).__name__}: {exc}"
                self._conn = None
                rlog.error("ResearchStore tidak dapat dibuka: %s", self.error)
            return self.ok

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
                    self.ok = False

    _IDENTITY = {
        "schema_migrations": {"version"}, "candidates": {"candidate_id"}, "candidate_events": {"event_id", "candidate_id"},
        "replay_jobs": {"job_id", "candidate_id"}, "replay_results": {"replay_id", "candidate_id"},
        "candle_cache_meta": {"source", "pair", "timeframe"}, "research_runs": {"run_id"},
    }
    _EXPECTED = {
        "candidates": {"schema_version": "INTEGER NOT NULL DEFAULT 1", "created_at_utc": "TEXT NOT NULL DEFAULT ''", "updated_at_utc": "TEXT NOT NULL DEFAULT ''",
                       "pair": "TEXT NOT NULL DEFAULT ''", "state": "TEXT NOT NULL DEFAULT ''", "strategy_version": "TEXT", "payload_json": "TEXT NOT NULL DEFAULT '{}'",
                       "next_evaluate_after_utc": "TEXT", "expires_at_utc": "TEXT", "record_type": "TEXT NOT NULL DEFAULT 'CANDIDATE'"},
        "candidate_events": {"timestamp_utc": "TEXT NOT NULL DEFAULT ''", "event_type": "TEXT NOT NULL DEFAULT ''", "reason_code": "TEXT", "payload_json": "TEXT NOT NULL DEFAULT '{}'"},
        "replay_jobs": {"status": "TEXT NOT NULL DEFAULT 'PENDING'", "source": "TEXT NOT NULL DEFAULT 'BINANCE_USDM'", "from_ms": "INTEGER NOT NULL DEFAULT 0",
                        "to_ms": "INTEGER NOT NULL DEFAULT 0", "cursor_ms": "INTEGER NOT NULL DEFAULT 0", "attempts": "INTEGER NOT NULL DEFAULT 0", "last_error": "TEXT",
                        "updated_at_utc": "TEXT NOT NULL DEFAULT ''", "state_json": "TEXT"},
        "replay_results": {"strategy_version": "TEXT", "replay_engine_version": "TEXT NOT NULL DEFAULT ''", "outcome": "TEXT NOT NULL DEFAULT ''", "realized_r": "REAL",
                           "gross_pnl": "REAL", "net_pnl": "REAL", "metrics_json": "TEXT NOT NULL DEFAULT '{}'", "created_at_utc": "TEXT NOT NULL DEFAULT ''",
                           "parameter_set_id": "TEXT NOT NULL DEFAULT 'baseline'", "record_type": "TEXT NOT NULL DEFAULT 'REPLAY'"},
        "candle_cache_meta": {"first_ms": "INTEGER", "last_ms": "INTEGER", "row_count": "INTEGER", "path_or_table": "TEXT", "updated_at_utc": "TEXT NOT NULL DEFAULT ''"},
        "research_runs": {"parameter_set_id": "TEXT", "data_window": "TEXT", "metrics_json": "TEXT NOT NULL DEFAULT '{}'", "created_at_utc": "TEXT NOT NULL DEFAULT ''"},
    }

    def _columns(self, table: str) -> set[str]:
        assert self._conn is not None
        return {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}

    def _migrate(self) -> None:
        """Idempoten & self-healing: buat tabel, tambah kolom hilang, simpan tabel lama yang tidak kompatibel."""
        assert self._conn is not None
        conn = self._conn
        tables = [s for s in _SCHEMA_V1 if not s.lstrip().upper().startswith("CREATE INDEX")]
        indexes = [s for s in _SCHEMA_V1 if s.lstrip().upper().startswith("CREATE INDEX")]
        conn.execute("PRAGMA legacy_alter_table=ON")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table, ident in self._IDENTITY.items():
                if self._table_exists(table) and not ident <= self._columns(table):
                    legacy = f"{table}_legacy_{int(time.time())}"
                    conn.execute(f"ALTER TABLE {table} RENAME TO {legacy}")
                    rlog.warning("Skema %s tidak kompatibel; tabel lama disimpan sebagai %s", table, legacy)
            for stmt in tables:
                conn.execute(stmt)
            for table, cols in self._EXPECTED.items():
                have = self._columns(table)
                for name, decl in cols.items():
                    if name not in have:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                        rlog.warning("Kolom %s.%s ditambahkan (self-heal skema).", table, name)
            for stmt in indexes:
                conn.execute(stmt)
            conn.execute("INSERT OR IGNORE INTO schema_migrations(version, applied_at_utc) VALUES (1, ?)", (_r_iso(),))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("PRAGMA legacy_alter_table=OFF")

    def _table_exists(self, name: str) -> bool:
        assert self._conn is not None
        return self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    def _tx(self, fn):  # noqa: ANN001
        with self._lock:
            if self._conn is None:
                raise RuntimeError("ResearchStore tertutup")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                out = fn(self._conn)
                self._conn.execute("COMMIT")
                return out
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _q(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("ResearchStore tertutup")
            return list(self._conn.execute(sql, params).fetchall())

    # -- candidates & events (idempotent)
    @staticmethod
    def _upsert_candidate_sql(conn: sqlite3.Connection, rec: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO candidates(candidate_id, schema_version, created_at_utc, updated_at_utc, pair, state,
                   strategy_version, payload_json, next_evaluate_after_utc, expires_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(candidate_id) DO UPDATE SET updated_at_utc=excluded.updated_at_utc, state=excluded.state,
                   payload_json=excluded.payload_json, next_evaluate_after_utc=excluded.next_evaluate_after_utc,
                   expires_at_utc=excluded.expires_at_utc""",
            (rec["candidate_id"], int(rec.get("schema_version", 1)), rec.get("created_at_utc") or _r_iso(),
             rec.get("updated_at_utc") or _r_iso(), rec["pair"], rec["state"], rec.get("strategy_version"),
             json.dumps(rec, ensure_ascii=False, default=str), rec.get("next_evaluate_after_utc"), rec.get("expires_at_utc")),
        )

    def write_batch(self, candidates: list[dict[str, Any]], events: list[CandidateEvent]) -> int:
        def run(conn: sqlite3.Connection) -> int:
            for rec in candidates:
                self._upsert_candidate_sql(conn, rec)
            n = 0
            for ev in events:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO candidate_events(event_id, candidate_id, timestamp_utc, event_type, reason_code, payload_json) VALUES (?,?,?,?,?,?)",
                    (ev.event_id, ev.candidate_id, ev.timestamp_utc, ev.event_type, ev.reason_code,
                     json.dumps(asdict(ev), ensure_ascii=False, default=str)),
                )
                n += cur.rowcount
            return n
        return self._tx(run)

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        rows = self._q("SELECT payload_json FROM candidates WHERE candidate_id=?", (candidate_id,))
        return json.loads(rows[0][0]) if rows else None

    def counts_by_state(self) -> dict[str, int]:
        return {r[0]: r[1] for r in self._q("SELECT state, COUNT(*) FROM candidates GROUP BY state")}

    def list_candidates(self, states: tuple[str, ...] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if states:
            marks = ",".join("?" for _ in states)
            rows = self._q(f"SELECT payload_json FROM candidates WHERE state IN ({marks}) ORDER BY updated_at_utc DESC LIMIT ?", (*states, limit))
        else:
            rows = self._q("SELECT payload_json FROM candidates ORDER BY updated_at_utc DESC LIMIT ?", (limit,))
        return [json.loads(r[0]) for r in rows]

    def event_count(self) -> int:
        return int(self._q("SELECT COUNT(*) FROM candidate_events")[0][0])

    def load_active(self, limit: int) -> list[dict[str, Any]]:
        marks = ",".join("?" for _ in ACTIVE_CANDIDATE_STATES)
        rows = self._q(f"SELECT payload_json FROM candidates WHERE state IN ({marks}) ORDER BY updated_at_utc DESC LIMIT ?", (*sorted(ACTIVE_CANDIDATE_STATES), limit))
        return [json.loads(r[0]) for r in rows]

    # -- replay jobs/results
    def create_replay_job(self, job: ReplayJob) -> bool:
        def run(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "INSERT OR IGNORE INTO replay_jobs(job_id, candidate_id, status, source, from_ms, to_ms, cursor_ms, attempts, updated_at_utc) VALUES (?,?,?,?,?,?,?,?,?)",
                (job.job_id, job.candidate_id, job.status, job.source, job.from_ms, job.to_ms, job.cursor_ms or job.from_ms, 0, _r_iso()),
            )
            return cur.rowcount > 0
        return self._tx(run)

    def claim_job(self, max_attempts: int = 3) -> dict[str, Any] | None:
        def run(conn: sqlite3.Connection) -> dict[str, Any] | None:
            row = conn.execute(
                "SELECT job_id, candidate_id, from_ms, to_ms, cursor_ms, attempts, state_json FROM replay_jobs WHERE status='PENDING' AND attempts<? ORDER BY updated_at_utc LIMIT 1",
                (max_attempts,),
            ).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE replay_jobs SET status='RUNNING', updated_at_utc=? WHERE job_id=?", (_r_iso(), row[0]))
            return {"job_id": row[0], "candidate_id": row[1], "from_ms": row[2], "to_ms": row[3], "cursor_ms": row[4], "attempts": row[5], "state_json": row[6]}
        return self._tx(run)

    def update_job(self, job_id: str, *, status: str | None = None, cursor_ms: int | None = None, last_error: str | None = None, state_json: str | None = None) -> None:
        def run(conn: sqlite3.Connection) -> None:
            sets, params = ["updated_at_utc=?"], [_r_iso()]
            for col, val in (("status", status), ("cursor_ms", cursor_ms), ("last_error", last_error), ("state_json", state_json)):
                if val is not None:
                    sets.append(f"{col}=?")
                    params.append(val)
            conn.execute(f"UPDATE replay_jobs SET {', '.join(sets)} WHERE job_id=?", (*params, job_id))
        self._tx(run)

    def fail_job(self, job_id: str, error: str, max_attempts: int = 3) -> str:
        def run(conn: sqlite3.Connection) -> str:
            row = conn.execute("SELECT attempts FROM replay_jobs WHERE job_id=?", (job_id,)).fetchone()
            attempts = (row[0] if row else 0) + 1
            status = "FAILED" if attempts >= max_attempts else "PENDING"
            conn.execute("UPDATE replay_jobs SET status=?, attempts=?, last_error=?, updated_at_utc=? WHERE job_id=?",
                         (status, attempts, error[:300], _r_iso(), job_id))
            return status
        return self._tx(run)

    def reset_running_jobs(self) -> int:
        def run(conn: sqlite3.Connection) -> int:
            return conn.execute("UPDATE replay_jobs SET status='PENDING', updated_at_utc=? WHERE status='RUNNING'", (_r_iso(),)).rowcount
        return self._tx(run)

    def job_counts(self) -> dict[str, int]:
        return {r[0]: r[1] for r in self._q("SELECT status, COUNT(*) FROM replay_jobs GROUP BY status")}

    def pending_job_ranges(self) -> list[tuple[str, int, int]]:
        rows = self._q("SELECT c.pair, j.cursor_ms, j.to_ms FROM replay_jobs j JOIN candidates c ON c.candidate_id=j.candidate_id WHERE j.status IN ('PENDING','RUNNING')")
        return [(r[0], int(r[1]), int(r[2])) for r in rows]

    def save_replay_result(self, candidate_id: str, strategy_version: str, outcome: str, metrics: dict[str, Any], *, parameter_set_id: str = "baseline") -> bool:
        replay_id = hashlib.sha1(f"{candidate_id}|{REPLAY_ENGINE_VERSION}|{parameter_set_id}".encode()).hexdigest()[:24]

        def run(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                """INSERT OR IGNORE INTO replay_results(replay_id, candidate_id, strategy_version, replay_engine_version, outcome,
                       realized_r, gross_pnl, net_pnl, metrics_json, created_at_utc, parameter_set_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (replay_id, candidate_id, strategy_version, REPLAY_ENGINE_VERSION, outcome, metrics.get("net_r"),
                 metrics.get("gross_r"), metrics.get("net_r"), json.dumps(metrics, ensure_ascii=False, default=str), _r_iso(), parameter_set_id),
            )
            return cur.rowcount > 0
        return self._tx(run)

    def replay_rows(self, limit: int = 5000) -> list[dict[str, Any]]:
        rows = self._q(
            """SELECT r.candidate_id, r.outcome, r.metrics_json, c.payload_json, c.created_at_utc
               FROM replay_results r JOIN candidates c ON c.candidate_id=r.candidate_id
               ORDER BY c.created_at_utc LIMIT ?""", (limit,))
        out = []
        for cid, outcome, mj, pj, created in rows:
            out.append({"candidate_id": cid, "outcome": outcome, "metrics": json.loads(mj), "candidate": json.loads(pj), "created_at_utc": created})
        return out

    def replay_counts(self) -> dict[str, int]:
        return {r[0]: r[1] for r in self._q("SELECT outcome, COUNT(*) FROM replay_results GROUP BY outcome")}

    # -- candle meta & runs
    def upsert_candle_meta(self, source: str, pair: str, timeframe: str, first_ms: int, last_ms: int, row_count: int, path: str) -> None:
        def run(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO candle_cache_meta(source, pair, timeframe, first_ms, last_ms, row_count, path_or_table, updated_at_utc)
                   VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(source, pair, timeframe) DO UPDATE SET
                   first_ms=MIN(first_ms, excluded.first_ms), last_ms=MAX(last_ms, excluded.last_ms),
                   row_count=excluded.row_count, path_or_table=excluded.path_or_table, updated_at_utc=excluded.updated_at_utc""",
                (source, pair, timeframe, first_ms, last_ms, row_count, path, _r_iso()),
            )
        self._tx(run)

    def candle_meta(self, source: str, pair: str, timeframe: str = "1m") -> dict[str, Any] | None:
        rows = self._q("SELECT first_ms, last_ms, row_count FROM candle_cache_meta WHERE source=? AND pair=? AND timeframe=?", (source, pair, timeframe))
        return {"first_ms": rows[0][0], "last_ms": rows[0][1], "row_count": rows[0][2]} if rows else None

    def save_run(self, run_id: str, parameter_set_id: str, data_window: str, metrics: dict[str, Any]) -> None:
        def run(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO research_runs(run_id, parameter_set_id, data_window, metrics_json, created_at_utc) VALUES (?,?,?,?,?)",
                (run_id, parameter_set_id, data_window, json.dumps(metrics, ensure_ascii=False, default=str), _r_iso()),
            )
        self._tx(run)

    def stats(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "path": str(self.path), "wal": self.wal, "error": self.error,
            "candidates": sum(self.counts_by_state().values()) if self.ok else 0,
            "events": self.event_count() if self.ok else 0,
            "jobs": self.job_counts() if self.ok else {},
        }


# ------------------------------------------------------------ M1 candle files (bukan GitHub)

class CandleFileStore:
    """candles/<PAIR>/<YYYYMMDD>.csv.gz — baca per hari agar RAM tetap kecil."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "candles"

    def _path(self, pair: str, day: str) -> Path:
        return self.root / pair.upper() / f"{day}.csv.gz"

    @staticmethod
    def _day(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y%m%d")

    def _read_day(self, path: Path) -> dict[int, tuple]:
        rows: dict[int, tuple] = {}
        if not path.exists():
            return rows
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    rows[int(parts[0])] = (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        return rows

    def write(self, pair: str, rows: list[tuple]) -> tuple[int, int, int]:
        by_day: dict[str, list[tuple]] = {}
        for row in rows:
            by_day.setdefault(self._day(row[0]), []).append(row)
        first = last = None
        for day, chunk in by_day.items():
            path = self._path(pair, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged = self._read_day(path)
            for row in chunk:
                merged[row[0]] = tuple(row)
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                for key in sorted(merged):
                    r = merged[key]
                    fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
            os.replace(tmp, path)
            keys = sorted(merged)
            first = keys[0] if first is None else min(first, keys[0])
            last = keys[-1] if last is None else max(last, keys[-1])
        return (first or 0, last or 0, len(rows))

    def read_range(self, pair: str, from_ms: int, to_ms: int):
        day = datetime.fromtimestamp(from_ms / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        end = datetime.fromtimestamp(to_ms / 1000, tz=timezone.utc)
        while day <= end:
            rows = self._read_day(self._path(pair, day.strftime("%Y%m%d")))
            for key in sorted(rows):
                if from_ms <= key <= to_ms:
                    yield rows[key]
            day += timedelta(days=1)

    def disk_usage_bytes(self) -> int:
        total = 0
        if self.root.exists():
            for p in self.root.rglob("*.csv.gz"):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def prune(self, max_bytes: int, protected: list[tuple[str, int, int]]) -> int:
        """Hapus hari tertua yang tidak dirujuk replay/audit sampai di bawah kuota."""
        if self.disk_usage_bytes() <= max_bytes or not self.root.exists():
            return 0
        files = sorted(self.root.rglob("*.csv.gz"), key=lambda p: p.name)
        removed = 0
        for path in files:
            if self.disk_usage_bytes() <= max_bytes:
                break
            pair = path.parent.name
            day = path.name[:8]
            try:
                day_start = int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError:
                continue
            day_end = day_start + 86_400_000 - 1
            if any(p == pair and a <= day_end and b >= day_start for p, a, b in protected):
                continue
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


# ------------------------------------------------------------ replay M1 deterministik

TRAIL_LADDER_REPLAY = tuple(TRAIL_R_LADDER)


class ReplaySim:
    """Replay satu kandidat candle-per-candle (semantik eksplisit; ambigu dilabeli konservatif)."""

    def __init__(self, cand: dict[str, Any], *, trailing: bool, fee_pct: float, slippage_pct: float,
                 expiry_ms: int | None = None, trail_fee_buffer_pct: float = 0.12) -> None:
        self.buy = str(cand.get("direction")).upper() == "BUY"
        self.entry = float(cand["entry"])
        self.sl0 = float(cand["sl"])
        self.tp = float(cand["tp"])
        self.price_exp = float(cand.get("price_exp") or 0.0)
        self.decision_ms = int(cand.get("decision_ms") or 0)
        self.expiry_ms = expiry_ms
        self.trailing = trailing
        self.cost_rate = (fee_pct + slippage_pct) / 100.0
        self.trail_fee_buffer_pct = trail_fee_buffer_pct
        self.risk = abs(self.entry - self.sl0)
        self.phase = "PENDING"
        self.sl = self.sl0
        self.entry_ms: int | None = None
        self.exit_ms: int | None = None
        self.exit_price: float | None = None
        self.outcome: str | None = None
        self.mfe = 0.0
        self.mae = 0.0
        self.ambiguity: list[str] = []
        self.optimistic_exit: float | None = None
        self.last_ms = 0
        self.gap_minutes = 0
        self.candles = 0
        self._pending_sl: float | None = None

    @property
    def done(self) -> bool:
        return self.phase == "DONE"

    def _finish(self, outcome: str, t: int, price: float | None) -> None:
        self.phase = "DONE"
        self.outcome = outcome
        self.exit_ms = t
        self.exit_price = price

    def _touch_sl(self, low: float, high: float, sl: float) -> bool:
        return low <= sl if self.buy else high >= sl

    def _touch_tp(self, low: float, high: float) -> bool:
        return high >= self.tp if self.buy else low <= self.tp

    def feed(self, candle: tuple) -> bool:
        t, o, h, l, c = int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4])
        if self.done or t < self.decision_ms:
            return self.done
        if self.last_ms and t > self.last_ms + 60_000:
            self.gap_minutes += int((t - self.last_ms) // 60_000) - 1
        self.last_ms = t
        self.candles += 1
        if self.phase == "PENDING":
            if self.expiry_ms is not None and t >= self.expiry_ms:
                self._finish("EXPIRED", t, None)
                return True
            can_fill = (l <= self.entry) if self.buy else (h >= self.entry)
            exp_touched = self.price_exp > 0 and ((h >= self.price_exp) if self.buy else (l <= self.price_exp))
            if not can_fill:
                if exp_touched:
                    self._finish("EXPIRED", t, None)
                    return True
                return False
            self.phase = "OPEN"
            self.entry_ms = t
            if exp_touched:
                self.ambiguity.append("fill_and_price_exp_same_candle")
            stop_hit = self._touch_sl(l, h, self.sl)
            tp_hit = self._touch_tp(l, h)
            if stop_hit:
                self.ambiguity.append("entry_and_stop_same_candle")
                if tp_hit:
                    self.ambiguity.append("entry_stop_tp_same_candle")
                    self.optimistic_exit = self.tp
                self._finish("SL", t, self.sl)
                return True
            if tp_hit:
                self.ambiguity.append("entry_and_tp_same_candle_no_credit")
            return False
        # OPEN (candle setelah fill)
        if self._pending_sl is not None:
            self.sl = self._pending_sl
            self._pending_sl = None
        stop_hit = self._touch_sl(l, h, self.sl)
        tp_hit = self._touch_tp(l, h)
        if stop_hit and tp_hit:
            self.ambiguity.append("sl_and_tp_same_candle")
            self.optimistic_exit = self.tp
        if stop_hit:
            self._finish("SL" if self.sl == self.sl0 else "TRAIL", t, self.sl)
            return True
        if tp_hit:
            self._finish("TP", t, self.tp)
            return True
        move_fav = (h - self.entry) if self.buy else (self.entry - l)
        move_adv = (self.entry - l) if self.buy else (h - self.entry)
        self.mfe = max(self.mfe, move_fav)
        self.mae = max(self.mae, move_adv)
        if self.trailing and self.risk > 0:
            r_max = self.mfe / self.risk
            lock = None
            for trigger, lock_r in TRAIL_LADDER_REPLAY:
                if r_max >= trigger:
                    lock = lock_r
            if lock is not None:
                gain = max(lock * self.risk, self.entry * self.trail_fee_buffer_pct / 100.0)
                level = self.entry + gain if self.buy else self.entry - gain
                better = level > self.sl if self.buy else level < self.sl
                if better:
                    self._pending_sl = level  # berlaku mulai candle berikutnya
        return False

    def finalize(self, horizon_reached: bool, last_close: float | None, last_ms: int | None) -> bool:
        """Panggil saat data horizon habis. Return True bila hasil final."""
        if self.done:
            return True
        if not horizon_reached:
            return False
        if self.phase == "PENDING":
            self._finish("NOT_FILLED", last_ms or self.last_ms, None)
        else:
            self.ambiguity.append("open_at_horizon_marked_to_market")
            self._finish("OPEN_AT_HORIZON", last_ms or self.last_ms, last_close)
        return True

    def result(self) -> dict[str, Any]:
        filled = self.entry_ms is not None
        gross_r = net_r = 0.0
        if filled and self.exit_price is not None and self.risk > 0:
            unit = (self.exit_price - self.entry) if self.buy else (self.entry - self.exit_price)
            cost = (self.entry + self.exit_price) * self.cost_rate
            gross_r = unit / self.risk
            net_r = (unit - cost) / self.risk
        optimistic_r = None
        if self.optimistic_exit is not None and filled and self.risk > 0:
            unit = (self.optimistic_exit - self.entry) if self.buy else (self.entry - self.optimistic_exit)
            optimistic_r = (unit - (self.entry + self.optimistic_exit) * self.cost_rate) / self.risk
        return {
            "outcome": self.outcome, "filled": filled, "ambiguous": bool(self.ambiguity), "ambiguity": list(self.ambiguity),
            "entry_ms": self.entry_ms, "exit_ms": self.exit_ms, "entry_price": self.entry if filled else None,
            "exit_price": self.exit_price, "gross_r": round(gross_r, 4), "net_r": round(net_r, 4),
            "optimistic_net_r": None if optimistic_r is None else round(optimistic_r, 4),
            "mfe_price": self.mfe, "mae_price": self.mae,
            "mfe_r": round(self.mfe / self.risk, 4) if self.risk > 0 else None,
            "mae_r": round(self.mae / self.risk, 4) if self.risk > 0 else None,
            "mfe_pct": round(self.mfe / self.entry * 100, 4) if self.entry > 0 else None,
            "mae_pct": round(self.mae / self.entry * 100, 4) if self.entry > 0 else None,
            "time_to_entry_ms": (self.entry_ms - self.decision_ms) if filled else None,
            "time_to_exit_ms": (self.exit_ms - self.entry_ms) if filled and self.exit_ms else None,
            "trailing_model": "R_LADDER_CANDLE_APPROX" if self.trailing else "NONE",
            "gap_minutes": self.gap_minutes, "candles": self.candles,
        }

    def to_state(self) -> dict[str, Any]:
        keys = ("phase", "sl", "entry_ms", "exit_ms", "exit_price", "outcome", "mfe", "mae", "ambiguity",
                "optimistic_exit", "last_ms", "gap_minutes", "candles", "_pending_sl")
        return {k: getattr(self, k) for k in keys}

    def restore(self, state: dict[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)


class ReplayRunner:
    """Dua simulasi paralel: trailing (mirip live) dan fixed SL/TP (untuk varian)."""

    def __init__(self, cand: dict[str, Any], *, fee_pct: float, slippage_pct: float, expiry_ms: int | None) -> None:
        self.cand = cand
        self.sims = {
            "trail": ReplaySim(cand, trailing=True, fee_pct=fee_pct, slippage_pct=slippage_pct, expiry_ms=expiry_ms),
            "fixed": ReplaySim(cand, trailing=False, fee_pct=fee_pct, slippage_pct=slippage_pct, expiry_ms=expiry_ms),
        }

    def feed(self, candle: tuple) -> bool:
        return all([s.feed(candle) for s in self.sims.values()])

    def finalize(self, horizon_reached: bool, last_close: float | None, last_ms: int | None) -> bool:
        return all([s.finalize(horizon_reached, last_close, last_ms) for s in self.sims.values()])

    @property
    def done(self) -> bool:
        return all(s.done for s in self.sims.values())

    def result(self) -> dict[str, Any]:
        head = self.sims["trail"].result()
        fixed = self.sims["fixed"].result()
        head["net_r_fixed"] = fixed["net_r"]
        head["outcome_fixed"] = fixed["outcome"]
        head["replay_engine_version"] = REPLAY_ENGINE_VERSION
        head["fee_pct_per_side"] = round(self.sims["trail"].cost_rate * 100.0 - EST_SLIPPAGE_PCT_PER_SIDE, 6)
        head["slippage_pct_per_side"] = EST_SLIPPAGE_PCT_PER_SIDE
        head["funding_modeled"] = False
        head["data_source"] = "BINANCE_USDM_M1"
        head["limitations"] = ["structural/RSI trailing tidak direplikasi", "funding tidak dimodelkan", "urutan intra-candle M1 tidak diketahui"]
        return head

    def to_state(self) -> dict[str, Any]:
        return {name: sim.to_state() for name, sim in self.sims.items()}

    def restore(self, state: dict[str, Any]) -> None:
        for name, sim_state in state.items():
            if name in self.sims:
                self.sims[name].restore(sim_state)


def replay_candidate(cand: dict[str, Any], candles, *, fee_pct: float, slippage_pct: float, horizon_end_ms: int, expiry_ms: int | None = None) -> dict[str, Any]:
    """Pembungkus sinkron (untuk tes & batch kecil)."""
    runner = ReplayRunner(cand, fee_pct=fee_pct, slippage_pct=slippage_pct, expiry_ms=expiry_ms)
    last_close, last_ms = None, None
    for candle in candles:
        if candle[0] > horizon_end_ms:
            break
        last_close, last_ms = float(candle[4]), int(candle[0])
        if runner.feed(candle):
            break
    runner.finalize(True, last_close, last_ms)
    return runner.result()


# ------------------------------------------------------------ resource / memory pressure

def detect_memory_limit_mb() -> float | None:
    """Batas memori container (cgroup v2/v1) bila terdeteksi."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            if raw and raw != "max":
                value = int(raw) / 1024 / 1024
                if 16 <= value < 1_000_000:
                    return value
        except (OSError, ValueError):
            continue
    return None


class ResourceMonitor:
    LEVELS = ("NORMAL", "WARNING", "HIGH", "CRITICAL")

    def __init__(self, configured_limit_mb: float, *, thresholds: tuple[float, float, float] = (70.0, 80.0, 90.0),
                 hysteresis: float = 5.0, alarm_cooldown: float = 300.0) -> None:
        detected = detect_memory_limit_mb()
        self.detected_limit_mb = detected
        self.limit_mb = detected if detected else max(configured_limit_mb, 64.0)
        self.limit_source = "CGROUP" if detected else "CONFIG"
        self.thresholds = thresholds
        self.hysteresis = hysteresis
        self.alarm_cooldown = alarm_cooldown
        self.level = "NORMAL"
        self.peak_rss_mb = 0.0
        self._last_alarm = 0.0

    def pressure_pct(self, rss_mb: float) -> float:
        return rss_mb / self.limit_mb * 100.0 if self.limit_mb > 0 else 0.0

    def update(self, rss_mb: float, now: float | None = None) -> tuple[str, bool, bool]:
        """Return (level, changed, should_alarm). Hysteresis mencegah bolak-balik."""
        now = time.monotonic() if now is None else now
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
        pct = self.pressure_pct(rss_mb)
        idx = self.LEVELS.index(self.level)
        up = sum(1 for th in self.thresholds if pct >= th)
        if up > idx:
            new_idx = up
        else:
            new_idx = idx
            while new_idx > 0 and pct < self.thresholds[new_idx - 1] - self.hysteresis:
                new_idx -= 1
        changed = new_idx != idx
        self.level = self.LEVELS[new_idx]
        alarm = changed and new_idx > idx and (now - self._last_alarm >= self.alarm_cooldown)
        if alarm:
            self._last_alarm = now
        return self.level, changed, alarm

    def allows_research(self) -> bool:
        return self.level == "NORMAL"

    def allows_deep_analysis(self, near_zone: bool) -> bool:
        return self.level in {"NORMAL", "WARNING"} or (self.level == "HIGH" and near_zone)


# ------------------------------------------------------------ waktu & event-driven recheck

H4_MS_MAIN = 4 * 3600 * 1000
M15_MS_MAIN = 15 * 60 * 1000


def next_boundary_ts(now_ts: float, interval_seconds: int) -> float:
    return (int(now_ts // interval_seconds) + 1) * interval_seconds


def next_evaluate_ts(state: str, now_ts: float, retries: int = 0) -> float:
    """Jadwal evaluasi ulang berbasis event (candle close), bukan ban panjang."""
    if state in WAITING_CANDIDATE_STATES:
        return max(now_ts + 30.0, min(next_boundary_ts(now_ts, 900), now_ts + SCAN_WATCHLIST_INTERVAL))
    if state == "DATA_UNAVAILABLE":
        return now_ts + min(60.0 * (2 ** min(retries, 4)), 900.0)
    if state == "READY":
        return now_ts + SCAN_READY_RETRY_SECONDS
    return max(next_boundary_ts(now_ts, 4 * 3600), now_ts + 30.0)


class RecheckBook:
    """Cooldown sementara per pair (bukan ban). Habis saat candle relevan maju atau jadwal lewat."""

    def __init__(self, max_size: int = 3000) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.max_size = max_size

    def set(self, pair: str, until_ts: float, reason: str, state: str, **extra: Any) -> None:
        self.items[pair] = {"until": float(until_ts), "reason": reason, "state": state, **extra}
        if len(self.items) > self.max_size:
            for key in sorted(self.items, key=lambda k: self.items[k]["until"])[: len(self.items) - self.max_size]:
                self.items.pop(key, None)

    def active(self, pair: str, now_ts: float) -> dict[str, Any] | None:
        item = self.items.get(pair)
        if item is None:
            return None
        if item["until"] <= now_ts:
            self.items.pop(pair, None)
            return None
        return item

    def purge(self, now_ts: float) -> int:
        gone = [p for p, it in self.items.items() if it["until"] <= now_ts]
        for pair in gone:
            self.items.pop(pair, None)
        return len(gone)

    def invalidate_btc_mismatch_pairs(self, btc_trend: str) -> list[str]:
        gone = [p for p, it in self.items.items() if it.get("reason") == "STRUCTURE_MISMATCH" and it.get("btc_trend") != btc_trend]
        for pair in gone:
            self.items.pop(pair, None)
        return gone

    def invalidate_btc_mismatch(self, btc_trend: str) -> int:
        return len(self.invalidate_btc_mismatch_pairs(btc_trend))

    def counts_by_reason(self, now_ts: float) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in self.items.values():
            if it["until"] > now_ts:
                out[it["reason"]] = out.get(it["reason"], 0) + 1
        return out

    def next_due_ts(self) -> float | None:
        now = time.time()
        return min((it["until"] for it in self.items.values() if it["until"] > now), default=None)

    def export(self, now_ts: float, limit: int = 1500) -> dict[str, Any]:
        live = {p: it for p, it in self.items.items() if it["until"] > now_ts}
        keep = sorted(live, key=lambda k: live[k]["until"])[:limit]
        return {p: live[p] for p in keep}

    def load(self, data: dict[str, Any], now_ts: float) -> int:
        n = 0
        for pair, it in (data or {}).items():
            try:
                if float(it["until"]) > now_ts:
                    self.items[str(pair)] = dict(it)
                    n += 1
            except (KeyError, TypeError, ValueError):
                continue
        return n


# ------------------------------------------------------------ candidate builder, priority, ledger

def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def build_candidate_from_setup(pair: str, setup: dict[str, Any], *, cycle_id: str, now: datetime, state_override: str | None = None,
                               reason_override: str | None = None, strategy_name: str = "", ttl_hours: float | None = None) -> CandidateRecord:
    analysis = setup.get("analysis") if isinstance(setup.get("analysis"), dict) else {}
    data = setup.get("data") if isinstance(setup.get("data"), dict) else {}
    strat = setup.get("strategy") if isinstance(setup.get("strategy"), dict) else {}
    signature = str(setup.get("thesis_signature") or "")
    direction = str(setup.get("direction") or "").upper()
    if not signature:
        signature = hashlib.sha1(f"{pair}|{direction}|{setup.get('entry')}|{setup.get('sl')}|{setup.get('tp')}".encode()).hexdigest()[:20]
    now_iso = _r_iso(now)
    expires = now + timedelta(hours=ttl_hours if ttl_hours is not None else CANDIDATE_TTL_HOURS)
    state = state_override or str(setup.get("candidate_state") or "READY")
    reason = reason_override or str(setup.get("state_reason") or "")
    rec = CandidateRecord(
        candidate_id=f"C{signature}", pair=pair, state=state, scan_cycle_id=str(cycle_id),
        created_at_utc=now_iso, updated_at_utc=now_iso,
        analysis_source=str(setup.get("analysis_source") or "BYBIT_PUBLIC"),
        strategy_name=str(strat.get("name") or strategy_name), strategy_version=str(strat.get("version") or ""),
        direction=direction, btc_h4_regime=str(data.get("directional_bias") or ""),
        pair_h4_regime=str(((analysis.get("macro") or {}).get("pair_h4") or {}).get("trend") or ""),
        btc_alignment=str(setup.get("btc_alignment") or ""), setup_archetype=str(setup.get("setup_archetype") or ""),
        confidence=_f(setup.get("confidence")), confidence_components=dict(setup.get("confidence_components") or {}),
        entry_zone=dict(setup.get("entry_zone") or {}), entry=_f(setup.get("entry")), sl=_f(setup.get("sl")),
        tp=_f(setup.get("tp")), price_exp=_f(setup.get("price_exp")),
        planned_rr_gross=setup.get("planned_rr_gross"), planned_rr_after_costs=setup.get("planned_rr_after_costs"),
        data_asof=dict(setup.get("data_asof") or {}), gate_reasons=[g.get("code") if isinstance(g, dict) else g for g in (setup.get("gate_reasons") or [])][:8],
        state_reason=reason, price_vs_zone=str(setup.get("price_vs_zone") or ""),
        expires_at_utc=_r_iso(expires), thesis_signature=signature, decision_ms=int(now.timestamp() * 1000),
    )
    rec.next_evaluate_after_utc = _r_iso(datetime.fromtimestamp(next_evaluate_ts(state, now.timestamp()), tz=timezone.utc))
    return rec


def synthetic_observation(pair: str, state: str, reason: str, *, cycle_id: str, now: datetime, bucket_ms: int = H4_MS_MAIN,
                          strategy_version: str = "", retries: int = 0, btc_h4_regime: str = "", pair_h4_regime: str = "") -> CandidateRecord:
    """Hasil scan tanpa setup (mismatch, data tak tersedia, dsb.) tetap tercatat dan terhitung."""
    bucket = int(now.timestamp() * 1000) // bucket_ms
    cid = "S" + hashlib.sha1(f"{pair}|{state}|{reason}|{bucket}".encode()).hexdigest()[:20]
    now_iso = _r_iso(now)
    rec = CandidateRecord(
        candidate_id=cid, pair=pair, state=state, scan_cycle_id=str(cycle_id), created_at_utc=now_iso, updated_at_utc=now_iso,
        strategy_version=strategy_version, state_reason=reason, btc_h4_regime=btc_h4_regime, pair_h4_regime=pair_h4_regime,
        expires_at_utc=_r_iso(now + timedelta(hours=CANDIDATE_TTL_HOURS)), retries=retries, decision_ms=int(now.timestamp() * 1000),
    )
    rec.next_evaluate_after_utc = _r_iso(datetime.fromtimestamp(next_evaluate_ts(state, now.timestamp(), retries), tz=timezone.utc))
    return rec


def candidate_priority(rec: CandidateRecord, *, now_ts: float | None = None, distance_pct: float | None = None,
                       bucket_expectancy: float | None = None, bucket_n: int = 0) -> float:
    """Prioritas jadwal re-evaluasi/validasi (0-100). BUKAN pengganti confidence."""
    conf = max(0.0, min(rec.confidence, 100.0)) / 100.0
    rr = rec.planned_rr_after_costs if rec.planned_rr_after_costs is not None else (rec.planned_rr_gross or 0.0)
    rr_n = max(0.0, min(_f(rr), 4.0)) / 4.0
    prox = 0.3 if distance_pct is None else max(0.0, 1.0 - min(abs(distance_pct), 5.0) / 5.0)
    zone_bonus = {"IN_ZONE": 1.0, "ABOVE_ZONE": 0.5, "BELOW_ZONE": 0.5}.get(rec.price_vs_zone, 0.3)
    prox = max(prox, zone_bonus) if distance_pct is None else prox
    fresh = 1.0
    if now_ts is not None and rec.data_asof.get("M15"):
        age = max(0.0, now_ts - rec.data_asof["M15"] / 1000.0)
        fresh = max(0.0, 1.0 - age / 3600.0)
    align = {"ALIGNED": 1.0, "NEUTRAL": 0.6, "COUNTER_TREND": 0.2}.get(rec.btc_alignment, 0.4)
    score = 0.35 * conf + 0.20 * rr_n + 0.25 * prox + 0.10 * fresh + 0.10 * align
    if bucket_expectancy is not None and bucket_n >= RESEARCH_MIN_BUCKET_SAMPLE:
        score = 0.9 * score + 0.1 * max(0.0, min(1.0, 0.5 + bucket_expectancy / 4.0))
    return round(score * 100.0, 3)


class CandidateLedger:
    """Queue terbatas + antrean tulis. Persistensi: SQLite bila tersedia, selain itu ekspor ringkas GitHub."""

    def __init__(self, max_size: int = CANDIDATE_QUEUE_MAX, store: ResearchStore | None = None) -> None:
        self.max_size = max_size
        self.store = store
        self.items: dict[str, CandidateRecord] = {}
        self.by_pair: dict[str, str] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._events: list[CandidateEvent] = []
        self.recent_events: deque[CandidateEvent] = deque(maxlen=300)
        self.evictions = 0
        self.persist_errors = 0
        self.dirty_export = False

    # -- observasi & transisi
    def observe(self, rec: CandidateRecord, event_type: str | None = None, metrics: dict[str, Any] | None = None) -> CandidateRecord:
        existing = self.items.get(rec.candidate_id)
        old_state = existing.state if existing else ""
        prev_id = self.by_pair.get(rec.pair)
        if prev_id and prev_id != rec.candidate_id:
            prev = self.items.get(prev_id)
            if prev and prev.state in (WAITING_CANDIDATE_STATES | {"REJECTED_QUALITY", "READY"}) and not prev.linked_trade_id:
                self._transition(prev, "SUPERSEDED", "CANDIDATE_SUPERSEDED", "SUPERSEDED_BY_NEW_THESIS", {"new_candidate": rec.candidate_id})
        if existing is not None:
            rec.created_at_utc = existing.created_at_utc
            rec.linked_trade_id = existing.linked_trade_id or rec.linked_trade_id
            rec.replay_eligible = existing.replay_eligible or rec.replay_eligible
            rec.replay_sampling_reason = existing.replay_sampling_reason or rec.replay_sampling_reason
            if existing.state in {"PENDING", "FILLED", "CLOSED"} and rec.state in {"READY", "WAITING_PRICE", "WAITING_TRIGGER"}:
                rec.state = existing.state  # state trading tidak diturunkan oleh scan ulang
        self.items[rec.candidate_id] = rec
        self.by_pair[rec.pair] = rec.candidate_id
        material = (not existing) or existing.state != rec.state or existing.state_reason != rec.state_reason
        if material:
            etype = event_type or self._event_for(rec.state, bool(existing))
            self._emit(rec, etype, old_state, rec.state_reason, metrics)
        self._pending[rec.candidate_id] = rec.to_dict()
        self.dirty_export = True
        self._evict()
        return rec

    @staticmethod
    def _event_for(state: str, existed: bool) -> str:
        return {
            "WAITING_PRICE": "CANDIDATE_WAITING", "WAITING_TRIGGER": "CANDIDATE_WAITING", "REJECTED_HARD": "CANDIDATE_REJECTED",
            "REJECTED_QUALITY": "CANDIDATE_REJECTED", "DATA_UNAVAILABLE": "CANDIDATE_REJECTED", "EXPIRED": "CANDIDATE_EXPIRED",
            "READY": "CANDIDATE_VALIDATED" if existed else "CANDIDATE_CREATED",
        }.get(state, "CANDIDATE_RECHECKED" if existed else "CANDIDATE_CREATED")

    def _emit(self, rec: CandidateRecord, etype: str, old_state: str, reason: str, metrics: dict[str, Any] | None) -> None:
        ev = CandidateEvent(candidate_id=rec.candidate_id, event_type=etype, new_state=rec.state, old_state=old_state,
                            reason_code=reason, strategy_version=rec.strategy_version, scan_cycle_id=rec.scan_cycle_id,
                            metrics={"confidence": rec.confidence, "rr_after_costs": rec.planned_rr_after_costs, **(metrics or {})})
        self._events.append(ev)
        self.recent_events.append(ev)

    def _transition(self, rec: CandidateRecord, state: str, etype: str, reason: str, metrics: dict[str, Any] | None = None) -> None:
        old = rec.state
        rec.state = state
        rec.state_reason = reason
        rec.updated_at_utc = _r_iso()
        self._emit(rec, etype, old, reason, metrics)
        self._pending[rec.candidate_id] = rec.to_dict()
        self.dirty_export = True

    def link_trade(self, candidate_id: str, trade_id: str, state: str = "PENDING") -> None:
        rec = self.items.get(candidate_id)
        if rec is None:
            return
        rec.linked_trade_id = trade_id
        self._transition(rec, state, "TRADE_LINKED", "TRADE_CREATED", {"trade_id": trade_id})

    def annotate(self, candidate_id: str, etype: str, reason: str, metrics: dict[str, Any] | None = None) -> None:
        """Catat alasan baru tanpa mengubah state (mis. RISK_REJECTED)."""
        rec = self.items.get(candidate_id)
        if rec is None:
            return
        rec.state_reason = reason
        rec.updated_at_utc = _r_iso()
        self._emit(rec, etype, rec.state, reason, metrics)
        self._pending[rec.candidate_id] = rec.to_dict()

    def set_state(self, candidate_id: str, state: str, etype: str, reason: str) -> None:
        rec = self.items.get(candidate_id)
        if rec is not None and rec.state != state:
            self._transition(rec, state, etype, reason)

    def mark_replay(self, candidate_id: str, eligible: bool, reason: str | None) -> None:
        rec = self.items.get(candidate_id)
        if rec is not None and rec.replay_eligible != eligible:
            rec.replay_eligible = eligible
            rec.replay_sampling_reason = reason
            self._pending[rec.candidate_id] = rec.to_dict()

    def expire_due(self, now: datetime) -> int:
        n = 0
        for rec in list(self.items.values()):
            exp = _r_parse_iso(rec.expires_at_utc)
            if exp and exp <= now and rec.state in (WAITING_CANDIDATE_STATES | {"READY", "REJECTED_QUALITY", "DATA_UNAVAILABLE"}) and not rec.linked_trade_id:
                self._transition(rec, "EXPIRED", "CANDIDATE_EXPIRED", "TTL_REACHED")
                n += 1
        return n

    def due_watch(self, now: datetime, limit: int) -> list[CandidateRecord]:
        out = []
        for rec in self.items.values():
            if rec.state in WAITING_CANDIDATE_STATES and not rec.linked_trade_id:
                nxt = _r_parse_iso(rec.next_evaluate_after_utc)
                if nxt is None or nxt <= now:
                    out.append(rec)
        out.sort(key=lambda r: r.priority, reverse=True)
        return out[:limit]

    def next_watch_due_ts(self) -> float | None:
        now = time.time()
        times = [
            t.timestamp() for r in self.items.values()
            if r.state in WAITING_CANDIDATE_STATES and not r.linked_trade_id
            and (t := _r_parse_iso(r.next_evaluate_after_utc)) and t.timestamp() > now
        ]
        return min(times, default=None)

    def postpone(self, rec: CandidateRecord, seconds: float) -> None:
        rec.next_evaluate_after_utc = _r_iso(_r_now() + timedelta(seconds=seconds))
        self._pending[rec.candidate_id] = rec.to_dict()

    def _evict(self) -> None:
        if len(self.items) <= self.max_size:
            return
        order = {"SUPERSEDED": 0, "EXPIRED": 0, "CLOSED": 0, "DATA_UNAVAILABLE": 1, "REJECTED_HARD": 1, "REJECTED_QUALITY": 2, "WAITING_PRICE": 3, "WAITING_TRIGGER": 3}
        evictable = [r for r in self.items.values() if r.state in order and not r.linked_trade_id]
        evictable.sort(key=lambda r: (order[r.state], r.priority))
        for rec in evictable[: len(self.items) - self.max_size]:
            self.items.pop(rec.candidate_id, None)
            if self.by_pair.get(rec.pair) == rec.candidate_id:
                self.by_pair.pop(rec.pair, None)
            self.evictions += 1

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.items.values():
            out[rec.state] = out.get(rec.state, 0) + 1
        return out

    def oldest_active_age_seconds(self, now: datetime) -> float | None:
        ages = [(now - t).total_seconds() for r in self.items.values() if r.state in ACTIVE_CANDIDATE_STATES and (t := _r_parse_iso(r.created_at_utc))]
        return max(ages, default=None)

    def top(self, limit: int = 10, states: set[str] | None = None) -> list[CandidateRecord]:
        pool = [r for r in self.items.values() if states is None or r.state in states]
        return sorted(pool, key=lambda r: r.priority, reverse=True)[:limit]

    # -- persistensi
    async def flush(self) -> bool:
        """Tulis batch ke SQLite (thread). Gagal -> tidak mengklaim tersimpan."""
        if not self._pending and not self._events:
            return True
        if self.store is None or not self.store.ok:
            self._pending.clear()
            self._events.clear()
            return False
        recs, evs = list(self._pending.values()), list(self._events)
        try:
            await asyncio.to_thread(self.store.write_batch, recs, evs)
        except Exception as exc:  # noqa: BLE001
            self.persist_errors += 1
            rlog.error("[%s] flush ledger gagal: %s", ErrorCategory.PERSISTENCE_ERROR, exc)
            return False
        self._pending.clear()
        del self._events[: len(evs)]
        return True

    def export_compact(self, limit: int = 150) -> dict[str, Any]:
        keep = self.top(limit, ACTIVE_CANDIDATE_STATES)
        fields = ("candidate_id", "pair", "state", "direction", "confidence", "entry", "sl", "tp", "price_exp", "state_reason",
                  "btc_alignment", "next_evaluate_after_utc", "expires_at_utc", "created_at_utc", "updated_at_utc", "thesis_signature",
                  "strategy_version", "linked_trade_id", "decision_ms", "planned_rr_after_costs", "entry_zone", "replay_eligible")
        return {"version": 1, "saved_at": _r_iso(), "evictions": self.evictions,
                "candidates": [{k: getattr(r, k) for k in fields} for r in keep]}

    def load_compact(self, data: dict[str, Any], now: datetime) -> int:
        n = 0
        for row in (data or {}).get("candidates", []):
            try:
                rec = CandidateRecord.from_dict(row)
                exp = _r_parse_iso(rec.expires_at_utc)
                if exp and exp > now and rec.state in WAITING_CANDIDATE_STATES:
                    self.items[rec.candidate_id] = rec
                    self.by_pair[rec.pair] = rec.candidate_id
                    n += 1
            except (TypeError, ValueError, KeyError):
                continue
        return n


def replay_eligibility(rec: CandidateRecord, threshold: float) -> tuple[bool, str | None]:
    """Sampling replay yang dapat direproduksi + alasannya (tanpa M1 untuk semua pair)."""
    if not (rec.entry > 0 and rec.sl > 0 and rec.tp > 0 and rec.direction in {"BUY", "SELL"}):
        return False, None
    hard_codes = {"GEOMETRY_INVALID", "PRICE_INVALID", "ZONE_INVERTED", "DATA_STALE"}
    if rec.state in {"REJECTED_HARD", "DATA_UNAVAILABLE"} and (set(rec.gate_reasons) & hard_codes or rec.state == "DATA_UNAVAILABLE"):
        return False, None
    if rec.state in {"READY", "PENDING", "FILLED", "CLOSED"} or rec.linked_trade_id:
        return True, "ACCEPTED"
    if rec.state in (WAITING_CANDIDATE_STATES | {"REJECTED_QUALITY"}) and threshold - RESEARCH_NEAR_THRESHOLD_POINTS <= rec.confidence < threshold:
        return True, "NEAR_THRESHOLD"
    if RESEARCH_SAMPLE_REJECT_RATE > 0:
        bucket = int(hashlib.sha1(rec.candidate_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < RESEARCH_SAMPLE_REJECT_RATE:
            return True, f"SAMPLED_{RESEARCH_SAMPLE_REJECT_RATE:g}"
    return False, None


# ------------------------------------------------------------ risk sizing (opt-in)

def _d(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def compute_risk_quantity(*, meta: Any, direction: str, entry: Decimal, sl: Decimal, equity: Decimal | None, risk_pct: float | None,
                          leverage: int, margin_cap_usdt: Decimal | None = None, available_balance: Decimal | None = None,
                          max_notional: Decimal | None = None, fee_pct: float | None = None, slippage_pct: float | None = None) -> dict[str, Any]:
    """Qty dari rencana loss di SL; tidak pernah menaikkan qty demi minimum exchange."""
    out: dict[str, Any] = {"ok": False, "reason": "", "mode": "RISK_BUDGET"}
    if equity is None or equity <= 0:
        out["reason"] = "EQUITY_UNKNOWN"
        return out
    if risk_pct is None or risk_pct <= 0:
        out["reason"] = "RISK_PCT_NOT_CONFIGURED"
        return out
    entry, sl = _d(entry), _d(sl)
    dist = abs(entry - sl)
    if entry <= 0 or sl <= 0 or dist <= 0:
        out["reason"] = "SL_DISTANCE_INVALID"
        return out
    if str(direction).upper() == "BUY" and not sl < entry or str(direction).upper() == "SELL" and not sl > entry:
        out["reason"] = "SL_WRONG_SIDE"
        return out
    fee = EST_FEE_PCT_PER_SIDE if fee_pct is None else fee_pct
    slip = EST_SLIPPAGE_PCT_PER_SIDE if slippage_pct is None else slippage_pct
    c = Decimal(str((fee + slip) / 100.0))
    loss_per_unit = dist + (entry + sl) * c
    budget = _d(equity) * Decimal(str(risk_pct)) / Decimal(100)
    raw_qty = budget / loss_per_unit
    caps: list[tuple[str, Decimal]] = []
    lev = Decimal(max(int(leverage), 1))
    if margin_cap_usdt is not None and margin_cap_usdt > 0:
        caps.append(("MARGIN_CAP", margin_cap_usdt * lev / entry))
    if available_balance is not None:
        caps.append(("AVAILABLE_BALANCE", available_balance * lev * Decimal("0.95") / entry))
    if max_notional is not None and max_notional > 0:
        caps.append(("MAX_NOTIONAL", max_notional / entry))
    if getattr(meta, "max_qty", 0) and meta.max_qty > 0:
        caps.append(("MAX_QTY", meta.max_qty))
    qty, binding = raw_qty, "RISK_BUDGET"
    for name, cap in caps:
        if cap < qty:
            qty, binding = cap, name
    step = meta.step_size if meta.step_size and meta.step_size > 0 else Decimal("0")
    if step > 0:
        qty = (qty / step).to_integral_value(rounding=ROUND_DOWN) * step
    min_qty = meta.min_qty if meta.min_qty and meta.min_qty > 0 else Decimal("0")
    smallest = max(min_qty, (meta.min_notional / entry) if meta.min_notional and meta.min_notional > 0 else Decimal("0"))
    if step > 0 and smallest > 0:
        smallest = (smallest / step).to_integral_value(rounding=ROUND_CEILING) * step
    out.update({"risk_budget_usdt": budget, "raw_qty": raw_qty, "loss_per_unit": loss_per_unit, "binding_cap": binding,
                "smallest_legal_qty": smallest, "est_cost_per_unit": (entry + sl) * c})
    if smallest > 0 and smallest * loss_per_unit > budget:
        out["reason"] = "MIN_ORDER_EXCEEDS_RISK_BUDGET"
        out["min_order_loss_usdt"] = smallest * loss_per_unit
        return out
    if qty <= 0 or (smallest > 0 and qty < smallest):
        out["reason"] = f"BELOW_MIN_ORDER_AFTER_{binding}"
        return out
    planned = qty * loss_per_unit
    if planned > budget * Decimal("1.000001"):
        out["reason"] = "RISK_BUDGET_BREACH_AFTER_ROUNDING"
        return out
    out.update({"ok": True, "reason": "OK", "qty": qty, "planned_loss_usdt": planned, "notional": qty * entry,
                "margin_required": qty * entry / lev, "risk_pct_of_equity": float(planned / _d(equity) * 100)})
    return out


def portfolio_risk_check(*, active: list[tuple[str, Decimal]], new_direction: str, new_loss: Decimal, equity: Decimal | None,
                         max_portfolio_pct: float | None, max_direction_pct: float | None = None,
                         daily_drawdown_pct: float | None = None, max_daily_dd_pct: float | None = None,
                         protection_ok: bool = True) -> tuple[bool, str, dict[str, Any]]:
    info: dict[str, Any] = {}
    if not protection_ok:
        return False, "PROTECTION_ERROR_BLOCKS_NEW_EXPOSURE", info
    if equity is None or equity <= 0:
        return False, "EQUITY_UNKNOWN", info
    total = sum((loss for _, loss in active), Decimal(0)) + new_loss
    same = sum((loss for d, loss in active if d == new_direction), Decimal(0)) + new_loss
    info.update({"portfolio_risk_pct": float(total / equity * 100), "direction_risk_pct": float(same / equity * 100)})
    if max_portfolio_pct is not None and total > equity * Decimal(str(max_portfolio_pct)) / 100:
        return False, "PORTFOLIO_RISK_CAP", info
    if max_direction_pct is not None and same > equity * Decimal(str(max_direction_pct)) / 100:
        return False, "DIRECTION_RISK_CAP", info
    if max_daily_dd_pct is not None and daily_drawdown_pct is not None and daily_drawdown_pct >= max_daily_dd_pct:
        return False, "DAILY_DRAWDOWN_LIMIT", info
    return True, "OK", info


# ------------------------------------------------------------ statistik R, bucket, sweep, tournament

def r_metrics(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return {"n": 0, "reliable": False, "warning": "tanpa sampel"}
    mean = sum(vals) / n
    srt = sorted(vals)
    median = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v < 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    peak = cum = mdd = 0.0
    for v in vals:
        cum += v
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    half = 1.96 * sd / math.sqrt(n) if n > 1 else None
    reliable = n >= RESEARCH_MIN_BUCKET_SAMPLE
    return {
        "n": n, "mean_r": round(mean, 4), "median_r": round(median, 4), "expectancy_r": round(mean, 4),
        "win_rate": round(len(wins) / n * 100, 2), "avg_win_r": round(gross_win / len(wins), 4) if wins else 0.0,
        "avg_loss_r": round(-gross_loss / len(losses), 4) if losses else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        "max_drawdown_r": round(mdd, 4), "sd_r": round(sd, 4),
        "ci95_r": None if half is None else [round(mean - half, 4), round(mean + half, 4)],
        "reliable": reliable,
        "warning": None if reliable else f"sampel {n} < {RESEARCH_MIN_BUCKET_SAMPLE}: jangan disimpulkan",
    }


def confidence_bucket(conf: float) -> str:
    lo = int(max(50, min(90, (conf // 10) * 10)))
    return f"{lo}-{lo + 9}" if lo < 90 else "90-100"


def bucket_report(rows: list[dict[str, Any]], key: str, r_key: str = "r") -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        k = confidence_bucket(row["confidence"]) if key == "confidence" else str(row.get(key) or "UNKNOWN")
        groups.setdefault(k, []).append(row[r_key])
    return {k: r_metrics(v) for k, v in sorted(groups.items())}


def threshold_sweep(rows: list[dict[str, Any]], thresholds: list[float] | None = None, r_key: str = "r") -> list[dict[str, Any]]:
    """Dari snapshot tersimpan: tanpa fetch ulang market data."""
    out = []
    for th in thresholds or [55, 60, 65, 70, 75, 80, 85]:
        subset = [r[r_key] for r in rows if r["confidence"] >= th]
        out.append({"threshold": th, **r_metrics(subset), "signal_share_pct": round(len(subset) / len(rows) * 100, 1) if rows else 0.0})
    return out


def split_windows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda r: r["created_at"])
    n = len(ordered)
    a, b = int(n * 0.5), int(n * 0.75)
    return {"development": ordered[:a], "validation": ordered[a:b], "out_of_sample": ordered[b:]}


def walk_forward(rows: list[dict[str, Any]], folds: int = 4) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: r["created_at"])
    n = len(ordered)
    if n < folds * 2:
        return []
    size = n // (folds + 1)
    out = []
    for i in range(1, folds + 1):
        test = ordered[i * size:(i + 1) * size]
        out.append({"fold": i, "train_n": i * size, **r_metrics([r["r"] for r in test])})
    return out


def default_variants(baseline_threshold: float) -> list[dict[str, Any]]:
    """Satu perubahan per varian, data & biaya identik."""
    base = {"id": "baseline", "min_conf": baseline_threshold, "min_rr_after_costs": 0.0, "alignment": ("ALIGNED", "NEUTRAL"), "r_key": "r"}
    variants = [base]
    for delta in (-5, 5, 10):
        variants.append({**base, "id": f"threshold{delta:+d}", "min_conf": baseline_threshold + delta})
    for rr in (1.2, 1.5):
        variants.append({**base, "id": f"rr_after_cost>={rr}", "min_rr_after_costs": rr})
    variants.append({**base, "id": "allow_counter_trend", "alignment": None})
    variants.append({**base, "id": "no_trailing", "r_key": "r_fixed"})
    return variants


def evaluate_variants(rows: list[dict[str, Any]], variants: list[dict[str, Any]]) -> dict[str, Any]:
    windows = split_windows(rows)
    report: dict[str, Any] = {"promotion": "TIDAK ADA promosi otomatis; keputusan manual.", "sample_total": len(rows), "variants": []}
    for var in variants:
        def pick(subset: list[dict[str, Any]]) -> list[float]:
            vals = []
            for r in subset:
                if r["confidence"] < var["min_conf"]:
                    continue
                if (r.get("rr_after_costs") or 0.0) < var["min_rr_after_costs"]:
                    continue
                if var["alignment"] is not None and r.get("btc_alignment") not in var["alignment"]:
                    continue
                vals.append(r[var["r_key"]])
            return vals
        report["variants"].append({"id": var["id"], **{name: r_metrics(pick(sub)) for name, sub in windows.items()}})
    report["walk_forward_baseline"] = walk_forward([r for r in rows if r["confidence"] >= variants[0]["min_conf"]])
    return report


def replay_rows_for_stats(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Baris replay -> format statistik; hasil ambigu memakai nilai konservatif (SL-first)."""
    out = []
    for row in raw_rows:
        m, c = row["metrics"], row["candidate"]
        if m.get("outcome") in {"NOT_FILLED", "EXPIRED"} or not m.get("filled"):
            continue
        out.append({"created_at": row["created_at_utc"], "confidence": _f(c.get("confidence")), "direction": c.get("direction"),
                    "btc_alignment": c.get("btc_alignment"), "archetype": c.get("setup_archetype"), "strategy_version": c.get("strategy_version"),
                    "rr_after_costs": _f(c.get("planned_rr_after_costs")), "r": _f(m.get("net_r")), "r_fixed": _f(m.get("net_r_fixed")),
                    "ambiguous": bool(m.get("ambiguous"))})
    return out


class ResearchWorker:
    """Worker prioritas-4: satu job, cooperative, resumable. Tidak memegang kredensial/klien real."""

    def __init__(self, engine: Any) -> None:
        self.e = engine
        self.state = "OFF"
        self.pause_reason: str | None = None
        self.task: asyncio.Task | None = None
        self.current_job: str | None = None
        self.last_error: str | None = None
        self.last_ok_ts: float | None = None
        self.jobs_done = 0
        self.api_calls = 0
        self.api_weight = 0
        self.rate_limited_until = 0.0

    @property
    def enabled(self) -> bool:
        return self.state in {"RUNNING", "PAUSED"}

    def start(self) -> None:
        if self.task is not None and not self.task.done():
            self.state = "RUNNING"
            return
        self.state = "RUNNING"
        self.task = asyncio.create_task(self._run(), name="research-worker")

    async def stop(self) -> None:
        task, self.task = self.task, None
        self.state = "OFF"
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    def pause(self, reason: str) -> None:
        if self.state == "RUNNING":
            self.state = "PAUSED"
        self.pause_reason = reason

    def resume(self) -> None:
        if self.state == "PAUSED":
            self.state = "RUNNING"
        self.pause_reason = None

    def gate(self) -> tuple[bool, str]:
        e = self.e
        if e.research_store is None or not e.research_store.ok:
            return False, "STORE_UNAVAILABLE"
        if not e._protection_healthy():
            return False, "PROTECTION_UNHEALTHY"
        if not e._resource.allows_research():
            return False, f"MEMORY_{e._resource.level}"
        if RESEARCH_MEMORY_BUDGET_MB > 0 and rss_mb() > RESEARCH_MEMORY_BUDGET_MB:
            return False, "MEMORY_BUDGET"
        if time.monotonic() < self.rate_limited_until:
            return False, "RATE_LIMIT_COOLDOWN"
        if getattr(e, "_scan_cycle_active", False):
            return False, "SCAN_ACTIVE"
        if e.real.cooldown_remaining > 0:
            return False, "API_COOLDOWN"
        head = e._api_headroom_pct()
        if head is not None and head < RESEARCH_MIN_API_HEADROOM_PCT:
            return False, "API_HEADROOM_LOW"
        return True, "OK"

    async def _run(self) -> None:
        while True:
            try:
                ok, why = self.gate()
                if self.state == "PAUSED" or not ok:
                    if not ok:
                        self.pause_reason = why
                    await asyncio.sleep(15)
                    continue
                self.pause_reason = None
                job = await asyncio.to_thread(self.e.research_store.claim_job)
                if job is None:
                    await asyncio.sleep(45)
                    continue
                await self._process(job)
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                rlog.error("[%s] worker: %s", ErrorCategory.RESEARCH_ERROR, self.last_error)
                await asyncio.sleep(30)

    async def _download_page(self, pair: str, start_ms: int, end_ms: int) -> int:
        e = self.e
        raw = await e.rest._get("/fapi/v1/klines", {"symbol": pair, "interval": "1m", "startTime": start_ms, "endTime": end_ms, "limit": 1000})
        self.api_calls += 1
        self.api_weight += 5
        now_ms = int(time.time() * 1000)
        rows = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in raw if len(r) >= 7 and int(r[6]) < now_ms]
        if not rows:
            return 0
        first, last, _ = await asyncio.to_thread(e.candle_files.write, pair, rows)
        await asyncio.to_thread(e.research_store.upsert_candle_meta, "BINANCE_USDM", pair, "1m", first, last, (last - first) // 60000 + 1, str(e.candle_files.root / pair))
        quota = int(min(RESEARCH_MAX_DISK_MB, 100.0 if e.research_ephemeral else RESEARCH_MAX_DISK_MB) * 1024 * 1024)
        if e.candle_files.disk_usage_bytes() > quota:
            protected = await asyncio.to_thread(e.research_store.pending_job_ranges)
            await asyncio.to_thread(e.candle_files.prune, quota, protected)
        return len(rows)

    async def _ensure_candles(self, pair: str, a: int, b: int) -> None:
        meta = await asyncio.to_thread(self.e.research_store.candle_meta, "BINANCE_USDM", pair, "1m")
        start = a
        if meta and meta["first_ms"] is not None and meta["first_ms"] <= a:
            start = max(a, int(meta["last_ms"]) + 60000)
        while start <= b:
            ok, why = self.gate_inner()
            if not ok:
                raise RuntimeError(f"RESEARCH_GATE:{why}")
            n = await self._download_page(pair, start, min(b, start + 1000 * 60000 - 1))
            if n == 0:
                break
            start += n * 60000
            await asyncio.sleep(1.0)

    def gate_inner(self) -> tuple[bool, str]:
        ok, why = self.gate()
        # Job yang sudah berjalan boleh lanjut walau scan aktif (jeda 1 dtk antar halaman).
        return (True, "OK") if why == "SCAN_ACTIVE" else (ok, why)

    async def _process(self, job: dict[str, Any]) -> None:
        e, store, files = self.e, self.e.research_store, self.e.candle_files
        jid, cid = job["job_id"], job["candidate_id"]
        self.current_job = jid
        try:
            cand = await asyncio.to_thread(store.get_candidate, cid)
            if cand is None:
                await asyncio.to_thread(store.update_job, jid, status="FAILED", last_error="candidate hilang")
                return
            pair = str(cand["pair"])
            expiry = int(_r_parse_iso(cand.get("expires_at_utc")).timestamp() * 1000) if _r_parse_iso(cand.get("expires_at_utc")) else None
            runner = ReplayRunner(cand, fee_pct=EST_FEE_PCT_PER_SIDE, slippage_pct=EST_SLIPPAGE_PCT_PER_SIDE, expiry_ms=expiry)
            if job.get("state_json"):
                runner.restore(json.loads(job["state_json"]))
            started, fed = time.monotonic(), 0
            cursor, to_ms = int(job["cursor_ms"]), int(job["to_ms"])
            end_ms = min(to_ms, int(time.time() * 1000) - 120_000)
            last_close = last_ms = None
            while cursor <= end_ms and fed < RESEARCH_MAX_CANDLES_PER_JOB and time.monotonic() - started < RESEARCH_MAX_RUNTIME_SECONDS and not runner.done:
                span_end = min(end_ms, cursor + 1000 * 60000 - 1)
                await self._ensure_candles(pair, cursor, span_end)
                rows = await asyncio.to_thread(lambda: list(files.read_range(pair, cursor, span_end)))
                if not rows:
                    break
                for row in rows:
                    last_close, last_ms = row[4], row[0]
                    fed += 1
                    if runner.feed(row):
                        break
                cursor = rows[-1][0] + 60000
                await asyncio.to_thread(store.update_job, jid, status="RUNNING", cursor_ms=cursor, state_json=json.dumps(runner.to_state()))
                await asyncio.sleep(0)
            horizon_reached = cursor > to_ms
            if runner.done or runner.finalize(horizon_reached, last_close, last_ms):
                result = runner.result()
                await asyncio.to_thread(store.save_replay_result, cid, str(cand.get("strategy_version") or ""), str(result["outcome"]), result)
                await asyncio.to_thread(store.update_job, jid, status="DONE", cursor_ms=cursor)
                e._ledger_replay_done(cid, result)
                self.jobs_done += 1
                self.last_ok_ts = time.time()
            else:
                await asyncio.to_thread(store.update_job, jid, status="PENDING", cursor_ms=cursor, state_json=json.dumps(runner.to_state()))
        except asyncio.CancelledError:
            try:
                await asyncio.shield(asyncio.to_thread(store.update_job, jid, status="PENDING"))
            except Exception:  # noqa: BLE001
                pass
            raise
        except BinanceRateLimitError as exc:
            self.rate_limited_until = time.monotonic() + float(getattr(exc, "server_cooldown_seconds", 60.0) or 60.0)
            self.last_error = f"{ErrorCategory.RATE_LIMIT}: {exc}"[:200]
            await asyncio.to_thread(store.update_job, jid, status="PENDING", last_error=self.last_error)
        except RuntimeError as exc:
            if str(exc).startswith("RESEARCH_GATE:"):
                await asyncio.to_thread(store.update_job, jid, status="PENDING")
                self.pause_reason = str(exc).split(":", 1)[1]
            else:
                self.last_error = f"{ErrorCategory.RESEARCH_ERROR}: {exc}"[:200]
                await asyncio.to_thread(store.fail_job, jid, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{ErrorCategory.RESEARCH_ERROR}: {type(exc).__name__}: {exc}"[:200]
            await asyncio.to_thread(store.fail_job, jid, self.last_error)
        finally:
            self.current_job = None

    def status(self) -> dict[str, Any]:
        e = self.e
        jobs = e.research_store.job_counts() if e.research_store and e.research_store.ok else {}
        return {"state": self.state, "pause_reason": self.pause_reason, "current_job": self.current_job, "jobs": jobs,
                "queue": jobs.get("PENDING", 0) + jobs.get("RUNNING", 0), "done": self.jobs_done, "last_error": self.last_error,
                "api_calls": self.api_calls, "api_weight_est": self.api_weight,
                "last_ok_age_s": None if self.last_ok_ts is None else round(time.time() - self.last_ok_ts)}


# ------------------------------------------------------------ R dari histori trade (real/sim)

def record_realized_r(rec: dict[str, Any], *, net: bool = True) -> float | None:
    """R terealisasi dari record histori. net=True memotong estimasi biaya round-trip."""
    try:
        base = float(rec.get("fill_price") or rec.get("entry") or 0)
        pnl = rec.get("pnl_percent")
        if base <= 0 or pnl in (None, ""):
            return None
        hist = rec.get("trail_history") or []
        isl = float(hist[0]["old_sl"]) if hist and isinstance(hist[0], dict) and hist[0].get("old_sl") else float(rec.get("sl") or 0)
        if isl <= 0:
            return None
        risk_pct = abs(base - isl) / base * 100.0
        if risk_pct <= 0:
            return None
        cost = 2.0 * (EST_FEE_PCT_PER_SIDE + EST_SLIPPAGE_PCT_PER_SIDE) if net else 0.0
        return (float(pnl) - cost) / risk_pct
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return None


def history_r_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec.get("result") not in {"TP", "SL", "TRAIL"}:
            continue
        r = record_realized_r(rec, net=True)
        if r is None:
            continue
        try:
            conf = float(rec.get("strategy_confidence"))
        except (TypeError, ValueError):
            conf = 0.0
        rows.append({"created_at": str(rec.get("filled_at") or rec.get("created_at") or ""), "confidence": conf,
                     "direction": rec.get("direction"), "btc_alignment": (rec.get("strategy_analysis") or {}).get("btc_alignment") if isinstance(rec.get("strategy_analysis"), dict) else None,
                     "archetype": rec.get("strategy_version"), "strategy_version": rec.get("strategy_version"),
                     "rr_after_costs": 0.0, "r": r, "r_fixed": r, "ambiguous": False,
                     "mode": "REAL" if rec.get("real_enabled") else "SIM"})
    return rows


def fmt_r_line(m: dict[str, Any]) -> str:
    if not m.get("n"):
        return "belum ada sampel"
    pf = "-" if m.get("profit_factor") is None else f"{m['profit_factor']:.2f}"
    warn = "" if m.get("reliable") else "  ⚠️ sampel kecil"
    ci = "" if not m.get("ci95_r") else f" CI95[{m['ci95_r'][0]:+.2f},{m['ci95_r'][1]:+.2f}]"
    return f"n={m['n']} • E={m['expectancy_r']:+.2f}R{ci} • win {m['win_rate']:.0f}% • PF {pf} • DD {m['max_drawdown_r']:.2f}R{warn}"


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
        self.drawdown_percent: Decimal | None = None
        self._drawdown_triggered = False
        self.sim_balance: Decimal = SIM_START_BALANCE
        self._sim_dirty = False

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
        self._bans_dirty = False
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
        await self._purge_balance_bans()
        self._init_research_layer()
        await self._research_startup()
        self._start_autostop_task()

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
        await self._flush_bans_if_dirty()
        await self._research_shutdown()

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

    def _shed_memory(self) -> None:
        """Katup darurat: buang detail analisis trade lama di RAM agar proses tidak di-kill OOM."""
        dropped = 0
        for record in self.history_records[:-40]:
            analysis = record.get("strategy_analysis")
            if isinstance(analysis, dict) and len(analysis) > 2:
                record["strategy_analysis"] = {k: analysis[k] for k in ("confidence", "scan") if k in analysis}
                dropped += 1
        release_memory()
        if dropped:
            log.warning(
                "[MEM] RAM %.0f MB di atas batas %.0f MB; analisis %s trade lama dipangkas.",
                rss_mb(),
                MEMORY_SOFT_LIMIT_MB,
                dropped,
            )

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
                    for record in payload:
                        if isinstance(record, dict) and isinstance(record.get("strategy_analysis"), dict):
                            record["strategy_analysis"] = compact_analysis(record["strategy_analysis"])
                    self.history_records = payload
                    raw_trades = None
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
            if str(reason).startswith("INSUFFICIENT_AVAILABLE_BALANCE"):
                raise InsufficientBalanceError(f"{meta.symbol}: {reason}")
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

            if trade.sizing_mode == "RISK_BUDGET":
                quantity, target, actual = self._risk_quantity_real(meta, trade, usdt, available, max_notional)
            else:
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
            except (InsufficientBalanceError, MarginInfluenceError, RiskRejectError) as exc:
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
            elif isinstance(last_exc, RiskRejectError):
                log.info("RISK REJECT | pair=%s | %s", trade.pair, last_exc)
            elif isinstance(last_exc, InsufficientBalanceError):
                log.info("SALDO KURANG | pair=%s | %s", trade.pair, last_exc)
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
                compact_analysis(record.get("strategy_analysis"))
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
            candidate_id=str(record.get("candidate_id") or ""),
            sizing_mode=str(record.get("sizing_mode") or ""),
            risk_info=record.get("risk_info") if isinstance(record.get("risk_info"), dict) else {},
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

    async def _purge_balance_bans(self) -> None:
        """Hapus ban permanen salah kaprah akibat saldo tersedia kurang (bukan sifat pair)."""
        stale = [
            pair
            for pair, item in self.banned_pairs.items()
            if item.get("source") == "AUTO_MARGIN_INFLUENCE"
            and (
                "INSUFFICIENT_AVAILABLE_BALANCE" in str(item.get("reason") or "")
                or "cycle berturut-turut" in str(item.get("reason") or "")
            )
        ]
        for pair in stale:
            self.banned_pairs.pop(pair, None)
        if stale:
            try:
                await self._persist_banned_pairs()
            except Exception:
                log.exception("Gagal menyimpan ban setelah pembersihan ban saldo-kurang.")
            log.info("[BAN] %s ban margin hasil deteksi awal/saldo-kurang dibersihkan", len(stale))

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
                            **({"btc_trend": item["btc_trend"]} if item.get("btc_trend") else {}),
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
                            **({"btc_trend": item["btc_trend"]} if item.get("btc_trend") else {}),
                        }
            except Exception:
                log.exception("Gagal membaca %s; daftar ban di-reset di RAM.", BANNED_PATH)
                loaded = {}

        self.banned_pairs = loaded
        await self._purge_expired_bans(persist=False)
        log.info("[BAN] Loaded active banned pairs: %s", len(self.banned_pairs))

    async def _persist_banned_pairs(self) -> None:
        async with self._ban_lock:
            self._bans_dirty = False
            rows = [self.banned_pairs[pair] for pair in sorted(self.banned_pairs)]
            payload = {
                "version": 1,
                "saved_at": iso_utc(),
                "bans": rows,
            }
            try:
                await self.github.replace_file(
                    BANNED_PATH,
                    json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    f"ban: update {len(rows)} pair(s)",
                )
            except Exception:
                self._bans_dirty = True
                raise

    async def _flush_bans_if_dirty(self) -> None:
        """Simpan ban ke GitHub sekali untuk banyak perubahan."""
        if not self._bans_dirty:
            return
        try:
            await self._persist_banned_pairs()
        except Exception:
            log.exception("Gagal menyimpan daftar ban ke GitHub.")

    @staticmethod
    def _hours_to_next_h4_close() -> Decimal:
        """Jam menuju close candle H4 berikutnya (batas UTC kelipatan 4 jam)."""
        now_ts = now_utc().timestamp()
        next_ts = (int(now_ts // 14400) + 1) * 14400
        return Decimal(str(round(max((next_ts - now_ts) / 3600, 0.001), 4)))

    def _purge_stale_structure_bans(self, btc_trend: str) -> None:
        """Ban mismatch hanya valid untuk tren BTC saat ban dibuat."""
        stale = [
            pair
            for pair, item in self.banned_pairs.items()
            if item.get("source") == "AUTO_STRUCTURE_MISMATCH"
            and item.get("btc_trend") != btc_trend
        ]
        for pair in stale:
            self.banned_pairs.pop(pair, None)
        if stale:
            self._bans_dirty = True
            log.info("[BAN] %s ban mismatch dibersihkan (BTC H4 kini %s)", len(stale), btc_trend)

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
            # Tidak ditulis; loader membuang ban kedaluwarsa saat start.

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
        extra: dict[str, Any] | None = None,
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

        entry = {
            "pair": pair,
            "banned_at": existing.get("banned_at") if existing and new_source != "MANUAL" else iso_utc(),
            "until": until,
            "reason": new_reason,
            "source": new_source,
        }
        if extra and not preserve_existing:
            entry.update(extra)
        self.banned_pairs[pair] = entry
        if str(source or "MANUAL").strip() == "MANUAL":
            await self._persist_banned_pairs()
        else:
            self._bans_dirty = True
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
        await self._flush_bans_if_dirty()

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

    async def _refresh_balance_cache(self) -> None:
        """Segarkan saldo wallet setelah PnL terealisasi (1 request, bobot 5)."""
        if not self.real_mode or self.real.cooldown_remaining > 0:
            return
        try:
            await self.real.get_futures_balance()
        except BinanceRateLimitError as exc:
            self.real.apply_cooldown(exc.server_cooldown_seconds, exc.bot_cooldown_seconds)
            self._notify_rate_limit(exc, "REFRESH BALANCE")
        except Exception as exc:
            log.info("[REAL] refresh saldo gagal: %s", exc)

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

    # -- saldo simulasi, drawdown
    def _sim_notional(self, trade: Trade) -> Decimal:
        qty = (trade.risk_info or {}).get("qty")
        if qty:
            try:
                return Decimal(str(qty)) * (trade.fill_price or trade.entry)
            except InvalidOperation:
                pass
        return trade.margin_usdt * Decimal(trade.leverage)

    def _sim_unrealized(self) -> Decimal:
        total = Decimal("0")
        for t in self.active_trades.values():
            if t.real_enabled or t.status != "FILLED":
                continue
            snap = self.prices.get(t.pair)
            fill = t.fill_price or t.entry
            if snap is None or fill <= 0:
                continue
            total += self._sim_notional(t) * pct_change(t.direction, fill, snap.price) / Decimal("100")
        return total

    def _sim_used_margin(self) -> Decimal:
        return sum((t.margin_usdt for t in self.active_trades.values() if not t.real_enabled and t.status in {"PENDING", "FILLED"}), Decimal("0"))

    def _sim_equity(self) -> Decimal:
        return self.sim_balance + self._sim_unrealized()

    def _sim_has_margin(self, trade: Trade) -> bool:
        return self._sim_equity() - self._sim_used_margin() >= trade.margin_usdt

    def _sim_apply_close(self, trade: Trade) -> None:
        """Sinkron; dipanggil di dalam trade lock. PnL USDT = notional x % - estimasi biaya."""
        if trade.fill_price is None or trade.pnl_percent is None:
            return
        notional = self._sim_notional(trade)
        gross = notional * trade.pnl_percent / Decimal("100")
        cost = notional * Decimal(str(2 * (EST_FEE_PCT_PER_SIDE + EST_SLIPPAGE_PCT_PER_SIDE))) / Decimal("100")
        net = gross - cost
        self.sim_balance += net
        trade.risk_info = {**(trade.risk_info or {}), "sim_pnl_usdt": decimal_to_str(net), "sim_notional": decimal_to_str(notional)}
        self._sim_dirty = True

    async def _save_sim_account(self) -> None:
        payload = {"version": 1, "saved_at": iso_utc(), "balance": decimal_to_str(self.sim_balance), "start_balance": decimal_to_str(SIM_START_BALANCE)}
        try:
            await self.github.replace_file(SIM_ACCOUNT_PATH, json.dumps(payload).encode("utf-8"), "sim: account")
            self._sim_dirty = False
        except Exception as exc:  # noqa: BLE001
            log.warning("[SIM] saldo simulasi gagal disimpan: %s", exc)

    async def _load_sim_account(self) -> None:
        try:
            raw, _ = await self.github.get_file(SIM_ACCOUNT_PATH)
            if raw:
                value = Decimal(str(json.loads(raw.decode("utf-8")).get("balance")))
                if value.is_finite():
                    self.sim_balance = value
        except Exception as exc:  # noqa: BLE001
            log.info("[SIM] saldo simulasi tidak dimuat: %s", exc)

    def _guard_equity(self) -> Decimal:
        return self._current_equity_cached() if self.real_mode else self._sim_equity()

    def _drawdown_floor(self) -> Decimal | None:
        if self.drawdown_percent is None or self._equity_peak is None:
            return None
        return self._equity_peak * (Decimal("100") - self.drawdown_percent) / Decimal("100")

    def _drawdown_status_text(self) -> str:
        lines = ["🧨 DRAWDOWN", ""]
        lines.append("Threshold: OFF" if self.drawdown_percent is None else f"Threshold: {decimal_to_str(self.drawdown_percent)}% dari puncak equity")
        lines.append("Mode: REAL" if self.real_mode else f"Mode: SIMULASI (saldo {self._fmt_usd(self.sim_balance)} USDT)")
        lines.append(f"Peak Equity: {self._fmt_usd(self._equity_peak)} USDT")
        lines.append(f"Equity Terakhir: {self._fmt_usd(self._equity_last)} USDT")
        floor = self._drawdown_floor()
        if floor is not None:
            lines.append(f"Batas: {self._fmt_usd(floor)} USDT")
        lines.append(f"Status: {'TERPICU' if self._drawdown_triggered else 'AMAN'}")
        lines.append("Aksi saat tersentuh: tutup SEMUA isi /trade + scan OFF.")
        return "\n".join(lines)

    async def _trigger_drawdown(self, equity: Decimal, floor: Decimal) -> None:
        self._drawdown_triggered = True
        self._autostop_triggered = True
        self._scan_user_enabled = False
        self._scan_auto_paused = False
        peak = self._equity_peak or equity
        dd = (peak - equity) / peak * Decimal("100") if peak > 0 else Decimal("0")
        log.info("[DRAWDOWN] terpicu: equity=%s peak=%s", equity, peak)
        await self.reply(
            "🧨 DRAWDOWN TERPICU\n\n"
            f"Equity turun {decimal_to_str(dd.quantize(Decimal('0.01')))}% dari puncak "
            f"(batas {decimal_to_str(self.drawdown_percent)}%).\n"
            f"Peak: {self._fmt_usd(peak)} • Sekarang: {self._fmt_usd(equity)} • Batas: {self._fmt_usd(floor)} USDT\n\n"
            "SCAN dimatikan. Menutup seluruh isi /trade..."
        )
        trades = list(self.active_trades.values())
        ok_n = fail_n = 0
        details: list[str] = []
        self._close_batch_in_progress = True
        try:
            for trade in trades:
                try:
                    ok, detail = await self._close_one_trade(trade)
                    ok_n += 1 if ok else 0
                    fail_n += 0 if ok else 1
                    details.append(f"{trade.pair} → {detail}")
                except Exception as exc:  # noqa: BLE001
                    fail_n += 1
                    log.exception("[DRAWDOWN] gagal menutup %s", trade.trade_id)
                    details.append(f"{trade.pair} → ERROR: {exc}")
        finally:
            self._close_batch_in_progress = False
        await self.reply(
            f"🧨 DRAWDOWN: /trade dibersihkan\nDiproses {len(trades)} • berhasil {ok_n} • gagal {fail_n}\n\n"
            + "\n".join(details[:25])
            + ("\n\n⚠️ Ada yang gagal: periksa /trade dan gunakan /del." if fail_n else "")
            + "\n\nMulai lagi: /drawdown reset lalu /scan on"
        )

    async def _handle_drawdown_command(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            await self.reply(self._drawdown_status_text())
            return
        arg = parts[1].lower()
        if arg == "off":
            self.drawdown_percent = None
            self._drawdown_triggered = False
            await self.reply("🧨 DRAWDOWN OFF")
            return
        if arg == "reset":
            await self._autostop_rebaseline()
            self._drawdown_triggered = False
            await self.reply("♻️ Puncak equity diset ulang; drawdown aktif lagi. Scan tetap OFF sampai /scan on.\n\n" + self._drawdown_status_text())
            return
        try:
            value = Decimal(arg.replace(",", "."))
        except InvalidOperation:
            raise ValueError("Gunakan /drawdown [persen], /drawdown off, atau /drawdown reset.")
        if not value.is_finite() or value <= 0 or value >= 100:
            raise ValueError("Drawdown harus di antara 0 dan 100 (persen).")
        self.drawdown_percent = value
        if self._drawdown_triggered or self._equity_peak is None:
            await self._autostop_rebaseline()
            self._drawdown_triggered = False
        await self.reply(f"✅ DRAWDOWN = {decimal_to_str(value)}%\n\n" + self._drawdown_status_text()
                         + "\n\n⚠️ Bila tersentuh, SEMUA setup di /trade ditutup (real: posisi market-close + order dibatalkan).")

    async def _handle_resetbalance_command(self) -> None:
        if self.real_mode:
            await self.reply("❌ /resetbalance hanya untuk simulasi. Matikan /real terlebih dahulu.")
            return
        active = sum(1 for t in self.active_trades.values() if not t.real_enabled)
        self.sim_balance = SIM_START_BALANCE
        self._equity_peak = None
        self._equity_last = None
        self._autostop_triggered = False
        self._drawdown_triggered = False
        await self._save_sim_account()
        note = f"\n⚠️ {active} setup simulasi masih aktif; hasilnya masuk ke saldo baru." if active else ""
        await self.reply(f"♻️ SALDO SIMULASI DIRESET\n\nSaldo: {self._fmt_usd(SIM_START_BALANCE)} USDT\nPeak/autostop/drawdown di-reset.{note}")

    async def _show_balance(self) -> None:
        if self.real_mode:
            try:
                equity = self._current_equity_cached()
                rows = [f"Mode REAL", f"Equity {self._fmt_usd(equity)} USDT", f"PnL sementara {self._fmt_usd(self._temporary_real_pnl_usdt())} USDT"]
            except Exception as exc:  # noqa: BLE001
                rows = ["Mode REAL", f"Saldo belum tersedia: {exc}"]
        else:
            unreal = self._sim_unrealized()
            used = self._sim_used_margin()
            equity = self.sim_balance + unreal
            rows = [
                "Mode SIMULASI",
                f"Saldo      {self._fmt_usd(self.sim_balance)} USDT",
                f"Unrealized {self._fmt_usd(unreal)} USDT",
                f"Equity     {self._fmt_usd(equity)} USDT",
                f"Margin pakai {self._fmt_usd(used)} • bebas {self._fmt_usd(equity - used)}",
                f"PnL sejak reset {self._fmt_usd(self.sim_balance - SIM_START_BALANCE)} USDT",
            ]
        rows.append(f"Autostop {self._autostop_short()}")
        rows.append(f"Drawdown {'OFF' if self.drawdown_percent is None else decimal_to_str(self.drawdown_percent) + '% dari puncak'}")
        await self.reply(card("💼 SALDO", rows))

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
        if self.real_mode:
            lines.append("Mode: REAL")
        else:
            lines.append(f"Mode: SIMULASI (saldo {self._fmt_usd(self.sim_balance)} USDT)")
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
        while self._running:
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
        try:
            equity = self._guard_equity()
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
        dd_floor = self._drawdown_floor()
        if dd_floor is not None and not self._drawdown_triggered and equity <= dd_floor:
            await self._trigger_drawdown(equity, dd_floor)
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
        equity = self._guard_equity()
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
        if self._autostop_triggered or self._equity_peak is None:
            await self._autostop_rebaseline()
        suffix = ""
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
        hint = self._margin_hint(pair, price)
        return (
            f"Quantity terdekat menghasilkan notional {decimal_to_str(best_notional)}, "
            f"di luar target {decimal_to_str(target)} ±{decimal_to_str(SCAN_MARGIN_TOLERANCE*100)}%."
            + (f" Margin pas ≈ {hint}." if hint else "")
        )

    def _margin_hint(self, pair: str, price: Decimal) -> str | None:
        """Margin yang menghasilkan qty legal untuk pair ini (leverage saat ini)."""
        margin, leverage = self._margin_config()
        meta = self.symbols.get(pair)
        if margin is None or leverage is None or meta is None or leverage <= 0 or price <= 0:
            return None
        try:
            def legal(qty: Decimal) -> bool:
                if qty <= 0:
                    return False
                if meta.min_qty > 0 and qty < meta.min_qty:
                    return False
                if meta.max_qty > 0 and qty > meta.max_qty:
                    return False
                return not (meta.min_notional > 0 and qty * price < meta.min_notional)

            raw_qty = margin * leverage / price
            if meta.step_size > 0:
                floor_qty = (raw_qty / meta.step_size).to_integral_value(rounding=ROUND_DOWN) * meta.step_size
                qtys = [floor_qty, floor_qty + meta.step_size]
            else:
                qtys = [raw_qty]
            found = [q for q in qtys if legal(q)]
            if not found:
                q0 = meta.min_qty if meta.min_qty > 0 else meta.step_size
                if meta.min_notional > 0:
                    q0 = max(q0, meta.min_notional / price)
                if meta.step_size > 0:
                    q0 = (q0 / meta.step_size).to_integral_value(rounding=ROUND_CEILING) * meta.step_size
                found = [q0] if q0 > 0 else []
            values = sorted({(q * price / leverage).quantize(Decimal("0.01")) for q in found})
            return "/".join(f"${v}" for v in values) or None
        except (InvalidOperation, ArithmeticError):
            return None

    async def _process_margin_filter(self, pair: str, price: Decimal) -> tuple[bool, str | None]:
        # Deteksi awal dimatikan; margin influence hanya diban saat entry real gagal.
        return False, None

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
                release_memory()
                await self._update_pressure()

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
                await asyncio.sleep(self._next_scan_delay())
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

    # ------------------------------------------------------------------
    # RESEARCH / RESILIENCE LAYER (blueprint V1)
    # ------------------------------------------------------------------

    def _init_research_layer(self) -> None:
        self._resource = ResourceMonitor(MEMORY_SOFT_LIMIT_MB)
        self._recheck = RecheckBook()
        self.research_store: ResearchStore | None = None
        self.candle_files: CandleFileStore | None = None
        self.research_ephemeral = False
        self.ledger = CandidateLedger(store=None)
        self._funnel_last: dict[str, Any] = {}
        self._funnel_history: deque[dict[str, Any]] = deque(maxlen=48)
        self._scan_durations: deque[float] = deque(maxlen=20)
        self._retry_counts: dict[str, int] = {}
        self._scan_cycle_active = False
        self._last_cycle_analyzed = 0
        self._last_state_export = 0.0
        self._strategy_module_ref: Any = None
        self._replay_pending_ids: set[str] = set()
        self._sweep: dict[str, Any] = {"slot": None, "done": set()}
        self._research_day: tuple[str, int] = ("", 0)
        self._day_anchor: tuple[str, Decimal] | None = None
        self._risk_cfg: dict[str, Any] = {
            "mode": RISK_SIZING_MODE, "per_trade": RISK_PER_TRADE_PCT, "portfolio": MAX_PORTFOLIO_RISK_PCT,
            "direction": MAX_DIRECTION_RISK_PCT, "dd": MAX_DAILY_DRAWDOWN_PCT, "paper_equity": PAPER_EQUITY_USDT,
        }
        self.research_worker = ResearchWorker(self)

    async def _open_research_store(self, ephemeral_ok: bool = False) -> tuple[bool, str]:
        directory, ephemeral = BOT_DATA_DIR, False
        if not directory:
            if not ephemeral_ok:
                return False, "BOT_DATA_DIR belum diatur (Render free tidak punya disk persisten)."
            directory, ephemeral = str(Path(tempfile.gettempdir()) / "bot_research"), True
        store = ResearchStore(directory)
        if not await asyncio.to_thread(store.open):
            return False, str(store.error)
        self.research_store, self.candle_files, self.research_ephemeral = store, CandleFileStore(Path(directory)), ephemeral
        self.ledger.store = store
        for rec in self.ledger.items.values():
            self.ledger._pending[rec.candidate_id] = rec.to_dict()
        await asyncio.to_thread(store.reset_running_jobs)
        return True, directory

    async def _research_startup(self) -> None:
        await self._load_runtime_settings()
        await self._load_sim_account()
        await self._load_research_state()
        if RESEARCH_ENABLED:
            ok, info = await self._open_research_store(ephemeral_ok=False)
            if ok:
                self.research_worker.start()
                log.info("[RESEARCH] aktif | dir=%s", info)
            else:
                log.warning("[RESEARCH] diminta aktif tetapi storage tidak tersedia: %s", info)

    async def _research_shutdown(self) -> None:
        try:
            await self.research_worker.stop()
            await self.ledger.flush()
            await self._export_research_state(force=True)
        finally:
            if self.research_store is not None:
                await asyncio.to_thread(self.research_store.close)

    async def _load_runtime_settings(self) -> None:
        try:
            raw, _ = await self.github.get_file(RUNTIME_SETTINGS_PATH)
            if raw:
                data = json.loads(raw.decode("utf-8"))
                guard = data.get("guard") or {}
                for key, attr in (("autostop", "autostop_percent"), ("drawdown", "drawdown_percent")):
                    raw_v = guard.get(key)
                    if raw_v not in (None, ""):
                        val = Decimal(str(raw_v))
                        if val.is_finite() and 0 < val < 100:
                            setattr(self, attr, val)
                risk = data.get("risk") or {}
                mode = str(risk.get("mode") or "").upper()
                if mode in {"LEGACY_NOTIONAL", "RISK_BUDGET"}:
                    self._risk_cfg.update({k: risk.get(k) for k in ("per_trade", "portfolio", "direction", "dd", "paper_equity")})
                    self._risk_cfg["mode"] = mode
                    self._validate_risk_cfg(announce=False)
        except Exception as exc:  # noqa: BLE001
            log.info("[RISK] runtime settings tidak dimuat: %s", exc)

    def _validate_risk_cfg(self, announce: bool = True) -> str | None:
        cfg = self._risk_cfg
        if cfg["mode"] == "RISK_BUDGET" and (not cfg.get("per_trade") or not cfg.get("portfolio")):
            cfg["mode"] = "LEGACY_NOTIONAL"
            return "RISK_BUDGET butuh /risk pertrade dan /risk portfolio; kembali ke LEGACY_NOTIONAL."
        return None

    async def _save_runtime_settings(self) -> None:
        payload = {"version": 1, "saved_at": iso_utc(), "risk": dict(self._risk_cfg),
                   "guard": {"autostop": None if self.autostop_percent is None else str(self.autostop_percent),
                             "drawdown": None if self.drawdown_percent is None else str(self.drawdown_percent)}}
        await self.github.replace_file(RUNTIME_SETTINGS_PATH, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), "settings: risk")

    async def _load_research_state(self) -> None:
        try:
            raw, _ = await self.github.get_file(CANDIDATES_STATE_PATH)
            if raw:
                data = json.loads(raw.decode("utf-8"))
                n = self.ledger.load_compact(data.get("ledger") or {}, now_utc())
                m = self._recheck.load(data.get("recheck") or {}, time.time())
                self._funnel_history.extend(data.get("funnel_history") or [])
                sw = data.get("sweep") or {}
                if sw.get("slot") == int(time.time() // (4 * 3600)):
                    self._sweep = {"slot": sw["slot"], "done": set(sw.get("done") or [])}
                log.info("[RESEARCH] state dipulihkan: kandidat=%s cooldown=%s", n, m)
        except Exception as exc:  # noqa: BLE001
            log.info("[RESEARCH] state tidak dimuat: %s", exc)

    async def _export_research_state(self, force: bool = False) -> None:
        """Ringkasan kecil ke GitHub (bukan data candle), paling sering tiap RESEARCH_EXPORT_INTERVAL_SECONDS."""
        now = time.monotonic()
        if not force and (not self.ledger.dirty_export or now - self._last_state_export < RESEARCH_EXPORT_INTERVAL_SECONDS):
            return
        payload = {"version": 1, "saved_at": iso_utc(), "ledger": self.ledger.export_compact(),
                   "recheck": self._recheck.export(time.time()), "funnel_history": list(self._funnel_history)[-24:],
                   "sweep": {"slot": self._sweep.get("slot"), "done": sorted(self._sweep.get("done", set()))}}
        try:
            await self.github.replace_file(CANDIDATES_STATE_PATH, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "research: compact state")
            self._last_state_export = now
            self.ledger.dirty_export = False
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] export state gagal: %s", ErrorCategory.PERSISTENCE_ERROR, exc)

    async def _flush_research(self) -> None:
        ok = await self.ledger.flush()
        if ok and self._replay_pending_ids and self.research_store is not None and self.research_worker.enabled:
            day = now_utc().strftime("%Y-%m-%d")
            if self._research_day[0] != day:
                self._research_day = (day, 0)
            for cid in list(self._replay_pending_ids):
                self._replay_pending_ids.discard(cid)
                rec = self.ledger.items.get(cid)
                if rec is None or self._research_day[1] >= RESEARCH_MAX_CANDIDATES_PER_DAY:
                    continue
                job = ReplayJob(job_id=f"rj-{cid}", candidate_id=cid, from_ms=rec.decision_ms,
                                to_ms=rec.decision_ms + int(RESEARCH_REPLAY_HORIZON_HOURS * 3600 * 1000))
                try:
                    if await asyncio.to_thread(self.research_store.create_replay_job, job):
                        self._research_day = (day, self._research_day[1] + 1)
                except Exception as exc:  # noqa: BLE001
                    log.warning("[%s] gagal antre replay %s: %s", ErrorCategory.PERSISTENCE_ERROR, cid, exc)
        await self._export_research_state()

    def _ledger_replay_done(self, cid: str, result: dict[str, Any]) -> None:
        rec = self.ledger.items.get(cid)
        if rec is not None:
            self.ledger._emit(rec, "REPLAY_COMPLETED", rec.state, str(result.get("outcome")),
                              {"net_r": result.get("net_r"), "ambiguous": result.get("ambiguous")})

    def _new_funnel(self) -> dict[str, Any]:
        keys = ("universe", "usable_data", "structure_analyzed", "direction_thesis", "candidates_evaluated", "hard_rejects",
                "quality_rejects", "waiting_price", "waiting_trigger", "ready", "validation_pass", "validation_reject",
                "validation_error", "risk_rejects", "pending_created", "structure_mismatch", "data_unavailable", "data_stale",
                "budget_stopped", "memory_skipped", "validation_skipped_topn", "rate_limited", "sweep_covered", "sweep_total", "sim_margin_blocked")
        out: dict[str, Any] = {k: 0 for k in keys}
        out["errors"] = {}
        return out

    def _observe_candidate(self, rec: CandidateRecord) -> CandidateRecord:
        rec.priority = candidate_priority(rec, now_ts=time.time())
        rec = self.ledger.observe(rec)
        eligible, why = replay_eligibility(rec, float(self.scan_threshold))
        self.ledger.mark_replay(rec.candidate_id, eligible, why)
        if eligible and self.research_worker.enabled:
            self._replay_pending_ids.add(rec.candidate_id)
        return rec

    def _priority_of(self, candidate: dict[str, Any]) -> float:
        rec = self.ledger.items.get(str(candidate.get("_candidate_id") or ""))
        return rec.priority if rec is not None else float(candidate.get("confidence") or 0)

    def _after_validation(self, candidate: dict[str, Any], reason: str, state: str, now_ts: float) -> None:
        cid = str(candidate.get("_candidate_id") or "")
        if cid:
            self.ledger.set_state(cid, state, "CANDIDATE_REJECTED", reason)
        self._recheck.set(str(candidate.get("pair") or ""), now_ts + SCAN_WATCHLIST_INTERVAL, reason, state)

    def _sync_ledger_trades(self) -> None:
        live = {t.candidate_id: t for t in self.active_trades.values() if getattr(t, "candidate_id", "")}
        for cid, trade in live.items():
            self.ledger.set_state(cid, "FILLED" if trade.status == "FILLED" else "PENDING", "TRADE_STATE", f"TRADE_{trade.status}")

    def _next_scan_delay(self) -> float:
        if self._last_cycle_analyzed > 0:
            return SCAN_CYCLE_DELAY_SECONDS
        now = time.time()
        dues = [d for d in (self._recheck.next_due_ts(), self.ledger.next_watch_due_ts()) if d is not None]
        delay = min(SCAN_WATCHLIST_INTERVAL, 300.0) if not dues else min(min(dues) - now, SCAN_WATCHLIST_INTERVAL)
        delay = max(SCAN_CYCLE_DELAY_SECONDS, delay)
        # Jangan tidur melewati close H4 berikutnya (slot /H4 harus tepat waktu).
        return max(1.0, min(delay, next_boundary_ts(now, 4 * 3600) - now))

    def _api_headroom_pct(self) -> float | None:
        if self.real.cooldown_remaining > 0:
            return 0.0
        used, at = API_STATS.used_weight, API_STATS.used_weight_at
        if used is None or at is None or time.monotonic() - at > 75:
            return None
        return max(0.0, 100.0 - used / BINANCE_IP_WEIGHT_LIMIT * 100.0)

    def _protection_healthy(self) -> bool:
        return not any(t.real_enabled and (t.real_error or t.real_state in {"REAL_ERROR", "PROTECTION_ERROR"}) for t in self.active_trades.values())

    async def _update_pressure(self) -> None:
        rss = rss_mb()
        level, changed, alarm = self._resource.update(rss)
        module = self._strategy_module_ref
        if level in {"HIGH", "CRITICAL"}:
            if module is not None and hasattr(module, "series_cache_shrink"):
                module.series_cache_shrink(0.5 if level == "HIGH" else 1.0)
            if level == "CRITICAL":
                self._shed_memory()
        if alarm:
            await self.reply(card("🧠 TEKANAN MEMORI", [
                f"Level  {level}", f"RSS    {rss:.0f}/{self._resource.limit_mb:.0f} MB ({self._resource.pressure_pct(rss):.0f}%)",
                "Cache candle dikurangi; research ditunda; scan tetap jalan.",
            ]))

    # -- risk sizing (opt-in)
    def _planned_loss_of(self, trade: Trade) -> Decimal:
        try:
            if trade.risk_info and trade.risk_info.get("planned_loss_usdt"):
                return Decimal(str(trade.risk_info["planned_loss_usdt"]))
            if trade.quantity:
                return Decimal(str(trade.quantity)) * abs(trade.entry - trade.sl)
        except (InvalidOperation, TypeError):
            pass
        return Decimal("0")

    def _equity_for_risk(self) -> Decimal | None:
        if self.real_mode:
            try:
                return self._current_equity_cached()
            except Exception:  # noqa: BLE001
                return None
        paper = self._risk_cfg.get("paper_equity")
        return Decimal(str(paper)) if paper else None

    def _daily_drawdown_pct(self) -> float | None:
        equity = self._equity_for_risk()
        if equity is None:
            return None
        day = now_local().strftime("%Y-%m-%d")
        if self._day_anchor is None or self._day_anchor[0] != day:
            self._day_anchor = (day, equity)
            return 0.0
        base = self._day_anchor[1]
        return float(max(Decimal(0), (base - equity) / base * 100)) if base > 0 else None

    def _risk_gate_for_candidate(self, trade: Trade) -> tuple[bool, dict[str, Any]]:
        cfg = self._risk_cfg
        meta = self.symbols.get(trade.pair)
        equity = self._equity_for_risk()
        if meta is None:
            return False, {"reason": "SYMBOL_META_UNAVAILABLE"}
        available = None
        if self.real_mode:
            bal = self.real.cached_usdt_balance()
            if bal and bal.get("availableBalance") is not None:
                available = parse_signed_decimal(str(bal.get("availableBalance")))
        res = compute_risk_quantity(meta=meta, direction=trade.direction, entry=trade.entry, sl=trade.sl, equity=equity,
                                    risk_pct=cfg.get("per_trade"), leverage=trade.leverage, margin_cap_usdt=trade.margin_usdt,
                                    available_balance=available)
        if not res["ok"]:
            return False, {"reason": res["reason"], "mode": "RISK_BUDGET"}
        active = [(t.direction, self._planned_loss_of(t)) for t in self.active_trades.values() if t.status in {"PENDING", "FILLED"}]
        ok, why, info = portfolio_risk_check(active=active, new_direction=trade.direction, new_loss=res["planned_loss_usdt"], equity=equity,
                                             max_portfolio_pct=cfg.get("portfolio"), max_direction_pct=cfg.get("direction"),
                                             daily_drawdown_pct=self._daily_drawdown_pct(), max_daily_dd_pct=cfg.get("dd"),
                                             protection_ok=self._protection_healthy())
        out = {"mode": "RISK_BUDGET", "reason": why, "risk_budget_usdt": str(res["risk_budget_usdt"]), "qty": str(res["qty"]),
               "planned_loss_usdt": str(res["planned_loss_usdt"]), "binding_cap": res["binding_cap"],
               "equity": str(equity), "assumptions": {"fee_pct": EST_FEE_PCT_PER_SIDE, "slippage_pct": EST_SLIPPAGE_PCT_PER_SIDE}, **info}
        return ok, out

    def _risk_quantity_real(self, meta: Any, trade: Trade, usdt_row: dict[str, Any], available: Decimal, max_notional: Decimal | None) -> tuple[Decimal, Decimal, Decimal]:
        equity = self._equity_from_balance_row(usdt_row) + self._temporary_real_pnl_usdt()
        res = compute_risk_quantity(meta=meta, direction=trade.direction, entry=trade.entry, sl=trade.sl, equity=equity,
                                    risk_pct=self._risk_cfg.get("per_trade"), leverage=trade.leverage, margin_cap_usdt=trade.margin_usdt,
                                    available_balance=available, max_notional=max_notional)
        if not res["ok"]:
            raise RiskRejectError(f"{trade.pair}: {res['reason']}")
        trade.risk_info = {**(trade.risk_info or {}), "mode": "RISK_BUDGET", "qty": str(res["qty"]), "planned_loss_usdt": str(res["planned_loss_usdt"]),
                           "risk_budget_usdt": str(res["risk_budget_usdt"]), "binding_cap": res["binding_cap"], "equity": str(equity)}
        return res["qty"], trade.margin_usdt * Decimal(trade.leverage), res["qty"] * trade.entry

    # -- laporan
    def _pct_bar(self, value: float | None) -> str:
        return "-" if value is None else f"{value:.0f}%"

    def _status_extra_report(self) -> str:
        rss = rss_mb()
        self._resource.update(rss)
        try:
            import sys as _sys
            mod = self._strategy_module_ref
            cache = mod.series_cache_stats() if mod is not None and hasattr(mod, "series_cache_stats") else {}
        except Exception:  # noqa: BLE001
            cache = {}
        durs = list(self._scan_durations)
        counts = self.ledger.counts()
        now_dt = now_utc()
        age = self.ledger.oldest_active_age_seconds(now_dt)
        rw = self.research_worker.status()
        rows = [
            f"🧠 RAM {rss:.0f}/{self._resource.limit_mb:.0f} MB ({self._resource.pressure_pct(rss):.0f}%) • {self._resource.level} • puncak {self._resource.peak_rss_mb:.0f} • batas {self._resource.limit_source}",
            f"🗃 Cache candle {cache.get('entries', 0)} entri • {cache.get('approx_mb', 0)} MB • hit {('-' if cache.get('hit_rate') is None else str(round(cache['hit_rate'] * 100)) + '%')} • evict {cache.get('evictions', 0)} • gagal {cache.get('failures', 0)}",
            f"🛰 Scan terakhir {('-' if not durs else f'{durs[-1]:.0f}s')} • rata² {('-' if not durs else f'{sum(durs) / len(durs):.0f}s')} • analisis {self._last_cycle_analyzed} pair",
            f"📋 Kandidat {sum(counts.values())}: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "kosong"),
            f"⏳ Kandidat aktif tertua {'-' if age is None else f'{age / 60:.0f} mnt'} • cooldown {sum(self._recheck.counts_by_reason(time.time()).values())} pair",
            f"🔬 Research {rw['state']} • antrean {rw['queue']} • selesai {rw['done']}" + (f" • jeda: {rw['pause_reason']}" if rw["pause_reason"] else ""),
            f"📡 API headroom {self._pct_bar(self._api_headroom_pct())} • 429/418 {API_STATS.limit_hits}x",
            f"🛡 Proteksi real {'OK' if self._protection_healthy() else 'ADA ERROR — entry baru ditahan'}",
            f"⚖️ Sizing {self._risk_cfg['mode']}",
        ]
        if not self.real_mode:
            rows.append(f"💼 Saldo simulasi {self._fmt_usd(self.sim_balance)} • equity {self._fmt_usd(self._sim_equity())} USDT")
        rows.append(f"🧨 Drawdown {'OFF' if self.drawdown_percent is None else decimal_to_str(self.drawdown_percent) + '%'} • 🛡 Autostop {'OFF' if self.autostop_percent is None else decimal_to_str(self.autostop_percent) + '%'}")
        stale = self._funnel_last.get("data_stale", 0)
        if stale:
            rows.append(f"⚠️ Data basi pada {stale} pair di scan terakhir")
        return card("🧭 DIAGNOSTIK", rows)

    def _api_extra_report(self) -> str:
        rw = self.research_worker
        head = self._api_headroom_pct()
        rows = [f"🔬 Research  {rw.api_calls} req ≈{rw.api_weight} bobot sejak start • {rw.state}",
                f"📉 Headroom  {self._pct_bar(head)} (research butuh ≥ {RESEARCH_MIN_API_HEADROOM_PCT:.0f}%)",
                "Prioritas: proteksi/real > scan > validasi > research (research berhenti dulu)."]
        return "\n\n" + card("🧭 PRIORITAS API", rows)

    def _stats_r_lines(self) -> list[str]:
        rows = history_r_rows(self.history_records)
        m = r_metrics([r["r"] for r in rows])
        lines = ["", "📐 R-METRIK (setelah estimasi biaya)", f"└ {fmt_r_line(m)}"]
        if m.get("n"):
            lines.insert(2, f"├ Avg Win {m['avg_win_r']:+.2f}R • Avg Loss {m['avg_loss_r']:+.2f}R • median {m['median_r']:+.2f}R")
            lines[-1] = f"└ {fmt_r_line(m)}"
        return lines

    def _research_summary(self) -> dict[str, Any]:
        return {"candidates": self.ledger.counts(), "funnel_last": self._funnel_last, "funnel_history": list(self._funnel_history)[-12:],
                "cooldowns": self._recheck.counts_by_reason(time.time()), "research": self.research_worker.status(),
                "store": self.research_store.stats() if self.research_store else None, "risk": {k: v for k, v in self._risk_cfg.items()},
                "scan_durations": list(self._scan_durations)}

    def _research_report_md(self) -> str:
        rows = history_r_rows(self.history_records)
        lines = ["", "## Riset & Resiliensi", "", f"- Kandidat: {self.ledger.counts()}", f"- Funnel scan terakhir: {self._funnel_last}",
                 f"- R-metrik histori: {fmt_r_line(r_metrics([r['r'] for r in rows]))}",
                 f"- Cooldown recheck: {self._recheck.counts_by_reason(time.time())}"]
        return "\n".join(lines) + "\n"

    async def show_candidate_detail(self, pair: str) -> None:
        pair = normalize_symbol(pair)
        recs = [r for r in self.ledger.items.values() if r.pair == pair]
        cd = self._recheck.items.get(pair)
        ban = self.banned_pairs.get(pair)
        if not recs and cd is None and ban is None:
            await self.reply(f"Belum ada catatan kandidat untuk {pair} di sesi ini.")
            return
        rows = []
        for r in sorted(recs, key=lambda x: x.updated_at_utc, reverse=True)[:3]:
            comps = ", ".join(f"{k}={float(v):.0f}" for k, v in list(r.confidence_components.items())[:10]) if r.confidence_components else "-"
            rows += [f"{r.state} • {r.direction or '-'} • conf {r.confidence:.0f} • {r.state_reason or '-'}",
                     f"  entry {r.entry} sl {r.sl} tp {r.tp} • RR {r.planned_rr_gross}/{r.planned_rr_after_costs}",
                     f"  gate {r.gate_reasons or '-'} • BTC {r.btc_alignment or '-'}", f"  komponen {comps}",
                     f"  evaluasi ulang {r.next_evaluate_after_utc or '-'}"]
        if cd:
            rows.append(f"Cooldown recheck: {cd['reason']} s/d {datetime.fromtimestamp(cd['until'], tz=timezone.utc).strftime('%H:%M:%S')} UTC")
        if ban:
            rows.append(f"BAN: {ban.get('source')} • {ban.get('reason')}")
        await self.reply(card(f"🔎 {pair}", rows))

    async def show_candidates(self) -> None:
        self._sync_ledger_trades()
        counts = self.ledger.counts()
        f = self._funnel_last
        funnel_rows = [
            f"🌐 Universe {f.get('universe', 0)} → data {f.get('usable_data', 0)} → struktur {f.get('structure_analyzed', 0)} → searah {f.get('direction_thesis', 0)}",
            f"🧪 Dievaluasi {f.get('candidates_evaluated', 0)} • hard {f.get('hard_rejects', 0)} • kualitas {f.get('quality_rejects', 0)} (mismatch {f.get('structure_mismatch', 0)})",
            f"👀 Waiting price {f.get('waiting_price', 0)} • trigger {f.get('waiting_trigger', 0)} • ready {f.get('ready', 0)}",
            f"✅ Validasi lolos {f.get('validation_pass', 0)} • tolak {f.get('validation_reject', 0)} • error {f.get('validation_error', 0)} • risk-reject {f.get('risk_rejects', 0)}",
            f"📥 Pending dibuat {f.get('pending_created', 0)}",
            f"🧹 Tersapu H4 ini {f.get('sweep_covered', 0)}/{f.get('sweep_total', 0)} koin",
        ]
        top = self.ledger.top(8, ACTIVE_CANDIDATE_STATES)
        top_rows = [f"{r.pair} {r.direction or '-'} {r.state} conf {r.confidence:.0f} • {r.state_reason or '-'}" for r in top] or ["belum ada"]
        persisted = "SQLite" if self.research_store and self.research_store.ok else "RAM + ringkasan GitHub"
        errs = f.get("errors") or {}
        lines = [
            card("🧬 KANDIDAT", [f"Total {sum(counts.values())} • penyimpanan {persisted}"] + [f"{k} {v}" for k, v in sorted(counts.items())]),
            "", card("🪜 FUNNEL (scan terakhir)", funnel_rows),
            "", "🥇 Teratas", *[f"• {x}" for x in top_rows],
            "", f"⏸ Cooldown recheck: {self._recheck.counts_by_reason(time.time()) or 'tidak ada'}",
        ]
        if errs:
            lines.append(f"⚠️ Error per kategori: {errs}")
        await self.reply("\n".join(lines))

    async def _handle_research_command(self, text: str) -> None:
        parts = text.split()
        sub = parts[1].lower() if len(parts) > 1 else "status"
        rw = self.research_worker
        if sub == "on":
            ok, info = await self._open_research_store(ephemeral_ok=True)
            if not ok:
                await self.reply(card("🔬 RESEARCH", ["Tidak bisa diaktifkan.", str(info)]))
                return
            rw.start()
            note = "⚠️ MODE SEMENTARA: data hilang saat Render restart/tidur. Ringkasan kandidat tetap disimpan ke GitHub." if self.research_ephemeral else "Penyimpanan permanen aktif."
            await self.reply(card("🔬 RESEARCH ON", [f"Dir {info}", note, f"Batas disk {min(RESEARCH_MAX_DISK_MB, 100.0 if self.research_ephemeral else RESEARCH_MAX_DISK_MB):.0f} MB", "Berhenti otomatis bila RAM/API ketat atau proteksi real bermasalah."]))
            return
        if sub == "off":
            await rw.stop()
            await self.reply("🔬 Research dimatikan. Data yang sudah tersimpan tidak dihapus.")
            return
        st = rw.status()
        rows = [f"Status  {st['state']}" + (f" ({st['pause_reason']})" if st["pause_reason"] else ""), f"Antrean {st['queue']} • selesai {st['done']} • job {st['jobs']}",
                f"API     {st['api_calls']} req ≈{st['api_weight_est']} bobot", f"Storage {'ephemeral' if self.research_ephemeral else ('permanen' if self.research_store else 'tidak aktif')}"]
        if self.research_store and self.research_store.ok:
            try:
                rows.append(f"Hasil replay {self.research_store.replay_counts()}")
                rows.append(f"Disk candle {self.candle_files.disk_usage_bytes() / 1024 / 1024:.1f} MB")
            except Exception as exc:  # noqa: BLE001
                rows.append(f"Storage bermasalah: {type(exc).__name__}")
        if st["last_error"]:
            rows.append(f"Error terakhir {st['last_error']}")
        await self.reply(card("🔬 RESEARCH", rows) + "\n\nPerintah: /research on | off | status")

    async def _build_edge_text(self) -> str:
        hist = history_r_rows(self.history_records)
        lines = [card("🧪 EDGE", ["Hasil real/sim & replay dilaporkan TERPISAH.", "Jangan ubah parameter dari sampel kecil."]), "",
                 "📚 Histori trade (R setelah estimasi biaya)", f"└ {fmt_r_line(r_metrics([r['r'] for r in hist]))}"]
        if hist:
            lines.append("Per confidence:")
            for k, m in bucket_report(hist, "confidence").items():
                lines.append(f"• {k}: {fmt_r_line(m)}")
            for key, label in (("direction", "Arah"),):
                for k, m in bucket_report(hist, key).items():
                    lines.append(f"• {label} {k}: {fmt_r_line(m)}")
        if self.research_store and self.research_store.ok:
            try:
                raw = await asyncio.to_thread(self.research_store.replay_rows)
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] replay tidak terbaca: %s", ErrorCategory.PERSISTENCE_ERROR, exc)
                raw = None
                lines += ["", f"⚠️ Replay tidak dapat dibaca ({type(exc).__name__}); laporan histori tetap ditampilkan."]
            rrows = replay_rows_for_stats(raw or [])
            amb = sum(1 for r in rrows if r["ambiguous"])
            if raw is not None:
                lines += ["", f"🎞 Replay M1 (konservatif; ambigu={amb}/{len(rrows)})", f"└ {fmt_r_line(r_metrics([r['r'] for r in rrows]))}"]
            if rrows:
                lines.append("Sapuan threshold (conf ≥ T):")
                for row in threshold_sweep(rrows):
                    lines.append(f"• ≥{row['threshold']}: {fmt_r_line(row)} • {row['signal_share_pct']}% sinyal")
                ev = evaluate_variants(rrows, default_variants(float(self.scan_threshold)))
                lines.append("Varian (satu perubahan; OOS):")
                for v in ev["variants"]:
                    lines.append(f"• {v['id']}: {fmt_r_line(v['out_of_sample'])}")
                lines.append(ev["promotion"])
            lines.append("Keterbatasan: trailing struktur/RSI & funding tidak direplikasi; urutan intra-candle M1 tak diketahui.")
        else:
            lines += ["", "🎞 Replay: belum aktif (gunakan /research on)."]
        return "\n".join(lines)

    async def _handle_risk_command(self, text: str) -> None:
        parts = text.split()
        cfg = self._risk_cfg
        if len(parts) == 1:
            await self.reply(card("⚖️ RISK SIZING", [
                f"Mode      {cfg['mode']}", f"Per trade {cfg.get('per_trade') or '-'}% equity", f"Portfolio {cfg.get('portfolio') or '-'}% equity",
                f"Arah      {cfg.get('direction') or '-'}%", f"Drawdown  {cfg.get('dd') or '-'}% / hari", f"Paper eq  {cfg.get('paper_equity') or '-'} USDT",
            ]) + "\n\n/risk mode legacy|budget\n/risk pertrade <%>\n/risk portfolio <%>\n/risk direction <%>\n/risk dd <%>\n/risk paper <USDT>")
            return
        sub = parts[1].lower()
        val = parts[2] if len(parts) > 2 else ""
        keymap = {"pertrade": ("per_trade", 0.01, 10.0), "portfolio": ("portfolio", 0.01, 50.0), "direction": ("direction", 0.01, 50.0), "dd": ("dd", 0.1, 50.0), "paper": ("paper_equity", 1.0, 1e9)}
        if sub == "mode":
            if val.lower() in {"legacy", "legacy_notional"}:
                cfg["mode"] = "LEGACY_NOTIONAL"
            elif val.lower() in {"budget", "risk_budget"}:
                cfg["mode"] = "RISK_BUDGET"
                err = self._validate_risk_cfg()
                if err:
                    await self.reply(f"❌ {err}")
                    return
            else:
                await self.reply("Gunakan /risk mode legacy atau /risk mode budget.")
                return
        elif sub in keymap:
            key, lo, hi = keymap[sub]
            try:
                num = float(val)
                if not math.isfinite(num) or num < lo or num > hi:
                    raise ValueError
            except ValueError:
                await self.reply(f"❌ Nilai tidak valid. Rentang {lo} – {hi}.")
                return
            cfg[key] = num
        else:
            await self.reply("Sub-perintah tidak dikenal. Ketik /risk.")
            return
        await self._save_runtime_settings()
        warn = "\n⚠️ RISK_BUDGET aktif: qty dihitung dari rencana loss di SL; trade yang tak muat anggaran ditolak (bukan dinaikkan)." if cfg["mode"] == "RISK_BUDGET" else ""
        await self.reply(f"✅ Risk diperbarui: mode {cfg['mode']}.{warn}")

    async def _run_scan_cycle(self) -> None:
        self._scan_cycle_active = True
        try:
            await self._run_scan_cycle_inner()
        finally:
            self._scan_cycle_active = False
            await self._flush_bans_if_dirty()
            await self._flush_research()

    async def _run_scan_cycle_inner(self) -> None:
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
        self._strategy_module_ref = module
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
        if btc_trend in {"BULLISH", "BEARISH", "RANGE"}:
            self._purge_stale_structure_bans(btc_trend)
            self._sweep["done"] -= set(self._recheck.invalidate_btc_mismatch_pairs(btc_trend))

        universe = await self._build_scan_universe()
        rows = universe["rows"]
        funnel = self._new_funnel()
        funnel["universe"] = len(rows)
        cycle_now = now_utc()
        now_ts = cycle_now.timestamp()
        cycle_id = f"{self.session_id}-{cycle}"
        self._recheck.purge(now_ts)
        self.ledger.expire_due(cycle_now)
        self._sync_ledger_trades()
        due_watch = self.ledger.due_watch(cycle_now, SCAN_WATCH_MAX_PER_CYCLE)
        active_now = {t.pair for t in self.active_trades.values()}
        for r_ in list(due_watch):
            if r_.pair in active_now or self._is_pair_banned(r_.pair) or self._recheck.active(r_.pair, now_ts) is not None:
                self.ledger.postpone(r_, SCAN_WATCHLIST_INTERVAL)
                due_watch.remove(r_)
        due_set = {r.pair for r in due_watch}
        waiting_pairs = {r.pair for r in self.ledger.items.values() if r.state in WAITING_CANDIDATE_STATES and not r.linked_trade_id}
        slot = int(now_ts // (4 * 3600))
        if self._sweep.get("slot") != slot:
            self._sweep = {"slot": slot, "done": set()}
        ready_due = {
            r.pair for r in self.ledger.items.values()
            if r.state == "READY" and not r.linked_trade_id
            and (_r_parse_iso(r.next_evaluate_after_utc) or cycle_now) <= cycle_now
        }
        sweep_targets: list[str] = []
        swept_skipped = 0
        watch_deferred = 0
        cooldown_count = 0
        final_risk_rejected = 0
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
            sweep_targets.append(pair)
            if pair in waiting_pairs and pair not in due_set:
                watch_deferred += 1
                continue
            if self._recheck.active(pair, now_ts) is not None:
                cooldown_count += 1
                continue
            if pair in self._sweep["done"] and pair not in due_set and pair not in ready_due:
                swept_skipped += 1
                continue

            eligible.append(pair)
        due_first = due_set | ready_due
        eligible = (
            [p for p in eligible if p in due_first]
            + [p for p in eligible if p not in due_first and p not in self._retry_counts]
            + [p for p in eligible if p not in due_first and p in self._retry_counts]
        )

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

                if time.monotonic() - started > SCAN_CYCLE_TIME_BUDGET_SECONDS:
                    funnel["budget_stopped"] += 1
                    log.info("[SCAN] Cycle #%s berhenti: anggaran waktu %.0fs habis.", cycle, SCAN_CYCLE_TIME_BUDGET_SECONDS)
                    break
                if self._resource.level == "CRITICAL" or (self._resource.level == "HIGH" and pair not in due_set):
                    funnel["memory_skipped"] += 1
                    continue
                if funnel["rate_limited"] >= 3:
                    log.info("[SCAN] Cycle #%s berhenti: rate-limit berulang.", cycle)
                    break
                headroom = self._api_headroom_pct()
                if headroom is not None and headroom < SCAN_MIN_API_HEADROOM_PCT and pair not in due_set:
                    funnel["api_stopped"] = funnel.get("api_stopped", 0) + 1
                    log.info("[SCAN] Cycle #%s berhenti: headroom API %.0f%% < %.0f%%.", cycle, headroom, SCAN_MIN_API_HEADROOM_PCT)
                    break

                scanned += 1
                self._sweep["done"].add(pair)
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
                        funnel["structure_mismatch"] += 1
                        funnel["quality_rejects"] += 1
                        self._recheck.set(pair, next_boundary_ts(now_ts, 4 * 3600), "STRUCTURE_MISMATCH", "REJECTED_QUALITY", btc_trend=btc_trend)
                        self._observe_candidate(synthetic_observation(
                            pair, "REJECTED_QUALITY", "STRUCTURE_MISMATCH", cycle_id=cycle_id, now=cycle_now,
                            btc_h4_regime=btc_trend, pair_h4_regime=pair_trend,
                        ))
                        log.info(
                            "[SCAN] %s rejected + recheck s/d close H4 | pair=%s BTC=%s",
                            pair,
                            pair_trend,
                            btc_trend,
                        )
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue

                    directional.append(pair)
                    funnel["structure_analyzed"] += 1
                    funnel["direction_thesis"] += 1

                    candidate = reused_setup
                    if candidate is None:
                        candidate = await self._scan_generate_for_pair(module, pair, cycle)
                    funnel["candidates_evaluated"] += 1

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
                    cstate = str(candidate.get("candidate_state") or "READY")
                    cand_rec = build_candidate_from_setup(pair, candidate, cycle_id=cycle_id, now=cycle_now, strategy_name="SMC_VLT_RSI")
                    self._retry_counts.pop(pair, None)
                    candidate["_candidate_id"] = cand_rec.candidate_id
                    if cstate in {"DATA_UNAVAILABLE", "REJECTED_HARD", "REJECTED_QUALITY"}:
                        key = {"DATA_UNAVAILABLE": "data_unavailable", "REJECTED_HARD": "hard_rejects"}.get(cstate, "quality_rejects")
                        funnel[key] += 1
                        if cstate == "DATA_UNAVAILABLE":
                            funnel["data_stale"] += 1
                        self._observe_candidate(cand_rec)
                        self._recheck.set(pair, next_evaluate_ts(cstate, now_ts, self._retry_counts.get(pair, 0)), cand_rec.state_reason or cstate, cstate)
                        log.info("[SCAN] %s %s (%s) | confidence=%s", pair, cstate, cand_rec.state_reason, decimal_to_str(confidence))
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue
                    if confidence < self.scan_threshold:
                        threshold_banned += 1
                        gates = (
                            ((candidate.get("analysis") or {}).get("selected_setup") or {}).get("evidence")
                            or {}
                        ).get("gates") or {}
                        waiting = bool(gates.get("waiting"))
                        near = float(confidence) >= float(self.scan_threshold) - RESEARCH_NEAR_THRESHOLD_POINTS
                        if waiting or cstate == "WAITING_TRIGGER":
                            new_state, why = "WAITING_TRIGGER", "TRIGGER_NOT_CONFIRMED"
                            funnel["waiting_trigger"] += 1
                        elif near:
                            new_state, why = "WAITING_PRICE", "BELOW_THRESHOLD_WATCH"
                            funnel["waiting_price"] += 1
                        else:
                            new_state, why = "REJECTED_QUALITY", "BELOW_THRESHOLD"
                            funnel["quality_rejects"] += 1
                        cand_rec.state, cand_rec.state_reason = new_state, why
                        cand_rec.next_evaluate_after_utc = _r_iso(datetime.fromtimestamp(next_evaluate_ts(new_state, now_ts), tz=timezone.utc))
                        self._observe_candidate(cand_rec)
                        if new_state == "REJECTED_QUALITY":
                            self._recheck.set(pair, next_boundary_ts(now_ts, 4 * 3600), "BELOW_THRESHOLD", new_state)
                        log.info(
                            "[SCAN] %s below threshold -> %s (%s) | confidence=%s threshold=%s",
                            pair, new_state, why, decimal_to_str(confidence), decimal_to_str(self.scan_threshold),
                        )
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue

                    threshold_candidates.append(candidate)
                    funnel["ready"] += 1
                    self._observe_candidate(cand_rec)
                    log.info(
                        "[SCAN] %s threshold PASS confidence=%s",
                        pair,
                        decimal_to_str(confidence),
                    )
                except Exception as exc:
                    analysis_errors += 1
                    self._sweep["done"].discard(pair)
                    err_cat = classify_error(exc)
                    funnel["errors"][err_cat] = funnel["errors"].get(err_cat, 0) + 1
                    if err_cat == ErrorCategory.RATE_LIMIT or getattr(exc, "bybit_rate_limited", False):
                        funnel["rate_limited"] += 1
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
                        funnel["data_unavailable"] += 1
                        self._recheck.set(pair, now_ts + float(SCAN_BANNED_HISTORY_HOURS) * 3600.0, "INSUFFICIENT_HISTORY", "DATA_UNAVAILABLE")
                        self._observe_candidate(synthetic_observation(pair, "DATA_UNAVAILABLE", "DATA_INSUFFICIENT", cycle_id=cycle_id, now=cycle_now))
                        log.info("[SCAN] %s dilewati, riwayat candle kurang | %s", pair, exc)
                    else:
                        log.exception("[SCAN] analysis error %s", pair)
                        retries = self._retry_counts[pair] = min(self._retry_counts.get(pair, 0) + 1, 6)
                        funnel["data_unavailable"] += 1
                        self._recheck.set(pair, next_evaluate_ts("DATA_UNAVAILABLE", now_ts, retries), err_cat, "DATA_UNAVAILABLE")
                        self._observe_candidate(synthetic_observation(pair, "DATA_UNAVAILABLE", err_cat, cycle_id=cycle_id, now=cycle_now, retries=retries))

                await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)

        # Kandidat yang threshold-pass selalu divalidasi berdasarkan confidence
        # terbesar terlebih dahulu.
        threshold_candidates.sort(
            key=lambda item: Decimal(str(item.get("confidence") or "0")),
            reverse=True,
        )
        threshold_candidates.sort(key=self._priority_of, reverse=True)
        if len(threshold_candidates) > SCAN_VALIDATE_TOP_N:
            funnel["validation_skipped_topn"] = len(threshold_candidates) - SCAN_VALIDATE_TOP_N
            threshold_candidates = threshold_candidates[:SCAN_VALIDATE_TOP_N]
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

            if time.monotonic() - started > SCAN_CYCLE_TIME_BUDGET_SECONDS:
                funnel["budget_stopped"] += 1
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
                    funnel["validation_reject"] += 1
                    self._after_validation(candidate, "VALIDATION_REJECT", "REJECTED_QUALITY", now_ts)
                else:
                    checked["_candidate_id"] = candidate.get("_candidate_id")
                    validated.append(checked)
                    funnel["validation_pass"] += 1
            except Exception as exc:
                validator_rejects += 1
                v_cat = classify_error(exc)
                funnel["validation_error"] += 1
                funnel["errors"][v_cat] = funnel["errors"].get(v_cat, 0) + 1
                self._after_validation(candidate, v_cat, "DATA_UNAVAILABLE" if v_cat != ErrorCategory.STRATEGY_CONTRACT_ERROR else "REJECTED_QUALITY", now_ts)
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
        final_balance_note: str | None = None

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
                    "strategy_analysis": compact_analysis(candidate.get("analysis")),
                }
                meta["margin_usdt"] = self.margin_usdt
                meta["leverage"] = self.leverage
                trade = self._build_trade_from_setup(
                    candidate,
                    strategy_meta=meta,
                )
                trade.candidate_id = str(candidate.get("_candidate_id") or "")
                trade.sizing_mode = self._risk_cfg["mode"]
                if self._risk_cfg["mode"] == "RISK_BUDGET":
                    risk_ok, risk_info = self._risk_gate_for_candidate(trade)
                    trade.risk_info = risk_info
                    if not risk_ok:
                        final_risk_rejected += 1
                        funnel["risk_rejects"] += 1
                        self.ledger.annotate(trade.candidate_id, "RISK_REJECTED", f"RISK_REJECT:{risk_info.get('reason')}")
                        log.info("[SCAN] %s ditolak risk gate: %s", trade.pair, risk_info.get("reason"))
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue
                if not self.real_mode and not self._sim_has_margin(trade):
                    funnel["sim_margin_blocked"] += 1
                    self.ledger.annotate(trade.candidate_id, "SIM_MARGIN_BLOCKED", "SIM_MARGIN_BLOCKED")
                    log.info("[SCAN] %s ditahan: margin simulasi tidak cukup.", trade.pair)
                    await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                    continue
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
                    except InsufficientBalanceError as exc:
                        # Kondisi akun, bukan sifat pair: tanpa ban; hentikan entry real siklus ini.
                        final_balance_note = str(exc).split(": ", 1)[-1]
                        log.info("[SCAN] saldo tersedia kurang, entry real ditunda: %s", exc)
                        break
                    except MarginInfluenceError as exc:
                        hint = self._margin_hint(trade.pair, trade.entry)
                        await self._ban_pair(
                            trade.pair,
                            hours=None,
                            reason=(
                                f"Real quantity tidak memenuhi target margin ±10%: {exc}"
                                + (f" Margin pas ≈ {hint}." if hint else "")
                            ),
                            source="AUTO_MARGIN_INFLUENCE",
                        )
                        final_margin_banned.append(trade.pair)
                        log.info("[SCAN] %s setup dibatalkan, diban permanen (margin influence): %s", trade.pair, exc)
                        await asyncio.sleep(SCAN_PAIR_DELAY_SECONDS)
                        continue
                    except RiskRejectError as exc:
                        final_risk_rejected += 1
                        funnel["risk_rejects"] += 1
                        self.ledger.annotate(trade.candidate_id, "RISK_REJECTED", f"RISK_REJECT:{exc}")
                        log.info("[SCAN] %s ditolak risk sizing: %s", trade.pair, exc)
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
                if trade.candidate_id:
                    self.ledger.link_trade(trade.candidate_id, trade.trade_id, "PENDING")
                funnel["pending_created"] += 1
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

        funnel["usable_data"] = max(0, scanned - analysis_errors)
        target_set = set(sweep_targets)
        analyzed_cd = {
            pair_ for pair_, cd_ in self._recheck.items.items()
            if cd_["until"] > now_ts and (cd_.get("state") != "DATA_UNAVAILABLE" or cd_.get("reason") == "INSUFFICIENT_HISTORY")
        }
        funnel["sweep_covered"] = len(target_set & (self._sweep["done"] | analyzed_cd | waiting_pairs))
        funnel["sweep_total"] = len(target_set)
        sweep_rest = funnel["sweep_total"] - funnel["sweep_covered"]
        sweep_pct = round(funnel["sweep_covered"] / funnel["sweep_total"] * 100) if funnel["sweep_total"] else 100
        if sweep_rest == 0 and funnel["sweep_total"]:
            self._sweep["completed_at"] = self._sweep.get("completed_at") or iso_utc()
        self._scan_durations.append(time.monotonic() - started)
        self._funnel_last = funnel
        self._funnel_history.append({"cycle": cycle, "at": iso_utc(), "duration_s": round(self._scan_durations[-1], 1), **{k: v for k, v in funnel.items() if k != "errors"}, "errors": funnel["errors"]})
        self._last_cycle_analyzed = scanned
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
        balance_text = (
            f"├ Saldo kurang     entry real ditunda ({final_balance_note})\n" if final_balance_note else ""
        )
        await self.reply(
            f"╭─ 🔄 SCAN #{cycle} SELESAI ─╮\n"
            f"│ ₿ BTC H4   {btc_trend}\n"
            f"│ ⏱ {duration:.0f}s  •  siklus berikut {self._next_scan_delay():.0f}s\n"
            f"│ 💾 RAM {(f'{rss_mb():.0f} MB' if rss_mb() > 0 else '-')}\n"
            "╰──────────────────╯\n\n"
            "🌐 UNIVERSE\n"
            f"├ Binance ∩ Bybit  {universe['common_count']}\n"
            f"├ Banned           {banned_count}\n"
            f"├ Sudah di /trade  {already_trade}\n"
            f"└ Eligible         {len(eligible)}  (batch {len(batch)}/{SCAN_MAX_PAIRS_PER_CYCLE}, tunda {deferred})\n\n"
            "🧭 ARAH vs BTC\n"
            f"├ Searah {len(directional)}  •  tidak searah {rejected_structure} (recheck s/d close H4)\n"
            f"└ H4  🟢 {direction_counts['BULLISH']}  🔴 {direction_counts['BEARISH']}"
            f"  ⚪ {direction_counts['RANGE']}  ❓ {direction_counts['UNKNOWN']}\n\n"
            "🔬 ANALISIS\n"
            f"├ Dipindai         {scanned}/{SCAN_MAX_PAIRS_PER_CYCLE}\n"
            f"├ ≥ Threshold      {len(threshold_candidates)}  (di bawah {threshold_banned}, watchlist/recheck)\n"
            f"├ Validator lolos  {len(validated)}  (tolak {validator_missing + validator_rejects})\n"
            f"├ Masuk /trade     {len(final_added)}  (error {final_add_errors}, dibatasi {final_capped})\n"
            f"├ Watch/cooldown   {len(waiting_pairs)} (tunda {watch_deferred}) / {cooldown_count}  • risk-reject {final_risk_rejected}\n"
            f"├ Tersapu H4 ini   {funnel['sweep_covered']}/{funnel['sweep_total']} ({sweep_pct}%) • sisa {sweep_rest}" + (f" (~{-(-sweep_rest // SCAN_MAX_PAIRS_PER_CYCLE)} siklus)" if sweep_rest else " ✅") + "\n"
            f"{margin_line}\n"
            f"{leverage_line}\n"
            f"{balance_text}"
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
            "analysis": self._json_safe(compact_analysis(analysis)),
            "raw_result": self._json_safe(
                {k: v for k, v in compact_analysis(payload).items() if k != "analysis"}
            ),
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
                "analysis": compact_analysis(result.get("analysis")),
                "raw_result": compact_analysis(result.get("raw_result")),
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
                compact_analysis(meta.get("strategy_analysis"))
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
                pnl_usdt = self._temporary_trade_pnl_usdt(trade, snapshot.price)
                if pnl_usdt is not None:
                    margin_used = (
                        trade.actual_notional / Decimal(trade.leverage)
                        if trade.actual_notional and trade.leverage
                        else trade.margin_usdt
                    )
                    sign = "+" if pnl_usdt >= 0 else ""
                    pnl_text += f"  (≈{sign}{self._fmt_usd(pnl_usdt)} USDT"
                    if margin_used > 0:
                        pnl_text += f" • ROI {format_pct(pnl_usdt / margin_used * Decimal('100'))}"
                    pnl_text += ")"
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

        if self.real_mode and (
            self.real.last_balance_at is None
            or time.monotonic() - self.real.last_balance_at > BALANCE_STALE_SECONDS
        ):
            await self._refresh_balance_cache()

        real_filled = sum(
            1
            for trade in self.active_trades.values()
            if trade.status == "FILLED" and trade.real_enabled
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
            f"🧊 Balance Cache       {balance_cache_age} ago • {temp_pnl_count}/{real_filled} REAL FILLED",
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
                if not current.real_enabled:
                    self._sim_apply_close(current)

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
        if self._sim_dirty:
            await self._save_sim_account()

        if trade.real_enabled and result in {"TP", "SL", "TRAIL"}:
            await self._cleanup_real_leftovers(trade)

        if getattr(trade, "candidate_id", ""):
            self.ledger.set_state(
                trade.candidate_id,
                "EXPIRED" if result in {"EXPIRED", "DELETED"} and trade.fill_price is None else "CLOSED",
                "TRADE_CLOSED", str(result),
            )

        # PnL terealisasi mengubah saldo wallet; cache lama bikin saldo/autostop meleset.
        if trade.real_enabled and trade.fill_price is not None:
            await self._refresh_balance_cache()

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

        loss_sum = sum(losses, Decimal("0"))
        profit_factor = (
            sum(wins, Decimal("0")) / abs(loss_sum) if loss_sum < 0 else None
        )
        expectancy = gross_net / Decimal(len(pnls)) if pnls else None

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
            "profit_factor": profit_factor,
            "expectancy_percent": expectancy,
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
        stats = self._calculate_stats(self.history_records)

        active_pending = sum(1 for t in self.active_trades.values() if t.status == "PENDING")
        active_filled = sum(1 for t in self.active_trades.values() if t.status == "FILLED")

        def num(value: Decimal | None) -> str:
            return "-" if value is None else decimal_to_str(value)

        def pct(value: Decimal) -> str:
            return format_pct(value).replace("+", "")

        def dur(value: Decimal | None) -> str:
            return "-" if value is None else duration_text(float(value))

        gap = stats["confidence_gap_tp_sl"]
        if gap is None:
            confidence_conclusion = "Belum cukup data confidence di kedua outcome."
        elif gap > 0:
            confidence_conclusion = f"Rerata confidence TP lebih tinggi {decimal_to_str(gap)} poin dari SL."
        elif gap < 0:
            confidence_conclusion = f"Rerata confidence TP lebih rendah {decimal_to_str(abs(gap))} poin dari SL."
        else:
            confidence_conclusion = "Rerata confidence TP dan SL sama."

        if stats["total_filled"]:
            outcome_conclusion = (
                f"Dari {stats['total_filled']} trade TP/SL: "
                f"TP {stats['tp']} ({pct(stats['tp_rate'])}), "
                f"SL {stats['sl']} ({pct(stats['sl_rate'])})."
            )
        else:
            outcome_conclusion = "Belum ada trade yang berakhir TP/SL."

        coverage = (
            Decimal(stats["confidence_count_entered"]) / Decimal(stats["entered_records_count"]) * Decimal("100")
            if stats["entered_records_count"]
            else Decimal("0")
        )

        if stats["profit_factor"] is not None:
            pf_text = f"{float(stats['profit_factor']):.2f}"
        elif stats["average_win_percent"] > 0:
            pf_text = "∞ (belum ada loss)"
        else:
            pf_text = "-"
        expectancy = stats["expectancy_percent"]
        expectancy_text = "-" if expectancy is None else f"{format_pct(expectancy)} / trade"

        lines = [
            card(
                "📊 STATS",
                [
                    f"🗂 Setup {stats['total_records']}  •  ✅ Entry {stats['entered_records_count']}  •  🏁 Selesai {stats['total_filled']}",
                    f"🟡 Pending {active_pending}  •  🟢 Filled {active_filled}",
                ],
            ),
            "",
            "🎯 OUTCOME",
            f"├ Win Rate   {gauge(stats['win_rate'])}  {pct(stats['win_rate'])}",
            f"├ ✅ TP       {stats['tp']}  ({pct(stats['tp_rate'])})  ↳ TRAIL {stats.get('trail_exit', 0)}",
            f"├ ❌ SL       {stats['sl']}  ({pct(stats['sl_rate'])})",
            f"├ ⌛ Expired  {stats['expired']}  ({pct(stats['expired_rate_all'])} history)",
            f"├ 🗑 Deleted  {stats['deleted']}  ({pct(stats['deleted_rate_all'])} history)",
            f"└ ✋ Manual   pending {stats['manual_close_pending']}  •  filled@0% {stats['manual_close_filled']}",
            "",
            "💰 PERFORMA",
            f"├ Net PnL        {format_pct(stats['gross_pnl_percent'])}",
            f"├ Avg Win        {format_pct(stats['average_win_percent'])}",
            f"├ Avg Loss       {format_pct(stats['average_loss_percent'])}",
            f"├ Profit Factor  {pf_text}",
            f"├ Expectancy     {expectancy_text}",
            f"└ Avg Planned RR {num(stats['average_planned_rr_entered'])}",
            "",
            "🧠 CONFIDENCE",
            f"├ Semua history  {num(stats['average_confidence_all'])}",
            f"├ TP  {num(stats['average_confidence_tp'])}  •  SL  {num(stats['average_confidence_sl'])}",
            f"├ Setup entry    {num(stats['average_confidence_entered'])}",
            f"└ Coverage       {gauge(coverage)}  {decimal_to_str(coverage)}%  (tanpa data {stats['confidence_missing_entered']})",
            "",
            "⏱ WAKTU",
            f"├ Pending  {dur(stats['average_pending_entered_seconds'])}",
            f"├ Hold TP  {dur(stats['average_holding_tp_seconds'])}",
            f"└ Hold SL  {dur(stats['average_holding_sl_seconds'])}",
            "",
            "🧬 SUMBER",
            f"└ AUTO {stats['auto_records']}  •  MANUAL {stats['manual_records']}  •  Trail event {stats['trail_count']}",
            "",
            "🔬 KESIMPULAN",
            f"• {outcome_conclusion}",
            f"• {confidence_conclusion}",
            "• Expired, Deleted, dan manual close non-TP/SL tidak dihitung win/loss.",
        ]
        lines += self._stats_r_lines()
        await self.reply("\n".join(lines))

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
        full_data["research"] = self._research_summary()
        report += self._research_report_md()

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
    # RESET STATS
    # --------------------------------------------------------

    async def _start_resetstats(self) -> None:
        if self.flow is not None:
            await self.reply(
                "Masih ada sesi yang sedang berjalan.\n"
                "Gunakan /back terlebih dahulu."
            )
            return

        self.flow = {"kind": "RESETSTATS", "step": "CONFIRM"}
        await self.reply(
            card(
                "⚠️ RESET STATS",
                [
                    "📦 Diarsipkan ke data/archive/, lalu dihapus:",
                    "• trades.json • events.jsonl • trade_history.md",
                    "• hasil /analyze",
                    "✅ Catatan, ban, dan active setup tetap aman",
                ],
            )
            + "\n\n1. Ya, reset stats\n2. Batal"
        )

    async def _handle_resetstats_input(self, text: str) -> None:
        answer = str(text or "").strip()
        if answer == "2":
            self.flow = None
            await self.reply("✅ RESET STATS dibatalkan.")
            return
        if answer != "1":
            await self.reply("Jawab 1 untuk reset stats atau 2 untuk batal.")
            return

        self.flow = None
        try:
            await self.reset_stats_records()
        except Exception as exc:
            log.exception("RESET STATS gagal.")
            await self.reply(f"❌ /resetstats gagal. Tidak ada data yang dihapus bila arsip gagal.\n\n{exc}")

    async def reset_stats_records(self) -> None:
        """Arsipkan histori ke data/archive/<waktu>/ lalu hapus; catatan tidak disentuh."""
        stamp = now_local().strftime("%Y%m%d-%H%M%S")
        history_paths = (HISTORY_TRADES_PATH, HISTORY_EVENTS_PATH, HISTORY_MARKDOWN_PATH)
        archived: list[str] = []
        deleted = 0

        async with self._history_lock:
            # Arsip semua dulu; gagal satu = batal tanpa menghapus apa pun.
            for path in history_paths:
                raw, _ = await self.github.get_file(path)
                if not raw:
                    continue
                target = f"data/archive/{stamp}/{Path(path).name}"
                await self.github.replace_file(target, raw, f"archive: {path} sebelum reset stats")
                archived.append(target)

            for path in (*history_paths, ANALYSIS_JSON_PATH, ANALYSIS_MD_PATH):
                if await self.github.delete_file(path, f"resetstats: delete {path}"):
                    deleted += 1

            self.history_records.clear()
            self.history_events.clear()
            self._history_dirty = False
            self.last_history_refresh = now_utc()

        await self.reply(
            card(
                "♻️ STATS DIRESET",
                [
                    f"📦 Arsip    {len(archived)} file" + (f" → data/archive/{stamp}/" if archived else ""),
                    f"🗑 Dihapus  {deleted} file",
                    "📊 Stats    mulai dari 0",
                ],
            )
            + "\n\nActive setup /trade tidak disentuh; trade lama yang masih aktif tetap tercatat saat selesai."
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
            "/stats - statistik histori + R-metrik\n"
            "/drawdown [%|off|reset] - tutup SEMUA /trade bila equity turun dari puncak\n"
            "/balance - saldo (simulasi $100 atau real)\n"
            "/resetbalance - reset saldo simulasi ke $100\n"
            "/candidates [PAIR] - kandidat, funnel scan, alasan per pair\n"
            "/edge - R, bucket confidence, sapuan threshold, varian\n"
            "/research on|off|status - replay M1 (default OFF)\n"
            "/risk - sizing berbasis risiko (default LEGACY)\n"
            "/resetstats - arsipkan + reset stats (untuk tes strategy baru)\n"
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
                    "/resetstats",
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
                await self.reply(self._api_report() + self._api_extra_report())
                return

            if command == "/autostop":
                try:
                    await self._handle_autostop_command(text)
                    try:
                        await self._save_runtime_settings()
                    except Exception as exc:
                        log.warning("[AUTOSTOP] pengaturan gagal disimpan: %s", exc)
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

            if command == "/resetstats":
                await self._start_resetstats()
                return

            if command == "/status":
                await self.show_status()
                await self.reply(self._status_extra_report())
                return

            if command == "/drawdown":
                try:
                    await self._handle_drawdown_command(text)
                except Exception as exc:
                    log.exception("DRAWDOWN command gagal.")
                    await self.reply(f"❌ /drawdown gagal.\n\n{exc}")
                    return
                try:
                    await self._save_runtime_settings()
                except Exception as exc:
                    log.warning("[DRAWDOWN] pengaturan gagal disimpan: %s", exc)
                return

            if command == "/resetbalance":
                try:
                    await self._handle_resetbalance_command()
                except Exception as exc:
                    log.exception("RESETBALANCE command gagal.")
                    await self.reply(f"❌ /resetbalance gagal.\n\n{exc}")
                return

            if command == "/balance":
                await self._show_balance()
                return

            if command == "/candidates":
                arg = text.split()[1] if len(text.split()) > 1 else ""
                if arg:
                    await self.show_candidate_detail(arg)
                else:
                    await self.show_candidates()
                return

            if command == "/research":
                try:
                    await self._handle_research_command(text)
                except Exception as exc:
                    log.exception("RESEARCH command gagal.")
                    await self.reply(f"❌ /research gagal.\n\n{exc}")
                return

            if command == "/edge":
                try:
                    await self.reply(await self._build_edge_text())
                except Exception as exc:
                    log.exception("EDGE command gagal.")
                    await self.reply(f"❌ /edge gagal.\n\n{exc}")
                return

            if command == "/risk":
                try:
                    await self._handle_risk_command(text)
                except Exception as exc:
                    log.exception("RISK command gagal.")
                    await self.reply(f"❌ /risk gagal.\n\n{exc}")
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

            if self.flow["kind"] == "RESETSTATS":
                await self._handle_resetstats_input(text)
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
