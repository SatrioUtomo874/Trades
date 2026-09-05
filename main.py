"""
main.py — Adaptive Trading Bot — Infrastruktur Utama
======================================================================
Tanggung jawab (§1 & §65):
    - SATU-SATUNYA modul yang boleh mengakses API market (Bybit & Binance)
      dan WebSocket. strategy.py & learn.py TIDAK PERNAH memanggil API.
    - Trading engine, Telegram command handler, position/pending
      management, risk management, state management, worker orchestration.

Kredensial WAJIB diambil dari environment variable (.env), TIDAK PERNAH
ditulis langsung ke source code (§61). Jalankan dengan file `.env` di
folder yang sama (lihat `.env.example`), atau export manual sebelum start:

    BINANCE_API_KEY=...
    BINANCE_API_SECRET=...
    BYBIT_API_KEY=...           # opsional untuk endpoint publik
    BYBIT_API_SECRET=...        # opsional untuk endpoint publik
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...
    OLLAMA_URL=http://localhost:11434     # opsional
    OLLAMA_MODEL=llama3                   # opsional
    GITHUB_TOKEN=...                       # opsional, untuk git autosave
    GIT_AUTOSAVE=false                     # true/false

PENTING: bot ini SELALU terhubung ke Binance Futures MAINNET (uang
sungguhan) — tidak ada mode testnet. Pastikan BINANCE_API_KEY/SECRET
adalah API key Binance asli dengan permission Futures aktif, dan IP
server sudah masuk whitelist key tersebut (kalau key diberi IP restriction).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import logging.handlers
import math
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()  # muat file .env di direktori kerja (§61 — kredensial via env var)
except ImportError:
    pass  # python-dotenv opsional; env var manual tetap berfungsi

import requests

try:
    import websocket  # pip install websocket-client
except ImportError:
    websocket = None  # akan dicek saat startup; simulasi tetap bisa jalan tanpa WS live

import strategy as strategy
import learn as learn


# =============================================================================
# 0. CONFIG & LOGGING
# =============================================================================

def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    binance_api_key: str = field(default_factory=lambda: os.environ.get("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.environ.get("BINANCE_API_SECRET", ""))
    bybit_api_key: str = field(default_factory=lambda: os.environ.get("BYBIT_API_KEY", ""))
    bybit_api_secret: str = field(default_factory=lambda: os.environ.get("BYBIT_API_SECRET", ""))
    telegram_bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    allowed_user_id: str = field(default_factory=lambda: os.environ.get("ALLOWED_USER_ID", ""))
    ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ollama_api_key: str = field(default_factory=lambda: os.environ.get("OLLAMA_API_KEY", ""))
    github_token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    git_autosave: bool = field(default_factory=lambda: _env_bool("GIT_AUTOSAVE", False))
    state_dir: str = field(default_factory=lambda: os.environ.get("STATE_DIR", "state"))

    def __post_init__(self) -> None:
        # §61 revisi — BUG LAMA: telegram_chat_id sempat ke-tuker baca dari
        # ALLOWED_USER_ID sehingga TELEGRAM_CHAT_ID tidak pernah kepakai dan
        # notifikasi (termasuk balance/resetbalance/error) tidak pernah
        # terkirim kalau ALLOWED_USER_ID belum diisi. Sekarang keduanya
        # independen; TAPI untuk bot personal (1 user), chat_id == user_id,
        # jadi kalau TELEGRAM_CHAT_ID kosong, fallback otomatis ke
        # ALLOWED_USER_ID (cukup isi salah satu di .env).
        if not self.telegram_chat_id and self.allowed_user_id:
            self.telegram_chat_id = self.allowed_user_id
        if not self.allowed_user_id and self.telegram_chat_id:
            self.allowed_user_id = self.telegram_chat_id

    def validate_for_real_mode(self) -> List[str]:
        missing = []
        if not self.binance_api_key:
            missing.append("BINANCE_API_KEY")
        if not self.binance_api_secret:
            missing.append("BINANCE_API_SECRET")
        return missing

    def validate_telegram(self) -> List[str]:
        missing = []
        if not self.telegram_bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.telegram_chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        return missing


class TerminalFormatter(logging.Formatter):
    """Terminal operational log yang ringkas dan mudah dibaca."""

    def format(self, record: logging.LogRecord) -> str:
        seq = getattr(record, "log_seq", None)
        if seq is None:
            seq = getattr(logging, "_adaptive_log_seq", 0) + 1
            logging._adaptive_log_seq = seq

        symbol = getattr(record, "symbol", None) or getattr(record, "coin", None)
        scope = f"[{str(symbol).upper()}]" if symbol else "[GLOBAL]"
        level = record.levelname
        message = record.getMessage().replace("\n", " | ")
        return f"[{int(seq):05d}] {scope} [{level}] {message}"


def setup_logging(state_dir: str) -> logging.Logger:
    os.makedirs(state_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    terminal_fmt = TerminalFormatter()
    has_console = False
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.INFO)
            handler.setFormatter(terminal_fmt)
            has_console = True

    if not has_console:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(terminal_fmt)
        root.addHandler(console)

    # Library network DEBUG tidak boleh bocor ke terminal.
    for name in ("urllib3", "requests", "websocket", "websocket._logging"):
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = True

    has_file = any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, "_adaptive_bot_file_handler", False)
        for h in root.handlers
    )
    if not has_file:
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(state_dir, "bot.log"), maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        file_handler._adaptive_bot_file_handler = True
        root.addHandler(file_handler)

    root._adaptive_bot_configured = True
    return root


logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Runtime compatibility guard
# ---------------------------------------------------------------------------
# The launcher syncs the GitHub repository before loading MAIN_FILE.  A
# versioned module name such as ``strategy_vnext2`` can therefore crash before
# the bot even starts when that auxiliary file was not committed.  Keep the
# runtime dependency names stable (strategy.py / learn.py) and patch only the
# deterministic helper symbols that older strategy.py revisions may lack.
def _ensure_strategy_runtime_compat() -> None:
    if not hasattr(strategy, "classify_volatility_regime"):
        def classify_volatility_regime(candles, params):
            if not candles:
                return "NORMAL"
            try:
                period = max(2, int(params.get("atr_period", 14)))
                lookback = max(period + 5, int(params.get("vol_regime_lookback", 100)))
            except Exception:
                period, lookback = 14, 100
            work = list(candles)[-lookback:]
            closes_fn = getattr(strategy, "_closes", None)
            atr_fn = getattr(strategy, "atr_series", None)
            if not callable(closes_fn) or not callable(atr_fn):
                return "NORMAL"
            try:
                closes = closes_fn(work)
                atrs = atr_fn(work, period)
            except Exception:
                return "NORMAL"
            if not closes or not atrs:
                return "NORMAL"
            try:
                price = float(closes[-1] or 0.0)
                atr_now = float(atrs[-1] or 0.0)
            except Exception:
                return "NORMAL"
            if price <= 0 or atr_now <= 0:
                return "NORMAL"
            try:
                atr_pct = (atr_now / price) * 100.0
                low_pct = float(params.get("low_vol_pct", 0.15))
                high_pct = float(params.get("high_vol_pct", 3.0))
            except Exception:
                return "NORMAL"
            if atr_pct < low_pct:
                return "LOW_VOLATILITY"
            if atr_pct > high_pct:
                return "HIGH_VOLATILITY"
            return "NORMAL"
        strategy.classify_volatility_regime = classify_volatility_regime

    if not hasattr(strategy, "validate_trailing_geometry"):
        def validate_trailing_geometry(direction, current_sl, proposed_sl, price, entry, tp):
            import math as _math
            try:
                cur = float(current_sl); new = float(proposed_sl)
                px = float(price); en = float(entry); target = float(tp)
            except Exception:
                return False, "NON_FINITE_TRAIL_GEOMETRY"
            if not all(_math.isfinite(x) for x in (cur, new, px, en, target)):
                return False, "NON_FINITE_TRAIL_GEOMETRY"
            d = str(direction or "").upper()
            if d == "BUY":
                if cur >= en and new < cur:
                    return False, "TRAIL_WOULD_REDUCE_PROTECTION"
                if new >= px:
                    return False, "TRAIL_SL_NOT_BELOW_PRICE"
                if target > en and new >= target:
                    return False, "TRAIL_SL_CROSSES_TP"
                if new <= 0 or en <= 0:
                    return False, "INVALID_PRICE_GEOMETRY"
                return True, "OK"
            if d == "SELL":
                if cur <= en and new > cur:
                    return False, "TRAIL_WOULD_REDUCE_PROTECTION"
                if new <= px:
                    return False, "TRAIL_SL_NOT_ABOVE_PRICE"
                if target < en and new <= target:
                    return False, "TRAIL_SL_CROSSES_TP"
                if new <= 0 or en <= 0:
                    return False, "INVALID_PRICE_GEOMETRY"
                return True, "OK"
            return False, "INVALID_DIRECTION"
        strategy.validate_trailing_geometry = validate_trailing_geometry

    required = ("Strategy", "Setup", "classify_regime", "validate_geometry")
    missing = [name for name in required if not hasattr(strategy, name)]
    if missing:
        raise RuntimeError(
            "strategy.py tidak kompatibel dengan main ini; simbol wajib hilang: "
            + ", ".join(missing)
        )


_ensure_strategy_runtime_compat()

# Binance REST safety governor. 418 means the IP was auto-banned after
# continued rate-limit violations; we therefore bias toward serialization,
# conservative headroom, and zero retries after 429/418.
BINANCE_POST_LIMIT_SAFETY_SECONDS = 15.0
# Binance reports REQUEST_WEIGHT against the source IP, not against this bot
# instance/API key.  Keep a large reserve so one new trade cannot be sent
# into an already-hot IP budget.
BINANCE_WEIGHT_HARD_STOP = 1800
BINANCE_WEIGHT_ORDER_SOFT_STOP = 1500
BINANCE_WEIGHT_RECOVERY_MARGIN = 300
BINANCE_DEFAULT_429_COOLDOWN = 75.0
# Trade-related REST calls are deliberately much slower than ordinary REST.
# This spacing covers leverage/order/cancel as one lane, not just POST /order.
BINANCE_ORDER_INTERVAL = 6.0
BINANCE_REQUEST_INTERVAL = 1.5


class SecretRedactingFilter(logging.Filter):
    """§61 — pastikan secret tidak pernah bocor ke log/telegram/traceback."""

    def __init__(self, secrets: Sequence[str]):
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for s in self._secrets:
            if s in msg:
                msg = msg.replace(s, "***REDACTED***")
        record.msg = msg
        record.args = ()
        return True


# =============================================================================
# 1. EXCEPTIONS
# =============================================================================

class RateLimitError(Exception):
    """Binance/Bybit rate-limit exception with recovery metadata."""

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 retry_after: Optional[float] = None,
                 banned_until_ms: Optional[int] = None,
                 code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.banned_until_ms = banned_until_ms
        self.code = code


class ExchangeError(Exception):
    pass


# =============================================================================
# 2. BYBIT REST + WEBSOCKET (sumber data market — §5)
# =============================================================================

class BybitClient:
    BASE_URL = "https://api.bybit.com"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()

    def _get(self, path: str, params: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
        url = self.BASE_URL + path
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code == 429:
                    raise RateLimitError("Bybit rate limit")
                data = resp.json()
                if data.get("retCode") not in (0, None):
                    raise ExchangeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
                return data
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise ExchangeError(f"Bybit request gagal setelah retry: {last_err}")

    def get_klines(self, symbol: str, interval: str = "15", limit: int = 672) -> List[Dict[str, float]]:
        """Ambil candle M15. §5: 672 candle ~ 7 hari data M15."""
        data = self._get(
            "/v5/market/kline",
            {"category": "linear", "symbol": symbol, "interval": interval, "limit": min(limit, 1000)},
        )
        rows = data.get("result", {}).get("list", [])
        candles = [
            {
                "t": float(r[0]),
                "o": float(r[1]),
                "h": float(r[2]),
                "l": float(r[3]),
                "c": float(r[4]),
                "v": float(r[5]),
            }
            for r in rows
        ]
        candles.sort(key=lambda c: c["t"])  # Bybit v5 mengembalikan newest-first
        return candles

    def get_ranked_symbols(self) -> List[Tuple[str, float]]:
        """Ranking berdasarkan turnover 24h (linear perpetual, pair *USDT)."""
        data = self._get("/v5/market/tickers", {"category": "linear"})
        rows = data.get("result", {}).get("list", [])
        ranked = []
        for r in rows:
            symbol = r.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            try:
                turnover = float(r.get("turnover24h", 0.0))
            except (TypeError, ValueError):
                turnover = 0.0
            ranked.append((symbol, turnover))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_last_price(self, symbol: str) -> Optional[float]:
        data = self._get("/v5/market/tickers", {"category": "linear", "symbol": symbol})
        rows = data.get("result", {}).get("list", [])
        if not rows:
            return None
        try:
            return float(rows[0]["lastPrice"])
        except (KeyError, TypeError, ValueError):
            return None


class BybitWebSocket:
    """WebSocket monitoring harga real-time (§16). Reconnect otomatis,
    heartbeat, tanpa duplicate subscription, aman dari race condition
    (semua akses `self.subscribed` dilindungi lock)."""

    URL = "wss://stream.bybit.com/v5/public/linear"

    def __init__(self, on_tick, on_kline):
        self.on_tick = on_tick
        self.on_kline = on_kline
        self._lock = threading.Lock()
        self.subscribed: set = set()
        self._ws = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat = time.time()

    def start(self) -> None:
        if websocket is None:
            logger.error("Package 'websocket-client' belum terinstal — WebSocket monitoring nonaktif. "
                         "Jalankan: pip install websocket-client")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="Worker2-WebSocket", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def _run_forever(self) -> None:
        backoff = 1
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    self.URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:  # pragma: no cover
                logger.warning("BybitWebSocket error: %s", e)
            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def invalidate_listen_key(self) -> None:
        with self._lock:
            self._listen_key = None

    def _on_open(self, ws) -> None:
        logger.info("BybitWebSocket terhubung")
        with self._lock:
            symbols = list(self.subscribed)
        if symbols:
            self._send_subscribe(symbols)

    def _send_subscribe(self, symbols: Sequence[str]) -> None:
        args = []
        for s in symbols:
            args.append(f"tickers.{s}")
            args.append(f"kline.15.{s}")
        try:
            self._ws.send(json.dumps({"op": "subscribe", "args": args}))
        except Exception as e:
            logger.warning("Gagal subscribe WS: %s", e)

    def subscribe(self, symbol: str) -> None:
        with self._lock:
            if symbol in self.subscribed:
                return  # cegah duplicate subscription (§16)
            self.subscribed.add(symbol)
        if self._ws:
            self._send_subscribe([symbol])

    def unsubscribe(self, symbol: str) -> None:
        with self._lock:
            self.subscribed.discard(symbol)
        if self._ws:
            try:
                self._ws.send(json.dumps({"op": "unsubscribe", "args": [f"tickers.{symbol}", f"kline.15.{symbol}"]}))
            except Exception:
                pass

    def _on_message(self, ws, message: str) -> None:
        self._last_heartbeat = time.time()
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        topic = data.get("topic", "")
        if topic.startswith("tickers."):
            symbol = topic.split(".", 1)[1]
            payload = data.get("data", {})
            price = payload.get("lastPrice")
            if price is not None:
                try:
                    self.on_tick(symbol, float(price), data.get("ts", time.time() * 1000))
                except Exception as e:
                    logger.error("on_tick handler error: %s", e)
        elif topic.startswith("kline."):
            symbol = topic.split(".")[-1]
            for k in data.get("data", []):
                try:
                    candle = {
                        "t": float(k["start"]), "o": float(k["open"]), "h": float(k["high"]),
                        "l": float(k["low"]), "c": float(k["close"]), "v": float(k["volume"]),
                    }
                    self.on_kline(symbol, candle, bool(k.get("confirm", False)))
                except Exception as e:
                    logger.error("on_kline handler error: %s", e)

    def _on_error(self, ws, error) -> None:  # pragma: no cover
        logger.warning("BybitWebSocket error: %s", error)

    def _on_close(self, ws, code, msg) -> None:
        logger.warning("BybitWebSocket tertutup (code=%s, msg=%s) — akan reconnect", code, msg)


class BinancePositionMarketWebSocket:
    """Public Binance USDⓈ-M market stream used ONLY for position display.

    This stream is intentionally isolated from trading logic:
    - no REST calls;
    - no order/fill decisions;
    - no candle/strategy processing;
    - only maintains the freshest Binance Futures mark price for symbols that
      have an active/pending local position.

    Binance's public market streams support live SUBSCRIBE/UNSUBSCRIBE without
    credentials. We use ``@markPrice@1s`` because the /trade PnL view is meant
    to track the Binance Futures mark price rather than a Bybit last price.
    """

    URL = "wss://fstream.binance.com/stream"

    def __init__(self, on_price):
        self.on_price = on_price
        self._lock = threading.RLock()
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._subscribed: set[str] = set()
        self._request_id = 0

    @property
    def connected(self) -> bool:
        with self._lock:
            return bool(self._ws and not self._stop.is_set())

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    @staticmethod
    def _stream(symbol: str) -> str:
        return f"{symbol.lower()}@markPrice@1s"

    def start(self) -> None:
        if websocket is None:
            logger.error("Package 'websocket-client' belum terinstal — Binance position market WS nonaktif")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="BinancePositionMarketWS", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            ws = self._ws
            self._ws = None
        if ws:
            try:
                ws.close()
            except Exception:
                pass

    def subscribe(self, symbol: str) -> None:
        symbol = str(symbol or "").upper()
        if not symbol:
            return
        with self._lock:
            if symbol in self._subscribed:
                return
            self._subscribed.add(symbol)
            ws = self._ws
        if ws:
            self._send_subscription("SUBSCRIBE", [self._stream(symbol)])
            logger.info("[BINANCE-MARKET-WS] SUBSCRIBE %s", symbol)

    def unsubscribe(self, symbol: str) -> None:
        symbol = str(symbol or "").upper()
        if not symbol:
            return
        with self._lock:
            self._subscribed.discard(symbol)
            ws = self._ws
        if ws:
            self._send_subscription("UNSUBSCRIBE", [self._stream(symbol)])
            logger.info("[BINANCE-MARKET-WS] UNSUBSCRIBE %s", symbol)

    def sync_subscriptions(self, symbols: Iterable[str]) -> None:
        desired = {str(s or "").upper() for s in symbols if str(s or "").strip()}
        with self._lock:
            current = set(self._subscribed)
        for symbol in sorted(current - desired):
            self.unsubscribe(symbol)
        for symbol in sorted(desired - current):
            self.subscribe(symbol)

    def _send_subscription(self, method: str, streams: Sequence[str]) -> None:
        if not streams:
            return
        with self._lock:
            ws = self._ws
        if ws is None:
            return
        payload = {"method": method, "params": list(streams), "id": self._next_id()}
        try:
            ws.send(json.dumps(payload))
        except Exception as exc:
            logger.warning("[BINANCE-MARKET-WS] %s gagal: %s", method, exc)

    def _on_open(self, ws) -> None:
        logger.info("[BINANCE-MARKET-WS] CONNECTED")
        with self._lock:
            symbols = sorted(self._subscribed)
        # Keep batches small and far below Binance's websocket message-rate
        # limit. One subscription request is enough for all active positions.
        streams = [self._stream(s) for s in symbols]
        if streams:
            for start in range(0, len(streams), 50):
                self._send_subscription("SUBSCRIBE", streams[start:start + 50])

    def _on_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return
        if "stream" not in payload:
            return
        data = payload.get("data") or {}
        if data.get("e") != "markPriceUpdate":
            return
        symbol = str(data.get("s") or "").upper()
        mark = data.get("p")
        if not symbol or mark is None:
            return
        try:
            price = float(mark)
            event_ts = float(data.get("E") or data.get("T") or time.time() * 1000)
            if not math.isfinite(price) or price <= 0:
                return
            self.on_price(symbol, price, event_ts)
        except (TypeError, ValueError, OverflowError) as exc:
            logger.debug("[BINANCE-MARKET-WS] price parse gagal %s: %s", symbol, exc)

    def _on_error(self, ws, error) -> None:
        logger.warning("[BINANCE-MARKET-WS] error: %s", error)

    def _on_close(self, ws, code, msg) -> None:
        logger.warning("[BINANCE-MARKET-WS] closed code=%s msg=%s — reconnect", code, msg)

    def _run_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                ws = websocket.WebSocketApp(
                    self.URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                with self._lock:
                    self._ws = ws
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:  # pragma: no cover
                logger.warning("[BINANCE-MARKET-WS] run error: %s", exc)
            finally:
                with self._lock:
                    if self._ws is ws:
                        self._ws = None
            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


class BinanceUserDataStream:
    """Binance USD-M Futures user-data stream.

    Uses a single long-lived listenKey WebSocket for ORDER_TRADE_UPDATE and
    ACCOUNT_UPDATE. The stream is deliberately independent from Binance REST
    rate-limit handling so fills/position changes can still arrive while REST
    is paused or HTTP 418 banned.
    """

    REST_URL = "https://fapi.binance.com/fapi/v1/listenKey"
    WS_BASE = "wss://fstream.binance.com/ws"

    def __init__(self, api_key: str, on_event, rest_request=None, rest_allowed=None, on_rate_limit=None):
        self.api_key = api_key
        self.on_event = on_event
        self._rest_request = rest_request
        self._rest_allowed = rest_allowed or (lambda: True)
        self._on_rate_limit = on_rate_limit or (lambda err: None)
        self._ws = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._listen_key: Optional[str] = None
        self._lock = threading.RLock()
        self._last_event_ts = 0.0
        self._last_keepalive = 0.0

    @property
    def connected(self) -> bool:
        return bool(self._ws and self._listen_key and not self._stop.is_set())

    def start(self) -> None:
        if websocket is None:
            logger.error("websocket-client tidak terpasang; Binance user-data stream nonaktif")
            return
        if not self.api_key:
            logger.warning("BINANCE_API_KEY kosong; Binance user-data stream tidak dapat dimulai")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="BinanceUserData", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def _create_listen_key(self) -> Optional[str]:
        try:
            if self._rest_request is not None:
                data = self._rest_request("POST", "/fapi/v1/listenKey", signed=False)
                key = data.get("listenKey") if isinstance(data, dict) else None
                if not key:
                    raise ExchangeError("Binance listenKey response tidak berisi listenKey")
                return str(key)
            resp = requests.post(
                self.REST_URL,
                headers={"X-MBX-APIKEY": self.api_key},
                timeout=15,
            )
            if resp.status_code in (429, 418):
                raise RateLimitError(
                    f"Binance listenKey rate-limit HTTP {resp.status_code}: {resp.text[:300]}",
                    status_code=resp.status_code,
                    retry_after=float(resp.headers.get("Retry-After")) if resp.headers.get("Retry-After") else None,
                )
            resp.raise_for_status()
            data = resp.json()
            key = data.get("listenKey")
            if not key:
                raise ExchangeError("Binance listenKey response tidak berisi listenKey")
            return str(key)
        except RateLimitError:
            raise
        except Exception as e:
            logger.warning("Gagal membuat Binance listenKey: %s", e)
            return None

    def _keepalive(self) -> None:
        if not self._listen_key or not self._rest_allowed():
            return
        try:
            if self._rest_request is not None:
                self._rest_request("PUT", "/fapi/v1/listenKey", signed=False)
            else:
                resp = requests.put(
                    self.REST_URL,
                    headers={"X-MBX-APIKEY": self.api_key},
                    timeout=15,
                )
                if resp.status_code in (429, 418):
                    raise RateLimitError(
                        f"Binance listenKey keepalive rate-limit HTTP {resp.status_code}",
                        status_code=resp.status_code,
                        retry_after=float(resp.headers.get("Retry-After")) if resp.headers.get("Retry-After") else None,
                    )
                if resp.status_code >= 400:
                    raise ExchangeError(f"listenKey keepalive HTTP {resp.status_code}: {resp.text[:300]}")
            self._last_keepalive = time.time()
        except RateLimitError as e:
            self._on_rate_limit(e)
            logger.warning("Binance user-data keepalive terkena rate-limit: %s", e)
        except Exception as e:
            logger.warning("Binance user-data keepalive gagal: %s", e)

    def _run_forever(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            if not self._listen_key and not self._rest_allowed():
                time.sleep(5)
                continue
            if not self._listen_key:
                try:
                    self._listen_key = self._create_listen_key()
                except RateLimitError as e:
                    self._on_rate_limit(e)
                    logger.warning("Binance user-data listenKey terkena rate-limit: %s", e)
                    time.sleep(min(30.0, backoff))
                    backoff = min(backoff * 2.0, 60.0)
                    continue
                if not self._listen_key:
                    time.sleep(min(30.0, backoff))
                    backoff = min(backoff * 2.0, 60.0)
                    continue

            url = f"{self.WS_BASE}/{self._listen_key}"
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=0, ping_timeout=None)
            except Exception as e:  # pragma: no cover
                logger.warning("Binance user-data websocket error: %s", e)
            finally:
                self._ws = None

            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)

    def _on_open(self, ws) -> None:
        backoff = 2.0
        self._last_event_ts = time.time()
        self._last_keepalive = time.time()
        logger.info("Binance User Data WS CONNECTED")
        # A listenKey expires after 60 minutes. Keepalive dilakukan sekitar 50 menit sekali untuk menghindari request REST yang tidak perlu.
        # Keepalive is performed by this same thread without touching BinanceClient REST gate.
        def keepalive_loop():
            while not self._stop.is_set() and ws is self._ws:
                now = time.time()
                if now - self._last_keepalive >= 50 * 60:
                    self._keepalive()
                time.sleep(15)
        threading.Thread(target=keepalive_loop, name="BinanceUserDataKeepalive", daemon=True).start()

    def _on_message(self, ws, message: str) -> None:
        self._last_event_ts = time.time()
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        try:
            self.on_event(data)
        except Exception:
            logger.exception("Binance user-data event handler error")

    def _on_error(self, ws, error) -> None:
        logger.warning("Binance User Data WS error: %s", error)

    def _on_close(self, ws, code, msg) -> None:
        logger.warning("Binance User Data WS closed code=%s msg=%s", code, msg)
        # Expired stream must get a new listenKey after reconnect.
        if code in (1000, 1001) and self._stop.is_set():
            return
        if code == 1000 and self._last_event_ts and (time.time() - self._last_event_ts) > 3600:
            self._listen_key = None


# =============================================================================
# 3. BINANCE FUTURES REST (eksekusi order nyata — §61)
#    NB: "Algo Order" pada spesifikasi (TP/SL/Trail) diimplementasikan
#    dengan conditional order Binance Futures asli (STOP_MARKET /
#    TAKE_PROFIT_MARKET / TRAILING_STOP_MARKET via endpoint /fapi/v1/order),
#    BUKAN endpoint /sapi/v1/algo/* (yang ditujukan untuk TWAP/eksekusi
#    besar) — endpoint tersebut tidak relevan untuk TP/SL/Trail per posisi.
# =============================================================================

class BinanceClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode() if api_secret else b""
        self.base_url = "https://fapi.binance.com"  # selalu MAINNET — pakai API key/secret asli
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})
        self._exchange_info_cache: Optional[Dict[str, Any]] = None
        self._exchange_info_ts: float = 0.0
        self._time_offset_ms: int = 0
        self._time_offset_synced_at: float = 0.0
        self._hedge_mode: Optional[bool] = None
        # Semua Binance REST melewati SATU serial choke-point.
        # Network call juga diserialkan, bukan hanya jadwalnya, sehingga jika
        # satu worker menerima 429/418 worker lain tidak bisa menyelipkan request
        # sebelum governor sempat memasang hard-stop.
        self._request_gate_lock = threading.Lock()
        self._network_lock = threading.RLock()
        self._next_request_mono = 0.0
        self._min_request_interval = BINANCE_REQUEST_INTERVAL
        self._order_request_interval = BINANCE_ORDER_INTERVAL
        self._next_order_request_mono = 0.0
        self._used_weight_1m = 0
        self._used_weight_1m_ts = 0.0
        self._last_response_ts = 0.0
        self._last_request_path = ""
        self._consecutive_429 = 0
        self._last_rate_limit_ts = 0.0
        self._last_rate_limit_status = 0
        self._used_order_10s = 0
        self._used_order_1m = 0
        self._order_limit_10s = 300
        self._order_limit_1m = 1200
        self._blocked_until_mono = 0.0
        self._blocked_error: Optional[RateLimitError] = None
        self._leverage_cache: Dict[str, int] = {}

    def health_check(self) -> Dict[str, Any]:
        """Explicit operator health probe; never called automatically at startup."""
        return self._request("GET", "/fapi/v1/time", signed=False)

    # -- revisi: sinkronisasi waktu server (§4) -----------------------------
    # Penyebab klasik "API key sama, IP sama, tiba-tiba semua order error"
    # adalah timestamp lokal drift terhadap server Binance (-1021 Timestamp
    # for this request is outside of the recvWindow). Ini sinkronkan offset
    # sekali di awal & auto-refresh tiap 30 menit / saat error -1021 muncul.
    def sync_server_time(self) -> None:
        """Sync server time through the SAME REST governor.

        Versi lama memakai session.get() langsung di sini sehingga endpoint
        /time dapat melewati gate 429/418. Pada IP yang sedang ditekan Binance,
        itu justru dapat memperburuk keadaan.
        """
        data = self._request("GET", "/fapi/v1/time", signed=False, _skip_time_sync=True)
        server_ms = int(data["serverTime"])
        local_ms = int(time.time() * 1000)
        self._time_offset_ms = server_ms - local_ms
        self._time_offset_synced_at = time.time()
        logger.info(
            "[BINANCE] TIME SYNC | offset=%dms | weight_1m=%s",
            self._time_offset_ms, self._used_weight_1m,
        )

    def _timestamp(self) -> int:
        # Timestamp is intentionally local-first. Binance time is synchronized
        # only on explicit startup/operator demand or after HTTP -1021.
        return int(time.time() * 1000) + self._time_offset_ms

    def _sign(self, params: Dict[str, Any]) -> str:
        query = urllib.parse.urlencode(params, doseq=True)
        sig = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    def _local_rate_limit_error(self) -> RateLimitError:
        err = self._blocked_error or RateLimitError(
            "Binance REST locally blocked by governor", status_code=429, code=-1003
        )
        remaining = max(0.0, self._blocked_until_mono - time.monotonic())
        return RateLimitError(
            str(err),
            status_code=err.status_code or 429,
            retry_after=remaining,
            banned_until_ms=err.banned_until_ms,
            code=err.code,
        )

    @staticmethod
    def _header_float(headers: Any, name: str) -> Optional[float]:
        raw = headers.get(name)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _response_rate_limit_block(self, resp: Any, code: Optional[int], msg: str) -> RateLimitError:
        retry_after = self._header_float(resp.headers, "Retry-After")
        banned_until_ms = None
        match = re.search(r"banned until (\d+)", msg or "", re.I)
        if match:
            banned_until_ms = int(match.group(1))

        status = int(resp.status_code)
        if banned_until_ms:
            block_seconds = max(1.0, banned_until_ms / 1000.0 - time.time())
        elif retry_after is not None:
            block_seconds = max(1.0, retry_after)
        elif code == -1015:
            block_seconds = BINANCE_DEFAULT_429_COOLDOWN
        elif status == 429:
            block_seconds = BINANCE_DEFAULT_429_COOLDOWN
        else:
            block_seconds = 300.0

        err = RateLimitError(
            f"Binance rate limit (HTTP {status}): {msg}",
            status_code=status, retry_after=retry_after,
            banned_until_ms=banned_until_ms, code=code,
        )
        self._blocked_until_mono = max(
            self._blocked_until_mono, time.monotonic() + block_seconds
        )
        self._blocked_error = err
        self._last_rate_limit_ts = time.time()
        self._last_rate_limit_status = status
        self._consecutive_429 = self._consecutive_429 + 1 if status == 429 else self._consecutive_429
        logger.warning(
            "[BINANCE] REST RATE LIMIT | path=%s HTTP=%s code=%s block=%.1fs "
            "retry_after=%s weight_1m=%s order10s=%s order1m=%s msg=%s",
            self._last_request_path, status, code, block_seconds, retry_after,
            self._used_weight_1m, self._used_order_10s, self._used_order_1m,
            msg[:220],
        )
        return err

    def _refresh_local_usage_windows(self) -> None:
        """Expire locally remembered Binance counters when their windows elapsed.

        Response headers are authoritative when a request is made. These local
        resets only prevent stale counters from causing a permanent self-block
        after an idle period.
        """
        now = time.time()
        if self._used_weight_1m_ts and now - self._used_weight_1m_ts >= 60.0:
            self._used_weight_1m = 0
            self._used_weight_1m_ts = 0.0
        # ORDER-COUNT is maintained by Binance in rolling 10s/1m windows.
        if self._last_response_ts and now - self._last_response_ts >= 10.0:
            self._used_order_10s = 0
        if self._last_response_ts and now - self._last_response_ts >= 60.0:
            self._used_order_1m = 0

    def _request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None,
        signed: bool = False, _retry_on_time_drift: bool = True,
        _skip_time_sync: bool = False,
    ) -> Any:
        """Single choke-point for ALL Binance REST requests.

        Safety properties:
        - one network request at a time;
        - no retry after 429/418;
        - local hard-stop is checked again immediately before sending;
        - response usage headers update the governor;
        - time-sync uses this same choke-point;
        - socket timeouts are NEVER treated as rate-limit events.
        """
        params = dict(params or {})
        is_order_request = path in (
            "/fapi/v1/order", "/fapi/v1/allOpenOrders", "/fapi/v1/batchOrders",
            "/fapi/v1/leverage", "/fapi/v1/positionSide/dual"
        )

        with self._network_lock:
            self._refresh_local_usage_windows()
            now_mono = time.monotonic()
            if now_mono < self._blocked_until_mono:
                raise self._local_rate_limit_error()

            # Trade requests get an earlier local stop. REQUEST_WEIGHT is
            # shared at the source-IP level, so "first order of this process"
            # does not mean "first request of the IP in this minute".
            # Never spend the last part of the budget on an entry.
            if is_order_request and self._used_weight_1m >= BINANCE_WEIGHT_ORDER_SOFT_STOP:
                block_seconds = 30.0
                self._blocked_until_mono = max(
                    self._blocked_until_mono, time.monotonic() + block_seconds
                )
                err = RateLimitError(
                    f"Binance trade REST deferred by local governor: "
                    f"used_weight_1m={self._used_weight_1m} >= {BINANCE_WEIGHT_ORDER_SOFT_STOP}",
                    status_code=429, code=-1003, retry_after=block_seconds,
                )
                self._blocked_error = err
                logger.warning(
                    "[BINANCE] TRADE PREBLOCK | weight_1m=%s >= %s | path=%s | "
                    "order10s=%s order1m=%s | request not sent",
                    self._used_weight_1m, BINANCE_WEIGHT_ORDER_SOFT_STOP, path,
                    self._used_order_10s, self._used_order_1m,
                )
                raise err

            # Conservative pre-flight headroom. A Binance 429 must be treated
            # as a stop signal, not something we retry into a 418.
            if self._used_weight_1m >= BINANCE_WEIGHT_HARD_STOP:
                block_seconds = 60.0
                self._blocked_until_mono = max(
                    self._blocked_until_mono, time.monotonic() + block_seconds
                )
                err = RateLimitError(
                    f"Binance REST governor pre-block: used_weight_1m={self._used_weight_1m}",
                    status_code=429, code=-1003, retry_after=block_seconds,
                )
                self._blocked_error = err
                logger.warning(
                    "[BINANCE] REST PREBLOCK | weight_1m=%s >= %s",
                    self._used_weight_1m, BINANCE_WEIGHT_HARD_STOP,
                )
                raise err

            if is_order_request and (
                self._used_order_10s >= int(self._order_limit_10s * 0.70)
                or self._used_order_1m >= int(self._order_limit_1m * 0.85)
            ):
                block_seconds = 15.0
                self._blocked_until_mono = max(
                    self._blocked_until_mono, time.monotonic() + block_seconds
                )
                err = RateLimitError(
                    "Binance order governor pre-block: "
                    f"10s={self._used_order_10s}/{self._order_limit_10s}, "
                    f"1m={self._used_order_1m}/{self._order_limit_1m}",
                    status_code=429, code=-1015, retry_after=block_seconds,
                )
                self._blocked_error = err
                raise err

            with self._request_gate_lock:
                now_mono = time.monotonic()
                wait = max(
                    self._next_request_mono - now_mono,
                    (self._next_order_request_mono - now_mono) if is_order_request else 0.0,
                )
                if wait > 0:
                    time.sleep(wait)
                # Re-check after sleeping: another thread cannot be sending
                # concurrently because _network_lock is held, but this guard
                # keeps the invariant explicit.
                if time.monotonic() < self._blocked_until_mono:
                    raise self._local_rate_limit_error()
                send_mono = time.monotonic()
                self._next_request_mono = send_mono + self._min_request_interval
                if is_order_request:
                    self._next_order_request_mono = send_mono + self._order_request_interval

            self._last_request_path = path
            logger.debug(
                "[BINANCE] REST SEND | %s %s | signed=%s | weight1m=%s order10s=%s order1m=%s",
                method, path, signed, self._used_weight_1m, self._used_order_10s, self._used_order_1m,
            )

            try:
                if signed:
                    # Do not poll /fapi/v1/time periodically. Signed requests use
                    # the locally maintained offset; an actual -1021 is the only
                    # event that triggers one explicit resynchronization.
                    params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
                    params["recvWindow"] = 10000
                    query = self._sign(params)
                    resp = self.session.request(
                        method, f"{self.base_url}{path}?{query}", timeout=15
                    )
                else:
                    resp = self.session.request(
                        method, self.base_url + path, params=params, timeout=15
                    )
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning("[BINANCE] REST NETWORK TIMEOUT/CONNECTION | path=%s | %s", path, e)
                raise ExchangeError(f"Binance connection error: {e}")

            self._last_response_ts = time.time()
            raw_weight = resp.headers.get("X-MBX-USED-WEIGHT-1M") or resp.headers.get("X-MBX-USED-WEIGHT-1m")
            if raw_weight:
                try:
                    self._used_weight_1m = int(float(raw_weight))
                    self._used_weight_1m_ts = time.time()
                except (TypeError, ValueError):
                    pass
            for header_name, attr in (
                ("X-MBX-ORDER-COUNT-10S", "_used_order_10s"),
                ("X-MBX-ORDER-COUNT-1M", "_used_order_1m"),
            ):
                raw = resp.headers.get(header_name)
                if raw is not None:
                    try:
                        setattr(self, attr, int(float(raw)))
                    except (TypeError, ValueError):
                        pass

            if resp.status_code in (429, 418):
                msg = resp.text[:1000]
                code = None
                try:
                    body = resp.json()
                    if isinstance(body, dict):
                        code = body.get("code")
                        msg = body.get("msg") or msg
                except ValueError:
                    pass
                raise self._response_rate_limit_block(resp, code, msg)

            try:
                data = resp.json()
            except ValueError:
                raise ExchangeError(
                    f"Binance HTTP {resp.status_code} malformed response: {resp.text[:200]}"
                )

            if isinstance(data, dict) and "code" in data and data.get("code", 0) < 0:
                code = data["code"]
                msg = data.get("msg")
                if code in (-1003, -1015):
                    class _SyntheticResponse:
                        status_code = resp.status_code
                        headers = resp.headers
                        text = resp.text
                        def json(self_inner):
                            return data
                    raise self._response_rate_limit_block(_SyntheticResponse(), code, msg or "")
                if code == -1021 and signed and _retry_on_time_drift:
                    logger.warning("[BINANCE] -1021 timestamp drift — sync + retry once")
                    # Clear stale offset before the one allowed retry. The retry
                    # remains behind the same serial network gate.
                    self._time_offset_synced_at = 0.0
                    self.sync_server_time()
                    return self._request(
                        method, path, params, signed=signed,
                        _retry_on_time_drift=False, _skip_time_sync=True,
                    )
                if code == -2015:
                    raise ExchangeError(
                        f"Binance error -2015 (Invalid API-key/IP/permissions): {msg}. "
                        f"Endpoint: {self.base_url} (MAINNET). "
                        "Penyebab tersering: IP restriction key tidak mencakup IP server ini, "
                        "atau permission Futures belum diaktifkan di API key ini."
                    )
                raise ExchangeError(f"Binance error {code}: {msg} (HTTP {resp.status_code})")

            # A successful response resets the consecutive-429 counter; usage
            # itself remains visible so the next request can still be preblocked.
            self._consecutive_429 = 0
            logger.debug(
                "[BINANCE] REST OK | %s %s | HTTP=%s | weight1m=%s order10s=%s order1m=%s",
                method, path, resp.status_code, self._used_weight_1m, self._used_order_10s, self._used_order_1m,
            )
            return data

    # -- account -----------------------------------------------------------
    def get_balance_usdt(self) -> float:
        data = self._request("GET", "/fapi/v2/balance", signed=True)
        for row in data:
            if row.get("asset") == "USDT":
                return float(row.get("balance", 0.0))
        raise ExchangeError("Asset USDT tidak ditemukan di response /fapi/v2/balance — cek permission API key")

    def get_position_risk(self, symbol: str) -> Optional[Dict[str, Any]]:
        data = self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
        for row in data:
            if row.get("symbol") == symbol and abs(float(row.get("positionAmt", 0))) > 0:
                return row
        return None

    def get_all_position_risk(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        return [r for r in data if abs(float(r.get("positionAmt", 0))) > 0]

    def is_hedge_mode(self, force: bool = False) -> bool:
        """§4 revisi — deteksi Hedge Mode (dual position side). Kalau akun
        dalam Hedge Mode tapi order dikirim tanpa positionSide yang sesuai,
        Binance menolak dengan -4061 (position side does not match) — ini
        salah satu penyebab umum 'order tiba-tiba error' yang TIDAK ada
        hubungannya dengan API key atau IP."""
        if self._hedge_mode is not None and not force:
            return self._hedge_mode
        try:
            data = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
            self._hedge_mode = bool(data.get("dualSidePosition", False))
        except ExchangeError as e:
            logger.error("Gagal deteksi position mode Binance: %s", e)
            raise
        return self._hedge_mode

    def _position_side_param(self, direction: str, closing: bool = False) -> Dict[str, str]:
        """Kembalikan {'positionSide': ...} kalau Hedge Mode aktif, kosong
        kalau One-way (One-way TIDAK boleh dikirimi positionSide selain
        default). `closing` diabaikan sengaja: di Hedge Mode, order penutup
        posisi BUY tetap pakai positionSide=LONG (bukan SHORT!) — kesalahan
        umum yang bikin order TP/SL gagal terpasang."""
        if not self.is_hedge_mode():
            return {}
        return {"positionSide": "LONG" if direction == "BUY" else "SHORT"}

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        if isinstance(leverage, bool) or int(leverage) != leverage:
            raise ExchangeError(f"Leverage {leverage} harus bilangan bulat")
        leverage = int(leverage)
        if leverage < 1 or leverage > 125:
            raise ExchangeError(f"Leverage {leverage} di luar rentang 1-125x")
        if self._leverage_cache.get(symbol) == leverage:
            return {"symbol": symbol, "leverage": leverage, "_cached": True}
        data = self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)
        self._leverage_cache[symbol] = leverage
        return data

    def exchange_info(self, force: bool = False) -> Dict[str, Any]:
        if self._exchange_info_cache and not force and time.time() - self._exchange_info_ts < 3600:
            return self._exchange_info_cache
        data = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
        symbols = {s["symbol"]: s for s in data.get("symbols", [])}
        self._exchange_info_cache = symbols
        self._exchange_info_ts = time.time()
        return symbols

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        # Binance exchangeInfo is intentionally lazy: scanning does not call it.
        # It is fetched only when an actual execution candidate needs exchange filters.
        info = self.exchange_info().get(symbol)
        if not info:
            raise ExchangeError(f"Symbol {symbol} tidak ditemukan di Binance exchangeInfo")
        filters = {f["filterType"]: f for f in info["filters"]}
        return {
            "step_size": Decimal(filters["LOT_SIZE"]["stepSize"]),
            "min_qty": Decimal(filters["LOT_SIZE"]["minQty"]),
            "tick_size": Decimal(filters["PRICE_FILTER"]["tickSize"]),
            "min_notional": Decimal((filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}).get("minNotional", (filters.get("MIN_NOTIONAL") or {}).get("notional", "5"))),
            "quantity_precision": info.get("quantityPrecision", 3),
            "price_precision": info.get("pricePrecision", 2),
        }

    # -- orders ----------------------------------------------------------------
    # NB: `direction` di sini = arah POSISI yang sedang dibuka/ditutup ("BUY"
    # untuk posisi LONG, "SELL" untuk posisi SHORT) — dipakai HANYA untuk
    # menentukan positionSide di Hedge Mode, terlepas dari `side` order itu
    # sendiri (order penutup LONG punya side=SELL tapi positionSide=LONG).
    def place_market_order(self, symbol: str, side: str, quantity: Decimal, direction: str) -> Any:
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": str(quantity)}
        params.update(self._position_side_param(direction))
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_limit_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal, direction: str,
        client_order_id: Optional[str] = None,
    ) -> Any:
        params = {
            "symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTC",
            "quantity": str(quantity), "price": str(price),
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        params.update(self._position_side_param(direction))
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_stop_market(self, symbol: str, side: str, stop_price: Decimal, direction: str, close_position: bool = True, quantity: Optional[Decimal] = None) -> Any:
        hedge = self.is_hedge_mode()
        params = {"symbol": symbol, "side": side, "type": "STOP_MARKET", "stopPrice": str(stop_price)}
        if hedge:
            if quantity is None or quantity <= 0:
                raise ExchangeError(f"Hedge Mode memerlukan quantity untuk STOP_MARKET {symbol}")
            params["quantity"] = str(quantity)
        elif close_position:
            params["closePosition"] = "true"
        params.update(self._position_side_param(direction))
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_take_profit_market(self, symbol: str, side: str, stop_price: Decimal, direction: str, close_position: bool = True, quantity: Optional[Decimal] = None) -> Any:
        hedge = self.is_hedge_mode()
        params = {"symbol": symbol, "side": side, "type": "TAKE_PROFIT_MARKET", "stopPrice": str(stop_price)}
        if hedge:
            if quantity is None or quantity <= 0:
                raise ExchangeError(f"Hedge Mode memerlukan quantity untuk TAKE_PROFIT_MARKET {symbol}")
            params["quantity"] = str(quantity)
        elif close_position:
            params["closePosition"] = "true"
        params.update(self._position_side_param(direction))
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_trailing_stop(self, symbol: str, side: str, callback_rate: float, activation_price: Decimal, direction: str) -> Any:
        params = {
            "symbol": symbol, "side": side, "type": "TRAILING_STOP_MARKET",
            "callbackRate": str(callback_rate), "activationPrice": str(activation_price),
        }
        if not self.is_hedge_mode():
            params["closePosition"] = "true"
        params.update(self._position_side_param(direction))
        return self._request(
            "POST", "/fapi/v1/order", params,
            signed=True,
        )

    def cancel_order(self, symbol: str, order_id: Any) -> Any:
        return self._request("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def cancel_all_open_orders(self, symbol: str) -> Any:
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)

    def get_open_orders(self, symbol: str) -> Any:
        return self._request("GET", "/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def get_all_open_orders(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/fapi/v1/openOrders", signed=True)


# =============================================================================
# 4. PRECISION / QUANTITY / GEOMETRY (§12, §58, §59, §60)
# =============================================================================

def round_step(value: Decimal, step: Decimal, rounding=ROUND_DOWN) -> Decimal:
    if step == 0:
        return value
    return (value / step).quantize(Decimal("1"), rounding=rounding) * step


def compute_quantity(
    entry: float, margin: float, leverage: float, filters: Dict[str, Any]
) -> Tuple[Optional[Decimal], str]:
    """Hitung quantity dari margin & leverage, lalu normalisasi terhadap
    step size / min qty / min notional, dan validasi margin tidak boleh
    menyimpang > ±50% dari target (§12)."""
    try:
        entry_d = Decimal(str(entry))
        margin_d = Decimal(str(margin))
        leverage_d = Decimal(str(leverage))
    except InvalidOperation:
        return None, "INVALID_NUMERIC_INPUT"

    if entry_d <= 0 or margin_d <= 0 or leverage_d <= 0:
        return None, "INVALID_INPUT_RANGE"

    notional_target = margin_d * leverage_d
    raw_qty = notional_target / entry_d

    step = filters["step_size"]
    qty = round_step(raw_qty, step)
    if qty < filters["min_qty"]:
        qty = filters["min_qty"]

    notional = qty * entry_d
    if notional < filters["min_notional"]:
        needed_qty = filters["min_notional"] / entry_d
        qty = round_step(needed_qty, step, rounding=ROUND_DOWN)
        # pastikan tetap >= min_notional setelah rounding-down step
        while qty * entry_d < filters["min_notional"]:
            qty += step
        notional = qty * entry_d

    if qty <= 0:
        return None, "QUANTITY_ZERO_AFTER_NORMALIZATION"

    resulting_margin = (qty * entry_d) / leverage_d
    lower = margin_d * Decimal("0.5")
    upper = margin_d * Decimal("1.5")
    if not (lower <= resulting_margin <= upper):
        return None, f"MARGIN_DEVIATION_OUT_OF_BOUND (target={margin_d}, actual={resulting_margin})"

    return qty, "OK"


def round_price(price: float, tick_size: Decimal, rounding=ROUND_DOWN) -> Decimal:
    return round_step(Decimal(str(price)), tick_size, rounding=rounding)

def normalize_order_prices(setup: Dict[str, Any], tick_size: Decimal) -> Dict[str, float]:
    """Round prices directionally so Binance tick constraints and geometry stay intact."""
    direction = setup["direction"]
    if direction == "BUY":
        entry_r = round_price(float(setup["entry"]), tick_size, ROUND_DOWN)
        sl_r = round_price(float(setup["sl"]), tick_size, ROUND_DOWN)
        tp_r = round_price(float(setup["tp"]), tick_size, rounding=ROUND_UP)
    else:
        entry_r = round_price(float(setup["entry"]), tick_size, rounding=ROUND_UP)
        sl_r = round_price(float(setup["sl"]), tick_size, rounding=ROUND_UP)
        tp_r = round_price(float(setup["tp"]), tick_size, ROUND_DOWN)
    return {"entry": float(entry_r), "sl": float(sl_r), "tp": float(tp_r)}


# =============================================================================
# 5. STATE MANAGEMENT — thread-safe, state machine, idempotency (§55,56,57)
# =============================================================================

ALLOWED_TRANSITIONS = {
    "BINANCE_WAITING": {"PENDING", "CANCELLED"},
    "PENDING": {"FILLED", "TIMEOUT", "CANCELLED"},
    "FILLED": {"PROTECTED", "CLOSED"},
    "PROTECTED": {"TRAILING", "CLOSED"},
    "TRAILING": {"CLOSED"},
}
TERMINAL_STATES = {"CLOSED", "TIMEOUT", "CANCELLED"}


class StateStore:
    """Menyimpan seluruh posisi/pending + state global bot. Semua mutasi
    dilindungi lock. Transisi status divalidasi terhadap ALLOWED_TRANSITIONS
    dan bersifat idempotent (event duplikat diabaikan dengan aman)."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.backup_path = checkpoint_path + ".backup"
        self._lock = threading.RLock()
        self._symbol_locks: Dict[str, threading.Lock] = {}

        self.mode = "SIMULASI"  # atau REAL
        self.auto = False
        self.margin = 1.0
        self.leverage = 5.0
        self.max_positions = 5
        self.autostop_pct: Optional[float] = None
        self.highest_balance: Optional[float] = None
        self.current_balance: Optional[float] = None
        # REAL balance model: one Binance snapshot is the anchor; runtime PnL
        # is accumulated locally from verified fills/closes. ACCOUNT_UPDATE WS
        # may add a reconciliation adjustment without polling REST balance.
        self.real_balance_snapshot: Optional[float] = None
        self.real_localized_pnl: float = 0.0
        self.real_balance_adjustment: float = 0.0
        self.real_balance_snapshot_ts: float = 0.0
        self.real_last_balance_source: str = "NONE"
        self.sim_balance = 10.0
        self.sim_balance_anchor = 10.0

        self.binance_paused = False
        self.binance_pause_ts: Optional[float] = None
        self.binance_pause_until: Optional[float] = None
        self.binance_pause_reason: str = ""

        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position dict
        self.scanned_coins: List[str] = []
        self.scan_history: List[Dict[str, Any]] = []
        self.bans: Dict[str, Dict[str, Any]] = {}  # symbol -> {reason, expiry(None=permanent), permanent}
        self.strategy_state: Dict[str, Any] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.processed_events: set = set()  # idempotency guard: f"{symbol}:{event}:{ts}"

    def symbol_lock(self, symbol: str) -> threading.Lock:
        with self._lock:
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = threading.Lock()
            return self._symbol_locks[symbol]

    # -- position lifecycle ---------------------------------------------------
    def add_pending(
        self, setup: Dict[str, Any], qty: Decimal, margin_used: float,
        status: str = "PENDING",
    ) -> None:
        with self._lock:
            entry = float(setup["entry"])
            sl = float(setup["sl"])
            now = time.time()
            self.positions[setup["pair"]] = {
                **setup,
                "status": status,
                "qty": str(qty),
                "margin_used": margin_used,
                "leverage": self.leverage,
                "created_at": now,
                "trail_count": 0,
                "binance_order_ids": {},
                "peak_price": entry,
                "initial_sl": sl,
                "initial_risk": abs(entry - sl),
                "real_fill_confirmed": False,
                "binance_leverage_confirmed": False,
                "binance_entry_client_order_id": "",
                "binance_entry_confirmed_at": None,
            }

    def transition(self, symbol: str, new_status: str, event_id: str, **updates) -> bool:
        """Return True jika transisi benar-benar diterapkan (bukan duplikat)."""
        with self.symbol_lock(symbol):
            if event_id in self.processed_events:
                return False  # §57 idempotency — event sudah pernah diproses
            pos = self.positions.get(symbol)
            if not pos:
                return False
            current = pos["status"]
            if current in TERMINAL_STATES:
                return False
            if new_status not in ALLOWED_TRANSITIONS.get(current, set()):
                logger.warning("Transisi ilegal ditolak: %s %s -> %s", symbol, current, new_status)
                return False
            pos["status"] = new_status
            pos.update(updates)
            self.processed_events.add(event_id)
            if len(self.processed_events) > 20000:
                # jaga agar set tidak tumbuh tanpa batas
                self.processed_events = set(list(self.processed_events)[-10000:])
            if new_status in TERMINAL_STATES:
                self.closed_trades.append(dict(pos))
            return True

    def get_active_count(self) -> int:
        with self._lock:
            return sum(1 for p in self.positions.values() if p["status"] not in TERMINAL_STATES)

    def remove_terminal(self, symbol: str) -> None:
        with self._lock:
            pos = self.positions.get(symbol)
            if pos and pos["status"] in TERMINAL_STATES:
                del self.positions[symbol]

    def discard_pending(self, symbol: str) -> None:
        """Batalkan pending yang gagal dipasang di exchange (order Binance
        error) — TIDAK dihitung sbg trade/timeout, murni rollback state."""
        with self._lock:
            pos = self.positions.get(symbol)
            if pos and pos["status"] == "PENDING":
                del self.positions[symbol]

    def snapshot_positions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(p) for p in self.positions.values()]

    # -- ban management (§26,27,28) -------------------------------------------
    def ban(self, symbol: str, reason: str, duration_sec: Optional[float] = None) -> None:
        symbol = symbol.upper()
        with self._lock:
            self.bans[symbol] = {
                "reason": reason,
                "expiry": None if duration_sec is None else time.time() + float(duration_sec),
                "permanent": duration_sec is None,
            }

    def unban(self, symbol: Optional[str]) -> None:
        with self._lock:
            if symbol is None or symbol.upper() == "ALL":
                self.bans.clear()
            else:
                self.bans.pop(symbol, None)

    def is_banned(self, symbol: str) -> bool:
        with self._lock:
            b = self.bans.get(symbol)
            if not b:
                return False
            if b.get("expiry") is None or b.get("permanent"):
                return True
            if b["expiry"] <= time.time():
                del self.bans[symbol]
                return False
            return True

    def cleanup_expired_bans(self) -> List[str]:
        expired = []
        with self._lock:
            for symbol in list(self.bans.keys()):
                expiry = self.bans[symbol].get("expiry")
                if expiry is not None and expiry <= time.time():
                    del self.bans[symbol]
                    expired.append(symbol)
        return expired

    def active_bans(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self.bans.items()}

    # -- checkpoint (state utama, terpisah dari learn checkpoint) -------------
    def export_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode, "auto": self.auto, "margin": self.margin, "leverage": self.leverage, "max_positions": self.max_positions,
                "autostop_pct": self.autostop_pct, "highest_balance": self.highest_balance, "current_balance": self.current_balance,
                "real_balance_snapshot": self.real_balance_snapshot,
                "real_localized_pnl": self.real_localized_pnl,
                "real_balance_adjustment": self.real_balance_adjustment,
                "real_balance_snapshot_ts": self.real_balance_snapshot_ts,
                "real_last_balance_source": self.real_last_balance_source,
                "sim_balance": self.sim_balance, "sim_balance_anchor": self.sim_balance_anchor,
                "positions": self.positions, "scanned_coins": self.scanned_coins,
                "scan_history": self.scan_history[-20:],
                "bans": self.bans, "closed_trades": self.closed_trades[-2000:],
                "strategy_state": self.strategy_state,
                "binance_paused": self.binance_paused,
                "binance_pause_ts": self.binance_pause_ts,
                "binance_pause_until": self.binance_pause_until,
                "binance_pause_reason": self.binance_pause_reason,
                "saved_at": time.time(),
            }

    def load_state(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self.mode = data.get("mode", self.mode)
            self.auto = data.get("auto", False)
            self.margin = data.get("margin", self.margin)
            self.leverage = data.get("leverage", self.leverage)
            try:
                self.max_positions = max(1, min(20, int(data.get("max_positions", self.max_positions))))
            except (TypeError, ValueError):
                self.max_positions = 5
            self.autostop_pct = data.get("autostop_pct")
            self.highest_balance = data.get("highest_balance")
            self.current_balance = data.get("current_balance")
            self.real_balance_snapshot = data.get("real_balance_snapshot")
            self.real_localized_pnl = float(data.get("real_localized_pnl", 0.0) or 0.0)
            self.real_balance_adjustment = float(data.get("real_balance_adjustment", 0.0) or 0.0)
            self.real_balance_snapshot_ts = float(data.get("real_balance_snapshot_ts", 0.0) or 0.0)
            self.real_last_balance_source = str(data.get("real_last_balance_source", "NONE") or "NONE")
            self.sim_balance = data.get("sim_balance", 10.0)
            self.sim_balance_anchor = data.get("sim_balance_anchor", 10.0)
            self.positions = data.get("positions", {})
            self.scanned_coins = data.get("scanned_coins", [])
            self.scan_history = data.get("scan_history", [])
            self.bans = data.get("bans", {})
            self.closed_trades = data.get("closed_trades", [])
            self.strategy_state = data.get("strategy_state", {}) if isinstance(data.get("strategy_state", {}), dict) else {}
            self.binance_paused = bool(data.get("binance_paused", False))
            self.binance_pause_ts = data.get("binance_pause_ts")
            self.binance_pause_until = data.get("binance_pause_until")
            self.binance_pause_reason = data.get("binance_pause_reason", "")
            if self.binance_paused and self.binance_pause_until is not None and self.binance_pause_until <= time.time():
                self.binance_paused = False

    def save_checkpoint(self) -> None:
        data = self.export_state()
        tmp = self.checkpoint_path + ".tmp"
        try:
            if os.path.exists(self.checkpoint_path):
                import shutil
                shutil.copyfile(self.checkpoint_path, self.backup_path)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, self.checkpoint_path)
        except OSError as e:
            logger.error("Gagal simpan state checkpoint: %s", e)

    def clear_runtime_checkpoint(self) -> None:
        """Delete main runtime checkpoint for a genuinely fresh /try session.

        Learning memory is intentionally NOT deleted. This only resets trading
        runtime state so a new /try starts with a fresh main session.
        """
        with self._lock:
            for path in (self.checkpoint_path, self.backup_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError as e:
                    logger.warning("Gagal menghapus runtime checkpoint %s: %s", path, e)

    def load_checkpoint(self) -> str:
        for path, label in ((self.checkpoint_path, "primary"), (self.backup_path, "backup")):
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.load_state(data)
                return label
            except (json.JSONDecodeError, OSError):
                continue
        return "empty"


# =============================================================================
# 6. TELEGRAM
# =============================================================================

IMPORTANT_EVENTS = {
    "BOT_START", "BOT_STOP", "BINANCE_PAUSE", "BINANCE_READY", "SIGNAL_PASSED",
    "PENDING", "FILLED", "TRAIL", "TP", "SL", "TIMEOUT", "BANNED", "UNBANNED",
    "MARGIN_SUCCESS", "LEVERAGE_SUCCESS", "AUTOSTOP", "ERROR", "WARNING",
}


class TelegramErrorLogHandler(logging.Handler):
    """Bridge ERROR log records to the bot Telegram queue.

    Non-recursive: records emitted by the Telegram subsystem itself are ignored,
    and delivery is queued rather than doing network I/O from the logging call.
    """
    def __init__(self, notifier_getter):
        super().__init__(level=logging.ERROR)
        self._notifier_getter = notifier_getter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.name.startswith("telegram") or record.name.startswith("urllib3"):
                return
            notifier = self._notifier_getter()
            if notifier is None:
                return
            symbol = getattr(record, "symbol", None) or getattr(record, "coin", None)
            scope = f"[{str(symbol).upper()}] " if symbol else ""
            text = self.format(record).replace("\n", " | ")
            notifier.send(f"⚠️ ERROR LOG\n{scope}{text}", "ERROR")
        except Exception:
            # Logging must never be allowed to crash the trading worker.
            pass


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        self._queue: "queue.Queue[str]" = queue.Queue()

    def start_sender(self) -> None:
        """Compatibility no-op: queue Telegram sekarang dipompa Worker 3/4 agar total worker <= 5."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram belum dikonfigurasi (.env) — notifikasi nonaktif")

    def start(self) -> None:
        self.start_sender()

    def stop(self) -> None:
        self.flush(max_messages=100)

    def send(self, text: str, event_type: str = "INFO") -> None:
        """Kirim hanya event penting ke Telegram (§53); pesan dipecah agar <=4096 karakter."""
        if event_type not in IMPORTANT_EVENTS and event_type != "INFO":
            return
        if not self.token or not self.chat_id:
            return
        text = str(text)
        max_len = 3900
        if len(text) <= max_len:
            self._queue.put(text)
            return
        chunk = ""
        for line in text.splitlines(True):
            if len(chunk) + len(line) > max_len and chunk:
                self._queue.put(chunk.rstrip())
                chunk = ""
            chunk += line
        if chunk:
            self._queue.put(chunk.rstrip())

    def flush(self, max_messages: int = 5) -> int:
        """Kirim queue secara bounded; dipanggil oleh worker yang sudah ada."""
        if not self.token or not self.chat_id:
            return 0
        sent = 0
        while sent < max_messages:
            try:
                text = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                resp = requests.post(
                    f"{self.base}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text},
                    timeout=5,
                )
                if not resp.ok:
                    logger.warning("Telegram sendMessage HTTP %s: %s", resp.status_code, resp.text[:200])
            except Exception as e:
                logger.warning("Gagal kirim Telegram: %s", e)
            sent += 1
        return sent

    def get_updates(self, offset: Optional[int], timeout: int = 25) -> List[Dict[str, Any]]:
        try:
            resp = requests.get(
                f"{self.base}/getUpdates",
                params={"offset": offset, "timeout": timeout},
                timeout=timeout + 10,
            )
            data = resp.json()
            return data.get("result", []) if data.get("ok") else []
        except Exception as e:
            logger.warning("Telegram getUpdates gagal: %s", e)
            time.sleep(2)
            return []


def get_server_ip() -> str:
    """Ambil IP publik keluar. PENTING: fallback socket-trick di bawah
    seringkali mengembalikan IP INTERNAL container (bukan IP publik asli
    yang dilihat Binance) di platform seperti Render/Railway/Heroku — kalau
    fallback ini yang terpakai, JANGAN dijadikan acuan untuk IP whitelist
    API key. Selalu ditandai jelas supaya tidak menyesatkan."""
    try:
        resp = requests.get("https://api.ipify.org", timeout=5)
        if resp.status_code == 200:
            return resp.text.strip()
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return f"{ip} (⚠️ fallback lokal — mungkin BUKAN IP publik asli, ipify.org tidak terjangkau)"
    except Exception:
        return "UNKNOWN"


# =============================================================================
# 7. UNIVERSE SCANNER (§6, §7)
# =============================================================================

def build_universe(
    bybit: BybitClient, binance_symbols: Optional[set], state: StateStore, top_n: int = 50
) -> List[str]:
    """Build scan universe from Bybit only. Binance symbols are optional compatibility input.

    Scanning must not require Binance REST. Binance contract filters are resolved only
    when a real candidate is actually about to be submitted.
    """
    ranked = bybit.get_ranked_symbols()
    if binance_symbols is None:
        ranked_symbols = [s for s, _ in ranked]
        available_symbols = {s for s, _ in ranked}
    else:
        ranked_symbols = [s for s, _ in ranked if s in binance_symbols]
        available_symbols = set(binance_symbols)

    excluded = set(state.positions.keys()) | set(state.bans.keys())
    universe: List[str] = []
    if "BTCUSDT" in available_symbols:
        universe.append("BTCUSDT")  # BTC context remains first without Binance REST.

    for sym in ranked_symbols:
        if len(universe) >= top_n:
            break
        if sym in excluded or sym in universe:
            continue
        universe.append(sym)
    return universe[:top_n]


# =============================================================================
# 8. BOT (orkestrasi worker & command handler)
# =============================================================================

def _preflight_local_contract() -> None:
    """Fail early with a precise message if main.py is paired with an incompatible strategy.py/learn.py."""
    import inspect
    missing = []
    required_strategy = ["Setup", "classify_volatility_regime", "validate_trailing_geometry"]
    for name in required_strategy:
        if not hasattr(strategy, name):
            missing.append(f"strategy.{name}")
    if hasattr(strategy, "Setup"):
        try:
            sig = inspect.signature(strategy.Setup)
            params = sig.parameters
            for name in ("viability", "quality_score", "execution_score", "context_score", "freshness_score", "expected_value_score"):
                if name not in params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    missing.append(f"strategy.Setup({name}=...)" )
        except Exception as exc:
            raise RuntimeError(f"Tidak bisa memeriksa kontrak strategy.Setup: {exc}") from exc
    if missing:
        raise RuntimeError(
            "Kontrak main.py tidak cocok dengan strategy.py. Missing/unsupported: " + ", ".join(missing)
            + ". Gunakan strategy.py dari paket revisi yang sama."
        )


class TradingBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.state_dir, exist_ok=True)

        self.state = StateStore(os.path.join(cfg.state_dir, "main_checkpoint.json"))
        self.strategy_engine = strategy.Strategy()
        self.learn_engine = learn.LearnEngine(
            checkpoint_path=os.path.join(cfg.state_dir, "learn_checkpoint.json"),
            ollama_url=cfg.ollama_url or None,
            ollama_api_key=cfg.ollama_api_key or None,
            git_enabled=cfg.git_autosave,
        )
        self.bybit = BybitClient(cfg.bybit_api_key, cfg.bybit_api_secret)
        self.binance = BinanceClient(cfg.binance_api_key, cfg.binance_api_secret)
        self.telegram = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id)
        self._telegram_error_handler = TelegramErrorLogHandler(lambda: getattr(self, "telegram", None))
        self._telegram_error_handler.setFormatter(TerminalFormatter())
        logging.getLogger().addHandler(self._telegram_error_handler)
        # Learn tidak melakukan network sendiri. Bila LearnEngine vNext
        # menyediakan notification sink, checkpoint/audit notification
        # dialirkan ke Telegram milik main.py. Tetap kompatibel dengan
        # LearnEngine lama agar launcher tidak crash saat repository belum
        # tersinkron penuh.
        self._learn_notification_sink_attached = False
        set_sink = getattr(self.learn_engine, "set_notification_sink", None)
        if callable(set_sink):
            set_sink(lambda text, level="INFO": self.telegram.send(text, level))
            self._learn_notification_sink_attached = True
        else:
            logger.warning("[LEARN] notification sink API belum tersedia; checkpoint Telegram fallback aktif")
        self.ws = BybitWebSocket(on_tick=self._on_tick, on_kline=self._on_kline)
        self.binance_ws = BinanceUserDataStream(
            cfg.binance_api_key,
            self._on_binance_user_event,
            self.binance._request,
            lambda: self.state.mode == "REAL" and not self.state.binance_paused,
            self._enter_binance_pause,
        )

        # Public Binance market WS is display-only. It never replaces Bybit
        # candles/strategy processing and never makes trading decisions.
        self.binance_position_ws = BinancePositionMarketWebSocket(self._on_binance_position_price)

        self._candle_cache: Dict[str, List[Dict[str, float]]] = {}
        # cache harga realtime dari WebSocket untuk log Telegram
        self._last_prices: Dict[str, float] = {}
        self._binance_mark_prices: Dict[str, float] = {}
        self._binance_mark_price_ts: Dict[str, float] = {}
        self._market_context: Dict[str, Any] = {}
        self._candle_lock = threading.Lock()
        self._trail_queue: "queue.Queue[str]" = queue.Queue()
        # Binance REST work that survives a rate-limit pause.
        self._binance_pending_queue: "queue.Queue[str]" = queue.Queue()
        self._binance_protection_queue: "queue.Queue[str]" = queue.Queue()
        self._binance_queue_lock = threading.Lock()
        self._binance_pending_queued: set = set()
        self._binance_protection_queued: set = set()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._last_update_id: Optional[int] = None
        self._last_freq_status: Optional[str] = None
        self._last_freq_alert_ts: float = 0.0
        self._shadow_candidates: Dict[str, Dict[str, Any]] = {}
        self._shadow_lock = threading.Lock()

    # -------------------------------------------------------------------
    # Binance position display market-data bridge
    # -------------------------------------------------------------------
    def _on_binance_position_price(self, symbol: str, price: float, event_ts: float) -> None:
        # Display/cache only. This callback never touches strategy, orders,
        # Binance REST, or local PnL accounting.
        self._binance_mark_prices[symbol] = price
        self._binance_mark_price_ts[symbol] = event_ts

    def _sync_binance_position_market_subscriptions(self) -> None:
        active_symbols = {
            str(p.get("pair") or "").upper()
            for p in self.state.snapshot_positions()
            if p.get("status") not in TERMINAL_STATES and p.get("pair")
        }
        self.binance_position_ws.sync_subscriptions(active_symbols)

    def _trade_display_price(self, symbol: str, fallback: float) -> tuple[float, str]:
        symbol = str(symbol or "").upper()
        if self.state.mode == "REAL":
            price = self._binance_mark_prices.get(symbol)
            ts = self._binance_mark_price_ts.get(symbol, 0.0)
            # Reject an obviously stale market stream so /trade never freezes
            # on an old price when Binance WS is reconnecting.
            if price is not None and (time.time() * 1000 - ts) <= 5000:
                return float(price), "BINANCE MARK"
        return float(fallback), "BYBIT WS"

    # -------------------------------------------------------------------
    # Startup / shutdown
    # -------------------------------------------------------------------
    def startup(self) -> None:
        label_main = self.state.load_checkpoint()
        label_learn = self.learn_engine.load()
        saved_strategy = self.learn_engine.strategy_state or self.state.strategy_state
        if saved_strategy:
            self.strategy_engine.load_state(saved_strategy)
        else:
            self.learn_engine.set_strategy_state(self.strategy_engine.export_state())
        # Jika restart terjadi saat Binance masih pause, pertahankan cooldown yang tersimpan.
        if self.state.mode == "REAL" and not self.state.binance_paused:
            try:
                # Reconcile positions/orders only. Balance REST is intentionally NOT
                # polled on startup; the saved local balance model is reused.
                self._reconcile_real_account_on_startup()
            except RateLimitError as e:
                self._enter_binance_pause(e)
            except Exception as e:
                logger.error("Startup reconciliation Binance gagal: %s", e)
                self.state.auto = False
                self.telegram.send(f"⚠️ STARTUP SAFETY HALT\nGagal sinkronisasi akun Binance: {e}\nAUTO = OFF", "ERROR")
        self._rebuild_binance_waiting_lists()
        logger.info("State dimuat (main=%s, learn=%s, strategy=%s)", label_main, label_learn, self.strategy_engine.version)
        logger.info("[BALANCE] policy=ONE_SNAPSHOT_PLUS_LOCAL_PNL | source=%s | snapshot=%s | current=%s",
                    self.state.real_last_balance_source, self.state.real_balance_snapshot, self.state.current_balance)

        ip = "(cek /ip bila diperlukan)"
        # Jika dijalankan lewat try.py Render, launcher sudah menjadi pemilik Telegram getUpdates.
        # Hindari 409 Conflict karena dua polling berjalan bersamaan.
        launcher_mode = os.environ.get("RUN_WITH_LAUNCHER", "false").lower() == "true"
        # Saat dijalankan melalui try.py: polling Telegram milik launcher.
        # Telegram outbound memakai queue yang dipompa oleh worker yang sudah ada; tidak membuat thread ke-6.
        self.telegram.start_sender()
        if self.state.binance_paused:
            now = time.time()
            remaining = max(0, int((self.state.binance_pause_until or now) - now))
            ban_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.state.binance_pause_ts)) if self.state.binance_pause_ts else "-"
            ready_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.state.binance_pause_until)) if self.state.binance_pause_until else "-"
            binance_status = (
                "🔴 BAN / RATE-LIMIT\n"
                f"Mulai: {ban_at}\n"
                f"READY setelah: {ready_at}\n"
                f"Cooldown: {remaining} detik"
            )
        else:
            binance_status = "🟢 READY"
        self.telegram.send(
            f"🤖 BOT STARTED — NEW MAIN SESSION\n\n"
            f"Status: ONLINE\nMode: {self.state.mode}\nServer IP: {ip}\n"
            f"Binance REST: {binance_status}\n\n"
            "Ketik /healthz untuk status lengkap.\n"
            "Ketik /auto untuk memulai scanning.",
            "BOT_START",
        )
        logger.info("[MAIN] START — new session | mode=%s | binance=%s", self.state.mode, "BAN" if self.state.binance_paused else "READY")

        self.ws.start()
        self.binance_position_ws.start()
        if self.state.mode == "REAL":
            self.binance_ws.start()
        for pos in self.state.snapshot_positions():
            if pos.get("status") in ("PENDING", "FILLED", "PROTECTED", "TRAILING"):
                self.ws.subscribe(pos["pair"])
        self._sync_binance_position_market_subscriptions()

        for target, name in (
            (self._worker_scanner, "Worker1-Scanner"),
            (self._worker_learn, "Worker3-Learn"),
            # Worker command internal dimatikan saat Render launcher aktif.
            # try.py akan meneruskan update ke handle_update().
            *(([(self._worker_command_handler, "Worker4-Command")] ) if not launcher_mode else []),
            (self._worker_ban_timer, "Worker5-BanTimer"),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def _reconcile_real_account_on_startup(self) -> None:
        """Reconcile local bot state with the authoritative Binance account before AUTO resumes."""
        if self.state.mode != "REAL":
            return
        live_positions = self.binance.get_all_position_risk()
        live_orders = self.binance.get_all_open_orders()
        live_pos_symbols = {str(r.get("symbol", "")) for r in live_positions if abs(float(r.get("positionAmt", 0) or 0)) > 0}
        live_order_symbols = {str(o.get("symbol", "")) for o in live_orders if o.get("symbol")}
        local_active = {p["pair"]: p for p in self.state.snapshot_positions() if p.get("status") not in TERMINAL_STATES}
        unknown_positions = sorted(live_pos_symbols - set(local_active))
        unknown_orders = sorted(live_order_symbols - set(local_active))
        missing_local = sorted(set(local_active) - live_pos_symbols - live_order_symbols)
        if unknown_positions or unknown_orders or missing_local:
            self.state.auto = False
            self.telegram.send(
                "⚠️ STARTUP RECONCILIATION — AUTO = OFF\n"
                f"Untracked Binance position: {', '.join(unknown_positions) or '-'}\n"
                f"Untracked Binance order: {', '.join(unknown_orders) or '-'}\n"
                f"Local state tanpa order/position Binance: {', '.join(missing_local) or '-'}\n"
                "Periksa /trade lalu sinkronkan/bersihkan akun sebelum /auto.",
                "ERROR",
            )
            logger.error("Startup reconciliation mismatch: unknown_positions=%s unknown_orders=%s missing_local=%s", unknown_positions, unknown_orders, missing_local)

    def _rebuild_binance_waiting_lists(self) -> None:
        """Reconstruct transient REST queues after restart from persistent position state."""
        if self.state.mode != "REAL":
            return
        for pos in self.state.snapshot_positions():
            symbol = pos.get("pair")
            if not symbol:
                continue
            status = pos.get("status")
            ids = pos.get("binance_order_ids", {}) or {}
            if status == "BINANCE_WAITING" and not ids.get("entry"):
                self._queue_binance_pending(symbol)
            elif status == "PENDING" and not ids.get("entry"):
                self._queue_binance_pending(symbol)
            elif status == "FILLED" and (not ids.get("sl") or not ids.get("tp")):
                self._queue_binance_protection(symbol)
            elif status in ("PROTECTED", "TRAILING") and not ids.get("sl"):
                self._queue_binance_protection(symbol)
            if pos.get("_binance_cleanup_pending"):
                self._queue_binance_protection(symbol)
            if pos.get("_pending_trail_sl") is not None:
                self._trail_queue.put(symbol)

    def shutdown(self, fresh_session: bool = False) -> None:
        """Stop every main worker. For launcher /end, optionally start fresh next time.

        SIMULASI can safely discard its runtime checkpoint. REAL runtime state is
        retained when there are active/reserved positions so a stop cannot orphan
        live Binance risk. Learning memory is always preserved.
        """
        self._stop.set()
        self.state.auto = False
        self.telegram.send("🛑 BOT STOP\n\nStatus: OFFLINE", "BOT_STOP")
        self.ws.stop()
        self.binance_position_ws.stop()
        self.binance_ws.stop()
        # Give owned worker threads a short grace period so /end really releases
        # the current main.py runtime before the launcher drops the module.
        deadline = time.monotonic() + 3.0
        for t in list(self._threads):
            if t is threading.current_thread():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if t.is_alive():
                t.join(timeout=min(1.0, remaining))
        active = self.state.get_active_count()
        if fresh_session and (self.state.mode != "REAL" or active == 0):
            self.learn_engine.save_checkpoint()
            self.telegram.flush(max_messages=20)
            self.state.clear_runtime_checkpoint()
            logger.info("[MAIN] COLD STOP — runtime checkpoint dihapus; learning memory dipertahankan")
        else:
            self.state.save_checkpoint()
            self.learn_engine.save_checkpoint()
            self.telegram.flush(max_messages=20)
            if fresh_session and self.state.mode == "REAL" and active > 0:
                logger.warning("[MAIN] COLD STOP ditahan untuk REAL aktif: %s posisi/reserve tetap disimpan sebagai safety checkpoint", active)
                self.telegram.send(
                    f"⚠️ REAL SAFETY CHECKPOINT DIPERTAHANKAN\nAda {active} posisi/reserve aktif.\nRuntime tidak dihapus agar posisi Binance tidak menjadi orphan saat /try berikutnya.",
                    "WARNING",
                )
        try:
            logging.getLogger().removeHandler(self._telegram_error_handler)
            self._telegram_error_handler.close()
        except Exception:
            pass
        self.telegram.stop()

    # -------------------------------------------------------------------
    # WebSocket callbacks (Worker 2)
    # -------------------------------------------------------------------
    def _on_binance_user_event(self, data: Dict[str, Any]) -> None:
        """Authoritative REAL-mode order/position events from Binance WS.

        This callback intentionally performs NO Binance REST request. Therefore
        a 418/429 REST ban cannot stop fill/exit detection.
        """
        event = str(data.get("e", ""))
        event_ts = float(data.get("E") or data.get("T") or time.time() * 1000)
        if event == "listenKeyExpired":
            self.binance_ws.invalidate_listen_key()
            logger.error("[BINANCE] USERDATA LISTENKEY EXPIRED")
            self.telegram.send("⚠️ Binance User Data WS expired — akan reconnect setelah stream tersedia.", "ERROR")
            return
        if event == "ACCOUNT_UPDATE":
            account = data.get("a") or {}
            for bal in account.get("B") or []:
                if str(bal.get("a") or "") == "USDT":
                    try:
                        wallet_balance = float(bal.get("wb") or 0.0)
                        self._reconcile_real_balance_from_ws(wallet_balance, "BINANCE_WS_ACCOUNT_UPDATE")
                    except (TypeError, ValueError):
                        pass
            for item in account.get("P") or []:
                symbol = str(item.get("s") or "")
                if not symbol:
                    continue
                try:
                    amt = float(item.get("pa") or 0.0)
                except (TypeError, ValueError):
                    amt = 0.0
                pos = self.state.positions.get(symbol)
                if pos is None:
                    continue
                pos["binance_position_amt"] = amt
                pos["binance_entry_price"] = float(item.get("ep") or pos.get("fill_price") or pos.get("entry") or 0.0)
                pos["binance_unrealized_pnl"] = float(item.get("up") or 0.0)
                pending_outcome = pos.get("_pending_real_close_outcome")
                if pending_outcome and abs(amt) <= 0:
                    self._finalize_real_close_from_user_event(symbol, pos, pending_outcome,
                                                              float(pos.get("_pending_real_close_price") or pos.get("fill_price") or pos.get("entry") or 0.0),
                                                              float(pos.get("_pending_real_close_time") or event_ts))
            return

        if event != "ORDER_TRADE_UPDATE":
            return
        order = data.get("o") or {}
        symbol = str(order.get("s") or "")
        if not symbol:
            return
        client_id = str(order.get("c") or "")
        order_id = order.get("i")
        status = str(order.get("X") or "")
        exec_type = str(order.get("x") or "")
        order_type = str(order.get("o") or "")
        avg_price = float(order.get("ap") or order.get("L") or 0.0)
        cumulative_qty = float(order.get("z") or 0.0)
        pos = self.state.positions.get(symbol)
        if not pos:
            return

        ids = pos.setdefault("binance_order_ids", {})
        role = None
        priority_keys = ["entry", "tp", "trail", "sl"]
        for k in priority_keys:
            v = ids.get(k)
            if v is not None and str(v) == str(order_id):
                role = k
                break
        if role is None:
            if client_id and client_id == str(pos.get("binance_entry_client_order_id") or ""):
                role = "entry"
        if role is None:
            # Exit orders may arrive after restart. Match known stop/tp ids by string only.
            return

        if role == "entry":
            if status in ("PARTIALLY_FILLED", "FILLED") and cumulative_qty > 0:
                pos["real_filled_qty"] = cumulative_qty
                pos["fill_price"] = avg_price or pos.get("fill_price") or pos.get("entry")
                pos["real_fill_confirmed"] = True
                pos["binance_entry_status"] = status
                pos["binance_entry_exec_type"] = exec_type
                if pos.get("status") in ("BINANCE_WAITING", "PENDING"):
                    event_id = f"{symbol}:FILLED_WS:{pos['created_at']}:{status}:{cumulative_qty}"
                    if self.state.transition(
                        symbol, "FILLED", event_id,
                        fill_time=event_ts,
                        fill_price=pos["fill_price"],
                        real_fill_confirmed=True,
                    ):
                        with self._shadow_lock:
                            self._shadow_candidates.pop(symbol, None)
                        self.telegram.send(
                            f"✅ FILLED — {symbol}\nEntry: {float(pos['fill_price']):.8f}\nArah: {pos['direction']}\nSource: Binance WS",
                            "FILLED",
                        )
                        self._attach_real_protection(symbol, pos)
                elif pos.get("status") in ("PROTECTED", "TRAILING"):
                    # Additional fill quantity: protection must be reconciled later.
                    self._queue_binance_protection(symbol)
                return
            if status in ("CANCELED", "EXPIRED", "EXPIRED_IN_MATCH") and pos.get("status") == "PENDING":
                self.state.transition(symbol, "CANCELLED", f"{symbol}:ENTRY_CANCELLED_WS:{pos['created_at']}:{status}", close_reason=f"BINANCE_{status}")
                self.state.remove_terminal(symbol)
                self.ws.unsubscribe(symbol)
                self.telegram.send(f"⚠️ ENTRY {status} — {symbol}\nBinance membatalkan order.", "WARNING")
            return

        if role in ("sl", "tp", "trail") and status == "FILLED":
            outcome = "TP" if role == "tp" else ("TRAIL" if role == "trail" else "INITIAL_SL")
            exit_price = avg_price or float(order.get("L") or pos.get("sl") or pos.get("tp") or pos.get("entry") or 0.0)
            pos["_pending_real_close_outcome"] = outcome
            pos["_pending_real_close_price"] = exit_price
            pos["_pending_real_close_time"] = event_ts
            pos["binance_exit_order_id"] = order_id
            logger.info("[BINANCE] %s EXIT FILLED — %s price=%.8f", symbol, outcome, exit_price)
            # ACCOUNT_UPDATE will confirm flat; if it has already arrived, finalize immediately.
            if abs(float(pos.get("binance_position_amt", 0.0) or 0.0)) <= 0:
                self._finalize_real_close_from_user_event(symbol, pos, outcome, exit_price, event_ts)

    def _finalize_real_close_from_user_event(self, symbol: str, pos: Dict[str, Any], outcome: str, price: float, ts_ms: float) -> None:
        if pos.get("status") not in ("FILLED", "PROTECTED", "TRAILING"):
            return
        event_id = f"{symbol}:CLOSED_WS:{outcome}:{pos['created_at']}"
        risk = abs(float(pos["entry"]) - float(pos.get("initial_sl", pos["sl"]))) or 1e-9
        pnl_r = (price - pos["entry"]) / risk if pos["direction"] == "BUY" else (pos["entry"] - price) / risk
        pnl_pct = ((price - pos["entry"]) / pos["entry"] * 100) if pos["direction"] == "BUY" else ((pos["entry"] - price) / pos["entry"] * 100)
        applied = self.state.transition(
            symbol, "CLOSED", event_id,
            close_price=price, close_time=ts_ms, close_reason=outcome,
            pnl_pct=pnl_pct, pnl_r=pnl_r,
        )
        if not applied:
            return
        self.learn_engine.record_trade_outcome(pos, outcome, {
            "pnl_pct": pnl_pct, "pnl_r": pnl_r, "close_time": ts_ms, "trail_count": pos.get("trail_count", 0),
        })
        self.state.ban(symbol, outcome, 24 * 3600)
        self.telegram.send(
            f"{'🟢' if pnl_pct >= 0 else '🔴'} {outcome} {pnl_pct:+.2f}% — {symbol} | C{pos['confidence']:.0f}%\nSource: Binance WS",
            outcome if outcome in IMPORTANT_EVENTS else "INFO",
        )
        pos.pop("_pending_real_close_outcome", None)
        pos.pop("_pending_real_close_price", None)
        pos.pop("_pending_real_close_time", None)
        # Do not use REST while banned. Remaining protective orders are harmless close-only
        # orders; cleanup is attempted later by the recovery queue/startup reconciliation.
        if self.state.binance_paused:
            pos["_binance_cleanup_pending"] = True
        else:
            try:
                self._cleanup_real_symbol(symbol, pos, close_position=False)
            except Exception as e:
                pos["_binance_cleanup_pending"] = True
                logger.warning("Post-close cleanup %s ditunda: %s", symbol, e)
        self.state.remove_terminal(symbol)
        self.ws.unsubscribe(symbol)
        self.binance_position_ws.unsubscribe(symbol)

    def _on_kline(self, symbol: str, candle: Dict[str, float], confirm: bool) -> None:
        with self._candle_lock:
            buf = self._candle_cache.setdefault(symbol, [])
            if buf and buf[-1]["t"] == candle["t"]:
                buf[-1] = candle
            else:
                buf.append(candle)
            if len(buf) > 700:
                del buf[: len(buf) - 700]
        if confirm:
            self._evaluate_position_monitoring(symbol)

    def _on_tick(self, symbol: str, price: float, ts: float) -> None:
        self._last_prices[symbol] = price
        self._check_shadow_candidate(symbol, price, ts)
        pos = self.state.positions.get(symbol)
        if not pos:
            return
        if pos["status"] == "PENDING":
            self._check_pending_fill(symbol, pos, price, ts)
        elif pos["status"] in ("FILLED", "PROTECTED", "TRAILING"):
            self._check_tp_sl(symbol, pos, price, ts)

    def _check_shadow_candidate(self, symbol: str, price: float, ts: float) -> None:
        with self._shadow_lock:
            cand = self._shadow_candidates.get(symbol)
            if not cand:
                return
            if cand.get("expires_at", 0.0) < time.time():
                self._shadow_candidates.pop(symbol, None)
                return
            direction = cand["direction"]
            state = cand["shadow_state"]
            if state == "WAIT_ENTRY":
                entered = (direction == "BUY" and price <= cand["entry"]) or (direction == "SELL" and price >= cand["entry"])
                tp_first = (direction == "BUY" and price >= cand["tp"]) or (direction == "SELL" and price <= cand["tp"])
                if tp_first and not entered:
                    self.learn_engine.record_shadow_outcome(cand, "TIMEOUT", 0.0)
                    self._shadow_candidates.pop(symbol, None)
                    return
                if entered:
                    cand["shadow_state"] = "ENTERED"
                    cand["fill_price"] = cand["entry"]
                    return
            elif state == "ENTERED":
                risk = abs(float(cand["entry"]) - float(cand["sl"])) or 1e-9
                hit_tp = (direction == "BUY" and price >= cand["tp"]) or (direction == "SELL" and price <= cand["tp"])
                hit_sl = (direction == "BUY" and price <= cand["sl"]) or (direction == "SELL" and price >= cand["sl"])
                if hit_tp or hit_sl:
                    pnl_r = ((price - cand["entry"]) / risk) if direction == "BUY" else ((cand["entry"] - price) / risk)
                    outcome = "TP" if hit_tp else "INITIAL_SL"
                    self.learn_engine.record_shadow_outcome(cand, outcome, pnl_r)
                    self._shadow_candidates.pop(symbol, None)

    def _check_pending_fill(self, symbol: str, pos: Dict[str, Any], price: float, ts: float) -> None:
        direction = pos["direction"]
        entry = float(pos["entry"])
        if self.state.mode == "REAL" and not pos.get("binance_order_ids", {}).get("entry"):
            return  # tidak ada order Binance nyata => harga Bybit tidak boleh membuat phantom-fill
        filled_by_market = (direction == "BUY" and price <= entry) or (direction == "SELL" and price >= entry)
        if not filled_by_market:
            tp_hit_first = (direction == "BUY" and price >= float(pos["tp"])) or (direction == "SELL" and price <= float(pos["tp"]))
            if tp_hit_first and (self.state.mode != "REAL" or pos.get("binance_order_ids", {}).get("entry")):
                self._handle_timeout(symbol, pos)
            return

        if self.state.mode == "REAL":
            # REAL fill is authoritative from Binance ORDER_TRADE_UPDATE.
            # Do not poll positionRisk from the Bybit market tick path: that was
            # the primary source of request amplification leading to 418 bans.
            return
        else:
            event_id = f"{symbol}:FILLED:{pos['created_at']}"
            if not self.state.transition(symbol, "FILLED", event_id, fill_time=ts, fill_price=price, real_fill_confirmed=False):
                return

        with self._shadow_lock:
            self._shadow_candidates.pop(symbol, None)
        self.telegram.send(f"✅ FILLED — {symbol}\nEntry: {pos.get('fill_price', price)}\nArah: {direction}", "FILLED")
        if self.state.mode == "REAL":
            self._attach_real_protection(symbol, pos)
        else:
            self.state.transition(symbol, "PROTECTED", f"{symbol}:PROTECTED:{pos['created_at']}")

    def _queue_binance_protection(self, symbol: str) -> None:
        with self._binance_queue_lock:
            if symbol not in self._binance_protection_queued:
                self._binance_protection_queued.add(symbol)
                self._binance_protection_queue.put(symbol)

    def _queue_binance_pending(self, symbol: str) -> None:
        with self._binance_queue_lock:
            if symbol not in self._binance_pending_queued:
                self._binance_pending_queued.add(symbol)
                self._binance_pending_queue.put(symbol)

    def _attach_real_protection(self, symbol: str, pos: Dict[str, Any]) -> bool:
        """Pasang TP/SL setelah FILLED Binance sudah dikonfirmasi."""
        try:
            if not pos.get("real_fill_confirmed"):
                logger.warning("Protection %s ditunda: Binance fill belum confirmed", symbol)
                self._queue_binance_protection(symbol)
                return False

            side_close = "SELL" if pos["direction"] == "BUY" else "BUY"
            ids = pos.setdefault("binance_order_ids", {})
            protection_qty = Decimal(str(pos.get("real_filled_qty") or pos.get("qty") or "0"))
            if not ids.get("sl"):
                filters = self.binance.get_symbol_filters(symbol)
                prices = normalize_order_prices(pos, filters["tick_size"])
                pos["entry"], pos["sl"], pos["tp"] = prices["entry"], prices["sl"], prices["tp"]
                sl_order = self.binance.place_stop_market(
                    symbol, side_close, Decimal(str(pos["sl"])), pos["direction"],
                    quantity=protection_qty
                )
                ids["sl"] = sl_order.get("orderId")
                time.sleep(1)
            if not ids.get("tp"):
                tp_order = self.binance.place_take_profit_market(
                    symbol, side_close, Decimal(str(pos["tp"])), pos["direction"],
                    quantity=protection_qty
                )
                ids["tp"] = tp_order.get("orderId")
            self.state.transition(symbol, "PROTECTED", f"{symbol}:PROTECTED:{pos['created_at']}")
            self.telegram.send(
                f"🛡️ PROTECTION ACTIVE — {symbol}\nTP: {pos['tp']}\nSL: {pos['sl']}", "INFO"
            )
            return True
        except RateLimitError as e:
            self._queue_binance_protection(symbol)
            self._enter_binance_pause(e)
            return False
        except ExchangeError as e:
            logger.error("Gagal pasang protective order %s: %s", symbol, e)
            self.telegram.send(f"⚠️ ERROR — gagal pasang TP/SL {symbol}: {e}", "ERROR")
            return False

    def _process_binance_waiting_lists(self) -> None:
        """Drain REAL Binance work after cooldown, one exchange step at a time."""
        if self.state.binance_paused or self.state.mode != "REAL":
            return

        try:
            symbol = self._binance_protection_queue.get_nowait()
        except queue.Empty:
            symbol = None
        if symbol:
            with self._binance_queue_lock:
                self._binance_protection_queued.discard(symbol)
            pos = self.state.positions.get(symbol)
            if pos and pos.get("status") == "FILLED":
                self._attach_real_protection(symbol, pos)
            time.sleep(1.0)
            return

        try:
            symbol = self._binance_pending_queue.get_nowait()
        except queue.Empty:
            symbol = None
        if not symbol:
            return

        with self._binance_queue_lock:
            self._binance_pending_queued.discard(symbol)
        pos = self.state.positions.get(symbol)
        if not pos or pos.get("status") != "BINANCE_WAITING":
            return

        try:
            filters = self.binance.get_symbol_filters(symbol)
            leverage = int(pos.get("leverage", self.state.leverage))
            if not pos.get("binance_leverage_confirmed"):
                self.binance.set_leverage(symbol, leverage)
                pos["binance_leverage_confirmed"] = True

            side = "BUY" if pos["direction"] == "BUY" else "SELL"
            tick = filters["tick_size"]
            entry_price = round_price(
                pos["entry"], tick, ROUND_DOWN if pos["direction"] == "BUY" else ROUND_UP
            )
            client_id = pos.get("binance_entry_client_order_id") or self._make_client_order_id(symbol, pos)
            pos["binance_entry_client_order_id"] = client_id
            order = self.binance.place_limit_order(
                symbol, side, Decimal(str(pos["qty"])), entry_price, pos["direction"],
                client_order_id=client_id,
            )
            order_id = order.get("orderId")
            if not order_id:
                raise ExchangeError(f"Binance limit {symbol} diterima tanpa orderId")
            pos["binance_order_ids"]["entry"] = order_id
            pos["binance_entry_confirmed_at"] = time.time()
            event_id = f"{symbol}:BINANCE_ENTRY_CONFIRMED:{pos['created_at']}"
            if self.state.transition(symbol, "PENDING", event_id, binance_entry_confirmed=True):
                self.ws.subscribe(symbol)
                self.binance_position_ws.subscribe(symbol)
                self.telegram.send(
                    f"🎯 PENDING ORDER — {symbol}\n"
                    f"Binance order terkonfirmasi\nEntry: {pos['entry']}",
                    "PENDING",
                )
        except RateLimitError as e:
            self._queue_binance_pending(symbol)
            self._enter_binance_pause(e)
            return
        except ExchangeError as e:
            logger.error("Waiting limit order gagal %s: %s", symbol, e)
            self.telegram.send(f"⚠️ ERROR — waiting limit {symbol}: {e}", "ERROR")
            self.state.discard_pending(symbol)
            self.ws.unsubscribe(symbol)
        time.sleep(1.0)

    def _check_tp_sl(self, symbol: str, pos: Dict[str, Any], price: float, ts: float) -> None:
        direction = pos["direction"]
        pos["peak_price"] = max(pos.get("peak_price", price), price) if direction == "BUY" else min(pos.get("peak_price", price), price)

        hit_tp = (direction == "BUY" and price >= pos["tp"]) or (direction == "SELL" and price <= pos["tp"])
        hit_sl = (direction == "BUY" and price <= pos["sl"]) or (direction == "SELL" and price >= pos["sl"])

        if self.state.mode == "REAL":
            # Binance protective orders are authoritative. Bybit price is used only
            # as monitoring/trailing context, never as a REAL close confirmation.
            return
        if hit_tp:
            self._close_position(symbol, pos, "TP", price, ts)
        elif hit_sl:
            outcome = "TRAIL" if pos.get("trail_count", 0) > 0 else "INITIAL_SL"
            self._close_position(symbol, pos, outcome, price, ts)

    def _set_real_balance_snapshot(self, balance: float, source: str = "BINANCE_REST") -> None:
        """Initialize/reset the local REAL balance model from one Binance snapshot."""
        bal = float(balance)
        if not math.isfinite(bal) or bal < 0:
            raise ValueError(f"Invalid REAL balance snapshot: {balance}")
        with self.state._lock:
            self.state.real_balance_snapshot = bal
            self.state.real_localized_pnl = 0.0
            self.state.real_balance_adjustment = 0.0
            self.state.real_balance_snapshot_ts = time.time()
            self.state.real_last_balance_source = source
            self.state.current_balance = bal
            self.state.highest_balance = bal if self.state.highest_balance is None else max(self.state.highest_balance, bal)
        logger.info("[BALANCE] REAL SNAPSHOT | %.8f source=%s", bal, source)

    def _apply_real_balance_event(self, price_pnl: float, reason: str) -> None:
        """Apply verified trade PnL locally; NEVER call Binance balance REST here."""
        pnl = float(price_pnl)
        if not math.isfinite(pnl):
            return
        with self.state._lock:
            if self.state.real_balance_snapshot is None:
                # Cannot invent a base balance. Preserve current value only.
                logger.warning("[BALANCE] LOCAL PNL skipped without REAL snapshot | reason=%s pnl=%.8f", reason, pnl)
                return
            self.state.real_localized_pnl += pnl
            local_balance = (self.state.real_balance_snapshot
                             + self.state.real_localized_pnl
                             + self.state.real_balance_adjustment)
            self.state.current_balance = local_balance
            if self.state.highest_balance is None:
                self.state.highest_balance = local_balance
            else:
                self.state.highest_balance = max(self.state.highest_balance, local_balance)
            source = self.state.real_last_balance_source or "LOCAL"
            self.state.real_last_balance_source = f"{source}|LOCAL_PNL"
        logger.info("[BALANCE] LOCAL PNL | %.8f | balance=%.8f | reason=%s", pnl, local_balance, reason)
        self._check_local_autostop(reason)

    def _reconcile_real_balance_from_ws(self, wallet_balance: float, source: str = "BINANCE_WS") -> None:
        """Use Binance ACCOUNT_UPDATE as an optional correction, without REST polling."""
        bal = float(wallet_balance)
        if not math.isfinite(bal) or bal < 0:
            return
        with self.state._lock:
            if self.state.real_balance_snapshot is None:
                self.state.real_balance_snapshot = bal
                self.state.real_balance_snapshot_ts = time.time()
                self.state.real_localized_pnl = 0.0
                self.state.real_balance_adjustment = 0.0
            expected_local = (self.state.real_balance_snapshot
                              + self.state.real_localized_pnl)
            self.state.real_balance_adjustment = bal - expected_local
            self.state.current_balance = bal
            self.state.real_last_balance_source = source
            if self.state.highest_balance is None:
                self.state.highest_balance = bal
            else:
                self.state.highest_balance = max(self.state.highest_balance, bal)
        logger.info("[BALANCE] WS RECONCILE | wallet=%.8f adjustment=%+.8f source=%s",
                    bal, self.state.real_balance_adjustment, source)
        self._check_local_autostop("ACCOUNT_UPDATE")

    def _check_local_autostop(self, reason: str) -> None:
        with self.state._lock:
            bal = self.state.current_balance
            high = self.state.highest_balance
            limit = self.state.autostop_pct
            auto = self.state.auto
        if bal is None or high is None or limit is None or high <= 0 or not auto:
            return
        dd = (high - bal) / high * 100.0
        if dd >= limit:
            with self.state._lock:
                self.state.auto = False
            self.telegram.send(
                f"🛑 AUTOSTOP TERPICU\nDrawdown: {dd:.2f}% >= batas {limit}%\nAUTO = OFF\nSource: {reason}",
                "AUTOSTOP",
            )

    def _refresh_real_balance_after_event(self, reason: str) -> None:
        """Compatibility hook: balance is LOCAL now; no Binance REST polling."""
        if self.state.mode != "REAL":
            return
        logger.info("[BALANCE] REST POLL SKIPPED | balance model local | reason=%s", reason)
        self._check_local_autostop(reason)

    def _close_position(self, symbol: str, pos: Dict[str, Any], outcome: str, price: float, ts: float) -> None:
        if self.state.mode == "REAL":
            # REAL exits are finalized from Binance ORDER_TRADE_UPDATE + ACCOUNT_UPDATE.
            return
        if self.state.mode == "REAL":
            try:
                self._cleanup_real_symbol(symbol, pos, close_position=True)
            except RateLimitError as e:
                self._enter_binance_pause(e)
                self.telegram.send(f"⚠️ {outcome} {symbol} terdeteksi dari harga WS, tetapi Binance belum bisa diverifikasi karena rate-limit. Posisi TIDAK ditutup di state lokal.", "ERROR")
                return
            except ExchangeError as e:
                logger.error("Real close belum terverifikasi %s: %s", symbol, e)
                self.telegram.send(f"⚠️ ERROR CLOSE — {symbol} belum dipastikan flat di Binance: {e}", "ERROR")
                return
        event_id = f"{symbol}:CLOSED:{outcome}:{pos['created_at']}"
        risk = abs(float(pos["entry"]) - float(pos.get("initial_sl", pos["sl"]))) or 1e-9
        pnl_r = (price - pos["entry"]) / risk if pos["direction"] == "BUY" else (pos["entry"] - price) / risk
        pnl_pct = ((price - pos["entry"]) / pos["entry"] * 100) if pos["direction"] == "BUY" else ((pos["entry"] - price) / pos["entry"] * 100)

        applied = self.state.transition(
            symbol, "CLOSED", event_id, close_price=price, close_time=ts, close_reason=outcome,
            pnl_pct=pnl_pct, pnl_r=pnl_r,
        )
        if not applied:
            return

        qty = Decimal(str(pos.get("real_filled_qty") or pos.get("qty") or "0"))
        entry_d = Decimal(str(pos["entry"]))
        exit_d = Decimal(str(price))
        price_pnl = (exit_d - entry_d) * qty if pos["direction"] == "BUY" else (entry_d - exit_d) * qty
        if self.state.mode == "REAL":
            self._apply_real_balance_event(float(price_pnl), f"close:{symbol}:{outcome}")
        else:
            self.state.sim_balance += float(price_pnl)

        with self._candle_lock:
            trace_candles = list(self._candle_cache.get(symbol, []))
        fill_ms = float(pos.get("fill_time", 0) or 0)
        trace_path = [c for c in trace_candles if fill_ms <= 0 or float(c.get("t", 0)) >= fill_ms]
        self.learn_engine.record_trade_outcome(pos, outcome, {
            "pnl_pct": pnl_pct, "pnl_r": pnl_r, "close_time": ts, "trail_count": pos.get("trail_count", 0),
            "trail_history": list(pos.get("trail_history", [])), "path_candles": trace_path[-500:],
        })
        self.state.ban(symbol, outcome, 24 * 3600)  # §26 post-trade ban 24 jam
        self.telegram.send(
            f"{'🟢' if pnl_pct >= 0 else '🔴'} {outcome} {pnl_pct:+.2f}% — {symbol} | C{pos['confidence']:.0f}%",
            outcome if outcome in IMPORTANT_EVENTS else "INFO",
        )
        self.state.remove_terminal(symbol)
        self.ws.unsubscribe(symbol)

    def _handle_timeout(self, symbol: str, pos: Dict[str, Any]) -> None:
        if self.state.mode == "REAL":
            if self.state.binance_paused:
                logger.warning("Timeout %s tertahan: Binance sedang pause; state dipertahankan", symbol)
                return
            try:
                self._cleanup_real_symbol(symbol, pos, close_position=False)
            except (RateLimitError, ExchangeError) as e:
                if isinstance(e, RateLimitError):
                    self._enter_binance_pause(e)
                self.telegram.send(f"⚠️ TIMEOUT {symbol} belum ditutup lokal karena cleanup Binance gagal: {e}", "ERROR")
                return
        event_id = f"{symbol}:TIMEOUT:{pos['created_at']}"
        if not self.state.transition(symbol, "TIMEOUT", event_id, close_time=time.time() * 1000, close_reason="TP_BEFORE_ENTRY"):
            return
        self.learn_engine.record_trade_outcome(pos, "TIMEOUT", {"pnl_pct": 0.0, "pnl_r": 0.0, "close_time": time.time() * 1000, "trail_count": 0})
        self.state.ban(symbol, "TIMEOUT", 12 * 3600)
        self.telegram.send(f"⏱️ TIMEOUT — {symbol}\nTP tersentuh sebelum entry terisi.", "TIMEOUT")
        self.state.remove_terminal(symbol)
        self.ws.unsubscribe(symbol)
        self.binance_position_ws.unsubscribe(symbol)

    def _cleanup_real_account(self) -> None:
        """Emergency/manual timeout cleanup for the entire Binance Futures account."""
        open_orders = self.binance.get_all_open_orders()
        symbols = {str(o.get("symbol", "")) for o in open_orders if o.get("symbol")}
        positions = self.binance.get_all_position_risk()
        symbols.update(str(r.get("symbol", "")) for r in positions if r.get("symbol"))
        for idx, symbol in enumerate(sorted(symbols), 1):
            logger.info("[TIMEOUT ALL] CANCEL OPEN ORDERS %02d/%02d | %s", idx, len(symbols), symbol, extra={"symbol": symbol})
            self.binance.cancel_all_open_orders(symbol)
            # Explicit delay is intentionally redundant with the governor:
            # cleanup is a safety operation, not latency-sensitive trading.
            time.sleep(0.5)
        # Flatten every real position, including positions not present in local state.
        for risk in positions:
            symbol = str(risk.get("symbol", ""))
            amt = Decimal(str(risk.get("positionAmt", "0")))
            if not symbol or amt == 0:
                continue
            actual_dir = "BUY" if amt > 0 else "SELL"
            # Hedge Mode exposes positionSide; preserve that side when closing.
            position_side = str(risk.get("positionSide", ""))
            if position_side == "LONG":
                actual_dir = "BUY"
            elif position_side == "SHORT":
                actual_dir = "SELL"
            side_close = "SELL" if actual_dir == "BUY" else "BUY"
            filters = self.binance.get_symbol_filters(symbol)
            qty = round_step(abs(amt), filters["step_size"])
            if qty > 0:
                logger.info("[TIMEOUT ALL] FLATTEN %s qty=%s", symbol, qty, extra={"symbol": symbol})
                self.binance.place_market_order(symbol, side_close, qty, actual_dir)
                time.sleep(0.75)
        # Verify globally empty account state.
        for risk in self.binance.get_all_position_risk():
            if abs(float(risk.get("positionAmt", 0))) > 0:
                raise ExchangeError(f"Posisi Binance masih aktif: {risk.get('symbol')} {risk.get('positionAmt')}")
        leftovers = self.binance.get_all_open_orders()
        if leftovers:
            for o in leftovers:
                sym = o.get("symbol")
                if sym:
                    self.binance.cancel_all_open_orders(sym)
            leftovers = self.binance.get_all_open_orders()
        if leftovers:
            raise ExchangeError(f"Masih ada {len(leftovers)} open order Binance setelah cleanup")

    def _cleanup_real_symbol(self, symbol: str, pos: Dict[str, Any], close_position: bool) -> None:
        """Cancel bot/open orders and optionally flatten a real position. Raises on uncertainty."""
        self.binance.cancel_all_open_orders(symbol)
        risk = self.binance.get_position_risk(symbol)
        if risk:
            if not close_position:
                raise ExchangeError(f"Binance masih memiliki posisi aktif {symbol}; cleanup timeout ditolak")
            amt = Decimal(str(risk.get("positionAmt", "0")))
            if amt == 0:
                return
            actual_dir = "BUY" if amt > 0 else "SELL"
            side_close = "SELL" if actual_dir == "BUY" else "BUY"
            filters = self.binance.get_symbol_filters(symbol)
            qty = round_step(abs(amt), filters["step_size"])
            if qty <= 0:
                raise ExchangeError(f"Quantity posisi {symbol} nol setelah normalisasi")
            self.binance.place_market_order(symbol, side_close, qty, actual_dir)
            time.sleep(0.5)
            verify = self.binance.get_position_risk(symbol)
            if verify:
                raise ExchangeError(f"Posisi {symbol} masih aktif setelah market close")
        open_orders = self.binance.get_open_orders(symbol)
        if open_orders:
            self.binance.cancel_all_open_orders(symbol)
            verify_orders = self.binance.get_open_orders(symbol)
            if verify_orders:
                raise ExchangeError(f"Masih ada {len(verify_orders)} open order Binance pada {symbol}")


    def _evaluate_position_monitoring(self, symbol: str) -> None:
        pos = self.state.positions.get(symbol)
        if not pos or pos["status"] not in ("PROTECTED", "TRAILING"):
            return
        with self._candle_lock:
            candles = list(self._candle_cache.get(symbol, []))
        if not candles:
            return
        btc_candles = []
        with self._candle_lock:
            btc_candles = list(self._candle_cache.get("BTCUSDT", []))
        decision = self.strategy_engine.monitor_position(pos, candles, btc_candles=btc_candles, market_context=self._market_context)
        logger.info("[MONITOR] %s action=%s profit_r=%.2f weakness=%s", symbol, decision.get("action"), float(decision.get("profit_r", 0.0)), decision.get("weakness_score", 0), extra={"symbol": symbol})
        if decision["action"] == "TRAIL" and decision["new_sl"] is not None:
            self._trail_queue.put(symbol)
            pos["_pending_trail_sl"] = decision["new_sl"]
            pos["_pending_trail_reason"] = decision["reason"]

    def _process_trail_queue(self) -> None:
        """§23 — trail queue, eksekusi 2 detik/coin agar tidak burst request."""
        try:
            symbol = self._trail_queue.get(timeout=1)
        except queue.Empty:
            return
        pos = self.state.positions.get(symbol)
        if not pos or "_pending_trail_sl" not in pos:
            return
        new_sl = pos.get("_pending_trail_sl")
        reasons = pos.get("_pending_trail_reason", [])
        if new_sl is None:
            return

        old_sl = pos["sl"]
        more_protective = (new_sl > old_sl) if pos["direction"] == "BUY" else (new_sl < old_sl)
        if not more_protective:
            return

        if self.state.mode == "REAL":
            ok = self._safe_trail_update_real(symbol, pos, new_sl)
            if not ok:
                pos["_pending_trail_sl"] = new_sl
                pos["_pending_trail_reason"] = reasons
                logger.info("[BINANCE] TRAIL QUEUED — %s new_sl=%s", symbol, new_sl)
                return

        pos.pop("_pending_trail_sl", None)
        pos.pop("_pending_trail_reason", None)
        pos["trail_count"] = pos.get("trail_count", 0) + 1
        pos.setdefault("trail_history", []).append({"timestamp": time.time()*1000, "old_sl": old_sl, "new_sl": new_sl, "reason": list(reasons), "profit_r": pos.get("profit_r")})
        if pos["status"] != "TRAILING":
            # transisi PROTECTED -> TRAILING sekaligus meng-update SL (§55)
            self.state.transition(
                symbol, "TRAILING", f"{symbol}:TRAIL:{pos['trail_count']}:{pos['created_at']}",
                sl=new_sl, trail_count=pos["trail_count"],
            )
        else:
            # sudah berstatus TRAILING sebelumnya — cukup update nilai SL langsung
            pos["sl"] = new_sl
        self.telegram.send(
            f"🔒 TRAILING UPDATE — {symbol}\nSL: {old_sl} -> {new_sl}\nAlasan: {', '.join(reasons)}",
            "TRAIL",
        )
        time.sleep(2)  # §23 — 2 detik / coin

    def _safe_trail_update_real(self, symbol: str, pos: Dict[str, Any], new_sl: float) -> bool:
        """Create new SL first, then cancel old SL to minimize protection gap."""
        try:
            side_close = "SELL" if pos["direction"] == "BUY" else "BUY"
            new_order = self.binance.place_stop_market(
                symbol, side_close, Decimal(str(new_sl)), pos["direction"]
            )
            if not new_order.get("orderId"):
                return False
            old_order_id = pos["binance_order_ids"].get("sl")
            if old_order_id:
                try:
                    time.sleep(1)
                    self.binance.cancel_order(symbol, old_order_id)
                except RateLimitError as e:
                    # New SL is already live. Keep it and retry old-order cleanup later.
                    logger.error("SL baru %s terpasang tetapi cancel SL lama kena rate-limit: %s", symbol, e)
                    self._enter_binance_pause(e)
                    pos["binance_order_ids"]["sl"] = new_order.get("orderId")
                    return True
                except Exception as e:
                    logger.error("SL baru %s terpasang tapi gagal hapus order lama %s: %s — REVIEW MANUAL", symbol, old_order_id, e)
                    self.telegram.send(f"⚠️ WARNING — order SL lama {symbol} gagal dihapus, cek manual!", "WARNING")
            pos["binance_order_ids"]["sl"] = new_order.get("orderId")
            pos["binance_order_ids"]["trail"] = new_order.get("orderId")
            return True
        except RateLimitError as e:
            self._enter_binance_pause(e)
            return False
        except ExchangeError as e:
            logger.error("Gagal update trailing SL %s: %s", symbol, e)
            return False

    def _enter_binance_pause(self, error: Any) -> None:
        """Pause REST until Binance cooldown + 60s safety margin."""
        now = time.time()
        if isinstance(error, RateLimitError):
            if error.banned_until_ms:
                until = error.banned_until_ms / 1000.0
            elif error.retry_after is not None:
                until = now + max(1.0, error.retry_after)
            else:
                until = now + 60.0
            status = error.status_code or 429
            reason = str(error)
        else:
            until = now + 60.0
            status = 429
            reason = str(error)

        until += BINANCE_POST_LIMIT_SAFETY_SECONDS  # post-limit safety delay

        with self.state._lock:
            was_paused = self.state.binance_paused
            self.state.binance_paused = True
            self.state.binance_pause_ts = now
            self.state.binance_pause_until = max(self.state.binance_pause_until or 0.0, until)
            self.state.binance_pause_reason = reason
            remaining = max(0, int(self.state.binance_pause_until - now))

        if not was_paused:
            self.telegram.send(
                f"⏸️ Binance RATE LIMIT/BAN\n"
                f"Entry baru Binance dihentikan. Posisi aktif tetap dipantau via WS.\n"
                f"Cooldown: {remaining} detik\n"
                f"HTTP: {status}\n"
                f"Reason: {reason[:500]}\n"
                f"Weight 1m: {self.binance._used_weight_1m} | "
                f"Order 10s: {self.binance._used_order_10s} | Order 1m: {self.binance._used_order_1m}\n"
                f"Waiting list: protection/entry/trail disimpan",
                "BINANCE_PAUSE",
            )
        else:
            logger.warning("Binance pause diperpanjang %ss: %s", remaining, reason)

    def _check_binance_recovery(self) -> bool:
        """Timer-only recovery. Atomic so READY is emitted once."""
        with self.state._lock:
            if not self.state.binance_paused:
                return True
            now = time.time()
            until = self.state.binance_pause_until or ((self.state.binance_pause_ts or now) + 60)
            if now < until:
                return False
            self.state.binance_paused = False
            self.state.binance_pause_until = None
            self.state.binance_pause_reason = ""
        self.telegram.send("🟢 BINANCE READY — REST recovery window selesai", "BINANCE_READY")
        return True

    # -------------------------------------------------------------------
    # Worker 1 — Scanner + Strategy (§4, §6-§14)
    # -------------------------------------------------------------------
    def _worker_scanner(self) -> None:
        while not self._stop.is_set():
            try:
                self._check_binance_recovery()
                if not self.state.auto or self.state.binance_paused:
                    time.sleep(1)
                    continue
                if self.state.get_active_count() >= self.state.max_positions:
                    time.sleep(2)
                    continue
                self._run_scan_cycle()
            except RateLimitError as e:
                self._enter_binance_pause(e)
            except Exception as e:  # pragma: no cover
                logger.error("Worker1 scanner error: %s: %s", type(e).__name__, e)
                time.sleep(2)

    def _run_scan_cycle(self) -> None:
        cycle_id = int(time.time() * 1000)
        logger.info("[SCAN] CYCLE START id=%s", cycle_id)
        # Scanner market-data path is Bybit-only. Do not ask Binance for exchangeInfo
        # just to build the candidate universe; Binance filters are fetched only
        # when an actual candidate is selected for execution.
        self.state.cleanup_expired_bans()
        universe = build_universe(self.bybit, None, self.state)
        self.state.scanned_coins = universe
        logger.info("[SCAN] UNIVERSE READY | count=%s | max_slots=%s/%s", len(universe), self.state.get_active_count(), self.state.max_positions)

        btc_candles = None
        candidates: List[strategy.Setup] = []
        processed = 0
        valid_strategy = 0
        reject_counts: Dict[str, int] = {}

        # BTCUSDT tetap menjadi konteks korelasi walau sedang active/banned dan tidak boleh ditradingkan.
        if "BTCUSDT" in universe:
            btc_index = universe.index("BTCUSDT")
            if not self.state.is_banned("BTCUSDT") and "BTCUSDT" not in self.state.positions:
                pass
            else:
                try:
                    btc_candles = self.bybit.get_klines("BTCUSDT", "15", 672)
                    with self._candle_lock:
                        self._candle_cache["BTCUSDT"] = btc_candles
                    processed += 1
                except (ExchangeError, RateLimitError) as e:
                    if isinstance(e, RateLimitError):
                        raise
                    logger.warning("Gagal ambil candle BTCUSDT sebagai context: %s", e)

        for idx, symbol in enumerate(universe, 1):
            if self._stop.is_set() or not self.state.auto:
                logger.info("[SCAN] STOPPED MID-CYCLE at=%s/%s", idx - 1, len(universe))
                break
            logger.info("[SCAN %02d/%02d] START", idx, len(universe), extra={"symbol": symbol})
            if self.state.is_banned(symbol) or symbol in self.state.positions:
                continue
            if symbol == "BTCUSDT" and btc_candles is not None:
                candles = btc_candles
            else:
                try:
                    candles = self.bybit.get_klines(symbol, "15", 672)
                except (ExchangeError, RateLimitError) as e:
                    if isinstance(e, RateLimitError):
                        raise
                    logger.warning("Gagal ambil candle %s: %s", symbol, e)
                    time.sleep(1)
                    continue
            with self._candle_lock:
                self._candle_cache[symbol] = candles
            logger.info("[SCAN %02d/%02d] CANDLES READY | n=%s", idx, len(universe), len(candles), extra={"symbol": symbol})

            if symbol == "BTCUSDT":
                btc_candles = candles
            processed += 1

            setup, analysis_diag = self.strategy_engine.analyze_with_diagnostics(
                symbol, candles, btc_candles, market_context=self._market_context, enforce_threshold=False
            )
            if setup:
                valid_strategy += 1
                logger.info("[SCAN %02d/%02d] STRATEGY OK | %s %.1f%%", idx, len(universe), setup.direction, setup.confidence, extra={"symbol": symbol})
                logger.info("[SCAN %02d/%02d] DIAGNOSTICS | structure=%s liquidity=%s entry_dist=%.2fATR rr=%.2f btc=%s freshness=%.2f",
                            idx, len(universe), (analysis_diag.get("structure") or {}).get("bos"),
                            "SWEEP" if (analysis_diag.get("liquidity") or {}).get("sweep") else "NONE",
                            float((analysis_diag.get("entry") or {}).get("distance_atr", 0.0)),
                            float((analysis_diag.get("tp") or {}).get("rr", 0.0)),
                            (analysis_diag.get("btc") or {}).get("aligned"),
                            float((analysis_diag.get("freshness") or {}).get("score", analysis_diag.get("freshness_score", 0.0))),
                            extra={"symbol": symbol})
                threshold = self.strategy_engine.get_active_threshold()
                eligible_now = setup.confidence >= threshold
                self.learn_engine.record_scan_candidate(setup.to_dict(), eligible_now, threshold, "PASS" if eligible_now else "BELOW_ACTIVE_THRESHOLD")
                if eligible_now:
                    candidates.append(setup)
                else:
                    reject_counts["BELOW_ACTIVE_THRESHOLD"] = reject_counts.get("BELOW_ACTIVE_THRESHOLD", 0) + 1
                    logger.info("[SCAN %02d/%02d] BELOW THRESHOLD | %.1f%% < %.1f%%", idx, len(universe), setup.confidence, threshold, extra={"symbol": symbol})
                    # mulai low-confidence ban hanya setelah sistem threshold mencapai 40%.
                    if threshold >= 40.0 and setup.confidence < threshold and setup.confidence >= 0:
                        self.state.ban(symbol, "LOW_CONFIDENCE", 4 * 3600)
                        self.telegram.send(f"🚫 LOW-CONF BAN — {symbol}\nConfidence: {setup.confidence:.1f}% < threshold {threshold:.1f}%\nDurasi: 4 jam", "BANNED")
                # simpan kandidat sebagai shadow untuk evaluasi threshold berikutnya
                if not eligible_now:
                    with self._shadow_lock:
                        self._shadow_candidates[symbol] = {**setup.to_dict(), "shadow_state": "WAIT_ENTRY", "expires_at": time.time() + 24 * 3600}
            else:
                reject_counts["NO_VALID_ENTRY_CANDIDATE"] = reject_counts.get("NO_VALID_ENTRY_CANDIDATE", 0) + 1

            time.sleep(1)  # §5 — jeda 1 detik / coin

        candidates.sort(key=lambda s: s.confidence, reverse=True)
        if self._stop.is_set() or not self.state.auto or self.state.binance_paused:
            # /stop (atau Binance pause) harus memotong cycle tanpa mengubah
            # kandidat yang sudah dianalisis menjadi order baru.
            eligible = []
        else:
            slots_left = self.state.max_positions - self.state.get_active_count()
            eligible = candidates[: max(0, slots_left)]

            for setup in eligible:
                if self._stop.is_set() or not self.state.auto or self.state.binance_paused:
                    break
                self._create_pending(setup)

        avg_conf = sum(s.confidence for s in candidates) / len(candidates) if candidates else 0.0
        buy_n = sum(1 for s in candidates if s.direction == "BUY")
        breadth_buy = (buy_n / len(candidates) * 100) if candidates else 0.0
        if candidates:
            regime = candidates[0].regime
        elif btc_candles:
            regime = strategy.classify_regime(btc_candles, self.strategy_engine.params)
        else:
            regime = "SIDEWAYS"

        summary = {
            "requested": len(universe), "available": len(universe), "processed": processed,
            "valid_strategy": valid_strategy, "candidate": valid_strategy, "eligible": len(eligible),
            "threshold_rejected": reject_counts.get("BELOW_ACTIVE_THRESHOLD", 0),
            "avg_confidence": avg_conf, "rejects": reject_counts, "breadth_buy": breadth_buy,
            "breadth_sell": 100 - breadth_buy, "regime": regime,
            "btc_price": (btc_candles[-1].get("c") if btc_candles else None),
        }
        self._market_context = dict(summary)
        self.learn_engine.record_market_snapshot({
            "timestamp": time.time(), "btc_price": summary.get("btc_price"),
            "btc_regime": regime, "breadth_buy": breadth_buy, "breadth_sell": 100 - breadth_buy,
            "candidate_rate": valid_strategy, "eligible_rate": len(eligible),
        })
        self.learn_engine.record_scan_summary(summary)
        logger.info("[SCAN] CYCLE DONE | processed=%s valid=%s candidate=%s eligible=%s avg_conf=%.1f", processed, valid_strategy, len(candidates), len(eligible), avg_conf)
        self.state.scan_history.append({
            "timestamp": time.time(), "coins": list(universe), "requested": len(universe),
            "processed": processed, "valid_strategy": valid_strategy, "eligible": len(eligible),
            "threshold": self.strategy_engine.get_active_threshold(),
        })
        self.state.scan_history = self.state.scan_history[-20:]

        if eligible:
            lines = [f"✅ {len(eligible)} DECISION BRAIN ELIGIBLE\n"]
            for s in eligible:
                lines.append(f"• {s.pair} {s.direction} — {s.confidence:.0f}%")
            lines.append(f"\n📊 Scan\n{len(universe)} requested | {len(universe)} available")
            lines.append(f"{processed} processed | {valid_strategy} valid strategy")
            lines.append(f"\n🧠 Average confidence: {avg_conf:.1f}%")
            lines.append(f"\n🎯 Candidate: {len(candidates)}\nEligible: {len(eligible)}")
            rejects_str = "\n".join(f"{k}={v}" for k, v in reject_counts.items())
            lines.append(f"\n🚫 Main rejects:\n{rejects_str}")
            lines.append(f"\n📈 Breadth\nBUY {breadth_buy:.1f}%\nSELL {100 - breadth_buy:.1f}%")
            lines.append(f"\nRegime: {regime}")
            self.telegram.send("\n".join(lines), "SIGNAL_PASSED")

    def _make_client_order_id(self, symbol: str, pos_or_setup: Dict[str, Any]) -> str:
        raw = f"{symbol}-{pos_or_setup.get('created_at', time.time())}"
        safe = re.sub(r"[^A-Za-z0-9_-]", "", raw)
        return ("SMC" + safe)[-32:]

    def _create_pending(self, setup: strategy.Setup) -> None:
        logger.info("[ENTRY] PREPARE | direction=%s confidence=%.1f%% mode=%s", setup.direction, setup.confidence, self.state.mode, extra={"symbol": setup.pair})
        # SIMULASI never touches Binance. REAL exposes PENDING only after
        # Binance acknowledges the limit order with an orderId.
        if self.state.get_active_count() >= self.state.max_positions:
            logger.info("[SYSTEM] MAX 20 — %s tidak dibuat", setup.pair)
            return

        if self.state.mode == "REAL":
            try:
                filters = self.binance.get_symbol_filters(setup.pair)
            except RateLimitError as e:
                self._enter_binance_pause(e)
                return
            except ExchangeError as e:
                logger.error("Filter Binance gagal %s: %s", setup.pair, e)
                self.telegram.send(f"⚠️ ERROR — filter Binance {setup.pair}: {e}", "ERROR")
                return
        else:
            filters = {
                "step_size": Decimal("0.000001"), "min_qty": Decimal("0.000001"),
                "tick_size": Decimal("0.000001"), "min_notional": Decimal("0.01"),
            }

        normalized = normalize_order_prices(setup.to_dict(), filters["tick_size"])
        ok, geom_reason = strategy.validate_geometry(
            setup.direction, normalized["entry"], normalized["sl"], normalized["tp"],
            float(filters["tick_size"]), setup.atr
        )
        if not ok:
            logger.info("Setup %s ditolak validasi geometry: %s", setup.pair, geom_reason)
            return
        setup.entry, setup.sl, setup.tp = normalized["entry"], normalized["sl"], normalized["tp"]

        qty, reason = compute_quantity(setup.entry, self.state.margin, self.state.leverage, filters)
        if qty is None:
            logger.info("Setup %s ditolak validasi quantity/margin: %s", setup.pair, reason)
            return

        if self.state.mode == "SIMULASI":
            self.state.add_pending(setup.to_dict(), qty, self.state.margin, status="PENDING")
            with self._shadow_lock:
                self._shadow_candidates.pop(setup.pair, None)
            self.ws.subscribe(setup.pair)
            self.binance_position_ws.subscribe(setup.pair)
            current = self._last_prices.get(setup.pair, setup.entry)
            self.telegram.send(
                f"🎯 PENDING ORDER — {setup.pair}\n\n"
                f"{'🟢' if setup.direction == 'BUY' else '🔴'} {setup.direction}\n"
                f"Harga Saat Ini: {current:.6f}\n\nConfidence: {setup.confidence:.1f}%\n\n"
                f"Entry Zone: {setup.entry:.6f}\nTP: {setup.tp:.6f}\nSL: {setup.sl:.6f}",
                "PENDING",
            )
            return

        self.state.add_pending(setup.to_dict(), qty, self.state.margin, status="BINANCE_WAITING")
        pos = self.state.positions[setup.pair]
        pos["binance_entry_client_order_id"] = self._make_client_order_id(setup.pair, pos)
        with self._shadow_lock:
            self._shadow_candidates.pop(setup.pair, None)

        try:
            leverage = int(self.state.leverage)
            self.binance.set_leverage(setup.pair, leverage)
            pos["binance_leverage_confirmed"] = True
            side = "BUY" if setup.direction == "BUY" else "SELL"
            entry_price = round_price(
                setup.entry, filters["tick_size"],
                ROUND_DOWN if setup.direction == "BUY" else ROUND_UP,
            )
            order = self.binance.place_limit_order(
                setup.pair, side, qty, entry_price, setup.direction,
                client_order_id=pos["binance_entry_client_order_id"],
            )
            order_id = order.get("orderId")
            if not order_id:
                raise ExchangeError(f"Binance limit {setup.pair} diterima tanpa orderId")
            pos["binance_order_ids"]["entry"] = order_id
            pos["binance_entry_confirmed_at"] = time.time()
            event_id = f"{setup.pair}:BINANCE_ENTRY_CONFIRMED:{pos['created_at']}"
            if not self.state.transition(setup.pair, "PENDING", event_id, binance_entry_confirmed=True):
                raise ExchangeError(f"State PENDING gagal diterapkan setelah order Binance {setup.pair}")
            self.ws.subscribe(setup.pair)
            self.binance_position_ws.subscribe(setup.pair)
            logger.info("[ENTRY] BINANCE LIMIT CONFIRMED | orderId=%s | state=PENDING", order_id, extra={"symbol": setup.pair})
        except RateLimitError as e:
            logger.warning("[ENTRY] BINANCE RATE-LIMIT | moved to waiting queue", extra={"symbol": setup.pair})
            self._queue_binance_pending(setup.pair)
            self._enter_binance_pause(e)
            return
        except ExchangeError as e:
            logger.error("Gagal pasang limit order %s: %s", setup.pair, e)
            self.telegram.send(f"⚠️ ERROR — gagal pasang order {setup.pair}: {e}", "ERROR")
            self.state.discard_pending(setup.pair)
            self.ws.unsubscribe(setup.pair)
            return

        current = self._last_prices.get(setup.pair, setup.entry)
        self.telegram.send(
            f"🎯 PENDING ORDER — {setup.pair}\n\n"
            f"{'🟢' if setup.direction == 'BUY' else '🔴'} {setup.direction}\n"
            f"Harga Saat Ini: {current:.6f}\n\nConfidence: {setup.confidence:.1f}%\n\n"
            f"Entry Zone: {setup.entry:.6f}\nTP: {setup.tp:.6f}\nSL: {setup.sl:.6f}",
            "PENDING",
        )

    def _notify_learn_checkpoint_fallback(self, reason: str, ok: bool) -> None:
        """Telegram fallback untuk LearnEngine lama tanpa notification sink.

        Tidak membaca/mengambil data eksternal; hanya memberi tahu hasil
        checkpoint yang baru saja diminta oleh main.py. LearnEngine vNext
        menangani notifikasi sendiri melalui sink, sehingga fungsi ini hanya
        aktif bila sink API belum tersedia.
        """
        if self._learn_notification_sink_attached:
            return
        try:
            status = "✅ PASS" if ok else "❌ FAIL"
            self.telegram.send(
                f"🧠 LEARN CHECKPOINT\nStatus: {status}\nReason: {reason}",
                "LEARN_CHECKPOINT",
            )
        except Exception as exc:
            logger.warning("[LEARN] checkpoint Telegram fallback gagal: %s", exc)

    # -------------------------------------------------------------------
    # Worker 3 — Learn (§4, §39-§51)
    # -------------------------------------------------------------------
    def _worker_learn(self) -> None:
        last_audit = 0.0
        last_autosave = 0.0
        while not self._stop.is_set():
            try:
                self._check_binance_recovery()
                self._process_binance_waiting_lists()
                self._process_trail_queue()
                now = time.time()
                if now - last_audit > 300:  # audit tiap 5 menit
                    logger.info("[LEARN] AUDIT START")
                    report = self.learn_engine.audit(self.strategy_engine)
                    self.learn_engine.set_strategy_state(self.strategy_engine.export_state())
                    self.state.strategy_state = self.strategy_engine.export_state()
                    self._notify_audit_report(report)
                    logger.info("[LEARN] AUDIT DONE | action=%s", report.get("action"))
                    last_audit = now
                if now - last_autosave > 120:  # autosave tiap 2 menit, tidak boleh ganggu trading (§40)
                    logger.info("[LEARN] AUTOSAVE START")
                    learn_ok = True
                    try:
                        result = self.learn_engine.autosave(reason="periodic_worker")
                        if isinstance(result, bool):
                            learn_ok = result
                    except TypeError:
                        # Compatibility with the previous autosave() signature.
                        result = self.learn_engine.autosave()
                        if isinstance(result, bool):
                            learn_ok = result
                    except Exception as exc:
                        learn_ok = False
                        logger.warning("[LEARN] autosave gagal: %s", exc)
                    self.state.save_checkpoint()
                    self._notify_learn_checkpoint_fallback("periodic_worker", learn_ok)
                    logger.info("[LEARN] AUTOSAVE DONE")
                    last_autosave = now
                self._check_autostop()
                self.telegram.flush(max_messages=5)
            except Exception as e:  # pragma: no cover
                logger.error("Worker3 learn error: %s", e)
            time.sleep(0.5)

    def _notify_audit_report(self, report: Dict[str, Any]) -> None:
        """§50/§51 — pastikan diagnosis frequency SELALU sampai ke user, tidak
        cuma saat threshold benar-benar berubah (APPLIED)."""
        action = report.get("action")
        if action == "APPLIED":
            evidence = report.get("evidence", {})
            is_exploratory = evidence.get("type", "").startswith("EXPLORATORY") or evidence.get("type", "").startswith("LOWER_THRESHOLD")
            label = "🧭 STRATEGY EXPLORATORY CHANGE" if is_exploratory else "🧠 STRATEGY UPDATED"
            self.telegram.send(
                f"{label} — v{report['strategy_version']}\n"
                f"Threshold: {report['old_threshold']:.1f}% -> {report['new_threshold']:.1f}%\n"
                f"Evidence: {evidence.get('validation_note') or evidence.get('note')}",
                "INFO",
            )
        elif action == "REJECTED":
            logger.info("Audit: perubahan diusulkan tapi ditolak validasi — %s", report.get("reason"))

        freq = report.get("frequency") or {}
        status = freq.get("status")
        now = time.time()
        status_changed = status != self._last_freq_status
        stale_repeat = now - self._last_freq_alert_ts > 1800  # jangan spam, ulang tiap 30 menit
        if status in ("POSSIBLY_BROKEN", "HEALTHY_LOW_FREQUENCY_OR_STRICT_THRESHOLD") and (status_changed or stale_repeat):
            self.telegram.send(
                f"⚠️ FREQUENCY WARNING — {status}\n{freq.get('note', '')}\n"
                f"Rata-rata candidate: {freq.get('avg_candidate', 0):.2f} | eligible: {freq.get('avg_eligible', 0):.2f}",
                "WARNING",
            )
            self._last_freq_alert_ts = now
        self._last_freq_status = status

    def _check_autostop(self) -> None:
        """Check the last known Binance balance; never poll every loop."""
        if self.state.mode != "REAL" or self.state.autostop_pct is None:
            return
        balance = self.state.current_balance
        high = self.state.highest_balance
        if balance is None or high is None or high <= 0:
            return
        drawdown_pct = (high - balance) / high * 100
        if drawdown_pct >= self.state.autostop_pct and self.state.auto:
            self.state.auto = False
            self.telegram.send(
                f"🛑 AUTOSTOP TERPICU\nDrawdown: {drawdown_pct:.2f}% >= batas {self.state.autostop_pct}%\nAUTO = OFF",
                "AUTOSTOP",
            )

    # -------------------------------------------------------------------
    # Worker 5 — Ban Timer (§4, §26-§28) — tetap jalan walau scanner OFF
    # -------------------------------------------------------------------
    def _worker_ban_timer(self) -> None:
        while not self._stop.is_set():
            try:
                expired = self.state.cleanup_expired_bans()
                for symbol in expired:
                    self.telegram.send(f"✅ UNBANNED — {symbol} (masa ban berakhir)", "UNBANNED")
            except Exception as e:  # pragma: no cover
                logger.error("Worker5 ban timer error: %s", e)
            time.sleep(30)

    # -------------------------------------------------------------------
    # Worker 4 — Telegram Command Handler (§4, §54)
    # -------------------------------------------------------------------
    def _worker_command_handler(self) -> None:
        if not self.cfg.telegram_bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN kosong — command handler nonaktif")
            return
        if not self.cfg.allowed_user_id:
            logger.warning(
                "ALLOWED_USER_ID kosong — SEMUA orang yang tahu bot token bisa mengontrol bot ini! "
                "Sangat disarankan set ALLOWED_USER_ID di .env."
            )
        while not self._stop.is_set():
            updates = self.telegram.get_updates(self._last_update_id)
            for u in updates:
                self._last_update_id = u["update_id"] + 1
                msg = u.get("message", {})
                text = msg.get("text", "")
                sender_id = str(msg.get("from", {}).get("id", ""))
                if not text.startswith("/"):
                    continue
                if not self._is_authorized(sender_id):
                    logger.warning("Command '%s' dari user tidak diotorisasi (id=%s) — diabaikan", text, sender_id)
                    continue
                try:
                    self._dispatch_command(text)
                    self.telegram.flush(max_messages=10)
                except Exception as e:
                    logger.error("Command error '%s': %s", text, e)
                    self.telegram.send(f"⚠️ ERROR memproses command: {e}", "ERROR")

    def _is_authorized(self, sender_id: str) -> bool:
        """§54 — hanya ALLOWED_USER_ID yang boleh mengontrol bot lewat Telegram.
        Jika ALLOWED_USER_ID tidak diset, bot fail-open (mengizinkan semua) demi
        kompatibilitas mundur, tapi ini sudah diberi warning keras saat startup."""
        if not self.cfg.allowed_user_id:
            return False
        return sender_id != "" and sender_id == str(self.cfg.allowed_user_id)

    def _dispatch_command(self, text: str) -> None:
        parts = text.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        handlers = {
            "/auto": self._cmd_auto, "/stop": self._cmd_stop, "/mode": self._cmd_mode,
            "/margin": self._cmd_margin, "/leverage": self._cmd_leverage, "/max": self._cmd_max,
            "/resetbalance": self._cmd_resetbalance, "/trade": self._cmd_trade,
            "/order": self._cmd_order, "/stats": self._cmd_stats, "/koin": self._cmd_koin,
            "/ip": self._cmd_ip, "/banned": self._cmd_banned, "/unban": self._cmd_unban,
            "/timeout": self._cmd_timeout, "/autostop": self._cmd_autostop, "/open": self._cmd_open,
            "/healthz": self._cmd_healthz, "/help": self._cmd_help,
        }
        handler = handlers.get(cmd)
        if not handler:
            self.telegram.send(f"⚠️ Command tidak dikenal: {cmd}\nGunakan /help untuk daftar command yang valid.", "WARNING")
            return
        logger.info("[COMMAND] EXECUTE %s %s", cmd, " ".join(args) if args else "")
        handler(args)

    def _cmd_healthz(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /healthz", "INFO")
            return
        now = time.time()
        with self.state._lock:
            paused = bool(self.state.binance_paused)
            pause_ts = self.state.binance_pause_ts
            pause_until = self.state.binance_pause_until
            reason = self.state.binance_pause_reason or "-"
            auto = self.state.auto
            mode = self.state.mode
            slots = self.state.get_active_count()
            max_slots = self.state.max_positions
        lines = [
            "🩺 HEALTHZ",
            f"Main: 🟢 ONLINE",
            f"Mode: {mode}",
            f"AUTO: {'ON' if auto else 'OFF'}",
            f"Posisi/reserve: {slots}/{max_slots}",
        ]
        if paused:
            remaining = max(0, int((pause_until or now) - now))
            ban_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pause_ts)) if pause_ts else "-"
            ready_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pause_until)) if pause_until else "-"
            lines += [
                "Binance REST: 🔴 BAN / RATE-LIMIT",
                f"Catat mulai: {ban_at}",
                f"Ready setelah: {ready_at}",
                f"Sisa cooldown: {remaining} detik",
                f"Reason: {reason[:300]}",
                "User Data WS: 🟢 tetap aktif" if mode == "REAL" else "User Data WS: ⚪ tidak aktif (SIMULASI)",
                f"REST governor: weight1m={self.binance._used_weight_1m} | order10s={self.binance._used_order_10s} | order1m={self.binance._used_order_1m}",
            ]
        else:
            lines += [
                "Binance REST: 🟢 READY",
                "User Data WS: 🟢 aktif" if mode == "REAL" else "User Data WS: ⚪ tidak aktif (SIMULASI)",
                f"REST governor: weight1m={self.binance._used_weight_1m} | order10s={self.binance._used_order_10s} | order1m={self.binance._used_order_1m}",
            ]
        self.telegram.send("\n".join(lines), "INFO")

    def _cmd_help(self, args: List[str]) -> None:
        self.telegram.send(
            "🤖 COMMAND BOT\n\n"
            "/auto - Aktifkan AUTO scanning\n"
            "/stop - Matikan AUTO scanning\n"
            "/mode on|off - REAL / SIMULASI\n"
            "/margin <USDT> - Atur margin\n"
            "/leverage <integer> - Atur leverage (1-125x)\n"
            "/resetbalance - Reset balance anchor\n"
            "/trade - Posisi aktif/pending\n"
            "/order - Order aktif\n"
            "/open - Buka kembali memory learning dari checkpoint/backup\n"
            "/stats - Statistik bot\n"
            "/koin - Universe coin\n"
            "/banned - Daftar coin banned\n"
            "/unban <COIN> - Hapus ban coin\n"
            "/timeout All|COIN - Bersihkan satu/semua posisi dan order\n"
            "/autostop - Pengaturan auto stop\n"
            "/ip - IP server\n"
            "/healthz - Status Binance, ban/cooldown, WS, dan posisi\n"
            "/help - Bantuan command",
            "INFO",
        )

    def _cmd_auto(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /auto", "INFO")
            return
        if self.state.mode == "REAL" and self.state.binance_paused:
            self.telegram.send("⚠️ AUTO belum bisa dinyalakan: Binance sedang rate-limited.", "WARNING")
            return
        self.state.auto = True
        logger.info("[AUTO] ENABLE | mode=%s | binance_paused=%s | weight1m=%s", self.state.mode, self.state.binance_paused, self.binance._used_weight_1m)
        if self.state.mode == "REAL":
            if self.state.current_balance is None or self.state.real_balance_snapshot is None:
                self.telegram.send(
                    "⚠️ AUTO belum bisa dinyalakan: belum ada snapshot saldo REAL lokal. "
                    "Gunakan /mode on atau /resetbalance untuk mengambil 1 snapshot Binance.",
                    "WARNING",
                )
                self.state.auto = False
                return
            self.state.highest_balance = self.state.current_balance if self.state.highest_balance is None else max(self.state.highest_balance, self.state.current_balance)
        self.telegram.send("▶️ AUTO = ON — scanning dimulai", "INFO")

    def _cmd_stop(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /stop", "INFO")
            return
        self.state.auto = False
        self.telegram.send("⏹️ AUTO = OFF — scanning dihentikan (WebSocket & trailing tetap aktif)", "INFO")

    def _cmd_mode(self, args: List[str]) -> None:
        if len(args) != 1 or args[0].lower() not in ("on", "off"):
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/mode on   → REAL TRADE\n/mode off  → SIMULASI", "INFO")
            return
        target = args[0].lower()
        active = self.state.get_active_count()
        if active > 0:
            self.telegram.send("⚠️ MODE tidak dapat diubah saat masih ada posisi/order aktif.\n\nGunakan /timeout all terlebih dahulu agar state dan Binance benar-benar bersih.", "WARNING")
            return
        if target == "on":
            if self.state.binance_paused:
                self.telegram.send("⚠️ MODE REAL TIDAK DAPAT DIAKTIFKAN\n\nBinance sedang rate-limited. Tunggu BINANCE READY.", "WARNING")
                return
            missing = self.cfg.validate_for_real_mode()
            if missing:
                self.telegram.send(f"⚠️ Tidak bisa mengaktifkan REAL: env var kosong: {', '.join(missing)}", "ERROR")
                return
            try:
                # Exactly ONE Binance balance snapshot when entering REAL mode.
                bal = self.binance.get_balance_usdt()
                self._set_real_balance_snapshot(bal, "BINANCE_REST_MODE_ON")
                self.state.mode = "REAL"
                self.binance_ws.start()
                self.learn_engine.set_strategy_state(self.strategy_engine.export_state())
                self.telegram.send(
                    f"🔴 MODE REAL TRADE AKTIF\n\nSaldo snapshot Binance: ${bal:.4f}\n"
                    "PnL berikutnya dihitung lokal dari trade + koreksi ACCOUNT_UPDATE WS.",
                    "INFO",
                )
            except RateLimitError as e:
                self._enter_binance_pause(e)
                self.state.mode = "SIMULASI"
                self.telegram.send("⚠️ MODE REAL gagal diaktifkan karena Binance rate-limited. Tetap SIMULASI.", "ERROR")
            except Exception as e:
                self.state.mode = "SIMULASI"
                self.telegram.send(f"⚠️ MODE REAL gagal diaktifkan: {e}", "ERROR")
        else:
            if self.state.mode == "REAL":
                self.telegram.send("🧪 MODE SIMULASI AKTIF", "INFO")
            else:
                self.telegram.send("🧪 MODE SIMULASI SUDAH AKTIF", "INFO")
            self.state.mode = "SIMULASI"

    def _cmd_margin(self, args: List[str]) -> None:
        if len(args) != 1:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/margin <USDT>\n\nContoh:\n/margin 1", "INFO")
            return
        try:
            value = float(args[0])
            if value <= 0 or not math.isfinite(value):
                raise ValueError
        except ValueError:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/margin <USDT>\n\nContoh:\n/margin 1", "INFO")
            return
        self.state.margin = value  # §60 — perubahan margin memicu recalculation quantity utk pending berikutnya
        self.telegram.send(f"✅ MARGIN SUCCESS — margin diatur ke ${value}", "MARGIN_SUCCESS")

    def _cmd_leverage(self, args: List[str]) -> None:
        if len(args) != 1:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/leverage <angka>\n\nContoh:\n/leverage 5", "INFO")
            return
        try:
            value_float = float(args[0])
            if value_float <= 0 or not math.isfinite(value_float) or not value_float.is_integer():
                raise ValueError
            value = int(value_float)
            if value > 125:
                raise ValueError
        except ValueError:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/leverage <integer>\n\nContoh:\n/leverage 5", "INFO")
            return
        self.state.leverage = value
        self.telegram.send(f"✅ LEVERAGE SUCCESS — leverage diatur ke {value}x", "LEVERAGE_SUCCESS")

    def _cmd_max(self, args: List[str]) -> None:
        """Atur hard cap jumlah posisi aktif/reserve. Rentang 1..20; default 5."""
        if len(args) != 1:
            self.telegram.send(
                "⚠️ Format salah.\n\nGunakan:\n/max <angka>\n\nContoh:\n/max 5\n\nRentang yang diizinkan: 1–20\nDefault: 5",
                "INFO",
            )
            return
        try:
            value = int(args[0])
            if value < 1 or value > 20 or str(value) != args[0].lstrip("+"):
                raise ValueError
        except ValueError:
            self.telegram.send(
                "⚠️ MAX tidak valid.\n\nGunakan angka bulat 1–20.\nContoh: /max 5",
                "INFO",
            )
            return
        old = self.state.max_positions
        self.state.max_positions = value
        active = self.state.get_active_count()
        self.state.save_checkpoint()
        if active > value:
            self.telegram.send(
                f"⚠️ MAX POSISI DIUBAH\n\nLimit baru: {value}\nPosisi/reserve aktif saat ini: {active}\n\nBot tidak akan menambah posisi baru sampai jumlah aktif turun ke ≤ {value}.",
                "WARNING",
            )
        else:
            self.telegram.send(
                f"✅ MAX POSISI SUCCESS\n\nSebelumnya: {old}\nSekarang: {value}\nAktif/reserve: {active}/{value}",
                "INFO",
            )

    def _cmd_resetbalance(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /resetbalance", "INFO")
            return
        if self.state.binance_paused:
            self.telegram.send("⚠️ Tidak bisa reset balance — Binance sedang rate-limited.", "INFO")
            return
        if self.state.mode == "SIMULASI":
            self.state.sim_balance = 10.0
            self.state.sim_balance_anchor = 10.0
            self.telegram.send("✅ Balance simulasi direset ke $10.0000", "INFO")
        else:
            try:
                # Exactly ONE fresh Binance snapshot on explicit /resetbalance.
                bal = self.binance.get_balance_usdt()
                old_anchor = self.state.highest_balance
                self._set_real_balance_snapshot(bal, "BINANCE_REST_RESETBALANCE")
                prefix = f"Sebelumnya: ${old_anchor:.4f}\n" if old_anchor is not None else ""
                self.telegram.send(
                    f"✅ Anchor balance REAL direset\n\n{prefix}Snapshot Binance: ${bal:.4f}\n"
                    "PnL berikutnya kembali dihitung lokal.",
                    "INFO",
                )
            except RateLimitError as e:
                self._enter_binance_pause(e)
                self.telegram.send("⚠️ /resetbalance gagal karena Binance rate-limited. Anchor lama dipertahankan.", "ERROR")
            except Exception as e:
                logger.error("Gagal ambil balance Binance saat /resetbalance: %s", e)
                self.telegram.send(f"⚠️ Gagal ambil balance Binance: {e}", "ERROR")

    def _cmd_trade(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /trade", "INFO")
            return
        positions = self.state.snapshot_positions()
        active = [p for p in positions if p["status"] not in TERMINAL_STATES]
        lines = [f"📡 Posisi / Reserve ({len(active)}/{self.state.max_positions})\n"]
        for p in active:
            icon = "🟢" if p["direction"] == "BUY" else "🔴"
            if p["status"] == "BINANCE_WAITING":
                lines.append(
                    f"⏳ {p['pair']} — MENUNGGU BINANCE\n{icon} {p['direction']}\n"
                    f"Entry zone: {p['entry']:.6f}\nTP: {p['tp']:.6f}\nSL: {p['sl']:.6f}\n"
                    f"Confidence: {p['confidence']:.0f}%\n"
                    "⚠️ Belum PENDING sampai Binance mengembalikan orderId.\n"
                )
            elif p["status"] == "PENDING":
                fallback = self._last_prices.get(p['pair'], p['entry'])
                current, source = self._trade_display_price(p['pair'], fallback)
                jarak = abs(current - p['entry']) / p['entry'] * 100 if p['entry'] else 0.0
                lines.append(
                    f"⏳ {p['pair']} — PENDING\n{icon} {p['direction']}\n"
                    f"Entry zone: {p['entry']:.6f}\nHarga kini: {current:.6f}\nSumber harga: {source}\nJarak: {jarak:.2f}%\n"
                    f"TP: {p['tp']:.6f}\nSL: {p['sl']:.6f}\nConfidence: {p['confidence']:.0f}%\n"
                )
            else:
                fallback = self._last_prices.get(p['pair'], p['entry'])
                current, source = self._trade_display_price(p['pair'], fallback)
                if p['direction'] == 'BUY':
                    pnl = ((current - p['entry']) / p['entry']) * 100
                else:
                    pnl = ((p['entry'] - current) / p['entry']) * 100
                fill_time = p.get("fill_time")
                jam = time.strftime("%H:%M", time.localtime(fill_time / 1000)) if fill_time else "-"
                lines.append(
                    f"{icon} {p['pair']} — {p['status']}\n"
                    f"Entry: {p['entry']:.6f}\nHarga: {current:.6f}\nSumber harga: {source}\n"
                    f"TP: {p['tp']:.6f}\nSL: {p['sl']:.6f}\n"
                    f"Confidence: {p['confidence']:.0f}%\nPnL: {pnl:+.2f}%\n🕐 Entry: {jam}\n"
                )
        self.telegram.send("\n".join(lines) if active else "📡 Tidak ada posisi aktif/pending.", "INFO")

    def _cmd_order(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /order", "INFO")
            return
        positions = [p for p in self.state.snapshot_positions() if p["status"] not in TERMINAL_STATES]
        if not positions:
            self.telegram.send("📋 ORDER\n\nTidak ada order aktif.", "INFO")
            return
        lines = [f"📋 ORDER — {len(positions)}/{self.state.max_positions}\n"]
        for i, p in enumerate(positions, 1):
            icon = "🟢" if p["direction"] == "BUY" else "🔴"
            sl_label = "Trail SL" if p.get("trail_count", 0) > 0 else "SL"
            fallback = self._last_prices.get(p["pair"], p["entry"])
            current, _source = self._trade_display_price(p["pair"], fallback)
            status = p["status"]
            if status == "BINANCE_WAITING":
                status = "WAITING BINANCE CONFIRM"
            lines.append(
                f"{i}. {p['pair']} — {status}\n"
                f"{icon} {p['direction']} | Confidence: {p['confidence']:.0f}%\n"
                f"Entry: {p['entry']:.6f}\n"
                f"TP: {p['tp']:.6f}\n"
                f"{sl_label}: {p['sl']:.6f}\n"
            )
        self.telegram.send("\n".join(lines), "INFO")

    def _cmd_stats(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /stats", "INFO")
            return
        stats = self.learn_engine.overall_stats()
        counts = stats.get("outcome_counts", {})
        economic_n = stats.get("n", 0)
        total_recorded = sum(counts.values())
        wins = sum(1 for t in stats.get("last_trades", []) if float(t.get("pnl_r", 0)) > 0)
        anchor = self.state.sim_balance_anchor if self.state.mode == "SIMULASI" else self.state.highest_balance
        balance = self.state.sim_balance if self.state.mode == "SIMULASI" else self.state.current_balance
        dd_line = ""
        if anchor is not None and anchor > 0 and balance is not None:
            dd = (balance - anchor) / anchor * 100
            dd_line = f"\nModal anchor: ${anchor:.4f} → Saldo: ${balance:.4f} ({dd:+.2f}%)"
        trail_n = counts.get("TRAIL", 0)
        tp_n = counts.get("TP", 0)
        sl_n = counts.get("INITIAL_SL", 0)
        trail_pct = (trail_n / economic_n * 100) if economic_n else 0.0
        tp_pct = (tp_n / economic_n * 100) if economic_n else 0.0
        sl_pct = (sl_n / economic_n * 100) if economic_n else 0.0
        lines = [
            f"📊 Statistik — {economic_n} trade | TP {tp_n} | Initial SL {sl_n} | Trail {trail_n}",
            f"Mode: {'🧪 SIMULASI' if self.state.mode == 'SIMULASI' else '🔴 REAL TRADE'}",
            f"Hasil ekonomi: WR {stats['win_rate']:.1f}% | Expectancy {stats['expectancy']:.3f}R | PF {stats['profit_factor']}",
            dd_line,
            (f"Balance model: snapshot ${self.state.real_balance_snapshot:.4f} | local PnL ${self.state.real_localized_pnl:+.4f} | adjustment ${self.state.real_balance_adjustment:+.4f}"
             if self.state.mode == "REAL" and self.state.real_balance_snapshot is not None else ""),
            f"Confidence rata-rata closed: {stats.get('confidence_avg_closed', 0.0):.1f}%",
            f"🎯 TP: {tp_pct:.1f}% | 🔒 Trail: {trail_pct:.1f}% | 🛑 SL: {sl_pct:.1f}%",
            f"⏱️ Timeout: {counts.get('TIMEOUT', 0)} (tidak dihitung WIN/LOSS)",
            "\n5 terakhir:",
        ]
        for t in stats.get("last_trades", []):
            outcome = t.get("outcome", "?")
            emoji = "🟢" if float(t.get("pnl_r", 0)) > 0 else ("🔴" if outcome != "TIMEOUT" else "⏱️")
            lines.append(f"{emoji} {outcome} {float(t.get('pnl_pct', 0.0)):+.2f}% → C{float(t.get('confidence', 0.0)):.0f}%")
        lines.append(f"\n🚫 Banned: {len(self.state.active_bans())} | 🧠 Threshold: {self.strategy_engine.get_active_threshold():.1f}%")
        self.telegram.send("\n".join(x for x in lines if x != ""), "INFO")

    def _cmd_koin(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /koin", "INFO")
            return
        if not self.state.scanned_coins:
            self.telegram.send("🪙 Belum ada coin yang discan.", "INFO")
            return
        lines = ["🪙 COIN TERAKHIR DISCANN\n"]
        for i, s in enumerate(self.state.scanned_coins, 1):
            lines.append(f"{i}. {s}")
        if self.state.scan_history:
            last = self.state.scan_history[-1]
            lines.append(f"\n📊 Scan terakhir: {last.get('processed', 0)}/{last.get('requested', 0)} diproses | {last.get('valid_strategy', 0)} candidate")
        self.telegram.send("\n".join(lines), "INFO")

    def _cmd_ip(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /ip", "INFO")
            return
        ip = "(cek /ip bila diperlukan)"
        self.telegram.send(f"🌐 SERVER INFO\n\nIP: {ip}\nStatus: ONLINE", "INFO")

    def _cmd_banned(self, args: List[str]) -> None:
        if args:
            if len(args) != 1 or not args[0].upper().endswith("USDT"):
                self.telegram.send("⚠️ Format salah. Gunakan /banned BTCUSDT untuk ban permanen, atau /banned untuk melihat daftar.", "INFO")
                return
            symbol = args[0].upper()
            if symbol in self.state.positions:
                self.telegram.send(f"⚠️ {symbol} sedang aktif/pending — tidak diban sekarang. Bersihkan posisi dulu.", "WARNING")
                return
            self.state.ban(symbol, "MANUAL_PERMANENT", None)
            self.telegram.send(f"🚫 BANNED PERMANEN — {symbol}", "BANNED")
            return
        bans = self.state.active_bans()
        if not bans:
            self.telegram.send("🚫 BANNED COINS\n\nTidak ada coin yang diban.", "INFO")
            return
        lines = ["🚫 BANNED COINS\n"]
        now = time.time()
        for symbol, info in bans.items():
            if info.get("permanent") or info.get("expiry") is None:
                remaining = "PERMANEN"
            else:
                remaining_s = max(0.0, float(info["expiry"]) - now)
                h, m = int(remaining_s // 3600), int((remaining_s % 3600) // 60)
                remaining = f"{h:02d}h {m:02d}m"
            lines.append(f"{symbol}\nReason: {info['reason']}\nRemaining: {remaining}\n")
        self.telegram.send("\n".join(lines), "INFO")

    def _cmd_unban(self, args: List[str]) -> None:
        if not args:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/unban BTCUSDT\natau\n/unban All", "INFO")
            return
        self.state.unban(args[0])
        self.telegram.send(f"✅ UNBANNED — {args[0]}", "UNBANNED")

    def _cmd_timeout(self, args: List[str]) -> None:
        if len(args) != 1:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/timeout All\natau\n/timeout BTCUSDT", "INFO")
            return
        target = args[0].upper()
        if self.state.mode == "REAL" and self.state.binance_paused:
            self.telegram.send("⚠️ /timeout REAL ditahan karena Binance sedang rate-limited. State TIDAK dihapus agar tidak meninggalkan posisi tak terlacak.", "WARNING")
            return
        try:
            # Acknowledge immediately so the user knows the cleanup job has started.
            self.telegram.send(
                f"⏳ TIMEOUT {'ALL' if target == 'ALL' else target} — akan mencoba timeout dan membersihkan state/order.",
                "INFO",
            )
            self.telegram.flush(max_messages=1)
            logger.info("[TIMEOUT] START | target=%s | mode=%s", target, self.state.mode)
            if target == "ALL" and self.state.mode == "REAL":
                logger.info("[TIMEOUT] REAL ALL | cleanup Binance START")
                self._cleanup_real_account()
                for symbol in list(self.state.positions.keys()):
                    self.state.transition(symbol, "CANCELLED", f"{symbol}:MANUAL_TIMEOUT:{time.time_ns()}", close_reason="MANUAL_TIMEOUT_ALL")
                    self.state.remove_terminal(symbol)
                    self.ws.unsubscribe(symbol)
                    self.binance_position_ws.unsubscribe(symbol)
                    with self._shadow_lock:
                        self._shadow_candidates.pop(symbol, None)
                self.state.auto = False
                logger.info("[TIMEOUT] REAL ALL | cleanup Binance DONE | local state cleared")
                self.telegram.send("✅ TIMEOUT ALL SELESAI\nBinance: semua posisi dan open order sudah diverifikasi bersih.\nAUTO: OFF", "INFO")
                return

            targets = list(self.state.positions.keys()) if target == "ALL" else [target]
            done, missing = [], []
            for idx, symbol in enumerate(targets, 1):
                logger.info("[TIMEOUT %02d/%02d] START", idx, len(targets), extra={"symbol": symbol})
                pos = self.state.positions.get(symbol)
                if not pos:
                    missing.append(symbol)
                    continue
                if self.state.mode == "REAL" and pos.get("status") != "BINANCE_WAITING":
                    self._cleanup_real_symbol(symbol, pos, close_position=(pos.get("status") != "PENDING"))
                if self.state.transition(symbol, "CANCELLED", f"{symbol}:MANUAL_TIMEOUT:{time.time_ns()}", close_reason="MANUAL_TIMEOUT"):
                    self.state.remove_terminal(symbol)
                    self.ws.unsubscribe(symbol)
                    self.binance_position_ws.unsubscribe(symbol)
                    with self._shadow_lock:
                        self._shadow_candidates.pop(symbol, None)
                    done.append(symbol)
                    logger.info("[TIMEOUT %02d/%02d] DONE", idx, len(targets), extra={"symbol": symbol})
            if missing and not done:
                self.telegram.send(f"⚠️ Tidak ditemukan order/posisi lokal: {', '.join(missing)}", "WARNING")
            else:
                self.telegram.send(f"✅ Timeout selesai\nBerhasil dibersihkan: {', '.join(done) or '-'}", "INFO")
        except RateLimitError as e:
            self._enter_binance_pause(e)
            self.telegram.send("⚠️ Cleanup timeout berhenti karena Binance rate-limit. Tidak ada state lokal yang dihapus untuk posisi yang belum terverifikasi.", "ERROR")
        except Exception as e:
            logger.error("Manual timeout gagal: %s", e)
            self.telegram.send(f"⚠️ ERROR TIMEOUT — {e}\nState dipertahankan sampai Binance bisa diverifikasi bersih.", "ERROR")

    def _cmd_autostop(self, args: List[str]) -> None:
        if len(args) != 1:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/autostop <percentage>\n\nContoh:\n/autostop 10", "INFO")
            return
        try:
            pct = float(args[0])
            if pct <= 0 or pct >= 100:
                raise ValueError
        except ValueError:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/autostop <percentage>\n\nContoh: /autostop 10", "INFO")
            return
        if self.state.mode != "REAL":
            self.telegram.send("⚠️ /autostop hanya berlaku pada MODE REAL. Aktifkan /mode on terlebih dahulu.", "WARNING")
            return
        self.state.autostop_pct = pct
        self.telegram.send(f"✅ Autostop diatur ke {pct}% (REAL TRADE)", "INFO")

    def _cmd_open(self, args: List[str]) -> None:
        if args:
            self.telegram.send("⚠️ Format salah. Gunakan: /open", "INFO")
            return
        label = self.learn_engine.load()
        if self.learn_engine.strategy_state:
            self.strategy_engine.load_state(self.learn_engine.strategy_state)
            self.state.strategy_state = self.strategy_engine.export_state()
        self.telegram.send(f"📂 Learning memory dibuka: {label}\n🧠 Strategy: v{self.strategy_engine.version} | Threshold: {self.strategy_engine.get_active_threshold():.1f}%", "INFO")


# =============================================================================
# 9. SELF TEST (offline, tanpa network) — pelengkap §63 sebisa mungkin tanpa
#    koneksi exchange nyata. Jalankan: python main.py --selftest
# =============================================================================

def run_selftest() -> bool:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}")
        if not cond:
            ok = False

    # geometry test (§58)
    valid, _ = strategy.validate_geometry("BUY", 100, 95, 110, atr_val=2)
    check("geometry valid BUY", valid)
    invalid, _ = strategy.validate_geometry("BUY", 100, 105, 110, atr_val=2)
    check("geometry invalid BUY terdeteksi", not invalid)

    # quantity/margin test (§12, §59)
    filters = {
        "step_size": Decimal("0.001"), "min_qty": Decimal("0.001"),
        "tick_size": Decimal("0.0001"), "min_notional": Decimal("5"),
    }
    qty, reason = compute_quantity(entry=100.0, margin=1.0, leverage=5.0, filters=filters)
    check(f"quantity valid dihitung ({reason})", qty is not None)

    qty_bad, reason_bad = compute_quantity(entry=100.0, margin=1.0, leverage=0.00001, filters=filters)
    check("quantity margin-deviation ditolak", qty_bad is None)

    # state machine test (§55, §57)
    st = StateStore("/tmp/_selftest_checkpoint.json")
    st.add_pending({"pair": "TESTUSDT", "direction": "BUY", "entry": 1, "tp": 2, "sl": 0.5, "confidence": 50, "reason": [], "components": {}, "setup_type": "x", "regime": "x", "session": "x", "atr": 0.1, "timestamp": 0, "strategy_version": "1.00"}, Decimal("1"), 1.0)
    t1 = st.transition("TESTUSDT", "FILLED", "evt1")
    t1_dup = st.transition("TESTUSDT", "FILLED", "evt1")  # idempotency check
    check("transisi PENDING->FILLED sukses", t1)
    check("event duplikat diabaikan (idempotent)", not t1_dup)
    illegal = st.transition("TESTUSDT", "TRAILING", "evt2")  # FILLED->TRAILING tidak diizinkan langsung
    check("transisi ilegal ditolak", not illegal)

    # ban timer test (§26)
    st.ban("BANUSDT", "TEST", 0.01)
    time.sleep(0.02)
    check("ban expired terdeteksi", not st.is_banned("BANUSDT"))

    # strategy engine sanity test — tidak boleh crash pada data sintetis
    import random
    random.seed(42)
    synthetic = []
    price = 100.0
    for i in range(700):
        o = price
        price += random.uniform(-1, 1.2)
        h, l, c = max(o, price) + 0.2, min(o, price) - 0.2, price
        synthetic.append({"t": i * 900000, "o": o, "h": h, "l": l, "c": c, "v": 100.0})
    strat = strategy.Strategy()
    try:
        result = strat.analyze("TESTUSDT", synthetic, synthetic)
        check("strategy.analyze tidak crash pada data sintetis", True)
    except Exception as e:
        print(f"   exception: {e}")
        check("strategy.analyze tidak crash pada data sintetis", False)

    # learn engine sanity test
    le = learn.LearnEngine(checkpoint_path="/tmp/_selftest_learn.json")
    for i in range(50):
        le.record_trade_outcome(
            {"pair": "TESTUSDT", "direction": "BUY", "confidence": 30 + i, "setup_type": "x",
             "regime": "SIDEWAYS", "session": "ASIA", "components": {}, "strategy_version": "1.00",
             "timestamp": (i * 900000)},
            "TP" if i % 2 == 0 else "INITIAL_SL",
            {"pnl_pct": 1.0 if i % 2 == 0 else -1.0, "pnl_r": 1.0 if i % 2 == 0 else -1.0, "close_time": time.time() * 1000},
        )
    audit_report = le.audit(strat)
    check("learn.audit tidak crash", isinstance(audit_report, dict))
    check("learn.save_checkpoint sukses", le.save_checkpoint())

    # Binance public position-market WS regression: mark price must be accepted
    # into the display cache without touching Binance REST.
    class _FakePositionMarketWS:
        def subscribe(self, *args, **kwargs):
            pass
        def unsubscribe(self, *args, **kwargs):
            pass
    check("Binance position market WS display cache object tersedia", BinancePositionMarketWebSocket is not None)
    check("Binance position WS stream format valid", BinancePositionMarketWebSocket._stream("BTCUSDT") == "btcusdt@markPrice@1s")

    _received_position_prices = []
    _pmws = BinancePositionMarketWebSocket(lambda symbol, price, ts: _received_position_prices.append((symbol, price, ts)))
    _pmws._on_message(None, json.dumps({"stream": "ethusdt@markPrice@1s", "data": {"e": "markPriceUpdate", "E": 123, "s": "ETHUSDT", "p": "2510.1234"}}))
    check("Binance mark-price WS parser valid", _received_position_prices == [("ETHUSDT", 2510.1234, 123.0)])

    # Binance user-data WS regression: a FILLED entry must be accepted without
    # a positionRisk REST probe. This is the key property during HTTP 418/429.
    class _FakeTelegram:
        def send(self, *args, **kwargs):
            pass
    class _FakeWS:
        def unsubscribe(self, *args, **kwargs):
            pass
        def subscribe(self, *args, **kwargs):
            pass
    class _FakeBinanceWS:
        def invalidate_listen_key(self):
            pass
    class _FakeLearn:
        def record_trade_outcome(self, *args, **kwargs):
            pass
    bot = TradingBot.__new__(TradingBot)
    bot.state = StateStore("/tmp/_selftest_ws_state.json")
    bot.state.mode = "REAL"
    bot.state.auto = False
    setup_ws = {
        "pair": "WSUSDT", "direction": "BUY", "entry": 100.0, "tp": 110.0, "sl": 95.0,
        "confidence": 60.0, "reason": [], "components": {}, "setup_type": "x",
        "regime": "SIDEWAYS", "session": "ASIA", "atr": 1.0, "timestamp": 0, "strategy_version": "1.00"
    }
    bot.state.add_pending(setup_ws, Decimal("0.05"), 1.0, status="PENDING")
    pos = bot.state.positions["WSUSDT"]
    pos["binance_order_ids"]["entry"] = 12345
    pos["binance_entry_client_order_id"] = "SMCWSUSDT"
    bot.telegram = _FakeTelegram()
    bot.ws = _FakeWS()
    bot.binance_ws = _FakeBinanceWS()
    bot.learn_engine = _FakeLearn()
    bot._shadow_lock = threading.RLock()
    bot._shadow_candidates = {}
    bot._attach_real_protection = lambda symbol, pos: True
    bot._refresh_real_balance_after_event = lambda *args, **kwargs: None
    bot._finalize_real_close_from_user_event = lambda *args, **kwargs: None
    bot._on_binance_user_event({
        "e": "ORDER_TRADE_UPDATE", "E": 1000,
        "o": {"s": "WSUSDT", "c": "SMCWSUSDT", "i": 12345, "X": "FILLED",
              "x": "TRADE", "o": "LIMIT", "ap": "100.1", "L": "100.1", "z": "0.05"}
    })
    check("REAL fill Binance WS tanpa REST", bot.state.positions["WSUSDT"]["status"] == "FILLED")
    check("REAL fill confirmation tersimpan", bool(bot.state.positions["WSUSDT"].get("real_fill_confirmed")))

    # Governor hard-stop regression: after a synthetic 418, a second request
    # must be blocked locally without issuing another network request.
    bc = BinanceClient("KEY", "SECRET")
    bc._blocked_until_mono = time.monotonic() + 5
    bc._blocked_error = RateLimitError("synthetic 418", status_code=418, retry_after=5.0, code=-1003)
    try:
        bc._request("GET", "/fapi/v2/balance", signed=True)
        governor_blocked = False
    except RateLimitError as e:
        governor_blocked = True and e.status_code == 418
    check("REST governor hard-stop after 418", governor_blocked)

    # Local governor window regression: stale usage must not permanently self-block.
    bc2 = BinanceClient("KEY", "SECRET")
    bc2._used_weight_1m = BINANCE_WEIGHT_HARD_STOP + 10
    bc2._used_weight_1m_ts = time.time() - 61.0
    bc2._used_order_10s = 999
    bc2._used_order_1m = 999
    bc2._last_response_ts = time.time() - 61.0
    bc2._refresh_local_usage_windows()
    check("REST governor stale windows reset", bc2._used_weight_1m == 0 and bc2._used_order_10s == 0 and bc2._used_order_1m == 0)

    # Local REAL balance model regression: one snapshot + local realized PnL,
    # no balance REST refresh on close.
    local_bot = TradingBot.__new__(TradingBot)
    local_bot.state = StateStore("/tmp/_selftest_local_balance.json")
    local_bot.state.mode = "REAL"
    local_bot.state.autostop_pct = None
    local_bot.telegram = _FakeTelegram()
    local_bot._set_real_balance_snapshot(100.0, "TEST_SNAPSHOT")
    local_bot._apply_real_balance_event(2.5, "TEST_CLOSE")
    check("local REAL balance snapshot tersimpan", local_bot.state.real_balance_snapshot == 100.0)
    check("local REAL PnL diterapkan", abs(local_bot.state.current_balance - 102.5) < 1e-9)

    return ok


# =============================================================================
# 10. ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Trading Bot")
    parser.add_argument("--selftest", action="store_true", help="Jalankan self-test offline lalu keluar")
    args = parser.parse_args()

    cfg = Config()
    setup_logging(cfg.state_dir)
    secrets = [cfg.binance_api_secret, cfg.binance_api_key, cfg.bybit_api_secret,
               cfg.bybit_api_key, cfg.telegram_bot_token, cfg.github_token]
    logging.getLogger().addFilter(SecretRedactingFilter(secrets))

    if args.selftest:
        success = run_selftest()
        sys.exit(0 if success else 1)

    bot = TradingBot(cfg)

    def _handle_signal(signum, frame):
        logger.info("Sinyal %s diterima, shutdown...", signum)
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    bot.startup()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.shutdown()



# =============================================================================
# Render launcher compatibility adapter (try.py)
# =============================================================================

_LAUNCHER_BOT = None

async def on_start(context: dict):
    """Dipanggil oleh try.py saat /try."""
    global _LAUNCHER_BOT

    os.environ["RUN_WITH_LAUNCHER"] = "true"
    cfg = Config()
    setup_logging(cfg.state_dir)

    _LAUNCHER_BOT = TradingBot(cfg)
    # /try is a new MAIN session. Never delete a live REAL checkpoint here;
    # shutdown() already handles safe cold-stop semantics. For a clean SIMULASI
    # session, no main checkpoint means startup naturally begins from defaults.
    _LAUNCHER_BOT.startup()

    return True


async def handle_update(update: dict, context: dict = None):
    """Bridge command Telegram dari try.py ke dispatcher lama."""
    global _LAUNCHER_BOT

    try:
        if _LAUNCHER_BOT is None:
            logger.warning("handle_update dipanggil tapi bot belum aktif")
            return None

        # try.py mengirim raw Telegram update. Ambil format standar dan fallback.
        message = update.get("message") or update.get("edited_message") or {}
        text = str(message.get("text") or message.get("caption") or "").strip()
        sender_id = str(message.get("from", {}).get("id", ""))

        logger.info("[COMMAND BRIDGE] menerima: %s (from=%s)", text, sender_id)

        if not text.startswith("/"):
            return None
        if not _LAUNCHER_BOT._is_authorized(sender_id):
            logger.warning("[COMMAND BRIDGE] command dari user tidak diotorisasi (id=%s) — diabaikan", sender_id)
            return None
        await asyncio.to_thread(_LAUNCHER_BOT._dispatch_command, text)

    except Exception as e:
        logger.exception("handle_update gagal: %s", e)
    return None


async def on_stop(context: dict):
    """Dipanggil oleh try.py saat /end."""
    global _LAUNCHER_BOT

    if _LAUNCHER_BOT is not None:
        _LAUNCHER_BOT.shutdown(fresh_session=True)
        _LAUNCHER_BOT = None


if __name__ == "__main__":
    main()
