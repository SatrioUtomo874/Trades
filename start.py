from __future__ import annotations

"""
SMCAutoTrade - start.py

Data infrastructure only.

Responsibilities:
- expose a small Telegram command surface through main.py
- detect the public IP of the running server
- build a common USDT perpetual symbol universe from Bybit + Binance
- bootstrap 700 x 15m OHLCV candles from Bybit REST, one symbol per second
- keep the dataset in memory for strategy.py
- switch to Bybit public linear WebSocket kline updates after bootstrap
- maintain reconnect/heartbeat logic

main.py owns Telegram polling. This file MUST NOT call getUpdates.

Expected launcher context:
    {
        "chat_id": int | None,
        "user_id": int | None,
        "stop_event": threading.Event,
        "send_message": callable,
        "launcher": "main.py",
        "start_file": str,
        "is_running": callable,
    }

Environment variables:
- DATA_MAX_SYMBOLS=250
- DATA_CANDLES=700
- DATA_TIMEFRAME=15
- DATA_LOAD_INTERVAL=1.0
- BYBIT_BASE_URL=https://api.bybit.com
- BYBIT_WS_URL=wss://stream.bybit.com/v5/public/linear
- BINANCE_BASE_URL=https://fapi.binance.com
- IP_URL=https://api.ipify.org
- REQUEST_TIMEOUT=20
- WS_PING_INTERVAL=20
- WS_RECONNECT_MAX=30
- LOG_EVERY_TICK_SECONDS=15
- DATA_RETENTION=750

Dependencies:
- requests
- websocket-client
"""

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import requests
import websocket

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BYBIT_BASE_URL = (os.getenv("BYBIT_BASE_URL") or "https://api.bybit.com").rstrip("/")
BYBIT_WS_URL = (os.getenv("BYBIT_WS_URL") or "wss://stream.bybit.com/v5/public/linear").strip()
BINANCE_BASE_URL = (os.getenv("BINANCE_BASE_URL") or "https://fapi.binance.com").rstrip("/")
IP_URL = (os.getenv("IP_URL") or "https://api.ipify.org").strip()

DATA_MAX_SYMBOLS = max(1, int(os.getenv("DATA_MAX_SYMBOLS", "250")))
DATA_CANDLES = max(1, min(1000, int(os.getenv("DATA_CANDLES", "700"))))
DATA_TIMEFRAME = (os.getenv("DATA_TIMEFRAME") or "15").strip()
DATA_LOAD_INTERVAL = max(0.0, float(os.getenv("DATA_LOAD_INTERVAL", "1.0")))
DATA_RETENTION = max(DATA_CANDLES, int(os.getenv("DATA_RETENTION", "750")))
REQUEST_TIMEOUT = max(5, int(os.getenv("REQUEST_TIMEOUT", "20")))
WS_PING_INTERVAL = max(5, int(os.getenv("WS_PING_INTERVAL", "20")))
WS_RECONNECT_MAX = max(5, int(os.getenv("WS_RECONNECT_MAX", "30")))
LOG_EVERY_TICK_SECONDS = max(1, int(os.getenv("LOG_EVERY_TICK_SECONDS", "15")))

# Pair discovery is intentionally based on USDT perpetuals on both venues.
# This makes the universe comparable with Bybit category=linear and Binance
# USD-M futures rather than mixing spot and derivative symbols.
BYBIT_CATEGORY = "linear"
BINANCE_SYMBOLS_ENDPOINT = "/fapi/v1/exchangeInfo"
BYBIT_INSTRUMENTS_ENDPOINT = "/v5/market/instruments-info"
BYBIT_KLINE_ENDPOINT = "/v5/market/kline"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=(os.getenv("LOG_LEVEL") or "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("data-engine")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float = 0.0
    confirmed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "turnover": self.turnover,
            "confirmed": self.confirmed,
        }


class DataStore:
    """Thread-safe in-memory OHLCV store used by strategy.py later."""

    def __init__(self, retention: int) -> None:
        self._retention = retention
        self._lock = threading.RLock()
        self._candles: dict[str, list[Candle]] = {}
        self._prices: dict[str, float] = {}
        self._last_update: dict[str, int] = {}

    def set_history(self, symbol: str, candles: list[Candle]) -> None:
        ordered = sorted(candles, key=lambda c: c.timestamp)
        with self._lock:
            self._candles[symbol] = ordered[-self._retention :]
            if ordered:
                self._prices[symbol] = ordered[-1].close
                self._last_update[symbol] = ordered[-1].timestamp

    def upsert_candle(self, symbol: str, candle: Candle) -> None:
        with self._lock:
            series = self._candles.setdefault(symbol, [])
            if series and candle.timestamp == series[-1].timestamp:
                series[-1] = candle
            elif not series or candle.timestamp > series[-1].timestamp:
                series.append(candle)
            else:
                # Rare out-of-order update: replace by timestamp or insert.
                for idx, existing in enumerate(series):
                    if existing.timestamp == candle.timestamp:
                        series[idx] = candle
                        break
                    if existing.timestamp > candle.timestamp:
                        series.insert(idx, candle)
                        break
            if len(series) > self._retention:
                del series[: len(series) - self._retention]
            self._prices[symbol] = candle.close
            self._last_update[symbol] = candle.timestamp

    def get_candles(self, symbol: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            series = list(self._candles.get(symbol, []))
        if limit is not None:
            series = series[-max(1, limit) :]
        return [c.as_dict() for c in series]

    def get_latest(self, symbol: str) -> dict[str, Any] | None:
        with self._lock:
            series = self._candles.get(symbol, [])
            if not series:
                return None
            return series[-1].as_dict()

    def get_price(self, symbol: str) -> float | None:
        with self._lock:
            return self._prices.get(symbol)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {symbol: [c.as_dict() for c in candles] for symbol, candles in self._candles.items()}

    def symbol_count(self) -> int:
        with self._lock:
            return len(self._candles)

    def candle_count(self, symbol: str) -> int:
        with self._lock:
            return len(self._candles.get(symbol, []))


class DataEngine:
    """Owns exchange discovery, historical loading, and live streaming."""

    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context
        self.stop_event: threading.Event = context["stop_event"]
        self.send_message: Callable[[int, Any], None] = context["send_message"]
        self.chat_id: int | None = context.get("chat_id")

        self.store = DataStore(DATA_RETENTION)
        self.symbols: list[str] = []
        self._symbol_lock = threading.RLock()
        self._run_lock = threading.RLock()
        self._auto_running = False
        self._bootstrap_done = False
        self._ws_thread: threading.Thread | None = None
        self._bootstrap_thread: threading.Thread | None = None
        self._ws: websocket.WebSocketApp | None = None
        self._last_tick_log: dict[str, float] = {}
        self._last_ws_message = 0.0
        self._restart_requested = threading.Event()

    # ------------------------- common helpers -------------------------

    def _notify(self, text: str) -> None:
        chat_id = self.chat_id
        if chat_id is None:
            return
        try:
            self.send_message(chat_id, text)
        except Exception:
            log.exception("Telegram notification failed")

    @staticmethod
    def _http_get(url: str, params: dict[str, Any]) -> requests.Response:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response

    def _public_ip(self) -> str:
        response = requests.get(IP_URL, timeout=10)
        response.raise_for_status()
        return response.text.strip()

    # ------------------------- exchange discovery -------------------------

    def discover_bybit_symbols(self) -> set[str]:
        symbols: set[str] = set()
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {
                "category": BYBIT_CATEGORY,
                "status": "Trading",
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor

            data = self._http_get(
                f"{BYBIT_BASE_URL}{BYBIT_INSTRUMENTS_ENDPOINT}", params
            ).json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit instruments error: {data}")

            result = data.get("result") or {}
            for item in result.get("list") or []:
                if item.get("status") != "Trading":
                    continue
                if item.get("contractType") != "LinearPerpetual":
                    continue
                if item.get("quoteCoin") != "USDT" or item.get("settleCoin") != "USDT":
                    continue
                symbol = str(item.get("symbol") or "").upper()
                if symbol:
                    symbols.add(symbol)

            cursor = result.get("nextPageCursor") or None
            if not cursor:
                break

        log.info("[DISCOVERY] Bybit linear USDT perpetuals: %d", len(symbols))
        return symbols

    def discover_binance_symbols(self) -> set[str]:
        data = self._http_get(
            f"{BINANCE_BASE_URL}{BINANCE_SYMBOLS_ENDPOINT}", {}
        ).json()
        symbols: set[str] = set()
        for item in data.get("symbols") or []:
            if item.get("status") != "TRADING":
                continue
            if item.get("contractType") != "PERPETUAL":
                continue
            if item.get("quoteAsset") != "USDT" or item.get("marginAsset") != "USDT":
                continue
            symbol = str(item.get("symbol") or "").upper()
            if symbol:
                symbols.add(symbol)
        log.info("[DISCOVERY] Binance USD-M USDT perpetuals: %d", len(symbols))
        return symbols

    def build_universe(self) -> list[str]:
        log.info("[DISCOVERY] Querying Bybit instruments...")
        bybit = self.discover_bybit_symbols()

        log.info("[DISCOVERY] Querying Binance exchangeInfo...")
        binance = self.discover_binance_symbols()

        common = sorted(bybit & binance)
        selected = common[:DATA_MAX_SYMBOLS]

        with self._symbol_lock:
            self.symbols = selected

        log.info(
            "[DISCOVERY] Common symbols=%d | selected=%d | max=%d",
            len(common),
            len(selected),
            DATA_MAX_SYMBOLS,
        )

        if common:
            log.info("[DISCOVERY] First symbols: %s", ", ".join(common[:20]))
        return selected

    # ------------------------- historical bootstrap -------------------------

    def fetch_bybit_klines(self, symbol: str) -> list[Candle]:
        response = self._http_get(
            f"{BYBIT_BASE_URL}{BYBIT_KLINE_ENDPOINT}",
            {
                "category": BYBIT_CATEGORY,
                "symbol": symbol,
                "interval": DATA_TIMEFRAME,
                "limit": DATA_CANDLES,
            },
        )
        data = response.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit kline error for {symbol}: {data}")

        raw = ((data.get("result") or {}).get("list")) or []
        candles: list[Candle] = []
        for row in raw:
            if len(row) < 6:
                continue
            candles.append(
                Candle(
                    timestamp=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    turnover=float(row[6]) if len(row) > 6 else 0.0,
                    confirmed=True,
                )
            )
        candles.sort(key=lambda c: c.timestamp)
        return candles[-DATA_CANDLES:]

    def bootstrap_symbol(self, symbol: str) -> bool:
        started = time.monotonic()
        try:
            log.info(
                "[REST] %s | requesting %d candles | timeframe=%s",
                symbol,
                DATA_CANDLES,
                DATA_TIMEFRAME,
            )
            candles = self.fetch_bybit_klines(symbol)
            if len(candles) < DATA_CANDLES:
                log.warning(
                    "[DATA] %s | received %d/%d candles",
                    symbol,
                    len(candles),
                    DATA_CANDLES,
                )
            else:
                log.info("[DATA] %s | %d candles received", symbol, len(candles))
            self.store.set_history(symbol, candles)
            elapsed = time.monotonic() - started
            log.info("[DATA] %s | history stored | %.2fs", symbol, elapsed)
            return bool(candles)
        except Exception as exc:
            log.exception("[REST] %s | bootstrap failed: %s", symbol, exc)
            return False

    def bootstrap_all(self) -> None:
        with self._symbol_lock:
            symbols = list(self.symbols)

        total = len(symbols)
        if total == 0:
            self._notify("❌ /auto gagal: tidak ada common USDT perpetual pair.")
            return

        success = 0
        failure = 0
        log.info("[BOOTSTRAP] Starting historical load: %d symbols", total)

        for index, symbol in enumerate(symbols, start=1):
            if self.stop_event.is_set() or not self._auto_running:
                log.info("[BOOTSTRAP] stopped at %d/%d", index - 1, total)
                return

            log.info("[BOOTSTRAP] %d/%d | %s", index, total, symbol)
            ok = self.bootstrap_symbol(symbol)
            if ok:
                success += 1
            else:
                failure += 1

            # User requested a paced load: one symbol per second.
            if index < total:
                remaining = DATA_LOAD_INTERVAL
                deadline = time.monotonic() + remaining
                while remaining > 0 and not self.stop_event.is_set() and self._auto_running:
                    time.sleep(min(0.25, remaining))
                    remaining = deadline - time.monotonic()

        if self.stop_event.is_set() or not self._auto_running:
            return

        self._bootstrap_done = True
        log.info(
            "[BOOTSTRAP] COMPLETE | success=%d | failed=%d | symbols=%d | candles/symbol=%d",
            success,
            failure,
            total,
            DATA_CANDLES,
        )
        self._notify(
            "✅ Historical data ready\n"
            f"{success}/{total} symbols loaded\n"
            f"{DATA_CANDLES} candles/symbol\n"
            f"Timeframe: {DATA_TIMEFRAME}M\n\n"
            "Starting Bybit WebSocket..."
        )
        self.start_websocket()

    # ------------------------- websocket -------------------------

    def _ws_topics(self) -> list[str]:
        with self._symbol_lock:
            symbols = list(self.symbols)
        return [f"kline.{DATA_TIMEFRAME}.{symbol}" for symbol in symbols]

    def _ws_on_open(self, ws: websocket.WebSocketApp) -> None:
        self._last_ws_message = time.monotonic()
        topics = self._ws_topics()
        log.info("[WS] Connected | subscribing to %d kline topics", len(topics))

        # Bybit currently has no args-count limit for futures, but it does cap
        # the total args payload. Sending one subscribe packet keeps the code
        # simple and comfortably below the documented 21,000-character cap for
        # this 250-symbol design.
        payload = {"op": "subscribe", "args": topics}
        ws.send(json.dumps(payload))
        log.info("[WS] Subscribe request sent | topics=%d", len(topics))
        self._notify(f"🟢 MARKET DATA ONLINE\nWebSocket: Bybit\nSymbols: {len(topics)}")

    def _ws_on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        self._last_ws_message = time.monotonic()
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            log.warning("[WS] Non-JSON message ignored: %r", message[:300])
            return

        if payload.get("op") in {"pong", "subscribe", "unsubscribe"}:
            if payload.get("op") == "subscribe":
                log.info("[WS] Subscription response: %s", payload)
            return

        topic = str(payload.get("topic") or "")
        if not topic.startswith("kline."):
            return

        data = payload.get("data") or []
        for item in data:
            symbol = self._symbol_from_topic(topic, item)
            if not symbol:
                continue
            candle = self._parse_ws_candle(item)
            if candle is None:
                continue
            self.store.upsert_candle(symbol, candle)
            self._log_live_update(symbol, candle)

    @staticmethod
    def _symbol_from_topic(topic: str, item: dict[str, Any]) -> str:
        symbol = str(item.get("symbol") or "").upper()
        if symbol:
            return symbol
        parts = topic.split(".")
        return parts[-1].upper() if parts else ""

    @staticmethod
    def _parse_ws_candle(item: dict[str, Any]) -> Candle | None:
        try:
            return Candle(
                timestamp=int(item["start"]),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item["volume"]),
                turnover=float(item.get("turnover") or 0.0),
                confirmed=bool(item.get("confirm", False)),
            )
        except (KeyError, TypeError, ValueError):
            log.exception("[WS] Invalid kline payload: %r", item)
            return None

    def _log_live_update(self, symbol: str, candle: Candle) -> None:
        now = time.monotonic()
        last = self._last_tick_log.get(symbol, 0.0)
        if now - last < LOG_EVERY_TICK_SECONDS:
            return
        self._last_tick_log[symbol] = now
        state = "CLOSED" if candle.confirmed else "LIVE"
        log.info(
            "[TICK] %s | %s | O=%s H=%s L=%s C=%s V=%s",
            symbol,
            state,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )

    def _ws_on_error(self, _ws: websocket.WebSocketApp, error: Any) -> None:
        log.warning("[WS] error: %s", error)

    def _ws_on_close(self, _ws: websocket.WebSocketApp, status_code: Any, msg: Any) -> None:
        log.warning("[WS] closed | code=%s | msg=%s", status_code, msg)

    def _ws_ping_loop(self, ws: websocket.WebSocketApp) -> None:
        while self._auto_running and not self.stop_event.is_set() and self._ws is ws:
            if self.stop_event.wait(WS_PING_INTERVAL):
                return
            try:
                ws.send(json.dumps({"op": "ping"}))
            except Exception:
                return

    def _ws_worker(self) -> None:
        backoff = 2
        while self._auto_running and not self.stop_event.is_set():
            try:
                topics = self._ws_topics()
                if not topics:
                    log.warning("[WS] No topics available")
                    return

                ws = websocket.WebSocketApp(
                    BYBIT_WS_URL,
                    on_open=self._ws_on_open,
                    on_message=self._ws_on_message,
                    on_error=self._ws_on_error,
                    on_close=self._ws_on_close,
                )
                self._ws = ws

                ping_thread: threading.Thread | None = None

                def _open_with_ping(app: websocket.WebSocketApp) -> None:
                    self._ws_on_open(app)
                    nonlocal ping_thread
                    ping_thread = threading.Thread(
                        target=self._ws_ping_loop,
                        args=(app,),
                        name="bybit-ws-ping",
                        daemon=True,
                    )
                    ping_thread.start()

                ws.on_open = _open_with_ping

                log.info("[WS] Connecting to %s", BYBIT_WS_URL)
                ws.run_forever(
                    ping_interval=None,
                    ping_timeout=None,
                    skip_utf8_validation=True,
                )

                self._ws = None
                if self._auto_running and not self.stop_event.is_set():
                    log.warning("[WS] Disconnected; reconnecting in %ss", backoff)
                    self.stop_event.wait(backoff)
                    backoff = min(backoff * 2, WS_RECONNECT_MAX)
                else:
                    return
            except Exception:
                log.exception("[WS] worker crashed")
                self._ws = None
                if self._auto_running and not self.stop_event.is_set():
                    log.warning("[WS] retrying in %ss", backoff)
                    self.stop_event.wait(backoff)
                    backoff = min(backoff * 2, WS_RECONNECT_MAX)
                else:
                    return
            else:
                backoff = 2

    def start_websocket(self) -> None:
        if self._ws_thread and self._ws_thread.is_alive():
            log.info("[WS] Already running")
            return
        self._ws_thread = threading.Thread(
            target=self._ws_worker,
            name="bybit-kline-ws",
            daemon=True,
        )
        self._ws_thread.start()

    def stop_websocket(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                log.exception("[WS] close failed")

    # ------------------------- lifecycle -------------------------

    def start_auto(self) -> str:
        with self._run_lock:
            if self._auto_running:
                return "ℹ️ /auto sudah aktif atau sedang berjalan."
            self._auto_running = True
            self._bootstrap_done = False

        log.info("[AUTO] Starting scan mode")
        self._notify("🤖 AUTO MODE STARTED\nScanning Bybit + Binance symbols...")

        try:
            selected = self.build_universe()
        except Exception as exc:
            log.exception("[AUTO] Symbol discovery failed")
            self._auto_running = False
            self._notify(f"❌ Symbol discovery gagal: {exc}")
            return f"❌ /auto gagal: {exc}"

        if not selected:
            self._auto_running = False
            return "❌ Tidak menemukan pair common Bybit + Binance."

        self._notify(
            "✅ Symbol scan selesai\n"
            f"Common pairs selected: {len(selected)}\n"
            "Source historical: Bybit REST\n"
            f"Timeframe: {DATA_TIMEFRAME}M\n"
            f"History: {DATA_CANDLES} candles/pair\n\n"
            "Historical loading dimulai..."
        )

        self._bootstrap_thread = threading.Thread(
            target=self.bootstrap_all,
            name="historical-bootstrap",
            daemon=True,
        )
        self._bootstrap_thread.start()
        return f"🟢 /auto aktif. Scanning + bootstrap dimulai untuk {len(selected)} pair."

    def stop(self) -> None:
        with self._run_lock:
            self._auto_running = False
        self.stop_websocket()
        self._notify("⏹️ Data engine dihentikan.")
        log.info("[ENGINE] stopped")

    # ------------------------- public data API -------------------------

    def get_symbols(self) -> list[str]:
        with self._symbol_lock:
            return list(self.symbols)

    def get_candles(self, symbol: str, limit: int | None = None) -> list[dict[str, Any]]:
        return self.store.get_candles(symbol.upper(), limit)

    def get_latest_candle(self, symbol: str) -> dict[str, Any] | None:
        return self.store.get_latest(symbol.upper())

    def get_price(self, symbol: str) -> float | None:
        return self.store.get_price(symbol.upper())

    def get_market_snapshot(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        return {
            "symbol": symbol,
            "timeframe": DATA_TIMEFRAME,
            "source": "Bybit",
            "candle_count": self.store.candle_count(symbol),
            "price": self.store.get_price(symbol),
            "latest_candle": self.store.get_latest(symbol),
        }

    def get_status(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "auto_running": self._auto_running,
            "bootstrap_done": self._bootstrap_done,
            "websocket_connected": self._ws is not None,
            "symbols": len(self.symbols),
            "symbols_loaded": self.store.symbol_count(),
            "timeframe": DATA_TIMEFRAME,
            "history_target": DATA_CANDLES,
            "retention": DATA_RETENTION,
            "last_ws_activity_seconds": None
            if not self._last_ws_message
            else round(now - self._last_ws_message, 1),
        }


# ---------------------------------------------------------------------------
# Module-level lifecycle required by main.py
# ---------------------------------------------------------------------------

_ENGINE_LOCK = threading.RLock()
_ENGINE: DataEngine | None = None
_PUBLIC_IP: str | None = None


def _server_ip_text() -> str:
    global _PUBLIC_IP
    if _PUBLIC_IP:
        return _PUBLIC_IP
    try:
        response = requests.get(IP_URL, timeout=10)
        response.raise_for_status()
        _PUBLIC_IP = response.text.strip()
    except Exception as exc:
        log.warning("[NET] Public IP lookup failed: %s", exc)
        # Local fallback helps diagnose deployments with blocked external IP APIs.
        try:
            _PUBLIC_IP = socket.gethostbyname(socket.gethostname())
        except Exception:
            _PUBLIC_IP = "unknown"
    return _PUBLIC_IP


def on_start(context: dict[str, Any]) -> bool:
    global _ENGINE
    ip = _server_ip_text()
    log.info("[START] start.py initialized")
    log.info("[NET] Server public IP: %s", ip)
    log.info("[CONFIG] Bybit REST: %s", BYBIT_BASE_URL)
    log.info("[CONFIG] Bybit WS: %s", BYBIT_WS_URL)
    log.info("[CONFIG] Binance REST: %s", BINANCE_BASE_URL)
    log.info(
        "[CONFIG] max_symbols=%d candles=%d timeframe=%s load_interval=%.2fs retention=%d",
        DATA_MAX_SYMBOLS,
        DATA_CANDLES,
        DATA_TIMEFRAME,
        DATA_LOAD_INTERVAL,
        DATA_RETENTION,
    )

    with _ENGINE_LOCK:
        _ENGINE = DataEngine(context)
        engine = _ENGINE

    chat_id = context.get("chat_id")
    if chat_id is not None:
        try:
            context["send_message"](
                chat_id,
                "🟢 start.py aktif\n"
                f"Server IP: {ip}\n"
                "Data engine: READY\n"
                "Use /IP untuk melihat IP lagi.\n"
                "Use /auto untuk mulai scanning mode.",
            )
        except Exception:
            log.exception("[TELEGRAM] startup notification failed")

    return engine is not None


def on_stop(context: dict[str, Any]) -> bool:
    global _ENGINE
    with _ENGINE_LOCK:
        engine = _ENGINE
        _ENGINE = None
    if engine is not None:
        engine.stop()
    log.info("[STOP] start.py cleaned up")
    return True


def _engine_or_raise() -> DataEngine:
    with _ENGINE_LOCK:
        engine = _ENGINE
    if engine is None:
        raise RuntimeError("Data engine belum diinisialisasi.")
    return engine


def handle_update(update: dict[str, Any], context: dict[str, Any]) -> str | None:
    """Main.py forwards complete Telegram updates here."""
    message = update.get("message") or {}
    text = str(message.get("text") or message.get("caption") or "").strip()
    if not text:
        return None

    command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()

    if command == "/ip":
        ip = _server_ip_text()
        log.info("[CMD] /IP -> %s", ip)
        return f"🌐 Server IP\n{ip}"

    if command == "/auto":
        return _engine_or_raise().start_auto()

    if command == "/status":
        engine = _engine_or_raise()
        status = engine.get_status()
        return (
            "📊 DATA ENGINE STATUS\n"
            f"AUTO: {'ON' if status['auto_running'] else 'OFF'}\n"
            f"BOOTSTRAP: {'DONE' if status['bootstrap_done'] else 'RUNNING/WAITING'}\n"
            f"WEBSOCKET: {'CONNECTED' if status['websocket_connected'] else 'OFF'}\n"
            f"Symbols: {status['symbols']}\n"
            f"Loaded: {status['symbols_loaded']}\n"
            f"History target: {status['history_target']}\n"
            f"Timeframe: {status['timeframe']}M\n"
            f"Retention: {status['retention']}"
        )

    if command == "/symbols":
        engine = _engine_or_raise()
        symbols = engine.get_symbols()
        if not symbols:
            return "ℹ️ Symbol universe belum dibuat. Jalankan /auto."
        preview = ", ".join(symbols[:50])
        suffix = " ..." if len(symbols) > 50 else ""
        return f"🪙 Symbols: {len(symbols)}\n{preview}{suffix}"

    if command == "/candles":
        parts = text.split()
        if len(parts) < 2:
            return "Usage: /candles BTCUSDT [limit]"
        symbol = parts[1].upper()
        limit = None
        if len(parts) >= 3:
            try:
                limit = int(parts[2])
            except ValueError:
                return "❌ Limit harus berupa angka."
        engine = _engine_or_raise()
        candles = engine.get_candles(symbol, limit)
        if not candles:
            return f"❌ Tidak ada data candle untuk {symbol}."
        latest = candles[-1]
        return (
            f"📈 {symbol} | {len(candles)} candles\n"
            f"TF: {DATA_TIMEFRAME}M\n"
            f"Last: {latest['close']}\n"
            f"Timestamp: {latest['timestamp']}"
        )

    if command == "/price":
        parts = text.split()
        if len(parts) < 2:
            return "Usage: /price BTCUSDT"
        symbol = parts[1].upper()
        price = _engine_or_raise().get_price(symbol)
        return f"💹 {symbol}: {price}" if price is not None else f"❌ Price {symbol} belum tersedia."

    if command == "/help":
        return (
            "🤖 DATA ENGINE COMMANDS\n\n"
            "/IP — server public IP\n"
            "/auto — scan pair + load 700 candle 15M + start WebSocket\n"
            "/status — engine status\n"
            "/symbols — selected symbols\n"
            "/candles BTCUSDT [limit] — inspect candle count/latest\n"
            "/price BTCUSDT — latest stored price"
        )

    # Ignore unknown commands for now; strategy.py will later own its own
    # command namespace through this same forwarding mechanism.
    return None


# ---------------------------------------------------------------------------
# Optional local smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("start.py is designed to be loaded by main.py. Use /try in Telegram.")
