from __future__ import annotations

"""SMCAutoTrade start_v4.py - market data infrastructure.

VERSION: 4.0

Launcher contract:
    on_start(context)
    on_stop(context)
    handle_update(update, context)

This module owns data only. It never calls Telegram getUpdates.

Data backbone:
    15m: 700 candles per symbol (primary historical series)
    5m : 500 candles per symbol
    1m : 500 candles per symbol

Live source after bootstrap:
    Bybit public linear WebSocket for 15m, 5m and 1m kline streams.

Strategy integration:
    strategy.py is loaded by this module after bootstrap and receives a
    DataAPI object. Strategy code is kept out of the data engine itself.
"""

import importlib.util
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests
import websocket

BYBIT_BASE_URL = (os.getenv("BYBIT_BASE_URL") or "https://api.bybit.com").rstrip("/")
BYBIT_WS_URL = (os.getenv("BYBIT_WS_URL") or "wss://stream.bybit.com/v5/public/linear").strip()
BINANCE_BASE_URL = (os.getenv("BINANCE_BASE_URL") or "https://fapi.binance.com").rstrip("/")
IP_URL = (os.getenv("IP_URL") or "https://api.ipify.org").strip()

MAX_SYMBOLS = max(1, int(os.getenv("DATA_MAX_SYMBOLS", "250")))
TF_CONFIG = {"15": 700, "5": 500, "1": 500}
LOAD_INTERVAL = max(0.0, float(os.getenv("DATA_LOAD_INTERVAL", "1.0")))
RETENTION_EXTRA = max(50, int(os.getenv("DATA_RETENTION_EXTRA", "50")))
REQUEST_TIMEOUT = max(5, int(os.getenv("REQUEST_TIMEOUT", "20")))
WS_PING_INTERVAL = max(5, int(os.getenv("WS_PING_INTERVAL", "20")))
WS_RECONNECT_MAX = max(5, int(os.getenv("WS_RECONNECT_MAX", "30")))
LOG_EVERY_SYMBOL_TICK = max(2, int(os.getenv("LOG_EVERY_SYMBOL_TICK", "15")))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_FILE = os.getenv("STRATEGY_FILE", "strategy.py")

BYBIT_CATEGORY = "linear"
BYBIT_INSTRUMENTS = "/v5/market/instruments-info"
BYBIT_KLINE = "/v5/market/kline"

logging.basicConfig(
    level=(os.getenv("LOG_LEVEL") or "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("data-engine")


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
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, dict[str, list[Candle]]] = {}
        self._prices: dict[str, float] = {}
        self._last_event: dict[str, int] = {}
        self._price_ts: dict[str, int] = {}

    def set_history(self, symbol: str, timeframe: str, candles: list[Candle]) -> None:
        ordered = sorted(candles, key=lambda c: c.timestamp)
        keep = TF_CONFIG[timeframe] + RETENTION_EXTRA
        with self._lock:
            self._data.setdefault(symbol, {})[timeframe] = ordered[-keep:]
            if timeframe == "1" and ordered:
                self._prices[symbol] = ordered[-1].close
                self._price_ts[symbol] = ordered[-1].timestamp
                self._last_event[symbol] = ordered[-1].timestamp

    def upsert(self, symbol: str, timeframe: str, candle: Candle) -> None:
        with self._lock:
            series = self._data.setdefault(symbol, {}).setdefault(timeframe, [])
            if series and candle.timestamp == series[-1].timestamp:
                series[-1] = candle
            elif not series or candle.timestamp > series[-1].timestamp:
                series.append(candle)
            else:
                for i, old in enumerate(series):
                    if old.timestamp == candle.timestamp:
                        series[i] = candle
                        break
                    if old.timestamp > candle.timestamp:
                        series.insert(i, candle)
                        break
            keep = TF_CONFIG.get(timeframe, 1000) + RETENTION_EXTRA
            if len(series) > keep:
                del series[: len(series) - keep]
            if candle.timestamp >= self._price_ts.get(symbol, -1):
                self._prices[symbol] = candle.close
                self._price_ts[symbol] = candle.timestamp
            self._last_event[symbol] = max(candle.timestamp, self._last_event.get(symbol, -1))

    def get(self, symbol: str, timeframe: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            series = list(self._data.get(symbol.upper(), {}).get(timeframe, []))
        if limit is not None:
            series = series[-max(1, int(limit)):]
        return [c.as_dict() for c in series]

    def latest(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        with self._lock:
            series = self._data.get(symbol.upper(), {}).get(timeframe, [])
            return series[-1].as_dict() if series else None

    def price(self, symbol: str) -> float | None:
        with self._lock:
            return self._prices.get(symbol.upper())

    def counts(self, symbol: str) -> dict[str, int]:
        with self._lock:
            d = self._data.get(symbol.upper(), {})
            return {tf: len(d.get(tf, [])) for tf in TF_CONFIG}


class DataAPI:
    """Read-only facade presented to strategy.py."""

    def __init__(self, engine: "DataEngine") -> None:
        self._engine = engine

    def get_symbols(self) -> list[str]:
        return self._engine.get_symbols()

    def get_candles(self, symbol: str, timeframe: str = "15", limit: int | None = None) -> list[dict[str, Any]]:
        return self._engine.store.get(symbol.upper(), str(timeframe), limit)

    def get_latest_candle(self, symbol: str, timeframe: str = "15") -> dict[str, Any] | None:
        return self._engine.store.latest(symbol.upper(), str(timeframe))

    def get_price(self, symbol: str) -> float | None:
        return self._engine.store.price(symbol.upper())

    def get_snapshot(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        return {
            "symbol": symbol,
            "price": self.get_price(symbol),
            "timeframes": {tf: self.get_candles(symbol, tf) for tf in TF_CONFIG},
        }

    def is_bootstrap_complete(self) -> bool:
        return self._engine.bootstrap_complete

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._engine.add_data_callback(callback)


class DataEngine:
    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context
        self.stop_event: threading.Event = context["stop_event"]
        self.send_message: Callable[[int, Any], None] = context["send_message"]
        self.chat_id: int | None = context.get("chat_id")

        self.store = DataStore()
        self.api = DataAPI(self)
        self.symbols: list[str] = []
        self._symbol_lock = threading.RLock()
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._run_lock = threading.RLock()
        self.auto_running = False
        self.bootstrap_complete = False
        self.ws: websocket.WebSocketApp | None = None
        self.ws_thread: threading.Thread | None = None
        self.bootstrap_thread: threading.Thread | None = None
        self.strategy: Any = None
        self.strategy_error: str | None = None
        self._tick_logs: dict[tuple[str, str], float] = {}

    def _notify(self, text: str) -> None:
        if self.chat_id is None:
            return
        try:
            self.send_message(self.chat_id, text)
        except Exception:
            log.exception("telegram notification failed")

    @staticmethod
    def _get(url: str, params: dict[str, Any]) -> requests.Response:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r

    def public_ip(self) -> str:
        r = requests.get(IP_URL, timeout=10)
        r.raise_for_status()
        return r.text.strip()

    # ---------------- symbol discovery ----------------
    def bybit_symbols(self) -> set[str]:
        out: set[str] = set()
        cursor = None
        while True:
            params: dict[str, Any] = {
                "category": BYBIT_CATEGORY,
                "status": "Trading",
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor
            data = self._get(f"{BYBIT_BASE_URL}{BYBIT_INSTRUMENTS}", params).json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit instruments error: {data}")
            result = data.get("result") or {}
            for x in result.get("list") or []:
                if x.get("status") == "Trading" and x.get("contractType") == "LinearPerpetual" and x.get("quoteCoin") == "USDT" and x.get("settleCoin") == "USDT":
                    s = str(x.get("symbol") or "").upper()
                    if s:
                        out.add(s)
            cursor = result.get("nextPageCursor") or None
            if not cursor:
                break
        return out

    def binance_symbols(self) -> set[str]:
        data = self._get(f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo", {}).json()
        out: set[str] = set()
        for x in data.get("symbols") or []:
            if x.get("status") == "TRADING" and x.get("contractType") == "PERPETUAL" and x.get("quoteAsset") == "USDT" and x.get("marginAsset") == "USDT":
                s = str(x.get("symbol") or "").upper()
                if s:
                    out.add(s)
        return out

    def build_universe(self) -> list[str]:
        b1 = self.bybit_symbols()
        log.info("[DISCOVERY] Bybit USDT perpetuals=%d", len(b1))
        b2 = self.binance_symbols()
        log.info("[DISCOVERY] Binance USDT perpetuals=%d", len(b2))
        common = sorted(b1 & b2)
        selected = common[:MAX_SYMBOLS]
        with self._symbol_lock:
            self.symbols = selected
        log.info("[DISCOVERY] common=%d selected=%d", len(common), len(selected))
        return selected

    # ---------------- historical ----------------
    def fetch_klines(self, symbol: str, timeframe: str, limit: int) -> list[Candle]:
        data = self._get(
            f"{BYBIT_BASE_URL}{BYBIT_KLINE}",
            {"category": BYBIT_CATEGORY, "symbol": symbol, "interval": timeframe, "limit": limit},
        ).json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit kline error {symbol}/{timeframe}: {data}")
        candles: list[Candle] = []
        for row in ((data.get("result") or {}).get("list") or []):
            if len(row) < 6:
                continue
            candles.append(Candle(int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]) if len(row) > 6 else 0.0, True))
        candles.sort(key=lambda c: c.timestamp)
        return candles[-limit:]

    def bootstrap_symbol(self, symbol: str) -> bool:
        ok = True
        for tf, limit in TF_CONFIG.items():
            if self.stop_event.is_set() or not self.auto_running:
                return False
            t0 = time.monotonic()
            try:
                candles = self.fetch_klines(symbol, tf, limit)
                self.store.set_history(symbol, tf, candles)
                if len(candles) < limit:
                    ok = False
                    log.warning("[REST] %s %s | got %d/%d", symbol, tf, len(candles), limit)
                else:
                    log.info("[REST] %s %s | loaded %d | %.2fs", symbol, tf, len(candles), time.monotonic() - t0)
            except Exception:
                ok = False
                log.exception("[REST] %s %s failed", symbol, tf)
        return ok

    def bootstrap_all(self) -> None:
        symbols = self.get_symbols()
        success = 0
        failure = 0
        total = len(symbols)
        log.info("[BOOTSTRAP] start %d symbols", total)
        for idx, symbol in enumerate(symbols, 1):
            if self.stop_event.is_set() or not self.auto_running:
                return
            started = time.monotonic()
            log.info("[BOOTSTRAP] %d/%d %s", idx, total, symbol)
            if self.bootstrap_symbol(symbol):
                success += 1
            else:
                failure += 1
            if idx == 1 or idx % 25 == 0 or idx == total:
                log.info("[BOOTSTRAP] progress %d/%d | success=%d failure=%d", idx, total, success, failure)
                self._notify(
                    "📥 BOOTSTRAP PROGRESS\n"
                    f"{idx}/{total} pairs\n"
                    f"Success: {success} | Failed: {failure}"
                )
            elapsed = time.monotonic() - started
            sleep_for = max(0.0, LOAD_INTERVAL - elapsed)
            if idx < total and sleep_for:
                self.stop_event.wait(sleep_for)
        if self.stop_event.is_set() or not self.auto_running:
            return
        self.bootstrap_complete = success > 0
        log.info("[BOOTSTRAP] complete success=%d failure=%d/%d", success, failure, total)
        self._load_strategy()
        self._notify(
            "✅ DATA READY\n"
            f"Symbols: {total}\n"
            f"Loaded: {success}\n"
            f"Failed: {failure}\n"
            "Historical: 15M/700 + 5M/500 + 1M/500\n"
            f"Strategy: {'READY' if self.strategy else 'NOT READY'}\n"
            "Running initial strategy scan..."
        )
        if self.strategy and hasattr(self.strategy, "on_data_ready"):
            try:
                result = self.strategy.on_data_ready()
                message = str(result).strip() if result else "🔎 Initial scan selesai\nSetup baru: 0"
                self._notify(message)
                log.info("[STRATEGY] initial scan finished")
            except Exception as exc:
                log.exception("[STRATEGY] on_data_ready failed")
                self._notify(f"❌ INITIAL STRATEGY SCAN ERROR\n{type(exc).__name__}: {exc}")
        else:
            self._notify(
                "⚠️ INITIAL STRATEGY SCAN SKIPPED\n"
                f"Reason: {self.strategy_error or 'strategy.py tidak aktif'}"
            )
        self.start_websocket()

    # ---------------- strategy ----------------
    def _load_strategy(self) -> None:
        raw_path = str(STRATEGY_FILE).strip()
        path = raw_path if os.path.isabs(raw_path) else os.path.join(BASE_DIR, raw_path)
        path = os.path.abspath(path)
        self.strategy = None
        self.strategy_error = None
        if not os.path.isfile(path):
            self.strategy_error = f"strategy file not found: {path}"
            log.error("[STRATEGY] %s", self.strategy_error)
            self._notify(
                "❌ STRATEGY LOAD FAILED\n"
                f"File: {os.path.basename(path)}\n"
                f"Path: {path}"
            )
            return
        try:
            spec = importlib.util.spec_from_file_location(f"smc_strategy_runtime_{int(time.time() * 1000)}", path)
            if spec is None or spec.loader is None:
                raise ImportError("cannot create module spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.strategy = module
            self.strategy_error = None
            if hasattr(module, "initialize"):
                module.initialize(self.api, self.context)
            log.info("[STRATEGY] loaded %s", path)
            self._notify(f"✅ STRATEGY LOADED\n{os.path.basename(path)}")
        except Exception as exc:
            self.strategy = None
            self.strategy_error = f"{type(exc).__name__}: {exc}"
            log.exception("[STRATEGY] load failed")
            self._notify(
                "❌ STRATEGY LOAD ERROR\n"
                f"File: {os.path.basename(path)}\n"
                f"Error: {self.strategy_error}"
            )

    def reset_strategy(self) -> str:
        """Reload only strategy.py using the current local file.

        Data history and the live Bybit WebSocket stay untouched. This is intended
        for replacing a strategy while /auto is already running.
        """
        with self._run_lock:
            if not self.auto_running:
                return "ℹ️ /auto belum aktif. Jalankan /auto terlebih dahulu."

        old_strategy = self.strategy
        old_error = self.strategy_error

        if old_strategy and hasattr(old_strategy, "shutdown"):
            try:
                old_strategy.shutdown()
            except Exception:
                log.exception("[STRATEGY] old shutdown failed during reset")

        self.strategy = None
        self.strategy_error = None

        try:
            # Force a fresh module namespace every reset so Python never reuses
            # stale module state from the previous strategy.
            raw_path = str(STRATEGY_FILE).strip()
            path = raw_path if os.path.isabs(raw_path) else os.path.join(BASE_DIR, raw_path)
            path = os.path.abspath(path)

            if not os.path.isfile(path):
                raise FileNotFoundError(f"strategy file not found: {path}")

            module_name = f"smc_strategy_runtime_{int(time.time() * 1000)}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("cannot create strategy module spec")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self.strategy = module
            self.strategy_error = None

            if hasattr(module, "initialize"):
                module.initialize(self.api, self.context)

            log.info(
                "[STRATEGY RESET] loaded=%s old=%s",
                path,
                getattr(old_strategy, "__file__", None) if old_strategy else old_error,
            )
            self._notify(
                "🔄 STRATEGY RESET\n"
                f"Loaded: {os.path.basename(path)}\n"
                "Data/WebSocket tetap berjalan.\n"
                "Running fresh initial scan..."
            )

            if hasattr(module, "on_data_ready"):
                result = module.on_data_ready()
                message = str(result).strip() if result else "🔎 Initial scan selesai\nSetup baru: 0"
            else:
                message = "✅ Strategy reloaded\nTidak ada on_data_ready()."

            self._notify(message)
            log.info("[STRATEGY RESET] initial scan finished")
            return "✅ Strategy berhasil di-reset dan di-scan ulang dari data yang sudah ada."

        except Exception as exc:
            self.strategy = None
            self.strategy_error = f"{type(exc).__name__}: {exc}"
            log.exception("[STRATEGY RESET] failed")
            self._notify(
                "❌ STRATEGY RESET FAILED\n"
                f"{type(exc).__name__}: {exc}"
            )
            return f"❌ Strategy reset gagal: {type(exc).__name__}: {exc}"

    def add_data_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._callbacks.append(callback)

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        if self.strategy and hasattr(self.strategy, "on_market_event"):
            try:
                result = self.strategy.on_market_event(event)
                if result and self.chat_id is not None:
                    self._notify(str(result))
            except Exception:
                log.exception("[STRATEGY] on_market_event failed")
        for cb in tuple(self._callbacks):
            try:
                cb(event)
            except Exception:
                log.exception("data callback failed")

    # ---------------- websocket ----------------
    def _topics(self) -> list[str]:
        symbols = self.get_symbols()
        return [f"kline.{tf}.{s}" for tf in TF_CONFIG for s in symbols]

    @staticmethod
    def _topic_batches(topics: list[str], max_chars: int = 19000) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        current_chars = 2
        for topic in topics:
            add = len(topic) + (1 if current else 0)
            if current and current_chars + add > max_chars:
                batches.append(current)
                current = []
                current_chars = 2
            current.append(topic)
            current_chars += add
        if current:
            batches.append(current)
        return batches

    def _ws_open(self, ws: websocket.WebSocketApp) -> None:
        topics = self._topics()
        batches = self._topic_batches(topics)
        for idx, batch in enumerate(batches, 1):
            payload = {"op": "subscribe", "req_id": f"smc-{idx}", "args": batch}
            ws.send(json.dumps(payload, separators=(",", ":")))
            log.info("[WS] subscribe batch %d/%d | topics=%d", idx, len(batches), len(batch))
        log.info("[WS] connected | total_topics=%d | batches=%d", len(topics), len(batches))
        self._notify(
            "🟢 MARKET DATA LIVE\n"
            "Bybit WS\n"
            f"Symbols: {len(self.symbols)}\n"
            "Streams: 15M + 5M + 1M\n"
            f"Subscriptions: {len(topics)} in {len(batches)} batches"
        )

    def _ws_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        try:
            p = json.loads(message)
        except json.JSONDecodeError:
            return
        if p.get("op") in {"subscribe", "pong", "unsubscribe"}:
            return
        topic = str(p.get("topic") or "")
        parts = topic.split(".")
        if len(parts) != 3 or parts[0] != "kline":
            return
        tf = parts[1]
        for item in p.get("data") or []:
            symbol = str(item.get("symbol") or parts[2]).upper()
            try:
                c = Candle(int(item["start"]), float(item["open"]), float(item["high"]), float(item["low"]), float(item["close"]), float(item["volume"]), float(item.get("turnover") or 0), bool(item.get("confirm", False)))
            except (KeyError, TypeError, ValueError):
                log.exception("[WS] bad kline payload")
                continue
            self.store.upsert(symbol, tf, c)
            key = (symbol, tf)
            now = time.monotonic()
            if now - self._tick_logs.get(key, 0) >= LOG_EVERY_SYMBOL_TICK:
                self._tick_logs[key] = now
                log.info("[TICK] %s %s %s C=%s confirm=%s", symbol, tf, datetime_utc(c.timestamp), c.close, c.confirmed)
            self._dispatch_event({"type": "candle", "symbol": symbol, "timeframe": tf, "candle": c.as_dict()})

    def _ws_error(self, _ws: websocket.WebSocketApp, error: Any) -> None:
        log.warning("[WS] error=%s", error)

    def _ws_close(self, _ws: websocket.WebSocketApp, code: Any, msg: Any) -> None:
        log.warning("[WS] close code=%s msg=%s", code, msg)

    def _ping_loop(self, ws: websocket.WebSocketApp) -> None:
        while self.auto_running and not self.stop_event.is_set() and self.ws is ws:
            if self.stop_event.wait(WS_PING_INTERVAL):
                return
            try:
                ws.send(json.dumps({"op": "ping"}))
            except Exception:
                return

    def _ws_worker(self) -> None:
        backoff = 2
        while self.auto_running and not self.stop_event.is_set():
            try:
                ws = websocket.WebSocketApp(BYBIT_WS_URL, on_open=self._ws_open, on_message=self._ws_message, on_error=self._ws_error, on_close=self._ws_close)
                self.ws = ws
                threading.Thread(target=self._ping_loop, args=(ws,), name="ws-ping", daemon=True).start()
                log.info("[WS] connecting %s", BYBIT_WS_URL)
                ws.run_forever(ping_interval=None, ping_timeout=None, skip_utf8_validation=True)
            except Exception:
                log.exception("[WS] worker error")
            finally:
                self.ws = None
            if self.auto_running and not self.stop_event.is_set():
                log.warning("[WS] reconnect in %ss", backoff)
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, WS_RECONNECT_MAX)
        log.info("[WS] worker stopped")

    def start_websocket(self) -> None:
        if self.ws_thread and self.ws_thread.is_alive():
            return
        self.ws_thread = threading.Thread(target=self._ws_worker, name="bybit-market-ws", daemon=True)
        self.ws_thread.start()

    def stop_websocket(self) -> None:
        ws = self.ws
        self.ws = None
        if ws:
            try:
                ws.close()
            except Exception:
                log.exception("[WS] close failed")

    # ---------------- commands / lifecycle ----------------
    def start_auto(self) -> str:
        with self._run_lock:
            if self.auto_running:
                return "ℹ️ /auto sudah aktif."
            self.auto_running = True
            self.bootstrap_complete = False
        try:
            ip = self.public_ip()
        except Exception as exc:
            ip = f"unavailable ({exc})"
        log.info("[AUTO] start | server_ip=%s", ip)
        self._notify(f"🤖 AUTO MODE\nServer IP: {ip}\n\nDiscovering Bybit + Binance...")
        try:
            symbols = self.build_universe()
            if not symbols:
                raise RuntimeError("common symbol universe kosong")
        except Exception as exc:
            self.auto_running = False
            log.exception("[AUTO] discovery failed")
            return f"❌ /auto gagal: {exc}"
        self._notify(f"✅ Universe ready\nCommon selected: {len(symbols)}\nMax: {MAX_SYMBOLS}\n\nBootstrap 15M/5M/1M dimulai...")
        self.bootstrap_thread = threading.Thread(target=self.bootstrap_all, name="historical-bootstrap", daemon=True)
        self.bootstrap_thread.start()
        return f"🟢 /auto aktif — {len(symbols)} pair masuk data pipeline."

    def stop(self) -> None:
        with self._run_lock:
            self.auto_running = False
        self.stop_websocket()
        if self.strategy and hasattr(self.strategy, "shutdown"):
            try:
                self.strategy.shutdown()
            except Exception:
                log.exception("[STRATEGY] shutdown failed")
        log.info("[ENGINE] stopped")

    def get_symbols(self) -> list[str]:
        with self._symbol_lock:
            return list(self.symbols)

    def status(self) -> dict[str, Any]:
        return {
            "auto": self.auto_running,
            "bootstrap": self.bootstrap_complete,
            "symbols": len(self.symbols),
            "strategy": bool(self.strategy),
            "strategy_error": self.strategy_error,
            "ws": self.ws is not None,
        }


def datetime_utc(ms: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000))


ENGINE: DataEngine | None = None


def on_start(context: dict[str, Any]) -> None:
    global ENGINE
    ENGINE = DataEngine(context)
    try:
        ip = ENGINE.public_ip()
    except Exception as exc:
        ip = f"unavailable ({exc})"
    log.info("[START] start_v3 ready | server_ip=%s | base_dir=%s", ip, BASE_DIR)
    ENGINE._notify(
        "🟢 START.PY V3 READY\n"
        f"Server IP: {ip}\n"
        f"File: {os.path.basename(__file__)}\n"
        f"Strategy target: {STRATEGY_FILE}\n\n"
        "/auto → mulai market scanner"
    )


def on_stop(context: dict[str, Any]) -> None:
    global ENGINE
    if ENGINE:
        ENGINE.stop()
    ENGINE = None


def _help() -> str:
    return (
        "📡 DATA ENGINE V3\n\n"
        "/IP — public server IP\n"
        "/auto — start discovery + bootstrap + websocket\n"
        "/status — data/strategy status\n"
        "/symbols — jumlah dan daftar pair\n"
        "/candles BTCUSDT 15 20 — lihat candle\n"
        "/price BTCUSDT — harga terakhir\n"
        "/reset — reload strategy.py + scan ulang tanpa reset data/WS\n"
    )


def handle_update(update: dict[str, Any], context: dict[str, Any]) -> str | None:
    engine = ENGINE
    if engine is None:
        return "❌ Data engine belum aktif."
    msg = update.get("message") or {}
    text = str(msg.get("text") or msg.get("caption") or "").strip()
    if not text:
        return None
    cmd, *args = text.split()
    cmd = cmd.split("@", 1)[0].lower()
    if cmd == "/ip":
        try:
            return f"🌐 Server public IP\n{engine.public_ip()}"
        except Exception as exc:
            return f"❌ IP gagal diambil: {exc}"
    if cmd in {"/help", "/start"}:
        return _help()
    if cmd == "/auto":
        return engine.start_auto()
    if cmd == "/status":
        s = engine.status()
        return (
            "📊 STATUS\n"
            f"AUTO: {s['auto']}\n"
            f"BOOTSTRAP: {s['bootstrap']}\n"
            f"SYMBOLS: {s['symbols']}\n"
            f"WS: {s['ws']}\n"
            f"STRATEGY: {s['strategy']}\n"
            f"STRATEGY ERROR: {s['strategy_error'] or '-'}"
        )
    if cmd == "/symbols":
        syms = engine.get_symbols()
        preview = ", ".join(syms[:80])
        return f"🪙 Symbols: {len(syms)}\n{preview}" if preview else "🪙 Symbols: 0"
    if cmd == "/candles":
        if len(args) < 2:
            return "Format: /candles BTCUSDT 15 [limit]"
        symbol, tf = args[0].upper(), args[1]
        limit = int(args[2]) if len(args) > 2 else 10
        candles = engine.api.get_candles(symbol, tf, min(limit, 50))
        if not candles:
            return f"❌ Tidak ada data {symbol} TF {tf}."
        lines = [f"{c['timestamp']} O={c['open']} H={c['high']} L={c['low']} C={c['close']} V={c['volume']}" for c in candles]
        return f"📈 {symbol} {tf}M\n" + "\n".join(lines)
    if cmd == "/price":
        if not args:
            return "Format: /price BTCUSDT"
        p = engine.api.get_price(args[0].upper())
        return f"💵 {args[0].upper()} = {p}" if p is not None else "❌ Harga belum tersedia."
    if cmd == "/reset":
        return engine.reset_strategy()
    if engine.strategy and hasattr(engine.strategy, "handle_command"):
        try:
            result = engine.strategy.handle_command(text)
            return None if result is None else str(result)
        except Exception:
            log.exception("[STRATEGY] command failed")
            return "❌ Strategy command error. Lihat terminal."
    return None
