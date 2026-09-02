from __future__ import annotations

"""

# VERSION: 14.0
SMCAutoTrade start_v14.py

Data + execution orchestration layer.

- Bybit REST/WS remain the market-data source.
- strategy.py remains strategy-only and emits confirmed signal objects.
- SIMULATION is the default execution mode.
- REAL mode uses Binance USDⓈ-M Futures REST for entry and conditional exits.
- Quantity is auto-adjusted to valid Binance step/minimum filters, constrained
  to +/- 50% of configured per-trade margin.
- /reset reloads strategy only; data, trades and bans remain intact.
- Bans persist for 8 hours after every closed bot-managed trade.
"""

import hashlib
import hmac
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import urllib.parse
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import websocket

VERSION = "14.0"

BYBIT_BASE_URL = (os.getenv("BYBIT_BASE_URL") or "https://api.bybit.com").rstrip("/")
BYBIT_WS_URL = (os.getenv("BYBIT_WS_URL") or "wss://stream.bybit.com/v5/public/linear").strip()
BINANCE_BASE_URL = (os.getenv("BINANCE_BASE_URL") or "https://fapi.binance.com").rstrip("/")
BINANCE_WS_BASE = (os.getenv("BINANCE_WS_BASE") or "wss://fstream.binance.com/ws").rstrip("/")
IP_URL = (os.getenv("IP_URL") or "https://api.ipify.org").strip()

TF_CONFIG = {"15": 700, "5": 500, "1": 500}
LOAD_INTERVAL = max(0.0, float(os.getenv("DATA_LOAD_INTERVAL", "0.0")))
RETENTION_EXTRA = max(50, int(os.getenv("DATA_RETENTION_EXTRA", "50")))
REQUEST_TIMEOUT = max(5, int(os.getenv("REQUEST_TIMEOUT", "20")))
BINANCE_EXTRA_COOLDOWN = max(0, int(os.getenv("BINANCE_EXTRA_COOLDOWN", "60")))
BINANCE_RECOVERY_POLL = max(5, int(os.getenv("BINANCE_RECOVERY_POLL", "30")))
WS_PING_INTERVAL = max(5, int(os.getenv("WS_PING_INTERVAL", "20")))
WS_RECONNECT_MAX = max(5, int(os.getenv("WS_RECONNECT_MAX", "30")))
LOG_EVERY_SYMBOL_TICK = max(2, int(os.getenv("LOG_EVERY_SYMBOL_TICK", "15")))
WS_MAX_ARG_CHARS = max(5000, int(os.getenv("WS_MAX_ARG_CHARS", "18000")))
WS_RECONNECT_JITTER = max(0.0, float(os.getenv("WS_RECONNECT_JITTER", "1.5")))
WS_NOTIFY_BATCH_EVERY = max(1, int(os.getenv("WS_NOTIFY_BATCH_EVERY", "5")))
STRATEGY_FILE = (os.getenv("STRATEGY_FILE") or "strategy.py").strip()
LEARN_FILE = (os.getenv("LEARN_FILE") or "learn.py").strip()
BASE_DIR = Path(__file__).resolve().parent

BINANCE_API_KEY = (os.getenv("BINANCE_API_KEY") or "").strip()
BINANCE_API_SECRET = (os.getenv("BINANCE_API_SECRET") or "").strip()

DEFAULT_MARGIN = max(0.0, float(os.getenv("TRADE_MARGIN", "10")))
DEFAULT_LEVERAGE = max(1, int(os.getenv("TRADE_LEVERAGE", "10")))
DEFAULT_MAX_ACTIVE = max(0, int(os.getenv("TRADE_MAX_ACTIVE", "5")))
BAN_HOURS = max(0.25, float(os.getenv("TRADE_BAN_HOURS", "8")))
AUTO_QTY_RANGE = 0.50
STATE_FILE = Path(os.getenv("TRADE_STATE_FILE", str(BASE_DIR / "trade_state_v6.json"))).resolve()
STATE_LOCK = threading.RLock()

BINANCE_META_CACHE_FILE = Path(
    os.getenv("BINANCE_META_CACHE_FILE", str(BASE_DIR / "binance_exchange_cache_v7.json"))
).resolve()
BINANCE_META_TTL = max(300, int(os.getenv("BINANCE_META_TTL", str(6 * 3600))))
BINANCE_WEIGHT_LIMIT = max(100, int(os.getenv("BINANCE_WEIGHT_LIMIT", "2400")))
BINANCE_WEIGHT_SOFT_LIMIT = max(100, min(BINANCE_WEIGHT_LIMIT - 1, int(os.getenv("BINANCE_WEIGHT_SOFT_LIMIT", "1800"))))
BINANCE_REQUEST_MIN_GAP = max(0.0, float(os.getenv("BINANCE_REQUEST_MIN_GAP", "0.08")))
BINANCE_BLACKOUT_EXTRA = max(0, int(os.getenv("BINANCE_BLACKOUT_EXTRA", "60")))
BINANCE_429_BASE_BACKOFF = max(5, int(os.getenv("BINANCE_429_BASE_BACKOFF", "30")))

MAX_WORKERS = max(1, min(5, int(os.getenv("MAX_WORKERS", "5"))))
BYBIT_REQUEST_MIN_GAP = max(0.05, float(os.getenv("BYBIT_REQUEST_MIN_GAP", "0.12")))
BYBIT_HTTP_WINDOW = max(1.0, float(os.getenv("BYBIT_HTTP_WINDOW", "5")))
BYBIT_HTTP_MAX_REQUESTS = max(1, int(os.getenv("BYBIT_HTTP_MAX_REQUESTS", "500")))
BYBIT_429_BACKOFF = max(1.0, float(os.getenv("BYBIT_429_BACKOFF", "5")))
BOOTSTRAP_BATCH_SIZE = max(MAX_WORKERS, int(os.getenv("BOOTSTRAP_BATCH_SIZE", "50")))


logging.basicConfig(
    level=(os.getenv("LOG_LEVEL") or "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("start-v13")


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
        return asdict(self)


@dataclass
class Trade:
    id: str
    signal_id: str
    symbol: str
    direction: str
    order_type: str
    status: str
    mode: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confirmation_price: float | None
    quantity: float
    margin_target: float
    margin_actual: float
    leverage: int
    created_ts: int
    opened_ts: int | None = None
    closed_ts: int | None = None
    actual_entry: float | None = None
    actual_exit: float | None = None
    pnl: float | None = None
    r_multiple: float | None = None
    outcome: str | None = None
    entry_order_id: str | None = None
    sl_order_id: str | None = None
    tp_order_id: str | None = None
    sl_algo_id: str | None = None
    tp_algo_id: str | None = None
    error: str | None = None


class DataStore:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.data: dict[str, dict[str, list[Candle]]] = {}
        self.prices: dict[str, float] = {}
        self.price_ts: dict[str, int] = {}

    def set_history(self, symbol: str, tf: str, candles: list[Candle]) -> None:
        with self.lock:
            keep = TF_CONFIG[tf] + RETENTION_EXTRA
            ordered = sorted(candles, key=lambda c: c.timestamp)
            self.data.setdefault(symbol, {})[tf] = ordered[-keep:]
            if candles:
                last = ordered[-1]
                self.prices[symbol] = last.close
                self.price_ts[symbol] = last.timestamp

    def upsert(self, symbol: str, tf: str, candle: Candle) -> None:
        with self.lock:
            s = self.data.setdefault(symbol, {}).setdefault(tf, [])
            if s and candle.timestamp == s[-1].timestamp:
                s[-1] = candle
            elif not s or candle.timestamp > s[-1].timestamp:
                s.append(candle)
            else:
                replaced = False
                for i, old in enumerate(s):
                    if old.timestamp == candle.timestamp:
                        s[i] = candle; replaced = True; break
                    if old.timestamp > candle.timestamp:
                        s.insert(i, candle); replaced = True; break
                if not replaced:
                    s.append(candle)
            keep = TF_CONFIG.get(tf, 1000) + RETENTION_EXTRA
            if len(s) > keep:
                del s[: len(s) - keep]
            if candle.timestamp >= self.price_ts.get(symbol, -1):
                self.prices[symbol] = candle.close
                self.price_ts[symbol] = candle.timestamp

    def get(self, symbol: str, tf: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self.lock:
            rows = list(self.data.get(symbol.upper(), {}).get(tf, []))
        if limit is not None:
            rows = rows[-max(1, int(limit)):]
        return [c.as_dict() for c in rows]

    def latest(self, symbol: str, tf: str) -> dict[str, Any] | None:
        with self.lock:
            rows = self.data.get(symbol.upper(), {}).get(tf, [])
            return rows[-1].as_dict() if rows else None

    def price(self, symbol: str) -> float | None:
        with self.lock:
            return self.prices.get(symbol.upper())

    def remove_symbol(self, symbol: str) -> None:
        symbol = symbol.upper()
        with self.lock:
            self.data.pop(symbol, None)
            self.prices.pop(symbol, None)
            self.price_ts.pop(symbol, None)


class DataAPI:
    def __init__(self, engine: "DataEngine") -> None:
        self.engine = engine

    def get_symbols(self) -> list[str]:
        return self.engine.get_symbols()

    def get_candles(self, symbol: str, timeframe: str = "15", limit: int | None = None) -> list[dict[str, Any]]:
        return self.engine.store.get(symbol, str(timeframe), limit)

    def get_latest_candle(self, symbol: str, timeframe: str = "15") -> dict[str, Any] | None:
        return self.engine.store.latest(symbol, str(timeframe))

    def get_price(self, symbol: str) -> float | None:
        return self.engine.store.price(symbol)

    def get_snapshot(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol.upper(), "price": self.get_price(symbol), "timeframes": {tf: self.get_candles(symbol, tf) for tf in TF_CONFIG}}

    def is_bootstrap_complete(self) -> bool:
        return self.engine.bootstrap_complete

    def is_symbol_banned(self, symbol: str) -> bool:
        return self.engine.trade_manager.is_banned(symbol)

    def get_global_context(self) -> dict[str, Any]:
        if self.engine.learning and hasattr(self.engine.learning, "get_global_context"):
            try:
                return dict(self.engine.learning.get_global_context())
            except Exception:
                log.exception("[LEARN] global context read failed")
        return {}

    def get_learning_snapshot(self) -> dict[str, Any]:
        if self.engine.learning and hasattr(self.engine.learning, "status"):
            try:
                return {"learning": self.engine.learning.status()}
            except Exception:
                pass
        return {}

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.engine.add_data_callback(callback)


class BinanceError(RuntimeError):
    pass


class BinanceRateLimited(BinanceError):
    def __init__(self, status: int, retry_after_seconds: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after_seconds = retry_after_seconds


class BinanceCircuitOpen(BinanceError):
    pass


class BinanceRequestManager:
    """Central Binance request gate.

    All Binance REST traffic passes here. The manager tracks Binance's IP weight
    headers, enforces a local request gap, and opens a circuit after 429/418.
    """
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_request = 0.0
        self.used_weight_1m = 0
        self.used_weight_1s = 0
        self.blackout_until = 0.0
        self.last_status: int | None = None
        self.last_retry_after = 0
        self.backoff_count = 0

    @property
    def blocked(self) -> bool:
        return time.time() < self.blackout_until

    @property
    def remaining_seconds(self) -> int:
        return max(0, int(self.blackout_until - time.time()))

    def status_text(self) -> str:
        if self.blocked:
            return f"BINANCE: BLACKOUT ({self.remaining_seconds}s remaining)"
        if self.used_weight_1m >= BINANCE_WEIGHT_SOFT_LIMIT:
            return f"BINANCE: THROTTLED ({self.used_weight_1m}/{BINANCE_WEIGHT_LIMIT} weight/1m)"
        return f"BINANCE: OK ({self.used_weight_1m}/{BINANCE_WEIGHT_LIMIT} weight/1m)"

    def _parse_retry_after(self, response: requests.Response) -> int:
        raw = response.headers.get("Retry-After")
        if raw is not None:
            try:
                return max(1, int(float(raw)))
            except ValueError:
                pass
        # Binance may omit Retry-After on some responses. Use conservative
        # exponential backoff instead of hammering the IP.
        self.backoff_count = min(self.backoff_count + 1, 6)
        return BINANCE_429_BASE_BACKOFF * (2 ** (self.backoff_count - 1))

    def open_blackout(self, seconds: int, status: int) -> int:
        seconds = max(1, int(seconds))
        until = time.time() + seconds + BINANCE_BLACKOUT_EXTRA
        with self.lock:
            self.blackout_until = max(self.blackout_until, until)
            self.last_status = status
            self.last_retry_after = seconds
        return max(0, int(self.blackout_until - time.time()))

    def before_request(self) -> None:
        with self.lock:
            remaining = self.blackout_until - time.time()
            if remaining > 0:
                raise BinanceCircuitOpen(
                    f"Binance temporarily disabled; {int(remaining)}s remaining after HTTP {self.last_status}"
                )
            gap = time.monotonic() - self.last_request
            if gap < BINANCE_REQUEST_MIN_GAP:
                time.sleep(BINANCE_REQUEST_MIN_GAP - gap)
            # Once the soft threshold is reached, pause briefly rather than
            # continue producing a burst that can push the IP into 429.
            if self.used_weight_1m >= BINANCE_WEIGHT_SOFT_LIMIT:
                sleep_for = min(2.0, max(0.2, (self.used_weight_1m - BINANCE_WEIGHT_SOFT_LIMIT + 1) * 0.002))
                time.sleep(sleep_for)
            self.last_request = time.monotonic()

    def after_response(self, response: requests.Response) -> None:
        with self.lock:
            for key, attr in (
                ("X-MBX-USED-WEIGHT-1M", "used_weight_1m"),
                ("X-MBX-USED-WEIGHT-1S", "used_weight_1s"),
            ):
                raw = response.headers.get(key)
                if raw is not None:
                    try:
                        setattr(self, attr, int(raw))
                    except ValueError:
                        pass
            if response.status_code < 429:
                self.backoff_count = 0
            self.last_status = response.status_code

    def request(self, method: str, url: str, *, params=None, headers=None) -> requests.Response:
        self.before_request()
        try:
            response = requests.request(
                method,
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise BinanceError(f"network error: {exc}") from exc
        self.after_response(response)
        if response.status_code in (418, 429):
            retry = self._parse_retry_after(response)
            blocked_for = self.open_blackout(retry, response.status_code)
            log.error(
                "[BINANCE] HTTP %s rate-limit circuit OPEN | retry-after=%ss | blackout=%ss | used_1m=%s",
                response.status_code, retry, blocked_for, self.used_weight_1m,
            )
            raise BinanceRateLimited(
                response.status_code,
                blocked_for,
                f"Binance HTTP {response.status_code}; blackout {blocked_for}s",
            )
        return response


BINANCE_GATE = BinanceRequestManager()


class BinanceClient:
    def __init__(self) -> None:
        self.key = BINANCE_API_KEY
        self.secret = BINANCE_API_SECRET
        self.base_url = BINANCE_BASE_URL
        self.meta: dict[str, dict[str, Any]] = {}
        self.meta_loaded = False
        self.meta_loaded_ts = 0.0
        self._load_meta_cache()

    @property
    def configured(self) -> bool:
        return bool(self.key and self.secret)

    @property
    def available(self) -> bool:
        return not BINANCE_GATE.blocked

    def _signed(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = True) -> Any:
        params = dict(params or {})
        if signed:
            params.setdefault("timestamp", int(time.time() * 1000))
            params.setdefault("recvWindow", 5000)
            query = urllib.parse.urlencode([(k, v) for k, v in params.items() if v is not None], doseq=True)
            signature = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            params["signature"] = signature
        headers = {"X-MBX-APIKEY": self.key} if self.key else {}
        url = self.base_url + path
        response = BINANCE_GATE.request(method, url, params=params, headers=headers)
        try:
            body = response.json()
        except ValueError:
            body = response.text
        if response.status_code >= 400:
            raise BinanceError(f"HTTP {response.status_code}: {body}")
        if isinstance(body, dict) and "code" in body and isinstance(body.get("code"), int) and body["code"] < 0:
            raise BinanceError(str(body))
        return body

    def _load_meta_cache(self) -> None:
        if not BINANCE_META_CACHE_FILE.exists():
            return
        try:
            raw = json.loads(BINANCE_META_CACHE_FILE.read_text(encoding="utf-8"))
            symbols = raw.get("symbols") or {}
            if isinstance(symbols, dict) and symbols:
                self.meta = symbols
                self.meta_loaded = True
                self.meta_loaded_ts = float(raw.get("saved_at", 0.0))
                log.info(
                    "[BINANCE] exchangeInfo cache loaded symbols=%d age=%ss",
                    len(self.meta), max(0, int(time.time() - self.meta_loaded_ts))
                )
        except Exception:
            log.exception("[BINANCE] exchangeInfo cache load failed")

    def _save_meta_cache(self) -> None:
        try:
            payload = {"saved_at": time.time(), "symbols": self.meta}
            tmp = BINANCE_META_CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(BINANCE_META_CACHE_FILE)
        except Exception:
            log.exception("[BINANCE] exchangeInfo cache save failed")

    def exchange_info(self, force: bool = False) -> dict[str, Any]:
        fresh = self.meta_loaded and (time.time() - self.meta_loaded_ts) < BINANCE_META_TTL
        if fresh and not force:
            return {"symbols": list(self.meta.values())}

        if BINANCE_GATE.blocked:
            if self.meta_loaded:
                log.warning("[BINANCE] using stale exchangeInfo cache while circuit is open")
                return {"symbols": list(self.meta.values()), "stale": True}
            raise BinanceCircuitOpen(
                f"Binance unavailable and no exchangeInfo cache exists; {BINANCE_GATE.remaining_seconds}s remaining"
            )

        data = self._signed("GET", "/fapi/v1/exchangeInfo", signed=False)
        self.meta = {
            str(s["symbol"]).upper(): s
            for s in data.get("symbols", [])
            if s.get("status") == "TRADING"
        }
        self.meta_loaded = True
        self.meta_loaded_ts = time.time()
        self._save_meta_cache()
        log.info(
            "[BINANCE] exchangeInfo refreshed symbols=%d ttl=%ss",
            len(self.meta), BINANCE_META_TTL
        )
        return {"symbols": list(self.meta.values())}

    def symbol_meta(self, symbol: str) -> dict[str, Any]:
        self.exchange_info()
        s = self.meta.get(symbol.upper())
        if not s:
            raise BinanceError(f"Binance symbol {symbol} not found in exchangeInfo cache")
        return s

    def account(self) -> dict[str, Any]:
        if not self.configured:
            raise BinanceError("BINANCE_API_KEY / BINANCE_API_SECRET belum diset")
        return self._signed("GET", "/fapi/v2/account", signed=True)

    def position_mode(self) -> bool:
        if not self.configured:
            raise BinanceError("Binance credentials missing")
        data = self._signed("GET", "/fapi/v1/positionSide/dual", signed=True)
        return bool(data.get("dualSidePosition"))

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return self._signed("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": int(leverage)}, signed=True)

    def new_order(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._signed("POST", "/fapi/v1/order", params, signed=True)

    def query_order(self, symbol: str, order_id: str | int | None = None, client_order_id: str | None = None) -> dict[str, Any]:
        p: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            p["orderId"] = order_id
        if client_order_id:
            p["origClientOrderId"] = client_order_id
        return self._signed("GET", "/fapi/v1/order", p, signed=True)

    def cancel_order(self, symbol: str, order_id: str | int | None = None, client_order_id: str | None = None) -> dict[str, Any]:
        p: dict[str, Any] = {"symbol": symbol}
        if order_id is not None: p["orderId"] = order_id
        if client_order_id: p["origClientOrderId"] = client_order_id
        return self._signed("DELETE", "/fapi/v1/order", p, signed=True)

    def new_algo_order(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._signed("POST", "/fapi/v1/algoOrder", params, signed=True)

    def query_algo_order(self, symbol: str, algo_id: str | int | None = None, client_algo_id: str | None = None) -> dict[str, Any]:
        p: dict[str, Any] = {"symbol": symbol}
        if algo_id is not None: p["algoId"] = algo_id
        if client_algo_id: p["clientAlgoId"] = client_algo_id
        return self._signed("GET", "/fapi/v1/algoOrder", p, signed=True)

    def cancel_algo_order(self, symbol: str, algo_id: str | int | None = None, client_algo_id: str | None = None) -> dict[str, Any]:
        p: dict[str, Any] = {"symbol": symbol}
        if algo_id is not None: p["algoId"] = algo_id
        if client_algo_id: p["clientAlgoId"] = client_algo_id
        return self._signed("DELETE", "/fapi/v1/algoOrder", p, signed=True)

    def user_trades(self, symbol: str, start_time: int | None = None) -> list[dict[str, Any]]:
        p: dict[str, Any] = {"symbol": symbol, "limit": 1000}
        if start_time is not None: p["startTime"] = max(0, int(start_time))
        return self._signed("GET", "/fapi/v1/userTrades", p, signed=True)


class TradeManager:
    def __init__(self, engine: "DataEngine", context: dict[str, Any]) -> None:
        self.engine = engine
        self.send_message = context["send_message"]
        self.chat_id = context.get("chat_id")
        self.lock = threading.RLock()
        self.binance = BinanceClient()
        self.mode = "OFF"  # hard default after process restart
        self.margin = DEFAULT_MARGIN
        self.leverage = DEFAULT_LEVERAGE
        self.max_active = DEFAULT_MAX_ACTIVE
        self.banned: dict[str, int] = {}
        self.trades: dict[str, Trade] = {}
        self.closed: list[dict[str, Any]] = []
        self._last_real_poll = 0.0
        self._load_state()

    def binance_status(self) -> str:
        return BINANCE_GATE.status_text()

    def _notify(self, text: str) -> None:
        if self.chat_id is None:
            return
        try:
            self.send_message(self.chat_id, text)
        except Exception:
            log.exception("[TG] trade notification failed")

    def _save_state(self) -> None:
        with self.lock:
            payload = {
                "banned": self.banned,
                "trades": {k: asdict(v) for k, v in self.trades.items()},
                "closed": self.closed[-5000:],
            }
        tmp = STATE_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(STATE_FILE)
        except Exception:
            log.exception("[STATE] save failed")

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            self.banned = {str(k).upper(): int(v) for k, v in (data.get("banned") or {}).items()}
            for k, raw in (data.get("trades") or {}).items():
                self.trades[k] = Trade(**raw)
            self.closed = list(data.get("closed") or [])
            self._purge_bans()
            log.info("[STATE] loaded trades=%d closed=%d bans=%d", len(self.trades), len(self.closed), len(self.banned))
        except Exception:
            log.exception("[STATE] load failed; starting clean runtime state")

    def _purge_bans(self) -> None:
        now = int(time.time() * 1000)
        expired = [s for s, until in self.banned.items() if until <= now]
        for s in expired:
            self.banned.pop(s, None)

    def is_banned(self, symbol: str) -> bool:
        with self.lock:
            self._purge_bans()
            return self.banned.get(symbol.upper(), 0) > int(time.time() * 1000)

    def ban(self, symbol: str) -> None:
        until = int((time.time() + BAN_HOURS * 3600) * 1000)
        self.banned[symbol.upper()] = until
        self._save_state()

    def active_count(self) -> int:
        return sum(1 for t in self.trades.values() if t.status in {"WAITING_ENTRY", "ENTRY_SUBMITTED", "OPEN", "PROTECTION_PENDING"})

    def _symbol_filters(self, symbol: str, order_type: str) -> dict[str, Any]:
        s = self.binance.symbol_meta(symbol)
        filters = {f.get("filterType"): f for f in s.get("filters", [])}
        lot = filters.get("MARKET_LOT_SIZE") if order_type == "MARKET" else filters.get("LOT_SIZE")
        lot = lot or filters.get("LOT_SIZE") or {}
        price = filters.get("PRICE_FILTER") or {}
        notion = filters.get("MIN_NOTIONAL") or {}
        return {
            "tick_size": Decimal(str(price.get("tickSize", "0.00000001"))),
            "step_size": Decimal(str(lot.get("stepSize", "0.00000001"))),
            "min_qty": Decimal(str(lot.get("minQty", "0"))),
            "max_qty": Decimal(str(lot.get("maxQty", "999999999"))),
            "min_notional": Decimal(str(notion.get("notional", notion.get("minNotional", "0")))),
        }

    @staticmethod
    def _round_step(value: Decimal, step: Decimal, rounding=ROUND_DOWN) -> Decimal:
        if step <= 0:
            return value
        units = (value / step).quantize(Decimal("1"), rounding=rounding)
        return units * step

    def normalize_price(self, symbol: str, value: float, direction: str | None = None) -> float:
        f = self._symbol_filters(symbol, "LIMIT")
        step = f["tick_size"]
        return float(self._round_step(Decimal(str(value)), step, ROUND_DOWN))

    def calculate_quantity(self, symbol: str, entry_price: float, order_type: str) -> tuple[float, float, float]:
        if self.margin <= 0:
            raise BinanceError("margin harus > 0")
        if entry_price <= 0:
            raise BinanceError("entry price tidak valid")
        f = self._symbol_filters(symbol, order_type)
        target_notional = Decimal(str(self.margin)) * Decimal(str(self.leverage))
        raw_qty = target_notional / Decimal(str(entry_price))
        step = f["step_size"]
        candidates: list[Decimal] = []
        for rounding in (ROUND_DOWN, ROUND_UP):
            q = self._round_step(raw_qty, step, rounding)
            if q > 0:
                candidates.append(q)
        if f["min_qty"] > 0:
            candidates.append(f["min_qty"])
        # Ensure minimum notional is respected, then test +/-50% margin band.
        if f["min_notional"] > 0:
            q_min_notional = f["min_notional"] / Decimal(str(entry_price))
            candidates.append(self._round_step(q_min_notional, step, ROUND_UP))
        lo_margin = Decimal(str(self.margin * (1 - AUTO_QTY_RANGE)))
        hi_margin = Decimal(str(self.margin * (1 + AUTO_QTY_RANGE)))
        valid: list[tuple[Decimal, Decimal]] = []
        seen = set()
        for q in candidates:
            q = max(f["min_qty"], min(q, f["max_qty"]))
            if q in seen or q <= 0:
                continue
            seen.add(q)
            actual_margin = (q * Decimal(str(entry_price))) / Decimal(str(self.leverage))
            if lo_margin <= actual_margin <= hi_margin:
                valid.append((q, actual_margin))
        if not valid:
            raise BinanceError(
                f"no valid quantity within ±50% margin; target={self.margin:.8f} "
                f"allowed=[{float(lo_margin):.8f},{float(hi_margin):.8f}] "
                f"step={step} minQty={f['min_qty']} minNotional={f['min_notional']}"
            )
        q, actual_margin = min(valid, key=lambda x: abs(float(x[1]) - self.margin))
        return float(q), float(actual_margin), float(q * Decimal(str(entry_price)))

    def _trade_from_signal(self, signal: dict[str, Any], mode: str) -> Trade:
        symbol = str(signal["symbol"]).upper()
        direction = str(signal["direction"]).upper()
        order_type = str(signal.get("entry_type") or "LIMIT").upper()
        entry = float(signal["entry_price"])
        qty, actual_margin, _ = self.calculate_quantity(symbol, entry, order_type)
        return Trade(
            id=f"T7-{symbol}-{uuid.uuid4().hex[:10]}",
            signal_id=str(signal["id"]),
            symbol=symbol,
            direction=direction,
            order_type=order_type,
            status="WAITING_ENTRY",
            mode=mode,
            entry_price=entry,
            stop_loss=float(signal["stop_loss"]),
            take_profit=float(signal["take_profit"]),
            confirmation_price=(float(signal["confirmation_price"]) if signal.get("confirmation_price") is not None else None),
            quantity=qty,
            margin_target=self.margin,
            margin_actual=actual_margin,
            leverage=self.leverage,
            created_ts=int(time.time() * 1000),
        )

    def accept_signal(self, signal: dict[str, Any]) -> str | None:
        symbol = str(signal.get("symbol") or "").upper()
        if not symbol:
            return None
        with self.lock:
            self._purge_bans()
            if self.is_banned(symbol):
                log.info("[TRADE] blocked banned symbol=%s", symbol)
                return None
            if self.max_active == 0 or self.active_count() >= self.max_active:
                log.warning("[TRADE] max active reached current=%d max=%d signal=%s", self.active_count(), self.max_active, signal.get("id"))
                self._notify(
                    "🚫 ORDER BLOCKED\n\n"
                    f"{symbol} {signal.get('direction')}\n"
                    f"Max active orders/positions: {self.max_active}\n"
                    f"Current: {self.active_count()}"
                )
                return None
            try:
                trade = self._trade_from_signal(signal, self.mode)
            except Exception as exc:
                log.exception("[TRADE] quantity validation failed for %s", symbol)
                self._notify(f"❌ TRADE REJECTED\n{symbol}\nReason: {type(exc).__name__}: {exc}")
                return None
            self.trades[trade.id] = trade
            self._save_state()
            if self.mode == "OFF":
                if trade.order_type == "MARKET":
                    self._sim_open(trade, self.engine.store.price(symbol) or trade.entry_price)
                else:
                    self._notify(_format_trade(trade, "🟡 SIMULATION ORDER CREATED"))
                    log.info("[SIM] pending %s %s entry=%s qty=%s margin=%s", symbol, trade.direction, trade.entry_price, trade.quantity, trade.margin_actual)
            else:
                self._submit_real_entry(trade)
            return trade.id

    def _sim_open(self, trade: Trade, price: float) -> None:
        trade.status = "OPEN"
        trade.opened_ts = int(time.time() * 1000)
        trade.actual_entry = float(price)
        self._notify(_format_trade(trade, "🟢 SIMULATION TRADE OPEN"))
        log.info("[SIM] OPEN %s %s @ %.8f qty=%s", trade.symbol, trade.direction, price, trade.quantity)
        self._save_state()
        self.engine._emit_learning("on_trade_event", {"trade_id": trade.id, "signal_id": trade.signal_id, "symbol": trade.symbol, "direction": trade.direction, "status": trade.status, "mode": trade.mode, "ts": int(time.time()*1000), "pnl": trade.pnl, "r_multiple": trade.r_multiple})

    def _submit_real_entry(self, trade: Trade) -> None:
        try:
            if not self.binance.configured:
                raise BinanceError("BINANCE_API_KEY / BINANCE_API_SECRET belum diset")
            if self.binance.position_mode():
                raise BinanceError("Binance account berada di Hedge Mode; v5 live execution memakai One-way/BOTH")
            self.binance.set_leverage(trade.symbol, trade.leverage)
            side = "BUY" if trade.direction == "LONG" else "SELL"
            params: dict[str, Any] = {
                "symbol": trade.symbol,
                "side": side,
                "positionSide": "BOTH",
                "type": trade.order_type,
                "quantity": self._format_decimal(trade.quantity),
                "newOrderRespType": "ACK",
                "newClientOrderId": f"smc5_{trade.id[-16:]}",
            }
            if trade.order_type == "LIMIT":
                trade.entry_price = self.normalize_price(trade.symbol, trade.entry_price)
                params.update({"price": self._format_decimal(trade.entry_price), "timeInForce": "GTC"})
            response = self.binance.new_order(params)
            trade.entry_order_id = str(response.get("orderId")) if response.get("orderId") is not None else None
            trade.status = "ENTRY_SUBMITTED" if trade.order_type == "MARKET" else "WAITING_ENTRY"
            self._notify(_format_trade(trade, "📤 REAL ENTRY ORDER SENT"))
            log.info("[REAL] entry sent %s %s type=%s orderId=%s", trade.symbol, trade.direction, trade.order_type, trade.entry_order_id)
            self._poll_real_trade(trade, force=True)
        except BinanceRateLimited as exc:
            trade.status = "ERROR"
            trade.error = str(exc)
            self._notify(
                "🚨 BINANCE RATE LIMIT / BLACKOUT\n\n"
                f"HTTP {exc.status}\n"
                f"Binance disabled for ~{BINANCE_GATE.remaining_seconds}s.\n"
                "Bybit/strategy data tetap berjalan."
            )
        except Exception as exc:
            trade.status = "ERROR"
            trade.error = f"{type(exc).__name__}: {exc}"
            log.exception("[REAL] entry failed %s", trade.symbol)
            self._notify(f"❌ REAL ORDER FAILED\n{trade.symbol} {trade.direction}\nReason: {trade.error}")
        finally:
            self._save_state()

    @staticmethod
    def _format_decimal(value: float | Decimal) -> str:
        d = value if isinstance(value, Decimal) else Decimal(str(value))
        return format(d.normalize(), "f")

    def _price_touched(self, trade: Trade, price: float) -> bool:
        if trade.direction == "LONG":
            return price <= trade.entry_price
        return price >= trade.entry_price

    def on_market_price(self, symbol: str, price: float, candle: dict[str, Any] | None = None) -> None:
        with self.lock:
            now = int(time.time() * 1000)
            high = float((candle or {}).get("high", price))
            low = float((candle or {}).get("low", price))
            for trade in list(self.trades.values()):
                if trade.symbol != symbol.upper():
                    continue
                if trade.mode == "OFF":
                    if trade.status == "WAITING_ENTRY" and trade.order_type == "LIMIT":
                        touched = low <= trade.entry_price <= high
                        if touched:
                            self._sim_open(trade, trade.entry_price)
                    elif trade.status == "OPEN":
                        self._sim_check_exit_range(trade, high, low, price, now)
            # Real entry/protection state is confirmed against Binance, not Bybit price.
            if self.mode == "ON" and not BINANCE_GATE.blocked and time.monotonic() - self._last_real_poll >= 5.0:
                self._last_real_poll = time.monotonic()
                for trade in list(self.trades.values()):
                    if trade.mode == "ON" and trade.status in {"WAITING_ENTRY", "ENTRY_SUBMITTED", "OPEN", "PROTECTION_PENDING"}:
                        self._poll_real_trade(trade)
            self._save_state()

    def _sim_check_exit_range(self, trade: Trade, high: float, low: float, last_price: float, now: int) -> None:
        hit_sl = low <= trade.stop_loss if trade.direction == "LONG" else high >= trade.stop_loss
        hit_tp = high >= trade.take_profit if trade.direction == "LONG" else low <= trade.take_profit
        if hit_sl and hit_tp:
            # OHLC/live event does not reveal intrabar order. Conservative assumption: SL first.
            self._close_trade(trade, trade.stop_loss, "SL", now)
        elif hit_sl:
            self._close_trade(trade, trade.stop_loss, "SL", now)
        elif hit_tp:
            self._close_trade(trade, trade.take_profit, "TP", now)

    def _sim_check_exit(self, trade: Trade, price: float, now: int) -> None:
        hit_sl = price <= trade.stop_loss if trade.direction == "LONG" else price >= trade.stop_loss
        hit_tp = price >= trade.take_profit if trade.direction == "LONG" else price <= trade.take_profit
        if hit_sl and hit_tp:
            # Ambiguous intrabar ordering from tick stream: prefer SL conservatively.
            self._close_trade(trade, price, "SL", now)
        elif hit_sl:
            self._close_trade(trade, trade.stop_loss, "SL", now)
        elif hit_tp:
            self._close_trade(trade, trade.take_profit, "TP", now)

    def _close_trade(self, trade: Trade, exit_price: float, outcome: str, ts: int, realized_pnl: float | None = None) -> None:
        if trade.status == "CLOSED":
            return
        if trade.mode == "ON":
            for algo_id in (trade.sl_algo_id, trade.tp_algo_id):
                if algo_id:
                    try:
                        self.binance.cancel_algo_order(trade.symbol, algo_id=algo_id)
                    except Exception:
                        log.info("[REAL] protection already closed/triggered %s algo=%s", trade.symbol, algo_id)
        trade.status = "CLOSED"
        trade.closed_ts = ts
        trade.actual_exit = float(exit_price)
        if realized_pnl is None:
            if trade.direction == "LONG":
                realized_pnl = (exit_price - float(trade.actual_entry or trade.entry_price)) * trade.quantity
            else:
                realized_pnl = (float(trade.actual_entry or trade.entry_price) - exit_price) * trade.quantity
        trade.pnl = float(realized_pnl)
        risk_cash = abs(trade.entry_price - trade.stop_loss) * trade.quantity
        trade.r_multiple = trade.pnl / risk_cash if risk_cash > 0 else None
        trade.outcome = outcome
        self.closed.append(asdict(trade))
        if len(self.closed) > 5000:
            self.closed = self.closed[-5000:]
        self.ban(trade.symbol)
        self._notify(_format_close(trade))
        log.warning("[TRADE CLOSED] %s %s %s pnl=%.8f R=%s", trade.symbol, trade.direction, outcome, trade.pnl, trade.r_multiple)
        self._save_state()
        self.engine._emit_learning("on_trade_event", {"trade_id": trade.id, "signal_id": trade.signal_id, "symbol": trade.symbol, "direction": trade.direction, "status": trade.status, "mode": trade.mode, "ts": int(time.time()*1000), "pnl": trade.pnl, "r_multiple": trade.r_multiple, "outcome": trade.outcome})

    def _place_protection(self, trade: Trade) -> None:
        if trade.mode != "ON" or trade.status not in {"OPEN", "PROTECTION_PENDING"}:
            return
        try:
            side = "SELL" if trade.direction == "LONG" else "BUY"
            qty = self._format_decimal(trade.quantity)
            sl_price = self.normalize_price(trade.symbol, trade.stop_loss)
            tp_price = self.normalize_price(trade.symbol, trade.take_profit)
            common = {
                "algoType": "CONDITIONAL",
                "symbol": trade.symbol,
                "side": side,
                "positionSide": "BOTH",
                "quantity": qty,
                "workingType": "MARK_PRICE",
                "timeInForce": "GTC",
                "reduceOnly": "true",
                "newOrderRespType": "ACK",
            }
            if not trade.sl_algo_id:
                p = dict(common)
                p.update({
                    "type": "STOP_MARKET",
                    "triggerPrice": self._format_decimal(sl_price),
                    "clientAlgoId": f"smc5sl_{trade.id[-18:]}",
                })
                r = self.binance.new_algo_order(p)
                trade.sl_algo_id = str(r.get("algoId")) if r.get("algoId") is not None else None
            if not trade.tp_algo_id:
                p = dict(common)
                p.update({
                    "type": "TAKE_PROFIT_MARKET",
                    "triggerPrice": self._format_decimal(tp_price),
                    "clientAlgoId": f"smc5tp_{trade.id[-18:]}",
                })
                r = self.binance.new_algo_order(p)
                trade.tp_algo_id = str(r.get("algoId")) if r.get("algoId") is not None else None
            trade.status = "OPEN"
            self._notify(
                "🛡️ REAL PROTECTION ACTIVE\n\n"
                f"{trade.symbol} {trade.direction}\n"
                f"SL: {sl_price:.8f}\nTP: {tp_price:.8f}\n"
                f"Qty: {trade.quantity}\n"
                f"Margin: {trade.margin_actual:.8f}"
            )
        except Exception as exc:
            trade.status = "PROTECTION_PENDING"
            trade.error = f"{type(exc).__name__}: {exc}"
            log.exception("[REAL] protection failed %s", trade.symbol)
            self._notify(f"🚨 CRITICAL PROTECTION ERROR\n{trade.symbol} {trade.direction}\n{trade.error}\nRetrying automatically.")

    def _poll_real_trade(self, trade: Trade, force: bool = False) -> None:
        if not self.binance.configured or not trade.entry_order_id:
            return
        try:
            order = self.binance.query_order(trade.symbol, order_id=trade.entry_order_id)
            status = str(order.get("status") or "")
            executed = float(order.get("executedQty") or 0)
            avg_price = float(order.get("avgPrice") or 0)
            if trade.status in {"WAITING_ENTRY", "ENTRY_SUBMITTED"}:
                if status == "FILLED":
                    trade.actual_entry = avg_price if avg_price > 0 else trade.entry_price
                    trade.opened_ts = int(time.time() * 1000)
                    trade.status = "PROTECTION_PENDING"
                    self._notify(_format_trade(trade, "🟢 REAL ENTRY FILLED"))
                    log.warning("[REAL] FILLED %s %s avg=%s qty=%s", trade.symbol, trade.direction, trade.actual_entry, executed)
                    self._place_protection(trade)
                elif status in {"CANCELED", "REJECTED", "EXPIRED"}:
                    trade.status = "ERROR"
                    trade.error = f"entry order status={status}"
                    self._notify(f"❌ REAL ENTRY ENDED\n{trade.symbol}\n{trade.error}")
            elif trade.status in {"OPEN", "PROTECTION_PENDING"}:
                self._place_protection(trade)
                # Detect zero position as a close and derive realized PnL from user trades.
                account = self.binance.account()
                pos = next((p for p in account.get("positions", []) if p.get("symbol") == trade.symbol), None)
                position_amt = float(pos.get("positionAmt") or 0) if pos else 0.0
                if abs(position_amt) < 1e-15 and trade.opened_ts:
                    trades = self.binance.user_trades(trade.symbol, max(trade.opened_ts - 2000, 0))
                    realized = 0.0
                    exit_prices = []
                    for rt in trades:
                        realized += float(rt.get("realizedPnl") or 0)
                        if float(rt.get("qty") or 0) > 0:
                            exit_prices.append(float(rt.get("price") or 0))
                    if abs(realized) > 0 or exit_prices:
                        exit_price = exit_prices[-1] if exit_prices else trade.entry_price
                        outcome = "TP" if realized > 0 else "SL"
                        self._close_trade(trade, exit_price, outcome, int(time.time() * 1000), realized_pnl=realized)
        except Exception as exc:
            log.warning("[REAL POLL] %s failed: %s", trade.symbol, exc)

    def mode_text(self) -> str:
        return f"MODE: {self.mode}\nExecution: {'REAL Binance Futures' if self.mode == 'ON' else 'SIMULATION'}\nMargin: {self.margin}\nLeverage: {self.leverage}x\nMax active: {self.max_active}\n{BINANCE_GATE.status_text()}"

    def set_mode(self, value: str) -> str:
        value = value.upper()
        if value not in {"ON", "OFF"}:
            return "Format: /mode on|off"
        with self.lock:
            if value == "ON":
                if not self.binance.configured:
                    return "❌ MODE ON ditolak: Binance API key/secret belum tersedia."
                try:
                    if self.binance.position_mode():
                        return "❌ MODE ON ditolak: Binance masih Hedge Mode. V5 memakai One-way Mode."
                except Exception as exc:
                    return f"❌ MODE ON gagal memeriksa Binance: {type(exc).__name__}: {exc}"
                self.mode = "ON"
                self._notify("🔴 REAL MODE ACTIVE\nNew confirmed signals may create real Binance Futures orders.")
                return self.mode_text()
            self.mode = "OFF"
            # Cancel only bot-owned pending real entry orders; never close live positions.
            canceled = 0
            for trade in list(self.trades.values()):
                if trade.mode == "ON" and trade.status in {"WAITING_ENTRY", "ENTRY_SUBMITTED"} and trade.entry_order_id:
                    try:
                        self.binance.cancel_order(trade.symbol, order_id=trade.entry_order_id)
                        trade.status = "ERROR"
                        trade.error = "canceled_by_mode_off"
                        canceled += 1
                    except Exception:
                        log.exception("[MODE OFF] cancel failed %s", trade.symbol)
            self._save_state()
            return f"{self.mode_text()}\nCanceled pending real entries: {canceled}"

    def stats(self) -> str:
        with self.lock:
            closed = self.closed[-5000:]
            wins = sum(1 for x in closed if x.get("outcome") == "TP")
            losses = sum(1 for x in closed if x.get("outcome") == "SL")
            pnl = sum(float(x.get("pnl") or 0) for x in closed)
            rs = [float(x["r_multiple"]) for x in closed if x.get("r_multiple") is not None]
            wr = (wins / len(closed) * 100) if closed else 0.0
            avg_r = sum(rs) / len(rs) if rs else 0.0
            active = self.active_count()
            lines = [
                "📊 TRADING STATS",
                "",
                f"Mode: {self.mode}",
                f"Closed trades: {len(closed)}",
                f"Wins: {wins}",
                f"Losses: {losses}",
                f"Win rate: {wr:.2f}%",
                f"Net PnL: {pnl:+.8f}",
                f"Average R: {avg_r:+.3f}R",
                f"Active orders/positions: {active}/{self.max_active}",
                "",
                "Recent closes:",
            ]
            for x in reversed(closed[-15:]):
                lines.append(
                    f"{x['symbol']} {x['direction']} | {x.get('outcome')} | "
                    f"PnL={float(x.get('pnl') or 0):+.8f} | R={float(x.get('r_multiple') or 0):+.2f}"
                )
            return "\n".join(lines)[:3900]

    def trades_text(self) -> str:
        with self.lock:
            rows = [t for t in self.trades.values() if t.status in {"WAITING_ENTRY", "ENTRY_SUBMITTED", "OPEN", "PROTECTION_PENDING"}]
            if not rows:
                return "📭 Tidak ada active trade/order."
            lines = [f"📊 ACTIVE TRADES ({len(rows)}/{self.max_active})"]
            for i, t in enumerate(sorted(rows, key=lambda x: x.created_ts), 1):
                lines.append(
                    f"\n{i}. {t.symbol} {t.direction}\n"
                    f"{t.mode} | {t.order_type} | {t.status}\n"
                    f"Entry {t.entry_price:.8f}\nSL {t.stop_loss:.8f}\nTP {t.take_profit:.8f}\n"
                    f"Qty {t.quantity} | Margin {t.margin_actual:.8f} | Lev {t.leverage}x"
                )
            return "\n".join(lines)[:3900]

    def bans_text(self) -> str:
        with self.lock:
            self._purge_bans()
            if not self.banned:
                return "🟢 BAN LIST KOSONG"
            now = int(time.time() * 1000)
            lines = ["🚫 BANNED SYMBOLS"]
            for symbol, until in sorted(self.banned.items(), key=lambda x: x[1]):
                mins = max(0, int((until - now) / 60000))
                lines.append(f"{symbol} | {mins} min remaining")
            return "\n".join(lines)

    def unban(self, symbol: str) -> str:
        symbol = symbol.upper()
        with self.lock:
            if symbol not in self.banned:
                return f"ℹ️ {symbol} tidak sedang banned."
            self.banned.pop(symbol, None)
            self._save_state()
            return f"✅ {symbol} UNBANNED"

    def reset_bans(self) -> str:
        with self.lock:
            n = len(self.banned)
            self.banned.clear()
            self._save_state()
            return f"✅ BAN LIST RESET\nRemoved: {n} symbols"


class BybitHttpGate:
    """Conservative shared HTTP gate for Bybit requests."""
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_request = 0.0
        self.starts: list[float] = []
        self.backoff_until = 0.0

    def before(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                if now < self.backoff_until:
                    wait = self.backoff_until - now
                else:
                    cutoff = now - BYBIT_HTTP_WINDOW
                    self.starts = [x for x in self.starts if x >= cutoff]
                    gap = BYBIT_REQUEST_MIN_GAP - (now - self.last_request)
                    window_wait = 0.0
                    if len(self.starts) >= BYBIT_HTTP_MAX_REQUESTS:
                        window_wait = max(0.0, self.starts[0] + BYBIT_HTTP_WINDOW - now)
                    wait = max(gap, window_wait, 0.0)
                if wait <= 0:
                    now = time.monotonic()
                    self.last_request = now
                    self.starts.append(now)
                    return
            time.sleep(min(wait, 1.0))

    def after(self, response: requests.Response) -> None:
        if response.status_code == 429:
            with self.lock:
                self.backoff_until = max(self.backoff_until, time.monotonic() + BYBIT_429_BACKOFF)
            log.warning("[BYBIT] 429 received; backing off %.1fs", BYBIT_429_BACKOFF)

BYBIT_GATE = BybitHttpGate()


class DataEngine:
    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context
        self.stop_event: threading.Event = context["stop_event"]
        self.send_message: Callable[[int, Any], None] = context["send_message"]
        self.chat_id = context.get("chat_id")
        self.store = DataStore()
        self.api = DataAPI(self)
        self.trade_manager = TradeManager(self, context)
        self.symbols: list[str] = []
        self.symbol_lock = threading.RLock()
        self.callbacks: list[Callable[[dict[str, Any]], None]] = []
        self.run_lock = threading.RLock()
        self.auto_running = False
        self.auto_armed = False
        self.binance_cooldown_until = 0.0
        self.binance_cooldown_reason: str | None = None
        self.binance_recovery_thread: threading.Thread | None = None
        self._recovery_lock = threading.RLock()
        self._auto_pipeline_started = False
        self.bootstrap_complete = False
        # WebSocket is a pool: one connection per subscription batch.
        self.ws_apps: dict[int, websocket.WebSocketApp] = {}
        self.ws_threads: dict[int, threading.Thread] = {}
        self.ws_batch_topics: dict[int, list[str]] = {}
        self.ws_connected: bool = False
        self.ws: websocket.WebSocketApp | None = None
        self.ws_thread: threading.Thread | None = None
        self.bootstrap_thread: threading.Thread | None = None
        self.strategy: Any = None
        self.strategy_error: str | None = None
        self.tick_logs: dict[tuple[str, str], float] = {}
        self.worker_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="smc-worker")
        self.learning: Any = None
        self.learning_error: str | None = None
        self.load_learning()
        # learn.py receives a callback to reload a promoted strategy without
        # rebuilding the data engine or websocket pool.
        self.context["reset_strategy"] = self.reset_strategy

    def _notify(self, text: str) -> None:
        if self.chat_id is None:
            return
        try:
            self.send_message(self.chat_id, text)
        except Exception:
            log.exception("[TG] notify failed")

    def _strategy_path(self) -> Path:
        raw = STRATEGY_FILE
        if raw.startswith("[") and "](" in raw:
            raw = raw.split("](", 1)[0].lstrip("[")
        path = Path(raw)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    def _learning_path(self) -> Path:
        raw = LEARN_FILE
        if raw.startswith("[") and "](" in raw:
            raw = raw.split("](", 1)[0].lstrip("[")
        p = Path(raw)
        return p if p.is_absolute() else (BASE_DIR / p).resolve()

    def load_learning(self) -> None:
        path = self._learning_path()
        self.learning = None
        self.learning_error = None
        if not path.is_file():
            self.learning_error = f"learn file not found: {path}"
            log.warning("[LEARN] %s", self.learning_error)
            return
        try:
            name = f"smc_learn_v3_{int(time.time()*1000)}"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError("cannot create learn spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self.learning = module
            if hasattr(module, "initialize"):
                module.initialize(self.api, self.context)
            # Learning memory is opened automatically on startup.  /open remains
            # available as an explicit restore command.
            if hasattr(module, "open_memory"):
                try:
                    restored = module.open_memory()
                    log.info("[LEARN] startup memory open: %s", str(restored).replace("\n", " | "))
                except Exception:
                    log.exception("[LEARN] startup memory open failed")
            log.info("[LEARN] loaded %s", path)
            self._notify(f"✅ LEARN ENGINE LOADED + MEMORY OPENED\n{path.name}")
        except Exception as exc:
            self.learning = None
            self.learning_error = f"{type(exc).__name__}: {exc}"
            log.exception("[LEARN] load failed")
            self._notify(f"⚠️ LEARN ENGINE UNAVAILABLE\n{self.learning_error}")

    def _emit_learning(self, method: str, payload: dict[str, Any]) -> None:
        module = self.learning
        if not module:
            return
        try:
            fn = getattr(module, method, None)
            if callable(fn):
                fn(payload)
        except Exception:
            log.exception("[LEARN] %s failed", method)

    def public_ip(self) -> str:
        r = requests.get(IP_URL, timeout=10)
        r.raise_for_status()
        return r.text.strip()

    def _get(self, url: str, params: dict[str, Any]) -> requests.Response:
        if url.startswith(BYBIT_BASE_URL):
            BYBIT_GATE.before()
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP request failed: {exc}") from exc
        if url.startswith(BYBIT_BASE_URL):
            BYBIT_GATE.after(r)
        r.raise_for_status()
        return r

    # ---- Bybit discovery / historical ----
    def bybit_symbols(self) -> set[str]:
        out: set[str] = set(); cursor = None
        while True:
            p: dict[str, Any] = {"category": "linear", "status": "Trading", "limit": 1000}
            if cursor: p["cursor"] = cursor
            d = self._get(f"{BYBIT_BASE_URL}/v5/market/instruments-info", p).json()
            if d.get("retCode") != 0: raise RuntimeError(f"Bybit instruments: {d}")
            result = d.get("result") or {}
            for x in result.get("list") or []:
                if x.get("contractType") == "LinearPerpetual" and x.get("quoteCoin") == "USDT" and x.get("settleCoin") == "USDT" and x.get("status") == "Trading":
                    out.add(str(x.get("symbol") or "").upper())
            cursor = result.get("nextPageCursor") or None
            if not cursor: break
        return out

    def binance_symbols(self) -> set[str]:
        data = self.trade_manager.binance.exchange_info()
        out: set[str] = set()
        for x in data.get("symbols") or []:
            if (
                x.get("status") == "TRADING"
                and x.get("contractType") == "PERPETUAL"
                and x.get("quoteAsset") == "USDT"
                and x.get("marginAsset") == "USDT"
            ):
                s = str(x.get("symbol") or "").upper()
                if s:
                    out.add(s)
        if data.get("stale"):
            log.warning("[DISCOVERY] Binance symbols are from stale local cache")
        return out

    def build_universe(self) -> list[str]:
        b1 = self.bybit_symbols()
        log.info("[DISCOVERY] Bybit USDT perpetuals=%d", len(b1))
        try:
            b2 = self.binance_symbols()
            log.info("[DISCOVERY] Binance USDT perpetuals=%d", len(b2))
            common = sorted(b1 & b2)
        except BinanceCircuitOpen as exc:
            # Keep the Bybit market-data pipeline alive. A previous universe is
            # reused if available; otherwise discovery cannot form an intersection.
            with self.symbol_lock:
                existing = list(self.symbols)
            if existing:
                log.warning("[DISCOVERY] Binance circuit open; reusing previous universe=%d", len(existing))
                common = existing
            else:
                raise RuntimeError(f"Binance unavailable and no cached universe: {exc}") from exc
        with self.symbol_lock:
            self.symbols = common
        log.info("[DISCOVERY] common=%d selected=%d", len(common), len(common))
        return common

    def fetch_klines(self, symbol: str, tf: str, limit: int) -> list[Candle]:
        d = self._get(f"{BYBIT_BASE_URL}/v5/market/kline", {"category": "linear", "symbol": symbol, "interval": tf, "limit": limit}).json()
        if d.get("retCode") != 0: raise RuntimeError(f"Bybit kline {symbol}/{tf}: {d}")
        rows = []
        for r in ((d.get("result") or {}).get("list") or []):
            if len(r) < 6: continue
            rows.append(Candle(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]) if len(r) > 6 else 0.0, True))
        rows.sort(key=lambda c: c.timestamp)
        return rows[-limit:]

    def bootstrap_symbol(self, symbol: str) -> bool:
        ok = True
        for tf, limit in TF_CONFIG.items():
            if self.stop_event.is_set() or not self.auto_running: return False
            try:
                candles = self.fetch_klines(symbol, tf, limit)
                self.store.set_history(symbol, tf, candles)
                ok = ok and len(candles) >= limit
                log.info("[REST] %s %s %d/%d", symbol, tf, len(candles), limit)
            except Exception:
                ok = False; log.exception("[REST] %s %s failed", symbol, tf)
        return ok

    def bootstrap_all(self) -> None:
        syms = self.get_symbols()
        total = len(syms)
        success = 0
        failed = 0
        completed = 0
        failed_symbols: list[str] = []
        started = time.monotonic()
        log.info(
            "[BOOTSTRAP] START %d symbols | workers=%d | batch=%d",
            total, MAX_WORKERS, BOOTSTRAP_BATCH_SIZE
        )

        def load_one(symbol: str) -> tuple[str, bool]:
            if self.stop_event.is_set() or not self.auto_running:
                return symbol, False
            return symbol, self.bootstrap_symbol(symbol)

        # Bounded queue: at most BOOTSTRAP_BATCH_SIZE Future objects exist at once.
        # Actual REST concurrency is capped globally by the shared worker pool.
        for offset in range(0, total, BOOTSTRAP_BATCH_SIZE):
            if self.stop_event.is_set() or not self.auto_running:
                return
            batch = syms[offset: offset + BOOTSTRAP_BATCH_SIZE]
            futures = {self.worker_pool.submit(load_one, symbol): symbol for symbol in batch}
            try:
                for future in as_completed(futures):
                    if self.stop_event.is_set() or not self.auto_running:
                        break
                    symbol = futures[future]
                    completed += 1
                    try:
                        _, ok = future.result()
                    except Exception:
                        ok = False
                        log.exception("[BOOTSTRAP] %s failed in worker", symbol)
                    if ok:
                        success += 1
                    else:
                        failed += 1
                        failed_symbols.append(symbol)
                        # Remove any partial data from a symbol that failed one or more
                        # timeframes. It must never leak incomplete history into strategy.
                        self.store.remove_symbol(symbol)
                    if completed == 1 or completed % 25 == 0 or completed == total:
                        elapsed = max(0.001, time.monotonic() - started)
                        rate = completed / elapsed
                        eta = (total - completed) / rate if rate else 0
                        log.info(
                            "[BOOTSTRAP] %d/%d | success=%d failed=%d | %.2f pair/s | ETA %.0fs",
                            completed, total, success, failed, rate, eta
                        )
                        self._notify(
                            "📥 BOOTSTRAP PROGRESS\n"
                            f"{completed}/{total} pairs\n"
                            f"Success: {success} | Failed: {failed}\n"
                            f"Workers: {MAX_WORKERS}"
                        )
            finally:
                if self.stop_event.is_set() or not self.auto_running:
                    for future in futures:
                        future.cancel()

        if not self.auto_running or self.stop_event.is_set():
            return

        # Best-effort bootstrap policy:
        # failed pairs are discarded, successful pairs remain active, and the
        # strategy/WS continue as long as at least one complete pair remains.
        if failed_symbols:
            failed_set = set(failed_symbols)
            with self.symbol_lock:
                self.symbols = [s for s in self.symbols if s not in failed_set]

        active_total = len(self.get_symbols())
        elapsed = time.monotonic() - started
        self.bootstrap_complete = active_total > 0 and success > 0
        log.info(
            "[BOOTSTRAP] COMPLETE usable=%d/%d failed=%d elapsed=%.1fs",
            active_total, total, failed, elapsed
        )

        if not self.bootstrap_complete:
            self._notify(
                "❌ DATA BOOTSTRAP FAILED\n"
                f"Usable pairs: {active_total}/{total}\n"
                f"Failed/discarded: {failed}\n"
                "No complete pair is available; strategy/WebSocket not started."
            )
            return

        if failed_symbols:
            preview = ", ".join(failed_symbols[:12])
            if len(failed_symbols) > 12:
                preview += f" … +{len(failed_symbols) - 12} more"
            self._notify(
                "⚠️ BOOTSTRAP BEST-EFFORT\n"
                f"Usable pairs: {active_total}/{total}\n"
                f"Discarded failed pairs: {failed}\n"
                f"Dropped: {preview}\n"
                "Continuing with successful pairs."
            )

        self.load_strategy()
        if self.learning and hasattr(self.learning, "build_global_context"):
            try:
                ctx = self.learning.build_global_context()
                log.info(
                    "[LEARN] initial global context regime=%s label=%s breadth=%.2f",
                    ctx.get("regime", "-"), ctx.get("market_label", "-"),
                    float(ctx.get("breadth", 0.0))
                )
            except Exception:
                log.exception("[LEARN] initial global context build failed")

        self._notify(
            "✅ DATA READY\n"
            f"Universe before bootstrap: {total}\n"
            f"Usable pairs: {active_total}\n"
            f"Discarded: {failed}\n"
            "Historical: 15M/700 + 5M/500 + 1M/500\n"
            f"Bootstrap time: {elapsed:.1f}s\n"
            f"Workers: {MAX_WORKERS}\n"
            f"Strategy: {'READY' if self.strategy else 'NOT READY'}"
        )

        if self.strategy and hasattr(self.strategy, "on_data_ready"):
            try:
                summary = self.strategy.on_data_ready()
                self._notify(str(summary or "🔎 Initial scan selesai\nConfirmed signals: 0"))
                self._accept_strategy_queue()
            except Exception as exc:
                log.exception("[STRATEGY] initial scan failed")
                self._notify(f"❌ INITIAL STRATEGY SCAN ERROR\n{type(exc).__name__}: {exc}")
        else:
            self._notify(
                "⚠️ INITIAL STRATEGY SCAN SKIPPED\n"
                f"Reason: {self.strategy_error or 'strategy unavailable'}"
            )

        self.start_websocket()

    def reset_strategy(self) -> str:
        """Reload strategy.py while preserving market data, trades and bans."""
        with self.run_lock:
            if not self.auto_running:
                return "ℹ️ /reset hanya boleh dijalankan saat /auto aktif."

        old = self.strategy
        old_path = getattr(old, "__file__", None) if old else None

        # Stop only the old strategy runtime. Do NOT stop data/WS/trade manager.
        if old and hasattr(old, "shutdown"):
            try:
                old.shutdown()
            except Exception:
                log.exception("[STRATEGY] old shutdown failed during reset")

        self.strategy = None
        self.strategy_error = None

        try:
            path = self._strategy_path()
            if not path.is_file():
                raise FileNotFoundError(f"strategy file not found: {path}")

            name = f"smc_strategy_reset_v13_{int(time.time()*1000)}"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError("cannot create strategy spec")

            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)

            if hasattr(module, "initialize"):
                module.initialize(self.api, self.context)

            self.strategy = module
            self.strategy_error = None

            if self.learning and hasattr(self.learning, "on_strategy_loaded"):
                try:
                    self.learning.on_strategy_loaded(module, path)
                except Exception:
                    log.exception("[LEARN] on_strategy_loaded failed during reset")

            log.info("[STRATEGY RESET] old=%s new=%s", old_path, path)

            # Fresh analysis over already-collected candles.
            result = None
            if hasattr(module, "on_data_ready"):
                result = module.on_data_ready()

            self._notify(
                "🔄 STRATEGY RESET COMPLETE\n"
                f"Loaded: {path.name}\n"
                "Data/WebSocket/Trades/Bans tetap berjalan.\n"
                + (str(result).strip() if result else "🔎 Initial scan selesai\nSetup baru: 0")
            )
            self._accept_strategy_queue()
            return "✅ Strategy berhasil di-reset dan di-scan ulang."
        except Exception as exc:
            self.strategy = None
            self.strategy_error = f"{type(exc).__name__}: {exc}"
            log.exception("[STRATEGY RESET] failed")
            self._notify(
                "🚨 STRATEGY RESET FAILED\n"
                f"{type(exc).__name__}: {exc}\n"
                "Data/WebSocket tetap berjalan."
            )
            return f"❌ Strategy reset gagal: {type(exc).__name__}: {exc}"

    def load_strategy(self) -> None:
        path = self._strategy_path()
        self.strategy = None; self.strategy_error = None
        if not path.is_file():
            self.strategy_error = f"strategy file not found: {path}"
            log.error("[STRATEGY] %s", self.strategy_error)
            self._notify(f"❌ STRATEGY LOAD FAILED\nFile: {path.name}\nPath: {path}")
            return
        try:
            name = f"smc_strategy_runtime_v13_{int(time.time()*1000)}"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None: raise ImportError("cannot create strategy spec")
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module
            spec.loader.exec_module(module)
            self.strategy = module
            if hasattr(module, "initialize"): module.initialize(self.api, self.context)
            if self.learning and hasattr(self.learning, "on_strategy_loaded"):
                self.learning.on_strategy_loaded(module, path)
            log.info("[STRATEGY] loaded %s", path)
            self._notify(f"✅ STRATEGY LOADED\n{path.name}")
        except Exception as exc:
            self.strategy_error = f"{type(exc).__name__}: {exc}"
            log.exception("[STRATEGY] load failed")
            self._notify(f"❌ STRATEGY LOAD ERROR\nFile: {path.name}\nError: {self.strategy_error}")

    def _accept_strategy_queue(self) -> None:
        if self.strategy and hasattr(self.strategy, "drain_signals"):
            try:
                for signal in self.strategy.drain_signals() or []:
                    self._accept_signal(signal)
            except Exception:
                log.exception("[STRATEGY] drain_signals failed")

    def _accept_signal(self, signal: dict[str, Any]) -> None:
        signal_id = str(signal.get("id") or "")
        if not signal_id: return
        # prevent duplicate signal processing in same runtime/state
        if any(t.signal_id == signal_id and t.status not in {"CLOSED", "ERROR"} for t in self.trade_manager.trades.values()):
            return
        log.warning(
            "[SIGNAL] CONFIRMED %s %s entry=%s @ %.8f SL=%.8f TP=%.8f RR=%.2f score=%s",
            signal.get("symbol"), signal.get("direction"), signal.get("entry_type"),
            float(signal.get("entry_price")), float(signal.get("stop_loss")), float(signal.get("take_profit")),
            float(signal.get("rr", 0)), signal.get("score"),
        )
        self._emit_learning("on_signal", dict(signal))
        tid = self.trade_manager.accept_signal(signal)
        if tid:
            trade = self.trade_manager.trades.get(tid)
            if trade:
                self._emit_learning("on_trade_event", {"trade_id": trade.id, "signal_id": trade.signal_id, "symbol": trade.symbol, "direction": trade.direction, "status": trade.status, "mode": trade.mode, "ts": int(time.time()*1000), "pnl": trade.pnl, "r_multiple": trade.r_multiple})

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        symbol = str(event.get("symbol") or "").upper()
        candle = event.get("candle") or {}
        if symbol and candle.get("close") is not None:
            self.trade_manager.on_market_price(symbol, float(candle["close"]), candle)
        self._emit_learning("on_market_event", event)
        if self.strategy and hasattr(self.strategy, "on_market_event"):
            try:
                result = self.strategy.on_market_event(event)
                if isinstance(result, dict) and result.get("type") == "signal":
                    self._accept_signal(result.get("signal") or {})
                elif isinstance(result, str) and result.strip():
                    # Backward compatibility: allow old strategy strings to reach Telegram.
                    self._notify(result.strip())
                self._accept_strategy_queue()
            except Exception:
                log.exception("[STRATEGY] on_market_event failed")
        for cb in tuple(self.callbacks):
            try: cb(event)
            except Exception: log.exception("[DATA CALLBACK] failed")

    # ---- websocket ----
    def add_data_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.callbacks.append(callback)

    def _topics(self) -> list[str]:
        return [f"kline.{tf}.{s}" for tf in TF_CONFIG for s in self.get_symbols()]

    @staticmethod
    def _topic_batches(topics: list[str], max_chars: int) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        chars = 2
        for topic in topics:
            add = len(topic) + (1 if current else 0)
            if current and chars + add > max_chars:
                batches.append(current)
                current = []
                chars = 2
            current.append(topic)
            chars += add
        if current:
            batches.append(current)
        return batches

    def _ws_open(self, ws: websocket.WebSocketApp) -> None:
        topics = self._topics()
        batches = self._topic_batches(topics, WS_MAX_ARG_CHARS)
        self.ws = ws
        self.ws_connected = True
        log.info("[WS] connected | topics=%d subscribe_messages=%d | one I/O thread", len(topics), len(batches))
        for idx, batch in enumerate(batches, 1):
            payload = {"op": "subscribe", "req_id": f"smc9-{idx}-{uuid.uuid4().hex[:6]}", "args": batch}
            ws.send(json.dumps(payload, separators=(",", ":")))
            log.info("[WS] subscribe %d/%d | topics=%d chars=%d", idx, len(batches), len(batch), sum(len(x) for x in batch))
            if idx < len(batches):
                time.sleep(0.05)
        self._notify(
            "🟢 MARKET DATA LIVE\nBybit WS\n"
            f"Symbols: {len(self.symbols)}\nStreams: 15M + 5M + 1M\n"
            f"Topics: {len(topics)}\nSubscribe messages: {len(batches)}\nWS connections: 1"
        )

    def _ws_message(self, _ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            p = json.loads(raw)
        except json.JSONDecodeError:
            return
        if p.get("op") in {"subscribe", "pong", "unsubscribe"}:
            if p.get("op") == "subscribe" and p.get("success") is False:
                log.error("[WS] subscription rejected: %s", p)
            return
        parts = str(p.get("topic") or "").split(".")
        if len(parts) != 3 or parts[0] != "kline":
            return
        tf = parts[1]
        for x in p.get("data") or []:
            try:
                c = Candle(int(x["start"]), float(x["open"]), float(x["high"]), float(x["low"]), float(x["close"]), float(x["volume"]), float(x.get("turnover") or 0), bool(x.get("confirm", False)))
                symbol = str(x.get("symbol") or parts[2]).upper()
            except (KeyError, TypeError, ValueError):
                log.exception("[WS] bad kline payload")
                continue
            self.store.upsert(symbol, tf, c)
            key = (symbol, tf)
            now = time.monotonic()
            if now - self.tick_logs.get(key, 0) >= LOG_EVERY_SYMBOL_TICK:
                self.tick_logs[key] = now
                log.info("[TICK WS] %s %s C=%.8f confirm=%s", symbol, tf, c.close, c.confirmed)
            self._dispatch_event({"type": "candle", "symbol": symbol, "timeframe": tf, "candle": c.as_dict()})

    def _ws_error(self, _ws: websocket.WebSocketApp, error: Any) -> None:
        log.warning("[WS] error=%s", error)

    def _ws_close(self, _ws: websocket.WebSocketApp, code: Any, msg: Any) -> None:
        self.ws_connected = False
        log.warning("[WS] closed code=%s msg=%s", code, msg)

    def _ws_worker(self) -> None:
        backoff = 2.0
        while self.auto_running and not self.stop_event.is_set():
            try:
                ws = websocket.WebSocketApp(BYBIT_WS_URL, on_open=self._ws_open, on_message=self._ws_message, on_error=self._ws_error, on_close=self._ws_close)
                self.ws = ws
                log.info("[WS] connecting %s", BYBIT_WS_URL)
                ws.run_forever(ping_interval=WS_PING_INTERVAL, ping_timeout=max(2, WS_PING_INTERVAL - 2), skip_utf8_validation=True)
            except Exception:
                log.exception("[WS] worker error")
            finally:
                self.ws_connected = False
                self.ws = None
            if self.auto_running and not self.stop_event.is_set():
                jitter = (uuid.uuid4().int % 1000) / 1000 * WS_RECONNECT_JITTER
                delay = min(WS_RECONNECT_MAX, backoff + jitter)
                log.warning("[WS] reconnect in %.1fs", delay)
                if self.stop_event.wait(delay):
                    return
                backoff = min(backoff * 2.0, WS_RECONNECT_MAX)

    def start_websocket(self) -> None:
        if self.ws_thread and self.ws_thread.is_alive():
            return
        self.ws_thread = threading.Thread(target=self._ws_worker, name="bybit-ws", daemon=True)
        self.ws_thread.start()

    def stop_websocket(self) -> None:
        # Close every active websocket in the pool, plus the legacy single
        # connection reference used by older worker code.
        apps = list(getattr(self, "ws_apps", {}).items())
        for idx, ws_app in apps:
            try:
                ws_app.close()
            except Exception:
                log.exception("[WS] close failed batch=%s", idx)
        if hasattr(self, "ws_apps"):
            self.ws_apps.clear()
        if hasattr(self, "ws_batch_topics"):
            self.ws_batch_topics.clear()

        ws = getattr(self, "ws", None)
        self.ws = None
        self.ws_connected = False
        if ws:
            try:
                ws.close()
            except Exception:
                log.exception("[WS] legacy ws close failed")

    # ---- lifecycle / command ----

    # ---------------- Binance blackout / auto-resume ----------------
    def _binance_blackout_remaining(self) -> int:
        return max(0, int(self.binance_cooldown_until - time.time()))

    def _binance_blackout_active(self) -> bool:
        return self.binance_cooldown_until > time.time()

    def _arm_binance_blackout(self, retry_after: float, reason: str) -> None:
        total = max(1.0, float(retry_after)) + BINANCE_EXTRA_COOLDOWN
        with self._recovery_lock:
            self.binance_cooldown_until = max(
                self.binance_cooldown_until,
                time.time() + total,
            )
            self.binance_cooldown_reason = reason
            self.auto_armed = True

        log.warning(
            "[BINANCE] blackout armed | reason=%s | retry_after=%.0fs | safety=%ss | total=%ss",
            reason, retry_after, BINANCE_EXTRA_COOLDOWN, total
        )
        self._notify(
            "⚠️ BINANCE COOLDOWN\n"
            f"Reason: {reason}\n"
            f"Duration: {int(total)}s (incl. +{BINANCE_EXTRA_COOLDOWN}s safety)\n"
            "Bybit/strategy tetap berjalan.\n"
            "AUTO tetap armed — akan resume otomatis."
        )
        self._ensure_recovery_worker()

    def _ensure_recovery_worker(self) -> None:
        with self._recovery_lock:
            if self.binance_recovery_thread and self.binance_recovery_thread.is_alive():
                return
            self.binance_recovery_thread = threading.Thread(
                target=self._binance_recovery_loop,
                name="binance-auto-recovery",
                daemon=True,
            )
            self.binance_recovery_thread.start()

    def _binance_recovery_loop(self) -> None:
        while self.auto_armed and not self.stop_event.is_set():
            remaining = self._binance_blackout_remaining()
            if remaining > 0:
                self.stop_event.wait(min(BINANCE_RECOVERY_POLL, remaining))
                continue

            try:
                # Health check must not cause a recursive cooldown through the
                # same handler; use a very lightweight unauthenticated endpoint.
                if hasattr(self, "_raw_binance_get"):
                    self._raw_binance_get("/fapi/v1/time", {})
                else:
                    r = requests.get(
                        f"{BINANCE_BASE_URL}/fapi/v1/time",
                        timeout=REQUEST_TIMEOUT,
                    )
                    r.raise_for_status()

                with self._recovery_lock:
                    self.binance_cooldown_until = 0.0
                    self.binance_cooldown_reason = None
                self._notify(
                    "🟢 BINANCE RECOVERY\n"
                    "Cooldown selesai.\n"
                    "Health check: ✅\n"
                    "AUTO pipeline dilanjutkan."
                )
                self._resume_auto_pipeline()
                return
            except Exception as exc:
                log.warning("[BINANCE] recovery health check failed: %s", exc)
                with self._recovery_lock:
                    self.binance_cooldown_until = time.time() + BINANCE_RECOVERY_POLL
                self.stop_event.wait(BINANCE_RECOVERY_POLL)

    def _resume_auto_pipeline(self) -> None:
        with self.run_lock:
            if not self.auto_running or not self.auto_armed:
                return
            self._auto_pipeline_started = False

        try:
            symbols = self.build_universe()
            if not symbols:
                raise RuntimeError("common symbol universe still empty")
            self._notify(
                "🔄 AUTO RESUME\n"
                f"Common pairs: {len(symbols)}\n"
                "Melanjutkan bootstrap/data pipeline."
            )
            self._start_bootstrap_once()
        except Exception as exc:
            log.exception("[AUTO RESUME] failed")
            if "418" in str(exc) or "429" in str(exc):
                # The request path should already have armed a new blackout.
                return
            self._notify(f"❌ AUTO RESUME FAILED\n{type(exc).__name__}: {exc}")

    def _start_bootstrap_once(self) -> None:
        with self.run_lock:
            if not self.auto_running or not self.auto_armed:
                return
            if self._auto_pipeline_started:
                return
            if self._binance_blackout_active():
                self._ensure_recovery_worker()
                return
            self._auto_pipeline_started = True

        self.bootstrap_thread = threading.Thread(
            target=self.bootstrap_all,
            name="historical-bootstrap",
            daemon=True,
        )
        self.bootstrap_thread.start()

    def start_auto(self) -> str:
        with self.run_lock:
            if self.auto_running and self.auto_armed:
                if self._binance_blackout_active():
                    return (
                        "⏳ /auto armed.\n"
                        f"Binance cooldown: {self._binance_blackout_remaining()}s\n"
                        "Bybit/strategy tetap berjalan.\n"
                        "Auto-resume aktif."
                    )
                return "ℹ️ /auto sudah aktif."

            self.auto_running = True
            self.auto_armed = True
            self.bootstrap_complete = False
            self._auto_pipeline_started = False

        try:
            ip = self.public_ip()
        except Exception as exc:
            ip = f"unavailable ({exc})"

        log.info("[AUTO] armed | server_ip=%s", ip)
        self._notify(
            "🤖 AUTO MODE ARMED\n"
            f"Server IP: {ip}\n"
            "Discovering Bybit + Binance..."
        )

        if self._binance_blackout_active():
            self._ensure_recovery_worker()
            return (
                "⏳ /auto armed.\n"
                f"Binance cooldown: {self._binance_blackout_remaining()}s\n"
                "Auto akan lanjut otomatis."
            )

        try:
            symbols = self.build_universe()
            if not symbols:
                raise RuntimeError("common symbol universe kosong")
            self._notify(
                f"✅ Universe ready\n"
                f"Common pairs: {len(symbols)}\n"
                "No pair hard-cap\n\n"
                "Bootstrap 15M/5M/1M dimulai..."
            )
            self._start_bootstrap_once()
            return f"🟢 /auto aktif — {len(symbols)} pair masuk pipeline."

        except Exception as exc:
            text = str(exc)
            if "418" in text or "429" in text:
                self._ensure_recovery_worker()
                return (
                    "⏳ /auto armed.\n"
                    f"Binance cooldown: {self._binance_blackout_remaining()}s\n"
                    "Bybit/strategy tetap berjalan; auto akan resume sendiri."
                )
            log.exception("[AUTO] discovery failed")
            with self.run_lock:
                self.auto_running = False
                self.auto_armed = False
            return f"❌ /auto gagal: {exc}"

    def stop(self) -> None:
        with self.run_lock:
            self.auto_running = False
            self.auto_armed = False
            self._auto_pipeline_started = False
        self.stop_websocket()
        try:
            self.worker_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            log.exception("[WORKERS] shutdown failed")
        if self.strategy and hasattr(self.strategy, "shutdown"):
            try: self.strategy.shutdown()
            except Exception: log.exception("[STRATEGY] shutdown failed")
        self.trade_manager._save_state()
        if self.learning and hasattr(self.learning, "shutdown"):
            try: self.learning.shutdown()
            except Exception: log.exception("[LEARN] shutdown failed")
        elif self.learning and hasattr(self.learning, "_save_state"):
            try: self.learning._save_state()
            except Exception: log.exception("[LEARN] final save failed")
        log.info("[ENGINE] stopped")

    def get_symbols(self) -> list[str]:
        with self.symbol_lock: return list(self.symbols)

    def status(self) -> dict[str, Any]:
        return {
            "auto": self.auto_running, "bootstrap": self.bootstrap_complete, "symbols": len(self.symbols),
            "strategy": bool(self.strategy), "strategy_error": self.strategy_error, "ws": int(self.ws_connected),
            "ws_connections": 1 if self.ws_connected else 0,
            "mode": self.trade_manager.mode, "active": self.trade_manager.active_count(), "max": self.trade_manager.max_active,
            "auto_armed": self.auto_armed,
            "binance_cooldown": self._binance_blackout_remaining(),

        }


def _format_trade(t: Trade, header: str) -> str:
    return (
        f"{header}\n\n{t.symbol} {t.direction}\n"
        f"Mode: {t.mode}\nType: {t.order_type}\nStatus: {t.status}\n\n"
        f"📍 Entry: {t.entry_price:.8f}\n"
        f"🛑 SL: {t.stop_loss:.8f}\n"
        f"🎯 TP: {t.take_profit:.8f}\n"
        f"Qty: {t.quantity}\n"
        f"Margin: {t.margin_actual:.8f} / target {t.margin_target:.8f}\n"
        f"Leverage: {t.leverage}x\n"
        f"Trade ID: {t.id}"
    )


def _format_close(t: Trade) -> str:
    icon = "✅" if t.outcome == "TP" else "🛑"
    return (
        f"{icon} TRADE CLOSED\n\n{t.symbol} {t.direction}\n"
        f"Result: {t.outcome}\n"
        f"Entry: {float(t.actual_entry or t.entry_price):.8f}\n"
        f"Exit: {float(t.actual_exit or t.entry_price):.8f}\n"
        f"PnL: {float(t.pnl or 0):+.8f}\n"
        f"R: {float(t.r_multiple or 0):+.2f}\n\n"
        f"🚫 {t.symbol} banned for {BAN_HOURS:g}h\n"
        f"Trade ID: {t.id}"
    )


ENGINE: DataEngine | None = None


def on_start(context: dict[str, Any]) -> None:
    global ENGINE
    ENGINE = DataEngine(context)
    try: ip = ENGINE.public_ip()
    except Exception as exc: ip = f"unavailable ({exc})"
    log.info("[START] V14 ready | ip=%s | base=%s | strategy=%s", ip, BASE_DIR, STRATEGY_FILE)
    ENGINE._notify(
        f"🟢 START.PY V{VERSION} READY\n"
        f"Server IP: {ip}\n"
        f"Strategy target: {STRATEGY_FILE}\n"
        "Default mode: OFF (SIMULATION)\n\n"
        "/auto → start scanner"
    )


def on_stop(context: dict[str, Any]) -> None:
    global ENGINE
    if ENGINE: ENGINE.stop()
    ENGINE = None


def _help() -> str:
    return (
        "🤖 SMCAutoTrade V14\n\n"
        "/auto — ALL common pairs + historical + websocket pool\n"
        "/mode — show mode\n/mode on — REAL Binance\n/mode off — SIMULATION\n"
        "/margin 10 — target margin/trade\n/leverage 10 — leverage\n/max 5 — max active orders/positions\n"
        "/trade — active simulated/real orders\n/stats — closed trade statistics\n"
        "/banned — banned symbols\n/unban BTCUSDT — remove ban\n/resetban — clear all bans\n"
        "/reset — reload strategy.py using existing market data\n"
        "/open — restore learning memory\n/full — run autonomous learning cycle\n/learn — learning status\n/save — checkpoint learning memory\n"
        "/status — data + execution status\n"
    )


def handle_update(update: dict[str, Any], context: dict[str, Any]) -> str | None:
    engine = ENGINE
    if engine is None: return "❌ Data engine belum aktif."
    msg = update.get("message") or {}
    text = str(msg.get("text") or msg.get("caption") or "").strip()
    if not text: return None
    parts = text.split(); cmd = parts[0].split("@", 1)[0].lower(); args = parts[1:]

    try:
        if cmd in {"/help", "/start"}: return _help()
        if cmd == "/ip":
            return f"🌐 Server public IP\n{engine.public_ip()}"
        if cmd == "/auto": return engine.start_auto()
        if cmd == "/binance":
            return engine.trade_manager.binance_status() + f"\nBlackout extra: +{BINANCE_BLACKOUT_EXTRA}s"
        if cmd == "/status":
            s = engine.status()
            return (
                "📊 SYSTEM STATUS\n"
                f"AUTO: {s['auto']}\nBOOTSTRAP: {s['bootstrap']}\nSYMBOLS: {s['symbols']}\n"
                f"WS connections: {s['ws']}/runtime {s['ws_connections']}\nSTRATEGY: {s['strategy']}\nSTRATEGY ERROR: {s['strategy_error'] or '-'}\n"
                f"MODE: {s['mode']}\nACTIVE: {s['active']}/{s['max']}\n"
                f"LEARN: {'READY' if engine.learning else 'OFF'}"
            )
        if cmd == "/mode":
            return engine.trade_manager.set_mode(args[0]) if args else engine.trade_manager.mode_text()
        if cmd == "/margin":
            if not args: return f"💵 Margin/trade: {engine.trade_manager.margin}"
            engine.trade_manager.margin = float(args[0]); return f"✅ Margin set: {engine.trade_manager.margin:g}"
        if cmd == "/leverage":
            if not args: return f"⚡ Leverage: {engine.trade_manager.leverage}x"
            value = int(args[0]);
            if value < 1 or value > 125: return "❌ Leverage harus 1–125."
            engine.trade_manager.leverage = value; return f"✅ Leverage set: {value}x"
        if cmd == "/max":
            if not args: return f"📊 MAX ACTIVE: {engine.trade_manager.max_active}\nCurrent: {engine.trade_manager.active_count()}"
            value = int(args[0])
            if value < 0: return "❌ /max tidak boleh negatif."
            engine.trade_manager.max_active = value
            return f"✅ Max active orders/positions: {value}"
        if cmd == "/trade": return engine.trade_manager.trades_text()
        if cmd == "/stats": return engine.trade_manager.stats()
        if cmd == "/banned": return engine.trade_manager.bans_text()
        if cmd == "/unban": return engine.trade_manager.unban(args[0]) if args else "Format: /unban BTCUSDT"
        if cmd == "/resetban": return engine.trade_manager.reset_bans()
        if cmd == "/reset": return engine.reset_strategy()
        if cmd == "/symbols":
            syms = engine.get_symbols(); return f"🪙 Symbols: {len(syms)}\n" + ", ".join(syms[:100])
        if cmd == "/candles":
            if len(args) < 2: return "Format: /candles BTCUSDT 15 [limit]"
            sym, tf = args[0].upper(), args[1]; limit = min(int(args[2]) if len(args) > 2 else 10, 50)
            rows = engine.api.get_candles(sym, tf, limit)
            if not rows: return f"❌ Tidak ada data {sym} {tf}M"
            return f"📈 {sym} {tf}M\n" + "\n".join(f"{r['timestamp']} O={r['open']} H={r['high']} L={r['low']} C={r['close']}" for r in rows)
        if cmd == "/price":
            if not args: return "Format: /price BTCUSDT"
            p = engine.api.get_price(args[0].upper()); return f"💵 {args[0].upper()} = {p}" if p is not None else "❌ Harga belum tersedia."
        if cmd in {"/full", "/open", "/learn", "/save", "/learningreport"}:
            if not engine.learning or not hasattr(engine.learning, "handle_command"):
                return f"⚠️ Learn engine unavailable: {engine.learning_error or 'not loaded'}"
            if cmd == "/full" and not engine.auto_running:
                return "ℹ️ /full hanya boleh dijalankan saat /auto aktif."
            return engine.learning.handle_command(text)

        if engine.strategy and hasattr(engine.strategy, "handle_command"):
            result = engine.strategy.handle_command(text)
            return None if result is None else str(result)
        return None
    except Exception as exc:
        log.exception("[COMMAND] %s failed", cmd)
        engine._notify(f"🚨 HANDLER ERROR\nCommand: {cmd}\n{type(exc).__name__}: {exc}")
        return f"❌ Command gagal: {type(exc).__name__}: {exc}"
