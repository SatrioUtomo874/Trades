"""
main.py — infrastructure / orchestrator layer for the SMC/ICT auto-trading bot.

Compatible with try.py's launcher contract:
    async def on_start(context) -> None | False
    async def handle_update(update, context) -> None
    async def on_stop(context) -> None

main.py is the ONLY module that talks to Bybit, Binance, or Telegram.
strategy.py and learn.py never make external API calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import logging
import os
import time
import uuid
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests

import strategy
import learn

log = logging.getLogger("main")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
LOG_DIR = BASE_DIR / "logs"
STATE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MAIN_CKPT = STATE_DIR / "main_checkpoint.json"
MAIN_CKPT_BACKUP = STATE_DIR / "main_checkpoint.json.backup"

# ----------------------------------------------------------------------
# configuration (from .env, loaded by try.py's process-wide load_dotenv)
# ----------------------------------------------------------------------

def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


BINANCE_API_KEY = _env("BINANCE_API_KEY")
BINANCE_API_SECRET = _env("BINANCE_API_SECRET")
BINANCE_API_KEY_READ = _env("BINANCE_API_KEY_1") or BINANCE_API_KEY
BINANCE_API_SECRET_READ = _env("BINANCE_API_SECRET_1") or BINANCE_API_SECRET

GITHUB_TOKEN = _env("GITHUB_TOKEN")
REPO_NAME = _env("REPO_NAME")
GITHUB_BRANCH = _env("GITHUB_BRANCH", "main")

BINANCE_FAPI = "https://fapi.binance.com"
BYBIT_API = "https://api.bybit.com"

SCAN_TARGET_SYMBOLS = 50
CANDLES_PER_SYMBOL = 672  # ~7 days of M15
SCAN_PER_COIN_DELAY = 1.0
SCAN_INTERVAL_SEC = 120
MAX_POSITIONS = 20
TRAIL_QUEUE_DELAY = 2.0
WAITING_QUEUE_DELAY = 1.0
BINANCE_RATE_LIMIT_RECOVERY_SEC = 60

DEFAULT_SIM_BALANCE = 10.0
DEFAULT_MARGIN = 1.0
DEFAULT_LEVERAGE = 10

BAN_AFTER_TRADE_HOURS = 24
BAN_AFTER_TIMEOUT_HOURS = 12
BAN_LOW_CONFIDENCE_HOURS = 4
LOW_CONFIDENCE_BAN_MIN_THRESHOLD = 40.0

STRATEGY_VERSION_FALLBACK = "1.0.0"
MAIN_VERSION = "1.0.0"

STATE_LOCK = asyncio.Lock()
PY_EOF_MARK = True


# ----------------------------------------------------------------------
# rounding helpers (Decimal-based, per validated real-trade experience)
# ----------------------------------------------------------------------

def round_to_tick(value: float, tick: float, mode=ROUND_DOWN) -> float:
    if tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    steps = (d_value / d_tick).to_integral_value(rounding=mode)
    return float(steps * d_tick)


def round_step(value: float, step: float, mode=ROUND_DOWN) -> float:
    return round_to_tick(value, step, mode)


# ----------------------------------------------------------------------
# Binance USD-M Futures client
# ----------------------------------------------------------------------

class BinanceError(RuntimeError):
    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class BinanceClient:
    """Wraps signed/unsigned Binance USD-M Futures REST calls.

    TP/SL/Trailing conditional orders go through /fapi/v1/algoOrder — the
    mandatory Algo Service migration (effective 2025-12-09). The legacy
    /fapi/v1/order endpoint rejects STOP_MARKET/TAKE_PROFIT_MARKET with -4120.
    """

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self._banned_until = 0.0
        self._filters_cache: dict[str, dict] = {}
        self._filters_cache_at = 0.0

    # -- low level --

    def _sign(self, params: dict) -> dict:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params.setdefault("recvWindow", 10000)
        query = urlencode(params, doseq=True)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self.api_key}

    def _wait_if_banned(self) -> None:
        remaining = self._banned_until - time.time()
        if remaining > 0:
            time.sleep(min(remaining, 5))

    def _register_ban(self, message: str) -> None:
        # Binance embeds "banned until <ms epoch>" in -1003 error messages.
        import re
        match = re.search(r"banned until (\d+)", message)
        if match:
            self._banned_until = int(match.group(1)) / 1000.0
        else:
            self._banned_until = time.time() + BINANCE_RATE_LIMIT_RECOVERY_SEC

    def is_rate_limited(self) -> bool:
        return time.time() < self._banned_until

    def ready_in_seconds(self) -> float:
        return max(0.0, self._banned_until - time.time())

    def _request(self, method: str, path: str, signed: bool = False, params: Optional[dict] = None) -> Any:
        self._wait_if_banned()
        params = params or {}
        headers = {}
        if signed:
            params = self._sign(params)
            headers = self._headers()
        elif self.api_key:
            headers = self._headers()

        url = f"{BINANCE_FAPI}{path}"
        try:
            if method == "GET":
                resp = self.session.get(url, params=params, headers=headers, timeout=20)
            elif method == "POST":
                resp = self.session.post(url, data=params, headers=headers, timeout=20)
            elif method == "DELETE":
                resp = self.session.delete(url, params=params, headers=headers, timeout=20)
            else:
                raise ValueError(f"unsupported method {method}")
        except requests.RequestException as exc:
            raise BinanceError(f"network error: {exc}") from exc

        if resp.status_code == 418 or resp.status_code == 429:
            self._register_ban(resp.text)
            raise BinanceError(f"rate limited: {resp.text[:300]}", code=-1003)

        try:
            body = resp.json()
        except ValueError:
            raise BinanceError(f"invalid JSON response: {resp.text[:300]}")

        if isinstance(body, dict) and "code" in body and int(body["code"]) < 0:
            code = int(body["code"])
            msg = str(body.get("msg", ""))
            if code == -1003:
                self._register_ban(msg)
            raise BinanceError(f"{code}: {msg}", code=code)

        return body

    # -- market data / symbol info --

    def exchange_info(self, force: bool = False) -> dict:
        if not force and self._filters_cache and (time.time() - self._filters_cache_at) < 3600:
            return self._filters_cache
        data = self._request("GET", "/fapi/v1/exchangeInfo")
        filters: dict[str, dict] = {}
        for sym in data.get("symbols", []):
            f = {"symbol": sym["symbol"], "status": sym.get("status")}
            for flt in sym.get("filters", []):
                if flt["filterType"] == "PRICE_FILTER":
                    f["tickSize"] = float(flt["tickSize"])
                elif flt["filterType"] == "LOT_SIZE":
                    f["stepSize"] = float(flt["stepSize"])
                    f["minQty"] = float(flt["minQty"])
                elif flt["filterType"] == "MIN_NOTIONAL":
                    f["minNotional"] = float(flt.get("notional", flt.get("minNotional", 0)))
            filters[sym["symbol"]] = f
        self._filters_cache = filters
        self._filters_cache_at = time.time()
        return filters

    def get_filters(self, symbol: str) -> Optional[dict]:
        return self.exchange_info().get(symbol)

    def usable_symbols(self) -> set[str]:
        return {s for s, f in self.exchange_info().items() if f.get("status") == "TRADING"}

    def mark_price(self, symbol: str) -> float:
        data = self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol})
        return float(data["markPrice"])

    # -- account --

    def account_balance_usdt(self) -> float:
        data = self._request("GET", "/fapi/v2/balance", signed=True)
        for entry in data:
            if entry.get("asset") == "USDT":
                return float(entry.get("availableBalance", entry.get("balance", 0.0)))
        return 0.0

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._request("POST", "/fapi/v1/leverage", signed=True,
                              params={"symbol": symbol, "leverage": leverage})

    def position_risk(self, symbol: str) -> Optional[dict]:
        data = self._request("GET", "/fapi/v3/positionRisk", signed=True, params={"symbol": symbol})
        for p in data:
            if p.get("symbol") == symbol and abs(float(p.get("positionAmt", 0))) > 0:
                return p
        return None

    # -- orders --

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        return self._request("POST", "/fapi/v1/order", signed=True, params={
            "symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTC",
            "quantity": quantity, "price": price,
        })

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._request("DELETE", "/fapi/v1/order", signed=True,
                              params={"symbol": symbol, "orderId": order_id})

    def get_order(self, symbol: str, order_id: int) -> dict:
        return self._request("GET", "/fapi/v1/order", signed=True,
                              params={"symbol": symbol, "orderId": order_id})

    def market_close(self, symbol: str, side: str, quantity: float) -> dict:
        """side = the closing side (opposite of the position direction)."""
        return self._request("POST", "/fapi/v1/order", signed=True, params={
            "symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity, "reduceOnly": "true",
        })

    # -- algo (conditional TP/SL/trailing) orders --

    def place_algo_order(self, symbol: str, side: str, order_type: str, quantity: float,
                          trigger_price: float, close_position: bool = False,
                          reduce_only: bool = True) -> dict:
        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "triggerPrice": trigger_price,
            "workingType": "MARK_PRICE",
        }
        if close_position:
            params["closePosition"] = "true"
        else:
            params["quantity"] = quantity
            params["reduceOnly"] = "true" if reduce_only else "false"
        return self._request("POST", "/fapi/v1/algoOrder", signed=True, params=params)

    def cancel_algo_order(self, algo_id: int) -> dict:
        try:
            return self._request("DELETE", "/fapi/v1/algoOrder", signed=True, params={"algoId": algo_id})
        except BinanceError as exc:
            if exc.code in (-2011, -2013):  # unknown order -> already gone, treat as clean
                return {"already_clean": True}
            raise

    def query_algo_order(self, algo_id: int) -> dict:
        return self._request("GET", "/fapi/v1/algoOrder", signed=True, params={"algoId": algo_id})

    def open_algo_orders(self, symbol: str) -> list[dict]:
        return self._request("GET", "/fapi/v1/openAlgoOrders", signed=True, params={"symbol": symbol})


# ----------------------------------------------------------------------
# Bybit public REST client (market-data source per spec section 9)
# ----------------------------------------------------------------------

class BybitClient:
    def __init__(self):
        self.session = requests.Session()

    def _get(self, path: str, params: dict) -> dict:
        resp = self.session.get(f"{BYBIT_API}{path}", params=params, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        if body.get("retCode") not in (0, None):
            raise RuntimeError(f"Bybit {path}: {body.get('retMsg')}")
        return body

    def klines_m15(self, symbol: str, limit: int = CANDLES_PER_SYMBOL) -> list[dict]:
        candles: list[dict] = []
        end_time = None
        remaining = limit
        while remaining > 0:
            batch = min(remaining, 1000)
            params = {"category": "linear", "symbol": symbol, "interval": "15", "limit": batch}
            if end_time is not None:
                params["end"] = end_time
            body = self._get("/v5/market/kline", params)
            rows = body.get("result", {}).get("list", [])
            if not rows:
                break
            # Bybit returns newest-first: [start, open, high, low, close, volume, turnover]
            for r in rows:
                candles.append({
                    "timestamp": int(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                })
            remaining -= len(rows)
            end_time = int(rows[-1][0]) - 1
            if len(rows) < batch:
                break
        candles.sort(key=lambda c: c["timestamp"])
        return candles[-limit:]

    def top_volume_symbols(self, quote: str = "USDT", limit: int = 200) -> list[tuple[str, float]]:
        body = self._get("/v5/market/tickers", {"category": "linear"})
        rows = body.get("result", {}).get("list", [])
        out = []
        for r in rows:
            symbol = r.get("symbol", "")
            if not symbol.endswith(quote):
                continue
            try:
                turnover = float(r.get("turnover24h", 0.0))
            except (TypeError, ValueError):
                turnover = 0.0
            out.append((symbol, turnover))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:limit]


BYBIT = BybitClient()
BINANCE_EXEC = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET) if BINANCE_API_KEY and BINANCE_API_SECRET else None
BINANCE_READ = BinanceClient(BINANCE_API_KEY_READ, BINANCE_API_SECRET_READ) if BINANCE_API_KEY_READ and BINANCE_API_SECRET_READ else BINANCE_EXEC


# ----------------------------------------------------------------------
# quantity / geometry / PnL helpers
# ----------------------------------------------------------------------

def validate_geometry(direction: str, entry: float, sl: float, tp: float) -> bool:
    if entry <= 0 or sl <= 0 or tp <= 0:
        return False
    if direction == "BUY":
        return sl < entry < tp
    if direction == "SELL":
        return tp < entry < sl
    return False


def calc_auto_quantity(entry: float, margin_usd: float, leverage: int, filters: Optional[dict]) -> Optional[float]:
    """Quantity = margin*leverage / entry, normalized to exchange filters.
    If minNotional/minQty cannot be met, margin is allowed to scale up to
    max(margin*3, margin+$5) before giving up (documented cap from real-trade tuning)."""
    if entry <= 0 or leverage <= 0:
        return None
    step = (filters or {}).get("stepSize", 0.0) or 0.0
    min_qty = (filters or {}).get("minQty", 0.0) or 0.0
    min_notional = (filters or {}).get("minNotional", 5.0) or 5.0

    cap = max(margin_usd * 3, margin_usd + 5.0)
    trial_margin = margin_usd
    while trial_margin <= cap:
        notional = trial_margin * leverage
        qty = notional / entry
        if step > 0:
            qty = round_step(qty, step, ROUND_DOWN)
        if qty <= 0 or qty < min_qty or qty * entry < min_notional:
            trial_margin += max(margin_usd * 0.25, 0.25)
            continue
        return qty
    log.warning("calc_auto_quantity: unable to satisfy filters up to cap=%.4f for entry=%.8f", cap, entry)
    return None


def calc_pnl(direction: str, entry: float, exit_price: float, quantity: float) -> float:
    if direction == "BUY":
        return (exit_price - entry) * quantity
    return (entry - exit_price) * quantity


def calc_pnl_pct_of_margin(direction: str, entry: float, exit_price: float, margin_usd: float, leverage: int) -> float:
    if entry <= 0 or margin_usd <= 0:
        return 0.0
    move = (exit_price - entry) / entry if direction == "BUY" else (entry - exit_price) / entry
    return move * leverage * 100.0


def calc_r_multiple(direction: str, entry: float, exit_price: float, stop_loss: float) -> Optional[float]:
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return None
    if direction == "BUY":
        return (exit_price - entry) / risk
    return (entry - exit_price) / risk


# ----------------------------------------------------------------------
# runtime state + checkpoint persistence
# ----------------------------------------------------------------------

def _new_state() -> dict:
    return {
        "mode": "sim",                 # "sim" | "real"
        "sim_balance": DEFAULT_SIM_BALANCE,
        "anchor_balance": DEFAULT_SIM_BALANCE,
        "margin": DEFAULT_MARGIN,
        "leverage": DEFAULT_LEVERAGE,
        "max_positions": MAX_POSITIONS,
        "autostop_pct": None,
        "peak_balance": DEFAULT_SIM_BALANCE,
        "autostop_triggered": False,
        "scanning": False,
        "pending": {},                 # id -> pending order dict
        "positions": {},               # id -> filled position dict
        "bans": {},                    # symbol -> {until: epoch|None, permanent: bool, reason: str}
        "scanned_coins": [],           # cache for /koin
        "pending_cancel_stats": {"tp_before_entry": 0, "expired": 0, "binance_reject": 0},
        "confidence_threshold": 0.0,
        "chat_id": None,
        "banned_count_lifetime": 0,
        "binance_paused_until": 0.0,
    }


STATE: dict = _new_state()


def _atomic_write_json(path: Path, payload: dict) -> None:
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_main_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def save_state() -> None:
    try:
        _atomic_write_json(MAIN_CKPT, STATE)
        _atomic_write_json(MAIN_CKPT_BACKUP, STATE)
    except OSError:
        log.exception("failed to save main checkpoint")


def load_state() -> bool:
    global STATE
    for path in (MAIN_CKPT, MAIN_CKPT_BACKUP):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            merged = _new_state()
            merged.update(data)
            STATE = merged
            return True
        except (OSError, json.JSONDecodeError):
            continue
    return False


# ----------------------------------------------------------------------
# ban manager
# ----------------------------------------------------------------------

def is_banned(symbol: str) -> bool:
    entry = STATE["bans"].get(symbol)
    if not entry:
        return False
    if entry.get("permanent"):
        return True
    until = entry.get("until")
    return bool(until and until > time.time())


def apply_ban(symbol: str, hours: Optional[float], reason: str, permanent: bool = False) -> None:
    STATE["bans"][symbol] = {
        "until": None if permanent else time.time() + hours * 3600,
        "permanent": permanent,
        "reason": reason,
        "applied_at": time.time(),
    }
    STATE["banned_count_lifetime"] = STATE.get("banned_count_lifetime", 0) + 1


def unban(symbol: str) -> bool:
    return STATE["bans"].pop(symbol, None) is not None


def unban_all() -> int:
    n = len(STATE["bans"])
    STATE["bans"] = {}
    return n


def purge_expired_bans() -> list[str]:
    now = time.time()
    expired = []
    for symbol, entry in list(STATE["bans"].items()):
        if entry.get("permanent"):
            continue
        until = entry.get("until")
        if until and until <= now:
            expired.append(symbol)
            STATE["bans"].pop(symbol, None)
    return expired


def active_ban_count() -> int:
    return len(STATE["bans"])


# ----------------------------------------------------------------------
# notifications (captured from launcher context at on_start)
# ----------------------------------------------------------------------

_SEND_MESSAGE = None  # callable(chat_id:int, text:str)
_CHAT_ID: Optional[int] = None


def notify(text: str) -> None:
    if _SEND_MESSAGE and _CHAT_ID:
        try:
            _SEND_MESSAGE(_CHAT_ID, text)
        except Exception:
            log.exception("notify failed")


# ----------------------------------------------------------------------
# price feed (Bybit public WS ticker) with REST fallback
# ----------------------------------------------------------------------

PRICE_CACHE: dict[str, dict] = {}   # symbol -> {"price": float, "ts": float}
_WS_SYMBOLS: set[str] = set()
_WS_SYMBOLS_LOCK = asyncio.Lock()
PRICE_FRESH_SEC = 20


async def ws_track(symbol: str) -> None:
    async with _WS_SYMBOLS_LOCK:
        _WS_SYMBOLS.add(symbol)


async def ws_untrack(symbol: str) -> None:
    async with _WS_SYMBOLS_LOCK:
        _WS_SYMBOLS.discard(symbol)


def get_price(symbol: str) -> Optional[float]:
    entry = PRICE_CACHE.get(symbol)
    if entry and (time.time() - entry["ts"]) < PRICE_FRESH_SEC:
        return entry["price"]
    # REST fallback (rare — WS should normally be fresh)
    try:
        resp = requests.get(f"{BYBIT_API}/v5/market/tickers",
                             params={"category": "linear", "symbol": symbol}, timeout=10)
        rows = resp.json().get("result", {}).get("list", [])
        if rows:
            price = float(rows[0]["lastPrice"])
            PRICE_CACHE[symbol] = {"price": price, "ts": time.time()}
            return price
    except Exception:
        log.exception("REST price fallback failed for %s", symbol)
    return entry["price"] if entry else None


async def ws_feed_task(stop_flag: "asyncio.Event") -> None:
    """Maintains a Bybit public WS ticker subscription for all tracked symbols."""
    try:
        import websockets
    except ImportError:
        log.warning("websockets package not installed — falling back to REST-only price polling")
        while not stop_flag.is_set():
            async with _WS_SYMBOLS_LOCK:
                symbols = list(_WS_SYMBOLS)
            for sym in symbols:
                await asyncio.to_thread(get_price, sym)
            await asyncio.sleep(5)
        return

    backoff = 2
    while not stop_flag.is_set():
        try:
            async with websockets.connect("wss://stream.bybit.com/v5/public/linear", ping_interval=20) as ws:
                subscribed: set[str] = set()
                backoff = 2
                last_resync = 0.0
                while not stop_flag.is_set():
                    if time.time() - last_resync > 10:
                        async with _WS_SYMBOLS_LOCK:
                            desired = set(_WS_SYMBOLS)
                        to_add = desired - subscribed
                        to_remove = subscribed - desired
                        if to_add:
                            await ws.send(json.dumps({"op": "subscribe", "args": [f"tickers.{s}" for s in to_add]}))
                            subscribed |= to_add
                        if to_remove:
                            await ws.send(json.dumps({"op": "unsubscribe", "args": [f"tickers.{s}" for s in to_remove]}))
                            subscribed -= to_remove
                        last_resync = time.time()
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    topic = msg.get("topic", "")
                    if topic.startswith("tickers."):
                        data = msg.get("data") or {}
                        symbol = data.get("symbol") or topic.split(".", 1)[1]
                        price = data.get("lastPrice")
                        if symbol and price:
                            PRICE_CACHE[symbol] = {"price": float(price), "ts": time.time()}
        except Exception:
            log.warning("ws_feed_task reconnecting after error", exc_info=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ----------------------------------------------------------------------
# candle cache (Bybit REST) — used for scanning and trailing analysis
# ----------------------------------------------------------------------

_BTC_CANDLE_CACHE: dict = {"candles": [], "fetched_at": 0.0}
_RECENT_CANDLE_CACHE: dict[str, dict] = {}  # symbol -> {"candles":[...], "fetched_at": ts}
RECENT_CANDLE_TTL_SEC = 45


async def get_btc_candles() -> list[dict]:
    if time.time() - _BTC_CANDLE_CACHE["fetched_at"] < 300 and _BTC_CANDLE_CACHE["candles"]:
        return _BTC_CANDLE_CACHE["candles"]
    try:
        candles = await asyncio.to_thread(BYBIT.klines_m15, "BTCUSDT", CANDLES_PER_SYMBOL)
        _BTC_CANDLE_CACHE["candles"] = candles
        _BTC_CANDLE_CACHE["fetched_at"] = time.time()
        return candles
    except Exception:
        log.exception("failed to refresh BTC candles")
        return _BTC_CANDLE_CACHE["candles"]


async def get_recent_candles(symbol: str, limit: int = 120) -> list[dict]:
    cached = _RECENT_CANDLE_CACHE.get(symbol)
    if cached and (time.time() - cached["fetched_at"]) < RECENT_CANDLE_TTL_SEC:
        return cached["candles"]
    try:
        candles = await asyncio.to_thread(BYBIT.klines_m15, symbol, limit)
        _RECENT_CANDLE_CACHE[symbol] = {"candles": candles, "fetched_at": time.time()}
        return candles
    except Exception:
        log.exception("failed to refresh recent candles for %s", symbol)
        return cached["candles"] if cached else []


# ----------------------------------------------------------------------
# top-50 scanner universe
# ----------------------------------------------------------------------

async def build_scan_universe() -> list[str]:
    """Top-volume symbols present on both Bybit and Binance, excluding banned/active/pending,
    padded up to SCAN_TARGET_SYMBOLS. BTCUSDT is always included when available."""
    try:
        bybit_ranked = await asyncio.to_thread(BYBIT.top_volume_symbols, "USDT", 300)
    except Exception:
        log.exception("failed to fetch Bybit top-volume list")
        return []

    try:
        binance_symbols = await asyncio.to_thread((BINANCE_READ or BINANCE_EXEC).usable_symbols) \
            if (BINANCE_READ or BINANCE_EXEC) else set()
    except Exception:
        log.exception("failed to fetch Binance exchange info")
        binance_symbols = set()

    occupied_symbols = {v["symbol"] for v in STATE["pending"].values()} | \
                       {v["symbol"] for v in STATE["positions"].values()}

    eligible: list[str] = []

    def usable(sym: str) -> bool:
        if binance_symbols and sym not in binance_symbols:
            return False
        if is_banned(sym):
            return False
        if sym in occupied_symbols:
            return False
        return True

    if usable("BTCUSDT"):
        eligible.append("BTCUSDT")

    for sym, _turnover in bybit_ranked:
        if len(eligible) >= SCAN_TARGET_SYMBOLS:
            break
        if sym in eligible:
            continue
        if usable(sym):
            eligible.append(sym)

    return eligible[:SCAN_TARGET_SYMBOLS]


def capacity_remaining() -> int:
    used = len(STATE["pending"]) + len(STATE["positions"])
    return max(0, STATE["max_positions"] - used)


def is_binance_ready() -> bool:
    if not BINANCE_EXEC:
        return True  # no real client configured -> only sim mode matters
    return not BINANCE_EXEC.is_rate_limited()


# ----------------------------------------------------------------------
# pending order creation (sim + real)
# ----------------------------------------------------------------------

PENDING_TIMEOUT_HOURS = 8.0


async def open_pending(decision: dict) -> Optional[dict]:
    symbol = decision["pair"]
    direction = decision["direction"]
    entry = float(decision["entry"])
    tp = float(decision["take_profit"])
    sl = float(decision["stop_loss"])

    if not validate_geometry(direction, entry, sl, tp):
        return None

    pending_id = uuid.uuid4().hex[:12]
    record = {
        "id": pending_id,
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "take_profit": tp,
        "stop_loss": sl,
        "confidence": decision.get("confidence"),
        "risk_reward": decision.get("risk_reward"),
        "setup_type": decision.get("setup_type"),
        "market_regime": decision.get("market_regime"),
        "feature_scores": decision.get("feature_scores"),
        "strategy_version": decision.get("strategy_version"),
        "created_at": time.time(),
        "expires_at": time.time() + PENDING_TIMEOUT_HOURS * 3600,
        "mode": STATE["mode"],
        "margin": STATE["margin"],
        "leverage": STATE["leverage"],
        "state": "PENDING",
        "binance_order_id": None,
        "last_binance_poll": 0.0,
    }

    if STATE["mode"] == "real":
        if not BINANCE_EXEC:
            notify("⚠️ Mode real aktif tapi BINANCE_API_KEY/SECRET belum diset. Order dilewati.")
            return None
        filters = BINANCE_EXEC.get_filters(symbol)
        if not filters:
            return None
        tick = filters.get("tickSize", 0.0) or 0.0
        entry_r = round_to_tick(entry, tick) if tick else entry
        qty = calc_auto_quantity(entry_r, STATE["margin"], STATE["leverage"], filters)
        if not qty:
            notify(f"⚠️ Gagal hitung quantity untuk {symbol} (margin/filter tidak cocok).")
            return None
        try:
            await asyncio.to_thread(BINANCE_EXEC.set_leverage, symbol, STATE["leverage"])
            side = "BUY" if direction == "BUY" else "SELL"
            order = await asyncio.to_thread(BINANCE_EXEC.place_limit_order, symbol, side, qty, entry_r)
        except BinanceError as exc:
            notify(f"❌ Binance order gagal untuk {symbol}: <code>{html.escape(str(exc))}</code>")
            return None
        record["binance_order_id"] = order.get("orderId")
        record["quantity"] = qty
        record["entry"] = entry_r

    STATE["pending"][pending_id] = record
    ws_track_sync(symbol)
    learn.record_candidate({**decision, "accepted": True, "pending_id": pending_id})
    return record


def ws_track_sync(symbol: str) -> None:
    _WS_SYMBOLS.add(symbol)


def ws_untrack_sync(symbol: str) -> None:
    still_needed = any(v["symbol"] == symbol for v in STATE["pending"].values()) or \
                   any(v["symbol"] == symbol for v in STATE["positions"].values())
    if not still_needed:
        _WS_SYMBOLS.discard(symbol)


# ----------------------------------------------------------------------
# scan cycle
# ----------------------------------------------------------------------

def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "—"


async def scan_once(should_stop) -> None:
    universe = await build_scan_universe()
    STATE["scanned_coins"] = [{"symbol": s, "scanned_at": time.time()} for s in universe]
    if not universe:
        return

    btc_candles = await get_btc_candles()
    threshold = STATE["confidence_threshold"]

    eligible: list[dict] = []
    rejects: dict[str, int] = {}
    confidences: list[float] = []
    buy_count = 0
    sell_count = 0
    regimes: dict[str, int] = {}
    low_conf_bans = 0
    processed = 0

    for symbol in universe:
        if should_stop():
            break
        processed += 1
        try:
            candles = await asyncio.to_thread(BYBIT.klines_m15, symbol, CANDLES_PER_SYMBOL)
        except Exception:
            log.exception("scan: failed to fetch candles for %s", symbol)
            await asyncio.sleep(SCAN_PER_COIN_DELAY)
            continue

        context = {"symbol": symbol, "btc_candles": btc_candles, "active_threshold": threshold}
        decision = None
        try:
            decision = strategy.analyze(candles, context)
        except Exception:
            log.exception("strategy.analyze failed for %s", symbol)

        if decision is None:
            rejects["NO_VALID_ENTRY_CANDIDATE"] = rejects.get("NO_VALID_ENTRY_CANDIDATE", 0) + 1
            await asyncio.sleep(SCAN_PER_COIN_DELAY)
            continue

        confidences.append(decision["confidence"])
        regimes[decision["market_regime"]] = regimes.get(decision["market_regime"], 0) + 1
        if decision["direction"] == "BUY":
            buy_count += 1
        else:
            sell_count += 1

        if decision["confidence"] < threshold:
            rejects["BELOW_ACTIVE_THRESHOLD"] = rejects.get("BELOW_ACTIVE_THRESHOLD", 0) + 1
            learn.record_candidate({**decision, "accepted": False, "rejected_reason": "BELOW_ACTIVE_THRESHOLD"})
            if threshold >= LOW_CONFIDENCE_BAN_MIN_THRESHOLD:
                apply_ban(symbol, BAN_LOW_CONFIDENCE_HOURS, "low_confidence")
                low_conf_bans += 1
            await asyncio.sleep(SCAN_PER_COIN_DELAY)
            continue

        eligible.append(decision)
        await asyncio.sleep(SCAN_PER_COIN_DELAY)

    accepted = 0
    for decision in eligible:
        if capacity_remaining() <= 0:
            break
        record = await open_pending(decision)
        if record:
            accepted += 1

    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
    total_breadth = buy_count + sell_count
    buy_pct = round(buy_count / total_breadth * 100, 1) if total_breadth else 0.0
    sell_pct = round(100 - buy_pct, 1) if total_breadth else 0.0
    top_regime = max(regimes.items(), key=lambda kv: kv[1])[0] if regimes else "unknown"

    summary = {
        "requested": len(universe),
        "available": len(universe),
        "processed": processed,
        "valid_analyses": len(confidences),
        "avg_confidence": avg_conf,
        "candidate_count": len(eligible),
        "eligible_count": accepted,
        "low_confidence_bans": low_conf_bans,
        "rejects": rejects,
        "buy_pct": buy_pct,
        "sell_pct": sell_pct,
        "regime": top_regime,
        "threshold": threshold,
    }
    learn.record_scan(summary)

    lines = [f"✅ <b>{accepted} DECISION BRAIN ELIGIBLE</b>", ""]
    for d in eligible[:accepted]:
        lines.append(f"• {d['pair']} {d['direction']} — {d['confidence']:.0f}%")
    lines += [
        "",
        "📊 <b>Scan:</b>",
        f"{len(universe)} requested",
        f"{len(universe)} available",
        f"{processed} processed",
        f"{len(confidences)} valid strategy analyses",
        "",
        f"🧠 Average confidence: {avg_conf}%",
        "",
        f"🎯 Candidate: {len(eligible)}",
        f"Eligible: {accepted}",
        f"Low-confidence bans: {low_conf_bans}",
        "",
        "🚫 Main rejects:",
    ]
    for reason, count in rejects.items():
        lines.append(f"{reason} = {count}")
    lines += [
        "",
        "📈 Breadth:",
        f"BUY {buy_pct}%",
        f"SELL {sell_pct}%",
        "",
        f"Regime: {top_regime}",
    ]
    notify("\n".join(lines))
    save_state()


async def scanner_loop(stop_flag: "asyncio.Event") -> None:
    last_cycle = 0.0
    while not stop_flag.is_set():
        try:
            if STATE["scanning"] and is_binance_ready() and capacity_remaining() > 0:
                if time.time() - last_cycle >= SCAN_INTERVAL_SEC:
                    last_cycle = time.time()

                    def should_stop():
                        return stop_flag.is_set() or not STATE["scanning"]

                    await scan_once(should_stop)
        except Exception:
            log.exception("scanner_loop iteration failed")
        await asyncio.sleep(2)


# ----------------------------------------------------------------------
# real-mode protection (TP/SL algo orders) + emergency close
# ----------------------------------------------------------------------

REAL_TRADE_POLL_SLEEP = 30.0  # steady per-item Binance poll cadence (rate-limit lesson learned)


async def _install_protection_real(pos: dict) -> bool:
    symbol = pos["symbol"]
    close_side = "SELL" if pos["direction"] == "BUY" else "BUY"
    qty = pos["quantity"]
    for attempt in range(3):
        try:
            tp_order = await asyncio.to_thread(
                BINANCE_EXEC.place_algo_order, symbol, close_side, "TAKE_PROFIT_MARKET", qty, pos["take_profit"])
            sl_order = await asyncio.to_thread(
                BINANCE_EXEC.place_algo_order, symbol, close_side, "STOP_MARKET", qty, pos["stop_loss"])
            pos["tp_algo_id"] = tp_order.get("algoId")
            pos["sl_algo_id"] = sl_order.get("algoId")
            return True
        except BinanceError:
            log.exception("place TP/SL attempt %s failed for %s", attempt + 1, symbol)
            await asyncio.sleep(2)
    return False


async def _emergency_close_real(pos: dict, reason: str) -> None:
    symbol = pos["symbol"]
    try:
        risk = await asyncio.to_thread(BINANCE_EXEC.position_risk, symbol)
        qty = abs(float(risk["positionAmt"])) if risk else pos.get("quantity", 0.0)
        if qty > 0:
            close_side = "SELL" if pos["direction"] == "BUY" else "BUY"
            await asyncio.to_thread(BINANCE_EXEC.market_close, symbol, close_side, qty)
        notify(f"🚨 <b>AUTO-OUT {symbol}</b>\nAlasan: {html.escape(reason)}")
    except BinanceError as exc:
        notify(f"❌ Emergency close GAGAL untuk {symbol}: <code>{html.escape(str(exc))}</code>\n"
               f"⚠️ Posisi mungkin masih terbuka tanpa proteksi — cek manual di Binance.")


async def _replace_sl_real(pos: dict, new_sl: float) -> bool:
    symbol = pos["symbol"]
    close_side = "SELL" if pos["direction"] == "BUY" else "BUY"
    old_algo_id = pos.get("sl_algo_id")
    try:
        if old_algo_id:
            await asyncio.to_thread(BINANCE_EXEC.cancel_algo_order, old_algo_id)
        new_order = await asyncio.to_thread(
            BINANCE_EXEC.place_algo_order, symbol, close_side, "STOP_MARKET", pos["quantity"], new_sl)
        pos["sl_algo_id"] = new_order.get("algoId")
        pos["stop_loss"] = new_sl
        return True
    except BinanceError:
        log.exception("SL replace failed for %s — keeping previous SL", symbol)
        return False


# ----------------------------------------------------------------------
# closing / recording trades
# ----------------------------------------------------------------------

def _session_and_regime(pos: dict) -> tuple[str, str]:
    ts_ms = int(pos.get("opened_at", time.time()) * 1000)
    return strategy.session_of(ts_ms), pos.get("market_regime", "unknown")


async def _finalize_trade(pos: dict, result: str, exit_price: float, mode: str) -> None:
    direction = pos["direction"]
    entry = pos["entry"]
    margin = pos.get("margin", DEFAULT_MARGIN)
    leverage = pos.get("leverage", DEFAULT_LEVERAGE)
    quantity = pos.get("quantity") or (margin * leverage / entry if entry else 0.0)

    pnl_usd = calc_pnl(direction, entry, exit_price, quantity) if mode == "real" else \
        margin * (calc_pnl_pct_of_margin(direction, entry, exit_price, margin, leverage) / 100.0)
    pnl_pct = calc_pnl_pct_of_margin(direction, entry, exit_price, margin, leverage)
    r_multiple = calc_r_multiple(direction, entry, exit_price, pos.get("initial_stop_loss", pos["stop_loss"]))
    session, regime = _session_and_regime(pos)

    if mode == "sim":
        STATE["sim_balance"] = round(STATE["sim_balance"] + pnl_usd, 6)

    trade = {
        "symbol": pos["symbol"],
        "direction": direction,
        "mode": mode,
        "entry": entry,
        "exit_price": exit_price,
        "result": result,
        "r_multiple": r_multiple,
        "pnl_pct": round(pnl_pct, 4),
        "pnl_usd": round(pnl_usd, 6),
        "confidence": pos.get("confidence"),
        "session": session,
        "market_regime": regime,
        "setup_type": pos.get("setup_type"),
        "feature_scores": pos.get("feature_scores"),
        "opened_at": int(pos.get("opened_at", time.time()) * 1000),
        "closed_at": int(time.time() * 1000),
        "trail_count": pos.get("trail_count", 0),
    }
    learn.record_trade(trade)
    apply_ban(pos["symbol"], BAN_AFTER_TRADE_HOURS, f"post_trade_{result}")

    icon = "🟢" if pnl_usd >= 0 else "🔴"
    label = {"tp": "TP", "initial_sl": "SL", "trail": "TRAIL"}.get(result, result.upper())
    balance_txt = f"${STATE['sim_balance']:.4f}" if mode == "sim" else "(real)"
    notify(f"{icon} <b>{label} — {pos['symbol']}</b>\n"
           f"Entry: {entry} → Exit: {exit_price}\n"
           f"PnL: {pnl_pct:+.2f}% → {balance_txt}\n"
           f"Confidence: {pos.get('confidence')}%")


async def close_position(pos_id: str, result: str, exit_price: float) -> None:
    pos = STATE["positions"].pop(pos_id, None)
    if not pos:
        return
    await _finalize_trade(pos, result, exit_price, pos["mode"])
    ws_untrack_sync(pos["symbol"])
    save_state()


async def cancel_pending(pending_id: str, reason: str) -> None:
    """reason: 'manual' | 'auto_timeout' | 'tp_before_entry' | 'binance_reject'"""
    pending = STATE["pending"].pop(pending_id, None)
    if not pending:
        return
    symbol = pending["symbol"]

    if pending.get("mode") == "real" and pending.get("binance_order_id") and BINANCE_EXEC:
        try:
            await asyncio.to_thread(BINANCE_EXEC.cancel_order, symbol, pending["binance_order_id"])
        except BinanceError as exc:
            if not (exc.code in (-2011, -2013)):  # already gone -> fine
                log.warning("cancel_order failed for %s: %s", symbol, exc)

    if reason == "manual":
        pass  # administrative cleanup: never sent to learn.py
    else:
        bucket = "expired" if reason == "auto_timeout" else reason
        if bucket in STATE["pending_cancel_stats"]:
            STATE["pending_cancel_stats"][bucket] += 1
        if reason in ("auto_timeout", "tp_before_entry"):
            apply_ban(symbol, BAN_AFTER_TIMEOUT_HOURS, reason)
            learn.record_timeout({
                "symbol": symbol, "direction": pending["direction"], "confidence": pending.get("confidence"),
                "reason": reason, "entry": pending["entry"], "take_profit": pending["take_profit"],
                "stop_loss": pending["stop_loss"],
            })
            notify(f"⏱️ <b>TIMEOUT — {symbol}</b>\nAlasan: {reason}\nDiban {BAN_AFTER_TIMEOUT_HOURS} jam.")

    ws_untrack_sync(symbol)
    save_state()


# ----------------------------------------------------------------------
# pending -> filled transition
# ----------------------------------------------------------------------

async def _promote_to_position(pending: dict, actual_entry: float) -> None:
    pos_id = pending["id"]
    pos = dict(pending)
    pos["entry"] = actual_entry
    pos["initial_stop_loss"] = pending["stop_loss"]
    pos["opened_at"] = time.time()
    pos["best_price"] = actual_entry
    pos["trail_count"] = 0
    pos["sl_replace_count"] = 0
    pos["state"] = "FILLED"
    pos["last_binance_poll"] = 0.0
    pos["last_trail_check"] = 0.0

    if pos["mode"] == "real":
        protected = await _install_protection_real(pos)
        if not protected:
            notify(f"🚨 Gagal pasang TP/SL untuk {pos['symbol']} setelah 3x percobaan — menutup posisi demi keamanan.")
            await _emergency_close_real(pos, "place_tp_sl gagal 3x berturut-turut")
            STATE["pending"].pop(pos_id, None)
            ws_untrack_sync(pos["symbol"])
            return
        pos["state"] = "PROTECTED"

    STATE["pending"].pop(pos_id, None)
    STATE["positions"][pos_id] = pos
    notify(f"✅ <b>FILLED — {pos['symbol']}</b>\n{pos['direction']} @ {actual_entry}\n"
           f"TP: {pos['take_profit']} | SL: {pos['stop_loss']}\nConfidence: {pos.get('confidence')}%")
    save_state()


async def _process_pending_sim(pid: str, pending: dict) -> None:
    symbol = pending["symbol"]
    price = get_price(symbol)
    if price is None:
        return
    direction = pending["direction"]
    entry, tp = pending["entry"], pending["take_profit"]

    tp_hit_first = (direction == "BUY" and price >= tp) or (direction == "SELL" and price <= tp)
    if tp_hit_first:
        await cancel_pending(pid, "tp_before_entry")
        return

    if time.time() >= pending["expires_at"]:
        await cancel_pending(pid, "auto_timeout")
        return

    filled = (direction == "BUY" and price <= entry) or (direction == "SELL" and price >= entry)
    if filled:
        await _promote_to_position(pending, entry)


async def _process_pending_real(pid: str, pending: dict) -> None:
    symbol = pending["symbol"]
    now = time.time()

    price = get_price(symbol)
    if price is not None:
        direction, tp = pending["direction"], pending["take_profit"]
        tp_hit_first = (direction == "BUY" and price >= tp) or (direction == "SELL" and price <= tp)
        if tp_hit_first:
            await cancel_pending(pid, "tp_before_entry")
            return

    if now >= pending["expires_at"]:
        await cancel_pending(pid, "auto_timeout")
        return

    if now - pending.get("last_binance_poll", 0) < REAL_TRADE_POLL_SLEEP:
        return
    pending["last_binance_poll"] = now
    if not BINANCE_EXEC or not pending.get("binance_order_id"):
        return
    try:
        order = await asyncio.to_thread(BINANCE_EXEC.get_order, symbol, pending["binance_order_id"])
    except BinanceError:
        log.exception("get_order failed for %s", symbol)
        return

    status = order.get("status")
    if status == "FILLED":
        actual_entry = float(order.get("avgPrice") or pending["entry"])
        if not validate_geometry(pending["direction"], actual_entry, pending["stop_loss"], pending["take_profit"]):
            notify(f"⚠️ Geometri invalid setelah fill {symbol} (slippage) — AUTO-OUT.")
            await cancel_pending(pid, "binance_reject")
            return
        await _promote_to_position(pending, actual_entry)
    elif status in ("CANCELED", "EXPIRED", "REJECTED"):
        STATE["pending"].pop(pid, None)
        ws_untrack_sync(symbol)
        STATE["pending_cancel_stats"]["binance_reject"] += 1
        save_state()


async def _process_position_sim(pid: str, pos: dict) -> None:
    symbol = pos["symbol"]
    price = get_price(symbol)
    if price is None:
        return
    direction = pos["direction"]
    pos["best_price"] = max(pos["best_price"], price) if direction == "BUY" else min(pos["best_price"], price)

    sl_hit = (direction == "BUY" and price <= pos["stop_loss"]) or (direction == "SELL" and price >= pos["stop_loss"])
    tp_hit = (direction == "BUY" and price >= pos["take_profit"]) or (direction == "SELL" and price <= pos["take_profit"])

    if sl_hit:
        result = "trail" if pos["trail_count"] > 0 else "initial_sl"
        await close_position(pid, result, pos["stop_loss"])
        return
    if tp_hit:
        await close_position(pid, "tp", pos["take_profit"])
        return

    now = time.time()
    if now - pos.get("last_trail_check", 0) < 30:
        return
    pos["last_trail_check"] = now
    candles = await get_recent_candles(symbol)
    if not candles:
        return
    trail = strategy.update_position(
        {"candles": candles},
        {"direction": direction, "entry": pos["entry"], "current_sl": pos["stop_loss"],
         "current_tp": pos["take_profit"], "best_price": pos["best_price"]},
    )
    if trail and trail.get("new_sl"):
        pos["stop_loss"] = trail["new_sl"]
        pos["trail_count"] += 1
        pos["state"] = "TRAILING"
        notify(_format_trail_notice(pos, trail))


async def _process_position_real(pid: str, pos: dict) -> None:
    symbol = pos["symbol"]
    price = get_price(symbol)
    if price is not None:
        direction = pos["direction"]
        pos["best_price"] = max(pos["best_price"], price) if direction == "BUY" else min(pos["best_price"], price)

    now = time.time()
    if now - pos.get("last_binance_poll", 0) < REAL_TRADE_POLL_SLEEP:
        return
    pos["last_binance_poll"] = now

    try:
        risk = await asyncio.to_thread(BINANCE_EXEC.position_risk, symbol)
    except BinanceError:
        log.exception("position_risk failed for %s", symbol)
        return

    if risk is None:
        # position closed by exchange — classify which algo order triggered
        result = "initial_sl"
        exit_price = pos["stop_loss"]
        try:
            if pos.get("tp_algo_id"):
                tp_status = await asyncio.to_thread(BINANCE_EXEC.query_algo_order, pos["tp_algo_id"])
                if tp_status.get("algoStatus") == "TRIGGERED":
                    result = "tp"
                    exit_price = pos["take_profit"]
            if result == "initial_sl" and pos.get("trail_count", 0) > 0:
                result = "trail"
        except BinanceError:
            log.exception("query_algo_order failed while classifying close for %s", symbol)
        await close_position(pid, result, exit_price)
        return

    # health-check: verify SL algo order is still live every poll cycle
    sl_algo_id = pos.get("sl_algo_id")
    if sl_algo_id:
        try:
            sl_status = await asyncio.to_thread(BINANCE_EXEC.query_algo_order, sl_algo_id)
            if sl_status.get("algoStatus") not in ("NEW",):
                pos["sl_replace_count"] = pos.get("sl_replace_count", 0) + 1
                if pos["sl_replace_count"] > 3:
                    await _emergency_close_real(pos, "SL hilang berulang kali — circuit breaker")
                    STATE["positions"].pop(pid, None)
                    ws_untrack_sync(symbol)
                    return
                ok = await _replace_sl_real(pos, pos["stop_loss"])
                notify(f"{'🔁' if ok else '⚠️'} SL {'dipasang ulang' if ok else 'GAGAL dipasang ulang'} untuk {symbol}")
        except BinanceError:
            log.exception("SL health-check failed for %s", symbol)

    if now - pos.get("last_trail_check", 0) < 30:
        return
    pos["last_trail_check"] = now
    candles = await get_recent_candles(symbol)
    if not candles:
        return
    trail = strategy.update_position(
        {"candles": candles},
        {"direction": pos["direction"], "entry": pos["entry"], "current_sl": pos["stop_loss"],
         "current_tp": pos["take_profit"], "best_price": pos["best_price"]},
    )
    if trail and trail.get("new_sl"):
        ok = await _replace_sl_real(pos, trail["new_sl"])
        if ok:
            pos["trail_count"] += 1
            pos["state"] = "TRAILING"
            notify(_format_trail_notice(pos, trail))


def _format_trail_notice(pos: dict, trail: dict) -> str:
    return (
        f"🔒 <b>TRAILING UPDATE — {pos['symbol']}</b>\n\n"
        f"Direction: {pos['direction']}\n"
        f"State: {trail['state']}\n\n"
        f"Entry: {trail['entry']}\n"
        f"Price: {trail['current_price']}\n\n"
        f"SL: {trail['old_sl']} → {trail['new_sl']}\n"
        f"Profit: {trail['profit_r']:+.2f}R\n"
        f"TP: {trail['tp']}\n\n"
        f"ATR M15: {trail['atr']}\n"
        f"Weakness Score: {trail['weakness_score']}\n"
        f"Engine: {trail['engine']}\n\n"
        "Reasons:\n" + "\n".join(f"• {r}" for r in trail["reason_codes"])
    )


# ----------------------------------------------------------------------
# monitor loop (pending fills, active positions, ban expiry, autostop)
# ----------------------------------------------------------------------

async def monitor_loop(stop_flag: "asyncio.Event") -> None:
    while not stop_flag.is_set():
        try:
            expired = purge_expired_bans()
            if expired:
                log.info("bans expired: %s", expired)

            for pid, pending in list(STATE["pending"].items()):
                try:
                    if pending.get("mode") == "real":
                        await _process_pending_real(pid, pending)
                    else:
                        await _process_pending_sim(pid, pending)
                except Exception:
                    log.exception("pending processing failed for %s", pid)
                await asyncio.sleep(0.05)

            for pid, pos in list(STATE["positions"].items()):
                try:
                    if pos.get("mode") == "real":
                        await _process_position_real(pid, pos)
                    else:
                        await _process_position_sim(pid, pos)
                except Exception:
                    log.exception("position processing failed for %s", pid)
                await asyncio.sleep(0.05)

            await _check_autostop()
            await _check_binance_recovery()
        except Exception:
            log.exception("monitor_loop iteration failed")
        await asyncio.sleep(3)


async def _check_binance_recovery() -> None:
    if not BINANCE_EXEC:
        return
    if BINANCE_EXEC.is_rate_limited():
        if STATE.get("binance_paused_until", 0) == 0:
            notify(f"⏸️ Binance API dibatasi sementara. Pemulihan dalam ~{BINANCE_EXEC.ready_in_seconds():.0f}s.")
        STATE["binance_paused_until"] = BINANCE_EXEC._banned_until
    elif STATE.get("binance_paused_until", 0):
        STATE["binance_paused_until"] = 0.0
        notify("✅ Binance API sudah pulih (READY).")


async def _current_balance() -> float:
    if STATE["mode"] == "real" and BINANCE_EXEC:
        try:
            return await asyncio.to_thread(BINANCE_EXEC.account_balance_usdt)
        except BinanceError:
            log.exception("failed to fetch real balance for autostop")
            return STATE.get("peak_balance", DEFAULT_SIM_BALANCE)
    return STATE["sim_balance"]


async def _check_autostop() -> None:
    if STATE.get("autostop_pct") is None or STATE.get("autostop_triggered"):
        return
    balance = await _current_balance()
    STATE["peak_balance"] = max(STATE.get("peak_balance", balance), balance)
    peak = STATE["peak_balance"]
    if peak <= 0:
        return
    drawdown_pct = (peak - balance) / peak * 100
    if drawdown_pct >= STATE["autostop_pct"]:
        STATE["scanning"] = False
        STATE["autostop_triggered"] = True
        notify(f"🛑 <b>AUTO STOP</b>\nDrawdown {drawdown_pct:.2f}% dari peak ${peak:.4f}.\n"
               f"Scanner dihentikan. WebSocket & proteksi posisi tetap aktif.\nGunakan /auto untuk reset & lanjut.")
        save_state()


# ----------------------------------------------------------------------
# learn autosave (+ optional GitHub sync of the checkpoint only)
# ----------------------------------------------------------------------

def _github_headers() -> dict:
    return {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}


def _github_push_learn_checkpoint(content: bytes) -> None:
    if not (GITHUB_TOKEN and REPO_NAME):
        return
    import base64
    path = "state/learn_checkpoint.json"
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{path}"
    try:
        get_resp = requests.get(url, headers=_github_headers(), params={"ref": GITHUB_BRANCH}, timeout=20)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
        payload = {
            "message": "learn.py autosave",
            "content": base64.b64encode(content).decode("ascii"),
            "branch": GITHUB_BRANCH,
        }
        if sha:
            payload["sha"] = sha
        requests.put(url, headers=_github_headers(), json=payload, timeout=30)
    except Exception:
        log.exception("GitHub autosave of learn checkpoint failed (non-fatal)")


async def autosave_loop(stop_flag: "asyncio.Event") -> None:
    while not stop_flag.is_set():
        try:
            saved = learn.maybe_autosave(min_interval_sec=120)
            if saved:
                save_state()
                data = learn.get_checkpoint_bytes_for_sync()
                if data:
                    await asyncio.to_thread(_github_push_learn_checkpoint, data)
        except Exception:
            log.exception("autosave_loop iteration failed")
        await asyncio.sleep(60)


# ----------------------------------------------------------------------
# formatting helpers for Telegram output
# ----------------------------------------------------------------------

def get_public_ip() -> str:
    try:
        resp = requests.get("https://api.ipify.org", timeout=10)
        return resp.text.strip()
    except Exception:
        return "unknown"


def _fmt_trade_dashboard() -> str:
    total = len(STATE["pending"]) + len(STATE["positions"])
    lines = [f"📡 <b>ACTIVE POSITIONS ({total}/{STATE['max_positions']})</b>", ""]
    for pos in STATE["positions"].values():
        price = get_price(pos["symbol"]) or pos["entry"]
        pnl_pct = calc_pnl_pct_of_margin(pos["direction"], pos["entry"], price, pos.get("margin", 1.0), pos.get("leverage", 1))
        icon = "🟢" if pos["direction"] == "BUY" else "🔴"
        lines += [
            f"{icon} {pos['symbol']} — {pos['state']}",
            f"Entry: {pos['entry']}",
            f"Price: {price}",
            f"TP: {pos['take_profit']}",
            f"SL: {pos['stop_loss']}",
            f"Confidence: {pos.get('confidence')}%",
            f"PnL: {pnl_pct:+.2f}%",
            "",
        ]
    for pend in STATE["pending"].values():
        price = get_price(pend["symbol"])
        dist = None
        if price:
            dist = abs(price - pend["entry"]) / price * 100
        icon = "🟢" if pend["direction"] == "BUY" else "🔴"
        lines += [
            f"⏳ {pend['symbol']} — PENDING",
            f"{icon} {pend['direction']}",
            f"Entry zone: {pend['entry']}",
            f"Current: {price if price is not None else '—'}",
            f"Distance: {dist:.2f}%" if dist is not None else "Distance: —",
            f"TP: {pend['take_profit']}",
            f"SL: {pend['stop_loss']}",
            f"Confidence: {pend.get('confidence')}%",
            "",
        ]
    if total == 0:
        lines.append("(tidak ada posisi/pending)")
    return "\n".join(lines)


def _fmt_orders() -> str:
    if not STATE["pending"]:
        return "🎯 Tidak ada pending order."
    blocks = []
    for pend in STATE["pending"].values():
        icon = "🟢" if pend["direction"] == "BUY" else "🔴"
        blocks.append(
            f"🎯 <b>PENDING ORDER — {pend['symbol']}</b>\n\n"
            f"📡 {pend['symbol']} | {icon} {pend['direction']}\n"
            f"Confidence: {pend.get('confidence')}%\n\n"
            f"Entry: {pend['entry']}\n"
            f"TP: {pend['take_profit']}\n"
            f"SL: {pend['stop_loss']}"
        )
    return "\n\n".join(blocks)


def _fmt_stats() -> str:
    summary = learn.get_stats_summary(mode=STATE["mode"])
    count = summary["count"]
    mode_label = "🧪 SIMULASI" if STATE["mode"] == "sim" else "💰 REAL"
    anchor = STATE.get("anchor_balance", DEFAULT_SIM_BALANCE)
    current = STATE["sim_balance"] if STATE["mode"] == "sim" else None
    dd = summary["drawdown"]

    lines = [
        f"📊 <b>Statistics — {count} trades</b>",
        f"TP {summary['tp_count']} | Initial SL {summary['sl_count']} | Trail {summary['trail_count']}",
        "",
        f"Mode: {mode_label}",
        "",
        "Economic Result:",
        f"{summary['wins']} WIN / {summary['losses']} LOSS / {summary['breakeven']} BE",
        f"WR: {_fmt_pct(summary['win_rate'])}",
        "",
        "Anchor Capital:",
        f"${anchor:.4f}",
    ]
    if current is not None:
        pct = (current - anchor) / anchor * 100 if anchor else 0.0
        lines += ["", "Statistical Balance:", f"${current:.4f} ({pct:+.2f}%)"]
    lines += [
        "",
        f"Average Closed Confidence: {_fmt_pct(summary['avg_closed_confidence'])}",
        "",
        f"🎯 TP: {_fmt_pct(summary['tp_pct'])}",
        f"🔒 Trail: {_fmt_pct(summary['trail_pct'])}",
        f"🛑 SL: {_fmt_pct(summary['sl_pct'])}",
        "",
        "Last 5:",
    ]
    for t in reversed(summary["last_5"]):
        icon = "🟢" if t.get("pnl_usd", 0) >= 0 else "🔴"
        label = {"tp": "TP", "initial_sl": "SL", "trail": "TRAIL"}.get(t.get("result"), t.get("result", "?").upper())
        lines.append(f"{icon} {label} {t.get('pnl_pct', 0):+.2f}% | C{int(t.get('confidence') or 0)}%")
    lines += [
        "",
        f"🚫 Banned: {active_ban_count()}",
        f"🛡️ Early Reject Remaining: {STATE['pending_cancel_stats'].get('binance_reject', 0)}",
        f"Max drawdown: {dd['max_drawdown_pct']}%",
    ]
    return "\n".join(lines)


def _fmt_koin() -> str:
    coins = STATE.get("scanned_coins") or []
    if not coins:
        return "🔍 Belum ada hasil scan."
    lines = ["🔍 <b>Koin terpantau (scan terakhir)</b>", ""]
    for c in coins[:50]:
        lines.append(f"• {c['symbol']}")
    return "\n".join(lines)


def _fmt_banned() -> str:
    if not STATE["bans"]:
        return "🚫 Tidak ada koin yang diban."
    lines = ["🚫 <b>Daftar Ban</b>", ""]
    now = time.time()
    for symbol, entry in STATE["bans"].items():
        if entry.get("permanent"):
            lines.append(f"• {symbol} — PERMANEN ({entry.get('reason')})")
        else:
            remaining = max(0, entry.get("until", now) - now)
            lines.append(f"• {symbol} — sisa {remaining/3600:.1f} jam ({entry.get('reason')})")
    return "\n".join(lines)


HELP_TEXT = (
    "🤖 <b>SMC/ICT Auto Trading Bot</b>\n\n"
    "/auto — mulai/lanjutkan scanner\n"
    "/stop — hentikan scanner (posisi & proteksi tetap jalan)\n"
    "/mode on|off — real trade / simulasi\n"
    "/resetbalance — reset saldo simulasi ke $10\n"
    "/margin <usd> — set margin per trade\n"
    "/leverage <n> — set leverage\n"
    "/autostop <pct> — set drawdown auto-stop\n"
    "/trade — dashboard posisi & pending\n"
    "/order — detail pending order\n"
    "/stats — statistik performa\n"
    "/koin — daftar koin hasil scan terakhir\n"
    "/banned [symbol] — lihat/ban permanen\n"
    "/unban <symbol|all> — hapus ban\n"
    "/timeout all|<symbol> — cleanup manual\n"
    "/open — restore learning checkpoint\n"
    "/IP — IP server saat ini\n"
)


# ----------------------------------------------------------------------
# /timeout manual cleanup
# ----------------------------------------------------------------------

async def _cleanup_symbol(symbol: str) -> None:
    for pid, pend in list(STATE["pending"].items()):
        if pend["symbol"] == symbol:
            await cancel_pending(pid, "manual")
    for pid, pos in list(STATE["positions"].items()):
        if pos["symbol"] == symbol:
            if pos["mode"] == "real" and BINANCE_EXEC:
                try:
                    if pos.get("tp_algo_id"):
                        await asyncio.to_thread(BINANCE_EXEC.cancel_algo_order, pos["tp_algo_id"])
                    if pos.get("sl_algo_id"):
                        await asyncio.to_thread(BINANCE_EXEC.cancel_algo_order, pos["sl_algo_id"])
                    risk = await asyncio.to_thread(BINANCE_EXEC.position_risk, symbol)
                    if risk and abs(float(risk["positionAmt"])) > 0:
                        close_side = "SELL" if pos["direction"] == "BUY" else "BUY"
                        await asyncio.to_thread(BINANCE_EXEC.market_close, symbol, close_side,
                                                 abs(float(risk["positionAmt"])))
                except BinanceError:
                    log.exception("manual cleanup failed for %s", symbol)
            STATE["positions"].pop(pid, None)
            ws_untrack_sync(symbol)
    save_state()


async def cmd_timeout(args: str) -> str:
    args = args.strip()
    if not args:
        return "❌ Invalid command.\n\nUsage:\n/timeout all\n/timeout BTCUSDT"
    if args.lower() == "all":
        symbols = {p["symbol"] for p in STATE["pending"].values()} | {p["symbol"] for p in STATE["positions"].values()}
        for symbol in symbols:
            await _cleanup_symbol(symbol)
        return "✅ Semua pending & posisi dibersihkan (manual timeout)."
    symbol = args.upper()
    await _cleanup_symbol(symbol)
    return f"✅ {symbol} dibersihkan (manual timeout)."


# ----------------------------------------------------------------------
# command dispatch
# ----------------------------------------------------------------------

async def cmd_auto(_args: str) -> str:
    STATE["scanning"] = True
    STATE["autostop_triggered"] = False
    STATE["peak_balance"] = await _current_balance()
    save_state()
    return "▶️ Scanner diaktifkan. High-water mark autostop direset."


async def cmd_stop(_args: str) -> str:
    STATE["scanning"] = False
    save_state()
    return "⏹️ Scanner dihentikan. WebSocket, posisi aktif, dan learning tetap berjalan."


async def cmd_mode(args: str) -> str:
    args = args.strip().lower()
    if args not in ("on", "off"):
        return "❌ Invalid /mode usage.\n\nUse:\n/mode on\n/mode off"
    if args == "on":
        if not is_binance_ready():
            return "⏸️ Binance API sedang dibatasi.\n/mode on sementara tidak tersedia. Tunggu hingga READY."
        if not BINANCE_EXEC:
            return "❌ BINANCE_API_KEY/SECRET belum diset — tidak bisa aktifkan mode real."
        STATE["mode"] = "real"
        STATE["peak_balance"] = await _current_balance()
        STATE["autostop_triggered"] = False
        save_state()
        return "💰 Mode REAL aktif."
    STATE["mode"] = "sim"
    save_state()
    return "🧪 Mode SIMULASI aktif."


async def cmd_resetbalance(_args: str) -> str:
    if not is_binance_ready():
        return "⏸️ Binance API sedang dibatasi.\n/resetbalance sementara tidak tersedia."
    STATE["sim_balance"] = DEFAULT_SIM_BALANCE
    STATE["anchor_balance"] = DEFAULT_SIM_BALANCE
    STATE["peak_balance"] = DEFAULT_SIM_BALANCE
    STATE["autostop_triggered"] = False
    save_state()
    return f"🔄 Saldo simulasi direset ke ${DEFAULT_SIM_BALANCE:.2f}."


async def cmd_margin(args: str) -> str:
    try:
        value = float(args.strip())
        if value <= 0:
            raise ValueError
    except ValueError:
        return "❌ Invalid /margin usage.\n\nUse:\n/margin 1.5"
    STATE["margin"] = value
    save_state()
    return f"✅ Margin diset ke ${value:.4f} per trade."


async def cmd_leverage(args: str) -> str:
    try:
        value = int(args.strip())
        if not (1 <= value <= 125):
            raise ValueError
    except ValueError:
        return "❌ Invalid /leverage usage.\n\nUse:\n/leverage 10 (1-125)"
    STATE["leverage"] = value
    save_state()
    return f"✅ Leverage diset ke {value}x."


async def cmd_autostop(args: str) -> str:
    args = args.strip()
    if not args:
        current = STATE.get("autostop_pct")
        return f"ℹ️ Autostop saat ini: {current if current is not None else 'nonaktif'}%.\n\nUse:\n/autostop 10"
    try:
        value = float(args)
        if value <= 0:
            raise ValueError
    except ValueError:
        return "❌ Invalid /autostop usage.\n\nUse:\n/autostop 10"
    STATE["autostop_pct"] = value
    STATE["autostop_triggered"] = False
    save_state()
    return f"✅ Autostop diset ke {value}% drawdown dari peak."


async def cmd_unban(args: str) -> str:
    args = args.strip()
    if not args:
        return "❌ Invalid /unban usage.\n\nUse:\n/unban BTCUSDT\n/unban all"
    if args.lower() == "all":
        n = unban_all()
        save_state()
        return f"✅ {n} ban dihapus."
    symbol = args.upper()
    ok = unban(symbol)
    save_state()
    return f"✅ {symbol} di-unban." if ok else f"ℹ️ {symbol} tidak sedang diban."


async def cmd_banned(args: str) -> str:
    args = args.strip()
    if not args:
        return _fmt_banned()
    symbol = args.upper()
    apply_ban(symbol, None, "manual_permanent", permanent=True)
    save_state()
    return f"🚫 {symbol} diban PERMANEN."


async def cmd_open(_args: str) -> str:
    report = learn.load_checkpoint()
    if report["restored"]:
        return f"✅ Learning checkpoint dipulihkan dari {report['source']}."
    return f"⚠️ Gagal memulihkan checkpoint: {report['reason']}. Memulai dari state kosong."


COMMANDS = {
    "/start": lambda args: HELP_TEXT,
    "/help": lambda args: HELP_TEXT,
    "/ip": lambda args: get_public_ip(),
    "/auto": cmd_auto,
    "/stop": cmd_stop,
    "/mode": cmd_mode,
    "/resetbalance": cmd_resetbalance,
    "/margin": cmd_margin,
    "/leverage": cmd_leverage,
    "/autostop": cmd_autostop,
    "/trade": lambda args: _fmt_trade_dashboard(),
    "/order": lambda args: _fmt_orders(),
    "/stats": lambda args: _fmt_stats(),
    "/koin": lambda args: _fmt_koin(),
    "/banned": cmd_banned,
    "/unban": cmd_unban,
    "/timeout": cmd_timeout,
    "/open": cmd_open,
}


# ----------------------------------------------------------------------
# evidence-gated threshold calibration loop
# ----------------------------------------------------------------------

CALIBRATION_INTERVAL_SEC = 6 * 3600


async def calibration_loop(stop_flag: "asyncio.Event") -> None:
    while not stop_flag.is_set():
        try:
            await asyncio.sleep(CALIBRATION_INTERVAL_SEC)
            if stop_flag.is_set():
                break
            result = learn.maybe_calibrate(STATE["confidence_threshold"])
            if result["action"] == "adjust":
                learn.apply_calibration(result["new_threshold"], result["reason"])
                STATE["confidence_threshold"] = result["new_threshold"]
                save_state()
                notify(f"🧠 <b>Threshold calibration</b>\nBaru: {result['new_threshold']}%\n"
                       f"Alasan: {html.escape(result['reason'])}")
        except asyncio.CancelledError:
            break
        except Exception:
            log.exception("calibration_loop iteration failed")


# ----------------------------------------------------------------------
# background task lifecycle
# ----------------------------------------------------------------------

_STOP_FLAG: Optional[asyncio.Event] = None
_TASKS: list[asyncio.Task] = []


async def on_start(context: dict):
    global _SEND_MESSAGE, _CHAT_ID, _STOP_FLAG, _TASKS

    _SEND_MESSAGE = context.get("send_message")
    _CHAT_ID = context.get("chat_id")

    load_state()
    open_report = learn.load_checkpoint()
    STATE["confidence_threshold"] = learn.get_threshold()

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        log.warning("BINANCE_API_KEY/SECRET not set — real trading disabled, sim-only.")

    _STOP_FLAG = asyncio.Event()
    _TASKS = [
        asyncio.create_task(ws_feed_task(_STOP_FLAG), name="ws_feed"),
        asyncio.create_task(scanner_loop(_STOP_FLAG), name="scanner"),
        asyncio.create_task(monitor_loop(_STOP_FLAG), name="monitor"),
        asyncio.create_task(autosave_loop(_STOP_FLAG), name="autosave"),
        asyncio.create_task(calibration_loop(_STOP_FLAG), name="calibration"),
    ]

    ip = await asyncio.to_thread(get_public_ip)
    learn_status = f"dipulihkan dari {open_report['source']}" if open_report["restored"] else "kosong (baru)"
    notify(
        "🚀 <b>main.py AKTIF</b>\n\n"
        f"Server IP: <code>{ip}</code>\n"
        f"Mode: {'💰 REAL' if STATE['mode'] == 'real' else '🧪 SIMULASI'}\n"
        f"Learning checkpoint: {learn_status}\n"
        f"Confidence threshold: {STATE['confidence_threshold']}%\n\n"
        "Gunakan /auto untuk mulai scanning."
    )


async def handle_update(update: dict, context: dict) -> None:
    message = update.get("message") or {}
    chat_id = context.get("chat_id") or (message.get("chat") or {}).get("id")
    text = str(message.get("text") or message.get("caption") or "").strip()
    if not text or not text.startswith("/"):
        return

    parts = text.split(maxsplit=1)
    command = parts[0].split("@", 1)[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handler = COMMANDS.get(command)
    if handler is None:
        return  # unknown command — silently ignore, launcher already handles /try /end /ganti

    send = context.get("send_message") or _SEND_MESSAGE
    try:
        result = handler(args)
        if asyncio.iscoroutine(result):
            result = await result
        if result and send and chat_id:
            send(chat_id, result)
    except Exception as exc:
        log.exception("command %s failed", command)
        if send and chat_id:
            send(chat_id, f"❌ Command gagal: <code>{html.escape(str(exc)[:600])}</code>")


async def on_stop(context: dict) -> None:
    global _TASKS
    if _STOP_FLAG:
        _STOP_FLAG.set()
    for task in _TASKS:
        task.cancel()
    for task in _TASKS:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    _TASKS = []

    learn.save_checkpoint()
    save_state()
    log.info("main.py stopped cleanly; state persisted.")
