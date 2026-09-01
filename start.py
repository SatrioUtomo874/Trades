from __future__ import annotations

"""SMCAutoTrade start_v5.py

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

import requests
import websocket

VERSION = "5.0"

BYBIT_BASE_URL = (os.getenv("BYBIT_BASE_URL") or "https://api.bybit.com").rstrip("/")
BYBIT_WS_URL = (os.getenv("BYBIT_WS_URL") or "wss://stream.bybit.com/v5/public/linear").strip()
BINANCE_BASE_URL = (os.getenv("BINANCE_BASE_URL") or "https://fapi.binance.com").rstrip("/")
BINANCE_WS_BASE = (os.getenv("BINANCE_WS_BASE") or "wss://fstream.binance.com/ws").rstrip("/")
IP_URL = (os.getenv("IP_URL") or "https://api.ipify.org").strip()

MAX_SYMBOLS = max(1, int(os.getenv("DATA_MAX_SYMBOLS", "250")))
TF_CONFIG = {"15": 700, "5": 500, "1": 500}
LOAD_INTERVAL = max(0.0, float(os.getenv("DATA_LOAD_INTERVAL", "1.0")))
RETENTION_EXTRA = max(50, int(os.getenv("DATA_RETENTION_EXTRA", "50")))
REQUEST_TIMEOUT = max(5, int(os.getenv("REQUEST_TIMEOUT", "20")))
WS_PING_INTERVAL = max(5, int(os.getenv("WS_PING_INTERVAL", "20")))
WS_RECONNECT_MAX = max(5, int(os.getenv("WS_RECONNECT_MAX", "30")))
LOG_EVERY_SYMBOL_TICK = max(2, int(os.getenv("LOG_EVERY_SYMBOL_TICK", "15")))
STRATEGY_FILE = (os.getenv("STRATEGY_FILE") or "strategy.py").strip()
BASE_DIR = Path(__file__).resolve().parent

BINANCE_API_KEY = (os.getenv("BINANCE_API_KEY") or "").strip()
BINANCE_API_SECRET = (os.getenv("BINANCE_API_SECRET") or "").strip()

DEFAULT_MARGIN = max(0.0, float(os.getenv("TRADE_MARGIN", "10")))
DEFAULT_LEVERAGE = max(1, int(os.getenv("TRADE_LEVERAGE", "10")))
DEFAULT_MAX_ACTIVE = max(0, int(os.getenv("TRADE_MAX_ACTIVE", "5")))
BAN_HOURS = max(0.25, float(os.getenv("TRADE_BAN_HOURS", "8")))
AUTO_QTY_RANGE = 0.50
STATE_FILE = Path(os.getenv("TRADE_STATE_FILE", str(BASE_DIR / "trade_state_v5.json"))).resolve()
STATE_LOCK = threading.RLock()

logging.basicConfig(
    level=(os.getenv("LOG_LEVEL") or "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("start-v5")


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

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.engine.add_data_callback(callback)


class BinanceError(RuntimeError):
    pass


class BinanceClient:
    def __init__(self) -> None:
        self.key = BINANCE_API_KEY
        self.secret = BINANCE_API_SECRET
        self.base_url = BINANCE_BASE_URL
        self.meta: dict[str, dict[str, Any]] = {}
        self.meta_loaded = False

    @property
    def configured(self) -> bool:
        return bool(self.key and self.secret)

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
        try:
            if method == "GET":
                r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            elif method == "POST":
                r = requests.post(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            elif method == "DELETE":
                r = requests.delete(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            else:
                raise BinanceError(f"Unsupported HTTP method {method}")
        except requests.RequestException as exc:
            raise BinanceError(f"network error: {exc}") from exc
        try:
            body = r.json()
        except ValueError:
            body = r.text
        if r.status_code >= 400:
            raise BinanceError(f"HTTP {r.status_code}: {body}")
        if isinstance(body, dict) and "code" in body and isinstance(body.get("code"), int) and body["code"] < 0:
            raise BinanceError(str(body))
        return body

    def public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._signed("GET", path, params, signed=False)

    def exchange_info(self) -> dict[str, Any]:
        if not self.meta_loaded:
            data = self.public_get("/fapi/v1/exchangeInfo")
            self.meta = {str(s["symbol"]).upper(): s for s in data.get("symbols", []) if s.get("status") == "TRADING"}
            self.meta_loaded = True
            log.info("[BINANCE] exchangeInfo loaded symbols=%d", len(self.meta))
        return {"symbols": list(self.meta.values())}

    def symbol_meta(self, symbol: str) -> dict[str, Any]:
        self.exchange_info()
        s = self.meta.get(symbol.upper())
        if not s:
            raise BinanceError(f"Binance symbol {symbol} not found in exchangeInfo")
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
        self.binance.exchange_info()  # public; needed for quantity validation in both modes

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
            id=f"T5-{symbol}-{uuid.uuid4().hex[:10]}",
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
            if self.mode == "ON" and time.monotonic() - self._last_real_poll >= 2.0:
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
        return f"MODE: {self.mode}\nExecution: {'REAL Binance Futures' if self.mode == 'ON' else 'SIMULATION'}\nMargin: {self.margin}\nLeverage: {self.leverage}x\nMax active: {self.max_active}"

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
        self.bootstrap_complete = False
        self.ws: websocket.WebSocketApp | None = None
        self.ws_thread: threading.Thread | None = None
        self.bootstrap_thread: threading.Thread | None = None
        self.strategy: Any = None
        self.strategy_error: str | None = None
        self.tick_logs: dict[tuple[str, str], float] = {}

    def _notify(self, text: str) -> None:
        if self.chat_id is None:
            return
        try:
            self.send_message(self.chat_id, text)
        except Exception:
            log.exception("[TG] notify failed")

    def public_ip(self) -> str:
        r = requests.get(IP_URL, timeout=10)
        r.raise_for_status()
        return r.text.strip()

    def _get(self, url: str, params: dict[str, Any]) -> requests.Response:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
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
        d = self._get(f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo", {}).json()
        return {
            str(x.get("symbol") or "").upper()
            for x in d.get("symbols") or []
            if x.get("status") == "TRADING" and x.get("contractType") == "PERPETUAL" and x.get("quoteAsset") == "USDT" and x.get("marginAsset") == "USDT"
        }

    def build_universe(self) -> list[str]:
        a = self.bybit_symbols(); log.info("[DISCOVERY] Bybit=%d", len(a))
        b = self.binance_symbols(); log.info("[DISCOVERY] Binance=%d", len(b))
        common = sorted(a & b)
        selected = common[:MAX_SYMBOLS]
        with self.symbol_lock: self.symbols = selected
        log.info("[DISCOVERY] common=%d selected=%d", len(common), len(selected))
        return selected

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
        syms = self.get_symbols(); total = len(syms); success = 0; failed = 0
        log.info("[BOOTSTRAP] START %d symbols", total)
        for i, s in enumerate(syms, 1):
            if self.stop_event.is_set() or not self.auto_running: return
            t0 = time.monotonic(); log.info("[BOOTSTRAP] %d/%d %s", i, total, s)
            if self.bootstrap_symbol(s): success += 1
            else: failed += 1
            if i == 1 or i % 25 == 0 or i == total:
                self._notify(f"📥 BOOTSTRAP PROGRESS\n{i}/{total}\nSuccess: {success} | Failed: {failed}")
            elapsed = time.monotonic() - t0
            wait = max(0.0, LOAD_INTERVAL - elapsed)
            if i < total and wait: self.stop_event.wait(wait)
        if not self.auto_running or self.stop_event.is_set(): return
        self.bootstrap_complete = success == total and total > 0
        if not self.bootstrap_complete:
            log.error("[BOOTSTRAP] incomplete success=%d failed=%d", success, failed)
            self._notify(f"❌ DATA BOOTSTRAP INCOMPLETE\nLoaded: {success}/{total}\nStrategy/WebSocket not started.")
            return
        self.load_strategy()
        self._notify(
            "✅ DATA READY\n"
            f"Symbols: {total}\nLoaded: {success}\nFailed: {failed}\n"
            "Historical: 15M/700 + 5M/500 + 1M/500\n"
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
            self._notify(f"⚠️ INITIAL STRATEGY SCAN SKIPPED\nReason: {self.strategy_error or 'strategy unavailable'}")
        self.start_websocket()

    # ---- strategy ----
    def _strategy_path(self) -> Path:
        raw = STRATEGY_FILE
        if raw.startswith("[") and "](" in raw:  # defensive normalization for prior env mistakes
            raw = raw.split("](", 1)[0].lstrip("[")
        p = Path(raw)
        return p if p.is_absolute() else (BASE_DIR / p).resolve()

    def load_strategy(self) -> None:
        path = self._strategy_path()
        self.strategy = None; self.strategy_error = None
        if not path.is_file():
            self.strategy_error = f"strategy file not found: {path}"
            log.error("[STRATEGY] %s", self.strategy_error)
            self._notify(f"❌ STRATEGY LOAD FAILED\nFile: {path.name}\nPath: {path}")
            return
        try:
            name = f"smc_strategy_v5_{int(time.time()*1000)}"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None: raise ImportError("cannot create strategy spec")
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module
            spec.loader.exec_module(module)
            self.strategy = module
            if hasattr(module, "initialize"): module.initialize(self.api, self.context)
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
        tid = self.trade_manager.accept_signal(signal)
        if tid:
            trade = self.trade_manager.trades.get(tid)

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        symbol = str(event.get("symbol") or "").upper()
        candle = event.get("candle") or {}
        if symbol and candle.get("close") is not None:
            self.trade_manager.on_market_price(symbol, float(candle["close"]), candle)
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
    def _topic_batches(topics: list[str], max_chars: int = 19000) -> list[list[str]]:
        batches: list[list[str]] = []; cur: list[str] = []; n = 2
        for t in topics:
            add = len(t) + (1 if cur else 0)
            if cur and n + add > max_chars:
                batches.append(cur); cur = []; n = 2
            cur.append(t); n += add
        if cur: batches.append(cur)
        return batches

    def _ws_open(self, ws: websocket.WebSocketApp) -> None:
        topics = self._topics(); batches = self._topic_batches(topics)
        for i, batch in enumerate(batches, 1):
            ws.send(json.dumps({"op": "subscribe", "req_id": f"smc5-{i}", "args": batch}, separators=(",", ":")))
            log.info("[WS] subscription batch=%d/%d topics=%d", i, len(batches), len(batch))
        self._notify(f"🟢 MARKET DATA LIVE\nBybit WS\nSymbols: {len(self.symbols)}\nStreams: 15M + 5M + 1M\nSubscriptions: {len(topics)} in {len(batches)} batches")

    def _ws_message(self, _ws: websocket.WebSocketApp, raw: str) -> None:
        try: p = json.loads(raw)
        except json.JSONDecodeError: return
        if p.get("op") in {"subscribe", "pong"}: return
        parts = str(p.get("topic") or "").split(".")
        if len(parts) != 3 or parts[0] != "kline": return
        tf = parts[1]
        for x in p.get("data") or []:
            try:
                c = Candle(int(x["start"]), float(x["open"]), float(x["high"]), float(x["low"]), float(x["close"]), float(x["volume"]), float(x.get("turnover") or 0), bool(x.get("confirm", False)))
                symbol = str(x.get("symbol") or parts[2]).upper()
            except (KeyError, TypeError, ValueError):
                log.exception("[WS] bad kline payload"); continue
            self.store.upsert(symbol, tf, c)
            key = (symbol, tf); now = time.monotonic()
            if now - self.tick_logs.get(key, 0) >= LOG_EVERY_SYMBOL_TICK:
                self.tick_logs[key] = now
                log.info("[TICK] %s %s C=%.8f confirm=%s", symbol, tf, c.close, c.confirmed)
            self._dispatch_event({"type": "candle", "symbol": symbol, "timeframe": tf, "candle": c.as_dict()})

    def _ws_error(self, _ws: websocket.WebSocketApp, error: Any) -> None:
        log.warning("[WS] error=%s", error)

    def _ws_close(self, _ws: websocket.WebSocketApp, code: Any, msg: Any) -> None:
        log.warning("[WS] closed code=%s msg=%s", code, msg)

    def _ws_worker(self) -> None:
        backoff = 2
        while self.auto_running and not self.stop_event.is_set():
            try:
                ws = websocket.WebSocketApp(BYBIT_WS_URL, on_open=self._ws_open, on_message=self._ws_message, on_error=self._ws_error, on_close=self._ws_close)
                self.ws = ws
                log.info("[WS] connecting %s", BYBIT_WS_URL)
                ws.run_forever(ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_INTERVAL - 2, skip_utf8_validation=True)
            except Exception:
                log.exception("[WS] worker error")
            finally:
                self.ws = None
            if self.auto_running and not self.stop_event.is_set():
                log.warning("[WS] reconnect in %ss", backoff)
                self.stop_event.wait(backoff); backoff = min(backoff * 2, WS_RECONNECT_MAX)

    def start_websocket(self) -> None:
        if self.ws_thread and self.ws_thread.is_alive(): return
        self.ws_thread = threading.Thread(target=self._ws_worker, name="bybit-ws", daemon=True); self.ws_thread.start()

    def stop_websocket(self) -> None:
        ws = self.ws; self.ws = None
        if ws:
            try: ws.close()
            except Exception: log.exception("[WS] close failed")

    # ---- lifecycle / command ----
    def start_auto(self) -> str:
        with self.run_lock:
            if self.auto_running: return "ℹ️ /auto sudah aktif."
            self.auto_running = True; self.bootstrap_complete = False
        try: ip = self.public_ip()
        except Exception as exc: ip = f"unavailable ({exc})"
        self._notify(f"🤖 AUTO MODE\nServer IP: {ip}\n\nDiscovering Bybit + Binance...")
        try: syms = self.build_universe()
        except Exception as exc:
            self.auto_running = False; log.exception("[AUTO] discovery failed"); return f"❌ /auto gagal: {exc}"
        if not syms:
            self.auto_running = False; return "❌ /auto gagal: common symbol universe kosong"
        self._notify(f"✅ Universe ready\nCommon selected: {len(syms)}\n\nBootstrap 15M/5M/1M dimulai...")
        self.bootstrap_thread = threading.Thread(target=self.bootstrap_all, name="bootstrap", daemon=True); self.bootstrap_thread.start()
        return f"🟢 /auto aktif — {len(syms)} pair masuk data pipeline."

    def reset_strategy(self) -> str:
        with self.run_lock:
            if not self.auto_running or not self.bootstrap_complete:
                return "ℹ️ /auto belum aktif atau historical data belum ready."
        old = self.strategy
        if old and hasattr(old, "shutdown"):
            try: old.shutdown()
            except Exception: log.exception("[STRATEGY RESET] shutdown failed")
        self.load_strategy()
        if not self.strategy: return f"❌ Strategy reset gagal: {self.strategy_error or 'unknown error'}"
        try:
            summary = self.strategy.on_data_ready() if hasattr(self.strategy, "on_data_ready") else "✅ Strategy reloaded"
            self._notify(f"🔄 STRATEGY RESET\nLoaded: {self._strategy_path().name}\nData + WebSocket tetap berjalan.\n\n{summary or 'Scan selesai.'}")
            self._accept_strategy_queue()
            return "✅ Strategy berhasil di-reset dan scan ulang."
        except Exception as exc:
            log.exception("[STRATEGY RESET] scan failed")
            return f"❌ Strategy reset scan gagal: {type(exc).__name__}: {exc}"

    def stop(self) -> None:
        with self.run_lock: self.auto_running = False
        self.stop_websocket()
        if self.strategy and hasattr(self.strategy, "shutdown"):
            try: self.strategy.shutdown()
            except Exception: log.exception("[STRATEGY] shutdown failed")
        self.trade_manager._save_state()
        log.info("[ENGINE] stopped")

    def get_symbols(self) -> list[str]:
        with self.symbol_lock: return list(self.symbols)

    def status(self) -> dict[str, Any]:
        return {
            "auto": self.auto_running, "bootstrap": self.bootstrap_complete, "symbols": len(self.symbols),
            "strategy": bool(self.strategy), "strategy_error": self.strategy_error, "ws": self.ws is not None,
            "mode": self.trade_manager.mode, "active": self.trade_manager.active_count(), "max": self.trade_manager.max_active,
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
    log.info("[START] V5 ready | ip=%s | base=%s | strategy=%s", ip, BASE_DIR, STRATEGY_FILE)
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
        "🤖 SMCAutoTrade V5\n\n"
        "/auto — discovery + historical + websocket\n"
        "/mode — show mode\n/mode on — REAL Binance\n/mode off — SIMULATION\n"
        "/margin 10 — target margin/trade\n/leverage 10 — leverage\n/max 5 — max active orders/positions\n"
        "/trade — active simulated/real orders\n/stats — closed trade statistics\n"
        "/banned — banned symbols\n/unban BTCUSDT — remove ban\n/resetban — clear all bans\n"
        "/reset — reload strategy.py using existing market data\n"
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
        if cmd == "/status":
            s = engine.status()
            return (
                "📊 SYSTEM STATUS\n"
                f"AUTO: {s['auto']}\nBOOTSTRAP: {s['bootstrap']}\nSYMBOLS: {s['symbols']}\n"
                f"WS: {s['ws']}\nSTRATEGY: {s['strategy']}\nSTRATEGY ERROR: {s['strategy_error'] or '-'}\n"
                f"MODE: {s['mode']}\nACTIVE: {s['active']}/{s['max']}"
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

        if engine.strategy and hasattr(engine.strategy, "handle_command"):
            result = engine.strategy.handle_command(text)
            return None if result is None else str(result)
        return None
    except Exception as exc:
        log.exception("[COMMAND] %s failed", cmd)
        engine._notify(f"🚨 HANDLER ERROR\nCommand: {cmd}\n{type(exc).__name__}: {exc}")
        return f"❌ Command gagal: {type(exc).__name__}: {exc}"
