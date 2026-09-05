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
import signal
import socket
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal, InvalidOperation
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

import strategy
import learn


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


def setup_logging(state_dir: str) -> logging.Logger:
    os.makedirs(state_dir, exist_ok=True)
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    log.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(state_dir, "bot.log"), maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)
    return log


logger = logging.getLogger("main")


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
    pass


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

    # -- revisi: sinkronisasi waktu server (§4) -----------------------------
    # Penyebab klasik "API key sama, IP sama, tiba-tiba semua order error"
    # adalah timestamp lokal drift terhadap server Binance (-1021 Timestamp
    # for this request is outside of the recvWindow). Ini sinkronkan offset
    # sekali di awal & auto-refresh tiap 30 menit / saat error -1021 muncul.
    def sync_server_time(self) -> None:
        try:
            resp = self.session.get(f"{self.base_url}/fapi/v1/time", timeout=10)
            server_ms = resp.json()["serverTime"]
            local_ms = int(time.time() * 1000)
            self._time_offset_ms = server_ms - local_ms
            self._time_offset_synced_at = time.time()
            logger.info("Binance time offset disinkronkan: %d ms (endpoint=%s)", self._time_offset_ms, self.base_url)
        except Exception as e:
            logger.warning("Gagal sinkronisasi waktu server Binance (%s) — pakai waktu lokal", e)

    def _timestamp(self) -> int:
        if time.time() - self._time_offset_synced_at > 1800:
            self.sync_server_time()
        return int(time.time() * 1000) + self._time_offset_ms

    def _sign(self, params: Dict[str, Any]) -> str:
        query = urllib.parse.urlencode(params, doseq=True)
        sig = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False, _retry_on_time_drift: bool = True) -> Any:
        params = dict(params or {})
        url = self.base_url + path
        try:
            if signed:
                params["timestamp"] = self._timestamp()
                params["recvWindow"] = 10000  # lebih longgar dari default 5000, toleran thd sedikit latensi
                query = self._sign(params)
                url = f"{url}?{query}"
                resp = self.session.request(method, url, timeout=15)
            else:
                resp = self.session.request(method, url, params=params, timeout=15)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise ExchangeError(f"Binance connection error: {e}")

        if resp.status_code in (429, 418):
            raise RateLimitError(f"Binance rate limit (HTTP {resp.status_code}): {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError:
            raise ExchangeError(f"Binance HTTP {resp.status_code} malformed response: {resp.text[:200]}")

        if isinstance(data, dict) and "code" in data and data.get("code", 0) < 0:
            code = data["code"]
            msg = data.get("msg")
            if code in (-1003, -1015):  # too many requests
                raise RateLimitError(f"Binance rate limit code {code}: {msg}")
            if code == -1021 and signed and _retry_on_time_drift:
                # timestamp drift — resync sekali lalu retry SEKALI saja (hindari infinite loop)
                logger.warning("Binance -1021 (timestamp drift) — resync & retry sekali: %s", msg)
                self.sync_server_time()
                return self._request(method, path, params, signed=signed, _retry_on_time_drift=False)
            if code == -2015:
                raise ExchangeError(
                    f"Binance error -2015 (Invalid API-key/IP/permissions): {msg}. "
                    f"Endpoint: {self.base_url} (MAINNET). "
                    f"Penyebab tersering: IP restriction key tidak mencakup IP server ini, "
                    f"atau permission Futures belum diaktifkan di API key ini."
                )
            raise ExchangeError(f"Binance error {code}: {msg} (HTTP {resp.status_code})")
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
            logger.warning("Gagal deteksi position mode, asumsikan One-way: %s", e)
            self._hedge_mode = False
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
        return self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    def exchange_info(self, force: bool = False) -> Dict[str, Any]:
        if self._exchange_info_cache and not force and time.time() - self._exchange_info_ts < 3600:
            return self._exchange_info_cache
        data = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
        symbols = {s["symbol"]: s for s in data.get("symbols", [])}
        self._exchange_info_cache = symbols
        self._exchange_info_ts = time.time()
        return symbols

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        info = self.exchange_info().get(symbol)
        if not info:
            raise ExchangeError(f"Symbol {symbol} tidak ditemukan di Binance exchangeInfo")
        filters = {f["filterType"]: f for f in info["filters"]}
        return {
            "step_size": Decimal(filters["LOT_SIZE"]["stepSize"]),
            "min_qty": Decimal(filters["LOT_SIZE"]["minQty"]),
            "tick_size": Decimal(filters["PRICE_FILTER"]["tickSize"]),
            "min_notional": Decimal(filters.get("MIN_NOTIONAL", {}).get("notional", "5")),
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

    def place_limit_order(self, symbol: str, side: str, quantity: Decimal, price: Decimal, direction: str) -> Any:
        params = {
            "symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTC",
            "quantity": str(quantity), "price": str(price),
        }
        params.update(self._position_side_param(direction))
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_stop_market(self, symbol: str, side: str, stop_price: Decimal, direction: str, close_position: bool = True) -> Any:
        params = {"symbol": symbol, "side": side, "type": "STOP_MARKET", "stopPrice": str(stop_price)}
        if close_position and not self.is_hedge_mode():
            params["closePosition"] = "true"  # closePosition tidak valid dikombinasi positionSide di Hedge Mode
        params.update(self._position_side_param(direction))
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_take_profit_market(self, symbol: str, side: str, stop_price: Decimal, direction: str, close_position: bool = True) -> Any:
        params = {"symbol": symbol, "side": side, "type": "TAKE_PROFIT_MARKET", "stopPrice": str(stop_price)}
        if close_position and not self.is_hedge_mode():
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


def round_price(price: float, tick_size: Decimal) -> Decimal:
    return round_step(Decimal(str(price)), tick_size, rounding=ROUND_DOWN)


# =============================================================================
# 5. STATE MANAGEMENT — thread-safe, state machine, idempotency (§55,56,57)
# =============================================================================

ALLOWED_TRANSITIONS = {
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
        self.autostop_pct: Optional[float] = None
        self.highest_balance: Optional[float] = None
        self.sim_balance = 10.0
        self.sim_balance_anchor = 10.0

        self.binance_paused = False
        self.binance_pause_ts: Optional[float] = None

        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position dict
        self.scanned_coins: List[str] = []
        self.bans: Dict[str, Dict[str, Any]] = {}  # symbol -> {reason, expiry}
        self.closed_trades: List[Dict[str, Any]] = []
        self.processed_events: set = set()  # idempotency guard: f"{symbol}:{event}:{ts}"

    def symbol_lock(self, symbol: str) -> threading.Lock:
        with self._lock:
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = threading.Lock()
            return self._symbol_locks[symbol]

    # -- position lifecycle ---------------------------------------------------
    def add_pending(self, setup: Dict[str, Any], qty: Decimal, margin_used: float) -> None:
        with self._lock:
            self.positions[setup["pair"]] = {
                **setup,
                "status": "PENDING",
                "qty": str(qty),
                "margin_used": margin_used,
                "leverage": self.leverage,
                "created_at": time.time(),
                "trail_count": 0,
                "binance_order_ids": {},
                "peak_price": setup["entry"],
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
    def ban(self, symbol: str, reason: str, duration_sec: float) -> None:
        with self._lock:
            self.bans[symbol] = {"reason": reason, "expiry": time.time() + duration_sec}

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
            if b["expiry"] <= time.time():
                del self.bans[symbol]
                return False
            return True

    def cleanup_expired_bans(self) -> List[str]:
        expired = []
        with self._lock:
            for symbol in list(self.bans.keys()):
                if self.bans[symbol]["expiry"] <= time.time():
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
                "mode": self.mode, "auto": self.auto, "margin": self.margin, "leverage": self.leverage,
                "autostop_pct": self.autostop_pct, "highest_balance": self.highest_balance,
                "sim_balance": self.sim_balance, "sim_balance_anchor": self.sim_balance_anchor,
                "positions": self.positions, "scanned_coins": self.scanned_coins,
                "bans": self.bans, "closed_trades": self.closed_trades[-2000:],
                "saved_at": time.time(),
            }

    def load_state(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self.mode = data.get("mode", self.mode)
            self.auto = data.get("auto", False)
            self.margin = data.get("margin", self.margin)
            self.leverage = data.get("leverage", self.leverage)
            self.autostop_pct = data.get("autostop_pct")
            self.highest_balance = data.get("highest_balance")
            self.sim_balance = data.get("sim_balance", 10.0)
            self.sim_balance_anchor = data.get("sim_balance_anchor", 10.0)
            self.positions = data.get("positions", {})
            self.scanned_coins = data.get("scanned_coins", [])
            self.bans = data.get("bans", {})
            self.closed_trades = data.get("closed_trades", [])

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


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start_sender(self) -> None:
        """Start hanya pengirim pesan keluar. Tidak melakukan getUpdates."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram belum dikonfigurasi (.env) — notifikasi nonaktif")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._sender_loop, name="TelegramSender", daemon=True)
        self._thread.start()

    def start(self) -> None:
        # Backward compatible: saat standalone hanya perlu sender.
        self.start_sender()

    def stop(self) -> None:
        self._stop.set()

    def send(self, text: str, event_type: str = "INFO") -> None:
        """Kirim hanya event penting ke Telegram (§53); log detail tetap di file."""
        if event_type not in IMPORTANT_EVENTS and event_type != "INFO":
            return
        if not self.token:
            return
        self._queue.put(text)

    def _sender_loop(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                requests.post(
                    f"{self.base}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text},
                    timeout=10,
                )
            except Exception as e:
                logger.warning("Gagal kirim Telegram: %s", e)

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
    bybit: BybitClient, binance_symbols: set, state: StateStore, top_n: int = 50
) -> List[str]:
    ranked = bybit.get_ranked_symbols()
    ranked_symbols = [s for s, _ in ranked if s in binance_symbols]

    excluded = set(state.positions.keys()) | set(state.bans.keys())
    universe: List[str] = []
    if "BTCUSDT" in binance_symbols and "BTCUSDT" not in excluded:
        universe.append("BTCUSDT")  # §6 — BTCUSDT wajib selalu masuk

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
        self.ws = BybitWebSocket(on_tick=self._on_tick, on_kline=self._on_kline)

        self._candle_cache: Dict[str, List[Dict[str, float]]] = {}
        # cache harga realtime dari WebSocket untuk log Telegram
        self._last_prices: Dict[str, float] = {}
        self._candle_lock = threading.Lock()
        self._trail_queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._last_update_id: Optional[int] = None
        self._last_freq_status: Optional[str] = None
        self._last_freq_alert_ts: float = 0.0

    # -------------------------------------------------------------------
    # Startup / shutdown
    # -------------------------------------------------------------------
    def startup(self) -> None:
        label_main = self.state.load_checkpoint()
        label_learn = self.learn_engine.load()
        logger.info("State dimuat (main=%s, learn=%s)", label_main, label_learn)

        ip = get_server_ip()
        # Jika dijalankan lewat try.py Render, launcher sudah menjadi pemilik Telegram getUpdates.
        # Hindari 409 Conflict karena dua polling berjalan bersamaan.
        launcher_mode = os.environ.get("RUN_WITH_LAUNCHER", "false").lower() == "true"
        # Saat dijalankan melalui try.py: polling Telegram milik launcher.
        # Tetapi sender notifikasi tetap wajib hidup.
        self.telegram.start_sender()
        self.telegram.send(
            f"🤖 BOT STARTED\n\nStatus: ONLINE\nMode: {self.state.mode}\nServer IP: {ip}\n\n"
            f"Ketik /auto untuk memulai scanning.",
            "BOT_START",
        )
        logger.info("Bot started. Server IP: %s", ip)

        self.ws.start()

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

    def shutdown(self) -> None:
        self._stop.set()
        self.telegram.send("🛑 BOT STOP\n\nStatus: OFFLINE", "BOT_STOP")
        self.ws.stop()
        self.state.save_checkpoint()
        self.learn_engine.save_checkpoint()
        self.telegram.stop()

    # -------------------------------------------------------------------
    # WebSocket callbacks (Worker 2)
    # -------------------------------------------------------------------
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
        # simpan harga terakhir agar command/log memakai data WebSocket
        self._last_prices[symbol] = price
        pos = self.state.positions.get(symbol)
        if not pos:
            return
        if pos["status"] == "PENDING":
            self._check_pending_fill(symbol, pos, price, ts)
        elif pos["status"] in ("FILLED", "PROTECTED", "TRAILING"):
            self._check_tp_sl(symbol, pos, price, ts)

    # -------------------------------------------------------------------
    # Pending -> Filled -> TP/SL/Trail lifecycle (§17-§23, §55-§58)
    # -------------------------------------------------------------------
    def _check_pending_fill(self, symbol: str, pos: Dict[str, Any], price: float, ts: float) -> None:
        direction = pos["direction"]
        entry = pos["entry"]
        filled = (direction == "BUY" and price <= entry) or (direction == "SELL" and price >= entry)
        if not filled:
            # cek timeout: TP tersentuh duluan sebelum entry (§25)
            tp_hit_first = (direction == "BUY" and price >= pos["tp"]) or (direction == "SELL" and price <= pos["tp"])
            if tp_hit_first:
                self._handle_timeout(symbol, pos)
            return

        event_id = f"{symbol}:FILLED:{pos['created_at']}"
        applied = self.state.transition(symbol, "FILLED", event_id, fill_time=ts, fill_price=price)
        if not applied:
            return  # idempotent — sudah diproses / transisi ilegal

        self.telegram.send(f"✅ FILLED — {symbol}\nEntry: {price}\nArah: {direction}", "FILLED")

        if self.state.mode == "REAL":
            self._attach_real_protection(symbol, pos)
        else:
            self.state.transition(symbol, "PROTECTED", f"{symbol}:PROTECTED:{pos['created_at']}")

    def _attach_real_protection(self, symbol: str, pos: Dict[str, Any]) -> None:
        """§17 — jangan pasang TP/SL sebelum posisi Binance benar-benar
        terkonfirmasi aktif."""
        try:
            confirmed = self.binance.get_position_risk(symbol)
            if not confirmed:
                logger.warning("Posisi %s belum terkonfirmasi di Binance, tunda pemasangan TP/SL", symbol)
                return
            side_close = "SELL" if pos["direction"] == "BUY" else "BUY"
            sl_order = self.binance.place_stop_market(symbol, side_close, Decimal(str(pos["sl"])), pos["direction"])
            tp_order = self.binance.place_take_profit_market(symbol, side_close, Decimal(str(pos["tp"])), pos["direction"])
            pos["binance_order_ids"]["sl"] = sl_order.get("orderId")
            pos["binance_order_ids"]["tp"] = tp_order.get("orderId")
            self.state.transition(symbol, "PROTECTED", f"{symbol}:PROTECTED:{pos['created_at']}")
        except RateLimitError as e:
            self._enter_binance_pause(str(e))
        except ExchangeError as e:
            logger.error("Gagal pasang protective order %s: %s", symbol, e)
            self.telegram.send(f"⚠️ ERROR — gagal pasang TP/SL {symbol}: {e}", "ERROR")

    def _check_tp_sl(self, symbol: str, pos: Dict[str, Any], price: float, ts: float) -> None:
        direction = pos["direction"]
        pos["peak_price"] = max(pos.get("peak_price", price), price) if direction == "BUY" else min(pos.get("peak_price", price), price)

        hit_tp = (direction == "BUY" and price >= pos["tp"]) or (direction == "SELL" and price <= pos["tp"])
        hit_sl = (direction == "BUY" and price <= pos["sl"]) or (direction == "SELL" and price >= pos["sl"])

        if hit_tp:
            self._close_position(symbol, pos, "TP", price, ts)
        elif hit_sl:
            outcome = "TRAIL" if pos.get("trail_count", 0) > 0 else "INITIAL_SL"
            self._close_position(symbol, pos, outcome, price, ts)

    def _close_position(self, symbol: str, pos: Dict[str, Any], outcome: str, price: float, ts: float) -> None:
        event_id = f"{symbol}:CLOSED:{outcome}:{pos['created_at']}"
        risk = abs(pos["entry"] - pos["sl"]) or 1e-9
        pnl_r = (price - pos["entry"]) / risk if pos["direction"] == "BUY" else (pos["entry"] - price) / risk
        pnl_pct = ((price - pos["entry"]) / pos["entry"] * 100) if pos["direction"] == "BUY" else ((pos["entry"] - price) / pos["entry"] * 100)

        applied = self.state.transition(
            symbol, "CLOSED", event_id, close_price=price, close_time=ts, close_reason=outcome,
            pnl_pct=pnl_pct, pnl_r=pnl_r,
        )
        if not applied:
            return

        if self.state.mode == "REAL":
            try:
                self.binance.cancel_all_open_orders(symbol)
            except Exception as e:
                logger.warning("Gagal cancel sisa order %s setelah close: %s", symbol, e)
        else:
            self.state.sim_balance += self.state.sim_balance * (pnl_pct / 100.0)

        self.learn_engine.record_trade_outcome(pos, outcome, {
            "pnl_pct": pnl_pct, "pnl_r": pnl_r, "close_time": ts, "trail_count": pos.get("trail_count", 0),
        })
        self.state.ban(symbol, outcome, 24 * 3600)  # §26 post-trade ban 24 jam
        self.telegram.send(
            f"{'🟢' if pnl_pct >= 0 else '🔴'} {outcome} {pnl_pct:+.2f}% — {symbol} | C{pos['confidence']:.0f}%",
            outcome if outcome in IMPORTANT_EVENTS else "INFO",
        )
        self.state.remove_terminal(symbol)
        self.ws.unsubscribe(symbol)

    def _handle_timeout(self, symbol: str, pos: Dict[str, Any]) -> None:
        event_id = f"{symbol}:TIMEOUT:{pos['created_at']}"
        applied = self.state.transition(symbol, "TIMEOUT", event_id, close_time=time.time() * 1000)
        if not applied:
            return
        if self.state.mode == "REAL":
            try:
                self.binance.cancel_all_open_orders(symbol)
            except Exception as e:
                logger.warning("Gagal cancel limit order timeout %s: %s", symbol, e)
        self.state.ban(symbol, "TIMEOUT", 12 * 3600)  # §25/§26
        self.telegram.send(f"⏱️ TIMEOUT — {symbol}\nTP tersentuh sebelum entry terisi.", "TIMEOUT")
        self.state.remove_terminal(symbol)
        self.ws.unsubscribe(symbol)

    def _evaluate_position_monitoring(self, symbol: str) -> None:
        pos = self.state.positions.get(symbol)
        if not pos or pos["status"] not in ("PROTECTED", "TRAILING"):
            return
        with self._candle_lock:
            candles = list(self._candle_cache.get(symbol, []))
        if not candles:
            return
        decision = self.strategy_engine.monitor_position(pos, candles)
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
        new_sl = pos.pop("_pending_trail_sl")
        reasons = pos.pop("_pending_trail_reason", [])

        old_sl = pos["sl"]
        more_protective = (new_sl > old_sl) if pos["direction"] == "BUY" else (new_sl < old_sl)
        if not more_protective:
            return

        if self.state.mode == "REAL":
            ok = self._safe_trail_update_real(symbol, pos, new_sl)
            if not ok:
                return

        pos["trail_count"] = pos.get("trail_count", 0) + 1
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
        """§21 — validate new -> create new -> verify new -> cancel old,
        untuk meminimalkan protection gap."""
        try:
            side_close = "SELL" if pos["direction"] == "BUY" else "BUY"
            new_order = self.binance.place_stop_market(symbol, side_close, Decimal(str(new_sl)), pos["direction"])
            if not new_order.get("orderId"):
                return False
            old_order_id = pos["binance_order_ids"].get("sl")
            if old_order_id:
                try:
                    self.binance.cancel_order(symbol, old_order_id)
                except Exception as e:
                    logger.error("Order SL baru %s terpasang tapi gagal hapus order lama %s: %s — REVIEW MANUAL", symbol, old_order_id, e)
                    self.telegram.send(f"⚠️ WARNING — order SL lama {symbol} gagal dihapus, cek manual!", "WARNING")
            pos["binance_order_ids"]["sl"] = new_order.get("orderId")
            return True
        except RateLimitError as e:
            self._enter_binance_pause(str(e))
            return False
        except ExchangeError as e:
            logger.error("Gagal update trailing SL %s: %s", symbol, e)
            return False

    def _enter_binance_pause(self, reason: str) -> None:
        if self.state.binance_paused:
            return
        self.state.binance_paused = True
        self.state.binance_pause_ts = time.time()
        self.telegram.send(
            f"⏸️ BINANCE PAUSE\n\nReason: {reason}\nScanner: OFF\nWebSocket: ACTIVE\nTrail monitor: ACTIVE\n\nWaiting for recovery...",
            "BINANCE_PAUSE",
        )

    def _check_binance_recovery(self) -> None:
        if not self.state.binance_paused:
            return
        if time.time() - (self.state.binance_pause_ts or 0) >= 60:
            self.state.binance_paused = False
            self.telegram.send("🟢 BINANCE READY", "BINANCE_READY")

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
                if self.state.get_active_count() >= 20:
                    time.sleep(2)
                    continue
                self._run_scan_cycle()
            except RateLimitError as e:
                self._enter_binance_pause(str(e))
            except Exception as e:  # pragma: no cover
                logger.error("Worker1 scanner error: %s", e)
                time.sleep(2)

    def _run_scan_cycle(self) -> None:
        try:
            binance_symbols = set(self.binance.exchange_info().keys())
        except Exception as e:
            logger.error("Gagal ambil exchangeInfo Binance: %s", e)
            time.sleep(5)
            return

        universe = build_universe(self.bybit, binance_symbols, self.state)
        self.state.scanned_coins = universe

        btc_candles = None
        candidates: List[strategy.Setup] = []
        processed = 0
        valid_strategy = 0
        reject_counts: Dict[str, int] = {}

        for symbol in universe:
            if self._stop.is_set() or not self.state.auto:
                break
            if self.state.is_banned(symbol) or symbol in self.state.positions:
                continue
            try:
                candles = self.bybit.get_klines(symbol, "15", 672)
            except (ExchangeError, RateLimitError) as e:
                logger.warning("Gagal ambil candle %s: %s", symbol, e)
                time.sleep(1)
                continue

            with self._candle_lock:
                self._candle_cache[symbol] = candles

            if symbol == "BTCUSDT":
                btc_candles = candles
            processed += 1

            setup = self.strategy_engine.analyze(symbol, candles, btc_candles)
            if setup:
                valid_strategy += 1
                if setup.confidence >= self.strategy_engine.get_active_threshold():
                    candidates.append(setup)
                else:
                    reject_counts["BELOW_ACTIVE_THRESHOLD"] = reject_counts.get("BELOW_ACTIVE_THRESHOLD", 0) + 1
            else:
                reject_counts["NO_VALID_ENTRY_CANDIDATE"] = reject_counts.get("NO_VALID_ENTRY_CANDIDATE", 0) + 1

            time.sleep(1)  # §5 — jeda 1 detik / coin

        candidates.sort(key=lambda s: s.confidence, reverse=True)
        slots_left = 20 - self.state.get_active_count()
        eligible = candidates[: max(0, slots_left)]

        for setup in eligible:
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
            "valid_strategy": valid_strategy, "candidate": len(candidates), "eligible": len(eligible),
            "avg_confidence": avg_conf, "rejects": reject_counts, "breadth_buy": breadth_buy,
            "breadth_sell": 100 - breadth_buy, "regime": regime,
        }
        self.learn_engine.record_scan_summary(summary)

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

    def _create_pending(self, setup: strategy.Setup) -> None:
        try:
            filters = self.binance.get_symbol_filters(setup.pair) if self.cfg.binance_api_key else {
                "step_size": Decimal("0.001"), "min_qty": Decimal("0.001"),
                "tick_size": Decimal("0.0001"), "min_notional": Decimal("5"),
            }
        except Exception as e:
            logger.warning("Gagal ambil filter %s, pakai default konservatif: %s", setup.pair, e)
            filters = {
                "step_size": Decimal("0.001"), "min_qty": Decimal("0.001"),
                "tick_size": Decimal("0.0001"), "min_notional": Decimal("5"),
            }

        ok, geom_reason = strategy.validate_geometry(
            setup.direction, setup.entry, setup.sl, setup.tp, float(filters["tick_size"]), setup.atr
        )
        if not ok:
            logger.info("Setup %s ditolak validasi geometry: %s", setup.pair, geom_reason)
            return

        qty, reason = compute_quantity(setup.entry, self.state.margin, self.state.leverage, filters)
        if qty is None:
            logger.info("Setup %s ditolak validasi quantity/margin: %s", setup.pair, reason)
            return

        self.state.add_pending(setup.to_dict(), qty, self.state.margin)
        self.ws.subscribe(setup.pair)

        if self.state.mode == "REAL":
            try:
                self.binance.set_leverage(setup.pair, int(self.state.leverage))
            except RateLimitError as e:
                self._enter_binance_pause(str(e))
                return
            except ExchangeError as e:
                # leverage gagal di-set BUKAN berarti order harus batal total —
                # tapi user harus tahu leverage aktual mungkin beda dari yang diminta.
                logger.warning("Gagal set leverage %s ke %sx: %s (lanjut pakai leverage saat ini di exchange)", setup.pair, self.state.leverage, e)
                self.telegram.send(f"⚠️ WARNING — gagal set leverage {setup.pair}: {e}", "WARNING")
            try:
                side = "BUY" if setup.direction == "BUY" else "SELL"
                self.binance.place_limit_order(
                    setup.pair, side, qty, round_price(setup.entry, filters["tick_size"]), setup.direction
                )
            except RateLimitError as e:
                self._enter_binance_pause(str(e))
                return
            except ExchangeError as e:
                logger.error("Gagal pasang limit order %s: %s", setup.pair, e)
                self.telegram.send(f"⚠️ ERROR — gagal pasang order {setup.pair}: {e}", "ERROR")
                # order Binance gagal terpasang — jangan simpan state pending yang salah (yatim, tanpa order asli)
                self.state.discard_pending(setup.pair)
                self.ws.unsubscribe(setup.pair)
                return

        current = self._last_prices.get(setup.pair, setup.entry)
        self.telegram.send(
            f"🎯 PENDING ORDER — {setup.pair}\n\n"
            f"{'🟢' if setup.direction == 'BUY' else '🔴'} {setup.direction}\n"
            f"Harga Saat Ini: {current:.6f}\n\n"
            f"Confidence: {setup.confidence:.1f}%\n\nEntry Zone: {setup.entry:.6f}\n"
            f"TP: {setup.tp:.6f}\nSL: {setup.sl:.6f}",
            "PENDING",
        )

    # -------------------------------------------------------------------
    # Worker 3 — Learn (§4, §39-§51)
    # -------------------------------------------------------------------
    def _worker_learn(self) -> None:
        last_audit = 0.0
        last_autosave = 0.0
        while not self._stop.is_set():
            try:
                self._process_trail_queue()
                now = time.time()
                if now - last_audit > 300:  # audit tiap 5 menit
                    report = self.learn_engine.audit(self.strategy_engine)
                    self._notify_audit_report(report)
                    last_audit = now
                if now - last_autosave > 120:  # autosave tiap 2 menit, tidak boleh ganggu trading (§40)
                    self.learn_engine.autosave()
                    self.state.save_checkpoint()
                    last_autosave = now
                self._check_autostop()
            except Exception as e:  # pragma: no cover
                logger.error("Worker3 learn error: %s", e)
            time.sleep(0.5)

    def _notify_audit_report(self, report: Dict[str, Any]) -> None:
        """§50/§51 — pastikan diagnosis frequency SELALU sampai ke user, tidak
        cuma saat threshold benar-benar berubah (APPLIED)."""
        action = report.get("action")
        if action == "APPLIED":
            evidence = report.get("evidence", {})
            is_exploratory = evidence.get("type", "").startswith("EXPLORATORY")
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
        """§32 — Auto Stop / Maximum Drawdown, hanya berlaku REAL TRADE."""
        if self.state.mode != "REAL" or self.state.autostop_pct is None:
            return
        try:
            balance = self.binance.get_balance_usdt()
        except Exception:
            return
        if self.state.highest_balance is None:
            self.state.highest_balance = balance
        self.state.highest_balance = max(self.state.highest_balance, balance)
        drawdown_pct = (self.state.highest_balance - balance) / self.state.highest_balance * 100
        if drawdown_pct >= self.state.autostop_pct:
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
                except Exception as e:
                    logger.error("Command error '%s': %s", text, e)
                    self.telegram.send(f"⚠️ ERROR memproses command: {e}", "ERROR")

    def _is_authorized(self, sender_id: str) -> bool:
        """§54 — hanya ALLOWED_USER_ID yang boleh mengontrol bot lewat Telegram.
        Jika ALLOWED_USER_ID tidak diset, bot fail-open (mengizinkan semua) demi
        kompatibilitas mundur, tapi ini sudah diberi warning keras saat startup."""
        if not self.cfg.allowed_user_id:
            return True
        return sender_id != "" and sender_id == str(self.cfg.allowed_user_id)

    def _dispatch_command(self, text: str) -> None:
        parts = text.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        handlers = {
            "/auto": self._cmd_auto, "/stop": self._cmd_stop, "/mode": self._cmd_mode,
            "/margin": self._cmd_margin, "/leverage": self._cmd_leverage,
            "/resetbalance": self._cmd_resetbalance, "/trade": self._cmd_trade,
            "/order": self._cmd_order, "/stats": self._cmd_stats, "/koin": self._cmd_koin,
            "/ip": self._cmd_ip, "/banned": self._cmd_banned, "/unban": self._cmd_unban,
            "/timeout": self._cmd_timeout, "/autostop": self._cmd_autostop, "/open": self._cmd_open,
            "/help": self._cmd_help,
        }
        handler = handlers.get(cmd)
        if not handler:
            return
        handler(args)

    def _cmd_help(self, args: List[str]) -> None:
        self.telegram.send(
            "🤖 COMMAND BOT\n\n"
            "/auto - Aktifkan AUTO scanning\n"
            "/stop - Matikan AUTO scanning\n"
            "/mode on|off - REAL / SIMULASI\n"
            "/margin <USDT> - Atur margin\n"
            "/leverage <angka> - Atur leverage\n"
            "/resetbalance - Reset balance anchor\n"
            "/trade - Posisi aktif/pending\n"
            "/order - Order aktif\n"
            "/open - Posisi/order terbuka\n"
            "/stats - Statistik bot\n"
            "/koin - Universe coin\n"
            "/banned - Daftar coin banned\n"
            "/unban <COIN> - Hapus ban coin\n"
            "/timeout <detik> - Atur timeout\n"
            "/autostop - Pengaturan auto stop\n"
            "/ip - IP server\n"
            "/help - Bantuan command",
            "INFO",
        )

    def _cmd_auto(self, args: List[str]) -> None:
        self.state.auto = True
        self.state.highest_balance = None  # §31 — reset anchor autostop saat /auto dijalankan
        self.telegram.send("▶️ AUTO = ON — scanning dimulai", "INFO")

    def _cmd_stop(self, args: List[str]) -> None:
        self.state.auto = False
        self.telegram.send("⏹️ AUTO = OFF — scanning dihentikan (WebSocket & trailing tetap aktif)", "INFO")

    def _cmd_mode(self, args: List[str]) -> None:
        if not args or args[0].lower() not in ("on", "off"):
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/mode on   → REAL TRADE\n/mode off  → SIMULASI", "INFO")
            return
        if args[0].lower() == "on":
            if self.state.binance_paused:
                self.telegram.send(
                    "⚠️ MODE REAL TIDAK DAPAT DIAKTIFKAN\n\nBinance sedang rate-limited.\n\nStatus:\n⏸️ PAUSED\n\nSilakan tunggu Binance kembali READY.",
                    "INFO",
                )
                return
            missing = self.cfg.validate_for_real_mode()
            if missing:
                self.telegram.send(f"⚠️ Tidak bisa mengaktifkan REAL: env var kosong: {', '.join(missing)}", "ERROR")
                return
            self.state.mode = "REAL"
            try:
                bal = self.binance.get_balance_usdt()
                self.state.highest_balance = bal
                self.telegram.send(f"🔴 MODE REAL TRADE AKTIF\n\nSaldo Binance: ${bal:.4f}\n(dijadikan anchor autostop)", "INFO")
            except Exception as e:
                logger.error("Gagal ambil balance awal REAL mode: %s", e)
                self.telegram.send(
                    f"🔴 MODE REAL TRADE AKTIF\n\n⚠️ Gagal ambil saldo Binance: {e}\n"
                    f"Autostop TIDAK punya anchor sampai berhasil — cek /resetbalance setelah masalah teratasi.",
                    "ERROR",
                )
        else:
            self.state.mode = "SIMULASI"
            self.telegram.send("🧪 MODE SIMULASI AKTIF", "INFO")

    def _cmd_margin(self, args: List[str]) -> None:
        if not args:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/margin <USDT>\n\nContoh:\n/margin 1", "INFO")
            return
        try:
            value = float(args[0])
            if value <= 0:
                raise ValueError
        except ValueError:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/margin <USDT>\n\nContoh:\n/margin 1", "INFO")
            return
        self.state.margin = value  # §60 — perubahan margin memicu recalculation quantity utk pending berikutnya
        self.telegram.send(f"✅ MARGIN SUCCESS — margin diatur ke ${value}", "MARGIN_SUCCESS")

    def _cmd_leverage(self, args: List[str]) -> None:
        if not args:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/leverage <angka>\n\nContoh:\n/leverage 5", "INFO")
            return
        try:
            value = float(args[0])
            if value <= 0:
                raise ValueError
        except ValueError:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/leverage <angka>\n\nContoh:\n/leverage 5", "INFO")
            return
        self.state.leverage = value
        self.telegram.send(f"✅ LEVERAGE SUCCESS — leverage diatur ke {value}x", "LEVERAGE_SUCCESS")

    def _cmd_resetbalance(self, args: List[str]) -> None:
        if self.state.binance_paused:
            self.telegram.send("⚠️ Tidak bisa reset balance — Binance sedang rate-limited.", "INFO")
            return
        if self.state.mode == "SIMULASI":
            self.state.sim_balance = 10.0
            self.state.sim_balance_anchor = 10.0
            self.telegram.send("✅ Balance simulasi direset ke $10.0000", "INFO")
        else:
            try:
                bal = self.binance.get_balance_usdt()
                old_anchor = self.state.highest_balance
                self.state.highest_balance = bal
                prefix = f"Sebelumnya: ${old_anchor:.4f}\n" if old_anchor is not None else ""
                self.telegram.send(f"✅ Anchor balance REAL direset\n\n{prefix}Sekarang: ${bal:.4f}", "INFO")
            except Exception as e:
                logger.error("Gagal ambil balance Binance saat /resetbalance: %s", e)
                self.telegram.send(f"⚠️ Gagal ambil balance Binance: {e}", "ERROR")

    def _cmd_trade(self, args: List[str]) -> None:
        positions = self.state.snapshot_positions()
        active = [p for p in positions if p["status"] not in TERMINAL_STATES]
        lines = [f"📡 Posisi Aktif ({len(active)}/20)\n"]
        for p in active:
            icon = "🟢" if p["direction"] == "BUY" else "🔴"
            if p["status"] == "PENDING":
                current = self._last_prices.get(p['pair'], p['entry'])
                jarak = abs(current - p['entry']) / p['entry'] * 100 if p['entry'] else 0.0
                lines.append(
                    f"⏳ {p['pair']} — PENDING\n{icon} {p['direction']}\n"
                    f"Entry zone: {p['entry']:.6f}\nHarga kini: {current:.6f}\nJarak: {jarak:.2f}%\n"
                    f"TP: {p['tp']:.6f}\nSL: {p['sl']:.6f}\nConfidence: {p['confidence']:.0f}%\n"
                )
            else:
                current = self._last_prices.get(p['pair'], p['entry'])
                if p['direction'] == 'BUY':
                    pnl = ((current - p['entry']) / p['entry']) * 100
                else:
                    pnl = ((p['entry'] - current) / p['entry']) * 100
                fill_time = p.get("fill_time")
                jam = time.strftime("%H:%M", time.localtime(fill_time / 1000)) if fill_time else "-"
                lines.append(
                    f"{icon} {p['pair']} — {p['status']}\n"
                    f"Entry: {p['entry']:.6f}\nHarga: {current:.6f}\n"
                    f"TP: {p['tp']:.6f}\nSL: {p['sl']:.6f}\n"
                    f"Confidence: {p['confidence']:.0f}%\nPnL: {pnl:+.2f}%\n🕐 Entry: {jam}\n"
                )
        self.telegram.send("\n".join(lines) if active else "📡 Tidak ada posisi aktif/pending.", "INFO")

    def _cmd_order(self, args: List[str]) -> None:
        positions = [p for p in self.state.snapshot_positions() if p["status"] not in TERMINAL_STATES]
        if not positions:
            self.telegram.send("📋 ORDER\n\nTidak ada order aktif.", "INFO")
            return
        lines = ["📋 ORDER\n"]
        for i, p in enumerate(positions, 1):
            icon = "🟢" if p["direction"] == "BUY" else "🔴"
            current = self._last_prices.get(p['pair'], p['entry'])
            sl_label = f"Trail SL: {p['sl']:.6f}" if p.get("trail_count", 0) > 0 else f"SL: {p['sl']:.6f}"
            if p["status"] == "PENDING":
                pnl_line = ""
            else:
                pnl = ((current - p['entry']) / p['entry']) * 100 if p['direction'] == 'BUY' else ((p['entry'] - current) / p['entry']) * 100
                pnl_line = f"PnL: {pnl:+.2f}%\n"
            lines.append(
                f"{i}. {p['pair']} — {p['status']}\n{icon} {p['direction']}\n"
                f"Entry: {p['entry']:.6f}\nTP: {p['tp']:.6f}\n{sl_label}\n"
                f"{pnl_line}Confidence: {p['confidence']:.0f}%\n"
            )
        self.telegram.send("\n".join(lines), "INFO")

    def _cmd_stats(self, args: List[str]) -> None:
        stats = self.learn_engine.overall_stats()
        n = stats["n"]
        outcome_counts = stats.get("outcome_counts", {})
        lines = [f"📊 Statistik — {n} trade"]
        lines.append(f"TP {outcome_counts.get('TP',0)} | Initial SL {outcome_counts.get('INITIAL_SL',0)} | Trail {outcome_counts.get('TRAIL',0)}")
        lines.append(f"\nMode: {'🧪 SIMULASI' if self.state.mode=='SIMULASI' else '🔴 REAL TRADE'}")
        lines.append(f"\nWin rate: {stats['win_rate']:.1f}%\nExpectancy: {stats['expectancy']:.3f}R\nProfit factor: {stats['profit_factor']}")
        if self.state.mode == "SIMULASI":
            lines.append(f"\nModal anchor:\n${self.state.sim_balance_anchor:.4f} → Saldo statistik: ${self.state.sim_balance:.4f}")
        lines.append(f"\n🚫 Banned: {len(self.state.active_bans())}")
        self.telegram.send("\n".join(lines), "INFO")

    def _cmd_koin(self, args: List[str]) -> None:
        lines = ["🪙 COIN SCANNED\n"]
        for i, s in enumerate(self.state.scanned_coins, 1):
            lines.append(f"{i}. {s}")
        self.telegram.send("\n".join(lines) if self.state.scanned_coins else "🪙 Belum ada coin yang discan.", "INFO")

    def _cmd_ip(self, args: List[str]) -> None:
        ip = get_server_ip()
        self.telegram.send(f"🌐 SERVER INFO\n\nIP: {ip}\nStatus: ONLINE", "INFO")

    def _cmd_banned(self, args: List[str]) -> None:
        bans = self.state.active_bans()
        if not bans:
            self.telegram.send("🚫 BANNED COINS\n\nTidak ada coin yang diban.", "INFO")
            return
        lines = ["🚫 BANNED COINS\n"]
        now = time.time()
        for symbol, info in bans.items():
            remaining = max(0, info["expiry"] - now)
            h, m = int(remaining // 3600), int((remaining % 3600) // 60)
            lines.append(f"{symbol}\nReason: {info['reason']}\nRemaining: {h:02d}h {m:02d}m\n")
        self.telegram.send("\n".join(lines), "INFO")

    def _cmd_unban(self, args: List[str]) -> None:
        if not args:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/unban BTCUSDT\natau\n/unban All", "INFO")
            return
        self.state.unban(args[0])
        self.telegram.send(f"✅ UNBANNED — {args[0]}", "UNBANNED")

    def _cmd_timeout(self, args: List[str]) -> None:
        if not args:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/timeout All\natau\n/timeout BTCUSDT", "INFO")
            return
        targets = list(self.state.positions.keys()) if args[0].lower() == "all" else [args[0]]
        for symbol in targets:
            pos = self.state.positions.get(symbol)
            if not pos:
                continue
            if self.state.mode == "REAL":
                try:
                    self.binance.cancel_all_open_orders(symbol)
                except Exception as e:
                    logger.warning("Gagal bersihkan order %s saat manual timeout: %s", symbol, e)
            # manual timeout TIDAK dimasukkan sbg data pembelajaran learn.py (§29)
            self.state.transition(symbol, "CANCELLED", f"{symbol}:MANUAL_TIMEOUT:{time.time()}")
            self.state.remove_terminal(symbol)
            self.ws.unsubscribe(symbol)
        self.telegram.send(f"✅ Timeout selesai untuk: {', '.join(targets)}", "INFO")

    def _cmd_autostop(self, args: List[str]) -> None:
        if not args:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/autostop <percentage>\n\nContoh:\n/autostop 10", "INFO")
            return
        try:
            pct = float(args[0])
        except ValueError:
            self.telegram.send("⚠️ Format salah.\n\nGunakan:\n/autostop <percentage>", "INFO")
            return
        self.state.autostop_pct = pct
        self.telegram.send(f"✅ Autostop diatur ke {pct}% (hanya berlaku untuk REAL TRADE)", "INFO")

    def _cmd_open(self, args: List[str]) -> None:
        label = self.learn_engine.load()
        self.telegram.send(f"📂 Learning memory dimuat dari checkpoint: {label}", "INFO")


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
        _LAUNCHER_BOT.shutdown()
        _LAUNCHER_BOT = None


if __name__ == "__main__":
    main()
