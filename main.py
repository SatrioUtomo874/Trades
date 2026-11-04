from __future__ import annotations

"""Volumatic Trend Futures bot.

Launcher contract preserved: on_start(context), handle_update(update, context), on_stop(context).
The legacy SMC/M5 strategy engine is intentionally not included.
"""
import asyncio, base64, hashlib, hmac, json, logging, math, os, re, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
import traceback
from dotenv import load_dotenv
from websockets.asyncio.client import connect as ws_connect

import strategy

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / "trades.env")
load_dotenv(BASE_DIR / ".env")

ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0") or "0")
if not ALLOWED_USER_ID:
    raise RuntimeError("ALLOWED_USER_ID belum diset; bot tidak akan menerima command tanpa pembatas user.")
BINANCE_API_KEY = (os.getenv("BINANCE_API_KEY") or "").strip()
BINANCE_API_SECRET = (os.getenv("BINANCE_API_SECRET") or "").strip()
BINANCE_REST_BASE = "https://fapi.binance.com"
BINANCE_USER_WS_BASE = "wss://fstream.binance.com/ws/"
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()
STATE_PATH = "data/volumatic_state.json"
LOCAL_STATE = BASE_DIR / "volumatic_state.json"
DEFAULT_TIMEFRAME = "H4"
DEFAULT_MAX = int(os.getenv("MAX_ACTIVE_TRADES", "100") or "100")
DEFAULT_MARGIN = Decimal(os.getenv("MARGIN_USD", "0.5") or "0.5")
DEFAULT_LEVERAGE = int(os.getenv("LEVERAGE", "10") or "10")
SCAN_BATCH_SIZE = 50
SCAN_KLINE_LIMIT = 260
SCAN_PAIR_DELAY = float(os.getenv("SCAN_PAIR_DELAY_SECONDS", "0.12") or "0.12")
MONITOR_SECONDS = float(os.getenv("ORDER_MONITOR_SECONDS", "20") or "20")
UNIVERSE_REFRESH_SECONDS = 6 * 60 * 60
HTTP_TIMEOUT = 20
RECV_WINDOW = 5000
TF_MS = {"H4": 14_400_000, "D1": 86_400_000}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("volumatic.bot")


def utc_ms() -> int:
    return int(time.time() * 1000)


def iso(ms: int | None) -> str:
    if not ms:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def dec(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default


def dstr(value: Decimal) -> str:
    return format(value.normalize(), "f")


def safe_symbol(value: str) -> str:
    s = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    if not s.endswith("USDT"):
        raise ValueError("Simbol harus berupa pair USDT")
    return s


class BinanceRateLimit(RuntimeError):
    pass


class BinanceClient:
    def __init__(self) -> None:
        self.key = BINANCE_API_KEY
        self.secret = BINANCE_API_SECRET
        self.offset_ms = 0
        self.cooldown_until = 0.0
        self._lock = asyncio.Lock()
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.key} if self.key else {})
        self.exchange_info: dict[str, dict[str, Any]] = {}
        self.exchange_info_at = 0.0
        self.last_weight = "?"

    @property
    def cooldown(self) -> float:
        return max(0.0, self.cooldown_until - time.monotonic())

    def _signed_params(self, params: dict[str, Any]) -> dict[str, Any]:
        p = dict(params)
        p["timestamp"] = utc_ms() + self.offset_ms
        p["recvWindow"] = RECV_WINDOW
        query = urlencode(p, doseq=True)
        p["signature"] = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return p

    async def request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        async with self._lock:
            wait = self.cooldown
            if wait > 0:
                await asyncio.sleep(wait)
            p = dict(params or {})
            if signed:
                if not self.key or not self.secret:
                    raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET belum tersedia")
                p = self._signed_params(p)
            def _do():
                return self.session.request(method, BINANCE_REST_BASE + path, params=p, timeout=HTTP_TIMEOUT)
            response = await asyncio.to_thread(_do)
            self.last_weight = response.headers.get("X-MBX-USED-WEIGHT-1M", self.last_weight)
            try:
                payload = response.json()
            except Exception:
                payload = {"msg": response.text[:300]}
            if response.status_code in (418, 429) or (isinstance(payload, dict) and payload.get("code") in (-1003, -1015)):
                retry = dec(response.headers.get("Retry-After"), Decimal("60"))
                seconds = max(30.0, float(retry) + 5.0)
                self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)
                raise BinanceRateLimit(f"Binance API limit ({response.status_code}); cooldown {seconds:.0f}s")
            if response.status_code >= 400:
                msg = payload.get("msg", payload) if isinstance(payload, dict) else payload
                raise RuntimeError(f"Binance {path} HTTP {response.status_code}: {msg}")
            return payload

    async def sync_time(self) -> None:
        data = await self.request("GET", "/fapi/v1/time")
        self.offset_ms = int(data["serverTime"]) - utc_ms()

    async def klines(self, symbol: str, interval: str, limit: int = SCAN_KLINE_LIMIT) -> list[list[Any]]:
        return await self.request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    async def tickers(self) -> list[dict[str, Any]]:
        return await self.request("GET", "/fapi/v1/ticker/24hr")

    async def get_exchange_info(self, force: bool = False) -> dict[str, dict[str, Any]]:
        if self.exchange_info and not force and time.monotonic() - self.exchange_info_at < 6 * 3600:
            return self.exchange_info
        data = await self.request("GET", "/fapi/v1/exchangeInfo")
        out = {}
        for item in data.get("symbols", []):
            if item.get("status") != "TRADING" or item.get("contractType") != "PERPETUAL" or item.get("quoteAsset") != "USDT":
                continue
            filters = {f.get("filterType"): f for f in item.get("filters", [])}
            lot = filters.get("LOT_SIZE", {})
            price = filters.get("PRICE_FILTER", {})
            notional = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {}))
            out[item["symbol"]] = {
                "tick_size": dec(price.get("tickSize")),
                "step_size": dec(lot.get("stepSize")),
                "min_qty": dec(lot.get("minQty")),
                "max_qty": dec(lot.get("maxQty")),
                "min_notional": dec(notional.get("notional", notional.get("minNotional", 5))),
            }
        self.exchange_info, self.exchange_info_at = out, time.monotonic()
        return out

    async def account_config(self) -> dict[str, Any]:
        return await self.request("GET", "/fapi/v1/accountConfig", signed=True)

    async def balance(self) -> list[dict[str, Any]]:
        return await self.request("GET", "/fapi/v2/balance", signed=True)

    async def positions(self) -> list[dict[str, Any]]:
        return await self.request("GET", "/fapi/v2/positionRisk", signed=True)

    async def open_orders(self) -> list[dict[str, Any]]:
        return await self.request("GET", "/fapi/v1/openOrders", signed=True)

    async def order(self, symbol: str, order_id: int | None = None, client_id: str | None = None) -> dict[str, Any]:
        p: dict[str, Any] = {"symbol": symbol}
        if order_id is not None: p["orderId"] = order_id
        if client_id: p["origClientOrderId"] = client_id
        return await self.request("GET", "/fapi/v1/order", p, signed=True)

    async def cancel(self, symbol: str, order_id: int) -> dict[str, Any]:
        return await self.request("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    async def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return await self.request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    async def limit(self, symbol: str, side: str, qty: Decimal, price: Decimal, client_id: str) -> dict[str, Any]:
        return await self.request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTC",
            "quantity": dstr(qty), "price": dstr(price), "newClientOrderId": client_id,
            "newOrderRespType": "RESULT",
        }, signed=True)

    async def market_close(self, symbol: str, side: str, qty: Decimal) -> dict[str, Any]:
        return await self.request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side, "type": "MARKET", "quantity": dstr(qty),
            "reduceOnly": "true", "newOrderRespType": "RESULT",
        }, signed=True)

    async def create_listen_key(self) -> str:
        result = await self.request("POST", "/fapi/v1/listenKey")
        return str(result["listenKey"])

    async def keepalive_listen_key(self) -> None:
        await self.request("PUT", "/fapi/v1/listenKey")

    async def close_listen_key(self) -> None:
        await self.request("DELETE", "/fapi/v1/listenKey")


class GitHubState:
    def __init__(self):
        self.token, self.repo, self.branch = GITHUB_TOKEN, REPO_NAME, GITHUB_BRANCH
        self.sha: str | None = None

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

    def load(self) -> dict[str, Any] | None:
        if not (self.token and "/" in self.repo): return None
        url = f"https://api.github.com/repos/{self.repo}/contents/{STATE_PATH}"
        r = requests.get(url, headers=self._headers(), params={"ref": self.branch}, timeout=20)
        if r.status_code == 404: return None
        r.raise_for_status()
        item = r.json(); self.sha = item.get("sha")
        return json.loads(base64.b64decode(item["content"]).decode("utf-8"))

    def save(self, state: dict[str, Any]) -> None:
        if not (self.token and "/" in self.repo):
            LOCAL_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8"); return
        url = f"https://api.github.com/repos/{self.repo}/contents/{STATE_PATH}"
        payload = {"message": "Update Volumatic bot state", "branch": self.branch,
                   "content": base64.b64encode(json.dumps(state, separators=(",", ":")).encode()).decode()}
        if self.sha: payload["sha"] = self.sha
        r = requests.put(url, headers=self._headers(), json=payload, timeout=25)
        if r.status_code >= 400:
            try: detail = (r.json() or {}).get("message", r.text[:500])
            except Exception: detail = r.text[:500]
            remaining = r.headers.get("X-RateLimit-Remaining", "?")
            reset = r.headers.get("X-RateLimit-Reset", "?")
            if r.status_code == 403 and remaining == "0":
                hint = f"GitHub API rate limit habis (reset epoch {reset})."
            elif r.status_code == 403:
                hint = "GitHub menolak penulisan. Pastikan token memiliki Contents: Read and write, akses ke repo, dan branch tidak dilindungi aturan yang memblokir commit."
            else:
                hint = "GitHub Contents API gagal menyimpan state."
            raise RuntimeError(f"{hint} HTTP {r.status_code}: {detail}")
        self.sha = (r.json().get("content") or {}).get("sha")


@dataclass
class Pending:
    symbol: str
    side: str
    timeframe: str
    signal_time: int
    next_open_time: int
    expiry_time: int
    limit_price: str
    quantity: str
    client_order_id: str
    order_id: int | None = None
    status: str = "NEW"
    reversal: bool = False


class VolumaticEngine:
    def __init__(self, context: dict[str, Any]):
        self.context = dict(context)
        self.chat_id = int(context.get("chat_id") or ALLOWED_USER_ID)
        self.send_message = context["send_message"]
        self.binance = BinanceClient()
        self.github = GitHubState()
        self.timeframe = DEFAULT_TIMEFRAME
        self.max_slots = DEFAULT_MAX
        self.margin = DEFAULT_MARGIN
        self.min_mode = False
        self.leverage = DEFAULT_LEVERAGE
        self.real_mode = False  # always OFF after process restart
        self.scan_enabled = True
        self.stats_notify = False
        self.stats = {"scan_cycles": 0, "symbols_scanned": 0, "scan_errors": 0,
                      "long_bubbles": 0, "short_bubbles": 0, "orders_created": 0,
                      "filled_orders": 0, "closed_trades": 0, "wins": 0, "losses": 0,
                      "net_pnl": "0", "gross_profit": "0", "gross_loss": "0",
                      "realized_equity": "0", "equity_peak": "0", "max_drawdown": "0",
                      "best_trade": "0", "worst_trade": "0"}
        self.closed_trades: list[dict[str, Any]] = []
        self._error_last_sent: dict[str, float] = {}
        self._github_error_notified = False
        self._github_retry_after = 0.0
        self.universe: list[str] = []
        self.filters: dict[str, dict[str, Any]] = {}
        self.last_processed: dict[str, int] = {}
        self.scan_baseline_ms = utc_ms()
        self.pending: dict[str, Pending] = {}
        self.positions: dict[str, dict[str, Any]] = {}
        self.managed_symbols: set[str] = set()
        self.position_wait_until: dict[str, int] = {}
        self.deferred_reversals: dict[str, dict[str, Any]] = {}
        self.deferred_entries: dict[str, dict[str, Any]] = {}
        self.deferred_signals: dict[str, dict[str, Any]] = {}
        self.close_requests: set[str] = set()
        self.scan_task: asyncio.Task | None = None
        self.monitor_task: asyncio.Task | None = None
        self.user_ws_task: asyncio.Task | None = None
        self.keepalive_task: asyncio.Task | None = None
        self.listen_key: str | None = None
        self.stop_event = asyncio.Event()
        self.scan_wakeup = asyncio.Event()
        self.command_queue: list[str] = []
        self.scan_lock = asyncio.Lock()
        self.state_lock = asyncio.Lock()
        self.last_universe_refresh = 0.0
        self.started = False
        self._last_state_fingerprint: str | None = None

    async def say(self, text: str):
        try: await asyncio.to_thread(self.send_message, self.chat_id, str(text))
        except Exception: log.exception("Telegram send failed")

    async def report_error(self, where: str, exc: BaseException, *, cooldown: float = 60.0):
        """Log + forward unique errors to Telegram, with per-error throttling."""
        key = f"{type(exc).__name__}|{str(exc)[:240]}"
        now = time.monotonic()
        if now - self._error_last_sent.get(key, 0.0) < cooldown:
            return
        self._error_last_sent[key] = now
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log.error("%s: %s\n%s", where, exc, detail[-5000:])
        message = f"🚨 <b>BOT ERROR</b>\nLokasi: <code>{where}</code>\nTipe: <code>{type(exc).__name__}</code>\n<pre>{str(exc)[:1800]}</pre>"
        await self.say(message)

    async def persist(self):
        data = {
            "version": 3, "timeframe": self.timeframe, "max_slots": self.max_slots,
            "margin": str(self.margin), "min_mode": self.min_mode, "leverage": self.leverage,
            "scan_enabled": self.scan_enabled,
            "stats_notify": self.stats_notify, "stats": self.stats,
            "closed_trades": self.closed_trades[-500:],
            "last_processed": self.last_processed, "scan_baseline_ms": self.scan_baseline_ms,
            "pending": {s: asdict(p) for s, p in self.pending.items()},
            "positions": {s: {k:v for k,v in p.items() if k != "unRealizedProfit"} for s,p in self.positions.items()},
            "managed_symbols": sorted(self.managed_symbols), "position_wait_until": self.position_wait_until,
            "deferred_reversals": self.deferred_reversals,
            "deferred_entries": self.deferred_entries, "deferred_signals": self.deferred_signals,
            "close_requests": sorted(self.close_requests),
        }
        fingerprint = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        if fingerprint == self._last_state_fingerprint:
            return
        if time.monotonic() < self._github_retry_after:
            try: LOCAL_STATE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            except Exception as local_exc: await self.report_error("persist/local-fallback", local_exc)
            return
        try:
            await asyncio.to_thread(self.github.save, data)
            self._last_state_fingerprint = fingerprint
            self._github_error_notified = False
            self._github_retry_after = 0.0
        except Exception as exc:
            self._github_retry_after = time.monotonic() + 300.0
            log.warning("GitHub state persistence failed; using local fallback for 5 minutes: %s", exc)
            try:
                LOCAL_STATE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            except Exception as local_exc:
                await self.report_error("persist/local-fallback", local_exc)
            if not self._github_error_notified:
                self._github_error_notified = True
                await self.report_error("persist/github", exc, cooldown=3600)

    async def restore(self):
        try: data = await asyncio.to_thread(self.github.load)
        except Exception as exc:
            await self.report_error("restore/github-state", exc, cooldown=3600)
            data = None
        if not data and LOCAL_STATE.exists():
            try: data = json.loads(LOCAL_STATE.read_text(encoding="utf-8"))
            except Exception: data = None
        if not isinstance(data, dict): return
        tf = str(data.get("timeframe", DEFAULT_TIMEFRAME)).upper()
        self.timeframe = tf if tf in TF_MS else DEFAULT_TIMEFRAME
        self.max_slots = max(1, int(data.get("max_slots", DEFAULT_MAX)))
        self.margin = max(Decimal("0.01"), dec(data.get("margin"), DEFAULT_MARGIN))
        self.min_mode = bool(data.get("min_mode", False))
        self.leverage = max(1, min(125, int(data.get("leverage", DEFAULT_LEVERAGE))))
        self.scan_enabled = bool(data.get("scan_enabled", True))
        self.stats_notify = bool(data.get("stats_notify", False))
        self.stats.update(data.get("stats") or {})
        self.closed_trades = list(data.get("closed_trades") or [])[-500:]
        self.last_processed = {str(k): int(v) for k, v in (data.get("last_processed") or {}).items()}
        for sym, raw in (data.get("pending") or {}).items():
            try: self.pending[sym] = Pending(**raw)
            except Exception: log.warning("Invalid pending state %s", sym)
        self.positions = data.get("positions") or {}
        self.managed_symbols = set(data.get("managed_symbols") or []) | set(self.positions) | set(self.pending)
        self.position_wait_until = {str(k):int(v) for k,v in (data.get("position_wait_until") or {}).items()}
        self.deferred_reversals = data.get("deferred_reversals") or {}
        self.deferred_entries = data.get("deferred_entries") or {}
        self.deferred_signals = data.get("deferred_signals") or {}
        self.close_requests = set(data.get("close_requests") or [])

    async def start(self):
        await self.restore()
        self.scan_baseline_ms = utc_ms()
        if BINANCE_API_KEY and BINANCE_API_SECRET:
            try:
                await self.binance.sync_time()
                await self._refresh_universe(force=True)
                await self._reconcile()
            except Exception as exc:
                await self.report_error("startup/binance-sync", exc)
        self.stop_event.clear()
        self.scan_task = asyncio.create_task(self._scan_loop(), name="volumatic-scan")
        self.monitor_task = asyncio.create_task(self._monitor_loop(), name="volumatic-monitor")
        self.started = True
        await self.say(f"<b>Volumatic Trend bot aktif</b>\nTimeframe: <b>{self.timeframe}</b>\nMode REAL: OFF\nUniverse: {len(self.universe)} pair\nStrategi: LONG/SHORT bubble reversal, tanpa SL/TP.")

    async def stop(self):
        self.stop_event.set()
        for task in (self.scan_task, self.monitor_task, self.user_ws_task, self.keepalive_task):
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in (self.scan_task, self.monitor_task, self.user_ws_task, self.keepalive_task) if t), return_exceptions=True)
        if self.listen_key and self.binance.key:
            try: await asyncio.wait_for(self.binance.close_listen_key(), timeout=5)
            except Exception: log.warning("listenKey close skipped/failed during shutdown")
        await self.persist()
        self.started = False

    async def _refresh_universe(self, force: bool = False):
        if self.universe and not force and time.monotonic() - self.last_universe_refresh < UNIVERSE_REFRESH_SECONDS:
            return
        info = await self.binance.get_exchange_info(force=force)
        tickers = await self.binance.tickers()
        volume = {x.get("symbol"): dec(x.get("quoteVolume")) for x in tickers if x.get("symbol")}
        # Preserve the existing altcoin-universe intent: USDT perpetuals, excluding BTC/ETH.
        symbols = [s for s in info if s not in {"BTCUSDT", "ETHUSDT"}]
        symbols.sort(key=lambda s: volume.get(s, Decimal("0")), reverse=True)
        # Exclude the five largest 24h quote-volume pairs, preserving the user's
        # altcoin-universe preference from the previous scanner.
        excluded_top = set(symbols[:5])
        symbols = [s for s in symbols if s not in excluded_top]
        self.universe, self.filters = symbols, info
        self.last_universe_refresh = time.monotonic()
        log.info("Universe refreshed: %d USDT perpetual symbols", len(symbols))

    def _next_close_ms(self, now: int | None = None) -> int:
        now = int(now or utc_ms()); interval = TF_MS[self.timeframe]
        return ((now // interval) + 1) * interval

    async def _scan_loop(self):
        while not self.stop_event.is_set():
            try:
                if not self.scan_enabled:
                    self.scan_wakeup.clear()
                    try:
                        await asyncio.wait_for(self.scan_wakeup.wait(), timeout=60)
                    except asyncio.TimeoutError:
                        pass
                    continue
                await self._refresh_universe()
                target = self._next_close_ms() + 5000
                self.scan_wakeup.clear()
                try:
                    await asyncio.wait_for(self.scan_wakeup.wait(), timeout=max(1.0, (target - utc_ms()) / 1000))
                    continue
                except asyncio.TimeoutError:
                    pass
                if self.stop_event.is_set() or not self.scan_enabled:
                    continue
                await self.scan_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self.report_error("scan-loop", exc)
                await asyncio.sleep(30)

    async def scan_cycle(self, manual: bool = False):
        """Scan all symbols; rate limits are handled here so task exceptions never leak."""
        async with self.scan_lock:
            try:
                await self._refresh_universe()
                tf = self.timeframe
                interval = TF_MS[tf]
                interval_name = "4h" if tf == "H4" else "1d"
                symbols = list(self.universe)
                if not symbols:
                    await self.say("⚠️ Scan dibatalkan: universe kosong.")
                    return
                total_batches = (len(symbols) + SCAN_BATCH_SIZE - 1) // SCAN_BATCH_SIZE
                cycle_counts = {"scanned": 0, "valid": 0, "long": 0, "short": 0, "none": 0, "errors": 0}
                log.info("Starting %s scan: %d symbols in batches of %d", tf, len(symbols), SCAN_BATCH_SIZE)
                await self.say(f"🔎 <b>SCAN #{int(self.stats.get('scan_cycles', 0)) + 1} DIMULAI</b>\nTimeframe: {tf} | {len(symbols)} koin | {total_batches} batch")
                for batch_index, start in enumerate(range(0, len(symbols), SCAN_BATCH_SIZE), start=1):
                    if self.stop_event.is_set():
                        break
                    if not manual and not self.scan_enabled:
                        log.info("Auto scan paused by /scan off")
                        break
                    batch = symbols[start:start + SCAN_BATCH_SIZE]
                    b = {"scanned": 0, "valid": 0, "long": 0, "short": 0, "none": 0, "errors": 0}
                    coin_lines: list[str] = []
                    batch_rate_abort = False
                    log.info("Scan batch %d/%d | symbols %d-%d / %d (%s)", batch_index, total_batches, start + 1, start + len(batch), len(symbols), tf)
                    for symbol in batch:
                        if self.stop_event.is_set():
                            break
                        attempts = 0
                        while True:
                            try:
                                rows = await self.binance.klines(symbol, interval_name, SCAN_KLINE_LIMIT)
                                break
                            except BinanceRateLimit as exc:
                                attempts += 1
                                await self.report_error(f"scan/rate-limit/{symbol}", exc, cooldown=180)
                                wait = max(5.0, self.binance.cooldown + 1.0)
                                await self.say(f"⏸️ Binance 429 saat {symbol}. Menunggu {wait:.0f} detik lalu melanjutkan simbol yang sama; scan tidak crash dan order exchange tidak disentuh.")
                                if attempts >= 3:
                                    rows = None
                                    break
                                await asyncio.sleep(wait)
                            except Exception as exc:
                                rows = None
                                b["errors"] += 1; cycle_counts["errors"] += 1
                                self.stats["scan_errors"] = int(self.stats.get("scan_errors", 0)) + 1
                                coin_lines.append(f"❌ {symbol}: ERROR {type(exc).__name__} — {str(exc)[:120]}")
                                await self.report_error(f"scan/{symbol}", exc)
                                break
                        if rows is None:
                            if attempts >= 3:
                                b["errors"] += 1; cycle_counts["errors"] += 1
                                self.stats["scan_errors"] = int(self.stats.get("scan_errors", 0)) + 1
                                coin_lines.append(f"⚠️ {symbol}: ditunda setelah 3 kali rate-limit; dicoba lagi pada cycle berikutnya")
                                batch_rate_abort = True
                                await asyncio.sleep(SCAN_PAIR_DELAY)
                                break
                            await asyncio.sleep(SCAN_PAIR_DELAY)
                            continue
                        b["scanned"] += 1; cycle_counts["scanned"] += 1
                        closed = [r for r in rows if len(r) >= 7 and int(r[6]) < utc_ms()]
                        if len(closed) < strategy.ATR_LENGTH + strategy.LENGTH + 5:
                            b["errors"] += 1; cycle_counts["errors"] += 1
                            self.stats["scan_errors"] = int(self.stats.get("scan_errors", 0)) + 1
                            coin_lines.append(f"⚠️ {symbol}: data candle belum cukup")
                            await asyncio.sleep(SCAN_PAIR_DELAY)
                            continue
                        latest_open = int(closed[-1][0])
                        previous = self.last_processed.get(symbol)
                        if previous is not None and latest_open <= previous:
                            b["valid"] += 1; cycle_counts["valid"] += 1
                            b["none"] += 1; cycle_counts["none"] += 1
                            coin_lines.append(f"· {symbol}: tidak ada candle baru sejak scan sebelumnya")
                            await asyncio.sleep(SCAN_PAIR_DELAY)
                            continue
                        candles = [{"open_time":int(r[0]),"open":float(r[1]),"high":float(r[2]),"low":float(r[3]),"close":float(r[4]),"volume":float(r[5])} for r in closed[-SCAN_KLINE_LIMIT:]]
                        signals = strategy.signals_since(symbol, tf, candles, interval,
                            after_open_time=previous,
                            after_close_time_ms=self.scan_baseline_ms if previous is None else None)
                        self.last_processed[symbol] = latest_open
                        b["valid"] += 1; cycle_counts["valid"] += 1
                        if signals:
                            for sig in signals:
                                direction = str(sig.get("direction", "")).upper()
                                if direction == "LONG": b["long"] += 1; cycle_counts["long"] += 1
                                elif direction == "SHORT": b["short"] += 1; cycle_counts["short"] += 1
                                await self._on_signal(sig.as_dict())
                            coin_lines.append(f"🫧 {symbol}: {', '.join(str(x.direction) for x in signals)} bubble terkonfirmasi")
                        else:
                            b["none"] += 1; cycle_counts["none"] += 1
                            coin_lines.append(f"· {symbol}: tidak ada bubble baru")
                        await asyncio.sleep(SCAN_PAIR_DELAY)
                    self.stats["symbols_scanned"] = int(self.stats.get("symbols_scanned", 0)) + b["scanned"]
                    self.stats["scan_errors"] = int(self.stats.get("scan_errors", 0))
                    log.info("Batch %d/%d conclusion: scanned=%d valid=%d LONG=%d SHORT=%d no_signal=%d errors=%d", batch_index, total_batches, b["scanned"], b["valid"], b["long"], b["short"], b["none"], b["errors"])
                    summary = (f"📦 <b>Batch {batch_index}/{total_batches} selesai</b> ({tf})\n"
                               f"Koin: {b['scanned']}/{len(batch)} | LONG: {b['long']} | SHORT: {b['short']}\n"
                               f"Tanpa bubble/baru: {b['none']} | Error/kurang data: {b['errors']}")
                    if coin_lines:
                        summary += "\n\n<b>Kesimpulan setiap koin:</b>\n" + "\n".join(coin_lines)
                    await self.say(summary)
                    await self.persist()
                    if batch_rate_abort:
                        await self.say("⏸️ Scan dijeda karena Binance masih mengembalikan 429 setelah 3 percobaan. Scanner akan melanjutkan pada jadwal candle berikutnya; tidak ada order yang dibatalkan.")
                        break
                self.stats["scan_cycles"] = int(self.stats.get("scan_cycles", 0)) + 1
                await self.persist()
                log.info("Scan completed: %d symbols | timeframe=%s | LONG=%d SHORT=%d errors=%d", cycle_counts["scanned"], tf, cycle_counts["long"], cycle_counts["short"], cycle_counts["errors"])
                await self.say(f"✅ <b>SCAN SELESAI</b>\n{tf} | {cycle_counts['scanned']}/{len(symbols)} koin\nLONG {cycle_counts['long']} | SHORT {cycle_counts['short']} | tanpa bubble {cycle_counts['none']} | error {cycle_counts['errors']}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self.report_error("scan-cycle", exc)
                await self.say(f"❌ Scan dihentikan karena error: <code>{type(exc).__name__}: {str(exc)[:500]}</code>")
                await self.persist()

    async def _on_signal(self, sig: dict[str, Any]):
        symbol, direction = sig["symbol"], sig["direction"]
        # A newer confirmed flip supersedes any unsent deferred entry for this pair.
        self.deferred_entries.pop(symbol, None)
        candle_open = int(sig["candle_open_time"])
        interval = TF_MS[sig["timeframe"]]
        next_open = candle_open + interval
        expiry = next_open + interval
        price = dec(sig["signal_price"])
        log.info("BUBBLE %s %s %s at %s", symbol, sig["bubble"], direction, iso(candle_open))
        key = "long_bubbles" if direction == "LONG" else "short_bubbles"
        self.stats[key] = int(self.stats.get(key, 0)) + 1
        if symbol in self.pending:
            try:
                await self._expire_pending(symbol, force_check=True)
            except BinanceRateLimit:
                # Do not decide reversal until the pending order's true fill state is known.
                self.deferred_signals[symbol] = sig
                await self.persist()
                return
        pos = self.positions.get(symbol)
        if pos:
            if pos.get("side") == direction:
                return
            # Safety exit remains enabled even after /real off. New reverse entry
            # is separately gated by real_mode in _release_reversal.
            self.deferred_reversals[symbol] = {"signal":sig,"next_open_time":next_open,
                "expiry_time":expiry,"limit_price":str(price),"close_attempt_at":0}
            await self._close_position(symbol, pos)
            return
        if utc_ms() >= expiry:
            log.info("Ignore expired historical entry signal %s %s", symbol, direction)
            return
        if not self.real_mode:
            await self.say(f"🔔 {symbol} {sig['bubble']} ({direction}) terkonfirmasi pada {sig['timeframe']}. Entry tidak dikirim karena REAL OFF.")
            return
        try:
            await self._create_entry(sig, next_open, expiry, price, reversal=False)
        except (BinanceRateLimit, requests.RequestException) as exc:
            self.deferred_entries[symbol] = {"signal":sig,"next_open_time":next_open,"expiry_time":expiry,"limit_price":str(price)}
            await self.persist()
            await self.say(f"⏳ {symbol}: entry {direction} masuk antrean ({exc}). Entry hanya akan dibuat jika candle entry belum expired; clientOrderId tetap untuk mencegah duplikasi.")
        except Exception as exc:
            await self.report_error(f"signal-entry/{symbol}", exc)
            await self.say(f"⚠️ {symbol}: sinyal {direction} tidak dapat dibuat menjadi order: {exc}")

    async def _create_entry(self, sig: dict[str, Any], next_open: int, expiry: int, price: Decimal, reversal: bool):
        symbol, direction = sig["symbol"], sig["direction"]
        if utc_ms() >= expiry:
            log.info("Skip expired deferred entry %s %s", symbol, direction)
            self.deferred_reversals.pop(symbol, None)
            self.deferred_entries.pop(symbol, None)
            if symbol not in self.positions and symbol not in self.pending: self.managed_symbols.discard(symbol)
            await self.persist(); return
        if symbol in self.pending: return
        occupied_symbols = set(self.managed_symbols) | set(self.pending) | set(self.positions) | set(self.deferred_entries) | set(self.deferred_reversals)
        occupied_symbols.discard(symbol)  # this entry/reversal already owns its slot
        if len(occupied_symbols) >= self.max_slots:
            self.deferred_reversals.pop(symbol, None)
            self.deferred_entries.pop(symbol, None)
            if symbol not in self.positions and symbol not in self.pending: self.managed_symbols.discard(symbol)
            await self.say(f"{symbol}: sinyal {direction} dilewati karena kapasitas /max {self.max_slots} penuh.")
            await self.persist()
            return
        meta = self.filters.get(symbol)
        if not meta:
            await self._refresh_universe(force=True); meta = self.filters.get(symbol)
        if not meta: raise RuntimeError(f"Filter symbol {symbol} tidak ditemukan")
        tick, step = meta["tick_size"], meta["step_size"]
        rounding = ROUND_DOWN if direction == "LONG" else ROUND_UP
        if price <= 0: raise ValueError(f"Harga limit {symbol} tidak valid")
        if tick > 0: price = (price / tick).to_integral_value(rounding=rounding) * tick
        if self.min_mode:
            # Minimum valid quantity is the greater of LOT_SIZE minQty and
            # MIN_NOTIONAL / entry price, rounded UP to the symbol step size.
            required_qty = max(meta["min_qty"], meta["min_notional"] / price)
            qty = required_qty
            if step > 0:
                qty = (qty / step).to_integral_value(rounding=ROUND_UP) * step
            actual_notional = qty * price
            if qty < meta["min_qty"] or actual_notional < meta["min_notional"]:
                raise RuntimeError(f"Tidak dapat memenuhi minimum quantity/notional Binance untuk {symbol}")
            if meta.get("max_qty", Decimal("0")) > 0 and qty > meta["max_qty"]:
                raise RuntimeError(f"Minimum order {symbol} melebihi maxQty filter Binance")
            effective_margin = actual_notional / Decimal(self.leverage)
        else:
            notional = self.margin * Decimal(self.leverage)
            qty = notional / price
            if step > 0: qty = (qty / step).to_integral_value(rounding=ROUND_DOWN) * step
            if qty < meta["min_qty"] or qty * price < meta["min_notional"]:
                raise RuntimeError(f"Quantity/notional {symbol} di bawah batas minimum Binance; naikkan /margin atau aktifkan /min on")
            if meta.get("max_qty", Decimal("0")) > 0 and qty > meta["max_qty"]:
                raise RuntimeError(f"Quantity {symbol} melebihi maxQty filter Binance; turunkan /margin")
            effective_margin = self.margin
        await self.binance.set_leverage(symbol, self.leverage)
        side = "BUY" if direction == "LONG" else "SELL"
        client_id = f"VT{sig['timeframe']}{symbol[:8]}{candle_open_id(sig['candle_open_time'])}{'L' if side=='BUY' else 'S'}"
        # Idempotency guard: after a timeout/cooldown, the original POST may have
        # reached Binance even if the client did not receive its response.
        order = None
        try:
            order = await self.binance.order(symbol, client_id=client_id)
        except Exception as lookup_error:
            if "-2013" not in str(lookup_error) and "does not exist" not in str(lookup_error).lower():
                raise
        if order is None:
            order = await self.binance.limit(symbol, side, qty, price, client_id)
        self.pending[symbol] = Pending(symbol, side, sig["timeframe"], int(sig["candle_open_time"]),
            next_open, expiry, str(price), str(qty), client_id, int(order.get("orderId") or 0),
            str(order.get("status") or "NEW"), reversal)
        self.managed_symbols.add(symbol)
        self.stats["orders_created"] = int(self.stats.get("orders_created", 0)) + 1
        self.deferred_reversals.pop(symbol, None)
        self.deferred_entries.pop(symbol, None)
        await self.persist()
        margin_label = "Margin minimum" if self.min_mode else "Margin set"
        await self.say(f"🟦 <b>LIMIT {direction} dibuat</b>\n{symbol} | {sig['timeframe']}\nHarga: {dstr(price)}\nQty: {dstr(qty)}\nNotional: {dstr(qty * price)} USDT\n{margin_label}: ~{dstr(effective_margin)} USDT | Leverage: {self.leverage}x\nExpired: {iso(expiry)}")

    async def _close_position(self, symbol: str, pos: dict[str, Any]):
        amt = dec(pos.get("positionAmt"))
        if amt == 0:
            self.positions.pop(symbol, None); self.close_requests.discard(symbol); return
        side = "SELL" if amt > 0 else "BUY"
        try:
            result = await self.binance.market_close(symbol, side, abs(amt))
            log.info("Market close sent %s %s qty=%s order=%s", symbol, side, amt, result.get("orderId"))
            # Verify actual position is zero before placing reverse entry.
            await asyncio.sleep(0.5)
            await self._reconcile_symbol(symbol)
            if symbol not in self.positions:
                self.close_requests.discard(symbol)
                entry = dec(pos.get("entryPrice")); exit_price = dec(result.get("avgPrice"))
                qty = abs(amt); direction = str(pos.get("side") or ("LONG" if amt > 0 else "SHORT"))
                if entry > 0 and exit_price > 0 and qty > 0:
                    pnl = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
                    trade = {"symbol": symbol, "side": direction, "entry": str(entry), "exit": str(exit_price),
                             "qty": str(qty), "pnl": str(pnl), "closed_at": utc_ms()}
                    self.closed_trades.append(trade); self.closed_trades = self.closed_trades[-500:]
                    self.stats["closed_trades"] = int(self.stats.get("closed_trades", 0)) + 1
                    self.stats["net_pnl"] = str(dec(self.stats.get("net_pnl")) + pnl)
                    if pnl > 0:
                        self.stats["wins"] = int(self.stats.get("wins", 0)) + 1
                        self.stats["gross_profit"] = str(dec(self.stats.get("gross_profit")) + pnl)
                    elif pnl < 0:
                        self.stats["losses"] = int(self.stats.get("losses", 0)) + 1
                        self.stats["gross_loss"] = str(dec(self.stats.get("gross_loss")) + abs(pnl))
                    realized_equity = dec(self.stats.get("realized_equity")) + pnl
                    equity_peak = max(dec(self.stats.get("equity_peak")), realized_equity, Decimal("0"))
                    drawdown = equity_peak - realized_equity
                    self.stats["realized_equity"] = str(realized_equity)
                    self.stats["equity_peak"] = str(equity_peak)
                    self.stats["max_drawdown"] = str(max(dec(self.stats.get("max_drawdown")), drawdown))
                    old_best = dec(self.stats.get("best_trade"))
                    old_worst = dec(self.stats.get("worst_trade"))
                    self.stats["best_trade"] = str(max(old_best, pnl))
                    self.stats["worst_trade"] = str(min(old_worst, pnl))
                    if self.stats_notify:
                        await self.say(f"📊 <b>TRADE CLOSED</b>\n{symbol} {direction}\nEntry {dstr(entry)} → Exit {dstr(exit_price)}\nGross PnL: <b>{dstr(pnl)} USDT</b> (belum termasuk fee/funding)")
                    await self.persist()
                await self._release_reversal(symbol)
            else:
                self.positions[symbol]["closing"] = True
                if symbol not in self.deferred_reversals: self.close_requests.add(symbol)
                await self.persist()
        except (BinanceRateLimit, requests.RequestException):
            log.warning("Close %s deferred by API/network issue", symbol)
            if symbol not in self.deferred_reversals: self.close_requests.add(symbol)
            await self.persist()
            raise

    async def _release_reversal(self, symbol: str):
        item = self.deferred_reversals.get(symbol)
        if not item: return
        if utc_ms() >= int(item["expiry_time"]):
            self.deferred_reversals.pop(symbol, None)
            if symbol not in self.positions and symbol not in self.pending: self.managed_symbols.discard(symbol)
            await self.persist(); return
        if not self.real_mode:
            # Position has been closed, but /real off forbids the new entry.
            self.deferred_reversals.pop(symbol, None)
            if symbol not in self.positions and symbol not in self.pending: self.managed_symbols.discard(symbol)
            await self.persist(); return
        sig = item["signal"]
        try:
            await self._create_entry(sig, int(item["next_open_time"]), int(item["expiry_time"]), dec(item["limit_price"]), reversal=True)
        except (BinanceRateLimit, requests.RequestException):
            raise
        except Exception as exc:
            self.deferred_reversals.pop(symbol, None)
            if symbol not in self.positions and symbol not in self.pending: self.managed_symbols.discard(symbol)
            await self.persist()
            await self.report_error(f"reversal-entry/{symbol}", exc)
            await self.say(f"⚠️ Reversal {symbol} tidak dapat dibuat: {exc}")

    async def _expire_pending(self, symbol: str, force_check: bool = False):
        p = self.pending.get(symbol)
        if not p or (not force_check and utc_ms() < p.expiry_time): return
        # Check status first: never cancel an order that has already filled.
        try:
            status = await self.binance.order(symbol, order_id=p.order_id) if p.order_id else await self.binance.order(symbol, client_id=p.client_order_id)
            st = str(status.get("status") or "").upper()
            if st == "FILLED":
                self.position_wait_until[symbol] = utc_ms() + 30_000
                self.pending.pop(symbol, None)
                self.stats["filled_orders"] = int(self.stats.get("filled_orders", 0)) + 1
                await self._reconcile_symbol(symbol)
                await self.persist(); return
            cancel_result = None
            if st in {"NEW", "PARTIALLY_FILLED", "PENDING_NEW"} and p.order_id:
                cancel_result = await self.binance.cancel(symbol, p.order_id)
                cancel_status = str(cancel_result.get("status") or "").upper()
                if cancel_status == "FILLED":
                    self.position_wait_until[symbol] = utc_ms() + 30_000
                    self.pending.pop(symbol, None)
                    self.stats["filled_orders"] = int(self.stats.get("filled_orders", 0)) + 1
                    await self._reconcile_symbol(symbol)
                    await self.persist()
                    return
                if cancel_status == "PARTIALLY_FILLED" or dec(cancel_result.get("executedQty")) > 0:
                    self.position_wait_until[symbol] = utc_ms() + 30_000
                    await self._reconcile_symbol(symbol)
            self.pending.pop(symbol, None)
            log.info("Pending order expired %s status=%s", symbol, st)
            await self.persist()
        except BinanceRateLimit:
            # Keep it tracked; retry after cooldown. Exchange-resident LIMIT stays untouched.
            raise
        except Exception as exc:
            await self.report_error(f"pending-expiry/{symbol}", exc)

    async def _reconcile_symbol(self, symbol: str):
        rows = await self.binance.positions()
        found = None
        for row in rows:
            if row.get("symbol") != symbol: continue
            amt = dec(row.get("positionAmt"))
            if amt != 0: found = row; break
        if found:
            amt = dec(found.get("positionAmt"))
            self.managed_symbols.add(symbol)
            self.position_wait_until.pop(symbol, None)
            self.positions[symbol] = {"side":"LONG" if amt > 0 else "SHORT", "positionAmt":str(amt),
                "entryPrice":str(found.get("entryPrice") or "0"), "unRealizedProfit":str(found.get("unRealizedProfit") or "0")}
        else:
            self.positions.pop(symbol, None)
            if self.position_wait_until.get(symbol, 0) <= utc_ms() and symbol not in self.pending and symbol not in self.deferred_reversals and symbol not in self.deferred_entries:
                self.managed_symbols.discard(symbol)
            self.close_requests.discard(symbol)
        await self.persist()

    async def _reconcile(self):
        if not (self.binance.key and self.binance.secret): return
        try:
            cfg = await self.binance.account_config()
            if str(cfg.get("dualSidePosition", "false")).lower() == "true":
                log.error("Binance account is in Hedge Mode; bot requires One-way Mode")
                await self.say("⚠️ Akun Binance masih Hedge Mode. Bot ini memerlukan One-way Mode; ubah mode posisi di Binance sebelum /real on.")
                return
            rows = await self.binance.positions()
            actual = {}
            for row in rows:
                amt = dec(row.get("positionAmt"))
                if amt and row.get("symbol") in self.managed_symbols:
                    actual[row["symbol"]] = {"side":"LONG" if amt > 0 else "SHORT", "positionAmt":str(amt),
                        "entryPrice":str(row.get("entryPrice") or "0"), "unRealizedProfit":str(row.get("unRealizedProfit") or "0")}
            self.positions = actual
            open_orders = await self.binance.open_orders()
            open_by_id = {str(o.get("clientOrderId")):o for o in open_orders}
            # Retain only persisted pending orders still present at Binance.
            for sym, p in list(self.pending.items()):
                if p.client_order_id not in open_by_id:
                    try:
                        od = await self.binance.order(sym, order_id=p.order_id) if p.order_id else await self.binance.order(sym, client_id=p.client_order_id)
                        if od.get("status") == "FILLED":
                            self.position_wait_until[sym] = utc_ms() + 30_000
                            self.pending.pop(sym, None)
                        elif utc_ms() >= p.expiry_time: self.pending.pop(sym, None)
                    except Exception: pass
            await self.persist()
        except Exception as exc:
            await self.report_error("startup-reconciliation", exc)

    async def _monitor_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.binance.key and self.binance.secret and (self.pending or self.positions or self.deferred_reversals or self.deferred_entries or self.deferred_signals or self.close_requests or self.managed_symbols):
                    await self._monitor_once()
                else:
                    await self._flush_command_queue()
                await asyncio.sleep(MONITOR_SECONDS)
            except asyncio.CancelledError: raise
            except BinanceRateLimit:
                await asyncio.sleep(max(5.0, self.binance.cooldown))
            except Exception as exc:
                await self.report_error("monitor-loop", exc)
                await asyncio.sleep(MONITOR_SECONDS)

    async def _monitor_once(self):
        open_orders = await self.binance.open_orders()
        open_ids = {str(x.get("clientOrderId")): x for x in open_orders}
        positions_rows = await self.binance.positions()
        actual = {}
        for row in positions_rows:
            amt = dec(row.get("positionAmt"))
            if amt and row.get("symbol") in self.managed_symbols:
                actual[row["symbol"]] = {"side":"LONG" if amt > 0 else "SHORT", "positionAmt":str(amt),
                    "entryPrice":str(row.get("entryPrice") or "0"), "unRealizedProfit":str(row.get("unRealizedProfit") or "0")}
        self.positions = actual
        for sym in list(self.managed_symbols):
            if sym in actual:
                self.position_wait_until.pop(sym, None)
            elif self.position_wait_until.get(sym, 0) <= utc_ms() and sym not in self.pending and sym not in self.deferred_reversals and sym not in self.deferred_entries:
                self.managed_symbols.discard(sym)
        for sym, p in list(self.pending.items()):
            if p.client_order_id not in open_ids:
                try:
                    od = await self.binance.order(sym, order_id=p.order_id) if p.order_id else await self.binance.order(sym, client_id=p.client_order_id)
                    status = str(od.get("status") or "").upper()
                    if status == "FILLED" or sym in actual:
                        if status == "FILLED": self.position_wait_until[sym] = utc_ms() + 30_000
                        self.pending.pop(sym, None)
                        self.stats["filled_orders"] = int(self.stats.get("filled_orders", 0)) + 1
                        await self.say(f"✅ <b>LIMIT FILLED</b>\n{sym} {p.side} | Avg: {od.get('avgPrice','-')} | Qty: {od.get('executedQty','-')}\nPosisi sekarang dipantau sampai bubble berlawanan.")
                    elif status in {"CANCELED", "EXPIRED", "REJECTED"}:
                        self.pending.pop(sym, None)
                        await self.say(f"⌛ {sym}: pending order {status}.")
                except Exception as exc: await self.report_error(f"order-reconcile/{sym}", exc)
            elif utc_ms() >= p.expiry_time:
                await self._expire_pending(sym)
        # Re-evaluate a bubble that arrived while a previous pending order's
        # fill/expiry state could not be checked due to REST cooldown.
        for sym, sig in list(self.deferred_signals.items()):
            if sym in self.pending:
                continue
            deadline = int(sig["candle_open_time"]) + 2 * TF_MS[sig["timeframe"]]
            self.deferred_signals.pop(sym, None)
            if utc_ms() < deadline:
                await self._on_signal(sig)
        # A closing request or deferred reverse is resumed after REST recovers.
        for sym in list(self.close_requests):
            pos = self.positions.get(sym)
            if not pos:
                self.close_requests.discard(sym)
            else:
                await self._close_position(sym, pos)
        for sym, pos in list(self.positions.items()):
            reversal = self.deferred_reversals.get(sym)
            if reversal:
                last_attempt = int(reversal.get("close_attempt_at") or 0)
                if utc_ms() - last_attempt >= 30_000:
                    reversal["close_attempt_at"] = utc_ms()
                    await self.persist()
                    await self._close_position(sym, pos)
        for sym in list(self.deferred_reversals):
            if sym not in self.positions:
                await self._release_reversal(sym)
        for sym, item in list(self.deferred_entries.items()):
            if sym in self.positions or sym in self.pending:
                self.deferred_entries.pop(sym, None); continue
            if not self.real_mode or utc_ms() >= int(item["expiry_time"]):
                self.deferred_entries.pop(sym, None); continue
            try:
                await self._create_entry(item["signal"], int(item["next_open_time"]), int(item["expiry_time"]), dec(item["limit_price"]), False)
            except (BinanceRateLimit, requests.RequestException):
                raise
            except Exception as exc:
                self.deferred_entries.pop(sym, None)
                await self.persist()
                await self.report_error(f"deferred-entry/{sym}", exc)
                await self.say(f"⚠️ Deferred entry {sym} dibuang: {exc}")
        await self._flush_command_queue()
        await self.persist()

    async def _flush_command_queue(self):
        if self.binance.cooldown > 0 or not self.command_queue:
            return
        command = self.command_queue.pop(0)
        fake_update = {"message": {"text": command, "from": {"id": ALLOWED_USER_ID}}}
        await self.handle_update(fake_update, self.context)

    async def _start_private_ws(self):
        if self.user_ws_task and not self.user_ws_task.done(): return
        self.listen_key = await self.binance.create_listen_key()
        self.user_ws_task = asyncio.create_task(self._private_ws_loop(), name="volumatic-user-ws")
        self.keepalive_task = asyncio.create_task(self._listen_key_keepalive(), name="volumatic-listen-key")

    async def _listen_key_keepalive(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(25 * 60)
            if self.listen_key:
                try: await self.binance.keepalive_listen_key()
                except asyncio.CancelledError: raise
                except Exception as exc: await self.report_error("listenKey-keepalive", exc)

    async def _private_ws_loop(self):
        while not self.stop_event.is_set() and self.listen_key:
            try:
                async with ws_connect(BINANCE_USER_WS_BASE + self.listen_key, ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                    log.info("Private user-data WebSocket connected")
                    async for raw in ws:
                        if self.stop_event.is_set(): break
                        event = json.loads(raw)
                        if event.get("e") == "listenKeyExpired":
                            log.warning("Binance listenKey expired; reconnecting")
                            raise RuntimeError("listenKey expired")
                        if event.get("e") == "ORDER_TRADE_UPDATE":
                            o = event.get("o") or {}; sym = str(o.get("s") or "")
                            p = self.pending.get(sym)
                            if p and str(o.get("c")) == p.client_order_id:
                                p.status = str(o.get("X") or p.status)
                                await self.persist()
                                if p.status == "FILLED":
                                    self.stats["filled_orders"] = int(self.stats.get("filled_orders", 0)) + 1
                                    self.position_wait_until[sym] = utc_ms() + 30_000
                                    await self.say(f"✅ {sym} LIMIT terisi (WebSocket). Konfirmasi posisi sedang dilakukan.")
                                    try: await self._reconcile_symbol(sym); self.pending.pop(sym, None)
                                    except Exception as exc: await self.report_error(f"ws-position-reconcile/{sym}", exc)
                                    await self.persist()
            except asyncio.CancelledError: raise
            except Exception as exc:
                await self.report_error("private-user-websocket", exc)
                await asyncio.sleep(5)
                try: self.listen_key = await self.binance.create_listen_key()
                except Exception as reconnect_exc:
                    await self.report_error("private-ws-reconnect", reconnect_exc)
                    await asyncio.sleep(20)

    async def handle_update(self, update: dict[str, Any], context: dict[str, Any]):
        message = update.get("message") or update.get("edited_message") or {}
        text = str(message.get("text") or "").strip()
        uid = int((message.get("from") or {}).get("id") or 0)
        if uid and ALLOWED_USER_ID and uid != ALLOWED_USER_ID: return
        if not text: return
        parts = text.split(); cmd = parts[0].split("@")[0].lower(); args = parts[1:]
        try:
            if cmd in {"/start", "/help"}: await self._help()
            elif cmd == "/timeframe": await self._timeframe(args)
            elif cmd == "/max": await self._max(args)
            elif cmd == "/real": await self._real(args)
            elif cmd == "/margin": await self._margin(args)
            elif cmd == "/min": await self._min(args)
            elif cmd == "/leverage": await self._leverage(args)
            elif cmd == "/scan": await self._scan_command(args)
            elif cmd == "/stats": await self._stats_command(args)
            elif cmd in {"/status", "/api"}: await self._status()
            elif cmd == "/trade": await self._trade()
            elif cmd == "/balance": await self._balance()
            elif cmd == "/close": await self._close_command(args)
            elif cmd == "/cancel": await self._cancel_command(args)
            else: await self.say("Command tidak dikenal. Gunakan /help.")
        except BinanceRateLimit as exc:
            await self.say(f"⚠️ {exc}. Command akan dicoba ulang setelah cooldown.")
            if not (cmd == "/real" and args and args[0].lower() == "off") and cmd not in {"/timeframe", "/max", "/help", "/start"} and text not in self.command_queue:
                if len(self.command_queue) < 100: self.command_queue.append(text)
        except Exception as exc:
            await self.report_error(f"telegram-command/{cmd}", exc)
            await self.say(f"❌ Command {cmd} gagal: <code>{type(exc).__name__}: {str(exc)[:500]}</code>")

    async def _scan_command(self, args: list[str]):
        if not args:
            if self.scan_lock.locked():
                await self.say("Scan sedang berjalan.")
            else:
                asyncio.create_task(self.scan_cycle(manual=True), name="volumatic-manual-scan")
                await self.say("🔎 Scan manual dimulai. Progress dan kesimpulan akan dikirim per batch.")
            return
        mode = args[0].lower()
        if mode not in {"on", "off"}:
            raise ValueError("Gunakan /scan, /scan on, atau /scan off")
        self.scan_enabled = mode == "on"
        if self.scan_enabled:
            self.scan_wakeup.set()
            await self.say("✅ Auto scanner ON. Scan terjadwal berjalan setiap candle H4/D1 yang confirmed.")
        else:
            self.scan_wakeup.set()
            await self.say("⏸️ Auto scanner OFF. Siklus yang sedang berjalan akan berhenti setelah batch saat ini; order dan posisi tetap dipantau.")
        await self.persist()

    async def _stats_report(self, automatic: bool = False):
        # Statistics derive from the retained closed-trade history (maximum 500).
        trades = sorted(self.closed_trades, key=lambda t: int(t.get("closed_at", 0)))
        pnls = [dec(t.get("pnl")) for t in trades]
        wins_pnl = [p for p in pnls if p > 0]
        losses_pnl = [p for p in pnls if p < 0]
        total = int(self.stats.get("closed_trades", len(trades)))
        wins = int(self.stats.get("wins", len(wins_pnl)))
        losses = int(self.stats.get("losses", len(losses_pnl)))
        decided = wins + losses
        winrate = wins / decided * 100 if decided else 0.0
        gross_profit = dec(self.stats.get("gross_profit"), sum(wins_pnl, Decimal("0")))
        gross_loss_abs = dec(self.stats.get("gross_loss"), abs(sum(losses_pnl, Decimal("0"))))
        net_pnl = dec(self.stats.get("net_pnl"), sum(pnls, Decimal("0")))
        avg_win = gross_profit / wins if wins else Decimal("0")
        avg_loss = -gross_loss_abs / losses if losses else Decimal("0")
        expectancy = net_pnl / total if total else Decimal("0")
        profit_factor = gross_profit / gross_loss_abs if gross_loss_abs else None
        best = dec(self.stats.get("best_trade"), max(pnls) if pnls else Decimal("0"))
        worst = dec(self.stats.get("worst_trade"), min(pnls) if pnls else Decimal("0"))
        max_dd = dec(self.stats.get("max_drawdown"))
        if not self.stats.get("max_drawdown"):
            equity = Decimal("0"); peak = Decimal("0")
            for pnl in pnls:
                equity += pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)
        floating = sum((dec(pos.get("unRealizedProfit")) for pos in self.positions.values()), Decimal("0"))
        combined = net_pnl + floating
        pf_text = f"{profit_factor:.3f}" if profit_factor is not None else ("∞" if gross_profit > 0 else "—")
        title = "STATISTIK OTOMATIS" if automatic else "STATISTIK CLOSED TRADES"
        lines = [
            f"<b>{title}</b>",
            f"Total closed trades: <b>{total}</b> | Win: {wins} | Loss: {losses}",
            f"Win rate: <b>{winrate:.2f}%</b> | Profit factor: <b>{pf_text}</b>",
            f"Net realized PnL: <b>{net_pnl:.6f} USDT</b>",
            f"Floating PnL posisi aktif: <b>{floating:.6f} USDT</b>",
            f"Total PnL (realized + floating): <b>{combined:.6f} USDT</b>",
            f"Average win: <b>{avg_win:.6f} USDT</b> | Average loss: <b>{avg_loss:.6f} USDT</b>",
            f"Expectancy/trade: {expectancy:.6f} USDT",
            f"Max drawdown: <b>{max_dd:.6f} USDT</b> (equity closed-trade)",
            f"Best trade: {best:.6f} USDT | Worst trade: {worst:.6f} USDT",
            f"Gross profit: {gross_profit:.6f} USDT | Gross loss: {-gross_loss_abs:.6f} USDT",
            f"Posisi aktif: {len(self.positions)} | Pending LIMIT: {len(self.pending)}",
            f"Order dibuat: {self.stats.get('orders_created', 0)} | Filled: {self.stats.get('filled_orders', 0)}",
            f"Scan cycles: {self.stats.get('scan_cycles', 0)} | Scan errors: {self.stats.get('scan_errors', 0)}",
            "Catatan: PnL estimasi gross dari entry dan average exit, belum dikurangi fee/funding. Max DD dihitung kumulatif sejak statistik bot mulai direkam; riwayat detail menampilkan maksimal 500 trade.",
        ]
        if trades:
            lines.append("\n<b>10 closed trades terakhir:</b>")
            for trade in trades[-10:][::-1]:
                pnl = dec(trade.get("pnl")); mark = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                lines.append(f"{mark} {trade.get('symbol')} {trade.get('side')} | {pnl:.6f} USDT | {iso(int(trade.get('closed_at', 0)))}")
        await self.say("\n".join(lines))

    async def _stats_command(self, args: list[str]):
        if not args:
            await self._stats_report()
            return
        mode = args[0].lower()
        if mode not in {"on", "off"}:
            raise ValueError("Gunakan /stats, /stats on, atau /stats off")
        self.stats_notify = mode == "on"
        await self.persist()
        await self.say("✅ Notifikasi statistik otomatis ON saat trade ditutup." if self.stats_notify else "⏸️ Notifikasi statistik otomatis OFF. /stats tetap bisa dipanggil kapan saja.")

    async def _help(self):
        await self.say("<b>VOLUMATIC TREND BOT</b>\n/timeframe H4|D1 — timeframe aktif\n/max 10 — batas posisi + pending\n/real on|off — eksekusi live\n/margin 0.5 — margin tetap USDT\n/min on|off — margin minimum Binance\n/leverage 10 — leverage\n/scan — scan manual\n/scan on|off — hidup/matikan auto scanner\n/stats — statistik closed trades (PnL, DD, winrate, average win/loss, dll.)\n/stats on|off — notifikasi statistik otomatis\n/status — status bot/API\n/trade — posisi dan pending\n/balance — saldo Futures\n/close SYMBOL|all — tutup posisi\n/cancel SYMBOL — batalkan pending LIMIT\n\nStrategi: blue=LONG, red=SHORT; LIMIT satu candle; tanpa SL/TP.")

    async def _timeframe(self, args: list[str]):
        if not args:
            await self.say(f"Timeframe aktif: <b>{self.timeframe}</b> (default H4).")
            return
        tf = args[0].upper()
        if tf not in TF_MS: raise ValueError("Timeframe hanya H4 atau D1")
        if self.scan_lock.locked():
            raise ValueError("Scan sedang berjalan. Ubah timeframe setelah siklus scan selesai.")
        if self.pending or self.positions or self.managed_symbols or self.deferred_reversals or self.deferred_entries or self.deferred_signals:
            raise ValueError("Tidak bisa mengganti timeframe selama masih ada posisi, pending order, atau reversal tertunda. Selesaikan semuanya terlebih dahulu agar aturan exit tidak berubah di tengah trade.")
        self.timeframe = tf
        self.last_processed = {}
        self.scan_baseline_ms = utc_ms()
        self.scan_wakeup.set()
        await self.persist()
        await self.say(f"✅ Timeframe diubah menjadi <b>{tf}</b>. Baseline candle akan disinkronkan pada scan berikutnya.")

    async def _max(self, args: list[str]):
        if not args: await self.say(f"Max slot saat ini: {self.max_slots} (posisi + pending order).") ; return
        n = int(args[0])
        if n < 1 or n > 500: raise ValueError("/max harus 1 sampai 500")
        occupied = len(set(self.managed_symbols) | set(self.pending) | set(self.positions) | set(self.deferred_entries) | set(self.deferred_reversals))
        if n < occupied: raise ValueError(f"Tidak bisa /max {n}; saat ini sudah ada {occupied} slot terpakai")
        self.max_slots = n; await self.persist(); await self.say(f"✅ Max slot diubah menjadi {n}.")

    async def _real(self, args: list[str]):
        if not args: await self.say(f"REAL mode: {'ON' if self.real_mode else 'OFF'}") ; return
        mode = args[0].lower()
        if mode == "off":
            self.real_mode = False
            self.deferred_entries.clear()
            await self.persist()
            await self.say("REAL OFF. Tidak ada entry baru dan entry tertunda yang belum dikirim dibuang. Order LIMIT yang sudah berada di Binance tidak otomatis dibatalkan; posisi aktif tetap dipantau dan exit bubble tetap dijalankan.")
            return
        if mode != "on": raise ValueError("Gunakan /real on atau /real off")
        if not (self.binance.key and self.binance.secret): raise ValueError("API key/secret Binance belum diset")
        await self.binance.sync_time()
        cfg = await self.binance.account_config()
        if str(cfg.get("dualSidePosition", "false")).lower() == "true":
            raise ValueError("Akun Binance masih Hedge Mode. Ubah ke One-way Mode terlebih dahulu.")
        await self.binance.balance()
        self.real_mode = True
        await self._start_private_ws()
        await self.say("<b>REAL ON</b> — eksekusi LIMIT LONG/SHORT aktif. Tidak ada SL/TP. Pastikan margin dan risiko sudah dipahami.")

    async def _margin(self, args: list[str]):
        if not args: await self.say(f"Margin per posisi: {self.margin} USDT") ; return
        x = dec(args[0], Decimal("-1"))
        if x <= 0: raise ValueError("Margin harus lebih besar dari 0")
        self.margin = x; await self.persist(); await self.say(f"Margin per posisi: {x} USDT")

    async def _min(self, args: list[str]):
        if not args:
            await self.say(f"Mode margin minimum: {'ON' if self.min_mode else 'OFF'}\nLeverage aktif: {self.leverage}x\n{'Margin dihitung otomatis per simbol dari minQty, stepSize, dan minNotional Binance.' if self.min_mode else f'Margin tetap saat ini: {self.margin} USDT per posisi.'}")
            return
        mode = args[0].lower()
        if mode not in {"on", "off"}:
            raise ValueError("Gunakan /min on atau /min off")
        self.min_mode = mode == "on"
        await self.persist()
        if self.min_mode:
            await self.say(f"<b>MIN ON</b> — margin dihitung otomatis seminimal mungkin untuk setiap simbol berdasarkan filter LOT_SIZE/MIN_NOTIONAL Binance, lalu dibagi leverage {self.leverage}x. /margin disimpan tetapi tidak digunakan selama MIN ON.")
        else:
            await self.say(f"<b>MIN OFF</b> — kembali memakai margin tetap {self.margin} USDT per posisi dengan leverage {self.leverage}x.")

    async def _leverage(self, args: list[str]):
        if not args: await self.say(f"Leverage: {self.leverage}x") ; return
        n = int(args[0])
        if n < 1 or n > 125: raise ValueError("Leverage harus 1–125")
        self.leverage = n; await self.persist(); await self.say(f"Leverage baru: {n}x. Akan diterapkan sebelum entry berikutnya.")

    async def _status(self):
        await self.say(f"<b>STATUS VOLUMATIC</b>\nTimeframe: {self.timeframe}\nMode: {'REAL ON' if self.real_mode else 'REAL OFF'}\nUniverse: {len(self.universe)}\nPositions: {len(self.positions)}\nPending: {len(self.pending)}\nDeferred reversal: {len(self.deferred_reversals)}\nDeferred entry: {len(self.deferred_entries)}\nMax slots: {self.max_slots}\nMargin mode: {'MIN' if self.min_mode else 'FIXED'} | Fixed margin: {self.margin} USDT | Leverage: {self.leverage}x\nAPI cooldown: {self.binance.cooldown:.1f}s\nREST weight: {self.binance.last_weight}")

    async def _trade(self):
        lines = [f"<b>TRADE | {self.timeframe}</b>"]
        if not self.positions and not self.pending and not self.deferred_reversals and not self.deferred_entries and not self.deferred_signals:
            lines.append("Belum ada posisi atau pending order.")
        for s,p in self.positions.items(): lines.append(f"🟢 {s} {p.get('side')} | qty {p.get('positionAmt')} | entry {p.get('entryPrice')} | uPnL {p.get('unRealizedProfit')}")
        for s,p in self.pending.items(): lines.append(f"🕒 {s} {p.side} | LIMIT {p.limit_price} | {p.status} | exp {iso(p.expiry_time)}")
        for s,p in self.deferred_reversals.items(): lines.append(f"🔄 {s} reversal tertunda | exp {iso(int(p['expiry_time']))}")
        for s,p in self.deferred_entries.items(): lines.append(f"⏳ {s} entry tertunda | exp {iso(int(p['expiry_time']))}")
        for s,p in self.deferred_signals.items(): lines.append(f"🧭 {s} signal menunggu rekonsiliasi pending order")
        await self.say("\n".join(lines))

    async def _balance(self):
        rows = await self.binance.balance()
        usdt = next((x for x in rows if x.get("asset") == "USDT"), {})
        await self.say(f"USDT Wallet: {usdt.get('balance','-')}\nAvailable: {usdt.get('availableBalance','-')}\nUnrealized: {usdt.get('crossUnPnl','-')}")

    async def _close_command(self, args: list[str]):
        if not args: raise ValueError("Gunakan /close SYMBOL atau /close all")
        targets = list(self.positions) if args[0].lower() == "all" else [safe_symbol(args[0])]
        for s in targets:
            p = self.positions.get(s)
            if p: await self._close_position(s, p)
        await self.say("Permintaan close sudah diproses. Periksa /trade untuk konfirmasi posisi.")

    async def _cancel_command(self, args: list[str]):
        if not args: raise ValueError("Gunakan /cancel SYMBOL")
        s = safe_symbol(args[0]); p = self.pending.get(s)
        if not p: raise ValueError(f"Tidak ada pending order untuk {s}")
        await self._expire_pending(s, force_check=True)
        await self.say(f"Pemeriksaan pembatalan pending {s} selesai.")


_ENGINE: VolumaticEngine | None = None

async def on_start(context: dict[str, Any]):
    global _ENGINE
    if _ENGINE: await _ENGINE.stop()
    _ENGINE = VolumaticEngine(context)
    await _ENGINE.start()
    return True

async def handle_update(update: dict[str, Any], context: dict[str, Any]):
    if _ENGINE: await _ENGINE.handle_update(update, context)

async def on_stop(context: dict[str, Any]):
    global _ENGINE
    if _ENGINE:
        await _ENGINE.stop(); _ENGINE = None


def candle_open_id(value: int) -> str:
    # Base36 keeps Binance newClientOrderId under its 36-character limit.
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = max(0, int(value)); out = "0"
    if n:
        parts = []
        while n:
            n, r = divmod(n, 36); parts.append(chars[r])
        out = "".join(reversed(parts))
    return out[-10:]

if __name__ == "__main__":
    raise SystemExit("Gunakan launcher bot yang memanggil on_start(), handle_update(), dan on_stop().")
