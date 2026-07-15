"""
API Client - Binance REST (dengan throttle), Bybit Fallback, WebSocket, CoinGecko
Tier 1: Binance Futures REST (dengan throttle weight)
Tier 2: Bybit REST (prioritas untuk backfill berat)
Tier 3: Binance WebSocket (fallback terakhir)
Tier 4: CoinGecko (darurat harga saja)
"""
import os, time, json, threading, logging
from collections import deque
import pandas as pd
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger(__name__)

# ==================== KONSTANTA ====================
FAPI = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
BINANCE_WS_URL = "wss://fstream.binance.com/ws"
TOP_N_COINS = 50
MAX_PRICE = 80.0

# ==================== THROTTLE ====================
_request_timestamps = deque(maxlen=100)
_last_request_time = 0

def _throttle(weight=1):
    """
    Jaga agar total weight per menit tidak melewati 2400 (Binance limit).
    weight: estimasi weight request (klines=5, ticker=1, ping=1).
    """
    global _last_request_time
    now = time.time()
    # Bersihkan timestamp lebih dari 60 detik
    while _request_timestamps and now - _request_timestamps[0] > 60:
        _request_timestamps.popleft()
    
    # Estimasi weight yang sudah terpakai dalam 60 detik terakhir
    # (dikalikan weight rata-rata 3 sebagai estimasi konservatif)
    used_weight = len(_request_timestamps) * 3
    if used_weight >= 2000:  # sisakan 400 weight untuk safety
        wait = 60 - (now - _request_timestamps[0]) + 1
        if wait > 0:
            log.warning(f"[throttle] Rate limit mendekati batas ({used_weight}/2400), sleep {wait:.1f}s")
            time.sleep(wait)
    
    # Jeda minimal 0.05 detik antar request
    elapsed = now - _last_request_time
    if elapsed < 0.05:
        time.sleep(0.05 - elapsed)
    
    _request_timestamps.append(now)
    _last_request_time = time.time()

# ==================== BINANCE REST ====================
def fapi_get(path, params=None, retries=2):
    """Binance REST dengan deteksi 418/429 (langsung stop retry Binance)."""
    for i in range(retries):
        try:
            _throttle(weight=5 if "klines" in path else 1)
            r = requests.get(f"{FAPI}{path}", params=params, timeout=10, verify=False)
            if r.status_code in (418, 429):
                raise ConnectionError(f"Binance rate limit/ban (HTTP {r.status_code}) — stop retry Binance")
            d = r.json()
            if isinstance(d, dict) and "code" in d:
                raise ValueError(f"Binance error {d['code']}: {d.get('msg')}")
            return d
        except ConnectionError:
            raise  # langsung lempar ke caller untuk fallback
        except Exception as e:
            log.warning(f"[binance] {i+1}/{retries}: {e}")
            time.sleep(2)
    raise ConnectionError(f"Binance gagal: {path}")

def _binance_klines(symbol, interval, limit):
    raw = fapi_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not raw or len(raw) < 10:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume","cts","qvol","trades","tbv","tbq","ign"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open","high","low","close","volume"]].dropna()

def _binance_price(symbol):
    d = fapi_get("/fapi/v1/ticker/price", {"symbol": symbol})
    return float(d["price"])

def _binance_top_coins(exclude_syms=()):
    tickers = fapi_get("/fapi/v1/ticker/24hr")
    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t["quoteVolume"]) > 5_000_000
        and abs(float(t.get("priceChangePercent", "0"))) < 15
        and t["symbol"] not in exclude_syms
    ]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]

# ==================== BYBIT REST (backfill berat) ====================
INTERVAL_MAP = {"1m":"1","15m":"15","1h":"60","4h":"240","1d":"D"}

def _bybit_klines(symbol, interval, limit):
    iv = INTERVAL_MAP.get(interval, "15")
    try:
        r = requests.get(f"{BYBIT}/v5/market/kline",
                         params={"category":"linear","symbol":symbol,"interval":iv,"limit":limit},
                         timeout=15)
        d = r.json()
        if d.get("retCode") != 0:
            raise ValueError(f"Bybit error: {d.get('retMsg')}")
        rows = d["result"]["list"]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.index = pd.to_datetime(df["ts"].astype(float), unit="ms")
        df = df.sort_index()
        return df[["open","high","low","close","volume"]].dropna()
    except Exception as e:
        log.warning(f"[bybit] {symbol} {interval}: {e}")
        return pd.DataFrame()

def _bybit_price(symbol):
    r = requests.get(f"{BYBIT}/v5/market/tickers", params={"category":"linear","symbol":symbol}, timeout=10)
    d = r.json()
    if d.get("retCode") != 0:
        raise ValueError(f"Bybit ticker error: {d.get('retMsg')}")
    return float(d["result"]["list"][0]["lastPrice"])

def _bybit_top_coins(exclude_syms=()):
    r = requests.get(f"{BYBIT}/v5/market/tickers", params={"category":"linear"}, timeout=10)
    d = r.json()
    items = d.get("result", {}).get("list", [])
    usdt = [
        t for t in items
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t.get("turnover24h", "0")) > 5_000_000
        and abs(float(t.get("price24hPcnt", "0"))) < 0.15
        and t["symbol"] not in exclude_syms
    ]
    usdt.sort(key=lambda x: float(x.get("turnover24h", "0")), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]

# ==================== COINGECKO (darurat harga) ====================
COINGECKO_ID_MAP = {
    "BTCUSDT":"bitcoin", "ETHUSDT":"ethereum", "BNBUSDT":"binancecoin",
    "SOLUSDT":"solana", "XRPUSDT":"ripple", "ADAUSDT":"cardano",
    "DOGEUSDT":"dogecoin", "AVAXUSDT":"avalanche-2", "LINKUSDT":"chainlink",
    "DOTUSDT":"polkadot", "LTCUSDT":"litecoin", "TRXUSDT":"tron",
    "ATOMUSDT":"cosmos", "NEARUSDT":"near", "APTUSDT":"aptos",
    "ARBUSDT":"arbitrum", "OPUSDT":"optimism", "SUIUSDT":"sui",
    "TONUSDT":"the-open-network", "BCHUSDT":"bitcoin-cash",
}
def _coingecko_price(symbol):
    cid = COINGECKO_ID_MAP.get(symbol)
    if not cid: return None
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids": cid, "vs_currencies": "usd"}, timeout=8)
        p = r.json().get(cid, {}).get("usd")
        return float(p) if p else None
    except Exception:
        return None

# ==================== WEBSOCKET FEED ====================
try:
    import websocket
    _WS_LIB_OK = True
except ImportError:
    _WS_LIB_OK = False

class BinanceWSFeed:
    """
    Satu koneksi WS gabungan Binance Futures.
    BUKAN sumber utama — hanya fallback terakhir.
    Subscribe dinamis, cleanup stream idle >30 menit.
    """
    KLINE_INTERVALS = ("1m", "15m", "1h", "1d")
    MAX_CANDLES = {"1m": 300, "15m": 300, "1h": 300, "1d": 150}
    STALE_AFTER_SEC = 30
    STREAM_IDLE_SEC = 1800

    def __init__(self):
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._klines = {}      # {(sym,itv): deque}
        self._ticker = {}      # {sym: {"symbol","price","qvol","chg"}}
        self._last_used = {}   # {(sym,itv): timestamp}
        self._subscribed = set()
        self._ws = None
        self._last_msg = 0.0
        self._connected = False
        self._stop = False
        self._backoff = 1

    def start(self):
        if not _WS_LIB_OK:
            log.error("[ws] Modul websocket-client tidak terpasang. WS fallback nonaktif.")
            return
        threading.Thread(target=self._run_forever, daemon=True).start()

    def is_fresh(self):
        return self._connected and (time.time() - self._last_msg) < self.STALE_AFTER_SEC

    def get_price(self, symbol):
        with self._lock:
            d = self._ticker.get(symbol)
            return d["price"] if d else None

    def get_top_coins_raw(self):
        with self._lock:
            return list(self._ticker.values())

    def get_klines(self, symbol, interval, limit):
        with self._lock:
            buf = self._klines.get((symbol, interval))
            if not buf:
                return None
            rows = list(buf)[-limit:]
        if len(rows) < min(limit, 40):
            return None
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], unit="ms")
        return df[["o","h","l","c","v"]].rename(
            columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})

    def ensure_symbol_interval(self, symbol, interval):
        if not _WS_LIB_OK:
            return
        with self._lock:
            have = (symbol, interval) in self._klines
            self._last_used[(symbol, interval)] = time.time()
        if not have:
            self._backfill(symbol, interval)
        self._subscribe_kline(symbol, interval)

    def cleanup_stale_streams(self):
        now = time.time()
        with self._lock:
            stale = [k for k, ts in self._last_used.items() if now - ts > self.STREAM_IDLE_SEC]
        for sym, itv in stale:
            self._unsubscribe_kline(sym, itv)
            with self._lock:
                self._klines.pop((sym, itv), None)
                self._last_used.pop((sym, itv), None)
        if stale:
            log.info(f"[ws] cleanup {len(stale)} stream idle")

    # --- internal ---
    def _run_forever(self):
        while not self._stop:
            try:
                self._connect()
            except Exception as e:
                log.warning(f"[ws] koneksi error: {e}")
            self._connected = False
            if self._stop:
                break
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, 30)

    def _connect(self):
        self._ws = websocket.WebSocketApp(
            BINANCE_WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self._ws.run_forever(ping_interval=180, ping_timeout=10)

    def _on_open(self, ws):
        self._connected = True
        self._backoff = 1
        self._last_msg = time.time()
        log.info("[ws] Binance Futures WS terhubung")
        self._send_subscribe(["!ticker@arr"])
        with self._lock:
            keys = list(self._klines.keys())
        if keys:
            streams = [f"{sym.lower()}@kline_{itv}" for sym, itv in keys]
            self._send_subscribe(streams)

    def _on_message(self, ws, raw):
        self._last_msg = time.time()
        try:
            msg = json.loads(raw)
        except Exception:
            return
        if isinstance(msg, list):
            self._handle_ticker_array(msg)
        elif isinstance(msg, dict) and msg.get("e") == "24hrTicker":
            self._handle_ticker_array([msg])
        elif isinstance(msg, dict) and msg.get("e") == "kline":
            self._handle_kline(msg)

    def _handle_ticker_array(self, arr):
        with self._lock:
            for t in arr:
                try:
                    sym = t["s"]
                    self._ticker[sym] = {
                        "symbol": sym,
                        "price": float(t["c"]),
                        "qvol": float(t["q"]),
                        "chg": float(t["P"]),
                    }
                except Exception:
                    continue

    def _handle_kline(self, msg):
        k = msg["k"]
        sym = msg["s"]
        itv = k["i"]
        key = (sym, itv)
        row = {"t": k["t"], "o": float(k["o"]), "h": float(k["h"]),
               "l": float(k["l"]), "c": float(k["c"]), "v": float(k["v"])}
        with self._lock:
            buf = self._klines.get(key)
            if buf is None:
                return
            if buf and buf[-1]["t"] == row["t"]:
                buf[-1] = row
            else:
                buf.append(row)

    def _on_error(self, ws, err):
        log.warning(f"[ws] error: {err}")

    def _on_close(self, ws, code, msg):
        self._connected = False
        log.warning(f"[ws] tertutup (code={code})")

    def _send_subscribe(self, streams):
        if not streams or not self._ws:
            return
        try:
            with self._send_lock:
                self._ws.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": int(time.time()*1000) % 100000
                }))
            with self._lock:
                self._subscribed |= set(streams)
        except Exception as e:
            log.warning(f"[ws] gagal subscribe: {e}")

    def _subscribe_kline(self, symbol, interval):
        stream = f"{symbol.lower()}@kline_{interval}"
        with self._lock:
            already = stream in self._subscribed
        if not already:
            self._send_subscribe([stream])

    def _unsubscribe_kline(self, symbol, interval):
        stream = f"{symbol.lower()}@kline_{interval}"
        try:
            with self._send_lock:
                if self._ws:
                    self._ws.send(json.dumps({
                        "method": "UNSUBSCRIBE",
                        "params": [stream],
                        "id": int(time.time()*1000) % 100000
                    }))
            with self._lock:
                self._subscribed.discard(stream)
        except Exception:
            pass

    def _backfill(self, symbol, interval):
        limit = self.MAX_CANDLES.get(interval, 250)
        df = _bybit_klines(symbol, interval, limit)
        src = "bybit"
        if df.empty:
            try:
                df = _binance_klines(symbol, interval, limit)
                src = "binance"
            except Exception:
                pass
        if df.empty:
            log.warning(f"[ws-backfill] {symbol} {interval} GAGAL")
            return
        rows = deque(maxlen=limit)
        for ts, r in df.iterrows():
            rows.append({
                "t": int(ts.timestamp()*1000),
                "o": float(r.open), "h": float(r.high),
                "l": float(r.low), "c": float(r.close),
                "v": float(r.volume)
            })
        with self._lock:
            self._klines[(symbol, interval)] = rows
        log.info(f"[ws-backfill] {symbol} {interval} OK via {src} ({len(rows)} candle)")

ws_feed = BinanceWSFeed()

# ==================== BAN KOIN ====================
ban_lock = threading.Lock()
banned_coins = {}
scan_counter = 0
BAN_DURATION_SCANS = 15
BAN_DURATION_TRADE_CLOSED = 500

def ban_coin(sym, reason="", duration=None):
    d = duration if duration is not None else BAN_DURATION_SCANS
    with ban_lock:
        banned_coins[sym] = (scan_counter, d)
    log.info(f"[ban] {sym} diban {d} scan" + (f" ({reason})" if reason else ""))

def get_banned_coins():
    with ban_lock:
        return dict(banned_coins)

def get_scan_counter():
    global scan_counter
    with ban_lock:
        scan_counter += 1
        now = scan_counter
        to_unban = [s for s, (banned_at, dur) in banned_coins.items() if now - banned_at >= dur]
        for s in to_unban:
            del banned_coins[s]
            log.info(f"[unban] {s} kembali aktif")
        return now, set(banned_coins.keys())

# ==================== PUBLIK ====================
def get_price(symbol):
    """Tier1 Binance REST → Tier2 Bybit REST → Tier3 WS → Tier4 CoinGecko."""
    try:
        return _binance_price(symbol)
    except Exception:
        pass
    try:
        return _bybit_price(symbol)
    except Exception:
        pass
    if ws_feed.is_fresh():
        p = ws_feed.get_price(symbol)
        if p is not None:
            return p
    p = _coingecko_price(symbol)
    if p is not None:
        return p
    return None

def get_klines(symbol, interval, limit=250):
    """
    Tier1 Bybit REST (prioritas untuk backfill berat) → Tier2 Binance REST → Tier3 WS.
    Untuk /analyze backfill 3 bulan, Bybit jadi prioritas agar Binance tetap ringan.
    """
    ws_feed.ensure_symbol_interval(symbol, interval)
    df = _bybit_klines(symbol, interval, limit)
    if not df.empty:
        return df
    try:
        df = _binance_klines(symbol, interval, limit)
        if not df.empty:
            return df
    except Exception:
        pass
    if ws_feed.is_fresh():
        df = ws_feed.get_klines(symbol, interval, limit)
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()

def get_top_coins(exclude_syms=()):
    """
    Tier1 Binance REST → Tier2 Bybit REST → Tier3 WS.
    exclude_syms: set koin yang tidak boleh dipilih (banned + posisi aktif).
    """
    cur_ban = get_banned_coins()[1] if hasattr(get_banned_coins, '__call__') else set()
    # Gabungkan exclude_syms dengan banned
    all_exclude = set(exclude_syms) | cur_ban
    
    try:
        coins = _binance_top_coins(all_exclude)
        if coins:
            return coins
    except Exception:
        pass
    try:
        coins = _bybit_top_coins(all_exclude)
        if coins:
            return coins
    except Exception:
        pass
    if ws_feed.is_fresh():
        raw = ws_feed.get_top_coins_raw()
        usdt = [
            t for t in raw
            if t["symbol"].endswith("USDT")
            and 0.0001 < t["price"] < MAX_PRICE
            and t["qvol"] > 5_000_000
            and abs(t["chg"]) < 15
            and t["symbol"] not in all_exclude
        ]
        usdt.sort(key=lambda x: x["qvol"], reverse=True)
        if usdt:
            return [t["symbol"] for t in usdt[:TOP_N_COINS]]
    return []

def _price_cache_loop():
    """Watchdog: cek WS freshness, cleanup stale streams."""
    while True:
        try:
            ws_feed.cleanup_stale_streams()
        except Exception as e:
            log.error(f"[ws-watchdog] {e}")
        time.sleep(10)