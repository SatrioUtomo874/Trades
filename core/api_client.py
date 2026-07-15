"""
API Client - Binance REST (PRIORITAS UTAMA), Bybit Fallback, WebSocket, CoinGecko
"""
import os, time, json, threading, logging, re
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
    global _last_request_time
    now = time.time()
    while _request_timestamps and now - _request_timestamps[0] > 60:
        _request_timestamps.popleft()
    used_weight = len(_request_timestamps) * 3
    if used_weight >= 2000:
        wait = 60 - (now - _request_timestamps[0]) + 1
        if wait > 0:
            log.warning(f"[throttle] Rate limit mendekati batas ({used_weight}/2400), sleep {wait:.1f}s")
            time.sleep(wait)
    elapsed = now - _last_request_time
    if elapsed < 0.05:
        time.sleep(0.05 - elapsed)
    _request_timestamps.append(now)
    _last_request_time = time.time()

# ==================== VALIDASI SIMBOL ====================
_VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]+USDT$')

def is_valid_symbol(symbol):
    return bool(_VALID_SYMBOL_PATTERN.match(symbol))

# ==================== BINANCE REST ====================
def fapi_get(path, params=None, retries=1):
    """Binance REST dengan deteksi 418/429 (langsung lempar error)."""
    for i in range(retries):
        try:
            _throttle(weight=5 if "klines" in path else 1)
            r = requests.get(f"{FAPI}{path}", params=params, timeout=10, verify=False)
            if r.status_code in (418, 429):
                log.warning(f"[binance] HTTP {r.status_code} — IP BAN/RATE LIMIT! Pindah fallback.")
                raise ConnectionError(f"Binance rate limit/ban (HTTP {r.status_code})")
            d = r.json()
            if isinstance(d, dict) and "code" in d:
                raise ValueError(f"Binance error {d['code']}: {d.get('msg')}")
            return d
        except ConnectionError:
            raise
        except Exception as e:
            log.warning(f"[binance] {i+1}/{retries}: {e}")
            time.sleep(2)
    raise ConnectionError(f"Binance gagal: {path}")

def _binance_klines(symbol, interval, limit):
    if not is_valid_symbol(symbol):
        return pd.DataFrame()
    try:
        raw = fapi_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        if not raw or len(raw) < 10:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume","cts","qvol","trades","tbv","tbq","ign"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.index = pd.to_datetime(df["ts"], unit="ms")
        return df[["open","high","low","close","volume"]].dropna()
    except ConnectionError:
        raise  # lempar ke caller untuk fallback
    except Exception as e:
        log.warning(f"[binance/klines] {symbol}: {e}")
        return pd.DataFrame()

def _binance_price(symbol):
    if not is_valid_symbol(symbol):
        return None
    try:
        d = fapi_get("/fapi/v1/ticker/price", {"symbol": symbol})
        return float(d["price"])
    except ConnectionError:
        raise
    except Exception as e:
        log.warning(f"[binance/price] {symbol}: {e}")
        return None

def _binance_top_coins(exclude_syms=()):
    try:
        tickers = fapi_get("/fapi/v1/ticker/24hr")
        usdt = []
        for t in tickers:
            sym = t["symbol"]
            if not is_valid_symbol(sym):
                continue
            if (sym.endswith("USDT") 
                and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
                and float(t["quoteVolume"]) > 5_000_000
                and abs(float(t.get("priceChangePercent", "0"))) < 15
                and sym not in exclude_syms):
                usdt.append(t)
        usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        return [t["symbol"] for t in usdt[:TOP_N_COINS]]
    except ConnectionError:
        raise  # lempar ke caller untuk fallback
    except Exception as e:
        log.warning(f"[binance/top] {e}")
        return []

# ==================== BYBIT REST (FALLBACK) ====================
INTERVAL_MAP = {"1m":"1","15m":"15","1h":"60","4h":"240","1d":"D"}

def _bybit_klines(symbol, interval, limit):
    if not is_valid_symbol(symbol):
        return pd.DataFrame()
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
    if not is_valid_symbol(symbol):
        return None
    try:
        r = requests.get(f"{BYBIT}/v5/market/tickers", params={"category":"linear","symbol":symbol}, timeout=10)
        d = r.json()
        if d.get("retCode") != 0:
            raise ValueError(f"Bybit ticker error: {d.get('retMsg')}")
        return float(d["result"]["list"][0]["lastPrice"])
    except Exception as e:
        log.warning(f"[bybit/price] {symbol}: {e}")
        return None

def _bybit_top_coins(exclude_syms=()):
    try:
        r = requests.get(f"{BYBIT}/v5/market/tickers", params={"category":"linear"}, timeout=10)
        d = r.json()
        items = d.get("result", {}).get("list", [])
        usdt = []
        for t in items:
            sym = t["symbol"]
            if not is_valid_symbol(sym):
                continue
            if (sym.endswith("USDT")
                and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
                and float(t.get("turnover24h", "0")) > 5_000_000
                and abs(float(t.get("price24hPcnt", "0"))) < 0.15
                and sym not in exclude_syms):
                usdt.append(t)
        usdt.sort(key=lambda x: float(x.get("turnover24h", "0")), reverse=True)
        return [t["symbol"] for t in usdt[:TOP_N_COINS]]
    except Exception as e:
        log.warning(f"[bybit/top] {e}")
        return []

# ==================== COINGECKO (DARURAT - HARGA SAJA) ====================
COINGECKO_ID_MAP = {
    "BTCUSDT":"bitcoin", "ETHUSDT":"ethereum", "BNBUSDT":"binancecoin",
    "SOLUSDT":"solana", "XRPUSDT":"ripple", "ADAUSDT":"cardano",
    "DOGEUSDT":"dogecoin", "AVAXUSDT":"avalanche-2", "LINKUSDT":"chainlink",
    "DOTUSDT":"polkadot", "LTCUSDT":"litecoin", "TRXUSDT":"tron",
    "ATOMUSDT":"cosmos", "NEARUSDT":"near", "APTUSDT":"aptos",
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

# ==================== WEBSOCKET (FALLBACK TERAKHIR) ====================
# ... (WebSocket class tetap sama seperti sebelumnya, tidak berubah)

# ==================== PUBLIK ====================
def get_top_coins(exclude_syms=()):
    """
    PRIORITAS: Binance REST -> Bybit REST -> (tidak ada WS untuk top coins)
    Jika semua gagal, return []
    """
    banned_set = get_banned_set()
    all_exclude = set(exclude_syms) | banned_set
    
    # 1. Coba Binance
    try:
        coins = _binance_top_coins(all_exclude)
        if coins:
            log.info(f"[top] Binance: {len(coins)} koin")
            return coins
    except ConnectionError as e:
        log.warning(f"[top/binance] {e} — pindah Bybit")
    except Exception as e:
        log.warning(f"[top/binance] {e} — pindah Bybit")
    
    # 2. Fallback ke Bybit
    try:
        coins = _bybit_top_coins(all_exclude)
        if coins:
            log.info(f"[top] Bybit fallback: {len(coins)} koin")
            return coins
    except Exception as e:
        log.warning(f"[top/bybit] {e}")
    
    # 3. Tidak ada fallback untuk top coins (WS tidak support)
    log.error("[top] SEMUA SUMBER GAGAL! Tidak ada koin.")
    return []

def get_price(symbol):
    """PRIORITAS: Binance -> Bybit -> WebSocket -> CoinGecko."""
    if not is_valid_symbol(symbol):
        return None
    
    # 1. Binance
    try:
        p = _binance_price(symbol)
        if p is not None:
            return p
    except ConnectionError:
        pass
    except Exception:
        pass
    
    # 2. Bybit
    try:
        p = _bybit_price(symbol)
        if p is not None:
            return p
    except Exception:
        pass
    
    # 3. WebSocket
    if ws_feed.is_fresh():
        p = ws_feed.get_price(symbol)
        if p is not None:
            return p
    
    # 4. CoinGecko (darurat)
    p = _coingecko_price(symbol)
    if p is not None:
        return p
    
    return None

def get_klines(symbol, interval, limit=250):
    """PRIORITAS: Binance -> Bybit -> WebSocket."""
    if not is_valid_symbol(symbol):
        return pd.DataFrame()
    
    ws_feed.ensure_symbol_interval(symbol, interval)
    
    # 1. Binance
    try:
        df = _binance_klines(symbol, interval, limit)
        if not df.empty:
            return df
    except ConnectionError:
        pass
    except Exception as e:
        log.warning(f"[klines/binance] {symbol}: {e}")
    
    # 2. Bybit
    try:
        df = _bybit_klines(symbol, interval, limit)
        if not df.empty:
            log.info(f"[klines/bybit fallback] {symbol} {interval} OK")
            return df
    except Exception as e:
        log.warning(f"[klines/bybit] {symbol}: {e}")
    
    # 3. WebSocket (fallback terakhir)
    if ws_feed.is_fresh():
        df = ws_feed.get_klines(symbol, interval, limit)
        if df is not None and not df.empty:
            return df
    
    return pd.DataFrame()
