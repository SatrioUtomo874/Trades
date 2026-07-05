#!/usr/bin/env python3
"""
SMC Signal Broadcaster — Forward Entry Strategy
Logika: Analisis H1+M15+D1 → sinyal searah → entry diskon OB/FVG/Fib → TP/SL struktural
Render.com | python main.py
"""

import os, time, logging, threading, re
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
MAX_PRICE       = 80.0
TOP_N_COINS     = 50
MIN_RR              = 2.0
MONITOR_SLEEP       = 10
MAX_POSITIONS       = 20
MONITOR_INTERVAL    = 15 * 60
KLINE_VERIFY_THROTTLE = 20  # detik — min jarak antar fetch klines verifikasi
                            # SL saat sweep berkepanjangan (cegah hammering API)
SWEEP_PULL_FACTOR   = 1     # 0 = entry tetap di OB/FVG/EQL/Fib asli, 1 = entry persis di level Liquidity Sweep
MIN_CONFIDENCE      = 50    # ambang confidence minimum sinyal — diatur via /confidence_min
TP_MAX_RR_MULT      = 1.8   # batas atas pencarian TP "lebih kuat": RR boleh naik sampai MIN_RR × ini
WIB = timezone(timedelta(hours=7))   # untuk format jam entry di /trade
# ── Fibonacci Extension TP (gated H4 confluence) ──
# Dipakai HANYA saat level struktural biasa sudah habis diperiksa DAN
# konteks H4 (trend besar) + RSI H4 (momentum belum jenuh) mendukung.
# Bukan cabang "penyelamat" RR gagal — ini kandidat TP tambahan yang
# dievaluasi berdampingan dengan level struktural lain di _select_best_tp.
FIB_EXT_1           = 0.272  # ekstensi 1.272 — butuh H4 trend + RSI band saja
FIB_EXT_2           = 0.618  # ekstensi 1.618 — butuh confluence penuh (+ CHoCH M15 searah)
H4_RSI_BUY_MIN      = 45     # RSI H4 BUY: momentum sudah established (bukan baru mulai)
H4_RSI_BUY_MAX      = 68     # tapi belum overbought / jenuh
H4_RSI_SELL_MIN     = 32     # RSI H4 SELL: kebalikan dari BUY
H4_RSI_SELL_MAX     = 55
# ─────────────────────────────────────────────

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN tidak ditemukan di environment. Cek file .env")

import requests, pandas as pd, numpy as np, urllib3
from flask import Flask

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


class TelegramLogHandler(logging.Handler):
    """
    Forward log ERROR/CRITICAL ke Telegram.
    Throttle: maks 1 pesan per 30 detik per pesan unik
    agar tidak flood saat error berulang.
    """
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self._last_sent: dict = {}   # {msg_key: timestamp}
        self._throttle  = 30         # detik

    def emit(self, record):
        # Hindari rekursi (error saat kirim TG itu sendiri)
        if "TG" in record.getMessage(): return
        try:
            msg_key = record.getMessage()[:80]
            now = time.time()
            if now - self._last_sent.get(msg_key, 0) < self._throttle:
                return
            self._last_sent[msg_key] = now

            cid = active_chat_id
            if not cid or not TELEGRAM_TOKEN: return

            level_em = "🔴" if record.levelno >= logging.CRITICAL else "⚠️"
            text = (
                f"{level_em} <b>[{record.levelname}]</b>\n"
                f"<code>{record.getMessage()[:400]}</code>"
            )
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "HTML"},
                timeout=5
            )
        except Exception:
            pass   # jangan pernah raise dari handler log


_tg_log_handler = TelegramLogHandler()
log.addHandler(_tg_log_handler)

auto_mode      = False
auto_thread    = None
active_chat_id = None
timeout_flag   = False
active_trade   = None   # dict posisi yang sedang dipantau, None jika tidak ada

STARTING_BALANCE = 10.0   # modal awal simulasi dalam USD

stat_lock = threading.Lock()
stats = {
    "tp":0, "sl":0, "total":0,
    "balance"    : STARTING_BALANCE,
    "pnl_history": [],
}

# Ban koin berbasis SCAN CYCLE (bukan jumlah trade nyata — koin yang selalu
# ke-skip di tahap pending tidak pernah menambah hitungan trade, jadi ban
# berbasis trade tidak akan pernah relevan untuk kasus itu).
ban_lock = threading.Lock()
banned_coins: dict = {}      # {symbol: scan_counter saat diban}
scan_counter = 0             # bertambah 1 setiap get_top_coins() dipanggil
BAN_DURATION_SCANS = 8

def _ban_coin(sym, reason=""):
    """Ban koin selama BAN_DURATION_SCANS siklus scan berikutnya."""
    with ban_lock:
        banned_coins[sym] = scan_counter
    log.info(f"[ban] {sym} diban {BAN_DURATION_SCANS} scan" + (f" ({reason})" if reason else ""))

FAPI = "https://fapi.binance.com"

# ── Flask ─────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    with stat_lock:
        t=stats["total"]; tp=stats["tp"]; sl=stats["sl"]
    with ban_lock:
        n_banned = len(banned_coins)
    wr=f"{tp/(tp+sl)*100:.1f}%" if (tp+sl)>0 else "–"
    return (f"<h3>SMC Signal Broadcaster</h3>"
            f"<p>Auto:{auto_mode} | Banned:{n_banned}</p>"
            f"<p>Total:{t} TP:{tp} SL:{sl} WR:{wr}</p>"), 200

@app.route("/health")
def health(): return "OK", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═════════════════════════════════════════════
# TELEGRAM
# ═════════════════════════════════════════════
def tg_send(chat_id, text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id":chat_id,"text":text,"parse_mode":"HTML"},
            timeout=10)
    except Exception as e:
        log.error(f"[TG] {e}")

def tg_updates(offset=None):
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"timeout":8,"offset":offset}, timeout=12)
        d = r.json()
        return d.get("result",[]) if d.get("ok") else []
    except:
        return []


# ═════════════════════════════════════════════
# DATA LAYER — Binance utama, Bybit fallback
# ═════════════════════════════════════════════
BYBIT = "https://api.bybit.com"

# Konversi interval Binance → Bybit
INTERVAL_MAP = {
    "1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
    "1h":"60","2h":"120","4h":"240","1d":"D","1w":"W",
}

def _raw_get(url, params=None, retries=3):
    """HTTP GET dengan retry — digunakan oleh kedua exchange."""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10, verify=False)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"[http] {i+1}/{retries} {url}: {e}")
            time.sleep(2)
    raise ConnectionError(f"GET gagal: {url}")


# ── BINANCE ───────────────────────────────────
# Circuit-breaker global untuk ban IP (-1003). Binance memperpanjang durasi
# ban setiap kali endpoint tetap dihantam SELAGI masih diban — jadi begitu
# satu thread mendeteksi ban, SEMUA thread (price cache, scan, monitor,
# startup ping) harus langsung tahu dan berhenti request ke Binance sampai
# waktu ban lewat, bukan retry buta yang justru memperpanjang ban itu sendiri.
_binance_ban_lock = threading.Lock()
_binance_banned_until = 0.0   # epoch detik; 0 = tidak sedang diban

def _binance_ban_remaining():
    """Detik tersisa sebelum ban Binance berakhir (0 kalau tidak diban)."""
    with _binance_ban_lock:
        return max(0.0, _binance_banned_until - time.time())

def _register_binance_ban(msg):
    """Parse 'banned until <epoch_ms>' dari pesan error -1003 dan simpan
    sebagai state global — tidak pernah memperpendek ban yang sudah ada."""
    global _binance_banned_until
    m = re.search(r"banned until (\d+)", msg or "")
    if not m: return
    until_s = int(m.group(1)) / 1000.0 + 1.0   # +1 detik buffer aman
    with _binance_ban_lock:
        if until_s > _binance_banned_until:
            _binance_banned_until = until_s
            log.warning(f"[binance] IP diban, semua request Binance "
                        f"dihentikan selama {until_s - time.time():.0f}s")

def fapi_get(path, params=None):
    """Binance Futures GET — tetap dipakai utama."""
    remaining = _binance_ban_remaining()
    if remaining > 0:
        # Circuit breaker aktif — jangan hantam API sama sekali selagi
        # ban masih berlaku. Biarkan caller fallback ke Bybit kalau ada.
        raise ConnectionError(f"Binance masih diban {remaining:.0f}s lagi, skip.")

    for i in range(3):
        try:
            r = requests.get(f"{FAPI}{path}", params=params,
                             timeout=10, verify=False)
            d = r.json()
            if isinstance(d, dict) and "code" in d:
                msg = f"Binance {d['code']}: {d.get('msg')}"
                if d["code"] == -1003:
                    _register_binance_ban(d.get("msg", ""))
                    raise ConnectionError(msg)   # jangan retry — endpoint lagi diban
                raise ValueError(msg)
            return d
        except ConnectionError:
            raise   # ban terdeteksi, keluar langsung tanpa retry lagi
        except Exception as e:
            log.warning(f"[binance] {i+1}/3: {e}")
            time.sleep(2)
    raise ConnectionError(f"Binance gagal: {path}")

def _binance_klines(symbol, interval, limit):
    raw = fapi_get("/fapi/v1/klines",
                   {"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(raw, list) or len(raw) < min(limit, 40):
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=[
        "ts","open","high","low","close","volume",
        "cts","qvol","trades","tbv","tbq","ign"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open","high","low","close","volume"]].dropna()

def _binance_price(symbol):
    d = fapi_get("/fapi/v1/ticker/price", {"symbol": symbol})
    return float(d["price"])

def _binance_top_coins(exclude_syms):
    tickers = fapi_get("/fapi/v1/ticker/24hr")
    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t["quoteVolume"]) > 5_000_000          # min 5jt USDT/24j
        and abs(float(t.get("priceChangePercent","0"))) < 15  # skip pump/dump ekstrem
        and t["symbol"] not in exclude_syms
    ]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]


# ── BYBIT FALLBACK ────────────────────────────
def _bybit_klines(symbol, interval, limit):
    """Bybit Linear (USDT Perpetual) klines."""
    iv = INTERVAL_MAP.get(interval, "15")
    d = _raw_get(f"{BYBIT}/v5/market/kline", {
        "category":"linear","symbol":symbol,
        "interval":iv,"limit":limit
    })
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit kline error: {d.get('retMsg')}")
    rows = d["result"]["list"]
    if not rows or len(rows) < min(limit, 40):
        return pd.DataFrame()
    # Bybit returns: [startTime, open, high, low, close, volume, turnover]
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["ts"].astype(float), unit="ms")
    df = df.sort_index()   # Bybit returns newest-first
    return df[["open","high","low","close","volume"]].dropna()

def _bybit_price(symbol):
    d = _raw_get(f"{BYBIT}/v5/market/tickers",
                 {"category":"linear","symbol":symbol})
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit ticker error: {d.get('retMsg')}")
    return float(d["result"]["list"][0]["lastPrice"])

def _bybit_top_coins(exclude_syms):
    d = _raw_get(f"{BYBIT}/v5/market/tickers", {"category":"linear"})
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit tickers error: {d.get('retMsg')}")
    items = d["result"]["list"]
    usdt = [
        t for t in items
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t.get("turnover24h","0")) > 5_000_000   # min 5jt USDT/24j
        and abs(float(t.get("price24hPcnt","0"))) < 0.15  # skip pump/dump ekstrem
        and t["symbol"] not in exclude_syms
    ]
    usdt.sort(key=lambda x: float(x.get("turnover24h","0")), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]


# ── PRICE CACHE — satu thread terpusat, update tiap 10 detik ──
# Menghindari IP ban akibat banyak request price per-posisi per-detik
_price_cache: dict = {}          # {symbol: float}
_price_cache_lock = threading.Lock()
_PRICE_REFRESH_SEC = 10          # interval refresh cache (detik)

def _price_cache_loop():
    """
    Thread tunggal yang refresh harga semua koin aktif + posisi terbuka
    setiap _PRICE_REFRESH_SEC detik via satu batch request.
    Jauh lebih hemat rate limit daripada per-posisi per-tick.
    """
    while True:
        try:
            # Kumpulkan semua simbol yang perlu dipantau
            with positions_lock:
                syms = set(positions.keys())

            if syms:
                # Binance batch: ambil semua sekaligus dalam satu request
                try:
                    data = fapi_get("/fapi/v1/ticker/price")
                    if isinstance(data, list):
                        batch = {d["symbol"]: float(d["price"]) for d in data}
                        with _price_cache_lock:
                            for s in syms:
                                if s in batch:
                                    _price_cache[s] = batch[s]
                    else:
                        raise ValueError("Format tidak dikenal")
                except Exception as e:
                    log.warning(f"[price_cache/binance] batch gagal: {e} — coba Bybit")
                    # Bybit fallback: per-simbol (jarang terjadi)
                    for s in syms:
                        try:
                            p = _bybit_price(s)
                            with _price_cache_lock:
                                _price_cache[s] = p
                        except Exception:
                            pass
        except Exception as e:
            log.error(f"[price_cache_loop] {e}")

        time.sleep(_PRICE_REFRESH_SEC)

def get_price(symbol):
    """
    Ambil harga dari cache. Kalau belum ada (posisi baru), fetch sekali langsung.
    Setelah itu cache_loop yang handle update berkala.
    """
    with _price_cache_lock:
        cached = _price_cache.get(symbol)
    if cached is not None:
        return cached

    # Fetch langsung untuk posisi yang baru masuk cache
    for _ in range(2):
        try:
            p = _binance_price(symbol)
            with _price_cache_lock:
                _price_cache[symbol] = p
            return p
        except Exception as e:
            log.warning(f"[price/binance] {symbol}: {e}")
            time.sleep(1)
    for _ in range(2):
        try:
            p = _bybit_price(symbol)
            with _price_cache_lock:
                _price_cache[symbol] = p
            return p
        except Exception as e:
            log.warning(f"[price/bybit] {symbol}: {e}")
            time.sleep(1)
    return None

def get_klines(symbol, interval, limit=250):
    """Ambil klines. Binance dulu, fallback Bybit."""
    # Binance
    try:
        df = _binance_klines(symbol, interval, limit)
        if not df.empty:
            return df
        log.warning(f"[klines/binance] {symbol} kosong, coba Bybit...")
    except Exception as e:
        log.warning(f"[klines/binance] {symbol}: {e} — coba Bybit...")
    # Bybit fallback
    try:
        df = _bybit_klines(symbol, interval, limit)
        if not df.empty:
            log.info(f"[klines/bybit fallback] {symbol} {interval} OK")
            return df
    except Exception as e:
        log.warning(f"[klines/bybit] {symbol}: {e}")
    return pd.DataFrame()

def get_top_coins():
    """Ambil top coins. Binance dulu, fallback Bybit.

    Koin yang sedang pending/aktif di posisi, ATAU sedang diban (lihat
    _ban_coin), DIKELUARKAN dari pool SEBELUM slicing ke TOP_N_COINS —
    bukan disaring setelahnya. Efeknya: pool tetap genap TOP_N_COINS
    koin tradeable, koin peringkat ke-51, 52, dst otomatis naik
    menggantikan slot yang "diambil" oleh posisi/ban yang sedang berjalan.
    """
    global scan_counter
    with ban_lock:
        scan_counter += 1
        to_unban = [s for s, banned_at in banned_coins.items()
                    if scan_counter - banned_at >= BAN_DURATION_SCANS]
        for s in to_unban:
            del banned_coins[s]
            log.info(f"[unban] {s} kembali aktif setelah {BAN_DURATION_SCANS} scan")
        cur_ban = set(banned_coins.keys())

    with positions_lock:
        active_syms = set(positions.keys())   # pending + aktif

    exclude_syms = cur_ban | active_syms

    # Binance
    try:
        coins = _binance_top_coins(exclude_syms)
        if coins:
            return coins
        log.warning("[top_coins/binance] kosong, coba Bybit...")
    except Exception as e:
        log.warning(f"[top_coins/binance] {e} — coba Bybit...")
    # Bybit fallback
    try:
        coins = _bybit_top_coins(exclude_syms)
        log.info(f"[top_coins/bybit fallback] {len(coins)} koin")
        return coins
    except Exception as e:
        log.warning(f"[top_coins/bybit] {e}")
    return []


# ═════════════════════════════════════════════
# INDIKATOR
# ═════════════════════════════════════════════
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d=s.diff()
    g=d.clip(lower=0).rolling(n).mean()
    l=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def macd(s):
    line=ema(s,12)-ema(s,26); sig=ema(line,9)
    return line, sig, line-sig

def atr_fn(df, n=14):
    tr=pd.concat([
        df["high"]-df["low"],
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build_df(df):
    if len(df)<60: return None
    df=df.copy()
    df["ema9"]   = ema(df["close"],9)
    df["ema21"]  = ema(df["close"],21)
    df["ema50"]  = ema(df["close"],50)
    df["ema200"] = ema(df["close"],200) if len(df)>=200 else ema(df["close"],50)
    df["rsi"]    = rsi(df["close"])
    df["ml"],df["ms"],df["mh"] = macd(df["close"])
    df["atr"]    = atr_fn(df)
    df["vol_sma"]= df["volume"].rolling(20).mean()
    bm=df["close"].rolling(20).mean(); bs=df["close"].rolling(20).std()
    df["bb_up"]=bm+2*bs; df["bb_lo"]=bm-2*bs; df["bb_mid"]=bm
    return df.dropna()


# ═════════════════════════════════════════════
# SMC / PRICE ACTION TOOLS
# ═════════════════════════════════════════════
def swing_pts(df, lb=5):
    sh,sl=[],[]
    for i in range(lb, len(df)-lb):
        if df["high"].iloc[i]==df["high"].iloc[i-lb:i+lb+1].max(): sh.append(i)
        if df["low"].iloc[i]==df["low"].iloc[i-lb:i+lb+1].min():   sl.append(i)
    return sh, sl

def mkt_struct(df, sh, sl):
    if len(sh)<2 or len(sl)<2: return "ranging"
    hh=df["high"].iloc[sh[-1]]>df["high"].iloc[sh[-2]]
    hl=df["low"].iloc[sl[-1]]>df["low"].iloc[sl[-2]]
    lh=df["high"].iloc[sh[-1]]<df["high"].iloc[sh[-2]]
    ll=df["low"].iloc[sl[-1]]<df["low"].iloc[sl[-2]]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

def detect_bos(df, sh, sl):
    res={"bb":False,"bs":False,"cb":False,"cs":False}
    p=df["close"].iloc[-1]
    if len(sh)>=2:
        ph=df["high"].iloc[sh[-2]]; lh=df["high"].iloc[sh[-1]]
        if p>ph: res["bb" if lh>ph else "cb"]=True
    if len(sl)>=2:
        pl=df["low"].iloc[sl[-2]]; ll=df["low"].iloc[sl[-1]]
        if p<pl: res["bs" if ll<pl else "cs"]=True
    return res

def find_snr_levels(df, lb=80):
    """
    Cari level Support & Resistance dari swing points.
    Level yang paling banyak disentuh = level terkuat.
    """
    sh, sl = swing_pts(df, lb=5)
    levels = []
    for i in sh:
        levels.append(("R", df["high"].iloc[i]))
    for i in sl:
        levels.append(("S", df["low"].iloc[i]))
    return levels

def find_supply_demand(df, direction, lb=40):
    """
    Supply zone  = area di mana harga turun tajam (bearish OB)
    Demand zone  = area di mana harga naik tajam (bullish OB)
    Ini adalah zona institusi menempatkan order besar.
    """
    sub=df.iloc[-lb:]
    avg_body=(sub["close"]-sub["open"]).abs().mean()
    zones=[]
    for i in range(1, len(sub)-2):
        c=sub.iloc[i]; nx=sub.iloc[i+1]
        body=abs(nx["close"]-nx["open"])
        if body < avg_body*1.3: continue
        if direction=="supply":
            # Supply: candle bullish besar diikuti drop tajam
            if c["close"]>c["open"] and nx["close"]<nx["open"]:
                zones.append({
                    "top": max(c["open"],c["close"]),
                    "bot": min(c["open"],c["close"]),
                    "high": c["high"], "low": c["low"]
                })
        else:
            # Demand: candle bearish besar diikuti rally
            if c["close"]<c["open"] and nx["close"]>nx["open"]:
                zones.append({
                    "top": max(c["open"],c["close"]),
                    "bot": min(c["open"],c["close"]),
                    "high": c["high"], "low": c["low"]
                })
    return zones[-3:] if zones else []

def find_fvg(df, direction, lb=40):
    sub=df.iloc[-lb:]; out=[]
    for i in range(len(sub)-2):
        c0,c2=sub.iloc[i],sub.iloc[i+2]
        if direction=="bull" and c2["low"]>c0["high"]:
            out.append({"top":c2["low"],"bot":c0["high"],"mid":(c2["low"]+c0["high"])/2})
        if direction=="bear" and c2["high"]<c0["low"]:
            out.append({"top":c0["low"],"bot":c2["high"],"mid":(c0["low"]+c2["high"])/2})
    return out[-3:] if out else []

def find_equal_highs_lows(df, kind="high", lb=60, tol=0.0025):
    """
    Equal Highs/Lows = zona likuiditas (banyak stop loss retail di sana).
    Institusi sering sweeping level ini sebelum berbalik.
    """
    sub=df.iloc[-lb:]
    vals=sub["high"] if kind=="high" else sub["low"]
    clusters=[]
    visited=set()
    for i in range(len(vals)):
        if i in visited: continue
        group=[vals.iloc[i]]
        for j in range(i+1, len(vals)):
            if abs(vals.iloc[i]-vals.iloc[j])/max(vals.iloc[i],0.0001)<tol:
                group.append(vals.iloc[j])
                visited.add(j)
        if len(group)>=2:
            clusters.append(sum(group)/len(group))
    return sorted(clusters)

def nearest_snr(df, price, direction, margin=0.015):
    """
    Cari level S/R terdekat yang relevan untuk TP/SL.
    direction='above' → cari resistance di atas harga
    direction='below' → cari support di bawah harga
    """
    sh, sl = swing_pts(df, lb=4)
    if direction=="above":
        candidates = [df["high"].iloc[i] for i in sh
                      if df["high"].iloc[i] > price*(1+margin*0.3)]
        candidates += find_equal_highs_lows(df,"high")
        candidates = [c for c in candidates if c > price*(1+margin*0.3)]
        return min(candidates) if candidates else None
    else:
        candidates = [df["low"].iloc[i] for i in sl
                      if df["low"].iloc[i] < price*(1-margin*0.3)]
        candidates += find_equal_highs_lows(df,"low")
        candidates = [c for c in candidates if c < price*(1-margin*0.3)]
        return max(candidates) if candidates else None


def detect_choch(df, sh, sl):
    """
    CHoCH (Change of Character) — konfirmasi perubahan arah NYATA.
    Bearish CHoCH: harga break di bawah HL terakhir setelah LH terbentuk.
    Bullish CHoCH: harga break di atas LH terakhir setelah HL terbentuk.
    Lebih ketat dari BOS biasa — perlu dua swing point terkonfirmasi.
    """
    result = {"bearish_choch": False, "bullish_choch": False}
    close = df["close"].iloc[-1]

    # Bearish CHoCH: ada LH (lower high) DAN harga sekarang break bawah swing low sebelumnya
    if len(sh) >= 2 and len(sl) >= 2:
        prev_high = df["high"].iloc[sh[-2]]
        last_high = df["high"].iloc[sh[-1]]
        prev_low  = df["low"].iloc[sl[-2]]
        last_low  = df["low"].iloc[sl[-1]]

        lh_formed = last_high < prev_high          # LH terbentuk
        if lh_formed and close < prev_low:         # break bawah HL
            result["bearish_choch"] = True

        hh_formed = last_high > prev_high          # HH terbentuk
        if hh_formed and close > prev_low and last_low > prev_low:  # break atas + HL
            result["bullish_choch"] = True

    return result


def detect_failed_retest(df, sh, sl, atr):
    """
    Failed Retest — harga naik ke resistance/level struktural lalu ditolak keras.
    Ini trigger entry SELL yang paling valid di SMC.
    Syarat:
    - Ada resistance level yang jelas (swing high sebelumnya)
    - Harga candle sebelumnya menyentuh atau mendekati resistance (dalam 0.5 ATR)
    - Candle sekarang close jauh di bawah resistance (rejection)
    - Candle sekarang bearish (close < open)
    """
    result = {"failed_retest_sell": False, "failed_retest_buy": False,
              "resistance": None, "support": None}
    if len(df) < 3: return result

    L   = df.iloc[-1]   # candle sekarang
    P   = df.iloc[-2]   # candle sebelumnya

    # Failed retest SELL: candle sebelumnya menyentuh resistance, sekarang rejected
    if len(sh) >= 2:
        resistance = df["high"].iloc[sh[-2]]   # swing high terakhir = resistance
        touched    = P["high"] >= resistance - atr * 0.5   # candle sebelum menyentuh
        rejected   = L["close"] < resistance - atr * 0.3  # sekarang jauh di bawah
        bearish_c  = L["close"] < L["open"]               # candle bearish
        if touched and rejected and bearish_c:
            result["failed_retest_sell"] = True
            result["resistance"] = resistance

    # Failed retest BUY: candle sebelumnya menyentuh support, sekarang bounced
    if len(sl) >= 2:
        support  = df["low"].iloc[sl[-2]]      # swing low terakhir = support
        touched  = P["low"] <= support + atr * 0.5
        bounced  = L["close"] > support + atr * 0.3
        bullish_c = L["close"] > L["open"]
        if touched and bounced and bullish_c:
            result["failed_retest_buy"] = True
            result["support"] = support

    return result


# ═════════════════════════════════════════════
# TAHAP 1: SCORING NORMAL — cari sinyal terkuat
# ═════════════════════════════════════════════
def score_direction(df_h1, df_m15):
    """
    Analisis normal untuk menentukan arah dan koin terbaik.
    Return: dict dengan symbol, direction asli, confidence, price
    """
    h1=build_df(df_h1); m15=build_df(df_m15)
    if h1 is None or m15 is None: return None

    L1=h1.iloc[-1]; P1=h1.iloc[-2]
    L15=m15.iloc[-1]; P15=m15.iloc[-2]
    rv=L15["rsi"]

    bull=bear=0

    # Trend H1
    if L1["ema9"]>L1["ema21"]>L1["ema50"]:  bull+=15
    elif L1["ema9"]>L1["ema21"]:             bull+=8
    if L1["ema9"]<L1["ema21"]<L1["ema50"]:  bear+=15
    elif L1["ema9"]<L1["ema21"]:             bear+=8
    if L1["close"]>L1["ema200"]:             bull+=8
    else:                                     bear+=8

    # RSI M15
    if rv<35:    bull+=12
    elif rv<45:  bull+=6
    if rv>65:    bear+=12
    elif rv>55:  bear+=6

    # MACD M15
    if L15["mh"]>0 and P15["mh"]<=0:  bull+=12
    elif L15["mh"]>0:                  bull+=5
    if L15["mh"]<0 and P15["mh"]>=0:  bear+=12
    elif L15["mh"]<0:                  bear+=5

    # EMA M15
    if L15["ema9"]>L15["ema21"]>L15["ema50"]: bull+=10
    elif L15["ema9"]>L15["ema21"]:             bull+=5
    if L15["ema9"]<L15["ema21"]<L15["ema50"]: bear+=10
    elif L15["ema9"]<L15["ema21"]:             bear+=5

    # Bollinger
    if L15["close"]<=L15["bb_lo"]:    bull+=10
    elif L15["close"]<L15["bb_mid"]:  bull+=4
    if L15["close"]>=L15["bb_up"]:    bear+=10
    elif L15["close"]>L15["bb_mid"]:  bear+=4

    # Volume
    if L15["volume"]>L15["vol_sma"]*1.5:
        if L15["close"]>L15["open"]:  bull+=8
        else:                          bear+=8
    elif L15["volume"]>L15["vol_sma"]:
        if L15["close"]>L15["open"]:  bull+=3
        else:                          bear+=3

    # Market Structure H1
    sh1,sl1=swing_pts(h1,5)
    struct_h1=mkt_struct(h1,sh1,sl1)
    if struct_h1=="bullish": bull+=12
    if struct_h1=="bearish": bear+=12

    # SMC M15 — BOS / CHoCH
    sh15,sl15=swing_pts(m15,5)
    bos=detect_bos(m15,sh15,sl15)
    if bos["bb"]: bull+=15
    if bos["cb"]: bull+=10
    if bos["bs"]: bear+=15
    if bos["cs"]: bear+=10

    # ── CHoCH M15 — bobot tinggi, ini konfirmasi perubahan arah nyata ──
    atr_val=max(L15["atr"], L15["close"]*0.003)
    choch = detect_choch(m15, sh15, sl15)
    if choch["bullish_choch"]: bull+=20
    if choch["bearish_choch"]: bear+=20

    # ── CHoCH H1 — lebih kuat lagi ──
    choch_h1 = detect_choch(h1, sh1, sl1)
    if choch_h1["bullish_choch"]: bull+=25
    if choch_h1["bearish_choch"]: bear+=25

    # ── Failed Retest M15 — trigger entry paling valid ──
    fr = detect_failed_retest(m15, sh15, sl15, atr_val)
    if fr["failed_retest_sell"]: bear+=25
    if fr["failed_retest_buy"]:  bull+=25

    # ── Failed Retest H1 ──
    fr_h1 = detect_failed_retest(h1, sh1, sl1, atr_val)
    if fr_h1["failed_retest_sell"]: bear+=20
    if fr_h1["failed_retest_buy"]:  bull+=20

    # Candle pattern
    body=L15["close"]-L15["open"]
    low_wick=min(L15["open"],L15["close"])-L15["low"]
    up_wick=L15["high"]-max(L15["open"],L15["close"])
    if low_wick>abs(body)*1.5: bull+=8  # hammer
    if up_wick>abs(body)*1.5:  bear+=8  # shooting star

    direction="bull" if bull>=bear else "bear"
    raw=bull if direction=="bull" else bear
    conf=min(int(raw/230*100),99)

    # ── Konfirmasi D1 ──────────────────────────────────────────────
    d1_bias = "neutral"
    try:
        df_d1 = build_df(df_h1.resample("1D").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).dropna())
        if df_d1 is not None and len(df_d1) >= 10:
            LD = df_d1.iloc[-1]
            sh_d, sl_d = swing_pts(df_d1, lb=3)
            struct_d1  = mkt_struct(df_d1, sh_d, sl_d)
            if LD["ema9"] < LD["ema21"] < LD["ema50"] and struct_d1 == "bearish":
                d1_bias = "bearish"
            elif LD["ema9"] > LD["ema21"] > LD["ema50"] and struct_d1 == "bullish":
                d1_bias = "bullish"
    except Exception:
        pass

    # D1 berlawanan dengan sinyal → hard block, bukan hanya penalty
    if d1_bias == "bearish" and direction == "bull": return None
    if d1_bias == "bullish" and direction == "bear": return None

    # CHoCH dan failed retest hanya valid jika searah H1 structure
    # Sudah dihitung di atas — tapi kalau CHoCH berlawanan H1, kurangi bobotnya
    if struct_h1 == "bearish" and choch["bullish_choch"]:
        bull = max(0, bull - 20)
    if struct_h1 == "bullish" and choch["bearish_choch"]:
        bear = max(0, bear - 20)
    if struct_h1 == "bearish" and choch_h1["bullish_choch"]:
        bull = max(0, bull - 25)
    if struct_h1 == "bullish" and choch_h1["bearish_choch"]:
        bear = max(0, bear - 25)

    # Recalc direction dan conf setelah koreksi
    direction = "bull" if bull >= bear else "bear"
    raw  = bull if direction == "bull" else bear
    conf = min(int(raw / 230 * 100), 99)

    return {
        "direction"       : direction,
        "confidence"      : conf,
        "price"           : L15["close"],
        "atr"             : atr_val,
        "struct_h1"       : struct_h1,
        "d1_bias"         : d1_bias,
        "rsi"             : round(rv,1),
        "bull_pts"        : bull,
        "bear_pts"        : bear,
        "choch_m15"       : choch,
        "choch_h1"        : choch_h1,
        "failed_retest"   : fr,
    }


# ═════════════════════════════════════════════
# TAHAP 2: ANALISIS ULANG — SL DULU, LALU TP
# ═════════════════════════════════════════════
# ── Tier kekuatan level untuk pemilihan TP ──────────────────────────
# Tier lebih rendah = level lebih kuat/reliable sebagai target liquidity.
# 1=liquidity pool (eq highs/lows) — paling sering jadi tujuan harga institusi
# 2=supply/demand zone — area order block besar
# 3=FVG — gap yang cenderung "ditarik" tapi lebih lemah dari zone
# 4=swing point — sekadar local extremum, paling lemah
TP_TIER = {
    "eq_low_m15": 1, "eq_high_m15": 1, "eq_low_h1": 1, "eq_high_h1": 1,
    "demand_top_m15": 2, "supply_bot_m15": 2, "demand_top_h1": 2, "supply_bot_h1": 2,
    "fvg_bear_m15": 3, "fvg_bull_m15": 3,
    "sw_low_m15": 4, "sw_high_m15": 4, "sw_low_h1": 4, "sw_high_h1": 4,
    # 5-6 = proyeksi Fibonacci extension — level yang BELUM pernah "dibuktikan"
    # market (beda dari swing yang memang sudah jadi titik balik harga
    # sebelumnya). Sengaja ditandai paling lemah dan hanya aktif kalau
    # gate H4 confluence lolos — lihat _h4_confluence().
    "fib_ext_127": 5, "fib_ext_162": 6,
}


def _h4_confluence(df_h1, direction, choch_m15=None):
    """
    Konfirmasi H4 untuk membuka kandidat TP Fibonacci extension.
    Resample dari H1 yang sudah di-fetch — TIDAK ada API call tambahan
    (pola sama persis dengan d1_bias di score_direction()).

    Syarat 'confluence' (unlock fib 1.272):
      BUY  : EMA9>EMA21>EMA50 H4 + struktur H4 bullish + RSI H4 di [45,68]
      SELL : EMA9<EMA21<EMA50 H4 + struktur H4 bearish + RSI H4 di [32,55]

    Syarat 'full_confluence' (unlock fib 1.618, tambahan):
      confluence di atas TERPENUHI + CHoCH M15 searah trade.
      Ini level paling jauh/spekulatif — baru boleh dipakai kalau H4
      DAN M15 dan RSI semuanya sepakat, bukan cuma H4 saja.

    Return: {"confluence": bool, "full_confluence": bool}
    """
    result = {"confluence": False, "full_confluence": False}
    try:
        df_h4 = build_df(df_h1.resample("4h").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).dropna())
        if df_h4 is None or len(df_h4) < 20:
            return result

        L4 = df_h4.iloc[-1]
        sh4, sl4 = swing_pts(df_h4, lb=3)
        struct_h4 = mkt_struct(df_h4, sh4, sl4)
        rsi_h4 = L4["rsi"]

        if direction == "bull":
            ema_ok = L4["ema9"] > L4["ema21"] > L4["ema50"]
            struct_ok = struct_h4 == "bullish"
            rsi_ok = H4_RSI_BUY_MIN <= rsi_h4 <= H4_RSI_BUY_MAX
        else:
            ema_ok = L4["ema9"] < L4["ema21"] < L4["ema50"]
            struct_ok = struct_h4 == "bearish"
            rsi_ok = H4_RSI_SELL_MIN <= rsi_h4 <= H4_RSI_SELL_MAX

        result["confluence"] = bool(ema_ok and struct_ok and rsi_ok)

        if result["confluence"] and choch_m15:
            choch_agrees = (
                (direction == "bull" and choch_m15.get("bullish_choch")) or
                (direction == "bear" and choch_m15.get("bearish_choch"))
            )
            result["full_confluence"] = bool(choch_agrees)
    except Exception:
        pass
    return result


def _fib_extension_levels(h1, sh1, sl1, direction):
    """
    Proyeksi Fibonacci extension dari leg swing H1 terakhir (low→high untuk
    BUY, high→low untuk SELL). Bukan angka dikarang — ini proyeksi dari
    RENTANG pergerakan H1 yang sudah benar-benar terjadi di chart.

    Return: (fib_127_price, fib_162_price) atau (None, None) kalau swing
    H1 belum cukup terbentuk.
    """
    if not sh1 or not sl1:
        return None, None
    swing_high = h1["high"].iloc[sh1[-1]]
    swing_low  = h1["low"].iloc[sl1[-1]]
    leg = swing_high - swing_low
    if leg <= 0:
        return None, None

    if direction == "bull":
        return swing_high + leg * FIB_EXT_1, swing_high + leg * FIB_EXT_2
    else:
        return swing_low - leg * FIB_EXT_1, swing_low - leg * FIB_EXT_2


def _select_best_tp(tp_pool, entry_price, risk):
    """
    Pilih TP terbaik dari tp_pool — list berisi (label, price, tier).

    Floor RR >= MIN_RR WAJIB dipenuhi, tidak pernah dilanggar.
    Di antara semua kandidat yang lolos floor itu, utamakan level paling
    KUAT (tier terendah) selama RR-nya masih masuk akal — dibatasi sampai
    MIN_RR × TP_MAX_RR_MULT supaya tidak mengejar target yang terlalu jauh
    / tidak realistis. Kalau level kuat tidak ada dalam rentang itu,
    fallback ke kandidat TERDEKAT yang lolos RR >= MIN_RR (perilaku lama).

    Hasilnya: RR sering > MIN_RR saat ada level kuat yang mendukung,
    tapi tidak pernah < MIN_RR.
    """
    qualifying = []
    for lbl, v, tier in tp_pool:
        rr_c = abs(v - entry_price) / risk
        if rr_c >= MIN_RR:
            qualifying.append((lbl, v, tier, rr_c))
    if not qualifying:
        return None, None

    # Kandidat paling dekat yang lolos minimum — ini fallback paling aman
    nearest = min(qualifying, key=lambda x: x[3])

    # Kandidat dengan RR yang masih dalam batas wajar
    bounded = [c for c in qualifying if c[3] <= MIN_RR * TP_MAX_RR_MULT]
    if bounded:
        # Tier terkuat (terendah) menang; kalau seri tier, ambil RR tertinggi
        best = min(bounded, key=lambda x: (x[2], -x[3]))
        return round(best[1], 8), best[0]

    return round(nearest[1], 8), nearest[0]


def analyze_setup(df_h1, df_m15, direction, entry_price, score=None):
    """
    Tentukan SL dan TP dari level struktural chart.

    Urutan WAJIB:
    1. Tentukan SL dari LIQUIDITY POOL (equal highs/lows) atau OB/supply-demand zone.
       SL harus di level di mana jika harga sampai ke sana, analisis sudah TERBUKTI SALAH.
    2. Hitung risk = abs(entry - SL)
    3. Iterasi TP candidates dari terdekat ke terjauh.
       Ambil level pertama yang menghasilkan RR >= 2.0.

    SELL: SL di atas equal highs / OB top / swing high
    BUY : SL di bawah equal lows / demand bot / swing low

    score (opsional): dict hasil score_direction() — dipakai untuk ambil
    choch_m15 saat menentukan apakah fib extension 1.618 boleh dipakai
    (full_confluence). Kalau None, fib 1.618 tidak akan pernah aktif
    (fib 1.272 tetap bisa aktif dari H4 trend+RSI saja).
    """
    h1  = build_df(df_h1)
    m15 = build_df(df_m15)
    if h1 is None or m15 is None: return None

    L15 = m15.iloc[-1]
    L1  = h1.iloc[-1]
    atr_m15 = max(L15["atr"], entry_price * 0.002)
    atr_h1  = max(L1["atr"],  entry_price * 0.004)
    # Pakai ATR H1 sebagai referensi minimum SL — lebih representatif dari volatilitas nyata
    atr = atr_m15
    # Jaring pengaman noise/spread saja, BUKAN acuan jarak SL — SL tetap
    # harus datang dari level struktural, floor ini cuma turun tangan
    # kalau level tersebut benar-benar terlalu mepet dari entry.
    sl_min = max(atr_h1 * 0.3, atr_m15 * 0.6)

    sh15, sl15 = swing_pts(m15, lb=5)
    sh1,  sl1  = swing_pts(h1,  lb=5)

    # Kumpulkan semua level struktural
    supply_zones = find_supply_demand(m15, "supply")
    demand_zones = find_supply_demand(m15, "demand")
    eq_highs     = find_equal_highs_lows(m15, "high", lb=80)
    eq_lows      = find_equal_highs_lows(m15, "low",  lb=80)
    fvg_bear     = find_fvg(m15, "bear")
    fvg_bull     = find_fvg(m15, "bull")

    # H1 levels untuk konteks lebih besar
    eq_highs_h1  = find_equal_highs_lows(h1, "high", lb=50)
    eq_lows_h1   = find_equal_highs_lows(h1, "low",  lb=50)
    supply_h1    = find_supply_demand(h1, "supply")
    demand_h1    = find_supply_demand(h1, "demand")

    # ── Fibonacci Extension TP (gated H4 confluence) ────────────────────
    # Dihitung SEKALI di sini, dipakai kedua cabang (SELL/BUY) di bawah.
    # h4_gate menentukan level mana yang BOLEH masuk tp_pool — bukan
    # dipanggil hanya saat level struktural gagal (itu akan jadi bias
    # seleksi/"penyelamat"). Kandidat ini selalu dievaluasi berdampingan
    # dengan level struktural lain via _select_best_tp, untuk SEMUA sinyal.
    choch_m15_for_gate = (score or {}).get("choch_m15", {})
    h4_gate = _h4_confluence(df_h1, direction, choch_m15_for_gate)
    fib_127, fib_162 = _fib_extension_levels(h1, sh1, sl1, direction)

    sl_price = None
    sl_label = ""
    reasons  = []

    # ══════════════════════════════════════════════════════════════
    # SELL — sinyal bear, kita SELL
    # SL di atas liquidity pool / supply zone terdekat di atas entry
    # TP di bawah entry (demand zone / equal lows / swing low)
    # ══════════════════════════════════════════════════════════════
    if direction == "bear":

        # Kumpulkan kandidat SL — level di atas entry
        sl_pool = []

        # Prioritas 1: Equal highs M15 — zona liquidity sweep paling sering
        # SL di ATAS equal highs dengan buffer lebih besar agar tidak kena sweep
        for eh in sorted(eq_highs):
            if eh > entry_price + atr * 0.3:
                sl_pool.append(("eq_high_m15", eh + atr * 0.6))
                break

        # Prioritas 2: Supply zone top M15
        for z in supply_zones:
            if z["top"] > entry_price + atr * 0.2:
                sl_pool.append(("supply_top_m15", z["top"] + atr * 0.5))

        # Prioritas 3: Swing high M15 paling dekat di atas entry
        sh_above = sorted([m15["high"].iloc[i] for i in sh15
                           if m15["high"].iloc[i] > entry_price + atr * 0.2])
        if sh_above:
            sl_pool.append(("swing_h_m15", sh_above[0] + atr * 0.5))

        # Prioritas 4: Equal highs H1 (level lebih kuat)
        for eh in sorted(eq_highs_h1):
            if eh > entry_price + atr * 0.5:
                sl_pool.append(("eq_high_h1", eh + atr * 0.7))
                break

        # Prioritas 5: Supply zone H1
        for z in supply_h1:
            if z["top"] > entry_price + atr * 0.5:
                sl_pool.append(("supply_top_h1", z["top"] + atr * 0.6))

        # Pilih SL terdekat di atas entry dari pool
        sl_pool_valid = [(lbl, v) for lbl, v in sl_pool
                         if v > entry_price + atr * 0.15]
        if not sl_pool_valid:
            sl_price = entry_price + sl_min
            sl_label = "atr_fallback"
        else:
            sl_label, sl_price = min(sl_pool_valid, key=lambda x: x[1])

        risk = abs(sl_price - entry_price)
        if risk < sl_min:
            # Level struktural terlalu mepet — TAMBAH jarak secukupnya di
            # atas level itu (tetap dari analisa), bukan diganti angka ATR
            # yang lepas dari chart. (Untuk kasus atr_fallback, risk sudah
            # persis == sl_min sehingga baris ini tidak pernah tereksekusi.)
            sl_price += (sl_min - risk)
            risk = sl_min
            sl_label += "_topup"

        reasons.append(f"SL@{sl_price:.5g}({sl_label})")
        min_reward = risk * MIN_RR

        # Kumpulkan semua TP candidates di bawah entry — setiap kandidat
        # ditandai tier kekuatannya (lihat TP_TIER) untuk pemilihan TP
        tp_pool = []

        # Equal lows M15 (liquidity target — sering menjadi tujuan harga)
        for el in sorted(eq_lows, reverse=True):
            if el < entry_price - atr * 0.5:
                tp_pool.append(("eq_low_m15", el, TP_TIER["eq_low_m15"]))

        # Demand zone top M15 (harga sering berhenti di atas demand)
        for z in sorted(demand_zones, key=lambda x: x["top"], reverse=True):
            if z["top"] < entry_price - atr * 0.5:
                tp_pool.append(("demand_top_m15", z["top"], TP_TIER["demand_top_m15"]))

        # FVG bear mid M15
        for fvg in sorted(fvg_bear, key=lambda x: x["mid"], reverse=True):
            if fvg["mid"] < entry_price - atr * 0.5:
                tp_pool.append(("fvg_bear_m15", fvg["mid"], TP_TIER["fvg_bear_m15"]))

        # Swing low M15
        for v in sorted([m15["low"].iloc[i] for i in sl15], reverse=True):
            if v < entry_price - atr * 0.5:
                tp_pool.append(("sw_low_m15", v, TP_TIER["sw_low_m15"]))

        # Equal lows H1 (target lebih jauh)
        for el in sorted(eq_lows_h1, reverse=True):
            if el < entry_price - atr * 1.0:
                tp_pool.append(("eq_low_h1", el, TP_TIER["eq_low_h1"]))

        # Demand zone H1
        for z in sorted(demand_h1, key=lambda x: x["top"], reverse=True):
            if z["top"] < entry_price - atr * 1.0:
                tp_pool.append(("demand_top_h1", z["top"], TP_TIER["demand_top_h1"]))

        # Swing low H1
        for v in sorted([h1["low"].iloc[i] for i in sl1], reverse=True):
            if v < entry_price - atr * 1.0:
                tp_pool.append(("sw_low_h1", v, TP_TIER["sw_low_h1"]))

        # Fibonacci extension (bearish, proyeksi di bawah swing low H1) —
        # HANYA masuk pool kalau gate H4 confluence lolos. Tetap dievaluasi
        # lewat mekanisme nearest/tier-first yang sama seperti level lain.
        if fib_127 is not None and fib_127 < entry_price - atr * 0.5:
            if h4_gate["confluence"]:
                tp_pool.append(("fib_ext_127", fib_127, TP_TIER["fib_ext_127"]))
            if h4_gate["full_confluence"] and fib_162 is not None \
               and fib_162 < entry_price - atr * 0.5:
                tp_pool.append(("fib_ext_162", fib_162, TP_TIER["fib_ext_162"]))

        # Pilih TP: lolos RR >= MIN_RR WAJIB, tapi utamakan level paling kuat
        # (tier terendah) dalam batas RR wajar — RR bisa > MIN_RR kalau level
        # kuat itu ada lebih jauh, fallback ke nearest-qualifying kalau tidak.
        tp_price, tp_label = _select_best_tp(tp_pool, entry_price, risk)

        if tp_price is None:
            return None  # tidak ada level TP yang menghasilkan RR >= 2

        reasons.append(f"TP@{tp_price:.5g}({tp_label})")

    # ══════════════════════════════════════════════════════════════
    # BUY — sinyal bull, kita BUY
    # SL di bawah liquidity pool / demand zone terdekat di bawah entry
    # TP di atas entry (supply zone / equal highs / swing high)
    # ══════════════════════════════════════════════════════════════
    else:

        sl_pool = []

        # Prioritas 1: Equal lows M15 — liquidity pool di bawah
        # SL di BAWAH equal lows dengan buffer lebih besar agar tidak kena sweep
        for el in sorted(eq_lows, reverse=True):
            if el < entry_price - atr * 0.3:
                sl_pool.append(("eq_low_m15", el - atr * 0.6))
                break

        # Prioritas 2: Demand zone bot M15
        for z in demand_zones:
            if z["bot"] < entry_price - atr * 0.2:
                sl_pool.append(("demand_bot_m15", z["bot"] - atr * 0.5))

        # Prioritas 3: Swing low M15 paling dekat di bawah
        sl_below = sorted([m15["low"].iloc[i] for i in sl15
                           if m15["low"].iloc[i] < entry_price - atr * 0.2],
                          reverse=True)
        if sl_below:
            sl_pool.append(("swing_l_m15", sl_below[0] - atr * 0.5))

        # Prioritas 4: Equal lows H1
        for el in sorted(eq_lows_h1, reverse=True):
            if el < entry_price - atr * 0.5:
                sl_pool.append(("eq_low_h1", el - atr * 0.7))
                break

        # Prioritas 5: Demand zone H1
        for z in demand_h1:
            if z["bot"] < entry_price - atr * 0.5:
                sl_pool.append(("demand_bot_h1", z["bot"] - atr * 0.6))

        sl_pool_valid = [(lbl, v) for lbl, v in sl_pool
                         if v < entry_price - atr * 0.15]
        if not sl_pool_valid:
            sl_price = entry_price - sl_min
            sl_label = "atr_fallback"
        else:
            sl_label, sl_price = max(sl_pool_valid, key=lambda x: x[1])

        risk = abs(entry_price - sl_price)
        if risk < sl_min:
            # Level struktural terlalu mepet — TAMBAH jarak secukupnya di
            # bawah level itu (tetap dari analisa), bukan diganti angka
            # ATR yang lepas dari chart. (Kasus atr_fallback: risk sudah
            # persis == sl_min, baris ini tidak pernah tereksekusi.)
            sl_price -= (sl_min - risk)
            risk = sl_min
            sl_label += "_topup"

        reasons.append(f"SL@{sl_price:.5g}({sl_label})")
        min_reward = risk * MIN_RR

        # TP candidates di atas entry — setiap kandidat ditandai tier
        # kekuatannya (lihat TP_TIER) untuk pemilihan TP
        tp_pool = []

        # Equal highs M15
        for eh in sorted(eq_highs):
            if eh > entry_price + atr * 0.5:
                tp_pool.append(("eq_high_m15", eh, TP_TIER["eq_high_m15"]))

        # Supply zone bot M15
        for z in sorted(supply_zones, key=lambda x: x["bot"]):
            if z["bot"] > entry_price + atr * 0.5:
                tp_pool.append(("supply_bot_m15", z["bot"], TP_TIER["supply_bot_m15"]))

        # FVG bull mid M15
        for fvg in sorted(fvg_bull, key=lambda x: x["mid"]):
            if fvg["mid"] > entry_price + atr * 0.5:
                tp_pool.append(("fvg_bull_m15", fvg["mid"], TP_TIER["fvg_bull_m15"]))

        # Swing high M15
        for v in sorted([m15["high"].iloc[i] for i in sh15]):
            if v > entry_price + atr * 0.5:
                tp_pool.append(("sw_high_m15", v, TP_TIER["sw_high_m15"]))

        # Equal highs H1
        for eh in sorted(eq_highs_h1):
            if eh > entry_price + atr * 1.0:
                tp_pool.append(("eq_high_h1", eh, TP_TIER["eq_high_h1"]))

        # Supply zone H1
        for z in sorted(supply_h1, key=lambda x: x["bot"]):
            if z["bot"] > entry_price + atr * 1.0:
                tp_pool.append(("supply_bot_h1", z["bot"], TP_TIER["supply_bot_h1"]))

        # Swing high H1
        for v in sorted([h1["high"].iloc[i] for i in sh1]):
            if v > entry_price + atr * 1.0:
                tp_pool.append(("sw_high_h1", v, TP_TIER["sw_high_h1"]))

        # Fibonacci extension (bullish, proyeksi di atas swing high H1) —
        # HANYA masuk pool kalau gate H4 confluence lolos. Tetap dievaluasi
        # lewat mekanisme nearest/tier-first yang sama seperti level lain.
        if fib_127 is not None and fib_127 > entry_price + atr * 0.5:
            if h4_gate["confluence"]:
                tp_pool.append(("fib_ext_127", fib_127, TP_TIER["fib_ext_127"]))
            if h4_gate["full_confluence"] and fib_162 is not None \
               and fib_162 > entry_price + atr * 0.5:
                tp_pool.append(("fib_ext_162", fib_162, TP_TIER["fib_ext_162"]))

        # Pilih TP: lolos RR >= MIN_RR WAJIB, tapi utamakan level paling kuat
        # (tier terendah) dalam batas RR wajar — RR bisa > MIN_RR kalau level
        # kuat itu ada lebih jauh, fallback ke nearest-qualifying kalau tidak.
        tp_price, tp_label = _select_best_tp(tp_pool, entry_price, risk)

        if tp_price is None:
            return None

        reasons.append(f"TP@{tp_price:.5g}({tp_label})")

    # ── Hitung RR final ───────────────────────────────────────────────
    risk   = abs(entry_price - sl_price)
    reward = abs(tp_price - entry_price)
    if risk == 0: return None
    rr = round(reward / risk, 2)
    if rr < MIN_RR: return None

    return {
        "sl"    : round(sl_price, 8),
        "tp"    : round(tp_price, 8),
        "rr"    : rr,
        "reason": " | ".join(reasons),
    }


def find_ob(df, direction, lb=40):
    """
    Order Block (OB) — candle terakhir sebelum impulse move besar.
    Bullish OB: candle bearish terakhir sebelum rally kuat ke atas.
    Bearish OB: candle bullish terakhir sebelum drop kuat ke bawah.
    Entry ideal ada di retrace ke zona OB.
    """
    sub = df.iloc[-lb:]
    avg_body = (sub["close"] - sub["open"]).abs().mean()
    obs = []
    for i in range(1, len(sub) - 3):
        c   = sub.iloc[i]
        nx  = sub.iloc[i + 1]
        nx2 = sub.iloc[i + 2]
        impulse = abs(nx["close"] - nx["open"])
        if impulse < avg_body * 1.5: continue
        if direction == "bull":
            if (c["close"] < c["open"] and
                    nx["close"] > nx["open"] and
                    nx2["close"] > nx2["open"]):
                obs.append({"top": max(c["open"], c["close"]),
                            "bot": min(c["open"], c["close"]),
                            "mid": (c["open"] + c["close"]) / 2})
        else:
            if (c["close"] > c["open"] and
                    nx["close"] < nx["open"] and
                    nx2["close"] < nx2["open"]):
                obs.append({"top": max(c["open"], c["close"]),
                            "bot": min(c["open"], c["close"]),
                            "mid": (c["open"] + c["close"]) / 2})
    return obs[-3:] if obs else []


def _find_sweep_level(m15, h1, direction, ref_price, atr):
    """
    Cari level Liquidity Sweep RAW (tanpa buffer SL) terdekat di luar
    ref_price — sumbernya sama persis dengan SL pool di analyze_setup():
    eq highs/lows M15 → supply/demand zone M15 → swing M15
    → eq highs/lows H1 → supply/demand H1

    Ini level "mentah" tempat institusi biasa sweep liquidity retail —
    BUKAN posisi SL final (SL final = level ini + buffer, tetap dihitung
    terpisah di analyze_setup() dan tidak diubah oleh fungsi ini).

    direction='bear' → cari level di ATAS ref_price (untuk tarik entry SELL)
    direction='bull' → cari level di BAWAH ref_price (untuk tarik entry BUY)
    """
    if m15 is None: return None
    sh15, sl15 = swing_pts(m15, lb=5)
    levels = []

    if direction == "bear":
        for eh in sorted(find_equal_highs_lows(m15, "high", lb=80)):
            if eh > ref_price + atr * 0.2:
                levels.append(eh); break
        for z in find_supply_demand(m15, "supply"):
            if z["top"] > ref_price + atr * 0.2:
                levels.append(z["top"])
        sh_above = sorted([m15["high"].iloc[i] for i in sh15
                           if m15["high"].iloc[i] > ref_price + atr * 0.2])
        if sh_above:
            levels.append(sh_above[0])
        if h1 is not None:
            for eh in sorted(find_equal_highs_lows(h1, "high", lb=50)):
                if eh > ref_price + atr * 0.5:
                    levels.append(eh); break
            for z in find_supply_demand(h1, "supply"):
                if z["top"] > ref_price + atr * 0.5:
                    levels.append(z["top"])
        levels = [v for v in levels if v > ref_price]
        return min(levels) if levels else None

    else:
        for el in sorted(find_equal_highs_lows(m15, "low", lb=80), reverse=True):
            if el < ref_price - atr * 0.2:
                levels.append(el); break
        for z in find_supply_demand(m15, "demand"):
            if z["bot"] < ref_price - atr * 0.2:
                levels.append(z["bot"])
        sl_below = sorted([m15["low"].iloc[i] for i in sl15
                           if m15["low"].iloc[i] < ref_price - atr * 0.2], reverse=True)
        if sl_below:
            levels.append(sl_below[0])
        if h1 is not None:
            for el in sorted(find_equal_highs_lows(h1, "low", lb=50), reverse=True):
                if el < ref_price - atr * 0.5:
                    levels.append(el); break
            for z in find_supply_demand(h1, "demand"):
                if z["bot"] < ref_price - atr * 0.5:
                    levels.append(z["bot"])
        levels = [v for v in levels if v < ref_price]
        return max(levels) if levels else None


def calc_discount_entry(df_h1, df_m15, direction, current_price, atr):
    """
    Hitung harga entry diskon dari zona struktural:
    SELL → entry di zona premium (lebih tinggi) sebelum turun
    BUY  → entry di zona diskon  (lebih rendah) sebelum naik

    Preferensi (priority 1 = terbaik):
    1. OB (Order Block)
    2. FVG (Fair Value Gap)
    3. Equal Highs/Lows
    4. Fibonacci 50% retracement

    SWEEP PULL (tambahan):
    Setelah kandidat terbaik (base_entry) ditemukan, entry ditarik
    SWEEP_PULL_FACTOR dari jarak base_entry → level Liquidity Sweep
    terdekat (level raw yang sama dipakai analyze_setup() untuk SL,
    tapi di sini TANPA buffer SL). Alasannya: wick harga terbukti sering
    sampai ke level sweep itu (lihat notif "Liquidity Sweep" berulang)
    tanpa pernah dianggap entry — level yang sedikit lebih jauh ini
    justru lebih sering tersentuh daripada base_entry (OB/FVG) yang
    terlalu dekat dengan harga sekarang.
    SL TIDAK diubah oleh ini — analyze_setup() tetap menaruh SL di level
    sweep + buffernya sendiri, jadi urutannya selalu:
    entry (baru) < level sweep < SL (untuk SELL, sebaliknya untuk BUY).
    SL tidak pernah menyentuh level sweep karena bufer SL selalu
    ditambahkan DI LUAR level itu.
    """
    m15 = build_df(df_m15)
    h1  = build_df(df_h1)
    if m15 is None: return current_price, "market"

    sh15, sl15 = swing_pts(m15, lb=5)
    eq_highs   = find_equal_highs_lows(m15, "high", lb=80)
    eq_lows    = find_equal_highs_lows(m15, "low",  lb=80)
    fvg_bear   = find_fvg(m15, "bear")
    fvg_bull   = find_fvg(m15, "bull")
    ob_bull    = find_ob(m15, "bull")
    ob_bear    = find_ob(m15, "bear")
    candidates = []

    if direction == "bear":
        for ob in reversed(ob_bear):
            if ob["top"] > current_price + atr * 0.1:
                candidates.append((ob["top"], "ob_bear_top", 1))
            elif ob["mid"] > current_price + atr * 0.1:
                candidates.append((ob["mid"], "ob_bear_mid", 1))
        for fvg in reversed(fvg_bear):
            if fvg["top"] > current_price + atr * 0.1:
                candidates.append((fvg["mid"], "fvg_bear", 2))
        for eh in sorted(eq_highs):
            if current_price + atr * 0.2 < eh < current_price + atr * 3.0:
                candidates.append((eh, "eq_high_m15", 3))
                break
        if len(sh15) >= 2 and len(sl15) >= 1:
            fib50 = m15["low"].iloc[sl15[-1]] + (
                m15["high"].iloc[sh15[-1]] - m15["low"].iloc[sl15[-1]]) * 0.5
            if current_price + atr * 0.1 < fib50 < current_price + atr * 4.0:
                candidates.append((fib50, "fib50", 4))
        above = [(p, l, r) for p, l, r in candidates if p > current_price]
        if above:
            above.sort(key=lambda x: (x[2], x[0]))
            base_entry, base_label = above[0][0], above[0][1]

            sweep_level = _find_sweep_level(m15, h1, "bear", base_entry, atr)
            if sweep_level is not None and sweep_level > base_entry + atr * 0.1:
                pulled = base_entry + (sweep_level - base_entry) * SWEEP_PULL_FACTOR
                return round(pulled, 8), f"{base_label}+sweep_pull"
            return round(base_entry, 8), base_label

        # Tidak ada kandidat OB/FVG/EQH/Fib valid — sebelum jatuh ke market,
        # coba tarik dari current_price langsung ke arah level sweep terdekat.
        sweep_level = _find_sweep_level(m15, h1, "bear", current_price, atr)
        if sweep_level is not None and sweep_level > current_price + atr * 0.1:
            pulled = current_price + (sweep_level - current_price) * SWEEP_PULL_FACTOR
            return round(pulled, 8), "sweep_pull_only"
    else:
        for ob in reversed(ob_bull):
            if ob["top"] < current_price - atr * 0.1:
                candidates.append((ob["top"], "ob_bull_top", 1))
            elif ob["mid"] < current_price - atr * 0.1:
                candidates.append((ob["mid"], "ob_bull_mid", 1))
        for fvg in reversed(fvg_bull):
            if fvg["bot"] < current_price - atr * 0.1:
                candidates.append((fvg["mid"], "fvg_bull", 2))
        for el in sorted(eq_lows, reverse=True):
            if current_price - atr * 3.0 < el < current_price - atr * 0.2:
                candidates.append((el, "eq_low_m15", 3))
                break
        if len(sl15) >= 2 and len(sh15) >= 1:
            fib50 = m15["high"].iloc[sh15[-1]] - (
                m15["high"].iloc[sh15[-1]] - m15["low"].iloc[sl15[-1]]) * 0.5
            if current_price - atr * 4.0 < fib50 < current_price - atr * 0.1:
                candidates.append((fib50, "fib50", 4))
        below = [(p, l, r) for p, l, r in candidates if p < current_price]
        if below:
            below.sort(key=lambda x: (x[2], -x[0]))
            base_entry, base_label = below[0][0], below[0][1]

            sweep_level = _find_sweep_level(m15, h1, "bull", base_entry, atr)
            if sweep_level is not None and sweep_level < base_entry - atr * 0.1:
                pulled = base_entry - (base_entry - sweep_level) * SWEEP_PULL_FACTOR
                return round(pulled, 8), f"{base_label}+sweep_pull"
            return round(base_entry, 8), base_label

        # Tidak ada kandidat OB/FVG/EQL/Fib valid — sebelum jatuh ke market,
        # coba tarik dari current_price langsung ke arah level sweep terdekat.
        sweep_level = _find_sweep_level(m15, h1, "bull", current_price, atr)
        if sweep_level is not None and sweep_level < current_price - atr * 0.1:
            pulled = current_price - (current_price - sweep_level) * SWEEP_PULL_FACTOR
            return round(pulled, 8), "sweep_pull_only"

    return current_price, "market"


# ═════════════════════════════════════════════
# PIPELINE ANALISIS LENGKAP
# ═════════════════════════════════════════════
def full_analyze(symbol):
    """
    1. Score arah sinyal (H1 + M15 + D1 bias)
    2. Hitung entry diskon dari OB/FVG/EQL/Fib
    3. Hitung SL/TP dari entry diskon
    Entry = zona struktural, bukan market price
    """
    try:
        df_h1  = get_klines(symbol, "1h",  250)
        df_m15 = get_klines(symbol, "15m", 250)
        if df_h1.empty or df_m15.empty: return None

        score = score_direction(df_h1, df_m15)
        if score is None: return None

        original_dir  = score["direction"]
        current_price = score["price"]
        atr_val       = score["atr"]
        decision      = "BUY"  if original_dir == "bull" else "SELL"

        # Entry diskon dari zona struktural
        discount_entry, entry_label = calc_discount_entry(
            df_h1, df_m15, original_dir, current_price, atr_val)

        # SL/TP dihitung dari entry diskon
        setup = analyze_setup(df_h1, df_m15, original_dir, discount_entry, score=score)
        if setup is None: return None

        # TP wajib MASIH di depan harga sekarang. Kalau entry diskon
        # dihitung dari zona struktural yang sudah ditinggalkan jauh oleh
        # rally/dump kuat (biasanya RSI sudah ekstrem), TP hasil analisa
        # dari zona lama itu bisa sudah KELEWAT harga sekarang — sinyal
        # ini mati sebelum pending order sempat dibuat. Tolak di sini,
        # bukan menunggu pending-cancel logic menangkapnya belakangan.
        if original_dir == "bull" and current_price >= setup["tp"]:
            return None
        if original_dir == "bear" and current_price <= setup["tp"]:
            return None

        return {
            "symbol"       : symbol,
            "original_dir" : original_dir,
            "decision"     : decision,
            "confidence"   : score["confidence"],
            "price"        : current_price,
            "entry"        : discount_entry,
            "entry_label"  : entry_label,
            "sl"           : setup["sl"],
            "tp"           : setup["tp"],
            "rr"           : setup["rr"],
            "rsi"          : score["rsi"],
            "struct_h1"    : score["struct_h1"],
            "d1_bias"      : score.get("d1_bias", "neutral"),
            "choch_m15"    : score.get("choch_m15", {}),
            "choch_h1"     : score.get("choch_h1", {}),
            "failed_retest": score.get("failed_retest", {}),
            "tp_sl_reason" : f"Entry@{discount_entry:.5g}({entry_label}) | {setup['reason']}",
        }
    except Exception as e:
        log.debug(f"[full_analyze] {symbol}: {e}")
        return None


# ═════════════════════════════════════════════
# SCAN — 1 sinyal terbaik
# ═════════════════════════════════════════════
def run_scan_once(chat_id):
    tg_send(chat_id,f"🔍 Scanning {TOP_N_COINS} koin...")
    try:
        symbols=get_top_coins()
    except Exception as e:
        tg_send(chat_id,f"⚠️ Binance error: <code>{str(e)[:150]}</code>")
        return None

    if not symbols:
        tg_send(chat_id,"⚠️ Tidak ada koin tersedia untuk di-scan saat ini.")
        return None

    results=[]
    for idx,sym in enumerate(symbols,1):
        log.info(f"[{idx:02d}/{len(symbols)}] {sym}")
        r=full_analyze(sym)
        if r: results.append(r)
        time.sleep(0.08)

    if not results:
        tg_send(chat_id,"⚠️ Tidak ada setup valid dari semua koin.")
        return None

    # Filter: hanya koin dengan confidence >= MIN_CONFIDENCE (diatur via /confidence_min)
    results = [r for r in results if r["confidence"] >= MIN_CONFIDENCE]
    if not results:
        tg_send(chat_id,f"⚠️ Tidak ada koin dengan confidence cukup (≥{MIN_CONFIDENCE}%). Retry...")
        return None

    # Ranking: confidence DESC → rr DESC
    results.sort(key=lambda x:(x["confidence"],x["rr"]),reverse=True)
    best=results[0]
    log.info(f"Best: {best['symbol']} {best['decision']} "
             f"conf={best['confidence']}% RR=1:{best['rr']}")
    return best



# ═════════════════════════════════════════════
# STATISTIK + BALANCE
# ═════════════════════════════════════════════
POSITION_SIZE_PCT = 100.0  # ukuran posisi per trade = 100% saldo (setara 1× leverage)
                            # P&L murni dari jarak SL/TP yang ditetapkan analisis:
                            #   TP hit → gain = posisi × (tp_dist / entry)
                            #   SL hit → loss = posisi × (sl_dist / entry)
                            # Nilai ini TIDAK mempengaruhi PENEMPATAN SL/TP —
                            # hanya memengaruhi simulasi saldo.

def update_stats(result, entry=None, sl_p=None, tp_p=None, close_price=None):
    """
    Hitung P&L simulasi murni dari jarak harga analisis.

    Model: alokasikan POSITION_SIZE_PCT % dari saldo ke setiap trade.
    Gain/loss = posisi × (jarak harga tutup dari entry / entry).

    close_price:
      - None (TP/SL alami)  → P&L dihitung dari jarak penuh ke tp_p/sl_p,
        sesuai hasilnya (result="tp" pakai tp_p, result="sl" pakai sl_p).
      - Diberikan (mis. /timeout paksa) → P&L dihitung dari harga RIIL
        saat ditutup, bukan target penuh — posisi yang belum sempat
        capai TP/SL tidak dicatat seolah sudah full target.

    result ("tp"/"sl") menentukan kategori statistik (counter tp/sl),
    tapi arah gain/loss selalu disimpulkan dari posisi tp_p relatif ke
    entry (BUY: tp_p > entry, SELL: tp_p < entry) — jadi tanda P&L tetap
    benar untuk harga tutup manapun, tidak cuma pas persis di tp_p/sl_p.
    """
    with stat_lock:
        stats["total"] += 1
        if result in ("tp", "sl"):
            stats[result] += 1

        if not entry or tp_p is None:
            return

        balance      = stats["balance"]
        position_usd = round(balance * POSITION_SIZE_PCT / 100, 6)
        direction_sign = 1 if tp_p > entry else -1

        if close_price is not None:
            ref_price = close_price
        elif result == "tp":
            ref_price = tp_p
        elif result == "sl" and sl_p is not None:
            ref_price = sl_p
        else:
            return

        pnl_pct = (ref_price - entry) / entry * direction_sign
        pnl_usd = round(position_usd * pnl_pct, 4)
        pct     = round(pnl_pct * 100, 3)
        stats["balance"] = round(balance + pnl_usd, 4)
        stats["pnl_history"].append({
            "result": result, "pct": pct,
            "pnl_usd": pnl_usd, "balance_after": stats["balance"],
        })

def fmt_stats():
    with stat_lock:
        t   = stats["total"]
        tp  = stats["tp"]
        sl  = stats["sl"]
        bal = stats["balance"]
        hist= list(stats["pnl_history"])

    if t == 0:
        return (f"📊 <b>Statistik Simulasi</b>\n\n"
                f"Belum ada trade.\n"
                f"💵 Modal awal: <b>${STARTING_BALANCE:.2f}</b>")

    wr      = tp/(tp+sl)*100 if (tp+sl)>0 else 0
    pnl_tot = round(bal - STARTING_BALANCE, 4)
    pnl_pct = round(pnl_tot / STARTING_BALANCE * 100, 2)
    pnl_em  = "📈" if pnl_tot >= 0 else "📉"
    pnl_sgn = "+" if pnl_tot >= 0 else ""

    # Riwayat 5 trade terakhir
    recent = hist[-5:] if hist else []
    hist_lines = []
    for h in reversed(recent):
        em = "✅" if h["result"] == "tp" else "❌"
        sgn = "+" if h["pnl_usd"] >= 0 else ""
        hist_lines.append(
            f"  {em} {sgn}{h['pct']:.2f}%  {sgn}${h['pnl_usd']:.4f}  "
            f"→ ${h['balance_after']:.4f}"
        )
    hist_str = "\n".join(hist_lines) if hist_lines else "  (belum ada)"

    return (
        f"📊 <b>Statistik Simulasi</b>\n\n"
        f"Total trade  : {t}\n"
        f"🎯 TP        : {tp} ({tp/t*100:.1f}%)\n"
        f"🛑 SL        : {sl} ({sl/t*100:.1f}%)\n"
        f"📈 Win Rate  : <b>{wr:.1f}%</b> (dari {tp+sl} trade)\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 <b>Modal Awal : ${STARTING_BALANCE:.2f}</b>\n"
        f"💰 <b>Saldo Kini : ${bal:.4f}</b>\n"
        f"{pnl_em} <b>Total P&L  : {pnl_sgn}${pnl_tot:.4f} "
        f"({pnl_sgn}{pnl_pct:.2f}%)</b>\n"
        f"⚖️ Model P&L  : posisi {POSITION_SIZE_PCT:.0f}% saldo × % gerakan SL/TP\n\n"
        f"📋 5 Trade Terakhir:\n{hist_str}\n\n"
        f"🚫 Banned    : {len(banned_coins)}"
    )

def fmt_signal_msg(sig):
    em  = "🟢" if sig["decision"]=="BUY" else "🔴"
    bar = "█"*(sig["confidence"]//10)+"░"*(10-sig["confidence"]//10)
    dir_label = "BULLISH" if sig["original_dir"]=="bull" else "BEARISH"
    d1_em = {"bullish":"📈","bearish":"📉","neutral":"➡️"}.get(sig.get("d1_bias","neutral"),"➡️")
    d1_str = sig.get("d1_bias","neutral").upper()

    triggers = []
    choch_m15 = sig.get("choch_m15", {})
    choch_h1  = sig.get("choch_h1", {})
    fr        = sig.get("failed_retest", {})
    if choch_h1.get("bearish_choch"):   triggers.append("CHoCH Bearish H1")
    if choch_h1.get("bullish_choch"):   triggers.append("CHoCH Bullish H1")
    if choch_m15.get("bearish_choch"):  triggers.append("CHoCH Bearish M15")
    if choch_m15.get("bullish_choch"):  triggers.append("CHoCH Bullish M15")
    if fr.get("failed_retest_sell"):    triggers.append("Failed Retest Sell")
    if fr.get("failed_retest_buy"):     triggers.append("Failed Retest Buy")
    trigger_str = " | ".join(triggers) if triggers else "—"

    entry_label = sig.get("entry_label", "market")
    price_now   = sig.get("price", sig["entry"])
    entry_zone  = sig["entry"]
    entry_str = (
        f"📍 Harga kini: <code>{price_now:.6g}</code>\n"
        f"🎯 Entry zone: <code>{entry_zone:.6g}</code> ({entry_label})"
        if abs(price_now - entry_zone) / max(price_now, 0.0001) > 0.002
        else f"💰 Entry     : <code>{entry_zone:.6g}</code> ({entry_label})"
    )

    return (
        f"📡 <b>SINYAL DITEMUKAN</b>\n\n"
        f"Koin       : <b>{sig['symbol']}</b>\n"
        f"Analisis   : <b>{dir_label}</b> (confidence {sig['confidence']}% {bar})\n"
        f"Eksekusi   : {em} <b>{sig['decision']}</b>\n\n"
        f"{entry_str}\n"
        f"✅ TP      : <code>{sig['tp']:.6g}</code>\n"
        f"🛑 SL      : <code>{sig['sl']:.6g}</code>\n"
        f"⚖️ RR      : <b>1:{sig['rr']}</b>\n"
        f"RSI        : {sig['rsi']} | H1: {sig['struct_h1'].upper()} | D1: {d1_em} {d1_str}\n"
        f"🎯 Trigger : {trigger_str}\n\n"
        f"📝 Basis:\n{sig['tp_sl_reason']}"
    )


# ═════════════════════════════════════════════
# MULTI-POSITION BROADCASTER
# ═════════════════════════════════════════════
# MAX_POSITIONS dikontrol lewat /max — lihat konstanta di bagian atas file
MONITOR_INTERVAL = 15 * 60  # cek posisi tiap 15 menit (detik)

positions_lock = threading.Lock()
positions: dict = {}   # {sym: {signal, entry, tp, sl, entry_time, thread}}

def close_position(sym, result, close_price=None):
    """Tutup posisi, catat statistik, ban koin sementara, kirim notif."""
    global active_trade
    with positions_lock:
        pos = positions.pop(sym, None)
    if pos is None: return

    sig   = pos["signal"]
    entry = pos["entry"]
    sl_p  = sig["sl"]
    tp_p  = sig["tp"]
    cid   = pos["chat_id"]

    update_stats(result, entry=entry, sl_p=sl_p, tp_p=tp_p, close_price=close_price)
    _ban_coin(sym, f"trade closed ({result})")

    # Update active_trade jika ini yang sedang dipantau
    with positions_lock:
        if not positions:
            active_trade = None

    emoji = {"tp":"🎯","sl":"🛑"}.get(result,"❓")
    label = {"tp":"TAKE PROFIT","sl":"STOP LOSS"}.get(result, result.upper())
    tg_send(cid, f"{emoji} <b>{label}</b> — {sym}\n\n" + fmt_stats())


def check_tp_sl_order(sym, tp_p, sl_p, is_buy, lookback_min=15, df=None):
    """
    Ambil candle M1 dalam N menit terakhir, periksa urutan:
    mana yang kena duluan — TP atau SL?

    df (opsional): dataframe klines M1 yang SUDAH di-fetch caller —
    dipakai supaya tidak fetch ulang candle yang sama dua kali (caller
    sering butuh dataframe ini juga untuk keperluan lain, mis. verifikasi
    confirmed_sl di monitor_position).

    Return: "tp", "sl", atau None (tidak ada yang tersentuh)
    """
    try:
        if df is None:
            df = get_klines(sym, "1m", lookback_min + 2)
        if df is None or df.empty: return None

        # Ambil hanya candle dalam lookback_min menit terakhir
        df = df.tail(lookback_min)

        for _, row in df.iterrows():
            high = row["high"]
            low  = row["low"]
            if is_buy:
                # Untuk BUY: TP di atas, SL di bawah
                # Kalau high >= TP dan low <= SL di candle yang sama → cek open lebih dekat ke mana
                if high >= tp_p and low <= sl_p:
                    # Harga open candle ini lebih dekat ke TP atau SL?
                    dist_tp = abs(row["open"] - tp_p)
                    dist_sl = abs(row["open"] - sl_p)
                    return "tp" if dist_tp < dist_sl else "sl"
                elif high >= tp_p:
                    return "tp"
                elif low <= sl_p:
                    return "sl"
            else:
                # Untuk SELL: TP di bawah, SL di atas
                if low <= tp_p and high >= sl_p:
                    dist_tp = abs(row["open"] - tp_p)
                    dist_sl = abs(row["open"] - sl_p)
                    return "tp" if dist_tp < dist_sl else "sl"
                elif low <= tp_p:
                    return "tp"
                elif high >= sl_p:
                    return "sl"
    except Exception as e:
        log.debug(f"[check_tp_sl_order] {sym}: {e}")
    return None


def monitor_position(sym, pos):
    """
    Thread per-posisi: cek harga/TP/SL setiap MONITOR_SLEEP (10 detik),
    kirim pesan update ke Telegram tiap MONITOR_INTERVAL (15 menit) TANPA
    pernah menghentikan pengecekan harga di antaranya.
    Posisi hanya ditutup saat TP atau SL — tidak ada timeout otomatis.
    """
    sig     = pos["signal"]
    chat_id = pos["chat_id"]
    entry   = pos["entry"]
    tp_p    = sig["tp"]
    sl_p    = sig["sl"]
    is_buy  = sig["decision"] == "BUY"

    next_update_at = time.time() + MONITOR_INTERVAL

    while True:
        with positions_lock:
            if sym not in positions: return

        # Manual /timeout SYMBOL — tutup paksa sesuai PnL riil saat ini:
        # floating positif dicatat sebagai TP, floating negatif sebagai SL.
        # Bukan selalu "SL" — itu akan mencatat kerugian penuh meski posisi
        # sedang untung saat ditutup.
        if pos.get("timeout_flag"):
            pos["timeout_flag"] = False
            price = get_price(sym) or entry
            pnl_pct = (price - entry) / entry * (1 if is_buy else -1)
            result  = "tp" if pnl_pct >= 0 else "sl"
            emoji   = "🎯" if result == "tp" else "🛑"
            tg_send(chat_id,
                f"⏭ <b>Ditutup Manual</b> — {sym} {emoji}\n"
                f"Harga: <code>{price:.6g}</code> | PnL: <b>{pnl_pct*100:+.2f}%</b>\n"
                f"Dicatat sebagai {result.upper()} (sesuai PnL riil saat ditutup)")
            close_position(sym, result, close_price=price)
            return

        price = get_price(sym)
        if price is None:
            time.sleep(MONITOR_SLEEP); continue

        # ── Cek TP / SL — verifikasi via candle M1 ─────────────────
        hit_tp = (price >= tp_p) if is_buy else (price <= tp_p)
        hit_sl = (price <= sl_p) if is_buy else (price >= sl_p)

        if hit_tp or hit_sl:
            # Kalau masih dalam episode sweep yang sama (sudah kirim notif
            # sekali) dan belum lewat KLINE_VERIFY_THROTTLE detik sejak
            # verifikasi terakhir, JANGAN fetch klines lagi — cukup tunggu.
            # Tanpa ini, selama harga grinding di SL, kode akan hammer
            # Binance dengan 2 request klines setiap 10 detik tanpa henti.
            now_ts = time.time()
            last_check = pos.get("last_kline_check", 0)
            if pos.get("sweep_notified") and (now_ts - last_check) < KLINE_VERIFY_THROTTLE:
                time.sleep(MONITOR_SLEEP)
                continue

            pos["last_kline_check"] = now_ts
            try:
                df_m1 = get_klines(sym, "1m", 5)
            except Exception:
                df_m1 = None

            # Satu fetch di atas dipakai ulang utk 2 keperluan (dulu fetch
            # 2x terpisah untuk hal yang hampir identik):
            order = check_tp_sl_order(sym, tp_p, sl_p, is_buy, lookback_min=3, df=df_m1)
            if order is None:
                order = "tp" if hit_tp else "sl"

            if order == "tp":
                pct = abs(tp_p - entry) / entry * 100
                tg_send(chat_id,
                    f"🎯 <b>TAKE PROFIT</b> — {sym} 🎉\n"
                    f"TP: <code>{tp_p:.6g}</code>\n"
                    f"Profit: +{pct:.2f}%")
                close_position(sym, "tp")
                return
            else:
                confirmed_sl = False
                try:
                    if df_m1 is not None and not df_m1.empty:
                        last_closes = df_m1["close"].tail(3)
                        confirmed_sl = any(
                            (c <= sl_p) if is_buy else (c >= sl_p)
                            for c in last_closes
                        )
                    else:
                        # Tidak bisa fetch candle M1 — gunakan harga cache
                        # sebagai fallback agar SL tetap bisa terpicu
                        confirmed_sl = hit_sl
                except Exception:
                    confirmed_sl = hit_sl

                if confirmed_sl:
                    tg_send(chat_id,
                        f"🛑 <b>STOP LOSS</b> — {sym}\n"
                        f"Harga: <code>{price:.6g}</code> | SL: <code>{sl_p:.6g}</code>")
                    close_position(sym, "sl")
                    return
                else:
                    # Notif dikirim sekali per episode sweep (flag reset
                    # begitu kondisi sweep hilang). Iterasi berikutnya
                    # dalam window throttle akan skip fetch sama sekali
                    # (lihat pengecekan sweep_notified di atas).
                    if not pos.get("sweep_notified"):
                        tg_send(chat_id,
                            f"🔄 <b>Liquidity Sweep — {sym}</b>\n"
                            f"Wick menyentuh SL, candle M1 belum konfirmasi. Lanjut...")
                        pos["sweep_notified"] = True
                    time.sleep(MONITOR_SLEEP)
                    continue

        # Harga sudah tidak lagi menyentuh SL → reset flag notif sweep
        pos["sweep_notified"] = False

        # ── Update periodik — dikirim tanpa menghentikan pengecekan
        # harga. Loop tetap kembali ke atas tiap MONITOR_SLEEP dan tetap
        # mengecek TP/SL; hanya PESAN-nya yang dijadwalkan tiap 15 menit.
        if time.time() >= next_update_at:
            pnl_pct = (price - entry) / entry * 100 * (1 if is_buy else -1)
            tg_send(chat_id,
                f"📊 <b>Update 15m — {sym}</b>\n"
                f"Arah  : {'🟢 BUY' if is_buy else '🔴 SELL'}\n"
                f"Entry : <code>{entry:.6g}</code>\n"
                f"Harga : <code>{price:.6g}</code>\n"
                f"TP    : <code>{tp_p:.6g}</code>\n"
                f"SL    : <code>{sl_p:.6g}</code>\n"
                f"PnL   : <b>{pnl_pct:+.2f}%</b>")
            next_update_at = time.time() + MONITOR_INTERVAL

        time.sleep(MONITOR_SLEEP)


def simulation_loop(chat_id):
    """
    Broadcaster utama — non-blocking:
    - Scan berjalan di thread terpisah agar tidak block loop utama
    - Monitor per-posisi juga thread terpisah (sudah ada)
    - Loop utama hanya koordinasi: cek slot, launch scan/monitor
    """
    global auto_mode
    tg_send(chat_id,
        "🤖 <b>SMC Signal Broadcaster dimulai!</b>\n\n"
        "• Scan koin → catat sinyal → pantau tiap 15 menit\n"
        f"• Maks {MAX_POSITIONS} posisi bersamaan\n"
        "• Posisi ditutup hanya saat TP atau SL\n\n"
        "/stop untuk berhenti | /timeout SYMBOL untuk tutup paksa\n"
        "/trade untuk lihat semua posisi aktif")

    scanning = False          # flag: apakah scan sedang berjalan
    scan_lock = threading.Lock()

    def _do_scan():
        nonlocal scanning
        try:
            signal = run_scan_once(chat_id)
            if not auto_mode or signal is None:
                return

            sym = signal["symbol"]
            with positions_lock:
                if sym in positions: return
                if len(positions) >= MAX_POSITIONS: return

            entry_target = signal["entry"]
            current      = signal["price"]
            is_buy       = signal["decision"] == "BUY"
            tp_p         = signal["tp"]
            entry_label  = signal.get("entry_label", "market")

            already_at_entry = (
                (is_buy     and current <= entry_target * 1.002) or
                (not is_buy and current >= entry_target * 0.998)
            )

            if already_at_entry or entry_label == "market":
                # Langsung masuk
                actual_entry = get_price(sym) or current
                _open_position(sym, signal, actual_entry, chat_id, "langsung")
            else:
                # Daftarkan dulu sebagai pending agar tidak di-scan ulang
                with positions_lock:
                    if sym in positions: return
                    if len(positions) >= MAX_POSITIONS: return
                    positions[sym] = {
                        "signal"      : signal,
                        "entry"       : entry_target,
                        "chat_id"     : chat_id,
                        "entry_time"  : None,        # belum entry, set saat terpicu
                        "timeout_flag": False,
                        "status"      : "pending",
                    }

                dist_pct = abs(entry_target - current) / current * 100
                tg_send(chat_id,
                    f"🎯 <b>PENDING ORDER</b> — {sym}\n\n"
                    f"{fmt_signal_msg(signal)}\n\n"
                    f"⏳ Menunggu harga ke zona entry\n"
                    f"Harga kini : <code>{current:.6g}</code>\n"
                    f"Entry zone : <code>{entry_target:.6g}</code> ({entry_label})\n"
                    f"Jarak      : {dist_pct:.2f}%")
                threading.Thread(
                    target=_wait_entry,
                    args=(sym, signal, chat_id),
                    daemon=True
                ).start()
        finally:
            with scan_lock:
                scanning = False

    def _wait_entry(sym, signal, chat_id):
        """Thread terpisah — tunggu harga ke zona entry. /stop tidak
        membatalkan pending; hanya menghentikan scan koin baru."""
        entry_target = signal["entry"]
        is_buy       = signal["decision"] == "BUY"
        tp_p         = signal["tp"]
        sl_p         = signal["sl"]
        deadline     = time.time() + 8 * 3600

        while time.time() < deadline:
            with positions_lock:
                if sym not in positions: return

            price_now = get_price(sym)
            if price_now is None:
                time.sleep(MONITOR_SLEEP); continue

            # TP tersentuh sebelum entry → sinyal basi, hapus pending
            tp_hit = (price_now >= tp_p) if is_buy else (price_now <= tp_p)
            if tp_hit:
                with positions_lock:
                    positions.pop(sym, None)
                _ban_coin(sym, "TP sebelum entry")
                tg_send(chat_id,
                    f"⏭ <b>Pending Batal</b> — {sym}\n"
                    f"TP tersentuh sebelum entry. Skip.")
                return

            # SL tersentuh/dilewati sebelum entry → setup sudah tidak valid
            # (harga sudah membuktikan analisa salah sebelum posisi sempat
            # dibuka). Tanpa cek ini, harga bisa gap lewat SL dan entry_hit
            # tetap terpicu di harga yang geometrinya sudah rusak.
            sl_hit = (price_now <= sl_p) if is_buy else (price_now >= sl_p)
            if sl_hit:
                with positions_lock:
                    positions.pop(sym, None)
                _ban_coin(sym, "SL sebelum entry")
                tg_send(chat_id,
                    f"⏭ <b>Pending Batal</b> — {sym}\n"
                    f"SL tersentuh sebelum entry. Skip.")
                return

            # Harga mencapai zona entry
            entry_hit = (
                (is_buy     and price_now <= entry_target * 1.003) or
                (not is_buy and price_now >= entry_target * 0.997)
            )
            if entry_hit:
                _open_position(sym, signal, price_now, chat_id, "terpicu")
                return

            time.sleep(MONITOR_SLEEP)

        # Expired — hapus pending
        with positions_lock:
            positions.pop(sym, None)
        _ban_coin(sym, "pending expired")
        tg_send(chat_id,
            f"⏰ <b>Pending Expired</b> — {sym}\n"
            f"Harga tidak mencapai zona entry dalam 8 jam. Skip.")

    def _open_position(sym, signal, actual_entry, chat_id, mode_label):
        """Upgrade posisi dari pending ke aktif dan mulai monitor."""
        is_buy = signal["decision"] == "BUY"
        sl_v, tp_v = signal["sl"], signal["tp"]

        # Validasi geometri dulu — SL dan TP wajib di sisi yang benar dari
        # entry aktual. Wajib dicek sebelum rasio RR, karena rasio abs(jarak)
        # bisa tampak valid (>= MIN_RR) walau posisinya sebenarnya terbalik
        # (mis. harga gap lewat SL sebelum entry sempat tersentuh).
        geometry_ok = (sl_v < actual_entry < tp_v) if is_buy else (tp_v < actual_entry < sl_v)
        if not geometry_ok:
            with positions_lock:
                positions.pop(sym, None)
            _ban_coin(sym, "geometri invalid")
            tg_send(chat_id,
                f"⚠️ <b>Skip {sym}</b> — Geometri SL/TP tidak valid di entry aktual\n"
                f"Entry: <code>{actual_entry:.6g}</code> | "
                f"TP: <code>{tp_v:.6g}</code> | SL: <code>{sl_v:.6g}</code>")
            return

        # Verifikasi RR masih valid di harga entry aktual.
        # TP/SL dihitung dari discount_entry (analisis), tapi posisi
        # dibuka di harga nyata — selisihnya bisa membuat RR < MIN_RR.
        sl_dist = abs(actual_entry - sl_v)
        tp_dist = abs(tp_v - actual_entry)
        actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if actual_rr < MIN_RR:
            with positions_lock:
                positions.pop(sym, None)
            _ban_coin(sym, "RR gagal di entry aktual")
            tg_send(chat_id,
                f"⚠️ <b>Skip {sym}</b> — RR tidak memenuhi di entry aktual\n"
                f"Entry: <code>{actual_entry:.6g}</code> | "
                f"TP: <code>{tp_v:.6g}</code> | SL: <code>{sl_v:.6g}</code>\n"
                f"RR aktual: <b>1:{actual_rr:.2f}</b> (min 1:{MIN_RR})")
            return

        with positions_lock:
            if sym not in positions: return   # sudah dihapus (expired/batal)
            pos = positions[sym]
            pos["entry"]      = actual_entry
            pos["entry_time"] = time.time()
            pos["status"]     = "active"

        tg_send(chat_id,
            f"⚡ <b>ENTRY {mode_label.upper()}</b> — {sym}\n"
            f"Entry aktual: <code>{actual_entry:.6g}</code>\n"
            f"TP: <code>{tp_v:.6g}</code> | SL: <code>{sl_v:.6g}</code>\n"
            f"RR: <b>1:{actual_rr:.2f}</b> | 📡 Dipantau tiap 15 menit...")

        threading.Thread(
            target=monitor_position,
            args=(sym, pos),
            daemon=True
        ).start()

    while auto_mode:
        with positions_lock:
            n_pos = len(positions)

        # Slot penuh — tunggu saja
        if n_pos >= MAX_POSITIONS:
            time.sleep(5)
            continue

        # Kalau scan sedang berjalan — jangan launch scan baru
        with scan_lock:
            already_scanning = scanning
            if not already_scanning:
                scanning = True

        if already_scanning:
            time.sleep(5)
            continue

        # Launch scan di background
        threading.Thread(target=_do_scan, daemon=True).start()

        # Jeda antar scan agar tidak langsung re-scan begitu selesai
        time.sleep(5)

    tg_send(chat_id, "⏹ <b>Scanning dihentikan.</b>\n\n" + fmt_stats())



# ═════════════════════════════════════════════
# PESAN STATIS
# ═════════════════════════════════════════════
GREETING=(
    "👋 <b>SMC Signal Broadcaster</b>\n\n"
    f"Scan → sinyal → pantau max {MAX_POSITIONS} posisi bersamaan (update tiap 15 menit)\n"
    "Posisi ditutup hanya saat TP atau SL\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "/start               — Menu ini\n"
    "/auto                — Mulai broadcaster\n"
    "/stop                — Hentikan scanning (posisi aktif tetap dipantau)\n"
    "/trade               — Lihat semua posisi aktif\n"
    "/max                 — Lihat/ubah max posisi + info batas API\n"
    "/confidence_min      — Lihat/ubah ambang confidence minimum\n"
    "/timeout SYMBOL      — Tutup paksa posisi tertentu\n"
    "/timeout             — Tutup paksa semua posisi\n"
    "/stats               — Statistik + saldo\n"
    "/banned              — Daftar koin ban\n"
    "/resetban            — Hapus semua ban\n"
    "/resetbalance        — Reset saldo ke $10\n"
    "/info                — Detail metode analisis\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "⚠️ <i>Simulasi saja — bukan saran finansial.</i>"
)

def get_info_msg():
    return (
        "ℹ️ <b>Metode Analisis</b>\n\n"
        "<b>Tahap 1 — Scoring arah sinyal:</b>\n"
        "• EMA 9/21/50/200 alignment (H1 + M15)\n"
        "• RSI 14 oversold/overbought\n"
        "• MACD crossover momentum\n"
        "• Bollinger Bands posisi\n"
        "• Volume vs rata-rata\n"
        "• Market Structure H1 (HH/HL vs LH/LL)\n"
        "• BOS + CHoCH (M15)\n"
        "• Candle pattern (hammer, shooting star)\n\n"
        "<b>Tahap 2 — Penentuan SL (prioritas):</b>\n"
        "BUY: SL di bawah equal lows → demand zone → swing low\n"
        "SELL: SL di atas equal highs → supply zone → swing high\n"
        "SL minimum: 1.2× ATR agar tidak kena noise\n\n"
        "<b>Tahap 3 — Pemilihan TP (tier-based):</b>\n"
        "RR ≥ 1:2 WAJIB, tapi utamakan level PALING KUAT:\n"
        "1) eq highs/lows  2) supply/demand  3) FVG\n"
        "4) swing H1  5-6) Fibonacci extension (1.272/1.618)*\n"
        "*hanya aktif kalau H4 trend + RSI H4 + CHoCH M15 mendukung —\n"
        " level ini belum 'terbukti' market, jadi paling lemah & butuh\n"
        " konfirmasi ekstra. Selalu dievaluasi bareng level lain, bukan\n"
        " cabang khusus penyelamat RR gagal.\n\n"
        f"Min RR: 1:{MIN_RR} | Min Confidence: {MIN_CONFIDENCE}%\n"
        f"TF: H1 (bias) + M15 (entry) + H4 (fib gate)\n"
        f"Model P&L   : posisi {POSITION_SIZE_PCT:.0f}% saldo × % jarak SL/TP aktual\n"
        f"  → SL dekat (0.5%) = loss kecil | SL jauh (4%) = loss lebih besar\n"
        f"  → P&L murni dari level struktural analisis, bukan fixed -2%\n"
        f"Modal simulasi: ${STARTING_BALANCE:.2f}"
    )


# ═════════════════════════════════════════════
# BOT LOOP
# ═════════════════════════════════════════════
def bot_loop():
    global auto_mode, auto_thread, active_chat_id, timeout_flag, MAX_POSITIONS, MIN_CONFIDENCE

    log.info("Test koneksi Binance...")
    for i in range(10):
        remaining = _binance_ban_remaining()
        if remaining > 0:
            wait = min(remaining + 2, 300)   # tunggu sesuai sisa ban, maks 5 menit per iterasi
            log.warning(f"[startup] Binance masih diban {remaining:.0f}s — menunggu {wait:.0f}s...")
            time.sleep(wait)
            continue
        try:
            fapi_get("/fapi/v1/ping")
            log.info("Binance OK!")
            break
        except Exception as e:
            log.warning(f"Retry {i+1}/10: {e}")
            time.sleep(10)
    else:
        log.critical("Binance tidak bisa dijangkau.")
        return

    offset=None
    log.info("Bot siap.")

    while True:
        try:
            for upd in tg_updates(offset):
                offset=upd["update_id"]+1
                msg=upd.get("message",{})
                uid=msg.get("from",{}).get("id")
                chat_id=msg.get("chat",{}).get("id")
                text=msg.get("text","").strip().lower()
                if not uid or not chat_id or not text: continue
                if uid!=ALLOWED_USER_ID:
                    tg_send(chat_id,"⛔ Akses ditolak."); continue
                active_chat_id=chat_id

                if text in ("/start","start"):
                    tg_send(chat_id,GREETING)
                elif text in ("/info","info"):
                    tg_send(chat_id,get_info_msg())
                elif text in ("/stats","stats"):
                    tg_send(chat_id,fmt_stats())
                elif text in ("/banned","banned"):
                    with ban_lock:
                        cur_scan = scan_counter
                        b = sorted(banned_coins.items())
                    if b:
                        lines = []
                        for sym, banned_at in b:
                            remaining = max(0, BAN_DURATION_SCANS - (cur_scan - banned_at))
                            lines.append(f"• {sym} (unban dalam {remaining} scan)")
                        tg_send(chat_id,
                            f"🚫 <b>Banned ({len(b)}):</b>\n" + "\n".join(lines))
                    else:
                        tg_send(chat_id, "✅ Belum ada ban.")
                elif text in ("/resetban","resetban"):
                    with ban_lock: n=len(banned_coins); banned_coins.clear()
                    tg_send(chat_id,f"✅ Ban direset ({n} dihapus).")
                elif text in ("/resetbalance","resetbalance"):
                    with stat_lock:
                        stats["balance"]     = STARTING_BALANCE
                        stats["pnl_history"] = []
                        stats["tp"]          = 0
                        stats["sl"]          = 0
                        stats["total"]       = 0
                    tg_send(chat_id,
                        f"✅ Saldo & statistik direset.\n"
                        f"💵 Modal awal: <b>${STARTING_BALANCE:.2f}</b>")
                elif text in ("/auto","auto"):
                    if auto_mode:
                        tg_send(chat_id,"⚙️ Broadcaster sudah berjalan.")
                    else:
                        auto_mode=True
                        auto_thread=threading.Thread(
                            target=simulation_loop,args=(chat_id,),daemon=True)
                        auto_thread.start()
                elif text in ("/stop","stop"):
                    # /stop hanya mematikan scanning sinyal baru — posisi
                    # yang sudah berjalan tetap dipantau sampai TP/SL alami.
                    if auto_mode:
                        auto_mode = False
                        with positions_lock:
                            n_active = len(positions)
                        tg_send(chat_id,
                            f"⏹ <b>Scanning dihentikan.</b>\n"
                            f"Posisi aktif ({n_active}) tetap dipantau sampai TP/SL.\n"
                            f"Pakai /timeout SYMBOL kalau mau tutup paksa.")
                    else:
                        tg_send(chat_id,"ℹ️ Broadcaster tidak berjalan.")
                elif text in ("/trade","trade"):
                    with positions_lock:
                        pos_list = list(positions.items())
                    if not pos_list:
                        tg_send(chat_id,"ℹ️ Tidak ada posisi aktif.")
                    else:
                        lines = [f"📡 <b>Posisi Aktif ({len(pos_list)}/{MAX_POSITIONS})</b>\n"]
                        for s, p in pos_list:
                            sig    = p["signal"]
                            is_buy = sig["decision"] == "BUY"
                            em     = "🟢" if is_buy else "🔴"
                            status = p.get("status", "active")

                            if status == "pending":
                                pr       = get_price(s) or p["entry"]
                                dist_pct = abs(p["entry"] - pr) / pr * 100
                                lines.append(
                                    f"\n⏳ <b>{s}</b> — PENDING\n"
                                    f"{em} {sig['decision']} | Entry zone: <code>{p['entry']:.6g}</code>\n"
                                    f"Harga kini: <code>{pr:.6g}</code> | Jarak: {dist_pct:.2f}%\n"
                                    f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{sig['sl']:.6g}</code>"
                                )
                            else:
                                pr  = get_price(s) or p["entry"]
                                pnl = (pr - p["entry"]) / p["entry"] * 100 * (1 if is_buy else -1)
                                entry_clock = datetime.fromtimestamp(
                                    p["entry_time"], tz=WIB).strftime("%H:%M")
                                lines.append(
                                    f"\n{em} <b>{s}</b> — AKTIF\n"
                                    f"Entry: <code>{p['entry']:.6g}</code> | Harga: <code>{pr:.6g}</code>\n"
                                    f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{sig['sl']:.6g}</code>\n"
                                    f"PnL: <b>{pnl:+.2f}%</b> | 🕐 Entry jam {entry_clock}"
                                )
                        tg_send(chat_id,"\n".join(lines))
                elif text.startswith("/timeout") or (not text.startswith("/") and text.startswith("timeout")):
                    parts = text.split()
                    target_sym = parts[1].upper() if len(parts) > 1 else None
                    with positions_lock:
                        syms = list(positions.keys())
                    if not syms:
                        tg_send(chat_id,"ℹ️ Tidak ada posisi aktif.")
                    elif target_sym:
                        if target_sym in syms:
                            with positions_lock:
                                if target_sym in positions:
                                    positions[target_sym]["timeout_flag"] = True
                            tg_send(chat_id,f"⏭ Timeout → {target_sym}.")
                        else:
                            tg_send(chat_id,
                                f"❓ {target_sym} tidak ditemukan.\n"
                                f"Aktif: {', '.join(syms)}")
                    else:
                        with positions_lock:
                            for s in syms:
                                if s in positions:
                                    positions[s]["timeout_flag"] = True
                        tg_send(chat_id,f"⏭ Timeout semua ({len(syms)}) posisi.")
                elif text.startswith("/max"):
                    parts = text.split()
                    # ── /max (tampilkan info) ──────────────────────────────
                    if len(parts) == 1:
                        # Estimasi beban API saat ini
                        scan_weight_per_min  = 836   # ~100 kline req × weight5 / ~34s scan
                        price_weight_per_min = 12    # 1 batch ticker/price tiap 10 detik
                        total_weight         = scan_weight_per_min + price_weight_per_min
                        binance_limit        = 2400
                        usage_pct            = total_weight / binance_limit * 100
                        headroom_pct         = 100 - usage_pct
                        threads_now          = 4 + MAX_POSITIONS * 2   # bot+cache+flask+scan + monitor+wait_entry

                        # Batas aman: scan mendominasi, bukan jumlah posisi
                        # Posisi hanya menambah ~0.02 weight/mnt per posisi (SL check jarang)
                        # Batas praktis sebelum scan overload:
                        #   sisa headroom = 1552 weight/mnt, scan = 836/mnt
                        #   bisa ~2 scan paralel tapi kode hanya 1 scan sekaligus → aman tak terbatas dari sisi API
                        # Batas rekomendasi dari sisi KUALITAS SINYAL: ≤ 20
                        tg_send(chat_id,
                            f"⚙️ <b>Max Posisi</b>\n\n"
                            f"Saat ini     : <b>{MAX_POSITIONS} posisi</b>\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📡 <b>Info Beban API (Binance Futures)</b>\n\n"
                            f"Limit Binance    : <b>2.400 weight/mnt</b>\n"
                            f"Scan 50 koin     : ~{scan_weight_per_min} weight/mnt\n"
                            f"Price cache      : ~{price_weight_per_min} weight/mnt (1 batch/10 dtk)\n"
                            f"Total dipakai    : ~{total_weight} weight/mnt "
                            f"(<b>{usage_pct:.0f}%</b> dari limit)\n"
                            f"Headroom tersisa : ~{headroom_pct:.0f}%\n\n"
                            f"⚠️ <b>Penting:</b> MAX_POSITIONS <b>tidak</b> menambah beban\n"
                            f"API secara signifikan. Beban didominasi scan koin,\n"
                            f"bukan jumlah posisi yang dipantau.\n"
                            f"Monitor thread baca harga dari cache lokal — bukan API.\n\n"
                            f"🧵 Thread aktif est. : ~{threads_now}\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 <b>Batas yang Disarankan</b>\n\n"
                            f"API weight  : ✅ aman hingga 50+ posisi\n"
                            f"Thread      : ✅ aman hingga 50+ posisi\n"
                            f"Kualitas sinyal: ⚠️  disarankan ≤ 20\n"
                            f"  (lebih dari itu, scanner makin susah\n"
                            f"  temukan setup berkualitas karena koin\n"
                            f"  terbaik sudah terpakai)\n\n"
                            f"<b>Ubah: /max 5 | /max 10 | /max 15 | /max 20</b>")
                    # ── /max N (ubah nilai) ────────────────────────────────
                    elif len(parts) == 2:
                        try:
                            n = int(parts[1])
                            if n < 1 or n > 50:
                                tg_send(chat_id,
                                    f"❌ Nilai harus antara 1–50.\n"
                                    f"Contoh: /max 10")
                            else:
                                old = MAX_POSITIONS
                                MAX_POSITIONS = n
                                with positions_lock:
                                    n_active = len(positions)
                                note = ""
                                if n < n_active:
                                    note = (f"\n\n⚠️ Ada {n_active} posisi aktif saat ini.\n"
                                            f"Posisi yang sudah buka tetap dipantau.\n"
                                            f"Scan baru berhenti sampai posisi tutup ke ≤ {n}.")
                                tg_send(chat_id,
                                    f"✅ Max posisi diubah: <b>{old} → {MAX_POSITIONS}</b>{note}")
                        except ValueError:
                            tg_send(chat_id,"❌ Format salah. Contoh: /max 10")
                    else:
                        tg_send(chat_id,"❌ Format: /max  atau  /max 10")
                elif text.startswith("/confidence_min"):
                    parts = text.split()
                    # ── /confidence_min (tampilkan nilai saat ini) ─────────
                    if len(parts) == 1:
                        tg_send(chat_id,
                            f"🎯 <b>Confidence Minimum</b>\n\n"
                            f"Saat ini: <b>{MIN_CONFIDENCE}%</b>\n\n"
                            f"Sinyal dengan confidence di bawah angka ini akan\n"
                            f"diabaikan sebelum masuk pertimbangan RR/entry.\n"
                            f"Makin tinggi → sinyal lebih jarang tapi lebih\n"
                            f"selektif. Makin rendah → sinyal lebih sering\n"
                            f"tapi makin banyak setup lemah ikut lolos.\n\n"
                            f"<b>Ubah: /confidence_min 50</b>")
                    # ── /confidence_min N (ubah nilai) ─────────────────────
                    elif len(parts) == 2:
                        try:
                            n = int(parts[1])
                            if n < 0 or n > 99:
                                tg_send(chat_id,
                                    f"❌ Nilai harus antara 0–99.\n"
                                    f"Contoh: /confidence_min 50")
                            else:
                                old = MIN_CONFIDENCE
                                MIN_CONFIDENCE = n
                                tg_send(chat_id,
                                    f"✅ Confidence minimum diubah: "
                                    f"<b>{old}% → {MIN_CONFIDENCE}%</b>")
                        except ValueError:
                            tg_send(chat_id,"❌ Format salah. Contoh: /confidence_min 50")
                    else:
                        tg_send(chat_id,"❌ Format: /confidence_min  atau  /confidence_min 50")
                else:
                    tg_send(chat_id,"❓ Tidak dikenal. /start")

            time.sleep(1)
        except Exception as e:
            log.error(f"[bot] {e}")
            time.sleep(5)


if __name__=="__main__":
    threading.Thread(target=_price_cache_loop, daemon=True).start()
    threading.Thread(target=bot_loop, daemon=True).start()
    run_flask()
