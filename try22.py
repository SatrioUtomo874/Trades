#!/usr/bin/env python3
"""
SMC Simulasi Trading Bot v5 — Counter Entry Strategy
Logika: Analisis normal → temukan sinyal → BALIK arah → analisis ulang TP/SL
Render.com | python main.py
"""

# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
ALLOWED_USER_ID = 8041197505
MAX_PRICE       = 80.0
TOP_N_COINS     = 50
MIN_RR              = 2.0
MONITOR_SLEEP       = 2
AUTO_TIMEOUT_HOURS  = 5      # otomatis timeout setelah 5 jam
# ─────────────────────────────────────────────

import os, time, logging, threading
from datetime import datetime

import requests, pandas as pd, numpy as np, urllib3
from flask import Flask

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

auto_mode      = False
auto_thread    = None
active_chat_id = None
timeout_flag   = False

STARTING_BALANCE = 10.0   # modal awal simulasi dalam USD

stat_lock = threading.Lock()
stats = {
    "tp":0, "sl":0, "timeout":0, "total":0,
    "balance"    : STARTING_BALANCE,
    "pnl_history": [],   # list dict {result, pnl_usd, pct, balance_after}
}

ban_lock = threading.Lock()
banned_coins: dict = {}   # {symbol: trade_count_saat_ban}
UNBAN_AFTER_TRADES = 5    # otomatis unban setelah N trade selesai

FAPI = "https://fapi.binance.com"

# ── Flask ─────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    with stat_lock:
        t=stats["total"]; tp=stats["tp"]; sl=stats["sl"]
    wr=f"{tp/(tp+sl)*100:.1f}%" if (tp+sl)>0 else "–"
    return (f"<h3>SMC Counter Bot v5</h3>"
            f"<p>Auto:{auto_mode} | Banned:{len(banned_coins)}</p>"
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
def fapi_get(path, params=None):
    """Binance Futures GET — tetap dipakai utama."""
    for i in range(3):
        try:
            r = requests.get(f"{FAPI}{path}", params=params,
                             timeout=10, verify=False)
            d = r.json()
            if isinstance(d, dict) and "code" in d:
                raise ValueError(f"Binance {d['code']}: {d.get('msg')}")
            return d
        except Exception as e:
            log.warning(f"[binance] {i+1}/3: {e}")
            time.sleep(2)
    raise ConnectionError(f"Binance gagal: {path}")

def _binance_klines(symbol, interval, limit):
    raw = fapi_get("/fapi/v1/klines",
                   {"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(raw, list) or len(raw) < 40:
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

def _binance_top_coins(cur_ban):
    tickers = fapi_get("/fapi/v1/ticker/24hr")
    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t["quoteVolume"]) > 100_000
        and t["symbol"] not in cur_ban
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
    if not rows or len(rows) < 40:
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

def _bybit_top_coins(cur_ban):
    d = _raw_get(f"{BYBIT}/v5/market/tickers", {"category":"linear"})
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit tickers error: {d.get('retMsg')}")
    items = d["result"]["list"]
    usdt = [
        t for t in items
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t.get("turnover24h","0")) > 100_000
        and t["symbol"] not in cur_ban
    ]
    usdt.sort(key=lambda x: float(x.get("turnover24h","0")), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]


# ── PUBLIC API — digunakan oleh seluruh program ──
def get_price(symbol):
    """Ambil harga real-time. Binance dulu, fallback Bybit."""
    # Binance
    for _ in range(2):
        try:
            return _binance_price(symbol)
        except Exception as e:
            log.warning(f"[price/binance] {symbol}: {e}")
            time.sleep(1)
    # Bybit fallback
    for _ in range(2):
        try:
            p = _bybit_price(symbol)
            log.info(f"[price/bybit fallback] {symbol}: {p}")
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

def get_real_trade_count():
    """Hitung trade nyata: tp + sl + timeout saja, tidak termasuk no_entry."""
    with stat_lock:
        return stats["tp"] + stats["sl"] + stats["timeout"]

def get_top_coins():
    """Ambil top coins. Binance dulu, fallback Bybit.
    Auto-unban koin yang sudah melewati UNBAN_AFTER_TRADES trade nyata.
    """
    with ban_lock:
        real_trades = get_real_trade_count()
        # Auto-unban koin yang sudah UNBAN_AFTER_TRADES trade nyata sejak diban
        to_unban = [sym for sym, ban_at in banned_coins.items()
                    if real_trades - ban_at >= UNBAN_AFTER_TRADES]
        for sym in to_unban:
            del banned_coins[sym]
            log.info(f"[unban] {sym} kembali aktif setelah {UNBAN_AFTER_TRADES} trade")
        cur_ban = set(banned_coins.keys())
    # Binance
    try:
        coins = _binance_top_coins(cur_ban)
        if coins:
            return coins
        log.warning("[top_coins/binance] kosong, coba Bybit...")
    except Exception as e:
        log.warning(f"[top_coins/binance] {e} — coba Bybit...")
    # Bybit fallback
    try:
        coins = _bybit_top_coins(cur_ban)
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

    # SMC M15
    sh15,sl15=swing_pts(m15,5)
    bos=detect_bos(m15,sh15,sl15)
    if bos["bb"]: bull+=15
    if bos["cb"]: bull+=10
    if bos["bs"]: bear+=15
    if bos["cs"]: bear+=10

    # Candle pattern
    body=L15["close"]-L15["open"]
    low_wick=min(L15["open"],L15["close"])-L15["low"]
    up_wick=L15["high"]-max(L15["open"],L15["close"])
    if low_wick>abs(body)*1.5: bull+=8  # hammer
    if up_wick>abs(body)*1.5:  bear+=8  # shooting star

    direction="bull" if bull>=bear else "bear"
    raw=bull if direction=="bull" else bear
    conf=min(int(raw/160*100),99)
    atr_val=max(L15["atr"], L15["close"]*0.003)

    return {
        "direction"  : direction,
        "confidence" : conf,
        "price"      : L15["close"],
        "atr"        : atr_val,
        "struct_h1"  : struct_h1,
        "rsi"        : round(rv,1),
        "bull_pts"   : bull,
        "bear_pts"   : bear,
    }


# ═════════════════════════════════════════════
# TAHAP 2: ANALISIS ULANG — SL DULU, LALU TP
# ═════════════════════════════════════════════
def analyze_counter_setup(df_h1, df_m15, counter_dir, entry_price):
    """
    Urutan WAJIB:
    1. Tentukan SL dulu berdasarkan LIQUIDITY POOL (equal highs/lows)
       atau OB/supply-demand zone — bukan wick biasa yang mudah tersapu.
       SL harus di level di mana jika harga sampai ke sana,
       analisis counter sudah TERBUKTI SALAH.

    2. Hitung risk = abs(entry - SL)

    3. Iterasi TP candidates dari terdekat ke terjauh.
       Ambil level pertama yang menghasilkan RR >= 2.0
       Jika TP terdekat RR < 2.0 → lewati, cari level berikutnya.
       TP tidak boleh asal, harus level nyata dari chart.

    SL SELL counter: di atas equal highs / OB top / swing high — BUKAN wick 3 candle
    SL BUY  counter: di bawah equal lows / demand bot / swing low
    """
    h1  = build_df(df_h1)
    m15 = build_df(df_m15)
    if h1 is None or m15 is None: return None

    L15 = m15.iloc[-1]
    atr = max(L15["atr"], entry_price * 0.002)

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

    sl_price = None
    sl_label = ""
    reasons  = []

    # ══════════════════════════════════════════════════════════════
    # SELL — sinyal bear, kita SELL
    # SL di atas liquidity pool / supply zone terdekat di atas entry
    # TP di bawah entry (demand zone / equal lows / swing low)
    # ══════════════════════════════════════════════════════════════
    if counter_dir == "bear":

        # Kumpulkan kandidat SL — level di atas entry
        sl_pool = []

        # Prioritas 1: Equal highs M15 — ini zona liquidity sweep paling sering
        # SL di ATAS equal highs agar tidak kena sweep
        for eh in sorted(eq_highs):
            if eh > entry_price + atr * 0.3:
                sl_pool.append(("eq_high_m15", eh + atr * 0.25))
                break  # ambil yang paling dekat

        # Prioritas 2: Supply zone top M15 — level institusi jual
        for z in supply_zones:
            if z["top"] > entry_price + atr * 0.2:
                sl_pool.append(("supply_top_m15", z["top"] + atr * 0.2))

        # Prioritas 3: Swing high M15 paling dekat di atas entry
        sh_above = sorted([m15["high"].iloc[i] for i in sh15
                           if m15["high"].iloc[i] > entry_price + atr * 0.2])
        if sh_above:
            sl_pool.append(("swing_h_m15", sh_above[0] + atr * 0.2))

        # Prioritas 4: Equal highs H1 (level lebih kuat)
        for eh in sorted(eq_highs_h1):
            if eh > entry_price + atr * 0.5:
                sl_pool.append(("eq_high_h1", eh + atr * 0.3))
                break

        # Prioritas 5: Supply zone H1
        for z in supply_h1:
            if z["top"] > entry_price + atr * 0.5:
                sl_pool.append(("supply_top_h1", z["top"] + atr * 0.25))

        # Pilih SL terdekat di atas entry dari pool
        sl_pool_valid = [(lbl, v) for lbl, v in sl_pool
                         if v > entry_price + atr * 0.15]
        if not sl_pool_valid:
            # Fallback: ATR-based jika tidak ada level struktural
            sl_price = entry_price + atr * 2.0
            sl_label = "atr_fallback"
        else:
            sl_label, sl_price = min(sl_pool_valid, key=lambda x: x[1])

        risk = abs(sl_price - entry_price)
        # SL minimum = 1.2 ATR agar tidak mudah tersapu noise intraday
        if risk < atr * 1.2:
            sl_price = entry_price + atr * 2.0
            risk = atr * 2.0
            sl_label += "_expanded"

        reasons.append(f"SL@{sl_price:.5g}({sl_label})")
        min_reward = risk * MIN_RR

        # Kumpulkan semua TP candidates di bawah entry, urutkan terdekat ke terjauh
        tp_pool = []

        # Equal lows M15 (liquidity target — sering menjadi tujuan harga)
        for el in sorted(eq_lows, reverse=True):
            if el < entry_price - atr * 0.5:
                tp_pool.append(("eq_low_m15", el))

        # Demand zone top M15 (harga sering berhenti di atas demand)
        for z in sorted(demand_zones, key=lambda x: x["top"], reverse=True):
            if z["top"] < entry_price - atr * 0.5:
                tp_pool.append(("demand_top_m15", z["top"]))

        # FVG bear mid M15
        for fvg in sorted(fvg_bear, key=lambda x: x["mid"], reverse=True):
            if fvg["mid"] < entry_price - atr * 0.5:
                tp_pool.append(("fvg_bear_m15", fvg["mid"]))

        # Swing low M15
        for v in sorted([m15["low"].iloc[i] for i in sl15], reverse=True):
            if v < entry_price - atr * 0.5:
                tp_pool.append(("sw_low_m15", v))

        # Equal lows H1 (target lebih jauh)
        for el in sorted(eq_lows_h1, reverse=True):
            if el < entry_price - atr * 1.0:
                tp_pool.append(("eq_low_h1", el))

        # Demand zone H1
        for z in sorted(demand_h1, key=lambda x: x["top"], reverse=True):
            if z["top"] < entry_price - atr * 1.0:
                tp_pool.append(("demand_top_h1", z["top"]))

        # Swing low H1
        for v in sorted([h1["low"].iloc[i] for i in sl1], reverse=True):
            if v < entry_price - atr * 1.0:
                tp_pool.append(("sw_low_h1", v))

        # Iterasi dari terdekat ke terjauh — ambil yang pertama RR >= MIN_RR
        tp_price = None
        tp_label = ""
        for lbl, v in sorted(tp_pool, key=lambda x: x[1], reverse=True):
            rr_candidate = abs(entry_price - v) / risk
            if rr_candidate >= MIN_RR:
                tp_price = v
                tp_label = lbl
                break

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
        for el in sorted(eq_lows, reverse=True):
            if el < entry_price - atr * 0.3:
                sl_pool.append(("eq_low_m15", el - atr * 0.25))
                break

        # Prioritas 2: Demand zone bot M15
        for z in demand_zones:
            if z["bot"] < entry_price - atr * 0.2:
                sl_pool.append(("demand_bot_m15", z["bot"] - atr * 0.2))

        # Prioritas 3: Swing low M15 paling dekat di bawah
        sl_below = sorted([m15["low"].iloc[i] for i in sl15
                           if m15["low"].iloc[i] < entry_price - atr * 0.2],
                          reverse=True)
        if sl_below:
            sl_pool.append(("swing_l_m15", sl_below[0] - atr * 0.2))

        # Prioritas 4: Equal lows H1
        for el in sorted(eq_lows_h1, reverse=True):
            if el < entry_price - atr * 0.5:
                sl_pool.append(("eq_low_h1", el - atr * 0.3))
                break

        # Prioritas 5: Demand zone H1
        for z in demand_h1:
            if z["bot"] < entry_price - atr * 0.5:
                sl_pool.append(("demand_bot_h1", z["bot"] - atr * 0.25))

        sl_pool_valid = [(lbl, v) for lbl, v in sl_pool
                         if v < entry_price - atr * 0.15]
        if not sl_pool_valid:
            sl_price = entry_price - atr * 2.0
            sl_label = "atr_fallback"
        else:
            sl_label, sl_price = max(sl_pool_valid, key=lambda x: x[1])

        risk = abs(entry_price - sl_price)
        # SL minimum = 1.2 ATR agar tidak mudah tersapu noise intraday
        if risk < atr * 1.2:
            sl_price = entry_price - atr * 2.0
            risk = atr * 2.0
            sl_label += "_expanded"

        reasons.append(f"SL@{sl_price:.5g}({sl_label})")
        min_reward = risk * MIN_RR

        # TP candidates di atas entry
        tp_pool = []

        # Equal highs M15
        for eh in sorted(eq_highs):
            if eh > entry_price + atr * 0.5:
                tp_pool.append(("eq_high_m15", eh))

        # Supply zone bot M15
        for z in sorted(supply_zones, key=lambda x: x["bot"]):
            if z["bot"] > entry_price + atr * 0.5:
                tp_pool.append(("supply_bot_m15", z["bot"]))

        # FVG bull mid M15
        for fvg in sorted(fvg_bull, key=lambda x: x["mid"]):
            if fvg["mid"] > entry_price + atr * 0.5:
                tp_pool.append(("fvg_bull_m15", fvg["mid"]))

        # Swing high M15
        for v in sorted([m15["high"].iloc[i] for i in sh15]):
            if v > entry_price + atr * 0.5:
                tp_pool.append(("sw_high_m15", v))

        # Equal highs H1
        for eh in sorted(eq_highs_h1):
            if eh > entry_price + atr * 1.0:
                tp_pool.append(("eq_high_h1", eh))

        # Supply zone H1
        for z in sorted(supply_h1, key=lambda x: x["bot"]):
            if z["bot"] > entry_price + atr * 1.0:
                tp_pool.append(("supply_bot_h1", z["bot"]))

        # Swing high H1
        for v in sorted([h1["high"].iloc[i] for i in sh1]):
            if v > entry_price + atr * 1.0:
                tp_pool.append(("sw_high_h1", v))

        # Iterasi dari terdekat ke terjauh — ambil yang pertama RR >= MIN_RR
        tp_price = None
        tp_label = ""
        for lbl, v in sorted(tp_pool, key=lambda x: x[1]):
            rr_candidate = abs(v - entry_price) / risk
            if rr_candidate >= MIN_RR:
                tp_price = v
                tp_label = lbl
                break

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


# ═════════════════════════════════════════════
# PIPELINE ANALISIS LENGKAP
# ═════════════════════════════════════════════
def full_analyze(symbol):
    """
    1. Ambil data H1 + M15
    2. Score arah sinyal normal
    3. Gunakan arah sinyal SEARAH (BUY jika bull, SELL jika bear)
    4. analyze_counter_setup menerima "bear" untuk SELL, "bull" untuk BUY
       — nama fungsi dipertahankan, tapi sekarang dipakai searah sinyal
    """
    try:
        df_h1  = get_klines(symbol, "1h",  250)
        df_m15 = get_klines(symbol, "15m", 250)
        if df_h1.empty or df_m15.empty: return None

        # Tahap 1: scoring arah
        score = score_direction(df_h1, df_m15)
        if score is None: return None

        original_dir = score["direction"]          # "bull" atau "bear"
        entry_price  = score["price"]

        # BUY jika sinyal bull, SELL jika sinyal bear
        decision  = "BUY"  if original_dir == "bull" else "SELL"
        # analyze_counter_setup: blok "bear" = logika SELL, blok "bull" = logika BUY
        setup_dir = original_dir   # "bull" → blok BUY, "bear" → blok SELL

        # Tahap 2: analisis TP/SL
        setup = analyze_counter_setup(df_h1, df_m15, setup_dir, entry_price)
        if setup is None: return None

        return {
            "symbol"       : symbol,
            "original_dir" : original_dir,
            "decision"     : decision,
            "confidence"   : score["confidence"],
            "price"        : entry_price,
            "entry"        : entry_price,
            "sl"           : setup["sl"],
            "tp"           : setup["tp"],
            "rr"           : setup["rr"],
            "rsi"          : score["rsi"],
            "struct_h1"    : score["struct_h1"],
            "tp_sl_reason" : setup["reason"],
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
        tg_send(chat_id,"⚠️ Semua koin diban. /resetban untuk reset.")
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

    # Filter: hanya koin dengan confidence >= 45%
    results = [r for r in results if r["confidence"] >= 45]
    if not results:
        tg_send(chat_id,"⚠️ Tidak ada koin dengan confidence cukup (≥45%). Retry...")
        return None

    # Ranking: confidence DESC → rr DESC
    results.sort(key=lambda x:(x["confidence"],x["rr"]),reverse=True)
    best=results[0]
    log.info(f"Best: {best['symbol']} {best['decision']} "
             f"conf={best['confidence']}% RR=1:{best['rr']}")
    return best


# ═════════════════════════════════════════════
# MONITORING
# ═════════════════════════════════════════════
def em_dir(decision): return "🟢" if decision=="BUY" else "🔴"

def monitor_trade(chat_id, signal):
    global timeout_flag
    sym=signal["symbol"]
    is_buy=signal["decision"]=="BUY"
    entry=signal["entry"]
    sl_p=signal["sl"]
    tp_p=signal["tp"]

    actual = get_price(sym) or entry

    risk   = abs(actual - sl_p)
    reward = abs(tp_p - actual)
    rr_now = round(reward / risk, 2) if risk > 0 else 0

    dir_label = "BULLISH" if signal["original_dir"] == "bull" else "BEARISH"

    tg_send(chat_id,
        f"⚡ <b>ENTRY — {sym}</b>\n\n"
        f"Arah      : {em_dir(signal['decision'])} <b>{signal['decision']}</b>\n"
        f"Analisis  : <b>{dir_label}</b>\n"
        f"Entry     : <code>{actual:.6g}</code>\n"
        f"✅ TP     : <code>{tp_p:.6g}</code>\n"
        f"🛑 SL     : <code>{sl_p:.6g}</code>\n"
        f"⚖️ RR     : 1:{rr_now}\n"
        f"📝 Basis  : {signal['tp_sl_reason']}\n\n"
        f"📡 Monitor tiap {MONITOR_SLEEP}s... /timeout untuk skip.")

    last_log      = time.time()
    log_interval  = 90
    entry_time    = time.time()
    auto_timeout_sec = AUTO_TIMEOUT_HOURS * 3600

    # fase: "normal" → setelah 5 jam berubah ke "wait_profit"
    phase           = "normal"
    notified_timeout = False   # pastikan notif timeout hanya dikirim sekali

    while True:
        # ── Manual timeout via /timeout ──────────────────────────────
        if timeout_flag:
            timeout_flag = False
            price = get_price(sym) or 0
            tg_send(chat_id,
                f"⏭ <b>Timeout Manual</b> — {sym}\n"
                f"Harga: <code>{price:.6g}</code>")
            signal["timeout_close_price"] = None  # timeout manual, tidak ada profit
            return "timeout"

        price = get_price(sym)
        if price is None:
            time.sleep(MONITOR_SLEEP); continue

        elapsed = time.time() - entry_time
        is_profit = (price > actual) if is_buy else (price < actual)

        # ── Cek apakah sudah melewati batas waktu auto-timeout ───────
        if phase == "normal" and elapsed >= auto_timeout_sec:
            phase = "wait_profit"
            if not notified_timeout:
                notified_timeout = True
                pnl_pct = (price - actual) / actual * 100 * (1 if is_buy else -1)
                pnl_sign = "+" if pnl_pct >= 0 else ""
                tg_send(chat_id,
                    f"⏰ <b>Auto-Timeout {AUTO_TIMEOUT_HOURS}j — {sym}</b>\n\n"
                    f"Posisi  : {em_dir(signal['decision'])} {signal['decision']}\n"
                    f"Entry   : <code>{actual:.6g}</code>\n"
                    f"Harga   : <code>{price:.6g}</code>\n"
                    f"PnL saat ini: <b>{pnl_sign}{pnl_pct:.3f}%</b>\n\n"
                    f"{'✅ PnL positif — menutup posisi sekarang...' if is_profit else '⏳ PnL masih negatif — menahan posisi, tunggu harga kembali profit...'}")

        # ── Fase wait_profit: tutup segera saat PnL pertama kali + ──
        if phase == "wait_profit":
            if is_profit:
                close_pct = (price - actual) / actual * 100 * (1 if is_buy else -1)
                # Hitung pnl_usd pakai harga close aktual (bukan tp_p)
                # update_stats akan dipanggil dari simulation_loop, kita return
                # "timeout" tapi sertakan close_price agar update_stats bisa hitung
                tg_send(chat_id,
                    f"⏱ <b>TIMEOUT — Posisi Ditutup (Profit)</b> — {sym} 💰\n\n"
                    f"Entry   : <code>{actual:.6g}</code>\n"
                    f"Tutup   : <code>{price:.6g}</code>\n"
                    f"Profit  : <b>+{close_pct:.3f}%</b>\n\n"
                    f"⏰ Ditutup otomatis setelah {AUTO_TIMEOUT_HOURS} jam\n"
                    f"karena harga kembali ke posisi profit.")
                # Simpan harga penutupan ke signal agar simulation_loop bisa pakai
                signal["timeout_close_price"] = price
                return "timeout"

            # Belum profit → cek SL masih berlaku
            hit_sl = (price <= sl_p) if is_buy else (price >= sl_p)
            if hit_sl:
                pct = abs(price - actual) / actual * 100
                tg_send(chat_id,
                    f"🛑 <b>STOP LOSS — {sym}</b>\n"
                    f"Harga: <code>{price:.6g}</code>\n"
                    f"SL   : <code>{sl_p:.6g}</code>\n"
                    f"Loss : -{pct:.2f}%\n"
                    f"<i>(Timeout aktif, tapi SL tetap dihormati)</i>")
                return "sl"

            # Update log tiap 90 detik selama menunggu profit
            if time.time() - last_log >= log_interval:
                pnl_pct = (price - actual) / actual * 100 * (1 if is_buy else -1)
                tg_send(chat_id,
                    f"⏳ <b>Menunggu Profit — {sym}</b>\n"
                    f"Harga : <code>{price:.6g}</code>\n"
                    f"Entry : <code>{actual:.6g}</code>\n"
                    f"PnL   : <b>{pnl_pct:+.3f}%</b>\n"
                    f"SL    : <code>{sl_p:.6g}</code> (masih aktif)")
                last_log = time.time()

            time.sleep(MONITOR_SLEEP)
            continue

        # ── Fase normal: cek TP / SL biasa ───────────────────────────
        hit_tp = (price >= tp_p) if is_buy else (price <= tp_p)
        hit_sl = (price <= sl_p) if is_buy else (price >= sl_p)

        if hit_tp:
            pct = abs(price - actual) / actual * 100
            tg_send(chat_id,
                f"🎯 <b>TAKE PROFIT — {sym}</b> 🎉\n"
                f"Harga: <code>{price:.6g}</code>\n"
                f"TP   : <code>{tp_p:.6g}</code>\n"
                f"Profit: +{pct:.2f}%")
            return "tp"

        if hit_sl:
            pct = abs(price - actual) / actual * 100
            tg_send(chat_id,
                f"🛑 <b>STOP LOSS — {sym}</b>\n"
                f"Harga: <code>{price:.6g}</code>\n"
                f"SL   : <code>{sl_p:.6g}</code>\n"
                f"Loss : -{pct:.2f}%")
            return "sl"

        # Update log berkala fase normal
        if time.time() - last_log >= log_interval:
            pct_tp = abs(tp_p - price) / abs(tp_p - actual) * 100 if abs(tp_p - actual) > 0 else 0
            sisa_jam = max(0, (auto_timeout_sec - elapsed) / 3600)
            tg_send(chat_id,
                f"📊 <b>Update {sym}</b>\n"
                f"Harga : <code>{price:.6g}</code>\n"
                f"TP    : <code>{tp_p:.6g}</code> (sisa {pct_tp:.1f}%)\n"
                f"SL    : <code>{sl_p:.6g}</code>\n"
                f"⏰ Auto-timeout dalam {sisa_jam:.1f} jam")
            last_log = time.time()

        time.sleep(MONITOR_SLEEP)


# ═════════════════════════════════════════════
# STATISTIK + BALANCE
# ═════════════════════════════════════════════
RISK_PER_TRADE_PCT = 2.0   # risiko maksimal per trade = 2% dari saldo

def update_stats(result, entry=None, sl_p=None, tp_p=None, timeout_profit=False):
    """
    Hitung P&L dalam USD berdasarkan fixed risk per trade.
    Risk per trade = RISK_PER_TRADE_PCT % dari saldo saat ini.

    - tp      → profit  = risk × (tp_dist / sl_dist)
    - sl      → loss    = -risk
    - timeout + timeout_profit=True → profit = risk × (close_dist / sl_dist)
                                      tp_p di sini = harga tutup aktual
    - timeout biasa (manual /timeout) → tidak ada perubahan balance
    """
    with stat_lock:
        stats["total"] += 1
        if result in ("tp", "sl", "timeout"):
            stats[result] += 1

        if result == "tp" and entry and sl_p and tp_p:
            balance  = stats["balance"]
            risk_usd = round(balance * RISK_PER_TRADE_PCT / 100, 4)
            sl_dist  = abs(entry - sl_p)
            tp_dist  = abs(tp_p  - entry)
            rr_actual = tp_dist / sl_dist if sl_dist > 0 else MIN_RR
            pnl_usd  = round(risk_usd * rr_actual, 4)
            pct      = round(tp_dist / entry * 100, 3)
            stats["balance"] = round(balance + pnl_usd, 4)
            stats["pnl_history"].append({
                "result"       : "tp",
                "pct"          : pct,
                "pnl_usd"      : pnl_usd,
                "balance_after": stats["balance"],
            })

        elif result == "sl" and entry and sl_p:
            balance  = stats["balance"]
            risk_usd = round(balance * RISK_PER_TRADE_PCT / 100, 4)
            sl_dist  = abs(entry - sl_p)
            pnl_usd  = -risk_usd
            pct      = -round(sl_dist / entry * 100, 3)
            stats["balance"] = round(balance + pnl_usd, 4)
            stats["pnl_history"].append({
                "result"       : "sl",
                "pct"          : pct,
                "pnl_usd"      : pnl_usd,
                "balance_after": stats["balance"],
            })

        elif result == "timeout" and timeout_profit and entry and sl_p and tp_p:
            # tp_p di sini = harga close aktual (bukan TP asli)
            balance   = stats["balance"]
            risk_usd  = round(balance * RISK_PER_TRADE_PCT / 100, 4)
            sl_dist   = abs(entry - sl_p)
            close_dist = abs(tp_p - entry)   # jarak entry → close price
            rr_actual  = close_dist / sl_dist if sl_dist > 0 else 0.1
            pnl_usd   = round(risk_usd * rr_actual, 4)
            pct       = round(close_dist / entry * 100, 3)
            stats["balance"] = round(balance + pnl_usd, 4)
            stats["pnl_history"].append({
                "result"       : "timeout",   # tetap timeout di history
                "pct"          : pct,         # profit kecil tapi positif
                "pnl_usd"      : pnl_usd,
                "balance_after": stats["balance"],
            })
        # timeout manual → tidak ada entri pnl_history, balance tidak berubah

def fmt_stats():
    with stat_lock:
        t   = stats["total"]
        tp  = stats["tp"]
        sl  = stats["sl"]
        to  = stats["timeout"]
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
        if h["result"] == "tp":
            em = "✅"
        elif h["result"] == "sl":
            em = "❌"
        elif h["result"] == "timeout":
            em = "⏱✅" if h["pnl_usd"] > 0 else "⏱"
        else:
            em = "➖"
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
        f"⏱ Timeout   : {to} ({to/t*100:.1f}%)\n"
        f"📈 Win Rate  : <b>{wr:.1f}%</b> (dari {tp+sl} trade)\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 <b>Modal Awal : ${STARTING_BALANCE:.2f}</b>\n"
        f"💰 <b>Saldo Kini : ${bal:.4f}</b>\n"
        f"{pnl_em} <b>Total P&L  : {pnl_sgn}${pnl_tot:.4f} "
        f"({pnl_sgn}{pnl_pct:.2f}%)</b>\n"
        f"⚠️ Risk/trade : {RISK_PER_TRADE_PCT}% saldo\n\n"
        f"📋 5 Trade Terakhir:\n{hist_str}\n\n"
        f"🚫 Banned    : {len(banned_coins)}"
    )

def fmt_signal_msg(sig):
    em  = "🟢" if sig["decision"]=="BUY" else "🔴"
    bar = "█"*(sig["confidence"]//10)+"░"*(10-sig["confidence"]//10)
    dir_label = "BULLISH" if sig["original_dir"]=="bull" else "BEARISH"
    return (
        f"📡 <b>SINYAL DITEMUKAN</b>\n\n"
        f"Koin       : <b>{sig['symbol']}</b>\n"
        f"Analisis   : <b>{dir_label}</b> (confidence {sig['confidence']}% {bar})\n"
        f"Eksekusi   : {em} <b>{sig['decision']}</b>\n\n"
        f"💰 Entry   : <code>{sig['entry']:.6g}</code>\n"
        f"✅ TP      : <code>{sig['tp']:.6g}</code>\n"
        f"🛑 SL      : <code>{sig['sl']:.6g}</code>\n"
        f"⚖️ RR      : <b>1:{sig['rr']}</b>\n"
        f"RSI        : {sig['rsi']} | H1: {sig['struct_h1'].upper()}\n\n"
        f"📝 Basis TP/SL:\n{sig['tp_sl_reason']}"
    )


# ═════════════════════════════════════════════
# SIMULATION LOOP
# ═════════════════════════════════════════════
def simulation_loop(chat_id):
    global auto_mode, timeout_flag
    tg_send(chat_id,
        "🤖 <b>Simulasi Trading dimulai!</b>\n\n"
        "Alur:\n"
        "1. Scan & scoring 50 koin\n"
        "2. Ambil sinyal terkuat (BUY/SELL)\n"
        "3. Tentukan SL dari struktur chart\n"
        "4. Iterasi TP hingga RR ≥ 1:2\n"
        "5. Monitor hingga TP / SL\n\n"
        "/stop untuk berhenti | /timeout untuk skip")

    while auto_mode:
        timeout_flag=False

        signal=run_scan_once(chat_id)
        if not auto_mode: break

        if signal is None:
            tg_send(chat_id,"⚠️ Tidak ada setup. Retry 5 detik...")
            for _ in range(5):
                if not auto_mode: break
                time.sleep(1)
            continue

        tg_send(chat_id, fmt_signal_msg(signal))

        sym    = signal["symbol"]
        result = monitor_trade(chat_id, signal)

        # Ban koin setelah trade selesai (tp, sl, timeout — semua di-ban)
        with ban_lock:
            banned_coins[sym] = get_real_trade_count()

        # Timeout otomatis → pakai close price aktual (bukan tp_p)
        timeout_close = signal.pop("timeout_close_price", None)
        if result == "timeout" and timeout_close is not None:
            update_stats("timeout",
                         entry=signal.get("entry"),
                         sl_p=signal.get("sl"),
                         tp_p=timeout_close,
                         timeout_profit=True)
        else:
            update_stats(result,
                         entry=signal.get("entry"),
                         sl_p=signal.get("sl"),
                         tp_p=signal.get("tp"))

        emoji = {"tp":"🎯","sl":"🛑","timeout":"⏱"}.get(result,"❓")
        label = {"tp":"TAKE PROFIT","sl":"STOP LOSS","timeout":"TIMEOUT"}.get(result,result.upper())
        tg_send(chat_id, f"{emoji} <b>{label}</b> — {sym}\n\n"+fmt_stats())

        if not auto_mode: break
        # langsung lanjut scan tanpa jeda

    tg_send(chat_id,"⏹ <b>Simulasi dihentikan.</b>\n\n"+fmt_stats())


# ═════════════════════════════════════════════
# PESAN STATIS
# ═════════════════════════════════════════════
GREETING=(
    "👋 <b>SMC Signal Trading Bot</b>\n\n"
    "Scan koin → analisis sinyal → entry searah → TP/SL via SNR+SND+SMC\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "/start        — Menu ini\n"
    "/auto         — Mulai simulasi otomatis\n"
    "/stop         — Hentikan simulasi\n"
    "/timeout      — Skip monitoring, lanjut scan\n"
    "/stats        — Statistik + saldo\n"
    "/banned       — Daftar koin ban\n"
    "/resetban     — Hapus semua ban\n"
    "/resetbalance — Reset saldo ke $10\n"
    "/info         — Detail metode analisis\n"
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
        "<b>Tahap 3 — Iterasi TP:</b>\n"
        "Dari level terdekat ke terjauh, ambil TP pertama\n"
        "yang menghasilkan RR ≥ 1:2\n"
        "Level TP: eq highs/lows, supply/demand, FVG, swing H1\n\n"
        f"Min RR: 1:{MIN_RR} | Min Confidence: 45%\n"
        f"TF: H1 (bias) + M15 (entry)\n"
        f"Risk per trade: {RISK_PER_TRADE_PCT}% dari saldo\n"
        f"Modal simulasi: ${STARTING_BALANCE:.2f}"
    )


# ═════════════════════════════════════════════
# BOT LOOP
# ═════════════════════════════════════════════
def bot_loop():
    global auto_mode, auto_thread, active_chat_id, timeout_flag

    log.info("Test koneksi Binance...")
    for i in range(10):
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
                        real_trades = get_real_trade_count()
                        b = sorted(banned_coins.items())
                    if b:
                        lines = []
                        for sym, ban_at in b:
                            remaining = max(0, UNBAN_AFTER_TRADES - (real_trades - ban_at))
                            lines.append(f"• {sym} (unban dalam {remaining} trade)")
                        tg_send(chat_id,
                            f"🚫 <b>Banned ({len(b)}):</b>\n" + "\n".join(lines))
                    else:
                        tg_send(chat_id, "✅ Belum ada ban.")
                elif text in ("/resetban","resetban"):
                    with ban_lock: n=len(banned_coins); banned_coins.clear()
                    tg_send(chat_id,f"✅ Ban direset ({n} dihapus).")
                elif text in ("/resetbalance","resetbalance"):
                    with stat_lock:
                        stats["balance"]      = STARTING_BALANCE
                        stats["pnl_history"]  = []
                        stats["tp"]           = 0
                        stats["sl"]           = 0
                        stats["timeout"]      = 0
                        stats["total"]        = 0
                    tg_send(chat_id,
                        f"✅ Saldo & statistik direset.\n"
                        f"💵 Modal awal: <b>${STARTING_BALANCE:.2f}</b>")
                elif text in ("/auto","auto"):
                    if auto_mode:
                        tg_send(chat_id,"⚙️ Simulasi sudah berjalan.")
                    else:
                        auto_mode=True
                        auto_thread=threading.Thread(
                            target=simulation_loop,args=(chat_id,),daemon=True)
                        auto_thread.start()
                elif text in ("/stop","stop"):
                    if auto_mode:
                        auto_mode=False; timeout_flag=True
                        tg_send(chat_id,"⏹ Menghentikan...")
                    else:
                        tg_send(chat_id,"ℹ️ Simulasi tidak berjalan.")
                elif text in ("/timeout","timeout"):
                    if auto_mode:
                        timeout_flag=True
                        tg_send(chat_id,"⏭ Timeout — monitoring dilewati.")
                    else:
                        tg_send(chat_id,"ℹ️ Tidak ada monitoring aktif.")
                else:
                    tg_send(chat_id,"❓ Tidak dikenal. /start")

            time.sleep(1)
        except Exception as e:
            log.error(f"[bot] {e}")
            time.sleep(5)


if __name__=="__main__":
    threading.Thread(target=bot_loop, daemon=True).start()
    run_flask()
