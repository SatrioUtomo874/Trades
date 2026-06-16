#!/usr/bin/env python3
"""
SMC Signal Bot — Simulasi Trading
Render.com Web Service | python main.py
"""

# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
ALLOWED_USER_ID = 8041197505
MAX_PRICE       = 80.0
TOP_N_COINS     = 50
LOOP_INTERVAL   = 300   # detik antar scan
MIN_RR          = 2.0
MONITOR_SLEEP   = 2     # detik antar cek harga saat monitoring
ENTRY_TIMEOUT   = 3600  # detik maks tunggu limit order (1 jam)
# ─────────────────────────────────────────────

import os, time, logging, threading
from datetime import datetime, timezone

import requests, pandas as pd, numpy as np, urllib3
from flask import Flask

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── State global ──────────────────────────────
auto_mode       = False
auto_thread     = None
active_chat_id  = None
timeout_flag    = False   # /timeout → hentikan monitoring sekarang

# ── Statistik simulasi ────────────────────────
stat_lock = threading.Lock()
stats = {"tp": 0, "sl": 0, "no_entry": 0, "total": 0}

# ── Daftar ban koin ───────────────────────────
ban_lock    = threading.Lock()
banned_coins: set = set()

FAPI = "https://fapi.binance.com"

# ── Flask ─────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    with stat_lock:
        total = stats["total"]
        tp    = stats["tp"]
        sl    = stats["sl"]
        ne    = stats["no_entry"]
    pct_tp = f"{tp/total*100:.1f}%" if total else "–"
    pct_sl = f"{sl/total*100:.1f}%" if total else "–"
    banned = len(banned_coins)
    return (
        f"<h3>SMC Sim Bot</h3>"
        f"<p>Auto: {auto_mode} | Banned: {banned}</p>"
        f"<p>Total: {total} | TP: {tp} ({pct_tp}) | "
        f"SL: {sl} ({pct_sl}) | No Entry: {ne}</p>"
    ), 200

@app.route("/health")
def health():
    return "OK", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    log.info(f"Flask port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═════════════════════════════════════════════
# TELEGRAM
# ═════════════════════════════════════════════
def tg_send(chat_id, text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10)
    except Exception as e:
        log.error(f"[TG] {e}")

def tg_updates(offset=None):
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"timeout": 8, "offset": offset}, timeout=12)
        d = r.json()
        return d.get("result", []) if d.get("ok") else []
    except Exception as e:
        log.warning(f"[TG] {e}")
        return []


# ═════════════════════════════════════════════
# BINANCE DATA
# ═════════════════════════════════════════════
def fapi_get(path, params=None):
    for i in range(3):
        try:
            r = requests.get(f"{FAPI}{path}", params=params,
                             timeout=10, verify=False)
            d = r.json()
            if isinstance(d, dict) and "code" in d:
                raise ValueError(f"Binance {d['code']}: {d.get('msg')}")
            return d
        except Exception as e:
            log.warning(f"[fapi] {i+1}/3: {e}")
            time.sleep(2)
    raise ConnectionError(f"fapi gagal: {path}")

def get_price(symbol: str) -> float | None:
    """Ambil harga terkini satu koin — ringan, cepat."""
    try:
        d = fapi_get("/fapi/v1/ticker/price", {"symbol": symbol})
        return float(d["price"])
    except Exception as e:
        log.warning(f"[get_price] {symbol}: {e}")
        return None

def get_klines(symbol, interval, limit=200):
    raw = fapi_get("/fapi/v1/klines",
                   {"symbol": symbol, "interval": interval, "limit": limit})
    if not isinstance(raw, list) or len(raw) < 30:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=[
        "ts","open","high","low","close","volume",
        "cts","qvol","trades","tbv","tbq","ign"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open","high","low","close","volume"]].dropna()

def get_top_coins():
    tickers = fapi_get("/fapi/v1/ticker/24hr")
    with ban_lock:
        current_ban = set(banned_coins)
    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t["quoteVolume"]) > 100_000
        and t["symbol"] not in current_ban
    ]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]


# ═════════════════════════════════════════════
# INDIKATOR
# ═════════════════════════════════════════════
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd(s):
    line = ema(s,12) - ema(s,26)
    sig  = ema(line, 9)
    return line, sig, line - sig

def atr_series(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def indicators(df):
    if len(df) < 50: return None
    df = df.copy()
    df["ema9"]   = ema(df["close"], 9)
    df["ema21"]  = ema(df["close"], 21)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200) if len(df)>=200 else ema(df["close"],50)
    df["rsi"]    = rsi(df["close"])
    df["macd_line"], df["macd_sig"], df["macd_hist"] = macd(df["close"])
    df["atr"]    = atr_series(df)
    df["vol_sma"]= df["volume"].rolling(20).mean()
    bb_mid       = df["close"].rolling(20).mean()
    bb_std       = df["close"].rolling(20).std()
    df["bb_up"]  = bb_mid + 2*bb_std
    df["bb_lo"]  = bb_mid - 2*bb_std
    df["bb_mid"] = bb_mid
    return df.dropna()


# ═════════════════════════════════════════════
# SMC
# ═════════════════════════════════════════════
def swing_pts(df, lb=5):
    sh, sl = [], []
    for i in range(lb, len(df)-lb):
        if df["high"].iloc[i] == df["high"].iloc[i-lb:i+lb+1].max(): sh.append(i)
        if df["low"].iloc[i]  == df["low"].iloc[i-lb:i+lb+1].min():  sl.append(i)
    return sh, sl

def mkt_struct(df, sh, sl):
    if len(sh)<2 or len(sl)<2: return "ranging"
    hh = df["high"].iloc[sh[-1]] > df["high"].iloc[sh[-2]]
    hl = df["low"].iloc[sl[-1]]  > df["low"].iloc[sl[-2]]
    lh = df["high"].iloc[sh[-1]] < df["high"].iloc[sh[-2]]
    ll = df["low"].iloc[sl[-1]]  < df["low"].iloc[sl[-2]]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

def find_ob(df, direction, lb=30):
    sub = df.iloc[-lb:]
    avg = (sub["close"]-sub["open"]).abs().mean()
    obs = []
    for i in range(1, len(sub)-1):
        c, nx = sub.iloc[i], sub.iloc[i+1]
        if abs(nx["close"]-nx["open"]) < avg*1.2: continue
        if direction=="bull" and c["close"]<c["open"] and nx["close"]>nx["open"]:
            obs.append({"top":c["high"],"bot":c["low"]})
        if direction=="bear" and c["close"]>c["open"] and nx["close"]<nx["open"]:
            obs.append({"top":c["high"],"bot":c["low"]})
    return obs[-2:] if obs else []

def find_fvg(df, direction, lb=30):
    sub = df.iloc[-lb:]
    out = []
    for i in range(len(sub)-2):
        c0, c2 = sub.iloc[i], sub.iloc[i+2]
        if direction=="bull" and c2["low"]>c0["high"]:
            out.append({"top":c2["low"],"bot":c0["high"]})
        if direction=="bear" and c2["high"]<c0["low"]:
            out.append({"top":c0["low"],"bot":c2["high"]})
    return out[-2:] if out else []

def liq_sweep(df, sh, sl):
    bull=bear=False
    cur,prv = df.iloc[-1], df.iloc[-2]
    if sl:
        ll = df["low"].iloc[sl[-1]]
        if prv["low"]<ll<cur["close"]: bull=True
    if sh:
        lh = df["high"].iloc[sh[-1]]
        if prv["high"]>lh>cur["close"]: bear=True
    return bull, bear

def choch_bos(df, sh, sl):
    res = dict(bos_bull=False,bos_bear=False,choch_bull=False,choch_bear=False)
    p = df["close"].iloc[-1]
    if len(sh)>=2:
        ph=df["high"].iloc[sh[-2]]; lh=df["high"].iloc[sh[-1]]
        if p>ph: res["bos_bull" if lh>ph else "choch_bull"]=True
    if len(sl)>=2:
        pl=df["low"].iloc[sl[-2]]; ll=df["low"].iloc[sl[-1]]
        if p<pl: res["bos_bear" if ll<pl else "choch_bear"]=True
    return res


# ═════════════════════════════════════════════
# SCORING
# ═════════════════════════════════════════════
def score_coin(df_h1, df_m15):
    h1  = indicators(df_h1)
    m15 = indicators(df_m15)
    if h1 is None or m15 is None: return None

    L1=h1.iloc[-1]; P1=h1.iloc[-2]
    L15=m15.iloc[-1]; P15=m15.iloc[-2]

    bull_pts = bear_pts = 0

    # EMA H1
    if L1["ema9"]>L1["ema21"]>L1["ema50"]:   bull_pts+=15
    elif L1["ema9"]>L1["ema21"]:              bull_pts+=8
    if L1["ema9"]<L1["ema21"]<L1["ema50"]:   bear_pts+=15
    elif L1["ema9"]<L1["ema21"]:              bear_pts+=8

    # EMA200 H1
    if L1["close"]>L1["ema200"]: bull_pts+=8
    else:                         bear_pts+=8

    # RSI M15
    rv = L15["rsi"]
    if rv<40:   bull_pts+=10
    elif rv<50: bull_pts+=5
    if rv>60:   bear_pts+=10
    elif rv>50: bear_pts+=5

    # MACD M15
    if L15["macd_hist"]>0 and P15["macd_hist"]<=0: bull_pts+=12
    elif L15["macd_hist"]>0:                        bull_pts+=6
    if L15["macd_hist"]<0 and P15["macd_hist"]>=0: bear_pts+=12
    elif L15["macd_hist"]<0:                        bear_pts+=6

    # EMA M15
    if L15["ema9"]>L15["ema21"]>L15["ema50"]: bull_pts+=10
    elif L15["ema9"]>L15["ema21"]:             bull_pts+=5
    if L15["ema9"]<L15["ema21"]<L15["ema50"]: bear_pts+=10
    elif L15["ema9"]<L15["ema21"]:             bear_pts+=5

    # Bollinger M15
    if L15["close"]<L15["bb_lo"]:   bull_pts+=8
    elif L15["close"]<L15["bb_mid"]:bull_pts+=3
    if L15["close"]>L15["bb_up"]:   bear_pts+=8
    elif L15["close"]>L15["bb_mid"]:bear_pts+=3

    # Volume M15
    if L15["volume"]>L15["vol_sma"]*1.3:
        if L15["close"]>L15["open"]: bull_pts+=7
        else:                         bear_pts+=7
    elif L15["volume"]>L15["vol_sma"]:
        if L15["close"]>L15["open"]: bull_pts+=3
        else:                         bear_pts+=3

    # Market structure H1
    sh1,sl1  = swing_pts(h1,5)
    struct_h1= mkt_struct(h1,sh1,sl1)
    if struct_h1=="bullish": bull_pts+=10
    if struct_h1=="bearish": bear_pts+=10

    # SMC M15
    sh15,sl15    = swing_pts(m15,5)
    bos          = choch_bos(m15,sh15,sl15)
    sw_bull,sw_bear = liq_sweep(m15,sh15,sl15)
    obs_bull = find_ob(m15,"bull"); obs_bear = find_ob(m15,"bear")
    fvg_bull = find_fvg(m15,"bull"); fvg_bear= find_fvg(m15,"bear")

    if bos["bos_bull"] or bos["choch_bull"]: bull_pts+=10
    if bos["bos_bear"] or bos["choch_bear"]: bear_pts+=10
    if sw_bull: bull_pts+=8
    if sw_bear: bear_pts+=8
    if obs_bull: bull_pts+=7
    if obs_bear: bear_pts+=7
    if fvg_bull: bull_pts+=5
    if fvg_bear: bear_pts+=5

    direction = "bull" if bull_pts>=bear_pts else "bear"
    raw_conf  = bull_pts if direction=="bull" else bear_pts
    confidence= min(int(raw_conf/130*100), 99)

    obs  = obs_bull  if direction=="bull" else obs_bear
    fvgs = fvg_bull  if direction=="bull" else fvg_bear

    price   = L15["close"]
    atr_val = L15["atr"] if L15["atr"]>0 else price*0.005

    entry,sl_p,tp_p,etype,conf_lvl = calc_setup(
        m15, direction, price, atr_val, obs, fvgs, sh15, sl15)
    if entry is None: return None

    risk   = abs(entry-sl_p)
    reward = abs(tp_p-entry)
    rr     = round(reward/risk,2) if risk>0 else 0

    why = []
    if struct_h1!="ranging":       why.append(f"H1:{struct_h1.upper()}")
    if L1["close"]>L1["ema200"]:   why.append("Above EMA200")
    elif L1["close"]<L1["ema200"]: why.append("Below EMA200")
    if bos["bos_bull"] or bos["bos_bear"]:     why.append("BOS✔")
    if bos["choch_bull"] or bos["choch_bear"]: why.append("CHoCH✔")
    if sw_bull or sw_bear:                      why.append("LiqSweep✔")
    if obs:  why.append(f"OB:{obs[-1]['bot']:.4g}–{obs[-1]['top']:.4g}")
    if fvgs: why.append(f"FVG:{fvgs[-1]['bot']:.4g}–{fvgs[-1]['top']:.4g}")
    mc = (direction=="bull" and L15["macd_hist"]>0 and P15["macd_hist"]<=0) or \
         (direction=="bear" and L15["macd_hist"]<0 and P15["macd_hist"]>=0)
    if mc: why.append("MACD Cross✔")
    why.append(f"RSI:{rv:.0f}")

    return {
        "decision":direction=="bull" and "BUY" or "SELL",
        "confidence":confidence,
        "price":price,
        "entry":round(entry,8),
        "sl":round(sl_p,8),
        "tp":round(tp_p,8),
        "rr":rr,
        "etype":etype,
        "conf_lvl":conf_lvl,
        "reason":" | ".join(why),
        "rsi":round(rv,1),
    }


def calc_setup(df, direction, price, atr_val, obs, fvgs, sh, sl_pts):
    def try_zone(ztop, zbot):
        buf  = atr_val*0.3
        in_z = zbot<=price<=ztop
        if direction=="bull":
            entry    = price if in_z else ztop
            etype    = "MARKET" if in_z else "LIMIT"
            conf_lvl = None if in_z else round(ztop,8)
            sl_p     = zbot-buf
            cands    = [df["high"].iloc[i] for i in sh if df["high"].iloc[i]>entry]
            tp_p     = min(cands) if cands else entry+abs(entry-sl_p)*MIN_RR
        else:
            entry    = price if in_z else zbot
            etype    = "MARKET" if in_z else "LIMIT"
            conf_lvl = None if in_z else round(zbot,8)
            sl_p     = ztop+buf
            cands    = [df["low"].iloc[i] for i in sl_pts if df["low"].iloc[i]<entry]
            tp_p     = max(cands) if cands else entry-abs(sl_p-entry)*MIN_RR
        risk=abs(entry-sl_p)
        if risk==0: return None
        if abs(tp_p-entry)/risk<MIN_RR: return None
        return entry,sl_p,tp_p,etype,conf_lvl

    for ob in reversed(obs):
        for fvg in reversed(fvgs):
            ot=min(ob["top"],fvg["top"]); ob_=max(ob["bot"],fvg["bot"])
            if ot>ob_:
                r=try_zone(ot,ob_)
                if r: return r
    for ob in reversed(obs):
        r=try_zone(ob["top"],ob["bot"])
        if r: return r
    for fvg in reversed(fvgs):
        r=try_zone(fvg["top"],fvg["bot"])
        if r: return r

    # Fallback ATR
    if direction=="bull":
        entry=price; sl_p=price-atr_val*1.5; tp_p=price+atr_val*1.5*MIN_RR
    else:
        entry=price; sl_p=price+atr_val*1.5; tp_p=price-atr_val*1.5*MIN_RR
    return entry,sl_p,tp_p,"MARKET",None


def analyze(symbol):
    try:
        h1  = get_klines(symbol,"1h",200)
        m15 = get_klines(symbol,"15m",200)
        if h1.empty or m15.empty: return None
        r = score_coin(h1, m15)
        if r: r["symbol"]=symbol
        return r
    except Exception as e:
        log.debug(f"[analyze] {symbol}: {e}")
        return None


# ═════════════════════════════════════════════
# SCAN — ambil 1 sinyal terbaik
# ═════════════════════════════════════════════
def run_scan_once(chat_id) -> dict | None:
    """Scan 50 koin, kembalikan 1 sinyal terbaik atau None."""
    tg_send(chat_id, f"🔍 Scanning {TOP_N_COINS} koin... mohon tunggu.")
    try:
        symbols = get_top_coins()
    except Exception as e:
        tg_send(chat_id, f"⚠️ Gagal ambil data Binance: <code>{str(e)[:200]}</code>")
        return None

    if not symbols:
        tg_send(chat_id, "⚠️ Semua koin masuk daftar ban atau tidak ada data.")
        return None

    results = []
    for idx, sym in enumerate(symbols, 1):
        log.info(f"[{idx:02d}/{len(symbols)}] {sym}")
        r = analyze(sym)
        if r: results.append(r)
        time.sleep(0.08)

    if not results:
        tg_send(chat_id, "⚠️ Tidak ada data valid dari semua koin.")
        return None

    results.sort(key=lambda x: (x["confidence"], x["rr"]), reverse=True)
    return results[0]


# ═════════════════════════════════════════════
# MONITORING — cek harga tiap N detik
# ═════════════════════════════════════════════
def monitor_trade(chat_id, signal: dict) -> str:
    """
    Monitor harga hingga TP/SL/timeout.
    Return: 'tp' | 'sl' | 'timeout' | 'no_entry'
    """
    global timeout_flag
    sym       = signal["symbol"]
    direction = signal["decision"]   # "BUY" / "SELL"
    entry     = signal["entry"]
    sl_p      = signal["sl"]
    tp_p      = signal["tp"]
    etype     = signal["etype"]
    is_buy    = direction == "BUY"

    # ── LIMIT ORDER: tunggu harga menyentuh entry ──
    if etype == "LIMIT":
        tg_send(chat_id,
            f"⏳ <b>Menunggu LIMIT entry</b>\n"
            f"Koin    : <b>{sym}</b>\n"
            f"Entry   : <code>{entry:.6g}</code>\n"
            f"TP      : <code>{tp_p:.6g}</code>\n"
            f"SL      : <code>{sl_p:.6g}</code>\n"
            f"Timeout : {ENTRY_TIMEOUT//60} menit\n\n"
            f"Ketik /timeout untuk lewati.")

        start = time.time()
        entry_hit = False
        while time.time()-start < ENTRY_TIMEOUT:
            if timeout_flag:
                timeout_flag = False
                tg_send(chat_id, f"⏭ <b>/timeout</b> — limit {sym} dibatalkan.")
                return "timeout"

            price = get_price(sym)
            if price is None:
                time.sleep(MONITOR_SLEEP)
                continue

            # Cek apakah TP sudah tersentuh sebelum entry → batalkan
            if is_buy and price >= tp_p:
                tg_send(chat_id,
                    f"❌ <b>Entry Dibatalkan — {sym}</b>\n"
                    f"TP <code>{tp_p:.6g}</code> tersentuh sebelum entry limit.\n"
                    f"Harga sekarang: <code>{price:.6g}</code>")
                return "no_entry"
            if not is_buy and price <= tp_p:
                tg_send(chat_id,
                    f"❌ <b>Entry Dibatalkan — {sym}</b>\n"
                    f"TP <code>{tp_p:.6g}</code> tersentuh sebelum entry limit.\n"
                    f"Harga sekarang: <code>{price:.6g}</code>")
                return "no_entry"

            # Entry tersentuh?
            if is_buy and price <= entry:
                entry_hit = True; break
            if not is_buy and price >= entry:
                entry_hit = True; break

            time.sleep(MONITOR_SLEEP)

        if not entry_hit:
            tg_send(chat_id,
                f"⏰ <b>Limit timeout — {sym}</b>\n"
                f"Entry <code>{entry:.6g}</code> tidak tercapai dalam "
                f"{ENTRY_TIMEOUT//60} menit.")
            return "no_entry"

        tg_send(chat_id,
            f"✅ <b>LIMIT ENTRY TERISI — {sym}</b>\n"
            f"Entry : <code>{entry:.6g}</code>\n"
            f"TP    : <code>{tp_p:.6g}</code>\n"
            f"SL    : <code>{sl_p:.6g}</code>")

    else:
        # ── MARKET ORDER: langsung masuk ──────────
        # Ambil harga aktual saat ini (bisa sedikit berbeda dari analisis)
        actual_entry = get_price(sym) or entry
        tg_send(chat_id,
            f"⚡ <b>MARKET ENTRY — {sym}</b>\n"
            f"Keputusan : <b>{direction}</b>\n"
            f"Entry analisis: <code>{entry:.6g}</code>\n"
            f"Entry aktual  : <code>{actual_entry:.6g}</code>\n"
            f"TP    : <code>{tp_p:.6g}</code>\n"
            f"SL    : <code>{sl_p:.6g}</code>\n"
            f"RR    : 1:{signal['rr']}\n\n"
            f"📡 Monitoring harga setiap {MONITOR_SLEEP} detik...\n"
            f"Ketik /timeout untuk hentikan monitoring.")

    # ── MONITORING setelah entry ───────────────────
    log_interval = 60   # kirim update tiap N detik
    last_log     = time.time()

    while True:
        if timeout_flag:
            timeout_flag = False
            price = get_price(sym) or 0
            tg_send(chat_id,
                f"⏭ <b>/timeout</b> — monitoring {sym} dihentikan.\n"
                f"Harga terakhir: <code>{price:.6g}</code>")
            return "timeout"

        price = get_price(sym)
        if price is None:
            time.sleep(MONITOR_SLEEP)
            continue

        # Cek TP
        if is_buy and price >= tp_p:
            tg_send(chat_id,
                f"🎯 <b>TAKE PROFIT — {sym}</b> 🎉\n"
                f"TP tercapai: <code>{price:.6g}</code>\n"
                f"Target TP : <code>{tp_p:.6g}</code>")
            return "tp"

        if not is_buy and price <= tp_p:
            tg_send(chat_id,
                f"🎯 <b>TAKE PROFIT — {sym}</b> 🎉\n"
                f"TP tercapai: <code>{price:.6g}</code>\n"
                f"Target TP : <code>{tp_p:.6g}</code>")
            return "tp"

        # Cek SL
        if is_buy and price <= sl_p:
            tg_send(chat_id,
                f"🛑 <b>STOP LOSS — {sym}</b>\n"
                f"SL tercapai: <code>{price:.6g}</code>\n"
                f"Target SL : <code>{sl_p:.6g}</code>")
            return "sl"

        if not is_buy and price >= sl_p:
            tg_send(chat_id,
                f"🛑 <b>STOP LOSS — {sym}</b>\n"
                f"SL tercapai: <code>{price:.6g}</code>\n"
                f"Target SL : <code>{sl_p:.6g}</code>")
            return "sl"

        # Update berkala
        if time.time()-last_log >= log_interval:
            dist_tp = abs(tp_p-price)
            dist_sl = abs(price-sl_p)
            pct_tp  = dist_tp/abs(tp_p-entry)*100 if abs(tp_p-entry)>0 else 0
            tg_send(chat_id,
                f"📊 <b>Monitor {sym}</b>\n"
                f"Harga  : <code>{price:.6g}</code>\n"
                f"TP     : <code>{tp_p:.6g}</code> (sisa {pct_tp:.1f}%)\n"
                f"SL     : <code>{sl_p:.6g}</code>")
            last_log = time.time()

        time.sleep(MONITOR_SLEEP)


# ═════════════════════════════════════════════
# STATISTIK
# ═════════════════════════════════════════════
def update_stats(result: str):
    with stat_lock:
        stats["total"] += 1
        if result in ("tp","sl","no_entry","timeout"):
            stats[result if result in stats else "no_entry"] += 1


def fmt_stats() -> str:
    with stat_lock:
        total = stats["total"]
        tp    = stats["tp"]
        sl    = stats["sl"]
        ne    = stats["no_entry"]
        to    = stats.get("timeout", 0)
    if total == 0:
        return "Belum ada simulasi selesai."
    pct_tp = tp/total*100
    pct_sl = sl/total*100
    return (
        f"📊 <b>Statistik Simulasi</b>\n\n"
        f"Total trade  : {total}\n"
        f"✅ TP        : {tp} ({pct_tp:.1f}%)\n"
        f"🛑 SL        : {sl} ({pct_sl:.1f}%)\n"
        f"❌ No Entry  : {ne}\n"
        f"⏭ Timeout   : {to}\n\n"
        f"🚫 Koin diban: {len(banned_coins)}"
    )


def fmt_signal_msg(sig: dict) -> str:
    em    = "🟢" if sig["decision"]=="BUY" else "🔴"
    bar   = "█"*(sig["confidence"]//10) + "░"*(10-sig["confidence"]//10)
    etype = (
        f"⏳ LIMIT — tunggu harga menyentuh <code>{sig['conf_lvl']}</code>"
        if sig["etype"]=="LIMIT" and sig.get("conf_lvl")
        else "⚡ MARKET — entry sekarang"
    )
    return (
        f"📡 <b>SINYAL TERBAIK</b>\n\n"
        f"{em} <b>{sig['symbol']}</b>\n"
        f"Keputusan   : <b>{sig['decision']}</b>\n"
        f"Confidence  : <b>{sig['confidence']}%</b> {bar}\n"
        f"Harga kini  : <code>{sig['price']:.6g}</code>\n"
        f"🎯 Entry    : <code>{sig['entry']:.6g}</code>\n"
        f"Tipe        : {etype}\n"
        f"🛑 Stop Loss: <code>{sig['sl']:.6g}</code>\n"
        f"✅ Take Profit: <code>{sig['tp']:.6g}</code>\n"
        f"⚖️ RR       : <b>1:{sig['rr']}</b>\n"
        f"📝 Analisis : {sig['reason']}"
    )


# ═════════════════════════════════════════════
# AUTO LOOP SIMULASI
# ═════════════════════════════════════════════
def simulation_loop(chat_id):
    global auto_mode, timeout_flag
    tg_send(chat_id, "🤖 <b>Simulasi Trading dimulai!</b>\nBot akan scan → simulasi → scan → ...")

    while auto_mode:
        timeout_flag = False

        # 1. SCAN — cari 1 sinyal terbaik
        signal = run_scan_once(chat_id)

        if not auto_mode: break

        if signal is None:
            tg_send(chat_id, "⚠️ Tidak ada sinyal valid. Scan ulang dalam 60 detik...")
            for _ in range(60):
                if not auto_mode: break
                time.sleep(1)
            continue

        # 2. Tampilkan sinyal
        tg_send(chat_id, fmt_signal_msg(signal))

        # 3. Ban koin ini (tidak boleh masuk scan berikutnya)
        sym = signal["symbol"]
        with ban_lock:
            banned_coins.add(sym)
        log.info(f"[ban] {sym} ditambahkan. Total ban: {len(banned_coins)}")

        # 4. SIMULASI TRADE — monitoring sampai TP/SL
        result = monitor_trade(chat_id, signal)

        # 5. Update statistik
        update_stats(result)
        with stat_lock:
            stats.setdefault("timeout", 0)
            if result == "timeout":
                stats["timeout"] += 1

        # 6. Kirim ringkasan trade ini + statistik kumulatif
        result_emoji = {"tp":"🎯","sl":"🛑","no_entry":"❌","timeout":"⏭"}.get(result,"❓")
        result_label = {"tp":"TAKE PROFIT","sl":"STOP LOSS",
                        "no_entry":"NO ENTRY","timeout":"TIMEOUT"}.get(result,result.upper())
        tg_send(chat_id,
            f"{result_emoji} <b>Hasil: {result_label}</b> — {sym}\n\n"
            + fmt_stats())

        if not auto_mode: break

        # 7. Jeda singkat sebelum scan berikutnya
        tg_send(chat_id, "⏳ Scan berikutnya dalam 10 detik...")
        for _ in range(10):
            if not auto_mode: break
            time.sleep(1)

    tg_send(chat_id, "⏹ <b>Simulasi dihentikan.</b>\n\n" + fmt_stats())


# ═════════════════════════════════════════════
# PESAN STATIS
# ═════════════════════════════════════════════
GREETING = (
    "👋 <b>SMC Simulasi Trading Bot</b>\n\n"
    "Bot ini menscan koin → menemukan sinyal terbaik → mensimulasikan trade.\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "📌 <b>Perintah:</b>\n"
    "/start    — Pesan ini\n"
    "/auto     — Mulai simulasi otomatis\n"
    "/stop     — Hentikan simulasi\n"
    "/timeout  — Skip monitoring saat ini, lanjut scan\n"
    "/stats    — Lihat statistik TP/SL\n"
    "/banned   — Lihat daftar koin yang diban\n"
    "/resetban — Reset daftar ban\n"
    "/info     — Detail metode analisis\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "⚠️ <i>Ini adalah simulasi. Bukan saran finansial.</i>"
)

INFO_MSG = (
    "ℹ️ <b>Metode Analisis</b>\n\n"
    "• EMA 9/21/50/200 (H1 + M15)\n"
    "• RSI 14 oversold/overbought\n"
    "• MACD crossover\n"
    "• Bollinger Bands posisi\n"
    "• Volume vs rata-rata\n"
    "• Market Structure H1 (HH/HL)\n"
    "• BOS + CHoCH\n"
    "• Order Block + FVG\n"
    "• Liquidity Sweep\n\n"
    "Entry: MARKET (langsung) atau LIMIT (tunggu zona)\n"
    f"Min RR: 1:{MIN_RR} | Fallback: ATR-based\n\n"
    "Alur simulasi:\n"
    "Scan → Sinyal #1 → Ban koin → Monitor TP/SL → Stats → Scan lagi"
)


# ═════════════════════════════════════════════
# BOT LOOP (command handler)
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

    offset = None
    log.info("Bot siap. Menunggu perintah Telegram...")

    while True:
        try:
            for upd in tg_updates(offset):
                offset  = upd["update_id"] + 1
                msg     = upd.get("message", {})
                uid     = msg.get("from", {}).get("id")
                chat_id = msg.get("chat", {}).get("id")
                text    = msg.get("text", "").strip().lower()

                if not uid or not chat_id or not text: continue
                if uid != ALLOWED_USER_ID:
                    tg_send(chat_id, "⛔ Akses ditolak.")
                    continue

                active_chat_id = chat_id

                if text in ("/start","start"):
                    tg_send(chat_id, GREETING)

                elif text in ("/info","info"):
                    tg_send(chat_id, INFO_MSG)

                elif text in ("/stats","stats"):
                    tg_send(chat_id, fmt_stats())

                elif text in ("/banned","banned"):
                    with ban_lock:
                        b = sorted(banned_coins)
                    if b:
                        tg_send(chat_id,
                            f"🚫 <b>Koin diban ({len(b)}):</b>\n" + ", ".join(b))
                    else:
                        tg_send(chat_id, "✅ Belum ada koin yang diban.")

                elif text in ("/resetban","resetban"):
                    with ban_lock:
                        n = len(banned_coins)
                        banned_coins.clear()
                    tg_send(chat_id, f"✅ Daftar ban direset. ({n} koin dihapus)")

                elif text in ("/auto","auto"):
                    if auto_mode:
                        tg_send(chat_id, "⚙️ Simulasi sudah berjalan.")
                    else:
                        auto_mode   = True
                        auto_thread = threading.Thread(
                            target=simulation_loop, args=(chat_id,), daemon=True)
                        auto_thread.start()

                elif text in ("/stop","stop"):
                    if auto_mode:
                        auto_mode    = False
                        timeout_flag = True   # hentikan monitoring juga
                        tg_send(chat_id, "⏹ Menghentikan simulasi...")
                    else:
                        tg_send(chat_id, "ℹ️ Simulasi tidak sedang berjalan.")

                elif text in ("/timeout","timeout"):
                    if auto_mode:
                        timeout_flag = True
                        tg_send(chat_id, "⏭ Timeout diterima — monitoring akan dilewati.")
                    else:
                        tg_send(chat_id, "ℹ️ Tidak ada monitoring aktif.")

                else:
                    tg_send(chat_id, "❓ Tidak dikenal. Ketik /start.")

            time.sleep(1)

        except Exception as e:
            log.error(f"[bot loop] {e}")
            time.sleep(5)


# ═════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════
if __name__ == "__main__":
    threading.Thread(target=bot_loop, daemon=True).start()
    run_flask()
