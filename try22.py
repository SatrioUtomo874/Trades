#!/usr/bin/env python3
"""
SMC Simulasi Trading Bot v4
Render.com | python main.py
"""

# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
ALLOWED_USER_ID = 8041197505
MAX_PRICE       = 80.0
TOP_N_COINS     = 50
MIN_RR          = 2.0
MONITOR_SLEEP   = 2
ENTRY_WAIT_MAX  = 1800   # 30 menit tunggu limit, lalu cancel
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

stat_lock = threading.Lock()
stats = {"tp":0,"sl":0,"no_entry":0,"timeout":0,"total":0}

ban_lock = threading.Lock()
banned_coins: set = set()

FAPI = "https://fapi.binance.com"

# ── Flask ──────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    with stat_lock:
        t=stats["total"]; tp=stats["tp"]; sl=stats["sl"]
    wr=f"{tp/(tp+sl)*100:.1f}%" if (tp+sl)>0 else "–"
    return (f"<h3>SMC Sim Bot v4</h3>"
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
# BINANCE
# ═════════════════════════════════════════════
def fapi_get(path, params=None):
    for i in range(4):
        try:
            r = requests.get(f"{FAPI}{path}", params=params,
                             timeout=10, verify=False)
            d = r.json()
            if isinstance(d, dict) and "code" in d:
                raise ValueError(f"Binance {d['code']}: {d.get('msg')}")
            return d
        except Exception as e:
            log.warning(f"[fapi] {i+1}/4: {e}")
            time.sleep(3)
    raise ConnectionError(f"fapi gagal: {path}")

def get_price(symbol):
    """Ambil harga real-time — robust, tidak crash."""
    for _ in range(3):
        try:
            d = fapi_get("/fapi/v1/ticker/price", {"symbol": symbol})
            return float(d["price"])
        except:
            time.sleep(1)
    return None

def get_klines(symbol, interval, limit=250):
    raw = fapi_get("/fapi/v1/klines",
                   {"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(raw,list) or len(raw)<40:
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
        cur_ban = set(banned_coins)
    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t["quoteVolume"]) > 100_000
        and t["symbol"] not in cur_ban
    ]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]


# ═════════════════════════════════════════════
# INDIKATOR
# ═════════════════════════════════════════════
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d=s.diff()
    g=d.clip(lower=0).rolling(n).mean()
    l=(-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1 + g/l.replace(0, np.nan))

def macd(s):
    line=ema(s,12)-ema(s,26)
    sig=ema(line,9)
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
# SMC TOOLS
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

def find_ob(df, direction, lb=30):
    sub=df.iloc[-lb:]
    avg=(sub["close"]-sub["open"]).abs().mean()
    obs=[]
    for i in range(1, len(sub)-1):
        c,nx=sub.iloc[i],sub.iloc[i+1]
        if abs(nx["close"]-nx["open"])<avg*1.2: continue
        if direction=="bull" and c["close"]<c["open"] and nx["close"]>nx["open"]:
            obs.append({"top":c["high"],"bot":c["low"],"mid":(c["high"]+c["low"])/2})
        if direction=="bear" and c["close"]>c["open"] and nx["close"]<nx["open"]:
            obs.append({"top":c["high"],"bot":c["low"],"mid":(c["high"]+c["low"])/2})
    # Kembalikan OB yang paling dekat dengan harga sekarang
    price=df["close"].iloc[-1]
    if direction=="bull":
        obs=[o for o in obs if o["top"]<=price*1.01]  # OB di bawah atau dekat harga
    else:
        obs=[o for o in obs if o["bot"]>=price*0.99]
    return obs[-2:] if obs else []

def find_fvg(df, direction, lb=30):
    sub=df.iloc[-lb:]; out=[]
    price=df["close"].iloc[-1]
    for i in range(len(sub)-2):
        c0,c2=sub.iloc[i],sub.iloc[i+2]
        if direction=="bull" and c2["low"]>c0["high"]:
            fvg={"top":c2["low"],"bot":c0["high"],"mid":(c2["low"]+c0["high"])/2}
            if fvg["top"]<=price*1.01:
                out.append(fvg)
        if direction=="bear" and c2["high"]<c0["low"]:
            fvg={"top":c0["low"],"bot":c2["high"],"mid":(c0["low"]+c2["high"])/2}
            if fvg["bot"]>=price*0.99:
                out.append(fvg)
    return out[-2:] if out else []

def sweep_rejection(df, sh, sl):
    """
    Deteksi liquidity sweep + rejection candle.
    Ini adalah sinyal reversal terkuat — institusi ambil likuiditas lalu balik.
    """
    if len(df)<3: return False, False, "", ""
    cur=df.iloc[-1]; prv=df.iloc[-2]
    bull_sweep=bear_sweep=False
    rb=rs=""

    if sl:
        ll=df["low"].iloc[sl[-1]]
        # Prv: spike ke bawah melewati swing low, lalu cur: close di atas (rejection)
        if prv["low"]<ll and cur["close"]>ll:
            bull_pinbar=(cur["close"]-cur["open"])>0  # candle bullish
            lower_wick=cur["open"]-cur["low"] if cur["close"]>cur["open"] else cur["close"]-cur["low"]
            body=abs(cur["close"]-cur["open"])
            if lower_wick>body*0.5 or bull_pinbar:
                bull_sweep=True
                rb=f"Sweep {ll:.4g}+rejection"

    if sh:
        lh=df["high"].iloc[sh[-1]]
        if prv["high"]>lh and cur["close"]<lh:
            bear_pinbar=(cur["close"]-cur["open"])<0
            upper_wick=cur["high"]-cur["open"] if cur["close"]<cur["open"] else cur["high"]-cur["close"]
            body=abs(cur["close"]-cur["open"])
            if upper_wick>body*0.5 or bear_pinbar:
                bear_sweep=True
                rs=f"Sweep {lh:.4g}+rejection"

    return bull_sweep, bear_sweep, rb, rs


# ═════════════════════════════════════════════
# SETUP ENTRY/SL/TP
# ═════════════════════════════════════════════
def build_setup(df, direction, sh, sl, obs, fvgs, atr_val, swept):
    """
    Filosofi:
    - Entry SELALU di harga pasar saat ini jika kondisi terpenuhi.
      LIMIT hanya jika ada OB/FVG yang belum disentuh dan sangat dekat (<2*ATR).
    - SL = di luar swing struktural + buffer ATR kecil.
    - TP = level swing berikutnya yang realistis, pastikan RR>=2.
    """
    price = df["close"].iloc[-1]

    # ── SL struktural ─────────────────────────────────────────────────
    if direction=="bull":
        # SL: swing low terdekat YANG VALID di bawah harga
        candidates=[df["low"].iloc[i] for i in sl
                    if df["low"].iloc[i] < price - atr_val*0.5]
        if candidates:
            structural_low=max(candidates)  # swing low terdekat
            sl_price=structural_low - atr_val*0.3
        else:
            sl_price=price - atr_val*2.5
    else:
        candidates=[df["high"].iloc[i] for i in sh
                    if df["high"].iloc[i] > price + atr_val*0.5]
        if candidates:
            structural_high=min(candidates)
            sl_price=structural_high + atr_val*0.3
        else:
            sl_price=price + atr_val*2.5

    risk = abs(price - sl_price)
    if risk < atr_val*0.3:
        # SL terlalu dekat → perluas
        sl_price = price - atr_val*2 if direction=="bull" else price + atr_val*2
        risk = abs(price - sl_price)

    # ── TP: level nyata berikutnya ─────────────────────────────────────
    min_reward = risk * MIN_RR

    if direction=="bull":
        # TP = swing high terdekat di atas entry + min_reward
        tp_candidates=[df["high"].iloc[i] for i in sh
                       if df["high"].iloc[i] > price + min_reward*0.7]
        if tp_candidates:
            tp=min(tp_candidates)
            if tp < price+min_reward:
                tp=price+min_reward
        else:
            tp=price+min_reward
    else:
        tp_candidates=[df["low"].iloc[i] for i in sl
                       if df["low"].iloc[i] < price - min_reward*0.7]
        if tp_candidates:
            tp=max(tp_candidates)
            if tp > price-min_reward:
                tp=price-min_reward
        else:
            tp=price-min_reward

    # ── Tentukan tipe entry ────────────────────────────────────────────
    # Default: MARKET (entry sekarang)
    entry=price; etype="MARKET"; conf_lvl=None; entry_reason=""

    if swept:
        # Setelah sweep → market entry langsung
        entry_reason="Post-sweep"
        etype="MARKET"

    elif obs:
        ob=obs[-1]
        dist=abs(price-ob["mid"])
        if dist < atr_val*1.5:
            # OB sangat dekat → tetap market entry (sudah di zona)
            entry=price; etype="MARKET"; entry_reason=f"OB zone (dekat)"
        else:
            # OB jauh → tetap market entry, catat OB sebagai referensi
            entry=price; etype="MARKET"; entry_reason=f"OB ref {ob['bot']:.4g}–{ob['top']:.4g}"

    elif fvgs:
        fvg=fvgs[-1]
        dist=abs(price-fvg["mid"])
        if dist < atr_val*1.5:
            entry=price; etype="MARKET"; entry_reason="FVG zone (dekat)"
        else:
            entry=price; etype="MARKET"; entry_reason=f"FVG ref {fvg['bot']:.4g}–{fvg['top']:.4g}"
    else:
        entry_reason="EMA+momentum"

    # Hitung RR final
    actual_risk=abs(entry-sl_price)
    actual_reward=abs(tp-entry)
    if actual_risk==0: return None
    rr=round(actual_reward/actual_risk,2)
    if rr<MIN_RR: return None

    return {
        "entry"       : round(entry,8),
        "sl"          : round(sl_price,8),
        "tp"          : round(tp,8),
        "rr"          : rr,
        "etype"       : etype,
        "conf_lvl"    : conf_lvl,
        "entry_reason": entry_reason,
    }


# ═════════════════════════════════════════════
# SCORING & ANALISIS
# ═════════════════════════════════════════════
def score_and_analyze(df_h1, df_m15):
    h1=build_df(df_h1); m15=build_df(df_m15)
    if h1 is None or m15 is None: return None

    L1=h1.iloc[-1]; P1=h1.iloc[-2]
    L15=m15.iloc[-1]; P15=m15.iloc[-2]
    rv=L15["rsi"]; atr_val=max(L15["atr"], L15["close"]*0.003)

    bull=bear=0

    # ── Trend H1 ─────────
    if L1["ema9"]>L1["ema21"]>L1["ema50"]:  bull+=15
    elif L1["ema9"]>L1["ema21"]:             bull+=8
    if L1["ema9"]<L1["ema21"]<L1["ema50"]:  bear+=15
    elif L1["ema9"]<L1["ema21"]:             bear+=8
    if L1["close"]>L1["ema200"]:             bull+=8
    else:                                     bear+=8

    # ── RSI M15 ──────────
    if rv<35:    bull+=12
    elif rv<45:  bull+=6
    if rv>65:    bear+=12
    elif rv>55:  bear+=6

    # ── MACD M15 ─────────
    if L15["mh"]>0 and P15["mh"]<=0:  bull+=12
    elif L15["mh"]>0:                  bull+=5
    if L15["mh"]<0 and P15["mh"]>=0:  bear+=12
    elif L15["mh"]<0:                  bear+=5

    # ── EMA M15 ──────────
    if L15["ema9"]>L15["ema21"]>L15["ema50"]: bull+=10
    elif L15["ema9"]>L15["ema21"]:             bull+=5
    if L15["ema9"]<L15["ema21"]<L15["ema50"]: bear+=10
    elif L15["ema9"]<L15["ema21"]:             bear+=5

    # ── Bollinger ────────
    if L15["close"]<=L15["bb_lo"]:    bull+=10
    elif L15["close"]<L15["bb_mid"]:  bull+=4
    if L15["close"]>=L15["bb_up"]:    bear+=10
    elif L15["close"]>L15["bb_mid"]:  bear+=4

    # ── Volume ───────────
    if L15["volume"]>L15["vol_sma"]*1.5:
        if L15["close"]>L15["open"]:  bull+=8
        else:                          bear+=8
    elif L15["volume"]>L15["vol_sma"]:
        if L15["close"]>L15["open"]:  bull+=3
        else:                          bear+=3

    # ── Market Structure H1 ──
    sh1,sl1=swing_pts(h1,5)
    struct_h1=mkt_struct(h1,sh1,sl1)
    if struct_h1=="bullish": bull+=12
    if struct_h1=="bearish": bear+=12

    # ── SMC M15 ──────────
    sh15,sl15=swing_pts(m15,5)
    bos=detect_bos(m15,sh15,sl15)
    sw_bull,sw_bear,rsn_b,rsn_s=sweep_rejection(m15,sh15,sl15)
    obs_b=find_ob(m15,"bull"); obs_s=find_ob(m15,"bear")
    fvg_b=find_fvg(m15,"bull"); fvg_s=find_fvg(m15,"bear")

    if bos["bb"]:  bull+=15
    if bos["cb"]:  bull+=10
    if bos["bs"]:  bear+=15
    if bos["cs"]:  bear+=10
    if sw_bull:    bull+=18
    if sw_bear:    bear+=18
    if obs_b:      bull+=7
    if obs_s:      bear+=7
    if fvg_b:      bull+=4
    if fvg_s:      bear+=4

    # ── Candle konfirmasi terakhir ──
    # Bullish engulfing / hammer → bull
    body=L15["close"]-L15["open"]
    prev_body=P15["close"]-P15["open"]
    if body>0 and body>abs(prev_body)*0.8: bull+=6
    if body<0 and abs(body)>abs(prev_body)*0.8: bear+=6
    lower_wick=min(L15["open"],L15["close"])-L15["low"]
    upper_wick=L15["high"]-max(L15["open"],L15["close"])
    if lower_wick>abs(body)*1.5: bull+=6   # hammer
    if upper_wick>abs(body)*1.5: bear+=6   # shooting star

    # ── Tentukan arah ────
    direction="bull" if bull>=bear else "bear"
    raw_conf=bull if direction=="bull" else bear
    confidence=min(int(raw_conf/160*100),99)

    obs  = obs_b if direction=="bull" else obs_s
    fvgs = fvg_b if direction=="bull" else fvg_s
    swept= sw_bull if direction=="bull" else sw_bear

    setup=build_setup(m15, direction, sh15, sl15, obs, fvgs, atr_val, swept)
    if setup is None: return None

    # Narasi
    why=[]
    if struct_h1!="ranging":           why.append(f"H1:{struct_h1.upper()}")
    if L1["close"]>L1["ema200"]:       why.append("AboveEMA200")
    else:                               why.append("BelowEMA200")
    if bos["bb"] or bos["bs"]:         why.append("BOS✔")
    if bos["cb"] or bos["cs"]:         why.append("CHoCH✔")
    if sw_bull:                         why.append(rsn_b)
    if sw_bear:                         why.append(rsn_s)
    if obs:   why.append(f"OB:{obs[-1]['bot']:.4g}–{obs[-1]['top']:.4g}")
    if fvgs:  why.append(f"FVG:{fvgs[-1]['bot']:.4g}–{fvgs[-1]['top']:.4g}")
    mc=(direction=="bull" and L15["mh"]>0 and P15["mh"]<=0) or \
       (direction=="bear" and L15["mh"]<0 and P15["mh"]>=0)
    if mc: why.append("MACD-X✔")
    why.append(f"RSI:{rv:.0f}")
    why.append(f"[{setup['entry_reason']}]")

    return {
        "symbol":"", "decision":"BUY" if direction=="bull" else "SELL",
        "confidence":confidence, "price":L15["close"],
        "entry":setup["entry"], "sl":setup["sl"], "tp":setup["tp"],
        "rr":setup["rr"], "etype":setup["etype"],
        "conf_lvl":setup["conf_lvl"], "reason":" | ".join(why),
        "rsi":round(rv,1),
    }

def analyze(symbol):
    try:
        h1=get_klines(symbol,"1h",250)
        m15=get_klines(symbol,"15m",250)
        if h1.empty or m15.empty: return None
        r=score_and_analyze(h1,m15)
        if r: r["symbol"]=symbol
        return r
    except Exception as e:
        log.debug(f"[analyze] {symbol}: {e}")
        return None


# ═════════════════════════════════════════════
# SCAN — 1 sinyal terbaik
# ═════════════════════════════════════════════
def run_scan_once(chat_id):
    tg_send(chat_id, f"🔍 Scanning {TOP_N_COINS} koin...")
    try:
        symbols=get_top_coins()
    except Exception as e:
        tg_send(chat_id,f"⚠️ Binance error: <code>{str(e)[:150]}</code>")
        return None

    if not symbols:
        tg_send(chat_id,"⚠️ Semua koin diban. Ketik /resetban untuk reset.")
        return None

    results=[]
    for idx,sym in enumerate(symbols,1):
        log.info(f"[{idx:02d}/{len(symbols)}] {sym}")
        r=analyze(sym)
        if r: results.append(r)
        time.sleep(0.08)

    if not results:
        tg_send(chat_id,"⚠️ Tidak ada sinyal valid dari semua koin.")
        return None

    results.sort(key=lambda x:(x["confidence"],x["rr"]),reverse=True)
    best=results[0]
    log.info(f"Sinyal terbaik: {best['symbol']} {best['decision']} conf={best['confidence']}%")
    return best


# ═════════════════════════════════════════════
# MONITORING
# ═════════════════════════════════════════════
def monitor_trade(chat_id, signal):
    global timeout_flag
    sym=signal["symbol"]; is_buy=signal["decision"]=="BUY"
    entry=signal["entry"]; sl_p=signal["sl"]; tp_p=signal["tp"]

    # Semua entry sekarang adalah MARKET
    actual=get_price(sym) or entry

    # Validasi: jika harga sudah melewati TP sebelum bisa masuk → skip
    if is_buy and actual>=tp_p:
        tg_send(chat_id,f"⏭ {sym}: Harga <code>{actual:.6g}</code> sudah di atas TP <code>{tp_p:.6g}</code>. Skip.")
        return "no_entry"
    if not is_buy and actual<=tp_p:
        tg_send(chat_id,f"⏭ {sym}: Harga <code>{actual:.6g}</code> sudah di bawah TP <code>{tp_p:.6g}</code>. Skip.")
        return "no_entry"
    # Validasi: jika harga sudah di SL
    if is_buy and actual<=sl_p:
        tg_send(chat_id,f"⏭ {sym}: Harga <code>{actual:.6g}</code> sudah di bawah SL <code>{sl_p:.6g}</code>. Skip.")
        return "no_entry"
    if not is_buy and actual>=sl_p:
        tg_send(chat_id,f"⏭ {sym}: Harga <code>{actual:.6g}</code> sudah di atas SL <code>{sl_p:.6g}</code>. Skip.")
        return "no_entry"

    risk=abs(actual-sl_p)
    reward=abs(tp_p-actual)
    rr_actual=round(reward/risk,2) if risk>0 else 0

    tg_send(chat_id,
        f"⚡ <b>ENTRY — {sym}</b>\n"
        f"Arah      : <b>{signal['decision']}</b>\n"
        f"Entry     : <code>{actual:.6g}</code>\n"
        f"✅ TP     : <code>{tp_p:.6g}</code>\n"
        f"🛑 SL     : <code>{sl_p:.6g}</code> (struktur invalid jika kena)\n"
        f"⚖️ RR     : 1:{rr_actual}\n"
        f"📝 Alasan : {signal['reason']}\n\n"
        f"📡 Monitor tiap {MONITOR_SLEEP}s... /timeout untuk skip.")

    last_log=time.time(); log_interval=90

    while True:
        if timeout_flag:
            timeout_flag=False
            price=get_price(sym) or 0
            tg_send(chat_id,
                f"⏭ <b>Timeout</b> — {sym} dihentikan.\n"
                f"Harga terakhir: <code>{price:.6g}</code>")
            return "timeout"

        price=get_price(sym)
        if price is None:
            time.sleep(MONITOR_SLEEP)
            continue

        if is_buy:
            if price>=tp_p:
                profit_pct=round((price-actual)/actual*100,2)
                tg_send(chat_id,
                    f"🎯 <b>TAKE PROFIT — {sym}</b> 🎉\n"
                    f"Harga: <code>{price:.6g}</code> ≥ TP: <code>{tp_p:.6g}</code>\n"
                    f"Profit: +{profit_pct}%")
                return "tp"
            if price<=sl_p:
                loss_pct=round((actual-price)/actual*100,2)
                tg_send(chat_id,
                    f"🛑 <b>STOP LOSS — {sym}</b>\n"
                    f"Harga: <code>{price:.6g}</code> ≤ SL: <code>{sl_p:.6g}</code>\n"
                    f"Loss: -{loss_pct}% | Struktur bullish invalid.")
                return "sl"
        else:
            if price<=tp_p:
                profit_pct=round((actual-price)/actual*100,2)
                tg_send(chat_id,
                    f"🎯 <b>TAKE PROFIT — {sym}</b> 🎉\n"
                    f"Harga: <code>{price:.6g}</code> ≤ TP: <code>{tp_p:.6g}</code>\n"
                    f"Profit: +{profit_pct}%")
                return "tp"
            if price>=sl_p:
                loss_pct=round((price-actual)/actual*100,2)
                tg_send(chat_id,
                    f"🛑 <b>STOP LOSS — {sym}</b>\n"
                    f"Harga: <code>{price:.6g}</code> ≥ SL: <code>{sl_p:.6g}</code>\n"
                    f"Loss: -{loss_pct}% | Struktur bearish invalid.")
                return "sl"

        if time.time()-last_log>=log_interval:
            pct_tp=abs(tp_p-price)/abs(tp_p-actual)*100 if abs(tp_p-actual)>0 else 0
            tg_send(chat_id,
                f"📊 <b>Update {sym}</b>\n"
                f"Harga: <code>{price:.6g}</code>\n"
                f"TP   : <code>{tp_p:.6g}</code> (sisa {pct_tp:.1f}%)\n"
                f"SL   : <code>{sl_p:.6g}</code>")
            last_log=time.time()

        time.sleep(MONITOR_SLEEP)


# ═════════════════════════════════════════════
# STATISTIK & FORMAT
# ═════════════════════════════════════════════
def update_stats(result):
    with stat_lock:
        stats["total"]+=1
        if result in stats: stats[result]+=1

def fmt_stats():
    with stat_lock:
        t=stats["total"]; tp=stats["tp"]
        sl=stats["sl"]; ne=stats["no_entry"]; to=stats["timeout"]
    if t==0: return "Belum ada simulasi."
    wr=tp/(tp+sl)*100 if (tp+sl)>0 else 0
    return (
        f"📊 <b>Statistik Simulasi</b>\n\n"
        f"Total      : {t}\n"
        f"🎯 TP      : {tp} ({tp/t*100:.1f}%)\n"
        f"🛑 SL      : {sl} ({sl/t*100:.1f}%)\n"
        f"❌ No Entry: {ne}\n"
        f"⏭ Timeout : {to}\n"
        f"📈 Win Rate: <b>{wr:.1f}%</b> (dari {tp+sl} trade)\n\n"
        f"🚫 Banned  : {len(banned_coins)}"
    )

def fmt_signal_msg(sig):
    em="🟢" if sig["decision"]=="BUY" else "🔴"
    bar="█"*(sig["confidence"]//10)+"░"*(10-sig["confidence"]//10)
    return (
        f"📡 <b>SINYAL TERBAIK</b>\n\n"
        f"{em} <b>{sig['symbol']}</b>\n"
        f"Arah       : <b>{sig['decision']}</b>\n"
        f"Confidence : <b>{sig['confidence']}%</b> {bar}\n"
        f"Harga kini : <code>{sig['price']:.6g}</code>\n"
        f"⚡ Entry   : <code>{sig['entry']:.6g}</code> (MARKET)\n"
        f"✅ TP      : <code>{sig['tp']:.6g}</code>\n"
        f"🛑 SL      : <code>{sig['sl']:.6g}</code>\n"
        f"⚖️ RR      : <b>1:{sig['rr']}</b>\n"
        f"📝 Analisis: {sig['reason']}"
    )


# ═════════════════════════════════════════════
# SIMULATION LOOP
# ═════════════════════════════════════════════
def simulation_loop(chat_id):
    global auto_mode, timeout_flag
    tg_send(chat_id,
        "🤖 <b>Simulasi dimulai!</b>\n"
        "Scan → Entry → Monitor TP/SL → Stats → Scan lagi\n"
        "/stop untuk berhenti | /timeout untuk skip monitoring")

    while auto_mode:
        timeout_flag=False
        signal=run_scan_once(chat_id)
        if not auto_mode: break

        if signal is None:
            tg_send(chat_id,"⚠️ Tidak ada sinyal. Retry 30 detik...")
            for _ in range(30):
                if not auto_mode: break
                time.sleep(1)
            continue

        tg_send(chat_id, fmt_signal_msg(signal))
        sym=signal["symbol"]
        with ban_lock:
            banned_coins.add(sym)

        result=monitor_trade(chat_id, signal)
        update_stats(result)

        emoji={"tp":"🎯","sl":"🛑","no_entry":"❌","timeout":"⏭"}.get(result,"❓")
        label={"tp":"TAKE PROFIT","sl":"STOP LOSS",
               "no_entry":"NO ENTRY","timeout":"TIMEOUT"}.get(result,result.upper())
        tg_send(chat_id,f"{emoji} <b>{label}</b> — {sym}\n\n"+fmt_stats())

        if not auto_mode: break
        for _ in range(5):
            if not auto_mode: break
            time.sleep(1)

    tg_send(chat_id,"⏹ <b>Simulasi dihentikan.</b>\n\n"+fmt_stats())


# ═════════════════════════════════════════════
# PESAN STATIS
# ═════════════════════════════════════════════
GREETING=(
    "👋 <b>SMC Simulasi Trading Bot v4</b>\n\n"
    "Scan koin → sinyal terbaik → simulasi trade nyata\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "/start    — Menu ini\n"
    "/auto     — Mulai simulasi\n"
    "/stop     — Hentikan\n"
    "/timeout  — Skip monitoring aktif\n"
    "/stats    — Statistik TP/SL/Winrate\n"
    "/banned   — Daftar koin ban\n"
    "/resetban — Hapus semua ban\n"
    "/info     — Detail metode\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "⚠️ <i>Simulasi saja — bukan saran finansial.</i>"
)
INFO_MSG=(
    "ℹ️ <b>Metode v4</b>\n\n"
    "<b>Entry:</b> Selalu MARKET (harga sekarang)\n"
    "→ Tidak ada limit order yang bikin No Entry\n\n"
    "<b>SL:</b> Swing low/high struktural + buffer ATR kecil\n"
    "→ Jika kena SL = struktur sudah rusak secara objektif\n\n"
    "<b>TP:</b> Swing high/low berikutnya (level nyata)\n"
    "→ Bukan target arbitrary\n\n"
    "<b>Filter sebelum entry:</b>\n"
    "→ Harga tidak boleh sudah melewati TP\n"
    "→ Harga tidak boleh sudah di SL\n\n"
    f"Min RR: 1:{MIN_RR} | TF: H1+M15"
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

                if text in ("/start","start"):       tg_send(chat_id,GREETING)
                elif text in ("/info","info"):        tg_send(chat_id,INFO_MSG)
                elif text in ("/stats","stats"):      tg_send(chat_id,fmt_stats())
                elif text in ("/banned","banned"):
                    with ban_lock: b=sorted(banned_coins)
                    tg_send(chat_id,
                        f"🚫 <b>Banned ({len(b)}):</b>\n"+", ".join(b) if b
                        else "✅ Belum ada ban.")
                elif text in ("/resetban","resetban"):
                    with ban_lock: n=len(banned_coins); banned_coins.clear()
                    tg_send(chat_id,f"✅ Ban direset ({n} dihapus).")
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
