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
MIN_RR          = 2.0
MONITOR_SLEEP   = 2
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
# TAHAP 2: ANALISIS ULANG — cari TP/SL untuk
#          posisi COUNTER (dibalik dari sinyal)
# ═════════════════════════════════════════════
def analyze_counter_setup(df_h1, df_m15, counter_dir, entry_price):
    """
    Setelah menentukan counter direction, analisis ulang chart
    untuk menemukan TP dan SL yang tepat berdasarkan:
    - Support/Resistance (SNR)
    - Supply/Demand zone (SND)
    - FVG (Fair Value Gap)
    - Equal Highs/Lows (liquidity pools)
    - Swing structure

    counter_dir: "bull" (counter dari sinyal bear → kita BUY)
                 "bear" (counter dari sinyal bull → kita SELL)
    entry_price: harga aktual saat masuk
    """
    h1=build_df(df_h1); m15=build_df(df_m15)
    if h1 is None or m15 is None: return None

    L15=m15.iloc[-1]
    atr_val=max(L15["atr"], entry_price*0.003)
    sh15,sl15=swing_pts(m15,5)
    sh1,sl1=swing_pts(h1,5)

    reasons=[]

    if counter_dir=="bear":
        # Kita SELL di entry_price
        # SL: di atas resistance/supply terdekat
        # TP: support/demand terkuat di bawah

        # Cari SL: resistance terdekat di atas entry
        # 1. Dari swing high M15
        res_above=[df_m15["high"].iloc[i]
                   for i in sh15
                   if df_m15["high"].iloc[i] > entry_price]
        # 2. Dari supply zone
        supply_zones=find_supply_demand(m15,"supply")
        for z in supply_zones:
            if z["high"]>entry_price:
                res_above.append(z["high"])
        # 3. Dari equal highs (liquidity)
        eq_highs=find_equal_highs_lows(m15,"high")
        res_above+=[h for h in eq_highs if h>entry_price]
        # 4. Dari H1 swing high
        res_above+=[df_h1["high"].iloc[i]
                    for i in sh1
                    if df_h1["high"].iloc[i]>entry_price]

        if res_above:
            nearest_res=min(res_above)
            sl_price=nearest_res+atr_val*0.3
            reasons.append(f"SL di atas resistance {nearest_res:.4g}")
        else:
            sl_price=entry_price+atr_val*2.5
            reasons.append("SL ATR-based (no resistance found)")

        risk=abs(sl_price-entry_price)
        min_tp=entry_price-risk*MIN_RR

        # Cari TP: support/demand terkuat di bawah
        sup_below=[df_m15["low"].iloc[i]
                   for i in sl15
                   if df_m15["low"].iloc[i] < entry_price-risk*0.5]
        # Demand zones
        demand_zones=find_supply_demand(m15,"demand")
        for z in demand_zones:
            if z["bot"]<entry_price-risk*0.5:
                sup_below.append(z["bot"])
        # Equal lows (liquidity target)
        eq_lows=find_equal_highs_lows(m15,"low")
        sup_below+=[l for l in eq_lows if l<entry_price-risk*0.5]
        # FVG bearish
        fvgs=find_fvg(m15,"bear")
        for fvg in fvgs:
            if fvg["mid"]<entry_price-risk*0.5:
                sup_below.append(fvg["mid"])
        # H1 support
        sup_below+=[df_h1["low"].iloc[i]
                    for i in sl1
                    if df_h1["low"].iloc[i]<entry_price-risk*0.5]

        if sup_below:
            best_tp=max(sup_below)  # support terdekat di bawah
            # Pastikan RR >= MIN_RR
            if abs(best_tp-entry_price)/risk >= MIN_RR:
                tp_price=best_tp
                reasons.append(f"TP di support/demand {best_tp:.4g}")
            else:
                tp_price=min_tp
                reasons.append(f"TP min RR 1:{MIN_RR}")
        else:
            tp_price=min_tp
            reasons.append(f"TP min RR 1:{MIN_RR}")

    else:
        # counter_dir=="bull" → kita BUY di entry_price
        # SL: di bawah support/demand terdekat
        # TP: resistance/supply terkuat di atas

        sup_below=[df_m15["low"].iloc[i]
                   for i in sl15
                   if df_m15["low"].iloc[i]<entry_price]
        demand_zones=find_supply_demand(m15,"demand")
        for z in demand_zones:
            if z["low"]<entry_price:
                sup_below.append(z["low"])
        eq_lows=find_equal_highs_lows(m15,"low")
        sup_below+=[l for l in eq_lows if l<entry_price]
        sup_below+=[df_h1["low"].iloc[i]
                    for i in sl1
                    if df_h1["low"].iloc[i]<entry_price]

        if sup_below:
            nearest_sup=max(sup_below)
            sl_price=nearest_sup-atr_val*0.3
            reasons.append(f"SL di bawah support {nearest_sup:.4g}")
        else:
            sl_price=entry_price-atr_val*2.5
            reasons.append("SL ATR-based (no support found)")

        risk=abs(entry_price-sl_price)
        min_tp=entry_price+risk*MIN_RR

        res_above=[df_m15["high"].iloc[i]
                   for i in sh15
                   if df_m15["high"].iloc[i]>entry_price+risk*0.5]
        supply_zones=find_supply_demand(m15,"supply")
        for z in supply_zones:
            if z["top"]>entry_price+risk*0.5:
                res_above.append(z["top"])
        eq_highs=find_equal_highs_lows(m15,"high")
        res_above+=[h for h in eq_highs if h>entry_price+risk*0.5]
        fvgs=find_fvg(m15,"bull")
        for fvg in fvgs:
            if fvg["mid"]>entry_price+risk*0.5:
                res_above.append(fvg["mid"])
        res_above+=[df_h1["high"].iloc[i]
                    for i in sh1
                    if df_h1["high"].iloc[i]>entry_price+risk*0.5]

        if res_above:
            best_tp=min(res_above)
            if abs(best_tp-entry_price)/risk>=MIN_RR:
                tp_price=best_tp
                reasons.append(f"TP di resistance/supply {best_tp:.4g}")
            else:
                tp_price=min_tp
                reasons.append(f"TP min RR 1:{MIN_RR}")
        else:
            tp_price=min_tp
            reasons.append(f"TP min RR 1:{MIN_RR}")

    risk=abs(entry_price-sl_price)
    reward=abs(tp_price-entry_price)
    if risk==0: return None
    rr=round(reward/risk,2)
    if rr<MIN_RR: return None

    return {
        "sl"    : round(sl_price,8),
        "tp"    : round(tp_price,8),
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
    3. Balik arah → counter direction
    4. Analisis ulang TP/SL untuk posisi counter
    """
    try:
        df_h1=get_klines(symbol,"1h",250)
        df_m15=get_klines(symbol,"15m",250)
        if df_h1.empty or df_m15.empty: return None

        # Tahap 1: scoring arah
        score=score_direction(df_h1,df_m15)
        if score is None: return None

        # Tahap 2: balik arah
        original_dir=score["direction"]
        counter_dir="bear" if original_dir=="bull" else "bull"
        counter_decision="SELL" if counter_dir=="bear" else "BUY"
        entry_price=score["price"]

        # Tahap 3: analisis ulang untuk counter TP/SL
        setup=analyze_counter_setup(df_h1,df_m15,counter_dir,entry_price)
        if setup is None: return None

        return {
            "symbol"       : symbol,
            "original_dir" : original_dir,
            "decision"     : counter_decision,
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

    # Ranking: confidence DESC → rr DESC
    results.sort(key=lambda x:(x["confidence"],x["rr"]),reverse=True)
    best=results[0]
    log.info(f"Best: {best['symbol']} {best['decision']} "
             f"conf={best['confidence']}% RR=1:{best['rr']}")
    return best


# ═════════════════════════════════════════════
# MONITORING
# ═════════════════════════════════════════════
def monitor_trade(chat_id, signal):
    global timeout_flag
    sym=signal["symbol"]
    is_buy=signal["decision"]=="BUY"
    entry=signal["entry"]
    sl_p=signal["sl"]
    tp_p=signal["tp"]

    actual=get_price(sym) or entry

    # Validasi pre-entry
    if is_buy:
        if actual>=tp_p:
            tg_send(chat_id,f"⏭ {sym}: Harga sudah di atas TP. Skip.")
            return "no_entry"
        if actual<=sl_p:
            tg_send(chat_id,f"⏭ {sym}: Harga sudah di bawah SL. Skip.")
            return "no_entry"
    else:
        if actual<=tp_p:
            tg_send(chat_id,f"⏭ {sym}: Harga sudah di bawah TP. Skip.")
            return "no_entry"
        if actual>=sl_p:
            tg_send(chat_id,f"⏭ {sym}: Harga sudah di atas SL. Skip.")
            return "no_entry"

    risk=abs(actual-sl_p)
    reward=abs(tp_p-actual)
    rr_now=round(reward/risk,2) if risk>0 else 0

    orig_label="BULLISH" if signal["original_dir"]=="bull" else "BEARISH"

    tg_send(chat_id,
        f"⚡ <b>COUNTER ENTRY — {sym}</b>\n\n"
        f"📊 Analisis chart: <b>{orig_label}</b>\n"
        f"↩️ Eksekusi      : <b>{signal['decision']}</b> (dibalik)\n\n"
        f"💰 Entry  : <code>{actual:.6g}</code>\n"
        f"✅ TP     : <code>{tp_p:.6g}</code>\n"
        f"🛑 SL     : <code>{sl_p:.6g}</code>\n"
        f"⚖️ RR     : 1:{rr_now}\n\n"
        f"📝 TP/SL basis:\n{signal['tp_sl_reason']}\n\n"
        f"📡 Monitor tiap {MONITOR_SLEEP}s... /timeout untuk skip.")

    last_log=time.time(); log_interval=90

    while True:
        if timeout_flag:
            timeout_flag=False
            price=get_price(sym) or 0
            tg_send(chat_id,
                f"⏭ <b>Timeout</b> — {sym}\n"
                f"Harga: <code>{price:.6g}</code>")
            return "timeout"

        price=get_price(sym)
        if price is None:
            time.sleep(MONITOR_SLEEP); continue

        hit_tp=False; hit_sl=False
        if is_buy:
            hit_tp=price>=tp_p
            hit_sl=price<=sl_p
        else:
            hit_tp=price<=tp_p
            hit_sl=price>=sl_p

        if hit_tp:
            pct=abs(price-actual)/actual*100
            tg_send(chat_id,
                f"🎯 <b>TAKE PROFIT — {sym}</b> 🎉\n"
                f"Harga: <code>{price:.6g}</code>\n"
                f"TP   : <code>{tp_p:.6g}</code>\n"
                f"Profit: +{pct:.2f}%")
            return "tp"

        if hit_sl:
            pct=abs(price-actual)/actual*100
            tg_send(chat_id,
                f"🛑 <b>STOP LOSS — {sym}</b>\n"
                f"Harga: <code>{price:.6g}</code>\n"
                f"SL   : <code>{sl_p:.6g}</code>\n"
                f"Loss : -{pct:.2f}%")
            return "sl"

        if time.time()-last_log>=log_interval:
            pct_tp=abs(tp_p-price)/abs(tp_p-actual)*100 if abs(tp_p-actual)>0 else 0
            tg_send(chat_id,
                f"📊 <b>Update {sym}</b>\n"
                f"Harga : <code>{price:.6g}</code>\n"
                f"TP    : <code>{tp_p:.6g}</code> (sisa {pct_tp:.1f}%)\n"
                f"SL    : <code>{sl_p:.6g}</code>")
            last_log=time.time()

        time.sleep(MONITOR_SLEEP)


# ═════════════════════════════════════════════
# STATISTIK
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
    em="🔴" if sig["decision"]=="SELL" else "🟢"
    bar="█"*(sig["confidence"]//10)+"░"*(10-sig["confidence"]//10)
    orig="BULLISH" if sig["original_dir"]=="bull" else "BEARISH"
    return (
        f"📡 <b>COUNTER SIGNAL DITEMUKAN</b>\n\n"
        f"Koin        : <b>{sig['symbol']}</b>\n"
        f"Chart bicara: <b>{orig}</b> (confidence {sig['confidence']}% {bar})\n"
        f"↩️ Eksekusi : {em} <b>{sig['decision']}</b> (counter)\n\n"
        f"💰 Entry  : <code>{sig['entry']:.6g}</code>\n"
        f"✅ TP     : <code>{sig['tp']:.6g}</code>\n"
        f"🛑 SL     : <code>{sig['sl']:.6g}</code>\n"
        f"⚖️ RR     : <b>1:{sig['rr']}</b>\n"
        f"RSI         : {sig['rsi']} | H1: {sig['struct_h1'].upper()}\n\n"
        f"📝 Basis TP/SL:\n{sig['tp_sl_reason']}"
    )


# ═════════════════════════════════════════════
# SIMULATION LOOP
# ═════════════════════════════════════════════
def simulation_loop(chat_id):
    global auto_mode, timeout_flag
    tg_send(chat_id,
        "🤖 <b>Counter Trading Simulasi dimulai!</b>\n\n"
        "Alur:\n"
        "1. Scan & analisis 50 koin\n"
        "2. Temukan sinyal terkuat (misal: BULL)\n"
        "3. Balik arah → eksekusi SELL\n"
        "4. Analisis ulang TP/SL via SNR/SND/SMC\n"
        "5. Monitor hingga TP atau SL\n\n"
        "/stop untuk berhenti | /timeout untuk skip")

    while auto_mode:
        timeout_flag=False

        signal=run_scan_once(chat_id)
        if not auto_mode: break

        if signal is None:
            tg_send(chat_id,"⚠️ Tidak ada setup. Retry 30 detik...")
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
    "👋 <b>SMC Counter Trading Bot v5</b>\n\n"
    "Strategi: Analisis chart → temukan sinyal → BALIK arah\n"
    "→ Cari TP/SL via SNR + SND + SMC + FVG\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "/start    — Menu ini\n"
    "/auto     — Mulai simulasi counter trading\n"
    "/stop     — Hentikan simulasi\n"
    "/timeout  — Skip monitoring, lanjut scan\n"
    "/stats    — Statistik TP/SL/Winrate\n"
    "/banned   — Daftar koin ban\n"
    "/resetban — Hapus semua ban\n"
    "/info     — Detail strategi\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "⚠️ <i>Simulasi saja — bukan saran finansial.</i>"
)
INFO_MSG=(
    "ℹ️ <b>Strategi Counter Trading v5</b>\n\n"
    "<b>Kenapa dibalik?</b>\n"
    "Ketika banyak indikator menunjuk BULL,\n"
    "artinya retail trader sudah banyak BUY.\n"
    "Institusi sering mengambil likuiditas mereka\n"
    "dengan mendorong harga berlawanan dulu.\n\n"
    "<b>Flow:</b>\n"
    "1. Analisis normal → sinyal BULL/BEAR\n"
    "2. Balik → eksekusi SELL/BUY\n"
    "3. Analisis ulang untuk TP/SL:\n"
    "   • SNR: Support & Resistance terdekat\n"
    "   • SND: Supply & Demand zone\n"
    "   • FVG: Fair Value Gap\n"
    "   • Equal Highs/Lows: Liquidity pools\n"
    "   • H1 swing structure\n\n"
    f"Min RR: 1:{MIN_RR} | SL: di luar S/R struktural"
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
                    tg_send(chat_id,INFO_MSG)
                elif text in ("/stats","stats"):
                    tg_send(chat_id,fmt_stats())
                elif text in ("/banned","banned"):
                    with ban_lock: b=sorted(banned_coins)
                    tg_send(chat_id,
                        f"🚫 <b>Banned ({len(b)}):</b>\n"+", ".join(b)
                        if b else "✅ Belum ada ban.")
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
