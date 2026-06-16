#!/usr/bin/env python3
"""
SMC Simulasi Trading Bot — Logika Entry/TP/SL Diperbaiki
Render.com Web Service | python main.py
"""

# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
ALLOWED_USER_ID = 8041197505
MAX_PRICE       = 80.0
TOP_N_COINS     = 50
MIN_RR          = 2.0
MONITOR_SLEEP   = 2
ENTRY_TIMEOUT   = 3600
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
    ptp = f"{tp/t*100:.1f}%" if t else "–"
    psl = f"{sl/t*100:.1f}%" if t else "–"
    return (f"<h3>SMC Sim Bot</h3>"
            f"<p>Auto:{auto_mode} | Banned:{len(banned_coins)}</p>"
            f"<p>Total:{t} TP:{tp}({ptp}) SL:{sl}({psl})</p>"), 200

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
    except Exception as e:
        log.warning(f"[TG] {e}")
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
    try:
        d = fapi_get("/fapi/v1/ticker/price", {"symbol": symbol})
        return float(d["price"])
    except Exception as e:
        log.warning(f"[price] {symbol}: {e}")
        return None

def get_klines(symbol, interval, limit=300):
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
    d=s.diff(); g=d.clip(lower=0).rolling(n).mean()
    l=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def macd(s):
    line=ema(s,12)-ema(s,26); sig=ema(line,9)
    return line, sig, line-sig

def atr_s(df, n=14):
    tr=pd.concat([df["high"]-df["low"],
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def indicators(df):
    if len(df)<60: return None
    df=df.copy()
    df["ema9"]   = ema(df["close"],9)
    df["ema21"]  = ema(df["close"],21)
    df["ema50"]  = ema(df["close"],50)
    df["ema200"] = ema(df["close"],200) if len(df)>=200 else ema(df["close"],50)
    df["rsi"]    = rsi(df["close"])
    df["ml"],df["ms"],df["mh"] = macd(df["close"])
    df["atr"]    = atr_s(df)
    df["vol_sma"]= df["volume"].rolling(20).mean()
    bm=df["close"].rolling(20).mean(); bs=df["close"].rolling(20).std()
    df["bb_up"]=bm+2*bs; df["bb_lo"]=bm-2*bs; df["bb_mid"]=bm
    return df.dropna()


# ═════════════════════════════════════════════
# STRUKTUR PASAR & SMC
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
    """Deteksi BOS dan CHoCH untuk konfirmasi arah."""
    res={"bos_bull":False,"bos_bear":False,"choch_bull":False,"choch_bear":False}
    p=df["close"].iloc[-1]
    if len(sh)>=2:
        ph=df["high"].iloc[sh[-2]]; lh=df["high"].iloc[sh[-1]]
        if p>ph: res["bos_bull" if lh>ph else "choch_bull"]=True
    if len(sl)>=2:
        pl=df["low"].iloc[sl[-2]]; ll=df["low"].iloc[sl[-1]]
        if p<pl: res["bos_bear" if ll<pl else "choch_bear"]=True
    return res

def find_ob(df, direction, lb=40):
    """
    Order Block = candle terakhir yang berlawanan sebelum impulse besar.
    Ini zona di mana institusi menempatkan order → harga sering kembali ke sini.
    """
    sub=df.iloc[-lb:]; avg=(sub["close"]-sub["open"]).abs().mean()
    obs=[]
    for i in range(1, len(sub)-1):
        c,nx=sub.iloc[i],sub.iloc[i+1]
        body=abs(nx["close"]-nx["open"])
        if body<avg*1.3: continue
        if direction=="bull" and c["close"]<c["open"] and nx["close"]>nx["open"]:
            obs.append({"top":c["high"],"bot":c["low"],
                        "mid":(c["high"]+c["low"])/2,"idx":i})
        if direction=="bear" and c["close"]>c["open"] and nx["close"]<nx["open"]:
            obs.append({"top":c["high"],"bot":c["low"],
                        "mid":(c["high"]+c["low"])/2,"idx":i})
    return obs

def find_fvg(df, direction, lb=40):
    """Fair Value Gap — celah harga yang cenderung diisi sebelum lanjut."""
    sub=df.iloc[-lb:]; out=[]
    for i in range(len(sub)-2):
        c0,c2=sub.iloc[i],sub.iloc[i+2]
        if direction=="bull" and c2["low"]>c0["high"]:
            out.append({"top":c2["low"],"bot":c0["high"],
                        "mid":(c2["low"]+c0["high"])/2})
        if direction=="bear" and c2["high"]<c0["low"]:
            out.append({"top":c0["low"],"bot":c2["high"],
                        "mid":(c0["low"]+c2["high"])/2})
    return out

def find_equal_levels(df, kind="high", lb=50, tolerance=0.002):
    """
    Equal Highs/Lows = zona likuiditas yang sering di-sweep institusi.
    Setelah sweep → harga berbalik → entry di sana.
    """
    sub=df.iloc[-lb:]
    vals=sub["high"] if kind=="high" else sub["low"]
    levels=[]
    for i in range(len(vals)-1):
        for j in range(i+1, len(vals)):
            v1,v2=vals.iloc[i],vals.iloc[j]
            if v1>0 and abs(v1-v2)/v1 < tolerance:
                levels.append((v1+v2)/2)
    return list(set([round(l,6) for l in levels]))

def liquidity_sweep_confirmed(df, sh, sl):
    """
    Sweep + rejection yang terkonfirmasi:
    Harga menembus swing high/low lama lalu close kembali di dalam range.
    Ini adalah setup entry terkuat dalam SMC.
    """
    bull=bear=False; reason_b=reason_s=""
    if len(df)<3: return bull,bear,reason_b,reason_s
    cur,prv,prv2=df.iloc[-1],df.iloc[-2],df.iloc[-3]

    if sl:
        ll=df["low"].iloc[sl[-1]]
        # Sweep: prv menembus level lama, cur close di atasnya (rejection)
        if prv["low"]<ll and cur["close"]>ll and cur["close"]>cur["open"]:
            bull=True
            reason_b=f"Liquidity sweep di {ll:.4g} + rejection bullish"

    if sh:
        lh=df["high"].iloc[sh[-1]]
        if prv["high"]>lh and cur["close"]<lh and cur["close"]<cur["open"]:
            bear=True
            reason_s=f"Liquidity sweep di {lh:.4g} + rejection bearish"

    return bull,bear,reason_b,reason_s

def is_retracement_not_reversal(df, direction, sh, sl):
    """
    Bedakan retracement (sementara) vs reversal (balik arah).
    Retracement: pullback ke 38–62% Fibonacci dari impulse terakhir,
    struktur belum berubah (tidak ada BOS berlawanan).
    Return True jika ini retracement → bisa jadi zona entry.
    """
    if len(sh)<2 or len(sl)<2: return False
    bos=detect_bos(df,sh,sl)
    if direction=="bull":
        # Jika ada BOS bearish → ini bukan retracement, ini reversal
        if bos["bos_bear"] or bos["choch_bear"]: return False
        # Cek apakah pullback masih dalam 62% dari impulse terakhir
        if len(sh)>=1 and len(sl)>=1:
            swing_high=df["high"].iloc[sh[-1]]
            swing_low=df["low"].iloc[sl[-1]]
            current=df["close"].iloc[-1]
            fib62=swing_high-(swing_high-swing_low)*0.62
            if current>=fib62: return True  # masih dalam zona retracement
    else:
        if bos["bos_bull"] or bos["choch_bull"]: return False
        if len(sh)>=1 and len(sl)>=1:
            swing_high=df["high"].iloc[sh[-1]]
            swing_low=df["low"].iloc[sl[-1]]
            current=df["close"].iloc[-1]
            fib62=swing_low+(swing_high-swing_low)*0.62
            if current<=fib62: return True
    return False


# ═════════════════════════════════════════════
# LOGIKA ENTRY / SL / TP YANG BENAR
# ═════════════════════════════════════════════
def build_precise_setup(df, direction, sh, sl, obs, fvgs,
                        sweep_bull, sweep_bear, atr_val):
    """
    Logika entry/SL/TP yang diperbaiki:

    ENTRY:
    - Prioritas 1: Setelah liquidity sweep + rejection (entry terkuat)
    - Prioritas 2: Di zona OB yang bertepatan dengan FVG (konfluensi)
    - Prioritas 3: Di zona OB saja
    - Prioritas 4: Di zona FVG saja
    - Fallback: Harga sekarang dengan konfirmasi EMA

    SL:
    - BUY: Di bawah swing low TERBARU yang signifikan
             (bukan swing low sembarangan — harus swing low yang membatalkan analisis)
             Jika harga turun ke SL ini, itu berarti struktur bullish sudah rusak.
    - SELL: Di atas swing high terbaru yang signifikan

    TP:
    - BUY: Swing high terdekat di atas entry, atau next liquidity pool
    - SELL: Swing low terdekat di bawah entry, atau next liquidity pool
    - Dipastikan RR >= MIN_RR
    """
    price = df["close"].iloc[-1]

    # ── Tentukan SL yang benar ─────────────────────────────────────────
    # SL harus di luar struktur — di bawah swing low terakhir (buy)
    # atau di atas swing high terakhir (sell)
    # Buffer = ATR * 0.3 saja (kecil, hanya untuk noise)
    buf = atr_val * 0.3

    if direction == "bull":
        # SL = di bawah swing low terakhir yang valid
        if sl:
            # Ambil swing low terdekat di bawah harga sekarang
            valid_sl = [df["low"].iloc[i] for i in sl
                        if df["low"].iloc[i] < price]
            if valid_sl:
                structural_sl = max(valid_sl) - buf  # swing low terdekat
            else:
                structural_sl = price - atr_val * 2
        else:
            structural_sl = price - atr_val * 2

    else:  # bear
        if sh:
            valid_sh = [df["high"].iloc[i] for i in sh
                        if df["high"].iloc[i] > price]
            if valid_sh:
                structural_sl = min(valid_sh) + buf
            else:
                structural_sl = price + atr_val * 2
        else:
            structural_sl = price + atr_val * 2

    # ── Tentukan ENTRY ─────────────────────────────────────────────────
    entry       = None
    etype       = "MARKET"
    conf_lvl    = None
    entry_reason= ""
    zone_top    = zone_bot = None

    # Prioritas 1: Setelah sweep — entry di harga sekarang
    if direction == "bull" and sweep_bull:
        entry       = price
        etype       = "MARKET"
        entry_reason= "Post-sweep market entry"

    elif direction == "bear" and sweep_bear:
        entry       = price
        etype       = "MARKET"
        entry_reason= "Post-sweep market entry"

    # Prioritas 2: OB + FVG konfluensi — entry limit di zona
    if entry is None and obs and fvgs:
        for ob in reversed(obs):
            for fvg in reversed(fvgs):
                ot = min(ob["top"], fvg["top"])
                ob_= max(ob["bot"], fvg["bot"])
                if ot > ob_:  # ada overlap
                    if direction=="bull" and ob_ > structural_sl:
                        zone_top=ot; zone_bot=ob_
                        in_z = zone_bot<=price<=zone_top
                        entry     = price if in_z else zone_top
                        etype     = "MARKET" if in_z else "LIMIT"
                        conf_lvl  = None if in_z else zone_top
                        entry_reason = "OB+FVG konfluensi"
                        break
                    elif direction=="bear" and ob_ < structural_sl:
                        zone_top=ot; zone_bot=ob_
                        in_z = zone_bot<=price<=zone_top
                        entry     = price if in_z else zone_bot
                        etype     = "MARKET" if in_z else "LIMIT"
                        conf_lvl  = None if in_z else zone_bot
                        entry_reason = "OB+FVG konfluensi"
                        break
            if entry: break

    # Prioritas 3: OB saja
    if entry is None and obs:
        for ob in reversed(obs):
            if direction=="bull" and ob["bot"] > structural_sl:
                zone_top=ob["top"]; zone_bot=ob["bot"]
                in_z = zone_bot<=price<=zone_top
                entry     = price if in_z else zone_top
                etype     = "MARKET" if in_z else "LIMIT"
                conf_lvl  = None if in_z else zone_top
                entry_reason = "Order Block"
                break
            elif direction=="bear" and ob["top"] < structural_sl:
                zone_top=ob["top"]; zone_bot=ob["bot"]
                in_z = zone_bot<=price<=zone_top
                entry     = price if in_z else zone_bot
                etype     = "MARKET" if in_z else "LIMIT"
                conf_lvl  = None if in_z else zone_bot
                entry_reason = "Order Block"
                break

    # Prioritas 4: FVG saja
    if entry is None and fvgs:
        for fvg in reversed(fvgs):
            if direction=="bull" and fvg["bot"] > structural_sl:
                in_z = fvg["bot"]<=price<=fvg["top"]
                entry     = price if in_z else fvg["top"]
                etype     = "MARKET" if in_z else "LIMIT"
                conf_lvl  = None if in_z else fvg["top"]
                entry_reason = "Fair Value Gap"
                break
            elif direction=="bear" and fvg["top"] < structural_sl:
                in_z = fvg["bot"]<=price<=fvg["top"]
                entry     = price if in_z else fvg["bot"]
                etype     = "MARKET" if in_z else "LIMIT"
                conf_lvl  = None if in_z else fvg["bot"]
                entry_reason = "Fair Value Gap"
                break

    # Fallback: market entry harga sekarang
    if entry is None:
        entry = price
        etype = "MARKET"
        entry_reason = "Market entry (EMA konfirmasi)"

    # ── Tentukan TP ────────────────────────────────────────────────────
    # TP = level likuiditas berikutnya (swing high/low sebelumnya)
    # Pastikan RR >= MIN_RR, jika tidak → sesuaikan TP
    if direction == "bull":
        risk = abs(entry - structural_sl)
        min_tp = entry + risk * MIN_RR

        # Cari swing high di atas entry sebagai TP natural
        natural_tps = sorted([df["high"].iloc[i] for i in sh
                               if df["high"].iloc[i] > entry + risk * 0.5])
        if natural_tps:
            tp = natural_tps[0]
            if tp < min_tp:  # TP natural terlalu dekat → gunakan min_tp
                tp = min_tp
        else:
            tp = min_tp

        # Cek equal highs sebagai TP (zona likuiditas yang sering dituju)
        eq_highs = [l for l in find_equal_levels(df,"high")
                    if l > entry and l > min_tp*0.99]
        if eq_highs:
            eq_tp = min(eq_highs)
            if eq_tp > min_tp:
                tp = eq_tp  # TP ke zona likuiditas lebih akurat

    else:
        risk = abs(structural_sl - entry)
        min_tp = entry - risk * MIN_RR

        natural_tps = sorted([df["low"].iloc[i] for i in sl
                               if df["low"].iloc[i] < entry - risk * 0.5],
                              reverse=True)
        if natural_tps:
            tp = natural_tps[0]
            if tp > min_tp:
                tp = min_tp
        else:
            tp = min_tp

        eq_lows = [l for l in find_equal_levels(df,"low")
                   if l < entry and l < min_tp*1.01]
        if eq_lows:
            eq_tp = max(eq_lows)
            if eq_tp < min_tp:
                tp = eq_tp

    # Validasi final RR
    risk   = abs(entry - structural_sl)
    reward = abs(tp - entry)
    if risk == 0: return None
    rr = round(reward/risk, 2)
    if rr < MIN_RR:
        return None

    return {
        "entry"       : round(entry, 8),
        "sl"          : round(structural_sl, 8),
        "tp"          : round(tp, 8),
        "rr"          : rr,
        "etype"       : etype,
        "conf_lvl"    : round(conf_lvl, 8) if conf_lvl else None,
        "entry_reason": entry_reason,
    }


# ═════════════════════════════════════════════
# SCORING KOIN
# ═════════════════════════════════════════════
def score_coin(df_h1, df_m15):
    h1  = indicators(df_h1)
    m15 = indicators(df_m15)
    if h1 is None or m15 is None: return None

    L1=h1.iloc[-1]; P1=h1.iloc[-2]
    L15=m15.iloc[-1]; P15=m15.iloc[-2]

    bull_pts = bear_pts = 0

    # EMA H1 (trend utama)
    if L1["ema9"]>L1["ema21"]>L1["ema50"]:   bull_pts+=15
    elif L1["ema9"]>L1["ema21"]:              bull_pts+=8
    if L1["ema9"]<L1["ema21"]<L1["ema50"]:   bear_pts+=15
    elif L1["ema9"]<L1["ema21"]:              bear_pts+=8
    if L1["close"]>L1["ema200"]:              bull_pts+=8
    else:                                      bear_pts+=8

    # RSI M15
    rv=L15["rsi"]
    if rv<35:    bull_pts+=12  # oversold kuat
    elif rv<45:  bull_pts+=7
    if rv>65:    bear_pts+=12  # overbought kuat
    elif rv>55:  bear_pts+=7

    # MACD M15
    if L15["mh"]>0 and P15["mh"]<=0: bull_pts+=12
    elif L15["mh"]>0:                 bull_pts+=6
    if L15["mh"]<0 and P15["mh"]>=0: bear_pts+=12
    elif L15["mh"]<0:                 bear_pts+=6

    # EMA M15
    if L15["ema9"]>L15["ema21"]>L15["ema50"]: bull_pts+=10
    elif L15["ema9"]>L15["ema21"]:             bull_pts+=5
    if L15["ema9"]<L15["ema21"]<L15["ema50"]: bear_pts+=10
    elif L15["ema9"]<L15["ema21"]:             bear_pts+=5

    # Bollinger posisi
    if L15["close"]<L15["bb_lo"]:    bull_pts+=10  # di bawah lower → reversal
    elif L15["close"]<L15["bb_mid"]: bull_pts+=4
    if L15["close"]>L15["bb_up"]:    bear_pts+=10
    elif L15["close"]>L15["bb_mid"]: bear_pts+=4

    # Volume
    if L15["volume"]>L15["vol_sma"]*1.5:
        if L15["close"]>L15["open"]: bull_pts+=10
        else:                         bear_pts+=10
    elif L15["volume"]>L15["vol_sma"]:
        if L15["close"]>L15["open"]: bull_pts+=4
        else:                         bear_pts+=4

    # Market structure H1
    sh1,sl1=swing_pts(h1,5)
    struct_h1=mkt_struct(h1,sh1,sl1)
    if struct_h1=="bullish": bull_pts+=10
    if struct_h1=="bearish": bear_pts+=10

    # SMC M15
    sh15,sl15=swing_pts(m15,5)
    bos=detect_bos(m15,sh15,sl15)
    sw_bull,sw_bear,rsn_b,rsn_s=liquidity_sweep_confirmed(m15,sh15,sl15)
    obs_bull=find_ob(m15,"bull"); obs_bear=find_ob(m15,"bear")
    fvg_bull=find_fvg(m15,"bull"); fvg_bear=find_fvg(m15,"bear")

    # BOS/CHoCH adalah konfirmasi terkuat
    if bos["bos_bull"]:   bull_pts+=15
    if bos["choch_bull"]: bull_pts+=12
    if bos["bos_bear"]:   bear_pts+=15
    if bos["choch_bear"]: bear_pts+=12

    # Liquidity sweep = sinyal pembalikan terkuat
    if sw_bull: bull_pts+=18
    if sw_bear: bear_pts+=18

    if obs_bull: bull_pts+=8
    if obs_bear: bear_pts+=8
    if fvg_bull: bull_pts+=5
    if fvg_bear: bear_pts+=5

    # Retracement check — jika sedang retracement (bukan reversal)
    # maka tambah poin untuk arah utama
    is_retrace_bull = is_retracement_not_reversal(m15,"bull",sh15,sl15)
    is_retrace_bear = is_retracement_not_reversal(m15,"bear",sh15,sl15)
    if is_retrace_bull: bull_pts+=10
    if is_retrace_bear: bear_pts+=10

    direction = "bull" if bull_pts>=bear_pts else "bear"
    raw_conf  = bull_pts if direction=="bull" else bear_pts
    confidence= min(int(raw_conf/150*100), 99)

    obs  = obs_bull if direction=="bull" else obs_bear
    fvgs = fvg_bull if direction=="bull" else fvg_bear

    price   = L15["close"]
    atr_val = L15["atr"] if L15["atr"]>0 else price*0.005

    setup = build_precise_setup(
        m15, direction, sh15, sl15, obs, fvgs,
        sw_bull, sw_bear, atr_val
    )
    if setup is None: return None

    # Narasi alasan
    why=[]
    if struct_h1!="ranging":      why.append(f"H1:{struct_h1.upper()}")
    if bos["bos_bull"] or bos["bos_bear"]:     why.append("BOS✔")
    if bos["choch_bull"] or bos["choch_bear"]: why.append("CHoCH✔")
    if sw_bull: why.append(rsn_b)
    if sw_bear: why.append(rsn_s)
    if is_retrace_bull or is_retrace_bear:     why.append("Retracement✔")
    if obs:  why.append(f"OB:{obs[-1]['bot']:.4g}–{obs[-1]['top']:.4g}")
    if fvgs: why.append(f"FVG:{fvgs[-1]['bot']:.4g}–{fvgs[-1]['top']:.4g}")
    mc=(direction=="bull" and L15["mh"]>0 and P15["mh"]<=0) or \
       (direction=="bear" and L15["mh"]<0 and P15["mh"]>=0)
    if mc: why.append("MACD Cross✔")
    why.append(f"RSI:{rv:.0f}")
    why.append(f"[{setup['entry_reason']}]")

    return {
        "symbol"    : "",
        "decision"  : "BUY" if direction=="bull" else "SELL",
        "confidence": confidence,
        "price"     : price,
        "entry"     : setup["entry"],
        "sl"        : setup["sl"],
        "tp"        : setup["tp"],
        "rr"        : setup["rr"],
        "etype"     : setup["etype"],
        "conf_lvl"  : setup["conf_lvl"],
        "reason"    : " | ".join(why),
        "rsi"       : round(rv,1),
    }

def analyze(symbol):
    try:
        h1  = get_klines(symbol,"1h",300)
        m15 = get_klines(symbol,"15m",300)
        if h1.empty or m15.empty: return None
        r=score_coin(h1,m15)
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
        symbols = get_top_coins()
    except Exception as e:
        tg_send(chat_id, f"⚠️ Binance error: <code>{str(e)[:200]}</code>")
        return None

    if not symbols:
        tg_send(chat_id, "⚠️ Semua koin diban atau tidak ada data.")
        return None

    results=[]
    for idx,sym in enumerate(symbols,1):
        log.info(f"[{idx:02d}/{len(symbols)}] {sym}")
        r=analyze(sym)
        if r: results.append(r)
        time.sleep(0.08)

    if not results:
        tg_send(chat_id,"⚠️ Tidak ada data valid.")
        return None

    results.sort(key=lambda x:(x["confidence"],x["rr"]),reverse=True)
    return results[0]


# ═════════════════════════════════════════════
# MONITORING HARGA
# ═════════════════════════════════════════════
def monitor_trade(chat_id, signal):
    global timeout_flag
    sym=signal["symbol"]; is_buy=signal["decision"]=="BUY"
    entry=signal["entry"]; sl_p=signal["sl"]; tp_p=signal["tp"]
    etype=signal["etype"]

    # ── LIMIT ORDER ───────────────────────────
    if etype=="LIMIT":
        tg_send(chat_id,
            f"⏳ <b>Menunggu LIMIT — {sym}</b>\n"
            f"Entry   : <code>{entry:.6g}</code>\n"
            f"TP      : <code>{tp_p:.6g}</code>\n"
            f"SL      : <code>{sl_p:.6g}</code>\n"
            f"Timeout : {ENTRY_TIMEOUT//60} menit\n"
            f"/timeout untuk skip.")

        start=time.time(); entry_hit=False
        while time.time()-start<ENTRY_TIMEOUT:
            if timeout_flag:
                timeout_flag=False
                tg_send(chat_id,f"⏭ Timeout — limit {sym} dibatalkan.")
                return "no_entry"
            price=get_price(sym)
            if price is None: time.sleep(MONITOR_SLEEP); continue

            # TP sudah tercapai sebelum entry → no entry
            if is_buy and price>=tp_p:
                tg_send(chat_id,
                    f"❌ <b>No Entry — {sym}</b>\n"
                    f"TP <code>{tp_p:.6g}</code> tercapai sebelum entry limit.\n"
                    f"Harga: <code>{price:.6g}</code>")
                return "no_entry"
            if not is_buy and price<=tp_p:
                tg_send(chat_id,
                    f"❌ <b>No Entry — {sym}</b>\n"
                    f"TP <code>{tp_p:.6g}</code> tercapai sebelum entry limit.\n"
                    f"Harga: <code>{price:.6g}</code>")
                return "no_entry"

            # Entry tersentuh
            if is_buy and price<=entry: entry_hit=True; break
            if not is_buy and price>=entry: entry_hit=True; break
            time.sleep(MONITOR_SLEEP)

        if not entry_hit:
            tg_send(chat_id,
                f"⏰ <b>Limit Timeout — {sym}</b>\n"
                f"Entry <code>{entry:.6g}</code> tidak tercapai.")
            return "no_entry"

        tg_send(chat_id,
            f"✅ <b>Entry LIMIT terisi — {sym}</b>\n"
            f"Entry: <code>{entry:.6g}</code> | TP: <code>{tp_p:.6g}</code> | SL: <code>{sl_p:.6g}</code>")

    else:
        # ── MARKET ORDER ──────────────────────
        actual=get_price(sym) or entry
        tg_send(chat_id,
            f"⚡ <b>MARKET ENTRY — {sym}</b>\n"
            f"Keputusan : <b>{signal['decision']}</b>\n"
            f"Entry analisis : <code>{entry:.6g}</code>\n"
            f"Entry aktual   : <code>{actual:.6g}</code>\n"
            f"TP : <code>{tp_p:.6g}</code>\n"
            f"SL : <code>{sl_p:.6g}</code>  ← Invalidasi struktur\n"
            f"RR : 1:{signal['rr']}\n\n"
            f"📡 Monitoring tiap {MONITOR_SLEEP}s... /timeout untuk skip.")

    # ── MONITORING ──────────────────────────────
    last_log=time.time(); log_interval=60

    while True:
        if timeout_flag:
            timeout_flag=False
            price=get_price(sym) or 0
            tg_send(chat_id,
                f"⏭ <b>Timeout</b> — monitoring {sym} dihentikan.\n"
                f"Harga: <code>{price:.6g}</code>")
            return "timeout"

        price=get_price(sym)
        if price is None: time.sleep(MONITOR_SLEEP); continue

        if is_buy and price>=tp_p:
            tg_send(chat_id,
                f"🎯 <b>TAKE PROFIT — {sym}</b> 🎉\n"
                f"Harga: <code>{price:.6g}</code> ≥ TP: <code>{tp_p:.6g}</code>")
            return "tp"
        if not is_buy and price<=tp_p:
            tg_send(chat_id,
                f"🎯 <b>TAKE PROFIT — {sym}</b> 🎉\n"
                f"Harga: <code>{price:.6g}</code> ≤ TP: <code>{tp_p:.6g}</code>")
            return "tp"
        if is_buy and price<=sl_p:
            tg_send(chat_id,
                f"🛑 <b>STOP LOSS — {sym}</b>\n"
                f"Harga: <code>{price:.6g}</code> ≤ SL: <code>{sl_p:.6g}</code>\n"
                f"Struktur bullish invalidasi di level ini.")
            return "sl"
        if not is_buy and price>=sl_p:
            tg_send(chat_id,
                f"🛑 <b>STOP LOSS — {sym}</b>\n"
                f"Harga: <code>{price:.6g}</code> ≥ SL: <code>{sl_p:.6g}</code>\n"
                f"Struktur bearish invalidasi di level ini.")
            return "sl"

        if time.time()-last_log>=log_interval:
            risk=abs(entry-sl_p); reward_left=abs(tp_p-price)
            rr_left=round(reward_left/risk,2) if risk>0 else 0
            pct_to_tp=abs(tp_p-price)/abs(tp_p-entry)*100 if abs(tp_p-entry)>0 else 0
            tg_send(chat_id,
                f"📊 <b>Update {sym}</b>\n"
                f"Harga    : <code>{price:.6g}</code>\n"
                f"TP       : <code>{tp_p:.6g}</code> (sisa {pct_to_tp:.1f}%)\n"
                f"SL       : <code>{sl_p:.6g}</code>")
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
    if t==0: return "Belum ada simulasi selesai."
    wr=tp/(tp+sl)*100 if (tp+sl)>0 else 0
    return (
        f"📊 <b>Statistik Simulasi</b>\n\n"
        f"Total trade  : {t}\n"
        f"🎯 TP        : {tp} ({tp/t*100:.1f}%)\n"
        f"🛑 SL        : {sl} ({sl/t*100:.1f}%)\n"
        f"❌ No Entry  : {ne}\n"
        f"⏭ Timeout   : {to}\n"
        f"📈 Win Rate  : {wr:.1f}% (dari {tp+sl} trade aktual)\n\n"
        f"🚫 Koin diban: {len(banned_coins)}"
    )

def fmt_signal_msg(sig):
    em="🟢" if sig["decision"]=="BUY" else "🔴"
    bar="█"*(sig["confidence"]//10)+"░"*(10-sig["confidence"]//10)
    et=(f"⏳ LIMIT — tunggu <code>{sig['conf_lvl']}</code>"
        if sig["etype"]=="LIMIT" and sig.get("conf_lvl")
        else "⚡ MARKET — entry sekarang")
    risk=abs(sig["entry"]-sig["sl"])
    return (
        f"📡 <b>SINYAL TERBAIK</b>\n\n"
        f"{em} <b>{sig['symbol']}</b>\n"
        f"Keputusan  : <b>{sig['decision']}</b>\n"
        f"Confidence : <b>{sig['confidence']}%</b> {bar}\n"
        f"Harga kini : <code>{sig['price']:.6g}</code>\n"
        f"🎯 Entry   : <code>{sig['entry']:.6g}</code>\n"
        f"Tipe       : {et}\n"
        f"🛑 SL      : <code>{sig['sl']:.6g}</code> (invalidasi struktur)\n"
        f"✅ TP      : <code>{sig['tp']:.6g}</code>\n"
        f"⚖️ RR      : <b>1:{sig['rr']}</b> | Risk: {risk:.4g}\n"
        f"📝 Analisis: {sig['reason']}"
    )


# ═════════════════════════════════════════════
# SIMULATION LOOP
# ═════════════════════════════════════════════
def simulation_loop(chat_id):
    global auto_mode, timeout_flag
    tg_send(chat_id,
        "🤖 <b>Simulasi Trading dimulai!</b>\n"
        "Alur: Scan → Sinyal #1 → Simulasi → Statistik → Scan lagi\n"
        "Ketik /stop untuk berhenti.")

    while auto_mode:
        timeout_flag=False

        signal=run_scan_once(chat_id)
        if not auto_mode: break

        if signal is None:
            tg_send(chat_id,"⚠️ Tidak ada sinyal. Coba lagi 60 detik...")
            for _ in range(60):
                if not auto_mode: break
                time.sleep(1)
            continue

        tg_send(chat_id, fmt_signal_msg(signal))

        # Ban koin
        sym=signal["symbol"]
        with ban_lock:
            banned_coins.add(sym)

        # Simulasi
        result=monitor_trade(chat_id, signal)
        update_stats(result)

        emoji={"tp":"🎯","sl":"🛑","no_entry":"❌","timeout":"⏭"}.get(result,"❓")
        label={"tp":"TAKE PROFIT","sl":"STOP LOSS",
               "no_entry":"NO ENTRY","timeout":"TIMEOUT"}.get(result,result.upper())
        tg_send(chat_id, f"{emoji} <b>Hasil: {label}</b> — {sym}\n\n"+fmt_stats())

        if not auto_mode: break
        for _ in range(10):
            if not auto_mode: break
            time.sleep(1)

    tg_send(chat_id,"⏹ <b>Simulasi dihentikan.</b>\n\n"+fmt_stats())


# ═════════════════════════════════════════════
# PESAN STATIS
# ═════════════════════════════════════════════
GREETING=(
    "👋 <b>SMC Simulasi Trading Bot</b>\n\n"
    "Scan → Sinyal terbaik → Simulasi trade → Statistik\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "📌 <b>Perintah:</b>\n"
    "/start    — Menu ini\n"
    "/auto     — Mulai simulasi otomatis\n"
    "/stop     — Hentikan simulasi\n"
    "/timeout  — Skip monitoring, lanjut scan\n"
    "/stats    — Statistik TP/SL/Winrate\n"
    "/banned   — Daftar koin yang diban\n"
    "/resetban — Reset daftar ban\n"
    "/info     — Detail metode analisis\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "⚠️ <i>Simulasi saja — bukan saran finansial.</i>"
)
INFO_MSG=(
    "ℹ️ <b>Metode Analisis</b>\n\n"
    "<b>Entry:</b>\n"
    "1. Post-sweep entry (terkuat)\n"
    "2. OB + FVG konfluensi\n"
    "3. Order Block saja\n"
    "4. Fair Value Gap saja\n"
    "5. Market entry EMA konfirmasi\n\n"
    "<b>SL:</b> Di luar swing low/high struktural\n"
    "→ Jika harga ke sini, struktur sudah rusak\n"
    "→ Bukan SL arbitrary (ATR*1.5)\n\n"
    "<b>TP:</b> Swing high/low berikutnya atau equal highs/lows\n\n"
    "<b>Retracement vs Reversal:</b>\n"
    "Jika pullback dalam fib 38–62% & tidak ada BOS berlawanan\n"
    "→ dianggap retracement → poin entry tambahan\n\n"
    f"Min RR: 1:{MIN_RR} | TF: H1 (bias) + M15 (entry)"
)


# ═════════════════════════════════════════════
# BOT COMMAND LOOP
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
    log.info("Bot siap. Menunggu perintah...")

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
                    tg_send(chat_id,"⛔ Akses ditolak.")
                    continue
                active_chat_id=chat_id

                if text in ("/start","start"):
                    tg_send(chat_id,GREETING)
                elif text in ("/info","info"):
                    tg_send(chat_id,INFO_MSG)
                elif text in ("/stats","stats"):
                    tg_send(chat_id,fmt_stats())
                elif text in ("/banned","banned"):
                    with ban_lock:
                        b=sorted(banned_coins)
                    tg_send(chat_id,
                        f"🚫 <b>Banned ({len(b)}):</b>\n"+", ".join(b) if b
                        else "✅ Belum ada ban.")
                elif text in ("/resetban","resetban"):
                    with ban_lock:
                        n=len(banned_coins); banned_coins.clear()
                    tg_send(chat_id,f"✅ Ban direset ({n} koin dihapus).")
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
                        tg_send(chat_id,"⏹ Menghentikan simulasi...")
                    else:
                        tg_send(chat_id,"ℹ️ Simulasi tidak berjalan.")
                elif text in ("/timeout","timeout"):
                    if auto_mode:
                        timeout_flag=True
                        tg_send(chat_id,"⏭ Timeout — monitoring dilewati.")
                    else:
                        tg_send(chat_id,"ℹ️ Tidak ada monitoring aktif.")
                else:
                    tg_send(chat_id,"❓ Tidak dikenal. Ketik /start.")

            time.sleep(1)
        except Exception as e:
            log.error(f"[bot loop] {e}")
            time.sleep(5)


# ═════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════
if __name__=="__main__":
    threading.Thread(target=bot_loop, daemon=True).start()
    run_flask()
