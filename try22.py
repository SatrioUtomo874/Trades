#!/usr/bin/env python3
"""
SMC Signal Broadcasting Bot — Render.com Ready
Start command: python main.py
"""

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
ALLOWED_USER_ID = 8041197505

MAX_PRICE      = 80.0
TOP_N_COINS    = 50
TOP_SIGNALS    = 3
LOOP_INTERVAL  = 300  # detik
MIN_RR         = 2.0  # minimum RR untuk setup valid
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
FAPI           = "https://fapi.binance.com"

# ─── Flask (wajib untuk Render) ───────────────
app = Flask(__name__)

@app.route("/")
def index():
    return f"SMC Bot OK | Auto: {auto_mode}", 200

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
            params={"timeout": 10, "offset": offset}, timeout=15)
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
            r = requests.get(f"{FAPI}{path}", params=params, timeout=10, verify=False)
            d = r.json()
            if isinstance(d, dict) and "code" in d:
                raise ValueError(f"Binance {d['code']}: {d.get('msg')}")
            return d
        except Exception as e:
            log.warning(f"[fapi] {i+1}/3: {e}")
            time.sleep(2)
    raise ConnectionError(f"fapi gagal: {path}")

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
    usdt = [t for t in tickers
            if t["symbol"].endswith("USDT")
            and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
            and float(t["quoteVolume"]) > 100_000]
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

def atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def indicators(df):
    if len(df) < 50:
        return None
    df = df.copy()
    df["ema9"]   = ema(df["close"], 9)
    df["ema21"]  = ema(df["close"], 21)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200) if len(df)>=200 else ema(df["close"],50)
    df["rsi"]    = rsi(df["close"])
    df["macd_line"], df["macd_sig"], df["macd_hist"] = macd(df["close"])
    df["atr"]    = atr(df)
    df["vol_sma"]= df["volume"].rolling(20).mean()
    bb_mid       = df["close"].rolling(20).mean()
    bb_std       = df["close"].rolling(20).std()
    df["bb_up"]  = bb_mid + 2*bb_std
    df["bb_lo"]  = bb_mid - 2*bb_std
    df["bb_mid"] = bb_mid
    return df.dropna()


# ═════════════════════════════════════════════
# SMC HELPERS
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
        body = abs(nx["close"]-nx["open"])
        if body < avg*1.2: continue
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
# SKOR ANALISIS — tidak ada batas minimum
# Setiap koin selalu punya skor, kita ambil terbaik
# ═════════════════════════════════════════════
def score_coin(df_h1, df_m15):
    """
    Kembalikan dict lengkap dengan skor dan arah.
    Tidak ada filter ketat — semua koin dinilai, terbaik yang menang.
    """
    # ── Indikator ──────────────────────────────
    h1  = indicators(df_h1)
    m15 = indicators(df_m15)
    if h1 is None or m15 is None:
        return None

    L1  = h1.iloc[-1];  P1  = h1.iloc[-2]
    L15 = m15.iloc[-1]; P15 = m15.iloc[-2]

    # ── Poin positif untuk bull / bear ─────────
    # Tiap kriteria diberi bobot, dikumpul jadi skor 0–100
    bull_pts = 0
    bear_pts = 0

    # 1. EMA trend H1
    if L1["ema9"] > L1["ema21"] > L1["ema50"]:   bull_pts += 15
    elif L1["ema9"] > L1["ema21"]:                bull_pts += 8
    if L1["ema9"] < L1["ema21"] < L1["ema50"]:   bear_pts += 15
    elif L1["ema9"] < L1["ema21"]:                bear_pts += 8

    # 2. Harga vs EMA200 H1
    if L1["close"] > L1["ema200"]:  bull_pts += 8
    else:                            bear_pts += 8

    # 3. RSI M15
    rsi_v = L15["rsi"]
    if rsi_v < 40:   bull_pts += 10   # oversold → potensi rebound
    elif rsi_v < 50: bull_pts += 5
    if rsi_v > 60:   bear_pts += 10   # overbought → potensi turun
    elif rsi_v > 50: bear_pts += 5

    # 4. MACD M15
    if L15["macd_hist"] > 0 and P15["macd_hist"] <= 0: bull_pts += 12  # crossover
    elif L15["macd_hist"] > 0:                          bull_pts += 6
    if L15["macd_hist"] < 0 and P15["macd_hist"] >= 0: bear_pts += 12
    elif L15["macd_hist"] < 0:                          bear_pts += 6

    # 5. EMA alignment M15
    if L15["ema9"] > L15["ema21"] > L15["ema50"]: bull_pts += 10
    elif L15["ema9"] > L15["ema21"]:               bull_pts += 5
    if L15["ema9"] < L15["ema21"] < L15["ema50"]: bear_pts += 10
    elif L15["ema9"] < L15["ema21"]:               bear_pts += 5

    # 6. Bollinger posisi M15
    if L15["close"] < L15["bb_lo"]:  bull_pts += 8  # di bawah lower → potential reversal
    elif L15["close"] < L15["bb_mid"]: bull_pts += 3
    if L15["close"] > L15["bb_up"]:  bear_pts += 8
    elif L15["close"] > L15["bb_mid"]: bear_pts += 3

    # 7. Volume konfirmasi M15
    if L15["volume"] > L15["vol_sma"] * 1.3: 
        # volume tinggi searah candle
        if L15["close"] > L15["open"]: bull_pts += 7
        else:                           bear_pts += 7
    elif L15["volume"] > L15["vol_sma"]:
        if L15["close"] > L15["open"]: bull_pts += 3
        else:                           bear_pts += 3

    # 8. Market structure H1
    sh1, sl1   = swing_pts(h1, 5)
    struct_h1  = mkt_struct(h1, sh1, sl1)
    if struct_h1 == "bullish": bull_pts += 10
    if struct_h1 == "bearish": bear_pts += 10

    # 9. SMC M15
    sh15, sl15 = swing_pts(m15, 5)
    bos        = choch_bos(m15, sh15, sl15)
    sw_bull, sw_bear = liq_sweep(m15, sh15, sl15)
    obs_bull   = find_ob(m15, "bull")
    obs_bear   = find_ob(m15, "bear")
    fvg_bull   = find_fvg(m15, "bull")
    fvg_bear   = find_fvg(m15, "bear")

    if bos["bos_bull"] or bos["choch_bull"]: bull_pts += 10
    if bos["bos_bear"] or bos["choch_bear"]: bear_pts += 10
    if sw_bull:   bull_pts += 8
    if sw_bear:   bear_pts += 8
    if obs_bull:  bull_pts += 7
    if obs_bear:  bear_pts += 7
    if fvg_bull:  bull_pts += 5
    if fvg_bear:  bear_pts += 5

    # ── Tentukan arah dominan ──────────────────
    total = bull_pts + bear_pts
    if total == 0:
        return None

    if bull_pts >= bear_pts:
        direction = "bull"
        raw_conf  = bull_pts
        obs       = obs_bull
        fvgs      = fvg_bull
    else:
        direction = "bear"
        raw_conf  = bear_pts
        obs       = obs_bear
        fvgs      = fvg_bear

    # Normalisasi skor ke 0–100 (max poin teoritis ~130)
    confidence = min(int(raw_conf / 130 * 100), 99)

    # ── Entry / SL / TP ───────────────────────
    price   = L15["close"]
    atr_val = L15["atr"] if L15["atr"] > 0 else price * 0.005

    entry, sl_p, tp_p, etype, conf_lvl = calc_setup(
        m15, direction, price, atr_val, obs, fvgs, sh15, sl15
    )
    if entry is None:
        return None

    risk   = abs(entry - sl_p)
    reward = abs(tp_p - entry)
    rr     = round(reward / risk, 2) if risk > 0 else 0

    # ── Narasi alasan ─────────────────────────
    why = []
    if struct_h1 != "ranging":       why.append(f"H1:{struct_h1.upper()}")
    if L1["close"] > L1["ema200"]:   why.append("Di atas EMA200")
    elif L1["close"] < L1["ema200"]: why.append("Di bawah EMA200")
    if bos["bos_bull"] or bos["bos_bear"]:          why.append("BOS✔")
    if bos["choch_bull"] or bos["choch_bear"]:      why.append("CHoCH✔")
    if sw_bull or sw_bear:                           why.append("LiqSweep✔")
    if obs:  why.append(f"OB:{obs[-1]['bot']:.4g}–{obs[-1]['top']:.4g}")
    if fvgs: why.append(f"FVG:{fvgs[-1]['bot']:.4g}–{fvgs[-1]['top']:.4g}")
    macd_cross = (direction=="bull" and L15["macd_hist"]>0 and P15["macd_hist"]<=0) or \
                 (direction=="bear" and L15["macd_hist"]<0 and P15["macd_hist"]>=0)
    if macd_cross: why.append("MACD Cross✔")
    why.append(f"RSI:{rsi_v:.0f}")

    return {
        "decision"   : "BUY" if direction=="bull" else "SELL",
        "confidence" : confidence,
        "price"      : price,
        "entry"      : round(entry, 8),
        "sl"         : round(sl_p, 8),
        "tp"         : round(tp_p, 8),
        "rr"         : rr,
        "etype"      : etype,
        "conf_lvl"   : conf_lvl,
        "reason"     : " | ".join(why),
        "rsi"        : round(rsi_v, 1),
        "bull_pts"   : bull_pts,
        "bear_pts"   : bear_pts,
    }


def calc_setup(df, direction, price, atr_val, obs, fvgs, sh, sl_pts):
    """Hitung entry/SL/TP. Fallback ke ATR jika tidak ada OB/FVG."""

    def try_zone(ztop, zbot):
        buf  = atr_val * 0.3
        in_z = zbot <= price <= ztop
        if direction == "bull":
            entry    = price if in_z else ztop
            etype    = "MARKET" if in_z else "STOP LIMIT"
            conf_lvl = None if in_z else round(ztop, 8)
            sl_p     = zbot - buf
            cands    = [df["high"].iloc[i] for i in sh if df["high"].iloc[i] > entry]
            tp_p     = min(cands) if cands else entry + abs(entry-sl_p)*MIN_RR
        else:
            entry    = price if in_z else zbot
            etype    = "MARKET" if in_z else "STOP LIMIT"
            conf_lvl = None if in_z else round(zbot, 8)
            sl_p     = ztop + buf
            cands    = [df["low"].iloc[i] for i in sl_pts if df["low"].iloc[i] < entry]
            tp_p     = max(cands) if cands else entry - abs(sl_p-entry)*MIN_RR
        risk = abs(entry-sl_p)
        if risk == 0: return None
        rr = abs(tp_p-entry)/risk
        if rr < MIN_RR: return None
        return entry, sl_p, tp_p, etype, conf_lvl

    # Prioritas 1: OB + FVG
    for ob in reversed(obs):
        for fvg in reversed(fvgs):
            ot = min(ob["top"], fvg["top"]); ob_ = max(ob["bot"], fvg["bot"])
            if ot > ob_:
                r = try_zone(ot, ob_)
                if r: return r

    # Prioritas 2: OB
    for ob in reversed(obs):
        r = try_zone(ob["top"], ob["bot"])
        if r: return r

    # Prioritas 3: FVG
    for fvg in reversed(fvgs):
        r = try_zone(fvg["top"], fvg["bot"])
        if r: return r

    # Fallback: ATR-based setup (selalu ada)
    if direction == "bull":
        entry  = price
        sl_p   = price - atr_val * 1.5
        tp_p   = price + atr_val * 1.5 * MIN_RR
        etype  = "MARKET"
    else:
        entry  = price
        sl_p   = price + atr_val * 1.5
        tp_p   = price - atr_val * 1.5 * MIN_RR
        etype  = "MARKET"

    return entry, sl_p, tp_p, etype, None


# ═════════════════════════════════════════════
# ANALISIS SATU KOIN
# ═════════════════════════════════════════════
def analyze(symbol):
    try:
        df_h1  = get_klines(symbol, "1h",  200)
        df_m15 = get_klines(symbol, "15m", 200)
        if df_h1.empty or df_m15.empty:
            return None
        result = score_coin(df_h1, df_m15)
        if result:
            result["symbol"] = symbol
        return result
    except Exception as e:
        log.debug(f"[analyze] {symbol}: {e}")
        return None


# ═════════════════════════════════════════════
# FORMAT PESAN
# ═════════════════════════════════════════════
GREETING = (
    "👋 <b>SMC Signal Bot — Aktif!</b>\n\n"
    "Menscan <b>50 koin USDT Futures</b> volume tertinggi (harga &lt; $80)\n"
    "Analisis: <b>SMC + Price Action + Multi-TF (H1 + M15)</b>\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "📌 <b>Perintah:</b>\n"
    "/start  — Tampilkan pesan ini\n"
    "/scan   — Scan manual sekarang\n"
    "/auto   — Scan otomatis tiap 5 menit\n"
    "/stop   — Hentikan scan otomatis\n"
    "/status — Status bot\n"
    "/info   — Detail metode analisis\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "⚠️ <i>Sinyal bersifat edukatif. Bukan saran finansial.</i>"
)

INFO_MSG = (
    "ℹ️ <b>Metode Analisis</b>\n\n"
    "Setiap koin dinilai dengan sistem poin:\n\n"
    "• EMA 9/21/50/200 alignment (H1+M15)\n"
    "• RSI 14 — oversold/overbought\n"
    "• MACD crossover momentum\n"
    "• Bollinger Bands posisi\n"
    "• Volume vs rata-rata\n"
    "• Market Structure H1 (HH/HL vs LH/LL)\n"
    "• BOS + CHoCH (Break/Change of Structure)\n"
    "• Order Block detection\n"
    "• Fair Value Gap\n"
    "• Liquidity Sweep\n\n"
    "Semua poin dijumlah → 3 koin skor tertinggi\n"
    "dikirim setiap putaran.\n\n"
    f"⚖️ Min RR: 1:{MIN_RR} | Fallback: ATR-based setup"
)


def fmt_signals(results, scan_time, total):
    lines = [
        "📡 <b>SMC SIGNAL BROADCAST</b>",
        f"🕐 {scan_time}  |  🔍 {total} koin discan\n",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for i, r in enumerate(results, 1):
        em    = "🟢" if r["decision"]=="BUY" else "🔴"
        bar   = "█" * (r["confidence"]//10) + "░" * (10 - r["confidence"]//10)
        etype = (
            f"⏳ <b>Tipe:</b> STOP LIMIT\n   ↳ Konfirmasi di <code>{r['conf_lvl']}</code>"
            if r["etype"]=="STOP LIMIT" and r["conf_lvl"]
            else "⚡ <b>Tipe:</b> MARKET"
        )
        lines.append(
            f"\n{em} <b>#{i} {r['symbol']}</b>\n"
            f"💰 Harga      : <code>{r['price']:.6g}</code>\n"
            f"📊 Keputusan  : <b>{r['decision']}</b>\n"
            f"📈 Confidence : <b>{r['confidence']}%</b> {bar}\n"
            f"🎯 Entry      : <code>{r['entry']:.6g}</code>\n"
            f"{etype}\n"
            f"🛑 Stop Loss  : <code>{r['sl']:.6g}</code>\n"
            f"✅ Take Profit: <code>{r['tp']:.6g}</code>\n"
            f"⚖️ RR         : <b>1:{r['rr']}</b>\n"
            f"📝 Analisis   : {r['reason']}"
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("\n⚠️ <i>Edukatif — bukan saran finansial.</i>")
    return "\n".join(lines)


# ═════════════════════════════════════════════
# SCAN RUNNER
# ═════════════════════════════════════════════
def run_scan(chat_id, silent=False):
    scan_time = datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC")
    if not silent:
        tg_send(chat_id, f"🔄 Scan {TOP_N_COINS} koin dimulai...")

    try:
        symbols = get_top_coins()
    except Exception as e:
        tg_send(chat_id, f"⚠️ <b>Error Binance:</b> <code>{str(e)[:200]}</code>")
        return

    results = []
    for idx, sym in enumerate(symbols, 1):
        log.info(f"[{idx:02d}/{len(symbols)}] {sym}")
        r = analyze(sym)
        if r:
            results.append(r)
        time.sleep(0.08)

    if not results:
        tg_send(chat_id, f"⚠️ Tidak ada data valid dari {len(symbols)} koin.")
        return

    # Ranking: confidence DESC → rr DESC
    results.sort(key=lambda x: (x["confidence"], x["rr"]), reverse=True)
    top = results[:TOP_SIGNALS]

    log.info(f"Scan selesai — {len(results)} valid, top {len(top)} dikirim.")
    tg_send(chat_id, fmt_signals(top, scan_time, len(symbols)))


# ═════════════════════════════════════════════
# AUTO LOOP
# ═════════════════════════════════════════════
def auto_loop(chat_id):
    global auto_mode
    while auto_mode:
        run_scan(chat_id, silent=True)
        for _ in range(LOOP_INTERVAL):
            if not auto_mode: break
            time.sleep(1)


# ═════════════════════════════════════════════
# BOT LOOP
# ═════════════════════════════════════════════
def bot_loop():
    global auto_mode, auto_thread, active_chat_id

    log.info("Test koneksi Binance Futures...")
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
    log.info("Menunggu perintah Telegram...")

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

                elif text in ("/scan","scan"):
                    threading.Thread(target=run_scan, args=(chat_id,), daemon=True).start()

                elif text in ("/auto","auto"):
                    if auto_mode:
                        tg_send(chat_id, "⚙️ Auto scan sudah aktif.")
                    else:
                        auto_mode   = True
                        auto_thread = threading.Thread(
                            target=auto_loop, args=(chat_id,), daemon=True)
                        auto_thread.start()
                        tg_send(chat_id,
                            f"✅ Auto scan aktif — tiap {LOOP_INTERVAL//60} menit.\n"
                            "Scan pertama dimulai sekarang...")

                elif text in ("/stop","stop"):
                    if auto_mode:
                        auto_mode = False
                        tg_send(chat_id, "⏹ Auto scan dihentikan.")
                    else:
                        tg_send(chat_id, "ℹ️ Auto scan tidak aktif.")

                elif text in ("/status","status"):
                    tg_send(chat_id, (
                        f"📶 <b>Status Bot</b>\n\n"
                        f"Mode    : {'🟢 AUTO' if auto_mode else '⚪ Manual'}\n"
                        f"Endpoint: Binance Futures\n"
                        f"TF      : H1 + M15\n"
                        f"Interval: {LOOP_INTERVAL//60} menit"
                    ))

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
    run_flask()  # main thread — wajib untuk Render
