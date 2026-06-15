#!/usr/bin/env python3
"""
SMC Signal Broadcasting Bot
Deploy: Render.com Web Service
Start command: python main.py
"""

# ─────────────────────────────────────────────
# KONFIGURASI — edit bagian ini saja
# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
ALLOWED_USER_ID = 8041197505

MAX_PRICE       = 80.0
TOP_N_COINS     = 50
MIN_CONFIDENCE  = 60
MIN_RR          = 3.0
LOOP_INTERVAL   = 300   # detik (5 menit)
TOP_SIGNALS     = 3
# ─────────────────────────────────────────────

import os
import time
import logging
import threading
from datetime import datetime

import requests
import pandas as pd
import numpy as np
import urllib3
from flask import Flask

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

auto_mode      = False
auto_thread    = None
active_chat_id = None

FAPI = "https://fapi.binance.com"

# ─────────────────────────────────────────────
# FLASK — wajib untuk Render agar tidak error
# "no port detected"
# ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    mode = "AUTO" if auto_mode else "MANUAL"
    return f"SMC Signal Bot — OK | Mode: {mode}", 200

@app.route("/health")
def health():
    return "OK", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    log.info(f"Flask berjalan di port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═════════════════════════════════════════════
# TELEGRAM
# ═════════════════════════════════════════════
def tg_send(chat_id: int, text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.error(f"[TG] {e}")


def tg_error(chat_id: int, ctx: str, err: Exception):
    log.error(f"[{ctx}] {err}")
    tg_send(chat_id, f"⚠️ <b>Error — {ctx}</b>\n<code>{str(err)[:300]}</code>")


def tg_updates(offset=None) -> list:
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"timeout": 10, "offset": offset},
            timeout=15,
        )
        data = r.json()
        return data.get("result", []) if data.get("ok") else []
    except Exception as e:
        log.warning(f"[TG] getUpdates: {e}")
        return []


def authorized(uid: int) -> bool:
    return uid == ALLOWED_USER_ID


# ═════════════════════════════════════════════
# BINANCE FUTURES DATA
# ═════════════════════════════════════════════
def fapi_get(path: str, params: dict = None):
    for attempt in range(3):
        try:
            r = requests.get(
                f"{FAPI}{path}", params=params, timeout=10, verify=False
            )
            data = r.json()
            if isinstance(data, dict) and "code" in data:
                raise ValueError(f"Binance error {data['code']}: {data.get('msg')}")
            return data
        except Exception as e:
            log.warning(f"[fapi] attempt {attempt+1}: {e}")
            time.sleep(2)
    raise ConnectionError(f"fapi gagal setelah 3x: {path}")


def get_klines(symbol: str, interval: str, limit: int = 150) -> pd.DataFrame:
    raw = fapi_get(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )
    if not isinstance(raw, list) or len(raw) < 30:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "cts", "qvol", "trades", "tbv", "tbq", "ign",
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open", "high", "low", "close", "volume"]].dropna()


def get_top_coins() -> list:
    tickers = fapi_get("/fapi/v1/ticker/24hr")
    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t["quoteVolume"]) > 500_000
    ]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    symbols = [t["symbol"] for t in usdt[:TOP_N_COINS]]
    log.info(f"[Binance] {len(symbols)} koin dipilih (harga < ${MAX_PRICE})")
    return symbols


# ═════════════════════════════════════════════
# INDIKATOR
# ═════════════════════════════════════════════
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd(s: pd.Series):
    line = ema(s, 12) - ema(s, 26)
    sig  = ema(line, 9)
    return line, sig, line - sig

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_indicators(df: pd.DataFrame):
    if len(df) < 60:
        return None
    df = df.copy()
    df["ema20"]    = ema(df["close"], 20)
    df["ema50"]    = ema(df["close"], 50)
    df["ema200"]   = ema(df["close"], 200) if len(df) >= 200 else ema(df["close"], 50)
    df["rsi"]      = rsi(df["close"])
    df["macd_line"], df["macd_sig"], df["macd_hist"] = macd(df["close"])
    df["atr"]      = atr(df)
    df["vol_sma"]  = df["volume"].rolling(20).mean()
    bb_mid         = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    return df.dropna()


# ═════════════════════════════════════════════
# SMC
# ═════════════════════════════════════════════
def swing_points(df: pd.DataFrame, lb: int = 5):
    sh, sl = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i-lb:i+lb+1].max():
            sh.append(i)
        if df["low"].iloc[i] == df["low"].iloc[i-lb:i+lb+1].min():
            sl.append(i)
    return sh, sl


def market_structure(df: pd.DataFrame, sh: list, sl: list) -> str:
    if len(sh) < 2 or len(sl) < 2:
        return "ranging"
    hh = df["high"].iloc[sh[-1]] > df["high"].iloc[sh[-2]]
    hl = df["low"].iloc[sl[-1]]  > df["low"].iloc[sl[-2]]
    lh = df["high"].iloc[sh[-1]] < df["high"].iloc[sh[-2]]
    ll = df["low"].iloc[sl[-1]]  < df["low"].iloc[sl[-2]]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"


def detect_bos_choch(df: pd.DataFrame, sh: list, sl: list) -> dict:
    res   = dict(bos_bull=False, bos_bear=False, choch_bull=False, choch_bear=False)
    price = df["close"].iloc[-1]
    if len(sh) >= 2:
        ph = df["high"].iloc[sh[-2]]
        lh = df["high"].iloc[sh[-1]]
        if price > ph:
            res["bos_bull" if lh > ph else "choch_bull"] = True
    if len(sl) >= 2:
        pl = df["low"].iloc[sl[-2]]
        ll = df["low"].iloc[sl[-1]]
        if price < pl:
            res["bos_bear" if ll < pl else "choch_bear"] = True
    return res


def order_blocks(df: pd.DataFrame, direction: str, lb: int = 30) -> list:
    sub = df.iloc[-lb:]
    avg = (sub["close"] - sub["open"]).abs().mean()
    obs = []
    for i in range(1, len(sub) - 1):
        c, nx = sub.iloc[i], sub.iloc[i+1]
        if abs(nx["close"] - nx["open"]) < avg * 1.3:
            continue
        if direction == "bull" and c["close"] < c["open"] and nx["close"] > nx["open"]:
            obs.append({"top": c["high"], "bot": c["low"]})
        if direction == "bear" and c["close"] > c["open"] and nx["close"] < nx["open"]:
            obs.append({"top": c["high"], "bot": c["low"]})
    return obs[-3:]


def fair_value_gaps(df: pd.DataFrame, direction: str, lb: int = 30) -> list:
    sub  = df.iloc[-lb:]
    fvgs = []
    for i in range(len(sub) - 2):
        c0, c2 = sub.iloc[i], sub.iloc[i+2]
        if direction == "bull" and c2["low"] > c0["high"]:
            fvgs.append({"top": c2["low"], "bot": c0["high"]})
        if direction == "bear" and c2["high"] < c0["low"]:
            fvgs.append({"top": c0["low"], "bot": c2["high"]})
    return fvgs[-3:]


def liquidity_sweep(df: pd.DataFrame, sh: list, sl: list):
    bull = bear = False
    cur, prv = df.iloc[-1], df.iloc[-2]
    if sl:
        ll = df["low"].iloc[sl[-1]]
        if prv["low"] < ll < cur["close"]:
            bull = True
    if sh:
        lh = df["high"].iloc[sh[-1]]
        if prv["high"] > lh > cur["close"]:
            bear = True
    return bull, bear


def premium_discount(df: pd.DataFrame) -> str:
    mid = (df["high"].max() + df["low"].min()) / 2
    return "premium" if df["close"].iloc[-1] > mid else "discount"


# ═════════════════════════════════════════════
# ENTRY / SL / TP
# ═════════════════════════════════════════════
def build_setup(df: pd.DataFrame, direction: str, obs: list, fvgs: list,
                sh: list, sl_pts: list):
    price   = df["close"].iloc[-1]
    atr_val = df["atr"].iloc[-1]

    def try_zone(ztop, zbot):
        buf  = atr_val * 0.5
        in_z = zbot <= price <= ztop
        if direction == "bull":
            entry    = price if in_z else ztop
            etype    = "MARKET" if in_z else "STOP LIMIT"
            conf_lvl = None if in_z else ztop
            sl_p     = zbot - buf
            above    = [df["high"].iloc[i] for i in sh if df["high"].iloc[i] > entry]
            tp_p     = min(above) if above else entry + abs(entry - sl_p) * MIN_RR
        else:
            entry    = price if in_z else zbot
            etype    = "MARKET" if in_z else "STOP LIMIT"
            conf_lvl = None if in_z else zbot
            sl_p     = ztop + buf
            below    = [df["low"].iloc[i] for i in sl_pts if df["low"].iloc[i] < entry]
            tp_p     = max(below) if below else entry - abs(sl_p - entry) * MIN_RR

        risk = abs(entry - sl_p)
        if risk == 0:
            return None
        rr = abs(tp_p - entry) / risk
        if rr < MIN_RR:
            return None
        return {
            "entry": round(entry, 8),
            "sl"   : round(sl_p, 8),
            "tp"   : round(tp_p, 8),
            "rr"   : round(rr, 2),
            "etype": etype,
            "conf" : round(conf_lvl, 8) if conf_lvl else None,
        }

    # Prioritas 1: OB + FVG konfluensi
    for ob in reversed(obs):
        for fvg in reversed(fvgs):
            ot  = min(ob["top"], fvg["top"])
            ob_ = max(ob["bot"], fvg["bot"])
            if ot > ob_:
                s = try_zone(ot, ob_)
                if s:
                    return s, "OB + FVG"
    # Prioritas 2: OB saja
    for ob in reversed(obs):
        s = try_zone(ob["top"], ob["bot"])
        if s:
            return s, "Order Block"
    # Prioritas 3: FVG saja
    for fvg in reversed(fvgs):
        s = try_zone(fvg["top"], fvg["bot"])
        if s:
            return s, "Fair Value Gap"

    return None, None


# ═════════════════════════════════════════════
# CONFIDENCE SCORE
# ═════════════════════════════════════════════
def confidence(df: pd.DataFrame, direction: str, bos: dict, struct_h1: str,
               obs: list, fvgs: list, sw_bull: bool, sw_bear: bool, zone: str) -> int:
    sc   = 0
    L, P = df.iloc[-1], df.iloc[-2]

    if direction == "bull":
        if L["ema20"] > L["ema50"] > L["ema200"]: sc += 18
        elif L["ema20"] > L["ema50"]:             sc += 9
        if 40 <= L["rsi"] <= 65:   sc += 12
        elif 30 <= L["rsi"] < 40:  sc += 6
        if L["macd_hist"] > 0 and P["macd_hist"] <= 0: sc += 12
        elif L["macd_hist"] > 0:                        sc += 6
        if L["volume"] > L["vol_sma"] * 1.2: sc += 8
        if bos["bos_bull"]:    sc += 12
        if bos["choch_bull"]:  sc += 10
        if obs:                sc += 12
        if fvgs:               sc += 8
        if sw_bull:            sc += 6
        if struct_h1 == "bullish": sc += 8
        if zone == "discount": sc += 4
    else:
        if L["ema20"] < L["ema50"] < L["ema200"]: sc += 18
        elif L["ema20"] < L["ema50"]:             sc += 9
        if 35 <= L["rsi"] <= 60:   sc += 12
        elif 60 < L["rsi"] <= 70:  sc += 6
        if L["macd_hist"] < 0 and P["macd_hist"] >= 0: sc += 12
        elif L["macd_hist"] < 0:                        sc += 6
        if L["volume"] > L["vol_sma"] * 1.2: sc += 8
        if bos["bos_bear"]:    sc += 12
        if bos["choch_bear"]:  sc += 10
        if obs:                sc += 12
        if fvgs:               sc += 8
        if sw_bear:            sc += 6
        if struct_h1 == "bearish": sc += 8
        if zone == "premium":  sc += 4

    return min(sc, 100)


# ═════════════════════════════════════════════
# ANALISIS SATU KOIN (multi-timeframe H1+M15)
# ═════════════════════════════════════════════
def analyze(symbol: str):
    try:
        df_h1  = get_klines(symbol, "1h",  150)
        df_m15 = get_klines(symbol, "15m", 150)

        if df_h1.empty or df_m15.empty:
            return None

        df_h1  = add_indicators(df_h1)
        df_m15 = add_indicators(df_m15)

        if df_h1 is None or df_m15 is None:
            return None

        # Filter volatilitas ekstrem
        atr_now  = df_m15["atr"].iloc[-1]
        atr_mean = df_m15["atr"].rolling(50).mean().iloc[-1]
        if pd.notna(atr_mean) and atr_now > 2.5 * atr_mean:
            return None

        # Struktur H1 sebagai bias utama
        sh_h1, sl_h1 = swing_points(df_h1, lb=5)
        struct_h1    = market_structure(df_h1, sh_h1, sl_h1)
        if struct_h1 == "ranging":
            return None

        direction = "bull" if struct_h1 == "bullish" else "bear"

        # Konfirmasi EMA M15 searah bias
        L15 = df_m15.iloc[-1]
        if direction == "bull" and L15["ema20"] <= L15["ema50"]:
            return None
        if direction == "bear" and L15["ema20"] >= L15["ema50"]:
            return None

        # SMC M15
        sh_m15, sl_m15   = swing_points(df_m15, lb=5)
        bos              = detect_bos_choch(df_m15, sh_m15, sl_m15)
        sw_bull, sw_bear = liquidity_sweep(df_m15, sh_m15, sl_m15)
        obs_list         = order_blocks(df_m15, direction)
        fvg_list         = fair_value_gaps(df_m15, direction)
        zone             = premium_discount(df_m15)

        # Liquidity sweep wajib ada
        if direction == "bull" and not sw_bull:
            return None
        if direction == "bear" and not sw_bear:
            return None

        conf = confidence(
            df_m15, direction, bos, struct_h1,
            obs_list, fvg_list, sw_bull, sw_bear, zone,
        )
        if conf < MIN_CONFIDENCE:
            return None

        setup, setup_type = build_setup(
            df_m15, direction, obs_list, fvg_list, sh_m15, sl_m15
        )
        if setup is None:
            return None

        # Narasi alasan
        why = []
        if struct_h1 != "ranging":                      why.append(f"H1 {struct_h1.upper()}")
        if bos["bos_bull"] or bos["bos_bear"]:          why.append("BOS ✔")
        if bos["choch_bull"] or bos["choch_bear"]:      why.append("CHoCH ✔")
        if obs_list: why.append(f"OB {obs_list[-1]['bot']:.5g}–{obs_list[-1]['top']:.5g}")
        if fvg_list: why.append(f"FVG {fvg_list[-1]['bot']:.5g}–{fvg_list[-1]['top']:.5g}")
        if sw_bull or sw_bear:                           why.append("Liq.Sweep ✔")
        if zone in ("discount", "premium"):              why.append(f"Zone:{zone}")

        return {
            "symbol"    : symbol,
            "price"     : L15["close"],
            "decision"  : "BUY" if direction == "bull" else "SELL",
            "confidence": conf,
            "entry"     : setup["entry"],
            "sl"        : setup["sl"],
            "tp"        : setup["tp"],
            "rr"        : setup["rr"],
            "etype"     : setup["etype"],
            "conf_lvl"  : setup["conf"],
            "setup_type": setup_type,
            "reason"    : " | ".join(why),
            "rsi"       : round(L15["rsi"], 1),
        }

    except Exception as e:
        log.debug(f"[analyze] {symbol}: {e}")
        return None


# ═════════════════════════════════════════════
# FORMAT PESAN TELEGRAM
# ═════════════════════════════════════════════
GREETING = (
    "👋 <b>SMC Signal Bot — Aktif!</b>\n\n"
    "Menscan <b>50 koin USDT Futures</b> volume tertinggi (harga &lt; $80)\n"
    "Analisis: <b>SMC + Price Action + Multi-Timeframe (H1 + M15)</b>\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "📌 <b>Perintah:</b>\n"
    "/start  — Tampilkan pesan ini\n"
    "/scan   — Scan manual sekarang\n"
    "/auto   — Scan otomatis tiap 5 menit\n"
    "/stop   — Hentikan scan otomatis\n"
    "/status — Status &amp; mode bot\n"
    "/info   — Detail konfigurasi\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "⚠️ <i>Sinyal bersifat edukatif. Bukan saran finansial.</i>"
)

INFO_MSG = (
    "ℹ️ <b>Konfigurasi Bot</b>\n\n"
    f"📊 Timeframe     : H1 (bias) + M15 (entry)\n"
    f"🔢 Jumlah koin   : {TOP_N_COINS}\n"
    f"💵 Filter harga  : &lt; ${MAX_PRICE}\n"
    f"🎯 Min confidence: {MIN_CONFIDENCE}%\n"
    f"⚖️ Min RR         : 1:{int(MIN_RR)}\n"
    f"🏆 Top sinyal    : {TOP_SIGNALS}\n\n"
    "🧠 <b>Metode:</b>\n"
    "• Market Structure H1 (bias utama)\n"
    "• EMA 20/50/200 alignment\n"
    "• RSI 14 + MACD momentum\n"
    "• BOS + CHoCH (M15)\n"
    "• Order Block + FVG\n"
    "• Liquidity Sweep (wajib ada)\n"
    "• Premium/Discount Zone\n"
    "• ATR filter volatilitas ekstrem\n"
    "• Volume confirmation"
)


def fmt_signals(results: list, scan_time: str, total: int) -> str:
    lines = [
        "📡 <b>SMC SIGNAL BROADCAST</b>",
        f"🕐 {scan_time}  |  🔍 {total} koin discan\n",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for i, r in enumerate(results, 1):
        em    = "🟢" if r["decision"] == "BUY" else "🔴"
        etype = (
            f"⏳ <b>Tipe:</b> STOP LIMIT — tunggu <code>{r['conf_lvl']}</code>"
            if r["etype"] == "STOP LIMIT" and r["conf_lvl"]
            else "⚡ <b>Tipe:</b> MARKET / LIMIT"
        )
        lines.append(
            f"\n{em} <b>#{i} {r['symbol']}</b> <i>({r['setup_type']})</i>\n"
            f"💰 Harga     : <code>{r['price']:.6g}</code>\n"
            f"📊 Sinyal    : <b>{r['decision']}</b> | Conf: <b>{r['confidence']}%</b> | RSI: {r['rsi']}\n"
            f"🎯 Entry     : <code>{r['entry']:.6g}</code>\n"
            f"{etype}\n"
            f"🛑 Stop Loss : <code>{r['sl']:.6g}</code>\n"
            f"✅ Take Profit: <code>{r['tp']:.6g}</code>\n"
            f"⚖️ RR        : <b>1:{r['rr']}</b>\n"
            f"📝 Alasan    : {r['reason']}"
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("\n⚠️ <i>Edukatif saja — bukan saran finansial.</i>")
    return "\n".join(lines)


def fmt_no_signal(scan_time: str, total: int) -> str:
    return (
        f"📡 <b>SMC SIGNAL BROADCAST</b>\n"
        f"🕐 {scan_time}  |  🔍 {total} koin\n\n"
        f"⏸ Tidak ada sinyal memenuhi kriteria:\n"
        f"Conf ≥ {MIN_CONFIDENCE}%  |  RR ≥ 1:{int(MIN_RR)}\n\n"
        f"Scan ulang dalam {LOOP_INTERVAL // 60} menit."
    )


# ═════════════════════════════════════════════
# SCAN RUNNER
# ═════════════════════════════════════════════
def run_scan(chat_id: int, silent: bool = False):
    scan_time = datetime.now().strftime("%d/%m/%Y %H:%M UTC")
    if not silent:
        tg_send(chat_id, f"🔄 Scan {TOP_N_COINS} koin dimulai... mohon tunggu.")

    try:
        symbols = get_top_coins()
    except Exception as e:
        tg_error(chat_id, "Ambil data Binance", e)
        return

    results = []
    for idx, sym in enumerate(symbols, 1):
        log.info(f"[{idx:02d}/{len(symbols)}] {sym}")
        r = analyze(sym)
        if r:
            results.append(r)
        time.sleep(0.05)

    results.sort(key=lambda x: (x["confidence"], x["rr"]), reverse=True)
    top = results[:TOP_SIGNALS]

    msg = fmt_signals(top, scan_time, len(symbols)) if top else fmt_no_signal(scan_time, len(symbols))
    tg_send(chat_id, msg)
    log.info(f"Scan selesai — {len(results)} sinyal, {len(top)} dikirim.")


# ═════════════════════════════════════════════
# AUTO LOOP
# ═════════════════════════════════════════════
def auto_loop(chat_id: int):
    global auto_mode
    while auto_mode:
        run_scan(chat_id, silent=True)
        for _ in range(LOOP_INTERVAL):
            if not auto_mode:
                break
            time.sleep(1)
    log.info("Auto loop berhenti.")


# ═════════════════════════════════════════════
# BOT LOOP (berjalan di thread terpisah)
# ═════════════════════════════════════════════
def bot_loop():
    global auto_mode, auto_thread, active_chat_id

    log.info("Test koneksi Binance Futures...")
    for attempt in range(10):
        try:
            fapi_get("/fapi/v1/ping")
            log.info("Binance Futures OK!")
            break
        except Exception as e:
            log.warning(f"Binance belum bisa dijangkau (attempt {attempt+1}/10): {e}")
            time.sleep(10)
    else:
        log.critical("Binance tidak bisa dijangkau setelah 10x percobaan.")
        return

    log.info("Menunggu perintah Telegram...")
    offset = None

    while True:
        try:
            updates = tg_updates(offset)
            for upd in updates:
                offset  = upd["update_id"] + 1
                msg     = upd.get("message", {})
                uid     = msg.get("from", {}).get("id")
                chat_id = msg.get("chat", {}).get("id")
                text    = msg.get("text", "").strip().lower()

                if not uid or not chat_id or not text:
                    continue

                if not authorized(uid):
                    tg_send(chat_id, "⛔ Akses ditolak.")
                    log.warning(f"Unauthorized: {uid}")
                    continue

                active_chat_id = chat_id

                if text in ("/start", "start"):
                    tg_send(chat_id, GREETING)

                elif text in ("/info", "info"):
                    tg_send(chat_id, INFO_MSG)

                elif text in ("/scan", "scan"):
                    threading.Thread(
                        target=run_scan, args=(chat_id,), daemon=True
                    ).start()

                elif text in ("/auto", "auto"):
                    if auto_mode:
                        tg_send(chat_id, "⚙️ Auto scan sudah aktif.")
                    else:
                        auto_mode   = True
                        auto_thread = threading.Thread(
                            target=auto_loop, args=(chat_id,), daemon=True
                        )
                        auto_thread.start()
                        tg_send(
                            chat_id,
                            f"✅ Auto scan aktif — tiap {LOOP_INTERVAL // 60} menit.",
                        )

                elif text in ("/stop", "stop"):
                    if auto_mode:
                        auto_mode = False
                        tg_send(chat_id, "⏹ Auto scan dihentikan.")
                    else:
                        tg_send(chat_id, "ℹ️ Auto scan tidak aktif.")

                elif text in ("/status", "status"):
                    mode = "🟢 AUTO aktif" if auto_mode else "⚪ Manual"
                    tg_send(chat_id, (
                        f"📶 <b>Status Bot</b>\n\n"
                        f"Mode     : {mode}\n"
                        f"Endpoint : Binance Futures (fapi)\n"
                        f"Interval : {LOOP_INTERVAL // 60} menit\n"
                        f"Timeframe: H1 + M15"
                    ))

                else:
                    tg_send(chat_id, "❓ Tidak dikenal. Ketik /start untuk bantuan.")

            time.sleep(1)

        except Exception as e:
            log.error(f"[bot loop] {e}")
            time.sleep(5)


# ═════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════
if __name__ == "__main__":
    # Bot loop di background thread
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()

    # Flask di main thread — wajib untuk Render
    run_flask()
