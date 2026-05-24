#!/usr/bin/env python3
"""
SIGNAL BROADCASTER + SIMPLE ORDER EXECUTION (FIXED SCANNING)
"""

import os, time, hmac, hashlib, math, threading, json, requests, pandas as pd, numpy as np
from datetime import datetime
from flask import Flask

# ================== KONFIGURASI ==================
TELEGRAM_TOKEN = "8094484109:AAF9Z3lQUxdQFqqeG6NKV9O1EC0vrxzJy0U"
CHAT_ID = "8041197505"
API_KEY = os.environ.get("BINANCE_API_KEY", "")
SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")

settings = {
    "leverage": 5,
    "min_order_usd": 1.0,
    "max_price": 100.0,
    "min_confidence": 55,
    "scan_interval": 3,
    "top_coins": 30,
}

banned = {}
perma_banned = set()
bot_running = True

app = Flask(__name__)
@app.route('/')
def home():
    return "Bot is alive", 200

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except: pass

def log_activity(msg):
    send_telegram(f"📋 {msg}")

# ---------- BINANCE REST API (STABLE) ----------
def binance_request(endpoint, params=None, method="GET", auth=True):
    if params is None: params = {}
    if auth:
        try:
            server_time = requests.get("https://fapi.binance.com/fapi/v1/time", timeout=5).json()["serverTime"]
        except:
            server_time = int(time.time() * 1000)
        params["timestamp"] = server_time
        params["recvWindow"] = 10000
        qs = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(SECRET_KEY.encode(), qs.encode(), hashlib.sha256).hexdigest()
        url = f"https://fapi.binance.com{endpoint}?{qs}&signature={signature}"
    else:
        url = f"https://fapi.binance.com{endpoint}"
        if params:
            url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])

    headers = {"X-MBX-APIKEY": API_KEY} if auth else {}
    for attempt in range(5):
        try:
            time.sleep(0.3 * (attempt + 1))
            if method == "GET": r = requests.get(url, headers=headers, timeout=15)
            elif method == "POST": r = requests.post(url, headers=headers, timeout=15)
            elif method == "DELETE": r = requests.delete(url, headers=headers, timeout=15)
            else: return None

            if r.status_code == 200: return r.json()
            elif r.status_code == 418:
                log_activity("⛔ IP dibanned! Bot berhenti 15 menit.")
                time.sleep(900)
                continue
            elif r.status_code == 429:
                wait = 15 * (attempt + 1)
                log_activity(f"⏳ Rate limited, tunggu {wait}s")
                time.sleep(wait)
                continue
            else:
                try:
                    err = r.json()
                    log_activity(f"❌ API {r.status_code}: {err.get('msg','')}")
                except:
                    log_activity(f"❌ API {r.status_code}")
                return None
        except Exception as e:
            log_activity(f"⚠️ Network: {e}")
            time.sleep(5)
    return None

# ---------- ORDER FUNCTIONS ----------
def place_limit_order(symbol, side, quantity, price):
    params = {"symbol":symbol,"side":side,"type":"LIMIT","quantity":quantity,"price":price,"timeInForce":"GTC"}
    res = binance_request("/fapi/v1/order", params, method="POST")
    return res["orderId"] if res and "orderId" in res else None

def place_market_order(symbol, side, quantity):
    params = {"symbol":symbol,"side":side,"type":"MARKET","quantity":quantity}
    res = binance_request("/fapi/v1/order", params, method="POST")
    return res["orderId"] if res and "orderId" in res else None

def set_leverage(symbol, lev):
    binance_request("/fapi/v1/leverage", {"symbol":symbol,"leverage":lev}, method="POST")

def get_mark_price(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}", timeout=5)
        return float(r.json()["price"])
    except: return None

def get_symbol_filters(symbol):
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        for s in r.json()["symbols"]:
            if s["symbol"] == symbol:
                f = {}
                for fl in s["filters"]:
                    if fl["filterType"]=="PRICE_FILTER": f["tickSize"]=float(fl["tickSize"])
                    elif fl["filterType"]=="LOT_SIZE": f["stepSize"]=float(fl["stepSize"]); f["minQty"]=float(fl["minQty"])
                    elif fl["filterType"]=="MIN_NOTIONAL": f["minNotional"]=float(fl["notional"])
                return f
    except: pass
    return {"tickSize":0.01,"stepSize":0.001,"minQty":0.001,"minNotional":5.0}

def round_to_tick(v, tick):
    if tick == 0: return v
    prec = int(round(-math.log10(tick), 0))
    return round(round(v/tick)*tick, prec)

def round_to_step(v, step):
    if step == 0: return v
    prec = int(round(-math.log10(step), 0))
    return round(round(v/step)*step, prec)

def fmt_price(v, tick):
    if tick == 0: return str(v)
    prec = int(round(-math.log10(tick), 0))
    return f"{v:.{prec}f}"

def fmt_qty(v, step):
    if step == 0: return str(v)
    prec = int(round(-math.log10(step), 0))
    return f"{v:.{prec}f}"

# ---------- ANALISA TEKNIKAL LENGKAP ----------
def fetch_klines(symbol, interval, limit=200):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    for _ in range(3):
        try:
            time.sleep(0.5)
            data = binance_request(f"/fapi/v1/klines", {"symbol":symbol,"interval":interval,"limit":limit}, auth=False)
            if not data or isinstance(data, dict): return None
            df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
            df.set_index("timestamp", inplace=True)
            return df[["open","high","low","close","volume"]]
        except: time.sleep(5)
    return None

def add_indicators(df):
    if len(df) < 80: return None
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean() if len(df)>=200 else df["ema50"]
    df["tr"] = np.maximum(df["high"]-df["low"], np.maximum(abs(df["high"]-df["close"].shift()), abs(df["low"]-df["close"].shift())))
    df["atr"] = df["tr"].rolling(14).mean()
    delta = df["close"].diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain/loss; df["rsi"] = 100-(100/(1+rs))
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    return df

def market_structure(df, window=3):
    if len(df) < window*2+2: return "ranging"
    highs, lows = df["high"], df["low"]
    sh, sl = [], []
    for i in range(window, len(df)-window):
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max(): sh.append(highs.iloc[i])
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min(): sl.append(lows.iloc[i])
    if len(sh)<2 or len(sl)<2: return "ranging"
    hh = sh[-1]>sh[-2]; hl = sl[-1]>sl[-2]; lh = sh[-1]<sh[-2]; ll = sl[-1]<sl[-2]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

def detect_liquidity_sweep(df, direction):
    last = df.iloc[-2]
    if direction=="buy":
        for i in range(len(df)-5,3,-1):
            if df["low"].iloc[i]==df["low"].iloc[i-3:i+4].min():
                if last["low"] < df["low"].iloc[i] and last["close"] > df["low"].iloc[i]: return True, df["low"].iloc[i]
    else:
        for i in range(len(df)-5,3,-1):
            if df["high"].iloc[i]==df["high"].iloc[i-3:i+4].max():
                if last["high"] > df["high"].iloc[i] and last["close"] < df["high"].iloc[i]: return True, df["high"].iloc[i]
    return False, None

def find_fvg(df, direction):
    last_idx = len(df)-2
    if last_idx<3: return None
    if direction=="buy":
        if df["low"].iloc[last_idx] > df["high"].iloc[last_idx-2]: return df["high"].iloc[last_idx-2], df["low"].iloc[last_idx]
    else:
        if df["high"].iloc[last_idx] < df["low"].iloc[last_idx-2]: return df["low"].iloc[last_idx-2], df["high"].iloc[last_idx]
    return None

def find_order_block(df, direction):
    last_idx = len(df)-2; last_close = df["close"].iloc[last_idx]
    if direction=="buy":
        for i in range(last_idx-1, max(last_idx-20,0),-1):
            if df["close"].iloc[i] < df["open"].iloc[i]:
                if i+1<=last_idx and df["close"].iloc[i+1]>df["open"].iloc[i+1] and last_close>df["high"].iloc[i]: return df["high"].iloc[i], df["low"].iloc[i]
    else:
        for i in range(last_idx-1, max(last_idx-20,0),-1):
            if df["close"].iloc[i] > df["open"].iloc[i]:
                if i+1<=last_idx and df["close"].iloc[i+1]<df["open"].iloc[i+1] and last_close<df["low"].iloc[i]: return df["high"].iloc[i], df["low"].iloc[i]
    return None

def get_levels(df):
    highs, lows = df["high"], df["low"]; sh, sl = [], []
    for i in range(2, len(df)-2):
        if highs.iloc[i]==highs.iloc[i-2:i+3].max(): sh.append(highs.iloc[i])
        if lows.iloc[i]==lows.iloc[i-2:i+3].min(): sl.append(lows.iloc[i])
    return sorted(sl, reverse=True), sorted(sh)

def has_bullish_confirmation(df):
    last = df.iloc[-2]; prev = df.iloc[-3]
    if prev["close"]<prev["open"] and last["close"]>last["open"]:
        if last["close"]>prev["open"] and last["open"]<prev["close"]: return True
    body = abs(last["close"]-last["open"]); lower = min(last["open"],last["close"])-last["low"]; upper = last["high"]-max(last["open"],last["close"])
    return lower>body*1.5 and upper<body*0.5

def has_bearish_confirmation(df):
    last = df.iloc[-2]; prev = df.iloc[-3]
    if prev["close"]>prev["open"] and last["close"]<last["open"]:
        if last["open"]>prev["close"] and last["close"]<prev["open"]: return True
    body = abs(last["close"]-last["open"]); upper = last["high"]-max(last["open"],last["close"]); lower = min(last["open"],last["close"])-last["low"]
    return upper>body*1.5 and lower<body*0.5

def find_best_entry_tp_sl(df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr):
    supports_h1, resistances_h1 = get_levels(df_h1)
    supports_m15, resistances_m15 = get_levels(df_m15)
    all_supports = sorted(supports_h1+supports_m15, reverse=True)
    all_resistances = sorted(resistances_h1+resistances_m15)
    ob_h1 = find_order_block(df_h1, "buy" if bias_bull else "sell")
    ob_m15 = find_order_block(df_m15, "buy" if bias_bull else "sell")
    fvg_h1 = find_fvg(df_h1, "buy" if bias_bull else "sell")
    fvg_m15 = find_fvg(df_m15, "buy" if bias_bull else "sell")
    sup_cand, res_cand = [], []
    if bias_bull:
        if sweep_level: sup_cand.append(sweep_level)
        if ob_h1: sup_cand.append(ob_h1[0])
        if ob_m15: sup_cand.append(ob_m15[0])
        if fvg_h1: sup_cand.append(fvg_h1[0])
        if fvg_m15: sup_cand.append(fvg_m15[0])
        sup_cand.extend(all_supports); res_cand.extend(all_resistances)
    else:
        if sweep_level: res_cand.append(sweep_level)
        if ob_h1: res_cand.append(ob_h1[1])
        if ob_m15: res_cand.append(ob_m15[1])
        if fvg_h1: res_cand.append(fvg_h1[1])
        if fvg_m15: res_cand.append(fvg_m15[1])
        res_cand.extend(all_resistances); sup_cand.extend(all_supports)

    if bias_bull:
        valid_sup = [s for s in sup_cand if s < entry_raw]
        final_entry = round(max(valid_sup)+atr*0.2,6) if valid_sup else entry_raw
        sl_levels = [s-atr*0.3 for s in sup_cand if s < final_entry]
        sl = round(min(sl_levels),6) if sl_levels else round(final_entry-atr*1.5,6)
        tp_levels = [r for r in res_cand if r > final_entry]
        tp = round(min(tp_levels)*0.999,6) if tp_levels else round(final_entry+atr*2.0,6)
    else:
        valid_res = [r for r in res_cand if r > entry_raw]
        final_entry = round(min(valid_res)-atr*0.2,6) if valid_res else entry_raw
        sl_levels = [r+atr*0.3 for r in res_cand if r > final_entry]
        sl = round(max(sl_levels),6) if sl_levels else round(final_entry+atr*1.5,6)
        tp_levels = [s for s in sup_cand if s < final_entry]
        tp = round(max(tp_levels)*1.001,6) if tp_levels else round(final_entry-atr*2.0,6)
    if bias_bull:
        if sl>=final_entry: sl=round(final_entry-atr*1.0,6)
        if tp<=final_entry: tp=round(final_entry+atr*0.5,6)
    else:
        if sl<=final_entry: sl=round(final_entry+atr*1.0,6)
        if tp>=final_entry: tp=round(final_entry-atr*0.5,6)
    return final_entry, tp, sl

def analyze_signal(symbol):
    df_d1 = fetch_klines(symbol,"1d",200); df_h4 = fetch_klines(symbol,"4h",200)
    df_h1 = fetch_klines(symbol,"1h",150); df_m15 = fetch_klines(symbol,"15m",150); df_m5 = fetch_klines(symbol,"5m",150)
    if any(x is None for x in [df_d1,df_h4,df_h1,df_m15,df_m5]): return None
    df_d1=add_indicators(df_d1); df_h4=add_indicators(df_h4); df_h1=add_indicators(df_h1); df_m15=add_indicators(df_m15); df_m5=add_indicators(df_m5)
    if any(x is None for x in [df_d1,df_h4,df_h1,df_m15,df_m5]): return None
    struct_d1 = market_structure(df_d1,5)
    if struct_d1=="ranging": return None
    bias_bull = struct_d1=="bullish"; direction="BUY" if bias_bull else "SELL"; score=0
    last_d1=df_d1.iloc[-1]
    if bias_bull and last_d1["close"]>last_d1["ema50"]: score+=10
    elif not bias_bull and last_d1["close"]<last_d1["ema50"]: score+=10
    if bias_bull and 40<last_d1["rsi"]<70: score+=5
    elif not bias_bull and 30<last_d1["rsi"]<60: score+=5
    struct_h4=market_structure(df_h4,3)
    if struct_h4==struct_d1: score+=10
    if bias_bull and has_bullish_confirmation(df_h4): score+=5
    elif not bias_bull and has_bearish_confirmation(df_h4): score+=5
    sweep_h4,_=detect_liquidity_sweep(df_h4,"buy" if bias_bull else "sell")
    if sweep_h4: score+=10
    struct_h1=market_structure(df_h1,2)
    h1_aligned = (struct_h1==struct_d1)
    if h1_aligned: score+=10
    else: score-=10
    last_h1=df_h1.iloc[-2]
    if last_h1["volume"]>last_h1["vol_avg20"]: score+=5
    if bias_bull and last_h1["rsi"]>=72: return None
    if not bias_bull and last_h1["rsi"]<=28: return None
    last_m15=df_m15.iloc[-2]
    if bias_bull and last_m15["ema12"]>last_m15["ema26"]: score+=10
    elif not bias_bull and last_m15["ema12"]<last_m15["ema26"]: score+=10
    else: return None
    sweep_m15, sweep_level_m15=detect_liquidity_sweep(df_m15,"buy" if bias_bull else "sell")
    if sweep_m15: score+=15; sweep_level=sweep_level_m15
    elif sweep_h4: sweep_level=None
    else: return None
    if bias_bull and last_m15["rsi"]>=75: return None
    if not bias_bull and last_m15["rsi"]<=25: return None
    if bias_bull and has_bullish_confirmation(df_m15): score+=5
    elif not bias_bull and has_bearish_confirmation(df_m15): score+=5
    last_m5=df_m5.iloc[-2]
    if bias_bull and last_m5["close"]>last_m5["ema12"]: score+=5
    elif not bias_bull and last_m5["close"]<last_m5["ema12"]: score+=5
    else: return None
    confidence=min(score,100)
    if confidence<settings["min_confidence"]: return None
    atr=last_m15["atr"] if not np.isnan(last_m15["atr"]) else last_m15["close"]*0.002
    entry_raw=round(last_m15["close"],6)
    final_entry,tp,sl=find_best_entry_tp_sl(df_h1,df_m15,bias_bull,entry_raw,sweep_level,atr)
    if bias_bull: risk=final_entry-sl; reward=tp-final_entry
    else: risk=sl-final_entry; reward=final_entry-tp
    if risk>0 and reward/risk<1.3: return None
    return {"symbol":symbol,"signal":direction,"entry":final_entry,"tp":tp,"sl":sl,"confidence":confidence,"atr":round(atr,6),"rr":round(reward/risk,2) if risk>0 else 0}

# ---------- SCANNING (DIPERBAIKI) ----------
def get_coins():
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=15)
        if r.status_code != 200:
            log_activity(f"⚠️ Gagal ambil daftar koin (HTTP {r.status_code})")
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            log_activity("⚠️ Data koin kosong.")
            return None
        tickers = [t for t in data if t["symbol"].endswith("USDT")]
        tickers.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        res = []
        for t in tickers:
            sym = t["symbol"]
            if sym in perma_banned: continue
            if float(t["lastPrice"]) <= settings["max_price"]:
                res.append(sym)
            if len(res) >= settings["top_coins"]: break
        if not res:
            log_activity("⚠️ Tidak ada koin yang memenuhi filter harga.")
        return res
    except Exception as e:
        log_activity(f"❌ Error ambil daftar koin: {e}")
        return None

def update_banned():
    to_del = [k for k,v in banned.items() if v<=0]
    for k in to_del: del banned[k]
    for k in list(banned.keys()): banned[k] -= 1

def scan_signals():
    update_banned()
    coins = get_coins()
    if coins is None:
        log_activity("😴 Gagal ambil koin, jeda 30 detik...")
        time.sleep(30)
        return []
    if len(coins) == 0:
        log_activity("😴 Tidak ada koin tersedia, jeda 10 detik...")
        time.sleep(10)
        return []
    log_activity(f"🔍 Scanning {len(coins)} koin...")
    signals = []
    for sym in coins:
        if sym in banned or sym in perma_banned: continue
        try:
            sig = analyze_signal(sym)
            if sig:
                sig["symbol"] = sym
                signals.append(sig)
        except: pass
        time.sleep(settings["scan_interval"])
    if signals:
        best = max(signals, key=lambda x: x["confidence"])
        log_activity(f"🏆 Sinyal terbaik: {best['signal']} {best['symbol']} (Conf: {best['confidence']}%)")
        return [best]
    return []

# ---------- EKSEKUSI ORDER ----------
def execute_signal(sig):
    symbol = sig["symbol"]
    side = "BUY" if sig["signal"]=="BUY" else "SELL"
    entry, tp, sl = sig["entry"], sig["tp"], sig["sl"]
    filters = get_symbol_filters(symbol); tick=filters.get("tickSize",0.01); step=filters.get("stepSize",0.001)
    min_notional = max(filters.get("minNotional",5.0), settings["min_order_usd"])
    qty = round_to_step(min_notional/entry, step)
    if entry*qty < min_notional: qty = round_to_step(qty+step, step)
    entry_str = fmt_price(entry,tick); qty_str = fmt_qty(qty,step)
    set_leverage(symbol, settings["leverage"])
    log_activity(f"🚀 {side} {symbol} Entry: {entry_str} Qty: {qty_str} Conf: {sig['confidence']}%")
    order_id = place_limit_order(symbol, side, qty_str, entry_str)
    if not order_id:
        current = get_mark_price(symbol)
        if current:
            if (side=="BUY" and current<entry) or (side=="SELL" and current>entry):
                order_id = place_market_order(symbol, side, qty_str)
                log_activity(f"⚠️ Limit gagal, Market Order di {current}")
            else:
                order_id = place_market_order(symbol, side, qty_str)
                log_activity(f"⚠️ Limit gagal, Market Order fallback")
    if order_id:
        log_activity(f"✅ Order {side} {symbol} terpasang (ID: {order_id})")
        banned[symbol] = 20
        msg = (
            f"<b>📊 {sig['signal']} {symbol}</b>\n"
            f"Entry: {entry}\nTP: {tp} | SL: {sl}\n"
            f"Conf: {sig['confidence']}% | RR: 1:{sig['rr']} | ATR: {sig['atr']}"
        )
        send_telegram(msg)
    else:
        log_activity(f"❌ Gagal eksekusi order {symbol}")

# ---------- LOOP UTAMA ----------
def main_loop():
    global bot_running
    log_activity("🔄 Bot mulai scanning...")
    while True:
        if bot_running:
            try:
                signals = scan_signals()
                if signals:
                    execute_signal(signals[0])
                time.sleep(2)  # jeda kecil setelah siklus
            except Exception as e:
                log_activity(f"⚠️ Error: {e}")
                time.sleep(10)
        else:
            time.sleep(2)

# ---------- TELEGRAM POLLING ----------
def telegram_polling():
    global bot_running
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            params = {"timeout": 30, "offset": offset}
            resp = requests.get(url, params=params, timeout=35)
            data = resp.json()
            if not data.get("ok"): continue
            for update in data["result"]:
                offset = update["update_id"] + 1
                msg = update.get("message")
                if not msg: continue
                text = msg.get("text", "")
                if str(msg["chat"]["id"]) != CHAT_ID: continue
                if text == "/stop":
                    bot_running = False
                    send_telegram("⏹️ Bot dihentikan.")
                elif text == "/start":
                    bot_running = True
                    send_telegram("▶️ Bot dimulai.")
                elif text.startswith("/set "):
                    parts = text.split()
                    if len(parts) == 3:
                        key = parts[1]
                        try:
                            value = float(parts[2])
                            if key in settings:
                                settings[key] = value
                                send_telegram(f"⚙️ {key} diset ke {value}")
                            else: send_telegram(f"❌ Key tidak dikenal: {key}")
                        except ValueError: send_telegram("❌ Value harus berupa angka.")
                elif text == "/settings":
                    send_telegram(f"<pre>{json.dumps(settings, indent=2)}</pre>")
                elif text == "/status":
                    send_telegram(f"📊 Bot: {'Running' if bot_running else 'Stopped'} | Banned: {len(banned)} koin")
                elif text == "/menu":
                    send_telegram("""<b>Command List:</b>
/start - Mulai bot
/stop - Hentikan bot
/status - Status bot
/settings - Lihat pengaturan
/set key value - Ubah pengaturan
/menu - Tampilkan menu""")
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}"); time.sleep(5)

# ---------- STARTUP ----------
if __name__ == "__main__":
    log_activity("🤖 Bot starting...")
    try:
        ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
        log_activity(f"🚀 Bot dimulai!\nIP: {ip}\nPastikan IP di-whitelist di Binance.")
    except: log_activity("🚀 Bot dimulai! (IP tidak terdeteksi)")
    if not API_KEY or not SECRET_KEY:
        log_activity("❌ API Key/Secret belum diset.")
    threading.Thread(target=telegram_polling, daemon=True).start()
    threading.Thread(target=main_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
