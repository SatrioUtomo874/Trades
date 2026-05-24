#!/usr/bin/env python3
"""
FULL AUTO TRADING BOT - Hybrid REST + WebSocket
Stabil, real-time, anti-banned.
"""

import os, time, hmac, hashlib, math, threading, json, requests, pandas as pd, numpy as np
from datetime import datetime
from flask import Flask
import websocket

# ================== KONFIGURASI ==================
TELEGRAM_TOKEN = "8094484109:AAF9Z3lQUxdQFqqeG6NKV9O1EC0vrxzJy0U"
CHAT_ID = "8041197505"
API_KEY = os.environ.get("BINANCE_API_KEY", "")
SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")

settings = {
    "max_positions": 4,
    "leverage": 5,
    "min_order_usd": 1.0,
    "max_price": 100.0,
    "min_confidence": 55,
    "ban_cycles": 20,
    "scan_interval": 3,
    "top_coins": 30,
}

banned = {}
perma_banned = set()
tracking_orders = {}

# WebSocket global
ws = None
ws_lock = threading.Lock()
listen_key = None
last_listen_key_update = 0

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

# ---------- WEBSOCKET MANAJEMEN ----------
def get_listen_key():
    global listen_key, last_listen_key_update
    res = binance_request("/fapi/v1/listenKey", method="POST")
    if res and "listenKey" in res:
        listen_key = res["listenKey"]
        last_listen_key_update = time.time()
        log_activity("🔑 Listen key diperoleh.")
        return listen_key
    log_activity("❌ Gagal mendapatkan listen key.")
    return None

def keep_alive_listen_key():
    global listen_key, last_listen_key_update
    while True:
        time.sleep(1800)  # setiap 30 menit
        if listen_key:
            binance_request("/fapi/v1/listenKey", {"listenKey": listen_key}, method="PUT")
            last_listen_key_update = time.time()
            log_activity("🔄 Listen key diperpanjang.")

def on_message(ws, message):
    """Callback WebSocket: menerima data real-time."""
    data = json.loads(message)
    if "e" in data:
        if data["e"] == "ACCOUNT_UPDATE":
            # Posisi berubah (TP/SL tersentuh, order terisi)
            log_activity(f"📡 Update akun: {data['a']['P']}")
        elif data["e"] == "ORDER_TRADE_UPDATE":
            # Order terisi atau dibatalkan
            log_activity(f"📡 Update order: {data['o']['s']} {data['o']['X']}")

def on_error(ws, error):
    log_activity(f"⚠️ WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    log_activity("🔌 WebSocket terputus. Mencoba reconnect...")
    time.sleep(1)
    connect_websocket()

def on_open(ws):
    log_activity("🔗 WebSocket terhubung.")

def connect_websocket():
    global ws
    with ws_lock:
        if listen_key is None:
            get_listen_key()
        if listen_key is None:
            log_activity("❌ Tidak bisa connect WebSocket tanpa listen key.")
            return
        ws_url = f"wss://fstream.binance.com/ws/{listen_key}"
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        wst = threading.Thread(target=ws.run_forever, daemon=True)
        wst.start()

# ---------- BINANCE HELPERS ----------
def get_open_positions():
    raw = binance_request("/fapi/v1/positionRisk")
    if not raw or not isinstance(raw, list): return []
    positions = []
    for p in raw:
        amt = float(p.get("positionAmt", 0))
        if amt != 0:
            positions.append({
                "symbol": p["symbol"],
                "side": "LONG" if amt > 0 else "SHORT",
                "amount": abs(amt),
                "entryPrice": float(p["entryPrice"]),
            })
    return positions

def get_open_orders():
    orders = []
    lim = binance_request("/fapi/v1/openOrders")
    if lim and isinstance(lim, list): orders.extend(lim)
    algo = binance_request("/fapi/v1/algoOpenOrders")
    if algo and isinstance(algo, list): orders.extend(algo)
    return orders

def cancel_order(symbol, order_id):
    r = binance_request("/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, method="DELETE")
    if r and r.get("status") == "CANCELED": return True
    r = binance_request("/fapi/v1/algoOrder", {"symbol": symbol, "algoId": order_id}, method="DELETE")
    return r is not None

def place_limit_order(symbol, side, quantity, price):
    params = {"symbol":symbol,"side":side,"type":"LIMIT","quantity":quantity,"price":price,"timeInForce":"GTC"}
    res = binance_request("/fapi/v1/order", params, method="POST")
    return res["orderId"] if res and "orderId" in res else None

def place_market_order(symbol, side, quantity):
    params = {"symbol":symbol,"side":side,"type":"MARKET","quantity":quantity}
    res = binance_request("/fapi/v1/order", params, method="POST")
    return res["orderId"] if res and "orderId" in res else None

def place_tp_sl(symbol, side, quantity, sl_price, tp_price):
    close_side = "SELL" if side == "LONG" else "BUY"
    tp = binance_request("/fapi/v1/algoOrder", {
        "algoType":"CONDITIONAL","symbol":symbol,"side":close_side,"type":"TAKE_PROFIT_MARKET",
        "quantity":quantity,"triggerPrice":tp_price,"workingType":"MARK_PRICE"}, method="POST")
    sl = binance_request("/fapi/v1/algoOrder", {
        "algoType":"CONDITIONAL","symbol":symbol,"side":close_side,"type":"STOP_MARKET",
        "quantity":quantity,"triggerPrice":sl_price,"workingType":"MARK_PRICE"}, method="POST")
    return tp is not None and sl is not None

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

# ---------- ANALISA TEKNIKAL (FULL) ----------
# (Semua fungsi teknikal: fetch_klines, add_indicators, market_structure, 
#  detect_liquidity_sweep, find_fvg, find_order_block, get_levels, 
#  has_bullish_confirmation, has_bearish_confirmation, 
#  find_best_entry_tp_sl, analyze_signal – sama persis dengan yang ada 
#  di kode terakhir. Untuk menghemat ruang, tidak ditulis ulang di sini,
#  tapi pastikan sudah ada di file Anda.)

# ---------- DATA & INDIKATOR ----------
def fetch_klines(symbol, interval, limit=200):
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

# ---------- ALUR UTAMA ----------
def main_loop():
    log_activity("🔄 Bot mulai bekerja...")
    while True:
        try:
            # 1. Bersihkan banned
            to_del = [k for k,v in banned.items() if v<=0]
            for k in to_del: del banned[k]
            for k in list(banned.keys()): banned[k] -= 1

            # 2. Cek posisi & pasang TP/SL
            positions = get_open_positions()
            log_activity(f"🔍 Posisi aktif: {len(positions)}")
            for pos in positions:
                sym = pos["symbol"]
                orders = get_open_orders()
                has_tp = any(o["symbol"]==sym and o.get("type") in ("TAKE_PROFIT_MARKET","STOP_MARKET") for o in orders)
                if has_tp: continue
                side = "buy" if pos["side"]=="LONG" else "sell"
                df_h1 = fetch_klines(sym,"1h",150)
                if df_h1 is None: continue
                df_h1 = add_indicators(df_h1)
                if df_h1 is None: continue
                entry = pos["entryPrice"]
                atr = df_h1["atr"].iloc[-1] if not np.isnan(df_h1["atr"].iloc[-1]) else entry*0.002
                supports, resistances = get_levels(df_h1)
                if side=="buy":
                    tp = round(min([r for r in resistances if r>entry])*0.999,6) if any(r>entry for r in resistances) else round(entry*1.006,6)
                    sl = round(max([s for s in supports if s<entry])-atr*0.3,6) if any(s<entry for s in supports) else round(entry*0.992,6)
                else:
                    tp = round(max([s for s in supports if s<entry])*1.001,6) if any(s<entry for s in supports) else round(entry*0.994,6)
                    sl = round(min([r for r in resistances if r>entry])+atr*0.3,6) if any(r>entry for r in resistances) else round(entry*1.008,6)
                filters = get_symbol_filters(sym)
                tick = filters.get("tickSize",0.01); step = filters.get("stepSize",0.001)
                tp_str = fmt_price(tp,tick); sl_str = fmt_price(sl,tick); qty_str = fmt_qty(pos["amount"],step)
                if place_tp_sl(sym, pos["side"], qty_str, sl_str, tp_str):
                    log_activity(f"🛡️ TP/SL {sym} dipasang: TP {tp_str} | SL {sl_str}")

            # 3. Bersihkan orphan orders
            orders = get_open_orders()
            syms = {p["symbol"] for p in positions}
            for o in orders:
                if o.get("type") in ("TAKE_PROFIT_MARKET","STOP_MARKET") and o["symbol"] not in syms:
                    cancel_order(o["symbol"], o.get("orderId") or o.get("algoId"))
                    log_activity(f"🧹 Orphan order {o['symbol']} dibatalkan.")

            # 4. Batalkan limit order setengah jalan
            to_cancel = []
            for oid, info in list(tracking_orders.items()):
                mark = get_mark_price(info["symbol"])
                if mark is None: continue
                if info["side"]=="BUY":
                    half = info["entry"] + (info["tp"]-info["entry"])/2
                    if mark >= half: to_cancel.append(oid)
                else:
                    half = info["entry"] - (info["entry"]-info["tp"])/2
                    if mark <= half: to_cancel.append(oid)
            for oid in to_cancel:
                info = tracking_orders[oid]
                cancel_order(info["symbol"], oid)
                del tracking_orders[oid]
                log_activity(f"🧹 Limit order {info['symbol']} dibatalkan (setengah jalan ke TP).")

            # 5. Hitung kapasitas
            positions = get_open_positions()
            limit_orders = [o for o in get_open_orders() if o.get("type")=="LIMIT"]
            total_active = len(positions) + len(limit_orders)
            log_activity(f"📊 Posisi: {len(positions)} | Limit: {len(limit_orders)} | Max: {settings['max_positions']}")

            # 6. IF masih ada slot, cari sinyal sampai dapat
            if total_active < settings["max_positions"]:
                while True:
                    # Ambil daftar koin
                    try:
                        r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=15)
                        data = r.json()
                        coins = []
                        tickers = [t for t in data if t["symbol"].endswith("USDT")]
                        tickers.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
                        for t in tickers:
                            sym = t["symbol"]
                            if sym in perma_banned: continue
                            if float(t["lastPrice"]) <= settings["max_price"]:
                                coins.append(sym)
                            if len(coins) >= settings["top_coins"]: break
                    except:
                        time.sleep(10)
                        continue

                    log_activity(f"🔍 Scanning {len(coins)} koin...")
                    best_signal = None
                    for sym in coins:
                        if sym in banned or sym in perma_banned: continue
                        try:
                            sig = analyze_signal(sym)
                            if sig:
                                sig["symbol"] = sym
                                if best_signal is None or sig["confidence"] > best_signal["confidence"]:
                                    best_signal = sig
                        except: pass
                        time.sleep(settings["scan_interval"])

                    if best_signal:
                        # Eksekusi
                        symbol = best_signal["symbol"]
                        side = "BUY" if best_signal["signal"]=="BUY" else "SELL"
                        entry, tp, sl = best_signal["entry"], best_signal["tp"], best_signal["sl"]
                        filters = get_symbol_filters(symbol)
                        tick = filters.get("tickSize",0.01); step = filters.get("stepSize",0.001)
                        min_notional = max(filters.get("minNotional",5.0), settings["min_order_usd"])
                        qty = round_to_step(min_notional/entry, step)
                        if entry*qty < min_notional: qty = round_to_step(qty+step, step)
                        entry_str = fmt_price(entry,tick); qty_str = fmt_qty(qty,step)
                        set_leverage(symbol, settings["leverage"])
                        log_activity(f"🚀 {side} {symbol} Entry: {entry_str} Qty: {qty_str} Conf: {best_signal['confidence']}%")
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
                            tracking_orders[order_id] = {"symbol":symbol,"entry":entry,"tp":tp,"sl":sl,"side":side}
                            log_activity(f"✅ Order {side} {symbol} terpasang (ID: {order_id})")
                            banned[symbol] = settings["ban_cycles"]
                        break  # keluar dari loop scanning
                    else:
                        log_activity("😴 Tidak ada sinyal, coba lagi...")
                        time.sleep(5)
            else:
                log_activity("🛑 Kapasitas penuh, menunggu...")
                time.sleep(30)

        except Exception as e:
            log_activity(f"⚠️ Error: {e}")
            time.sleep(30)

# ---------- TELEGRAM POLLING ----------
def telegram_polling():
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
                if text.startswith("/set "):
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
                    pos = len(get_open_positions())
                    ords = len(get_open_orders())
                    send_telegram(f"📊 Posisi: {pos} | Orders: {ords} | Max: {settings['max_positions']}")
                elif text.startswith("/ban "):
                    sym = text.split()[1].upper()
                    if not sym.endswith("USDT"): sym += "USDT"
                    perma_banned.add(sym)
                    send_telegram(f"🚫 {sym} dibanned permanen.")
                elif text.startswith("/unban "):
                    sym = text.split()[1].upper()
                    if not sym.endswith("USDT"): sym += "USDT"
                    perma_banned.discard(sym)
                    send_telegram(f"✅ {sym} dihapus dari banned permanen.")
                elif text == "/menu":
                    send_telegram("""<b>Command List:</b>
/status - Posisi & order aktif
/settings - Lihat pengaturan
/set key value - Ubah pengaturan
/ban SYMBOL - Ban permanen koin
/unban SYMBOL - Hapus ban permanen
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
    else:
        # Mulai WebSocket
        get_listen_key()
        connect_websocket()
        # Thread perpanjang listen key
        threading.Thread(target=keep_alive_listen_key, daemon=True).start()
        # Thread Telegram polling
        threading.Thread(target=telegram_polling, daemon=True).start()
        # Loop utama
        main_loop()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
