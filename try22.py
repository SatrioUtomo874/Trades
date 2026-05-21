#!/usr/bin/env python3
"""
AUTO TRADING BOT – LIMIT ORDER + AUTO TP/SL + BAN 20 SIKLUS
Langsung pasang limit order, setelah terisi pasang TP/SL, lalu lanjut scan.
"""

import os, time, hmac, hashlib, math, threading, requests, pandas as pd, numpy as np
from datetime import datetime
from flask import Flask

# ================== KONFIGURASI ==================
TELEGRAM_TOKEN = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
CHAT_ID = "8041197505"
TP_PERCENT = 0.6
SL_PERCENT = 0.85
MIN_CONFIDENCE = 65
LEVERAGE = 5
TIMEOUT_MINUTES = 15
SCAN_INTERVAL = 60
MAX_PRICE_USDT = 100.0
BAN_CYCLES = 20
API_KEY = os.environ.get("BINANCE_API_KEY", "")
SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")
# =================================================

app = Flask(__name__)
@app.route('/')
def home():
    return "Bot is alive", 200

# ---------- GLOBAL BAN STATE ----------
banned_coins = {}  # {symbol: cycles_left}

def update_banned():
    to_delete = []
    for sym in list(banned_coins.keys()):
        banned_coins[sym] -= 1
        if banned_coins[sym] <= 0:
            to_delete.append(sym)
    for sym in to_delete:
        del banned_coins[sym]

def ban_coin(symbol):
    banned_coins[symbol] = BAN_CYCLES
    send_telegram(f"📛 {symbol} di-ban {BAN_CYCLES} siklus.")

# ---------- TELEGRAM ----------
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=8)
    except: pass

def get_public_ip():
    try: return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except: return "IP tidak terdeteksi"

def signed_request(endpoint, params, method="GET"):
    if not API_KEY or not SECRET_KEY:
        send_telegram("❌ API Key/Secret belum diisi."); return None
    params["timestamp"] = int(time.time()*1000)
    qs = "&".join([f"{k}={v}" for k,v in sorted(params.items())])
    sig = hmac.new(SECRET_KEY.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"https://fapi.binance.com{endpoint}?{qs}&signature={sig}"
    try:
        if method=="GET": r = requests.get(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
        elif method=="POST": r = requests.post(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
        elif method=="DELETE": r = requests.delete(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
        else: return None
        if r.status_code==200: return r.json()
        else:
            try: msg = r.json().get("msg","")
            except: msg = r.text
            send_telegram(f"❌ API Error {r.status_code}\n{msg}")
            return None
    except Exception as e:
        send_telegram(f"⚠️ Network: {e}"); return None

# ---------- DATA ----------
def fetch_klines(symbol, interval, limit=100):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    for _ in range(3):
        try:
            resp = requests.get(url, timeout=8); data = resp.json()
            if isinstance(data, dict) and "code" in data: return None
            df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
            df.set_index("timestamp", inplace=True)
            return df[["open","high","low","close","volume"]]
        except: time.sleep(10)
    return None

def add_indicators(df):
    if len(df)<80: return None
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean() if len(df)>=200 else df["ema50"]
    df["atr"] = df["high"].sub(df["low"]).rolling(14).mean()
    delta = df["close"].diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain/loss; df["rsi"] = 100 - (100/(1+rs))
    return df

def market_structure(df, w=3):
    if len(df)<w*2+2: return "ranging"
    h,l = df["high"], df["low"]; sh,sl = [],[]
    for i in range(w, len(df)-w):
        if h.iloc[i]==h.iloc[i-w:i+w+1].max(): sh.append(h.iloc[i])
        if l.iloc[i]==l.iloc[i-w:i+w+1].min(): sl.append(l.iloc[i])
    if len(sh)<2 or len(sl)<2: return "ranging"
    hh = sh[-1]>sh[-2]; hl = sl[-1]>sl[-2]; lh = sh[-1]<sh[-2]; ll = sl[-1]<sl[-2]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

def liquidity_sweep(df, direction):
    last = df.iloc[-2]
    if direction=="buy":
        for i in range(len(df)-4,3,-1):
            if df["low"].iloc[i]==df["low"].iloc[i-3:i+4].min():
                if last["low"] < df["low"].iloc[i] and last["close"] > df["low"].iloc[i]: return True, df["low"].iloc[i]
    else:
        for i in range(len(df)-4,3,-1):
            if df["high"].iloc[i]==df["high"].iloc[i-3:i+4].max():
                if last["high"] > df["high"].iloc[i] and last["close"] < df["high"].iloc[i]: return True, df["high"].iloc[i]
    return False, None

def order_block_or_fvg(df, direction):
    last_idx = len(df)-2; last_close = df["close"].iloc[last_idx]
    if direction=="buy":
        for i in range(last_idx-1, max(last_idx-20,0),-1):
            if df["close"].iloc[i] < df["open"].iloc[i]:
                if i+1<=last_idx and df["close"].iloc[i+1] > df["open"].iloc[i+1] and last_close > df["high"].iloc[i]: return True, df["high"].iloc[i]
        if last_idx>=2 and df["low"].iloc[last_idx] > df["high"].iloc[last_idx-2]: return True, df["high"].iloc[last_idx-2]
    else:
        for i in range(last_idx-1, max(last_idx-20,0),-1):
            if df["close"].iloc[i] > df["open"].iloc[i]:
                if i+1<=last_idx and df["close"].iloc[i+1] < df["open"].iloc[i+1] and last_close < df["low"].iloc[i]: return True, df["low"].iloc[i]
        if last_idx>=2 and df["high"].iloc[last_idx] < df["low"].iloc[last_idx-2]: return True, df["low"].iloc[last_idx-2]
    return False, None

def premium_discount_zone(df):
    mid = (df["high"].max() + df["low"].min())/2
    return "premium" if df["close"].iloc[-2] > mid else "discount"

def generate_signal(df_h1, df_m15, df_m5):
    df_h1 = add_indicators(df_h1); df_m15 = add_indicators(df_m15); df_m5 = add_indicators(df_m5)
    if df_h1 is None or df_m15 is None or df_m5 is None: return None
    atr_now = df_m15["atr"].iloc[-2]; atr_mean = df_m15["atr"].rolling(50).mean().iloc[-2]
    if pd.notna(atr_mean) and atr_now > 2.5*atr_mean: return None
    struct_h1 = market_structure(df_h1,5)
    if struct_h1=="ranging": return None
    bias_bull = struct_h1=="bullish"; direction = "buy" if bias_bull else "sell"
    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["ema12"] <= last_m5["ema26"]: return None
    if not bias_bull and last_m5["ema12"] >= last_m5["ema26"]: return None
    score = 0.25
    sweep_ok, _ = liquidity_sweep(df_m15, direction)
    if not sweep_ok: return None
    score += 0.25
    area_ok, _ = order_block_or_fvg(df_m15, direction)
    if area_ok: score += 0.15
    zone = premium_discount_zone(df_m15)
    if (bias_bull and zone=="discount") or (not bias_bull and zone=="premium"): score += 0.1
    struct_m5 = market_structure(df_m5,2)
    if (bias_bull and struct_m5=="bullish") or (not bias_bull and struct_m5=="bearish"): score += 0.1
    if score*100 < MIN_CONFIDENCE: return None
    last = df_m15.iloc[-2]; price = last["close"]
    return {"signal":"BUY" if bias_bull else "SELL", "entry_signal":round(price,6), "confidence":int(score*100)}

def get_coins_by_volume(top=50, max_price=100.0):
    for _ in range(3):
        try:
            resp = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10)
            tickers = [t for t in resp.json() if t["symbol"].endswith("USDT")]
            tickers.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
            res = []
            for t in tickers:
                if float(t["lastPrice"]) <= max_price: res.append(t["symbol"])
                if len(res)>=top: break
            return res
        except: time.sleep(10)
    return []

# ---------- PRESISI ----------
symbol_filters_cache = {}
def get_symbol_filters(symbol):
    if symbol in symbol_filters_cache: return symbol_filters_cache[symbol]
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        for s in resp.json()["symbols"]:
            if s["symbol"]==symbol:
                f = {}
                for fl in s["filters"]:
                    if fl["filterType"]=="PRICE_FILTER": f["tickSize"] = float(fl["tickSize"])
                    elif fl["filterType"]=="LOT_SIZE": f["stepSize"] = float(fl["stepSize"]); f["minQty"] = float(fl["minQty"])
                    elif fl["filterType"]=="MIN_NOTIONAL": f["minNotional"] = float(fl["notional"])
                symbol_filters_cache[symbol]=f; return f
    except: pass
    return None

def round_to_tick(v, tick):
    if tick==0: return v
    prec = int(round(-math.log10(tick),0))
    return round(round(v/tick)*tick, prec)

def round_to_step(v, step):
    if step==0: return v
    prec = int(round(-math.log10(step),0))
    return round(round(v/step)*step, prec)

def fmt_tick(v, tick):
    if tick==0: return str(v)
    prec = int(round(-math.log10(tick),0))
    return f"{v:.{prec}f}"

def fmt_step(v, step):
    if step==0: return str(v)
    prec = int(round(-math.log10(step),0))
    return f"{v:.{prec}f}"

# ---------- ORDER FUNCTIONS ----------
def get_mark_price(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}", timeout=5)
        return float(r.json()["price"])
    except: return None

def set_leverage(symbol, leverage):
    params = {"symbol":symbol, "leverage":leverage}
    signed_request("/fapi/v1/leverage", params, method="POST")

def place_limit_order(symbol, side, quantity, price):
    params = {
        "symbol":symbol, "side":side, "type":"LIMIT",
        "quantity":quantity, "price":price, "timeInForce":"GTC"
    }
    res = signed_request("/fapi/v1/order", params, method="POST")
    if res and "orderId" in res: return res["orderId"]
    return None

def cancel_order(symbol, order_id):
    params = {"symbol":symbol, "orderId":order_id}
    return signed_request("/fapi/v1/order", params, method="DELETE") is not None

def get_order_status(symbol, order_id):
    params = {"symbol":symbol, "orderId":order_id}
    res = signed_request("/fapi/v1/order", params, method="GET")
    if res and "status" in res: return res["status"]
    return None

def place_tp_sl_algo(symbol, side, quantity, stop_loss_price, take_profit_price):
    # SL: STOP_MARKET, TP: TAKE_PROFIT_MARKET
    res1 = signed_request("/fapi/v1/algoOrder", {
        "algoType":"CONDITIONAL","symbol":symbol,"side":"SELL" if side=="BUY" else "BUY",
        "type":"STOP_MARKET","quantity":quantity,"triggerPrice":stop_loss_price,
        "workingType":"MARK_PRICE"
    }, method="POST")
    res2 = signed_request("/fapi/v1/algoOrder", {
        "algoType":"CONDITIONAL","symbol":symbol,"side":"SELL" if side=="BUY" else "BUY",
        "type":"TAKE_PROFIT_MARKET","quantity":quantity,"triggerPrice":take_profit_price,
        "workingType":"MARK_PRICE"
    }, method="POST")
    return (res1 is not None and "algoId" in res1) and (res2 is not None and "algoId" in res2)

# ---------- TRADING CYCLE ----------
def trading_cycle():
    update_banned()  # Kurangi hitungan ban

    coins = get_coins_by_volume(50, MAX_PRICE_USDT)
    if not coins:
        send_telegram("❌ Gagal ambil daftar koin."); return

    signals = []
    for sym in coins:
        if sym in banned_coins: continue  # skip koin yang sedang diban
        try:
            h1 = fetch_klines(sym, "1h", 100); m15 = fetch_klines(sym, "15m", 100); m5 = fetch_klines(sym, "5m", 100)
            if not all([h1 is not None, m15 is not None, m5 is not None]): continue
            sig = generate_signal(h1, m15, m5)
            if sig: sig["symbol"]=sym; signals.append(sig)
        except: pass
        time.sleep(0.03)

    if not signals:
        send_telegram("❌ Tidak ada sinyal dengan Confidence ≥ 65%"); return

    best = max(signals, key=lambda x: x["confidence"])
    symbol = best["symbol"]; side = "BUY" if best["signal"]=="BUY" else "SELL"
    entry_raw = best["entry_signal"]; confidence = best["confidence"]

    filters = get_symbol_filters(symbol)
    if not filters: send_telegram("❌ Gagal ambil filter."); return
    tick = filters["tickSize"]; step = filters["stepSize"]; min_qty = filters["minQty"]; min_notional = filters["minNotional"]

    entry = round_to_tick(entry_raw, tick)
    if side=="BUY":
        tp_raw = entry*(1+TP_PERCENT/100); sl_raw = entry*(1-SL_PERCENT/100)
    else:
        tp_raw = entry*(1-TP_PERCENT/100); sl_raw = entry*(1+SL_PERCENT/100)
    tp = round_to_tick(tp_raw, tick); sl = round_to_tick(sl_raw, tick)

    raw_qty = max(min_notional/entry, min_qty)
    qty = round_to_step(raw_qty, step)
    if qty < min_qty:
        qty = min_qty; qty = round_to_step(qty, step)
        if qty < min_qty: qty += step; qty = round_to_step(qty, step)

    entry_str = fmt_tick(entry, tick); tp_str = fmt_tick(tp, tick); sl_str = fmt_tick(sl, tick); qty_str = fmt_step(qty, step)

    send_telegram(f"📊 {best['signal']} {symbol} | Entry: {entry_str} | TP: {tp_str} | SL: {sl_str} | Qty: {qty_str} | Conf: {confidence}%")

    set_leverage(symbol, LEVERAGE)

    # Langsung pasang limit order
    order_id = place_limit_order(symbol, side, qty_str, entry_str)
    if not order_id:
        send_telegram("❌ Gagal pasang Limit Order."); return
    send_telegram(f"📌 Limit Order terpasang (ID: {order_id}) – menunggu harga sentuh {entry_str}")

    # Pantau hingga terisi, TP/SL tersentuh duluan, atau timeout
    start_time = time.time(); last_notify = time.time()
    while True:
        time.sleep(1)
        mark = get_mark_price(symbol)
        if mark is None: continue

        if side=="BUY":
            if mark >= tp:
                cancel_order(symbol, order_id); send_telegram(f"⚠️ TP ({tp_str}) tersentuh sebelum entry. Order dibatalkan."); ban_coin(symbol); return
            if mark <= sl:
                cancel_order(symbol, order_id); send_telegram(f"⚠️ SL ({sl_str}) tersentuh sebelum entry. Order dibatalkan."); ban_coin(symbol); return
        else:
            if mark <= tp:
                cancel_order(symbol, order_id); send_telegram(f"⚠️ TP ({tp_str}) tersentuh sebelum entry. Order dibatalkan."); ban_coin(symbol); return
            if mark >= sl:
                cancel_order(symbol, order_id); send_telegram(f"⚠️ SL ({sl_str}) tersentuh sebelum entry. Order dibatalkan."); ban_coin(symbol); return

        status = get_order_status(symbol, order_id)
        if status == "FILLED":
            send_telegram(f"✅ Limit Order {symbol} terisi di {entry_str}. Memasang TP/SL...")
            if place_tp_sl_algo(symbol, side, qty_str, sl_str, tp_str):
                send_telegram(f"🎯 TP/SL terpasang: TP {tp_str} | SL {sl_str}")
            else:
                send_telegram("⚠️ Gagal pasang TP/SL otomatis, cek manual.")
            ban_coin(symbol)  # ban setelah order terisi
            return  # Langsung lanjut scan (siklus selesai)
        elif status in ("CANCELED", "EXPIRED", "REJECTED"):
            send_telegram(f"⚠️ Order {symbol} berstatus {status}."); ban_coin(symbol); return

        if time.time() - last_notify >= 180:
            send_telegram(f"⏳ {symbol} | Mark: {mark:.6f} | Limit: {entry_str} | TP: {tp_str} | SL: {sl_str}")
            last_notify = time.time()

        if time.time() - start_time > TIMEOUT_MINUTES*60:
            cancel_order(symbol, order_id); send_telegram(f"⏰ Timeout, order {symbol} dibatalkan."); ban_coin(symbol); return

# ---------- LOOP ----------
def run_loop():
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Siklus...")
            trading_cycle()
        except Exception as e:
            send_telegram(f"⚠️ Bot error: {e}")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    ip = get_public_ip()
    send_telegram(f"🚀 Bot Auto-Trading (Limit Order) dimulai!\nIP: {ip}\nPastikan IP di-whitelist.")
    if not API_KEY or not SECRET_KEY:
        send_telegram("❌ API Key/Secret belum diset di environment.")
    else:
        t = threading.Thread(target=run_loop, daemon=True); t.start()
    port = int(os.environ.get("PORT",8080))
    app.run(host="0.0.0.0", port=port)
