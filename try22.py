#!/usr/bin/env python3
"""
AUTO TRADING BOT – Order Bersyarat (Algo) + Presisi Harga/Quantity
"""

import os
import time
import hmac
import hashlib
import json
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask

# ================== KONFIGURASI ==================
TELEGRAM_TOKEN = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
CHAT_ID = "8041197505"

TP_PERCENT = 0.6
SL_PERCENT = 0.80
MIN_CONFIDENCE = 60
LEVERAGE = 5
TIMEOUT_MINUTES = 15
SCAN_INTERVAL = 60
MAX_PRICE_USDT = 100.0

API_KEY = os.environ.get("BINANCE_API_KEY", "")
SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")
# =================================================

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive", 200

# ---------- TELEGRAM ----------
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=8)
    except:
        pass

def get_public_ip():
    try:
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except:
        return "IP tidak terdeteksi"

# ---------- BINANCE SIGNED REQUEST ----------
def signed_request(endpoint, params, method="GET"):
    if not API_KEY or not SECRET_KEY:
        send_telegram("❌ API Key/Secret belum diisi.")
        return None

    params["timestamp"] = int(time.time() * 1000)
    query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    signature = hmac.new(SECRET_KEY.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    full_url = f"https://fapi.binance.com{endpoint}?{query_string}&signature={signature}"

    try:
        if method == "GET":
            resp = requests.get(full_url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
        elif method == "POST":
            resp = requests.post(full_url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
        elif method == "DELETE":
            resp = requests.delete(full_url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
        else:
            return None

        if resp.status_code == 200:
            return resp.json()
        else:
            try:
                err = resp.json()
                msg = err.get("msg", "Tidak ada pesan error")
            except:
                msg = resp.text
            send_telegram(f"❌ API Error {resp.status_code} pada {endpoint}\n{msg}")
            return None
    except Exception as e:
        send_telegram(f"⚠️ Network error: {e}")
        return None

# ---------- DATA & INDIKATOR ----------
def fetch_klines(symbol, interval, limit=100):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    for _ in range(3):
        try:
            resp = requests.get(url, timeout=8)
            data = resp.json()
            if isinstance(data, dict) and "code" in data:
                return None
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.set_index("timestamp", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]
        except:
            time.sleep(10)
    return None

def add_indicators(df):
    if len(df) < 80:
        return None
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean() if len(df) >= 200 else df["ema50"]
    df["atr"] = df["high"].sub(df["low"]).rolling(14).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def market_structure(df, window=3):
    if len(df) < window*2+2:
        return "ranging"
    highs, lows = df["high"], df["low"]
    sh, sl = [], []
    for i in range(window, len(df)-window):
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
            sh.append(highs.iloc[i])
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
            sl.append(lows.iloc[i])
    if len(sh) < 2 or len(sl) < 2:
        return "ranging"
    hh = sh[-1] > sh[-2]
    hl = sl[-1] > sl[-2]
    lh = sh[-1] < sh[-2]
    ll = sl[-1] < sl[-2]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

def liquidity_sweep(df, direction):
    last = df.iloc[-2]
    if direction == "buy":
        for i in range(len(df)-4, 3, -1):
            if df["low"].iloc[i] == df["low"].iloc[i-3:i+4].min():
                if last["low"] < df["low"].iloc[i] and last["close"] > df["low"].iloc[i]:
                    return True, df["low"].iloc[i]
    else:
        for i in range(len(df)-4, 3, -1):
            if df["high"].iloc[i] == df["high"].iloc[i-3:i+4].max():
                if last["high"] > df["high"].iloc[i] and last["close"] < df["high"].iloc[i]:
                    return True, df["high"].iloc[i]
    return False, None

def order_block_or_fvg(df, direction):
    last_idx = len(df)-2
    last_close = df["close"].iloc[last_idx]
    if direction == "buy":
        for i in range(last_idx-1, max(last_idx-20,0), -1):
            if df["close"].iloc[i] < df["open"].iloc[i]:
                if i+1 <= last_idx and df["close"].iloc[i+1] > df["open"].iloc[i+1] and last_close > df["high"].iloc[i]:
                    return True, df["high"].iloc[i]
        if last_idx >= 2 and df["low"].iloc[last_idx] > df["high"].iloc[last_idx-2]:
            return True, df["high"].iloc[last_idx-2]
    else:
        for i in range(last_idx-1, max(last_idx-20,0), -1):
            if df["close"].iloc[i] > df["open"].iloc[i]:
                if i+1 <= last_idx and df["close"].iloc[i+1] < df["open"].iloc[i+1] and last_close < df["low"].iloc[i]:
                    return True, df["low"].iloc[i]
        if last_idx >= 2 and df["high"].iloc[last_idx] < df["low"].iloc[last_idx-2]:
            return True, df["low"].iloc[last_idx-2]
    return False, None

def premium_discount_zone(df):
    mid = (df["high"].max() + df["low"].min()) / 2
    return "premium" if df["close"].iloc[-2] > mid else "discount"

def generate_signal(df_h1, df_m15, df_m5):
    df_h1 = add_indicators(df_h1)
    df_m15 = add_indicators(df_m15)
    df_m5 = add_indicators(df_m5)
    if df_h1 is None or df_m15 is None or df_m5 is None:
        return None

    atr_now = df_m15["atr"].iloc[-2]
    atr_mean = df_m15["atr"].rolling(50).mean().iloc[-2]
    if pd.notna(atr_mean) and atr_now > 2.5 * atr_mean:
        return None

    struct_h1 = market_structure(df_h1, 5)
    if struct_h1 == "ranging":
        return None
    bias_bull = struct_h1 == "bullish"
    direction = "buy" if bias_bull else "sell"

    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["ema12"] <= last_m5["ema26"]:
        return None
    if not bias_bull and last_m5["ema12"] >= last_m5["ema26"]:
        return None

    score = 0.25
    sweep_ok, _ = liquidity_sweep(df_m15, direction)
    if not sweep_ok:
        return None
    score += 0.25

    area_ok, _ = order_block_or_fvg(df_m15, direction)
    if area_ok:
        score += 0.15

    zone = premium_discount_zone(df_m15)
    if (bias_bull and zone == "discount") or (not bias_bull and zone == "premium"):
        score += 0.1

    struct_m5 = market_structure(df_m5, 2)
    if (bias_bull and struct_m5 == "bullish") or (not bias_bull and struct_m5 == "bearish"):
        score += 0.1

    if score * 100 < MIN_CONFIDENCE:
        return None

    last = df_m15.iloc[-2]
    price = last["close"]
    return {
        "signal": "BUY" if bias_bull else "SELL",
        "entry_signal": round(price, 6),
        "confidence": int(score * 100),
    }

def get_coins_by_volume(top=50, max_price=50.0):
    for _ in range(3):
        try:
            resp = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10)
            tickers = [t for t in resp.json() if t["symbol"].endswith("USDT")]
            tickers.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
            res = []
            for t in tickers:
                if float(t["lastPrice"]) <= max_price:
                    res.append(t["symbol"])
                if len(res) >= top:
                    break
            return res
        except:
            time.sleep(10)
    return []

# ---------- PRESISI HARGA & QUANTITY ----------
symbol_filters_cache = {}

def get_symbol_filters(symbol):
    if symbol in symbol_filters_cache:
        return symbol_filters_cache[symbol]
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        for s in resp.json()["symbols"]:
            if s["symbol"] == symbol:
                filters = {}
                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        filters["tickSize"] = float(f["tickSize"])
                    elif f["filterType"] == "LOT_SIZE":
                        filters["stepSize"] = float(f["stepSize"])
                        filters["minQty"] = float(f["minQty"])
                    elif f["filterType"] == "MIN_NOTIONAL":
                        filters["minNotional"] = float(f["notional"])
                symbol_filters_cache[symbol] = filters
                return filters
    except:
        pass
    return None

def round_to_tick(value, tick_size):
    return round(value / tick_size) * tick_size

def round_to_step(value, step_size):
    return round(value / step_size) * step_size

# ---------- ALGO ORDER FUNCTIONS ----------
def get_mark_price(symbol):
    try:
        resp = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}", timeout=5)
        return float(resp.json()["price"])
    except:
        return None

def set_leverage(symbol, leverage):
    params = {"symbol": symbol, "leverage": leverage}
    signed_request("/fapi/v1/leverage", params, method="POST")

def place_algo_stop_order(symbol, side, quantity, trigger_price, limit_price):
    params = {
        "algoType": "CONDITIONAL",
        "symbol": symbol,
        "side": side,
        "type": "STOP",
        "quantity": quantity,
        "triggerPrice": trigger_price,
        "price": limit_price,
        "timeInForce": "GTC",
        "workingType": "MARK_PRICE"
    }
    res = signed_request("/fapi/v1/algoOrder", params, method="POST")
    if res and "algoId" in res:
        return res["algoId"]
    return None

def cancel_algo_order(symbol, algo_id):
    params = {"symbol": symbol, "algoId": algo_id}
    return signed_request("/fapi/v1/algoOrder", params, method="DELETE") is not None

def get_algo_order_status(symbol, algo_id):
    params = {"symbol": symbol, "algoId": algo_id}
    res = signed_request("/fapi/v1/algoOrder", params, method="GET")
    if res and "algoStatus" in res:
        return res["algoStatus"]
    return None

# ---------- SIKLUS TRADING (LOGIKA PERSILANGAN + PRESISI) ----------
def trading_cycle():
    coins = get_coins_by_volume(50, MAX_PRICE_USDT)
    if not coins:
        send_telegram("❌ Gagal ambil daftar koin.")
        return

    signals = []
    for sym in coins:
        try:
            h1 = fetch_klines(sym, "1h", 100)
            m15 = fetch_klines(sym, "15m", 100)
            m5 = fetch_klines(sym, "5m", 100)
            if not all([h1 is not None, m15 is not None, m5 is not None]):
                continue
            sig = generate_signal(h1, m15, m5)
            if sig:
                sig["symbol"] = sym
                signals.append(sig)
        except:
            pass
        time.sleep(0.03)

    if not signals:
        send_telegram("❌ Tidak ada sinyal.")
        return

    best = max(signals, key=lambda x: x["confidence"])
    symbol = best["symbol"]
    side = "BUY" if best["signal"] == "BUY" else "SELL"
    entry_raw = best["entry_signal"]
    confidence = best["confidence"]

    # Ambil filter presisi
    filters = get_symbol_filters(symbol)
    if not filters:
        send_telegram("❌ Gagal ambil filter presisi.")
        return
    tick = filters["tickSize"]
    step = filters["stepSize"]
    min_qty = filters["minQty"]
    min_notional = filters["minNotional"]

    # Bulatkan entry
    entry = round_to_tick(entry_raw, tick)

    # Hitung TP & SL dari entry yang sudah dibulatkan
    if side == "BUY":
        tp_raw = entry * (1 + TP_PERCENT/100)
        sl_raw = entry * (1 - SL_PERCENT/100)
    else:
        tp_raw = entry * (1 - TP_PERCENT/100)
        sl_raw = entry * (1 + SL_PERCENT/100)

    tp = round_to_tick(tp_raw, tick)
    sl = round_to_tick(sl_raw, tick)

    send_telegram(f"📊 {best['signal']} {symbol} | Entry: {entry} | TP: {tp} | SL: {sl} | Conf: {confidence}%")

    # --- TUNGGU HARGA VALID ---
    send_telegram(f"⏳ Menunggu posisi valid untuk {side} {symbol}...")
    start_time = time.time()
    last_notify = time.time()

    while time.time() - start_time < TIMEOUT_MINUTES * 60:
        mark = get_mark_price(symbol)
        if mark is None:
            time.sleep(1)
            continue

        if side == "BUY":
            if mark >= tp:
                send_telegram(f"⚠️ TP ({tp}) tersentuh sebelum entry.")
                return
            if mark <= sl:
                send_telegram(f"⚠️ SL ({sl}) tersentuh sebelum entry.")
                return
            if mark < entry:   # harga di bawah entry → valid untuk BUY STOP
                send_telegram(f"✅ Harga di bawah entry ({mark:.6f}). Memasang STOP BUY...")
                break
        else:
            if mark <= tp:
                send_telegram(f"⚠️ TP ({tp}) tersentuh sebelum entry.")
                return
            if mark >= sl:
                send_telegram(f"⚠️ SL ({sl}) tersentuh sebelum entry.")
                return
            if mark > entry:   # harga di atas entry → valid untuk SELL STOP
                send_telegram(f"✅ Harga di atas entry ({mark:.6f}). Memasang STOP SELL...")
                break

        if time.time() - last_notify >= 180:
            send_telegram(f"⏳ {symbol} | Current: {mark:.6f} | Entry: {entry}")
            last_notify = time.time()

        time.sleep(1)

    if time.time() - start_time >= TIMEOUT_MINUTES * 60:
        send_telegram(f"⏰ Timeout menunggu posisi valid.")
        return

    # --- HITUNG QUANTITY DENGAN PRESISI ---
    raw_qty = max(min_notional / entry, min_qty)
    qty = round_to_step(raw_qty, step)
    if qty < min_qty:
        qty = min_qty
        qty = round_to_step(qty, step)  # pastikan kelipatan
    # Jika masih di bawah min_qty setelah round, naikkan satu step
    if qty < min_qty:
        qty += step
        qty = round_to_step(qty, step)

    # Set leverage
    set_leverage(symbol, LEVERAGE)

    # Pasang order
    algo_id = place_algo_stop_order(symbol, side, qty, entry, entry)
    if not algo_id:
        send_telegram("❌ Gagal pasang Order Bersyarat.")
        return

    send_telegram(f"🎯 Order Bersyarat terpasang (ID: {algo_id})")

    # --- MONITORING ---
    last_notify = time.time()
    while True:
        time.sleep(1)
        mark = get_mark_price(symbol)
        if mark is None:
            continue

        # Cek TP/SL setelah order
        if side == "BUY":
            if mark >= tp:
                cancel_algo_order(symbol, algo_id)
                send_telegram(f"🛑 TP ({tp}) tercapai, order dibatalkan.")
                return
            if mark <= sl:
                cancel_algo_order(symbol, algo_id)
                send_telegram(f"🛑 SL ({sl}) tercapai, order dibatalkan.")
                return
        else:
            if mark <= tp:
                cancel_algo_order(symbol, algo_id)
                send_telegram(f"🛑 TP ({tp}) tercapai, order dibatalkan.")
                return
            if mark >= sl:
                cancel_algo_order(symbol, algo_id)
                send_telegram(f"🛑 SL ({sl}) tercapai, order dibatalkan.")
                return

        # Cek status order
        status = get_algo_order_status(symbol, algo_id)
        if status == "FILLED":
            send_telegram(f"✅ Order {symbol} terisi di {mark:.6f}")
            return
        elif status in ("CANCELED", "REJECTED", "EXPIRED"):
            send_telegram(f"⚠️ Order {symbol} berstatus {status}.")
            return

        if time.time() - last_notify >= 180:
            send_telegram(f"⏳ {symbol} | Mark: {mark:.6f} | TP: {tp} | SL: {sl}")
            last_notify = time.time()

        if time.time() - start_time > TIMEOUT_MINUTES * 60:
            cancel_algo_order(symbol, algo_id)
            send_telegram(f"⏰ Timeout, order dibatalkan.")
            return

# ---------- LOOP UTAMA ----------
def run_loop():
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Siklus trading...")
            trading_cycle()
        except Exception as e:
            send_telegram(f"⚠️ Bot error: {e}")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    ip = get_public_ip()
    send_telegram(f"🚀 Bot Auto-Trading dimulai!\nIP Publik Render: {ip}\nPastikan IP sudah di-whitelist di Binance.")

    if not API_KEY or not SECRET_KEY:
        send_telegram("❌ API Key/Secret belum diset di environment Render.")
    else:
        t = threading.Thread(target=run_loop, daemon=True)
        t.start()

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
