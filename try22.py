#!/usr/bin/env python3
"""
AUTO TRADING BOT – Render Web Service
Menggunakan API Binance Futures (Stop-Limit Order)
IP dinamis Render → kirim log error ke Telegram jika API menolak.
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

# Parameter trading
TP_PERCENT = 0.6      # 0.6%
SL_PERCENT = 0.85     # 0.85%
MIN_CONFIDENCE = 65
LEVERAGE = 5
TIMEOUT_MINUTES = 15
SCAN_INTERVAL = 60    # jeda antar siklus (detik)
MAX_PRICE_USDT = 50.0  # hanya koin ≤ $50

# API Binance (diisi dari environment variable Render)
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

# ---------- BINANCE REST SIGNATURE ----------
def signed_request(endpoint, params, method="GET"):
    """Tambah signature & kirim request ke fapi.binance.com. Kirim detail error ke Telegram jika gagal."""
    if not API_KEY or not SECRET_KEY:
        send_telegram("❌ API Key/Secret belum diisi di environment variable Render.")
        return None

    params["timestamp"] = int(time.time() * 1000)
    # Buat query string tanpa signature dulu
    query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    # Tanda tangan
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

        # Jika sukses, kembalikan JSON
        if resp.status_code == 200:
            return resp.json()
        else:
            # Gagal → kirim detail ke Telegram
            try:
                err_json = resp.json()
                err_msg = err_json.get("msg", "Tidak ada pesan error")
            except:
                err_msg = resp.text
            send_telegram(f"❌ API Error {resp.status_code} pada {endpoint}\n{err_msg}")
            return None
    except Exception as e:
        send_telegram(f"⚠️ Network/Exception: {e}")
        return None

# ---------- DATA FETCHER (seperti sebelumnya) ----------
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

# ---------- TRADING FUNCTIONS ----------
def get_mark_price(symbol):
    try:
        resp = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}", timeout=5)
        return float(resp.json()["price"])
    except:
        return None

def get_min_quantity(symbol):
    """Ambil quantity minimum dari exchange info."""
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        data = resp.json()
        for s in data["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "MIN_NOTIONAL":
                        min_notional = float(f["notional"])
                        return min_notional
        return 5.0  # default
    except:
        return 5.0

def set_leverage(symbol, leverage):
    params = {"symbol": symbol, "leverage": leverage}
    res = signed_request("/fapi/v1/leverage", params, method="POST")
    if res is None:
        send_telegram(f"⚠️ Gagal set leverage {symbol}. API mungkin menolak (cek IP).")

def place_stop_limit_order(symbol, side, quantity, entry_price):
    """Pasang stop-limit order di entry_price."""
    params = {
        "symbol": symbol,
        "side": side,
        "type": "STOP",
        "quantity": quantity,
        "price": entry_price,
        "stopPrice": entry_price,
        "timeInForce": "GTC"
    }
    res = signed_request("/fapi/v1/order", params, method="POST")
    if res and "orderId" in res:
        return res["orderId"]
    else:
        if res and "msg" in res:
            send_telegram(f"❌ Order gagal: {res['msg']} (mungkin IP ditolak)")
        else:
            send_telegram("❌ Order gagal tanpa pesan. Cek API key/IP.")
        return None

def cancel_order(symbol, order_id):
    params = {"symbol": symbol, "orderId": order_id}
    res = signed_request("/fapi/v1/order", params, method="DELETE")
    return res is not None

def check_order_status(symbol, order_id):
    """Return status: NEW, FILLED, CANCELED, EXPIRED, atau None jika error."""
    params = {"symbol": symbol, "orderId": order_id}
    res = signed_request("/fapi/v1/order", params, method="GET")
    if res and "status" in res:
        return res["status"]
    return None

# ---------- MAIN TRADING CYCLE ----------
def trading_cycle():
    """Satu siklus: scan sinyal terbaik → eksekusi → monitoring."""
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
        send_telegram("❌ Tidak ada sinyal dengan Confidence ≥ 65%")
        return

    # Pilih sinyal confidence tertinggi
    best = max(signals, key=lambda x: x["confidence"])
    symbol = best["symbol"]
    side = "BUY" if best["signal"] == "BUY" else "SELL"
    entry = best["entry_signal"]
    confidence = best["confidence"]

    # Hitung TP & SL
    if side == "BUY":
        tp = round(entry * (1 + TP_PERCENT/100), 6)
        sl = round(entry * (1 - SL_PERCENT/100), 6)
    else:
        tp = round(entry * (1 - TP_PERCENT/100), 6)
        sl = round(entry * (1 + SL_PERCENT/100), 6)

    # Ambil mark price terbaru untuk validasi stopPrice
    mark = get_mark_price(symbol)
    if mark is None:
        send_telegram("❌ Gagal ambil harga pasar.")
        return

    # Validasi stopPrice (harus di atas mark untuk BUY, di bawah mark untuk SELL)
    if side == "BUY" and entry <= mark:
        send_telegram(f"⚠️ Entry {entry} ≤ mark {mark}, stopPrice tidak valid. Skip.")
        return
    if side == "SELL" and entry >= mark:
        send_telegram(f"⚠️ Entry {entry} ≥ mark {mark}, stopPrice tidak valid. Skip.")
        return

    # Hitung quantity minimum
    min_notional = get_min_quantity(symbol)
    quantity = max(min_notional / entry, 0.001)  # minimal 0.001 untuk keamanan
    # Ambil step size dari exchange info? Kita sederhanakan: bulatkan ke 1 desimal, atau gunakan quantity mentah.
    # Untuk kebanyakan pair, quantity bisa 1 desimal. Untuk aman, kita bulatkan ke 3 desimal.
    quantity = round(quantity, 3)

    # Set leverage
    set_leverage(symbol, LEVERAGE)

    # Pasang order
    send_telegram(f"📊 {best['signal']} {symbol}\nEntry: {entry}\nTP: {tp} | SL: {sl}\nConf: {confidence}%\nQty: {quantity}")
    order_id = place_stop_limit_order(symbol, side, quantity, entry)
    if not order_id:
        send_telegram(f"❌ Gagal pasang order {symbol}.")
        return

    send_telegram(f"✅ Stop-Limit Order terpasang (ID: {order_id})")

    # Monitoring loop
    start_time = time.time()
    last_notify = time.time()
    while True:
        time.sleep(1)
        mark = get_mark_price(symbol)
        if mark is None:
            continue

        # Cek TP / SL sebelum entry
        if side == "BUY":
            if mark >= tp:
                cancel_order(symbol, order_id)
                send_telegram(f"⚠️ {symbol} TP ({tp}) tersentuh sebelum entry. Order dibatalkan.")
                return
            if mark <= sl:
                cancel_order(symbol, order_id)
                send_telegram(f"⚠️ {symbol} SL ({sl}) tersentuh sebelum entry. Order dibatalkan.")
                return
        else:
            if mark <= tp:
                cancel_order(symbol, order_id)
                send_telegram(f"⚠️ {symbol} TP ({tp}) tersentuh sebelum entry. Order dibatalkan.")
                return
            if mark >= sl:
                cancel_order(symbol, order_id)
                send_telegram(f"⚠️ {symbol} SL ({sl}) tersentuh sebelum entry. Order dibatalkan.")
                return

        # Cek status order
        status = check_order_status(symbol, order_id)
        if status == "FILLED":
            send_telegram(f"🏆 Order {symbol} terisi di {mark:.6f}. Manajemen risiko oleh platform.")
            return
        elif status in ("CANCELED", "EXPIRED"):
            send_telegram(f"⚠️ Order {symbol} {status} sebelum sempat terisi.")
            return

        # Notifikasi tiap 3 menit
        if time.time() - last_notify >= 180:
            send_telegram(f"⏳ {symbol} {side} LIMIT\nEntry: {entry}\nCurrent: {mark:.6f}\nTP: {tp} | SL: {sl}")
            last_notify = time.time()

        # Timeout
        if time.time() - start_time > TIMEOUT_MINUTES * 60:
            cancel_order(symbol, order_id)
            send_telegram(f"⏰ Timeout {TIMEOUT_MINUTES} menit, order {symbol} dibatalkan.")
            return

def run_loop():
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Siklus auto-trading...")
            trading_cycle()
        except Exception as e:
            send_telegram(f"⚠️ Bot error: {e}")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    if not API_KEY or not SECRET_KEY:
        send_telegram("❌ API Key/Secret belum diset di environment Render.")
    # Jalankan trading loop di thread terpisah
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
