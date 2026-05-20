#!/usr/bin/env python3
"""
SIGNAL BROADCASTER – Render Web Service
Loop nonstop + Flask untuk health-check.
"""

import os
import time
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
SL_PERCENT = 0.85
MIN_CONFIDENCE = 65
# =================================================

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive", 200

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=8)
    except:
        pass

# -------------------------------------------------------------------
# Data & Indikator (sama persis seperti sebelumnya, tidak diubah)
# -------------------------------------------------------------------
def fetch_klines(symbol, interval, limit=100):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    for attempt in range(3):
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
    for attempt in range(3):
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

def main_scan():
    coins = get_coins_by_volume(50, 50.0)
    if not coins:
        send_telegram("❌ Gagal mengambil daftar koin setelah 3 percobaan.")
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
    else:
        send_telegram(f"🔔 Ditemukan {len(signals)} sinyal (Conf ≥ 65%):")
        for sig in signals:
            tp_val = round(sig["entry_signal"] * (1 + TP_PERCENT/100), 6) if sig["signal"]=="BUY" else round(sig["entry_signal"] * (1 - TP_PERCENT/100), 6)
            sl_val = round(sig["entry_signal"] * (1 - SL_PERCENT/100), 6) if sig["signal"]=="BUY" else round(sig["entry_signal"] * (1 + SL_PERCENT/100), 6)
            msg = (
                f"<b>📊 {sig['signal']} {sig['symbol']}</b>\n"
                f"Entry: {sig['entry_signal']}\n"
                f"TP: {tp_val} (+{TP_PERCENT}%) | SL: {sl_val} (∓{SL_PERCENT}%)\n"
                f"Confidence: {sig['confidence']}%"
            )
            send_telegram(msg)

def run_loop():
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Memulai scan...")
            main_scan()
        except Exception as e:
            print(f"Error: {e}")
            send_telegram(f"⚠️ Bot error: {e}")
        time.sleep(60)

if __name__ == "__main__":
    # Jalankan loop di thread terpisah
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    # Jalankan Flask
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
