#!/usr/bin/env python3
"""
SIGNAL BROADCASTER OPTIMIZED – Dual API (Binance/Bybit)
Dengan adjustable aggression level via Telegram:
  /up   -> perketat sinyal (confidence naik, RR naik, volume naik)
  /down -> longgarkan sinyal (confidence turun, RR turun, volume turun)
  /status -> tampilkan level saat ini
"""

import time
import random
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask

# ================== KONFIGURASI ==================
TELEGRAM_TOKEN = "8094484109:AAF9Z3lQUxdQFqqeG6NKV9O1EC0vrxzJy0U"
CHAT_ID = "8041197505"
MAX_PRICE = 100.0
BAN_CYCLES = 20
SCAN_INTERVAL = 60
TOP_COINS = 50
# =================================================

# ---------- ADJUSTABLE AGGRESSION ----------
# Level: 0 = normal, positif = lebih ketat, negatif = lebih longgar
AGGRESSION_LEVEL = 0
LEVEL_LOCK = threading.Lock()

def get_aggression_params():
    """Kembalikan parameter berdasarkan level agresi saat ini."""
    with LEVEL_LOCK:
        lvl = AGGRESSION_LEVEL

    # Base normal
    base = {
        "min_confidence": 70,
        "min_rr": 1.8,
        "volume_mult": 1.5,
        "tp_distance_atr": 2.0
    }

    if lvl > 0:  # lebih ketat
        base["min_confidence"] += 5 * lvl
        base["min_rr"] += 0.2 * lvl
        base["volume_mult"] += 0.5 * lvl
        base["tp_distance_atr"] += 0.5 * lvl
    elif lvl < 0:  # lebih longgar
        base["min_confidence"] += 5 * lvl  # lvl negatif → confidence turun
        base["min_rr"] += 0.2 * lvl
        base["volume_mult"] += 0.3 * lvl
        base["tp_distance_atr"] += 0.5 * lvl

    # Batas aman
    base["min_confidence"] = max(55, min(85, base["min_confidence"]))
    base["min_rr"] = max(1.2, min(2.5, base["min_rr"]))
    base["volume_mult"] = max(0.8, min(3.0, base["volume_mult"]))
    base["tp_distance_atr"] = max(1.0, min(4.0, base["tp_distance_atr"]))

    return base

def set_aggression(direction):
    """Ubah level agresi. direction: 'up' atau 'down'"""
    global AGGRESSION_LEVEL
    with LEVEL_LOCK:
        old = AGGRESSION_LEVEL
        if direction == "up":
            AGGRESSION_LEVEL = min(2, AGGRESSION_LEVEL + 1)  # max +2
        else:
            AGGRESSION_LEVEL = max(-3, AGGRESSION_LEVEL - 1)  # max -3
        new = AGGRESSION_LEVEL

    # Kirim notifikasi
    if old != new:
        params = get_aggression_params()
        msg = (
            f"🔧 <b>Level Agresi Berubah</b>\n"
            f"Dari level {old} → {new}\n"
            f"Confidence min: {params['min_confidence']}%\n"
            f"RR min: 1:{params['min_rr']}\n"
            f"Volume min: {params['volume_mult']}x\n"
            f"Jarak TP min: {params['tp_distance_atr']}x ATR"
        )
        send_telegram(msg)
    else:
        send_telegram(f"ℹ️ Level sudah di batas {'atas' if direction=='up' else 'bawah'} (level {new}).")

# ---------- TELEGRAM POLLING ----------
def telegram_polling():
    """Thread terpisah: poll update dari Telegram untuk perintah /up /down /status"""
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            params = {"timeout": 30, "offset": offset}
            resp = requests.get(url, params=params, timeout=35)
            data = resp.json()
            if not data.get("ok"):
                time.sleep(5)
                continue

            for update in data["result"]:
                offset = update["update_id"] + 1
                msg = update.get("message")
                if not msg:
                    continue
                text = msg.get("text", "")
                chat_id = str(msg["chat"]["id"])

                # Hanya proses perintah dari CHAT_ID yang terdaftar
                if chat_id != CHAT_ID:
                    continue

                if text == "/up":
                    set_aggression("up")
                elif text == "/down":
                    set_aggression("down")
                elif text == "/status":
                    params = get_aggression_params()
                    send_telegram(
                        f"📊 <b>Level Saat Ini: {AGGRESSION_LEVEL}</b>\n"
                        f"Confidence: {params['min_confidence']}%\n"
                        f"RR: 1:{params['min_rr']}\n"
                        f"Volume: {params['volume_mult']}x\n"
                        f"Jarak TP: {params['tp_distance_atr']}x ATR"
                    )
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(5)

# ---------- TELEGRAM SEND ----------
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except:
        pass

# ========== BINANCE FUNCTIONS ==========
def get_coins_binance(limit=50, max_price=100.0):
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        data = requests.get(url, timeout=15).json()
        if not isinstance(data, list):
            return None
        tickers = [t for t in data if t["symbol"].endswith("USDT")]
        tickers.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        res = []
        for t in tickers:
            price = float(t["lastPrice"])
            if price <= max_price:
                res.append(t["symbol"])
            if len(res) >= limit:
                break
        return res if res else None
    except:
        return None

def fetch_klines_binance(symbol, interval, limit=200):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if isinstance(data, dict) and "code" in data:
            return None
        df = pd.DataFrame(data, columns=[
            "timestamp","open","high","low","close","volume",
            "close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]]
    except:
        return None

# ========== BYBIT FUNCTIONS ==========
INTERVAL_MAP = {"1d":"D","4h":"240","1h":"60","15m":"15","5m":"5"}

def get_coins_bybit(limit=50, max_price=100.0):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        data = requests.get(url, timeout=15).json()
        if data.get("retCode") != 0:
            return None
        tickers = data["result"]["list"]
        filtered = []
        for t in tickers:
            if t["symbol"].endswith("USDT"):
                try:
                    price = float(t["lastPrice"])
                    if price <= max_price:
                        filtered.append(t)
                except:
                    pass
        filtered.sort(key=lambda x: float(x.get("turnover24h", 0)), reverse=True)
        return [t["symbol"] for t in filtered[:limit]]
    except:
        return None

def fetch_klines_bybit(symbol, interval, limit=200):
    bybit_interval = INTERVAL_MAP.get(interval, "60")
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={bybit_interval}&limit={limit}"
    try:
        data = requests.get(url, timeout=15).json()
        if data.get("retCode") != 0:
            return None
        rows = data["result"]["list"]
        if not rows:
            return None
        rows = rows[::-1]
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]]
    except:
        return None

FALLBACK_SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
    "AVAXUSDT","DOTUSDT","LINKUSDT","MATICUSDT","UNIUSDT","ATOMUSDT","LTCUSDT",
    "ETCUSDT","OPUSDT","ARBUSDT","INJUSDT","TIAUSDT","SUIUSDT","SEIUSDT",
    "NEARUSDT","APTUSDT","RNDRUSDT","FETUSDT","AGIXUSDT","OCEANUSDT","GRTUSDT",
    "THETAUSDT","SANDUSDT","MANAUSDT","GALAUSDT","AXSUSDT","CHZUSDT","FLOWUSDT",
    "EGLDUSDT","QNTUSDT","SNXUSDT","CRVUSDT","COMPUSDT","AAVEUSDT","MKRUSDT",
    "RUNEUSDT","LDOUSDT","FXSUSDT","1INCHUSDT","ZRXUSDT","BATUSDT","ENJUSDT","ANKRUSDT"
]

# ========== INDIKATOR & SMC TOOLS ==========
def add_all_indicators(df):
    if len(df) < 80:
        return None
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean() if len(df) >= 200 else df["ema50"]
    df["tr"] = np.maximum(df["high"] - df["low"],
                          np.maximum(abs(df["high"] - df["close"].shift()),
                                     abs(df["low"] - df["close"].shift())))
    df["atr"] = df["tr"].rolling(14).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    return df

def find_swings(df, window=2):
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False
    for i in range(window, len(df) - window):
        if df["high"].iloc[i] == df["high"].iloc[i-window:i+window+1].max():
            df.loc[df.index[i], "swing_high"] = True
        if df["low"].iloc[i] == df["low"].iloc[i-window:i+window+1].min():
            df.loc[df.index[i], "swing_low"] = True
    return df

def market_structure(df, window=3):
    if len(df) < window*2+2:
        return "ranging"
    df_swing = find_swings(df, window)
    swing_highs = df_swing[df_swing["swing_high"]]["high"]
    swing_lows = df_swing[df_swing["swing_low"]]["low"]
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "ranging"
    hh = swing_highs.iloc[-1] > swing_highs.iloc[-2]
    hl = swing_lows.iloc[-1] > swing_lows.iloc[-2]
    lh = swing_highs.iloc[-1] < swing_highs.iloc[-2]
    ll = swing_lows.iloc[-1] < swing_lows.iloc[-2]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

def detect_liquidity_sweep(df, direction):
    last = df.iloc[-2]
    if direction == "buy":
        for i in range(len(df)-5, 3, -1):
            if df["low"].iloc[i] == df["low"].iloc[i-3:i+4].min():
                if last["low"] < df["low"].iloc[i] and last["close"] > df["low"].iloc[i]:
                    return True, df["low"].iloc[i]
    else:
        for i in range(len(df)-5, 3, -1):
            if df["high"].iloc[i] == df["high"].iloc[i-3:i+4].max():
                if last["high"] > df["high"].iloc[i] and last["close"] < df["high"].iloc[i]:
                    return True, df["high"].iloc[i]
    return False, None

def find_order_block(df, direction):
    last_idx = len(df) - 2
    last_close = df["close"].iloc[last_idx]
    if direction == "buy":
        for i in range(last_idx-1, max(last_idx-20, 0), -1):
            if df["close"].iloc[i] < df["open"].iloc[i]:
                if i+1 <= last_idx and df["close"].iloc[i+1] > df["open"].iloc[i+1]:
                    if last_close > df["high"].iloc[i]:
                        return df["high"].iloc[i], df["low"].iloc[i]
        if last_idx >= 2 and df["low"].iloc[last_idx] > df["high"].iloc[last_idx-2]:
            return df["high"].iloc[last_idx-2], df["low"].iloc[last_idx]
    else:
        for i in range(last_idx-1, max(last_idx-20, 0), -1):
            if df["close"].iloc[i] > df["open"].iloc[i]:
                if i+1 <= last_idx and df["close"].iloc[i+1] < df["open"].iloc[i+1]:
                    if last_close < df["low"].iloc[i]:
                        return df["high"].iloc[i], df["low"].iloc[i]
        if last_idx >= 2 and df["high"].iloc[last_idx] < df["low"].iloc[last_idx-2]:
            return df["high"].iloc[last_idx], df["low"].iloc[last_idx-2]
    return None, None

def get_nearest_levels(df):
    df_swing = find_swings(df, window=2)
    swings_low = df_swing[df_swing["swing_low"]]["low"]
    swings_high = df_swing[df_swing["swing_high"]]["high"]
    current = df["close"].iloc[-2]
    supports = sorted([v for v in swings_low if v < current], reverse=True)
    resistances = sorted([v for v in swings_high if v > current])
    return supports, resistances

# ========== ANALISA SINYAL (ADJUSTABLE) ==========
def analyze_signal(symbol, fetch_klines_func):
    params = get_aggression_params()
    min_conf = params["min_confidence"]
    min_rr = params["min_rr"]
    vol_mult = params["volume_mult"]
    tp_dist = params["tp_distance_atr"]

    df_d1 = fetch_klines_func(symbol, "1d", 200)
    df_h4 = fetch_klines_func(symbol, "4h", 200)
    df_h1 = fetch_klines_func(symbol, "1h", 150)
    df_m15 = fetch_klines_func(symbol, "15m", 150)
    df_m5 = fetch_klines_func(symbol, "5m", 150)

    if any([df_d1 is None, df_h4 is None, df_h1 is None, df_m15 is None, df_m5 is None]):
        return None

    df_d1 = add_all_indicators(df_d1)
    df_h4 = add_all_indicators(df_h4)
    df_h1 = add_all_indicators(df_h1)
    df_m15 = add_all_indicators(df_m15)
    df_m5 = add_all_indicators(df_m5)

    if any([df_d1 is None, df_h4 is None, df_h1 is None, df_m15 is None, df_m5 is None]):
        return None

    # D1
    struct_d1 = market_structure(df_d1, window=5)
    if struct_d1 == "ranging":
        return None
    bias_bull = struct_d1 == "bullish"
    direction = "BUY" if bias_bull else "SELL"
    score = 0

    last_d1 = df_d1.iloc[-1]
    if bias_bull and last_d1["close"] > last_d1["ema50"]:
        score += 0.15
    elif not bias_bull and last_d1["close"] < last_d1["ema50"]:
        score += 0.15

    if bias_bull and 40 < last_d1["rsi"] < 70:
        score += 0.05
    elif not bias_bull and 30 < last_d1["rsi"] < 60:
        score += 0.05

    # H4
    struct_h4 = market_structure(df_h4, window=3)
    if struct_h4 != struct_d1:
        return None
    score += 0.10

    sweep_h4, _ = detect_liquidity_sweep(df_h4, "buy" if bias_bull else "sell")
    if sweep_h4:
        score += 0.10

    ob_h4_high, _ = find_order_block(df_h4, "buy" if bias_bull else "sell")
    if ob_h4_high is not None:
        score += 0.10

    # H1
    struct_h1 = market_structure(df_h1, window=2)
    if struct_h1 != struct_d1:
        return None
    score += 0.10

    last_h1 = df_h1.iloc[-2]
    if last_h1["volume"] <= vol_mult * last_h1["vol_avg20"]:
        return None
    score += 0.05

    if bias_bull and last_h1["rsi"] >= 60:
        return None
    if not bias_bull and last_h1["rsi"] <= 40:
        return None

    # M15
    last_m15 = df_m15.iloc[-2]
    if bias_bull and last_m15["ema12"] > last_m15["ema26"]:
        score += 0.10
    elif not bias_bull and last_m15["ema12"] < last_m15["ema26"]:
        score += 0.10
    else:
        return None

    sweep_m15, _ = detect_liquidity_sweep(df_m15, "buy" if bias_bull else "sell")
    if not sweep_m15:
        return None
    score += 0.15

    if bias_bull and last_m15["rsi"] < 65:
        score += 0.05
    elif not bias_bull and last_m15["rsi"] > 35:
        score += 0.05

    # M5
    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["close"] > last_m5["ema12"]:
        score += 0.05
    elif not bias_bull and last_m5["close"] < last_m5["ema12"]:
        score += 0.05
    else:
        return None

    # TP/SL dengan parameter adjustable
    supports_h1, resistances_h1 = get_nearest_levels(df_h1)
    supports_m15, resistances_m15 = get_nearest_levels(df_m15)

    entry = round(last_m15["close"], 6)
    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else entry * 0.002

    if direction == "BUY":
        if supports_h1:
            sl_base = supports_h1[0]
        elif supports_m15:
            sl_base = supports_m15[0]
        else:
            sl_base = entry - atr * 1.5

        sl = round(sl_base - atr * 0.5, 6)

        if resistances_h1:
            tp_base = resistances_h1[0]
        elif resistances_m15:
            tp_base = resistances_m15[0]
        else:
            tp_base = entry + atr * 2.0

        tp = round(tp_base * 0.998, 6)

        if tp - entry < tp_dist * atr:
            return None

        risk = entry - sl
        reward = tp - entry
        if reward / risk < min_rr:
            tp = round(entry + risk * min_rr, 6)
    else:
        if resistances_h1:
            sl_base = resistances_h1[0]
        elif resistances_m15:
            sl_base = resistances_m15[0]
        else:
            sl_base = entry + atr * 1.5

        sl = round(sl_base + atr * 0.5, 6)

        if supports_h1:
            tp_base = supports_h1[0]
        elif supports_m15:
            tp_base = supports_m15[0]
        else:
            tp_base = entry - atr * 2.0

        tp = round(tp_base * 1.002, 6)

        if entry - tp < tp_dist * atr:
            return None

        risk = sl - entry
        reward = entry - tp
        if reward / risk < min_rr:
            tp = round(entry - risk * min_rr, 6)

    confidence = min(int(score * 100), 95)
    if confidence < min_conf:
        return None

    return {
        "symbol": symbol,
        "signal": direction,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "confidence": confidence,
        "atr": round(atr, 6),
        "rr": round(abs(tp - entry) / abs(entry - sl), 2) if abs(entry - sl) > 0 else 0
    }

# ========== LOOP UTAMA ==========
banned = {}

def main_loop():
    global banned
    while True:
        to_del = [k for k, v in banned.items() if v <= 0]
        for k in to_del:
            del banned[k]
        for k in list(banned.keys()):
            banned[k] -= 1

        coins = None
        fetch_func = None
        api_source = ""

        coins = get_coins_binance(TOP_COINS, MAX_PRICE)
        if coins:
            fetch_func = fetch_klines_binance
            api_source = "Binance"
        else:
            coins = get_coins_bybit(TOP_COINS, MAX_PRICE)
            if coins:
                fetch_func = fetch_klines_bybit
                api_source = "Bybit"
            else:
                coins = FALLBACK_SYMBOLS[:TOP_COINS]
                test_df = fetch_klines_binance(coins[0], "1h", 10)
                if test_df is not None:
                    fetch_func = fetch_klines_binance
                    api_source = "Fallback + Binance"
                else:
                    fetch_func = fetch_klines_bybit
                    api_source = "Fallback + Bybit"

        if not coins or not fetch_func:
            send_telegram("❌ Semua API tidak tersedia. Coba lagi nanti.")
            time.sleep(SCAN_INTERVAL)
            continue

        params = get_aggression_params()
        print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] Scan {len(coins)} koin ({api_source}) | Level {AGGRESSION_LEVEL}")
        signals = []
        for sym in coins:
            if sym in banned:
                continue
            try:
                sig = analyze_signal(sym, fetch_func)
                if sig:
                    signals.append(sig)
            except:
                pass
            time.sleep(0.02)

        if not signals:
            send_telegram(f"❌ Tidak ada sinyal (Conf ≥ {params['min_confidence']}%, RR ≥ 1:{params['min_rr']})")
            time.sleep(SCAN_INTERVAL)
            continue

        send_telegram(f"🔔 <b>Ditemukan {len(signals)} sinyal ({api_source}):</b>")
        for sig in signals:
            msg = (
                f"<b>📊 {sig['signal']} {sig['symbol']}</b>\n"
                f"Entry: {sig['entry']}\n"
                f"TP: {sig['tp']} | SL: {sig['sl']}\n"
                f"Confidence: {sig['confidence']}%\n"
                f"RR: 1:{sig['rr']} | ATR: {sig['atr']}"
            )
            send_telegram(msg)
            banned[sig["symbol"]] = BAN_CYCLES

        send_telegram(f"📛 {len(signals)} koin di-ban {BAN_CYCLES} siklus.")
        time.sleep(SCAN_INTERVAL)

# ========== FLASK APP ==========
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive", 200

def run_flask():
    app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    print("=" * 60)
    print("  SIGNAL BROADCASTER OPTIMIZED – ADJUSTABLE")
    print("  /up (ketat) | /down (longgar) | /status")
    print("=" * 60)
    send_telegram("🚀 <b>Signal Broadcaster Adjustable dimulai!</b>\nKetik /up, /down, atau /status di sini.")

    # Thread Flask
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Thread polling Telegram
    polling_thread = threading.Thread(target=telegram_polling, daemon=True)
    polling_thread.start()

    # Loop utama
    main_loop()
