#!/usr/bin/env python3
"""
SIGNAL BROADCASTER – Adjustable Aggression (Optimized)
Dual API (Binance / Bybit) + Telegram polling untuk /up, /down, /status.
Level bawah meningkatkan frekuensi drastis, level atas tetap ketat.
"""

import time
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
AGGRESSION_LEVEL = 0
LEVEL_LOCK = threading.Lock()

def get_aggression_params():
    with LEVEL_LOCK:
        lvl = AGGRESSION_LEVEL

    # Parameter dasar (level 0)
    p = {
        "min_confidence": 70,
        "min_rr": 1.8,
        "volume_mult": 1.5,        # 0 = nonaktif
        "tp_distance_atr": 2.0,    # 0 = nonaktif
        "rsi_h1_buy_max": 60,
        "rsi_h1_sell_min": 40,
        "rsi_m15_buy_max": 65,
        "rsi_m15_sell_min": 35,
        "require_h4_structure": True,
        "require_h1_structure": True,
        "sweep_mode": "m15",       # "m15" (wajib M15), "any" (H4 atau M15)
    }

    # Modifikasi berdasarkan level
    if lvl > 0:  # lebih ketat
        p["min_confidence"] += 5 * lvl
        p["min_rr"] += 0.2 * lvl
        p["volume_mult"] += 0.5 * lvl
        p["tp_distance_atr"] += 0.5 * lvl
        p["rsi_h1_buy_max"] -= 5 * lvl
        p["rsi_h1_sell_min"] += 5 * lvl
        p["rsi_m15_buy_max"] -= 5 * lvl
        p["rsi_m15_sell_min"] += 5 * lvl
    elif lvl < 0:  # lebih longgar
        p["min_confidence"] += 5 * lvl  # negatif → turun
        p["min_rr"] += 0.2 * lvl
        p["volume_mult"] += 0.3 * lvl
        p["tp_distance_atr"] += 0.5 * lvl
        p["rsi_h1_buy_max"] -= 5 * lvl   # lvl negatif → naikkan batas (longgar)
        p["rsi_h1_sell_min"] += 5 * lvl   # lvl negatif → turunkan batas
        p["rsi_m15_buy_max"] -= 5 * lvl
        p["rsi_m15_sell_min"] += 5 * lvl

        # Level -2 ke bawah: struktur H4/H1 tidak wajib
        if lvl <= -2:
            p["require_h4_structure"] = False
            p["require_h1_structure"] = False
            p["sweep_mode"] = "any"
        elif lvl == -1:
            p["require_h4_structure"] = True
            p["require_h1_structure"] = False  # hanya H1 yang dilonggarkan
            p["sweep_mode"] = "any"

    # Batas aman
    p["min_confidence"] = max(50, min(85, p["min_confidence"]))
    p["min_rr"] = max(1.0, min(2.5, p["min_rr"]))
    p["volume_mult"] = max(0.0, min(3.0, p["volume_mult"]))
    p["tp_distance_atr"] = max(0.0, min(4.0, p["tp_distance_atr"]))
    p["rsi_h1_buy_max"] = max(60, min(100, p["rsi_h1_buy_max"]))
    p["rsi_h1_sell_min"] = max(0, min(40, p["rsi_h1_sell_min"]))
    p["rsi_m15_buy_max"] = max(65, min(100, p["rsi_m15_buy_max"]))
    p["rsi_m15_sell_min"] = max(0, min(35, p["rsi_m15_sell_min"]))

    return p

def set_aggression(direction):
    global AGGRESSION_LEVEL
    with LEVEL_LOCK:
        old = AGGRESSION_LEVEL
        if direction == "up":
            AGGRESSION_LEVEL = min(2, AGGRESSION_LEVEL + 1)
        else:
            AGGRESSION_LEVEL = max(-3, AGGRESSION_LEVEL - 1)
        new = AGGRESSION_LEVEL

    if old != new:
        params = get_aggression_params()
        msg = (
            f"🔧 <b>Level Agresi Berubah</b>\n"
            f"Dari level {old} → {new}\n"
            f"Conf: {params['min_confidence']}% | RR: 1:{params['min_rr']} | Vol: {params['volume_mult']}x\n"
            f"TP dist: {params['tp_distance_atr']}x ATR | Sweep: {params['sweep_mode']}\n"
            f"H4 struct: {params['require_h4_structure']} | H1 struct: {params['require_h1_structure']}"
        )
        send_telegram(msg)
    else:
        send_telegram(f"ℹ️ Level sudah di batas {'atas' if direction=='up' else 'bawah'} (level {new}).")

# ---------- TELEGRAM POLLING ----------
def telegram_polling():
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
                        f"Conf: {params['min_confidence']}% | RR: 1:{params['min_rr']}\n"
                        f"Vol: {params['volume_mult']}x | TP dist: {params['tp_distance_atr']}x ATR\n"
                        f"Sweep: {params['sweep_mode']} | H4: {params['require_h4_structure']} | H1: {params['require_h1_structure']}"
                    )
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(5)

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
        if not isinstance(data, list): return None
        tickers = [t for t in data if t["symbol"].endswith("USDT")]
        tickers.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        res = []
        for t in tickers:
            if float(t["lastPrice"]) <= max_price:
                res.append(t["symbol"])
            if len(res) >= limit: break
        return res if res else None
    except:
        return None

def fetch_klines_binance(symbol, interval, limit=200):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if isinstance(data, dict) and "code" in data: return None
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
        if data.get("retCode") != 0: return None
        tickers = data["result"]["list"]
        filtered = []
        for t in tickers:
            if t["symbol"].endswith("USDT"):
                try:
                    if float(t["lastPrice"]) <= max_price: filtered.append(t)
                except: pass
        filtered.sort(key=lambda x: float(x.get("turnover24h", 0)), reverse=True)
        return [t["symbol"] for t in filtered[:limit]]
    except:
        return None

def fetch_klines_bybit(symbol, interval, limit=200):
    bybit_interval = INTERVAL_MAP.get(interval, "60")
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={bybit_interval}&limit={limit}"
    try:
        data = requests.get(url, timeout=15).json()
        if data.get("retCode") != 0: return None
        rows = data["result"]["list"]
        if not rows: return None
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
    if len(df) < 80: return None
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
    if len(df) < window*2+2: return "ranging"
    df_swing = find_swings(df, window)
    swing_highs = df_swing[df_swing["swing_high"]]["high"]
    swing_lows = df_swing[df_swing["swing_low"]]["low"]
    if len(swing_highs) < 2 or len(swing_lows) < 2: return "ranging"
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

def get_nearest_levels(df):
    df_swing = find_swings(df, window=2)
    swings_low = df_swing[df_swing["swing_low"]]["low"]
    swings_high = df_swing[df_swing["swing_high"]]["high"]
    current = df["close"].iloc[-2]
    supports = sorted([v for v in swings_low if v < current], reverse=True)
    resistances = sorted([v for v in swings_high if v > current])
    return supports, resistances

# ========== ANALISA SINYAL (FULLY ADJUSTABLE) ==========
def analyze_signal(symbol, fetch_func):
    p = get_aggression_params()

    df_d1 = fetch_func(symbol, "1d", 200)
    df_h4 = fetch_func(symbol, "4h", 200)
    df_h1 = fetch_func(symbol, "1h", 150)
    df_m15 = fetch_func(symbol, "15m", 150)
    df_m5 = fetch_func(symbol, "5m", 150)

    if any(d is None for d in [df_d1, df_h4, df_h1, df_m15, df_m5]):
        return None

    df_d1 = add_all_indicators(df_d1)
    df_h4 = add_all_indicators(df_h4)
    df_h1 = add_all_indicators(df_h1)
    df_m15 = add_all_indicators(df_m15)
    df_m5 = add_all_indicators(df_m5)
    if any(d is None for d in [df_d1, df_h4, df_h1, df_m15, df_m5]):
        return None

    # D1 selalu wajib trending
    struct_d1 = market_structure(df_d1, window=5)
    if struct_d1 == "ranging":
        return None
    bias_bull = struct_d1 == "bullish"
    direction = "BUY" if bias_bull else "SELL"
    score = 0

    # EMA D1
    last_d1 = df_d1.iloc[-1]
    if bias_bull and last_d1["close"] > last_d1["ema50"]:
        score += 0.15
    elif not bias_bull and last_d1["close"] < last_d1["ema50"]:
        score += 0.15

    # RSI D1
    if bias_bull and 40 < last_d1["rsi"] < 70:
        score += 0.05
    elif not bias_bull and 30 < last_d1["rsi"] < 60:
        score += 0.05

    # H4
    struct_h4 = market_structure(df_h4, window=3)
    if p["require_h4_structure"] and struct_h4 != struct_d1:
        return None
    if struct_h4 == struct_d1:
        score += 0.10

    sweep_h4, _ = detect_liquidity_sweep(df_h4, "buy" if bias_bull else "sell")
    if sweep_h4:
        score += 0.10

    # H1
    struct_h1 = market_structure(df_h1, window=2)
    if p["require_h1_structure"] and struct_h1 != struct_d1:
        return None
    if struct_h1 == struct_d1:
        score += 0.10

    # Volume H1 (jika multiplier > 0)
    last_h1 = df_h1.iloc[-2]
    if p["volume_mult"] > 0 and last_h1["volume"] <= p["volume_mult"] * last_h1["vol_avg20"]:
        return None
    if last_h1["volume"] > last_h1["vol_avg20"]:
        score += 0.05

    # RSI H1
    if bias_bull and last_h1["rsi"] >= p["rsi_h1_buy_max"]:
        return None
    if not bias_bull and last_h1["rsi"] <= p["rsi_h1_sell_min"]:
        return None

    # M15
    last_m15 = df_m15.iloc[-2]
    if bias_bull and last_m15["ema12"] > last_m15["ema26"]:
        score += 0.10
    elif not bias_bull and last_m15["ema12"] < last_m15["ema26"]:
        score += 0.10
    else:
        return None

    # Sweep requirement
    sweep_m15, _ = detect_liquidity_sweep(df_m15, "buy" if bias_bull else "sell")
    if p["sweep_mode"] == "m15":
        if not sweep_m15:
            return None
        score += 0.15
    elif p["sweep_mode"] == "any":
        if not (sweep_m15 or sweep_h4):
            return None
        if sweep_m15:
            score += 0.15
        elif sweep_h4:
            score += 0.10

    # RSI M15
    if bias_bull and last_m15["rsi"] >= p["rsi_m15_buy_max"]:
        return None
    if not bias_bull and last_m15["rsi"] <= p["rsi_m15_sell_min"]:
        return None
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

    # TP / SL
    supports_h1, resistances_h1 = get_nearest_levels(df_h1)
    supports_m15, resistances_m15 = get_nearest_levels(df_m15)
    entry = round(last_m15["close"], 6)
    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else entry * 0.002

    if direction == "BUY":
        sl_base = supports_h1[0] if supports_h1 else (supports_m15[0] if supports_m15 else entry - atr * 1.5)
        sl = round(sl_base - atr * 0.5, 6)
        tp_base = resistances_h1[0] if resistances_h1 else (resistances_m15[0] if resistances_m15 else entry + atr * 2.0)
        tp = round(tp_base * 0.998, 6)

        if p["tp_distance_atr"] > 0 and (tp - entry) < p["tp_distance_atr"] * atr:
            return None

        risk = entry - sl
        reward = tp - entry
        if reward / risk < p["min_rr"]:
            tp = round(entry + risk * p["min_rr"], 6)
    else:
        sl_base = resistances_h1[0] if resistances_h1 else (resistances_m15[0] if resistances_m15 else entry + atr * 1.5)
        sl = round(sl_base + atr * 0.5, 6)
        tp_base = supports_h1[0] if supports_h1 else (supports_m15[0] if supports_m15 else entry - atr * 2.0)
        tp = round(tp_base * 1.002, 6)

        if p["tp_distance_atr"] > 0 and (entry - tp) < p["tp_distance_atr"] * atr:
            return None

        risk = sl - entry
        reward = entry - tp
        if reward / risk < p["min_rr"]:
            tp = round(entry - risk * p["min_rr"], 6)

    confidence = min(int(score * 100), 95)
    if confidence < p["min_confidence"]:
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
        # Update ban
        to_del = [k for k, v in banned.items() if v <= 0]
        for k in to_del: del banned[k]
        for k in list(banned.keys()): banned[k] -= 1

        # Tentukan sumber data
        coins = get_coins_binance(TOP_COINS, MAX_PRICE)
        fetch_func = fetch_klines_binance
        api_source = "Binance"
        if not coins:
            coins = get_coins_bybit(TOP_COINS, MAX_PRICE)
            fetch_func = fetch_klines_bybit
            api_source = "Bybit"
        if not coins:
            coins = FALLBACK_SYMBOLS[:TOP_COINS]
            test = fetch_klines_binance(coins[0], "1h", 10)
            if test is not None:
                fetch_func = fetch_klines_binance
                api_source = "Fallback+Binance"
            else:
                fetch_func = fetch_klines_bybit
                api_source = "Fallback+Bybit"

        if not coins or not fetch_func:
            send_telegram("❌ Semua API tidak tersedia.")
            time.sleep(SCAN_INTERVAL)
            continue

        print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] {api_source} | Level {AGGRESSION_LEVEL} | {len(coins)} koin")
        signals = []
        fetch_ok = 0
        fetch_fail = 0
        for sym in coins:
            if sym in banned: continue
            try:
                sig = analyze_signal(sym, fetch_func)
                if sig:
                    signals.append(sig)
                    fetch_ok += 1
                else:
                    # Tidak ada sinyal, tapi data mungkin berhasil
                    fetch_ok += 1  # kita anggap berhasil fetch karena analyze_signal tidak error
            except Exception as e:
                fetch_fail += 1
                print(f"  ⚠️ Error {sym}: {e}")
            time.sleep(0.02)

        # Ringkasan
        if fetch_fail > 0:
            send_telegram(f"⚠️ {fetch_fail} koin gagal fetch data.")
        if fetch_ok == 0 and fetch_fail == len(coins):
            send_telegram("❌ Semua koin gagal fetch. API down.")
            time.sleep(SCAN_INTERVAL)
            continue

        if not signals:
            params = get_aggression_params()
            send_telegram(f"❌ Tidak ada sinyal (Conf ≥ {params['min_confidence']}%, RR ≥ 1:{params['min_rr']})")
            time.sleep(SCAN_INTERVAL)
            continue

        send_telegram(f"🔔 <b>Ditemukan {len(signals)} sinyal ({api_source}):</b>")
        for sig in signals:
            msg = (
                f"<b>📊 {sig['signal']} {sig['symbol']}</b>\n"
                f"Entry: {sig['entry']}\n"
                f"TP: {sig['tp']} | SL: {sig['sl']}\n"
                f"Conf: {sig['confidence']}% | RR: 1:{sig['rr']} | ATR: {sig['atr']}"
            )
            send_telegram(msg)
            banned[sig["symbol"]] = BAN_CYCLES

        send_telegram(f"📛 {len(signals)} koin di-ban.")
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
    print("  SIGNAL BROADCASTER – Adjustable (Longgar/Ketat)")
    print("=" * 60)
    send_telegram("🚀 <b>Bot Adjustable siap!</b> /up /down /status")

    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=telegram_polling, daemon=True).start()
    main_loop()
