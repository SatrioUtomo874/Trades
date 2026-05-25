#!/usr/bin/env python3
"""
SIGNAL BROADCASTER – Single Mode Adjustable (Full Code)
Longgar secara default (seperti level -2). Semua parameter bisa diubah via /set.
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

# ---------- SETTINGS DEFAULT (LONGGAR) ----------
settings = {
    "min_confidence": 50,          # minimal confidence %
    "min_rr": 1.3,                 # risk/reward minimal
    "base_score": 60,              # skor awal
    "h1_penalty": 5,               # penalti jika H1 melawan tren
    "entry_shift_pips": 4,         # pergeseran entry (agresif)
    "rsi_h1_buy_max": 70,         # batas RSI H1 untuk BUY
    "rsi_h1_sell_min": 30,        # batas RSI H1 untuk SELL
    "rsi_m15_buy_max": 75,        # batas RSI M15 untuk BUY
    "rsi_m15_sell_min": 25,       # batas RSI M15 untuk SELL
}

# ---------- TELEGRAM ----------
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except: pass

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
                            else:
                                send_telegram(f"❌ Key tidak dikenal: {key}\nGunakan: {', '.join(settings.keys())}")
                        except ValueError:
                            send_telegram("❌ Value harus berupa angka.")
                elif text == "/status":
                    s = "\n".join([f"{k}: {v}" for k, v in settings.items()])
                    send_telegram(f"📊 <b>Pengaturan Saat Ini:</b>\n<pre>{s}</pre>")
                elif text == "/menu":
                    send_telegram("""<b>Command List:</b>
/status - Lihat pengaturan
/set key value - Ubah pengaturan
/menu - Tampilkan menu""")
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}"); time.sleep(5)

# ========== BINANCE / BYBIT DATA FUNCTIONS ==========
def get_coins_binance(limit=50, max_price=100.0):
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        data = requests.get(url, timeout=15).json()
        if not isinstance(data, list): return None
        tickers = [t for t in data if t["symbol"].endswith("USDT")]
        tickers.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        res = []
        for t in tickers:
            if float(t["lastPrice"]) <= max_price: res.append(t["symbol"])
            if len(res) >= limit: break
        return res if res else None
    except: return None

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
        for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]]
    except: return None

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
    except: return None

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
        for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]]
    except: return None

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
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_diff"] = df["macd"] - df["macd_signal"]
    return df

def market_structure(df, window=3):
    if len(df) < window*2+2: return "ranging"
    highs, lows = df["high"], df["low"]
    sh, sl = [], []
    for i in range(window, len(df)-window):
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max(): sh.append(highs.iloc[i])
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min(): sl.append(lows.iloc[i])
    if len(sh) < 2 or len(sl) < 2: return "ranging"
    hh = sh[-1] > sh[-2]; hl = sl[-1] > sl[-2]; lh = sh[-1] < sh[-2]; ll = sl[-1] < sl[-2]
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

def find_fvg(df, direction):
    last_idx = len(df) - 2
    if last_idx < 3: return None
    if direction == "buy":
        if df["low"].iloc[last_idx] > df["high"].iloc[last_idx-2]:
            return df["high"].iloc[last_idx-2], df["low"].iloc[last_idx]
    else:
        if df["high"].iloc[last_idx] < df["low"].iloc[last_idx-2]:
            return df["low"].iloc[last_idx-2], df["high"].iloc[last_idx]
    return None

def find_order_block(df, direction):
    last_idx = len(df) - 2; last_close = df["close"].iloc[last_idx]
    if direction == "buy":
        for i in range(last_idx-1, max(last_idx-20, 0), -1):
            if df["close"].iloc[i] < df["open"].iloc[i]:
                if i+1 <= last_idx and df["close"].iloc[i+1] > df["open"].iloc[i+1] and last_close > df["high"].iloc[i]:
                    return df["high"].iloc[i], df["low"].iloc[i]
    else:
        for i in range(last_idx-1, max(last_idx-20, 0), -1):
            if df["close"].iloc[i] > df["open"].iloc[i]:
                if i+1 <= last_idx and df["close"].iloc[i+1] < df["open"].iloc[i+1] and last_close < df["low"].iloc[i]:
                    return df["high"].iloc[i], df["low"].iloc[i]
    return None

def has_bullish_confirmation(df):
    last = df.iloc[-2]; prev = df.iloc[-3]
    if prev["close"] < prev["open"] and last["close"] > last["open"]:
        if last["close"] > prev["open"] and last["open"] < prev["close"]: return True
    body = abs(last["close"] - last["open"])
    lower_shadow = min(last["open"], last["close"]) - last["low"]
    upper_shadow = last["high"] - max(last["open"], last["close"])
    if lower_shadow > body * 1.5 and upper_shadow < body * 0.5: return True
    return False

def has_bearish_confirmation(df):
    last = df.iloc[-2]; prev = df.iloc[-3]
    if prev["close"] > prev["open"] and last["close"] < last["open"]:
        if last["open"] > prev["close"] and last["close"] < prev["open"]: return True
    body = abs(last["close"] - last["open"])
    upper_shadow = last["high"] - max(last["open"], last["close"])
    lower_shadow = min(last["open"], last["close"]) - last["low"]
    if upper_shadow > body * 1.5 and lower_shadow < body * 0.5: return True
    return False

def get_levels(df):
    highs, lows = df["high"], df["low"]
    sh, sl = [], []
    for i in range(2, len(df)-2):
        if highs.iloc[i] == highs.iloc[i-2:i+3].max(): sh.append(highs.iloc[i])
        if lows.iloc[i] == lows.iloc[i-2:i+3].min(): sl.append(lows.iloc[i])
    return sorted(sl, reverse=True), sorted(sh)

# ========== ANALISA SINYAL ==========
def analyze_signal(symbol, fetch_func):
    df_d1 = fetch_func(symbol, "1d", 200)
    df_h4 = fetch_func(symbol, "4h", 200)
    df_h1 = fetch_func(symbol, "1h", 150)
    df_m15 = fetch_func(symbol, "15m", 150)
    df_m5 = fetch_func(symbol, "5m", 150)
    if any(d is None for d in [df_d1, df_h4, df_h1, df_m15, df_m5]): return None

    df_d1 = add_all_indicators(df_d1)
    df_h4 = add_all_indicators(df_h4)
    df_h1 = add_all_indicators(df_h1)
    df_m15 = add_all_indicators(df_m15)
    df_m5 = add_all_indicators(df_m5)
    if any(d is None for d in [df_d1, df_h4, df_h1, df_m15, df_m5]): return None

    # 1. D1 trend (wajib)
    struct_d1 = market_structure(df_d1, 5)
    if struct_d1 == "ranging": return None
    bias_bull = struct_d1 == "bullish"
    direction = "BUY" if bias_bull else "SELL"
    score = settings["base_score"]

    # 2. Sweep (wajib)
    sweep_m15, sweep_level_m15 = detect_liquidity_sweep(df_m15, "buy" if bias_bull else "sell")
    sweep_h4, _ = detect_liquidity_sweep(df_h4, "buy" if bias_bull else "sell")
    if not (sweep_m15 or sweep_h4): return None
    if sweep_m15: score += 15
    else: score += 8

    # 3. D1 bonus
    last_d1 = df_d1.iloc[-1]
    if bias_bull and last_d1["close"] > last_d1["ema50"]: score += 5
    elif not bias_bull and last_d1["close"] < last_d1["ema50"]: score += 5
    if bias_bull and 40 < last_d1["rsi"] < 70: score += 3
    elif not bias_bull and 30 < last_d1["rsi"] < 60: score += 3

    # 4. H4 bonus
    struct_h4 = market_structure(df_h4, 3)
    if struct_h4 == struct_d1: score += 8
    elif has_bullish_confirmation(df_h4) if bias_bull else has_bearish_confirmation(df_h4): score += 4
    if sweep_h4: score += 5

    # 5. H1 penalti & bonus
    struct_h1 = market_structure(df_h1, 2)
    if struct_h1 == struct_d1: score += 8
    else: score -= settings["h1_penalty"]

    last_h1 = df_h1.iloc[-2]
    if last_h1["volume"] > last_h1["vol_avg20"]: score += 5
    if bias_bull and last_h1["rsi"] < settings["rsi_h1_buy_max"]: score += 3
    elif not bias_bull and last_h1["rsi"] > settings["rsi_h1_sell_min"]: score += 3

    # 6. M15 bonus
    last_m15 = df_m15.iloc[-2]
    if bias_bull and last_m15["ema12"] > last_m15["ema26"]: score += 8
    elif not bias_bull and last_m15["ema12"] < last_m15["ema26"]: score += 8
    else: return None

    if bias_bull and last_m15["rsi"] < settings["rsi_m15_buy_max"]: score += 3
    elif not bias_bull and last_m15["rsi"] > settings["rsi_m15_sell_min"]: score += 3

    # 7. Konfirmasi candlestick bonus
    if bias_bull and has_bullish_confirmation(df_m15): score += 5
    elif not bias_bull and has_bearish_confirmation(df_m15): score += 5

    # 8. OB/FVG bonus
    ob_m15 = find_order_block(df_m15, "buy" if bias_bull else "sell")
    fvg_m15 = find_fvg(df_m15, "buy" if bias_bull else "sell")
    if ob_m15: score += 5
    elif fvg_m15: score += 5

    # 9. M5 konfirmasi
    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["close"] > last_m5["ema12"]: score += 3
    elif not bias_bull and last_m5["close"] < last_m5["ema12"]: score += 3
    else: return None

    confidence = min(score, 100)
    if confidence < settings["min_confidence"]: return None

    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else last_m15["close"] * 0.002
    entry_raw = round(last_m15["close"], 6)

    supports, resistances = get_levels(df_h1 if struct_h1 == struct_d1 else df_m15)
    sweep_level = sweep_level_m15 if sweep_m15 else None
    sup_cand = [s for s in supports if s < entry_raw]
    res_cand = [r for r in resistances if r > entry_raw]

    if ob_m15:
        if bias_bull: sup_cand.append(ob_m15[0])
        else: res_cand.append(ob_m15[1])
    if fvg_m15:
        if bias_bull: sup_cand.append(fvg_m15[0])
        else: res_cand.append(fvg_m15[1])

    shift_pct = settings["entry_shift_pips"] * 0.0001
    shift = shift_pct * entry_raw if settings["entry_shift_pips"] > 0 else 0

    if bias_bull:
        best_support = max(sup_cand) if sup_cand else entry_raw - atr
        final_entry = round(best_support + atr * 0.2 + shift, 6)
        sl = round(min(sup_cand + [sweep_level] if sweep_level else sup_cand) - atr * 0.5, 6) if sup_cand else round(final_entry - atr * 1.5, 6)
        tp = round(min(res_cand) * 0.999, 6) if res_cand else round(final_entry + atr * 2.0, 6)
    else:
        best_resistance = min(res_cand) if res_cand else entry_raw + atr
        final_entry = round(best_resistance - atr * 0.2 - shift, 6)
        sl = round(max(res_cand + [sweep_level] if sweep_level else res_cand) + atr * 0.5, 6) if res_cand else round(final_entry + atr * 1.5, 6)
        tp = round(max(sup_cand) * 1.001, 6) if sup_cand else round(final_entry - atr * 2.0, 6)

    risk = abs(final_entry - sl)
    reward = abs(tp - final_entry)
    if risk > 0 and reward / risk < settings["min_rr"]: return None

    return {
        "symbol": symbol,
        "signal": direction,
        "entry": final_entry,
        "tp": tp,
        "sl": sl,
        "confidence": confidence,
        "atr": round(atr, 6),
        "rr": round(reward/risk, 2) if risk > 0 else 0
    }

# ========== LOOP UTAMA ==========
banned = {}

def main_loop():
    global banned
    while True:
        to_del = [k for k, v in banned.items() if v <= 0]
        for k in to_del: del banned[k]
        for k in list(banned.keys()): banned[k] -= 1

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

        print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] {api_source} | {len(coins)} koin")
        signals = []
        for sym in coins:
            if sym in banned: continue
            try:
                sig = analyze_signal(sym, fetch_func)
                if sig:
                    signals.append(sig)
            except: pass
            time.sleep(0.02)

        if not signals:
            send_telegram(f"❌ Tidak ada sinyal (Conf ≥ {settings['min_confidence']}%)")
            time.sleep(SCAN_INTERVAL)
            continue

        send_telegram(f"🔔 <b>{len(signals)} sinyal ({api_source})</b>")
        for sig in signals:
            msg = (
                f"<b>📊 {sig['signal']} {sig['symbol']}</b>\n"
                f"Entry: {sig['entry']}\nTP: {sig['tp']} | SL: {sig['sl']}\n"
                f"Conf: {sig['confidence']}% | RR: 1:{sig['rr']} | ATR: {sig['atr']}"
            )
            send_telegram(msg)
            banned[sig["symbol"]] = BAN_CYCLES
        send_telegram(f"📛 {len(signals)} koin di-ban.")
        time.sleep(SCAN_INTERVAL)

# ========== FLASK ==========
app = Flask(__name__)
@app.route('/')
def home(): return "Bot is alive", 200
def run_flask(): app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    print("=" * 60)
    print("  SIGNAL BROADCASTER – Single Adjustable Mode (Full)")
    print("=" * 60)
    send_telegram("🚀 <b>Bot Sinyal Adjustable siap!</b>\nGunakan /set key value untuk mengubah parameter.\n/menu untuk bantuan.")
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=telegram_polling, daemon=True).start()
    main_loop()
