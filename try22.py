#!/usr/bin/env python3
"""
SIGNAL BROADCASTER – Three‑Level Aggression (0 = Perfect, -1 = High Freq, -2 = Ultra Freq)
Level -2: volume & jarak TP nonaktif agar sinyal tetap muncul, confidence 55% & RR 1.3 tetap dijaga.
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

# ---------- AGGRESSION (0, -1, -2) ----------
AGGRESSION_LEVEL = 0
LEVEL_LOCK = threading.Lock()

def get_aggression_params():
    with LEVEL_LOCK:
        lvl = AGGRESSION_LEVEL

    if lvl == 0:
        return {
            "min_confidence": 75,
            "min_rr": 2.0,
            "volume_mult": 2.0,
            "tp_distance_atr": 2.5,
            "rsi_h1_buy_max": 50,
            "rsi_h1_sell_min": 50,
            "rsi_m15_buy_max": 60,
            "rsi_m15_sell_min": 40,
            "require_h4_structure": True,
            "require_h1_structure": True,
            "sweep_mode": "m15",
            "entry_shift_pips": 0,
            "require_confirmation": True,
        }
    elif lvl == -1:
        return {
            "min_confidence": 60,
            "min_rr": 1.5,
            "volume_mult": 1.0,
            "tp_distance_atr": 1.5,
            "rsi_h1_buy_max": 65,
            "rsi_h1_sell_min": 35,
            "rsi_m15_buy_max": 70,
            "rsi_m15_sell_min": 30,
            "require_h4_structure": False,
            "require_h1_structure": False,
            "sweep_mode": "any",
            "entry_shift_pips": 2,
            "require_confirmation": False,
        }
    else:  # -2 (ULTRA FREQUENCY, CONFIDENCE TETAP 55%, RR 1.3)
        return {
            "min_confidence": 55,
            "min_rr": 1.3,
            "volume_mult": 0.0,            # nonaktif
            "tp_distance_atr": 0.0,        # nonaktif
            "rsi_h1_buy_max": 75,          # longgar
            "rsi_h1_sell_min": 25,
            "rsi_m15_buy_max": 78,
            "rsi_m15_sell_min": 22,
            "require_h4_structure": False,
            "require_h1_structure": False,
            "sweep_mode": "any",
            "entry_shift_pips": 5,         # lebih berani
            "require_confirmation": False,
        }

def set_aggression(direction):
    global AGGRESSION_LEVEL
    with LEVEL_LOCK:
        old = AGGRESSION_LEVEL
        if direction == "up":
            AGGRESSION_LEVEL = min(0, AGGRESSION_LEVEL + 1)
        else:
            AGGRESSION_LEVEL = max(-2, AGGRESSION_LEVEL - 1)
        new = AGGRESSION_LEVEL
    if old != new:
        p = get_aggression_params()
        send_telegram(f"🔧 Level {old} → {new}\nConf: {p['min_confidence']}% | RR: 1:{p['min_rr']} | Vol: {p['volume_mult']}x | TP dist: {p['tp_distance_atr']}xATR")
    else:
        send_telegram(f"ℹ️ Sudah di level {new}.")

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
                if text == "/up": set_aggression("up")
                elif text == "/down": set_aggression("down")
                elif text == "/status":
                    p = get_aggression_params()
                    send_telegram(f"📊 Level {AGGRESSION_LEVEL}\nConf: {p['min_confidence']}% | RR: 1:{p['min_rr']} | Vol: {p['volume_mult']}x | TP dist: {p['tp_distance_atr']}xATR")
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}"); time.sleep(5)

# ========== BINANCE / BYBIT FUNCTIONS ==========
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

def find_best_entry_tp_sl(df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr, p):
    supports_h1, resistances_h1 = get_levels(df_h1)
    supports_m15, resistances_m15 = get_levels(df_m15)
    all_supports = sorted(supports_h1 + supports_m15, reverse=True)
    all_resistances = sorted(resistances_h1 + resistances_m15)

    shift_pct = p["entry_shift_pips"] * 0.0001
    shift = shift_pct * entry_raw if p["entry_shift_pips"] > 0 else 0

    if bias_bull:
        nearest_support = None
        for sup in all_supports:
            if sup < entry_raw:
                nearest_support = sup
                break
        final_entry = round(nearest_support + atr * 0.2 + shift, 6) if nearest_support else entry_raw + shift

        sl_candidates = []
        if sweep_level and sweep_level < final_entry:
            sl_candidates.append(sweep_level - atr * 0.3)
        count = 0
        for sup in all_supports:
            if sup < final_entry:
                sl_candidates.append(sup - atr * 0.3)
                count += 1
                if count >= 2: break
        if not sl_candidates:
            sl_candidates.append(final_entry - atr * 1.2)
        sl = round(min(sl_candidates), 6)

        tp = None
        for res in all_resistances:
            if res > final_entry:
                tp = round(res * 0.999, 6)
                break
        if tp is None or tp <= final_entry:
            tp = round(final_entry + atr * 2.0, 6)

        risk = final_entry - sl
        reward = tp - final_entry
        if reward / risk < p["min_rr"]:
            for i, res in enumerate(all_resistances):
                if res > final_entry and i > 0:
                    tp = round(res * 0.999, 6)
                    reward = tp - final_entry
                    if reward / risk >= p["min_rr"]:
                        break
    else:
        nearest_resistance = None
        for res in all_resistances:
            if res > entry_raw:
                nearest_resistance = res
                break
        final_entry = round(nearest_resistance - atr * 0.2 - shift, 6) if nearest_resistance else entry_raw - shift

        sl_candidates = []
        if sweep_level and sweep_level > final_entry:
            sl_candidates.append(sweep_level + atr * 0.3)
        count = 0
        for res in all_resistances:
            if res > final_entry:
                sl_candidates.append(res + atr * 0.3)
                count += 1
                if count >= 2: break
        if not sl_candidates:
            sl_candidates.append(final_entry + atr * 1.2)
        sl = round(max(sl_candidates), 6)

        tp = None
        for sup in all_supports:
            if sup < final_entry:
                tp = round(sup * 1.001, 6)
                break
        if tp is None or tp >= final_entry:
            tp = round(final_entry - atr * 2.0, 6)

        risk = sl - final_entry
        reward = final_entry - tp
        if reward / risk < p["min_rr"]:
            for i, sup in enumerate(all_supports):
                if sup < final_entry and i > 0:
                    tp = round(sup * 1.001, 6)
                    reward = final_entry - tp
                    if reward / risk >= p["min_rr"]:
                        break

    if bias_bull:
        if sl >= final_entry: sl = round(final_entry - atr * 1.0, 6)
        if tp <= final_entry: tp = round(final_entry + atr * 0.5, 6)
    else:
        if sl <= final_entry: sl = round(final_entry + atr * 1.0, 6)
        if tp >= final_entry: tp = round(final_entry - atr * 0.5, 6)

    return final_entry, tp, sl

def analyze_signal(symbol, fetch_func):
    p = get_aggression_params()

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

    struct_d1 = market_structure(df_d1, 5)
    if struct_d1 == "ranging": return None
    bias_bull = struct_d1 == "bullish"
    direction = "BUY" if bias_bull else "SELL"
    score = 0

    last_d1 = df_d1.iloc[-1]
    if bias_bull and last_d1["close"] > last_d1["ema50"]: score += 0.15
    elif not bias_bull and last_d1["close"] < last_d1["ema50"]: score += 0.15
    if bias_bull and 40 < last_d1["rsi"] < 70: score += 0.05
    elif not bias_bull and 30 < last_d1["rsi"] < 60: score += 0.05

    struct_h4 = market_structure(df_h4, 3)
    if p["require_h4_structure"] and struct_h4 != struct_d1: return None
    if struct_h4 == struct_d1: score += 0.10
    sweep_h4, _ = detect_liquidity_sweep(df_h4, "buy" if bias_bull else "sell")
    if sweep_h4: score += 0.10

    struct_h1 = market_structure(df_h1, 2)
    if p["require_h1_structure"] and struct_h1 != struct_d1: return None
    if struct_h1 == struct_d1: score += 0.10
    last_h1 = df_h1.iloc[-2]
    if p["volume_mult"] > 0 and last_h1["volume"] <= p["volume_mult"] * last_h1["vol_avg20"]: return None
    if last_h1["volume"] > last_h1["vol_avg20"]: score += 0.05
    if bias_bull and last_h1["rsi"] >= p["rsi_h1_buy_max"]: return None
    if not bias_bull and last_h1["rsi"] <= p["rsi_h1_sell_min"]: return None

    last_m15 = df_m15.iloc[-2]
    if bias_bull and last_m15["ema12"] > last_m15["ema26"]: score += 0.10
    elif not bias_bull and last_m15["ema12"] < last_m15["ema26"]: score += 0.10
    else: return None

    sweep_m15, sweep_level = detect_liquidity_sweep(df_m15, "buy" if bias_bull else "sell")
    if p["sweep_mode"] == "m15" and not sweep_m15: return None
    if p["sweep_mode"] == "any" and not (sweep_m15 or sweep_h4): return None
    if sweep_m15: score += 0.15
    elif sweep_h4: score += 0.10

    if bias_bull and last_m15["rsi"] >= p["rsi_m15_buy_max"]: return None
    if not bias_bull and last_m15["rsi"] <= p["rsi_m15_sell_min"]: return None
    if bias_bull and last_m15["rsi"] < 65: score += 0.05
    elif not bias_bull and last_m15["rsi"] > 35: score += 0.05

    if p["require_confirmation"]:
        if bias_bull and not has_bullish_confirmation(df_m15): return None
        if not bias_bull and not has_bearish_confirmation(df_m15): return None
        score += 0.05

    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["close"] > last_m5["ema12"]: score += 0.05
    elif not bias_bull and last_m5["close"] < last_m5["ema12"]: score += 0.05
    else: return None

    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else last_m15["close"] * 0.002
    entry_raw = round(last_m15["close"], 6)

    final_entry, tp, sl = find_best_entry_tp_sl(
        df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr, p
    )

    # Jarak TP hanya dicek jika > 0
    if p["tp_distance_atr"] > 0:
        if bias_bull and (tp - final_entry) < p["tp_distance_atr"] * atr: return None
        if not bias_bull and (final_entry - tp) < p["tp_distance_atr"] * atr: return None

    confidence = min(int(score * 100), 95)
    if confidence < p["min_confidence"]: return None

    return {
        "symbol": symbol,
        "signal": direction,
        "entry": final_entry,
        "tp": tp,
        "sl": sl,
        "confidence": confidence,
        "atr": round(atr, 6),
        "rr": round(abs(tp - final_entry) / abs(final_entry - sl), 2) if abs(final_entry - sl) > 0 else 0
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

        print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] {api_source} | Level {AGGRESSION_LEVEL} | {len(coins)} koin")
        signals = []
        fetch_ok = fetch_fail = 0
        for sym in coins:
            if sym in banned: continue
            try:
                sig = analyze_signal(sym, fetch_func)
                if sig:
                    signals.append(sig)
                    fetch_ok += 1
                else: fetch_ok += 1
            except Exception as e:
                fetch_fail += 1
                print(f"  ⚠️ Error {sym}: {e}")
            time.sleep(0.02)

        if fetch_fail > 0: send_telegram(f"⚠️ {fetch_fail} koin gagal fetch.")
        if fetch_ok == 0 and fetch_fail == len(coins):
            send_telegram("❌ Semua koin gagal fetch.")
            time.sleep(SCAN_INTERVAL)
            continue

        if not signals:
            p = get_aggression_params()
            send_telegram(f"❌ Tidak ada sinyal (Conf ≥ {p['min_confidence']}%)")
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
    print("  THREE-LEVEL SIGNAL BROADCASTER")
    print("=" * 60)
    send_telegram("🚀 <b>Bot Sinyal 3-Level siap!</b> /up /down /status")
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=telegram_polling, daemon=True).start()
    main_loop()
