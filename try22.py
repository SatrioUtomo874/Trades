#!/usr/bin/env python3
"""
SIGNAL BROADCASTER – High Winrate Entry Logic
- Entry dari level teknikal + konfirmasi candlestick
- SL diperkuat multi-level
- Agresi adjustable (level -3 lebih berani, level 0 normal)
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
    p = {
        "min_confidence": 70,
        "min_rr": 1.8,
        "volume_mult": 1.5,
        "tp_distance_atr": 2.0,
        "rsi_h1_buy_max": 60,
        "rsi_h1_sell_min": 40,
        "rsi_m15_buy_max": 65,
        "rsi_m15_sell_min": 35,
        "require_h4_structure": True,
        "require_h1_structure": True,
        "sweep_mode": "m15",
        "entry_shift_pips": 0,
        "require_confirmation": True,   # konfirmasi candlestick wajib di level normal
    }
    if lvl > 0:
        p["min_confidence"] += 5 * lvl
        p["min_rr"] += 0.2 * lvl
        p["volume_mult"] += 0.5 * lvl
        p["tp_distance_atr"] += 0.5 * lvl
        p["rsi_h1_buy_max"] -= 5 * lvl
        p["rsi_h1_sell_min"] += 5 * lvl
        p["rsi_m15_buy_max"] -= 5 * lvl
        p["rsi_m15_sell_min"] += 5 * lvl
    elif lvl < 0:
        p["min_confidence"] += 5 * lvl
        p["min_rr"] += 0.2 * lvl
        p["volume_mult"] += 0.3 * lvl
        p["tp_distance_atr"] += 0.5 * lvl
        p["rsi_h1_buy_max"] -= 5 * lvl
        p["rsi_h1_sell_min"] += 5 * lvl
        p["rsi_m15_buy_max"] -= 5 * lvl
        p["rsi_m15_sell_min"] += 5 * lvl
        if lvl <= -2:
            p["entry_shift_pips"] = 4
            p["require_h4_structure"] = False
            p["require_h1_structure"] = False
            p["sweep_mode"] = "any"
            p["require_confirmation"] = False  # di level -3, konfirmasi tidak wajib
        elif lvl == -1:
            p["entry_shift_pips"] = 2
            p["require_h4_structure"] = True
            p["require_h1_structure"] = False
            p["sweep_mode"] = "any"
            p["require_confirmation"] = False
    # Batas aman
    p["min_confidence"] = max(50, min(85, p["min_confidence"]))
    p["min_rr"] = max(1.2, min(2.5, p["min_rr"]))
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
        if direction == "up": AGGRESSION_LEVEL = min(2, AGGRESSION_LEVEL + 1)
        else: AGGRESSION_LEVEL = max(-3, AGGRESSION_LEVEL - 1)
        new = AGGRESSION_LEVEL
    if old != new:
        p = get_aggression_params()
        send_telegram(f"🔧 Level {old} → {new}\nConf: {p['min_confidence']}% | RR: 1:{p['min_rr']} | Vol: {p['volume_mult']}x | Confirm: {p['require_confirmation']}")
    else:
        send_telegram(f"ℹ️ Level sudah di batas ({new}).")

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
                    send_telegram(f"📊 Level {AGGRESSION_LEVEL}\nConf: {p['min_confidence']}% | RR: 1:{p['min_rr']} | Confirm: {p['require_confirmation']}")
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}"); time.sleep(5)

# ========== BINANCE / BYBIT FUNCTIONS (sama seperti sebelumnya) ==========
# ... (gunakan kode yang sudah ada, tidak diubah)
# Untuk ringkas, saya tidak menyalin ulang. Asumsikan fungsi get_coins_* dan fetch_klines_* sudah ada.

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
    # MACD
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

# ---------- KONFIRMASI CANDLESTICK ----------
def has_bullish_confirmation(df):
    """Candle terakhir (M15) adalah bullish engulfing atau bullish pin bar."""
    last = df.iloc[-2]  # closed candle
    prev = df.iloc[-3]
    # Bullish engulfing: prev bearish, last bullish, last close > prev open, last open < prev close
    if prev["close"] < prev["open"] and last["close"] > last["open"]:
        if last["close"] > prev["open"] and last["open"] < prev["close"]:
            return True
    # Bullish pin bar: long lower shadow, small body, close near high
    body = abs(last["close"] - last["open"])
    lower_shadow = min(last["open"], last["close"]) - last["low"]
    upper_shadow = last["high"] - max(last["open"], last["close"])
    if lower_shadow > body * 1.5 and upper_shadow < body * 0.5:
        return True
    return False

def has_bearish_confirmation(df):
    last = df.iloc[-2]
    prev = df.iloc[-3]
    # Bearish engulfing
    if prev["close"] > prev["open"] and last["close"] < last["open"]:
        if last["open"] > prev["close"] and last["close"] < prev["open"]:
            return True
    # Bearish pin bar
    body = abs(last["close"] - last["open"])
    upper_shadow = last["high"] - max(last["open"], last["close"])
    lower_shadow = min(last["open"], last["close"]) - last["low"]
    if upper_shadow > body * 1.5 and lower_shadow < body * 0.5:
        return True
    return False

# ---------- ENTRY/TP/SL DARI LEVEL TEKNIKAL (DIPERBAIKI) ----------
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
        # Entry: di atas support terdekat, tapi tidak terlalu jauh
        nearest_support = None
        for sup in all_supports:
            if sup < entry_raw:
                nearest_support = sup
                break
        if nearest_support is not None:
            final_entry = round(nearest_support + atr * 0.2 + shift, 6)  # sedikit di atas support
        else:
            final_entry = entry_raw + shift

        # SL: di bawah support terkuat (cari 2 level jika ada)
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
        sl = round(min(sl_candidates), 6)  # SL terendah

        # TP: resistance terdekat di atas
        tp = None
        for res in all_resistances:
            if res > final_entry:
                tp = round(res * 0.999, 6)
                break
        if tp is None or tp <= final_entry:
            tp = round(final_entry + atr * 2.0, 6)

        # Pastikan RR minimal (tidak memaksa TP, hanya cek)
        risk = final_entry - sl
        reward = tp - final_entry
        if reward / risk < p["min_rr"]:
            # Coba resistance berikutnya
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
        if nearest_resistance is not None:
            final_entry = round(nearest_resistance - atr * 0.2 - shift, 6)
        else:
            final_entry = entry_raw - shift

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

    # Safety check
    if bias_bull:
        if sl >= final_entry: sl = round(final_entry - atr * 1.0, 6)
        if tp <= final_entry: tp = round(final_entry + atr * 0.5, 6)
    else:
        if sl <= final_entry: sl = round(final_entry + atr * 1.0, 6)
        if tp >= final_entry: tp = round(final_entry - atr * 0.5, 6)

    return final_entry, tp, sl

# ========== ANALISA SINYAL (DENGAN PERBAIKAN) ==========
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

    # D1
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

    # H4
    struct_h4 = market_structure(df_h4, 3)
    if p["require_h4_structure"] and struct_h4 != struct_d1: return None
    if struct_h4 == struct_d1: score += 0.10
    sweep_h4, _ = detect_liquidity_sweep(df_h4, "buy" if bias_bull else "sell")
    if sweep_h4: score += 0.10

    # H1
    struct_h1 = market_structure(df_h1, 2)
    if p["require_h1_structure"] and struct_h1 != struct_d1: return None
    if struct_h1 == struct_d1: score += 0.10
    last_h1 = df_h1.iloc[-2]
    if p["volume_mult"] > 0 and last_h1["volume"] <= p["volume_mult"] * last_h1["vol_avg20"]: return None
    if last_h1["volume"] > last_h1["vol_avg20"]: score += 0.05
    if bias_bull and last_h1["rsi"] >= p["rsi_h1_buy_max"]: return None
    if not bias_bull and last_h1["rsi"] <= p["rsi_h1_sell_min"]: return None

    # M15
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

    # Konfirmasi candlestick (jika diwajibkan)
    if p["require_confirmation"]:
        if bias_bull and not has_bullish_confirmation(df_m15): return None
        if not bias_bull and not has_bearish_confirmation(df_m15): return None
        score += 0.05  # bonus jika ada konfirmasi

    # M5
    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["close"] > last_m5["ema12"]: score += 0.05
    elif not bias_bull and last_m5["close"] < last_m5["ema12"]: score += 0.05
    else: return None

    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else last_m15["close"] * 0.002
    entry_raw = round(last_m15["close"], 6)

    final_entry, tp, sl = find_best_entry_tp_sl(
        df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr, p
    )

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

# ========== LOOP UTAMA (tidak berubah) ==========
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
    print("  HIGH WINRATE SIGNAL BROADCASTER")
    print("=" * 60)
    send_telegram("🚀 <b>Bot High Winrate siap!</b> /up /down /status")
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=telegram_polling, daemon=True).start()
    main_loop()
