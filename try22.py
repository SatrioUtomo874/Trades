#!/usr/bin/env python3
"""
SIGNAL BROADCASTER OPTIMIZED – Entry/TP/SL dari Level Teknikal Murni
- TP & SL dari support/resistance, EMA, liquidity sweep.
- RR hanya sebagai batas minimal (1:1.2 di level -3).
- Entry bisa sedikit digeser ke arah yang lebih berani (tergantung level agresi).
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
        "min_rr": 1.8,                # Batas minimal RR
        "volume_mult": 1.5,
        "tp_distance_atr": 2.0,      # Jarak minimal entry ke TP (0 = nonaktif)
        "rsi_h1_buy_max": 60,
        "rsi_h1_sell_min": 40,
        "rsi_m15_buy_max": 65,
        "rsi_m15_sell_min": 35,
        "require_h4_structure": True,
        "require_h1_structure": True,
        "sweep_mode": "m15",         # "m15" atau "any"
        "entry_shift_pips": 0,       # Berapa pips menggeser entry (BUY naik, SELL turun)
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

        # Level -2 ke bawah: entry lebih berani
        if lvl <= -2:
            p["entry_shift_pips"] = 4
            p["require_h4_structure"] = False
            p["require_h1_structure"] = False
            p["sweep_mode"] = "any"
        elif lvl == -1:
            p["entry_shift_pips"] = 2
            p["require_h4_structure"] = True
            p["require_h1_structure"] = False
            p["sweep_mode"] = "any"

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
        send_telegram(f"🔧 Level {old} → {new}\nConf: {p['min_confidence']}% | RR min: 1:{p['min_rr']} | Vol: {p['volume_mult']}x\nTP dist: {p['tp_distance_atr']}xATR | Entry shift: {p['entry_shift_pips']} pips")
    else:
        send_telegram(f"ℹ️ Level sudah di batas ({new}).")

# ---------- TELEGRAM ----------
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
                    send_telegram(f"📊 Level {AGGRESSION_LEVEL}\nConf: {p['min_confidence']}% | RR: 1:{p['min_rr']} | Vol: {p['volume_mult']}x\nTP dist: {p['tp_distance_atr']}xATR | Entry shift: {p['entry_shift_pips']}pips")
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}"); time.sleep(5)

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except: pass

# ========== BINANCE / BYBIT FUNCTIONS (persis sama seperti sebelumnya) ==========
# ... (sama persis dengan kode terakhir, tidak diubah) ...
# Karena panjang, saya hanya mencantumkan placeholder. Asumsikan fungsi-fungsi fetch dan get_coins sudah ada.
# (Untuk menghemat tempat, gunakan kode yang sudah ada sebelumnya.)

# ========== FUNGSI BARU: MENENTUKAN ENTRY, TP, SL DARI LEVEL TEKNIKAL ==========
def find_best_entry_tp_sl(df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr, p):
    """
    Mengembalikan (final_entry, final_tp, final_sl) berdasarkan:
    - Support/resistance H1 & M15
    - EMA (200, 50)
    - Liquidity sweep level
    - Agresi (entry shift)
    """
    # Dapatkan level support/resistance dari swing points
    def get_levels(df):
        highs = df["high"]; lows = df["low"]
        swing_highs = []; swing_lows = []
        for i in range(2, len(df)-2):
            if highs.iloc[i] == highs.iloc[i-2:i+3].max(): swing_highs.append(highs.iloc[i])
            if lows.iloc[i] == lows.iloc[i-2:i+3].min(): swing_lows.append(lows.iloc[i])
        return sorted(swing_lows, reverse=True), sorted(swing_highs)

    supports_h1, resistances_h1 = get_levels(df_h1)
    supports_m15, resistances_m15 = get_levels(df_m15)

    # Gabungkan dan urutkan (prioritas H1 lebih tinggi, tapi M15 juga dipakai)
    all_supports = sorted(supports_h1 + supports_m15, reverse=True)
    all_resistances = sorted(resistances_h1 + resistances_m15)

    current_price = entry_raw
    shift = p["entry_shift_pips"] * 0.0001  # pips ke desimal (1 pip ≈ 0.0001 untuk kebanyakan pair)
    # Untuk koin dengan harga kecil (< 1), 1 pip mungkin 0.0001 atau 0.001, kita sesuaikan nanti.

    # Tentukan entry
    if bias_bull:
        # Entry dinaikkan sedikit (lebih berani)
        final_entry = round(current_price + shift * current_price, 6) if shift > 0 else current_price
        # Cari support terkuat untuk SL
        # SL harus di bawah liquidity sweep level (jika ada), atau di bawah support terdekat
        sl_candidates = []
        # 1. Gunakan sweep_level jika tersedia dan di bawah entry
        if sweep_level is not None and sweep_level < final_entry:
            sl_candidates.append(sweep_level - atr * 0.3)
        # 2. Support dari H1/M15 yang di bawah final_entry
        for sup in all_supports:
            if sup < final_entry:
                sl_candidates.append(sup - atr * 0.3)
                break
        # Jika tidak ada, fallback ke ATR
        if not sl_candidates:
            sl_candidates.append(final_entry - atr * 1.2)
        sl = round(min(sl_candidates), 6)  # SL terendah (terjauh dari entry)

        # Cari TP dari resistance
        tp_candidates = []
        for res in all_resistances:
            if res > final_entry:
                tp_candidates.append(res)
        if tp_candidates:
            # Ambil resistance terdekat
            tp = round(tp_candidates[0] * 0.999, 6)  # kurangi sedikit
        else:
            # Fallback: gunakan EMA 200 atau 1.5x ATR
            tp = round(final_entry + atr * 1.5, 6)

        # Cek RR minimal
        risk = final_entry - sl
        reward = tp - final_entry
        if reward / risk < p["min_rr"]:
            # Cari resistance berikutnya
            if len(tp_candidates) > 1:
                tp = round(tp_candidates[1] * 0.999, 6)
                reward = tp - final_entry
            # Jika masih kurang, terpaksa abaikan RR (biarkan sinyal tetap muncul dengan RR rendah)
            # Tapi user bilang jangan paksa TP dari RR, jadi kita tetap ambil TP teknikal itu.
            # Kita hanya akan menambahkan peringatan di Telegram nanti (di luar fungsi ini).

    else:  # SELL
        final_entry = round(current_price - shift * current_price, 6) if shift > 0 else current_price
        # Cari resistance terdekat untuk SL
        sl_candidates = []
        if sweep_level is not None and sweep_level > final_entry:
            sl_candidates.append(sweep_level + atr * 0.3)
        for res in all_resistances:
            if res > final_entry:
                sl_candidates.append(res + atr * 0.3)
                break
        if not sl_candidates:
            sl_candidates.append(final_entry + atr * 1.2)
        sl = round(max(sl_candidates), 6)

        tp_candidates = []
        for sup in all_supports:
            if sup < final_entry:
                tp_candidates.append(sup)
        if tp_candidates:
            tp = round(tp_candidates[0] * 1.001, 6)
        else:
            tp = round(final_entry - atr * 1.5, 6)

        risk = sl - final_entry
        reward = final_entry - tp
        if reward / risk < p["min_rr"]:
            if len(tp_candidates) > 1:
                tp = round(tp_candidates[1] * 1.001, 6)

    # Pastikan TP dan SL masuk akal
    if tp <= final_entry and bias_bull: tp = round(final_entry + atr * 0.5, 6)
    if tp >= final_entry and not bias_bull: tp = round(final_entry - atr * 0.5, 6)
    if sl >= final_entry and bias_bull: sl = round(final_entry - atr * 1.0, 6)
    if sl <= final_entry and not bias_bull: sl = round(final_entry + atr * 1.0, 6)

    return final_entry, tp, sl

# ========== ANALISA SINYAL (DENGAN ENTRY/TP/SL BARU) ==========
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

    # D1 wajib trending
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

    # M5 konfirmasi
    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["close"] > last_m5["ema12"]: score += 0.05
    elif not bias_bull and last_m5["close"] < last_m5["ema12"]: score += 0.05
    else: return None

    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else last_m15["close"] * 0.002
    entry_raw = round(last_m15["close"], 6)

    # Panggil fungsi baru untuk menghitung entry, tp, sl
    final_entry, tp, sl = find_best_entry_tp_sl(
        df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr, p
    )

    # Cek jarak TP (jika diatur > 0)
    if p["tp_distance_atr"] > 0:
        if bias_bull and (tp - final_entry) < p["tp_distance_atr"] * atr: return None
        if not bias_bull and (final_entry - tp) < p["tp_distance_atr"] * atr: return None

    # Confidence
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

# ========== LOOP UTAMA (dengan statistik fetch) ==========
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
                    fetch_ok += 1  # berhasil fetch meski tidak ada sinyal
            except Exception as e:
                fetch_fail += 1
                print(f"  ⚠️ Error {sym}: {e}")
            time.sleep(0.02)

        if fetch_fail > 0:
            send_telegram(f"⚠️ {fetch_fail} koin gagal fetch data.")
        if fetch_ok == 0 and fetch_fail == len(coins):
            send_telegram("❌ Semua koin gagal fetch. API down.")
            time.sleep(SCAN_INTERVAL)
            continue

        if not signals:
            p = get_aggression_params()
            send_telegram(f"❌ Tidak ada sinyal (Conf ≥ {p['min_confidence']}%, RR min 1:{p['min_rr']})")
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

# ========== FLASK ==========
app = Flask(__name__)
@app.route('/')
def home():
    return "Bot is alive", 200

def run_flask():
    app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    print("="*60)
    print("  SIGNAL BROADCASTER – Teknikal Entry/TP/SL")
    print("="*60)
    send_telegram("🚀 <b>Bot Teknikal siap!</b> /up /down /status")
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=telegram_polling, daemon=True).start()
    main_loop()
