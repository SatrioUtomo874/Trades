#!/usr/bin/env python3
"""
SIMULATION TRADING BOT – FVG/SMC Logic + Real‑time Monitoring
Notifikasi Telegram setiap 5 menit saat memantau harga.
No RSI, no API keys. Uses Bybit public data.
Stats tracking (TP/SL/NO_ENTRY), adjustable via /set.
Runs 24/7 on Render.
"""

import os
import time
import threading
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask

# ================== CONFIG ==================
TELEGRAM_TOKEN = "8094484109:AAF9Z3lQUxdQFqqeG6NKV9O1EC0vrxzJy0U"
CHAT_ID = "8041197505"

settings = {
    "min_confidence": 40,
    "min_rr": 1.6,
    "entry_shift_pips": 5,
    "max_price": 50.0,
    "scan_interval": 3,
    "top_coins": 50,
    "ban_cycles": 20,
    # stats
    "tp": 0,
    "sl": 0,
    "no_entry": 0,
}

banned = {}
perma_banned = set()
bot_running = True

app = Flask(__name__)
@app.route('/')
def home():
    return "Bot is alive", 200

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except:
        pass

def log_activity(msg):
    send_telegram(f"📋 {msg}")

# ========== BYBIT PUBLIC API ==========
def get_coins_bybit():
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        r = requests.get(url, timeout=15)
        if r.status_code != 200: return None
        data = r.json()
        if data.get("retCode") != 0: return None
        tickers = data["result"]["list"]
        filtered = []
        for t in tickers:
            sym = t["symbol"]
            if sym in perma_banned or not sym.endswith("USDT"): continue
            try:
                price = float(t["lastPrice"])
                if price <= settings["max_price"]:
                    filtered.append((sym, float(t.get("turnover24h", 0))))
            except: pass
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in filtered[:int(settings["top_coins"])]]
    except: return None

FALLBACK_COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
    "AVAXUSDT","DOTUSDT","LINKUSDT","MATICUSDT","UNIUSDT","ATOMUSDT","LTCUSDT",
    "ETCUSDT","OPUSDT","ARBUSDT","INJUSDT","TIAUSDT","SUIUSDT","SEIUSDT",
    "NEARUSDT","APTUSDT","RNDRUSDT","FETUSDT","AGIXUSDT","OCEANUSDT","GRTUSDT",
    "THETAUSDT","SANDUSDT","MANAUSDT","GALAUSDT","AXSUSDT","CHZUSDT","FLOWUSDT",
    "EGLDUSDT","QNTUSDT","SNXUSDT","CRVUSDT","COMPUSDT","AAVEUSDT","MKRUSDT",
    "RUNEUSDT","LDOUSDT","FXSUSDT","1INCHUSDT","ZRXUSDT","BATUSDT","ENJUSDT","ANKRUSDT"
]

def get_coins():
    coins = get_coins_bybit()
    if coins: return coins
    log_activity("🔄 Fallback ke daftar koin statis...")
    return [c for c in FALLBACK_COINS if c not in perma_banned][:int(settings["top_coins"])]

def get_mark_price(symbol):
    """Ambil harga mark dari Bybit public API."""
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get("retCode") == 0:
            return float(data["result"]["list"][0]["lastPrice"])
    except: pass
    return None

INTERVAL_MAP = {"1d":"D","4h":"240","1h":"60","15m":"15","5m":"5"}

def fetch_klines(symbol, interval, limit=200):
    bybit_interval = INTERVAL_MAP.get(interval, "60")
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={bybit_interval}&limit={limit}"
    for _ in range(3):
        try:
            time.sleep(0.5)
            r = requests.get(url, timeout=15)
            if r.status_code != 200: continue
            data = r.json()
            if data.get("retCode") != 0: continue
            rows = data["result"]["list"]
            if not rows: continue
            rows = rows[::-1]
            df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
            df.set_index("timestamp", inplace=True)
            return df[["open","high","low","close","volume"]]
        except: time.sleep(5)
    return None

# ========== SMC INDICATORS (NO RSI) ==========
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
    df["vol_avg20"] = df["volume"].rolling(20).mean()
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

def get_levels(df):
    highs, lows = df["high"], df["low"]
    sh, sl = [], []
    for i in range(2, len(df)-2):
        if highs.iloc[i] == highs.iloc[i-2:i+3].max(): sh.append(highs.iloc[i])
        if lows.iloc[i] == lows.iloc[i-2:i+3].min(): sl.append(lows.iloc[i])
    return sorted(sl, reverse=True), sorted(sh)

def analyze_signal(symbol):
    df_d1 = fetch_klines(symbol,"1d",200); df_h4 = fetch_klines(symbol,"4h",200)
    df_h1 = fetch_klines(symbol,"1h",150); df_m15 = fetch_klines(symbol,"15m",150); df_m5 = fetch_klines(symbol,"5m",150)
    if any(d is None for d in [df_d1,df_h4,df_h1,df_m15,df_m5]): return None

    df_d1 = add_all_indicators(df_d1); df_h4 = add_all_indicators(df_h4)
    df_h1 = add_all_indicators(df_h1); df_m15 = add_all_indicators(df_m15); df_m5 = add_all_indicators(df_m5)
    if any(d is None for d in [df_d1,df_h4,df_h1,df_m15,df_m5]): return None

    struct_d1 = market_structure(df_d1, 5)
    if struct_d1 == "ranging": return None
    bias_bull = struct_d1 == "bullish"
    direction = "BUY" if bias_bull else "SELL"
    score = 0

    last_d1 = df_d1.iloc[-1]
    if bias_bull and last_d1["close"] > last_d1["ema50"]: score += 10
    elif not bias_bull and last_d1["close"] < last_d1["ema50"]: score += 10

    struct_h4 = market_structure(df_h4, 3)
    if struct_h4 == struct_d1: score += 10
    sweep_h4, _ = detect_liquidity_sweep(df_h4, "buy" if bias_bull else "sell")
    if sweep_h4: score += 10

    struct_h1 = market_structure(df_h1, 2)
    if struct_h1 == struct_d1: score += 10

    last_m15 = df_m15.iloc[-2]
    if bias_bull and last_m15["ema12"] > last_m15["ema26"]: score += 10
    elif not bias_bull and last_m15["ema12"] < last_m15["ema26"]: score += 10
    else: return None

    sweep_m15, sweep_level_m15 = detect_liquidity_sweep(df_m15, "buy" if bias_bull else "sell")
    if not (sweep_m15 or sweep_h4): return None
    if sweep_m15: score += 15
    else: score += 10

    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["close"] > last_m5["ema12"]: score += 5
    elif not bias_bull and last_m5["close"] < last_m5["ema12"]: score += 5
    else: return None

    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else last_m15["close"] * 0.002
    entry_raw = round(last_m15["close"], 6)

    # ---- Entry/TP/SL based on H4 FVG/OB + M15 support ----
    ob_h4 = find_order_block(df_h4, "buy" if bias_bull else "sell")
    fvg_h4 = find_fvg(df_h4, "buy" if bias_bull else "sell")
    zone_high = None; zone_low = None
    if ob_h4:
        zone_high, zone_low = ob_h4
    elif fvg_h4:
        zone_high, zone_low = fvg_h4

    supports_m15, resistances_m15 = get_levels(df_m15)
    supports_m5, resistances_m5 = get_levels(df_m5)
    all_supports = sorted(supports_m15 + supports_m5, reverse=True)
    all_resistances = sorted(resistances_m15 + resistances_m5)

    shift_pct = settings["entry_shift_pips"] * 0.0001
    shift = shift_pct * entry_raw if settings["entry_shift_pips"] > 0 else 0

    if bias_bull:
        sup_cand = [s for s in all_supports if s < entry_raw]
        if zone_low is not None:
            sup_cand = [s for s in sup_cand if s >= zone_low]
        best_support = max(sup_cand) if sup_cand else (zone_low if zone_low else entry_raw - atr)
        final_entry = round(best_support + atr * 0.2 + shift, 6)

        sl_candidates = []
        if sweep_level_m15: sl_candidates.append(sweep_level_m15)
        if zone_low: sl_candidates.append(zone_low)
        if sup_cand: sl_candidates.append(min(sup_cand))
        sl_base = min(sl_candidates) if sl_candidates else (final_entry - atr * 1.5)
        sl = round(sl_base - atr * 0.5, 6)

        res_cand = [r for r in all_resistances if r > final_entry]
        tp = round(min(res_cand) * 0.999, 6) if res_cand else round(final_entry + atr * 2.0, 6)

    else:
        res_cand = [r for r in all_resistances if r > entry_raw]
        if zone_high is not None:
            res_cand = [r for r in res_cand if r <= zone_high]
        best_resistance = min(res_cand) if res_cand else (zone_high if zone_high else entry_raw + atr)
        final_entry = round(best_resistance - atr * 0.2 - shift, 6)

        sl_candidates = []
        if sweep_level_m15: sl_candidates.append(sweep_level_m15)
        if zone_high: sl_candidates.append(zone_high)
        if res_cand: sl_candidates.append(max(res_cand))
        sl_base = max(sl_candidates) if sl_candidates else (final_entry + atr * 1.5)
        sl = round(sl_base + atr * 0.5, 6)

        sup_cand = [s for s in all_supports if s < final_entry]
        tp = round(max(sup_cand) * 1.001, 6) if sup_cand else round(final_entry - atr * 2.0, 6)

    risk = abs(final_entry - sl)
    reward = abs(tp - final_entry)
    if risk > 0 and reward / risk < settings["min_rr"]:
        return None

    confidence = min(score, 100)
    if confidence < settings["min_confidence"]: return None

    return {
        "symbol": symbol,
        "signal": direction,
        "entry": final_entry,
        "tp": tp,
        "sl": sl,
        "confidence": confidence,
        "atr": round(atr, 6),
        "rr": round(reward/risk, 2) if risk > 0 else 0,
    }

# ========== SIMULATION ENGINE (WITH 5-MIN NOTIFICATIONS) ==========
def simulate_trade(sig):
    symbol = sig["symbol"]
    direction = sig["signal"]
    entry = sig["entry"]
    tp = sig["tp"]
    sl = sig["sl"]
    confidence = sig["confidence"]
    rr = sig["rr"]

    msg = (
        f"<b>📊 {direction} {symbol} (SIMULATION)</b>\n"
        f"Entry: {entry}\nTP: {tp} | SL: {sl}\n"
        f"Conf: {confidence}% | RR: 1:{rr} | ATR: {sig['atr']}"
    )
    log_activity(msg)

    if direction == "BUY":
        halfway = entry + (tp - entry) / 2
    else:
        halfway = entry - (entry - tp) / 2

    log_activity(f"⏳ Memantau {symbol} (halfway: {halfway:.6f})...")

    # ---- FASE 1: TUNGGU ENTRY ----
    start_time = time.time()
    last_notify = time.time()
    entry_hit = False
    while time.time() - start_time < 300:  # timeout 5 menit
        price = get_mark_price(symbol)
        if price is None:
            time.sleep(1)
            continue

        # Notifikasi tiap 5 menit
        if time.time() - last_notify >= 300:
            log_activity(f"⏳ {symbol} | Price: {price:.6f} | Entry: {entry:.6f} | Halfway: {halfway:.6f}")
            last_notify = time.time()

        # Check halfway BEFORE entry
        if direction == "BUY" and price >= halfway:
            log_activity(f"⚠️ Harga mencapai halfway ({halfway}) sebelum entry. No Entry.")
            settings["no_entry"] += 1
            return "NO_ENTRY"
        if direction == "SELL" and price <= halfway:
            log_activity(f"⚠️ Harga mencapai halfway ({halfway}) sebelum entry. No Entry.")
            settings["no_entry"] += 1
            return "NO_ENTRY"

        # Check entry
        if direction == "BUY" and price <= entry:
            entry_hit = True
            log_activity(f"✅ Entry {symbol} tercapai di {price}")
            break
        if direction == "SELL" and price >= entry:
            entry_hit = True
            log_activity(f"✅ Entry {symbol} tercapai di {price}")
            break

        time.sleep(1)

    if not entry_hit:
        log_activity(f"⏰ Timeout 5 menit, harga tidak menyentuh entry. No Entry.")
        settings["no_entry"] += 1
        return "NO_ENTRY"

    # ---- FASE 2: PANTAU TP/SL ----
    log_activity(f"🔄 Memantau TP/SL {symbol}...")
    last_notify = time.time()
    while True:
        price = get_mark_price(symbol)
        if price is None:
            time.sleep(1)
            continue

        # Notifikasi tiap 5 menit
        if time.time() - last_notify >= 300:
            log_activity(f"⏳ {symbol} | Price: {price:.6f} | TP: {tp:.6f} | SL: {sl:.6f}")
            last_notify = time.time()

        if direction == "BUY":
            if price <= sl:
                log_activity(f"❌ Stop Loss tercapai di {price}")
                settings["sl"] += 1
                return "SL"
            if price >= tp:
                log_activity(f"🏆 Take Profit tercapai di {price}")
                settings["tp"] += 1
                return "TP"
        else:
            if price >= sl:
                log_activity(f"❌ Stop Loss tercapai di {price}")
                settings["sl"] += 1
                return "SL"
            if price <= tp:
                log_activity(f"🏆 Take Profit tercapai di {price}")
                settings["tp"] += 1
                return "TP"
        time.sleep(1)

# ========== SCANNING ==========
def update_banned():
    to_del = [k for k,v in banned.items() if v<=0]
    for k in to_del: del banned[k]
    for k in list(banned.keys()): banned[k] -= 1

def scan_signals():
    if not bot_running: return []
    update_banned()
    coins = get_coins()
    if not coins:
        log_activity("😴 Tidak ada koin, jeda 30 detik...")
        time.sleep(30)
        return []
    log_activity(f"🔍 Scanning {len(coins)} koin...")
    signals = []
    for sym in coins:
        if not bot_running: break
        if sym in banned or sym in perma_banned: continue
        try:
            sig = analyze_signal(sym)
            if sig:
                sig["symbol"] = sym
                signals.append(sig)
        except: pass
        time.sleep(settings["scan_interval"])
    if signals and bot_running:
        best = max(signals, key=lambda x: x["confidence"])
        log_activity(f"🏆 Sinyal terbaik: {best['signal']} {best['symbol']} (Conf: {best['confidence']}%)")
        return [best]
    return []

# ========== MAIN LOOP ==========
def main_loop():
    global bot_running
    log_activity("🔄 Bot simulasi mulai...")
    while True:
        if not bot_running:
            time.sleep(2)
            continue
        try:
            signals = scan_signals()
            if signals:
                sig = signals[0]
                simulate_trade(sig)
                banned[sig["symbol"]] = settings["ban_cycles"]
            else:
                time.sleep(5)
        except Exception as e:
            log_activity(f"⚠️ Error: {e}")
            time.sleep(10)

# ========== TELEGRAM POLLING ==========
def telegram_polling():
    global bot_running
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
                if text == "/stop":
                    bot_running = False
                    send_telegram("⏹️ Bot dihentikan.")
                elif text == "/start":
                    bot_running = True
                    send_telegram("▶️ Bot dimulai.")
                elif text.startswith("/set "):
                    parts = text.split()
                    if len(parts) == 3:
                        key = parts[1]
                        try:
                            value = float(parts[2])
                            if key in settings:
                                settings[key] = value
                                send_telegram(f"⚙️ {key} diset ke {value}")
                            else:
                                send_telegram(f"❌ Key tidak dikenal: {key}")
                        except ValueError:
                            send_telegram("❌ Value harus berupa angka.")
                elif text == "/settings":
                    send_telegram(f"<pre>{json.dumps(settings, indent=2)}</pre>")
                elif text == "/status":
                    send_telegram(
                        f"📊 Bot: {'Running' if bot_running else 'Stopped'}\n"
                        f"TP: {settings['tp']} | SL: {settings['sl']} | No Entry: {settings['no_entry']}"
                    )
                elif text == "/menu":
                    send_telegram("""<b>Command List:</b>
/start - Mulai bot
/stop - Hentikan bot
/status - Statistik TP/SL/No Entry
/settings - Lihat pengaturan
/set key value - Ubah pengaturan
/menu - Tampilkan menu""")
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}"); time.sleep(5)

# ========== STARTUP ==========
if __name__ == "__main__":
    log_activity("🤖 Bot simulasi starting...")
    try:
        ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
        log_activity(f"🚀 Bot dimulai!\nIP: {ip}")
    except:
        log_activity("🚀 Bot dimulai! (IP tidak terdeteksi)")
    threading.Thread(target=telegram_polling, daemon=True).start()
    threading.Thread(target=main_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
