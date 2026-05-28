#!/usr/bin/env python3
"""
SIMULATION SCALPING BOT – Improved Entry Logic
Timeout 5 menit, market order jika harga sudah lewat, halfway check tetap ada.
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
    "min_confidence": 30,
    "min_rr": 1.4,
    "entry_shift_pips": 4,
    "max_price": 100.0,
    "scan_interval": 2,
    "top_coins": 80,
    "ban_cycles": 10,
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
    except: pass

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
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get("retCode") == 0:
            return float(data["result"]["list"][0]["lastPrice"])
    except: pass
    return None

INTERVAL_MAP = {"1d":"D","4h":"240","1h":"60","15m":"15","5m":"5"}

def fetch_klines(symbol, interval, limit=120):
    bybit_interval = INTERVAL_MAP.get(interval, "60")
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={bybit_interval}&limit={limit}"
    for _ in range(2):
        try:
            time.sleep(0.3)
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
        except: time.sleep(3)
    return None

# ========== SMC INDICATORS ==========
def add_all_indicators(df):
    if len(df) < 50: return None
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["tr"] = np.maximum(df["high"] - df["low"],
                          np.maximum(abs(df["high"] - df["close"].shift()),
                                     abs(df["low"] - df["close"].shift())))
    df["atr"] = df["tr"].rolling(14).mean()
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
        for i in range(len(df)-4, 2, -1):
            if df["low"].iloc[i] == df["low"].iloc[i-2:i+3].min():
                if last["low"] < df["low"].iloc[i] and last["close"] > df["low"].iloc[i]:
                    return True, df["low"].iloc[i]
    else:
        for i in range(len(df)-4, 2, -1):
            if df["high"].iloc[i] == df["high"].iloc[i-2:i+3].max():
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
        for i in range(last_idx-1, max(last_idx-15, 0), -1):
            if df["close"].iloc[i] < df["open"].iloc[i]:
                if i+1 <= last_idx and df["close"].iloc[i+1] > df["open"].iloc[i+1] and last_close > df["high"].iloc[i]:
                    return df["high"].iloc[i], df["low"].iloc[i]
    else:
        for i in range(last_idx-1, max(last_idx-15, 0), -1):
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
    df_h4 = fetch_klines(symbol,"4h",100)
    df_h1 = fetch_klines(symbol,"1h",100)
    df_m15 = fetch_klines(symbol,"15m",100)
    df_m5 = fetch_klines(symbol,"5m",100)
    if any(d is None for d in [df_h4, df_h1, df_m15, df_m5]): return None

    df_h4 = add_all_indicators(df_h4); df_h1 = add_all_indicators(df_h1)
    df_m15 = add_all_indicators(df_m15); df_m5 = add_all_indicators(df_m5)
    if any(d is None for d in [df_h4, df_h1, df_m15, df_m5]): return None

    # 1. H4 trend (wajib)
    struct_h4 = market_structure(df_h4, 5)
    if struct_h4 == "ranging": return None
    bias_bull = struct_h4 == "bullish"
    direction = "BUY" if bias_bull else "SELL"
    score = 50

    # 2. Sweep (wajib)
    sweep_m15, sweep_level_m15 = detect_liquidity_sweep(df_m15, "buy" if bias_bull else "sell")
    sweep_h4, _ = detect_liquidity_sweep(df_h4, "buy" if bias_bull else "sell")
    if not (sweep_m15 or sweep_h4): return None
    if sweep_m15: score += 20
    else: score += 10

    # 3. FVG/OB H4 sebagai zona
    ob_h4 = find_order_block(df_h4, "buy" if bias_bull else "sell")
    fvg_h4 = find_fvg(df_h4, "buy" if bias_bull else "sell")
    zone_high = None; zone_low = None
    if ob_h4:
        zone_high, zone_low = ob_h4
    elif fvg_h4:
        zone_high, zone_low = fvg_h4

    # 4. Filter EMA50 M15
    last_m15 = df_m15.iloc[-2]
    if bias_bull and last_m15["close"] <= last_m15["ema50"]: return None
    if not bias_bull and last_m15["close"] >= last_m15["ema50"]: return None

    # 5. M15 momentum
    if bias_bull and last_m15["ema12"] > last_m15["ema26"]: score += 10
    elif not bias_bull and last_m15["ema12"] < last_m15["ema26"]: score += 10

    # 6. M5 konfirmasi
    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["close"] > last_m5["ema12"]: score += 5
    elif not bias_bull and last_m5["close"] < last_m5["ema12"]: score += 5

    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else last_m15["close"] * 0.002
    entry_raw = round(last_m15["close"], 6)

    shift_pct = settings["entry_shift_pips"] * 0.0001
    shift = shift_pct * entry_raw if settings["entry_shift_pips"] > 0 else 0

    if bias_bull:
        if zone_low is not None:
            final_entry = round(zone_low + atr * 0.3 + shift, 6)
            sl = round(zone_low - atr * 0.7, 6)
        else:
            final_entry = round(entry_raw + shift, 6)
            sl = round(final_entry - atr * 1.5, 6)

        _, resistances_m15 = get_levels(df_m15)
        tp_cand = [r for r in resistances_m15 if r > final_entry]
        if tp_cand:
            tp = round(min(tp_cand) * 1.001, 6)
        else:
            tp = round(final_entry + atr * 2.5, 6)
    else:
        if zone_high is not None:
            final_entry = round(zone_high - atr * 0.3 - shift, 6)
            sl = round(zone_high + atr * 0.7, 6)
        else:
            final_entry = round(entry_raw - shift, 6)
            sl = round(final_entry + atr * 1.5, 6)

        supports_m15, _ = get_levels(df_m15)
        tp_cand = [s for s in supports_m15 if s < final_entry]
        if tp_cand:
            tp = round(max(tp_cand) * 0.999, 6)
        else:
            tp = round(final_entry - atr * 2.5, 6)

    risk = abs(final_entry - sl)
    reward = abs(tp - final_entry)
    if risk <= 0 or reward / risk < settings["min_rr"]:
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
        "rr": round(reward/risk, 2),
    }

# ========== SIMULATION ENGINE (IMPROVED ENTRY) ==========
def simulate_trade(sig):
    symbol = sig["symbol"]
    direction = sig["signal"]
    entry = sig["entry"]
    tp = sig["tp"]
    sl = sig["sl"]
    confidence = sig["confidence"]
    rr = sig["rr"]

    msg = (
        f"<b>📊 {direction} {symbol} (SIM)</b>\n"
        f"Entry: {entry}\nTP: {tp} | SL: {sl}\n"
        f"Conf: {confidence}% | RR: 1:{rr} | ATR: {sig['atr']}"
    )
    log_activity(msg)

    halfway = entry + (tp - entry) / 2 if direction == "BUY" else entry - (entry - tp) / 2
    log_activity(f"⏳ Memantau {symbol} (halfway: {halfway:.6f})...")

    # Cek harga saat ini
    current = get_mark_price(symbol)
    if current is None:
        log_activity("❌ Gagal ambil harga, skip.")
        return "NO_ENTRY"

    # Jika harga sudah di bawah entry (BUY) atau di atas entry (SELL), langsung eksekusi di harga saat ini
    if direction == "BUY" and current < entry:
        log_activity(f"⚡ Harga sudah di bawah entry ({current:.6f}), eksekusi market order simulasi.")
        entry = current
    elif direction == "SELL" and current > entry:
        log_activity(f"⚡ Harga sudah di atas entry ({current:.6f}), eksekusi market order simulasi.")
        entry = current
    else:
        # Fase 1: tunggu entry (timeout 5 menit)
        start_time = time.time()
        last_notify = time.time()
        entry_hit = False
        while time.time() - start_time < 300:
            price = get_mark_price(symbol)
            if price is None:
                time.sleep(1)
                continue

            if time.time() - last_notify >= 300:
                log_activity(f"⏳ {symbol} | Price: {price:.6f} | Entry: {entry:.6f} | Halfway: {halfway:.6f}")
                last_notify = time.time()

            # Check halfway
            if direction == "BUY" and price >= halfway:
                log_activity(f"⚠️ Halfway tercapai sebelum entry. No Entry.")
                settings["no_entry"] += 1
                return "NO_ENTRY"
            if direction == "SELL" and price <= halfway:
                log_activity(f"⚠️ Halfway tercapai sebelum entry. No Entry.")
                settings["no_entry"] += 1
                return "NO_ENTRY"

            # Check entry
            if direction == "BUY" and price <= entry:
                entry = price
                entry_hit = True
                log_activity(f"✅ Entry di {price}")
                break
            if direction == "SELL" and price >= entry:
                entry = price
                entry_hit = True
                log_activity(f"✅ Entry di {price}")
                break
            time.sleep(1)

        if not entry_hit:
            log_activity(f"⏰ Timeout 5 menit. No Entry.")
            settings["no_entry"] += 1
            return "NO_ENTRY"

    # Fase 2: pantau TP/SL
    log_activity(f"🔄 Memantau TP/SL (Entry: {entry})...")
    last_notify = time.time()
    while True:
        price = get_mark_price(symbol)
        if price is None:
            time.sleep(1)
            continue

        if time.time() - last_notify >= 300:
            log_activity(f"⏳ {symbol} | Price: {price:.6f} | TP: {tp:.6f} | SL: {sl:.6f}")
            last_notify = time.time()

        if direction == "BUY":
            if price <= sl:
                log_activity(f"❌ SL di {price}")
                settings["sl"] += 1
                return "SL"
            if price >= tp:
                log_activity(f"🏆 TP di {price}")
                settings["tp"] += 1
                return "TP"
        else:
            if price >= sl:
                log_activity(f"❌ SL di {price}")
                settings["sl"] += 1
                return "SL"
            if price <= tp:
                log_activity(f"🏆 TP di {price}")
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
        log_activity("😴 Tidak ada koin.")
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
        log_activity(f"🏆 {best['signal']} {best['symbol']} (Conf: {best['confidence']}%)")
        return [best]
    return []

# ========== MAIN LOOP ==========
def main_loop():
    global bot_running
    log_activity("🔄 Bot simulasi (improved entry) mulai...")
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
                    total = settings["tp"] + settings["sl"] + settings["no_entry"]
                    wr = f"{(settings['tp']/total*100):.1f}%" if total > 0 else "N/A"
                    send_telegram(
                        f"📊 Bot: {'Running' if bot_running else 'Stopped'}\n"
                        f"TP: {settings['tp']} | SL: {settings['sl']} | No Entry: {settings['no_entry']}\n"
                        f"Total: {total} | WR: {wr}"
                    )
                elif text == "/menu":
                    send_telegram("""<b>Command List:</b>
/start - Mulai bot
/stop - Hentikan bot
/status - Statistik TP/SL/No Entry & Winrate
/settings - Lihat pengaturan
/set key value - Ubah pengaturan
/menu - Tampilkan menu""")
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}"); time.sleep(5)

# ========== STARTUP ==========
if __name__ == "__main__":
    log_activity("🤖 Bot simulasi improved starting...")
    try:
        ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
        log_activity(f"🚀 IP: {ip}")
    except:
        log_activity("🚀 Bot dimulai!")
    threading.Thread(target=telegram_polling, daemon=True).start()
    threading.Thread(target=main_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
