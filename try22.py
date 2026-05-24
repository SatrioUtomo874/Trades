#!/usr/bin/env python3
"""
FULL AUTO TRADING BOT - Binance Futures
- Manajemen posisi & TP/SL teknis (H1/H4)
- Sinyal teknikal dengan FVG/OB, bonus poin, RR wajib
- Batas posisi maks, auto quantity, margin
- Order tracking, pembersihan orphan orders
- Konfigurasi via command Telegram (/set key value, /settings)
- Ban permanen via /ban <SYMBOL>, /unban <SYMBOL>
- Menu bantuan via /menu
- Scanning terus-menerus hingga sinyal ditemukan
- Limit order dibatalkan jika harga sudah setengah jalan ke TP
"""

import os
import time
import hmac
import hashlib
import math
import threading
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask

# ================== KONFIGURASI DEFAULT ==================
TELEGRAM_TOKEN = "8094484109:AAF9Z3lQUxdQFqqeG6NKV9O1EC0vrxzJy0U"
CHAT_ID = "8041197505"

settings = {
    "max_positions": 5,
    "leverage": 5,
    "min_order_usd": 1.0,
    "max_price": 100.0,
    "min_confidence": 55,
    "ban_cycles": 20,
    "scan_interval": 10,   # detik antar siklus jika tidak ada sinyal
    "top_coins": 80,
}

API_KEY = os.environ.get("BINANCE_API_KEY", "")
SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")

# ---------- STATE ----------
banned = {}               # sementara (siklus)
perma_banned = set()      # permanen
tracking_orders = {}       # order_id -> info

app = Flask(__name__)
@app.route('/')
def home():
    return "Bot is alive", 200

# ---------- TELEGRAM ----------
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except:
        pass

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
                                send_telegram(f"❌ Key tidak dikenal: {key}")
                        except ValueError:
                            send_telegram("❌ Value harus berupa angka.")
                elif text == "/settings":
                    s = json.dumps(settings, indent=2)
                    send_telegram(f"<pre>{s}</pre>")
                elif text == "/status":
                    pos_count = len(get_open_positions())
                    orders = len(get_open_orders())
                    send_telegram(f"📊 Posisi: {pos_count} | Limit Orders: {orders} | Max: {settings['max_positions']}")
                elif text.startswith("/ban "):
                    symbol = text.split()[1].upper()
                    if not symbol.endswith("USDT"):
                        symbol += "USDT"
                    perma_banned.add(symbol)
                    send_telegram(f"🚫 {symbol} dibanned permanen.")
                elif text.startswith("/unban "):
                    symbol = text.split()[1].upper()
                    if not symbol.endswith("USDT"):
                        symbol += "USDT"
                    perma_banned.discard(symbol)
                    send_telegram(f"✅ {symbol} dihapus dari banned permanen.")
                elif text == "/menu":
                    help_text = """
<b>Command List:</b>
/status - Posisi & order aktif
/settings - Lihat pengaturan
/set key value - Ubah pengaturan (max_positions, leverage, min_order_usd, max_price, min_confidence, ban_cycles, scan_interval, top_coins)
/ban SYMBOL - Ban permanen koin (contoh: /ban BTCUSDT)
/unban SYMBOL - Hapus ban permanen
/menu - Tampilkan menu ini
"""
                    send_telegram(help_text)
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(5)

# ---------- BINANCE SIGNED REQUEST ----------
def signed_request(endpoint, params=None, method="GET"):
    if params is None:
        params = {}
    params["timestamp"] = int(time.time() * 1000)
    query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    signature = hmac.new(SECRET_KEY.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    url = f"https://fapi.binance.com{endpoint}?{query_string}&signature={signature}"

    try:
        if method == "GET":
            r = requests.get(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
        elif method == "POST":
            r = requests.post(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
        elif method == "DELETE":
            r = requests.delete(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
        else:
            return None
        if r.status_code == 200:
            return r.json()
        else:
            try:
                err = r.json()
                send_telegram(f"❌ API Error {r.status_code}: {err.get('msg', '')}")
            except:
                send_telegram(f"❌ API Error {r.status_code}: {r.text}")
            return None
    except Exception as e:
        send_telegram(f"⚠️ Network error: {e}")
        return None

# ---------- BINANCE HELPERS ----------
def get_open_positions():
    data = signed_request("/fapi/v2/positionRisk", method="GET")
    if not data:
        return []
    positions = []
    for p in data:
        if float(p["positionAmt"]) != 0:
            positions.append({
                "symbol": p["symbol"],
                "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
                "amount": abs(float(p["positionAmt"])),
                "entryPrice": float(p["entryPrice"]),
                "unrealizedProfit": float(p["unRealizedProfit"]),
                "leverage": int(p["leverage"]),
            })
    return positions

def get_open_orders():
    all_orders = []
    limit = signed_request("/fapi/v1/openOrders", method="GET")
    if limit and isinstance(limit, list):
        all_orders.extend(limit)
    algo = signed_request("/fapi/v1/algoOpenOrders", method="GET")
    if algo and isinstance(algo, list):
        all_orders.extend(algo)
    return all_orders

def cancel_order(symbol, order_id):
    res = signed_request("/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, method="DELETE")
    if res and res.get("status") == "CANCELED":
        return True
    res = signed_request("/fapi/v1/algoOrder", {"symbol": symbol, "algoId": order_id}, method="DELETE")
    return res is not None

def place_limit_order(symbol, side, quantity, price):
    params = {
        "symbol": symbol,
        "side": side,
        "type": "LIMIT",
        "quantity": quantity,
        "price": price,
        "timeInForce": "GTC"
    }
    res = signed_request("/fapi/v1/order", params, method="POST")
    if res and "orderId" in res:
        return res["orderId"]
    return None

def place_market_order(symbol, side, quantity):
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
    }
    res = signed_request("/fapi/v1/order", params, method="POST")
    if res and "orderId" in res:
        return res["orderId"]
    return None

def place_tp_sl_for_position(symbol, side, quantity, stop_loss_price, take_profit_price):
    if side == "LONG":
        tp_side = "SELL"
        sl_side = "SELL"
    else:
        tp_side = "BUY"
        sl_side = "BUY"

    tp_res = signed_request("/fapi/v1/algoOrder", {
        "algoType": "CONDITIONAL",
        "symbol": symbol,
        "side": tp_side,
        "type": "TAKE_PROFIT_MARKET",
        "quantity": quantity,
        "triggerPrice": take_profit_price,
        "workingType": "MARK_PRICE"
    }, method="POST")

    sl_res = signed_request("/fapi/v1/algoOrder", {
        "algoType": "CONDITIONAL",
        "symbol": symbol,
        "side": sl_side,
        "type": "STOP_MARKET",
        "quantity": quantity,
        "triggerPrice": stop_loss_price,
        "workingType": "MARK_PRICE"
    }, method="POST")

    return tp_res is not None and sl_res is not None

def set_leverage(symbol, leverage):
    signed_request("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, method="POST")

def get_mark_price(symbol):
    try:
        resp = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}", timeout=5)
        return float(resp.json()["price"])
    except:
        return None

def get_symbol_filters(symbol):
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        for s in resp.json()["symbols"]:
            if s["symbol"] == symbol:
                filters = {}
                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        filters["tickSize"] = float(f["tickSize"])
                    elif f["filterType"] == "LOT_SIZE":
                        filters["stepSize"] = float(f["stepSize"])
                        filters["minQty"] = float(f["minQty"])
                    elif f["filterType"] == "MIN_NOTIONAL":
                        filters["minNotional"] = float(f["notional"])
                return filters
    except:
        pass
    return {"tickSize": 0.01, "stepSize": 0.001, "minQty": 0.001, "minNotional": 5.0}

def round_to_tick(value, tick):
    if tick == 0:
        return value
    prec = int(round(-math.log10(tick), 0))
    return round(round(value / tick) * tick, prec)

def round_to_step(value, step):
    if step == 0:
        return value
    prec = int(round(-math.log10(step), 0))
    return round(round(value / step) * step, prec)

def format_price(value, tick):
    if tick == 0:
        return str(value)
    prec = int(round(-math.log10(tick), 0))
    return f"{value:.{prec}f}"

def format_qty(value, step):
    if step == 0:
        return str(value)
    prec = int(round(-math.log10(step), 0))
    return f"{value:.{prec}f}"

# ---------- INDIKATOR & SMC (FULL) ----------
def fetch_klines(symbol, interval, limit=200):
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
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_diff"] = df["macd"] - df["macd_signal"]
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
    hh = sh[-1] > sh[-2]; hl = sl[-1] > sl[-2]
    lh = sh[-1] < sh[-2]; ll = sl[-1] < sl[-2]
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
    if last_idx < 3:
        return None
    if direction == "buy":
        if df["low"].iloc[last_idx] > df["high"].iloc[last_idx-2]:
            return df["high"].iloc[last_idx-2], df["low"].iloc[last_idx]
    else:
        if df["high"].iloc[last_idx] < df["low"].iloc[last_idx-2]:
            return df["low"].iloc[last_idx-2], df["high"].iloc[last_idx]
    return None

def find_order_block(df, direction):
    last_idx = len(df) - 2
    last_close = df["close"].iloc[last_idx]
    if direction == "buy":
        for i in range(last_idx-1, max(last_idx-20, 0), -1):
            if df["close"].iloc[i] < df["open"].iloc[i]:
                if i+1 <= last_idx and df["close"].iloc[i+1] > df["open"].iloc[i+1]:
                    if last_close > df["high"].iloc[i]:
                        return df["high"].iloc[i], df["low"].iloc[i]
    else:
        for i in range(last_idx-1, max(last_idx-20, 0), -1):
            if df["close"].iloc[i] > df["open"].iloc[i]:
                if i+1 <= last_idx and df["close"].iloc[i+1] < df["open"].iloc[i+1]:
                    if last_close < df["low"].iloc[i]:
                        return df["high"].iloc[i], df["low"].iloc[i]
    return None

def get_levels(df):
    highs, lows = df["high"], df["low"]
    sh, sl = [], []
    for i in range(2, len(df)-2):
        if highs.iloc[i] == highs.iloc[i-2:i+3].max(): sh.append(highs.iloc[i])
        if lows.iloc[i] == lows.iloc[i-2:i+3].min(): sl.append(lows.iloc[i])
    return sorted(sl, reverse=True), sorted(sh)

def has_bullish_confirmation(df):
    last = df.iloc[-2]; prev = df.iloc[-3]
    if prev["close"] < prev["open"] and last["close"] > last["open"]:
        if last["close"] > prev["open"] and last["open"] < prev["close"]:
            return True
    body = abs(last["close"] - last["open"])
    lower_shadow = min(last["open"], last["close"]) - last["low"]
    upper_shadow = last["high"] - max(last["open"], last["close"])
    if lower_shadow > body * 1.5 and upper_shadow < body * 0.5:
        return True
    return False

def has_bearish_confirmation(df):
    last = df.iloc[-2]; prev = df.iloc[-3]
    if prev["close"] > prev["open"] and last["close"] < last["open"]:
        if last["open"] > prev["close"] and last["close"] < prev["open"]:
            return True
    body = abs(last["close"] - last["open"])
    upper_shadow = last["high"] - max(last["open"], last["close"])
    lower_shadow = min(last["open"], last["close"]) - last["low"]
    if upper_shadow > body * 1.5 and lower_shadow < body * 0.5:
        return True
    return False

def find_best_entry_tp_sl(df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr):
    supports_h1, resistances_h1 = get_levels(df_h1)
    supports_m15, resistances_m15 = get_levels(df_m15)
    all_supports = sorted(supports_h1 + supports_m15, reverse=True)
    all_resistances = sorted(resistances_h1 + resistances_m15)

    ob_h1 = find_order_block(df_h1, "buy" if bias_bull else "sell")
    ob_m15 = find_order_block(df_m15, "buy" if bias_bull else "sell")
    fvg_h1 = find_fvg(df_h1, "buy" if bias_bull else "sell")
    fvg_m15 = find_fvg(df_m15, "buy" if bias_bull else "sell")

    support_candidates = []
    resistance_candidates = []
    if bias_bull:
        if sweep_level: support_candidates.append(sweep_level)
        if ob_h1: support_candidates.append(ob_h1[0])
        if ob_m15: support_candidates.append(ob_m15[0])
        if fvg_h1: support_candidates.append(fvg_h1[0])
        if fvg_m15: support_candidates.append(fvg_m15[0])
        support_candidates.extend(all_supports)
        resistance_candidates.extend(all_resistances)
    else:
        if sweep_level: resistance_candidates.append(sweep_level)
        if ob_h1: resistance_candidates.append(ob_h1[1])
        if ob_m15: resistance_candidates.append(ob_m15[1])
        if fvg_h1: resistance_candidates.append(fvg_h1[1])
        if fvg_m15: resistance_candidates.append(fvg_m15[1])
        resistance_candidates.extend(all_resistances)
        support_candidates.extend(all_supports)

    if bias_bull:
        valid_supports = [s for s in support_candidates if s < entry_raw]
        if valid_supports:
            best_support = max(valid_supports)
            final_entry = round(best_support + atr * 0.2, 6)
        else:
            final_entry = entry_raw

        sl_levels = [s - atr * 0.3 for s in support_candidates if s < final_entry]
        if sl_levels:
            sl = round(min(sl_levels), 6)
        else:
            sl = round(final_entry - atr * 1.5, 6)

        tp_levels = [r for r in resistance_candidates if r > final_entry]
        if tp_levels:
            tp = round(min(tp_levels) * 0.999, 6)
        else:
            tp = round(final_entry + atr * 2.0, 6)
    else:
        valid_resistances = [r for r in resistance_candidates if r > entry_raw]
        if valid_resistances:
            best_resistance = min(valid_resistances)
            final_entry = round(best_resistance - atr * 0.2, 6)
        else:
            final_entry = entry_raw

        sl_levels = [r + atr * 0.3 for r in resistance_candidates if r > final_entry]
        if sl_levels:
            sl = round(max(sl_levels), 6)
        else:
            sl = round(final_entry + atr * 1.5, 6)

        tp_levels = [s for s in support_candidates if s < final_entry]
        if tp_levels:
            tp = round(max(tp_levels) * 1.001, 6)
        else:
            tp = round(final_entry - atr * 2.0, 6)

    if bias_bull:
        if sl >= final_entry: sl = round(final_entry - atr * 1.0, 6)
        if tp <= final_entry: tp = round(final_entry + atr * 0.5, 6)
    else:
        if sl <= final_entry: sl = round(final_entry + atr * 1.0, 6)
        if tp >= final_entry: tp = round(final_entry - atr * 0.5, 6)

    return final_entry, tp, sl

def analyze_signal(symbol):
    df_d1 = fetch_klines(symbol, "1d", 200)
    df_h4 = fetch_klines(symbol, "4h", 200)
    df_h1 = fetch_klines(symbol, "1h", 150)
    df_m15 = fetch_klines(symbol, "15m", 150)
    df_m5 = fetch_klines(symbol, "5m", 150)
    if any(d is None for d in [df_d1, df_h4, df_h1, df_m15, df_m5]):
        return None

    df_d1 = add_all_indicators(df_d1)
    df_h4 = add_all_indicators(df_h4)
    df_h1 = add_all_indicators(df_h1)
    df_m15 = add_all_indicators(df_m15)
    df_m5 = add_all_indicators(df_m5)
    if any(d is None for d in [df_d1, df_h4, df_h1, df_m15, df_m5]):
        return None

    struct_d1 = market_structure(df_d1, 5)
    if struct_d1 == "ranging":
        return None
    bias_bull = struct_d1 == "bullish"
    direction = "BUY" if bias_bull else "SELL"
    score = 0

    last_d1 = df_d1.iloc[-1]
    if bias_bull and last_d1["close"] > last_d1["ema50"]:
        score += 10
    elif not bias_bull and last_d1["close"] < last_d1["ema50"]:
        score += 10
    if bias_bull and 40 < last_d1["rsi"] < 70:
        score += 5
    elif not bias_bull and 30 < last_d1["rsi"] < 60:
        score += 5

    struct_h4 = market_structure(df_h4, 3)
    if struct_h4 == struct_d1:
        score += 10
    if bias_bull and has_bullish_confirmation(df_h4):
        score += 5
    elif not bias_bull and has_bearish_confirmation(df_h4):
        score += 5

    sweep_h4, _ = detect_liquidity_sweep(df_h4, "buy" if bias_bull else "sell")
    if sweep_h4:
        score += 10

    struct_h1 = market_structure(df_h1, 2)
    h1_aligned = (struct_h1 == struct_d1)
    if h1_aligned:
        score += 10
    else:
        score -= 10

    last_h1 = df_h1.iloc[-2]
    if last_h1["volume"] > last_h1["vol_avg20"]:
        score += 5

    if bias_bull and last_h1["rsi"] >= 72:
        return None
    if not bias_bull and last_h1["rsi"] <= 28:
        return None

    last_m15 = df_m15.iloc[-2]
    if bias_bull and last_m15["ema12"] > last_m15["ema26"]:
        score += 10
    elif not bias_bull and last_m15["ema12"] < last_m15["ema26"]:
        score += 10
    else:
        return None

    sweep_m15, sweep_level_m15 = detect_liquidity_sweep(df_m15, "buy" if bias_bull else "sell")
    if sweep_m15:
        score += 15
        sweep_level = sweep_level_m15
    elif sweep_h4:
        sweep_level = None
    else:
        return None

    if bias_bull and last_m15["rsi"] >= 75:
        return None
    if not bias_bull and last_m15["rsi"] <= 25:
        return None

    if bias_bull and has_bullish_confirmation(df_m15):
        score += 5
    elif not bias_bull and has_bearish_confirmation(df_m15):
        score += 5

    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["close"] > last_m5["ema12"]:
        score += 5
    elif not bias_bull and last_m5["close"] < last_m5["ema12"]:
        score += 5
    else:
        return None

    confidence = min(score, 100)
    if confidence < settings["min_confidence"]:
        return None

    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else last_m15["close"] * 0.002
    entry_raw = round(last_m15["close"], 6)

    final_entry, tp, sl = find_best_entry_tp_sl(
        df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr
    )

    if bias_bull:
        risk = final_entry - sl
        reward = tp - final_entry
    else:
        risk = sl - final_entry
        reward = final_entry - tp
    if risk > 0 and reward / risk < 1.3:
        return None

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

# ---------- MANAJEMEN POSISI ----------
def manage_existing_positions():
    positions = get_open_positions()
    for pos in positions:
        symbol = pos["symbol"]
        orders = get_open_orders()
        has_tp_sl = any(o["symbol"] == symbol and o.get("type") in ("TAKE_PROFIT_MARKET", "STOP_MARKET") for o in orders)
        if has_tp_sl:
            continue

        side = "buy" if pos["side"] == "LONG" else "sell"
        df_h1 = fetch_klines(symbol, "1h", 150)
        df_h4 = fetch_klines(symbol, "4h", 150)
        if df_h1 is None or df_h4 is None:
            continue
        df_h1 = add_all_indicators(df_h1)
        df_h4 = add_all_indicators(df_h4)
        if df_h1 is None or df_h4 is None:
            continue

        entry_price = pos["entryPrice"]
        atr = df_h1["atr"].iloc[-1] if not np.isnan(df_h1["atr"].iloc[-1]) else entry_price * 0.002

        supports, resistances = get_levels(df_h1)
        if side == "buy":
            tp_candidates = [r for r in resistances if r > entry_price]
            tp = round(min(tp_candidates) * 0.999, 6) if tp_candidates else round(entry_price * 1.006, 6)
            sl_candidates = [s for s in supports if s < entry_price]
            sl = round(max(sl_candidates) - atr * 0.3, 6) if sl_candidates else round(entry_price * 0.992, 6)
        else:
            tp_candidates = [s for s in supports if s < entry_price]
            tp = round(max(tp_candidates) * 1.001, 6) if tp_candidates else round(entry_price * 0.994, 6)
            sl_candidates = [r for r in resistances if r > entry_price]
            sl = round(min(sl_candidates) + atr * 0.3, 6) if sl_candidates else round(entry_price * 1.008, 6)

        filters = get_symbol_filters(symbol)
        tick = filters.get("tickSize", 0.01)
        step = filters.get("stepSize", 0.001)
        qty = pos["amount"]
        tp_str = format_price(tp, tick)
        sl_str = format_price(sl, tick)
        qty_str = format_qty(qty, step)

        if place_tp_sl_for_position(symbol, pos["side"], qty_str, sl_str, tp_str):
            send_telegram(f"🛡️ TP/SL {symbol} dipasang: TP {tp_str} | SL {sl_str}")

def cleanup_orphan_orders():
    orders = get_open_orders()
    positions = get_open_positions()
    position_symbols = {p["symbol"] for p in positions}
    for o in orders:
        if o.get("type") in ("TAKE_PROFIT_MARKET", "STOP_MARKET"):
            if o["symbol"] not in position_symbols:
                cancel_order(o["symbol"], o.get("orderId") or o.get("algoId"))
                send_telegram(f"🧹 Orphan order {o['symbol']} dibatalkan.")

# ---------- PEMBATALAN ORDER JIKA HARGA SETENGAH JALAN KE TP ----------
def cancel_orders_half_tp():
    """Batalkan limit order jika harga sudah setengah perjalanan ke TP."""
    to_cancel = []
    for order_id, info in list(tracking_orders.items()):
        symbol = info["symbol"]
        entry = info["entry"]
        tp = info["tp"]
        side = info["side"]
        mark = get_mark_price(symbol)
        if mark is None:
            continue

        if side == "BUY":
            half_way = entry + (tp - entry) / 2
            if mark >= half_way:
                to_cancel.append(order_id)
        else:  # SELL
            half_way = entry - (entry - tp) / 2
            if mark <= half_way:
                to_cancel.append(order_id)

    for oid in to_cancel:
        info = tracking_orders[oid]
        cancel_order(info["symbol"], oid)
        del tracking_orders[oid]
        send_telegram(f"🧹 Limit order {info['symbol']} dibatalkan (harga setengah jalan ke TP).")

# ---------- SCANNING ----------
def get_coins():
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        data = requests.get(url, timeout=15).json()
        if not isinstance(data, list):
            return None
        tickers = [t for t in data if t["symbol"].endswith("USDT")]
        tickers.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        res = []
        for t in tickers:
            symbol = t["symbol"]
            if symbol in perma_banned:
                continue
            if float(t["lastPrice"]) <= settings["max_price"]:
                res.append(symbol)
            if len(res) >= settings["top_coins"]:
                break
        return res if res else None
    except:
        return None

def update_banned():
    to_del = [k for k, v in banned.items() if v <= 0]
    for k in to_del:
        del banned[k]
    for k in list(banned.keys()):
        banned[k] -= 1

def scan_signals():
    update_banned()
    coins = get_coins()
    if not coins:
        return []

    signals = []
    for sym in coins:
        if sym in banned or sym in perma_banned:
            continue
        try:
            sig = analyze_signal(sym)
            if sig:
                sig["symbol"] = sym
                signals.append(sig)
        except:
            pass
        time.sleep(0.02)

    if not signals:
        return []
    best = max(signals, key=lambda x: x["confidence"])
    return [best]

# ---------- EKSEKUSI ----------
def execute_signal(sig):
    symbol = sig["symbol"]
    side = "BUY" if sig["signal"] == "BUY" else "SELL"
    entry_price = sig["entry"]
    tp = sig["tp"]
    sl = sig["sl"]

    filters = get_symbol_filters(symbol)
    tick = filters.get("tickSize", 0.01)
    step = filters.get("stepSize", 0.001)
    min_notional = max(filters.get("minNotional", 5.0), settings["min_order_usd"])

    mark = get_mark_price(symbol)
    if mark is None:
        send_telegram("❌ Gagal ambil harga pasar.")
        return None
    raw_qty = min_notional / entry_price
    qty = round_to_step(raw_qty, step)
    if entry_price * qty < min_notional:
        qty += step
        qty = round_to_step(qty, step)

    entry_str = format_price(entry_price, tick)
    qty_str = format_qty(qty, step)

    set_leverage(symbol, settings["leverage"])

    order_id = place_limit_order(symbol, side, qty_str, entry_str)
    if not order_id:
        current_price = get_mark_price(symbol)
        if current_price:
            if (side == "BUY" and current_price < entry_price) or (side == "SELL" and current_price > entry_price):
                order_id = place_market_order(symbol, side, qty_str)
                send_telegram(f"⚠️ Limit gagal, Market Order di {current_price}")
            else:
                order_id = place_market_order(symbol, side, qty_str)
                send_telegram(f"⚠️ Limit gagal, Market Order sebagai fallback")

    if order_id:
        tracking_orders[order_id] = {
            "symbol": symbol,
            "entry": entry_price,
            "tp": tp,
            "sl": sl,
            "side": side,
        }
        send_telegram(f"✅ Order {side} {symbol} terpasang (ID: {order_id})")
        banned[symbol] = settings["ban_cycles"]
        return order_id
    else:
        send_telegram(f"❌ Gagal eksekusi order {symbol}")
        return None

# ---------- MAIN LOOP ----------
def main_loop():
    while True:
        update_banned()
        manage_existing_positions()
        cleanup_orphan_orders()
        cancel_orders_half_tp()  # ganti dari cancel_orders_tp_reached

        positions = get_open_positions()
        limit_orders = [o for o in get_open_orders() if o.get("type") == "LIMIT"]
        total_active = len(positions) + len(limit_orders)

        if total_active >= settings["max_positions"]:
            send_telegram(f"🛑 Kapasitas penuh ({total_active}/{settings['max_positions']}). Menunggu...")
            time.sleep(settings["scan_interval"])
            continue

        # Scanning sinyal sampai dapat
        while True:
            signals = scan_signals()
            if signals:
                break
            time.sleep(settings["scan_interval"])

        best_sig = signals[0]
        execute_signal(best_sig)

        msg = (
            f"📊 <b>{best_sig['signal']} {best_sig['symbol']}</b>\n"
            f"Entry: {best_sig['entry']}\nTP: {best_sig['tp']} | SL: {best_sig['sl']}\n"
            f"Conf: {best_sig['confidence']}% | RR: 1:{best_sig['rr']} | ATR: {best_sig['atr']}"
        )
        send_telegram(msg)

        time.sleep(2)

# ========== STARTUP ==========
if __name__ == "__main__":
    print("=" * 60)
    print("  FULL AUTO TRADING BOT")
    print("=" * 60)
    try:
        ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
        send_telegram(f"🚀 Bot dimulai!\nIP: {ip}\nPastikan IP di-whitelist di Binance.")
    except:
        send_telegram("🚀 Bot dimulai! (IP tidak terdeteksi)")

    if not API_KEY or not SECRET_KEY:
        send_telegram("❌ API Key/Secret belum diset di environment.")
    else:
        threading.Thread(target=telegram_polling, daemon=True).start()
        main_loop()

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
