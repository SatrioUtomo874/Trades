#!/usr/bin/env python3
"""
FULL AUTO TRADING BOT – Alur Lengkap
Manajemen posisi aktif, TP/SL otomatis, limit order tracking,
scanning sinyal SMC, eksekusi order, pengaturan via /set.
"""

import os, time, hmac, hashlib, math, threading, json, requests, pandas as pd, numpy as np
from datetime import datetime
from flask import Flask

# ================== KONFIGURASI ==================
TELEGRAM_TOKEN = "8094484109:AAF9Z3lQUxdQFqqeG6NKV9O1EC0vrxzJy0U"
CHAT_ID = "8041197505"
API_KEY = os.environ.get("BINANCE_API_KEY", "")
SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")

settings = {
    "leverage": 8,
    "min_order_usd": 1.0,
    "max_price": 50.0,
    "min_confidence": 40,
    "min_rr": 1.6,
    "entry_shift_pips": 5,
    "volume_mult": 0.0,
    "tp_distance_atr": 0.0,
    "rsi_h1_buy_max": 75,
    "rsi_h1_sell_min": 25,
    "rsi_m15_buy_max": 78,
    "rsi_m15_sell_min": 22,
    "sweep_mode": "any",
    "require_confirmation": False,
    "ban_cycles": 20,
    "scan_interval": 3,
    "top_coins": 50,
    "max_positions": 4,
}

banned = {}
perma_banned = set()
bot_running = True
tracking_orders = {}  # order_id -> {symbol, entry, tp, sl, side}

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

# ========== BINANCE API ==========
def binance_request(endpoint, params=None, method="GET", auth=True):
    if params is None: params = {}
    if auth:
        try:
            server_time = requests.get("https://fapi.binance.com/fapi/v1/time", timeout=5).json()["serverTime"]
        except:
            server_time = int(time.time() * 1000)
        params["timestamp"] = server_time
        params["recvWindow"] = 10000
        qs = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(SECRET_KEY.encode(), qs.encode(), hashlib.sha256).hexdigest()
        url = f"https://fapi.binance.com{endpoint}?{qs}&signature={signature}"
    else:
        url = f"https://fapi.binance.com{endpoint}"
        if params:
            url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])

    headers = {"X-MBX-APIKEY": API_KEY} if auth else {}
    for attempt in range(5):
        try:
            time.sleep(0.3 * (attempt + 1))
            if method == "GET": r = requests.get(url, headers=headers, timeout=15)
            elif method == "POST": r = requests.post(url, headers=headers, timeout=15)
            elif method == "DELETE": r = requests.delete(url, headers=headers, timeout=15)
            else: return None

            if r.status_code == 200: return r.json()
            elif r.status_code == 418:
                log_activity("⛔ IP dibanned! Bot berhenti 15 menit.")
                time.sleep(900)
                continue
            elif r.status_code == 429:
                wait = 15 * (attempt + 1)
                log_activity(f"⏳ Rate limited, tunggu {wait}s")
                time.sleep(wait)
                continue
            else:
                try:
                    err = r.json()
                    log_activity(f"❌ API {r.status_code}: {err.get('msg','')}")
                except:
                    log_activity(f"❌ API {r.status_code}")
                return None
        except Exception as e:
            log_activity(f"⚠️ Network: {e}")
            time.sleep(5)
    return None

# ---------- Order & Account Helpers ----------
def get_open_positions():
    raw = binance_request("/fapi/v2/positionRisk")
    if not raw or not isinstance(raw, list): return []
    positions = []
    for p in raw:
        amt = float(p.get("positionAmt", 0))
        if amt != 0:
            positions.append({
                "symbol": p["symbol"],
                "side": "LONG" if amt > 0 else "SHORT",
                "amount": abs(amt),
                "entryPrice": float(p["entryPrice"]),
            })
    return positions

def get_open_orders():
    orders = []
    lim = binance_request("/fapi/v1/openOrders")
    if lim and isinstance(lim, list): orders.extend(lim)
    algo = binance_request("/fapi/v1/algoOpenOrders")
    if algo and isinstance(algo, list): orders.extend(algo)
    return orders

def cancel_order(symbol, order_id):
    r = binance_request("/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, method="DELETE")
    if r and r.get("status") == "CANCELED": return True
    r = binance_request("/fapi/v1/algoOrder", {"symbol": symbol, "algoId": order_id}, method="DELETE")
    return r is not None

def place_limit_order(symbol, side, quantity, price):
    params = {"symbol":symbol,"side":side,"type":"LIMIT","quantity":quantity,"price":price,"timeInForce":"GTC"}
    res = binance_request("/fapi/v1/order", params, method="POST")
    return res["orderId"] if res and "orderId" in res else None

def place_market_order(symbol, side, quantity):
    params = {"symbol":symbol,"side":side,"type":"MARKET","quantity":quantity}
    res = binance_request("/fapi/v1/order", params, method="POST")
    return res["orderId"] if res and "orderId" in res else None

def place_tp_sl_for_position(symbol, side, quantity, sl_price, tp_price):
    close_side = "SELL" if side == "LONG" else "BUY"
    tp = binance_request("/fapi/v1/algoOrder", {
        "algoType":"CONDITIONAL","symbol":symbol,"side":close_side,"type":"TAKE_PROFIT_MARKET",
        "quantity":quantity,"triggerPrice":tp_price,"workingType":"MARK_PRICE"}, method="POST")
    sl = binance_request("/fapi/v1/algoOrder", {
        "algoType":"CONDITIONAL","symbol":symbol,"side":close_side,"type":"STOP_MARKET",
        "quantity":quantity,"triggerPrice":sl_price,"workingType":"MARK_PRICE"}, method="POST")
    return tp is not None and sl is not None

def set_leverage(symbol, lev):
    res = binance_request("/fapi/v1/leverage", {"symbol": symbol, "leverage": int(lev)}, method="POST")
    return res is not None and "leverage" in res

def get_mark_price(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}", timeout=5)
        return float(r.json()["price"])
    except: return None

def get_symbol_filters(symbol):
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        for s in r.json()["symbols"]:
            if s["symbol"] == symbol:
                f = {}
                for fl in s["filters"]:
                    if fl["filterType"]=="PRICE_FILTER": f["tickSize"]=float(fl["tickSize"])
                    elif fl["filterType"]=="LOT_SIZE": f["stepSize"]=float(fl["stepSize"]); f["minQty"]=float(fl["minQty"])
                    elif fl["filterType"]=="MIN_NOTIONAL": f["minNotional"]=float(fl["notional"])
                return f
    except: pass
    return {"tickSize":0.01,"stepSize":0.001,"minQty":0.001,"minNotional":5.0}

def round_to_tick(v, tick):
    if tick == 0: return v
    prec = int(round(-math.log10(tick), 0))
    return round(round(v/tick)*tick, prec)

def round_to_step(v, step):
    if step == 0: return v
    prec = int(round(-math.log10(step), 0))
    return round(round(v/step)*step, prec)

def fmt_price(v, tick):
    if tick == 0: return str(v)
    prec = int(round(-math.log10(tick), 0))
    return f"{v:.{prec}f}"

def fmt_qty(v, step):
    if step == 0: return str(v)
    prec = int(round(-math.log10(step), 0))
    return f"{v:.{prec}f}"

# ========== DATA PUBLIK ==========
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

# ========== KLINE BYBIT ==========
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

# ========== INDIKATOR & SMC ==========
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

def find_best_entry_tp_sl(df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr):
    supports_h1, resistances_h1 = get_levels(df_h1)
    supports_m15, resistances_m15 = get_levels(df_m15)
    all_supports = sorted(supports_h1 + supports_m15, reverse=True)
    all_resistances = sorted(resistances_h1 + resistances_m15)

    shift_pct = settings["entry_shift_pips"] * 0.0001
    shift = shift_pct * entry_raw if settings["entry_shift_pips"] > 0 else 0

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
        if reward / risk < settings["min_rr"]:
            for i, res in enumerate(all_resistances):
                if res > final_entry and i > 0:
                    tp = round(res * 0.999, 6)
                    reward = tp - final_entry
                    if reward / risk >= settings["min_rr"]:
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
        if reward / risk < settings["min_rr"]:
            for i, sup in enumerate(all_supports):
                if sup < final_entry and i > 0:
                    tp = round(sup * 1.001, 6)
                    reward = final_entry - tp
                    if reward / risk >= settings["min_rr"]:
                        break

    if bias_bull:
        if sl >= final_entry: sl = round(final_entry - atr * 1.0, 6)
        if tp <= final_entry: tp = round(final_entry + atr * 0.5, 6)
    else:
        if sl <= final_entry: sl = round(final_entry + atr * 1.0, 6)
        if tp >= final_entry: tp = round(final_entry - atr * 0.5, 6)

    return final_entry, tp, sl

def analyze_signal(symbol):
    df_d1 = fetch_klines(symbol,"1d",200); df_h4 = fetch_klines(symbol,"4h",200)
    df_h1 = fetch_klines(symbol,"1h",150); df_m15 = fetch_klines(symbol,"15m",150); df_m5 = fetch_klines(symbol,"5m",150)
    if any(d is None for d in [df_d1,df_h4,df_h1,df_m15,df_m5]): return None

    df_d1 = add_all_indicators(df_d1)
    df_h4 = add_all_indicators(df_h4)
    df_h1 = add_all_indicators(df_h1)
    df_m15 = add_all_indicators(df_m15)
    df_m5 = add_all_indicators(df_m5)
    if any(d is None for d in [df_d1,df_h4,df_h1,df_m15,df_m5]): return None

    struct_d1 = market_structure(df_d1, 5)
    if struct_d1 == "ranging": return None
    bias_bull = struct_d1 == "bullish"
    direction = "BUY" if bias_bull else "SELL"
    score = 0

    last_d1 = df_d1.iloc[-1]
    if bias_bull and last_d1["close"] > last_d1["ema50"]: score += 10
    elif not bias_bull and last_d1["close"] < last_d1["ema50"]: score += 10
    if bias_bull and 40 < last_d1["rsi"] < 70: score += 5
    elif not bias_bull and 30 < last_d1["rsi"] < 60: score += 5

    struct_h4 = market_structure(df_h4, 3)
    if struct_h4 == struct_d1: score += 10
    sweep_h4, _ = detect_liquidity_sweep(df_h4, "buy" if bias_bull else "sell")
    if sweep_h4: score += 10

    struct_h1 = market_structure(df_h1, 2)
    if struct_h1 == struct_d1: score += 10
    last_h1 = df_h1.iloc[-2]
    if settings["volume_mult"] > 0 and last_h1["volume"] <= settings["volume_mult"] * last_h1["vol_avg20"]: return None
    if last_h1["volume"] > last_h1["vol_avg20"]: score += 5
    if bias_bull and last_h1["rsi"] >= settings["rsi_h1_buy_max"]: return None
    if not bias_bull and last_h1["rsi"] <= settings["rsi_h1_sell_min"]: return None

    last_m15 = df_m15.iloc[-2]
    if bias_bull and last_m15["ema12"] > last_m15["ema26"]: score += 10
    elif not bias_bull and last_m15["ema12"] < last_m15["ema26"]: score += 10
    else: return None

    sweep_m15, sweep_level = detect_liquidity_sweep(df_m15, "buy" if bias_bull else "sell")
    if settings["sweep_mode"] == "m15" and not sweep_m15: return None
    if settings["sweep_mode"] == "any" and not (sweep_m15 or sweep_h4): return None
    if sweep_m15: score += 15
    elif sweep_h4: score += 10

    if bias_bull and last_m15["rsi"] >= settings["rsi_m15_buy_max"]: return None
    if not bias_bull and last_m15["rsi"] <= settings["rsi_m15_sell_min"]: return None
    if bias_bull and last_m15["rsi"] < 65: score += 5
    elif not bias_bull and last_m15["rsi"] > 35: score += 5

    if settings["require_confirmation"]:
        if bias_bull and not has_bullish_confirmation(df_m15): return None
        if not bias_bull and not has_bearish_confirmation(df_m15): return None
        score += 5

    last_m5 = df_m5.iloc[-2]
    if bias_bull and last_m5["close"] > last_m5["ema12"]: score += 5
    elif not bias_bull and last_m5["close"] < last_m5["ema12"]: score += 5
    else: return None

    atr = last_m15["atr"] if not np.isnan(last_m15["atr"]) else last_m15["close"] * 0.002
    entry_raw = round(last_m15["close"], 6)

    final_entry, tp, sl = find_best_entry_tp_sl(
        df_h1, df_m15, bias_bull, entry_raw, sweep_level, atr
    )

    if settings["tp_distance_atr"] > 0:
        if bias_bull and (tp - final_entry) < settings["tp_distance_atr"] * atr: return None
        if not bias_bull and (final_entry - tp) < settings["tp_distance_atr"] * atr: return None

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
        "rr": round(abs(tp - final_entry) / abs(final_entry - sl), 2) if abs(final_entry - sl) > 0 else 0
    }

# ========== MANAJEMEN POSISI AKTIF ==========
def manage_positions():
    positions = get_open_positions()
    if not positions:
        log_activity("ℹ️ Tidak ada posisi aktif.")
        return
    log_activity(f"🔍 Memeriksa {len(positions)} posisi...")
    for pos in positions:
        sym = pos["symbol"]
        orders = get_open_orders()
        has_tp_sl = any(o["symbol"]==sym and o.get("type") in ("TAKE_PROFIT_MARKET","STOP_MARKET") for o in orders)
        if has_tp_sl:
            continue
        # Analisa untuk TP/SL posisi
        side = "buy" if pos["side"]=="LONG" else "sell"
        df_h1 = fetch_klines(sym,"1h",150)
        df_h4 = fetch_klines(sym,"4h",200)
        if df_h1 is None or df_h4 is None:
            continue
        df_h1 = add_all_indicators(df_h1)
        df_h4 = add_all_indicators(df_h4)
        if df_h1 is None:
            continue
        entry = pos["entryPrice"]
        atr = df_h1["atr"].iloc[-1] if not np.isnan(df_h1["atr"].iloc[-1]) else entry*0.002
        supports, resistances = get_levels(df_h1)
        if side=="buy":
            tp_cand = [r for r in resistances if r>entry]
            tp = round(min(tp_cand)*0.999,6) if tp_cand else round(entry*1.01,6)
            sl_cand = [s for s in supports if s<entry]
            sl = round(max(sl_cand)-atr*0.3,6) if sl_cand else round(entry*0.99,6)
        else:
            tp_cand = [s for s in supports if s<entry]
            tp = round(max(tp_cand)*1.001,6) if tp_cand else round(entry*0.99,6)
            sl_cand = [r for r in resistances if r>entry]
            sl = round(min(sl_cand)+atr*0.3,6) if sl_cand else round(entry*1.01,6)
        filters = get_symbol_filters(sym)
        tick = filters.get("tickSize",0.01)
        step = filters.get("stepSize",0.001)
        qty_str = fmt_qty(pos["amount"], step)
        tp_str = fmt_price(tp, tick)
        sl_str = fmt_price(sl, tick)
        if place_tp_sl_for_position(sym, pos["side"], qty_str, sl_str, tp_str):
            log_activity(f"🛡️ TP/SL {sym} dipasang: TP {tp_str} | SL {sl_str}")
        else:
            log_activity(f"⚠️ Gagal pasang TP/SL {sym}")

# ========== SCANNING ==========
def update_banned():
    to_del = [k for k,v in banned.items() if v<=0]
    for k in to_del: del banned[k]
    for k in list(banned.keys()): banned[k] -= 1

def scan_signals():
    if not bot_running:
        return []
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

# ========== EKSEKUSI ORDER ==========
def execute_signal(sig):
    if not bot_running: return
    symbol = sig["symbol"]
    side = "BUY" if sig["signal"]=="BUY" else "SELL"
    entry, tp, sl = sig["entry"], sig["tp"], sig["sl"]
    filters = get_symbol_filters(symbol)
 tick = filters.get("tickSize", 0.01)
 step = filters.get("stepSize", 0.001)
 exchange_min_notional = filters.get("minNotional", 5.0)
# Pastikan minimal $5 digunakan jika exchange tidak memberikan angka
 if exchange_min_notional < 5.0:
     exchange_min_notional = 5.0
 min_notional = max(exchange_min_notional, settings["min_order_usd"])
 log_activity(f"ℹ️ {symbol} minNotional: {min_notional} USD (exchange: {exchange_min_notional}, setting: {settings['min_order_usd']})")

 qty = round_to_step(min_notional / entry, step)
 while entry * qty < min_notional:
     qty += step
     qty = round_to_step(qty, step)
 if qty < filters.get("minQty", 0.001):
     qty = filters["minQty"]
     qty = round_to_step(qty, step)
     lev_ok = set_leverage(symbol, settings["leverage"])
 if not lev_ok:
        log_activity(f"⚠️ Leverage gagal, sinyal dikirim tanpa order.")
        msg = (
            f"<b>📊 {sig['signal']} {symbol} (NO ORDER)</b>\n"
            f"Entry: {entry}\nTP: {tp} | SL: {sl}\n"
            f"Conf: {sig['confidence']}% | RR: 1:{sig['rr']} | ATR: {sig['atr']}\n"
            f"⚠️ Leverage gagal."
        )
        send_telegram(msg)
        banned[symbol] = settings["ban_cycles"]
        return

    log_activity(f"🚀 {side} {symbol} Entry: {entry_str} Qty: {qty_str} Conf: {sig['confidence']}%")

    order_id = place_limit_order(symbol, side, qty_str, entry_str)
    if not order_id:
        current = get_mark_price(symbol)
        if current:
            if (side == "BUY" and current < entry) or (side == "SELL" and current > entry):
                order_id = place_market_order(symbol, side, qty_str)
                if order_id:
                    log_activity(f"⚠️ Limit gagal, Market Order di {current}")
                    entry = current  # update entry untuk tracking
                else:
                    log_activity(f"❌ Market order juga gagal.")
            else:
                log_activity(f"ℹ️ Harga tidak menguntungkan, limit order batal.")
                # tetap kirim sinyal tanpa order
                msg = (
                    f"<b>📊 {sig['signal']} {symbol} (NO ORDER)</b>\n"
                    f"Entry: {entry}\nTP: {tp} | SL: {sl}\n"
                    f"Conf: {sig['confidence']}% | RR: 1:{sig['rr']} | ATR: {sig['atr']}\n"
                    f"⚠️ Limit gagal, harga tidak menguntungkan."
                )
                send_telegram(msg)
                banned[symbol] = settings["ban_cycles"]
                return

    if order_id:
        tracking_orders[order_id] = {
            "symbol": symbol,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "side": side,
        }
        log_activity(f"✅ Order {side} {symbol} terpasang (ID: {order_id})")
        banned[symbol] = settings["ban_cycles"]
        msg = (
            f"<b>📊 {sig['signal']} {symbol}</b>\n"
            f"Entry: {entry}\nTP: {tp} | SL: {sl}\n"
            f"Conf: {sig['confidence']}% | RR: 1:{sig['rr']} | ATR: {sig['atr']}"
        )
        send_telegram(msg)
    else:
        log_activity(f"❌ Gagal eksekusi order {symbol}")

# ========== PEMBATALAN ORDER TRACKING ==========
def cancel_tracked_orders():
    to_cancel = []
    for oid, info in list(tracking_orders.items()):
        symbol = info["symbol"]
        tp = info["tp"]
        mark = get_mark_price(symbol)
        if mark is None: continue
        if info["side"] == "BUY" and mark >= tp:
            to_cancel.append(oid)
        elif info["side"] == "SELL" and mark <= tp:
            to_cancel.append(oid)
    for oid in to_cancel:
        info = tracking_orders[oid]
        cancel_order(info["symbol"], oid)
        log_activity(f"🧹 Limit order {info['symbol']} dibatalkan (TP tercapai).")
        del tracking_orders[oid]

# ========== LOOP UTAMA ==========
def main_loop():
    global bot_running
    log_activity("🔄 Bot mulai full auto trading...")
    while True:
        if not bot_running:
            time.sleep(2)
            continue
        try:
            # 1. Manajemen posisi aktif
            manage_positions()
            # 2. Cek posisi aktif & limit order
            positions = get_open_positions()
            pos_count = len(positions)
            limit_orders = [o for o in get_open_orders() if o.get("type") == "LIMIT"]
            limit_count = len(limit_orders)
            total_active = pos_count + limit_count
            log_activity(f"📊 Posisi: {pos_count} | Limit: {limit_count} | Max: {settings['max_positions']}")

            # 3. Bersihkan order yang TP-nya sudah tercapai
            cancel_tracked_orders()

            # 4. Jika kapasitas penuh, tunggu
            if total_active >= settings["max_positions"]:
                log_activity("🛑 Kapasitas penuh, menunggu...")
                time.sleep(30)
                continue

            # 5. Scanning sinyal sampai dapat
            while True:
                if not bot_running: break
                signals = scan_signals()
                if signals:
                    execute_signal(signals[0])
                    break  # kembali ke awal setelah eksekusi
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
                    send_telegram(f"📊 Bot: {'Running' if bot_running else 'Stopped'} | Banned: {len(banned)} koin")
                elif text == "/menu":
                    send_telegram("""<b>Command List:</b>
/start - Mulai bot
/stop - Hentikan bot
/status - Status bot
/settings - Lihat pengaturan
/set key value - Ubah pengaturan
/menu - Tampilkan menu""")
            time.sleep(1)
        except Exception as e:
            print(f"Polling error: {e}"); time.sleep(5)

# ========== STARTUP ==========
if __name__ == "__main__":
    log_activity("🤖 Bot starting...")
    try:
        ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
        log_activity(f"🚀 Bot dimulai!\nIP: {ip}\nPastikan IP di-whitelist di Binance.")
    except:
        log_activity("🚀 Bot dimulai! (IP tidak terdeteksi)")
    if not API_KEY or not SECRET_KEY:
        log_activity("❌ API Key/Secret belum diset. Bot hanya bisa scanning tanpa eksekusi.")
    threading.Thread(target=telegram_polling, daemon=True).start()
    threading.Thread(target=main_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
