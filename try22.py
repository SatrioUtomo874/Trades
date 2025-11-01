# indicator.py
import pandas as pd
import numpy as np
from binance.spot import Spot
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from flask import Flask, jsonify
import threading
import time

load_dotenv()

# Flask app untuk web service
app = Flask(__name__)

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Inisialisasi Binance client
client = Spot(api_key=API_KEY, api_secret=API_SECRET, base_url="https://api.binance.com")

# -------------------------------------------------------------------
# ✅ Helper — Fetch klines
# -------------------------------------------------------------------
def get_klines(client, symbol: str, interval: str, limit: int = 500):
    raw = client.klines(symbol, interval, limit=limit)

    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)

    return df

# -------------------------------------------------------------------
# ✅ 1. TREND DETECTOR (EMA50 vs EMA200)
# -------------------------------------------------------------------
def get_trend(client, symbol: str, interval: str = "1h", limit: int = 500):
    df = get_klines(client, symbol, interval, limit)

    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    ema50 = df["ema50"].iloc[-1]
    ema200 = df["ema200"].iloc[-1]

    if ema50 > ema200 * 1.01:
        return "UPTREND"
    elif ema50 < ema200 * 0.99:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

# -------------------------------------------------------------------
# ✅ 2. RSI (default RSI-14)
# -------------------------------------------------------------------
def get_rsi(client, symbol: str, interval: str = "1h", period: int = 14, limit: int = 200):
    df = get_klines(client, symbol, interval, limit)

    delta = df["close"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    return float(RSI.iloc[-1])

# -------------------------------------------------------------------
# ✅ 3. SMC TREND (Swing, BOS, CHoCH)
# -------------------------------------------------------------------
def find_swing_high_low(df, left=2, right=2):
    swing_high = []
    swing_low = []

    for i in range(left, len(df) - right):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        if high == max(df["high"].iloc[i-left:i+right+1]):
            swing_high.append(i)

        if low == min(df["low"].iloc[i-left:i+right+1]):
            swing_low.append(i)

    return swing_high, swing_low

def detect_structure(df, swing_high, swing_low):
    last_close = df["close"].iloc[-1]

    bos = None
    choch = None

    if len(swing_high) > 0:
        last_sh = df["high"].iloc[swing_high[-1]]
        if last_close > last_sh:
            bos = "BOS_UP"

    if len(swing_low) > 0:
        last_sl = df["low"].iloc[swing_low[-1]]
        if last_close < last_sl:
            bos = "BOS_DOWN"

    # CHoCH logic (reversal)
    if len(swing_low) > 0 and len(swing_high) > 0:
        last_sh = df["high"].iloc[swing_high[-1]]
        last_sl = df["low"].iloc[swing_low[-1]]

        if last_close > last_sh:
            choch = "CHOCH_UP"
        if last_close < last_sl:
            choch = "CHOCH_DOWN"

    return bos, choch

def get_trend_smc(client, symbol: str, interval: str = "1h", limit: int = 500):
    df = get_klines(client, symbol, interval, limit)

    swing_high, swing_low = find_swing_high_low(df)
    bos, choch = detect_structure(df, swing_high, swing_low)

    # Priority: CHoCH > BOS > Sideways
    if choch == "CHOCH_UP":
        return "UPTREND (CHOCH-UP)"
    if choch == "CHOCH_DOWN":
        return "DOWNTREND (CHOCH-DOWN)"

    if bos == "BOS_UP":
        return "UPTREND"
    if bos == "BOS_DOWN":
        return "DOWNTREND"

    return "SIDEWAYS"

def get_support_resistance(client, symbol: str, interval: str = "1h", lookback: int = 50):
    """
    Cari support & resistance dari swing high & low
    """
    klines = client.klines(symbol, interval, limit=lookback)
    if not klines:
        return None

    highs = np.array([float(k[2]) for k in klines])  # high setiap candle
    lows = np.array([float(k[3]) for k in klines])   # low setiap candle

    resistance = np.max(highs)  # titik tertinggi → resistance
    support = np.min(lows)      # titik terendah → support

    return {"support": support, "resistance": resistance}

def get_volume(client, symbol: str, interval: str = "1h", limit: int = 1):
    """
    Mengambil volume terakhir dari symbol di timeframe tertentu.
    Return: float (volume)
    """
    raw = client.klines(symbol, interval, limit=limit)
    
    if not raw:
        return None

    last_candle = raw[-1]
    volume = float(last_candle[5])  # index 5 = volume

    return volume

def get_volume_average(client, symbol: str, interval: str = "1h", limit: int = 20):
    """
    Mengambil rata-rata volume untuk perbandingan
    """
    raw = client.klines(symbol, interval, limit=limit)
    
    if not raw:
        return None

    volumes = [float(candle[5]) for candle in raw]
    return sum(volumes) / len(volumes)

def get_current_price(client, symbol: str):
    """
    Ambil harga terakhir (last price) dari symbol
    Return: float
    """
    data = client.ticker_price(symbol)
    return float(data['price'])

def find_swing(client, symbol: str, interval: str = "15m", limit: int = 100, left=2, right=2):
    """
    Cari swing high dan swing low dari symbol & timeframe tertentu
    Return: tuple (list of swing high indices, list of swing low indices)
    """
    df = get_klines(client, symbol, interval, limit)
    
    swing_high_idx = []
    swing_low_idx = []

    for i in range(left, len(df) - right):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        if high == max(df["high"].iloc[i-left:i+right+1]):
            swing_high_idx.append(i)

        if low == min(df["low"].iloc[i-left:i+right+1]):
            swing_low_idx.append(i)

    return swing_high_idx, swing_low_idx

def get_trade_levels_with_percent(client, symbol: str, interval: str = "15m", buffer=0.001, sl_buffer=0.0015):
    """
    Hitung entry, TP, SL untuk scalping dengan pendekatan realistis
    """
    df = get_klines(client, symbol, interval, limit=100)
    trend = get_trend(client, symbol, interval)
    smc = get_trend_smc(client, symbol, interval)
    current_price = get_current_price(client, symbol)

    swing_high, swing_low = find_swing(client, symbol, interval, limit=len(df))
    
    # Cari swing terdekat di atas/bawah harga sekarang
    nearest_swing_high = min([h for h in swing_high if df["high"].iloc[h] > current_price], default=None)
    nearest_swing_low = max([l for l in swing_low if df["low"].iloc[l] < current_price], default=None)

    # Tentukan level
    if "UPTREND" in trend or "UPTREND" in smc:
        entry = current_price * (1 + buffer)
        tp = df["high"].iloc[nearest_swing_high] if nearest_swing_high else current_price * 1.003
        sl = df["low"].iloc[nearest_swing_low] * (1 - sl_buffer) if nearest_swing_low else current_price * (1 - sl_buffer)
        direction = "LONG"
    elif "DOWNTREND" in trend or "DOWNTREND" in smc:
        entry = current_price * (1 - buffer)
        tp = df["low"].iloc[nearest_swing_low] if nearest_swing_low else current_price * 0.997
        sl = df["high"].iloc[nearest_swing_high] * (1 + sl_buffer) if nearest_swing_high else current_price * (1 + sl_buffer)
        direction = "SHORT"
    else:
        entry = current_price
        tp = None
        sl = None
        direction = "SIDEWAYS"

    tp_percent = ((tp - entry) / entry * 100) if tp else None
    sl_percent = ((entry - sl) / entry * 100) if sl else None
    if direction == "SHORT":
        tp_percent = ((entry - tp) / entry * 100) if tp else None
        sl_percent = ((sl - entry) / entry * 100) if sl else None

    return {
        "entry_price": round(entry, 8),
        "take_profit": round(tp, 8) if tp else None,
        "stop_loss": round(sl, 8) if sl else None,
        "tp_percent": round(tp_percent, 2) if tp_percent else None,
        "sl_percent": round(sl_percent, 2) if sl_percent else None,
        "direction": direction
    }

# -------------------------------------------------------------------
# Flask Routes untuk Web Service
# -------------------------------------------------------------------
@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "Crypto Trading Indicator API is running!",
        "endpoints": {
            "/health": "Health check",
            "/analysis/<symbol>": "Get trading analysis for a symbol",
            "/price/<symbol>": "Get current price"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/analysis/<symbol>')
def analysis(symbol):
    try:
        symbol = symbol.upper()
        
        # Ambil data
        trend = get_trend(client, symbol, "1d")
        rsi = get_rsi(client, symbol, "15m")
        smc = get_trend_smc(client, symbol, "4h")
        volume = get_volume(client, symbol, "15m")
        sr_levels = get_support_resistance(client, symbol, "4h")
        price = get_current_price(client, symbol)
        trade_levels = get_trade_levels_with_percent(client, symbol, "5m")
        
        # Hitung confidence
        confidence = 0
        
        # 1️⃣ Trend EMA
        if "UPTREND" in trend:
            confidence += 30
        elif "DOWNTREND" in trend:
            confidence += 20

        # 2️⃣ Trend SMC
        if "UPTREND" in smc:
            confidence += 30
        elif "DOWNTREND" in smc:
            confidence += 15

        # 3️⃣ Volume (dibandingkan rata-rata historis)
        avg_volume = get_volume_average(client, symbol, "4h", limit=20)
        if volume and avg_volume and volume >= avg_volume * 0.8:
            confidence += 20
        elif volume and avg_volume:
            confidence += 15

        # 4️⃣ RSI
        if rsi < 30:  # oversold
            confidence += 5
        elif rsi > 70:  # overbought
            confidence -= 5
        else:
            confidence += 15  # netral, aman untuk masuk

        # Clamp confidence antara 0 - 100
        confidence = max(0, min(confidence, 100))
        
        response = {
            "symbol": symbol,
            "analysis": {
                "trend_ema": trend,
                "rsi": round(rsi, 2),
                "trend_smc": smc,
                "volume": volume,
                "current_price": price,
                "support": sr_levels["support"] if sr_levels else None,
                "resistance": sr_levels["resistance"] if sr_levels else None
            },
            "trade_levels": trade_levels,
            "confidence": round(confidence, 2)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/price/<symbol>')
def price(symbol):
    try:
        price = get_current_price(client, symbol.upper())
        return jsonify({"symbol": symbol.upper(), "price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# Telegram Bot Functions
# -------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("/start")  # cetak ke terminal
    await update.message.reply_text("Bot sudah di-start!")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("/stop")  # cetak ke terminal
    await update.message.reply_text("Bot sudah di-stop!")

async def print_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        text_to_print = " ".join(context.args)
        print(text_to_print)  # cetak ke terminal
        await update.message.reply_text(f"Printed to terminal: {text_to_print}")
    else:
        await update.message.reply_text("Tidak ada teks untuk dicetak.")

async def coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        symbol = context.args[0].upper()

        try:
            # Ambil data
            trend = get_trend(client, symbol, "1d")
            rsi = get_rsi(client, symbol, "15m")
            smc = get_trend_smc(client, symbol, "4h")
            volume = get_volume(client, symbol, "15m")
            sr_levels = get_support_resistance(client, symbol, "4h")
            price = get_current_price(client, symbol)
            trade_levels = get_trade_levels_with_percent(client, symbol, "5m")
            
            # Hitung confidence
            confidence = 0
            
            if "UPTREND" in trend:
                confidence += 30
            elif "DOWNTREND" in trend:
                confidence += 20

            if "UPTREND" in smc:
                confidence += 30
            elif "DOWNTREND" in smc:
                confidence += 15

            avg_volume = get_volume_average(client, symbol, "4h", limit=20)
            if volume and avg_volume and volume >= avg_volume * 0.8:
                confidence += 20
            elif volume and avg_volume:
                confidence += 15

            if rsi < 30:
                confidence += 5
            elif rsi > 70:
                confidence -= 5
            else:
                confidence += 15

            confidence = max(0, min(confidence, 100))

            # Format output rapi
            message = f"📊 *{symbol} Analysis*\n"
            message += f"-------------------------\n"
            message += f"🔥 Trend (EMA): {trend}\n"
            message += f"📈 RSI-14: {rsi:.2f}\n"
            message += f"💹 SMC Trend: {smc}\n"
            message += f"📊 Volume: {volume}\n"
            message += f"🛡 Support: {sr_levels['support'] if sr_levels else 'N/A'}\n"
            message += f"⛔ Resistance: {sr_levels['resistance'] if sr_levels else 'N/A'}\n"
            message += f"💰 Current Price: {price}\n\n"

            message += f"💎 Trade Levels ({trade_levels['direction']}):\n"
            message += f"• Entry: {trade_levels['entry_price']}\n"
            message += f"• Take Profit: {trade_levels['take_profit']} ({trade_levels['tp_percent']}%)\n"
            message += f"• Stop Loss: {trade_levels['stop_loss']} ({trade_levels['sl_percent']}%)\n"
            message += f"• Confidence: {confidence}%\n"

            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error analyzing {symbol}: {str(e)}")
    else:
        await update.message.reply_text("❌ Tolong masukkan simbol koin, contoh: /coin BTCUSDT")

def run_telegram_bot():
    """Jalankan Telegram bot di thread terpisah"""
    if BOT_TOKEN:
        app_bot = ApplicationBuilder().token(BOT_TOKEN).build()

        app_bot.add_handler(CommandHandler("start", start))
        app_bot.add_handler(CommandHandler("stop", stop))
        app_bot.add_handler(CommandHandler("print", print_command))
        app_bot.add_handler(CommandHandler("coin", coin))

        print("Telegram bot running...")
        app_bot.run_polling()
    else:
        print("BOT_TOKEN tidak ditemukan, Telegram bot tidak dijalankan")

# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def main():
    port = int(os.environ.get("PORT", 5000))
    
    # Jalankan Telegram bot di thread terpisah jika BOT_TOKEN ada
    if BOT_TOKEN:
        bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
        bot_thread.start()
        print(f"Telegram bot started in separate thread")
    
    # Jalankan Flask app
    print(f"Starting Flask web service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()
