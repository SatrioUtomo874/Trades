# app.py
import pandas as pd
import numpy as np
from binance.client import Client
import os
from dotenv import load_dotenv
from flask import Flask, jsonify
import time
import logging
import asyncio
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Flask app untuk web service
app = Flask(__name__)

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Inisialisasi Binance client dengan error handling
try:
    client = Client(api_key=API_KEY, api_secret=API_SECRET)
    logger.info("✅ Binance client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Binance client: {e}")
    client = None

# -------------------------------------------------------------------
# ✅ Helper — Fetch klines dengan error handling
# -------------------------------------------------------------------
def get_klines(symbol: str, interval: str, limit: int = 500):
    if client is None:
        logger.error("Binance client not initialized")
        return None
        
    try:
        raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)

        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
        ])

        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)

        return df
    except Exception as e:
        logger.error(f"Error getting klines for {symbol}: {e}")
        return None

# -------------------------------------------------------------------
# ✅ 1. TREND DETECTOR (EMA50 vs EMA200)
# -------------------------------------------------------------------
def get_trend(symbol: str, interval: str = "1h", limit: int = 500):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        return "ERROR"

    try:
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
    except Exception as e:
        logger.error(f"Error calculating trend for {symbol}: {e}")
        return "ERROR"

# -------------------------------------------------------------------
# ✅ 2. RSI (default RSI-14)
# -------------------------------------------------------------------
def get_rsi(symbol: str, interval: str = "1h", period: int = 14, limit: int = 200):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        return 50  # Default value jika error

    try:
        delta = df["close"].diff()

        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))

        return float(RSI.iloc[-1])
    except Exception as e:
        logger.error(f"Error calculating RSI for {symbol}: {e}")
        return 50

# -------------------------------------------------------------------
# ✅ 3. SMC TREND (Swing, BOS, CHoCH)
# -------------------------------------------------------------------
def find_swing_high_low(df, left=2, right=2):
    if df is None or len(df) == 0:
        return [], []

    swing_high = []
    swing_low = []

    try:
        for i in range(left, len(df) - right):
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]

            if high == max(df["high"].iloc[i-left:i+right+1]):
                swing_high.append(i)

            if low == min(df["low"].iloc[i-left:i+right+1]):
                swing_low.append(i)
    except Exception as e:
        logger.error(f"Error finding swing highs/lows: {e}")

    return swing_high, swing_low

def detect_structure(df, swing_high, swing_low):
    if df is None or len(df) == 0:
        return None, None

    try:
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
    except Exception as e:
        logger.error(f"Error detecting structure: {e}")
        return None, None

def get_trend_smc(symbol: str, interval: str = "1h", limit: int = 500):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        return "ERROR"

    try:
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
    except Exception as e:
        logger.error(f"Error calculating SMC trend for {symbol}: {e}")
        return "ERROR"

def get_support_resistance(symbol: str, interval: str = "1h", lookback: int = 50):
    """
    Cari support & resistance dari swing high & low
    """
    df = get_klines(symbol, interval, lookback)
    if df is None or len(df) == 0:
        return None

    try:
        highs = df["high"].values
        lows = df["low"].values

        resistance = np.max(highs)  # titik tertinggi → resistance
        support = np.min(lows)      # titik terendah → support

        return {"support": support, "resistance": resistance}
    except Exception as e:
        logger.error(f"Error calculating support/resistance for {symbol}: {e}")
        return None

def get_volume(symbol: str, interval: str = "1h", limit: int = 1):
    """
    Mengambil volume terakhir dari symbol di timeframe tertentu.
    Return: float (volume)
    """
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        return None

    try:
        return float(df["volume"].iloc[-1])
    except Exception as e:
        logger.error(f"Error getting volume for {symbol}: {e}")
        return None

def get_volume_average(symbol: str, interval: str = "1h", limit: int = 20):
    """
    Mengambil rata-rata volume untuk perbandingan
    """
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        return None

    try:
        volumes = df["volume"].astype(float)
        return float(volumes.mean())
    except Exception as e:
        logger.error(f"Error calculating volume average for {symbol}: {e}")
        return None

def get_current_price(symbol: str):
    """
    Ambil harga terakhir (last price) dari symbol
    Return: float
    """
    if client is None:
        return None
        
    try:
        data = client.get_symbol_ticker(symbol=symbol)
        return float(data['price'])
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        return None

def get_trade_levels_with_percent(symbol: str, interval: str = "15m", buffer=0.001, sl_buffer=0.0015):
    """
    Hitung entry, TP, SL untuk scalping dengan pendekatan realistis
    """
    try:
        df = get_klines(symbol, interval, limit=100)
        if df is None or len(df) == 0:
            return {
                "entry_price": None,
                "take_profit": None,
                "stop_loss": None,
                "tp_percent": None,
                "sl_percent": None,
                "direction": "ERROR"
            }

        trend = get_trend(symbol, interval)
        smc = get_trend_smc(symbol, interval)
        current_price = get_current_price(symbol)
        
        if current_price is None:
            return {
                "entry_price": None,
                "take_profit": None,
                "stop_loss": None,
                "tp_percent": None,
                "sl_percent": None,
                "direction": "ERROR"
            }

        # Tentukan level sederhana berdasarkan trend
        if "UPTREND" in trend or "UPTREND" in smc:
            entry = current_price * (1 + buffer)
            tp = current_price * 1.005  # 0.5% profit
            sl = current_price * (1 - sl_buffer)  # 0.15% stop loss
            direction = "LONG"
        elif "DOWNTREND" in trend or "DOWNTREND" in smc:
            entry = current_price * (1 - buffer)
            tp = current_price * 0.995  # 0.5% profit  
            sl = current_price * (1 + sl_buffer)  # 0.15% stop loss
            direction = "SHORT"
        else:
            entry = current_price
            tp = current_price * 1.002  # small profit for sideways
            sl = current_price * 0.998  # small stop loss for sideways
            direction = "SIDEWAYS"

        tp_percent = ((tp - entry) / entry * 100) if tp else None
        sl_percent = ((entry - sl) / entry * 100) if sl else None
        if direction == "SHORT":
            tp_percent = ((entry - tp) / entry * 100) if tp else None
            sl_percent = ((sl - entry) / entry * 100) if sl else None

        return {
            "entry_price": round(entry, 8) if entry else None,
            "take_profit": round(tp, 8) if tp else None,
            "stop_loss": round(sl, 8) if sl else None,
            "tp_percent": round(tp_percent, 2) if tp_percent else None,
            "sl_percent": round(sl_percent, 2) if sl_percent else None,
            "direction": direction
        }
    except Exception as e:
        logger.error(f"Error calculating trade levels for {symbol}: {e}")
        return {
            "entry_price": None,
            "take_profit": None,
            "stop_loss": None,
            "tp_percent": None,
            "sl_percent": None,
            "direction": "ERROR"
        }

# -------------------------------------------------------------------
# Telegram Bot Functions
# -------------------------------------------------------------------
def run_telegram_bot():
    """Jalankan Telegram bot dalam thread terpisah"""
    if not BOT_TOKEN:
        logger.warning("❌ BOT_TOKEN tidak ditemukan, Telegram bot tidak dijalankan")
        return

    try:
        from telegram import Update
        from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
        
        logger.info("🔄 Memulai Telegram bot...")
        
        # Buat application untuk bot
        application = ApplicationBuilder().token(BOT_TOKEN).build()
        
        async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handler untuk command /start"""
            await update.message.reply_text(
                "🤖 **Crypto Trading Bot Aktif!**\n\n"
                "Gunakan perintah:\n"
                "• /coin [SYMBOL] - Analisis trading coin\n"
                "• /price [SYMBOL] - Cek harga saat ini\n"
                "• /trend [SYMBOL] - Analisis trend\n\n"
                "Contoh: `/coin BTCUSDT`",
                parse_mode='Markdown'
            )
        
        async def coin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handler untuk command /coin"""
            if not context.args:
                await update.message.reply_text(
                    "❌ **Usage:** `/coin [SYMBOL]`\n"
                    "Contoh: `/coin BTCUSDT` atau `/coin ETHUSDT`",
                    parse_mode='Markdown'
                )
                return
            
            symbol = context.args[0].upper()
            
            try:
                # Kirim pesan sedang memproses
                processing_msg = await update.message.reply_text(f"🔄 Menganalisis {symbol}...")
                
                # Ambil data
                trend = get_trend(symbol, "1d")
                rsi = get_rsi(symbol, "15m")
                smc = get_trend_smc(symbol, "4h")
                volume = get_volume(symbol, "15m")
                sr_levels = get_support_resistance(symbol, "4h")
                price = get_current_price(symbol)
                trade_levels = get_trade_levels_with_percent(symbol, "5m")
                
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
                avg_volume = get_volume_average(symbol, "4h", limit=20)
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
                
                # Format output untuk Telegram
                message = f"📊 **{symbol} ANALYSIS**\n\n"
                message += f"💰 **Current Price:** ${price:,.2f}\n\n"
                
                message += f"**TREND ANALYSIS**\n"
                message += f"• EMA Trend: {trend}\n"
                message += f"• SMC Trend: {smc}\n"
                message += f"• RSI (14): {rsi:.2f}\n"
                message += f"• Volume: {volume:.2f}\n\n"
                
                message += f"**SUPPORT & RESISTANCE**\n"
                message += f"• Support: ${sr_levels['support']:,.2f}\n" if sr_levels else "• Support: N/A\n"
                message += f"• Resistance: ${sr_levels['resistance']:,.2f}\n\n" if sr_levels else "• Resistance: N/A\n\n"
                
                message += f"**TRADE SETUP** ({trade_levels['direction']})\n"
                message += f"• Entry: ${trade_levels['entry_price']:.2f}\n"
                if trade_levels['take_profit']:
                    message += f"• Take Profit: ${trade_levels['take_profit']:.2f} ({trade_levels['tp_percent']}%)\n"
                if trade_levels['stop_loss']:
                    message += f"• Stop Loss: ${trade_levels['stop_loss']:.2f} ({trade_levels['sl_percent']}%)\n"
                
                message += f"\n**CONFIDENCE LEVEL:** {confidence}%"
                
                # Hapus pesan processing dan kirim hasil
                await processing_msg.delete()
                await update.message.reply_text(message, parse_mode='Markdown')
                
            except Exception as e:
                error_msg = f"❌ Error analyzing {symbol}: {str(e)}"
                await update.message.reply_text(error_msg)
                logger.error(f"Telegram bot error: {e}")
        
        async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handler untuk command /price"""
            if not context.args:
                await update.message.reply_text("❌ Usage: /price [SYMBOL]")
                return
            
            symbol = context.args[0].upper()
            try:
                price = get_current_price(symbol)
                if price:
                    await update.message.reply_text(f"💰 **{symbol} Price:** ${price:,.2f}", parse_mode='Markdown')
                else:
                    await update.message.reply_text(f"❌ Cannot get price for {symbol}")
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {str(e)}")
        
        async def trend_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handler untuk command /trend"""
            if not context.args:
                await update.message.reply_text("❌ Usage: /trend [SYMBOL]")
                return
            
            symbol = context.args[0].upper()
            try:
                trend_ema = get_trend(symbol, "1h")
                trend_smc = get_trend_smc(symbol, "1h")
                rsi = get_rsi(symbol, "1h")
                
                message = f"📈 **{symbol} TREND ANALYSIS**\n\n"
                message += f"• EMA Trend: {trend_ema}\n"
                message += f"• SMC Trend: {trend_smc}\n"
                message += f"• RSI: {rsi:.2f}"
                
                await update.message.reply_text(message, parse_mode='Markdown')
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {str(e)}")
        
        # Tambahkan handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("coin", coin_command))
        application.add_handler(CommandHandler("price", price_command))
        application.add_handler(CommandHandler("trend", trend_command))
        
        # Jalankan bot
        logger.info("✅ Telegram bot started successfully")
        application.run_polling()
        
    except ImportError as e:
        logger.error(f"❌ Telegram library not installed: {e}")
    except Exception as e:
        logger.error(f"❌ Telegram bot error: {e}")

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
            "/price/<symbol>": "Get current price",
            "/trend/<symbol>": "Get trend analysis"
        },
        "timestamp": time.time()
    })

@app.route('/health')
def health():
    status = "healthy" if client else "unhealthy"
    return jsonify({
        "status": status,
        "timestamp": time.time(),
        "binance_connected": client is not None,
        "telegram_bot": BOT_TOKEN is not None
    })

@app.route('/analysis/<symbol>')
def analysis(symbol):
    try:
        symbol = symbol.upper()
        
        # Ambil data dengan error handling
        trend = get_trend(symbol, "1d")
        rsi = get_rsi(symbol, "15m")
        smc = get_trend_smc(symbol, "4h")
        volume = get_volume(symbol, "15m")
        sr_levels = get_support_resistance(symbol, "4h")
        price = get_current_price(symbol)
        trade_levels = get_trade_levels_with_percent(symbol, "5m")
        
        # Hitung confidence
        confidence = 0
        
        try:
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
            avg_volume = get_volume_average(symbol, "4h", limit=20)
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
        except Exception as e:
            logger.error(f"Error calculating confidence for {symbol}: {e}")
            confidence = 50  # Default confidence
        
        response = {
            "symbol": symbol,
            "analysis": {
                "trend_ema": trend,
                "rsi": round(rsi, 2) if rsi else None,
                "trend_smc": smc,
                "volume": volume,
                "current_price": price,
                "support": sr_levels["support"] if sr_levels else None,
                "resistance": sr_levels["resistance"] if sr_levels else None
            },
            "trade_levels": trade_levels,
            "confidence": round(confidence, 2),
            "timestamp": time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analysis endpoint for {symbol}: {e}")
        return jsonify({
            "error": str(e), 
            "symbol": symbol,
            "timestamp": time.time()
        }), 500

@app.route('/price/<symbol>')
def price(symbol):
    try:
        price = get_current_price(symbol.upper())
        return jsonify({
            "symbol": symbol.upper(), 
            "price": price,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Error in price endpoint for {symbol}: {e}")
        return jsonify({
            "error": str(e), 
            "symbol": symbol.upper(),
            "timestamp": time.time()
        }), 500

@app.route('/trend/<symbol>')
def trend(symbol):
    try:
        trend_ema = get_trend(symbol.upper(), "1h")
        trend_smc = get_trend_smc(symbol.upper(), "1h")
        rsi = get_rsi(symbol.upper(), "1h")
        
        return jsonify({
            "symbol": symbol.upper(),
            "trend_ema": trend_ema,
            "trend_smc": trend_smc,
            "rsi": rsi,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Error in trend endpoint for {symbol}: {e}")
        return jsonify({
            "error": str(e), 
            "symbol": symbol.upper(),
            "timestamp": time.time()
        }), 500

@app.route('/test')
def test():
    """Endpoint untuk testing koneksi dan fungsi dasar"""
    try:
        # Test Binance connection
        price = get_current_price("BTCUSDT")
        
        return jsonify({
            "status": "success",
            "binance_connected": client is not None,
            "telegram_bot_available": BOT_TOKEN is not None,
            "btc_price": price,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }), 500

# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def main():
    port = int(os.environ.get("PORT", 5000))
    
    # Test connection
    try:
        if client:
            # Simple API call to test connection
            client.get_symbol_ticker(symbol="BTCUSDT")
            logger.info("✅ Binance connection successful")
        else:
            logger.warning("⚠️ Binance client not available")
    except Exception as e:
        logger.error(f"❌ Binance connection failed: {e}")
    
    # Jalankan Telegram bot di thread terpisah
    if BOT_TOKEN:
        try:
            bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
            bot_thread.start()
            logger.info("✅ Telegram bot started in separate thread")
        except Exception as e:
            logger.error(f"❌ Failed to start Telegram bot: {e}")
    else:
        logger.warning("⚠️ BOT_TOKEN not found, Telegram bot disabled")
    
    # Jalankan Flask app
    logger.info(f"🚀 Starting Flask web service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()
