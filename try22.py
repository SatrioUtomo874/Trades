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
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Flask app untuk web service
app = Flask(__name__)

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Inisialisasi Binance client dengan error handling
try:
    client = Client(api_key=API_KEY, api_secret=API_SECRET, testnet=False)
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

        # Convert to float
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df
    except Exception as e:
        logger.error(f"Error getting klines for {symbol}: {e}")
        return None

# -------------------------------------------------------------------
# ✅ 1. TREND DETECTOR (EMA50 vs EMA200)
# -------------------------------------------------------------------
def get_trend(symbol: str, interval: str = "1h", limit: int = 500):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) < 200:
        return "ERROR: Insufficient data"

    try:
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

        ema50 = df["ema50"].iloc[-1]
        ema200 = df["ema200"].iloc[-1]

        if ema50 > ema200 * 1.01:
            return "🟢 UPTREND"
        elif ema50 < ema200 * 0.99:
            return "🔴 DOWNTREND"
        else:
            return "🟡 SIDEWAYS"
    except Exception as e:
        logger.error(f"Error calculating trend for {symbol}: {e}")
        return f"ERROR: {str(e)}"

# -------------------------------------------------------------------
# ✅ 2. RSI (default RSI-14)
# -------------------------------------------------------------------
def get_rsi(symbol: str, interval: str = "1h", period: int = 14, limit: int = 200):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) < period:
        return 50  # Default value jika error

    try:
        delta = df["close"].diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

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
        return "ERROR: No data"

    try:
        swing_high, swing_low = find_swing_high_low(df)
        bos, choch = detect_structure(df, swing_high, swing_low)

        # Priority: CHoCH > BOS > Sideways
        if choch == "CHOCH_UP":
            return "🟢 UPTREND (CHOCH-UP)"
        if choch == "CHOCH_DOWN":
            return "🔴 DOWNTREND (CHOCH-DOWN)"
        if bos == "BOS_UP":
            return "🟢 UPTREND"
        if bos == "BOS_DOWN":
            return "🔴 DOWNTREND"

        return "🟡 SIDEWAYS"
    except Exception as e:
        logger.error(f"Error calculating SMC trend for {symbol}: {e}")
        return f"ERROR: {str(e)}"

def get_support_resistance(symbol: str, interval: str = "1h", lookback: int = 50):
    """Cari support & resistance dari swing high & low"""
    df = get_klines(symbol, interval, lookback)
    if df is None or len(df) == 0:
        return None

    try:
        highs = df["high"].values
        lows = df["low"].values

        resistance = np.max(highs)
        support = np.min(lows)

        return {
            "support": round(support, 4),
            "resistance": round(resistance, 4)
        }
    except Exception as e:
        logger.error(f"Error calculating support/resistance for {symbol}: {e}")
        return None

def get_volume(symbol: str, interval: str = "1h", limit: int = 1):
    """Mengambil volume terakhir dari symbol"""
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        return None

    try:
        return float(df["volume"].iloc[-1])
    except Exception as e:
        logger.error(f"Error getting volume for {symbol}: {e}")
        return None

def get_volume_average(symbol: str, interval: str = "1h", limit: int = 20):
    """Mengambil rata-rata volume untuk perbandingan"""
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
    """Ambil harga terakhir (last price) dari symbol"""
    if client is None:
        return None
        
    try:
        data = client.get_symbol_ticker(symbol=symbol)
        return float(data['price'])
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        return None

def calculate_confidence(symbol: str, trend: str, smc: str, rsi: float, volume: float):
    """Hitung confidence level untuk trading"""
    confidence = 0
    
    try:
        # 1️⃣ Trend EMA (30 points)
        if "UPTREND" in trend:
            confidence += 30
        elif "DOWNTREND" in trend:
            confidence += 20

        # 2️⃣ Trend SMC (30 points)
        if "UPTREND" in smc:
            confidence += 30
        elif "DOWNTREND" in smc:
            confidence += 15

        # 3️⃣ Volume (20 points)
        avg_volume = get_volume_average(symbol, "4h", limit=20)
        if volume and avg_volume:
            if volume >= avg_volume * 1.2:  # Volume 20% di atas rata-rata
                confidence += 20
            elif volume >= avg_volume * 0.8:  # Volume minimal 80% dari rata-rata
                confidence += 15
            else:
                confidence += 5

        # 4️⃣ RSI (20 points)
        if rsi < 30:  # oversold
            confidence += 5
        elif rsi > 70:  # overbought
            confidence -= 5
        elif 40 <= rsi <= 60:  # zona netral optimal
            confidence += 20
        else:
            confidence += 10

        # Clamp antara 0 - 100
        confidence = max(0, min(confidence, 100))
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        confidence = 50  # Default
        
    return confidence

def get_trade_levels(symbol: str, interval: str = "15m"):
    """Hitung entry, TP, SL untuk trading"""
    try:
        current_price = get_current_price(symbol)
        if current_price is None:
            return error_trade_levels()

        trend = get_trend(symbol, interval)
        smc = get_trend_smc(symbol, interval)

        # Tentukan arah trading
        if "UPTREND" in trend or "UPTREND" in smc:
            direction = "LONG"
            entry = current_price * 1.001  # 0.1% di atas current price
            tp = current_price * 1.015    # 1.5% profit
            sl = current_price * 0.995    # 0.5% stop loss
        elif "DOWNTREND" in trend or "DOWNTREND" in smc:
            direction = "SHORT" 
            entry = current_price * 0.999  # 0.1% di bawah current price
            tp = current_price * 0.985    # 1.5% profit
            sl = current_price * 1.005    # 0.5% stop loss
        else:
            direction = "SIDEWAYS"
            entry = current_price
            tp = current_price * 1.008    # 0.8% profit kecil
            sl = current_price * 0.992    # 0.8% stop loss

        # Hitung persentase
        if direction == "LONG":
            tp_percent = ((tp - entry) / entry) * 100
            sl_percent = ((entry - sl) / entry) * 100
        else:  # SHORT
            tp_percent = ((entry - tp) / entry) * 100
            sl_percent = ((sl - entry) / entry) * 100

        return {
            "entry_price": round(entry, 4),
            "take_profit": round(tp, 4),
            "stop_loss": round(sl, 4),
            "tp_percent": round(tp_percent, 2),
            "sl_percent": round(sl_percent, 2),
            "direction": direction,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error calculating trade levels for {symbol}: {e}")
        return error_trade_levels()

def error_trade_levels():
    """Return trade levels ketika error"""
    return {
        "entry_price": None,
        "take_profit": None,
        "stop_loss": None,
        "tp_percent": None,
        "sl_percent": None,
        "direction": "ERROR",
        "success": False
    }

# -------------------------------------------------------------------
# Telegram Bot Functions
# -------------------------------------------------------------------
class TelegramBot:
    def __init__(self):
        self.application = None
        self.is_running = False
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk command /start"""
        welcome_text = """
🤖 **Crypto Trading Bot Aktif!**

**Perintah yang tersedia:**
/coin [SYMBOL] - Analisis trading lengkap
/price [SYMBOL] - Cek harga saat ini  
/trend [SYMBOL] - Analisis trend
/help - Menampilkan bantuan

**Contoh:**
`/coin BTCUSDT`
`/price ETHUSDT`
`/trend ADAUSDT`

Bot ini memberikan analisis teknikal berdasarkan:
• Trend EMA (50 vs 200)
• RSI (14 period)
• Smart Money Concept (SMC)
• Support & Resistance
        """
        await update.message.reply_text(welcome_text, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk command /help"""
        help_text = """
📖 **Panduan Penggunaan Bot**

**1. Analisis Coin (/coin)**
Memberikan analisis lengkap untuk trading:
- Trend berdasarkan EMA dan SMC
- Level RSI saat ini
- Support & Resistance
- Rekomendasi entry, TP, SL
- Confidence level

**2. Cek Harga (/price)**
Menampilkan harga real-time coin

**3. Analisis Trend (/trend)**
Analisis trend teknikal singkat

**Format Symbol:** BTCUSDT, ETHUSDT, ADAUSDT, dll.
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def coin_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk command /coin"""
        if not context.args:
            await update.message.reply_text(
                "❌ **Format salah!**\n"
                "Gunakan: `/coin [SYMBOL]`\n"
                "Contoh: `/coin BTCUSDT`",
                parse_mode='Markdown'
            )
            return
        
        symbol = context.args[0].upper().strip()
        
        try:
            # Kirim pesan sedang memproses
            processing_msg = await update.message.reply_text(f"🔍 Menganalisis **{symbol}**...", parse_mode='Markdown')
            
            # Ambil data
            trend_ema = get_trend(symbol, "1h")
            rsi = get_rsi(symbol, "15m")
            trend_smc = get_trend_smc(symbol, "4h")
            volume = get_volume(symbol, "1h")
            sr_levels = get_support_resistance(symbol, "4h")
            current_price = get_current_price(symbol)
            trade_levels = get_trade_levels(symbol, "15m")
            
            # Hitung confidence
            confidence = calculate_confidence(symbol, trend_ema, trend_smc, rsi, volume)
            
            # Format output untuk Telegram
            message = f"📊 **ANALISIS {symbol}**\n\n"
            
            # Harga saat ini
            if current_price:
                message += f"💰 **Harga Saat Ini:** `${current_price:,.4f}`\n\n"
            
            # Analisis Teknikal
            message += f"**📈 ANALISIS TEKNIKAL**\n"
            message += f"• **Trend EMA:** {trend_ema}\n"
            message += f"• **Trend SMC:** {trend_smc}\n"
            message += f"• **RSI (14):** `{rsi:.2f}`"
            
            # Warna RSI
            if rsi < 30:
                message += " 🟢 (Oversold)\n"
            elif rsi > 70:
                message += " 🔴 (Overbought)\n"
            else:
                message += " 🟡 (Netral)\n"
                
            if volume:
                message += f"• **Volume:** `{volume:,.0f}`\n"
            
            # Support & Resistance
            if sr_levels:
                message += f"• **Support:** `${sr_levels['support']}`\n"
                message += f"• **Resistance:** `${sr_levels['resistance']}`\n"
            
            message += f"\n**🎯 SETUP TRADING**\n"
            message += f"• **Arah:** {trade_levels['direction']}\n"
            
            if trade_levels['success']:
                message += f"• **Entry:** `${trade_levels['entry_price']}`\n"
                message += f"• **TP:** `${trade_levels['take_profit']}` (+{trade_levels['tp_percent']}%)\n"
                message += f"• **SL:** `${trade_levels['stop_loss']}` (-{trade_levels['sl_percent']}%)\n"
            
            # Confidence Level
            message += f"\n**📊 CONFIDENCE LEVEL**\n"
            if confidence >= 70:
                message += f"🟢 **{confidence}%** (Tinggi)\n"
            elif confidence >= 50:
                message += f"🟡 **{confidence}%** (Sedang)\n"
            else:
                message += f"🔴 **{confidence}%** (Rendah)\n"
            
            # Risk Disclaimer
            message += f"\n___\n"
            message += f"⚠️ *Disclaimer: Ini bukan financial advice. Trading mengandung risiko.*"
            
            # Hapus pesan processing dan kirim hasil
            await processing_msg.delete()
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            error_msg = f"❌ **Error menganalisis {symbol}:**\n`{str(e)}`"
            await update.message.reply_text(error_msg, parse_mode='Markdown')
            logger.error(f"Telegram bot error in coin_analysis: {e}")
    
    async def price_check(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk command /price"""
        if not context.args:
            await update.message.reply_text("❌ Gunakan: `/price [SYMBOL]`", parse_mode='Markdown')
            return
        
        symbol = context.args[0].upper().strip()
        try:
            price = get_current_price(symbol)
            if price:
                # Dapatkan perubahan harga 24 jam
                try:
                    ticker = client.get_24hr_ticker(symbol=symbol)
                    price_change = float(ticker['priceChange'])
                    price_change_percent = float(ticker['priceChangePercent'])
                    
                    change_emoji = "🟢" if price_change >= 0 else "🔴"
                    change_text = f"{change_emoji} {price_change_percent:+.2f}% (24h)"
                    
                    await update.message.reply_text(
                        f"💰 **{symbol}**\n"
                        f"**Harga:** `${price:,.4f}`\n"
                        f"**Perubahan:** {change_text}",
                        parse_mode='Markdown'
                    )
                except:
                    await update.message.reply_text(f"💰 **{symbol}:** `${price:,.4f}`", parse_mode='Markdown')
            else:
                await update.message.reply_text(f"❌ Tidak dapat mengambil harga untuk {symbol}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: `{str(e)}`", parse_mode='Markdown')
    
    async def trend_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk command /trend"""
        if not context.args:
            await update.message.reply_text("❌ Gunakan: `/trend [SYMBOL]`", parse_mode='Markdown')
            return
        
        symbol = context.args[0].upper().strip()
        try:
            processing_msg = await update.message.reply_text(f"🔍 Menganalisis trend **{symbol}**...", parse_mode='Markdown')
            
            trend_ema = get_trend(symbol, "1h")
            trend_smc = get_trend_smc(symbol, "4h")
            rsi = get_rsi(symbol, "1h")
            current_price = get_current_price(symbol)
            
            message = f"📈 **TREND ANALYSIS - {symbol}**\n\n"
            
            if current_price:
                message += f"💰 **Harga:** `${current_price:,.4f}`\n\n"
            
            message += f"**Trend Indicators:**\n"
            message += f"• **EMA (1h):** {trend_ema}\n"
            message += f"• **SMC (4h):** {trend_smc}\n"
            message += f"• **RSI (1h):** `{rsi:.2f}`"
            
            # Analisis sederhana
            if "UPTREND" in trend_ema and "UPTREND" in trend_smc:
                message += f"\n\n🎯 **Kesimpulan:** Trend Naik kuat"
            elif "DOWNTREND" in trend_ema and "DOWNTREND" in trend_smc:
                message += f"\n\n🎯 **Kesimpulan:** Trend Turun kuat"
            else:
                message += f"\n\n🎯 **Kesimpulan:** Market Sideways/Ranging"
            
            await processing_msg.delete()
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: `{str(e)}`", parse_mode='Markdown')
    
    def run_bot(self):
        """Jalankan Telegram bot"""
        if not BOT_TOKEN:
            logger.warning("❌ BOT_TOKEN tidak ditemukan, Telegram bot tidak dijalankan")
            return
        
        try:
            # Buat application untuk bot
            self.application = ApplicationBuilder().token(BOT_TOKEN).build()
            
            # Tambahkan handlers
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("coin", self.coin_analysis))
            self.application.add_handler(CommandHandler("price", self.price_check))
            self.application.add_handler(CommandHandler("trend", self.trend_analysis))
            
            # Jalankan bot
            logger.info("✅ Telegram bot started successfully")
            self.is_running = True
            self.application.run_polling()
            
        except Exception as e:
            logger.error(f"❌ Telegram bot error: {e}")
            self.is_running = False

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
        "telegram_bot_available": BOT_TOKEN is not None
    })

@app.route('/analysis/<symbol>')
def analysis(symbol):
    try:
        symbol = symbol.upper()
        
        # Ambil data
        trend_ema = get_trend(symbol, "1h")
        rsi = get_rsi(symbol, "15m")
        trend_smc = get_trend_smc(symbol, "4h")
        volume = get_volume(symbol, "1h")
        sr_levels = get_support_resistance(symbol, "4h")
        current_price = get_current_price(symbol)
        trade_levels = get_trade_levels(symbol, "15m")
        
        # Hitung confidence
        confidence = calculate_confidence(symbol, trend_ema, trend_smc, rsi, volume)
        
        response = {
            "symbol": symbol,
            "analysis": {
                "trend_ema": trend_ema,
                "rsi": round(rsi, 2) if rsi else None,
                "trend_smc": trend_smc,
                "volume": volume,
                "current_price": current_price,
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
        trend_smc = get_trend_smc(symbol.upper(), "4h")
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
    """Endpoint untuk testing koneksi"""
    try:
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
    
    # Test koneksi Binance
    try:
        if client:
            client.get_symbol_ticker(symbol="BTCUSDT")
            logger.info("✅ Binance connection successful")
        else:
            logger.warning("⚠️ Binance client not available")
    except Exception as e:
        logger.error(f"❌ Binance connection failed: {e}")
    
    # Jalankan Telegram bot di thread terpisah
    if BOT_TOKEN:
        try:
            telegram_bot = TelegramBot()
            bot_thread = threading.Thread(target=telegram_bot.run_bot, daemon=True)
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
