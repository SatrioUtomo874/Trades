# bot.py - Telegram Bot (Jalankan di lokal/VPS terpisah)
import pandas as pd
import numpy as np
from binance.client import Client
import os
from dotenv import load_dotenv
import time
import logging
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Inisialisasi Binance client
try:
    client = Client(api_key=API_KEY, api_secret=API_SECRET, testnet=False)
    logger.info("✅ Binance client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Binance client: {e}")
    client = None

# -------------------------------------------------------------------
# FUNGSI ANALISIS (Sama seperti di app.py)
# -------------------------------------------------------------------
def get_klines(symbol: str, interval: str, limit: int = 500):
    if client is None:
        return None
        
    try:
        raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        logger.error(f"Error getting klines for {symbol}: {e}")
        return None

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
        return f"ERROR: {str(e)}"

def get_rsi(symbol: str, interval: str = "1h", period: int = 14, limit: int = 200):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) < period:
        return 50
    try:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))
        return float(RSI.iloc[-1])
    except Exception as e:
        return 50

def get_trend_smc(symbol: str, interval: str = "1h", limit: int = 500):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        return "ERROR: No data"
    try:
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        
        if len(closes) >= 3:
            if closes[-1] > closes[-2] > closes[-3] and lows[-1] > lows[-2]:
                return "🟢 UPTREND"
            elif closes[-1] < closes[-2] < closes[-3] and highs[-1] < highs[-2]:
                return "🔴 DOWNTREND"
        return "🟡 SIDEWAYS"
    except Exception as e:
        return f"ERROR: {str(e)}"

def get_support_resistance(symbol: str, interval: str = "1h", lookback: int = 50):
    df = get_klines(symbol, interval, lookback)
    if df is None or len(df) == 0:
        return None
    try:
        highs = df["high"].values
        lows = df["low"].values
        return {
            "support": round(np.min(lows), 4),
            "resistance": round(np.max(highs), 4)
        }
    except Exception as e:
        return None

def get_volume(symbol: str, interval: str = "1h", limit: int = 1):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        return None
    try:
        return float(df["volume"].iloc[-1])
    except Exception as e:
        return None

def get_volume_average(symbol: str, interval: str = "1h", limit: int = 20):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        return None
    try:
        volumes = df["volume"].astype(float)
        return float(volumes.mean())
    except Exception as e:
        return None

def get_current_price(symbol: str):
    if client is None:
        return None
    try:
        data = client.get_symbol_ticker(symbol=symbol)
        return float(data['price'])
    except Exception as e:
        return None

def calculate_confidence(symbol: str, trend: str, smc: str, rsi: float, volume: float):
    confidence = 0
    try:
        if "UPTREND" in trend:
            confidence += 30
        elif "DOWNTREND" in trend:
            confidence += 20

        if "UPTREND" in smc:
            confidence += 30
        elif "DOWNTREND" in smc:
            confidence += 15

        avg_volume = get_volume_average(symbol, "4h", limit=20)
        if volume and avg_volume:
            if volume >= avg_volume * 1.2:
                confidence += 20
            elif volume >= avg_volume * 0.8:
                confidence += 15
            else:
                confidence += 5

        if rsi < 30:
            confidence += 5
        elif rsi > 70:
            confidence -= 5
        elif 40 <= rsi <= 60:
            confidence += 20
        else:
            confidence += 10

        confidence = max(0, min(confidence, 100))
    except Exception as e:
        confidence = 50
    return confidence

def get_trade_levels(symbol: str, interval: str = "15m"):
    try:
        current_price = get_current_price(symbol)
        if current_price is None:
            return {"direction": "ERROR", "success": False}

        trend = get_trend(symbol, interval)
        smc = get_trend_smc(symbol, interval)

        if "UPTREND" in trend or "UPTREND" in smc:
            direction = "LONG"
            entry = current_price * 1.001
            tp = current_price * 1.015
            sl = current_price * 0.995
        elif "DOWNTREND" in trend or "DOWNTREND" in smc:
            direction = "SHORT" 
            entry = current_price * 0.999
            tp = current_price * 0.985
            sl = current_price * 1.005
        else:
            direction = "SIDEWAYS"
            entry = current_price
            tp = current_price * 1.008
            sl = current_price * 0.992

        if direction == "LONG":
            tp_percent = ((tp - entry) / entry) * 100
            sl_percent = ((entry - sl) / entry) * 100
        else:
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
        return {"direction": "ERROR", "success": False}

# -------------------------------------------------------------------
# TELEGRAM BOT HANDLERS
# -------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def coin_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /coin"""
    if not context.args:
        await update.message.reply_text(
            "❌ **Format salah!**\nGunakan: `/coin [SYMBOL]`\nContoh: `/coin BTCUSDT`",
            parse_mode='Markdown'
        )
        return
    
    symbol = context.args[0].upper().strip()
    
    try:
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
        
        if current_price:
            message += f"💰 **Harga Saat Ini:** `${current_price:,.4f}`\n\n"
        
        message += f"**📈 ANALISIS TEKNIKAL**\n"
        message += f"• **Trend EMA:** {trend_ema}\n"
        message += f"• **Trend SMC:** {trend_smc}\n"
        message += f"• **RSI (14):** `{rsi:.2f}`"
        
        if rsi < 30:
            message += " 🟢 (Oversold)\n"
        elif rsi > 70:
            message += " 🔴 (Overbought)\n"
        else:
            message += " 🟡 (Netral)\n"
            
        if volume:
            message += f"• **Volume:** `{volume:,.0f}`\n"
        
        if sr_levels:
            message += f"• **Support:** `${sr_levels['support']}`\n"
            message += f"• **Resistance:** `${sr_levels['resistance']}`\n"
        
        message += f"\n**🎯 SETUP TRADING**\n"
        message += f"• **Arah:** {trade_levels['direction']}\n"
        
        if trade_levels['success']:
            message += f"• **Entry:** `${trade_levels['entry_price']}`\n"
            message += f"• **TP:** `${trade_levels['take_profit']}` (+{trade_levels['tp_percent']}%)\n"
            message += f"• **SL:** `${trade_levels['stop_loss']}` (-{trade_levels['sl_percent']}%)\n"
        
        message += f"\n**📊 CONFIDENCE LEVEL**\n"
        if confidence >= 70:
            message += f"🟢 **{confidence}%** (Tinggi)\n"
        elif confidence >= 50:
            message += f"🟡 **{confidence}%** (Sedang)\n"
        else:
            message += f"🔴 **{confidence}%** (Rendah)\n"
        
        message += f"\n___\n"
        message += f"⚠️ *Disclaimer: Ini bukan financial advice. Trading mengandung risiko.*"
        
        await processing_msg.delete()
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        error_msg = f"❌ **Error menganalisis {symbol}:**\n`{str(e)}`"
        await update.message.reply_text(error_msg, parse_mode='Markdown')
        logger.error(f"Telegram bot error in coin_analysis: {e}")

async def price_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /price"""
    if not context.args:
        await update.message.reply_text("❌ Gunakan: `/price [SYMBOL]`", parse_mode='Markdown')
        return
    
    symbol = context.args[0].upper().strip()
    try:
        price = get_current_price(symbol)
        if price:
            await update.message.reply_text(f"💰 **{symbol}:** `${price:,.4f}`", parse_mode='Markdown')
        else:
            await update.message.reply_text(f"❌ Tidak dapat mengambil harga untuk {symbol}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{str(e)}`", parse_mode='Markdown')

async def trend_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

# -------------------------------------------------------------------
# MAIN BOT FUNCTION
# -------------------------------------------------------------------
def main():
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN tidak ditemukan!")
        return
    
    try:
        # Buat application untuk bot
        application = ApplicationBuilder().token(BOT_TOKEN).build()
        
        # Tambahkan handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("coin", coin_analysis))
        application.add_handler(CommandHandler("price", price_check))
        application.add_handler(CommandHandler("trend", trend_analysis))
        
        # Jalankan bot
        logger.info("✅ Telegram bot started successfully")
        print("Bot sedang berjalan... Tekan Ctrl+C untuk menghentikan.")
        application.run_polling()
        
    except Exception as e:
        logger.error(f"❌ Telegram bot error: {e}")

if __name__ == "__main__":
    import threading
    from webserver import app

    # Jalankan webserver di thread terpisah
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))).start()

    # Jalankan bot Telegram
    run_bot()


