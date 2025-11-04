import os
import asyncio
import pandas as pd
import numpy as np
from binance.client import Client
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import flask
from flask import request
import threading
import warnings
warnings.filterwarnings('ignore')

# Flask app untuk render.com
app = flask.Flask(__name__)

@app.route('/')
def index():
    return "Bot Trading is Running!"

@app.route('/health')
def health():
    return "OK"

# Konfigurasi
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'your_api_key_here')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', 'your_secret_key_here')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_telegram_bot_token_here')

# Inisialisasi client Binance
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

class TechnicalIndicators:
    @staticmethod
    def ema(data, period):
        """Menghitung Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(data, period):
        """Menghitung Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data, period=14):
        """Menghitung Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Handle division by zero
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Default to 50 if NaN

class TradingAnalyzer:
    def __init__(self):
        self.timeframes = ['1W', '4H', '1H', '30M']
        self.indicators = TechnicalIndicators()
        
    def get_data(self, symbol, timeframe):
        """Mengambil data dari Binance"""
        try:
            # Map timeframe ke interval Binance
            timeframe_map = {
                '1W': Client.KLINE_INTERVAL_1WEEK,
                '4H': Client.KLINE_INTERVAL_4HOUR,
                '1H': Client.KLINE_INTERVAL_1HOUR,
                '30M': Client.KLINE_INTERVAL_30MINUTE
            }
            
            interval = timeframe_map.get(timeframe, Client.KLINE_INTERVAL_1HOUR)
            klines = client.get_klines(symbol=symbol, interval=interval, limit=100)
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Konversi tipe data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            print(f"Error getting data for {symbol} {timeframe}: {e}")
            return None

    def calculate_indicators(self, df):
        """Menghitung semua indikator teknikal secara manual"""
        if df is None or len(df) < 20:
            return None
            
        try:
            # EMA
            df['ema_9'] = self.indicators.ema(df['close'], 9)
            df['ema_21'] = self.indicators.ema(df['close'], 21)
            df['ema_50'] = self.indicators.ema(df['close'], 50)
            
            # RSI
            df['rsi'] = self.indicators.rsi(df['close'], 14)
            
            # Moving Averages
            df['ma_5'] = self.indicators.sma(df['close'], 5)
            df['ma_10'] = self.indicators.sma(df['close'], 10)
            
            # Volume
            df['volume_sma'] = self.indicators.sma(df['volume'], 20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
            
            return df
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None

    def detect_fvg(self, df, lookback=10):
        """Mendeteksi Fair Value Gap"""
        try:
            fvg_bullish = []
            fvg_bearish = []
            
            for i in range(2, min(lookback, len(df))):
                current_idx = -i
                prev_idx = -(i-1)
                
                # Bullish FVG: Candle bearish diikuti candle bullish dengan gap
                if (df['close'].iloc[current_idx] < df['open'].iloc[current_idx] and  # Bearish candle
                    df['close'].iloc[prev_idx] > df['open'].iloc[prev_idx] and  # Bullish candle berikutnya
                    df['low'].iloc[prev_idx] > df['high'].iloc[current_idx]):  # Gap
                    fvg_bullish.append({
                        'price_level': (df['high'].iloc[current_idx] + df['low'].iloc[prev_idx]) / 2,
                        'strength': abs(df['close'].iloc[prev_idx] - df['open'].iloc[prev_idx])
                    })
                
                # Bearish FVG: Candle bullish diikuti candle bearish dengan gap
                elif (df['close'].iloc[current_idx] > df['open'].iloc[current_idx] and  # Bullish candle
                      df['close'].iloc[prev_idx] < df['open'].iloc[prev_idx] and  # Bearish candle berikutnya
                      df['high'].iloc[prev_idx] < df['low'].iloc[current_idx]):  # Gap
                    fvg_bearish.append({
                        'price_level': (df['low'].iloc[current_idx] + df['high'].iloc[prev_idx]) / 2,
                        'strength': abs(df['close'].iloc[prev_idx] - df['open'].iloc[prev_idx])
                    })
            
            return fvg_bullish[-3:], fvg_bearish[-3:]  # Return 3 terbaru
        except Exception as e:
            print(f"Error detecting FVG: {e}")
            return [], []

    def detect_liquidity_sweep(self, df, lookback=20):
        """Mendeteksi Liquidity Sweep"""
        try:
            sweeps = []
            
            for i in range(2, min(lookback, len(df))):
                current_idx = -i
                current_high = df['high'].iloc[current_idx]
                current_low = df['low'].iloc[current_idx]
                current_open = df['open'].iloc[current_idx]
                current_close = df['close'].iloc[current_idx]
                prev_high = df['high'].iloc[-(i+1)]
                prev_low = df['low'].iloc[-(i+1)]
                
                # Sweep ke atas
                if (current_high > prev_high and 
                    current_close < current_open and  # Bearish rejection
                    (current_high - max(current_open, current_close)) > 
                    (max(current_open, current_close) - current_low) * 1.5):
                    sweeps.append({
                        'type': 'bullish_sweep',
                        'level': current_high,
                        'timestamp': df['timestamp'].iloc[current_idx]
                    })
                
                # Sweep ke bawah
                elif (current_low < prev_low and 
                      current_close > current_open and  # Bullish rejection
                      (min(current_open, current_close) - current_low) > 
                      (current_high - min(current_open, current_close)) * 1.5):
                    sweeps.append({
                        'type': 'bearish_sweep',
                        'level': current_low,
                        'timestamp': df['timestamp'].iloc[current_idx]
                    })
            
            return sweeps[-2:]  # Return 2 terbaru
        except Exception as e:
            print(f"Error detecting liquidity sweep: {e}")
            return []

    def detect_order_block(self, df, lookback=20):
        """Mendeteksi Order Block"""
        try:
            order_blocks = []
            
            for i in range(2, min(lookback, len(df))):
                current_idx = -i
                next_idx = -(i-1)
                
                # Bullish order block: Bearish candle diikuti bullish candle
                if (df['close'].iloc[current_idx] < df['open'].iloc[current_idx] and  # Bearish
                    df['close'].iloc[next_idx] > df['open'].iloc[next_idx] and  # Bullish berikutnya
                    df['low'].iloc[next_idx] > df['low'].iloc[current_idx]):  # Higher low
                    order_blocks.append({
                        'type': 'bullish',
                        'high': df['high'].iloc[current_idx],
                        'low': df['low'].iloc[current_idx],
                        'strength': abs(df['close'].iloc[current_idx] - df['open'].iloc[current_idx])
                    })
                
                # Bearish order block: Bullish candle diikuti bearish candle
                elif (df['close'].iloc[current_idx] > df['open'].iloc[current_idx] and  # Bullish
                      df['close'].iloc[next_idx] < df['open'].iloc[next_idx] and  # Bearish berikutnya
                      df['high'].iloc[next_idx] < df['high'].iloc[current_idx]):  # Lower high
                    order_blocks.append({
                        'type': 'bearish',
                        'high': df['high'].iloc[current_idx],
                        'low': df['low'].iloc[current_idx],
                        'strength': abs(df['close'].iloc[current_idx] - df['open'].iloc[current_idx])
                    })
            
            return order_blocks[-3:]  # Return 3 terbaru
        except Exception as e:
            print(f"Error detecting order block: {e}")
            return []

    def analyze_candlestick_patterns(self, df):
        """Analisis pola candlestick"""
        try:
            patterns = []
            if len(df) < 2:
                return patterns
                
            current_close = df['close'].iloc[-1]
            current_open = df['open'].iloc[-1]
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            prev_open = df['open'].iloc[-2]
            prev_high = df['high'].iloc[-2]
            prev_low = df['low'].iloc[-2]
            
            # Doji
            body = abs(current_close - current_open)
            total_range = current_high - current_low
            if total_range > 0 and body / total_range < 0.1:
                patterns.append("DOJI")
            
            # Hammer
            if body > 0 and total_range > 0:
                lower_shadow = current_close - current_low if current_close > current_open else current_open - current_low
                upper_shadow = current_high - current_close if current_close > current_open else current_high - current_open
                
                if lower_shadow > 2 * body and upper_shadow < body * 0.5:
                    patterns.append("HAMMER")
                
                # Shooting Star
                if upper_shadow > 2 * body and lower_shadow < body * 0.5:
                    patterns.append("SHOOTING_STAR")
            
            # Engulfing
            if (current_close > current_open and prev_close < prev_open and
                current_open < prev_close and current_close > prev_open):
                patterns.append("BULLISH_ENGULFING")
            elif (current_close < current_open and prev_close > prev_open and
                  current_open > prev_close and current_close < prev_open):
                patterns.append("BEARISH_ENGULFING")
            
            return patterns
        except Exception as e:
            print(f"Error analyzing candlestick patterns: {e}")
            return []

    def detect_double_top_bottom(self, df, lookback=30):
        """Mendeteksi Double Top/Bottom"""
        try:
            if len(df) < lookback:
                return False, False
                
            highs = df['high'].tail(lookback)
            lows = df['low'].tail(lookback)
            
            # Cari local maxima dan minima
            local_max = []
            local_min = []
            
            for i in range(1, len(highs)-1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    local_max.append(highs.iloc[i])
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    local_min.append(lows.iloc[i])
            
            # Double Top
            double_top = False
            if len(local_max) >= 2:
                max1, max2 = sorted(local_max[-2:], reverse=True)
                if abs(max1 - max2) / max1 < 0.02:  # Dalam 2%
                    double_top = True
            
            # Double Bottom
            double_bottom = False
            if len(local_min) >= 2:
                min1, min2 = sorted(local_min[-2:])
                if abs(min1 - min2) / min1 < 0.02:  # Dalam 2%
                    double_bottom = True
            
            return double_top, double_bottom
        except Exception as e:
            print(f"Error detecting double top/bottom: {e}")
            return False, False

    def analyze_trend(self, df):
        """Analisis trend berdasarkan EMA"""
        try:
            if len(df) < 2:
                return "UNKNOWN"
                
            current_price = df['close'].iloc[-1]
            ema_9 = df['ema_9'].iloc[-1] if not pd.isna(df['ema_9'].iloc[-1]) else current_price
            ema_21 = df['ema_21'].iloc[-1] if not pd.isna(df['ema_21'].iloc[-1]) else current_price
            ema_50 = df['ema_50'].iloc[-1] if not pd.isna(df['ema_50'].iloc[-1]) else current_price
            
            if current_price > ema_9 > ema_21 > ema_50:
                return "STRONG_BULLISH"
            elif current_price < ema_9 < ema_21 < ema_50:
                return "STRONG_BEARISH"
            elif ema_9 > ema_21 > ema_50:
                return "BULLISH"
            elif ema_9 < ema_21 < ema_50:
                return "BEARISH"
            else:
                return "SIDEWAYS"
        except Exception as e:
            print(f"Error analyzing trend: {e}")
            return "UNKNOWN"

    def generate_analysis(self, symbol):
        """Generate analisis lengkap untuk semua timeframe"""
        analysis_result = f"📊 **ANALYSIS FOR {symbol}**\n\n"
        
        for timeframe in self.timeframes:
            analysis_result += f"**⏰ {timeframe} TIMEFRAME**\n"
            analysis_result += "─" * 40 + "\n"
            
            # Get data
            df = self.get_data(symbol, timeframe)
            if df is None or len(df) < 50:
                analysis_result += "❌ Data tidak cukup untuk analisis\n\n"
                continue
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            if df is None:
                analysis_result += "❌ Error dalam perhitungan indikator\n\n"
                continue
            
            try:
                # Current price info
                current_price = df['close'].iloc[-1]
                price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                
                analysis_result += f"💰 Price: ${current_price:.4f} ({price_change:+.2f}%)\n"
                
                # Trend Analysis
                trend = self.analyze_trend(df)
                trend_emoji = "🟢" if "BULL" in trend else "🔴" if "BEAR" in trend else "🟡"
                analysis_result += f"📈 Trend: {trend_emoji} {trend}\n"
                
                # RSI
                rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
                rsi_status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
                analysis_result += f"📊 RSI: {rsi:.1f} ({rsi_status})\n"
                
                # Moving Averages
                ma_5 = df['ma_5'].iloc[-1] if not pd.isna(df['ma_5'].iloc[-1]) else current_price
                ma_10 = df['ma_10'].iloc[-1] if not pd.isna(df['ma_10'].iloc[-1]) else current_price
                analysis_result += f"📏 MA5: ${ma_5:.4f} | MA10: ${ma_10:.4f}\n"
                
                # Volume
                volume_ratio = df['volume_ratio'].iloc[-1] if not pd.isna(df['volume_ratio'].iloc[-1]) else 1
                volume_status = "HIGH 📈" if volume_ratio > 1.5 else "LOW 📉" if volume_ratio < 0.5 else "NORMAL ➡️"
                analysis_result += f"📦 Volume: {volume_status} ({volume_ratio:.2f}x)\n"
                
                # Smart Money Concepts
                fvg_bullish, fvg_bearish = self.detect_fvg(df)
                if fvg_bullish:
                    analysis_result += f"🟢 FVG Bullish: {len(fvg_bullish)} terdeteksi\n"
                if fvg_bearish:
                    analysis_result += f"🔴 FVG Bearish: {len(fvg_bearish)} terdeteksi\n"
                
                liquidity_sweeps = self.detect_liquidity_sweep(df)
                if liquidity_sweeps:
                    analysis_result += f"💧 Liquidity Sweep: {len(liquidity_sweeps)} terdeteksi\n"
                
                order_blocks = self.detect_order_block(df)
                if order_blocks:
                    bull_blocks = len([ob for ob in order_blocks if ob['type'] == 'bullish'])
                    bear_blocks = len([ob for ob in order_blocks if ob['type'] == 'bearish'])
                    analysis_result += f"📦 Order Blocks: 🟢{bull_blocks} 🔴{bear_blocks}\n"
                
                # Candlestick Patterns
                patterns = self.analyze_candlestick_patterns(df)
                if patterns:
                    analysis_result += f"🕯️ Patterns: {', '.join(patterns)}\n"
                
                # Double Top/Bottom
                double_top, double_bottom = self.detect_double_top_bottom(df)
                if double_top:
                    analysis_result += "⛰️ DOUBLE TOP Terdeteksi\n"
                if double_bottom:
                    analysis_result += "🏞️ DOUBLE BOTTOM Terdeteksi\n"
                
                analysis_result += "\n"
                
            except Exception as e:
                analysis_result += f"❌ Error dalam analisis: {str(e)}\n\n"
                continue
        
        return analysis_result

# Global analyzer instance
analyzer = TradingAnalyzer()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /start"""
    welcome_text = """
🤖 **Binance Trading Bot**

Commands:
/coin [symbol1] [symbol2] ... - Analisis multiple coins
Example: /coin BTCUSDT ETHUSDT ADAUSDT

Supported timeframes: 1W, 4H, 1H, 30M

Indicators included:
• EMA Trend Analysis (9, 21, 50)
• RSI (14)
• Moving Averages (MA5, MA10)
• Volume Analysis
• Smart Money Concepts:
  - Fair Value Gap (FVG)
  - Liquidity Sweep
  - Order Blocks
• Candlestick Patterns
• Double Top/Bottom
    """
    await update.message.reply_text(welcome_text)

async def analyze_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /coin"""
    if not context.args:
        await update.message.reply_text("❌ Please provide coin symbols. Example: /coin BTCUSDT ETHUSDT")
        return
    
    symbols = [sym.upper() for sym in context.args]
    
    for symbol in symbols:
        try:
            # Validasi symbol
            client.get_symbol_ticker(symbol=symbol)
            
            await update.message.reply_text(f"🔄 Analyzing {symbol}...")
            
            # Generate analysis
            analysis = analyzer.generate_analysis(symbol)
            
            # Split long messages (Telegram limit)
            if len(analysis) > 4096:
                for i in range(0, len(analysis), 4096):
                    await update.message.reply_text(analysis[i:i+4096])
                    await asyncio.sleep(0.5)
            else:
                await update.message.reply_text(analysis)
                
            await asyncio.sleep(1)  # Delay antara coins
            
        except Exception as e:
            error_msg = f"❌ Error analyzing {symbol}: {str(e)}"
            await update.message.reply_text(error_msg)
            continue

def run_flask():
    """Jalankan Flask server"""
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

async def main():
    """Main function"""
    # Jalankan Flask di thread terpisah
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Inisialisasi bot Telegram
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("coin", analyze_coin))
    
    # Start bot
    print("Bot started!")
    await application.run_polling()

if __name__ == "__main__":
    # Untuk kompatibilitas dengan Render.com
    port = int(os.environ.get('PORT', 5000))
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)).start()
    asyncio.run(main())
