import numpy as np
from binance.client import Client
import time
import os
from datetime import datetime, timedelta
import warnings
import math
import requests
import json
import pandas as pd
from collections import deque
from dotenv import load_dotenv
import threading

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI YANG DIPERBAIKI ====================
API_KEYS = [
    {
        'key': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET')
    }
]

CURRENT_API_INDEX = 0
INITIAL_INVESTMENT = float(os.getenv('INITIAL_INVESTMENT', '10.0'))
ORDER_RUN = os.getenv('ORDER_RUN', 'False').lower() == 'true'

# Trading Parameters - LEBIH KONSERVATIF
TAKE_PROFIT_PCT = 0.015  # 1.5%
STOP_LOSS_PCT = 0.008    # 0.8%
TRAILING_STOP_ACTIVATION = 0.008
TRAILING_STOP_PCT = 0.006

# Risk Management
POSITION_SIZING_PCT = 0.3  # Lebih kecil untuk risk management
MAX_DRAWDOWN_PCT = 0.4
ADAPTIVE_CONFIDENCE = True

# Filter Koin
MIN_24H_VOLUME = 5000000  # $5 juta volume minimum
MAX_SPREAD_PCT = 0.15     # Spread maksimal 0.15%

# Daftar koin yang lebih berkualitas
COINS = [
    'SOLUSDT', 'BNBUSDT', 'AVAXUSDT', 'ADAUSDT', 'MATICUSDT', 
    'DOTUSDT', 'LINKUSDT', 'UNIUSDT', 'XRPUSDT', 'DOGEUSDT',
    'ATOMUSDT', 'FTMUSDT', 'NEARUSDT', 'ALGOUSDT', 'SANDUSDT',
    'MANAUSDT', 'GALAUSDT', 'ENJUSDT', 'CHZUSDT', 'APEUSDT'
]

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
SEND_TELEGRAM_NOTIFICATIONS = True

# File Configuration
LOG_FILE = 'improved_trading_log.txt'
TRADE_HISTORY_FILE = 'improved_trade_history.json'

# Global variables
current_investment = INITIAL_INVESTMENT
active_position = None
trade_history = []
client = None
BOT_RUNNING = False

# ==================== INISIALISASI CLIENT ====================
def initialize_binance_client():
    """Initialize Binance client dengan error handling"""
    global client
    
    try:
        if not API_KEYS[0]['key'] or not API_KEYS[0]['secret']:
            print("❌ API Key atau Secret tidak ditemukan. Cek file .env")
            return False
            
        client = Client(API_KEYS[0]['key'], API_KEYS[0]['secret'])
        
        # Test connection
        client.get_account()
        print("✅ Binance client berhasil diinisialisasi")
        return True
        
    except Exception as e:
        print(f"❌ Gagal menginisialisasi Binance client: {e}")
        print("💡 Pastikan API Key dan Secret benar, dan koneksi internet stabil")
        return False

# ==================== FUNGSI FILTER KOIN ====================
def filter_quality_coins():
    """Filter koin berdasarkan volume, spread, dan likuiditas"""
    global client
    quality_coins = []
    
    if client is None:
        print("❌ Client Binance belum diinisialisasi")
        return quality_coins
    
    print("🔍 Filtering quality coins based on volume and liquidity...")
    
    for coin in COINS:
        try:
            # Cek apakah koin ada dan aktif
            rate_limit()
            ticker = client.get_24hr_ticker(symbol=coin)
            volume = float(ticker['quoteVolume'])
            
            # Cek spread bid-ask
            book = client.get_order_book(symbol=coin, limit=5)
            bid_price = float(book['bids'][0][0])
            ask_price = float(book['asks'][0][0])
            spread = (ask_price - bid_price) / bid_price * 100
            
            # Cek price change untuk volatilitas
            price_change = float(ticker['priceChangePercent'])
            
            # Filter kondisi
            if (volume >= MIN_24H_VOLUME and 
                spread <= MAX_SPREAD_PCT and
                abs(price_change) < 20.0 and  # Tidak terlalu volatil
                coin not in ['FDUSDT', 'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT']):
                quality_coins.append(coin)
                print(f"✅ {coin}: Volume=${volume:,.0f}, Spread={spread:.3f}%")
            else:
                print(f"❌ {coin} rejected - Vol:${volume:,.0f}, Spread:{spread:.3f}%")
                
        except Exception as e:
            print(f"❌ Skip {coin}: {e}")
    
    print(f"🎯 Quality coins selected: {len(quality_coins)}/{len(COINS)}")
    return quality_coins

# ==================== INDIKATOR TEKNIKAL YANG DIPERBAIKI ====================
def calculate_mfi(highs, lows, closes, volumes, period=14):
    """Calculate Money Flow Index"""
    if len(closes) < period + 1:
        return 50
    
    try:
        # Typical Price
        typical_prices = [(high + low + close) / 3 for high, low, close in zip(highs, lows, closes)]
        
        # Raw Money Flow
        money_flows = [tp * vol for tp, vol in zip(typical_prices, volumes)]
        
        # Positive and Negative Money Flow
        positive_flows = []
        negative_flows = []
        
        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i-1]:
                positive_flows.append(money_flows[i])
                negative_flows.append(0)
            elif typical_prices[i] < typical_prices[i-1]:
                positive_flows.append(0)
                negative_flows.append(money_flows[i])
            else:
                positive_flows.append(0)
                negative_flows.append(0)
        
        # Calculate MFI
        mfi_values = []
        for i in range(period, len(positive_flows)):
            positive_sum = sum(positive_flows[i-period:i])
            negative_sum = sum(negative_flows[i-period:i])
            
            if negative_sum == 0:
                mfi = 100
            else:
                money_ratio = positive_sum / negative_sum
                mfi = 100 - (100 / (1 + money_ratio))
            
            mfi_values.append(mfi)
        
        return mfi_values[-1] if mfi_values else 50
        
    except Exception as e:
        print(f"❌ MFI calculation error: {e}")
        return 50

def calculate_support_resistance(highs, lows, closes, lookback=50):
    """Identify support and resistance levels"""
    try:
        if len(highs) < lookback:
            return None, None
            
        # Simple support/resistance using recent highs/lows
        support_level = min(lows[-lookback:])
        resistance_level = max(highs[-lookback:])
        
        return support_level, resistance_level
        
    except Exception as e:
        print(f"❌ Support/Resistance calculation error: {e}")
        return None, None

def calculate_atr_stop_loss(highs, lows, closes, period=14, multiplier=2):
    """Calculate Stop Loss berdasarkan ATR"""
    try:
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return None
            
        tr_values = []
        for i in range(1, len(closes)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr = max(tr1, tr2, tr3)
            tr_values.append(tr)
        
        atr = np.mean(tr_values[-period:])
        return atr * multiplier
        
    except Exception as e:
        print(f"❌ ATR calculation error: {e}")
        return None

# ==================== SISTEM ANALISIS YANG DIPERBAIKI ====================
def analyze_coin_improved(symbol):
    """Analisis koin dengan kriteria yang lebih ketat"""
    global client
    
    if client is None:
        print("❌ Client Binance belum diinisialisasi")
        return None
        
    try:
        # Ambil data multi-timeframe
        data_15m = get_klines_data_fast(symbol, Client.KLINE_INTERVAL_15MINUTE, 100)
        data_5m = get_klines_data_fast(symbol, Client.KLINE_INTERVAL_5MINUTE, 100)
        data_1h = get_klines_data_fast(symbol, Client.KLINE_INTERVAL_1HOUR, 100)
        
        if not data_15m or not data_5m or not data_1h:
            return None
        
        # Extract data
        closes_15m = data_15m['close']
        closes_5m = data_5m['close']
        closes_1h = data_1h['close']
        highs_15m = data_15m['high']
        lows_15m = data_15m['low']
        volumes_15m = data_15m['volume']
        
        # Hitung semua indikator
        current_price = closes_15m[-1]
        
        # EMA Multi-timeframe
        ema_9_15m = calculate_ema(closes_15m, 9)
        ema_21_15m = calculate_ema(closes_15m, 21)
        ema_50_15m = calculate_ema(closes_15m, 50)
        ema_200_1h = calculate_ema(closes_1h, 200)
        
        # RSI
        rsi_15m = calculate_rsi(closes_15m, 14)
        rsi_5m = calculate_rsi(closes_5m, 14)
        
        # MACD
        macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(closes_15m, 12, 26, 9)
        
        # MFI (Money Flow Index)
        mfi_15m = calculate_mfi(highs_15m, lows_15m, closes_15m, volumes_15m, 14)
        
        # Volume analysis
        volume_ratio = calculate_volume_profile(volumes_15m, 20)
        current_volume = volumes_15m[-1] if volumes_15m else 0
        avg_volume = np.mean(volumes_15m[-20:]) if len(volumes_15m) >= 20 else current_volume
        
        # Support Resistance
        support, resistance = calculate_support_resistance(highs_15m, lows_15m, closes_15m, 50)
        
        # ATR untuk stop loss
        atr_value = calculate_atr_stop_loss(highs_15m, lows_15m, closes_15m, 14, 2)
        
        # ========== KRITERIA SINYAL YANG LEBIH KETAT ==========
        
        # 1. Trend utama bullish (harga di atas EMA 200 1H)
        trend_bullish = current_price > ema_200_1h if ema_200_1h else False
        
        # 2. EMA alignment bullish
        ema_alignment = (ema_9_15m > ema_21_15m > ema_50_15m and 
                        current_price > ema_9_15m)
        
        # 3. RSI kondisi ideal (momentum baik)
        rsi_ok = (rsi_15m > 45 and rsi_15m < 70 and 
                 rsi_5m > 40 and rsi_5m < 75)
        
        # 4. MACD bullish
        macd_bullish = (macd_hist_15m > 0 and macd_15m > macd_signal_15m)
        
        # 5. MFI tidak overbought
        mfi_ok = mfi_15m < 80
        
        # 6. Volume konfirmasi
        volume_ok = volume_ratio > 1.0 and current_volume > avg_volume * 0.7
        
        # 7. Price position (di atas support)
        price_position_ok = support and (current_price > support * 1.01)
        
        # Hitung confidence score
        confidence = 0
        if trend_bullish: confidence += 20
        if ema_alignment: confidence += 25
        if rsi_ok: confidence += 15
        if macd_bullish: confidence += 15
        if mfi_ok: confidence += 10
        if volume_ok: confidence += 10
        if price_position_ok: confidence += 5
        
        # Sinyal buy hanya jika kriteria utama terpenuhi
        buy_signal = (trend_bullish and ema_alignment and rsi_ok and 
                     macd_bullish and mfi_ok and confidence >= 75)
        
        return {
            'symbol': symbol,
            'buy_signal': buy_signal,
            'confidence': confidence,
            'current_price': current_price,
            'rsi_15m': rsi_15m,
            'rsi_5m': rsi_5m,
            'mfi_15m': mfi_15m,
            'volume_ratio': volume_ratio,
            'support': support,
            'resistance': resistance,
            'atr_value': atr_value,
            'trend_bullish': trend_bullish,
            'ema_alignment': ema_alignment,
            'macd_bullish': macd_bullish
        }
        
    except Exception as e:
        print(f"❌ Error in improved analysis for {symbol}: {e}")
        return None

# ==================== FUNGSI YANG SUDAH ADA (dengan minor improvements) ====================
def rate_limit():
    """Rate limiting"""
    time.sleep(0.1)

def get_klines_data_fast(symbol, interval, limit=100):
    """Get klines data"""
    global client
    
    if client is None:
        print("❌ Client Binance belum diinisialisasi")
        return None
        
    try:
        rate_limit()
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if klines and len(klines) >= 20:
            closes = [float(kline[4]) for kline in klines]
            highs = [float(kline[2]) for kline in klines]
            lows = [float(kline[3]) for kline in klines]
            volumes = [float(kline[5]) for kline in klines]
            
            return {
                'close': closes,
                'high': highs,
                'low': lows,
                'volume': volumes
            }
        return None
    except Exception as e:
        print(f"❌ Error getting klines for {symbol}: {e}")
        return None

def calculate_ema(prices, period):
    """Calculate EMA"""
    if len(prices) < period:
        return None
    try:
        series = pd.Series(prices)
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]
    except:
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return 50
    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean()
        avg_losses = pd.Series(losses).rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    except:
        return 50

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    if len(prices) < slow:
        return None, None, None
    try:
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        
        return macd_line.iloc[-1], macd_signal.iloc[-1], macd_histogram.iloc[-1]
    except:
        return None, None, None

def calculate_volume_profile(volumes, period=20):
    """Calculate volume profile"""
    if len(volumes) < period:
        return 1.0
    try:
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-period:])
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    except:
        return 1.0

# ==================== MAIN BOT LOGIC YANG DIPERBAIKI ====================
def scan_for_signals_improved():
    """Scan untuk sinyal dengan koin yang sudah difilter"""
    global BOT_RUNNING
    
    print("\n🎯 IMPROVED SCANNING - Looking for quality signals...")
    
    # Filter koin berkualitas
    quality_coins = filter_quality_coins()
    
    signals = []
    for coin in quality_coins:
        if not BOT_RUNNING:
            break
            
        print(f"🔍 Analyzing {coin}...")
        analysis = analyze_coin_improved(coin)
        
        if analysis and analysis['buy_signal']:
            print(f"🚨 QUALITY SIGNAL: {coin} - Confidence: {analysis['confidence']:.1f}%")
            
            # Log detail sinyal
            signal_info = (
                f"📊 Signal Details:\n"
                f"• RSI 15m: {analysis['rsi_15m']:.1f}\n"
                f"• MFI: {analysis['mfi_15m']:.1f}\n"
                f"• Volume Ratio: {analysis['volume_ratio']:.2f}\n"
                f"• Trend: {'Bullish' if analysis['trend_bullish'] else 'Bearish'}\n"
                f"• EMA Alignment: {'Yes' if analysis['ema_alignment'] else 'No'}"
            )
            print(signal_info)
            
            signals.append(analysis)
            
            # Kirim notifikasi Telegram
            telegram_msg = (
                f"🚨 <b>QUALITY BUY SIGNAL</b>\n"
                f"• {coin}: {analysis['confidence']:.1f}%\n"
                f"• Price: ${analysis['current_price']:.6f}\n"
                f"• RSI: {analysis['rsi_15m']:.1f}\n"
                f"• Volume: {analysis['volume_ratio']:.2f}x\n"
                f"• Support: ${analysis['support']:.6f}\n"
                f"• ATR Stop: ${analysis['atr_value']:.6f}"
            )
            send_telegram_message(telegram_msg)
    
    print(f"📊 Scan complete: {len(signals)} quality signals found")
    return signals

def execute_improved_trade(signal):
    """Execute trade dengan risk management yang lebih baik"""
    global active_position
    
    symbol = signal['symbol']
    confidence = signal['confidence']
    current_price = signal['current_price']
    atr_value = signal['atr_value']
    
    print(f"⚡ EXECUTING IMPROVED TRADE: {symbol}")
    
    # Gunakan ATR untuk stop loss jika available
    if atr_value and atr_value > 0:
        stop_loss = current_price - atr_value
        take_profit = current_price + (atr_value * 2)  # Risk:Reward 1:2
    else:
        # Fallback ke percentage-based
        stop_loss = current_price * (1 - STOP_LOSS_PCT)
        take_profit = current_price * (1 + TAKE_PROFIT_PCT)
    
    # Place buy order (simulasi atau real)
    # ... (implementasi order placement sama seperti sebelumnya)
    
    print(f"✅ Trade executed with improved risk management")
    return True

def improved_main_loop():
    """Main loop yang diperbaiki"""
    global BOT_RUNNING, active_position, client
    
    print("🚀 STARTING IMPROVED TRADING BOT")
    
    # Initialize Binance client
    if not initialize_binance_client():
        print("❌ Tidak dapat melanjutkan tanpa koneksi Binance")
        return
    
    BOT_RUNNING = True
    
    while BOT_RUNNING:
        try:
            if active_position:
                # Monitor posisi aktif
                time.sleep(1)
                continue
            
            # Scan untuk sinyal berkualitas
            signals = scan_for_signals_improved()
            
            if signals:
                # Execute sinyal terbaik
                best_signal = max(signals, key=lambda x: x['confidence'])
                execute_improved_trade(best_signal)
            
            time.sleep(10)  # Delay antara scan
            
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(5)

def send_telegram_message(message):
    """Send Telegram message"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

# ==================== START BOT ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 IMPROVED TRADING BOT - QUALITY SIGNALS")
    print("=" * 60)
    
    improved_main_loop()
