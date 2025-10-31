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
from flask import Flask
import signal
import sys

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI YANG LEBIH AGRESIF ====================
API_KEYS = [
    {
        'key': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET')
    }
]

CURRENT_API_INDEX = 0
INITIAL_INVESTMENT = float(os.getenv('INITIAL_INVESTMENT', '10.0'))
ORDER_RUN = os.getenv('ORDER_RUN', 'False').lower() == 'true'

# Trading Parameters - LEBIH AGRESIF TAPI MASIH AMAN
TAKE_PROFIT_PCT = 0.025  # 2.5% (naik dari 1.5%)
STOP_LOSS_PCT = 0.012    # 1.2% (naik dari 0.8%)
TRAILING_STOP_ACTIVATION = 0.012
TRAILING_STOP_PCT = 0.008

# Risk Management
POSITION_SIZING_PCT = 0.4  # Lebih besar untuk agresif
MAX_DRAWDOWN_PCT = 0.4
ADAPTIVE_CONFIDENCE = True

# Filter Koin - LEBIH FLEKSIBEL
MIN_24H_VOLUME = 3000000  # $3 juta volume minimum (turun dari $5 juta)
MAX_SPREAD_PCT = 0.2      # Spread maksimal 0.2% (naik dari 0.15%)

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
SEND_TELEGRAM_NOTIFICATIONS = True

# File Configuration
LOG_FILE = 'aggressive_trading_log.txt'
TRADE_HISTORY_FILE = 'aggressive_trade_history.json'

# Global variables
current_investment = INITIAL_INVESTMENT
active_position = None
trade_history = []
client = None
BOT_RUNNING = False
last_signal_time = {}

# Flask app untuk health check
app = Flask(__name__)

@app.route('/')
def health_check():
    """Health check endpoint untuk Render"""
    return {
        'status': 'running',
        'service': 'Aggressive Trading Bot Signal',
        'timestamp': datetime.now().isoformat(),
        'message': 'Bot is aggressively scanning for trading signals'
    }

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}

def start_web_server():
    """Start simple web server untuk health checks"""
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Starting health check server on port {port}")
    # Run in background thread agar tidak blocking main bot
    from waitress import serve
    serve(app, host="0.0.0.0", port=port)

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

# ==================== FUNGSI UNTUK MENDAPATKAN SEMUA KOIN ====================
def get_all_quality_coins():
    """Dapatkan semua koin USDT dari Binance kecuali yang dikecualikan"""
    global client
    
    if client is None:
        print("❌ Client Binance belum diinisialisasi")
        return []
    
    try:
        print("🔄 Mengambil semua koin dari Binance...")
        
        # Dapatkan info semua trading pairs
        exchange_info = client.get_exchange_info()
        all_symbols = exchange_info['symbols']
        
        # Filter hanya USDT pairs yang aktif trading
        usdt_pairs = []
        excluded_symbols = ['BTCUSDT', 'ETHUSDT', 'FDUSDT', 'USDCUSDT', 'FDUSUSDT', 'BUSDUSDT', 'TUSDUSDT']
        
        for symbol_info in all_symbols:
            symbol = symbol_info['symbol']
            status = symbol_info['status']
            
            # Filter kriteria - LEBIH FLEKSIBEL
            if (symbol.endswith('USDT') and 
                status == 'TRADING' and 
                symbol not in excluded_symbols and
                not any(excluded in symbol for excluded in ['UP', 'DOWN', 'BEAR', 'BULL'])):  # Exclude leveraged tokens
                usdt_pairs.append(symbol)
        
        print(f"📊 Ditemukan {len(usdt_pairs)} koin USDT")
        return usdt_pairs
        
    except Exception as e:
        print(f"❌ Error mendapatkan daftar koin: {e}")
        return []

# ==================== FUNGSI FILTER KOIN YANG LEBIH FLEKSIBEL ====================
def filter_quality_coins():
    """Filter koin berdasarkan volume, spread, dan likuiditas - LEBIH FLEKSIBEL"""
    global client
    quality_coins = []
    
    if client is None:
        print("❌ Client Binance belum diinisialisasi")
        return quality_coins
    
    # Dapatkan semua koin USDT
    all_coins = get_all_quality_coins()
    if not all_coins:
        print("❌ Tidak ada koin yang ditemukan")
        return quality_coins
    
    print(f"🔍 Filtering {len(all_coins)} coins based on volume and liquidity...")
    
    # Batasi jumlah koin yang di-scan untuk menghindari rate limit
    coins_to_scan = all_coins[:80]  # Scan 80 koin pertama (lebih banyak)
    
    for coin in coins_to_scan:
        try:
            # Gunakan get_ticker() yang benar
            rate_limit()
            ticker = client.get_ticker(symbol=coin)
            volume = float(ticker['quoteVolume'])
            
            # Cek spread bid-ask menggunakan order book
            book = client.get_order_book(symbol=coin, limit=5)
            bid_price = float(book['bids'][0][0])
            ask_price = float(book['asks'][0][0])
            spread = (ask_price - bid_price) / bid_price * 100
            
            # Cek price change untuk volatilitas
            price_change = float(ticker['priceChangePercent'])
            current_price = float(ticker['lastPrice'])
            
            # Filter kondisi - LEBIH FLEKSIBEL
            if (volume >= MIN_24H_VOLUME and 
                spread <= MAX_SPREAD_PCT and
                abs(price_change) < 30.0 and  # Lebih toleran terhadap volatilitas
                current_price > 0.0005):      # Harga minimum lebih rendah
                
                quality_coins.append(coin)
                print(f"✅ {coin}: Volume=${volume:,.0f}, Spread={spread:.3f}%, Price=${current_price:.6f}")
                    
        except Exception as e:
            # Skip error untuk koin tertentu, lanjut ke koin berikutnya
            print(f"⚠️  Skip {coin}: {e}")
            continue
    
    # Urutkan berdasarkan volume (descending) dan ambil top 30
    quality_coins = get_sorted_coins_by_volume(quality_coins)[:30]
    
    print(f"🎯 Quality coins selected: {len(quality_coins)}")
    return quality_coins

def get_sorted_coins_by_volume(coins):
    """Urutkan koin berdasarkan volume 24h"""
    global client
    coin_volumes = []
    
    for coin in coins:
        try:
            rate_limit()
            ticker = client.get_ticker(symbol=coin)
            volume = float(ticker['quoteVolume'])
            coin_volumes.append((coin, volume))
        except:
            continue
    
    # Urutkan berdasarkan volume descending
    coin_volumes.sort(key=lambda x: x[1], reverse=True)
    return [coin for coin, volume in coin_volumes]

# ==================== INDIKATOR TEKNIKAL YANG LEBIH SENSITIF ====================
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

def calculate_momentum(closes, period=10):
    """Calculate price momentum"""
    if len(closes) < period:
        return 0
    try:
        momentum = ((closes[-1] - closes[-period]) / closes[-period]) * 100
        return momentum
    except:
        return 0

def calculate_volatility(closes, period=20):
    """Calculate price volatility"""
    if len(closes) < period:
        return 0
    try:
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * 100
        return volatility
    except:
        return 0

# ==================== SISTEM ANALISIS YANG LEBIH AGRESIF ====================
def analyze_coin_aggressive(symbol):
    """Analisis koin dengan kriteria yang lebih agresif"""
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
        
        # EMA Multi-timeframe - PERIODE LEBIH PENDEK UNTUK RESPONSIF
        ema_5_15m = calculate_ema(closes_15m, 5)
        ema_9_15m = calculate_ema(closes_15m, 9)
        ema_21_15m = calculate_ema(closes_15m, 21)
        ema_50_15m = calculate_ema(closes_15m, 50)
        ema_100_1h = calculate_ema(closes_1h, 100)  # EMA lebih pendek untuk trend
        
        # RSI - RANGE LEBIH LUAS
        rsi_15m = calculate_rsi(closes_15m, 14)
        rsi_5m = calculate_rsi(closes_5m, 14)
        
        # MACD
        macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(closes_15m, 8, 21, 9)  # MACD lebih sensitif
        
        # MFI (Money Flow Index) - RANGE LEBIH LUAS
        mfi_15m = calculate_mfi(highs_15m, lows_15m, closes_15m, volumes_15m, 14)
        
        # Volume analysis - LEBIH FLEKSIBEL
        volume_ratio = calculate_volume_profile(volumes_15m, 20)
        current_volume = volumes_15m[-1] if volumes_15m else 0
        avg_volume = np.mean(volumes_15m[-20:]) if len(volumes_15m) >= 20 else current_volume
        
        # Support Resistance
        support, resistance = calculate_support_resistance(highs_15m, lows_15m, closes_15m, 50)
        
        # ATR untuk stop loss
        atr_value = calculate_atr_stop_loss(highs_15m, lows_15m, closes_15m, 14, 1.8)  # Multiplier lebih kecil
        
        # Momentum dan Volatility
        momentum_15m = calculate_momentum(closes_15m, 10)
        volatility_15m = calculate_volatility(closes_15m, 20)
        
        # ========== KRITERIA SINYAL YANG LEBIH AGRESIF ==========
        
        # 1. Trend utama bullish (harga di atas EMA 100 1H) - LEBIH FLEKSIBEL
        trend_bullish = current_price > ema_100_1h if ema_100_1h else True  # Tidak wajib
        
        # 2. EMA alignment bullish - LEBIH FLEKSIBEL
        ema_alignment = (ema_5_15m > ema_9_15m and ema_9_15m > ema_21_15m and
                        current_price > ema_5_15m)
        
        # 3. RSI kondisi ideal (momentum baik) - RANGE LEBIH LUAS
        rsi_ok = (rsi_15m > 40 and rsi_15m < 75 and 
                 rsi_5m > 35 and rsi_5m < 80)
        
        # 4. MACD bullish - LEBIH SENSITIF
        macd_bullish = (macd_hist_15m > 0 or (macd_15m > macd_signal_15m))
        
        # 5. MFI tidak overbought - RANGE LEBIH LUAS
        mfi_ok = mfi_15m < 85
        
        # 6. Volume konfirmasi - LEBIH FLEKSIBEL
        volume_ok = volume_ratio > 0.8 or current_volume > avg_volume * 0.6
        
        # 7. Price position (di atas support) - LEBIH FLEKSIBEL
        price_position_ok = (support is None) or (current_price > support * 1.005)
        
        # 8. Momentum positif
        momentum_ok = momentum_15m > -2.0  # Boleh sedikit negatif
        
        # 9. Volatility acceptable
        volatility_ok = volatility_15m < 8.0  # Volatilitas tidak terlalu tinggi
        
        # Hitung confidence score - BOBOT LEBIH SEIMBANG
        confidence = 0
        if trend_bullish: confidence += 15
        if ema_alignment: confidence += 20
        if rsi_ok: confidence += 15
        if macd_bullish: confidence += 15
        if mfi_ok: confidence += 10
        if volume_ok: confidence += 10
        if price_position_ok: confidence += 5
        if momentum_ok: confidence += 5
        if volatility_ok: confidence += 5
        
        # Sinyal buy dengan kriteria lebih fleksibel
        required_criteria = 5  # Minimal 5 dari 9 kriteria terpenuhi
        criteria_met = sum([
            trend_bullish, ema_alignment, rsi_ok, macd_bullish, 
            mfi_ok, volume_ok, price_position_ok, momentum_ok, volatility_ok
        ])
        
        buy_signal = (criteria_met >= required_criteria and confidence >= 65)
        
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
            'momentum': momentum_15m,
            'volatility': volatility_15m,
            'trend_bullish': trend_bullish,
            'ema_alignment': ema_alignment,
            'macd_bullish': macd_bullish,
            'criteria_met': f"{criteria_met}/9"
        }
        
    except Exception as e:
        print(f"❌ Error in aggressive analysis for {symbol}: {e}")
        return None

# ==================== FUNGSI YANG SUDAH ADA (dengan minor improvements) ====================
def rate_limit():
    """Rate limiting"""
    time.sleep(0.15)  # Sedikit lebih cepat untuk scanning agresif

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

# ==================== MAIN BOT LOGIC YANG LEBIH AGRESIF ====================
def scan_for_signals_aggressive():
    """Scan untuk sinyal dengan pendekatan lebih agresif"""
    global BOT_RUNNING, last_signal_time
    
    print("\n🎯 AGGRESSIVE SCANNING - Looking for frequent signals...")
    
    # Filter koin berkualitas
    quality_coins = filter_quality_coins()
    
    if not quality_coins:
        print("❌ Tidak ada koin berkualitas yang ditemukan")
        return []
    
    signals = []
    for coin in quality_coins:
        if not BOT_RUNNING:
            break
            
        print(f"🔍 Analyzing {coin}...")
        analysis = analyze_coin_aggressive(coin)
        
        if analysis and analysis['buy_signal']:
            # Cek cooldown untuk coin yang sama (5 menit)
            current_time = time.time()
            last_time = last_signal_time.get(coin, 0)
            
            if current_time - last_time < 300:  # 5 menit cooldown
                print(f"⏳ Skip {coin} - dalam cooldown period")
                continue
                
            print(f"🚨 AGGRESSIVE SIGNAL: {coin} - Confidence: {analysis['confidence']:.1f}%")
            
            # Log detail sinyal
            signal_info = (
                f"📊 Signal Details:\n"
                f"• RSI 15m: {analysis['rsi_15m']:.1f}\n"
                f"• MFI: {analysis['mfi_15m']:.1f}\n"
                f"• Volume Ratio: {analysis['volume_ratio']:.2f}\n"
                f"• Momentum: {analysis['momentum']:.2f}%\n"
                f"• Criteria Met: {analysis['criteria_met']}\n"
                f"• Trend: {'Bullish' if analysis['trend_bullish'] else 'Bearish'}\n"
                f"• EMA Alignment: {'Yes' if analysis['ema_alignment'] else 'No'}"
            )
            print(signal_info)
            
            signals.append(analysis)
            last_signal_time[coin] = current_time
            
            # Kirim notifikasi Telegram
            if SEND_TELEGRAM_NOTIFICATIONS:
                telegram_msg = (
                    f"🚨 <b>AGGRESSIVE BUY SIGNAL</b>\n"
                    f"• {coin}: {analysis['confidence']:.1f}%\n"
                    f"• Price: ${analysis['current_price']:.6f}\n"
                    f"• RSI: {analysis['rsi_15m']:.1f}\n"
                    f"• Volume: {analysis['volume_ratio']:.2f}x\n"
                    f"• Momentum: {analysis['momentum']:.2f}%\n"
                    f"• Criteria: {analysis['criteria_met']}\n"
                    f"• ATR Stop: ${analysis['atr_value']:.6f if analysis['atr_value'] else 'N/A'}"
                )
                send_telegram_message(telegram_msg)
    
    print(f"📊 Scan complete: {len(signals)} aggressive signals found")
    return signals

def execute_aggressive_trade(signal):
    """Execute trade dengan approach lebih agresif"""
    global active_position
    
    symbol = signal['symbol']
    confidence = signal['confidence']
    current_price = signal['current_price']
    atr_value = signal['atr_value']
    
    print(f"⚡ EXECUTING AGGRESSIVE TRADE: {symbol}")
    
    # Gunakan ATR untuk stop loss jika available
    if atr_value and atr_value > 0:
        stop_loss = current_price - atr_value
        take_profit = current_price + (atr_value * 2.5)  # Risk:Reward 1:2.5
    else:
        # Fallback ke percentage-based
        stop_loss = current_price * (1 - STOP_LOSS_PCT)
        take_profit = current_price * (1 + TAKE_PROFIT_PCT)
    
    # Simulasi trade execution (untuk sinyal telegram only)
    trade_info = {
        'symbol': symbol,
        'entry_price': current_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"✅ Trade signal generated for {symbol}")
    print(f"   Entry: ${current_price:.6f}")
    print(f"   Stop Loss: ${stop_loss:.6f}")
    print(f"   Take Profit: ${take_profit:.6f}")
    
    # Kirim detail trade ke Telegram
    if SEND_TELEGRAM_NOTIFICATIONS:
        trade_msg = (
            f"📈 <b>AGGRESSIVE TRADE EXECUTED</b>\n"
            f"• {symbol}\n"
            f"• Entry: ${current_price:.6f}\n"
            f"• Stop Loss: ${stop_loss:.6f}\n"
            f"• Take Profit: ${take_profit:.6f}\n"
            f"• Confidence: {confidence:.1f}%\n"
            f"• Risk/Reward: 1:2.5"
        )
        send_telegram_message(trade_msg)
    
    return True

def aggressive_main_loop():
    """Main loop yang lebih agresif"""
    global BOT_RUNNING, active_position, client
    
    print("🚀 STARTING AGGRESSIVE TRADING BOT - FREQUENT TELEGRAM SIGNALS")
    
    # Initialize Binance client
    if not initialize_binance_client():
        print("❌ Tidak dapat melanjutkan tanpa koneksi Binance")
        return
    
    BOT_RUNNING = True
    
    # Kirim notifikasi start bot
    if SEND_TELEGRAM_NOTIFICATIONS:
        send_telegram_message("🤖 <b>AGGRESSIVE TRADING BOT STARTED</b>\n• Mode: Frequent Telegram Signals\n• Aggressively scanning for buy signals...")
    
    scan_count = 0
    while BOT_RUNNING:
        try:
            scan_count += 1
            print(f"\n=== AGGRESSIVE SCAN #{scan_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            
            # Scan untuk sinyal berkualitas
            signals = scan_for_signals_aggressive()
            
            if signals:
                # Execute semua sinyal yang memenuhi kriteria (tidak hanya yang terbaik)
                for signal in signals:
                    if signal['confidence'] >= 65:  # Minimum confidence threshold lebih rendah
                        execute_aggressive_trade(signal)
                        time.sleep(2)  # Delay kecil antara eksekusi
            
            # Delay antara scan - lebih pendek untuk scanning lebih sering
            print(f"⏳ Waiting 30 seconds before next aggressive scan...")
            time.sleep(30)
            
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(30)  # Delay lebih pendek jika error

def send_telegram_message(message):
    """Send Telegram message"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("❌ Telegram bot token atau chat ID tidak ditemukan")
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Telegram message sent")
            return True
        else:
            print(f"❌ Failed to send Telegram message: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error sending Telegram message: {e}")
        return False

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    global BOT_RUNNING
    print(f"\n🛑 Received shutdown signal...")
    BOT_RUNNING = False
    if SEND_TELEGRAM_NOTIFICATIONS:
        send_telegram_message("🛑 <b>AGGRESSIVE TRADING BOT STOPPED</b>\n• Shutdown signal received")
    time.sleep(2)
    sys.exit(0)

# ==================== START BOT ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 AGGRESSIVE TRADING BOT - FREQUENT TELEGRAM SIGNALS")
    print("=" * 60)
    
    # Register signal handlers untuk graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start web server in background thread untuk health checks
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    try:
        aggressive_main_loop()
    except KeyboardInterrupt:
        print("\n🛑 Bot dihentikan oleh user")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("🛑 <b>TRADING BOT STOPPED</b>\n• Stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"❌ <b>BOT CRASHED</b>\n• Error: {str(e)}")
