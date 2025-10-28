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

# Load environment variables dari file .env
load_dotenv()
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
# Multiple API Keys untuk fallback
API_KEYS = [
    {
        'key': os.getenv('BINANCE_API_KEY_1'),
        'secret': os.getenv('BINANCE_API_SECRET_1')
    },
    {
        'key': os.getenv('BINANCE_API_KEY_2'), 
        'secret': os.getenv('BINANCE_API_SECRET_2')
    },
    {
        'key': os.getenv('BINANCE_API_KEY_3'),
        'secret': os.getenv('BINANCE_API_SECRET_3')
    }
]

# Default ke API key lama untuk kompatibilitas
if not API_KEYS[0]['key']:
    API_KEYS[0] = {
        'key': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET')
    }

CURRENT_API_INDEX = 0
INITIAL_INVESTMENT = float(os.getenv('INITIAL_INVESTMENT', '5.5'))
ORDER_RUN = os.getenv('ORDER_RUN', 'False').lower() == 'true'

# Trading Parameters - SESUAI PERMINTAAN
TAKE_PROFIT_PCT = 0.0062  # 0.62%
STOP_LOSS_PCT = 0.0160    # 1.6%
TRAILING_STOP_ACTIVATION = 0.0040
TRAILING_STOP_PCT = 0.0080

# Risk Management
POSITION_SIZING_PCT = 0.4
MAX_DRAWDOWN_PCT = 0.6
ADAPTIVE_CONFIDENCE = True

# PARAMETER TIMEFRAME - SESUAI PERMINTAAN
RSI_MIN_15M = 35
RSI_MAX_15M = 65
EMA_SHORT_15M = 12
EMA_LONG_15M = 26
MACD_FAST_15M = 7
MACD_SLOW_15M = 21
MACD_SIGNAL_15M = 7
LRO_PERIOD_15M = 20
VOLUME_PERIOD_15M = 15

RSI_MIN_5M = 35
RSI_MAX_5M = 68
EMA_SHORT_5M = 5
EMA_LONG_5M = 20
MACD_FAST_5M = 8
MACD_SLOW_5M = 21
MACD_SIGNAL_5M = 8
LRO_PERIOD_5M = 20
VOLUME_PERIOD_5M = 10

VOLUME_RATIO_MIN = 0.8

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
SEND_TELEGRAM_NOTIFICATIONS = True

# ==================== TELEGRAM CONTROL SYSTEM ====================
TELEGRAM_CONTROL_ENABLED = True
BOT_RUNNING = False
ADMIN_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Configuration file
CONFIG_FILE = 'bot_config.json'

# File Configuration
LOG_FILE = 'trading_log1.txt'
TRADE_HISTORY_FILE = 'trade_history1.json'
BOT_STATE_FILE = 'bot_state1.json'

# Coin List - dipersingkat untuk testing
COINS = [
    'PENGUUSDT','WALUSDT','MIRAUSDT','HEMIUSDT','PUMPUSDT','TRXUSDT','LTCUSDT','FFUSDT',
    'SUIUSDT','ASTERUSDT','ZECUSDT','CAKEUSDT','BNBUSDT','AVNTUSDT','DOGEUSDT','ADAUSDT',
    'XPLUSDT','XRPUSDT','DASHUSDT','SOLUSDT','LINKUSDT','AVAXUSDT', 'PEPEUSDT'
]

# ==================== INISIALISASI VARIABEL GLOBAL ====================
current_investment = INITIAL_INVESTMENT
active_position = None
buy_signals = []
trade_history = []
client = None

coin_performance = {}
dynamic_threshold = 30
recent_trades = deque(maxlen=10)
failed_coins = {}

performance_state = {
    'ema_pnl': 0.0,
    'loss_streak': 0,
    'win_streak': 0,
    'paused_until': 0,
    'last_adaptation_time': 0,
    'total_trades': 0,
    'total_wins': 0
}

# Hyperparameters untuk Feedback Loop
RECENT_WINDOW = 10
LOSS_STREAK_LIMIT = 2
LOSS_STREAK_THRESHOLD_INCREASE = 8
LOSS_STREAK_POSITION_SIZER_MULT = 0.7
LOSS_STREAK_COOLDOWN = 600
WIN_STREAK_LIMIT = 3
WIN_STREAK_THRESHOLD_DECREASE = 5
EMA_ALPHA = 0.3
MIN_POSITION_SIZING = 0.1
MAX_ADAPTATION_PERCENT = 0.25

# ==================== DELAY CONFIGURATION ====================
DELAY_BETWEEN_COINS = 0.3
DELAY_BETWEEN_REQUESTS = 0.1
DELAY_AFTER_ERROR = 1.0
DELAY_BETWEEN_SCANS = 3.0
DELAY_WHEN_PAUSED = 1.0

# Rate limiting
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 0.2

if os.environ.get('RENDER'):
    DELAY_BETWEEN_COINS = 0.5
    DELAY_BETWEEN_REQUESTS = 0.2
    DELAY_BETWEEN_SCANS = 5.0

# ==================== FUNGSI UTAMA ====================
def get_public_ip():
    """Mendapatkan IP publik"""
    try:
        services = ['https://api.ipify.org', 'https://ident.me', 'https://checkip.amazonaws.com']
        
        for service in services:
            try:
                response = requests.get(service, timeout=5)
                if response.status_code == 200:
                    ip = response.text.strip()
                    print(f"✅ Public IP: {ip}")
                    return ip
            except:
                continue
        return None
    except Exception as e:
        print(f"❌ Error getting public IP: {e}")
        return None

def rate_limit():
    """Rate limiting untuk menghindari ban Binance"""
    global LAST_REQUEST_TIME
    current_time = time.time()
    elapsed = current_time - LAST_REQUEST_TIME
    
    if elapsed < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - elapsed
        time.sleep(sleep_time)
    
    LAST_REQUEST_TIME = time.time()

def create_simple_health_endpoint():
    """Health endpoint untuk Render"""
    try:
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/')
        def health_check():
            return {'status': 'running', 'timestamp': datetime.now().isoformat()}
        
        @app.route('/health')
        def health():
            return {'status': 'healthy'}
        
        port = int(os.environ.get('PORT', 5000))
        
        def run_flask():
            app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"⚠️ Health endpoint error: {e}")
        return False

def check_render_environment():
    """Check jika running di Render"""
    return os.environ.get('RENDER', False)

def check_binance_connection():
    """Test koneksi ke Binance"""
    try:
        client.ping()
        print("✅ Binance connection test: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Binance connection test failed: {e}")
        return False

# ==================== MULTIPLE API KEY MANAGEMENT ====================
def rotate_api_key():
    """Rotate ke API key berikutnya"""
    global CURRENT_API_INDEX, client
    
    old_index = CURRENT_API_INDEX
    CURRENT_API_INDEX = (CURRENT_API_INDEX + 1) % len(API_KEYS)
    
    print(f"🔄 Rotating API key: {old_index} -> {CURRENT_API_INDEX}")
    
    if initialize_binance_client():
        send_telegram_message(f"🔄 <b>API KEY DIPERBARUI</b>\nBerpindah ke API key {CURRENT_API_INDEX + 1}")
        return True
    else:
        print(f"❌ Failed to rotate to API key {CURRENT_API_INDEX}")
        return False

def initialize_binance_client():
    """Initialize Binance client dengan API key yang aktif"""
    global client, CURRENT_API_INDEX
    
    try:
        api_key = API_KEYS[CURRENT_API_INDEX]['key']
        api_secret = API_KEYS[CURRENT_API_INDEX]['secret']
        
        if not api_key or not api_secret:
            print(f"❌ API key {CURRENT_API_INDEX} tidak valid")
            return False
            
        client = Client(api_key, api_secret, {"timeout": 20})
        print(f"✅ Binance client initialized dengan API key {CURRENT_API_INDEX + 1}")
        
        # Test connection
        client.ping()
        print(f"✅ Binance connection test successful dengan API key {CURRENT_API_INDEX + 1}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Binance client dengan API key {CURRENT_API_INDEX + 1}: {e}")
        return False

def handle_binance_error():
    """Handle error Binance dan rotate API key"""
    global BOT_RUNNING
    
    print("🔴 Binance error detected, attempting API key rotation...")
    
    # Coba rotate API key
    for attempt in range(len(API_KEYS)):
        if rotate_api_key():
            print("✅ API key rotation successful, continuing operations...")
            return True
    
    # Jika semua API key gagal
    print("❌ All API keys failed, stopping bot...")
    send_telegram_message("🔴 <b>SEMUA API KEY GAGAL</b>\nBot dihentikan otomatis.")
    BOT_RUNNING = False
    return False

# ==================== TELEGRAM & LOGGING ====================
def load_config():
    """Load configuration dari file"""
    default_config = {
        'trading_params': {
            'TAKE_PROFIT_PCT': 0.0062,
            'STOP_LOSS_PCT': 0.0160,
            'TRAILING_STOP_ACTIVATION': 0.0040,
            'TRAILING_STOP_PCT': 0.0080,
            'POSITION_SIZING_PCT': 0.4,
            'MAX_DRAWDOWN_PCT': 0.6,
            'ADAPTIVE_CONFIDENCE': True
        },
        'timeframe_params': {
            'RSI_MIN_15M': 35, 'RSI_MAX_15M': 65, 'EMA_SHORT_15M': 12, 'EMA_LONG_15M': 26,
            'MACD_FAST_15M': 7, 'MACD_SLOW_15M': 21, 'MACD_SIGNAL_15M': 7,
            'LRO_PERIOD_15M': 20, 'VOLUME_PERIOD_15M': 15,
            'RSI_MIN_5M': 35, 'RSI_MAX_5M': 68, 'EMA_SHORT_5M': 5, 'EMA_LONG_5M': 20,
            'MACD_FAST_5M': 8, 'MACD_SLOW_5M': 21, 'MACD_SIGNAL_5M': 8,
            'LRO_PERIOD_5M': 20, 'VOLUME_PERIOD_5M': 10, 'VOLUME_RATIO_MIN': 0.8
        }
    }
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            save_config(default_config)
            return default_config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return default_config

def save_config(config):
    """Save configuration ke file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False

def update_global_variables_from_config():
    """Update global variables dari config file"""
    global TAKE_PROFIT_PCT, STOP_LOSS_PCT, TRAILING_STOP_ACTIVATION, TRAILING_STOP_PCT
    global POSITION_SIZING_PCT, MAX_DRAWDOWN_PCT, ADAPTIVE_CONFIDENCE
    global RSI_MIN_15M, RSI_MAX_15M, EMA_SHORT_15M, EMA_LONG_15M, MACD_FAST_15M, MACD_SLOW_15M, MACD_SIGNAL_15M
    global LRO_PERIOD_15M, VOLUME_PERIOD_15M, RSI_MIN_5M, RSI_MAX_5M, EMA_SHORT_5M, EMA_LONG_5M
    global MACD_FAST_5M, MACD_SLOW_5M, MACD_SIGNAL_5M, LRO_PERIOD_5M, VOLUME_PERIOD_5M, VOLUME_RATIO_MIN
    
    config = load_config()
    trading_params = config['trading_params']
    timeframe_params = config['timeframe_params']
    
    TAKE_PROFIT_PCT = trading_params['TAKE_PROFIT_PCT']
    STOP_LOSS_PCT = trading_params['STOP_LOSS_PCT']
    TRAILING_STOP_ACTIVATION = trading_params['TRAILING_STOP_ACTIVATION']
    TRAILING_STOP_PCT = trading_params['TRAILING_STOP_PCT']
    POSITION_SIZING_PCT = trading_params['POSITION_SIZING_PCT']
    MAX_DRAWDOWN_PCT = trading_params['MAX_DRAWDOWN_PCT']
    ADAPTIVE_CONFIDENCE = trading_params['ADAPTIVE_CONFIDENCE']
    
    RSI_MIN_15M = timeframe_params['RSI_MIN_15M']
    RSI_MAX_15M = timeframe_params['RSI_MAX_15M']
    EMA_SHORT_15M = timeframe_params['EMA_SHORT_15M']
    EMA_LONG_15M = timeframe_params['EMA_LONG_15M']
    MACD_FAST_15M = timeframe_params['MACD_FAST_15M']
    MACD_SLOW_15M = timeframe_params['MACD_SLOW_15M']
    MACD_SIGNAL_15M = timeframe_params['MACD_SIGNAL_15M']
    LRO_PERIOD_15M = timeframe_params['LRO_PERIOD_15M']
    VOLUME_PERIOD_15M = timeframe_params['VOLUME_PERIOD_15M']
    RSI_MIN_5M = timeframe_params['RSI_MIN_5M']
    RSI_MAX_5M = timeframe_params['RSI_MAX_5M']
    EMA_SHORT_5M = timeframe_params['EMA_SHORT_5M']
    EMA_LONG_5M = timeframe_params['EMA_LONG_5M']
    MACD_FAST_5M = timeframe_params['MACD_FAST_5M']
    MACD_SLOW_5M = timeframe_params['MACD_SLOW_5M']
    MACD_SIGNAL_5M = timeframe_params['MACD_SIGNAL_5M']
    LRO_PERIOD_5M = timeframe_params['LRO_PERIOD_5M']
    VOLUME_PERIOD_5M = timeframe_params['VOLUME_PERIOD_5M']
    VOLUME_RATIO_MIN = timeframe_params['VOLUME_RATIO_MIN']

def handle_telegram_command():
    """Check untuk Telegram commands"""
    global BOT_RUNNING
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data['ok'] and data['result']:
                for update in data['result']:
                    if 'message' in update and 'text' in update['message']:
                        message = update['message']['text']
                        chat_id = update['message']['chat']['id']
                        
                        if str(chat_id) != ADMIN_CHAT_ID:
                            continue
                            
                        if message.startswith('/'):
                            process_telegram_command(message, chat_id, update['update_id'])
    except Exception as e:
        print(f"❌ Telegram command error: {e}")

def process_telegram_command(command, chat_id, update_id):
    """Process individual Telegram commands"""
    global BOT_RUNNING
    
    try:
        if command == '/start':
            if not BOT_RUNNING:
                BOT_RUNNING = True
                send_telegram_message("🤖 <b>BOT DIAKTIFKAN</b>\nBot trading sekarang berjalan.")
                print("✅ Bot started via Telegram command")
            else:
                send_telegram_message("⚠️ Bot sudah berjalan.")
                
        elif command == '/stop':
            if BOT_RUNNING:
                BOT_RUNNING = False
                send_telegram_message("🛑 <b>BOT DIHENTIKAN</b>\nTrading dihentikan.")
                print("🛑 Bot stopped via Telegram command")
            else:
                send_telegram_message("⚠️ Bot sudah dalam keadaan berhenti.")
                
        elif command == '/status':
            send_bot_status(chat_id)
            
        elif command == '/config':
            send_current_config(chat_id)
            
        elif command.startswith('/set '):
            handle_set_command(command, chat_id)
            
        elif command == '/help':
            send_help_message(chat_id)
            
        elif command.startswith('/sell'):
            handle_sell_command(command, chat_id)
            
        elif command.startswith('/coins '):
            handle_coins_command(command, chat_id)
            
        elif command == '/info':
            handle_info_command(chat_id)
            
        elif command == '/modal':
            handle_modal_command(command, chat_id)
            
        mark_update_processed(update_id)
        
    except Exception as e:
        send_telegram_message(f"❌ <b>ERROR PROCESSING COMMAND</b>\n{str(e)}")

def handle_modal_command(command, chat_id):
    """Handle /modal command untuk mengubah modal"""
    global current_investment, BOT_RUNNING
    
    if BOT_RUNNING:
        send_telegram_message("❌ <b>BOT MASIH BERJALAN</b>\nHentikan bot terlebih dahulu dengan /stop sebelum mengubah modal.")
        return
    
    try:
        parts = command.split()
        if len(parts) < 2:
            send_telegram_message(f"❌ Format: /modal [jumlah_usdt]\nContoh: /modal 10.5\nModal saat ini: ${current_investment:.2f}")
            return
            
        # Handle berbagai format angka
        modal_str = parts[1].replace(',', '.')  # Ganti koma dengan titik
        # Hapus karakter non-digit kecuali titik
        modal_str = ''.join(c for c in modal_str if c.isdigit() or c == '.')
        
        if not modal_str or modal_str == '.':
            send_telegram_message("❌ Format angka tidak valid. Contoh: /modal 5.4")
            return
        
        new_investment = float(modal_str)

        if new_investment <= 0:  # Untuk modal command  
            send_telegram_message("❌ Modal harus lebih besar dari 0")
            return
        
        if new_investment <= 0:
            send_telegram_message("❌ Modal harus lebih besar dari 0")
            return
        
        old_investment = current_investment
        current_investment = new_investment
        
        send_telegram_message(f"✅ <b>MODAL DIPERBARUI</b>\nModal sebelumnya: ${old_investment:.2f}\nModal baru: <b>${current_investment:.2f}</b>")
        print(f"✅ Investment updated: ${old_investment:.2f} -> ${current_investment:.2f}")
        
    except ValueError:
        send_telegram_message("❌ Format angka tidak valid. Gunakan format: /modal 10.5")
    except Exception as e:
        send_telegram_message(f"❌ Error mengubah modal: {str(e)}")

def handle_sell_command(command, chat_id):
    """Handle /sell command untuk menjual posisi aktif atau mengubah TP/SL"""
    global active_position, BOT_RUNNING
    
    if not active_position:
        send_telegram_message("❌ <b>TIDAK ADA POSISI AKTIF</b>\nTidak ada posisi yang bisa dijual atau diubah.")
        return
    
    if not BOT_RUNNING:
        send_telegram_message("❌ <b>BOT SEDANG BERHENTI</b>\nAktifkan bot terlebih dahulu dengan /start.")
        return
    
    symbol = active_position['symbol']
    quantity = active_position['quantity']
    entry_price = active_position['entry_price']
    current_price = get_current_price(symbol)
    
    # Parse command
    parts = command.split()
    
    if len(parts) == 1:
        # Hanya /sell - execute manual sell
        send_telegram_message(f"🔄 <b>MENJALANKAN SELL MANUAL</b>\nSymbol: {symbol}\nQuantity: {quantity:.6f}")
        
        success = execute_market_sell(symbol, quantity, entry_price, "MANUAL SELL")
        
        if success:
            send_telegram_message(f"✅ <b>SELL MANUAL BERHASIL</b>\n{symbol} telah dijual.")
        else:
            send_telegram_message(f"❌ <b>SELL MANUAL GAGAL</b>\nGagal menjual {symbol}.")
    
    elif len(parts) >= 3:
        # /sell dengan parameter (tp/sl dan harga)
        action = parts[1].lower()
        try:
            # Handle berbagai format angka (5.4, 5,4, dll)
            price_str = parts[2].replace(',', '.')  # Ganti koma dengan titik
            # Hapus karakter non-digit kecuali titik
            price_str = ''.join(c for c in price_str if c.isdigit() or c == '.')
            
            if not price_str or price_str == '.':
                send_telegram_message("❌ Format harga tidak valid. Contoh: /sell tp 0.153420")
                return
                
            price_value = float(price_str)
            
            if price_value <= 0:  # Untuk sell command
                send_telegram_message("❌ Harga harus lebih besar dari 0")
                return
            
            if action == 'tp':
                if price_value <= entry_price:
                    send_telegram_message(f"❌ <b>TAKE PROFIT HARUS DI ATAS HARGA BELI</b>\nHarga beli: ${entry_price:.6f}\nTP yang diminta: ${price_value:.6f}")
                    return
        
                old_tp = active_position['take_profit']
                active_position['take_profit'] = price_value
                send_telegram_message(f"✅ <b>TAKE PROFIT DIPERBARUI</b>\n{symbol}\nTP sebelumnya: ${old_tp:.6f}\nTP baru: <b>${price_value:.6f}</b>")
                print(f"✅ TP updated for {symbol}: {price_value}")
                
            elif action == 'sl':
                if price_value >= entry_price:
                    send_telegram_message(f"❌ <b>STOP LOSS HARUS DI BAWAH HARGA BELI</b>\nHarga beli: ${entry_price:.6f}\nSL yang diminta: ${price_value:.6f}")
                    return
                
                active_position['stop_loss'] = price_value
                send_telegram_message(f"✅ <b>STOP LOSS DIPERBARUI</b>\n{symbol}\nSL sebelumnya: ${active_position.get('stop_loss_old', 'N/A'):.6f}\nSL baru: <b>${price_value:.6f}</b>")
                print(f"✅ SL updated for {symbol}: {price_value}")
                
            else:
                send_telegram_message("❌ Format: /sell [tp/sl] [harga]\nContoh: /sell tp 2.987\n/sell sl 2.500")
                
        except ValueError:
            send_telegram_message("❌ Format harga tidak valid. Gunakan angka.\nContoh: /sell tp 2.987")
    else:
        send_telegram_message("❌ Format: /sell [tp/sl] [harga]\nContoh: /sell tp 2.987\n/sell sl 2.500\n/sell (untuk jual manual)")

def handle_coins_command(command, chat_id):
    """Handle /coins command untuk menambah/hapus coin"""
    global BOT_RUNNING, COINS
    
    if BOT_RUNNING:
        send_telegram_message("❌ <b>BOT MASIH BERJALAN</b>\nHentikan bot terlebih dahulu dengan /stop sebelum mengubah daftar coin.")
        return
    
    try:
        parts = command.split()
        if len(parts) < 3:
            send_telegram_message("❌ Format: /coins [add/del] [coin_name]\nContoh: /coins add BTCUSDT")
            return
            
        action = parts[1].lower()
        coin_name = parts[2].upper()
        
        if not coin_name.endswith('USDT'):
            coin_name += 'USDT'
        
        if action == 'add':
            if coin_name in COINS:
                send_telegram_message(f"⚠️ <b>COIN SUDAH ADA</b>\n{coin_name} sudah ada dalam daftar.")
            else:
                COINS.append(coin_name)
                send_telegram_message(f"✅ <b>COIN DITAMBAHKAN</b>\n{coin_name} telah ditambahkan ke daftar.\nTotal coin: {len(COINS)}")
                print(f"✅ Coin added: {coin_name}")
                
        elif action == 'del' or action == 'remove':
            if coin_name in COINS:
                COINS.remove(coin_name)
                send_telegram_message(f"✅ <b>COIN DIHAPUS</b>\n{coin_name} telah dihapus dari daftar.\nTotal coin: {len(COINS)}")
                print(f"✅ Coin removed: {coin_name}")
            else:
                send_telegram_message(f"❌ <b>COIN TIDAK DITEMUKAN</b>\n{coin_name} tidak ada dalam daftar.")
                
        else:
            send_telegram_message("❌ Format: /coins [add/del] [coin_name]\nContoh: /coins add BTCUSDT")
            
    except Exception as e:
        send_telegram_message(f"❌ Error mengubah daftar coin: {str(e)}")

def handle_info_command(chat_id):
    """Handle /info command untuk menampilkan daftar coin"""
    global COINS
    
    coins_count = len(COINS)
    
    # Format daftar coin untuk tampilan yang rapi
    if coins_count <= 20:
        coins_list = "\n".join([f"• {coin}" for coin in sorted(COINS)])
    else:
        # Tampilkan sebagian saja jika terlalu banyak
        coins_list = "\n".join([f"• {coin}" for coin in sorted(COINS)[:20]]) + f"\n• ... dan {coins_count - 20} coin lainnya"
    
    info_msg = (
        f"📊 <b>INFORMASI COIN</b>\n"
        f"Total Coin: {coins_count}\n"
        f"────────────────────\n"
        f"{coins_list}"
    )
    
    send_telegram_message(info_msg)

def handle_set_command(command, chat_id):
    """Handle /set command untuk mengubah konfigurasi"""
    global BOT_RUNNING
    
    # Cek apakah bot sedang berjalan
    if BOT_RUNNING:
        send_telegram_message("❌ <b>BOT MASIH BERJALAN</b>\nHentikan bot terlebih dahulu dengan /stop sebelum mengubah konfigurasi.")
        return
    
    try:
        parts = command.split()
        if len(parts) < 3:
            send_telegram_message("❌ Format: /set [parameter] [value]\nContoh: /set TAKE_PROFIT_PCT 0.008")
            return
            
        param_name = parts[1]
        param_value = ' '.join(parts[2:])
        
        config = load_config()
        updated = False
        
        # Daftar parameter yang valid
        valid_params = {
            # Trading parameters
            'TAKE_PROFIT_PCT': ('trading_params', float),
            'STOP_LOSS_PCT': ('trading_params', float),
            'TRAILING_STOP_ACTIVATION': ('trading_params', float),
            'TRAILING_STOP_PCT': ('trading_params', float),
            'POSITION_SIZING_PCT': ('trading_params', float),
            'MAX_DRAWDOWN_PCT': ('trading_params', float),
            'ADAPTIVE_CONFIDENCE': ('trading_params', bool),
            
            # 15M timeframe parameters
            'RSI_MIN_15M': ('timeframe_params', int),
            'RSI_MAX_15M': ('timeframe_params', int),
            'EMA_SHORT_15M': ('timeframe_params', int),
            'EMA_LONG_15M': ('timeframe_params', int),
            'MACD_FAST_15M': ('timeframe_params', int),
            'MACD_SLOW_15M': ('timeframe_params', int),
            'MACD_SIGNAL_15M': ('timeframe_params', int),
            'LRO_PERIOD_15M': ('timeframe_params', int),
            'VOLUME_PERIOD_15M': ('timeframe_params', int),
            
            # 5M timeframe parameters
            'RSI_MIN_5M': ('timeframe_params', int),
            'RSI_MAX_5M': ('timeframe_params', int),
            'EMA_SHORT_5M': ('timeframe_params', int),
            'EMA_LONG_5M': ('timeframe_params', int),
            'MACD_FAST_5M': ('timeframe_params', int),
            'MACD_SLOW_5M': ('timeframe_params', int),
            'MACD_SIGNAL_5M': ('timeframe_params', int),
            'LRO_PERIOD_5M': ('timeframe_params', int),
            'VOLUME_PERIOD_5M': ('timeframe_params', int),
            'VOLUME_RATIO_MIN': ('timeframe_params', float)
        }
        
        if param_name in valid_params:
            section, value_type = valid_params[param_name]
            old_value = config[section][param_name]
            
            try:
                # Convert value ke tipe yang sesuai
                if value_type == bool:
                    if param_value.lower() in ('true', '1', 'yes', 'y', 'ya'):
                        new_value = True
                    elif param_value.lower() in ('false', '0', 'no', 'n', 'tidak'):
                        new_value = False
                    else:
                        send_telegram_message(f"❌ Nilai boolean tidak valid: {param_value}\nGunakan: true/false, yes/no, 1/0")
                        return
                else:
                    new_value = value_type(param_value)
                
                # Validasi nilai
                if param_name.endswith('_PCT') and new_value <= 0:
                    send_telegram_message(f"❌ Nilai {param_name} harus lebih besar dari 0")
                    return
                elif param_name.startswith('RSI_') and (new_value < 0 or new_value > 100):
                    send_telegram_message(f"❌ Nilai {param_name} harus antara 0-100")
                    return
                elif param_name.startswith(('EMA_', 'MACD_', 'LRO_', 'VOLUME_')) and new_value <= 0:
                    send_telegram_message(f"❌ Nilai {param_name} harus lebih besar dari 0")
                    return
                
                config[section][param_name] = new_value
                updated = True
                
            except ValueError:
                send_telegram_message(f"❌ Format nilai tidak valid untuk {param_name}\nTipe yang diharapkan: {value_type.__name__}")
                return
        
        if updated:
            if save_config(config):
                update_global_variables_from_config()
                send_telegram_message(f"✅ <b>KONFIGURASI DIPERBARUI</b>\n{param_name}: {old_value} → {new_value}")
                print(f"✅ Configuration updated: {param_name} = {new_value}")
            else:
                send_telegram_message("❌ Gagal menyimpan konfigurasi.")
        else:
            send_telegram_message(f"❌ Parameter '{param_name}' tidak ditemukan.\nGunakan /config untuk melihat parameter yang tersedia.")
            
    except Exception as e:
        send_telegram_message(f"❌ Error mengubah konfigurasi: {str(e)}")

def send_bot_status(chat_id):
    """Send current bot status"""
    global BOT_RUNNING, active_position, current_investment, trade_history
    
    status_msg = (
        f"🤖 <b>BOT STATUS</b>\n"
        f"Status: {'🟢 BERJALAN' if BOT_RUNNING else '🔴 BERHENTI'}\n"
        f"Modal: ${current_investment:.2f}\n"
        f"Total Trade: {len(trade_history)}\n"
        f"Win Rate: {calculate_winrate():.1f}%\n"
        f"API Key: {CURRENT_API_INDEX + 1}\n"
    )
    
    if active_position:
        current_price = get_current_price(active_position['symbol'])
        if current_price:
            pnl_pct = (current_price - active_position['entry_price']) / active_position['entry_price'] * 100
            status_msg += f"Posisi Aktif: {active_position['symbol']}\n"
            status_msg += f"Entry: ${active_position['entry_price']:.6f}\n"
            status_msg += f"Current: ${current_price:.6f}\n"
            status_msg += f"TP: ${active_position['take_profit']:.6f}\n"
            status_msg += f"SL: ${active_position['stop_loss']:.6f}\n"
            status_msg += f"PnL: {pnl_pct:+.2f}%\n"
            status_msg += f"Gunakan /sell untuk close manual\n"
            status_msg += f"Gunakan /sell tp [harga] untuk ubah TP\n"
            status_msg += f"Gunakan /sell sl [harga] untuk ubah SL\n"
    else:
        status_msg += "Posisi Aktif: Tidak ada\n"
    
    send_telegram_message(status_msg)

def send_current_config(chat_id):
    """Send current configuration"""
    config = load_config()
    trading_params = config['trading_params']
    timeframe_params = config['timeframe_params']
    
    config_msg = (
        f"⚙️ <b>KONFIGURASI SAAT INI</b>\n"
        f"┌────────────────────────────┐\n"
        f"│ <b>TRADING PARAMS</b>\n"
        f"│ TP: {trading_params['TAKE_PROFIT_PCT']*100:.2f}%\n"
        f"│ SL: {trading_params['STOP_LOSS_PCT']*100:.2f}%\n"
        f"│ Position: {trading_params['POSITION_SIZING_PCT']*100:.1f}%\n"
        f"│ Trailing Act: {trading_params['TRAILING_STOP_ACTIVATION']*100:.2f}%\n"
        f"│ Trailing SL: {trading_params['TRAILING_STOP_PCT']*100:.2f}%\n"
        f"│ Adaptive: {trading_params['ADAPTIVE_CONFIDENCE']}\n"
        f"├────────────────────────────┤\n"
        f"│ <b>M15 PARAMS</b>\n"
        f"│ RSI: {timeframe_params['RSI_MIN_15M']}-{timeframe_params['RSI_MAX_15M']}\n"
        f"│ EMA: {timeframe_params['EMA_SHORT_15M']}/{timeframe_params['EMA_LONG_15M']}\n"
        f"│ MACD: {timeframe_params['MACD_FAST_15M']}/{timeframe_params['MACD_SLOW_15M']}/{timeframe_params['MACD_SIGNAL_15M']}\n"
        f"│ LRO: {timeframe_params['LRO_PERIOD_15M']}\n"
        f"│ Volume: {timeframe_params['VOLUME_PERIOD_15M']}\n"
        f"├────────────────────────────┤\n"
        f"│ <b>M5 PARAMS</b>\n"
        f"│ RSI: {timeframe_params['RSI_MIN_5M']}-{timeframe_params['RSI_MAX_5M']}\n"
        f"│ EMA: {timeframe_params['EMA_SHORT_5M']}/{timeframe_params['EMA_LONG_5M']}\n"
        f"│ MACD: {timeframe_params['MACD_FAST_5M']}/{timeframe_params['MACD_SLOW_5M']}/{timeframe_params['MACD_SIGNAL_5M']}\n"
        f"│ LRO: {timeframe_params['LRO_PERIOD_5M']}\n"
        f"│ Volume: {timeframe_params['VOLUME_PERIOD_5M']}\n"
        f"│ Volume Ratio: {timeframe_params['VOLUME_RATIO_MIN']}\n"
        f"└────────────────────────────┘\n"
        f"Gunakan /set [parameter] [value] untuk mengubah konfigurasi"
    )
    
    send_telegram_message(config_msg)

def send_help_message(chat_id):
    """Send help message"""
    help_msg = (
        f"📖 <b>BOT TRADING COMMANDS</b>\n"
        f"┌────────────────────────────┐\n"
        f"│ /start - Mulai bot trading\n"
        f"│ /stop - Hentikan bot\n"
        f"│ /status - Status bot saat ini\n"
        f"│ /config - Tampilkan konfigurasi\n"
        f"│ /help - Tampilkan pesan bantuan\n"
        f"│ /modal - Ubah modal (saat bot berhenti)\n"
        f"│ /info - Tampilkan daftar coin\n"
        f"├────────────────────────────┤\n"
        f"│ <b>SELL COMMANDS</b>\n"
        f"│ /sell - Jual posisi aktif (manual)\n"
        f"│ /sell tp [harga] - Ubah Take Profit\n"
        f"│ /sell sl [harga] - Ubah Stop Loss\n"
        f"│ Contoh:\n"
        f"│ /sell tp 2.987\n"
        f"│ /sell sl 2.500\n"
        f"├────────────────────────────┤\n"
        f"│ <b>UBAH KONFIGURASI</b>\n"
        f"│ /set [param] [value]\n"
        f"│ Contoh:\n"
        f"│ /set TAKE_PROFIT_PCT 0.008\n"
        f"│ /set RSI_MIN_15M 30\n"
        f"│ /set POSITION_SIZING_PCT 0.3\n"
        f"│ /set ADAPTIVE_CONFIDENCE false\n"
        f"├────────────────────────────┤\n"
        f"│ <b>KELOLA COIN</b>\n"
        f"│ /coins [add/del] [coin_name]\n"
        f"│ Contoh:\n"
        f"│ /coins add BTCUSDT\n"
        f"│ /coins del ETHUSDT\n"
        f"└────────────────────────────┘\n"
        f"<i>Hanya bisa diubah saat bot berhenti</i>"
    )
    
    send_telegram_message(help_msg)

def mark_update_processed(update_id):
    """Mark Telegram update as processed"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        params = {'offset': update_id + 1}
        requests.get(url, params=params, timeout=3)
    except:
        pass

def send_telegram_message(message):
    """Send notification ke Telegram"""
    if not SEND_TELEGRAM_NOTIFICATIONS:
        return False
        
    try:
        if len(message) > 4000:
            message = message[:4000] + "\n... (message truncated)"
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': ADMIN_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        response = requests.post(url, data=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

def write_log(entry):
    """Write trading log ke file"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {entry}"
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
        print(log_entry)
    except Exception as e:
        print(f"❌ Error writing to log file: {e}")

def initialize_logging():
    """Initialize log file"""
    try:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                f.write("🚀 ADVANCED TRADING BOT\n")
                f.write(f"Bot Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Initial Capital: ${INITIAL_INVESTMENT}\n")
            print(f"✅ Log file initialized: {LOG_FILE}")
    except Exception as e:
        print(f"❌ Error initializing log file: {e}")

def log_position_closed(symbol, entry_price, exit_price, quantity, exit_type):
    """Log ketika position closed"""
    global current_investment, trade_history
    
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        current_investment += pnl
        
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_type': exit_type,
            'capital_after': current_investment
        }
        trade_history.append(trade_record)
        
        recent_trades.append(pnl_pct)
        update_dynamic_threshold()
        trade_performance_feedback_loop()
        save_trade_history()
        
        winrate = calculate_winrate()
        total_trades = len(trade_history)
        
        log_entry = (f"📉 POSITION CLOSED | {symbol} | Exit: {exit_type} | "
                    f"Entry: ${entry_price:.6f} | Exit: ${exit_price:.6f} | "
                    f"PnL: ${pnl:.4f} ({pnl_pct:+.2f}%) | New Capital: ${current_investment:.2f}")
        
        write_log(log_entry)
        
        telegram_msg = (f"📉 <b>POSITION CLOSED - {exit_type}</b>\n"
                      f"Symbol: {symbol}\n"
                      f"Entry: ${entry_price:.6f} | Exit: ${exit_price:.6f}\n"
                      f"PnL: <b>{pnl_pct:+.2f}%</b> | Amount: ${pnl:.4f}\n"
                      f"New Capital: <b>${current_investment:.2f}</b>\n"
                      f"Win Rate: {winrate:.1f}% ({total_trades} trades)")
        
        send_telegram_message(telegram_msg)
        
    except Exception as e:
        print(f"❌ Error logging position close: {e}")

def log_position_opened(symbol, entry_price, quantity, take_profit, stop_loss, confidence):
    """Log ketika position opened"""
    try:
        log_entry = (f"📈 POSITION OPENED | {symbol} | "
                    f"Entry: ${entry_price:.6f} | Qty: {quantity:.6f} | "
                    f"TP: ${take_profit:.6f} | SL: ${stop_loss:.6f} | "
                    f"Confidence: {confidence:.1f}%")
        
        write_log(log_entry)
        
        telegram_msg = (f"📈 <b>POSITION OPENED</b>\n"
                      f"Symbol: {symbol}\n"
                      f"Entry: ${entry_price:.6f}\n"
                      f"Quantity: {quantity:.6f}\n"
                      f"Take Profit: ${take_profit:.6f}\n"
                      f"Stop Loss: ${stop_loss:.6f}\n"
                      f"Confidence: {confidence:.1f}%\n"
                      f"Gunakan /sell tp [harga] untuk ubah TP\n"
                      f"Gunakan /sell sl [harga] untuk ubah SL")
        
        send_telegram_message(telegram_msg)
        
    except Exception as e:
        print(f"❌ Error logging position open: {e}")

# ==================== MANAJEMEN DATA ====================
def load_trade_history():
    """Load trade history dari file"""
    global trade_history
    try:
        if os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE, 'r') as f:
                trade_history = json.load(f)
            print(f"✅ Loaded {len(trade_history)} previous trades")
    except Exception as e:
        print(f"❌ Error loading trade history: {e}")
        trade_history = []

def save_trade_history():
    """Save trade history ke file"""
    try:
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(trade_history, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving trade history: {e}")

def calculate_winrate():
    """Calculate winrate dari trade history"""
    if not trade_history:
        return 0.0
    
    wins = sum(1 for trade in trade_history if trade.get('pnl_pct', 0) > 0)
    return (wins / len(trade_history)) * 100

def load_bot_state():
    """Load bot state"""
    global coin_performance, dynamic_threshold, recent_trades, failed_coins, performance_state
    
    try:
        if os.path.exists(BOT_STATE_FILE):
            with open(BOT_STATE_FILE, 'r') as f:
                state = json.load(f)
                
            coin_performance = state.get('coin_performance', {})
            dynamic_threshold = max(30, state.get('dynamic_threshold', 30))
            recent_trades = deque(state.get('recent_trades', []), maxlen=10)
            performance_state = state.get('performance_state', performance_state)
            
            current_time = time.time()
            failed_coins = {
                coin: timestamp for coin, timestamp in state.get('failed_coins', {}).items()
                if current_time - timestamp < 86400
            }
            
            print(f"✅ Loaded bot state: threshold: {dynamic_threshold}")
    except Exception as e:
        print(f"❌ Error loading bot state: {e}")

def save_bot_state():
    """Save bot state"""
    try:
        state = {
            'coin_performance': coin_performance,
            'dynamic_threshold': dynamic_threshold,
            'recent_trades': list(recent_trades),
            'failed_coins': failed_coins,
            'performance_state': performance_state,
            'last_update': time.time()
        }
        
        with open(BOT_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving bot state: {e}")

def update_dynamic_threshold():
    """Update confidence threshold berdasarkan recent performance"""
    global dynamic_threshold
    
    if len(recent_trades) < 5:
        return
    
    wins = sum(1 for trade in recent_trades if trade > 0)
    winrate = wins / len(recent_trades)
    
    if winrate < 0.5:
        dynamic_threshold = min(40, dynamic_threshold + 2)
        print(f"📊 Winrate rendah ({winrate:.1%}), naikkan threshold ke {dynamic_threshold}")
    elif winrate > 0.7:
        dynamic_threshold = max(30, dynamic_threshold - 1)
        print(f"📊 Winrate tinggi ({winrate:.1%}), turunkan threshold ke {dynamic_threshold}")

# ==================== BINANCE UTILITIES ====================
def trade_performance_feedback_loop():
    """Adaptive feedback loop"""
    global recent_trades, performance_state, dynamic_threshold, POSITION_SIZING_PCT

    try:
        now = time.time()
        
        if len(recent_trades) < 2:
            return

        recent = list(recent_trades)
        wins = sum(1 for v in recent if v > 0)
        losses = sum(1 for v in recent if v <= 0)
        rolling_winrate = wins / len(recent) if recent else 0
        avg_pnl = sum(recent) / len(recent) if recent else 0

        ema_prev = performance_state.get('ema_pnl', 0.0)
        ema_new = EMA_ALPHA * avg_pnl + (1 - EMA_ALPHA) * ema_prev
        performance_state['ema_pnl'] = ema_new

        if trade_history:
            loss_streak = 0
            win_streak = 0
            
            for trade in reversed(trade_history[-5:]):
                pnl_pct = trade.get('pnl_pct', 0)
                if pnl_pct <= 0:
                    if win_streak == 0:
                        loss_streak += 1
                    else:
                        break
                else:
                    if loss_streak == 0:
                        win_streak += 1
                    else:
                        break
            
            performance_state['loss_streak'] = loss_streak
            performance_state['win_streak'] = win_streak
            performance_state['total_trades'] = len(trade_history)
            performance_state['total_wins'] = wins

        print(f"[FEEDBACK] Winrate: {rolling_winrate:.1%}, Avg PnL: {avg_pnl:.3f}%, Loss Streak: {performance_state['loss_streak']}")

        if (performance_state['loss_streak'] >= LOSS_STREAK_LIMIT and 
            now - performance_state.get('last_adaptation_time', 0) > 120):
            
            old_threshold = dynamic_threshold
            old_position_size = POSITION_SIZING_PCT
            
            dynamic_threshold = dynamic_threshold + LOSS_STREAK_THRESHOLD_INCREASE
            new_pos_size = max(MIN_POSITION_SIZING, POSITION_SIZING_PCT * LOSS_STREAK_POSITION_SIZER_MULT)
            max_allowed_reduction = POSITION_SIZING_PCT * (1 - MAX_ADAPTATION_PERCENT)
            POSITION_SIZING_PCT = max(new_pos_size, max_allowed_reduction)
            
            performance_state['paused_until'] = now + LOSS_STREAK_COOLDOWN
            performance_state['last_adaptation_time'] = now

            adaptation_msg = (
                f"⚠️ <b>ADAPTATION TRIGGERED - LOSS STREAK</b>\n"
                f"Loss Streak: {performance_state['loss_streak']}\n"
                f"Threshold: {old_threshold} → {dynamic_threshold}\n"
                f"Position Size: {old_position_size:.1%} → {POSITION_SIZING_PCT:.1%}"
            )
            send_telegram_message(adaptation_msg)

        if (performance_state['win_streak'] >= WIN_STREAK_LIMIT and 
            now - performance_state.get('last_adaptation_time', 0) > 120) and not performance_state.get('reward_triggered', False):
            
            old_threshold = dynamic_threshold
            old_position_size = POSITION_SIZING_PCT
            
            dynamic_threshold = max(30, dynamic_threshold - WIN_STREAK_THRESHOLD_DECREASE)
            new_pos_size = min(0.6, POSITION_SIZING_PCT * 1.15)
            max_allowed_increase = POSITION_SIZING_PCT * (1 + MAX_ADAPTATION_PERCENT)
            POSITION_SIZING_PCT = min(new_pos_size, max_allowed_increase)
            
            performance_state['last_adaptation_time'] = now
            performance_state['reward_triggered'] = True

            reward_msg = (
                f"✅ <b>PERFORMANCE REWARD - WIN STREAK</b>\n"
                f"Win Streak: {performance_state['win_streak']}\n"
                f"Threshold: {old_threshold} → {dynamic_threshold}\n"
                f"Position Size: {old_position_size:.1%} → {POSITION_SIZING_PCT:.1%}"
            )
            send_telegram_message(reward_msg)

        if (len(recent_trades) >= 5 and rolling_winrate < 0.2 and 
            now - performance_state.get('last_adaptation_time', 0) > 300):
            
            performance_state['paused_until'] = now + 1800
            emergency_msg = (
                f"🛑 <b>EMERGENCY STOP - POOR PERFORMANCE</b>\n"
                f"Win Rate: {rolling_winrate:.1%}\n"
                f"Bot paused for 30 minutes"
            )
            send_telegram_message(emergency_msg)

        save_bot_state()

    except Exception as e:
        print(f"❌ Feedback loop error: {e}")

def is_trading_paused():
    """Cek jika trading sedang dipause"""
    global performance_state
    
    now = time.time()
    paused_until = performance_state.get('paused_until', 0)
    
    if now < paused_until:
        remaining = paused_until - now
        if remaining > 60:
            print(f"🔒 Trading paused. Resuming in {int(remaining/60)} minutes")
        return True
    
    return False

def get_symbol_info(symbol):
    """Get symbol information"""
    try:
        rate_limit()
        info = client.get_symbol_info(symbol)
        return info
    except Exception as e:
        print(f"❌ Error getting symbol info: {e}")
        return None

def get_price_precision(symbol):
    """Get price precision untuk symbol"""
    try:
        info = get_symbol_info(symbol)
        if info:
            for f in info['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    tick_size = float(f['tickSize'])
                    precision = 0
                    while tick_size < 1:
                        tick_size *= 10
                        precision += 1
                    return precision
        return 6
    except Exception as e:
        return 6

def get_quantity_precision(symbol):
    """Get quantity precision untuk symbol"""
    try:
        info = get_symbol_info(symbol)
        if info:
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
                    precision = 0
                    while step_size < 1:
                        step_size *= 10
                        precision += 1
                    return precision
        return 2
    except Exception as e:
        return 2

def get_min_notional(symbol):
    """Get minimum notional requirement"""
    try:
        symbol_info = client.get_symbol_info(symbol)
        if symbol_info:
            for f in symbol_info['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = float(f.get('minNotional', 10.0))
                    return min_notional
                elif f['filterType'] == 'NOTIONAL':
                    min_notional = float(f.get('minNotional', 10.0))
                    return min_notional
        return 10.0
    except Exception as e:
        return 10.0

def round_step_size(quantity, symbol):
    """Round quantity ke step size yang tepat"""
    try:
        if quantity <= 0:
            return None
            
        symbol_info = get_symbol_info(symbol)
        if not symbol_info:
            return math.floor(quantity * 1000000) / 1000000
            
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                step_size = float(filter['stepSize'])
                min_qty = float(filter.get('minQty', step_size))
                if quantity < min_qty:
                    return None
                
                precision = int(round(-math.log(step_size, 10), 0))
                rounded_qty = math.floor(quantity / step_size) * step_size
                rounded_qty = round(rounded_qty, precision)
                
                if rounded_qty <= 0 or rounded_qty < min_qty:
                    return None
                    
                return float(rounded_qty)
                
        return math.floor(quantity * 1000000) / 1000000
        
    except Exception as e:
        print(f"❌ Error in round_step_size for {symbol}: {e}")
        return math.floor(quantity * 1000000) / 1000000

def get_current_price(symbol):
    """Get current price"""
    try:
        rate_limit()
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        precision = get_price_precision(symbol)
        return round(price, precision)
    except Exception as e:
        print(f"❌ Error getting current price for {symbol}: {e}")
        return None

# ==================== INDIKATOR TEKNIKAL ====================
def calculate_ema(prices, period):
    """Calculate EMA"""
    if prices is None or len(prices) < period:
        return None
    
    try:
        series = pd.Series(prices)
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1] if not ema.empty else None
    except Exception as e:
        print(f"❌ EMA calculation error: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return None
    
    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean()
        avg_losses = pd.Series(losses).rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    except Exception as e:
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
        
        return (
            macd_line.iloc[-1] if not macd_line.empty else None,
            macd_signal.iloc[-1] if not macd_signal.empty else None,
            macd_histogram.iloc[-1] if not macd_histogram.empty else None
        )
    except Exception as e:
        return None, None, None

def calculate_linear_regression(prices, period=20):
    """Calculate Linear Regression Oscillator"""
    if len(prices) < period:
        return 0
    
    try:
        x = np.arange(period)
        y = prices[-period:]
        
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        
        regression_line = m * x + b
        current_regression = regression_line[-1]
        current_price = prices[-1]
        
        lro = ((current_price - current_regression) / abs(current_regression)) * 100 if current_regression != 0 else 0
        
        return lro
    except Exception as e:
        return 0

def calculate_volume_profile(volumes, period=20):
    """Calculate volume profile"""
    if len(volumes) < period:
        return 1.0
    
    try:
        current_volume = volumes[-1] if volumes else 0
        avg_volume = np.mean(volumes[-period:]) if len(volumes) >= period else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        return volume_ratio
    except Exception as e:
        return 1.0

# ==================== DATA FETCHING ====================
def get_klines_data_fast(symbol, interval, limit=50):
    """Get klines data dengan cepat"""
    try:
        rate_limit()
        
        for attempt in range(2):
            try:
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
                else:
                    time.sleep(0.5)
            except Exception as e:
                if attempt == 0:
                    print(f"   ⚠️ Attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)
        
        return None
    except Exception as e:
        print(f"❌ Error get_klines_data_fast untuk {symbol}: {str(e)}")
        return None

def get_two_timeframe_data_fast(symbol):
    """Get data kedua timeframe"""
    try:
        test_price = get_current_price(symbol)
        if not test_price:
            return None, None
        
        data_15m = get_klines_data_fast(symbol, Client.KLINE_INTERVAL_15MINUTE, 40)
        data_5m = get_klines_data_fast(symbol, Client.KLINE_INTERVAL_5MINUTE, 40)
        
        if data_15m is None or data_5m is None:
            return None, None
            
        min_data_points = 20
        if (len(data_15m['close']) < min_data_points or 
            len(data_5m['close']) < min_data_points):
            return None, None
            
        return data_15m, data_5m
        
    except Exception as e:
        print(f"❌ Error get_two_timeframe_data_fast untuk {symbol}: {e}")
        return None, None

# ==================== SISTEM SINYAL ====================
def analyze_timeframe_fast(data, timeframe, symbol):
    """Analisis timeframe dengan cepat"""
    if data is None:
        return False, 0
        
    if not isinstance(data, dict):
        return False, 0
        
    required_keys = ['close', 'high', 'low', 'volume']
    for key in required_keys:
        if key not in data or data[key] is None or len(data[key]) < 20:
            return False, 0

    try:
        closes = data['close']

        if timeframe == '15m':
            ema_short_period = EMA_SHORT_15M
            ema_long_period = EMA_LONG_15M
            rsi_min = RSI_MIN_15M
            rsi_max = RSI_MAX_15M
            macd_fast = MACD_FAST_15M
            macd_slow = MACD_SLOW_15M
            macd_signal = MACD_SIGNAL_15M
            lro_period = LRO_PERIOD_15M
            volume_period = VOLUME_PERIOD_15M
        else:
            ema_short_period = EMA_SHORT_5M
            ema_long_period = EMA_LONG_5M
            rsi_min = RSI_MIN_5M
            rsi_max = RSI_MAX_5M
            macd_fast = MACD_FAST_5M
            macd_slow = MACD_SLOW_5M
            macd_signal = MACD_SIGNAL_5M
            lro_period = LRO_PERIOD_5M
            volume_period = VOLUME_PERIOD_5M

        ema_short = calculate_ema(closes, ema_short_period)
        ema_long = calculate_ema(closes, ema_long_period)
        rsi = calculate_rsi(closes, 14)
        macd_line, macd_signal, macd_histogram = calculate_macd(closes, macd_fast, macd_slow, macd_signal)
        lro = calculate_linear_regression(closes, lro_period)
        volume_ratio = calculate_volume_profile(data['volume'], volume_period)

        if any(x is None for x in [ema_short, ema_long, rsi, macd_line]):
            return False, 0

        current_price = closes[-1] if closes else 0

        price_above_ema_short = current_price > ema_short if ema_short is not None else False
        price_above_ema_long = current_price > ema_long if ema_long is not None else False
        ema_bullish = ema_short > ema_long if (ema_short is not None and ema_long is not None) else False
        
        rsi_ok = (rsi_min <= rsi <= rsi_max) if rsi is not None else False
        macd_bullish = macd_line > macd_signal if (macd_signal is not None and macd_line is not None) else False
        volume_ok = volume_ratio > VOLUME_RATIO_MIN if volume_ratio is not None else False
        lro_positive = lro > -2 if lro is not None else False

        score = 0
        if price_above_ema_short: score += 20
        if price_above_ema_long: score += 15
        if ema_bullish: score += 15
        if rsi_ok: score += 20
        if macd_bullish: score += 15
        if volume_ok: score += 10
        if lro_positive: score += 5

        signal_ok = (price_above_ema_short and rsi_ok and macd_bullish and 
                    volume_ok and lro_positive)

        return signal_ok, min(score, 100)

    except Exception as e:
        print(f"❌ Error in analyze_timeframe_fast for {symbol} ({timeframe}): {str(e)}")
        return False, 0

def analyze_coin_fast(symbol):
    """Analisis coin dengan cepat"""
    if symbol in failed_coins:
        fail_time = failed_coins[symbol]
        if time.time() - fail_time < 300:
            return None
        else:
            failed_coins.pop(symbol, None)
    
    try:
        data_15m, data_5m = get_two_timeframe_data_fast(symbol)
        
        if data_15m is None or data_5m is None:
            return None
        
        m15_ok, m15_score = analyze_timeframe_fast(data_15m, '15m', symbol)
        m5_ok, m5_score = analyze_timeframe_fast(data_5m, '5m', symbol)
        
        if m15_score is None: m15_score = 0
        if m5_score is None: m5_score = 0
        
        confidence = (m15_score * 0.6) + (m5_score * 0.4)
        confidence = min(95, max(confidence, 0))
        
        current_price = get_current_price(symbol)
        if current_price is None:
            return None
        
        buy_signal = (m15_ok and m5_ok) and confidence >= dynamic_threshold
        
        return {
            'symbol': symbol,
            'buy_signal': buy_signal,
            'confidence': confidence,
            'current_price': current_price,
            'm15_signal': m15_ok,
            'm5_signal': m5_ok,
            'm15_score': m15_score,
            'm5_score': m5_score
        }
        
    except Exception as e:
        print(f"❌ Error in analyze_coin_fast for {symbol}: {str(e)}")
        failed_coins[symbol] = time.time()
        return None

# ==================== ORDER MANAGEMENT ====================
def get_precise_quantity(symbol, investment_amount):
    """Dapatkan quantity yang tepat"""
    try:
        current_price = get_current_price(symbol)
        if not current_price or current_price <= 0:
            return None
            
        theoretical_quantity = investment_amount / current_price
        
        if theoretical_quantity <= 0:
            return None
            
        if not ORDER_RUN:
            return round(theoretical_quantity * 0.998, 6)
            
        precise_quantity = round_step_size(theoretical_quantity, symbol)
        return precise_quantity
        
    except Exception as e:
        print(f"❌ Error di get_precise_quantity: {e}")
        return None

def calculate_position_size(symbol):
    """Hitung ukuran position"""
    global current_investment
    
    position_value = current_investment * POSITION_SIZING_PCT
    
    if position_value < INITIAL_INVESTMENT:
        position_value = INITIAL_INVESTMENT
    
    max_position = current_investment * 0.6
    if position_value > max_position:
        position_value = max_position
    
    if position_value > current_investment:
        position_value = current_investment
    
    return position_value

def place_market_buy_order(symbol, investment_amount):
    """Place market buy order"""
    global current_investment
    
    try:
        print(f"🔹 BUY ORDER: {symbol}")
        
        free_balance = 0
        if ORDER_RUN:
            try:
                rate_limit()
                balance = client.get_asset_balance(asset='USDT')
                free_balance = float(balance['free'])
                if free_balance < investment_amount:
                    print(f"❌ Insufficient balance. Need: ${investment_amount:.2f}, Available: ${free_balance:.2f}")
                    return None
            except Exception as e:
                free_balance = investment_amount * 2
        
        current_price = get_current_price(symbol)
        if not current_price or current_price <= 0:
            return None

        theoretical_quantity = investment_amount / current_price
        
        if theoretical_quantity <= 0:
            return None

        min_notional = get_min_notional(symbol)
        calculated_value = theoretical_quantity * current_price
        
        required_minimum = max(min_notional, INITIAL_INVESTMENT)
        if calculated_value < required_minimum:
            required_investment = required_minimum * 1.02
            theoretical_quantity = required_investment / current_price
            
            if ORDER_RUN and required_investment > free_balance:
                return None

        if not ORDER_RUN:
            precise_quantity = round(theoretical_quantity, 6)
            
            simulated_order = {
                'status': 'FILLED',
                'symbol': symbol,
                'executedQty': str(precise_quantity),
                'fills': [{'price': str(current_price), 'qty': str(precise_quantity)}]
            }
            return simulated_order

        precise_quantity = round_step_size(theoretical_quantity, symbol)
        
        if not precise_quantity:
            return None
            
        final_order_value = precise_quantity * current_price
        
        if final_order_value < required_minimum:
            return None
        
        rate_limit()
        order = client.order_market_buy(
            symbol=symbol,
            quantity=precise_quantity
        )
        
        if order and order.get('status') == 'FILLED':
            print(f"✅ BUY order executed for {symbol}")
            return order
        else:
            print(f"❌ BUY order failed for {symbol}")
            return None
            
    except Exception as e:
        print(f"❌ BUY error: {e}")
        return None

def execute_market_sell(symbol, quantity, entry_price, exit_type):
    """Execute market sell order"""
    global current_investment, active_position
    
    try:
        print(f"🔹 SELL: {symbol} ({exit_type})")
        
        current_price = get_current_price(symbol)
        if not current_price:
            return False

        if ORDER_RUN:
            asset = symbol.replace('USDT', '')
            try:
                rate_limit()
                balance_info = client.get_asset_balance(asset=asset)
                if balance_info:
                    available_balance = float(balance_info['free'])
                    
                    if available_balance <= 0:
                        active_position = None
                        return False
                        
                    if quantity > available_balance:
                        quantity = available_balance
            except Exception as e:
                print(f"   ⚠️ Balance check error: {e}")

        if not ORDER_RUN:
            log_position_closed(symbol, entry_price, current_price, quantity, exit_type)
            active_position = None
            return True

        precise_quantity = round_step_size(quantity, symbol)
        if not precise_quantity:
            precise_quantity = round(quantity, 6)

        if precise_quantity <= 0:
            active_position = None
            return False

        rate_limit()
        sell_order = client.order_market_sell(
            symbol=symbol,
            quantity=precise_quantity
        )

        if sell_order and sell_order.get('status') == 'FILLED':
            executed_qty = float(sell_order.get('executedQty', 0))
            exit_price = current_price
            
            if sell_order.get('fills') and len(sell_order['fills']) > 0:
                exit_price = float(sell_order['fills'][0]['price'])
            
            log_position_closed(symbol, entry_price, exit_price, executed_qty, exit_type)
            active_position = None
            print(f"   ✅ SELL successful: {executed_qty} {symbol} at ${exit_price}")
            return True
            
        active_position = None
        return False

    except Exception as e:
        print(f"❌ SELL error: {e}")
        active_position = None
        return False

# ==================== MONITORING & EXECUTION ====================
def monitor_coins_until_signal():
    """Monitor coins sampai menemukan sinyal buy"""
    global buy_signals, BOT_RUNNING
    
    try:
        client.ping()
        print(f"\n🔍 SCANNING coins until signal found...")
    except Exception as e:
        print(f"❌ BINANCE CONNECTION ERROR: {e}")
        if not handle_binance_error():
            return []
    
    buy_signals = []
    
    for i, coin in enumerate(COINS):
        if not BOT_RUNNING:
            print("🛑 Scanning interrupted by /stop command")
            return []
        
        try:
            analysis = analyze_coin_fast(coin)
            
            if not BOT_RUNNING:
                print("🛑 Analysis interrupted by /stop command")
                return []
            
            if analysis and analysis['buy_signal']:
                print(f"   🚨 SIGNAL FOUND: {coin} - Confidence: {analysis['confidence']:.1f}%")
                buy_signals.append(analysis)
                
                telegram_msg = f"🚨 <b>BUY SIGNAL DETECTED</b>\n• {coin}: {analysis['confidence']:.1f}%\n• Price: ${analysis['current_price']:.6f}"
                send_telegram_message(telegram_msg)
                
                print(f"   ⚡ STOPPING SCAN - Processing {coin} immediately")
                return buy_signals
            
            elif analysis:
                print(f"   ❌ {coin}: No signal ({analysis['confidence']:.1f}%)")
            else:
                print(f"   💀 {coin}: No data")
            
            if TELEGRAM_CONTROL_ENABLED:
                handle_telegram_command()
            
            if i < len(COINS) - 1:
                time.sleep(DELAY_BETWEEN_COINS)
            
        except Exception as e:
            print(f"   ⚠️ {coin}: Error - {e}")
            time.sleep(DELAY_AFTER_ERROR)
            
            if not BOT_RUNNING:
                print("🛑 Error handling interrupted by /stop command")
                return []
    
    print(f"\n📊 SCAN COMPLETE: No signals found in {len(COINS)} coins")
    return []

def fast_execute_signal(signal):
    """Execute trade dengan cepat"""
    global active_position

    if not BOT_RUNNING:
        return False
    
    symbol = signal['symbol']
    confidence = signal['confidence']
    current_price = signal['current_price']
    
    print(f"⚡ EXECUTION: {symbol} (Confidence: {confidence:.1f}%)")
    
    print(f"   🔍 Quick re-confirmation...")
    quick_check = analyze_coin_fast(symbol)

    if not BOT_RUNNING:
        return False
    
    if not quick_check or not quick_check['buy_signal']:
        print(f"   ❌ Signal disappeared, aborting...")
        failed_coins[symbol] = time.time()
        return False
    
    investment_amount = calculate_position_size(symbol)
    
    if not investment_amount:
        return False
    
    buy_order = place_market_buy_order(symbol, investment_amount)
    
    if buy_order and buy_order.get('status') == 'FILLED':
        executed_qty = float(buy_order.get('executedQty', 0))
        
        if buy_order.get('fills') and len(buy_order['fills']) > 0:
            entry_price = float(buy_order['fills'][0]['price'])
        else:
            entry_price = current_price
        
        price_precision = get_price_precision(symbol)
        entry_price = round(entry_price, price_precision)
        take_profit = round(entry_price * (1 + TAKE_PROFIT_PCT), price_precision)
        stop_loss = round(entry_price * (1 - STOP_LOSS_PCT), price_precision)
        
        active_position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'quantity': executed_qty,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'highest_price': entry_price,
            'trailing_active': False,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        log_position_opened(symbol, entry_price, executed_qty, take_profit, stop_loss, confidence)
        
        print(f"   ✅ EXECUTION COMPLETE: {symbol}")
        return True
    
    print(f"   ❌ Execution failed for {symbol}")
    failed_coins[symbol] = time.time()
    return False

# ==================== POSITION MONITORING ====================
def update_trailing_stop(current_price):
    """Update trailing stop"""
    global active_position
    
    if not active_position:
        return
    
    entry_price = active_position['entry_price']
    highest_price = active_position.get('highest_price', entry_price)
    
    if current_price > highest_price:
        active_position['highest_price'] = current_price
        highest_price = current_price
    
    price_increase_pct = (highest_price - entry_price) / entry_price
    
    if price_increase_pct >= TRAILING_STOP_ACTIVATION:
        new_stop_loss = highest_price * (1 - TRAILING_STOP_PCT)
        
        if new_stop_loss > active_position['stop_loss']:
            active_position['stop_loss'] = new_stop_loss
            active_position['trailing_active'] = True

def check_position_exit():
    """Check jika position perlu di-exit"""
    global active_position
    
    if not active_position:
        return
    
    symbol = active_position['symbol']
    
    try:
        current_price = get_current_price(symbol)
        
        if not current_price:
            return
        
        entry_price = active_position['entry_price']
        quantity = active_position['quantity']
        
        update_trailing_stop(current_price)
        
        stop_loss = active_position['stop_loss']
        take_profit = active_position['take_profit']
        
        if current_price <= stop_loss:
            print(f"🛑 Stop Loss hit for {symbol} at ${current_price:.6f}")
            execute_market_sell(symbol, quantity, entry_price, "STOP LOSS")
            return
        
        if current_price >= take_profit:
            print(f"🎯 Take Profit hit for {symbol} at ${current_price:.6f}")
            execute_market_sell(symbol, quantity, entry_price, "TAKE PROFIT")
            return
    except Exception as e:
        print(f"❌ Error checking position exit: {e}")

# ==================== MAIN BOT LOGIC ====================
def main_improved_fast():
    """Main bot function"""
    global current_investment, active_position, BOT_RUNNING
    
    print("🚀 Starting BOT - API ONLY VERSION")

    update_global_variables_from_config()
    initialize_logging()
    load_trade_history()
    load_bot_state()
    
    if not initialize_binance_client():
        return
    
    startup_msg = f"🤖 <b>BOT STARTED - API MODE</b>\nCoins: {len(COINS)}\nMode: {'LIVE' if ORDER_RUN else 'SIMULATION'}\nAPI Key: {CURRENT_API_INDEX + 1}\nModal: ${current_investment:.2f}\nStatus: MENUNGGU PERINTAH /start"
    send_telegram_message(startup_msg)
    
    consecutive_errors = 0
    last_feedback_check = 0
    
    print("✅ Bot siap. Gunakan Telegram untuk mengontrol (/start untuk mulai)")
    
    while True:
        try:
            if TELEGRAM_CONTROL_ENABLED:
                handle_telegram_command()
            
            if not BOT_RUNNING:
                time.sleep(0.5)
                continue
            
            if is_trading_paused():
                time.sleep(DELAY_WHEN_PAUSED)
                continue
            
            try:
                client.ping()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Koneksi error #{consecutive_errors}: {e}")
                if consecutive_errors >= 3:
                    if not handle_binance_error():
                        break
                time.sleep(5)
                continue
            
            current_time = time.time()
            if current_time - last_feedback_check > 300:
                trade_performance_feedback_loop()
                last_feedback_check = current_time
            
            if active_position:
                check_position_exit()
                time.sleep(1)
                continue
            
            print(f"\n🕐 Scanning for signals... (Threshold: {dynamic_threshold}%)")
            signals = monitor_coins_until_signal()
            
            if signals:
                signal = signals[0]
                if fast_execute_signal(signal):
                    print("✅ Trade executed successfully!")
            
            time.sleep(DELAY_BETWEEN_SCANS)
                
            save_bot_state()
                
        except KeyboardInterrupt:
            print(f"\n🛑 Bot stopped by user")
            send_telegram_message("🛑 <b>BOT DIHENTIKAN OLEH PENGGUNA</b>")
            BOT_RUNNING = False
            break
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(5)
    
    save_bot_state()
    print("✅ Bot stopped completely")

def main():
    main_improved_fast()

def keep_alive_ping():
    """Keep-alive ping untuk Render"""
    while True:
        try:
            requests.get("http://localhost:5000/health", timeout=5)
        except:
            print("💓 Keep-alive heartbeat - Bot still running")
        time.sleep(300)

def safe_run_worker():
    """Loop utama bot dengan proteksi error"""
    print("🔄 Running trading bot worker (safe mode)...")
    
    restart_count = 0
    max_restarts = 5
    
    while restart_count < max_restarts:
        try:
            print(f"🔄 Starting bot iteration #{restart_count + 1}")
            main_improved_fast()
            
            if not BOT_RUNNING:
                print("🛑 Bot stopped normally via command")
                break
                
        except Exception as e:
            restart_count += 1
            print(f"⚠️ Worker crashed (attempt {restart_count}/{max_restarts}): {e}")
            time.sleep(10)
        
        BOT_RUNNING = False
        
    if restart_count >= max_restarts:
        error_msg = f"🔴 <b>BOT CRASHED TOO MANY TIMES</b>\nRestarted {max_restarts} times."
        send_telegram_message(error_msg)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ADVANCED TRADING BOT - API ONLY VERSION")
    print("=" * 60)
    
    is_render = check_render_environment()
    
    if is_render:
        health_thread = threading.Thread(target=create_simple_health_endpoint, daemon=True)
        health_thread.start()
        print("🌐 Health endpoint started")
    
    keep_alive_thread = threading.Thread(target=keep_alive_ping, daemon=True)
    keep_alive_thread.start()
    print("💓 Keep-alive thread started")
    
    try:
        safe_run_worker()
    except KeyboardInterrupt:
        print("\n🛑 Bot shutdown requested")
        BOT_RUNNING = False
    except Exception as e:
        print(f"💀 Fatal error: {e}")
        send_telegram_message(f"🔴 <b>FATAL ERROR</b>\n{str(e)}")
    
    print("✅ Bot shutdown complete")


