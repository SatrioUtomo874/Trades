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
import websocket
from flask import Flask

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
# Gunakan API key dari environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

INITIAL_INVESTMENT = float(os.getenv('INITIAL_INVESTMENT', '5.5'))
global ORDER_RUN
ORDER_RUN = os.getenv('ORDER_RUN', 'False').lower() == 'true'

# Trading Parameters
TAKE_PROFIT_PCT = 0.0062
STOP_LOSS_PCT = 0.0160
TRAILING_STOP_ACTIVATION = 0.0040
TRAILING_STOP_PCT = 0.0080
POSITION_SIZING_PCT = 0.4

# Technical Parameters
RSI_MIN = 35
RSI_MAX = 65
EMA_SHORT = 12
EMA_LONG = 26
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_PERIOD = 20
VOLUME_RATIO_MIN = 0.8

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
SEND_TELEGRAM_NOTIFICATIONS = True

# Telegram Control
TELEGRAM_CONTROL_ENABLED = True
BOT_RUNNING = False
ADMIN_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Configuration file
CONFIG_FILE = 'bot_config.json'

# File Configuration
LOG_FILE = 'trading_log1.txt'
TRADE_HISTORY_FILE = 'trade_history1.json'

# Coin List (disederhanakan untuk testing)
COINS = [
    'PENGUUSDT','WALUSDT','MIRAUSDT','HEMIUSDT','PUMPUSDT','TRXUSDT','LTCUSDT','FFUSDT',
    'SUIUSDT','ASTERUSDT','ZECUSDT','CAKEUSDT','BNBUSDT','AVNTUSDT','DOGEUSDT','ADAUSDT',
    'XPLUSDT','XRPUSDT','DASHUSDT','SOLUSDT','LINKUSDT','AVAXUSDT', 'PEPEUSDT'
]
# ==================== VARIABEL GLOBAL ====================
current_investment = INITIAL_INVESTMENT
active_position = None
trade_history = []
client = None

# WebSocket Data Storage
websocket_data = {}
price_data = {}

# Performance tracking
performance_state = {
    'total_trades': 0,
    'total_wins': 0,
    'consecutive_losses': 0
}

# ==================== HEALTH ENDPOINT ====================
def create_health_endpoint():
    """Health endpoint untuk Render dengan port yang benar"""
    try:
        app = Flask(__name__)
        
        @app.route('/')
        def health_check():
            return {
                'status': 'running', 
                'timestamp': datetime.now().isoformat(),
                'bot_running': BOT_RUNNING,
                'coins_monitored': len(COINS)
            }
        
        @app.route('/health')
        def health():
            return {'status': 'healthy'}
        
        # Gunakan port dari environment variable PORT, default 8080
        port = int(os.environ.get('PORT', 8080))
        
        def run_flask():
            print(f"🌐 Starting health endpoint on port {port}")
            app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        print(f"✅ Health endpoint started on port {port}")
        return True
    except Exception as e:
        print(f"⚠️ Health endpoint error: {e}")
        return False

# ==================== WEB SOCKET MANAGER ====================
class BinanceWebSocketManager:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    def start_kline_stream(self, symbols, interval='5m'):
        """Start WebSocket for kline data dengan reconnection logic"""
        try:
            streams = [f"{symbol.lower()}@kline_{interval}" for symbol in symbols]
            stream_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
            
            print(f"🔗 Connecting WebSocket for {len(symbols)} symbols...")
            
            self.ws = websocket.WebSocketApp(
                stream_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            def run_websocket():
                try:
                    self.ws.run_forever()
                except Exception as e:
                    print(f"WebSocket run_forever error: {e}")
                    self.schedule_reconnect()
            
            ws_thread = threading.Thread(target=run_websocket, daemon=True)
            ws_thread.start()
            
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            self.schedule_reconnect()
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            stream = data.get('stream', '')
            
            if 'kline_' in stream:
                symbol = stream.split('@')[0].upper()
                kline = data['data']['k']
                
                if symbol not in websocket_data:
                    websocket_data[symbol] = {
                        'close': deque(maxlen=100),
                        'high': deque(maxlen=100),
                        'low': deque(maxlen=100),
                        'volume': deque(maxlen=100),
                        'timestamp': deque(maxlen=100)
                    }
                
                # Always update current price
                current_price = float(kline['c'])
                price_data[symbol] = current_price
                
                # Update historical data only when kline is closed
                if kline['x']:  # Kline is closed
                    websocket_data[symbol]['close'].append(current_price)
                    websocket_data[symbol]['high'].append(float(kline['h']))
                    websocket_data[symbol]['low'].append(float(kline['l']))
                    websocket_data[symbol]['volume'].append(float(kline['v']))
                    websocket_data[symbol]['timestamp'].append(kline['t'])
                
        except Exception as e:
            print(f"WebSocket message error: {e}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        self.connected = False

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed")
        self.connected = False
        self.schedule_reconnect()

    def on_open(self, ws):
        print("✅ WebSocket connected successfully")
        self.connected = True
        self.reconnect_attempts = 0  # Reset reconnect attempts on successful connection

    def schedule_reconnect(self):
        """Schedule WebSocket reconnection"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(30, 5 * self.reconnect_attempts)  # Exponential backoff: 5, 10, 15, 20, 30 seconds
            print(f"🔄 Attempting WebSocket reconnect #{self.reconnect_attempts} in {delay} seconds...")
            threading.Timer(delay, self.reconnect).start()
        else:
            print("❌ Max reconnection attempts reached. WebSocket connection failed.")

    def reconnect(self):
        """Reconnect WebSocket"""
        print("Attempting to reconnect WebSocket...")
        self.start_kline_stream(COINS)

# ==================== BINANCE CLIENT MANAGEMENT ====================
def initialize_binance_client():
    """Initialize Binance client dengan error handling"""
    global client, ORDER_RUN
    
    if not ORDER_RUN:
        print("🔄 Simulation mode - No Binance client needed")
        return True
        
    try:
        if not API_KEY or not API_SECRET:
            print("❌ API key or secret not found")
            return False
            
        print("🔧 Initializing Binance client...")
        
        # Gunakan timeout yang lebih pendek dan configuration yang lebih aman
        client = Client(
            API_KEY, 
            API_SECRET, 
            {
                "timeout": 15,
                "requests_params": {"timeout": 15}
            }
        )
        
        # Test connection dengan method yang lebih ringan
        server_time = client.get_server_time()
        print(f"✅ Binance client initialized successfully. Server time: {server_time['serverTime']}")
        return True
        
    except Exception as e:
        print(f"❌ Binance client initialization failed: {e}")
        
        # Jika terkena IP ban, gunakan mode simulasi
        if "IP banned" in str(e) or "too much request" in str(e):
            print("🚫 IP Banned detected. Switching to simulation mode...")
            ORDER_RUN = False
            send_telegram_message("⚠️ <b>IP BAN DETECTED</b>\nSwitching to simulation mode automatically.")
            return True
            
        return False

# ==================== TELEGRAM COMMAND SYSTEM ====================
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
        response = requests.post(url, data=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

def handle_telegram_command():
    """Check untuk Telegram commands"""
    global BOT_RUNNING
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        response = requests.get(url, timeout=10)
        
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
    global BOT_RUNNING, COINS, current_investment
    
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
            
        elif command.startswith('/modal'):
            handle_modal_command(command, chat_id)

        mark_update_processed(update_id)
        
    except Exception as e:
        send_telegram_message(f"❌ <b>ERROR PROCESSING COMMAND</b>\n{str(e)}")

def parse_float_value(input_str):
    """Parse string menjadi float, handle berbagai format angka"""
    try:
        cleaned = input_str.replace(',', '.').replace(' ', '')
        if not cleaned.replace('.', '').isdigit():
            return None
        value = float(cleaned)
        return value
    except (ValueError, AttributeError):
        return None

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
            
        new_investment = parse_float_value(parts[1])
        
        if new_investment is None:
            send_telegram_message("❌ Format angka tidak valid. Gunakan angka dengan titik atau koma.\nContoh: /modal 5.4 atau /modal 5,4")
            return
        
        if new_investment <= 0:
            send_telegram_message("❌ Modal harus lebih besar dari 0")
            return
        
        old_investment = current_investment
        current_investment = new_investment
        
        send_telegram_message(f"✅ <b>MODAL DIPERBARUI</b>\nModal sebelumnya: ${old_investment:.2f}\nModal baru: <b>${current_investment:.2f}</b>")
        print(f"✅ Investment updated: ${old_investment:.2f} -> ${current_investment:.2f}")
        
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
    current_price = price_data.get(symbol, entry_price)
    
    parts = command.split()
    
    if len(parts) == 1:
        send_telegram_message(f"🔄 <b>MENJALANKAN SELL MANUAL</b>\nSymbol: {symbol}\nQuantity: {quantity:.6f}")
        
        success = execute_market_sell(symbol, quantity, entry_price, "MANUAL SELL")
        
        if success:
            send_telegram_message(f"✅ <b>SELL MANUAL BERHASIL</b>\n{symbol} telah dijual.")
        else:
            send_telegram_message(f"❌ <b>SELL MANUAL GAGAL</b>\nGagal menjual {symbol}.")
    
    elif len(parts) >= 3:
        action = parts[1].lower()
        price_value = parse_float_value(parts[2])
        
        if price_value is None:
            send_telegram_message("❌ Format harga tidak valid. Gunakan angka dengan titik atau koma.\nContoh: /sell sl 0.0539 atau /sell sl 0,0539")
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
            
            old_sl = active_position['stop_loss']
            active_position['stop_loss'] = price_value
            send_telegram_message(f"✅ <b>STOP LOSS DIPERBARUI</b>\n{symbol}\nSL sebelumnya: ${old_sl:.6f}\nSL baru: <b>${price_value:.6f}</b>")
            print(f"✅ SL updated for {symbol}: {price_value}")
            
        else:
            send_telegram_message("❌ Format: /sell [tp/sl] [harga]\nContoh: /sell tp 2.987\n/sell sl 2.500")
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
    
    if coins_count <= 20:
        coins_list = "\n".join([f"• {coin}" for coin in sorted(COINS)])
    else:
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
        
        valid_params = {
            'TAKE_PROFIT_PCT': ('trading_params', float),
            'STOP_LOSS_PCT': ('trading_params', float),
            'TRAILING_STOP_ACTIVATION': ('trading_params', float),
            'TRAILING_STOP_PCT': ('trading_params', float),
            'POSITION_SIZING_PCT': ('trading_params', float),
            'RSI_MIN': ('technical_params', int),
            'RSI_MAX': ('technical_params', int),
            'EMA_SHORT': ('technical_params', int),
            'EMA_LONG': ('technical_params', int),
            'MACD_FAST': ('technical_params', int),
            'MACD_SLOW': ('technical_params', int),
            'MACD_SIGNAL': ('technical_params', int),
            'VOLUME_RATIO_MIN': ('technical_params', float)
        }
        
        if param_name in valid_params:
            section, value_type = valid_params[param_name]
            old_value = config[section][param_name]
            
            try:
                if value_type == float:
                    parsed_value = parse_float_value(param_value)
                    if parsed_value is None:
                        send_telegram_message(f"❌ Format angka tidak valid untuk {param_name}\nGunakan titik atau koma sebagai desimal")
                        return
                    new_value = parsed_value
                else:
                    new_value = value_type(param_value)
                
                if param_name.endswith('_PCT') and new_value <= 0:
                    send_telegram_message(f"❌ Nilai {param_name} harus lebih besar dari 0")
                    return
                elif param_name.startswith('RSI_') and (new_value < 0 or new_value > 100):
                    send_telegram_message(f"❌ Nilai {param_name} harus antara 0-100")
                    return
                elif param_name.startswith(('EMA_', 'MACD_')) and new_value <= 0:
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
    
    winrate = calculate_winrate()
    
    status_msg = (
        f"🤖 <b>BOT STATUS</b>\n"
        f"Status: {'🟢 BERJALAN' if BOT_RUNNING else '🔴 BERHENTI'}\n"
        f"Modal: ${current_investment:.2f}\n"
        f"Total Trade: {len(trade_history)}\n"
        f"Win Rate: {winrate:.1f}%\n"
        f"Mode: {'LIVE' if ORDER_RUN else 'SIMULATION'}\n"
    )
    
    if active_position:
        current_price = price_data.get(active_position['symbol'], active_position['entry_price'])
        pnl_pct = (current_price - active_position['entry_price']) / active_position['entry_price'] * 100
        status_msg += f"Posisi Aktif: {active_position['symbol']}\n"
        status_msg += f"Entry: ${active_position['entry_price']:.6f}\n"
        status_msg += f"Current: ${current_price:.6f}\n"
        status_msg += f"TP: ${active_position['take_profit']:.6f}\n"
        status_msg += f"SL: ${active_position['stop_loss']:.6f}\n"
        status_msg += f"PnL: {pnl_pct:+.2f}%\n"
        status_msg += f"Gunakan /sell untuk close manual\n"
    else:
        status_msg += "Posisi Aktif: Tidak ada\n"
    
    send_telegram_message(status_msg)

def send_current_config(chat_id):
    """Send current configuration"""
    config_msg = (
        f"⚙️ <b>KONFIGURASI SAAT INI</b>\n"
        f"┌────────────────────────────┐\n"
        f"│ <b>TRADING PARAMS</b>\n"
        f"│ TP: {TAKE_PROFIT_PCT*100:.2f}%\n"
        f"│ SL: {STOP_LOSS_PCT*100:.2f}%\n"
        f"│ Position: {POSITION_SIZING_PCT*100:.1f}%\n"
        f"│ Trailing Act: {TRAILING_STOP_ACTIVATION*100:.2f}%\n"
        f"│ Trailing SL: {TRAILING_STOP_PCT*100:.2f}%\n"
        f"├────────────────────────────┤\n"
        f"│ <b>TECHNICAL PARAMS</b>\n"
        f"│ RSI: {RSI_MIN}-{RSI_MAX}\n"
        f"│ EMA: {EMA_SHORT}/{EMA_LONG}\n"
        f"│ MACD: {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}\n"
        f"│ Volume Ratio: {VOLUME_RATIO_MIN}\n"
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
        f"│ /set RSI_MIN 30\n"
        f"│ /set POSITION_SIZING_PCT 0.3\n"
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

# ==================== CONFIGURATION MANAGEMENT ====================
def load_config():
    """Load configuration dari file"""
    default_config = {
        'trading_params': {
            'TAKE_PROFIT_PCT': TAKE_PROFIT_PCT,
            'STOP_LOSS_PCT': STOP_LOSS_PCT,
            'TRAILING_STOP_ACTIVATION': TRAILING_STOP_ACTIVATION,
            'TRAILING_STOP_PCT': TRAILING_STOP_PCT,
            'POSITION_SIZING_PCT': POSITION_SIZING_PCT
        },
        'technical_params': {
            'RSI_MIN': RSI_MIN,
            'RSI_MAX': RSI_MAX,
            'EMA_SHORT': EMA_SHORT,
            'EMA_LONG': EMA_LONG,
            'MACD_FAST': MACD_FAST,
            'MACD_SLOW': MACD_SLOW,
            'MACD_SIGNAL': MACD_SIGNAL,
            'VOLUME_RATIO_MIN': VOLUME_RATIO_MIN
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
    global TAKE_PROFIT_PCT, STOP_LOSS_PCT, TRAILING_STOP_ACTIVATION, TRAILING_STOP_PCT, POSITION_SIZING_PCT
    global RSI_MIN, RSI_MAX, EMA_SHORT, EMA_LONG, MACD_FAST, MACD_SLOW, MACD_SIGNAL, VOLUME_RATIO_MIN
    
    config = load_config()
    trading_params = config['trading_params']
    technical_params = config['technical_params']
    
    TAKE_PROFIT_PCT = trading_params['TAKE_PROFIT_PCT']
    STOP_LOSS_PCT = trading_params['STOP_LOSS_PCT']
    TRAILING_STOP_ACTIVATION = trading_params['TRAILING_STOP_ACTIVATION']
    TRAILING_STOP_PCT = trading_params['TRAILING_STOP_PCT']
    POSITION_SIZING_PCT = trading_params['POSITION_SIZING_PCT']
    
    RSI_MIN = technical_params['RSI_MIN']
    RSI_MAX = technical_params['RSI_MAX']
    EMA_SHORT = technical_params['EMA_SHORT']
    EMA_LONG = technical_params['EMA_LONG']
    MACD_FAST = technical_params['MACD_FAST']
    MACD_SLOW = technical_params['MACD_SLOW']
    MACD_SIGNAL = technical_params['MACD_SIGNAL']
    VOLUME_RATIO_MIN = technical_params['VOLUME_RATIO_MIN']

# ==================== DATA MANAGEMENT ====================
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

# ==================== TECHNICAL INDICATORS (WebSocket-based) ====================
def calculate_ema(prices, period):
    """Calculate EMA dari data WebSocket"""
    if len(prices) < period:
        return None
    try:
        series = pd.Series(prices)
        return series.ewm(span=period, adjust=False).mean().iloc[-1]
    except:
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI dari data WebSocket"""
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
    """Calculate MACD dari data WebSocket"""
    if len(prices) < slow:
        return None, None, None
    
    try:
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        
        return (
            macd_line.iloc[-1],
            macd_signal.iloc[-1],
            macd_histogram.iloc[-1]
        )
    except:
        return None, None, None

def calculate_volume_ratio(volumes, period=20):
    """Calculate volume ratio dari data WebSocket"""
    if len(volumes) < period:
        return 1.0
    
    try:
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-period:])
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    except:
        return 1.0

# ==================== TRADING LOGIC (WebSocket-based) ====================
def analyze_symbol(symbol):
    """Analyze symbol menggunakan data WebSocket"""
    if symbol not in websocket_data or len(websocket_data[symbol]['close']) < 26:
        return None
    
    try:
        closes = list(websocket_data[symbol]['close'])
        volumes = list(websocket_data[symbol]['volume'])
        current_price = price_data.get(symbol, closes[-1] if closes else 0)
        
        # Calculate indicators from WebSocket data
        ema_short = calculate_ema(closes, EMA_SHORT)
        ema_long = calculate_ema(closes, EMA_LONG)
        rsi = calculate_rsi(closes)
        macd_line, macd_signal, _ = calculate_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        volume_ratio = calculate_volume_ratio(volumes)
        
        if any(x is None for x in [ema_short, ema_long, rsi, macd_line]):
            return None
        
        # Trading conditions
        price_above_ema_short = current_price > ema_short
        price_above_ema_long = current_price > ema_long
        ema_bullish = ema_short > ema_long
        rsi_ok = RSI_MIN <= rsi <= RSI_MAX
        macd_bullish = macd_line > macd_signal if macd_signal else False
        volume_ok = volume_ratio > VOLUME_RATIO_MIN
        
        # Calculate confidence score
        score = 0
        if price_above_ema_short: score += 25
        if price_above_ema_long: score += 20
        if ema_bullish: score += 20
        if rsi_ok: score += 20
        if macd_bullish: score += 10
        if volume_ok: score += 5
        
        buy_signal = (price_above_ema_short and price_above_ema_long and 
                     ema_bullish and rsi_ok and macd_bullish and volume_ok)
        
        return {
            'symbol': symbol,
            'buy_signal': buy_signal,
            'confidence': min(score, 100),
            'current_price': current_price,
            'rsi': rsi,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'volume_ratio': volume_ratio
        }
        
    except Exception as e:
        print(f"Analysis error for {symbol}: {e}")
        return None

# ==================== ORDER MANAGEMENT ====================
def calculate_position_size():
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
        
        current_price = price_data.get(symbol)
        if not current_price or current_price <= 0:
            return None

        theoretical_quantity = investment_amount / current_price
        
        if theoretical_quantity <= 0:
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

        # Live trading
        try:
            # Get balance
            balance = client.get_asset_balance(asset='USDT')
            free_balance = float(balance['free'])
            
            if free_balance < investment_amount:
                print(f"❌ Insufficient balance. Need: ${investment_amount:.2f}, Available: ${free_balance:.2f}")
                return None
                
            # Place order
            order = client.order_market_buy(
                symbol=symbol,
                quoteOrderQty=investment_amount  # Use quote order quantity for precise amount
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
            
    except Exception as e:
        print(f"❌ BUY execution error: {e}")
        return None

def execute_market_sell(symbol, quantity, entry_price, exit_type):
    """Execute market sell order"""
    global current_investment, active_position
    
    try:
        print(f"🔹 SELL: {symbol} ({exit_type})")
        
        current_price = price_data.get(symbol, entry_price)

        if not ORDER_RUN:
            log_position_closed(symbol, entry_price, current_price, quantity, exit_type)
            active_position = None
            return True

        # Live trading
        try:
            # Get balance
            asset = symbol.replace('USDT', '')
            balance_info = client.get_asset_balance(asset=asset)
            if balance_info:
                available_balance = float(balance_info['free'])
                
                if available_balance <= 0:
                    active_position = None
                    return False
                    
                if quantity > available_balance:
                    quantity = available_balance

            # Place sell order
            sell_order = client.order_market_sell(
                symbol=symbol,
                quantity=round(quantity, 6)  # Simple rounding for demo
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
            print(f"❌ SELL execution error: {e}")
            active_position = None
            return False

    except Exception as e:
        print(f"❌ SELL error: {e}")
        active_position = None
        return False

# ==================== LOGGING ====================
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
        
        performance_state['total_trades'] += 1
        if pnl > 0:
            performance_state['total_wins'] += 1
            performance_state['consecutive_losses'] = 0
        else:
            performance_state['consecutive_losses'] += 1
        
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
        current_price = price_data.get(symbol, active_position['entry_price'])
        
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
def main():
    global BOT_RUNNING, current_investment, active_position, ORDER_RUN
    
    print("🚀 Starting BOT - WebSocket Version")
    
    # Start health endpoint first
    create_health_endpoint()
    
    # Load configuration and data
    update_global_variables_from_config()
    load_trade_history()
    
    # Initialize Binance client (only if ORDER_RUN is True)
    if not initialize_binance_client():
        print("❌ Failed to initialize Binance client. Continuing in simulation mode...")
        global ORDER_RUN
        ORDER_RUN = False
    
    # Initialize WebSocket
    ws_manager = BinanceWebSocketManager()
    ws_manager.start_kline_stream(COINS)
    
    # Wait for WebSocket to connect and collect initial data
    print("⏳ Collecting initial market data (30 seconds)...")
    for i in range(30):
        if not BOT_RUNNING:
            break
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/30 seconds...")
    
    connected_symbols = len([s for s in COINS if s in websocket_data])
    print(f"✅ WebSocket data collected for {connected_symbols}/{len(COINS)} symbols")
    
    startup_msg = (f"🤖 <b>BOT STARTED - WebSocket MODE</b>\n"
                  f"Coins: {len(COINS)} ({connected_symbols} connected)\n"
                  f"Mode: {'LIVE' if ORDER_RUN else 'SIMULATION'}\n"
                  f"Modal: ${current_investment:.2f}\n"
                  f"Status: MENUNGGU PERINTAH /start")
    send_telegram_message(startup_msg)
    
    last_scan_time = 0
    scan_interval = 10  # seconds
    
    print("✅ Bot siap. Gunakan Telegram untuk mengontrol (/start untuk mulai)")
    
    while True:
        try:
            if TELEGRAM_CONTROL_ENABLED:
                handle_telegram_command()
            
            if not BOT_RUNNING:
                time.sleep(1)
                continue
            
            current_time = time.time()
            
            # Monitor active position
            if active_position:
                check_position_exit()
                time.sleep(1)
                continue
            
            # Scan for new signals
            if current_time - last_scan_time >= scan_interval:
                print("🔍 Scanning for signals...")
                
                best_signal = None
                signals_found = 0
                
                for symbol in COINS:
                    if not BOT_RUNNING:
                        break
                    
                    analysis = analyze_symbol(symbol)
                    
                    if analysis and analysis['buy_signal']:
                        signals_found += 1
                        if not best_signal or analysis['confidence'] > best_signal['confidence']:
                            best_signal = analysis
                
                if best_signal and best_signal['confidence'] > 60:
                    print(f"🎯 Signal found: {best_signal['symbol']} (Confidence: {best_signal['confidence']:.1f}%)")
                    
                    investment_amount = calculate_position_size()
                    buy_order = place_market_buy_order(best_signal['symbol'], investment_amount)
                    
                    if buy_order and buy_order.get('status') == 'FILLED':
                        executed_qty = float(buy_order.get('executedQty', 0))
                        
                        if buy_order.get('fills') and len(buy_order['fills']) > 0:
                            entry_price = float(buy_order['fills'][0]['price'])
                        else:
                            entry_price = best_signal['current_price']
                        
                        price_precision = 6
                        entry_price = round(entry_price, price_precision)
                        take_profit = round(entry_price * (1 + TAKE_PROFIT_PCT), price_precision)
                        stop_loss = round(entry_price * (1 - STOP_LOSS_PCT), price_precision)
                        
                        active_position = {
                            'symbol': best_signal['symbol'],
                            'entry_price': entry_price,
                            'quantity': executed_qty,
                            'take_profit': take_profit,
                            'stop_loss': stop_loss,
                            'highest_price': entry_price,
                            'trailing_active': False,
                            'confidence': best_signal['confidence'],
                            'timestamp': time.time()
                        }
                        
                        log_position_opened(best_signal['symbol'], entry_price, executed_qty, take_profit, stop_loss, best_signal['confidence'])
                
                print(f"📊 Scan complete: {signals_found} signals found")
                last_scan_time = current_time
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            send_telegram_message("🛑 <b>BOT DIHENTIKAN OLEH PENGGUNA</b>")
            BOT_RUNNING = False
            break
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(5)
    
    print("✅ Bot stopped completely")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ADVANCED TRADING BOT - WebSocket VERSION")
    print("=" * 60)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot shutdown requested")
        BOT_RUNNING = False
    except Exception as e:
        print(f"💀 Fatal error: {e}")
        send_telegram_message(f"🔴 <b>FATAL ERROR</b>\n{str(e)}")
    
    print("✅ Bot shutdown complete")




