import os
import time
import json
import threading
from datetime import datetime
from dotenv import load_dotenv
import websocket
import requests
from binance.client import Client
import pandas as pd
import numpy as np
from collections import deque

# Load environment variables
load_dotenv()

# ==================== KONFIGURASI ====================
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

INITIAL_INVESTMENT = float(os.getenv('INITIAL_INVESTMENT', '5.5'))
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

# Coin List
COINS = [
    'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
    'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'XRPUSDT', 'EOSUSDT'
]

# ==================== VARIABEL GLOBAL ====================
current_investment = INITIAL_INVESTMENT
active_position = None
trade_history = []
client = None
BOT_RUNNING = False

# WebSocket data storage
market_data = {}
price_data = {}

# Performance tracking
performance_state = {
    'total_trades': 0,
    'total_wins': 0,
    'consecutive_losses': 0
}

# ==================== WEB SOCKET IMPLEMENTATION ====================
class BinanceWebSocket:
    def __init__(self):
        self.ws = None
        self.connected = False
        
    def start_kline_stream(self, symbols, interval='5m'):
        """Start WebSocket for kline data"""
        streams = [f"{symbol.lower()}@kline_{interval}" for symbol in symbols]
        stream_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        self.ws = websocket.WebSocketApp(
            stream_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        def run_websocket():
            self.ws.run_forever()
            
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            stream = data.get('stream', '')
            
            if 'kline_' in stream:
                symbol = stream.split('@')[0].upper()
                kline = data['data']['k']
                
                if symbol not in market_data:
                    market_data[symbol] = {
                        'close': deque(maxlen=100),
                        'high': deque(maxlen=100),
                        'low': deque(maxlen=100),
                        'volume': deque(maxlen=100),
                        'timestamp': deque(maxlen=100)
                    }
                
                # Update market data
                if kline['x']:  # Kline is closed
                    market_data[symbol]['close'].append(float(kline['c']))
                    market_data[symbol]['high'].append(float(kline['h']))
                    market_data[symbol]['low'].append(float(kline['l']))
                    market_data[symbol]['volume'].append(float(kline['v']))
                    market_data[symbol]['timestamp'].append(kline['t'])
                
                # Update current price
                price_data[symbol] = float(kline['c'])
                
        except Exception as e:
            print(f"WebSocket message error: {e}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed")
        self.connected = False
        # Attempt reconnect after 10 seconds
        threading.Timer(10, self.reconnect).start()

    def on_open(self, ws):
        print("WebSocket connected")
        self.connected = True

    def reconnect(self):
        """Reconnect WebSocket"""
        print("Attempting to reconnect WebSocket...")
        self.start_kline_stream(COINS)

# ==================== TELEGRAM FUNCTIONS ====================
def send_telegram_message(message):
    """Send message to Telegram"""
    try:
        if len(message) > 4000:
            message = message[:4000] + "..."
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")

def handle_telegram_command():
    """Handle Telegram commands"""
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
                        
                        if str(chat_id) != TELEGRAM_CHAT_ID:
                            continue
                            
                        if message == '/start':
                            global BOT_RUNNING
                            BOT_RUNNING = True
                            send_telegram_message("🤖 <b>BOT DIAKTIFKAN</b>")
                            
                        elif message == '/stop':
                            BOT_RUNNING = False
                            send_telegram_message("🛑 <b>BOT DIHENTIKAN</b>")
                            
                        elif message == '/status':
                            send_status()
                            
    except Exception as e:
        print(f"Telegram command error: {e}")

# ==================== TECHNICAL INDICATORS ====================
def calculate_ema(prices, period):
    """Calculate EMA"""
    if len(prices) < period:
        return None
    try:
        series = pd.Series(prices)
        return series.ewm(span=period, adjust=False).mean().iloc[-1]
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

def calculate_macd(prices):
    """Calculate MACD"""
    if len(prices) < MACD_SLOW:
        return None, None, None
    
    try:
        ema_fast = pd.Series(prices).ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=MACD_SLOW, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        
        return (
            macd_line.iloc[-1],
            macd_signal.iloc[-1],
            macd_histogram.iloc[-1]
        )
    except:
        return None, None, None

def calculate_volume_ratio(volumes, period=VOLUME_PERIOD):
    """Calculate volume ratio"""
    if len(volumes) < period:
        return 1.0
    
    try:
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-period:])
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    except:
        return 1.0

# ==================== TRADING LOGIC ====================
def analyze_symbol(symbol):
    """Analyze symbol for trading signals"""
    if symbol not in market_data or len(market_data[symbol]['close']) < 26:
        return None
    
    try:
        closes = list(market_data[symbol]['close'])
        volumes = list(market_data[symbol]['volume'])
        current_price = price_data.get(symbol, closes[-1] if closes else 0)
        
        # Calculate indicators
        ema_short = calculate_ema(closes, EMA_SHORT)
        ema_long = calculate_ema(closes, EMA_LONG)
        rsi = calculate_rsi(closes)
        macd_line, macd_signal, macd_histogram = calculate_macd(closes)
        volume_ratio = calculate_volume_ratio(volumes)
        
        if any(x is None for x in [ema_short, ema_long, rsi, macd_line]):
            return None
        
        # Trading conditions
        price_above_ema_short = current_price > ema_short
        price_above_ema_long = current_price > ema_long
        ema_bullish = ema_short > ema_long
        rsi_ok = RSI_MIN <= rsi <= RSI_MAX
        macd_bullish = macd_line > macd_signal if macd_signal else False
        volume_ok = volume_ratio > 1.0
        
        # Calculate confidence score
        score = 0
        if price_above_ema_short: score += 25
        if price_above_ema_long: score += 20
        if ema_bullish: score += 20
        if rsi_ok: score += 20
        if macd_bullish: score += 10
        if volume_ok: score += 5
        
        buy_signal = (price_above_ema_short and price_above_ema_long and 
                     ema_bullish and rsi_ok and macd_bullish)
        
        return {
            'symbol': symbol,
            'buy_signal': buy_signal,
            'confidence': min(score, 100),
            'current_price': current_price,
            'rsi': rsi,
            'ema_short': ema_short,
            'ema_long': ema_long
        }
        
    except Exception as e:
        print(f"Analysis error for {symbol}: {e}")
        return None

def calculate_position_size():
    """Calculate position size"""
    global current_investment
    return current_investment * POSITION_SIZING_PCT

def execute_buy(symbol, analysis):
    """Execute buy order"""
    global active_position, current_investment
    
    try:
        investment_amount = calculate_position_size()
        current_price = analysis['current_price']
        
        if ORDER_RUN:
            # Real order execution
            quantity = investment_amount / current_price
            precision = get_quantity_precision(symbol)
            quantity = round(quantity, precision)
            
            order = client.order_market_buy(
                symbol=symbol,
                quantity=quantity
            )
            
            if order and order.get('status') == 'FILLED':
                executed_qty = float(order.get('executedQty', 0))
                entry_price = current_price
                
                if order.get('fills'):
                    entry_price = float(order['fills'][0]['price'])
        else:
            # Simulation
            executed_qty = investment_amount / current_price
            entry_price = current_price
        
        # Set take profit and stop loss
        take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
        stop_loss = entry_price * (1 - STOP_LOSS_PCT)
        
        active_position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'quantity': executed_qty,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'highest_price': entry_price
        }
        
        # Log the trade
        log_trade('BUY', symbol, entry_price, executed_qty, analysis['confidence'])
        
        message = (f"📈 <b>POSITION OPENED</b>\n"
                  f"Symbol: {symbol}\n"
                  f"Entry: ${entry_price:.6f}\n"
                  f"Quantity: {executed_qty:.6f}\n"
                  f"TP: ${take_profit:.6f}\n"
                  f"SL: ${stop_loss:.6f}")
        send_telegram_message(message)
        
        return True
        
    except Exception as e:
        print(f"Buy execution error: {e}")
        return False

def execute_sell(exit_type):
    """Execute sell order"""
    global active_position, current_investment
    
    if not active_position:
        return False
    
    try:
        symbol = active_position['symbol']
        quantity = active_position['quantity']
        entry_price = active_position['entry_price']
        current_price = price_data.get(symbol, entry_price)
        
        if ORDER_RUN:
            # Real order execution
            precision = get_quantity_precision(symbol)
            quantity = round(quantity, precision)
            
            order = client.order_market_sell(
                symbol=symbol,
                quantity=quantity
            )
            
            if order and order.get('status') == 'FILLED':
                if order.get('fills'):
                    current_price = float(order['fills'][0]['price'])
        # For simulation, we use current price
        
        # Calculate PnL
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        current_investment += pnl
        
        # Update performance
        performance_state['total_trades'] += 1
        if pnl > 0:
            performance_state['total_wins'] += 1
            performance_state['consecutive_losses'] = 0
        else:
            performance_state['consecutive_losses'] += 1
        
        # Log the trade
        log_trade('SELL', symbol, current_price, quantity, 0, exit_type, pnl, pnl_pct)
        
        message = (f"📉 <b>POSITION CLOSED - {exit_type}</b>\n"
                  f"Symbol: {symbol}\n"
                  f"Exit: ${current_price:.6f}\n"
                  f"PnL: {pnl_pct:+.2f}% (${pnl:.4f})\n"
                  f"Capital: ${current_investment:.2f}")
        send_telegram_message(message)
        
        active_position = None
        return True
        
    except Exception as e:
        print(f"Sell execution error: {e}")
        return False

def monitor_position():
    """Monitor active position for exit conditions"""
    if not active_position:
        return
    
    symbol = active_position['symbol']
    current_price = price_data.get(symbol, 0)
    entry_price = active_position['entry_price']
    
    if current_price <= active_position['stop_loss']:
        print(f"Stop loss hit for {symbol}")
        execute_sell("STOP LOSS")
        return
    
    if current_price >= active_position['take_profit']:
        print(f"Take profit hit for {symbol}")
        execute_sell("TAKE PROFIT")
        return
    
    # Update trailing stop
    if current_price > active_position['highest_price']:
        active_position['highest_price'] = current_price
        
        price_increase = (current_price - entry_price) / entry_price
        if price_increase >= TRAILING_STOP_ACTIVATION:
            new_stop_loss = current_price * (1 - TRAILING_STOP_PCT)
            if new_stop_loss > active_position['stop_loss']:
                active_position['stop_loss'] = new_stop_loss

# ==================== UTILITY FUNCTIONS ====================
def get_quantity_precision(symbol):
    """Get quantity precision for symbol"""
    try:
        info = client.get_symbol_info(symbol)
        if info:
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
                    precision = 0
                    while step_size < 1:
                        step_size *= 10
                        precision += 1
                    return precision
        return 6
    except:
        return 6

def log_trade(action, symbol, price, quantity, confidence=0, exit_type="", pnl=0, pnl_pct=0):
    """Log trade to file"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if action == 'BUY':
            log_entry = f"BUY | {symbol} | Price: ${price:.6f} | Qty: {quantity:.6f} | Confidence: {confidence:.1f}%"
        else:
            log_entry = f"SELL | {symbol} | Price: ${price:.6f} | Qty: {quantity:.6f} | PnL: {pnl_pct:+.2f}% | Type: {exit_type}"
        
        print(f"[{timestamp}] {log_entry}")
        
        with open('trading_log.txt', 'a') as f:
            f.write(f"[{timestamp}] {log_entry}\n")
            
    except Exception as e:
        print(f"Logging error: {e}")

def send_status():
    """Send status update"""
    global active_position, current_investment
    
    win_rate = (performance_state['total_wins'] / performance_state['total_trades'] * 100) if performance_state['total_trades'] > 0 else 0
    
    status_msg = (f"🤖 <b>BOT STATUS</b>\n"
                  f"Status: {'🟢 RUNNING' if BOT_RUNNING else '🔴 STOPPED'}\n"
                  f"Capital: ${current_investment:.2f}\n"
                  f"Trades: {performance_state['total_trades']}\n"
                  f"Win Rate: {win_rate:.1f}%")
    
    if active_position:
        symbol = active_position['symbol']
        current_price = price_data.get(symbol, active_position['entry_price'])
        pnl_pct = ((current_price - active_position['entry_price']) / active_position['entry_price']) * 100
        
        status_msg += (f"\n\n📊 <b>ACTIVE POSITION</b>\n"
                      f"Symbol: {symbol}\n"
                      f"Entry: ${active_position['entry_price']:.6f}\n"
                      f"Current: ${current_price:.6f}\n"
                      f"PnL: {pnl_pct:+.2f}%")
    
    send_telegram_message(status_msg)

def initialize_binance():
    """Initialize Binance client"""
    global client
    try:
        client = Client(API_KEY, API_SECRET, {"timeout": 20})
        client.ping()
        print("✅ Binance client initialized")
        return True
    except Exception as e:
        print(f"❌ Binance initialization failed: {e}")
        return False

# ==================== MAIN BOT LOGIC ====================
def main():
    global BOT_RUNNING
    
    print("🚀 Starting Trading Bot with WebSocket")
    
    if not initialize_binance():
        return
    
    # Initialize WebSocket
    ws_manager = BinanceWebSocket()
    ws_manager.start_kline_stream(COINS)
    
    # Wait for WebSocket to connect and collect initial data
    print("⏳ Collecting initial market data...")
    time.sleep(30)
    
    startup_msg = (f"🤖 <b>BOT STARTED</b>\n"
                  f"Mode: {'LIVE' if ORDER_RUN else 'SIMULATION'}\n"
                  f"Coins: {len(COINS)}\n"
                  f"Capital: ${current_investment:.2f}")
    send_telegram_message(startup_msg)
    
    last_scan_time = 0
    scan_interval = 10  # seconds
    
    while True:
        try:
            handle_telegram_command()
            
            if not BOT_RUNNING:
                time.sleep(1)
                continue
            
            current_time = time.time()
            
            # Monitor active position
            if active_position:
                monitor_position()
                time.sleep(1)
                continue
            
            # Scan for new signals
            if current_time - last_scan_time >= scan_interval:
                print("🔍 Scanning for signals...")
                
                best_signal = None
                
                for symbol in COINS:
                    if not BOT_RUNNING:
                        break
                    
                    analysis = analyze_symbol(symbol)
                    
                    if analysis and analysis['buy_signal']:
                        if not best_signal or analysis['confidence'] > best_signal['confidence']:
                            best_signal = analysis
                
                if best_signal and best_signal['confidence'] > 60:
                    print(f"🎯 Signal found: {best_signal['symbol']} (Confidence: {best_signal['confidence']:.1f}%)")
                    execute_buy(best_signal['symbol'], best_signal)
                
                last_scan_time = current_time
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            BOT_RUNNING = False
            break
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
