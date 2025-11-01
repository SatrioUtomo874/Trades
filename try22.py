import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from binance import AsyncClient, BinanceSocketManager
from telegram import Bot
from telegram.error import TelegramError
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
import pandas_ta as ta  # Ganti dari ta ke pandas_ta
import signal
import sys

# Load environment variables
load_dotenv()

# Configuration
class Config:
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    UPDATE_INTERVAL_HOURS = int(os.getenv('UPDATE_INTERVAL_HOURS', '4'))
    
    # Get all USDT pairs excluding BTC and ETH
    PAIR_LIST = []  # Will be populated dynamically
    
    # Indicator parameters
    EMA_FAST = 50
    EMA_SLOW = 200
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14
    
    # Confidence weights
    WEIGHT_TREND = 0.4
    WEIGHT_RSI = 0.2
    WEIGHT_MACD = 0.2
    WEIGHT_VOLUME = 0.2

class SmartMoneyAnalyzer:
    """Class untuk analisis Smart Money Concept"""
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, lookback: int = 100) -> Tuple[float, float, List[float], List[float]]:
        """Temukan level support dan resistance menggunakan swing points"""
        highs = df['high'].tail(lookback)
        lows = df['low'].tail(lookback)
        
        # Find swing highs and lows
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(highs)-2):
            if (highs.iloc[i] > highs.iloc[i-1] and 
                highs.iloc[i] > highs.iloc[i-2] and
                highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.iloc[i+2]):
                resistance_levels.append(highs.iloc[i])
            
            if (lows.iloc[i] < lows.iloc[i-1] and 
                lows.iloc[i] < lows.iloc[i-2] and
                lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.iloc[i+2]):
                support_levels.append(lows.iloc[i])
        
        # Get the most relevant levels (closest to current price)
        current_price = df['close'].iloc[-1]
        
        # Filter support levels below current price and resistance above
        valid_support = [s for s in support_levels if s < current_price]
        valid_resistance = [r for r in resistance_levels if r > current_price]
        
        if valid_support:
            relevant_support = max(valid_support)  # Highest support below current price
        else:
            # Fallback: use recent low as support
            relevant_support = lows.tail(20).min()
            
        if valid_resistance:
            relevant_resistance = min(valid_resistance)  # Lowest resistance above current price
        else:
            # Fallback: use recent high as resistance
            relevant_resistance = highs.tail(20).max()
        
        return relevant_support, relevant_resistance, support_levels, resistance_levels
    
    @staticmethod
    def calculate_fair_value_gap(df: pd.DataFrame) -> Optional[float]:
        """Hitung Fair Value Gap untuk entry optimal"""
        if len(df) < 3:
            return None
            
        current = df.iloc[-1]
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Bullish FVG detection (price gap down)
        if (prev1['high'] < prev2['low'] and 
            current['low'] > prev1['high']):
            return (prev1['high'] + current['low']) / 2
        
        return None
    
    @staticmethod
    def find_liquidity_zones(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Temukan zona liquidity (equal highs/lows)"""
        highs = df['high'].tail(50)
        lows = df['low'].tail(50)
        
        # Find equal highs (liquidity above)
        equal_highs = []
        for i in range(len(highs)):
            for j in range(i+1, len(highs)):
                if abs(highs.iloc[i] - highs.iloc[j]) / highs.iloc[i] < 0.002:  # 0.2% tolerance
                    equal_highs.append((highs.iloc[i] + highs.iloc[j]) / 2)
        
        # Find equal lows (liquidity below)
        equal_lows = []
        for i in range(len(lows)):
            for j in range(i+1, len(lows)):
                if abs(lows.iloc[i] - lows.iloc[j]) / lows.iloc[i] < 0.002:  # 0.2% tolerance
                    equal_lows.append((lows.iloc[i] + lows.iloc[j]) / 2)
        
        return equal_lows, equal_highs

    @staticmethod
    def calculate_price_targets(df: pd.DataFrame, entry: float, support: float, resistance: float, atr: float) -> Tuple[float, float, float]:
        """Hitung target harga berdasarkan struktur market dan volatilitas"""
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        # Method 1: Berdasarkan resistance levels
        resistance_target = resistance
        
        # Method 2: Berdasarkan ATR (volatilitas)
        atr_target_1 = entry + (atr * 1.5)
        atr_target_2 = entry + (atr * 2.5)
        
        # Method 3: Berdasarkan recent swing high
        swing_high_target = recent_high
        
        # Method 4: Fibonacci extension (simplified)
        fib_target = entry + ((resistance - support) * 0.618)
        
        # Kombinasi semua method untuk mendapatkan target yang realistis
        tp1 = min(resistance_target, atr_target_1, swing_high_target, fib_target)
        tp2 = max(resistance_target, atr_target_2, swing_high_target, fib_target)
        
        # Pastikan TP reasonable
        min_profit = entry * 1.01  # Minimal 1% profit
        max_profit = entry * 1.10  # Maksimal 10% profit untuk TP1
        
        tp1 = max(min(tp1, max_profit), min_profit)
        tp2 = max(tp2, tp1 * 1.02)  # TP2 minimal 2% di atas TP1
        
        return tp1, tp2

class TechnicalAnalyzer:
    """Class untuk menghitung indikator teknikal dengan pandas_ta"""
    
    def __init__(self):
        self.smart_money = SmartMoneyAnalyzer()
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
        return ta.ema(data['close'], length=period)
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int) -> pd.Series:
        return ta.rsi(data['close'], length=period)
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        macd_result = ta.macd(data['close'], fast=12, slow=26, signal=9)
        if macd_result is not None:
            return macd_result.iloc[:, 0], macd_result.iloc[:, 1], macd_result.iloc[:, 2]
        return None, None, None
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int) -> pd.Series:
        return ta.atr(data['high'], data['low'], data['close'], length=period)
    
    @staticmethod
    def calculate_volume_sma(data: pd.DataFrame, period: int) -> pd.Series:
        return data['volume'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.Series:
        return ta.obv(data['close'], data['volume'])

class SignalAnalyzer:
    """Class untuk menganalisis sinyal trading dengan Smart Money Concept"""
    
    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()
    
    async def analyze_pair(self, client: AsyncClient, pair: str) -> Optional[Dict]:
        """Analisis satu pair untuk sinyal BUY dengan Smart Money Concept"""
        try:
            # Skip excluded pairs
            excluded_pairs = ['FDUSDT', 'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'USDPUSDT', 'USTUSDT']
            if pair in excluded_pairs:
                return None
                
            # Ambil data multi-timeframe
            timeframes = ['1d', '4h', '1h']
            all_data = {}
            
            for tf in timeframes:
                klines = await client.get_klines(
                    symbol=pair,
                    interval=tf,
                    limit=300
                )
                
                # Check if we got enough data
                if len(klines) < 100:
                    logging.warning(f"Insufficient data for {pair} on {tf} timeframe")
                    return None
                
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove any NaN values
                df = df.dropna()
                
                if len(df) < 100:
                    logging.warning(f"Not enough valid data for {pair} on {tf}")
                    return None
                
                all_data[tf] = df
            
            return self._generate_smart_money_signal(pair, all_data)
            
        except Exception as e:
            logging.error(f"Error analyzing {pair}: {str(e)}")
            return None
    
    def _generate_smart_money_signal(self, pair: str, data: Dict) -> Dict:
        """Generate sinyal dengan Smart Money Concept"""
        df_1d = data['1d']
        df_4h = data['4h']
        df_1h = data['1h']
        
        # Hitung indikator untuk timeframe 4h (utama)
        try:
            ema_fast_4h = self.tech_analyzer.calculate_ema(df_4h, Config.EMA_FAST)
            ema_slow_4h = self.tech_analyzer.calculate_ema(df_4h, Config.EMA_SLOW)
            rsi_4h = self.tech_analyzer.calculate_rsi(df_4h, Config.RSI_PERIOD)
            macd_4h, macd_signal_4h, macd_hist_4h = self.tech_analyzer.calculate_macd(df_4h)
            atr_4h = self.tech_analyzer.calculate_atr(df_4h, Config.ATR_PERIOD)
            volume_sma_4h = self.tech_analyzer.calculate_volume_sma(df_4h, 20)
            obv_4h = self.tech_analyzer.calculate_obv(df_4h)
        except Exception as e:
            logging.error(f"Error calculating indicators for {pair}: {str(e)}")
            return None
        
        # Check for NaN values in indicators
        if (ema_fast_4h is None or ema_slow_4h is None or rsi_4h is None or 
            macd_4h is None or atr_4h is None or 
            ema_fast_4h.isna().iloc[-1] or ema_slow_4h.isna().iloc[-1] or 
            rsi_4h.isna().iloc[-1] or atr_4h.isna().iloc[-1]):
            logging.warning(f"NaN values in indicators for {pair}")
            return None
        
        # Data terbaru
        current_close = df_4h['close'].iloc[-1]
        current_rsi = rsi_4h.iloc[-1]
        current_atr = atr_4h.iloc[-1]
        current_volume = df_4h['volume'].iloc[-1]
        avg_volume = volume_sma_4h.iloc[-1] if not volume_sma_4h.isna().iloc[-1] else current_volume
        current_obv = obv_4h.iloc[-1] if obv_4h is not None else 0
        obv_trend = current_obv > obv_4h.iloc[-5] if obv_4h is not None and len(obv_4h) > 5 else False
        
        # Smart Money Analysis
        support, resistance, all_supports, all_resistances = self.tech_analyzer.smart_money.find_support_resistance(df_4h)
        fvg_entry = self.tech_analyzer.smart_money.calculate_fair_value_gap(df_4h)
        liquidity_below, liquidity_above = self.tech_analyzer.smart_money.find_liquidity_zones(df_4h)
        
        # Trend analysis
        trend_direction = self._get_trend_direction(ema_fast_4h, ema_slow_4h)
        macd_signal = self._get_macd_signal(macd_4h, macd_signal_4h)
        rsi_signal = self._get_rsi_signal(current_rsi)
        volume_signal = current_volume > avg_volume * 1.2
        
        # Smart Money Conditions
        smart_money_conditions = self._check_smart_money_conditions(
            df_4h, current_close, support, resistance, obv_trend
        )
        
        # Hitung confidence score dengan tambahan smart money factors
        confidence = self._calculate_smart_money_confidence(
            trend_direction, rsi_signal, macd_signal, volume_signal, smart_money_conditions
        )
        
        # Hanya proses jika confidence cukup tinggi
        if confidence < 0.7:  # Increased threshold for better signals
            return None
        
        # Kalkulasi level trading dengan Smart Money Concept
        entry, sl, tp1, tp2, rr_ratio = self._calculate_dynamic_levels(
            df_4h, current_close, current_atr, trend_direction, 
            support, resistance, all_supports, all_resistances
        )
        
        # Validasi: SL harus selalu di bawah entry
        if sl >= entry:
            logging.warning(f"Invalid SL for {pair}: SL {sl} >= Entry {entry}. Adjusting...")
            # Adjust SL berdasarkan ATR
            sl = entry - (current_atr * 1.5)
            if sl >= entry:
                sl = entry * 0.98  # Fallback 2% di bawah entry
        
        # Validasi ulang TP
        if tp1 <= entry:
            tp1 = entry * 1.02
        if tp2 <= tp1:
            tp2 = tp1 * 1.02
        
        return {
            'pair': pair,
            'trend': trend_direction,
            'rsi': round(current_rsi, 2),
            'macd_signal': macd_signal,
            'entry': round(entry, 4),
            'stop_loss': round(sl, 4),
            'take_profit_1': round(tp1, 4),
            'take_profit_2': round(tp2, 4),
            'confidence': round(confidence * 100),
            'volume_boost': volume_signal,
            'support_level': round(support, 4),
            'resistance_level': round(resistance, 4),
            'obv_bullish': obv_trend,
            'timestamp': datetime.now(timezone.utc),
            'risk_reward': f'1:{rr_ratio:.1f}',
            'atr_percentage': round((current_atr / current_close) * 100, 2)
        }
    
    def _calculate_dynamic_levels(self, df: pd.DataFrame, current_close: float, 
                                atr: float, trend: str, support: float, 
                                resistance: float, all_supports: List[float], 
                                all_resistances: List[float]) -> Tuple[float, float, float, float, float]:
        """Hitung level trading dinamis berdasarkan struktur market"""
        
        # Hitung entry price
        if trend == "Bullish":
            # Untuk trend bullish, entry di pullback ke support atau FVG
            entry = max(support * 1.005, current_close * 0.995)
        else:
            # Untuk sideways/bearish, entry lebih konservatif
            entry = current_close * 0.99
        
        # Hitung Stop Loss berdasarkan multiple factors
        sl = self._calculate_dynamic_sl(entry, support, all_supports, atr, current_close)
        
        # Hitung Take Profit berdasarkan multiple factors
        tp1, tp2 = self._calculate_dynamic_tp(entry, resistance, all_resistances, atr, current_close, sl)
        
        # Hitung risk-reward ratio
        risk = entry - sl
        if risk > 0:
            reward = tp1 - entry
            rr_ratio = round(reward / risk, 1)
        else:
            rr_ratio = 2.0  # Default
        
        return entry, sl, tp1, tp2, rr_ratio
    
    def _calculate_dynamic_sl(self, entry: float, support: float, all_supports: List[float], 
                            atr: float, current_close: float) -> float:
        """Hitung Stop Loss dinamis berdasarkan multiple factors"""
        
        # Factor 1: Di bawah support terdekat
        sl_support = support * 0.995
        
        # Factor 2: Berdasarkan ATR (volatilitas)
        sl_atr = entry - (atr * 1.5)
        
        # Factor 3: Berdasarkan support level berikutnya (jika ada)
        if all_supports:
            # Cari support di bawah support saat ini
            lower_supports = [s for s in all_supports if s < support]
            if lower_supports:
                sl_next_support = max(lower_supports) * 0.995
            else:
                sl_next_support = support * 0.99
        else:
            sl_next_support = support * 0.99
        
        # Factor 4: Maximum risk (5% dari entry)
        sl_max_risk = entry * 0.95
        
        # Factor 5: Minimum risk (1% dari entry)
        sl_min_risk = entry * 0.99
        
        # Kombinasikan semua factors, pilih yang terbaik (paling aman)
        sl_candidates = [sl_support, sl_atr, sl_next_support, sl_max_risk]
        valid_sl_candidates = [sl for sl in sl_candidates if sl < entry]
        
        if valid_sl_candidates:
            sl = max(valid_sl_candidates)  # Pilih SL tertinggi yang masih di bawah entry
        else:
            sl = sl_min_risk
        
        # Pastikan SL reasonable
        if sl > sl_min_risk:
            sl = sl_min_risk
        if sl < sl_max_risk:
            sl = sl_max_risk
        
        return sl
    
    def _calculate_dynamic_tp(self, entry: float, resistance: float, all_resistances: List[float],
                            atr: float, current_close: float, sl: float) -> Tuple[float, float]:
        """Hitung Take Profit dinamis berdasarkan multiple factors"""
        
        # Factor 1: Resistance terdekat
        tp_resistance = resistance * 0.995  # Slight buffer
        
        # Factor 2: Berdasarkan ATR (volatilitas)
        tp_atr_1 = entry + (atr * 2.0)
        tp_atr_2 = entry + (atr * 3.0)
        
        # Factor 3: Berdasarkan risk-reward ratio
        risk = entry - sl
        tp_rr_1 = entry + (risk * 2.0)
        tp_rr_2 = entry + (risk * 3.0)
        
        # Factor 4: Berdasarkan resistance level berikutnya
        if all_resistances:
            # Cari resistance di atas resistance saat ini
            higher_resistances = [r for r in all_resistances if r > resistance]
            if higher_resistances:
                tp_next_resistance = min(higher_resistances) * 0.995
            else:
                tp_next_resistance = resistance * 1.02
        else:
            tp_next_resistance = resistance * 1.02
        
        # Factor 5: Berdasarkan persentase profit wajar
        tp_percent_1 = entry * 1.03  # 3% profit
        tp_percent_2 = entry * 1.06  # 6% profit
        
        # Kombinasikan semua factors untuk TP1
        tp1_candidates = [tp_resistance, tp_atr_1, tp_rr_1, tp_percent_1, tp_next_resistance]
        valid_tp1_candidates = [tp for tp in tp1_candidates if tp > entry]
        
        if valid_tp1_candidates:
            tp1 = min(valid_tp1_candidates)  # Pilih TP1 terendah yang realistis
        else:
            tp1 = tp_percent_1
        
        # Kombinasikan semua factors untuk TP2
        tp2_candidates = [tp_atr_2, tp_rr_2, tp_percent_2, tp_next_resistance * 1.02]
        valid_tp2_candidates = [tp for tp in tp2_candidates if tp > tp1]
        
        if valid_tp2_candidates:
            tp2 = min(valid_tp2_candidates)
        else:
            tp2 = tp1 * 1.03  # 3% di atas TP1
        
        # Pastikan TP reasonable
        max_tp = current_close * 1.15  # Maksimal 15% profit
        tp1 = min(tp1, max_tp)
        tp2 = min(tp2, max_tp * 1.05)
        
        return tp1, tp2
    
    def _check_smart_money_conditions(self, df: pd.DataFrame, current_price: float, 
                                    support: float, resistance: float, obv_trend: bool) -> Dict:
        """Cek kondisi Smart Money"""
        conditions = {
            'near_support': current_price <= support * 1.02,  # Dalam 2% dari support
            'obv_bullish': obv_trend,
            'volume_spike': df['volume'].iloc[-1] > df['volume'].iloc[-5:].mean() * 1.5,
            'price_above_ema': current_price > df['close'].rolling(50).mean().iloc[-1]
        }
        
        # Hitung score kondisi
        score = sum(conditions.values())
        conditions['score'] = score
        
        return conditions
    
    def _get_trend_direction(self, ema_fast: pd.Series, ema_slow: pd.Series) -> str:
        """Tentukan arah trend"""
        if ema_fast is None or ema_slow is None or len(ema_fast) < 2 or len(ema_slow) < 2:
            return "Unknown"
            
        fast_current = ema_fast.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_current = ema_slow.iloc[-1]
        
        if fast_current > slow_current and fast_prev <= slow_current:
            return "Bullish Cross"
        elif fast_current > slow_current:
            return "Bullish"
        elif fast_current < slow_current and fast_prev >= slow_current:
            return "Bearish Cross"
        else:
            return "Bearish"
    
    def _get_macd_signal(self, macd: pd.Series, macd_signal: pd.Series) -> str:
        """Analisis sinyal MACD"""
        if macd is None or macd_signal is None or len(macd) < 2 or len(macd_signal) < 2:
            return "Unknown"
            
        macd_current = macd.iloc[-1]
        macd_prev = macd.iloc[-2]
        signal_current = macd_signal.iloc[-1]
        
        if macd_current > signal_current and macd_prev <= signal_current:
            return "Cross UP"
        elif macd_current < signal_current and macd_prev >= signal_current:
            return "Cross DOWN"
        elif macd_current > signal_current:
            return "Above Signal"
        else:
            return "Below Signal"
    
    def _get_rsi_signal(self, rsi: float) -> bool:
        """Cek kondisi RSI untuk entry"""
        return 40 <= rsi <= 60  # RSI di area netral untuk accumulation
    
    def _calculate_smart_money_confidence(self, trend: str, rsi_signal: bool, 
                                        macd_signal: str, volume_signal: bool,
                                        smart_conditions: Dict) -> float:
        """Hitung confidence score dengan Smart Money factors"""
        confidence = 0.0
        
        # Trend weight
        if "Bullish" in trend:
            confidence += Config.WEIGHT_TREND
        
        # RSI weight
        if rsi_signal:
            confidence += Config.WEIGHT_RSI
        
        # MACD weight
        if "Cross UP" in macd_signal or "Above Signal" in macd_signal:
            confidence += Config.WEIGHT_MACD
        
        # Volume weight
        if volume_signal:
            confidence += Config.WEIGHT_VOLUME
        
        # Smart Money conditions (additional points)
        smart_money_score = smart_conditions['score'] * 0.1  # Convert to 0-0.4 scale
        confidence += min(smart_money_score, 0.4)
        
        return min(confidence, 1.0)  # Cap at 100%

# ... (TelegramNotifier, BinanceDataManager, AISignalBot classes tetap sama seperti sebelumnya)
# Hanya mengganti import dan TechnicalAnalyzer saja

class TelegramNotifier:
    """Class untuk mengirim notifikasi ke Telegram"""
    
    def __init__(self):
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.chat_id = Config.TELEGRAM_CHAT_ID
    
    async def send_signal(self, signal: Dict):
        """Kirim sinyal ke Telegram"""
        try:
            message = self._format_smart_money_signal(signal)
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            logging.info(f"Smart Money Signal sent to Telegram for {signal['pair']}")
        except TelegramError as e:
            logging.error(f"Failed to send Telegram message: {str(e)}")
    
    def _format_smart_money_signal(self, signal: Dict) -> str:
        """Format pesan sinyal Smart Money untuk Telegram"""
        volume_icon = "📈" if signal['volume_boost'] else "📊"
        obv_icon = "🟢" if signal['obv_bullish'] else "🔴"
        
        # Calculate risk percentage
        risk_pct = ((signal['entry'] - signal['stop_loss']) / signal['entry']) * 100
        profit_pct_1 = ((signal['take_profit_1'] - signal['entry']) / signal['entry']) * 100
        profit_pct_2 = ((signal['take_profit_2'] - signal['entry']) / signal['entry']) * 100
        
        return f"""
🎯 <b>DYNAMIC SMART MONEY SIGNAL</b> 🎯

🪙 <b>Coin:</b> {signal['pair']}
📊 <b>Trend:</b> {signal['trend']}
{volume_icon} <b>RSI:</b> {signal['rsi']}
⚡ <b>MACD:</b> {signal['macd_signal']}
{obv_icon} <b>OBV:</b> {'Bullish' if signal['obv_bullish'] else 'Bearish'}
📈 <b>ATR Volatility:</b> {signal['atr_percentage']}%

💎 <b>KEY LEVELS:</b>
🏠 <b>Support:</b> {signal['support_level']}
🚧 <b>Resistance:</b> {signal['resistance_level']}

🎯 <b>TRADING PLAN:</b>
💰 <b>Entry:</b> {signal['entry']}
🛑 <b>SL:</b> {signal['stop_loss']} ({risk_pct:.2f}%)
🎯 <b>TP1:</b> {signal['take_profit_1']} (+{profit_pct_1:.2f}%)
🎯 <b>TP2:</b> {signal['take_profit_2']} (+{profit_pct_2:.2f}%)

💪 <b>Confidence:</b> {signal['confidence']}%
⚖️ <b>Risk/Reward:</b> {signal['risk_reward']}
📅 <b>Time:</b> {signal['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}

<i>Dynamic Smart Money Concept - Adaptif terhadap market conditions</i>
"""
    
    async def send_alert(self, message: str):
        """Kirim alert umum ke Telegram"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"⚠️ <b>DYNAMIC SMART MONEY BOT ALERT</b> ⚠️\n\n{message}",
                parse_mode='HTML'
            )
        except TelegramError as e:
            logging.error(f"Failed to send alert: {str(e)}")
    
    async def send_summary(self, signals: List[Dict]):
        """Kirim summary 3 sinyal terbaik"""
        if not signals:
            await self.send_alert("📊 Analysis Complete: No high-quality signals found.")
            return
            
        try:
            summary_message = "🚀 <b>TOP 3 DYNAMIC SMART MONEY SIGNALS</b> 🚀\n\n"
            
            for i, signal in enumerate(signals[:3], 1):
                risk_pct = ((signal['entry'] - signal['stop_loss']) / signal['entry']) * 100
                profit_pct_1 = ((signal['take_profit_1'] - signal['entry']) / signal['entry']) * 100
                
                summary_message += f"{i}. <b>{signal['pair']}</b> - Confidence: {signal['confidence']}%\n"
                summary_message += f"   📍 Entry: {signal['entry']} | SL: {signal['stop_loss']} ({risk_pct:.2f}%)\n"
                summary_message += f"   🎯 TP1: {signal['take_profit_1']} (+{profit_pct_1:.2f}%)\n"
                summary_message += f"   📊 RSI: {signal['rsi']} | Trend: {signal['trend']}\n"
                summary_message += f"   ⚖️ R/R: {signal['risk_reward']} | ATR: {signal['atr_percentage']}%\n\n"
            
            summary_message += f"📈 Total pairs scanned: {len(Config.PAIR_LIST)}\n"
            summary_message += f"✅ Quality signals found: {len(signals)}\n"
            summary_message += f"⏰ Next update in {Config.UPDATE_INTERVAL_HOURS} hours"
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=summary_message,
                parse_mode='HTML'
            )
            
        except TelegramError as e:
            logging.error(f"Failed to send summary: {str(e)}")

class BinanceDataManager:
    """Manager untuk koneksi dan data Binance"""
    
    def __init__(self):
        self.client = None
        self.bm = None
    
    async def initialize(self):
        """Initialize koneksi Binance"""
        try:
            self.client = await AsyncClient.create(
                Config.BINANCE_API_KEY,
                Config.BINANCE_API_SECRET
            )
            self.bm = BinanceSocketManager(self.client)
            
            # Get all USDT pairs excluding BTC and ETH
            await self._load_all_usdt_pairs()
            
            logging.info("Binance client initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Binance client: {str(e)}")
            raise
    
    async def _load_all_usdt_pairs(self):
        """Load semua pair USDT kecuali BTC, ETH, dan pair yang dikecualikan"""
        try:
            exchange_info = await self.client.get_exchange_info()
            usdt_pairs = []
            
            # Pair yang dikecualikan
            excluded_pairs = ['BTCUSDT', 'ETHUSDT', 'FDUSDT', 'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'USDPUSDT', 'USTUSDT']
            
            for symbol in exchange_info['symbols']:
                if (symbol['symbol'].endswith('USDT') and 
                    symbol['status'] == 'TRADING' and
                    symbol['symbol'] not in excluded_pairs):
                    usdt_pairs.append(symbol['symbol'])
            
            # Sort by volume and take top 50 for efficiency
            tickers = await self.client.get_ticker()
            volume_data = []
            
            for ticker in tickers:
                if ticker['symbol'] in usdt_pairs:
                    volume = float(ticker['quoteVolume'])
                    volume_data.append((ticker['symbol'], volume))
            
            # Sort by volume descending
            volume_data.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 50 pairs by volume
            Config.PAIR_LIST = [pair for pair, volume in volume_data[:50]]
            
            logging.info(f"Loaded {len(Config.PAIR_LIST)} trading pairs (excluding BTC/ETH/Stablecoins)")
            
        except Exception as e:
            logging.error(f"Error loading USDT pairs: {str(e)}")
            # Fallback to some popular pairs (excluding the ones we don't want)
            Config.PAIR_LIST = [
                'SOLUSDT', 'BNBUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOTUSDT',
                'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 'XRPUSDT', 'DOGEUSDT',
                'ATOMUSDT', 'FTMUSDT', 'NEARUSDT', 'CAKEUSDT', 'SUIUSDT',
                'LTCUSDT', 'XLMUSDT', 'ALGOUSDT', 'VETUSDT', 'THETAUSDT'
            ]
    
    async def close(self):
        """Tutup koneksi Binance"""
        if self.client:
            await self.client.close_connection()

class AISignalBot:
    """Main class untuk AI Signal Bot dengan Smart Money"""
    
    def __init__(self):
        self.data_manager = BinanceDataManager()
        self.analyzer = SignalAnalyzer()
        self.notifier = TelegramNotifier()
        self.is_running = False
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/dynamic_smart_money_signals.txt', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    async def analyze_all_pairs(self):
        """Analisis semua pair yang ditentukan"""
        signals = []
        analyzed_count = 0
        
        logging.info(f"Starting Dynamic Smart Money analysis for {len(Config.PAIR_LIST)} pairs...")
        
        for pair in Config.PAIR_LIST:
            try:
                signal = await self.analyzer.analyze_pair(self.data_manager.client, pair)
                analyzed_count += 1
                
                if signal:
                    # Validasi tambahan: pastikan SL < Entry dan TP > Entry
                    if (signal['stop_loss'] < signal['entry'] and 
                        signal['take_profit_1'] > signal['entry'] and
                        signal['take_profit_2'] > signal['take_profit_1']):
                        signals.append(signal)
                        logging.info(f"🎯 Dynamic Signal for {pair}: Confidence {signal['confidence']}%, R/R {signal['risk_reward']}")
                    else:
                        logging.warning(f"❌ Rejected {pair}: Invalid levels")
                else:
                    logging.debug(f"❌ No quality signal for {pair}")
                    
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
                    
            except Exception as e:
                logging.error(f"Error analyzing {pair}: {str(e)}")
                continue
        
        # Sort by confidence dan ambil top 3
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        top_signals = signals[:3]
        
        logging.info(f"Dynamic analysis complete: {analyzed_count} pairs analyzed, {len(signals)} quality signals found")
        
        return top_signals
    
    async def run_analysis_cycle(self):
        """Jalankan satu siklus analisis"""
        logging.info("Starting Dynamic Smart Money analysis cycle...")
        
        try:
            signals = await self.analyze_all_pairs()
            
            if signals:
                # Kirim summary dulu
                await self.notifier.send_summary(signals)
                await asyncio.sleep(2)
                
                # Kirim sinyal individual untuk 3 terbaik
                for signal in signals:
                    if signal['confidence'] >= 70:  # Higher threshold for quality
                        await self.notifier.send_signal(signal)
                        await asyncio.sleep(2)  # Delay antar pesan
            else:
                logging.info("No quality signals generated in this cycle")
                await self.notifier.send_alert("📊 Analysis Complete: No high-quality signals found in this cycle.")
                
        except Exception as e:
            logging.error(f"Error in analysis cycle: {str(e)}")
            await self.notifier.send_alert(f"Error in analysis cycle: {str(e)}")
    
    async def start(self):
        """Start the Dynamic Smart Money Bot"""
        logging.info("Starting Dynamic Smart Money Signal Bot...")
        
        try:
            # Initialize connections
            await self.data_manager.initialize()
            await self.notifier.send_alert("🤖 DYNAMIC SMART MONEY BOT Started Successfully!\n\n"
                                         "🔍 Scanning all USDT pairs (excluding BTC/ETH/Stablecoins)\n"
                                         "🎯 Using Dynamic Smart Money Concept + Adaptive Risk Management\n"
                                         "📊 Multiple factors: Support/Resistance, ATR, Market Structure\n"
                                         "⏰ Updates every 4 hours")
            
            self.is_running = True
            
            # Main loop
            while self.is_running:
                try:
                    await self.run_analysis_cycle()
                    
                    # Tunggu untuk interval berikutnya
                    logging.info(f"Waiting {Config.UPDATE_INTERVAL_HOURS} hours for next analysis...")
                    await asyncio.sleep(Config.UPDATE_INTERVAL_HOURS * 3600)
                    
                except Exception as e:
                    logging.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logging.error(f"Failed to start bot: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the bot"""
        self.is_running = False
        await self.data_manager.close()
        logging.info("Dynamic Smart Money Bot stopped")

# Web service untuk Render
from flask import Flask, jsonify
import threading

app = Flask(__name__)
bot_instance = None
bot_thread = None

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Dynamic Smart Money Bot",
        "description": "AI-powered cryptocurrency trading signals with Smart Money Concept"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/start')
def start_bot():
    global bot_instance, bot_thread
    if bot_thread and bot_thread.is_alive():
        return jsonify({"status": "already running"})
    
    bot_instance = AISignalBot()
    bot_thread = threading.Thread(target=lambda: asyncio.run(bot_instance.start()))
    bot_thread.daemon = True
    bot_thread.start()
    
    return jsonify({"status": "bot started"})

@app.route('/stop')
def stop_bot():
    global bot_instance, bot_thread
    if bot_instance:
        asyncio.run(bot_instance.stop())
    if bot_thread:
        bot_thread.join(timeout=10)
    
    return jsonify({"status": "bot stopped"})

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("Received shutdown signal...")
    if bot_instance:
        asyncio.run(bot_instance.stop())
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check environment variables
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        exit(1)
    
    # Start the web server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
