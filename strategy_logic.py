# Strategy Logic for Cryptocurrency Trading
# Hipotesis: Optimized volatility-based entry with volume filters and dynamic SL/TP
# Confidence Score: 75% (moderate improvement in volatility handling)

import numpy as np
import pandas as pd

# Constants
VOLATILITY_THRESHOLD = 0.12
ENTRY_VOLUME_FILTER = 1.5
TRAILING_STOP = 0.05
MAX_DRAWDOWN = 0.08

# Indicators
def calculate_volatility(price_data):
    """Calculate volatility using standard deviation of price changes."""
    return np.std(price_data.diff())

def get_price_range(price_data):
    """Determine price range based on historical data."""
    return price_data.iloc[-1] - price_data.iloc[-2] if len(price_data) > 1 else 0

# SMC (Stop and Limit) function
def smc(price, volatility, entry_volume):
    """Determine stop and limit based on volatility and entry volume."""
    if volatility > VOLATILITY_THRESHOLD:
        return price + ENTRY_VOLUME_FILTER * entry_volume
    return price + 0.01 * entry_volume

# Scoring function
def scoring(price, volatility, entry_volume):
    """Calculate score based on volatility and entry volume."""
    vol_score = 1 - (volatility / VOLATILITY_THRESHOLD)
    vol_score = max(vol_score, 0.1)
    entry_score = 1 - (entry_volume / 1000)
    entry_score = max(entry_score, 0.1)
    return vol_score * entry_score

# Entry function
def entry(price, volatility, entry_volume):
    """Determine entry point based on volatility and volume."""
    if volatility > VOLATILITY_THRESHOLD:
        return price + ENTRY_VOLUME_FILTER * entry_volume
    return price + 0.01 * entry_volume

# Stop Loss and Take Profit
def sl_tp(price, volatility, entry_volume):
    """Calculate SL/TP based on volatility and entry volume."""
    if volatility > VOLATILITY_THRESHOLD:
        return price - 0.02 * entry_volume
    return price - 0.01 * entry_volume

# Strategy Logic
def strategy_logic(price_data, entry_volume):
    """Main strategy logic with volatility-based adjustments."""
    price = price_data.iloc[-1]
    volatility = calculate_volatility(price_data)
    entry_price = entry(price, volatility, entry_volume)
    sl_price = sl_tp(price, volatility, entry_volume)
    tp_price = entry_price + 0.05 * entry_volume
    return {
        'entry': entry_price,
        'stop': sl_price,
        'take_profit': tp_price
    }

# Scoring function
def calculate_score(price, volatility, entry_volume):
    """Calculate overall score based on volatility and entry volume."""
    vol_score = 1 - (volatility / VOLATILITY_THRESHOLD)
    entry_score = 1 - (entry_volume / 1000)
    return vol_score * entry_score

# Indicator function
def indicator(price_data):
    """Determine indicator based on price range and volatility."""
    price_range = get_price_range(price_data)
    volatility = calculate_volatility(price_data)
    return price_range - volatility * 0.05