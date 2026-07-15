#!/usr/bin/env python3
"""
ENTRY POINT RENDER
Memuat core modules, mencoba load strategy_logic (kaset AI),
lalu menjalankan bot.
"""
import os
import sys
import time
import threading
import logging

# Tambahkan path ke root agar import core/ dan strategy_logic bisa ditemukan
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# Import core modules
from core import api_client
from core import stats_keeper
from core import monitor_engine
from core import telegram_bot
from core import context_builder

# ==================== LOAD STRATEGY LOGIC (KASET AI) ====================
strategy_logic = None
try:
    import strategy_logic as logic
    # Inject ke monitor_engine agar bisa dipakai di scanning
    monitor_engine.strategy_logic = logic
    # Timpa parameter default (jika ada)
    if hasattr(logic, 'MIN_CONFIDENCE'):
        monitor_engine.DEFAULTS['MIN_CONFIDENCE'] = logic.MIN_CONFIDENCE
    if hasattr(logic, 'MIN_RR'):
        monitor_engine.DEFAULTS['MIN_RR'] = logic.MIN_RR
    if hasattr(logic, 'MAX_POSITIONS'):
        monitor_engine.DEFAULTS['MAX_POSITIONS'] = logic.MAX_POSITIONS
    if hasattr(logic, 'TRAIL_R_LADDER'):
        monitor_engine.DEFAULTS['TRAIL_R_LADDER'] = logic.TRAIL_R_LADDER
    strategy_logic = logic
    print("[bootstrap] ✅ Strategy logic loaded from AI.")
except ImportError:
    print("[bootstrap] ℹ️ No strategy_logic.py found, using built-in defaults.")
except Exception as e:
    print(f"[bootstrap] ❌ Error loading strategy_logic: {e}")
    # Tetap jalan dengan fallback

# ==================== START SERVICES ====================
# 1. WebSocket feed (background)
api_client.ws_feed.start()

# 2. Price cache watchdog (background)
threading.Thread(target=api_client._price_cache_loop, daemon=True).start()

# 3. Flask (background)
threading.Thread(target=telegram_bot.run_flask, daemon=True).start()

# 4. Bot loop (main thread)
telegram_bot.bot_loop()

# ==================== KEEP ALIVE ====================
# Jika bot_loop() berhenti (jarang), keep process alive
while True:
    time.sleep(3600)