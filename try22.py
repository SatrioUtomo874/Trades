#!/usr/bin/env python3
"""
SIMPLE TRADING SIMULATOR BOT – Educational Game
Chart historis, entry TP/SL, next candle.
Token & Chat ID sudah diatur.
"""

import os
import json
import io
import random
import asyncio
import logging
import threading
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode

# ================== KONFIGURASI ==================
TELEGRAM_TOKEN = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
CHAT_ID = "8041197505"
BINANCE_API = "https://api.binance.com/api/v3/klines"
CACHE_DIR = "cache"
SESSIONS_FILE = "sessions.json"

SUPPORTED_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
INTERVAL = "4h"  # timeframe tetap H4 untuk kemudahan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(CACHE_DIR, exist_ok=True)

# ================== FLASK UNTUK RENDER ==================
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

# ================== FUNGSI DATA ==================
def fetch_klines(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Ambil data historis, cache di file parquet."""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{INTERVAL}.parquet")
    try:
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            if len(df) >= limit:
                return df.tail(limit)
    except:
        pass

    try:
        params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
        resp = requests.get(BINANCE_API, params=params, timeout=30)
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df.to_parquet(cache_file)
        return df.tail(limit)
    except Exception as e:
        logger.error(f"Gagal fetch: {e}")
        return pd.DataFrame()

def generate_chart(df: pd.DataFrame, symbol: str, title: str = "") -> io.BytesIO:
    """Buat chart candlestick sederhana."""
    mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="inherit", wick="inherit", volume="in")
    style = mpf.make_mpf_style(marketcolors=mc, facecolor="#1e1e1e", gridcolor="#444", edgecolor="#444", figcolor="#1e1e1e")

    fig, axes = mpf.plot(df, type="candle", style=style, volume=False,
                         title=f"{symbol} {title}", ylabel="Price",
                         returnfig=True, figsize=(8, 5), tight_layout=True)

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = round(rsi.iloc[-1], 2)
    fig.text(0.5, 0.02, f"RSI: {rsi_val}", ha="center", va="center",
             color="white", fontsize=12, fontweight="bold",
             bbox=dict(facecolor="#333333", alpha=0.8, boxstyle="round"))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#1e1e1e")
    plt.close(fig)
    buf.seek(0)
    return buf

# ================== SESI PENGGUNA ==================
def load_sessions():
    if not os.path.exists(SESSIONS_FILE):
        return {}
    with open(SESSIONS_FILE, "r") as f:
        return json.load(f)

def save_sessions(sessions):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2, default=str)

def get_session(user_id: int):
    return load_sessions().get(str(user_id))

def update_session(user_id: int, data):
    sessions = load_sessions()
    sessions[str(user_id)] = data
    save_sessions(sessions)

# ================== GAME LOGIC ==================
def new_game():
    symbol = random.choice(SUPPORTED_PAIRS)
    df = fetch_klines(symbol, 400)
    if len(df) < 100:
        raise ValueError("Data tidak cukup")
    start_idx = random.randint(80, len(df) - 20)
    current_time = df.index[start_idx].to_pydatetime()
    return {
        "symbol": symbol,
        "current_time": current_time.isoformat(),
        "position": None,
        "stats": {"wins": 0, "losses": 0, "total": 0}
    }

def get_visible_df(session):
    df = fetch_klines(session["symbol"], 500)
    if df.empty:
        return df
    current_time = pd.Timestamp(session["current_time"])
    return df[df.index <= current_time].copy()

def advance_time(session):
    current = pd.Timestamp(session["current_time"])
    session["current_time"] = (current + timedelta(hours=4)).isoformat()
    update_session(session["user_id"], session)

def check_tp_sl(session):
    pos = session.get("position")
    if not pos:
        return None
    df = get_visible_df(session)
    if len(df) < 2:
        return None
    last_close = df["close"].iloc[-1]
    if pos["type"] == "buy":
        if last_close >= pos["tp"]:
            return "tp"
        elif last_close <= pos["sl"]:
            return "sl"
    else:
        if last_close <= pos["tp"]:
            return "tp"
        elif last_close >= pos["sl"]:
            return "sl"
    return None

# ================== HANDLER TELEGRAM ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        session = new_game()
        session["user_id"] = user_id
        update_session(user_id, session)
    except Exception as e:
        await update.message.reply_text(f"Gagal memulai: {e}")
        return
    await kirim_chart(update, session)

async def kirim_chart(update: Update, session, extra=""):
    df = get_visible_df(session)
    if df.empty:
        await update.message.reply_text("Data tidak tersedia")
        return
    pos = session.get("position")
    title = f"POSITION: {pos['type'].upper()} @ {pos['entry']:.4f}" if pos else ""
    buf = generate_chart(df, session["symbol"], title)
    caption = f"{session['symbol']} {INTERVAL}"
    if pos:
        caption += f"\nEntry: {pos['entry']:.4f} | TP: {pos['tp']:.4f} | SL: {pos['sl']:.4f}"
    if extra:
        caption += f"\n{extra}"
    await update.message.reply_photo(buf, caption=caption)

async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session:
        await update.message.reply_text("Ketik /start dulu")
        return
    if session.get("position"):
        await update.message.reply_text("Anda sudah punya posisi")
        return
    try:
        tp = float(context.args[0])
        sl = float(context.args[1])
    except:
        await update.message.reply_text("Format: /buy <tp> <sl>\nContoh: /buy 50000 49000")
        return
    df = get_visible_df(session)
    if df.empty:
        await update.message.reply_text("Data tidak tersedia")
        return
    entry = df["close"].iloc[-1]
    if sl >= entry or tp <= entry:
        await update.message.reply_text("SL harus < entry, TP > entry")
        return
    session["position"] = {"type": "buy", "entry": entry, "tp": tp, "sl": sl}
    update_session(user_id, session)
    await kirim_chart(update, session, "BUY terpasang")

async def sell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session:
        await update.message.reply_text("Ketik /start dulu")
        return
    if session.get("position"):
        await update.message.reply_text("Anda sudah punya posisi")
        return
    try:
        tp = float(context.args[0])
        sl = float(context.args[1])
    except:
        await update.message.reply_text("Format: /sell <tp> <sl>\nContoh: /sell 45000 46000")
        return
    df = get_visible_df(session)
    if df.empty:
        await update.message.reply_text("Data tidak tersedia")
        return
    entry = df["close"].iloc[-1]
    if sl <= entry or tp >= entry:
        await update.message.reply_text("SL harus > entry, TP < entry")
        return
    session["position"] = {"type": "sell", "entry": entry, "tp": tp, "sl": sl}
    update_session(user_id, session)
    await kirim_chart(update, session, "SELL terpasang")

async def next_candle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session:
        await update.message.reply_text("Ketik /start dulu")
        return
    if not session.get("position"):
        await update.message.reply_text("Belum ada posisi. Gunakan /buy atau /sell")
        return

    advance_time(session)
    result = check_tp_sl(session)
    if result:
        session["position"] = None
        session["stats"]["total"] += 1
        if result == "tp":
            session["stats"]["wins"] += 1
            msg = "TP TERCAPAI! Anda MENANG"
        else:
            session["stats"]["losses"] += 1
            msg = "SL TERCAPAI! Anda KALAH"
        update_session(user_id, session)
        await kirim_chart(update, session, msg)
        await update.message.reply_text(
            f"{msg}\nWins: {session['stats']['wins']} Losses: {session['stats']['losses']} Total: {session['stats']['total']}"
        )
    else:
        update_session(user_id, session)
        await kirim_chart(update, session, "Lanjut... TP/SL belum kena")

async def close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session or not session.get("position"):
        await update.message.reply_text("Tidak ada posisi")
        return
    session["position"] = None
    session["stats"]["total"] += 1
    update_session(user_id, session)
    await update.message.reply_text("Posisi ditutup manual")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session:
        await update.message.reply_text("Belum ada sesi")
        return
    s = session["stats"]
    total = s["total"]
    wr = f"{s['wins']/total*100:.1f}%" if total > 0 else "N/A"
    await update.message.reply_text(
        f"Total: {total}\nWins: {s['wins']}\nLosses: {s['losses']}\nWin Rate: {wr}"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start - Mulai sesi baru\n"
        "/buy <tp> <sl> - Entry BUY\n"
        "/sell <tp> <sl> - Entry SELL\n"
        "/next - Maju 1 candle\n"
        "/close - Tutup posisi manual\n"
        "/stats - Lihat statistik\n"
        "/help - Bantuan\n\n"
        "Data historis nyata dari Binance. Latih analisa teknikal Anda!"
    )

# ================== MAIN ==================
async def main():
    app_bot = Application.builder().token(TELEGRAM_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CommandHandler("buy", buy))
    app_bot.add_handler(CommandHandler("sell", sell))
    app_bot.add_handler(CommandHandler("next", next_candle))
    app_bot.add_handler(CommandHandler("close", close))
    app_bot.add_handler(CommandHandler("stats", stats))
    app_bot.add_handler(CommandHandler("help", help_cmd))

    # Kirim pesan selamat datang ke chat ID yang ditentukan
    await app_bot.bot.send_message(
        chat_id=CHAT_ID,
        text="Bot Simulasi Trading siap!\nKetik /start untuk mulai berlatih.\n/help untuk bantuan."
    )

    # Mulai polling
    await app_bot.run_polling()

if __name__ == "__main__":
    # Jalankan Flask di thread terpisah
    threading.Thread(target=run_flask, daemon=True).start()

    # Jalankan bot di thread utama
    asyncio.run(main())
