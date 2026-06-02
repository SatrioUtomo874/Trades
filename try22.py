#!/usr/bin/env python3
"""
TELEGRAM TRADING SIMULATOR BOT – Educational Game
Uses real historical Binance data, no real money.
Multi-TF mode (H4/H1/M15) + Scalping mode (M15).
RSI displayed on charts, strict no-future-data rule.
Sessions stored locally in JSON.
Token hardcoded for immediate use.
Supports Render 24/7 deployment with Flask health-check.
"""

import os
import json
import io
import random
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode

# ================== CONFIG ==================
TELEGRAM_TOKEN = "7585154530:AAHk9gwv8i2KnAf14kniYtBL9RclZt4Tt0o"
CHAT_ID = "8041197505"
BINANCE_API = "https://api.binance.com/api/v3/klines"
CACHE_DIR = "cache"
SESSIONS_FILE = "sessions.json"

SUPPORTED_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
TIMEFRAMES = {"H4": "4h", "H1": "1h", "M15": "15m"}
INTERVALS_SEC = {"H4": 4*3600, "H1": 3600, "M15": 900}

COLORS = {
    "up": "#26a69a",
    "down": "#ef5350",
    "bg": "#1e1e1e",
    "grid": "#444444",
    "text": "#ffffff"
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(CACHE_DIR, exist_ok=True)

# ================== FLASK APP ==================
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

# ================== FUNGSI DATA ==================
def fetch_klines(symbol: str, interval: str, limit: int = 500) -> Optional[pd.DataFrame]:
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}.parquet")
    try:
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            if len(df) >= limit:
                return df.tail(limit)
    except Exception:
        pass

    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(BINANCE_API, params=params, timeout=30)
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
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
        logger.error(f"Gagal fetch kline: {e}")
        return None

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

# ================== GENERATE CHART ==================
def generate_chart(df: pd.DataFrame, symbol: str, interval: str, title: str = "") -> io.BytesIO:
    mc = mpf.make_marketcolors(
        up=COLORS["up"], down=COLORS["down"],
        edge="inherit", wick="inherit", volume="in"
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor=COLORS["bg"],
        gridcolor=COLORS["grid"],
        edgecolor=COLORS["grid"],
        figcolor=COLORS["bg"],
    )

    rsi_val = calculate_rsi(df)

    fig, axes = mpf.plot(
        df, type="candle", style=style, volume=False,
        title=f"{symbol} {interval} {title}",
        ylabel="Price", ylabel_lower="",
        returnfig=True, figsize=(8, 5), tight_layout=True
    )

    fig.text(0.5, 0.02, f"RSI: {rsi_val}", ha="center", va="center",
             color=COLORS["text"], fontsize=12, fontweight="bold",
             bbox=dict(facecolor="#333333", alpha=0.8, boxstyle="round"))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf

# ================== SESI PENGGUNA ==================
def load_sessions() -> Dict[int, Any]:
    if not os.path.exists(SESSIONS_FILE):
        return {}
    with open(SESSIONS_FILE, "r") as f:
        return json.load(f)

def save_sessions(sessions: Dict[int, Any]):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2, default=str)

def get_session(user_id: int) -> Optional[Dict]:
    sessions = load_sessions()
    return sessions.get(str(user_id))

def update_session(user_id: int, data: Dict):
    sessions = load_sessions()
    sessions[str(user_id)] = data
    save_sessions(sessions)

# ================== GAME LOGIC ==================
def start_new_game(user_id: int) -> Dict:
    symbol = random.choice(SUPPORTED_PAIRS)
    mode = random.choice(["multi_tf", "scalp"])
    if mode == "scalp":
        interval = "M15"
    else:
        interval = "H4"

    df_h4 = fetch_klines(symbol, "4h", 400)
    if df_h4 is None or len(df_h4) < 100:
        raise ValueError("Data tidak mencukupi")

    max_start = len(df_h4) - 20
    min_start = 80
    start_idx = random.randint(min_start, max_start)
    current_time = df_h4.index[start_idx].to_pydatetime()

    session = {
        "symbol": symbol,
        "mode": mode,
        "interval": interval,
        "current_time": current_time.isoformat(),
        "position": None,
        "stats": {"wins": 0, "losses": 0, "total": 0},
        "game_over": False,
        "start_idx_h4": start_idx,
    }
    update_session(user_id, session)
    return session

def get_visible_data(session: Dict) -> pd.DataFrame:
    symbol = session["symbol"]
    interval = TIMEFRAMES[session["interval"]]
    df = fetch_klines(symbol, interval, 500)
    if df is None:
        return pd.DataFrame()
    current_time = pd.Timestamp(session["current_time"])
    return df[df.index <= current_time].copy()

def advance_time(session: Dict):
    interval = session["interval"]
    sec = INTERVALS_SEC[interval]
    current = pd.Timestamp(session["current_time"])
    session["current_time"] = (current + timedelta(seconds=sec)).isoformat()
    update_session(session["user_id"], session)

def check_tp_sl(session: Dict) -> Optional[str]:
    pos = session.get("position")
    if not pos:
        return None
    df = get_visible_data(session)
    if len(df) < 2:
        return None
    latest_close = df["close"].iloc[-1]
    if pos["type"] == "buy":
        if latest_close >= pos["tp"]:
            return "tp"
        elif latest_close <= pos["sl"]:
            return "sl"
    else:
        if latest_close <= pos["tp"]:
            return "tp"
        elif latest_close >= pos["sl"]:
            return "sl"
    return None

# ================== HANDLER TELEGRAM ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        session = start_new_game(user_id)
        session["user_id"] = user_id
        update_session(user_id, session)
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal memulai: {e}")
        return
    await kirim_chart(update, context, session)

async def kirim_chart(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Dict, extra_text: str = ""):
    df = get_visible_data(session)
    if df.empty:
        await update.message.reply_text("❌ Data tidak tersedia.")
        return

    pos = session.get("position")
    title = ""
    if pos:
        title = f"POSITION: {pos['type'].upper()} @ {pos['entry']:.4f}"

    buf = generate_chart(df, session["symbol"], session["interval"], title)
    caption = f"📊 {session['symbol']} {session['interval']} | Mode: {session['mode']}"
    if pos:
        caption += f"\n🔵 Entry: {pos['entry']:.4f} | TP: {pos['tp']:.4f} | SL: {pos['sl']:.4f}"
    if extra_text:
        caption += f"\n{extra_text}"

    await update.message.reply_photo(buf, caption=caption)

async def next_candle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session:
        await update.message.reply_text("⚠️ Tidak ada sesi. /start")
        return

    pos = session.get("position")
    if not pos:
        await update.message.reply_text("⚠️ Belum ada posisi. Gunakan /buy atau /sell.")
        return

    advance_time(session)
    update_session(user_id, session)

    result = check_tp_sl(session)
    if result:
        session["position"] = None
        session["stats"]["total"] += 1
        if result == "tp":
            session["stats"]["wins"] += 1
            msg = "🏆 TP tersentuh! Anda MENANG."
        else:
            session["stats"]["losses"] += 1
            msg = "💔 SL tersentuh! Anda KALAH."
        update_session(user_id, session)
        await kirim_chart(update, context, session, extra_text=msg)
        await update.message.reply_text(
            f"{msg}\n📈 Stat: Wins:{session['stats']['wins']} Losses:{session['stats']['losses']} Total:{session['stats']['total']}"
        )
    else:
        await kirim_chart(update, context, session, extra_text="Lanjut... TP/SL belum tersentuh.")

async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session:
        await update.message.reply_text("⚠️ /start dulu.")
        return
    if session.get("position"):
        await update.message.reply_text("⚠️ Sudah punya posisi. /close dulu.")
        return
    try:
        args = context.args
        if len(args) != 2:
            await update.message.reply_text("Format: /buy <tp> <sl>\nContoh: /buy 50000 49000")
            return
        tp = float(args[0])
        sl = float(args[1])
    except ValueError:
        await update.message.reply_text("❌ TP/SL harus angka.")
        return

    df = get_visible_data(session)
    if df.empty:
        await update.message.reply_text("❌ Gagal baca harga.")
        return
    entry_price = df["close"].iloc[-1]
    if sl >= entry_price or tp <= entry_price:
        await update.message.reply_text("❌ SL harus < entry, TP > entry.")
        return

    session["position"] = {
        "type": "buy",
        "entry": entry_price,
        "tp": tp,
        "sl": sl,
    }
    update_session(user_id, session)
    await kirim_chart(update, context, session, extra_text=f"✅ BUY @ {entry_price:.4f}")

async def sell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session:
        await update.message.reply_text("⚠️ /start dulu.")
        return
    if session.get("position"):
        await update.message.reply_text("⚠️ Sudah punya posisi.")
        return
    try:
        args = context.args
        if len(args) != 2:
            await update.message.reply_text("Format: /sell <tp> <sl>\nContoh: /sell 45000 46000")
            return
        tp = float(args[0])
        sl = float(args[1])
    except ValueError:
        await update.message.reply_text("❌ TP/SL harus angka.")
        return

    df = get_visible_data(session)
    if df.empty:
        await update.message.reply_text("❌ Gagal baca harga.")
        return
    entry_price = df["close"].iloc[-1]
    if sl <= entry_price or tp >= entry_price:
        await update.message.reply_text("❌ SL harus > entry, TP < entry.")
        return

    session["position"] = {
        "type": "sell",
        "entry": entry_price,
        "tp": tp,
        "sl": sl,
    }
    update_session(user_id, session)
    await kirim_chart(update, context, session, extra_text=f"✅ SELL @ {entry_price:.4f}")

async def close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session or not session.get("position"):
        await update.message.reply_text("⚠️ Tidak ada posisi.")
        return
    session["position"] = None
    session["stats"]["total"] += 1
    update_session(user_id, session)
    await update.message.reply_text("🔒 Posisi ditutup manual.")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session:
        await update.message.reply_text("⚠️ Belum ada sesi.")
        return
    s = session["stats"]
    total = s["total"]
    wr = f"{s['wins']/total*100:.1f}%" if total > 0 else "N/A"
    await update.message.reply_text(
        f"📊 Statistik:\nTotal: {total}\nWins: {s['wins']}\nLosses: {s['losses']}\nWin Rate: {wr}"
    )

async def tf_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    if not session or session["mode"] != "multi_tf":
        await update.message.reply_text("⚠️ Hanya untuk mode Multi-TF.")
        return
    try:
        new_tf = context.args[0].upper()
        if new_tf not in TIMEFRAMES:
            raise ValueError
    except (IndexError, ValueError):
        await update.message.reply_text("Gunakan: /tf h4, /tf h1, /tf m15")
        return
    session["interval"] = new_tf
    update_session(user_id, session)
    await kirim_chart(update, context, session, extra_text=f"⏱ Timeframe: {new_tf}")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎮 <b>Trading Simulator Bot</b>\n\n"
        "Perintah:\n"
        "/start - Mulai sesi baru\n"
        "/buy <tp> <sl> - Entry BUY\n"
        "/sell <tp> <sl> - Entry SELL\n"
        "/next - Maju 1 candle\n"
        "/close - Tutup posisi manual\n"
        "/stats - Lihat statistik\n"
        "/tf <h4/h1/m15> - Ganti timeframe (Multi-TF)\n"
        "/help - Bantuan\n\n"
        "Data historis asli dari Binance. Latih analisa teknikal!",
        parse_mode=ParseMode.HTML
    )

# ================== MAIN ==================
def main():
    """Jalankan bot di thread utama."""
    app_bot = Application.builder().token(TELEGRAM_TOKEN).build()

    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CommandHandler("next", next_candle))
    app_bot.add_handler(CommandHandler("buy", buy))
    app_bot.add_handler(CommandHandler("sell", sell))
    app_bot.add_handler(CommandHandler("close", close))
    app_bot.add_handler(CommandHandler("stats", stats))
    app_bot.add_handler(CommandHandler("tf", tf_switch))
    app_bot.add_handler(CommandHandler("help", help_cmd))

    # Kirim welcome message
    async def send_welcome():
        await app_bot.bot.send_message(
            chat_id=CHAT_ID,
            text="🎮 <b>Bot Simulasi Trading siap digunakan!</b>\n\nGunakan /start untuk memulai sesi permainan baru.\nKetik /help untuk bantuan.",
            parse_mode=ParseMode.HTML
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_welcome())

    # Mulai polling
    app_bot.run_polling()

if __name__ == "__main__":
    # Jalankan Flask di thread terpisah
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Bot berjalan di thread utama
    main()
