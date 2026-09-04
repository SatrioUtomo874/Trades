import os
import json
import time
import asyncio
import threading
from datetime import datetime

from dotenv import load_dotenv

import strategy_V1 as strategy
import learn_V1 as learn

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes
)

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")

MODE = "off"
AUTO = False

MARGIN = 1
LEVERAGE = 10
SCAN_LIMIT = 50
TIMER = 3

ACTIVE_TRADES = {}
BANNED = {}

SIMULATED_ORDERS = []

WS_RUNNING = False


def log(msg):
    print(f"[{datetime.now()}] {msg}")


def ban_coin(symbol, duration):
    BANNED[symbol] = time.time() + duration


def is_banned(symbol):
    if symbol not in BANNED:
        return False

    if time.time() > BANNED[symbol]:
        del BANNED[symbol]
        return False

    return True


def calculate_quantity(price):
    value = MARGIN * LEVERAGE
    return round(value / price, 6)


def create_pending_order(symbol, side, entry, tp, sl, quantity=None):

    if quantity is None:
        quantity = calculate_quantity(entry)

    order = {
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "qty": quantity,
        "status": "PENDING",
        "mode": MODE,
        "time": str(datetime.now())
    }

    if MODE == "off":
        SIMULATED_ORDERS.append(order)
        ACTIVE_TRADES[symbol] = order
        return order


    # Binance execution placeholder
    # dibuat terpisah agar tidak bergantung pada strategy

    ACTIVE_TRADES[symbol] = order
    return order



async def cmd_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global AUTO

    AUTO = True

    await update.message.reply_text(
        "🤖 Auto mode ON"
    )



async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global AUTO

    AUTO = False

    await update.message.reply_text(
        "🛑 Auto mode OFF"
    )



async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global MODE

    if context.args:
        MODE = context.args[0]

    await update.message.reply_text(
        f"Mode: {MODE}"
    )



async def cmd_pending(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if len(context.args) < 5:
        await update.message.reply_text(
            "/pending SYMBOL SIDE ENTRY TP SL"
        )
        return

    symbol = context.args[0]
    side = context.args[1]

    entry = float(context.args[2])
    tp = float(context.args[3])
    sl = float(context.args[4])

    order = create_pending_order(
        symbol,
        side,
        entry,
        tp,
        sl
    )

    learn.record_candidate(order)

    await update.message.reply_text(
        json.dumps(order, indent=2)
    )



async def cmd_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        json.dumps(
            ACTIVE_TRADES,
            indent=2
        )
    )



async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        json.dumps(
            learn.get_stats(),
            indent=2
        )
    )



async def cmd_full(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        json.dumps(
            learn.full_command(),
            indent=2
        )
    )



async def cmd_save(update: Update, context: ContextTypes.DEFAULT_TYPE):

    learn.save_state()

    await update.message.reply_text(
        "Saved"
    )



async def cmd_open(update: Update, context: ContextTypes.DEFAULT_TYPE):

    learn.open_state()

    await update.message.reply_text(
        "Opened"
    )



async def auto_loop():

    while True:

        if AUTO:

            log("AUTO scanning")

            # scanner Bybit + strategy bridge
            # ditambahkan saat exchange layer masuk

        await asyncio.sleep(TIMER)



def websocket_worker():

    global WS_RUNNING

    WS_RUNNING = True

    while WS_RUNNING:

        # single websocket monitor
        # cek ACTIVE_TRADES

        time.sleep(1)



async def start():

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("auto", cmd_auto))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("pending", cmd_pending))
    app.add_handler(CommandHandler("trade", cmd_trade))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("full", cmd_full))
    app.add_handler(CommandHandler("save", cmd_save))
    app.add_handler(CommandHandler("open", cmd_open))

    threading.Thread(
        target=websocket_worker,
        daemon=True
    ).start()

    asyncio.create_task(auto_loop())

    await app.run_polling()



if __name__ == "__main__":
    asyncio.run(start())
