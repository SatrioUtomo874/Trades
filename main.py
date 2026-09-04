import os
import asyncio
import logging
import requests
import time
import json
import websocket
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import random
import threading
import hashlib
import hmac
from strategy import strategy_logic

load_dotenv("trades.env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")


TRADE_STATS = {
    "TP": 0,
    "SL": 0,
    "TRAIL": 0
}

BOT = None
AUTO_MODE = False
SCAN_LIMIT = 50
SCAN_TIMER = 5
BANNED_COINS = {}
TOP_50_COINS = []
CANDLE_DATA = {}
MODE = "off"
MARGIN = 1.0
LEVERAGE = 10
MONITOR_TASKS = {}
WS_THREADS = {}

# Launcher lifecycle: try.py owns Telegram polling; main.py exposes handlers/lifecycle.
MAIN_APP = None
MAIN_RUNNING = False
AUTO_TASK = None



PENDING_ORDERS = []
CURRENT_PRICES = {}
MONITOR_TASK = None
STOP_MONITORS = {}
WS_THREAD = None
WS_STOP_EVENT = None

def is_allowed(update):
    return str(update.effective_user.id) == ALLOWED_USER_ID


def get_public_ip():
    try:
        response = requests.get(
            "https://api.ipify.org?format=json",
            timeout=10
        )
        response.raise_for_status()
        return response.json()["ip"]

    except Exception as e:
        logging.error("IP error: %s", e)
        return None


def get_bybit_futures_coins():
    url = "https://api.bybit.com/v5/market/instruments-info"

    params = {
        "category": "linear",
        "limit": 1000
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if data["retCode"] != 0:
        raise Exception(data["retMsg"])

    symbols = []

    for item in data["result"]["list"]:
        if item["status"] == "Trading":
            symbols.append(item["symbol"])

    return symbols


def get_binance_futures_coins():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    symbols = []

    for item in data["symbols"]:
        if item["status"] == "TRADING":
            symbols.append(item["symbol"])

    return symbols


def get_common_coins():
    bybit = set(get_bybit_futures_coins())
    binance = set(get_binance_futures_coins())

    return sorted(bybit & binance)


def get_top_volume_coins(common_coins):
    url = "https://api.bybit.com/v5/market/tickers"

    params = {
        "category": "linear"
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if data["retCode"] != 0:
        raise Exception(data["retMsg"])

    volume_data = []

    for item in data["result"]["list"]:
        symbol = item["symbol"]

        if symbol in common_coins:
            volume = float(item["volume24h"])

            volume_data.append({
                "symbol": symbol,
                "volume": volume
            })

        volume_data = [
        coin
        for coin in volume_data
        if not is_coin_banned(coin["symbol"])
    ]

    btc = next(
        (coin for coin in volume_data if coin["symbol"] == "BTCUSDT"),
        None
    )

    if btc:
        volume_data.remove(btc)
        volume_data = [btc] + volume_data

    return volume_data[:50]


def get_bybit_672_candles(symbol):
    url = "https://api.bybit.com/v5/market/kline"

    candles = []

    end_time = int(
        datetime.now().timestamp() * 1000
    )

    while len(candles) < 672:

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": "15",
            "limit": 200,
            "end": end_time
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if data["retCode"] != 0:
            raise Exception(data["retMsg"])

        batch = data["result"]["list"]

        if not batch:
            break

        candles.extend(batch)

        oldest_time = min(
            int(candle[0])
            for candle in batch
        )

        end_time = oldest_time - 1

    candles = sorted(
        candles,
        key=lambda x: int(x[0])
    )

    current_time = int(
        datetime.now().timestamp() * 1000
    )

    interval_ms = 15 * 60 * 1000

    candles = [
        candle
        for candle in candles
        if int(candle[0]) + interval_ms <= current_time
    ]

    return candles[-672:]


async def scan_coin(symbol):
    try:
        candles = await asyncio.to_thread(
            get_bybit_672_candles,
            symbol
        )

        if len(candles) < 672:
            return False

        CANDLE_DATA[symbol] = candles
        signal = strategy_logic(
            symbol,
            candles
        )

        if signal:

            quantity = auto_quantity(
                signal["entry"]
            )

            success, result = pending_order(
                symbol,
                signal["side"],
                signal["entry"],
                signal["tp"],
                signal["sl"],
                quantity
            )

            if success:

                order = result

                order["trail_enabled"] = (
                    signal["trail"]["enabled"]
                )

                order["trail_activation_r"] = (
                    signal["trail"]["activation_r"]
                )

                order["trail_distance_r"] = (
                    signal["trail"]["distance_r"]
                )

                order["initial_risk"] = abs(
                    order["entry"]
                    - order["sl"]
                )

                order["trail_active"] = False

                await start_trade_monitor()

                print(
                    f"[STRATEGY ORDER] "
                    f"{symbol} {signal['side']} | "
                    f"Entry={signal['entry']} | "
                    f"TP={signal['tp']} | "
                    f"SL={signal['sl']}"
                )

        return True

    except Exception as e:
        logging.error(
            "[SCAN ERROR] %s: %s",
            symbol,
            e
        )

        return False


async def timer_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SCAN_TIMER

    if not is_allowed(update):
        return

    if not context.args:
        await update.message.reply_text(
            f"⏱️ Timer scanning saat ini: {SCAN_TIMER} detik"
        )
        return

    try:
        timer = float(context.args[0])

        if timer < 0:
            await update.message.reply_text(
                "❌ Timer tidak boleh kurang dari 0."
            )
            return

        SCAN_TIMER = timer

        await update.message.reply_text(
            f"✅ Timer scanning diubah menjadi {SCAN_TIMER} detik."
        )

    except ValueError:
        await update.message.reply_text(
            "❌ Format: /timer 3"
        )

async def send_log(bot, message):
    await bot.send_message(
        chat_id=ALLOWED_USER_ID,
        text=f"🤖 {message}"
    )

async def margin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MARGIN

    if not is_allowed(update):
        return

    if not context.args:
        await update.message.reply_text(
            f"💰 Margin saat ini: ${MARGIN}"
        )
        return

    try:
        margin = float(context.args[0])

        if margin <= 0:
            await update.message.reply_text(
                "❌ Margin harus lebih dari 0."
            )
            return

        MARGIN = margin

        await update.message.reply_text(
            f"✅ Margin diubah menjadi ${MARGIN}"
        )

    except ValueError:
        await update.message.reply_text(
            "❌ Format: /margin 1"
        )

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_allowed(update):
        return

    await update.message.reply_text(
        "Bot connected.\n\n"
        "/ip - cek public IP\n"
        "/auto - aktifkan auto mode\n"
        "/stop - matikan auto mode\n"
        "/scan <jumlah> - jumlah koin yang discan\n"
        "/koin - lihat Top 50"
    )

async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MODE

    if not is_allowed(update):
        return

    if not context.args:
        await update.message.reply_text(
            f"⚙️ Mode saat ini: {MODE}"
        )
        return

    mode = context.args[0].lower()

    if mode not in ["on", "off"]:
        await update.message.reply_text(
            "❌ Format: /mode on atau /mode off"
        )
        return

    MODE = mode

    if MODE == "off":

        await update.message.reply_text(
            "🟢 Mode simulasi: ON"
        )

    else:

        try:
            balance = await asyncio.to_thread(
                get_binance_balance
            )

            await update.message.reply_text(
                f"🔴 Real trade mode: ON\n"
                f"💰 Binance Futures Balance: "
                f"${balance:.2f}"
            )

        except Exception as e:

            MODE = "off"

            await update.message.reply_text(
                f"❌ Gagal mengakses Binance.\n"
                f"Mode dikembalikan ke OFF.\n\n"
                f"Error: {e}"
            )

async def ip_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_allowed(update):
        return

    ip = get_public_ip()

    if ip:
        await update.message.reply_text(
            f"🌐 Public IP: {ip}"
        )
    else:
        await update.message.reply_text(
            "❌ Gagal mendapatkan IP."
        )

async def monitor_trade():
    try:
        while True:

            active_orders = [
                order
                for order in PENDING_ORDERS
                if order["status"] in ["PENDING", "FILLED"]
            ]

            if not active_orders:
                await asyncio.sleep(0.1)
                continue

            for order in active_orders:

                symbol = order["symbol"]
                price = CURRENT_PRICES.get(symbol)

                if price is None:
                    continue

                entry = order["entry"]
                tp = order["tp"]
                sl = order["sl"]
                quantity = order["quantity"]
                side = order["side"]

                if order["status"] == "PENDING":

                    if side == "Buy":

                        if price >= tp:
                            await finish_order(
                                order,
                                "EXPIRED",
                                price
                            )
                            continue

                        if price <= entry:
                            order["status"] = "FILLED"
                            order["entry_time"] = time.time()

                            await send_log(
                                BOT,
                                f"🟢 {symbol} BUY FILLED\n"
                                f"Entry: {price}"
                            )

                    else:

                        if price <= tp:
                            await finish_order(
                                order,
                                "EXPIRED",
                                price
                            )
                            continue

                        if price >= entry:
                            order["status"] = "FILLED"
                            order["entry_time"] = time.time()

                            await send_log(
                                BOT,
                                f"🔴 {symbol} SELL FILLED\n"
                                f"Entry: {price}"
                            )

                elif order["status"] == "FILLED":
                    if order.get("trail_enabled"):

                        risk = order["initial_risk"]

                        activation_r = (
                            order["trail_activation_r"]
                        )

                        distance_r = (
                            order["trail_distance_r"]
                        )

                        activation_distance = (
                            risk * activation_r
                        )

                        trail_distance = (
                            risk * distance_r
                        )

                        if order["side"] == "Buy":

                            activation_price = (
                                order["entry"]
                                + activation_distance
                            )

                            if price >= activation_price:

                                trail_order(
                                    symbol,
                                    trail_distance
                                )

                        else:

                            activation_price = (
                                order["entry"]
                                - activation_distance
                            )

                            if price <= activation_price:

                                trail_order(
                                    symbol,
                                    trail_distance
                                )
                    if side == "Buy":

                        order["pnl"] = (
                            price - entry
                        ) * quantity

                        if price >= tp:
                            await finish_order(
                                order,
                                "TP",
                                price
                            )
                            continue

                        if price <= sl:
                            await finish_order(
                                order,
                                "TRAIL"
                                if order.get("trail_active")
                                else "SL",
                                price
                            )
                            continue

                    else:

                        order["pnl"] = (
                            entry - price
                        ) * quantity

                        if price <= tp:
                            await finish_order(
                                order,
                                "TP",
                                price
                            )
                            continue

                        if price >= sl:
                            await finish_order(
                                order,
                                "TRAIL"
                                if order.get("trail_active")
                                else "SL",
                                price
                            )
                            continue

            await asyncio.sleep(0.05)

    except asyncio.CancelledError:
        logging.info("[MONITOR] Cancelled")
        raise

    except Exception as e:
        logging.error(
            "[MONITOR ERROR] %s",
            e
        )

async def auto_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    global AUTO_MODE

    if not is_allowed(update):
        return

    AUTO_MODE = True

    await update.message.reply_text(
        "🤖 Auto mode: ON\n"
        "Bot siap menjalankan pencarian pair."
    )

async def trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return

    if not PENDING_ORDERS:
        await update.message.reply_text(
            "📭 Tidak ada trade aktif."
        )
        return

    lines = ["📊 ACTIVE TRADES\n"]

    for index, order in enumerate(PENDING_ORDERS, start=1):
        symbol = order["symbol"]
        price = CURRENT_PRICES.get(
            symbol,
            "N/A"
        )

        lines.append(
            f"{index}. {symbol}\n"
            f"   {order['side']}\n"
            f"   Entry : {order['entry']}\n"
            f"   TP    : {order['tp']}\n"
            f"   SL    : {order['sl']}\n"
            f"   Qty   : {order['quantity']}\n"
            f"   Now   : {price}\n"
            f"   Status: {order['status']}\n"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )

async def stop_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    global AUTO_MODE

    if not is_allowed(update):
        return

    AUTO_MODE = False

    await update.message.reply_text(
        "🛑 Auto mode: OFF"
    )

async def pending_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return

    if not TOP_50_COINS:
        await update.message.reply_text(
            "❌ TOP volume belum tersedia."
        )
        return

    available = [
        coin["symbol"]
        for coin in TOP_50_COINS
        if not is_coin_banned(coin["symbol"])
        and coin["symbol"] not in [
            order["symbol"] for order in PENDING_ORDERS
        ]
    ]

    if not available:
        await update.message.reply_text(
            "❌ Tidak ada koin yang tersedia untuk pending test."
        )
        return

    symbol = random.choice(available)

    try:
        price = await asyncio.to_thread(
            websocket_price,
            symbol
        )

        side = random.choice(["Buy", "Sell"])

        if side == "Buy":
            entry = price * 0.995
            tp = entry * 1.01
            sl = entry * 0.995
        else:
            entry = price * 1.005
            tp = entry * 0.99
            sl = entry * 1.005

        quantity = auto_quantity(entry)

        success, result = pending_order(
            symbol,
            side,
            entry,
            tp,
            sl,
            quantity
        )

        if not success:
            await update.message.reply_text(
                f"❌ Pending gagal: {result}"
            )
            return

        await update.message.reply_text(
            f"🟡 Pending dibuat\n"
            f"Pair: {symbol}\n"
            f"Side: {side}\n"
            f"Entry: {entry}\n"
            f"TP: {tp}\n"
            f"SL: {sl}\n"
            f"Qty: {quantity}"
        )

        start_trade_monitor()

    except Exception as e:
        logging.error("[PENDING ERROR] %s", e)

        await update.message.reply_text(
            f"❌ Pending error: {e}"
        )

async def scan_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    global SCAN_LIMIT

    if not is_allowed(update):
        return

    if not context.args:
        await update.message.reply_text(
            f"🔢 Scan limit saat ini: {SCAN_LIMIT}"
        )
        return

    try:
        total = int(context.args[0])

        if total < 1 or total > 50:
            await update.message.reply_text(
                "❌ Jumlah scan harus antara 1-50."
            )
            return

        SCAN_LIMIT = total

        await update.message.reply_text(
            f"✅ Scan limit diubah menjadi {SCAN_LIMIT} koin."
        )

    except ValueError:
        await update.message.reply_text(
            "❌ Format: /scan 20"
        )

async def leverage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LEVERAGE

    if not is_allowed(update):
        return

    if not context.args:
        await update.message.reply_text(
            f"⚙️ Leverage saat ini: {LEVERAGE}x"
        )
        return

    try:
        leverage = int(context.args[0])

        if leverage < 1:
            await update.message.reply_text(
                "❌ Leverage minimal 1x."
            )
            return

        LEVERAGE = leverage

        await update.message.reply_text(
            f"✅ Leverage diubah menjadi {LEVERAGE}x"
        )

    except ValueError:
        await update.message.reply_text(
            "❌ Format: /leverage 10"
        )

async def koin_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_allowed(update):
        return

    if not TOP_50_COINS:
        await update.message.reply_text(
            "❌ Belum ada data Top 50."
        )
        return

    text = "\n".join(
        f"{i}. {coin['symbol']}"
        for i, coin in enumerate(
            TOP_50_COINS,
            start=1
        )
    )

    await update.message.reply_text(
        f"📊 Top {len(TOP_50_COINS)} Volume:\n\n{text}"
    )

async def unban_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return

    if not context.args:
        await update.message.reply_text(
            "❌ Format: /unban BTCUSDT"
        )
        return

    symbol = context.args[0].upper()

    if symbol not in BANNED_COINS:
        await update.message.reply_text(
            f"ℹ️ {symbol} tidak sedang banned."
        )
        return

    del BANNED_COINS[symbol]

    await update.message.reply_text(
        f"✅ {symbol} berhasil di-unban."
    )

async def test_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_allowed(update):
        return

    try:
        await update.message.reply_text(
            "🧪 Test dimulai..."
        )

        common_coins = await asyncio.to_thread(
            get_common_coins
        )

        if not common_coins:
            await update.message.reply_text(
                "❌ Tidak ditemukan common pair Bybit + Binance."
            )
            return

        symbol = common_coins[0]

        await update.message.reply_text(
            f"🔎 Pair yang dipilih: {symbol}\n"
            f"📊 Mengambil 672 candle M15..."
        )

        candles = await asyncio.to_thread(
            get_bybit_672_candles,
            symbol
        )

        if len(candles) < 672:
            await update.message.reply_text(
                f"❌ Hanya mendapatkan {len(candles)} candle."
            )
            return

        CANDLE_DATA[symbol] = candles

        await update.message.reply_text(
            f"✅ Selesai!\n\n"
            f"Pair: {symbol}\n"
            f"Candle: {len(candles)}\n"
            f"Timeframe: M15\n"
            f"Data disimpan di memory."
        )

    except Exception as e:
        await update.message.reply_text(
            f"❌ Test error: {e}"
        )

def ban_coin(symbol, duration):
    expire_time = time.time() + duration
    BANNED_COINS[symbol] = expire_time

async def auto_loop(bot):
    global AUTO_MODE
    global TOP_50_COINS

    while True:

        if AUTO_MODE:

            try:
                await send_log(
                    bot,
                    "Mulai mencari common pairs..."
                )

                await asyncio.sleep(SCAN_TIMER)

                if not AUTO_MODE:
                    continue

                common_coins = await asyncio.to_thread(
                    get_common_coins
                )

                await send_log(
                    bot,
                    f"Common pairs ditemukan: "
                    f"{len(common_coins)}"
                )

                TOP_50_COINS = await asyncio.to_thread(
                    get_top_volume_coins,
                    common_coins
                )

                total = min(
                    SCAN_LIMIT,
                    len(TOP_50_COINS)
                )

                success = 0
                failed = 0

                await send_log(
                    bot,
                    f"Mulai scanning {total} koin..."
                )

                for index, coin in enumerate(
                    TOP_50_COINS[:total],
                    start=1
                ):

                    if not AUTO_MODE:
                        break

                    symbol = coin["symbol"]

                    result = await scan_coin(symbol)

                    if result:
                        status = "Success"
                        success += 1
                    else:
                        status = "Failed"
                        failed += 1

                    print(
                        f"[SCAN] [{index}/{total}] "
                        f"{symbol} {status}"
                    )

                    await asyncio.sleep(1)

                if AUTO_MODE:
                    await send_log(
                        bot,
                        f"Scanning selesai.\n"
                        f"Total: {total}\n"
                        f"Success: {success}\n"
                        f"Failed: {failed}"
                    )

            except Exception as e:

                await send_log(
                    bot,
                    f"❌ Auto error: {e}"
                )

        await asyncio.sleep(1)

def is_coin_banned(symbol):
    expire_time = BANNED_COINS.get(symbol)

    if expire_time is None:
        return False

    if time.time() >= expire_time:
        del BANNED_COINS[symbol]
        return False

    return True

def auto_quantity(entry_price):
    notional = MARGIN * LEVERAGE
    quantity = notional / entry_price

    return quantity

async def post_init(app):
    global BOT, AUTO_TASK

    BOT = app.bot

    if AUTO_TASK is None or AUTO_TASK.done():
        AUTO_TASK = app.create_task(auto_loop(app.bot))

async def banned_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return

    now = time.time()

    active_bans = []

    for symbol, expire_time in list(BANNED_COINS.items()):

        remaining = expire_time - now

        if remaining <= 0:
            del BANNED_COINS[symbol]
            continue

        active_bans.append(
            (symbol, remaining)
        )

    if not active_bans:
        await update.message.reply_text(
            "🔓 Tidak ada koin yang sedang banned."
        )
        return

    active_bans.sort(
        key=lambda x: x[1]
    )

    lines = ["🔒 KOIN BANNED\n"]

    for index, (symbol, remaining) in enumerate(
        active_bans,
        start=1
    ):
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        seconds = int(remaining % 60)

        lines.append(
            f"{index}. {symbol} — "
            f"{hours}j {minutes}m {seconds}d"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )

def pending_order(symbol, side, entry_price, tp_price, sl_price, quantity):
    symbol = symbol.upper()
    side = side.capitalize()

    if side not in ["Buy", "Sell"]:
        return False, "Side harus Buy atau Sell."

    if is_coin_banned(symbol):
        return False, f"{symbol} sedang banned."

    active_orders = [
        order
        for order in PENDING_ORDERS
        if order["status"] in ["PENDING", "FILLED"]
    ]

    if len(active_orders) >= 20:
        return False, "Maksimal 20 pair aktif."



    if quantity <= 0:
        return False, "Quantity harus lebih dari 0."

    if entry_price <= 0 or tp_price <= 0 or sl_price <= 0:
        return False, "Harga Entry, TP, dan SL harus lebih dari 0."

    if side == "Buy":
        if not (sl_price < entry_price < tp_price):
            return False, "Buy harus memenuhi: SL < Entry < TP."

    if side == "Sell":
        if not (tp_price < entry_price < sl_price):
            return False, "Sell harus memenuhi: TP < Entry < SL."

    for order in PENDING_ORDERS:
        if order["symbol"] == symbol:
            return False, f"{symbol} sudah memiliki order aktif."

    order = {
        "id": str(int(time.time() * 1000)),
        "symbol": symbol,
        "side": side,
        "entry": float(entry_price),
        "tp": float(tp_price),
        "sl": float(sl_price),
        "quantity": float(quantity),
        "status": "PENDING",
        "current_price": None,
        "entry_time": None,
        "exit_time": None,
        "exit_price": None,
        "pnl": 0.0,
        "result": None,
        "created_at": time.time()
    }

    PENDING_ORDERS.append(order)

    logging.info(
        "[PENDING] %s %s | Entry=%s TP=%s SL=%s Qty=%s",
        symbol,
        side,
        entry_price,
        tp_price,
        sl_price,
        quantity
    )

    return True, order

def websocket_monitor(stop_event):
    while not stop_event.is_set():

        ws = None

        try:
            symbols = list({
                order["symbol"]
                for order in PENDING_ORDERS
                if order["status"] in ["PENDING", "FILLED"]
            })

            if not symbols:
                time.sleep(1)
                continue

            ws = websocket.create_connection(
                "wss://stream.bybit.com/v5/public/linear",
                timeout=2
            )

            ws.send(json.dumps({
                "op": "subscribe",
                "args": [
                    f"tickers.{symbol}"
                    for symbol in symbols
                ]
            }))

            logging.info(
                "[WS] Monitoring %s pair",
                len(symbols)
            )

            while not stop_event.is_set():

                try:
                    message = json.loads(ws.recv())

                except websocket.WebSocketTimeoutException:
                    continue

                topic = message.get("topic")

                if not topic or not topic.startswith("tickers."):
                    continue

                symbol = topic.replace(
                    "tickers.",
                    ""
                )

                data = message.get("data", {})
                price = data.get("lastPrice")

                if price is None:
                    continue

                price = float(price)

                CURRENT_PRICES[symbol] = price

                for order in PENDING_ORDERS:
                    if order["symbol"] == symbol:
                        order["current_price"] = price

        except Exception as e:

            if not stop_event.is_set():
                logging.error(
                    "[WS ERROR] %s",
                    e
                )

        finally:

            if ws:
                try:
                    ws.close()
                except Exception:
                    pass

        if not stop_event.is_set():
            time.sleep(1)

    logging.info("[WS] Thread stopped")

def start_websocket():
    global WS_THREAD
    global WS_STOP_EVENT

    if WS_THREAD and WS_THREAD.is_alive():
        return

    WS_STOP_EVENT = threading.Event()

    WS_THREAD = threading.Thread(
        target=websocket_monitor,
        args=(WS_STOP_EVENT,),
        daemon=True
    )

    WS_THREAD.start()

    logging.info("[WS] Started")

def stop_websocket():
    global WS_THREAD
    global WS_STOP_EVENT

    if WS_STOP_EVENT:
        WS_STOP_EVENT.set()

    WS_THREAD = None
    WS_STOP_EVENT = None

    logging.info("[WS] Stopped")

async def start_trade_monitor():
    global MONITOR_TASK

    start_websocket()

    if MONITOR_TASK and not MONITOR_TASK.done():
        return

    MONITOR_TASK = asyncio.create_task(
        monitor_trade()
    )

    logging.info("[MONITOR] Started")

async def finish_order(order, result, exit_price):
    symbol = order["symbol"]

    order["exit_price"] = exit_price
    order["exit_time"] = time.time()
    order["result"] = result
    if result in TRADE_STATS:
        TRADE_STATS[result] += 1

    if order["status"] == "FILLED":

        if order["side"] == "Buy":
            order["pnl"] = (
                exit_price - order["entry"]
            ) * order["quantity"]

        else:
            order["pnl"] = (
                order["entry"] - exit_price
            ) * order["quantity"]

    else:
        order["pnl"] = 0.0

    if result == "EXPIRED":

        message = (
            f"⏱️ {symbol} EXPIRED\n"
            f"Price: {exit_price}\n"
            f"TP tersentuh sebelum Entry.\n"
            f"🔒 Ban 24 jam."
        )

    else:

        message = (
            f"🏁 {symbol} {result}\n"
            f"Entry: {order['entry']}\n"
            f"Exit: {exit_price}\n"
            f"Qty: {order['quantity']}\n"
            f"PnL: {order['pnl']:.6f}\n"
            f"🔒 Ban 24 jam."
        )

    await send_log(
        BOT,
        message
    )

    ban_coin(
        symbol,
        24 * 3600
    )

    CURRENT_PRICES.pop(symbol, None)

    if order in PENDING_ORDERS:
        PENDING_ORDERS.remove(order)

    logging.info(
        "[TRADE CLOSED] %s | %s",
        symbol,
        result
    )

async def close_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return

    if not context.args:
        await update.message.reply_text(
            "❌ Format: /close BTCUSDT"
        )
        return

    symbol = context.args[0].upper()

    order = next(
        (
            item
            for item in PENDING_ORDERS
            if item["symbol"] == symbol
        ),
        None
    )

    if order is None:
        await update.message.reply_text(
            f"❌ Tidak ada trade aktif {symbol}."
        )
        return

    price = CURRENT_PRICES.get(
        symbol,
        order["entry"]
    )

    if order["status"] == "FILLED":

        if order["side"] == "Buy":
            order["pnl"] = (
                price - order["entry"]
            ) * order["quantity"]

        else:
            order["pnl"] = (
                order["entry"] - price
            ) * order["quantity"]

    else:
        order["pnl"] = 0.0

    await send_log(
        BOT,
        f"🔒 {symbol} MANUAL CLOSE\n"
        f"Status: {order['status']}\n"
        f"Price: {price}\n"
        f"PnL: {order['pnl']:.6f}\n"
        f"🔒 Ban 8 jam."
    )

    stop_event = STOP_MONITORS.get(symbol)

    if stop_event:
        stop_event.set()

    task = MONITOR_TASKS.get(symbol)

    if task:
        task.cancel()

    MONITOR_TASKS.pop(symbol, None)
    STOP_MONITORS.pop(symbol, None)
    WS_THREADS.pop(symbol, None)
    CURRENT_PRICES.pop(symbol, None)

    if order in PENDING_ORDERS:
        PENDING_ORDERS.remove(order)

    ban_coin(
        symbol,
        8 * 3600
    )

    await update.message.reply_text(
        f"✅ {symbol} berhasil ditutup.\n"
        f"🔒 Banned 8 jam."
    )

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_allowed(update):
        return

    tp = TRADE_STATS["TP"]
    sl = TRADE_STATS["SL"]
    trail = TRADE_STATS["TRAIL"]

    total = tp + sl + trail

    if total == 0:
        await update.message.reply_text(
            "📊 Belum ada statistik trade."
        )
        return

    tp_percent = (
        tp / total
    ) * 100

    sl_percent = (
        sl / total
    ) * 100

    trail_percent = (
        trail / total
    ) * 100

    await update.message.reply_text(
        f"📊 TRADE STATS\n\n"
        f"Total : {total}\n"
        f"TP    : {tp} ({tp_percent:.1f}%)\n"
        f"SL    : {sl} ({sl_percent:.1f}%)\n"
        f"Trail : {trail} ({trail_percent:.1f}%)"
    )

def get_binance_balance():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise Exception(
            "BINANCE_API_KEY atau BINANCE_API_SECRET belum tersedia."
        )

    timestamp = int(time.time() * 1000)

    params = {
        "timestamp": timestamp,
        "recvWindow": 5000
    }

    query_string = "&".join(
        f"{key}={value}"
        for key, value in params.items()
    )

    signature = hmac.new(
        api_secret.encode(),
        query_string.encode(),
        hashlib.sha256
    ).hexdigest()

    params["signature"] = signature

    response = requests.get(
        "https://fapi.binance.com/fapi/v2/balance",
        params=params,
        headers={
            "X-MBX-APIKEY": api_key
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    usdt = next(
        (
            item
            for item in data
            if item["asset"] == "USDT"
        ),
        None
    )

    if usdt is None:
        return 0.0

    return float(usdt["balance"])

def trail_order(symbol, distance):
    symbol = symbol.upper()

    order = next(
        (
            item
            for item in PENDING_ORDERS
            if item["symbol"] == symbol
            and item["status"] == "FILLED"
        ),
        None
    )

    if order is None:
        return False

    price = CURRENT_PRICES.get(symbol)

    if price is None:
        return False

    if distance <= 0:
        return False

    if order["side"] == "Buy":

        new_sl = price - distance

        if new_sl > order["sl"]:
            order["sl"] = new_sl
            order["trail_active"] = True

    else:

        new_sl = price + distance

        if new_sl < order["sl"]:
            order["sl"] = new_sl
            order["trail_active"] = True

    return True
    
def websocket_price(symbol):
    ws = websocket.create_connection(
        "wss://stream.bybit.com/v5/public/linear",
        timeout=10
    )

    subscribe = {
        "op": "subscribe",
        "args": [f"tickers.{symbol}"]
    }

    ws.send(json.dumps(subscribe))

    while True:
        message = json.loads(ws.recv())

        if message.get("topic") == f"tickers.{symbol}":
            price = float(message["data"]["lastPrice"])
            ws.close()
            return price

async def on_start(context=None):
    """Initialize main.py as a handler module for try.py."""
    global MAIN_APP, MAIN_RUNNING, AUTO_TASK

    if MAIN_RUNNING and MAIN_APP is not None:
        return True

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    _register_handlers(app)
    await app.initialize()
    await app.start()

    MAIN_APP = app
    MAIN_RUNNING = True
    AUTO_TASK = None
    await post_init(app)
    logging.info("[MAIN] handler engine started")
    return True


async def handle_update(update_data, context=None):
    """Accept one raw Telegram update from the launcher."""
    if not MAIN_RUNNING or MAIN_APP is None:
        return None

    update = Update.de_json(update_data, MAIN_APP.bot)
    await MAIN_APP.process_update(update)
    return None


async def on_stop(context=None):
    """Stop background tasks, monitor threads, and the PTB application."""
    global MAIN_APP, MAIN_RUNNING, AUTO_TASK, MONITOR_TASK

    AUTO_MODE = False
    try:
        globals()["AUTO_MODE"] = False
    except Exception:
        pass

    if AUTO_TASK is not None and not AUTO_TASK.done():
        AUTO_TASK.cancel()
        try:
            await AUTO_TASK
        except asyncio.CancelledError:
            pass
        except Exception:
            logging.exception("[MAIN] auto task stop gagal")
    AUTO_TASK = None

    if MONITOR_TASK is not None and not MONITOR_TASK.done():
        MONITOR_TASK.cancel()
        try:
            await MONITOR_TASK
        except asyncio.CancelledError:
            pass
        except Exception:
            logging.exception("[MAIN] monitor task stop gagal")
    MONITOR_TASK = None

    try:
        stop_websocket()
    except Exception:
        logging.exception("[MAIN] websocket stop gagal")

    app = MAIN_APP
    MAIN_APP = None
    MAIN_RUNNING = False

    if app is not None:
        try:
            await app.stop()
        finally:
            await app.shutdown()

    logging.info("[MAIN] handler engine stopped")
    return True


def _register_handlers(app):
    """Register the same Telegram handlers used by standalone mode."""
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("close", close_command))
    app.add_handler(CommandHandler("ip", ip_command))
    app.add_handler(CommandHandler("auto", auto_command))
    app.add_handler(CommandHandler("unban", unban_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("trade", trade_command))
    app.add_handler(CommandHandler("pending", pending_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("koin", koin_command))
    app.add_handler(CommandHandler("leverage", leverage_command))
    app.add_handler(CommandHandler("test", test_command))
    app.add_handler(CommandHandler("banned", banned_command))
    app.add_handler(CommandHandler("timer", timer_command))
    app.add_handler(CommandHandler("margin", margin_command))
    app.add_handler(CommandHandler("mode", mode_command))


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logging.getLogger("httpx").setLevel(
        logging.WARNING
    )

    logging.getLogger("httpcore").setLevel(
        logging.WARNING
    )

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    _register_handlers(app)

    app.run_polling()


if __name__ == "__main__":
    main()