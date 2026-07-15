"""
Monitor Engine - Threading, Scan 50 Koin, Pending Order, Posisi Aktif
"""
import time
import threading
import logging
from datetime import datetime, timezone, timedelta

from . import api_client
from . import stats_keeper

log = logging.getLogger(__name__)

# ==================== KONSTANTA DEFAULT ====================
# Nilai ini akan di-override oleh strategy_logic (jika ada)
DEFAULTS = {
    "MIN_CONFIDENCE": 50,
    "MIN_RR": 2.0,
    "MAX_POSITIONS": 20,
    "MONITOR_SLEEP": 10,
    "MONITOR_INTERVAL": 15 * 60,  # 15 menit
    "TRAIL_R_LADDER": [(0.5, 0.15), (1.0, 0.35), (1.5, 0.50), (2.0, 0.65), (2.8, 0.80), (3.5, 0.85)],
    "STRUCT_TRAIL_LB": 2,
    "STRUCT_TRAIL_BUF_PCT": 0.0015,
    "STRUCT_TRAIL_LOOKBACK": 60,
}

# ==================== STATE GLOBAL ====================
positions = {}
positions_lock = threading.Lock()
auto_mode = False
auto_thread = None

# Hook untuk strategy_logic (di-inject oleh bootstrap.py)
strategy_logic = None
WIB = timezone(timedelta(hours=7))

# ==================== FUNGSI PUBLIK ====================
def start_monitor(chat_id):
    """Mulai scanner otomatis (broadcaster)."""
    global auto_mode, auto_thread
    if auto_mode:
        return "⚠️ Monitor sudah berjalan."
    auto_mode = True
    auto_thread = threading.Thread(target=_simulation_loop, args=(chat_id,), daemon=True)
    auto_thread.start()
    return "✅ Monitor dimulai."

def stop_monitor():
    """Hentikan scanner (posisi aktif tetap dipantau)."""
    global auto_mode
    auto_mode = False
    return "⏹️ Scanning dihentikan. Posisi aktif tetap dipantau."

def timeout_position(sym=None):
    """
    Flag timeout untuk posisi tertentu atau semua posisi.
    Posisi akan ditutup dengan PnL riil saat ini di loop berikutnya.
    """
    with positions_lock:
        if sym:
            if sym in positions:
                positions[sym]["timeout_flag"] = True
                return f"⏭ Timeout → {sym}"
            return f"❌ {sym} tidak ditemukan."
        else:
            for s in positions:
                positions[s]["timeout_flag"] = True
            return f"⏭ Timeout semua ({len(positions)}) posisi."

def get_active_positions():
    """Ambil daftar posisi aktif (thread-safe)."""
    with positions_lock:
        return dict(positions)

def is_auto_running():
    return auto_mode

# ==================== FUNGSI INTI (DARI try22.py) ====================
def close_position(sym, result, close_price=None):
    """Tutup posisi, catat statistik, ban koin."""
    with positions_lock:
        pos = positions.pop(sym, None)
    if pos is None:
        return

    sig = pos["signal"]
    entry = pos["entry"]
    sl_p = sig["sl"]
    tp_p = sig["tp"]
    cid = pos["chat_id"]

    stats_keeper.update_stats(
        result, entry=entry, sl_p=sl_p, tp_p=tp_p,
        close_price=close_price, sym=sym,
        decision=sig.get("decision"), entry_time=pos.get("entry_time")
    )
    api_client.ban_coin(sym, f"trade closed ({result})", duration=api_client.BAN_DURATION_TRADE_CLOSED)

    # Update active_trade jika ini yang sedang dipantau
    with positions_lock:
        if not positions:
            pass  # active_trade tidak dipakai lagi

    emoji = {"tp": "🎯", "sl": "🛑", "trail": "🔒"}.get(result, "❓")
    label = {"tp": "TAKE PROFIT", "sl": "STOP LOSS", "trail": "TRAILING STOP"}.get(result, result.upper())
    api_client.tg_send(cid, f"{emoji} <b>{label}</b> — {sym}\n\n{stats_keeper.fmt_stats()}")

def check_tp_sl_order(sym, tp_p, sl_p, is_buy, lookback_min=15):
    """
    Ambil candle M1 dalam N menit terakhir, periksa mana yang kena duluan.
    Return: "tp", "sl", atau None.
    """
    try:
        df = api_client.get_klines(sym, "1m", lookback_min + 2)
        if df is None or df.empty:
            return None
        df = df.tail(lookback_min)
        for _, row in df.iterrows():
            high = row["high"]
            low = row["low"]
            if is_buy:
                if high >= tp_p and low <= sl_p:
                    dist_tp = abs(row["open"] - tp_p)
                    dist_sl = abs(row["open"] - sl_p)
                    return "tp" if dist_tp < dist_sl else "sl"
                elif high >= tp_p:
                    return "tp"
                elif low <= sl_p:
                    return "sl"
            else:
                if low <= tp_p and high >= sl_p:
                    dist_tp = abs(row["open"] - tp_p)
                    dist_sl = abs(row["open"] - sl_p)
                    return "tp" if dist_tp < dist_sl else "sl"
                elif low <= tp_p:
                    return "tp"
                elif high >= sl_p:
                    return "sl"
    except Exception as e:
        log.debug(f"[check_tp_sl] {sym}: {e}")
    return None

def monitor_position(sym, pos):
    """
    Thread per-posisi. Cek harga/TP/SL tiap MONITOR_SLEEP (10 detik).
    Kirim update ke Telegram tiap MONITOR_INTERVAL (15 menit).
    """
    sig = pos["signal"]
    chat_id = pos["chat_id"]
    entry = pos["entry"]
    tp_p = sig["tp"]
    sl_p = sig["sl"]  # SL berjalan
    is_buy = sig["decision"] == "BUY"
    risk0 = abs(entry - sig["sl"])
    locked_r_reached = 0.0
    next_struct_check = 0.0
    next_update_at = time.time() + DEFAULTS["MONITOR_INTERVAL"]

    while True:
        with positions_lock:
            if sym not in positions:
                return

        # Timeout manual
        if pos.get("timeout_flag"):
            pos["timeout_flag"] = False
            price = api_client.get_price(sym) or entry
            pnl_pct = (price - entry) / entry * (1 if is_buy else -1)
            result = "tp" if pnl_pct >= 0 else "sl"
            emoji = "🎯" if result == "tp" else "🛑"
            api_client.tg_send(chat_id,
                f"⏭ <b>Ditutup Manual</b> — {sym} {emoji}\n"
                f"Harga: <code>{price:.6g}</code> | PnL: <b>{pnl_pct*100:+.2f}%</b>")
            close_position(sym, result, close_price=price)
            return

        price = api_client.get_price(sym)
        if price is None:
            time.sleep(DEFAULTS["MONITOR_SLEEP"])
            continue

        # --- Kandidat A: R-ladder ---
        cand_a = None
        if risk0 > 0:
            pnl_r_now = (price - entry) / risk0 * (1 if is_buy else -1)
            best_r = 0.0
            for thr, lock in DEFAULTS["TRAIL_R_LADDER"]:
                if pnl_r_now >= thr:
                    best_r = max(best_r, thr * lock)
            if best_r > locked_r_reached:
                locked_r_reached = best_r
                cand_a = entry + best_r * risk0 * (1 if is_buy else -1)

        # --- Kandidat B: Structure (swing point M15) ---
        cand_b = None
        if time.time() >= next_struct_check:
            next_struct_check = time.time() + 120
            try:
                df_recent = api_client.get_klines(sym, "15m", DEFAULTS["STRUCT_TRAIL_LOOKBACK"])
                if df_recent is not None and len(df_recent) >= DEFAULTS["STRUCT_TRAIL_LB"] * 2 + 1:
                    sh_r, sl_r = strategy_logic.swing_pts(df_recent, lb=DEFAULTS["STRUCT_TRAIL_LB"])
                    if is_buy and sl_r:
                        cand_b = float(df_recent["low"].iloc[sl_r[-1]]) - entry * DEFAULTS["STRUCT_TRAIL_BUF_PCT"]
                    elif not is_buy and sh_r:
                        cand_b = float(df_recent["high"].iloc[sh_r[-1]]) + entry * DEFAULTS["STRUCT_TRAIL_BUF_PCT"]
            except Exception:
                cand_b = None
            pos["_struct_sl_cache"] = cand_b
        else:
            cand_b = pos.get("_struct_sl_cache")

        # SL baru = kandidat paling protektif, hanya boleh mengunci profit
        cands = [c for c in (cand_a, cand_b) if c is not None]
        if cands:
            new_sl = max(cands) if is_buy else min(cands)
            improves = (new_sl > sl_p) if is_buy else (new_sl < sl_p)
            within_tp = (new_sl < tp_p) if is_buy else (new_sl > tp_p)
            if improves and within_tp:
                sl_p = new_sl
                pos["current_sl"] = sl_p
                src = "R-ladder" if (cand_a is not None and new_sl == cand_a) else "structure"
                api_client.tg_send(chat_id,
                    f"🔒 <b>Trailing SL — {sym}</b> ({src})\n"
                    f"SL dikunci ke <code>{sl_p:.6g}</code> "
                    f"({(sl_p-entry)/entry*100*(1 if is_buy else -1):+.2f}%)")

        # --- Cek TP / SL ---
        hit_tp = (price >= tp_p) if is_buy else (price <= tp_p)
        hit_sl = (price <= sl_p) if is_buy else (price >= sl_p)

        if hit_tp or hit_sl:
            order = check_tp_sl_order(sym, tp_p, sl_p, is_buy, lookback_min=3)
            if order is None:
                order = "tp" if hit_tp else "sl"

            if order == "tp":
                pct = abs(tp_p - entry) / entry * 100
                api_client.tg_send(chat_id,
                    f"🎯 <b>TAKE PROFIT</b> — {sym} 🎉\n"
                    f"TP: <code>{tp_p:.6g}</code>\n"
                    f"Profit: +{pct:.2f}%")
                close_position(sym, "tp")
                return
            else:
                # Verifikasi SL via candle M1
                confirmed_sl = False
                try:
                    df_m1 = api_client.get_klines(sym, "1m", 5)
                    if df_m1 is not None and not df_m1.empty:
                        last_closes = df_m1["close"].tail(3)
                        confirmed_sl = any((c <= sl_p) if is_buy else (c >= sl_p) for c in last_closes)
                except Exception:
                    confirmed_sl = hit_sl
                if confirmed_sl:
                    pct_final = (sl_p - entry) / entry * 100 * (1 if is_buy else -1)
                    is_profit_lock = pct_final >= 0
                    result_final = "trail" if is_profit_lock else "sl"
                    label = "TRAILING STOP (profit terkunci)" if is_profit_lock else "STOP LOSS"
                    emoji = "🔒" if is_profit_lock else "🛑"
                    api_client.tg_send(chat_id,
                        f"{emoji} <b>{label}</b> — {sym}\n"
                        f"Harga: <code>{price:.6g}</code> | SL: <code>{sl_p:.6g}</code> | "
                        f"PnL: <b>{pct_final:+.2f}%</b>")
                    close_position(sym, result_final, close_price=sl_p)
                    return
                else:
                    if not pos.get("sweep_notified"):
                        api_client.tg_send(chat_id,
                            f"🔄 <b>Liquidity Sweep — {sym}</b>\n"
                            f"Wick menyentuh SL, candle M1 belum konfirmasi. Lanjut...")
                        pos["sweep_notified"] = True
                    time.sleep(DEFAULTS["MONITOR_SLEEP"])
                    continue

        pos["sweep_notified"] = False

        # Update periodik 15 menit
        if time.time() >= next_update_at:
            pnl_pct = (price - entry) / entry * 100 * (1 if is_buy else -1)
            api_client.tg_send(chat_id,
                f"📊 <b>Update 15m — {sym}</b>\n"
                f"Arah  : {'🟢 BUY' if is_buy else '🔴 SELL'}\n"
                f"Entry : <code>{entry:.6g}</code>\n"
                f"Harga : <code>{price:.6g}</code>\n"
                f"TP    : <code>{tp_p:.6g}</code>\n"
                f"SL    : <code>{sl_p:.6g}</code>\n"
                f"PnL   : <b>{pnl_pct:+.2f}%</b>")
            next_update_at = time.time() + DEFAULTS["MONITOR_INTERVAL"]

        time.sleep(DEFAULTS["MONITOR_SLEEP"])

def _wait_entry(sym, signal, chat_id):
    """Thread tunggu harga ke zona entry (pending order)."""
    entry_target = signal["entry"]
    is_buy = signal["decision"] == "BUY"
    tp_p = signal["tp"]
    sl_p = signal["sl"]
    deadline = time.time() + 8 * 3600
    next_sl_check = 0.0
    last_m15_ts = None

    while time.time() < deadline:
        with positions_lock:
            if sym not in positions:
                return

        price_now = api_client.get_price(sym)
        if price_now is None:
            time.sleep(DEFAULTS["MONITOR_SLEEP"])
            continue

        # TP tersentuh sebelum entry
        tp_hit = (price_now >= tp_p) if is_buy else (price_now <= tp_p)
        if tp_hit:
            with positions_lock:
                positions.pop(sym, None)
            api_client.ban_coin(sym, "TP sebelum entry")
            api_client.tg_send(chat_id, f"⏭ <b>Pending Batal</b> — {sym}\nTP tersentuh sebelum entry. Skip.")
            return

        # SL sebelum entry (butuh konfirmasi candle M15)
        if time.time() >= next_sl_check:
            next_sl_check = time.time() + 60
            try:
                df_chk = api_client.get_klines(sym, "15m", 3)
                if df_chk is not None and len(df_chk) >= 2:
                    closed_row = df_chk.iloc[-2]
                    ts_closed = df_chk.index[-2]
                    if last_m15_ts is None or ts_closed != last_m15_ts:
                        last_m15_ts = ts_closed
                        close_v = float(closed_row["close"])
                        sl_confirmed = (close_v <= sl_p) if is_buy else (close_v >= sl_p)
                        if sl_confirmed:
                            with positions_lock:
                                positions.pop(sym, None)
                            api_client.ban_coin(sym, "SL sebelum entry")
                            api_client.tg_send(chat_id, f"⏭ <b>Pending Batal</b> — {sym}\nCandle M15 close mengonfirmasi SL. Skip.")
                            return
            except Exception:
                pass

        # Entry hit
        entry_hit = (is_buy and price_now <= entry_target * 1.003) or (not is_buy and price_now >= entry_target * 0.997)
        if entry_hit:
            _open_position(sym, signal, price_now, chat_id, "terpicu")
            return

        time.sleep(DEFAULTS["MONITOR_SLEEP"])

    # Expired
    with positions_lock:
        positions.pop(sym, None)
    api_client.ban_coin(sym, "pending expired")
    api_client.tg_send(chat_id, f"⏰ <b>Pending Expired</b> — {sym}\nHarga tidak mencapai zona entry dalam 8 jam. Skip.")

def _open_position(sym, signal, actual_entry, chat_id, mode_label):
    """Upgrade posisi dari pending ke aktif."""
    is_buy = signal["decision"] == "BUY"
    sl_v, tp_v = signal["sl"], signal["tp"]

    # Validasi geometri
    geometry_ok = (sl_v < actual_entry < tp_v) if is_buy else (tp_v < actual_entry < sl_v)
    if not geometry_ok:
        with positions_lock:
            positions.pop(sym, None)
        api_client.ban_coin(sym, "geometri invalid")
        api_client.tg_send(chat_id,
            f"⚠️ <b>Skip {sym}</b> — Geometri SL/TP invalid\n"
            f"Entry: {actual_entry:.6g} | TP: {tp_v:.6g} | SL: {sl_v:.6g}")
        return

    # Verifikasi RR aktual
    sl_dist = abs(actual_entry - sl_v)
    tp_dist = abs(tp_v - actual_entry)
    actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0
    if actual_rr < DEFAULTS["MIN_RR"]:
        with positions_lock:
            positions.pop(sym, None)
        api_client.ban_coin(sym, "RR gagal di entry aktual")
        api_client.tg_send(chat_id,
            f"⚠️ <b>Skip {sym}</b> — RR tidak memenuhi di entry aktual\n"
            f"RR aktual: <b>1:{actual_rr:.2f}</b> (min 1:{DEFAULTS['MIN_RR']})")
        return

    with positions_lock:
        if sym not in positions:
            return
        pos = positions[sym]
        pos["entry"] = actual_entry
        pos["entry_time"] = time.time()
        pos["status"] = "active"
        pos["timeout_flag"] = False
        pos["current_sl"] = sl_v

    api_client.tg_send(chat_id,
        f"⚡ <b>ENTRY {mode_label.upper()}</b> — {sym}\n"
        f"Entry: <code>{actual_entry:.6g}</code>\n"
        f"TP: <code>{tp_v:.6g}</code> | SL: <code>{sl_v:.6g}</code>\n"
        f"RR: <b>1:{actual_rr:.2f}</b> | 📡 Dipantau...")

    threading.Thread(target=monitor_position, args=(sym, pos), daemon=True).start()

# ==================== SIMULATION LOOP ====================
def _simulation_loop(chat_id):
    """Loop utama broadcaster."""
    api_client.tg_send(chat_id,
        "🤖 <b>SMC Signal Broadcaster dimulai!</b>\n\n"
        f"• Scan 50 koin → catat sinyal → pantau tiap 15 menit\n"
        f"• Maks {DEFAULTS['MAX_POSITIONS']} posisi bersamaan\n"
        "• Posisi ditutup hanya saat TP atau SL\n\n"
        "/stop untuk berhenti | /timeout SYMBOL untuk tutup paksa")

    scanning = False
    scan_lock = threading.Lock()

    def _do_scan():
        nonlocal scanning
        try:
            # Panggil full_analyze dari strategy_logic
            if strategy_logic is None:
                log.error("[scan] strategy_logic belum di-load!")
                return

            # Ambil 50 koin (minus banned + posisi aktif)
            with positions_lock:
                active_syms = set(positions.keys())
            banned = api_client.get_banned_coins()[1]  # (scan_counter, set)
            exclude = active_syms | banned

            symbols = api_client.get_top_coins(exclude_syms=exclude)
            if not symbols:
                return

            # Scan sampai dapat sinyal atau habis
            for sym in symbols:
                if not auto_mode:
                    return
                signal = strategy_logic.full_analyze(sym)
                if signal is None:
                    continue
                # Cek confidence
                if signal.get("confidence", 0) < DEFAULTS["MIN_CONFIDENCE"]:
                    continue

                # Cek slot
                with positions_lock:
                    if sym in positions:
                        continue
                    if len(positions) >= DEFAULTS["MAX_POSITIONS"]:
                        return

                entry_target = signal["entry"]
                current = signal["price"]
                is_buy = signal["decision"] == "BUY"
                tp_p = signal["tp"]
                entry_label = signal.get("entry_label", "market")

                already_at_entry = (is_buy and current <= entry_target * 1.002) or (not is_buy and current >= entry_target * 0.998)

                if already_at_entry or entry_label == "market":
                    actual_entry = api_client.get_price(sym) or current
                    with positions_lock:
                        if sym in positions:
                            return
                        if len(positions) >= DEFAULTS["MAX_POSITIONS"]:
                            return
                        positions[sym] = {
                            "signal": signal,
                            "entry": entry_target,
                            "chat_id": chat_id,
                            "entry_time": None,
                            "timeout_flag": False,
                            "status": "pending",
                        }
                    _open_position(sym, signal, actual_entry, chat_id, "langsung")
                else:
                    with positions_lock:
                        if sym in positions:
                            return
                        if len(positions) >= DEFAULTS["MAX_POSITIONS"]:
                            return
                        positions[sym] = {
                            "signal": signal,
                            "entry": entry_target,
                            "chat_id": chat_id,
                            "entry_time": None,
                            "timeout_flag": False,
                            "status": "pending",
                        }
                    dist_pct = abs(entry_target - current) / current * 100
                    api_client.tg_send(chat_id,
                        f"🎯 <b>PENDING ORDER</b> — {sym}\n\n"
                        f"{_fmt_signal_msg(signal)}\n\n"
                        f"⏳ Menunggu harga ke zona entry\n"
                        f"Harga kini : <code>{current:.6g}</code>\n"
                        f"Entry zone : <code>{entry_target:.6g}</code> ({entry_label})\n"
                        f"Jarak      : {dist_pct:.2f}%")
                    threading.Thread(target=_wait_entry, args=(sym, signal, chat_id), daemon=True).start()
                return  # satu sinyal per scan

        finally:
            with scan_lock:
                scanning = False

    while auto_mode:
        with positions_lock:
            n_pos = len(positions)
        if n_pos >= DEFAULTS["MAX_POSITIONS"]:
            time.sleep(5)
            continue

        with scan_lock:
            if scanning:
                time.sleep(5)
                continue
            scanning = True

        threading.Thread(target=_do_scan, daemon=True).start()
        time.sleep(5)

    api_client.tg_send(chat_id, "⏹ <b>Scanning dihentikan.</b>\n\n" + stats_keeper.fmt_stats())

# ==================== HELPERS ====================
def _fmt_signal_msg(sig):
    """Format sinyal untuk Telegram (copy dari try22.py)."""
    em = "🟢" if sig["decision"] == "BUY" else "🔴"
    bar = "█" * (sig["confidence"] // 10) + "░" * (10 - sig["confidence"] // 10)
    dir_label = "BULLISH" if sig["original_dir"] == "bull" else "BEARISH"
    d1_em = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(sig.get("d1_bias", "neutral"), "➡️")

    triggers = []
    ch15, ch1, fr = sig.get("choch_m15", {}), sig.get("choch_h1", {}), sig.get("failed_retest", {})
    if ch1.get("bearish_choch"): triggers.append("CHoCH Bear H1")
    if ch1.get("bullish_choch"): triggers.append("CHoCH Bull H1")
    if ch15.get("bearish_choch"): triggers.append("CHoCH Bear M15")
    if ch15.get("bullish_choch"): triggers.append("CHoCH Bull M15")
    if fr.get("failed_retest_sell"): triggers.append("Failed Retest Sell")
    if fr.get("failed_retest_buy"): triggers.append("Failed Retest Buy")

    entry_label = sig.get("entry_label", "market")
    price_now, entry_zone = sig.get("price", sig["entry"]), sig["entry"]
    entry_str = (
        f"📍 Harga: <code>{price_now:.6g}</code> → 🎯 Entry: <code>{entry_zone:.6g}</code> ({entry_label})"
        if abs(price_now - entry_zone) / max(price_now, 0.0001) > 0.002
        else f"💰 Entry: <code>{entry_zone:.6g}</code> ({entry_label})"
    )

    return (
        f"📡 <b>{sig['symbol']}</b> — {dir_label} ({sig['confidence']}% {bar})\n"
        f"{em} <b>{sig['decision']}</b>\n"
        f"{entry_str}\n"
        f"✅ TP: <code>{sig['tp']:.6g}</code>  🛑 SL: <code>{sig['sl']:.6g}</code>  "
        f"⚖️ RR 1:{sig['rr']}\n"
        f"RSI {sig['rsi']} | H1 {sig['struct_h1'].upper()} | D1 {d1_em}{sig.get('d1_bias','neutral').upper()}\n"
        f"🎯 {' | '.join(triggers) if triggers else '—'}\n"
        f"📝 {sig['tp_sl_reason']}"
    )