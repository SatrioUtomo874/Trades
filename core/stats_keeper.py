"""
Stats Keeper - Manajemen Saldo Simulasi, PnL, dan Statistik
"""
import time
import threading
from collections import deque

# ==================== KONSTANTA ====================
STARTING_BALANCE = 10.0          # Modal awal simulasi
POSITION_SIZE_PCT = 100.0        # 100% saldo per trade (1x leverage)

# ==================== STATE GLOBAL ====================
stat_lock = threading.Lock()
stats = {
    "tp": 0,                     # Jumlah take profit
    "sl": 0,                     # Jumlah stop loss (loss)
    "trail": 0,                  # Jumlah trailing stop (profit terkunci)
    "total": 0,                  # Total trade selesai
    "balance": STARTING_BALANCE, # Saldo terkini
    "pnl_history": deque(maxlen=20),  # 20 trade terakhir
}

# ==================== FUNGSI INTI ====================
def update_stats(result, entry=None, sl_p=None, tp_p=None, close_price=None,
                 sym=None, decision=None, entry_time=None):
    """
    Catat hasil trade dan update saldo simulasi.

    Args:
        result (str): "tp" | "sl" | "trail"
        entry (float): Harga entry
        sl_p (float): Harga stop loss (saat ini / trailing)
        tp_p (float): Harga take profit
        close_price (float): Harga tutup aktual (untuk trailing)
        sym (str): Symbol koin
        decision (str): "BUY" atau "SELL"
        entry_time (float): Timestamp entry
    """
    with stat_lock:
        stats["total"] += 1
        if result in ("tp", "sl", "trail"):
            stats[result] = stats.get(result, 0) + 1

        # Jika entry atau tp_p None, tidak bisa hitung PnL
        if not entry or tp_p is None:
            return

        balance = stats["balance"]
        position_usd = round(balance * POSITION_SIZE_PCT / 100.0, 6)

        # Arah posisi: BUY -> profit jika harga naik, SELL -> profit jika harga turun
        direction_sign = 1 if tp_p > entry else -1

        # Tentukan harga referensi untuk hitung PnL
        if close_price is not None:
            ref_price = close_price
        elif result == "tp":
            ref_price = tp_p
        elif result == "sl" and sl_p is not None:
            ref_price = sl_p
        else:
            return

        # Persentase PnL
        pnl_pct = (ref_price - entry) / entry * direction_sign
        pnl_usd = round(position_usd * pnl_pct, 4)
        pct = round(pnl_pct * 100, 3)

        # Update saldo
        stats["balance"] = round(balance + pnl_usd, 4)

        # Simpan ke history
        stats["pnl_history"].append({
            "result": result,
            "pct": pct,
            "pnl_usd": pnl_usd,
            "balance_after": stats["balance"],
            "symbol": sym,
            "decision": decision,
            "entry_time": entry_time,
            "exit_time": time.time(),
            "entry": entry,
            "tp": tp_p,
            "sl": sl_p,
            "exit_price": ref_price,
        })

def reset_stats():
    """Reset semua statistik ke awal."""
    with stat_lock:
        stats["tp"] = 0
        stats["sl"] = 0
        stats["trail"] = 0
        stats["total"] = 0
        stats["balance"] = STARTING_BALANCE
        stats["pnl_history"] = deque(maxlen=20)

# ==================== FORMAT UNTUK TELEGRAM ====================
def fmt_stats():
    """Format statistik ringkas untuk Telegram."""
    with stat_lock:
        t = stats["total"]
        tp = stats["tp"]
        sl = stats["sl"]
        trail = stats.get("trail", 0)
        bal = stats["balance"]
        hist = list(stats["pnl_history"])

    if t == 0:
        return (f"📊 <b>Statistik</b>\n"
                f"Belum ada trade. Modal: ${STARTING_BALANCE:.2f}")

    wins = tp + trail
    wr = wins / (wins + sl) * 100 if (wins + sl) > 0 else 0
    pnl = round(bal - STARTING_BALANCE, 4)
    pnl_pct = round(pnl / STARTING_BALANCE * 100, 2)
    sgn = "+" if pnl >= 0 else ""

    hist_str = "\n".join(
        f"  {'✅' if h['result'] in ('tp','trail') else '❌'} "
        f"{'+' if h['pnl_usd']>=0 else ''}{h['pct']:.2f}% "
        f"→ ${h['balance_after']:.4f}"
        for h in reversed(hist[-5:])
    ) or "  (belum ada)"

    return (
        f"📊 <b>Statistik</b> — {t} trade | "
        f"TP {tp} SL {sl} Trail {trail}\n"
        f"Win Rate: <b>{wr:.1f}%</b>\n\n"
        f"Modal: ${STARTING_BALANCE:.2f} → "
        f"Saldo: <b>${bal:.4f}</b> ({sgn}{pnl_pct:.2f}%)\n\n"
        f"5 terakhir:\n{hist_str}"
    )

def fmt_backtest():
    """Format 20 trade terakhir untuk Telegram (evaluasi detail)."""
    with stat_lock:
        hist = list(stats["pnl_history"])

    if not hist:
        return "📋 <b>Backtest</b>\nBelum ada trade."

    lines = []
    for h in reversed(hist):
        em = "✅" if h["result"] in ("tp", "trail") else "❌"
        sym = h.get("symbol") or "?"
        dec = h.get("decision") or "?"
        res = h["result"].upper()
        pct = h["pct"]
        entry_v = h.get("entry")
        tp_v = h.get("tp")
        sl_v = h.get("sl")
        exit_v = h.get("exit_price")

        # Format level
        if entry_v is not None and tp_v is not None and sl_v is not None:
            levels = (f"Entry: <code>{entry_v:.6g}</code> | "
                      f"TP: <code>{tp_v:.6g}</code> | "
                      f"SL: <code>{sl_v:.6g}</code>\n"
                      f"Exit: <code>{exit_v:.6g}</code>\n")
        else:
            levels = ""

        lines.append(
            f"{em} <b>{sym}</b> {dec} | {res} {pct:+.2f}%\n"
            f"{levels}"
        )

    return f"📋 <b>Backtest ({len(hist)} trade terakhir)</b>\n\n" + "\n".join(lines)

def get_balance():
    """Ambil saldo terkini (thread-safe)."""
    with stat_lock:
        return stats["balance"]

def get_total_trades():
    """Ambil total trade (thread-safe)."""
    with stat_lock:
        return stats["total"]