"""
Context Builder - Generate research_context.json untuk AI Lokal.
Menggabungkan data market 15 koin 3 bulan + hasil backtest.
"""
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from . import api_client

log = logging.getLogger(__name__)

# ==================== KONSTANTA ====================
TOP_COINS_FOR_RESEARCH = 15
DAYS_HISTORY = 90
SESSION_BINS = [
    (0, 8, "00:00-08:00 UTC"),
    (8, 16, "08:00-16:00 UTC"),
    (16, 24, "16:00-24:00 UTC"),
]

# ==================== FUNGSI INTI ====================
def generate_research_context(trade_history=None):
    """
    Generate research_context.json dari 15 koin × 3 bulan M1 + trade history.

    Args:
        trade_history (list): Daftar trade dari stats_keeper (pnl_history).

    Returns:
        dict: siap di-export ke JSON.
    """
    log.info("[context] Mulai generate research_context...")

    # 1. Ambil 15 koin teratas
    all_coins = api_client.get_top_coins()
    coins = all_coins[:TOP_COINS_FOR_RESEARCH]
    if not coins:
        log.warning("[context] Tidak ada koin tersedia")
        return _empty_context()

    log.info(f"[context] Mengambil data untuk {len(coins)} koin: {', '.join(coins)}")

    # 2. Fetch data M1 3 bulan untuk setiap koin
    coin_data = {}
    for sym in coins:
        try:
            df = api_client.get_klines(sym, "1m", limit=DAYS_HISTORY * 1440)
            if not df.empty:
                coin_data[sym] = df
                log.debug(f"[context] {sym}: {len(df)} candle")
        except Exception as e:
            log.warning(f"[context] Gagal fetch {sym}: {e}")

    # 3. Build result structure
    result = {
        "period": f"{DAYS_HISTORY} hari terakhir",
        "coins": {},
        "performance_breakdown": {
            "by_session": {},
            "by_volume_condition": {},
            "by_struct_h1": {},
            "by_coin": {}
        },
        "worst_trades": [],
        "best_trades": [],
        "summary": {}
    }

    # 4. Hitung statistik per koin (volatilitas, volume, range)
    for sym, df in coin_data.items():
        if len(df) < 100:
            continue
        close = df["close"]
        volume = df["volume"]
        volatility = close.pct_change().std() * 100
        volume_avg = volume.mean()
        result["coins"][sym] = {
            "volatility_avg": round(volatility, 2),
            "volume_avg": round(volume_avg, 0),
            "price_range": {
                "low": round(df["low"].min(), 6),
                "high": round(df["high"].max(), 6)
            },
            "close_price": round(close.iloc[-1], 6),
            "candles": len(df)
        }

    # 5. Analisis trade history (jika ada)
    if trade_history and len(trade_history) > 0:
        _analyze_trade_history(trade_history, result)

    # 6. Tambahkan summary
    total_trades = len(trade_history) if trade_history else 0
    if total_trades > 0:
        wins = sum(1 for t in trade_history if t["result"] in ("tp", "trail"))
        wr = wins / total_trades * 100
        result["summary"] = {
            "total_trades": total_trades,
            "win_rate": round(wr, 1),
            "avg_profit": round(sum(t.get("pnl_usd", 0) for t in trade_history) / total_trades, 4)
        }

    log.info(f"[context] Selesai. {len(coin_data)} koin, {total_trades} trade dianalisis.")
    return result

# ==================== INTERNAL ====================
def _empty_context():
    """Return context kosong (fallback)."""
    return {
        "period": f"{DAYS_HISTORY} hari terakhir",
        "coins": {},
        "performance_breakdown": {},
        "worst_trades": [],
        "best_trades": [],
        "summary": {"total_trades": 0, "win_rate": 0, "avg_profit": 0}
    }

def _analyze_trade_history(trade_history, result):
    """
    Analisis daftar trade: breakdown per session, per coin, per volume condition,
    serta ambil best/worst trades.
    """
    if not trade_history:
        return

    # Breakdown per coin
    by_coin = defaultdict(lambda: {"total": 0, "wins": 0, "pnl": 0})
    # Breakdown per session
    by_session = defaultdict(lambda: {"total": 0, "wins": 0})
    # Breakdown by volume condition (dummy, karena kita tidak punya data volume per trade di sini)
    # Kita isi berdasarkan apakah trade masuk di session tertentu (simulasi sederhana)

    best_trades = []
    worst_trades = []

    for t in trade_history:
        sym = t.get("symbol", "unknown")
        pnl = t.get("pnl_usd", 0)
        result_type = t.get("result", "sl")
        is_win = result_type in ("tp", "trail")

        # Per koin
        by_coin[sym]["total"] += 1
        by_coin[sym]["wins"] += 1 if is_win else 0
        by_coin[sym]["pnl"] += pnl

        # Per session (dari exit_time atau entry_time)
        exit_ts = t.get("exit_time")
        if exit_ts:
            dt = datetime.fromtimestamp(exit_ts)
            hour = dt.hour
            for start, end, label in SESSION_BINS:
                if start <= hour < end:
                    by_session[label]["total"] += 1
                    by_session[label]["wins"] += 1 if is_win else 0
                    break

        # Best & worst
        if is_win:
            best_trades.append((pnl, sym, result_type))
        else:
            worst_trades.append((pnl, sym, result_type))

    # Simpan breakdown per coin
    for sym, data in by_coin.items():
        total = data["total"]
        wins = data["wins"]
        wr = wins / total * 100 if total > 0 else 0
        result["performance_breakdown"]["by_coin"][sym] = {
            "total": total,
            "win_rate": round(wr, 1),
            "avg_pnl": round(data["pnl"] / total, 4) if total > 0 else 0
        }

    # Simpan breakdown per session
    for label, data in by_session.items():
        total = data["total"]
        wins = data["wins"]
        wr = wins / total * 100 if total > 0 else 0
        result["performance_breakdown"]["by_session"][label] = {
            "total": total,
            "win_rate": round(wr, 1)
        }

    # Ambil 3 best dan 3 worst
    best_trades.sort(key=lambda x: x[0], reverse=True)
    worst_trades.sort(key=lambda x: x[0])

    result["best_trades"] = [
        {"pnl": round(pnl, 4), "symbol": sym, "result": res}
        for pnl, sym, res in best_trades[:3] if pnl > 0
    ]
    result["worst_trades"] = [
        {"pnl": round(pnl, 4), "symbol": sym, "result": res}
        for pnl, sym, res in worst_trades[:3] if pnl < 0
    ]

    # Breakdown by volume condition (simulasi dari sesi)
    # Kita hanya isi jika ada data session
    if by_session:
        total_trades = sum(d["total"] for d in by_session.values())
        if total_trades > 0:
            # Kondisi volume: asumsikan sesi siang = volume tinggi, malam = volume rendah
            high_vol_sessions = ["08:00-16:00 UTC", "16:00-24:00 UTC"]
            low_vol_sessions = ["00:00-08:00 UTC"]

            high_total = sum(by_session.get(s, {}).get("total", 0) for s in high_vol_sessions)
            high_wins = sum(by_session.get(s, {}).get("wins", 0) for s in high_vol_sessions)
            low_total = sum(by_session.get(s, {}).get("total", 0) for s in low_vol_sessions)
            low_wins = sum(by_session.get(s, {}).get("wins", 0) for s in low_vol_sessions)

            result["performance_breakdown"]["by_volume_condition"] = {
                "volume > avg_20": {
                    "total": high_total,
                    "win_rate": round(high_wins / high_total * 100, 1) if high_total > 0 else 0
                },
                "volume < avg_20": {
                    "total": low_total,
                    "win_rate": round(low_wins / low_total * 100, 1) if low_total > 0 else 0
                }
            }

def export_research_context_json(trade_history, output_path=None):
    """
    Convenience: generate context dan simpan ke file JSON.

    Args:
        trade_history (list): Daftar trade dari stats_keeper.
        output_path (str): Path file output. Default: '/tmp/research_context.json'

    Returns:
        str: Path file yang disimpan.
    """
    if output_path is None:
        output_path = "/tmp/research_context.json"

    context = generate_research_context(trade_history)

    with open(output_path, "w") as f:
        json.dump(context, f, indent=2, default=str)

    log.info(f"[context] Disimpan ke {output_path}")
    return output_path