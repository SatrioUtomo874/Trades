"""
Telegram Bot - Flask App + Semua Handler
"""
import os
import sys
import time
import json
import base64
import logging
import threading
import importlib
from datetime import datetime, timezone, timedelta
from flask import Flask

import requests
import pandas as pd

from . import api_client
from . import stats_keeper
from . import monitor_engine
from . import context_builder

log = logging.getLogger(__name__)

# ==================== KONSTANTA ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME")  # format: "username/repo"

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN tidak ditemukan di environment")

# ==================== FLASK APP ====================
app = Flask(__name__)

@app.route("/")
def index():
    return "Trading Research Engine v2 - Aktif", 200

@app.route("/health")
def health():
    return "OK", 200

def run_flask():
    """Jalankan Flask di thread terpisah."""
    port = int(os.environ.get("PORT", 8080))
    log.info(f"[flask] binding port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ==================== TELEGRAM HELPERS ====================
def tg_send(chat_id, text, parse_mode="HTML"):
    """Kirim pesan ke Telegram."""
    if not chat_id or not TELEGRAM_TOKEN:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            timeout=10
        )
    except Exception as e:
        log.error(f"[TG send] {e}")

def tg_send_document(chat_id, file_path, caption=""):
    """Kirim file ke Telegram."""
    if not chat_id or not TELEGRAM_TOKEN:
        return
    try:
        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {"chat_id": chat_id, "caption": caption}
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
                files=files, data=data, timeout=30
            )
    except Exception as e:
        log.error(f"[TG doc] {e}")

def tg_updates(offset=None):
    """Ambil update dari Telegram."""
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"timeout": 8, "offset": offset}, timeout=12
        )
        d = r.json()
        return d.get("result", []) if d.get("ok") else []
    except Exception:
        return []

# ==================== GITHUB API HELPERS ====================
def _commit_to_github(content, path, commit_msg="Update via Telegram"):
    """Commit file ke GitHub menggunakan API."""
    if not GITHUB_TOKEN or not REPO_NAME:
        raise ValueError("GITHUB_TOKEN atau REPO_NAME tidak diset.")
    
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    # 1. Get current SHA (untuk update)
    sha = None
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception:
        pass
    
    # 2. Commit baru
    data = {
        "message": commit_msg,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": "main"
    }
    if sha:
        data["sha"] = sha
    
    resp = requests.put(url, headers=headers, json=data)
    if resp.status_code not in (200, 201):
        raise ValueError(f"GitHub commit gagal: {resp.status_code} {resp.text}")
    
    return True

def _reload_strategy_logic():
    """Reload strategy_logic dari disk (setelah commit GitHub)."""
    try:
        # Path absolut ke strategy_logic.py di root repo
        # (direktori ini adalah root karena bootstrap.py di sini)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import strategy_logic
        importlib.reload(strategy_logic)
        # Inject ke monitor_engine
        monitor_engine.strategy_logic = strategy_logic
        log.info("[reload] strategy_logic berhasil di-reload!")
        return True
    except Exception as e:
        log.error(f"[reload] Gagal reload strategy_logic: {e}")
        return False

# ==================== GENERATE CSV ====================
def _generate_statistics_csv(chat_id):
    """Generate statistics.csv dari stats_keeper."""
    with stats_keeper.stat_lock:
        s = stats_keeper.stats
        data = {
            "total_trades": [s["total"]],
            "tp": [s["tp"]],
            "sl": [s["sl"]],
            "trail": [s.get("trail", 0)],
            "balance": [s["balance"]],
            "starting_balance": [stats_keeper.STARTING_BALANCE]
        }
        df = pd.DataFrame(data)
        path = "/tmp/statistics.csv"
        df.to_csv(path, index=False)
        return path

def _generate_trade_csv(chat_id):
    """Generate trade.csv dari pnl_history."""
    with stats_keeper.stat_lock:
        hist = list(stats_keeper.stats["pnl_history"])
    
    if not hist:
        # Buat CSV kosong dengan header
        df = pd.DataFrame(columns=["symbol", "decision", "result", "pnl_pct", "pnl_usd", "entry", "tp", "sl", "exit_price"])
        path = "/tmp/trade.csv"
        df.to_csv(path, index=False)
        return path
    
    rows = []
    for h in hist:
        rows.append({
            "symbol": h.get("symbol", ""),
            "decision": h.get("decision", ""),
            "result": h["result"],
            "pnl_pct": h["pct"],
            "pnl_usd": h["pnl_usd"],
            "entry": h.get("entry", 0),
            "tp": h.get("tp", 0),
            "sl": h.get("sl", 0),
            "exit_price": h.get("exit_price", 0),
            "entry_time": datetime.fromtimestamp(h.get("entry_time", 0)).strftime("%Y-%m-%d %H:%M:%S") if h.get("entry_time") else "",
            "exit_time": datetime.fromtimestamp(h.get("exit_time", 0)).strftime("%Y-%m-%d %H:%M:%S") if h.get("exit_time") else "",
        })
    df = pd.DataFrame(rows)
    path = "/tmp/trade.csv"
    df.to_csv(path, index=False)
    return path

# ==================== HANDLER ANALYZE (BACKGROUND) ====================
def _run_analyze(chat_id):
    """Background thread untuk /analyze."""
    try:
        tg_send(chat_id, "🔄 Memulai riset historis untuk 15 koin (3 bulan)...\nIni bisa memakan waktu 3-5 menit.")

        # 1. Generate research_context
        with stats_keeper.stat_lock:
            trade_hist = list(stats_keeper.stats["pnl_history"])
        
        context = context_builder.generate_research_context(trade_hist)
        context_path = "/tmp/research_context.json"
        with open(context_path, "w") as f:
            json.dump(context, f, indent=2, default=str)
        
        # 2. Generate CSV
        stats_csv = _generate_statistics_csv(chat_id)
        trade_csv = _generate_trade_csv(chat_id)
        
        # 3. Kirim ke Telegram
        tg_send(chat_id, "✅ Riset selesai! Mengirim file...")
        
        tg_send_document(chat_id, stats_csv, caption="📊 statistics.csv")
        tg_send_document(chat_id, trade_csv, caption="📋 trade.csv")
        tg_send_document(chat_id, context_path, caption="🧠 research_context.json")
        
        # 4. Ringkasan singkat
        summary = context.get("summary", {})
        total = summary.get("total_trades", 0)
        wr = summary.get("win_rate", 0)
        avg_pnl = summary.get("avg_profit", 0)
        tg_send(chat_id,
            f"📊 <b>Ringkasan Riset</b>\n"
            f"Total trade: {total}\n"
            f"Win Rate: {wr}%\n"
            f"Avg PnL per trade: ${avg_pnl:.4f}\n\n"
            f"File sudah dikirim. Jalankan researcher.py di laptop untuk analisis lebih lanjut."
        )
    except Exception as e:
        log.error(f"[analyze] Error: {e}")
        tg_send(chat_id, f"❌ Error saat menjalankan riset:\n<code>{str(e)[:200]}</code>")

# ==================== BOT LOOP ====================
def bot_loop():
    """Loop utama bot Telegram."""
    # Pastikan active_chat_id ke ALLOWED_USER_ID sejak awal
    active_chat_id = ALLOWED_USER_ID

    # Kirim notifikasi siap
    if ALLOWED_USER_ID:
        tg_send(ALLOWED_USER_ID,
            "✅ <b>Bot Siap</b>\n"
            "Trading Research Engine v2 aktif.\n"
            "Ketik /start untuk menu.")

    offset = None

    # GREETING
    GREETING = (
        "👋 <b>SMC Signal Broadcaster + Research Engine</b>\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "/start               — Menu ini\n"
        "/auto                — Mulai broadcaster (scan 50 koin)\n"
        "/stop                — Hentikan scanning (posisi aktif tetap dipantau)\n"
        "/trade               — Lihat semua posisi aktif\n"
        "/analyze             — Jalankan riset historis 15 koin (3 bulan) → export CSV+JSON\n"
        "/ganti               — Upload strategy_logic.py baru (kaset AI)\n"
        "/timeout SYMBOL      — Tutup paksa posisi tertentu\n"
        "/timeout             — Tutup paksa semua posisi\n"
        "/stats               — Statistik + saldo\n"
        "/backtest            — 20 trade terakhir (evaluasi)\n"
        "/banned              — Daftar koin ban\n"
        "/resetban            — Hapus semua ban\n"
        "/resetbalance        — Reset saldo ke $10\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "⚠️ <i>Simulasi saja — bukan saran finansial.</i>"
    )

    while True:
        try:
            updates = tg_updates(offset)
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message", {})
                uid = msg.get("from", {}).get("id")
                chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "").strip().lower()
                doc = msg.get("document")

                if not uid or not chat_id:
                    continue
                if uid != ALLOWED_USER_ID:
                    tg_send(chat_id, "⛔ Akses ditolak.")
                    continue

                active_chat_id = chat_id

                # --- COMMANDS ---
                if text == "/start":
                    tg_send(chat_id, GREETING)

                elif text == "/auto":
                    resp = monitor_engine.start_monitor(chat_id)
                    tg_send(chat_id, resp)

                elif text == "/stop":
                    resp = monitor_engine.stop_monitor()
                    tg_send(chat_id, resp)

                elif text == "/stats":
                    tg_send(chat_id, stats_keeper.fmt_stats())

                elif text == "/backtest":
                    tg_send(chat_id, stats_keeper.fmt_backtest())

                elif text == "/trade":
                    pos_list = monitor_engine.get_active_positions()
                    if not pos_list:
                        tg_send(chat_id, "ℹ️ Tidak ada posisi aktif.")
                    else:
                        lines = [f"📡 <b>Posisi Aktif ({len(pos_list)}/{monitor_engine.DEFAULTS['MAX_POSITIONS']})</b>\n"]
                        for s, p in pos_list.items():
                            sig = p["signal"]
                            is_buy = sig["decision"] == "BUY"
                            em = "🟢" if is_buy else "🔴"
                            status = p.get("status", "active")
                            pr = api_client.get_price(s) or p["entry"]
                            if status == "pending":
                                dist_pct = abs(p["entry"] - pr) / pr * 100
                                lines.append(
                                    f"\n⏳ <b>{s}</b> — PENDING\n"
                                    f"{em} {sig['decision']} | Entry zone: <code>{p['entry']:.6g}</code>\n"
                                    f"Harga: <code>{pr:.6g}</code> | Jarak: {dist_pct:.2f}%\n"
                                    f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{sig['sl']:.6g}</code>"
                                )
                            else:
                                pnl = (pr - p["entry"]) / p["entry"] * 100 * (1 if is_buy else -1)
                                cur_sl = p.get("current_sl", sig["sl"])
                                trail_note = " 🔒trailing" if cur_sl != sig["sl"] else ""
                                lines.append(
                                    f"\n{em} <b>{s}</b> — AKTIF\n"
                                    f"Entry: <code>{p['entry']:.6g}</code> | Harga: <code>{pr:.6g}</code>\n"
                                    f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{cur_sl:.6g}</code>{trail_note}\n"
                                    f"PnL: <b>{pnl:+.2f}%</b>"
                                )
                        tg_send(chat_id, "\n".join(lines))

                elif text.startswith("/timeout"):
                    parts = text.split()
                    target = parts[1].upper() if len(parts) > 1 else None
                    resp = monitor_engine.timeout_position(target)
                    tg_send(chat_id, resp)

                elif text == "/banned":
                    banned = api_client.get_banned_coins()
                    if banned:
                        lines = [f"🚫 <b>Banned ({len(banned)}):</b>"]
                        for sym, (banned_at, dur) in banned.items():
                            remaining = max(0, dur - (api_client.scan_counter - banned_at))
                            lines.append(f"• {sym} (unban dalam {remaining} scan)")
                        tg_send(chat_id, "\n".join(lines))
                    else:
                        tg_send(chat_id, "✅ Belum ada ban.")

                elif text == "/resetban":
                    with api_client.ban_lock:
                        api_client.banned_coins.clear()
                    tg_send(chat_id, "✅ Ban direset.")

                elif text == "/resetbalance":
                    stats_keeper.reset_stats()
                    tg_send(chat_id, f"✅ Saldo & statistik direset ke ${stats_keeper.STARTING_BALANCE:.2f}")

                elif text == "/analyze":
                    # Jalankan di background thread agar tidak block loop
                    threading.Thread(target=_run_analyze, args=(chat_id,), daemon=True).start()
                    tg_send(chat_id, "⏳ Riset dimulai di background. Anda akan menerima file dalam beberapa menit.")

                elif text == "/ganti" and doc:
                    # Handler untuk upload file (via document)
                    # Note: untuk command /ganti, user harus kirim file dengan caption /ganti
                    # atau kita handle di bagian file handler di bawah.
                    pass  # Ditangani di bagian document handler

                elif text.startswith("/ganti"):
                    tg_send(chat_id, "📤 Kirim file strategy_logic.py dengan caption /ganti (atau kirim sebagai file).")

                elif text.startswith("/max"):
                    parts = text.split()
                    if len(parts) == 2:
                        try:
                            n = int(parts[1])
                            if 1 <= n <= 50:
                                old = monitor_engine.DEFAULTS["MAX_POSITIONS"]
                                monitor_engine.DEFAULTS["MAX_POSITIONS"] = n
                                tg_send(chat_id, f"✅ Max posisi diubah: {old} → {n}")
                            else:
                                tg_send(chat_id, "❌ Nilai harus 1-50.")
                        except ValueError:
                            tg_send(chat_id, "❌ Format salah. Contoh: /max 10")
                    else:
                        tg_send(chat_id,
                            f"⚙️ Max posisi saat ini: <b>{monitor_engine.DEFAULTS['MAX_POSITIONS']}</b>\n"
                            "Ubah: /max 10")

                elif text.startswith("/confidence_min"):
                    parts = text.split()
                    if len(parts) == 2:
                        try:
                            n = int(parts[1])
                            if 0 <= n <= 99:
                                old = monitor_engine.DEFAULTS["MIN_CONFIDENCE"]
                                monitor_engine.DEFAULTS["MIN_CONFIDENCE"] = n
                                tg_send(chat_id, f"✅ Confidence minimum diubah: {old}% → {n}%")
                            else:
                                tg_send(chat_id, "❌ Nilai harus 0-99.")
                        except ValueError:
                            tg_send(chat_id, "❌ Format salah. Contoh: /confidence_min 50")
                    else:
                        tg_send(chat_id,
                            f"🎯 Confidence minimum saat ini: <b>{monitor_engine.DEFAULTS['MIN_CONFIDENCE']}%</b>\n"
                            "Ubah: /confidence_min 50")

                else:
                    tg_send(chat_id, "❓ Perintah tidak dikenal. /start")

                # --- HANDLE DOCUMENT UPLOAD (untuk /ganti) ---
                if doc and doc.get("file_name", "").endswith(".py"):
                    # Cek apakah user mengirim file Python (kemungkinan /ganti)
                    try:
                        file_id = doc["file_id"]
                        file_info = requests.get(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile",
                            params={"file_id": file_id}, timeout=10
                        ).json()
                        file_path = file_info["result"]["file_path"]
                        file_content = requests.get(
                            f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}",
                            timeout=10
                        ).text

                        # Validasi sintaks
                        try:
                            compile(file_content, "strategy_logic.py", "exec")
                        except SyntaxError as e:
                            tg_send(chat_id, f"❌ Error sintaks di file:\n<code>{e}</code>")
                            continue

                        # Commit ke GitHub
                        try:
                            _commit_to_github(file_content, "strategy_logic.py", "Update strategy_logic via Telegram /ganti")
                            tg_send(chat_id, "✅ File berhasil di-commit ke GitHub!")
                            # Reload
                            if _reload_strategy_logic():
                                tg_send(chat_id, "✅ Strategy logic baru aktif tanpa restart server!")
                            else:
                                tg_send(chat_id, "⚠️ Commit berhasil, tapi reload gagal. Restart manual diperlukan.")
                        except Exception as e:
                            tg_send(chat_id, f"❌ Gagal commit ke GitHub:\n<code>{str(e)[:200]}</code>")

                    except Exception as e:
                        log.error(f"[ganti] {e}")
                        tg_send(chat_id, f"❌ Error processing file: {e}")

            time.sleep(1)

        except Exception as e:
            log.error(f"[bot_loop] {e}")
            time.sleep(5)