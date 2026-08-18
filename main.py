#!/usr/bin/env python3
"""
main.py V5 — MESIN (engine). Telegram handler, API client, monitoring,
stats, export /analyze, hot-swap /ganti. Logika analisa ada di
strategy_logic.py ("Otak"), diimpor di bawah.

Diekstrak dari try22__2_.py, 3 perubahan:
1. Setup logging dipindah ke awal (sebelumnya log dipanggil sebelum
   didefinisikan -> selalu NameError saat start).
2. Fallback strategy_logic gagal load: full_analyze jadi no-op (tidak
   entry baru), tapi TRAIL_R_LADDER dkk tetap terisi biar posisi yang
   sudah terbuka tetap ke-trailing.
3. full_analyze() terima df_h1/df_m15/df_d1 langsung, bukan symbol.
"""
import sys
import os, time, logging, threading
from collections import deque
from datetime import datetime, timezone, timedelta

import requests, pandas as pd, numpy as np, urllib3, json
from flask import Flask

try:
    import websocket   # pip: websocket-client
    _WS_LIB_OK = True
except ImportError:
    _WS_LIB_OK = False

# ── Logging: WAJIB disiapkan sebelum baris lain yang mungkin logging ──
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
MAX_PRICE       = 80.0
TOP_N_COINS     = 50
MONITOR_SLEEP       = 10
# Polling API SIGNED (posisi/order real) sengaja lebih jarang dari MONITOR_SLEEP
# biasa — TP/SL sudah dieksekusi Binance sendiri otomatis begitu tersentuh,
# polling di sini cuma buat TAHU KAPAN itu terjadi (pencatatan), bukan buat
# memicu eksekusinya. Terlalu sering polling = boros weight API tanpa manfaat
# nyata, malah berisiko kena limit/ban (lihat _binance_wait_if_banned).
REAL_TRADE_POLL_SLEEP = 30

# Telegram long-polling / Render watchdog. Telegram polling harus tetap hidup
# walaupun Binance sedang pause; error polling TIDAK boleh ditelan diam-diam.
TELEGRAM_LONGPOLL_TIMEOUT = 20
TELEGRAM_HTTP_TIMEOUT = 30
TELEGRAM_ERROR_BACKOFF_MAX = 60
TELEGRAM_KEEPALIVE_SEC = 300
# Jeda minimum antar-request HTTP ke Binance agar scan tidak menghantam API beruntun.
# 1 request / detik masih cukup untuk scan 50 koin tanpa burst besar.
BINANCE_REQUEST_INTERVAL = 2.5
# Setelah cooldown/ban Binance selesai, tunggu tambahan 60 detik sebelum request pertama.
BINANCE_POST_COOLDOWN_GRACE = 60.0
# Safety governor berbasis header usage; berhenti sebelum mendekati limit 1 menit.
BINANCE_WEIGHT_SOFT_LIMIT = 1800
BINANCE_WEIGHT_HARD_LIMIT = 2100
_binance_request_lock = threading.Lock()
_binance_last_request_at = 0.0
_binance_weight_1m = None
_binance_weight_seen_at = 0.0
MAX_POSITIONS       = 20   # runtime via /max — jangan pindah ke strategy_logic
MONITOR_INTERVAL    = 15 * 60
STRATEGY_MANAGE_INTERVAL = 60
STRATEGY_CONFIDENCE_THRESHOLD = 60  # filter orchestration; strategy tetap menghitung confidence
WIB = timezone(timedelta(hours=7))   # format jam entry di /trade
MAIN_ENGINE_VERSION = "V5"
# ─────────────────────────────────────────────

# Import OTAK — kalau gagal ATAU full_analyze() tidak ada di dalamnya
# (misal file strategy_logic.py yang salah/lama ke-upload), fallback aman.
try:
    from strategy_logic import *
    if "full_analyze" not in dir() or not callable(full_analyze):
        raise ImportError(
            "strategy_logic.py ke-import tapi TIDAK ADA fungsi full_analyze() di dalamnya "
            "— kemungkinan file yang salah/versi lama ter-upload.")
    log.info("[OTAK] strategy_logic.py berhasil dimuat & full_analyze() terverifikasi ada.")
except Exception as e:
    log.error(f"[OTAK] Gagal memuat strategy_logic.py ({e}) — fallback aman aktif.")
    # Engine fallback tidak memiliki aturan trading.
    def full_analyze(df_h1, df_m15, df_d1=None, symbol=None):
        return None
    _STRATEGY_LOAD_ERROR = str(e)
else:
    _STRATEGY_LOAD_ERROR = None

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN tidak ditemukan di environment. Cek file .env")

class TelegramLogHandler(logging.Handler):
    """
    Forward log ERROR/CRITICAL ke Telegram.
    Throttle: maks 1 pesan per 30 detik per pesan unik
    agar tidak flood saat error berulang.
    """
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self._last_sent: dict = {}   # {msg_key: timestamp}
        self._throttle  = 30         # detik

    def emit(self, record):
        # Hindari rekursi (error saat kirim TG itu sendiri)
        if "TG" in record.getMessage(): return
        try:
            msg_key = record.getMessage()[:80]
            now = time.time()
            if now - self._last_sent.get(msg_key, 0) < self._throttle:
                return
            self._last_sent[msg_key] = now

            cid = active_chat_id
            if not cid or not TELEGRAM_TOKEN: return

            level_em = "🔴" if record.levelno >= logging.CRITICAL else "⚠️"
            text = (
                f"{level_em} <b>[{record.levelname}]</b>\n"
                f"<code>{record.getMessage()[:400]}</code>"
            )
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "HTML"},
                timeout=5
            )
        except Exception:
            pass   # jangan pernah raise dari handler log


_tg_log_handler = TelegramLogHandler()
log.addHandler(_tg_log_handler)

auto_mode      = False
auto_thread    = None
active_chat_id = None
timeout_flag   = False
active_trade   = None   # dict posisi yang sedang dipantau, None jika tidak ada

STARTING_BALANCE = 10.0   # modal awal simulasi dalam USD

stat_lock = threading.Lock()
stats = {
    "tp":0, "sl":0, "trail":0, "total":0,
    "balance"    : STARTING_BALANCE,
    "pnl_history": deque(maxlen=20),   # 20 trade terakhir untuk /backtest
}

# Ban koin berbasis SCAN CYCLE (bukan jumlah trade nyata — koin yang selalu
# ke-skip di tahap pending tidak pernah menambah hitungan trade, jadi ban
# berbasis trade tidak akan pernah relevan untuk kasus itu).
ban_lock = threading.Lock()
banned_coins: dict = {}      # {symbol: (scan_counter saat diban, durasi ban itu)}
scan_counter = 0             # bertambah 1 setiap get_top_coins() dipanggil
BAN_DURATION_SCANS = 15
BAN_DURATION_TRADE_CLOSED = 300   # ban khusus setelah trade BENAR-BENAR closed (TP/SL/Trail)

# ── REAL TRADE (Binance Futures) — aktif otomatis kalau API key/secret diset ──
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
# BINANCE_KEYS_PRESENT: apakah kredensial ADA (tetap, tidak berubah runtime).
# REAL_TRADE_ENABLED: mode AKTIF SEKARANG (bisa di-toggle runtime via /mode
# on|off). Default = BINANCE_KEYS_PRESENT, sama seperti perilaku lama (kalau
# key diset, langsung real trade) — supaya tidak ada perubahan perilaku
# default untuk siapa pun yang belum pernah pakai /mode.
BINANCE_KEYS_PRESENT = bool(BINANCE_API_KEY and BINANCE_API_SECRET)
REAL_TRADE_ENABLED   = BINANCE_KEYS_PRESENT

LEVERAGE          = 5      # runtime, via /leverage
MARGIN_USD        = 5.0    # runtime, via /margin
AUTOSTOP_PCT      = 3.0    # runtime, via /autostop
peak_real_balance = None   # diisi saat fetch balance real pertama kali sukses
autostop_lock     = threading.Lock()

def _ban_coin(sym, reason="", duration=None):
    """
    Ban koin selama `duration` siklus scan berikutnya (default
    BAN_DURATION_SCANS). Dipakai dengan duration=BAN_DURATION_TRADE_CLOSED
    khusus di close_position() — trade yang benar-benar closed (TP/SL/
    Trail) dibanned jauh lebih lama daripada kasus pending batal/RR
    gagal/geometri invalid, supaya bot tidak langsung coba koin yang
    sama lagi setelah baru saja selesai trade sungguhan.
    """
    d = duration if duration is not None else BAN_DURATION_SCANS
    with ban_lock:
        banned_coins[sym] = (scan_counter, d)
    log.info(f"[ban] {sym} diban {d} scan" + (f" ({reason})" if reason else ""))

FAPI = "https://fapi.binance.com"
BINANCE_WS_URL = "wss://fstream.binance.com/ws"

# ── Flask ─────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    with stat_lock:
        t=stats["total"]; tp=stats["tp"]; sl=stats["sl"]; trail=stats.get("trail",0)
    with ban_lock:
        n_banned = len(banned_coins)
    wins = tp + trail
    wr=f"{wins/(wins+sl)*100:.1f}%" if (wins+sl)>0 else "–"
    ws_state = "REST (WS fallback siaga)" if ws_feed.is_fresh() else "REST (WS fallback belum siap)"
    return (f"<h3>SMC Signal Broadcaster</h3>"
            f"<p>Auto:{auto_mode} | Banned:{n_banned} | Data:{ws_state}</p>"
            f"<p>Total:{t} TP:{tp} SL:{sl} Trail:{trail} WR:{wr}</p>"), 200

@app.route("/health")
@app.route("/healthz")
def health():
    # Endpoint ini sengaja TIDAK memanggil Binance/API eksternal.
    # Aman dipakai Render/uptime monitor untuk menjaga service tetap hidup.
    with _telegram_state_lock:
        tg_ok = _telegram_polling_alive
        tg_last = _telegram_last_success_at
    with _binance_pause_lock:
        paused = _binance_scan_paused or _binance_recovering
    body = {
        "status": "ok",
        "telegram_polling": bool(tg_ok),
        "telegram_last_success_age": (round(time.time()-tg_last, 1) if tg_last else None),
        "binance_scan_paused": bool(paused),
        "timestamp": time.time(),
    }
    return body, 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    log.info(f"[flask] binding port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═════════════════════════════════════════════
# TELEGRAM
# ═════════════════════════════════════════════
def tg_send(chat_id, text):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id":chat_id,"text":text,"parse_mode":"HTML"},
            timeout=10)
        if r.status_code >= 400:
            log.warning(f"[TG/sendMessage] HTTP {r.status_code}: {r.text[:300]}")
    except Exception as e:
        log.error(f"[TG/sendMessage] {e}")

# ============================================================
# TAMBAHAN BARU (START) — Helper kirim file ke Telegram
# ============================================================
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
# ============================================================
# TAMBAHAN BARU (END)
# ============================================================


# ============================================================
# TAMBAHAN BARU (START) — GitHub API untuk /ganti
# ============================================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME")  # format: "username/repo"

def _commit_to_github(content, path="strategy_logic.py", commit_msg="Update strategy_logic via Telegram /ganti"):
    """Commit file ke GitHub menggunakan API."""
    if not GITHUB_TOKEN or not REPO_NAME:
        raise ValueError("GITHUB_TOKEN atau REPO_NAME tidak diset di environment.")
    
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
    import base64
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
# ============================================================
# TAMBAHAN BARU (END)
# ============================================================

# ── TELEGRAM POLLING STATE ───────────────────────────────────────────────
_telegram_state_lock = threading.Lock()
_telegram_polling_alive = False
_telegram_last_success_at = 0.0
_telegram_last_error_at = 0.0
_telegram_last_conflict_alert_at = 0.0

class TelegramPollingConflict(ConnectionError):
    """Telegram 409: webhook/instance lain bentrok dengan getUpdates."""


def _telegram_mark_success():
    global _telegram_polling_alive, _telegram_last_success_at
    with _telegram_state_lock:
        _telegram_polling_alive = True
        _telegram_last_success_at = time.time()


def _telegram_mark_error():
    global _telegram_polling_alive, _telegram_last_error_at
    with _telegram_state_lock:
        _telegram_polling_alive = False
        _telegram_last_error_at = time.time()


def _telegram_bootstrap():
    """Pastikan token memakai long polling, bukan webhook lama yang tertinggal."""
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo",
            timeout=10)
        d = r.json()
        if not d.get("ok"):
            log.warning(f"[TG] getWebhookInfo gagal: {d}")
            return
        info = d.get("result", {})
        url = info.get("url") or ""
        if url:
            log.warning(f"[TG] Webhook aktif ({url}) — hapus agar long polling tidak 409.")
            rr = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook",
                params={"drop_pending_updates": False}, timeout=10)
            rd = rr.json()
            if rd.get("ok"):
                log.info("[TG] Webhook lama dihapus; long polling siap.")
            else:
                log.error(f"[TG] deleteWebhook gagal: {rd}")
    except Exception as e:
        log.warning(f"[TG] bootstrap error: {e}")


def tg_updates(offset=None):
    """Long poll Telegram dengan error visibility + backoff signal.

    Tidak pernah mengembalikan [] secara diam-diam saat HTTP/JSON gagal, karena
    itu membuat bot terlihat hidup padahal command tidak diterima.
    """
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"timeout": TELEGRAM_LONGPOLL_TIMEOUT, "offset": offset},
            timeout=TELEGRAM_HTTP_TIMEOUT)
        if r.status_code == 409:
            _telegram_mark_error()
            raise TelegramPollingConflict(
                "Telegram 409 Conflict: ada webhook atau instance bot lain yang memakai token ini.")
        if r.status_code == 429:
            _telegram_mark_error()
            try:
                d = r.json(); retry_after = int(d.get("parameters", {}).get("retry_after", 5))
            except Exception:
                retry_after = 5
            raise ConnectionError(f"Telegram rate limit 429; retry_after={retry_after}s")
        if r.status_code >= 500:
            _telegram_mark_error()
            raise ConnectionError(f"Telegram server HTTP {r.status_code}")
        r.raise_for_status()
        d = r.json()
        if not d.get("ok"):
            _telegram_mark_error()
            raise ConnectionError(f"Telegram getUpdates error: {d}")
        _telegram_mark_success()
        return d.get("result", [])
    except TelegramPollingConflict:
        raise
    except Exception as e:
        _telegram_mark_error()
        log.warning(f"[TG/getUpdates] {e}")
        raise


# ═════════════════════════════════════════════
# DATA LAYER — REST sebagai sumber UTAMA, WS cuma fallback TERAKHIR
#   Tier 1: Binance Futures REST        (sumber utama)
#   Tier 2: Bybit REST                  (kalau Binance REST error/kena
#           limit/ban — lihat fapi_get(): begitu Binance balas 418/429,
#           retry ke Binance langsung dihentikan, tidak ditunggu2)
#   Tier 3: Binance Futures WebSocket   (fallback TERAKHIR, dipakai hanya
#           kalau Tier 1 & Tier 2 dua-duanya gagal. WS tetap disubscribe
#           & di-backfill terus di background — lihat ensure_symbol_
#           interval() — supaya buffernya SIAP dipakai sewaktu-waktu,
#           tapi TIDAK dijadikan sumber utama krn koneksinya sering
#           putus-nyambung di lingkungan hosting ini)
#   Tier 4: CoinGecko REST — DARURAT, HARGA SAJA, hanya koin-koin di
#           COINGECKO_ID_MAP. TIDAK dipakai untuk klines: granularitas
#           candle CoinGecko (30m/4h/4hari tergantung rentang) tidak
#           cocok dengan kebutuhan M1/M15/H1/D1 presisi bot ini — kalau
#           dipaksakan, sinyal SMC yang butuh candle presisi (BOS/CHoCH/
#           swing point) bisa salah baca. Kalau semua REST+WS gagal
#           total, get_klines() balikin DataFrame kosong (sama seperti
#           perilaku lama) alih-alih pura-pura pakai data CoinGecko yang
#           tidak akurat.
# ═════════════════════════════════════════════
BYBIT = "https://api.bybit.com"

# Konversi interval Binance → Bybit
INTERVAL_MAP = {
    "1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
    "1h":"60","2h":"120","4h":"240","1d":"D","1w":"W",
}

# Simbol Binance Futures -> id CoinGecko, HANYA koin-koin besar yang aman
# di-mapping manual (ticker collision antar chain bikin auto-match ke
# CoinGecko berisiko fatal — bisa ambil harga koin yang salah). Tambah
# manual kalau perlu koin lain, JANGAN pernah generate otomatis dari nama.
COINGECKO_ID_MAP = {
    "BTCUSDT":"bitcoin", "ETHUSDT":"ethereum", "BNBUSDT":"binancecoin",
    "SOLUSDT":"solana", "XRPUSDT":"ripple", "ADAUSDT":"cardano",
    "DOGEUSDT":"dogecoin", "AVAXUSDT":"avalanche-2", "LINKUSDT":"chainlink",
    "DOTUSDT":"polkadot", "LTCUSDT":"litecoin", "TRXUSDT":"tron",
    "ATOMUSDT":"cosmos", "NEARUSDT":"near", "APTUSDT":"aptos",
    "ARBUSDT":"arbitrum", "OPUSDT":"optimism", "SUIUSDT":"sui",
    "TONUSDT":"the-open-network", "BCHUSDT":"bitcoin-cash",
}

import re

# ── State ban IP Binance — DIBAGI antara fapi_get (publik) & _binance_signed
# (private), karena ban itu per-IP, bukan per-endpoint/per-key. Begitu satu
# sisi kena ban, sisi lain juga harus tahu & berhenti nembak, bukan lanjut
# jalan sendiri-sendiri (itu yang bikin log kebanjiran "Skip ... HTTP 418").
_binance_ban_lock = threading.Lock()
_binance_banned_until = 0.0   # unix timestamp detik; 0 = tidak sedang ban
_BINANCE_BAN_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".binance_ban_state.json")
# Global circuit breaker: saat Binance 429/418, NEW SCAN/ENTRY berhenti.
# WS tetap hidup untuk memantau posisi aktif.
_binance_scan_paused = False
_binance_pause_reason = ""
_binance_recovering = False
_binance_pause_lock = threading.Lock()
# Pending trail: satu state terbaru per simbol, bukan queue order lama berantai.
_pending_trails = {}   # {symbol: {sl, tp, quantity, updated_at, reason, side}}
_pending_trails_lock = threading.Lock()
_pending_protections = {}  # filled position awaiting TP/SL after Binance recovery
_pending_protections_lock = threading.Lock()

class BinanceCooldownError(ConnectionError):
    """Tidak mengirim request Binance selama cooldown aktif."""


def _load_binance_ban_state():
    global _binance_banned_until
    try:
        with open(_BINANCE_BAN_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        until = float(data.get("banned_until", 0.0))
        if until > time.time():
            _binance_banned_until = until
            with _binance_pause_lock:
                globals()["_binance_scan_paused"] = True
                globals()["_binance_pause_reason"] = "persisted Binance cooldown"
            log.warning(f"[binance] cooldown dipulihkan: {until-time.time():.0f} detik tersisa")
    except Exception:
        pass


def _save_binance_ban_state(until):
    try:
        tmp = _BINANCE_BAN_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"banned_until": float(until)}, f)
        os.replace(tmp, _BINANCE_BAN_STATE_FILE)
    except Exception as e:
        log.debug(f"[binance] gagal simpan cooldown: {e}")

_load_binance_ban_state()


def _binance_register_ban(msg="", fallback_seconds=60, retry_after=None):
    """Aktifkan circuit breaker global saat Binance rate-limit/ban.
    retry_after (detik) diprioritaskan bila tersedia dari header/API response.
    """
    global _binance_banned_until, _binance_scan_paused, _binance_pause_reason
    m = re.search(r"banned until (\d+)", msg or "")
    candidates = [time.time() + fallback_seconds]
    if retry_after is not None:
        try:
            candidates.append(time.time() + max(float(retry_after), 0.0))
        except (TypeError, ValueError):
            pass
    if m:
        candidates.append(int(m.group(1)) / 1000)
    until = max(candidates)
    until += BINANCE_POST_COOLDOWN_GRACE
    with _binance_ban_lock:
        _binance_banned_until = max(_binance_banned_until, until)
        current_until = _binance_banned_until
    with _binance_pause_lock:
        _binance_scan_paused = True
        _binance_pause_reason = (msg or "Binance rate limit / ban")[:180]
    _save_binance_ban_state(current_until)
    log.error(f"[BINANCE PAUSE] Scanner & entry BARU dihentikan selama {max(current_until-time.time(),0):.0f} detik. WS tetap memantau posisi.")


def _binance_is_scan_paused():
    with _binance_pause_lock:
        paused = _binance_scan_paused or _binance_recovering
    if paused:
        return True
    return _binance_cooldown_remaining() > 0


def _binance_cooldown_remaining():
    with _binance_ban_lock:
        return max(0.0, _binance_banned_until - time.time())


def _binance_try_resume():
    global _binance_scan_paused, _binance_pause_reason
    if _binance_cooldown_remaining() > 0:
        return False
    with _binance_pause_lock:
        _binance_scan_paused = False
        _binance_pause_reason = ""
    return True


def _queue_pending_trail(sym, new_sl, new_tp, qty, reason="strategy", side=None):
    """Simpan state trail terbaru per simbol; order lama tidak ditumpuk."""
    with _pending_trails_lock:
        old = _pending_trails.get(sym)
        if old is None:
            _pending_trails[sym] = {
                "sl": new_sl, "tp": new_tp, "quantity": qty,
                "updated_at": time.time(), "reason": reason, "side": side,
            }
            return
        buy = (side or old.get("side")) == "BUY"
        old_sl = old.get("sl")
        better_sl = (new_sl is not None and old_sl is None) or (new_sl is not None and ((new_sl > old_sl) if buy else (new_sl < old_sl)))
        if better_sl or new_tp != old.get("tp") or (qty and qty != old.get("quantity")):
            old.update({"sl": new_sl, "tp": new_tp, "quantity": qty, "updated_at": time.time(), "reason": reason, "side": side or old.get("side")})


def _get_pending_trail(sym):
    with _pending_trails_lock:
        v = _pending_trails.get(sym)
        return dict(v) if v else None


def _clear_pending_trail(sym):
    with _pending_trails_lock:
        _pending_trails.pop(sym, None)


def _binance_wait_if_banned():
    with _binance_ban_lock:
        until = _binance_banned_until
    remaining = until - time.time()
    if remaining > 0:
        raise BinanceCooldownError(f"Binance cooldown aktif {remaining:.0f}s")

def _binance_update_weight_from_response(r):
    """Catat request-weight 1m dari header Binance bila tersedia."""
    global _binance_weight_1m, _binance_weight_seen_at
    raw = None
    for key in ("X-MBX-USED-WEIGHT-1M", "x-mbx-used-weight-1m"):
        if key in r.headers:
            raw = r.headers.get(key)
            break
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    _binance_weight_1m = value
    _binance_weight_seen_at = time.time()
    return value


def _binance_request_pause():
    """Throttle semua request HTTP Binance + weight governor.
    429/418 tidak pernah di-retry; hard/soft usage hanya untuk mencegah ban lebih awal.
    """
    global _binance_last_request_at
    _binance_wait_if_banned()
    now = time.monotonic()
    with _binance_request_lock:
        # Kalau observed 1m weight sudah tinggi, tunggu masuk window menit berikutnya.
        # Ini jauh lebih aman daripada terus mengirim request sampai Binance yang 429.
        if _binance_weight_1m is not None and _binance_weight_1m >= BINANCE_WEIGHT_SOFT_LIMIT:
            wall_now = time.time()
            wait_window = max(0.0, 62.0 - (wall_now % 60.0))
            log.warning(f"[binance-weight] {_binance_weight_1m} weight/1m — throttle {wait_window:.1f}s ke window berikutnya.")
            time.sleep(wait_window)
        wait = BINANCE_REQUEST_INTERVAL - (time.monotonic() - _binance_last_request_at)
        if wait > 0:
            time.sleep(wait)
        _binance_last_request_at = time.monotonic()


def _raw_get(url, params=None, retries=3):
    """HTTP GET dengan retry — digunakan oleh Bybit & CoinGecko."""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10, verify=False)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"[http] {i+1}/{retries} {url}: {e}")
            time.sleep(2)
    raise ConnectionError(f"GET gagal: {url}")


# ── BINANCE REST (backfill awal WS + fallback tier-2) ─────────────────
def fapi_get(path, params=None):
    # Satu request saja per pemanggilan. Retry REST publik ke Binance dihapus:
    # saat gagal langsung gunakan fallback agar error tidak berubah menjadi burst.
    _binance_wait_if_banned()
    try:
        _binance_request_pause()
        r = requests.get(f"{FAPI}{path}", params=params, timeout=10, verify=False)
        used = _binance_update_weight_from_response(r)
        if used is not None and used >= BINANCE_WEIGHT_HARD_LIMIT:
            log.warning(f"[binance-weight] {used} weight/1m — menahan request baru sampai window mereda.")
        if r.status_code in (418, 429):
            retry_after = r.headers.get("Retry-After")
            _binance_register_ban(r.text or "", retry_after=retry_after)
            raise BinanceCooldownError(f"Binance kena limit/ban (HTTP {r.status_code})")
        d = r.json()
        if isinstance(d, dict) and "code" in d:
            if d["code"] == -1003:
                retry_after = None
                _binance_register_ban(d.get("msg", ""), retry_after=retry_after)
                raise BinanceCooldownError(f"Binance {d['code']}: {d.get('msg')}")
            raise ValueError(f"Binance {d['code']}: {d.get('msg')}")
        return d
    except BinanceCooldownError:
        raise
    except Exception as e:
        log.warning(f"[binance] {path} gagal: {e} — langsung fallback")
        raise ConnectionError(f"Binance gagal: {path}: {e}") from e


# ============================================================
# REAL TRADE — Binance Futures signed API (order/leverage/posisi)
# Dipakai TERPISAH dari fapi_get di atas (yang publik, untuk cari
# sinyal) supaya limit rate keduanya tidak bercampur.
# ============================================================
import hmac, hashlib, urllib.parse, math
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN

def _binance_signed(method, path, params=None):
    if not REAL_TRADE_ENABLED:
        raise RuntimeError("BINANCE_API_KEY/SECRET tidak diset")
    _binance_wait_if_banned()
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    query = urllib.parse.urlencode(params, safe=",")
    sig = hmac.new(BINANCE_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url = f"{FAPI}{path}?{query}&signature={sig}"
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    last_err = None
    for attempt in range(3):
        try:
            _binance_request_pause()
            r = requests.request(method, url, headers=headers, timeout=10, verify=False)
            used = _binance_update_weight_from_response(r)
            if used is not None and used >= BINANCE_WEIGHT_HARD_LIMIT:
                log.warning(f"[binance-weight] {used} weight/1m setelah signed {method} {path}")
            if r.status_code in (418, 429):
                retry_after = r.headers.get("Retry-After")
                _binance_register_ban(r.text, retry_after=retry_after)
                raise BinanceCooldownError(f"Binance kena limit/ban (HTTP {r.status_code})")
            data = r.json()
            if isinstance(data, dict) and "code" in data and data["code"] < 0:
                if data["code"] == -1003:
                    _binance_register_ban(data.get("msg", ""))
                    raise BinanceCooldownError(f"Binance {data['code']}: {data.get('msg')}")
                raise RuntimeError(f"Binance {data['code']}: {data.get('msg')}")
            return data
        except BinanceCooldownError:
            # WAJIB: 418/429 TIDAK PERNAH masuk retry loop.
            raise
        except RuntimeError:
            raise
        except Exception as e:
            last_err = e
            log.warning(f"[binance-signed] {method} {path} percobaan {attempt+1}: {e}")
            time.sleep(1.5)
    raise RuntimeError(f"Gagal request signed {method} {path}: {last_err}")


_symbol_filters_cache = {}
_exchange_info_cache = {"fetched_at": 0.0}
_exchange_info_lock = threading.Lock()

def _load_all_symbol_filters():
    """Fetch /fapi/v1/exchangeInfo SEKALI, parse SEMUA simbol sekaligus ke
    cache — supaya koin baru berikutnya tidak perlu fetch ulang endpoint
    berat ini. Refresh tiap 1 jam (filter simbol jarang berubah)."""
    with _exchange_info_lock:
        if time.time() - _exchange_info_cache["fetched_at"] < 3600 and _symbol_filters_cache:
            return
        data = fapi_get("/fapi/v1/exchangeInfo")
        for s in data["symbols"]:
            f = {x["filterType"]: x for x in s["filters"]}
            if "LOT_SIZE" not in f or "PRICE_FILTER" not in f:
                continue
            _symbol_filters_cache[s["symbol"]] = {
                "stepSize": float(f["LOT_SIZE"]["stepSize"]),
                "minQty": float(f["LOT_SIZE"]["minQty"]),
                "minNotional": float(f.get("MIN_NOTIONAL", {}).get("notional", 5.0)),
                "tickSize": float(f["PRICE_FILTER"]["tickSize"]),
                "qtyPrecision": s["quantityPrecision"],
                "pricePrecision": s["pricePrecision"],
            }
        _exchange_info_cache["fetched_at"] = time.time()

def get_symbol_filters(symbol):
    if symbol not in _symbol_filters_cache:
        _load_all_symbol_filters()
    if symbol not in _symbol_filters_cache:
        raise ValueError(f"{symbol} tidak ada di exchangeInfo")
    return _symbol_filters_cache[symbol]


def round_to_tick(price, tick_size):
    """Bulatkan ke kelipatan PERSIS tickSize (bukan cuma jumlah desimal —
    dua hal beda, sumber error -4014 'Price not increased by tick size').
    Pakai Decimal supaya tidak kena noise floating point (mis. 0.0005)."""
    if not tick_size or tick_size <= 0:
        return price
    d_price, d_tick = Decimal(str(price)), Decimal(str(tick_size))
    steps = (d_price / d_tick).to_integral_value(rounding=ROUND_HALF_UP)
    return float(steps * d_tick)


def round_qty(quantity, step, qty_prec, rounding=ROUND_HALF_UP):
    """Bulatkan quantity ke kelipatan PERSIS stepSize pakai Decimal.

    KENAPA INI PENTING (bug 'posisi tidak 100% tertutup saat SL'):
    ─────────────────────────────────────────────────────────────
    `math.floor(quantity / step) * step` pakai float biasa KENA noise
    binary floating point. Contoh nyata: quantity=1.2, step=0.1 →
    1.2/0.1 di Python = 11.999999999999998 (BUKAN 12.0), lalu
    math.floor() membulatkannya jadi 11 → hasil 1.1, padahal quantity
    aslinya 1.2. Order SL (reduceOnly=quantity) yang dipasang jadi
    0.1 koin LEBIH KECIL dari posisi riil → saat SL ter-trigger, 0.1
    koin itu TIDAK IKUT TERTUTUP dan posisi tersisa terbuka selamanya
    (tanpa proteksi SL sama sekali untuk sisa itu).

    Fix: pakai Decimal(str(...)) supaya representasi desimalnya EXACT
    (bukan biner), dan default ROUND_HALF_UP (bulatkan ke kelipatan
    step TERDEKAT) — karena quantity yang masuk ke sini (qty posisi
    aktif) SEHARUSNYA sudah persis kelipatan step sejak awal dibuka
    (lihat calc_auto_quantity), jadi tidak perlu di-floor lagi; floor
    kedua itulah sumber bug di atas. Kalau memang perlu floor murni
    (mis. saat menghitung qty MAKSIMUM yang boleh dibeli dari suatu
    notional — lihat calc_auto_quantity), panggil dengan
    rounding=ROUND_DOWN secara eksplisit.
    """
    if not step or step <= 0:
        return round(quantity, qty_prec)
    d_qty, d_step = Decimal(str(quantity)), Decimal(str(step))
    steps = (d_qty / d_step).to_integral_value(rounding=rounding)
    return float(round(steps * d_step, qty_prec))


def calc_auto_quantity(symbol, entry_price, margin_usd, leverage):
    """
    Quantity dari margin x leverage, dibulatkan ke stepSize Binance.
    Kalau di bawah minQty/minNotional (error -1013 LOT_SIZE / -4164
    MIN_NOTIONAL), margin dinaikkan SEDIKIT supaya order tetap valid.
    Cap kenaikan = mana yang LEBIH BESAR antara 3x margin awal ATAU
    margin awal + $5 — kombinasi ini supaya margin kecil (mis. $1) tetap
    dapat headroom wajar (cuma 1.5x dari $1 = $1.5, kelewat sempit utk
    banyak koin), sementara margin besar tidak melonjak tak terkendali.
    Return (qty, margin_terpakai, dinaikkan?) atau (None, None, False)
    kalau tetap gagal walau sudah disesuaikan.
    """
    info = get_symbol_filters(symbol)
    step, min_qty, min_notional = info["stepSize"], info["minQty"], info["minNotional"]

    def qty_from_notional(notional):
        # Sama seperti place_sl_order() — pakai Decimal (ROUND_DOWN, exact)
        # bukan math.floor(float) supaya tidak kehilangan 1 step ekstra
        # akibat noise floating point (mis. 1.2/0.1 = 11.999999999998 di
        # Python murni). Floor tetap dipertahankan di sini (memang harus
        # floor, bukan nearest) karena tujuannya membatasi qty MAKSIMUM
        # yang boleh dibeli dari notional yang tersedia.
        q = round_qty(notional / entry_price, step, info["qtyPrecision"], rounding=ROUND_DOWN)
        return q

    qty = qty_from_notional(margin_usd * leverage)
    if qty >= min_qty and qty * entry_price >= min_notional:
        return qty, margin_usd, False

    needed_notional = max(min_notional, min_qty * entry_price) * 1.01
    bumped_margin = needed_notional / leverage
    cap = max(margin_usd * 3, margin_usd + 5)
    if bumped_margin > cap:
        log.warning(f"[calc_auto_quantity] {symbol}: butuh margin ${bumped_margin:.4f} "
                    f"tapi cap cuma ${cap:.4f} (margin awal ${margin_usd:.2f}, leverage {leverage}x)")
        return None, None, False
    qty = qty_from_notional(needed_notional)
    if qty < min_qty or qty * entry_price < min_notional:
        return None, None, False
    return qty, round(bumped_margin, 4), True


def set_leverage(symbol, leverage):
    return _binance_signed("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})


def place_limit_order(symbol, side, quantity, price):
    tick = get_symbol_filters(symbol)["tickSize"]
    return _binance_signed("POST", "/fapi/v1/order", {
        "symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTC",
        "quantity": quantity, "price": round_to_tick(price, tick),
    })


def place_market_order(symbol, side, quantity, reduce_only=False):
    params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity}
    if reduce_only:
        params["reduceOnly"] = "true"
    return _binance_signed("POST", "/fapi/v1/order", params)


def cancel_order(symbol, order_id):
    """Cancel ORDER BIASA (limit/market entry) — bukan TP/SL, lihat cancel_algo_order."""
    if not order_id: return None
    try:
        return _binance_signed("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id})
    except Exception as e:
        log.warning(f"[cancel_order] {symbol} #{order_id}: {e}")
        return None


def get_order_status(symbol, order_id):
    return _binance_signed("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id})


def get_real_position(symbol):
    rows = _binance_signed("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
    for p in rows:
        if p["symbol"] == symbol and abs(float(p["positionAmt"])) > 0:
            return p
    return None


# ── TP/SL sekarang WAJIB lewat Algo Order API (Binance migrasi order kondisional
# ke /fapi/v1/algoOrder per 9 Des 2025 — endpoint /fapi/v1/order lama menolaknya
# dengan error -4120). Field beda dari order biasa: stopPrice->triggerPrice,
# orderId->algoId, status->algoStatus. ──

def place_sl_order(symbol, is_buy, sl_price, quantity):
    """
    Pasang SL sebagai Conditional Algo Order di Binance Futures.

    KENAPA quantity + reduceOnly, BUKAN closePosition=true:
    ─────────────────────────────────────────────────────────
    Binance Futures hanya mengizinkan SATU order dengan `closePosition=true`
    per sisi (side) per simbol secara bersamaan. TP sudah dipasang dengan
    `closePosition=true` (SELL untuk posisi BUY). Kalau SL juga pakai
    `closePosition=true`, Binance otomatis mem-cancel salah satunya —
    biasanya SL yang dipasang kedua. Inilah penyebab SL terus "hilang"
    segera setelah dipasang ulang.

    Solusi: SL pakai `quantity` (jumlah lot yang sama dengan posisi) +
    `reduceOnly=true`. Ini setara dengan menutup seluruh posisi saat
    ter-trigger, TANPA konflik dengan TP. Ketika TP atau SL ter-trigger,
    Binance otomatis meng-cancel order lainnya (karena posisinya sudah nol
    dan order reduce-only tidak punya posisi untuk di-reduce).
    """
    close_side = "SELL" if is_buy else "BUY"
    info = get_symbol_filters(symbol)
    tick = info["tickSize"]
    step = info["stepSize"]
    qty_prec = info.get("qtyPrecision", 8)
    # FIX bug "posisi tidak 100% tertutup saat SL": sebelumnya pakai
    # math.floor(quantity/step)*step dengan float biasa, yang kena noise
    # binary floating point (mis. 1.2/0.1 = 11.999999999998 → floor jadi
    # 1.1, bukan 1.2) sehingga qty SL selalu bisa 1 step LEBIH KECIL dari
    # posisi riil dan menyisakan sebagian posisi tanpa proteksi saat SL
    # ter-trigger. round_qty() pakai Decimal (exact) + bulat ke TERDEKAT,
    # bukan floor lagi — qty yang masuk ke sini sudah kelipatan step sejak
    # awal (dari calc_auto_quantity), jadi tidak boleh di-floor kedua kali.
    qty_rounded = round_qty(quantity, step, qty_prec)
    return _binance_signed("POST", "/fapi/v1/algoOrder", {
        "algoType": "CONDITIONAL", "symbol": symbol, "side": close_side, "type": "STOP_MARKET",
        "triggerPrice": round_to_tick(sl_price, tick),
        "quantity": qty_rounded,
        "reduceOnly": "true",
        "workingType": "MARK_PRICE",
    })


def place_tp_sl(symbol, is_buy, tp_price, sl_price, quantity):
    """
    Pasang TP + SL sekaligus.

    KEDUANYA quantity + reduceOnly=true (BUKAN closePosition=true untuk TP
    lagi — lihat catatan di bawah). Simetris dengan place_sl_order().

    ── KENAPA TP JUGA DIUBAH (bukan cuma SL) ──────────────────────────────
    Sebelumnya cuma SL yang diubah ke quantity+reduceOnly, TP tetap pakai
    closePosition=true, dengan asumsi itu sudah cukup untuk menghindari
    konflik "1 closePosition per side". Tapi laporan user menunjukkan SL
    tetap hilang berulang meski sudah dipisah begitu — artinya closePosition
    di sisi TP kemungkinan MASIH bisa memicu Binance meng-cancel order lain
    (mis. reduceOnly quantity yang jumlahnya persis = seluruh posisi bisa
    tetap dianggap "menutup posisi penuh" oleh sebagian implementasi, sama
    seperti closePosition). Fix paling aman & pasti tidak ambigu: TP dan SL
    SAMA-SAMA quantity+reduceOnly biasa — tidak ada mekanisme closePosition
    sama sekali di simbol ini, jadi tidak ada jalur bagi Binance untuk
    menganggap salah satu order "menggantikan" yang lain.
    """
    close_side = "SELL" if is_buy else "BUY"
    info = get_symbol_filters(symbol)
    tick, step = info["tickSize"], info["stepSize"]
    qty_prec = info.get("qtyPrecision", 8)
    qty_rounded = round_qty(quantity, step, qty_prec)
    tp = _binance_signed("POST", "/fapi/v1/algoOrder", {
        "algoType": "CONDITIONAL", "symbol": symbol, "side": close_side, "type": "TAKE_PROFIT_MARKET",
        "triggerPrice": round_to_tick(tp_price, tick),
        "quantity": qty_rounded,
        "reduceOnly": "true",
        "workingType": "MARK_PRICE",
    })
    sl = place_sl_order(symbol, is_buy, sl_price, quantity)
    return tp, sl


def cancel_algo_order(algo_id):
    if not algo_id: return None
    try:
        return _binance_signed("DELETE", "/fapi/v1/algoOrder", {"algoId": algo_id})
    except Exception as e:
        log.warning(f"[cancel_algo_order] #{algo_id}: {e}")
        return None


def get_algo_order_status(algo_id):
    return _binance_signed("GET", "/fapi/v1/algoOrder", {"algoId": algo_id})


def cancel_all_algo_orders(symbol):
    """Bersihkan SEMUA algo order (TP/SL) tersisa di suatu koin — dipakai
    sebagai jaring pengaman setelah posisi closed, jaga-jaga salah satu
    order (TP atau SL) belum ke-cancel otomatis oleh Binance."""
    try:
        return _binance_signed("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol})
    except Exception as e:
        log.warning(f"[cancel_all_algo_orders] {symbol}: {e}")
        return None


def get_real_balance():
    """Return (available, total) USDT, atau (None, None) kalau gagal."""
    try:
        rows = _binance_signed("GET", "/fapi/v2/balance", {})
        for r in rows:
            if r["asset"] == "USDT":
                return float(r["availableBalance"]), float(r["balance"])
    except Exception as e:
        log.warning(f"[get_real_balance] {e}")
    return None, None


def get_public_ip():
    try:
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except Exception:
        return "unknown"


def _binance_klines(symbol, interval, limit):
    raw = fapi_get("/fapi/v1/klines",
                   {"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(raw, list) or len(raw) < min(limit, 40):
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=[
        "ts","open","high","low","close","volume",
        "cts","qvol","trades","tbv","tbq","ign"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open","high","low","close","volume"]].dropna()

def _binance_price(symbol):
    d = fapi_get("/fapi/v1/ticker/price", {"symbol": symbol})
    return float(d["price"])

def _binance_top_coins(exclude_syms):
    tickers = fapi_get("/fapi/v1/ticker/24hr")
    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t["quoteVolume"]) > 5_000_000
        and abs(float(t.get("priceChangePercent","0"))) < 15
        and t["symbol"] not in exclude_syms
    ]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]


# ── BYBIT (fallback tier-3) ────────────────────────────────────────────
def _bybit_klines(symbol, interval, limit):
    iv = INTERVAL_MAP.get(interval, "15")
    d = _raw_get(f"{BYBIT}/v5/market/kline", {
        "category":"linear","symbol":symbol,
        "interval":iv,"limit":limit
    })
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit kline error: {d.get('retMsg')}")
    rows = d["result"]["list"]
    if not rows or len(rows) < min(limit, 40):
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df["ts"].astype(float), unit="ms")
    df = df.sort_index()
    return df[["open","high","low","close","volume"]].dropna()

def _bybit_price(symbol):
    d = _raw_get(f"{BYBIT}/v5/market/tickers",
                 {"category":"linear","symbol":symbol})
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit ticker error: {d.get('retMsg')}")
    return float(d["result"]["list"][0]["lastPrice"])

def _bybit_top_coins(exclude_syms):
    d = _raw_get(f"{BYBIT}/v5/market/tickers", {"category":"linear"})
    if d.get("retCode", -1) != 0:
        raise ValueError(f"Bybit tickers error: {d.get('retMsg')}")
    items = d["result"]["list"]
    usdt = [
        t for t in items
        if t["symbol"].endswith("USDT")
        and 0.0001 < float(t["lastPrice"]) < MAX_PRICE
        and float(t.get("turnover24h","0")) > 5_000_000
        and abs(float(t.get("price24hPcnt","0"))) < 0.15
        and t["symbol"] not in exclude_syms
    ]
    usdt.sort(key=lambda x: float(x.get("turnover24h","0")), reverse=True)
    return [t["symbol"] for t in usdt[:TOP_N_COINS]]


# ── COINGECKO (fallback tier-4, DARURAT — harga saja) ──────────────────
def _coingecko_price(symbol):
    cid = COINGECKO_ID_MAP.get(symbol)
    if not cid:
        return None
    try:
        d = _raw_get("https://api.coingecko.com/api/v3/simple/price",
                     {"ids": cid, "vs_currencies": "usd"}, retries=1)
        p = d.get(cid, {}).get("usd")
        return float(p) if p is not None else None
    except Exception as e:
        log.warning(f"[price/coingecko] {symbol}: {e}")
        return None


# ── WEBSOCKET FEED (tier-1) ─────────────────────────────────────────────
class BinanceWSFeed:
    """
    Satu koneksi WS gabungan (raw stream endpoint, subscribe dinamis) ke
    Binance Futures:
      - !ticker@arr        → harga + statistik 24 jam SEMUA simbol tiap
                              ~1 detik. Menggantikan polling REST batch
                              utk get_price() & get_top_coins() sepenuhnya
                              begitu WS ini live — jauh lebih hemat rate
                              limit/risiko IP ban dibanding sebelumnya.
      - <sym>@kline_<itv>  → update candle real-time, HANYA utk pasangan
                              (simbol, interval) yang benar-benar diminta
                              get_klines() — subscribe on-demand (lazy),
                              bukan semua 50 koin x semua interval sekaligus,
                              biar hemat kuota stream & bandwidth.

    Catatan penting: WS TIDAK BISA memberi histori candle sebelum koneksi
    dibuka — itu keterbatasan protokol, bukan celah desain. Karena itu
    setiap (simbol, interval) yang baru pertama kali diminta di-backfill
    SEKALI via REST (Binance → Bybit), baru setelah itu WS yang menjaga
    buffer tetap update tanpa REST lagi.

    Auto-reconnect dgn exponential backoff (1s→30s), auto re-subscribe
    semua stream yang lagi aktif begitu reconnect berhasil.
    """
    KLINE_INTERVALS = ("1m", "15m", "1h", "1d")
    MAX_CANDLES  = {"1m": 300, "15m": 300, "1h": 300, "1d": 150}
    STALE_AFTER_SEC   = 30     # >30s tanpa pesan masuk → anggap WS mati
    STREAM_IDLE_SEC   = 1800   # (simbol,interval) tak dipakai 30menit → unsubscribe

    def __init__(self):
        self._lock       = threading.Lock()
        self._send_lock  = threading.Lock()
        self._klines     = {}     # {(sym,itv): deque([{t,o,h,l,c,v}, ...])}
        self._ticker     = {}     # {sym: {"symbol","price","qvol","chg"}}
        self._last_used  = {}     # {(sym,itv): timestamp terakhir diminta}
        self._subscribed = set()  # stream string yg lagi aktif di WS
        self._ws         = None
        self._last_msg   = 0.0
        self._connected  = False
        self._stop       = False
        self._backoff    = 1

    # ── public ──
    def start(self):
        if not _WS_LIB_OK:
            log.error("[ws] Modul 'websocket-client' belum terpasang — "
                      "TAMBAHKAN 'websocket-client' ke requirements.txt. "
                      "Bot tetap jalan tapi full REST-only (Binance→Bybit) "
                      "sampai modul ini ada.")
            return
        threading.Thread(target=self._run_forever, daemon=True).start()

    def is_fresh(self):
        return self._connected and (time.time() - self._last_msg) < self.STALE_AFTER_SEC

    def get_price(self, symbol):
        with self._lock:
            d = self._ticker.get(symbol)
            return d["price"] if d else None

    def get_top_coins_raw(self):
        with self._lock:
            return list(self._ticker.values())

    def get_klines(self, symbol, interval, limit=250):
        """Return klines dari buffer WS internal (data yg sudah di-backfill & live-update).
        Dipanggil dari module-level get_klines() sebagai fallback setelah REST gagal."""
        with self._lock:
            buf = self._klines.get((symbol, interval))
            if not buf:
                return pd.DataFrame()
            rows = list(buf)[-limit:]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], unit="ms")
        df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"}, inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    def ensure_symbol_interval(self, symbol, interval):
        """Dipanggil tiap get_klines() — backfill SEKALI kalau baru,
        subscribe stream kalau belum, update timestamp pemakaian terakhir."""
        if not _WS_LIB_OK:
            return
        with self._lock:
            have = (symbol, interval) in self._klines
            self._last_used[(symbol, interval)] = time.time()
        if not have:
            self._backfill(symbol, interval)
        self._subscribe_kline(symbol, interval)

    def cleanup_stale_streams(self):
        """Unsubscribe & buang buffer (simbol,interval) yg tidak dipakai
        >30menit — dipanggil berkala dari watchdog thread, biar jumlah
        stream aktif tetap proporsional dgn pool koin yang sedang jalan."""
        now = time.time()
        with self._lock:
            stale = [k for k, ts in self._last_used.items()
                     if now - ts > self.STREAM_IDLE_SEC]
        for (sym, itv) in stale:
            self._unsubscribe_kline(sym, itv)
            with self._lock:
                self._klines.pop((sym, itv), None)
                self._last_used.pop((sym, itv), None)
        if stale:
            log.info(f"[ws] cleanup {len(stale)} stream idle >30menit")

    # ── internal: backfill histori awal via REST ──
    def _backfill(self, symbol, interval):
        limit = self.MAX_CANDLES.get(interval, 250)
        df, src = pd.DataFrame(), None
        try:
            df = _binance_klines(symbol, interval, limit)
            if not df.empty: src = "binance"
        except Exception as e:
            log.warning(f"[ws-backfill/binance] {symbol} {interval}: {e}")
        if df.empty:
            try:
                df = _bybit_klines(symbol, interval, limit)
                if not df.empty: src = "bybit"
            except Exception as e:
                log.warning(f"[ws-backfill/bybit] {symbol} {interval}: {e}")
        if df.empty:
            log.warning(f"[ws-backfill] {symbol} {interval} GAGAL TOTAL "
                        f"(binance+bybit) — coba lagi di pemanggilan berikutnya")
            return
        rows = deque(maxlen=limit)
        for ts, r in df.iterrows():
            rows.append({"t": int(ts.timestamp()*1000), "o": float(r.open),
                         "h": float(r.high), "l": float(r.low),
                         "c": float(r.close), "v": float(r.volume)})
        with self._lock:
            self._klines[(symbol, interval)] = rows
        log.info(f"[ws-backfill] {symbol} {interval} OK via {src} ({len(rows)} candle)")

    # ── internal: lifecycle WS ──
    def _run_forever(self):
        while not self._stop:
            try:
                self._connect()
            except Exception as e:
                log.warning(f"[ws] koneksi error: {e}")
            self._connected = False
            if self._stop:
                break
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, 30)

    def _connect(self):
        self._ws = websocket.WebSocketApp(
            BINANCE_WS_URL,
            on_open=self._on_open, on_message=self._on_message,
            on_error=self._on_error, on_close=self._on_close)
        self._ws.run_forever(ping_interval=180, ping_timeout=10)

    def _on_open(self, ws):
        self._connected = True
        self._backoff = 1
        self._last_msg = time.time()
        log.info("[ws] Binance Futures WS terhubung")
        self._send_subscribe(["!ticker@arr"])
        with self._lock:
            keys = list(self._klines.keys())
        if keys:
            streams = [f"{sym.lower()}@kline_{itv}" for sym, itv in keys]
            self._send_subscribe(streams)

    def _on_message(self, ws, raw):
        self._last_msg = time.time()
        try:
            msg = json.loads(raw)
        except Exception:
            return
        if isinstance(msg, list):
            self._handle_ticker_array(msg)
        elif isinstance(msg, dict) and msg.get("e") == "24hrTicker":
            self._handle_ticker_array([msg])
        elif isinstance(msg, dict) and msg.get("e") == "kline":
            self._handle_kline(msg)

    def _handle_ticker_array(self, arr):
        with self._lock:
            for t in arr:
                try:
                    sym = t["s"]
                    self._ticker[sym] = {
                        "symbol": sym, "price": float(t["c"]),
                        "qvol": float(t["q"]), "chg": float(t["P"]),
                    }
                except Exception:
                    continue

    def _handle_kline(self, msg):
        k = msg["k"]; sym = msg["s"]; itv = k["i"]
        key = (sym, itv)
        row = {"t": k["t"], "o": float(k["o"]), "h": float(k["h"]),
               "l": float(k["l"]), "c": float(k["c"]), "v": float(k["v"])}
        with self._lock:
            buf = self._klines.get(key)
            if buf is None:
                return   # belum di-backfill — abaikan sampai diminta
            if buf and buf[-1]["t"] == row["t"]:
                buf[-1] = row
            else:
                buf.append(row)

    def _on_error(self, ws, err):
        log.warning(f"[ws] error: {err}")

    def _on_close(self, ws, code, msg):
        self._connected = False
        log.warning(f"[ws] tertutup (code={code})")

    def _send_subscribe(self, streams):
        if not streams or not self._ws:
            return   # belum connect — akan di-resubscribe otomatis di _on_open
        try:
            with self._send_lock:
                self._ws.send(json.dumps({
                    "method":"SUBSCRIBE","params":streams,
                    "id": int(time.time()*1000) % 100000}))
            with self._lock:
                self._subscribed |= set(streams)
        except Exception as e:
            log.warning(f"[ws] gagal subscribe {streams}: {e}")

    def _subscribe_kline(self, symbol, interval):
        stream = f"{symbol.lower()}@kline_{interval}"
        with self._lock:
            already = stream in self._subscribed
        if not already:
            self._send_subscribe([stream])

    def _unsubscribe_kline(self, symbol, interval):
        stream = f"{symbol.lower()}@kline_{interval}"
        try:
            with self._send_lock:
                if self._ws:
                    self._ws.send(json.dumps({
                        "method":"UNSUBSCRIBE","params":[stream],
                        "id": int(time.time()*1000) % 100000}))
            with self._lock:
                self._subscribed.discard(stream)
        except Exception:
            pass


ws_feed = BinanceWSFeed()


# ── FUNGSI PUBLIK — signature SAMA PERSIS dgn sebelumnya, jadi seluruh
#    kode bot (scoring, monitor posisi, dsb) TIDAK perlu diubah sama sekali ──
def get_price(symbol):
    """Saat Binance pause: jangan hit REST fallback. Gunakan WS/local cache saja."""
    if _binance_is_scan_paused():
        p = ws_feed.get_price(symbol)
        return p
    try:
        return _binance_price(symbol)
    except Exception as e:
        log.warning(f"[price/binance] {symbol}: {e} — fallback")
        if _binance_is_scan_paused():
            return ws_feed.get_price(symbol)
    for _ in range(2):
        try:
            return _bybit_price(symbol)
        except Exception as e:
            log.warning(f"[price/bybit] {symbol}: {e}")
            time.sleep(1)
    if ws_feed.is_fresh():
        p = ws_feed.get_price(symbol)
        if p is not None:
            return p
    p = _coingecko_price(symbol)
    if p is not None:
        return p
    return None

def get_klines(symbol, interval, limit=250):
    """WS-first. Saat Binance global pause, hanya buffer WS yang sudah ada dipakai.
    Tidak boleh memicu backfill Binance/Bybit baru selama circuit breaker aktif.
    """
    if _binance_is_scan_paused():
        df = ws_feed.get_klines(symbol, interval, limit) if ws_feed.is_fresh() else pd.DataFrame()
        return df if df is not None else pd.DataFrame()
    ws_feed.ensure_symbol_interval(symbol, interval)
    if ws_feed.is_fresh():
        df = ws_feed.get_klines(symbol, interval, limit)
        if df is not None and not df.empty:
            return df
    try:
        df = _binance_klines(symbol, interval, limit)
        if not df.empty:
            return df
    except Exception as e:
        log.warning(f"[klines/binance] {symbol}: {e}")
        if _binance_is_scan_paused():
            return ws_feed.get_klines(symbol, interval, limit) if ws_feed.is_fresh() else pd.DataFrame()
    try:
        df = _bybit_klines(symbol, interval, limit)
        if not df.empty:
            log.info(f"[klines/bybit fallback] {symbol} {interval} OK")
            return df
    except Exception as e:
        log.warning(f"[klines/bybit] {symbol}: {e}")
    return pd.DataFrame()

last_scanned_coins = []
last_scanned_at = None
_last_scanned_lock = threading.Lock()

def get_top_coins():
    """Wrapper: panggil _get_top_coins_impl() lalu cache hasilnya ke
    last_scanned_coins — dipakai command /koin supaya bisa nampilin daftar
    koin yang di-scan TANPA perlu fetch ulang / ikut nambah scan_counter
    (yang dipakai buat hitung durasi ban)."""
    coins = _get_top_coins_impl()
    global last_scanned_coins, last_scanned_at
    with _last_scanned_lock:
        last_scanned_coins = coins
        last_scanned_at = time.time()
    return coins

def _get_top_coins_impl():
    """Ambil top coins. Saat Binance pause, seluruh scan berhenti.
    Fallback Bybit/WS hanya boleh dipakai ketika Binance tidak sedang dalam global pause.
    """
    if _binance_is_scan_paused():
        log.warning(f"[scan] DITAHAN — Binance cooldown aktif {_binance_cooldown_remaining():.0f}s")
        return []
    global scan_counter
    with ban_lock:
        scan_counter += 1
        to_unban = [s for s, (banned_at, dur) in banned_coins.items()
                    if scan_counter - banned_at >= dur]
        for s in to_unban:
            dur = banned_coins[s][1]
            del banned_coins[s]
            log.info(f"[unban] {s} kembali aktif setelah {dur} scan")
        cur_ban = set(banned_coins.keys())

    with positions_lock:
        active_syms = set(positions.keys())

    exclude_syms = cur_ban | active_syms

    # Binance REST
    try:
        coins = _binance_top_coins(exclude_syms)
        if coins:
            return coins
        if _binance_is_scan_paused():
            log.warning("[top_coins/binance] kosong karena circuit breaker aktif — TIDAK fallback.")
            return []
        log.warning("[top_coins/binance] kosong, coba Bybit...")
    except BinanceCooldownError:
        log.warning("[top_coins/binance] rate-limit/ban — TIDAK fallback, scan cycle dihentikan.")
        return []
    except Exception as e:
        log.warning(f"[top_coins/binance] {e}")
        if _binance_is_scan_paused():
            return []
    # Bybit fallback
    try:
        coins = _bybit_top_coins(exclude_syms)
        if coins:
            log.info(f"[top_coins/bybit fallback] {len(coins)} koin")
            return coins
        log.warning("[top_coins/bybit] kosong, coba WS...")
    except Exception as e:
        log.warning(f"[top_coins/bybit] {e} — coba WS...")
    # WS fallback TERAKHIR
    if ws_feed.is_fresh():
        raw = ws_feed.get_top_coins_raw()
        usdt = [
            t for t in raw
            if t["symbol"].endswith("USDT")
            and 0.0001 < t["price"] < MAX_PRICE
            and t["qvol"] > 5_000_000
            and abs(t["chg"]) < 15
            and t["symbol"] not in exclude_syms
        ]
        if usdt:
            usdt.sort(key=lambda x: x["qvol"], reverse=True)
            log.warning("[top_coins/ws fallback] REST Binance & Bybit gagal")
            return [t["symbol"] for t in usdt[:TOP_N_COINS]]
    return []


_PRICE_REFRESH_SEC = 10   # interval cek watchdog (detik)

def _price_cache_loop():
    """
    DULU: thread polling REST batch tiap 10 detik utk cache harga posisi.
    SEKARANG: REST (Binance→Bybit) adalah sumber data UTAMA di get_price/
    get_klines/get_top_coins; WS cuma buffer fallback TERAKHIR yang
    disiapkan di background. Karena WS bukan sumber utama lagi, hidup-
    matinya WS BUKAN kejadian penting bagi operasional bot — jadi TIDAK
    lagi dikirim ke Telegram tiap kali flap (dulu ini yang bikin spam
    notifikasi "WS pulih"/"WS terputus" berulang-ulang). Status WS tetap
    dicatat di log untuk keperluan debug, dan stream kline yang sudah
    tidak dipakai >30 menit tetap dibersihkan di sini.
    """
    was_fresh = None   # None = belum pernah dicek
    while True:
        try:
            fresh = ws_feed.is_fresh()
            if was_fresh is not None and was_fresh != fresh:
                if fresh:
                    log.info("[ws-watchdog] WS fallback tersedia lagi (buffer siap)")
                else:
                    log.info("[ws-watchdog] WS fallback tidak tersedia — tidak masalah, REST tetap sumber utama")
            was_fresh = fresh
            ws_feed.cleanup_stale_streams()
        except Exception as e:
            log.error(f"[ws-watchdog] {e}")
        time.sleep(_PRICE_REFRESH_SEC)

# ═════════════════════════════════════════════
# INDIKATOR
# ═════════════════════════════════════════════
def run_scan_once(chat_id):
    """
    Scan seluruh universe dan kembalikan SEMUA setup yang lolos threshold.
    main.py adalah tubuh/orchestrator: mengumpulkan hasil, menerapkan threshold,
    lalu menyerahkan setiap setup valid ke execution. Semua keputusan market
    (Entry/SL/TP/Trail/Confidence) tetap berasal dari strategy_logic.py.
    """
    if _binance_is_scan_paused():
        tg_send(chat_id, f"⏸️ <b>SCAN PAUSED</b> — Binance sedang rate-limit/ban. Posisi aktif tetap dipantau via WS.")
        return []
    tg_send(chat_id, f"🔍 Scanning {TOP_N_COINS} koin...")
    if _binance_is_scan_paused():
        return [], [], "Binance cooldown/rate-limit aktif — /analyze ditahan"
    try:
        symbols = get_top_coins()
    except BinanceCooldownError as e:
        tg_send(chat_id, f"⏸️ <b>Scan dihentikan</b> — Binance rate-limit/ban aktif.\n<code>{str(e)[:180]}</code>")
        return []
    except Exception as e:
        tg_send(chat_id, f"⚠️ Market data error: <code>{str(e)[:150]}</code>")
        return []
    if not symbols:
        if _binance_is_scan_paused():
            tg_send(chat_id, "⏸️ <b>Scan dihentikan</b> — Binance rate-limit/ban aktif. Posisi aktif tetap dipantau via WS.")
        else:
            tg_send(chat_id, "⚠️ Tidak ada koin tersedia untuk di-scan.")
        return []

    results = []
    for idx, sym in enumerate(symbols, 1):
        if _binance_is_scan_paused():
            log.warning("[scan] Binance pause aktif — scan cycle dihentikan di tengah jalan.")
            break
        log.info(f"[{idx:02d}/{len(symbols)}] {sym}")
        try:
            h1 = get_klines(sym, "1h", 250)
            m15 = get_klines(sym, "15m", 250)
            try:
                d1 = get_klines(sym, "1d", 100)
            except Exception:
                d1 = None
            r = full_analyze(h1, m15, d1, symbol=sym)
            if isinstance(r, dict):
                conf = float(r.get("confidence", 0) or 0)
                if conf >= STRATEGY_CONFIDENCE_THRESHOLD:
                    results.append(r)
                    log.info(f"[SIGNAL] {sym} {r.get('decision')} confidence={conf:.1f}")
                else:
                    log.info(f"[FILTER] {sym} confidence={conf:.1f} < {STRATEGY_CONFIDENCE_THRESHOLD}")
        except Exception as e:
            log.debug(f"[scan] {sym}: {e}")

        # Request throttle global sudah mengatur Binance; jeda kecil ini hanya
        # memberi scheduler/thread lain kesempatan berjalan.
        time.sleep(0.15)

    results.sort(key=lambda x: float(x.get("confidence", 0) or 0), reverse=True)
    if not results:
        tg_send(chat_id, f"⚠️ Tidak ada setup dengan confidence ≥ {STRATEGY_CONFIDENCE_THRESHOLD}%.")
        return []

    summary = "\n".join(
        f"• {r.get('symbol','?')} {r.get('decision','?')} — {float(r.get('confidence',0)):.0f}%"
        for r in results
    )
    tg_send(chat_id, f"✅ <b>{len(results)} sinyal lolos</b> (threshold {STRATEGY_CONFIDENCE_THRESHOLD}%)\n\n{summary}")
    return results




# ═════════════════════════════════════════════
# STATISTIK + BALANCE
# ═════════════════════════════════════════════

# ── Fee trading — dipakai update_stats() untuk PnL simulasi & real ────────
# Standar Binance USDT-M Futures VIP0, TANPA diskon BNB. SESUAIKAN kalau
# tier akun kamu beda (VIP lebih tinggi / diskon BNB aktif / dsb) —
# semakin akurat angka ini, semakin dekat statistik bot ke kenyataan.
ENTRY_FEE_PCT = 0.0002   # 0.02% — entry via limit order (biasanya maker)
EXIT_FEE_PCT  = 0.0005   # 0.05% — exit via SL/TP market-trigger (taker)
                            # P&L murni dari jarak SL/TP yang ditetapkan analisis:
                            #   TP hit → gain = posisi × (tp_dist / entry)
                            #   SL hit → loss = posisi × (sl_dist / entry)
                            # Nilai ini TIDAK mempengaruhi PENEMPATAN SL/TP —
                            # hanya memengaruhi simulasi saldo.
# POSITION_SIZE_PCT: SUDAH TIDAK DIPAKAI (lihat fix di update_stats di bawah)
# — dipertahankan sebagai konstanta supaya tidak menghapus definisi yang
# mungkin masih direferensikan dari luar, tapi update_stats() sekarang
# pakai MARGIN_USD × LEVERAGE (persis logika real trade), bukan ini lagi.
POSITION_SIZE_PCT = 100.0  # DEPRECATED — lihat catatan di atas

def update_stats(result, entry=None, sl_p=None, tp_p=None, close_price=None,
                 sym=None, decision=None, entry_time=None,
                 confidence=None, entry_label=None, rr=None, rsi=None,
                 struct_h1=None, d1_bias=None):
    """
    Hitung P&L simulasi murni dari jarak harga analisis (lihat komentar
    lama untuk detail model close_price). Tambahan: catat sym/decision/
    entry_time/exit_time + detail sinyal (confidence/entry_label/rr/rsi/
    struct_h1/d1_bias) ke pnl_history — bahan diagnosis strategy_logic.py
    tanpa perlu data tambahan lain (lihat /analyze).

    result: "tp" | "sl" | "trail" — "trail" = trailing stop mengunci
    profit (SL bergerak, tapi ditutup di atas entry utk BUY / di bawah
    entry utk SELL). Dihitung terpisah dari "sl" murni supaya statistik
    tidak salah mengira profit sebagai kerugian.
    """
    with stat_lock:
        stats["total"] += 1
        if result in ("tp", "sl", "trail"):
            stats[result] = stats.get(result, 0) + 1

        if not entry or tp_p is None:
            return

        balance      = stats["balance"]
        # ── FIX "buat semirip mungkin" ──────────────────────────────────
        # Sebelumnya: position_usd = balance × 100% — simulasi selalu
        # bertaruh SELURUH saldo tiap trade (full compounding), padahal
        # real trading pakai MARGIN_USD × LEVERAGE (jumlah dolar FIXED,
        # kecil, diatur via /margin & /leverage), TIDAK ikut membesar
        # walau saldo real sudah tumbuh. Ini bikin bentuk kurva ekuitas
        # simulasi sama sekali beda dari real (simulasi: compounding
        # agresif; real: flat sizing) — bukan cuma soal fee/entry lagi,
        # tapi soal skala taruhan itu sendiri.
        # Sekarang KEDUA mode pakai rumus yang SAMA PERSIS seperti real
        # trade sizing, supaya kalau kamu ubah /margin atau /leverage,
        # simulasi otomatis ikut menyesuaikan — selaras terus dengan real.
        position_usd = round(MARGIN_USD * LEVERAGE, 6)
        direction_sign = 1 if tp_p > entry else -1

        if close_price is not None:
            ref_price = close_price
        elif result == "tp":
            ref_price = tp_p
        elif result == "sl" and sl_p is not None:
            ref_price = sl_p
        else:
            return

        pnl_pct_raw = (ref_price - entry) / entry * direction_sign
        # ── FIX "simulasi tidak real / win rate kelewat bagus" ──────────
        # Sebelumnya PnL dihitung MURNI dari selisih harga — nol biaya
        # trading. Di real trading, Binance SELALU potong fee tiap kali
        # entry (limit order → biasanya maker) DAN exit (SL/TP → market-
        # trigger → taker), otomatis kepotong dari saldo asli. Simulasi
        # tidak pernah mengurangi ini, jadi untuk trade RR ketat (SL 1-2%
        # dari harga, khas bot ini), fee round-trip yang kelihatannya kecil
        # bisa membalik hasil "breakeven/rugi tipis di real" jadi "menang"
        # di simulasi — bias sistemik yang bikin win rate simulasi selalu
        # kelihatan lebih bagus dari kenyataan.
        #
        # Angka ENTRY_FEE_PCT/EXIT_FEE_PCT di bawah = tarif standar Binance
        # USDT-M Futures VIP0 tanpa diskon BNB. Kalau akun kamu VIP lebih
        # tinggi / pakai diskon BNB / fee-nya beda, SESUAIKAN angka ini
        # (dekat bagian atas file) supaya makin presisi ke kondisi akunmu.
        # Diterapkan ke SIMULASI *dan* REAL supaya keduanya konsisten
        # mencerminkan biaya riil (real trading sebenarnya sudah kepotong
        # otomatis di Binance — ini menyamakan angka yang DITAMPILKAN bot
        # dengan kenyataan itu, bukan menambah biaya baru yang sungguhan).
        fee_pct = ENTRY_FEE_PCT + EXIT_FEE_PCT
        pnl_pct = pnl_pct_raw - fee_pct
        pnl_usd = round(position_usd * pnl_pct, 4)
        pct     = round(pnl_pct * 100, 3)
        stats["balance"] = round(balance + pnl_usd, 4)
        stats["pnl_history"].append({
            "result": result, "pct": pct,
            "pnl_usd": pnl_usd, "balance_after": stats["balance"],
            "symbol": sym, "decision": decision,
            "entry_time": entry_time, "exit_time": time.time(),
            "entry": entry, "tp": tp_p, "sl": sl_p, "exit_price": ref_price,
            "confidence": confidence, "entry_label": entry_label, "rr": rr,
            "rsi": rsi, "struct_h1": struct_h1, "d1_bias": d1_bias,
        })

# Hitung alasan pending dibatalkan — biar bisa didiagnosis dari data,
# bukan tebak-tebakan (mis. "kenapa banyak batal?" jadi terjawab dari /stats).
pending_cancel_stats = {"tp_before_entry": 0, "expired": 0, "binance_reject": 0}
pending_cancel_lock = threading.Lock()

def _record_pending_cancel(reason_key):
    with pending_cancel_lock:
        pending_cancel_stats[reason_key] = pending_cancel_stats.get(reason_key, 0) + 1


def fmt_stats():
    with stat_lock:
        t, tp, sl = stats["total"], stats["tp"], stats["sl"]
        trail, bal = stats.get("trail", 0), stats["balance"]
        hist = list(stats["pnl_history"])

    if t == 0:
        return f"📊 <b>Statistik</b>\nBelum ada trade. Modal: ${STARTING_BALANCE:.2f}"

    wins = tp + trail   # trailing stop yang mengunci profit dihitung menang
    wr = wins/(wins+sl)*100 if (wins+sl) > 0 else 0
    pnl = round(bal - STARTING_BALANCE, 4)
    pnl_pct = round(pnl / STARTING_BALANCE * 100, 2)
    sgn = "+" if pnl >= 0 else ""

    hist_str = "\n".join(
        f"  {'✅' if h['result'] in ('tp','trail') else '❌'} {'+' if h['pnl_usd']>=0 else ''}{h['pct']:.2f}% "
        f"→ ${h['balance_after']:.4f}"
        for h in reversed(hist[-5:])
    ) or "  (belum ada)"

    with pending_cancel_lock:
        pc = dict(pending_cancel_stats)
    total_cancel = sum(pc.values())
    cancel_line = ""
    if total_cancel > 0:
        cancel_line = (f"\n\n⏭ Pending batal: {total_cancel} total\n"
                        f"  TP sebelum entry: {pc['tp_before_entry']} | "
                        f"Expired: {pc['expired']} | Ditolak Binance: {pc['binance_reject']}")

    return (
        f"📊 <b>Statistik</b> — {t} trade | TP {tp} SL {sl} Trail {trail}\n"
        f"Win Rate: <b>{wr:.1f}%</b> (TP+Trail vs SL)\n\n"
        f"Modal: ${STARTING_BALANCE:.2f} → Saldo: <b>${bal:.4f}</b> "
        f"({sgn}{pnl_pct:.2f}%)\n\n"
        f"5 terakhir:\n{hist_str}\n\n"
        f"🚫 Banned: {len(banned_coins)}"
        f"{cancel_line}"
    )

def fmt_backtest():
    """20 trade terakhir: koin, arah, hasil, entry/TP/SL, jam masuk-keluar — bahan evaluasi."""
    with stat_lock:
        hist = list(stats["pnl_history"])
    if not hist:
        return "📋 <b>Backtest</b>\nBelum ada trade."

    lines = []
    for h in reversed(hist):
        em  = "✅" if h["result"] in ("tp", "trail") else "❌"
        dec = h.get("decision") or "?"
        sym = h.get("symbol") or "?"
        et  = h.get("entry_time")
        xt  = h.get("exit_time")
        t_in  = datetime.fromtimestamp(et, WIB).strftime("%d/%m/%Y %H:%M") if et else "?"
        t_out = datetime.fromtimestamp(xt, WIB).strftime("%d/%m/%Y %H:%M") if xt else "?"
        sgn = "+" if h["pnl_usd"] >= 0 else ""
        entry_v, tp_v, sl_v = h.get("entry"), h.get("tp"), h.get("sl")
        # Untuk trade "trail", SL yang relevan ditampilkan adalah SL
        # TRAILING aktual saat ditutup (exit_price), bukan SL original —
        # supaya konsisten dgn PnL yang tercatat (sudah untung, bukan rugi).
        sl_display = h.get("exit_price") if h.get("result") == "trail" else sl_v
        levels = (f"Entry: <code>{entry_v:.6g}</code> | TP: <code>{tp_v:.6g}</code> | "
                  f"SL: <code>{sl_display:.6g}</code>\n"
                  if entry_v is not None and tp_v is not None and sl_display is not None else "")
        lines.append(
            f"{em} <b>{sym}</b> {dec} | {h['result'].upper()} {sgn}{h['pct']:.2f}%\n"
            f"{levels}"
            f"{t_in}→{t_out}"
        )
    return f"📋 <b>Backtest ({len(hist)} trade terakhir)</b>\n\n" + "\n\n".join(lines)

# ============================================================
# ANALYZE — DIAGNOSTIC SNAPSHOT (DUA FILE: MD + CSV)
# ============================================================
def _json_compact(value):
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            return str(value)
    return value


def _analysis_row(symbol, sig=None, status="ok", error=""):
    """Normalisasi hasil full_analyze() menjadi satu baris diagnostik."""
    sig = sig if isinstance(sig, dict) else {}
    row = {
        "symbol": symbol,
        "status": status,
        "error": error,
        "decision": sig.get("decision", ""),
        "confidence": sig.get("confidence", ""),
        "above_threshold": bool(
            isinstance(sig.get("confidence"), (int, float))
            and float(sig.get("confidence")) >= STRATEGY_CONFIDENCE_THRESHOLD
        ),
        "entry": sig.get("entry", ""),
        "entry_label": sig.get("entry_label", ""),
        "entry_state": sig.get("entry_state", ""),
        "sl": sig.get("sl", ""),
        "tp": sig.get("tp", ""),
        "rr": sig.get("rr", ""),
        "atr": sig.get("atr", ""),
        "risk_atr": sig.get("risk_atr", ""),
        "rsi": sig.get("rsi", ""),
        "struct_h1": sig.get("struct_h1", ""),
        "struct_m15": sig.get("struct_m15", ""),
        "d1_bias": sig.get("d1_bias", ""),
        "htf_bias": sig.get("htf_bias", ""),
        "h1_bias": sig.get("h1_bias", ""),
        "choch_m15": _json_compact(sig.get("choch_m15", {})),
        "choch_h1": _json_compact(sig.get("choch_h1", {})),
        "cisd_m15": _json_compact(sig.get("cisd_m15", {})),
        "failed_retest": _json_compact(sig.get("failed_retest", {})),
        "poi_reacted": sig.get("poi_reacted", ""),
        "htf_overlap": sig.get("htf_overlap", ""),
        "selected_sweep": sig.get("selected_sweep", ""),
        "sweep_context": _json_compact(sig.get("sweep_context", {})),
        "trigger_count": sig.get("trigger_count", ""),
        "entry_confirmation": _json_compact(sig.get("entry_confirmation", {})),
        "tp_sl_reason": sig.get("tp_sl_reason", ""),
    }
    return row


def _analyze_snapshot():
    """Scan satu universe sekali dan simpan SEMUA hasil strategy, termasuk yang di bawah threshold."""
    try:
        symbols = get_top_coins()
    except Exception as e:
        return [], [], f"market-data error: {e}"
    if not symbols:
        return [], [], "universe kosong"

    rows = []
    passing = []
    for idx, sym in enumerate(symbols, 1):
        log.info(f"[analyze {idx:02d}/{len(symbols)}] {sym}")
        try:
            h1 = get_klines(sym, "1h", 250)
            m15 = get_klines(sym, "15m", 250)
            try:
                d1 = get_klines(sym, "1d", 100)
            except Exception:
                d1 = None
            sig = full_analyze(h1, m15, d1, symbol=sym)
            if isinstance(sig, dict):
                row = _analysis_row(sym, sig)
                rows.append(row)
                if row["above_threshold"]:
                    passing.append(sig)
            else:
                rows.append(_analysis_row(sym, status="no_setup"))
        except Exception as e:
            rows.append(_analysis_row(sym, status="error", error=str(e)[:250]))
        time.sleep(0.15)
    return rows, passing, ""


def _analyze_runtime_stats():
    with stat_lock:
        hist = list(stats["pnl_history"])
        balance = stats["balance"]
    if not hist:
        return {
            "trades": 0, "balance": balance, "net": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "max_dd": 0.0, "expectancy": 0.0,
        }
    wins = [float(t.get("pnl_usd", 0.0)) for t in hist if float(t.get("pnl_usd", 0.0)) >= 0]
    losses = [float(t.get("pnl_usd", 0.0)) for t in hist if float(t.get("pnl_usd", 0.0)) < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    equity = [STARTING_BALANCE] + [float(t.get("balance_after", STARTING_BALANCE)) for t in hist]
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)
    net = gross_profit - gross_loss
    return {
        "trades": len(hist),
        "balance": balance,
        "net": net,
        "win_rate": len(wins) / len(hist) * 100.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
        "max_dd": max_dd,
        "expectancy": net / len(hist),
    }


def _write_analyze_csv(rows):
    path = "/tmp/analyze_data.csv"
    cols = [
        "symbol", "status", "error", "decision", "confidence", "above_threshold",
        "entry", "entry_label", "entry_state", "sl", "tp", "rr", "atr", "risk_atr", "rsi",
        "struct_h1", "struct_m15", "d1_bias", "htf_bias", "h1_bias", "choch_m15", "choch_h1",
        "cisd_m15", "failed_retest", "poi_reacted", "htf_overlap", "selected_sweep", "sweep_context",
        "trigger_count", "entry_confirmation", "tp_sl_reason",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def _write_analyze_report(rows, passing, universe_error=""):
    path = "/tmp/analyze_report.md"
    now = datetime.now(WIB).strftime("%Y-%m-%d %H:%M:%S WIB")
    rt = _analyze_runtime_stats()
    analyzed = len(rows)
    setup_rows = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] == "error"]
    no_setup = [r for r in rows if r["status"] == "no_setup"]
    passing_sorted = sorted(passing, key=lambda x: float(x.get("confidence", 0) or 0), reverse=True)

    lines = [
        "# SMCAutoTrade — Analysis Report",
        "",
        f"**Waktu snapshot:** {now}",
        f"**Confidence threshold:** {STRATEGY_CONFIDENCE_THRESHOLD}%",
        "**Mode:** Diagnostic snapshot saat ini; bukan backtest historis.",
        "",
        "## Ringkasan Market",
        "",
        "| Metrik | Nilai |",
        "|---|---:|",
        f"| Koin dipindai | {analyzed} |",
        f"| Setup dari strategy | {len(setup_rows)} |",
        f"| Lolos threshold | {len(passing_sorted)} |",
        f"| Tidak ada setup | {len(no_setup)} |",
        f"| Error analisis | {len(errors)} |",
        f"| Threshold | {STRATEGY_CONFIDENCE_THRESHOLD}% |",
        "",
        "## Kandidat Lolos",
        "",
        "| Rank | Koin | Decision | Confidence | Entry | SL | TP | RR | Entry Type |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    if passing_sorted:
        for i, s in enumerate(passing_sorted, 1):
            lines.append(
                f"| {i} | {s.get('symbol','')} | {s.get('decision','')} | "
                f"{float(s.get('confidence',0) or 0):.0f}% | {s.get('entry','')} | {s.get('sl','')} | "
                f"{s.get('tp','')} | {float(s.get('rr',0) or 0):.2f} | {s.get('entry_label','')} |"
            )
    else:
        lines.append("| — | Tidak ada | — | — | — | — | — | — | — |")

    lines += [
        "",
        "## Diagnostik Per Koin",
        "",
        "| Koin | Status | Direction | Conf. | D1 | H1 | M15 | Sweep | POI React | RSI | RR |",
        "|---|---|---|---:|---|---|---|---|---|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: float(x.get("confidence") or -1), reverse=True):
        lines.append(
            f"| {r['symbol']} | {r['status']} | {r.get('decision','')} | "
            f"{r.get('confidence','')} | {r.get('d1_bias','')} | {r.get('struct_h1','')} | "
            f"{r.get('struct_m15','')} | {r.get('selected_sweep','')} | {r.get('poi_reacted','')} | "
            f"{r.get('rsi','')} | {r.get('rr','')} |"
        )

    lines += [
        "",
        "## Runtime Trading Snapshot",
        "",
        "| Metrik | Nilai |",
        "|---|---:|",
        f"| Trade tercatat di runtime | {rt['trades']} |",
        f"| Balance | ${rt['balance']:.4f} |",
        f"| Net PnL | ${rt['net']:.4f} |",
        f"| Win rate | {rt['win_rate']:.2f}% |",
        f"| Profit factor | {rt['profit_factor']:.3f} |",
        f"| Max drawdown | ${rt['max_dd']:.4f} |",
        f"| Expectancy/trade | ${rt['expectancy']:.4f} |",
        "",
        "## Interpretasi",
        "",
        "- `Confidence` adalah output strategy; threshold hanya dipakai engine untuk menentukan setup yang layak diproses.",
        "- `Entry`, `SL`, `TP`, dan Trail tetap berasal dari strategy; report ini tidak menghitung ulang level tersebut.",
        "- Data diagnostik lengkap per koin tersedia di `analyze_data.csv`.",
    ]
    if universe_error:
        lines += ["", f"> **Market data warning:** {universe_error}"]
    if errors:
        lines += ["", "## Error Analysis", ""]
        for r in errors:
            lines.append(f"- **{r['symbol']}**: {r['error']}")
    path = "/tmp/analyze_report.md"
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

# ============================================================
# END ANALYZE
# ============================================================

def fmt_signal_msg(sig):
    d=sig.get("decision","?"); em="🟢" if d=="BUY" else "🔴" if d=="SELL" else "⚪"
    return (f"📡 <b>{sig.get('symbol','?')}</b> | {em} <b>{d}</b> | Confidence: {sig.get('confidence','—')}%\n"
            f"Entry: <code>{sig.get('entry',0):.8g}</code> | TP: <code>{sig.get('tp',0):.8g}</code> | SL: <code>{sig.get('sl',0):.8g}</code>")



# ═════════════════════════════════════════════
# MULTI-POSITION BROADCASTER
# ═════════════════════════════════════════════
# MAX_POSITIONS dikontrol lewat /max — lihat konstanta di bagian atas file
MONITOR_INTERVAL = 15 * 60  # cek posisi tiap 15 menit (detik)

positions_lock = threading.Lock()
positions: dict = {}   # {sym: {signal, entry, tp, sl, entry_time, thread}}

def close_position(sym, result, close_price=None):
    """Tutup posisi, catat statistik, ban koin sementara, kirim notif."""
    global active_trade
    with positions_lock:
        pos = positions.pop(sym, None)
    if pos is None: return

    sig   = pos["signal"]
    entry = pos["entry"]
    sl_p  = sig["sl"]
    tp_p  = sig["tp"]
    cid   = pos["chat_id"]

    update_stats(result, entry=entry, sl_p=sl_p, tp_p=tp_p, close_price=close_price,
                 sym=sym, decision=sig.get("decision"), entry_time=pos.get("entry_time"),
                 confidence=sig.get("confidence"), entry_label=sig.get("entry_label"),
                 rr=sig.get("rr"), rsi=sig.get("rsi"), struct_h1=sig.get("struct_h1"),
                 d1_bias=sig.get("d1_bias"))
    _ban_coin(sym, f"trade closed ({result})", duration=BAN_DURATION_TRADE_CLOSED)

    # Update active_trade jika ini yang sedang dipantau
    with positions_lock:
        if not positions:
            active_trade = None

    with stat_lock:
        last = stats["pnl_history"][-1] if stats["pnl_history"] else None

    emoji = {"tp":"🎯","sl":"🛑","trail":"🔒"}.get(result,"❓")
    label = {"tp":"TAKE PROFIT","sl":"STOP LOSS","trail":"TRAILING STOP"}.get(result, result.upper())
    detail = ""
    if last and last.get("symbol") == sym:
        sgn = "+" if last["pct"] >= 0 else ""
        detail = (f"Entry: <code>{last['entry']:.6g}</code> → Exit: <code>{last['exit_price']:.6g}</code>\n"
                   f"Hasil: <b>{sgn}{last['pct']:.2f}%</b> (${sgn}{last['pnl_usd']:.4f})\n\n")
    tg_send(cid, f"{emoji} <b>{label}</b> — {sym}\n\n{detail}" + fmt_stats())


def check_tp_sl_order(sym, tp_p, sl_p, is_buy, lookback_min=15):
    """
    Ambil candle M1 dalam N menit terakhir, periksa urutan:
    mana yang kena duluan — TP atau SL?

    Return: "tp", "sl", atau None (tidak ada yang tersentuh)
    """
    try:
        df = get_klines(sym, "1m", lookback_min + 2)
        if df is None or df.empty: return None

        # Ambil hanya candle dalam lookback_min menit terakhir
        df = df.tail(lookback_min)

        for _, row in df.iterrows():
            high = row["high"]
            low  = row["low"]
            if is_buy:
                # Untuk BUY: TP di atas, SL di bawah
                # Kalau high >= TP dan low <= SL di candle yang sama → cek open lebih dekat ke mana
                if high >= tp_p and low <= sl_p:
                    # Harga open candle ini lebih dekat ke TP atau SL?
                    dist_tp = abs(row["open"] - tp_p)
                    dist_sl = abs(row["open"] - sl_p)
                    return "tp" if dist_tp < dist_sl else "sl"
                elif high >= tp_p:
                    return "tp"
                elif low <= sl_p:
                    return "sl"
            else:
                # Untuk SELL: TP di bawah, SL di atas
                if low <= tp_p and high >= sl_p:
                    dist_tp = abs(row["open"] - tp_p)
                    dist_sl = abs(row["open"] - sl_p)
                    return "tp" if dist_tp < dist_sl else "sl"
                elif low <= tp_p:
                    return "tp"
                elif high >= sl_p:
                    return "sl"
    except Exception as e:
        log.debug(f"[check_tp_sl_order] {sym}: {e}")
    return None




# ============================================================
# STRATEGY DISPATCH — ENGINE TIDAK MEMILIKI OTAK TRADING
# ============================================================

def _strategy_position_update(sym,pos):
    if _binance_is_scan_paused():
        return None
    manager=globals().get("manage_position")
    if not callable(manager): return None
    try:
        m15=get_klines(sym,"15m",250); h1=get_klines(sym,"1h",250)
        try: d1=get_klines(sym,"1d",100)
        except Exception: d1=None
        return manager(state=dict(pos),df_m15=m15,df_h1=h1,df_d1=d1,symbol=sym)
    except Exception as e:
        log.warning(f"[strategy/manage] {sym}: {e}"); return None

def _apply_strategy_update(sym,pos,update):
    if not isinstance(update,dict): return False
    sig=pos["signal"]; changed=False
    if update.get("tp") is not None:
        sig["tp"]=float(update["tp"]); changed=True
    if update.get("sl") is not None:
        new=float(update["sl"]); old=float(pos.get("current_sl",sig["sl"]))
        buy=sig["decision"]=="BUY"
        if (new>old) if buy else (new<old):
            pos["current_sl"]=new; sig["sl"]=new; changed=True
    return changed

def monitor_position(sym,pos):
    """Execution monitor. Tidak menentukan Entry/TP/SL/Trail."""
    next_strategy=0
    while True:
        with positions_lock:
            if sym not in positions:return
            pos=positions[sym]
        if pos.get("timeout_flag"):
            price=get_price(sym) or pos["entry"]; buy=pos["signal"]["decision"]=="BUY"
            result="tp" if (price-pos["entry"])*(1 if buy else -1)>=0 else "sl"
            close_position(sym,result,close_price=price); return
        if time.time()>=next_strategy:
            upd=_strategy_position_update(sym,pos); next_strategy=time.time()+STRATEGY_MANAGE_INTERVAL
            if isinstance(upd,dict):
                if upd.get("close"):
                    price=upd.get("close_price") or get_price(sym) or pos["entry"]
                    reason=str(upd.get("reason") or "strategy")
                    close_position(sym,"trail" if reason=="trail" else "strategy",close_price=price); return
                _apply_strategy_update(sym,pos,upd)
        price=get_price(sym)
        if price is None: time.sleep(MONITOR_SLEEP); continue
        sig=pos["signal"]; buy=sig["decision"]=="BUY"; tp=sig.get("tp"); sl=pos.get("current_sl",sig.get("sl"))
        hit_tp=tp is not None and ((price>=tp) if buy else (price<=tp))
        hit_sl=sl is not None and ((price<=sl) if buy else (price>=sl))
        if hit_tp or hit_sl:
            result="tp" if hit_tp and not hit_sl else "sl"
            if hit_tp and hit_sl: result=check_tp_sl_order(sym,tp,sl,buy,3) or "tp"
            close_position(sym,result,close_price=tp if result=="tp" else sl); return
        time.sleep(MONITOR_SLEEP)

def _open_position(sym,signal,actual_entry,chat_id,mode_label="strategy"):
    buy=signal["decision"]=="BUY"; sl=signal.get("sl"); tp=signal.get("tp")
    if sl is None or tp is None:
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,"strategy tidak mengirim SL/TP"); return
    valid=(sl<actual_entry<tp) if buy else (tp<actual_entry<sl)
    if not valid:
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,"level strategy invalid")
        tg_send(chat_id,f"⚠️ <b>Skip {sym}</b> — geometri level strategy invalid.")
        return
    with positions_lock:
        if sym not in positions:return
        pos=positions[sym]
        pos.update({"entry":actual_entry,"entry_time":time.time(),"status":"active",
                    "timeout_flag":False,"current_sl":sl})
    tg_send(chat_id,f"⚡ <b>ENTRY {mode_label.upper()}</b> — {sym}\n"
                    f"Entry: <code>{actual_entry:.8g}</code>\n"
                    f"TP: <code>{tp:.8g}</code> | SL: <code>{sl:.8g}</code>")
    threading.Thread(target=monitor_position,args=(sym,pos),daemon=True).start()


# ============================================================
# REAL TRADE — alur pending order, monitoring posisi, auto-stop
# ============================================================

def _open_pending_real(sym,signal,chat_id):
    if _binance_is_scan_paused():
        log.warning(f"[entry] {sym} ditahan — Binance pause aktif")
        return
    buy=signal["decision"]=="BUY"; entry=signal["entry"]; sl=signal.get("sl"); tp=signal.get("tp")
    if sl is None or tp is None:
        _ban_coin(sym,"strategy tidak mengirim SL/TP"); return
    valid=(sl<entry<tp) if buy else (tp<entry<sl)
    if not valid:
        _ban_coin(sym,"geometri strategy invalid"); tg_send(chat_id,f"⏭ <b>Skip {sym}</b> — geometri strategy invalid."); return
    side="BUY" if buy else "SELL"
    with positions_lock:
        if sym in positions or len(positions)>=MAX_POSITIONS:return
        positions[sym]={"signal":signal,"entry":entry,"chat_id":chat_id,"entry_time":None,
                        "timeout_flag":False,"status":"pending"}
    try:
        avail,_=get_real_balance()
        if avail is not None and avail<MARGIN_USD: raise RuntimeError(f"saldo ${avail:.2f} < margin ${MARGIN_USD:.2f}")
        qty,margin,bumped=calc_auto_quantity(sym,entry,MARGIN_USD,LEVERAGE)
        if qty is None: raise RuntimeError("quantity di bawah minimum Binance")
        set_leverage(sym,LEVERAGE); order=place_limit_order(sym,side,qty,entry)
        with positions_lock: positions[sym].update({"order_id":order["orderId"],"quantity":qty,"margin_used":margin})
        tg_send(chat_id,f"🎯 <b>PENDING ORDER REAL</b> — {sym}\n\n{fmt_signal_msg(signal)}")
        threading.Thread(target=_wait_entry_real,args=(sym,signal,chat_id,order["orderId"]),daemon=True).start()
    except Exception as e:
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,f"gagal pasang order real ({e})"); tg_send(chat_id,f"⚠️ <b>Skip {sym}</b> — {e}")



def _wait_entry_real(sym,signal,chat_id,order_id):
    deadline=time.time()+8*3600
    while time.time()<deadline:
        with positions_lock:
            if sym not in positions:return
            if positions[sym].get("timeout_flag"):
                try: cancel_order(sym,order_id); cancel_all_algo_orders(sym)
                except Exception: pass
                positions.pop(sym,None); return
        try: order=get_order_status(sym,order_id)
        except Exception as e:
            log.warning(f"[wait_entry_real] {sym}: {e}"); time.sleep(REAL_TRADE_POLL_SLEEP); continue
        status=order.get("status")
        if status=="FILLED":
            actual=float(order.get("avgPrice") or 0) or signal["entry"]
            _open_position_real(sym,signal,actual,chat_id,order); return
        if status in ("CANCELED","EXPIRED","REJECTED"):
            with positions_lock: positions.pop(sym,None)
            _ban_coin(sym,f"order {status.lower()}"); _record_pending_cancel("binance_reject"); return
        time.sleep(REAL_TRADE_POLL_SLEEP)
    try: cancel_order(sym,order_id)
    except Exception: pass
    with positions_lock: positions.pop(sym,None)
    _ban_coin(sym,"pending expired"); _record_pending_cancel("expired")



def _emergency_close(sym, is_buy, qty, chat_id, reason):
    """Auto-out: tutup posisi market SEKARANG. Fallback untuk kondisi
    bahaya (geometri invalid / harga sudah lewat SL) setelah order FILLED."""
    try:
        place_market_order(sym, "SELL" if is_buy else "BUY", qty, reduce_only=True)
        tg_send(chat_id, f"🚨 <b>AUTO-OUT</b> — {sym}\nAlasan: {reason}\nPosisi ditutup market segera.")
    except Exception as e:
        tg_send(chat_id, f"🚨 <b>GAGAL AUTO-OUT</b> — {sym}: {e}\n"
                          f"❗ CEK MANUAL SEGERA DI BINANCE, posisi mungkin masih terbuka!")
    with positions_lock:
        positions.pop(sym, None)
    _ban_coin(sym, reason)


def _open_position_real(sym,signal,actual_entry,chat_id,order_info):
    buy=signal["decision"]=="BUY"; sl=signal.get("sl"); tp=signal.get("tp")
    qty=abs(float(order_info.get("executedQty",0)))
    if not qty:
        with positions_lock: qty=positions.get(sym,{}).get("quantity",0)
    if sl is None or tp is None:
        _emergency_close(sym,buy,qty,chat_id,"strategy tidak mengirim SL/TP"); return
    valid=(sl<actual_entry<tp) if buy else (tp<actual_entry<sl)
    if not valid:
        _emergency_close(sym,buy,qty,chat_id,"level strategy invalid setelah fill"); return
    tick=get_symbol_filters(sym)["tickSize"]; sl=round_to_tick(sl,tick); tp=round_to_tick(tp,tick)
    last=None; tpo=slo=None
    for attempt in range(1,4):
        try:
            t,s=place_tp_sl(sym,buy,tp,sl,qty); tpo=t["algoId"]; slo=s["algoId"]; last=None; break
        except Exception as e:
            last=e; log.warning(f"[open_position_real] proteksi {attempt}/3 gagal: {e}")
            if attempt<3: time.sleep(2)
    if last is not None or slo is None:
        if isinstance(last, BinanceCooldownError):
            with positions_lock:
                if sym in positions:
                    positions[sym].update({"entry": actual_entry, "entry_time": time.time(),
                                           "status": "active", "current_sl": sl, "quantity": qty,
                                           "tp_order_id": None, "sl_order_id": None})
            _queue_pending_protection(sym, buy, sl, tp, qty)
            tg_send(chat_id, f"⏸️ <b>PROTEKSI DITUNDA</b> — {sym}\nBinance sedang rate-limit/ban. TP/SL dicatat dan akan dipasang setelah recovery +60 detik.")
            threading.Thread(target=monitor_position_real,args=(sym,positions[sym]),daemon=True).start()
            return
        _emergency_close(sym,buy,qty,chat_id,f"gagal pasang SL ({last})"); return
    with positions_lock:
        if sym not in positions:return
        positions[sym].update({"entry":actual_entry,"entry_time":time.time(),"status":"active",
                               "current_sl":sl,"quantity":qty,"tp_order_id":tpo,"sl_order_id":slo})
    tg_send(chat_id,f"⚡ <b>ENTRY REAL</b> — {sym}\nEntry: <code>{actual_entry:.8g}</code>\n"
                     f"TP: <code>{tp:.8g}</code> | SL: <code>{sl:.8g}</code>")
    threading.Thread(target=monitor_position_real,args=(sym,positions[sym]),daemon=True).start()



def _infer_close_reason(tp_algo_id, sl_algo_id):
    """Cek algo order mana yang TRIGGERED/FINISHED untuk tahu sebab posisi
    closed (tp/sl). TP/SL sekarang algo order (lihat place_tp_sl), jadi
    query-nya lewat get_algo_order_status, bukan get_order_status biasa."""
    tp_status = sl_status = None
    try:
        if tp_algo_id: tp_status = get_algo_order_status(tp_algo_id).get("algoStatus")
    except Exception: pass
    try:
        if sl_algo_id: sl_status = get_algo_order_status(sl_algo_id).get("algoStatus")
    except Exception: pass
    if tp_status in ("TRIGGERED", "FINISHED"): return "tp"
    if sl_status in ("TRIGGERED", "FINISHED"): return "sl"
    return "unknown"


def monitor_position_real(sym,pos):
    next_strategy=0
    while True:
        with positions_lock:
            if sym not in positions:return
            pos=positions[sym]
        if pos.get("timeout_flag"):
            qty=pos.get("quantity",0); buy=pos["signal"]["decision"]=="BUY"; price=get_price(sym) or pos["entry"]
            try: cancel_all_algo_orders(sym); place_market_order(sym,"SELL" if buy else "BUY",qty,reduce_only=True)
            except Exception as e: log.error(f"[monitor_real] manual close {sym}: {e}")
            close_position(sym,"strategy",close_price=price); return
        if _binance_is_scan_paused():
            # Saat Binance pause, hanya pantau harga dari WS. Jangan kirim/cancel/query REST.
            price = get_price(sym)
            if price is not None:
                sig = pos["signal"]; buy = sig["decision"] == "BUY"
                tp = sig.get("tp"); sl = pos.get("current_sl", sig.get("sl"))
                hit_tp = tp is not None and ((price >= tp) if buy else (price <= tp))
                hit_sl = sl is not None and ((price <= sl) if buy else (price >= sl))
                if hit_tp or hit_sl:
                    log.critical(f"[monitor_real] {sym} menyentuh level saat Binance pause; REST ditahan. Cek Binance/WebSocket/account state.")
            time.sleep(MONITOR_SLEEP)
            continue
        try: real=get_real_position(sym)
        except BinanceCooldownError:
            time.sleep(REAL_TRADE_POLL_SLEEP); continue
        except Exception as e: log.warning(f"[monitor_real] {sym}: {e}"); time.sleep(REAL_TRADE_POLL_SLEEP); continue
        if real is None:
            reason=_infer_close_reason(pos.get("tp_order_id"),pos.get("sl_order_id"))
            sig=pos["signal"]; close= sig.get("tp") if reason=="tp" else pos.get("current_sl",sig.get("sl"))
            cancel_all_algo_orders(sym); close_position(sym,reason if reason in ("tp","sl") else "strategy",close_price=close); return
        live=abs(float(real.get("positionAmt",0)))
        if live: pos["quantity"]=live
        if time.time()>=next_strategy:
            upd=_strategy_position_update(sym,pos); next_strategy=time.time()+STRATEGY_MANAGE_INTERVAL
            if isinstance(upd,dict):
                if upd.get("close"):
                    price=upd.get("close_price") or get_price(sym) or pos["entry"]; buy=pos["signal"]["decision"]=="BUY"
                    try: cancel_all_algo_orders(sym); place_market_order(sym,"SELL" if buy else "BUY",pos["quantity"],reduce_only=True)
                    except Exception as e: log.error(f"[strategy close] {sym}: {e}")
                    close_position(sym,"trail" if upd.get("reason")=="trail" else "strategy",close_price=price); return
                oldsl=pos.get("current_sl",pos["signal"].get("sl")); oldtp=pos["signal"].get("tp")
                _apply_strategy_update(sym,pos,upd)
                newsl=pos.get("current_sl",pos["signal"].get("sl")); newtp=pos["signal"].get("tp")
                if newsl!=oldsl or newtp!=oldtp:
                    try:
                        if _binance_is_scan_paused():
                            _queue_pending_trail(sym, newsl, newtp, pos.get("quantity",0), reason="strategy", side=pos["signal"]["decision"])
                            log.warning(f"[trail] {sym} queued — Binance pause aktif; SL={newsl} TP={newtp}")
                        else:
                            cancel_algo_order(pos.get("tp_order_id")); cancel_algo_order(pos.get("sl_order_id"))
                            t,s=place_tp_sl(sym,pos["signal"]["decision"]=="BUY",newtp,newsl,pos["quantity"])
                            pos["tp_order_id"]=t["algoId"]; pos["sl_order_id"]=s["algoId"]
                            _clear_pending_trail(sym)
                    except BinanceCooldownError:
                        _queue_pending_trail(sym, newsl, newtp, pos.get("quantity",0), reason="binance_cooldown", side=pos["signal"]["decision"])
                    except Exception as e: log.warning(f"[strategy/manage real] {sym}: {e}")
        time.sleep(REAL_TRADE_POLL_SLEEP)



def _queue_pending_protection(sym, buy, sl, tp, qty):
    with _pending_protections_lock:
        _pending_protections[sym] = {
            "side": "BUY" if buy else "SELL", "sl": sl, "tp": tp,
            "quantity": qty, "updated_at": time.time()
        }


def _clear_pending_protection(sym):
    with _pending_protections_lock:
        _pending_protections.pop(sym, None)


def _resume_binance_and_flush_pending(chat_id=None):
    'Strict Binance recovery. Scanner tetap PAUSED sampai seluruh recovery gate sukses.'
    global _binance_recovering, _binance_scan_paused, _binance_pause_reason
    if _binance_cooldown_remaining() > 0:
        return False
    with _binance_pause_lock:
        _binance_recovering = True
        _binance_scan_paused = True
        _binance_pause_reason = 'recovery in progress'
    log.info('[BINANCE RESUME] Cooldown + grace selesai. Strict sync & protection recovery dimulai...')
    failures = []
    try:
        # 1) STRICT POSITION SYNC
        with positions_lock:
            items = list(positions.items())
        for sym, pos in items:
            if pos.get('status') != 'active':
                continue
            try:
                real = get_real_position(sym)
                if real is None:
                    failures.append(f'{sym}: position sync returned no position')
                    log.error(f'[resume] SYNC GAGAL {sym}: posisi tidak terkonfirmasi')
                    continue
                live_qty = abs(float(real.get('positionAmt', 0)))
                if live_qty <= 0:
                    failures.append(f'{sym}: position quantity=0')
                    log.error(f'[resume] SYNC GAGAL {sym}: quantity=0')
                    continue
                with positions_lock:
                    if sym in positions:
                        positions[sym]['quantity'] = live_qty
                        positions[sym]['exchange_synced_at'] = time.time()
            except Exception as e:
                failures.append(f'{sym}: sync {e}')
                log.error(f'[resume] SYNC GAGAL {sym}: {e}')

        # 2) STRICT PENDING PROTECTION
        with _pending_protections_lock:
            protections = [(sym, dict(v)) for sym, v in _pending_protections.items()]
        for sym, pr in protections:
            try:
                with positions_lock:
                    pos = positions.get(sym)
                if not pos or pos.get('status') != 'active':
                    _clear_pending_protection(sym)
                    continue
                qty = pos.get('quantity') or pr.get('quantity')
                buy = pr.get('side') == 'BUY'
                if not qty or pr.get('tp') is None or pr.get('sl') is None:
                    raise RuntimeError('pending protection tidak lengkap')
                t, s = place_tp_sl(sym, buy, pr['tp'], pr['sl'], qty)
                with positions_lock:
                    if sym in positions:
                        positions[sym].update({'tp_order_id': t['algoId'], 'sl_order_id': s['algoId']})
                _clear_pending_protection(sym)
                log.info(f'[protection-resume] {sym} TP/SL berhasil dipasang kembali.')
            except Exception as e:
                failures.append(f'{sym}: protection {e}')
                log.error(f'[protection-resume] {sym} GAGAL: {e}')

        # 3) STRICT PENDING TRAIL
        with _pending_trails_lock:
            pending = [(sym, dict(v)) for sym, v in _pending_trails.items()]
        for sym, tr in pending:
            try:
                with positions_lock:
                    pos = positions.get(sym)
                if not pos or pos.get('status') != 'active':
                    _clear_pending_trail(sym)
                    continue
                buy = pos['signal']['decision'] == 'BUY'
                qty = pos.get('quantity') or tr.get('quantity')
                tp = tr.get('tp') or pos['signal'].get('tp')
                sl = tr.get('sl') or pos.get('current_sl')
                if not qty or sl is None or tp is None:
                    raise RuntimeError('pending trail tidak lengkap')
                cancel_algo_order(pos.get('tp_order_id'))
                cancel_algo_order(pos.get('sl_order_id'))
                t, s = place_tp_sl(sym, buy, tp, sl, qty)
                with positions_lock:
                    if sym in positions:
                        positions[sym].update({'tp_order_id': t['algoId'], 'sl_order_id': s['algoId'], 'exchange_synced_at': time.time()})
                _clear_pending_trail(sym)
                log.info(f'[trail-resume] {sym} pending trail berhasil dipasang kembali.')
            except Exception as e:
                failures.append(f'{sym}: trail {e}')
                log.error(f'[trail-resume] {sym} GAGAL: {e}')

        # 4) HARD GATE: partial recovery tidak boleh menyalakan scanner.
        if failures:
            with _binance_pause_lock:
                _binance_recovering = False
                _binance_scan_paused = True
                _binance_pause_reason = 'recovery incomplete'
            msg = ' | '.join(failures[:6])
            log.error(f'[BINANCE RECOVERY] BELUM SELESAI — scanner tetap PAUSED. {msg}')
            if chat_id:
                tg_send(chat_id, '⚠️ <b>Binance recovery belum selesai.</b>\nScanner tetap dihentikan karena sync/protection masih gagal.\nDetail: <code>' + msg[:500] + '</code>')
            return False

        # Semua gate sukses → scanner baru boleh resume.
        with _binance_pause_lock:
            _binance_recovering = False
            _binance_scan_paused = False
            _binance_pause_reason = ''
        if chat_id:
            tg_send(chat_id, '✅ <b>Binance recovery selesai.</b>\nSemua posisi berhasil disinkronkan dan pending protection/trailing berhasil diproses.\nScanning boleh resume.')
        return True

    except Exception as e:
        with _binance_pause_lock:
            _binance_recovering = False
            _binance_scan_paused = True
            _binance_pause_reason = 'recovery exception'
        log.error(f'[BINANCE RECOVERY] exception — scanner tetap PAUSED: {e}', exc_info=True)
        return False

def _binance_recovery_loop(chat_id_getter=lambda: active_chat_id):
    """Watchdog global. Setelah cooldown+60s, lakukan strict recovery; scanner hanya resume jika seluruh gate sukses."""
    notified = False
    while True:
        try:
            if _binance_is_scan_paused():
                if not notified:
                    tg_send(chat_id_getter(), "⏸️ <b>Binance RATE LIMIT/BAN</b> — scanning & entry baru dihentikan. Posisi aktif tetap dipantau via WS.")
                    notified = True
                if _binance_cooldown_remaining() <= 0 and not _binance_recovering:
                    if _resume_binance_and_flush_pending(chat_id_getter()):
                        notified = False
                else:
                    time.sleep(5)
                continue
        except Exception as e:
            log.warning(f"[binance-recovery] {e}")
        time.sleep(5)


def autostop_loop(chat_id):
    """Background: pantau saldo real, auto /stop kalau drawdown dari peak > AUTOSTOP_PCT."""
    global auto_mode, peak_real_balance
    while True:
        try:
            if REAL_TRADE_ENABLED and not _binance_is_scan_paused():
                _, total = get_real_balance()
                if total is not None:
                    with autostop_lock:
                        if peak_real_balance is None or total > peak_real_balance:
                            peak_real_balance = total
                        drawdown_pct = (peak_real_balance - total) / peak_real_balance * 100 if peak_real_balance else 0
                    if auto_mode and drawdown_pct >= AUTOSTOP_PCT:
                        auto_mode = False
                        tg_send(chat_id,
                            f"🛑 <b>AUTO-STOP TERPICU</b>\n\n"
                            f"Saldo turun <b>{drawdown_pct:.2f}%</b> dari peak "
                            f"(${peak_real_balance:.2f} → ${total:.2f})\n"
                            f"Threshold: {AUTOSTOP_PCT}%\n\n"
                            f"Scan sinyal baru dihentikan. Posisi aktif tetap dipantau.\n"
                            f"Jalankan lagi manual dengan /auto")
        except Exception as e:
            log.warning(f"[autostop_loop] {e}")
        time.sleep(60)


def simulation_loop(chat_id):
    """Koordinator runtime; seluruh keputusan trading berasal dari strategy."""
    global auto_mode
    tg_send(chat_id,"🤖 <b>Engine dimulai.</b>\nStrategy mengendalikan Entry/TP/SL/Trail.")
    scanning=False; scan_lock=threading.Lock(); last_scan=0.0

    def do_scan():
        nonlocal scanning
        try:
            signals = run_scan_once(chat_id)
            if not auto_mode or not signals:
                return

            opened = 0
            for signal in signals:
                if not auto_mode or _binance_is_scan_paused():
                    break
                sym = signal.get("symbol")
                if not sym:
                    continue
                with positions_lock:
                    if sym in positions or len(positions) >= MAX_POSITIONS:
                        continue

                if REAL_TRADE_ENABLED:
                    _open_pending_real(sym, signal, chat_id)
                    opened += 1
                    continue

                price = signal.get("price") or get_price(sym)
                entry = signal.get("entry")
                if price is None or entry is None:
                    continue
                mode = str(signal.get("execution_mode", "")).lower() or ("market" if signal.get("entry_label") == "market" else "limit")
                with positions_lock:
                    if sym in positions or len(positions) >= MAX_POSITIONS:
                        continue
                    positions[sym] = {"signal": signal, "entry": entry, "chat_id": chat_id,
                                      "entry_time": None, "timeout_flag": False, "status": "pending"}
                if mode == "market":
                    _open_position(sym, signal, get_price(sym) or price, chat_id, "strategy")
                else:
                    tg_send(chat_id, f"🎯 <b>PENDING ORDER</b> — {sym}\n\n{fmt_signal_msg(signal)}")
                    threading.Thread(target=wait_entry, args=(sym, signal, chat_id), daemon=True).start()
                opened += 1

            log.info(f"[scan] {len(signals)} signal lolos, {opened} dikirim ke execution")
        finally:
            with scan_lock:
                scanning = False

    def wait_entry(sym,signal,chat_id):
        entry=signal["entry"]; buy=signal["decision"]=="BUY"; deadline=time.time()+8*3600
        while time.time()<deadline:
            with positions_lock:
                if sym not in positions:return
                if positions[sym].get("timeout_flag"): positions.pop(sym,None); return
            price=get_price(sym)
            if price is not None and ((price<=entry) if buy else (price>=entry)):
                fill=min(entry,price) if buy else max(entry,price)
                _open_position(sym,signal,fill,chat_id,"strategy"); return
            time.sleep(MONITOR_SLEEP)
        with positions_lock: positions.pop(sym,None)
        _ban_coin(sym,"pending expired")

    while auto_mode:
        if _binance_is_scan_paused():
            time.sleep(5)
            continue
        with positions_lock: full=len(positions)>=MAX_POSITIONS
        if full: time.sleep(5); continue
        with scan_lock:
            if scanning: time.sleep(5); continue
            scanning=True
        if time.time()-last_scan<120:
            with scan_lock: scanning=False
            time.sleep(5); continue
        last_scan=time.time(); threading.Thread(target=do_scan,daemon=True).start(); time.sleep(5)
    tg_send(chat_id,"⏹ <b>Scanning dihentikan.</b>\n\n"+fmt_stats())




# ═════════════════════════════════════════════
# PESAN STATIS
# ═════════════════════════════════════════════
def get_start_msg():
    return (
        "👋 <b>SMC Signal Broadcaster</b>\n\n"
        f"Scan → multi-signal → max {MAX_POSITIONS} posisi bersamaan\n"
        f"Confidence minimum: <b>{STRATEGY_CONFIDENCE_THRESHOLD}%</b>\n"
        "Posisi ditutup hanya saat TP, SL, atau keputusan Trail dari strategy\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "/start               — Menu + konfigurasi aktif\n"
        "/auto                — Mulai broadcaster\n"
        "/stop                — Hentikan scanning (posisi aktif tetap dipantau)\n"
        "/trade               — Lihat semua posisi aktif\n"
        "/max                 — Lihat/ubah max posisi\n"
        "/confidence_min      — Lihat ambang confidence minimum\n"
        "/confidence_min 70   — Ubah threshold menjadi 70%\n"
        "/leverage            — Lihat/ubah leverage (real trade)\n"
        "/margin              — Lihat/ubah margin awal per trade (real trade)\n"
        "/autostop            — Lihat/ubah threshold auto-stop drawdown\n"
        "/mode                — Lihat mode aktif (real/simulasi)\n"
        "/mode on             — Real trade\n"
        "/mode off            — Simulasi\n"
        "/timeout SYMBOL      — Tutup paksa posisi tertentu\n"
        "/timeout             — Tutup paksa semua posisi\n"
        "/stats               — Statistik + saldo\n"
        "/backtest            — 20 trade terakhir (evaluasi)\n"
        "/banned              — Daftar koin ban\n"
        "/koin                — Daftar koin yang sedang di-scan\n"
        "/resetban            — Hapus semua ban\n"
        "/resetbalance        — Reset saldo ke $10\n"
        "/info                — Detail metode analisis\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        + ("🔴 <b>REAL TRADE AKTIF</b> — order sungguhan di Binance Futures, uang beneran."
           if REAL_TRADE_ENABLED else
           "⚠️ <i>Simulasi saja — bukan saran finansial.</i>")
    )

def get_info_msg():
    return ("ℹ️ <b>Engine</b>\n\n"
            "Strategy: Entry • Stop Loss • Take Profit • Trail • Confidence • Setup selection\n"
            "Engine: data transport • Telegram • order execution • position state • monitoring • statistics.")



# ═════════════════════════════════════════════
# RENDER KEEP-ALIVE / TELEGRAM WATCHDOG
# ═════════════════════════════════════════════
def _render_keepalive_loop():
    """Best-effort self health ping untuk mengurangi risiko Render Free idle.

    Render_EXTERNAL_URL dipakai kalau tersedia. Endpoint /healthz tidak menyentuh
    Binance, jadi loop ini tidak menambah Binance weight. Untuk jaminan terhadap
    spin-down Free, external uptime monitor tetap lebih kuat; loop ini adalah
    lapisan tambahan, bukan satu-satunya mekanisme.
    """
    base = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
    if not base:
        log.info("[render] RENDER_EXTERNAL_URL tidak tersedia — keepalive internal off")
        return
    url = f"{base}/healthz"
    while True:
        try:
            r = requests.get(url, timeout=10)
            if r.ok:
                log.debug("[render] keepalive OK")
            else:
                log.warning(f"[render] keepalive HTTP {r.status_code}")
        except Exception as e:
            log.debug(f"[render] keepalive gagal: {e}")
        time.sleep(TELEGRAM_KEEPALIVE_SEC)


def _telegram_watchdog_alert(cid, text):
    global _telegram_last_conflict_alert_at
    now = time.time()
    if now - _telegram_last_conflict_alert_at < 300:
        return
    _telegram_last_conflict_alert_at = now
    if cid:
        tg_send(cid, text)


# ═════════════════════════════════════════════
# BOT LOOP
# ═════════════════════════════════════════════
def bot_loop():
    global auto_mode, auto_thread, active_chat_id, timeout_flag, MAX_POSITIONS, LEVERAGE, MARGIN_USD, AUTOSTOP_PCT, peak_real_balance, REAL_TRADE_ENABLED, STRATEGY_CONFIDENCE_THRESHOLD

    # Set active_chat_id ke ALLOWED_USER_ID SEJAK AWAL — di chat pribadi
    # Telegram, chat_id sama dengan user_id, jadi bot bisa kirim pesan
    # proaktif (termasuk "Bot Siap" & notifikasi darurat) SEBELUM user
    # mengirim perintah apa pun. Sebelumnya active_chat_id cuma None
    # sampai user chat duluan, jadi notifikasi penting tidak pernah sampai.
    if ALLOWED_USER_ID:
        active_chat_id = ALLOWED_USER_ID

    # Tidak ada startup ping Binance. Request hanya dilakukan saat benar-benar dibutuhkan.

    _telegram_bootstrap()
    offset=None
    poll_backoff=1
    log.info(f"Bot siap — main.py {MAIN_ENGINE_VERSION}.")
    if ALLOWED_USER_ID:
        tg_send(ALLOWED_USER_ID,
            "✅ <b>Bot Siap</b>\n"
            "Semua sistem sudah menyala dan siap menerima perintah.\n"
            "Ketik /start untuk melihat menu.")

    while True:
        try:
            updates = tg_updates(offset)
            poll_backoff = 1
            for upd in updates:
                offset=upd["update_id"]+1
                msg=upd.get("message",{})
                uid=msg.get("from",{}).get("id")
                chat_id=msg.get("chat",{}).get("id")
                # Pesan berisi DOKUMEN pakai field "caption", bukan "text" —
                # "text" cuma ada di pesan teks polos tanpa lampiran. Sebelumnya
                # cuma baca "text", jadi /ganti (dikirim sbg dokumen + caption)
                # selalu ke-skip diam-diam di baris `if ... not text: continue`
                # di bawah, sebelum sempat sampai ke handler manapun.
                text=(msg.get("text") or msg.get("caption") or "").strip().lower()
                if not uid or not chat_id or not text: continue
                if uid!=ALLOWED_USER_ID:
                    tg_send(chat_id,"⛔ Akses ditolak."); continue
                active_chat_id=chat_id

                if text in ("/start","start"):
                    tg_send(chat_id,get_start_msg())
                elif text.startswith("/confidence_min") or text.startswith("confidence_min"):
                    parts = text.split()
                    if len(parts) == 1:
                        tg_send(chat_id, f"🎯 <b>Confidence minimum:</b> {STRATEGY_CONFIDENCE_THRESHOLD}%\nGunakan <code>/confidence_min 70</code> untuk mengubahnya.")
                    else:
                        try:
                            val = float(parts[1].replace("%", ""))
                            if not (0 <= val <= 100):
                                raise ValueError("rentang 0-100")
                            STRATEGY_CONFIDENCE_THRESHOLD = int(round(val))
                            tg_send(chat_id, f"✅ Confidence minimum diubah menjadi <b>{STRATEGY_CONFIDENCE_THRESHOLD}%</b>.")
                        except Exception:
                            tg_send(chat_id, "❌ Format salah. Gunakan <code>/confidence_min 70</code> (0-100).")
                elif text in ("/info","info"):
                    tg_send(chat_id,get_info_msg())
                elif text in ("/stats","stats"):
                    tg_send(chat_id,fmt_stats())
                elif text in ("/backtest","backtest"):
                    tg_send(chat_id,fmt_backtest())
                # ============================================================
                # TAMBAHAN BARU (START) — Handler /analyze
                # ============================================================
                elif text in ("/analyze","analyze"):
                    # Diagnostic snapshot satu universe. Background supaya loop Telegram tetap responsif.
                    def _run_analyze(cid):
                        try:
                            tg_send(cid,
                                f"🔎 <b>Mulai /analyze</b>\n"
                                f"Scan hingga {TOP_N_COINS} koin + detail Entry/SL/TP/Structure/Liquidity/Confidence.\n"
                                f"Threshold: {STRATEGY_CONFIDENCE_THRESHOLD}%\n"
                                f"Dibuat menjadi 2 file: report Markdown + data CSV.")
                            rows, passing, universe_error = _analyze_snapshot()
                            report_path = _write_analyze_report(rows, passing, universe_error)
                            csv_path = _write_analyze_csv(rows)

                            tg_send(cid,
                                f"✅ <b>/analyze selesai</b>\n"
                                f"Koin dipindai: {len(rows)}\n"
                                f"Lolos threshold: {len(passing)}\n\n"
                                f"Mengirim 2 file...")
                            tg_send_document(cid, report_path, caption="📊 analyze_report.md — diagnosis strategy & market snapshot")
                            tg_send_document(cid, csv_path, caption="📋 analyze_data.csv — data lengkap per koin")
                        except Exception as e:
                            log.error(f"[analyze] Error: {e}", exc_info=True)
                            tg_send(cid, f"❌ Error saat /analyze:\n<code>{str(e)[:300]}</code>")

                    threading.Thread(target=_run_analyze, args=(chat_id,), daemon=True).start()
                    tg_send(chat_id, "⏳ /analyze berjalan di background. Bot tetap menerima perintah lain.")
# ============================================================
# TAMBAHAN BARU (END)
# ============================================================
                # ============================================================
# TAMBAHAN BARU (START) — Handler /ganti (Upload Otak Baru via GitHub API)
# ============================================================
                elif text in ("/ganti","ganti"):
                   doc = msg.get("document")
                   if not doc:
                       tg_send(chat_id, "📤 Kirim file strategy_logic.py sebagai dokumen dengan caption /ganti")
                       continue
               
                   file_name = doc.get("file_name", "")
                   if not file_name.endswith(".py"):
                       tg_send(chat_id, "❌ Harus file .py")
                       continue
               
                   try:
                       # 1. Download file dari Telegram
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
               
                       # 2. Validasi sintaks
                       try:
                           compiled = compile(file_content, "strategy_logic.py", "exec")
                       except SyntaxError as e:
                           tg_send(chat_id, f"❌ Error sintaks di file:\n<code>{e}</code>")
                           continue
               
                       # 3. Validasi full_analyze() ADA
                       check_ns = {}
                       try:
                           exec(compiled, check_ns)
                       except Exception as e:
                           tg_send(chat_id, f"❌ File error saat dijalankan (bukan cuma sintaks):\n<code>{e}</code>")
                           continue
                       if "full_analyze" not in check_ns or not callable(check_ns["full_analyze"]):
                           tg_send(chat_id, "❌ File ini tidak punya fungsi full_analyze() — ditolak.")
                           continue
               
                       # 4. Commit ke GitHub
                       try:
                           _commit_to_github(file_content, "strategy_logic.py", f"Update strategy_logic via Telegram /ganti")
                           tg_send(chat_id, "✅ File berhasil di-commit ke GitHub!")
                       except Exception as e:
                           tg_send(chat_id, f"❌ Gagal commit ke GitHub:\n<code>{str(e)[:200]}</code>")
                           continue
               
                       # 5. Tulis ke file LOKAL
                       local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_logic.py")
                       with open(local_path, "w", encoding="utf-8") as f:
                           f.write(file_content)
               
                       # 6. ========== ADAPTIVE RELOAD: Bind apa yang ADA, pertahankan yang tidak ==========
                       import importlib, sys

                       # Hapus modul dari cache supaya reload benar-benar dari disk
                       if "strategy_logic" in sys.modules:
                           del sys.modules["strategy_logic"]

                       import strategy_logic as sl

                       # --- SENTINEL untuk membedakan "tidak ada" vs None ---
                       _SL_SENTINEL = object()

                       def _sl_bind(name):
                           """Bind ke global HANYA kalau ada di modul baru.
                           Kalau tidak ada -> global lama tetap aktif, return False."""
                           val = getattr(sl, name, _SL_SENTINEL)
                           if val is not _SL_SENTINEL:
                               globals()[name] = val
                               return True
                           return False

                       # WAJIB: full_analyze sudah divalidasi ada di atas
                       globals()["full_analyze"] = sl.full_analyze

                       # -- Fungsi opsional --------------------------------------------------
                       # Kalau tidak ada di file baru -> versi lama di global tetap aktif.
                       # Kamu bebas ganti nama, tambah, atau hapus fungsi apapun
                       # selama full_analyze() tetap ada.
                       _OPT_FNS = ["manage_position"]
                       _bound_fns, _kept_fns = [], []
                       for _fn in _OPT_FNS:
                           (_bound_fns if _sl_bind(_fn) else _kept_fns).append(_fn)

                       # Tangkap semua public callable BARU yang tidak ada di daftar atas
                       for _attr in dir(sl):
                           if _attr.startswith("__"):
                               continue
                           if _attr not in _OPT_FNS and _attr != "full_analyze":
                               _v = getattr(sl, _attr, None)
                               if callable(_v):
                                   globals()[_attr] = _v
                                   if _attr not in _bound_fns:
                                       _bound_fns.append(f"✨{_attr}")

                       # -- Konstanta opsional -----------------------------------------------
                       # Kalau tidak ada di file baru, nilai lama dipertahankan.
                       _OPT_CONSTS = []
                       _bound_consts, _kept_consts = [], []
                       for _k in _OPT_CONSTS:
                           (_bound_consts if _sl_bind(_k) else _kept_consts).append(_k)

                       # -- Laporan ke user --------------------------------------------------
                       _rpt = ["✅ <b>Strategy logic aktif!</b>"]
                       if _bound_fns:
                           _rpt.append(f"🔄 Diperbarui: <code>{', '.join(_bound_fns)}</code>")
                       if _kept_fns:
                           _rpt.append(f"♻️ Versi lama dipertahankan: <code>{', '.join(_kept_fns)}</code>")
                       if _bound_consts:
                           _rpt.append(f"📐 Konstanta diperbarui: <code>{', '.join(_bound_consts)}</code>")
                       if _kept_consts:
                           _rpt.append(f"📌 Konstanta lama dipertahankan: <code>{', '.join(_kept_consts)}</code>")

                       log.info("[OTAK] Strategy logic di-reload (adaptive bind).")
                       tg_send(chat_id, "\n".join(_rpt))
               
                   except Exception as e:
                       log.error(f"[ganti] Error: {e}")
                       tg_send(chat_id, f"❌ Gagal mengganti strategy_logic:\n<code>{str(e)[:200]}</code>")
                # ============================================================
                # TAMBAHAN BARU (END)
                # ============================================================
                elif text.startswith("/banned") or text.startswith("banned"):
                    parts = text.split()
                    if len(parts) > 1:
                        # /banned <koin> -> ban PERMANEN (duration=inf, tidak pernah auto-unban)
                        target_sym = parts[1].upper()
                        with ban_lock:
                            banned_coins[target_sym] = (scan_counter, float("inf"))
                        log.info(f"[ban] {target_sym} diban PERMANEN (manual via /banned)")
                        tg_send(chat_id, f"🚫 <b>{target_sym} diban PERMANEN.</b>\nLepas lagi dengan /resetban.")
                    else:
                        with ban_lock:
                            cur_scan = scan_counter
                            b = sorted(banned_coins.items())
                        if b:
                            lines = []
                            for sym, (banned_at, dur) in b:
                                if dur == float("inf"):
                                    lines.append(f"• {sym} (PERMANEN)")
                                else:
                                    remaining = max(0, dur - (cur_scan - banned_at))
                                    lines.append(f"• {sym} (unban dalam {remaining} scan)")
                            tg_send(chat_id,
                                f"🚫 <b>Banned ({len(b)}):</b>\n" + "\n".join(lines) +
                                f"\n\n<i>Ban permanen: /banned SYMBOL</i>")
                        else:
                            tg_send(chat_id, "✅ Belum ada ban.\n\n<i>Ban permanen: /banned SYMBOL</i>")
                elif text in ("/koin","koin"):
                    with _last_scanned_lock:
                        coins = list(last_scanned_coins)
                        scanned_at = last_scanned_at
                    if not coins:
                        tg_send(chat_id, "⏳ Belum ada data — tunggu siklus scan pertama selesai.")
                    else:
                        age_min = (time.time() - scanned_at) / 60 if scanned_at else 0
                        tg_send(chat_id,
                            f"📋 <b>Koin yang di-scan ({len(coins)})</b> — update {age_min:.0f} menit lalu:\n\n"
                            + ", ".join(coins))
                elif text in ("/resetban","resetban"):
                    with ban_lock: n=len(banned_coins); banned_coins.clear()
                    tg_send(chat_id,f"✅ Ban direset ({n} dihapus).")
                elif text in ("/resetbalance","resetbalance"):
                    with stat_lock:
                        stats["balance"]     = STARTING_BALANCE
                        stats["pnl_history"] = deque(maxlen=20)
                        stats["tp"]          = 0
                        stats["sl"]          = 0
                        stats["trail"]       = 0
                        stats["total"]       = 0
                    tg_send(chat_id,
                        f"✅ Saldo & statistik direset.\n"
                        f"💵 Modal awal: <b>${STARTING_BALANCE:.2f}</b>")
                elif text in ("/auto","auto"):
                    if auto_mode:
                        tg_send(chat_id,"⚙️ Broadcaster sudah berjalan.")
                    else:
                        # Reset referensi peak ke saldo SEKARANG — supaya drawdown
                        # dihitung ulang dari titik ini, bukan dari peak lama yang
                        # bikin auto-stop langsung kepicu lagi begitu /auto ditekan.
                        if REAL_TRADE_ENABLED:
                            _, total = get_real_balance()
                            with autostop_lock:
                                peak_real_balance = total
                        auto_mode=True
                        auto_thread=threading.Thread(
                            target=simulation_loop,args=(chat_id,),daemon=True)
                        auto_thread.start()
                elif text in ("/stop","stop"):
                    # /stop hanya mematikan scanning sinyal baru — posisi
                    # yang sudah berjalan tetap dipantau sampai TP/SL alami.
                    if auto_mode:
                        auto_mode = False
                        with positions_lock:
                            n_active = len(positions)
                        tg_send(chat_id,
                            f"⏹ <b>Scanning dihentikan.</b>\n"
                            f"Posisi aktif ({n_active}) tetap dipantau sampai TP/SL.\n"
                            f"Pakai /timeout SYMBOL kalau mau tutup paksa.")
                    else:
                        tg_send(chat_id,"ℹ️ Broadcaster tidak berjalan.")
                elif text in ("/trade","trade"):
                    with positions_lock:
                        pos_list = list(positions.items())
                    if not pos_list:
                        tg_send(chat_id,"ℹ️ Tidak ada posisi aktif.")
                    else:
                        lines = [f"📡 <b>Posisi Aktif ({len(pos_list)}/{MAX_POSITIONS})</b>\n"]
                        for s, p in pos_list:
                            sig    = p["signal"]
                            is_buy = sig["decision"] == "BUY"
                            em     = "🟢" if is_buy else "🔴"
                            status = p.get("status", "active")

                            if status == "pending":
                                pr       = get_price(s) or p["entry"]
                                dist_pct = abs(p["entry"] - pr) / pr * 100
                                lines.append(
                                    f"\n⏳ <b>{s}</b> — PENDING\n"
                                    f"{em} {sig['decision']} | Entry zone: <code>{p['entry']:.6g}</code>\n"
                                    f"Harga kini: <code>{pr:.6g}</code> | Jarak: {dist_pct:.2f}%\n"
                                    f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{sig['sl']:.6g}</code>"
                                )
                            else:
                                pr  = get_price(s) or p["entry"]
                                pnl = (pr - p["entry"]) / p["entry"] * 100 * (1 if is_buy else -1)
                                entry_clock = datetime.fromtimestamp(
                                    p["entry_time"], tz=WIB).strftime("%H:%M")
                                cur_sl = p.get("current_sl", sig["sl"])
                                trail_note = " 🔒trailing" if cur_sl != sig["sl"] else ""
                                lines.append(
                                    f"\n{em} <b>{s}</b> — AKTIF\n"
                                    f"Entry: <code>{p['entry']:.6g}</code> | Harga: <code>{pr:.6g}</code>\n"
                                    f"TP: <code>{sig['tp']:.6g}</code> | SL: <code>{cur_sl:.6g}</code>{trail_note}\n"
                                    f"PnL: <b>{pnl:+.2f}%</b> | 🕐 Entry jam {entry_clock}"
                                )
                        tg_send(chat_id,"\n".join(lines))
                elif text.startswith("/timeout") or (not text.startswith("/") and text.startswith("timeout")):
                    parts = text.split()
                    target_sym = parts[1].upper() if len(parts) > 1 else None
                    with positions_lock:
                        syms = list(positions.keys())
                    if not syms:
                        tg_send(chat_id,"ℹ️ Tidak ada posisi aktif.")
                    elif target_sym:
                        if target_sym in syms:
                            with positions_lock:
                                if target_sym in positions:
                                    positions[target_sym]["timeout_flag"] = True
                            tg_send(chat_id,f"⏭ Timeout → {target_sym}.")
                        else:
                            tg_send(chat_id,
                                f"❓ {target_sym} tidak ditemukan.\n"
                                f"Aktif: {', '.join(syms)}")
                    else:
                        with positions_lock:
                            for s in syms:
                                if s in positions:
                                    positions[s]["timeout_flag"] = True
                        tg_send(chat_id,f"⏭ Timeout semua ({len(syms)}) posisi.")
                elif text.startswith("/mode"):
                    # /mode          → tampilkan status sekarang
                    # /mode on       → aktifkan REAL TRADE (butuh API key)
                    # /mode off      → aktifkan mode SIMULASI (backtest strategy_logic)
                    #
                    # PENTING: toggle ini HANYA memengaruhi posisi BARU yang
                    # dibuka SETELAH perintah ini. Posisi yang sudah terbuka
                    # tetap dipantau oleh thread monitor yang sama seperti
                    # saat dibuka (monitor_position untuk simulasi,
                    # monitor_position_real untuk real trade) — thread itu
                    # sudah "terkunci" ke fungsinya sejak posisi dibuat, jadi
                    # toggle mode di tengah jalan TIDAK mengubah/mengganggu
                    # posisi yang sedang berjalan sama sekali (tidak ada
                    # posisi yang tiba-tiba pindah rezim atau kehilangan
                    # monitoring). Pipeline pencarian sinyal, monitoring, dan
                    # pencatatan statistik 100% sama persis di kedua mode —
                    # satu-satunya cabang beda cuma di titik "buka posisi"
                    # (line ~2742: REAL_TRADE_ENABLED → _open_pending_real
                    # vs jalur simulasi), jadi tidak ada logic baru yang
                    # perlu diduplikasi/di-maintain terpisah.
                    parts = text.split()
                    arg = parts[1].lower() if len(parts) > 1 else None
                    with positions_lock:
                        n_open = len(positions)
                    if arg is None:
                        status = "🔴 REAL TRADE (uang beneran)" if REAL_TRADE_ENABLED else "🧪 SIMULASI (backtest strategy_logic)"
                        tg_send(chat_id,
                            f"⚙️ Mode saat ini: <b>{status}</b>\n\n"
                            f"Ganti dengan <code>/mode on</code> (real) atau "
                            f"<code>/mode off</code> (simulasi).")
                    elif arg == "on":
                        if not BINANCE_KEYS_PRESENT:
                            tg_send(chat_id,
                                "❌ Tidak bisa aktifkan mode real — "
                                "BINANCE_API_KEY/BINANCE_API_SECRET belum diset di server.")
                        elif REAL_TRADE_ENABLED:
                            tg_send(chat_id, "🔴 Mode real sudah aktif.")
                        else:
                            REAL_TRADE_ENABLED = True
                            extra = (f"\n\nℹ️ {n_open} posisi simulasi yang sedang berjalan "
                                     f"tetap dipantau sebagai simulasi sampai selesai (TP/SL/timeout) — "
                                     f"tidak ikut berubah jadi real." if n_open else "")
                            tg_send(chat_id, f"🔴 <b>Mode REAL TRADE diaktifkan.</b> "
                                              f"Posisi baru mulai sekarang akan pakai uang beneran di Binance.{extra}")
                    elif arg == "off":
                        if not REAL_TRADE_ENABLED:
                            tg_send(chat_id, "🧪 Mode simulasi sudah aktif.")
                        else:
                            REAL_TRADE_ENABLED = False
                            extra = (f"\n\nℹ️ {n_open} posisi real yang sedang berjalan "
                                     f"tetap dipantau & ditutup normal via Binance — "
                                     f"tidak ikut berubah jadi simulasi." if n_open else "")
                            tg_send(chat_id, f"🧪 <b>Mode SIMULASI diaktifkan.</b> "
                                              f"Posisi baru mulai sekarang cuma backtest strategy_logic, "
                                              f"tidak ada order sungguhan.{extra}")
                    else:
                        tg_send(chat_id, "❓ Pakai <code>/mode</code>, <code>/mode on</code>, atau <code>/mode off</code>.")
                elif text.startswith("/max"):
                    parts = text.split()
                    # ── /max (tampilkan info) ──────────────────────────────
                    if len(parts) == 1:
                        # Estimasi beban API saat ini
                        scan_weight_per_min  = 836   # ~100 kline req × weight5 / ~34s scan
                        price_weight_per_min = 12    # 1 batch ticker/price tiap 10 detik
                        total_weight         = scan_weight_per_min + price_weight_per_min
                        binance_limit        = 2400
                        usage_pct            = total_weight / binance_limit * 100
                        headroom_pct         = 100 - usage_pct
                        threads_now          = 4 + MAX_POSITIONS * 2   # bot+cache+flask+scan + monitor+wait_entry

                        # Batas aman: scan mendominasi, bukan jumlah posisi
                        # Posisi hanya menambah ~0.02 weight/mnt per posisi (SL check jarang)
                        # Batas praktis sebelum scan overload:
                        #   sisa headroom = 1552 weight/mnt, scan = 836/mnt
                        #   bisa ~2 scan paralel tapi kode hanya 1 scan sekaligus → aman tak terbatas dari sisi API
                        # Batas rekomendasi dari sisi KUALITAS SINYAL: ≤ 20
                        tg_send(chat_id,
                            f"⚙️ <b>Max Posisi</b>\n\n"
                            f"Saat ini     : <b>{MAX_POSITIONS} posisi</b>\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📡 <b>Info Beban API (Binance Futures)</b>\n\n"
                            f"Limit Binance    : <b>2.400 weight/mnt</b>\n"
                            f"Scan 50 koin     : ~{scan_weight_per_min} weight/mnt\n"
                            f"Price cache      : ~{price_weight_per_min} weight/mnt (1 batch/10 dtk)\n"
                            f"Total dipakai    : ~{total_weight} weight/mnt "
                            f"(<b>{usage_pct:.0f}%</b> dari limit)\n"
                            f"Headroom tersisa : ~{headroom_pct:.0f}%\n\n"
                            f"⚠️ <b>Penting:</b> MAX_POSITIONS <b>tidak</b> menambah beban\n"
                            f"API secara signifikan. Beban didominasi scan koin,\n"
                            f"bukan jumlah posisi yang dipantau.\n"
                            f"Monitor thread baca harga dari cache lokal — bukan API.\n\n"
                            f"🧵 Thread aktif est. : ~{threads_now}\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 <b>Batas yang Disarankan</b>\n\n"
                            f"API weight  : ✅ aman hingga 50+ posisi\n"
                            f"Thread      : ✅ aman hingga 50+ posisi\n"
                            f"Kualitas sinyal: ⚠️  disarankan ≤ 20\n"
                            f"  (lebih dari itu, scanner makin susah\n"
                            f"  temukan setup berkualitas karena koin\n"
                            f"  terbaik sudah terpakai)\n\n"
                            f"<b>Ubah: /max 5 | /max 10 | /max 15 | /max 20</b>")
                    # ── /max N (ubah nilai) ────────────────────────────────
                    elif len(parts) == 2:
                        try:
                            n = int(parts[1])
                            if n < 1 or n > 50:
                                tg_send(chat_id,
                                    f"❌ Nilai harus antara 1–50.\n"
                                    f"Contoh: /max 10")
                            else:
                                old = MAX_POSITIONS
                                MAX_POSITIONS = n
                                with positions_lock:
                                    n_active = len(positions)
                                note = ""
                                if n < n_active:
                                    note = (f"\n\n⚠️ Ada {n_active} posisi aktif saat ini.\n"
                                            f"Posisi yang sudah buka tetap dipantau.\n"
                                            f"Scan baru berhenti sampai posisi tutup ke ≤ {n}.")
                                tg_send(chat_id,
                                    f"✅ Max posisi diubah: <b>{old} → {MAX_POSITIONS}</b>{note}")
                        except ValueError:
                            tg_send(chat_id,"❌ Format salah. Contoh: /max 10")
                    else:
                        tg_send(chat_id,"❌ Format: /max  atau  /max 10")

                elif text.startswith("/leverage"):
                    parts = text.split()
                    if len(parts) == 1:
                        tg_send(chat_id,
                            f"⚙️ <b>Leverage</b>\n\nSaat ini: <b>{LEVERAGE}x</b>\n\n"
                            f"<b>Ubah: /leverage 5</b>")
                    elif len(parts) == 2:
                        try:
                            n = int(parts[1])
                            if n < 1 or n > 125:
                                tg_send(chat_id, "❌ Nilai harus antara 1–125.\nContoh: /leverage 5")
                            else:
                                old = LEVERAGE
                                LEVERAGE = n
                                tg_send(chat_id, f"✅ Leverage diubah: <b>{old}x → {LEVERAGE}x</b>")
                        except ValueError:
                            tg_send(chat_id, "❌ Format salah. Contoh: /leverage 5")
                    else:
                        tg_send(chat_id, "❌ Format: /leverage  atau  /leverage 5")

                elif text.startswith("/margin"):
                    parts = text.split()
                    if len(parts) == 1:
                        tg_send(chat_id,
                            f"⚙️ <b>Margin Awal</b>\n\nSaat ini: <b>${MARGIN_USD:.2f}</b>\n\n"
                            f"Kalau margin ini terlalu kecil untuk suatu koin (kena batas minimum\n"
                            f"quantity/notional Binance), bot otomatis menaikkan SEDIKIT (maks 1.5x)\n"
                            f"khusus untuk trade itu — bukan mengubah setting ini secara permanen.\n\n"
                            f"<b>Ubah: /margin 5</b>")
                    elif len(parts) == 2:
                        try:
                            n = float(parts[1])
                            if n <= 0 or n > 10000:
                                tg_send(chat_id, "❌ Nilai harus antara 0–10000.\nContoh: /margin 5")
                            else:
                                old = MARGIN_USD
                                MARGIN_USD = n
                                tg_send(chat_id, f"✅ Margin awal diubah: <b>${old:.2f} → ${MARGIN_USD:.2f}</b>")
                        except ValueError:
                            tg_send(chat_id, "❌ Format salah. Contoh: /margin 5")
                    else:
                        tg_send(chat_id, "❌ Format: /margin  atau  /margin 5")

                elif text.startswith("/autostop"):
                    parts = text.split()
                    if len(parts) == 1:
                        with autostop_lock:
                            peak_txt = f"${peak_real_balance:.2f}" if peak_real_balance else "belum ada data"
                        tg_send(chat_id,
                            f"⚙️ <b>Auto-Stop Drawdown</b>\n\nThreshold: <b>{AUTOSTOP_PCT}%</b>\n"
                            f"Peak saldo tercatat: {peak_txt}\n\n"
                            f"Kalau saldo turun segini persen dari peak, scan sinyal baru otomatis\n"
                            f"berhenti (posisi aktif tetap dipantau). Jalankan lagi manual dengan /auto.\n\n"
                            f"<b>Ubah: /autostop 3</b>")
                    elif len(parts) == 2:
                        try:
                            n = float(parts[1])
                            if n <= 0 or n > 100:
                                tg_send(chat_id, "❌ Nilai harus antara 0–100.\nContoh: /autostop 3")
                            else:
                                old = AUTOSTOP_PCT
                                AUTOSTOP_PCT = n
                                tg_send(chat_id, f"✅ Threshold auto-stop diubah: <b>{old}% → {AUTOSTOP_PCT}%</b>")
                        except ValueError:
                            tg_send(chat_id, "❌ Format salah. Contoh: /autostop 3")
                    else:
                        tg_send(chat_id, "❌ Format: /autostop  atau  /autostop 3")
                else:
                    tg_send(chat_id,"❓ Tidak dikenal. /start")

            time.sleep(0.2)
        except TelegramPollingConflict as e:
            log.error(f"[TG POLLING CONFLICT] {e}")
            _telegram_watchdog_alert(
                active_chat_id,
                "🚨 <b>Telegram polling conflict</b>\n\n"
                "Bot masih hidup, tetapi <code>getUpdates</code> bentrok. "
                "Pastikan hanya 1 instance bot memakai TELEGRAM_TOKEN ini."
            )
            time.sleep(min(max(poll_backoff, 5), TELEGRAM_ERROR_BACKOFF_MAX))
            poll_backoff = min(max(poll_backoff * 2, 5), TELEGRAM_ERROR_BACKOFF_MAX)
        except Exception as e:
            log.error(f"[TG/BOT LOOP] {e}", exc_info=True)
            time.sleep(min(max(poll_backoff, 2), TELEGRAM_ERROR_BACKOFF_MAX))
            poll_backoff = min(max(poll_backoff * 2, 2), TELEGRAM_ERROR_BACKOFF_MAX)


if __name__=="__main__":
    # Flask dijalankan di thread sendiri PALING AWAL supaya port langsung
    # bind & terdeteksi Render, tidak menunggu inisialisasi bot/WS selesai.
    threading.Thread(target=run_flask, daemon=True).start()
    ws_feed.start()
    threading.Thread(target=_price_cache_loop, daemon=True).start()
    threading.Thread(target=_binance_recovery_loop, daemon=True).start()
    threading.Thread(target=_render_keepalive_loop, daemon=True).start()
    threading.Thread(target=bot_loop, daemon=True).start()

    if _STRATEGY_LOAD_ERROR and ALLOWED_USER_ID:
        tg_send(ALLOWED_USER_ID,
            f"🚨 <b>strategy_logic.py BERMASALAH</b>\n\n"
            f"{_STRATEGY_LOAD_ERROR}\n\n"
            f"Bot jalan pakai fallback AMAN (tidak akan cari/entry sinyal baru)\n"
            f"sampai file yang benar di-upload lewat /ganti.")

    if REAL_TRADE_ENABLED and ALLOWED_USER_ID:
        ip = get_public_ip()
        tg_send(ALLOWED_USER_ID,
            f"🔴 <b>REAL TRADE MODE</b>\n\n"
            f"IP Render saat ini: <code>{ip}</code>\n\n"
            f"Whitelist IP ini di Binance API Management dulu kalau belum,\n"
            f"lalu kirim /auto untuk mulai. Bot TIDAK akan mulai cari sinyal\n"
            f"sampai kamu kirim /auto secara manual.")
        threading.Thread(target=autostop_loop, args=(ALLOWED_USER_ID,), daemon=True).start()

    # Semua thread di atas daemon=True — main thread harus tetap hidup,
    # kalau tidak proses langsung exit begitu baris ini selesai.
    while True:
        time.sleep(3600)
