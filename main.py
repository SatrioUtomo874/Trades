#!/usr/bin/env python3
"""
main.py — MESIN (engine). Telegram handler, API client, monitoring,
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
MAX_POSITIONS       = 20   # runtime via /max — jangan pindah ke strategy_logic
MONITOR_INTERVAL    = 15 * 60
MIN_CONFIDENCE      = 50   # runtime via /confidence_min — jangan pindah ke strategy_logic
WIB = timezone(timedelta(hours=7))   # format jam entry di /trade
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
    MIN_RR = 2.0
    TRAIL_R_LADDER = [(0.5, 0.15), (1.0, 0.35), (1.5, 0.50),
                       (2.0, 0.65), (2.8, 0.80), (3.5, 0.85)]
    STRUCT_TRAIL_LB, STRUCT_TRAIL_BUF_PCT, STRUCT_TRAIL_LOOKBACK = 2, 0.0015, 60
    FIB_EXT_1, FIB_EXT_2 = 0.272, 0.618
    H4_RSI_BUY_MIN, H4_RSI_BUY_MAX = 45, 68
    H4_RSI_SELL_MIN, H4_RSI_SELL_MAX = 32, 55
    def full_analyze(df_h1, df_m15, df_d1=None, symbol=None):
        return None  # fallback: tidak pernah hasilkan sinyal baru
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
REAL_TRADE_ENABLED = bool(BINANCE_API_KEY and BINANCE_API_SECRET)

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
def health(): return "OK", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    log.info(f"[flask] binding port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═════════════════════════════════════════════
# TELEGRAM
# ═════════════════════════════════════════════
def tg_send(chat_id, text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id":chat_id,"text":text,"parse_mode":"HTML"},
            timeout=10)
    except Exception as e:
        log.error(f"[TG] {e}")

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

def tg_updates(offset=None):
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"timeout":8,"offset":offset}, timeout=12)
        d = r.json()
        return d.get("result",[]) if d.get("ok") else []
    except:
        return []


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

def _binance_wait_if_banned():
    with _binance_ban_lock:
        until = _binance_banned_until
    remaining = until - time.time()
    if remaining > 0:
        log.warning(f"[binance] Masih dalam masa ban, menunggu {remaining:.0f} detik lagi sebelum request baru...")
        time.sleep(remaining + 1)   # +1 detik buffer

def _binance_register_ban(msg="", fallback_seconds=60):
    """Catat waktu ban global. Coba parse 'banned until <ms epoch>' dari
    pesan error Binance (paling akurat); kalau tidak ada, mundur konservatif
    (fallback_seconds, makin lama tiap kena berturut-turut)."""
    global _binance_banned_until
    m = re.search(r"banned until (\d+)", msg)
    until = int(m.group(1)) / 1000 if m else (time.time() + fallback_seconds)
    with _binance_ban_lock:
        if until > _binance_banned_until:
            _binance_banned_until = until
    wait = until - time.time()
    log.error(f"[binance] Kena limit/ban — semua request Binance dijeda {max(wait,0):.0f} detik.")


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
    _binance_wait_if_banned()
    for i in range(3):
        try:
            r = requests.get(f"{FAPI}{path}", params=params,
                             timeout=10, verify=False)
            if r.status_code in (418, 429):
                # Kena rate-limit/ban IP dari Binance — JANGAN retry lagi
                # ke Binance (mengulang request saat sedang kena ban malah
                # berisiko memperpanjang durasi ban). Catat state ban
                # global (dipakai fapi_get & _binance_signed) lalu lempar
                # ke caller supaya pindah ke tier fallback (Bybit → WS).
                try:
                    body_msg = r.text
                except Exception:
                    body_msg = ""
                _binance_register_ban(body_msg)
                raise ConnectionError(
                    f"Binance kena limit/ban (HTTP {r.status_code})")
            d = r.json()
            if isinstance(d, dict) and "code" in d:
                if d["code"] == -1003:
                    _binance_register_ban(d.get("msg", ""))
                raise ValueError(f"Binance {d['code']}: {d.get('msg')}")
            return d
        except ConnectionError as e:
            log.warning(f"[binance] {e} — stop retry Binance, pindah fallback")
            raise
        except Exception as e:
            log.warning(f"[binance] {i+1}/3: {e}")
            time.sleep(2)
    raise ConnectionError(f"Binance gagal: {path}")


# ============================================================
# REAL TRADE — Binance Futures signed API (order/leverage/posisi)
# Dipakai TERPISAH dari fapi_get di atas (yang publik, untuk cari
# sinyal) supaya limit rate keduanya tidak bercampur.
# ============================================================
import hmac, hashlib, urllib.parse, math
from decimal import Decimal, ROUND_HALF_UP

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
            r = requests.request(method, url, headers=headers, timeout=10, verify=False)
            if r.status_code in (418, 429):
                _binance_register_ban(r.text)
                raise RuntimeError(f"Binance kena limit/ban (HTTP {r.status_code})")
            data = r.json()
            if isinstance(data, dict) and "code" in data and data["code"] < 0:
                if data["code"] == -1003:
                    _binance_register_ban(data.get("msg", ""))
                raise RuntimeError(f"Binance {data['code']}: {data.get('msg')}")
            return data
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
        q = math.floor((notional / entry_price) / step) * step
        return round(q, info["qtyPrecision"])

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

def place_sl_order(symbol, is_buy, sl_price):
    close_side = "SELL" if is_buy else "BUY"
    tick = get_symbol_filters(symbol)["tickSize"]
    return _binance_signed("POST", "/fapi/v1/algoOrder", {
        "algoType": "CONDITIONAL", "symbol": symbol, "side": close_side, "type": "STOP_MARKET",
        "triggerPrice": round_to_tick(sl_price, tick), "closePosition": "true", "workingType": "MARK_PRICE",
    })


def place_tp_sl(symbol, is_buy, tp_price, sl_price):
    close_side = "SELL" if is_buy else "BUY"
    tick = get_symbol_filters(symbol)["tickSize"]
    tp = _binance_signed("POST", "/fapi/v1/algoOrder", {
        "algoType": "CONDITIONAL", "symbol": symbol, "side": close_side, "type": "TAKE_PROFIT_MARKET",
        "triggerPrice": round_to_tick(tp_price, tick), "closePosition": "true", "workingType": "MARK_PRICE",
    })
    sl = place_sl_order(symbol, is_buy, sl_price)
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

    def get_klines(self, symbol, interval, limit):
        with self._lock:
            buf = self._klines.get((symbol, interval))
            if not buf:
                return None
            rows = list(buf)[-limit:]
        if len(rows) < min(limit, 40):
            return None
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], unit="ms")
        return df[["o","h","l","c","v"]].rename(
            columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})

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
    """Tier1 Binance REST → Tier2 Bybit REST → Tier3 WS (fallback TERAKHIR,
    hanya dipakai kalau REST Binance & Bybit gagal/error/kena ban) →
    Tier4 CoinGecko (darurat paling akhir, hanya koin di COINGECKO_ID_MAP)."""
    for _ in range(2):
        try:
            return _binance_price(symbol)
        except Exception as e:
            log.warning(f"[price/binance] {symbol}: {e}")
            time.sleep(1)
    for _ in range(2):
        try:
            return _bybit_price(symbol)
        except Exception as e:
            log.warning(f"[price/bybit] {symbol}: {e}")
            time.sleep(1)
    if ws_feed.is_fresh():
        p = ws_feed.get_price(symbol)
        if p is not None:
            log.warning(f"[price/ws fallback] {symbol} — REST Binance & Bybit gagal")
            return p
    p = _coingecko_price(symbol)
    if p is not None:
        log.warning(f"[price/coingecko DARURAT] {symbol} — semua sumber lain gagal")
        return p
    return None

def get_klines(symbol, interval, limit=250):
    """Tier1 buffer WS (GRATIS, live-updated di background) → Tier2 Binance
    REST → Tier3 Bybit REST. Sebelumnya REST Binance dipanggil DULUAN tiap
    kali (WS cuma fallback terakhir) — padahal WS-nya sudah jalan terus,
    live, dan nol biaya rate-limit. Itu penyebab utama sering kena
    limit/ban meski jumlah posisi cuma sedikit: setiap scan/monitor tetap
    nembak REST walau datanya sebenarnya sudah ada gratis di buffer WS."""
    ws_feed.ensure_symbol_interval(symbol, interval)

    if ws_feed.is_fresh():
        df = ws_feed.get_klines(symbol, interval, limit)
        if df is not None and not df.empty:
            return df

    try:
        df = _binance_klines(symbol, interval, limit)
        if not df.empty:
            return df
        log.warning(f"[klines/binance] {symbol} kosong, coba Bybit...")
    except Exception as e:
        log.warning(f"[klines/binance] {symbol}: {e} — coba Bybit...")
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
    """Ambil top coins. Tier1 Binance REST → Tier2 Bybit REST → Tier3 WS
    ticker cache (fallback TERAKHIR, hanya kalau REST Binance & Bybit
    gagal/error/kena ban). Logika exclude/ban SAMA PERSIS seperti
    sebelumnya."""
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
        log.warning("[top_coins/binance] kosong, coba Bybit...")
    except Exception as e:
        log.warning(f"[top_coins/binance] {e} — coba Bybit...")
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
    tg_send(chat_id,f"🔍 Scanning {TOP_N_COINS} koin...")
    try:
        symbols=get_top_coins()
    except Exception as e:
        tg_send(chat_id,f"⚠️ Binance error: <code>{str(e)[:150]}</code>")
        return None

    if not symbols:
        tg_send(chat_id,"⚠️ Tidak ada koin tersedia untuk di-scan saat ini.")
        return None

    results=[]
    for idx,sym in enumerate(symbols,1):
        log.info(f"[{idx:02d}/{len(symbols)}] {sym}")
        try:
            df_h1  = get_klines(sym, "1h",  250)
            df_m15 = get_klines(sym, "15m", 250)
            try:
                df_d1 = get_klines(sym, "1d", 100)
            except Exception:
                df_d1 = None
            r = full_analyze(df_h1, df_m15, df_d1, symbol=sym)
        except Exception as e:
            log.debug(f"[scan] {sym}: {e}")
            r = None
        if r: results.append(r)
        time.sleep(0.15)

    if not results:
        tg_send(chat_id,"⚠️ Tidak ada setup valid dari semua koin.")
        return None

    # Filter: hanya koin dengan confidence >= MIN_CONFIDENCE (diatur via /confidence_min)
    results = [r for r in results if r["confidence"] >= MIN_CONFIDENCE]
    if not results:
        tg_send(chat_id,f"⚠️ Tidak ada koin dengan confidence cukup (≥{MIN_CONFIDENCE}%). Retry...")
        return None

    # Ranking: confidence DESC → rr DESC
    results.sort(key=lambda x:(x["confidence"],x["rr"]),reverse=True)
    best=results[0]
    log.info(f"Best: {best['symbol']} {best['decision']} "
             f"conf={best['confidence']}% RR=1:{best['rr']}")
    return best



# ═════════════════════════════════════════════
# STATISTIK + BALANCE
# ═════════════════════════════════════════════
POSITION_SIZE_PCT = 100.0  # ukuran posisi per trade = 100% saldo (setara 1× leverage)
                            # P&L murni dari jarak SL/TP yang ditetapkan analisis:
                            #   TP hit → gain = posisi × (tp_dist / entry)
                            #   SL hit → loss = posisi × (sl_dist / entry)
                            # Nilai ini TIDAK mempengaruhi PENEMPATAN SL/TP —
                            # hanya memengaruhi simulasi saldo.

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
        position_usd = round(balance * POSITION_SIZE_PCT / 100, 6)
        direction_sign = 1 if tp_p > entry else -1

        if close_price is not None:
            ref_price = close_price
        elif result == "tp":
            ref_price = tp_p
        elif result == "sl" and sl_p is not None:
            ref_price = sl_p
        else:
            return

        pnl_pct = (ref_price - entry) / entry * direction_sign
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
# TAMBAHAN BARU (START) — Fungsi Generator untuk /analyze
# ============================================================

def _session_of(entry_time):
    if not entry_time:
        return "unknown"
    h = datetime.fromtimestamp(entry_time, WIB).hour
    if h < 8:  return "Asia"
    if h < 16: return "London"
    return "New York"

def _calc_full_metrics(hist, starting_balance):
    """Analisa lengkap dari pnl_history: profit factor, drawdown, expectancy, sharpe, dst."""
    if not hist:
        return {}
    n = len(hist)
    wins   = [h["pnl_usd"] for h in hist if h["pnl_usd"] >= 0]
    losses = [h["pnl_usd"] for h in hist if h["pnl_usd"] < 0]
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    net_profit   = gross_profit - gross_loss

    equity = [starting_balance] + [h["balance_after"] for h in hist]
    peak, max_dd_usd = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd_usd = max(max_dd_usd, peak - e)
    max_dd_pct = (max_dd_usd / peak * 100) if peak > 0 else 0

    returns = [h["pct"] for h in hist]
    mean_r  = sum(returns) / n
    std_r   = (sum((r - mean_r) ** 2 for r in returns) / n) ** 0.5 if n > 1 else 0

    return {
        "total_trades": n,
        "net_profit": round(net_profit, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0,
        "win_rate_pct": round(len(wins) / n * 100, 2),
        "avg_win_usd": round(gross_profit / len(wins), 4) if wins else 0,
        "avg_loss_usd": round(gross_loss / len(losses), 4) if losses else 0,
        "expectancy_usd": round(net_profit / n, 4),
        "max_drawdown_usd": round(max_dd_usd, 4),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "recovery_factor": round(net_profit / max_dd_usd, 3) if max_dd_usd > 0 else 0,
        "sharpe_ratio": round(mean_r / std_r, 3) if std_r > 0 else 0,
        "return_pct": round(net_profit / starting_balance * 100, 2),
    }

def _generate_statistics_csv():
    """Generate statistics.csv — analisa performa lengkap (net profit, drawdown, sharpe, dst)."""
    with stat_lock:
        hist = list(stats["pnl_history"])
        balance = stats["balance"]

    metrics = _calc_full_metrics(hist, STARTING_BALANCE)
    metrics["balance"] = balance
    metrics["starting_balance"] = STARTING_BALANCE
    if not metrics:
        metrics = {"total_trades": 0, "balance": balance, "starting_balance": STARTING_BALANCE}

    df = pd.DataFrame([metrics])
    path = "/tmp/statistics.csv"
    df.to_csv(path, index=False)
    return path

def _generate_trade_csv():
    """Generate trade.csv dari pnl_history — termasuk detail sinyal
    (confidence/entry_label/rr/rsi/struct_h1) biar cukup buat diagnosis
    strategy_logic.py tanpa perlu data tambahan."""
    with stat_lock:
        hist = list(stats["pnl_history"])
    
    cols = ["symbol", "decision", "result", "pnl_pct", "pnl_usd",
            "entry", "tp", "sl", "exit_price", "entry_time", "exit_time",
            "confidence", "entry_label", "rr", "rsi", "struct_h1", "d1_bias"]
    if not hist:
        df = pd.DataFrame(columns=cols)
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
            "confidence": h.get("confidence", ""),
            "entry_label": h.get("entry_label", ""),
            "rr": h.get("rr", ""),
            "rsi": h.get("rsi", ""),
            "struct_h1": h.get("struct_h1", ""),
            "d1_bias": h.get("d1_bias", ""),
        })
    df = pd.DataFrame(rows)
    path = "/tmp/trade.csv"
    df.to_csv(path, index=False)
    return path

def _confidence_bucket(c):
    if c is None: return "unknown"
    if c < 50: return "<50"
    if c < 65: return "50-64"
    if c < 80: return "65-79"
    return "80+"

def _generate_research_context():
    """
    Generate research_context.json — analisa performa trading lengkap,
    dilengkapi breakdown per entry_label & confidence, dan data pending-
    cancel — supaya file ini SENDIRI cukup dipakai untuk review
    strategy_logic.py tanpa perlu data tambahan lain.
    (Data chart M1 TIDAK di sini lagi — sudah ada sumber terpisah dari
    candle_fetcher.py, jadi tidak perlu fetch ulang & redundan.)
    """
    log.info("[research] Generating research_context.json...")

    with stat_lock:
        trade_hist = list(stats["pnl_history"])
        balance = stats["balance"]
    with pending_cancel_lock:
        pc = dict(pending_cancel_stats)

    result = {
        "period": f"{len(trade_hist)} trade terakhir",
        "summary": _calc_full_metrics(trade_hist, STARTING_BALANCE),
        "performance_breakdown": {"by_coin": {}, "by_session": {}, "by_entry_label": {}, "by_confidence": {}},
        "pending_cancel": {**pc, "total": sum(pc.values())},
        "data_quality_notes": [],
        "worst_trades": [],
        "best_trades": [],
    }

    if trade_hist:
        by_coin, by_session, by_label, by_conf = {}, {}, {}, {}
        mislabeled = []
        for t in trade_hist:
            sym = t.get("symbol", "unknown")
            by_coin.setdefault(sym, []).append(t)
            by_session.setdefault(_session_of(t.get("entry_time")), []).append(t)
            by_label.setdefault(t.get("entry_label") or "unknown", []).append(t)
            by_conf.setdefault(_confidence_bucket(t.get("confidence")), []).append(t)
            # Sanity check: result="sl" mestinya selalu pnl<0, "trail"/"tp" pnl>=0.
            # Kalau tidak, kemungkinan data dari versi main.py lama (sebelum fix
            # reklasifikasi trail-vs-sl) — jangan dipercaya labelnya mentah-mentah.
            if t["result"] == "sl" and t["pnl_usd"] >= 0:
                mislabeled.append(t.get("symbol"))

        if mislabeled:
            result["data_quality_notes"].append(
                f"{len(mislabeled)} trade berlabel 'sl' tapi pnl positif ({', '.join(mislabeled)}) — "
                f"kemungkinan dari versi main.py sebelum fix reklasifikasi trail-vs-sl. "
                f"Jangan simpulkan pola menang/kalah dari label result mentah-mentah utk trade ini.")

        for sym, rows in by_coin.items():
            m = _calc_full_metrics(rows, STARTING_BALANCE)
            result["performance_breakdown"]["by_coin"][sym] = {
                "total": m["total_trades"], "win_rate": m["win_rate_pct"],
                "net_profit": m["net_profit"], "avg_pnl": round(m["net_profit"] / m["total_trades"], 4),
            }
        for sess, rows in by_session.items():
            m = _calc_full_metrics(rows, STARTING_BALANCE)
            result["performance_breakdown"]["by_session"][sess] = {
                "total": m["total_trades"], "win_rate": m["win_rate_pct"], "net_profit": m["net_profit"],
            }
        for label, rows in by_label.items():
            m = _calc_full_metrics(rows, STARTING_BALANCE)
            result["performance_breakdown"]["by_entry_label"][label] = {
                "total": m["total_trades"], "win_rate": m["win_rate_pct"], "net_profit": m["net_profit"],
            }
        for bucket, rows in by_conf.items():
            m = _calc_full_metrics(rows, STARTING_BALANCE)
            result["performance_breakdown"]["by_confidence"][bucket] = {
                "total": m["total_trades"], "win_rate": m["win_rate_pct"], "net_profit": m["net_profit"],
            }

        best  = sorted([t for t in trade_hist if t["pnl_usd"] >= 0], key=lambda x: x["pnl_usd"], reverse=True)[:3]
        worst = sorted([t for t in trade_hist if t["pnl_usd"] < 0], key=lambda x: x["pnl_usd"])[:3]
        result["best_trades"]  = [{"pnl": round(t["pnl_usd"], 4), "symbol": t.get("symbol"), "result": t["result"],
                                    "entry_label": t.get("entry_label"), "confidence": t.get("confidence")} for t in best]
        result["worst_trades"] = [{"pnl": round(t["pnl_usd"], 4), "symbol": t.get("symbol"), "result": t["result"],
                                    "entry_label": t.get("entry_label"), "confidence": t.get("confidence")} for t in worst]

    path = "/tmp/research_context.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info(f"[research] Selesai, disimpan ke {path}")
    return path

# ============================================================
# TAMBAHAN BARU (END)
# ============================================================

def fmt_signal_msg(sig):
    em  = "🟢" if sig["decision"]=="BUY" else "🔴"
    bar = "█"*(sig["confidence"]//10)+"░"*(10-sig["confidence"]//10)
    dir_label = "BULLISH" if sig["original_dir"]=="bull" else "BEARISH"
    d1_em = {"bullish":"📈","bearish":"📉","neutral":"➡️"}.get(sig.get("d1_bias","neutral"),"➡️")

    triggers = []
    ch15, ch1, fr = sig.get("choch_m15",{}), sig.get("choch_h1",{}), sig.get("failed_retest",{})
    if ch1.get("bearish_choch"):  triggers.append("CHoCH Bear H1")
    if ch1.get("bullish_choch"):  triggers.append("CHoCH Bull H1")
    if ch15.get("bearish_choch"): triggers.append("CHoCH Bear M15")
    if ch15.get("bullish_choch"): triggers.append("CHoCH Bull M15")
    if fr.get("failed_retest_sell"): triggers.append("Failed Retest Sell")
    if fr.get("failed_retest_buy"):  triggers.append("Failed Retest Buy")

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


def monitor_position(sym, pos):
    """
    Thread per-posisi: cek harga/TP/SL setiap MONITOR_SLEEP (10 detik),
    kirim pesan update ke Telegram tiap MONITOR_INTERVAL (15 menit) TANPA
    pernah menghentikan pengecekan harga di antaranya.
    Posisi hanya ditutup saat TP atau SL — tidak ada timeout otomatis.

    TRAILING STOP — DUA KOMPONEN, dipakai yang PALING PROTEKTIF:
      A) R-multiple ladder (TRAIL_R_LADDER): tiap profit capai ambang R
         tertentu (RELATIF ke risk trade itu sendiri, bukan persen
         absolut), SL dikunci ke sebagian dari R yang tercapai — proteksi
         cepat sejak awal, dicek tiap loop (tick-based). Redesign dari
         versi persen absolut setelah analisa mendalam menemukan: 51%
         trade py risk <0.6%, jadi threshold absolut lama butuh >1R dulu
         baru dapat proteksi; sementara 80.8% trade yg akhirnya SL
         SEMPAT profit dulu (median 0.56R) sebelum berbalik tanpa pernah
         terlindungi. R-ladder relatif memperbaiki ini utk semua ukuran
         risk sekaligus.
      B) Structure (swing point M15): SL mengikuti higher-low (BUY) /
         lower-high (SELL) terkonfirmasi terbaru — mengikuti price action
         asli, tidak overfit ke satu angka. Dicek tiap ~2 menit (throttled
         — swing point cuma berubah tiap candle M15 baru).
    Analisa forward-replay 375 trade yg exit via Trail: 62.4% memang akan
    balik ke SL asli kalau tidak ditrail (trail benar menyelamatkan),
    37.6% malah lanjut ke TP kalau tidak ditrail — TP cap BUKAN penyebab
    trade Trail terpotong (Trail selalu terjadi sebelum harga sempat ke
    TP), makanya fix-nya di kalibrasi trail (R-relatif), bukan hapus TP.
    SL trailing (dari kandidat manapun) HANYA boleh mengunci profit
    (searah entry->TP), tidak pernah mundur mendekati entry lagi.
    """
    sig     = pos["signal"]
    chat_id = pos["chat_id"]
    entry   = pos["entry"]
    tp_p    = sig["tp"]
    sl_p    = sig["sl"]           # SL berjalan — bisa naik oleh trailing
    is_buy  = sig["decision"] == "BUY"
    risk0   = abs(entry - sig["sl"])   # risk ASLI (SL awal, tidak ikut bergerak) — basis R-multiple
    locked_r_reached   = 0.0      # R terbesar yang sudah dikunci via TRAIL_R_LADDER
    next_struct_check  = 0.0      # throttle fetch M15 utk komponen structure

    next_update_at = time.time() + MONITOR_INTERVAL

    while True:
        with positions_lock:
            if sym not in positions: return

        # Manual /timeout SYMBOL — tutup paksa sesuai PnL riil saat ini:
        # floating positif dicatat sebagai TP, floating negatif sebagai SL.
        # Bukan selalu "SL" — itu akan mencatat kerugian penuh meski posisi
        # sedang untung saat ditutup.
        if pos.get("timeout_flag"):
            pos["timeout_flag"] = False
            price = get_price(sym) or entry
            pnl_pct = (price - entry) / entry * (1 if is_buy else -1)
            result  = "tp" if pnl_pct >= 0 else "sl"
            emoji   = "🎯" if result == "tp" else "🛑"
            tg_send(chat_id,
                f"⏭ <b>Ditutup Manual</b> — {sym} {emoji}\n"
                f"Harga: <code>{price:.6g}</code> | PnL: <b>{pnl_pct*100:+.2f}%</b>\n"
                f"Dicatat sebagai {result.upper()} (sesuai PnL riil saat ditutup)")
            close_position(sym, result, close_price=price)
            return

        price = get_price(sym)
        if price is None:
            time.sleep(MONITOR_SLEEP); continue

        # ── Kandidat A: R-multiple ladder (proteksi relatif ke risk trade
        # ini sendiri, bukan persen absolut) — lihat catatan TRAIL_R_LADDER
        # di atas utk alasan redesign ini. Dicek SEBELUM cek TP/SL supaya
        # SL baru langsung berlaku di iterasi yang sama.
        cand_a = None
        proxy_now = price
        pnl_r_now = (proxy_now - entry) / risk0 * (1 if is_buy else -1) if risk0 > 0 else 0
        best_r = 0.0
        for thr, lock in TRAIL_R_LADDER:
            if pnl_r_now >= thr:
                best_r = max(best_r, thr * lock)
        if best_r > locked_r_reached:
            locked_r_reached = best_r
            cand_a = entry + best_r * risk0 * (1 if is_buy else -1)

        # ── Kandidat B: structure (swing point M15), throttled ~2 menit ──
        cand_b = None
        if time.time() >= next_struct_check:
            next_struct_check = time.time() + 120
            try:
                df_recent = get_klines(sym, "15m", STRUCT_TRAIL_LOOKBACK)
                if df_recent is not None and len(df_recent) >= STRUCT_TRAIL_LB * 2 + 1:
                    sh_r, sl_r = swing_pts(df_recent, lb=STRUCT_TRAIL_LB)
                    if is_buy and sl_r:
                        cand_b = float(df_recent["low"].iloc[sl_r[-1]]) - entry * STRUCT_TRAIL_BUF_PCT
                    elif not is_buy and sh_r:
                        cand_b = float(df_recent["high"].iloc[sh_r[-1]]) + entry * STRUCT_TRAIL_BUF_PCT
            except Exception:
                cand_b = None
            pos["_struct_sl_cache"] = cand_b
        else:
            cand_b = pos.get("_struct_sl_cache")

        # SL baru = kandidat PALING PROTEKTIF di antara A & B yang ada,
        # cuma boleh mengunci profit (searah TP), tidak pernah melewati TP.
        cands = [c for c in (cand_a, cand_b) if c is not None]
        if cands:
            new_sl = max(cands) if is_buy else min(cands)
            improves = (new_sl > sl_p) if is_buy else (new_sl < sl_p)
            within_tp = (new_sl < tp_p) if is_buy else (new_sl > tp_p)
            if improves and within_tp:
                sl_p = new_sl
                pos["current_sl"] = sl_p   # sync ke shared state utk /trade
                src = "R-ladder" if (cand_a is not None and new_sl == cand_a) else "structure"
                tg_send(chat_id,
                    f"🔒 <b>Trailing SL — {sym}</b> ({src})\n"
                    f"SL dikunci ke <code>{sl_p:.6g}</code> "
                    f"({(sl_p-entry)/entry*100*(1 if is_buy else -1):+.2f}%)")

        # ── Cek TP / SL — verifikasi via candle M1 ─────────────────
        hit_tp = (price >= tp_p) if is_buy else (price <= tp_p)
        hit_sl = (price <= sl_p) if is_buy else (price >= sl_p)

        if hit_tp or hit_sl:
            order = check_tp_sl_order(sym, tp_p, sl_p, is_buy, lookback_min=3)
            if order is None:
                order = "tp" if hit_tp else "sl"

            if order == "tp":
                pct = abs(tp_p - entry) / entry * 100
                tg_send(chat_id,
                    f"🎯 <b>TAKE PROFIT</b> — {sym} 🎉\n"
                    f"TP: <code>{tp_p:.6g}</code>\n"
                    f"Profit: +{pct:.2f}%")
                close_position(sym, "tp")
                return
            else:
                confirmed_sl = False
                try:
                    df_m1 = get_klines(sym, "1m", 5)
                    if df_m1 is not None and not df_m1.empty:
                        last_closes = df_m1["close"].tail(3)
                        confirmed_sl = any(
                            (c <= sl_p) if is_buy else (c >= sl_p)
                            for c in last_closes
                        )
                    else:
                        # Tidak bisa fetch candle M1 — gunakan harga cache
                        # sebagai fallback agar SL tetap bisa terpicu
                        confirmed_sl = hit_sl
                except Exception:
                    confirmed_sl = hit_sl

                if confirmed_sl:
                    pct_final = (sl_p - entry) / entry * 100 * (1 if is_buy else -1)
                    is_profit_lock = pct_final >= 0
                    result_final = "trail" if is_profit_lock else "sl"
                    label = "TRAILING STOP (profit terkunci)" if is_profit_lock else "STOP LOSS"
                    emoji = "🔒" if is_profit_lock else "🛑"
                    tg_send(chat_id,
                        f"{emoji} <b>{label}</b> — {sym}\n"
                        f"Harga: <code>{price:.6g}</code> | SL: <code>{sl_p:.6g}</code> | "
                        f"PnL: <b>{pct_final:+.2f}%</b>")
                    # close_price = sl_p (SL AKTUAL yang sudah di-trail),
                    # bukan sig["sl"] asli — supaya P&L tercatat sesuai
                    # level SL sebenarnya. result dibedakan "trail" vs "sl"
                    # supaya win-rate tidak salah hitung profit sbg loss.
                    close_position(sym, result_final, close_price=sl_p)
                    return
                else:
                    # Notif dikirim sekali per episode sweep (flag reset
                    # begitu kondisi sweep hilang), loop istirahat
                    # MONITOR_SLEEP detik sebelum cek lagi.
                    if not pos.get("sweep_notified"):
                        tg_send(chat_id,
                            f"🔄 <b>Liquidity Sweep — {sym}</b>\n"
                            f"Wick menyentuh SL, candle M1 belum konfirmasi. Lanjut...")
                        pos["sweep_notified"] = True
                    time.sleep(MONITOR_SLEEP)
                    continue

        # Harga sudah tidak lagi menyentuh SL → reset flag notif sweep
        pos["sweep_notified"] = False

        # ── Update periodik — dikirim tanpa menghentikan pengecekan
        # harga. Loop tetap kembali ke atas tiap MONITOR_SLEEP dan tetap
        # mengecek TP/SL; hanya PESAN-nya yang dijadwalkan tiap 15 menit.
        if time.time() >= next_update_at:
            pnl_pct = (price - entry) / entry * 100 * (1 if is_buy else -1)
            tg_send(chat_id,
                f"📊 <b>Update 15m — {sym}</b>\n"
                f"Arah  : {'🟢 BUY' if is_buy else '🔴 SELL'}\n"
                f"Entry : <code>{entry:.6g}</code>\n"
                f"Harga : <code>{price:.6g}</code>\n"
                f"TP    : <code>{tp_p:.6g}</code>\n"
                f"SL    : <code>{sl_p:.6g}</code>\n"
                f"PnL   : <b>{pnl_pct:+.2f}%</b>")
            next_update_at = time.time() + MONITOR_INTERVAL

        time.sleep(MONITOR_SLEEP)


# ============================================================
# REAL TRADE — alur pending order, monitoring posisi, auto-stop
# ============================================================

def _open_pending_real(sym, signal, chat_id):
    """Pasang LIMIT order asli di Binance untuk entry (real trade)."""
    is_buy = signal["decision"] == "BUY"
    entry_target = signal["entry"]
    side = "BUY" if is_buy else "SELL"

    with positions_lock:
        if sym in positions: return
        if len(positions) >= MAX_POSITIONS: return
        positions[sym] = {
            "signal": signal, "entry": entry_target, "chat_id": chat_id,
            "entry_time": None, "timeout_flag": False, "status": "pending",
        }

    try:
        avail, _ = get_real_balance()
        if avail is not None and avail < MARGIN_USD:
            raise RuntimeError(f"saldo tersedia ${avail:.2f} < margin ${MARGIN_USD:.2f}")

        qty, margin_used, bumped = calc_auto_quantity(sym, entry_target, MARGIN_USD, LEVERAGE)
        if qty is None:
            raise RuntimeError("quantity di bawah minimum Binance meski margin sudah disesuaikan")

        set_leverage(sym, LEVERAGE)
        order = place_limit_order(sym, side, qty, entry_target)
        order_id = order["orderId"]

        with positions_lock:
            if sym not in positions: return
            positions[sym]["order_id"] = order_id
            positions[sym]["quantity"] = qty
            positions[sym]["margin_used"] = margin_used

        note = f" (margin disesuaikan ${MARGIN_USD:.2f}→${margin_used:.2f})" if bumped else ""
        tg_send(chat_id,
            f"🎯 <b>PENDING ORDER REAL</b> — {sym}\n\n"
            f"{fmt_signal_msg(signal)}\n\n"
            f"Qty: <code>{qty}</code> | Leverage: {LEVERAGE}x{note}\n"
            f"Order #{order_id} terpasang di Binance, menunggu terisi (maks 8 jam)")

        threading.Thread(target=_wait_entry_real, args=(sym, signal, chat_id, order_id), daemon=True).start()

    except Exception as e:
        with positions_lock:
            positions.pop(sym, None)
        _ban_coin(sym, f"gagal pasang order real ({e})")
        tg_send(chat_id, f"⚠️ <b>Skip {sym}</b> — Gagal pasang order: {e}")


def _wait_entry_real(sym, signal, chat_id, order_id):
    """Poll status order Binance sampai FILLED/CANCELED/expired 8 jam.
    Beda dari versi simulasi: TIDAK ada pengecekan 'SL sebelum entry'
    (tidak relevan untuk limit order asli — order pasti kena/fill dulu
    sebelum harga bisa lanjut ke level SL yang lebih jauh; exchange
    yang menangani itu sendiri)."""
    is_buy = signal["decision"] == "BUY"
    tp_p, entry_target = signal["tp"], signal["entry"]
    deadline = time.time() + 8 * 3600

    while time.time() < deadline:
        with positions_lock:
            if sym not in positions: return

        try:
            order = get_order_status(sym, order_id)
        except Exception as e:
            log.warning(f"[wait_entry_real] {sym}: {e}")
            time.sleep(REAL_TRADE_POLL_SLEEP); continue

        status = order.get("status")
        if status == "FILLED":
            avg_price = float(order.get("avgPrice") or 0) or entry_target
            _open_position_real(sym, signal, avg_price, chat_id, order)
            return
        if status in ("CANCELED", "EXPIRED", "REJECTED"):
            with positions_lock:
                positions.pop(sym, None)
            _ban_coin(sym, f"order {status.lower()}")
            _record_pending_cancel("binance_reject")
            tg_send(chat_id, f"⏭ <b>Pending Batal</b> — {sym}\nStatus order: {status}")
            return

        price_now = get_price(sym)
        if price_now is not None and status in ("NEW", "PARTIALLY_FILLED"):
            tp_hit = (price_now >= tp_p) if is_buy else (price_now <= tp_p)
            if tp_hit:
                cancel_order(sym, order_id)
                with positions_lock:
                    positions.pop(sym, None)
                _ban_coin(sym, "TP sebelum entry")
                _record_pending_cancel("tp_before_entry")
                tg_send(chat_id, f"⏭ <b>Pending Batal</b> — {sym}\nTP tersentuh sebelum entry, order dibatalkan.")
                return

        time.sleep(REAL_TRADE_POLL_SLEEP)

    cancel_order(sym, order_id)
    with positions_lock:
        positions.pop(sym, None)
    _ban_coin(sym, "pending expired")
    _record_pending_cancel("expired")
    tg_send(chat_id, f"⏰ <b>Pending Expired</b> — {sym}\nOrder dibatalkan (8 jam tidak terisi).")


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


def _open_position_real(sym, signal, actual_entry, chat_id, order_info):
    is_buy = signal["decision"] == "BUY"
    sl_v, tp_v = signal["sl"], signal["tp"]
    with positions_lock:
        fallback_qty = positions.get(sym, {}).get("quantity", 0)
    qty = abs(float(order_info.get("executedQty", 0))) or fallback_qty

    # Toleransi kecil (0.15%) sebelum dianggap "invalid" — batas tegas (<)
    # gampang salah tangkap selisih pembulatan tick antara harga sinyal
    # awal vs avgPrice hasil fill Binance, padahal setup-nya sebenarnya
    # masih valid. Gap yang BENERAN besar tetap ke-tangkap normal.
    tol = actual_entry * 0.0015
    geometry_ok = (sl_v - tol < actual_entry < tp_v + tol) if is_buy else (tp_v - tol < actual_entry < sl_v + tol)
    if not geometry_ok:
        _emergency_close(sym, is_buy, qty, chat_id,
            f"geometri invalid setelah order terisi (entry={actual_entry:.6g}, sl={sl_v:.6g}, tp={tp_v:.6g})")
        return

    # Fallback: harga sudah lewat SL sesaat setelah fill (gap/slippage buruk) ->
    # jangan pasang SL yang sudah basi, langsung auto-out.
    price_now = get_price(sym) or actual_entry
    sl_already_breached = (price_now <= sl_v) if is_buy else (price_now >= sl_v)
    if sl_already_breached:
        _emergency_close(sym, is_buy, qty, chat_id, "harga sudah melewati SL segera setelah order terisi")
        return

    sl_dist = abs(actual_entry - sl_v)
    tp_dist = abs(tp_v - actual_entry)
    actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0

    # KRITIS: posisi TIDAK BOLEH dibiarkan aktif tanpa SL terpasang di Binance.
    # Coba 3x (kadang gagal transient), dan kalau tetap gagal semua, WAJIB
    # auto-out — sebelumnya di sini cuma nge-warn lalu lanjut treat posisi
    # sebagai aktif normal, padahal SL-nya nggak pernah benar-benar ada di
    # Binance (ini penyebab kasus SL "hilang" & harga tembus tanpa nutup).
    tp_order_id = sl_order_id = None
    last_err = None
    for attempt in range(1, 4):
        try:
            tp_order, sl_order = place_tp_sl(sym, is_buy, tp_v, sl_v)
            tp_order_id, sl_order_id = tp_order["algoId"], sl_order["algoId"]
            last_err = None
            break
        except Exception as e:
            last_err = e
            log.warning(f"[open_position_real] percobaan {attempt}/3 gagal pasang TP/SL {sym}: {e}")
            if attempt < 3:
                time.sleep(2)

    if last_err is not None or sl_order_id is None:
        tg_send(chat_id, f"🚨 {sym}: GAGAL pasang SL setelah 3x percobaan ({last_err}) — "
                          f"posisi ditutup paksa, TIDAK dibiarkan tanpa proteksi.")
        _emergency_close(sym, is_buy, qty, chat_id, f"gagal pasang SL setelah 3x percobaan ({last_err})")
        return

    with positions_lock:
        if sym not in positions: return
        pos = positions[sym]
        pos.update({
            "entry": actual_entry, "entry_time": time.time(), "status": "active",
            "current_sl": sl_v, "quantity": qty,
            "tp_order_id": tp_order_id, "sl_order_id": sl_order_id,
        })

    tg_send(chat_id,
        f"⚡ <b>ENTRY REAL</b> — {sym}\n"
        f"Entry aktual: <code>{actual_entry:.6g}</code> | Qty: <code>{qty}</code>\n"
        f"TP: <code>{tp_v:.6g}</code> | SL: <code>{sl_v:.6g}</code>\n"
        f"RR: <b>1:{actual_rr:.2f}</b> | 📡 Dipantau...")

    threading.Thread(target=monitor_position_real, args=(sym, pos), daemon=True).start()


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


def monitor_position_real(sym, pos):
    """Pantau posisi real: deteksi closed (TP/SL Binance eksekusi sendiri
    otomatis, tidak perlu polling harga tiap detik untuk itu) + jalankan
    trailing (cancel+replace SL order kalau membaik)."""
    sig = pos["signal"]
    is_buy = sig["decision"] == "BUY"
    entry = pos["entry"]
    tp_p = sig["tp"]
    sl_p = pos["current_sl"]
    qty = pos["quantity"]
    chat_id = pos["chat_id"]
    tp_order_id = pos.get("tp_order_id")
    sl_order_id = pos.get("sl_order_id")
    risk0 = abs(entry - sl_p)
    locked_r = 0.0
    next_struct_check = 0.0
    next_sl_health_check = 0.0
    sl_replace_count = 0   # circuit-breaker: kalau kejadian "SL hilang" berulang terus, jangan spam selamanya

    while True:
        with positions_lock:
            if sym not in positions: return
            manual_close = positions[sym].get("timeout_flag")

        if manual_close:
            price = get_price(sym) or entry
            pnl_pct = (price - entry) / entry * (1 if is_buy else -1)
            result = "tp" if pnl_pct >= 0 else "sl"
            try:
                cancel_algo_order(tp_order_id); cancel_algo_order(sl_order_id)
                place_market_order(sym, "SELL" if is_buy else "BUY", qty, reduce_only=True)
            except Exception as e:
                log.error(f"[monitor_real] gagal tutup manual {sym}: {e}")
            tg_send(chat_id, f"⏭ <b>Ditutup Manual</b> — {sym}\nPnL: <b>{pnl_pct*100:+.2f}%</b>")
            close_position(sym, result, close_price=price)
            return

        try:
            real_pos = get_real_position(sym)
        except Exception as e:
            log.warning(f"[monitor_real] {sym} cek posisi gagal: {e}")
            time.sleep(REAL_TRADE_POLL_SLEEP); continue

        if real_pos is None:
            reason = _infer_close_reason(tp_order_id, sl_order_id)
            # Reklasifikasi jalan utk 'sl' MAUPUN 'unknown' (algoStatus query
            # gagal/ambigu) — sebelumnya cuma dicek utk reason=='sl', jadi
            # kasus 'unknown' lolos tanpa dicek & selalu jatuh ke label "sl"
            # walau SL-nya sebenarnya sudah ke-trail ke zona untung.
            if reason != "tp":
                if sl_p >= entry if is_buy else sl_p <= entry:
                    reason = "trail"
                elif reason == "unknown":
                    reason = "sl"   # fallback konservatif, masih rugi & status genuinely tidak jelas
            close_price = tp_p if reason == "tp" else sl_p
            # Jaring pengaman: cek & bersihkan sisa algo order (TP atau SL)
            # yang mungkin belum ke-cancel otomatis oleh Binance.
            cancel_all_algo_orders(sym)
            close_position(sym, reason, close_price=close_price)
            return

        # ── Jaring pengaman: verifikasi berkala SL BENERAN masih aktif di
        # Binance selagi posisi masih terbuka (bukan cuma pas awal buka).
        # Kalau ternyata hilang (order ke-cancel sendiri oleh Binance,
        # error yang tidak ketangkap sebelumnya, dsb) dan harga sudah
        # lewat level SL, auto-out SEKARANG — jangan biarkan posisi
        # tanpa proteksi sampai user sadar sendiri.
        if time.time() >= next_sl_health_check:
            next_sl_health_check = time.time() + 60
            sl_missing = sl_order_id is None
            if not sl_missing:
                try:
                    st = get_algo_order_status(sl_order_id).get("algoStatus")
                    sl_missing = st not in ("NEW",)
                except Exception as e:
                    log.warning(f"[monitor_real sl-check] {sym}: {e}")
            if sl_missing:
                sl_replace_count += 1
                if sl_replace_count > 3:
                    tg_send(chat_id, f"🚨 <b>SL berulang kali hilang</b> — {sym}\n"
                                      f"Sudah {sl_replace_count}x, ada yang tidak beres — auto-out daripada terus berulang.")
                    _emergency_close(sym, is_buy, qty, chat_id, f"SL hilang berulang {sl_replace_count}x (kemungkinan masalah lain)")
                    return
                price_now = get_price(sym) or entry
                sl_breached = (price_now <= sl_p) if is_buy else (price_now >= sl_p)
                if sl_breached:
                    tg_send(chat_id, f"🚨 <b>SL HILANG</b> — {sym}\nHarga sudah lewat level SL — auto-out sekarang.")
                    _emergency_close(sym, is_buy, qty, chat_id, "SL hilang saat posisi aktif & harga sudah tembus")
                    return
                try:
                    # Cancel dulu yang lama SEBELUM pasang baru (jaga-jaga
                    # ternyata masih ada & cuma keliru kedeteksi "hilang" —
                    # kalau langsung pasang baru tanpa cancel, bisa ada 2 SL
                    # order aktif barengan, salah satunya kena auto-cancel
                    # Binance, lalu health-check berikutnya deteksi "hilang"
                    # lagi -> siklus spam berulang. cancel_algo_order aman
                    # dipanggil walau order-nya memang sudah tidak ada.
                    cancel_algo_order(sl_order_id)
                    new_sl_order = place_sl_order(sym, is_buy, sl_p)
                    sl_order_id = new_sl_order["algoId"]
                    with positions_lock:
                        if sym in positions:
                            positions[sym]["sl_order_id"] = sl_order_id
                    tg_send(chat_id, f"⚠️ <b>SL dipasang ulang</b> — {sym}\nSempat hilang, sudah dipasang ulang di <code>{sl_p:.6g}</code>.")
                except Exception as e:
                    tg_send(chat_id, f"🚨 <b>SL HILANG & GAGAL dipasang ulang</b> — {sym}: {e}\n❗ TUTUP MANUAL SEKARANG!")

        price = get_price(sym) or entry
        pnl_r_now = (price - entry) / risk0 * (1 if is_buy else -1) if risk0 > 0 else 0
        best_r = locked_r
        for thr, lock in TRAIL_R_LADDER:
            if pnl_r_now >= thr:
                best_r = max(best_r, thr * lock)
        cand_a = entry + best_r * risk0 * (1 if is_buy else -1) if best_r > locked_r else None
        if best_r > locked_r:
            locked_r = best_r

        cand_b = None
        if time.time() >= next_struct_check:
            next_struct_check = time.time() + 120
            try:
                df_m15 = get_klines(sym, "15m", STRUCT_TRAIL_LOOKBACK)
                sh, sl_pts = swing_pts(df_m15, lb=STRUCT_TRAIL_LB)
                if is_buy and sl_pts:
                    cand_b = float(df_m15["low"].iloc[sl_pts[-1]]) - entry * STRUCT_TRAIL_BUF_PCT
                elif not is_buy and sh:
                    cand_b = float(df_m15["high"].iloc[sh[-1]]) + entry * STRUCT_TRAIL_BUF_PCT
            except Exception as e:
                log.debug(f"[monitor_real trail] {sym}: {e}")

        cands = [c for c in (cand_a, cand_b) if c is not None]
        if cands:
            proposed = max(cands) if is_buy else min(cands)
            improves = (proposed > sl_p) if is_buy else (proposed < sl_p)
            within_tp = (proposed < tp_p) if is_buy else (proposed > tp_p)
            if improves and within_tp:
                try:
                    old_sl = sl_p
                    cancel_algo_order(sl_order_id)   # SL lama WAJIB dihapus dulu sebelum pasang yang baru
                    new_sl_order = place_sl_order(sym, is_buy, proposed)
                    sl_order_id = new_sl_order["algoId"]
                    sl_p = proposed
                    with positions_lock:
                        if sym in positions:
                            positions[sym]["current_sl"] = sl_p
                            positions[sym]["sl_order_id"] = sl_order_id
                    locked_pct = (sl_p - entry) / entry * 100 * (1 if is_buy else -1)
                    label = "Profit terkunci" if locked_pct >= 0 else "Risiko dikurangi"
                    tg_send(chat_id,
                        f"🔒 <b>TRAILING SL</b> — {sym}\n"
                        f"SL digeser: <code>{old_sl:.6g}</code> → <code>{sl_p:.6g}</code>\n"
                        f"{label}: <b>{locked_pct:+.2f}%</b>")
                except Exception as e:
                    log.warning(f"[monitor_real trail] gagal update SL {sym}: {e}")

        time.sleep(REAL_TRADE_POLL_SLEEP)


def autostop_loop(chat_id):
    """Background: pantau saldo real, auto /stop kalau drawdown dari peak > AUTOSTOP_PCT."""
    global auto_mode, peak_real_balance
    while True:
        try:
            if REAL_TRADE_ENABLED:
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
    """
    Broadcaster utama — non-blocking:
    - Scan berjalan di thread terpisah agar tidak block loop utama
    - Monitor per-posisi juga thread terpisah (sudah ada)
    - Loop utama hanya koordinasi: cek slot, launch scan/monitor
    """
    global auto_mode
    tg_send(chat_id,
        "🤖 <b>SMC Signal Broadcaster dimulai!</b>\n\n"
        "• Scan koin → catat sinyal → pantau tiap 15 menit\n"
        f"• Maks {MAX_POSITIONS} posisi bersamaan\n"
        "• Posisi ditutup hanya saat TP atau SL\n\n"
        "/stop untuk berhenti | /timeout SYMBOL untuk tutup paksa\n"
        "/trade untuk lihat semua posisi aktif")

    scanning = False          # flag: apakah scan sedang berjalan
    scan_lock = threading.Lock()

    def _do_scan():
        nonlocal scanning
        try:
            signal = run_scan_once(chat_id)
            if not auto_mode or signal is None:
                return

            sym = signal["symbol"]
            with positions_lock:
                if sym in positions: return
                if len(positions) >= MAX_POSITIONS: return

            # ── REAL TRADE: pasang limit order asli, exchange yang urus fill ──
            # (tidak butuh split langsung/pending seperti simulasi — order
            # limit otomatis fill instan kalau harga sudah di zona entry)
            if REAL_TRADE_ENABLED:
                _open_pending_real(sym, signal, chat_id)
                return

            entry_target = signal["entry"]
            current      = signal["price"]
            is_buy       = signal["decision"] == "BUY"
            tp_p         = signal["tp"]
            entry_label  = signal.get("entry_label", "market")

            already_at_entry = (
                (is_buy     and current <= entry_target * 1.002) or
                (not is_buy and current >= entry_target * 0.998)
            )

            if already_at_entry or entry_label == "market":
                # Langsung masuk — daftarkan dulu di positions supaya
                # _open_position (yang mengasumsikan entry sudah ada
                # sebagai pending) tidak langsung return diam-diam.
                actual_entry = get_price(sym) or current
                with positions_lock:
                    if sym in positions: return
                    if len(positions) >= MAX_POSITIONS: return
                    positions[sym] = {
                        "signal"      : signal,
                        "entry"       : entry_target,
                        "chat_id"     : chat_id,
                        "entry_time"  : None,
                        "timeout_flag": False,
                        "status"      : "pending",
                    }
                _open_position(sym, signal, actual_entry, chat_id, "langsung")
            else:
                # Daftarkan dulu sebagai pending agar tidak di-scan ulang
                with positions_lock:
                    if sym in positions: return
                    if len(positions) >= MAX_POSITIONS: return
                    positions[sym] = {
                        "signal"      : signal,
                        "entry"       : entry_target,
                        "chat_id"     : chat_id,
                        "entry_time"  : None,        # belum entry, set saat terpicu
                        "timeout_flag": False,
                        "status"      : "pending",
                    }

                dist_pct = abs(entry_target - current) / current * 100
                tg_send(chat_id,
                    f"🎯 <b>PENDING ORDER</b> — {sym}\n\n"
                    f"{fmt_signal_msg(signal)}\n\n"
                    f"⏳ Menunggu harga ke zona entry\n"
                    f"Harga kini : <code>{current:.6g}</code>\n"
                    f"Entry zone : <code>{entry_target:.6g}</code> ({entry_label})\n"
                    f"Jarak      : {dist_pct:.2f}%")
                threading.Thread(
                    target=_wait_entry,
                    args=(sym, signal, chat_id),
                    daemon=True
                ).start()
        finally:
            with scan_lock:
                scanning = False

    def _wait_entry(sym, signal, chat_id):
        """Thread terpisah — tunggu harga ke zona entry. /stop tidak
        membatalkan pending; hanya menghentikan scan koin baru.

        PATCH PENDING-CONFIRM: SL-sebelum-entry dulu dicek dari tick
        price mentah tiap 10 detik (price_now<=sl_p) — terlalu sensitif.
        sl_p di sini = level INVALIDASI ZONA itu sendiri (tepi jauh
        OB/FVG + noise buffer kecil, lihat analyze_setup()), seringkali
        cuma noise-buffer kecil (0.3-0.8×ATR) dari entry — wick sesaat
        gampang menyentuhnya lalu balik lagi padahal zona sebenarnya
        masih valid & akan terisi. Sekarang butuh KONFIRMASI CANDLE
        CLOSE M15 (meniru proteksi anti-whipsaw yang sebelumnya cuma ada
        di posisi aktif via check_tp_sl_order — sekarang juga berlaku di
        fase pending). TP-before-entry & entry-fill TETAP tick-based
        (permisif) — tidak ada ruginya di situ: TP kena berarti peluang
        memang lewat, dan entry di sentuhan wick MENGUNTUNGKAN trader.
        """
        entry_target = signal["entry"]
        is_buy       = signal["decision"] == "BUY"
        tp_p         = signal["tp"]
        sl_p         = signal["sl"]
        deadline     = time.time() + 8 * 3600
        next_sl_check = 0.0        # throttle fetch M15 (candle baru tiap 15 menit)
        last_m15_ts   = None

        while time.time() < deadline:
            with positions_lock:
                if sym not in positions: return

            price_now = get_price(sym)
            if price_now is None:
                time.sleep(MONITOR_SLEEP); continue

            # TP tersentuh sebelum entry → sinyal basi, hapus pending
            tp_hit = (price_now >= tp_p) if is_buy else (price_now <= tp_p)
            if tp_hit:
                with positions_lock:
                    positions.pop(sym, None)
                _ban_coin(sym, "TP sebelum entry")
                tg_send(chat_id,
                    f"⏭ <b>Pending Batal</b> — {sym}\n"
                    f"TP tersentuh sebelum entry. Skip.")
                return

            # SL sebelum entry — BUTUH KONFIRMASI CANDLE CLOSE M15 (lihat
            # docstring). Dicek setiap ~60 detik saja (cukup, candle M15
            # baru muncul tiap 15 menit) supaya tidak fetch klines tiap
            # 10 detik terus-menerus.
            if time.time() >= next_sl_check:
                next_sl_check = time.time() + 60
                try:
                    df_chk = get_klines(sym, "15m", 3)
                    if df_chk is not None and len(df_chk) >= 2:
                        closed_row = df_chk.iloc[-2]   # candle terakhir yg SUDAH close
                        ts_closed  = df_chk.index[-2]
                        if last_m15_ts is None or ts_closed != last_m15_ts:
                            last_m15_ts = ts_closed
                            close_v = float(closed_row["close"])
                            sl_confirmed = (close_v <= sl_p) if is_buy else (close_v >= sl_p)
                            if sl_confirmed:
                                with positions_lock:
                                    positions.pop(sym, None)
                                _ban_coin(sym, "SL sebelum entry")
                                tg_send(chat_id,
                                    f"⏭ <b>Pending Batal</b> — {sym}\n"
                                    f"Candle M15 close mengonfirmasi SL sebelum entry. Skip.")
                                return
                except Exception as e:
                    log.debug(f"[_wait_entry sl-confirm] {sym}: {e}")

            # Harga mencapai zona entry
            entry_hit = (
                (is_buy     and price_now <= entry_target * 1.003) or
                (not is_buy and price_now >= entry_target * 0.997)
            )
            if entry_hit:
                _open_position(sym, signal, price_now, chat_id, "terpicu")
                return

            time.sleep(MONITOR_SLEEP)

        # Expired — hapus pending
        with positions_lock:
            positions.pop(sym, None)
        _ban_coin(sym, "pending expired")
        tg_send(chat_id,
            f"⏰ <b>Pending Expired</b> — {sym}\n"
            f"Harga tidak mencapai zona entry dalam 8 jam. Skip.")

    def _open_position(sym, signal, actual_entry, chat_id, mode_label):
        """Upgrade posisi dari pending ke aktif dan mulai monitor."""
        is_buy = signal["decision"] == "BUY"
        sl_v, tp_v = signal["sl"], signal["tp"]

        # Validasi geometri dulu — SL dan TP wajib di sisi yang benar dari
        # entry aktual. Wajib dicek sebelum rasio RR, karena rasio abs(jarak)
        # bisa tampak valid (>= MIN_RR) walau posisinya sebenarnya terbalik
        # (mis. harga gap lewat SL sebelum entry sempat tersentuh).
        geometry_ok = (sl_v < actual_entry < tp_v) if is_buy else (tp_v < actual_entry < sl_v)
        if not geometry_ok:
            with positions_lock:
                positions.pop(sym, None)
            _ban_coin(sym, "geometri invalid")
            tg_send(chat_id,
                f"⚠️ <b>Skip {sym}</b> — Geometri SL/TP tidak valid di entry aktual\n"
                f"Entry: <code>{actual_entry:.6g}</code> | "
                f"TP: <code>{tp_v:.6g}</code> | SL: <code>{sl_v:.6g}</code>")
            return

        # Verifikasi RR masih valid di harga entry aktual.
        # TP/SL dihitung dari discount_entry (analisis), tapi posisi
        # dibuka di harga nyata — selisihnya bisa membuat RR < MIN_RR.
        sl_dist = abs(actual_entry - sl_v)
        tp_dist = abs(tp_v - actual_entry)
        actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if actual_rr < MIN_RR:
            with positions_lock:
                positions.pop(sym, None)
            _ban_coin(sym, "RR gagal di entry aktual")
            tg_send(chat_id,
                f"⚠️ <b>Skip {sym}</b> — RR tidak memenuhi di entry aktual\n"
                f"Entry: <code>{actual_entry:.6g}</code> | "
                f"TP: <code>{tp_v:.6g}</code> | SL: <code>{sl_v:.6g}</code>\n"
                f"RR aktual: <b>1:{actual_rr:.2f}</b> (min 1:{MIN_RR})")
            return

        with positions_lock:
            if sym not in positions: return   # sudah dihapus (expired/batal)
            pos = positions[sym]
            pos["entry"]      = actual_entry
            pos["entry_time"] = time.time()
            pos["status"]     = "active"
            pos["timeout_flag"] = False   # reset — flag lama (saat masih pending) tidak boleh menutup posisi baru ini
            pos["current_sl"] = sl_v      # SL awal = SL asli, akan naik oleh trailing di monitor_position

        tg_send(chat_id,
            f"⚡ <b>ENTRY {mode_label.upper()}</b> — {sym}\n"
            f"Entry aktual: <code>{actual_entry:.6g}</code>\n"
            f"TP: <code>{tp_v:.6g}</code> | SL: <code>{sl_v:.6g}</code>\n"
            f"RR: <b>1:{actual_rr:.2f}</b> | 📡 Dipantau tiap 15 menit...")

        threading.Thread(
            target=monitor_position,
            args=(sym, pos),
            daemon=True
        ).start()

    SCAN_INTERVAL_SEC = 120   # jeda minimum antar SIKLUS scan penuh (50 koin).
    # Sebelumnya cuma dikasih jeda 5 detik antar PERCOBAAN launch -- kalau satu
    # scan selesai lebih cepat dari itu, langsung scan lagi nyaris tanpa henti
    # (150+ request/scan × berkali-kali/menit). Ini penyebab utama sering kena
    # limit/ban Binance. M15 candle baru cuma muncul tiap 15 menit, jadi scan
    # tiap 2 menit sudah lebih dari cukup responsif tanpa membebani API.
    last_scan_started_at = 0.0

    while auto_mode:
        with positions_lock:
            n_pos = len(positions)

        # Slot penuh — tunggu saja
        if n_pos >= MAX_POSITIONS:
            time.sleep(5)
            continue

        # Kalau scan sedang berjalan — jangan launch scan baru
        with scan_lock:
            already_scanning = scanning
            if not already_scanning:
                scanning = True

        if already_scanning:
            time.sleep(5)
            continue

        # Jeda minimum antar SIKLUS scan (bukan cuma antar percobaan) —
        # kalau belum waktunya, lepas flag scanning lagi & tunggu.
        elapsed = time.time() - last_scan_started_at
        if elapsed < SCAN_INTERVAL_SEC:
            with scan_lock:
                scanning = False
            time.sleep(min(5, SCAN_INTERVAL_SEC - elapsed))
            continue

        # Launch scan di background
        last_scan_started_at = time.time()
        threading.Thread(target=_do_scan, daemon=True).start()

        # Jeda antar percobaan launch (flag scanning yang cegah overlap)
        time.sleep(5)

    tg_send(chat_id, "⏹ <b>Scanning dihentikan.</b>\n\n" + fmt_stats())



# ═════════════════════════════════════════════
# PESAN STATIS
# ═════════════════════════════════════════════
GREETING=(
    "👋 <b>SMC Signal Broadcaster</b>\n\n"
    f"Scan → sinyal → pantau max {MAX_POSITIONS} posisi bersamaan (update tiap 15 menit)\n"
    "Posisi ditutup hanya saat TP atau SL\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "/start               — Menu ini\n"
    "/auto                — Mulai broadcaster\n"
    "/stop                — Hentikan scanning (posisi aktif tetap dipantau)\n"
    "/trade               — Lihat semua posisi aktif\n"
    "/max                 — Lihat/ubah max posisi + info batas API\n"
    "/confidence_min      — Lihat/ubah ambang confidence minimum\n"
    "/leverage            — Lihat/ubah leverage (real trade)\n"
    "/margin              — Lihat/ubah margin awal per trade (real trade)\n"
    "/autostop            — Lihat/ubah threshold auto-stop drawdown\n"
    "/timeout SYMBOL      — Tutup paksa posisi tertentu\n"
    "/timeout             — Tutup paksa semua posisi\n"
    "/stats               — Statistik + saldo\n"
    "/backtest             — 20 trade terakhir (evaluasi)\n"
    "/banned              — Daftar koin ban (+ SYMBOL utk ban permanen)\n"
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
    return (
        "ℹ️ <b>Metode Analisis</b>\n\n"
        "<b>Tahap 1 — BIAS (struktur besar dulu):</b>\n"
        "• Market Structure H1 (HH/HL vs LH/LL) — bobot terbesar\n"
        "• CHoCH H1 (wajib body close) — perubahan bias/karakter pasar\n"
        "• D1 bias (EMA ATAU struktur harian, salah satu cukup) — konteks\n"
        "  makro yang H1 sendiri tak bisa lihat; ikut scoring + hard block\n"
        "  kalau berlawanan total dengan arah akhir\n"
        "• EMA H1 trend alignment (9/21/50/200)\n"
        "• RSI 14 M15 — momentum filter tambahan\n\n"
        "<b>Tahap 2 — SETUP (konfirmasi entry price-action/SMC):</b>\n"
        "• BOS (cukup shadow) + CHoCH M15 (wajib body close)\n"
        "• Failed Retest M15 & H1 — trigger entry paling valid\n"
        "• Validitas & tipe pullback (corrective/sweeping/aggressive)\n"
        "• Pin bar rejection + pola Fakey (false breakout)\n"
        "• Liquidity Run vs Sweep/Swift\n"
        "• OTE 0.62-0.79 (hanya bonus, wajib CHoCH/FVG pendukung)\n"
        "• MACD/Bollinger/Volume M15 — momentum confluence ringan\n\n"
        "<b>Tahap 3 — GATE:</b>\n"
        "Setup yang berlawanan arah dengan bias struktural dilemahkan\n"
        "drastis (bukan dijumlah rata seperti indikator lepas biasa) —\n"
        "struktur besar diperlakukan sebagai filter wajib.\n"
        "Inducement-aware: turunkan confidence jika breakout baru\n"
        "terjadi tanpa CHoCH konfirmasi.\n\n"
        "<b>Tahap 4 — Penentuan SL (invalidation level):</b>\n"
        "SL = seberang titik entry (OB/FVG) itu sendiri — kalau tersentuh,\n"
        "struktur TERBUKTI gagal, bukan liquidity pool jauh.\n"
        "Buffer noise dari ATR gabungan M15+H1 (bukan M15 saja) — mencegah\n"
        "SL kena wick biasa saat harga baru keluar dari candle spike besar\n"
        "lalu masuk fase konsolidasi sempit (M15 'tenang' tapi semu).\n\n"
        "<b>Tahap 5 — Pemilihan TP (tier-based):</b>\n"
        "RR ≥ 1:2 WAJIB, tapi utamakan level PALING KUAT:\n"
        "1) eq highs/lows  2) supply/demand  3) FVG\n"
        "4) swing H1  5-6) Fibonacci extension (1.272/1.618)*\n"
        "*hanya aktif kalau H4 trend + RSI H4 + CHoCH M15 mendukung —\n"
        " level ini belum 'terbukti' market, jadi paling lemah & butuh\n"
        " konfirmasi ekstra. Selalu dievaluasi bareng level lain, bukan\n"
        " cabang khusus penyelamat RR gagal.\n"
        "Supply/demand & FVG diprioritaskan yang FRESH (belum tersentuh)\n"
        "dan FVG breakaway (candle-3 searah) di atas rejection.\n\n"
        "<b>Tahap 6 — Entry diskon (skor kualitas − penalti jarak):</b>\n"
        "1) OB fresh & selaras fib diskon/premium  2) FVG breakaway/fresh\n"
        "3) Equal highs/lows  4) Fibonacci ADAPTIF (0.236-0.382 trend\n"
        "SANGAT kuat, 0.382-0.5 trend kuat, 0.618-0.786 trend lemah) —\n"
        "keempatnya kini SATU pool skor yang sama, zona lebih dekat\n"
        "lebih diprioritaskan drpd zona jauh dgn kualitas sebanding\n\n"
        "<b>Tahap 7 — Trailing Stop (setelah posisi aktif):</b>\n"
        "Dua komponen, dipakai yang PALING PROTEKTIF:\n"
        "• R-ladder: 0.5R→kunci15% | 1.0R→35% | 1.5R→50% | 2.0R→65% |\n"
        "  2.8R→80% | 3.5R→85% (R = kelipatan risk/jarak-SL trade itu\n"
        "  sendiri, BUKAN persen absolut — proteksi tetap dini walau SL rapat)\n"
        "• Structure: SL mengikuti higher-low/lower-high M15 terbaru\n"
        "SL trailing cuma boleh mengunci profit (searah TP), tak pernah\n"
        "mundur ke entry. Kalau SL trailing tersentuh dgn profit terkunci,\n"
        "dicatat 'Trail' (bukan 'SL') — tetap dihitung menang di win-rate.\n\n"
        f"Min RR: 1:{MIN_RR} | Min Confidence: {MIN_CONFIDENCE}%\n"
        f"TF: H1 (bias) + M15 (entry) + H4 (fib gate)\n"
        f"Model P&L   : posisi {POSITION_SIZE_PCT:.0f}% saldo × % jarak SL/TP aktual\n"
        f"  → SL dekat (0.5%) = loss kecil | SL jauh (4%) = loss lebih besar\n"
        f"  → P&L murni dari level struktural analisis, bukan fixed -2%\n"
        f"Modal simulasi: ${STARTING_BALANCE:.2f}"
    )


# ═════════════════════════════════════════════
# BOT LOOP
# ═════════════════════════════════════════════
def bot_loop():
    global auto_mode, auto_thread, active_chat_id, timeout_flag, MAX_POSITIONS, MIN_CONFIDENCE, LEVERAGE, MARGIN_USD, AUTOSTOP_PCT, peak_real_balance

    # Set active_chat_id ke ALLOWED_USER_ID SEJAK AWAL — di chat pribadi
    # Telegram, chat_id sama dengan user_id, jadi bot bisa kirim pesan
    # proaktif (termasuk "Bot Siap" & notifikasi darurat) SEBELUM user
    # mengirim perintah apa pun. Sebelumnya active_chat_id cuma None
    # sampai user chat duluan, jadi notifikasi penting tidak pernah sampai.
    if ALLOWED_USER_ID:
        active_chat_id = ALLOWED_USER_ID

    # Cek koneksi Binance dipindah ke THREAD TERPISAH di background —
    # TIDAK BOLEH memblokir atau mematikan polling Telegram. SEBELUMNYA
    # cek ini ada di jalur utama bot_loop(): kalau ping gagal 10x
    # (mis. IP Render kena rate-limit/geo-block sementara oleh Binance),
    # baris "return" bikin SELURUH bot_loop() — termasuk polling
    # Telegram — berhenti total dan tidak pernah jalan lagi. Itulah
    # penyebab utama bot "tidak bisa diakses lewat Telegram" sebelumnya.
    def _check_binance():
        for i in range(10):
            try:
                fapi_get("/fapi/v1/ping")
                log.info("Binance OK!")
                return
            except Exception as e:
                log.warning(f"[binance-ping] retry {i+1}/10: {e}")
                time.sleep(10)
        log.error("Binance tidak bisa dijangkau setelah 10x percobaan. "
                   "Bot tetap jalan — scan & harga otomatis fallback ke "
                   "Bybit/CoinGecko selama Binance bermasalah.")
    threading.Thread(target=_check_binance, daemon=True).start()

    offset=None
    log.info("Bot siap.")
    if ALLOWED_USER_ID:
        tg_send(ALLOWED_USER_ID,
            "✅ <b>Bot Siap</b>\n"
            "Semua sistem sudah menyala dan siap menerima perintah.\n"
            "Ketik /start untuk melihat menu.")

    while True:
        try:
            for upd in tg_updates(offset):
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
                    tg_send(chat_id,GREETING)
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
                    # Jalankan di background thread agar tidak block loop
                    def _run_analyze(cid):
                        try:
                            tg_send(cid, "🔄 Memulai riset historis 15 koin (3 bulan)...\nIni bisa memakan waktu 3-5 menit.")
                            
                            # Generate 3 file
                            stats_csv = _generate_statistics_csv()
                            trade_csv = _generate_trade_csv()
                            context_json = _generate_research_context()
                            
                            # Kirim ke Telegram
                            tg_send(cid, "✅ Riset selesai! Mengirim file...")
                            tg_send_document(cid, stats_csv, caption="📊 statistics.csv")
                            tg_send_document(cid, trade_csv, caption="📋 trade.csv")
                            tg_send_document(cid, context_json, caption="🧠 research_context.json")
                            
                            # Ringkasan
                            with open(context_json, "r") as f:
                                ctx = json.load(f)
                            summary = ctx.get("summary", {})
                            total = summary.get("total_trades", 0)
                            wr = summary.get("win_rate", 0)
                            tg_send(cid,
                                f"📊 <b>Ringkasan Riset</b>\n"
                                f"Total trade: {total}\n"
                                f"Win Rate: {wr}%\n\n"
                                f"File sudah dikirim. Jalankan researcher.py di laptop untuk analisis lebih lanjut.")
                        except Exception as e:
                            log.error(f"[analyze] Error: {e}")
                            tg_send(cid, f"❌ Error saat menjalankan riset:\n<code>{str(e)[:200]}</code>")
                    
                    threading.Thread(target=_run_analyze, args=(chat_id,), daemon=True).start()
                    tg_send(chat_id, "⏳ Riset dimulai di background. Anda akan menerima file dalam beberapa menit.")
# ============================================================
# TAMBAHAN BARU (END)
# ============================================================
                # ============================================================
# TAMBAHAN BARU (START) — Handler /ganti (Upload Otak Baru via GitHub API)
# ============================================================
                elif text == "/ganti":
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

                        # 2b. Validasi full_analyze() ADA di file baru — tolak dari awal
                        # kalau tidak ada, jangan sampai ke-commit/reload file yang salah.
                        check_ns = {}
                        try:
                            exec(compiled, check_ns)
                        except Exception as e:
                            tg_send(chat_id, f"❌ File error saat dijalankan (bukan cuma sintaks):\n<code>{e}</code>")
                            continue
                        if "full_analyze" not in check_ns or not callable(check_ns["full_analyze"]):
                            tg_send(chat_id, "❌ File ini tidak punya fungsi full_analyze() — "
                                              "ditolak, bukan strategy_logic.py yang valid.")
                            continue
                        
                        # 3. Commit ke GitHub via API
                        try:
                            _commit_to_github(file_content, "strategy_logic.py", f"Update strategy_logic via Telegram /ganti")
                            tg_send(chat_id, "✅ File berhasil di-commit ke GitHub!")
                        except Exception as e:
                            tg_send(chat_id, f"❌ Gagal commit ke GitHub:\n<code>{str(e)[:200]}</code>")
                            continue

                        # 4. Tulis ke file LOKAL yang sedang jalan di Render — WAJIB sebelum
                        # reload, kalau tidak importlib.reload() cuma baca ulang isi lama
                        # dari disk (ini penyebab /ganti kelihatan sukses tapi strategi
                        # sebenarnya tidak pernah berubah).
                        local_path = sys.modules["strategy_logic"].__file__ if "strategy_logic" in sys.modules \
                                     else os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_logic.py")
                        with open(local_path, "w", encoding="utf-8") as f:
                            f.write(file_content)

                        # 5. Reload modul strategy_logic dari disk (sekarang sudah ter-update)
                        import importlib
                        if "strategy_logic" in sys.modules:
                            importlib.reload(sys.modules["strategy_logic"])
                        else:
                            import strategy_logic
                            importlib.reload(strategy_logic)
                        
                        # 6. Update global namespace main.py dengan fungsi-fungsi baru
                        strat_mod = sys.modules["strategy_logic"]
                        for attr in dir(strat_mod):
                            if not attr.startswith("_"):
                                globals()[attr] = getattr(strat_mod, attr)
                        
                        tg_send(chat_id, "✅ Strategy logic berhasil di-reload dan AKTIF tanpa restart!")
                        log.info("[OTAK] Strategy logic di-reload via /ganti (GitHub commit + file lokal)")
                        
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
                elif text.startswith("/confidence_min"):
                    parts = text.split()
                    # ── /confidence_min (tampilkan nilai saat ini) ─────────
                    if len(parts) == 1:
                        tg_send(chat_id,
                            f"🎯 <b>Confidence Minimum</b>\n\n"
                            f"Saat ini: <b>{MIN_CONFIDENCE}%</b>\n\n"
                            f"Sinyal dengan confidence di bawah angka ini akan\n"
                            f"diabaikan sebelum masuk pertimbangan RR/entry.\n"
                            f"Makin tinggi → sinyal lebih jarang tapi lebih\n"
                            f"selektif. Makin rendah → sinyal lebih sering\n"
                            f"tapi makin banyak setup lemah ikut lolos.\n\n"
                            f"<b>Ubah: /confidence_min 50</b>")
                    # ── /confidence_min N (ubah nilai) ─────────────────────
                    elif len(parts) == 2:
                        try:
                            n = int(parts[1])
                            if n < 0 or n > 99:
                                tg_send(chat_id,
                                    f"❌ Nilai harus antara 0–99.\n"
                                    f"Contoh: /confidence_min 50")
                            else:
                                old = MIN_CONFIDENCE
                                MIN_CONFIDENCE = n
                                tg_send(chat_id,
                                    f"✅ Confidence minimum diubah: "
                                    f"<b>{old}% → {MIN_CONFIDENCE}%</b>")
                        except ValueError:
                            tg_send(chat_id,"❌ Format salah. Contoh: /confidence_min 50")
                    else:
                        tg_send(chat_id,"❌ Format: /confidence_min  atau  /confidence_min 50")

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

            time.sleep(1)
        except Exception as e:
            log.error(f"[bot] {e}")
            time.sleep(5)


if __name__=="__main__":
    # Flask dijalankan di thread sendiri PALING AWAL supaya port langsung
    # bind & terdeteksi Render, tidak menunggu inisialisasi bot/WS selesai.
    threading.Thread(target=run_flask, daemon=True).start()
    ws_feed.start()
    threading.Thread(target=_price_cache_loop, daemon=True).start()
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
