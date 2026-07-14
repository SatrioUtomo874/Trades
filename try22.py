#!/usr/bin/env python3
"""
SMC Signal Broadcaster — Forward Entry Strategy
Logika: Analisis H1+M15+D1 → sinyal searah → entry diskon OB/FVG/Fib → TP/SL struktural
Render.com | python main.py
"""

import os, time, logging, threading
from collections import deque
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
MAX_PRICE       = 80.0
TOP_N_COINS     = 50
MIN_RR              = 2.0
MONITOR_SLEEP       = 10
MAX_POSITIONS       = 20
MONITOR_INTERVAL    = 15 * 60
MIN_CONFIDENCE      = 50    # ambang confidence minimum sinyal — diatur via /confidence_min
# TRAIL_R_LADDER = [(ambang_R, lock_ratio), ...] — begitu profit (dlm
# kelipatan risk R trade itu sendiri) capai ambang, kunci lock_ratio dari
# level itu (dlm R juga). SL final = kandidat PALING PROTEKTIF antara
# R-ladder ini vs komponen structure (swing-point) di bawah.
#
# RIWAYAT TUNING (evolusi, dari kecil ke besar sampel):
#   v1 flat-percent (bukan R)         → banyak masalah, digantikan R-based
#   v2 R-based (0.5/1.0/1.5/2.2/3.0R) → validasi 543 trade (M15 bar)
#   v3 ekor diperketat (2.0/2.8R)     → validasi silang 110+543 trade (M15)
#   v4 lock dinaikkan semua tahap     → validasi 356 trade (M1 presisi,
#                                        5 koin) — TERNYATA overfit, PnL
#                                        malah lebih rendah di sampel besar
#   v5 FINAL — kembali ke lock v3 + tambah 1 tahap ekor (3.5R) → divalidasi
#      di 1113 trade NYATA (15 koin, M1 presisi penuh menit-per-menit,
#      dataset PALING BESAR & PALING AKURAT sejauh ini): win rate PERSIS
#      SAMA (72.2%) dengan ladder v4, PnL lebih tinggi (144.35%→151.56%,
#      +7.2pp) — dan SL count SAMA SEKALI TIDAK BERUBAH (309), artinya
#      perbaikan ini murni menangkap lebih banyak upside dari trade yang
#      memang sudah menang, BUKAN mengambil risiko baru.
#
# Pelajaran dari v4→v5: tuning di sampel kecil (356 trade) bisa menyesatkan
# — begitu divalidasi ulang di sampel 3x lebih besar, hasilnya justru lebih
# baik pakai parameter yang lebih dekat ke versi SEBELUM v4. Sampel besar
# menang. Sudah dicoba juga menurunkan threshold R pertama (banyak trade
# SL cuma sempat MFE 0.28-0.36R sebelum reversal, di bawah 0.5R) — TERBUKTI
# menaikkan win rate signifikan (sampai 78-83%) TAPI PnL SELALU turun
# (127-146%) — tidak diambil krn bukan perbaikan bersih di kedua sisi,
# cuma trade-off WR-vs-PnL. Sudah dicoba juga grid search lebih luas di
# tahap ekor (3.2R/3.5R/4.5R/5.0R dgn macam2 lock) — hasil konvergen di
# kisaran 149-151.5%, F3b (di bawah) adalah titik terbaik yang ditemukan.
TRAIL_R_LADDER = [
    (0.5, 0.15),   # profit capai 0.5R → kunci 15% dari 0.5R
    (1.0, 0.35),   # 1.0R → kunci 35%
    (1.5, 0.50),   # 1.5R → kunci 50%
    (2.0, 0.65),   # 2.0R → kunci 65%
    (2.8, 0.80),   # 2.8R → kunci 80%
    (3.5, 0.85),   # 3.5R → kunci 85% (tahap tambahan v5 — tangkap sisa upside
                   #   trade yang sudah lari jauh, avg RR planned ~3.5-3.6R
                   #   jadi di titik ini biasanya sudah dekat TP)
]
# Trailing stop — KOMPONEN STRUKTUR (tetap dipakai, TIDAK berubah dari
# sebelumnya — divalidasi terpisah dan tetap jadi kandidat independen yg
# dibandingkan dgn ladder R di atas, SL final = paling protektif dari
# keduanya). Dibandingkan head-to-head di Datasheet.csv: fixed-pct
# SENDIRIAN WR 70.7% PnL 131.44%; structure SENDIRIAN WR 40-49% PnL
# 140-156% (lebih besar tapi jarang menang krn butuh lb*2+1 candle utk
# konfirmasi swing pertama); KOMBO WR 70.3% PnL 137.82% — lebih tahan-
# overfit krn separuh keputusan dari price action riil.
STRUCT_TRAIL_LB       = 2       # swing pivot lookback (kanan-kiri) di M15
STRUCT_TRAIL_BUF_PCT  = 0.0015  # buffer 0.15% di bawah/atas swing point
STRUCT_TRAIL_LOOKBACK = 60      # jumlah candle M15 ke belakang utk deteksi swing
WIB = timezone(timedelta(hours=7))   # untuk format jam entry di /trade
# ── Fibonacci Extension TP (gated H4 confluence) ──
# Dipakai HANYA saat level struktural biasa sudah habis diperiksa DAN
# konteks H4 (trend besar) + RSI H4 (momentum belum jenuh) mendukung.
# Bukan cabang "penyelamat" RR gagal — ini kandidat TP tambahan yang
# dievaluasi berdampingan dengan level struktural lain di _select_best_tp.
FIB_EXT_1           = 0.272  # ekstensi 1.272 — butuh H4 trend + RSI band saja
FIB_EXT_2           = 0.618  # ekstensi 1.618 — butuh confluence penuh (+ CHoCH M15 searah)
H4_RSI_BUY_MIN      = 45     # RSI H4 BUY: momentum sudah established (bukan baru mulai)
H4_RSI_BUY_MAX      = 68     # tapi belum overbought / jenuh
H4_RSI_SELL_MIN     = 32     # RSI H4 SELL: kebalikan dari BUY
H4_RSI_SELL_MAX     = 55
# ─────────────────────────────────────────────

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN tidak ditemukan di environment. Cek file .env")

import requests, pandas as pd, numpy as np, urllib3, json
from flask import Flask

try:
    import websocket   # pip: websocket-client
    _WS_LIB_OK = True
except ImportError:
    _WS_LIB_OK = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


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
BAN_DURATION_TRADE_CLOSED = 500   # ban khusus setelah trade BENAR-BENAR closed (TP/SL/Trail)

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
    for i in range(3):
        try:
            r = requests.get(f"{FAPI}{path}", params=params,
                             timeout=10, verify=False)
            if r.status_code in (418, 429):
                # Kena rate-limit/ban IP dari Binance — JANGAN retry lagi
                # ke Binance (mengulang request saat sedang kena ban malah
                # berisiko memperpanjang durasi ban). Langsung lempar ke
                # caller supaya pindah ke tier fallback (Bybit → WS).
                raise ConnectionError(
                    f"Binance kena limit/ban (HTTP {r.status_code})")
            d = r.json()
            if isinstance(d, dict) and "code" in d:
                raise ValueError(f"Binance {d['code']}: {d.get('msg')}")
            return d
        except ConnectionError as e:
            log.warning(f"[binance] {e} — stop retry Binance, pindah fallback")
            raise
        except Exception as e:
            log.warning(f"[binance] {i+1}/3: {e}")
            time.sleep(2)
    raise ConnectionError(f"Binance gagal: {path}")

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
    """Tier1 Binance REST → Tier2 Bybit REST → Tier3 buffer WS (fallback
    TERAKHIR, hanya dipakai kalau REST Binance & Bybit gagal/error/kena
    ban). ensure_symbol_interval() tetap dipanggil di awal supaya WS terus
    subscribe & backfill di background — bukan supaya jadi sumber utama,
    tapi supaya buffer-nya SIAP dipakai sewaktu-waktu REST bermasalah."""
    ws_feed.ensure_symbol_interval(symbol, interval)
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
    if ws_feed.is_fresh():
        df = ws_feed.get_klines(symbol, interval, limit)
        if df is not None and not df.empty:
            log.warning(f"[klines/ws fallback] {symbol} {interval} — REST Binance & Bybit gagal")
            return df
    return pd.DataFrame()

def get_top_coins():
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
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d=s.diff()
    g=d.clip(lower=0).rolling(n).mean()
    l=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def macd(s):
    line=ema(s,12)-ema(s,26); sig=ema(line,9)
    return line, sig, line-sig

def atr_fn(df, n=14):
    tr=pd.concat([
        df["high"]-df["low"],
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build_df(df):
    if len(df)<60: return None
    df=df.copy()
    df["ema9"]   = ema(df["close"],9)
    df["ema21"]  = ema(df["close"],21)
    df["ema50"]  = ema(df["close"],50)
    df["ema200"] = ema(df["close"],200) if len(df)>=200 else ema(df["close"],50)
    df["rsi"]    = rsi(df["close"])
    df["ml"],df["ms"],df["mh"] = macd(df["close"])
    df["atr"]    = atr_fn(df)
    df["vol_sma"]= df["volume"].rolling(20).mean()
    bm=df["close"].rolling(20).mean(); bs=df["close"].rolling(20).std()
    df["bb_up"]=bm+2*bs; df["bb_lo"]=bm-2*bs; df["bb_mid"]=bm
    return df.dropna()


# ═════════════════════════════════════════════
# SMC / PRICE ACTION TOOLS
# ═════════════════════════════════════════════
def swing_pts(df, lb=5):
    sh,sl=[],[]
    for i in range(lb, len(df)-lb):
        if df["high"].iloc[i]==df["high"].iloc[i-lb:i+lb+1].max(): sh.append(i)
        if df["low"].iloc[i]==df["low"].iloc[i-lb:i+lb+1].min():   sl.append(i)
    return sh, sl

def mkt_struct(df, sh, sl):
    if len(sh)<2 or len(sl)<2: return "ranging"
    hh=df["high"].iloc[sh[-1]]>df["high"].iloc[sh[-2]]
    hl=df["low"].iloc[sl[-1]]>df["low"].iloc[sl[-2]]
    lh=df["high"].iloc[sh[-1]]<df["high"].iloc[sh[-2]]
    ll=df["low"].iloc[sl[-1]]<df["low"].iloc[sl[-2]]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

def detect_bos(df, sh, sl):
    """
    BOS (Break of Structure) — konfirmasi kelanjutan trend.
    Sesuai materi: BOS valid CUKUP dengan shadow/wick candle menembus
    swing sebelumnya (tidak wajib body close, beda dengan CHoCH yang
    lebih ketat — lihat detect_choch()).
    """
    res={"bb":False,"bs":False,"cb":False,"cs":False}
    hi=df["high"].iloc[-1]; lo=df["low"].iloc[-1]
    if len(sh)>=2:
        ph=df["high"].iloc[sh[-2]]; lh=df["high"].iloc[sh[-1]]
        if hi>ph: res["bb" if lh>ph else "cb"]=True
    if len(sl)>=2:
        pl=df["low"].iloc[sl[-2]]; ll=df["low"].iloc[sl[-1]]
        if lo<pl: res["bs" if ll<pl else "cs"]=True
    return res

def find_snr_levels(df, lb=80):
    """
    Cari level Support & Resistance dari swing points.
    Level yang paling banyak disentuh = level terkuat.
    """
    sh, sl = swing_pts(df, lb=5)
    levels = []
    for i in sh:
        levels.append(("R", df["high"].iloc[i]))
    for i in sl:
        levels.append(("S", df["low"].iloc[i]))
    return levels

def find_zones(df, direction, lb=40, strict=False):
    """
    Deteksi ZONA TERPADU (Order Block = Supply/Demand Zone) — satu model
    sesuai materi: OB pada dasarnya adalah versi "dasar/minimal" dari
    Supply & Demand zone, bukan konsep terpisah. Fungsi ini menggantikan
    find_supply_demand() dan find_ob() versi lama yang terpisah.

    direction: "bull"/"demand" → base candle bearish diikuti rally (cari
               zona untuk BUY)
               "bear"/"supply" → base candle bullish diikuti drop (cari
               zona untuk SELL)
    strict   : True  → wajib 2 candle konfirmasi lanjutan searah setelah
                        impulse (perilaku find_ob lama, base candle lebih
                        "murni"/sempit — dipakai untuk entry precision)
               False → cukup 1 candle impulse (perilaku find_supply_demand
                        lama, zona bisa sedikit lebih lebar — dipakai
                        untuk SL/TP pool yang butuh lebih banyak kandidat)

    Setiap zona disertai VALIDASI 3-KRITERIA dari materi (dianggap valid
    kalau minimal salah satu terpenuhi, quality = jumlah yang terpenuhi):
    1. has_fvg   — ada Fair Value Gap yang menyertai impulse move
    2. has_bos   — impulse move menghasilkan break of structure
    3. is_fresh  — zona belum pernah disentuh ulang sejak terbentuk
    Plus:
    - pattern         : RBR/DBR (demand) atau DBD/RBD (supply)
    - strong_move_away: candle impulse body besar (bukan sekadar koreksi
      kecil — penanda smart money benar2 eksekusi order besar di sana)
    - fib_zone/fib_ratio/fib_aligned: posisi zona relatif ke range swing
      lb candle terakhir. Zona demand/bull idealnya di DISKON, zona
      supply/bear idealnya di PREMIUM (fib_aligned=True kalau selaras).
    """
    is_demand = direction in ("bull", "demand")
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean()
    swing_hi = sub["high"].max()
    swing_lo = sub["low"].min()
    zones = []
    sh_all, sl_all = swing_pts(df, lb=5)

    end_range = len(sub) - 3 if strict else len(sub) - 2
    for i in range(1, end_range):
        c   = sub.iloc[i]
        nx  = sub.iloc[i + 1]
        nx2 = sub.iloc[i + 2] if i + 2 < len(sub) else None
        impulse_body = abs(nx["close"] - nx["open"])
        strong_move_away = impulse_body >= avg_body * 1.3
        min_impulse = avg_body * (1.5 if strict else 1.3)
        if impulse_body < min_impulse:
            continue

        if is_demand:
            is_match = c["close"] < c["open"] and nx["close"] > nx["open"]
            if strict and is_match:
                is_match = nx2 is not None and nx2["close"] > nx2["open"]
        else:
            is_match = c["close"] > c["open"] and nx["close"] < nx["open"]
            if strict and is_match:
                is_match = nx2 is not None and nx2["close"] < nx2["open"]
        if not is_match:
            continue

        top = max(c["open"], c["close"])
        bot = min(c["open"], c["close"])
        df_idx = base_offset + i

        # Kriteria 1: FVG menyertai impulse (celah antara c dan nx2)
        has_fvg = False
        if nx2 is not None:
            if is_demand and nx2["low"] > c["high"]:
                has_fvg = True
            if (not is_demand) and nx2["high"] < c["low"]:
                has_fvg = True

        # Kriteria 2: impulse ini menghasilkan BOS (harga break swing sebelumnya)
        has_bos = False
        try:
            if is_demand and len(sh_all) >= 1:
                prior_highs = [df["high"].iloc[k] for k in sh_all if k < df_idx]
                if prior_highs and nx["high"] > max(prior_highs[-1:] or [float("-inf")]):
                    has_bos = True
            if (not is_demand) and len(sl_all) >= 1:
                prior_lows = [df["low"].iloc[k] for k in sl_all if k < df_idx]
                if prior_lows and nx["low"] < min(prior_lows[-1:] or [float("inf")]):
                    has_bos = True
        except Exception:
            has_bos = False

        # Kriteria 3: fresh — belum pernah disentuh ulang sejak terbentuk
        fresh = is_zone_fresh(df, top, bot, df_idx)

        pattern = classify_sd_pattern(df, df_idx, "demand" if is_demand else "supply")

        fib = get_fib_zone((top + bot) / 2, swing_lo, swing_hi)
        fib_aligned = fib["zone"] in (("discount", "equilibrium") if is_demand
                                       else ("premium", "equilibrium"))

        zones.append({
            "top": top, "bot": bot,
            "mid": (top + bot) / 2,
            "high": c["high"], "low": c["low"],
            "idx": df_idx,
            "has_fvg": bool(has_fvg),
            "has_bos": bool(has_bos),
            "is_fresh": bool(fresh),
            "strong_move_away": bool(strong_move_away),
            "pattern": pattern,
            "fib_zone": fib["zone"],
            "fib_ratio": fib["ratio"],
            "fib_aligned": bool(fib_aligned),
            # quality: berapa dari 3 kriteria utama terpenuhi (fvg, bos, fresh)
            "quality": int(has_fvg) + int(has_bos) + int(fresh),
        })
    return zones[-3:] if zones else []


def find_supply_demand(df, direction, lb=40):
    """Kompatibilitas: alias tipis ke find_zones (mode non-strict/S&D)."""
    return find_zones(df, "demand" if direction == "demand" else "supply", lb=lb, strict=False)


def find_ob(df, direction, lb=40):
    """Kompatibilitas: alias tipis ke find_zones (mode strict/OB murni)."""
    return find_zones(df, direction, lb=lb, strict=True)



def find_fvg(df, direction, lb=40):
    """
    Fair Value Gap (FVG) — celah 3-candle yang menandakan pergerakan
    impulsif tak seimbang antara buyer/seller.

    Setiap FVG kini disertai:
    - is_fresh   : belum pernah disentuh ulang (bahkan oleh shadow) sejak terbentuk
    - candle3    : klasifikasi "breakaway" (ideal, searah & impulsif) vs
                   "rejection" (hindari, candle ke-3 melawan arah gap)
    - fib_zone   : apakah gap ini berada di area diskon/premium relatif
                   terhadap range swing lb candle terakhir (dipakai utk
                   preferensi entry FVG di area diskon utk BUY / premium
                   utk SELL, sesuai materi)
    """
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    out = []
    swing_hi = sub["high"].max()
    swing_lo = sub["low"].min()

    for i in range(len(sub) - 2):
        c0, c1, c2 = sub.iloc[i], sub.iloc[i + 1], sub.iloc[i + 2]
        df_idx_c0 = base_offset + i
        df_idx_c2 = base_offset + i + 2

        gap = None
        if direction == "bull" and c2["low"] > c0["high"]:
            gap = {"top": c2["low"], "bot": c0["high"]}
        if direction == "bear" and c2["high"] < c0["low"]:
            gap = {"top": c0["low"], "bot": c2["high"]}
        if gap is None:
            continue

        gap["mid"] = (gap["top"] + gap["bot"]) / 2
        gap["idx"] = df_idx_c2
        gap["is_fresh"] = is_zone_fresh(df, gap["top"], gap["bot"], df_idx_c0, end_idx=len(df)-1)
        gap["candle3"] = classify_fvg_candle3(df, df_idx_c2, direction)
        gap["fib_zone"] = get_fib_zone(gap["mid"], swing_lo, swing_hi)["zone"]
        out.append(gap)

    return out[-3:] if out else []

def find_equal_highs_lows(df, kind="high", lb=60, tol=0.0025):
    """
    Equal Highs/Lows = zona likuiditas (banyak stop loss retail di sana).
    Institusi sering sweeping level ini sebelum berbalik.
    """
    sub=df.iloc[-lb:]
    vals=sub["high"] if kind=="high" else sub["low"]
    clusters=[]
    visited=set()
    for i in range(len(vals)):
        if i in visited: continue
        group=[vals.iloc[i]]
        for j in range(i+1, len(vals)):
            if abs(vals.iloc[i]-vals.iloc[j])/max(vals.iloc[i],0.0001)<tol:
                group.append(vals.iloc[j])
                visited.add(j)
        if len(group)>=2:
            clusters.append(sum(group)/len(group))
    return sorted(clusters)

def nearest_snr(df, price, direction, margin=0.015):
    """
    Cari level S/R terdekat yang relevan untuk TP/SL.
    direction='above' → cari resistance di atas harga
    direction='below' → cari support di bawah harga
    """
    sh, sl = swing_pts(df, lb=4)
    if direction=="above":
        candidates = [df["high"].iloc[i] for i in sh
                      if df["high"].iloc[i] > price*(1+margin*0.3)]
        candidates += find_equal_highs_lows(df,"high")
        candidates = [c for c in candidates if c > price*(1+margin*0.3)]
        return min(candidates) if candidates else None
    else:
        candidates = [df["low"].iloc[i] for i in sl
                      if df["low"].iloc[i] < price*(1-margin*0.3)]
        candidates += find_equal_highs_lows(df,"low")
        candidates = [c for c in candidates if c < price*(1-margin*0.3)]
        return max(candidates) if candidates else None


def detect_choch(df, sh, sl):
    """
    CHoCH (Change of Character) — konfirmasi perubahan arah NYATA.
    Bearish CHoCH: harga break di bawah HL terakhir setelah LH terbentuk.
    Bullish CHoCH: harga break di atas LH terakhir setelah HL terbentuk.
    Lebih ketat dari BOS biasa — perlu dua swing point terkonfirmasi
    DAN wajib BODY CLOSE candle menembus level (bukan sekadar shadow/wick),
    karena CHoCH menandakan pembalikan karakter pasar yang butuh bukti
    lebih kuat dibanding BOS yang hanya kelanjutan trend. Fungsi ini
    sudah pakai df["close"] (bukan high/low) sehingga syarat body-close
    otomatis terpenuhi.
    """
    result = {"bearish_choch": False, "bullish_choch": False}
    close = df["close"].iloc[-1]

    # Bearish CHoCH: ada LH (lower high) DAN harga sekarang break bawah swing low sebelumnya
    if len(sh) >= 2 and len(sl) >= 2:
        prev_high = df["high"].iloc[sh[-2]]
        last_high = df["high"].iloc[sh[-1]]
        prev_low  = df["low"].iloc[sl[-2]]
        last_low  = df["low"].iloc[sl[-1]]

        lh_formed = last_high < prev_high          # LH terbentuk
        if lh_formed and close < prev_low:         # break bawah HL
            result["bearish_choch"] = True

        hh_formed = last_high > prev_high          # HH terbentuk
        if hh_formed and close > prev_low and last_low > prev_low:  # break atas + HL
            result["bullish_choch"] = True

    return result


def detect_failed_retest(df, sh, sl, atr):
    """
    Failed Retest — harga naik ke resistance/level struktural lalu ditolak keras.
    Ini trigger entry SELL yang paling valid di SMC.
    Syarat:
    - Ada resistance level yang jelas (swing high sebelumnya)
    - Harga candle sebelumnya menyentuh atau mendekati resistance (dalam 0.5 ATR)
    - Candle sekarang close jauh di bawah resistance (rejection)
    - Candle sekarang bearish (close < open)
    """
    result = {"failed_retest_sell": False, "failed_retest_buy": False,
              "resistance": None, "support": None}
    if len(df) < 3: return result

    L   = df.iloc[-1]   # candle sekarang
    P   = df.iloc[-2]   # candle sebelumnya

    # Failed retest SELL: candle sebelumnya menyentuh resistance, sekarang rejected
    if len(sh) >= 2:
        resistance = df["high"].iloc[sh[-2]]   # swing high terakhir = resistance
        touched    = P["high"] >= resistance - atr * 0.5   # candle sebelum menyentuh
        rejected   = L["close"] < resistance - atr * 0.3  # sekarang jauh di bawah
        bearish_c  = L["close"] < L["open"]               # candle bearish
        if touched and rejected and bearish_c:
            result["failed_retest_sell"] = True
            result["resistance"] = resistance

    # Failed retest BUY: candle sebelumnya menyentuh support, sekarang bounced
    if len(sl) >= 2:
        support  = df["low"].iloc[sl[-2]]      # swing low terakhir = support
        touched  = P["low"] <= support + atr * 0.5
        bounced  = L["close"] > support + atr * 0.3
        bullish_c = L["close"] > L["open"]
        if touched and bounced and bullish_c:
            result["failed_retest_buy"] = True
            result["support"] = support

    return result


# ═════════════════════════════════════════════
# SMC LANJUTAN — Ilmu dari materi edukasi:
# fresh/mitigated zone, fib diskon/premium, breakaway
# vs rejection FVG, validitas pullback, price action
# confirmation (pin bar/fakey), pola RBR/DBR/DBD/RBD,
# inducement & liquidity sweep/run.
# ═════════════════════════════════════════════

def is_zone_fresh(df, top, bot, formed_idx, end_idx=None):
    """
    Cek apakah sebuah zona (OB/S&D/FVG) masih FRESH — belum pernah
    disentuh oleh harga sejak zona itu terbentuk.

    "Disentuh" didefinisikan longgar (bahkan wick/shadow saja dianggap
    sudah memitigasi zona — sesuai penjelasan di materi FVG: "meskipun
    hanya tersentuh sedikit dengan shadow, kita tetap menganggapnya
    sudah tersentuh").

    formed_idx: index candle tempat zona ini terbentuk (posisi dalam df).
    end_idx   : index terakhir yang mau diperiksa (default: candle
                terakhir df). start diambil 2 candle setelah formed_idx
                supaya candle pembentuk zona itu sendiri tidak dihitung.

    Return: True jika fresh (belum tersentuh), False jika sudah termitigasi.
    """
    if formed_idx is None or top is None or bot is None:
        return True
    n = len(df)
    end_idx = end_idx if end_idx is not None else n - 1
    start = formed_idx + 2
    if start >= end_idx:
        return True
    sub = df.iloc[start:end_idx]
    if sub.empty:
        return True
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)


def get_fib_zone(price, swing_low, swing_high):
    """
    Tentukan posisi harga dalam rentang swing (retracement ratio) serta
    apakah harga berada di area DISKON, PREMIUM, atau EQUILIBRIUM.

    ratio dihitung sebagai posisi price relatif terhadap [swing_low, swing_high]:
      ratio kecil (<=0.45) → dekat swing_low  → "discount"
      ratio besar (>=0.55) → dekat swing_high → "premium"
      di antaranya         → "equilibrium"

    Return dict: {"ratio": float, "zone": str}
    """
    rng = swing_high - swing_low
    if rng <= 0:
        return {"ratio": 0.5, "zone": "equilibrium"}
    ratio = (price - swing_low) / rng
    if ratio <= 0.45:
        zone = "discount"
    elif ratio >= 0.55:
        zone = "premium"
    else:
        zone = "equilibrium"
    return {"ratio": round(ratio, 4), "zone": zone}


def adaptive_fib_target(df, sh, sl, direction):
    """
    Tentukan target retracement Fibonacci secara ADAPTIF berdasarkan
    kekuatan trend & kedalaman pullback (bukan angka fix 50%):

    - Trend SANGAT kuat (impuls dominan, nyaris tanpa pullback)
      → fokus area retracement 0.236 - 0.382 (paling dangkal)
    - Trend kuat (impuls dominan, pullback dangkal & lemah)
      → fokus area retracement 0.382 - 0.5 (dangkal)
    - Trend lemah (pullback agresif & dalam)
      → fokus area retracement 0.618 - 0.786 (dalam, termasuk OTE)

    Kekuatan trend diestimasi dari rasio panjang leg pullback vs leg
    impuls terakhir (di TF yang sama, m15/h1 tergantung caller).

    CATATAN (fix presisi-entry): tier "SANGAT kuat" ditambahkan setelah
    analisa data menemukan >600 sinyal (dari 2 backtest independen)
    gagal terisi krn TP sudah kena duluan sebelum harga sempat pullback
    ke zona entry — median jarak ke entry 2.2-2.5× lebih jauh drpd jarak
    ke TP itu sendiri. Root cause: trend yg SANGAT kuat (pullback_ratio
    mendekati 0, artinya harga nyaris tidak pullback sama sekali) tetap
    diminta retrace ke 0.382-0.5 — utk momentum se-ekstrem itu, itu
    sendiri sudah terlalu dalam & sering tidak pernah kejadian sebelum
    harga lanjut ke TP. Tier baru ini TIDAK melonggarkan syarat kualitas
    apa pun (freshness/FVG/BOS tetap sama) — cuma target retracement yg
    lebih realistis utk kondisi momentum paling ekstrem.

    Return: (fib_lo, fib_hi) sebagai rasio retracement (0..1).
    """
    default = (0.5, 0.618)   # fallback netral kalau data belum cukup
    if len(sh) < 2 or len(sl) < 2:
        return default
    try:
        if direction == "bull":
            impulse_len   = df["high"].iloc[sh[-1]] - df["low"].iloc[sl[-2]]
            pullback_len  = df["high"].iloc[sh[-1]] - df["close"].iloc[-1]
        else:
            impulse_len   = df["high"].iloc[sh[-2]] - df["low"].iloc[sl[-1]]
            pullback_len  = df["close"].iloc[-1] - df["low"].iloc[sl[-1]]
        if impulse_len <= 0:
            return default
        pullback_ratio = abs(pullback_len) / impulse_len
    except Exception:
        return default

    if pullback_ratio <= 0.12:
        return (0.236, 0.382)   # trend SANGAT kuat, pullback minimal
    elif pullback_ratio <= 0.30:
        return (0.382, 0.5)     # trend kuat, pullback dangkal
    elif pullback_ratio >= 0.55:
        return (0.618, 0.786)   # trend lemah, pullback dalam (OTE)
    else:
        return (0.5, 0.618)


def classify_fvg_candle3(df, fvg_idx_c2, direction):
    """
    Klasifikasi FVG berdasarkan candle ke-3 (candle "c2" pembentuk gap):
    - Breakaway Gap : candle ke-3 SEARAH gap (impulsif, melanjutkan) → IDEAL untuk entry
    - Rejection Gap : candle ke-3 BERLAWANAN arah gap → HINDARI, sinyal lemah

    direction: "bull" (bullish FVG) atau "bear" (bearish FVG)
    Return: "breakaway" atau "rejection"
    """
    if fvg_idx_c2 is None or fvg_idx_c2 >= len(df):
        return "unknown"
    c2 = df.iloc[fvg_idx_c2]
    is_bull_candle = c2["close"] > c2["open"]
    if direction == "bull":
        return "breakaway" if is_bull_candle else "rejection"
    else:
        return "rejection" if is_bull_candle else "breakaway"


def is_valid_pullback(df, direction, lookback=8):
    """
    Validasi pullback sesuai definisi price action yang ketat:
    pullback valid HANYA jika candle koreksi benar-benar men-BREAK
    high/low dari candle sebelumnya (bukan sekadar candle berganti warna).

    Bullish trend: pullback valid jika ada candle bearish yang close-nya
    menembus LOW dari candle bullish terakhir sebelum koreksi dimulai.
    Bearish trend: sebaliknya, candle bullish menembus HIGH candle
    bearish terakhir.

    Return: bool
    """
    if len(df) < lookback + 2:
        return False
    sub = df.iloc[-lookback:]

    if direction == "bull":
        last_bull_low = None
        found_i = None
        for i in range(len(sub) - 1, -1, -1):
            c = sub.iloc[i]
            if c["close"] > c["open"]:
                last_bull_low = c["low"]
                found_i = i
                break
        if last_bull_low is None:
            return False
        after = sub.iloc[found_i+1:]
        return bool((after["close"] < last_bull_low).any())
    else:
        last_bear_high = None
        found_i = None
        for i in range(len(sub) - 1, -1, -1):
            c = sub.iloc[i]
            if c["close"] < c["open"]:
                last_bear_high = c["high"]
                found_i = i
                break
        if last_bear_high is None:
            return False
        after = sub.iloc[found_i+1:]
        return bool((after["close"] > last_bear_high).any())


def classify_pullback_type(df, direction, atr, lookback=6):
    """
    Klasifikasi tipe pullback: aggressive / corrective / sweeping.

    - Aggressive : koreksi cepat & besar (candle body rerata > 1.2x ATR),
      momentum kuat melawan trend → probabilitas reaksi di zona RENDAH,
      sebaiknya tidak entry langsung.
    - Sweeping   : ada equal high/low (double top/bottom) tepat sebelum
      area, menandakan liquidity pool yang disapu dulu → probabilitas
      TINGGI setelah sweep + shift struktur.
    - Corrective : koreksi bertahap, beberapa struktur kecil → probabilitas
      entry paling ideal, terutama dengan konfirmasi CHoCH TF rendah.

    Return: "aggressive" | "corrective" | "sweeping"
    """
    if len(df) < lookback + 1:
        return "corrective"
    sub = df.iloc[-lookback:]
    bodies = (sub["close"] - sub["open"]).abs()
    avg_body = bodies.mean()

    highs = sub["high"].values
    lows  = sub["low"].values
    tol = atr * 0.15
    has_equal_high = False
    has_equal_low  = False
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            if abs(highs[i] - highs[j]) < tol:
                has_equal_high = True
            if abs(lows[i] - lows[j]) < tol:
                has_equal_low = True

    if direction == "bull" and has_equal_low:
        return "sweeping"
    if direction == "bear" and has_equal_high:
        return "sweeping"

    if avg_body > atr * 1.2:
        return "aggressive"

    return "corrective"


def detect_pinbar(candle, min_wick_ratio=1.5):
    """
    Deteksi pola Pin Bar (Deception Candle): body kecil di salah satu
    ujung, shadow panjang di sisi berlawanan — menandakan rejection kuat.

    Return: {"is_pinbar": bool, "bullish_pinbar": bool, "bearish_pinbar": bool}
    """
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body = abs(c - o)
    rng  = h - l
    if rng <= 0:
        return {"is_pinbar": False, "bullish_pinbar": False, "bearish_pinbar": False}
    low_wick = min(o, c) - l
    up_wick  = h - max(o, c)

    bullish_pinbar = low_wick > body * min_wick_ratio and low_wick > up_wick * 1.5
    bearish_pinbar = up_wick > body * min_wick_ratio and up_wick > low_wick * 1.5
    return {
        "is_pinbar": bool(bullish_pinbar or bearish_pinbar),
        "bullish_pinbar": bool(bullish_pinbar),
        "bearish_pinbar": bool(bearish_pinbar),
    }


def detect_fakey(df):
    """
    Deteksi pola Fakey (false breakout dari inside bar):
    1. Ada inside bar (candle tertutup penuh dalam range candle sebelumnya)
    2. Harga breakout ke salah satu sisi (menembus high/low mother bar)
    3. Harga berbalik dan close kembali DI DALAM range mother bar

    Return: {"is_fakey": bool, "bullish_fakey": bool, "bearish_fakey": bool}
    bullish_fakey = false breakout ke bawah lalu balik naik (sinyal BUY)
    bearish_fakey = false breakout ke atas lalu balik turun (sinyal SELL)
    """
    result = {"is_fakey": False, "bullish_fakey": False, "bearish_fakey": False}
    if len(df) < 3:
        return result

    mother = df.iloc[-3]
    inside = df.iloc[-2]
    last   = df.iloc[-1]

    is_inside = inside["high"] <= mother["high"] and inside["low"] >= mother["low"]
    if not is_inside:
        return result

    broke_up   = last["high"] > mother["high"]
    broke_down = last["low"]  < mother["low"]
    closed_inside = mother["low"] <= last["close"] <= mother["high"]

    if broke_down and closed_inside and last["close"] > last["open"]:
        result["is_fakey"] = True
        result["bullish_fakey"] = True
    elif broke_up and closed_inside and last["close"] < last["open"]:
        result["is_fakey"] = True
        result["bearish_fakey"] = True

    return result


def classify_sd_pattern(df, zone_idx, direction, lb=6):
    """
    Klasifikasi pola pembentukan supply/demand berdasarkan rally/drop/base:
    - Demand: RBR (Rally-Base-Rally) atau DBR (Drop-Base-Rally)
    - Supply: DBD (Drop-Base-Drop) atau RBD (Rally-Base-Drop)

    zone_idx: index candle "base" (candle dasar pembentuk OB) dalam df.
    Return label string atau "unknown" kalau tidak cukup data.
    """
    if zone_idx is None or zone_idx < lb or zone_idx + lb >= len(df):
        return "unknown"

    before = df.iloc[max(0, zone_idx - lb):zone_idx]
    after  = df.iloc[zone_idx + 1: zone_idx + 1 + lb]
    if before.empty or after.empty:
        return "unknown"

    move_before = before["close"].iloc[-1] - before["close"].iloc[0]
    move_after  = after["close"].iloc[-1] - after["close"].iloc[0]

    before_up = move_before > 0
    after_up  = move_after > 0

    if direction == "demand":
        if before_up and after_up:       return "RBR"
        if (not before_up) and after_up: return "DBR"
        return "unknown"
    else:
        if (not before_up) and (not after_up): return "DBD"
        if before_up and (not after_up):        return "RBD"
        return "unknown"


def detect_liquidity_run_or_sweep(df, sh, sl, direction):
    """
    Bedakan Liquidity RUN (breakout bersih, close di luar swing) vs
    Liquidity SWEEP/SWIFT (wick menembus tapi GAGAL close di luar swing —
    liquidity grab, arah sebenarnya kemungkinan BERLAWANAN).

    direction: "bull" → cek terhadap swing high terdekat
               "bear" → cek terhadap swing low terdekat

    Return: {"type": "run"/"sweep"/"none", "level": float atau None}
    """
    result = {"type": "none", "level": None}
    if direction == "bull" and len(sh) >= 1:
        level = df["high"].iloc[sh[-1]]
        last  = df.iloc[-1]
        if last["high"] > level and last["close"] > level:
            result = {"type": "run", "level": level}
        elif last["high"] > level and last["close"] <= level:
            result = {"type": "sweep", "level": level}
    elif direction == "bear" and len(sl) >= 1:
        level = df["low"].iloc[sl[-1]]
        last  = df.iloc[-1]
        if last["low"] < level and last["close"] < level:
            result = {"type": "run", "level": level}
        elif last["low"] < level and last["close"] >= level:
            result = {"type": "sweep", "level": level}
    return result


def detect_inducement_move(df, direction, atr, lookback=5):
    """
    Deteksi kemungkinan inducement — gerakan kecil BERLAWANAN arah trend
    yang muncul TEPAT SEBELUM harga menyentuh level penting (OB/FVG/EQH/EQL).
    Ciri: gerakan kecil (< 0.6 ATR), searah pullback minor, terjadi di
    2-3 candle terakhir sebelum candle sekarang.

    Ini dipakai sebagai FLAG (bukan hard block) — kalau inducement barusan
    terjadi, kita minta konfirmasi CHoCH tambahan sebelum entry, alih-alih
    entry di breakout/gerakan pertama begitu saja.

    Return: bool (True = terindikasi inducement baru saja terjadi)
    """
    if len(df) < lookback + 1:
        return False
    sub = df.iloc[-lookback:-1]   # tidak termasuk candle sekarang
    if sub.empty:
        return False
    small_moves = ((sub["close"] - sub["open"]).abs() < atr * 0.6)
    if direction == "bull":
        counter = sub["close"] < sub["open"]
    else:
        counter = sub["close"] > sub["open"]
    return bool((small_moves & counter).tail(3).any())


# ═════════════════════════════════════════════
# TAHAP 1: SCORING NORMAL — cari sinyal terkuat
# ═════════════════════════════════════════════
def score_direction(df_h1, df_m15, df_d1=None):
    """
    Analisis HIERARKIS (bukan lagi additive scoring flat dari banyak
    indikator lepas): sesuai filosofi price-action/SMC di materi —
    tentukan BIAS dari struktur besar dulu, baru cari KONFIRMASI ENTRY
    dari price-action M15. Bias dan konfirmasi yang berlawanan saling
    melemahkan drastis (gate), bukan sekadar dijumlah rata.

    LAYER 1 — BIAS (konteks arah, dari struktur & momentum besar):
      • Market Structure H1 (HH/HL vs LH/LL) — bobot terbesar
      • D1 bias (EMA + struktur harian) — konfirmasi/veto di layer akhir
      • EMA H1 trend alignment
      • RSI M15 (dipertahankan sesuai preferensi — momentum filter,
        bukan sinyal SMC, tapi tetap berguna sebagai extra confluence)

    LAYER 2 — SETUP/KONFIRMASI (price-action & SMC murni dari 10 materi):
      • BOS (shadow) & CHoCH (wajib body-close) M15+H1
      • Failed Retest M15+H1
      • Validitas & tipe pullback (corrective/sweeping/aggressive)
      • Pin bar rejection, pola Fakey
      • Liquidity Run vs Sweep/Swift
      • OTE 0.62-0.79 + FVG/CHoCH pendukung
      • MACD/BB/Volume M15 — momentum confluence tambahan (ringan)

    LAYER 3 — GATE: kalau LAYER 2 (konfirmasi entry) berlawanan arah
    dengan LAYER 1 (bias struktural), konfirmasi itu dilemahkan drastis
    alih-alih dijumlah rata — mencegah sinyal yang sebenarnya melawan
    struktur besar tapi lolos hanya karena numpuk banyak micro-signal M15.

    df_d1 (BARU): klines D1 ASLI (bukan hasil resample df_h1). df_h1 di
    sini cuma window pendek (get_klines limit=250 ≈ 10 hari), resample
    ke "1D" dari situ cuma menghasilkan ~10 bar harian — DI BAWAH syarat
    minimum build_df (60 bar), jadi d1_bias SELALU "neutral" dan fitur
    ini tidak pernah benar-benar berkontribusi (bug lama). Kalau df_d1
    disediakan (fetch terpisah, histori panjang), d1_bias dihitung dari
    situ — kalau tidak, fallback ke cara lama (tetap sering neutral).

    Return: dict dengan symbol, direction asli, confidence, price
    """
    h1=build_df(df_h1); m15=build_df(df_m15)
    if h1 is None or m15 is None: return None

    L1=h1.iloc[-1]; P1=h1.iloc[-2]
    L15=m15.iloc[-1]; P15=m15.iloc[-2]
    rv=L15["rsi"]
    atr_val=max(L15["atr"], L15["close"]*0.003)

    sh1,sl1   = swing_pts(h1,5)
    sh15,sl15 = swing_pts(m15,5)
    struct_h1 = mkt_struct(h1,sh1,sl1)
    choch_h1  = detect_choch(h1, sh1, sl1)   # dihitung di sini krn ikut Layer 1

    # ── D1 bias — dihitung di awal supaya ikut Layer 1 scoring (bukan
    # cuma hard-block di akhir seperti sebelumnya). Syarat dilonggarkan
    # jadi OR (EMA alignment ATAU struct D1, tidak wajib bersamaan) —
    # sebelumnya syarat AND terlalu ketat sehingga d1_bias hampir selalu
    # "neutral" dan tidak pernah benar-benar berkontribusi ke arah,
    # sehingga H1 jadi satu-satunya penentu bias tanpa konteks D1 sama
    # sekali (H1 bisa saja cuma pullback minor dari downtrend D1 besar).
    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            df_d1_built = build_df(df_d1)
        else:
            # fallback lama (hampir selalu gagal krn window terlalu pendek)
            df_d1_built = build_df(df_h1.resample("1D").agg({
                "open":"first","high":"max","low":"min",
                "close":"last","volume":"sum"
            }).dropna())
        if df_d1_built is not None and len(df_d1_built) >= 10:
            LD = df_d1_built.iloc[-1]
            sh_d, sl_d = swing_pts(df_d1_built, lb=3)
            struct_d1  = mkt_struct(df_d1_built, sh_d, sl_d)
            ema_bear_d1 = LD["ema9"] < LD["ema21"] < LD["ema50"]
            ema_bull_d1 = LD["ema9"] > LD["ema21"] > LD["ema50"]
            if struct_d1 == "bearish" or ema_bear_d1:
                d1_bias = "bearish"
            elif struct_d1 == "bullish" or ema_bull_d1:
                d1_bias = "bullish"
            # kalau struct_d1 dan ema saling kontradiksi (mis. struct
            # bullish tapi ema bearish), biarkan struct_d1 menang karena
            # itu representasi price action nyata, EMA cuma turunan lag.
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════
    # LAYER 1 — BIAS: arah besar dari struktur, bukan micro-indicator.
    # CHoCH H1 dimasukkan di layer ini (bukan Layer 2/setup) karena
    # secara konsep CHoCH = perubahan KARAKTER/BIAS pasar itu sendiri
    # (lihat materi video #6), bukan sekadar trigger entry M15 seperti
    # pin bar/fakey. struct_h1 dari swing HH/HL cenderung lagging —
    # CHoCH H1 sering jadi sinyal PALING AWAL bahwa bias lama sudah
    # tidak berlaku, jadi wajib ikut menentukan bias_dir, bukan cuma
    # jadi setup yang dipotong buta kalau "melawan" struct_h1 yang
    # sebenarnya sudah usang.
    # ══════════════════════════════════════════════════════════════
    bias_bull = bias_bear = 0

    if struct_h1=="bullish": bias_bull += 30
    if struct_h1=="bearish": bias_bear += 30

    if choch_h1["bullish_choch"]: bias_bull += 26
    if choch_h1["bearish_choch"]: bias_bear += 26

    if L1["ema9"]>L1["ema21"]>L1["ema50"]:  bias_bull += 15
    elif L1["ema9"]>L1["ema21"]:             bias_bull += 7
    if L1["ema9"]<L1["ema21"]<L1["ema50"]:  bias_bear += 15
    elif L1["ema9"]<L1["ema21"]:             bias_bear += 7
    if L1["close"]>L1["ema200"]: bias_bull += 8
    else:                          bias_bear += 8

    # D1 bias — bobot besar sengaja, karena ini representasi konteks
    # MAKRO/harian yang H1 sendiri tidak bisa lihat (H1 rentan salah baca
    # pullback lokal sebagai reversal, padahal trend besarnya masih sama).
    if d1_bias == "bullish": bias_bull += 24
    if d1_bias == "bearish": bias_bear += 24

    # RSI M15 — dipertahankan sebagai momentum filter (bukan SMC, tapi
    # tetap relevan): oversold/overbought memberi confluence ke bias.
    if rv<35:    bias_bull += 12
    elif rv<45:  bias_bull += 6
    if rv>65:    bias_bear += 12
    elif rv>55:  bias_bear += 6

    bias_dir = "bull" if bias_bull >= bias_bear else "bear"

    # ══════════════════════════════════════════════════════════════
    # LAYER 2 — SETUP: konfirmasi entry price-action/SMC (10 materi)
    # ══════════════════════════════════════════════════════════════
    setup_bull = setup_bear = 0

    # BOS (shadow cukup) & CHoCH (wajib body close) — inti SMC
    bos = detect_bos(m15, sh15, sl15)
    if bos["bb"]: setup_bull += 12
    if bos["cb"]: setup_bull += 7
    if bos["bs"]: setup_bear += 12
    if bos["cs"]: setup_bear += 7

    choch = detect_choch(m15, sh15, sl15)
    if choch["bullish_choch"]: setup_bull += 22
    if choch["bearish_choch"]: setup_bear += 22

    # (CHoCH H1 sudah dihitung di Layer 1/bias di atas — lihat komentar
    # di bagian bias_bull/bias_bear)

    # Failed Retest — trigger entry paling valid di SMC
    fr = detect_failed_retest(m15, sh15, sl15, atr_val)
    if fr["failed_retest_sell"]: setup_bear += 24
    if fr["failed_retest_buy"]:  setup_bull += 24

    fr_h1 = detect_failed_retest(h1, sh1, sl1, atr_val)
    if fr_h1["failed_retest_sell"]: setup_bear += 18
    if fr_h1["failed_retest_buy"]:  setup_bull += 18

    # Validitas & tipe pullback — corrective=ideal, sweeping=ideal
    # setelah sweep, aggressive=risiko tinggi (bobot dikurangi)
    pullback_valid_bull = is_valid_pullback(m15, "bull")
    pullback_valid_bear = is_valid_pullback(m15, "bear")
    pullback_type_bull  = classify_pullback_type(m15, "bull", atr_val)
    pullback_type_bear  = classify_pullback_type(m15, "bear", atr_val)

    if pullback_valid_bull:
        if pullback_type_bull == "aggressive": setup_bull += 3
        elif pullback_type_bull == "sweeping":  setup_bull += 14
        else:                                    setup_bull += 9
    if pullback_valid_bear:
        if pullback_type_bear == "aggressive": setup_bear += 3
        elif pullback_type_bear == "sweeping":  setup_bear += 14
        else:                                    setup_bear += 9

    # Pin bar rejection & pola Fakey — konfirmasi price action di zona
    pinbar = detect_pinbar(L15)
    if pinbar["bullish_pinbar"]: setup_bull += 10
    if pinbar["bearish_pinbar"]: setup_bear += 10

    fakey = detect_fakey(m15)
    if fakey["bullish_fakey"]: setup_bull += 10
    if fakey["bearish_fakey"]: setup_bear += 10

    # Liquidity Run vs Sweep/Swift
    liq_bull = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bear")
    if liq_bull["type"] == "run":    setup_bull += 10
    elif liq_bull["type"] == "sweep": setup_bear += 8
    if liq_bear["type"] == "run":    setup_bear += 10
    elif liq_bear["type"] == "sweep": setup_bull += 8

    # Inducement — flag saja, dipakai full_analyze/calc_discount_entry
    # untuk menunda entry, bukan mengubah skor di sini.
    inducement_bull = detect_inducement_move(m15, "bull", atr_val)
    inducement_bear = detect_inducement_move(m15, "bear", atr_val)

    # OTE (0.62-0.79) — TIDAK boleh berdiri sendiri, wajib CHoCH atau
    # FVG fresh searah sebagai pendamping.
    ote_bull = ote_bear = False
    if len(sh15) >= 1 and len(sl15) >= 1:
        swing_hi_m15 = m15["high"].iloc[sh15[-1]]
        swing_lo_m15 = m15["low"].iloc[sl15[-1]]
        fib_now = get_fib_zone(L15["close"], swing_lo_m15, swing_hi_m15)
        if 0.62 <= (1 - fib_now["ratio"]) <= 0.79: ote_bull = True
        if 0.62 <= fib_now["ratio"] <= 0.79:        ote_bear = True

    if ote_bull and (choch["bullish_choch"] or any(f.get("is_fresh") for f in find_fvg(m15, "bull", lb=30))):
        setup_bull += 10
    if ote_bear and (choch["bearish_choch"] or any(f.get("is_fresh") for f in find_fvg(m15, "bear", lb=30))):
        setup_bear += 10

    # Candle pattern dasar (hammer/shooting star) — pelengkap price action
    body=L15["close"]-L15["open"]
    low_wick=min(L15["open"],L15["close"])-L15["low"]
    up_wick=L15["high"]-max(L15["open"],L15["close"])
    if low_wick>abs(body)*1.5: setup_bull += 6
    if up_wick>abs(body)*1.5:  setup_bear += 6

    # Momentum confluence ringan (MACD/BB/Volume M15) — bukan SMC, tapi
    # masih dipakai selama relevan sesuai preferensi (bobot lebih kecil
    # dari layer SMC di atas, hanya sebagai pelengkap).
    if L15["mh"]>0 and P15["mh"]<=0:  setup_bull += 8
    elif L15["mh"]>0:                  setup_bull += 3
    if L15["mh"]<0 and P15["mh"]>=0:  setup_bear += 8
    elif L15["mh"]<0:                  setup_bear += 3

    if L15["close"]<=L15["bb_lo"]:    setup_bull += 7
    elif L15["close"]<L15["bb_mid"]:  setup_bull += 3
    if L15["close"]>=L15["bb_up"]:    setup_bear += 7
    elif L15["close"]>L15["bb_mid"]:  setup_bear += 3

    if L15["volume"]>L15["vol_sma"]*1.5:
        if L15["close"]>L15["open"]:  setup_bull += 6
        else:                          setup_bear += 6
    elif L15["volume"]>L15["vol_sma"]:
        if L15["close"]>L15["open"]:  setup_bull += 2
        else:                          setup_bear += 2

    # ══════════════════════════════════════════════════════════════
    # LAYER 3 — GATE: konfirmasi yang BERLAWANAN dengan bias struktural
    # dilemahkan drastis (dipotong separuh), bukan dijumlah rata.
    # Ini yang membedakan dari additive scoring lama — struktur besar
    # (bias) diperlakukan sebagai FILTER wajib, bukan sekadar satu
    # sumber poin di antara puluhan sumber poin lain yang setara.
    # ══════════════════════════════════════════════════════════════
    if bias_dir == "bull":
        setup_bear = setup_bear * 0.5
    else:
        setup_bull = setup_bull * 0.5

    bull = bias_bull + setup_bull
    bear = bias_bear + setup_bear

    direction="bull" if bull>=bear else "bear"
    raw=bull if direction=="bull" else bear
    conf=min(int(raw/264*100),99)

    # D1 berlawanan TOTAL dengan sinyal akhir → tetap hard block (bukan
    # cuma penalty scoring) — kalau sampai lolos scoring pun (krn Layer 2
    # setup sangat kuat) tapi D1 benar2 berlawanan, lebih aman ditolak.
    if d1_bias == "bearish" and direction == "bull": return None
    if d1_bias == "bullish" and direction == "bear": return None

    return {
        "direction"       : direction,
        "confidence"      : conf,
        "price"           : L15["close"],
        "atr"             : atr_val,
        "struct_h1"       : struct_h1,
        "d1_bias"         : d1_bias,
        "rsi"             : round(rv,1),
        "bull_pts"        : bull,
        "bear_pts"        : bear,
        "bias_dir"        : bias_dir,
        "choch_m15"       : choch,
        "choch_h1"        : choch_h1,
        "failed_retest"   : fr,
        "pullback_valid"  : pullback_valid_bull if direction == "bull" else pullback_valid_bear,
        "pullback_type"   : pullback_type_bull if direction == "bull" else pullback_type_bear,
        "pinbar"          : pinbar,
        "fakey"           : fakey,
        "liquidity_bull"  : liq_bull,
        "liquidity_bear"  : liq_bear,
        "inducement"      : inducement_bull if direction == "bull" else inducement_bear,
    }


# ═════════════════════════════════════════════
# TAHAP 2: ANALISIS ULANG — SL DULU, LALU TP
# ═════════════════════════════════════════════
# ── Tier kekuatan level untuk pemilihan TP ──────────────────────────
# Tier lebih rendah = level lebih kuat/reliable sebagai target liquidity.
def _h4_confluence(df_h1, direction, choch_m15=None):
    """
    Konfirmasi H4 untuk membuka kandidat TP Fibonacci extension.
    Resample dari H1 yang sudah di-fetch — TIDAK ada API call tambahan
    (pola sama persis dengan d1_bias di score_direction()).

    Syarat 'confluence' (unlock fib 1.272):
      BUY  : EMA9>EMA21>EMA50 H4 + struktur H4 bullish + RSI H4 di [45,68]
      SELL : EMA9<EMA21<EMA50 H4 + struktur H4 bearish + RSI H4 di [32,55]

    Syarat 'full_confluence' (unlock fib 1.618, tambahan):
      confluence di atas TERPENUHI + CHoCH M15 searah trade.
      Ini level paling jauh/spekulatif — baru boleh dipakai kalau H4
      DAN M15 dan RSI semuanya sepakat, bukan cuma H4 saja.

    Return: {"confluence": bool, "full_confluence": bool}
    """
    result = {"confluence": False, "full_confluence": False}
    try:
        df_h4 = build_df(df_h1.resample("4h").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).dropna())
        if df_h4 is None or len(df_h4) < 20:
            return result

        L4 = df_h4.iloc[-1]
        sh4, sl4 = swing_pts(df_h4, lb=3)
        struct_h4 = mkt_struct(df_h4, sh4, sl4)
        rsi_h4 = L4["rsi"]

        if direction == "bull":
            ema_ok = L4["ema9"] > L4["ema21"] > L4["ema50"]
            struct_ok = struct_h4 == "bullish"
            rsi_ok = H4_RSI_BUY_MIN <= rsi_h4 <= H4_RSI_BUY_MAX
        else:
            ema_ok = L4["ema9"] < L4["ema21"] < L4["ema50"]
            struct_ok = struct_h4 == "bearish"
            rsi_ok = H4_RSI_SELL_MIN <= rsi_h4 <= H4_RSI_SELL_MAX

        result["confluence"] = bool(ema_ok and struct_ok and rsi_ok)

        if result["confluence"] and choch_m15:
            choch_agrees = (
                (direction == "bull" and choch_m15.get("bullish_choch")) or
                (direction == "bear" and choch_m15.get("bearish_choch"))
            )
            result["full_confluence"] = bool(choch_agrees)
    except Exception:
        pass
    return result


def _fib_extension_levels(h1, sh1, sl1, direction):
    """
    Proyeksi Fibonacci extension dari leg swing H1 terakhir (low→high untuk
    BUY, high→low untuk SELL). Bukan angka dikarang — ini proyeksi dari
    RENTANG pergerakan H1 yang sudah benar-benar terjadi di chart.

    Return: (fib_127_price, fib_162_price) atau (None, None) kalau swing
    H1 belum cukup terbentuk.
    """
    if not sh1 or not sl1:
        return None, None
    swing_high = h1["high"].iloc[sh1[-1]]
    swing_low  = h1["low"].iloc[sl1[-1]]
    leg = swing_high - swing_low
    if leg <= 0:
        return None, None

    if direction == "bull":
        return swing_high + leg * FIB_EXT_1, swing_high + leg * FIB_EXT_2
    else:
        return swing_low - leg * FIB_EXT_1, swing_low - leg * FIB_EXT_2


TP_RR_CAP = MIN_RR * 2   # RR di atas ini ditarik mundur ke titik RR=cap, arah tetap ke level terkuat

def _select_best_tp(tp_pool, entry_price, risk):
    """
    Pilih TP dari level PALING KUAT (tier terendah) di antara semua
    kandidat yang lolos floor RR >= MIN_RR. Seri tier → ambil RR tertinggi.
    Kalau RR ke level itu > TP_RR_CAP, TP ditarik mundur (searah entry->level)
    ke titik yang menghasilkan RR = TP_RR_CAP — arah tetap ke level kuat itu,
    cuma jaraknya dipersingkat supaya lebih realistis tersentuh.
    """
    qualifying = []
    for lbl, v, tier in tp_pool:
        rr_c = abs(v - entry_price) / risk
        if rr_c >= MIN_RR:
            qualifying.append((lbl, v, tier, rr_c))
    if not qualifying:
        return None, None
    best_lbl, best_v, best_tier, best_rr = min(qualifying, key=lambda x: (x[2], -x[3]))
    if best_rr > TP_RR_CAP:
        sgn = 1 if best_v > entry_price else -1
        best_v = entry_price + sgn * risk * TP_RR_CAP
        best_lbl += "_capped"
    return round(best_v, 8), best_lbl


def _build_tp_pool(m15, h1, direction, entry_price, atr, sh15, sl15, sh1, sl1, h4_gate, fib_127, fib_162):
    """TP pool searah entry (bear=bawah, bull=atas), tier makin kecil makin kuat."""
    up = direction == "bull"
    zones_m15 = find_zones(m15, "demand" if up else "supply")
    zones_h1  = find_zones(h1, "demand" if up else "supply")
    fvgs      = find_fvg(m15, "bull" if up else "bear")
    eqs_m15   = find_equal_highs_lows(m15, "high" if up else "low", lb=80)
    eqs_h1    = find_equal_highs_lows(h1, "high" if up else "low", lb=50)
    sw_m15    = [m15["high" if up else "low"].iloc[i] for i in (sh15 if up else sl15)]
    sw_h1     = [h1["high" if up else "low"].iloc[i] for i in (sh1 if up else sl1)]
    sgn = 1 if up else -1
    pool = []

    for v in eqs_h1:
        if sgn*(v - entry_price) > atr*1.0: pool.append(("eq_h1", v, 1))
    for z in zones_h1:
        edge = z["bot"] if up else z["top"]
        if sgn*(edge - entry_price) > atr*1.0: pool.append(("zone_h1", edge, 2))
    for v in sw_h1:
        if sgn*(v - entry_price) > atr*1.0: pool.append(("sw_h1", v, 3))
    for v in eqs_m15:
        if sgn*(v - entry_price) > atr*0.5: pool.append(("eq_m15", v, 4))
    for z in zones_m15:
        edge = z["bot"] if up else z["top"]
        if sgn*(edge - entry_price) > atr*0.5:
            pool.append(("zone_m15", edge, 5 - (0.4 if z.get("is_fresh") else 0)))
    for f in fvgs:
        if sgn*(f["mid"] - entry_price) > atr*0.5:
            t = 6 - (0.4 if f.get("candle3") == "breakaway" else 0) - (0.2 if f.get("is_fresh") else 0)
            pool.append(("fvg_m15", f["mid"], t))
    for v in sw_m15:
        if sgn*(v - entry_price) > atr*0.5: pool.append(("sw_m15", v, 7))

    if fib_127 is not None and sgn*(fib_127 - entry_price) > atr*0.5 and h4_gate["confluence"]:
        pool.append(("fib127", fib_127, 8))
        if h4_gate["full_confluence"] and fib_162 is not None and sgn*(fib_162 - entry_price) > atr*0.5:
            pool.append(("fib162", fib_162, 9))
    return pool


def analyze_setup(df_h1, df_m15, direction, entry_price, score=None, invalid_level=None):
    """
    SL = seberang titik entry itu sendiri (invalid_level dari
    calc_discount_entry) + buffer noise kecil. Kalau harga sentuh SL,
    itu artinya struktur di entry ini TERBUKTI invalid (bukan sekadar
    "belum dikonfirmasi") — bukan liquidity pool berikutnya yang jauh.
    Tidak ada level jelas → return None (skip, cari koin lain).
    TP = tier pool terkuat dengan floor RR >= MIN_RR (lihat _select_best_tp).
    """
    h1, m15 = build_df(df_h1), build_df(df_m15)
    if h1 is None or m15 is None: return None

    # ATR M15 saja bisa under-estimate kalau harga baru selesai fase
    # spike/impulsif besar lalu masuk konsolidasi sempit (M15 "tenang"
    # tapi itu semu — koin baru saja terbukti bisa bergerak liar).
    # Pakai ATR H1 juga sebagai pembanding supaya buffer SL tetap
    # proporsional terhadap volatilitas riil koin, bukan cuma window
    # M15 saat ini yang mungkin kebetulan sempit.
    atr_m15 = m15["atr"].iloc[-1]
    atr_h1  = h1["atr"].iloc[-1] / 4   # ATR H1 diskalakan kasar ke basis M15
    atr = max(atr_m15, atr_h1, entry_price * 0.002)
    noise = atr * 0.6   # buffer anti-noise — dinaikkan dari 0.25x krn ATR
                         # M15 sendirian gampang under-estimate saat harga
                         # baru keluar dari candle spike besar (lihat kasus
                         # KAITOUSDT: SL kena dlm 4 menit oleh wick biasa)

    if invalid_level is None:
        return None

    sl_price = invalid_level + (noise if direction == "bear" else -noise)
    risk = abs(sl_price - entry_price)
    risk_floor = max(atr * 0.8, entry_price * 0.003)   # dinaikkan dari 0.4x/0.0015
    if risk < risk_floor:
        sl_price += (risk_floor - risk) * (1 if direction == "bear" else -1)
        risk = risk_floor
    if risk <= 0: return None

    sh15, sl15 = swing_pts(m15, lb=5)
    sh1, sl1   = swing_pts(h1, lb=5)
    choch_m15  = (score or {}).get("choch_m15", {})
    h4_gate    = _h4_confluence(df_h1, direction, choch_m15)
    fib_127, fib_162 = _fib_extension_levels(h1, sh1, sl1, direction)

    tp_pool = _build_tp_pool(m15, h1, direction, entry_price, atr,
                              sh15, sl15, sh1, sl1, h4_gate, fib_127, fib_162)
    tp_price, tp_label = _select_best_tp(tp_pool, entry_price, risk)
    if tp_price is None: return None

    reward = abs(tp_price - entry_price)
    rr = round(reward / risk, 2)
    if rr < MIN_RR: return None

    return {
        "sl": round(sl_price, 8), "tp": round(tp_price, 8), "rr": rr,
        "reason": f"SL@{sl_price:.5g}(invalidation) | TP@{tp_price:.5g}({tp_label})",
    }




def _zone_score(z):
    """Skor kekuatan zona OB/S&D: fresh + fvg + bos + breakaway fib align."""
    return z.get("quality", 0) + int(z.get("fib_aligned", False))


def _collect_entry_candidates(m15, direction, entry_ref, atr):
    """
    Kumpulkan semua kandidat entry (OB, FVG, sweep raw level, fib adaptif)
    dengan skor kekuatan masing-masing. direction: 'bull' cari di bawah
    entry_ref, 'bear' cari di atas entry_ref.
    sweep_side: sisi zona yang jadi TITIK ENTRY (ujung sweep, dekat harga)
    invalid_side: sisi seberang zona (dipakai sebagai basis SL nanti)

    FIX PRESISI-ENTRY (v2, REBALANCED): sebelumnya kandidat cuma
    dibandingkan lewat skor kualitas mentah (freshness/FVG/BOS), TANPA
    mempertimbangkan seberapa jauh zona itu dari harga saat ini — analisa
    2 backtest independen (>600 sinyal gabungan) menemukan median jarak
    ke entry 2.2-2.5× lebih jauh drpd jarak ke TP itu sendiri, bikin TP
    sering kesentuh duluan sebelum harga sempat pullback. v1 (penalti
    jarak besar + fib_adaptive ikut bersaing bebas) TERBUKTI kelewatan:
    trade naik 56% tapi SL naik 150%, Profit Factor nyaris separuh. v2
    ini menurunkan bobot penalti jarak jadi cuma tie-break TIPIS antar
    kandidat SEJENIS yang sudah sebanding kualitasnya, dan fib_adaptif
    dikembalikan jadi last-resort murni (tapi bug invalid_level=None-nya
    tetap diperbaiki, jadi minimal BISA menghasilkan trade saat memang
    tidak ada OB/FVG/EQ sama sekali — dulu jalur itu mati total).
    """
    up = direction == "bear"
    obs = find_zones(m15, direction, strict=True)
    fvgs = find_fvg(m15, direction)
    eqs = find_equal_highs_lows(m15, "high" if up else "low", lb=80)
    cands = []

    def _dist_penalty(price):
        # REBALANCE: bobot diturunkan 0.4→0.15 setelah data run pertama
        # menunjukkan versi 0.4 terlalu agresif — SL naik 150% padahal
        # trade cuma naik 56%, Profit Factor nyaris separuh (10.2→5.3).
        # Sekarang cuma nge-geser tie-break TIPIS antar kandidat SEJENIS
        # yang kualitasnya sudah sebanding (OB vs OB, FVG vs FVG dst),
        # bukan lagi cukup besar utk bikin zona lemah-tapi-dekat ngalahin
        # zona kuat-tapi-agak-jauh.
        if atr <= 0: return 0.0
        return (abs(price - entry_ref) / atr) * 0.15

    for z in obs:
        entry_pt, invalid_pt = (z["top"], z["bot"]) if up else (z["bot"], z["top"])
        if (up and entry_pt > entry_ref + atr*0.1) or (not up and entry_pt < entry_ref - atr*0.1):
            cands.append({"price": entry_pt, "invalid": invalid_pt, "label": "ob",
                           "score": 3 + _zone_score(z) - _dist_penalty(entry_pt)})
    for f in fvgs:
        if (up and f["mid"] > entry_ref + atr*0.1) or (not up and f["mid"] < entry_ref - atr*0.1):
            sc = 2 + int(f.get("is_fresh", False)) + 2*int(f.get("candle3") == "breakaway")
            invalid_pt = f["top"] if up else f["bot"]
            cands.append({"price": f["mid"], "invalid": invalid_pt, "label": "fvg",
                           "score": sc - _dist_penalty(f["mid"])})
    eqs_sorted = sorted(eqs) if up else sorted(eqs, reverse=True)
    for lv in eqs_sorted[:1]:
        if (up and lv > entry_ref + atr*0.2) or (not up and lv < entry_ref - atr*0.2):
            cands.append({"price": lv, "invalid": lv + (atr*0.6 if up else -atr*0.6),
                           "label": "eq", "score": 2 - _dist_penalty(lv)})

    # Fib adaptif — REBALANCE: dikembalikan jadi TRUE LAST RESORT (cuma
    # dipakai kalau BENAR-BENAR tidak ada OB/FVG/EQ sama sekali), bukan
    # ikut bersaing bebas di pool utama lagi. Data run pertama (versi
    # "ikut bersaing") menunjukkan fib generik terlalu sering menang
    # padahal secara struktural lebih lemah drpd OB/FVG asli — itu
    # kontributor utama SL melonjak. Yang TETAP diperbaiki dari versi
    # asli: dulu invalid_level selalu None di jalur ini → analyze_setup
    # SELALU menolaknya (bug lama, fib_adaptive tidak pernah benar2
    # menghasilkan trade). Sekarang dikasih invalid_level yang benar
    # (tepi dalam zona) supaya minimal BISA dipakai saat memang tidak
    # ada alternatif lain — lebih baik drpd skip sepenuhnya.
    if not cands:
        try:
            sh15, sl15 = swing_pts(m15, lb=5)
            if len(sh15) >= 1 and len(sl15) >= 1:
                lo, hi = adaptive_fib_target(m15, sh15, sl15, direction)
                swing_hi = m15["high"].iloc[sh15[-1]]
                swing_lo = m15["low"].iloc[sl15[-1]]
                leg = swing_hi - swing_lo
                px = (swing_lo + leg*lo) if up else (swing_hi - leg*lo)   # tepi dangkal = lo
                invalid_fib = (swing_lo + leg*hi) if up else (swing_hi - leg*hi)  # tepi dalam = SL basis
                if (up and px > entry_ref + atr*0.1) or (not up and px < entry_ref - atr*0.1):
                    cands.append({"price": px, "invalid": invalid_fib, "label": "fib_adaptive",
                                   "score": 1.5})
        except Exception:
            pass

    return cands


def calc_discount_entry(df_h1, df_m15, direction, current_price, atr):
    """
    Entry = kandidat terkuat (OB fresh > FVG breakaway > EQ > fib adaptif),
    dibandingkan lewat skor YANG SUDAH memperhitungkan jarak dari harga
    (lihat _collect_entry_candidates) — bukan cuma kualitas mentah. Fib
    adaptif sekarang ikut bersaing di pool yang sama (bukan fallback
    terakhir saja), jadi kalau OB/FVG yang ada semuanya jauh, alternatif
    fib yang lebih reachable bisa menang.
    Return (entry_price, label, invalid_level) — invalid_level dipakai
    analyze_setup() sebagai basis SL (seberang titik entry ini sendiri).
    """
    m15 = build_df(df_m15)
    if m15 is None: return current_price, "market", None
    cands = _collect_entry_candidates(m15, direction, current_price, atr)
    if cands:
        best = max(cands, key=lambda c: c["score"])
        return round(best["price"], 8), best["label"], best["invalid"]
    return current_price, "market", None


# ═════════════════════════════════════════════
# PIPELINE ANALISIS LENGKAP
# ═════════════════════════════════════════════
def full_analyze(symbol):
    """
    1. Score arah sinyal (H1 + M15 + D1 bias)
    2. Hitung entry diskon dari OB/FVG/EQL/Fib
    3. Hitung SL/TP dari entry diskon
    Entry = zona struktural, bukan market price
    """
    try:
        df_h1  = get_klines(symbol, "1h",  250)
        df_m15 = get_klines(symbol, "15m", 250)
        if df_h1.empty or df_m15.empty: return None
        # D1 ASLI (histori panjang, bukan resample window 250-jam) —
        # lihat catatan di score_direction(). 100 candle harian, no
        # extra API load karena cuma dipanggil sekali per scan simbol.
        try:
            df_d1 = get_klines(symbol, "1d", 100)
        except Exception:
            df_d1 = None

        score = score_direction(df_h1, df_m15, df_d1)
        if score is None: return None

        original_dir  = score["direction"]
        current_price = score["price"]
        atr_val       = score["atr"]
        decision      = "BUY"  if original_dir == "bull" else "SELL"

        # ── Inducement-aware confidence adjustment ───────────────────────
        # Kalau terindikasi inducement (gerakan kecil pancingan) BARU SAJA
        # terjadi dan belum ada CHoCH searah yang mengkonfirmasi shift
        # struktur, turunkan confidence sedikit — mendorong sinyal ini
        # untuk tidak lolos MIN_CONFIDENCE kalau memang masih marginal,
        # alih-alih entry di gerakan/breakout pertama yang berisiko jadi
        # jebakan (bukan hard block, supaya sinyal yang memang sangat
        # kuat dari indikator lain tetap bisa lolos).
        confidence = score["confidence"]
        choch_confirms = (
            (original_dir == "bull" and score.get("choch_m15", {}).get("bullish_choch")) or
            (original_dir == "bear" and score.get("choch_m15", {}).get("bearish_choch"))
        )
        if score.get("inducement") and not choch_confirms:
            confidence = max(0, confidence - 8)

        # Kalau pullback yang mendasari sinyal ini AGGRESSIVE (momentum
        # kuat melawan, reaksi di zona rendah probabilitasnya) turunkan
        # sedikit juga, kecuali sudah ada CHoCH searah yang menguatkan.
        if score.get("pullback_type") == "aggressive" and not choch_confirms:
            confidence = max(0, confidence - 5)

        # Entry diskon dari zona struktural
        discount_entry, entry_label, invalid_level = calc_discount_entry(
            df_h1, df_m15, original_dir, current_price, atr_val)

        # SL/TP dihitung dari entry diskon
        setup = analyze_setup(df_h1, df_m15, original_dir, discount_entry,
                               score=score, invalid_level=invalid_level)
        if setup is None: return None

        # TP wajib MASIH di depan harga sekarang. Kalau entry diskon
        # dihitung dari zona struktural yang sudah ditinggalkan jauh oleh
        # rally/dump kuat (biasanya RSI sudah ekstrem), TP hasil analisa
        # dari zona lama itu bisa sudah KELEWAT harga sekarang — sinyal
        # ini mati sebelum pending order sempat dibuat. Tolak di sini,
        # bukan menunggu pending-cancel logic menangkapnya belakangan.
        if original_dir == "bull" and current_price >= setup["tp"]:
            return None
        if original_dir == "bear" and current_price <= setup["tp"]:
            return None

        return {
            "symbol"       : symbol,
            "original_dir" : original_dir,
            "decision"     : decision,
            "confidence"   : confidence,
            "price"        : current_price,
            "entry"        : discount_entry,
            "entry_label"  : entry_label,
            "sl"           : setup["sl"],
            "tp"           : setup["tp"],
            "rr"           : setup["rr"],
            "rsi"          : score["rsi"],
            "struct_h1"    : score["struct_h1"],
            "d1_bias"      : score.get("d1_bias", "neutral"),
            "choch_m15"    : score.get("choch_m15", {}),
            "choch_h1"     : score.get("choch_h1", {}),
            "failed_retest": score.get("failed_retest", {}),
            "tp_sl_reason" : f"Entry@{discount_entry:.5g}({entry_label}) | {setup['reason']}",
        }
    except Exception as e:
        log.debug(f"[full_analyze] {symbol}: {e}")
        return None


# ═════════════════════════════════════════════
# SCAN — 1 sinyal terbaik
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
        r=full_analyze(sym)
        if r: results.append(r)
        time.sleep(0.08)

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
                 sym=None, decision=None, entry_time=None):
    """
    Hitung P&L simulasi murni dari jarak harga analisis (lihat komentar
    lama untuk detail model close_price). Tambahan: catat sym/decision/
    entry_time/exit_time ke pnl_history untuk /backtest.

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
        })

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

    return (
        f"📊 <b>Statistik</b> — {t} trade | TP {tp} SL {sl} Trail {trail}\n"
        f"Win Rate: <b>{wr:.1f}%</b> (TP+Trail vs SL)\n\n"
        f"Modal: ${STARTING_BALANCE:.2f} → Saldo: <b>${bal:.4f}</b> "
        f"({sgn}{pnl_pct:.2f}%)\n\n"
        f"5 terakhir:\n{hist_str}\n\n"
        f"🚫 Banned: {len(banned_coins)}"
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
                 sym=sym, decision=sig.get("decision"), entry_time=pos.get("entry_time"))
    _ban_coin(sym, f"trade closed ({result})", duration=BAN_DURATION_TRADE_CLOSED)

    # Update active_trade jika ini yang sedang dipantau
    with positions_lock:
        if not positions:
            active_trade = None

    emoji = {"tp":"🎯","sl":"🛑","trail":"🔒"}.get(result,"❓")
    label = {"tp":"TAKE PROFIT","sl":"STOP LOSS","trail":"TRAILING STOP"}.get(result, result.upper())
    tg_send(cid, f"{emoji} <b>{label}</b> — {sym}\n\n" + fmt_stats())


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

        # Launch scan di background
        threading.Thread(target=_do_scan, daemon=True).start()

        # Jeda antar scan agar tidak langsung re-scan begitu selesai
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
    "/timeout SYMBOL      — Tutup paksa posisi tertentu\n"
    "/timeout             — Tutup paksa semua posisi\n"
    "/stats               — Statistik + saldo\n"
    "/backtest             — 20 trade terakhir (evaluasi)\n"
    "/banned              — Daftar koin ban\n"
    "/resetban            — Hapus semua ban\n"
    "/resetbalance        — Reset saldo ke $10\n"
    "/info                — Detail metode analisis\n"
    "━━━━━━━━━━━━━━━━━━━━\n\n"
    "⚠️ <i>Simulasi saja — bukan saran finansial.</i>"
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
    global auto_mode, auto_thread, active_chat_id, timeout_flag, MAX_POSITIONS, MIN_CONFIDENCE

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
                text=msg.get("text","").strip().lower()
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
                elif text in ("/banned","banned"):
                    with ban_lock:
                        cur_scan = scan_counter
                        b = sorted(banned_coins.items())
                    if b:
                        lines = []
                        for sym, (banned_at, dur) in b:
                            remaining = max(0, dur - (cur_scan - banned_at))
                            lines.append(f"• {sym} (unban dalam {remaining} scan)")
                        tg_send(chat_id,
                            f"🚫 <b>Banned ({len(b)}):</b>\n" + "\n".join(lines))
                    else:
                        tg_send(chat_id, "✅ Belum ada ban.")
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
    # Semua thread di atas daemon=True — main thread harus tetap hidup,
    # kalau tidak proses langsung exit begitu baris ini selesai.
    while True:
        time.sleep(3600)
