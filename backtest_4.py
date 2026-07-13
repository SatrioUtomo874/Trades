#!/usr/bin/env python3
"""
backtest-4.py — SMC Signal Backtester (offline, RESOLUSI M1)
============================================================
Sama seperti backtest-3.py (per-koin, stake fixed, R-ladder+structure
trail, D1-bias, pending-confirm, dll — semua fix sebelumnya TETAP ada,
diimpor langsung dari try22.py jadi otomatis ikut) — TAPI manajemen
posisi (entry-fill/SL-before-entry/TP-SL-order/trailing) sekarang jalan
di resolusi M1 (candle 1 menit), bukan M15. Analisa sinyal (score_
direction dkk) tetap di M15/H1/D1 seperti biasa — cuma presisi EKSEKUSI
yang naik ~15×.

Sumber data: crypto_m1_3bulan.csv(.gz) — cuma 5 koin (DASH, NEAR, SOL,
XPL, XRP), 3 bulan, TANPA kolom volume (diisi konstan, lihat load_csv).

Cara pakai:
  python backtest-4.py --coins SOL
  python backtest-4.py --coins SOL XRP --verbose
  python backtest-4.py                         (semua 5 koin M1)
"""

import os, sys, argparse, time
from collections import defaultdict

# ── 1. Setup env SEBELUM import try22 ──────────────────────────────────────
os.environ.setdefault("TELEGRAM_TOKEN", "backtest_dummy_token")
os.environ.setdefault("ALLOWED_USER_ID", "0")

# ── 2. Mock modul yang tidak dibutuhkan ────────────────────────────────────
from unittest.mock import MagicMock
for _m in ["flask", "requests", "urllib3", "urllib3.exceptions",
           "urllib3.util", "urllib3.util.retry", "dotenv"]:
    sys.modules.setdefault(_m, MagicMock())
sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None

# ── 3. Resolve path try22.py (folder sama atau attached_assets/) ───────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_SCRIPT_DIR, os.path.join(_SCRIPT_DIR, "attached_assets")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import importlib
    _try22 = importlib.import_module("try22_1783656381597")
    print("✅  try22.py berhasil di-import")
except Exception as _e:
    print(f"❌  Gagal import try22: {_e}")
    print("    Letakkan try22_1783656381597.py satu folder dengan backtest-4.py")
    sys.exit(1)

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as _swv


# ════════════════════════════════════════════════════════════════════════════
# PATCH KRITIS — HARUS DIPASANG SEBELUM FUNGSI LAIN DIPAKAI
# ════════════════════════════════════════════════════════════════════════════

# ── PATCH 1: swing_pts → numpy vectorized ─────────────────────────────────
# Masalah asal: pandas iloc-loop O(N×lb) → 93% dari waktu score_direction.
# Solusi: numpy sliding_window_view, kompleksitas sama tapi 20× lebih cepat.
# Output identik dengan versi asli try22.py.

def _fast_swing_pts(df, lb=5):
    hi = df["high"].values
    lo = df["low"].values
    n = len(hi)
    if n < 2 * lb + 1:
        return [], []
    wh = _swv(hi, 2 * lb + 1)
    wl = _swv(lo, 2 * lb + 1)
    # Posisi lokal pivot: nilai tengah = max/min dari jendela
    sh = (lb + np.where(wh[:, lb] == wh.max(axis=1))[0]).tolist()
    sl = (lb + np.where(wl[:, lb] == wl.min(axis=1))[0]).tolist()
    return sh, sl

_try22.swing_pts = _fast_swing_pts


# ── PATCH 2: build_df → skip jika indikator sudah ada ─────────────────────
# Masalah asal: score_direction, calc_discount_entry, analyze_setup semua
# memanggil build_df lagi meski kita sudah passing DataFrame yang sudah
# punya kolom EMA/RSI/MACD/ATR — memboroskan ~40ms per panggilan.
# Solusi: kalau "ema9" sudah ada, langsung return (tidak recompute).
#
# BUG NYATA (penyebab backtest ≠ chart real-time) — PATCH 2b:
# Indikator di-precompute SEKALI dari SELURUH histori CSV (ribuan candle),
# lalu tiap scan tinggal di-slice 250 bar terakhir. Untuk RSI/ATR/MACD/BB/
# vol_sma ini AMAN (rolling window ≤26 bar, hasil di bar terakhir sama
# persis baik dihitung dari histori penuh maupun dari window 250 bar saja).
# TAPI untuk EMA9/21/50/200, try22.py LIVE selalu fetch cuma
# get_klines(limit=250) lalu build_df dari situ — "cold start", EMA
# hanya "matang" dari histori pendek itu. EMA200 (span 200) di window
# 250 bar BELUM konvergen (sisa bobot dari elemen pertama window masih
# ~8%), sedangkan versi precompute-histori-penuh di sini sudah matang.
# Hasilnya: L1["close"]>L1["ema200"] (bobot ±8 poin, bisa nge-flip
# bias_dir di skor yang tipis) sering BEDA antara backtest vs live —
# diverifikasi langsung di data: ~3% dari seluruh titik scan berbeda
# arahnya. INI SALAH SATU SUMBER UTAMA backtest ≠ chart real-time.
# Fix: timpa HANYA baris terakhir kolom ema9/21/50/200 dengan nilai
# "cold start" asli dari window yang sama (meniru persis live),
# numpy vectorized (closed-form, bukan loop python) supaya tetap cepat.

_orig_build_df = _try22.build_df

def _ema_coldstart_last(close_arr, span):
    """Nilai EMA di elemen TERAKHIR, cold-start HANYA dari close_arr
    (seed = elemen pertama array) — identik dengan
    pd.Series(close_arr).ewm(span=span, adjust=False).mean().iloc[-1]
    tapi vectorized numpy murni (tervalidasi cocok s.d. 1e-14)."""
    n = len(close_arr)
    if n < 2:
        return close_arr[-1] if n else np.nan
    alpha = 2.0 / (span + 1)
    beta  = 1.0 - alpha
    w = beta ** np.arange(n - 2, -1, -1)
    return beta ** (n - 1) * close_arr[0] + alpha * np.dot(w, close_arr[1:])

def _fast_build_df(df):
    if df is None or len(df) < 60:
        return None
    if "ema9" in df.columns:          # sudah prebuilt → skip rekomputasi berat
        d = df.dropna()
        if len(d) < 60:
            return None
        d = d.copy()                  # jangan mutasi cache precompute asli
        close_arr = d["close"].to_numpy()
        for col, span in (("ema9", 9), ("ema21", 21), ("ema50", 50)):
            d.iloc[-1, d.columns.get_loc(col)] = _ema_coldstart_last(close_arr, span)
        span200 = 200 if len(close_arr) >= 200 else 50   # replika kondisi asli try22.py
        d.iloc[-1, d.columns.get_loc("ema200")] = _ema_coldstart_last(close_arr, span200)
        return d
    return _orig_build_df(df)

_try22.build_df = _fast_build_df
build_df = _fast_build_df            # alias lokal


# ── Bind fungsi analisis yang akan dipakai ─────────────────────────────────
score_direction     = _try22.score_direction
calc_discount_entry = _try22.calc_discount_entry
analyze_setup       = _try22.analyze_setup


# ════════════════════════════════════════════════════════════════════════════
# KONFIGURASI BACKTEST
# ════════════════════════════════════════════════════════════════════════════
CSV_DEFAULT       = "crypto_m1_3bulan.csv"   # .csv, bukan .gz (loader tetap terima .gz juga kalau ada)
STARTING_BALANCE  = 10.0     # sama dengan try22.py
# MIN_CONFIDENCE: dinaikkan dari 40 ke 45. Analisis 182 trade real dari
# backtest_result.csv sebelumnya menunjukkan bucket confidence 40-44
# WR=65.0% vs 45-49 WR=73.3%; retest retroaktif dgn trail baru:
# conf>=45 → WR 94.9% (n=39) vs conf>=40 → WR 81.9% (n=182). Trade-off:
# jumlah sinyal jauh lebih sedikit (frekuensi turun ~5x). Live try22.py
# defaultnya malah 50 (lebih konservatif lagi) — kalau mau win rate
# setinggi mungkin & terima frekuensi lebih rendah, coba --conf 50.
MIN_CONFIDENCE    = 45
# POSITION SIZE — FIXED, TIDAK COMPOUNDING (permintaan user):
# "modal awal itu adalah quantity itu sendiri" → tiap trade SELALU pakai
# $STARTING_BALANCE sebagai stake, apapun saldo kumulatif saat itu. Jadi
# tiap koin benar2 independen (hasil koin A tidak mempengaruhi ukuran
# posisi koin B), dan "saldo akhir" = STARTING_BALANCE + jumlah semua
# pnl_usd tiap trade (penjumlahan murni, bukan kurva ekuitas compounding).
POSITION_SIZE_USD = STARTING_BALANCE
POSITION_SIZE_PCT = 100.0    # disimpan utk kompatibilitas tampilan saja

# Window data per scan — sama dengan try22.py: get_klines(symbol, "1h", 250)
# Setelah numpy patch kecepatan hampir sama dengan 120 bar (5.5ms vs 5.8ms)
M15_WINDOW  = 250
H1_WINDOW   = 250

# Trailing stop — REDESIGN KE R-MULTIPLE (bukan persen absolut lagi).
# Analisa mendalam thd backtest_result_New.csv (543 trade + forward-
# replay harga asli di Datasheet.csv) menemukan step PERSEN ABSOLUT itu
# sendiri cacat desain:
#   • 51% dari SEMUA trade py risk (jarak SL) < 0.6% dari entry — utk
#     trade begini, threshold step-1 lama (0.6% profit) = butuh >1R
#     gerakan favorable dulu baru dapat proteksi APA PUN.
#   • Dari 151 trade yg akhirnya SL: 80.8%-nya SEMPAT profit dulu
#     (median 0.56R, avg 0.85R) sebelum berbalik ke SL — bukan salah
#     arah, cuma tidak sempat terkunci krn ambang absolut kelewat jauh
#     dari risk trade itu sendiri.
#   • Dari 375 trade yg exit via Trail: 62.4% MEMANG akan balik ke SL
#     asli kalau tidak ditrail (trail bekerja benar) — tapi 37.6% MALAH
#     lanjut ke TP kalau tidak ditrail (rata2 +1.7%/trade hilang sia-sia).
#     → TP cap BUKAN penyebabnya (Trail selalu terjadi SEBELUM harga
#     sempat ke TP) — jadi solusinya kalibrasi trail ke R-multiple,
#     BUKAN menghapus TP.
# TRAIL_R_LADDER = [(ambang_R, lock_ratio), ...] — begitu profit (dlm
# kelipatan risk R trade itu sendiri) capai ambang, kunci lock_ratio dari
# level itu (dlm R juga). Tervalidasi via replay 543 trade: WR 69.4%→
# 82.1%, PnL 165.05%→154.11% (turun wajar — sebagian dulunya SL penuh
# sekarang jadi trail kecil, trade-off sepadan utk win rate jauh lebih
# tinggi & proteksi merata ke semua ukuran risk).
# RETUNE ekor (2.2R/3.0R → 2.0R/2.8R, lock 60%/70% → 65%/80%): forward-
# replay thd backtest_result.csv (110 trade, 3 bulan terpisah dari dataset
# tuning awal) menemukan 60% trade exit-via-Trail MEMANG akan balik ke SL
# asli kalau tidak ditrail (dipertahankan), tapi 40% lanjut ke TP kalau
# tidak ditrail (avg +1.93%/trade hilang) — ekor lama melepas terlalu
# banyak di rentang 2-3R. Ekor baru tervalidasi silang di DUA dataset
# independen (110 trade baru & 543 trade lama): win rate SAMA PERSIS di
# keduanya (90.0% & 82.1%), PnL naik konsisten (+4.7pp & +10.7pp). Tahap
# 0.5-1.5R (paling berpengaruh ke win rate) TIDAK diubah.
# RETUNE FINAL (validasi M1 resolusi penuh, backtest_result_m1.csv, 356
# trade): lock ratio dinaikkan di semua tahap (threshold R tetap) — WR
# sama (71.1%), PnL naik tipis (66.98%→67.32%). Sudah dicoba turunkan
# threshold R pertama (banyak SL di M1 cuma sempat MFE~0.28R) — WR bisa
# naik s.d. 83% tapi PnL SELALU turun (51-64%), jadi TIDAK diambil —
# bukan perbaikan bersih, cuma trade-off yang sudah pernah dieksplorasi.
TRAIL_R_LADDER = [
    (0.5, 0.18),   # profit capai 0.5R → kunci 18% dari 0.5R
    (1.0, 0.38),   # 1.0R → kunci 38%
    (1.5, 0.55),   # 1.5R → kunci 55%
    (2.0, 0.70),   # 2.0R → kunci 70%
    (2.8, 0.85),   # 2.8R → kunci 85%
]

# ── TRAILING STOP — KOMPONEN STRUKTUR (tetap dipakai, tidak berubah) ───
# Dibandingkan head-to-head di data Datasheet.csv (417 trade riil, replay
# ulang M15) dengan beberapa pendekatan:
#   no_trail (murni TP/SL)        WR 35.3%  PnL 143.74%
#   fixed_pct step0.6/lock0.6     WR 70.7%  PnL 131.44%   ← versi lama
#   ATR-based (k=1..3)            WR 46.8-64.7%  PnL 104-148%
#   breakeven→ATR/structure       WR 21-56%  PnL 58-118%  (SEMUA LEBIH BURUK
#                                  — breakeven terlalu dini justru men-scratch
#                                  banyak trade yang sebenarnya lanjut profit)
#   structure murni (swing point) WR 40.0-49.2%  PnL 140-156%
#   KOMBO fixed_pct + structure   WR 70.3%  PnL 137.82%
#
# Structure murni SENDIRIAN (tanpa komponen R-ladder) menang PnL tipis
# tapi win rate jauh lebih rendah (40-49%) krn butuh lb*2+1 candle utk
# konfirmasi swing pertama — di awal trade (paling rawan whipsaw) tidak
# ada proteksi sama sekali. Kombo menutupi kelemahan itu: R-ladder
# menjaga di awal (relatif ke risk masing-masing trade), structure
# mengambil alih begitu ada swing point valid yg lebih baik.
STRUCT_TRAIL_LB       = 2       # swing pivot lookback (kanan-kiri) di M15
STRUCT_TRAIL_BUF_PCT  = 0.0015  # buffer 0.15% di bawah/atas swing point
STRUCT_TRAIL_LOOKBACK = 60      # jumlah candle M15 ke belakang utk deteksi swing

# Pending: max 8 jam = 32 candle M15
PENDING_MAX_CANDLES = 32

# Live try22.py scan TERUS-MENERUS (tiap ~5-15 detik, lihat simulation_loop
# di try22.py, ditambah sampai MAX_POSITIONS=20 posisi paralel di ~50 koin)
# — BUKAN cuma 1x/jam, dan candle H1/M15 terakhirnya seringkali MASIH
# BERJALAN (belum close) saat itu di-scan. Data CSV di sini resolusinya
# M15 (candle 15 menit) — itu batas paling rapat yang bisa dipakai TANPA
# lookahead. DENSE_SCAN=True (default) membuat backtest scan tiap candle
# M15 (bukan cuma tiap jam) pakai H1/M15 TERAKHIR YANG SUDAH CLOSE — jauh
# lebih dekat ke frekuensi live dibanding mode lama (1x/jam). TAPI TETAP
# TIDAK akan menyamai live yang scan tiap detik dgn candle yang masih
# berjalan — itu butuh data tick/sub-menit yang tidak ada di CSV ini,
# jadi jumlah trade backtest akan selalu lebih sedikit dari live secara
# struktural, bukan karena bug. Set False kalau mau balik ke mode 1x/jam
# (lebih cepat, tapi sinyal lebih sedikit lagi).
DENSE_SCAN = True



# ════════════════════════════════════════════════════════════════════════════
# WRAPPER ANALISIS — replika full_analyze() dari try22.py (tanpa API)
# ════════════════════════════════════════════════════════════════════════════

def analyze_from_df(df_h1, df_m15, df_d1=None):
    """
    Replika full_analyze() dari try22.py tapi menerima DataFrame
    langsung (tidak ada get_klines / get_price).

    Alur identik:
      1. score_direction        → confidence + arah + ATR
      2. Penyesuaian confidence (inducement / aggressive pullback)
      3. Early exit jika < MIN_CONFIDENCE
      4. calc_discount_entry    → entry zona diskon
      5. analyze_setup          → SL / TP struktural
      6. Validasi TP masih di depan harga sekarang
    """
    try:
        score = score_direction(df_h1, df_m15, df_d1)
        if score is None:
            return None

        direction     = score["direction"]
        current_price = score["price"]
        atr_val       = score["atr"]
        confidence    = score["confidence"]

        # Penyesuaian confidence (identik try22.py full_analyze)
        choch_confirms = (
            (direction == "bull" and score.get("choch_m15", {}).get("bullish_choch")) or
            (direction == "bear" and score.get("choch_m15", {}).get("bearish_choch"))
        )
        if score.get("inducement") and not choch_confirms:
            confidence = max(0, confidence - 8)
        if score.get("pullback_type") == "aggressive" and not choch_confirms:
            confidence = max(0, confidence - 5)

        # Early exit — hemat ~15ms per panggilan yang tidak lolos
        if confidence < MIN_CONFIDENCE:
            return None

        entry, entry_label, invalid_level = calc_discount_entry(
            df_h1, df_m15, direction, current_price, atr_val)

        setup = analyze_setup(
            df_h1, df_m15, direction, entry,
            score=score, invalid_level=invalid_level)
        if setup is None:
            return None

        # TP harus masih di depan harga (bukan sudah kelewatan rally/dump)
        if direction == "bull" and current_price >= setup["tp"]:
            return None
        if direction == "bear" and current_price <= setup["tp"]:
            return None

        return {
            "decision"    : "BUY"  if direction == "bull" else "SELL",
            "direction"   : direction,
            "confidence"  : confidence,
            "price"       : current_price,
            "entry"       : entry,
            "entry_label" : entry_label,
            "sl"          : setup["sl"],
            "tp"          : setup["tp"],
            "rr"          : setup["rr"],
            "rsi"         : score.get("rsi"),
            "d1_bias"     : score.get("d1_bias", "neutral"),
            "struct_h1"   : score.get("struct_h1", "ranging"),
        }
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# LOAD & PRECOMPUTE
# ════════════════════════════════════════════════════════════════════════════

def load_csv(path, months=None):
    """Muat CSV M1 -> {coin: DataFrame M1 DatetimeIndex UTC}.
    Data M1 (crypto_m1_3bulan.csv) TIDAK punya kolom volume -- diisi
    konstan 1.0 (build_df/vol_sma butuh kolom ini tetap ada supaya tidak
    crash; efeknya cuma bonus confluence "volume > vol_sma*1.5" di
    score_direction jadi netral/tidak pernah nyala, bukan bug/crash).
    months: sama seperti versi M15 -- potong N bulan terakhir kalau diisi."""
    print(f"\n📂  Memuat (M1): {path}")
    if path.endswith(".gz"):
        df = pd.read_csv(path, compression="gzip", parse_dates=["Timestamp"])
    else:
        df = pd.read_csv(path, parse_dates=["Timestamp"])
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df = df.sort_values(["coin", "timestamp"])

    data = {}
    for coin, grp in df.groupby("coin"):
        g = (grp[["timestamp", "open", "high", "low", "close", "volume"]]
             .set_index("timestamp").sort_index())
        if g.index.tz is None:
            g.index = g.index.tz_localize("UTC")
        data[coin] = g[~g.index.duplicated(keep="last")]

    if months is not None and data:
        global_max = max(v.index[-1] for v in data.values())
        cutoff = global_max - pd.DateOffset(months=months)
        for coin in list(data.keys()):
            sliced = data[coin].loc[cutoff:]
            if len(sliced) == 0:
                del data[coin]
            else:
                data[coin] = sliced

    dates = [v.index[0] for v in data.values()] + [v.index[-1] for v in data.values()]
    print(f"    -> {len(data)} koin  |  "
          f"{min(dates).date()} s/d {max(dates).date()}"
          + (f"  (dipotong {months} bulan terakhir)" if months is not None else ""))
    return data


def _resample_m15(m1_df):
    """Resample M1 -> M15 (sama dengan try22 get_klines '15m'). Diturunkan
    dari data M1 ASLI -- lebih akurat drpd M15 native yang mungkin sudah
    melalui proses agregasi/approximasi di sumber datanya."""
    return m1_df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last",  "volume": "sum"
    }).dropna()


def _resample_h1(m15_df):
    """Resample M15 -> H1 (sama dengan try22 get_klines '1h')."""
    return m15_df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last",  "volume": "sum"
    }).dropna()


def _resample_d1(m15_df):
    """Resample M15 -> D1 (dipakai utk D1 bias ASLI)."""
    return m15_df.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last",  "volume": "sum"
    }).dropna()


def precompute_indicators(coin_data):
    """
    coin_data di sini adalah RAW M1 per koin (beda dari backtest-3.py yang
    RAW-nya sudah M15). M15/H1/D1 semua DITURUNKAN dari M1 via resample --
    lebih presisi drpd M15 native krn dibangun dari candle 1-menit asli.

    BUG FIX: build_df dipanggil dengan _orig_build_df (bukan _fast) karena
    series ini belum punya kolom ema9 -- perlu dicompute dari awal.

    D1 FIX: d1_ind di-precompute SEKALI dari SELURUH histori (sama seperti
    m15_ind/h1_ind) -- score_direction akan panggil build_df lagi pada
    window 100-bar yang di-slice per scan, tapi karena kolom ema9 sudah
    ada, _fast_build_df (PATCH 2b) ambil jalur cepat + auto-koreksi
    cold-start EMA di baris terakhir -- presisi setara get_klines(1d,100)
    di try22.py tapi tetap cepat.
    """
    print("⚙️   Pre-komputasi indikator (M1->M15->H1->D1)...", end="", flush=True)
    t0 = time.time()
    m15_ind, h1_ind, d1_ind = {}, {}, {}

    for coin, raw_m1 in coin_data.items():
        rm15 = _resample_m15(raw_m1[["open","high","low","close","volume"]])
        bm = _orig_build_df(rm15)
        if bm is None:
            continue
        m15_ind[coin] = bm

        rh1 = _resample_h1(rm15)
        bh1 = _orig_build_df(rh1)
        if bh1 is not None:
            h1_ind[coin] = bh1

        rd1 = _resample_d1(rm15)
        bd1 = _orig_build_df(rd1)
        if bd1 is not None:
            d1_ind[coin] = bd1

    ready = sum(1 for c in coin_data if c in m15_ind and c in h1_ind)
    print(f" selesai ({time.time()-t0:.1f}s)  |  {ready}/{len(coin_data)} koin siap")
    return m15_ind, h1_ind, d1_ind


# ============================================================================
# BACKTEST ENGINE (resolusi M1 -- lihat catatan _run_one_coin)
# ============================================================================

def _run_one_coin(coin, m1_raw, m15_ind_coin, h1_full, d1_full, verbose=False):
    """
    Backtest SATU KOIN, sepenuhnya independen dari koin lain — VERSI M1.

    BEDA dari versi M15 (backtest-3.py): manajemen posisi (cek TP/SL/
    trailing/entry-fill/SL-sebelum-entry) sekarang jalan di RESOLUSI M1,
    bukan M15. Ini menghilangkan hampir semua ambiguitas "TP dan SL
    kena di candle yang sama, mana duluan?" yang terpaksa ditebak pakai
    midpoint-tiebreak di versi M15 (1 candle M15 = 15 candle M1, jadi
    presisi ordering naik ~15x). Konfirmasi SL-sebelum-entry pending
    juga sekarang pakai M1 CLOSE — PERSIS meniru mekanisme live try22.py
    yang memang cek candle M1 (get_klines(sym,"1m",5)), bukan lagi
    aproksimasi pakai M15 close.

    Analisa/sinyal (score_direction dkk) TETAP di timeframe M15/H1/D1
    yang sama seperti sebelumnya — TIDAK berubah, cuma dipicu di momen
    M1 yang tepat (M15 close = menit ke-14/29/44/59, H1 close = menit
    ke-59) supaya presisi waktu pemicu scan juga naik (dulu presisinya
    "per candle M15", sekarang "per menit").

    Posisi (kalau ada) cuma satu slot: None / 'pending' / 'active'.
    """
    c_stats = {
        "total": 0, "tp": 0, "sl": 0, "trail": 0,
        "pending_cancelled": 0, "scans_done": 0, "signals_found": 0,
        "pnl_history": [], "pending_history": [],
    }
    if h1_full is None or m15_ind_coin is None or m1_raw is None:
        return c_stats

    ts_list  = m1_raw.index
    position = None                       # None | dict (status pending/active)

    # Structure-trail tetap di TIMEFRAME M15 (bukan M1 -- swing point di
    # M1 cuma noise, bukan struktur beneran). Array high/low M15 di-
    # precompute sekali dari m15_ind_coin (sudah lengkap kolom OHLC-nya
    # krn build_df tidak membuang kolom asli, cuma nambah indikator).
    _hi_np15 = m15_ind_coin["high"].to_numpy()
    _lo_np15 = m15_ind_coin["low"].to_numpy()
    _m15_index = m15_ind_coin.index

    def _fast_swing_from_np(hi_arr, lo_arr, lb):
        n = len(hi_arr)
        if n < 2 * lb + 1:
            return [], []
        wh = _swv(hi_arr, 2 * lb + 1)
        wl = _swv(lo_arr, 2 * lb + 1)
        sh = (lb + np.where(wh[:, lb] == wh.max(axis=1))[0]).tolist()
        sl = (lb + np.where(wl[:, lb] == wl.min(axis=1))[0]).tolist()
        return sh, sl

    # -- Helper: catat pending yang gagal --------------------------------
    def log_pending_fail(pos, reason, ts_cancel):
        c_stats["pending_cancelled"] += 1
        sig = pos["sig"]
        c_stats["pending_history"].append({
            "coin": coin, "decision": sig["decision"], "reason": reason,
            "confidence": sig["confidence"], "rr_planned": sig["rr"],
            "entry_target": pos["entry"], "tp": pos["tp"], "sl": pos["sl"],
            "signal_ts": pos.get("signal_ts"), "cancel_ts": ts_cancel,
            "d1_bias": sig.get("d1_bias", "?"), "struct_h1": sig.get("struct_h1", "?"),
        })

    # -- Helper: buka posisi aktif ---------------------------------------
    def open_position(sig, actual_entry, ts_open):
        is_buy     = sig["decision"] == "BUY"
        sl_v, tp_v = sig["sl"], sig["tp"]
        if is_buy     and not (sl_v < actual_entry < tp_v): return None
        if not is_buy and not (tp_v < actual_entry < sl_v): return None
        rr_actual = abs(tp_v - actual_entry) / max(abs(actual_entry - sl_v), 1e-12)
        if rr_actual < _try22.MIN_RR:
            return None
        return {
            "sig": sig, "entry": actual_entry, "sl": sl_v, "tp": tp_v,
            "is_buy": is_buy, "locked_r": 0.0, "status": "active",
            "open_ts": ts_open, "pos_usd": POSITION_SIZE_USD,
        }

    # -- Helper: cek TP/SL + trailing stop (R-ladder + structure M15) ---
    def check_candle(pos, hi, lo, m15_pos_idx):
        entry  = pos["entry"]
        is_buy = pos["is_buy"]
        tp_p   = pos["tp"]
        sl_p   = pos["sl"]

        cand_a = None
        risk0  = abs(entry - pos["sig"]["sl"])
        if risk0 > 0:
            proxy = hi if is_buy else lo
            pnl_r = (proxy - entry) / risk0 * (1 if is_buy else -1)
            best_r = 0.0
            for thr, lock in TRAIL_R_LADDER:
                if pnl_r >= thr:
                    best_r = max(best_r, thr * lock)
            if best_r > pos["locked_r"]:
                pos["locked_r"] = best_r
                cand_a = entry + best_r * risk0 * (1 if is_buy else -1)

        cand_b = None
        w_lo = max(0, m15_pos_idx - STRUCT_TRAIL_LOOKBACK)
        hi_w = _hi_np15[w_lo:m15_pos_idx + 1]
        lo_w = _lo_np15[w_lo:m15_pos_idx + 1]
        if len(hi_w) >= STRUCT_TRAIL_LB * 2 + 1:
            sh_w, sl_w = _fast_swing_from_np(hi_w, lo_w, STRUCT_TRAIL_LB)
            if is_buy and sl_w:
                cand_b = float(lo_w[sl_w[-1]]) - entry * STRUCT_TRAIL_BUF_PCT
            elif not is_buy and sh_w:
                cand_b = float(hi_w[sh_w[-1]]) + entry * STRUCT_TRAIL_BUF_PCT

        cands = [c for c in (cand_a, cand_b) if c is not None]
        if cands:
            new_sl = max(cands) if is_buy else min(cands)
            if ((is_buy and sl_p < new_sl < tp_p) or
                    (not is_buy and tp_p < new_sl < sl_p)):
                pos["sl"] = new_sl
                sl_p      = new_sl

        # Resolusi M1: candle jauh lebih sempit drpd M15, jadi kasus
        # "TP & SL kena di candle yang sama" jauh lebih jarang -- tapi
        # midpoint-tiebreak tetap dipertahankan sbg fallback konservatif
        # utk kasus langka candle M1 yang masih memuat keduanya (spike).
        hit_tp = (hi >= tp_p) if is_buy else (lo <= tp_p)
        hit_sl = (lo <= sl_p) if is_buy else (hi >= sl_p)

        if hit_tp and hit_sl:
            mid = (hi + lo) / 2
            if abs(mid - tp_p) <= abs(mid - sl_p):
                return "tp", tp_p
            profit_sl = (sl_p > entry) if is_buy else (sl_p < entry)
            return ("trail" if profit_sl else "sl"), sl_p
        if hit_tp:
            return "tp", tp_p
        if hit_sl:
            profit_sl = (sl_p > entry) if is_buy else (sl_p < entry)
            return ("trail" if profit_sl else "sl"), sl_p
        return None, None

    # -- Helper: tutup posisi & catat PnL --------------------------------
    def close_position(pos, result, exit_price, exit_ts):
        entry   = pos["entry"]
        tp_v    = pos["tp"]
        pos_usd = pos["pos_usd"]
        sign    = 1 if tp_v > entry else -1
        pnl_pct = (exit_price - entry) / entry * sign
        pnl_usd = round(pos_usd * pnl_pct, 6)
        pct     = round(pnl_pct * 100, 3)
        c_stats["total"] += 1
        c_stats[result]   = c_stats.get(result, 0) + 1

        if verbose:
            em = "OK" if result in ("tp", "trail") else "XX"
            s  = "+" if pct >= 0 else ""
            print(f"  {em} [{coin:10s}] {pos['sig']['decision']:4s}  "
                  f"conf={pos['sig']['confidence']:2d}%  "
                  f"entry={entry:.5g} -> {result.upper():5s}@{exit_price:.5g}  "
                  f"{s}{pct:.2f}%  pnl=${pnl_usd:+.4f}")

        c_stats["pnl_history"].append({
            "result": result, "pct": pct, "pnl_usd": pnl_usd,
            "balance_after": None,
            "coin": coin,
            "decision": pos["sig"]["decision"],
            "confidence": pos["sig"]["confidence"],
            "rr_planned": pos["sig"]["rr"],
            "entry": entry, "tp": tp_v, "sl": pos["sig"]["sl"],
            "exit_price": exit_price,
            "open_ts": pos["open_ts"], "exit_ts": exit_ts,
            "d1_bias": pos["sig"].get("d1_bias", "?"),
            "struct_h1": pos["sig"].get("struct_h1", "?"),
        })

    # ====================================================================
    # Cache h1_sl/d1_sl per cutoff (lihat PATCH SPEED 2 versi M15) --
    # sama pentingnya di sini krn scan di mode DENSE dipicu tiap M15
    # close (menit 14/29/44/59), h1_cutoff cuma berubah 1x/jam.
    _h1_cache_key, _h1_cache_val = None, None
    _d1_cache_key, _d1_cache_val = None, None
    m15_pos_idx = -1   # posisi bar M15 TERAKHIR YANG SUDAH CLOSE, utk structure-trail

    PENDING_MAX_DUR = pd.Timedelta(hours=8)

    # LOOP UTAMA -- resolusi M1, timeline koin INI saja
    # ====================================================================
    for ts in ts_list:
        row = m1_raw.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        hi, lo, close_v = float(row["high"]), float(row["low"]), float(row["close"])

        is_m15_close = (ts.minute % 15 == 14)
        if is_m15_close:
            m15_cutoff_now = ts.floor("15min")
            idx = _m15_index.searchsorted(m15_cutoff_now, side="right") - 1
            if idx >= 0:
                m15_pos_idx = int(idx)

        # -- 1. Posisi AKTIF: cek TP/SL/trailing (resolusi M1) ---------
        if position is not None and position["status"] == "active":
            result, exit_price = check_candle(position, hi, lo, m15_pos_idx)
            if result is not None:
                close_position(position, result, exit_price, ts)
                position = None

        # -- 2. Posisi PENDING: cek entry/cancel/expired (resolusi M1) --
        elif position is not None and position["status"] == "pending":
            sig    = position["sig"]
            is_buy = sig["decision"] == "BUY"
            ep, tp_p, sl_p = position["entry"], position["tp"], position["sl"]

            if (ts - position["pending_start_ts"]) >= PENDING_MAX_DUR:
                log_pending_fail(position, "timeout", ts)
                position = None
            else:
                if is_buy:
                    tp_hit_before_entry = hi >= tp_p and lo > ep
                    # Konfirmasi SL-sebelum-entry pakai CLOSE CANDLE M1 --
                    # cocok PERSIS dgn mekanisme live try22.py (cek candle
                    # M1 asli, bukan lagi aproksimasi M15 close spt di
                    # backtest-3.py).
                    sl_hit = close_v <= sl_p
                else:
                    tp_hit_before_entry = lo <= tp_p and hi < ep
                    sl_hit               = close_v >= sl_p

                if tp_hit_before_entry:
                    log_pending_fail(position, "tp_before_entry", ts)
                    position = None
                elif sl_hit:
                    log_pending_fail(position, "sl_before_entry", ts)
                    position = None
                else:
                    hit_entry = (is_buy and lo <= ep * 1.0005) or \
                                (not is_buy and hi >= ep * 0.9995)
                    if hit_entry:
                        newpos = open_position(sig, ep, ts)
                        if newpos is None:
                            log_pending_fail(position, "rr_fail_actual_entry", ts)
                            position = None
                        else:
                            position = newpos

        # -- 3. Kalau slot kosong: scan sinyal baru -- HANYA di M15 close
        # (dense) atau H1 close (hourly), sama seperti versi M15 tapi
        # dipicu berdasar menit M1 sekarang (presisi lebih tinggi) -----
        if position is None:
            scan_now = is_m15_close if DENSE_SCAN else (ts.minute == 59)
            if scan_now:
                if ts.minute == 59:
                    c_stats["scans_done"] += 1

                h1_cutoff = ts.floor("h") if ts.minute == 59 else (ts.floor("h") - pd.Timedelta(hours=1))
                m15_cutoff = ts.floor("15min")
                m15_sl = m15_ind_coin.loc[:m15_cutoff].iloc[-M15_WINDOW:]

                if h1_cutoff == _h1_cache_key:
                    h1_sl = _h1_cache_val
                else:
                    h1_sl = h1_full.loc[:h1_cutoff].iloc[-H1_WINDOW:]
                    _h1_cache_key, _h1_cache_val = h1_cutoff, h1_sl

                if len(m15_sl) >= 60 and len(h1_sl) >= 60:
                    d1_cutoff = ts.normalize() - pd.Timedelta(days=1)
                    if d1_cutoff == _d1_cache_key:
                        d1_sl = _d1_cache_val
                    else:
                        d1_sl = None
                        if d1_full is not None and len(d1_full):
                            d1_sl = d1_full.loc[:d1_cutoff].iloc[-100:]
                            if len(d1_sl) < 65:
                                d1_sl = None
                        _d1_cache_key, _d1_cache_val = d1_cutoff, d1_sl

                    sig = analyze_from_df(h1_sl, m15_sl, d1_sl)
                    if sig is not None:
                        c_stats["signals_found"] += 1
                        is_buy = sig["decision"] == "BUY"
                        ep     = sig["entry"]
                        cp     = sig["price"]

                        at_zone = (is_buy and cp <= ep * 1.002) or \
                                  (not is_buy and cp >= ep * 0.998)

                        if at_zone or sig["entry_label"] == "market":
                            newpos = open_position(sig, cp, ts)
                            if newpos is not None:
                                position = newpos
                        else:
                            position = {
                                "sig": sig, "entry": ep, "sl": sig["sl"], "tp": sig["tp"],
                                "is_buy": is_buy, "locked_r": 0.0, "status": "pending",
                                "open_ts": None, "pos_usd": POSITION_SIZE_USD,
                                "pending_start_ts": ts, "signal_ts": ts,
                            }

    # -- Tutup sisa posisi di akhir data koin ini ------------------------
    if position is not None:
        if position["status"] == "pending":
            log_pending_fail(position, "data_habis", ts_list[-1])
        elif len(m1_raw):
            last_cl = float(m1_raw["close"].iloc[-1])
            sign    = 1 if position["tp"] > position["entry"] else -1
            result  = "tp" if (last_cl - position["entry"]) * sign >= 0 else "sl"
            close_position(position, result, last_cl, ts_list[-1])

    return c_stats


def run_backtest(coin_data, m15_ind, h1_ind, d1_ind, verbose=False):
    """
    Koordinator: backtest SATU PER SATU per koin (independen penuh — hasil
    koin A tidak mempengaruhi koin B sama sekali, sejalan dengan stake
    fixed $POSITION_SIZE_USD/trade, bukan compounding), lalu JUMLAHKAN
    semua hasilnya jadi satu ringkasan. Ini permintaan eksplisit: "backtest
    satu per satu data koin lalu menjumlahkan total tradenya".
    """
    all_coins = sorted(c for c in coin_data if c in m15_ind and c in h1_ind)
    if not all_coins:
        print("❌  Tidak ada koin dengan cukup history.")
        return None

    print(f"\n🪙  Backtest PER KOIN (satu-satu, lalu dijumlah): {len(all_coins)} koin")
    print(f"    MIN_CONF    : {MIN_CONFIDENCE}%  |  Window: {M15_WINDOW} bar M15 / {H1_WINDOW} bar H1")
    print(f"    Stake/trade : ${POSITION_SIZE_USD:.2f} FIXED (bukan compounding — tiap koin independen)")
    print(f"    Resolusi    : M1 (manajemen posisi/entry-fill/SL-confirm presisi menit)")
    print(f"    Scan mode   : {'DENSE (tiap M15 close)' if DENSE_SCAN else 'H1-close saja (tiap jam)'}\n")

    stats = {
        "balance": STARTING_BALANCE,
        "total": 0, "tp": 0, "sl": 0, "trail": 0,
        "pending_cancelled": 0, "scans_done": 0, "signals_found": 0,
        "pnl_history": [], "pending_history": [],
    }
    t_start = time.time()

    for ci, coin in enumerate(all_coins, 1):
        c_stats = _run_one_coin(coin, coin_data[coin], m15_ind[coin],
                                 h1_ind.get(coin), d1_ind.get(coin), verbose)
        stats["total"]              += c_stats["total"]
        stats["tp"]                 += c_stats["tp"]
        stats["sl"]                 += c_stats["sl"]
        stats["trail"]              += c_stats.get("trail", 0)
        stats["pending_cancelled"]  += c_stats["pending_cancelled"]
        stats["scans_done"]         += c_stats["scans_done"]
        stats["signals_found"]      += c_stats["signals_found"]
        stats["pnl_history"].extend(c_stats["pnl_history"])
        stats["pending_history"].extend(c_stats["pending_history"])

        coin_pnl = sum(h["pnl_usd"] for h in c_stats["pnl_history"])
        elapsed  = time.time() - t_start
        print(f"  [{ci:2d}/{len(all_coins)}] {coin:10s}  trade={c_stats['total']:3d}  "
              f"(TP {c_stats['tp']} | Trail {c_stats.get('trail',0)} | SL {c_stats['sl']})  "
              f"pnl=${coin_pnl:+.4f}  [{elapsed:.0f}s elapsed]")

    # Urutkan seluruh trade (gabungan semua koin) secara kronologis, lalu
    # hitung ulang "balance_after" berjalan (fixed stake, penjumlahan murni
    # — BUKAN kurva ekuitas compounding) supaya trade list & equity curve
    # tetap masuk akal dibaca secara waktu meski sumbernya per-koin.
    stats["pnl_history"].sort(
        key=lambda h: h["open_ts"] if h["open_ts"] is not None else h["exit_ts"])
    stats["pending_history"].sort(
        key=lambda h: h["signal_ts"] if h["signal_ts"] is not None else h["cancel_ts"])
    running = STARTING_BALANCE
    for h in stats["pnl_history"]:
        running = round(running + h["pnl_usd"], 6)
        h["balance_after"] = running
    stats["balance"] = running

    elapsed = time.time() - t_start
    print(f"\n  Waktu total: {elapsed:.1f}s  ({elapsed/max(len(all_coins),1):.1f}s/koin)")
    return stats


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def compute_extra_stats(stats):
    """Max Drawdown, Profit Factor, Expectancy — dihitung dari equity
    curve (stats['pnl_history'], sudah terurut kronologis dgn
    balance_after berjalan, fixed-stake/non-compounding)."""
    hist = stats["pnl_history"]
    if not hist:
        return dict(max_dd_usd=0.0, max_dd_pct=0.0, profit_factor=0.0,
                    expectancy_usd=0.0, expectancy_pct=0.0,
                    gross_profit=0.0, gross_loss=0.0)

    balances = [STARTING_BALANCE] + [h["balance_after"] for h in hist]
    peak = balances[0]
    max_dd_usd = max_dd_pct = 0.0
    for b in balances:
        if b > peak:
            peak = b
        dd = peak - b
        dd_pct = (dd / peak * 100) if peak > 0 else 0.0
        max_dd_usd = max(max_dd_usd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)

    gross_profit = sum(h["pnl_usd"] for h in hist if h["pnl_usd"] > 0)
    gross_loss   = abs(sum(h["pnl_usd"] for h in hist if h["pnl_usd"] < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    expectancy_usd = sum(h["pnl_usd"] for h in hist) / len(hist)
    expectancy_pct = sum(h["pct"]     for h in hist) / len(hist)

    return dict(max_dd_usd=max_dd_usd, max_dd_pct=max_dd_pct,
                profit_factor=profit_factor,
                expectancy_usd=expectancy_usd, expectancy_pct=expectancy_pct,
                gross_profit=gross_profit, gross_loss=gross_loss)


def write_equity_curve_png(stats, out_path="equity_curve.png"):
    """Grafik equity curve — opsional, butuh matplotlib. Kalau tidak ada,
    di-skip dengan pesan (data equity curve tetap ada di kolom
    balance_after pada CSV trade list & tabel mingguan di markdown)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ℹ️   matplotlib tidak terinstall — equity_curve.png dilewati "
              "(data equity curve tetap ada di kolom 'balance_after' CSV & "
              "tabel mingguan di file .md). Install dgn: pip install matplotlib")
        return None

    hist = stats["pnl_history"]
    if not hist:
        return None
    xs = [h["open_ts"] or h["exit_ts"] for h in hist]
    ys = [h["balance_after"] for h in hist]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot([xs[0]], [STARTING_BALANCE], marker="o", color="#2563eb")
    ax.plot(xs, ys, color="#2563eb", linewidth=1.2)
    ax.axhline(STARTING_BALANCE, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Equity Curve (stake fixed $%.2f/trade, non-compounding)" % STARTING_BALANCE)
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Saldo kumulatif ($)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"💾  Equity curve: {out_path}")
    return out_path


def print_summary(stats, coins_tested):
    hist  = stats["pnl_history"]
    t     = stats["total"]
    tp    = stats["tp"]
    sl    = stats["sl"]
    trail = stats.get("trail", 0)
    bal   = stats["balance"]
    wins  = tp + trail
    wr    = wins / max(wins + sl, 1) * 100
    pnl   = round(bal - STARTING_BALANCE, 6)
    pct   = round(pnl / STARTING_BALANCE * 100, 2)
    sgn   = "+" if pnl >= 0 else ""

    W = 64
    ex = compute_extra_stats(stats)
    print(); print("═" * W)
    print(f"  📊  HASIL BACKTEST — SMC Signal Bot ({len(coins_tested)} koin)")
    print("═" * W)
    print(f"  Koin diuji        : {', '.join(sorted(coins_tested))}")
    print(f"  Modal awal        : ${STARTING_BALANCE:.2f}")
    print(f"  Saldo akhir       : ${bal:.4f}   ({sgn}{pct:.2f}%)")
    print(f"  P&L absolut       : {sgn}${abs(pnl):.4f}")
    print(f"  Total trade       : {t}   (TP {tp} | Trail {trail} | SL {sl})")
    print(f"  Win rate          : {wr:.1f}%   (TP + Trail dihitung menang)")
    pf_str = "∞ (tidak ada loss)" if ex["profit_factor"] == float("inf") else f"{ex['profit_factor']:.2f}"
    print(f"  Max Drawdown      : ${ex['max_dd_usd']:.4f}  ({ex['max_dd_pct']:.2f}% dari puncak)")
    print(f"  Profit Factor     : {pf_str}   (gross profit ${ex['gross_profit']:.4f} / gross loss ${ex['gross_loss']:.4f})")
    print(f"  Expectancy/trade  : ${ex['expectancy_usd']:+.4f}  ({ex['expectancy_pct']:+.3f}%)")
    print(f"  Pending batal     : {stats['pending_cancelled']}")
    pend_reason = defaultdict(int)
    for p in stats.get("pending_history", []):
        pend_reason[p["reason"]] += 1
    for reason, label in [
        ("sl_before_entry", "SL sebelum entry"),
        ("tp_before_entry", "TP sebelum entry"),
        ("timeout",          "Timeout 8 jam"),
        ("rr_fail_actual_entry", "RR gagal di entry aktual"),
        ("data_habis",       "Data habis"),
    ]:
        n = pend_reason.get(reason, 0)
        if n:
            print(f"    - {label:24s}: {n}")
    print(f"  Scan H1 dilakukan : {stats['scans_done']}")
    print(f"  Sinyal ≥ conf{MIN_CONFIDENCE}%  : {stats['signals_found']}")

    if hist:
        winner = [h for h in hist if h["result"] in ("tp", "trail")]
        loser  = [h for h in hist if h["result"] == "sl"]
        pcts   = [h["pct"] for h in hist]
        avg_w  = sum(h["pct"] for h in winner) / len(winner) if winner else 0
        avg_l  = sum(h["pct"] for h in loser)  / len(loser)  if loser  else 0
        avg_rr = sum(h["rr_planned"] for h in hist) / len(hist)

        print(f"\n  Avg profit (TP/Trail) : +{avg_w:.2f}%")
        print(f"  Avg loss  (SL)        :  {avg_l:.2f}%")
        print(f"  Best trade            : +{max(pcts):.2f}%")
        print(f"  Worst trade           :  {min(pcts):.2f}%")
        print(f"  Avg RR direncanakan   : 1:{avg_rr:.2f}")

        # Equity curve ringkas
        if len(hist) >= 5:
            step = max(1, len(hist) // 10)
            print(f"\n  Equity curve (tiap ~{step} trade):")
            for i in range(0, len(hist), step):
                b   = hist[i]["balance_after"]
                cum = (b - STARTING_BALANCE) / STARTING_BALANCE * 100
                s   = "+" if cum >= 0 else ""
                bar = "█" * int(abs(cum) / 2 + 0.5)
                print(f"    #{i+1:4d}: ${b:.4f}  ({s}{cum:.2f}%)  {bar}")
            b   = hist[-1]["balance_after"]
            cum = (b - STARTING_BALANCE) / STARTING_BALANCE * 100
            s   = "+" if cum >= 0 else ""
            bar = "█" * int(abs(cum) / 2 + 0.5)
            print(f"    #{len(hist):4d}: ${b:.4f}  ({s}{cum:.2f}%)  {bar}  ← akhir")
    print("═" * W)


def print_per_coin(stats):
    hist = stats["pnl_history"]
    if not hist:
        return
    cs = defaultdict(lambda: {"tp": 0, "sl": 0, "trail": 0, "pnl": 0.0})
    for h in hist:
        c = cs[h["coin"]]
        c[h["result"]] = c.get(h["result"], 0) + 1
        c["pnl"] += h["pnl_usd"]
    rows = []
    for coin, c in cs.items():
        n    = c["tp"] + c["sl"] + c["trail"]
        wins = c["tp"] + c["trail"]
        wr   = wins / n * 100 if n else 0
        rows.append((coin, n, c["tp"], c["trail"], c["sl"], wr, c["pnl"]))
    rows.sort(key=lambda x: -x[5])

    print(f"\n  📈  Statistik per Koin:")
    print(f"  {'Koin':12s} {'N':>4} {'TP':>4} {'Trail':>5} {'SL':>4} {'WR%':>6} {'PnL USD':>9}")
    print(f"  {'-'*54}")
    for r in rows:
        s = "+" if r[6] >= 0 else ""
        print(f"  {r[0]:12s} {r[1]:>4d} {r[2]:>4d} {r[3]:>5d} {r[4]:>4d} "
              f"{r[5]:>5.1f}% {s}{r[6]:>8.4f}")
    print(f"  {'-'*54}")


def print_trade_list(stats, n=20):
    hist = stats["pnl_history"]
    if not hist:
        return
    show = hist[-n:]
    print(f"\n  📋  {len(show)} trade terakhir (dari {len(hist)} total):")
    for h in show:
        em  = "✅" if h["result"] in ("tp", "trail") else "❌"
        s   = "+" if h["pct"] >= 0 else ""
        ts  = h["open_ts"].strftime("%d/%m %H:%M") if h.get("open_ts") else "?"
        print(f"  {em} [{h['coin']:10s}] {h['decision']:4s}  "
              f"conf={h['confidence']:2d}%  "
              f"{h['result'].upper():5s} {s}{h['pct']:.2f}%  "
              f"→ ${h['balance_after']:.4f}   ({ts})")


def write_csv(stats, coins_tested, out_path="backtest_result_m1.csv"):
    """
    Ekspor hasil ke CSV:
      Baris 1-N  : ringkasan win rate (header + nilai)
      Baris N+1  : baris kosong pemisah
      Baris N+2+ : daftar tiap trade (header + data)
    """
    import csv

    hist  = stats["pnl_history"]
    pend  = stats["pending_history"]
    t     = stats["total"]
    tp    = stats["tp"]
    sl    = stats["sl"]
    trail = stats.get("trail", 0)
    wins  = tp + trail
    wr    = wins / max(wins + sl, 1) * 100
    bal   = stats["balance"]
    pnl   = round(bal - STARTING_BALANCE, 6)
    pct   = round(pnl / STARTING_BALANCE * 100, 2)

    winner = [h for h in hist if h["result"] in ("tp", "trail")]
    loser  = [h for h in hist if h["result"] == "sl"]
    pcts   = [h["pct"] for h in hist] if hist else [0]
    avg_w  = round(sum(h["pct"] for h in winner) / len(winner), 3) if winner else 0
    avg_l  = round(sum(h["pct"] for h in loser)  / len(loser),  3) if loser  else 0
    avg_rr = round(sum(h["rr_planned"] for h in hist) / len(hist), 3) if hist else 0
    ex     = compute_extra_stats(stats)

    # Breakdown alasan pending gagal (task 3: cari tau penyebab pending
    # gagal — angka per-alasan ini bahan analisanya)
    pend_reason = defaultdict(int)
    for p in pend:
        pend_reason[p["reason"]] += 1
    pend_total = len(pend)

    summary_rows = [
        ["=== RINGKASAN WIN RATE ==="],
        ["Koin diuji",          " | ".join(sorted(coins_tested))],
        ["Modal awal ($)",      STARTING_BALANCE],
        ["Saldo akhir ($)",     round(bal, 4)],
        ["PnL absolut ($)",     round(pnl, 4)],
        ["PnL (%)",             pct],
        ["Total trade",         t],
        ["TP",                  tp],
        ["Trail",               trail],
        ["SL",                  sl],
        ["Win rate (%)",        round(wr, 2)],
        ["Avg profit TP/Trail (%)", avg_w],
        ["Avg loss SL (%)",     avg_l],
        ["Best trade (%)",      round(max(pcts), 3)],
        ["Worst trade (%)",     round(min(pcts), 3)],
        ["Max Drawdown ($)",    round(ex["max_dd_usd"], 4)],
        ["Max Drawdown (%)",    round(ex["max_dd_pct"], 2)],
        ["Profit Factor",       "inf" if ex["profit_factor"] == float("inf") else round(ex["profit_factor"], 3)],
        ["Gross Profit ($)",    round(ex["gross_profit"], 4)],
        ["Gross Loss ($)",      round(ex["gross_loss"], 4)],
        ["Expectancy/trade ($)", round(ex["expectancy_usd"], 4)],
        ["Expectancy/trade (%)", round(ex["expectancy_pct"], 3)],
        ["Avg RR direncanakan", avg_rr],
        ["MIN_CONFIDENCE (%)",  MIN_CONFIDENCE],
        ["MIN_RR",              _try22.MIN_RR],
        ["TRAIL_R_LADDER",      str(TRAIL_R_LADDER)],
        ["STRUCT_TRAIL_LB",     STRUCT_TRAIL_LB],
        ["STRUCT_TRAIL_BUF (%)", STRUCT_TRAIL_BUF_PCT * 100],
        ["Scan mode",           "DENSE (tiap M15)" if DENSE_SCAN else "H1-close saja"],
        ["Pending batal (total)", pend_total],
        ["  - timeout (8 jam)",          pend_reason.get("timeout", 0)],
        ["  - SL sebelum entry",         pend_reason.get("sl_before_entry", 0)],
        ["  - TP sebelum entry (lewat)", pend_reason.get("tp_before_entry", 0)],
        ["  - RR gagal di entry aktual", pend_reason.get("rr_fail_actual_entry", 0)],
        ["  - data habis (akhir dataset)", pend_reason.get("data_habis", 0)],
        ["Scan H1",             stats["scans_done"]],
        ["Sinyal ditemukan",    stats["signals_found"]],
    ]

    trade_header = [
        "no", "coin", "decision", "result",
        "confidence", "d1_bias", "struct_h1",
        "open_ts", "exit_ts",
        "entry", "tp_planned", "sl_planned", "exit_price",
        "rr_planned", "pct_pnl", "pnl_usd", "balance_after",
    ]

    pending_header = [
        "no", "coin", "decision", "reason", "confidence", "rr_planned",
        "d1_bias", "struct_h1", "entry_target", "tp", "sl",
        "signal_ts", "cancel_ts",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in summary_rows:
            w.writerow(row)
        w.writerow([])          # baris kosong pemisah
        w.writerow([])
        w.writerow(["=== DAFTAR TRADE ==="])
        w.writerow(trade_header)
        for i, h in enumerate(hist, 1):
            ots = h["open_ts"].strftime("%Y-%m-%d %H:%M") if h.get("open_ts") else ""
            ets = h["exit_ts"].strftime("%Y-%m-%d %H:%M") if h.get("exit_ts") else ""
            w.writerow([
                i,
                h["coin"],
                h["decision"],
                h["result"].upper(),
                h["confidence"],
                h.get("d1_bias", ""),
                h.get("struct_h1", ""),
                ots,
                ets,
                round(h["entry"],      8),
                round(h["tp"],         8),
                round(h["sl"],         8),
                round(h["exit_price"], 8),
                round(h["rr_planned"], 3),
                round(h["pct"],        4),
                round(h["pnl_usd"],    6),
                round(h["balance_after"], 6),
            ])

        # ── Task 4: trade pending yang GAGAL, sekarang ikut ditulis ──
        w.writerow([])
        w.writerow([])
        w.writerow(["=== DAFTAR PENDING GAGAL ==="])
        w.writerow(pending_header)
        for i, p in enumerate(pend, 1):
            sts = p["signal_ts"].strftime("%Y-%m-%d %H:%M") if p.get("signal_ts") else ""
            cts = p["cancel_ts"].strftime("%Y-%m-%d %H:%M") if p.get("cancel_ts") else ""
            w.writerow([
                i, p["coin"], p["decision"], p["reason"], p["confidence"],
                round(p["rr_planned"], 3), p.get("d1_bias",""), p.get("struct_h1",""),
                round(p["entry_target"], 8), round(p["tp"], 8), round(p["sl"], 8),
                sts, cts,
            ])

    print(f"\n💾  Hasil disimpan ke: {out_path}  ({len(hist)} trade, {len(pend)} pending gagal)")
    write_markdown_report(stats, coins_tested, out_path.replace(".csv", ".md"))
    write_equity_curve_png(stats, out_path.replace(".csv", "_equity.png"))


def write_markdown_report(stats, coins_tested, out_path="backtest_result.md"):
    """
    Task 2: parameter-parameter penting dalam format markdown, sebagai
    bahan analisa (lebih enak dibaca daripada CSV mentah untuk laporan/
    dokumentasi, dan gampang di-diff antar-run kalau parameter diubah).
    """
    hist  = stats["pnl_history"]
    pend  = stats["pending_history"]
    t     = stats["total"]
    tp, sl, trail = stats["tp"], stats["sl"], stats.get("trail", 0)
    wins  = tp + trail
    wr    = wins / max(wins + sl, 1) * 100
    bal   = stats["balance"]
    pnl   = round(bal - STARTING_BALANCE, 6)
    pct   = round(pnl / STARTING_BALANCE * 100, 2)
    winner = [h for h in hist if h["result"] in ("tp", "trail")]
    loser  = [h for h in hist if h["result"] == "sl"]
    avg_w  = round(sum(h["pct"] for h in winner) / len(winner), 3) if winner else 0
    avg_l  = round(sum(h["pct"] for h in loser)  / len(loser),  3) if loser  else 0
    pcts   = [h["pct"] for h in hist] if hist else [0]

    pend_reason = defaultdict(int)
    for p in pend: pend_reason[p["reason"]] += 1
    pend_total = len(pend)
    ex = compute_extra_stats(stats)
    pf_str = "∞ (tidak ada loss)" if ex["profit_factor"] == float("inf") else f"{ex['profit_factor']:.2f}"

    # Breakdown per koin (win rate + pnl)
    cs = defaultdict(lambda: {"tp": 0, "sl": 0, "trail": 0, "pnl": 0.0, "pending_fail": 0})
    for h in hist:
        c = cs[h["coin"]]
        c[h["result"]] = c.get(h["result"], 0) + 1
        c["pnl"] += h["pnl_usd"]
    for p in pend:
        cs[p["coin"]]["pending_fail"] += 1

    lines = []
    A = lines.append
    A(f"# Laporan Backtest — {out_path.replace('.md','')}")
    A("")
    A(f"Koin diuji: {', '.join(sorted(coins_tested))} ({len(coins_tested)} koin)")
    A("")
    A("## Ringkasan")
    A("")
    A("| Metrik | Nilai |")
    A("|---|---|")
    A(f"| Modal awal | ${STARTING_BALANCE:.2f} |")
    A(f"| Saldo akhir | ${bal:.4f} |")
    A(f"| PnL | ${pnl:+.4f} ({pct:+.2f}%) |")
    A(f"| Total trade (closed) | {t} |")
    A(f"| TP / Trail / SL | {tp} / {trail} / {sl} |")
    A(f"| Win rate | {wr:.2f}% |")
    A(f"| Avg profit (TP/Trail) | +{avg_w:.3f}% |")
    A(f"| Avg loss (SL) | {avg_l:.3f}% |")
    A(f"| Best / Worst trade | {max(pcts):+.3f}% / {min(pcts):+.3f}% |")
    A(f"| **Max Drawdown** | ${ex['max_dd_usd']:.4f} ({ex['max_dd_pct']:.2f}% dari puncak equity) |")
    A(f"| **Profit Factor** | {pf_str} (gross profit ${ex['gross_profit']:.4f} / gross loss ${ex['gross_loss']:.4f}) |")
    A(f"| **Expectancy per trade** | ${ex['expectancy_usd']:+.4f} ({ex['expectancy_pct']:+.3f}%) |")
    A(f"| Pending gagal (total) | {pend_total} |")
    A(f"| Rasio pending-gagal : trade-jadi | {pend_total}:{t} "
      f"({pend_total/max(t,1):.2f}× lebih banyak gagal daripada jadi trade)" if t else "")
    A("")
    A("## Parameter yang dipakai run ini")
    A("")
    A("| Parameter | Nilai | Catatan |")
    A("|---|---|---|")
    A(f"| MIN_CONFIDENCE | {MIN_CONFIDENCE}% | ambang sinyal minimum |")
    A(f"| MIN_RR | {_try22.MIN_RR} | RR minimum setup valid |")
    A(f"| TRAIL_R_LADDER | {TRAIL_R_LADDER} | komponen R-multiple trailing (relatif ke risk tiap trade) |")
    A(f"| STRUCT_TRAIL_LB / BUF | {STRUCT_TRAIL_LB} / {STRUCT_TRAIL_BUF_PCT*100:.2f}% | komponen structure (swing-point) trailing |")
    A(f"| PENDING_MAX_CANDLES | {PENDING_MAX_CANDLES} (= {PENDING_MAX_CANDLES/4:.0f} jam) | batas tunggu entry pending |")
    A(f"| M15_WINDOW / H1_WINDOW | {M15_WINDOW} / {H1_WINDOW} | jumlah candle per window analisa |")
    A(f"| Scan mode | {'DENSE (tiap candle M15)' if DENSE_SCAN else 'H1-close saja (tiap jam)'} | |")
    A(f"| Stake per trade | ${POSITION_SIZE_USD:.2f} FIXED | tidak compounding, per-koin independen |")
    A("")
    A("## Kenapa pending gagal (breakdown alasan)")
    A("")
    A("| Alasan | Jumlah | % dari total pending gagal |")
    A("|---|---|---|")
    for reason, label in [
        ("sl_before_entry", "SL tersentuh sebelum entry (candle close confirm)"),
        ("tp_before_entry", "TP sudah kena duluan (peluang lewat, tidak sempat entry)"),
        ("timeout", "Timeout 8 jam (harga tak pernah sampai zona entry)"),
        ("rr_fail_actual_entry", "RR gagal di harga entry aktual"),
        ("data_habis", "Data historis habis saat masih pending"),
    ]:
        n = pend_reason.get(reason, 0)
        p = n/pend_total*100 if pend_total else 0
        A(f"| {label} | {n} | {p:.1f}% |")
    A("")
    A("**Catatan diagnosa:** SL selama fase pending = level invalidasi ZONA itu sendiri "
      "(tepi jauh OB/FVG + noise buffer kecil, lihat `analyze_setup()` di try22.py) — "
      "jaraknya ke entry sering cuma `risk_floor` (0.3-0.8×ATR), jadi wajar kalau porsi "
      "\"SL sebelum entry\" dominan: itu bukan bug, itu konsekuensi structural dari entry "
      "di tepi zona sempit. Fix yang sudah diterapkan: SL selama pending sekarang butuh "
      "KONFIRMASI CANDLE CLOSE (bukan cuma wick sesaat) — meniru proteksi anti-whipsaw yang "
      "sebelumnya cuma ada di posisi aktif, sekarang juga berlaku di fase pending.")
    A("")
    A("## Equity Curve")
    A("")
    png_name = out_path.replace(".md", "_equity.png")
    A(f"![Equity Curve]({os.path.basename(png_name)})")
    A("")
    A("(Kalau gambar di atas tidak muncul — butuh `pip install matplotlib` lalu "
      "jalankan ulang. Data mentahnya tetap ada di kolom `balance_after` pada "
      "CSV trade list, tinggal di-plot pakai tool apa saja.)")
    A("")
    A("Snapshot saldo per minggu (dari trade pertama tiap minggu):")
    A("")
    A("| Minggu mulai | Saldo | PnL sejak awal |")
    A("|---|---|---|")
    if hist:
        seen_weeks = set()
        for h in hist:
            ts = h.get("open_ts") or h.get("exit_ts")
            if ts is None: continue
            wk = ts.strftime("%Y-W%U")
            if wk in seen_weeks: continue
            seen_weeks.add(wk)
            b = h["balance_after"]
            p = b - STARTING_BALANCE
            A(f"| {ts.strftime('%Y-%m-%d')} | ${b:.4f} | {p:+.4f} ({p/STARTING_BALANCE*100:+.1f}%) |")
        last = hist[-1]
        A(f"| **{(last.get('exit_ts') or last.get('open_ts')).strftime('%Y-%m-%d')} (akhir)** "
          f"| **${last['balance_after']:.4f}** | **{pnl:+.4f} ({pct:+.2f}%)** |")
    A("")
    A("## Breakdown per Koin")
    A("")
    A("| Koin | Trade | TP | Trail | SL | WR% | PnL $ | Pending Gagal |")
    A("|---|---|---|---|---|---|---|---|")
    for coin in sorted(cs, key=lambda c: -(cs[c]["tp"]+cs[c]["trail"]+cs[c]["sl"])):
        c = cs[coin]
        n = c["tp"] + c["trail"] + c["sl"]
        if n == 0 and c["pending_fail"] == 0: continue
        wr_c = (c["tp"]+c["trail"])/n*100 if n else 0
        A(f"| {coin} | {n} | {c['tp']} | {c['trail']} | {c['sl']} | "
          f"{wr_c:.1f}% | {c['pnl']:+.4f} | {c['pending_fail']} |")
    A("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"💾  Laporan markdown: {out_path}")


def print_timing_estimate(n_coins):
    """CATATAN: fungsi estimasi presisi sebelumnya (versi ms_per_scan hardcoded)
    sudah 2x meleset jauh dari realita (pertama terlalu optimis, lalu setelah
    dikoreksi malah terlalu pesimis) — karena angkanya cuma tebakan tanpa
    profiling nyata (backtest ini tidak dijalankan di sisi pengembang).
    Daripada terus menebak dan menyesatkan, sekarang cuma kasih patokan
    kasar + saran: lihat waktu elapsed koin PERTAMA yang muncul di log,
    itu jauh lebih akurat drpd tebakan apa pun di sini."""
    print(f"\n  ⏱️  Estimasi waktu: TIDAK ditampilkan dgn angka pasti — perkiraan")
    print(f"      sebelumnya (di kedua arah) terbukti meleset jauh dari realita.")
    print(f"      Kecepatan aktual tergantung jumlah trade/pending per koin,")
    print(f"      jadi variasinya besar. Cara paling akurat: tunggu koin PERTAMA")
    print(f"      selesai (baris '[1/{n_coins}] ... elapsed' di bawah), lalu kalikan")
    print(f"      dgn {n_coins} koin buat estimasi kasar totalnya.")
    print(f"      (--hourly buat matikan DENSE_SCAN kalau butuh jauh lebih cepat)")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="SMC Backtest offline RESOLUSI M1 — import langsung dari try22.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python backtest-4.py --coins SOL
  python backtest-4.py --coins SOL XRP --verbose
  python backtest-4.py --coins SOL --conf 35
  python backtest-4.py                         (semua 5 koin, seluruh 3 bulan data M1)
        """)
    ap.add_argument("--csv",     default=CSV_DEFAULT,
                    help=f"Path CSV M1 (default: {CSV_DEFAULT})")
    ap.add_argument("--months",  type=float, default=0,
                    help="Pakai N bulan TERAKHIR data saja (default: 0 = semua, "
                         "krn data M1 memang sudah persis 3 bulan).")
    ap.add_argument("--coins",   nargs="*", metavar="COIN",
                    help="Filter koin: SOL  atau  SOL XRP NEAR")
    ap.add_argument("--conf",    type=int, default=MIN_CONFIDENCE,
                    metavar="N", help=f"Confidence minimum (default: {MIN_CONFIDENCE})")
    ap.add_argument("--verbose", action="store_true",
                    help="Tampilkan setiap trade saat terjadi")
    ap.add_argument("--hourly",  action="store_true",
                    help="Scan cuma 1x/jam (H1 close) — mode LAMA, lebih cepat "
                         "tapi sinyal jauh lebih sedikit. Default sekarang DENSE "
                         "(scan tiap candle M15) supaya frekuensi trade lebih "
                         "dekat ke live try22.py — tetap pakai H1 terakhir yang "
                         "sudah close (tidak look-ahead), tapi TETAP TIDAK identik "
                         "dgn live krn live bisa entry di candle yg belum final "
                         "dan scan tiap detik (data CSV cuma resolusi M15).")
    ap.add_argument("--bench",   action="store_true",
                    help="Tampilkan benchmark kecepatan scan lalu keluar")
    args = ap.parse_args()

    # Override globals dari argumen
    globals()["MIN_CONFIDENCE"] = args.conf
    if args.hourly:
        globals()["DENSE_SCAN"] = False

    # ── Benchmark mode ────────────────────────────────────────────────────
    if args.bench:
        _run_benchmark()
        return

    # ── Cek CSV ───────────────────────────────────────────────────────────
    csv_path = args.csv
    if not os.path.exists(csv_path):
        # Coba juga di folder script
        alt = os.path.join(_SCRIPT_DIR, args.csv)
        if os.path.exists(alt):
            csv_path = alt
        else:
            print(f"❌  File tidak ditemukan: {args.csv}")
            sys.exit(1)

    months_arg = None if args.months == 0 else args.months
    coin_data = load_csv(csv_path, months=months_arg)

    if args.coins:
        wanted    = {c.upper() for c in args.coins}
        coin_data = {k: v for k, v in coin_data.items() if k in wanted}
        if not coin_data:
            avail = sorted(load_csv(csv_path).keys())
            print(f"❌  Koin tidak ditemukan: {wanted}")
            print(f"    Tersedia: {avail}")
            sys.exit(1)
        print(f"    Filter    : {sorted(coin_data.keys())}")

    print_timing_estimate(len(coin_data))

    m15_ind, h1_ind, d1_ind = precompute_indicators(coin_data)
    stats = run_backtest(coin_data, m15_ind, h1_ind, d1_ind, verbose=args.verbose)
    if not stats:
        sys.exit(1)

    print_summary(stats, set(coin_data.keys()))
    print_per_coin(stats)
    print_trade_list(stats, n=20)
    write_csv(stats, set(coin_data.keys()))
    print(f"\n✅  Selesai.\n")


def _run_benchmark():
    """Ukur kecepatan scan aktual setelah patch."""
    csv_path = CSV_DEFAULT
    if not os.path.exists(csv_path):
        csv_path = os.path.join(_SCRIPT_DIR, CSV_DEFAULT)
    if not os.path.exists(csv_path):
        print("❌  CSV tidak ditemukan untuk benchmark")
        return

    if csv_path.endswith(".gz"):
        df = pd.read_csv(csv_path, compression="gzip", parse_dates=["Timestamp"])
    else:
        df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    sample_coin = sorted(df["coin"].unique())[0]
    trx = df[df["coin"] == sample_coin].sort_values("timestamp").set_index("timestamp")
    if trx.index.tz is None:
        trx.index = trx.index.tz_localize("UTC")
    trx = trx[~trx.index.duplicated()]

    raw_m1 = trx[["open","high","low","close","volume"]]
    m15_raw  = _resample_m15(raw_m1)
    m15_full = _orig_build_df(m15_raw)
    h1_raw   = _resample_h1(m15_raw)
    h1_full  = _orig_build_df(h1_raw)

    idx = int(len(m15_full) * 0.7)
    ts  = m15_full.index[idx]

    print(f"\n⏱️  Benchmark scan @ {ts.date()} (70% data {sample_coin}, M15 derived from M1)")
    for W in [80, 120, 150, 250]:
        m15_s = m15_full.loc[:ts].iloc[-W:]
        h1_s  = h1_full.loc[:ts].iloc[-W:]
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            score_direction(h1_s, m15_s)
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times) * 1000
        print(f"  window={W:3d}: {avg:5.1f}ms/scan")

    print(f"\n  Menggunakan window={M15_WINDOW} untuk scan.")
    print(f"  (Estimasi total waktu tidak dihitung di sini — tergantung jumlah")
    print(f"   candle M1 & posisi aktif/pending per koin, lihat catatan di")
    print(f"   print_timing_estimate().)\n")


# ============================================================
# FUNGSI UNTUK DIPANGGIL DARI OPTIMIZER
# ============================================================
def run_backtest_for_params(params, csv_path=CSV_DEFAULT, verbose=False):
    """
    Jalankan backtest dengan parameter tertentu.
    params: dict dengan key:
        - min_conf: int
        - trail_ladder: list of tuples (threshold, lock)
        - struct_trail_lb: int
        - struct_trail_buf_pct: float
        - dense_scan: bool (opsional)
    Return: dict berisi metrik:
        - total_trades, win_rate, profit_factor, pnl_pct, max_drawdown_pct, etc.
    """
    # Override global variables
    global MIN_CONFIDENCE, TRAIL_R_LADDER, STRUCT_TRAIL_LB, STRUCT_TRAIL_BUF_PCT, DENSE_SCAN
    MIN_CONFIDENCE = params.get('min_conf', 45)
    TRAIL_R_LADDER = params.get('trail_ladder', TRAIL_R_LADDER)
    STRUCT_TRAIL_LB = params.get('struct_trail_lb', 2)
    STRUCT_TRAIL_BUF_PCT = params.get('struct_trail_buf_pct', 0.0015)
    DENSE_SCAN = params.get('dense_scan', True)

    # Load data (gunakan CSV yang sama)
    coin_data = load_csv(csv_path)
    m15_ind, h1_ind, d1_ind = precompute_indicators(coin_data)
    stats = run_backtest(coin_data, m15_ind, h1_ind, d1_ind, verbose=verbose)

    # Ekstrak metrik
    total = stats['total']
    tp = stats['tp']
    sl = stats['sl']
    trail = stats.get('trail', 0)
    wins = tp + trail
    win_rate = (wins / max(wins + sl, 1)) * 100 if (wins + sl) > 0 else 0
    pnl_pct = (stats['balance'] - STARTING_BALANCE) / STARTING_BALANCE * 100
    # profit factor
    gross_profit = sum(h['pnl_usd'] for h in stats['pnl_history'] if h['pnl_usd'] > 0)
    gross_loss = abs(sum(h['pnl_usd'] for h in stats['pnl_history'] if h['pnl_usd'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown (dari compute_extra_stats)
    ex = compute_extra_stats(stats)
    max_dd_pct = ex['max_dd_pct']

    return {
        'total_trades': total,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'pnl_pct': pnl_pct,
        'max_drawdown_pct': max_dd_pct,
        'stats': stats  # untuk detail jika perlu
    }


if __name__ == "__main__":
    main()
