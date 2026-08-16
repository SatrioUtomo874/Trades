"""
strategy_logic.py — OTAK v5 "LIQUIDITY-FIRST"
=================================================================================
Rewrite total (permintaan eksplisit: strategy_logic lama dibuang, dibangun ulang
dari nol). Sumber ilmu: seluruh 49 transkrip video trading di combined.txt
(market structure, order block, FVG, liquidity sweep & jenis-jenisnya,
inducement, ChoCH/BOS, CISD, OTE & Fibonacci, external vs internal liquidity,
candle range theory, breaker block / unicorn model, kualitas zona supply &
demand, trend strength, top-down multi-timeframe). Konsep dipelajari dan
ditulis ulang dalam bentuk aturan kuantitatif, BUKAN disalin verbatim.

Urutan proses tiap koin — SESUAI PERMINTAAN, dieksekusi persis dalam urutan
ini di dalam full_analyze():
  1) BIAS  — arah HTF (D1 → H1) + kekuatan tren (trend_strength).
  2) ENTRY — kumpulkan POI (Order Block / FVG / Breaker Block / Equal High-
     Low) searah bias, WAJIB mempertimbangkan Liquidity Sweep & Inducement,
     lalu pilih kandidat terbaik & tentukan harga entry presisi.
  3) SL    — ditempatkan di luar titik invalidasi SEBENARNYA (bukan di luar
     zona semata): kalau ada liquidity sweep yang terjadi di dekat POI, SL
     diletakkan di luar UJUNG WICK sweep tersebut + buffer kecil, supaya SL
     baru tersentuh kalau sweep itu berulang LEBIH DALAM — indikasi arah
     memang salah, bukan sekadar false break sebelumnya.
  4) TP    — dibangun dari pool level "draw on liquidity" (external liquidity
     H1 diprioritaskan atas internal M15, sesuai video liquidity eksternal/
     internal), lalu dipilih mengikuti aturan RR: target <2R TIDAK langsung
     ditolak — cari target lebih jauh di pool sampai RR≥MIN_RR, dan kalau
     target valid >MAX_RR maka RR dipotong (dicap), bukan ditolak.
  5) TRAIL — TIDAK dihitung di sini karena main.py memakai swing_pts() +
     STRUCT_TRAIL_* miliknya SENDIRI secara langsung tiap kali harga
     bergerak (lihat monitor_position() main.py). Modul ini hanya
     menyediakan swing_pts() yang presisi dan TRAIL_R_LADDER = [] supaya
     SATU-SATUNYA mekanisme trailing yang aktif adalah pergeseran SL
     mengikuti struktur M15 — bukan profit-lock berbasis R — persis seperti
     yang diminta ("Trail bukan pengaman profit, tapi update SL mengikuti
     struktur market M15, sedikit di bawah/atas swing").
  6) CONFIDENCE — dihitung TERAKHIR, dari seluruh bukti yang sudah
     dikumpulkan di langkah 1–4 (bias, structure, liquidity, kualitas POI,
     timing RSI+Volume). Semua bobot dijumlah = tepat 100 secara konstruksi
     (bukan hasil kalibrasi ulang setelah observasi) sehingga skor yang
     tampil selalu berarti sesuatu yang bisa ditelusuri ke aturan
     spesifik — tidak ada komponen skor "abu-abu"/tanpa dasar.

Filosofi kuantitas sinyal: setiap koin yang datanya cukup HARUS menghasilkan
sebuah kandidat (bahkan kalau lemah) — bukan biner "ada setup / tidak ada
setup". Kualitas rendah direpresentasikan lewat confidence rendah, difilter
oleh MIN_CONFIDENCE di main.py (/confidence_min), bukan oleh penolakan diam-
diam di sini. full_analyze() hanya mengembalikan None kalau data benar-benar
tidak cukup untuk membentuk geometri Entry/SL/TP yang valid sama sekali
(termasuk tidak ada target TP nyata yang mencapai MIN_RR — lihat select_tp).

Kompatibilitas dengan main.py (WAJIB dipertahankan bila memodifikasi file ini):
  - full_analyze(df_h1, df_m15, df_d1=None, symbol=None) -> dict | None
      dict wajib berisi minimal:
      symbol, decision("BUY"/"SELL"), original_dir("bull"/"bear"),
      confidence(int 0-100), price, entry, sl, tp, rr, rsi, struct_h1(str),
      tp_sl_reason(str); opsional tapi dipakai main.py bila ada: atr,
      entry_label, d1_bias, choch_m15, choch_h1, failed_retest.
  - swing_pts(df, lb) -> (swing_high_idx_list, swing_low_idx_list)
      DIPANGGIL LANGSUNG oleh main.py di monitor_position() untuk trailing —
      signature & makna return TIDAK BOLEH berubah.
  - validate_and_adjust_geometry(entry, sl, tp, current_price, atr, direction)
      -> dict{entry,sl,tp,rr,adjusted} | None
      DIPANGGIL LANGSUNG oleh main.py setelah order terisi (real trade) untuk
      mengoreksi geometri & mendeteksi Liquidity Sweep pasca-fill.
  - Konstanta yang dibaca LANGSUNG oleh main.py:
      MIN_RR, MAX_RR, TRAIL_R_LADDER, STRUCT_TRAIL_LB, STRUCT_TRAIL_BUF_PCT,
      STRUCT_TRAIL_LOOKBACK, FIB_EXT_1, FIB_EXT_2.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# =================================================================================
# KONFIGURASI — dibaca main.py langsung, atau dipakai internal di bawah
# =================================================================================

# ── Risk:Reward (permintaan #1) ──────────────────────────────────────────────
MIN_RR = 2.0     # RR di bawah ini TIDAK auto-tolak — lihat select_tp()
MAX_RR = 4.0     # RR di atas ini DIPOTONG ke sini, bukan ditolak — select_tp()

# ── Trail (permintaan #4) — KOSONGKAN SENGAJA ────────────────────────────────
# main.py memakai kandidat "paling protektif" antara TRAIL_R_LADDER (profit
# ladder berbasis R) dan structure (swing_pts M15). Trail yang diminta murni
# structural — bukan pengaman profit — jadi ladder-nya dikosongkan supaya
# SATU-SATUNYA sumber trailing adalah pergeseran SL mengikuti swing M15.
TRAIL_R_LADDER: List[Tuple[float, float]] = []
STRUCT_TRAIL_LB       = 3        # lookback swing_pts saat trailing (candle M15)
STRUCT_TRAIL_BUF_PCT  = 0.0025   # 0.25% "sedikit di bawah/atas swing" (contoh user)
STRUCT_TRAIL_LOOKBACK = 60       # jumlah candle M15 yang diambil untuk trailing

# ── Fibonacci extension (dipakai pool TP) ────────────────────────────────────
FIB_EXT_1 = 0.272   # 127.2%
FIB_EXT_2 = 0.618   # 161.8%

# ── Sinkron dengan gate eksekusi main.py (_validate_signal_before_entry) ────
# main.py menolak entry kalau |price_now - entry| > atr * 1.50. Nilai ini
# HARUS sama supaya strategy tidak pernah mengembalikan kandidat yang pasti
# ditolak main.py setelah scan selesai (harga bergerak sedikit di antaranya).
MAIN_ENTRY_MAX_ATR = 1.50

# ── Toleransi Liquidity Sweep pasca-fill (permintaan #3) ────────────────────
# Dipakai validate_and_adjust_geometry(). Angka ini SENGAJA disamakan dengan
# comment main.py sendiri di _open_position_real ("Liquidity Sweep depth
# ≤3×ATR") — sebelumnya angka itu hanya ada di komentar main.py tapi TIDAK
# PERNAH benar-benar diimplementasikan di strategy_logic (bug nyata: SL yang
# tersentuh dangkal selalu dianggap reversal penuh & auto-out, padahal
# menurut desain main.py seharusnya bisa diselamatkan sebagai sweep).
SWEEP_TOLERANCE_ATR = 3.0
SWEEP_RELOCATE_BUFFER_ATR = 0.5

# ── Struktur & swing ──────────────────────────────────────────────────────
SWING_LB_H1   = 5
SWING_LB_M15  = 5
SWING_LB_MINOR = 2     # swing minor untuk inducement / equal-level clustering

# ── Zona (OB/FVG/Breaker) ───────────────────────────────────────────────────
ZONE_LOOKBACK_H1   = 90
ZONE_LOOKBACK_M15  = 70
MIN_DISPLACEMENT_ATR = 0.25   # body minimum candle "impulsif" (video 34: ciri #2)
FVG_LOOKBACK = 60
EQUAL_LEVEL_TOL = 0.0025      # toleransi cluster equal high/low (~0.25%)

# ── Liquidity sweep & inducement ────────────────────────────────────────────
INDUCEMENT_LOOKBACK  = 40
CISD_LOOKBACK = 10

# ── Candle Range Theory (baru — video "Candle Range Theory") ────────────────
CRT_LOOKBACK = 4

# ── RSI & Volume timing — "sinyal jangan telat" (permintaan eksplisit) ──────
RSI_TIMING_SLOPE   = 1.5     # perubahan RSI minimum supaya dianggap "baru berbalik"
RSI_LATE_CEILING   = 72.0    # BUY: RSI di atas ini + slope turun = sudah telat
VOL_EXPANSION_MIN  = 1.15    # candle konfirmasi harus ≥15% di atas volume SMA20
VOL_CONTRACTION_MAX = 0.85   # pullback ideal: volume MENGECIL sebelum entry

# ── Entry-location anti-chase ────────────────────────────────────────────────
ENTRY_LOCATION_LOOKBACK = 16
ENTRY_CHASE_HIGH   = 0.82
ENTRY_PREFERRED_BUY  = 0.55
ENTRY_PREFERRED_SELL = 0.45

# ── SL — batas risk terhadap ATR/entry (permintaan #3) ──────────────────────
SL_BUFFER_ATR = 0.5      # "sedikit di luar" level invalidasi
SL_MIN_RISK_ATR = 1.0
SL_MAX_RISK_ATR = 2.0
SL_MAX_RISK_PCT_OF_ENTRY = 0.035

# ── Macro bias (BTC H1) — konteks tambahan opsional, bukan veto keras ───────
MACRO_ALIGN_BONUS  = 6
MACRO_AGAINST_MULT = 0.75


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# =================================================================================
# SECTION 1 — INDIKATOR & PERSIAPAN DATA
# =================================================================================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    # Kalau loss rolling = 0 murni (candle naik terus tanpa turun sekalipun
    # dalam window, misal pump kuat) itu RSI seharusnya mendekati 100, BUKAN
    # NaN. replace(0, np.nan) pada versi naif membuat baris itu ter-drop oleh
    # dropna() di build_df() — pakai epsilon kecil supaya tetap jadi angka
    # valid (RSI≈100) alih-alih diam-diam menghilangkan candle dari analisa.
    rs = gain / loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)


def atr_fn(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _closed_candles(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Buang candle yang masih berjalan (belum close) supaya structure/EMA/RSI
    tidak "repaint" di tengah scan yang sama."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    idx = out.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    boundary = pd.Timestamp.now(tz="UTC").floor(f"{interval_minutes}min")
    if idx[-1] < boundary:
        return out
    return out.loc[idx < boundary].copy()


def build_df(df: pd.DataFrame, interval_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Lengkapi OHLCV dengan EMA/RSI/ATR/volume SMA & STD.

    vol_sma/vol_std dipakai volume_confirmation() — di versi lama kolom ini
    dihitung tapi TIDAK PERNAH benar-benar dipakai di manapun; di sini
    benar-benar dipakai untuk gate timing entry (permintaan "RSI dan Volume").
    """
    if df is None or len(df) < 60:
        return None
    df = df.copy()
    if interval_minutes is not None:
        df = _closed_candles(df, interval_minutes)
    if len(df) < 60:
        return None
    df["ema9"]   = ema(df["close"], 9)
    df["ema21"]  = ema(df["close"], 21)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200) if len(df) >= 200 else ema(df["close"], 50)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr_fn(df)
    df["vol_sma"] = df["volume"].rolling(20).mean()
    df["vol_std"] = df["volume"].rolling(20).std()
    return df.dropna(subset=["ema9", "ema21", "ema50", "rsi", "atr"])


def swing_pts(df: pd.DataFrame, lb: int = 5):
    """Swing high & swing low (fractal lb-bar). DIPAKAI LANGSUNG main.py
    untuk trailing — signature & return TIDAK BOLEH berubah."""
    sh, sl = [], []
    if df is None or len(df) < lb * 2 + 1:
        return sh, sl
    high = df["high"].values
    low = df["low"].values
    n = len(high)
    for i in range(lb, n - lb):
        wh = high[i - lb: i + lb + 1]
        wl = low[i - lb: i + lb + 1]
        if high[i] == wh.max():
            sh.append(i)
        if low[i] == wl.min():
            sl.append(i)
    return sh, sl


def volume_confirmation(df: pd.DataFrame, idx: int = -1) -> float:
    """Rasio volume candle ke rata-rata SMA20-nya. >1 = ekspansi (partisipasi
    order flow nyata, sesuai video CISD: "harus ada volume ikut meningkat")."""
    if df is None or "vol_sma" not in df.columns or len(df) < abs(idx) + 1:
        return 1.0
    v = _safe_float(df["volume"].iloc[idx], 0.0)
    avg = _safe_float(df["vol_sma"].iloc[idx], 0.0)
    if avg <= 0:
        return 1.0
    return v / avg


# =================================================================================
# SECTION 2 — MARKET STRUCTURE (HH/HL/LH/LL, BOS, ChoCH)
# =================================================================================
# Referensi: "Market Structure Smart Money", "This is the basic foundation...
# market structure", "The Secret of Pullbacks... Complete Guide to Market
# Structure", "ARE YOU SURE THE METHOD OF MARKING KEY LEVEL IS CORRECT?"
# (kriteria: pakai wick untuk liquidity/level, body untuk arah candle).

def market_structure(df: pd.DataFrame, sh: list, sl: list) -> str:
    """HH+HL berturut2 = bullish, LH+LL = bearish, selain itu ranging."""
    if len(sh) < 2 or len(sl) < 2:
        return "ranging"
    hh = df["high"].iloc[sh[-1]] > df["high"].iloc[sh[-2]]
    hl = df["low"].iloc[sl[-1]] > df["low"].iloc[sl[-2]]
    lh = df["high"].iloc[sh[-1]] < df["high"].iloc[sh[-2]]
    ll = df["low"].iloc[sl[-1]] < df["low"].iloc[sl[-2]]
    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"


def detect_bos(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """Break of Structure — penerusan tren: candle terakhir CLOSE menembus
    swing SEARAH tren yang sedang berjalan (bukan swing terhadap arah)."""
    out = {"bullish_bos": False, "bearish_bos": False}
    if df is None or df.empty:
        return out
    last_close = float(df["close"].iloc[-1])
    if len(sh) >= 2:
        out["bullish_bos"] = last_close > float(df["high"].iloc[sh[-2]])
    if len(sl) >= 2:
        out["bearish_bos"] = last_close < float(df["low"].iloc[sl[-2]])
    return out


def detect_choch(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """Change of Character — pembalikan struktur: candle CLOSE menembus
    swing point TERDEKAT yang berlawanan arah dengan struktur saat ini
    (tanda awal bahwa struktur dominan mulai gagal)."""
    out = {"bullish_choch": False, "bearish_choch": False}
    if df is None or df.empty:
        return out
    struct = market_structure(df, sh, sl)
    last_close = float(df["close"].iloc[-1])
    if struct == "bearish" and sh:
        out["bullish_choch"] = last_close > float(df["high"].iloc[sh[-1]])
    elif struct == "bullish" and sl:
        out["bearish_choch"] = last_close < float(df["low"].iloc[sl[-1]])
    elif struct == "ranging":
        # Di dalam range, ChoCH = breakout tegas dari sisi yang baru saja gagal.
        if sh and len(sh) >= 1:
            out["bullish_choch"] = last_close > float(df["high"].iloc[sh[-1]])
        if sl and len(sl) >= 1:
            out["bearish_choch"] = last_close < float(df["low"].iloc[sl[-1]])
    return out


# =================================================================================
# SECTION 3 — LIQUIDITY (external/internal, equal high-low, sweep, inducement)
# =================================================================================
# Referensi: "Liquidity Is the Language of the Market", "3 Types of Liquidity
# Targeted by Smart Money" (equal high/low, trendline liquidity, internal
# range liquidity), "The Secret of Price Movement: External vs Internal
# Liquidity" (BSL/SSL di luar swing = external; FVG/OB di dalam range =
# internal — market menyapu external lalu rebalance ke internal),
# "How the Market Traps Traders with Inducement" & "...[SNIPER ENTRY]".

def detect_equal_levels(df: pd.DataFrame, kind: str = "high",
                        lb: int = 100, tol: float = EQUAL_LEVEL_TOL) -> list:
    """Equal Highs / Equal Lows — cluster titik harga yang disentuh berulang
    (liquidity pool tipe #1 dari "3 Types of Liquidity")."""
    if df is None or df.empty:
        return []
    sub = df.iloc[-lb:]
    vals = sub["high"] if kind == "high" else sub["low"]
    clusters, visited = [], set()
    for i in range(len(vals)):
        if i in visited:
            continue
        group = [float(vals.iloc[i])]
        for j in range(i + 1, len(vals)):
            if j in visited:
                continue
            if abs(vals.iloc[i] - vals.iloc[j]) / max(abs(float(vals.iloc[i])), 1e-10) < tol:
                group.append(float(vals.iloc[j]))
                visited.add(j)
        if len(group) >= 2:
            clusters.append(sum(group) / len(group))
    return sorted(clusters)


def external_liquidity(df: pd.DataFrame, sh: list, sl: list,
                       direction: str) -> dict:
    """Level liquidity EKSTERNAL terdekat searah draw (di atas swing high
    signifikan untuk bull/buyside, di bawah swing low untuk bear/sellside).
    Ini yang dituju harga SETELAH internal liquidity di dalam range
    diselesaikan — dipakai sebagai magnet TP, bukan entry."""
    out = {"level": None, "swept": False}
    if direction == "bull" and sh:
        level = float(df["high"].iloc[sh[-1]])
        last_high = float(df["high"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        out["level"] = level
        out["swept"] = bool(last_high > level and last_close < level)
    elif direction == "bear" and sl:
        level = float(df["low"].iloc[sl[-1]])
        last_low = float(df["low"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        out["level"] = level
        out["swept"] = bool(last_low < level and last_close > level)
    return out


def detect_liquidity_sweep(df: pd.DataFrame, sh: list, sl: list,
                           direction: str, atr: float) -> dict:
    """Liquidity Sweep: wick menembus swing point, tapi CLOSE kembali ke
    sisi yang benar. Depth dilaporkan dalam satuan ATR (dipakai SL & CISD).

    direction="bull" → sweep di bawah swing LOW (sellside liquidity diambil,
    lalu reversal ke atas). direction="bear" → sweep di atas swing HIGH.
    """
    result = {"type": "none", "level": None, "wick_extreme": None, "depth_atr": 0.0}
    atr = max(_safe_float(atr), 1e-10)
    if direction == "bull" and sl:
        level = float(df["low"].iloc[sl[-1]])
        last_low = float(df["low"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_low < level and last_close > level:
            result = {
                "type": "sweep", "level": level, "wick_extreme": last_low,
                "depth_atr": round((level - last_low) / atr, 3),
            }
    elif direction == "bear" and sh:
        level = float(df["high"].iloc[sh[-1]])
        last_high = float(df["high"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_high > level and last_close < level:
            result = {
                "type": "sweep", "level": level, "wick_extreme": last_high,
                "depth_atr": round((last_high - level) / atr, 3),
            }
    return result


def detect_inducement(df: pd.DataFrame, direction: str,
                      lb: int = INDUCEMENT_LOOKBACK) -> dict:
    """Inducement: gerakan minor (bukan zona statis) yang menyapu liquidity
    KECIL sebelum harga benar-benar bereaksi di POI utama. Kalau minor swing
    ini SUDAH disapu (wick tembus, close kembali) sebelum harga sampai POI
    kita, itu bonus konfirmasi — bukan syarat wajib (supaya sinyal tidak
    berkurang untuk koin yang tidak menunjukkan pola ini secara eksplisit)."""
    out = {"found": False, "level": None, "swept": False}
    if df is None or len(df) < lb + SWING_LB_MINOR * 2 + 1:
        return out
    sub = df.iloc[-lb:].reset_index(drop=True)
    sh_m, sl_m = swing_pts(sub, lb=SWING_LB_MINOR)
    try:
        if direction == "bull" and sl_m:
            idx = sl_m[-1]
            if idx >= len(sub) - 2:
                return out
            level = float(sub["low"].iloc[idx])
            after = sub.iloc[idx + 1:]
            swept = bool((after["low"] < level).any() and float(after["close"].iloc[-1]) > level)
            out = {"found": True, "level": level, "swept": swept}
        elif direction == "bear" and sh_m:
            idx = sh_m[-1]
            if idx >= len(sub) - 2:
                return out
            level = float(sub["high"].iloc[idx])
            after = sub.iloc[idx + 1:]
            swept = bool((after["high"] > level).any() and float(after["close"].iloc[-1]) < level)
            out = {"found": True, "level": level, "swept": swept}
    except Exception:
        return {"found": False, "level": None, "swept": False}
    return out


# =================================================================================
# SECTION 4 — POI: ORDER BLOCK, FAIR VALUE GAP, BREAKER BLOCK
# =================================================================================
# Referensi: "How to Choose the Best Order Block When All Zones Look Valid"
# (filter: searah tren, disertai FVG+BOS, Fibonacci diskon ≤0.618/premium
# ≥0.618, fresh/unmitigated, turun time-frame untuk konfirmasi structure
# shift), "3 Characteristics of High Quality Supply & Demand Zones" (fresh +
# strong displacement dengan FVG + diikuti BOS/ChoCH), "How to determine FVG
# with high probability", "3 simple tips for finding high probability order
# blocks", "Teknik Scalping... Unicorn Model + breaker block, FVG".

def fib_position(price: float, lo: float, hi: float) -> float:
    """Posisi harga dalam range [lo, hi], 0=lo, 1=hi."""
    rng = hi - lo
    if rng <= 0:
        return 0.5
    return max(0.0, min(1.0, (price - lo) / rng))


def in_ote(price: float, lo: float, hi: float, direction: str) -> bool:
    """OTE (Optimal Trade Entry) 61.8%-78.6% retracement dari impulse leg."""
    if hi <= lo:
        return False
    r = fib_position(price, lo, hi)
    return (0.214 <= r <= 0.382) if direction == "bull" else (0.618 <= r <= 0.786)


def is_zone_fresh(df: pd.DataFrame, top: float, bot: float, formed_idx: int,
                  direction: Optional[str] = None) -> bool:
    """Zona masih fresh = body candle setelahnya belum menembus close lewat
    sisi luar zona (video 34, ciri #1: "belum pernah disentuh/dimitigasi")."""
    if formed_idx is None or formed_idx + 2 >= len(df):
        return True
    sub = df.iloc[formed_idx + 2:]
    if sub.empty:
        return True
    if direction == "bull":
        return not bool((sub["close"] < bot).any())
    if direction == "bear":
        return not bool((sub["close"] > top).any())
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)


def detect_fvg(df: pd.DataFrame, direction: str, lb: int = FVG_LOOKBACK) -> list:
    """Fair Value Gap — celah 3-candle akibat ketidakseimbangan buyer/seller
    yang bergerak terlalu cepat. Hanya return yang masih fresh."""
    if df is None or len(df) < 5:
        return []
    sub = df.iloc[-lb:]
    base = len(df) - len(sub)
    out = []
    for i in range(len(sub) - 2):
        c0, c2 = sub.iloc[i], sub.iloc[i + 2]
        gap = None
        if direction == "bull" and float(c2["low"]) > float(c0["high"]):
            gap = {"top": float(c2["low"]), "bot": float(c0["high"])}
        elif direction == "bear" and float(c2["high"]) < float(c0["low"]):
            gap = {"top": float(c0["low"]), "bot": float(c2["high"])}
        if gap:
            gap["mid"] = (gap["top"] + gap["bot"]) / 2
            gap["idx"] = base + i + 2
            gap["is_fresh"] = is_zone_fresh(df, gap["top"], gap["bot"], gap["idx"], direction)
            out.append(gap)
    fresh = [g for g in out if g["is_fresh"]]
    return fresh[-4:]


def _scan_ob_raw(df: pd.DataFrame, direction: str,
                 sh: Optional[list] = None, sl: Optional[list] = None,
                 lb: int = ZONE_LOOKBACK_M15) -> list:
    """Pemindaian mentah pola Order Block (trigger candle + impulse candle
    searah), TANPA filter freshness. Dipakai bersama oleh detect_order_block
    (yang MEWAJIBKAN fresh — POI aktif) dan detect_breaker_block (yang justru
    BUTUH zona yang sudah tidak fresh/sudah ditembus — itu esensi breaker)."""
    if df is None or len(df) < 10:
        return []
    is_demand = direction == "bull"
    sub = df.iloc[-lb:]
    base = len(df) - len(sub)
    avg_body = float((sub["close"] - sub["open"]).abs().mean()) or 1e-8

    fib_sh = float(df["high"].iloc[sh[-1]]) if (sh and len(sh) > 0) else None
    fib_sl = float(df["low"].iloc[sl[-1]]) if (sl and len(sl) > 0) else None

    has_bos_global = False
    if sh and sl:
        if is_demand and len(sh) >= 2:
            has_bos_global = float(df["high"].iloc[-1]) > float(df["high"].iloc[sh[-2]])
        elif not is_demand and len(sl) >= 2:
            has_bos_global = float(df["low"].iloc[-1]) < float(df["low"].iloc[sl[-2]])

    zones = []
    for i in range(1, len(sub) - 2):
        c, nx = sub.iloc[i], sub.iloc[i + 1]
        if is_demand:
            if not (c["close"] < c["open"] and nx["close"] > nx["open"]):
                continue
        else:
            if not (c["close"] > c["open"] and nx["close"] < nx["open"]):
                continue

        impulse_body = abs(float(nx["close"]) - float(nx["open"]))
        if impulse_body < avg_body * 1.2:
            continue

        ob_top = float(max(c["open"], c["close"]))
        ob_bot = float(min(c["open"], c["close"]))
        df_idx = base + i

        quality = 0.0
        if impulse_body >= avg_body * 1.5:
            quality += 1
        if impulse_body >= avg_body * 2.5:
            quality += 1

        has_fvg = False
        if i + 2 < len(sub):
            c2 = sub.iloc[i + 2]
            if is_demand and float(c2["low"]) > float(c["high"]):
                has_fvg = True
            elif not is_demand and float(c2["high"]) < float(c["low"]):
                has_fvg = True
        if has_fvg:
            quality += 1.5

        if has_bos_global:
            quality += 1

        fib_r = 0.5
        if fib_sh is not None and fib_sl is not None and fib_sh > fib_sl:
            ob_mid = (ob_top + ob_bot) / 2
            fib_r = fib_position(ob_mid, fib_sl, fib_sh)
            if is_demand and fib_r <= 0.618:
                quality += 1.5
            elif not is_demand and fib_r >= 0.382:
                quality += 1.5

        zones.append({
            "top": ob_top, "bot": ob_bot, "idx": df_idx,
            "quality": round(quality, 2), "has_fvg": has_fvg,
            "fib_r": round(fib_r, 3),
        })
    return zones


def detect_order_block(df: pd.DataFrame, direction: str,
                       sh: Optional[list] = None, sl: Optional[list] = None,
                       lb: int = ZONE_LOOKBACK_M15) -> list:
    """Order Block berkualitas tinggi. Kriteria (video "How to Choose the
    Best Order Block" + "3 Characteristics of High Quality S&D Zones"):
      1. Candle terakhir berlawanan warna sebelum candle impulsif searah
         trade (trigger candle), body impulsif ≥1.2x rata2 body lokal.
      2. Zona masih fresh (belum dimitigasi close) — INI yang membedakan
         dari _scan_ob_raw: OB aktif WAJIB fresh, kalau tidak itu sudah
         bukan POI yang valid (justru jadi kandidat breaker, lihat
         detect_breaker_block).
      3. Diprioritaskan bila diikuti FVG (ciri displacement kuat, ciri #2).
      4. Diprioritaskan bila berada di sisi diskon (≤0.618 fib) untuk
         BUY / premium (≥0.618 fib) untuk SELL (filter Fibonacci OB).
      5. Bonus bila terjadi BOS global searah (ciri #3: mengubah struktur).
    Return terurut kualitas tertinggi dulu, field 'idx' = index candle
    trigger (dipakai detect_breaker_block).
    """
    zones = _scan_ob_raw(df, direction, sh=sh, sl=sl, lb=lb)
    fresh = [z for z in zones
             if is_zone_fresh(df, z["top"], z["bot"], z["idx"], direction=direction)]
    fresh.sort(key=lambda z: -z["quality"])
    return fresh[:4]


def detect_breaker_block(df: pd.DataFrame, direction: str,
                         sh: Optional[list] = None, sl: Optional[list] = None,
                         lb: int = ZONE_LOOKBACK_M15) -> list:
    """Breaker Block ("Unicorn Model"): Order Block LAWAN arah yang gagal —
    ditembus penuh (body close melewati sisi jauhnya) — lalu BELUM di-reclaim
    lagi sejak itu. Zona bekas OB gagal ini sering jadi POI kuat saat
    diretest dari sisi berlawanan, terutama bila overlap dengan FVG
    ("Unicorn Model + breaker block, FVG").

    PENTING: pakai _scan_ob_raw (BUKAN detect_order_block) sebagai sumber,
    karena breaker justru butuh OB yang SUDAH tidak fresh (sudah ditembus) —
    kalau memakai detect_order_block yang mewajibkan fresh, fungsi ini tidak
    akan pernah menemukan apa pun (bug yang sempat ditemukan saat testing).
    """
    if df is None or len(df) < 10:
        return []
    opp = "bear" if direction == "bull" else "bull"
    try:
        candidates = _scan_ob_raw(df, opp, sh=sh, sl=sl, lb=lb)
    except Exception:
        return []
    out = []
    for z in candidates:
        top, bot, idx = z["top"], z["bot"], z.get("idx")
        if idx is None or idx + 2 >= len(df):
            continue
        after = df.iloc[idx + 1:]
        if direction == "bull":
            broke_mask = after["close"] > top
        else:
            broke_mask = after["close"] < bot
        if not bool(broke_mask.any()):
            continue
        break_pos = int(np.argmax(broke_mask.values))
        post_break = after.iloc[break_pos + 1:]
        if post_break.empty:
            continue
        if direction == "bull":
            reclaimed = bool((post_break["close"] < bot).any())
        else:
            reclaimed = bool((post_break["close"] > top).any())
        if reclaimed:
            continue
        out.append({
            "top": top, "bot": bot, "idx": idx,
            "quality": z["quality"], "has_fvg": z.get("has_fvg", False),
            "label": "breaker",
        })
    return out[-2:]


def zones_overlap(a_top: float, a_bot: float, b_top: float, b_bot: float) -> bool:
    """True kalau dua rentang harga [bot, top] saling overlap — dipakai cek
    konfluensi POI M15 dengan zona H1 ("How to Choose the Best Order Block")."""
    lo = max(min(a_bot, a_top), min(b_bot, b_top))
    hi = min(max(a_bot, a_top), max(b_bot, b_top))
    return lo <= hi


# =================================================================================
# SECTION 5 — KONFIRMASI: CISD, CANDLE RANGE THEORY, RSI DIVERGENCE
# =================================================================================
# Referensi: "CISD Secret: The Earliest Reversal Signal in ICT Strategy"
# (tandai OPEN candle pertama dari runtun searah sebagai garis CISD; valid
# kalau (a) ada liquidity sweep sebelumnya, (b) reaksi terjadi di zona HTF
# penting, (c) didukung volume/FVG), "\"How to Entry Precisely Using Candle
# Range Theory\"" & "Candle Range Theory Strategy: 3 Simple Steps" (range
# candle acuan = liquidity; probe candle yang wick tembus tapi GAGAL close
# di luar range = sinyal menuju sisi berlawanan range).

def detect_cisd(df: pd.DataFrame, sh: list, sl: list, atr: float,
                lb: int = CISD_LOOKBACK) -> dict:
    """Change In State of Delivery.

    1. Cari runtun candle searah terakhir (>=3 candle) sebelum candle
       penutup berlawanan arah muncul.
    2. Tandai OPEN candle PERTAMA runtun itu sebagai garis CISD.
    3. Valid kalau candle terakhir CLOSE menembus garis itu ke arah
       berlawanan (delivery benar-benar berpindah tangan) — bukan cuma
       retracement biasa.
    4. Bonus validitas (dilaporkan, bukan syarat mutlak, supaya sinyal tetap
       ada untuk koin tanpa pola ini): didahului liquidity sweep, dan candle
       konfirmasi didukung volume ekspansi.
    """
    result = {"bullish_cisd": False, "bearish_cisd": False, "level": None,
              "preceded_by_sweep": False}
    if df is None or len(df) < lb + 1:
        return result
    sub = df.iloc[-lb:]
    opens = sub["open"].values
    closes = sub["close"].values
    n = len(closes)
    if n < 4:
        return result
    last_bull = closes[-1] > opens[-1]
    last_bear = closes[-1] < opens[-1]

    if last_bull:
        bear_run = 0
        for j in range(n - 2, -1, -1):
            if closes[j] < opens[j]:
                bear_run += 1
            else:
                break
        if bear_run >= 3:
            first_idx = n - 1 - bear_run
            if first_idx >= 0:
                level = float(opens[first_idx])
                if closes[-1] > level:
                    result["bullish_cisd"] = True
                    result["level"] = level
                    sweep = detect_liquidity_sweep(df, sh, sl, "bull", atr)
                    result["preceded_by_sweep"] = sweep.get("type") == "sweep"
    elif last_bear:
        bull_run = 0
        for j in range(n - 2, -1, -1):
            if closes[j] > opens[j]:
                bull_run += 1
            else:
                break
        if bull_run >= 3:
            first_idx = n - 1 - bull_run
            if first_idx >= 0:
                level = float(opens[first_idx])
                if closes[-1] < level:
                    result["bearish_cisd"] = True
                    result["level"] = level
                    sweep = detect_liquidity_sweep(df, sh, sl, "bear", atr)
                    result["preceded_by_sweep"] = sweep.get("type") == "sweep"
    return result


def detect_candle_range_rejection(df: pd.DataFrame, direction: str,
                                  lookback: int = CRT_LOOKBACK) -> dict:
    """Candle Range Theory: range candle acuan (high/low beberapa candle
    terakhir SEBELUM candle probe) adalah level liquidity. Kalau candle
    probe (candle M15 terakhir yang sudah closed) menembus salah satu sisi
    dengan wick tapi GAGAL close di luar, maka probabilitas mengarah ke sisi
    berlawanan dari range itu — sinyal ini SEGAR (pakai candle terakhir),
    cocok untuk timing entry yang tidak telat."""
    out = {"triggered": False, "range_high": None, "range_low": None}
    if df is None or len(df) < lookback + 2:
        return out
    window = df.iloc[-(lookback + 1):]
    ref = window.iloc[:-1]
    probe = window.iloc[-1]
    ref_high = float(ref["high"].max())
    ref_low = float(ref["low"].min())
    out["range_high"] = ref_high
    out["range_low"] = ref_low
    if direction == "bull":
        out["triggered"] = bool(
            float(probe["low"]) < ref_low
            and float(probe["close"]) > ref_low
            and float(probe["close"]) > float(probe["open"])
        )
    else:
        out["triggered"] = bool(
            float(probe["high"]) > ref_high
            and float(probe["close"]) < ref_high
            and float(probe["close"]) < float(probe["open"])
        )
    return out


def detect_rsi_divergence(df: pd.DataFrame, direction: str, lb: int = 30) -> dict:
    """RSI Divergence — harga membuat extreme baru, RSI tidak konfirmasi."""
    result = {"bull_div": False, "bear_div": False, "strong": False}
    if df is None or len(df) < lb + 1 or "rsi" not in df.columns:
        return result
    sub = df.iloc[-lb:]
    price = sub["close"].values
    rsi_v = sub["rsi"].values
    n = len(price)
    w = 3

    lows = [i for i in range(w, n - w) if price[i] == min(price[max(0, i - w): i + w + 1])]
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if price[i2] < price[i1] and rsi_v[i2] > rsi_v[i1]:
            result["bull_div"] = True
            if rsi_v[i2] < 35:
                result["strong"] = True

    highs = [i for i in range(w, n - w) if price[i] == max(price[max(0, i - w): i + w + 1])]
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if price[i2] > price[i1] and rsi_v[i2] < rsi_v[i1]:
            result["bear_div"] = True
            if rsi_v[i2] > 65:
                result["strong"] = True

    if direction == "bull" and not result["bull_div"]:
        return {"bull_div": False, "bear_div": False, "strong": False}
    if direction == "bear" and not result["bear_div"]:
        return {"bull_div": False, "bear_div": False, "strong": False}
    return result


def detect_failed_retest(df: pd.DataFrame, sh: list, sl: list, atr: float) -> dict:
    """Failed retest sederhana — dipertahankan untuk kompatibilitas pesan
    Telegram main.py (fmt_signal_msg membaca sig['failed_retest'])."""
    result = {"failed_retest_sell": False, "failed_retest_buy": False}
    if df is None or len(df) < 3 or not sh or not sl:
        return result
    atr = max(_safe_float(atr), 1e-10)
    L, P = df.iloc[-1], df.iloc[-2]
    if len(sh) >= 2:
        res = float(df["high"].iloc[sh[-2]])
        if P["high"] >= res - atr * 0.5 and L["close"] < res - atr * 0.3 and L["close"] < L["open"]:
            result["failed_retest_sell"] = True
    if len(sl) >= 2:
        sup = float(df["low"].iloc[sl[-2]])
        if P["low"] <= sup + atr * 0.5 and L["close"] > sup + atr * 0.3 and L["close"] > L["open"]:
            result["failed_retest_buy"] = True
    return result


# =================================================================================
# SECTION 6 — KEKUATAN TREN
# =================================================================================
# Referensi: "The Secret to Measuring Trend Strength That Traders Rarely
# Discuss" — kualitas entry ditentukan kekuatan tren pendukungnya, bukan
# cuma bentuk setup. Diukur dari efisiensi arah (net move / total move,
# menangkap dimensi WAKTU yang disebut video ini), kekuatan impuls
# (body/ATR), kualitas pullback, kualitas struktur (HH/HL vs LH/LL), dan
# momentum RSI (level + slope, penalti kalau divergence).

def trend_strength(df: pd.DataFrame, direction: str, lb: int = 60) -> dict:
    if df is None or len(df) < 35:
        return {"strength": 0.5, "regime": "unknown"}
    d = df.tail(max(35, lb)).copy()
    close = d["close"].astype(float)
    atr_s = d["atr"] if "atr" in d.columns else atr_fn(d)
    atr_now = _safe_float(atr_s.iloc[-1])
    if atr_now <= 0:
        atr_now = max(float(close.iloc[-1]) * 0.001, 1e-10)

    sign = 1.0 if direction == "bull" else -1.0
    net = sign * (float(close.iloc[-1]) - float(close.iloc[-21]))
    path = float(close.iloc[-20:].diff().abs().sum())
    efficiency = _clip01(net / max(path, 1e-10))

    bodies = (d["close"] - d["open"]).abs().tail(8)
    atr_tail = atr_s.tail(8).replace(0, np.nan)
    body_atr = (bodies / atr_tail).replace([np.inf, -np.inf], np.nan).dropna()
    impulse = _clip01((float(body_atr.mean()) - 0.15) / 0.85) if not body_atr.empty else 0.5

    sh, sl = swing_pts(d, lb=3)
    pullback_quality = 0.5
    structure_quality = 0.5
    if direction == "bull" and sl and sh:
        last_low = float(d["low"].iloc[sl[-1]])
        last_high = float(d["high"].iloc[sh[-1]])
        prev_low = float(d["low"].iloc[sl[-2]]) if len(sl) >= 2 else last_low
        prev_high = float(d["high"].iloc[sh[-2]]) if len(sh) >= 2 else last_high
        impulse_leg = max(last_high - prev_low, atr_now)
        pullback = max(0.0, last_high - last_low)
        pullback_quality = 1.0 - _clip01(pullback / max(impulse_leg, atr_now))
        hh = last_high > prev_high if len(sh) >= 2 else False
        hl = last_low > prev_low if len(sl) >= 2 else False
        structure_quality = 0.75 if (hh and hl) else (0.45 if (hh or hl) else 0.25)
    elif direction == "bear" and sh and sl:
        last_high = float(d["high"].iloc[sh[-1]])
        last_low = float(d["low"].iloc[sl[-1]])
        prev_high = float(d["high"].iloc[sh[-2]]) if len(sh) >= 2 else last_high
        prev_low = float(d["low"].iloc[sl[-2]]) if len(sl) >= 2 else last_low
        impulse_leg = max(prev_high - last_low, atr_now)
        pullback = max(0.0, last_high - last_low)
        pullback_quality = 1.0 - _clip01(pullback / max(impulse_leg, atr_now))
        lh = last_high < prev_high if len(sh) >= 2 else False
        ll = last_low < prev_low if len(sl) >= 2 else False
        structure_quality = 0.75 if (lh and ll) else (0.45 if (lh or ll) else 0.25)

    ema9 = float(d["ema9"].iloc[-1]) if "ema9" in d else float(ema(close, 9).iloc[-1])
    ema21 = float(d["ema21"].iloc[-1]) if "ema21" in d else float(ema(close, 21).iloc[-1])
    ema50 = float(d["ema50"].iloc[-1]) if "ema50" in d else float(ema(close, 50).iloc[-1])
    ema21_prev = float(d["ema21"].iloc[-6]) if "ema21" in d and len(d) >= 6 else ema21
    slope = sign * (ema21 - ema21_prev) / max(atr_now * 5.0, 1e-10)
    ema_slope = _clip01(0.5 + slope)
    ema_align = _clip01(0.5 + sign * (ema9 - ema21) / max(atr_now * 2.0, 1e-10)
                        + sign * (ema21 - ema50) / max(atr_now * 4.0, 1e-10))
    ema_quality = 0.55 * ema_slope + 0.45 * ema_align

    rv = d["rsi"].astype(float) if "rsi" in d else rsi(close)
    r_now = float(rv.iloc[-1])
    r_prev = float(rv.iloc[-4]) if len(rv) >= 4 else r_now
    r_slope = sign * (r_now - r_prev)
    r_level = _clip01((sign * (r_now - 50.0) + 10.0) / 35.0)
    r_momentum = _clip01(0.5 + r_slope / 12.0)

    div = detect_rsi_divergence(d, direction, lb=min(30, len(d) - 1))
    divergence_penalty = 0.18 if div.get("strong") else (
        0.10 if (div.get("bull_div") if direction == "bull" else div.get("bear_div")) else 0.0)

    raw = (0.20 * efficiency + 0.16 * impulse + 0.15 * pullback_quality
           + 0.14 * structure_quality + 0.15 * ema_quality + 0.20 * r_momentum)
    strength = _clip01(raw - divergence_penalty)
    if r_level < 0.35:
        strength *= 0.90

    if strength >= 0.88:
        regime = "very_strong"
    elif strength >= 0.78:
        regime = "strong"
    elif strength >= 0.68:
        regime = "healthy"
    elif strength >= 0.48:
        regime = "transition"
    else:
        regime = "weak"

    return {
        "strength": round(strength, 4), "regime": regime,
        "efficiency": round(efficiency, 4), "impulse": round(impulse, 4),
        "pullback_quality": round(pullback_quality, 4),
        "structure_quality": round(structure_quality, 4),
        "ema_quality": round(ema_quality, 4),
        "rsi_momentum": round(r_momentum, 4), "atr": atr_now,
    }


# =================================================================================
# SECTION 7 — BIAS TOP-DOWN (D1 -> H1) & MACRO BTC
# =================================================================================
# Referensi: "Are You Sure Your Top-Down Analysis Is Correct?" — arah HTF
# adalah GATE, bukan bonus yang bisa dikalahkan skor lokal M15.

def htf_bias(df_h1: pd.DataFrame, df_d1: Optional[pd.DataFrame],
            h1_built: pd.DataFrame) -> dict:
    """Tentukan bias D1 + H1. Return dict berisi arah masing2 timeframe dan
    'combined' — d1 kalau selaras dengan h1, konflik kalau berlawanan,
    salah satu kalau yang lain tidak tersedia, neutral kalau keduanya tidak
    jelas."""
    L1 = h1_built.iloc[-1]
    sh1, sl1 = swing_pts(h1_built, lb=SWING_LB_H1)
    struct_h1 = market_structure(h1_built, sh1, sl1)
    ema_h1_bull = L1["ema9"] > L1["ema21"] > L1["ema50"]
    ema_h1_bear = L1["ema9"] < L1["ema21"] < L1["ema50"]
    h1_bias = "bullish" if struct_h1 == "bullish" else ("bearish" if struct_h1 == "bearish" else "neutral")

    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            d1 = build_df(df_d1, interval_minutes=1440)
        elif df_h1 is not None and isinstance(df_h1.index, pd.DatetimeIndex):
            d1 = build_df(
                df_h1.resample("1D").agg(
                    {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
                ).dropna()
            )
        else:
            d1 = None
        if d1 is not None and len(d1) >= 10:
            LD = d1.iloc[-1]
            shd, sld = swing_pts(d1, lb=3)
            sd1 = market_structure(d1, shd, sld)
            bull_d1 = sd1 == "bullish" or (LD["ema9"] > LD["ema21"] > LD["ema50"])
            bear_d1 = sd1 == "bearish" or (LD["ema9"] < LD["ema21"] < LD["ema50"])
            d1_bias = "bullish" if bull_d1 else ("bearish" if bear_d1 else "neutral")
    except Exception:
        pass

    if d1_bias in ("bullish", "bearish") and h1_bias in ("bullish", "bearish"):
        combined = d1_bias if d1_bias == h1_bias else "conflict"
    elif d1_bias in ("bullish", "bearish"):
        combined = d1_bias
    elif h1_bias in ("bullish", "bearish"):
        combined = h1_bias
    else:
        combined = "neutral"

    return {
        "struct_h1": struct_h1, "h1_bias": h1_bias, "d1_bias": d1_bias,
        "combined": combined, "ema_h1_bull": bool(ema_h1_bull),
        "ema_h1_bear": bool(ema_h1_bear), "sh1": sh1, "sl1": sl1,
    }


def macro_bias(df_btc_h1: Optional[pd.DataFrame]) -> str:
    """Bias BTC H1 sebagai proksi kondisi market keseluruhan — konteks
    tambahan MODERAT (bonus/penalti kecil), bukan veto keras. Return
    "unknown" kalau data tidak dikasih; caller HARUS treat unknown = no-op."""
    if df_btc_h1 is None or len(df_btc_h1) < 60:
        return "unknown"
    try:
        btc = build_df(df_btc_h1, interval_minutes=60)
        if btc is None or len(btc) < 60:
            return "unknown"
        L = btc.iloc[-1]
        shb, slb = swing_pts(btc, lb=5)
        struct_btc = market_structure(btc, shb, slb)
        ema_bull = L["ema9"] > L["ema21"] > L["ema50"]
        ema_bear = L["ema9"] < L["ema21"] < L["ema50"]
        if struct_btc == "bullish" or ema_bull:
            return "bullish"
        if struct_btc == "bearish" or ema_bear:
            return "bearish"
        return "ranging"
    except Exception:
        return "unknown"


# =================================================================================
# SECTION 8 — ENTRY TIMING: RSI + VOLUME ("sinyal jangan telat")
# =================================================================================
# Permintaan eksplisit: gunakan RSI DAN Volume supaya sinyal tidak telat.
# RSI slope menandakan momentum baru mulai berbalik (bukan sudah exhausted
# di ujung ekstrem — itu artinya kita sudah telat, bukan di awal gerakan).
# Volume dipakai sesuai video CISD ("harus ada volume ikut meningkat") &
# semangat Wyckoff (effort vs result): candle konfirmasi harus didukung
# partisipasi order flow di atas rata-rata, bukan noise tipis.

def entry_timing(m15: pd.DataFrame, direction: str) -> dict:
    if m15 is None or len(m15) < 6:
        return {"rsi": 50.0, "rsi_slope": 0.0, "rsi_ok": False,
                "vol_ratio": 1.0, "vol_ok": False, "late": False}

    rsi_now = float(m15["rsi"].iloc[-1])
    rsi_prev = float(m15["rsi"].iloc[-4]) if len(m15) >= 4 else rsi_now
    slope = rsi_now - rsi_prev
    vol_ratio = volume_confirmation(m15, -1)

    if direction == "bull":
        rsi_ok = slope >= RSI_TIMING_SLOPE and rsi_now <= RSI_LATE_CEILING
        late = rsi_now > RSI_LATE_CEILING and slope <= 0
    else:
        rsi_ok = slope <= -RSI_TIMING_SLOPE and rsi_now >= (100 - RSI_LATE_CEILING)
        late = rsi_now < (100 - RSI_LATE_CEILING) and slope >= 0

    vol_ok = vol_ratio >= VOL_EXPANSION_MIN
    return {
        "rsi": round(rsi_now, 2), "rsi_slope": round(slope, 2), "rsi_ok": bool(rsi_ok),
        "vol_ratio": round(vol_ratio, 2), "vol_ok": bool(vol_ok), "late": bool(late),
    }


# =================================================================================
# SECTION 9 — ENTRY LOCATION (anti-chase)
# =================================================================================

def entry_location(m15: pd.DataFrame, direction: str, entry: float, atr: float) -> dict:
    """Nilai lokasi entry relatif range M15 lokal. Arah HTF boleh benar,
    tapi entry di ujung range yang salah (chasing) tetap entry yang buruk —
    ini penalti/hard-block, bukan pembatalan arah."""
    if m15 is None or len(m15) < 8:
        return {"location_score": 50, "location_state": "unknown",
                "range_position": 0.5, "hard_block": False}

    n = min(ENTRY_LOCATION_LOOKBACK, len(m15))
    sub = m15.iloc[-n:]
    rh = float(sub["high"].max())
    rl = float(sub["low"].min())
    width = max(rh - rl, max(_safe_float(atr), 1e-10))
    pos = max(0.0, min(1.0, (float(entry) - rl) / width))

    score = 50
    if direction == "bull":
        if pos <= 0.35:
            score += 15
        elif pos <= ENTRY_PREFERRED_BUY:
            score += 10
        elif pos <= 0.70:
            score += 0
        elif pos <= ENTRY_CHASE_HIGH:
            score -= 10
        else:
            score -= 22
        hard_block = pos >= ENTRY_CHASE_HIGH
    else:
        if pos >= 0.65:
            score += 15
        elif pos >= ENTRY_PREFERRED_SELL:
            score += 10
        elif pos >= 0.30:
            score += 0
        elif pos >= (1.0 - ENTRY_CHASE_HIGH):
            score -= 10
        else:
            score -= 22
        hard_block = pos <= (1.0 - ENTRY_CHASE_HIGH)

    score = int(max(0, min(100, score)))
    if hard_block:
        state = "WAIT_ENTRY"
    elif score >= 70:
        state = "GOOD"
    elif score >= 50:
        state = "ACCEPTABLE"
    else:
        state = "WEAK"

    return {
        "location_score": score, "location_state": state,
        "range_position": round(pos, 3), "range_low": rl, "range_high": rh,
        "hard_block": bool(hard_block),
    }


# =================================================================================
# SECTION 10 — KANDIDAT ENTRY (POI cascade: OB+FVG "unicorn" > OB > Breaker >
#               FVG > Equal High/Low) — WAJIB mempertimbangkan Liquidity Sweep
# =================================================================================
# Urutan prioritas kualitas POI (base score internal, dipakai HANYA untuk
# memilih SATU kandidat terbaik sebelum SL/TP dihitung — bukan confidence
# akhir):
#   1. OB yang overlap FVG ("unicorn model")       base 10
#   2. OB murni (kualitas dari detect_order_block) base 7 + quality
#   3. Breaker Block (retest zona gagal)            base 6
#   4. FVG murni                                    base 5
#   5. Equal High/Low (liquidity pool)               base 3
# Modifier (ditambahkan ke base):
#   + liquidity sweep terjadi di/dekat level     : +3   (permintaan #2/#3)
#   + inducement sudah disapu sebelum POI ini    : +1.5
#   + konfluensi zona H1 (overlap)               : +2.5
#   + posisi diskon/premium (fib ≤/≥0.618)       : +1.5
#   + Candle Range Theory rejection selaras      : +1.5
# Filter keras (kandidat dibuang, bukan cuma dikurangi skornya):
#   - tidak reachable dari harga sekarang (limit order harus genuine
#     retracement, bukan entry di seberang harga / sudah lewat)
#   - jarak ke harga sekarang > MAIN_ENTRY_MAX_ATR x ATR (akan ditolak
#     main.py juga — jangan kembalikan kandidat yang pasti gagal)
#   - lokasi entry hard-block (chasing jelas, lihat entry_location)

def _reachable(direction: str, entry_pt: float, zone_bot: float, zone_top: float,
              current_price: float) -> bool:
    """Limit order harus genuine retracement: BUY di bawah/at harga sekarang
    dan masih di dalam zona (belum tersapu penuh); SELL sebaliknya."""
    if direction == "bull":
        return zone_bot <= current_price and entry_pt <= current_price * 1.001
    return zone_top >= current_price and entry_pt >= current_price * 0.999


def collect_poi_candidates(h1: pd.DataFrame, m15: pd.DataFrame, direction: str,
                           current_price: float, atr: float, ctx: dict) -> list:
    up = direction == "bull"
    sh15, sl15 = ctx["sh15"], ctx["sl15"]
    sh1, sl1 = ctx["sh1"], ctx["sl1"]

    liq = ctx["liquidity_bull"] if up else ctx["liquidity_bear"]
    liq_ok = liq.get("type") == "sweep"
    choch = ctx["choch_m15"]
    choch_ok = choch.get("bullish_choch") if up else choch.get("bearish_choch")
    induce = ctx["inducement_bull"] if up else ctx["inducement_bear"]
    induce_ok = bool(induce.get("swept"))
    crt = ctx["crt_bull"] if up else ctx["crt_bear"]
    crt_ok = bool(crt.get("triggered"))

    fib_sh = float(m15["high"].iloc[sh15[-1]]) if sh15 else None
    fib_sl = float(m15["low"].iloc[sl15[-1]]) if sl15 else None

    # Zona H1 searah — untuk cek konfluensi ("How to Choose the Best OB")
    htf_zones = []
    try:
        for z in detect_order_block(h1, direction, sh=sh1, sl=sl1, lb=ZONE_LOOKBACK_H1):
            htf_zones.append((z["top"], z["bot"]))
        for f in detect_fvg(h1, direction, lb=ZONE_LOOKBACK_H1):
            htf_zones.append((f["top"], f["bot"]))
    except Exception:
        htf_zones = []

    def _confluence(top, bot):
        return any(zones_overlap(top, bot, zt, zb) for zt, zb in htf_zones)

    def _sweep_bonus(entry_pt):
        if not liq_ok or not liq.get("level"):
            return 0.0
        lev = float(liq["level"])
        return 3.0 if (up and entry_pt >= lev * 0.995) or (not up and entry_pt <= lev * 1.005) else 0.0

    def _fib_bonus(top, bot):
        if fib_sh is None or fib_sl is None or fib_sh <= fib_sl:
            return 0.0
        r = fib_position((top + bot) / 2, fib_sl, fib_sh)
        if up and r <= 0.618:
            return 1.5
        if not up and r >= 0.382:
            return 1.5
        return 0.0

    cands = []

    # ── OB (murni + "unicorn" bila overlap FVG) ─────────────────────────
    obs = detect_order_block(m15, direction, sh=sh15, sl=sl15, lb=ZONE_LOOKBACK_M15)
    for z in obs:
        entry_pt = float(z["top"]) if up else float(z["bot"])
        invalid_pt = float(z["bot"]) if up else float(z["top"])
        if not _reachable(direction, entry_pt, z["bot"], z["top"], current_price):
            continue
        base = 10.0 if z.get("has_fvg") else (7.0 + z["quality"])
        label = "ob_fvg" if z.get("has_fvg") else "ob"
        sweep_b = _sweep_bonus(entry_pt)
        fib_b = _fib_bonus(z["top"], z["bot"])
        conf = _confluence(z["top"], z["bot"])
        sc = base + sweep_b + fib_b + (2.5 if conf else 0.0) + (1.5 if induce_ok else 0.0)
        if choch_ok:
            sc += 2.0
        if crt_ok:
            sc += 1.5
        freshness = min(1.0, 0.55 + z.get("quality", 0) / 6.0) if label == "ob" else 1.0
        cands.append({"price": round(entry_pt, 8), "invalid": round(invalid_pt, 8),
                      "label": label, "score": sc, "zone": z,
                      "sweep_used": sweep_b > 0, "fib_ok": fib_b > 0,
                      "confluence": conf, "choch_used": bool(choch_ok),
                      "crt_used": crt_ok, "induce_used": induce_ok,
                      "freshness": freshness})

    # ── Breaker Block ────────────────────────────────────────────────────
    breakers = detect_breaker_block(m15, direction, sh=sh15, sl=sl15, lb=ZONE_LOOKBACK_M15)
    for z in breakers:
        entry_pt = float(z["top"]) if up else float(z["bot"])
        invalid_pt = float(z["bot"]) if up else float(z["top"])
        if not _reachable(direction, entry_pt, z["bot"], z["top"], current_price):
            continue
        sweep_b = _sweep_bonus(entry_pt)
        conf = _confluence(z["top"], z["bot"])
        sc = 6.0 + sweep_b + (2.5 if conf else 0.0) + (1.5 if induce_ok else 0.0)
        if choch_ok:
            sc += 2.0
        if crt_ok:
            sc += 1.5
        cands.append({"price": round(entry_pt, 8), "invalid": round(invalid_pt, 8),
                      "label": "breaker", "score": sc, "zone": z,
                      "sweep_used": sweep_b > 0, "fib_ok": False,
                      "confluence": conf, "choch_used": bool(choch_ok),
                      "crt_used": crt_ok, "induce_used": induce_ok,
                      "freshness": 0.65 if z.get("has_fvg") else 0.55})

    # ── FVG murni ───────────────────────────────────────────────────────
    fvgs = detect_fvg(m15, direction, lb=FVG_LOOKBACK)
    for f in fvgs:
        entry_pt = f["mid"]
        invalid_pt = f["top"] if up else f["bot"]
        if not _reachable(direction, entry_pt, f["bot"], f["top"], current_price):
            continue
        sweep_b = _sweep_bonus(entry_pt)
        fib_b = _fib_bonus(f["top"], f["bot"])
        conf = _confluence(f["top"], f["bot"])
        sc = 5.0 + sweep_b + fib_b + (2.5 if conf else 0.0) + (1.5 if induce_ok else 0.0)
        if choch_ok:
            sc += 1.0
        if crt_ok:
            sc += 1.5
        cands.append({"price": round(entry_pt, 8), "invalid": round(invalid_pt, 8),
                      "label": "fvg", "score": sc, "zone": f,
                      "sweep_used": sweep_b > 0, "fib_ok": fib_b > 0,
                      "confluence": conf, "choch_used": bool(choch_ok),
                      "crt_used": crt_ok, "induce_used": induce_ok,
                      "freshness": 0.5})

    # ── Equal High/Low (liquidity pool) ─────────────────────────────────
    eqs = detect_equal_levels(m15, "low" if up else "high", lb=80)
    for eq in eqs[-2:]:
        if not up and float(eq) < current_price * 0.999:
            continue
        if up and float(eq) > current_price * 1.001:
            continue
        invalid_pt = eq - atr * 0.8 if up else eq + atr * 0.8
        sweep_b = _sweep_bonus(eq)
        sc = 3.0 + sweep_b + (1.5 if induce_ok else 0.0)
        if crt_ok:
            sc += 1.0
        cands.append({"price": round(float(eq), 8), "invalid": round(float(invalid_pt), 8),
                      "label": "eq", "score": sc, "zone": {"top": eq, "bot": eq},
                      "sweep_used": sweep_b > 0, "fib_ok": False,
                      "confluence": False, "choch_used": False,
                      "crt_used": crt_ok, "induce_used": induce_ok,
                      "freshness": 0.35})

    # ── Filter: jarak ke harga & lokasi entry ───────────────────────────
    enriched = []
    for c in cands:
        if abs(c["price"] - current_price) > atr * MAIN_ENTRY_MAX_ATR:
            continue
        loc = entry_location(m15, direction, c["price"], atr)
        if loc["hard_block"]:
            continue
        c = dict(c)
        c["location"] = loc
        c["score"] = c["score"] + (loc["location_score"] - 50) * 0.05
        enriched.append(c)

    enriched.sort(key=lambda c: -c["score"])
    return enriched


# =================================================================================
# SECTION 11 — SL (permintaan #3): harus menandakan pembalikan arah SUNGGUHAN
# =================================================================================
# Prioritas kandidat SL (paling presisi/relevan dulu):
#   1. sweep_wick  — kalau ada liquidity sweep BENAR TERJADI di dekat POI,
#      SL diletakkan di luar UJUNG WICK sweep tsb + buffer kecil. Ini paling
#      tepat memenuhi permintaan: SL baru tersentuh kalau level itu disapu
#      LEBIH DALAM dari sweep yang sudah terjadi — bukti kuat arah salah,
#      bukan pengulangan sweep yang sama.
#   2. zone_invalid — sisi luar OB/FVG/Breaker yang dipakai sebagai entry.
#   3. struct_h1   — swing H1 terakhir (lebih tahan noise, invalidation besar).
#   4. struct_m15  — swing M15 terakhir (fallback presisi lebih rendah).
# Semua kandidat dibatasi ke rentang risk [SL_MIN_RISK_ATR, SL_MAX_RISK_ATR]
# ATR (dan tidak lebih dari SL_MAX_RISK_PCT_OF_ENTRY dari harga entry) supaya
# risk tetap proporsional — SL yang terlalu lebar bukan diperbaiki dengan
# menggeser ke dalam (itu memalsukan RR), melainkan kandidat entry itu yang
# akan dicoba gantikan oleh kandidat lain di collect_poi_candidates().

def compute_sl(direction: str, entry: float, invalid_level: float,
               liq_sweep: dict, m15: pd.DataFrame, h1: pd.DataFrame,
               sh15: list, sl15: list, sh1: list, sl1: list,
               atr: float) -> Optional[Tuple[float, float, str]]:
    up = direction == "bull"
    buf = atr * SL_BUFFER_ATR
    min_risk = atr * SL_MIN_RISK_ATR
    max_risk = min(atr * SL_MAX_RISK_ATR, entry * SL_MAX_RISK_PCT_OF_ENTRY)

    cands = []  # (priority, label, sl_price, risk)

    # 1. Ujung wick liquidity sweep (paling menandakan "arah benar-benar salah")
    if liq_sweep and liq_sweep.get("type") == "sweep" and liq_sweep.get("wick_extreme") is not None:
        extreme = float(liq_sweep["wick_extreme"])
        sl_raw = extreme - buf if up else extreme + buf
        risk = abs(entry - sl_raw)
        if min_risk <= risk <= max_risk and ((sl_raw < entry) if up else (sl_raw > entry)):
            cands.append((0, "sweep_wick", sl_raw, risk))

    # 2. Sisi luar zona (invalidation POI)
    if invalid_level is not None:
        sl_raw = invalid_level - buf if up else invalid_level + buf
        risk = abs(entry - sl_raw)
        if min_risk <= risk <= max_risk and ((sl_raw < entry) if up else (sl_raw > entry)):
            cands.append((1, "zone_invalid", sl_raw, risk))

    # 3. Swing H1 (invalidation besar, tahan noise)
    if up and sl1:
        h1_low = float(h1["low"].iloc[sl1[-1]])
        if h1_low < entry:
            sl_raw = h1_low - buf
            risk = abs(entry - sl_raw)
            if min_risk <= risk <= max_risk:
                cands.append((2, "struct_h1", sl_raw, risk))
    elif not up and sh1:
        h1_high = float(h1["high"].iloc[sh1[-1]])
        if h1_high > entry:
            sl_raw = h1_high + buf
            risk = abs(entry - sl_raw)
            if min_risk <= risk <= max_risk:
                cands.append((2, "struct_h1", sl_raw, risk))

    # 4. Swing M15 (fallback presisi lebih rendah)
    if up and sl15:
        m_low = float(m15["low"].iloc[sl15[-1]])
        if m_low < entry:
            sl_raw = m_low - buf
            risk = abs(entry - sl_raw)
            if min_risk <= risk <= max_risk:
                cands.append((3, "struct_m15", sl_raw, risk))
    elif not up and sh15:
        m_high = float(m15["high"].iloc[sh15[-1]])
        if m_high > entry:
            sl_raw = m_high + buf
            risk = abs(entry - sl_raw)
            if min_risk <= risk <= max_risk:
                cands.append((3, "struct_m15", sl_raw, risk))

    if not cands:
        return None

    # Prioritas ditentukan urutan di atas; dalam prioritas sama pilih risk
    # TERBESAR (invalidation paling jauh = paling tahan terhadap noise/sweep
    # ulang).
    cands.sort(key=lambda x: (x[0], -x[3]))
    _, label, sl_price, risk = cands[0]
    return round(sl_price, 8), risk, label


# =================================================================================
# SECTION 12 — TP POOL & SELEKSI (permintaan #1: RR 1:2..1:4, jangan auto-tolak)
# =================================================================================
# Pool "draw on liquidity", EXTERNAL (H1: OB/FVG/swing/EQ lawan arah, magnet
# harga sesungguhnya) diprioritaskan atas INTERNAL (EQ M15, sering cuma
# disapu di tengah jalan) — sesuai video external vs internal liquidity.
# Tier lebih kecil = lebih diutamakan bila beberapa target sama2 valid RR.

def build_tp_pool(h1: pd.DataFrame, m15: pd.DataFrame, direction: str,
                  entry: float, atr: float, sh1: list, sl1: list,
                  sh15: list, sl15: list) -> list:
    up = direction == "bull"
    sgn = 1 if up else -1
    pool = []
    opp_dir = "bear" if up else "bull"

    # Tier 1: OB H1 lawan arah — external, presisi tinggi
    for z in detect_order_block(h1, opp_dir, sh=sh1, sl=sl1, lb=ZONE_LOOKBACK_H1):
        edge = float(z["bot"]) if up else float(z["top"])
        if sgn * (edge - entry) > atr * 0.5:
            pool.append(("ob_h1", edge, 1))

    # Tier 2: FVG H1 lawan arah — external
    for f in detect_fvg(h1, opp_dir, lb=ZONE_LOOKBACK_H1):
        if sgn * (f["mid"] - entry) > atr * 0.5:
            pool.append(("fvg_h1", f["mid"], 2))

    # Tier 3: swing H1 terakhir (external, struktur murni — external liquidity)
    sw_vals = ([float(h1["high"].iloc[i]) for i in sh1] if up
              else [float(h1["low"].iloc[i]) for i in sl1])
    for v in sw_vals[-2:]:
        if sgn * (v - entry) > atr * 1.0:
            pool.append(("sw_h1", v, 3))

    # Tier 4: Equal High/Low H1 — liquidity pool besar (external, "draw on liquidity")
    for v in detect_equal_levels(h1, "high" if up else "low", lb=100):
        if sgn * (v - entry) > atr * 0.8:
            pool.append(("eq_h1", v, 4))

    # Tier 5: swing H1 lebih jauh — cadangan ekstensi kalau target dekat RR<2
    sw_all = ([float(h1["high"].iloc[i]) for i in sh1] if up
             else [float(h1["low"].iloc[i]) for i in sl1])
    for v in sw_all[:-2]:
        if sgn * (v - entry) > atr * 1.0:
            pool.append(("sw_h1_far", v, 5))

    # Tier 6: Candle Range Theory — sisi berlawanan dari range candle acuan M15
    crt = detect_candle_range_rejection(m15, direction)
    if crt.get("triggered"):
        target = crt["range_high"] if up else crt["range_low"]
        if target is not None and sgn * (target - entry) > atr * 0.3:
            pool.append(("crt", float(target), 6))

    # Tier 7: EQ M15 — internal, tetap dipakai (frekuensi sinyal) tapi prioritas rendah
    for v in detect_equal_levels(m15, "high" if up else "low", lb=80):
        if sgn * (v - entry) > atr * 0.3:
            pool.append(("eq_m15", v, 7))

    # Tier 8 & 9: Fibonacci extension — cadangan ekstensi terjauh
    if sh1 and sl1:
        sh_val = float(h1["high"].iloc[sh1[-1]])
        sl_val = float(h1["low"].iloc[sl1[-1]])
        leg = sh_val - sl_val
        if leg > 0:
            for ext, lbl, tier in ((FIB_EXT_1, "fib127", 8), (FIB_EXT_2, "fib162", 8),
                                   (1.0, "fib200", 9), (1.414, "fib241", 9)):
                tp_v = (sh_val + leg * ext) if up else (sl_val - leg * ext)
                if sgn * (tp_v - entry) > atr * 0.5:
                    pool.append((lbl, tp_v, tier))

    pool.sort(key=lambda x: abs(x[1] - entry))
    return pool


def select_tp(pool: list, entry: float, risk: float,
             direction: str) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    """Aturan Entry->SL->TP persis permintaan #1:
      * Target pertama dengan RR<MIN_RR TIDAK langsung ditolak — telusuri
        pool (dari terdekat ke terjauh) sampai ketemu target real yang
        RR-nya >= MIN_RR.
      * Kalau target valid itu RR > MAX_RR, TP dipotong tepat di MAX_RR
        (bukan ditolak).
      * Kalau TIDAK ADA target di pool yang mencapai MIN_RR sama sekali,
        tidak mengarang target — return None (caller akan coba kandidat
        entry lain).
    """
    if not pool or risk <= 0:
        return None, None, None
    sgn = 1 if direction == "bull" else -1

    targets = []
    for lbl, value, tier in pool:
        value = float(value)
        distance = sgn * (value - entry)
        if distance <= 0:
            continue
        rr = distance / risk
        targets.append((lbl, value, int(tier), rr))
    if not targets:
        return None, None, None

    targets.sort(key=lambda x: x[3])  # RR menaik = jarak menaik (risk konstan)
    for lbl, value, tier, rr in targets:
        if rr >= MIN_RR:
            if rr <= MAX_RR:
                return round(value, 8), lbl, round(rr, 2)
            capped = entry + sgn * risk * MAX_RR
            return round(capped, 8), lbl + "_capped", MAX_RR
    return None, None, None


# =================================================================================
# SECTION 13 — CONFIDENCE (permintaan #5): dihitung TERAKHIR, semua berdasar,
#               bobot dijumlah = 100 by construction (bukan hasil kalibrasi
#               ulang setelah observasi) — tidak ada komponen "abu-abu".
# =================================================================================
# Kategori & bobot (total = 100):
#   A) HTF Bias Alignment      35  — D1 bias searah 12, H1 struct searah 12,
#                                     EMA H1 stack searah 6, BOS H1 searah 5
#   B) M15 Structural Trigger  30  — ChoCH 10, CISD 8, Candle Range Theory 6,
#                                     BOS/failed-retest lanjutan 6
#   C) Liquidity Context       15  — Liquidity Sweep 8, Inducement 4,
#                                     Konfluensi H1 3
#   D) Kualitas Zona (POI)     10  — freshness+displacement+FVG 6, posisi
#                                     Fibonacci diskon/premium 4
#   E) Timing RSI + Volume     10  — RSI 5, Volume 5
# Modifier pasca-jumlah (HANYA mengurangi/menyesuaikan, tidak menambah bobot
# baru yang tidak berdasar):
#   - entry_location_score  → faktor pengali 0.55..1.00 (chase parah = 0.55)
#   - konflik D1 vs H1       → x0.85
#   - macro BTC H1 (opsional): searah +6 (dibatasi ke 100), berlawanan x0.75

def compute_confidence(
    d1_agree: bool, h1_agree: bool, ema_h1_agree: bool, bos_h1_agree: bool,
    choch_agree: bool, cisd_agree: bool, crt_agree: bool, bos_or_fr_agree: bool,
    sweep_agree: bool, inducement_agree: bool, htf_confluence: bool,
    zone_freshness: float, fib_ok: bool,
    rsi_ok: bool, vol_ok: bool,
    location_score: int, htf_conflict: bool,
    macro_align: Optional[bool],
) -> int:
    score = 0.0
    # A) HTF Bias Alignment — 35
    if d1_agree: score += 12
    if h1_agree: score += 12
    if ema_h1_agree: score += 6
    if bos_h1_agree: score += 5
    # B) M15 Structural Trigger — 30
    if choch_agree: score += 10
    if cisd_agree: score += 8
    if crt_agree: score += 6
    if bos_or_fr_agree: score += 6
    # C) Liquidity Context — 15
    if sweep_agree: score += 8
    if inducement_agree: score += 4
    if htf_confluence: score += 3
    # D) Kualitas Zona — 10
    score += 6.0 * _clip01(zone_freshness)
    if fib_ok: score += 4
    # E) Timing RSI + Volume — 10
    if rsi_ok: score += 5
    if vol_ok: score += 5

    loc_factor = 0.55 + 0.45 * _clip01(location_score / 100.0)
    score *= loc_factor
    if htf_conflict:
        score *= 0.85
    if macro_align is True:
        score = min(100.0, score + MACRO_ALIGN_BONUS)
    elif macro_align is False:
        score *= MACRO_AGAINST_MULT

    return int(max(0, min(100, round(score))))


# =================================================================================
# SECTION 14 — ORKESTRASI: score_direction (konteks) & full_analyze (utama)
# =================================================================================

def score_direction(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                    df_d1: Optional[pd.DataFrame] = None,
                    df_btc_h1: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """Bangun konteks lengkap (bias HTF + seluruh sinyal M15 dua arah) yang
    dipakai full_analyze() untuk memilih arah, lalu Entry->SL->TP->Confidence.
    Tidak dipanggil main.py secara langsung — murni internal/opsional hook
    (tetap diberi nama ini untuk kompatibilitas adaptive-bind /ganti)."""
    h1 = build_df(df_h1, interval_minutes=60)
    m15 = build_df(df_m15, interval_minutes=15)
    if h1 is None or m15 is None:
        return None

    L1, L15 = h1.iloc[-1], m15.iloc[-1]
    atr = max(float(L15["atr"]), float(L1["atr"]) / 4.0, float(L15["close"]) * 0.003)

    sh1, sl1 = swing_pts(h1, lb=SWING_LB_H1)
    sh15, sl15 = swing_pts(m15, lb=SWING_LB_M15)

    htf = htf_bias(df_h1, df_d1, h1)
    bos_h1 = detect_bos(h1, sh1, sl1)
    choch_h1 = detect_choch(h1, sh1, sl1)

    choch_m15 = detect_choch(m15, sh15, sl15)
    bos_m15 = detect_bos(m15, sh15, sl15)
    cisd_m15 = detect_cisd(m15, sh15, sl15, atr, lb=CISD_LOOKBACK)
    liq_bull = detect_liquidity_sweep(m15, sh15, sl15, "bull", atr)
    liq_bear = detect_liquidity_sweep(m15, sh15, sl15, "bear", atr)
    ext_liq_bull = external_liquidity(h1, sh1, sl1, "bull")
    ext_liq_bear = external_liquidity(h1, sh1, sl1, "bear")
    fr_m15 = detect_failed_retest(m15, sh15, sl15, atr)
    induce_bull = detect_inducement(m15, "bull")
    induce_bear = detect_inducement(m15, "bear")
    crt_bull = detect_candle_range_rejection(m15, "bull")
    crt_bear = detect_candle_range_rejection(m15, "bear")
    rdiv_bull = detect_rsi_divergence(m15, "bull", lb=30)
    rdiv_bear = detect_rsi_divergence(m15, "bear", lb=30)

    # ── Tally trigger M15 per arah — dipakai HANYA kalau HTF neutral/conflict
    # (gate tetap arah HTF; ini cuma tie-breaker lokal, bukan pengganti gate).
    bull_triggers = sum([
        choch_m15.get("bullish_choch", False), cisd_m15.get("bullish_cisd", False),
        crt_bull.get("triggered", False), bos_m15.get("bullish_bos", False),
        liq_bull.get("type") == "sweep",
    ])
    bear_triggers = sum([
        choch_m15.get("bearish_choch", False), cisd_m15.get("bearish_cisd", False),
        crt_bear.get("triggered", False), bos_m15.get("bearish_bos", False),
        liq_bear.get("type") == "sweep",
    ])

    combined = htf["combined"]
    if combined == "bullish":
        direction = "bull"
    elif combined == "bearish":
        direction = "bear"
    else:
        direction = "bull" if bull_triggers >= bear_triggers else "bear"

    macro = macro_bias(df_btc_h1)

    return {
        "direction": direction, "price": float(L15["close"]), "atr": atr,
        "struct_h1": htf["struct_h1"], "h1_bias": htf["h1_bias"],
        "d1_bias": htf["d1_bias"], "combined": combined,
        "ema_h1_bull": htf["ema_h1_bull"], "ema_h1_bear": htf["ema_h1_bear"],
        "bos_h1": bos_h1, "choch_h1": choch_h1,
        "choch_m15": choch_m15, "bos_m15": bos_m15, "cisd_m15": cisd_m15,
        "liquidity_bull": liq_bull, "liquidity_bear": liq_bear,
        "external_liquidity_bull": ext_liq_bull, "external_liquidity_bear": ext_liq_bear,
        "failed_retest": fr_m15,
        "inducement_bull": induce_bull, "inducement_bear": induce_bear,
        "crt_bull": crt_bull, "crt_bear": crt_bear,
        "rsi_div_bull": rdiv_bull, "rsi_div_bear": rdiv_bear,
        "macro_bias": macro,
        "sh1": sh1, "sl1": sl1, "sh15": sh15, "sl15": sl15,
        "bull_triggers": bull_triggers, "bear_triggers": bear_triggers,
    }


def get_best_signal(candidates: list) -> Optional[dict]:
    """Pilih sinyal terbaik dari beberapa dict hasil full_analyze() (opsional,
    dipertahankan untuk kompatibilitas — main.py sendiri melakukan ranking
    confidence→RR di run_scan_once())."""
    if not candidates:
        return None
    label_bonus = {"ob_fvg": 5, "ob": 4, "breaker": 3, "fvg": 2, "eq": 1}

    def _rank(sig):
        return sig["confidence"] + label_bonus.get(sig.get("entry_label", ""), 0) + sig.get("rr", 0) * 0.5

    return max(candidates, key=_rank)


def full_analyze(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                 df_d1: Optional[pd.DataFrame] = None,
                 symbol: Optional[str] = None,
                 df_btc_h1: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """Analisa satu koin, urutan PERSIS sesuai permintaan:
    Bias -> Entry (POI + Liquidity Sweep) -> SL (anti-sweep) -> TP (RR 2..4)
    -> Confidence (dihitung terakhir dari seluruh bukti di atas).
    Trail TIDAK dihitung di sini (lihat docstring modul) — main.py memakai
    swing_pts()+STRUCT_TRAIL_* langsung.
    """
    try:
        ctx = score_direction(df_h1, df_m15, df_d1=df_d1, df_btc_h1=df_btc_h1)
        if ctx is None:
            return None

        h1 = build_df(df_h1, interval_minutes=60)
        m15 = build_df(df_m15, interval_minutes=15)
        if h1 is None or m15 is None:
            return None

        direction = ctx["direction"]
        up = direction == "bull"
        cur_price = ctx["price"]
        atr = ctx["atr"]
        sh1, sl1, sh15, sl15 = ctx["sh1"], ctx["sl1"], ctx["sh15"], ctx["sl15"]

        candidates = collect_poi_candidates(h1, m15, direction, cur_price, atr, ctx)
        if not candidates:
            if symbol:
                log.debug(f"[{symbol}] tidak ada POI entry yang reachable/valid.")
            return None

        liq_sweep_ctx = ctx["liquidity_bull"] if up else ctx["liquidity_bear"]
        timing = entry_timing(m15, direction)

        d1_agree = ctx["d1_bias"] == ("bullish" if up else "bearish")
        h1_agree = ctx["h1_bias"] == ("bullish" if up else "bearish")
        ema_h1_agree = ctx["ema_h1_bull"] if up else ctx["ema_h1_bear"]
        bos_h1_agree = bool(ctx["bos_h1"].get("bullish_bos" if up else "bearish_bos"))
        choch_agree = bool(ctx["choch_m15"].get("bullish_choch" if up else "bearish_choch"))
        cisd_agree = bool(ctx["cisd_m15"].get("bullish_cisd" if up else "bearish_cisd"))
        bos_or_fr_agree = bool(ctx["bos_m15"].get("bullish_bos" if up else "bearish_bos")) or \
                          bool(ctx["failed_retest"].get("failed_retest_buy" if up else "failed_retest_sell"))
        htf_conflict = ctx["combined"] == "conflict"

        macro = ctx["macro_bias"]
        if macro in ("unknown", "ranging"):
            macro_align = None
        else:
            macro_align = macro == ("bullish" if up else "bearish")

        best_result = None
        for cand in candidates:
            entry = cand["price"]
            invalid_level = cand["invalid"]

            sl_result = compute_sl(direction, entry, invalid_level, liq_sweep_ctx,
                                   m15, h1, sh15, sl15, sh1, sl1, atr)
            if sl_result is None:
                continue
            sl_price, risk, sl_label = sl_result
            if risk <= 0:
                continue

            tp_pool = build_tp_pool(h1, m15, direction, entry, atr, sh1, sl1, sh15, sl15)
            tp_price, tp_label, rr = select_tp(tp_pool, entry, risk, direction)
            if tp_price is None:
                continue

            geo_ok = (sl_price < entry < tp_price) if up else (tp_price < entry < sl_price)
            if not geo_ok:
                continue

            loc = cand.get("location", {}) or entry_location(m15, direction, entry, atr)
            confidence = compute_confidence(
                d1_agree=d1_agree, h1_agree=h1_agree, ema_h1_agree=ema_h1_agree,
                bos_h1_agree=bos_h1_agree, choch_agree=choch_agree, cisd_agree=cisd_agree,
                crt_agree=bool(cand.get("crt_used")), bos_or_fr_agree=bos_or_fr_agree,
                sweep_agree=bool(cand.get("sweep_used")), inducement_agree=bool(cand.get("induce_used")),
                htf_confluence=bool(cand.get("confluence")), zone_freshness=cand.get("freshness", 0.5),
                fib_ok=bool(cand.get("fib_ok")), rsi_ok=timing["rsi_ok"], vol_ok=timing["vol_ok"],
                location_score=loc.get("location_score", 50), htf_conflict=htf_conflict,
                macro_align=macro_align,
            )

            rsi_val = round(float(m15["rsi"].iloc[-1]), 1)
            reason = (
                f"Entry@{entry:.6g}({cand['label']}) | SL@{sl_price:.6g}({sl_label}) | "
                f"TP@{tp_price:.6g}({tp_label}) | RR=1:{rr} | "
                f"Sweep={'yes' if cand.get('sweep_used') else 'no'} | "
                f"Induce={'yes' if cand.get('induce_used') else 'no'} | "
                f"RSI={rsi_val}({'rising' if timing['rsi_slope'] > 0 else 'falling'}) | "
                f"Vol={timing['vol_ratio']}x"
            )

            best_result = {
                "symbol": symbol,
                "decision": "BUY" if up else "SELL",
                "original_dir": direction,
                "confidence": confidence,
                "price": round(cur_price, 8),
                "entry": round(entry, 8),
                "entry_label": cand["label"],
                "sl": round(sl_price, 8),
                "tp": round(tp_price, 8),
                "rr": rr,
                "atr": round(atr, 8),
                "rsi": rsi_val,
                "struct_h1": ctx["struct_h1"],
                "d1_bias": ctx["d1_bias"],
                "htf_bias": ctx["combined"],
                "h1_bias": ctx["h1_bias"],
                "choch_m15": ctx["choch_m15"],
                "choch_h1": ctx["choch_h1"],
                "bos_m15": ctx["bos_m15"],
                "bos_h1": ctx["bos_h1"],
                "cisd_m15": ctx["cisd_m15"],
                "failed_retest": ctx["failed_retest"],
                "selected_sweep": bool(cand.get("sweep_used")),
                "entry_location_score": loc.get("location_score", 50),
                "entry_location_state": loc.get("location_state", "unknown"),
                "tp_sl_reason": reason,
            }
            break  # kandidat sudah terurut skor terbaik dulu; ambil yang pertama valid

        if best_result is None and symbol:
            log.debug(f"[{symbol}] semua kandidat gugur di SL/TP/geometry — tidak ada setup.")
        if best_result and symbol:
            log.info(
                f"[{symbol}] {best_result['decision']} entry={best_result['entry']:.6g} "
                f"({best_result['entry_label']}) sl={best_result['sl']:.6g} "
                f"tp={best_result['tp']:.6g} rr=1:{best_result['rr']} "
                f"conf={best_result['confidence']}%"
            )
        return best_result

    except Exception as e:
        if symbol:
            log.error(f"[full_analyze] {symbol}: {e}", exc_info=True)
        return None


# =================================================================================
# SECTION 15 — VALIDASI PRE/POST-ORDER (dipanggil LANGSUNG oleh main.py)
# =================================================================================

def validate_and_adjust_geometry(
    entry: float, sl: float, tp: float,
    current_price: float, atr: float, direction: str,
) -> Optional[dict]:
    """Dipanggil main.py setelah order terisi (real trade) untuk memvalidasi
    ulang geometri, DAN — permintaan #3 — membedakan Liquidity Sweep dangkal
    (bisa diselamatkan) dari reversal sungguhan (harus auto-out).

    Kasus:
      1. Geometri valid & SL belum ditembus → OK, RR dicek ≥ MIN_RR.
      2. SL sudah ditembus (current_price melewati sl):
         - Kedalaman ≤ SWEEP_TOLERANCE_ATR (3×ATR, sinkron dengan komentar
           main.py sendiri di _open_position_real) → dianggap Liquidity
           Sweep, BUKAN reversal. SL direlokasi ke luar current_price (bukti
           terbaru seberapa jauh sweep terjadi) + buffer kecil, TP tetap,
           RR dihitung ulang. Kalau RR baru masih ≥ MIN_RR → diselamatkan.
         - Kedalaman > toleransi → reversal sungguhan, fail-closed (None).
      3. Geometri tidak valid (SL/TP di sisi yang salah dari entry) tanpa
         SL ditembus → sinyal basi, TIDAK PERNAH menggeser entry secara
         retroaktif (None).
    CATATAN: sebelumnya kasus 2 SELALU return None (bug) — main.py sudah
    lama menampilkan pesan "Liquidity Sweep setelah fill" seolah fitur ini
    aktif, padahal fungsi lama tidak pernah benar-benar menyelamatkan
    posisi. Ini baru benar-benar diimplementasikan di sini.
    """
    try:
        up = direction == "bull"
        atr = max(_safe_float(atr), 1e-12)

        def geo_ok(e, s, t):
            return (s < e < t) if up else (t < e < s)

        def rr_of(e, s, t):
            return abs(t - e) / max(abs(e - s), 1e-10)

        breached = (current_price <= sl) if up else (current_price >= sl)

        # Kasus 1 — sudah valid, SL belum ditembus.
        if geo_ok(entry, sl, tp) and not breached:
            r = rr_of(entry, sl, tp)
            if r < MIN_RR:
                return None
            return {"entry": entry, "sl": sl, "tp": tp,
                    "rr": round(min(r, MAX_RR), 2), "adjusted": False}

        # Kasus 2 — SL ditembus: cek apakah ini Liquidity Sweep dangkal.
        if breached:
            depth = (sl - current_price) if up else (current_price - sl)
            depth_atr = depth / atr
            if depth_atr > SWEEP_TOLERANCE_ATR:
                log.info(
                    f"[validate_geo] breach {depth_atr:.2f}xATR > toleransi "
                    f"{SWEEP_TOLERANCE_ATR}xATR — reversal sungguhan, ditolak."
                )
                return None

            buf = atr * SWEEP_RELOCATE_BUFFER_ATR
            new_sl = current_price - buf if up else current_price + buf
            if (up and new_sl >= entry) or (not up and new_sl <= entry):
                return None  # sweep terlalu dalam relatif entry, tidak masuk akal direlokasi

            new_risk = abs(entry - new_sl)
            if new_risk <= 0:
                return None
            r = abs(tp - entry) / new_risk
            if r < MIN_RR:
                log.info(
                    f"[validate_geo] Liquidity Sweep {depth_atr:.2f}xATR terdeteksi, "
                    f"tapi RR pasca-relokasi {r:.2f} < MIN_RR — ditolak."
                )
                return None
            log.info(
                f"[validate_geo] Liquidity Sweep {depth_atr:.2f}xATR terdeteksi "
                f"(≤{SWEEP_TOLERANCE_ATR}xATR) — SL direlokasi {sl:.6g}→{new_sl:.6g}, "
                f"RR baru 1:{min(r, MAX_RR):.2f}."
            )
            return {"entry": entry, "sl": round(new_sl, 8), "tp": tp,
                    "rr": round(min(r, MAX_RR), 2), "adjusted": True}

        # Kasus 3 — geometri tidak valid & SL belum ditembus → sinyal basi.
        if not geo_ok(entry, sl, tp):
            return None
        r = rr_of(entry, sl, tp)
        if r < MIN_RR:
            return None
        return {"entry": round(entry, 8), "sl": round(sl, 8), "tp": tp,
                "rr": round(min(r, MAX_RR), 2), "adjusted": False}
    except Exception as e:
        log.error(f"[validate_and_adjust_geometry] {e}", exc_info=True)
        return None
