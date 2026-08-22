"""
strategy_logic.py — OTAK v7 (Adaptive RR + Intelligent Target Selection + Context-Aware Trailing)
========================================================================================
Dibangun dari corpus transkrip video SMC/ICT (channel RUANG TRADER, ~39 video:
market structure, order block, FVG, liquidity sweep, inducement, ChoCH/BOS,
CISD, OTE & Fibonacci, external vs internal liquidity, Wyckoff, dsb).

Urutan proses tiap sinyal (SESUAI PERMINTAAN): Entry → SL → TP → Confidence
global (tanpa bias sesi). Arah HTF adalah gate keputusan, bukan sekadar bonus
skor. Setup lokal hanya boleh dipakai setelah POI HTF disentuh dan ada
konfirmasi displacement/structure yang baru. Ini sengaja mengurangi sinyal:
skor agregat lama dapat membuat OTE/RSI/sweep mengalahkan tren utama.

Poin revisi v3:
  1) RR minimal 1:2, maksimal 1:4. Jika target RR<2 → JANGAN auto‑tolak,
     dulu cari target lebih jauh dari pool TP (swing H1 lebih lama, EQ H1,
     fib extension) sebelum menyerah; jika ada RR>4 → cap ke 4 (bukan tolak).
  2) Entry presisi: OB/FVG/EQ dengan bonus konfluen HTF (overlap zona H1) dan
     bonus Inducement (liquidity minor sudah disapu sebelum POI asli) —
     sesuai transkrip "How the Market Traps Traders with Inducement" &
     "trading strategy using inducement [SNIPER ENTRY]".
  3) SL struktural anti‑Liquidity Sweep: SL ditaruh di luar invalidation
     level (OB/struktur) + buffer, supaya jika benar‑benar tersentuh berarti
     arah memang salah, bukan sekadar sweep likuiditas minor. Penanganan
     "SL tersentuh tapi ternyata sweep" (harga lanjut sesuai arah semula)
     ditangani di validate_and_adjust_geometry() (dipanggil main.py saat
     order terisi/di-monitor) dan di monitor_position() main.py (verifikasi
     candle M1 sebelum SL dikonfirmasi).
  4) Trail Ladder: seluruh profit-ladder dihapus. Trail hanya mengikuti
     struktur M15 (HL/LH + buffer), sehingga pergeseran SL berarti struktur
     mulai gagal, bukan sekadar memaksa profit terkunci.
  5) Proses per‑koin: Entry → SL → TP → confidence (global, tanpa sesi).
  6) Tidak ada confidence per‑sesi — confidence murni dari kualitas
     struktur+konfluensi chart (skema sama seperti v2, ditambah bonus baru).
  7) Konsep tambahan (inducement, external/internal liquidity, konfluensi
     HTF) masuk sebagai bonus skor kecil, TIDAK menambah filter penolakan,
     supaya kuantitas sinyal tidak turun.
  8) Entry-location layer: arah bullish/bearish tidak otomatis berarti entry
     sekarang. Kandidat dinilai terhadap range M15, adverse swing, dan timing
     RSI. Harga yang terlalu dekat sisi salah range + momentum RSI yang masih
     melawan entry menjadi WAIT_ENTRY, bukan langsung dipaksa trade.
  9) Candidate fallback: jika kandidat terbaik punya invalidation terlalu lebar
     atau TP tidak mencapai 2R, candidate berikutnya dicoba. Tidak ada synthetic
     ATR stop untuk menyelamatkan entry yang buruk.
 10) Tidak menambahkan logika scalping M1/Silver Bullet/killzone sesi yang
     tidak relevan untuk swing H1/M15 — hanya diambil konsep yang dipakai
     bot ini (structure, OB/FVG, liquidity, CISD, Fibonacci).

Kompatibel dengan main.py:
  - full_analyze(df_h1, df_m15, df_d1=None, symbol=None) → dict | None
  - score_direction(df_h1, df_m15, df_d1=None) → dict | None
  - swing_pts(df, lb) → (sh, sl)
  - TRAIL_R_LADDER, STRUCT_TRAIL_LB, STRUCT_TRAIL_BUF_PCT, STRUCT_TRAIL_LOOKBACK
  - MIN_RR, MAX_RR, FIB_EXT_1, FIB_EXT_2
"""

import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# KONFIGURASI — Diimpor langsung oleh main.py
# =============================================================================

MIN_RR   = 2.0
MAX_RR   = None  # unlimited RR; target tetap harus struktural/berbasis liquidity

# ── Trail Ladder v3 ──────────────────────────────────────────────────────
# HANYA breakeven di 1.0R. Ini bukan "pengaman profit" — cuma menghilangkan
# risiko begitu trade sudah maju 1R, konsisten dengan permintaan: "Trail
# jangan dipaksa profit". Setelah breakeven tercapai, seluruh pergeseran SL
# selanjutnya murni mengikuti struktur M15 (lihat monitor_position main.py
# yang memakai kandidat "paling protektif" antara ladder ini vs structure —
# karena ladder cuma py 1 rung breakeven, structure yang akan mendominasi
# di hampir semua kasus setelah 1R).
TRAIL_R_LADDER = []  # Trail sepenuhnya struktural M15; tidak ada profit-lock ladder.

# Trailing struktural M15 — INI inti dari Trail (bukan ladder di atas).
# "Ketika harga menyentuh Trail artinya harga tidak kuat mengikuti trend
# sebelumnya" — SL baru = swing M15 terakhir (HL utk BUY / LH utk SELL)
# + buffer supaya tidak kena liquidity sweep biasa.
STRUCT_TRAIL_LB       = 3      # lookback swing_pts saat trailing
STRUCT_TRAIL_BUF_PCT  = 0.0025 # buffer 0.25% di luar swing agar tidak kena LS biasa
STRUCT_TRAIL_LOOKBACK = 60     # candle M15 untuk cari swing trailing

# Adaptive trailing engine V4
TRAIL_ENGINE_VERSION       = "4.0-adaptive"
TRAIL_ARM_R                = 0.80
TRAIL_PROTECT_R            = 1.00
TRAIL_MATURE_R             = 1.50
TRAIL_ATR_MULT             = 2.40
TRAIL_ATR_TIGHT_MULT       = 1.65
TRAIL_MIN_GAP_ATR          = 0.28
TRAIL_BREAK_EVEN_OFFSET_R  = 0.02
TRAIL_WEAKNESS_MODERATE    = 3
TRAIL_WEAKNESS_STRONG      = 6
TRAIL_LIQUIDITY_NEAR_R     = 0.75
TRAIL_VOLUME_EXPANSION     = 1.30
TRAIL_VOLUME_WEAK          = 0.75
TRAIL_MIN_IMPROVEMENT_ATR  = 0.10

# Fibonacci extension TP (level 1.272 dan 1.618 dari impulse leg)
FIB_EXT_1 = 0.272   # 127.2%
FIB_EXT_2 = 0.618   # 161.8%

# ── Inducement & konfluensi ──────────────────────────────────────────────
INDUCEMENT_LOOKBACK  = 40     # candle M15 untuk cari liquidity minor sebelum POI
INDUCEMENT_MINOR_LB  = 2      # lookback swing_pts untuk swing "minor" (inducement)
CONFLUENCE_BONUS      = 2     # bonus skor kalau OB/FVG M15 overlap dengan zona H1
POI_REACTION_LOOKBACK = 16     # candle M15 untuk menunggu reaksi setelah POI HTF
CONFIRMATION_LOOKBACK = 4      # candle M15 yang dipakai untuk displacement terbaru
MIN_DISPLACEMENT_ATR  = 0.25   # body minimum agar candle bukan noise

# Entry-location / RSI timing — ditambahkan setelah audit CAPUSDT.
# Tujuannya bukan membuat bot anti-trade, tetapi mencegah BUY/SELL di lokasi
# yang sudah terlalu dekat sisi salah dari range saat momentum M15 masih
# bergerak melawan entry. RSI dipakai sebagai timing/context, bukan sinyal
# tunggal dan bukan hard overbought/oversold rule.
ENTRY_LOCATION_LOOKBACK = 16     # 4 jam M15 untuk konteks lokasi
ENTRY_CHASE_HIGH = 0.82         # BUY di atas 82% range / SELL di bawah 18% = chase
ENTRY_PREFERRED_BUY = 0.55      # BUY ideal di bawah ~55% range lokal
ENTRY_PREFERRED_SELL = 0.45     # SELL ideal di atas ~45% range lokal
ENTRY_SWING_NEAR_ATR = 0.55     # dekat swing adverse => kurangi kualitas
RSI_TIMING_SLOPE = 1.5          # perubahan RSI minimum agar dianggap bermakna
RSI_BUY_WEAK = 48.0              # BUY + RSI di bawah ini yang masih jatuh = tunggu
RSI_SELL_WEAK = 52.0              # SELL + RSI di atas ini yang masih naik = tunggu
ENTRY_LOCATION_HARD_FLOOR = 30   # kandidat di bawah ini tidak executable

# Must match main.py _validate_signal_before_entry().  Keeping this in the
# strategy engine prevents full_analyze() from returning a candidate that the
# execution gate will immediately reject as ENTRY_TOO_FAR.
MAIN_ENTRY_MAX_ATR = 1.50


# =============================================================================
# UTILITY — Indikator teknikal
# =============================================================================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    lo = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / lo.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr_fn(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _closed_candles(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Keep only candles whose interval has fully closed.

    Binance REST and websocket kline feeds include the currently forming
    candle.  Using it for structure/EMA decisions makes a signal repaint
    during the same scan and was especially harmful on the fast altcoins in
    the supplied trade sample.  DataFrames with a non-datetime index are
    left untouched for compatibility with offline callers/tests.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    idx = out.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    boundary = now.floor(f"{interval_minutes}min")
    # Historical/offline datasets may be entirely older than the current
    # candle. In that case every row is already closed and must be preserved.
    if idx[-1] < boundary:
        return out
    return out.loc[idx < boundary].copy()


def build_df(df: pd.DataFrame, interval_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Tambahkan EMA, RSI, ATR, volume SMA ke DataFrame OHLCV."""
    if df is None or len(df) < 60:
        return None
    df = df.copy()
    if interval_minutes is not None:
        df = _closed_candles(df, interval_minutes)
    if len(df) < 60:
        return None
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200) if len(df) >= 200 else ema(df["close"], 50)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr_fn(df)
    df["vol_sma"] = df["volume"].rolling(20).mean()
    return df.dropna()

def swing_pts(df: pd.DataFrame, lb: int = 5):
    """Swing high & low — dipakai main.py untuk trailing."""
    sh, sl = [], []
    high = df["high"].values
    low = df["low"].values
    n = len(high)
    for i in range(lb, n - lb):
        window_h = high[max(0, i - lb): i + lb + 1]
        window_l = low[max(0, i - lb): i + lb + 1]
        if high[i] == window_h.max():
            sh.append(i)
        if low[i] == window_l.min():
            sl.append(i)
    return sh, sl

def _market_structure(df: pd.DataFrame, sh: list, sl: list) -> str:
    """HH+HL = bullish · LH+LL = bearish · else ranging."""
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

mkt_struct = _market_structure  # alias

def _macro_bias(df_btc_h1: Optional[pd.DataFrame]) -> str:
    """
    Bias arah MARKET SECARA KESELURUHAN (BTC H1 sebagai proxy), bukan
    struktur per-koin. Dipakai untuk meredam sinyal BUY di altcoin saat
    seluruh market lagi risk-off (dan sebaliknya) — analisa 7 Agu 2026:
    4/4 sinyal BUY kena SL, total market cap crypto turun 0.5% & BTC di
    bawah EMA50/EMA200 hari itu; satu-satunya SELL malah menang besar.
    Struktur per-koin (OB/FVG M15) bisa kelihatan valid sendiri, tapi
    kalah lawan gravitasi market kalau macro-nya jelas berlawanan.

    Return "bullish" / "bearish" / "ranging" / "unknown" (kalau data BTC
    tidak dikasih atau tidak cukup — caller HARUS treat "unknown" sebagai
    no-op, jangan menghukum arah manapun).
    """
    if df_btc_h1 is None or len(df_btc_h1) < 60:
        return "unknown"
    try:
        btc = build_df(df_btc_h1, interval_minutes=60)
        if btc is None or len(btc) < 60:
            return "unknown"
        LB = btc.iloc[-1]
        shb, slb = swing_pts(btc, lb=5)
        struct_btc = _market_structure(btc, shb, slb)
        ema_bull = LB["ema9"] > LB["ema21"] > LB["ema50"]
        ema_bear = LB["ema9"] < LB["ema21"] < LB["ema50"]
        if struct_btc == "bullish" or ema_bull:
            return "bullish"
        if struct_btc == "bearish" or ema_bear:
            return "bearish"
        return "ranging"
    except Exception:
        return "unknown"

# Penalti/bonus macro — sengaja MODERAT (bukan veto keras seperti d1_bias
# per-koin di atas), karena ini cuma satu lapis konteks tambahan. Tujuannya
# NGE-REM sinyal yang jelas melawan arus market, BUKAN mematikannya —
# kuantitas sinyal harus tetap terjaga (lihat instruksi awal strategy ini).
MACRO_ALIGN_BONUS   = 8      # searah macro → bonus kecil
MACRO_AGAINST_MULT  = 0.72   # berlawanan macro → skor sisi itu dikali ini (bukan 0)


# =============================================================================
# SMC / ICT DETECTORS (berdasarkan transkrip)
# =============================================================================

def is_zone_fresh(df: pd.DataFrame, top: float, bot: float,
                  formed_idx: int, end_idx: Optional[int] = None,
                  direction: Optional[str] = None) -> bool:
    """True jika zona masih aktif, bukan sekadar belum pernah disentuh.

    Untuk zona berarah, retest tidak otomatis membatalkan zona. Invalidasi
    terjadi bila candle close menembus sisi luar zona. Tanpa ``direction``
    fungsi mempertahankan perilaku konservatif lama untuk caller generik.
    """
    if formed_idx is None or formed_idx + 2 >= len(df):
        return True
    start = formed_idx + 2
    end = end_idx if end_idx is not None else len(df) - 1
    if start >= end:
        return True
    sub = df.iloc[start:end]
    if direction == "bull":
        return not bool((sub["close"] < bot).any())
    if direction == "bear":
        return not bool((sub["close"] > top).any())
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)

def fib_position(price: float, swing_low: float, swing_high: float) -> float:
    """Posisi harga dalam range 0–1, 0 = swing_low, 1 = swing_high."""
    rng = swing_high - swing_low
    if rng <= 0:
        return 0.5
    return max(0.0, min(1.0, (price - swing_low) / rng))

def is_in_ote(price: float, swing_low: float, swing_high: float,
              direction: str) -> bool:
    """
    OTE (Optimal Trade Entry) — zona 61.8%–78.6% retracement.
    Bull: fib_position antara 0.214 dan 0.382
    Bear: fib_position antara 0.618 dan 0.786
    """
    if swing_high <= swing_low:
        return False
    r = fib_position(price, swing_low, swing_high)
    if direction == "bull":
        return 0.214 <= r <= 0.382
    else:
        return 0.618 <= r <= 0.786

def detect_fvg(df: pd.DataFrame, direction: str, lb: int = 50) -> list:
    """
    Fair Value Gap (imbalance) — 3‑candle pattern.
    Hanya return zona yang FRESH.
    """
    sub = df.iloc[-lb:]
    base = len(df) - len(sub)
    out = []
    for i in range(len(sub) - 2):
        c0, c2 = sub.iloc[i], sub.iloc[i + 2]
        gap = None
        if direction == "bull" and c2["low"] > c0["high"]:
            gap = {"top": float(c2["low"]), "bot": float(c0["high"])}
        elif direction == "bear" and c2["high"] < c0["low"]:
            gap = {"top": float(c0["low"]), "bot": float(c2["high"])}
        if gap:
            gap["mid"] = (gap["top"] + gap["bot"]) / 2
            gap["idx"] = base + i + 2
            gap["is_fresh"] = is_zone_fresh(
                df, gap["top"], gap["bot"], gap["idx"], direction=direction
            )
            out.append(gap)
    fresh = [f for f in out if f["is_fresh"]]
    return fresh[-3:] if fresh else []

def detect_order_block(df: pd.DataFrame, direction: str, lb: int = 60,
                       sh: Optional[list] = None, sl: Optional[list] = None) -> list:
    """
    Order Block berkualitas (transkrip 1, 14, 26).
    Filter:
      - zona fresh
      - Fibonacci diskon (bull ≤ 0.618) / premium (bear ≥ 0.382)
      - kualitas: impulse kuat, FVG setelahnya, BOS, recency.
    Return list max 3, urut kualitas tertinggi.
    """
    is_demand = direction == "bull"
    sub = df.iloc[-lb:]
    base = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean() or 1e-8

    # Fibonacci context
    fib_sh = float(df["high"].iloc[sh[-1]]) if (sh and len(sh) > 0) else None
    fib_sl = float(df["low"].iloc[sl[-1]]) if (sl and len(sl) > 0) else None

    # BOS global
    has_bos_global = False
    if sh and sl:
        if is_demand and len(sh) >= 2:
            has_bos_global = float(df["high"].iloc[-1]) > float(df["high"].iloc[sh[-2]])
        elif not is_demand and len(sl) >= 2:
            has_bos_global = float(df["low"].iloc[-1]) < float(df["low"].iloc[sl[-2]])

    zones = []
    for i in range(1, len(sub) - 3):
        c, nx = sub.iloc[i], sub.iloc[i + 1]
        # Pola trigger + impulse
        if is_demand:
            if not (c["close"] < c["open"] and nx["close"] > nx["open"]):
                continue
        else:
            if not (c["close"] > c["open"] and nx["close"] < nx["open"]):
                continue

        impulse_body = abs(nx["close"] - nx["open"])
        if impulse_body < avg_body * 1.2:
            continue

        ob_top = float(max(c["open"], c["close"]))
        ob_bot = float(min(c["open"], c["close"]))
        df_idx = base + i

        if not is_zone_fresh(
            df, ob_top, ob_bot, df_idx, direction=direction
        ):
            continue

        q = 0
        if impulse_body >= avg_body * 1.5:
            q += 1
        if impulse_body >= avg_body * 2.5:
            q += 1

        # FVG setelah OB
        if i + 2 < len(sub):
            c2 = sub.iloc[i + 2]
            if is_demand and c2["low"] > c["high"]:
                q += 1
            elif not is_demand and c2["high"] < c["low"]:
                q += 1

        if has_bos_global:
            q += 1

        # Fibonacci diskon/premium (kunci dari transkrip 1)
        if fib_sh is not None and fib_sl is not None:
            ob_mid = (ob_top + ob_bot) / 2
            fib_r = fib_position(ob_mid, fib_sl, fib_sh)
            if is_demand and fib_r <= 0.618:
                q += 1
            elif not is_demand and fib_r >= 0.382:
                q += 1

        if df_idx >= len(df) - 20:
            q += 1

        if q >= 2:
            zones.append({
                "top": ob_top,
                "bot": ob_bot,
                "mid": (ob_top + ob_bot) / 2,
                "idx": df_idx,
                "quality": q,
                "has_bos": has_bos_global,
            })

    zones.sort(key=lambda z: (-z["quality"], -z["idx"]))
    return zones[:3]

def detect_choch(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """Change of Character (transkrip 17, 19, 20)."""
    result = {"bullish_choch": False, "bearish_choch": False}
    if len(sh) < 2 or len(sl) < 2:
        return result
    close = float(df["close"].iloc[-1])
    prev_high = float(df["high"].iloc[sh[-2]])
    last_high = float(df["high"].iloc[sh[-1]])
    prev_low = float(df["low"].iloc[sl[-2]])
    last_low = float(df["low"].iloc[sl[-1]])
    struct = _market_structure(df, sh, sl)

    # A ChoCH is a break of the *opposite* protected swing.  The old
    # implementation compared a bullish close with prev_low (and a bearish
    # close with prev_low), which is true on most ordinary candles and added
    # +20 points to both random pullbacks and continuations.
    if struct == "bearish" and close > last_high:
        result["bullish_choch"] = True
    if struct == "bullish" and close < last_low:
        result["bearish_choch"] = True

    # Raw reversal: require a close beyond the swing, never only a wick.
    if last_high > prev_high and last_low > prev_low and close > last_high:
        result["bullish_choch"] = True
    if last_high < prev_high and last_low < prev_low and close < last_low:
        result["bearish_choch"] = True
    return result

def detect_bos(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """Break of Structure (transkrip 17)."""
    result = {"bullish_bos": False, "bearish_bos": False}
    if len(sh) < 2 or len(sl) < 2:
        return result

    # Use the latest confirmed swing and a candle close.  Looking at the
    # previous swing plus the current wick made old structure look broken
    # long after the actual break had happened.
    close = float(df["close"].iloc[-1])
    last_high = float(df["high"].iloc[sh[-1]])
    last_low = float(df["low"].iloc[sl[-1]])
    prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else close
    result["bullish_bos"] = close > last_high and prev_close <= last_high
    result["bearish_bos"] = close < last_low and prev_close >= last_low
    return result

def detect_cisd(df: pd.DataFrame, lb: int = 8) -> dict:
    """Change In State of Delivery (transkrip 29)."""
    result = {"bullish_cisd": False, "bearish_cisd": False}
    if len(df) < lb + 1:
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
                bear_mid = (opens[first_idx] + closes[first_idx]) / 2
                if closes[-1] > bear_mid:
                    result["bullish_cisd"] = True
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
                bull_mid = (opens[first_idx] + closes[first_idx]) / 2
                if closes[-1] < bull_mid:
                    result["bearish_cisd"] = True
    return result

def detect_liquidity_sweep(df: pd.DataFrame, sh: list, sl: list,
                           direction: str) -> dict:
    """
    Liquidity Sweep (transkrip 4, 9, 15, 18).
    Valid: wick menembus swing, tapi close kembali di atas/bawah level.
    """
    result = {"type": "none", "level": None, "strength": 0}
    if direction == "bull" and sl:
        level = float(df["low"].iloc[sl[-1]])
        last_low = float(df["low"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_low < level and last_close > level:
            depth = (level - last_low) / max(level, 1e-10)
            result = {
                "type": "sweep",
                "level": level,
                "strength": min(3, int(depth / 0.002) + 1),
            }
    elif direction == "bear" and sh:
        level = float(df["high"].iloc[sh[-1]])
        last_high = float(df["high"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_high > level and last_close < level:
            depth = (last_high - level) / max(level, 1e-10)
            result = {
                "type": "sweep",
                "level": level,
                "strength": min(3, int(depth / 0.002) + 1),
            }
    return result

def detect_inducement(df: pd.DataFrame, direction: str,
                      lb: int = INDUCEMENT_LOOKBACK) -> dict:
    """
    Inducement (transkrip "How the Market Traps Traders with Inducement" &
    "trading strategy using inducement [SNIPER ENTRY]").

    Konsep: sebelum harga bereaksi di POI "asli" (OB/FVG yang lebih dalam),
    market sering menyapu dulu liquidity pool MINOR (swing kecil) yang
    memancing early entry ("inducement"). Kalau minor pool itu SUDAH disapu
    (wick tembus, close kembali ke sisi yang benar) sebelum harga mendekati
    POI kita, itu tanda bagus — bukan validasi wajib, cuma BONUS confidence,
    supaya kuantitas sinyal tidak berkurang untuk koin yang tidak
    menunjukkan pola ini secara eksplisit.

    Return: {"found": bool, "level": float|None, "swept": bool}
    """
    out = {"found": False, "level": None, "swept": False}
    if df is None or len(df) < lb + INDUCEMENT_MINOR_LB * 2 + 1:
        return out
    sub = df.iloc[-lb:].reset_index(drop=True)
    sh_m, sl_m = swing_pts(sub, lb=INDUCEMENT_MINOR_LB)
    try:
        if direction == "bull" and sl_m:
            idx = sl_m[-1]
            if idx >= len(sub) - 2:
                return out
            level = float(sub["low"].iloc[idx])
            after = sub.iloc[idx + 1:]
            swept = bool((after["low"] < level).any() and
                         float(after["close"].iloc[-1]) > level)
            out = {"found": True, "level": level, "swept": swept}
        elif direction == "bear" and sh_m:
            idx = sh_m[-1]
            if idx >= len(sub) - 2:
                return out
            level = float(sub["high"].iloc[idx])
            after = sub.iloc[idx + 1:]
            swept = bool((after["high"] > level).any() and
                         float(after["close"].iloc[-1]) < level)
            out = {"found": True, "level": level, "swept": swept}
    except Exception:
        return {"found": False, "level": None, "swept": False}
    return out

def zones_overlap(a_top: float, a_bot: float, b_top: float, b_bot: float) -> bool:
    """True jika dua rentang harga [bot, top] saling overlap (dipakai untuk
    cek konfluensi OB/FVG M15 dengan zona H1 — 'How to Choose the Best Order
    Block When All Zones Look Valid')."""
    lo = max(min(a_bot, a_top), min(b_bot, b_top))
    hi = min(max(a_bot, a_top), max(b_bot, b_top))
    return lo <= hi

def detect_equal_highs_lows(df: pd.DataFrame, kind: str = "high",
                            lb: int = 80, tol: float = 0.003) -> list:
    """Equal Highs/Lows (transkrip 15, 24)."""
    sub = df.iloc[-lb:]
    vals = sub["high"] if kind == "high" else sub["low"]
    clusters, visited = [], set()
    for i in range(len(vals)):
        if i in visited:
            continue
        group = [float(vals.iloc[i])]
        for j in range(i + 1, len(vals)):
            if abs(vals.iloc[i] - vals.iloc[j]) / max(abs(float(vals.iloc[i])), 1e-10) < tol:
                group.append(float(vals.iloc[j]))
                visited.add(j)
        if len(group) >= 2:
            clusters.append(sum(group) / len(group))
    return sorted(clusters)

def detect_rsi_divergence(df: pd.DataFrame, direction: str, lb: int = 30) -> dict:
    """RSI Divergence (transkrip 19)."""
    result = {"bull_div": False, "bear_div": False, "strong": False}
    if len(df) < lb + 1 or "rsi" not in df.columns:
        return result
    sub = df.iloc[-lb:]
    price = sub["close"].values
    rsi_v = sub["rsi"].values
    n = len(price)
    lb3 = 3

    lows = [i for i in range(lb3, n - lb3)
            if price[i] == min(price[max(0, i - lb3): i + lb3 + 1])]
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if price[i2] < price[i1] and rsi_v[i2] > rsi_v[i1]:
            result["bull_div"] = True
            if rsi_v[i2] < 35:
                result["strong"] = True

    highs = [i for i in range(lb3, n - lb3)
             if price[i] == max(price[max(0, i - lb3): i + lb3 + 1])]
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

def detect_failed_retest(df: pd.DataFrame, sh: list, sl: list,
                         atr: float) -> dict:
    """Failed retest (untuk kompatibilitas)."""
    result = {"failed_retest_sell": False, "failed_retest_buy": False}
    if len(df) < 3 or not sh or not sl:
        return result
    L = df.iloc[-1]
    P = df.iloc[-2]
    if len(sh) >= 2:
        res = float(df["high"].iloc[sh[-2]])
        if P["high"] >= res - atr * 0.5 and L["close"] < res - atr * 0.3 and L["close"] < L["open"]:
            result["failed_retest_sell"] = True
    if len(sl) >= 2:
        sup = float(df["low"].iloc[sl[-2]])
        if P["low"] <= sup + atr * 0.5 and L["close"] > sup + atr * 0.3 and L["close"] > L["open"]:
            result["failed_retest_buy"] = True
    return result


def detect_entry_confirmation(df: pd.DataFrame, direction: str, atr: float,
                              lb: int = CONFIRMATION_LOOKBACK) -> dict:
    """Konfirmasi displacement yang baru, bukan sekadar lokasi entry.

    OTE, RSI divergence, discount/premium, dan sweep menjelaskan lokasi atau
    kemungkinan reaksi. Mereka tidak membuktikan bahwa buyer/seller sudah
    mengambil alih. Konfirmasi ini membutuhkan candle close searah yang
    menembus range pendek sebelumnya dengan body yang cukup besar.
    """
    out = {
        "confirmed": False,
        "direction": direction,
        "kind": "none",
        "idx": None,
        "body_atr": 0.0,
    }
    if df is None or len(df) < max(lb + 2, 8):
        return out

    last = df.iloc[-1]
    prior = df.iloc[-(lb + 1):-1]
    body = abs(float(last["close"]) - float(last["open"]))
    local_atr = max(float(atr), 1e-10)
    body_atr = body / local_atr
    out["body_atr"] = round(body_atr, 3)

    recent_bodies = (df["close"] - df["open"]).abs().iloc[-(lb + 5):-1]
    median_body = float(recent_bodies.median()) if not recent_bodies.empty else 0.0
    min_body = max(local_atr * MIN_DISPLACEMENT_ATR, median_body * 0.8)

    if direction == "bull":
        confirmed = (
            float(last["close"]) > float(last["open"])
            and float(last["close"]) > float(prior["high"].max())
            and body >= min_body
        )
    else:
        confirmed = (
            float(last["close"]) < float(last["open"])
            and float(last["close"]) < float(prior["low"].min())
            and body >= min_body
        )

    if confirmed:
        out.update({
            "confirmed": True,
            "kind": "displacement_close",
            "idx": len(df) - 1,
        })
    return out


def _collect_htf_poi_zones(h1: pd.DataFrame, direction: str,
                           score_ctx: dict) -> list:
    """Kumpulkan POI H1 searah untuk validasi reaksi sebelum entry."""
    zones = []
    try:
        for z in detect_order_block(
            h1, direction, lb=80,
            sh=score_ctx.get("sh1", []),
            sl=score_ctx.get("sl1", []),
        ):
            zones.append({
                "top": float(z["top"]),
                "bot": float(z["bot"]),
                "kind": "ob_h1",
            })
        for f in detect_fvg(h1, direction, lb=60):
            zones.append({
                "top": float(f["top"]),
                "bot": float(f["bot"]),
                "kind": "fvg_h1",
            })
    except Exception:
        return []
    return zones


def _recent_poi_reaction(m15: pd.DataFrame, zones: list, direction: str,
                         atr: float,
                         lookback: int = POI_REACTION_LOOKBACK) -> bool:
    """Pastikan POI HTF sudah disentuh dan mendapat rejection baru.

    Retest saja tidak cukup. Setelah harga masuk POI, harus ada close kembali
    keluar dari sisi reaksi zona. Ini mencegah pending order dipasang hanya
    karena OB/FVG lama masih terlihat fresh.
    """
    if m15 is None or m15.empty or not zones:
        return False
    start = max(0, len(m15) - lookback)
    sub = m15.iloc[start:]
    body_floor = max(float(atr) * MIN_DISPLACEMENT_ATR, 1e-10)

    for zone in zones:
        top, bot = float(zone["top"]), float(zone["bot"])
        for i in range(len(sub)):
            candle = sub.iloc[i]
            touched = float(candle["low"]) <= top and float(candle["high"]) >= bot
            if not touched:
                continue
            for j in range(i, min(i + 5, len(sub))):
                reaction = sub.iloc[j]
                body = abs(float(reaction["close"]) - float(reaction["open"]))
                if direction == "bull":
                    ok = (
                        float(reaction["close"]) > top
                        and float(reaction["close"]) > float(reaction["open"])
                        and body >= body_floor
                    )
                else:
                    ok = (
                        float(reaction["close"]) < bot
                        and float(reaction["close"]) < float(reaction["open"])
                        and body >= body_floor
                    )
                if ok:
                    return True
    return False


# =============================================================================
# SCORING — Confidence global tanpa bias sesi
# =============================================================================

def score_direction(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                    df_d1: Optional[pd.DataFrame] = None,
                    df_btc_h1: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """
    Tentukan arah dan skor confidence global (tanpa bias sesi).
    Faktor scoring (max ~165):
      - D1 bias (struktur/EMA)         +20
      - H1 structure                   +20
      - H1 ChoCH                       +10
      - H1 BOS                         +5
      - H1 EMA stack                   +10
      - M15 ChoCH                      +20
      - M15 BOS                        +10
      - M15 CISD                       +15
      - M15 Liquidity Sweep            +15
      - M15 OTE                        +10
      - M15 Fibonacci diskon/premium   +10
      - M15 RSI divergence             +10 (+5 if strong)
      - M15 failed retest              +10
    Penalti: jika setup M15 berlawanan dengan bias D1 → kurangi 50%.
    """
    h1 = build_df(df_h1, interval_minutes=60)
    m15 = build_df(df_m15, interval_minutes=15)
    if h1 is None or m15 is None:
        return None

    L1 = h1.iloc[-1]
    L15 = m15.iloc[-1]
    atr = max(float(L15["atr"]),
              float(L1["atr"]) / 4,
              float(L15["close"]) * 0.003)

    sh1, sl1 = swing_pts(h1, lb=5)
    sh15, sl15 = swing_pts(m15, lb=5)
    struct_h1 = _market_structure(h1, sh1, sl1)

    # ── D1 bias ──────────────────────────────────────────────────
    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            d1 = build_df(df_d1, interval_minutes=1440)
        else:
            d1 = build_df(
                df_h1.resample("1D").agg(
                    {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
                ).dropna()
            )
        if d1 is not None and len(d1) >= 10:
            LD = d1.iloc[-1]
            shd, sld = swing_pts(d1, lb=3)
            sd1 = _market_structure(d1, shd, sld)
            bull_d1 = sd1 == "bullish" or (LD["ema9"] > LD["ema21"] > LD["ema50"])
            bear_d1 = sd1 == "bearish" or (LD["ema9"] < LD["ema21"] < LD["ema50"])
            if bull_d1:
                d1_bias = "bullish"
            elif bear_d1:
                d1_bias = "bearish"
    except Exception:
        pass

    # ── H1 indikator ─────────────────────────────────────────────
    ema_h1_bull = L1["ema9"] > L1["ema21"] > L1["ema50"]
    ema_h1_bear = L1["ema9"] < L1["ema21"] < L1["ema50"]
    choch_h1 = detect_choch(h1, sh1, sl1)
    bos_h1 = detect_bos(h1, sh1, sl1)

    # ── M15 indikator ─────────────────────────────────────────────
    choch_m15 = detect_choch(m15, sh15, sl15)
    bos_m15 = detect_bos(m15, sh15, sl15)
    cisd_m15 = detect_cisd(m15, lb=8)
    liq_bull = detect_liquidity_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_sweep(m15, sh15, sl15, "bear")
    fr_m15 = detect_failed_retest(m15, sh15, sl15, atr)

    # ── Fibonacci M15 context ─────────────────────────────────────
    fib_sh = float(m15["high"].iloc[sh15[-1]]) if sh15 else None
    fib_sl = float(m15["low"].iloc[sl15[-1]]) if sl15 else None
    fib_r = fib_position(float(L15["close"]), fib_sl or 0, fib_sh or 1) \
            if (fib_sh and fib_sl) else 0.5

    ote_bull = is_in_ote(float(L15["close"]), fib_sl or 0, fib_sh or 1, "bull") if (fib_sh and fib_sl) else False
    ote_bear = is_in_ote(float(L15["close"]), fib_sl or 0, fib_sh or 1, "bear") if (fib_sh and fib_sl) else False

    in_discount = fib_r < 0.45
    in_premium = fib_r > 0.55

    rdiv_bull = detect_rsi_divergence(m15, "bull", lb=30)
    rdiv_bear = detect_rsi_divergence(m15, "bear", lb=30)

    # ── Inducement dan konfirmasi entry ────────────────────────────
    induce_bull = detect_inducement(m15, "bull")
    induce_bear = detect_inducement(m15, "bear")
    confirm_bull = detect_entry_confirmation(m15, "bull", atr)
    confirm_bear = detect_entry_confirmation(m15, "bear", atr)

    # ── Score BULL ───────────────────────────────────────────────
    bull = 0
    if d1_bias == "bullish":                    bull += 20
    if struct_h1 == "bullish":                  bull += 20
    elif struct_h1 == "ranging":                bull += 5
    if choch_h1["bullish_choch"]:               bull += 10
    if bos_h1["bullish_bos"]:                   bull += 5
    if ema_h1_bull:                             bull += 10
    if choch_m15["bullish_choch"]:              bull += 20
    if bos_m15["bullish_bos"]:                  bull += 10
    if cisd_m15["bullish_cisd"]:                bull += 15
    if liq_bull["type"] == "sweep":             bull += 15
    if ote_bull:                                bull += 10
    if in_discount:                             bull += 10
    if rdiv_bull.get("bull_div"):
        bull += 10 + (5 if rdiv_bull.get("strong") else 0)
    if fr_m15["failed_retest_buy"]:             bull += 10
    if induce_bull.get("swept"):                bull += 8

    # ── Score BEAR ───────────────────────────────────────────────
    bear = 0
    if d1_bias == "bearish":                    bear += 20
    if struct_h1 == "bearish":                  bear += 20
    elif struct_h1 == "ranging":                bear += 5
    if choch_h1["bearish_choch"]:               bear += 10
    if bos_h1["bearish_bos"]:                   bear += 5
    if ema_h1_bear:                             bear += 10
    if choch_m15["bearish_choch"]:              bear += 20
    if bos_m15["bearish_bos"]:                  bear += 10
    if cisd_m15["bearish_cisd"]:                bear += 15
    if liq_bear["type"] == "sweep":             bear += 15
    if ote_bear:                                bear += 10
    if in_premium:                              bear += 10
    if rdiv_bear.get("bear_div"):
        bear += 10 + (5 if rdiv_bear.get("strong") else 0)
    if fr_m15["failed_retest_sell"]:            bear += 10
    if induce_bear.get("swept"):                bear += 8

    # ── Macro bias (BTC H1) — konteks tambahan, bukan pengganti HTF koin ──
    macro_bias = _macro_bias(df_btc_h1)
    if macro_bias == "bullish":
        bull += MACRO_ALIGN_BONUS
        bear = int(bear * MACRO_AGAINST_MULT)
    elif macro_bias == "bearish":
        bear += MACRO_ALIGN_BONUS
        bull = int(bull * MACRO_AGAINST_MULT)
    # macro_bias == "ranging"/"unknown" → tidak ada perubahan sama sekali

    # ── HTF direction gate ─────────────────────────────────────────
    # D1 dan H1 bukan lagi bonus independen yang bisa dikalahkan oleh RSI,
    # OTE, atau sweep. Jika keduanya berlawanan, market sedang tidak memiliki
    # bias yang cukup jelas untuk limit order baru; tunggu alignment berikutnya.
    h1_bias = "bullish" if struct_h1 == "bullish" else (
        "bearish" if struct_h1 == "bearish" else "neutral"
    )
    if d1_bias in ("bullish", "bearish") and h1_bias in ("bullish", "bearish"):
        htf_bias = d1_bias if d1_bias == h1_bias else "conflict"
    elif d1_bias in ("bullish", "bearish"):
        htf_bias = d1_bias
    elif h1_bias in ("bullish", "bearish"):
        htf_bias = h1_bias
    else:
        htf_bias = "neutral"

    if htf_bias == "bullish":
        direction = "bull"
    elif htf_bias == "bearish":
        direction = "bear"
    else:
        # Konflik/ranging bukan auto-reject. Pilih sisi dengan confluence
        # terbesar dan biarkan confidence yang menghukum setup yang lemah.
        direction = "bull" if bull >= bear else "bear"
    raw = bull if direction == "bull" else bear
    # ── FIX BUG KRITIS (scan 50 koin selalu "Tidak ada setup valid") ──
    # MAX_SCORE=173 lama = jumlah SEMUA bonus sekaligus (D1+H1+ChoCH+BOS+
    # EMA+M15 ChoCH+BOS+CISD+Liquidity+OTE+Fib+RSIdiv+failed_retest+
    # inducement, semua di arah yang sama). Dites empiris lewat 300
    # skenario pasar acak (termasuk tren paling bersih): skor mentah
    # MAKSIMUM yang pernah tercapai cuma ~88, rata-rata ~40, p99 ~75 —
    # skala 173 bikin confidence yang ditampilkan jauh lebih rendah dari
    # kualitas setup sebenarnya (bahkan setup terbaik pun mentok ~50%).
    # Direkalibrasi ke 100 (dekat batas atas realistis) supaya angka
    # confidence yang tampil BERARTI sesuatu — setup kuat bisa mendekati
    # 80-90%, bukan mentok di 50%.
    MAX_SCORE = 100
    confidence = min(int(raw / MAX_SCORE * 100), 99)
    if htf_bias == "conflict":
        confidence = max(0, confidence - 12)
    elif htf_bias == "neutral":
        confidence = max(0, confidence - 5)

    # Direction quality is separate from the absolute score.  A high score
    # assembled from unrelated bonuses is not enough when H1 and M15 disagree.
    edge = abs(bull - bear)
    m15_struct = _market_structure(m15, sh15, sl15)
    selected_choch = choch_m15["bullish_choch"] if direction == "bull" else choch_m15["bearish_choch"]
    selected_bos = bos_m15["bullish_bos"] if direction == "bull" else bos_m15["bearish_bos"]
    selected_cisd = cisd_m15["bullish_cisd"] if direction == "bull" else cisd_m15["bearish_cisd"]
    selected_confirm = (
        confirm_bull if direction == "bull" else confirm_bear
    )
    selected_sweep = (
        liq_bull if direction == "bull" else liq_bear
    ).get("type") == "sweep"
    # Sweep hanya lokasi liquidity, bukan bukti arah. Trigger utama harus
    # berupa structure/displacement yang terjadi pada candle terbaru.
    trigger_count = sum(bool(x) for x in (
        selected_choch, selected_bos, selected_cisd,
        selected_confirm.get("confirmed"),
    ))

    return {
        "direction": direction,
        "confidence": confidence,
        "bull_score": bull,
        "bear_score": bear,
        "direction_edge": edge,
        "m15_struct": m15_struct,
        "trigger_count": trigger_count,
        "htf_bias": htf_bias,
        "h1_bias": h1_bias,
        "entry_confirmation": selected_confirm,
        "entry_confirmation_bull": confirm_bull,
        "entry_confirmation_bear": confirm_bear,
        "selected_sweep": selected_sweep,
        "price": float(L15["close"]),
        "atr": atr,
        "struct_h1": struct_h1,
        "d1_bias": d1_bias,
        "macro_bias": macro_bias,
        "choch_m15": choch_m15,
        "choch_h1": choch_h1,
        "cisd_m15": cisd_m15,
        "bos_m15": bos_m15,
        "bos_h1": bos_h1,
        "m15_rsi": float(L15["rsi"]),
        "m15_rsi_slope": float(L15["rsi"] - m15["rsi"].iloc[-2]) if len(m15) >= 2 else 0.0,
        "m15_relative_volume": _relative_volume(m15, 20) if len(m15) >= 23 else 1.0,
        "m15_momentum_aligned": bool((direction == "bull" and _momentum_context(m15).get("bull")) or (direction == "bear" and _momentum_context(m15).get("bear"))),
        "m15_divergence_against": bool((direction == "bull" and rdiv_bear.get("bear_div")) or (direction == "bear" and rdiv_bull.get("bull_div"))),
        "failed_retest": fr_m15,
        "liquidity_bull": liq_bull,
        "liquidity_bear": liq_bear,
        "inducement_bull": induce_bull,
        "inducement_bear": induce_bear,
        "sh15": sh15, "sl15": sl15,
        "sh1": sh1, "sl1": sl1,
        "fib_r": round(fib_r, 3),
        "ote_bull": ote_bull,
        "ote_bear": ote_bear,
    }


# =============================================================================
# CONFIDENCE CALIBRATION V3.0 — TRADEABILITY
# =============================================================================
CONFIDENCE_MODEL_VERSION = "4.0_calibrated_ready"

def _rr_band(rr: float) -> str:
    rr = float(rr or 0.0)
    if rr < 3.0:
        return "2-3R"
    if rr < 5.0:
        return "3-5R"
    if rr < 7.0:
        return "5-7R"
    if rr < 9.0:
        return "7-9R"
    return "9R+"


def calibrate_confidence_from_history(history, base_score: float, decision: str,
                                      entry_label: str, rr: float, setup_quality: float) -> tuple[int, dict]:
    """Empirical probability calibration hook.

    `history` is optional and must contain closed trades with a numeric PnL or
    a boolean `profitable`. With fewer observations the model falls back to the
    structural score; it never pretends that a tiny sample is a precise probability.
    Bayesian smoothing keeps sparse buckets conservative.
    """
    try:
        rows = [r for r in (history or []) if isinstance(r, dict)]
        if not rows:
            return int(max(0, min(99, round(base_score)))), {
                "calibration": "structural_fallback", "samples": 0,
                "empirical_confidence": None, "calibration_weight": 0.0
            }

        rrband = _rr_band(rr)
        direction = str(decision or "").upper()
        label = str(entry_label or "").lower()
        candidates = []
        for r in rows:
            d = str(r.get("decision", r.get("direction", ""))).upper()
            el = str(r.get("entry_label", "")).lower()
            rb = _rr_band(float(r.get("rr", 0) or 0))
            profitable = r.get("profitable")
            if profitable is None:
                pnl = r.get("pnl", r.get("pnl_usd", r.get("net_pnl")))
                try:
                    profitable = float(pnl) > 0
                except Exception:
                    continue
            # Hierarchical relevance: same direction + label + RR band is strongest.
            match = 0.0
            if d == direction: match += 0.35
            if el == label: match += 0.30
            if rb == rrband: match += 0.25
            if match >= 0.35:
                candidates.append((match, bool(profitable)))

        if not candidates:
            candidates = [(0.25, bool(r.get("profitable"))) for r in rows if "profitable" in r]

        if not candidates:
            return int(max(0, min(99, round(base_score)))), {
                "calibration": "structural_fallback", "samples": 0,
                "empirical_confidence": None, "calibration_weight": 0.0
            }

        # Weighted Beta posterior. Prior is intentionally neutral, not a claim of
        # 60% win probability. More observations are required before it dominates.
        alpha = beta = 2.0
        weighted_n = 0.0
        for w, win in candidates:
            alpha += w if win else 0.0
            beta += 0.0 if win else w
            weighted_n += w
        empirical = 100.0 * alpha / max(alpha + beta, 1e-9)
        weight = min(0.70, weighted_n / 40.0)
        calibrated = (1.0 - weight) * float(base_score) + weight * empirical
        calibrated = max(0, min(99, int(round(calibrated))))
        return calibrated, {
            "calibration": "empirical_bayesian",
            "samples": len(candidates),
            "weighted_samples": round(weighted_n, 2),
            "empirical_confidence": round(empirical, 2),
            "calibration_weight": round(weight, 3),
            "rr_band": rrband,
            "direction": direction,
            "entry_label": label,
        }
    except Exception:
        return int(max(0, min(99, round(base_score)))), {
            "calibration": "structural_fallback_error", "samples": 0,
            "empirical_confidence": None, "calibration_weight": 0.0
        }


def _confidence_band(conf: int) -> str:
    if conf >= 85:
        return "ELITE"
    if conf >= 75:
        return "VERY_STRONG"
    if conf >= 65:
        return "STRONG"
    if conf >= 55:
        return "NORMAL"
    if conf >= 45:
        return "WEAK"
    return "VERY_WEAK"


def _calibrate_confidence(base_confidence: int, score_ctx: dict, loc: dict,
                          candidate: dict, rr: float, htf_poi: list,
                          poi_reacted: bool, tp_diag: Optional[dict] = None) -> tuple[int, int, dict]:
    """Confidence V3.0 = tradeability, not raw confluence count.

    Direction establishes the thesis, but entry location and current-move health
    have larger influence on the final number. RR is payoff, not probability.
    """
    direction = score_ctx.get("direction", "bull")
    htf_bias = score_ctx.get("htf_bias", "neutral")
    m15_struct = score_ctx.get("m15_struct", "ranging")
    h1_bias = score_ctx.get("h1_bias", "neutral")
    trigger_count = int(score_ctx.get("trigger_count", 0) or 0)
    selected_confirm = score_ctx.get("entry_confirmation", {}) or {}
    selected_sweep = bool(score_ctx.get("selected_sweep"))
    loc_score = int(loc.get("location_score", 50) or 50)
    candidate_score = float(candidate.get("score", 0) or 0)
    entry_label = str(candidate.get("label", ""))

    # 1) DIRECTION / STRUCTURAL THESIS: 0-25
    direction_q = 0.0
    if h1_bias == ("bullish" if direction == "bull" else "bearish"):
        direction_q += 10
    elif h1_bias == "neutral":
        direction_q += 6
    if htf_bias == ("bullish" if direction == "bull" else "bearish"):
        direction_q += 9
    elif htf_bias == "neutral":
        direction_q += 5
    elif htf_bias == "conflict":
        direction_q -= 4
    if m15_struct == ("bullish" if direction == "bull" else "bearish"):
        direction_q += 6
    elif m15_struct == "ranging":
        direction_q += 3
    direction_q = max(0.0, min(25.0, direction_q))

    # 2) ENTRY QUALITY: 0-30 -- deliberately larger than raw direction.
    entry_q = 0.0
    entry_q += max(0.0, min(18.0, (loc_score - 30) * 0.38))
    if htf_poi:
        entry_q += 3.0
    if poi_reacted:
        entry_q += 4.0
    if entry_label in ("ob", "ob_retest"):
        entry_q += 2.0
    elif entry_label in ("fvg", "fvg_retest"):
        entry_q += 1.5
    # Chase / bad location penalties.
    if loc_score < 45:
        entry_q -= 4
    if loc.get("location_state") in {"chase", "bad_location", "premium_chase", "discount_chase"}:
        entry_q -= 4
    entry_q = max(0.0, min(30.0, entry_q))

    # 3) CURRENT MOVE HEALTH: 0-25
    move_q = 12.0
    rsi_val = float(loc.get("rsi", score_ctx.get("m15_rsi", 50.0)) or 50.0)
    rv = float(score_ctx.get("m15_relative_volume", 1.0) or 1.0)
    rsi_timing = str(loc.get("rsi_timing", "unknown"))
    rsi_slope = float(loc.get("rsi_slope", score_ctx.get("m15_rsi_slope", 0.0)) or 0.0)
    momentum_aligned = bool(score_ctx.get("m15_momentum_aligned", False))
    divergence = bool(score_ctx.get("m15_divergence_against", False))

    if momentum_aligned:
        move_q += 6
    if rv >= 1.30 and momentum_aligned:
        move_q += 3
    elif rv < 0.70 and rr >= 2.0:
        move_q -= 2
    if divergence:
        move_q -= 5
    if direction == "bull" and rsi_val >= 72:
        move_q -= 4
    elif direction == "bear" and rsi_val <= 28:
        move_q -= 4
    if rsi_timing in {"rising", "falling", "favorable"}:
        move_q += 1
    if direction == "bull" and rsi_slope < -1.5:
        move_q -= 2
    if direction == "bear" and rsi_slope > 1.5:
        move_q -= 2
    move_q = max(0.0, min(25.0, move_q))

    # 4) TRIGGER / CONFIRMATION: 0-10
    trigger_q = min(10.0, 2.0 + min(trigger_count, 3) * 1.5)
    if selected_confirm.get("confirmed"):
        trigger_q += 2.0
    if selected_sweep:
        trigger_q += 0.5
    trigger_q = max(0.0, min(10.0, trigger_q))

    # 5) TARGET / PAYOFF QUALITY: 0-10. RR is not treated as win probability.
    target_q = 4.0
    selected = (tp_diag or {}).get("selected") if isinstance(tp_diag, dict) else None
    if selected:
        reach = float(selected.get("reachability_proxy", 0.0) or 0.0)
        path = float(selected.get("path_clear", 0.0) or 0.0)
        tq = float(selected.get("target_quality", 0.0) or 0.0)
        target_q = min(10.0, 2.0 + 3.0 * reach + 2.0 * path + 3.0 * tq)
    elif rr >= 2.0:
        target_q = 5.0
    target_q = max(0.0, min(10.0, target_q))

    # Research prior: OB was materially healthier than FVG/EQ in the supplied
    # ledger, while SELL was weaker. These are soft penalties only; they are not
    # hard filters and must be re-calibrated once a larger history is available.
    if entry_label in ("fvg", "fvg_retest"):
        entry_q -= 1.5
    elif entry_label in ("eq", "equal", "equal_high", "equal_low"):
        entry_q -= 3.0
    if direction == "bear":
        direction_q -= 2.0
    entry_q = max(0.0, min(30.0, entry_q))
    direction_q = max(0.0, min(25.0, direction_q))

    # Direction should matter, but cannot dominate. This is the key change that
    # addresses high-confidence SLs caused by strong HTF structure + poor timing.
    calibrated = int(round(
        0.25 * (base_confidence)
        + 0.30 * (direction_q + 10 * 0)
        + 0.30 * (entry_q / 30.0 * 100.0)
        + 0.15 * ((move_q + trigger_q + target_q) / 45.0 * 100.0)
    ))

    # Harder contradiction penalties are still soft, so frequency does not collapse.
    if htf_bias == "conflict":
        calibrated -= 7
    if m15_struct not in (("bullish" if direction == "bull" else "bearish"), "ranging"):
        calibrated -= 4
    if loc_score < 40:
        calibrated -= 5
    if divergence and (direction == "bull" and rsi_val >= 70 or direction == "bear" and rsi_val <= 30):
        calibrated -= 3

    calibrated = max(0, min(99, calibrated))
    setup_quality = int(round(0.20 * direction_q + 0.40 * entry_q + 0.25 * move_q + 0.10 * trigger_q + 0.05 * target_q))
    diagnostics = {
        "model_version": CONFIDENCE_MODEL_VERSION,
        "direction_component": int(base_confidence),
        "direction_quality": int(round(direction_q)),
        "entry_quality": int(round(entry_q)),
        "move_health": int(round(move_q)),
        "trigger_quality": int(round(trigger_q)),
        "target_quality": int(round(target_q)),
        "setup_quality": max(0, min(100, setup_quality)),
        "rr_band": _rr_band(rr),
        "entry_type_prior": "OB_BASELINE" if entry_label in ("ob", "ob_retest") else ("FVG_SOFT_PENALTY" if entry_label in ("fvg", "fvg_retest") else "OTHER_SOFT_PENALTY"),
        "rsi": round(rsi_val, 1),
        "relative_volume": round(rv, 3),
        "divergence_against": bool(divergence),
        "confidence_band": _confidence_band(calibrated),
    }
    return calibrated, max(0, min(100, setup_quality)), diagnostics


# =============================================================================
# ENTRY LOCATION + RSI TIMING
# =============================================================================

def _entry_location_metrics(m15: pd.DataFrame, direction: str,
                            entry: float, atr: float) -> dict:
    """Nilai lokasi entry relatif terhadap range M15 + timing RSI.

    Prinsip CAP: arah HTF boleh benar, tetapi entry di bagian atas range saat
    M15 sedang melemah adalah entry yang buruk. Kita tidak memaksa harga
    masuk di titik tertentu; kita hanya memberi ranking lebih tinggi pada
    retracement yang sehat dan memblokir chase yang jelas.
    """
    if m15 is None or len(m15) < 8:
        return {
            "location_score": 50, "location_state": "unknown",
            "range_position": 0.5, "rsi_timing": "unknown",
            "rsi": None, "rsi_slope": 0.0, "hard_block": False,
            "entry_zone_low": None, "entry_zone_high": None,
        }

    n = min(ENTRY_LOCATION_LOOKBACK, len(m15))
    sub = m15.iloc[-n:]
    rh = float(sub["high"].max())
    rl = float(sub["low"].min())
    width = max(rh - rl, max(float(atr), 1e-10))
    pos = (float(entry) - rl) / width
    pos = max(0.0, min(1.0, pos))

    rsi_now = float(m15["rsi"].iloc[-1])
    rsi_1 = float(m15["rsi"].iloc[-2])
    rsi_2 = float(m15["rsi"].iloc[-3])
    slope = rsi_now - rsi_2
    rising = slope >= RSI_TIMING_SLOPE
    falling = slope <= -RSI_TIMING_SLOPE

    # Adverse-side swing: BUY terlalu dekat high, SELL terlalu dekat low.
    recent = m15.iloc[-8:]
    adverse_high = float(recent["high"].max())
    adverse_low = float(recent["low"].min())
    if direction == "bull":
        swing_dist_atr = (adverse_high - float(entry)) / max(float(atr), 1e-10)
    else:
        swing_dist_atr = (float(entry) - adverse_low) / max(float(atr), 1e-10)

    score = 50
    notes = []
    if direction == "bull":
        if pos <= 0.35:
            score += 15; notes.append("deep_discount")
        elif pos <= ENTRY_PREFERRED_BUY:
            score += 10; notes.append("good_pullback")
        elif pos <= 0.70:
            score += 0; notes.append("mid_range")
        elif pos <= ENTRY_CHASE_HIGH:
            score -= 10; notes.append("upper_range")
        else:
            score -= 22; notes.append("chasing_high")

        if swing_dist_atr <= ENTRY_SWING_NEAR_ATR:
            score -= 10; notes.append("near_swing_high")
        if 42 <= rsi_now <= 58 and rising:
            score += 10; notes.append("rsi_recovery")
        elif rsi_now > 68 and falling:
            score -= 14; notes.append("rsi_exhaustion")
        elif rsi_now < RSI_BUY_WEAK and falling and pos > ENTRY_PREFERRED_BUY:
            score -= 18; notes.append("rsi_weak_while_high")
        elif falling:
            score -= 5; notes.append("rsi_falling")
    else:
        if pos >= 0.65:
            score += 15; notes.append("deep_premium")
        elif pos >= ENTRY_PREFERRED_SELL:
            score += 10; notes.append("good_pullback")
        elif pos >= 0.30:
            score += 0; notes.append("mid_range")
        elif pos >= (1.0 - ENTRY_CHASE_HIGH):
            score -= 10; notes.append("lower_range")
        else:
            score -= 22; notes.append("chasing_low")

        if swing_dist_atr <= ENTRY_SWING_NEAR_ATR:
            score -= 10; notes.append("near_swing_low")
        if 42 <= rsi_now <= 58 and falling:
            score += 10; notes.append("rsi_recovery")
        elif rsi_now < 32 and rising:
            score -= 8; notes.append("rsi_bounce_against_sell")
        elif rsi_now > RSI_SELL_WEAK and rising and pos < ENTRY_PREFERRED_SELL:
            score -= 18; notes.append("rsi_weak_while_low")
        elif rising:
            score -= 5; notes.append("rsi_rising")

    score = int(max(0, min(100, score)))
    # Hard block hanya untuk lokasi yang benar-benar mengejar harga.
    # RSI tetap menjadi penalty/confluence, bukan gate keras, supaya setup
    # yang bagus tidak hilang hanya karena momentum belum ideal.
    if direction == "bull":
        hard_block = pos >= ENTRY_CHASE_HIGH
    else:
        hard_block = pos <= (1.0 - ENTRY_CHASE_HIGH)

    # Zona entry berbasis range lokal, bukan titik matematis palsu.
    if direction == "bull":
        zone_low = rl
        zone_high = rl + width * 0.60
    else:
        zone_low = rl + width * 0.40
        zone_high = rh

    if hard_block:
        state = "WAIT_ENTRY"
    elif score >= 70:
        state = "GOOD"
    elif score >= 50:
        state = "ACCEPTABLE"
    else:
        state = "WEAK"

    return {
        "location_score": score,
        "location_state": state,
        "range_position": round(pos, 3),
        "range_low": rl, "range_high": rh,
        "entry_zone_low": zone_low, "entry_zone_high": zone_high,
        "rsi": round(rsi_now, 2),
        "rsi_prev": round(rsi_1, 2),
        "rsi_slope": round(slope, 2),
        "rsi_timing": "rising" if rising else ("falling" if falling else "flat"),
        "swing_dist_atr": round(swing_dist_atr, 3),
        "notes": notes,
        "hard_block": hard_block,
    }


# =============================================================================
# STEP 1 — ENTRY CANDIDATES
# =============================================================================

def _collect_entry_candidates(m15: pd.DataFrame, h1: pd.DataFrame,
                              direction: str, current_price: float,
                              atr: float, score_ctx: dict) -> list:
    """
    Kumpulkan kandidat entry berdasarkan SMC/ICT.
    Prioritas (score tertinggi):
      1. OB + Liquidity Sweep (12–15)
      2. OB + ChoCH (9–12)
      3. OB + OTE (8–10)
      4. OB saja (5–8)
      5. FVG + CISD + LiqSweep (6–9)
      6. FVG saja (3–5)
      7. Equal Highs/Lows (2–4)
      8. Market entry fallback (1)
    """
    up = direction == "bull"
    cands = []

    liq = score_ctx.get("liquidity_bull" if up else "liquidity_bear", {})
    choch = score_ctx.get("choch_m15", {})
    cisd = score_ctx.get("cisd_m15", {})

    choch_ok = choch.get("bullish_choch") if up else choch.get("bearish_choch")
    cisd_ok = cisd.get("bullish_cisd") if up else cisd.get("bearish_cisd")
    liq_ok = liq.get("type") == "sweep"

    fib_sh = float(m15["high"].iloc[score_ctx["sh15"][-1]]) if score_ctx.get("sh15") else None
    fib_sl = float(m15["low"].iloc[score_ctx["sl15"][-1]]) if score_ctx.get("sl15") else None

    induce = score_ctx.get("inducement_bull" if up else "inducement_bear", {})
    induce_ok = bool(induce.get("swept"))

    # Zona H1 (OB + FVG searah) untuk cek konfluensi — "How to Choose the
    # Best Order Block When All Zones Look Valid": OB M15 yang overlap
    # dengan zona HTF lebih dipercaya daripada OB M15 berdiri sendiri.
    htf_zones = []
    try:
        for z in detect_order_block(h1, direction, lb=80,
                                    sh=score_ctx.get("sh1", []),
                                    sl=score_ctx.get("sl1", [])):
            htf_zones.append((z["top"], z["bot"]))
        for f in detect_fvg(h1, direction, lb=60):
            htf_zones.append((f["top"], f["bot"]))
    except Exception:
        htf_zones = []

    def _htf_confluence(top: float, bot: float) -> bool:
        return any(zones_overlap(top, bot, zt, zb) for zt, zb in htf_zones)

    # ── Order Block ──────────────────────────────────────────────────
    obs = detect_order_block(m15, direction, lb=60,
                             sh=score_ctx.get("sh15", []),
                             sl=score_ctx.get("sl15", []))
    for z in obs:
        entry_pt = float(z["top"]) if up else float(z["bot"])
        invalid_pt = float(z["bot"]) if up else float(z["top"])

        # A limit order must be a genuine retracement.  The previous 1%
        # tolerance allowed a BUY above market or a SELL below market;
        # Binance can fill those immediately at a different price, leaving
        # the original SL/TP geometry invalid.
        if up:
            if current_price < z["bot"] or entry_pt > current_price * 1.001:
                continue
        else:
            if current_price > z["top"] or entry_pt < current_price * 0.999:
                continue

        sc = 3 + z["quality"]  # base 3 + quality

        if liq_ok:
            sweep_lev = liq.get("level", 0)
            if up and entry_pt >= float(sweep_lev) * 0.995:
                sc += 3
            elif not up and entry_pt <= float(sweep_lev) * 1.005:
                sc += 3
        if choch_ok:
            sc += 2
        if fib_sh and fib_sl and is_in_ote(entry_pt, fib_sl, fib_sh, direction):
            sc += 1
        if _htf_confluence(z["top"], z["bot"]):
            sc += CONFLUENCE_BONUS
        if induce_ok:
            sc += 1

        cands.append({
            "price": round(entry_pt, 8),
            "invalid": round(invalid_pt, 8),
            "label": "ob",
            "score": sc,
        })

    # ── FVG ──────────────────────────────────────────────────────────
    fvgs = detect_fvg(m15, direction, lb=50)
    for f in fvgs:
        if not f["is_fresh"]:
            continue
        entry_pt = f["mid"]
        invalid_pt = f["top"] if up else f["bot"]
        if up:
            if current_price < f["bot"] or entry_pt > current_price * 1.001:
                continue
        else:
            if current_price > f["top"] or entry_pt < current_price * 0.999:
                continue
        sc = 3
        if cisd_ok: sc += 2
        if liq_ok: sc += 2
        if choch_ok: sc += 1
        if _htf_confluence(f["top"], f["bot"]):
            sc += CONFLUENCE_BONUS
        if induce_ok:
            sc += 1
        cands.append({
            "price": round(entry_pt, 8),
            "invalid": round(invalid_pt, 8),
            "label": "fvg",
            "score": sc,
        })

    # ── Equal Highs/Lows ─────────────────────────────────────────────
    eqs = detect_equal_highs_lows(m15, "low" if up else "high", lb=80)
    for eq in eqs[:2]:
        # ── Proximity filter: EQ entry harus REACHABLE dari harga sekarang.
        #
        # Bug asal: tidak ada filter proximity → EQ yang harganya sudah
        # "tersapu" (liquidity sweep) tetap masuk sebagai kandidat entry.
        # Akibatnya SELL limit dipasang DI BAWAH harga pasar → Binance langsung
        # fill di harga pasar (slippage besar) → actual_entry > SL → geometri
        # rusak → auto-out terpicu.
        #
        # SELL (not up): entry di equal HIGH → level harus ≥ current_price
        #   supaya SELL limit menunggu harga naik ke sana, bukan fill sekarang.
        #   Toleransi 0.3%: jika eq < current_price * 0.997 → sudah tersapu.
        #
        # BUY (up): entry di equal LOW → level harus ≤ current_price
        #   supaya BUY limit menunggu harga turun ke sana, bukan fill sekarang.
        #   Toleransi 0.3%: jika eq > current_price * 1.003 → sudah tersapu.
        if not up and float(eq) < current_price * 0.999:
            continue   # EQ high sudah di bawah harga pasar → skip
        if up and float(eq) > current_price * 1.001:
            continue   # EQ low sudah di atas harga pasar → skip

        invalid_pt = eq - atr * 0.8 if up else eq + atr * 0.8
        sc = 2
        if liq_ok: sc += 1
        if induce_ok: sc += 1
        cands.append({
            "price": round(float(eq), 8),
            "invalid": round(float(invalid_pt), 8),
            "label": "eq",
            "score": sc,
        })

    # No synthetic market entry. If no fresh/reachable POI exists, wait for
    # price to come to a real zone.  Before ranking, evaluate LOCATION + RSI
    # timing for every candidate so a good-direction but bad-location setup
    # cannot become the best trade merely because its RR is attractive.
    enriched = []
    for c in cands:
        loc = _entry_location_metrics(m15, direction, c["price"], atr)
        c = dict(c)
        c["location"] = loc
        c["location_score"] = loc["location_score"]
        c["location_state"] = loc["location_state"]
        c["rsi_timing"] = loc["rsi_timing"]
        c["hard_location_block"] = loc["hard_block"]
        c["reject_reason"] = "CHASE_LOCATION" if loc["hard_block"] else None

        # Candidate-specific quality. Location is deliberately meaningful but
        # not dominant: structure/POI can still win, while an obvious chase is
        # removed before SL/TP are calculated.
        c["score"] = int(c.get("score", 0) + round((loc["location_score"] - 50) * 0.35))
        if loc["hard_block"]:
            continue
        enriched.append(c)

    enriched.sort(key=lambda c: (-c["score"], -c.get("location_score", 0)))
    return enriched


# =============================================================================
# STEP 2 — SL STRUKTURAL (anti‑Liquidity Sweep)
# =============================================================================

def _compute_sl(m15: pd.DataFrame, h1: pd.DataFrame, direction: str,
                entry: float, atr: float, liq_sweep: dict,
                invalid_level: Optional[float] = None) -> Tuple[float, float]:
    """
    Hitung SL yang tepat secara struktural, ditempatkan sedemikian rupa
    sehingga jika tersentuh berarti arah benar‑benar salah (bukan LS biasa).
    Buffer anti‑Liquidity Sweep = 0.5 ATR.
    Pilih kandidat dengan prioritas: ob_invalid > struct_h1 > ls_level > struct_m15,
    dan di antara prioritas yang sama pilih yang LEBIH JAUH dari entry
    (lebih tahan noise).
    """
    up = direction == "bull"
    ls_buffer = atr * 0.5
    min_risk = atr * 1.0
    # Risk width is a QUALITY signal, not an automatic rejection. If the
    # real structural invalidation is wide, keep that invalidation rather than
    # moving SL inward just to manufacture a better RR. TP selection will then
    # decide whether a genuine >=2R target exists.
    # A very wide structural invalidation is not repaired by pulling SL inward.
    # The caller will try another entry candidate. This is exactly the CAP lesson:
    # a 4% stop is a property of the chosen entry, not a reason to keep the entry.
    max_risk = min(float(atr) * 2.0, float(entry) * 0.035)

    cands = []

    # 1. invalid level dari OB/FVG
    if invalid_level is not None:
        sl_raw = invalid_level + (-ls_buffer if up else ls_buffer)
        risk = abs(sl_raw - entry)
        if min_risk <= risk <= max_risk:
            cands.append(("ob_invalid", sl_raw, risk))

    # 2. M15 swing (prioritas lebih rendah karena lebih rentan noise)
    sh15, sl15 = swing_pts(m15, lb=3)
    if up and sl15:
        struct_low = float(m15["low"].iloc[sl15[-1]])
        if struct_low < entry:
            sl_raw = struct_low - ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_m15", sl_raw, risk))
    elif not up and sh15:
        struct_high = float(m15["high"].iloc[sh15[-1]])
        if struct_high > entry:
            sl_raw = struct_high + ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_m15", sl_raw, risk))

    # 3. level yang disweep
    if liq_sweep and liq_sweep.get("type") == "sweep" and liq_sweep.get("level"):
        lev = float(liq_sweep["level"])
        if up and lev < entry:
            sl_raw = lev - ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("ls_level", sl_raw, risk))
        elif not up and lev > entry:
            sl_raw = lev + ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("ls_level", sl_raw, risk))

    # 4. H1 swing (lebih lebar, lebih tahan noise)
    sh1, sl1 = swing_pts(h1, lb=5)
    if up and sl1:
        h1_low = float(h1["low"].iloc[sl1[-1]])
        if h1_low < entry:
            sl_raw = h1_low - ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_h1", sl_raw, risk))
    elif not up and sh1:
        h1_high = float(h1["high"].iloc[sh1[-1]])
        if h1_high > entry:
            sl_raw = h1_high + ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_h1", sl_raw, risk))

    if cands:
        # Prioritas: ob_invalid (paling presisi), struct_h1 (paling tahan),
        # ls_level, struct_m15. Dalam prioritas sama, pilih risk terbesar.
        _PRIO = {"ob_invalid": 0, "struct_h1": 1, "ls_level": 2, "struct_m15": 3}
        cands.sort(key=lambda x: (_PRIO.get(x[0], 9), -x[2]))
        _, sl_price, risk = cands[0]
        return sl_price, risk

    # No compact structural invalidation. Do not manufacture an ATR stop.
    raise ValueError("NO_COMPACT_STRUCTURAL_SL")


# =============================================================================
# STEP 3 — TP POOL DAN SELEKSI (dengan ekstensi jika RR < 2.0)
# =============================================================================

def _target_context(h1: pd.DataFrame, m15: pd.DataFrame, direction: str,
                    entry: float, atr: float) -> dict:
    """Context lokal untuk menilai apakah target yang lebih jauh masih credible."""
    up = direction == "bull"
    sh1, sl1 = swing_pts(h1, lb=5)
    struct_h1 = _market_structure(h1, sh1, sl1)
    struct_m15 = _market_structure(m15, *swing_pts(m15, lb=5)) if len(m15) >= 12 else "ranging"
    rsi_val = float(m15["rsi"].iloc[-1]) if ("rsi" in m15.columns and len(m15)) else 50.0
    rv = _relative_volume(m15, 20) if len(m15) >= 23 else 1.0
    mom = _momentum_context(m15) if len(m15) >= 8 else {"bull": False, "bear": False}
    return {
        "h1_structure": struct_h1,
        "m15_structure": struct_m15,
        "rsi": rsi_val,
        "relative_volume": rv,
        "momentum": mom,
        "atr": max(float(atr), 1e-12),
        "entry": float(entry),
        "direction": direction,
        "sh1": sh1,
        "sl1": sl1,
    }


def _build_tp_pool(h1: pd.DataFrame, m15: pd.DataFrame, direction: str,
                   entry: float, atr: float,
                   sh1: list, sl1: list, sh15: list, sl15: list) -> list:
    """Build a rich target pool; selection is performed by _select_tp()."""
    up = direction == "bull"
    sgn = 1 if up else -1
    pool = []

    def add(value, label, tier, anchor_strength=1.0):
        try:
            v = float(value)
        except Exception:
            return
        if sgn * (v - entry) <= atr * 0.35:
            return
        pool.append({
            "value": v,
            "label": str(label),
            "tier": int(tier),
            "anchor_strength": float(anchor_strength),
        })

    opp_dir = "bear" if up else "bull"

    # External liquidity / opposing HTF zones.
    for z in detect_order_block(h1, opp_dir, lb=80, sh=sh1, sl=sl1):
        edge = float(z["bot"]) if up else float(z["top"])
        fresh = 1.0 if z.get("quality", 0) >= 4 else 0.75
        add(edge, "ob_h1", 1, 1.10 * fresh)

    for f in detect_fvg(h1, opp_dir, lb=60):
        add(float(f["mid"]), "fvg_h1", 2, 0.95 if f.get("is_fresh", False) else 0.75)

    sw_vals = ([float(h1["high"].iloc[i]) for i in sh1]
               if up else [float(h1["low"].iloc[i]) for i in sl1])
    for v in sw_vals[-2:]:
        add(v, "sw_h1", 3, 1.00)

    for v in detect_equal_highs_lows(h1, "high" if up else "low", lb=100):
        add(v, "eq_h1", 4, 1.08)

    sw_all = ([float(h1["high"].iloc[i]) for i in sh1]
              if up else [float(h1["low"].iloc[i]) for i in sl1])
    for v in sw_all[:-2]:
        add(v, "sw_h1_far", 5, 0.82)

    # Internal liquidity is useful as a waypoint/path obstacle, but may be a
    # weaker final target than external H1 liquidity.
    for v in detect_equal_highs_lows(m15, "high" if up else "low", lb=80):
        add(v, "eq_m15", 6, 0.62)

    # Extensions are valid farther targets, but start with lower credibility.
    if sh1 and sl1:
        sh_val = float(h1["high"].iloc[sh1[-1]])
        sl_val = float(h1["low"].iloc[sl1[-1]])
        leg = sh_val - sl_val
        if leg > atr * 0.8:
            for ext, lbl, tier, strength in [
                (FIB_EXT_1, "fib127", 7, 0.58),
                (FIB_EXT_2, "fib162", 7, 0.48),
                (1.0, "fib200", 8, 0.38),
                (1.414, "fib241", 8, 0.30),
            ]:
                tp_v = (sh_val + leg * ext) if up else (sl_val - leg * ext)
                add(tp_v, lbl, tier, strength)

    # De-duplicate very close targets; preserve the strongest anchor.
    pool.sort(key=lambda x: (x["value"], x["tier"]))
    dedup = []
    for item in pool:
        if dedup:
            last = dedup[-1]
            if abs(item["value"] - last["value"]) <= atr * 0.12:
                if item["anchor_strength"] > last["anchor_strength"]:
                    dedup[-1] = item
                continue
        dedup.append(item)
    return dedup


def _target_path_score(pool: list, target: dict, entry: float, direction: str,
                       atr: float) -> tuple[float, int]:
    """Score how clear the price path is before a proposed final target."""
    sgn = 1 if direction == "bull" else -1
    target_v = float(target["value"])
    obstacles = 0
    penalty = 0.0
    for other in pool:
        ov = float(other["value"])
        if other is target:
            continue
        if sgn * (ov - entry) <= 0:
            continue
        if sgn * (target_v - ov) <= atr * 0.15:
            strength = max(0.2, float(other.get("anchor_strength", 0.5)))
            tier = int(other.get("tier", 8))
            tier_w = 1.20 if tier <= 4 else (0.80 if tier <= 6 else 0.45)
            distance = abs(ov - entry) / max(atr, 1e-12)
            # Obstacles closer to entry are less damaging because they may be
            # swept/consumed on the way; major levels close to target matter more.
            target_proximity = 1.0 / (1.0 + abs(target_v - ov) / max(atr, 1e-12))
            penalty += strength * tier_w * (0.55 + 0.90 * target_proximity)
            obstacles += 1
    return max(0.0, 1.0 - min(0.78, penalty * 0.10)), obstacles


def _select_tp(pool: list, entry: float, risk: float, direction: str,
               h1: Optional[pd.DataFrame] = None,
               m15: Optional[pd.DataFrame] = None,
               atr: Optional[float] = None) -> Tuple[Optional[float], Optional[str], Optional[float], dict]:
    """Choose the target with the best risk-adjusted plausibility, not merely the first >=2R.

    There is no artificial maximum RR. The engine prefers a target when its structural
    quality, path clarity, market regime and distance form a better *expected-value proxy*.
    The proxy is a ranking heuristic, not a calibrated probability.
    """
    if not pool or risk <= 0:
        return None, None, None, {}

    atr_v = max(float(atr or risk), 1e-12)
    ctx = _target_context(h1, m15, direction, entry, atr_v) if h1 is not None and m15 is not None else {
        "h1_structure": "unknown", "m15_structure": "unknown", "rsi": 50.0,
        "relative_volume": 1.0, "momentum": {"bull": False, "bear": False},
        "atr": atr_v,
    }
    sgn = 1 if direction == "bull" else -1
    candidates = []

    aligned_struct = ctx["h1_structure"] == ("bullish" if direction == "bull" else "bearish")
    aligned_m15 = ctx["m15_structure"] == ("bullish" if direction == "bull" else "bearish")
    momentum_aligned = ((direction == "bull" and ctx["momentum"].get("bull")) or
                        (direction == "bear" and ctx["momentum"].get("bear")))
    rsi = float(ctx["rsi"])
    # RSI is soft context: avoid rewarding an already exhausted extreme.
    rsi_support = 1.0
    if direction == "bull" and rsi >= 75:
        rsi_support = 0.78
    elif direction == "bear" and rsi <= 25:
        rsi_support = 0.78

    for target in pool:
        value = float(target["value"])
        distance = sgn * (value - entry)
        if distance <= 0:
            continue
        rr = distance / risk
        if rr < MIN_RR:
            continue

        path_clear, obstacles = _target_path_score(pool, target, entry, direction, atr_v)
        quality = 0.50
        quality += 0.12 * min(1.0, max(0.0, float(target.get("anchor_strength", 0.5))))

        tier = int(target.get("tier", 8))
        tier_base = {1: 0.14, 2: 0.12, 3: 0.13, 4: 0.15, 5: 0.09, 6: 0.04, 7: 0.02, 8: 0.0}.get(tier, 0.0)
        quality += tier_base
        quality += 0.10 if aligned_struct else (-0.05 if ctx["h1_structure"] in ("bullish", "bearish") else 0.0)
        quality += 0.06 if aligned_m15 else (-0.04 if ctx["m15_structure"] in ("bullish", "bearish") else 0.0)
        quality += 0.06 if momentum_aligned else 0.0
        quality *= rsi_support
        quality *= (0.88 + 0.16 * min(1.0, path_clear))

        # Distance is valuable, but target reachability should decay as the
        # requested move becomes increasingly remote. This prevents a raw
        # "largest RR wins" behaviour while still allowing 6R+ targets when
        # structure, momentum and path quality genuinely support them.
        far_support = 0.0
        if aligned_struct:
            far_support += 0.35
        if aligned_m15:
            far_support += 0.20
        if momentum_aligned:
            far_support += 0.25
        if path_clear > 0.85:
            far_support += 0.20
        decay = max(0.028, 0.085 - 0.055 * far_support)
        distance_excess = max(0.0, rr - 3.0)
        reachability = quality * np.exp(-decay * distance_excess)
        reachability = max(0.06, min(0.93, reachability))

        # Expected-value proxy with diminishing reward. log(1+RR) prevents a
        # mathematically huge target from dominating solely because it is far.
        # It is a ranking tool, NOT a calibrated probability model.
        ev_proxy = reachability * np.log1p(rr) - (1.0 - reachability) * 0.45
        distance_bonus = 0.08 * min(1.0, max(0.0, rr - 2.0) / 4.0)
        far_penalty = 0.0
        # Research showed the useful zone is not simply "lowest RR" or "highest RR".
        # Keep unlimited RR, but require stronger path evidence for extreme targets.
        if rr > 7.0:
            evidence = far_support
            if rr > 9.0:
                far_penalty += min(0.38, 0.055 * (rr - 9.0))
                if evidence < 0.80:
                    far_penalty += min(0.22, (0.80 - evidence) * 0.55)
            elif evidence < 0.60:
                far_penalty += min(0.16, (0.60 - evidence) * 0.45)
        # Prefer the empirically interesting middle-distance region when its
        # structural/path quality is comparable, without hard-capping RR.
        if 5.0 <= rr <= 8.5:
            score += 0.035
        score = ev_proxy + distance_bonus - far_penalty

        # A target directly beyond several strong obstacles is less attractive.
        score -= min(0.30, obstacles * 0.035)

        candidates.append({
            "label": target["label"],
            "value": round(value, 10),
            "rr": round(rr, 3),
            "tier": tier,
            "target_quality": round(quality, 4),
            "path_clear": round(path_clear, 4),
            "obstacles": int(obstacles),
            "reachability_proxy": round(reachability, 4),
            "ev_proxy": round(ev_proxy, 4),
            "score": round(score, 4),
        })

    if not candidates:
        return None, None, None, {}

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    diagnostics = {
        "selected": best,
        "candidates": candidates[:12],
        "context": {
            "h1_structure": ctx["h1_structure"],
            "m15_structure": ctx["m15_structure"],
            "rsi": round(rsi, 1),
            "relative_volume": round(float(ctx["relative_volume"]), 3),
            "momentum_aligned": bool(momentum_aligned),
            "min_rr": MIN_RR,
            "max_rr": None,
        },
        "method": "structural_target_quality + path_clarity + regime/momentum context + diminishing_distance + EV_proxy",
    }
    return best["value"], best["label"], best["rr"], diagnostics


# =============================================================================
# FUNGSI UTAMA — Dipanggil oleh main.py
# =============================================================================

def full_analyze(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                 df_d1: Optional[pd.DataFrame] = None,
                 symbol: Optional[str] = None,
                 df_btc_h1: Optional[pd.DataFrame] = None,
                 trade_history: Optional[list] = None) -> Optional[dict]:
    """
    Analisa penuh satu koin: Entry → SL → TP.

    df_btc_h1: candle H1 BTCUSDT, OPSIONAL — kalau dikasih, dipakai sebagai
    filter macro (lihat _macro_bias) supaya sinyal yang jelas melawan arah
    market keseluruhan sedikit diredam, bukan diloloskan mentah-mentah cuma
    berdasar struktur koin itu sendiri. Kalau tidak dikasih (default None),
    perilaku 100% sama seperti sebelumnya — tidak ada filter tambahan.
    """
    try:
        trade_history = trade_history if isinstance(trade_history, list) else []
        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            return None

        if symbol:
            log.info(f"[{symbol}] h1={len(df_h1)} m15={len(df_m15)}")

        # Structural analysis MUST use only fully closed candles.
        # Keep the latest market price separately for execution/pending-entry
        # geometry; the forming candle must not rewrite M15/H1 structure.
        live_price = float(df_m15["close"].iloc[-1])
        h1_closed = _closed_candles(df_h1, 60)
        m15_closed = _closed_candles(df_m15, 15)
        d1_closed = _closed_candles(df_d1, 1440) if df_d1 is not None else None
        score = score_direction(h1_closed, m15_closed, d1_closed, df_btc_h1)
        if score is None:
            if symbol:
                log.debug(f"[{symbol}] score_direction=None (data kurang)")
            return None

        direction = score["direction"]
        cur_price = live_price
        atr = score["atr"]
        confidence = score["confidence"]
        up = direction == "bull"

        if symbol:
            log.info(
                f"[{symbol}] dir={direction} conf={confidence}% "
                f"struct_h1={score['struct_h1']} d1={score['d1_bias']} "
                f"macro={score.get('macro_bias', 'unknown')}"
            )

        h1 = build_df(h1_closed, interval_minutes=60)
        m15 = build_df(m15_closed, interval_minutes=15)
        if h1 is None or m15 is None:
            return None

        # ── STEP 1: ENTRY ────────────────────────────────────────
        # Coba kandidat satu per satu. Candidate #1 tidak boleh membunuh
        # seluruh koin hanya karena SL-nya lebar atau lokasi entry buruk.
        cands = _collect_entry_candidates(m15, h1, direction, cur_price, atr, score)
        if not cands:
            if symbol:
                log.debug(f"[{symbol}] no executable entry candidate (location/POI)")
            return None

        # HTF POI, displacement, sweep, CHoCH/BOS/CISD, inducement, OTE, dan
        # reaction adalah confluence. Geometri entry/SL/TP tetap dihitung per
        # candidate, sehingga candidate buruk bisa dilewati tanpa membuang koin.
        htf_poi = _collect_htf_poi_zones(h1, direction, score)
        poi_reacted = _recent_poi_reaction(m15, htf_poi, direction, atr) if htf_poi else False
        confirmation = score.get("entry_confirmation", {})
        selected_m15_struct = score.get("m15_struct", "ranging")
        opposite_m15 = (
            (direction == "bull" and selected_m15_struct == "bearish") or
            (direction == "bear" and selected_m15_struct == "bullish")
        )

        confluence_bonus = 0
        if htf_poi:
            confluence_bonus += 5
        if poi_reacted:
            confluence_bonus += 7
        if confirmation.get("confirmed"):
            confluence_bonus += 8
        if score.get("selected_sweep"):
            confluence_bonus += 9
        if score.get("trigger_count", 0) >= 2:
            confluence_bonus += 4
        elif score.get("trigger_count", 0) == 1:
            confluence_bonus += 2
        if selected_m15_struct == ("bullish" if up else "bearish"):
            confluence_bonus += 4
        if opposite_m15:
            confluence_bonus -= 6
        base_confidence = max(0, min(99, confidence + confluence_bonus))

        liq_ctx = score["liquidity_bull"] if up else score["liquidity_bear"]
        sh1 = score.get("sh1", [])
        sl1 = score.get("sl1", [])
        sh15 = score.get("sh15", [])
        sl15 = score.get("sl15", [])

        evaluated = []
        for candidate in cands:
            entry = float(candidate["price"])
            entry_lbl = candidate["label"]
            invalid = candidate.get("invalid")
            loc = candidate.get("location", {})

            # Geometry checks per candidate.
            if up and entry > cur_price * 1.005:
                continue
            if not up and entry < cur_price * 0.995:
                continue

            # Keep strategy output compatible with main.py's pre-entry gate.
            # If candidate #1 is too far, evaluate candidate #2/#3 instead of
            # returning a signal that will be rejected immediately after scan.
            # The price can move after this snapshot; main.py still performs the
            # final live validation immediately before placing the order.
            if abs(cur_price - entry) > atr * MAIN_ENTRY_MAX_ATR:
                if symbol:
                    log.debug(
                        f"[{symbol}] candidate {entry_lbl}@{entry:.6g} skipped: "
                        f"ENTRY_TOO_FAR "
                        f"({abs(cur_price-entry)/max(atr,1e-12):.2f} ATR > "
                        f"{MAIN_ENTRY_MAX_ATR:.2f} ATR)"
                    )
                continue

            try:
                sl_price, risk = _compute_sl(
                    m15, h1, direction, entry, atr, liq_ctx, invalid
                )
            except ValueError as exc:
                if symbol:
                    log.debug(
                        f"[{symbol}] candidate {entry_lbl}@{entry:.6g} skipped: {exc}"
                    )
                continue

            if up and sl_price >= entry:
                continue
            if not up and sl_price <= entry:
                continue
            if risk <= 0:
                continue
            if up and cur_price <= sl_price:
                continue
            if not up and cur_price >= sl_price:
                continue

            tp_pool = _build_tp_pool(
                h1, m15, direction, entry, atr, sh1, sl1, sh15, sl15
            )
            tp_price, tp_lbl, rr, tp_diag = _select_tp(tp_pool, entry, risk, direction, h1=h1, m15=m15, atr=atr)
            if tp_price is None or rr is None or rr < MIN_RR:
                if symbol:
                    log.debug(
                        f"[{symbol}] candidate {entry_lbl}@{entry:.6g} skipped: "
                        f"no TP >= {MIN_RR}R"
                    )
                continue

            if up and cur_price >= tp_price:
                continue
            if not up and cur_price <= tp_price:
                continue

            # V3.0: confidence menilai tradeability; entry location + move health lebih berat
            # daripada raw structural confluence. Tidak ada hard rejection baru.
            loc_score = int(loc.get("location_score", 50))
            final_conf, setup_quality, conf_diag = _calibrate_confidence(
                base_confidence, score, loc, candidate, float(rr), htf_poi, poi_reacted, tp_diag=tp_diag
            )
            empirical_conf, empirical_diag = calibrate_confidence_from_history(
                trade_history, final_conf, "BUY" if up else "SELL", entry_lbl, float(rr), setup_quality
            )
            final_conf = empirical_conf
            conf_diag.update(empirical_diag)

            execution_score = (
                final_conf
                + min(float(rr), 6.0) * 0.45
                + candidate.get("score", 0) * 0.20
                + setup_quality * 0.35
            )

            evaluated.append({
                "candidate": candidate,
                "entry": entry,
                "entry_lbl": entry_lbl,
                "invalid": invalid,
                "sl": float(sl_price),
                "risk": float(risk),
                "tp": float(tp_price),
                "tp_lbl": tp_lbl,
                "rr": float(rr),
                "confidence": final_conf,
                "setup_quality": setup_quality,
                "confidence_diagnostics": conf_diag,
                "execution_score": execution_score,
                "location": loc,
            })

        if not evaluated:
            if symbol:
                log.debug(
                    f"[{symbol}] semua candidate gugur di geometry/TP/location/"
                    f"entry-distance; tidak mengembalikan setup yang main.py pasti tolak"
                )
            return None

        evaluated.sort(key=lambda x: x["execution_score"], reverse=True)
        best_eval = evaluated[0]
        best = best_eval["candidate"]
        entry = best_eval["entry"]
        entry_lbl = best_eval["entry_lbl"]
        sl_price = best_eval["sl"]
        risk = best_eval["risk"]
        tp_price = best_eval["tp"]
        tp_lbl = best_eval["tp_lbl"]
        rr = best_eval["rr"]
        confidence = best_eval["confidence"]
        loc = best_eval["location"]

        if symbol:
            log.info(
                f"[{symbol}] ENTRY={entry:.6f} label={entry_lbl} "
                f"loc={loc.get('location_score')}%/{loc.get('location_state')} "
                f"RSI={loc.get('rsi')}({loc.get('rsi_timing')})"
            )
            log.info(f"[{symbol}] SL={sl_price:.6f} risk={risk:.6f}")
            log.info(f"[{symbol}] TP={tp_price:.6f} label={tp_lbl} RR={rr:.2f}")

        rsi_val = round(float(m15["rsi"].iloc[-1]), 1)

        return {
            "symbol": symbol,
            "original_dir": direction,
            "decision": "BUY" if up else "SELL",
            "confidence": confidence,
            "direction_confidence": max(0, min(99, base_confidence)),
            "setup_quality": int(best_eval.get("setup_quality", 0)),
            "confidence_band": best_eval.get("confidence_diagnostics", {}).get("confidence_band", _confidence_band(confidence)),
            "confidence_diagnostics": best_eval.get("confidence_diagnostics", {}),
            "confidence_model": CONFIDENCE_MODEL_VERSION,
            "confidence_is_probability": bool(best_eval.get("confidence_diagnostics", {}).get("calibration") == "empirical_bayesian"),
            "rr_band": _rr_band(rr),
            "entry_location_score": loc.get("location_score", 50),
            "entry_location_state": loc.get("location_state", "unknown"),
            "entry_range_position": loc.get("range_position", 0.5),
            "entry_zone_low": loc.get("entry_zone_low"),
            "entry_zone_high": loc.get("entry_zone_high"),
            "rsi_timing": loc.get("rsi_timing", "unknown"),
            "rsi_slope": loc.get("rsi_slope", 0.0),
            "entry": round(entry, 8),
            "price": cur_price,
            "entry_label": entry_lbl,
            "sl": round(sl_price, 8),
            "initial_sl": round(sl_price, 8),
            "initial_risk": round(abs(entry - sl_price), 8),
            "tp": round(tp_price, 8),
            "rr": round(rr, 2),
            "atr": round(atr, 8),
            "rsi": rsi_val,
            "struct_h1": score["struct_h1"],
            "d1_bias": score.get("d1_bias", "neutral"),
            "choch_m15": score["choch_m15"],
            "choch_h1": score["choch_h1"],
            "cisd_m15": score["cisd_m15"],
            "failed_retest": score.get("failed_retest", {}),
            "htf_bias": score.get("htf_bias", "unknown"),
            "h1_bias": score.get("h1_bias", "unknown"),
            "poi_reacted": poi_reacted,
            "entry_confirmation": confirmation,
            "selected_sweep": score.get("selected_sweep", False),
            "trigger_count": score.get("trigger_count", 0),
            "tp_sl_reason": (
                f"Entry@{entry:.5g}({entry_lbl}) | "
                f"SL@{sl_price:.5g}(struct) | "
                f"TP@{tp_price:.5g}({tp_lbl}) | RR={rr:.2f} | "
                f"Loc={loc.get('location_score')}({loc.get('location_state')}) | "
                f"RSI={rsi_val}/{loc.get('rsi_timing')}"
            ),
        }

    except Exception as e:
        if symbol:
            log.error(f"[full_analyze] {symbol}: {e}", exc_info=True)
        return None

def get_best_signal(candidates: list) -> Optional[dict]:
    """Pilih sinyal terbaik dari list kandidat."""
    if not candidates:
        return None
    label_bonus = {"ob": 4, "fvg": 2, "eq": 1, "market": 0}

    def _rank(sig):
        bonus = label_bonus.get(sig.get("entry_label", ""), 0)
        return sig["confidence"] + bonus + sig.get("rr", 0) * 0.5

    return max(candidates, key=_rank)


# =============================================================================
# ADAPTIVE POSITION MANAGEMENT — TRAILING BRAIN V5
# =============================================================================

def _relative_volume(df: pd.DataFrame, n: int = 20) -> float:
    """Relative volume proxy dari data OHLCV yang SUDAH DIMILIKI bot."""
    if df is None or df.empty or "volume" not in df.columns:
        return 1.0
    vol = pd.to_numeric(df["volume"], errors="coerce").dropna()
    if len(vol) < n + 3:
        return 1.0
    base = float(vol.iloc[-n-1:-1].mean())
    return float(vol.iloc[-1] / max(base, 1e-12))


def _momentum_context(df: pd.DataFrame) -> dict:
    """Momentum lokal: EMA slope + last/previous return, tanpa request API tambahan."""
    if df is None or len(df) < 8:
        return {"bull": False, "bear": False, "expanding": False, "slope_atr": 0.0}
    close = df["close"].astype(float)
    atr = max(float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0, 1e-12)
    e9 = ema(close, 9)
    slope = float(e9.iloc[-1] - e9.iloc[-4]) / atr
    ret3 = float(close.iloc[-1] - close.iloc[-4]) / atr
    last_body = abs(float(df["close"].iloc[-1] - df["open"].iloc[-1])) / atr
    return {
        "bull": slope > 0.20 and ret3 > 0.15,
        "bear": slope < -0.20 and ret3 < -0.15,
        "expanding": last_body >= 0.80,
        "slope_atr": round(slope, 3),
        "ret3_atr": round(ret3, 3),
        "last_body_atr": round(last_body, 3),
    }


def _adaptive_trail_candidates(df_m15: pd.DataFrame, direction: str, entry: float,
                              current_price: float, initial_risk: float,
                              profit_r: float, weakness_score: int) -> dict:
    """Generate عدة stop candidates; caller decides using market state."""
    up = direction == "bull"
    atr = max(float(df_m15["atr"].iloc[-1]), 1e-12)
    sh, sl = swing_pts(df_m15.iloc[-TRAIL_LOOKBACK_CANDLES:], lb=STRUCT_TRAIL_LB) if len(df_m15) >= TRAIL_LOOKBACK_CANDLES else swing_pts(df_m15, lb=STRUCT_TRAIL_LB)
    # translate local indices back into the sliced frame is unnecessary for values
    recent = df_m15.iloc[-TRAIL_LOOKBACK_CANDLES:] if len(df_m15) >= TRAIL_LOOKBACK_CANDLES else df_m15
    sh, sl = swing_pts(recent, lb=STRUCT_TRAIL_LB)

    vol = _relative_volume(df_m15, 20)
    mom = _momentum_context(df_m15)
    adaptive_buffer = atr * (0.40 if weakness_score < TRAIL_WEAKNESS_MODERATE else 0.30)
    if vol < TRAIL_VOLUME_WEAK:
        adaptive_buffer *= 0.90
    if vol > TRAIL_VOLUME_EXPANSION and ((up and mom["bull"]) or ((not up) and mom["bear"])):
        adaptive_buffer *= 1.15

    structure_stop = None
    if up and sl:
        structure_stop = float(recent["low"].iloc[sl[-1]]) - adaptive_buffer
    elif (not up) and sh:
        structure_stop = float(recent["high"].iloc[sh[-1]]) + adaptive_buffer

    extreme_hi = float(recent["high"].max())
    extreme_lo = float(recent["low"].min())
    atr_mult = TRAIL_ATR_TIGHT_MULT if weakness_score >= TRAIL_WEAKNESS_STRONG else TRAIL_ATR_MULT
    atr_stop = (extreme_hi - atr * atr_mult) if up else (extreme_lo + atr * atr_mult)

    be_stop = entry + (initial_risk * TRAIL_BREAK_EVEN_OFFSET_R if up else -initial_risk * TRAIL_BREAK_EVEN_OFFSET_R)
    min_gap = atr * TRAIL_MIN_GAP_ATR
    # Ensure the candidate is not so close that normal noise immediately takes it.
    if up:
        max_usable = current_price - min_gap
        structure_stop = min(structure_stop, max_usable) if structure_stop is not None else None
        atr_stop = min(atr_stop, max_usable)
        be_stop = min(be_stop, max_usable)
    else:
        min_usable = current_price + min_gap
        structure_stop = max(structure_stop, min_usable) if structure_stop is not None else None
        atr_stop = max(atr_stop, min_usable)
        be_stop = max(be_stop, min_usable)

    return {
        "structure": structure_stop,
        "atr": atr_stop,
        "breakeven": be_stop,
        "relative_volume": round(vol, 3),
        "momentum": mom,
        "atr": atr,
        "buffer": adaptive_buffer,
    }


# Number of M15 candles examined by the trailing engine. Kept moderate to avoid
# unnecessary CPU work while still covering several structural legs.
TRAIL_LOOKBACK_CANDLES = STRUCT_TRAIL_LOOKBACK


def manage_position(state: dict, df_m15: pd.DataFrame, df_h1: Optional[pd.DataFrame] = None,
                    df_d1: Optional[pd.DataFrame] = None, symbol: Optional[str] = None) -> dict:
    """
    Position-management brain V4.

    Important: this function performs NO REST/WebSocket calls. It only processes
    data already supplied by main.py and returns a decision. main.py remains the
    sole execution layer.

    Decision states:
      HOLD      = trend is healthy; do not touch SL
      PROTECT   = trade matured enough to consider BE/soft protection
      TRAIL     = tighten SL using adaptive structure/ATR candidate
      EXIT      = confirmed structural/momentum failure; main.py may close
    """
    if df_m15 is None or df_m15.empty:
        return {"action": "HOLD", "state": "NO_DATA", "reason": ["M15 unavailable"]}

    live_price = float(df_m15["close"].iloc[-1])
    m15 = build_df(_closed_candles(df_m15, 15), interval_minutes=15)
    if m15 is None or len(m15) < 55:
        return {"action": "HOLD", "state": "NO_DATA", "reason": ["insufficient M15 data"]}

    sig = state.get("signal", state) or {}
    decision = str(sig.get("decision") or state.get("decision") or "BUY").upper()
    direction = "bull" if decision == "BUY" else "bear"
    entry = float(state.get("entry") or sig.get("entry") or m15["close"].iloc[-1])
    current_price = float(state.get("current_price") or state.get("price") or live_price)
    current_sl = float(state.get("current_sl") or sig.get("sl") or entry)
    initial_sl = float(state.get("initial_sl") or sig.get("initial_sl") or sig.get("sl") or current_sl)
    tp = sig.get("tp")
    tp = float(tp) if tp is not None else None
    initial_risk = max(abs(entry - initial_sl), 1e-12)
    profit_r = ((current_price - entry) if direction == "bull" else (entry - current_price)) / initial_risk

    sh, sl = swing_pts(m15, lb=STRUCT_TRAIL_LB)
    structure = _market_structure(m15, sh, sl)
    rdiv = detect_rsi_divergence(m15, direction, lb=30)
    vol = _relative_volume(m15, 20)
    mom = _momentum_context(m15) if len(m15) >= 8 else {"bull": False, "bear": False}
    atr = max(float(m15["atr"].iloc[-1]), 1e-12)

    weakness = 0
    reasons = []

    # Structure is the primary authority. A healthy trend gives the trade room.
    aligned_structure = structure == ("bullish" if direction == "bull" else "bearish")
    opposite_structure = structure == ("bearish" if direction == "bull" else "bullish")
    if aligned_structure:
        reasons.append("structure_aligned")
    elif opposite_structure:
        weakness += 3
        reasons.append("structure_opposite")
    else:
        reasons.append("structure_ranging")

    # Momentum: strong aligned movement = patience; strong opposite movement = tighten.
    if (direction == "bull" and mom["bull"]) or (direction == "bear" and mom["bear"]):
        reasons.append("momentum_aligned")
    elif (direction == "bull" and mom["bear"]) or (direction == "bear" and mom["bull"]):
        weakness += 2
        reasons.append("momentum_reversal")

    # RSI divergence is a warning, not an automatic exit.
    div = (direction == "bull" and rdiv.get("bear_div")) or (direction == "bear" and rdiv.get("bull_div"))
    if div:
        weakness += 2 if not rdiv.get("strong") else 3
        reasons.append("rsi_divergence")

    # Volume/price interaction. High aligned volume supports continuation; falling
    # volume after extension is an exhaustion warning.
    aligned = (direction == "bull" and mom["bull"]) or (direction == "bear" and mom["bear"])
    if vol >= TRAIL_VOLUME_EXPANSION and aligned:
        reasons.append("volume_supports_move")
    elif vol <= TRAIL_VOLUME_WEAK and profit_r >= TRAIL_MATURE_R:
        weakness += 1
        reasons.append("volume_exhaustion")
    elif vol >= TRAIL_VOLUME_EXPANSION and not aligned:
        weakness += 2
        reasons.append("counter_volume_expansion")

    # If the intended target is close, protect more aggressively once there is
    # evidence of rejection. Until then, allow the winner to complete its path.
    target_near = False
    if tp is not None and initial_risk > 0:
        target_distance_r = abs(tp - current_price) / initial_risk
        target_near = target_distance_r <= TRAIL_LIQUIDITY_NEAR_R
        if target_near:
            reasons.append("target_near")

    # Major recent swing/liquidity was reached and rejected: treat as stronger warning.
    recent_high = float(m15["high"].iloc[-20:].max())
    recent_low = float(m15["low"].iloc[-20:].min())
    last = m15.iloc[-1]
    body_atr = abs(float(last["close"] - last["open"])) / atr
    if direction == "bull" and float(last["high"]) >= recent_high * (1 - 1e-9) and float(last["close"]) < float(last["open"]) and body_atr >= 0.8:
        weakness += 2
        reasons.append("upper_liquidity_rejection")
    if direction == "bear" and float(last["low"]) <= recent_low * (1 + 1e-9) and float(last["close"]) > float(last["open"]) and body_atr >= 0.8:
        weakness += 2
        reasons.append("lower_liquidity_rejection")

    # Trade maturity changes how much weakness we tolerate.
    if profit_r < TRAIL_ARM_R:
        return {
            "action": "HOLD", "state": "DEVELOPING", "profit_r": round(profit_r, 2),
            "reason": reasons, "relative_volume": round(vol, 3),
        }

    candidates = _adaptive_trail_candidates(
        m15, direction, entry, current_price, initial_risk, profit_r, weakness
    )

    # Strong continuation: don't crowd the trade. Only act if an objectively more
    # protective candidate exists and profit has matured enough.
    if aligned and weakness < TRAIL_WEAKNESS_MODERATE and profit_r < TRAIL_MATURE_R:
        return {
            "action": "PROTECT" if profit_r >= TRAIL_PROTECT_R else "HOLD",
            "state": "TREND_HEALTHY",
            "profit_r": round(profit_r, 2),
            "reason": reasons + ["winner_allowed_room"],
            "relative_volume": round(vol, 3),
        }

    # Choose the candidate that protects without sitting inside normal noise.
    valid = []
    for name, value in candidates.items():
        if name in {"relative_volume", "momentum", "atr", "buffer"} or value is None:
            continue
        v = float(value)
        if direction == "bull" and v < current_price:
            if v > current_sl + atr * TRAIL_MIN_IMPROVEMENT_ATR:
                valid.append((v, name))
        elif direction == "bear" and v > current_price:
            if v < current_sl - atr * TRAIL_MIN_IMPROVEMENT_ATR:
                valid.append((v, name))

    # On moderate weakness choose the tighter valid candidate; on strong weakness
    # the same rule applies, but ATR multiplier has already contracted.
    if valid:
        if direction == "bull":
            chosen, source = max(valid, key=lambda x: x[0])
        else:
            chosen, source = min(valid, key=lambda x: x[0])
        state_name = "EXHAUSTION" if weakness >= TRAIL_WEAKNESS_STRONG else ("MATURE" if profit_r >= TRAIL_MATURE_R else "PROTECT")
        action = "TRAIL"
        return {
            "action": action,
            "state": state_name,
            "sl": round(float(chosen), 10),
            "profit_r": round(profit_r, 2),
            "weakness_score": int(weakness),
            "trail_source": source,
            "relative_volume": round(vol, 3),
            "reason": reasons + [f"adaptive_trail:{source}"]
        }

    # If trend has actually broken and no candidate can improve current SL, signal a
    # controlled close only after multiple independent warnings agree.
    if weakness >= TRAIL_WEAKNESS_STRONG and profit_r >= TRAIL_MATURE_R:
        return {
            "action": "EXIT",
            "close": True,
            "state": "EXIT_WARNING",
            "profit_r": round(profit_r, 2),
            "weakness_score": int(weakness),
            "reason": reasons + ["multi-factor-structure-failure"]
        }

    return {
        "action": "HOLD",
        "state": "WAIT",
        "profit_r": round(profit_r, 2),
        "weakness_score": int(weakness),
        "reason": reasons + ["no_better_stop_candidate"]
    }


# =============================================================================
# VALIDASI PRE-ORDER & KOREKSI GEOMETRY (dipanggil oleh main.py)
# =============================================================================

def validate_and_adjust_geometry(
    entry: float, sl: float, tp: float,
    current_price: float, atr: float,
    direction: str,
) -> Optional[dict]:
    """
    Validasi dan (jika perlu) koreksi geometri entry/SL/TP sebelum order dipasang
    atau setelah order terisi di harga yang berbeda dari target.

    Mengapa fungsi ini penting:
    ─────────────────────────────────────────────────────────────────────────────
    Sinyal dihitung pada waktu T. Saat order terpasang (T+beberapa detik/menit),
    harga pasar bisa sudah bergerak — khususnya:

      • Kasus "geometri invalid setelah order terisi":
        SELL limit di entry_target, tapi actual_fill = harga pasar (lebih tinggi
        dari entry_target karena market sudah di atas limit sell). Akibatnya
        actual_fill > SL → geometri rusak.

      • Kasus "harga sudah melewati SL setelah order terisi":
        Sinyal sudah kedaluwarsa atau fill terjadi di sisi yang salah.
        Posisi tidak boleh diselamatkan dengan menggeser entry/SL
        secara retroaktif.

    Logika:
    ─────────────────────────────────────────────────────────────────────────────
    1. Cek geometri dasar: SL di sisi yang benar dari entry, TP di sisi lain.
    2. Cek SL belum ditembus current_price.
    3. Jika SL ditembus atau geometri fill berubah, tolak posisi secara aman.
       Wick M1 tidak cukup untuk membuktikan bahwa posisi masih valid.
    4. Cek RR ≥ MIN_RR tanpa mengubah level trade.

    Return:
      dict  {entry, sl, tp, rr, adjusted} jika valid / bisa diselamatkan
      None  jika tidak bisa diperbaiki → TOLAK sinyal / auto-out
    """
    up = direction == "bull"
    def _geo_ok(e: float, s: float, t: float) -> bool:
        return (s < e < t) if up else (t < e < s)

    def _rr(e: float, s: float, t: float) -> float:
        return abs(t - e) / max(abs(e - s), 1e-10)

    sl_breached = (current_price <= sl) if up else (current_price >= sl)

    # ─── Kasus 1: sudah valid ────────────────────────────────────────────────
    if _geo_ok(entry, sl, tp) and not sl_breached:
        rr = _rr(entry, sl, tp)
        if rr < MIN_RR:
            return None
        return {"entry": entry, "sl": sl, "tp": tp, "rr": round(rr, 2), "adjusted": False}

    # ─── Kasus 2: SL ditembus → fail closed ─────────────────────────────────
    if sl_breached:
        log.info(
            f"[validate_geo] SL sudah ditembus sebelum validasi "
            f"(entry={entry:.6g}, sl={sl:.6g}, price={current_price:.6g}) — ditolak"
        )
        return None

    # Never move entry or SL after a fill. A changed geometry is a stale
    # signal, not a new setup.
    if not _geo_ok(entry, sl, tp):
        return None

    rr = _rr(entry, sl, tp)
    if rr < MIN_RR:
        log.debug(
            f"[validate_geo] RR={rr:.2f} < MIN_RR={MIN_RR} setelah koreksi — ditolak"
        )
        return None

    return {
        "entry": round(entry, 8),
        "sl":    round(sl, 8),
        "tp":    tp,
        "rr":    round(rr, 2),
        "adjusted": False,
    }