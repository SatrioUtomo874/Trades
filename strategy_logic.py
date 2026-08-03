"""
strategy_logic.py — OTAK v2
============================
Dibangun ulang dari 30 transkrip video SMC/ICT (channel RUANG TRADER).

ALUR ANALISA — BERURUTAN (sesuai instruksi user):
  Step 1 → ENTRY  : OB di zona diskon/premium + LiqSweep + ChoCH + OTE
  Step 2 → SL     : Level struktural M15 + buffer anti-Liquidity Sweep
  Step 3 → TP     : Pool target H1; jika RR < 2.0 → extend, JANGAN tolak
                     Batasi sampai RR 4.0

PRINSIP UTAMA:
  • OB dipilih hanya di zona diskon (bull) atau premium (bear) — Fibonacci filter
  • SL hanya tersentuh jika arah benar-benar salah, bukan sekadar Liquidity Sweep
  • Jika RR < 2.0: cari target lebih jauh, naikkan TP, cap di 4.0
  • Confidence = skor GLOBAL tunggal, tanpa bias sesi
  • Trail Ladder = geser SL ke struktur M15, BUKAN profit-taker paksa

KOMPATIBEL DENGAN main.py:
  - full_analyze(df_h1, df_m15, df_d1=None, symbol=None) → dict | None
  - score_direction(df_h1, df_m15, df_d1=None)           → dict | None
  - swing_pts(df, lb)                                    → (sh, sl)
  - TRAIL_R_LADDER, STRUCT_TRAIL_LB, STRUCT_TRAIL_BUF_PCT, STRUCT_TRAIL_LOOKBACK
  - MIN_RR, MAX_RR, FIB_EXT_1, FIB_EXT_2
"""
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# KONFIGURASI — Diimpor langsung oleh main.py (jangan ubah nama variabel)
# =============================================================================

MIN_RR   = 2.0
MAX_RR   = 4.0

# Trail Ladder: (min_R_profit_untuk_aktif, fraksi_lock_dari_risk)
# (1.0, 0.30) → saat profit ≥ 1.0R: geser SL ke level struktural M15
# terdekat yang ≥ entry + 0.30*risk. Trail bukan profit-taker,
# tapi update SL karena harga tidak kuat pertahankan trend.
TRAIL_R_LADDER = [
    (0.5, 0.00),   # break-even
    (1.0, 0.30),   # lock 0.3 R
    (1.5, 0.50),   # lock 0.5 R
    (2.0, 0.65),   # lock 0.65 R
    (2.8, 0.80),   # lock 0.8 R
    (3.5, 0.85),   # lock 0.85 R
]

# Trailing struktural M15 — dipakai main.py untuk geser SL ke swing M15
STRUCT_TRAIL_LB       = 3      # lookback swing_pts saat trailing
STRUCT_TRAIL_BUF_PCT  = 0.002  # buffer 0.2% di luar swing agar tidak kena LS biasa
STRUCT_TRAIL_LOOKBACK = 60     # candle M15 untuk cari swing trailing

# Fibonacci extension TP (level 1.272 dan 1.618 dari impulse leg)
FIB_EXT_1 = 0.272   # 127.2%
FIB_EXT_2 = 0.618   # 161.8%


# =============================================================================
# UTILITY — Indikator teknikal
# =============================================================================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d  = s.diff()
    g  = d.clip(lower=0).rolling(n).mean()
    lo = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / lo.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr_fn(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Tambahkan EMA, RSI, ATR, volume SMA ke DataFrame OHLCV."""
    if df is None or len(df) < 60:
        return None
    df = df.copy()
    df["ema9"]   = ema(df["close"],   9)
    df["ema21"]  = ema(df["close"],  21)
    df["ema50"]  = ema(df["close"],  50)
    df["ema200"] = (
        ema(df["close"], 200) if len(df) >= 200 else ema(df["close"], 50)
    )
    df["rsi"]     = rsi(df["close"])
    df["atr"]     = atr_fn(df)
    df["vol_sma"] = df["volume"].rolling(20).mean()
    return df.dropna()


def swing_pts(df: pd.DataFrame, lb: int = 5):
    """
    Swing high & low.
    PENTING: main.py mengimpor fungsi ini langsung untuk trailing stop.
    Jangan ubah nama atau signature.
    """
    sh, sl = [], []
    high = df["high"].values
    low  = df["low"].values
    n    = len(high)
    for i in range(lb, n - lb):
        window_h = high[max(0, i - lb): i + lb + 1]
        window_l = low [max(0, i - lb): i + lb + 1]
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
    hl = df["low"].iloc[sl[-1]]  > df["low"].iloc[sl[-2]]
    lh = df["high"].iloc[sh[-1]] < df["high"].iloc[sh[-2]]
    ll = df["low"].iloc[sl[-1]]  < df["low"].iloc[sl[-2]]
    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"

# Alias publik dipakai score_direction
mkt_struct = _market_structure


# =============================================================================
# SMC / ICT DETECTORS
# =============================================================================

def is_zone_fresh(df: pd.DataFrame, top: float, bot: float,
                  formed_idx: int, end_idx: Optional[int] = None) -> bool:
    """
    True jika zona (OB/FVG) belum pernah ditembus setelah terbentuk.
    Zona 'disentuh' = candle mana pun setelah formed_idx yang wick-nya masuk ke zona.
    """
    if formed_idx is None or formed_idx + 2 >= len(df):
        return True
    start = formed_idx + 2
    end   = end_idx if end_idx is not None else len(df) - 1
    if start >= end:
        return True
    sub     = df.iloc[start:end]
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)


def fib_position(price: float, swing_low: float, swing_high: float) -> float:
    """
    Posisi harga dalam range swing_low–swing_high.
    0.0 = di swing low · 1.0 = di swing high
    < 0.5 = discount zone (bawah), > 0.5 = premium zone (atas).
    """
    rng = swing_high - swing_low
    if rng <= 0:
        return 0.5
    return max(0.0, min(1.0, (price - swing_low) / rng))


def is_in_ote(price: float, swing_low: float, swing_high: float,
              direction: str) -> bool:
    """
    OTE (Optimal Trade Entry) dari Section 5 & 20 transkrip:
    Zona 61.8%–78.6% retracement = level Fibonacci paling sering memantul.

    Bull OTE: harga di 61.8%–78.6% turun dari swing_high
              → fib_position antara 0.214 dan 0.382 (dari bawah)
    Bear OTE: harga di 61.8%–78.6% naik dari swing_low
              → fib_position antara 0.618 dan 0.786
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
    Fair Value Gap (imbalance / institutional gap):
    3-candle pattern di mana ada gap antara candle 1 dan candle 3.

    Bull FVG: candle[i+2].low > candle[i].high
    Bear FVG: candle[i+2].high < candle[i].low

    Hanya return zona yang SEGAR (belum terisi ulang).
    """
    sub  = df.iloc[-lb:]
    base = len(df) - len(sub)
    out  = []

    for i in range(len(sub) - 2):
        c0, c2 = sub.iloc[i], sub.iloc[i + 2]
        gap = None
        if direction == "bull" and c2["low"] > c0["high"]:
            gap = {"top": float(c2["low"]), "bot": float(c0["high"])}
        elif direction == "bear" and c2["high"] < c0["low"]:
            gap = {"top": float(c0["low"]), "bot": float(c2["high"])}

        if gap:
            gap["mid"]      = (gap["top"] + gap["bot"]) / 2
            gap["idx"]      = base + i + 2
            gap["is_fresh"] = is_zone_fresh(df, gap["top"], gap["bot"], gap["idx"])
            out.append(gap)

    fresh = [f for f in out if f["is_fresh"]]
    return fresh[-3:] if fresh else []


def detect_order_block(df: pd.DataFrame, direction: str, lb: int = 60,
                       sh: Optional[list] = None, sl: Optional[list] = None) -> list:
    """
    Order Block berkualitas (Section 1, 14 transkrip).

    OB = candle berlawanan (pemicu/trigger candle) sebelum candle impuls kuat.
    Filter wajib: zona segar (fresh).

    Quality scoring (0–6):
      +1  impulse body ≥ 1.5× rata-rata body
      +1  impulse body ≥ 2.5× rata-rata body (sangat kuat)
      +1  ada FVG tepat setelah OB (imbalance = smart money speed)
      +1  ada BOS setelah OB (konfirmasi market structure)
      +1  OB di zona diskon (bull) atau premium (bear) berdasarkan Fibonacci
          — konsep utama Section 1: "Pilih OB di bawah 0.618 untuk buy"
      +1  OB terbentuk dalam 20 candle terakhir (recency bonus)

    Minimum quality untuk lolos: 2
    Return: list terurut kualitas tertinggi, max 3 zona.
    """
    is_demand = direction == "bull"
    sub       = df.iloc[-lb:]
    base      = len(df) - len(sub)
    avg_body  = (sub["close"] - sub["open"]).abs().mean() or 1e-8

    # Fibonacci context untuk filter diskon/premium
    fib_sh = float(df["high"].iloc[sh[-1]]) if (sh and len(sh) > 0) else None
    fib_sl = float(df["low"].iloc[sl[-1]])  if (sl and len(sl) > 0) else None

    # BOS global: ada break of structure yang relevan setelah OB?
    has_bos_global = False
    if sh and sl:
        if is_demand and len(sh) >= 2:
            has_bos_global = float(df["high"].iloc[-1]) > float(df["high"].iloc[sh[-2]])
        elif not is_demand and len(sl) >= 2:
            has_bos_global = float(df["low"].iloc[-1]) < float(df["low"].iloc[sl[-2]])

    zones = []
    for i in range(1, len(sub) - 3):
        c, nx = sub.iloc[i], sub.iloc[i + 1]

        # Cek pola trigger-candle + impulse-candle
        if is_demand:
            # Bull demand OB: bearish trigger → bullish impulse
            if not (c["close"] < c["open"] and nx["close"] > nx["open"]):
                continue
        else:
            # Bear supply OB: bullish trigger → bearish impulse
            if not (c["close"] > c["open"] and nx["close"] < nx["open"]):
                continue

        impulse_body = abs(nx["close"] - nx["open"])
        if impulse_body < avg_body * 1.2:
            continue

        ob_top = float(max(c["open"], c["close"]))
        ob_bot = float(min(c["open"], c["close"]))
        df_idx = base + i

        if not is_zone_fresh(df, ob_top, ob_bot, df_idx):
            continue

        q = 0

        # 1. Impulse kuat
        if impulse_body >= avg_body * 1.5:
            q += 1
        if impulse_body >= avg_body * 2.5:
            q += 1  # double bonus untuk impulse sangat kuat

        # 2. FVG setelah OB (gap antara trigger candle dan candle ke-2 setelah trigger)
        if i + 2 < len(sub):
            c2 = sub.iloc[i + 2]
            if is_demand and c2["low"] > c["high"]:
                q += 1
            elif not is_demand and c2["high"] < c["low"]:
                q += 1

        # 3. BOS global
        if has_bos_global:
            q += 1

        # 4. Fibonacci zone filter — KUNCI dari Section 1 transkrip:
        #    "Kita hanya mempertimbangkan OB yang berada di area diskon (< 0.618)"
        if fib_sh is not None and fib_sl is not None:
            ob_mid  = (ob_top + ob_bot) / 2
            fib_r   = fib_position(ob_mid, fib_sl, fib_sh)
            # Bull: diskon = fib_r ≤ 0.618 (di bawah golden ratio)
            # Bear: premium = fib_r ≥ 0.382 (di atas 38.2% dari bawah = premium)
            if is_demand and fib_r <= 0.618:
                q += 1
            elif not is_demand and fib_r >= 0.382:
                q += 1

        # 5. Recency bonus
        if df_idx >= len(df) - 20:
            q += 1

        if q >= 2:
            zones.append({
                "top":      ob_top,
                "bot":      ob_bot,
                "mid":      (ob_top + ob_bot) / 2,
                "idx":      df_idx,
                "quality":  q,
                "has_bos":  has_bos_global,
            })

    # Urut: kualitas tertinggi, lalu lebih baru
    zones.sort(key=lambda z: (-z["quality"], -z["idx"]))
    return zones[:3]


def detect_choch(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """
    Change of Character (Section 17, 19, 20 transkrip).
    Sinyal awal perubahan struktur pasar — lebih awal dari BOS.

    Bullish ChoCH: dalam struktur bearish (LH+LL), harga close di atas
                   swing high sebelumnya → struktur mulai berubah bullish.
    Bearish ChoCH: dalam struktur bullish (HH+HL), harga close di bawah
                   swing low sebelumnya → struktur mulai berubah bearish.
    """
    result = {"bullish_choch": False, "bearish_choch": False}
    if len(sh) < 2 or len(sl) < 2:
        return result

    close     = float(df["close"].iloc[-1])
    prev_high = float(df["high"].iloc[sh[-2]])
    last_high = float(df["high"].iloc[sh[-1]])
    prev_low  = float(df["low"].iloc[sl[-2]])
    last_low  = float(df["low"].iloc[sl[-1]])
    struct    = _market_structure(df, sh, sl)

    # Bullish ChoCH: setelah LH+LL, close di atas prev_low (mulai reverse)
    if struct == "bearish" and close > prev_low:
        result["bullish_choch"] = True

    # Bearish ChoCH: setelah HH+HL, close di bawah prev_high
    if struct == "bullish" and close < prev_high:
        result["bearish_choch"] = True

    # Raw ChoCH (tanpa perlu strict structure): break dari pola swing terakhir
    if last_high > prev_high and last_low > prev_low and close > prev_low:
        result["bullish_choch"] = True
    if last_high < prev_high and last_low < prev_low and close < prev_low:
        result["bearish_choch"] = True

    return result


def detect_bos(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """
    Break of Structure — konfirmasi kelanjutan tren (Section 17 transkrip).
    BOS = harga menembus swing high/low sebelumnya dengan close body.
    """
    result = {"bullish_bos": False, "bearish_bos": False}
    if len(sh) < 2 or len(sl) < 2:
        return result
    if float(df["high"].iloc[-1]) > float(df["high"].iloc[sh[-2]]):
        result["bullish_bos"] = True
    if float(df["low"].iloc[-1]) < float(df["low"].iloc[sl[-2]]):
        result["bearish_bos"] = True
    return result


def detect_cisd(df: pd.DataFrame, lb: int = 8) -> dict:
    """
    Change In State of Delivery — sinyal reversal PALING AWAL dalam ICT
    (Section 29 transkrip).

    Dalam ICT, 'state of delivery' = karakter candle saat ini (bullish/bearish delivery).
    CISD terjadi saat 1 candle tiba-tiba mengubah karakter pengiriman:

    Bullish CISD: Minimal 3 candle bearish berturutan, lalu 1 candle bullish
                  yang menutup LEBIH TINGGI dari pertengahan candle bearish pertama
                  → pengiriman berubah dari bearish ke bullish.

    Bearish CISD: Minimal 3 candle bullish berturutan, lalu 1 candle bearish
                  yang menutup LEBIH RENDAH dari pertengahan candle bullish pertama
                  → pengiriman berubah dari bullish ke bearish.

    Kenapa bukan sekadar 'candle reversal'? Karena syarat 'menutup di atas/bawah
    pertengahan' memastikan perubahan karakter signifikan, bukan sekadar noise.
    """
    result = {"bullish_cisd": False, "bearish_cisd": False}
    if len(df) < lb + 1:
        return result

    sub    = df.iloc[-lb:]
    opens  = sub["open"].values
    closes = sub["close"].values
    n      = len(closes)
    if n < 4:
        return result

    last_bull = closes[-1] > opens[-1]
    last_bear = closes[-1] < opens[-1]

    if last_bull:
        # Hitung run bearish sebelum candle terakhir
        bear_run = 0
        for j in range(n - 2, -1, -1):
            if closes[j] < opens[j]:
                bear_run += 1
            else:
                break
        if bear_run >= 3:
            # Candle bullish harus close di atas midpoint candle bearish pertama
            first_idx = n - 1 - bear_run
            if first_idx >= 0:
                bear_mid = (opens[first_idx] + closes[first_idx]) / 2
                if closes[-1] > bear_mid:
                    result["bullish_cisd"] = True

    elif last_bear:
        # Hitung run bullish sebelum candle terakhir
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
    Liquidity Sweep — konsep utama dari Section 9, 14, 15, 18 transkrip.

    Market bergerak untuk MENGAMBIL stop-loss retail yang terkumpul di
    atas swing high (sell-side) atau di bawah swing low (buy-side).
    Setelah stop loss diambil, harga berbalik → itulah saat yang tepat untuk entry.

    Ciri sweep valid:
      Bull sweep: candle wick menembus swing low TAPI close KEMBALI di atas level
      Bear sweep: candle wick menembus swing high TAPI close KEMBALI di bawah level

    'strength' = seberapa jauh sweep (1-3), makin dalam makin banyak likuiditas terambil.
    """
    result = {"type": "none", "level": None, "strength": 0}

    if direction == "bull" and sl:
        level     = float(df["low"].iloc[sl[-1]])
        last_low  = float(df["low"].iloc[-1])
        last_close= float(df["close"].iloc[-1])
        # Wick tembus ke bawah swing low, tapi close kembali di atasnya
        if last_low < level and last_close > level:
            depth = (level - last_low) / max(level, 1e-10)
            result = {
                "type":     "sweep",
                "level":    level,
                "strength": min(3, int(depth / 0.002) + 1),
            }

    elif direction == "bear" and sh:
        level      = float(df["high"].iloc[sh[-1]])
        last_high  = float(df["high"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_high > level and last_close < level:
            depth = (last_high - level) / max(level, 1e-10)
            result = {
                "type":     "sweep",
                "level":    level,
                "strength": min(3, int(depth / 0.002) + 1),
            }

    return result


def detect_equal_highs_lows(df: pd.DataFrame, kind: str = "high",
                             lb: int = 80, tol: float = 0.003) -> list:
    """
    Equal Highs/Lows = internal liquidity (Section 15, 24 transkrip).
    Di mana stop-loss trader berkumpul karena level yang 'obvious'.

    tol: toleransi level dianggap 'equal' (0.3% default).
    Return: list level harga yang merupakan cluster ≥ 2 swing.
    """
    sub  = df.iloc[-lb:]
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
    """
    RSI Divergence (Section 19 transkrip — dikombinasikan dengan SMC).

    Bullish divergence: harga lower low, RSI higher low → momentum jual melemah.
    Bearish divergence: harga higher high, RSI lower high → momentum beli melemah.

    'strong': True jika divergence terjadi di zona extreme RSI
              (oversold < 35 untuk bull, overbought > 65 untuk bear).
    """
    result = {"bull_div": False, "bear_div": False, "strong": False}
    if len(df) < lb + 1 or "rsi" not in df.columns:
        return result

    sub   = df.iloc[-lb:]
    price = sub["close"].values
    rsi_v = sub["rsi"].values
    n     = len(price)
    lb3   = 3  # pivot detection lookback

    # ─── Bullish divergence ──────────────────────────────────────
    lows = [i for i in range(lb3, n - lb3)
            if price[i] == min(price[max(0, i - lb3): i + lb3 + 1])]
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if price[i2] < price[i1] and rsi_v[i2] > rsi_v[i1]:
            result["bull_div"] = True
            if rsi_v[i2] < 35:
                result["strong"] = True

    # ─── Bearish divergence ──────────────────────────────────────
    highs = [i for i in range(lb3, n - lb3)
             if price[i] == max(price[max(0, i - lb3): i + lb3 + 1])]
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if price[i2] > price[i1] and rsi_v[i2] < rsi_v[i1]:
            result["bear_div"] = True
            if rsi_v[i2] > 65:
                result["strong"] = True

    # Filter: kembalikan yang relevan dengan direction saja
    if direction == "bull" and not result["bull_div"]:
        return {"bull_div": False, "bear_div": False, "strong": False}
    if direction == "bear" and not result["bear_div"]:
        return {"bull_div": False, "bear_div": False, "strong": False}

    return result


def detect_failed_retest(df: pd.DataFrame, sh: list, sl: list,
                         atr: float) -> dict:
    """
    Failed retest: harga mencoba menembus level lama tapi gagal dan berbalik.
    Dipertahankan untuk kompatibilitas dengan format output main.py.
    """
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


# =============================================================================
# SCORING — Confidence global tanpa bias sesi
# =============================================================================

def score_direction(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                    df_d1: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """
    Tentukan arah dan skor confidence sinyal.

    Faktor scoring (max ~160 poin, dinormalisasi ke 0–99):
      [D1]  Struktur/EMA aligned            +20
      [H1]  Struktur market aligned         +20
      [H1]  ChoCH terkonfirmasi             +10
      [H1]  BOS terkonfirmasi               + 5
      [H1]  EMA stack (9>21>50)             +10
      [M15] ChoCH terkonfirmasi             +20
      [M15] BOS terkonfirmasi               +10
      [M15] CISD terdeteksi                 +15
      [M15] Liquidity Sweep terdeteksi      +15
      [M15] Harga di OTE zone (61.8-78.6%)  +10
      [M15] Fibonacci diskon/premium zone   +10
      [M15] RSI Divergence                  +10 (+5 jika strong)
      ─────────────────────────────────────────
      MAX POSSIBLE                         ~160

    Penalti: jika setup M15 berlawanan dengan bias D1, dikurangi 50%.
    """
    h1  = build_df(df_h1)
    m15 = build_df(df_m15)
    if h1 is None or m15 is None:
        return None

    L1  = h1.iloc[-1]
    L15 = m15.iloc[-1]
    atr = max(float(L15["atr"]),
              float(L1["atr"]) / 4,
              float(L15["close"]) * 0.003)

    sh1,  sl1  = swing_pts(h1,  lb=5)
    sh15, sl15 = swing_pts(m15, lb=5)
    struct_h1  = _market_structure(h1,  sh1,  sl1)

    # ── D1 bias ──────────────────────────────────────────────────
    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            d1 = build_df(df_d1)
        else:
            d1 = build_df(
                df_h1.resample("1D").agg(
                    {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
                ).dropna()
            )
        if d1 is not None and len(d1) >= 10:
            LD       = d1.iloc[-1]
            shd, sld = swing_pts(d1, lb=3)
            sd1      = _market_structure(d1, shd, sld)
            bull_d1  = sd1 == "bullish" or (
                LD["ema9"] > LD["ema21"] > LD["ema50"])
            bear_d1  = sd1 == "bearish" or (
                LD["ema9"] < LD["ema21"] < LD["ema50"])
            if bull_d1:
                d1_bias = "bullish"
            elif bear_d1:
                d1_bias = "bearish"
    except Exception:
        pass

    # ── H1 indikator ─────────────────────────────────────────────
    ema_h1_bull = L1["ema9"] > L1["ema21"] > L1["ema50"]
    ema_h1_bear = L1["ema9"] < L1["ema21"] < L1["ema50"]
    choch_h1    = detect_choch(h1, sh1, sl1)
    bos_h1      = detect_bos(h1, sh1, sl1)

    # ── M15 indikator ─────────────────────────────────────────────
    choch_m15 = detect_choch(m15, sh15, sl15)
    bos_m15   = detect_bos(m15, sh15, sl15)
    cisd_m15  = detect_cisd(m15, lb=8)
    liq_bull  = detect_liquidity_sweep(m15, sh15, sl15, "bull")
    liq_bear  = detect_liquidity_sweep(m15, sh15, sl15, "bear")
    fr_m15    = detect_failed_retest(m15, sh15, sl15, atr)

    # ── Fibonacci M15 context ─────────────────────────────────────
    fib_sh = float(m15["high"].iloc[sh15[-1]]) if sh15 else None
    fib_sl = float(m15["low"].iloc[sl15[-1]])  if sl15 else None
    fib_r  = fib_position(float(L15["close"]), fib_sl, fib_sh) \
             if (fib_sh and fib_sl) else 0.5

    ote_bull = is_in_ote(float(L15["close"]), fib_sl or 0,
                         fib_sh or 1, "bull") if (fib_sh and fib_sl) else False
    ote_bear = is_in_ote(float(L15["close"]), fib_sl or 0,
                         fib_sh or 1, "bear") if (fib_sh and fib_sl) else False

    in_discount = fib_r < 0.45   # zona diskon → cocok untuk buy
    in_premium  = fib_r > 0.55   # zona premium → cocok untuk sell

    # ── RSI divergence M15 ───────────────────────────────────────
    rdiv_bull = detect_rsi_divergence(m15, "bull", lb=30)
    rdiv_bear = detect_rsi_divergence(m15, "bear", lb=30)

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

    # ── Penalti cross-HTF ─────────────────────────────────────────
    # Jika setup M15 berlawanan dengan D1 → kurangi 50%
    if d1_bias == "bullish" and bear > bull:
        bear = int(bear * 0.5)
    elif d1_bias == "bearish" and bull > bear:
        bull = int(bull * 0.5)

    direction  = "bull" if bull >= bear else "bear"
    raw        = bull if direction == "bull" else bear
    MAX_SCORE  = 165
    confidence = min(int(raw / MAX_SCORE * 100), 99)

    return {
        "direction":      direction,
        "confidence":     confidence,
        "price":          float(L15["close"]),
        "atr":            atr,
        "struct_h1":      struct_h1,
        "d1_bias":        d1_bias,
        "choch_m15":      choch_m15,
        "choch_h1":       choch_h1,
        "cisd_m15":       cisd_m15,
        "bos_m15":        bos_m15,
        "bos_h1":         bos_h1,
        "failed_retest":  fr_m15,         # compat: main.py akses key ini
        "liquidity_bull": liq_bull,
        "liquidity_bear": liq_bear,
        "sh15": sh15, "sl15": sl15,
        "sh1":  sh1,  "sl1":  sl1,
        "fib_r": round(fib_r, 3),
        "ote_bull": ote_bull,
        "ote_bear": ote_bear,
    }


# =============================================================================
# STEP 1 — ENTRY CANDIDATES
# =============================================================================

def _collect_entry_candidates(m15: pd.DataFrame, h1: pd.DataFrame,
                               direction: str, current_price: float,
                               atr: float, score_ctx: dict) -> list:
    """
    Kumpulkan kandidat entry berdasarkan konsep SMC/ICT.

    PRIORITAS (score tertinggi = paling diprioritaskan):
      1. OB berkualitas + Liquidity Sweep sebelumnya         (12-15)
      2. OB berkualitas + ChoCH konfirmasi                   ( 9-12)
      3. OB berkualitas + OTE zone                           ( 8-10)
      4. OB berkualitas saja                                 ( 5- 8)
      5. FVG fresh + CISD konfirmasi + LiqSweep              ( 6- 9)
      6. FVG fresh saja                                      ( 3- 5)
      7. Equal Highs/Lows (liquidity target)                 ( 2- 4)
      8. Market entry fallback (confidence tinggi saja)      (   1 )

    Entry untuk Bull OB: harga di ob_top (top edge of demand zone)
    Entry untuk Bear OB: harga di ob_bot (bottom edge of supply zone)
    """
    up     = direction == "bull"
    cands  = []

    liq    = score_ctx.get("liquidity_bull" if up else "liquidity_bear", {})
    choch  = score_ctx.get("choch_m15", {})
    cisd   = score_ctx.get("cisd_m15",  {})
    sh15   = score_ctx.get("sh15", [])
    sl15   = score_ctx.get("sl15", [])

    choch_ok = choch.get("bullish_choch") if up else choch.get("bearish_choch")
    cisd_ok  = cisd.get("bullish_cisd")  if up else cisd.get("bearish_cisd")
    liq_ok   = liq.get("type") == "sweep"

    # Fibonacci context M15 untuk OTE check
    fib_sh = float(m15["high"].iloc[sh15[-1]]) if sh15 else None
    fib_sl = float(m15["low"].iloc[sl15[-1]])  if sl15 else None

    # ── Kandidat dari Order Block ─────────────────────────────────
    obs = detect_order_block(m15, direction, lb=60, sh=sh15, sl=sl15)
    for z in obs:
        # Untuk bull: entry di atas OB (ob_top = top edge demand zone)
        # Untuk bear: entry di bawah OB (ob_bot = bottom edge supply zone)
        entry_pt   = float(z["top"]) if up else float(z["bot"])
        invalid_pt = float(z["bot"]) if up else float(z["top"])

        # Jangan entry jika harga sudah terlalu jauh dari OB zone
        if up and current_price < z["bot"] * 0.99:
            continue   # price sudah breakdown jauh dari OB
        if not up and current_price > z["top"] * 1.01:
            continue

        sc = 3 + z["quality"]   # base 3 + quality (2-6) = range 5-9

        # Bonus: setelah Liquidity Sweep (setup paling ideal dari transkrip)
        if liq_ok:
            sweep_lev = liq.get("level", 0)
            if up and entry_pt >= float(sweep_lev) * 0.995:
                sc += 3
            elif not up and entry_pt <= float(sweep_lev) * 1.005:
                sc += 3

        # Bonus: ChoCH terkonfirmasi di M15
        if choch_ok:
            sc += 2

        # Bonus: OTE zone
        if fib_sh and fib_sl:
            if is_in_ote(entry_pt, fib_sl, fib_sh, direction):
                sc += 1

        cands.append({
            "price":   round(entry_pt, 8),
            "invalid": round(invalid_pt, 8),
            "label":   "ob",
            "score":   sc,
        })

    # ── Kandidat dari FVG ────────────────────────────────────────
    fvgs = detect_fvg(m15, direction, lb=50)
    for f in fvgs:
        if not f["is_fresh"]:
            continue
        entry_pt   = f["mid"]
        invalid_pt = f["top"] if up else f["bot"]

        sc = 3   # FVG base

        if cisd_ok:   sc += 2
        if liq_ok:    sc += 2
        if choch_ok:  sc += 1

        cands.append({
            "price":   round(entry_pt, 8),
            "invalid": round(invalid_pt, 8),
            "label":   "fvg",
            "score":   sc,
        })

    # ── Kandidat dari Equal Highs/Lows ───────────────────────────
    eqs = detect_equal_highs_lows(m15, "low" if up else "high", lb=80)
    for eq in eqs[:2]:
        invalid_pt = eq - atr * 0.8 if up else eq + atr * 0.8
        sc = 2
        if liq_ok: sc += 1

        cands.append({
            "price":   round(float(eq), 8),
            "invalid": round(float(invalid_pt), 8),
            "label":   "eq",
            "score":   sc,
        })

    # ── Market entry fallback ─────────────────────────────────────
    if not cands:
        invalid_pt = current_price - atr * 1.2 if up else current_price + atr * 1.2
        cands.append({
            "price":   round(current_price, 8),
            "invalid": round(float(invalid_pt), 8),
            "label":   "market",
            "score":   1,
        })

    # Urut score tertinggi pertama
    cands.sort(key=lambda c: -c["score"])
    return cands


# =============================================================================
# STEP 2 — SL STRUKTURAL
# =============================================================================

def _compute_sl(m15: pd.DataFrame, h1: pd.DataFrame, direction: str,
                entry: float, atr: float, liq_sweep: dict,
                invalid_level: Optional[float] = None) -> Tuple[float, float]:
    """
    Hitung SL yang tepat secara struktural.

    PRINSIP (dari instruksi user):
    'SL yang tersentuh = arah benar-benar salah dari analisa'
    → Tempatkan SL di level struktural yang jika ditembus,
      tren tidak lagi valid. Bukan sekadar ATR di bawah entry.

    Buffer anti-Liquidity Sweep (ATR × 0.35):
    → Cukup untuk selamat dari LS normal (wick biasa)
    → Jika SL tersentuh, itu bukan LS biasa tapi genuinely breakdown

    Kandidat (diurut, tightest valid dipilih untuk RR terbaik):
      1. Invalid level dari OB/FVG entry (paling presisi)
      2. M15 structural swing low/high terakhir yang valid
      3. Level yang disweep (jika ada LiqSweep sebelumnya)
      4. ATR fallback
    """
    up        = direction == "bull"
    sgn       = 1 if up else -1

    # Buffer anti-Liquidity-Sweep: dinaikkan 0.35→0.5 ATR.
    # 0.35 terlalu tipis — wick crypto normal bisa mencapai 0.4–0.6 ATR
    # tanpa harga benar-benar breakdown. 0.5 memberi ruang lebih lega
    # tanpa mengorbankan terlalu banyak RR.
    ls_buffer = atr * 0.5

    # SL tidak boleh lebih dekat dari 1.0 ATR (naik dari 0.8).
    # SL < 1 ATR dari entry hampir pasti terkena noise/spread harian.
    min_risk  = atr * 1.0

    # SL tidak boleh lebih jauh dari 4.5 ATR — di atas ini TP pool
    # kemungkinan besar tidak bisa mencapai RR 2.0 (no viable targets).
    max_risk  = atr * 4.5

    cands = []

    # Kandidat 1: invalid level dari OB/FVG
    if invalid_level is not None:
        sl_raw = invalid_level + (-ls_buffer if up else ls_buffer)
        risk   = abs(sl_raw - entry)
        if min_risk <= risk <= max_risk:
            cands.append(("ob_invalid", sl_raw, risk))

    # Kandidat 2: M15 swing struktural
    sh15, sl15 = swing_pts(m15, lb=3)
    if up and sl15:
        struct_low = float(m15["low"].iloc[sl15[-1]])
        if struct_low < entry:
            sl_raw = struct_low - ls_buffer
            risk   = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_m15", sl_raw, risk))
    elif not up and sh15:
        struct_high = float(m15["high"].iloc[sh15[-1]])
        if struct_high > entry:
            sl_raw = struct_high + ls_buffer
            risk   = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_m15", sl_raw, risk))

    # Kandidat 3: level yang disweep (jika ada LiqSweep)
    if liq_sweep and liq_sweep.get("type") == "sweep" and liq_sweep.get("level"):
        lev = float(liq_sweep["level"])
        if up and lev < entry:
            sl_raw = lev - ls_buffer
            risk   = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("ls_level", sl_raw, risk))
        elif not up and lev > entry:
            sl_raw = lev + ls_buffer
            risk   = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("ls_level", sl_raw, risk))

    # Kandidat 4: H1 structural swing (lebih luas, lebih tahan noise)
    sh1, sl1 = swing_pts(h1, lb=5)
    if up and sl1:
        h1_low = float(h1["low"].iloc[sl1[-1]])
        if h1_low < entry:
            sl_raw = h1_low - ls_buffer
            risk   = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_h1", sl_raw, risk))
    elif not up and sh1:
        h1_high = float(h1["high"].iloc[sh1[-1]])
        if h1_high > entry:
            sl_raw = h1_high + ls_buffer
            risk   = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_h1", sl_raw, risk))

    if cands:
        # Prioritas pemilihan SL (untuk meminimalkan false SL hit):
        #   ob_invalid  — invalidasi teknikal paling presisi (OB/FVG rusak = thesis salah)
        #   struct_h1   — swing H1 paling tahan noise (timeframe lebih tinggi)
        #   ls_level    — level sweep = structural juga
        #   struct_m15  — lebih rentan wick M15 daripada H1
        # Dalam satu prioritas yang sama → ambil yang LEBIH LEBAR
        # (lebih jauh dari entry = lebih tahan noise, lebih jarang false hit).
        _PRIO = {"ob_invalid": 0, "struct_h1": 1, "ls_level": 2, "struct_m15": 3}
        cands.sort(key=lambda x: (_PRIO.get(x[0], 9), -x[2]))   # priority asc, risk desc
        _, sl_price, risk = cands[0]
        return sl_price, risk

    # Fallback ATR (tidak ada kandidat struktural valid)
    sl_price = entry + (-min_risk if up else min_risk)
    return sl_price, min_risk


# =============================================================================
# STEP 3 — TP POOL DAN SELEKSI
# =============================================================================

def _build_tp_pool(h1: pd.DataFrame, m15: pd.DataFrame, direction: str,
                   entry: float, atr: float,
                   sh1: list, sl1: list, sh15: list, sl15: list) -> list:
    """
    Bangun pool target TP dari berbagai sumber, terurut terdekat ke terjauh.

    Sumber (tier = prioritas, lebih kecil = lebih diutamakan):
      Tier 1: Equal Highs/Lows M15  (internal liquidity terdekat)
      Tier 2: OB H1 edge             (supply/demand zone H1)
      Tier 3: FVG H1 mid             (imbalance H1)
      Tier 4: Swing H1               (external liquidity H1)
      Tier 5: Equal Highs/Lows H1    (internal liquidity H1)
      Tier 6: Fibonacci extension    (1.272 dan 1.618 dari impulse leg)
      Tier 7: Fibonacci 2.0 ext      (target jauh untuk RR extension)
    """
    up   = direction == "bull"
    sgn  = 1 if up else -1
    pool = []

    # Tier 1: EQ M15
    eqs_m15 = detect_equal_highs_lows(m15, "high" if up else "low", lb=80)
    for v in eqs_m15:
        if sgn * (v - entry) > atr * 0.3:
            pool.append(("eq_m15", v, 1))

    # Tier 2: OB H1 (edge berlawanan arah = resistance/support untuk TP)
    # Untuk bull TP: cari supply zone H1 di atas entry
    # Untuk bear TP: cari demand zone H1 di bawah entry
    opp_dir  = "bear" if up else "bull"
    obs_h1_opp = detect_order_block(h1, opp_dir, lb=80, sh=sh1, sl=sl1)
    for z in obs_h1_opp:
        edge = float(z["bot"]) if up else float(z["top"])
        if sgn * (edge - entry) > atr * 0.5:
            pool.append(("ob_h1", edge, 2))

    # Tier 3: FVG H1 (arah yang berlawanan = area yang masih perlu diisi)
    fvgs_h1 = detect_fvg(h1, opp_dir, lb=60)
    for f in fvgs_h1:
        if sgn * (f["mid"] - entry) > atr * 0.5:
            pool.append(("fvg_h1", f["mid"], 3))

    # Tier 4: Swing H1 (previous swing high/low = external liquidity target)
    sw_vals = ([float(h1["high"].iloc[i]) for i in sh1] if up
               else [float(h1["low"].iloc[i])  for i in sl1])
    for v in sw_vals:
        if sgn * (v - entry) > atr * 1.0:
            pool.append(("sw_h1", v, 4))

    # Tier 5: EQ H1
    eqs_h1 = detect_equal_highs_lows(h1, "high" if up else "low", lb=100)
    for v in eqs_h1:
        if sgn * (v - entry) > atr * 0.8:
            pool.append(("eq_h1", v, 5))

    # Tier 6 & 7: Fibonacci extensions dari impulse leg H1
    if sh1 and sl1:
        sh_val = float(h1["high"].iloc[sh1[-1]])
        sl_val = float(h1["low"].iloc[sl1[-1]])
        leg    = sh_val - sl_val
        if leg > 0:
            exts = [
                (FIB_EXT_1, "fib127", 6),
                (FIB_EXT_2, "fib162", 6),
                (1.0,       "fib200", 7),
                (1.414,     "fib241", 7),
            ]
            for ext, lbl, tier in exts:
                tp_v = (sh_val + leg * ext) if up else (sl_val - leg * ext)
                if sgn * (tp_v - entry) > atr * 0.5:
                    pool.append((lbl, tp_v, tier))

    # Sort by distance from entry (terdekat dulu)
    pool.sort(key=lambda x: abs(x[1] - entry))
    return pool


def _select_tp(pool: list, entry: float, risk: float,
               direction: str) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    """
    Pilih TP terbaik dari pool, dengan logika RR extension (instruksi user):

    1. Cari target dengan RR 2.0–4.0 (zona ideal) → ambil yang tier terkecil,
       jika tier sama ambil yang RR paling dekat ke 2.0 (lebih konservatif).

    2. Jika tidak ada di range ideal, tapi ada target RR > 4.0:
       → CAP ke RR 4.0 (sesuai instruksi user: "batasi hingga 1:4")

    3. Jika semua target RR < 2.0:
       → Cari lebih jauh (extend), user bilang JANGAN auto-tolak
       → Gunakan Fibonacci extension sebagai extended target
       → Jika masih tidak ada → return None (sinyal ditolak)

    Return: (tp_price, label, rr) atau (None, None, None)
    """
    if not pool:
        return None, None, None

    sgn       = 1 if direction == "bull" else -1
    qualified = []   # RR 2.0–4.0
    below_min = []   # RR < 2.0
    above_max = []   # RR > 4.0

    for lbl, v, tier in pool:
        if sgn * (v - entry) <= 0:
            continue
        rr = abs(v - entry) / max(risk, 1e-10)
        if MIN_RR <= rr <= MAX_RR:
            qualified.append((lbl, v, tier, rr))
        elif rr < MIN_RR:
            below_min.append((lbl, v, tier, rr))
        else:
            above_max.append((lbl, v, tier, rr))

    # 1. Ada target di zona ideal
    if qualified:
        # Prioritas: tier terkecil, lalu RR paling rendah (konservatif)
        best = min(qualified, key=lambda x: (x[2], x[3]))
        return round(best[1], 8), best[0], round(best[3], 2)

    # 2. Ada target terlalu jauh → cap ke 4.0
    if above_max:
        best    = min(above_max, key=lambda x: x[3])
        capped  = entry + sgn * risk * MAX_RR
        return round(capped, 8), best[0] + "_capped", MAX_RR

    # 3. Semua target terlalu dekat → sinyal tidak layak
    # (instruksi user: "jika ada koin yang RR-nya kurang dari 1:2,
    #  analisa kembali chartnya lalu lihat apakah masih bisa lebih naik")
    # Pool sudah berisi Fibonacci extensions jauh — jika tetap tidak ada
    # yang mencapai 2.0, maka sinyal memang tidak layak.
    return None, None, None


# =============================================================================
# FUNGSI UTAMA — Dipanggil oleh main.py
# =============================================================================

def full_analyze(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                 df_d1: Optional[pd.DataFrame] = None,
                 symbol: Optional[str] = None) -> Optional[dict]:
    """
    Analisa penuh satu koin dalam urutan: Entry → SL → TP.

    Args:
        df_h1:   DataFrame OHLCV H1 (minimal 60 candle)
        df_m15:  DataFrame OHLCV M15 (minimal 60 candle)
        df_d1:   DataFrame OHLCV D1 (opsional, untuk bias HTF)
        symbol:  Nama koin untuk logging (opsional)

    Returns:
        dict sinyal lengkap, atau None jika tidak ada setup valid.
    """
    try:
        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            return None

        if symbol:
            log.info(f"[{symbol}] h1={len(df_h1)} m15={len(df_m15)}")

        # ── Scoring & Arah ───────────────────────────────────────
        score = score_direction(df_h1, df_m15, df_d1)
        if score is None:
            if symbol:
                log.debug(f"[{symbol}] score_direction=None (data kurang)")
            return None

        direction  = score["direction"]
        cur_price  = score["price"]
        atr        = score["atr"]
        confidence = score["confidence"]
        up         = direction == "bull"

        if symbol:
            log.info(
                f"[{symbol}] dir={direction} conf={confidence}% "
                f"struct_h1={score['struct_h1']} d1={score['d1_bias']}"
            )

        # Build indicator DataFrames
        h1  = build_df(df_h1)
        m15 = build_df(df_m15)
        if h1 is None or m15 is None:
            return None

        # ── STEP 1: ENTRY ────────────────────────────────────────
        cands = _collect_entry_candidates(m15, h1, direction, cur_price, atr, score)
        if not cands:
            if symbol:
                log.debug(f"[{symbol}] no entry candidates")
            return None

        best       = cands[0]   # sorted by score desc
        entry      = best["price"]
        entry_lbl  = best["label"]
        invalid    = best["invalid"]

        # Market entry fallback hanya untuk confidence sangat tinggi
        if entry_lbl == "market" and confidence < 70:
            if symbol:
                log.debug(f"[{symbol}] market entry ditolak (conf={confidence}<70)")
            return None

        if symbol:
            log.info(
                f"[{symbol}] ENTRY={entry:.6f} label={entry_lbl} "
                f"score={best['score']}"
            )

        # ── STEP 2: SL ──────────────────────────────────────────
        liq_ctx = score["liquidity_bull"] if up else score["liquidity_bear"]
        sl_price, risk = _compute_sl(m15, h1, direction, entry, atr,
                                     liq_ctx, invalid)

        # Sanity check: SL harus di sisi yang benar dari entry
        if up  and sl_price >= entry:
            if symbol:
                log.debug(f"[{symbol}] SL={sl_price:.6f} ≥ entry={entry:.6f}, skip")
            return None
        if not up and sl_price <= entry:
            if symbol:
                log.debug(f"[{symbol}] SL={sl_price:.6f} ≤ entry={entry:.6f}, skip")
            return None
        if risk <= 0:
            return None

        if symbol:
            log.info(f"[{symbol}] SL={sl_price:.6f} risk={risk:.6f}")

        # ── STEP 3: TP ──────────────────────────────────────────
        sh1  = score.get("sh1",  [])
        sl1  = score.get("sl1",  [])
        sh15 = score.get("sh15", [])
        sl15 = score.get("sl15", [])

        tp_pool = _build_tp_pool(h1, m15, direction, entry, atr,
                                 sh1, sl1, sh15, sl15)
        tp_price, tp_lbl, rr = _select_tp(tp_pool, entry, risk, direction)

        # Fallback TP (hanya jika pool benar-benar kosong)
        if tp_price is None:
            sgn      = 1 if up else -1
            tp_price = entry + sgn * risk * MIN_RR
            tp_lbl   = "fallback_rr2"
            rr       = MIN_RR

        if symbol:
            log.info(f"[{symbol}] TP={tp_price:.6f} label={tp_lbl} RR={rr:.2f}")

        # Cegah TP yang sudah kelewatan
        if up  and cur_price >= tp_price:
            if symbol:
                log.debug(f"[{symbol}] TP sudah lewat (price={cur_price:.6f})")
            return None
        if not up and cur_price <= tp_price:
            if symbol:
                log.debug(f"[{symbol}] TP sudah lewat")
            return None

        # Cek RR minimum setelah semua proses
        if rr < MIN_RR:
            if symbol:
                log.debug(f"[{symbol}] RR={rr:.2f} < {MIN_RR}, skip")
            return None

        # ── BUILD SIGNAL ─────────────────────────────────────────
        rsi_val = round(float(m15["rsi"].iloc[-1]), 1)

        return {
            "symbol":       symbol,
            "original_dir": direction,
            "decision":     "BUY" if up else "SELL",
            "confidence":   confidence,
            "price":        cur_price,
            "entry":        round(entry, 8),
            "entry_label":  entry_lbl,
            "sl":           round(sl_price, 8),
            "tp":           round(tp_price, 8),
            "rr":           rr,
            "rsi":          rsi_val,
            "struct_h1":    score["struct_h1"],
            "d1_bias":      score.get("d1_bias", "neutral"),
            "choch_m15":    score["choch_m15"],
            "choch_h1":     score["choch_h1"],
            "cisd_m15":     score["cisd_m15"],
            "failed_retest": score.get("failed_retest", {}),
            "tp_sl_reason": (
                f"Entry@{entry:.5g}({entry_lbl}) | "
                f"SL@{sl_price:.5g}(struct) | "
                f"TP@{tp_price:.5g}({tp_lbl}) | RR={rr:.2f}"
            ),
        }

    except Exception as e:
        if symbol:
            log.error(f"[full_analyze] {symbol}: {e}", exc_info=True)
        return None


def get_best_signal(candidates: list) -> Optional[dict]:
    """
    Pilih sinyal terbaik dari list kandidat.
    Bobot: confidence + label bonus (ob > fvg > eq > market) + RR × 0.5
    """
    if not candidates:
        return None

    label_bonus = {"ob": 4, "fvg": 2, "eq": 1, "market": 0}

    def _rank(sig):
        bonus = label_bonus.get(sig.get("entry_label", ""), 0)
        return sig["confidence"] + bonus + sig.get("rr", 0) * 0.5

    return max(candidates, key=_rank)
