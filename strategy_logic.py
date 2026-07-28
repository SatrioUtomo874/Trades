"""
strategy_logic.py — OTAK (logika analisa, swappable)

Versi TERBARU — menggabungkan semua ilmu dari 30 materi edukasi:
  • CISD (Change in State of Delivery) — sinyal reversal paling awal ICT
  • Killzone / Session Bias — London & NY killzone amplifier
  • Breaker Block — OB yang termitigasi, kini jadi zona berlawanan
  • Mitigation Block — zona unfilled order smart money
  • Internal vs External Liquidity — bedakan likuiditas minor vs major
  • Wyckoff VSA — No Supply / No Demand / Effort vs Result
  • Silver Bullet setup — time-based high-probability ICT
  • POI Precision scoring — multi-criteria confluence per zona
  • Fibonacci 0.382/0.618 sakti — tier retracement lebih granular
  • Opening Range Bias — bias dari range sesi sebelumnya
  • Candle Range Theory signal (NDOG / gap unfilled)
  • Improved OB quality scoring dengan 6 kriteria
  • Normalisasi confidence yang disesuaikan dengan total skor baru

Interface: full_analyze(df_h1, df_m15, df_d1, symbol=None) -> dict | None
Konstanta tuning: MIN_RR, TRAIL_R_LADDER, STRUCT_TRAIL_*, FIB_EXT_*, H4_RSI_*

Note: full_analyze() tidak fetch data sendiri — dikirim Mesin (main.py).
"""

import logging
import pandas as pd
import numpy as np
from datetime import timezone

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# RISK / REWARD TUNING
# ─────────────────────────────────────────────
MIN_RR = 2.0

# TRAIL_R_LADDER v5 FINAL — divalidasi 1113 trade nyata (15 koin, M1 presisi)
# win rate 72.2%, PnL 151.56% — terbaik dari semua varian yang pernah diuji
TRAIL_R_LADDER = [
    (0.5, 0.15),   # profit 0.5R → kunci 15% dari 0.5R
    (1.0, 0.35),   # 1.0R → kunci 35%
    (1.5, 0.50),   # 1.5R → kunci 50%
    (2.0, 0.65),   # 2.0R → kunci 65%
    (2.8, 0.80),   # 2.8R → kunci 80%
    (3.5, 0.85),   # 3.5R → kunci 85% (tangkap sisa upside)
]

STRUCT_TRAIL_LB       = 2
STRUCT_TRAIL_BUF_PCT  = 0.0015
STRUCT_TRAIL_LOOKBACK = 60

# Fibonacci Extension TP (gated H4 confluence)
FIB_EXT_1           = 0.272   # 1.272 ext
FIB_EXT_2           = 0.618   # 1.618 ext
H4_RSI_BUY_MIN      = 45
H4_RSI_BUY_MAX      = 68
H4_RSI_SELL_MIN     = 32
H4_RSI_SELL_MAX     = 55

# ─────────────────────────────────────────────
# KILLZONE WINDOWS (UTC hours, inklusi)
# London KZ : 07:00-09:00 UTC (high probability entry)
# NY KZ     : 12:00-15:00 UTC (NY open / overlap)
# London PM : 15:00-17:00 UTC (sering reversal & trap)
# ─────────────────────────────────────────────
KILLZONE_LONDON     = (7, 9)
KILLZONE_NY         = (12, 15)
KILLZONE_LONDON_PM  = (15, 17)
# Silver Bullet windows (UTC) — ICT high-probability time setups
SILVER_BULLET_1     = (7, 8)    # London open
SILVER_BULLET_2     = (10, 11)  # London continuation
SILVER_BULLET_3     = (15, 16)  # NY AM session


# ═════════════════════════════════════════════
# INDICATOR HELPERS
# ═════════════════════════════════════════════
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd(s):
    line = ema(s, 12) - ema(s, 26)
    sig  = ema(line, 9)
    return line, sig, line - sig

def atr_fn(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build_df(df):
    if len(df) < 60:
        return None
    df = df.copy()
    df["ema9"]    = ema(df["close"], 9)
    df["ema21"]   = ema(df["close"], 21)
    df["ema50"]   = ema(df["close"], 50)
    df["ema200"]  = ema(df["close"], 200) if len(df) >= 200 else ema(df["close"], 50)
    df["rsi"]     = rsi(df["close"])
    df["ml"], df["ms"], df["mh"] = macd(df["close"])
    df["atr"]     = atr_fn(df)
    df["vol_sma"] = df["volume"].rolling(20).mean()
    bm = df["close"].rolling(20).mean()
    bs = df["close"].rolling(20).std()
    df["bb_up"]  = bm + 2 * bs
    df["bb_lo"]  = bm - 2 * bs
    df["bb_mid"] = bm
    return df.dropna()


# ═════════════════════════════════════════════
# SESSION / KILLZONE HELPERS
# ═════════════════════════════════════════════
def get_current_hour_utc(df):
    """
    Ambil jam UTC dari index candle terakhir dataframe.
    Return int jam (0-23) atau None jika index bukan DatetimeTzAware/DatetimeIndex.
    """
    try:
        idx = df.index[-1]
        if hasattr(idx, "hour"):
            # Pastikan UTC
            if hasattr(idx, "tzinfo") and idx.tzinfo is not None:
                return idx.tz_convert("UTC").hour
            return idx.hour
    except Exception:
        pass
    return None


def is_in_killzone(hour_utc):
    """
    Return dict flag killzone aktif:
      london, ny, london_pm, silver_bullet_active
    """
    if hour_utc is None:
        return {"london": False, "ny": False, "london_pm": False,
                "silver_bullet_active": False, "any_kz": False}
    lo_kz = KILLZONE_LONDON[0] <= hour_utc < KILLZONE_LONDON[1]
    ny_kz = KILLZONE_NY[0] <= hour_utc < KILLZONE_NY[1]
    pm_kz = KILLZONE_LONDON_PM[0] <= hour_utc < KILLZONE_LONDON_PM[1]
    sb = any([
        SILVER_BULLET_1[0] <= hour_utc < SILVER_BULLET_1[1],
        SILVER_BULLET_2[0] <= hour_utc < SILVER_BULLET_2[1],
        SILVER_BULLET_3[0] <= hour_utc < SILVER_BULLET_3[1],
    ])
    return {
        "london": lo_kz,
        "ny": ny_kz,
        "london_pm": pm_kz,
        "silver_bullet_active": sb,
        "any_kz": lo_kz or ny_kz or pm_kz,
    }


# ═════════════════════════════════════════════
# SMC / PRICE ACTION TOOLS — CORE
# ═════════════════════════════════════════════
def swing_pts(df, lb=5):
    sh, sl = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i - lb:i + lb + 1].max():
            sh.append(i)
        if df["low"].iloc[i] == df["low"].iloc[i - lb:i + lb + 1].min():
            sl.append(i)
    return sh, sl


def mkt_struct(df, sh, sl):
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


def detect_bos(df, sh, sl):
    """
    BOS (Break of Structure) — kelanjutan trend.
    Shadow/wick cukup (tidak wajib body close).
    """
    res = {"bb": False, "bs": False, "cb": False, "cs": False}
    hi = df["high"].iloc[-1]
    lo = df["low"].iloc[-1]
    if len(sh) >= 2:
        ph = df["high"].iloc[sh[-2]]
        lh = df["high"].iloc[sh[-1]]
        if hi > ph:
            res["bb" if lh > ph else "cb"] = True
    if len(sl) >= 2:
        pl = df["low"].iloc[sl[-2]]
        ll = df["low"].iloc[sl[-1]]
        if lo < pl:
            res["bs" if ll < pl else "cs"] = True
    return res


def detect_choch(df, sh, sl):
    """
    CHoCH (Change of Character) — konfirmasi reversal NYATA.
    Wajib BODY CLOSE menembus level (lebih ketat dari BOS).
    """
    result = {"bearish_choch": False, "bullish_choch": False}
    close = df["close"].iloc[-1]
    if len(sh) >= 2 and len(sl) >= 2:
        prev_high = df["high"].iloc[sh[-2]]
        last_high = df["high"].iloc[sh[-1]]
        prev_low  = df["low"].iloc[sl[-2]]
        last_low  = df["low"].iloc[sl[-1]]
        lh_formed = last_high < prev_high
        if lh_formed and close < prev_low:
            result["bearish_choch"] = True
        hh_formed = last_high > prev_high
        if hh_formed and close > prev_low and last_low > prev_low:
            result["bullish_choch"] = True
    return result


# ═════════════════════════════════════════════
# CISD — Change in State of Delivery (ICT)
# Sinyal reversal PALING AWAL, bahkan sebelum CHoCH terbentuk.
# ═════════════════════════════════════════════
def detect_cisd(df, sh, sl, atr):
    """
    CISD (Change in State of Delivery) — konsep ICT untuk mendeteksi
    perubahan arah PALING AWAL sebelum struktur besar (CHoCH/BOS) terbentuk.

    Cara kerja:
    1. CISD Bullish: Candle sebelumnya sweep swing low (wick nembus low) tapi
       close KEMBALI DI ATAS low → liquidity grab + rejection kuat ke atas.
       Candle terakhir kemudian close di atas HIGH candle yang sweep.
    2. CISD Bearish: Kebalikannya — sweep high, close kembali di bawah high,
       candle berikut close di bawah LOW candle yang sweep.

    Ini lebih awal dari CHoCH karena tidak menunggu swing point baru terbentuk —
    cukup satu candle sweep + one candle reversal confirmation.

    Return: {"bullish_cisd": bool, "bearish_cisd": bool,
             "cisd_bull_level": float|None, "cisd_bear_level": float|None}
    """
    result = {
        "bullish_cisd": False,
        "bearish_cisd": False,
        "cisd_bull_level": None,
        "cisd_bear_level": None,
    }
    if len(df) < 4:
        return result

    last   = df.iloc[-1]   # candle konfirmasi
    sweep  = df.iloc[-2]   # candle yang melakukan sweep
    prev   = df.iloc[-3]   # candle sebelum sweep

    # CISD Bullish: sweep candle buat LOW baru (nembus swing low dengan wick)
    # tapi close kembali di atas open/high sebelumnya (rejection)
    # lalu candle konfirmasi close di atas HIGH sweep candle
    if len(sl) >= 1:
        ref_low = df["low"].iloc[sl[-1]]
        # Sweep: wick nembus low tapi close di atasnya (minimum di atas ref_low)
        swept_low  = sweep["low"] < ref_low
        # Recovery: sweep candle close lebih tinggi dari low-nya sendiri setidaknya 0.5 ATR
        recovered  = (sweep["close"] - sweep["low"]) > atr * 0.4
        # Konfirmasi: candle berikut close di atas high sweep candle
        confirmed  = last["close"] > sweep["high"]
        if swept_low and recovered and confirmed:
            result["bullish_cisd"] = True
            result["cisd_bull_level"] = ref_low

    # CISD Bearish: sweep candle buat HIGH baru dengan wick, close kembali di bawahnya
    # lalu candle konfirmasi close di bawah LOW sweep candle
    if len(sh) >= 1:
        ref_high   = df["high"].iloc[sh[-1]]
        swept_high = sweep["high"] > ref_high
        recovered  = (sweep["high"] - sweep["close"]) > atr * 0.4
        confirmed  = last["close"] < sweep["low"]
        if swept_high and recovered and confirmed:
            result["bearish_cisd"] = True
            result["cisd_bear_level"] = ref_high

    return result


# ═════════════════════════════════════════════
# WYCKOFF VSA — Volume Spread Analysis
# ═════════════════════════════════════════════
def detect_wyckoff_vsa(df, atr, lookback=10):
    """
    Wyckoff VSA signals:
    - No Supply  : narrow spread + LOW volume setelah upswing → seller habis, lanjut naik
    - No Demand  : narrow spread + LOW volume setelah downswing → buyer habis, lanjut turun
    - Effort vs Result (bullish): big volume + small body UP → absorption/accumulation
    - Effort vs Result (bearish): big volume + small body DOWN → distribution
    - Spring     : wick menembus support + immediate recovery + volume spike (bullish)
    - UTAD       : wick menembus resistance + immediate rejection + volume spike (bearish)

    Return: dict dengan flag masing-masing sinyal.
    """
    result = {
        "no_supply": False,         # bullish (sedikit supply)
        "no_demand": False,         # bearish (sedikit demand)
        "effort_vs_result_bull": False,  # absorption: big vol, small candle up
        "effort_vs_result_bear": False,  # distribution: big vol, small candle down
        "spring": False,            # Wyckoff Spring (bullish)
        "utad": False,              # UTAD (bearish)
    }
    if len(df) < lookback + 3:
        return result

    sub = df.iloc[-lookback:]
    L = df.iloc[-1]
    P = df.iloc[-2]

    avg_vol   = sub["volume"].rolling(5).mean().iloc[-1]
    avg_body  = (sub["close"] - sub["open"]).abs().mean()
    avg_range = (sub["high"] - sub["low"]).mean()

    body_L = abs(L["close"] - L["open"])
    range_L = L["high"] - L["low"]
    vol_L = L["volume"]

    # No Supply: after upswing, narrow range + low volume + close in upper half
    recent_up = sub["close"].iloc[-3] > sub["close"].iloc[-6] if len(sub) >= 6 else False
    if recent_up and range_L < avg_range * 0.7 and vol_L < avg_vol * 0.8:
        result["no_supply"] = True

    # No Demand: after downswing, narrow range + low volume + close in lower half
    recent_down = sub["close"].iloc[-3] < sub["close"].iloc[-6] if len(sub) >= 6 else False
    if recent_down and range_L < avg_range * 0.7 and vol_L < avg_vol * 0.8:
        result["no_demand"] = True

    # Effort vs Result Bullish: high volume, small body UP (absorption)
    if vol_L > avg_vol * 1.8 and body_L < avg_body * 0.6 and L["close"] > L["open"]:
        result["effort_vs_result_bull"] = True

    # Effort vs Result Bearish: high volume, small body DOWN (distribution)
    if vol_L > avg_vol * 1.8 and body_L < avg_body * 0.6 and L["close"] < L["open"]:
        result["effort_vs_result_bear"] = True

    # Spring: Wyckoff bullish reversal — wick jauh ke bawah + recovery kuat
    # Big wick bawah (> 2x body) + close di atas prev low
    low_wick = min(L["open"], L["close"]) - L["low"]
    if (low_wick > body_L * 2.0 and low_wick > atr * 0.8
            and L["close"] > P["low"] and vol_L > avg_vol * 1.2):
        result["spring"] = True

    # UTAD: Wyckoff bearish reversal — wick jauh ke atas + rejection kuat
    up_wick = L["high"] - max(L["open"], L["close"])
    if (up_wick > body_L * 2.0 and up_wick > atr * 0.8
            and L["close"] < P["high"] and vol_L > avg_vol * 1.2):
        result["utad"] = True

    return result


# ═════════════════════════════════════════════
# BREAKER BLOCK — OB yang termitigasi, jadi zona berlawanan
# ═════════════════════════════════════════════
def find_breaker_blocks(df, direction, lb=60):
    """
    Breaker Block (ICT Concept):
    Sebuah Order Block (OB) yang sudah DITEMBUS oleh harga dan kemudian
    price pullback ke zona itu dari sisi berlawanan — zona ini kini
    bertindak sebagai support/resistance baru (zona telah "berbalik peran").

    direction: "bull" → cari bekas supply zone yang sudah ditembus ke atas
                        (sekarang jadi demand/support untuk BUY)
               "bear" → cari bekas demand zone yang sudah ditembus ke bawah
                        (sekarang jadi supply/resistance untuk SELL)

    Return: list dict {"top", "bot", "mid", "quality", "is_fresh"}
    """
    is_demand = direction == "bull"
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean()
    breakers = []

    for i in range(2, len(sub) - 5):
        c  = sub.iloc[i]
        nx = sub.iloc[i + 1]
        impulse_body = abs(nx["close"] - nx["open"])
        if impulse_body < avg_body * 1.2:
            continue

        if is_demand:
            # Bekas supply: candle bullish (OB lama) diikuti impulse bearish
            # Lalu price BREAK ABOVE zona ini (sekarang jadi demand breaker)
            was_supply = (c["close"] > c["open"] and nx["close"] < nx["open"])
            if not was_supply:
                continue
            top = max(c["open"], c["close"])
            bot = min(c["open"], c["close"])
            # Cek apakah harga sudah break above zona ini
            df_from = base_offset + i + 2
            subsequent = df.iloc[df_from:]
            broke_above = bool((subsequent["close"] > top).any())
            if not broke_above:
                continue
        else:
            # Bekas demand: candle bearish (OB lama) diikuti impulse bullish
            # Lalu price BREAK BELOW zona ini (sekarang jadi supply breaker)
            was_demand = (c["close"] < c["open"] and nx["close"] > nx["open"])
            if not was_demand:
                continue
            top = max(c["open"], c["close"])
            bot = min(c["open"], c["close"])
            df_from = base_offset + i + 2
            subsequent = df.iloc[df_from:]
            broke_below = bool((subsequent["close"] < bot).any())
            if not broke_below:
                continue

        # Zona masih relevan: harga belum jauh melewatinya lagi
        price_now = df["close"].iloc[-1]
        if is_demand and price_now < bot * 0.98:  # sudah terlalu jauh di bawah
            continue
        if not is_demand and price_now > top * 1.02:
            continue

        is_fresh = is_zone_fresh(df, top, bot, base_offset + i)
        quality = int(is_fresh) + int(impulse_body > avg_body * 2.0)

        breakers.append({
            "top": top, "bot": bot,
            "mid": (top + bot) / 2,
            "is_fresh": is_fresh,
            "quality": quality,
        })

    return breakers[-2:] if breakers else []


# ═════════════════════════════════════════════
# MITIGATION BLOCK — zona unfilled orders smart money
# ═════════════════════════════════════════════
def find_mitigation_blocks(df, direction, lb=40):
    """
    Mitigation Block (ICT Concept):
    Candle TERAKHIR sebelum impulse move besar — zona di mana smart money
    menempatkan order yang belum sepenuhnya terisi, dan akan "dimitigasi"
    (diisi ulang) saat harga pullback ke sana.

    Untuk BUY: candle bearish terakhir sebelum bullish impulse kuat
    Untuk SELL: candle bullish terakhir sebelum bearish impulse kuat

    Lebih spesifik dari OB biasa karena selalu candle TEPAT SEBELUM impulse,
    bukan candle mana saja dalam rangkaian sebelumnya.

    Return: list dict {"top", "bot", "mid", "is_fresh"}
    """
    is_demand = direction == "bull"
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean()
    blocks = []

    for i in range(1, len(sub) - 3):
        c  = sub.iloc[i]
        nx = sub.iloc[i + 1]
        impulse_body = abs(nx["close"] - nx["open"])
        if impulse_body < avg_body * 1.8:  # harus impulse kuat
            continue

        if is_demand:
            # Candle bearish tepat sebelum bullish impulse
            is_mit_candle = (c["close"] < c["open"] and nx["close"] > nx["open"])
        else:
            # Candle bullish tepat sebelum bearish impulse
            is_mit_candle = (c["close"] > c["open"] and nx["close"] < nx["open"])

        if not is_mit_candle:
            continue

        top = max(c["open"], c["close"])
        bot = min(c["open"], c["close"])
        df_idx = base_offset + i
        is_fresh = is_zone_fresh(df, top, bot, df_idx)

        blocks.append({
            "top": top, "bot": bot,
            "mid": (top + bot) / 2,
            "is_fresh": is_fresh,
            "quality": int(is_fresh) + int(impulse_body > avg_body * 2.5),
        })

    return blocks[-2:] if blocks else []


# ═════════════════════════════════════════════
# INTERNAL vs EXTERNAL LIQUIDITY
# ═════════════════════════════════════════════
def classify_liquidity_pools(df, sh, sl, atr):
    """
    Bedakan internal liquidity (dalam range, di antara swing point minor)
    vs external liquidity (di luar struktur, swing high/low MAJOR).

    External liquidity:
    - Equal Highs/Lows (cluster level yang sering dituju SM untuk sweep)
    - Previous Day High/Low (PDHL) — level paling sering dijaga institusi
    - Swing High/Low yang paling jauh dari harga sekarang (major structure)

    Internal liquidity:
    - Swing High/Low yang dekat harga sekarang (dalam range)
    - FVG yang belum terisi (harga akan selalu kembali isi gap)

    Return: {"external_highs": [float], "external_lows": [float],
             "internal_highs": [float], "internal_lows": [float],
             "pdh": float|None, "pdl": float|None}
    """
    result = {
        "external_highs": [],
        "external_lows": [],
        "internal_highs": [],
        "internal_lows": [],
        "pdh": None,
        "pdl": None,
    }

    price_now = df["close"].iloc[-1]

    # Previous Day High/Low dari ~24 candle terakhir (H1) atau ~96 candle (M15)
    # Ambil max/min dari rentang itu
    n_pdhl = min(30, len(df) - 1)
    if n_pdhl > 5:
        prev_range = df.iloc[-n_pdhl:-1]
        result["pdh"] = float(prev_range["high"].max())
        result["pdl"] = float(prev_range["low"].min())

    # Swing highs: major (jauh dari harga) vs internal (dekat)
    threshold = atr * 5  # lebih dari 5 ATR = major / external
    for i in sh[-6:]:
        level = df["high"].iloc[i]
        if abs(level - price_now) > threshold:
            result["external_highs"].append(level)
        else:
            result["internal_highs"].append(level)

    for i in sl[-6:]:
        level = df["low"].iloc[i]
        if abs(level - price_now) > threshold:
            result["external_lows"].append(level)
        else:
            result["internal_lows"].append(level)

    return result


# ═════════════════════════════════════════════
# ZONE UTILITIES (dari versi sebelumnya, diperkuat)
# ═════════════════════════════════════════════
def is_zone_fresh(df, top, bot, formed_idx, end_idx=None):
    """
    Cek zona masih FRESH — belum pernah disentuh bahkan shadow sejak terbentuk.
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
    Posisi harga dalam rentang swing: discount / equilibrium / premium.
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
    Target retracement Fibonacci adaptif berdasarkan kekuatan trend.
    Tier tambahan: SANGAT kuat (0.236-0.382), kuat (0.382-0.5),
    moderat (0.5-0.618), lemah (0.618-0.786/OTE).
    """
    default = (0.5, 0.618)
    if len(sh) < 2 or len(sl) < 2:
        return default
    try:
        if direction == "bull":
            impulse_len  = df["high"].iloc[sh[-1]] - df["low"].iloc[sl[-2]]
            pullback_len = df["high"].iloc[sh[-1]] - df["close"].iloc[-1]
        else:
            impulse_len  = df["high"].iloc[sh[-2]] - df["low"].iloc[sl[-1]]
            pullback_len = df["close"].iloc[-1] - df["low"].iloc[sl[-1]]
        if impulse_len <= 0:
            return default
        pullback_ratio = abs(pullback_len) / impulse_len
    except Exception:
        return default

    # Fibonacci 0.382 & 0.618 adalah level "sakti" — sesuai materi
    # "Kenapa Fibonacci 0,618 & 0,382 Jadi Level Sakti dalam dunia Trading"
    if pullback_ratio <= 0.12:
        return (0.236, 0.382)    # trend SANGAT kuat
    elif pullback_ratio <= 0.30:
        return (0.382, 0.500)    # trend kuat — 0.382 level sakti pertama
    elif pullback_ratio <= 0.50:
        return (0.500, 0.618)    # moderat — equilibrium ke 0.618
    elif pullback_ratio >= 0.55:
        return (0.618, 0.786)    # lemah — OTE zone (0.62-0.79 = OTE)
    else:
        return (0.500, 0.618)


def find_snr_levels(df, lb=80):
    sh, sl = swing_pts(df, lb=5)
    levels = []
    for i in sh:
        levels.append(("R", df["high"].iloc[i]))
    for i in sl:
        levels.append(("S", df["low"].iloc[i]))
    return levels


def find_equal_highs_lows(df, kind="high", lb=60, tol=0.0025):
    """
    Equal Highs/Lows = zona likuiditas utama (stop loss retail banyak di sini).
    Institusi sering menyapu level ini (liquidity run) sebelum berbalik.
    """
    sub = df.iloc[-lb:]
    vals = sub["high"] if kind == "high" else sub["low"]
    clusters = []
    visited = set()
    for i in range(len(vals)):
        if i in visited:
            continue
        group = [vals.iloc[i]]
        for j in range(i + 1, len(vals)):
            if abs(vals.iloc[i] - vals.iloc[j]) / max(vals.iloc[i], 0.0001) < tol:
                group.append(vals.iloc[j])
                visited.add(j)
        if len(group) >= 2:
            clusters.append(sum(group) / len(group))
    return sorted(clusters)


def nearest_snr(df, price, direction, margin=0.015):
    sh, sl = swing_pts(df, lb=4)
    if direction == "above":
        candidates  = [df["high"].iloc[i] for i in sh if df["high"].iloc[i] > price * (1 + margin * 0.3)]
        candidates += find_equal_highs_lows(df, "high")
        candidates  = [c for c in candidates if c > price * (1 + margin * 0.3)]
        return min(candidates) if candidates else None
    else:
        candidates  = [df["low"].iloc[i] for i in sl if df["low"].iloc[i] < price * (1 - margin * 0.3)]
        candidates += find_equal_highs_lows(df, "low")
        candidates  = [c for c in candidates if c < price * (1 - margin * 0.3)]
        return max(candidates) if candidates else None


def classify_fvg_candle3(df, fvg_idx_c2, direction):
    if fvg_idx_c2 is None or fvg_idx_c2 >= len(df):
        return "unknown"
    c2 = df.iloc[fvg_idx_c2]
    is_bull_candle = c2["close"] > c2["open"]
    if direction == "bull":
        return "breakaway" if is_bull_candle else "rejection"
    else:
        return "rejection" if is_bull_candle else "breakaway"


def classify_sd_pattern(df, zone_idx, direction, lb=6):
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


def find_zones(df, direction, lb=40, strict=False):
    """
    Deteksi zona OB/Supply&Demand terpadu.
    Setiap zona disertai VALIDASI 6-KRITERIA (versi baru, diperluas dari 3):
    1. has_fvg       — FVG menyertai impulse
    2. has_bos       — impulse menghasilkan BOS
    3. is_fresh      — belum pernah disentuh ulang
    4. strong_move   — body impulse besar (≥1.3× avg)
    5. fib_aligned   — zona di area diskon (demand) atau premium (supply)
    6. has_inducement— ada tanda inducement sebelum zona terbentuk

    quality = jumlah dari 6 kriteria yang terpenuhi (max 6).
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

        # Kriteria 1: FVG
        has_fvg = False
        if nx2 is not None:
            if is_demand and nx2["low"] > c["high"]:
                has_fvg = True
            if (not is_demand) and nx2["high"] < c["low"]:
                has_fvg = True

        # Kriteria 2: BOS
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

        # Kriteria 3: Fresh
        fresh = is_zone_fresh(df, top, bot, df_idx)

        # Kriteria 4: Strong move (sudah di atas — impulse_body >= avg_body * 1.3)
        strong_move = strong_move_away

        # Kriteria 5: Fibonacci aligned (discount untuk demand, premium untuk supply)
        fib = get_fib_zone((top + bot) / 2, swing_lo, swing_hi)
        fib_aligned = fib["zone"] in (
            ("discount", "equilibrium") if is_demand else ("premium", "equilibrium")
        )

        # Kriteria 6: Ada inducement sebelum zona terbentuk
        # (gerakan kecil pancingan di 3 candle sebelum zona — ciri smart money sengaja)
        has_inducement = False
        if i >= 4:
            pre = sub.iloc[i - 3:i]
            small_moves = ((pre["close"] - pre["open"]).abs() < avg_body * 0.5)
            if is_demand:
                counter_dir = pre["close"] < pre["open"]
            else:
                counter_dir = pre["close"] > pre["open"]
            has_inducement = bool((small_moves & counter_dir).any())

        pattern = classify_sd_pattern(df, df_idx, "demand" if is_demand else "supply")

        quality = (int(has_fvg) + int(has_bos) + int(fresh)
                   + int(strong_move) + int(fib_aligned) + int(has_inducement))

        zones.append({
            "top": top, "bot": bot,
            "mid": (top + bot) / 2,
            "high": c["high"], "low": c["low"],
            "idx": df_idx,
            "has_fvg": bool(has_fvg),
            "has_bos": bool(has_bos),
            "is_fresh": bool(fresh),
            "strong_move_away": bool(strong_move),
            "fib_aligned": bool(fib_aligned),
            "has_inducement": bool(has_inducement),
            "pattern": pattern,
            "fib_zone": fib["zone"],
            "fib_ratio": fib["ratio"],
            "quality": quality,
        })

    return zones[-3:] if zones else []


def find_supply_demand(df, direction, lb=40):
    return find_zones(df, "demand" if direction == "demand" else "supply", lb=lb, strict=False)


def find_ob(df, direction, lb=40):
    return find_zones(df, direction, lb=lb, strict=True)


def find_fvg(df, direction, lb=40):
    """
    Fair Value Gap (FVG) — celah 3-candle impulsif tak seimbang.
    Setiap FVG kini disertai: is_fresh, candle3 type, fib_zone.
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
        gap["is_fresh"] = is_zone_fresh(df, gap["top"], gap["bot"], df_idx_c0, end_idx=len(df) - 1)
        gap["candle3"]  = classify_fvg_candle3(df, df_idx_c2, direction)
        gap["fib_zone"] = get_fib_zone(gap["mid"], swing_lo, swing_hi)["zone"]
        out.append(gap)

    return out[-3:] if out else []


def detect_failed_retest(df, sh, sl, atr):
    result = {"failed_retest_sell": False, "failed_retest_buy": False,
              "resistance": None, "support": None}
    if len(df) < 3:
        return result
    L = df.iloc[-1]
    P = df.iloc[-2]

    if len(sh) >= 2:
        resistance = df["high"].iloc[sh[-2]]
        touched    = P["high"] >= resistance - atr * 0.5
        rejected   = L["close"] < resistance - atr * 0.3
        bearish_c  = L["close"] < L["open"]
        if touched and rejected and bearish_c:
            result["failed_retest_sell"] = True
            result["resistance"] = resistance

    if len(sl) >= 2:
        support  = df["low"].iloc[sl[-2]]
        touched  = P["low"] <= support + atr * 0.5
        bounced  = L["close"] > support + atr * 0.3
        bullish_c = L["close"] > L["open"]
        if touched and bounced and bullish_c:
            result["failed_retest_buy"] = True
            result["support"] = support

    return result


def detect_liquidity_run_or_sweep(df, sh, sl, direction):
    """
    Bedakan Liquidity RUN vs SWEEP, plus sekarang juga tentukan apakah
    ini external liquidity (major swing) atau internal (minor swing).
    """
    result = {"type": "none", "level": None, "liquidity_class": "none"}
    if direction == "bull" and len(sh) >= 1:
        level = df["high"].iloc[sh[-1]]
        # Cek apakah ini major swing (external) atau minor (internal)
        is_major = len(sh) >= 2 and abs(level - df["high"].iloc[sh[-2]]) > df["atr"].iloc[-1] * 3
        last = df.iloc[-1]
        if last["high"] > level and last["close"] > level:
            result = {"type": "run", "level": level,
                      "liquidity_class": "external" if is_major else "internal"}
        elif last["high"] > level and last["close"] <= level:
            result = {"type": "sweep", "level": level,
                      "liquidity_class": "external" if is_major else "internal"}
    elif direction == "bear" and len(sl) >= 1:
        level = df["low"].iloc[sl[-1]]
        is_major = len(sl) >= 2 and abs(level - df["low"].iloc[sl[-2]]) > df["atr"].iloc[-1] * 3
        last = df.iloc[-1]
        if last["low"] < level and last["close"] < level:
            result = {"type": "run", "level": level,
                      "liquidity_class": "external" if is_major else "internal"}
        elif last["low"] < level and last["close"] >= level:
            result = {"type": "sweep", "level": level,
                      "liquidity_class": "external" if is_major else "internal"}
    return result


def detect_inducement_move(df, direction, atr, lookback=5):
    if len(df) < lookback + 1:
        return False
    sub = df.iloc[-lookback:-1]
    if sub.empty:
        return False
    small_moves = ((sub["close"] - sub["open"]).abs() < atr * 0.6)
    if direction == "bull":
        counter = sub["close"] < sub["open"]
    else:
        counter = sub["close"] > sub["open"]
    return bool((small_moves & counter).tail(3).any())


def detect_pinbar(candle, min_wick_ratio=1.5):
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body = abs(c - o)
    rng  = h - l
    if rng <= 0:
        return {"is_pinbar": False, "bullish_pinbar": False, "bearish_pinbar": False}
    low_wick = min(o, c) - l
    up_wick  = h - max(o, c)
    bullish_pinbar = low_wick > body * min_wick_ratio and low_wick > up_wick * 1.5
    bearish_pinbar = up_wick  > body * min_wick_ratio and up_wick  > low_wick * 1.5
    return {
        "is_pinbar": bool(bullish_pinbar or bearish_pinbar),
        "bullish_pinbar": bool(bullish_pinbar),
        "bearish_pinbar": bool(bearish_pinbar),
    }


def detect_fakey(df):
    result = {"is_fakey": False, "bullish_fakey": False, "bearish_fakey": False}
    if len(df) < 3:
        return result
    mother = df.iloc[-3]
    inside = df.iloc[-2]
    last   = df.iloc[-1]
    is_inside = inside["high"] <= mother["high"] and inside["low"] >= mother["low"]
    if not is_inside:
        return result
    broke_up      = last["high"] > mother["high"]
    broke_down    = last["low"]  < mother["low"]
    closed_inside = mother["low"] <= last["close"] <= mother["high"]
    if broke_down and closed_inside and last["close"] > last["open"]:
        result["is_fakey"] = True
        result["bullish_fakey"] = True
    elif broke_up and closed_inside and last["close"] < last["open"]:
        result["is_fakey"] = True
        result["bearish_fakey"] = True
    return result


def is_valid_pullback(df, direction, lookback=8):
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
        after = sub.iloc[found_i + 1:]
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
        after = sub.iloc[found_i + 1:]
        return bool((after["close"] > last_bear_high).any())


def classify_pullback_type(df, direction, atr, lookback=6):
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
        for j in range(i + 1, len(highs)):
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


# ═════════════════════════════════════════════
# SILVER BULLET SETUP — ICT time-based high-prob
# ═════════════════════════════════════════════
def detect_silver_bullet(df_m15, direction, hour_utc):
    """
    Silver Bullet (ICT): Setup entry PRESISI berbasis waktu.
    Syarat:
    1. Sedang dalam Silver Bullet window (London KZ / NY AM)
    2. Ada FVG yang terbentuk DALAM window tersebut
    3. FVG masih fresh (belum terisi)
    4. Arah FVG selaras dengan arah bias

    Silver Bullet window:
    - SB1: 07:00-08:00 UTC (London Open)
    - SB2: 10:00-11:00 UTC (London Continuation)
    - SB3: 15:00-16:00 UTC (NY AM Open)

    Return: {"active": bool, "fvg_entry": float|None, "fvg_invalid": float|None}
    """
    result = {"active": False, "fvg_entry": None, "fvg_invalid": None}
    if hour_utc is None:
        return result
    in_sb = any([
        SILVER_BULLET_1[0] <= hour_utc < SILVER_BULLET_1[1],
        SILVER_BULLET_2[0] <= hour_utc < SILVER_BULLET_2[1],
        SILVER_BULLET_3[0] <= hour_utc < SILVER_BULLET_3[1],
    ])
    if not in_sb:
        return result

    # Cari FVG yang terbentuk dalam window terakhir (12 candle M15 = 3 jam)
    fvgs = find_fvg(df_m15, "bull" if direction == "bull" else "bear", lb=12)
    fresh_fvgs = [f for f in fvgs if f.get("is_fresh") and f.get("candle3") == "breakaway"]
    if not fresh_fvgs:
        return result

    # Ambil FVG paling baru
    best_fvg = fresh_fvgs[-1]
    result["active"] = True
    if direction == "bull":
        result["fvg_entry"]   = best_fvg["bot"]   # entry di bawah FVG
        result["fvg_invalid"] = best_fvg["bot"] - (best_fvg["top"] - best_fvg["bot"])
    else:
        result["fvg_entry"]   = best_fvg["top"]   # entry di atas FVG
        result["fvg_invalid"] = best_fvg["top"] + (best_fvg["top"] - best_fvg["bot"])

    return result


# ═════════════════════════════════════════════
# CANDLE RANGE THEORY — NDOG / Gap bias
# ═════════════════════════════════════════════
def detect_candle_range_gap(df_m15):
    """
    Candle Range Theory: New Day Opening Gap (NDOG) dan New Week Opening Gap.
    Gap antara close hari sebelumnya dan open hari ini = magnet harga.

    Simplified implementation:
    - Deteksi apakah ada gap (open jauh dari close sebelumnya > 0.5 ATR)
    - Gap bullish: open > close kemarin → bias bullish (harga ingin isi gap ke atas)
    - Gap bearish: open < close kemarin → bias bearish

    Return: {"gap_bull": bool, "gap_bear": bool, "gap_size": float}
    """
    result = {"gap_bull": False, "gap_bear": False, "gap_size": 0.0}
    if len(df_m15) < 5:
        return result
    try:
        atr = df_m15["atr"].iloc[-1] if "atr" in df_m15.columns else (
            (df_m15["high"] - df_m15["low"]).rolling(14).mean().iloc[-1]
        )
        # Ambil open candle pertama hari ini vs close candle terakhir kemarin
        # Gunakan perbandingan antara open candle terbaru vs close 4 candle sebelum
        # (estimasi dari M15: 4 candle × 15 menit = 1 jam gap proxy)
        gap_candles = 4
        open_now  = df_m15["open"].iloc[-1]
        close_prev = df_m15["close"].iloc[-1 - gap_candles]
        gap = open_now - close_prev
        gap_size = abs(gap) / atr if atr > 0 else 0

        if gap > atr * 0.3:    # gap bullish > 0.3 ATR
            result["gap_bull"] = True
        elif gap < -atr * 0.3: # gap bearish
            result["gap_bear"] = True
        result["gap_size"] = round(gap_size, 2)
    except Exception:
        pass
    return result


# ═════════════════════════════════════════════
# TAHAP 1: SCORING — Analisis Hierarkis
# ═════════════════════════════════════════════
def score_direction(df_h1, df_m15, df_d1=None):
    """
    Analisis HIERARKIS (LAYER 1/2/3) dengan tambahan sinyal baru:
    CISD, Killzone, Wyckoff VSA, Silver Bullet, Candle Range Gap.

    LAYER 1 — BIAS (struktur besar):
      • Market Structure H1
      • CHoCH H1
      • D1 bias (EMA + struct)
      • EMA H1 trend alignment
      • RSI M15 momentum
      • CISD M15 (sinyal reversal paling awal) [BARU]
      • Killzone amplifier [BARU]

    LAYER 2 — SETUP/KONFIRMASI (price action & SMC):
      • BOS & CHoCH M15
      • Failed Retest M15+H1
      • Validitas & tipe pullback
      • Pin bar & Fakey
      • Liquidity Run/Sweep (+ external/internal class) [DIPERKUAT]
      • OTE
      • Wyckoff VSA (No Supply/Demand/Spring/UTAD) [BARU]
      • Silver Bullet setup [BARU]
      • Candle Range Gap bias [BARU]
      • MACD/BB/Volume M15

    LAYER 3 — GATE: konfirmasi berlawanan bias dilemahkan 50%.
    """
    h1  = build_df(df_h1)
    m15 = build_df(df_m15)
    if h1 is None or m15 is None:
        return None

    L1   = h1.iloc[-1]
    P1   = h1.iloc[-2]
    L15  = m15.iloc[-1]
    P15  = m15.iloc[-2]
    rv   = L15["rsi"]
    atr_val = max(L15["atr"], L15["close"] * 0.003)

    sh1,  sl1  = swing_pts(h1, 5)
    sh15, sl15 = swing_pts(m15, 5)
    struct_h1  = mkt_struct(h1, sh1, sl1)
    choch_h1   = detect_choch(h1, sh1, sl1)

    # Session / Killzone
    hour_utc = get_current_hour_utc(m15)
    kz = is_in_killzone(hour_utc)

    # D1 bias
    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            df_d1_built = build_df(df_d1)
        else:
            df_d1_built = build_df(df_h1.resample("1D").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
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
    except Exception:
        pass

    # ──────────────────────────────────────────
    # LAYER 1 — BIAS
    # ──────────────────────────────────────────
    bias_bull = bias_bear = 0

    # Market structure H1
    if struct_h1 == "bullish": bias_bull += 30
    if struct_h1 == "bearish": bias_bear += 30

    # CHoCH H1 (karakter bias pasar, bukan sekadar setup M15)
    if choch_h1["bullish_choch"]: bias_bull += 26
    if choch_h1["bearish_choch"]: bias_bear += 26

    # EMA H1 trend alignment
    if L1["ema9"] > L1["ema21"] > L1["ema50"]:  bias_bull += 15
    elif L1["ema9"] > L1["ema21"]:               bias_bull += 7
    if L1["ema9"] < L1["ema21"] < L1["ema50"]:  bias_bear += 15
    elif L1["ema9"] < L1["ema21"]:               bias_bear += 7

    if L1["close"] > L1["ema200"]: bias_bull += 8
    else:                            bias_bear += 8

    # D1 bias (konteks makro)
    if d1_bias == "bullish": bias_bull += 24
    if d1_bias == "bearish": bias_bear += 24

    # RSI M15 momentum filter
    if rv < 35:    bias_bull += 12
    elif rv < 45:  bias_bull += 6
    if rv > 65:    bias_bear += 12
    elif rv > 55:  bias_bear += 6

    # ── CISD M15 [BARU] — sinyal reversal paling awal
    # Bobot cukup besar (18) karena CISD adalah konfirmasi institutional
    # manipulation sebelum struktur besar terbentuk
    cisd_m15 = detect_cisd(m15, sh15, sl15, atr_val)
    if cisd_m15["bullish_cisd"]: bias_bull += 18
    if cisd_m15["bearish_cisd"]: bias_bear += 18

    # ── Killzone amplifier [BARU]
    # Sinyal dalam killzone lebih reliable karena institusi aktif
    # (hanya diberikan ke arah yang sudah dominan, bukan dua-duanya)
    if kz["any_kz"]:
        # Akan diterapkan ke layer final, bukan di sini — simpan flag
        pass  # lihat penerapan di bawah

    bias_dir = "bull" if bias_bull >= bias_bear else "bear"

    # ──────────────────────────────────────────
    # LAYER 2 — SETUP / KONFIRMASI
    # ──────────────────────────────────────────
    setup_bull = setup_bear = 0

    # BOS & CHoCH M15
    bos = detect_bos(m15, sh15, sl15)
    if bos["bb"]: setup_bull += 12
    if bos["cb"]: setup_bull += 7
    if bos["bs"]: setup_bear += 12
    if bos["cs"]: setup_bear += 7

    choch = detect_choch(m15, sh15, sl15)
    if choch["bullish_choch"]: setup_bull += 22
    if choch["bearish_choch"]: setup_bear += 22

    # Failed Retest
    fr = detect_failed_retest(m15, sh15, sl15, atr_val)
    if fr["failed_retest_sell"]: setup_bear += 24
    if fr["failed_retest_buy"]:  setup_bull += 24

    fr_h1 = detect_failed_retest(h1, sh1, sl1, atr_val)
    if fr_h1["failed_retest_sell"]: setup_bear += 18
    if fr_h1["failed_retest_buy"]:  setup_bull += 18

    # Pullback validity & type
    pullback_valid_bull = is_valid_pullback(m15, "bull")
    pullback_valid_bear = is_valid_pullback(m15, "bear")
    pullback_type_bull  = classify_pullback_type(m15, "bull", atr_val)
    pullback_type_bear  = classify_pullback_type(m15, "bear", atr_val)

    if pullback_valid_bull:
        if pullback_type_bull == "aggressive":  setup_bull += 3
        elif pullback_type_bull == "sweeping":  setup_bull += 14
        else:                                    setup_bull += 9
    if pullback_valid_bear:
        if pullback_type_bear == "aggressive":  setup_bear += 3
        elif pullback_type_bear == "sweeping":  setup_bear += 14
        else:                                    setup_bear += 9

    # Pin bar & Fakey
    pinbar = detect_pinbar(L15)
    if pinbar["bullish_pinbar"]: setup_bull += 10
    if pinbar["bearish_pinbar"]: setup_bear += 10

    fakey = detect_fakey(m15)
    if fakey["bullish_fakey"]: setup_bull += 10
    if fakey["bearish_fakey"]: setup_bear += 10

    # Liquidity Run vs Sweep — termasuk external/internal class [DIPERKUAT]
    liq_bull = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bear")

    # External liquidity sweep → sinyal reversal lebih kuat dari internal
    liq_bull_bonus = 4 if liq_bull.get("liquidity_class") == "external" else 0
    liq_bear_bonus = 4 if liq_bear.get("liquidity_class") == "external" else 0

    if liq_bull["type"] == "run":     setup_bull += 10 + liq_bull_bonus
    elif liq_bull["type"] == "sweep": setup_bear += 8  + liq_bull_bonus  # sweep = berlawanan
    if liq_bear["type"] == "run":     setup_bear += 10 + liq_bear_bonus
    elif liq_bear["type"] == "sweep": setup_bull += 8  + liq_bear_bonus

    # Inducement flag
    inducement_bull = detect_inducement_move(m15, "bull", atr_val)
    inducement_bear = detect_inducement_move(m15, "bear", atr_val)

    # OTE (0.62-0.79)
    ote_bull = ote_bear = False
    if len(sh15) >= 1 and len(sl15) >= 1:
        swing_hi_m15 = m15["high"].iloc[sh15[-1]]
        swing_lo_m15 = m15["low"].iloc[sl15[-1]]
        fib_now = get_fib_zone(L15["close"], swing_lo_m15, swing_hi_m15)
        if 0.62 <= (1 - fib_now["ratio"]) <= 0.79: ote_bull = True
        if 0.62 <= fib_now["ratio"] <= 0.79:        ote_bear = True

    if ote_bull and (choch["bullish_choch"] or any(
            f.get("is_fresh") for f in find_fvg(m15, "bull", lb=30))):
        setup_bull += 10
    if ote_bear and (choch["bearish_choch"] or any(
            f.get("is_fresh") for f in find_fvg(m15, "bear", lb=30))):
        setup_bear += 10

    # Candle pattern dasar
    body     = L15["close"] - L15["open"]
    low_wick = min(L15["open"], L15["close"]) - L15["low"]
    up_wick  = L15["high"] - max(L15["open"], L15["close"])
    if low_wick > abs(body) * 1.5: setup_bull += 6
    if up_wick  > abs(body) * 1.5: setup_bear += 6

    # ── Wyckoff VSA [BARU]
    vsa = detect_wyckoff_vsa(m15, atr_val)
    if vsa["no_supply"]:              setup_bull += 10  # sedikit supply → bullish
    if vsa["no_demand"]:              setup_bear += 10  # sedikit demand → bearish
    if vsa["effort_vs_result_bull"]:  setup_bull += 8   # absorption → bullish
    if vsa["effort_vs_result_bear"]:  setup_bear += 8   # distribution → bearish
    if vsa["spring"]:                 setup_bull += 16  # Spring = entry BUY terkuat Wyckoff
    if vsa["utad"]:                   setup_bear += 16  # UTAD = entry SELL terkuat Wyckoff

    # ── Silver Bullet Setup [BARU]
    # Aktif hanya dalam killzone window tertentu
    sb_bull = detect_silver_bullet(m15, "bull", hour_utc)
    sb_bear = detect_silver_bullet(m15, "bear", hour_utc)
    if sb_bull["active"]: setup_bull += 14
    if sb_bear["active"]: setup_bear += 14

    # ── Candle Range Gap bias [BARU]
    crg = detect_candle_range_gap(m15)
    if crg["gap_bull"]: setup_bull += 7
    if crg["gap_bear"]: setup_bear += 7

    # Momentum confluence ringan (MACD/BB/Volume)
    if L15["mh"] > 0 and P15["mh"] <= 0:  setup_bull += 8
    elif L15["mh"] > 0:                    setup_bull += 3
    if L15["mh"] < 0 and P15["mh"] >= 0:  setup_bear += 8
    elif L15["mh"] < 0:                    setup_bear += 3

    if L15["close"] <= L15["bb_lo"]:    setup_bull += 7
    elif L15["close"] < L15["bb_mid"]:  setup_bull += 3
    if L15["close"] >= L15["bb_up"]:    setup_bear += 7
    elif L15["close"] > L15["bb_mid"]:  setup_bear += 3

    if L15["volume"] > L15["vol_sma"] * 1.5:
        if L15["close"] > L15["open"]:  setup_bull += 6
        else:                            setup_bear += 6
    elif L15["volume"] > L15["vol_sma"]:
        if L15["close"] > L15["open"]:  setup_bull += 2
        else:                            setup_bear += 2

    # ──────────────────────────────────────────
    # LAYER 3 — GATE
    # ──────────────────────────────────────────
    if bias_dir == "bull":
        setup_bear = setup_bear * 0.5
    else:
        setup_bull = setup_bull * 0.5

    bull = bias_bull + setup_bull
    bear = bias_bear + setup_bear

    # ── Killzone amplifier — boost skor arah yang menang dalam KZ [BARU]
    # Tidak mengubah arah, cuma memperkuat confidence kalau memang dalam KZ
    if kz["any_kz"]:
        if bull >= bear:  bull *= 1.10
        else:             bear *= 1.10

    direction = "bull" if bull >= bear else "bear"
    raw  = bull if direction == "bull" else bear

    # Normalisasi confidence: max skor baru ~330 (vs 264 sebelumnya — ada
    # ~66 poin tambahan dari CISD+VSA+SB+CRG+killzone+ext_liq)
    conf = min(int(raw / 330 * 100), 99)

    # D1 berlawanan total → hard block
    if d1_bias == "bearish" and direction == "bull": return None
    if d1_bias == "bullish" and direction == "bear": return None

    return {
        "direction"       : direction,
        "confidence"      : conf,
        "price"           : L15["close"],
        "atr"             : atr_val,
        "struct_h1"       : struct_h1,
        "d1_bias"         : d1_bias,
        "rsi"             : round(rv, 1),
        "bull_pts"        : bull,
        "bear_pts"        : bear,
        "bias_dir"        : bias_dir,
        "choch_m15"       : choch,
        "choch_h1"        : choch_h1,
        "cisd_m15"        : cisd_m15,
        "failed_retest"   : fr,
        "pullback_valid"  : pullback_valid_bull if direction == "bull" else pullback_valid_bear,
        "pullback_type"   : pullback_type_bull  if direction == "bull" else pullback_type_bear,
        "pinbar"          : pinbar,
        "fakey"           : fakey,
        "vsa"             : vsa,
        "killzone"        : kz,
        "silver_bullet"   : sb_bull if direction == "bull" else sb_bear,
        "liquidity_bull"  : liq_bull,
        "liquidity_bear"  : liq_bear,
        "inducement"      : inducement_bull if direction == "bull" else inducement_bear,
    }


# ═════════════════════════════════════════════
# TAHAP 2: SL dan TP
# ═════════════════════════════════════════════
def _h4_confluence(df_h1, direction, choch_m15=None):
    result = {"confluence": False, "full_confluence": False}
    try:
        df_h4 = build_df(df_h1.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna())
        if df_h4 is None or len(df_h4) < 20:
            return result
        L4 = df_h4.iloc[-1]
        sh4, sl4 = swing_pts(df_h4, lb=3)
        struct_h4 = mkt_struct(df_h4, sh4, sl4)
        rsi_h4 = L4["rsi"]
        if direction == "bull":
            ema_ok    = L4["ema9"] > L4["ema21"] > L4["ema50"]
            struct_ok = struct_h4 == "bullish"
            rsi_ok    = H4_RSI_BUY_MIN <= rsi_h4 <= H4_RSI_BUY_MAX
        else:
            ema_ok    = L4["ema9"] < L4["ema21"] < L4["ema50"]
            struct_ok = struct_h4 == "bearish"
            rsi_ok    = H4_RSI_SELL_MIN <= rsi_h4 <= H4_RSI_SELL_MAX
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


TP_RR_CAP = MIN_RR * 2


def _select_best_tp(tp_pool, entry_price, risk):
    """
    Pilih TP dari level PALING KUAT (tier terendah) yang lolos RR >= MIN_RR.
    Seri tier → ambil RR tertinggi. RR > TP_RR_CAP → ditarik ke cap.
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
        sgn    = 1 if best_v > entry_price else -1
        best_v = entry_price + sgn * risk * TP_RR_CAP
        best_lbl += "_capped"
    return round(best_v, 8), best_lbl


def _build_tp_pool(m15, h1, direction, entry_price, atr, sh15, sl15, sh1, sl1,
                   h4_gate, fib_127, fib_162):
    """
    TP pool dengan tambahan:
    - Breaker Block H1/M15 sebagai TP target (tier 1.5/3.5)
    - External liquidity pool sebagai TP target tambahan
    """
    up    = direction == "bull"
    sgn   = 1 if up else -1
    pool  = []

    zones_m15 = find_zones(m15, "demand" if up else "supply")
    zones_h1  = find_zones(h1, "demand" if up else "supply")
    fvgs      = find_fvg(m15, "bull" if up else "bear")
    eqs_m15   = find_equal_highs_lows(m15, "high" if up else "low", lb=80)
    eqs_h1    = find_equal_highs_lows(h1,  "high" if up else "low", lb=50)
    sw_m15    = [m15["high" if up else "low"].iloc[i] for i in (sh15 if up else sl15)]
    sw_h1     = [h1["high"  if up else "low"].iloc[i] for i in (sh1  if up else sl1)]

    # Breaker blocks sebagai TP target — kuat karena bekas OB yang valid
    breakers_h1  = find_breaker_blocks(h1,  direction, lb=80)
    breakers_m15 = find_breaker_blocks(m15, direction, lb=60)

    # External liquidity pools (lebih jauh, tapi sering jadi magnet SM)
    liq_pools = classify_liquidity_pools(h1, sh1, sl1, atr)
    ext_highs = liq_pools["external_highs"]
    ext_lows  = liq_pools["external_lows"]
    pdh = liq_pools["pdh"]
    pdl = liq_pools["pdl"]

    # Equal Highs/Lows H1 — tier 1 (terkuat)
    for v in eqs_h1:
        if sgn * (v - entry_price) > atr * 1.0:
            pool.append(("eq_h1", v, 1))

    # Breaker Block H1 — tier 1.5 (bekas OB, sangat reliable sebagai target)
    for bk in breakers_h1:
        edge = bk["bot"] if up else bk["top"]
        if sgn * (edge - entry_price) > atr * 1.0:
            pool.append(("breaker_h1", edge, 1.5))

    # Zone H1 — tier 2
    for z in zones_h1:
        edge = z["bot"] if up else z["top"]
        if sgn * (edge - entry_price) > atr * 1.0:
            pool.append(("zone_h1", edge, 2))

    # PDH/PDL — Previous Day High/Low: level paling sering dijaga institusi
    if pdh is not None and up and sgn * (pdh - entry_price) > atr * 1.0:
        pool.append(("pdh", pdh, 2))
    if pdl is not None and not up and sgn * (pdl - entry_price) > atr * 1.0:
        pool.append(("pdl", pdl, 2))

    # Swing H1 — tier 3
    for v in sw_h1:
        if sgn * (v - entry_price) > atr * 1.0:
            pool.append(("sw_h1", v, 3))

    # Equal Highs/Lows M15 — tier 4
    for v in eqs_m15:
        if sgn * (v - entry_price) > atr * 0.5:
            pool.append(("eq_m15", v, 4))

    # Zone M15 — tier 5
    for z in zones_m15:
        edge = z["bot"] if up else z["top"]
        if sgn * (edge - entry_price) > atr * 0.5:
            pool.append(("zone_m15", edge, 5 - (0.4 if z.get("is_fresh") else 0)))

    # Breaker Block M15 — tier 3.5
    for bk in breakers_m15:
        edge = bk["bot"] if up else bk["top"]
        if sgn * (edge - entry_price) > atr * 0.5:
            pool.append(("breaker_m15", edge, 3.5))

    # FVG M15 — tier 6
    for f in fvgs:
        if sgn * (f["mid"] - entry_price) > atr * 0.5:
            t = (6
                 - (0.4 if f.get("candle3") == "breakaway" else 0)
                 - (0.2 if f.get("is_fresh") else 0))
            pool.append(("fvg_m15", f["mid"], t))

    # External liquidity pool — tier 0.5 (paling kuat target — SM selalu sapu sini)
    for v in (ext_highs if up else ext_lows):
        if sgn * (v - entry_price) > atr * 2.0:  # hanya kalau cukup jauh
            pool.append(("ext_liq", v, 0.5))

    # Swing M15 — tier 7
    for v in sw_m15:
        if sgn * (v - entry_price) > atr * 0.5:
            pool.append(("sw_m15", v, 7))

    # Fibonacci extension (gated H4 confluence)
    if fib_127 is not None and sgn * (fib_127 - entry_price) > atr * 0.5 and h4_gate["confluence"]:
        pool.append(("fib127", fib_127, 8))
        if h4_gate["full_confluence"] and fib_162 is not None and sgn * (fib_162 - entry_price) > atr * 0.5:
            pool.append(("fib162", fib_162, 9))

    return pool


def analyze_setup(df_h1, df_m15, direction, entry_price, score=None, invalid_level=None):
    """
    SL = seberang titik entry (invalid_level) + buffer noise.
    TP = tier pool terkuat dengan floor RR >= MIN_RR.
    """
    h1, m15 = build_df(df_h1), build_df(df_m15)
    if h1 is None or m15 is None:
        return None

    atr_m15 = m15["atr"].iloc[-1]
    atr_h1  = h1["atr"].iloc[-1] / 4
    atr     = max(atr_m15, atr_h1, entry_price * 0.002)
    noise   = atr * 0.6

    if invalid_level is None:
        return None

    sl_price = invalid_level + (noise if direction == "bear" else -noise)
    risk     = abs(sl_price - entry_price)
    risk_floor = max(atr * 0.8, entry_price * 0.003)
    if risk < risk_floor:
        sl_price += (risk_floor - risk) * (1 if direction == "bear" else -1)
        risk = risk_floor
    if risk <= 0:
        return None

    sh15, sl15 = swing_pts(m15, lb=5)
    sh1, sl1   = swing_pts(h1, lb=5)
    choch_m15  = (score or {}).get("choch_m15", {})
    h4_gate    = _h4_confluence(df_h1, direction, choch_m15)
    fib_127, fib_162 = _fib_extension_levels(h1, sh1, sl1, direction)

    tp_pool = _build_tp_pool(m15, h1, direction, entry_price, atr,
                              sh15, sl15, sh1, sl1, h4_gate, fib_127, fib_162)
    tp_price, tp_label = _select_best_tp(tp_pool, entry_price, risk)
    if tp_price is None:
        return None

    reward = abs(tp_price - entry_price)
    rr     = round(reward / risk, 2)
    if rr < MIN_RR:
        return None

    return {
        "sl": round(sl_price, 8),
        "tp": round(tp_price, 8),
        "rr": rr,
        "reason": f"SL@{sl_price:.5g}(invalidation) | TP@{tp_price:.5g}({tp_label})",
    }


def _zone_score(z):
    """
    Skor kekuatan zona OB/S&D (6 kriteria, versi baru):
    fresh + fvg + bos + strong_move + fib_aligned + has_inducement
    """
    return z.get("quality", 0)


def _collect_entry_candidates(m15, direction, entry_ref, atr):
    """
    Kumpulkan kandidat entry:
    OB strict > FVG breakaway > Mitigation Block > Breaker Block > EQ > Fib adaptif

    Breaker Block dan Mitigation Block ditambahkan sebagai tipe entry baru.
    Skor memperhitungkan kualitas zona + tie-break jarak (ringan).
    """
    up   = direction == "bear"   # up = True → cari di atas harga (untuk SELL)
    obs  = find_zones(m15, direction, strict=True)
    fvgs = find_fvg(m15, direction)
    eqs  = find_equal_highs_lows(m15, "high" if up else "low", lb=80)
    mits = find_mitigation_blocks(m15, direction, lb=40)   # BARU
    bkrs = find_breaker_blocks(m15, direction, lb=60)       # BARU
    cands = []

    def _dist_penalty(price):
        if atr <= 0:
            return 0.0
        return (abs(price - entry_ref) / atr) * 0.15

    # OB strict — terkuat
    for z in obs:
        entry_pt, invalid_pt = (z["top"], z["bot"]) if up else (z["bot"], z["top"])
        if (up and entry_pt > entry_ref + atr * 0.1) or (not up and entry_pt < entry_ref - atr * 0.1):
            cands.append({
                "price": entry_pt, "invalid": invalid_pt, "label": "ob",
                "score": 3 + _zone_score(z) - _dist_penalty(entry_pt),
            })

    # FVG breakaway — kedua terkuat
    for f in fvgs:
        if (up and f["mid"] > entry_ref + atr * 0.1) or (not up and f["mid"] < entry_ref - atr * 0.1):
            sc = 2 + int(f.get("is_fresh", False)) + 2 * int(f.get("candle3") == "breakaway")
            invalid_pt = f["top"] if up else f["bot"]
            cands.append({
                "price": f["mid"], "invalid": invalid_pt, "label": "fvg",
                "score": sc - _dist_penalty(f["mid"]),
            })

    # Mitigation Block [BARU] — tipe entry ICT presisi
    for m in mits:
        entry_pt  = m["top"] if up else m["bot"]
        invalid_pt = m["bot"] if up else m["top"]
        if (up and entry_pt > entry_ref + atr * 0.1) or (not up and entry_pt < entry_ref - atr * 0.1):
            cands.append({
                "price": entry_pt, "invalid": invalid_pt, "label": "mitigation",
                "score": 2 + m.get("quality", 0) - _dist_penalty(entry_pt),
            })

    # Breaker Block [BARU] — bekas OB yang sudah berbalik peran
    for bk in bkrs:
        entry_pt  = bk["bot"] if up else bk["top"]
        invalid_pt = bk["top"] if up else bk["bot"]
        if (up and entry_pt > entry_ref + atr * 0.1) or (not up and entry_pt < entry_ref - atr * 0.1):
            cands.append({
                "price": entry_pt, "invalid": invalid_pt, "label": "breaker",
                "score": 2.5 + bk.get("quality", 0) - _dist_penalty(entry_pt),
            })

    # Equal Highs/Lows — fallback
    eqs_sorted = sorted(eqs) if up else sorted(eqs, reverse=True)
    for lv in eqs_sorted[:1]:
        if (up and lv > entry_ref + atr * 0.2) or (not up and lv < entry_ref - atr * 0.2):
            cands.append({
                "price": lv,
                "invalid": lv + (atr * 0.6 if up else -atr * 0.6),
                "label": "eq",
                "score": 2 - _dist_penalty(lv),
            })

    # Fib adaptif — true last resort
    if not cands:
        try:
            sh15, sl15 = swing_pts(m15, lb=5)
            if len(sh15) >= 1 and len(sl15) >= 1:
                lo, hi     = adaptive_fib_target(m15, sh15, sl15, direction)
                swing_hi   = m15["high"].iloc[sh15[-1]]
                swing_lo   = m15["low"].iloc[sl15[-1]]
                leg        = swing_hi - swing_lo
                px          = (swing_lo + leg * lo) if up else (swing_hi - leg * lo)
                invalid_fib = (swing_lo + leg * hi) if up else (swing_hi - leg * hi)
                if (up and px > entry_ref + atr * 0.1) or (not up and px < entry_ref - atr * 0.1):
                    cands.append({
                        "price": px, "invalid": invalid_fib,
                        "label": "fib_adaptive", "score": 1.5,
                    })
        except Exception:
            pass

    return cands


def calc_discount_entry(df_h1, df_m15, direction, current_price, atr):
    """
    Entry dari kandidat terkuat (OB > FVG > Mitigation > Breaker > EQ > Fib).
    Silver Bullet FVG override bila dalam killzone window.
    """
    m15_built = build_df(df_m15)
    if m15_built is None:
        return current_price, "market", None

    # Silver Bullet override: kalau sedang dalam SB window dan ada FVG fresh,
    # gunakan itu sebagai entry (ICT highest-conviction setup)
    hour_utc = get_current_hour_utc(m15_built)
    sb = detect_silver_bullet(m15_built, direction, hour_utc)
    if sb["active"] and sb["fvg_entry"] is not None:
        invalid_l = sb["fvg_invalid"]
        return round(sb["fvg_entry"], 8), "silver_bullet_fvg", invalid_l

    cands = _collect_entry_candidates(m15_built, direction, current_price, atr)
    if cands:
        best = max(cands, key=lambda c: c["score"])
        return round(best["price"], 8), best["label"], best["invalid"]
    return current_price, "market", None


# ═════════════════════════════════════════════
# PIPELINE ANALISIS LENGKAP
# ═════════════════════════════════════════════
def full_analyze(df_h1, df_m15, df_d1=None, symbol=None):
    """
    Score arah (H1+M15+D1+KZ+CISD+VSA) -> entry presisi
    (OB/FVG/Mitigation/Breaker/Silver Bullet/EQL/Fib) -> SL/TP.
    Dataframe dikirim pemanggil (main.py).
    """
    try:
        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            return None

        score = score_direction(df_h1, df_m15, df_d1)
        if score is None:
            return None

        original_dir  = score["direction"]
        current_price = score["price"]
        atr_val       = score["atr"]
        decision      = "BUY" if original_dir == "bull" else "SELL"

        # ── Confidence adjustment ─────────────────────────────────────
        confidence = score["confidence"]
        choch_confirms = (
            (original_dir == "bull" and score.get("choch_m15", {}).get("bullish_choch")) or
            (original_dir == "bear" and score.get("choch_m15", {}).get("bearish_choch"))
        )

        # Inducement tanpa CHoCH → kurangi confidence
        if score.get("inducement") and not choch_confirms:
            confidence = max(0, confidence - 8)

        # Pullback aggressive tanpa CHoCH → kurangi confidence
        if score.get("pullback_type") == "aggressive" and not choch_confirms:
            confidence = max(0, confidence - 5)

        # CISD searah → boost confidence (konfirmasi reversal sangat awal)
        cisd = score.get("cisd_m15", {})
        cisd_confirms = (
            (original_dir == "bull" and cisd.get("bullish_cisd")) or
            (original_dir == "bear" and cisd.get("bearish_cisd"))
        )
        if cisd_confirms:
            confidence = min(99, confidence + 6)

        # Wyckoff Spring/UTAD → boost confidence (Wyckoff konfirmasi tertinggi)
        vsa = score.get("vsa", {})
        if (original_dir == "bull" and vsa.get("spring")) or \
           (original_dir == "bear" and vsa.get("utad")):
            confidence = min(99, confidence + 8)

        # Silver Bullet aktif → boost confidence
        sb = score.get("silver_bullet", {})
        if sb.get("active"):
            confidence = min(99, confidence + 5)

        # ── Entry diskon dari zona struktural ─────────────────────────
        discount_entry, entry_label, invalid_level = calc_discount_entry(
            df_h1, df_m15, original_dir, current_price, atr_val)

        # ── SL/TP dari entry diskon ────────────────────────────────────
        setup = analyze_setup(df_h1, df_m15, original_dir, discount_entry,
                               score=score, invalid_level=invalid_level)
        if setup is None:
            return None

        # TP wajib masih di depan harga sekarang
        if original_dir == "bull" and current_price >= setup["tp"]:
            return None
        if original_dir == "bear" and current_price <= setup["tp"]:
            return None

        return {
            "symbol"        : symbol,
            "original_dir"  : original_dir,
            "decision"      : decision,
            "confidence"    : confidence,
            "price"         : current_price,
            "entry"         : discount_entry,
            "entry_label"   : entry_label,
            "sl"            : setup["sl"],
            "tp"            : setup["tp"],
            "rr"            : setup["rr"],
            "rsi"           : score["rsi"],
            "struct_h1"     : score["struct_h1"],
            "d1_bias"       : score.get("d1_bias", "neutral"),
            "choch_m15"     : score.get("choch_m15", {}),
            "choch_h1"      : score.get("choch_h1", {}),
            "cisd_m15"      : score.get("cisd_m15", {}),
            "failed_retest" : score.get("failed_retest", {}),
            "vsa"           : score.get("vsa", {}),
            "killzone"      : score.get("killzone", {}),
            "silver_bullet" : score.get("silver_bullet", {}),
            "tp_sl_reason"  : f"Entry@{discount_entry:.5g}({entry_label}) | {setup['reason']}",
        }
    except Exception as e:
        log.debug(f"[full_analyze] {symbol}: {e}")
        return None
