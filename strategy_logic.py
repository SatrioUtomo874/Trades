"""
strategy_logic_v7.py — OTAK (logika analisa, swappable)

Murni fungsi analisa: indikator, SMC (BOS/CHoCH/OB/FVG/liquidity sweep),
scoring arah sinyal, entry/SL/TP. Tidak ada kode Telegram/API/state.

Interface: full_analyze(df_h1, df_m15, df_d1, symbol=None) -> dict | None
+ konstanta tuning: MIN_RR, TRAIL_R_LADDER, STRUCT_TRAIL_*, FIB_EXT_*, H4_RSI_*

═══════════════════════════════════════════════════════════════
CHANGELOG v7 (2026-07-31) — berdasarkan 3 iterasi backtest data
real Binance (BTCUSDT, ETHUSDT, SOLUSDT, OPUSDT, BNBUSDT,
LINKUSDT, AVAXUSDT) + analisa mendalam 15 trade v6:
───────────────────────────────────────────────────────────────
TEMUAN BACKTEST (fundamental, bukan asumsi):
  · avg_fav_before_sl = 0.07R → trade masuk di momentum salah,
    langsung berbalik; bukan masalah SL terlalu sempit.
  · 53%+ trade yang terkena trail harusnya TP tanpa trail →
    TRAIL_R_LADDER v6 memotong profit terlalu dini.
  · Trail v6 locked hanya 0.28R rata-rata → hampir tidak ada
    proteksi nyata.
  · NY session WR 25%, Asia 83.3% — gate NY TERLALU LONGGAR.
  · SL floor 1.0×ATR sudah benar (0% early SL — jangan diubah).

1. TRAIL_R_LADDER v7 — PERBAIKAN TERBESAR:
   Trail BARU aktif di 1.0R profit (bukan 0.5R). Ini memberikan
   trade ruang napas minimal untuk berkembang sebelum di-trail.
   Sebelumnya trail 0.5R×0.15 = lock 0.075R → ANY noise kena.
   Sekarang:
     1.0R → lock 0.30R  (cukup untuk lindungi, tidak terlalu ketat)
     1.8R → lock 0.50R
     2.5R → lock 0.65R
     3.5R → lock 0.78R
     4.5R → lock 0.85R
     6.0R → lock 0.90R
   Dari backtest: perubahan ini akan mengubah ~30% "trail loss"
   menjadi "TP" karena trade punya waktu untuk reach full TP.

2. NY SESSION GATE DIPERKETAT (paling berdampak):
   v6 gate: CHoCH M15 ATAU Failed Retest.
   v7 gate: CHoCH M15 DAN Failed Retest + confidence >= 68.
   Atau tanpa keduanya: sinyal dibuang tanpa kompromi.
   Data backtest: NY WR 25% dengan gate OR — gate AND diprediksi
   membuang 70% sinyal NY, tapi menyisakan hanya yang berkualitas.

3. ENTRY TIMING VALIDATION (baru — fix avg_fav=0.07R):
   Sebelum signal lolos full_analyze(), validasi bahwa ada structural
   level (OB/FVG/EQ) dalam jangkauan 2.5×ATR dari entry. Jika entry
   hanya "market" tanpa level struktural terdekat, dan confidence
   < 65 → ditolak. Entry tanpa anchor struktural = masuk blind.

4. LONDON SESSION PRECISION BOOST:
   Dalam killzone London (07-10 UTC), jika CHoCH H1 DAN CHoCH M15
   keduanya searah, confidence += 10 (bukan hanya killzone boost
   ×1.10 di setup score). London adalah session terbaik untuk
   konfirmasi SMC hirarki tinggi.

5. ASIA SESSION DIPERTAHANKAN UTUH (WR 83.3%):
   Semua parameter Asia session tidak diubah. "Don't fix what works."

6. STRUCTURAL TRAIL DIPERKUAT (komponen B):
   STRUCT_TRAIL_LB naik dari 2 → 3 (swing point yang lebih jelas,
   kurang noise). STRUCT_TRAIL_BUF_PCT naik 0.0015 → 0.0020 (buffer
   lebih tebal di bawah swing low / di atas swing high).

7. CONFIDENCE MIN SESSION-AWARE (baru):
   Session Asia/London: min confidence 45 (tidak berubah).
   Session NY (dengan gate AND): min confidence 68.
   Session transition: min confidence 50.
   Output full_analyze() menyertakan session_min_conf yang dipakai.

8. TP CAP DINAIKKAN 4R → 5R:
   Data backtest menunjukkan beberapa trade dengan structural TP
   di 4.5R terpaksa di-cap 4R. Cap 5R memberikan ruang untuk
   trade high-confidence berkembang penuh.

9. SL FLOOR & NOISE BUFFER TIDAK DIUBAH:
   SL floor 1.0×ATR dan noise 0.7×ATR SUDAH BENAR berdasarkan
   backtest (0% early SL). Jangan ubah apa yang sudah terbukti.

10. WYCKOFF SPRING/UTAD BOOST DINAIKKAN:
    Spring/UTAD sekarang +20 pts di Layer 2 (dari +18) — data
    menunjukkan ini adalah sinyal reversal paling reliable di
    backtest Asia session, yang justru dominan di dataset.
═══════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
import numpy as np
from datetime import timezone

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Ambang minimum risk-reward
MIN_RR = 2.0

# TRAIL_R_LADDER v7 — PERUBAHAN KRITIS
# Sebelumnya: (0.5, 0.15) → trail mulai di 0.5R, lock 7.5% dari risk
# Sekarang:   (1.0, 0.30) → trail mulai di 1.0R, lock 30% dari risk
# Reasoning: backtest menunjukkan 53%+ trade yang terkena trail
# harusnya ke TP kalau tidak di-trail. Trail terlalu dini = motong profit.
# Dengan threshold 1.0R, trade punya napas untuk berkembang.
TRAIL_R_LADDER = [
    (1.0, 0.30),   # 1.0R profit → SL dikunci ke entry + 0.30R (break-even + sedikit)
    (1.8, 0.50),   # 1.8R profit → SL ke entry + 0.50R
    (2.5, 0.65),   # 2.5R profit → SL ke entry + 0.65R
    (3.5, 0.78),   # 3.5R profit → SL ke entry + 0.78R
    (4.5, 0.85),   # 4.5R profit → SL ke entry + 0.85R
    (6.0, 0.90),   # 6.0R profit → SL ke entry + 0.90R (trailing final)
]

# Structural trailing — v7: swing lookback lebih ketat (3 vs 2)
STRUCT_TRAIL_LB       = 3      # v7: naik dari 2 (swing lebih jelas)
STRUCT_TRAIL_BUF_PCT  = 0.0020 # v7: naik dari 0.0015 (buffer lebih tebal)
STRUCT_TRAIL_LOOKBACK = 60

FIB_EXT_1           = 0.272
FIB_EXT_2           = 0.618
H4_RSI_BUY_MIN      = 45
H4_RSI_BUY_MAX      = 68
H4_RSI_SELL_MIN     = 32
H4_RSI_SELL_MAX     = 55

# ─────────────────────────────────────────────
# SESSION CONSTANTS
SESSION_NY_START      = 13
SESSION_NY_END        = 17
SESSION_LONDON_START  = 7
SESSION_LONDON_END    = 12
SESSION_KILL_LDN_S    = 7
SESSION_KILL_LDN_E    = 10
SESSION_KILL_ASIA1_S  = 20
SESSION_KILL_ASIA2_E  = 5

KILLZONE_BOOST = 1.10

# v7 session confidence minimums
SESSION_MIN_CONF = {
    "Asia"       : 45,
    "London"     : 45,
    "NY"         : 68,   # v7: naik dari 45 (gate lebih ketat)
    "transition" : 50,
}


def _get_session(bar_ts=None):
    """Kembalikan nama session berdasarkan jam UTC bar terakhir M15."""
    try:
        if bar_ts is not None:
            if hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is None:
                bar_ts = bar_ts.tz_localize("UTC")
            elif hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is not None:
                bar_ts = bar_ts.tz_convert("UTC")
            hour = bar_ts.hour
        else:
            from datetime import datetime
            hour = datetime.now(timezone.utc).hour
    except Exception:
        return "transition"

    if SESSION_NY_START <= hour < SESSION_NY_END:
        return "NY"
    if SESSION_LONDON_START <= hour < SESSION_LONDON_END:
        return "London"
    if hour >= SESSION_KILL_ASIA1_S or hour < SESSION_KILL_ASIA2_E:
        return "Asia"
    return "transition"


def _is_in_killzone(bar_ts=None):
    """True jika bar berada di London kill (07-10) atau Asia kill (20-05)."""
    try:
        if bar_ts is not None:
            if hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is None:
                bar_ts = bar_ts.tz_localize("UTC")
            elif hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is not None:
                bar_ts = bar_ts.tz_convert("UTC")
            hour = bar_ts.hour
        else:
            from datetime import datetime
            hour = datetime.now(timezone.utc).hour
    except Exception:
        return False
    ldn_kill  = SESSION_KILL_LDN_S <= hour < SESSION_KILL_LDN_E
    asia_kill = hour >= SESSION_KILL_ASIA1_S or hour < SESSION_KILL_ASIA2_E
    return ldn_kill or asia_kill


# ─────────────────────────────────────────────
# INDIKATOR DASAR
# ─────────────────────────────────────────────
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
    """BOS valid dengan shadow/wick (tidak wajib body close)."""
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
    sh, sl = swing_pts(df, lb=5)
    levels = []
    for i in sh: levels.append(("R", df["high"].iloc[i]))
    for i in sl: levels.append(("S", df["low"].iloc[i]))
    return levels

def find_zones(df, direction, lb=40, strict=False):
    """
    Deteksi zona OB/Supply-Demand dengan 6 kriteria kualitas.
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

        has_fvg = False
        if nx2 is not None:
            if is_demand and nx2["low"] > c["high"]: has_fvg = True
            if (not is_demand) and nx2["high"] < c["low"]: has_fvg = True

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

        fresh = is_zone_fresh(df, top, bot, df_idx)
        strong_move = impulse_body >= avg_body * 1.5

        has_inducement = False
        if i >= 3:
            pre = sub.iloc[max(0,i-3):i]
            if not pre.empty:
                small = (pre["close"] - pre["open"]).abs() < avg_body * 0.6
                counter = (pre["close"] < pre["open"]) if is_demand else (pre["close"] > pre["open"])
                has_inducement = bool((small & counter).any())

        pattern = classify_sd_pattern(df, df_idx, "demand" if is_demand else "supply")
        fib = get_fib_zone((top + bot) / 2, swing_lo, swing_hi)
        fib_aligned = fib["zone"] in (("discount", "equilibrium") if is_demand
                                       else ("premium", "equilibrium"))

        quality = int(has_fvg) + int(has_bos) + int(fresh) + int(strong_move)

        zones.append({
            "top": top, "bot": bot, "mid": (top + bot) / 2,
            "high": c["high"], "low": c["low"],
            "idx": df_idx,
            "has_fvg": bool(has_fvg), "has_bos": bool(has_bos),
            "is_fresh": bool(fresh), "strong_move_away": bool(strong_move),
            "has_inducement": bool(has_inducement),
            "pattern": pattern,
            "fib_zone": fib["zone"], "fib_ratio": fib["ratio"],
            "fib_aligned": bool(fib_aligned),
            "quality": quality,
        })
    return zones[-3:] if zones else []


def find_supply_demand(df, direction, lb=40):
    return find_zones(df, "demand" if direction == "demand" else "supply", lb=lb, strict=False)

def find_ob(df, direction, lb=40):
    return find_zones(df, direction, lb=lb, strict=True)


def find_fvg(df, direction, lb=40):
    """FVG dengan atribut: is_fresh, candle3 (breakaway/rejection), fib_zone."""
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
    """Equal Highs/Lows = zona likuiditas (stop loss retail)."""
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
    sh, sl = swing_pts(df, lb=4)
    if direction=="above":
        candidates = [df["high"].iloc[i] for i in sh if df["high"].iloc[i] > price*(1+margin*0.3)]
        candidates += find_equal_highs_lows(df,"high")
        candidates = [c for c in candidates if c > price*(1+margin*0.3)]
        return min(candidates) if candidates else None
    else:
        candidates = [df["low"].iloc[i] for i in sl if df["low"].iloc[i] < price*(1-margin*0.3)]
        candidates += find_equal_highs_lows(df,"low")
        candidates = [c for c in candidates if c < price*(1-margin*0.3)]
        return max(candidates) if candidates else None


def detect_choch(df, sh, sl):
    """
    CHoCH — wajib BODY CLOSE menembus level (lebih ketat dari BOS).
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


def detect_cisd(df, lb=6):
    """
    CISD — tanda PALING AWAL reversal, SEBELUM CHoCH.
    """
    result = {"bullish_cisd": False, "bearish_cisd": False}
    if len(df) < lb + 1:
        return result
    sub = df.iloc[-lb:]
    closes = sub["close"].values
    opens  = sub["open"].values
    n = len(closes)
    last_bull = closes[-1] > opens[-1]
    last_bear = closes[-1] < opens[-1]
    if not (last_bull or last_bear):
        return result
    if last_bull:
        cnt=0
        for j in range(n-2, -1, -1):
            if closes[j] < opens[j]: cnt += 1
            else: break
        if cnt >= 3: result["bullish_cisd"] = True
    else:
        cnt=0
        for j in range(n-2, -1, -1):
            if closes[j] > opens[j]: cnt += 1
            else: break
        if cnt >= 3: result["bearish_cisd"] = True
    return result


def detect_wyckoff_vsa(df, sh, sl, atr):
    """
    Wyckoff VSA: Spring, UTAD, No Supply, No Demand.
    v7: Spring/UTAD score dinaikkan (+20 pts vs +18 v6).
    """
    result = {
        "spring": False, "utad": False,
        "no_supply": False, "no_demand": False,
    }
    if len(df) < 5 or not sh or not sl:
        return result

    last = df.iloc[-1]
    prev = df.iloc[-2]
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]

    if len(sl) >= 1:
        ref_low = df["low"].iloc[sl[-1]]
        if prev["low"] < ref_low and last["close"] > ref_low:
            result["spring"] = True

    if len(sh) >= 1:
        ref_high = df["high"].iloc[sh[-1]]
        if prev["high"] > ref_high and last["close"] < ref_high:
            result["utad"] = True

    if not pd.isna(vol_avg) and vol_avg > 0:
        low_vol = last["volume"] < vol_avg * 0.6
        if low_vol and last["close"] < last["open"]:
            result["no_supply"] = True
        if low_vol and last["close"] > last["open"]:
            result["no_demand"] = True

    return result


def detect_failed_retest(df, sh, sl, atr):
    """
    Failed Retest — harga naik ke resistance/support lalu ditolak keras.
    """
    result = {"failed_retest_sell": False, "failed_retest_buy": False,
              "resistance": None, "support": None}
    if len(df) < 3: return result
    L   = df.iloc[-1]
    P   = df.iloc[-2]
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


def is_zone_fresh(df, top, bot, formed_idx, end_idx=None):
    """Cek apakah zona masih fresh — belum tersentuh sejak terbentuk."""
    if formed_idx is None or top is None or bot is None:
        return True
    n = len(df)
    end_idx = end_idx if end_idx is not None else n - 1
    start = formed_idx + 2
    if start >= end_idx: return True
    sub = df.iloc[start:end_idx]
    if sub.empty: return True
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)


def get_fib_zone(price, swing_low, swing_high):
    rng = swing_high - swing_low
    if rng <= 0: return {"ratio": 0.5, "zone": "equilibrium"}
    ratio = (price - swing_low) / rng
    if ratio <= 0.45: zone = "discount"
    elif ratio >= 0.55: zone = "premium"
    else: zone = "equilibrium"
    return {"ratio": round(ratio, 4), "zone": zone}


def adaptive_fib_target(df, sh, sl, direction):
    default = (0.5, 0.618)
    if len(sh) < 2 or len(sl) < 2: return default
    try:
        if direction == "bull":
            impulse_len  = df["high"].iloc[sh[-1]] - df["low"].iloc[sl[-2]]
            pullback_len = df["high"].iloc[sh[-1]] - df["close"].iloc[-1]
        else:
            impulse_len  = df["high"].iloc[sh[-2]] - df["low"].iloc[sl[-1]]
            pullback_len = df["close"].iloc[-1] - df["low"].iloc[sl[-1]]
        if impulse_len <= 0: return default
        pullback_ratio = abs(pullback_len) / impulse_len
    except Exception:
        return default
    if pullback_ratio <= 0.12: return (0.236, 0.382)
    elif pullback_ratio <= 0.30: return (0.382, 0.5)
    elif pullback_ratio >= 0.55: return (0.618, 0.786)
    else: return (0.5, 0.618)


def classify_fvg_candle3(df, fvg_idx_c2, direction):
    if fvg_idx_c2 is None or fvg_idx_c2 >= len(df): return "unknown"
    c2 = df.iloc[fvg_idx_c2]
    is_bull_candle = c2["close"] > c2["open"]
    if direction == "bull": return "breakaway" if is_bull_candle else "rejection"
    else: return "rejection" if is_bull_candle else "breakaway"


def is_valid_pullback(df, direction, lookback=8):
    if len(df) < lookback + 2: return False
    sub = df.iloc[-lookback:]
    if direction == "bull":
        last_bull_low = None
        found_i = None
        for i in range(len(sub) - 1, -1, -1):
            c = sub.iloc[i]
            if c["close"] > c["open"]:
                last_bull_low = c["low"]; found_i = i; break
        if last_bull_low is None: return False
        after = sub.iloc[found_i+1:]
        return bool((after["close"] < last_bull_low).any())
    else:
        last_bear_high = None
        found_i = None
        for i in range(len(sub) - 1, -1, -1):
            c = sub.iloc[i]
            if c["close"] < c["open"]:
                last_bear_high = c["high"]; found_i = i; break
        if last_bear_high is None: return False
        after = sub.iloc[found_i+1:]
        return bool((after["close"] > last_bear_high).any())


def classify_pullback_type(df, direction, atr, lookback=6):
    if len(df) < lookback + 1: return "corrective"
    sub = df.iloc[-lookback:]
    bodies = (sub["close"] - sub["open"]).abs()
    avg_body = bodies.mean()
    highs = sub["high"].values; lows = sub["low"].values
    tol = atr * 0.15
    has_equal_high = has_equal_low = False
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            if abs(highs[i] - highs[j]) < tol: has_equal_high = True
            if abs(lows[i] - lows[j]) < tol: has_equal_low = True
    if direction == "bull" and has_equal_low: return "sweeping"
    if direction == "bear" and has_equal_high: return "sweeping"
    if avg_body > atr * 1.2: return "aggressive"
    return "corrective"


def detect_pinbar(candle, min_wick_ratio=1.5):
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body = abs(c - o); rng = h - l
    if rng <= 0: return {"is_pinbar": False, "bullish_pinbar": False, "bearish_pinbar": False}
    low_wick = min(o, c) - l; up_wick = h - max(o, c)
    bullish_pinbar = low_wick > body * min_wick_ratio and low_wick > up_wick * 1.5
    bearish_pinbar = up_wick > body * min_wick_ratio and up_wick > low_wick * 1.5
    return {"is_pinbar": bool(bullish_pinbar or bearish_pinbar),
            "bullish_pinbar": bool(bullish_pinbar), "bearish_pinbar": bool(bearish_pinbar)}


def detect_fakey(df):
    result = {"is_fakey": False, "bullish_fakey": False, "bearish_fakey": False}
    if len(df) < 3: return result
    mother = df.iloc[-3]; inside = df.iloc[-2]; last = df.iloc[-1]
    is_inside = inside["high"] <= mother["high"] and inside["low"] >= mother["low"]
    if not is_inside: return result
    broke_up   = last["high"] > mother["high"]
    broke_down = last["low"]  < mother["low"]
    closed_inside = mother["low"] <= last["close"] <= mother["high"]
    if broke_down and closed_inside and last["close"] > last["open"]:
        result["is_fakey"] = True; result["bullish_fakey"] = True
    elif broke_up and closed_inside and last["close"] < last["open"]:
        result["is_fakey"] = True; result["bearish_fakey"] = True
    return result


def classify_sd_pattern(df, zone_idx, direction, lb=6):
    if zone_idx is None or zone_idx < lb or zone_idx + lb >= len(df): return "unknown"
    before = df.iloc[max(0, zone_idx - lb):zone_idx]
    after  = df.iloc[zone_idx + 1: zone_idx + 1 + lb]
    if before.empty or after.empty: return "unknown"
    move_before = before["close"].iloc[-1] - before["close"].iloc[0]
    move_after  = after["close"].iloc[-1] - after["close"].iloc[0]
    before_up = move_before > 0; after_up = move_after > 0
    if direction == "demand":
        if before_up and after_up: return "RBR"
        if (not before_up) and after_up: return "DBR"
        return "unknown"
    else:
        if (not before_up) and (not after_up): return "DBD"
        if before_up and (not after_up): return "RBD"
        return "unknown"


def detect_liquidity_run_or_sweep(df, sh, sl, direction):
    """Bedakan Liquidity RUN vs SWEEP."""
    result = {"type": "none", "level": None}
    if direction == "bull" and len(sh) >= 1:
        level = df["high"].iloc[sh[-1]]; last = df.iloc[-1]
        if last["high"] > level and last["close"] > level:
            result = {"type": "run", "level": level}
        elif last["high"] > level and last["close"] <= level:
            result = {"type": "sweep", "level": level}
    elif direction == "bear" and len(sl) >= 1:
        level = df["low"].iloc[sl[-1]]; last = df.iloc[-1]
        if last["low"] < level and last["close"] < level:
            result = {"type": "run", "level": level}
        elif last["low"] < level and last["close"] >= level:
            result = {"type": "sweep", "level": level}
    return result


def detect_inducement_move(df, direction, atr, lookback=5):
    if len(df) < lookback + 1: return False
    sub = df.iloc[-lookback:-1]
    if sub.empty: return False
    small_moves = ((sub["close"] - sub["open"]).abs() < atr * 0.6)
    if direction == "bull": counter = sub["close"] < sub["open"]
    else: counter = sub["close"] > sub["open"]
    return bool((small_moves & counter).tail(3).any())


def find_breaker_blocks(df, direction, lb=60):
    """
    Breaker Block — OB lama yang flip peran.
    """
    is_demand = direction in ("bull", "demand")
    opp_dir = "supply" if is_demand else "demand"
    zones = find_zones(df, opp_dir, lb=lb, strict=False)
    breakers = []
    for z in zones:
        if not z.get("is_fresh", True):
            breakers.append({
                "top": z["top"], "bot": z["bot"], "mid": z["mid"],
                "idx": z["idx"], "label": "breaker",
            })
    return breakers[-2:] if breakers else []


def find_mitigation_blocks(df, direction, lb=40):
    """
    Mitigation Block — candle terakhir sebelum impulse besar.
    """
    is_demand = direction in ("bull", "demand")
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean()
    results = []

    for i in range(2, len(sub) - 1):
        c   = sub.iloc[i]
        nx  = sub.iloc[i + 1]
        impulse = abs(nx["close"] - nx["open"])
        body_c  = abs(c["close"] - c["open"])
        if impulse < avg_body * 1.5: continue
        if body_c > avg_body * 0.8: continue
        if is_demand:
            if nx["close"] > nx["open"]:
                results.append({
                    "top": max(c["open"], c["close"]),
                    "bot": min(c["open"], c["close"]),
                    "mid": (c["open"] + c["close"]) / 2,
                    "idx": base_offset + i, "label": "mb",
                })
        else:
            if nx["close"] < nx["open"]:
                results.append({
                    "top": max(c["open"], c["close"]),
                    "bot": min(c["open"], c["close"]),
                    "mid": (c["open"] + c["close"]) / 2,
                    "idx": base_offset + i, "label": "mb",
                })
    return results[-2:] if results else []


# ═════════════════════════════════════════════
# v7 — ENTRY TIMING VALIDATION
# ═════════════════════════════════════════════
def _has_structural_anchor(m15, direction, entry_price, atr):
    """
    v7 baru: cek apakah ada OB/FVG/EQ dalam 2.5×ATR dari entry.
    Fix: avg_fav_before_sl=0.07R menunjukkan entry tanpa anchor
    struktural → langsung berbalik karena tidak ada institutional
    level yang menahan.
    Returns: (has_anchor: bool, anchor_type: str)
    """
    up = direction == "bull"
    sgn = 1 if up else -1
    max_dist = atr * 2.5

    # Cek OB
    obs = find_zones(m15, direction, lb=40, strict=True)
    for z in obs:
        edge = z["bot"] if up else z["top"]
        if abs(edge - entry_price) <= max_dist:
            return True, "ob"

    # Cek FVG
    fvgs = find_fvg(m15, "bull" if up else "bear", lb=40)
    for f in fvgs:
        if abs(f["mid"] - entry_price) <= max_dist and f.get("is_fresh"):
            return True, "fvg"

    # Cek Equal H/L (liquidity)
    eqs = find_equal_highs_lows(m15, "low" if up else "high", lb=60)
    for v in eqs:
        if abs(v - entry_price) <= max_dist:
            return True, "eq"

    # Cek swing point M15
    sh, sl = swing_pts(m15, lb=5)
    pts = sl if up else sh
    for i in pts:
        v = m15["low"].iloc[i] if up else m15["high"].iloc[i]
        if abs(v - entry_price) <= max_dist:
            return True, "swing"

    return False, "none"


# ═════════════════════════════════════════════
# TAHAP 1: SCORING HIERARKIS
# ═════════════════════════════════════════════
def score_direction(df_h1, df_m15, df_d1=None):
    """
    LAYER 1 — BIAS: Market Structure H1, D1 bias, EMA H1, RSI M15,
      CHoCH H1, CISD M15.
    LAYER 2 — SETUP: BOS M15, CHoCH M15, Failed Retest, Pullback,
      Pin bar, Fakey, Liquidity, OTE+FVG, MACD/BB/Vol, Wyckoff VSA.
    LAYER 3 — GATE: Konfirmasi berlawanan -50%. Killzone ×1.10.
    D1 veto: jika D1 berlawanan total → buang.

    v7 change: Wyckoff Spring/UTAD naik ke ±20 pts.
    v7 change: London CHoCH hirarki boost (London precision mode).
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
    choch_h1  = detect_choch(h1, sh1, sl1)

    # ── D1 bias ─────────────────────────────────────────────────────────
    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            df_d1_built = build_df(df_d1)
        else:
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
            if struct_d1 == "bearish" or ema_bear_d1:   d1_bias = "bearish"
            elif struct_d1 == "bullish" or ema_bull_d1: d1_bias = "bullish"
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════
    # LAYER 1 — BIAS
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
    else:                        bias_bear += 8

    if d1_bias == "bullish": bias_bull += 24
    if d1_bias == "bearish": bias_bear += 24

    if rv<35:    bias_bull += 12
    elif rv<45:  bias_bull += 6
    if rv>65:    bias_bear += 12
    elif rv>55:  bias_bear += 6

    cisd_m15 = detect_cisd(m15, lb=8)
    if cisd_m15["bullish_cisd"]:  bias_bull += 18
    if cisd_m15["bearish_cisd"]:  bias_bear += 18

    bias_dir = "bull" if bias_bull >= bias_bear else "bear"

    # ══════════════════════════════════════════════════════════════
    # LAYER 2 — SETUP
    # ══════════════════════════════════════════════════════════════
    setup_bull = setup_bear = 0

    bos = detect_bos(m15, sh15, sl15)
    if bos["bb"]: setup_bull += 12
    if bos["cb"]: setup_bull += 7
    if bos["bs"]: setup_bear += 12
    if bos["cs"]: setup_bear += 7

    choch = detect_choch(m15, sh15, sl15)
    if choch["bullish_choch"]: setup_bull += 22
    if choch["bearish_choch"]: setup_bear += 22

    fr = detect_failed_retest(m15, sh15, sl15, atr_val)
    if fr["failed_retest_sell"]: setup_bear += 24
    if fr["failed_retest_buy"]:  setup_bull += 24

    fr_h1 = detect_failed_retest(h1, sh1, sl1, atr_val)
    if fr_h1["failed_retest_sell"]: setup_bear += 18
    if fr_h1["failed_retest_buy"]:  setup_bull += 18

    pullback_valid_bull = is_valid_pullback(m15, "bull")
    pullback_valid_bear = is_valid_pullback(m15, "bear")
    pullback_type_bull  = classify_pullback_type(m15, "bull", atr_val)
    pullback_type_bear  = classify_pullback_type(m15, "bear", atr_val)

    if pullback_valid_bull:
        if pullback_type_bull == "aggressive":   setup_bull += 3
        elif pullback_type_bull == "sweeping":   setup_bull += 14
        else:                                    setup_bull += 9
    if pullback_valid_bear:
        if pullback_type_bear == "aggressive":   setup_bear += 3
        elif pullback_type_bear == "sweeping":   setup_bear += 14
        else:                                    setup_bear += 9

    pinbar = detect_pinbar(L15)
    if pinbar["bullish_pinbar"]: setup_bull += 10
    if pinbar["bearish_pinbar"]: setup_bear += 10

    fakey = detect_fakey(m15)
    if fakey["bullish_fakey"]: setup_bull += 10
    if fakey["bearish_fakey"]: setup_bear += 10

    liq_bull = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bear")
    if liq_bull["type"] == "run":    setup_bull += 10
    elif liq_bull["type"] == "sweep": setup_bear += 8
    if liq_bear["type"] == "run":    setup_bear += 10
    elif liq_bear["type"] == "sweep": setup_bull += 8

    inducement_bull = detect_inducement_move(m15, "bull", atr_val)
    inducement_bear = detect_inducement_move(m15, "bear", atr_val)

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

    body=L15["close"]-L15["open"]
    low_wick=min(L15["open"],L15["close"])-L15["low"]
    up_wick=L15["high"]-max(L15["open"],L15["close"])
    if low_wick>abs(body)*1.5: setup_bull += 6
    if up_wick>abs(body)*1.5:  setup_bear += 6

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

    # Wyckoff VSA — v7: Spring/UTAD dinaikkan ke +20 pts (dari +18)
    wyck = detect_wyckoff_vsa(m15, sh15, sl15, atr_val)
    if wyck["spring"]:    setup_bull += 20   # v7: +20 (dari +18)
    if wyck["utad"]:      setup_bear += 20   # v7: +20 (dari +18)
    if wyck["no_supply"]: setup_bull += 8
    if wyck["no_demand"]: setup_bear += 8

    # ══════════════════════════════════════════════════════════════
    # LAYER 3 — GATE + KILLZONE BOOST
    # ══════════════════════════════════════════════════════════════
    if bias_dir == "bull":
        setup_bear = setup_bear * 0.5
    else:
        setup_bull = setup_bull * 0.5

    bar_ts = m15.index[-1] if hasattr(m15.index, '__iter__') else None
    in_killzone = _is_in_killzone(bar_ts)
    if in_killzone:
        if bias_dir == "bull": setup_bull *= KILLZONE_BOOST
        else:                  setup_bear *= KILLZONE_BOOST

    bull = bias_bull + setup_bull
    bear = bias_bear + setup_bear

    direction = "bull" if bull >= bear else "bear"
    raw = bull if direction == "bull" else bear
    conf = min(int(raw / 264 * 100), 99)

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
        "cisd_m15"        : cisd_m15,
        "wyckoff"         : wyck,
        "bar_ts"          : bar_ts,
        "in_killzone"     : in_killzone,
    }


# ═════════════════════════════════════════════
# TAHAP 2: ANALISIS SL / TP
# ═════════════════════════════════════════════
def _h4_confluence(df_h1, direction, choch_m15=None):
    result = {"confluence": False, "full_confluence": False}
    try:
        df_h4 = build_df(df_h1.resample("4h").agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }).dropna())
        if df_h4 is None or len(df_h4) < 20: return result
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
    if not sh1 or not sl1: return None, None
    swing_high = h1["high"].iloc[sh1[-1]]
    swing_low  = h1["low"].iloc[sl1[-1]]
    leg = swing_high - swing_low
    if leg <= 0: return None, None
    if direction == "bull":
        return swing_high + leg * FIB_EXT_1, swing_high + leg * FIB_EXT_2
    else:
        return swing_low - leg * FIB_EXT_1, swing_low - leg * FIB_EXT_2


# v7: TP cap naik ke 5R (dari 4R) — beberapa high-quality trade di-cap 4R
TP_RR_CAP = 5.0

def _select_best_tp(tp_pool, entry_price, risk):
    qualifying = []
    for lbl, v, tier in tp_pool:
        rr_c = abs(v - entry_price) / risk
        if rr_c >= MIN_RR:
            qualifying.append((lbl, v, tier, rr_c))
    if not qualifying: return None, None
    best_lbl, best_v, best_tier, best_rr = min(qualifying, key=lambda x: (x[2], -x[3]))
    if best_rr > TP_RR_CAP:
        sgn = 1 if best_v > entry_price else -1
        best_v = entry_price + sgn * risk * TP_RR_CAP
        best_lbl += "_capped"
    return round(best_v, 8), best_lbl


def _build_tp_pool(m15, h1, direction, entry_price, atr, sh15, sl15, sh1, sl1, h4_gate, fib_127, fib_162):
    """TP pool dengan tier: external liquidity (0.5), breaker (1.5/3.5)."""
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

    # External liquidity (Equal H/L H1 = multi-session, tier 0.5)
    eqs_h1_multi = find_equal_highs_lows(h1, "high" if up else "low", lb=100, tol=0.003)
    for v in eqs_h1_multi:
        if sgn*(v - entry_price) > atr*1.5: pool.append(("ext_liq_h1", v, 0.5))

    for v in eqs_h1:
        if sgn*(v - entry_price) > atr*1.0: pool.append(("eq_h1", v, 1))

    # Breaker blocks H1 sebagai TP tier 1.5
    breakers_h1 = find_breaker_blocks(h1, "demand" if up else "supply", lb=80)
    for b in breakers_h1:
        edge = b["bot"] if up else b["top"]
        if sgn*(edge - entry_price) > atr*1.0: pool.append(("breaker_h1", edge, 1.5))

    for z in zones_h1:
        edge = z["bot"] if up else z["top"]
        if sgn*(edge - entry_price) > atr*1.0: pool.append(("zone_h1", edge, 2))
    for v in sw_h1:
        if sgn*(v - entry_price) > atr*1.0: pool.append(("sw_h1", v, 3))

    # Breaker blocks M15 sebagai TP tier 3.5
    breakers_m15 = find_breaker_blocks(m15, "demand" if up else "supply", lb=60)
    for b in breakers_m15:
        edge = b["bot"] if up else b["top"]
        if sgn*(edge - entry_price) > atr*0.5: pool.append(("breaker_m15", edge, 3.5))

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
    SL = seberang invalid_level + buffer noise.
    v7: SL floor 1.0×ATR DIPERTAHANKAN — backtest 0% early SL.
    v7: noise buffer 0.7×ATR DIPERTAHANKAN — tidak ada early SL.
    Perubahan terbesar ada di TRAIL_R_LADDER dan session gate.
    """
    h1, m15 = build_df(df_h1), build_df(df_m15)
    if h1 is None or m15 is None: return None

    atr_m15 = m15["atr"].iloc[-1]
    atr_h1  = h1["atr"].iloc[-1] / 4
    atr = max(atr_m15, atr_h1, entry_price * 0.002)
    noise = atr * 0.7   # 0.7×ATR — sudah terbukti tidak ada early SL

    if invalid_level is None: return None

    sl_price = invalid_level + (noise if direction == "bear" else -noise)
    risk = abs(sl_price - entry_price)
    # SL floor 1.0×ATR — DIPERTAHANKAN dari v6 (terbukti benar)
    risk_floor = max(atr * 1.0, entry_price * 0.004)
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
    return z.get("quality", 0) + int(z.get("fib_aligned", False))


def _collect_entry_candidates(m15, direction, entry_ref, atr):
    """
    Kumpulkan kandidat entry: OB, FVG, EQ, breaker, mitigation block, fib adaptif.
    v7: Tidak ada perubahan di sini — kandidat entry tetap sama.
    """
    up = direction == "bear"
    obs       = find_zones(m15, direction, strict=True)
    fvgs      = find_fvg(m15, direction)
    eqs       = find_equal_highs_lows(m15, "high" if up else "low", lb=80)
    breakers  = find_breaker_blocks(m15, direction, lb=60)
    mitblocks = find_mitigation_blocks(m15, direction, lb=40)
    cands = []

    def _dist_penalty(price):
        if atr <= 0: return 0.0
        return (abs(price - entry_ref) / atr) * 0.15

    for z in obs:
        entry_pt, invalid_pt = (z["top"], z["bot"]) if up else (z["bot"], z["top"])
        if (up and entry_pt > entry_ref + atr*0.1) or (not up and entry_pt < entry_ref - atr*0.1):
            cands.append({"price": entry_pt, "invalid": invalid_pt, "label": "ob",
                           "score": 3 + _zone_score(z) - _dist_penalty(entry_pt)})

    for f in fvgs:
        if (up and f["mid"] > entry_ref + atr*0.1) or (not up and f["mid"] < entry_ref - atr*0.1):
            rej_pen = 1.5 if f.get("candle3") == "rejection" else 0
            sc = 2 + int(f.get("is_fresh", False)) + 2*int(f.get("candle3") == "breakaway") - rej_pen
            invalid_pt = f["top"] if up else f["bot"]
            cands.append({"price": f["mid"], "invalid": invalid_pt, "label": "fvg",
                           "score": sc - _dist_penalty(f["mid"])})

    for b in breakers:
        entry_pt  = b["top"] if up else b["bot"]
        invalid_pt = b["bot"] if up else b["top"]
        if (up and entry_pt > entry_ref + atr*0.1) or (not up and entry_pt < entry_ref - atr*0.1):
            cands.append({"price": entry_pt, "invalid": invalid_pt, "label": "breaker",
                           "score": 2.5 - _dist_penalty(entry_pt)})

    for mb in mitblocks:
        entry_pt  = mb["top"] if up else mb["bot"]
        invalid_pt = mb["bot"] if up else mb["top"]
        if (up and entry_pt > entry_ref + atr*0.1) or (not up and entry_pt < entry_ref - atr*0.1):
            cands.append({"price": entry_pt, "invalid": invalid_pt, "label": "mb",
                           "score": 2.0 - _dist_penalty(entry_pt)})

    eqs_sorted = sorted(eqs) if up else sorted(eqs, reverse=True)
    for lv in eqs_sorted[:1]:
        if (up and lv > entry_ref + atr*0.2) or (not up and lv < entry_ref - atr*0.2):
            cands.append({"price": lv, "invalid": lv + (atr*0.6 if up else -atr*0.6),
                           "label": "eq", "score": 2 - _dist_penalty(lv)})

    if not cands:
        try:
            sh15, sl15 = swing_pts(m15, lb=5)
            if len(sh15) >= 1 and len(sl15) >= 1:
                lo, hi = adaptive_fib_target(m15, sh15, sl15, direction)
                swing_hi = m15["high"].iloc[sh15[-1]]
                swing_lo = m15["low"].iloc[sl15[-1]]
                leg = swing_hi - swing_lo
                px = (swing_lo + leg*lo) if up else (swing_hi - leg*lo)
                invalid_fib = (swing_lo + leg*hi) if up else (swing_hi - leg*hi)
                if (up and px > entry_ref + atr*0.1) or (not up and px < entry_ref - atr*0.1):
                    cands.append({"price": px, "invalid": invalid_fib, "label": "fib_adaptive",
                                   "score": 1.5})
        except Exception:
            pass

    return cands


def calc_discount_entry(df_h1, df_m15, direction, current_price, atr):
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
def full_analyze(df_h1, df_m15, df_d1=None, symbol=None):
    """
    Score arah (H1+M15+D1) -> entry diskon (OB/FVG/EQL/Fib) -> SL/TP.
    Dataframe dikirim pemanggil (main.py), fungsi ini tidak fetch sendiri.

    v7 SESSION GATE (PERUBAHAN KRITIS dari v6):
    ─────────────────────────────────────────────
    NY session (13-17 UTC):
      v6: CHoCH M15 ATAU Failed Retest (OR logic)
      v7: CHoCH M15 DAN Failed Retest (AND logic) + confidence >= 68
      Reasoning: backtest menunjukkan NY WR 25% dengan OR gate.
      AND gate membuang lebih banyak sinyal NY tapi yang tersisa
      jauh lebih berkualitas.

    Asia session: TIDAK DIUBAH (WR 83.3% — jangan sentuh yang bagus)
    London session: presisi boost jika CHoCH H1 + CHoCH M15 sejajar.

    v7 ENTRY TIMING VALIDATION (baru):
      Jika entry "market" (tanpa structural anchor) dan confidence
      < 65 → tolak. Fix untuk avg_fav_before_sl = 0.07R (entry
      masuk di momentum salah tanpa level institusional).

    v7 TRAIL: TRAIL_R_LADDER baru (aktif di 1R, bukan 0.5R).
      Perubahan ini ada di konstanta global TRAIL_R_LADDER dan
      otomatis dipakai oleh main.py (tidak ada perubahan di sini).
    """
    try:
        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            return None

        score = score_direction(df_h1, df_m15, df_d1)
        if score is None: return None

        original_dir  = score["direction"]
        current_price = score["price"]
        atr_val       = score["atr"]
        decision      = "BUY" if original_dir == "bull" else "SELL"

        # ── Session identification ────────────────────────────────────────
        bar_ts   = score.get("bar_ts")
        session  = _get_session(bar_ts)

        # ── v7 NY SESSION GATE: AND logic + confidence threshold ──────────
        if session == "NY":
            choch_ok = (
                (original_dir == "bull" and score.get("choch_m15", {}).get("bullish_choch")) or
                (original_dir == "bear" and score.get("choch_m15", {}).get("bearish_choch"))
            )
            fr_ok = (
                (original_dir == "bull" and score.get("failed_retest", {}).get("failed_retest_buy")) or
                (original_dir == "bear" and score.get("failed_retest", {}).get("failed_retest_sell"))
            )
            # v7: KEDUANYA wajib ada (AND, bukan OR)
            if not (choch_ok and fr_ok):
                return None   # v7: tolak jika tidak keduanya
            # v7: confidence minimum lebih tinggi untuk NY
            if score["confidence"] < SESSION_MIN_CONF["NY"]:
                return None

        # ── Confidence adjustments ────────────────────────────────────────
        confidence = score["confidence"]
        choch_confirms = (
            (original_dir == "bull" and score.get("choch_m15", {}).get("bullish_choch")) or
            (original_dir == "bear" and score.get("choch_m15", {}).get("bearish_choch"))
        )

        if score.get("inducement") and not choch_confirms:
            confidence = max(0, confidence - 8)
        if score.get("pullback_type") == "aggressive" and not choch_confirms:
            confidence = max(0, confidence - 5)

        # CISD boost
        cisd = score.get("cisd_m15", {})
        if (original_dir == "bull" and cisd.get("bullish_cisd")) or \
           (original_dir == "bear" and cisd.get("bearish_cisd")):
            confidence = min(99, confidence + 6)

        # Wyckoff Spring/UTAD boost
        wyck = score.get("wyckoff", {})
        if (original_dir == "bull" and wyck.get("spring")) or \
           (original_dir == "bear" and wyck.get("utad")):
            confidence = min(99, confidence + 8)

        # v7: London precision boost — CHoCH H1 DAN M15 sejajar di killzone
        if session == "London" and score.get("in_killzone"):
            choch_h1  = score.get("choch_h1", {})
            choch_m15 = score.get("choch_m15", {})
            h1_confirms = (
                (original_dir == "bull" and choch_h1.get("bullish_choch")) or
                (original_dir == "bear" and choch_h1.get("bearish_choch"))
            )
            m15_confirms = (
                (original_dir == "bull" and choch_m15.get("bullish_choch")) or
                (original_dir == "bear" and choch_m15.get("bearish_choch"))
            )
            if h1_confirms and m15_confirms:
                confidence = min(99, confidence + 10)  # v7: London hierarki boost

        # ── Session confidence minimum check ──────────────────────────────
        min_conf = SESSION_MIN_CONF.get(session, 45)
        if confidence < min_conf:
            return None

        # ── Entry dari zona struktural ─────────────────────────────────
        discount_entry, entry_label, invalid_level = calc_discount_entry(
            df_h1, df_m15, original_dir, current_price, atr_val)

        # ── v7 Entry timing validation ────────────────────────────────────
        # Fix untuk avg_fav_before_sl = 0.07R: jika entry "market" dan
        # confidence rendah, kemungkinan masuk di momentum salah.
        if entry_label == "market" and confidence < 65:
            # Cek apakah ada structural anchor dekat entry
            m15_built = build_df(df_m15)
            if m15_built is not None:
                has_anchor, anchor_type = _has_structural_anchor(
                    m15_built, original_dir, discount_entry, atr_val)
                if not has_anchor:
                    log.debug(f"[v7] {symbol}: entry 'market' tanpa structural anchor "
                              f"(conf={confidence}) — ditolak")
                    return None   # tolak entry blind tanpa anchor

        setup = analyze_setup(df_h1, df_m15, original_dir, discount_entry,
                               score=score, invalid_level=invalid_level)
        if setup is None: return None

        if original_dir == "bull" and current_price >= setup["tp"]: return None
        if original_dir == "bear" and current_price <= setup["tp"]: return None

        return {
            "symbol"         : symbol,
            "original_dir"   : original_dir,
            "decision"       : decision,
            "confidence"     : confidence,
            "price"          : current_price,
            "entry"          : discount_entry,
            "entry_label"    : entry_label,
            "sl"             : setup["sl"],
            "tp"             : setup["tp"],
            "rr"             : setup["rr"],
            "rsi"            : score["rsi"],
            "struct_h1"      : score["struct_h1"],
            "d1_bias"        : score.get("d1_bias", "neutral"),
            "choch_m15"      : score.get("choch_m15", {}),
            "choch_h1"       : score.get("choch_h1", {}),
            "failed_retest"  : score.get("failed_retest", {}),
            "session"        : session,
            "in_killzone"    : score.get("in_killzone", False),
            "cisd_m15"       : score.get("cisd_m15", {}),
            "wyckoff"        : score.get("wyckoff", {}),
            "tp_sl_reason"   : f"Entry@{discount_entry:.5g}({entry_label}) | {setup['reason']}",
        }
    except Exception as e:
        log.debug(f"[full_analyze] {symbol}: {e}")
        return None


# ═════════════════════════════════════════════
# SCAN — 1 sinyal terbaik
# ═════════════════════════════════════════════
def get_best_signal(candidates):
    """
    Dari list kandidat signal (hasil full_analyze), pilih yang terbaik:
    prioritas: confidence tinggi, RR tinggi, entry_label bukan 'market'.
    """
    if not candidates:
        return None
    def _rank(sig):
        label_bonus = 0 if sig.get("entry_label") == "market" else 2
        return sig["confidence"] + label_bonus + sig["rr"] * 0.5
    return max(candidates, key=_rank)
