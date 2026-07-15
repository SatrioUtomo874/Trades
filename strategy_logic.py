"""
STRATEGY LOGIC — OTAK AI
============================================
FILE INI BERISI SEMUA FUNGSI ANALISA (INDIKATOR, SMC, SCORING, ENTRY, SL/TP)
DI-GENERATE / DIUBAH OLEH AI LOKAL (QWEN 1.7B)
AI BEBAS MENAMBAH, MENGURANGI, ATAU MENGUBAH FUNGSI DI BAWAH INI.
============================================
"""
import pandas as pd
import numpy as np
import logging
log = logging.getLogger(__name__)

# ==================== PARAMETER OVERRIDE ====================
# AI bisa mengubah nilai-nilai ini. try22.py akan membaca dan meng-override.
MIN_CONFIDENCE = 50
MIN_RR = 2.0
MAX_POSITIONS = 20
TRAIL_R_LADDER = [
    (0.5, 0.15),
    (1.0, 0.35),
    (1.5, 0.50),
    (2.0, 0.65),
    (2.8, 0.80),
    (3.5, 0.85),
]
FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618
H4_RSI_BUY_MIN = 45
H4_RSI_BUY_MAX = 68
H4_RSI_SELL_MIN = 32
H4_RSI_SELL_MAX = 55
TP_RR_CAP = 4.0
STRUCT_TRAIL_LB = 2
STRUCT_TRAIL_BUF_PCT = 0.0015
STRUCT_TRAIL_LOOKBACK = 60

# ==================== 1. INDIKATOR ====================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd(s):
    line = ema(s, 12) - ema(s, 26)
    sig = ema(line, 9)
    return line, sig, line - sig

def atr_fn(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build_df(df):
    if len(df) < 60:
        return None
    df = df.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200) if len(df) >= 200 else ema(df["close"], 50)
    df["rsi"] = rsi(df["close"])
    df["ml"], df["ms"], df["mh"] = macd(df["close"])
    df["atr"] = atr_fn(df)
    df["vol_sma"] = df["volume"].rolling(20).mean()
    bm = df["close"].rolling(20).mean()
    bs = df["close"].rolling(20).std()
    df["bb_up"] = bm + 2 * bs
    df["bb_lo"] = bm - 2 * bs
    df["bb_mid"] = bm
    return df.dropna()

# ==================== 2. SMC TOOLS ====================
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
    hl = df["low"].iloc[sl[-1]] > df["low"].iloc[sl[-2]]
    lh = df["high"].iloc[sh[-1]] < df["high"].iloc[sh[-2]]
    ll = df["low"].iloc[sl[-1]] < df["low"].iloc[sl[-2]]
    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"

def detect_bos(df, sh, sl):
    res = {"bb": False, "bs": False, "cb": False, "cs": False}
    hi, lo = df["high"].iloc[-1], df["low"].iloc[-1]
    if len(sh) >= 2:
        ph, lh = df["high"].iloc[sh[-2]], df["high"].iloc[sh[-1]]
        if hi > ph:
            res["bb" if lh > ph else "cb"] = True
    if len(sl) >= 2:
        pl, ll = df["low"].iloc[sl[-2]], df["low"].iloc[sl[-1]]
        if lo < pl:
            res["bs" if ll < pl else "cs"] = True
    return res

def detect_choch(df, sh, sl):
    result = {"bearish_choch": False, "bullish_choch": False}
    close = df["close"].iloc[-1]
    if len(sh) >= 2 and len(sl) >= 2:
        prev_high, last_high = df["high"].iloc[sh[-2]], df["high"].iloc[sh[-1]]
        prev_low, last_low = df["low"].iloc[sl[-2]], df["low"].iloc[sl[-1]]
        if last_high < prev_high and close < prev_low:
            result["bearish_choch"] = True
        if last_high > prev_high and close > prev_low and last_low > prev_low:
            result["bullish_choch"] = True
    return result

def is_zone_fresh(df, top, bot, formed_idx, end_idx=None):
    n = len(df)
    end_idx = end_idx if end_idx is not None else n - 1
    start = formed_idx + 2
    if start >= end_idx:
        return True
    sub = df.iloc[start:end_idx]
    if sub.empty:
        return True
    return not ((sub["low"] <= top) & (sub["high"] >= bot)).any()

def get_fib_zone(price, swing_low, swing_high):
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
    default = (0.5, 0.618)
    if len(sh) < 2 or len(sl) < 2:
        return default
    try:
        if direction == "bull":
            impulse_len = df["high"].iloc[sh[-1]] - df["low"].iloc[sl[-2]]
            pullback_len = df["high"].iloc[sh[-1]] - df["close"].iloc[-1]
        else:
            impulse_len = df["high"].iloc[sh[-2]] - df["low"].iloc[sl[-1]]
            pullback_len = df["close"].iloc[-1] - df["low"].iloc[sl[-1]]
        if impulse_len <= 0:
            return default
        pullback_ratio = abs(pullback_len) / impulse_len
    except Exception:
        return default
    if pullback_ratio <= 0.12:
        return (0.236, 0.382)
    elif pullback_ratio <= 0.30:
        return (0.382, 0.5)
    elif pullback_ratio >= 0.55:
        return (0.618, 0.786)
    else:
        return (0.5, 0.618)

def find_zones(df, direction, lb=40, strict=False):
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
        c = sub.iloc[i]
        nx = sub.iloc[i + 1]
        nx2 = sub.iloc[i + 2] if i + 2 < len(sub) else None
        impulse_body = abs(nx["close"] - nx["open"])
        min_impulse = avg_body * (1.5 if strict else 1.3)
        if impulse_body < min_impulse:
            continue
        if is_demand:
            is_match = c["close"] < c["open"] and nx["close"] > nx["open"]
            if strict and is_match and nx2 is not None:
                is_match = nx2["close"] > nx2["open"]
        else:
            is_match = c["close"] > c["open"] and nx["close"] < nx["open"]
            if strict and is_match and nx2 is not None:
                is_match = nx2["close"] < nx2["open"]
        if not is_match:
            continue
        top = max(c["open"], c["close"])
        bot = min(c["open"], c["close"])
        df_idx = base_offset + i
        has_fvg = False
        if nx2 is not None:
            if is_demand and nx2["low"] > c["high"]:
                has_fvg = True
            if (not is_demand) and nx2["high"] < c["low"]:
                has_fvg = True
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
        pattern = classify_sd_pattern(df, df_idx, "demand" if is_demand else "supply")
        fib = get_fib_zone((top + bot) / 2, swing_lo, swing_hi)
        fib_aligned = fib["zone"] in (("discount", "equilibrium") if is_demand else ("premium", "equilibrium"))
        zones.append({
            "top": top, "bot": bot, "mid": (top + bot) / 2,
            "idx": df_idx, "has_fvg": bool(has_fvg), "has_bos": bool(has_bos),
            "is_fresh": bool(fresh), "pattern": pattern,
            "fib_zone": fib["zone"], "fib_ratio": fib["ratio"],
            "fib_aligned": bool(fib_aligned),
            "quality": int(has_fvg) + int(has_bos) + int(fresh),
        })
    return zones[-3:] if zones else []

def find_fvg(df, direction, lb=40):
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

def detect_failed_retest(df, sh, sl, atr):
    result = {"failed_retest_sell": False, "failed_retest_buy": False, "resistance": None, "support": None}
    if len(df) < 3:
        return result
    L = df.iloc[-1]
    P = df.iloc[-2]
    if len(sh) >= 2:
        resistance = df["high"].iloc[sh[-2]]
        touched = P["high"] >= resistance - atr * 0.5
        rejected = L["close"] < resistance - atr * 0.3
        bearish_c = L["close"] < L["open"]
        if touched and rejected and bearish_c:
            result["failed_retest_sell"] = True
            result["resistance"] = resistance
    if len(sl) >= 2:
        support = df["low"].iloc[sl[-2]]
        touched = P["low"] <= support + atr * 0.5
        bounced = L["close"] > support + atr * 0.3
        bullish_c = L["close"] > L["open"]
        if touched and bounced and bullish_c:
            result["failed_retest_buy"] = True
            result["support"] = support
    return result

def classify_pullback_type(df, direction, atr, lookback=6):
    if len(df) < lookback + 1:
        return "corrective"
    sub = df.iloc[-lookback:]
    bodies = (sub["close"] - sub["open"]).abs()
    avg_body = bodies.mean()
    highs = sub["high"].values
    lows = sub["low"].values
    tol = atr * 0.15
    has_equal_high = False
    has_equal_low = False
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

def is_valid_pullback(df, direction, lookback=8):
    if len(df) < lookback + 2:
        return False
    sub = df.iloc[-lookback:]
    if direction == "bull":
        for i in range(len(sub) - 1, -1, -1):
            c = sub.iloc[i]
            if c["close"] > c["open"]:
                last_low = c["low"]
                found_i = i
                break
        else:
            return False
        after = sub.iloc[found_i + 1:]
        return bool((after["close"] < last_low).any())
    else:
        for i in range(len(sub) - 1, -1, -1):
            c = sub.iloc[i]
            if c["close"] < c["open"]:
                last_high = c["high"]
                found_i = i
                break
        else:
            return False
        after = sub.iloc[found_i + 1:]
        return bool((after["close"] > last_high).any())

def classify_fvg_candle3(df, fvg_idx_c2, direction):
    if fvg_idx_c2 is None or fvg_idx_c2 >= len(df):
        return "unknown"
    c2 = df.iloc[fvg_idx_c2]
    is_bull_candle = c2["close"] > c2["open"]
    if direction == "bull":
        return "breakaway" if is_bull_candle else "rejection"
    else:
        return "rejection" if is_bull_candle else "breakaway"

def detect_pinbar(candle, min_wick_ratio=1.5):
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return {"is_pinbar": False, "bullish_pinbar": False, "bearish_pinbar": False}
    low_wick = min(o, c) - l
    up_wick = h - max(o, c)
    bullish_pinbar = low_wick > body * min_wick_ratio and low_wick > up_wick * 1.5
    bearish_pinbar = up_wick > body * min_wick_ratio and up_wick > low_wick * 1.5
    return {"is_pinbar": bool(bullish_pinbar or bearish_pinbar), "bullish_pinbar": bool(bullish_pinbar), "bearish_pinbar": bool(bearish_pinbar)}

def detect_fakey(df):
    result = {"is_fakey": False, "bullish_fakey": False, "bearish_fakey": False}
    if len(df) < 3:
        return result
    mother = df.iloc[-3]
    inside = df.iloc[-2]
    last = df.iloc[-1]
    is_inside = inside["high"] <= mother["high"] and inside["low"] >= mother["low"]
    if not is_inside:
        return result
    broke_up = last["high"] > mother["high"]
    broke_down = last["low"] < mother["low"]
    closed_inside = mother["low"] <= last["close"] <= mother["high"]
    if broke_down and closed_inside and last["close"] > last["open"]:
        result["is_fakey"] = True
        result["bullish_fakey"] = True
    elif broke_up and closed_inside and last["close"] < last["open"]:
        result["is_fakey"] = True
        result["bearish_fakey"] = True
    return result

def detect_liquidity_run_or_sweep(df, sh, sl, direction):
    result = {"type": "none", "level": None}
    if direction == "bull" and len(sh) >= 1:
        level = df["high"].iloc[sh[-1]]
        last = df.iloc[-1]
        if last["high"] > level and last["close"] > level:
            result = {"type": "run", "level": level}
        elif last["high"] > level and last["close"] <= level:
            result = {"type": "sweep", "level": level}
    elif direction == "bear" and len(sl) >= 1:
        level = df["low"].iloc[sl[-1]]
        last = df.iloc[-1]
        if last["low"] < level and last["close"] < level:
            result = {"type": "run", "level": level}
        elif last["low"] < level and last["close"] >= level:
            result = {"type": "sweep", "level": level}
    return result

def detect_inducement_move(df, direction, atr, lookback=5):
    if len(df) < lookback + 1:
        return False
    sub = df.iloc[-lookback:-1]
    if sub.empty:
        return False
    small_moves = (sub["close"] - sub["open"]).abs() < atr * 0.6
    if direction == "bull":
        counter = sub["close"] < sub["open"]
    else:
        counter = sub["close"] > sub["open"]
    return bool((small_moves & counter).tail(3).any())

def classify_sd_pattern(df, zone_idx, direction, lb=6):
    if zone_idx is None or zone_idx < lb or zone_idx + lb >= len(df):
        return "unknown"
    before = df.iloc[max(0, zone_idx - lb):zone_idx]
    after = df.iloc[zone_idx + 1:zone_idx + 1 + lb]
    if before.empty or after.empty:
        return "unknown"
    move_before = before["close"].iloc[-1] - before["close"].iloc[0]
    move_after = after["close"].iloc[-1] - after["close"].iloc[0]
    before_up = move_before > 0
    after_up = move_after > 0
    if direction == "demand":
        if before_up and after_up:
            return "RBR"
        if (not before_up) and after_up:
            return "DBR"
        return "unknown"
    else:
        if (not before_up) and (not after_up):
            return "DBD"
        if before_up and (not after_up):
            return "RBD"
        return "unknown"

# ==================== 3. SCORING (Layer 1, 2, 3) ====================
def score_direction(df_h1, df_m15, df_d1=None):
    h1 = build_df(df_h1)
    m15 = build_df(df_m15)
    if h1 is None or m15 is None:
        return None
    L1 = h1.iloc[-1]
    L15 = m15.iloc[-1]
    atr_val = max(L15["atr"], L15["close"] * 0.003)
    sh1, sl1 = swing_pts(h1, 5)
    sh15, sl15 = swing_pts(m15, 5)
    struct_h1 = mkt_struct(h1, sh1, sl1)
    choch_h1 = detect_choch(h1, sh1, sl1)
    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            df_d1_built = build_df(df_d1)
        else:
            df_d1_built = build_df(df_h1.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna())
        if df_d1_built is not None and len(df_d1_built) >= 10:
            LD = df_d1_built.iloc[-1]
            sh_d, sl_d = swing_pts(df_d1_built, lb=3)
            struct_d1 = mkt_struct(df_d1_built, sh_d, sl_d)
            ema_bear_d1 = LD["ema9"] < LD["ema21"] < LD["ema50"]
            ema_bull_d1 = LD["ema9"] > LD["ema21"] > LD["ema50"]
            if struct_d1 == "bearish" or ema_bear_d1:
                d1_bias = "bearish"
            elif struct_d1 == "bullish" or ema_bull_d1:
                d1_bias = "bullish"
    except Exception:
        pass

    # LAYER 1 — BIAS
    bias_bull = bias_bear = 0
    if struct_h1 == "bullish":
        bias_bull += 30
    if struct_h1 == "bearish":
        bias_bear += 30
    if choch_h1["bullish_choch"]:
        bias_bull += 26
    if choch_h1["bearish_choch"]:
        bias_bear += 26
    if L1["ema9"] > L1["ema21"] > L1["ema50"]:
        bias_bull += 15
    elif L1["ema9"] > L1["ema21"]:
        bias_bull += 7
    if L1["ema9"] < L1["ema21"] < L1["ema50"]:
        bias_bear += 15
    elif L1["ema9"] < L1["ema21"]:
        bias_bear += 7
    if L1["close"] > L1["ema200"]:
        bias_bull += 8
    else:
        bias_bear += 8
    if d1_bias == "bullish":
        bias_bull += 24
    if d1_bias == "bearish":
        bias_bear += 24
    rv = L15["rsi"]
    if rv < 35:
        bias_bull += 12
    elif rv < 45:
        bias_bull += 6
    if rv > 65:
        bias_bear += 12
    elif rv > 55:
        bias_bear += 6
    bias_dir = "bull" if bias_bull >= bias_bear else "bear"

    # LAYER 2 — SETUP
    setup_bull = setup_bear = 0
    bos = detect_bos(m15, sh15, sl15)
    if bos["bb"]:
        setup_bull += 12
    if bos["cb"]:
        setup_bull += 7
    if bos["bs"]:
        setup_bear += 12
    if bos["cs"]:
        setup_bear += 7
    choch = detect_choch(m15, sh15, sl15)
    if choch["bullish_choch"]:
        setup_bull += 22
    if choch["bearish_choch"]:
        setup_bear += 22
    fr = detect_failed_retest(m15, sh15, sl15, atr_val)
    if fr["failed_retest_sell"]:
        setup_bear += 24
    if fr["failed_retest_buy"]:
        setup_bull += 24
    fr_h1 = detect_failed_retest(h1, sh1, sl1, atr_val)
    if fr_h1["failed_retest_sell"]:
        setup_bear += 18
    if fr_h1["failed_retest_buy"]:
        setup_bull += 18

    pullback_valid_bull = is_valid_pullback(m15, "bull")
    pullback_valid_bear = is_valid_pullback(m15, "bear")
    pullback_type_bull = classify_pullback_type(m15, "bull", atr_val)
    pullback_type_bear = classify_pullback_type(m15, "bear", atr_val)
    if pullback_valid_bull:
        if pullback_type_bull == "aggressive":
            setup_bull += 3
        elif pullback_type_bull == "sweeping":
            setup_bull += 14
        else:
            setup_bull += 9
    if pullback_valid_bear:
        if pullback_type_bear == "aggressive":
            setup_bear += 3
        elif pullback_type_bear == "sweeping":
            setup_bear += 14
        else:
            setup_bear += 9

    pinbar = detect_pinbar(L15)
    if pinbar["bullish_pinbar"]:
        setup_bull += 10
    if pinbar["bearish_pinbar"]:
        setup_bear += 10
    fakey = detect_fakey(m15)
    if fakey["bullish_fakey"]:
        setup_bull += 10
    if fakey["bearish_fakey"]:
        setup_bear += 10

    liq_bull = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bear")
    if liq_bull["type"] == "run":
        setup_bull += 10
    elif liq_bull["type"] == "sweep":
        setup_bear += 8
    if liq_bear["type"] == "run":
        setup_bear += 10
    elif liq_bear["type"] == "sweep":
        setup_bull += 8

    ote_bull = ote_bear = False
    if len(sh15) >= 1 and len(sl15) >= 1:
        swing_hi_m15 = m15["high"].iloc[sh15[-1]]
        swing_lo_m15 = m15["low"].iloc[sl15[-1]]
        fib_now = get_fib_zone(L15["close"], swing_lo_m15, swing_hi_m15)
        if 0.62 <= (1 - fib_now["ratio"]) <= 0.79:
            ote_bull = True
        if 0.62 <= fib_now["ratio"] <= 0.79:
            ote_bear = True
    if ote_bull and (choch["bullish_choch"] or any(f.get("is_fresh") for f in find_fvg(m15, "bull", lb=30))):
        setup_bull += 10
    if ote_bear and (choch["bearish_choch"] or any(f.get("is_fresh") for f in find_fvg(m15, "bear", lb=30))):
        setup_bear += 10

    body = L15["close"] - L15["open"]
    low_wick = min(L15["open"], L15["close"]) - L15["low"]
    up_wick = L15["high"] - max(L15["open"], L15["close"])
    if low_wick > abs(body) * 1.5:
        setup_bull += 6
    if up_wick > abs(body) * 1.5:
        setup_bear += 6

    if L15["mh"] > 0:
        setup_bull += 8
    elif L15["mh"] < 0:
        setup_bear += 8
    if L15["close"] <= L15["bb_lo"]:
        setup_bull += 7
    elif L15["close"] < L15["bb_mid"]:
        setup_bull += 3
    if L15["close"] >= L15["bb_up"]:
        setup_bear += 7
    elif L15["close"] > L15["bb_mid"]:
        setup_bear += 3
    if L15["volume"] > L15["vol_sma"] * 1.5:
        if L15["close"] > L15["open"]:
            setup_bull += 6
        else:
            setup_bear += 6
    elif L15["volume"] > L15["vol_sma"]:
        if L15["close"] > L15["open"]:
            setup_bull += 2
        else:
            setup_bear += 2

    # LAYER 3 — GATE
    if bias_dir == "bull":
        setup_bear = setup_bear * 0.5
    else:
        setup_bull = setup_bull * 0.5

    bull = bias_bull + setup_bull
    bear = bias_bear + setup_bear
    direction = "bull" if bull >= bear else "bear"
    raw = bull if direction == "bull" else bear
    conf = min(int(raw / 264 * 100), 99)

    if d1_bias == "bearish" and direction == "bull":
        return None
    if d1_bias == "bullish" and direction == "bear":
        return None

    return {
        "direction": direction,
        "confidence": conf,
        "price": L15["close"],
        "atr": atr_val,
        "struct_h1": struct_h1,
        "d1_bias": d1_bias,
        "rsi": round(rv, 1),
        "choch_m15": choch,
        "choch_h1": choch_h1,
        "failed_retest": fr,
        "pullback_valid": pullback_valid_bull if direction == "bull" else pullback_valid_bear,
        "pullback_type": pullback_type_bull if direction == "bull" else pullback_type_bear,
        "pinbar": pinbar,
        "fakey": fakey,
        "liquidity_bull": liq_bull,
        "liquidity_bear": liq_bear,
        "inducement": inducement_bull if direction == "bull" else inducement_bear,
    }

# ==================== 4. ENTRY & SL/TP ====================
def _zone_score(z):
    return z.get("quality", 0) + int(z.get("fib_aligned", False))

def _collect_entry_candidates(m15, direction, entry_ref, atr):
    up = direction == "bear"
    obs = find_zones(m15, direction, strict=True)
    fvgs = find_fvg(m15, direction)
    eqs = find_equal_highs_lows(m15, "high" if up else "low", lb=80)
    cands = []
    def _dist_penalty(price):
        if atr <= 0:
            return 0.0
        return (abs(price - entry_ref) / atr) * 0.15
    for z in obs:
        entry_pt, invalid_pt = (z["top"], z["bot"]) if up else (z["bot"], z["top"])
        if (up and entry_pt > entry_ref + atr * 0.1) or (not up and entry_pt < entry_ref - atr * 0.1):
            cands.append({"price": entry_pt, "invalid": invalid_pt, "label": "ob", "score": 3 + _zone_score(z) - _dist_penalty(entry_pt)})
    for f in fvgs:
        if (up and f["mid"] > entry_ref + atr * 0.1) or (not up and f["mid"] < entry_ref - atr * 0.1):
            sc = 2 + int(f.get("is_fresh", False)) + 2 * int(f.get("candle3") == "breakaway")
            invalid_pt = f["top"] if up else f["bot"]
            cands.append({"price": f["mid"], "invalid": invalid_pt, "label": "fvg", "score": sc - _dist_penalty(f["mid"])})
    eqs_sorted = sorted(eqs) if up else sorted(eqs, reverse=True)
    for lv in eqs_sorted[:1]:
        if (up and lv > entry_ref + atr * 0.2) or (not up and lv < entry_ref - atr * 0.2):
            cands.append({"price": lv, "invalid": lv + (atr * 0.6 if up else -atr * 0.6), "label": "eq", "score": 2 - _dist_penalty(lv)})
    if not cands:
        try:
            sh15, sl15 = swing_pts(m15, lb=5)
            if len(sh15) >= 1 and len(sl15) >= 1:
                lo, hi = adaptive_fib_target(m15, sh15, sl15, direction)
                swing_hi = m15["high"].iloc[sh15[-1]]
                swing_lo = m15["low"].iloc[sl15[-1]]
                leg = swing_hi - swing_lo
                px = (swing_lo + leg * lo) if up else (swing_hi - leg * lo)
                invalid_fib = (swing_lo + leg * hi) if up else (swing_hi - leg * hi)
                if (up and px > entry_ref + atr * 0.1) or (not up and px < entry_ref - atr * 0.1):
                    cands.append({"price": px, "invalid": invalid_fib, "label": "fib_adaptive", "score": 1.5})
        except Exception:
            pass
    return cands

def calc_discount_entry(df_h1, df_m15, direction, current_price, atr):
    m15 = build_df(df_m15)
    if m15 is None:
        return current_price, "market", None
    cands = _collect_entry_candidates(m15, direction, current_price, atr)
    if cands:
        best = max(cands, key=lambda c: c["score"])
        return round(best["price"], 8), best["label"], best["invalid"]
    return current_price, "market", None

def _h4_confluence(df_h1, direction, choch_m15=None):
    result = {"confluence": False, "full_confluence": False}
    try:
        df_h4 = build_df(df_h1.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna())
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
            choch_agrees = ((direction == "bull" and choch_m15.get("bullish_choch")) or
                            (direction == "bear" and choch_m15.get("bearish_choch")))
            result["full_confluence"] = bool(choch_agrees)
    except Exception:
        pass
    return result

def _fib_extension_levels(h1, sh1, sl1, direction):
    if not sh1 or not sl1:
        return None, None
    swing_high = h1["high"].iloc[sh1[-1]]
    swing_low = h1["low"].iloc[sl1[-1]]
    leg = swing_high - swing_low
    if leg <= 0:
        return None, None
    if direction == "bull":
        return swing_high + leg * FIB_EXT_1, swing_high + leg * FIB_EXT_2
    else:
        return swing_low - leg * FIB_EXT_1, swing_low - leg * FIB_EXT_2

def _select_best_tp(tp_pool, entry_price, risk):
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
    up = direction == "bull"
    zones_m15 = find_zones(m15, "demand" if up else "supply")
    zones_h1 = find_zones(h1, "demand" if up else "supply")
    fvgs = find_fvg(m15, "bull" if up else "bear")
    eqs_m15 = find_equal_highs_lows(m15, "high" if up else "low", lb=80)
    eqs_h1 = find_equal_highs_lows(h1, "high" if up else "low", lb=50)
    sw_m15 = [m15["high" if up else "low"].iloc[i] for i in (sh15 if up else sl15)]
    sw_h1 = [h1["high" if up else "low"].iloc[i] for i in (sh1 if up else sl1)]
    sgn = 1 if up else -1
    pool = []
    for v in eqs_h1:
        if sgn * (v - entry_price) > atr * 1.0:
            pool.append(("eq_h1", v, 1))
    for z in zones_h1:
        edge = z["bot"] if up else z["top"]
        if sgn * (edge - entry_price) > atr * 1.0:
            pool.append(("zone_h1", edge, 2))
    for v in sw_h1:
        if sgn * (v - entry_price) > atr * 1.0:
            pool.append(("sw_h1", v, 3))
    for v in eqs_m15:
        if sgn * (v - entry_price) > atr * 0.5:
            pool.append(("eq_m15", v, 4))
    for z in zones_m15:
        edge = z["bot"] if up else z["top"]
        if sgn * (edge - entry_price) > atr * 0.5:
            pool.append(("zone_m15", edge, 5 - (0.4 if z.get("is_fresh") else 0)))
    for f in fvgs:
        if sgn * (f["mid"] - entry_price) > atr * 0.5:
            t = 6 - (0.4 if f.get("candle3") == "breakaway" else 0) - (0.2 if f.get("is_fresh") else 0)
            pool.append(("fvg_m15", f["mid"], t))
    for v in sw_m15:
        if sgn * (v - entry_price) > atr * 0.5:
            pool.append(("sw_m15", v, 7))
    if fib_127 is not None and sgn * (fib_127 - entry_price) > atr * 0.5 and h4_gate["confluence"]:
        pool.append(("fib127", fib_127, 8))
        if h4_gate["full_confluence"] and fib_162 is not None and sgn * (fib_162 - entry_price) > atr * 0.5:
            pool.append(("fib162", fib_162, 9))
    return pool

def analyze_setup(df_h1, df_m15, direction, entry_price, score=None, invalid_level=None):
    h1, m15 = build_df(df_h1), build_df(df_m15)
    if h1 is None or m15 is None:
        return None
    atr_m15 = m15["atr"].iloc[-1]
    atr_h1 = h1["atr"].iloc[-1] / 4
    atr = max(atr_m15, atr_h1, entry_price * 0.002)
    noise = atr * 0.6
    if invalid_level is None:
        return None
    sl_price = invalid_level + (noise if direction == "bear" else -noise)
    risk = abs(sl_price - entry_price)
    risk_floor = max(atr * 0.8, entry_price * 0.003)
    if risk < risk_floor:
        sl_price += (risk_floor - risk) * (1 if direction == "bear" else -1)
        risk = risk_floor
    if risk <= 0:
        return None
    sh15, sl15 = swing_pts(m15, lb=5)
    sh1, sl1 = swing_pts(h1, lb=5)
    choch_m15 = (score or {}).get("choch_m15", {})
    h4_gate = _h4_confluence(df_h1, direction, choch_m15)
    fib_127, fib_162 = _fib_extension_levels(h1, sh1, sl1, direction)
    tp_pool = _build_tp_pool(m15, h1, direction, entry_price, atr, sh15, sl15, sh1, sl1, h4_gate, fib_127, fib_162)
    tp_price, tp_label = _select_best_tp(tp_pool, entry_price, risk)
    if tp_price is None:
        return None
    reward = abs(tp_price - entry_price)
    rr = round(reward / risk, 2)
    if rr < MIN_RR:
        return None
    return {"sl": round(sl_price, 8), "tp": round(tp_price, 8), "rr": rr, "reason": f"SL@{sl_price:.5g}(invalidation) | TP@{tp_price:.5g}({tp_label})"}

# ==================== 5. FULL ANALYZE (PIPELINE) ====================
def full_analyze(symbol):
    """
    1. Score arah sinyal (H1 + M15 + D1 bias)
    2. Hitung entry diskon dari OB/FVG/EQL/Fib
    3. Hitung SL/TP dari entry diskon
    Entry = zona struktural, bukan market price
    """
    # Fungsi get_klines akan diambil dari namespace try22.py (karena di-import di sana)
    # Ini adalah dependency ke TUBUH yang TIDAK BOLEH diubah.
    try:
        # Cari get_klines di namespace global (dari try22.py)
        import sys
        get_klines = sys.modules['__main__'].get_klines
    except (AttributeError, KeyError):
        # Fallback: coba import dari module utama
        try:
            from __main__ import get_klines
        except ImportError:
            log.error("[OTAK] Gagal mengakses get_klines dari TUBUH!")
            return None

    try:
        df_h1 = get_klines(symbol, "1h", 250)
        df_m15 = get_klines(symbol, "15m", 250)
        if df_h1.empty or df_m15.empty:
            return None
        try:
            df_d1 = get_klines(symbol, "1d", 100)
        except Exception:
            df_d1 = None

        score = score_direction(df_h1, df_m15, df_d1)
        if score is None:
            return None

        original_dir = score["direction"]
        current_price = score["price"]
        atr_val = score["atr"]
        confidence = score["confidence"]
        choch_confirms = ((original_dir == "bull" and score.get("choch_m15", {}).get("bullish_choch")) or
                          (original_dir == "bear" and score.get("choch_m15", {}).get("bearish_choch")))
        if score.get("inducement") and not choch_confirms:
            confidence = max(0, confidence - 8)
        if score.get("pullback_type") == "aggressive" and not choch_confirms:
            confidence = max(0, confidence - 5)

        discount_entry, entry_label, invalid_level = calc_discount_entry(
            df_h1, df_m15, original_dir, current_price, atr_val)
        setup = analyze_setup(df_h1, df_m15, original_dir, discount_entry, score=score, invalid_level=invalid_level)
        if setup is None:
            return None
        if original_dir == "bull" and current_price >= setup["tp"]:
            return None
        if original_dir == "bear" and current_price <= setup["tp"]:
            return None

        return {
            "symbol": symbol,
            "original_dir": original_dir,
            "decision": "BUY" if original_dir == "bull" else "SELL",
            "confidence": confidence,
            "price": current_price,
            "entry": discount_entry,
            "entry_label": entry_label,
            "sl": setup["sl"],
            "tp": setup["tp"],
            "rr": setup["rr"],
            "rsi": score["rsi"],
            "struct_h1": score["struct_h1"],
            "d1_bias": score.get("d1_bias", "neutral"),
            "choch_m15": score.get("choch_m15", {}),
            "choch_h1": score.get("choch_h1", {}),
            "failed_retest": score.get("failed_retest", {}),
            "tp_sl_reason": f"Entry@{discount_entry:.5g}({entry_label}) | {setup['reason']}",
        }
    except Exception as e:
        log.debug(f"[OTAK full_analyze] {symbol}: {e}")
        return None

# ==================== 6. PARAMETER OVERRIDE UNTUK TUBUH ====================
# Parameter ini akan dibaca oleh try22.py (jika ada) untuk meng-override nilai default.
# Nama variabel harus SAMA PERSIS dengan di try22.py.
MIN_CONFIDENCE = MIN_CONFIDENCE
MIN_RR = MIN_RR
MAX_POSITIONS = MAX_POSITIONS
TRAIL_R_LADDER = TRAIL_R_LADDER
