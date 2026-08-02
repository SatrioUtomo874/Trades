import logging
import pandas as pd
import numpy as np
from datetime import timezone

log = logging.getLogger(__name__)

# ============================================================
# KONFIGURASI
# ============================================================

MIN_RR = 1.5

TRAIL_R_LADDER = [
    (0.8, 0.20),
    (1.5, 0.40),
    (2.0, 0.55),
    (3.0, 0.70),
    (4.0, 0.85),
    (5.0, 0.90),
]

STRUCT_TRAIL_LB = 3
STRUCT_TRAIL_BUF_PCT = 0.002
STRUCT_TRAIL_LOOKBACK = 60

FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618

H4_RSI_BUY_MIN = 40
H4_RSI_BUY_MAX = 65
H4_RSI_SELL_MIN = 35
H4_RSI_SELL_MAX = 60

SESSION_NY_START = 13
SESSION_NY_END = 17
SESSION_LONDON_START = 7
SESSION_LONDON_END = 12
SESSION_KILL_LDN_S = 7
SESSION_KILL_LDN_E = 10
SESSION_KILL_ASIA1_S = 20
SESSION_KILL_ASIA2_E = 5
KILLZONE_BOOST = 1.1

# ============================================================
# SESSION MIN CONFIDENCE (DENGAN 2 KOMPROMI)
# ============================================================
SESSION_MIN_CONF = {
    "Asia": 45,        # KOMPROMI 1: 45 (turun dari 50, naik dari 40)
    "London": 50,
    "NY": 60,
    "transition": 45,
}

# ============================================================
# FUNGSI BANTU
# ============================================================

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
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
    if len(df) < 60: return None
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

def swing_pts(df, lb=5):
    sh, sl = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i - lb:i + lb + 1].max():
            sh.append(i)
        if df["low"].iloc[i] == df["low"].iloc[i - lb:i + lb + 1].min():
            sl.append(i)
    return sh, sl

def mkt_struct(df, sh, sl):
    if len(sh) < 2 or len(sl) < 2: return "ranging"
    hh = df["high"].iloc[sh[-1]] > df["high"].iloc[sh[-2]]
    hl = df["low"].iloc[sl[-1]] > df["low"].iloc[sl[-2]]
    lh = df["high"].iloc[sh[-1]] < df["high"].iloc[sh[-2]]
    ll = df["low"].iloc[sl[-1]] < df["low"].iloc[sl[-2]]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

def _get_session(bar_ts=None):
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
    if SESSION_NY_START <= hour < SESSION_NY_END: return "NY"
    if SESSION_LONDON_START <= hour < SESSION_LONDON_END: return "London"
    if hour >= SESSION_KILL_ASIA1_S or hour < SESSION_KILL_ASIA2_E: return "Asia"
    return "transition"

def _is_in_killzone(bar_ts=None):
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
    ldn_kill = SESSION_KILL_LDN_S <= hour < SESSION_KILL_LDN_E
    asia_kill = hour >= SESSION_KILL_ASIA1_S or hour < SESSION_KILL_ASIA2_E
    return ldn_kill or asia_kill

# ============================================================
# DETEKSI STRUKTUR SMC
# ============================================================

def detect_liquidity_sweep(df, sh, sl, direction):
    result = {"type": "none", "level": None}
    if direction == "bull" and len(sl) >= 1:
        low = df["low"].iloc[sl[-1]]
        if df["low"].iloc[-1] < low and df["close"].iloc[-1] > low:
            result = {"type": "sweep", "level": low}
    elif direction == "bear" and len(sh) >= 1:
        high = df["high"].iloc[sh[-1]]
        if df["high"].iloc[-1] > high and df["close"].iloc[-1] < high:
            result = {"type": "sweep", "level": high}
    return result

def detect_break_of_structure(df, sh, sl, direction):
    if direction == "bull" and len(sh) >= 2:
        prev_high = df["high"].iloc[sh[-2]]
        if df["high"].iloc[-1] > prev_high:
            return True
    elif direction == "bear" and len(sl) >= 2:
        prev_low = df["low"].iloc[sl[-2]]
        if df["low"].iloc[-1] < prev_low:
            return True
    return False

def detect_choch(df, sh, sl):
    result = {"bearish_choch": False, "bullish_choch": False}
    if len(sh) < 2 or len(sl) < 2: return result
    close = df["close"].iloc[-1]
    prev_high = df["high"].iloc[sh[-2]]
    last_high = df["high"].iloc[sh[-1]]
    prev_low = df["low"].iloc[sl[-2]]
    last_low = df["low"].iloc[sl[-1]]
    if last_high > prev_high and last_low > prev_low and close > prev_low:
        result["bullish_choch"] = True
    if last_high < prev_high and last_low < prev_low and close < prev_low:
        result["bearish_choch"] = True
    return result

def detect_cisd(df, lb=6):
    result = {"bullish_cisd": False, "bearish_cisd": False}
    if len(df) < lb + 1: return result
    sub = df.iloc[-lb:]
    closes = sub["close"].values
    opens = sub["open"].values
    n = len(closes)
    last_bull = closes[-1] > opens[-1]
    last_bear = closes[-1] < opens[-1]
    if not (last_bull or last_bear): return result
    if last_bull:
        cnt = 0
        for j in range(n-2, -1, -1):
            if closes[j] < opens[j]: cnt += 1
            else: break
        if cnt >= 3: result["bullish_cisd"] = True
    else:
        cnt = 0
        for j in range(n-2, -1, -1):
            if closes[j] > opens[j]: cnt += 1
            else: break
        if cnt >= 3: result["bearish_cisd"] = True
    return result

def detect_fvg(df, direction, lb=40, min_quality=False):
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    out = []
    for i in range(len(sub) - 2):
        c0, c1, c2 = sub.iloc[i], sub.iloc[i+1], sub.iloc[i+2]
        gap = None
        if direction == "bull" and c2["low"] > c0["high"]:
            gap = {"top": c2["low"], "bot": c0["high"]}
        elif direction == "bear" and c2["high"] < c0["low"]:
            gap = {"top": c0["low"], "bot": c2["high"]}
        if gap:
            gap["mid"] = (gap["top"] + gap["bot"]) / 2
            gap["idx"] = base_offset + i + 2
            gap["is_fresh"] = is_zone_fresh(df, gap["top"], gap["bot"], gap["idx"])
            out.append(gap)
    if min_quality:
        out = [f for f in out if f["is_fresh"]]
    return out[-3:] if out else []

def detect_order_block(df, direction, lb=40):
    is_demand = direction == "bull"
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean()
    zones = []
    for i in range(1, len(sub) - 2):
        c = sub.iloc[i]
        nx = sub.iloc[i+1]
        impulse_body = abs(nx["close"] - nx["open"])
        if impulse_body < avg_body * 1.3: continue
        is_match = (c["close"] < c["open"] and nx["close"] > nx["open"]) if is_demand else (c["close"] > c["open"] and nx["close"] < nx["open"])
        if not is_match: continue
        top = max(c["open"], c["close"])
        bot = min(c["open"], c["close"])
        df_idx = base_offset + i
        sh, sl = swing_pts(df, lb=5)
        has_fvg = False
        if i + 2 < len(sub):
            c2 = sub.iloc[i+2]
            if is_demand and c2["low"] > c["high"]: has_fvg = True
            if not is_demand and c2["high"] < c["low"]: has_fvg = True
        has_bos = detect_break_of_structure(df, sh, sl, direction)
        fresh = is_zone_fresh(df, top, bot, df_idx)
        quality = int(has_fvg) + int(has_bos) + int(fresh)
        if quality >= 2:
            zones.append({
                "top": top, "bot": bot, "mid": (top + bot) / 2,
                "idx": df_idx, "has_fvg": has_fvg, "has_bos": has_bos,
                "is_fresh": fresh, "quality": quality,
            })
    return zones[-3:] if zones else []

def detect_equal_highs_lows(df, kind="high", lb=60, tol=0.0025):
    sub = df.iloc[-lb:]
    vals = sub["high"] if kind == "high" else sub["low"]
    clusters = []
    visited = set()
    for i in range(len(vals)):
        if i in visited: continue
        group = [vals.iloc[i]]
        for j in range(i+1, len(vals)):
            if abs(vals.iloc[i] - vals.iloc[j]) / max(vals.iloc[i], 0.0001) < tol:
                group.append(vals.iloc[j])
                visited.add(j)
        if len(group) >= 2:
            clusters.append(sum(group) / len(group))
    return sorted(clusters)

def detect_failed_retest(df, sh, sl, atr):
    result = {"failed_retest_sell": False, "failed_retest_buy": False}
    if len(df) < 3: return result
    L = df.iloc[-1]
    P = df.iloc[-2]
    if len(sh) >= 2:
        resistance = df["high"].iloc[sh[-2]]
        touched = P["high"] >= resistance - atr * 0.5
        rejected = L["close"] < resistance - atr * 0.3
        bearish_c = L["close"] < L["open"]
        if touched and rejected and bearish_c:
            result["failed_retest_sell"] = True
    if len(sl) >= 2:
        support = df["low"].iloc[sl[-2]]
        touched = P["low"] <= support + atr * 0.5
        bounced = L["close"] > support + atr * 0.3
        bullish_c = L["close"] > L["open"]
        if touched and bounced and bullish_c:
            result["failed_retest_buy"] = True
    return result

def is_zone_fresh(df, top, bot, formed_idx, end_idx=None):
    if formed_idx is None or formed_idx + 2 >= len(df): return True
    start = formed_idx + 2
    end_idx = end_idx if end_idx is not None else len(df) - 1
    if start >= end_idx: return True
    sub = df.iloc[start:end_idx]
    if sub.empty: return True
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)

# ============================================================
# FIBONACCI & ENTRY/EXIT
# ============================================================

def get_fib_zone(price, swing_low, swing_high):
    rng = swing_high - swing_low
    if rng <= 0: return {"ratio": 0.5, "zone": "equilibrium"}
    ratio = (price - swing_low) / rng
    if ratio <= 0.45: zone = "discount"
    elif ratio >= 0.55: zone = "premium"
    else: zone = "equilibrium"
    return {"ratio": round(ratio, 4), "zone": zone}

def is_in_ote(df, direction, sh, sl):
    if len(sh) < 1 or len(sl) < 1: return False
    swing_high = df["high"].iloc[sh[-1]]
    swing_low = df["low"].iloc[sl[-1]]
    fib = get_fib_zone(df["close"].iloc[-1], swing_low, swing_high)
    if direction == "bull":
        return 0.62 <= (1 - fib["ratio"]) <= 0.79
    else:
        return 0.62 <= fib["ratio"] <= 0.79

def _fib_extension_levels(h1, sh1, sl1, direction):
    if not sh1 or not sl1: return None, None
    swing_high = h1["high"].iloc[sh1[-1]]
    swing_low = h1["low"].iloc[sl1[-1]]
    leg = swing_high - swing_low
    if leg <= 0: return None, None
    if direction == "bull":
        return swing_high + leg * FIB_EXT_1, swing_high + leg * FIB_EXT_2
    else:
        return swing_low - leg * FIB_EXT_1, swing_low - leg * FIB_EXT_2

def adaptive_fib_target(df, sh, sl, direction):
    default = (0.5, 0.618)
    if len(sh) < 2 or len(sl) < 2: return default
    try:
        if direction == "bull":
            impulse_len = df["high"].iloc[sh[-1]] - df["low"].iloc[sl[-2]]
            pullback_len = df["high"].iloc[sh[-1]] - df["close"].iloc[-1]
        else:
            impulse_len = df["high"].iloc[sh[-2]] - df["low"].iloc[sl[-1]]
            pullback_len = df["close"].iloc[-1] - df["low"].iloc[sl[-1]]
        if impulse_len <= 0: return default
        pullback_ratio = abs(pullback_len) / impulse_len
    except Exception:
        return default
    if pullback_ratio <= 0.12: return (0.236, 0.382)
    elif pullback_ratio <= 0.30: return (0.382, 0.5)
    elif pullback_ratio >= 0.55: return (0.618, 0.786)
    else: return (0.5, 0.618)

# ============================================================
# SCORING & ANALISIS
# ============================================================

def _h4_confluence(df_h1, direction, choch_m15=None):
    result = {"confluence": False, "full_confluence": False}
    try:
        df_h4 = build_df(df_h1.resample("4h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
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

def score_direction(df_h1, df_m15, df_d1=None):
    h1 = build_df(df_h1)
    m15 = build_df(df_m15)
    if h1 is None or m15 is None: return None

    L1 = h1.iloc[-1]
    L15 = m15.iloc[-1]
    atr_val = max(L15["atr"], L15["close"] * 0.003)

    sh1, sl1 = swing_pts(h1, 5)
    sh15, sl15 = swing_pts(m15, 5)
    struct_h1 = mkt_struct(h1, sh1, sl1)

    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            df_d1_built = build_df(df_d1)
        else:
            df_d1_built = build_df(df_h1.resample("1D").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna())
        if df_d1_built is not None and len(df_d1_built) >= 10:
            LD = df_d1_built.iloc[-1]
            sh_d, sl_d = swing_pts(df_d1_built, lb=3)
            struct_d1 = mkt_struct(df_d1_built, sh_d, sl_d)
            ema_bear_d1 = LD["ema9"] < LD["ema21"] < LD["ema50"]
            ema_bull_d1 = LD["ema9"] > LD["ema21"] > LD["ema50"]
            if struct_d1 == "bearish" or ema_bear_d1: d1_bias = "bearish"
            elif struct_d1 == "bullish" or ema_bull_d1: d1_bias = "bullish"
    except Exception:
        pass

    bias_bull = bias_bear = 0

    if struct_h1 == "bullish": bias_bull += 35
    elif struct_h1 == "bearish": bias_bear += 35

    choch_h1 = detect_choch(h1, sh1, sl1)
    if choch_h1["bullish_choch"]: bias_bull += 25
    if choch_h1["bearish_choch"]: bias_bear += 25

    if L1["ema9"] > L1["ema21"] > L1["ema50"]: bias_bull += 15
    elif L1["ema9"] < L1["ema21"] < L1["ema50"]: bias_bear += 15

    if d1_bias == "bullish": bias_bull += 20
    elif d1_bias == "bearish": bias_bear += 20

    setup_bull = setup_bear = 0

    choch_m15 = detect_choch(m15, sh15, sl15)
    if choch_m15["bullish_choch"]: setup_bull += 30
    if choch_m15["bearish_choch"]: setup_bear += 30

    cisd_m15 = detect_cisd(m15, lb=8)
    if cisd_m15["bullish_cisd"]: setup_bull += 20
    if cisd_m15["bearish_cisd"]: setup_bear += 20

    fr = detect_failed_retest(m15, sh15, sl15, atr_val)
    if fr["failed_retest_sell"]: setup_bear += 25
    if fr["failed_retest_buy"]: setup_bull += 25

    liq_bull = detect_liquidity_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_sweep(m15, sh15, sl15, "bear")
    if liq_bull["type"] == "sweep": setup_bull += 15
    if liq_bear["type"] == "sweep": setup_bear += 15

    if is_in_ote(m15, "bull", sh15, sl15): setup_bull += 10
    if is_in_ote(m15, "bear", sh15, sl15): setup_bear += 10

    if (struct_h1 == "bullish" and setup_bear > setup_bull):
        setup_bear = setup_bear * 0.5
    elif (struct_h1 == "bearish" and setup_bull > setup_bear):
        setup_bull = setup_bull * 0.5

    bar_ts = m15.index[-1] if hasattr(m15.index, '__iter__') else None
    in_killzone = _is_in_killzone(bar_ts)
    if in_killzone:
        if setup_bull > setup_bear: setup_bull *= KILLZONE_BOOST
        else: setup_bear *= KILLZONE_BOOST

    total_bull = bias_bull + setup_bull
    total_bear = bias_bear + setup_bear

    direction = "bull" if total_bull >= total_bear else "bear"
    raw = total_bull if direction == "bull" else total_bear
    conf = min(int(raw / 280 * 100), 99)

    d1_conflict = (d1_bias == "bearish" and direction == "bull") or (d1_bias == "bullish" and direction == "bear")
    if d1_conflict:
        conf = max(0, conf - 15)

    return {
        "direction": direction,
        "confidence": conf,
        "d1_conflict": d1_conflict,
        "price": L15["close"],
        "atr": atr_val,
        "struct_h1": struct_h1,
        "d1_bias": d1_bias,
        "choch_m15": choch_m15,
        "choch_h1": choch_h1,
        "cisd_m15": cisd_m15,
        "failed_retest": fr,
        "liquidity_bull": liq_bull,
        "liquidity_bear": liq_bear,
        "in_killzone": in_killzone,
        "bar_ts": bar_ts,
    }

# ============================================================
# ENTRY & TP
# ============================================================

def _collect_entry_candidates(m15, direction, entry_ref, atr, score=None):
    up = direction == "bull"
    cands = []

    # OB
    obs = detect_order_block(m15, direction, lb=40)
    for z in obs:
        entry_pt = z["top"] if not up else z["bot"]
        invalid_pt = z["bot"] if not up else z["top"]
        if (up and entry_pt < entry_ref + atr * 0.5) or (not up and entry_pt > entry_ref - atr * 0.5):
            cands.append({
                "price": entry_pt,
                "invalid": invalid_pt,
                "label": "ob",
                "score": 3 + z["quality"]
            })

    # FVG (KOMPROMI 2: HANYA JIKA ADA CHoCH, tanpa harus Liquidity Sweep)
    fvgs = detect_fvg(m15, direction, lb=40, min_quality=True)
    for f in fvgs:
        entry_pt = f["mid"]
        invalid_pt = f["top"] if up else f["bot"]
        if (up and entry_pt < entry_ref + atr * 0.5) or (not up and entry_pt > entry_ref - atr * 0.5):
            # Cek apakah ada CHoCH pendukung (HARUS)
            choch_support = False
            if score and score.get("choch_m15", {}).get("bullish_choch" if up else "bearish_choch"):
                choch_support = True
            
            # KOMPROMI 2: Hanya CHoCH yang cukup, Liquidity Sweep TIDAK WAJIB
            if choch_support:
                cands.append({
                    "price": entry_pt,
                    "invalid": invalid_pt,
                    "label": "fvg",
                    "score": 2 + int(f["is_fresh"])
                })

    # Equal Highs/Lows
    eqs = detect_equal_highs_lows(m15, "low" if up else "high", lb=80)
    for eq in eqs[:1]:
        invalid_pt = eq + atr * 0.6 if up else eq - atr * 0.6
        cands.append({
            "price": eq,
            "invalid": invalid_pt,
            "label": "eq",
            "score": 2
        })

    # Market entry sebagai fallback
    if not cands:
        invalid_pt = entry_ref - atr * 1.2 if up else entry_ref + atr * 1.2
        cands.append({
            "price": entry_ref,
            "invalid": invalid_pt,
            "label": "market",
            "score": 1
        })

    return cands

def _build_tp_pool(m15, h1, direction, entry_price, atr, sh15, sl15, sh1, sl1, h4_gate, fib_127, fib_162):
    up = direction == "bull"
    sgn = 1 if up else -1
    pool = []

    # EQH/EQL H1
    eqs_h1 = detect_equal_highs_lows(h1, "high" if up else "low", lb=100)
    for v in eqs_h1:
        if sgn * (v - entry_price) > atr * 0.5:
            pool.append(("eq_h1", v, 1))

    # OB H1
    obs_h1 = detect_order_block(h1, direction, lb=80)
    for z in obs_h1:
        edge = z["top"] if not up else z["bot"]
        if sgn * (edge - entry_price) > atr * 0.5:
            pool.append(("ob_h1", edge, 2))

    # FVG H1 (fresh)
    fvgs_h1 = detect_fvg(h1, direction, lb=60, min_quality=True)
    for f in fvgs_h1:
        if sgn * (f["mid"] - entry_price) > atr * 0.5:
            pool.append(("fvg_h1", f["mid"], 3))

    # Swing H1
    sw_h1 = [h1["high" if up else "low"].iloc[i] for i in (sh1 if up else sl1)]
    for v in sw_h1:
        if sgn * (v - entry_price) > atr * 1.0:
            pool.append(("sw_h1", v, 4))

    # Fib Extension
    if fib_127 is not None and sgn * (fib_127 - entry_price) > atr * 0.5 and h4_gate["confluence"]:
        pool.append(("fib127", fib_127, 5))
        if h4_gate["full_confluence"] and fib_162 is not None and sgn * (fib_162 - entry_price) > atr * 0.5:
            pool.append(("fib162", fib_162, 6))

    return pool

def _select_best_tp(tp_pool, entry_price, risk):
    qualifying = []
    for lbl, v, tier in tp_pool:
        rr_c = abs(v - entry_price) / risk
        if rr_c >= MIN_RR:
            qualifying.append((lbl, v, tier, rr_c))
    if not qualifying: return None, None
    best_lbl, best_v, best_tier, best_rr = min(qualifying, key=lambda x: (x[2], -x[3]))
    return round(best_v, 8), best_lbl

def analyze_setup(df_h1, df_m15, direction, entry_price, score=None, invalid_level=None):
    h1, m15 = build_df(df_h1), build_df(df_m15)
    if h1 is None or m15 is None: return None

    atr = max(m15["atr"].iloc[-1], h1["atr"].iloc[-1] / 4, entry_price * 0.002)
    noise = atr * 0.8

    if invalid_level is None:
        invalid_level = entry_price - atr * 1.2 if direction == "bull" else entry_price + atr * 1.2

    sl_price = invalid_level + (noise if direction == "bear" else -noise)
    risk = abs(sl_price - entry_price)
    risk_floor = max(atr * 0.8, entry_price * 0.003)
    if risk < risk_floor:
        sl_price += (risk_floor - risk) * (1 if direction == "bear" else -1)
        risk = risk_floor
    if risk <= 0: return None

    sh15, sl15 = swing_pts(m15, lb=5)
    sh1, sl1 = swing_pts(h1, lb=5)
    choch_m15 = (score or {}).get("choch_m15", {})
    h4_gate = _h4_confluence(df_h1, direction, choch_m15)
    fib_127, fib_162 = _fib_extension_levels(h1, sh1, sl1, direction)

    tp_pool = _build_tp_pool(m15, h1, direction, entry_price, atr,
                             sh15, sl15, sh1, sl1, h4_gate, fib_127, fib_162)
    tp_price, tp_label = _select_best_tp(tp_pool, entry_price, risk)
    if tp_price is None:
        sgn = 1 if direction == "bull" else -1
        tp_price = entry_price + sgn * risk * MIN_RR
        tp_label = "fallback_rr"

    reward = abs(tp_price - entry_price)
    rr = round(reward / risk, 2)
    if rr < MIN_RR: return None

    return {
        "sl": round(sl_price, 8),
        "tp": round(tp_price, 8),
        "rr": rr,
        "reason": f"SL@{sl_price:.5g}(invalidation) | TP@{tp_price:.5g}({tp_label})",
    }

# ============================================================
# FUNGSI UTAMA
# ============================================================

def full_analyze(df_h1, df_m15, df_d1=None, symbol=None):
    try:
        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            return None

        score = score_direction(df_h1, df_m15, df_d1)
        if score is None: return None

        direction = score["direction"]
        current_price = score["price"]
        atr = score["atr"]
        confidence = score["confidence"]

        # Session filter (dengan kompromi Asia 45)
        bar_ts = score.get("bar_ts")
        session = _get_session(bar_ts)
        min_conf = SESSION_MIN_CONF.get(session, 45)
        if confidence < min_conf:
            return None

        # NY khusus: butuh CHoCH atau Failed Retest
        if session == "NY":
            choch_ok = (
                (direction == "bull" and score["choch_m15"].get("bullish_choch")) or
                (direction == "bear" and score["choch_m15"].get("bearish_choch"))
            )
            fr_ok = (
                (direction == "bull" and score["failed_retest"].get("failed_retest_buy")) or
                (direction == "bear" and score["failed_retest"].get("failed_retest_sell"))
            )
            if not (choch_ok or fr_ok):
                return None

        # Entry candidates
        m15_built = build_df(df_m15)
        cands = _collect_entry_candidates(m15_built, direction, current_price, atr, score=score)
        best = max(cands, key=lambda c: c["score"])
        entry_price, entry_label, invalid_level = best["price"], best["label"], best["invalid"]

        # Filter tambahan: FVG dengan confidence < 60 ditolak
        if entry_label == "fvg" and confidence < 60:
            return None

        # Filter tambahan: Market entry dengan confidence < 65 ditolak
        if entry_label == "market" and confidence < 65:
            return None

        setup = analyze_setup(df_h1, df_m15, direction, entry_price,
                              score=score, invalid_level=invalid_level)
        if setup is None: return None

        if direction == "bull" and current_price >= setup["tp"]: return None
        if direction == "bear" and current_price <= setup["tp"]: return None

        return {
            "symbol": symbol,
            "original_dir": direction,
            "decision": "BUY" if direction == "bull" else "SELL",
            "confidence": confidence,
            "price": current_price,
            "entry": entry_price,
            "entry_label": entry_label,
            "sl": setup["sl"],
            "tp": setup["tp"],
            "rr": setup["rr"],
            "rsi": round(m15_built["rsi"].iloc[-1], 1) if m15_built is not None else 50,
            "struct_h1": score["struct_h1"],
            "d1_bias": score.get("d1_bias", "neutral"),
            "choch_m15": score["choch_m15"],
            "choch_h1": score["choch_h1"],
            "cisd_m15": score["cisd_m15"],
            "failed_retest": score["failed_retest"],
            "session": session,
            "in_killzone": score.get("in_killzone", False),
            "tp_sl_reason": f"Entry@{entry_price:.5g}({entry_label}) | {setup['reason']}",
        }

    except Exception as e:
        log.debug(f"[full_analyze] {symbol}: {e}")
        return None

def get_best_signal(candidates):
    if not candidates: return None
    def _rank(sig):
        label_bonus = 0 if sig.get("entry_label") in ["market", "fvg"] else 3
        return sig["confidence"] + label_bonus + sig["rr"] * 0.5
    return max(candidates, key=_rank)