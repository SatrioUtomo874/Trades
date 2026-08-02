import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ============================================================
# KONFIGURASI
# ============================================================

MIN_RR = 2.0
MAX_RR = 4.0

TRAIL_R_LADDER = [
    (1.0, 0.30),
    (2.0, 0.50),
    (3.0, 0.65),
    (4.0, 0.80),
]

STRUCT_TRAIL_LB = 3
STRUCT_TRAIL_BUF_PCT = 0.0015
STRUCT_TRAIL_LOOKBACK = 60

FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618

# ============================================================
# FUNGSI BANTU
# ============================================================

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
    if df is None or len(df) < 60:
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

# ============================================================
# DETEKSI STRUKTUR SMC
# ============================================================

def is_zone_fresh(df, top, bot, formed_idx, end_idx=None):
    if formed_idx is None or formed_idx + 2 >= len(df):
        return True
    start = formed_idx + 2
    end_idx = end_idx if end_idx is not None else len(df) - 1
    if start >= end_idx:
        return True
    sub = df.iloc[start:end_idx]
    if sub.empty:
        return True
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)

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
    if len(sh) < 2 or len(sl) < 2:
        return result
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
    if len(df) < lb + 1:
        return result
    sub = df.iloc[-lb:]
    closes = sub["close"].values
    opens = sub["open"].values
    n = len(closes)
    last_bull = closes[-1] > opens[-1]
    last_bear = closes[-1] < opens[-1]
    if not (last_bull or last_bear):
        return result
    if last_bull:
        cnt = 0
        for j in range(n - 2, -1, -1):
            if closes[j] < opens[j]:
                cnt += 1
            else:
                break
        if cnt >= 3:
            result["bullish_cisd"] = True
    else:
        cnt = 0
        for j in range(n - 2, -1, -1):
            if closes[j] > opens[j]:
                cnt += 1
            else:
                break
        if cnt >= 3:
            result["bearish_cisd"] = True
    return result

def detect_fvg(df, direction, lb=40):
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    out = []
    for i in range(len(sub) - 2):
        c0, c1, c2 = sub.iloc[i], sub.iloc[i + 1], sub.iloc[i + 2]
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
    return [f for f in out if f["is_fresh"]][-3:] if out else []

def detect_order_block(df, direction, lb=40):
    is_demand = direction == "bull"
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean()
    zones = []
    for i in range(1, len(sub) - 2):
        c = sub.iloc[i]
        nx = sub.iloc[i + 1]
        impulse_body = abs(nx["close"] - nx["open"])
        if impulse_body < avg_body * 1.3:
            continue
        is_match = (c["close"] < c["open"] and nx["close"] > nx["open"]) if is_demand else (c["close"] > c["open"] and nx["close"] < nx["open"])
        if not is_match:
            continue
        top = max(c["open"], c["close"])
        bot = min(c["open"], c["close"])
        df_idx = base_offset + i
        sh, sl = swing_pts(df, lb=5)
        has_fvg = False
        if i + 2 < len(sub):
            c2 = sub.iloc[i + 2]
            if is_demand and c2["low"] > c["high"]:
                has_fvg = True
            if not is_demand and c2["high"] < c["low"]:
                has_fvg = True
        has_bos = detect_break_of_structure(df, sh, sl, direction)
        fresh = is_zone_fresh(df, top, bot, df_idx)
        quality = int(has_fvg) + int(has_bos) + int(fresh)
        if quality >= 2:
            zones.append({
                "top": top,
                "bot": bot,
                "mid": (top + bot) / 2,
                "idx": df_idx,
                "has_fvg": has_fvg,
                "has_bos": has_bos,
                "is_fresh": fresh,
                "quality": quality,
            })
    return [z for z in zones if z["is_fresh"]][-3:] if zones else []

def detect_equal_highs_lows(df, kind="high", lb=60, tol=0.0025):
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
    result = {"failed_retest_sell": False, "failed_retest_buy": False}
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
    if len(sl) >= 2:
        support = df["low"].iloc[sl[-2]]
        touched = P["low"] <= support + atr * 0.5
        bounced = L["close"] > support + atr * 0.3
        bullish_c = L["close"] > L["open"]
        if touched and bounced and bullish_c:
            result["failed_retest_buy"] = True
    return result

# ============================================================
# FIBONACCI & OTE
# ============================================================

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

def is_in_ote(df, direction, sh, sl):
    if len(sh) < 1 or len(sl) < 1:
        return False
    swing_high = df["high"].iloc[sh[-1]]
    swing_low = df["low"].iloc[sl[-1]]
    fib = get_fib_zone(df["close"].iloc[-1], swing_low, swing_high)
    if direction == "bull":
        return 0.62 <= (1 - fib["ratio"]) <= 0.79
    else:
        return 0.62 <= fib["ratio"] <= 0.79

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

# ============================================================
# SCORING – TANPA SESSION BIAS
# ============================================================

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

    # BIAS H1
    bias_bull = bias_bear = 0
    if struct_h1 == "bullish":
        bias_bull += 30
    elif struct_h1 == "bearish":
        bias_bear += 30

    choch_h1 = detect_choch(h1, sh1, sl1)
    if choch_h1["bullish_choch"]:
        bias_bull += 20
    if choch_h1["bearish_choch"]:
        bias_bear += 20

    if L1["ema9"] > L1["ema21"] > L1["ema50"]:
        bias_bull += 10
    elif L1["ema9"] < L1["ema21"] < L1["ema50"]:
        bias_bear += 10

    # D1 BIAS
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
            if struct_d1 == "bearish" or ema_bear_d1:
                d1_bias = "bearish"
            elif struct_d1 == "bullish" or ema_bull_d1:
                d1_bias = "bullish"
    except Exception:
        pass

    if d1_bias == "bullish":
        bias_bull += 15
    elif d1_bias == "bearish":
        bias_bear += 15

    # SETUP M15
    setup_bull = setup_bear = 0

    choch_m15 = detect_choch(m15, sh15, sl15)
    if choch_m15["bullish_choch"]:
        setup_bull += 30
    if choch_m15["bearish_choch"]:
        setup_bear += 30

    cisd_m15 = detect_cisd(m15, lb=8)
    if cisd_m15["bullish_cisd"]:
        setup_bull += 20
    if cisd_m15["bearish_cisd"]:
        setup_bear += 20

    fr = detect_failed_retest(m15, sh15, sl15, atr_val)
    if fr["failed_retest_sell"]:
        setup_bear += 25
    if fr["failed_retest_buy"]:
        setup_bull += 25

    liq_bull = detect_liquidity_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_sweep(m15, sh15, sl15, "bear")
    if liq_bull["type"] == "sweep":
        setup_bull += 15
    if liq_bear["type"] == "sweep":
        setup_bear += 15

    if is_in_ote(m15, "bull", sh15, sl15):
        setup_bull += 10
    if is_in_ote(m15, "bear", sh15, sl15):
        setup_bear += 10

    # Jika bias H1 dan setup M15 bertentangan, setup dikurangi 50%
    if struct_h1 == "bullish" and setup_bear > setup_bull:
        setup_bear = setup_bear * 0.5
    elif struct_h1 == "bearish" and setup_bull > setup_bear:
        setup_bull = setup_bull * 0.5

    total_bull = bias_bull + setup_bull
    total_bear = bias_bear + setup_bear

    direction = "bull" if total_bull >= total_bear else "bear"
    raw = total_bull if direction == "bull" else total_bear
    conf = min(int(raw / 280 * 100), 99)

    return {
        "direction": direction,
        "confidence": conf,
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
        "sh15": sh15,
        "sl15": sl15,
    }

# ============================================================
# ENTRY CANDIDATES
# ============================================================

def _collect_entry_candidates(m15, direction, entry_ref, atr, score=None):
    up = direction == "bull"
    cands = []
    max_dist = atr * 0.3

    # OB
    obs = detect_order_block(m15, direction, lb=40)
    for z in obs:
        entry_pt = z["top"] if not up else z["bot"]
        invalid_pt = z["bot"] if not up else z["top"]
        if (up and entry_pt < entry_ref + max_dist) or (not up and entry_pt > entry_ref - max_dist):
            cands.append({
                "price": entry_pt,
                "invalid": invalid_pt,
                "label": "ob",
                "score": 5 + z["quality"]
            })

    # EQ
    eqs = detect_equal_highs_lows(m15, "low" if up else "high", lb=80)
    for eq in eqs[:1]:
        invalid_pt = eq + atr * 0.6 if up else eq - atr * 0.6
        cands.append({
            "price": eq,
            "invalid": invalid_pt,
            "label": "eq",
            "score": 4
        })

    # FVG (fresh + CISD)
    fvgs = detect_fvg(m15, direction, lb=40)
    for f in fvgs:
        entry_pt = f["mid"]
        invalid_pt = f["top"] if up else f["bot"]
        if (up and entry_pt < entry_ref + max_dist) or (not up and entry_pt > entry_ref - max_dist):
            cisd_support = False
            if score and score.get("cisd_m15", {}).get("bullish_cisd" if up else "bearish_cisd"):
                cisd_support = True
            if cisd_support and f["is_fresh"]:
                cands.append({
                    "price": entry_pt,
                    "invalid": invalid_pt,
                    "label": "fvg",
                    "score": 3
                })

    # Market fallback
    if not cands:
        invalid_pt = entry_ref - atr * 1.2 if up else entry_ref + atr * 1.2
        cands.append({
            "price": entry_ref,
            "invalid": invalid_pt,
            "label": "market",
            "score": 1
        })

    return cands

# ============================================================
# TP POOL
# ============================================================

def _build_tp_pool(m15, h1, direction, entry_price, atr, sh15, sl15, sh1, sl1):
    up = direction == "bull"
    sgn = 1 if up else -1
    pool = []

    eqs_h1 = detect_equal_highs_lows(h1, "high" if up else "low", lb=100)
    for v in eqs_h1:
        if sgn * (v - entry_price) > atr * 0.5:
            pool.append(("eq_h1", v, 1))

    obs_h1 = detect_order_block(h1, direction, lb=80)
    for z in obs_h1:
        edge = z["top"] if not up else z["bot"]
        if sgn * (edge - entry_price) > atr * 0.5:
            pool.append(("ob_h1", edge, 2))

    fvgs_h1 = detect_fvg(h1, direction, lb=60)
    for f in fvgs_h1:
        if sgn * (f["mid"] - entry_price) > atr * 0.5:
            pool.append(("fvg_h1", f["mid"], 3))

    sw_h1 = [h1["high" if up else "low"].iloc[i] for i in (sh1 if up else sl1)]
    for v in sw_h1:
        if sgn * (v - entry_price) > atr * 1.0:
            pool.append(("sw_h1", v, 4))

    fib_127, fib_162 = _fib_extension_levels(h1, sh1, sl1, direction)
    if fib_127 is not None and sgn * (fib_127 - entry_price) > atr * 0.5:
        pool.append(("fib127", fib_127, 5))
    if fib_162 is not None and sgn * (fib_162 - entry_price) > atr * 0.5:
        pool.append(("fib162", fib_162, 6))

    pool.sort(key=lambda x: abs(x[1] - entry_price))
    return pool

def _select_best_tp(tp_pool, entry_price, risk):
    if not tp_pool:
        return None, None

    qualified = []
    too_far = []
    for lbl, v, tier in tp_pool:
        rr_c = abs(v - entry_price) / risk
        if rr_c >= MIN_RR:
            if rr_c <= MAX_RR:
                qualified.append((lbl, v, tier, rr_c))
            else:
                too_far.append((lbl, v, tier, rr_c))

    if qualified:
        best = min(qualified, key=lambda x: (x[2], x[3]))
        return round(best[1], 8), best[0]

    if too_far:
        best = min(too_far, key=lambda x: x[3])
        lbl, v, tier, rr = best
        sgn = 1 if v > entry_price else -1
        capped_price = entry_price + sgn * risk * MAX_RR
        return round(capped_price, 8), lbl + "_capped"

    return None, None

# ============================================================
# SETUP: SL & TP (DIPERBAIKI)
# ============================================================

def analyze_setup(df_h1, df_m15, direction, entry_price, invalid_level, score=None):
    h1, m15 = build_df(df_h1), build_df(df_m15)
    if h1 is None or m15 is None:
        return None

    atr = max(m15["atr"].iloc[-1], h1["atr"].iloc[-1] / 4, entry_price * 0.002)
    noise = atr * 0.5

    if invalid_level is None:
        invalid_level = entry_price - atr * 1.2 if direction == "bull" else entry_price + atr * 1.2

    sl_price = invalid_level + (noise if direction == "bear" else -noise)
    risk = abs(sl_price - entry_price)
    risk_floor = max(atr * 0.6, entry_price * 0.002)
    if risk < risk_floor:
        sl_price += (risk_floor - risk) * (1 if direction == "bear" else -1)
        risk = risk_floor
    if risk <= 0:
        return None

    # Ambil sh15/sl15 dari score jika ada
    if score is not None:
        sh15 = score.get("sh15", [])
        sl15 = score.get("sl15", [])
    else:
        sh15, sl15 = [], []

    sh1, sl1 = swing_pts(h1, lb=5)
    tp_pool = _build_tp_pool(m15, h1, direction, entry_price, atr, sh15, sl15, sh1, sl1)
    tp_price, tp_label = _select_best_tp(tp_pool, entry_price, risk)

    if tp_price is None:
        sgn = 1 if direction == "bull" else -1
        tp_price = entry_price + sgn * risk * MIN_RR
        tp_label = "fallback_rr"

    reward = abs(tp_price - entry_price)
    rr = round(reward / risk, 2)

    if rr < MIN_RR:
        return None

    if rr > MAX_RR:
        sgn = 1 if direction == "bull" else -1
        tp_price = entry_price + sgn * risk * MAX_RR
        tp_label = tp_label + "_capped"
        rr = MAX_RR

    return {
        "sl": round(sl_price, 8),
        "tp": round(tp_price, 8),
        "rr": rr,
        "reason": f"SL@{sl_price:.5g}(invalidation) | TP@{tp_price:.5g}({tp_label})",
    }

# ============================================================
# FUNGSI UTAMA – DIPANGGIL main.py
# ============================================================

def full_analyze(df_h1, df_m15, df_d1=None, symbol=None):
    try:
        # ----- LOG DATA -----
        if symbol:
            h1_len = len(df_h1) if df_h1 is not None else 0
            m15_len = len(df_m15) if df_m15 is not None else 0
            log.info(f"[DEBUG] {symbol}: h1={h1_len}, m15={m15_len}")

        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            if symbol:
                log.warning(f"[DEBUG] {symbol}: data kosong, skip")
            return None

        score = score_direction(df_h1, df_m15, df_d1)
        if score is None:
            if symbol:
                log.warning(f"[DEBUG] {symbol}: score_direction None")
            return None

        direction = score["direction"]
        current_price = score["price"]
        atr = score["atr"]
        confidence = score["confidence"]

        if symbol:
            log.info(f"[DEBUG] {symbol}: dir={direction}, conf={confidence}, atr={atr:.6f}, price={current_price:.6f}")

        # ----- FILTER VOLATILITAS (DINONAKTIFKAN SEMENTARA) -----
        # if atr / current_price < 0.003:
        #     if symbol: log.debug(f"[DEBUG] {symbol}: ATR terlalu kecil, skip")
        #     return None

        # ----- ENTRY CANDIDATE -----
        m15_built = build_df(df_m15)
        cands = _collect_entry_candidates(m15_built, direction, current_price, atr, score=score)
        if not cands:
            if symbol:
                log.warning(f"[DEBUG] {symbol}: tidak ada entry candidate")
            return None

        best = max(cands, key=lambda c: c["score"])
        entry_price, entry_label, invalid_level = best["price"], best["label"], best["invalid"]

        if symbol:
            log.info(f"[DEBUG] {symbol}: entry_label={entry_label}, entry={entry_price:.6f}, score={best['score']}")

        # Tolak market entry jika confidence < 65
        if entry_label == "market" and confidence < 65:
            if symbol:
                log.info(f"[DEBUG] {symbol}: market entry ditolak (conf<65)")
            return None

        # ----- SETUP SL/TP (DI SINI BUG SEBELUMNYA) -----
        setup = analyze_setup(df_h1, df_m15, direction, entry_price, invalid_level, score=score)
        if setup is None:
            if symbol:
                log.warning(f"[DEBUG] {symbol}: analyze_setup None")
            return None

        if symbol:
            log.info(f"[DEBUG] {symbol}: sl={setup['sl']:.6f}, tp={setup['tp']:.6f}, rr={setup['rr']}")

        # Cegah TP sudah lewat
        if direction == "bull" and current_price >= setup["tp"]:
            if symbol:
                log.info(f"[DEBUG] {symbol}: TP sudah lewat (current={current_price:.6f} >= tp={setup['tp']:.6f})")
            return None
        if direction == "bear" and current_price <= setup["tp"]:
            if symbol:
                log.info(f"[DEBUG] {symbol}: TP sudah lewat (current={current_price:.6f} <= tp={setup['tp']:.6f})")
            return None

        # ----- RETURN SIGNAL -----
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
            "tp_sl_reason": f"Entry@{entry_price:.5g}({entry_label}) | {setup['reason']}",
        }

    except Exception as e:
        if symbol:
            log.error(f"[full_analyze] {symbol}: {e}")
        return None

def get_best_signal(candidates):
    if not candidates:
        return None
    def _rank(sig):
        label_bonus = 0 if sig.get("entry_label") in ["market", "fvg"] else 3
        return sig["confidence"] + label_bonus + sig["rr"] * 0.5
    return max(candidates, key=_rank)