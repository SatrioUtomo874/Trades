"""strategy_logic.py — SMC/price-action decision engine v14.

Satu jalur reasoning: market context -> direction -> setup -> location -> risk -> confidence.
Position management memakai state machine terpisah dan path metrics MFE/MAE/giveback.
Kompatibel dengan dispatcher main.py dan tidak membutuhkan dependency di luar pandas/numpy.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Public constants imported by main.py.
MIN_RR = 2.0
MAX_RR = None
FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618
TRAIL_R_LADDER = []
STRUCT_TRAIL_LB = 3
STRUCT_TRAIL_BUF_PCT = 0.0025
STRUCT_TRAIL_LOOKBACK = 60
TRAIL_ENGINE_VERSION = "14.0-single-path-brain"
TRAIL_EXECUTION_BUFFER_ATR = 0.08
TRAIL_MIN_MARKET_GAP_ATR = 0.35
MAIN_ENTRY_MAX_ATR = 1.50
CONFIDENCE_MODEL_VERSION = "14.0-monotonic-quality"

# Entry engine.
MIN_DISPLACEMENT_ATR = 0.30
ENTRY_LOOKBACK = 20
SWING_LB = 3
POI_LOOKBACK = 80
HTF_POI_LOOKBACK = 90
ENTRY_MAX_RISK_ATR = 2.20
ENTRY_MIN_RISK_ATR = 0.55
ENTRY_MIN_RISK_PCT = 0.08
ENTRY_MAX_RISK_PCT = 3.50
ENTRY_LOCATION_SOFT_HIGH = 0.82
ENTRY_LOCATION_SOFT_LOW = 0.18
ENTRY_PREFERRED_BUY = 0.55
ENTRY_PREFERRED_SELL = 0.45

# Trail engine. Values are guardrails, not profit-lock ladders.
TRAIL_ARM_R = 0.80
TRAIL_GIVEBACK_WARN = 0.30
TRAIL_GIVEBACK_STRONG = 0.50
TRAIL_GIVEBACK_CRITICAL = 0.70
TRAIL_MFE_STRONG = 1.00
TRAIL_MFE_EXTENDED = 1.50
TRAIL_MFE_DEEP = 2.00
TRAIL_LOCK_WARN_R = 0.08
TRAIL_LOCK_STRONG_R = 0.30
TRAIL_LOCK_CRITICAL_R = 0.60
TRAIL_MIN_UPDATE_R = 0.04
TRAIL_MAX_CHURN = 4
TRAIL_STRUCT_BUFFER_ATR = 0.36
TRAIL_RETRACE_BUFFER_ATR = 0.25
TRAIL_REVERSAL_BODY_ATR = 0.85
TRAIL_COUNTER_BODY_ATR = 0.55
TRAIL_VOLUME_EXHAUSTION = 0.72
TRAIL_VOLUME_COUNTER = 1.20
TRAIL_PEAK_LOOKBACK = 40


def _num(v, default=None):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def _clip(v, lo=0.0, hi=100.0):
    return float(np.clip(float(v), lo, hi))


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr_fn(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def _closed_candles(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    idx = out.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    boundary = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
    if idx[-1] < boundary:
        return out
    return out.loc[idx < boundary].copy()


def build_df(df: pd.DataFrame, interval_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 60:
        return None
    x = df.copy()
    if interval_minutes:
        x = _closed_candles(x, interval_minutes)
    if len(x) < 60:
        return None
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["ema9"] = ema(x["close"], 9)
    x["ema21"] = ema(x["close"], 21)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200) if len(x) >= 200 else ema(x["close"], 50)
    x["rsi"] = rsi(x["close"])
    x["atr"] = atr_fn(x)
    x["vol_sma"] = x["volume"].rolling(20).mean()
    return x.dropna()


def swing_pts(df: pd.DataFrame, lb: int = 5):
    if df is None or df.empty or len(df) < lb * 2 + 3:
        return [], []
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    sh, sl = [], []
    for i in range(lb, len(df) - lb):
        wh = h[i - lb : i + lb + 1]
        wl = l[i - lb : i + lb + 1]
        if h[i] >= wh.max():
            sh.append(i)
        if l[i] <= wl.min():
            sl.append(i)
    return sh, sl


def _market_structure(df, sh, sl):
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


mkt_struct = _market_structure


def _direction_score(df_h1, df_m15, df_d1=None, market_context=None, df_btc_h1=None):
    h1 = build_df(df_h1, 60)
    m15 = build_df(df_m15, 15)
    if h1 is None or m15 is None:
        return None
    d1 = None
    if df_d1 is not None and len(df_d1) >= 60:
        d1 = build_df(df_d1, 1440)
    if d1 is None and isinstance(h1.index, pd.DatetimeIndex):
        d1 = build_df(
            h1.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        )
    sh1, sl1 = swing_pts(h1, 5)
    sh15, sl15 = swing_pts(m15, 5)
    struct_h1 = _market_structure(h1, sh1, sl1)
    struct_m15 = _market_structure(m15, sh15, sl15)
    d1_bias = "neutral"
    if d1 is not None and len(d1) >= 10:
        shd, sld = swing_pts(d1, 3)
        sd1 = _market_structure(d1, shd, sld)
        bull = sd1 == "bullish" or bool(d1["ema9"].iloc[-1] > d1["ema21"].iloc[-1] > d1["ema50"].iloc[-1])
        bear = sd1 == "bearish" or bool(d1["ema9"].iloc[-1] < d1["ema21"].iloc[-1] < d1["ema50"].iloc[-1])
        d1_bias = "bullish" if bull and not bear else "bearish" if bear and not bull else "neutral"

    ema_bull = h1["ema9"].iloc[-1] > h1["ema21"].iloc[-1] > h1["ema50"].iloc[-1]
    ema_bear = h1["ema9"].iloc[-1] < h1["ema21"].iloc[-1] < h1["ema50"].iloc[-1]
    fast = (float(m15["close"].iloc[-1]) - float(m15["close"].iloc[-4])) / max(float(m15["atr"].iloc[-1]), 1e-12)
    slow = (float(m15["close"].iloc[-1]) - float(m15["close"].iloc[-9])) / max(float(m15["atr"].iloc[-1]), 1e-12)

    bull = 35.0
    bear = 35.0
    if d1_bias == "bullish": bull += 20
    if d1_bias == "bearish": bear += 20
    if struct_h1 == "bullish": bull += 24
    elif struct_h1 == "bearish": bear += 24
    else:
        bull += 6; bear += 6
    if ema_bull: bull += 10
    if ema_bear: bear += 10
    if fast > 0: bull += min(8, fast * 4)
    if fast < 0: bear += min(8, -fast * 4)
    if slow > 0: bull += min(8, slow * 2)
    if slow < 0: bear += min(8, -slow * 2)
    if struct_m15 == "bullish": bull += 8
    if struct_m15 == "bearish": bear += 8

    mc = market_context if isinstance(market_context, dict) else {}
    breadth = _num(mc.get("bullish_breadth_pct"), None)
    if breadth is not None:
        if breadth >= 70: bull += 6
        elif breadth <= 35: bear += 6
        if breadth >= 80: bull += 2
        elif breadth <= 20: bear += 2
    rs1 = _num(mc.get("relative_strength_1h_pct"), None)
    if rs1 is not None:
        if rs1 > 0.40: bull += 5
        elif rs1 < -0.40: bear += 5
    rv = _num(mc.get("relative_volume"), None)
    if rv is not None and rv >= 1.2:
        if fast > 0: bull += 4
        elif fast < 0: bear += 4

    macro = "unknown"
    if df_btc_h1 is not None:
        bh1 = build_df(df_btc_h1, 60)
        if bh1 is not None:
            macro = "bullish" if bh1["ema9"].iloc[-1] > bh1["ema21"].iloc[-1] > bh1["ema50"].iloc[-1] else "bearish" if bh1["ema9"].iloc[-1] < bh1["ema21"].iloc[-1] < bh1["ema50"].iloc[-1] else "ranging"
    elif str(mc.get("market_regime", "")).lower().startswith("bull"):
        macro = "bullish"
    elif str(mc.get("market_regime", "")).lower().startswith("bear"):
        macro = "bearish"
    if macro == "bullish": bull += 5; bear *= 0.88
    elif macro == "bearish": bear += 5; bull *= 0.88

    direction = "bull" if bull >= bear else "bear"
    edge = abs(bull - bear)
    direction_quality = _clip(50 + edge * 1.25)
    return {
        "direction": direction,
        "bull_score": round(bull, 2),
        "bear_score": round(bear, 2),
        "direction_edge": round(edge, 2),
        "direction_quality": round(direction_quality, 2),
        "struct_h1": struct_h1,
        "m15_struct": struct_m15,
        "d1_bias": d1_bias,
        "macro_bias": macro,
        "atr": max(float(m15["atr"].iloc[-1]), float(h1["atr"].iloc[-1]) / 4),
        "price": float(m15["close"].iloc[-1]),
        "sh1": sh1, "sl1": sl1, "sh15": sh15, "sl15": sl15,
        "h1": h1, "m15": m15, "d1": d1,
    }


def score_direction(df_h1, df_m15, df_d1=None, df_btc_h1=None):
    return _direction_score(df_h1, df_m15, df_d1, None, df_btc_h1)


def _zone_overlap(price, top, bot):
    return float(bot) <= float(price) <= float(top)


def _fresh_zone(df, formed_idx, top, bot, direction):
    if formed_idx >= len(df) - 2:
        return True
    sub = df.iloc[formed_idx + 2 :]
    if direction == "bull":
        return not bool((sub["close"] < bot).any())
    return not bool((sub["close"] > top).any())


def _find_order_blocks(df, direction, lookback=POI_LOOKBACK):
    sub = df.tail(lookback).reset_index(drop=True)
    base = len(df) - len(sub)
    avg_body = max(float((sub["close"] - sub["open"]).abs().median()), 1e-12)
    out = []
    for i in range(1, len(sub) - 4):
        c, nxt = sub.iloc[i], sub.iloc[i + 1]
        if direction == "bull" and not (c["close"] < c["open"] and nxt["close"] > nxt["open"]):
            continue
        if direction == "bear" and not (c["close"] > c["open"] and nxt["close"] < nxt["open"]):
            continue
        body = abs(float(nxt["close"] - nxt["open"]))
        if body < avg_body * 1.25:
            continue
        top = float(max(c["open"], c["close"]))
        bot = float(min(c["open"], c["close"]))
        idx = base + i
        if not _fresh_zone(df, idx, top, bot, direction):
            continue
        score = 0.0
        score += min(30.0, body / avg_body * 10.0)
        if i + 2 < len(sub):
            c2 = sub.iloc[i + 2]
            if direction == "bull" and c2["low"] > c["high"]: score += 20
            if direction == "bear" and c2["high"] < c["low"]: score += 20
        if idx >= len(df) - 20: score += 15
        mid = (top + bot) / 2
        if direction == "bull" and mid <= float(df["close"].iloc[-1]): score += 10
        if direction == "bear" and mid >= float(df["close"].iloc[-1]): score += 10
        out.append({"top": top, "bot": bot, "mid": mid, "idx": idx, "score": score, "kind": "ob"})
    out.sort(key=lambda z: (-z["score"], -z["idx"]))
    return out[:4]


def _find_fvg(df, direction, lookback=POI_LOOKBACK):
    sub = df.tail(lookback)
    base = len(df) - len(sub)
    out = []
    for i in range(len(sub) - 2):
        a, c = sub.iloc[i], sub.iloc[i + 2]
        if direction == "bull" and c["low"] > a["high"]:
            top, bot = float(c["low"]), float(a["high"])
        elif direction == "bear" and c["high"] < a["low"]:
            top, bot = float(a["low"]), float(c["high"])
        else:
            continue
        idx = base + i + 2
        if not _fresh_zone(df, idx, top, bot, direction):
            continue
        width = top - bot
        out.append({"top": top, "bot": bot, "mid": (top + bot) / 2, "idx": idx, "score": 50 + min(30, width / max(float(df["atr"].iloc[-1]), 1e-12) * 20), "kind": "fvg"})
    out.sort(key=lambda z: (-z["score"], -z["idx"]))
    return out[:4]


def _find_equal_levels(df, kind="low", lookback=80, tol=0.0025):
    sub = df.tail(lookback)
    vals = sub[kind].to_numpy(dtype=float)
    clusters = []
    for i in range(len(vals)):
        grp = [vals[i]]
        for j in range(i + 1, len(vals)):
            if abs(vals[i] - vals[j]) / max(abs(vals[i]), 1e-12) <= tol:
                grp.append(vals[j])
        if len(grp) >= 2:
            clusters.append(float(np.mean(grp)))
    return sorted(set(round(x, 10) for x in clusters))


def _sweep(df, direction, sh, sl):
    if direction == "bull" and sl:
        level = float(df["low"].iloc[sl[-1]])
        if float(df["low"].iloc[-1]) < level and float(df["close"].iloc[-1]) > level:
            depth = (level - float(df["low"].iloc[-1])) / max(float(df["atr"].iloc[-1]), 1e-12)
            return {"type": "sweep", "level": level, "strength": _clip(45 + depth * 25)}
    if direction == "bear" and sh:
        level = float(df["high"].iloc[sh[-1]])
        if float(df["high"].iloc[-1]) > level and float(df["close"].iloc[-1]) < level:
            depth = (float(df["high"].iloc[-1]) - level) / max(float(df["atr"].iloc[-1]), 1e-12)
            return {"type": "sweep", "level": level, "strength": _clip(45 + depth * 25)}
    return {"type": "none", "level": None, "strength": 0}


def _displacement(df, direction):
    last = df.iloc[-1]
    body = abs(float(last["close"] - last["open"]))
    atr = max(float(last["atr"]), 1e-12)
    recent = df["close"].iloc[-5:-1]
    med = max(float((df["close"] - df["open"]).abs().iloc[-8:-1].median()), atr * 0.20)
    if direction == "bull":
        ok = float(last["close"]) > float(last["open"]) and float(last["close"]) > float(recent.max()) and body >= max(atr * MIN_DISPLACEMENT_ATR, med)
    else:
        ok = float(last["close"]) < float(last["open"]) and float(last["close"]) < float(recent.min()) and body >= max(atr * MIN_DISPLACEMENT_ATR, med)
    return bool(ok), body / atr


def _pullback_quality(df, direction, atr):
    n = 10
    if len(df) < n + 4:
        return {"score": 45, "state": "unknown", "depth_atr": 0.0, "efficiency": 0.0}
    sub = df.tail(n + 1)
    move = float(sub["close"].iloc[-1] - sub["close"].iloc[0])
    if direction == "bear": move = -move
    gross = float(sub["close"].diff().abs().sum())
    eff = abs(move) / max(gross, 1e-12)
    adverse = min(float((sub["low"] - sub["close"].iloc[0]).min()), 0.0) if direction == "bull" else min(float((sub["close"].iloc[0] - sub["high"]).min()), 0.0)
    depth = abs(adverse) / max(atr, 1e-12)
    score = 55.0
    if 0.25 <= depth <= 1.50: score += 20
    elif depth > 2.0: score -= 20
    if eff >= 0.45: score += 15
    elif eff < 0.22: score -= 12
    if direction == "bull" and sub["close"].iloc[-1] > sub["close"].iloc[-2]: score += 8
    if direction == "bear" and sub["close"].iloc[-1] < sub["close"].iloc[-2]: score += 8
    state = "clean" if score >= 70 else "mixed" if score >= 45 else "damaged"
    return {"score": _clip(score), "state": state, "depth_atr": round(depth, 3), "efficiency": round(eff, 3)}


def _location_quality(df, direction, price, atr):
    look = df.tail(ENTRY_LOOKBACK)
    lo, hi = float(look["low"].min()), float(look["high"].max())
    pos = (price - lo) / max(hi - lo, 1e-12)
    score = 50.0
    if direction == "bull":
        score += 28 if pos <= ENTRY_PREFERRED_BUY else -24 if pos >= ENTRY_LOCATION_SOFT_HIGH else 8
    else:
        score += 28 if pos >= ENTRY_PREFERRED_SELL else -24 if pos <= ENTRY_LOCATION_SOFT_LOW else 8
    near_high = abs(hi - price) / max(atr, 1e-12)
    near_low = abs(price - lo) / max(atr, 1e-12)
    if direction == "bull" and near_high < 0.45: score -= 10
    if direction == "bear" and near_low < 0.45: score -= 10
    return {"score": _clip(score), "range_position": round(pos, 3), "range_low": lo, "range_high": hi}


def _candidate_levels(m15, h1, direction, current_price, atr):
    zones = _find_order_blocks(m15, direction) + _find_fvg(m15, direction)
    zones += _find_order_blocks(h1, direction, HTF_POI_LOOKBACK) + _find_fvg(h1, direction, HTF_POI_LOOKBACK)
    out = []
    seen = set()
    for z in zones:
        p = float(z["mid"])
        if direction == "bull" and p > current_price * 1.006:
            continue
        if direction == "bear" and p < current_price * 0.994:
            continue
        if abs(current_price - p) > atr * MAIN_ENTRY_MAX_ATR:
            continue
        key = round(p, 10)
        if key in seen:
            continue
        seen.add(key)
        out.append(z)
    if not out:
        out = [{"mid": current_price, "top": current_price, "bot": current_price, "score": 35, "kind": "market", "idx": len(m15)-1}]
    return sorted(out, key=lambda z: z.get("score", 0), reverse=True)[:6]


def _compute_geometry(m15, h1, direction, entry, atr):
    sh, sl = swing_pts(m15, SWING_LB)
    h1sh, h1sl = swing_pts(h1, 5)
    if direction == "bull":
        anchors = [float(m15["low"].iloc[sl[-1]]) for _ in [0] if sl]
        if h1sl: anchors.append(float(h1["low"].iloc[h1sl[-1]]))
        anchor = min([x for x in anchors if x < entry] or [entry - atr * 1.0])
        sl = anchor - atr * 0.18
        risk = entry - sl
    else:
        anchors = [float(m15["high"].iloc[sh[-1]]) for _ in [0] if sh]
        if h1sh: anchors.append(float(h1["high"].iloc[h1sh[-1]]))
        anchor = max([x for x in anchors if x > entry] or [entry + atr * 1.0])
        sl = anchor + atr * 0.18
        risk = sl - entry
    risk = float(risk)
    risk_pct = risk / max(entry, 1e-12) * 100.0
    risk_atr = risk / max(atr, 1e-12)
    if risk <= 0 or risk_atr < ENTRY_MIN_RISK_ATR or risk_atr > ENTRY_MAX_RISK_ATR or risk_pct < ENTRY_MIN_RISK_PCT or risk_pct > ENTRY_MAX_RISK_PCT:
        return None
    return float(sl), risk, sh, sl, h1sh, h1sl, risk_atr, risk_pct


def _tp_targets(m15, h1, direction, entry, risk):
    vals = []
    if direction == "bull":
        sh, _ = swing_pts(h1, 5)
        for i in reversed(sh[-6:]):
            x = float(h1["high"].iloc[i])
            if x > entry: vals.append((x, "h1_swing"))
    else:
        _, sl = swing_pts(h1, 5)
        for i in reversed(sl[-6:]):
            x = float(h1["low"].iloc[i])
            if x < entry: vals.append((x, "h1_swing"))
    atr = max(float(m15["atr"].iloc[-1]), 1e-12)
    ext1 = entry + risk * (1 + FIB_EXT_1) if direction == "bull" else entry - risk * (1 + FIB_EXT_1)
    ext2 = entry + risk * (1 + FIB_EXT_2) if direction == "bull" else entry - risk * (1 + FIB_EXT_2)
    vals += [(ext1, "fib_1.272"), (ext2, "fib_1.618")]
    if direction == "bull": vals = [x for x in vals if x[0] > entry + risk * 1.95]
    else: vals = [x for x in vals if x[0] < entry - risk * 1.95]
    if not vals:
        return None
    vals.sort(key=lambda x: abs(((x[0]-entry) if direction == "bull" else (entry-x[0])) / risk - 2.5))
    tp, label = vals[0]
    rr = ((tp-entry) if direction == "bull" else (entry-tp)) / risk
    return float(tp), label, float(rr), atr


def _context_quality(market_context, direction):
    if not isinstance(market_context, dict) or not market_context:
        return {"score": 50.0, "reasons": ["market_context_unavailable"], "conflicts": []}
    score = 50.0
    reasons, conflicts = [], []
    breadth = _num(market_context.get("bullish_breadth_pct"), None)
    if breadth is not None:
        aligned = breadth >= 60 if direction == "bull" else breadth <= 40
        opposed = breadth <= 40 if direction == "bull" else breadth >= 60
        if aligned: score += 15; reasons.append("breadth_aligned")
        elif opposed: score -= 15; conflicts.append("breadth_opposed")
    rs = _num(market_context.get("relative_strength_1h_pct"), None)
    if rs is not None:
        if (rs >= 0.35 and direction == "bull") or (rs <= -0.35 and direction == "bear"):
            score += 15; reasons.append("relative_strength_aligned")
        elif (rs <= -0.35 and direction == "bull") or (rs >= 0.35 and direction == "bear"):
            score -= 14; conflicts.append("relative_strength_opposed")
    rv = _num(market_context.get("relative_volume"), None)
    if rv is not None:
        if rv >= 1.2: score += 8; reasons.append("volume_participation")
        elif rv <= 0.55: score -= 6; conflicts.append("low_participation")
    regime = str(market_context.get("market_regime") or market_context.get("chart_regime") or "").lower()
    if "expansion" in regime or "trend" in regime:
        reasons.append("trend_regime")
        score += 5
    elif "range" in regime or "compression" in regime:
        reasons.append("range_regime")
        score -= 2
    return {"score": _clip(score), "reasons": reasons, "conflicts": conflicts}


def _archetype(direction, sweep, poi_kind, displacement, pullback, htf_overlap, market_regime):
    if sweep["type"] == "sweep" and displacement:
        return "LIQUIDITY_SWEEP_RECLAIM"
    if htf_overlap and pullback["score"] >= 65 and poi_kind == "ob":
        return "HTF_OB_PULLBACK_CONTINUATION"
    if poi_kind == "fvg" and displacement:
        return "FVG_DISPLACEMENT_RETEST"
    if "range" in str(market_regime).lower() and sweep["type"] == "sweep":
        return "RANGE_LIQUIDITY_REVERSAL"
    return "STRUCTURE_CONTINUATION" if direction in ("bull", "bear") else "CONTEXTUAL_SETUP"


def _confidence(components, contradictions):
    quality = (
        components["direction"] * 0.27
        + components["setup"] * 0.28
        + components["location"] * 0.15
        + components["risk"] * 0.12
        + components["context"] * 0.18
    )
    penalty = min(20.0, contradictions * 6.0)
    quality = _clip(quality - penalty)
    confidence = int(round(20.0 + 0.70 * quality))
    return int(np.clip(confidence, 20, 90)), round(quality, 2), round(penalty, 2)


def _history_context(history, direction, entry_label, archetype):
    rows = [r for r in (history or []) if isinstance(r, dict)]
    if len(rows) < 8:
        return {"samples": len(rows), "adjustment": 0.0, "note": "history_insufficient"}
    side = "BUY" if direction == "bull" else "SELL"
    matched = [r for r in rows if str(r.get("decision", "")).upper() == side]
    matched2 = [r for r in matched if str(r.get("entry_label", "")) == entry_label]
    matched = matched2 if len(matched2) >= 6 else matched
    if not matched:
        return {"samples": len(rows), "adjustment": 0.0, "note": "no_match"}
    wins = np.array([1.0 if float(r.get("pnl_usd", 0) or 0) > 0 else 0.0 for r in matched], dtype=float)
    base = float(wins.mean())
    shrink = min(0.55, len(matched) / 40.0)
    adjustment = (base - 0.50) * 8.0 * shrink
    return {"samples": len(rows), "matched": len(matched), "win_rate": round(base, 3), "adjustment": round(adjustment, 2), "note": "soft_history_only"}


def full_analyze(df_h1, df_m15, df_d1=None, symbol=None, df_btc_h1=None, trade_history=None, market_context=None):
    try:
        ctx = _direction_score(df_h1, df_m15, df_d1, market_context, df_btc_h1)
        if ctx is None:
            return None
        direction = ctx["direction"]
        h1, m15 = ctx["h1"], ctx["m15"]
        price, atr = ctx["price"], ctx["atr"]
        sweep = _sweep(m15, direction, ctx["sh15"], ctx["sl15"])
        displacement, displacement_atr = _displacement(m15, direction)
        pullback = _pullback_quality(m15, direction, atr)
        location = _location_quality(m15, direction, price, atr)
        candidates = _candidate_levels(m15, h1, direction, price, atr)
        context_q = _context_quality(market_context, direction)
        evaluated = []
        htf_zones = _find_order_blocks(h1, direction, HTF_POI_LOOKBACK) + _find_fvg(h1, direction, HTF_POI_LOOKBACK)
        for cand in candidates:
            entry = float(cand["mid"])
            if abs(price - entry) > atr * MAIN_ENTRY_MAX_ATR:
                continue
            geo = _compute_geometry(m15, h1, direction, entry, atr)
            if geo is None:
                continue
            sl_price, risk, *_rest = geo
            tp = _tp_targets(m15, h1, direction, entry, risk)
            if tp is None:
                continue
            tp_price, tp_label, rr, _ = tp
            if rr < MIN_RR:
                continue
            htf_overlap = any(_zone_overlap(entry, z["top"], z["bot"]) for z in htf_zones[:6])
            poi_quality = _clip(40 + float(cand.get("score", 0)) * 0.55 + (12 if htf_overlap else 0) + (10 if sweep["type"] == "sweep" else 0))
            if displacement:
                setup_score = _clip(0.45 * poi_quality + 0.28 * pullback["score"] + 0.17 * min(100, displacement_atr * 35) + 0.10 * sweep["strength"])
            else:
                setup_score = _clip(0.55 * poi_quality + 0.30 * pullback["score"] + 0.15 * sweep["strength"])
            risk_atr = risk / max(atr, 1e-12)
            risk_score = 100 - min(65, abs(risk_atr - 1.0) * 32) - (15 if risk_atr < ENTRY_MIN_RISK_ATR * 1.15 else 0)
            contradictions = 0
            reasons = []
            if ctx["struct_h1"] == "bullish" and direction == "bear": contradictions += 1; reasons.append("h1_conflict")
            if ctx["struct_h1"] == "bearish" and direction == "bull": contradictions += 1; reasons.append("h1_conflict")
            if pullback["state"] == "damaged": contradictions += 1; reasons.append("damaged_pullback")
            if location["score"] < 35: contradictions += 1; reasons.append("poor_location")
            if context_q["conflicts"]: contradictions += min(2, len(context_q["conflicts"])); reasons.extend(context_q["conflicts"][:2])
            if not displacement: reasons.append("displacement_not_confirmed")
            if htf_overlap: reasons.append("htf_poi_overlap")
            if sweep["type"] == "sweep": reasons.append("liquidity_sweep")
            archetype = _archetype(direction, sweep, cand.get("kind", "market"), displacement, pullback, htf_overlap, (market_context or {}).get("market_regime", ""))
            components = {
                "direction": ctx["direction_quality"],
                "setup": setup_score,
                "location": location["score"],
                "risk": _clip(risk_score),
                "context": context_q["score"],
            }
            conf, quality, penalty = _confidence(components, contradictions)
            hist = _history_context(trade_history, direction, cand.get("kind", "market"), archetype)
            hist_adj = float(hist.get("adjustment", 0.0))
            # Historical adjustment is deliberately tiny and cannot reorder the primary quality model aggressively.
            conf = int(np.clip(round(conf + hist_adj), 20, 90))
            execution_score = quality * 0.82 + min(rr, 6.0) * 2.2 + float(cand.get("score", 0)) * 0.08 - contradictions * 3.5
            evaluated.append({
                "entry": entry, "entry_label": cand.get("kind", "market"), "sl": sl_price, "risk": risk,
                "tp": tp_price, "tp_label": tp_label, "rr": rr, "confidence": conf, "quality": quality,
                "execution_score": execution_score, "contradictions": contradictions, "reason": reasons,
                "archetype": archetype, "hist": hist, "components": components, "penalty": penalty,
                "poi_quality": poi_quality, "location": location, "pullback": pullback,
                "risk_atr": risk_atr, "displacement_atr": displacement_atr,
            })
        if not evaluated:
            return None
        evaluated.sort(key=lambda x: x["execution_score"], reverse=True)
        best = evaluated[0]
        confidence_band = "ELITE" if best["confidence"] >= 75 else "STRONG" if best["confidence"] >= 65 else "VALID" if best["confidence"] >= 50 else "WEAK"
        return {
            "symbol": symbol,
            "decision": "BUY" if direction == "bull" else "SELL",
            "confidence": int(best["confidence"]),
            "direction_confidence": int(round(ctx["direction_quality"])),
            "setup_quality": int(round(best["quality"])),
            "confidence_band": confidence_band,
            "confidence_model": CONFIDENCE_MODEL_VERSION,
            "confidence_is_probability": False,
            "confidence_diagnostics": {
                "model": CONFIDENCE_MODEL_VERSION,
                "components": best["components"],
                "contradictions": best["contradictions"],
                "contradiction_reasons": best["reason"],
                "historical_context": best["hist"],
                "archetype": best["archetype"],
                "quality_before_history": best["quality"],
                "history_adjustment": best["hist"].get("adjustment", 0.0),
                "quality_penalty": best["penalty"],
                "market_context": context_q,
            },
            "market_thesis": {
                "direction": "BUY" if direction == "bull" else "SELL",
                "archetype": best["archetype"],
                "evidence": best["reason"],
                "market_regime": (market_context or {}).get("market_regime", ctx["macro_bias"]),
            },
            "entry_location_score": int(round(best["location"]["score"])),
            "entry_location_state": "preferred" if best["location"]["score"] >= 70 else "acceptable" if best["location"]["score"] >= 45 else "late",
            "entry_range_position": best["location"]["range_position"],
            "trend_strength": {"score": ctx["direction_quality"], "state": ctx["struct_h1"]},
            "pullback_quality": best["pullback"],
            "liquidity_context": {"sweep": sweep},
            "poi_quality": best["poi_quality"],
            "poi_state": "fresh" if best["poi_quality"] >= 70 else "usable",
            "market_regime": (market_context or {}).get("market_regime", ctx["macro_bias"]),
            "entry": round(best["entry"], 10),
            "price": round(price, 10),
            "entry_label": best["entry_label"],
            "sl": round(best["sl"], 10),
            "initial_sl": round(best["sl"], 10),
            "initial_risk": round(best["risk"], 10),
            "tp": round(best["tp"], 10),
            "rr": round(best["rr"], 3),
            "tp_label": best["tp_label"],
            "atr": round(atr, 10),
            "risk_atr": round(best["risk_atr"], 3),
            "rsi": round(float(m15["rsi"].iloc[-1]), 2),
            "struct_h1": ctx["struct_h1"],
            "d1_bias": ctx["d1_bias"],
            "htf_bias": ctx["d1_bias"] if ctx["d1_bias"] != "neutral" else ctx["struct_h1"],
            "h1_bias": ctx["struct_h1"],
            "choch_m15": {"bullish_choch": False, "bearish_choch": False},
            "choch_h1": {"bullish_choch": False, "bearish_choch": False},
            "cisd_m15": {"bullish_cisd": False, "bearish_cisd": False},
            "failed_retest": {},
            "entry_confirmation": {"confirmed": bool(displacement), "kind": "displacement_close" if displacement else "none", "body_atr": round(displacement_atr, 3)},
            "selected_sweep": bool(sweep["type"] == "sweep"),
            "trigger_count": int(bool(displacement)) + int(bool(sweep["type"] == "sweep")),
            "m15_relative_volume": float(m15["volume"].iloc[-1] / max(m15["vol_sma"].iloc[-1], 1e-12)),
            "m15_rsi": float(m15["rsi"].iloc[-1]),
            "m15_rsi_slope": float(m15["rsi"].iloc[-1] - m15["rsi"].iloc[-2]),
            "v11_quality": {"trend_strength": {"score": ctx["direction_quality"]}, "pullback_quality": best["pullback"], "poi_quality": best["poi_quality"]},
            "reasoning_engine": TRAIL_ENGINE_VERSION,
            "tp_sl_reason": f"Entry@{best['entry']:.8g}({best['entry_label']}) | SL@{best['sl']:.8g} | TP@{best['tp']:.8g}({best['tp_label']}) | RR={best['rr']:.2f} | quality={best['quality']:.1f} | conf={best['confidence']}%",
        }
    except Exception as exc:
        log.exception("[full_analyze] %s: %s", symbol or "?", exc)
        return None


def _current_price(df_m15, state):
    p = _num(state.get("current_price"), None)
    if p is not None:
        return p
    return float(df_m15["close"].iloc[-1]) if df_m15 is not None and not df_m15.empty else None


def _path_metrics(df_m15, state, direction, entry, risk, current_price):
    current_r = ((current_price - entry) / risk) if direction == "bull" else ((entry - current_price) / risk)
    state_mfe = _num(state.get("mfe_r"), 0.0) or 0.0
    state_mae = _num(state.get("mae_r"), 0.0) or 0.0
    window = df_m15.tail(TRAIL_PEAK_LOOKBACK)
    peak = float(window["high"].max()) if direction == "bull" else float(window["low"].min())
    peak_r = ((peak - entry) / risk) if direction == "bull" else ((entry - peak) / risk)
    mfe = max(0.0, state_mfe, peak_r)
    giveback_r = max(0.0, mfe - current_r)
    ratio = giveback_r / max(mfe, 0.25) if mfe > 0 else 0.0
    return {
        "current_r": round(current_r, 4),
        "mfe_r": round(mfe, 4),
        "mae_r": round(state_mae, 4),
        "giveback_r": round(giveback_r, 4),
        "giveback_ratio": round(ratio, 4),
    }


def _reversal_state(df, direction, current_r, giveback_ratio):
    atr = max(float(df["atr"].iloc[-1]), 1e-12)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(float(last["close"] - last["open"])) / atr
    counter = (float(last["close"]) < float(last["open"])) if direction == "bull" else (float(last["close"]) > float(last["open"]))
    prev_counter = (float(prev["close"]) < float(prev["open"])) if direction == "bull" else (float(prev["close"]) > float(prev["open"]))
    rv = float(last["volume"] / max(last["vol_sma"], 1e-12))
    if counter and prev_counter and body >= TRAIL_REVERSAL_BODY_ATR and giveback_ratio >= TRAIL_GIVEBACK_STRONG:
        return "REVERSAL_CONFIRMED", 3, rv
    if counter and body >= TRAIL_COUNTER_BODY_ATR and giveback_ratio >= TRAIL_GIVEBACK_WARN:
        return "WEAKENING", 2, rv
    if giveback_ratio >= TRAIL_GIVEBACK_STRONG or current_r < 0.5:
        return "CAUTION", 1, rv
    return "HEALTHY", 0, rv


def _structural_trail(df, direction, current_price, atr):
    sh, sl = swing_pts(df.tail(STRUCT_TRAIL_LOOKBACK), STRUCT_TRAIL_LB)
    sub = df.tail(STRUCT_TRAIL_LOOKBACK)
    if direction == "bull" and sl:
        level = float(sub["low"].iloc[sl[-1]]) - atr * TRAIL_STRUCT_BUFFER_ATR
        return level
    if direction == "bear" and sh:
        level = float(sub["high"].iloc[sh[-1]]) + atr * TRAIL_STRUCT_BUFFER_ATR
        return level
    return None


def _retracement_trail(df, direction, entry, risk, current_price, path, state):
    atr = max(float(df["atr"].iloc[-1]), 1e-12)
    mfe = path["mfe_r"]
    gb = path["giveback_ratio"]
    if direction == "bull":
        peak_price = float(df["high"].tail(TRAIL_PEAK_LOOKBACK).max())
        if mfe < 1.0 or gb < TRAIL_GIVEBACK_WARN:
            return None
        retrace = max(0.18, 0.35 if gb < TRAIL_GIVEBACK_STRONG else 0.25)
        target = peak_price - (peak_price - entry) * retrace
        return min(target, current_price - atr * TRAIL_RETRACE_BUFFER_ATR)
    peak_price = float(df["low"].tail(TRAIL_PEAK_LOOKBACK).min())
    if mfe < 1.0 or gb < TRAIL_GIVEBACK_WARN:
        return None
    retrace = max(0.18, 0.35 if gb < TRAIL_GIVEBACK_STRONG else 0.25)
    target = peak_price + (entry - peak_price) * retrace
    return max(target, current_price + atr * TRAIL_RETRACE_BUFFER_ATR)


def manage_position(state: dict, df_m15: pd.DataFrame, df_h1: Optional[pd.DataFrame] = None, df_d1: Optional[pd.DataFrame] = None, symbol: Optional[str] = None):
    try:
        if df_m15 is None or len(df_m15) < 40:
            return {"action": "PROTECT", "state": "UNKNOWN", "reason": ["insufficient_m15_data"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        m15 = build_df(df_m15, 15)
        if m15 is None or len(m15) < 40:
            return {"action": "PROTECT", "state": "UNKNOWN", "reason": ["insufficient_m15_data"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        sig = state.get("signal") or {}
        side = str(sig.get("decision") or "BUY").upper()
        direction = "bull" if side == "BUY" else "bear"
        entry = _num(state.get("entry") or sig.get("entry"), None)
        initial_sl = _num(state.get("initial_sl") or sig.get("initial_sl") or sig.get("sl"), None)
        current_sl = _num(state.get("current_sl") or sig.get("sl"), None)
        current_price = _current_price(m15, state)
        if entry is None or initial_sl is None or current_sl is None or current_price is None:
            return {"action": "PROTECT", "state": "UNKNOWN", "reason": ["missing_position_geometry"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        risk = abs(entry - initial_sl)
        if risk <= 0:
            return {"action": "PROTECT", "state": "UNKNOWN", "reason": ["invalid_initial_risk"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        if direction == "bull" and current_price <= current_sl:
            return {"action": "PROTECT", "state": "AT_STOP", "reason": ["market_at_or_below_stop"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        if direction == "bear" and current_price >= current_sl:
            return {"action": "PROTECT", "state": "AT_STOP", "reason": ["market_at_or_above_stop"], "reasoning_engine": TRAIL_ENGINE_VERSION}

        path = _path_metrics(m15, state, direction, entry, risk, current_price)
        cur_r = path["current_r"]
        mfe = path["mfe_r"]
        gb = path["giveback_ratio"]
        state_name, reversal_score, rv = _reversal_state(m15, direction, cur_r, gb)
        reasons = []
        if cur_r < TRAIL_ARM_R and mfe < TRAIL_ARM_R:
            return {"action": "HOLD", "state": "INITIAL" if cur_r < 0.35 else "PROVING", "profit_r": cur_r, "lifecycle": path, "weakness_score": 0, "relative_volume": rv, "reason": ["path_not_mature_for_trailing"], "reasoning_engine": TRAIL_ENGINE_VERSION}

        atr = max(float(m15["atr"].iloc[-1]), 1e-12)
        structural = _structural_trail(m15, direction, current_price, atr)
        retracement = _retracement_trail(m15, direction, entry, risk, current_price, path, state)
        candidates = []
        if structural is not None:
            candidates.append((float(structural), "structure"))
        if retracement is not None:
            candidates.append((float(retracement), "retracement"))

        # Conservative profit floor after realized excursion. Never move to a side that can be immediately triggered.
        if mfe >= TRAIL_MFE_EXTENDED:
            lock_r = TRAIL_LOCK_STRONG_R if gb >= TRAIL_GIVEBACK_WARN else TRAIL_LOCK_WARN_R
            if gb >= TRAIL_GIVEBACK_CRITICAL and mfe >= TRAIL_MFE_DEEP:
                lock_r = TRAIL_LOCK_CRITICAL_R
            lock = entry + risk * lock_r if direction == "bull" else entry - risk * lock_r
            candidates.append((float(lock), "path_protection"))
            reasons.append(f"mfe={mfe:.2f}R")
        if gb >= TRAIL_GIVEBACK_WARN:
            reasons.append(f"giveback={gb:.0%}")
        if reversal_score:
            reasons.append(f"reversal_signal={reversal_score}")
        if rv <= TRAIL_VOLUME_EXHAUSTION:
            reasons.append("volume_exhaustion")
        elif rv >= TRAIL_VOLUME_COUNTER and reversal_score:
            reasons.append("counter_volume")

        valid = []
        for cand, source in candidates:
            if direction == "bull":
                cand = min(cand, current_price - atr * TRAIL_MIN_MARKET_GAP_ATR)
                if cand <= current_sl:
                    continue
                if cand >= current_price:
                    continue
            else:
                cand = max(cand, current_price + atr * TRAIL_MIN_MARKET_GAP_ATR)
                if cand >= current_sl:
                    continue
                if cand <= current_price:
                    continue
            valid.append((cand, source))
        if not valid:
            return {"action": "PROTECT" if gb >= TRAIL_GIVEBACK_WARN or reversal_score >= 2 else "HOLD", "state": state_name, "profit_r": round(cur_r, 3), "lifecycle": path, "weakness_score": reversal_score, "relative_volume": rv, "reason": reasons + ["no_safe_trail_candidate"], "reasoning_engine": TRAIL_ENGINE_VERSION}

        # For BUY, highest safe stop is most protective; for SELL, lowest is most protective.
        best_cand, best_source = (max(valid, key=lambda x: x[0]) if direction == "bull" else min(valid, key=lambda x: x[0]))
        improvement_r = ((best_cand - current_sl) if direction == "bull" else (current_sl - best_cand)) / risk
        update_count = int(state.get("trail_update_count", 0) or 0)
        if improvement_r < TRAIL_MIN_UPDATE_R and update_count >= TRAIL_MAX_CHURN:
            return {"action": "PROTECT", "state": state_name, "profit_r": round(cur_r, 3), "lifecycle": path, "weakness_score": reversal_score, "relative_volume": rv, "reason": reasons + ["anti_churn"], "reasoning_engine": TRAIL_ENGINE_VERSION}

        locked_r = ((best_cand - entry) if direction == "bull" else (entry - best_cand)) / risk
        return {
            "action": "TRAIL",
            "state": state_name,
            "sl": round(float(best_cand), 10),
            "profit_r": round(cur_r, 3),
            "locked_r": round(float(locked_r), 3),
            "trail_source": best_source,
            "candidate_type": best_source,
            "weakness_score": reversal_score,
            "relative_volume": rv,
            "lifecycle": path,
            "reversal_diagnostics": {"state": state_name, "score": reversal_score, "relative_volume": rv},
            "reason": reasons + [f"source={best_source}"],
            "reasoning_engine": TRAIL_ENGINE_VERSION,
        }
    except Exception as exc:
        log.exception("[manage_position] %s: %s", symbol or "?", exc)
        return {"action": "PROTECT", "state": "ERROR", "reason": [f"management_exception:{type(exc).__name__}"], "reasoning_engine": TRAIL_ENGINE_VERSION}


def get_best_signal(candidates: list) -> Optional[dict]:
    if not candidates:
        return None
    return max(candidates, key=lambda x: float(x.get("confidence", 0) or 0) * 0.8 + float(x.get("rr", 0) or 0) * 2.0)
