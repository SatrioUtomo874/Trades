"""SMCAutoTrade strategy engine V15.

Single decision path for entry and a separate path-aware manager for open positions.
The engine uses only pandas/numpy and accepts the market_context produced by main.py
when available. No network/API access is performed here.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_RR = 2.0
MAX_RR = None
FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618
TRAIL_R_LADDER = []
STRUCT_TRAIL_LB = 3
STRUCT_TRAIL_BUF_PCT = 0.0025
STRUCT_TRAIL_LOOKBACK = 60
TRAIL_ENGINE_VERSION = "15.0-evidence-hierarchy-path-brain"
TRAIL_EXECUTION_BUFFER_ATR = 0.08
TRAIL_MIN_MARKET_GAP_ATR = 0.35
MAIN_ENTRY_MAX_ATR = 1.50
CONFIDENCE_MODEL_VERSION = "15.0-monotonic-evidence-quality"

MIN_DISPLACEMENT_ATR = 0.30
ENTRY_LOOKBACK = 20
SWING_LB = 3
POI_LOOKBACK = 80
HTF_POI_LOOKBACK = 90
ENTRY_MAX_RISK_ATR = 2.20
ENTRY_MIN_RISK_ATR = 0.55
ENTRY_MIN_RISK_PCT = 0.08
ENTRY_MAX_RISK_PCT = 3.50
ENTRY_PREFERRED_BUY = 0.55
ENTRY_PREFERRED_SELL = 0.45

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
        x = float(v)
        return default if not np.isfinite(x) else x
    except Exception:
        return default


def _clip(v, lo=0.0, hi=100.0):
    return float(np.clip(float(v), lo, hi))


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr_fn(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _closed_candles(df, minutes):
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    idx = out.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    boundary = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
    if idx[-1] < boundary:
        return out
    return out.loc[idx < boundary].copy()


def build_df(df, interval_minutes=None):
    if df is None or len(df) < 60:
        return None
    x = df.copy()
    if interval_minutes:
        x = _closed_candles(x, interval_minutes)
    if x is None or len(x) < 60:
        return None
    x["ema9"] = ema(x["close"], 9)
    x["ema21"] = ema(x["close"], 21)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200) if len(x) >= 200 else ema(x["close"], 50)
    x["rsi"] = rsi(x["close"])
    x["atr"] = atr_fn(x)
    x["vol_sma"] = x["volume"].rolling(20).mean()
    return x.dropna()


def swing_pts(df, lb=5):
    sh, sl = [], []
    if df is None or len(df) < lb * 2 + 3:
        return sh, sl
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    for i in range(lb, len(df) - lb):
        if hi[i] == np.max(hi[i - lb:i + lb + 1]):
            sh.append(i)
        if lo[i] == np.min(lo[i - lb:i + lb + 1]):
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


def _direction_context(h1, m15, d1, market_context):
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

    ema_bull = bool(h1["ema9"].iloc[-1] > h1["ema21"].iloc[-1] > h1["ema50"].iloc[-1])
    ema_bear = bool(h1["ema9"].iloc[-1] < h1["ema21"].iloc[-1] < h1["ema50"].iloc[-1])
    atr = max(float(m15["atr"].iloc[-1]), float(h1["atr"].iloc[-1]) / 4, float(m15["close"].iloc[-1]) * 0.003)
    fast = (float(m15["close"].iloc[-1]) - float(m15["close"].iloc[-4])) / max(atr, 1e-12)
    slow = (float(m15["close"].iloc[-1]) - float(m15["close"].iloc[-9])) / max(atr, 1e-12)

    bull = 50.0
    bear = 50.0
    if d1_bias == "bullish": bull += 18
    elif d1_bias == "bearish": bear += 18
    if struct_h1 == "bullish": bull += 22
    elif struct_h1 == "bearish": bear += 22
    if ema_bull: bull += 8
    if ema_bear: bear += 8
    if fast > 0: bull += min(7.0, fast * 3.0)
    if fast < 0: bear += min(7.0, -fast * 3.0)
    if slow > 0: bull += min(7.0, slow * 1.5)
    if slow < 0: bear += min(7.0, -slow * 1.5)
    if struct_m15 == "bullish": bull += 5
    elif struct_m15 == "bearish": bear += 5

    mc = market_context if isinstance(market_context, dict) else {}
    breadth = _num(mc.get("bullish_breadth_pct"), None)
    breadth_score = 50.0
    if breadth is not None:
        breadth_score = _clip(50 + (breadth - 50) * 1.5)
        if breadth >= 65: bull += 6
        elif breadth <= 35: bear += 6
    rs = _num(mc.get("relative_strength_1h_pct"), None)
    if rs is not None:
        if rs >= 0.35: bull += 5
        elif rs <= -0.35: bear += 5
    rv = _num(mc.get("relative_volume"), None)
    if rv is not None and rv >= 1.15:
        if fast > 0: bull += 3
        elif fast < 0: bear += 3

    regime = str(mc.get("market_regime") or mc.get("chart_regime") or "").lower()
    macro = "unknown"
    if "bull" in regime: macro = "bullish"
    elif "bear" in regime: macro = "bearish"
    elif "range" in regime or "compression" in regime: macro = "ranging"
    if macro == "bullish":
        bull += 4
        bear *= 0.90
    elif macro == "bearish":
        bear += 4
        bull *= 0.90

    direction = "bull" if bull >= bear else "bear"
    edge = abs(bull - bear)
    quality = _clip(45 + edge * 1.5)
    htf_alignment = (
        d1_bias == "neutral" or struct_h1 == "ranging" or d1_bias == struct_h1
    )
    htf_conflict = d1_bias in ("bullish", "bearish") and struct_h1 in ("bullish", "bearish") and d1_bias != struct_h1
    return {
        "direction": direction, "bull": round(bull, 2), "bear": round(bear, 2),
        "direction_quality": round(quality, 2), "edge": round(edge, 2),
        "struct_h1": struct_h1, "struct_m15": struct_m15, "d1_bias": d1_bias,
        "macro_bias": macro, "htf_alignment": htf_alignment, "htf_conflict": htf_conflict,
        "atr": atr, "price": float(m15["close"].iloc[-1]),
        "sh1": sh1, "sl1": sl1, "sh15": sh15, "sl15": sl15,
        "breadth_score": breadth_score,
    }


def score_direction(df_h1, df_m15, df_d1=None, df_btc_h1=None):
    h1, m15 = build_df(df_h1, 60), build_df(df_m15, 15)
    if h1 is None or m15 is None:
        return None
    d1 = build_df(df_d1, 1440) if df_d1 is not None and len(df_d1) >= 60 else None
    ctx = _direction_context(h1, m15, d1, {})
    ctx["h1"], ctx["m15"], ctx["d1"] = h1, m15, d1
    return ctx


def _zone_fresh(df, idx, top, bot, direction):
    if idx >= len(df) - 2:
        return True
    sub = df.iloc[idx + 2:]
    return not bool((sub["close"] < bot).any()) if direction == "bull" else not bool((sub["close"] > top).any())


def _find_ob(df, direction, lookback=POI_LOOKBACK):
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
        if body < avg_body * 1.20:
            continue
        top, bot = float(max(c["open"], c["close"])), float(min(c["open"], c["close"]))
        idx = base + i
        if not _zone_fresh(df, idx, top, bot, direction):
            continue
        score = min(45.0, body / avg_body * 15.0)
        if i + 2 < len(sub):
            c2 = sub.iloc[i + 2]
            if direction == "bull" and c2["low"] > c["high"]: score += 22
            if direction == "bear" and c2["high"] < c["low"]: score += 22
        if idx >= len(df) - 20: score += 15
        out.append({"top": top, "bot": bot, "mid": (top + bot) / 2, "idx": idx, "score": _clip(score), "kind": "ob"})
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
        if not _zone_fresh(df, idx, top, bot, direction):
            continue
        width = top - bot
        atr = max(float(df["atr"].iloc[-1]), 1e-12)
        recency = 12.0 if idx >= len(df) - 16 else 0.0
        score = _clip(40 + min(30, width / atr * 18) + recency)
        out.append({"top": top, "bot": bot, "mid": (top + bot) / 2, "idx": idx, "score": score, "kind": "fvg"})
    out.sort(key=lambda z: (-z["score"], -z["idx"]))
    return out[:4]


def _find_equal(df, kind, lookback=80, tol=0.0025):
    vals = df[kind].tail(lookback).to_numpy(dtype=float)
    out = []
    for i in range(len(vals)):
        grp = [vals[i]]
        for j in range(i + 1, len(vals)):
            if abs(vals[i] - vals[j]) / max(abs(vals[i]), 1e-12) <= tol:
                grp.append(vals[j])
        if len(grp) >= 2:
            out.append(float(np.mean(grp)))
    return sorted(set(round(x, 10) for x in out))


def _sweep(df, direction, sh, sl):
    if direction == "bull" and sl:
        level = float(df["low"].iloc[sl[-1]])
        lo, close = float(df["low"].iloc[-1]), float(df["close"].iloc[-1])
        if lo < level and close > level:
            return {"type": "sweep", "level": level, "strength": _clip(45 + (level - lo) / max(float(df["atr"].iloc[-1]), 1e-12) * 25)}
    if direction == "bear" and sh:
        level = float(df["high"].iloc[sh[-1]])
        hi, close = float(df["high"].iloc[-1]), float(df["close"].iloc[-1])
        if hi > level and close < level:
            return {"type": "sweep", "level": level, "strength": _clip(45 + (hi - level) / max(float(df["atr"].iloc[-1]), 1e-12) * 25)}
    return {"type": "none", "level": None, "strength": 0.0}


def _displacement(df, direction):
    if len(df) < 8:
        return {"confirmed": False, "body_atr": 0.0, "range_break": False}
    last = df.iloc[-1]
    atr = max(float(last["atr"]), 1e-12)
    body = abs(float(last["close"] - last["open"]))
    prior = df.iloc[-5:-1]
    if direction == "bull":
        rb = float(last["close"]) > float(prior["high"].max())
        conf = float(last["close"]) > float(last["open"]) and rb and body >= atr * MIN_DISPLACEMENT_ATR
    else:
        rb = float(last["close"]) < float(prior["low"].min())
        conf = float(last["close"]) < float(last["open"]) and rb and body >= atr * MIN_DISPLACEMENT_ATR
    return {"confirmed": bool(conf), "body_atr": round(body / atr, 3), "range_break": bool(rb)}


def _pullback(df, direction, atr):
    if len(df) < 14:
        return {"score": 45.0, "state": "unknown", "efficiency": 0.0, "depth_atr": 0.0}
    sub = df.tail(12)
    gross = float(sub["close"].diff().abs().sum())
    net = float(sub["close"].iloc[-1] - sub["close"].iloc[0])
    if direction == "bear": net = -net
    eff = abs(net) / max(gross, 1e-12)
    if direction == "bull":
        adverse = max(0.0, float(sub["high"].iloc[0] - sub["low"].min()))
    else:
        adverse = max(0.0, float(sub["high"].max() - sub["low"].iloc[0]))
    depth = adverse / max(atr, 1e-12)
    score = 52.0
    if 0.25 <= depth <= 1.6: score += 18
    elif depth > 2.0: score -= 22
    if eff >= 0.45: score += 18
    elif eff < 0.20: score -= 14
    aligned_last = (direction == "bull" and sub["close"].iloc[-1] > sub["close"].iloc[-2]) or (direction == "bear" and sub["close"].iloc[-1] < sub["close"].iloc[-2])
    if aligned_last: score += 7
    return {"score": _clip(score), "state": "clean" if score >= 72 else "mixed" if score >= 45 else "damaged", "efficiency": round(eff, 3), "depth_atr": round(depth, 3)}


def _location(df, direction, price, atr):
    look = df.tail(ENTRY_LOOKBACK)
    lo, hi = float(look["low"].min()), float(look["high"].max())
    pos = (price - lo) / max(hi - lo, 1e-12)
    if direction == "bull":
        score = 76 if pos <= ENTRY_PREFERRED_BUY else 62 if pos < 0.82 else 30
    else:
        score = 76 if pos >= ENTRY_PREFERRED_SELL else 62 if pos > 0.18 else 30
    edge_gap = (hi - price) / max(atr, 1e-12) if direction == "bull" else (price - lo) / max(atr, 1e-12)
    if edge_gap < 0.45: score -= 12
    return {"score": _clip(score), "range_position": round(pos, 3), "range_low": lo, "range_high": hi}


def _context_quality(mc, direction):
    if not isinstance(mc, dict) or not mc:
        return {"score": 50.0, "conflicts": [], "reasons": ["market_context_unavailable"]}
    score, reasons, conflicts = 50.0, [], []
    breadth = _num(mc.get("bullish_breadth_pct"), None)
    if breadth is not None:
        aligned = breadth >= 60 if direction == "bull" else breadth <= 40
        opposed = breadth <= 40 if direction == "bull" else breadth >= 60
        if aligned: score += 15; reasons.append("breadth_aligned")
        elif opposed: score -= 18; conflicts.append("breadth_opposed")
        elif (45 <= breadth <= 55): reasons.append("breadth_neutral")
    rs = _num(mc.get("relative_strength_1h_pct"), None)
    if rs is not None:
        if (direction == "bull" and rs >= 0.35) or (direction == "bear" and rs <= -0.35): score += 14; reasons.append("relative_strength_aligned")
        elif (direction == "bull" and rs <= -0.35) or (direction == "bear" and rs >= 0.35): score -= 14; conflicts.append("relative_strength_opposed")
    rv = _num(mc.get("relative_volume"), None)
    if rv is not None:
        if rv >= 1.20: score += 8; reasons.append("participation_expanded")
        elif rv <= 0.55: score -= 6; conflicts.append("participation_thin")
    regime = str(mc.get("market_regime") or mc.get("chart_regime") or "").lower()
    if "transition" in regime: score -= 4; conflicts.append("transition_regime")
    elif "range" in regime or "compression" in regime: score -= 2; reasons.append("range_regime")
    elif "trend" in regime or "expansion" in regime: score += 4; reasons.append("trend_regime")
    return {"score": _clip(score), "conflicts": conflicts, "reasons": reasons}


def _geometry(m15, h1, direction, entry, atr):
    sh, sl = swing_pts(m15, SWING_LB)
    hsh, hsl = swing_pts(h1, 5)
    anchors = []
    if direction == "bull":
        if sl: anchors.append(float(m15["low"].iloc[sl[-1]]))
        if hsl: anchors.append(float(h1["low"].iloc[hsl[-1]]))
        valid = [x for x in anchors if x < entry]
        anchor = min(valid) if valid else entry - atr
        stop = anchor - atr * 0.18
    else:
        if sh: anchors.append(float(m15["high"].iloc[sh[-1]]))
        if hsh: anchors.append(float(h1["high"].iloc[hsh[-1]]))
        valid = [x for x in anchors if x > entry]
        anchor = max(valid) if valid else entry + atr
        stop = anchor + atr * 0.18
    risk = (entry - stop) if direction == "bull" else (stop - entry)
    if risk <= 0: return None
    risk_atr, risk_pct = risk / max(atr, 1e-12), risk / max(entry, 1e-12) * 100
    if risk_atr < ENTRY_MIN_RISK_ATR or risk_atr > ENTRY_MAX_RISK_ATR or risk_pct < ENTRY_MIN_RISK_PCT or risk_pct > ENTRY_MAX_RISK_PCT:
        return None
    return float(stop), float(risk), float(risk_atr), float(risk_pct), sh, sl, hsh, hsl


def _targets(m15, h1, direction, entry, risk):
    vals = []
    if direction == "bull":
        sh, _ = swing_pts(h1, 5)
        for i in reversed(sh[-8:]):
            x = float(h1["high"].iloc[i])
            if x > entry: vals.append((x, "h1_swing"))
        vals += [(entry + risk * 1.272, "fib_1.272"), (entry + risk * 1.618, "fib_1.618")]
        vals = [v for v in vals if v[0] > entry + risk * 1.95]
    else:
        _, sl = swing_pts(h1, 5)
        for i in reversed(sl[-8:]):
            x = float(h1["low"].iloc[i])
            if x < entry: vals.append((x, "h1_swing"))
        vals += [(entry - risk * 1.272, "fib_1.272"), (entry - risk * 1.618, "fib_1.618")]
        vals = [v for v in vals if v[0] < entry - risk * 1.95]
    if not vals: return None
    vals.sort(key=lambda z: abs((((z[0] - entry) if direction == "bull" else (entry - z[0])) / risk) - 2.6))
    tp, label = vals[0]
    rr = ((tp - entry) if direction == "bull" else (entry - tp)) / risk
    return float(tp), label, float(rr)


def _candidate_quality(direction, cand, htf_overlap, sweep, displacement, pullback, location, context, geom, rr, ctx, market_regime):
    risk, risk_atr, risk_pct = geom[1], geom[2], geom[3]
    poi_base = _clip(cand.get("score", 35.0) * 0.95)
    poi = poi_base + (10 if htf_overlap else 0)
    if cand.get("kind") == "fvg": poi -= 4 if not displacement["confirmed"] else 0
    if cand.get("kind") == "market": poi = min(poi, 38.0)
    poi = _clip(poi)

    trigger = 52.0
    if displacement["confirmed"]: trigger += 35
    elif displacement["range_break"]: trigger += 12
    if sweep["type"] == "sweep": trigger += 8
    if pullback["state"] == "clean": trigger += 8
    elif pullback["state"] == "damaged": trigger -= 20
    trigger = _clip(trigger)

    risk_score = 78.0
    if risk_atr < 0.70: risk_score -= 24
    elif risk_atr > 1.85: risk_score -= 15
    elif 0.85 <= risk_atr <= 1.55: risk_score += 10
    if risk_pct < 0.10: risk_score -= 10
    risk_score = _clip(risk_score)

    direction_score = ctx["direction_quality"]
    htf_score = 82.0 if ctx["htf_alignment"] else 45.0 if ctx["htf_conflict"] else 60.0
    market_score = context["score"]
    location_score = location["score"]

    positive = [
        ("htf", htf_score, 0.20),
        ("direction", direction_score, 0.18),
        ("poi", poi, 0.17),
        ("trigger", trigger, 0.20),
        ("location", location_score, 0.08),
        ("risk", risk_score, 0.08),
        ("context", market_score, 0.09),
    ]
    quality = sum(v * w for _, v, w in positive)

    contradictions = []
    if ctx["htf_conflict"]: contradictions.append("d1_h1_conflict")
    if pullback["state"] == "damaged": contradictions.append("damaged_pullback")
    if location_score < 40: contradictions.append("poor_location")
    contradictions.extend(context.get("conflicts", [])[:3])
    if not displacement["confirmed"]: contradictions.append("no_fresh_displacement")
    if sweep["type"] == "sweep":
        quality += min(5.0, sweep["strength"] * 0.04)
    else:
        quality -= 2.0

    penalty = min(24.0, len(contradictions) * 4.5)
    quality = _clip(quality - penalty)

    completeness = (
        int(displacement["confirmed"]) +
        int(sweep["type"] == "sweep") +
        int(htf_overlap) +
        int(pullback["state"] == "clean")
    )
    if completeness == 0: quality -= 10
    elif completeness == 1: quality -= 3
    elif completeness >= 3: quality += 4
    quality = _clip(quality)

    # Monotonic mapping: all inputs are normalized evidence scores, and the same
    # quality score always maps to the same confidence. There is no post-hoc
    # history boost that can reorder candidates.
    confidence = int(round(18 + quality * 0.74))
    confidence = int(np.clip(confidence, 25, 92))
    return {
        "quality": round(quality, 2), "confidence": confidence,
        "components": {k: round(v, 2) for k, v, _ in positive},
        "contradictions": contradictions, "penalty": round(penalty, 2),
        "completeness": completeness, "risk_atr": round(risk_atr, 3), "risk_pct": round(risk_pct, 4),
        "poi_quality": round(poi, 2), "trigger_quality": round(trigger, 2),
    }


def _archetype(direction, cand_kind, sweep, displacement, pullback, htf_overlap, regime):
    if sweep["type"] == "sweep" and displacement["confirmed"]:
        return "LIQUIDITY_SWEEP_RECLAIM"
    if htf_overlap and cand_kind == "ob" and pullback["state"] == "clean":
        return "HTF_OB_PULLBACK_CONTINUATION"
    if cand_kind == "fvg" and displacement["confirmed"]:
        return "FVG_DISPLACEMENT_RETEST"
    if "range" in str(regime).lower() and sweep["type"] == "sweep":
        return "RANGE_LIQUIDITY_REVERSAL"
    if displacement["confirmed"] and pullback["state"] == "clean":
        return "STRUCTURE_CONTINUATION"
    return "EARLY_CONTEXTUAL_SETUP"


def _history_context(history, direction, entry_label, archetype):
    rows = [r for r in (history or []) if isinstance(r, dict)]
    if len(rows) < 10:
        return {"samples": len(rows), "matched": 0, "weight": 0.0, "note": "insufficient_history"}
    side = "BUY" if direction == "bull" else "SELL"
    matched = [r for r in rows if str(r.get("decision", "")).upper() == side and str(r.get("entry_label", "")) == entry_label]
    if len(matched) < 6:
        matched = [r for r in rows if str(r.get("decision", "")).upper() == side]
    if not matched:
        return {"samples": len(rows), "matched": 0, "weight": 0.0, "note": "no_match"}
    wins = np.array([1.0 if _num(r.get("pnl_usd"), 0.0) > 0 else 0.0 for r in matched])
    wr = float(wins.mean())
    # History is audit context only. It never reorders the primary score.
    return {"samples": len(rows), "matched": len(matched), "win_rate": round(wr, 3), "weight": round(min(0.10, len(matched) / 80), 3), "note": "audit_context_only"}


def full_analyze(df_h1, df_m15, df_d1=None, symbol=None, df_btc_h1=None, trade_history=None, market_context=None):
    try:
        h1, m15 = build_df(df_h1, 60), build_df(df_m15, 15)
        if h1 is None or m15 is None:
            return None
        d1 = build_df(df_d1, 1440) if df_d1 is not None and len(df_d1) >= 60 else None
        ctx = _direction_context(h1, m15, d1, market_context)
        direction = ctx["direction"]
        price, atr = ctx["price"], ctx["atr"]
        sweep = _sweep(m15, direction, ctx["sh15"], ctx["sl15"])
        displacement = _displacement(m15, direction)
        pullback = _pullback(m15, direction, atr)
        location = _location(m15, direction, price, atr)
        context = _context_quality(market_context, direction)

        zones = _find_ob(m15, direction) + _find_fvg(m15, direction)
        zones += _find_ob(h1, direction, HTF_POI_LOOKBACK) + _find_fvg(h1, direction, HTF_POI_LOOKBACK)
        zones = [z for z in zones if abs(price - float(z["mid"])) <= atr * MAIN_ENTRY_MAX_ATR and ((direction == "bull" and z["mid"] <= price * 1.006) or (direction == "bear" and z["mid"] >= price * 0.994))]
        zones.sort(key=lambda z: (-z.get("score", 0), -z.get("idx", 0)))
        if not zones:
            zones = [{"mid": price, "top": price, "bot": price, "score": 25.0, "kind": "market", "idx": len(m15) - 1}]
        htf_zones = _find_ob(h1, direction, HTF_POI_LOOKBACK) + _find_fvg(h1, direction, HTF_POI_LOOKBACK)

        evaluated = []
        regime = str((market_context or {}).get("market_regime") or (market_context or {}).get("chart_regime") or ctx["macro_bias"])
        for cand in zones[:6]:
            entry = float(cand["mid"])
            if abs(price - entry) > atr * MAIN_ENTRY_MAX_ATR:
                continue
            geo = _geometry(m15, h1, direction, entry, atr)
            if geo is None:
                continue
            tp = _targets(m15, h1, direction, entry, geo[1])
            if tp is None:
                continue
            tp_price, tp_label, rr = tp
            if rr < MIN_RR:
                continue
            htf_overlap = any(z["bot"] <= entry <= z["top"] for z in htf_zones)
            q = _candidate_quality(direction, cand, htf_overlap, sweep, displacement, pullback, location, context, geo, rr, ctx, regime)
            archetype = _archetype(direction, cand.get("kind", "market"), sweep, displacement, pullback, htf_overlap, regime)
            history = _history_context(trade_history, direction, cand.get("kind", "market"), archetype)
            execution_score = q["quality"] * 0.82 + min(rr, 6.0) * 2.0 + q["poi_quality"] * 0.08
            evaluated.append({
                "entry": entry, "entry_label": cand.get("kind", "market"), "sl": geo[0], "risk": geo[1],
                "risk_atr": geo[2], "risk_pct": geo[3], "tp": tp_price, "tp_label": tp_label, "rr": rr,
                "q": q, "archetype": archetype, "history": history, "execution_score": execution_score,
                "htf_overlap": htf_overlap,
            })
        if not evaluated:
            return None
        evaluated.sort(key=lambda x: x["execution_score"], reverse=True)
        best = evaluated[0]
        conf = best["q"]["confidence"]
        band = "ELITE" if conf >= 78 else "STRONG" if conf >= 68 else "VALID" if conf >= 55 else "WEAK"
        decision = "BUY" if direction == "bull" else "SELL"
        evidence = [
            f"HTF alignment: {ctx['htf_alignment']}",
            f"direction quality: {ctx['direction_quality']:.0f}",
            f"POI: {best['entry_label']} ({best['q']['poi_quality']:.0f})",
            f"pullback: {pullback['state']}",
            f"displacement: {displacement['confirmed']}",
            f"market context: {context['score']:.0f}",
        ]
        if sweep["type"] == "sweep": evidence.append("liquidity sweep")
        return {
            "symbol": symbol, "decision": decision, "confidence": conf,
            "direction_confidence": int(round(ctx["direction_quality"])), "setup_quality": int(round(best["q"]["quality"])),
            "confidence_band": band, "confidence_model": CONFIDENCE_MODEL_VERSION, "confidence_is_probability": False,
            "confidence_diagnostics": {
                "components": best["q"]["components"], "contradictions": best["q"]["contradictions"],
                "penalty": best["q"]["penalty"], "completeness": best["q"]["completeness"],
                "archetype": best["archetype"], "history": best["history"],
                "market_context": context, "direction": ctx,
            },
            "market_thesis": {"direction": decision, "archetype": best["archetype"], "market_regime": regime, "evidence": evidence, "contradictions": best["q"]["contradictions"]},
            "entry_location_score": int(round(location["score"])), "entry_location_state": "preferred" if location["score"] >= 70 else "acceptable" if location["score"] >= 45 else "late",
            "entry_range_position": location["range_position"], "entry_zone_low": location["range_low"], "entry_zone_high": location["range_high"],
            "trend_strength": {"score": ctx["direction_quality"], "state": ctx["struct_h1"]}, "pullback_quality": pullback,
            "liquidity_context": {"sweep": sweep}, "poi_quality": best["q"]["poi_quality"], "poi_state": "fresh" if best["q"]["poi_quality"] >= 70 else "usable",
            "market_regime": regime, "entry": round(best["entry"], 10), "price": round(price, 10), "entry_label": best["entry_label"],
            "sl": round(best["sl"], 10), "initial_sl": round(best["sl"], 10), "initial_risk": round(best["risk"], 10),
            "tp": round(best["tp"], 10), "rr": round(best["rr"], 3), "tp_label": best["tp_label"], "atr": round(atr, 10),
            "risk_atr": round(best["risk_atr"], 3), "risk_pct": round(best["risk_pct"], 4), "rsi": round(float(m15["rsi"].iloc[-1]), 2),
            "m15_rsi": float(m15["rsi"].iloc[-1]), "m15_rsi_slope": float(m15["rsi"].iloc[-1] - m15["rsi"].iloc[-2]),
            "m15_relative_volume": float(m15["volume"].iloc[-1] / max(m15["vol_sma"].iloc[-1], 1e-12)),
            "struct_h1": ctx["struct_h1"], "d1_bias": ctx["d1_bias"], "htf_bias": ctx["d1_bias"] if ctx["d1_bias"] != "neutral" else ctx["struct_h1"], "h1_bias": ctx["struct_h1"],
            "choch_m15": {"bullish_choch": False, "bearish_choch": False}, "choch_h1": {"bullish_choch": False, "bearish_choch": False},
            "cisd_m15": {"bullish_cisd": False, "bearish_cisd": False}, "failed_retest": {},
            "entry_confirmation": displacement, "selected_sweep": sweep["type"] == "sweep", "trigger_count": int(displacement["confirmed"]) + int(sweep["type"] == "sweep"),
            "v11_quality": {"trend_strength": {"score": ctx["direction_quality"]}, "pullback_quality": pullback, "poi_quality": best["q"]["poi_quality"]},
            "reasoning_engine": TRAIL_ENGINE_VERSION,
            "tp_sl_reason": f"Entry@{best['entry']:.8g}({best['entry_label']}) | SL@{best['sl']:.8g} | TP@{best['tp']:.8g}({best['tp_label']}) | RR={best['rr']:.2f} | quality={best['q']['quality']:.1f} | conf={conf}%",
        }
    except Exception as exc:
        log.exception("[full_analyze] %s: %s", symbol or "?", exc)
        return None


def _current_price(df, state):
    p = _num(state.get("current_price"), None)
    return p if p is not None else float(df["close"].iloc[-1]) if df is not None and not df.empty else None


def _path_metrics(df, state, direction, entry, risk, price):
    cur = (price - entry) / risk if direction == "bull" else (entry - price) / risk
    smfe = _num(state.get("mfe_r"), 0.0) or 0.0
    smae = _num(state.get("mae_r"), 0.0) or 0.0
    win = df.tail(TRAIL_PEAK_LOOKBACK)
    peak = float(win["high"].max()) if direction == "bull" else float(win["low"].min())
    pmfe = (peak - entry) / risk if direction == "bull" else (entry - peak) / risk
    mfe = max(0.0, smfe, pmfe)
    gb = max(0.0, mfe - cur)
    ratio = gb / max(mfe, 0.25) if mfe > 0 else 0.0
    return {"current_r": round(cur, 4), "mfe_r": round(mfe, 4), "mae_r": round(smae, 4), "giveback_r": round(gb, 4), "giveback_ratio": round(ratio, 4)}


def _reversal_state(df, direction, cur_r, gb_ratio):
    atr = max(float(df["atr"].iloc[-1]), 1e-12)
    a, b = df.iloc[-1], df.iloc[-2]
    body = abs(float(a["close"] - a["open"])) / atr
    counter = (a["close"] < a["open"]) if direction == "bull" else (a["close"] > a["open"])
    counter2 = (b["close"] < b["open"]) if direction == "bull" else (b["close"] > b["open"])
    rv = float(a["volume"] / max(a["vol_sma"], 1e-12))
    if counter and counter2 and body >= TRAIL_REVERSAL_BODY_ATR and gb_ratio >= TRAIL_GIVEBACK_STRONG:
        return "REVERSAL_CONFIRMED", 3, rv
    if counter and body >= TRAIL_COUNTER_BODY_ATR and gb_ratio >= TRAIL_GIVEBACK_WARN:
        return "WEAKENING", 2, rv
    if gb_ratio >= TRAIL_GIVEBACK_STRONG or cur_r < 0.5:
        return "CAUTION", 1, rv
    return "HEALTHY", 0, rv


def _structural_trail(df, direction, price, atr):
    sub = df.tail(STRUCT_TRAIL_LOOKBACK)
    sh, sl = swing_pts(sub, STRUCT_TRAIL_LB)
    if direction == "bull" and sl:
        return float(sub["low"].iloc[sl[-1]]) - atr * TRAIL_STRUCT_BUFFER_ATR
    if direction == "bear" and sh:
        return float(sub["high"].iloc[sh[-1]]) + atr * TRAIL_STRUCT_BUFFER_ATR
    return None


def _retracement_trail(df, direction, entry, risk, price, path):
    atr = max(float(df["atr"].iloc[-1]), 1e-12)
    if path["mfe_r"] < 1.0 or path["giveback_ratio"] < TRAIL_GIVEBACK_WARN:
        return None
    if direction == "bull":
        peak = float(df["high"].tail(TRAIL_PEAK_LOOKBACK).max())
        frac = 0.35 if path["giveback_ratio"] < TRAIL_GIVEBACK_STRONG else 0.25
        return min(peak - (peak - entry) * frac, price - atr * TRAIL_RETRACE_BUFFER_ATR)
    peak = float(df["low"].tail(TRAIL_PEAK_LOOKBACK).min())
    frac = 0.35 if path["giveback_ratio"] < TRAIL_GIVEBACK_STRONG else 0.25
    return max(peak + (entry - peak) * frac, price + atr * TRAIL_RETRACE_BUFFER_ATR)


def manage_position(state, df_m15, df_h1=None, df_d1=None, symbol=None):
    try:
        if df_m15 is None or len(df_m15) < 40:
            return {"action": "PROTECT", "state": "UNKNOWN", "reason": ["insufficient_m15_data"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        m15 = build_df(df_m15, 15)
        if m15 is None or len(m15) < 40:
            return {"action": "PROTECT", "state": "UNKNOWN", "reason": ["insufficient_m15_data"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        sig = state.get("signal") or {}
        direction = "bull" if str(sig.get("decision", "BUY")).upper() == "BUY" else "bear"
        entry = _num(state.get("entry") or sig.get("entry"), None)
        initial_sl = _num(state.get("initial_sl") or sig.get("initial_sl") or sig.get("sl"), None)
        current_sl = _num(state.get("current_sl") or sig.get("sl"), None)
        price = _current_price(m15, state)
        if entry is None or initial_sl is None or current_sl is None or price is None:
            return {"action": "PROTECT", "state": "UNKNOWN", "reason": ["missing_position_geometry"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        risk = abs(entry - initial_sl)
        if risk <= 0:
            return {"action": "PROTECT", "state": "UNKNOWN", "reason": ["invalid_initial_risk"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        if direction == "bull" and price <= current_sl:
            return {"action": "PROTECT", "state": "AT_STOP", "reason": ["market_at_or_below_stop"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        if direction == "bear" and price >= current_sl:
            return {"action": "PROTECT", "state": "AT_STOP", "reason": ["market_at_or_above_stop"], "reasoning_engine": TRAIL_ENGINE_VERSION}

        path = _path_metrics(m15, state, direction, entry, risk, price)
        cur_r, mfe, gb = path["current_r"], path["mfe_r"], path["giveback_ratio"]
        state_name, rev_score, rv = _reversal_state(m15, direction, cur_r, gb)
        reasons = [f"mfe={mfe:.2f}R", f"giveback={gb:.0%}"] if mfe > 0 else []
        if rev_score: reasons.append(f"reversal_signal={rev_score}")
        if rv <= TRAIL_VOLUME_EXHAUSTION: reasons.append("volume_exhaustion")
        elif rv >= TRAIL_VOLUME_COUNTER and rev_score: reasons.append("counter_volume")

        if cur_r < TRAIL_ARM_R and mfe < TRAIL_ARM_R:
            return {"action": "HOLD", "state": "INITIAL" if cur_r < 0.35 else "PROVING", "profit_r": cur_r, "lifecycle": path, "weakness_score": 0, "relative_volume": rv, "reason": ["path_not_mature_for_trailing"], "reasoning_engine": TRAIL_ENGINE_VERSION}

        atr = max(float(m15["atr"].iloc[-1]), 1e-12)
        candidates = []
        s = _structural_trail(m15, direction, price, atr)
        r = _retracement_trail(m15, direction, entry, risk, price, path)
        if s is not None: candidates.append((float(s), "structure"))
        if r is not None: candidates.append((float(r), "retracement"))
        if mfe >= TRAIL_MFE_EXTENDED:
            lock_r = TRAIL_LOCK_STRONG_R if gb >= TRAIL_GIVEBACK_WARN else TRAIL_LOCK_WARN_R
            if gb >= TRAIL_GIVEBACK_CRITICAL and mfe >= TRAIL_MFE_DEEP:
                lock_r = TRAIL_LOCK_CRITICAL_R
            lock = entry + risk * lock_r if direction == "bull" else entry - risk * lock_r
            candidates.append((float(lock), "path_protection"))

        valid = []
        for cand, source in candidates:
            if direction == "bull":
                cand = min(cand, price - atr * TRAIL_MIN_MARKET_GAP_ATR)
                if cand <= current_sl or cand >= price: continue
            else:
                cand = max(cand, price + atr * TRAIL_MIN_MARKET_GAP_ATR)
                if cand >= current_sl or cand <= price: continue
            valid.append((cand, source))
        if not valid:
            return {"action": "PROTECT" if gb >= TRAIL_GIVEBACK_WARN or rev_score >= 2 else "HOLD", "state": state_name, "profit_r": round(cur_r, 3), "lifecycle": path, "weakness_score": rev_score, "relative_volume": rv, "reason": reasons + ["no_safe_trail_candidate"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        best, source = max(valid, key=lambda x: x[0]) if direction == "bull" else min(valid, key=lambda x: x[0])
        improvement = ((best - current_sl) if direction == "bull" else (current_sl - best)) / risk
        count = int(state.get("trail_update_count", 0) or 0)
        if improvement < TRAIL_MIN_UPDATE_R and count >= TRAIL_MAX_CHURN:
            return {"action": "PROTECT", "state": state_name, "profit_r": round(cur_r, 3), "lifecycle": path, "weakness_score": rev_score, "relative_volume": rv, "reason": reasons + ["anti_churn"], "reasoning_engine": TRAIL_ENGINE_VERSION}
        locked_r = ((best - entry) if direction == "bull" else (entry - best)) / risk
        return {
            "action": "TRAIL", "state": state_name, "sl": round(float(best), 10), "profit_r": round(cur_r, 3),
            "locked_r": round(float(locked_r), 3), "trail_source": source, "candidate_type": source,
            "weakness_score": rev_score, "relative_volume": rv, "lifecycle": path,
            "reversal_diagnostics": {"state": state_name, "score": rev_score, "relative_volume": rv},
            "reason": reasons + [f"source={source}"], "reasoning_engine": TRAIL_ENGINE_VERSION,
        }
    except Exception as exc:
        log.exception("[manage_position] %s: %s", symbol or "?", exc)
        return {"action": "PROTECT", "state": "ERROR", "reason": [f"management_exception:{type(exc).__name__}"], "reasoning_engine": TRAIL_ENGINE_VERSION}


def get_best_signal(candidates: list) -> Optional[dict]:
    if not candidates:
        return None
    return max(candidates, key=lambda x: float(x.get("execution_score", x.get("confidence", 0)) or 0))
